# Deployment Migration — November 2026

**Status:** In progress · **Started:** 2026-11-24 · **Related ADR:** ADR-019 (CLAUDE.md and `diagrams and tracker/Architecture_Decision_Records.docx`)

This records why the live deployment broke, every option considered for fixing it, why each
rejected option was rejected, and exactly what was changed. Free-tier facts below were checked
on 2026-11-24; free tiers change often, so re-check before relying on them.

---

## 1. What broke

The system was deployed as:

| Layer | Platform | Who paid |
|---|---|---|
| Backend (FastAPI) | Railway | Railway free trial credit |
| Frontend (React/Vite) | Vercel Hobby | Free plan |
| Database (PostgreSQL + pgvector) | Neon | Free plan |
| LLM + embeddings | Azure OpenAI (GPT-4o mini, text-embedding-3-small) | NIIT-provided Azure subscription |
| Observability | Langfuse Cloud | Free Hobby plan |

Two things expired at the end of the NIIT course:

1. **Railway trial credit ran out** → backend API down. The Vercel frontend still loads
   (verified: HTTP 200) but every ticket fails because it has no API to call.
2. **The NIIT Azure subscription ended** → the Azure OpenAI resource was deleted.
   Verified: the endpoint hostname in `.env` no longer resolves in DNS (`getaddrinfo failed`),
   while other sites (api.openai.com) are reachable — so it is not a local network problem.

What still works (verified 2026-11-24):

- **Neon database** — connected with the production credentials: 120 customers, 1,000 tickets,
  50 incidents, and the `data_document_chunks` pgvector table are all intact.
- **Vercel frontend** — Hobby is a permanent free plan, not a trial.

So two replacements were needed: **a backend host** and **an LLM provider**.

---

## 2. Decision: backend host → Render (free plan)

### Options considered

| Option | Verdict | Reason |
|---|---|---|
| Railway (top up) | Rejected | Paid only after trial; the goal was a $0 host. |
| **Hugging Face Spaces (Docker)** | **Rejected — not available** | Initially recommended (16 GB RAM, free). On trying to create the Space, Docker and Gradio SDKs were marked **"Paid"**. Hugging Face changed its policy in mid-2026: Docker/Gradio Spaces now require a PRO plan; free accounts can only create Static Spaces, which cannot run a Python backend. |
| Azure Container Apps | Rejected | Has a generous monthly free grant, but requires an Azure subscription. The only one available was NIIT's, which has ended. |
| Google Cloud Run | Rejected (for now) | Free monthly allowance is enough, but requires a credit card / billing account. Kept as the fallback if Render's memory proves too small. |
| Koyeb | Rejected | Requires a card (reported ~$29 hold since Feb 2026); conflicting reports on whether the free instance sleeps. |
| Fly.io | Rejected | No free tier for new accounts. |
| **Render free web service** | **Chosen** | No card required. Runs the same Dockerfile. Auto-deploys from GitHub (like Railway did). |

### Region: Virginia (US East)

The Neon database is in AWS us-east-1, so the Render service is placed in Render's Virginia
region. Each ticket makes several database round-trips (SQL agent, pgvector search, escalation
incident lookup), so server↔database distance dominates; the user (in Europe) only crosses the
Atlantic once per request (~0.1 s). The OpenAI API is also US-hosted. Moving Neon to an EU region
was not worth a database migration for a demo.

### Known trade-offs of Render free

- **512 MB RAM.** Measured locally: the backend uses roughly **200–400 MB** after start-up
  (app imported, LangGraph built, BM25 corpus loaded). It should fit, with limited headroom.
  If it runs out of memory under real tickets, move to Cloud Run (needs a card).
- **Sleeps after 15 minutes idle; ~1 minute to wake.** The first visitor after a quiet period
  waits. Mitigations (planned): a free uptime pinger hitting `/health`, and a
  "waking up the server…" message in the frontend.
- **In-memory state is lost on restart/sleep:** chat history (`MemorySaver`), semantic cache,
  cached Langfuse prompts. Nothing in Neon is affected. (PostgresSaver is already on the
  post-deploy TODO list.)

---

## 3. Decision: LLM provider → OpenAI API (prepaid $5)

### Key constraint

The document embeddings already stored in Neon were produced by **text-embedding-3-small
(1536 dimensions)**. A different embedding model produces incompatible vectors, which would make
similarity search meaningless unless every chunk is re-embedded. Keeping the same models means
**no re-ingest**.

### Options considered

| Option | Verdict | Reason |
|---|---|---|
| Azure OpenAI (own subscription) | Rejected | No personal Azure subscription; NIIT's has ended. |
| **GitHub Models (free)** | **Rejected — service retired** | Initially recommended (free, no card, same models). A token was created and tested: every endpoint at `models.github.ai` — including the public model catalog — returns a bare `200 text/plain "OK"` instead of a model response. Confirmed: **GitHub retired GitHub Models on 2026-07-30** (closed to new customers 2026-06-16). |
| Google Gemini API (free tier) | Rejected | Free and no card, and has an OpenAI-compatible endpoint. But: (1) different embedding model → all 155 chunks would have to be re-embedded; (2) free-tier daily limits are unclear and some sources report very low caps; (3) agent prompts and confidence thresholds were tuned on GPT-4o mini, so classification quality (SLO #10/#11) could shift. Too much risk for a working demo. |
| **OpenAI API (prepaid credit)** | **Chosen** | Same models as before → stored embeddings stay valid, no re-ingest, no prompt re-tuning. API reachability verified. Prepaid billing: $5 minimum top-up, no subscription, stops at $0 (auto-recharge kept OFF). Estimated ~$0.003–0.005 per ticket at GPT-4o mini prices → roughly 1,000+ demo tickets, and enough to run the full 51-case golden-set eval. |

Cost notes: billed in USD from Europe, so VAT at the local rate (~19–25%) is added on top of the
$5 credit (≈ $6 charged, full $5 usable), plus any bank currency-conversion fee. OpenAI prepaid credit expires (believed 1 year after purchase — check the billing
page). ChatGPT Plus is a separate product and does not include API access.

---

## 4. How — code changes

### 4.1 Provider-agnostic LLM client (`src/llm.py`, new)

Previously every agent constructed `AzureChatOpenAI(...)` with four Azure-specific environment
variables, hard-wiring the code to Azure. All clients are now built in one module from:

```
LLM_BASE_URL     default https://api.openai.com/v1
LLM_API_KEY
LLM_CHAT_MODEL   optional, default gpt-4o-mini
LLM_EMBED_MODEL  optional, default text-embedding-3-small (must match stored vectors)
```

Switching to any OpenAI-compatible provider later is an environment change, not a code change.

It exposes three builders:

- `chat_llm(**kwargs)` → LangChain `ChatOpenAI`. `with_structured_output` keeps its default
  `json_schema` method — the same default `AzureChatOpenAI` used, so agent behaviour (ADR-011)
  is unchanged. `prompt_cache_key` (ADR-016) is only sent when the provider is api.openai.com,
  because other providers may reject unknown request fields.
- `CompatEmbedding` → LlamaIndex embedding that calls the `openai` client directly. Written
  because LlamaIndex's own `OpenAIEmbedding` rejects model names outside a hard-coded list,
  which would block non-OpenAI providers.
- `langchain_embeddings()` → LangChain `OpenAIEmbeddings` for the RAGAS evaluator.

### 4.2 Files switched off Azure

| File | Change |
|---|---|
| `src/agents/agent1_classify.py` | `AzureChatOpenAI(...)` → `chat_llm(temperature=0)` |
| `src/agents/agent2_rag.py` | same |
| `src/agents/agent4_severity.py` | same |
| `src/agents/agent5_escalation.py` | same |
| `src/agents/response_synthesizer.py` | same (structured + streaming clients) |
| `src/rag/sql.py` | same |
| `src/rag/retrieval.py` | `AzureOpenAIEmbedding` → `CompatEmbedding`; `Settings.llm = None` (LlamaIndex never calls an LLM here since `num_queries=1`; `None` avoids it silently creating a default OpenAI client) |
| `src/evaluation/ragas_eval.py` | uses `chat_llm` + `langchain_embeddings` |
| `eval/eval.py` | LLM judge uses `chat_llm` |
| `tests/conftest.py` | fake `LLM_*` env vars instead of `AZURE_*` |
| `requirements.txt` | removed `llama-index-embeddings-azure-openai`, `llama-index-llms-azure-openai` (unused; smaller image) |
| `.env.example`, `README.md`, `CLAUDE.md` | updated; ADR-019 added |

### 4.3 Deployment files

- **`Dockerfile`** (new) — Python 3.12 slim, installs `requirements.txt`, copies only `src/`,
  `Documents/` (PDFs served at `/docs`) and `nodes_cache.json` (BM25 corpus). Runs as a
  non-root user; uvicorn listens on `$PORT` (Render injects it; 8000 fallback locally).
- **`.dockerignore`** (new) — keeps `.env`, frontend, eval, tests, database scripts out of the image.
- **`render.yaml`** (new) — Render Blueprint: free plan, Docker runtime, `/health` health check,
  auto-deploy on push. Secrets are `sync: false`, so Render asks for them once in the dashboard.
- `railway.toml` — left in place for now; obsolete.
- Hugging Face deploy script — written, then deleted once Docker Spaces turned out to be paid.

### 4.4 Verification so far

- No `AZURE_*` references remain in `src/`, `eval/`, `tests/`.
- Unit tests: **236 passed, 18 failed — identical to before the change** (compared by running
  the suite on the original code). The 18 failures and two uncollectable test files
  (`test_semantic_cache.py`, `test_sql_safety.py` still import the old `src.retrieval` path) are
  pre-existing and unrelated.
- Neon data verified intact (see §1).
- Memory footprint measured (see §2).
- **Not yet verified:** Docker build, Render deploy.

### 4.5 Live verification against OpenAI (2026-11-24, run locally)

| Check | Result |
|---|---|
| Chat completion | OK |
| Structured output (`with_structured_output`, include_raw) | OK — parsed Pydantic object + token usage returned (needed for Langfuse cost, SLO #14) |
| Streaming | OK |
| Embeddings | OK — 1536 dimensions, matches stored vectors |
| Hybrid retrieval on Neon (BM25 + pgvector, RRF) | 5 chunks, top results from API_Integration_Authentication_Guide.pdf for an OAuth query — stored embeddings remain valid, no re-ingest needed |
| RAG ticket ("configure OAuth 2.0") | route=RAG, usage_configuration, conf 0.95, severity Low, cited answer |
| SQL ticket ("open tickets for CUST-0001") | route=SQL, SQL agent queried DB, answer from account data |
| Multi-Agent ticket ("production down 2h, enterprise") | route=Multi-Agent, production_incident, conf 0.98, Critical, escalated to SRE Team (P1) with ITIL + SLA sources |

Open item: these tickets took 12–15 s each end-to-end, above SLO #5/#6 (5 s / 10 s P95). This
run included cold-start costs (Neon wake, first Langfuse prompt fetches, retriever build);
warm-path latency still needs measuring on the deployed service.

---

## 4.6 OpenAI API key rotation

The key was created with a **30-day expiry** (deliberately — a non-expiring key with a payment
method attached was judged too risky). Created 2026-11-24 → **expires about 2026-12-24**.
Calendar reminder: renew on 2026-12-20.

When it expires, tickets fail with an authentication error; no data is lost and nothing is charged.

Renewal steps:
1. platform.openai.com → API keys → Create new secret key (Default project, 30-day expiry).
2. Update `LLM_API_KEY` in local `.env`.
3. Render → service → Environment → `LLM_API_KEY` → Save (Render redeploys automatically).
4. Delete the old key on platform.openai.com.

---

## 5. Remaining steps

| # | Step | Owner | Status |
|---|---|---|---|
| 1 | Create OpenAI account, add $5 credit (auto-recharge OFF, monthly limit), create API key, put it in `.env` as `LLM_API_KEY` | Komal | Done 2026-11-24 |
| 2 | Live test: chat, structured output, streaming, embeddings (1536 dims), real pgvector search on Neon, full tickets | Claude | Done 2026-11-24 — see §4.5 |
| 3 | Commit and push to GitHub | Claude (on approval) | Done 2026-11-24 (commit 17c1df3) |
| 4 | Render → New → Blueprint → select repo, fill secrets (incl. `AUTH0_DOMAIN`, `AUTH0_API_AUDIENCE` from Auth0/Railway) | Komal | Done 2026-11-24 — live at https://support-resolution-api.onrender.com; verified `/health` 200, PDFs served, CORS allows the Vercel origin, unauthenticated `/ticket/stream` → 401 (RBAC enforced) |
| 5 | Vercel: set `VITE_API_URL` to the Render URL, redeploy (Vite bakes it in at build time) | Komal | Pending |
| 6 | Delete the unused GitHub fine-grained token | Komal | Pending |
| 7 | Optional: `/health` uptime pinger, frontend "waking up" message, delete `railway.toml` | Claude | Pending |

---

## 6. Lessons

- Two of the first recommendations (Hugging Face Docker Spaces, GitHub Models) were based on
  outdated knowledge of free tiers and failed when tried. Free-tier availability should be
  checked against current sources **before** recommending, and each dependency's live status
  tested before building on it.
- Hard-coding a vendor's client class into every agent made the provider switch touch eight
  files. Centralising client construction (`src/llm.py`) makes the next switch a config change.
- Keeping the embedding model fixed was the deciding constraint: it is what made OpenAI the
  lowest-risk choice despite not being free.

---

## Sources (checked 2026-11-24)

- Hugging Face Spaces overview — Docker/Gradio require a paid plan: https://github.com/huggingface/hub-docs/blob/main/docs/hub/spaces-overview.md
- GitHub Models retirement: https://www.cloudkaramchari.com/blog/github-models-retirement-migration-guide/ , https://github.com/qte77/gha-ai-changelog/issues/45
- Free hosting limits 2026 (Render, Koyeb, Cloud Run, Fly.io): https://dev.to/octave_nkurunziza_afb0512/free-tier-hosting-for-side-projects-in-2026-real-limits-6ab , https://snapdeploy.dev/state-of-free-hosting
- Render FastAPI hosting: https://render.com/hosting/fastapi
- Azure Container Apps billing: https://learn.microsoft.com/en-us/azure/container-apps/billing
- Gemini API rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
