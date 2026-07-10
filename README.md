# Enterprise Software Support & Resolution Intelligence System

**Production-grade autonomous agentic AI system for enterprise SaaS support triage.**

Live demo → [enterprise-software-support-resolut.vercel.app](https://enterprise-software-support-resolut.vercel.app)

---

## What It Does

Enterprise support teams receive thousands of tickets monthly — usage questions, API failures, production outages, billing disputes. Manual triage is slow, inconsistent, and expensive.

This system replaces manual triage with a 5-agent AI pipeline that:

- **Classifies** the ticket into one of 7 categories with confidence scoring
- **Routes** to the right retrieval pipeline (RAG / SQL / Hybrid / Multi-Agent)
- **Retrieves** relevant documentation chunks and customer account data
- **Re-evaluates severity** using all retrieved evidence (not just ticket text)
- **Escalates** critical and low-confidence cases to a human team with full context
- **Streams** a cited response in real time via Server-Sent Events

---

## Architecture

```
User Ticket
    │
    ▼
Agent 1 — Classifier
(category · severity · routing_path · confidence)
    │
    ├── RAG ──────────────────────────────────► Agent 2 (RAG)
    │                                               │
    ├── SQL ──────────────► Agent 3 (SQL)           │
    │                           │                   │
    ├── Hybrid ──► Agent 3 ──► Agent 2              │
    │                                               │
    └── Multi-Agent ──► Agent 3 ──► Agent 2 ──► Agent 5 (Escalation)
                                        │
                                    Agent 4 (Severity Re-assessment) ← runs on EVERY ticket
                                        │
                                    Response Synthesizer → SSE stream → React UI
```

**4 Routing Paths:**

| Path | Trigger | Agents |
|---|---|---|
| RAG | Documentation / how-to queries | 1 → 2 → 4 → Synth |
| SQL | Account / billing / ticket lookups | 1 → 3 → 4 → Synth |
| Hybrid | Needs both docs + account context | 1 → 3 → 2 → 4 → Synth |
| Multi-Agent | High/Critical severity | 1 → 3 → 2 → 4 → 5 → Synth |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph with conditional routing) |
| LLM | Azure OpenAI GPT-4o mini |
| Embeddings | Azure OpenAI text-embedding-3-small |
| Vector DB | PostgreSQL + pgvector |
| Structured data | PostgreSQL (parameterized queries + sqlglot validation) |
| Document parsing | LlamaParse (multimodal — tables, images, headers) |
| API | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (SSE) via sse-starlette |
| Frontend | React + Vite |
| Auth | Auth0 JWT + RBAC (l1-agent / manager / admin) |
| Observability | Langfuse (traces, spans, costs, RAGAS scores) |
| Evaluation | RAGAS (faithfulness, answer relevance, context precision) |
| Rate limiting | slowapi (10 req/min per IP) |
| Deployment | Railway (backend) + Vercel (frontend) + Neon (PostgreSQL) |

---

## The 5 Agents

### Agent 1 — Classifier
Classifies every ticket into one of 7 categories using `with_structured_output` (OpenAI function calling — schema enforced at API level, no prompt parsing). Returns category, severity, routing path, confidence score, and reasoning. Falls back to `ambiguous + Multi-Agent` on LLM failure so tickets always reach a human.

**7 categories:** `usage_configuration` · `integration_api` · `performance_latency` · `production_incident` · `billing` · `security` · `ambiguous`

### Agent 2 — RAG (Knowledge Retrieval)
Hybrid retrieval: pgvector semantic search + BM25 keyword search, fused via Reciprocal Rank Fusion (RRF). Returns top 5 chunks with source attribution. Semantic cache (cosine ≥ 0.92) returns cached chunks instantly for repeated queries.

**RAG corpus:** 7 product PDFs → LlamaParse → MarkdownNodeParser → SentenceSplitter → 155 nodes in pgvector

### Agent 3 — SQL (Account Data)
Natural language → SQL via GPT-4o mini + sqlglot AST validation. Only SELECT statements allowed. Allowlisted tables only. MAX_ROWS=100. 5-second query timeout. ThreadedConnectionPool (1–5 connections) with auto-reconnect ping for Neon cold starts.

### Agent 4 — Severity Re-assessment *(runs on EVERY ticket)*
Two-phase safety net:
1. **Hard rules** (deterministic) — ambiguous category, confidence below per-category threshold, Critical severity, High + production/security → always escalate
2. **LLM re-evaluation** — re-reads ticket + RAG chunks + SQL rows to catch hidden signals Agent 1 missed

**Confidence thresholds:** `production_incident` 0.85 · `security` 0.85 · `integration_api` 0.70 · `performance_latency` 0.70 · `usage_configuration` 0.65 · `billing` 0.65 · `ambiguous` 0.00

### Agent 5 — Escalation Manager
Assembles a full context package: ticket history, account data, incident logs (last 5), documentation context, agent reasoning trace. Routes to correct human team (L2/L3). Generates escalation reference number and Jira ticket ID.

---

## Key Features

**Guardrails (Input + Output)**
- Input: blocks prompt injection, PII in ticket text, policy violations before any LLM call
- Output: redacts PII (email, phone, SSN, credit card) from AI responses

**Multi-turn conversation**
- MemorySaver persists context across turns within a session
- Ambiguous first messages trigger clarification prompt instead of pipeline
- Very short messages (< 5 words) skip the full pipeline — instant clarification

**Real-time SSE streaming**
- `status` events after each agent (live progress indicators)
- `token` events word-by-word during synthesis
- `done` event with full response, classification, severity, sources, latency
- `scores` event with RAGAS metrics after response is shown (zero UX latency impact)

**RBAC**
- `l1-agent` — submit tickets
- `manager` — view escalations, ingest documents
- `admin` — full access

**Online document ingestion**
- Upload PDF or Word (.docx/.doc) via Knowledge Base tab
- LlamaParse → chunk → embed → pgvector (persistent) + in-memory BM25 (session)

**Observability**
- Every ticket creates a Langfuse trace with child spans per agent
- Token counts, cost, latency, RAGAS scores attached to each trace
- Azure OpenAI prefix caching for system prompts > 1024 tokens

---

## 14 Service Level Objectives (SLOs)

| # | Category | SLO | Target |
|---|---|---|---|
| 1 | Quality | Task Success Rate | ≥ 90% |
| 2 | Quality | Answer Faithfulness | ≥ 95% |
| 3 | Quality | SQL Correctness | ≥ 95% |
| 4 | Quality | Answer Relevance | ≥ 0.85 |
| 5 | Speed | P95 Latency — Standard | ≤ 5s |
| 6 | Speed | P95 Latency — Multi-Agent | ≤ 10s |
| 7 | Retrieval | Recall@5 | ≥ 90% |
| 8 | Retrieval | Source Attribution Rate | 100% |
| 9 | Retrieval | Context Precision | ≥ 0.80 |
| 10 | Safety | Critical Misclassification Rate | < 3% |
| 11 | Safety | Escalation Recall | 100% |
| 12 | Safety | Unauthorized Data Access | 0 violations |
| 13 | Safety | Guardrail Effectiveness | 100% |
| 14 | Cost | Cost Per Ticket | ≤ $0.05 avg · $0.15 cap |

---

## Project Structure

```
src/
  agents/
    agent1_classify.py       — classification + routing decision
    agent2_rag.py            — RAG retrieval + semantic cache
    agent3_sql.py            — NL→SQL with sqlglot validation
    agent4_severity.py       — severity re-assessment + hard escalation rules
    agent5_escalation.py     — escalation package + full context transfer
    response_synthesizer.py  — streaming response synthesis
  graph/
    state.py                 — TicketState TypedDict
    graph.py                 — LangGraph wiring, 4 routing paths
  api/
    main.py                  — FastAPI app, CORS, OTel, startup warmup
    schemas.py               — Pydantic request/response models
    auth.py                  — Auth0 JWT + RBAC
    limiter.py               — slowapi rate limiter
    routers/
      health.py              — GET /health
      tickets.py             — POST /ticket, POST /ticket/stream (SSE)
      admin.py               — GET /escalations, POST /admin/ingest
  rag/
    ingest.py                — LlamaParse → chunk → embed → pgvector
    retrieval.py             — hybrid BM25 + pgvector + RRF + semantic cache
    semantic_cache.py        — in-memory LRU semantic cache (cosine ≥ 0.92)
    sql.py                   — NL→SQL agent with connection pool
  guardrails/
    guardrails.py            — input guardrails (injection, PII, policy)
    output_guardrails.py     — output PII redaction
  observability/
    langfuse_client.py       — traces, spans, cost calc, prompt cache

frontend-react/              — React + Vite chat UI
  src/components/
    ChatWindow.jsx           — SSE streaming, multi-turn chat
    AnalysisPanel.jsx        — agent pipeline visualization
    EscalationsTab.jsx       — manager escalation dashboard
    IngestTab.jsx            — PDF/Word document upload

eval/
  golden_set.json            — 51 test cases (TC-01 to TC-51)
  eval.py                    — classification + LLM judge + RAGAS + guardrail eval

database/
  schema.sql                 — customers, support_tickets, incident_logs, knowledge_article_usage
  generate_data.py           — 120 customers, 1000 tickets, 50 incidents
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Liveness probe |
| GET | `/escalations` | manager / admin | All escalated tickets |
| POST | `/ticket` | all roles | Synchronous ticket submission |
| POST | `/ticket/stream` | all roles | SSE streaming (used by UI) |
| POST | `/admin/ingest` | manager / admin | Upload PDF/Word to knowledge base |

---

## Running Locally

**Prerequisites:** Python 3.12, Node 18+, PostgreSQL 16 with pgvector

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Fill in: Azure OpenAI, PostgreSQL, Langfuse, LlamaCloud, Auth0

# 3. Set up database
psql -U postgres -c "CREATE DATABASE support_intelligence;"
psql -U postgres -d support_intelligence -f database/schema.sql
python database/generate_data.py

# 4. Ingest documents
python -m src.rag.ingest

# 5. Start backend
uvicorn src.api.main:app --reload --port 8000

# 6. Start frontend
cd frontend-react && npm install && npm run dev
```

---

## Deployment

| Service | Platform |
|---|---|
| Backend API | Railway (auto-deploy from GitHub) |
| Frontend | Vercel (auto-deploy from GitHub) |
| PostgreSQL + pgvector | Neon (serverless, never pauses) |

---

## Evaluation

```bash
# Run full golden set evaluation (51 test cases)
python -m eval.eval --pipeline

# Includes: classification accuracy, LLM judge, RAGAS, guardrail effectiveness
```

---

## Key Architecture Decisions

- **LangGraph over CrewAI** — deterministic routing guarantees 100% escalation recall (SLO #11). CrewAI's autonomous agents cannot make this guarantee.
- **LlamaParse over pypdf** — multimodal parsing preserves tables as markdown; pypdf garbles table structure making it useless for RAG.
- **pgvector over Pinecone** — single PostgreSQL database for both structured and vector data; simpler ops.
- **Agent 4 on every ticket** — even RAG-only tickets can hide critical signals (frustrated customer, SLA breach embedded in casual language).
- **Neon over Supabase** — Supabase free tier pauses after 7 days (20-30s cold start); Neon never pauses projects (~1-2s compute wake).

---

## Author

**Komal Bansal** — NIIT Agentic AI Programme  
[LinkedIn](https://linkedin.com/in/komalbansal) · [GitHub](https://github.com/komal1001)
