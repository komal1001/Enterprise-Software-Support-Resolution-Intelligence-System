"""
End-to-end smoke test against the real LLM provider and the real database.

Run after changing LLM_API_KEY / LLM_BASE_URL, rotating the OpenAI key, or
before a deploy:

    python scripts/smoke_test.py            # all three stages
    python scripts/smoke_test.py --quick    # stage 1 only (no DB, ~5 s)

Stage 1  LLM provider   chat, structured output, streaming, embeddings
Stage 2  Retrieval      hybrid BM25 + pgvector search on the stored documents
Stage 3  Full pipeline  one RAG, one SQL and one Multi-Agent ticket through the graph

Costs a fraction of a cent per run. Uses real credentials from .env.
"""

import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # nodes_cache.json is loaded by relative path
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

import nest_asyncio

nest_asyncio.apply()

failures = 0


def check(name, fn):
    global failures
    start = time.time()
    try:
        detail = fn()
        print(f"  [OK]   {name} ({time.time() - start:.1f}s) {detail}")
    except Exception as e:
        failures += 1
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")


def stage_llm():
    from typing import Literal

    from pydantic import BaseModel

    from src.llm import CompatEmbedding, chat_llm, chat_model_name, embed_model_name

    print(f"\nStage 1 — LLM provider ({os.environ.get('LLM_BASE_URL') or 'api.openai.com'}, "
          f"{chat_model_name()}, {embed_model_name()})")
    if not os.environ.get("LLM_API_KEY"):
        print("  [FAIL] LLM_API_KEY is empty in .env — paste the key and save the file")
        sys.exit(1)

    class Probe(BaseModel):
        category: Literal["billing", "security", "production_incident"]
        confidence: float

    def structured():
        r = chat_llm(temperature=0).with_structured_output(Probe, include_raw=True).invoke(
            "Our whole production API is down for all users!")
        return f"-> {r['parsed']}, tokens={r['raw'].usage_metadata['total_tokens']}"

    def embedding():
        dims = len(CompatEmbedding().get_query_embedding("reset OAuth token"))
        assert dims == 1536, f"got {dims} dims; stored vectors are 1536 — wrong embedding model"
        return f"-> {dims} dims (matches stored vectors)"

    check("chat", lambda: "-> " + chat_llm(temperature=0).invoke("Reply with exactly: pong").content)
    check("structured output", structured)
    check("streaming", lambda: "-> " + "".join(
        c.content for c in chat_llm(temperature=0, streaming=True).stream("Count 1 to 5 separated by spaces")))
    check("embedding", embedding)


def stage_retrieval():
    from src.rag.retrieval import retrieve

    print("\nStage 2 — Retrieval on the database")

    def search():
        chunks = retrieve("How do I rotate an OAuth client secret?", use_cache=False)
        assert chunks, "no chunks returned"
        sources = sorted({c.get("source") for c in chunks})
        return f"-> {len(chunks)} chunks from {sources}"

    check("hybrid search (BM25 + pgvector)", search)


def stage_pipeline():
    from src.graph.graph import build_graph

    print("\nStage 3 — Full agent pipeline")
    graph = build_graph()
    tickets = [
        ("RAG", "How do I configure OAuth 2.0 authentication for the API?"),
        ("SQL", "How many open tickets does customer CUST-0001 have?"),
        ("Multi-Agent", "URGENT: our entire production environment has been down for 2 hours, "
                        "all users affected, we are an enterprise customer CUST-0005"),
    ]
    for expected, text in tickets:
        def run(text=text, expected=expected):
            state = graph.invoke(
                {"ticket_text": text, "conversation_history": [], "classification": None,
                 "rag_result": None, "sql_result": None, "severity_assessment": None,
                 "escalation_package": None, "final_response": None, "reflection_count": 0},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )
            c = state.get("classification") or {}
            sev = state.get("severity_assessment") or {}
            assert state.get("final_response"), "no final response"
            route_note = "" if c.get("routing_path") == expected else f" (expected {expected})"
            team = (state.get("escalation_package") or {}).get("team")
            return (f"-> route={c.get('routing_path')}{route_note}, {c.get('category')}, "
                    f"conf={c.get('confidence')}, severity={sev.get('final_severity') or sev.get('severity')}, "
                    f"escalate={sev.get('escalate')}" + (f", team={team}" if team else "")
                    + f"\n         \"{state['final_response'][:140].replace(chr(10), ' ')}…\"")
        check(f"{expected} ticket", run)


if __name__ == "__main__":
    stage_llm()
    if "--quick" not in sys.argv:
        stage_retrieval()
        stage_pipeline()
    print("\nRESULT:", "all checks passed" if failures == 0 else f"{failures} check(s) failed")
    sys.exit(1 if failures else 0)
