import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langfuse import observe
from pydantic import BaseModel

from src.graph.state import TicketState
from src.rag.retrieval import retrieve
from src.observability.langfuse_client import get_langfuse, get_cached_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Output schema — LLM returns a single refined retrieval query
# ---------------------------------------------------------------------------

class RefinedQuery(BaseModel):
    query: str   # keyword-dense, technically precise retrieval query


# ---------------------------------------------------------------------------
# LLM singleton — query rewriting (light task, same model as other agents)
# ---------------------------------------------------------------------------

_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
).with_structured_output(RefinedQuery, include_raw=True)


# ---------------------------------------------------------------------------
# SQL-informed context — extract customer context from SQL rows to enrich query
#
# For Hybrid/Multi-Agent tickets, Agent 3 runs before Agent 2.
# SQL rows may contain subscription tier, SLA level, region, incident root cause —
# concrete facts that narrow the documentation search to this customer's environment.
# ---------------------------------------------------------------------------

def _build_sql_context(rows: list[dict]) -> str:
    if not rows:
        return ""

    context_parts = []
    first_row = rows[0]

    for field in ("subscription_tier", "sla_level", "region"):
        if first_row.get(field):
            context_parts.append(str(first_row[field]))

    if first_row.get("root_cause"):
        context_parts.append(str(first_row["root_cause"])[:100])

    if first_row.get("severity_level"):
        context_parts.append(str(first_row["severity_level"]))
    elif first_row.get("severity"):
        context_parts.append(str(first_row["severity"]))

    return " ".join(filter(None, context_parts))


# ---------------------------------------------------------------------------
# Query rewriting — fallback to raw concatenation if LLM call fails
# ---------------------------------------------------------------------------

def _refine_query(state: TicketState, sql_context: str) -> tuple[str, int, int]:
    """
    Call agent2-rag-query Langfuse prompt to produce a keyword-dense retrieval
    query. Falls back to raw ticket_text + reasoning + sql_context if it fails.
    Returns (query, input_tokens, output_tokens).
    """
    base_query = (
        f"{state['ticket_text']} "
        f"{state['classification']['reasoning']} "
        f"{sql_context}"
    ).strip()

    try:
        prompt_obj = get_cached_prompt("agent2-rag-query")
        prompt = prompt_obj.compile(
            ticket_text=state["ticket_text"],
            category=state["classification"].get("category", ""),
            severity=state["classification"].get("severity", ""),
            reasoning=state["classification"].get("reasoning", ""),
            sql_context=sql_context or "N/A",
        )
        raw = _llm.invoke(prompt)

        if raw.get("parsing_error") or raw["parsed"] is None:
            return base_query, 0, 0

        usage = raw["raw"].usage_metadata or {}
        return (
            raw["parsed"].query,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
    except Exception:
        return base_query, 0, 0


# ---------------------------------------------------------------------------
# Agent 2 — LangGraph node
# ---------------------------------------------------------------------------

@observe(name="agent2-rag", as_type="retriever", capture_input=False, capture_output=False)
def agent2_rag(state: TicketState) -> TicketState:
    lf = get_langfuse()

    sql_rows = (state.get("sql_result") or {}).get("rows", [])
    sql_context = _build_sql_context(sql_rows) if sql_rows else ""

    # LLM query rewriting — produces a keyword-dense, technically precise query
    # instead of raw concatenation. Falls back gracefully on failure.
    refined_query, in_tok, out_tok = _refine_query(state, sql_context)

    # Semantic cache is safe only on RAG-only route — docs don't change per customer.
    # Hybrid/Multi-Agent queries are enriched with customer-specific SQL context,
    # so two "similar" queries can need completely different documentation chunks.
    routing_path = (state.get("classification") or {}).get("routing_path", "")
    use_cache    = (routing_path == "RAG")

    lf.update_current_span(input={"query": refined_query})

    chunks = retrieve(refined_query, use_cache=use_cache)

    from src.rag.semantic_cache import cache_stats
    stats = cache_stats()

    state["rag_result"] = {"query": refined_query, "chunks": chunks}

    lf.update_current_span(
        output={"chunks_retrieved": len(chunks)},
        metadata={
            "sources":        [c.get("source") for c in chunks],
            "top_score":      chunks[0].get("score") if chunks else None,
            "cache_hit":      use_cache and stats["hits"] > 0,
            "cache_hit_rate": stats["hit_rate"],
            "query_rewrite_tokens": in_tok + out_tok,
        },
    )

    return state
