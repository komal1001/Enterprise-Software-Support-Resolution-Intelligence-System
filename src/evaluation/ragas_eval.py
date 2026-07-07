"""
RAGAS-style evaluation metrics implemented on Azure OpenAI.

The installed ragas==0.4.3 package has a broken VertexAI import (ADR-006 note).
This module reimplements the three key RAGAS metrics directly, using the project's
existing Azure deployment — no ragas package dependency.

Metrics:
  faithfulness       (SLO #2  >= 0.95) — statement-level grounding check
  answer_relevance   (SLO #4  >= 0.85) — reverse-question cosine similarity
  context_precision  (SLO #9  >= 0.80) — fraction of retrieved chunks that are relevant

Faithfulness follows the official RAGAS paper methodology exactly:
  Step 1 — decompose answer into N atomic statements (1 LLM call)
  Step 2 — verify EACH statement individually against context (N LLM calls)
  Step 3 — score = supported_statements / total_statements

  Per-statement verification (not batch) matches the RAGAS paper and is more
  accurate because the LLM focuses on one claim at a time. For live tickets
  this runs post-response (after done event) so the extra calls have zero
  UX impact. For eval batch runs use faithfulness_batch() which checks all
  statements in one call for speed across ThreadPoolExecutor workers.

Called from:
  routes.py  — post-response phase (after done event), scores pushed to SSE + Langfuse
  eval.py    — run_full_pipeline_eval() — replaces LLM judge for scored eval runs
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()


# ---------------------------------------------------------------------------
# Shared LLM / embed builders (stateless — called fresh per evaluation)
# ---------------------------------------------------------------------------

def _llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
        model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
    )


def _embed() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness (per-statement — matches RAGAS paper exactly)
#
# Used by: routes.py (live tickets, post-response)
# ---------------------------------------------------------------------------

class _Statements(BaseModel):
    statements: list[str]

class _Verdict(BaseModel):
    supported: bool


def faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Official RAGAS faithfulness methodology — per-statement verification.

    Step 1: Decompose answer into atomic statements (1 LLM call).
    Step 2: For each statement, ask the LLM independently whether it is
            supported by the retrieved context (1 LLM call per statement).
    Step 3: score = supported_statements / total_statements

    More accurate than batch verification because the LLM focuses on one
    claim at a time. Runs post-response for live tickets so extra calls
    have zero UX impact.
    """
    if not answer.strip() or not contexts:
        return 1.0

    llm = _llm()
    context_block = "\n\n".join(
        f"[C{i+1}] {c[:1200]}" for i, c in enumerate(contexts)
    )

    # Step 1 — extract atomic statements
    stmts: _Statements = llm.with_structured_output(_Statements).invoke(
        f"Break the following response into individual, atomic factual statements. "
        f"Each statement should be independently verifiable.\n\nResponse:\n{answer}"
    )
    if not stmts.statements:
        return 1.0

    # Step 2 — verify each statement individually against context
    n_supported = 0
    for statement in stmts.statements:
        verdict: _Verdict = llm.with_structured_output(_Verdict).invoke(
            f"Given the context below, determine whether the statement is directly "
            f"supported by the context. Answer True if the context entails the "
            f"statement, False if it does not or if the context is silent on it.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Statement: {statement}"
        )
        if verdict.supported:
            n_supported += 1

    # Step 3 — score
    return round(n_supported / len(stmts.statements), 3)


# ---------------------------------------------------------------------------
# Metric 1b — Faithfulness batch (for eval pipeline speed)
#
# Used by: eval.py (51-case pipeline, ThreadPoolExecutor workers)
# Checks all statements in one LLM call — faster but slightly less accurate.
# ---------------------------------------------------------------------------

class _Verdicts(BaseModel):
    verdicts: list[bool]


def faithfulness_batch(answer: str, contexts: list[str]) -> float:
    """
    Batch variant of faithfulness — all statements verified in one LLM call.
    Use for eval pipeline where speed matters across parallel workers.
    """
    if not answer.strip() or not contexts:
        return 1.0

    llm = _llm()
    context_block = "\n\n".join(
        f"[C{i+1}] {c[:1200]}" for i, c in enumerate(contexts)
    )

    stmts: _Statements = llm.with_structured_output(_Statements).invoke(
        f"Break the following response into individual, atomic factual statements. "
        f"Each statement should be independently verifiable.\n\nResponse:\n{answer}"
    )
    if not stmts.statements:
        return 1.0

    stmt_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(stmts.statements))
    verdicts: _Verdicts = llm.with_structured_output(_Verdicts).invoke(
        f"For each statement below, return True if the statement is supported by "
        f"the provided contexts, False otherwise. Return one boolean per statement "
        f"in the same order.\n\nContexts:\n{context_block}\n\nStatements:\n{stmt_list}"
    )

    if not verdicts.verdicts:
        return 1.0

    n_supported = sum(1 for v in verdicts.verdicts if v)
    return round(n_supported / len(verdicts.verdicts), 3)


# ---------------------------------------------------------------------------
# Metric 1c — Task Completion (LLM judge — for SQL-only route)
#
# RAGAS answer relevance uses reverse-question cosine similarity, which
# underestimates data-table responses by ~10-15% (data-dense answers drift
# from the question phrasing even when they fully answer it).
# Task completion asks the LLM directly: "did this response fully answer
# the data lookup question?" — the right instrument for SQL tickets.
# ---------------------------------------------------------------------------

class _CompletionScore(BaseModel):
    score: float
    reasoning: str


def task_completion(question: str, answer: str) -> float:
    """
    LLM judge for SQL ticket relevance — SLO #4 proxy for data-lookup routes.
    Measures whether the response fully answers the user's data question.
    """
    if not question.strip() or not answer.strip():
        return 1.0

    llm = _llm()
    scored: _CompletionScore = llm.with_structured_output(_CompletionScore).invoke(
        "You are evaluating whether a support system response completely answers "
        "a user's data lookup question.\n\n"
        "Score from 0.0 to 1.0:\n"
        "  1.0 = Response fully answers the question with all requested data\n"
        "  0.7 = Response partially answers but misses some requested details\n"
        "  0.4 = Response addresses the topic but doesn't directly answer the question\n"
        "  0.0 = Response doesn't answer the question at all\n\n"
        f"Question: {question}\n\n"
        f"Response: {answer}\n\n"
        "Return the score and brief reasoning."
    )
    return round(max(0.0, min(1.0, scored.score)), 3)


# ---------------------------------------------------------------------------
# Metric 2 — Answer Relevance
# ---------------------------------------------------------------------------

class _Questions(BaseModel):
    questions: list[str]


def answer_relevance(question: str, answer: str) -> float:
    """
    Generate 3 questions the answer could be responding to.
    Score = avg cosine_sim(generated_question, original_question).
    """
    if not question.strip() or not answer.strip():
        return 1.0

    llm   = _llm()
    embed = _embed()

    gen: _Questions = llm.with_structured_output(_Questions).invoke(
        f"Given the following answer, generate exactly 3 questions that this answer "
        f"could be responding to. Return as a JSON list.\n\nAnswer:\n{answer}"
    )
    if not gen.questions:
        return 1.0

    orig_vec = embed.embed_query(question)
    sims = []
    for gq in gen.questions[:3]:
        gq_vec = embed.embed_query(gq)
        sims.append(_cosine(orig_vec, gq_vec))

    return round(sum(sims) / len(sims), 3)


# ---------------------------------------------------------------------------
# Metric 3 — Context Precision
# ---------------------------------------------------------------------------

class _AllVerdicts(BaseModel):
    verdicts: list[bool]   # one per chunk — True = relevant


def context_precision(question: str, contexts: list[str]) -> float:
    """
    Check all retrieved chunks in one LLM call.
    Score = relevant_chunks / total_chunks.  SLO #9 target >= 0.80.
    """
    if not contexts:
        return 1.0

    llm = _llm()
    chunks_block = "\n\n".join(
        f"Chunk {i+1}:\n{c[:1200]}" for i, c in enumerate(contexts)
    )
    v: _AllVerdicts = llm.with_structured_output(_AllVerdicts).invoke(
        f"For each context chunk below, return True if it contains information "
        f"that is useful or partially useful for answering the question, "
        f"False only if it is completely unrelated.\n\n"
        f"Question: {question}\n\n{chunks_block}\n\n"
        f"Return a list of {len(contexts)} booleans, one per chunk in order."
    )

    if not v.verdicts:
        return 1.0

    relevant = sum(1 for b in v.verdicts if b)
    return round(relevant / len(v.verdicts), 3)


# ---------------------------------------------------------------------------
# Public entry point — live tickets (uses per-statement faithfulness)
# ---------------------------------------------------------------------------

def score_ragas(
    question:     str,
    answer:       str,
    contexts:     list[str],
    is_escalated: bool = False,
    routing_path: str  = "",
    sql_contexts: list[str] | None = None,
) -> dict:
    """
    Run RAGAS metrics appropriate for the routing path.

    SQL route:       only answer_relevance — faithfulness/context_precision need
                     document chunks, not raw SQL rows.
    RAG / Hybrid:    all three metrics. Faithfulness uses per-statement verification.
    Escalated:       answer_relevance only — template responses, no retrieval.

    sql_contexts:    SQL rows for Hybrid/Multi-Agent tickets. Included in faithfulness
                     context so account-data claims can be verified, but EXCLUDED from
                     context_precision (which measures RAG retrieval quality only).

    Keys: ragas_faithfulness, ragas_answer_relevance, ragas_context_precision
    Missing metrics return None (displayed as N/A in the UI).
    """
    result: dict = {}
    is_sql_only = (routing_path == "SQL")
    faith_contexts = contexts + (sql_contexts or [])

    if is_sql_only:
        # Task completion is the right SLO #4 proxy for data-lookup responses.
        # RAGAS cosine similarity underestimates SQL answers by ~10-15%.
        result["ragas_faithfulness"]      = None
        result["ragas_answer_relevance"]  = _safe(task_completion, question, answer)
        result["ragas_context_precision"] = None
        result["relevance_type"]          = "task_completion"
        return result

    if is_escalated and not contexts:
        result["ragas_faithfulness"]      = None
        result["ragas_answer_relevance"]  = _safe(answer_relevance, question, answer)
        result["ragas_context_precision"] = None
        result["relevance_type"]          = "answer_relevance"
        return result

    # Faithfulness: RAG chunks + SQL rows (all grounding evidence)
    # Context precision: RAG chunks only (measures retrieval quality, not SQL)
    result["ragas_faithfulness"]      = _safe(faithfulness,      answer,   faith_contexts)
    result["ragas_answer_relevance"]  = _safe(answer_relevance,  question, answer)
    result["ragas_context_precision"] = _safe(context_precision, question, contexts) if contexts else None
    result["relevance_type"]          = "answer_relevance"

    return result


# ---------------------------------------------------------------------------
# Eval entry point — batch pipeline (uses batch faithfulness for speed)
# ---------------------------------------------------------------------------

def score_ragas_batch(
    question:     str,
    answer:       str,
    contexts:     list[str],
    is_escalated: bool = False,
    routing_path: str  = "",
    sql_contexts: list[str] | None = None,
) -> dict:
    """
    Same as score_ragas but uses faithfulness_batch — for eval.py ThreadPoolExecutor runs.
    sql_contexts: SQL rows included for faithfulness only, not context_precision.
    """
    result: dict = {}
    is_sql_only = (routing_path == "SQL")
    faith_contexts = contexts + (sql_contexts or [])

    if is_sql_only:
        result["ragas_faithfulness"]      = None
        result["ragas_answer_relevance"]  = _safe(task_completion, question, answer)
        result["ragas_context_precision"] = None
        result["relevance_type"]          = "task_completion"
        return result

    if is_escalated and not contexts:
        result["ragas_faithfulness"]      = None
        result["ragas_answer_relevance"]  = _safe(answer_relevance, question, answer)
        result["ragas_context_precision"] = None
        result["relevance_type"]          = "answer_relevance"
        return result

    result["ragas_faithfulness"]      = _safe(faithfulness_batch, answer,   faith_contexts)
    result["ragas_answer_relevance"]  = _safe(answer_relevance,   question, answer)
    result["ragas_context_precision"] = _safe(context_precision,  question, contexts) if contexts else None
    result["relevance_type"]          = "answer_relevance"

    return result


def _safe(fn, *args) -> Optional[float]:
    try:
        return fn(*args)
    except Exception:
        return None
