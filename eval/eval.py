# Evaluation Framework
#
# Measures system quality across 9 dimensions:
#   1.  Classification accuracy — category, severity, routing_path vs golden set
#   2.  Per-category accuracy  — breakdown so a broken category isn't hidden in TSR
#   3.  Confidence calibration — does high confidence actually mean higher accuracy?
#   4.  Misclassification matrix — which categories get confused with which
#   5.  Escalation recall      — 100% required, SLO #11 (Sprint 6)
#   6.  Escalation precision   — over-escalation tracking (Sprint 6)
#   7.  ADR-008 detection rate — secondary signal in reasoning field (Sprint 5)
#   8.  Response quality       — LLM-as-judge scores against answer_rubric (Sprint 4)
#   9.  Failure log            — tc_id, expected vs actual, confidence, reasoning
#
# TSR formula (ADR-007): TSR = cases_passed / total_cases × 100
# Pass threshold: TSR >= 90%
#
# ADR-006 — LLM-as-judge for golden-set evaluation
# ADR-007 — TSR >= 90% binary pass/fail + escalation = success

import asyncio
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv

import nest_asyncio

# Auto-patch every event loop created in any thread (ADR-013).
# nest_asyncio.apply() patches only the *current* loop — LlamaIndex/asyncpg creates
# new loops inside ThreadPoolExecutor workers, so those would be unpatched.
# This policy subclass patches each new loop at construction time.
class _PatchedLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()
        nest_asyncio.apply(loop)
        return loop

asyncio.set_event_loop_policy(_PatchedLoopPolicy())
nest_asyncio.apply()  # patch the main thread's existing loop too

load_dotenv()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
TSR_PASS_THRESHOLD = 90.0

# Guards progress-line prints when cases run in parallel (ADR-013)
_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    tc_id: str
    input_text: str

    # Expected values from golden set
    expected_category: str
    expected_severity: str
    expected_route: str
    expected_escalation: bool
    answer_rubric: str

    # Agent 1 actual output
    actual_category: Optional[str] = None
    actual_severity: Optional[str] = None
    actual_route: Optional[str] = None
    actual_confidence: Optional[float] = None
    actual_reasoning: Optional[str] = None

    # Classification pass/fail flags
    category_match: bool = False
    severity_match: bool = False
    route_match: bool = False
    classification_passed: bool = False     # all three must match

    # Escalation (Sprint 6)
    actual_escalation: Optional[bool] = None

    # ADR-008 secondary signal present in reasoning (Sprint 5)
    secondary_signal_detected: Optional[bool] = None

    # Full pipeline output (populated by run_full_pipeline_eval)
    final_response:   Optional[str]       = None
    rag_context:      Optional[str]       = None   # raw text of retrieved chunks
    cited_sources:    Optional[list[str]] = None   # SLO #8: sources the synthesizer cited

    # Heuristic scores — fast, no LLM cost (populated by run_heuristic_scores)
    heuristic_citation_present: Optional[bool]  = None   # SLO #8
    heuristic_length_ok:        Optional[bool]  = None   # response >= 50 chars
    heuristic_rubric_coverage:  Optional[float] = None   # fraction of rubric keywords found

    # LLM-as-judge scores — semantic quality (populated by score_with_llm_judge)
    llm_judge_faithfulness: Optional[float] = None   # SLO #2: >= 0.95
    llm_judge_relevance:    Optional[float] = None   # SLO #4: >= 0.85
    llm_judge_reasoning:    Optional[str]   = None

    # Legacy field kept for API compatibility
    llm_judge_score: Optional[float] = None

    # Error during evaluation run
    error: Optional[str] = None


@dataclass
class EvalReport:
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    tsr: float = 0.0

    # Per-category: {category: {"passed": int, "total": int, "accuracy": float}}
    per_category: dict = field(default_factory=dict)

    # Confusion matrix for failures: {expected: {actual: count}}
    misclassification_matrix: dict = field(default_factory=dict)

    # Confidence calibration: {bucket: {"correct": int, "total": int}}
    # Buckets: "high" (>=0.85), "medium" (0.70–0.84), "low" (<0.70)
    confidence_calibration: dict = field(default_factory=dict)

    # Escalation metrics (Sprint 6)
    escalation_recall: Optional[float] = None
    escalation_precision: Optional[float] = None

    # ADR-008 (Sprint 5)
    adr008_detection_rate: Optional[float] = None

    # LLM-as-judge (Sprint 4)
    avg_judge_score: Optional[float] = None

    # Context Precision — SLO #9: >= 0.80 (Sprint 5)
    context_precision: Optional[float] = None

    # Answer Relevance — SLO #4: >= 0.85 (Sprint 4)
    answer_relevance: Optional[float] = None

    # All failed EvalResults for the failure log
    failures: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Load golden set
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Classification check — live (Agent 1 is built)
# ---------------------------------------------------------------------------

def run_classification_check(tc: dict, agent1_fn: Callable) -> EvalResult:
    """
    Runs Agent 1 on one golden set case and compares to expected values.

    Requires the golden set case to have an 'expected_category' field with the
    exact Literal value (e.g. 'usage_configuration', 'integration_api').
    The 'category' field in the golden set is a grouping label, not this value —
    TC-01 to TC-18 already have expected_category; TC-19+ need it added.
    """
    result = EvalResult(
        tc_id=tc["tc_id"],
        input_text=tc["input"],
        expected_category=tc.get("expected_category", ""),
        expected_severity=tc.get("severity", ""),
        expected_route=tc.get("route", ""),
        expected_escalation=tc.get("expected_escalation", False),
        answer_rubric=tc.get("answer_rubric", ""),
    )

    if tc.get("expected_outcome") == "Blocked":
        result.error = "guardrail test — evaluated by check_guardrail_blocks() in Sprint 7 (SLO #13)"
        return result

    if not result.expected_category:
        result.error = "missing 'expected_category' field — add exact Literal value to this golden set case"
        return result

    try:
        from src.graph.state import TicketState
        state: TicketState = {
            "ticket_text": tc["input"],
            "conversation_history": [],
            "classification": None,
            "rag_result": None,
            "sql_result": None,
            "severity_assessment": None,
            "escalation_package": None,
            "final_response": None,
        }
        state = agent1_fn(state)
        c = state["classification"]

        result.actual_category  = c["category"]
        result.actual_severity  = c["severity"]
        result.actual_route     = c["routing_path"]
        result.actual_confidence = c["confidence"]
        result.actual_reasoning  = c["reasoning"]

        result.category_match = result.actual_category == result.expected_category
        result.severity_match = result.actual_severity == result.expected_severity
        result.route_match    = result.actual_route    == result.expected_route
        result.classification_passed = (
            result.category_match and result.severity_match and result.route_match
        )

    except Exception as e:
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# 2. Per-category accuracy
# ---------------------------------------------------------------------------

def build_per_category_accuracy(results: list[EvalResult]) -> dict:
    counts: dict = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        if r.error:
            continue
        counts[r.expected_category]["total"] += 1
        if r.classification_passed:
            counts[r.expected_category]["passed"] += 1
    return {
        cat: {
            **v,
            "accuracy": round(v["passed"] / v["total"] * 100, 1) if v["total"] else 0.0,
        }
        for cat, v in sorted(counts.items())
    }


# ---------------------------------------------------------------------------
# 3. Confidence calibration
# ---------------------------------------------------------------------------

def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.70:
        return "medium"
    return "low"


def build_confidence_calibration(results: list[EvalResult]) -> dict:
    """
    Checks whether high confidence actually correlates with correct classification.
    If high-bucket accuracy < medium-bucket, the confidence scale is miscalibrated
    and the thresholds in agent4_severity.py need adjustment.
    """
    buckets: dict = {
        "high":   {"correct": 0, "total": 0},
        "medium": {"correct": 0, "total": 0},
        "low":    {"correct": 0, "total": 0},
    }
    for r in results:
        if r.actual_confidence is None or r.error:
            continue
        bucket = _confidence_bucket(r.actual_confidence)
        buckets[bucket]["total"] += 1
        if r.classification_passed:
            buckets[bucket]["correct"] += 1
    return buckets


# ---------------------------------------------------------------------------
# 4. Misclassification matrix
# ---------------------------------------------------------------------------

def build_misclassification_matrix(results: list[EvalResult]) -> dict:
    """
    For every failed case, records {expected_category: {actual_category: count}}.
    Reveals systematic confusions — e.g. production_incident mislabelled as
    integration_api repeatedly means the priority ladder needs a stronger example.
    """
    matrix: dict = defaultdict(lambda: defaultdict(int))
    for r in results:
        if not r.classification_passed and not r.error and r.actual_category:
            matrix[r.expected_category][r.actual_category] += 1
    return {k: dict(v) for k, v in matrix.items()}


# ---------------------------------------------------------------------------
# 5 & 6. Escalation recall + precision — Sprint 6
# ---------------------------------------------------------------------------

def check_guardrail_blocks(guardrail_fn: Callable, golden_set: list[dict]) -> list[EvalResult]:
    """
    SLO #13 — Guardrail Effectiveness: 100% of adversarial inputs must be blocked.
    Runs TC-09, TC-48, TC-49 (expected_outcome == "Blocked") through the guardrail layer.
    Pass = request blocked before reaching Agent 1.
    Fail = request leaked through.
    """
    from src.guardrails.guardrails import GuardrailResult

    blocked_cases = [tc for tc in golden_set if tc.get("expected_outcome") == "Blocked"]
    results = []

    for tc in blocked_cases:
        result = EvalResult(
            tc_id=tc["tc_id"],
            input_text=tc["input"],
            expected_category=tc.get("expected_category", ""),
            expected_severity=tc.get("severity", ""),
            expected_route=tc.get("route", ""),
            expected_escalation=tc.get("expected_escalation", False),
            answer_rubric=tc.get("answer_rubric", ""),
        )
        try:
            guardrail_result: GuardrailResult = guardrail_fn(tc["input"])
            if guardrail_result.blocked:
                result.classification_passed = True
            else:
                result.classification_passed = False
                result.error = f"NOT BLOCKED — leaked through as: {guardrail_result.message or 'no message'}"
        except Exception as e:
            result.error = str(e)
        results.append(result)

    return results


def check_escalation_recall(results: list[EvalResult]) -> float:
    """SLO #11: of all tickets that required escalation, what % were actually escalated? Target: 100%."""
    required = [r for r in results if r.expected_escalation and not r.error]
    if not required:
        return 1.0
    correctly_escalated = sum(1 for r in required if r.actual_escalation)
    return round(correctly_escalated / len(required), 3)


def check_escalation_precision(results: list[EvalResult]) -> float:
    """Of all tickets that were escalated, what % actually required escalation? Measures over-escalation."""
    actually_escalated = [r for r in results if r.actual_escalation and not r.error]
    if not actually_escalated:
        return 1.0
    correct = sum(1 for r in actually_escalated if r.expected_escalation)
    return round(correct / len(actually_escalated), 3)


# ---------------------------------------------------------------------------
# 7. ADR-008 secondary signal detection — Sprint 5
# ---------------------------------------------------------------------------

def check_adr008_detection_rate(_results: list[EvalResult]) -> float:
    # Of multi-topic tickets (expected_category has a secondary signal), what %
    # had the secondary signal mentioned in Agent 1's reasoning field
    raise NotImplementedError("ADR-008 secondary signal detection — Sprint 5 (full pipeline needed)")


# ---------------------------------------------------------------------------
# 7b. Heuristic scoring — fast, zero LLM cost
#
# Three checks run on every response before the LLM judge:
#   1. Citation presence  — SLO #8: does response contain "[Source" pattern?
#   2. Length sanity      — response >= 50 chars (catches empty/error responses)
#   3. Rubric coverage    — fraction of key terms from answer_rubric found in response
#
# Rubric coverage is a cheap proxy for relevance. It uses single-word tokens
# (len >= 4, not stopwords) so common filler words don't inflate the score.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "must", "should", "with", "from", "that", "this", "have", "will",
    "provide", "include", "response", "answer", "give", "show", "the",
    "and", "for", "not", "been", "used", "which", "also", "only",
}


def run_heuristic_scores(result: EvalResult) -> EvalResult:
    if not result.final_response:
        result.heuristic_citation_present = False
        result.heuristic_length_ok        = False
        result.heuristic_rubric_coverage  = 0.0
        return result

    resp_lower = result.final_response.lower()

    # SLO #8: check cited_sources list (set by synthesizer structured output) first,
    # fall back to scanning response text for inline [Source ...] markers.
    if result.cited_sources is not None:
        result.heuristic_citation_present = len(result.cited_sources) > 0
    else:
        result.heuristic_citation_present = bool(re.search(r"\[source", resp_lower))
    result.heuristic_length_ok        = len(result.final_response.strip()) >= 50

    rubric = result.answer_rubric or ""
    tokens = [
        w.lower() for w in re.findall(r"[a-zA-Z]+", rubric)
        if len(w) >= 4 and w.lower() not in _STOPWORDS
    ]
    if tokens:
        found = sum(1 for t in tokens if t in resp_lower)
        result.heuristic_rubric_coverage = round(found / len(tokens), 2)
    else:
        result.heuristic_rubric_coverage = 1.0  # no rubric = no penalty

    return result


# ---------------------------------------------------------------------------
# 8. LLM-as-judge — semantic quality scoring
#
# Runs after heuristics. Scores two dimensions in one LLM call (ADR-006):
#   faithfulness (SLO #2 >= 0.95): every claim grounded in retrieved context
#   relevance    (SLO #4 >= 0.85): response addresses the customer's question
#
# For escalated tickets rag_context is empty — judge scores relevance only
# and faithfulness is set to 1.0 (Agent 5 responses are template-based).
# ---------------------------------------------------------------------------

def score_with_llm_judge(
    tc: dict, actual_response: str, rag_context: str = "", is_escalated: bool = False
) -> EvalResult:
    """
    Scores actual_response against tc['answer_rubric'] using GPT-4o mini.
    Returns a populated EvalResult fragment (only judge fields set).
    Raises on LLM error — caller handles retry/skip.

    is_escalated: True when Agent 4 routed this ticket to Agent 5. Agent 5's
    response is a template-based escalation acknowledgment, not a RAG-grounded
    answer — RAG chunks retrieved earlier in the pipeline (e.g. on Hybrid/
    Multi-Agent routes) are irrelevant to judging it. Faithfulness is skipped
    entirely (1.0) and relevance is judged against "did it correctly
    acknowledge the issue and escalate", not "did it resolve the ticket".
    """
    from langchain_openai import AzureChatOpenAI
    from pydantic import BaseModel as _BaseModel

    class _JudgeOutput(_BaseModel):
        faithfulness: float   # 0.0–1.0
        relevance:    float   # 0.0–1.0
        reasoning:    str

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
    ).with_structured_output(_JudgeOutput)

    if is_escalated:
        context_section = (
            "This ticket was escalated to a human team — the response below is an "
            "escalation acknowledgment, not a documentation-grounded answer. "
            "Do not compare it against any retrieved documentation."
        )
        relevance_instruction = (
            "relevance: Does the response correctly acknowledge the customer's issue, "
            "convey appropriate urgency for the severity, and clearly state that it has "
            "been escalated (per the rubric)? Do not penalise it for not resolving the "
            "issue directly — escalation is the correct and expected action.\n"
        )
    else:
        context_section = (
            f"Retrieved context provided to the system:\n{rag_context}"
            if rag_context
            else "No documentation context was retrieved (SQL-only ticket)."
        )
        relevance_instruction = (
            "relevance: Does the response directly address what the customer asked "
            "and satisfy the rubric criteria?\n"
        )

    prompt = (
        f"You are evaluating an AI customer support response. Score it on two dimensions.\n\n"
        f"Customer question:\n{tc['input']}\n\n"
        f"Expected answer criteria (rubric):\n{tc.get('answer_rubric', 'No rubric provided')}\n\n"
        f"{context_section}\n\n"
        f"Actual response:\n{actual_response}\n\n"
        "Score 0.0 to 1.0 for each:\n"
        "faithfulness: Are all claims in the response supported by the retrieved context or the rubric? "
        "Penalise any hallucinated details, invented method names, or made-up URLs.\n"
        f"{relevance_instruction}"
        "reasoning: One sentence explaining the scores."
    )

    scored = llm.invoke(prompt)

    # Escalated tickets: Agent 5 responses are template-based, not RAG-grounded.
    # Faithfulness check doesn't apply — set to 1.0 to avoid false penalisation.
    faithfulness = 1.0 if is_escalated else scored.faithfulness

    # Return a plain dict so the caller can copy fields onto its EvalResult
    return {
        "llm_judge_faithfulness": round(faithfulness, 3),
        "llm_judge_relevance":    round(scored.relevance, 3),
        "llm_judge_reasoning":    scored.reasoning,
        "llm_judge_score":        round((faithfulness + scored.relevance) / 2, 3),
    }


def run_guardrail_eval() -> None:
    """
    SLO #13: Guardrail Effectiveness — 100% of adversarial test set blocked.
    Tests TC-09 (prompt injection), TC-48 (policy override), TC-49 (unauthorized action).
    Runs entirely offline — no LLM calls, no graph invocation.
    """
    from src.guardrails.guardrails import check_guardrails

    golden_set = load_golden_set()
    blocked_cases = [tc for tc in golden_set if tc.get("expected_outcome") == "Blocked"]

    print("\n" + "=" * 62)
    print("GUARDRAIL EVAL (SLO #13)")
    print("=" * 62)
    passed = 0
    for tc in blocked_cases:
        result = check_guardrails(tc["input"])
        status = "BLOCKED [PASS]" if result.blocked else "PASSED THROUGH [FAIL]"
        reason = result.reason.value if result.reason else "—"
        print(f"  {tc['tc_id']}  [{status}]  reason={reason}")
        print(f"         input: {tc['input'][:80]}...")
        if result.blocked:
            passed += 1

    total = len(blocked_cases)
    pct   = round(passed / total * 100, 1) if total else 0
    slo   = "PASS" if passed == total else "FAIL"
    print(f"\n  Blocked: {passed}/{total}  ({pct}%)  SLO #13: {slo}")
    print("=" * 62)


def check_context_precision(_retrieved_chunks: list, _relevant_chunk_ids: list) -> float:
    """SLO #9 — Context Precision >= 0.80. Requires labelled relevance per chunk."""
    raise NotImplementedError("Context precision — requires per-chunk relevance labels (future sprint)")


def check_answer_relevance(question: str, actual_response: str) -> float:
    """SLO #4 — delegates to score_with_llm_judge; use run_full_pipeline_eval instead."""
    raise NotImplementedError("Use run_full_pipeline_eval() which calls score_with_llm_judge per case")


# ---------------------------------------------------------------------------
# Full-pipeline evaluation
#
# Runs compiled_graph.invoke() on each golden set case, then applies
# heuristic scoring and LLM-as-judge. Skips Blocked cases (guardrail eval).
# Runs on TC-01 to TC-18 by default (original 18 cases, all have rubrics).
# Pass tc_ids=[...] to restrict to a subset.
# ---------------------------------------------------------------------------

def _eval_single_case(tc: dict, skip_judge: bool) -> EvalResult:
    """
    Runs one golden set case through the full pipeline and scores it.
    Designed to be called from a thread — each invocation gets its own
    graph thread_id (UUID) so MemorySaver state never leaks between cases.
    Each case opens a Langfuse trace so LLM judge scores appear alongside
    agent child spans in the Tracing dashboard (Option A — ADR-013).
    """
    import uuid
    from src.graph.graph import compiled_graph
    from src.graph.state import TicketState
    from src.observability.langfuse_client import start_trace, log_ticket_complete, timer as lf_timer

    result = EvalResult(
        tc_id=tc["tc_id"],
        input_text=tc["input"],
        expected_category=tc.get("expected_category", ""),
        expected_severity=tc.get("severity", ""),
        expected_route=tc.get("route", ""),
        expected_escalation=tc.get("expected_escalation", False),
        answer_rubric=tc.get("answer_rubric", ""),
    )

    t0 = lf_timer()
    root, lf_token = start_trace(tc["input"])

    try:
        state: TicketState = {
            "ticket_text":          tc["input"],
            "conversation_history": tc.get("conversation_history", []),
            "classification":       None,
            "rag_result":           None,
            "sql_result":           None,
            "severity_assessment":  None,
            "escalation_package":   None,
            "final_response":       None,
            "reflection_count":     0,
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state  = compiled_graph.invoke(state, config=config)

        result.final_response    = state.get("final_response") or ""
        result.actual_category   = (state.get("classification") or {}).get("category")
        result.actual_severity   = (state.get("classification") or {}).get("severity")
        result.actual_route      = (state.get("classification") or {}).get("routing_path")
        result.actual_confidence = (state.get("classification") or {}).get("confidence")
        result.actual_escalation = (state.get("severity_assessment") or {}).get("escalate", False)

        rag_result = state.get("rag_result") or {}
        chunks     = rag_result.get("chunks", [])
        if chunks:
            result.rag_context = "\n\n".join(
                f"[{c.get('source','?')}] {str(c.get('text',''))[:300]}"
                for c in chunks[:3]
            )
        else:
            # SQL-only route: use SQL result rows as context for faithfulness check.
            # Without this, the judge sees "no context" and marks every claim unsupported → 0.0.
            sql_result = state.get("sql_result") or {}
            rows       = (sql_result.get("rows") or [])[:20]
            result.rag_context = ("SQL query results:\n" + "\n".join(str(r) for r in rows)) if rows else ""
        result.cited_sources = rag_result.get("cited_sources") or []

        result = run_heuristic_scores(result)

        if not skip_judge and result.final_response:
            try:
                scored = score_with_llm_judge(
                    tc, result.final_response, result.rag_context,
                    is_escalated=bool(result.actual_escalation),
                )
                result.llm_judge_faithfulness = scored["llm_judge_faithfulness"]
                result.llm_judge_relevance    = scored["llm_judge_relevance"]
                result.llm_judge_reasoning    = scored["llm_judge_reasoning"]
                result.llm_judge_score        = scored["llm_judge_score"]

                # Push LLM judge scores to this trace — visible in Langfuse Scores tab
                root.score_trace(
                    name="faithfulness",
                    value=result.llm_judge_faithfulness,
                    comment="LLM judge — SLO #2 target >= 0.95",
                )
                root.score_trace(
                    name="relevance",
                    value=result.llm_judge_relevance,
                    comment="LLM judge — SLO #4 target >= 0.85",
                )
            except Exception as e:
                result.llm_judge_reasoning = f"Judge error: {e}"

    except Exception as e:
        result.error = str(e)

    finally:
        tsr_pass = not (result.actual_confidence == 0.0 and result.actual_category == "ambiguous")
        log_ticket_complete(
            root, lf_token,
            final_response=result.final_response or "",
            total_cost_usd=0.0,
            total_latency_ms=lf_timer() - t0,
            tsr_pass=tsr_pass if result.actual_category else False,
            routing_path=result.actual_route or "",
        )

    status = "OK" if not result.error else "ERR"
    with _print_lock:
        print(f"  [{status}] {result.tc_id}  faith={result.llm_judge_faithfulness}  rel={result.llm_judge_relevance}  rubric_cov={result.heuristic_rubric_coverage}")

    return result


def run_full_pipeline_eval(
    tc_ids: Optional[list[str]] = None,
    skip_judge: bool = False,
    max_workers: int = 1,
) -> list[EvalResult]:
    """
    Runs the full agent graph on golden set cases and scores each response.

    Args:
        tc_ids:      List of tc_id strings to run. Defaults to TC-01 through TC-18.
        skip_judge:  If True, run heuristics only (faster, no LLM cost).
        max_workers: Thread pool size. Default 1 (sequential) — asyncpg's internal
                     connection pool attaches futures to the creating event loop; multiple
                     threads each spin up their own loop, causing "Future attached to a
                     different loop" errors under concurrent execution. Sequential eval
                     avoids this entirely. Pass max_workers=4 only if you accept ~50%
                     flaky failures on RAG/Hybrid cases. (ADR-013)

    Returns list of EvalResult sorted by tc_id with final_response, heuristic,
    and judge fields set.
    """
    golden_set  = load_golden_set()
    default_ids = {f"TC-{str(i).zfill(2)}" for i in range(1, 56)}
    target_ids  = set(tc_ids) if tc_ids else default_ids

    target_cases = [
        tc for tc in golden_set
        if tc["tc_id"] in target_ids and tc.get("expected_outcome") != "Blocked"
    ]

    results: list[EvalResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_single_case, tc, skip_judge): tc for tc in target_cases}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r.tc_id)
    return results


def print_pipeline_report(results: list[EvalResult]) -> None:
    scored = [r for r in results if not r.error and r.llm_judge_faithfulness is not None]
    heuristic_only = [r for r in results if not r.error and r.heuristic_rubric_coverage is not None]

    print("\n" + "=" * 62)
    print("FULL PIPELINE EVAL REPORT")
    print("=" * 62)
    print(f"Cases run          : {len(results)}")
    print(f"Errors             : {sum(1 for r in results if r.error)}")

    if heuristic_only:
        cit  = sum(1 for r in heuristic_only if r.heuristic_citation_present)
        lng  = sum(1 for r in heuristic_only if r.heuristic_length_ok)
        avg_cov = round(sum(r.heuristic_rubric_coverage for r in heuristic_only) / len(heuristic_only), 2)
        print(f"\n--- Heuristic Scores (n={len(heuristic_only)}) ---")
        print(f"  Citation present (SLO #8) : {cit}/{len(heuristic_only)}  ({'PASS' if cit == len(heuristic_only) else 'FAIL'})")
        print(f"  Length OK (>= 50 chars)   : {lng}/{len(heuristic_only)}")
        print(f"  Avg rubric keyword cov    : {avg_cov}  (proxy for relevance)")

    # SLO #11: Escalation Recall & Precision
    all_valid = [r for r in results if not r.error and r.actual_escalation is not None]
    if all_valid:
        recall    = check_escalation_recall(all_valid)
        precision = check_escalation_precision(all_valid)
        required  = [r for r in all_valid if r.expected_escalation]
        missed    = [r for r in required if not r.actual_escalation]
        over_esc  = [r for r in all_valid if r.actual_escalation and not r.expected_escalation]
        print(f"\n--- Escalation (SLO #11) ---")
        print(f"  Recall    (required->escalated) : {recall:.1%}  target 100%  {'PASS' if recall == 1.0 else 'FAIL'}")
        print(f"  Precision (escalated->required) : {precision:.1%}  (over-escalation check)")
        if missed:
            print(f"  MISSED escalations : {', '.join(r.tc_id for r in missed)}")
        if over_esc:
            print(f"  Over-escalated     : {', '.join(r.tc_id for r in over_esc)}")

    if scored:
        avg_faith = round(sum(r.llm_judge_faithfulness for r in scored) / len(scored), 3)
        avg_rel   = round(sum(r.llm_judge_relevance    for r in scored) / len(scored), 3)
        faith_pass = sum(1 for r in scored if r.llm_judge_faithfulness >= 0.95)
        rel_pass   = sum(1 for r in scored if r.llm_judge_relevance    >= 0.85)
        print(f"\n--- LLM Judge Scores (n={len(scored)}) ---")
        print(f"  Avg faithfulness (SLO #2) : {avg_faith}  target >= 0.95  {'PASS' if avg_faith >= 0.95 else 'FAIL'}")
        print(f"  Avg relevance    (SLO #4) : {avg_rel}   target >= 0.85  {'PASS' if avg_rel   >= 0.85 else 'FAIL'}")
        print(f"  Cases faith >= 0.95       : {faith_pass}/{len(scored)}")
        print(f"  Cases rel   >= 0.85       : {rel_pass}/{len(scored)}")

    print("\n--- Per-Case Detail ---")
    print(f"  {'TC':6}  {'Route':12}  {'Cit':3}  {'Esc':3}  {'Faith':6}  {'Rel':6}  {'Cov':5}  Reasoning / Sources")
    for r in results:
        if r.error:
            print(f"  {r.tc_id:<6}  {'ERROR':12}  {'—':3}  {'—':3}  {'—':6}  {'—':6}  {'—':5}  {r.error[:50]}")
            continue
        route  = r.actual_route or r.expected_route or "?"
        cit    = "Y" if r.heuristic_citation_present else "N"
        esc    = "Y" if r.actual_escalation else "N"
        srcs   = ", ".join((r.cited_sources or [])[:2]) or "none"
        detail = (r.llm_judge_reasoning or f"sources: {srcs}")[:50]
        print(
            f"  {r.tc_id:<6}  {route:<12}  {cit:<3}  {esc:<3}  "
            f"{('—' if r.llm_judge_faithfulness is None else str(r.llm_judge_faithfulness)):<6}  "
            f"{('—' if r.llm_judge_relevance    is None else str(r.llm_judge_relevance)):<6}  "
            f"{('—' if r.heuristic_rubric_coverage is None else str(r.heuristic_rubric_coverage)):<5}  "
            f"{detail}"
        )
    print("=" * 62)


# ---------------------------------------------------------------------------
# TSR calculation (ADR-007)
# ---------------------------------------------------------------------------

def calculate_tsr(results: list[EvalResult]) -> float:
    scored = [r for r in results if not r.error]
    if not scored:
        return 0.0
    passed = sum(1 for r in scored if r.classification_passed)
    return round(passed / len(scored) * 100, 2)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(agent1_fn: Callable) -> EvalReport:
    from src.guardrails.guardrails import check_guardrails

    golden_set = load_golden_set()

    # Classification check (excludes Blocked cases — handled separately)
    classification_results = [run_classification_check(tc, agent1_fn) for tc in golden_set]

    # Guardrail check — TC-09, TC-48, TC-49
    guardrail_results = check_guardrail_blocks(check_guardrails, golden_set)

    # Merge: replace the "guardrail test — pending" placeholders with real results
    guardrail_ids = {r.tc_id for r in guardrail_results}
    results = [r for r in classification_results if r.tc_id not in guardrail_ids] + guardrail_results

    report = EvalReport()
    report.total_cases = len(results)
    report.passed      = sum(1 for r in results if r.classification_passed)
    report.failed      = report.total_cases - report.passed
    report.tsr         = calculate_tsr(results)

    report.per_category             = build_per_category_accuracy(results)
    report.misclassification_matrix = build_misclassification_matrix(results)
    report.confidence_calibration   = build_confidence_calibration(results)
    report.failures                 = [r for r in results if not r.classification_passed]

    return report


def print_report(report: EvalReport, results: list | None = None) -> None:
    if results is None:
        results = report.failures
    print("\n" + "=" * 62)
    print("EVALUATION REPORT")
    print("=" * 62)
    print(f"Total cases  : {report.total_cases}")
    print(f"Passed       : {report.passed}")
    print(f"Failed       : {report.failed}")
    print(f"TSR          : {report.tsr}%  (threshold: >= {TSR_PASS_THRESHOLD}%)")
    print(f"Status       : {'PASS' if report.tsr >= TSR_PASS_THRESHOLD else 'FAIL'}")

    print("\n--- Per-Category Accuracy ---")
    for cat, stats in report.per_category.items():
        bar = "OK" if stats["accuracy"] >= 90 else "!!"
        print(f"  [{bar}] {cat:<25}  {stats['passed']}/{stats['total']}  ({stats['accuracy']}%)")

    print("\n--- Confidence Calibration ---")
    for bucket, stats in report.confidence_calibration.items():
        if stats["total"] == 0:
            continue
        acc = round(stats["correct"] / stats["total"] * 100, 1)
        print(f"  {bucket:<8}  {stats['correct']}/{stats['total']}  ({acc}%)")

    if report.misclassification_matrix:
        print("\n--- Misclassification Matrix (failures only) ---")
        for expected, actuals in report.misclassification_matrix.items():
            for actual, count in actuals.items():
                print(f"  {expected:<25} -> {actual:<25}  x{count}")

    classification_failures = [r for r in report.failures if "guardrail test" not in (r.error or "")]
    if classification_failures:
        print("\n--- Failure Log ---")
        for r in classification_failures:
            print(f"\n  [{r.tc_id}]  {r.input_text[:65]}...")
            if r.error:
                print(f"  ERROR   : {r.error}")
            else:
                print(f"  Expected   : {r.expected_category} / {r.expected_severity} / {r.expected_route}")
                print(f"  Actual     : {r.actual_category} / {r.actual_severity} / {r.actual_route}")
                print(f"  Confidence : {r.actual_confidence}")
                print(f"  Reasoning  : {r.actual_reasoning}")

    guardrail_results = [r for r in results if r.tc_id in {"TC-09", "TC-48", "TC-49"}]
    guardrail_failures = [r for r in guardrail_results if not r.classification_passed]
    guardrail_passed   = len(guardrail_results) - len(guardrail_failures)
    print(f"\n--- Guardrail Effectiveness (SLO #13) ---")
    print(f"  Blocked correctly : {guardrail_passed}/{len(guardrail_results)}")
    print(f"  Status            : {'PASS (100%)' if not guardrail_failures else 'FAIL'}")
    for r in guardrail_results:
        status = "BLOCKED" if r.classification_passed else "LEAKED"
        print(f"  [{r.tc_id}] {status}  {r.input_text[:55]}...")
    if guardrail_failures:
        for r in guardrail_failures:
            print(f"  ERROR: {r.error}")

    print("\n--- Pending (later sprints) ---")
    print("  LLM-as-judge response quality  : Sprint 4  (SLO #2 faithfulness >= 95%)")
    print("  Answer relevance               : Sprint 4  (SLO #4 >= 0.85)")
    print("  ADR-008 secondary signal rate  : Sprint 5  (ADR-008)")
    print("  Context precision              : Sprint 5  (SLO #9 >= 0.80)")
    print("  Escalation recall / precision  : Sprint 6  (SLO #11 recall = 100%)")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.agents.agent1_classify import agent1_classify

    if "--push-dataset" in sys.argv:
        # python -m eval.eval --push-dataset
        from eval.langfuse_eval import push_golden_set_to_langfuse
        push_golden_set_to_langfuse()

    elif "--langfuse" in sys.argv:
        # python -m eval.eval --langfuse agent1-classify-v1
        from eval.langfuse_eval import run_evaluation_with_langfuse
        run_name = sys.argv[sys.argv.index("--langfuse") + 1] if len(sys.argv) > sys.argv.index("--langfuse") + 1 else "agent1-classify"
        report = run_evaluation_with_langfuse(agent1_classify, run_name=run_name)
        print_report(report)

    elif "--pipeline" in sys.argv:
        # python -m eval.eval --pipeline
        # python -m eval.eval --pipeline --heuristic-only     (skip LLM judge, faster)
        # python -m eval.eval --pipeline TC-01 TC-03 TC-06    (specific cases)
        import json as _json
        from datetime import datetime as _dt
        skip_judge = "--heuristic-only" in sys.argv
        tc_ids = [a for a in sys.argv[sys.argv.index("--pipeline") + 1:] if a.startswith("TC-")] or None
        print(f"Running full pipeline eval on {tc_ids or 'TC-01–TC-55'} (judge={'OFF' if skip_judge else 'ON'})...")
        pipeline_results = run_full_pipeline_eval(tc_ids=tc_ids, skip_judge=skip_judge)
        print_pipeline_report(pipeline_results)

        # Save results to JSON for historical comparison
        _scored   = [r for r in pipeline_results if not r.error and r.llm_judge_faithfulness is not None]
        _esc_cases = [r for r in pipeline_results if not r.error and r.actual_escalation is not None]
        _esc_recall = round(check_escalation_recall(_esc_cases), 3)
        _avg_faith  = round(sum(r.llm_judge_faithfulness for r in _scored) / max(1, len(_scored)), 3)
        _avg_rel    = round(sum(r.llm_judge_relevance    for r in _scored) / max(1, len(_scored)), 3)

        def _route_avg(route: str, attr: str) -> float | None:
            rs = [getattr(r, attr) for r in _scored if r.actual_route == route and getattr(r, attr) is not None]
            return round(sum(rs) / len(rs), 3) if rs else None

        out = {
            "run_at":    _dt.utcnow().isoformat() + "Z",
            "tc_ids":    tc_ids or "all",
            "skip_judge": skip_judge,
            "summary": {
                "total":  len(pipeline_results),
                "errors": sum(1 for r in pipeline_results if r.error),
                "scored": len(_scored),
                # Aggregate metrics
                "avg_faithfulness":  _avg_faith,
                "avg_relevance":     _avg_rel,
                "escalation_recall": _esc_recall,
                # SLO pass/fail
                "slo": {
                    "SLO2_faithfulness_095":   {"target": 0.95, "actual": _avg_faith,  "pass": _avg_faith  >= 0.95},
                    "SLO4_relevance_085":      {"target": 0.85, "actual": _avg_rel,    "pass": _avg_rel    >= 0.85},
                    "SLO11_escalation_recall": {"target": 1.0,  "actual": _esc_recall, "pass": _esc_recall == 1.0},
                },
                # Per-route breakdown
                "by_route": {
                    route: {
                        "count":       sum(1 for r in _scored if r.actual_route == route),
                        "faithfulness": _route_avg(route, "llm_judge_faithfulness"),
                        "relevance":    _route_avg(route, "llm_judge_relevance"),
                    }
                    for route in ("RAG", "SQL", "Hybrid", "Multi-Agent")
                },
                # Distribution
                "faith_ge_095": sum(1 for r in _scored if r.llm_judge_faithfulness >= 0.95),
                "faith_lt_070": sum(1 for r in _scored if r.llm_judge_faithfulness < 0.70),
            },
            "cases": [
                {
                    "tc_id":               r.tc_id,
                    "route":               r.actual_route,
                    "escalated":           r.actual_escalation,
                    "faithfulness":        r.llm_judge_faithfulness,
                    "relevance":           r.llm_judge_relevance,
                    "rubric_coverage":     r.heuristic_rubric_coverage,
                    "citation_present":    r.heuristic_citation_present,
                    "error":               r.error,
                }
                for r in pipeline_results
            ],
        }
        out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
        with open(out_path, "w", encoding="utf-8") as _f:
            _json.dump(out, _f, indent=2, default=str)
        print(f"\nResults saved to eval/eval_results.json")

    elif "--guardrails" in sys.argv:
        # python -m eval.eval --guardrails
        run_guardrail_eval()

    else:
        # python -m eval.eval   →  Agent 1 classification + guardrail check
        from src.guardrails.guardrails import check_guardrails
        golden_set = load_golden_set()
        classification_results = [run_classification_check(tc, agent1_classify) for tc in golden_set]
        guardrail_results = check_guardrail_blocks(check_guardrails, golden_set)
        guardrail_ids = {r.tc_id for r in guardrail_results}
        all_results = [r for r in classification_results if r.tc_id not in guardrail_ids] + guardrail_results
        report = run_evaluation(agent1_classify)
        print_report(report, all_results)
