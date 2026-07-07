import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langfuse import observe
from pydantic import BaseModel, Field

from src.graph.state import TicketState
from src.observability.langfuse_client import get_cached_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Confidence thresholds — per category (ADR-005)
#
# If Agent 1's confidence is below the threshold for that category,
# Agent 4 escalates regardless of severity level.
#
# Industry basis:
#   - production_incident / security: ServiceNow Now Intelligence (0.85)
#   - integration_api / performance:  Zendesk AI auto-respond threshold (0.70)
#   - usage_configuration / billing:  lower stakes, 0.65 sufficient
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS = {
    "production_incident": 0.85,   # SLO #10 — misclassification < 3%
    "security":            0.85,   # SLO #11 — escalation recall = 100%
    "integration_api":     0.70,
    "performance_latency": 0.70,
    "usage_configuration": 0.65,
    "billing":             0.65,
    "ambiguous":           0.00,   # always escalate
}

# Categories where High severity also triggers escalation (not just Critical)
_HIGH_RISK_CATEGORIES = frozenset({"production_incident", "security"})

# Keywords that signal the ticket likely needs SQL data (account/subscription context)
_SQL_SIGNALS = frozenset({
    "account", "customer", "subscription", "billing", "plan",
    "invoice", "tier", "sla", "payment", "renewal", "suspended",
})

# Keywords that signal the ticket likely needs RAG data (documentation/how-to context)
_RAG_SIGNALS = frozenset({
    "how", "setup", "configure", "install", "error", "api",
    "documentation", "guide", "steps", "integration", "sdk",
})


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class SeverityAssessment(BaseModel):
    severity:           Literal["Low", "Medium", "High", "Critical"]
    confidence:         float = Field(ge=0.0, le=1.0)
    escalate:           bool
    reasoning:          str
    escalation_trigger: Optional[str] = None   # which hard rule fired, if any


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
).with_structured_output(SeverityAssessment, include_raw=True)


# ---------------------------------------------------------------------------
# Fallback — used when all 3 LLM attempts fail.
# Escalate for safety rather than risk missing a Critical ticket (SLO #11).
# ---------------------------------------------------------------------------

_FALLBACK_ASSESSMENT = {
    "severity":           "High",
    "confidence":         0.0,
    "escalate":           True,
    "reasoning":          "Severity assessment unavailable — escalating for safety",
    "escalation_trigger": "assessment_failure",
    "_input_tokens":      0,
    "_output_tokens":     0,
}


# ---------------------------------------------------------------------------
# Phase 1 — Hard escalation rules (deterministic, no LLM)
#
# These rules guarantee SLO #11 (escalation recall = 100%).
# They cover the 8 mandatory high-risk scenarios from the README:
#   1. Production outage (Critical/High + production_incident)
#   2. Security vulnerability (Critical/High + security)
#   3. Any Critical severity ticket
#   4. Low-confidence classification on high-stakes categories
#   5. Ambiguous tickets (cannot risk missing a hidden critical signal)
#
# LLM cannot override these — hard_escalate=True forces escalate=True
# on the final assessment regardless of what the LLM returns.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reflection — detect when Agent 1 routed incorrectly and retrieval is missing
#
# Only triggers when:
#   1. routing_path was single-source (RAG or SQL) but the ticket has signals
#      suggesting the OTHER source would also be useful
#   2. That source was never retrieved (result is None, not just empty)
#   3. reflection_count == 0 — max 1 reflection per ticket (prevents infinite loop)
#
# Returns "sql" or "rag" to tell the graph which agent to re-run,
# or None if no reflection needed.
# ---------------------------------------------------------------------------

def _detect_missing_retrieval(state: TicketState) -> str | None:
    if state.get("reflection_count", 0) >= 1:
        return None

    routing_path  = state["classification"].get("routing_path", "RAG")
    ticket_lower  = state["ticket_text"].lower()

    if routing_path == "RAG" and state.get("sql_result") is None:
        if any(s in ticket_lower for s in _SQL_SIGNALS):
            return "sql"

    if routing_path == "SQL" and state.get("rag_result") is None:
        if any(s in ticket_lower for s in _RAG_SIGNALS):
            return "rag"

    if routing_path in ("SQL", "Hybrid", "Multi-Agent"):
        sql_error = (state.get("sql_result") or {}).get("error")
        if sql_error:
            return "sql"

    return None


def is_below_confidence_threshold(category: str, confidence: float) -> bool:
    threshold = CONFIDENCE_THRESHOLDS.get(category, 0.70)
    return confidence < threshold


def _check_hard_escalation(classification: dict) -> tuple[bool, str]:
    category   = classification.get("category", "ambiguous")
    severity   = classification.get("severity", "Low")
    confidence = classification.get("confidence", 0.0)

    if category == "ambiguous":
        return True, "ambiguous_category"

    if is_below_confidence_threshold(category, confidence):
        return True, f"low_confidence_{round(confidence, 2)}"

    if severity == "Critical":
        return True, "critical_severity"

    if severity == "High" and category in _HIGH_RISK_CATEGORIES:
        return True, f"high_severity_{category}"

    return False, ""


# ---------------------------------------------------------------------------
# Phase 2 — LLM re-evaluation prompt (fetched from Langfuse prompt management)
# ---------------------------------------------------------------------------

def _get_prompt_inputs(state: TicketState) -> dict:
    """Extract variables needed to compile the agent4-severity Langfuse prompt."""
    classification = state["classification"]

    rag_chunks = (state.get("rag_result") or {}).get("chunks", [])
    sql_rows   = (state.get("sql_result") or {}).get("rows", [])

    rag_text = (
        "\n".join(
            f"[{c.get('source', 'unknown')}]: {str(c.get('text', ''))[:300]}"
            for c in rag_chunks[:3]
        )
        if rag_chunks else "No documentation retrieved."
    )

    return {
        "ticket_text": state["ticket_text"],
        "category":    classification.get("category"),
        "severity":    classification.get("severity"),
        "confidence":  classification.get("confidence"),
        "reasoning":   classification.get("reasoning"),
        "rag_text":    rag_text,
        "sql_text":    str(sql_rows[:5]) if sql_rows else "No account data retrieved.",
    }


# ---------------------------------------------------------------------------
# Agent 4 — LangGraph node (ADR-005: runs on EVERY ticket)
# ---------------------------------------------------------------------------

@observe(name="agent4-severity", as_type="generation", capture_input=False, capture_output=False)
def agent4_severity(state: TicketState) -> TicketState:
    from src.observability.langfuse_client import get_langfuse, calculate_cost

    lf             = get_langfuse()
    classification = state["classification"]

    # Phase 1 — hard rules (deterministic, SLO #11 guarantee)
    hard_escalate, trigger = _check_hard_escalation(classification)

    # Phase 2 — LLM re-evaluation with retry
    last_error = None

    for attempt in range(3):
        try:
            prompt_obj    = get_cached_prompt("agent4-severity")
            prompt_inputs = _get_prompt_inputs(state)
            prompt        = prompt_obj.compile(**prompt_inputs)

            raw = _llm.invoke(prompt)

            if raw.get("parsing_error") or raw["parsed"] is None:
                raise ValueError(f"Parsing failed: {raw.get('parsing_error')}")

            result: SeverityAssessment = raw["parsed"]
            usage         = raw["raw"].usage_metadata or {}
            input_tokens  = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            assessment = result.model_dump()

            # Hard rule overrides LLM escalation decision — SLO #11
            if hard_escalate:
                assessment["escalate"]           = True
                assessment["escalation_trigger"] = trigger
            elif assessment["escalate"]:
                # LLM wants to escalate but no hard rule fired.
                # Only allow if LLM re-assessed to Critical, or High on a high-risk
                # category. Prevents false positives where the LLM misreads ticket
                # content (e.g. "show me escalation history" → escalate=True).
                llm_sev = result.severity
                llm_cat = classification.get("category", "ambiguous")
                if llm_sev != "Critical" and not (llm_sev == "High" and llm_cat in _HIGH_RISK_CATEGORIES):
                    assessment["escalate"]           = False
                    assessment["escalation_trigger"] = None

            assessment["_input_tokens"]  = input_tokens
            assessment["_output_tokens"] = output_tokens

            lf.update_current_generation(
                model="gpt-4o-mini",
                input={"role": "severity-assessor"},
                output=assessment,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": round(calculate_cost(input_tokens, output_tokens), 6)},
                prompt=prompt_obj,
                metadata={
                    "severity":   assessment.get("severity"),
                    "confidence": assessment.get("confidence"),
                    "escalate":   assessment.get("escalate"),
                },
            )

            # Reflection — check if retrieval is missing due to Agent 1 routing error
            reretrieval = _detect_missing_retrieval(state)
            assessment["needs_reretrieval"] = reretrieval
            if reretrieval:
                state["reflection_count"] = state.get("reflection_count", 0) + 1

            state["severity_assessment"] = assessment
            return state

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)

    lf.update_current_generation(
        output={"error": str(last_error)},
        level="ERROR",
        status_message=str(last_error),
    )
    state["severity_assessment"] = _FALLBACK_ASSESSMENT.copy()
    return state
