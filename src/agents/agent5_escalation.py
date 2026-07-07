import os
import time
import uuid
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langfuse import observe
from pydantic import BaseModel

from src.graph.state import TicketState
from src.observability.langfuse_client import get_cached_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Incident log lookup — direct parameterised query, no LLM involved.
# Fetches the 5 most recent unresolved/critical incidents for a customer.
# Called only for escalated tickets where customer_id is known from sql_result.
# ---------------------------------------------------------------------------

def _fetch_incident_logs(customer_id: int) -> list[dict]:
    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ.get("POSTGRES_READER_USER", os.environ["POSTGRES_USER"]),
            password=os.environ.get("POSTGRES_READER_PASSWORD", os.environ["POSTGRES_PASSWORD"]),
        )
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET statement_timeout = '5s'")
                cur.execute(
                    """
                    SELECT incident_id, incident_type, severity_level, status,
                           root_cause, resolution_notes, created_at
                    FROM   incident_logs
                    WHERE  customer_id = %s
                    ORDER  BY created_at DESC
                    LIMIT  5
                    """,
                    (customer_id,),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Team routing — maps ticket category to the correct human team
# Based on standard enterprise SaaS escalation structure
# ---------------------------------------------------------------------------

_TEAM_ROUTING = {
    "production_incident": "SRE Team",
    "security":            "Security Team",
    "integration_api":     "Backend Engineering",
    "performance_latency": "Platform Engineering",
    "billing":             "Account Management",
    "usage_configuration": "L2 Support",
    "ambiguous":           "L2 Support",
}

_SEVERITY_TO_PRIORITY = {
    "Critical": "P1",
    "High":     "P2",
    "Medium":   "P3",
    "Low":      "P4",
}

_AUTH_KEYWORDS     = ("oauth", "token", "403", "authentication failure", "authenticate", "identity")
_DATALOSS_KEYWORDS = ("data loss", "records", "disappeared", "deleted", "missing data", "cannot recover")
_BILLING_KEYWORDS  = ("subscription", "downgrade", "upgrade", "rate limit", "billing", "invoice", "charge")
_CONFLICT_KEYWORDS = ("conflicting", "documentation", "installation guide", "api integration guide")

_CATEGORY_ACK = {
    "production_incident": "We understand your service is experiencing a critical production incident",
    "security":            "We understand you've reported a security vulnerability or data exposure concern",
    "integration_api":     "We understand you're experiencing API or integration failures",
    "performance_latency": "We understand you're experiencing severe performance or latency issues",
    "billing":             "We understand you have an urgent billing or account concern",
    "usage_configuration": "We understand you need assistance with configuration",
    "ambiguous":           "We've received your urgent request and need to investigate further",
}


def _get_team(category: str, ticket_text: str) -> str:
    """
    Refines category-level routing with keyword matching on ticket text.
    Returns the most specific human team for the escalation.
    """
    text = ticket_text.lower()

    if category in ("integration_api", "production_incident"):
        if any(kw in text for kw in _AUTH_KEYWORDS):
            return "Identity & Access Team"

    if category == "production_incident":
        if any(kw in text for kw in _DATALOSS_KEYWORDS):
            return "Data Engineering Team"

    if category in ("integration_api", "billing"):
        if any(kw in text for kw in _BILLING_KEYWORDS):
            return "Account Management"

    if category == "usage_configuration":
        if any(kw in text for kw in _CONFLICT_KEYWORDS):
            return "Platform Engineering"

    return _TEAM_ROUTING.get(category, "L2 Support")


# ---------------------------------------------------------------------------
# Output schema — LLM generates the human-readable context summary
# ---------------------------------------------------------------------------

class EscalationSummary(BaseModel):
    context_summary:     str         # 2-3 paragraph briefing for the human engineer
    recommended_actions: list[str]   # bullet-point next steps


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
).with_structured_output(EscalationSummary, include_raw=True)


# ---------------------------------------------------------------------------
# Fallback — minimal package so the ticket always reaches a human
# ---------------------------------------------------------------------------

def _fallback_summary(ticket_text: str) -> EscalationSummary:
    return EscalationSummary(
        context_summary=f"Automated summary unavailable. Original ticket: {ticket_text[:500]}",
        recommended_actions=["Review original ticket", "Contact customer directly"],
    )


# ---------------------------------------------------------------------------
# Prompt inputs — compiled into agent5-escalation Langfuse prompt
# ---------------------------------------------------------------------------

def _get_prompt_inputs(state: TicketState, team: str, priority: str) -> dict:
    classification  = state["classification"]
    severity_assess = state.get("severity_assessment") or {}
    rag_chunks      = (state.get("rag_result") or {}).get("chunks", [])
    sql_rows        = (state.get("sql_result") or {}).get("rows", [])

    rag_text = (
        "\n".join(
            f"[{c.get('source', 'unknown')}]: {str(c.get('text', ''))[:300]}"
            for c in rag_chunks[:3]
        )
        if rag_chunks else "No documentation retrieved."
    )

    return {
        "ticket_text":        state["ticket_text"],
        "team":               team,
        "priority":           priority,
        "category":           classification.get("category"),
        "initial_severity":   classification.get("severity"),
        "reassessed_severity": severity_assess.get("severity"),
        "confidence":         classification.get("confidence"),
        "escalation_trigger": severity_assess.get("escalation_trigger", "n/a"),
        "rag_text":           rag_text,
        "sql_text":           str(sql_rows[:5]) if sql_rows else "No account data retrieved.",
    }


# ---------------------------------------------------------------------------
# Agent 5 — LangGraph node
# ---------------------------------------------------------------------------

@observe(name="agent5-escalation", as_type="generation", capture_input=False, capture_output=False)
def agent5_escalation(state: TicketState) -> TicketState:
    from src.observability.langfuse_client import get_langfuse, calculate_cost

    lf             = get_langfuse()
    classification = state["classification"]
    severity_assess = state.get("severity_assessment") or {}

    category = classification.get("category", "ambiguous")
    severity = severity_assess.get("severity") or classification.get("severity", "High")
    team     = _get_team(category, state["ticket_text"])
    priority = _SEVERITY_TO_PRIORITY.get(severity, "P2")
    ref_id   = f"ESC-{uuid.uuid4().hex[:8].upper()}"

    summary       = None
    input_tokens  = 0
    output_tokens = 0
    prompt_obj    = None
    last_error    = None

    for attempt in range(3):
        try:
            prompt_obj    = get_cached_prompt("agent5-escalation")
            prompt_inputs = _get_prompt_inputs(state, team, priority)
            prompt        = prompt_obj.compile(**prompt_inputs)

            raw = _llm.invoke(prompt)

            if raw.get("parsing_error") or raw["parsed"] is None:
                raise ValueError(f"Parsing failed: {raw.get('parsing_error')}")

            summary_obj: EscalationSummary = raw["parsed"]
            usage = raw["raw"].usage_metadata or {}
            input_tokens  = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            summary = summary_obj
            break

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)

    if summary is None:
        lf.update_current_generation(
            output={"error": str(last_error)},
            level="ERROR",
            status_message=str(last_error),
        )
        summary = _fallback_summary(state["ticket_text"])

    # ── Full context transfer (README requirement) ────────────────────────────
    sql_rows  = (state.get("sql_result") or {}).get("rows", [])
    rag_chunks = (state.get("rag_result") or {}).get("chunks", [])

    # Incident logs — direct parameterised lookup if customer_id is known
    customer_id   = (sql_rows[0].get("customer_id") if sql_rows else None)
    incident_logs = _fetch_incident_logs(customer_id) if customer_id else []

    escalation_package = {
        # Routing & identifiers
        "reference_id":        ref_id,
        "team":                team,
        "priority":            priority,
        "category":            category,
        "severity":            severity,
        "escalation_trigger":  severity_assess.get("escalation_trigger"),

        # LLM-generated briefing for the human engineer
        "context_summary":     summary.context_summary,
        "recommended_actions": summary.recommended_actions,

        # Full context transfer — what the human team needs
        "ticket_history":       state.get("conversation_history", []),
        "account_data":         sql_rows,
        "incident_logs":        incident_logs,
        "documentation_context": [
            {"source": c.get("source", ""), "text": str(c.get("text", ""))[:300]}
            for c in rag_chunks[:5]
        ],
        "agent_reasoning": {
            "classification_reasoning": (state.get("classification") or {}).get("reasoning", ""),
            "severity_reasoning":       severity_assess.get("reasoning", ""),
        },

        "_input_tokens":  input_tokens,
        "_output_tokens": output_tokens,
    }

    # Create Jira issue — non-blocking, failure does not abort the escalation
    jira_key = None
    try:
        from src.integrations.jira_client import create_jira_issue
        jira_key = create_jira_issue(escalation_package)
        escalation_package["jira_issue_key"] = jira_key
    except Exception as jira_err:
        escalation_package["jira_issue_key"] = None
        escalation_package["jira_error"]     = str(jira_err)

    lf.update_current_generation(
        model="gpt-4o-mini",
        input={"role": "escalation-manager"},
        output={"team": team, "priority": priority, "reference_id": ref_id, "jira_key": jira_key},
        usage_details={"input": input_tokens, "output": output_tokens},
        cost_details={"total": round(calculate_cost(input_tokens, output_tokens), 6)},
        prompt=prompt_obj,
        metadata={"escalation_team": team, "priority": priority, "jira_key": jira_key},
    )

    state["escalation_package"] = escalation_package

    ack = _CATEGORY_ACK.get(category, "We've received your urgent request")
    jira_suffix = f" Jira: {jira_key}." if jira_key else ""
    state["final_response"] = (
        f"{ack}. Your ticket has been escalated to our {team} ({priority}). "
        f"Reference: {ref_id}.{jira_suffix} "
        "A support engineer will contact you shortly with a full resolution plan."
    )

    return state
