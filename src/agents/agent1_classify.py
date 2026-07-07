import os
import time
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langfuse import observe

from src.graph.state import TicketState
from src.observability.langfuse_client import get_cached_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Output schema — what Agent 1 must return
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    category: Literal[
        "usage_configuration",
        "integration_api",
        "performance_latency",
        "production_incident",
        "billing",
        "security",
        "ambiguous"
    ]
    severity: Literal["Low", "Medium", "High", "Critical"]
    routing_path: Literal["RAG", "SQL", "Hybrid", "Multi-Agent"]
    confidence: float
    reasoning: str


# ---------------------------------------------------------------------------
# LLM singleton — created once at import time, reused on every ticket
# include_raw=True preserves the AIMessage so usage_metadata (token counts)
# is accessible for Langfuse cost tracking (SLO #14)
# ---------------------------------------------------------------------------

_structured_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
).with_structured_output(ClassificationResult, include_raw=True)


# ---------------------------------------------------------------------------
# Fallback — used when all 3 retry attempts fail
# ambiguous + Multi-Agent + confidence=0.0 guarantees Agent 5 escalates
# so the ticket always reaches a human (SLO #11)
# ---------------------------------------------------------------------------

_FALLBACK_CLASSIFICATION = {
    "category":       "ambiguous",
    "severity":       "Medium",
    "routing_path":   "Multi-Agent",
    "confidence":     0.0,
    "reasoning":      "Classification unavailable — routed for human review",
    "_input_tokens":  0,
    "_output_tokens": 0,
}


# ---------------------------------------------------------------------------
# Agent 1 — LangGraph node
# ---------------------------------------------------------------------------

@observe(name="agent1-classify", as_type="generation", capture_input=False, capture_output=False)
def agent1_classify(state: TicketState) -> TicketState:
    from src.observability.langfuse_client import get_langfuse, calculate_cost

    lf         = get_langfuse()
    last_error = None

    for attempt in range(3):
        try:
            prompt_obj = get_cached_prompt("agent1-classify")
            prompt     = prompt_obj.compile(ticket_text=state["ticket_text"])

            raw = _structured_llm.invoke(prompt)

            if raw.get("parsing_error") or raw["parsed"] is None:
                raise ValueError(f"Parsing failed: {raw.get('parsing_error')}")

            result: ClassificationResult = raw["parsed"]
            usage         = raw["raw"].usage_metadata or {}
            input_tokens  = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            classification = result.model_dump()
            classification["_input_tokens"]  = input_tokens
            classification["_output_tokens"] = output_tokens

            lf.update_current_generation(
                model="gpt-4o-mini",
                input={"role": "classifier"},
                output=classification,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": round(calculate_cost(input_tokens, output_tokens), 6)},
                prompt=prompt_obj,
                metadata={
                    "category":     classification.get("category"),
                    "severity":     classification.get("severity"),
                    "routing_path": classification.get("routing_path"),
                    "confidence":   classification.get("confidence"),
                },
            )

            state["classification"] = classification
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
    state["classification"] = _FALLBACK_CLASSIFICATION.copy()
    return state


# ---------------------------------------------------------------------------
# Quick test — python -m src.agents.agent1_classify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_tickets = [
        "My Python SDK stopped sending traces to Langfuse after I upgraded to v2.0.",
        "All our traces stopped appearing 2 hours ago — our AI system is completely blind in production.",
        "How do I set up tracing in LangGraph using the Langfuse Python SDK?",
        "Our API is failing and we are also being charged twice this month.",
    ]

    for ticket in test_tickets:
        state: TicketState = {
            "ticket_text": ticket,
            "conversation_history": [],
            "classification": None,
            "rag_result": None,
            "sql_result": None,
            "severity_assessment": None,
            "escalation_package": None,
            "final_response": None,
        }
        result = agent1_classify(state)
        c = result["classification"]
        print(f"\nTicket    : {ticket[:65]}...")
        print(f"Category  : {c['category']}")
        print(f"Severity  : {c['severity']}")
        print(f"Route     : {c['routing_path']}")
        print(f"Confidence: {c['confidence']}")
        print(f"Reasoning : {c['reasoning']}")
