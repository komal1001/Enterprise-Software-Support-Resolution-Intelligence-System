import os
import time

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langfuse import observe
from pydantic import BaseModel

from src.graph.state import TicketState
from src.observability.langfuse_client import get_cached_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# Output schema
#
# sources is required — every response must cite which documents were used.
# SLO #8: Source Attribution Rate = 100%
# SLO #2: Answer Groundedness ≥ 95% (enforced by prompt, measured by eval)
# ---------------------------------------------------------------------------

class SynthesizedResponse(BaseModel):
    response: str         # full customer-facing answer with inline citations
    sources:  list[str]   # document names cited (e.g. ["API_Error_Codes_Troubleshooting_Handbook"])


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
).with_structured_output(SynthesizedResponse, include_raw=True)

# Plain-text streaming LLM — used by stream_synthesizer_tokens() for SSE token streaming.
# No structured output so tokens arrive as readable words, not JSON fragments.
_llm_stream = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    streaming=True,
    model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt_variables(state: TicketState) -> dict:
    classification  = state["classification"]
    severity_assess = state.get("severity_assessment") or {}

    rag_chunks = (state.get("rag_result") or {}).get("chunks", [])
    sql_rows   = (state.get("sql_result") or {}).get("rows", [])

    rag_text = (
        "\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{str(c.get('text', ''))[:1000]}"
            for c in rag_chunks[:5]
        )
        if rag_chunks else ""
    )

    sql_text = str(sql_rows[:10]) if sql_rows else ""

    context_block = ""
    if rag_text:
        context_block += f"DOCUMENTATION SOURCES:\n{rag_text}\n\n"
    if sql_text:
        total_rows = len(sql_rows)
        truncation_note = f" (showing first 10 of {total_rows})" if total_rows > 10 else ""
        context_block += f"ACCOUNT & TICKET DATA{truncation_note}:\n{sql_text}\n\n"
    if not context_block:
        context_block = "No additional context retrieved.\n\n"

    severity = severity_assess.get("severity") or classification.get("severity", "Medium")

    tone_instruction = {
        "Critical": "Respond with urgency. Acknowledge the severity immediately.",
        "High":     "Respond promptly and thoroughly.",
        "Medium":   "Respond clearly with step-by-step guidance.",
        "Low":      "Respond helpfully with clear instructions.",
    }.get(severity, "Respond clearly and helpfully.")

    return {
        "tone_instruction": tone_instruction,
        "ticket_text":      state["ticket_text"],
        "category":         classification.get("category"),
        "severity":         severity,
        "context_block":    context_block,
    }


# ---------------------------------------------------------------------------
# Response Synthesizer — LangGraph node
#
# Runs when Agent 4 sets escalate=False (non-escalated tickets only).
# Agent 5 handles escalated tickets and sets final_response directly.
# ---------------------------------------------------------------------------

@observe(name="response-synthesizer", as_type="generation", capture_input=False, capture_output=False)
def response_synthesizer(state: TicketState) -> TicketState:
    from src.observability.langfuse_client import get_langfuse, calculate_cost

    lf = get_langfuse()

    # If SQL was the primary retrieval and returned no rows, the customer/entity
    # doesn't exist in the database — return a clear message instead of letting
    # the LLM fabricate generic instructions.
    sql_result = state.get("sql_result") or {}
    if (
        sql_result
        and sql_result.get("error") is None
        and sql_result.get("row_count", -1) == 0
        and not (state.get("rag_result") or {}).get("chunks")
    ):
        state["final_response"] = (
            "No matching records were found in the database for your query. "
            "Please verify the company name or account details and try again."
        )
        state["_synthesizer_tokens"] = {"input_tokens": 0, "output_tokens": 0}
        return state

    prompt_obj      = get_cached_prompt("response-synthesizer")
    prompt_vars     = _build_prompt_variables(state)
    compiled_prompt = prompt_obj.compile(**prompt_vars)

    lf.update_current_generation(
        model="gpt-4o-mini",
        input={"role": "synthesizer"},
        prompt=prompt_obj,
    )

    for attempt in range(3):
        try:
            raw = _llm.invoke(compiled_prompt)

            if raw.get("parsing_error") or raw["parsed"] is None:
                raise ValueError(f"Parsing failed: {raw.get('parsing_error')}")

            result: SynthesizedResponse = raw["parsed"]
            usage         = raw["raw"].usage_metadata or {}
            input_tokens  = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            state["final_response"] = result.response
            if not state.get("rag_result"):
                state["rag_result"] = {}
            state["rag_result"]["cited_sources"] = result.sources
            state["_synthesizer_tokens"] = {"input_tokens": input_tokens, "output_tokens": output_tokens}

            lf.update_current_generation(
                output={"response": result.response, "sources": result.sources},
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": round(calculate_cost(input_tokens, output_tokens), 6)},
                metadata={"source_count": len(result.sources)},
            )

            return state

        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    lf.update_current_generation(
        output={"error": "all retries failed"},
        level="ERROR",
        status_message="all retries failed",
    )
    state["final_response"] = (
        "We were unable to generate an automated response at this time. "
        "Your ticket has been logged and our support team will follow up shortly."
    )
    return state


# ---------------------------------------------------------------------------
# Token streaming — used by the SSE /ticket/stream endpoint
#
# Yields plain-text tokens as the LLM generates them so the frontend can
# show words appearing word-by-word. Bypasses structured output (which would
# stream JSON fragments) and calls the model with a direct chat prompt.
# ---------------------------------------------------------------------------

def stream_synthesizer_tokens(state: TicketState):
    """Generator that yields text tokens one by one. Caller runs this in a thread."""
    sql_result = state.get("sql_result") or {}
    if (
        sql_result
        and sql_result.get("error") is None
        and sql_result.get("row_count", -1) == 0
        and not (state.get("rag_result") or {}).get("chunks")
    ):
        yield (
            "No matching records were found in the database for your query. "
            "Please verify the company name or account details and try again."
        )
        return

    prompt_vars = _build_prompt_variables(state)

    messages = [
        SystemMessage(content=(
            f"{prompt_vars['tone_instruction']} "
            "You are a technical support specialist for enterprise SaaS software. "
            "Write a clear, professional response that directly addresses the customer's issue. "
            "Cite knowledge base sources inline using [Source Name] format. "
            "Do NOT use email greetings (Hello, Dear) or sign-offs. Start directly with the answer. "
            "Write plain prose — no JSON, no markdown code blocks."
        )),
        HumanMessage(content=(
            f"Customer ticket: {prompt_vars['ticket_text']}\n\n"
            f"Category: {prompt_vars['category']} | Severity: {prompt_vars['severity']}\n\n"
            f"{prompt_vars['context_block']}"
            "Write a helpful support response:"
        )),
    ]

    for chunk in _llm_stream.stream(messages):
        if chunk.content:
            yield chunk.content
