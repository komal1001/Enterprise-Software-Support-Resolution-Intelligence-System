"""
Langfuse v4 observability client

One root span per ticket (the trace in Langfuse).
Each agent uses @observe() to create a child generation/retriever span automatically.

The root span is opened with start_as_current_observation().__enter__() so that OTel
sets it as the "current" span — @observe() on agent functions then finds it as parent
and nests correctly in the Langfuse Timeline waterfall.

Usage:
    from src.observability.langfuse_client import observe, get_langfuse, start_trace, log_ticket_complete, timer

    root, lf_token = start_trace(ticket_text)
    # ... run agents decorated with @observe() ...
    log_ticket_complete(root, lf_token, final_response=..., ...)
"""

import os
import threading
import time
from contextvars import ContextVar
from typing import Optional

from dotenv import load_dotenv
from langfuse import Langfuse, observe  # noqa: F401 — re-exported for agents

load_dotenv()

# GPT-4o mini pricing (Azure, per token)
_INPUT_COST_PER_TOKEN  = 0.15 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

_client: Optional[Langfuse] = None

# _current_root — root span object, for backward-compat (score_trace, etc.)
# _ctx_mgr      — the context manager entered in start_trace, exited in log_ticket_complete
_current_root: ContextVar = ContextVar("langfuse_root",    default=None)
_ctx_mgr:      ContextVar = ContextVar("langfuse_ctx_mgr", default=None)


def get_langfuse() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _client


# ---------------------------------------------------------------------------
# Prompt cache — shared across all agents
#
# lf.get_prompt() makes an HTTP round-trip to Langfuse cloud on every call.
# Prompts are version-pinned in the dashboard and don't change during a run,
# so we fetch each one once and reuse it for the lifetime of the process.
# Trade-off: edits made in the Langfuse dashboard won't take effect until
# the backend is restarted.
# ---------------------------------------------------------------------------

_prompt_cache: dict = {}
_prompt_cache_lock  = threading.Lock()


def get_cached_prompt(name: str):
    """Return the named Langfuse prompt, fetching from cloud only on first call."""
    if name not in _prompt_cache:
        with _prompt_cache_lock:
            if name not in _prompt_cache:
                _prompt_cache[name] = get_langfuse().get_prompt(name)
    return _prompt_cache[name]


def get_current_root():
    """Return the root span for the current request context, or None."""
    return _current_root.get()


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * _INPUT_COST_PER_TOKEN) + (output_tokens * _OUTPUT_COST_PER_TOKEN)


def timer() -> float:
    """Return current time in ms — subtract two calls to get latency."""
    return time.time() * 1000


def start_trace(ticket_text: str):
    """
    Open the root trace span and make it the CURRENT OTel span.

    Using start_as_current_observation().__enter__() instead of start_observation()
    so that @observe()-decorated agent functions find this root as their parent and
    nest correctly in the Langfuse Timeline waterfall.

    Returns (root, token) — pass both to log_ticket_complete for cleanup.
    """
    ctx_mgr = get_langfuse().start_as_current_observation(
        name="ticket-resolution",
        as_type="span",
        input={"ticket_text": ticket_text},
    )
    root  = ctx_mgr.__enter__()   # sets root as current OTel span
    token = _current_root.set(root)
    _ctx_mgr.set(ctx_mgr)
    return root, token


def log_ticket_complete(
    root,
    token,
    final_response: str,
    total_cost_usd: float,
    total_latency_ms: float,
    tsr_pass: bool,
    routing_path: str = "",
) -> None:
    """
    Attach SLO scores to the root span, end it, and flush to Langfuse.
    __exit__() on the context manager ends the root span (not root.end() directly,
    since start_as_current_observation owns the span lifecycle).
    """
    slo_latency_pass = (
        total_latency_ms <= 5000 if routing_path != "Multi-Agent"
        else total_latency_ms <= 10000
    )

    root.update(
        output={"response": final_response},
        metadata={
            "total_latency_ms": round(total_latency_ms),
            "total_cost_usd":   round(total_cost_usd, 6),
            "routing_path":     routing_path,
        },
    )

    root.score_trace(name="slo_tsr",       value=1.0 if tsr_pass else 0.0,
                     comment="SLO #1 — Task Success Rate")
    root.score_trace(name="slo_latency",   value=1.0 if slo_latency_pass else 0.0,
                     comment=f"SLO #5/#6 — threshold {'10s' if routing_path == 'Multi-Agent' else '5s'}")
    root.score_trace(name="slo_cost_usd",  value=round(total_cost_usd, 6),
                     comment="SLO #14 — cost per ticket (warn >$0.05, cap $0.15)")
    root.score_trace(name="slo_cost_pass", value=1.0 if total_cost_usd <= 0.15 else 0.0,
                     comment="SLO #14 — hard cap $0.15")

    ctx_mgr = _ctx_mgr.get()
    if ctx_mgr is not None:
        ctx_mgr.__exit__(None, None, None)   # ends the root span cleanly

    _current_root.reset(token)
    get_langfuse().flush()


def log_guardrail_block(ticket_text: str, reason: str) -> None:
    """Log a guardrail block as a standalone trace (no LLM cost, no child spans)."""
    root = get_langfuse().start_observation(
        name="guardrail-blocked",
        as_type="span",
        input={"ticket_text": ticket_text[:200]},
        output={"blocked": True, "reason": reason},
        metadata={"block_reason": reason},
    )
    root.score_trace(name="slo_guardrail", value=1.0, comment="SLO #13 — guardrail blocked adversarial input")
    root.end()
    get_langfuse().flush()
