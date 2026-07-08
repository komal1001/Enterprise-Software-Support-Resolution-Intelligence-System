import asyncio
import contextvars
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

_log = logging.getLogger(__name__)

import tempfile
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.api.limiter import limiter
from src.api.auth import get_current_user, require_role

from src.graph.state import TicketState
from src.graph.graph import compiled_graph, pre_synth_compiled_graph
from src.agents.agent5_escalation import agent5_escalation
from src.agents.response_synthesizer import stream_synthesizer_tokens
from src.guardrails.guardrails import check_guardrails
from src.guardrails.output_guardrails import check_output
from src.observability.langfuse_client import start_trace, log_ticket_complete, calculate_cost, timer, log_guardrail_block

router = APIRouter()


def _json_serial(obj):
    """Fallback JSON serializer for types psycopg2 returns (datetime, date, Decimal)."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ticket_text: str
    thread_id: str
    conversation_history: list[dict] = []


class ClassificationOut(BaseModel):
    category: str
    severity: str
    routing_path: str
    confidence: float
    reasoning: str


class TicketResponse(BaseModel):
    final_response: str
    classification: ClassificationOut
    conversation_history: list[dict]
    trace_id: str
    submitted_at: str
    total_cost_usd: float   # SLO #14 — cost per ticket (avg <= $0.05, cap $0.15)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Returns the current user's identity and roles — useful for frontend role-based UI."""
    return {
        "sub":   user.get("sub"),
        "roles": user.get("https://support-resolution-api/roles", []),
    }


@router.get("/escalations")
def get_escalations(user: dict = Depends(require_role("manager", "admin"))):
    """Manager/Admin only — all escalated tickets ordered by newest first (SLO #12)."""
    import os, psycopg2, psycopg2.extras
    try:
        conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
        )
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET statement_timeout = '5s'")
                cur.execute("""
                    SELECT t.ticket_id, t.customer_id, c.company_name,
                           t.issue_category, t.severity_level, t.ticket_status,
                           t.assigned_team, t.escalation_flag, t.created_at
                    FROM   support_tickets t
                    LEFT JOIN customers c USING (customer_id)
                    WHERE  t.escalation_flag = true
                    ORDER  BY t.created_at DESC
                    LIMIT  100
                """)
                rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return {"escalations": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


# ---------------------------------------------------------------------------
# POST /ticket
# ---------------------------------------------------------------------------

@router.post("/ticket", response_model=TicketResponse)
@limiter.limit("10/minute")
def submit_ticket(request: Request, body: TicketRequest, user: dict = Depends(require_role("l1-agent", "manager", "admin"))):
    if not body.ticket_text.strip():
        raise HTTPException(status_code=400, detail="ticket_text cannot be empty")

    # Reject oversized inputs before they reach the LLM (SLO #14 cost guard).
    # 800 tokens ≈ 3200 chars. Agent 2 uses classification["reasoning"] for retrieval,
    # not raw ticket text, so downstream agents are unaffected by this limit.
    MAX_TICKET_CHARS = 3200
    if len(body.ticket_text) > MAX_TICKET_CHARS:
        raise HTTPException(
            status_code=400,
            detail="Ticket too long (max ~800 tokens). Please describe the problem in your own words — do not paste raw logs or stack traces."
        )

    # Guardrail check — runs before any LLM call (SLO #13)
    guardrail = check_guardrails(body.ticket_text)
    if guardrail.blocked:
        log_guardrail_block(body.ticket_text, guardrail.reason.value if guardrail.reason else "unknown")
        raise HTTPException(status_code=400, detail=guardrail.message)

    submitted_at = datetime.now(timezone.utc).isoformat()

    # TODO Sprint 9 — Ticket persistence (requires RBAC):
    # Once auth is in place, customer_id is known from the logged-in user.
    # Insert one row into support_tickets with all fields:
    #   customer_id, ticket_text, thread_id, issue_category, severity_level,
    #   routing_path, ticket_status, assigned_team, escalation_flag, trace_id, created_at
    # The schema already has these columns (ALTER TABLE ran in Sprint 7).
    # Return ticket_id in the response so the user has a reference number.
    # See: database/schema.sql — support_tickets table

    state: TicketState = {
        "ticket_text":          body.ticket_text,
        "conversation_history": body.conversation_history,
        "classification":       None,
        "rag_result":           None,
        "sql_result":           None,
        "severity_assessment":  None,
        "escalation_package":   None,
        "final_response":       None,
        "reflection_count":     0,
    }

    t0          = timer()
    root, token = start_trace(body.ticket_text)

    config = {"configurable": {"thread_id": body.thread_id}}
    t1 = timer()
    try:
        state = compiled_graph.invoke(state, config=config)
    except Exception as e:
        root.end()
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")
    pipeline_latency = timer() - t1

    classification    = state["classification"]
    severity_assess   = state.get("severity_assessment") or {}
    escalation_pkg    = state.get("escalation_package") or {}

    # Agents 1, 4, 5 log their own Langfuse generation spans internally (with prompt_obj
    # linked for version tracking). Routes.py only aggregates costs from token counts
    # stored in state by each agent.
    agent1_cost = calculate_cost(
        classification.pop("_input_tokens", 0),
        classification.pop("_output_tokens", 0),
    )

    sql_result  = state.get("sql_result") or {}
    agent3_cost = calculate_cost(
        sql_result.pop("_input_tokens", 0),
        sql_result.pop("_output_tokens", 0),
    )

    agent4_cost = calculate_cost(
        severity_assess.pop("_input_tokens", 0),
        severity_assess.pop("_output_tokens", 0),
    ) if severity_assess else 0.0

    agent5_cost = calculate_cost(
        escalation_pkg.pop("_input_tokens", 0),
        escalation_pkg.pop("_output_tokens", 0),
    ) if escalation_pkg else 0.0

    synth_tokens  = state.get("_synthesizer_tokens") or {}
    synth_cost    = calculate_cost(
        synth_tokens.get("input_tokens", 0),
        synth_tokens.get("output_tokens", 0),
    )

    tsr_pass = not (
        classification["confidence"] == 0.0 and classification["category"] == "ambiguous"
    )

    # Clarification flow — industry standard (Zendesk, Intercom, Salesforce Einstein):
    # Ask one follow-up question before routing to a human.
    # Only triggers on the FIRST message (no prior history) so we don't loop forever.
    # On the user's next reply, conversation_history has context → Agent 1 can classify properly.
    if (
        classification["category"] == "ambiguous"
        and not body.conversation_history
    ):
        final_response = (
            "To help us route your ticket to the right team, could you share a bit more detail?\n\n"
            "- Which product, feature, or integration are you having trouble with?\n"
            "- What error message or unexpected behavior are you seeing?\n"
            "- When did this issue start?\n\n"
            "The more context you provide, the faster we can resolve your issue."
        )
    elif not tsr_pass:
        final_response = (
            "We were unable to automatically classify your issue at this time. "
            "Your ticket has been escalated to our support team who will respond shortly."
        )
    else:
        final_response = state.get("final_response") or (
            "Your ticket has been received and is being processed."
        )

    total_cost    = agent1_cost + agent3_cost + agent4_cost + agent5_cost + synth_cost
    total_latency = timer() - t0

    if total_cost > 0.15:
        import logging
        logging.getLogger(__name__).warning(
            "SLO #14 BREACH — ticket cost $%.4f exceeds hard cap $0.15 (trace: %s)",
            total_cost, root.trace_id,
        )

    log_ticket_complete(
        root,
        token,
        final_response=final_response,
        total_cost_usd=total_cost,
        total_latency_ms=total_latency,
        tsr_pass=tsr_pass,
        routing_path=classification["routing_path"],
    )

    updated_history = body.conversation_history + [
        {"role": "user",      "content": body.ticket_text},
        {"role": "assistant", "content": final_response},
    ]

    return TicketResponse(
        final_response=final_response,
        classification=ClassificationOut(**classification),
        conversation_history=updated_history,
        trace_id=root.trace_id,
        submitted_at=submitted_at,
        total_cost_usd=round(total_cost, 6),
    )


# ---------------------------------------------------------------------------
# POST /admin/ingest — upload a PDF and index it into pgvector + BM25
# Admin-only (SLO #12 — no unauthorised data access).
# ---------------------------------------------------------------------------

@router.post("/admin/ingest")
@limiter.limit("5/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_role("admin")),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")

    def _run_ingest(pdf_bytes: bytes, filename: str) -> dict:
        import pathlib
        from src.retrieval.ingest import ingest_pdf_bytes

        result = ingest_pdf_bytes(pdf_bytes, filename)
        return result

    try:
        result = await asyncio.to_thread(_run_ingest, contents, file.filename)
        return {
            "status":     "ok",
            "filename":   file.filename,
            "chunks":     result.get("chunks", 0),
            "message":    f"Indexed {result.get('chunks', 0)} chunks from {file.filename}",
        }
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Ingestion pipeline not available.")
    except Exception as e:
        _log.exception("Ingest error: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /ticket/stream — SSE streaming endpoint
# Yields a status event after each agent completes, then a done event
# with the full response. User sees live progress instead of blank screen.
# ---------------------------------------------------------------------------

_AGENT_TIMEOUT = 50.0    # seconds per agent step — covers Neon cold start + LLM call

def _user_friendly_error(raw: str) -> str:
    """Map internal/library exception messages to user-readable strings."""
    r = raw.lower()
    if "nested async" in r or "nest_asyncio" in r:
        return "The system is temporarily busy. Please try again in a moment."
    if "connection" in r and ("error" in r or "refused" in r or "closed" in r):
        return "Unable to reach the database. Please try again shortly."
    if "timeout" in r or "timed out" in r:
        return "The request took too long. Please try again."
    if "rate limit" in r or "429" in r:
        return "Too many requests. Please wait a moment before submitting again."
    if "api key" in r or "authentication" in r or "unauthorized" in r:
        return "A configuration error occurred. Please contact your administrator."
    return "Something went wrong processing your ticket. Please try again."

_NODE_LABELS = {
    "agent1_classify":      "Classifying ticket...",
    "agent2_rag":           "Retrieving documentation...",
    "agent3_sql":           "Querying account data...",
    "agent4_severity":      "Assessing severity...",
    "agent5_escalation":    "Escalating to support team...",
    "response_synthesizer": "Generating response...",
}

@router.post("/ticket/stream")
@limiter.limit("10/minute")
async def submit_ticket_stream(
    request: Request,
    body: TicketRequest,
    user: dict = Depends(require_role("l1-agent", "manager", "admin")),
):
    if not body.ticket_text.strip():
        raise HTTPException(status_code=400, detail="ticket_text cannot be empty")

    if len(body.ticket_text) > 3200:
        raise HTTPException(status_code=400, detail="Ticket too long (max ~800 tokens).")

    guardrail = check_guardrails(body.ticket_text)
    if guardrail.blocked:
        log_guardrail_block(body.ticket_text, guardrail.reason.value if guardrail.reason else "unknown")
        raise HTTPException(status_code=400, detail=guardrail.message)

    state: TicketState = {
        "ticket_text":          body.ticket_text,
        "conversation_history": body.conversation_history,
        "classification":       None,
        "rag_result":           None,
        "sql_result":           None,
        "severity_assessment":  None,
        "escalation_package":   None,
        "final_response":       None,
        "reflection_count":     0,
    }
    config = {"configurable": {"thread_id": body.thread_id}}

    async def event_generator():
        loop         = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        submitted_at = datetime.now(timezone.utc).isoformat()
        t0           = timer()
        root, lf_token = start_trace(body.ticket_text)

        # Snapshot async task context NOW (after start_trace set _current_root)
        # so background threads run inside it and agents can call get_current_root().
        _ctx = contextvars.copy_context()

        final_response = ""
        merged_state   = dict(state)
        classification = {}

        try:
            # ── Phase 1: run agents 1–4 via pre_synth_compiled_graph ──────────

            def run_pre_synth():
                # Give this thread its own isolated event loop so LlamaIndex's
                # "Detected nested async" check doesn't fire. LlamaIndex sees
                # the FastAPI event loop via asyncio.get_event_loop() and raises
                # if that loop is_running(). A fresh per-thread loop is NOT
                # running, so LlamaIndex runs normally. Coroutines are still
                # scheduled onto the main `loop` via run_coroutine_threadsafe.
                _thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_thread_loop)
                try:
                    def _inner():
                        try:
                            for chunk in pre_synth_compiled_graph.stream(state, stream_mode="updates"):
                                asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop)
                        except Exception as e:
                            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
                        finally:
                            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                    _ctx.run(_inner)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                finally:
                    _thread_loop.close()

            threading.Thread(target=run_pre_synth, daemon=True).start()

            while True:
                try:
                    msg_type, data = await asyncio.wait_for(queue.get(), timeout=_AGENT_TIMEOUT)
                except asyncio.TimeoutError:
                    yield {"event": "error", "data": json.dumps({"message": "Agent timed out. Please try again."})}
                    return
                if msg_type == "error":
                    _log.error("PIPELINE ERROR: %s", data)
                    yield {"event": "error", "data": json.dumps({"message": _user_friendly_error(data)})}
                    return
                if msg_type == "done":
                    break

                node_name  = list(data.keys())[0]
                node_state = data[node_name]
                merged_state.update(node_state)

                event_data: dict = {"step": node_name, "message": _NODE_LABELS.get(node_name, node_name)}
                if node_name == "agent1_classify":
                    clf = node_state.get("classification") or {}
                    classification = {k: v for k, v in clf.items() if not k.startswith("_")}
                    event_data["classification"] = classification
                elif node_name == "agent3_sql":
                    sql = node_state.get("sql_result") or {}
                    event_data["row_count"] = sql.get("row_count", 0)
                    event_data["sql_error"] = bool(sql.get("error"))
                elif node_name == "agent2_rag":
                    rag = node_state.get("rag_result") or {}
                    event_data["chunk_count"] = len(rag.get("chunks") or [])
                elif node_name == "agent4_severity":
                    sev = node_state.get("severity_assessment") or {}
                    event_data["escalate"]  = sev.get("escalate", False)
                    event_data["severity"]  = sev.get("severity", "")

                yield {"event": "status", "data": json.dumps(event_data)}

            # ── Phase 2: escalation OR token streaming ─────────────────────────
            severity_assessment = merged_state.get("severity_assessment") or {}
            escalate            = severity_assessment.get("escalate", False)

            # Low-confidence or ambiguous tickets without prior context → ask for
            # clarification instead of escalating. Confidence < 0.50 means the
            # classifier couldn't make sense of the input (e.g. "hello", "test").
            _clf_confidence = classification.get("confidence", 1.0)
            _clf_category   = classification.get("category", "")
            if (
                (_clf_category == "ambiguous" or _clf_confidence < 0.50)
                and not body.conversation_history
            ):
                escalate = False
                final_response = (
                    "To help us route your ticket to the right team, could you share a bit more detail?\n\n"
                    "- Which product, feature, or integration are you having trouble with?\n"
                    "- What error message or unexpected behavior are you seeing?\n"
                    "- When did this issue start?\n\n"
                    "The more context you provide, the faster we can resolve your issue."
                )

            elif escalate:
                yield {"event": "status", "data": json.dumps({
                    "step": "agent5_escalation",
                    "message": _NODE_LABELS["agent5_escalation"],
                })}
                merged_state   = await asyncio.to_thread(agent5_escalation, merged_state)
                final_response = merged_state.get("final_response") or (
                    "Your ticket has been escalated to our support team. "
                    "A specialist will contact you within the SLA window."
                )
            else:
                yield {"event": "status", "data": json.dumps({
                    "step": "response_synthesizer",
                    "message": "Generating response...",
                })}

                token_queue: asyncio.Queue = asyncio.Queue()

                def run_token_stream():
                    _thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_thread_loop)
                    try:
                        def _inner():
                            try:
                                for token in stream_synthesizer_tokens(merged_state):
                                    asyncio.run_coroutine_threadsafe(token_queue.put(("token", token)), loop)
                            except Exception as e:
                                asyncio.run_coroutine_threadsafe(token_queue.put(("error", str(e))), loop)
                            finally:
                                asyncio.run_coroutine_threadsafe(token_queue.put(("done", None)), loop)
                        _ctx.run(_inner)
                    except Exception as e:
                        asyncio.run_coroutine_threadsafe(token_queue.put(("error", str(e))), loop)
                        asyncio.run_coroutine_threadsafe(token_queue.put(("done", None)), loop)
                    finally:
                        _thread_loop.close()

                threading.Thread(target=run_token_stream, daemon=True).start()

                accumulated = ""
                while True:
                    try:
                        t_type, t_data = await asyncio.wait_for(token_queue.get(), timeout=_AGENT_TIMEOUT)
                    except asyncio.TimeoutError:
                        yield {"event": "error", "data": json.dumps({"message": "Synthesizer timed out. Please try again."})}
                        return
                    if t_type == "error":
                        _log.error("SYNTHESIZER ERROR: %s", t_data)
                        yield {"event": "error", "data": json.dumps({"message": _user_friendly_error(t_data)})}
                        return
                    if t_type == "done":
                        break
                    accumulated += t_data
                    yield {"event": "token", "data": json.dumps({"token": t_data})}

                final_response = accumulated or (
                    "We were unable to generate an automated response at this time. "
                    "Your ticket has been logged and our support team will follow up shortly."
                )

            # ── Phase 3: done event ────────────────────────────────────────────
            rag_result   = merged_state.get("rag_result") or {}
            # Only surface sources for routes that actually ran Agent 2.
            # MemorySaver persists rag_result across turns — a SQL ticket following
            # a RAG ticket in the same thread would otherwise show stale sources.
            _route = (merged_state.get("classification") or {}).get("routing_path", "")
            sources = (
                list({
                    c.get("source", "") for c in (rag_result.get("chunks") or []) if c.get("source")
                })
                if _route in ("RAG", "Hybrid", "Multi-Agent") else []
            )

            updated_history = body.conversation_history + [
                {"role": "user",      "content": body.ticket_text},
                {"role": "assistant", "content": final_response},
            ]

            severity_out = {k: v for k, v in severity_assessment.items() if not k.startswith("_")}

            # Strip internal token-count fields from escalation package before sending
            raw_esc_pkg  = merged_state.get("escalation_package") or {}
            escalation_out = {k: v for k, v in raw_esc_pkg.items() if not k.startswith("_")}

            # Output guardrail — redact PII before the response reaches the user
            _out = check_output(final_response)
            if not _out.clean:
                _log.warning("OUTPUT GUARDRAIL: %s", _out.reason)
            final_response = _out.response

            _log.info("DONE EVENT: final_response length=%d preview=%r", len(final_response or ""), (final_response or "")[:80])
            yield {
                "event": "done",
                "data": json.dumps({
                    "final_response":       final_response,
                    "classification":       classification,
                    "severity_assessment":  severity_out,
                    "escalation_package":   escalation_out,
                    "conversation_history": updated_history,
                    "sources":              sources,
                    "submitted_at":         submitted_at,
                    "latency_ms":           round(timer() - t0),
                }, default=_json_serial),
            }

            # ── Post-response: RAGAS evaluation — yields scores event to frontend ──
            # Runs AFTER done so the user sees the response immediately.
            # RAGAS uses statement-level decomposition (not a single 0-1 rating)
            # which prevents the self-grading inflation seen with the LLM judge.
            # Scores pushed to Langfuse trace for SLO #2, #4, #9 tracking.
            try:
                # Skip RAGAS for multi-turn follow-ups: response draws on conversation
                # history not present in the current retrieval context, so faithfulness
                # and context_precision would score 0% even for correct answers.
                if body.conversation_history:
                    raise StopIteration  # jump to except block → no scores event sent

                from src.evaluation.ragas_eval import score_ragas
                rag_chunks   = (rag_result.get("chunks") or [])[:5]
                contexts     = [str(c.get("text", ""))[:1200] for c in rag_chunks if c.get("text")]

                # SQL rows passed separately: included in faithfulness so account-data
                # claims can be verified, but excluded from context_precision which
                # measures RAG retrieval quality only (ADR-017).
                sql_rows = (merged_state.get("sql_result") or {}).get("rows") or []
                sql_contexts = (
                    ["SQL query results:\n" + "\n".join(str(r) for r in sql_rows[:10])]
                    if sql_rows else []
                )
                routing_path = classification.get("routing_path", "")
                is_escalated = bool(severity_assessment.get("escalate", False))

                scored = await asyncio.to_thread(
                    score_ragas,
                    body.ticket_text,
                    final_response,
                    contexts,
                    is_escalated,
                    routing_path,
                    sql_contexts,
                )

                faith = scored.get("ragas_faithfulness")
                relev = scored.get("ragas_answer_relevance")
                prec  = scored.get("ragas_context_precision")

                if faith is not None:
                    root.score_trace(name="ragas_faithfulness",
                                     value=faith, comment="RAGAS — SLO #2 target >= 0.95")
                if relev is not None:
                    root.score_trace(name="ragas_answer_relevance",
                                     value=relev, comment="RAGAS — SLO #4 target >= 0.85")
                if prec is not None:
                    root.score_trace(name="ragas_context_precision",
                                     value=prec,  comment="RAGAS — SLO #9 target >= 0.80")

                yield {
                    "event": "scores",
                    "data": json.dumps({
                        "faithfulness":      faith,
                        "relevance":         relev,
                        "context_precision": prec,
                        "relevance_type":    scored.get("relevance_type", "answer_relevance"),
                        "source":            "ragas",
                    }),
                }
            except (Exception, StopIteration):
                pass

        finally:
            # Always close and flush the trace — even on timeout, error, or client disconnect
            clf      = merged_state.get("classification") or {}
            tsr_pass = not (clf.get("confidence") == 0.0 and clf.get("category") == "ambiguous")
            log_ticket_complete(
                root, lf_token,
                final_response=final_response,
                total_cost_usd=0.0,
                total_latency_ms=timer() - t0,
                tsr_pass=tsr_pass,
                routing_path=clf.get("routing_path", ""),
            )

    return EventSourceResponse(event_generator())
