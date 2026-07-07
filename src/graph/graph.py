# LangGraph Graph — wires all 5 agents with conditional routing
#
# 4 routing paths driven by Agent 1's routing_path output:
#
#   RAG         → Agent1 → Agent2 → Agent4 → Response
#   SQL         → Agent1 → Agent3 → Agent4 → Response
#   Hybrid      → Agent1 → Agent3 → Agent2(SQL-enriched) → Agent4 → Response
#   Multi-Agent → Agent1 → Agent3 → Agent2(SQL-enriched) → Agent4 → Agent5 → Response
#
# Agent 4 runs on EVERY ticket after retrieval (ADR-005).
# Agent 5 runs only when Agent 4 sets severity_assessment["escalate"] = True.
#
# Hybrid/Multi-Agent: Agent 3 runs BEFORE Agent 2 (SQL-informed RAG).
# Agent 3 retrieves customer tier, region, and incident context first.
# Agent 2 uses that SQL context to issue a more specific retrieval query.
#
# Reflection loop: Agent 4 can route back to Agent 2 or Agent 3 if it detects
# that Agent 1's routing path missed a retrieval source (max 1 reflection).
#
# ADR-001 — LangGraph chosen over CrewAI for deterministic routing and SLO enforcement

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import TicketState
from src.agents.agent1_classify import agent1_classify
from src.agents.agent2_rag import agent2_rag
from src.agents.agent3_sql import agent3_sql
from src.agents.agent4_severity import agent4_severity
from src.agents.agent5_escalation import agent5_escalation
from src.agents.response_synthesizer import response_synthesizer


# ---------------------------------------------------------------------------
# Routing function 1 — after Agent 1 (classification)
#
# RAG-only → Agent 2 (no SQL needed)
# SQL / Hybrid / Multi-Agent → Agent 3 first (SQL-informed RAG pattern)
# ---------------------------------------------------------------------------

def route_after_agent1(state: TicketState) -> str:
    routing_path = state["classification"]["routing_path"]
    if routing_path == "RAG":
        return "agent2_rag"
    return "agent3_sql"


# ---------------------------------------------------------------------------
# Routing function 2 — after Agent 3 (SQL)
#
# Hybrid / Multi-Agent → Agent 2 next (SQL context now available for enrichment)
# SQL-only or reflection back-edge → Agent 4 directly
# ---------------------------------------------------------------------------

def route_after_agent3(state: TicketState) -> str:
    routing_path = state["classification"]["routing_path"]
    if routing_path in ("Hybrid", "Multi-Agent"):
        return "agent2_rag"
    return "agent4_severity"


# ---------------------------------------------------------------------------
# Routing function 3 — after Agent 2 (RAG)
#
# Agent 3 has already run for Hybrid/Multi-Agent tickets before Agent 2.
# All paths go directly to Agent 4 from here.
# ---------------------------------------------------------------------------

def route_after_agent2(state: TicketState) -> str:
    return "agent4_severity"


# ---------------------------------------------------------------------------
# Routing function 4 — after Agent 4 (severity assessment)
#
# Reflection loop: if Agent 4 detected missing retrieval (Agent 1 routed
# incorrectly), route back to Agent 2 or Agent 3 to collect the missing data.
# reflection_count is capped at 1 — Agent 4 only reflects once per ticket.
#
# Normal paths:
#   escalate=True  → Agent 5 for human escalation
#   escalate=False → response_synthesizer
# ---------------------------------------------------------------------------

def route_after_agent4(state: TicketState) -> str:
    assessment = state.get("severity_assessment") or {}

    needs_reretrieval = assessment.get("needs_reretrieval")
    if needs_reretrieval == "sql":
        return "agent3_sql"
    if needs_reretrieval == "rag":
        return "agent2_rag"

    if assessment.get("escalate", False):
        return "agent5_escalation"
    return "response_synthesizer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(TicketState)

    # Nodes
    graph.add_node("agent1_classify",      agent1_classify)
    graph.add_node("agent2_rag",           agent2_rag)
    graph.add_node("agent3_sql",           agent3_sql)
    graph.add_node("agent4_severity",      agent4_severity)
    graph.add_node("agent5_escalation",    agent5_escalation)
    graph.add_node("response_synthesizer", response_synthesizer)

    # Entry point
    graph.set_entry_point("agent1_classify")

    # Conditional edges
    graph.add_conditional_edges(
        "agent1_classify",
        route_after_agent1,
        {
            "agent2_rag": "agent2_rag",
            "agent3_sql": "agent3_sql",
        }
    )

    graph.add_conditional_edges(
        "agent3_sql",
        route_after_agent3,
        {
            "agent2_rag":      "agent2_rag",       # Hybrid/Multi-Agent: SQL-informed RAG
            "agent4_severity": "agent4_severity",  # SQL-only or reflection
        }
    )

    graph.add_conditional_edges(
        "agent2_rag",
        route_after_agent2,
        {
            "agent4_severity": "agent4_severity",
        }
    )

    graph.add_conditional_edges(
        "agent4_severity",
        route_after_agent4,
        {
            "agent5_escalation":    "agent5_escalation",
            "response_synthesizer": "response_synthesizer",
            "agent3_sql":           "agent3_sql",    # reflection back-edge
            "agent2_rag":           "agent2_rag",    # reflection back-edge
        }
    )

    # Direct edges
    graph.add_edge("agent5_escalation",    END)
    graph.add_edge("response_synthesizer", END)

    # ---------------------------------------------------------------------------
    # Checkpointer — saves full TicketState after every node completes.
    # thread_id in the config ties turns of the same conversation together.
    # LangGraph restores the previous state automatically on the next turn.
    #
    # MemorySaver  → stores in RAM, lost on server restart (development only)
    # PostgresSaver → persists to PostgreSQL, survives restarts (production)
    #
    # TODO Sprint 6: replace MemorySaver with PostgresSaver
    #   from langgraph.checkpoint.postgres import PostgresSaver
    #   checkpointer = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
    # ---------------------------------------------------------------------------
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Compiled graph — imported by FastAPI (src/api/routes.py) and eval pipeline
# ---------------------------------------------------------------------------

compiled_graph = build_graph()


# ---------------------------------------------------------------------------
# Pre-synth graph — runs agents 1–4 only, stopping before response_synthesizer
# and agent5_escalation. Used by the SSE streaming endpoint so that the
# synthesizer LLM call can be streamed token-by-token after the graph exits.
# ---------------------------------------------------------------------------

def route_after_agent4_pre_synth(state: TicketState) -> str:
    """Same reflection logic as route_after_agent4, but both terminal outcomes go to END."""
    assessment = state.get("severity_assessment") or {}
    if assessment.get("needs_reretrieval") == "sql":
        return "agent3_sql"
    if assessment.get("needs_reretrieval") == "rag":
        return "agent2_rag"
    return "end"


def build_pre_synth_graph():
    graph = StateGraph(TicketState)

    graph.add_node("agent1_classify", agent1_classify)
    graph.add_node("agent2_rag",      agent2_rag)
    graph.add_node("agent3_sql",      agent3_sql)
    graph.add_node("agent4_severity", agent4_severity)

    graph.set_entry_point("agent1_classify")

    graph.add_conditional_edges(
        "agent1_classify",
        route_after_agent1,
        {"agent2_rag": "agent2_rag", "agent3_sql": "agent3_sql"},
    )
    graph.add_conditional_edges(
        "agent3_sql",
        route_after_agent3,
        {"agent2_rag": "agent2_rag", "agent4_severity": "agent4_severity"},
    )
    graph.add_conditional_edges(
        "agent2_rag",
        route_after_agent2,
        {"agent4_severity": "agent4_severity"},
    )
    graph.add_conditional_edges(
        "agent4_severity",
        route_after_agent4_pre_synth,
        {"agent3_sql": "agent3_sql", "agent2_rag": "agent2_rag", "end": END},
    )

    return graph.compile()


pre_synth_compiled_graph = build_pre_synth_graph()
