"""
Unit tests for LangGraph routing functions (src/graph/graph.py).

These are pure functions — they read from TicketState dicts and return
the next node name.  No LLM calls, no DB, runs in < 10ms.

Covers:
  - route_after_agent1: RAG vs SQL/Hybrid/Multi-Agent split
  - route_after_agent3: Hybrid/Multi-Agent adds RAG step; SQL-only skips it
  - route_after_agent2: always goes to agent4
  - route_after_agent4: escalate → agent5; no escalate → synthesizer; reflection loops
"""

import pytest
from src.graph.graph import (
    route_after_agent1,
    route_after_agent3,
    route_after_agent2,
    route_after_agent4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(routing_path="RAG", escalate=False, needs_reretrieval=None):
    return {
        "ticket_text":          "test ticket",
        "conversation_history": [],
        "classification":       {"routing_path": routing_path},
        "rag_result":           None,
        "sql_result":           None,
        "severity_assessment":  {
            "severity":   "Medium",
            "confidence": 0.80,
            "escalate":   escalate,
            "reasoning":  "test",
            "needs_reretrieval": needs_reretrieval,
        },
        "escalation_package":   None,
        "final_response":       None,
        "reflection_count":     0,
    }


# ---------------------------------------------------------------------------
# route_after_agent1 — determines first retrieval agent
# ---------------------------------------------------------------------------

class TestRouteAfterAgent1:
    def test_rag_path_goes_to_agent2(self):
        assert route_after_agent1(_state("RAG")) == "agent2_rag"

    def test_sql_path_goes_to_agent3(self):
        assert route_after_agent1(_state("SQL")) == "agent3_sql"

    def test_hybrid_path_goes_to_agent3_first(self):
        """Hybrid: SQL runs before RAG so Agent 2 gets SQL-enriched context."""
        assert route_after_agent1(_state("Hybrid")) == "agent3_sql"

    def test_multi_agent_path_goes_to_agent3_first(self):
        assert route_after_agent1(_state("Multi-Agent")) == "agent3_sql"


# ---------------------------------------------------------------------------
# route_after_agent3 — SQL done, decide next step
# ---------------------------------------------------------------------------

class TestRouteAfterAgent3:
    def test_sql_only_goes_directly_to_agent4(self):
        assert route_after_agent3(_state("SQL")) == "agent4_severity"

    def test_hybrid_goes_to_agent2_after_sql(self):
        assert route_after_agent3(_state("Hybrid")) == "agent2_rag"

    def test_multi_agent_goes_to_agent2_after_sql(self):
        assert route_after_agent3(_state("Multi-Agent")) == "agent2_rag"

    def test_rag_only_not_expected_but_goes_to_agent4(self):
        """RAG path never reaches agent3, but if it somehow did, it skips RAG again."""
        assert route_after_agent3(_state("RAG")) == "agent4_severity"


# ---------------------------------------------------------------------------
# route_after_agent2 — RAG done, always severity check
# ---------------------------------------------------------------------------

class TestRouteAfterAgent2:
    def test_rag_path_goes_to_agent4(self):
        assert route_after_agent2(_state("RAG")) == "agent4_severity"

    def test_hybrid_path_goes_to_agent4(self):
        assert route_after_agent2(_state("Hybrid")) == "agent4_severity"

    def test_multi_agent_path_goes_to_agent4(self):
        assert route_after_agent2(_state("Multi-Agent")) == "agent4_severity"


# ---------------------------------------------------------------------------
# route_after_agent4 — final routing: escalate, synthesize, or reflect
# ---------------------------------------------------------------------------

class TestRouteAfterAgent4:
    def test_no_escalate_goes_to_synthesizer(self):
        assert route_after_agent4(_state(escalate=False)) == "response_synthesizer"

    def test_escalate_true_goes_to_agent5(self):
        assert route_after_agent4(_state(escalate=True)) == "agent5_escalation"

    def test_reflection_sql_goes_to_agent3(self):
        state = _state(escalate=False, needs_reretrieval="sql")
        assert route_after_agent4(state) == "agent3_sql"

    def test_reflection_rag_goes_to_agent2(self):
        state = _state(escalate=False, needs_reretrieval="rag")
        assert route_after_agent4(state) == "agent2_rag"

    def test_reflection_takes_priority_over_escalation(self):
        """Reflection fires before escalation check — missing data must be collected first."""
        state = _state(escalate=True, needs_reretrieval="sql")
        assert route_after_agent4(state) == "agent3_sql"

    def test_missing_severity_assessment_goes_to_synthesizer(self):
        """If severity_assessment is None (should not happen), default to synthesizer."""
        state = _state()
        state["severity_assessment"] = None
        assert route_after_agent4(state) == "response_synthesizer"

    def test_empty_severity_assessment_goes_to_synthesizer(self):
        state = _state()
        state["severity_assessment"] = {}
        assert route_after_agent4(state) == "response_synthesizer"


# ---------------------------------------------------------------------------
# Full path coverage — verify the 4 routing paths end at the right agents
# ---------------------------------------------------------------------------

class TestRoutingPathCoverage:
    """
    Trace each path through the routing functions and verify every hop.

    RAG         → agent2_rag → agent4_severity → response_synthesizer
    SQL         → agent3_sql → agent4_severity → response_synthesizer
    Hybrid      → agent3_sql → agent2_rag → agent4_severity → response_synthesizer
    Multi-Agent → agent3_sql → agent2_rag → agent4_severity → agent5_escalation
    """

    def test_rag_path_sequence(self):
        state = _state("RAG", escalate=False)
        assert route_after_agent1(state)  == "agent2_rag"
        assert route_after_agent2(state)  == "agent4_severity"
        assert route_after_agent4(state)  == "response_synthesizer"

    def test_sql_path_sequence(self):
        state = _state("SQL", escalate=False)
        assert route_after_agent1(state)  == "agent3_sql"
        assert route_after_agent3(state)  == "agent4_severity"
        assert route_after_agent4(state)  == "response_synthesizer"

    def test_hybrid_path_sequence(self):
        state = _state("Hybrid", escalate=False)
        assert route_after_agent1(state)  == "agent3_sql"
        assert route_after_agent3(state)  == "agent2_rag"
        assert route_after_agent2(state)  == "agent4_severity"
        assert route_after_agent4(state)  == "response_synthesizer"

    def test_multi_agent_path_sequence(self):
        state = _state("Multi-Agent", escalate=True)
        assert route_after_agent1(state)  == "agent3_sql"
        assert route_after_agent3(state)  == "agent2_rag"
        assert route_after_agent2(state)  == "agent4_severity"
        assert route_after_agent4(state)  == "agent5_escalation"
