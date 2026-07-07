"""
Unit tests for Agent 2 (src/agents/agent2_rag.py).

Two layers:
  1. _build_sql_context() — pure function, no mocking needed
  2. agent2_rag() node — retrieve() and Langfuse mocked

Tests verify:
  - SQL context enriches the RAG query for Hybrid/Multi-Agent routes
  - RAG-only path passes use_cache=True; Hybrid passes use_cache=False (ADR-017)
  - rag_result is stored correctly in state
  - Other state fields are not modified
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent2_rag import _build_sql_context, agent2_rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(routing_path="RAG", reasoning="API auth error", sql_rows=None):
    return {
        "ticket_text":          "Test ticket",
        "conversation_history": [],
        "classification": {
            "category":     "integration_api",
            "severity":     "High",
            "routing_path": routing_path,
            "confidence":   0.90,
            "reasoning":    reasoning,
        },
        "rag_result":          None,
        "sql_result":          {"rows": sql_rows} if sql_rows is not None else None,
        "severity_assessment": None,
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    0,
    }


def _mock_chunks(n=3):
    return [{"source": f"doc_{i}.pdf", "text": f"chunk text {i}", "score": 0.9 - i * 0.1}
            for i in range(n)]


# ---------------------------------------------------------------------------
# _build_sql_context — pure function
# ---------------------------------------------------------------------------

class TestBuildSqlContext:
    def test_empty_rows_returns_empty_string(self):
        assert _build_sql_context([]) == ""

    def test_tier_sla_region_concatenated(self):
        rows = [{"subscription_tier": "Enterprise", "sla_level": "Priority", "region": "US-East"}]
        ctx = _build_sql_context(rows)
        assert "Enterprise" in ctx
        assert "Priority" in ctx
        assert "US-East" in ctx

    def test_root_cause_truncated_to_100_chars(self):
        long_cause = "X" * 200
        rows = [{"root_cause": long_cause}]
        ctx = _build_sql_context(rows)
        assert len(ctx) <= 100

    def test_severity_level_field_used(self):
        rows = [{"severity_level": "Critical"}]
        ctx = _build_sql_context(rows)
        assert "Critical" in ctx

    def test_severity_fallback_used_when_no_severity_level(self):
        rows = [{"severity": "High"}]
        ctx = _build_sql_context(rows)
        assert "High" in ctx

    def test_severity_level_takes_priority_over_severity(self):
        rows = [{"severity_level": "Critical", "severity": "Low"}]
        ctx = _build_sql_context(rows)
        assert "Critical" in ctx

    def test_missing_fields_skipped(self):
        rows = [{"subscription_tier": "Standard"}]
        ctx = _build_sql_context(rows)
        assert ctx == "Standard"

    def test_only_first_row_used(self):
        rows = [
            {"subscription_tier": "Enterprise"},
            {"subscription_tier": "Standard"},
        ]
        ctx = _build_sql_context(rows)
        assert "Enterprise" in ctx
        assert ctx.count("Enterprise") == 1


# ---------------------------------------------------------------------------
# agent2_rag() node
# ---------------------------------------------------------------------------

class TestAgent2RagNode:
    def test_rag_result_stored_in_state(self):
        state  = _state(routing_path="RAG")
        chunks = _mock_chunks(3)
        with patch("src.agents.agent2_rag.retrieve", return_value=chunks), \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            result = agent2_rag(state)

        assert result["rag_result"] is not None
        assert result["rag_result"]["chunks"] == chunks
        assert len(result["rag_result"]["chunks"]) == 3

    def test_rag_only_path_uses_cache(self):
        state = _state(routing_path="RAG")
        with patch("src.agents.agent2_rag.retrieve", return_value=[]) as mock_retrieve, \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            agent2_rag(state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs.get("use_cache") is True

    def test_hybrid_path_bypasses_cache(self):
        """Hybrid uses SQL-enriched query — customer-specific context must not be cached (ADR-017)."""
        state = _state(routing_path="Hybrid")
        with patch("src.agents.agent2_rag.retrieve", return_value=[]) as mock_retrieve, \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            agent2_rag(state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs.get("use_cache") is False

    def test_multi_agent_path_bypasses_cache(self):
        state = _state(routing_path="Multi-Agent")
        with patch("src.agents.agent2_rag.retrieve", return_value=[]) as mock_retrieve, \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            agent2_rag(state)

        _, kwargs = mock_retrieve.call_args
        assert kwargs.get("use_cache") is False

    def test_sql_context_enriches_query(self):
        """Hybrid: SQL rows should be appended to the reasoning to enrich the query."""
        sql_rows = [{"subscription_tier": "Enterprise", "sla_level": "Priority", "region": "EU"}]
        state = _state(routing_path="Hybrid", reasoning="API errors after migration", sql_rows=sql_rows)
        with patch("src.agents.agent2_rag.retrieve", return_value=[]) as mock_retrieve, \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            result = agent2_rag(state)

        query_sent = mock_retrieve.call_args[0][0]
        assert "Enterprise" in query_sent
        assert "API errors after migration" in query_sent

    def test_no_sql_context_uses_reasoning_only(self):
        state = _state(routing_path="RAG", reasoning="How do I configure OAuth?", sql_rows=None)
        with patch("src.agents.agent2_rag.retrieve", return_value=[]) as mock_retrieve, \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            agent2_rag(state)

        query_sent = mock_retrieve.call_args[0][0]
        assert query_sent == "How do I configure OAuth?"

    def test_other_state_fields_unchanged(self):
        state = _state(routing_path="RAG")
        with patch("src.agents.agent2_rag.retrieve", return_value=[]), \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            result = agent2_rag(state)

        assert result["severity_assessment"] is None
        assert result["final_response"]      is None
        assert result["escalation_package"]  is None

    def test_empty_chunks_stored(self):
        state = _state(routing_path="RAG")
        with patch("src.agents.agent2_rag.retrieve", return_value=[]), \
             patch("src.agents.agent2_rag.get_langfuse", return_value=MagicMock()), \
             patch("src.retrieval.semantic_cache.cache_stats", return_value={"hits": 0, "hit_rate": 0.0}):
            result = agent2_rag(state)

        assert result["rag_result"]["chunks"] == []
