"""
Unit tests for Agent 3 (src/agents/agent3_sql.py).

Agent 3 is a thin delegating node — it calls sql.query() and stores the result.
Tests verify: correct delegation, state storage, SQL result passthrough.
"""

from unittest.mock import patch

import pytest

from src.agents.agent3_sql import agent3_sql


def _state(ticket_text="Show me my account details.", category="billing"):
    return {
        "ticket_text":          ticket_text,
        "conversation_history": [],
        "classification": {
            "category":     category,
            "severity":     "Medium",
            "routing_path": "SQL",
            "confidence":   0.85,
            "reasoning":    "Account data required",
        },
        "rag_result":          None,
        "sql_result":          None,
        "severity_assessment": None,
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    0,
    }


class TestAgent3Sql:
    def test_sql_result_stored_in_state(self):
        state = _state()
        fake_result = {"rows": [{"customer_id": 1, "company_name": "Acme Corp"}], "row_count": 1}
        with patch("src.agents.agent3_sql.query", return_value=fake_result):
            result = agent3_sql(state)

        assert result["sql_result"] == fake_result
        assert result["sql_result"]["rows"][0]["company_name"] == "Acme Corp"

    def test_query_called_with_ticket_text_and_classification(self):
        state = _state(ticket_text="What is my renewal date?")
        with patch("src.agents.agent3_sql.query", return_value={}) as mock_query:
            agent3_sql(state)

        mock_query.assert_called_once_with(
            "What is my renewal date?",
            state["classification"],
        )

    def test_empty_rows_stored(self):
        state = _state()
        with patch("src.agents.agent3_sql.query", return_value={"rows": [], "row_count": 0}):
            result = agent3_sql(state)

        assert result["sql_result"]["rows"] == []
        assert result["sql_result"]["row_count"] == 0

    def test_sql_error_result_stored(self):
        state = _state()
        error_result = {"error": "relation does not exist", "rows": [], "row_count": 0}
        with patch("src.agents.agent3_sql.query", return_value=error_result):
            result = agent3_sql(state)

        assert result["sql_result"]["error"] == "relation does not exist"

    def test_other_state_fields_unchanged(self):
        state = _state()
        with patch("src.agents.agent3_sql.query", return_value={"rows": [], "row_count": 0}):
            result = agent3_sql(state)

        assert result["rag_result"]          is None
        assert result["severity_assessment"] is None
        assert result["final_response"]      is None
        assert result["escalation_package"]  is None

    def test_returns_state_object(self):
        state = _state()
        with patch("src.agents.agent3_sql.query", return_value={}):
            result = agent3_sql(state)

        assert result is state
