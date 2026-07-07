"""
Unit tests for Agent 5 (src/agents/agent5_escalation.py).

Two layers:
  1. Pure functions: _get_team(), _fallback_summary(), team/priority mappings
  2. agent5_escalation() node — LLM, Jira, and DB calls mocked

Tests verify:
  - Team routing maps categories to correct human teams
  - Keyword overrides route to specialist teams (Identity, Data Engineering)
  - escalation_package contains all required context fields (README: full context transfer)
  - final_response includes reference_id, team name, and priority
  - Jira key appended to final_response when Jira succeeds
  - Fallback EscalationSummary used when LLM fails
  - Jira failure is non-blocking — escalation still completes
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent5_escalation import (
    _get_team,
    _fallback_summary,
    _TEAM_ROUTING,
    _SEVERITY_TO_PRIORITY,
    agent5_escalation,
    EscalationSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(category="integration_api", severity_assessed="High",
           ticket_text="API keeps returning 401 errors.", sql_rows=None, rag_chunks=None):
    return {
        "ticket_text":          ticket_text,
        "conversation_history": [],
        "classification": {
            "category":     category,
            "severity":     "Medium",
            "routing_path": "Multi-Agent",
            "confidence":   0.90,
            "reasoning":    "escalation reason",
        },
        "rag_result":  {"chunks": rag_chunks or []},
        "sql_result":  {"rows": sql_rows or [], "row_count": len(sql_rows or [])},
        "severity_assessment": {
            "severity":           severity_assessed,
            "confidence":         0.90,
            "escalate":           True,
            "reasoning":          "critical signals",
            "escalation_trigger": "critical_severity",
        },
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    0,
    }


def _mock_llm_response(summary="Context summary.", actions=None):
    parsed = EscalationSummary(
        context_summary=summary,
        recommended_actions=actions or ["Review ticket", "Contact customer"],
    )
    raw_msg = MagicMock()
    raw_msg.usage_metadata = {"input_tokens": 600, "output_tokens": 120}
    return {"parsed": parsed, "parsing_error": None, "raw": raw_msg}


def _prompt_obj():
    obj = MagicMock()
    obj.compile.return_value = "mocked prompt"
    return obj


# ---------------------------------------------------------------------------
# _get_team — pure team routing function
# ---------------------------------------------------------------------------

class TestGetTeam:
    @pytest.mark.parametrize("category,expected", [
        ("production_incident", "SRE Team"),
        ("security",            "Security Team"),
        ("integration_api",     "Backend Engineering"),
        ("performance_latency", "Platform Engineering"),
        ("billing",             "Account Management"),
        ("usage_configuration", "L2 Support"),
        ("ambiguous",           "L2 Support"),
    ])
    def test_default_category_routing(self, category, expected):
        assert _get_team(category, "generic ticket text") == expected

    def test_auth_keyword_overrides_integration_api(self):
        assert _get_team("integration_api", "Getting 403 forbidden on OAuth endpoint") == "Identity & Access Team"

    def test_token_keyword_overrides_integration_api(self):
        assert _get_team("integration_api", "Bearer token is being rejected") == "Identity & Access Team"

    def test_auth_keyword_overrides_production_incident(self):
        assert _get_team("production_incident", "Authentication failure for all users") == "Identity & Access Team"

    def test_data_loss_keyword_overrides_production_incident(self):
        assert _get_team("production_incident", "Customer records disappeared from the database") == "Data Engineering Team"

    def test_billing_keyword_overrides_integration_api(self):
        assert _get_team("integration_api", "API rate limit tied to our billing subscription plan") == "Account Management"

    def test_conflict_keyword_overrides_usage_configuration(self):
        assert _get_team("usage_configuration", "The installation guide conflicts with the API integration guide") == "Platform Engineering"

    def test_unknown_category_falls_back_to_l2(self):
        assert _get_team("unknown_category", "some ticket text") == "L2 Support"

    def test_billing_keyword_does_not_override_security(self):
        """Security should stay as Security Team even with billing keywords."""
        assert _get_team("security", "invoice billing subscription issue") == "Security Team"


# ---------------------------------------------------------------------------
# _fallback_summary — pure function
# ---------------------------------------------------------------------------

class TestFallbackSummary:
    def test_returns_escalation_summary_instance(self):
        result = _fallback_summary("My API is broken.")
        assert isinstance(result, EscalationSummary)

    def test_context_summary_contains_ticket_text(self):
        result = _fallback_summary("My API is broken.")
        assert "My API is broken." in result.context_summary

    def test_long_ticket_text_truncated_at_500(self):
        long_text = "X" * 600
        result = _fallback_summary(long_text)
        assert len(result.context_summary) < 600

    def test_recommended_actions_not_empty(self):
        result = _fallback_summary("Any ticket")
        assert len(result.recommended_actions) >= 1


# ---------------------------------------------------------------------------
# _SEVERITY_TO_PRIORITY mapping
# ---------------------------------------------------------------------------

class TestSeverityPriorityMapping:
    @pytest.mark.parametrize("severity,expected", [
        ("Critical", "P1"),
        ("High",     "P2"),
        ("Medium",   "P3"),
        ("Low",      "P4"),
    ])
    def test_severity_maps_to_priority(self, severity, expected):
        assert _SEVERITY_TO_PRIORITY[severity] == expected


# ---------------------------------------------------------------------------
# agent5_escalation() node
# ---------------------------------------------------------------------------

class TestAgent5EscalationNode:
    def _run(self, state, llm_response=None, jira_key=None, jira_exception=None):
        resp = llm_response or _mock_llm_response()
        with patch("src.agents.agent5_escalation._llm") as mock_llm, \
             patch("src.agents.agent5_escalation.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("src.agents.agent5_escalation._fetch_incident_logs", return_value=[]), \
             patch("src.integrations.jira_client.create_jira_issue",
                   side_effect=jira_exception or (lambda pkg: jira_key)), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = resp
            return agent5_escalation(state)

    def test_escalation_package_stored(self):
        state = _state()
        result = self._run(state, jira_key="SUP-42")
        assert result["escalation_package"] is not None

    def test_escalation_package_has_required_fields(self):
        state = _state()
        result = self._run(state)
        pkg = result["escalation_package"]
        for field in ("reference_id", "team", "priority", "category", "severity",
                      "context_summary", "recommended_actions",
                      "ticket_history", "account_data", "incident_logs",
                      "documentation_context", "agent_reasoning"):
            assert field in pkg, f"Missing field: {field}"

    def test_reference_id_format(self):
        state = _state()
        result = self._run(state)
        ref_id = result["escalation_package"]["reference_id"]
        assert re.match(r"^ESC-[A-F0-9]{8}$", ref_id), f"Unexpected ref_id format: {ref_id}"

    def test_final_response_contains_team_and_priority(self):
        state = _state(category="production_incident", severity_assessed="Critical")
        result = self._run(state)
        resp = result["final_response"]
        assert "SRE Team" in resp
        assert "P1" in resp

    def test_final_response_contains_reference_id(self):
        state = _state()
        result = self._run(state)
        ref_id = result["escalation_package"]["reference_id"]
        assert ref_id in result["final_response"]

    def test_jira_key_appended_when_successful(self):
        state = _state()
        result = self._run(state, jira_key="SUP-7")
        assert "SUP-7" in result["final_response"]
        assert result["escalation_package"]["jira_issue_key"] == "SUP-7"

    def test_jira_failure_is_non_blocking(self):
        """Jira call fails but escalation still completes successfully."""
        state = _state()
        result = self._run(state, jira_exception=Exception("Jira is down"))
        assert result["escalation_package"] is not None
        assert result["final_response"] is not None
        assert result["escalation_package"]["jira_issue_key"] is None
        assert "jira_error" in result["escalation_package"]

    def test_no_jira_key_no_jira_suffix_in_response(self):
        state = _state()
        result = self._run(state, jira_key=None)
        assert "Jira:" not in result["final_response"]

    def test_fallback_used_when_llm_fails(self):
        state = _state(ticket_text="Our entire service is down.")
        with patch("src.agents.agent5_escalation._llm") as mock_llm, \
             patch("src.agents.agent5_escalation.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("src.agents.agent5_escalation._fetch_incident_logs", return_value=[]), \
             patch("src.integrations.jira_client.create_jira_issue", return_value=None), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("Azure down")
            result = agent5_escalation(state)

        pkg = result["escalation_package"]
        assert pkg is not None
        assert "Our entire service is down." in pkg["context_summary"]

    def test_sql_rows_included_in_account_data(self):
        sql_rows = [{"customer_id": 99, "company_name": "BigCorp", "subscription_tier": "Enterprise"}]
        state = _state(sql_rows=sql_rows)
        result = self._run(state)
        assert result["escalation_package"]["account_data"] == sql_rows

    def test_rag_chunks_included_in_documentation_context(self):
        rag_chunks = [{"source": "SLA_Policy.pdf", "text": "SLA terms here"}]
        state = _state(rag_chunks=rag_chunks)
        result = self._run(state)
        doc_ctx = result["escalation_package"]["documentation_context"]
        assert len(doc_ctx) == 1
        assert doc_ctx[0]["source"] == "SLA_Policy.pdf"

    def test_category_ack_in_final_response(self):
        state = _state(category="security")
        result = self._run(state)
        assert "security vulnerability" in result["final_response"].lower()
