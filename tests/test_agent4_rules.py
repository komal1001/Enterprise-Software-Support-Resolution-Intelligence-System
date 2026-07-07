"""
Unit tests for Agent 4 deterministic (hard-rule) logic.

These functions contain no LLM calls — pure Python logic that guarantees
SLO #11 (Escalation Recall = 100%) regardless of LLM output.

Tests cover:
  - _check_hard_escalation():  the 4 hard rules that force escalate=True
  - is_below_confidence_threshold(): per-category threshold values
  - _detect_missing_retrieval(): reflection loop trigger
"""

import pytest
from src.agents.agent4_severity import (
    _check_hard_escalation,
    is_below_confidence_threshold,
    _detect_missing_retrieval,
    CONFIDENCE_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# is_below_confidence_threshold — per ADR-005 threshold table
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    def test_production_incident_threshold_is_85(self):
        assert CONFIDENCE_THRESHOLDS["production_incident"] == 0.85

    def test_security_threshold_is_85(self):
        assert CONFIDENCE_THRESHOLDS["security"] == 0.85

    def test_integration_api_threshold_is_70(self):
        assert CONFIDENCE_THRESHOLDS["integration_api"] == 0.70

    def test_performance_latency_threshold_is_70(self):
        assert CONFIDENCE_THRESHOLDS["performance_latency"] == 0.70

    def test_usage_configuration_threshold_is_65(self):
        assert CONFIDENCE_THRESHOLDS["usage_configuration"] == 0.65

    def test_billing_threshold_is_65(self):
        assert CONFIDENCE_THRESHOLDS["billing"] == 0.65

    def test_ambiguous_threshold_is_0(self):
        assert CONFIDENCE_THRESHOLDS["ambiguous"] == 0.00

    def test_below_production_incident_threshold(self):
        assert is_below_confidence_threshold("production_incident", 0.84) is True

    def test_at_production_incident_threshold(self):
        assert is_below_confidence_threshold("production_incident", 0.85) is False

    def test_above_production_incident_threshold(self):
        assert is_below_confidence_threshold("production_incident", 0.90) is False

    def test_below_security_threshold(self):
        assert is_below_confidence_threshold("security", 0.79) is True

    def test_below_billing_threshold(self):
        assert is_below_confidence_threshold("billing", 0.60) is True

    def test_at_billing_threshold(self):
        assert is_below_confidence_threshold("billing", 0.65) is False

    def test_unknown_category_uses_default_70(self):
        assert is_below_confidence_threshold("unknown_category", 0.69) is True
        assert is_below_confidence_threshold("unknown_category", 0.70) is False


# ---------------------------------------------------------------------------
# _check_hard_escalation — the 4 hard rules
# ---------------------------------------------------------------------------

class TestHardEscalationRules:
    def _clf(self, category="integration_api", severity="Medium", confidence=0.85):
        return {"category": category, "severity": severity, "confidence": confidence}

    # Rule 1 — ambiguous always escalates
    def test_ambiguous_category_always_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(category="ambiguous", severity="Low", confidence=0.99))
        assert escalate is True
        assert trigger == "ambiguous_category"

    # Rule 2 — low confidence on high-stakes category
    def test_low_confidence_production_incident_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="production_incident", severity="Medium", confidence=0.80
        ))
        assert escalate is True
        assert "low_confidence" in trigger

    def test_low_confidence_security_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="security", severity="Medium", confidence=0.70
        ))
        assert escalate is True
        assert "low_confidence" in trigger

    def test_low_confidence_integration_api_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="integration_api", severity="Low", confidence=0.65
        ))
        assert escalate is True
        assert "low_confidence" in trigger

    def test_sufficient_confidence_does_not_trigger_low_confidence_rule(self):
        escalate, _ = _check_hard_escalation(self._clf(
            category="integration_api", severity="Low", confidence=0.75
        ))
        assert escalate is False

    # Rule 3 — Critical severity always escalates
    def test_critical_severity_always_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="billing", severity="Critical", confidence=0.99
        ))
        assert escalate is True
        assert trigger == "critical_severity"

    def test_critical_on_any_category_escalates(self):
        for category in ["usage_configuration", "billing", "integration_api",
                         "performance_latency", "production_incident", "security"]:
            escalate, trigger = _check_hard_escalation(self._clf(
                category=category, severity="Critical", confidence=0.99
            ))
            assert escalate is True, f"Expected Critical {category} to escalate"
            assert trigger == "critical_severity"

    # Rule 4 — High severity on high-risk category
    def test_high_production_incident_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="production_incident", severity="High", confidence=0.90
        ))
        assert escalate is True
        assert "high_severity_production_incident" in trigger

    def test_high_security_escalates(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="security", severity="High", confidence=0.90
        ))
        assert escalate is True
        assert "high_severity_security" in trigger

    def test_high_integration_api_does_not_hard_escalate(self):
        """integration_api is NOT in _HIGH_RISK_CATEGORIES — High severity alone is not enough."""
        escalate, _ = _check_hard_escalation(self._clf(
            category="integration_api", severity="High", confidence=0.90
        ))
        assert escalate is False

    def test_high_performance_latency_does_not_hard_escalate(self):
        escalate, _ = _check_hard_escalation(self._clf(
            category="performance_latency", severity="High", confidence=0.90
        ))
        assert escalate is False

    def test_high_billing_does_not_hard_escalate(self):
        escalate, _ = _check_hard_escalation(self._clf(
            category="billing", severity="High", confidence=0.90
        ))
        assert escalate is False

    def test_medium_severity_does_not_hard_escalate(self):
        escalate, _ = _check_hard_escalation(self._clf(
            category="production_incident", severity="Medium", confidence=0.90
        ))
        assert escalate is False

    def test_low_severity_does_not_hard_escalate(self):
        escalate, _ = _check_hard_escalation(self._clf(
            category="security", severity="Low", confidence=0.90
        ))
        assert escalate is False

    def test_no_hard_escalation_returns_empty_trigger(self):
        escalate, trigger = _check_hard_escalation(self._clf(
            category="usage_configuration", severity="Low", confidence=0.90
        ))
        assert escalate is False
        assert trigger == ""


# ---------------------------------------------------------------------------
# _detect_missing_retrieval — reflection loop trigger
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    base = {
        "ticket_text":          "",
        "conversation_history": [],
        "classification":       {"routing_path": "RAG"},
        "rag_result":           None,
        "sql_result":           None,
        "severity_assessment":  None,
        "escalation_package":   None,
        "final_response":       None,
        "reflection_count":     0,
    }
    base.update(overrides)
    return base


class TestDetectMissingRetrieval:
    def test_rag_path_with_sql_signals_triggers_sql_reflection(self):
        state = _make_state(
            ticket_text="My account subscription is showing the wrong tier.",
            classification={"routing_path": "RAG"},
            sql_result=None,
        )
        assert _detect_missing_retrieval(state) == "sql"

    def test_rag_path_without_sql_signals_no_reflection(self):
        state = _make_state(
            ticket_text="How do I configure the SDK to use OAuth?",
            classification={"routing_path": "RAG"},
            sql_result=None,
        )
        assert _detect_missing_retrieval(state) is None

    def test_sql_path_with_rag_signals_triggers_rag_reflection(self):
        state = _make_state(
            ticket_text="How do I fix the error in the API integration setup?",
            classification={"routing_path": "SQL"},
            rag_result=None,
        )
        assert _detect_missing_retrieval(state) == "rag"

    def test_sql_path_without_rag_signals_no_reflection(self):
        state = _make_state(
            ticket_text="What is my current subscription renewal date?",
            classification={"routing_path": "SQL"},
            rag_result=None,
        )
        assert _detect_missing_retrieval(state) is None

    def test_max_reflection_count_prevents_loop(self):
        """Reflection can only fire once per ticket (reflection_count cap = 1)."""
        state = _make_state(
            ticket_text="My account subscription error during API integration setup.",
            classification={"routing_path": "RAG"},
            sql_result=None,
            reflection_count=1,   # already reflected once
        )
        assert _detect_missing_retrieval(state) is None

    def test_sql_error_triggers_reflection_on_sql_path(self):
        # Ticket text must not contain RAG signals (no substrings of _RAG_SIGNALS)
        # so the SQL-error check (3rd branch) fires instead of the RAG signal check.
        state = _make_state(
            ticket_text="List all tickets for customer 42.",
            classification={"routing_path": "SQL"},
            sql_result={"error": "relation does not exist"},
        )
        assert _detect_missing_retrieval(state) == "sql"

    def test_hybrid_path_with_sql_error_triggers_reflection(self):
        state = _make_state(
            ticket_text="Show me my account.",
            classification={"routing_path": "Hybrid"},
            sql_result={"error": "timeout"},
        )
        assert _detect_missing_retrieval(state) == "sql"

    def test_rag_path_with_sql_already_retrieved_no_reflection(self):
        """If SQL was already retrieved, no reflection needed on RAG path."""
        state = _make_state(
            ticket_text="My account subscription billing is wrong.",
            classification={"routing_path": "RAG"},
            sql_result={"rows": [{"customer_id": 1}]},   # SQL already ran
        )
        assert _detect_missing_retrieval(state) is None
