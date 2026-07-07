"""
Unit tests for Agent 4 LLM phase (src/agents/agent4_severity.py).

Covers the agent4_severity() LangGraph node — LLM calls mocked.
Complements test_agent4_rules.py which covers only the pure functions.

Tests verify:
  - Hard escalation rules override LLM output (SLO #11)
  - Soft de-escalation guard blocks false positives (fixes "show escalation history" bug)
  - severity_assessment stored correctly on success
  - Retry logic: 3 attempts with fallback
  - Fallback escalates for safety (never drops a critical ticket)
  - needs_reretrieval attached to assessment from _detect_missing_retrieval
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent4_severity import agent4_severity, SeverityAssessment, _FALLBACK_ASSESSMENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(category="integration_api", severity="Medium", confidence=0.85,
           routing_path="RAG", ticket_text="My API keeps returning errors.",
           rag_result=None, sql_result=None, reflection_count=0):
    return {
        "ticket_text":          ticket_text,
        "conversation_history": [],
        "classification": {
            "category":     category,
            "severity":     severity,
            "routing_path": routing_path,
            "confidence":   confidence,
            "reasoning":    "test reasoning",
        },
        "rag_result":          rag_result,
        "sql_result":          sql_result,
        "severity_assessment": None,
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    reflection_count,
    }


def _mock_llm_response(severity="Medium", confidence=0.80,
                       escalate=False, reasoning="seems low risk",
                       escalation_trigger=None):
    parsed = SeverityAssessment(
        severity=severity,
        confidence=confidence,
        escalate=escalate,
        reasoning=reasoning,
        escalation_trigger=escalation_trigger,
    )
    raw_msg = MagicMock()
    raw_msg.usage_metadata = {"input_tokens": 400, "output_tokens": 60}
    return {"parsed": parsed, "parsing_error": None, "raw": raw_msg}


def _prompt_obj():
    obj = MagicMock()
    obj.compile.return_value = "mocked prompt"
    return obj


def _base_patches(llm_mock):
    """Returns a list of active patch context managers (already entered)."""
    return [
        patch("src.agents.agent4_severity._llm"),
        patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()),
        patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()),
        patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001),
        patch("time.sleep"),
    ]


# ---------------------------------------------------------------------------
# Normal path — no hard escalation, LLM runs
# ---------------------------------------------------------------------------

class TestAgent4NormalPath:
    def test_assessment_stored_in_state(self):
        state = _state()
        resp = _mock_llm_response(severity="Medium", confidence=0.80, escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = resp
            result = agent4_severity(state)

        sa = result["severity_assessment"]
        assert sa is not None
        assert sa["severity"]   == "Medium"
        assert sa["confidence"] == 0.80
        assert sa["escalate"]   is False

    def test_token_counts_stored(self):
        state = _state()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = agent4_severity(state)

        sa = result["severity_assessment"]
        assert sa["_input_tokens"]  == 400
        assert sa["_output_tokens"] == 60

    def test_other_state_fields_unchanged(self):
        state = _state()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = agent4_severity(state)

        assert result["rag_result"]        is None
        assert result["sql_result"]        is None
        assert result["final_response"]    is None
        assert result["escalation_package"] is None


# ---------------------------------------------------------------------------
# Hard escalation rules override LLM (SLO #11)
# ---------------------------------------------------------------------------

class TestAgent4HardRulesOverrideLLM:
    def test_critical_ticket_escalates_even_if_llm_says_no(self):
        """Hard rule: Critical severity → always escalate, LLM cannot override."""
        state = _state(category="billing", severity="Critical", confidence=0.99)
        llm_says_no = _mock_llm_response(severity="Low", escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_says_no
            result = agent4_severity(state)

        sa = result["severity_assessment"]
        assert sa["escalate"]           is True
        assert sa["escalation_trigger"] == "critical_severity"

    def test_ambiguous_category_escalates_always(self):
        state = _state(category="ambiguous", severity="Low", confidence=0.99)
        llm_says_no = _mock_llm_response(escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_says_no
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is True
        assert result["severity_assessment"]["escalation_trigger"] == "ambiguous_category"

    def test_high_production_incident_escalates_always(self):
        state = _state(category="production_incident", severity="High", confidence=0.95)
        llm_says_no = _mock_llm_response(severity="Medium", escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_says_no
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is True

    def test_low_confidence_security_escalates_always(self):
        """Below threshold (0.79 < 0.85) → hard escalate regardless of LLM."""
        state = _state(category="security", severity="Medium", confidence=0.79)
        llm_says_no = _mock_llm_response(escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_says_no
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is True


# ---------------------------------------------------------------------------
# Soft de-escalation guard (fixes false positive bug — "show escalation history")
# ---------------------------------------------------------------------------

class TestAgent4SoftDeEscalationGuard:
    def test_llm_false_positive_blocked_for_medium_severity(self):
        """
        LLM returns escalate=True for a Medium-severity billing ticket with no hard rule.
        Soft guard must suppress it — Medium billing cannot trigger escalation via LLM alone.
        """
        state = _state(category="billing", severity="Medium", confidence=0.80)
        llm_false_positive = _mock_llm_response(
            severity="Medium", escalate=True, escalation_trigger="some_reason"
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_false_positive
            result = agent4_severity(state)

        sa = result["severity_assessment"]
        assert sa["escalate"]           is False
        assert sa["escalation_trigger"] is None

    def test_llm_false_positive_blocked_for_high_non_risk_category(self):
        """High severity integration_api — NOT in _HIGH_RISK_CATEGORIES — soft guard blocks it."""
        state = _state(category="integration_api", severity="High", confidence=0.85)
        llm_false_positive = _mock_llm_response(
            severity="High", escalate=True, escalation_trigger="seems risky"
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_false_positive
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is False

    def test_llm_critical_escalation_allowed_when_no_hard_rule(self):
        """LLM re-assesses to Critical (upgraded from Medium) — soft guard allows it."""
        state = _state(category="integration_api", severity="Medium", confidence=0.85)
        llm_upgrades = _mock_llm_response(
            severity="Critical", escalate=True, escalation_trigger="found hidden critical"
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_upgrades
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is True

    def test_llm_high_production_incident_escalation_allowed(self):
        """LLM re-assesses to High + production_incident — soft guard allows (high-risk category)."""
        state = _state(category="production_incident", severity="Medium", confidence=0.90)
        llm_upgrades = _mock_llm_response(
            severity="High", escalate=True, escalation_trigger="service degraded"
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = llm_upgrades
            result = agent4_severity(state)

        assert result["severity_assessment"]["escalate"] is True


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestAgent4RetryLogic:
    def test_succeeds_on_second_attempt(self):
        state = _state()
        fail = {"parsed": None, "parsing_error": "parse error", "raw": MagicMock()}
        ok   = _mock_llm_response(severity="Low", escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [fail, ok]
            result = agent4_severity(state)

        assert mock_llm.invoke.call_count == 2
        assert result["severity_assessment"]["severity"] == "Low"

    def test_succeeds_on_third_attempt(self):
        state = _state()
        fail = {"parsed": None, "parsing_error": "error", "raw": MagicMock()}
        ok   = _mock_llm_response(severity="High", escalate=False)
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [fail, fail, ok]
            result = agent4_severity(state)

        assert mock_llm.invoke.call_count == 3

    def test_exception_triggers_retry(self):
        state = _state()
        ok = _mock_llm_response()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [RuntimeError("Azure timeout"), ok]
            result = agent4_severity(state)

        assert result["severity_assessment"] is not None


# ---------------------------------------------------------------------------
# Fallback — all 3 attempts fail → escalate for safety (SLO #11)
# ---------------------------------------------------------------------------

class TestAgent4Fallback:
    def test_all_attempts_fail_returns_safe_fallback(self):
        """Fallback always escalates — better to over-escalate than miss a critical ticket."""
        state = _state()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("Azure is down")
            result = agent4_severity(state)

        sa = result["severity_assessment"]
        assert sa["escalate"]           is True
        assert sa["escalation_trigger"] == "assessment_failure"
        assert sa["confidence"]         == 0.0

    def test_fallback_makes_exactly_3_attempts(self):
        state = _state()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("failure")
            agent4_severity(state)

        assert mock_llm.invoke.call_count == 3

    def test_fallback_does_not_raise(self):
        state = _state()
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("failure")
            try:
                result = agent4_severity(state)
                assert result["severity_assessment"] is not None
            except Exception as exc:
                pytest.fail(f"agent4_severity raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# needs_reretrieval — reflection loop wired into assessment
# ---------------------------------------------------------------------------

class TestAgent4ReflectionWiring:
    def test_needs_reretrieval_attached_to_assessment(self):
        """SQL signal in a RAG-routed ticket → needs_reretrieval='sql' in assessment."""
        state = _state(
            category="billing", severity="Medium", confidence=0.80,
            routing_path="RAG",
            ticket_text="My account subscription billing is wrong.",
            sql_result=None,
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = _mock_llm_response(escalate=False)
            result = agent4_severity(state)

        assert result["severity_assessment"]["needs_reretrieval"] == "sql"
        assert result["reflection_count"] == 1

    def test_no_reflection_when_signals_absent(self):
        state = _state(
            routing_path="RAG",
            ticket_text="How do I configure webhooks?",
            sql_result=None,
        )
        with patch("src.agents.agent4_severity._llm") as mock_llm, \
             patch("src.agents.agent4_severity.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.return_value = _mock_llm_response(escalate=False)
            result = agent4_severity(state)

        assert result["severity_assessment"]["needs_reretrieval"] is None
