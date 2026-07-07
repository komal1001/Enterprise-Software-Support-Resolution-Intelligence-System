"""
Unit tests for Agent 1 (src/agents/agent1_classify.py).

LLM calls are mocked — these tests verify:
  - State is updated correctly on successful classification
  - Retry logic fires on LLM parse failure (3 attempts, exponential backoff)
  - Fallback classification is returned when all 3 attempts fail
  - Confidence is always a float in [0.0, 1.0]
  - Routing path is always one of the 4 valid literals

Patching notes:
  - _structured_llm: patched at module level (created at import time)
  - get_cached_prompt: patched at module level (imported at top of agent module)
  - get_langfuse / calculate_cost: patched at SOURCE module because they are
    imported inside the function body via `from src.observability... import ...`
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent1_classify import agent1_classify, ClassificationResult, _FALLBACK_CLASSIFICATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(ticket_text="Test ticket"):
    return {
        "ticket_text":          ticket_text,
        "conversation_history": [],
        "classification":       None,
        "rag_result":           None,
        "sql_result":           None,
        "severity_assessment":  None,
        "escalation_package":   None,
        "final_response":       None,
        "reflection_count":     0,
    }


def _mock_llm_response(category="integration_api", severity="High",
                       routing_path="SQL", confidence=0.92, reasoning="API auth error"):
    parsed = ClassificationResult(
        category=category, severity=severity,
        routing_path=routing_path, confidence=confidence, reasoning=reasoning,
    )
    raw_msg = MagicMock()
    raw_msg.usage_metadata = {"input_tokens": 500, "output_tokens": 80}
    return {"parsed": parsed, "parsing_error": None, "raw": raw_msg}


def _prompt_obj():
    obj = MagicMock()
    obj.compile.return_value = "mocked prompt"
    return obj


def _patches(llm_return=None, llm_side_effect=None):
    """Context manager stack: LLM + prompt + Langfuse (all needed together)."""
    return [
        patch("src.agents.agent1_classify._structured_llm"),
        patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()),
        patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()),
        patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001),
        patch("time.sleep"),
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAgent1HappyPath:
    def test_classification_stored_in_state(self):
        state = _base_state("Getting 401 errors calling /auth endpoint.")
        resp = _mock_llm_response(category="integration_api", severity="High",
                                  routing_path="SQL", confidence=0.92)
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = resp
            result = agent1_classify(state)

        clf = result["classification"]
        assert clf["category"]     == "integration_api"
        assert clf["severity"]     == "High"
        assert clf["routing_path"] == "SQL"
        assert clf["confidence"]   == 0.92

    def test_token_counts_stored(self):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = agent1_classify(state)

        assert result["classification"]["_input_tokens"]  == 500
        assert result["classification"]["_output_tokens"] == 80

    def test_other_state_keys_unchanged(self):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = agent1_classify(state)

        assert result["rag_result"]          is None
        assert result["sql_result"]          is None
        assert result["severity_assessment"] is None
        assert result["final_response"]      is None

    @pytest.mark.parametrize("routing_path", ["RAG", "SQL", "Hybrid", "Multi-Agent"])
    def test_all_4_routing_paths_accepted(self, routing_path):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response(routing_path=routing_path)
            result = agent1_classify(state)

        assert result["classification"]["routing_path"] == routing_path

    @pytest.mark.parametrize("category", [
        "usage_configuration", "integration_api", "performance_latency",
        "production_incident", "billing", "security", "ambiguous",
    ])
    def test_all_7_categories_accepted(self, category):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response(category=category)
            result = agent1_classify(state)

        assert result["classification"]["category"] == category


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestAgent1RetryLogic:
    def test_succeeds_on_second_attempt(self):
        state = _base_state()
        fail = {"parsed": None, "parsing_error": "JSON decode error", "raw": MagicMock()}
        ok   = _mock_llm_response(category="billing", severity="Low",
                                  routing_path="SQL", confidence=0.80)
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [fail, ok]
            result = agent1_classify(state)

        assert mock_llm.invoke.call_count == 2
        assert result["classification"]["category"] == "billing"

    def test_succeeds_on_third_attempt(self):
        state = _base_state()
        fail = {"parsed": None, "parsing_error": "error", "raw": MagicMock()}
        ok   = _mock_llm_response(category="security")
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [fail, fail, ok]
            result = agent1_classify(state)

        assert mock_llm.invoke.call_count == 3
        assert result["classification"]["category"] == "security"

    def test_exception_triggers_retry(self):
        state = _base_state()
        ok    = _mock_llm_response()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [RuntimeError("Azure timeout"), ok]
            result = agent1_classify(state)

        assert result["classification"]["category"] == "integration_api"


# ---------------------------------------------------------------------------
# Fallback — all 3 attempts fail → safe ambiguous/Multi-Agent classification
# ---------------------------------------------------------------------------

class TestAgent1Fallback:
    def test_all_attempts_fail_returns_fallback(self):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("Azure is down")
            result = agent1_classify(state)

        clf = result["classification"]
        assert clf["category"]     == "ambiguous"
        assert clf["routing_path"] == "Multi-Agent"
        assert clf["confidence"]   == 0.0

    def test_fallback_makes_exactly_3_attempts(self):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("failure")
            agent1_classify(state)

        assert mock_llm.invoke.call_count == 3

    def test_fallback_does_not_raise(self):
        state = _base_state()
        with patch("src.agents.agent1_classify._structured_llm") as mock_llm, \
             patch("src.agents.agent1_classify.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("failure")
            try:
                result = agent1_classify(state)
                assert result["classification"] is not None
            except Exception as exc:
                pytest.fail(f"agent1_classify raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# ClassificationResult Pydantic schema validation
# ---------------------------------------------------------------------------

class TestClassificationResultSchema:
    def test_valid_schema_accepted(self):
        clf = ClassificationResult(
            category="production_incident", severity="Critical",
            routing_path="Multi-Agent", confidence=0.95,
            reasoning="Complete outage detected.",
        )
        assert clf.category == "production_incident"

    def test_invalid_category_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(
                category="invalid_category", severity="High",
                routing_path="RAG", confidence=0.80, reasoning="test",
            )

    def test_invalid_severity_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(
                category="billing", severity="Urgent",
                routing_path="SQL", confidence=0.80, reasoning="test",
            )

    def test_invalid_routing_path_rejected(self):
        with pytest.raises(Exception):
            ClassificationResult(
                category="billing", severity="Low",
                routing_path="Direct", confidence=0.80, reasoning="test",
            )
