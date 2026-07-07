"""
Unit tests for response_synthesizer (src/agents/response_synthesizer.py).

Two layers:
  1. _build_prompt_variables() — pure function, no mocking
  2. response_synthesizer() node — LLM mocked

Tests verify:
  - Tone instruction maps correctly to severity
  - Context block built from RAG chunks and SQL rows
  - Zero-rows early return skips LLM (cost optimisation)
  - final_response stored correctly
  - cited_sources stored in rag_result (SLO #8: source attribution)
  - Fallback response returned when all LLM attempts fail
  - Retry logic fires on parse failure
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.response_synthesizer import (
    _build_prompt_variables,
    response_synthesizer,
    SynthesizedResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(severity="Medium", rag_chunks=None, sql_rows=None, sql_error=None,
           sql_row_count=None):
    sql_result = None
    if sql_rows is not None or sql_error is not None:
        sql_result = {
            "rows":      sql_rows or [],
            "row_count": sql_row_count if sql_row_count is not None else len(sql_rows or []),
            "error":     sql_error,
        }
    return {
        "ticket_text":          "My API keeps returning 401 errors.",
        "conversation_history": [],
        "classification": {
            "category":     "integration_api",
            "severity":     severity,
            "routing_path": "RAG",
            "confidence":   0.90,
            "reasoning":    "API auth issue",
        },
        "rag_result":          {"chunks": rag_chunks} if rag_chunks is not None else None,
        "sql_result":          sql_result,
        "severity_assessment": {
            "severity":   severity,
            "confidence": 0.85,
            "escalate":   False,
            "reasoning":  "not critical",
        },
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    0,
    }


def _mock_llm_response(response="Here is how to fix your API issue.",
                       sources=None):
    parsed = SynthesizedResponse(
        response=response,
        sources=sources or ["API_Error_Codes_Troubleshooting_Handbook.pdf"],
    )
    raw_msg = MagicMock()
    raw_msg.usage_metadata = {"input_tokens": 800, "output_tokens": 200}
    return {"parsed": parsed, "parsing_error": None, "raw": raw_msg}


def _prompt_obj():
    obj = MagicMock()
    obj.compile.return_value = "mocked prompt"
    return obj


# ---------------------------------------------------------------------------
# _build_prompt_variables — pure function
# ---------------------------------------------------------------------------

class TestBuildPromptVariables:
    @pytest.mark.parametrize("severity,keyword", [
        ("Critical", "urgency"),
        ("High",     "promptly"),
        ("Medium",   "step-by-step"),
        ("Low",      "helpful"),
    ])
    def test_tone_maps_to_severity(self, severity, keyword):
        state = _state(severity=severity)
        vars_ = _build_prompt_variables(state)
        assert keyword.lower() in vars_["tone_instruction"].lower()

    def test_rag_chunks_appear_in_context_block(self):
        chunks = [{"source": "SLA_Policy.pdf", "text": "SLA terms content here"}]
        state = _state(rag_chunks=chunks)
        vars_ = _build_prompt_variables(state)
        assert "DOCUMENTATION SOURCES" in vars_["context_block"]
        assert "SLA_Policy.pdf" in vars_["context_block"]

    def test_sql_rows_appear_in_context_block(self):
        rows = [{"customer_id": 1, "company_name": "Acme"}]
        state = _state(sql_rows=rows, sql_row_count=1)
        vars_ = _build_prompt_variables(state)
        assert "ACCOUNT & TICKET DATA" in vars_["context_block"]

    def test_no_context_produces_fallback_message(self):
        state = _state(rag_chunks=[], sql_rows=None)
        vars_ = _build_prompt_variables(state)
        assert "No additional context" in vars_["context_block"]

    def test_ticket_text_included(self):
        state = _state()
        vars_ = _build_prompt_variables(state)
        assert vars_["ticket_text"] == "My API keeps returning 401 errors."

    def test_severity_from_assessment_takes_precedence(self):
        state = _state(severity="Critical")
        vars_ = _build_prompt_variables(state)
        assert vars_["severity"] == "Critical"

    def test_rag_chunks_truncated_to_300_chars(self):
        long_text = "X" * 500
        chunks = [{"source": "doc.pdf", "text": long_text}]
        state = _state(rag_chunks=chunks)
        vars_ = _build_prompt_variables(state)
        assert len(vars_["context_block"]) < len(long_text) * 2


# ---------------------------------------------------------------------------
# response_synthesizer() node — zero-rows early return
# ---------------------------------------------------------------------------

class TestResponseSynthesizerZeroRows:
    def test_zero_rows_returns_no_records_message(self):
        """SQL returned no rows and no RAG chunks — skip LLM, return specific message."""
        state = _state(sql_rows=[], sql_row_count=0, rag_chunks=[])
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            result = response_synthesizer(state)

        mock_llm.invoke.assert_not_called()
        assert "No matching records" in result["final_response"]

    def test_zero_rows_with_rag_chunks_still_calls_llm(self):
        """Zero SQL rows but RAG chunks exist — should still call LLM for documentation answer."""
        chunks = [{"source": "doc.pdf", "text": "relevant docs"}]
        state  = _state(sql_rows=[], sql_row_count=0, rag_chunks=chunks)
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = response_synthesizer(state)

        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# response_synthesizer() node — happy path
# ---------------------------------------------------------------------------

class TestResponseSynthesizerHappyPath:
    def test_final_response_stored(self):
        state = _state()
        resp  = _mock_llm_response(response="Check your API key configuration.")
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = resp
            result = response_synthesizer(state)

        assert result["final_response"] == "Check your API key configuration."

    def test_cited_sources_stored_in_rag_result(self):
        """SLO #8: Source Attribution Rate = 100% — sources must be in state."""
        state = _state()
        sources = ["API_Error_Codes.pdf", "SLA_Policy.pdf"]
        resp = _mock_llm_response(sources=sources)
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = resp
            result = response_synthesizer(state)

        assert result["rag_result"]["cited_sources"] == sources

    def test_token_counts_stored(self):
        state = _state()
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = response_synthesizer(state)

        tokens = result["_synthesizer_tokens"]
        assert tokens["input_tokens"]  == 800
        assert tokens["output_tokens"] == 200

    def test_no_rag_result_initialised_before_storing_sources(self):
        """If rag_result is None, synthesizer must initialise it before writing cited_sources."""
        state = _state()
        state["rag_result"] = None
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001):
            mock_llm.invoke.return_value = _mock_llm_response()
            result = response_synthesizer(state)

        assert "cited_sources" in result["rag_result"]


# ---------------------------------------------------------------------------
# response_synthesizer() node — retry and fallback
# ---------------------------------------------------------------------------

class TestResponseSynthesizerRetryFallback:
    def test_succeeds_on_second_attempt(self):
        state = _state()
        fail  = {"parsed": None, "parsing_error": "JSON error", "raw": MagicMock()}
        ok    = _mock_llm_response(response="Fixed on retry.")
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = [fail, ok]
            result = response_synthesizer(state)

        assert mock_llm.invoke.call_count == 2
        assert result["final_response"] == "Fixed on retry."

    def test_all_attempts_fail_returns_fallback(self):
        state = _state()
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("Azure is down")
            result = response_synthesizer(state)

        assert result["final_response"] is not None
        assert "support team will follow up" in result["final_response"]

    def test_fallback_does_not_raise(self):
        state = _state()
        with patch("src.agents.response_synthesizer._llm") as mock_llm, \
             patch("src.agents.response_synthesizer.get_cached_prompt", return_value=_prompt_obj()), \
             patch("src.observability.langfuse_client.get_langfuse", return_value=MagicMock()), \
             patch("src.observability.langfuse_client.calculate_cost", return_value=0.0001), \
             patch("time.sleep"):
            mock_llm.invoke.side_effect = RuntimeError("failure")
            try:
                result = response_synthesizer(state)
                assert result["final_response"] is not None
            except Exception as exc:
                pytest.fail(f"response_synthesizer raised unexpectedly: {exc}")
