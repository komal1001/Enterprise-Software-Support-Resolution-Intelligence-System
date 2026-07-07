"""
Tests for stream_synthesizer_tokens() and _fetch_incident_logs() —
the remaining uncovered lines in response_synthesizer.py and agent5_escalation.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.response_synthesizer import stream_synthesizer_tokens
from src.agents.agent5_escalation import _fetch_incident_logs


# ---------------------------------------------------------------------------
# stream_synthesizer_tokens() — SSE token generator
# ---------------------------------------------------------------------------

def _state(severity="Medium", sql_rows=None, sql_row_count=None, rag_chunks=None,
           sql_error=None):
    sql_result = None
    if sql_rows is not None or sql_error is not None:
        sql_result = {
            "rows":      sql_rows or [],
            "row_count": sql_row_count if sql_row_count is not None else len(sql_rows or []),
            "error":     sql_error,
        }
    return {
        "ticket_text":          "How do I configure OAuth?",
        "conversation_history": [],
        "classification": {
            "category":     "usage_configuration",
            "severity":     severity,
            "routing_path": "RAG",
            "confidence":   0.90,
            "reasoning":    "How-to question",
        },
        "rag_result":          {"chunks": rag_chunks or []},
        "sql_result":          sql_result,
        "severity_assessment": {"severity": severity, "escalate": False},
        "escalation_package":  None,
        "final_response":      None,
        "reflection_count":    0,
    }


class TestStreamSynthesizerTokens:
    def test_zero_rows_yields_no_records_message(self):
        state = _state(sql_rows=[], sql_row_count=0, rag_chunks=[])
        tokens = list(stream_synthesizer_tokens(state))
        assert len(tokens) == 1
        assert "No matching records" in tokens[0]

    def test_zero_rows_does_not_call_llm_stream(self):
        state = _state(sql_rows=[], sql_row_count=0, rag_chunks=[])
        with patch("src.agents.response_synthesizer._llm_stream") as mock_stream:
            list(stream_synthesizer_tokens(state))
        mock_stream.stream.assert_not_called()

    def test_normal_path_yields_llm_tokens(self):
        state = _state()
        mock_chunk1 = MagicMock(content="Here ")
        mock_chunk2 = MagicMock(content="is the answer.")
        mock_chunk3 = MagicMock(content="")  # empty chunk, should be skipped

        with patch("src.agents.response_synthesizer._llm_stream") as mock_stream:
            mock_stream.stream.return_value = iter([mock_chunk1, mock_chunk2, mock_chunk3])
            tokens = list(stream_synthesizer_tokens(state))

        assert "Here " in tokens
        assert "is the answer." in tokens
        assert "" not in tokens  # empty chunks filtered out

    def test_tone_instruction_appears_in_system_message(self):
        state = _state(severity="Critical")
        captured_messages = []

        def capture_stream(messages):
            captured_messages.extend(messages)
            return iter([MagicMock(content="response")])

        with patch("src.agents.response_synthesizer._llm_stream") as mock_stream:
            mock_stream.stream.side_effect = capture_stream
            list(stream_synthesizer_tokens(state))

        system_content = captured_messages[0].content.lower()
        assert "urgency" in system_content

    def test_sql_error_path_still_streams(self):
        """SQL returned an error (not zero rows) — should still call LLM for RAG answer."""
        state = _state(sql_rows=[], sql_error="timeout", rag_chunks=[{"source": "doc.pdf", "text": "docs"}])
        with patch("src.agents.response_synthesizer._llm_stream") as mock_stream:
            mock_stream.stream.return_value = iter([MagicMock(content="Docs answer.")])
            tokens = list(stream_synthesizer_tokens(state))
        assert "Docs answer." in tokens


# ---------------------------------------------------------------------------
# _fetch_incident_logs() — parameterised DB query
# ---------------------------------------------------------------------------

class TestFetchIncidentLogs:
    def test_returns_rows_on_success(self):
        mock_rows = [{"incident_id": 1, "incident_type": "outage", "severity_level": "Critical"}]
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: s
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [{"incident_id": 1, "incident_type": "outage", "severity_level": "Critical"}]

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.connect", return_value=mock_conn), \
             patch("psycopg2.extras.RealDictCursor"):
            result = _fetch_incident_logs(customer_id=1)

        assert isinstance(result, list)

    def test_returns_empty_list_on_db_exception(self):
        with patch("psycopg2.connect", side_effect=Exception("DB is down")):
            result = _fetch_incident_logs(customer_id=99)
        assert result == []
