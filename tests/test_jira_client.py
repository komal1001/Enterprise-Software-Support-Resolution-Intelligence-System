"""
Unit tests for src/integrations/jira_client.py.

HTTP calls mocked via urllib.request — no real Jira calls made.
Tests cover: _build_description(), create_jira_issue(), auth header, HTTP errors.
"""

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.integrations.jira_client import (
    _auth_header,
    _build_description,
    create_jira_issue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def jira_env(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL",         "https://test.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL",            "test@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN",        "test-token-abc123")
    monkeypatch.setenv("JIRA_SERVICE_DESK_ID",  "2")
    monkeypatch.setenv("JIRA_REQUEST_TYPE_ID",  "16")
    monkeypatch.setenv("JIRA_PROJECT_KEY",      "SUP")


def _pkg(extra=None):
    base = {
        "reference_id":        "ESC-ABCD1234",
        "severity":            "Critical",
        "priority":            "P1",
        "category":            "production_incident",
        "team":                "SRE Team",
        "escalation_trigger":  "critical_severity",
        "context_summary":     "Service is completely down.",
        "recommended_actions": ["Restart pods", "Check DB connections"],
        "ticket_history":      [{"role": "user", "content": "Everything is down!"}],
        "account_data":        [{"customer_id": 1, "company_name": "Acme"}],
        "incident_logs":       [{"incident_type": "outage", "severity_level": "Critical",
                                 "status": "Open", "created_at": "2026-07-01T10:00:00"}],
        "agent_reasoning":     {"severity_reasoning": "Multiple Critical signals detected."},
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# _auth_header
# ---------------------------------------------------------------------------

class TestAuthHeader:
    def test_produces_basic_auth(self):
        header = _auth_header()
        assert header.startswith("Basic ")

    def test_encodes_email_and_token(self):
        header = _auth_header()
        encoded = header.split(" ", 1)[1]
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "test@example.com:test-token-abc123"


# ---------------------------------------------------------------------------
# _build_description — pure function
# ---------------------------------------------------------------------------

class TestBuildDescription:
    def test_contains_reference_id(self):
        desc = _build_description(_pkg())
        assert "ESC-ABCD1234" in desc

    def test_contains_severity_and_priority(self):
        desc = _build_description(_pkg())
        assert "Critical" in desc
        assert "P1" in desc

    def test_contains_team(self):
        desc = _build_description(_pkg())
        assert "SRE Team" in desc

    def test_contains_context_summary(self):
        desc = _build_description(_pkg())
        assert "Service is completely down." in desc

    def test_contains_recommended_actions(self):
        desc = _build_description(_pkg())
        assert "Restart pods" in desc
        assert "Check DB connections" in desc

    def test_contains_original_ticket_from_history(self):
        desc = _build_description(_pkg())
        assert "Everything is down!" in desc

    def test_contains_account_data(self):
        desc = _build_description(_pkg())
        assert "ACCOUNT DATA" in desc

    def test_contains_incident_log(self):
        desc = _build_description(_pkg())
        assert "RECENT INCIDENTS" in desc
        assert "outage" in desc

    def test_contains_agent_reasoning(self):
        desc = _build_description(_pkg())
        assert "Multiple Critical signals detected." in desc

    def test_no_ticket_history_skips_original_ticket(self):
        pkg = _pkg({"ticket_history": []})
        desc = _build_description(pkg)
        assert "ORIGINAL TICKET" not in desc

    def test_no_account_data_skips_section(self):
        pkg = _pkg({"account_data": []})
        desc = _build_description(pkg)
        assert "ACCOUNT DATA" not in desc

    def test_no_incidents_skips_section(self):
        pkg = _pkg({"incident_logs": []})
        desc = _build_description(pkg)
        assert "RECENT INCIDENTS" not in desc

    def test_category_title_cased(self):
        desc = _build_description(_pkg())
        assert "Production Incident" in desc


# ---------------------------------------------------------------------------
# create_jira_issue
# ---------------------------------------------------------------------------

class TestCreateJiraIssue:
    def _mock_http(self, issue_key="SUP-5"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"issueKey": issue_key}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_issue_key_on_success(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http("SUP-5")):
            result = create_jira_issue(_pkg())
        assert result == "SUP-5"

    def test_sends_correct_service_desk_id(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http()) as mock_urlopen:
            create_jira_issue(_pkg())
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["serviceDeskId"] == "2"

    def test_sends_correct_request_type_id(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http()) as mock_urlopen:
            create_jira_issue(_pkg())
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["requestTypeId"] == "16"

    def test_summary_contains_severity_and_category(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http()) as mock_urlopen:
            create_jira_issue(_pkg())
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode())
        summary = body["requestFieldValues"]["summary"]
        assert "Critical" in summary
        assert "Production Incident" in summary

    def test_missing_base_url_returns_none(self, monkeypatch):
        monkeypatch.setenv("JIRA_BASE_URL", "")
        result = create_jira_issue(_pkg())
        assert result is None

    def test_http_error_raises_runtime_error(self):
        import urllib.error
        mock_err = urllib.error.HTTPError(
            url="https://test.atlassian.net/rest/servicedeskapi/request",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b"Unauthorized"),
        )
        with patch("urllib.request.urlopen", side_effect=mock_err):
            with pytest.raises(RuntimeError, match="Jira API error 401"):
                create_jira_issue(_pkg())

    def test_request_uses_post_method(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http()) as mock_urlopen:
            create_jira_issue(_pkg())
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"

    def test_request_has_experimental_api_header(self):
        with patch("urllib.request.urlopen", return_value=self._mock_http()) as mock_urlopen:
            create_jira_issue(_pkg())
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-experimentalapi") == "opt-in"
