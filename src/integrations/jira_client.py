"""
Jira Service Management integration — src/integrations/jira_client.py

Called by Agent 5 after building the escalation package.
Creates a JSM service request so the human team receives the full handoff context.

Uses the JSM Customer REST API (/rest/servicedeskapi/request) — not the standard
Jira issue API (/rest/api/3/issue) which doesn't work for JSM projects.

Credentials read from environment:
  JIRA_BASE_URL        — https://yoursite.atlassian.net
  JIRA_EMAIL           — Atlassian account email
  JIRA_API_TOKEN       — Classic API token (Basic Auth)
  JIRA_PROJECT_KEY     — project key (e.g. SUP)
  JIRA_SERVICE_DESK_ID — numeric service desk id (default: 2)
  JIRA_REQUEST_TYPE_ID — request type id for "Report a system problem" (default: 16)
"""

import os
import base64
import json
import urllib.request
import urllib.error
from typing import Optional


def _auth_header() -> str:
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    encoded = base64.b64encode(f"{email}:{token}".encode()).decode()
    return f"Basic {encoded}"


def _post(url: str, payload: dict) -> dict:
    data    = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": _auth_header(),
        "Content-Type":  "application/json",
        "Accept":        "application/json",
        "X-ExperimentalApi": "opt-in",   # required for JSM customer API
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_description(pkg: dict) -> str:
    """Plain text description for the JSM request body."""
    lines = []

    lines.append(f"REFERENCE: {pkg.get('reference_id', 'N/A')}")
    lines.append(f"SEVERITY:  {pkg.get('severity', 'N/A')} ({pkg.get('priority', 'N/A')})")
    lines.append(f"CATEGORY:  {pkg.get('category', 'N/A').replace('_', ' ').title()}")
    lines.append(f"TEAM:      {pkg.get('team', 'N/A')}")
    lines.append(f"TRIGGER:   {pkg.get('escalation_trigger', 'N/A')}")
    lines.append("")

    # Original ticket text
    history = pkg.get("ticket_history", [])
    for msg in history:
        if msg.get("role") == "user":
            lines.append("ORIGINAL TICKET:")
            lines.append(msg.get("content", ""))
            lines.append("")
            break

    # Context summary
    lines.append("CONTEXT SUMMARY:")
    lines.append(pkg.get("context_summary", ""))
    lines.append("")

    # Recommended actions
    actions = pkg.get("recommended_actions", [])
    if actions:
        lines.append("RECOMMENDED ACTIONS:")
        for a in actions:
            lines.append(f"  - {a}")
        lines.append("")

    # Account data
    sql_rows = pkg.get("account_data", [])
    if sql_rows:
        lines.append("ACCOUNT DATA:")
        for row in sql_rows[:3]:
            lines.append(f"  {row}")
        lines.append("")

    # Recent incidents
    incidents = pkg.get("incident_logs", [])
    if incidents:
        lines.append("RECENT INCIDENTS:")
        for inc in incidents[:3]:
            lines.append(
                f"  {inc.get('incident_type','?')} | {inc.get('severity_level','?')} | "
                f"{inc.get('status','?')} | {str(inc.get('created_at',''))[:10]}"
            )
        lines.append("")

    # Documentation context (RAG chunks)
    doc_context = pkg.get("documentation_context", [])
    if doc_context:
        lines.append("RELEVANT DOCUMENTATION:")
        for doc in doc_context:
            source = doc.get("source", "unknown")
            text   = doc.get("text", "")[:300]
            lines.append(f"  [{source}]")
            lines.append(f"  {text}")
            lines.append("")

    # Agent reasoning
    reasoning = pkg.get("agent_reasoning", {})
    if reasoning.get("severity_reasoning"):
        lines.append("AGENT 4 SEVERITY REASONING:")
        lines.append(reasoning["severity_reasoning"])

    return "\n".join(lines)


def create_jira_issue(escalation_package: dict) -> Optional[str]:
    """
    Create a JSM service request from the Agent 5 escalation package.
    Returns the issue key (e.g. 'SUP-1') on success, None on failure.
    """
    base_url         = os.environ.get("JIRA_BASE_URL", "").rstrip("/")
    service_desk_id  = os.environ.get("JIRA_SERVICE_DESK_ID", "2")
    request_type_id  = os.environ.get("JIRA_REQUEST_TYPE_ID", "16")

    if not base_url:
        return None

    category = escalation_package.get("category", "general")
    severity = escalation_package.get("severity", "High")
    ref_id   = escalation_package.get("reference_id", "")
    team     = escalation_package.get("team", "Support")
    priority = escalation_package.get("priority", "P2")

    summary = (
        f"[{severity}/{priority}] {category.replace('_', ' ').title()} escalation "
        f"— {ref_id} → {team}"
    )

    payload = {
        "serviceDeskId": service_desk_id,
        "requestTypeId": request_type_id,
        "requestFieldValues": {
            "summary":     summary,
            "description": _build_description(escalation_package),
        },
    }

    url = f"{base_url}/rest/servicedeskapi/request"
    try:
        response = _post(url, payload)
        return response.get("issueKey")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Jira API error {e.code}: {body}") from e
