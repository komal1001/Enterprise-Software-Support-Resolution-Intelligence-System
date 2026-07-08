"""
Output guardrails — run on the synthesised response before it reaches the user.

Checks:
  1. PII redaction  — emails, phone numbers, credit-card numbers, SSNs
  2. Response sanity — catches empty or suspiciously short responses
"""

import re
from dataclasses import dataclass


@dataclass
class OutputGuardrailResult:
    clean:    bool   # False if PII was found and redacted
    reason:   str    # human-readable description of what was found
    response: str    # cleaned response (PII replaced with [REDACTED])


# ---------------------------------------------------------------------------
# PII patterns  (label, compiled regex)
# ---------------------------------------------------------------------------

_PII = [
    ("email",       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')),
    ("phone",       re.compile(r'\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b')),
    ("credit-card", re.compile(r'\b(?:\d[ \-]?){13,16}\b')),
    ("SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
]


def check_output(response: str) -> OutputGuardrailResult:
    """
    Scan the response for PII and redact in-place.
    Returns OutputGuardrailResult with clean=False if anything was redacted.
    Always returns a usable response string.
    """
    if not response or len(response.strip()) < 10:
        return OutputGuardrailResult(
            clean=False,
            reason="Response too short — replaced with fallback",
            response=(
                "We were unable to generate a complete response. "
                "Your ticket has been logged and our support team will follow up shortly."
            ),
        )

    cleaned = response
    triggered: list[str] = []

    for label, pattern in _PII:
        new, count = pattern.subn("[REDACTED]", cleaned)
        if count:
            cleaned = new
            triggered.append(f"{label} ×{count}")

    if triggered:
        return OutputGuardrailResult(
            clean=False,
            reason=f"PII redacted: {', '.join(triggered)}",
            response=cleaned,
        )

    return OutputGuardrailResult(clean=True, reason="", response=response)
