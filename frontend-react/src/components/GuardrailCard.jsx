const REASONS = {
  prompt_injection:    { title: 'Prompt injection detected',    what: 'The message tried to override the system instructions.' },
  policy_override:     { title: 'Policy override attempt',      what: 'The message tried to bypass SLA or escalation policy.' },
  unauthorized_action: { title: 'Unauthorized action',          what: 'The message requested an action that needs documented authorization.' },
  pii_detected:        { title: 'Sensitive data detected',      what: 'The message contained credentials or card data, which must never enter the LLM.' },
};

export default function GuardrailCard({ reason, message }) {
  const r = REASONS[reason] || { title: 'Request blocked', what: '' };
  return (
    <div className="guardrail-card" role="alert">
      <div className="guardrail-head">
        <span className="guardrail-shield" aria-hidden="true">🛡️</span>
        <div>
          <div className="guardrail-title">Blocked by input guardrails — {r.title}</div>
          {r.what && <div className="guardrail-what">{r.what}</div>}
        </div>
      </div>
      <div className="guardrail-facts">
        <span>Checked <strong>before</strong> any LLM call · 0 tokens spent</span>
        <span>Logged to Langfuse for audit</span>
        <span>SLO #13 Guardrail Effectiveness — target: 100% of adversarial inputs blocked</span>
      </div>
      {message && <div className="guardrail-message">Message shown to the user: “{message}”</div>}
    </div>
  );
}
