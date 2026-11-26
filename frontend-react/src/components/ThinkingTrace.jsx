// Live reasoning trace: each agent's stated rationale and key intermediate
// outputs (generated SQL, rewritten search query, retrieved passages, severity
// decision), streamed in the SSE status events as each agent finishes.

const AGENTS = {
  agent1_classify:   { icon: '🧠', name: 'Classifier' },
  agent3_sql:        { icon: '🗄️', name: 'Data Agent' },
  agent2_rag:        { icon: '📚', name: 'Knowledge Agent' },
  agent4_severity:   { icon: '⚖️', name: 'Severity Agent' },
  agent5_escalation: { icon: '🚨', name: 'Escalation Agent' },
};

const pretty = s => (s || '').replace(/_/g, ' ');
const docName = f => (f || '').replace(/_/g, ' ').replace(/\.pdf$/i, '');

function Classifier({ ev }) {
  const c = ev.classification || {};
  return (
    <>
      <div className="tt-decision">
        <span className="tt-tag">{pretty(c.category)}</span>
        <span className="tt-tag">{c.severity}</span>
        <span className="tt-tag tt-tag-strong">→ {c.routing_path} route</span>
        {c.confidence != null && <span className="tt-tag">{Math.round(c.confidence * 100)}% confident</span>}
      </div>
      {ev.thinking?.reasoning && <p className="tt-thought">“{ev.thinking.reasoning}”</p>}
    </>
  );
}

function DataAgent({ ev }) {
  const t = ev.thinking || {};
  return (
    <>
      {t.intent && <p className="tt-thought">Goal: {t.intent}</p>}
      {t.sql && (
        <>
          <div className="tt-sub">
            Generated SQL {!t.error && <span className="tt-ok">✓ passed validation: SELECT-only, allow-listed tables</span>}
          </div>
          <pre className="tt-code">{t.sql}</pre>
        </>
      )}
      <div className="tt-result">
        {t.error ? `⚠️ Query error: ${t.error}` : `${t.row_count ?? 0} row${t.row_count === 1 ? '' : 's'} returned`}
      </div>
    </>
  );
}

function KnowledgeAgent({ ev }) {
  const t = ev.thinking || {};
  return (
    <>
      {t.query && <p className="tt-thought">Search query: “{t.query}”</p>}
      {(t.top || []).length > 0 && (
        <ol className="tt-passages">
          {t.top.map((p, i) => (
            <li key={i}>
              <span className="tt-src">{docName(p.source)}{p.page ? ` · p.${p.page}` : ''}</span>
              <span className="tt-score">score {p.score}</span>
              <div className="tt-snippet">{p.snippet}…</div>
            </li>
          ))}
        </ol>
      )}
    </>
  );
}

function SeverityAgent({ ev }) {
  const t = ev.thinking || {};
  const changed = t.initial_severity && ev.severity && t.initial_severity !== ev.severity;
  return (
    <>
      <div className="tt-decision">
        <span className="tt-tag">
          {changed ? `${t.initial_severity} → ${ev.severity}` : `${ev.severity} (confirmed)`}
        </span>
        <span className={`tt-tag ${ev.escalate ? 'tt-tag-alert' : ''}`}>
          {ev.escalate ? 'Escalate to humans' : 'No escalation'}
        </span>
        {t.reretrieve && <span className="tt-tag">Requested more evidence</span>}
      </div>
      {t.reasoning && <p className="tt-thought">“{t.reasoning}”</p>}
      {t.trigger && <div className="tt-result">Trigger: {pretty(t.trigger)}</div>}
    </>
  );
}

function EscalationAgent({ ev }) {
  const t = ev.thinking || {};
  if (!t.team) return <p className="tt-thought">Assembling the handoff package…</p>;
  return (
    <>
      <div className="tt-decision">
        <span className="tt-tag tt-tag-alert">{t.team}</span>
        {t.priority && <span className="tt-tag">{t.priority}</span>}
        {t.reference_id && <span className="tt-tag">{t.reference_id}</span>}
      </div>
      {(t.actions || []).length > 0 && (
        <>
          <div className="tt-sub">Recommended next steps for the engineer</div>
          <ul className="tt-actions">{t.actions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </>
      )}
    </>
  );
}

const BODY = {
  agent1_classify: Classifier,
  agent3_sql: DataAgent,
  agent2_rag: KnowledgeAgent,
  agent4_severity: SeverityAgent,
  agent5_escalation: EscalationAgent,
};

export default function ThinkingTrace({ steps = [], live = false }) {
  // Keep the latest event per agent but preserve first-seen order; a repeated
  // step (e.g. re-retrieval) updates in place.
  const order = [];
  const latest = {};
  for (const s of steps) {
    if (!AGENTS[s.step]) continue;
    if (!(s.step in latest)) order.push(s.step);
    latest[s.step] = s;
  }
  if (order.length === 0) return null;

  return (
    <div className="thinking-trace">
      {order.map(step => {
        const Body = BODY[step];
        const ev = latest[step];
        return (
          <div key={step} className="tt-step">
            <div className="tt-head">
              <span className="tt-icon" aria-hidden="true">{AGENTS[step].icon}</span>
              <span className="tt-name">{AGENTS[step].name}</span>
            </div>
            <div className="tt-body"><Body ev={ev} /></div>
          </div>
        );
      })}
      {live && <div className="tt-step tt-pending"><span className="dot" /><span className="dot" /><span className="dot" /></div>}
    </div>
  );
}
