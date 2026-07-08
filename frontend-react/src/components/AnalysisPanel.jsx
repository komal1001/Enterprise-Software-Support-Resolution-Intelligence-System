import PipelineFlow from './PipelineFlow';

const SEVERITY_COLORS = {
  Critical: '#dc2626', High: '#ea580c', Medium: '#ca8a04', Low: '#16a34a',
};
const ROUTE_COLORS = {
  RAG: '#2563eb', SQL: '#7c3aed', Hybrid: '#0891b2', 'Multi-Agent': '#dc2626',
};
const CATEGORY_LABELS = {
  production_incident:  'Production Incident',
  security:             'Security',
  integration_api:      'Integration / API',
  performance_latency:  'Performance / Latency',
  usage_configuration:  'Usage / Configuration',
  billing:              'Billing',
  ambiguous:            'Ambiguous',
};

const SEVERITY_TOOLTIPS = {
  Critical: 'Complete outage, data loss, or security breach. Production is down. Immediate escalation required.',
  High:     'Major feature broken or many users affected. Significant business impact. Escalation likely.',
  Medium:   'Degraded performance or partial outage. Some users affected. Workaround may exist.',
  Low:      'Minor issue, single user affected. Workaround available. No production impact.',
};

const ROUTE_TOOLTIPS = {
  RAG:           'Documentation-only question. Knowledge Agent retrieves relevant docs.',
  SQL:           'Account or ticket data lookup. Data Agent queries the database.',
  Hybrid:        'Needs both account data (SQL) and documentation (RAG) to answer.',
  'Multi-Agent': 'Critical or high-risk ticket. All agents run + human escalation.',
};

function Badge({ label, color, tooltip }) {
  return (
    <span title={tooltip} style={{
      background: color, color: '#fff', padding: '3px 12px',
      borderRadius: 12, fontSize: 13, fontWeight: 600, marginRight: 6,
      cursor: tooltip ? 'help' : 'default',
    }}>{label}</span>
  );
}

function ScoreBar({ label, value, threshold, na }) {
  if (value == null) {
    if (!na) return null;
    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="meta-label">{label}</span>
        <span style={{ fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>{na}</span>
      </div>
    );
  }
  const pct    = Math.round(value * 100);
  const passes = value >= threshold;
  const color  = passes ? '#16a34a' : '#dc2626';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="meta-label">{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>
          {pct}% {passes ? '✓' : '✗ below SLO'}
        </span>
      </div>
      <div style={{ height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function AnalysisPanel({ response, role, agentSteps = [] }) {
  const clf        = response.classification || {};
  const sev4       = response.severity_assessment || {};
  const scores     = response.scores || null;
  const severity   = clf.severity || '';
  const route      = clf.routing_path || '';
  const category   = clf.category || '';
  const confidence = Math.round((clf.confidence || 0) * 100);
  const finalResp   = response.final_response || '';
  const isEscalated = finalResp.includes('escalated to our') || finalResp.startsWith('Your ticket has been escalated');
  const sources     = (response.sources || []).filter(Boolean);
  const submittedAt = (response.submitted_at || '').slice(0, 19).replace('T', ' ');
  const latencyMs   = response.latency_ms;
  const latencyMin  = latencyMs != null ? (latencyMs / 60000).toFixed(2) : null;
  const latencyOk   = latencyMs != null && latencyMs <= (route === 'Multi-Agent' ? 10000 : 5000);
  const escPkg      = response.escalation_package || {};
  const jiraKey     = escPkg.jira_issue_key;
  const jiraUrl     = jiraKey ? `https://supportintelligence.atlassian.net/browse/${jiraKey}` : null;

  return (
    <div className="analysis-panel">
      {isEscalated && (
        <div className="escalation-alert">
          🚨 Escalated — routed to human team for review
          {jiraKey && (role === 'manager' || role === 'admin') && (
            <span style={{ marginLeft: 12 }}>
              |{' '}
              <a href={jiraUrl} target="_blank" rel="noopener noreferrer"
                style={{ color: '#1d4ed8', fontWeight: 700 }}>
                {jiraKey}
              </a>
            </span>
          )}
        </div>
      )}

      <div className="badges">
        <Badge label={severity} color={SEVERITY_COLORS[severity] || '#6b7280'} tooltip={SEVERITY_TOOLTIPS[severity]} />
        <Badge label={route}    color={ROUTE_COLORS[route]    || '#6b7280'}    tooltip={ROUTE_TOOLTIPS[route]} />
      </div>

      <div className="meta-grid">
        <div>
          <span className="meta-label">Category</span>
          <span>{CATEGORY_LABELS[category] || category}</span>
        </div>
        <div>
          <span className="meta-label">Confidence</span>
          <span>{confidence}%</span>
        </div>
        {latencyMin != null && (
          <div>
            <span className="meta-label">Latency</span>
            <span style={{ color: latencyOk ? '#16a34a' : '#dc2626', fontWeight: 600 }}>
              {latencyMin} min {latencyOk ? '✓' : '⚠ SLO miss'}
            </span>
          </div>
        )}
      </div>

      {/* RAGAS scores — arrive ~5s after response via scores SSE event */}
      {scores ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '4px 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#6366f1', textTransform: 'uppercase', letterSpacing: 1 }}>
              RAGAS
            </span>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>statement-level evaluation</span>
          </div>
          <ScoreBar label="Faithfulness (SLO #2 ≥ 95%)"        value={scores.faithfulness}      threshold={0.95} na="N/A — SQL route" />
          <ScoreBar
            label={scores.relevance_type === 'task_completion'
              ? 'Task Completion (SLO #4 ≥ 85%)'
              : 'Answer Relevance (SLO #4 ≥ 85%)'}
            value={scores.relevance}
            threshold={0.85}
          />
          <ScoreBar label="Context Precision (SLO #9 ≥ 80%)"   value={scores.context_precision} threshold={0.80} na="N/A — SQL route" />
        </div>
      ) : (
        <p style={{ fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>
          RAGAS scores calculating...
        </p>
      )}

      {sources.length > 0 && (
        <div className="sources">
          <span className="meta-label">Sources</span>
          <ul>
            {sources.map((s, i) => (
              <li key={i}>
                <a
                  href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs/${s}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {s.replace(/_/g, ' ').replace('.pdf', '')}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      {agentSteps.length > 0 && (
        <details className="reasoning">
          <summary>Agent Pipeline — Execution Flow</summary>
          <PipelineFlow steps={agentSteps} done={true} compact={true} />
        </details>
      )}

      {clf.reasoning && (
        <details className="reasoning">
          <summary>Agent 1 — Classification Reasoning</summary>
          <p>{clf.reasoning}</p>
        </details>
      )}

      {sev4.reasoning && (
        <details className="reasoning">
          <summary>Agent 4 — Severity Re-assessment</summary>
          <p>{sev4.reasoning}</p>
          {sev4.escalation_trigger && (
            <p style={{ color: '#dc2626', fontSize: 12, padding: '0 12px 10px' }}>
              Escalation trigger: {sev4.escalation_trigger}
            </p>
          )}
        </details>
      )}

      {submittedAt && (
        <p className="timestamp">Submitted: {submittedAt} UTC</p>
      )}
    </div>
  );
}
