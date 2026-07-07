const ROUTE_SEQUENCES = {
  RAG:           ['agent1_classify', 'agent2_rag',  'agent4_severity', 'response_synthesizer'],
  SQL:           ['agent1_classify', 'agent3_sql',  'agent4_severity', 'response_synthesizer'],
  Hybrid:        ['agent1_classify', 'agent3_sql',  'agent2_rag', 'agent4_severity', 'response_synthesizer'],
  'Multi-Agent': ['agent1_classify', 'agent3_sql',  'agent2_rag', 'agent4_severity', 'agent5_escalation'],
};

const NODE_META = {
  agent1_classify:      { label: 'Classifier',      sublabel: 'Intent & routing decision' },
  agent2_rag:           { label: 'Knowledge Agent', sublabel: 'Semantic doc retrieval' },
  agent3_sql:           { label: 'Data Agent',      sublabel: 'Account & ticket lookup' },
  agent4_severity:      { label: 'Severity Agent',  sublabel: 'Risk re-assessment' },
  agent5_escalation:    { label: 'Escalation Agent',sublabel: 'Human handoff package' },
  response_synthesizer: { label: 'Synthesizer',     sublabel: 'Response generation' },
};

const ROUTE_COLORS = {
  RAG: '#2563eb', SQL: '#7c3aed', Hybrid: '#0891b2', 'Multi-Agent': '#dc2626',
};

function nodeDetail(step, eventData) {
  if (!eventData) return null;
  switch (step) {
    case 'agent1_classify': {
      const clf = eventData.classification || {};
      if (!clf.category) return null;
      return `${clf.category?.replace(/_/g, ' ')} · ${clf.severity} · ${clf.routing_path}`;
    }
    case 'agent3_sql':
      if (eventData.sql_error) return 'Query error';
      return eventData.row_count != null ? `${eventData.row_count} row${eventData.row_count !== 1 ? 's' : ''} retrieved` : null;
    case 'agent2_rag':
      return eventData.chunk_count != null ? `${eventData.chunk_count} chunks retrieved` : null;
    case 'agent4_severity':
      if (eventData.severity == null) return null;
      return eventData.escalate ? `${eventData.severity} — escalating` : `${eventData.severity} — no escalation`;
    case 'agent5_escalation':
      return 'Handoff package assembled';
    case 'response_synthesizer':
      return 'Streaming response...';
    default:
      return null;
  }
}

function StepNode({ step, status, eventData, isLast, routeColor }) {
  const meta   = NODE_META[step] || { label: step, sublabel: '' };
  const detail = nodeDetail(step, eventData);

  const dotColor =
    status === 'done'    ? '#16a34a' :
    status === 'active'  ? routeColor || '#2563eb' :
    '#cbd5e1';

  const labelColor =
    status === 'done'   ? '#16a34a' :
    status === 'active' ? routeColor || '#2563eb' :
    '#94a3b8';

  return (
    <div style={{ display: 'flex', gap: 12 }}>
      {/* Dot + connecting line */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <div style={{
          width: 14, height: 14, borderRadius: '50%',
          background: dotColor,
          boxShadow: status === 'active' ? `0 0 0 3px ${dotColor}33` : 'none',
          animation: status === 'active' ? 'pulse-ring 1.4s ease-out infinite' : 'none',
          flexShrink: 0,
          transition: 'background 0.3s ease, box-shadow 0.3s ease',
        }} />
        {!isLast && (
          <div style={{
            width: 2, flexGrow: 1, minHeight: 20,
            background: status === 'done' ? '#16a34a44' : '#e2e8f0',
            margin: '3px 0',
          }} />
        )}
      </div>

      {/* Content */}
      <div style={{ paddingBottom: isLast ? 0 : 14, paddingTop: 0, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: labelColor, letterSpacing: 0.3 }}>
            {meta.label}
          </span>
          <span style={{ fontSize: 11, color: '#94a3b8' }}>{meta.sublabel}</span>
          {status === 'done' && (
            <span style={{ fontSize: 11, color: '#16a34a', marginLeft: 'auto', flexShrink: 0 }}>✓</span>
          )}
          {status === 'active' && (
            <span style={{ fontSize: 11, color: labelColor, marginLeft: 'auto', flexShrink: 0, fontStyle: 'italic' }}>
              running...
            </span>
          )}
        </div>
        {detail && (
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{detail}</div>
        )}
      </div>
    </div>
  );
}

/**
 * PipelineFlow — shows the agent execution path for a ticket.
 *
 * Props:
 *   steps        — array of status event objects received so far
 *   done         — bool: true when the done SSE event has fired
 *   compact      — bool: compact mode for AnalysisPanel (no header, smaller padding)
 */
export default function PipelineFlow({ steps = [], done = false, compact = false }) {
  // Determine routing path from agent1 classification
  const clf1       = steps.find(s => s.step === 'agent1_classify');
  const routePath  = clf1?.classification?.routing_path || null;
  const routeColor = ROUTE_COLORS[routePath] || '#2563eb';

  // Which sequence of nodes to show
  const sequence = routePath ? ROUTE_SEQUENCES[routePath] : ['agent1_classify'];

  // Map step name → event data received
  const stepData = {};
  steps.forEach(s => { stepData[s.step] = s; });

  const receivedSteps = new Set(steps.map(s => s.step));
  // The active node is the first node in sequence that hasn't been received yet
  const activeNode = done ? null : sequence.find(n => !receivedSteps.has(n)) || null;

  function statusOf(nodeStep) {
    if (nodeStep === activeNode) return 'active';
    if (receivedSteps.has(nodeStep)) return 'done';
    return 'pending';
  }

  // Before Agent 1 fires, show agent1 as active
  if (steps.length === 0 && !done) {
    return (
      <div style={{ padding: compact ? '6px 0' : '10px 14px' }}>
        <StepNode
          step="agent1_classify"
          status="active"
          eventData={null}
          isLast={true}
          routeColor="#2563eb"
        />
      </div>
    );
  }

  return (
    <div style={{ padding: compact ? '4px 0' : '10px 14px' }}>
      {!compact && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#64748b' }}>
            Agent Pipeline
          </span>
          {routePath && (
            <span style={{
              fontSize: 10, fontWeight: 700, padding: '2px 8px',
              borderRadius: 10, background: routeColor + '1a', color: routeColor,
              textTransform: 'uppercase', letterSpacing: 0.5,
            }}>
              {routePath}
            </span>
          )}
        </div>
      )}

      <div>
        {sequence.map((nodeStep, i) => (
          <StepNode
            key={nodeStep}
            step={nodeStep}
            status={statusOf(nodeStep)}
            eventData={stepData[nodeStep]}
            isLast={i === sequence.length - 1}
            routeColor={routeColor}
          />
        ))}
      </div>
    </div>
  );
}
