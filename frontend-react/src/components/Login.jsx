import { useAuth0 } from '@auth0/auth0-react';

const FLOW = [
  { icon: '📧', label: 'Support Ticket',      sub: 'User submits issue',               color: '#3b82f6' },
  { icon: '🧠', label: 'Agent 1 — Classify',  sub: 'Intent · Severity · Route',        color: '#8b5cf6' },
  { icon: '🔍', label: 'Retrieve Context',     sub: 'RAG · SQL · Hybrid',               color: '#06b6d4' },
  { icon: '⚖️', label: 'Agent 4 — Re-assess', sub: 'Severity validation on every ticket', color: '#f59e0b' },
  { icon: '📝', label: 'Response',             sub: 'Cited · Streamed · Logged',        color: '#10b981' },
];

export default function Login() {
  const { loginWithRedirect, isLoading } = useAuth0();

  return (
    <div className="login-container">

      {/* ── Left: sign-in panel ── */}
      <div className="login-left">
        <div className="login-badge">AI · Autonomous · Enterprise</div>
        <h1 className="login-title">Enterprise Support Intelligence</h1>
        <p className="login-sub">Autonomous AI triage built on LangGraph, Azure OpenAI, and pgvector. Classifies, retrieves, re-assesses, and responds — end to end.</p>
        <button className="login-btn" onClick={() => loginWithRedirect()} disabled={isLoading}>
          {isLoading ? 'Loading…' : 'Sign In to Portal →'}
        </button>
        <p className="login-hint">NIIT Capstone · AI Solutions Engineering</p>
      </div>

      {/* ── Right: 3-D flow diagram ── */}
      <div className="login-right">
        <div className="login-diagram">
          <p className="diagram-label">Ticket Resolution Pipeline</p>
          <div className="login-flow">
            {FLOW.map((node, i) => (
              <div key={node.label} className="flow-step">
                <div
                  className="flow-node"
                  style={{ '--accent': node.color, '--delay': `${i * 0.12}s` }}
                >
                  <span className="flow-icon">{node.icon}</span>
                  <div className="flow-text">
                    <div className="flow-label">{node.label}</div>
                    <div className="flow-sub">{node.sub}</div>
                  </div>
                </div>
                {i < FLOW.length - 1 && (
                  <div className="flow-connector">
                    <div className="flow-line" style={{ '--accent': node.color }} />
                    <div className="flow-dot" style={{ '--accent': node.color }} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
