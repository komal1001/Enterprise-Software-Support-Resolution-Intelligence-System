export default function Sidebar({ user, stats, onNewSession, onSignOut }) {
  const { ticketCount, passCount, routeCounts, latencies } = stats;
  const tsr    = ticketCount ? Math.round(passCount / ticketCount * 100) : 0;
  const avgMs  = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0;
  const tsrOk  = tsr >= 90;
  const latOk  = avgMs <= 5000;

  return (
    <aside className="sidebar">
      <div className="user-info">
        <div className="user-avatar">{user.name?.[0] ?? '?'}</div>
        <div>
          <div className="user-name">{user.name}</div>
          <div className="user-role">{user.role}</div>
        </div>
      </div>

      <hr />

      <div className="slo-section">
        <div className="slo-title">Session SLOs</div>

        <div className={`slo-metric ${tsrOk ? 'ok' : 'fail'}`}>
          <span>TSR (≥ 90%)</span>
          <span>{ticketCount ? `${tsr}%` : '—'}</span>
        </div>

        <div className={`slo-metric ${latOk ? 'ok' : 'fail'}`}>
          <span>Avg Latency (≤ 5s)</span>
          <span>{latencies.length ? `${(avgMs / 1000).toFixed(1)}s` : '—'}</span>
        </div>
      </div>

      <hr />

      <div className="route-section">
        <div className="slo-title">Routing Distribution</div>
        {Object.entries(routeCounts).map(([route, count]) => (
          count > 0 && (
            <div key={route} className="route-row">
              <span>{route}</span>
              <span>{count} ({ticketCount ? Math.round(count / ticketCount * 100) : 0}%)</span>
            </div>
          )
        ))}
        {ticketCount === 0 && <p className="empty-note">No tickets yet</p>}
      </div>

      <hr />

      <button className="btn-secondary" onClick={onNewSession}>New Session</button>
      <button className="btn-ghost"     onClick={onSignOut}>Sign Out</button>
    </aside>
  );
}
