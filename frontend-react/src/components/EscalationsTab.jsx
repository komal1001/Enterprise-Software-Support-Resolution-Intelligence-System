import { useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SEVERITY_CLASS = {
  Critical: 'sev-critical',
  High:     'sev-high',
  Medium:   'sev-medium',
  Low:      'sev-low',
};

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

export default function EscalationsTab() {
  const { getAccessTokenSilently } = useAuth0();
  const [rows,    setRows]    = useState([]);
  const [count,   setCount]   = useState(0);
  const [status,  setStatus]  = useState('idle'); // idle | loading | error
  const [error,   setError]   = useState('');

  async function fetchEscalations() {
    setStatus('loading');
    setError('');
    try {
      const token = await getAccessTokenSilently({
        authorizationParams: { audience: 'https://support-resolution-api/' },
      });
      const res = await fetch(`${BASE_URL}/escalations`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      setRows(data.escalations || []);
      setCount(data.count || 0);
      setStatus('idle');
    } catch (err) {
      setError(err.message || 'Failed to load escalations.');
      setStatus('error');
    }
  }

  useEffect(() => { fetchEscalations(); }, []);

  return (
    <div className="escalations-tab">
      <div className="esc-header">
        <div>
          <h2 className="esc-title">Escalated Tickets</h2>
          <p className="esc-sub">Tickets flagged by Agent 4 for human review — ordered by newest first.</p>
        </div>
        <button className="esc-refresh-btn" onClick={fetchEscalations} disabled={status === 'loading'}>
          {status === 'loading' ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      {status === 'error' && (
        <div className="esc-error">{error}</div>
      )}

      {status !== 'loading' && rows.length === 0 && status !== 'error' && (
        <div className="esc-empty">No escalated tickets found.</div>
      )}

      {rows.length > 0 && (
        <>
          <p className="esc-count">{count} escalated ticket{count !== 1 ? 's' : ''}</p>
          <div className="esc-table-wrap">
            <table className="esc-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Company</th>
                  <th>Category</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th>Assigned Team</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(row => (
                  <tr key={row.ticket_id}>
                    <td className="esc-id">#{row.ticket_id}</td>
                    <td>{row.company_name || '—'}</td>
                    <td className="esc-category">{(row.issue_category || '').replace(/_/g, ' ')}</td>
                    <td>
                      <span className={`sev-badge ${SEVERITY_CLASS[row.severity_level] || ''}`}>
                        {row.severity_level || '—'}
                      </span>
                    </td>
                    <td>{row.ticket_status || '—'}</td>
                    <td>{row.assigned_team || '—'}</td>
                    <td className="esc-date">{formatDate(row.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
