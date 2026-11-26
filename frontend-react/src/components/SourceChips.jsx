const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const SQL_ROUTES = ['SQL', 'Hybrid', 'Multi-Agent'];

// Always-visible citations under each answer (SLO #8 Source Attribution),
// so visitors don't have to open the analysis panel to see them.
export default function SourceChips({ response }) {
  const docs  = (response?.sources || []).filter(Boolean);
  const route = response?.classification?.routing_path;
  const usedSql = SQL_ROUTES.includes(route);
  if (docs.length === 0 && !usedSql) return null;

  return (
    <div className="source-chips">
      <span className="source-chips-label">Sources</span>
      {docs.map(file => (
        <a
          key={file}
          className="source-chip"
          href={`${API_URL}/docs/${file}`}
          target="_blank"
          rel="noopener noreferrer"
          title={`Open ${file}`}
        >
          📄 {file.replace(/_/g, ' ').replace(/\.pdf$/i, '')}
        </a>
      ))}
      {usedSql && (
        <span className="source-chip source-chip-sql" title="Live query against customers, tickets and incident logs">
          🗄️ Account database (SQL)
        </span>
      )}
    </div>
  );
}
