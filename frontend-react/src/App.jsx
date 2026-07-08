import { useAuth0 } from '@auth0/auth0-react';
import { useState, useEffect } from 'react';
import Login from './components/Login';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import IngestTab from './components/IngestTab';
import './App.css';

function makeThreadId() { return `session-${Date.now()}`; }

const DEFAULT_STATS = {
  ticketCount: 0,
  passCount: 0,
  routeCounts: { RAG: 0, SQL: 0, Hybrid: 0, 'Multi-Agent': 0 },
  latencies: [],
};

const ROLES_CLAIM = 'https://support-resolution-api/roles';

export default function App() {
  const { isAuthenticated, isLoading, user, logout, getAccessTokenSilently } = useAuth0();
  const [threadId,    setThreadId]    = useState(makeThreadId);
  const [stats,       setStats]       = useState(DEFAULT_STATS);
  const [userRole,    setUserRole]    = useState('');
  const [activeTab,   setActiveTab]   = useState('chat'); // 'chat' | 'ingest'

  useEffect(() => {
    if (!isAuthenticated) return;
    getAccessTokenSilently({ authorizationParams: { audience: 'https://support-resolution-api/' } })
      .then(token => {
        const payload = JSON.parse(atob(token.split('.')[1]));
        const roles   = payload[ROLES_CLAIM] || [];
        setUserRole(roles[0] || 'l1-agent');
      })
      .catch(() => setUserRole('l1-agent'));
  }, [isAuthenticated, getAccessTokenSilently]);

  function handleTicketComplete(response, latencyMs) {
    const route  = response.classification?.routing_path || 'RAG';
    const isPass = !(
      response.classification?.confidence === 0.0 &&
      response.classification?.category   === 'ambiguous'
    );
    setStats(prev => ({
      ticketCount: prev.ticketCount + 1,
      passCount:   prev.passCount + (isPass ? 1 : 0),
      routeCounts: { ...prev.routeCounts, [route]: (prev.routeCounts[route] || 0) + 1 },
      latencies:   [...prev.latencies, latencyMs],
    }));
  }

  function handleNewSession() {
    setThreadId(makeThreadId());
    setStats(DEFAULT_STATS);
  }

  function handleSignOut() {
    logout({ logoutParams: { returnTo: window.location.origin } });
  }

  if (isLoading) return <div className="login-container"><p>Loading...</p></div>;
  if (!isAuthenticated) return <Login />;

  return (
    <div className="app-layout">
      <Sidebar
        user={{ name: user.name, email: user.email }}
        stats={stats}
        onNewSession={handleNewSession}
        onSignOut={handleSignOut}
      />
      <main className="main-content">
        <header className="main-header">
          <div className="header-row">
            <div>
              <h2>Support Ticket Portal</h2>
              <p>Describe your issue. The AI system classifies, retrieves context, assesses severity, and responds.</p>
            </div>
            {(userRole === 'admin' || userRole === 'manager') && (
              <nav className="tab-nav">
                <button
                  className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveTab('chat')}
                >Chat</button>
                <button
                  className={`tab-btn ${activeTab === 'ingest' ? 'active' : ''}`}
                  onClick={() => setActiveTab('ingest')}
                >Knowledge Base</button>
              </nav>
            )}
          </div>
        </header>

        {activeTab === 'chat' ? (
          <ChatWindow
            key={threadId}
            user={{ name: user.name, email: user.email, role: userRole }}
            threadId={threadId}
            onTicketComplete={handleTicketComplete}
          />
        ) : (
          <IngestTab />
        )}
      </main>
    </div>
  );
}
