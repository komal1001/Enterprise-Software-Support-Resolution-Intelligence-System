import { useState, useRef, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import ReactMarkdown from 'react-markdown';
import { submitTicketStream } from '../utils/api';
import AnalysisPanel from './AnalysisPanel';
import PipelineFlow from './PipelineFlow';
import ScenarioCards, { ScenarioChips } from './ScenarioCards';
import GuardrailCard from './GuardrailCard';
import ServerWakeBanner from './ServerWakeBanner';
import SourceChips from './SourceChips';
import ThinkingTrace from './ThinkingTrace';

const SLOW_HINT_MS = 8000;

const BASE_URL = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs`;

const MD_COMPONENTS = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
  ),
};

// Convert inline [Source: filename.pdf] citations to markdown links
function linkifySources(text) {
  if (!text) return '';
  return text.replace(
    /\[Source:\s*([^\]]+\.pdf)\]/g,
    (_, filename) => {
      const label = filename.replace(/_/g, ' ').replace('.pdf', '');
      return `[${label}](${BASE_URL}/${filename.trim()})`;
    }
  );
}

const STEP_LABELS = {
  agent1_classify:      'Classifying ticket...',
  agent2_rag:           'Retrieving documentation...',
  agent3_sql:           'Querying account data...',
  agent4_severity:      'Assessing severity...',
  agent5_escalation:    'Escalating to support team...',
  response_synthesizer: 'Generating response...',
};

export default function ChatWindow({ user, threadId, onTicketComplete }) {
  const { getAccessTokenSilently } = useAuth0();
  const [messages,       setMessages]       = useState([]);
  const [history,        setHistory]        = useState([]);
  const [input,          setInput]          = useState('');
  const [loading,        setLoading]        = useState(false);
  const [status,         setStatus]         = useState('');
  const [streamingText,  setStreamingText]  = useState('');
  const [error,          setError]          = useState('');
  const [agentSteps,     setAgentSteps]     = useState([]);
  const [pipelineDone,   setPipelineDone]   = useState(false);
  const [slowHint,       setSlowHint]       = useState(false);
  const agentStepsRef    = useRef([]);
  const streamingTextRef = useRef('');
  const bottomRef        = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, status, streamingText]);

  // No agent has reported yet after a few seconds → most likely a cold start.
  useEffect(() => {
    if (!loading || agentSteps.length > 0) { setSlowHint(false); return; }
    const t = setTimeout(() => setSlowHint(true), SLOW_HINT_MS);
    return () => clearTimeout(t);
  }, [loading, agentSteps.length]);

  function handleSubmit(e) {
    e.preventDefault();
    submitText(input.trim());
  }

  async function submitText(text) {
    if (!text || loading) return;

    setInput('');
    setError('');
    setStatus('');
    setStreamingText('');
    streamingTextRef.current = '';
    setAgentSteps([]);
    agentStepsRef.current = [];
    setPipelineDone(false);
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    const start = Date.now();

    await submitTicketStream(
      text,
      threadId,
      history,
      getAccessTokenSilently,
      (statusEvent) => {
        setStatus(STEP_LABELS[statusEvent.step] || statusEvent.message);
        if (statusEvent.step === 'response_synthesizer') setStatus('');
        agentStepsRef.current = [...agentStepsRef.current, statusEvent];
        setAgentSteps([...agentStepsRef.current]);
      },
      (token) => {
        setStreamingText(prev => {
          const next = prev + token;
          streamingTextRef.current = next;
          return next;
        });
        setStatus('');
      },
      (doneEvent) => {
        const latencyMs   = Date.now() - start;
        const stepsSnapshot = [...agentStepsRef.current];
        const finalContent  = doneEvent.final_response || streamingTextRef.current || '';
        setPipelineDone(true);
        setStatus('Scoring quality...');
        streamingTextRef.current = '';
        setStreamingText('');
        setLoading(false);
        setMessages(prev => [...prev, {
          role:       'assistant',
          content:    finalContent,
          response:   doneEvent,
          scores:     null,
          agentSteps: stepsSnapshot,
        }]);
        setHistory(doneEvent.conversation_history || []);
        onTicketComplete(doneEvent, latencyMs);
      },
      (errMsg) => {
        setStatus('');
        setStreamingText('');
        setLoading(false);
        if (errMsg && typeof errMsg === 'object' && errMsg.blocked_by === 'guardrail') {
          setMessages(prev => [...prev, { role: 'blocked', reason: errMsg.reason, content: errMsg.message }]);
          return;
        }
        if (typeof errMsg === 'object') errMsg = errMsg?.message || JSON.stringify(errMsg);
        if (errMsg?.includes('guardrail') || errMsg?.includes('Blocked') || errMsg?.includes('blocked')) {
          setError('Blocked by security guardrails. Please submit a valid support ticket.');
        } else if (errMsg?.includes('Too many')) {
          setError('Too many requests. Please wait a moment before submitting again.');
        } else {
          setError(errMsg || 'Something went wrong. Please try again.');
        }
      },
      // onScores — arrives 2-3s after done, attach to the last assistant message
      (scoresData) => {
        setStatus('');
        setMessages(prev => {
          const copy = [...prev];
          for (let i = copy.length - 1; i >= 0; i--) {
            if (copy[i].role === 'assistant') {
              copy[i] = { ...copy[i], scores: scoresData };
              break;
            }
          }
          return copy;
        });
      },
    );
  }

  const lastAssistantIdx = messages.map(m => m.role).lastIndexOf('assistant');

  return (
    <div className="chat-window">
      <div className="messages">
        <ServerWakeBanner />

        {messages.length === 0 && (
          <div className="empty-state scenario-intro">
            <p className="scenario-welcome">Welcome, {user.name}.</p>
            <p>Pick a scenario to watch the agents work — or describe your own issue below.</p>
            <ScenarioCards onPick={submitText} disabled={loading} />
          </div>
        )}

        {messages.map((msg, i) => msg.role === 'blocked' ? (
          <div key={i} className="message assistant">
            <GuardrailCard reason={msg.reason} message={msg.content} />
          </div>
        ) : (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === 'assistant' && (msg.agentSteps || []).some(s => s.thinking) && (
              <details className="thinking-toggle" open={i === lastAssistantIdx}>
                <summary className="thinking-toggle-summary">🧠 Agent reasoning — what each agent decided and why</summary>
                <ThinkingTrace steps={msg.agentSteps} />
              </details>
            )}
            {msg.role === 'assistant' && msg.response && (
              <details className="analysis-toggle">
                <summary className="analysis-toggle-summary">
                  {(() => {
                    const clf = msg.response.classification || {};
                    return `📊 Quality scores & pipeline details — ${clf.severity || ''} · ${clf.routing_path || ''} route · ${Math.round((clf.confidence || 0) * 100)}% confidence`;
                  })()}
                </summary>
                <AnalysisPanel
                  response={{ ...msg.response, scores: msg.scores }}
                  role={user.role}
                  agentSteps={msg.agentSteps || []}
                />
              </details>
            )}
            <div className="bubble markdown-body">
              <ReactMarkdown components={MD_COMPONENTS}>{linkifySources(msg.content)}</ReactMarkdown>
            </div>
            {msg.role === 'assistant' && <SourceChips response={msg.response} />}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            {/* Live pipeline flow — stays visible throughout processing and streaming */}
            {agentSteps.length > 0 && (
              <div className="pipeline-live-panel">
                <PipelineFlow steps={agentSteps} done={pipelineDone} />
                <ThinkingTrace steps={agentSteps} live={!streamingText} />
              </div>
            )}
            <div className={`bubble ${streamingText ? '' : 'loading'}`}>
              {streamingText
                ? <div className="markdown-body"><ReactMarkdown components={MD_COMPONENTS}>{linkifySources(streamingText)}</ReactMarkdown><span className="cursor">▌</span></div>
                : <><span className="dot" /><span className="dot" /><span className="dot" /></>
              }
            </div>
            {slowHint && (
              <p className="slow-hint">Still working — the first request after the server has been idle can take up to a minute.</p>
            )}
          </div>
        )}

        {!loading && status === 'Scoring quality...' && (
          <p style={{ fontSize: 12, color: '#94a3b8', fontStyle: 'italic', padding: '0 28px' }}>
            Scoring quality...
          </p>
        )}

        {error && <div className="error-banner">{error}</div>}
        <div ref={bottomRef} />
      </div>

      {messages.length > 0 && <ScenarioChips onPick={submitText} disabled={loading} />}

      <form className="input-bar" onSubmit={handleSubmit}>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e); } }}
          placeholder="Describe your issue... (Enter to send, Shift+Enter for new line)"
          rows={2}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
