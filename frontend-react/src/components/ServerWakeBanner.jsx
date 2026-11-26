import { useEffect, useState } from 'react';
import { healthCheck } from '../utils/api';

// The backend runs on Render's free plan, which sleeps after 15 min idle and
// takes ~30-60 s to wake. Ping it as soon as the chat opens so visitors see
// why the first answer is slow instead of assuming the app is broken.
const SHOW_AFTER_MS = 2500;
const EXPECTED_S    = 60;

export default function ServerWakeBanner() {
  const [state,   setState]   = useState('checking'); // checking | waking | ready | down
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const start = Date.now();
    const showTimer = setTimeout(() => !cancelled && setState(s => (s === 'checking' ? 'waking' : s)), SHOW_AFTER_MS);
    const tick = setInterval(() => setElapsed(Math.round((Date.now() - start) / 1000)), 1000);

    (async () => {
      for (let attempt = 0; attempt < 4 && !cancelled; attempt++) {
        if (await healthCheck()) {
          if (cancelled) return;
          setState(s => (s === 'waking' ? 'ready' : 'hidden'));
          return;
        }
        await new Promise(r => setTimeout(r, 5000));
      }
      if (!cancelled) setState('down');
    })();

    return () => { cancelled = true; clearTimeout(showTimer); clearInterval(tick); };
  }, []);

  useEffect(() => {
    if (state !== 'ready') return;
    const t = setTimeout(() => setState('hidden'), 4000);
    return () => clearTimeout(t);
  }, [state]);

  if (state === 'checking' || state === 'hidden') return null;

  if (state === 'ready') {
    return <div className="wake-banner wake-ready">✅ Server is awake ({elapsed}s) — answers will be fast now.</div>;
  }
  if (state === 'down') {
    return <div className="wake-banner wake-down">⚠️ The AI server is not responding. Please try again in a minute.</div>;
  }
  const pct = Math.min(95, Math.round((elapsed / EXPECTED_S) * 100));
  return (
    <div className="wake-banner" role="status">
      <div>⏳ Waking up the free-tier server… {elapsed}s <span className="wake-sub">(it sleeps when idle; usually under a minute)</span></div>
      <div className="wake-bar"><div className="wake-bar-fill" style={{ width: `${pct}%` }} /></div>
    </div>
  );
}
