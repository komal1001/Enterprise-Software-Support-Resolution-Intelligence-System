const BASE_URL    = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TIMEOUT_MS  = 120_000;  // RAGAS runs 5-8 LLM calls post-response; done fires ~35s, scores ~15s later

async function getToken(getAccessTokenSilently) {
  return getAccessTokenSilently({ authorizationParams: { audience: 'https://support-resolution-api/' } });
}

export async function submitTicketStream(
  ticketText, threadId, conversationHistory = [],
  getAccessTokenSilently,
  onStatus, onToken, onDone, onError, onScores,
) {
  const token      = await getToken(getAccessTokenSilently);
  const controller = new AbortController();
  const timeoutId  = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${BASE_URL}/ticket/stream`, {
      method: 'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body:   JSON.stringify({ ticket_text: ticketText, thread_id: threadId, conversation_history: conversationHistory }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      onError(err.detail || 'Request failed');
      return;
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop();

      for (const block of blocks) {
        const lines = block.trim().split(/\r?\n/);
        let eventType = 'message';
        let dataStr   = '';

        for (const line of lines) {
          if (line.startsWith('event: '))     eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr   = line.slice(6).trim();
        }

        if (!dataStr) continue;
        let data;
        try { data = JSON.parse(dataStr); } catch { continue; }

        if      (eventType === 'status') onStatus(data);
        else if (eventType === 'token')  onToken(data.token);
        else if (eventType === 'done')   { onDone(data); clearTimeout(timeoutId); }
        else if (eventType === 'scores') { onScores?.(data); }
        else if (eventType === 'error')  onError(data.message);
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      onError('This is taking longer than usual — likely a cold start after a period of inactivity. Please try submitting again.');
    } else {
      onError('Unable to reach the support system. Please check your connection and try again.');
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function healthCheck() {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
