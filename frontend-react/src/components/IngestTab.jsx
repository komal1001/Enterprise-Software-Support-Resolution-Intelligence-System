import { useState, useRef } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function IngestTab() {
  const { getAccessTokenSilently } = useAuth0();
  const [file,     setFile]     = useState(null);
  const [status,   setStatus]   = useState('idle'); // idle | uploading | success | error
  const [message,  setMessage]  = useState('');
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  function pickFile(f) {
    if (!f) return;
    const allowed = ['.pdf', '.docx', '.doc'];
    if (!allowed.some(ext => f.name.toLowerCase().endsWith(ext))) {
      setMessage('Only PDF and Word (.docx / .doc) files are supported.');
      setStatus('error');
      return;
    }
    setFile(f);
    setStatus('idle');
    setMessage('');
  }

  async function handleUpload() {
    if (!file) return;
    setStatus('uploading');
    setMessage('Parsing and indexing — this may take 20–60 seconds…');

    try {
      const token = await getAccessTokenSilently({
        authorizationParams: { audience: 'https://support-resolution-api/' },
      });
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${BASE_URL}/admin/ingest`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });

      const data = await res.json().catch(() => ({ detail: 'Unknown error' }));
      if (!res.ok) {
        setStatus('error');
        setMessage(data.detail || 'Ingestion failed.');
      } else {
        setStatus('success');
        setMessage(data.message || `Indexed ${data.chunks} chunks.`);
        setFile(null);
      }
    } catch (err) {
      setStatus('error');
      setMessage('Upload failed — check your connection.');
    }
  }

  return (
    <div className="ingest-tab">
      <h2 className="ingest-heading">Knowledge Base Ingestion</h2>
      <p className="ingest-sub">
        Upload a PDF to parse, embed, and index it into the vector store.
        New documents are immediately searchable via semantic retrieval.
      </p>

      {/* Drop zone */}
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => {
          e.preventDefault();
          setDragOver(false);
          pickFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.doc"
          style={{ display: 'none' }}
          onChange={e => pickFile(e.target.files[0])}
        />
        {file ? (
          <div className="drop-zone-content">
            <span className="drop-icon">📄</span>
            <span className="drop-filename">{file.name}</span>
            <span className="drop-size">{(file.size / 1024).toFixed(0)} KB</span>
          </div>
        ) : (
          <div className="drop-zone-content">
            <span className="drop-icon">⬆</span>
            <span>Drag & drop a PDF or Word doc here or <u>click to browse</u></span>
          </div>
        )}
      </div>

      <button
        className="ingest-btn"
        disabled={!file || status === 'uploading'}
        onClick={handleUpload}
      >
        {status === 'uploading' ? 'Indexing…' : 'Upload & Index'}
      </button>

      {message && (
        <div className={`ingest-msg ${status}`}>
          {status === 'uploading' && <span className="spinner" />}
          {message}
        </div>
      )}

      <div className="ingest-note">
        <strong>Note:</strong> New documents are stored in pgvector (persistent) and
        added to the BM25 keyword index for the current server session. To make BM25
        indexing permanent, re-run the ingestion script and redeploy.
      </div>
    </div>
  );
}
