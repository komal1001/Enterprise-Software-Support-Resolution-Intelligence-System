import asyncio
import logging
import os

_log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.api.auth import require_role
from src.api.limiter import limiter

router = APIRouter(tags=["admin"])


@router.get("/escalations")
def get_escalations(user: dict = Depends(require_role("manager", "admin"))):
    """Manager/Admin only — all escalated tickets ordered by newest first (SLO #12)."""
    import psycopg2, psycopg2.extras
    try:
        conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            port=int(os.environ.get("POSTGRES_PORT", 5432)),
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
        )
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET statement_timeout = '5s'")
                cur.execute("""
                    SELECT t.ticket_id, t.customer_id, c.company_name,
                           t.issue_category, t.severity_level, t.ticket_status,
                           t.assigned_team, t.escalation_flag, t.created_at
                    FROM   support_tickets t
                    LEFT JOIN customers c USING (customer_id)
                    WHERE  t.escalation_flag = true
                    ORDER  BY t.created_at DESC
                    LIMIT  100
                """)
                rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return {"escalations": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@router.post("/admin/ingest")
@limiter.limit("5/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_role("manager", "admin")),
):
    """Upload a PDF or Word doc — parse, embed, and index into pgvector + BM25 (SLO #12)."""
    _ALLOWED_EXTS = {".pdf", ".docx", ".doc"}
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in _ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail="Only PDF and Word (.docx/.doc) files are supported.")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")

    def _run_ingest(doc_bytes: bytes, filename: str) -> dict:
        from src.retrieval.ingest import ingest_pdf_bytes
        return ingest_pdf_bytes(doc_bytes, filename)

    try:
        result = await asyncio.to_thread(_run_ingest, contents, file.filename)
        return {
            "status":   "ok",
            "filename": file.filename,
            "chunks":   result.get("chunks", 0),
            "message":  f"Indexed {result.get('chunks', 0)} chunks from {file.filename}",
        }
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Ingestion pipeline not available.")
    except Exception as e:
        _log.exception("Ingest error: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
