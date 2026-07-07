# FastAPI entry point
#
# Starts the application, registers routes, configures CORS for React frontend.
# CORS allows the React dev server (localhost:3000) to call the FastAPI backend.
#
# GET /health is the liveness probe for containerised deployments (Docker/Kubernetes).
# ADR-009  — React frontend calls this API via fetch/axios.

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI

# Patch every new event loop created in any thread — LlamaIndex/asyncpg spins up
# new event loops inside ThreadPoolExecutor workers; without this each new loop
# is unpatched and raises "Detected nested async" (same fix as eval.py ADR-013).
import nest_asyncio

class _PatchedLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()
        nest_asyncio.apply(loop)
        return loop

asyncio.set_event_loop_policy(_PatchedLoopPolicy())
nest_asyncio.apply()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.limiter import limiter

_DIST = Path(__file__).parent.parent.parent / "frontend-react" / "dist"
_DOCS = Path(__file__).parent.parent.parent / "Documents"

app = FastAPI(
    title="Enterprise Support & Resolution Intelligence System",
    description="SLO-bound autonomous agentic AI system for enterprise software support.",
    version="0.1.0",
)

# Register limiter with app so slowapi can intercept rate-limit violations
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# OpenTelemetry infrastructure instrumentation
#
# Langfuse v4 sets the global OTel TracerProvider on first init, so these
# instrumentors automatically send spans to Langfuse — no separate exporter.
# FastAPI spans  → HTTP request latency per endpoint (SLO #5/#6)
# psycopg2 spans → DB query times for SQL agent and pgvector retrieval
# ---------------------------------------------------------------------------

from src.observability.langfuse_client import get_langfuse as _get_lf
_get_lf()   # initialise Langfuse + global OTel provider before instrumentors attach

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

FastAPIInstrumentor.instrument_app(app)
Psycopg2Instrumentor().instrument()

# ---------------------------------------------------------------------------
# CORS — allow React frontend to call the API
# In production, replace * with the actual deployed frontend domain.
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
]
# Add deployed frontend URL from env var — set FRONTEND_URL on Railway/Render
_frontend_url = os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    _ALLOWED_ORIGINS.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "X-Accel-Buffering"],
)

# ---------------------------------------------------------------------------
# Serve PDFs from Documents/ at /docs/<filename>.pdf
# Used by the frontend to open source documents as clickable links.
# ---------------------------------------------------------------------------

if _DOCS.exists():
    app.mount("/docs", StaticFiles(directory=_DOCS), name="docs")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------------------------
# Shutdown — flush any buffered Langfuse spans before the process exits.
# Without this, spans from the last in-flight ticket may never reach Langfuse
# if the process is killed (Ctrl+C, Railway/Render restart, OOM kill).
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """
    Warm both DB connections on startup so the first ticket doesn't hit cold-start delay.

    Two separate connections need warming:
    1. psycopg2 pool (SQL agent) — direct TCP connection to Neon
    2. LlamaIndex asyncpg (RAG agent) — separate connection used by pgvector retriever.
       Without this, the first RAG ticket pays a 10-30s Neon cold start even if SQL is warm.
    """
    import psycopg2

    def _ping_psycopg2():
        try:
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                port=int(os.environ.get("POSTGRES_PORT", 5432)),
                dbname=os.environ["POSTGRES_DB"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
            )
            conn.cursor().execute("SELECT 1")
            conn.close()
        except Exception:
            pass

    # VectorStoreIndex is now per-thread — no global singleton to pre-warm.
    # psycopg2 ping is sufficient to wake Neon compute before first ticket.
    await asyncio.to_thread(_ping_psycopg2)


@app.on_event("shutdown")
async def shutdown_event():
    from src.observability.langfuse_client import get_langfuse
    try:
        get_langfuse().flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Serve React frontend from dist/ if it exists
# API routes take priority (registered above). SPA fallback catches all other
# paths so React Router works correctly on hard reload.
# ---------------------------------------------------------------------------

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(_DIST / "index.html")


# ---------------------------------------------------------------------------
# Run directly: python -m src.api.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
