"""
RAG retrieval pipeline — called by Agent 2 on every ticket (Sprint 5).

retrieve(query) — hybrid BM25 + vector search via Reciprocal Rank Fusion (RRF).
  BM25  : exact keyword matches  (error codes: 401, 429, 503)
  Vector: semantic matches        (meaning-based similarity)
  RRF   : each sub-retriever fetches 8 candidates; RRF fusion returns top 5

Shared helpers (_build_vector_store, _build_embed_model, _load_nodes) are also
imported by ingest.py so they are defined once and reused in both pipelines.

Performance (ADR-014):
  _nodes (BM25 corpus) is a global singleton — loaded once from nodes_cache.json,
  read-only after init, safe to share across threads.
  _thread_local holds one VectorStoreIndex + one QueryFusionRetriever per OS thread.
  VectorStoreIndex is per-thread because PGVectorStore uses asyncpg internally, and
  asyncpg connection pools are bound to the event loop that created them. Each request
  thread creates its own event loop; a global index would cause "Future attached to a
  different loop" on the second request. Per-thread index = per-thread asyncpg pool.
"""

import os
import json
import threading
from typing import List, Optional

from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as _LlamaAzureOpenAI

load_dotenv()

# Point LlamaIndex at our Azure deployment so it never falls back to the
# default OpenAI() client (which requires OPENAI_API_KEY).
# num_queries=1 in QueryFusionRetriever means this LLM is never actually
# called — but Settings.llm must be set to a real client to suppress
# "using mock llm" warnings and avoid KeyError on startup.
Settings.llm = _LlamaAzureOpenAI(
    model="gpt-4o-mini",
    deployment_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants — shared with ingest.py
# ---------------------------------------------------------------------------
TABLE_NAME     = "document_chunks"
NODES_CACHE    = "nodes_cache.json"
EMBED_DIM      = 1536
TOP_K          = 5   # chunks returned to synthesizer
RETRIEVER_K    = 8   # candidates each sub-retriever fetches before RRF fusion
MAX_NODE_CHARS = 2000


# ---------------------------------------------------------------------------
# Global nodes singleton — read-only after init, safe to share across threads
# _nodes_lock: double-checked locking so only one thread loads the JSON file
# ---------------------------------------------------------------------------
_nodes:      Optional[List[TextNode]] = None
_nodes_lock  = threading.Lock()

# Global embedding model singleton — shared across threads (stateless, thread-safe).
# Avoids recreating AzureOpenAIEmbedding on every retrieve() call.
# Used to embed the query ONCE, then reuse the vector for both semantic cache
# lookup and pgvector retrieval (prevents double embedding API call on cache miss).
_embed_model_singleton: Optional[AzureOpenAIEmbedding] = None
_embed_model_lock       = threading.Lock()

# VectorStoreIndex is stored per-thread (not global) because PGVectorStore uses
# asyncpg internally, and asyncpg connection pools are bound to the event loop
# that created them. Each request thread creates its own event loop; reusing a
# global index across threads causes "Future attached to a different loop" errors.
# Per-thread index means each request gets its own asyncpg pool bound to its loop.

# Per-thread retriever — QueryFusionRetriever wraps the shared index with a
# per-thread BM25 retriever. asyncpg handles concurrent calls internally;
# the per-thread pattern here is retained for BM25 isolation (ADR-014).
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Shared infrastructure helpers
# Used by both ingest.py (to store) and rag.py (to query)
# ---------------------------------------------------------------------------

def _build_vector_store() -> PGVectorStore:
    _host = os.environ.get("POSTGRES_HOST", "localhost")
    _port = int(os.environ.get("POSTGRES_PORT", 5432))
    _db   = os.environ["POSTGRES_DB"]
    _user = os.environ["POSTGRES_USER"]
    _pwd  = os.environ.get("POSTGRES_PASSWORD", "")
    _remote = _host not in ("localhost", "127.0.0.1")
    # psycopg2 uses sslmode=require; asyncpg (LlamaIndex) uses ssl=require
    _ssl_sync  = "?sslmode=require" if _remote else ""
    _ssl_async = "?ssl=require"     if _remote else ""
    return PGVectorStore(
        connection_string=f"postgresql://{_user}:{_pwd}@{_host}:{_port}/{_db}{_ssl_sync}",
        async_connection_string=f"postgresql+asyncpg://{_user}:{_pwd}@{_host}:{_port}/{_db}{_ssl_async}",
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,
        perform_setup=False,
    )


def _get_embed_model() -> AzureOpenAIEmbedding:
    """Return the global embedding model singleton — built once, reused across all threads."""
    global _embed_model_singleton
    if _embed_model_singleton is None:
        with _embed_model_lock:
            if _embed_model_singleton is None:
                _embed_model_singleton = _build_embed_model()
    return _embed_model_singleton


def _build_embed_model() -> AzureOpenAIEmbedding:
    return AzureOpenAIEmbedding(
        model="text-embedding-3-small",
        deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def _load_nodes() -> List[TextNode]:
    """Load nodes from nodes_cache.json for BM25 retrieval."""
    with open(NODES_CACHE, encoding="utf-8") as f:
        data = json.load(f)
    return [TextNode(text=d["text"], metadata=d["metadata"], id_=d["id_"]) for d in data]


def _get_nodes() -> List[TextNode]:
    """Return global nodes singleton — loaded once, reused across all threads."""
    global _nodes
    if _nodes is None:
        with _nodes_lock:
            if _nodes is None:
                _nodes = _load_nodes()
    return _nodes


# ---------------------------------------------------------------------------
# BM25 retriever — pure Python wrapper around rank_bm25
# Implements LlamaIndex BaseRetriever so it plugs into QueryFusionRetriever
# rank_bm25 chosen over pystemmer-based BM25: same algorithm, no C++ compiler needed
# ---------------------------------------------------------------------------

class RankBM25Retriever(BaseRetriever):
    def __init__(self, nodes: List[TextNode], similarity_top_k: int = TOP_K):
        self._nodes = nodes
        self._top_k = similarity_top_k
        tokenized = [n.text.lower().split() for n in nodes]
        self._bm25 = BM25Okapi(tokenized)
        super().__init__()

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        query_tokens = query_bundle.query_str.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self._top_k]
        return [
            NodeWithScore(node=self._nodes[i], score=float(scores[i]))
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# load_index — connects to the stored pgvector index.
# Returns the global singleton — opens the asyncpg connection once and reuses
# it across all threads. This is the key warmup target: calling this function
# during startup means the first real ticket never pays the Neon cold-start.
# ---------------------------------------------------------------------------

def load_index() -> VectorStoreIndex:
    if not hasattr(_thread_local, 'index') or _thread_local.index is None:
        vector_store    = _build_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model     = _get_embed_model()
        _thread_local.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    return _thread_local.index


# ---------------------------------------------------------------------------
# _get_fusion_retriever — returns this thread's QueryFusionRetriever.
# Both the pgvector index and BM25 retriever are per-thread (ADR-014).
# ---------------------------------------------------------------------------

def _get_fusion_retriever(top_k: int = TOP_K) -> QueryFusionRetriever:
    if not hasattr(_thread_local, 'retriever') or _thread_local.retriever is None:
        nodes = _get_nodes()
        bm25  = RankBM25Retriever(nodes=nodes, similarity_top_k=RETRIEVER_K)
        _thread_local.retriever = QueryFusionRetriever(
            retrievers=[load_index().as_retriever(similarity_top_k=RETRIEVER_K), bm25],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
        )
    return _thread_local.retriever


# ---------------------------------------------------------------------------
# retrieve — called by Agent 2 on every ticket
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K, use_cache: bool = True) -> list[dict]:
    """
    Hybrid BM25 + vector retrieval merged via Reciprocal Rank Fusion.
    Returns list of dicts: text, score, source (filename), page.

    use_cache: True for RAG-only tickets (safe — docs don't change per customer).
               False for SQL/Hybrid/Multi-Agent routes (customer-specific context).
    """
    from src.rag.semantic_cache import cache_lookup, cache_store

    query_vector = None
    if use_cache:
        # Embed once — reuse vector for both cache lookup and pgvector retrieval.
        # Prevents a double embedding API call on cache miss (ADR-017).
        query_vector = _get_embed_model().get_text_embedding(query)
        cached = cache_lookup(query_vector)
        if cached is not None:
            return cached

    result_nodes = _get_fusion_retriever(top_k).retrieve(query)

    chunks = [
        {
            "text":   n.node.get_content(),
            "score":  n.score,
            "source": n.node.metadata.get("file_name", "unknown"),
            "page":   n.node.metadata.get("page_label", ""),
        }
        for n in result_nodes
    ]

    if use_cache and query_vector is not None:
        cache_store(query_vector, chunks)

    return chunks
