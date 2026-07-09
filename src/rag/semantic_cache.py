"""
Semantic cache for RAG retrieval — src/rag/semantic_cache.py

Caches retrieved chunks for semantically similar queries so the embedding
API call + pgvector search + BM25 scoring are skipped on cache hits.

Safe ONLY for RAG-only tickets (documentation answers, not customer-specific).
SQL/Hybrid/Multi-Agent routes must bypass this cache — their responses depend
on customer account data that changes per ticket.
 are we having problem every
How it works:
  1. Incoming query is embedded using our Azure embedding model (same model as ingest).
  2. Cache is scanned for any stored query whose cosine similarity >= SIMILARITY_THRESHOLD.
  3. Hit  → return cached chunks immediately (~0ms, no network calls).
  4. Miss → run normal retrieve(), store (query_vector, chunks) in cache.

Storage: in-memory list — process lifetime only. Cache is lost on restart.
Max size: MAX_CACHE_SIZE entries (LRU eviction keeps memory bounded).
Threshold: 0.92 cosine similarity — high enough to only match genuinely
           equivalent questions, not loosely related ones.

ADR-017.
"""

import math
import threading
from collections import OrderedDict
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.92   # cosine similarity required for a cache hit
MAX_CACHE_SIZE       = 200    # LRU eviction after this many entries

# ---------------------------------------------------------------------------
# Cache storage — OrderedDict for O(1) LRU eviction
# Key: tuple(query_vector)  Value: list[dict] chunks
# _cache_lock: thread-safe for FastAPI + parallel eval ThreadPoolExecutor
# ---------------------------------------------------------------------------

_cache: OrderedDict = OrderedDict()
_cache_lock         = threading.Lock()

# Stats — useful for portfolio demos / eval reporting
_hits   = 0
_misses = 0


# ---------------------------------------------------------------------------
# Cosine similarity — pure Python, no extra dependency
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_lookup(query_vector: list[float]) -> Optional[list[dict]]:
    """
    Return cached chunks if a semantically equivalent query was seen before.
    Takes a pre-computed embedding vector — caller embeds once and reuses the
    same vector for both the cache check and the pgvector retrieval, avoiding
    a double embedding API call on cache miss.
    Returns None on cache miss.
    """
    global _hits, _misses

    with _cache_lock:
        for key, chunks in _cache.items():
            stored_vector = list(key)
            sim = _cosine_similarity(query_vector, stored_vector)
            if sim >= SIMILARITY_THRESHOLD:
                _cache.move_to_end(key)
                _hits += 1
                return chunks

    _misses += 1
    return None


def cache_store(query_vector: list[float], chunks: list[dict]) -> None:
    """Store retrieved chunks under the pre-computed query embedding vector."""
    key = tuple(query_vector)

    with _cache_lock:
        _cache[key] = chunks
        _cache.move_to_end(key)
        if len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)   # evict least recently used


def cache_stats() -> dict:
    """Return hit/miss counts — used by eval reporting and Langfuse metadata."""
    total = _hits + _misses
    return {
        "hits":     _hits,
        "misses":   _misses,
        "total":    total,
        "hit_rate": round(_hits / total, 3) if total else 0.0,
        "size":     len(_cache),
    }


def cache_clear() -> None:
    """Clear all cached entries — call between eval runs to avoid cross-contamination."""
    global _hits, _misses
    with _cache_lock:
        _cache.clear()
        _hits   = 0
        _misses = 0
