"""
Unit tests for src/retrieval/semantic_cache.py — 100% coverage target.

Pure Python — no LLM, no DB, no network. Runs in < 5ms.
"""

import math
import pytest

import src.retrieval.semantic_cache as cache_module
from src.retrieval.semantic_cache import (
    _cosine_similarity,
    cache_lookup,
    cache_store,
    cache_stats,
    cache_clear,
    SIMILARITY_THRESHOLD,
    MAX_CACHE_SIZE,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear cache and reset stats before every test."""
    cache_clear()
    yield
    cache_clear()


# ---------------------------------------------------------------------------
# _cosine_similarity — pure math
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus1(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_0(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_both_zero_vectors_return_0(self):
        assert _cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_similar_vectors_close_to_1(self):
        sim = _cosine_similarity([1.0, 0.01], [1.0, 0.02])
        assert sim > 0.99


# ---------------------------------------------------------------------------
# cache_store and cache_lookup
# ---------------------------------------------------------------------------

class TestCacheStoreAndLookup:
    def test_miss_returns_none(self):
        vector = [1.0, 0.0, 0.0]
        assert cache_lookup(vector) is None

    def test_exact_match_returns_chunks(self):
        vector = [1.0, 0.0, 0.0]
        chunks = [{"source": "doc.pdf", "text": "content"}]
        cache_store(vector, chunks)
        result = cache_lookup(vector)
        assert result == chunks

    def test_similar_vector_above_threshold_is_hit(self):
        base = [1.0, 0.0, 0.0, 0.0]
        similar = [0.9999, 0.0001, 0.0, 0.0]   # cosine sim > 0.92
        chunks = [{"source": "doc.pdf", "text": "hit"}]
        cache_store(base, chunks)
        result = cache_lookup(similar)
        assert result is not None

    def test_dissimilar_vector_below_threshold_is_miss(self):
        base    = [1.0, 0.0, 0.0, 0.0]
        dissim  = [0.0, 1.0, 0.0, 0.0]   # orthogonal → cos sim = 0
        cache_store(base, [{"text": "stored"}])
        assert cache_lookup(dissim) is None

    def test_multiple_entries_correct_one_returned(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        c1 = [{"text": "chunk1"}]
        c2 = [{"text": "chunk2"}]
        cache_store(v1, c1)
        cache_store(v2, c2)
        assert cache_lookup(v1) == c1
        assert cache_lookup(v2) == c2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial_stats_all_zero(self):
        stats = cache_stats()
        assert stats["hits"]   == 0
        assert stats["misses"] == 0
        assert stats["total"]  == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"]   == 0

    def test_miss_increments_misses(self):
        cache_lookup([1.0, 0.0])
        assert cache_stats()["misses"] == 1

    def test_hit_increments_hits(self):
        v = [1.0, 0.0]
        cache_store(v, [{"text": "x"}])
        cache_lookup(v)
        assert cache_stats()["hits"] == 1

    def test_hit_rate_calculated_correctly(self):
        v = [1.0, 0.0]
        cache_store(v, [{"text": "x"}])
        cache_lookup([0.0, 1.0])   # miss
        cache_lookup(v)             # hit
        stats = cache_stats()
        assert stats["hit_rate"] == pytest.approx(0.5, abs=0.01)

    def test_size_reflects_entries(self):
        cache_store([1.0, 0.0], [])
        cache_store([0.0, 1.0], [])
        assert cache_stats()["size"] == 2


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    def test_evicts_oldest_when_full(self):
        # Each unit vector is in a distinct dimension → all orthogonal (cos sim = 0)
        # so none are cache hits for each other.
        # MAX_CACHE_SIZE + 1 dimensions needed to store MAX_CACHE_SIZE unique vectors.
        dim = MAX_CACHE_SIZE + 1

        for i in range(MAX_CACHE_SIZE):
            v = [0.0] * dim
            v[i] = 1.0
            cache_store(v, [{"i": i}])

        assert cache_stats()["size"] == MAX_CACHE_SIZE

        # One more unique vector → triggers LRU eviction, size stays at MAX
        extra = [0.0] * dim
        extra[MAX_CACHE_SIZE] = 1.0
        cache_store(extra, [{"extra": True}])

        assert cache_stats()["size"] == MAX_CACHE_SIZE


# ---------------------------------------------------------------------------
# cache_clear
# ---------------------------------------------------------------------------

class TestCacheClear:
    def test_clear_removes_all_entries(self):
        cache_store([1.0, 0.0], [{"text": "x"}])
        cache_clear()
        assert cache_lookup([1.0, 0.0]) is None
        assert cache_stats()["size"] == 0

    def test_clear_resets_stats(self):
        cache_store([1.0, 0.0], [])
        cache_lookup([1.0, 0.0])   # hit
        cache_clear()
        stats = cache_stats()
        assert stats["hits"]   == 0
        assert stats["misses"] == 0
