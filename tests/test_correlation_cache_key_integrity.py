"""
Tests for correlation matrix cache key integrity.

Validates:
1. Same data produces the same cache key (determinism)
2. Different data produces different cache keys (collision resistance)
3. Global numpy RNG state is NOT polluted by cache key computation
4. Cache key is 32 hex chars (128-bit)
5. Source code contains no np.random.seed calls
6. Large-matrix fallback uses np.random.default_rng (local RNG)
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def _make_biomatrix(
    data: np.ndarray,
    feature_ids: list[str] | None = None,
    sample_ids: list[str] | None = None,
):
    """Helper to build a BioMatrix for testing."""
    from cliquefinder.core.biomatrix import BioMatrix

    n_features, n_samples = data.shape
    if feature_ids is None:
        feature_ids = [f"gene_{i}" for i in range(n_features)]
    if sample_ids is None:
        sample_ids = [f"sample_{j}" for j in range(n_samples)]

    return BioMatrix(
        data=data,
        feature_ids=pd.Index(feature_ids),
        sample_ids=pd.Index(sample_ids),
        sample_metadata=pd.DataFrame(index=pd.Index(sample_ids)),
        quality_flags=np.zeros_like(data, dtype=int),
    )


# =====================================================================
# _compute_cache_key correctness
# =====================================================================


class TestCacheKeyDeterminism:
    """Same data must always produce the same cache key."""

    def test_same_data_same_key(self):
        """Identical BioMatrix instances produce identical cache keys."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        m1 = _make_biomatrix(data.copy())
        m2 = _make_biomatrix(data.copy())

        assert _compute_cache_key(m1) == _compute_cache_key(m2)

    def test_repeated_calls_same_key(self):
        """Calling _compute_cache_key twice on the same matrix gives the same result."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.random.default_rng(99).standard_normal((10, 8))
        m = _make_biomatrix(data)

        k1 = _compute_cache_key(m)
        k2 = _compute_cache_key(m)
        assert k1 == k2


class TestCacheKeyCollisionResistance:
    """Different data must produce different cache keys."""

    def test_different_values_different_key(self):
        """Matrices with different values produce different keys."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data1 = np.ones((5, 3), dtype=np.float64)
        data2 = np.ones((5, 3), dtype=np.float64)
        data2[2, 1] = 999.0  # single element change

        m1 = _make_biomatrix(data1)
        m2 = _make_biomatrix(data2)

        assert _compute_cache_key(m1) != _compute_cache_key(m2)

    def test_different_feature_ids_different_key(self):
        """Same data but different gene IDs produce different keys."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.ones((3, 4), dtype=np.float64)
        m1 = _make_biomatrix(data, feature_ids=["A", "B", "C"])
        m2 = _make_biomatrix(data, feature_ids=["X", "Y", "Z"])

        assert _compute_cache_key(m1) != _compute_cache_key(m2)

    def test_different_sample_ids_different_key(self):
        """Same data but different sample IDs produce different keys."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.ones((3, 4), dtype=np.float64)
        m1 = _make_biomatrix(data, sample_ids=["s1", "s2", "s3", "s4"])
        m2 = _make_biomatrix(data, sample_ids=["t1", "t2", "t3", "t4"])

        assert _compute_cache_key(m1) != _compute_cache_key(m2)

    def test_subtle_value_difference_detected(self):
        """A tiny floating-point difference is caught (regression for 1% sampling bug)."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        rng = np.random.default_rng(123)
        data1 = rng.standard_normal((100, 50))
        data2 = data1.copy()
        # Change a single element by a tiny amount
        data2[50, 25] += 1e-15

        m1 = _make_biomatrix(data1)
        m2 = _make_biomatrix(data2)

        assert _compute_cache_key(m1) != _compute_cache_key(m2)


class TestCacheKeyFormat:
    """Cache key has the correct format (32 hex chars = 128-bit)."""

    def test_key_length_is_32(self):
        """Cache key is exactly 32 hex characters."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        m = _make_biomatrix(data)
        key = _compute_cache_key(m)

        assert len(key) == 32, f"Expected 32 hex chars, got {len(key)}: {key}"

    def test_key_is_valid_hex(self):
        """Cache key contains only valid hexadecimal characters."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        m = _make_biomatrix(data)
        key = _compute_cache_key(m)

        # int(..., 16) will raise ValueError if not valid hex
        int(key, 16)


# =====================================================================
# No global RNG pollution
# =====================================================================


class TestNoGlobalRNGPollution:
    """_compute_cache_key must not alter np.random global state."""

    def test_global_rng_state_preserved(self):
        """np.random.get_state() is unchanged after calling _compute_cache_key."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        m = _make_biomatrix(data)

        # Capture global RNG state before
        state_before = np.random.get_state()

        _compute_cache_key(m)

        # Capture global RNG state after
        state_after = np.random.get_state()

        # Compare the state tuples
        assert state_before[0] == state_after[0], "RNG algorithm changed"
        np.testing.assert_array_equal(
            state_before[1], state_after[1],
            err_msg="Global RNG internal state was mutated by _compute_cache_key"
        )
        assert state_before[2] == state_after[2], "RNG position changed"

    def test_global_rng_sequence_unaffected(self):
        """Random numbers drawn after _compute_cache_key are the same as without it."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        m = _make_biomatrix(data)

        # Set a known seed and draw a random number
        np.random.seed(12345)
        expected = np.random.random(5)

        # Reset to same seed, call _compute_cache_key in between, then draw
        np.random.seed(12345)
        _compute_cache_key(m)
        actual = np.random.random(5)

        np.testing.assert_array_equal(expected, actual)


# =====================================================================
# Source-level checks: no np.random.seed in correlation_matrix.py
# =====================================================================


class TestSourceCodeCleanliness:
    """Source code must not contain np.random.seed calls."""

    def test_no_np_random_seed_in_source(self):
        """correlation_matrix.py does not call np.random.seed()."""
        from cliquefinder.utils import correlation_matrix

        source = inspect.getsource(correlation_matrix)
        assert "np.random.seed" not in source, (
            "correlation_matrix.py still contains np.random.seed() — "
            "use np.random.default_rng() instead"
        )

    def test_uses_default_rng_for_fallback(self):
        """Large-matrix fallback path uses np.random.default_rng (local RNG)."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        source = inspect.getsource(_compute_cache_key)
        assert "default_rng" in source, (
            "_compute_cache_key should use np.random.default_rng for the "
            "large-matrix sampling fallback"
        )

    def test_cache_key_docstring_says_32_chars(self):
        """Docstring documents 32-character key."""
        from cliquefinder.utils.correlation_matrix import _compute_cache_key

        docstring = _compute_cache_key.__doc__
        assert "32" in docstring, (
            "Docstring should document 32-char key length"
        )
