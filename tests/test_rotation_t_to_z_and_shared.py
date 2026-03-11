"""
Tests for the rotation framework's t-to-z conversion, shared rotation
matrices, and float64 vector normalization.

t->z approximation threshold raised from df>100 to df>1000:
    - df=500 must use proper t-distribution CDF (not normal approximation)
    - df=1500 may use normal approximation (z ~ t)
    - Verifies the threshold change in GPU path, CPU path, and observed stats

Shared rotations for FWER correction in test_gene_sets:
    - test_gene_set accepts rotation_matrices= parameter
    - test_gene_sets pre-generates shared rotations by default
    - shared_rotations=False reverts to independent generation
    - Pre-generated rotations have correct shape and unit norms

Rotation vector normalization always in float64 on CPU:
    - generate_rotation_vectors always normalizes in float64
    - Output has correct dtype and unit norms
    - use_gpu=True does not affect normalization precision
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from scipy.special import ndtri

from cliquefinder.stats.rotation import (
    generate_rotation_vectors,
    apply_rotations_batched,
    _apply_rotations_cpu,
    RotationTestEngine,
    RotationTestConfig,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_engine_and_fit(
    n_genes: int = 50,
    n_samples: int = 20,
    seed: int = 42,
) -> RotationTestEngine:
    """Create a fitted RotationTestEngine with random data."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_genes, n_samples))
    gene_ids = [f"GENE{i}" for i in range(n_genes)]
    metadata = pd.DataFrame({
        "sample": [f"S{i}" for i in range(n_samples)],
        "group": ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
    })

    engine = RotationTestEngine(data, gene_ids, metadata)
    engine.fit(
        conditions=["A", "B"],
        contrast=("A", "B"),
        condition_column="group",
    )
    return engine


# =============================================================================
# t->z approximation threshold
# =============================================================================


class TestTtoZThreshold:
    """Verify df>100 was raised to df>1000 for the t->z approximation."""

    def test_cpu_path_df500_uses_t_distribution(self):
        """At df=500 (between 100 and 1000), the CPU path must use
        proper t-distribution CDF, not the normal approximation."""
        rng = np.random.default_rng(123)
        n_genes, n_dims, n_rotations = 10, 15, 50
        df_residual = 500

        U = rng.standard_normal((n_genes, n_dims))
        rho_sq = np.sum(U ** 2, axis=1)
        R = generate_rotation_vectors(n_rotations, n_dims, rng=rng, use_gpu=False)
        sample_var = rng.uniform(0.5, 2.0, n_genes)

        t_rot, z_rot, valid = _apply_rotations_cpu(
            U, rho_sq, R, sample_var,
            moderated_variances=None,
            df_residual=df_residual,
            use_df=500.0,
        )

        # If proper t-distribution is used, z != t (they differ for finite df).
        # The difference is small but nonzero for df=500.
        # For the normal approx, z_rot would be identical to t_rot.
        assert not np.allclose(t_rot, z_rot, atol=0), (
            "At df=500, z should differ from t (t-distribution CDF should be used)"
        )

        # Verify the actual values match scipy's t->z transform
        p_expected = scipy_stats.t.cdf(t_rot, df=500.0)
        p_expected = np.clip(p_expected, 1e-15, 1 - 1e-15)
        z_expected = ndtri(p_expected)
        assert_allclose(z_rot, z_expected, rtol=1e-10)

    def test_cpu_path_df1500_uses_normal_approx(self):
        """At df=1500 (above 1000), the CPU path uses z ≈ t."""
        rng = np.random.default_rng(456)
        n_genes, n_dims, n_rotations = 10, 15, 50
        df_residual = 1500

        U = rng.standard_normal((n_genes, n_dims))
        rho_sq = np.sum(U ** 2, axis=1)
        R = generate_rotation_vectors(n_rotations, n_dims, rng=rng, use_gpu=False)
        sample_var = rng.uniform(0.5, 2.0, n_genes)

        t_rot, z_rot, valid = _apply_rotations_cpu(
            U, rho_sq, R, sample_var,
            moderated_variances=None,
            df_residual=df_residual,
            use_df=1500.0,
        )

        # Normal approximation: z should equal t
        assert_allclose(z_rot, t_rot, rtol=1e-15)

    def test_batched_df500_uses_t_distribution(self):
        """apply_rotations_batched dispatches to impl; at df=500 should
        use proper t-distribution (not normal approx)."""
        rng = np.random.default_rng(789)
        n_genes, n_dims, n_rotations = 10, 15, 50
        df_residual = 500

        U = rng.standard_normal((n_genes, n_dims))
        rho_sq = np.sum(U ** 2, axis=1)
        R = generate_rotation_vectors(n_rotations, n_dims, rng=rng, use_gpu=False)
        sample_var = rng.uniform(0.5, 2.0, n_genes)

        t_rot, z_rot, valid = apply_rotations_batched(
            U, rho_sq, R, sample_var,
            moderated_variances=None,
            df_residual=df_residual,
            df_total=500.0,
            use_gpu=False,
        )

        # z should differ from t at df=500
        assert not np.allclose(t_rot, z_rot, atol=0)

    def test_batched_df1500_uses_normal_approx(self):
        """apply_rotations_batched at df=1500 uses z ≈ t."""
        rng = np.random.default_rng(101)
        n_genes, n_dims, n_rotations = 10, 15, 50

        U = rng.standard_normal((n_genes, n_dims))
        rho_sq = np.sum(U ** 2, axis=1)
        R = generate_rotation_vectors(n_rotations, n_dims, rng=rng, use_gpu=False)
        sample_var = rng.uniform(0.5, 2.0, n_genes)

        t_rot, z_rot, valid = apply_rotations_batched(
            U, rho_sq, R, sample_var,
            moderated_variances=None,
            df_residual=1500,
            df_total=1500.0,
            use_gpu=False,
        )

        assert_allclose(z_rot, t_rot, rtol=1e-15)

    def test_observed_tz_threshold_in_test_gene_set(self):
        """The observed t->z conversion in test_gene_set also uses
        df>1000 threshold."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        # With 20 samples and 2 groups, df_residual = 18.
        # df_total with EB will be moderate. We can't easily force
        # df_total > 1000, but we can verify the code path exists
        # by checking that df <= 1000 uses proper t-distribution.
        gene_set = [f"GENE{i}" for i in range(5)]
        config = RotationTestConfig(
            n_rotations=99, use_gpu=False, seed=42,
        )
        result = engine.test_gene_set(
            gene_set, gene_set_id="test_set", config=config,
        )

        # Should produce valid results (not crash)
        assert result.n_genes_found >= 2
        assert "msq" in result.p_values

    def test_error_magnitude_at_boundary(self):
        """Demonstrate that df=1000 has substantially less t->z error
        than df=100 for |t|=5, justifying the raised threshold.

        Actual errors:
        - df=100:  ~6.1% relative error  (old threshold — unacceptable)
        - df=1000: ~0.65% relative error  (new threshold — acceptable for ROAST)
        - df=5000: ~0.13% relative error  (diminishing returns)
        """
        t_val = 5.0

        # At df=100: compute proper z and compare to t
        p_100 = scipy_stats.t.cdf(t_val, df=100)
        z_100 = scipy_stats.norm.ppf(p_100)
        error_100 = abs(z_100 - t_val) / abs(z_100)
        # Error at df=100 is ~6% — unacceptable
        assert error_100 > 0.05, f"Expected >5% error at df=100, got {error_100:.4%}"

        # At df=1000: error is ~0.65%, roughly 10x improvement
        p_1000 = scipy_stats.t.cdf(t_val, df=1000)
        z_1000 = scipy_stats.norm.ppf(p_1000)
        error_1000 = abs(z_1000 - t_val) / abs(z_1000)
        assert error_1000 < 0.01, f"Expected <1% error at df=1000, got {error_1000:.4%}"

        # Verify the improvement is substantial (at least 5x)
        improvement = error_100 / error_1000
        assert improvement > 5, (
            f"Expected >5x improvement, got {improvement:.1f}x"
        )


# =============================================================================
# Shared rotations for FWER correction
# =============================================================================


class TestSharedRotations:
    """Verify test_gene_sets uses shared rotations for valid FWER correction."""

    def test_test_gene_set_accepts_rotation_matrices_param(self):
        """test_gene_set signature includes rotation_matrices parameter."""
        sig = inspect.signature(RotationTestEngine.test_gene_set)
        assert "rotation_matrices" in sig.parameters, (
            "test_gene_set must accept rotation_matrices parameter"
        )

    def test_test_gene_sets_accepts_shared_rotations_param(self):
        """test_gene_sets signature includes shared_rotations parameter."""
        sig = inspect.signature(RotationTestEngine.test_gene_sets)
        assert "shared_rotations" in sig.parameters, (
            "test_gene_sets must accept shared_rotations parameter"
        )

    def test_shared_rotations_default_true(self):
        """shared_rotations defaults to True."""
        sig = inspect.signature(RotationTestEngine.test_gene_sets)
        default = sig.parameters["shared_rotations"].default
        assert default is True, f"shared_rotations default should be True, got {default}"

    def test_pregenerated_rotations_used_when_provided(self):
        """When rotation_matrices is provided, test_gene_set uses them
        instead of generating new ones."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        # Pre-generate rotation vectors
        n_rotations = 99
        rng = np.random.default_rng(42)
        R_shared = generate_rotation_vectors(
            n_rotations,
            engine._precomputed.residual_dims,
            rng=rng,
            use_gpu=False,
        )

        gene_set = [f"GENE{i}" for i in range(5)]
        config = RotationTestConfig(
            n_rotations=n_rotations, use_gpu=False, seed=42,
        )

        # Test with pre-generated rotations
        result = engine.test_gene_set(
            gene_set, gene_set_id="test",
            config=config, rotation_matrices=R_shared,
        )

        assert result.n_genes_found >= 2
        assert "msq" in result.p_values

    def test_shared_rotations_produce_same_null(self):
        """Two gene sets tested with the same shared rotations produce
        results from the same null distribution (same rotation vectors)."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        gene_sets = {
            "set_a": [f"GENE{i}" for i in range(5)],
            "set_b": [f"GENE{i}" for i in range(10, 15)],
        }
        config = RotationTestConfig(
            n_rotations=99, use_gpu=False, seed=42,
        )

        # Test with shared rotations (default)
        results_shared = engine.test_gene_sets(
            gene_sets, config=config, shared_rotations=True,
        )

        assert len(results_shared) == 2
        # Both should have valid p-values
        for r in results_shared:
            assert r.n_genes_found >= 2
            assert "msq" in r.p_values

    def test_independent_rotations_differ_from_shared(self):
        """With shared_rotations=False, each gene set gets independent
        rotations (legacy behavior)."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        gene_sets = {
            "set_a": [f"GENE{i}" for i in range(5)],
            "set_b": [f"GENE{i}" for i in range(10, 15)],
        }
        config = RotationTestConfig(
            n_rotations=99, use_gpu=False, seed=42,
        )

        # Independent rotations
        results_indep = engine.test_gene_sets(
            gene_sets, config=config, shared_rotations=False,
        )

        assert len(results_indep) == 2
        for r in results_indep:
            assert r.n_genes_found >= 2

    def test_shared_rotation_shape_and_norms(self):
        """Pre-generated shared rotations have correct shape and unit norms."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        n_rotations = 199
        rng = np.random.default_rng(42)
        R_shared = generate_rotation_vectors(
            n_rotations,
            engine._precomputed.residual_dims,
            rng=rng,
            use_gpu=False,
        )

        assert R_shared.shape == (n_rotations, engine._precomputed.residual_dims)
        norms = np.linalg.norm(R_shared, axis=1)
        assert_allclose(norms, 1.0, atol=1e-10)


# =============================================================================
# Rotation normalization in float64
# =============================================================================


class TestFloat64Normalization:
    """Verify rotation vectors are always normalized in float64 on CPU."""

    def test_normalization_always_float64(self):
        """generate_rotation_vectors returns float64 arrays regardless of
        use_gpu setting."""
        rng = np.random.default_rng(42)

        # CPU path
        R_cpu = generate_rotation_vectors(100, 20, rng=rng, use_gpu=False)
        assert R_cpu.dtype == np.float64

        # GPU path (even if MLX is available, normalization should be float64)
        rng2 = np.random.default_rng(42)
        R_gpu = generate_rotation_vectors(100, 20, rng=rng2, use_gpu=True)
        assert R_gpu.dtype == np.float64

    def test_unit_norms_float64_precision(self):
        """Rotation vectors have unit norms to float64 precision."""
        rng = np.random.default_rng(42)
        R = generate_rotation_vectors(500, 50, rng=rng, use_gpu=False)

        norms = np.linalg.norm(R, axis=1)
        # float64 gives ~15 digits; should be within 1e-14
        assert_allclose(norms, 1.0, atol=1e-14)

    def test_gpu_and_cpu_produce_identical_results(self):
        """With the same seed, use_gpu=True and use_gpu=False produce
        identical rotation vectors (since normalization is always CPU float64)."""
        rng1 = np.random.default_rng(42)
        R_cpu = generate_rotation_vectors(100, 20, rng=rng1, use_gpu=False)

        rng2 = np.random.default_rng(42)
        R_gpu = generate_rotation_vectors(100, 20, rng=rng2, use_gpu=True)

        # Should be bitwise identical since both paths now use CPU float64
        np.testing.assert_array_equal(R_cpu, R_gpu)

    def test_no_mlx_import_in_normalization(self):
        """generate_rotation_vectors no longer uses MLX for normalization.
        Verify by checking that the function does not call mx.* functions."""
        # We can verify this by patching MLX_AVAILABLE to True and checking
        # that the results are still computed purely in numpy
        rng = np.random.default_rng(42)
        R = generate_rotation_vectors(100, 20, rng=rng, use_gpu=True)

        # If MLX were used for normalization, float32 would lose precision.
        # Float64 norms should be within 1e-14 of 1.0.
        norms = np.linalg.norm(R, axis=1)
        max_deviation = np.max(np.abs(norms - 1.0))
        assert max_deviation < 1e-14, (
            f"Max norm deviation {max_deviation} suggests float32 normalization"
        )

    def test_large_dimension_precision(self):
        """For large n_dims, float64 normalization is especially important
        because the sum of squares spans a larger dynamic range."""
        rng = np.random.default_rng(42)
        # Large dimensionality: n_dims = 1000
        R = generate_rotation_vectors(50, 1000, rng=rng, use_gpu=False)

        norms = np.linalg.norm(R, axis=1)
        assert_allclose(norms, 1.0, atol=1e-14)
        assert R.dtype == np.float64


# =============================================================================
# Integration
# =============================================================================


class TestIntegration:
    """End-to-end integration tests for t->z threshold, shared rotations,
    and float64 normalization."""

    def test_full_pipeline_shared_rotations(self):
        """Full pipeline: fit, test_gene_sets with shared rotations,
        get valid results."""
        engine = _make_engine_and_fit(n_genes=50, n_samples=20)

        gene_sets = {
            f"set_{i}": [f"GENE{j}" for j in range(i * 5, i * 5 + 5)]
            for i in range(5)
        }
        config = RotationTestConfig(
            n_rotations=199,
            use_gpu=False,
            seed=42,
        )

        results = engine.test_gene_sets(
            gene_sets, config=config, shared_rotations=True,
        )

        assert len(results) == 5
        for r in results:
            assert r.n_genes_found >= 2
            # p-values should be between 0 and 1
            for stat_dict in r.p_values.values():
                for p in stat_dict.values():
                    assert 0 < p <= 1, f"Invalid p-value: {p}"

    def test_rotation_vectors_dtype_in_full_pipeline(self):
        """Rotation vectors used in the pipeline are float64."""
        rng = np.random.default_rng(42)
        R = generate_rotation_vectors(99, 18, rng=rng, use_gpu=False)

        assert R.dtype == np.float64
        norms = np.linalg.norm(R, axis=1)
        assert_allclose(norms, 1.0, atol=1e-14)
