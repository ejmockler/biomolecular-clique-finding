"""
Tests for GPU numerics correctness:

STAT-1-OPT: MLX fast path for batched OLS
    - Single-pattern (complete data) uses MLX when available
    - Mixed-pattern data falls through to NumPy per-pattern loop
    - Single-pattern with uniform NaN still uses MLX fast path
    - Fallback to NumPy when MLX is unavailable
    - MLX and NumPy paths produce identical results

W1-SQUEEZE-VAR: Unified squeeze_var signature
    - Scalar df (backward compatibility)
    - Array df (new capability)
    - Mixed df values produce same results as per-element scalar calls
    - Integration: run_protein_differential uses per-feature df
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats

# MLX availability detection (mirrors production code)
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from cliquefinder.stats.permutation_gpu import squeeze_var, fit_f_dist
from cliquefinder.stats.differential import (
    batched_ols_gpu,
    build_contrast_matrix,
    MLX_AVAILABLE as DIFF_MLX_AVAILABLE,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_ols_problem(
    n_samples: int = 20,
    n_features: int = 50,
    n_conditions: int = 2,
    nan_fraction: float = 0.0,
    nan_pattern: str = "random",
    seed: int = 42,
) -> dict:
    """Build a reproducible OLS test problem.

    Parameters
    ----------
    nan_pattern
        "random"  : each feature gets random NaN positions
        "uniform" : all features share the *same* NaN rows
        "none"    : no NaN at all (complete data)
    """
    rng = np.random.RandomState(seed)

    # Conditions and design
    cond_labels = []
    for i in range(n_conditions):
        cond_labels += [f"C{i}"] * (n_samples // n_conditions)
    # Pad if n_samples not evenly divisible
    while len(cond_labels) < n_samples:
        cond_labels.append(f"C{n_conditions - 1}")
    conditions = sorted(set(cond_labels))

    # Dummy-coded design matrix
    n_params = n_conditions  # intercept + (n_cond - 1) dummies
    X = np.zeros((n_samples, n_params))
    X[:, 0] = 1.0
    for i, cond in enumerate(cond_labels):
        idx = conditions.index(cond)
        if idx > 0:
            X[i, idx] = 1.0

    # Simulate Y = X @ true_beta + noise
    true_beta = rng.randn(n_params, n_features) * 2
    noise = rng.randn(n_samples, n_features) * 0.5
    Y = X @ true_beta + noise

    # Inject NaN
    if nan_fraction > 0:
        if nan_pattern == "random":
            mask = rng.rand(n_samples, n_features) < nan_fraction
            Y[mask] = np.nan
        elif nan_pattern == "uniform":
            # Same rows missing across ALL features
            row_mask = rng.rand(n_samples) < nan_fraction
            Y[row_mask, :] = np.nan

    # Contrast: C1 - C0
    contrast_matrix, contrast_names = build_contrast_matrix(
        conditions,
        {f"{conditions[-1]}_vs_{conditions[0]}": (conditions[-1], conditions[0])},
    )

    feature_ids = [f"F{i:04d}" for i in range(n_features)]

    return dict(
        Y=Y,
        X=X,
        conditions=conditions,
        feature_ids=feature_ids,
        contrast_matrix=contrast_matrix,
        contrast_names=contrast_names,
        true_beta=true_beta,
    )


def _run_numpy_reference(prob: dict) -> list:
    """Run batched_ols_gpu with MLX forcibly disabled (NumPy-only path)."""
    with patch("cliquefinder.stats.differential.MLX_AVAILABLE", False):
        return batched_ols_gpu(
            Y=prob["Y"],
            X=prob["X"],
            conditions=prob["conditions"],
            feature_ids=prob["feature_ids"],
            contrast_matrix=prob["contrast_matrix"],
            contrast_names=prob["contrast_names"],
        )


def _run_mlx_path(prob: dict) -> list:
    """Run batched_ols_gpu with MLX enabled (if available)."""
    if not MLX_AVAILABLE:
        pytest.skip("MLX not available")
    with patch("cliquefinder.stats.differential.MLX_AVAILABLE", True):
        return batched_ols_gpu(
            Y=prob["Y"],
            X=prob["X"],
            conditions=prob["conditions"],
            feature_ids=prob["feature_ids"],
            contrast_matrix=prob["contrast_matrix"],
            contrast_names=prob["contrast_names"],
        )


# =============================================================================
# STAT-1-OPT: MLX Fast Path Tests
# =============================================================================


class TestBatchedOlsGpuPerPattern:
    """Core correctness tests for per-pattern NaN grouping (STAT-1 fix)."""

    def test_complete_data_matches_statsmodels(self):
        """With no NaN, batched OLS should match individual OLS fits."""
        prob = _make_ols_problem(n_samples=30, n_features=10, nan_fraction=0.0)
        results = _run_numpy_reference(prob)

        # Verify against manual per-feature OLS
        for j, result in enumerate(results):
            assert result.convergence, f"Feature {j} did not converge"
            y = prob["Y"][:, j]
            X = prob["X"]

            # Solve OLS manually
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta_hat
            df = len(y) - X.shape[1]
            sigma2 = np.sum(resid ** 2) / df

            # Compare residual variance
            assert_allclose(
                result.residual_variance, sigma2, rtol=1e-10,
                err_msg=f"Feature {j} residual variance mismatch",
            )

    def test_mixed_nan_patterns_correct(self):
        """Features with different NaN patterns get separate (X'X)^-1."""
        prob = _make_ols_problem(
            n_samples=20, n_features=10, nan_fraction=0.15, nan_pattern="random",
        )
        results = _run_numpy_reference(prob)

        # Verify against manual per-feature OLS (dropping NaN rows individually)
        for j, result in enumerate(results):
            if not result.convergence:
                continue
            y = prob["Y"][:, j]
            X = prob["X"]

            valid = ~np.isnan(y)
            y_v = y[valid]
            X_v = X[valid, :]

            if len(y_v) <= X_v.shape[1]:
                continue

            beta_hat = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
            resid = y_v - X_v @ beta_hat
            df = len(y_v) - X_v.shape[1]
            sigma2 = np.sum(resid ** 2) / df

            assert_allclose(
                result.residual_variance, sigma2, rtol=1e-10,
                err_msg=f"Feature {j} residual variance mismatch (mixed NaN)",
            )

    def test_uniform_nan_pattern_correct(self):
        """All features with same NaN rows — single pattern group."""
        prob = _make_ols_problem(
            n_samples=20, n_features=10, nan_fraction=0.1, nan_pattern="uniform",
        )
        results = _run_numpy_reference(prob)

        for j, result in enumerate(results):
            if not result.convergence:
                continue
            y = prob["Y"][:, j]
            X = prob["X"]

            valid = ~np.isnan(y)
            y_v = y[valid]
            X_v = X[valid, :]

            beta_hat = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
            resid = y_v - X_v @ beta_hat
            df = len(y_v) - X_v.shape[1]
            sigma2 = np.sum(resid ** 2) / df

            assert_allclose(
                result.residual_variance, sigma2, rtol=1e-10,
                err_msg=f"Feature {j} residual variance mismatch (uniform NaN)",
            )

    def test_insufficient_data_handled(self):
        """Features with too many NaN (< 3 valid obs) are marked as failed."""
        prob = _make_ols_problem(n_samples=6, n_features=5, nan_fraction=0.0)
        # Make feature 2 have almost all NaN
        prob["Y"][:5, 2] = np.nan  # Only 1 valid observation left

        results = _run_numpy_reference(prob)
        assert not results[2].convergence
        assert "Insufficient" in results[2].issue

    def test_per_feature_df_varies_with_nan(self):
        """Features with different NaN counts should have different df."""
        prob = _make_ols_problem(n_samples=20, n_features=5, nan_fraction=0.0)
        # Inject different amounts of NaN per feature
        prob["Y"][:2, 0] = np.nan   # feature 0: 18 valid, df=16
        prob["Y"][:5, 1] = np.nan   # feature 1: 15 valid, df=13
        # feature 2: 20 valid, df=18 (no NaN)
        prob["Y"][:3, 3] = np.nan   # feature 3: 17 valid, df=15

        results = _run_numpy_reference(prob)
        dfs = [r.contrasts[0].df for r in results if r.convergence]

        # Should have varying df values
        assert len(set(dfs)) > 1, "Expected different df for features with different NaN counts"


class TestMLXFastPath:
    """Tests for MLX GPU acceleration fast path."""

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
    def test_mlx_matches_numpy_complete_data(self):
        """MLX fast path produces near-identical results to NumPy for complete data.

        MLX uses float32 internally for some operations, so we allow
        slightly looser tolerance than pure NumPy comparisons.
        """
        prob = _make_ols_problem(n_samples=30, n_features=20, nan_fraction=0.0)
        results_np = _run_numpy_reference(prob)
        results_mlx = _run_mlx_path(prob)

        assert len(results_np) == len(results_mlx)
        for j, (r_np, r_mlx) in enumerate(zip(results_np, results_mlx)):
            assert r_np.convergence == r_mlx.convergence
            if not r_np.convergence:
                continue
            assert_allclose(
                r_mlx.residual_variance, r_np.residual_variance, rtol=1e-5,
                err_msg=f"Feature {j} residual variance MLX vs NumPy",
            )
            for c_np, c_mlx in zip(r_np.contrasts, r_mlx.contrasts):
                # MLX uses float32 for some intermediate operations,
                # so tolerance must accommodate ~1e-4 relative error.
                assert_allclose(c_mlx.log2_fc, c_np.log2_fc, rtol=5e-4, atol=1e-6)
                assert_allclose(c_mlx.t_value, c_np.t_value, rtol=5e-4, atol=1e-5)
                assert_allclose(c_mlx.p_value, c_np.p_value, rtol=1e-3, atol=1e-8)

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
    def test_mlx_matches_numpy_uniform_nan(self):
        """MLX fast path handles uniform NaN (single pattern) correctly."""
        prob = _make_ols_problem(
            n_samples=30, n_features=20, nan_fraction=0.1, nan_pattern="uniform",
        )
        results_np = _run_numpy_reference(prob)
        results_mlx = _run_mlx_path(prob)

        for j, (r_np, r_mlx) in enumerate(zip(results_np, results_mlx)):
            assert r_np.convergence == r_mlx.convergence
            if not r_np.convergence:
                continue
            assert_allclose(
                r_mlx.residual_variance, r_np.residual_variance, rtol=1e-5,
                err_msg=f"Feature {j} residual variance MLX vs NumPy (uniform NaN)",
            )

    @pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
    def test_mixed_pattern_skips_mlx(self):
        """Mixed NaN patterns should still produce correct results (NumPy fallback)."""
        prob = _make_ols_problem(
            n_samples=30, n_features=20, nan_fraction=0.1, nan_pattern="random",
        )
        results_mlx = _run_mlx_path(prob)
        results_np = _run_numpy_reference(prob)

        # Results should be numerically identical — mixed patterns don't use MLX
        for j, (r_np, r_mlx) in enumerate(zip(results_np, results_mlx)):
            assert r_np.convergence == r_mlx.convergence
            if not r_np.convergence:
                continue
            assert_allclose(
                r_mlx.residual_variance, r_np.residual_variance, rtol=1e-10,
                err_msg=f"Feature {j}: mixed-pattern should be identical",
            )

    def test_numpy_fallback_when_mlx_unavailable(self):
        """Function works correctly when MLX is not available."""
        prob = _make_ols_problem(n_samples=20, n_features=10, nan_fraction=0.0)
        results = _run_numpy_reference(prob)

        # All features should converge
        converged = [r for r in results if r.convergence]
        assert len(converged) == 10

        # Results should be numerically valid
        for r in converged:
            assert np.isfinite(r.residual_variance)
            assert r.residual_variance > 0
            for c in r.contrasts:
                assert np.isfinite(c.t_value)
                assert 0 <= c.p_value <= 1

    def test_single_feature_complete(self):
        """Edge case: single feature, no NaN."""
        prob = _make_ols_problem(n_samples=20, n_features=1, nan_fraction=0.0)
        results = _run_numpy_reference(prob)
        assert len(results) == 1
        assert results[0].convergence

    def test_single_feature_with_nan(self):
        """Edge case: single feature with some NaN."""
        prob = _make_ols_problem(n_samples=20, n_features=1, nan_fraction=0.0)
        prob["Y"][:3, 0] = np.nan
        results = _run_numpy_reference(prob)
        assert len(results) == 1
        assert results[0].convergence
        assert results[0].n_observations == 17


# =============================================================================
# W1-SQUEEZE-VAR: Unified squeeze_var Tests
# =============================================================================


class TestSqueezeVarScalar:
    """Backward-compatibility tests: scalar df (original API)."""

    def test_basic_shrinkage(self):
        """Scalar df produces correct posterior variance."""
        sigma2 = np.array([0.5, 1.0, 2.0, 4.0])
        df = 10
        d0 = 5.0
        s0_sq = 1.5

        s2_post, df_total = squeeze_var(sigma2, df, d0, s0_sq)

        expected = (d0 * s0_sq + df * sigma2) / (d0 + df)
        assert_allclose(s2_post, expected)
        assert df_total == d0 + df
        assert isinstance(df_total, float)

    def test_infinite_d0_no_shrinkage(self):
        """When d0=inf, original variances are returned unchanged."""
        sigma2 = np.array([0.5, 1.0, 2.0])
        df = 10
        d0 = np.inf
        s0_sq = 1.0

        s2_post, df_total = squeeze_var(sigma2, df, d0, s0_sq)

        assert_allclose(s2_post, sigma2)
        assert df_total == float(df)
        # Should be a copy, not the same object
        assert s2_post is not sigma2

    def test_high_d0_shrinks_toward_prior(self):
        """Large d0 pulls posterior toward s0_sq."""
        sigma2 = np.array([0.1, 10.0])
        df = 5
        d0 = 1000.0
        s0_sq = 1.0

        s2_post, _ = squeeze_var(sigma2, df, d0, s0_sq)

        # Both should be close to s0_sq = 1.0
        assert_allclose(s2_post, 1.0, atol=0.1)

    def test_return_type_scalar_df(self):
        """Scalar df returns float df_total."""
        sigma2 = np.array([1.0])
        s2_post, df_total = squeeze_var(sigma2, 10, 5.0, 1.0)
        assert isinstance(df_total, float)


class TestSqueezeVarArray:
    """New capability tests: array df."""

    def test_array_df_basic(self):
        """Array df produces per-feature posterior variances."""
        sigma2 = np.array([0.5, 1.0, 2.0, 4.0])
        df = np.array([8, 10, 12, 15], dtype=np.float64)
        d0 = 5.0
        s0_sq = 1.5

        s2_post, df_total = squeeze_var(sigma2, df, d0, s0_sq)

        expected_var = (d0 * s0_sq + df * sigma2) / (d0 + df)
        expected_df = d0 + df
        assert_allclose(s2_post, expected_var)
        assert_allclose(df_total, expected_df)
        assert isinstance(df_total, np.ndarray)

    def test_array_df_matches_scalar_loop(self):
        """Array df gives identical results to calling scalar df in a loop."""
        rng = np.random.RandomState(123)
        sigma2 = rng.exponential(1.0, size=20)
        df_array = rng.randint(5, 20, size=20).astype(np.float64)
        d0 = 8.0
        s0_sq = 1.2

        # Array call
        s2_post_arr, df_total_arr = squeeze_var(sigma2, df_array, d0, s0_sq)

        # Scalar loop
        s2_post_loop = np.empty(20)
        df_total_loop = np.empty(20)
        for i in range(20):
            s2_i, df_i = squeeze_var(
                sigma2[i:i + 1], int(df_array[i]), d0, s0_sq,
            )
            s2_post_loop[i] = float(s2_i[0])
            df_total_loop[i] = float(df_i)

        assert_allclose(s2_post_arr, s2_post_loop, rtol=1e-14)
        assert_allclose(df_total_arr, df_total_loop, rtol=1e-14)

    def test_array_df_with_infinite_d0(self):
        """Array df with d0=inf returns original variances and array df."""
        sigma2 = np.array([1.0, 2.0, 3.0])
        df = np.array([5.0, 10.0, 15.0])
        d0 = np.inf
        s0_sq = 1.0

        s2_post, df_total = squeeze_var(sigma2, df, d0, s0_sq)

        assert_allclose(s2_post, sigma2)
        assert isinstance(df_total, np.ndarray)
        assert_allclose(df_total, df)

    def test_mixed_df_values(self):
        """Features with very different df get appropriately different shrinkage."""
        sigma2 = np.array([2.0, 2.0])  # Same sample variance
        df = np.array([3.0, 100.0])    # Very different df
        d0 = 5.0
        s0_sq = 1.0

        s2_post, _ = squeeze_var(sigma2, df, d0, s0_sq)

        # Low-df feature gets more shrinkage toward prior (s0_sq=1.0)
        # High-df feature keeps closer to sample variance (2.0)
        assert s2_post[0] < s2_post[1], (
            "Low-df feature should be shrunk more toward prior"
        )
        # Low-df: (5*1 + 3*2)/(5+3) = 11/8 = 1.375
        # High-df: (5*1 + 100*2)/(5+100) = 205/105 ≈ 1.952
        assert_allclose(s2_post[0], 11.0 / 8.0, rtol=1e-14)
        assert_allclose(s2_post[1], 205.0 / 105.0, rtol=1e-14)

    def test_return_type_array_df(self):
        """Array df returns NDArray df_total."""
        sigma2 = np.array([1.0, 2.0])
        df = np.array([5.0, 10.0])
        s2_post, df_total = squeeze_var(sigma2, df, 3.0, 1.0)
        assert isinstance(df_total, np.ndarray)
        assert df_total.dtype == np.float64


class TestSqueezeVarIntegration:
    """Integration test: squeeze_var used in run_protein_differential."""

    def test_per_feature_df_in_protein_differential(self):
        """run_protein_differential uses per-feature df when features have NaN."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 30
        n_samples = 20

        # Create data: features x samples (note: transposed from OLS convention)
        data = rng.randn(n_features, n_samples) * 0.5 + 10
        # Add differential signal
        data[:5, 10:] += 2.0

        # Inject NaN in different amounts per feature
        data[0, :3] = np.nan    # 3 NaN
        data[1, :7] = np.nan    # 7 NaN
        # data[2] has no NaN

        conditions = np.array(["CTRL"] * 10 + ["CASE"] * 10)
        feature_ids = [f"P{i:04d}" for i in range(n_features)]

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            verbose=False,
        )

        # Features with different NaN counts should have different df
        df_values = result["df"].values
        # Feature 0 has 3 NaN (17 valid, df=15+d0)
        # Feature 1 has 7 NaN (13 valid, df=11+d0)
        # Feature 2 has 0 NaN (20 valid, df=18+d0)
        # They should differ because the base df (before +d0) differs
        # (unless d0 is very large, making differences negligible)
        assert result["df"].iloc[0] != result["df"].iloc[2], (
            "Features with different NaN counts should have different total df"
        )

    def test_eb_disabled_uses_per_feature_df(self):
        """Without EB moderation, df should still be per-feature."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 20
        n_samples = 20

        data = rng.randn(n_features, n_samples) * 0.5 + 10
        data[0, :5] = np.nan  # 5 NaN → 15 valid → df=13
        # Feature 1: no NaN → 20 valid → df=18

        conditions = np.array(["CTRL"] * 10 + ["CASE"] * 10)
        feature_ids = [f"P{i:04d}" for i in range(n_features)]

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # df should differ between feature 0 and feature 1
        df_0 = result["df"].iloc[0]
        df_1 = result["df"].iloc[1]
        assert df_0 < df_1, (
            f"Feature with NaN should have lower df ({df_0}) than "
            f"feature without ({df_1})"
        )


# =============================================================================
# Cross-cutting: batched_ols_gpu contrast & t-statistic tests
# =============================================================================


class TestBatchedOlsContrasts:
    """Verify contrasts and t-statistics from batched OLS."""

    def test_known_differential_detected(self):
        """Features with injected signal should have significant t-values."""
        rng = np.random.RandomState(42)
        n_samples = 40
        n_features = 20

        conditions = ["C0"] * 20 + ["C1"] * 20
        X = np.zeros((n_samples, 2))
        X[:, 0] = 1.0
        X[20:, 1] = 1.0

        # Inject strong signal in features 0-4
        Y = rng.randn(n_samples, n_features) * 0.3
        Y[20:, :5] += 3.0  # Strong effect

        contrast_matrix, contrast_names = build_contrast_matrix(
            ["C0", "C1"], {"C1_vs_C0": ("C1", "C0")}
        )

        results = _run_numpy_reference(dict(
            Y=Y, X=X, conditions=["C0", "C1"],
            feature_ids=[f"F{i}" for i in range(n_features)],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        ))

        # First 5 features should be significant
        for j in range(5):
            r = results[j]
            assert r.convergence
            assert abs(r.contrasts[0].t_value) > 5.0, (
                f"Feature {j} should be highly significant"
            )
            assert r.contrasts[0].p_value < 0.001

        # Remaining features should be non-significant
        for j in range(10, n_features):
            r = results[j]
            assert r.convergence
            assert abs(r.contrasts[0].t_value) < 5.0

    def test_log2fc_sign_correct(self):
        """log2FC sign matches the direction of the injected signal."""
        rng = np.random.RandomState(42)
        n_samples = 30

        conditions = ["C0"] * 15 + ["C1"] * 15
        X = np.zeros((n_samples, 2))
        X[:, 0] = 1.0
        X[15:, 1] = 1.0

        Y = rng.randn(n_samples, 2) * 0.1
        Y[15:, 0] += 2.0   # C1 > C0 for feature 0
        Y[15:, 1] -= 2.0   # C1 < C0 for feature 1

        contrast_matrix, contrast_names = build_contrast_matrix(
            ["C0", "C1"], {"C1_vs_C0": ("C1", "C0")}
        )

        results = _run_numpy_reference(dict(
            Y=Y, X=X, conditions=["C0", "C1"],
            feature_ids=["up", "down"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        ))

        assert results[0].contrasts[0].log2_fc > 0, "Upregulated feature should have positive FC"
        assert results[1].contrasts[0].log2_fc < 0, "Downregulated feature should have negative FC"
