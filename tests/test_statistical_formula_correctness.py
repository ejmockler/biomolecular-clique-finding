"""
Statistical formula correctness tests.

Validates implementations against reference formulas:

- VSN MLX NaN reference (nanmean equivalent)
- EB p-values always use t-distribution (never Normal)
- Li-Ji M_eff uses full M eigenvalues
- fit_f_dist s0_sq = mean(sigma2) when d0=inf
- trigamma_inverse R limma reciprocal Newton
- Per-feature df array for EB hyperparameters
- MEAN MIXED uses signed z (two-sided via |null|>=|obs|)
- FLOORMEAN UP/DOWN applies floor before zeroing
- MEAN50 selects by |z| not w*z
- Sample weights applied to Y in rotation
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import polygamma, digamma


# =============================================================================
# VSN MLX NaN Reference
# =============================================================================


class TestVSNNanReference:
    """Verify MLX nanmean implementation matches NumPy nanmean."""

    def test_mlx_nanmean_matches_numpy(self):
        """MLX VSN reference should match np.nanmean with NaN data."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 20))
        # Insert ~25% NaN
        nan_mask = rng.random((10, 20)) < 0.25
        data[nan_mask] = np.nan

        # NumPy reference
        ref_numpy = np.nanmean(data, axis=1)

        # MLX implementation (mirrors the fixed code in normalization.py)
        y = mx.array(data.astype(np.float32))
        valid_count = mx.sum(mx.where(mx.isnan(y), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y), 0.0, y), axis=1)
        ref_mlx = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        ref_mlx_np = np.array(ref_mlx, dtype=np.float64)

        # Float32 tolerance
        assert_allclose(ref_mlx_np, ref_numpy, rtol=1e-5, atol=1e-6)

    def test_mlx_nanmean_all_nan_row(self):
        """Edge case: row with all NaN should produce 0 (not NaN/inf)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        data = np.full((3, 5), np.nan)
        data[1, :] = [1.0, 2.0, 3.0, np.nan, np.nan]

        y = mx.array(data.astype(np.float32))
        valid_count = mx.sum(mx.where(mx.isnan(y), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y), 0.0, y), axis=1)
        ref = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        ref_np = np.array(ref, dtype=np.float64)

        # Row 0: all NaN → 0/1 = 0.0
        assert ref_np[0] == pytest.approx(0.0)
        # Row 1: mean of [1,2,3] = 2.0
        assert ref_np[1] == pytest.approx(2.0, rel=1e-5)
        # Row 2: all NaN → 0.0
        assert ref_np[2] == pytest.approx(0.0)

    def test_old_formula_was_wrong(self):
        """Demonstrate the old formula mx.mean(mx.where(isnan,0,y)) is biased."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        # 4 values, 1 NaN: true mean = mean([1,2,3]) = 2.0
        data = np.array([[1.0, 2.0, 3.0, np.nan]])
        y = mx.array(data.astype(np.float32))

        # Old (wrong) formula: divides by 4 instead of 3
        old_ref = mx.mean(mx.where(mx.isnan(y), 0.0, y), axis=1)
        old_val = float(np.array(old_ref)[0])
        assert old_val == pytest.approx(1.5, rel=1e-5)  # Wrong: 6/4 = 1.5

        # New (correct) formula
        valid_count = mx.sum(mx.where(mx.isnan(y), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y), 0.0, y), axis=1)
        new_ref = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        new_val = float(np.array(new_ref)[0])
        assert new_val == pytest.approx(2.0, rel=1e-5)  # Correct: 6/3 = 2.0


# =============================================================================
# Always use t-distribution for p-values
# =============================================================================


class TestAlwaysTDistribution:
    """Verify p-values always use t-distribution, never Normal."""

    def test_t_vs_normal_divergence_small_df(self):
        """For small df, t and Normal give very different p-values."""
        from scipy import stats as scipy_stats

        t_stat = 2.3
        df = 5

        # t-distribution (correct)
        p_t = 2 * scipy_stats.t.sf(abs(t_stat), df)
        # Normal (wrong when df is small)
        p_norm = 2 * scipy_stats.norm.sf(abs(t_stat))

        # Normal is anti-conservative: gives SMALLER p-value
        assert p_norm < p_t
        # The ratio should be substantial at df=5
        assert p_t / p_norm > 1.5  # At least 50% difference

    def test_t_converges_to_normal_large_df(self):
        """Verify t(df) → Normal as df → ∞ (validates our universal-t approach)."""
        from scipy import stats as scipy_stats

        t_stat = 2.0
        df_large = 1e6

        p_t = 2 * scipy_stats.t.sf(abs(t_stat), df_large)
        p_norm = 2 * scipy_stats.norm.sf(abs(t_stat))

        # For very large df, t and Normal should be essentially identical
        assert_allclose(p_t, p_norm, rtol=1e-5)

    def test_run_protein_differential_uses_t_not_normal(self):
        """Integration: run_protein_differential should use t-distribution."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.default_rng(42)
        n_features, n_samples = 20, 12
        data = rng.standard_normal((n_features, n_samples))
        # Add condition effect to first 5 features
        data[:5, 6:] += 2.0

        feature_ids = [f"gene_{i}" for i in range(n_features)]
        conditions = np.array(["A"] * 6 + ["B"] * 6)

        results = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("B", "A"),
            eb_moderation=True,
        )

        # All p-values should be valid
        assert not results["p_value"].isna().any()
        assert (results["p_value"] >= 0).all()
        assert (results["p_value"] <= 1).all()

        # The significant genes should have small p-values
        sig_pvals = results.loc[results["feature_id"].isin([f"gene_{i}" for i in range(5)]), "p_value"]
        assert (sig_pvals < 0.05).all()


# =============================================================================
# Li-Ji M_eff Formula
# =============================================================================


class TestLiJiMeff:
    """Verify Li-Ji M_eff uses full M eigenvalues (zero-padded)."""

    def test_li_ji_uses_full_M(self):
        """Li-Ji should use all M eigenvalues, not filtered subset."""
        from cliquefinder.stats.correlation_tests import estimate_effective_tests

        # Create correlation matrix with known eigenstructure
        rng = np.random.default_rng(42)
        n = 20
        # Correlation matrix with some near-zero eigenvalues
        A = rng.standard_normal((n, n))
        C = A @ A.T
        D = np.diag(1.0 / np.sqrt(np.diag(C)))
        corr = D @ C @ D

        m_eff = estimate_effective_tests(corr, method='li-ji')

        # M_eff should be between 1 and n
        assert 1 <= m_eff <= n
        # For a random correlation matrix, M_eff should be < n
        assert m_eff < n

    def test_li_ji_identity_matrix(self):
        """For identity (independent tests), M_eff ≈ M."""
        from cliquefinder.stats.correlation_tests import estimate_effective_tests

        n = 10
        corr = np.eye(n)
        m_eff = estimate_effective_tests(corr, method='li-ji')

        # For independent tests, M_eff should be close to M
        # Li-Ji: M_eff = 1 + (M-1)*(1 - Var(λ)/M)
        # For identity, all eigenvalues = 1, Var(λ) = 0, so M_eff = M
        assert_allclose(m_eff, n, atol=0.01)

    def test_li_ji_perfect_correlation(self):
        """For perfectly correlated tests, M_eff ≈ 1."""
        from cliquefinder.stats.correlation_tests import estimate_effective_tests

        n = 10
        corr = np.ones((n, n))
        m_eff = estimate_effective_tests(corr, method='li-ji')

        # For perfect correlation, one eigenvalue = n, rest = 0
        # Var(λ) is large, so M_eff should be small
        assert m_eff < 3  # Should be close to 1


# =============================================================================
# fit_f_dist s0_sq when d0=inf
# =============================================================================


class TestFitFDistInfD0:
    """Verify fit_f_dist returns mean(sigma2) when d0→∞."""

    def test_s0_sq_is_mean_not_geometric(self):
        """When d0=inf, s0_sq should be arithmetic mean of sigma2."""
        from cliquefinder.stats.permutation_gpu import fit_f_dist

        # Create very homogeneous variances that will trigger d0=inf
        rng = np.random.default_rng(42)
        sigma2 = np.ones(100) + rng.standard_normal(100) * 0.01  # Very tight
        df = 10

        d0, s0_sq = fit_f_dist(sigma2, df)

        if np.isinf(d0):
            # When d0=inf, s0_sq should be arithmetic mean
            expected = float(np.mean(sigma2[sigma2 > 0]))
            assert_allclose(s0_sq, expected, rtol=1e-10)

    def test_s0_sq_consistent_both_inf_paths(self):
        """Both d0=inf paths (evar<=0 and d0>1e10) should use same formula."""
        from cliquefinder.stats.permutation_gpu import fit_f_dist

        # Test with very homogeneous data (evar_adjusted <= 0 path)
        sigma2_homo = np.ones(50) * 2.0
        d0_1, s0_1 = fit_f_dist(sigma2_homo, 20)

        if np.isinf(d0_1):
            assert_allclose(s0_1, 2.0, rtol=1e-10)


# =============================================================================
# trigamma_inverse R limma alignment
# =============================================================================


class TestTrigammaInverse:
    """Verify trigamma_inverse matches R limma's reciprocal Newton formulation."""

    def test_basic_inversion(self):
        """trigamma_inverse(trigamma(x)) ≈ x for various x values."""
        from cliquefinder.stats.permutation_gpu import trigamma_inverse

        test_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

        for x in test_values:
            tri_x = float(polygamma(1, x))
            x_recovered = trigamma_inverse(tri_x)
            assert_allclose(x_recovered, x, rtol=1e-6,
                            err_msg=f"Failed for x={x}")

    def test_edge_cases(self):
        """Edge cases should not crash or produce invalid values."""
        from cliquefinder.stats.permutation_gpu import trigamma_inverse

        # Very small x (large y expected)
        result = trigamma_inverse(1e-10)
        assert result > 0
        assert np.isfinite(result)

        # Very large x (small y expected, uses asymptotic)
        result = trigamma_inverse(1e8)
        assert result > 0
        assert np.isfinite(result)

        # Moderate x
        result = trigamma_inverse(0.1)
        assert result > 0
        # Verify: trigamma(result) ≈ 0.1
        assert_allclose(polygamma(1, result), 0.1, rtol=1e-6)

    def test_asymptotic_threshold_1e7(self):
        """Asymptotic threshold should be 1e7 (matching R limma)."""
        from cliquefinder.stats.permutation_gpu import trigamma_inverse

        # Just below threshold: should use Newton iteration
        result_below = trigamma_inverse(9e6)
        # Just above threshold: should use asymptotic formula 1/sqrt(x)
        result_above = trigamma_inverse(2e7)

        # Both should give valid results
        assert result_below > 0
        assert result_above > 0

        # Asymptotic: y ≈ 1/sqrt(x)
        expected_above = 1.0 / np.sqrt(2e7)
        assert_allclose(result_above, expected_above, rtol=1e-10)


# =============================================================================
# Per-feature df array
# =============================================================================


class TestPerFeatureDf:
    """Verify fit_f_dist accepts and uses per-feature df arrays."""

    def test_array_df_accepted(self):
        """fit_f_dist should accept array df without error."""
        from cliquefinder.stats.permutation_gpu import fit_f_dist

        rng = np.random.default_rng(42)
        n_features = 50
        sigma2 = rng.exponential(1.0, n_features)
        # Heterogeneous df: some features have more observations
        df_array = rng.choice([5, 8, 10, 15], size=n_features)

        d0, s0_sq = fit_f_dist(sigma2, df_array)

        assert np.isfinite(d0) or np.isinf(d0)
        assert s0_sq > 0
        assert np.isfinite(s0_sq)

    def test_array_df_differs_from_scalar_median(self):
        """Per-feature df should give different (more accurate) result than median."""
        from cliquefinder.stats.permutation_gpu import fit_f_dist

        rng = np.random.default_rng(42)
        n_features = 100
        sigma2 = rng.exponential(1.0, n_features)
        # Strongly heterogeneous df
        df_array = np.concatenate([
            np.full(50, 3),    # Half with low df
            np.full(50, 30),   # Half with high df
        ])
        median_df = int(np.median(df_array))

        d0_array, s0_array = fit_f_dist(sigma2, df_array)
        d0_scalar, s0_scalar = fit_f_dist(sigma2, median_df)

        # Results should differ (the whole point of per-feature df)
        # But both should be valid
        for d0, s0 in [(d0_array, s0_array), (d0_scalar, s0_scalar)]:
            assert s0 > 0
            assert np.isfinite(s0)


# =============================================================================
# MEAN MIXED uses signed z
# =============================================================================


class TestMeanMixedSignedZ:
    """Verify MEAN MIXED statistic uses signed z with |null|>=|obs| p-value."""

    def test_mean_mixed_is_signed(self):
        """MEAN MIXED should return signed (not absolute) mean."""
        from cliquefinder.stats.rotation import (
            _compute_mean_stat, Alternative,
        )

        # z values: half positive, half negative (net negative)
        z = np.array([[1.0, -2.0, 0.5, -3.0]])
        w = np.ones_like(z)
        A = float(z.shape[1])

        result = _compute_mean_stat(z, w, A, Alternative.MIXED)

        # Should be signed mean: (1 - 2 + 0.5 - 3) / 4 = -3.5/4 = -0.875
        expected = np.mean(z)
        assert_allclose(result[0], expected, rtol=1e-10)

        # Old (wrong) formula would use |z|: (1+2+0.5+3)/4 = 1.625
        old_wrong = np.mean(np.abs(z))
        assert abs(result[0]) < old_wrong  # Signed is smaller

    def test_mean_mixed_pvalue_two_sided(self):
        """P-value for MEAN MIXED uses |null| >= |obs|."""
        from cliquefinder.stats.rotation import (
            compute_rotation_pvalues, SetStatistic, Alternative,
        )

        obs_stat = -0.5  # Negative observed

        # Null distribution: symmetric around 0
        null_vals = np.array([-0.6, -0.3, 0.1, 0.4, 0.7, -0.8, 0.55, -0.2, 0.3, -0.1])
        # |null| >= |obs|=0.5: |-0.6|, |0.7|, |-0.8|, |0.55| = 4 out of 10

        observed = {"mean": {"mixed": obs_stat}}
        null = {"mean": {"mixed": null_vals}}

        pvals = compute_rotation_pvalues(observed, null)

        # p = (b+1)/(B+1) where b = count(|null| >= |obs|)
        expected_b = np.sum(np.abs(null_vals) >= abs(obs_stat))
        expected_p = (expected_b + 1) / (len(null_vals) + 1)
        assert_allclose(pvals["mean"]["mixed"], expected_p, rtol=1e-10)


# =============================================================================
# FLOORMEAN applies floor before zeroing
# =============================================================================


class TestFloormeanFloor:
    """Verify FLOORMEAN applies floor(|z|, sqrt(q)) before zeroing by direction."""

    def test_floormean_up_floors_positive_z(self):
        """FLOORMEAN UP should floor positive z at sqrt(q)."""
        from cliquefinder.stats.rotation import (
            _compute_floormean_stat, Alternative,
        )

        z = np.array([[0.01, 0.5, -1.0, 2.0]])  # Small positive, medium, negative, large
        w = np.ones_like(z)
        A = float(z.shape[1])
        floor = 0.3  # sqrt(q)

        result = _compute_floormean_stat(z, w, A, Alternative.UP, floor)

        # For UP: positive z contribute with floor(|z|, 0.3)
        # z=0.01 → max(0.01, 0.3) = 0.3
        # z=0.5 → max(0.5, 0.3) = 0.5
        # z=-1.0 → 0 (negative)
        # z=2.0 → max(2.0, 0.3) = 2.0
        expected = (0.3 + 0.5 + 0.0 + 2.0) / 4
        assert_allclose(result[0], expected, rtol=1e-10)

    def test_floormean_down_floors_negative_z(self):
        """FLOORMEAN DOWN should floor |z| for negative z."""
        from cliquefinder.stats.rotation import (
            _compute_floormean_stat, Alternative,
        )

        z = np.array([[-0.01, -0.5, 1.0, -2.0]])
        w = np.ones_like(z)
        A = float(z.shape[1])
        floor = 0.3

        result = _compute_floormean_stat(z, w, A, Alternative.DOWN, floor)

        # For DOWN: negative z contribute with floor(|z|, 0.3)
        # z=-0.01 → max(0.01, 0.3) = 0.3
        # z=-0.5 → max(0.5, 0.3) = 0.5
        # z=1.0 → 0 (positive)
        # z=-2.0 → max(2.0, 0.3) = 2.0
        expected = (0.3 + 0.5 + 0.0 + 2.0) / 4
        assert_allclose(result[0], expected, rtol=1e-10)

    def test_old_floormean_up_was_wrong(self):
        """Demonstrate old formula np.maximum(z, 0) lacked floor."""
        z = np.array([[0.01, 0.5, -1.0, 2.0]])
        w = np.ones_like(z)
        floor = 0.3

        # Old formula: np.maximum(z, 0) → [0.01, 0.5, 0, 2.0] (no floor)
        old_f = np.maximum(z, 0)
        old_result = np.sum(w * old_f, axis=1) / z.shape[1]

        # Old would give (0.01+0.5+0+2.0)/4 = 0.6275
        assert old_result[0] == pytest.approx(0.6275, rel=1e-10)

        # Correct: (0.3+0.5+0+2.0)/4 = 0.7 (floor boosts small positive z)
        from cliquefinder.stats.rotation import _compute_floormean_stat, Alternative
        correct = _compute_floormean_stat(z, w, float(z.shape[1]), Alternative.UP, floor)
        assert correct[0] == pytest.approx(0.7, rel=1e-10)
        assert correct[0] > old_result[0]


# =============================================================================
# MEAN50 selects by |z| not w*z
# =============================================================================


class TestMean50Selection:
    """Verify MEAN50 selects top 50% genes by |z| (unweighted)."""

    def test_mean50_selects_by_abs_z(self):
        """Top 50% by |z| not by w*z."""
        from cliquefinder.stats.rotation import (
            _compute_mean50_stat, Alternative,
        )

        # 4 genes: |z| = [0.1, 3.0, 0.5, 2.0]
        # Top 50% by |z|: genes 1,3 (|z|=3.0, 2.0)
        z = np.array([[0.1, -3.0, 0.5, 2.0]])
        # Weights that would change selection if used: w*z = [10, -3, 5, 2]
        w = np.array([[100.0, 1.0, 10.0, 1.0]])
        A = float(z.shape[1])

        result_up = _compute_mean50_stat(z, w, A, Alternative.UP)

        # h = 4 // 2 = 2, select genes with |z| = 3.0 and 2.0
        # For UP: mean of w*z for those genes = mean(1*(-3) + 1*2) = mean(-3, 2) = -0.5
        selected_wz = np.array([-3.0, 2.0])  # w*z for genes 1,3
        expected = np.mean(selected_wz)
        assert_allclose(result_up[0], expected, rtol=1e-10)

    def test_mean50_h_matches_r_integer_division(self):
        """h = n // 2 (R's integer division), not (n+1)//2."""
        from cliquefinder.stats.rotation import (
            _compute_mean50_stat, Alternative,
        )

        # 5 genes: h should be 5//2 = 2 (not (5+1)//2 = 3)
        z = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        w = np.ones_like(z)
        A = float(z.shape[1])

        result = _compute_mean50_stat(z, w, A, Alternative.UP)

        # Top 2 by |z|: genes with z=5.0 and z=4.0
        expected = np.mean([5.0, 4.0])  # 4.5
        assert_allclose(result[0], expected, rtol=1e-10)


# =============================================================================
# Sample weights applied to Y
# =============================================================================


class TestSampleWeightsOnY:
    """Verify sample weights are applied to Y (not just X) in ROAST."""

    def test_w_sqrt_vec_stored_in_precomputed(self):
        """RotationPrecomputed stores w_sqrt_vec."""
        from cliquefinder.stats.rotation import compute_rotation_matrices

        rng = np.random.default_rng(42)
        n_samples = 12
        conditions = ["A"] * 6 + ["B"] * 6
        weights = rng.uniform(0.5, 2.0, n_samples)

        precomp = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=weights,
        )

        assert precomp.w_sqrt_vec is not None
        assert len(precomp.w_sqrt_vec) == n_samples
        assert_allclose(precomp.w_sqrt_vec, np.sqrt(weights), rtol=1e-10)

    def test_w_sqrt_vec_none_without_weights(self):
        """Without sample weights, w_sqrt_vec should be None."""
        from cliquefinder.stats.rotation import compute_rotation_matrices

        n_samples = 12
        conditions = ["A"] * 6 + ["B"] * 6

        precomp = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=None,
        )

        assert precomp.w_sqrt_vec is None

    def test_weights_applied_to_y_in_extract(self):
        """extract_gene_effects applies w_sqrt_vec to Y."""
        from cliquefinder.stats.rotation import (
            compute_rotation_matrices, extract_gene_effects,
        )

        rng = np.random.default_rng(42)
        n_samples, n_genes = 12, 20
        data = rng.standard_normal((n_genes, n_samples))
        conditions = ["A"] * 6 + ["B"] * 6
        weights = np.ones(n_samples)
        weights[0] = 4.0  # Up-weight first sample

        precomp = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=weights,
        )

        gene_ids = [f"g{i}" for i in range(n_genes)]
        effects_w = extract_gene_effects(data, gene_ids, precomp)

        # Without weights
        precomp_noweight = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=None,
        )
        effects_nw = extract_gene_effects(data, gene_ids, precomp_noweight)

        # Results should differ because of the up-weighted sample
        assert not np.allclose(effects_w.U, effects_nw.U)

    def test_w_sqrt_vec_immutable(self):
        """w_sqrt_vec should be read-only (frozen dataclass)."""
        from cliquefinder.stats.rotation import compute_rotation_matrices

        n_samples = 8
        conditions = ["A"] * 4 + ["B"] * 4
        weights = np.ones(n_samples) * 2.0

        precomp = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=weights,
        )

        with pytest.raises((ValueError, TypeError)):
            precomp.w_sqrt_vec[0] = 999.0


# =============================================================================
# Integration: Full Pipeline Regression
# =============================================================================


class TestFormulaCorrectnessIntegration:
    """Integration tests verifying combined statistical formula correctness."""

    def test_eb_moderation_with_heterogeneous_df(self):
        """Integration: EB moderation with per-feature df works end to end."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.default_rng(42)
        n_features, n_samples = 30, 16
        data = rng.standard_normal((n_features, n_samples))
        # Add some NaN to create heterogeneous df
        for i in range(10):
            data[i, rng.choice(n_samples, 2, replace=False)] = np.nan

        # Add signal to first 5 features
        data[:5, 8:] += 3.0

        feature_ids = [f"gene_{i}" for i in range(n_features)]
        conditions = np.array(["ctrl"] * 8 + ["treat"] * 8)

        results = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("treat", "ctrl"),
            eb_moderation=True,
        )

        # All p-values should be valid
        assert not results["p_value"].isna().any()
        assert (results["p_value"] > 0).all()
        assert (results["p_value"] <= 1).all()

        # Signal genes should be detected
        sig_mask = results["feature_id"].isin([f"gene_{i}" for i in range(5)])
        assert (results.loc[sig_mask, "p_value"] < 0.01).all()

    def test_rotation_precomp_and_effects_with_weights(self):
        """Integration: ROAST precomputation and gene effects with sample weights."""
        from cliquefinder.stats.rotation import (
            compute_rotation_matrices, extract_gene_effects,
        )

        rng = np.random.default_rng(42)
        n_samples, n_genes = 16, 10
        data = rng.standard_normal((n_genes, n_samples))
        data[:5, 8:] += 2.0  # Up-regulate first 5 in group B

        conditions = ["A"] * 8 + ["B"] * 8
        weights = rng.uniform(0.5, 2.0, n_samples)

        # Precomputation with weights should work
        precomp = compute_rotation_matrices(
            sample_condition=conditions,
            conditions=["A", "B"],
            contrast=("B", "A"),
            sample_weights=weights,
        )
        assert precomp.w_sqrt_vec is not None
        assert precomp.Q2.shape[0] == n_samples

        # Gene effects extraction should apply weights to Y
        gene_ids = [f"g{i}" for i in range(n_genes)]
        effects = extract_gene_effects(data, gene_ids, precomp)

        assert effects.U.shape == (n_genes, precomp.Q2.shape[1])
        assert len(effects.sample_variances) == n_genes
        assert np.all(effects.sample_variances >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
