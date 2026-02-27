"""Tests for STAT-2 (Camera VIF) and STAT-3 (z-score SE) fixes.

Validates that:
- The z-score denominator uses SE = sigma/sqrt(k) (STAT-3)
- The Camera VIF = 1 + (k-1)*rho_bar inflates SE correctly (STAT-2)
- estimate_inter_gene_correlation returns sensible values
- Robust mode applies both SE and VIF corrections
- Backward compatibility is preserved
"""

import numpy as np
import pytest

from cliquefinder.stats.enrichment_z import (
    compute_competitive_z,
    compute_robust_competitive_z,
    estimate_inter_gene_correlation,
)


class TestStat3SEDenominator:
    """STAT-3: z-score uses SE = sigma/sqrt(k) as denominator."""

    def test_z_score_uses_se_not_sigma(self):
        """Z-score denominator is bg_std/sqrt(k), not bg_std."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        # 20 targets with elevated |t|
        t_stats[:20] = rng.normal(3, 0.5, size=20)
        is_target = np.zeros(200, dtype=bool)
        is_target[:20] = True

        z = compute_competitive_z(t_stats, is_target)

        # Compute expected with SE denominator
        abs_t = np.abs(t_stats)
        target_mean = np.mean(abs_t[:20])
        bg_mean = np.mean(abs_t[20:])
        bg_std = np.std(abs_t[20:], ddof=1)
        k = 20
        expected_z = (target_mean - bg_mean) / (bg_std / np.sqrt(k))

        assert z == pytest.approx(expected_z, rel=1e-10)

    def test_z_score_scales_with_sqrt_k(self):
        """Doubling k should increase z by approximately sqrt(2) for same effect size."""
        rng = np.random.default_rng(42)
        n = 1000
        t_stats = rng.normal(0, 1, size=n)

        # 10 targets
        k1 = 10
        t_stats_1 = t_stats.copy()
        t_stats_1[:k1] = 3.0  # constant elevated signal
        is_target_1 = np.zeros(n, dtype=bool)
        is_target_1[:k1] = True

        # 40 targets (4x), same signal magnitude
        k2 = 40
        t_stats_2 = t_stats.copy()
        t_stats_2[:k2] = 3.0
        is_target_2 = np.zeros(n, dtype=bool)
        is_target_2[:k2] = True

        z1 = compute_competitive_z(t_stats_1, is_target_1)
        z2 = compute_competitive_z(t_stats_2, is_target_2)

        # z2/z1 should be approximately sqrt(k2/k1) = sqrt(4) = 2
        # (not exact because background changes slightly)
        ratio = z2 / z1
        assert 1.5 < ratio < 2.5  # approximately 2

    def test_single_target_uses_sigma_not_se(self):
        """With k=1, denominator falls back to sigma (not sigma/sqrt(1))."""
        t_stats = np.array([5.0, 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3])
        is_target = np.array([True, False, False, False, False, False, False, False])

        z = compute_competitive_z(t_stats, is_target)

        # k=1: use bg_std directly (SE of a single observation is sigma)
        abs_t = np.abs(t_stats)
        target_mean = float(abs_t[0])
        bg_mean = float(np.mean(abs_t[1:]))
        bg_std = float(np.std(abs_t[1:], ddof=1))
        expected = (target_mean - bg_mean) / bg_std
        assert z == pytest.approx(expected)


class TestStat2CameraVIF:
    """STAT-2: Camera VIF adjusts for inter-gene correlation."""

    def test_vif_reduces_z_score(self):
        """Positive inter-gene correlation should reduce |z-score|."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        t_stats[:20] = rng.normal(3, 0.5, size=20)
        is_target = np.zeros(200, dtype=bool)
        is_target[:20] = True

        z_no_corr = compute_competitive_z(t_stats, is_target)
        z_with_corr = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=0.1
        )

        assert abs(z_with_corr) < abs(z_no_corr)

    def test_vif_scaling_formula(self):
        """z_vif = z_se / sqrt(VIF) where VIF = 1 + (k-1)*rho_bar."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        t_stats[:20] = rng.normal(3, 0.5, size=20)
        is_target = np.zeros(200, dtype=bool)
        is_target[:20] = True

        rho_bar = 0.15
        k = 20
        vif = 1 + (k - 1) * rho_bar

        z_se = compute_competitive_z(t_stats, is_target)
        z_vif = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=rho_bar
        )

        expected_z_vif = z_se / np.sqrt(vif)
        assert z_vif == pytest.approx(expected_z_vif, rel=1e-10)

    def test_zero_correlation_equals_no_correlation(self):
        """rho_bar=0 gives same result as rho_bar=None (VIF=1)."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:10] = rng.normal(2, 0.5, size=10)
        is_target = np.zeros(100, dtype=bool)
        is_target[:10] = True

        z_none = compute_competitive_z(t_stats, is_target)
        z_zero = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=0.0
        )

        assert z_none == pytest.approx(z_zero)

    def test_negative_correlation_treated_as_zero(self):
        """Negative rho_bar is treated as 0 (no VIF)."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:10] = rng.normal(2, 0.5, size=10)
        is_target = np.zeros(100, dtype=bool)
        is_target[:10] = True

        z_none = compute_competitive_z(t_stats, is_target)
        z_neg = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=-0.05
        )

        # Negative correlation should not inflate VIF
        assert z_neg == pytest.approx(z_none)

    def test_large_rho_dramatically_reduces_z(self):
        """High inter-gene correlation (rho_bar=0.5) greatly reduces z."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        t_stats[:50] = rng.normal(3, 0.5, size=50)
        is_target = np.zeros(200, dtype=bool)
        is_target[:50] = True

        z_no_corr = compute_competitive_z(t_stats, is_target)
        z_high_corr = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=0.5
        )

        # VIF = 1 + 49*0.5 = 25.5, sqrt(VIF) ~ 5.05
        # z should decrease by about 5x
        ratio = z_no_corr / z_high_corr
        expected_ratio = np.sqrt(1 + 49 * 0.5)
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)


class TestRobustModeVIF:
    """Robust mode also applies SE + VIF corrections."""

    def test_robust_se_denominator(self):
        """Robust mode uses MAD/sqrt(k) as denominator."""
        bg = np.array([0.5, 1.0, 1.2, 1.5, 2.0, 0.8, 1.1, 1.3, 1.7, 0.9])
        t_stats = np.concatenate([np.array([3.0, -3.5, 2.8, -4.0, 3.2]), bg])
        is_target = np.zeros(len(t_stats), dtype=bool)
        is_target[:5] = True

        z = compute_competitive_z(t_stats, is_target, robust=True)

        # Manual: median target |t| = 3.2, median bg = 1.15
        # MAD = 1.4826 * 0.30 = 0.44478
        # SE = MAD / sqrt(5) = 0.198918
        k = 5
        mad = 1.4826 * 0.30
        se = mad / np.sqrt(k)
        expected = (3.2 - 1.15) / se
        assert z == pytest.approx(expected, rel=1e-4)

    def test_robust_vif_reduces_z(self):
        """Robust mode with VIF produces smaller |z| than without."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        t_stats[:20] = rng.normal(3, 0.5, size=20)
        is_target = np.zeros(200, dtype=bool)
        is_target[:20] = True

        z_no_corr = compute_competitive_z(t_stats, is_target, robust=True)
        z_with_corr = compute_competitive_z(
            t_stats, is_target, robust=True, inter_gene_correlation=0.1
        )

        assert abs(z_with_corr) < abs(z_no_corr)

    def test_robust_vif_scaling(self):
        """Robust VIF scaling: z_vif = z_se / sqrt(VIF)."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        t_stats[:20] = rng.normal(3, 0.5, size=20)
        is_target = np.zeros(200, dtype=bool)
        is_target[:20] = True

        rho_bar = 0.2
        k = 20
        vif = 1 + (k - 1) * rho_bar

        z_se = compute_competitive_z(t_stats, is_target, robust=True)
        z_vif = compute_competitive_z(
            t_stats, is_target, robust=True, inter_gene_correlation=rho_bar
        )

        expected_z_vif = z_se / np.sqrt(vif)
        assert z_vif == pytest.approx(expected_z_vif, rel=1e-10)

    def test_convenience_wrapper_passes_through_vif(self):
        """compute_robust_competitive_z passes inter_gene_correlation."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:10] = rng.normal(3, 0.5, size=10)
        is_target = np.zeros(100, dtype=bool)
        is_target[:10] = True

        z_direct = compute_competitive_z(
            t_stats, is_target, robust=True, inter_gene_correlation=0.1
        )
        z_wrapper = compute_robust_competitive_z(
            t_stats, is_target, inter_gene_correlation=0.1
        )

        assert z_direct == z_wrapper


class TestEstimateInterGeneCorrelation:
    """Tests for estimate_inter_gene_correlation()."""

    def test_uncorrelated_genes_near_zero(self):
        """Independent genes should give rho_bar near 0."""
        rng = np.random.default_rng(42)
        n_genes = 50
        n_samples = 100
        expression = rng.standard_normal((n_genes, n_samples))
        is_target = np.zeros(n_genes, dtype=bool)
        is_target[:20] = True

        rho_bar = estimate_inter_gene_correlation(expression, is_target)

        # Should be near zero (floor at 0, so >= 0)
        assert rho_bar >= 0.0
        assert rho_bar < 0.1  # statistical noise for 20 genes, 100 samples

    def test_perfectly_correlated_genes(self):
        """Perfectly correlated genes should give rho_bar = 1.0."""
        rng = np.random.default_rng(42)
        n_samples = 50
        # All target genes are the same row (perfect correlation)
        base_row = rng.standard_normal(n_samples)
        expression = np.vstack([
            np.tile(base_row, (5, 1)),  # 5 identical target genes
            rng.standard_normal((10, n_samples)),  # 10 background genes
        ])
        is_target = np.zeros(15, dtype=bool)
        is_target[:5] = True

        rho_bar = estimate_inter_gene_correlation(expression, is_target)

        assert rho_bar == pytest.approx(1.0, abs=1e-10)

    def test_correlated_genes_moderate(self):
        """Moderately correlated genes give expected rho_bar."""
        rng = np.random.default_rng(42)
        n_genes = 10
        n_samples = 500  # many samples for stable estimate
        target_rho = 0.3

        # Generate correlated expression data
        # Use factor model: x_i = sqrt(rho)*z + sqrt(1-rho)*e_i
        z = rng.standard_normal(n_samples)
        expression = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            noise = rng.standard_normal(n_samples)
            expression[i, :] = np.sqrt(target_rho) * z + np.sqrt(1 - target_rho) * noise

        # Add some background genes
        bg = rng.standard_normal((10, n_samples))
        full_expression = np.vstack([expression, bg])
        is_target = np.zeros(20, dtype=bool)
        is_target[:10] = True

        rho_bar = estimate_inter_gene_correlation(full_expression, is_target)

        # Should be close to target_rho
        assert rho_bar == pytest.approx(target_rho, abs=0.05)

    def test_fewer_than_two_targets_returns_zero(self):
        """Fewer than 2 target genes returns 0.0."""
        rng = np.random.default_rng(42)
        expression = rng.standard_normal((10, 50))

        # Zero targets
        is_target_zero = np.zeros(10, dtype=bool)
        assert estimate_inter_gene_correlation(expression, is_target_zero) == 0.0

        # One target
        is_target_one = np.zeros(10, dtype=bool)
        is_target_one[0] = True
        assert estimate_inter_gene_correlation(expression, is_target_one) == 0.0

    def test_negative_correlation_floored_at_zero(self):
        """Negative average correlation is floored at 0."""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Anti-correlated: one gene = -other gene
        expression = np.zeros((4, n_samples))
        expression[0, :] = rng.standard_normal(n_samples)
        expression[1, :] = -expression[0, :]  # perfectly anti-correlated
        expression[2, :] = rng.standard_normal(n_samples)
        expression[3, :] = rng.standard_normal(n_samples)

        is_target = np.array([True, True, False, False])

        rho_bar = estimate_inter_gene_correlation(expression, is_target)
        assert rho_bar == 0.0  # floored at 0


class TestBackwardCompatibility:
    """Ensure existing callers work without changes."""

    def test_no_inter_gene_correlation_param(self):
        """Calling without inter_gene_correlation still works."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:10] = rng.normal(2, 0.5, size=10)
        is_target = np.zeros(100, dtype=bool)
        is_target[:10] = True

        # Should work without the parameter
        z = compute_competitive_z(t_stats, is_target)
        assert isinstance(z, float)
        assert z != 0.0

    def test_robust_wrapper_no_inter_gene_correlation(self):
        """compute_robust_competitive_z works without inter_gene_correlation."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:10] = rng.normal(2, 0.5, size=10)
        is_target = np.zeros(100, dtype=bool)
        is_target[:10] = True

        z = compute_robust_competitive_z(t_stats, is_target)
        assert isinstance(z, float)

    def test_network_enrichment_result_backward_compat(self):
        """NetworkEnrichmentResult works with old constructor (no VIF fields)."""
        from cliquefinder.stats.differential import NetworkEnrichmentResult

        # Old-style construction without VIF fields
        result = NetworkEnrichmentResult(
            observed_mean_abs_t=2.5,
            null_mean=1.0,
            null_std=0.3,
            z_score=5.0,
            empirical_pvalue=0.001,
            n_targets=50,
            n_background=500,
            pct_down=60.0,
            direction_pvalue=0.1,
            mannwhitney_pvalue=0.005,
        )

        # Defaults should apply
        assert result.variance_inflation_factor == 1.0
        assert result.mean_pairwise_correlation == 0.0

        # Dict access should work
        d = result.to_dict()
        assert "variance_inflation_factor" in d
        assert "mean_pairwise_correlation" in d
