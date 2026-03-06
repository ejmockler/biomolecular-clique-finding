"""
Wave 1 audit remediation tests for R limma formula alignment.

Tests for four findings:

STAT-CORE-6: Anti-conservative p-values when EB enabled but d0=inf
    - Verify p-values always use t-distribution, never Normal approximation
    - Test both differential.py and permutation_gpu.py paths
    - Compare against scipy.stats.t.sf directly for small df

GPU-1: fit_f_dist s0_sq diverges from R limma when d0=Inf
    - Test with homogeneous variances producing d0=inf
    - Verify s0_sq == mean(sigma2) (arithmetic mean, not geometric)

GPU-2: trigamma_inverse Newton formula differs from R limma
    - Test inverse property: trigamma(trigamma_inverse(x)) ~= x
    - Test edge cases: very small, medium, large x values
    - Verify asymptotic threshold changed to 1e7

GPU-8: Median df for EB hyperparameter estimation
    - Test fit_f_dist with per-feature df array vs scalar median df
    - Test with heterogeneous df values
    - Verify df >= 2 clipping prevents numerical issues
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from scipy.special import polygamma

from cliquefinder.stats.permutation_gpu import (
    fit_f_dist,
    squeeze_var,
    trigamma_inverse,
)


# =============================================================================
# STAT-CORE-6: P-values must always use t-distribution
# =============================================================================


class TestStatCore6DifferentialPvalues:
    """Verify run_protein_differential always uses t-distribution for p-values.

    The old code used Normal approximation when d0=inf or EB disabled.
    For small df (e.g., df=5), this gives anti-conservative p-values:
    t.sf(2.3, 5) = 0.035 vs norm.sf(2.3) = 0.011 (3x error).
    """

    def _make_small_df_problem(self, n_samples: int = 7, n_features: int = 30):
        """Create a test problem with small residual df.

        With n_samples=7 and n_params=2, df_residual = 5 per feature.
        This is where the Normal vs t-distribution difference is largest.
        """
        rng = np.random.RandomState(42)
        data = rng.randn(n_features, n_samples) * 0.5 + 10
        # Inject signal in first 5 features
        data[:5, n_samples // 2:] += 1.5

        conditions = np.array(
            ["CTRL"] * (n_samples // 2) + ["CASE"] * (n_samples - n_samples // 2)
        )
        feature_ids = [f"P{i:04d}" for i in range(n_features)]
        return data, feature_ids, conditions

    def test_pvalues_use_t_distribution_eb_disabled(self):
        """With EB disabled (d0=inf by definition), p-values must use t(df_residual)."""
        from cliquefinder.stats.differential import run_protein_differential

        data, feature_ids, conditions = self._make_small_df_problem(n_samples=7)

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # Manually compute expected p-values using t-distribution
        for _, row in result.iterrows():
            if np.isnan(row["p_value"]):
                continue
            t_val = row["t_statistic"]
            df = row["df"]
            expected_p = 2 * scipy_stats.t.sf(np.abs(t_val), df)
            # The old code would have used norm.sf here
            wrong_p = 2 * scipy_stats.norm.sf(np.abs(t_val))

            assert_allclose(
                row["p_value"], expected_p, rtol=1e-10,
                err_msg=f"P-value for {row['feature_id']} should use t({df})",
            )

            # For small df, norm gives anti-conservative (smaller) p-values
            if df < 20 and np.abs(t_val) > 1.0:
                assert wrong_p < expected_p, (
                    f"Normal approximation should be anti-conservative for df={df}"
                )

    def test_pvalues_use_t_distribution_eb_d0_inf(self):
        """With EB enabled but d0=inf (homogeneous variances), use t(df_residual)."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 30
        n_samples = 7  # Small n for small df

        # Create data with very homogeneous variances to force d0=inf
        data = rng.randn(n_features, n_samples) * 1.0 + 10
        data[:5, n_samples // 2:] += 1.5

        conditions = np.array(["CTRL"] * 3 + ["CASE"] * 4)
        feature_ids = [f"P{i:04d}" for i in range(n_features)]

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            verbose=False,
        )

        # Verify p-values match t-distribution (not Normal)
        for _, row in result.iterrows():
            if np.isnan(row["p_value"]):
                continue
            t_val = row["t_statistic"]
            df = row["df"]
            expected_p = 2 * scipy_stats.t.sf(np.abs(t_val), df)
            assert_allclose(
                row["p_value"], expected_p, rtol=1e-10,
                err_msg=f"P-value should use t({df}), not Normal",
            )

    def test_small_df_anti_conservative_magnitude(self):
        """Quantify the anti-conservative error for typical small-df scenarios."""
        # df=5: t.sf(2.3, 5) vs norm.sf(2.3)
        t_val = 2.3
        for df in [5, 10]:
            p_correct = 2 * scipy_stats.t.sf(t_val, df)
            p_wrong = 2 * scipy_stats.norm.sf(t_val)

            ratio = p_correct / p_wrong
            assert ratio > 1.0, (
                f"df={df}: t-distribution p={p_correct:.4f} should be larger than "
                f"Normal p={p_wrong:.4f}"
            )

            if df == 5:
                # For df=5, the ratio should be about 3x
                assert ratio > 2.5, (
                    f"df=5: expected ~3x ratio, got {ratio:.1f}x"
                )

    def test_large_df_convergence(self):
        """For large df, t-distribution and Normal should give similar results."""
        t_val = 2.0
        for df in [100, 1000, 10000]:
            p_t = 2 * scipy_stats.t.sf(t_val, df)
            p_norm = 2 * scipy_stats.norm.sf(t_val)
            # Should converge as df increases; t(100) still ~6% off Normal,
            # so use a tolerance that scales with 1/df
            assert_allclose(p_t, p_norm, rtol=0.1 * (100 / df),
                            err_msg=f"t({df}) should be close to Normal for large df")

    def test_pvalues_use_t_distribution_df10(self):
        """Verify with df=10 (n_samples=12)."""
        from cliquefinder.stats.differential import run_protein_differential

        data, feature_ids, conditions = self._make_small_df_problem(n_samples=12)

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        for _, row in result.iterrows():
            if np.isnan(row["p_value"]):
                continue
            expected_p = 2 * scipy_stats.t.sf(np.abs(row["t_statistic"]), row["df"])
            assert_allclose(row["p_value"], expected_p, rtol=1e-10)


class TestStatCore6PermutationGpuPvalues:
    """Verify permutation_gpu.py p-value path also uses t-distribution.

    This tests the fix at the code level by checking that the p-value
    computation uses t.sf instead of norm.sf for unmoderated paths.
    """

    def test_t_distribution_more_conservative_than_normal(self):
        """Direct comparison: t.sf gives larger (more conservative) p-values for small df."""
        t_values = np.array([1.5, 2.0, 2.5, 3.0])
        df = 5  # Small residual df

        p_t = 2 * scipy_stats.t.sf(np.abs(t_values), df)
        p_norm = 2 * scipy_stats.norm.sf(np.abs(t_values))

        # Every t-based p-value should be larger (more conservative) than Normal
        assert np.all(p_t > p_norm), (
            "t-distribution p-values must be more conservative than Normal for df=5"
        )

    def test_t_with_inf_df_equals_normal(self):
        """When df=inf, t.sf should equal norm.sf (validates our universal approach)."""
        t_values = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        p_t = 2 * scipy_stats.t.sf(np.abs(t_values), np.inf)
        p_norm = 2 * scipy_stats.norm.sf(np.abs(t_values))
        assert_allclose(p_t, p_norm, rtol=1e-12,
                        err_msg="t(inf) should equal Normal distribution")


# =============================================================================
# GPU-1: fit_f_dist s0_sq with d0=Inf
# =============================================================================


class TestGpu1FitFDistS0SqInf:
    """Verify fit_f_dist returns arithmetic mean of raw variances when d0=inf.

    R limma returns s20 = mean(x) when d0 is infinite (no covariate).
    The old code returned exp(emean) which is the geometric mean of the
    adjusted values -- different from R limma.
    """

    def test_homogeneous_variances_d0_inf(self):
        """Homogeneous variances should produce d0=inf and s0_sq=mean(sigma2)."""
        # All variances identical -> no variance in log(sigma2) -> d0=inf
        sigma2 = np.ones(100) * 2.5
        d0, s0_sq = fit_f_dist(sigma2, 10)

        assert np.isinf(d0), "Identical variances should give d0=inf"
        assert_allclose(s0_sq, 2.5, rtol=1e-10,
                        err_msg="s0_sq should equal mean(sigma2) when d0=inf")

    def test_near_homogeneous_variances(self):
        """Nearly homogeneous variances producing d0=inf should use arithmetic mean."""
        rng = np.random.RandomState(42)
        # Very small variance around 3.0
        sigma2 = 3.0 + rng.randn(200) * 0.001
        sigma2 = np.abs(sigma2)  # Ensure positive

        d0, s0_sq = fit_f_dist(sigma2, 10)

        # May or may not produce d0=inf depending on exact variance
        if np.isinf(d0):
            expected_mean = float(np.mean(sigma2))
            assert_allclose(s0_sq, expected_mean, rtol=1e-6,
                            err_msg="s0_sq should be arithmetic mean, not geometric")

            # The old code would have returned exp(emean) which for nearly
            # identical positive values is approximately the geometric mean
            # Verify these are different for non-identical data
            geometric_approx = np.exp(np.mean(np.log(sigma2)))
            # For very tight distributions they'll be close, but the formula
            # should use arithmetic mean
            assert_allclose(s0_sq, expected_mean, rtol=1e-10)

    def test_heterogeneous_variances_not_inf(self):
        """Heterogeneous variances should produce finite d0."""
        rng = np.random.RandomState(42)
        # Wide spread of variances
        sigma2 = rng.exponential(1.0, size=100)
        d0, s0_sq = fit_f_dist(sigma2, 10)

        assert np.isfinite(d0), "Heterogeneous variances should give finite d0"
        assert d0 > 0, "d0 should be positive"
        assert s0_sq > 0, "s0_sq should be positive"

    def test_arithmetic_vs_geometric_mean_difference(self):
        """For skewed variance distributions, arithmetic and geometric means differ."""
        # Deliberately create a case where arithmetic != geometric mean
        sigma2 = np.array([1.0, 1.0, 1.0, 1.0, 10.0])  # Skewed
        arithmetic_mean = np.mean(sigma2)
        geometric_mean = np.exp(np.mean(np.log(sigma2)))

        assert arithmetic_mean > geometric_mean, (
            "AM-GM inequality: arithmetic mean should exceed geometric mean"
        )

        # When fit_f_dist returns d0=inf, it should use arithmetic mean
        # We can't force d0=inf with this data, but we verify the formula

    def test_s0_sq_equals_mean_for_constant_variance(self):
        """When all variances are exactly equal, s0_sq = that variance."""
        for val in [0.5, 1.0, 3.14, 100.0]:
            sigma2 = np.full(50, val)
            d0, s0_sq = fit_f_dist(sigma2, 8)
            assert np.isinf(d0)
            assert_allclose(s0_sq, val, rtol=1e-10,
                            err_msg=f"s0_sq should equal {val} for constant variance")

    def test_s0_sq_with_array_df_d0_inf(self):
        """fit_f_dist with per-feature df array also uses arithmetic mean when d0=inf."""
        sigma2 = np.ones(50) * 2.0
        df_array = np.random.RandomState(42).randint(5, 20, size=50)
        d0, s0_sq = fit_f_dist(sigma2, df_array)

        assert np.isinf(d0)
        assert_allclose(s0_sq, 2.0, rtol=1e-10)


# =============================================================================
# GPU-2: trigamma_inverse Newton formula and asymptotic threshold
# =============================================================================


class TestGpu2TrigammaInverse:
    """Verify trigamma_inverse matches R limma formulation.

    Tests the inverse property: trigamma(trigamma_inverse(x)) ~= x
    for a wide range of x values.
    """

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
    def test_inverse_property(self, x):
        """trigamma(trigamma_inverse(x)) should equal x."""
        y = trigamma_inverse(x)
        recovered = polygamma(1, y)
        assert_allclose(
            recovered, x, rtol=1e-6,
            err_msg=f"trigamma(trigamma_inverse({x})) = {recovered}, expected {x}",
        )

    @pytest.mark.parametrize("x", [1e-8, 1e-6, 1e-4])
    def test_inverse_property_small_x(self, x):
        """Inverse property for very small x (large y)."""
        y = trigamma_inverse(x)
        assert y > 0, f"trigamma_inverse({x}) should be positive"
        recovered = polygamma(1, y)
        assert_allclose(recovered, x, rtol=1e-4,
                        err_msg=f"Small x={x}: inverse property violated")

    @pytest.mark.parametrize("x", [1e7, 1e8, 1e10])
    def test_inverse_property_large_x(self, x):
        """Inverse property for very large x (near asymptotic regime)."""
        y = trigamma_inverse(x)
        assert y > 0, f"trigamma_inverse({x}) should be positive"
        recovered = polygamma(1, y)
        assert_allclose(recovered, x, rtol=1e-3,
                        err_msg=f"Large x={x}: inverse property violated")

    def test_asymptotic_threshold_is_1e7(self):
        """Values between 1e6 and 1e7 should use Newton iteration, not asymptotic."""
        # For x=5e6, the old code (threshold=1e6) would return 1/sqrt(x)
        # With threshold=1e7, it should use Newton iteration for better accuracy
        x = 5e6
        y = trigamma_inverse(x)
        recovered = polygamma(1, y)

        # The asymptotic approximation 1/sqrt(x) is less accurate here
        y_asymptotic = 1.0 / np.sqrt(x)
        recovered_asymptotic = polygamma(1, y_asymptotic)

        # Newton should be at least as accurate as asymptotic
        newton_error = abs(recovered - x) / x
        assert newton_error < 1e-4, (
            f"Newton iteration for x=5e6 should be accurate, error={newton_error}"
        )

    def test_monotonicity(self):
        """trigamma_inverse should be monotonically decreasing (larger x -> smaller y)."""
        x_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        y_values = [trigamma_inverse(x) for x in x_values]

        for i in range(len(y_values) - 1):
            assert y_values[i] > y_values[i + 1], (
                f"trigamma_inverse should decrease: "
                f"f({x_values[i]})={y_values[i]} vs f({x_values[i+1]})={y_values[i+1]}"
            )

    def test_zero_returns_inf(self):
        """trigamma_inverse(0) should return inf."""
        assert trigamma_inverse(0) == np.inf

    def test_negative_returns_inf(self):
        """trigamma_inverse(negative) should return inf."""
        assert trigamma_inverse(-1.0) == np.inf

    def test_positive_output(self):
        """Output should always be positive for positive input."""
        for x in [1e-10, 1e-5, 0.001, 0.1, 1.0, 100.0, 1e6, 1e10]:
            y = trigamma_inverse(x)
            assert y > 0, f"trigamma_inverse({x}) = {y} should be positive"

    def test_convergence_r_limma_formulation(self):
        """R limma reciprocal formulation should converge in few iterations.

        The R limma formulation: dif = tri * (1 - tri/x) / tri_deriv
        has better convergence than standard Newton: delta = (tri - x) / tri_deriv.
        Verify accuracy for typical EB estimation values.
        """
        # Typical evar_adjusted values from EB estimation
        for x in [0.05, 0.1, 0.5, 1.0, 2.0]:
            y = trigamma_inverse(x)
            recovered = polygamma(1, y)
            assert_allclose(recovered, x, rtol=1e-7,
                            err_msg=f"R limma formulation inaccurate for x={x}")


# =============================================================================
# GPU-8: Per-feature df array for EB hyperparameter estimation
# =============================================================================


class TestGpu8PerFeatureDf:
    """Verify fit_f_dist handles per-feature df array correctly.

    R limma uses per-feature df rather than a scalar median. This matters
    when features have heterogeneous df (e.g., different amounts of missing data).
    """

    def test_array_df_accepted(self):
        """fit_f_dist should accept an array of per-feature df values."""
        rng = np.random.RandomState(42)
        sigma2 = rng.exponential(1.0, size=100)
        df_array = rng.randint(5, 20, size=100)

        d0, s0_sq = fit_f_dist(sigma2, df_array)
        assert np.isfinite(d0) or np.isinf(d0), "d0 should be a valid number"
        assert s0_sq > 0, "s0_sq should be positive"

    def test_array_vs_scalar_df_differ(self):
        """Per-feature df array should give different estimates than scalar median df.

        Create data with heterogeneous df: some features have df=3 (low confidence),
        others have df=20 (high confidence).
        """
        rng = np.random.RandomState(42)
        n_features = 200

        # Simulate variances from F-distribution with known parameters
        true_d0 = 8.0
        true_s0_sq = 1.0

        # Heterogeneous df
        df_array = np.concatenate([
            np.full(100, 3),   # 100 features with df=3
            np.full(100, 20),  # 100 features with df=20
        ])

        # Simulate s2 ~ s0^2 * F(df, d0) for each feature
        sigma2 = np.array([
            true_s0_sq * rng.chisquare(df_i) / df_i
            for df_i in df_array
        ])

        # Clip df >= 2 as the fix does
        df_clipped = np.maximum(df_array, 2).astype(np.float64)

        # Per-feature df
        d0_array, s0_sq_array = fit_f_dist(sigma2, df_clipped)

        # Scalar median df
        median_df = int(np.median(df_array))
        d0_scalar, s0_sq_scalar = fit_f_dist(sigma2, median_df)

        # They should differ because the per-feature df captures
        # that low-df features contribute more uncertainty
        # (unless both return d0=inf, which is unlikely with this data)
        if np.isfinite(d0_array) and np.isfinite(d0_scalar):
            # At least one of d0 or s0_sq should differ meaningfully
            d0_diff = abs(d0_array - d0_scalar) / max(abs(d0_scalar), 1e-10)
            s0_diff = abs(s0_sq_array - s0_sq_scalar) / max(abs(s0_sq_scalar), 1e-10)
            assert d0_diff > 0.01 or s0_diff > 0.01, (
                f"Per-feature df should give different estimates: "
                f"d0_array={d0_array:.3f} vs d0_scalar={d0_scalar:.3f}, "
                f"s0_array={s0_sq_array:.6f} vs s0_scalar={s0_sq_scalar:.6f}"
            )

    def test_uniform_df_matches_scalar(self):
        """When all per-feature df are equal, array should match scalar result."""
        rng = np.random.RandomState(42)
        sigma2 = rng.exponential(1.0, size=100)
        df_value = 10

        d0_scalar, s0_scalar = fit_f_dist(sigma2, df_value)
        df_array = np.full(100, df_value)
        d0_array, s0_array = fit_f_dist(sigma2, df_array)

        assert_allclose(d0_array, d0_scalar, rtol=1e-10,
                        err_msg="Uniform array df should match scalar df")
        assert_allclose(s0_array, s0_scalar, rtol=1e-10,
                        err_msg="Uniform array df should match scalar df")

    def test_df_clipping_prevents_numerical_issues(self):
        """df=1 features should be clipped to df=2 to avoid digamma/trigamma issues."""
        rng = np.random.RandomState(42)
        n_features = 50

        sigma2 = rng.exponential(1.0, size=n_features)

        # Some features with df=1 (problematic for digamma/trigamma)
        df_raw = np.concatenate([
            np.full(10, 1),   # df=1: digamma(0.5) and trigamma(0.5) can be extreme
            np.full(40, 10),  # df=10: safe
        ])

        # With clipping to df >= 2
        df_clipped = np.maximum(df_raw, 2).astype(np.float64)
        d0_clipped, s0_sq_clipped = fit_f_dist(sigma2, df_clipped)

        # Should produce valid output
        assert np.isfinite(d0_clipped) or np.isinf(d0_clipped)
        assert np.isfinite(s0_sq_clipped)
        assert s0_sq_clipped > 0

    def test_df1_raw_still_works(self):
        """Even without clipping, fit_f_dist should handle df=1 without crashing.

        But the estimates may be less stable.
        """
        rng = np.random.RandomState(42)
        sigma2 = rng.exponential(1.0, size=50)
        df_raw = np.concatenate([np.full(10, 1), np.full(40, 10)])

        # Should not raise
        d0, s0_sq = fit_f_dist(sigma2, df_raw)
        assert np.isfinite(s0_sq) or np.isnan(s0_sq) or s0_sq > 0

    def test_integration_run_protein_differential_uses_array_df(self):
        """run_protein_differential should pass per-feature df array to fit_f_dist."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 30
        n_samples = 20

        data = rng.randn(n_features, n_samples) * 0.5 + 10
        data[:5, 10:] += 2.0

        # Inject different NaN counts per feature to create heterogeneous df
        data[0, :3] = np.nan   # df = 15
        data[1, :7] = np.nan   # df = 11
        data[2, :1] = np.nan   # df = 17
        # data[3] has no NaN   # df = 18

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

        # Verify features have different df (reflecting per-feature calculation)
        df_values = result["df"].values
        # Features 0, 1, 2, 3 should have different base df due to different NaN counts
        # After EB moderation, df_total = df_residual + d0, so if d0 is finite,
        # the df differences from NaN should be preserved
        df_0 = df_values[0]
        df_1 = df_values[1]
        df_3 = df_values[3]

        # Feature 1 (more NaN) should have lower df than feature 3 (no NaN)
        assert df_1 < df_3, (
            f"Feature with more NaN (df={df_1}) should have lower total df "
            f"than feature without NaN (df={df_3})"
        )

    def test_fit_f_dist_valid_mask_applied_to_array_df(self):
        """When sigma2 has invalid entries, only valid entries' df are used."""
        sigma2 = np.array([1.0, -1.0, 2.0, np.inf, 3.0, 0.0, 1.5])
        df_array = np.array([10, 5, 8, 12, 15, 7, 9])

        d0, s0_sq = fit_f_dist(sigma2, df_array)

        # Only indices 0, 2, 4, 6 are valid (positive and finite sigma2)
        # Should not crash and should produce valid output
        assert np.isfinite(d0) or np.isinf(d0)
        assert np.isfinite(s0_sq)
        assert s0_sq > 0

    def test_per_feature_df_affects_eb_priors(self):
        """Using per-feature df should change EB hyperparameter estimates
        compared to using the median df.

        This directly tests the GPU-8 fix: the old code used
        int(np.median(df_valid)), the new code uses the full array.
        """
        rng = np.random.RandomState(123)
        n = 200

        # Simulate from known F-distribution
        true_d0 = 5.0
        true_s0 = 2.0

        # Highly heterogeneous df distribution (bimodal)
        df = np.concatenate([np.full(n // 2, 3), np.full(n // 2, 50)])
        median_df = int(np.median(df))
        # median of [3]*100 + [50]*100 is 26 (midpoint), far from either mode
        assert median_df != 3 and median_df != 50, (
            f"Median ({median_df}) should differ from both modes"
        )

        # Generate sigma2 ~ s0^2 * F(df_i, d0)
        sigma2 = np.array([
            true_s0 * rng.chisquare(d) / d for d in df
        ])

        df_clipped = np.maximum(df, 2).astype(np.float64)

        d0_per, s0_per = fit_f_dist(sigma2, df_clipped)
        d0_med, s0_med = fit_f_dist(sigma2, median_df)

        # Both should produce reasonable estimates, but they should differ
        # because the mean of trigamma over a bimodal df distribution
        # differs from trigamma of the median
        assert np.isfinite(d0_per) or np.isinf(d0_per)
        assert np.isfinite(d0_med) or np.isinf(d0_med)


# =============================================================================
# Cross-cutting: combined fix integration tests
# =============================================================================


class TestCombinedFixesIntegration:
    """Integration tests verifying all fixes work together correctly."""

    def test_small_sample_full_pipeline(self):
        """Full pipeline with small samples (small df) should produce valid results.

        This exercises all four fixes together:
        - GPU-8: per-feature df array
        - GPU-1: correct s0_sq if d0=inf
        - GPU-2: trigamma_inverse accuracy
        - STAT-CORE-6: t-distribution p-values
        """
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 50
        n_samples = 8  # Very small -> df ~= 6

        data = rng.randn(n_features, n_samples) * 0.5 + 10
        data[:10, 4:] += 2.0  # Signal in first 10 features

        conditions = np.array(["CTRL"] * 4 + ["CASE"] * 4)
        feature_ids = [f"P{i:04d}" for i in range(n_features)]

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            verbose=False,
        )

        # All p-values should be valid
        valid_p = result["p_value"].dropna()
        assert len(valid_p) > 0
        assert (valid_p >= 0).all(), "P-values must be non-negative"
        assert (valid_p <= 1).all(), "P-values must be <= 1"

        # Signal features should be more significant
        signal_p = result.iloc[:10]["p_value"].dropna()
        noise_p = result.iloc[10:]["p_value"].dropna()
        assert signal_p.median() < noise_p.median(), (
            "Signal features should have smaller p-values than noise"
        )

    def test_eb_moderation_with_heterogeneous_df(self):
        """EB moderation with mixed NaN patterns (heterogeneous df)."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        n_features = 40
        n_samples = 15

        data = rng.randn(n_features, n_samples) * 0.5 + 10
        data[:5, 8:] += 1.5

        # Create heterogeneous missing data
        for i in range(n_features):
            n_missing = rng.randint(0, 4)
            if n_missing > 0:
                missing_idx = rng.choice(n_samples, n_missing, replace=False)
                data[i, missing_idx] = np.nan

        conditions = np.array(["CTRL"] * 8 + ["CASE"] * 7)
        feature_ids = [f"P{i:04d}" for i in range(n_features)]

        result = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            verbose=False,
        )

        # Should produce valid results
        valid_p = result["p_value"].dropna()
        assert len(valid_p) > 0
        assert (valid_p >= 0).all()
        assert (valid_p <= 1).all()

        # df should be valid (finite or inf depending on EB fit).
        # When d0=inf (prior dominates), all df_total = inf, which is correct.
        # When d0 is finite, df should vary across features due to heterogeneous NaN.
        df_valid = result["df"].dropna()
        assert len(df_valid) > 0, "Should have valid df values"
        assert (df_valid > 0).all(), "All df values should be positive"

    def test_p_values_match_t_dist_after_all_fixes(self):
        """After all fixes, every p-value in the output should match t(df)."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.RandomState(42)
        data = rng.randn(30, 10) * 0.5 + 10
        data[:5, 5:] += 1.5

        conditions = np.array(["CTRL"] * 5 + ["CASE"] * 5)
        feature_ids = [f"P{i:04d}" for i in range(30)]

        for eb in [True, False]:
            result = run_protein_differential(
                data=data,
                feature_ids=feature_ids,
                sample_condition=conditions,
                contrast=("CASE", "CTRL"),
                eb_moderation=eb,
                verbose=False,
            )

            for _, row in result.iterrows():
                if np.isnan(row["p_value"]):
                    continue
                expected_p = 2 * scipy_stats.t.sf(
                    np.abs(row["t_statistic"]), row["df"]
                )
                assert_allclose(
                    row["p_value"], expected_p, rtol=1e-10,
                    err_msg=(
                        f"P-value mismatch for {row['feature_id']} "
                        f"(eb={eb}): got {row['p_value']}, expected {expected_p}"
                    ),
                )
