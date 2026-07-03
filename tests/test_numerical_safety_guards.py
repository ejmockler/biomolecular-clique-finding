"""
Tests for numerical safety guards across statistical modules.

Covers:
- binomtest: deprecated binom_test replaced with scipy.stats.binomtest.
- logsumexp NaN propagation: NaN values excluded, all-NaN columns -> NaN.
- AFT imputation: uniform draw lower bound > 0 prevents ppf(0) = -inf.
- Mixed model fallback df: residual df uses n_obs - n_params (not off-by-one).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from scipy.special import logsumexp


# =============================================================================
# binomtest replaces deprecated binom_test
# =============================================================================


class TestBinomtestReplacement:
    """Verify binom_test is fully replaced with binomtest."""

    def test_binomtest_called_not_binom_test(self):
        """Ensure run_network_enrichment_test uses binomtest, not binom_test."""
        from cliquefinder.stats.differential import run_network_enrichment_test

        # Create minimal protein_results DataFrame
        rng = np.random.default_rng(42)
        n_total = 100
        n_targets = 20
        t_stats = rng.standard_normal(n_total)
        is_target = np.zeros(n_total, dtype=bool)
        is_target[:n_targets] = True

        protein_results = pd.DataFrame({
            't_statistic': t_stats,
            'is_target': is_target,
        })

        # Patch binomtest to track calls
        with patch.object(scipy_stats, 'binomtest', wraps=scipy_stats.binomtest) as mock_binomtest:
            result = run_network_enrichment_test(
                protein_results,
                n_permutations=100,
                seed=42,
                verbose=False,
            )
            # binomtest must have been called
            assert mock_binomtest.call_count == 1, (
                f"binomtest should be called exactly once, got {mock_binomtest.call_count}"
            )

    def test_binomtest_result_matches_expected(self):
        """Verify binomtest produces correct p-value for known inputs."""
        from cliquefinder.stats.differential import run_network_enrichment_test

        # Construct the data FIRST so the oracle can derive its null
        # probability from the same background the production code uses.
        # 15 of 20 targets are negative; the code (differential.py:1891-1894)
        # tests that against the observed genome-wide background
        # downregulation rate, NOT against a naive 50%.
        rng = np.random.default_rng(99)
        n_background = 80
        t_stats_targets = np.concatenate([
            -np.abs(rng.standard_normal(15)),  # 15 negative
            np.abs(rng.standard_normal(5)),     # 5 positive
        ])
        t_stats_background = rng.standard_normal(n_background)
        t_stats = np.concatenate([t_stats_targets, t_stats_background])
        is_target = np.concatenate([
            np.ones(20, dtype=bool),
            np.zeros(n_background, dtype=bool),
        ])

        protein_results = pd.DataFrame({
            't_statistic': t_stats,
            'is_target': is_target,
        })

        # Independent hand-computed oracle: mirror the production null
        # probability (differential.py:1888-1889) — the background
        # downregulation rate = 34/80 = 0.425 for rng=default_rng(99).
        n_negative = 15
        n_total = 20
        n_background_down = int(np.sum(t_stats_background < 0))
        background_down_rate = n_background_down / n_background
        expected_result = scipy_stats.binomtest(
            n_negative, n_total, p=background_down_rate, alternative='two-sided'
        )

        result = run_network_enrichment_test(
            protein_results,
            n_permutations=100,
            seed=42,
            verbose=False,
        )

        assert_allclose(result.direction_pvalue, expected_result.pvalue, rtol=1e-10)

    def test_no_binom_test_in_source(self):
        """Static check: binom_test should not appear in the source code
        (except in comments documenting the deprecation fix)."""
        import inspect
        from cliquefinder.stats import differential

        source = inspect.getsource(differential)
        # Count non-comment occurrences of binom_test (not binomtest)
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                continue
            # Check for binom_test but not binomtest
            if 'binom_test' in stripped and 'binomtest' not in stripped:
                pytest.fail(
                    f"Found deprecated binom_test in non-comment code: {stripped}"
                )


# =============================================================================
# logsumexp NaN propagation
# =============================================================================


class TestLogsumexpNanHandling:
    """Verify logsumexp handles NaN values correctly."""

    def test_logsumexp_with_nan_values(self):
        """NaN values should be excluded from the logsumexp sum."""
        from cliquefinder.stats.summarization import summarize_to_protein

        # 3 features, 4 samples; feature 0 sample 2 is NaN
        data = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [2.0, 3.0, 5.0,   1.0],
            [3.0, 1.0, 2.0,   3.0],
        ])

        result = summarize_to_protein(data, method="logsum")

        # Expected: logsumexp over non-NaN values per column
        # Column 0: logsumexp([1, 2, 3]) - all present
        # Column 1: logsumexp([2, 3, 1]) - all present
        # Column 2: logsumexp([-inf, 5, 2]) = logsumexp([5, 2]) - NaN masked
        # Column 3: logsumexp([4, 1, 3]) - all present
        expected_col0 = logsumexp([1.0, 2.0, 3.0])
        expected_col1 = logsumexp([2.0, 3.0, 1.0])
        expected_col2 = logsumexp([5.0, 2.0])  # NaN excluded
        expected_col3 = logsumexp([4.0, 1.0, 3.0])

        expected = np.array([expected_col0, expected_col1, expected_col2, expected_col3])
        assert_allclose(result, expected, rtol=1e-12)

    def test_logsumexp_all_nan_column_returns_nan(self):
        """If all values in a column are NaN, result should be NaN."""
        from cliquefinder.stats.summarization import summarize_to_protein

        data = np.array([
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
        ])

        result = summarize_to_protein(data, method="logsum")

        # Column 0: logsumexp([1, 2]) - valid
        # Column 1: all NaN -> should be NaN
        # Column 2: logsumexp([3, 4]) - valid
        assert np.isfinite(result[0])
        assert np.isnan(result[1]), "All-NaN column should produce NaN"
        assert np.isfinite(result[2])

    def test_logsumexp_no_nan_unchanged(self):
        """Without NaN, result should be identical to raw logsumexp."""
        from cliquefinder.stats.summarization import summarize_to_protein

        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        result = summarize_to_protein(data, method="logsum")
        expected = logsumexp(data, axis=0)
        assert_allclose(result, expected, rtol=1e-14)

    def test_logsumexp_single_non_nan_per_column(self):
        """Column with one valid value should return that value."""
        from cliquefinder.stats.summarization import summarize_to_protein

        data = np.array([
            [np.nan, 2.0],
            [3.0,    np.nan],
        ])

        result = summarize_to_protein(data, method="logsum")

        # Column 0: only 3.0 -> logsumexp([3.0]) = 3.0
        # Column 1: only 2.0 -> logsumexp([2.0]) = 2.0
        assert_allclose(result[0], 3.0, rtol=1e-14)
        assert_allclose(result[1], 2.0, rtol=1e-14)


# =============================================================================
# AFT imputation -inf draw prevention
# =============================================================================


class TestAftInfPrevention:
    """Verify AFT imputation never draws -inf from norm.ppf."""

    def test_uniform_draw_lower_bound_positive(self):
        """The uniform draw lower bound must be > 0 to prevent ppf(0) = -inf."""
        from cliquefinder.stats.missing import impute_aft_model

        # Create data with missing values that would trigger AFT imputation
        rng = np.random.default_rng(42)
        n_features, n_samples = 5, 20
        data = rng.standard_normal((n_features, n_samples)) + 5.0  # Moderate shift

        # Introduce missing values at low end
        threshold = 5.0
        data[data < threshold] = np.nan

        # Ensure we have some missing values
        assert np.any(np.isnan(data)), "Test data must have NaN values"

        result = impute_aft_model(data, censoring_threshold=threshold, random_state=42)

        # All imputed values must be finite (no -inf)
        assert np.all(np.isfinite(result.data)), (
            f"Found non-finite values in imputed data: "
            f"{result.data[~np.isfinite(result.data)]}"
        )

    def test_ppf_always_finite_with_tiny_lower_bound(self):
        """Direct test: ppf(tiny) should be finite, ppf(0) would be -inf."""
        tiny = np.finfo(np.float64).tiny  # ~2.2e-308

        # ppf(0) is -inf
        assert np.isinf(scipy_stats.norm.ppf(0.0)), "ppf(0) should be -inf"

        # ppf(tiny) is finite (approximately -37.5)
        result = scipy_stats.norm.ppf(tiny)
        assert np.isfinite(result), f"ppf(tiny) should be finite, got {result}"
        assert result < -30, f"ppf(tiny) should be very negative, got {result}"

    def test_aft_many_draws_all_finite(self):
        """With many draws (stress test), all values must remain finite."""
        from cliquefinder.stats.missing import impute_aft_model

        rng = np.random.default_rng(123)
        n_features, n_samples = 20, 50
        data = rng.standard_normal((n_features, n_samples)) + 5.0

        # Make ~40% missing (aggressive censoring)
        threshold = 5.0
        data[data < threshold] = np.nan
        n_missing = np.sum(np.isnan(data))
        assert n_missing > 100, f"Expected >100 missing values, got {n_missing}"

        result = impute_aft_model(data, censoring_threshold=threshold, random_state=0)

        assert np.all(np.isfinite(result.data)), (
            f"Found {np.sum(~np.isfinite(result.data))} non-finite values "
            f"out of {result.data.size} total"
        )

    def test_aft_imputed_values_below_threshold(self):
        """AFT imputed values should be at or below the censoring threshold."""
        from cliquefinder.stats.missing import impute_aft_model

        rng = np.random.default_rng(77)
        n_features, n_samples = 10, 30
        data = rng.standard_normal((n_features, n_samples)) + 8.0
        threshold = 7.0
        data[data < threshold] = np.nan

        assert np.any(np.isnan(data)), "Need missing values for this test"

        result = impute_aft_model(data, censoring_threshold=threshold, random_state=77)

        # Check imputed positions: all should be <= threshold
        mask = np.isnan(data)
        imputed_values = result.data[mask]
        assert np.all(imputed_values <= threshold + 1e-10), (
            f"Some imputed values exceed threshold: "
            f"max={np.max(imputed_values)}, threshold={threshold}"
        )


# =============================================================================
# Mixed model fallback df off-by-one
# =============================================================================


class TestMixedModelFallbackDf:
    """Verify the residual df computation in fit_linear_model's mixed model path."""

    def test_residual_df_no_extra_subtraction(self):
        """residual_df should be max(n_groups - n_fixed, n_obs - n_fixed),
        not max(n_groups - n_fixed, n_obs - n_fixed - 1)."""
        from cliquefinder.stats.differential import fit_linear_model

        # Create data with repeated measures that will trigger mixed model
        rng = np.random.default_rng(42)
        n_subjects = 6
        n_conditions = 2
        n_reps = 3  # replicates per subject per condition

        conditions = []
        subjects = []
        y_values = []

        for subj_idx in range(n_subjects):
            for cond_idx in range(n_conditions):
                for _ in range(n_reps):
                    conditions.append(f"cond_{cond_idx}")
                    subjects.append(f"subj_{subj_idx}")
                    y_values.append(rng.standard_normal() + cond_idx * 2.0)

        y = np.array(y_values)
        condition = np.array(conditions)
        subject = np.array(subjects)

        result = fit_linear_model(y, condition, subject, use_mixed=True)
        coef_df, model_type, residual_var, subject_var, converged, reason, cov_params, residual_df, n_obs_used, n_groups = result

        if converged and model_type.name == "MIXED":
            n_obs = n_subjects * n_conditions * n_reps  # 36
            n_fixed = len(coef_df)  # number of fixed effect params (intercept + condition dummies)

            # The correct formula: max(n_groups - n_fixed, n_obs - n_fixed)
            expected_df = max(n_groups - n_fixed, n_obs - n_fixed)

            # The old (wrong) formula: max(n_groups - n_fixed, n_obs - n_fixed - 1)
            wrong_df = max(n_groups - n_fixed, n_obs - n_fixed - 1)

            assert residual_df == expected_df, (
                f"residual_df={residual_df} should be {expected_df} "
                f"(n_obs={n_obs}, n_fixed={n_fixed}, n_groups={n_groups}). "
                f"Old wrong value would have been {wrong_df}."
            )
            # Verify the fix actually changed something (off-by-one matters)
            assert expected_df == wrong_df + 1, (
                "Expected the fix to differ by exactly 1 from the old formula"
            )
        else:
            pytest.skip("Mixed model did not converge; cannot test fallback df")

    def test_residual_df_formula_directly(self):
        """Unit test the formula in isolation, independent of mixed model fitting."""
        # Simulate known values
        n_obs = 36
        n_fixed = 2  # intercept + 1 condition dummy
        n_groups = 6

        # Correct formula (after fix)
        correct_df = max(n_groups - n_fixed, n_obs - n_fixed)
        assert correct_df == max(4, 34) == 34

        # Old formula (before fix) was off by 1
        old_df = max(n_groups - n_fixed, n_obs - n_fixed - 1)
        assert old_df == max(4, 33) == 33

        # The difference should be exactly 1
        assert correct_df - old_df == 1

    def test_residual_df_small_n_groups_path(self):
        """When n_groups - n_fixed dominates, the fix still applies correctly."""
        # Scenario: many groups, few fixed params, few obs per group
        n_obs = 10
        n_fixed = 8  # many condition dummies
        n_groups = 10

        # Correct: max(10 - 8, 10 - 8) = max(2, 2) = 2
        correct_df = max(n_groups - n_fixed, n_obs - n_fixed)
        assert correct_df == 2

        # Old: max(10 - 8, 10 - 8 - 1) = max(2, 1) = 2
        old_df = max(n_groups - n_fixed, n_obs - n_fixed - 1)
        assert old_df == 2

        # In this case both paths give the same result (n_groups path dominates)
        # This is expected - the fix only matters when n_obs - n_fixed path dominates
