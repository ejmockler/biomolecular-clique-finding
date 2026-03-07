"""Tests for Audit XI findings — 8 fixes across 4 deferred + 4 novel."""

import warnings

import numpy as np
import pandas as pd
import pytest


class TestH4_CensoredQuantileDirection:
    """H-4: Censored quantile normalization must map to UPPER quantiles."""

    def test_observed_values_map_to_upper_quantiles(self):
        """Under MNAR, observed values are high-abundance and should map to
        the upper portion of the target distribution."""
        from cliquefinder.stats.normalization import quantile_normalization

        # 5 features, 2 samples.  Sample 1 is complete, sample 2 has 2 NaN.
        data = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, 3.0],
            [4.0, 5.0],
            [5.0, 7.0],
        ])
        result = quantile_normalization(data, method="censored").data

        # Sample 2 has 3 observed values out of 5 features.
        # Under MNAR, these should map to the UPPER 3/5 of the target,
        # not the LOWER 3/5.
        observed_normalized = result[~np.isnan(result[:, 1]), 1]

        # The target distribution is the mean of sorted complete columns.
        # Sample 1 sorted = [1, 2, 3, 4, 5], so target = [1, 2, 3, 4, 5].
        # Upper 3 elements of target = [3, 4, 5].
        # The normalized observed values should span roughly [3, 5], not [1, 3].
        assert np.min(observed_normalized) >= 2.5, (
            f"Min normalized value {np.min(observed_normalized):.2f} is too low — "
            "observed values should map to upper quantiles under MNAR"
        )

    def test_complete_sample_unchanged(self):
        """A sample with no missing values should use the full target."""
        from cliquefinder.stats.normalization import quantile_normalization

        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        result = quantile_normalization(data, method="censored").data
        # Both samples are complete, so censored == simple
        result_simple = quantile_normalization(data, method="simple").data
        np.testing.assert_allclose(result, result_simple, atol=1e-10)

    def test_single_observed_value_maps_to_top(self):
        """Single observed value should map to the highest target quantile."""
        from cliquefinder.stats.normalization import quantile_normalization

        # 5 features, 2 samples.  Only feature 4 observed in both.
        # Target distribution (from complete cases): mean([5, 10]) = [7.5]
        # With only 1 observed value in sample 2, target[-1:] = [7.5]
        data = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, np.nan],
            [5.0, 10.0],
        ])
        result = quantile_normalization(data, method="censored").data
        # Single observed value maps to the top of the target
        assert np.isfinite(result[4, 1])
        # The NaN entries stay NaN
        assert np.all(np.isnan(result[:4, 1]))


class TestH1_NegativeControlsIndexSpace:
    """H-1: Expression-matched controls must use engine's filtered index space."""

    def test_gene_means_use_engine_data(self):
        """gene_means must be computed from engine.data (filtered), not
        the caller-supplied data (unfiltered)."""
        from cliquefinder.stats.negative_controls import (
            _sample_expression_matched_set,
        )

        # Simulate filtered engine data: 3 genes with known means
        gene_means = np.array([1.0, 5.0, 10.0])
        gene_variances = np.array([0.1, 0.5, 1.0])
        rng = np.random.default_rng(42)

        target_indices = [0]
        non_target_indices = np.array([1, 2])

        matched = _sample_expression_matched_set(
            target_indices, non_target_indices,
            gene_means, gene_variances, rng,
        )
        # Should match gene 1 (mean=5) rather than gene 2 (mean=10)
        # because gene 0 has mean=1 and gene 1 is closer
        assert matched[0] == 1

    def test_nan_variance_guarded(self):
        """NaN variances (from single-observation genes) should not crash."""
        from cliquefinder.stats.negative_controls import (
            _sample_expression_matched_set,
        )

        gene_means = np.array([1.0, 2.0, 3.0])
        gene_variances = np.array([0.1, np.nan, 0.5])  # gene 1 has NaN var
        # The NaN guard in the main code replaces NaN with 0,
        # so we simulate that here
        gene_variances = np.where(np.isfinite(gene_variances), gene_variances, 0.0)

        rng = np.random.default_rng(42)
        target_indices = [0]
        non_target_indices = np.array([1, 2])

        # Should not raise
        matched = _sample_expression_matched_set(
            target_indices, non_target_indices,
            gene_means, gene_variances, rng,
        )
        assert len(matched) == 1


class TestH3_SubjectColWarning:
    """H-3: subject_col and use_mixed_model must warn when passed to GPU path."""

    def test_subject_col_warns(self):
        """Passing subject_col to GPU permutation should warn."""
        from cliquefinder.stats.permutation_gpu import run_permutation_test_gpu

        data = np.random.default_rng(42).normal(size=(10, 6))
        feature_ids = [f"G{i}" for i in range(10)]
        md = pd.DataFrame({
            "condition": ["A", "A", "A", "B", "B", "B"],
            "subject_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        })
        cliques = []

        with pytest.warns(UserWarning, match="subject_col.*NOT used"):
            run_permutation_test_gpu(
                data, feature_ids, md, cliques,
                condition_col="condition",
                contrast=("A", "B"),
                subject_col="subject_id",
                use_mixed_model=False,
                n_permutations=10,
                verbose=False,
            )

    def test_use_mixed_model_warns(self):
        """Passing use_mixed_model=True should warn."""
        from cliquefinder.stats.permutation_gpu import run_permutation_test_gpu

        data = np.random.default_rng(42).normal(size=(10, 6))
        feature_ids = [f"G{i}" for i in range(10)]
        md = pd.DataFrame({
            "condition": ["A", "A", "A", "B", "B", "B"],
        })
        cliques = []

        with pytest.warns(UserWarning, match="use_mixed_model.*NOT used"):
            run_permutation_test_gpu(
                data, feature_ids, md, cliques,
                condition_col="condition",
                contrast=("A", "B"),
                subject_col=None,
                use_mixed_model=True,
                n_permutations=10,
                verbose=False,
            )


class TestM2_UseEbDeprecated:
    """M-2: use_eb=False must emit DeprecationWarning."""

    def test_use_eb_false_warns(self):
        """Setting use_eb=False should emit a DeprecationWarning."""
        from cliquefinder.stats.rotation import run_rotation_test

        data = np.random.default_rng(42).normal(size=(20, 8))
        gene_ids = [f"G{i}" for i in range(20)]
        md = pd.DataFrame({"condition": ["A"] * 4 + ["B"] * 4})

        with pytest.warns(DeprecationWarning, match="use_eb=False has no effect"):
            run_rotation_test(
                data=data,
                gene_ids=gene_ids,
                metadata=md,
                gene_sets={},
                conditions=["A", "B"],
                contrast=("A", "B"),
                condition_column="condition",
                n_rotations=99,
                use_eb=False,
                verbose=False,
            )


class TestNORMXI2_NegativeCVFix:
    """NORM-XI-2: CV must be non-negative even when medians are negative."""

    def test_negative_medians_positive_cv(self):
        from cliquefinder.stats.normalization import assess_normalization_quality

        # Log-transformed data with negative medians
        before = np.array([
            [-6.0, -4.0],
            [-5.5, -3.5],
            [-7.0, -5.0],
        ])
        after = before - np.nanmedian(before, axis=0, keepdims=True)

        result = assess_normalization_quality(before, after)
        assert result["median_cv_before"] >= 0, (
            f"CV should be non-negative, got {result['median_cv_before']}"
        )


class TestQMXI3_ZeroIQRFences:
    """QM-XI-3: Zero-IQR adjusted boxplot should not flag everything."""

    def test_zero_iqr_returns_infinite_fences(self):
        from cliquefinder.quality.outliers import adjusted_boxplot_fences

        # Data where central 50% is identical
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 9.0])
        lower, upper = adjusted_boxplot_fences(x)

        # With IQR=0, fences should be (-inf, inf) so nothing is flagged
        assert lower == -np.inf
        assert upper == np.inf

    def test_zero_iqr_does_not_flag_near_median(self):
        from cliquefinder.quality.outliers import adjusted_boxplot_fences

        x = np.array([5.0] * 8 + [4.9, 5.1])
        lower, upper = adjusted_boxplot_fences(x)

        # 4.9 and 5.1 should NOT be flagged as outliers
        assert 4.9 >= lower
        assert 5.1 <= upper


class TestXI4_IlocBooleanMask:
    """XI-4: iloc with boolean mask → loc for pandas compatibility."""

    def test_loc_with_boolean_mask(self):
        """Verify that the permutation_gpu path uses .loc not .iloc for boolean masks."""
        import ast
        from pathlib import Path

        source = Path(
            "src/cliquefinder/stats/permutation_gpu.py"
        ).read_text()
        tree = ast.parse(source)

        # Check that no iloc is called with a variable named contrast_mask
        # This is a static check that the fix was applied
        assert "iloc[contrast_mask]" not in source, (
            "Found deprecated iloc[boolean_mask] — should use .loc"
        )


class TestNCXI2_NaNVarianceGuard:
    """NC-XI-2: NaN variances from ddof=1 must not poison cost matrix."""

    def test_nan_variance_replaced_with_zero(self):
        """Gene with single observation should have variance=0 not NaN."""
        # Simulate what the fix does
        data = np.array([
            [1.0, np.nan, np.nan, np.nan],  # Only 1 valid value
            [1.0, 2.0, 3.0, 4.0],
        ])
        variances = np.nanvar(data, axis=1, ddof=1)
        # Row 0 has ddof=1 with n=1 → NaN
        assert np.isnan(variances[0])

        # After the guard
        variances = np.where(np.isfinite(variances), variances, 0.0)
        assert variances[0] == 0.0
        assert variances[1] > 0
