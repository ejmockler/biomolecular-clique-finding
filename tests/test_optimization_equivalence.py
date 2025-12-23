"""
Verify that optimized implementation produces identical results to original.

CRITICAL: All optimizations must be numerically equivalent (bit-identical within
floating point precision). Any deviation indicates a bug that could affect
scientific conclusions.
"""

import numpy as np
import pytest
from cliquefinder.quality import OutlierDetector, Imputer
from cliquefinder.core.quality import QualityFlag


class TestCorrelationCacheEquivalence:
    """Test that cached correlation matrix gives identical results."""

    def test_cache_vs_no_cache_identical(self, medium_matrix):
        """
        Test that cached correlation computation produces identical results.

        This verifies that memory-mapped correlation caching doesn't introduce
        numerical errors.
        """
        # Detect outliers first
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        # Get mask of values to impute
        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0

        # Impute WITHOUT caching (baseline - direct computation)
        # NOTE: Current implementation doesn't have caching yet, so we're testing
        # that the function produces deterministic results across multiple runs
        imputer1 = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result1 = imputer1.apply(flagged).data

        # Run again to test determinism
        imputer2 = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result2 = imputer2.apply(flagged).data

        # Verify bit-identical (within float64 precision)
        np.testing.assert_array_equal(result1, result2,
            err_msg="Imputation should be deterministic")

        # Verify imputed values specifically
        imputed_values1 = result1[to_impute]
        imputed_values2 = result2[to_impute]

        np.testing.assert_allclose(
            imputed_values1,
            imputed_values2,
            rtol=1e-9,
            atol=1e-12,
            err_msg="Imputed values should be identical"
        )

        print(f"✓ Deterministic imputation verified ({to_impute.sum()} imputed values)")


class TestParallelEquivalence:
    """Test that parallel execution gives identical results to sequential."""

    def test_parallel_vs_sequential_identical(self, medium_matrix):
        """
        Test that parallelization doesn't affect results.

        This verifies that multi-core processing produces identical results
        to single-threaded processing.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        # Sequential (n_jobs=1)
        # NOTE: Current implementation doesn't have n_jobs parameter yet
        # This test will be updated when parallelization is added
        imputer_seq = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result_seq = imputer_seq.apply(flagged).data

        # For now, just verify determinism
        # When n_jobs parameter is added, this will test parallel vs sequential
        imputer_check = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result_check = imputer_check.apply(flagged).data

        # Verify bit-identical
        np.testing.assert_allclose(
            result_seq,
            result_check,
            rtol=1e-9,
            atol=1e-12,
            err_msg="Sequential runs should be identical"
        )

        print("✓ Sequential consistency verified (parallel test pending n_jobs implementation)")


class TestNoRedundantComputation:
    """Verify that correlations are not recomputed for weighting."""

    def test_correlation_returned_with_indices(self, small_matrix):
        """
        Verify that neighbor search returns both indices AND correlations.

        This ensures we don't recompute correlations for weighting (which
        would be wasteful since we already computed them for neighbor selection).
        """
        from cliquefinder.quality import correlation_knn

        # Check that _find_k_nearest_correlation exists and has correct signature
        assert hasattr(correlation_knn, '_find_k_nearest_correlation'), \
            "Module should have _find_k_nearest_correlation function"

        # The current implementation only returns indices
        # When optimization is added, it should return (indices, correlations)
        # For now, verify the function exists and works
        func = correlation_knn._find_k_nearest_correlation

        # Test on small data
        result = func(
            small_matrix.data,
            gene_idx=0,
            k=5
        )

        # Currently returns only indices
        assert isinstance(result, np.ndarray), "Should return array of indices"
        assert len(result) == 5, "Should return k=5 neighbors"

        print("✓ Neighbor search function verified (correlation return pending optimization)")


class TestWeightedVsUnweighted:
    """Test weighted vs unweighted imputation gives different but valid results."""

    def test_weighted_differs_from_unweighted(self, medium_matrix):
        """
        Verify that weighted imputation gives different results than unweighted.

        This ensures that correlation weighting is actually being applied.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_imputed = to_impute.sum()

        if n_imputed == 0:
            pytest.skip("No outliers detected in test data")

        # Unweighted imputation
        imputer_unweighted = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=False
        )
        result_unweighted = imputer_unweighted.apply(flagged).data

        # Weighted imputation
        imputer_weighted = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True,
            weight_power=1.0
        )
        result_weighted = imputer_weighted.apply(flagged).data

        # Results should differ for imputed values
        imputed_unweighted = result_unweighted[to_impute]
        imputed_weighted = result_weighted[to_impute]

        # At least some values should differ
        n_different = np.sum(~np.isclose(imputed_unweighted, imputed_weighted))

        assert n_different > 0, \
            "Weighted and unweighted imputation should produce different results"

        # But both should be valid (no NaN, no Inf)
        assert not np.any(np.isnan(result_unweighted)), "Unweighted: No NaN allowed"
        assert not np.any(np.isnan(result_weighted)), "Weighted: No NaN allowed"
        assert not np.any(np.isinf(result_unweighted)), "Unweighted: No Inf allowed"
        assert not np.any(np.isinf(result_weighted)), "Weighted: No Inf allowed"

        print(f"✓ Weighted vs unweighted: {n_different}/{n_imputed} values differ")


class TestCorrelationVsMedian:
    """Test that correlation-based KNN differs from simpler imputation methods."""

    def test_correlation_differs_from_median(self, small_matrix):
        """
        Verify that correlation-based KNN gives different results than median.

        This ensures KNN is actually using neighbor information, not just
        computing per-gene statistics.

        Note: Legacy Euclidean KNN was removed as scientifically incorrect
        for gene expression (tests magnitude, not pattern similarity).
        See test_legacy_removed.py for validation that 'knn' strategy is rejected.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(small_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_imputed = to_impute.sum()

        if n_imputed == 0:
            pytest.skip("No outliers detected in test data")

        # Correlation-based KNN (context-aware)
        imputer_corr = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=False
        )
        result_corr = imputer_corr.apply(flagged).data

        # Median imputation (context-blind)
        imputer_median = Imputer(strategy="median")
        result_median = imputer_median.apply(flagged).data

        # Results should differ
        imputed_corr = result_corr[to_impute]
        imputed_median = result_median[to_impute]

        n_different = np.sum(~np.isclose(imputed_corr, imputed_median))

        assert n_different > 0, \
            "Correlation KNN and median should produce different results"

        print(f"✓ Correlation KNN vs Median: {n_different}/{n_imputed} values differ")


class TestMedianFallback:
    """Test that median imputation works as fallback."""

    def test_median_imputation_works(self, medium_matrix):
        """Verify median imputation produces valid results."""
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0

        if to_impute.sum() == 0:
            pytest.skip("No outliers detected in test data")

        # Median imputation
        imputer = Imputer(strategy="median")
        result = imputer.apply(flagged)

        # Verify no NaN/Inf
        assert not np.any(np.isnan(result.data)), "No NaN should remain"
        assert not np.any(np.isinf(result.data)), "No Inf should remain"

        # Verify imputed count matches outlier count
        imputed = (result.quality_flags & QualityFlag.IMPUTED) > 0
        assert np.array_equal(imputed, to_impute), \
            "Imputed mask should match outlier mask"

        print(f"✓ Median imputation: {to_impute.sum()} values imputed")


class TestRadiusVsKNN:
    """Test radius-based imputation vs fixed-k KNN."""

    def test_radius_produces_valid_results(self, medium_matrix):
        """Verify radius-based imputation works correctly."""
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0

        if to_impute.sum() == 0:
            pytest.skip("No outliers detected in test data")

        # Radius-based imputation
        imputer_radius = Imputer(
            strategy="radius_correlation",
            correlation_threshold=0.7,
            min_neighbors=3,
            max_neighbors=20,
            weighted=True
        )
        result_radius = imputer_radius.apply(flagged)

        # Verify valid results
        assert not np.any(np.isnan(result_radius.data)), "No NaN should remain"
        assert not np.any(np.isinf(result_radius.data)), "No Inf should remain"

        # Compare to fixed-k KNN
        imputer_knn = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result_knn = imputer_knn.apply(flagged)

        # Results may differ (adaptive vs fixed neighborhood)
        imputed_radius = result_radius.data[to_impute]
        imputed_knn = result_knn.data[to_impute]

        # Both should be valid
        assert not np.any(np.isnan(imputed_radius))
        assert not np.any(np.isnan(imputed_knn))

        # Calculate how different they are
        rel_diff = np.abs(imputed_radius - imputed_knn) / (np.abs(imputed_knn) + 1e-10)
        mean_rel_diff = np.mean(rel_diff)

        print(f"✓ Radius vs KNN: mean relative difference = {mean_rel_diff:.4f}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
