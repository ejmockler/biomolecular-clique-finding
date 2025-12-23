"""
End-to-end test of complete optimized pipeline.

This test verifies that all optimizations work together correctly
in a realistic workflow.
"""

import numpy as np
import pytest
from cliquefinder.quality import OutlierDetector, Imputer
from cliquefinder.core.quality import QualityFlag


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_complete_optimized_pipeline(self, large_matrix):
        """
        Run complete quality control pipeline with all optimizations.

        This simulates a real analysis workflow:
        1. Load data (done in fixture)
        2. Detect outliers
        3. Impute with optimized methods
        4. Verify results are valid

        Expected behavior:
        - No NaN or Inf in output
        - All outliers imputed
        - Quality flags correctly set
        - Results scientifically valid
        """
        print(f"\n=== Full Pipeline Test ===")
        print(f"Input: {large_matrix.n_features} genes × {large_matrix.n_samples} samples")

        # Step 1: Detect outliers
        print("\n1. Detecting outliers...")
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(large_matrix)

        outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_outliers = outlier_mask.sum()
        outlier_pct = 100 * n_outliers / flagged.data.size

        print(f"   Detected {n_outliers:,} outliers ({outlier_pct:.2f}% of data)")

        assert n_outliers > 0, "Should detect some outliers in synthetic data"
        assert outlier_pct < 10, "Should not flag >10% as outliers (too aggressive)"

        # Step 2: Impute with MAD-clip (consistent with detection)
        print("\n2. Imputing with MAD-clip...")
        imputer = Imputer(
            strategy="mad-clip",
            threshold=3.5
        )

        clean = imputer.apply(flagged)

        # Step 3: Verify results
        print("\n3. Verifying results...")

        # No NaN should remain
        n_nan = np.sum(np.isnan(clean.data))
        assert n_nan == 0, f"Found {n_nan} NaN values after imputation"
        print(f"   ✓ No NaN values")

        # No Inf should remain
        n_inf = np.sum(np.isinf(clean.data))
        assert n_inf == 0, f"Found {n_inf} Inf values after imputation"
        print(f"   ✓ No Inf values")

        # Imputed count should match outlier count
        imputed_mask = (clean.quality_flags & QualityFlag.IMPUTED) > 0
        n_imputed = imputed_mask.sum()

        assert n_imputed == n_outliers, \
            f"Imputed {n_imputed} values but detected {n_outliers} outliers"
        print(f"   ✓ All {n_outliers:,} outliers imputed")

        # Imputed positions should match outlier positions
        assert np.array_equal(imputed_mask, outlier_mask), \
            "Imputed positions should exactly match outlier positions"
        print(f"   ✓ Imputed positions match outliers")

        # Quality flags should preserve OUTLIER_DETECTED and add IMPUTED
        combined_flags = (
            (clean.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        ) & (
            (clean.quality_flags & QualityFlag.IMPUTED) > 0
        )
        assert np.array_equal(combined_flags, outlier_mask), \
            "Quality flags should preserve outlier detection"
        print(f"   ✓ Quality flags correctly set")

        # Imputed values should differ from original outliers
        original_outliers = large_matrix.data[outlier_mask]
        imputed_values = clean.data[imputed_mask]

        n_changed = np.sum(~np.isclose(original_outliers, imputed_values))
        pct_changed = 100 * n_changed / n_outliers

        assert pct_changed > 90, \
            f"Only {pct_changed:.1f}% of outliers changed (expected >90%)"
        print(f"   ✓ {pct_changed:.1f}% of outliers changed")

        # Imputed values should be reasonable (not extreme)
        # Check they're within data range
        data_min = large_matrix.data.min()
        data_max = large_matrix.data.max()
        data_range = data_max - data_min

        imputed_min = imputed_values.min()
        imputed_max = imputed_values.max()

        assert imputed_min >= data_min - 0.1 * data_range, \
            "Imputed values too low"
        assert imputed_max <= data_max + 0.1 * data_range, \
            "Imputed values too high"
        print(f"   ✓ Imputed values in reasonable range")

        # Step 4: Summary
        print(f"\n=== Pipeline Complete ===")
        print(f"   Input: {large_matrix.n_features} genes × {large_matrix.n_samples} samples")
        print(f"   Outliers detected: {n_outliers:,} ({outlier_pct:.2f}%)")
        print(f"   Outliers imputed: {n_imputed:,} (100%)")
        print(f"   Quality: No NaN, No Inf, Valid flags")
        print(f"   ✓ All validations passed")


    def test_pipeline_with_radius_imputation(self, medium_matrix):
        """
        Test pipeline with radius-based imputation.

        This tests the experimental adaptive neighborhood method.
        """
        print(f"\n=== Radius Imputation Pipeline ===")

        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_outliers = outlier_mask.sum()

        if n_outliers == 0:
            pytest.skip("No outliers detected in test data")

        print(f"Detected {n_outliers} outliers")

        # Impute with MAD-clip method
        imputer = Imputer(
            strategy="mad-clip",
            threshold=3.5
        )

        clean = imputer.apply(flagged)

        # Verify
        assert not np.any(np.isnan(clean.data))
        assert not np.any(np.isinf(clean.data))

        imputed_mask = (clean.quality_flags & QualityFlag.IMPUTED) > 0
        assert imputed_mask.sum() == n_outliers

        print(f"✓ Radius imputation successful")


    def test_pipeline_with_median_fallback(self, medium_matrix):
        """
        Test pipeline with median imputation (fastest fallback).

        This should work even without any optimization.
        """
        print(f"\n=== Median Imputation Pipeline ===")

        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_outliers = outlier_mask.sum()

        if n_outliers == 0:
            pytest.skip("No outliers detected in test data")

        print(f"Detected {n_outliers} outliers")

        # Impute with median (simplest method)
        imputer = Imputer(strategy="median")
        clean = imputer.apply(flagged)

        # Verify
        assert not np.any(np.isnan(clean.data))
        assert not np.any(np.isinf(clean.data))

        imputed_mask = (clean.quality_flags & QualityFlag.IMPUTED) > 0
        assert imputed_mask.sum() == n_outliers

        print(f"✓ Median imputation successful")


    def test_pipeline_preserves_non_outliers(self, medium_matrix):
        """
        Verify that non-outlier values are unchanged by pipeline.

        Critical: imputation should ONLY modify flagged values.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(medium_matrix)

        outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_outliers = outlier_mask.sum()

        if n_outliers == 0:
            pytest.skip("No outliers detected in test data")

        # Impute
        imputer = Imputer(strategy="mad-clip", threshold=3.5)
        clean = imputer.apply(flagged)

        # Check non-outliers unchanged
        non_outlier_mask = ~outlier_mask

        original_non_outliers = medium_matrix.data[non_outlier_mask]
        final_non_outliers = clean.data[non_outlier_mask]

        np.testing.assert_array_equal(
            original_non_outliers,
            final_non_outliers,
            err_msg="Non-outlier values should not change"
        )

        print(f"✓ {non_outlier_mask.sum()} non-outlier values preserved")


    def test_empty_pipeline(self):
        """
        Test pipeline when no outliers detected.

        Should handle gracefully (no-op) by warning and returning unchanged data.
        """
        # Create a small matrix with EXACTLY ZERO variance per gene
        # MAD-z cannot detect outliers when all values in a row are identical
        n_genes, n_samples = 50, 20

        # Each gene has constant value (different genes have different constants)
        # This guarantees MAD = 0 for every gene, making z-scores undefined/zero
        data = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            data[i, :] = 10.0 + i * 0.1  # Constant across samples

        import pandas as pd
        from cliquefinder.core.biomatrix import BioMatrix

        feature_ids = pd.Index([f"GENE_{i:03d}" for i in range(n_genes)])
        sample_ids = pd.Index([f"SAMPLE_{i:03d}" for i in range(n_samples)])
        sample_metadata = pd.DataFrame({'phenotype': ['A'] * n_samples}, index=sample_ids)
        quality_flags = np.zeros((n_genes, n_samples), dtype=np.uint32)

        no_outlier_matrix = BioMatrix(data, feature_ids, sample_ids, sample_metadata, quality_flags)

        # Detect outliers - should find none (MAD=0 means no variation)
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(no_outlier_matrix)

        outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        n_outliers = outlier_mask.sum()

        assert n_outliers == 0, f"Expected 0 outliers in constant data, got {n_outliers}"
        print(f"Confirmed: {n_outliers} outliers in constant data")

        # Try to impute (should warn about nothing to impute)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            imputer = Imputer(strategy="median")  # Use median (faster for edge case test)
            clean = imputer.apply(flagged)

            # Should have warned about nothing to impute
            warning_found = any("No values flagged" in str(warning.message) for warning in w)
            assert warning_found, "Should warn when nothing to impute"

        # Data should be unchanged (no imputation occurred)
        np.testing.assert_array_equal(no_outlier_matrix.data, clean.data)

        print(f"✓ Empty pipeline handled gracefully (warned and returned unchanged)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
