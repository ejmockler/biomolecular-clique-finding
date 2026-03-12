"""NaN resilience and degenerate-input edge cases for statistical modules.

Covers: VSN convergence parity, global standards normalization with missing
spike-ins, min-value imputation on empty data, median polish NaN propagation
through the permutation null distribution, and empirical p-value NaN exclusion.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# VSN normalization — GPU/CPU convergence parity
# ---------------------------------------------------------------------------

class TestVSNConvergenceParity:
    """GPU and CPU VSN paths must use the same relative convergence formula."""

    def test_gpu_uses_relative_a_change(self):
        """GPU path divides a_change by (|a| + eps), not absolute difference."""
        import inspect
        import cliquefinder.stats.normalization as norm_mod

        mod_source = inspect.getsource(norm_mod)
        assert "mx.abs(a_new - a) / (mx.abs(a) + 1e-10)" in mod_source, \
            "GPU path must use relative convergence for a_change"

    def test_cpu_gpu_formula_match(self):
        """Both paths must use structurally identical relative formulas."""
        import inspect
        import cliquefinder.stats.normalization as norm_mod

        source = inspect.getsource(norm_mod)
        assert "np.abs(a_new - a) / (np.abs(a) + 1e-10)" in source
        assert "mx.abs(a_new - a) / (mx.abs(a) + 1e-10)" in source


# ---------------------------------------------------------------------------
# Global standards normalization — NaN spike-in handling
# ---------------------------------------------------------------------------

class TestGlobalStandardsNaNResilience:
    """When spike-in standards are NaN, normalization must warn and degrade
    gracefully rather than propagating NaN through the entire sample."""

    def test_nan_standards_warns(self):
        """All standards NaN for one sample emits UserWarning."""
        from cliquefinder.stats.normalization import global_standards_normalization

        data = np.random.default_rng(42).standard_normal((10, 5))
        data[0, 2] = np.nan
        data[3, 2] = np.nan

        with pytest.warns(UserWarning, match="All standard proteins are NaN"):
            global_standards_normalization(data, [0, 3])

    def test_nan_standards_preserves_original(self):
        """Affected samples keep original values (norm factor = 0)."""
        from cliquefinder.stats.normalization import global_standards_normalization

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        data[0, 2] = np.nan
        data[3, 2] = np.nan
        original_col2 = data[:, 2].copy()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = global_standards_normalization(data, [0, 3])

        assert_allclose(result.data[:, 2], original_col2,
                        err_msg="NaN-standard sample should preserve original values")

    def test_valid_standards_unaffected(self):
        """Samples with valid standards normalize normally."""
        from cliquefinder.stats.normalization import global_standards_normalization

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))

        result = global_standards_normalization(data, [0, 3])

        assert not np.any(np.isnan(result.data))
        assert np.all(np.isfinite(result.normalization_factors))

    def test_all_samples_nan_standards(self):
        """When ALL samples have NaN standards, no normalization occurs."""
        from cliquefinder.stats.normalization import global_standards_normalization

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        data[0, :] = np.nan
        data[3, :] = np.nan
        original = data.copy()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = global_standards_normalization(data, [0, 3])

        assert_allclose(result.normalization_factors, 0.0,
                        err_msg="All-NaN standards should produce zero factors")
        assert_allclose(result.data, original,
                        err_msg="No normalization should be applied")


# ---------------------------------------------------------------------------
# Min-value imputation — all-NaN guards
# ---------------------------------------------------------------------------

class TestImputeMinValueNaNGuards:
    """impute_min_value must reject all-NaN matrices and handle partial NaN."""

    def test_all_nan_global_raises(self):
        from cliquefinder.stats.missing import impute_min_value
        with pytest.raises(ValueError, match="all values are NaN"):
            impute_min_value(np.full((5, 3), np.nan), method="global")

    def test_all_nan_feature_raises(self):
        from cliquefinder.stats.missing import impute_min_value
        with pytest.raises(ValueError, match="all values are NaN"):
            impute_min_value(np.full((5, 3), np.nan), method="feature")

    def test_all_nan_sample_raises(self):
        from cliquefinder.stats.missing import impute_min_value
        with pytest.raises(ValueError, match="all values are NaN"):
            impute_min_value(np.full((5, 3), np.nan), method="sample")

    def test_partial_nan_imputes(self):
        """Sparse NaN should fill with the appropriate minimum."""
        from cliquefinder.stats.missing import impute_min_value

        data = np.array([[1.0, 2.0, np.nan],
                         [np.nan, 3.0, 4.0],
                         [5.0, np.nan, 6.0]])

        result = impute_min_value(data, method="global")
        assert not np.any(np.isnan(result.data))
        assert result.data[0, 2] == 1.0
        assert result.data[1, 0] == 1.0

    def test_per_feature_all_nan_row_falls_back(self):
        """One all-NaN feature row falls back to global min."""
        from cliquefinder.stats.missing import impute_min_value

        data = np.array([[np.nan, np.nan, np.nan],  # all-NaN feature
                         [2.0, 3.0, np.nan],
                         [5.0, np.nan, 6.0]])

        result = impute_min_value(data, method="feature")
        assert not np.any(np.isnan(result.data))
        # Global min is 2.0; all-NaN feature should use that
        assert result.data[0, 0] == 2.0
        assert result.data[0, 1] == 2.0
        assert result.data[0, 2] == 2.0


# ---------------------------------------------------------------------------
# Batched median polish — NaN propagation semantics
# ---------------------------------------------------------------------------

class TestMedianPolishNaN:
    """NaN from all-NaN protein rows must propagate correctly through median
    polish: contained within the affected row, excluded from overall effect
    via nanmedian, and surfaced as NaN sample abundances only when ALL
    proteins are NaN."""

    def test_single_nan_row_contained(self):
        """One all-NaN protein in a batch element doesn't corrupt the result."""
        from cliquefinder.stats.permutation_gpu import batched_median_polish_gpu

        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 5, 8))
        data[1, 2, :] = np.nan  # one protein all-NaN

        result = batched_median_polish_gpu(data, use_gpu=False)

        assert np.all(np.isfinite(result[0]))
        assert np.all(np.isfinite(result[1])), \
            "NaN from one protein should be excluded by nanmedian"
        assert np.all(np.isfinite(result[2]))

    def test_all_proteins_nan_produces_nan(self):
        """All proteins NaN for a batch element → NaN abundances (not 0.0)."""
        from cliquefinder.stats.permutation_gpu import batched_median_polish_gpu

        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 4, 6))
        data[1, :, :] = np.nan

        result = batched_median_polish_gpu(data, use_gpu=False)

        assert np.all(np.isnan(result[1])), \
            "All-NaN batch element must produce NaN, not 0.0"
        assert np.all(np.isfinite(result[0]))
        assert np.all(np.isfinite(result[2]))

    def test_clean_data_unchanged(self):
        """No NaN in input → finite results, correct shape."""
        from cliquefinder.stats.permutation_gpu import batched_median_polish_gpu

        data = np.random.default_rng(42).standard_normal((5, 4, 8))
        result = batched_median_polish_gpu(data, use_gpu=False)

        assert result.shape == (5, 8)
        assert np.all(np.isfinite(result))

    def test_nan_does_not_block_convergence(self):
        """NaN rows must not prevent convergence for valid entries."""
        from cliquefinder.stats.permutation_gpu import batched_median_polish_gpu

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 6, 8))
        data[0, 3, :] = np.nan

        result = batched_median_polish_gpu(data, max_iter=50, use_gpu=False)
        assert np.all(np.isfinite(result[1]))


# ---------------------------------------------------------------------------
# Empirical p-values — NaN exclusion from null distribution
# ---------------------------------------------------------------------------

class TestEmpiricalPValueNaNExclusion:
    """NaN null t-statistics (from degenerate median polish) must be excluded
    from both numerator and denominator of empirical p-values."""

    def test_nan_null_values_excluded(self):
        """P-values identical whether NaN entries are present or absent."""
        from cliquefinder.stats.permutation_gpu import compute_empirical_pvalues

        observed_t = {"clique1": (1.5, 0.01, 3.0)}

        null_vals = np.concatenate([
            np.random.default_rng(42).standard_normal(95),
            np.array([3.5, 4.0, 3.1, 5.0, 3.2]),
        ])

        results_clean = compute_empirical_pvalues(
            observed_t, {"clique1": null_vals}, {"clique1": null_vals}, 0.05)

        null_with_nan = np.concatenate([null_vals, np.full(20, np.nan)])
        results_nan = compute_empirical_pvalues(
            observed_t, {"clique1": null_with_nan},
            {"clique1": null_with_nan}, 0.05)

        assert len(results_clean) > 0
        assert len(results_nan) > 0
        assert_allclose(
            results_clean[0].empirical_pvalue,
            results_nan[0].empirical_pvalue,
            err_msg="NaN null values should be excluded, not counted")


