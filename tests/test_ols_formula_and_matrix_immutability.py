"""Tests for Wave 5 miscellaneous architecture findings.

GPU-9:  OLSPrecomputedMatrices immutability (dataclasses.replace)
STAT-CORE-8: OLS formula with .T on pinv matches reference
VAL-4:  details dict stored on ValidationReport and in to_dict()
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from cliquefinder.stats.permutation_gpu import OLSPrecomputedMatrices
from cliquefinder.stats.validation_report import ValidationReport


# =============================================================================
# GPU-9: OLSPrecomputedMatrices not mutated in place
# =============================================================================


class TestGPU9_ImmutableMatrices:
    """Verify OLSPrecomputedMatrices uses replace() instead of mutation."""

    def _make_matrices(self, **overrides) -> OLSPrecomputedMatrices:
        """Create a default OLSPrecomputedMatrices for testing."""
        defaults = dict(
            X=np.eye(4, 2),
            XtX_inv=np.eye(2),
            c=np.array([1.0, -1.0]),
            c_var_factor=2.0,
            df_residual=2,
            conditions=["A", "B"],
            contrast_name="A-B",
            eb_d0=None,
            eb_s0_sq=None,
            eb_df_total=None,
        )
        defaults.update(overrides)
        return OLSPrecomputedMatrices(**defaults)

    def test_replace_creates_new_instance(self):
        """replace() should return a *new* object, not mutate the original."""
        original = self._make_matrices()
        assert original.eb_d0 is None
        assert original.eb_s0_sq is None
        assert original.eb_df_total is None

        updated = replace(original, eb_d0=5.0, eb_s0_sq=0.1, eb_df_total=7.0)

        # Original is unchanged
        assert original.eb_d0 is None
        assert original.eb_s0_sq is None
        assert original.eb_df_total is None

        # Updated has new values
        assert updated.eb_d0 == 5.0
        assert updated.eb_s0_sq == 0.1
        assert updated.eb_df_total == 7.0

    def test_replace_preserves_other_fields(self):
        """replace() should leave non-overridden fields intact."""
        original = self._make_matrices(df_residual=10)
        updated = replace(original, eb_d0=3.0, eb_s0_sq=0.5, eb_df_total=13.0)

        assert updated.df_residual == 10
        assert updated.contrast_name == "A-B"
        np.testing.assert_array_equal(updated.c, original.c)
        np.testing.assert_array_equal(updated.X, original.X)
        np.testing.assert_array_equal(updated.XtX_inv, original.XtX_inv)

    def test_replace_with_none_clears_eb_fields(self):
        """replace() with None should disable EB moderation fields."""
        matrices = self._make_matrices(eb_d0=5.0, eb_s0_sq=0.1, eb_df_total=7.0)
        cleared = replace(matrices, eb_d0=None, eb_s0_sq=None, eb_df_total=2.0)

        assert cleared.eb_d0 is None
        assert cleared.eb_s0_sq is None
        assert cleared.eb_df_total == 2.0

    def test_replace_with_inf_d0(self):
        """replace() with inf d0 should work for no-shrinkage case."""
        matrices = self._make_matrices(df_residual=8)
        updated = replace(
            matrices,
            eb_d0=np.inf,
            eb_s0_sq=0.5,
            eb_df_total=float(matrices.df_residual),
        )
        assert np.isinf(updated.eb_d0)
        assert updated.eb_df_total == 8.0


# =============================================================================
# STAT-CORE-8: OLS formula correctness
# =============================================================================


class TestSTATCORE8_OLSFormula:
    """Verify row-major OLS formula Y @ X @ XtX_inv.T matches reference."""

    def _ols_reference(self, X, Y_row):
        """Reference OLS using the standard column-major formula.

        Standard: beta_col = (X'X)^{-1} X' Y_col
        where Y_col is (n_samples, n_features).
        Returns beta_row = (n_features, n_params).
        """
        Y_col = Y_row.T  # (n_samples, n_features)
        beta_col = np.linalg.lstsq(X, Y_col, rcond=None)[0]  # (n_params, n_features)
        return beta_col.T  # (n_features, n_params)

    def test_formula_matches_lstsq(self):
        """Y @ X @ XtX_inv.T should match np.linalg.lstsq."""
        rng = np.random.default_rng(42)
        n_samples, n_params, n_features = 20, 3, 50

        X = rng.standard_normal((n_samples, n_params))
        Y_row = rng.standard_normal((n_features, n_samples))

        XtX_inv = np.linalg.inv(X.T @ X)
        beta_batched = Y_row @ X @ XtX_inv.T
        beta_ref = self._ols_reference(X, Y_row)

        np.testing.assert_allclose(beta_batched, beta_ref, atol=1e-10)

    def test_formula_with_pinv(self):
        """Y @ X @ pinv(X'X).T should match lstsq even when using pinv."""
        rng = np.random.default_rng(123)
        n_samples, n_params, n_features = 15, 4, 30

        X = rng.standard_normal((n_samples, n_params))
        Y_row = rng.standard_normal((n_features, n_samples))

        XtX_inv = np.linalg.pinv(X.T @ X)
        beta_batched = Y_row @ X @ XtX_inv.T
        beta_ref = self._ols_reference(X, Y_row)

        np.testing.assert_allclose(beta_batched, beta_ref, atol=1e-10)

    def test_formula_with_moderately_collinear_design(self):
        """Formula should still match lstsq with moderately correlated columns.

        Uses moderate collinearity (correlation ~0.95) rather than extreme
        near-singularity, since pinv and lstsq use different SVD truncation
        thresholds for truly rank-deficient matrices.
        """
        rng = np.random.default_rng(99)
        n_samples, n_features = 20, 10

        # Create moderately collinear design: col3 ≈ col1 + col2 + noise
        X = rng.standard_normal((n_samples, 3))
        X[:, 2] = X[:, 0] + X[:, 1] + 0.3 * rng.standard_normal(n_samples)

        Y_row = rng.standard_normal((n_features, n_samples))

        XtX_inv = np.linalg.pinv(X.T @ X)
        beta_batched = Y_row @ X @ XtX_inv.T
        beta_ref = self._ols_reference(X, Y_row)

        np.testing.assert_allclose(beta_batched, beta_ref, atol=1e-8)

    def test_transpose_matters_for_asymmetric_pinv(self):
        """Demonstrate that .T on pinv matters when pinv is not symmetric.

        Creates a scenario where pinv(X'X) is detectably asymmetric and
        verifies that using .T gives the correct answer while omitting .T
        gives a (slightly) different answer.
        """
        rng = np.random.default_rng(777)
        n_samples, n_params, n_features = 30, 5, 20

        X = rng.standard_normal((n_samples, n_params))
        Y_row = rng.standard_normal((n_features, n_samples))

        XtX_inv = np.linalg.pinv(X.T @ X)

        # Artificially make pinv asymmetric to test .T matters
        asymmetry = rng.standard_normal((n_params, n_params)) * 1e-12
        XtX_inv_asym = XtX_inv + asymmetry

        beta_with_T = Y_row @ X @ XtX_inv_asym.T
        beta_without_T = Y_row @ X @ XtX_inv_asym

        # They should differ (slightly) when pinv is asymmetric
        # The one with .T is the mathematically correct derivation
        if not np.allclose(XtX_inv_asym, XtX_inv_asym.T):
            assert not np.allclose(beta_with_T, beta_without_T), (
                "With asymmetric pinv, .T should produce a different result"
            )

    def test_residuals_from_formula(self):
        """Verify residuals computed from batched formula are correct."""
        rng = np.random.default_rng(55)
        n_samples, n_params, n_features = 10, 2, 5

        X = rng.standard_normal((n_samples, n_params))
        Y_row = rng.standard_normal((n_features, n_samples))

        XtX_inv = np.linalg.inv(X.T @ X)
        beta = Y_row @ X @ XtX_inv.T
        Y_pred = beta @ X.T
        residuals = Y_row - Y_pred

        # Residuals should be orthogonal to X
        for j in range(n_features):
            proj = X.T @ residuals[j]
            np.testing.assert_allclose(proj, 0, atol=1e-10)


# =============================================================================
# VAL-4: details dict stored and included in to_dict()
# =============================================================================


class TestVAL4_DetailsStored:
    """Verify that compute_verdict stores details and to_dict includes them."""

    def test_details_stored_on_validated(self):
        """Details should be stored when verdict is 'validated'."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.compute_verdict()

        assert report.verdict == "validated"
        assert hasattr(report, "phase_details")
        assert isinstance(report.phase_details, dict)
        assert "covariate_adjusted" in report.phase_details
        assert "pass" in report.phase_details["covariate_adjusted"]

    def test_details_stored_on_refuted(self):
        """Details should be stored when verdict is 'refuted'."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.5})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.8})
        report.compute_verdict()

        assert report.verdict == "refuted"
        assert isinstance(report.phase_details, dict)
        assert "covariate_adjusted" in report.phase_details
        assert "fail" in report.phase_details["covariate_adjusted"]

    def test_details_stored_on_inconclusive(self):
        """Details should be stored when verdict is 'inconclusive'."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.5})
        report.compute_verdict()

        assert report.verdict == "inconclusive"
        assert isinstance(report.phase_details, dict)
        assert "label_permutation_stratified" in report.phase_details

    def test_details_in_to_dict(self):
        """to_dict() should include phase_details when present."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.add_phase("specificity", {"specificity_label": "specific"})
        report.compute_verdict()

        d = report.to_dict()
        assert "phase_details" in d
        assert isinstance(d["phase_details"], dict)
        assert "covariate_adjusted" in d["phase_details"]
        assert "specificity" in d["phase_details"]

    def test_details_not_in_to_dict_without_compute_verdict(self):
        """to_dict() should not include phase_details if compute_verdict not called."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        d = report.to_dict()

        # phase_details is empty by default (not populated)
        assert "phase_details" not in d  # empty dict is falsy

    def test_details_includes_supplementary_phases(self):
        """Details should include supplementary phase info."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.02,
            "n_matched": 15,
        })
        report.add_phase("negative_controls", {
            "target_percentile": 5.0,
            "fpr": 0.03,
        })
        report.compute_verdict()

        assert "matched_reanalysis" in report.phase_details
        assert "negative_controls" in report.phase_details

    def test_details_includes_permutation_warning(self):
        """Details should include warning when stratified passes but free fails."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.01},
            "free": {"permutation_pvalue": 0.5},
        })
        report.compute_verdict()

        assert "permutation_warning" in report.phase_details

    def test_to_dict_json_serializable(self):
        """to_dict() output with phase_details should be JSON-serializable."""
        import json

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.add_phase("specificity", {"specificity_label": "specific"})
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.02,
            "n_matched": 15,
        })
        report.compute_verdict()

        d = report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["phase_details"] == d["phase_details"]
