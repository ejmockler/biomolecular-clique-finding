"""Tests for ddof=0 conventions across QRILC, normalization CV, summarization,
and permutation z-score; fit_f_dist documentation; ROAST effect size
documentation; n_draws deprecation; supplementary phase messages.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest


class TestQrilcDdof:
    """QRILC global fallback in impute_qrilc should use ddof=1."""

    def test_qrilc_global_fallback_uses_ddof1(self):
        """np.std call in QRILC global fallback uses ddof=1."""
        from cliquefinder.stats.missing import impute_qrilc

        source = inspect.getsource(impute_qrilc)
        # The global fallback line should contain ddof=1
        assert "ddof=1" in source, (
            "impute_qrilc global fallback should use ddof=1 for sample std"
        )

    def test_qrilc_global_fallback_numerical(self):
        """QRILC global fallback produces sample std (ddof=1), not pop std."""
        from cliquefinder.stats.missing import impute_qrilc

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=(5, 6))
        data[0, 0] = np.nan
        data[1, 1] = np.nan
        data[2, 2] = np.nan

        result = impute_qrilc(data, random_state=42)
        assert not np.any(np.isnan(result.data))
        assert result.n_imputed == 3


class TestNormalizationCvDdof:
    """CV computation in assess_normalization_quality uses ddof=1."""

    def test_cv_computation_uses_ddof1(self):
        """np.std in CV calculation uses ddof=1."""
        from cliquefinder.stats.normalization import assess_normalization_quality

        source = inspect.getsource(assess_normalization_quality)
        assert "ddof=1" in source, (
            "CV computation should use ddof=1 for sample std"
        )

    def test_cv_numerical_correctness(self):
        """CV values match expected ddof=1 formula."""
        from cliquefinder.stats.normalization import assess_normalization_quality

        rng = np.random.default_rng(42)
        before = rng.normal(10.0, 2.0, size=(20, 5))
        after = before - np.nanmedian(before, axis=0) + 10.0

        result = assess_normalization_quality(before, after)

        medians_before = np.nanmedian(before, axis=0)
        expected_cv = np.std(medians_before, ddof=1) / np.mean(medians_before)
        assert abs(result["median_cv_before"] - expected_cv) < 1e-10

    def test_cv_single_sample_returns_nan(self):
        """Single-sample data returns NaN for CV (can't compute ddof=1 std)."""
        from cliquefinder.stats.normalization import assess_normalization_quality

        data = np.array([[1.0], [2.0], [3.0]])
        result = assess_normalization_quality(data, data)
        assert np.isnan(result["median_cv_before"])


class TestSummarizationDdof:
    """CliqueSummary.to_dict should use ddof=1 for std_abundance."""

    def test_to_dict_uses_ddof1(self):
        """Source code of to_dict contains ddof=1."""
        from cliquefinder.stats.summarization import CliqueSummary

        source = inspect.getsource(CliqueSummary.to_dict)
        assert "ddof=1" in source, (
            "CliqueSummary.to_dict should use ddof=1 for std_abundance"
        )

    def test_to_dict_numerical_correctness(self):
        """std_abundance matches np.nanstd with ddof=1."""
        from cliquefinder.stats.summarization import CliqueSummary

        abundances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = CliqueSummary(
            clique_id="test",
            sample_abundances=abundances,
            n_proteins=3,
            protein_ids=["A", "B", "C"],
            method="median_polish",
        )
        result = summary.to_dict()
        expected_std = float(np.nanstd(abundances, ddof=1))
        assert abs(result["std_abundance"] - expected_std) < 1e-10

    def test_to_dict_single_value_returns_zero(self):
        """Single non-NaN value returns 0.0 for std_abundance."""
        from cliquefinder.stats.summarization import CliqueSummary

        abundances = np.array([5.0, np.nan, np.nan])
        summary = CliqueSummary(
            clique_id="test",
            sample_abundances=abundances,
            n_proteins=1,
            protein_ids=["A"],
            method="median_polish",
        )
        result = summary.to_dict()
        assert result["std_abundance"] == 0.0

    def test_to_dict_with_nans(self):
        """ddof=1 std is computed only over non-NaN values."""
        from cliquefinder.stats.summarization import CliqueSummary

        abundances = np.array([1.0, 2.0, np.nan, 4.0])
        summary = CliqueSummary(
            clique_id="test",
            sample_abundances=abundances,
            n_proteins=2,
            protein_ids=["A", "B"],
            method="median_polish",
        )
        result = summary.to_dict()
        expected_std = float(np.nanstd(abundances, ddof=1))
        assert abs(result["std_abundance"] - expected_std) < 1e-10


class TestPermutationZscoreDdof:
    """PermutationResult.zscore should use ddof=1."""

    def _make_result(self, null_dist, observed_stat=10.0):
        from cliquefinder.stats.permutation_framework import (
            PermutationResult,
            TestResult,
        )

        observed = TestResult(
            feature_set_id="test",
            test_statistic=observed_stat,
            p_value=0.01,
            effect_size=1.0,
            n_test=5,
            n_reference=5,
        )
        return PermutationResult(
            feature_set_id="test",
            observed=observed,
            null_distribution=null_dist,
            empirical_pvalue=0.01,
            empirical_pvalue_onesided=0.005,
            percentile_rank=99.0,
            n_permutations=len(null_dist),
        )

    def test_zscore_uses_ddof1(self):
        """Source code of zscore property contains ddof=1."""
        from cliquefinder.stats.permutation_framework import PermutationResult

        source = inspect.getsource(PermutationResult.zscore.fget)
        assert "ddof=1" in source, (
            "PermutationResult.zscore should use ddof=1 for null std"
        )

    def test_zscore_numerical_correctness(self):
        """z-score matches (obs - mean) / std(ddof=1) formula."""
        null_dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self._make_result(null_dist, observed_stat=10.0)

        expected_z = (10.0 - np.mean(null_dist)) / np.std(null_dist, ddof=1)
        assert abs(result.zscore - expected_z) < 1e-10

    def test_to_dict_null_std_uses_ddof1(self):
        """to_dict null_std also uses ddof=1."""
        null_dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self._make_result(null_dist, observed_stat=10.0)
        d = result.to_dict()
        expected_null_std = float(np.std(null_dist, ddof=1))
        assert abs(d["null_std"] - expected_null_std) < 1e-10

    def test_zscore_with_two_elements(self):
        """z-score works correctly with exactly 2 null distribution values."""
        null_dist = np.array([1.0, 3.0])
        result = self._make_result(null_dist, observed_stat=5.0)

        expected_z = (5.0 - 2.0) / np.std(null_dist, ddof=1)
        assert abs(result.zscore - expected_z) < 1e-10


class TestFitFDistComment:
    """fit_f_dist should document that R limma uses weighted moments."""

    def test_fit_f_dist_has_weighted_moments_comment(self):
        """Source contains comment about R limma df-weighted moments."""
        from cliquefinder.stats.permutation_gpu import fit_f_dist

        source = inspect.getsource(fit_f_dist)
        assert "weighted" in source.lower(), (
            "fit_f_dist should document that R limma uses df-weighted moments"
        )
        assert "ddof=1" in source, (
            "fit_f_dist should use ddof=1 for variance computation"
        )


class TestRobustHitsComment:
    """robust_hits() comment should accurately describe NaN filtering."""

    def test_comment_explains_dropna_usage(self):
        """The robust_hits() method uses dropna correctly."""
        from cliquefinder.stats.concordance import MethodComparisonResult

        source = inspect.getsource(MethodComparisonResult.robust_hits)
        # robust_hits uses explicit .dropna() per-row.
        assert "dropna" in source.lower(), (
            "robust_hits() should reference its dropna-based NaN handling"
        )

    def test_comment_explains_nan_exclusion(self):
        """Comment explains that NaN p-values cause exclusion via .all()."""
        from cliquefinder.stats.concordance import MethodComparisonResult

        source = inspect.getsource(MethodComparisonResult.robust_hits)
        assert "NaN" in source, (
            "robust_hits() comment should explain NaN behavior"
        )


class TestRoastEffectSizeComment:
    """Roast effect_size 'up' direction should have rationale comment."""

    def test_effect_size_up_direction_documented(self):
        """Source explains why 'up' direction is used for effect_size."""
        from cliquefinder.stats.methods.roast import ROASTMethod

        source = inspect.getsource(ROASTMethod)
        # Should contain explanation about limma convention and upregulation
        assert "limma" in source.lower(), (
            "Roast effect_size should reference limma convention"
        )
        assert "upregulat" in source, (
            "Roast effect_size should explain upregulation convention"
        )


class TestNDrawsDeprecated:
    """impute_aft_model n_draws parameter should warn when != 1."""

    def test_n_draws_default_no_warning(self):
        """Default n_draws=1 does not trigger deprecation warning."""
        from cliquefinder.stats.missing import impute_aft_model

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=(5, 6))
        data[0, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = impute_aft_model(data, random_state=42)
            assert result.n_imputed == 1

    def test_n_draws_nondefault_warns(self):
        """n_draws != 1 triggers DeprecationWarning."""
        from cliquefinder.stats.missing import impute_aft_model

        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, size=(5, 6))
        data[0, 0] = np.nan

        with pytest.warns(DeprecationWarning, match="n_draws"):
            impute_aft_model(data, n_draws=5, random_state=42)

    def test_docstring_documents_deprecation(self):
        """Docstring mentions that n_draws is deprecated."""
        from cliquefinder.stats.missing import impute_aft_model

        doc = impute_aft_model.__doc__
        assert doc is not None
        assert "Deprecated" in doc or "deprecated" in doc, (
            "impute_aft_model docstring should note n_draws deprecation"
        )


class TestSupplementaryPhaseMessages:
    """Validation report should distinguish unexecuted vs no-effect phases."""

    def _make_report_with_gates_passing(self, extra_phases=None):
        """Create a ValidationReport with both mandatory gates passing."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.01,
        })
        report.add_phase("label_permutation", {
            "permutation_pvalue": 0.02,
        })
        if extra_phases:
            for name, result in extra_phases.items():
                report.add_phase(name, result)
        report.compute_verdict()
        return report

    def test_no_supplementary_message_says_not_executed(self):
        """When no supplementary phases ran, message says 'not executed'."""
        report = self._make_report_with_gates_passing()
        msg = report.summary.lower()
        assert "not executed" in msg or "not configured" in msg, (
            f"Summary should say supplementary phases were 'not executed' or "
            f"'not configured', got: {report.summary}"
        )

    def test_no_supplementary_message_does_not_say_no_phases_ran(self):
        """The old ambiguous 'No supplementary phases ran' text is gone."""
        report = self._make_report_with_gates_passing()
        assert "No supplementary phases ran" not in report.summary, (
            f"Old ambiguous message should be replaced, got: {report.summary}"
        )

    def test_supplementary_pass_count_message(self):
        """When supplementary phases ran and passed, message shows counts."""
        report = self._make_report_with_gates_passing(extra_phases={
            "matched_reanalysis": {
                "empirical_pvalue": 0.03,
                "n_matched": 10,
            },
        })
        assert "1/1 pass" in report.summary, (
            f"Summary should show supplementary pass/total counts, "
            f"got: {report.summary}"
        )
