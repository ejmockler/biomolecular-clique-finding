"""
Tests for Wave 4 statistical documentation and convention fixes.

Covers:
- STAT-6:  SpecificityResult null_corr as top-level field + docstring notes
- STAT-10: P-value sidedness documentation in label_permutation,
           permutation_framework, and rotation modules
- STAT-13: FDR docstring revision in rotation.py
"""

from __future__ import annotations

import inspect

import pytest


# ============================================================================
# STAT-6: Specificity null_corr promotion and docstring notes
# ============================================================================

class TestStat6NullCorrelation:
    """STAT-6: null_corr should be a top-level field in SpecificityResult dict."""

    def test_null_corr_top_level_in_to_dict(self):
        """null_correlation must appear as a top-level key in to_dict() output."""
        from cliquefinder.stats.specificity import SpecificityResult, ContrastEnrichment

        result = SpecificityResult(
            primary_contrast="C9_vs_CTRL",
            contrasts={
                "C9_vs_CTRL": ContrastEnrichment(
                    contrast_name="C9_vs_CTRL",
                    z_score=3.0,
                    empirical_pvalue=0.001,
                ),
                "Sporadic_vs_CTRL": ContrastEnrichment(
                    contrast_name="Sporadic_vs_CTRL",
                    z_score=1.5,
                    empirical_pvalue=0.05,
                ),
            },
            specificity_ratio=2.0,
            specificity_label="specific",
            summary="test",
            interaction_z=1.5,
            interaction_pvalue=0.01,
            z_difference=1.5,
            z_difference_ci=(-0.5, 3.5),
            null_correlation=0.75,
        )

        d = result.to_dict()

        # null_correlation must be a TOP-LEVEL key, not only nested
        assert "null_correlation" in d, (
            "null_correlation should be a top-level key in to_dict() output"
        )
        assert d["null_correlation"] == 0.75

        # It should also still be available in interaction_test for
        # backward compatibility
        assert "interaction_test" in d
        assert d["interaction_test"]["null_correlation"] == 0.75

    def test_null_corr_absent_when_no_interaction(self):
        """When interaction test was not run, null_correlation should not appear."""
        from cliquefinder.stats.specificity import SpecificityResult, ContrastEnrichment

        result = SpecificityResult(
            primary_contrast="C9_vs_CTRL",
            contrasts={
                "C9_vs_CTRL": ContrastEnrichment(
                    contrast_name="C9_vs_CTRL",
                    z_score=3.0,
                    empirical_pvalue=0.001,
                ),
            },
            specificity_ratio=float("inf"),
            specificity_label="inconclusive",
            summary="Only one contrast provided.",
        )

        d = result.to_dict()
        assert "null_correlation" not in d
        assert "interaction_test" not in d

    def test_specificity_docstring_mentions_permutation_resolution(self):
        """SpecificityResult docstring should mention permutation resolution."""
        from cliquefinder.stats.specificity import SpecificityResult

        docstring = SpecificityResult.__doc__
        assert docstring is not None
        assert "permutation resolution" in docstring.lower(), (
            "SpecificityResult docstring should mention permutation resolution"
        )
        assert "n_perms" in docstring, (
            "SpecificityResult docstring should reference n_perms parameter"
        )

    def test_specificity_docstring_mentions_shared_sample_correlation(self):
        """SpecificityResult docstring should explain null_corr meaning."""
        from cliquefinder.stats.specificity import SpecificityResult

        docstring = SpecificityResult.__doc__
        assert docstring is not None
        assert "shared-sample correlation" in docstring.lower(), (
            "SpecificityResult docstring should explain shared-sample "
            "correlation structure"
        )

    def test_interaction_permutation_docstring_mentions_resolution(self):
        """_run_interaction_permutation docstring should mention resolution."""
        from cliquefinder.stats.specificity import _run_interaction_permutation

        docstring = _run_interaction_permutation.__doc__
        assert docstring is not None
        assert "permutation resolution" in docstring.lower()
        assert "n_perms" in docstring


# ============================================================================
# STAT-10: Inconsistent p-value sidedness documentation
# ============================================================================

class TestStat10PValueSidedness:
    """STAT-10: Each module should document its p-value sidedness convention."""

    def test_label_permutation_docstring_mentions_one_sided(self):
        """run_label_permutation_null docstring must state one-sided."""
        from cliquefinder.stats.label_permutation import run_label_permutation_null

        docstring = run_label_permutation_null.__doc__
        assert docstring is not None
        assert "one-sided" in docstring.lower(), (
            "run_label_permutation_null docstring must mention 'one-sided'"
        )

    def test_permutation_framework_docstring_mentions_two_sided(self):
        """run_competitive_test docstring must state two-sided."""
        from cliquefinder.stats.permutation_framework import PermutationTestEngine

        docstring = PermutationTestEngine.run_competitive_test.__doc__
        assert docstring is not None
        assert "two-sided" in docstring.lower(), (
            "run_competitive_test docstring must mention 'two-sided'"
        )

    def test_rotation_pvalue_docstring_mentions_sidedness(self):
        """compute_rotation_pvalues docstring should document sidedness."""
        from cliquefinder.stats.rotation import compute_rotation_pvalues

        docstring = compute_rotation_pvalues.__doc__
        assert docstring is not None
        # Should mention the sidedness convention for UP/DOWN/MIXED
        assert "one-sided" in docstring.lower() or "upper-tail" in docstring.lower(), (
            "compute_rotation_pvalues docstring should mention sidedness"
        )

    def test_rotation_alternative_docstring_documents_directions(self):
        """Alternative enum docstring must document UP, DOWN, MIXED."""
        from cliquefinder.stats.rotation import Alternative

        docstring = Alternative.__doc__
        assert docstring is not None
        assert "UP" in docstring
        assert "DOWN" in docstring
        assert "MIXED" in docstring
        assert "two-sided" in docstring.lower()

    def test_validation_report_sidedness_comment(self):
        """validation_report.py compute_verdict should have sidedness comment."""
        from cliquefinder.stats.validation_report import ValidationReport

        source = inspect.getsource(ValidationReport.compute_verdict)
        assert "sidedness" in source.lower() or "STAT-10" in source, (
            "compute_verdict should have a comment about p-value sidedness"
        )


# ============================================================================
# STAT-13: FDR docstring revision
# ============================================================================

class TestStat13FDRDocstring:
    """STAT-13: FDR docstrings should mention Benjamini-Hochberg for multiple testing."""

    def test_results_to_dataframe_fdr_docstring(self):
        """results_to_dataframe docstring must mention FDR correction."""
        from cliquefinder.stats.rotation import RotationTestEngine

        docstring = RotationTestEngine.results_to_dataframe.__doc__
        assert docstring is not None
        assert "Benjamini-Hochberg" in docstring or "FDR correction" in docstring, (
            "results_to_dataframe docstring must mention FDR correction "
            "for multiple gene set testing"
        )

    def test_results_to_dataframe_single_set_valid(self):
        """Docstring should clarify single gene-set case is valid without FDR."""
        from cliquefinder.stats.rotation import RotationTestEngine

        docstring = RotationTestEngine.results_to_dataframe.__doc__
        assert docstring is not None
        assert "single gene-set" in docstring.lower() or "single gene set" in docstring.lower(), (
            "Docstring should clarify that single gene-set validation "
            "does not require FDR correction"
        )

    def test_run_rotation_test_fdr_docstring(self):
        """run_rotation_test docstring must mention FDR/Benjamini-Hochberg."""
        from cliquefinder.stats.rotation import run_rotation_test

        docstring = run_rotation_test.__doc__
        assert docstring is not None
        assert "Benjamini-Hochberg" in docstring or "FDR correction" in docstring, (
            "run_rotation_test docstring must mention FDR correction "
            "for multiple gene set testing"
        )

    def test_fdr_docstring_mentions_multiple_gene_sets(self):
        """FDR docstring should mention the 'multiple gene sets' context."""
        from cliquefinder.stats.rotation import RotationTestEngine

        docstring = RotationTestEngine.results_to_dataframe.__doc__
        assert docstring is not None
        assert "multiple gene set" in docstring.lower() or "multiple gene-set" in docstring.lower(), (
            "FDR docstring should mention testing multiple gene sets"
        )
