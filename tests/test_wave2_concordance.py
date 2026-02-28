"""Tests for Audit II Wave 2: concordance and verdict fixes.

Covers:
    MCOMP-1: robust_hits() NaN poisoning from failed methods
    MCOMP-2: wide_format() include_invalid parameter
    VAL-1:   Verdict "refuted" gap when Phase 3 not run
    VAL-2:   Negative controls docstring inversion
"""

from __future__ import annotations

import numpy as np
import pytest

from cliquefinder.stats.concordance import MethodComparisonResult
from cliquefinder.stats.method_comparison_types import (
    ConcordanceMetrics,
    MethodName,
    UnifiedCliqueResult,
)
from cliquefinder.stats.validation_report import ValidationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    clique_id: str,
    method: MethodName,
    p_value: float,
    effect_size: float = 1.0,
) -> UnifiedCliqueResult:
    """Build a minimal UnifiedCliqueResult for testing."""
    return UnifiedCliqueResult(
        clique_id=clique_id,
        method=method,
        effect_size=effect_size,
        effect_size_se=0.1,
        p_value=p_value,
        statistic_value=2.0,
        statistic_type="t",
        degrees_of_freedom=10.0,
        n_proteins=10,
        n_proteins_found=8,
    )


def _make_comparison(
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]],
) -> MethodComparisonResult:
    """Build a MethodComparisonResult without computing real concordance."""
    return MethodComparisonResult(
        results_by_method=results_by_method,
        pairwise_concordance=[],
        mean_spearman_rho=0.0,
        mean_cohen_kappa=0.0,
        disagreement_cases=None,
        preprocessing_params={},
        methods_run=list(results_by_method.keys()),
        n_cliques_tested=len(
            {r.clique_id for rs in results_by_method.values() for r in rs}
        ),
    )


# ---------------------------------------------------------------------------
# MCOMP-1: robust_hits() tolerates failed (all-NaN) methods
# ---------------------------------------------------------------------------


class TestRobustHitsNaNTolerance:
    """MCOMP-1: robust_hits() should not be poisoned by a failed method."""

    def test_robust_hits_tolerates_failed_method(self):
        """One method produces all NaN p-values; robust_hits still works
        from the remaining methods that have valid results."""
        clique_ids = ["CQ1", "CQ2", "CQ3"]

        # Method A: all significant
        ols_results = [
            _make_result(cid, MethodName.OLS, p_value=0.001)
            for cid in clique_ids
        ]

        # Method B: all significant
        roast_results = [
            _make_result(cid, MethodName.ROAST_MSQ, p_value=0.002)
            for cid in clique_ids
        ]

        # Method C: FAILED (all NaN p-values → invalid results)
        perm_results = [
            _make_result(cid, MethodName.PERMUTATION_COMPETITIVE, p_value=float("nan"), effect_size=float("nan"))
            for cid in clique_ids
        ]

        comp = _make_comparison({
            MethodName.OLS: ols_results,
            MethodName.ROAST_MSQ: roast_results,
            MethodName.PERMUTATION_COMPETITIVE: perm_results,
        })

        # Before fix: robust_hits returns [] because NaN column poisons .all()
        # After fix:  NaN-only columns are excluded; CQ1-CQ3 are robust hits
        robust = comp.robust_hits(threshold=0.05)
        assert set(robust) == {"CQ1", "CQ2", "CQ3"}

    def test_robust_hits_empty_when_no_active_methods(self):
        """If ALL methods produce NaN, robust_hits returns empty."""
        clique_ids = ["CQ1"]
        nan_results_a = [
            _make_result("CQ1", MethodName.OLS, p_value=float("nan"), effect_size=float("nan"))
        ]
        nan_results_b = [
            _make_result("CQ1", MethodName.ROAST_MSQ, p_value=float("nan"), effect_size=float("nan"))
        ]
        comp = _make_comparison({
            MethodName.OLS: nan_results_a,
            MethodName.ROAST_MSQ: nan_results_b,
        })
        assert comp.robust_hits() == []

    def test_robust_hits_partial_nan_per_row(self):
        """A clique with NaN in one active method is still robust if all
        other active methods are significant."""
        # OLS: CQ1=0.001, CQ2=0.001
        ols = [
            _make_result("CQ1", MethodName.OLS, p_value=0.001),
            _make_result("CQ2", MethodName.OLS, p_value=0.001),
        ]
        # ROAST: CQ1=0.002, CQ2 not present (missing, not NaN)
        roast = [
            _make_result("CQ1", MethodName.ROAST_MSQ, p_value=0.002),
        ]
        comp = _make_comparison({
            MethodName.OLS: ols,
            MethodName.ROAST_MSQ: roast,
        })
        robust = comp.robust_hits(threshold=0.05)
        # CQ1 has both methods significant → robust
        # CQ2 has only OLS (ROAST is NaN) → still robust (only non-NaN checked)
        assert "CQ1" in robust
        assert "CQ2" in robust


# ---------------------------------------------------------------------------
# MCOMP-2: wide_format() invalid result filtering
# ---------------------------------------------------------------------------


class TestWideFormatInvalidFiltering:
    """MCOMP-2: wide_format() filters invalid results by default."""

    def _build_comparison_with_invalid(self):
        """Build a comparison where one result has NaN p-value."""
        valid = _make_result("CQ1", MethodName.OLS, p_value=0.01)
        invalid = _make_result(
            "CQ2", MethodName.OLS,
            p_value=float("nan"), effect_size=float("nan"),
        )
        return _make_comparison({
            MethodName.OLS: [valid, invalid],
        })

    def test_wide_format_excludes_invalid_by_default(self):
        """By default, wide_format() only includes valid results."""
        comp = self._build_comparison_with_invalid()
        wide = comp.wide_format()
        # CQ2 has NaN p-value → invalid → excluded by default
        assert "CQ1" in wide["clique_id"].values
        assert "CQ2" not in wide["clique_id"].values

    def test_wide_format_includes_invalid_when_requested(self):
        """include_invalid=True includes results with NaN/inf p-values."""
        comp = self._build_comparison_with_invalid()
        wide = comp.wide_format(include_invalid=True)
        assert "CQ1" in wide["clique_id"].values
        assert "CQ2" in wide["clique_id"].values

    def test_wide_format_include_invalid_false_explicit(self):
        """Explicitly passing include_invalid=False behaves like default."""
        comp = self._build_comparison_with_invalid()
        wide_default = comp.wide_format()
        wide_explicit = comp.wide_format(include_invalid=False)
        assert list(wide_default["clique_id"]) == list(wide_explicit["clique_id"])


# ---------------------------------------------------------------------------
# VAL-1: Verdict "inconclusive" when Phase 3 missing (not failed)
# ---------------------------------------------------------------------------


class TestVerdictPhase3Missing:
    """VAL-1: Phase 1 passes + Phase 3 not run → 'inconclusive', not 'refuted'."""

    def test_verdict_inconclusive_when_phase3_missing(self):
        """Phase 1 passes but Phase 3 was never added → inconclusive."""
        report = ValidationReport()
        # Phase 1: passes
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "z_score": 3.5,
        })
        # Phase 3 (label_permutation) is NOT added at all

        report.compute_verdict()

        assert report.verdict == "inconclusive", (
            f"Expected 'inconclusive' but got '{report.verdict}'"
        )
        assert "not run" in report.summary.lower() or "was not run" in report.summary.lower()

    def test_verdict_refuted_when_both_gates_fail(self):
        """Both Phase 1 and Phase 3 fail → still 'refuted' (regression guard)."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.5,
        })
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.6},
            "permutation_pvalue": 0.6,
        })
        report.compute_verdict()
        assert report.verdict == "refuted"

    def test_verdict_validated_when_both_gates_pass(self):
        """Both Phase 1 and Phase 3 pass → still 'validated' (regression guard)."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
        })
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })
        report.compute_verdict()
        assert report.verdict == "validated"

    def test_verdict_inconclusive_phase1_pass_phase3_failed(self):
        """Phase 1 passes, Phase 3 ran but failed (status=failed) →
        inconclusive (existing behavior, regression guard)."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
        })
        report.add_phase("label_permutation", {
            "status": "failed",
            "error": "Permutation crashed",
        })
        report.compute_verdict()
        # Phase 3 is present but has status "failed" → gate_permutation stays False
        # and perm is truthy (dict exists), so existing branch handles it
        assert report.verdict != "refuted"


# ---------------------------------------------------------------------------
# VAL-2: Negative controls docstring correctness
# ---------------------------------------------------------------------------


class TestNegativeControlsDocstring:
    """VAL-2: Verify docstring accurately reflects the computation."""

    def test_negative_controls_docstring(self):
        """competitive_z_percentile docstring says 0 = most enriched."""
        from cliquefinder.stats.negative_controls import NegativeControlResult

        docstring = NegativeControlResult.__doc__
        assert docstring is not None, "NegativeControlResult has no docstring"

        # The corrected docstring should say 0 = most enriched
        assert "0 = most enriched" in docstring, (
            "Docstring should state '0 = most enriched' for competitive_z_percentile"
        )
        # Should NOT say "100 = most enriched" (the old incorrect text)
        assert "100 = most enriched" not in docstring, (
            "Docstring should NOT state '100 = most enriched'"
        )
