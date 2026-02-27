"""
Tests for ARCH-1 fix: Silent Method Failure tracking in run_method_comparison().

Verifies that:
1. Failing methods are captured in MethodComparisonResult.failed_methods
2. summary() includes failure warnings when methods fail
3. Concordance is computed for surviving methods only
4. When ALL methods fail, result is properly reported
5. Backward compatibility: failed_methods defaults to empty dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.method_comparison import (
    MethodComparisonResult,
    MethodName,
    UnifiedCliqueResult,
    run_method_comparison,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight fake methods that satisfy the CliqueTestMethod protocol
# ---------------------------------------------------------------------------


class _SuccessMethod:
    """A fake method that always returns predetermined results."""

    def __init__(
        self,
        name: MethodName,
        results: list[UnifiedCliqueResult] | None = None,
    ):
        self._name = name
        self._results = results or []

    @property
    def name(self) -> MethodName:
        return self._name

    def test(self, experiment, **kwargs) -> list[UnifiedCliqueResult]:
        return self._results


class _FailingMethod:
    """A fake method that always raises."""

    def __init__(self, name: MethodName, error_msg: str = "kaboom"):
        self._name = name
        self._error_msg = error_msg

    @property
    def name(self) -> MethodName:
        return self._name

    def test(self, experiment, **kwargs) -> list[UnifiedCliqueResult]:
        raise RuntimeError(self._error_msg)


def _make_result(
    clique_id: str,
    method: MethodName,
    p_value: float = 0.01,
    effect_size: float = 1.5,
) -> UnifiedCliqueResult:
    """Create a minimal valid UnifiedCliqueResult."""
    return UnifiedCliqueResult(
        clique_id=clique_id,
        method=method,
        effect_size=effect_size,
        effect_size_se=0.3,
        p_value=p_value,
        statistic_value=5.0,
        statistic_type="t",
        degrees_of_freedom=10.0,
        n_proteins=10,
        n_proteins_found=8,
    )


def _make_fake_experiment():
    """Create a minimal mock PreparedCliqueExperiment."""
    mock_exp = MagicMock()
    mock_exp.preprocessing_params = {"normalization": "median"}
    return mock_exp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ols_results():
    """Three clique results from OLS (need >= 3 for concordance)."""
    return [
        _make_result("TP53", MethodName.OLS, p_value=0.001),
        _make_result("MYC", MethodName.OLS, p_value=0.5),
        _make_result("BRCA1", MethodName.OLS, p_value=0.03),
    ]


@pytest.fixture()
def roast_results():
    """Three clique results from ROAST_MSQ, agreeing with OLS on TP53."""
    return [
        _make_result("TP53", MethodName.ROAST_MSQ, p_value=0.005),
        _make_result("MYC", MethodName.ROAST_MSQ, p_value=0.6),
        _make_result("BRCA1", MethodName.ROAST_MSQ, p_value=0.04),
    ]


# ---------------------------------------------------------------------------
# Test 1: A single failing method is captured in failed_methods
# ---------------------------------------------------------------------------


class TestFailedMethodTracking:
    """Verify that method failures are captured, not silently swallowed."""

    def test_single_failure_captured(self, ols_results):
        """When one method crashes, its error is recorded in failed_methods."""
        good = _SuccessMethod(MethodName.OLS, ols_results)
        bad = _FailingMethod(MethodName.LMM, "singular matrix")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad],
                verbose=False,
            )

        assert "lmm" in result.failed_methods
        assert "singular matrix" in result.failed_methods["lmm"]

    def test_multiple_failures_captured(self):
        """When multiple methods crash, all are recorded."""
        bad1 = _FailingMethod(MethodName.LMM, "no convergence")
        bad2 = _FailingMethod(MethodName.ROAST_MSQ, "rotation error")
        good = _SuccessMethod(MethodName.OLS, [
            _make_result("X", MethodName.OLS),
        ])

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad1, bad2],
                verbose=False,
            )

        assert len(result.failed_methods) == 2
        assert "lmm" in result.failed_methods
        assert "roast_msq" in result.failed_methods
        assert "no convergence" in result.failed_methods["lmm"]
        assert "rotation error" in result.failed_methods["roast_msq"]

    def test_no_failures_empty_dict(self, ols_results):
        """When all methods succeed, failed_methods is empty."""
        good = _SuccessMethod(MethodName.OLS, ols_results)

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good],
                verbose=False,
            )

        assert result.failed_methods == {}

    def test_failed_method_not_in_methods_run(self, ols_results):
        """A failed method should NOT appear in methods_run (only succeeded)."""
        good = _SuccessMethod(MethodName.OLS, ols_results)
        bad = _FailingMethod(MethodName.LMM, "boom")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad],
                verbose=False,
            )

        assert MethodName.OLS in result.methods_run
        assert MethodName.LMM not in result.methods_run


# ---------------------------------------------------------------------------
# Test 2: summary() includes failure warnings
# ---------------------------------------------------------------------------


class TestSummaryFailureWarnings:
    """Verify that summary() output surfaces method failures."""

    def test_summary_contains_warning_when_methods_fail(self):
        """summary() should include a WARNING line listing failed methods."""
        # Build a minimal MethodComparisonResult directly
        mcr = MethodComparisonResult(
            results_by_method={
                MethodName.OLS: [_make_result("X", MethodName.OLS)],
            },
            pairwise_concordance=[],
            mean_spearman_rho=np.nan,
            mean_cohen_kappa=np.nan,
            disagreement_cases=pd.DataFrame(),
            preprocessing_params={},
            methods_run=[MethodName.OLS],
            n_cliques_tested=1,
            failed_methods={"lmm": "singular matrix", "roast_msq": "NaN in rotation"},
        )

        text = mcr.summary()
        assert "WARNING" in text
        assert "2 method(s) failed" in text
        assert "lmm: singular matrix" in text
        assert "roast_msq: NaN in rotation" in text

    def test_summary_no_warning_when_all_succeed(self):
        """summary() should NOT contain WARNING when no failures."""
        mcr = MethodComparisonResult(
            results_by_method={
                MethodName.OLS: [_make_result("X", MethodName.OLS)],
            },
            pairwise_concordance=[],
            mean_spearman_rho=np.nan,
            mean_cohen_kappa=np.nan,
            disagreement_cases=pd.DataFrame(),
            preprocessing_params={},
            methods_run=[MethodName.OLS],
            n_cliques_tested=1,
            failed_methods={},
        )

        text = mcr.summary()
        assert "WARNING" not in text
        assert "failed" not in text.lower()


# ---------------------------------------------------------------------------
# Test 3: Concordance is computed for surviving methods
# ---------------------------------------------------------------------------


class TestConcordanceWithFailures:
    """Verify concordance computation proceeds correctly despite failures."""

    def test_concordance_computed_for_surviving_methods(
        self, ols_results, roast_results
    ):
        """Two surviving methods should still produce concordance metrics."""
        good_ols = _SuccessMethod(MethodName.OLS, ols_results)
        good_roast = _SuccessMethod(MethodName.ROAST_MSQ, roast_results)
        bad = _FailingMethod(MethodName.LMM, "explosion")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good_ols, good_roast, bad],
                verbose=False,
            )

        # Should have exactly one pairwise concordance (OLS vs ROAST_MSQ)
        assert len(result.pairwise_concordance) == 1
        conc = result.pairwise_concordance[0]
        assert conc.method_a == MethodName.OLS
        assert conc.method_b == MethodName.ROAST_MSQ

        # Failed method should still be tracked
        assert "lmm" in result.failed_methods

        # Mean metrics should be computed from the surviving pair
        assert np.isfinite(result.mean_spearman_rho)
        assert np.isfinite(result.mean_cohen_kappa)

    def test_single_surviving_method_no_concordance(self, ols_results):
        """With only one surviving method, no pairwise concordance is possible."""
        good = _SuccessMethod(MethodName.OLS, ols_results)
        bad1 = _FailingMethod(MethodName.LMM, "err1")
        bad2 = _FailingMethod(MethodName.ROAST_MSQ, "err2")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad1, bad2],
                verbose=False,
            )

        assert len(result.pairwise_concordance) == 0
        assert np.isnan(result.mean_spearman_rho)
        assert np.isnan(result.mean_cohen_kappa)
        assert len(result.failed_methods) == 2


# ---------------------------------------------------------------------------
# Test 4: ALL methods fail
# ---------------------------------------------------------------------------


class TestAllMethodsFail:
    """Verify proper reporting when every method crashes."""

    def test_all_methods_fail_result(self):
        """When all methods fail, result has no methods_run, no concordance."""
        bad1 = _FailingMethod(MethodName.OLS, "ols crash")
        bad2 = _FailingMethod(MethodName.LMM, "lmm crash")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            result = run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[bad1, bad2],
                verbose=False,
            )

        # No methods succeeded
        assert result.methods_run == []
        assert result.n_cliques_tested == 0

        # All methods recorded as failed
        assert len(result.failed_methods) == 2
        assert "ols" in result.failed_methods
        assert "lmm" in result.failed_methods
        assert "ols crash" in result.failed_methods["ols"]

        # No concordance possible
        assert result.pairwise_concordance == []
        assert np.isnan(result.mean_spearman_rho)
        assert np.isnan(result.mean_cohen_kappa)

    def test_all_fail_summary_has_warning(self):
        """summary() should list all failures when everything crashes."""
        mcr = MethodComparisonResult(
            results_by_method={
                MethodName.OLS: [],
                MethodName.LMM: [],
            },
            pairwise_concordance=[],
            mean_spearman_rho=np.nan,
            mean_cohen_kappa=np.nan,
            disagreement_cases=pd.DataFrame(),
            preprocessing_params={},
            methods_run=[],
            n_cliques_tested=0,
            failed_methods={"ols": "ols crash", "lmm": "lmm crash"},
        )

        text = mcr.summary()
        assert "WARNING" in text
        assert "2 method(s) failed" in text
        assert "ols: ols crash" in text
        assert "lmm: lmm crash" in text


# ---------------------------------------------------------------------------
# Test 5: Backward compatibility — default_factory=dict
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure the new field has a default and doesn't break existing code."""

    def test_default_factory_is_empty_dict(self):
        """MethodComparisonResult without failed_methods arg defaults to {}."""
        mcr = MethodComparisonResult(
            results_by_method={},
            pairwise_concordance=[],
            mean_spearman_rho=np.nan,
            mean_cohen_kappa=np.nan,
            disagreement_cases=pd.DataFrame(),
            preprocessing_params={},
            methods_run=[],
            n_cliques_tested=0,
            # NOTE: NOT passing failed_methods — testing default
        )

        assert mcr.failed_methods == {}
        assert isinstance(mcr.failed_methods, dict)


# ---------------------------------------------------------------------------
# Test 6: Verbose output includes failure info
# ---------------------------------------------------------------------------


class TestVerboseOutput:
    """Verify that verbose mode prints failure diagnostics."""

    def test_verbose_prints_attempted_vs_succeeded(self, ols_results, capsys):
        """Verbose output should show attempted and succeeded counts."""
        good = _SuccessMethod(MethodName.OLS, ols_results)
        bad = _FailingMethod(MethodName.LMM, "no convergence")

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad],
                verbose=True,
            )

        captured = capsys.readouterr().out
        assert "Methods attempted: 2, succeeded: 1" in captured
        assert "FAILED methods:" in captured
        assert "lmm: no convergence" in captured

    def test_verbose_no_failure_section_when_all_succeed(
        self, ols_results, capsys
    ):
        """When all methods succeed, verbose output should not show FAILED."""
        good = _SuccessMethod(MethodName.OLS, ols_results)

        with patch(
            "cliquefinder.stats.method_comparison.prepare_experiment",
            return_value=_make_fake_experiment(),
        ):
            run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good],
                verbose=True,
            )

        captured = capsys.readouterr().out
        assert "Methods attempted: 1, succeeded: 1" in captured
        assert "FAILED methods:" not in captured


# ---------------------------------------------------------------------------
# Test 7: Logger warning is emitted on failure
# ---------------------------------------------------------------------------


class TestLoggerWarning:
    """Verify that logger.warning is called when a method fails."""

    def test_logger_warning_emitted(self, ols_results):
        """Failing method should emit a logger.warning."""
        good = _SuccessMethod(MethodName.OLS, ols_results)
        bad = _FailingMethod(MethodName.LMM, "crash!")

        with (
            patch(
                "cliquefinder.stats.method_comparison.prepare_experiment",
                return_value=_make_fake_experiment(),
            ),
            patch(
                "cliquefinder.stats.method_comparison.logger"
            ) as mock_logger,
        ):
            run_method_comparison(
                data=np.zeros((5, 4)),
                feature_ids=["a", "b", "c", "d", "e"],
                sample_metadata=pd.DataFrame(
                    {"cond": ["A", "A", "B", "B"]}
                ),
                cliques=[],
                condition_column="cond",
                contrast=("A", "B"),
                methods=[good, bad],
                verbose=False,
            )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "lmm" in str(call_args)
        assert "crash!" in str(call_args)
