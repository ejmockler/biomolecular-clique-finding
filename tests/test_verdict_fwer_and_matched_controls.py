"""Tests for FWER docstring accuracy and graph permutation verdict integration.

Validates:
- compute_verdict docstring uses bounded-FWER language with rho-dependent
  formulas instead of the incorrect alpha^2 claim.
- Graph permutation phase integrates correctly into supplementary verdict.
"""

import inspect
import numpy as np
import pytest

from cliquefinder.stats.validation_report import ValidationReport


# ===================================================================
# FWER docstring accuracy
# ===================================================================


class TestFWERDocstring:
    """Verify compute_verdict docstring uses bounded-FWER with rho formula."""

    def _get_docstring(self):
        return inspect.getdoc(ValidationReport.compute_verdict)

    def test_docstring_contains_bounded_fwer(self):
        """Docstring says 'bounded-FWER' (not 'controlled FWER')."""
        doc = self._get_docstring()
        assert "bounded-FWER" in doc.lower() or "Bounded-FWER" in doc or "bounded-fwer" in doc.lower()

    def test_docstring_contains_rho_dependent_formula(self):
        """Docstring includes the rho-dependent joint-pass formula."""
        doc = self._get_docstring()
        assert "rho" in doc
        assert "Phi" in doc
        assert "z_alpha" in doc
        assert "sqrt(1 - rho" in doc or "sqrt(1-rho" in doc

    def test_docstring_no_alpha_squared_claim(self):
        """Docstring does NOT claim joint probability IS alpha^2."""
        doc = self._get_docstring()
        assert "joint probability under the global null is alpha^2" not in doc.lower()

    def test_docstring_contains_quantitative_bounds(self):
        """Docstring gives the approximate FWER range for typical rho."""
        doc = self._get_docstring()
        assert "0.006" in doc
        assert "0.020" in doc

    def test_docstring_mentions_design_asymmetry(self):
        """Docstring contains the 'Design asymmetry note'."""
        doc = self._get_docstring()
        assert "Design asymmetry" in doc

    def test_docstring_mentions_rho_080_value(self):
        """Docstring notes the rho=0.8 case (approx 0.030)."""
        doc = self._get_docstring()
        assert "0.030" in doc


# ===================================================================
# Graph permutation verdict integration
# ===================================================================


class TestGraphPermutationVerdict:
    """Graph permutation phase should integrate as supplementary evidence."""

    def _make_report(self, phases):
        """Build a ValidationReport with given phases and mandatory gates passing."""
        report = ValidationReport.__new__(ValidationReport)
        report.phases = phases
        report.verdict = None
        report.summary = None
        report.details = {}
        return report

    def _base_phases(self):
        """Phases where mandatory gates pass."""
        return {
            "covariate_adjusted": {"empirical_pvalue": 0.001},
            "label_permutation": {"permutation_pvalue": 0.001},
        }

    def test_graph_permutation_pass_increments_supplementary(self):
        """Low graph_permutation percentile counts as supplementary pass."""
        phases = self._base_phases()
        phases["graph_permutation"] = {
            "target_percentile": 3.0,
            "fpr": 0.02,
        }
        report = self._make_report(phases)
        report.compute_verdict()

        # With mandatory gates passing and supplementary passing,
        # verdict should be "validated"
        assert report.verdict == "validated"
        assert "graph_permutation" in report.details or "Supplementary" in report.summary

    def test_graph_permutation_fail_counted(self):
        """High graph_permutation percentile counts as supplementary fail."""
        phases = self._base_phases()
        phases["graph_permutation"] = {
            "target_percentile": 80.0,
            "fpr": 0.5,
        }
        report = self._make_report(phases)
        report.compute_verdict()

        # Mandatory gates pass but only supplementary is graph_permutation
        # which fails. With 0 supplementary passes out of 1, verdict is inconclusive.
        assert report.verdict == "inconclusive"

    def test_graph_permutation_absent_no_effect(self):
        """Without graph_permutation in phases, verdict unaffected."""
        phases = self._base_phases()
        report = self._make_report(phases)
        report.compute_verdict()

        # No supplementary phases -> validated-with-caveat or validated
        # depending on implementation, but definitely not inconclusive
        # due to missing supplementary
        assert report.verdict in ("validated", "validated-limited")
