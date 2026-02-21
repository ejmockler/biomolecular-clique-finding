"""Tests for validation report verdict logic."""

import json
import tempfile
from pathlib import Path

import pytest

from cliquefinder.stats.validation_report import ValidationReport


class TestVerdictWithFailedPhases:
    """Tests for compute_verdict() handling of failed phases."""

    def _make_passing_report(self):
        """Build a report where all phases pass."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "z_score": 3.5,
        })
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "free": {"permutation_pvalue": 0.01},
            "permutation_pvalue": 0.002,
        })
        report.add_phase("specificity", {
            "specificity_label": "specific",
        })
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.01,
            "n_matched": 20,
        })
        report.add_phase("negative_controls", {
            "target_percentile": 3.0,
            "fpr": 0.02,
        })
        return report

    def test_verdict_with_failed_supplementary_phase(self):
        """Failed Phase 5 doesn't block 'validated' when gates pass."""
        report = self._make_passing_report()
        # Replace Phase 5 with a failure
        report.phases["negative_controls"] = {
            "status": "failed",
            "error": "RotationTestEngine exploded",
        }
        report.compute_verdict()

        # Gates (Phase 1 + 3) pass, so should still be validated
        assert report.verdict == "validated"
        assert "Phases failed: negative_controls" in report.summary

    def test_verdict_with_failed_mandatory_phase(self):
        """Failed Phase 3 (mandatory gate) → inconclusive."""
        report = self._make_passing_report()
        report.phases["label_permutation"] = {
            "status": "failed",
            "error": "All permutations crashed",
        }
        report.compute_verdict()

        # Phase 1 passes but Phase 3 failed (treated as not-passed)
        assert report.verdict != "validated"
        assert "Phases failed: label_permutation" in report.summary

    def test_partial_report_serialization(self):
        """Save/load roundtrip with subset of phases."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.01,
            "z_score": 2.8,
        })
        report.add_phase("specificity", {
            "status": "failed",
            "error": "Not enough contrasts",
        })
        report.compute_verdict()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            report.save(path)
            with open(path) as f:
                loaded = json.load(f)

            assert loaded["verdict"] == report.verdict
            assert "covariate_adjusted" in loaded["phases"]
            assert loaded["phases"]["specificity"]["status"] == "failed"
            assert loaded["summary"] == report.summary
        finally:
            path.unlink()


class TestVerdictCompetitiveZPreference:
    """Tests for Phase 5 competitive z-score preference in verdict."""

    def test_competitive_z_used_when_available(self):
        """Verdict uses competitive z percentile over ROAST percentile."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })
        # Phase 5: ROAST percentile says fail (50%), but competitive z says pass (3%)
        report.add_phase("negative_controls", {
            "target_percentile": 50.0,
            "fpr": 0.4,
            "competitive_z": {
                "percentile": 3.0,
                "fpr": 0.02,
                "target_z": 3.5,
            },
        })
        report.compute_verdict()

        assert report.verdict == "validated"
        # Supplementary should pass (competitive z percentile 3% < 10%)
        assert "1/1 pass" in report.summary

    def test_falls_back_to_roast_without_competitive_z(self):
        """Without competitive z, verdict uses ROAST percentile."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })
        # Phase 5: ROAST percentile passes (5%), no competitive z
        report.add_phase("negative_controls", {
            "target_percentile": 5.0,
            "fpr": 0.03,
        })
        report.compute_verdict()

        assert report.verdict == "validated"
        # Supplementary passes using ROAST percentile (5% < 10%)
        assert "1/1 pass" in report.summary

    def test_roast_fail_without_competitive_z(self):
        """Without competitive z, failing ROAST percentile is correctly used."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })
        # Phase 5: ROAST percentile says fail (50%), no competitive z
        report.add_phase("negative_controls", {
            "target_percentile": 50.0,
            "fpr": 0.4,
        })
        report.compute_verdict()

        # Gates pass but all supplementary fail → inconclusive
        assert report.verdict == "inconclusive"
        assert "0/1" in report.summary or "supplementary" in report.summary.lower()


class TestVerdictAlphaParameter:
    """Tests for configurable alpha threshold in compute_verdict()."""

    def test_verdict_custom_alpha(self):
        """With alpha=0.01, a phase with p=0.03 fails the gate."""
        report = ValidationReport()
        # Phase 1: p=0.03 — passes at alpha=0.05 but fails at alpha=0.01
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.03,
            "z_score": 2.2,
        })
        # Phase 3: clearly significant
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "free": {"permutation_pvalue": 0.005},
            "permutation_pvalue": 0.002,
        })
        report.add_phase("specificity", {
            "specificity_label": "specific",
        })

        # Default alpha=0.05: Phase 1 passes (0.03 < 0.05) → validated
        report.compute_verdict()
        assert report.verdict == "validated"

        # Stricter alpha=0.01: Phase 1 fails (0.03 >= 0.01)
        report.compute_verdict(alpha=0.01)
        assert report.verdict != "validated"

    def test_verdict_custom_alpha_phase3(self):
        """With alpha=0.01, Phase 3 with p=0.04 fails the gate."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
        })
        # Phase 3: p=0.04 — passes at default but fails at alpha=0.01
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.04},
            "permutation_pvalue": 0.04,
        })

        # Default: passes
        report.compute_verdict()
        assert report.verdict == "validated"

        # Stricter: fails
        report.compute_verdict(alpha=0.01)
        assert report.verdict == "inconclusive"

    def test_verdict_default_alpha_backward_compat(self):
        """Default alpha=0.05 produces identical results to previous behavior."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "z_score": 3.5,
        })
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "free": {"permutation_pvalue": 0.01},
            "permutation_pvalue": 0.002,
        })
        report.add_phase("specificity", {
            "specificity_label": "specific",
        })
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.01,
            "n_matched": 20,
        })
        report.add_phase("negative_controls", {
            "target_percentile": 3.0,
            "fpr": 0.02,
        })

        # Explicit alpha=0.05 should match no-arg call
        report.compute_verdict(alpha=0.05)
        verdict_explicit = report.verdict
        summary_explicit = report.summary

        report.compute_verdict()
        assert report.verdict == verdict_explicit
        assert report.summary == summary_explicit
        assert report.verdict == "validated"

    def test_verdict_custom_alpha_phase4_supplementary(self):
        """Custom alpha also affects Phase 4 supplementary threshold."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.001},
            "permutation_pvalue": 0.001,
        })
        # Phase 4: p=0.03 — passes at alpha=0.05, fails at alpha=0.01
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.03,
            "n_matched": 15,
        })

        # At default alpha=0.05, matched passes → "1/1 pass"
        report.compute_verdict()
        assert report.verdict == "validated"
        assert "1/1 pass" in report.summary

        # At alpha=0.01, matched fails → "0/1 pass" but still validated
        # because supplementary failure alone doesn't block (only
        # "all supplementary fail" leads to inconclusive)
        report.compute_verdict(alpha=0.01)
        # All 1 supplementary phase fails → inconclusive
        assert report.verdict == "inconclusive"


class TestCLIAlphaFlag:
    """Tests for N1: --alpha CLI flag on validate-baselines command."""

    def test_alpha_argument_registered(self):
        """Verify --alpha is accepted by the argparse parser."""
        import argparse
        from cliquefinder.cli.validate_baselines import register_parser

        parent = argparse.ArgumentParser()
        subparsers = parent.add_subparsers()
        register_parser(subparsers)

        # Default value
        args = parent.parse_args(["validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
        ])
        assert args.alpha == 0.05

        # Custom value
        args = parent.parse_args(["validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
            "--alpha", "0.01",
        ])
        assert args.alpha == 0.01
