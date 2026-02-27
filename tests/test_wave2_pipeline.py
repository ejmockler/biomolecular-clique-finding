"""Tests for wave-2 pipeline hardening and validation fixes.

Covers:
- ARCH-3:  Failure threshold tracking in run_stratified_analysis()
- ARCH-6:  Checkpoint/resume in validate_baselines orchestrator
- ARCH-15: Updated docstring for compute_verdict()
- STAT-15: Configurable constants (neg_ctrl_percentile, specificity-z-threshold, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cliquefinder.stats.validation_report import ValidationReport


# =========================================================================
# ARCH-3: Failure threshold tracking in run_stratified_analysis()
# =========================================================================

class TestARCH3FailureThreshold:
    """Test that run_stratified_analysis aborts or warns on high failure rates."""

    def _make_mock_module(self, name: str):
        """Create a minimal mock INDRAModule."""
        m = MagicMock()
        m.regulator_name = name
        m.regulator_id = f"HGNC:{hash(name) % 10000}"
        m.indra_target_names = {f"GENE{i}" for i in range(5)}
        return m

    def _make_mock_matrix(self):
        """Create a minimal mock BioMatrix."""
        m = MagicMock()
        m.feature_ids = [f"GENE{i}" for i in range(20)]
        m.sample_ids = [f"S{i}" for i in range(10)]
        m.data = np.random.default_rng(42).normal(size=(20, 10))
        m.sample_metadata = MagicMock()
        m.quality_flags = np.ones(10)
        m.n_features = 20
        m.n_samples = 10
        return m

    def _make_mock_validator(self):
        """Create a mock CliqueValidator with realistic cache stats."""
        mock_validator = MagicMock()
        mock_validator.get_available_conditions.return_value = ["CASE_M", "CTRL_M"]
        mock_validator.get_cache_stats.return_value = {
            "n_precomputed_corr": 0,
            "precomputed_corr_mb": 0.0,
            "n_precomputed_genes": 0,
        }
        return mock_validator

    @patch("cliquefinder.cli._analyze_core.INDRAModuleExtractor")
    @patch("cliquefinder.cli._analyze_core.CliqueValidator")
    @patch("cliquefinder.cli._analyze_core.MyGeneInfoMapper")
    @patch("cliquefinder.cli._analyze_core.analyze_regulator_module")
    def test_all_fail_raises_runtime_error(
        self, mock_analyze, mock_mapper, mock_validator_cls, mock_extractor_cls
    ):
        """When >50% of regulators fail, RuntimeError is raised."""
        from cliquefinder.cli._analyze_core import run_stratified_analysis

        modules = [self._make_mock_module(f"TF{i}") for i in range(10)]
        mock_extractor = MagicMock()
        mock_extractor.get_regulator_modules.return_value = modules
        mock_extractor_cls.return_value = mock_extractor

        mock_validator_cls.return_value = self._make_mock_validator()

        # ALL regulators fail
        mock_analyze.side_effect = RuntimeError("Neo4j connection lost")

        matrix = self._make_mock_matrix()
        cogex = MagicMock()

        with pytest.raises(RuntimeError, match="Analysis aborted"):
            run_stratified_analysis(
                matrix=matrix,
                regulators=[f"TF{i}" for i in range(10)],
                cogex_client=cogex,
                stratify_by=["phenotype"],
                n_workers=1,
            )

    @patch("cliquefinder.cli._analyze_core.INDRAModuleExtractor")
    @patch("cliquefinder.cli._analyze_core.CliqueValidator")
    @patch("cliquefinder.cli._analyze_core.MyGeneInfoMapper")
    @patch("cliquefinder.cli._analyze_core.analyze_regulator_module")
    def test_partial_failure_logs_warning(
        self, mock_analyze, mock_mapper, mock_validator_cls, mock_extractor_cls, caplog
    ):
        """When 10-50% of regulators fail, a WARNING is logged."""
        from cliquefinder.cli._analyze_core import run_stratified_analysis

        modules = [self._make_mock_module(f"TF{i}") for i in range(10)]
        mock_extractor = MagicMock()
        mock_extractor.get_regulator_modules.return_value = modules
        mock_extractor_cls.return_value = mock_extractor

        mock_validator_cls.return_value = self._make_mock_validator()

        # 2 out of 10 fail (20% > 10% threshold)
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient failure")
            result = MagicMock()
            result.regulator_name = kwargs.get("regulator_name", "TF")
            return result

        mock_analyze.side_effect = side_effect

        matrix = self._make_mock_matrix()
        cogex = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cliquefinder.cli._analyze_core"):
            results = run_stratified_analysis(
                matrix=matrix,
                regulators=[f"TF{i}" for i in range(10)],
                cogex_client=cogex,
                stratify_by=["phenotype"],
                n_workers=1,
            )

        # Should have 8 results (10 - 2 failures)
        assert len(results) == 8
        # Should have logged a warning about 20% failure rate
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("failed" in msg and "20%" in msg for msg in warning_msgs), (
            f"Expected warning about 20% failure rate. Got: {warning_msgs}"
        )

    @patch("cliquefinder.cli._analyze_core.INDRAModuleExtractor")
    @patch("cliquefinder.cli._analyze_core.CliqueValidator")
    @patch("cliquefinder.cli._analyze_core.MyGeneInfoMapper")
    @patch("cliquefinder.cli._analyze_core.analyze_regulator_module")
    def test_low_failure_no_warning(
        self, mock_analyze, mock_mapper, mock_validator_cls, mock_extractor_cls, caplog
    ):
        """When <10% of regulators fail, no WARNING is logged."""
        from cliquefinder.cli._analyze_core import run_stratified_analysis

        modules = [self._make_mock_module(f"TF{i}") for i in range(20)]
        mock_extractor = MagicMock()
        mock_extractor.get_regulator_modules.return_value = modules
        mock_extractor_cls.return_value = mock_extractor

        mock_validator = self._make_mock_validator()
        mock_validator.get_available_conditions.return_value = ["CASE_M"]
        mock_validator_cls.return_value = mock_validator

        # 1 out of 20 fails (5% < 10%)
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("one-off failure")
            result = MagicMock()
            return result

        mock_analyze.side_effect = side_effect

        matrix = self._make_mock_matrix()
        cogex = MagicMock()

        with caplog.at_level(logging.WARNING, logger="cliquefinder.cli._analyze_core"):
            results = run_stratified_analysis(
                matrix=matrix,
                regulators=[f"TF{i}" for i in range(20)],
                cogex_client=cogex,
                stratify_by=["phenotype"],
                n_workers=1,
            )

        assert len(results) == 19
        # No "Results may be incomplete" warning (the _FAILURE_WARN_THRESHOLD line)
        warn_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "Results may be incomplete" in r.message
        ]
        assert len(warn_records) == 0


# =========================================================================
# ARCH-6: Checkpoint/resume in validate_baselines
# =========================================================================

class TestARCH6Checkpoint:
    """Test checkpoint save/load round-trip and phase skipping."""

    def test_save_load_roundtrip(self):
        """Saved checkpoint can be loaded and contains the correct phases."""
        from cliquefinder.cli.validate_baselines import _save_checkpoint, _load_checkpoint

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "z_score": 3.5,
        })
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _save_checkpoint(report, output_dir)

            # Verify file exists
            checkpoint_path = output_dir / "validation_checkpoint.json"
            assert checkpoint_path.exists()

            # Load and verify
            loaded = _load_checkpoint(output_dir)
            assert "covariate_adjusted" in loaded.phases
            assert "label_permutation" in loaded.phases
            assert loaded.phases["covariate_adjusted"]["empirical_pvalue"] == 0.001

    def test_load_missing_checkpoint_returns_fresh(self):
        """When no checkpoint exists, returns a fresh ValidationReport."""
        from cliquefinder.cli.validate_baselines import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = _load_checkpoint(Path(tmpdir))
            assert len(loaded.phases) == 0

    def test_load_corrupt_checkpoint_returns_fresh(self):
        """Corrupt checkpoint file returns a fresh ValidationReport."""
        from cliquefinder.cli.validate_baselines import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "validation_checkpoint.json").write_text("not valid json{{{")
            loaded = _load_checkpoint(output_dir)
            assert len(loaded.phases) == 0

    def test_completed_phases_skipped_on_resume(self):
        """Phases already in checkpoint should not be re-run."""
        from cliquefinder.cli.validate_baselines import _save_checkpoint, _load_checkpoint

        # Simulate a checkpoint with Phase 1 done
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "z_score": 3.5,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _save_checkpoint(report, output_dir)

            loaded = _load_checkpoint(output_dir)

            # The orchestrator checks: if "covariate_adjusted" in report.phases
            # We verify this condition holds
            assert "covariate_adjusted" in loaded.phases
            assert "label_permutation" not in loaded.phases

    def test_force_restart_flag_registered(self):
        """Verify --force-restart is accepted by the argument parser."""
        from cliquefinder.cli.validate_baselines import register_parser

        parent = argparse.ArgumentParser()
        subparsers = parent.add_subparsers()
        register_parser(subparsers)

        # Default: False
        args = parent.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
        ])
        assert args.force_restart is False

        # With flag
        args = parent.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
            "--force-restart",
        ])
        assert args.force_restart is True


# =========================================================================
# ARCH-15: Docstring for compute_verdict()
# =========================================================================

class TestARCH15Docstring:
    """Test that the compute_verdict docstring documents the design asymmetry."""

    def test_docstring_contains_asymmetry_note(self):
        """Docstring must document the intentional asymmetry."""
        docstring = ValidationReport.compute_verdict.__doc__
        assert docstring is not None, "compute_verdict() has no docstring"
        assert "Design asymmetry" in docstring
        assert "VALIDATION" in docstring
        assert "inconclusive" in docstring
        assert "Bonferroni-Holm" in docstring

    def test_docstring_replaces_alpha_squared_claim(self):
        """The old alpha^2 independent-test claim should be replaced."""
        docstring = ValidationReport.compute_verdict.__doc__
        # The new docstring uses bounded-FWER with rho-dependent formula (STAT-8)
        assert "bounded-FWER" in docstring.lower() or "Bounded-FWER" in docstring
        # The old claim "alpha^2 when tests are independent" should be gone
        assert "joint probability under the\n           global null is alpha^2" not in docstring


# =========================================================================
# STAT-15: Configurable constants
# =========================================================================

class TestSTAT15ConfigurableConstants:
    """Test that verdict thresholds are parameterized and affect results."""

    def _make_passing_report(self):
        """Build a report where all phases pass at default thresholds."""
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
        report.add_phase("negative_controls", {
            "target_percentile": 8.0,
            "fpr": 0.05,
        })
        return report

    def test_compute_verdict_accepts_neg_ctrl_percentile(self):
        """compute_verdict() accepts neg_ctrl_percentile parameter."""
        report = self._make_passing_report()
        # With default threshold (10.0), percentile=8.0 passes
        report.compute_verdict(neg_ctrl_percentile=10.0)
        assert report.verdict == "validated"
        assert "1/1 pass" in report.summary

    def test_strict_neg_ctrl_percentile_changes_verdict(self):
        """Stricter percentile threshold changes Phase 5 pass/fail."""
        report = self._make_passing_report()

        # At default (10.0), percentile 8.0 passes
        report.compute_verdict(neg_ctrl_percentile=10.0)
        assert report.verdict == "validated"

        # At stricter threshold (5.0), percentile 8.0 fails
        report.compute_verdict(neg_ctrl_percentile=5.0)
        # All supplementary fail -> inconclusive
        assert report.verdict == "inconclusive"

    def test_lenient_neg_ctrl_percentile_passes_borderline(self):
        """Lenient percentile threshold passes a borderline case."""
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.001})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.002},
            "permutation_pvalue": 0.002,
        })
        # Percentile 15 fails at default (10.0) but passes at lenient (20.0)
        report.add_phase("negative_controls", {
            "target_percentile": 15.0,
            "fpr": 0.1,
        })

        report.compute_verdict(neg_ctrl_percentile=10.0)
        assert report.verdict == "inconclusive"

        report.compute_verdict(neg_ctrl_percentile=20.0)
        assert report.verdict == "validated"

    def test_cli_stat15_arguments_registered(self):
        """Verify STAT-15 CLI arguments are registered and have correct defaults."""
        from cliquefinder.cli.validate_baselines import register_parser

        parent = argparse.ArgumentParser()
        subparsers = parent.add_subparsers()
        register_parser(subparsers)

        args = parent.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
        ])

        # Check defaults
        assert args.specificity_z_threshold == 1.5
        assert args.neg_ctrl_percentile == 10.0
        assert args.interaction_n_perms == 200

    def test_cli_stat15_custom_values(self):
        """STAT-15 CLI arguments accept custom values."""
        from cliquefinder.cli.validate_baselines import register_parser

        parent = argparse.ArgumentParser()
        subparsers = parent.add_subparsers()
        register_parser(subparsers)

        args = parent.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out/", "--network-query", "TP53",
            "--specificity-z-threshold", "2.0",
            "--negative-control-percentile", "5.0",
            "--interaction-n-perms", "500",
        ])

        assert args.specificity_z_threshold == 2.0
        assert args.neg_ctrl_percentile == 5.0
        assert args.interaction_n_perms == 500

    def test_neg_ctrl_percentile_default_backward_compat(self):
        """Default neg_ctrl_percentile=10.0 matches previous hardcoded behavior."""
        report = self._make_passing_report()

        # Without parameter (uses default 10.0)
        report.compute_verdict()
        verdict_default = report.verdict
        summary_default = report.summary

        # With explicit 10.0
        report.compute_verdict(neg_ctrl_percentile=10.0)
        assert report.verdict == verdict_default
        assert report.summary == summary_default


class TestSpecificityZThreshold:
    """Test that compute_specificity respects z_threshold parameter."""

    def test_z_threshold_affects_specificity_label(self):
        """Different z_threshold values can change the specificity label."""
        from cliquefinder.stats.specificity import compute_specificity

        enrichment_by_contrast = {
            "primary": {
                "z_score": 2.0,
                "empirical_pvalue": 0.01,
                "n_targets": 15,
                "pct_down": 0.4,
                "direction_pvalue": 0.3,
            },
            "secondary": {
                "z_score": 0.5,
                "empirical_pvalue": 0.3,
                "n_targets": 15,
                "pct_down": 0.5,
                "direction_pvalue": 0.5,
            },
        }

        # With default z_threshold=1.5, primary (z=2.0) is above threshold
        result_low = compute_specificity(
            enrichment_by_contrast,
            primary_contrast="primary",
            z_threshold=1.5,
        )
        assert result_low.specificity_label == "specific"

        # With z_threshold=3.0, primary (z=2.0) is below threshold
        result_high = compute_specificity(
            enrichment_by_contrast,
            primary_contrast="primary",
            z_threshold=3.0,
        )
        assert result_high.specificity_label == "inconclusive"
