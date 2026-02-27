"""Tests for Wave 4 architecture/code quality improvements.

ARCH-8:  Protocol get_conditions (no isinstance on concrete)
ARCH-12: CLI parameter bounds checking
ARCH-18: Warning convention (warnings.warn vs logger.warning)
"""

from __future__ import annotations

import argparse
import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest


# =====================================================================
# ARCH-8: Protocol abstraction — get_conditions
# =====================================================================


class TestProtocolGetConditions:
    """Verify ExperimentalDesign Protocol has get_conditions and concrete
    implementations satisfy it."""

    def test_protocol_has_get_conditions(self):
        """ExperimentalDesign Protocol declares get_conditions."""
        from cliquefinder.stats.permutation_framework import ExperimentalDesign

        # Check that get_conditions is defined on the Protocol
        assert hasattr(ExperimentalDesign, "get_conditions"), (
            "ExperimentalDesign Protocol must define get_conditions()"
        )

    def test_two_group_design_get_conditions(self):
        """TwoGroupDesign.get_conditions returns condition column values."""
        from cliquefinder.stats.permutation_framework import TwoGroupDesign

        design = TwoGroupDesign(
            condition_column="group",
            test_condition="A",
            reference_condition="B",
        )
        meta = pd.DataFrame({"group": ["A", "B", "A", "B"]})
        conditions = design.get_conditions(meta)
        np.testing.assert_array_equal(conditions, ["A", "B", "A", "B"])

    def test_metadata_derived_design_get_conditions(self):
        """MetadataDerivedDesign.get_conditions returns derived labels."""
        from cliquefinder.stats.permutation_framework import MetadataDerivedDesign

        def derive_fn(row):
            return "treated" if row["dose"] > 0 else "control"

        design = MetadataDerivedDesign(
            derivation_fn=derive_fn,
            test_condition="treated",
            reference_condition="control",
        )
        meta = pd.DataFrame({"dose": [10, 0, 5, 0]})
        conditions = design.get_conditions(meta)
        np.testing.assert_array_equal(
            conditions, ["treated", "control", "treated", "control"]
        )

    def test_no_isinstance_on_metadata_derived_design(self):
        """Key functions in permutation_framework.py must not use
        isinstance(design, MetadataDerivedDesign)."""
        from cliquefinder.stats import permutation_framework

        source = inspect.getsource(permutation_framework)
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "isinstance" and len(node.args) >= 2:
                    # Check if second argument mentions MetadataDerivedDesign
                    second_arg = node.args[1]
                    if isinstance(second_arg, ast.Name) and second_arg.id == "MetadataDerivedDesign":
                        violations.append(node.lineno)

        assert not violations, (
            f"isinstance(design, MetadataDerivedDesign) found at lines {violations}. "
            "Use design.get_conditions(metadata) instead."
        )

    def test_get_design_conditions_helper_with_protocol(self):
        """_get_design_conditions works with objects implementing Protocol."""
        from cliquefinder.stats.permutation_framework import (
            TwoGroupDesign,
            _get_design_conditions,
        )

        design = TwoGroupDesign(
            condition_column="cond",
            test_condition="T",
            reference_condition="C",
        )
        meta = pd.DataFrame({"cond": ["T", "C", "T"]})
        result = _get_design_conditions(design, meta)
        np.testing.assert_array_equal(result, ["T", "C", "T"])

    def test_get_design_conditions_fallback(self):
        """_get_design_conditions falls back to condition_column when
        get_conditions is not available."""
        from cliquefinder.stats.permutation_framework import _get_design_conditions

        class LegacyDesign:
            condition_column = "grp"

        design = LegacyDesign()
        meta = pd.DataFrame({"grp": ["X", "Y", "X"]})
        result = _get_design_conditions(design, meta)
        np.testing.assert_array_equal(result, ["X", "Y", "X"])


# =====================================================================
# ARCH-12: CLI parameter bounds checking
# =====================================================================


class TestCLIBoundsChecking:
    """Verify that invalid numeric CLI parameters are rejected."""

    @staticmethod
    def _build_parser(register_fn):
        """Build an argparse parser with the given subcommand register function."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register_fn(subparsers)
        return parser

    # --- validate-baselines ---

    def test_alpha_rejects_above_1(self):
        """--alpha 2.0 must be rejected."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "validate-baselines",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out", "--network-query", "TP53",
                "--alpha", "2.0",
            ])

    def test_alpha_accepts_valid(self):
        """--alpha 0.05 must be accepted."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        args = parser.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out", "--network-query", "TP53",
            "--alpha", "0.05",
        ])
        assert args.alpha == pytest.approx(0.05)

    def test_alpha_rejects_zero(self):
        """--alpha 0.0 must be rejected (boundary)."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "validate-baselines",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out", "--network-query", "TP53",
                "--alpha", "0.0",
            ])

    def test_n_rotations_rejects_negative(self):
        """--n-rotations -5 must be rejected."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "validate-baselines",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out", "--network-query", "TP53",
                "--n-rotations", "-5",
            ])

    def test_n_rotations_accepts_valid(self):
        """--n-rotations 1000 must be accepted."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        args = parser.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out", "--network-query", "TP53",
            "--n-rotations", "1000",
        ])
        assert args.n_rotations == 1000

    def test_label_permutations_rejects_zero(self):
        """--label-permutations 0 must be rejected."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "validate-baselines",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out", "--network-query", "TP53",
                "--label-permutations", "0",
            ])

    def test_label_permutations_accepts_valid(self):
        """--label-permutations 500 must be accepted."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        args = parser.parse_args([
            "validate-baselines",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out", "--network-query", "TP53",
            "--label-permutations", "500",
        ])
        assert args.label_permutations == 500

    def test_min_evidence_rejects_zero(self):
        """--min-evidence 0 must be rejected."""
        from cliquefinder.cli.validate_baselines import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "validate-baselines",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out", "--network-query", "TP53",
                "--min-evidence", "0",
            ])

    # --- differential ---

    def test_differential_n_rotations_rejects_negative(self):
        """differential --n-rotations -5 must be rejected."""
        from cliquefinder.cli.differential import setup_parser

        parser = self._build_parser(setup_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "differential",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out",
                "--n-rotations", "-5",
            ])

    def test_differential_n_rotations_accepts_valid(self):
        """differential --n-rotations 9999 must be accepted."""
        from cliquefinder.cli.differential import setup_parser

        parser = self._build_parser(setup_parser)
        args = parser.parse_args([
            "differential",
            "--data", "d.csv", "--metadata", "m.csv",
            "--output", "out",
            "--n-rotations", "9999",
        ])
        assert args.n_rotations == 9999

    def test_differential_min_targets_rejects_zero(self):
        """differential --min-targets 0 must be rejected."""
        from cliquefinder.cli.differential import setup_parser

        parser = self._build_parser(setup_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "differential",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out",
                "--min-targets", "0",
            ])

    def test_differential_workers_rejects_zero(self):
        """differential --workers 0 must be rejected."""
        from cliquefinder.cli.differential import setup_parser

        parser = self._build_parser(setup_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "differential",
                "--data", "d.csv", "--metadata", "m.csv",
                "--output", "out",
                "--workers", "0",
            ])

    # --- analyze ---

    def test_analyze_min_evidence_rejects_negative(self):
        """analyze --min-evidence -1 must be rejected."""
        from cliquefinder.cli.analyze import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "analyze",
                "--input", "d.csv",
                "--min-evidence", "-1",
            ])

    def test_analyze_min_targets_rejects_zero(self):
        """analyze --min-targets 0 must be rejected."""
        from cliquefinder.cli.analyze import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "analyze",
                "--input", "d.csv",
                "--min-targets", "0",
            ])

    def test_analyze_workers_rejects_zero(self):
        """analyze --workers 0 must be rejected."""
        from cliquefinder.cli.analyze import register_parser

        parser = self._build_parser(register_parser)
        with pytest.raises(SystemExit):
            parser.parse_args([
                "analyze",
                "--input", "d.csv",
                "--workers", "0",
            ])


# =====================================================================
# ARCH-12: Validator unit tests
# =====================================================================


class TestValidatorFunctions:
    """Unit-test the shared validator functions directly."""

    def test_positive_int_rejects_zero(self):
        from cliquefinder.cli._validators import _positive_int

        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("0")

    def test_positive_int_rejects_negative(self):
        from cliquefinder.cli._validators import _positive_int

        with pytest.raises(argparse.ArgumentTypeError):
            _positive_int("-3")

    def test_positive_int_accepts_positive(self):
        from cliquefinder.cli._validators import _positive_int

        assert _positive_int("42") == 42

    def test_probability_rejects_zero(self):
        from cliquefinder.cli._validators import _probability

        with pytest.raises(argparse.ArgumentTypeError):
            _probability("0.0")

    def test_probability_rejects_one(self):
        from cliquefinder.cli._validators import _probability

        with pytest.raises(argparse.ArgumentTypeError):
            _probability("1.0")

    def test_probability_rejects_above_one(self):
        from cliquefinder.cli._validators import _probability

        with pytest.raises(argparse.ArgumentTypeError):
            _probability("2.0")

    def test_probability_accepts_valid(self):
        from cliquefinder.cli._validators import _probability

        assert _probability("0.05") == pytest.approx(0.05)

    def test_positive_float_rejects_zero(self):
        from cliquefinder.cli._validators import _positive_float

        with pytest.raises(argparse.ArgumentTypeError):
            _positive_float("0.0")

    def test_positive_float_accepts_positive(self):
        from cliquefinder.cli._validators import _positive_float

        assert _positive_float("3.14") == pytest.approx(3.14)

    def test_percentile_rejects_zero(self):
        from cliquefinder.cli._validators import _percentile

        with pytest.raises(argparse.ArgumentTypeError):
            _percentile("0")

    def test_percentile_rejects_100(self):
        from cliquefinder.cli._validators import _percentile

        with pytest.raises(argparse.ArgumentTypeError):
            _percentile("100")

    def test_percentile_accepts_50(self):
        from cliquefinder.cli._validators import _percentile

        assert _percentile("50") == pytest.approx(50.0)


# =====================================================================
# ARCH-18: Warning convention
# =====================================================================


class TestWarningConvention:
    """Verify key modules document the warning convention and use it
    correctly."""

    @pytest.mark.parametrize("module_path", [
        "src/cliquefinder/cli/validate_baselines.py",
        "src/cliquefinder/knowledge/cogex.py",
        "src/cliquefinder/stats/permutation_framework.py",
        "src/cliquefinder/stats/rotation.py",
        "src/cliquefinder/stats/negative_controls.py",
        "src/cliquefinder/stats/bootstrap_stability.py",
    ])
    def test_modules_have_warning_convention_comment(self, module_path):
        """Key modules must contain the warning convention comment."""
        from pathlib import Path

        # Navigate from tests/ up to repo root
        repo_root = Path(__file__).resolve().parent.parent
        full_path = repo_root / module_path
        source = full_path.read_text()
        assert "Warning convention:" in source, (
            f"{module_path} is missing the 'Warning convention:' comment"
        )

    def test_validate_baselines_uses_logger_for_phase_failures(self):
        """validate_baselines.py must use logger.warning (not warnings.warn)
        for phase failure messages."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        source = (repo_root / "src/cliquefinder/cli/validate_baselines.py").read_text()

        # Should NOT have warnings.warn for phase failures
        import re
        phase_failure_warns = re.findall(
            r"warnings\.warn\(.*Phase \d.*failed", source
        )
        assert not phase_failure_warns, (
            "validate_baselines.py should use logger.warning() for phase "
            f"failure messages, found: {phase_failure_warns}"
        )

    def test_negative_controls_uses_logger_for_control_set_failure(self):
        """negative_controls.py must not have actual warnings.warn() calls."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        source = (repo_root / "src/cliquefinder/stats/negative_controls.py").read_text()

        # Use AST to find actual warnings.warn() calls (ignores strings/comments)
        tree = ast.parse(source)
        warn_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "warn"
                        and isinstance(func.value, ast.Name) and func.value.id == "warnings"):
                    warn_calls.append(node.lineno)

        assert not warn_calls, (
            "negative_controls.py should use logger.warning() for internal "
            f"failures, not warnings.warn(). Found at lines: {warn_calls}"
        )

    def test_bootstrap_stability_uses_logger_for_bootstrap_failures(self):
        """bootstrap_stability.py must not have actual warnings.warn() calls."""
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        source = (repo_root / "src/cliquefinder/stats/bootstrap_stability.py").read_text()

        # Use AST to find actual warnings.warn() calls (ignores strings/comments)
        tree = ast.parse(source)
        warn_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "warn"
                        and isinstance(func.value, ast.Name) and func.value.id == "warnings"):
                    warn_calls.append(node.lineno)

        assert not warn_calls, (
            "bootstrap_stability.py should use logger.warning() for internal "
            f"failures, not warnings.warn(). Found at lines: {warn_calls}"
        )
