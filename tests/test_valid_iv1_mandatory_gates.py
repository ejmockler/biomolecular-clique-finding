"""Tests for VALID-IV-1: Mandatory gate abort in validate_baselines.

When Phase 1 or Phase 3 fails with a data-quality exception (ValueError,
np.linalg.LinAlgError), the pipeline must abort immediately — remaining
phases are skipped and no verdict is computed. Non-data exceptions
(e.g., RuntimeError) should still allow continuation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_args(tmp_path: Path) -> argparse.Namespace:
    """Build a minimal args namespace for validate_baselines."""
    return argparse.Namespace(
        data=tmp_path / "data.csv",
        metadata=tmp_path / "meta.csv",
        output=tmp_path / "out",
        network_query="C9ORF72",
        cohort_config=None,
        condition_col="phenotype",
        contrast=[("test_vs_ctrl", "CASE", "CTRL")],
        covariates=["Sex"],
        match_vars=["Sex"],
        label_permutations=10,
        stratify_col="Sex",
        n_neg_controls=10,
        min_evidence=1,
        indra_env_file=Path("/dev/null"),
        n_rotations=99,
        seed=42,
        gpu=False,
        bootstrap_stability=False,
        n_bootstraps=10,
        interaction=False,
        alpha=0.05,
        force_restart=True,
        specificity_z_threshold=1.5,
        neg_ctrl_percentile=10.0,
        interaction_n_perms=10,
    )


def _make_mock_matrix(n_features=20, n_samples=10):
    """Create a mock matrix object."""
    rng = np.random.default_rng(42)
    sample_ids = [f"S{i}" for i in range(n_samples)]
    feature_ids = [f"P{i:05d}" for i in range(n_features)]
    mock_matrix = MagicMock()
    mock_matrix.n_features = n_features
    mock_matrix.n_samples = n_samples
    mock_matrix.feature_ids = feature_ids
    mock_matrix.sample_ids = sample_ids
    mock_matrix.data = rng.normal(size=(n_features, n_samples))
    return mock_matrix, sample_ids, feature_ids


def _write_meta(tmp_path, sample_ids):
    """Write metadata with 2 conditions."""
    n = len(sample_ids)
    half = n // 2
    meta_df = pd.DataFrame(
        {
            "phenotype": ["CASE"] * half + ["CTRL"] * (n - half),
            "Sex": (["M", "F"] * ((n + 1) // 2))[:n],
        },
        index=sample_ids,
    )
    meta_df.to_csv(tmp_path / "meta.csv")


# =========================================================================
# Phase 1: Mandatory gate abort on ValueError / LinAlgError
# =========================================================================


class TestPhase1MandatoryGateAbort:
    """Phase 1 ValueError/LinAlgError aborts the pipeline immediately."""

    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase1_valueerror_aborts(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, tmp_path,
    ):
        """Phase 1 ValueError triggers abort; Phase 3+ never runs."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 raises ValueError (data quality issue)
        mock_diff.side_effect = ValueError("Singular design matrix")

        args = _make_args(tmp_path)
        result = run_validate_baselines(args)

        # Pipeline returns 0 (aborted, not error code 1)
        assert result == 0

        # Validation report should be saved with the failed phase
        report_path = tmp_path / "out" / "validation_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            report_data = json.load(f)
        assert report_data["phases"]["covariate_adjusted"]["status"] == "failed"
        # No Phase 3 or later phases should be present (pipeline aborted)
        assert "label_permutation" not in report_data["phases"]
        assert "matched_reanalysis" not in report_data["phases"]
        assert "negative_controls" not in report_data["phases"]

    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase1_linalgerror_aborts(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, tmp_path,
    ):
        """Phase 1 LinAlgError triggers abort; Phase 3+ never runs."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 raises LinAlgError (numerical issue)
        mock_diff.side_effect = np.linalg.LinAlgError("SVD did not converge")

        args = _make_args(tmp_path)
        result = run_validate_baselines(args)

        assert result == 0

        # Verify abort: only Phase 1 present, no later phases
        report_path = tmp_path / "out" / "validation_report.json"
        with open(report_path) as f:
            report_data = json.load(f)
        assert report_data["phases"]["covariate_adjusted"]["status"] == "failed"
        assert "label_permutation" not in report_data["phases"]

    @patch("cliquefinder.stats.label_permutation.run_label_permutation_null")
    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase1_runtimeerror_continues(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, mock_perm, tmp_path,
    ):
        """Phase 1 RuntimeError does NOT abort — pipeline continues to Phase 3."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 raises RuntimeError (NOT a data quality issue)
        mock_diff.side_effect = RuntimeError("Missing optional dependency")

        # Phase 3 mock (pipeline should reach this)
        mock_perm_result = MagicMock()
        mock_perm_result.to_dict.return_value = {"permutation_pvalue": 0.5}
        mock_perm_result.permutation_pvalue = 0.5
        mock_perm.return_value = mock_perm_result

        args = _make_args(tmp_path)

        # Patch Phase 4 and 5 so the pipeline can complete
        with patch(
            "cliquefinder.stats.matching.exact_match_covariates"
        ) as mock_match:
            mock_match_result = MagicMock()
            mock_match_result.n_original = 10
            mock_match_result.n_matched = 8
            mock_match_result.matched_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            mock_match_result.match_vars = ["Sex"]
            mock_match.return_value = mock_match_result

            # Phase 4 differential will also fail (same mock_diff), that's fine
            with patch(
                "cliquefinder.stats.negative_controls.run_negative_control_sets"
            ) as mock_neg, patch(
                "cliquefinder.stats.rotation.RotationTestEngine"
            ):
                mock_neg_result = MagicMock()
                mock_neg_result.to_dict.return_value = {"fpr": 0.05}
                mock_neg.return_value = mock_neg_result

                try:
                    run_validate_baselines(args)
                except Exception:
                    pass

        # Key check: Phase 3 WAS called (pipeline continued past Phase 1 RuntimeError)
        mock_perm.assert_called()


# =========================================================================
# Phase 3: Mandatory gate abort on ValueError / LinAlgError
# =========================================================================


class TestPhase3MandatoryGateAbort:
    """Phase 3 ValueError/LinAlgError aborts the pipeline immediately."""

    @patch("cliquefinder.stats.label_permutation.run_label_permutation_null")
    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase3_valueerror_aborts(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, mock_perm, tmp_path,
    ):
        """Phase 3 ValueError triggers abort; Phase 4+ never runs."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 succeeds
        mock_diff.return_value = pd.DataFrame({"z_score": [1.0]})
        mock_enrich_result = MagicMock()
        mock_enrich_result.to_dict.return_value = {"empirical_pvalue": 0.01, "z_score": 3.0}
        mock_enrich.return_value = mock_enrich_result

        # Phase 3 raises ValueError
        mock_perm.side_effect = ValueError("All permutation z-scores are NaN")

        args = _make_args(tmp_path)
        result = run_validate_baselines(args)

        # Pipeline returns 0 (aborted)
        assert result == 0

        # Validation report saved with both phases recorded
        report_path = tmp_path / "out" / "validation_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            report_data = json.load(f)
        assert "covariate_adjusted" in report_data["phases"]
        assert report_data["phases"]["label_permutation"]["status"] == "failed"
        # Phase 4 and 5 should NOT be present
        assert "matched_reanalysis" not in report_data["phases"]
        assert "negative_controls" not in report_data["phases"]

    @patch("cliquefinder.stats.label_permutation.run_label_permutation_null")
    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase3_linalgerror_aborts(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, mock_perm, tmp_path,
    ):
        """Phase 3 LinAlgError triggers abort; Phase 4+ never runs."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 succeeds
        mock_diff.return_value = pd.DataFrame({"z_score": [1.0]})
        mock_enrich_result = MagicMock()
        mock_enrich_result.to_dict.return_value = {"empirical_pvalue": 0.01, "z_score": 3.0}
        mock_enrich.return_value = mock_enrich_result

        # Phase 3 raises LinAlgError
        mock_perm.side_effect = np.linalg.LinAlgError("Eigendecomposition failed")

        args = _make_args(tmp_path)
        result = run_validate_baselines(args)

        assert result == 0

        report_path = tmp_path / "out" / "validation_report.json"
        with open(report_path) as f:
            report_data = json.load(f)
        assert report_data["phases"]["label_permutation"]["status"] == "failed"
        assert "matched_reanalysis" not in report_data["phases"]

    @patch("cliquefinder.stats.label_permutation.run_label_permutation_null")
    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase3_runtimeerror_continues(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, mock_perm, tmp_path,
    ):
        """Phase 3 RuntimeError does NOT abort — pipeline continues to Phase 4."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()

        # Phase 1 succeeds
        mock_diff.return_value = pd.DataFrame({"z_score": [1.0]})
        mock_enrich_result = MagicMock()
        mock_enrich_result.to_dict.return_value = {"empirical_pvalue": 0.01, "z_score": 3.0}
        mock_enrich.return_value = mock_enrich_result

        # Phase 3 raises RuntimeError (NOT a data quality issue)
        mock_perm.side_effect = RuntimeError("CUDA not available")

        args = _make_args(tmp_path)

        with patch(
            "cliquefinder.stats.matching.exact_match_covariates"
        ) as mock_match:
            mock_match_result = MagicMock()
            mock_match_result.n_original = 10
            mock_match_result.n_matched = 8
            mock_match_result.matched_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            mock_match_result.match_vars = ["Sex"]
            mock_match.return_value = mock_match_result

            with patch(
                "cliquefinder.stats.negative_controls.run_negative_control_sets"
            ) as mock_neg, patch(
                "cliquefinder.stats.rotation.RotationTestEngine"
            ):
                mock_neg_result = MagicMock()
                mock_neg_result.to_dict.return_value = {"fpr": 0.05}
                mock_neg.return_value = mock_neg_result

                try:
                    run_validate_baselines(args)
                except Exception:
                    pass

            # Key check: Phase 4 (matching) WAS called — pipeline continued
            mock_match.assert_called_once()


# =========================================================================
# Checkpoint persistence on abort
# =========================================================================


class TestAbortCheckpointPersistence:
    """Verify checkpoint and report are saved on mandatory gate abort."""

    @patch("cliquefinder.stats.differential.run_network_enrichment_test")
    @patch("cliquefinder.stats.differential.run_protein_differential")
    @patch("cliquefinder.stats.design_matrix.build_covariate_design_matrix")
    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_phase1_abort_saves_checkpoint(
        self, mock_load, mock_query, mock_design, mock_diff,
        mock_enrich, tmp_path,
    ):
        """Phase 1 abort persists checkpoint so pipeline can resume later."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = _make_mock_matrix()
        mock_load.return_value = mock_matrix
        _write_meta(tmp_path, sample_ids)

        mock_query.return_value = {f"G{i}": feature_ids[i] for i in range(5)}
        mock_design.return_value = MagicMock()
        mock_diff.side_effect = ValueError("Zero-variance features detected")

        args = _make_args(tmp_path)
        run_validate_baselines(args)

        # Both checkpoint and report should exist
        checkpoint_path = tmp_path / "out" / "validation_checkpoint.json"
        report_path = tmp_path / "out" / "validation_report.json"
        assert checkpoint_path.exists(), "Checkpoint should be saved on abort"
        assert report_path.exists(), "Report should be saved on abort"

        # Checkpoint should contain the failed phase
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        assert "covariate_adjusted" in checkpoint["phases"]
        assert checkpoint["phases"]["covariate_adjusted"]["status"] == "failed"
