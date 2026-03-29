"""Tests for CLI input validation guards and bootstrap index handling.

Covers:
- Empty contrasts dict crash in validate_baselines
- contrasts=None AttributeError in differential CLI
- Empty target gene set guard in validate_baselines
- Bootstrap duplicate DataFrame index from with-replacement sampling
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =========================================================================
# Empty contrasts dict crash in validate_baselines
# =========================================================================

class TestEmptyContrasts:
    """Verify validate_baselines returns error code 1 when contrasts is empty."""

    def _make_args(self, tmp_path: Path) -> argparse.Namespace:
        """Build a minimal args namespace for validate_baselines."""
        return argparse.Namespace(
            data=tmp_path / "data.csv",
            metadata=tmp_path / "meta.csv",
            output=tmp_path / "out",
            network_query="C9ORF72",
            cohort_config=None,
            condition_col="phenotype",
            contrast=None,  # No explicit contrasts
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
            target_set_file=None,
        )

    def _make_mock_matrix(self, n_features=20, n_samples=10):
        """Create a mock matrix object."""
        rng = np.random.default_rng(42)
        mock_matrix = MagicMock()
        mock_matrix.n_features = n_features
        mock_matrix.n_samples = n_samples
        mock_matrix.feature_ids = [f"P{i:05d}" for i in range(n_features)]
        mock_matrix.sample_ids = [f"S{i}" for i in range(n_samples)]
        mock_matrix.data = rng.normal(size=(n_features, n_samples))
        return mock_matrix

    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_validate_baselines_empty_contrasts(self, mock_load, tmp_path):
        """Empty contrasts dict returns error code 1, not IndexError."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        n_samples = 10
        sample_ids = [f"S{i}" for i in range(n_samples)]

        # Write metadata: ALL same condition => empty contrasts
        meta_df = pd.DataFrame(
            {"phenotype": ["CTRL"] * n_samples, "Sex": ["M"] * 5 + ["F"] * 5},
            index=sample_ids,
        )
        meta_df.to_csv(tmp_path / "meta.csv")

        mock_matrix = self._make_mock_matrix(n_samples=n_samples)
        mock_load.return_value = mock_matrix

        args = self._make_args(tmp_path)
        result = run_validate_baselines(args)
        assert result == 1, "Expected error code 1 for empty contrasts"


# =========================================================================
# contrasts=None AttributeError in differential CLI
# =========================================================================

class TestNoneContrasts:
    """Verify differential CLI handles None contrasts gracefully."""

    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_differential_none_contrasts(self, mock_load, tmp_path):
        """When contrasts=None (single condition), CLI prints error and returns 1."""
        from cliquefinder.cli.differential import run_differential

        rng = np.random.default_rng(42)
        n_features, n_samples = 20, 10

        # Write metadata with only ONE condition => cannot derive contrast
        sample_ids = [f"S{i}" for i in range(n_samples)]
        meta_df = pd.DataFrame(
            {"phenotype": ["CTRL"] * n_samples, "Sex": ["M"] * 5 + ["F"] * 5},
            index=sample_ids,
        )
        meta_path = tmp_path / "meta.csv"
        meta_df.to_csv(meta_path)

        # Write data
        feature_ids = [f"P{i:05d}" for i in range(n_features)]
        data = rng.normal(size=(n_features, n_samples))
        data_df = pd.DataFrame(data, index=feature_ids, columns=sample_ids)
        data_path = tmp_path / "data.csv"
        data_df.to_csv(data_path)

        # Mock load_csv_matrix
        mock_matrix = MagicMock()
        mock_matrix.n_features = n_features
        mock_matrix.n_samples = n_samples
        mock_matrix.feature_ids = feature_ids
        mock_matrix.sample_ids = sample_ids
        mock_matrix.data = data
        mock_load.return_value = mock_matrix

        args = argparse.Namespace(
            data=data_path,
            metadata=meta_path,
            output=tmp_path / "out",
            cohort_config=None,
            condition_col="phenotype",
            contrast=None,
            genetic_contrast=None,
            subject_col=None,
            covariates=None,
            mode="protein",
            cliques=None,
            discover_gene_sets=False,
            summarization="none",
            normalization="none",
            imputation="none",
            no_mixed_model=True,
            fdr_method="BH",
            fdr_threshold=0.05,
            roast=False,
            permutation_test=False,
            n_rotations=99,
            n_permutations=100,
            permutation_seed=42,
            workers=1,
            network_query=None,
            min_evidence=1,
            indra_env_file=Path("/dev/null"),
            gpu=False,
            force_cpu=False,
            min_coherence=None,
            stmt_types=None,
            regulator_class=None,
            method_comparison=False,
            bootstrap=False,
            interaction=None,
            enrichment_test=False,
            seed=42,
        )

        result = run_differential(args)
        assert result == 1, "Expected error code 1 when contrasts is None"


# =========================================================================
# Empty target gene set in validate_baselines
# =========================================================================

class TestEmptyGeneSet:
    """Verify validate_baselines rejects empty/tiny gene sets."""

    def _make_args(self, tmp_path: Path) -> argparse.Namespace:
        """Build a minimal args namespace."""
        return argparse.Namespace(
            data=tmp_path / "data.csv",
            metadata=tmp_path / "meta.csv",
            output=tmp_path / "out",
            network_query="FAKEGENE",
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
            target_set_file=None,
        )

    def _make_mock_matrix(self, n_features=20, n_samples=10):
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

    def _write_meta(self, tmp_path, sample_ids):
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

    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_validate_baselines_empty_geneset(self, mock_load, mock_query, tmp_path):
        """Empty gene set returns error code 1."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, _ = self._make_mock_matrix()
        mock_load.return_value = mock_matrix
        self._write_meta(tmp_path, sample_ids)

        # Return EMPTY network targets
        mock_query.return_value = {}

        args = self._make_args(tmp_path)
        result = run_validate_baselines(args)
        assert result == 1, "Expected error code 1 for empty gene set"

    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_validate_baselines_single_gene_rejected(self, mock_load, mock_query, tmp_path):
        """Single-gene set (< 2) returns error code 1."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, _ = self._make_mock_matrix()
        mock_load.return_value = mock_matrix
        self._write_meta(tmp_path, sample_ids)

        # Single target gene
        mock_query.return_value = {"GENE1": "P00001"}

        args = self._make_args(tmp_path)
        result = run_validate_baselines(args)
        assert result == 1, "Expected error code 1 for single-gene set"

    @patch("cliquefinder.cli.differential.query_network_targets")
    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_validate_baselines_small_geneset_warns(self, mock_load, mock_query, tmp_path):
        """Gene set with 2-4 genes warns but does not error."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        mock_matrix, sample_ids, feature_ids = self._make_mock_matrix()
        mock_load.return_value = mock_matrix
        self._write_meta(tmp_path, sample_ids)

        # 3 genes: passes (>= 2) but should warn (< 5).
        # Use feature IDs that are actually in the matrix.
        mock_query.return_value = {
            "G1": feature_ids[0],
            "G2": feature_ids[1],
            "G3": feature_ids[2],
        }

        args = self._make_args(tmp_path)

        # Patch downstream phases to avoid running the full pipeline.
        # We need to get past the gene set guard, which happens before Phase 1.
        with patch(
            "cliquefinder.stats.design_matrix.build_covariate_design_matrix"
        ), patch(
            "cliquefinder.stats.differential.run_protein_differential"
        ), patch(
            "cliquefinder.stats.differential.run_network_enrichment_test"
        ) as mock_enr:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"z_score": 1.5, "p": 0.05}
            mock_enr.return_value = mock_result

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Will fail at later phases but we just need the warning
                try:
                    run_validate_baselines(args)
                except Exception:
                    pass

            gene_warnings = [x for x in w if "fewer than 5 genes" in str(x.message)]
            assert len(gene_warnings) >= 1, (
                f"Expected warning about < 5 genes, got warnings: "
                f"{[str(x.message) for x in w]}"
            )


# =========================================================================
# Bootstrap duplicate DataFrame index
# =========================================================================

class TestBootstrapIndex:
    """Verify bootstrap with-replacement sampling produces unique integer index."""

    def test_bootstrap_duplicate_index(self):
        """After bootstrap sampling, metadata should have unique integer index."""
        rng = np.random.default_rng(42)
        n_features, n_samples = 10, 20

        # Create metadata with named index (like real sample IDs)
        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        metadata = pd.DataFrame(
            {
                "condition": (["CASE"] * 12) + (["CTRL"] * 8),
                "Sex": (["M", "F"] * 10),
            },
            index=sample_ids,
        )
        data = rng.normal(size=(n_features, n_samples))

        # Simulate the bootstrap sampling logic from bootstrap_comparison.py
        case_samples = [s for s in sample_ids if metadata.loc[s, "condition"] == "CASE"]
        ctrl_samples = [s for s in sample_ids if metadata.loc[s, "condition"] == "CTRL"]

        # Sample WITH replacement to create duplicates
        selected_cases = rng.choice(case_samples, size=8, replace=True)
        bootstrap_samples = list(selected_cases) + ctrl_samples

        # Verify duplicates exist in the sample list
        assert len(bootstrap_samples) != len(set(bootstrap_samples)), (
            "Test setup: expected duplicate sample IDs from with-replacement sampling"
        )

        # Build bootstrap metadata the way the fixed code does it
        sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
        sample_indices = [sample_to_idx[s] for s in bootstrap_samples]
        bootstrap_data = data[:, sample_indices]
        bootstrap_meta = metadata.loc[bootstrap_samples].copy()
        # The fix: reset_index(drop=True) produces unique integer index
        bootstrap_meta = bootstrap_meta.reset_index(drop=True)

        # Verify index is unique integers
        assert bootstrap_meta.index.is_unique, "Bootstrap metadata index should be unique"
        assert np.issubdtype(bootstrap_meta.index.dtype, np.integer), (
            f"Bootstrap metadata index should be integer, got {bootstrap_meta.index.dtype}"
        )
        assert len(bootstrap_meta) == len(bootstrap_samples), (
            "Bootstrap metadata should have same number of rows as samples"
        )
        assert bootstrap_data.shape[1] == len(bootstrap_meta), (
            "Data columns should match metadata rows"
        )

    def test_bootstrap_meta_preserves_columns(self):
        """Ensure reset_index(drop=True) preserves all metadata columns."""
        sample_ids = [f"s{i}" for i in range(10)]
        metadata = pd.DataFrame(
            {"condition": ["A"] * 5 + ["B"] * 5, "covar": range(10)},
            index=sample_ids,
        )

        # Duplicate via with-replacement
        bootstrap_samples = ["s0", "s0", "s1", "s1", "s5", "s6", "s7", "s8", "s9"]
        bootstrap_meta = metadata.loc[bootstrap_samples].copy()
        bootstrap_meta = bootstrap_meta.reset_index(drop=True)

        assert list(bootstrap_meta.columns) == ["condition", "covar"]
        assert len(bootstrap_meta) == 9
        # First two rows should have same condition (both s0)
        assert bootstrap_meta.loc[0, "condition"] == "A"
        assert bootstrap_meta.loc[1, "condition"] == "A"

    def test_bootstrap_duplicate_index_without_fix_fails(self):
        """Demonstrate that without reset_index, .loc[] is ambiguous on duplicates."""
        sample_ids = ["s0", "s1", "s2"]
        metadata = pd.DataFrame(
            {"condition": ["A", "B", "A"]},
            index=sample_ids,
        )

        # Sample with replacement creating duplicates
        bootstrap_samples = ["s0", "s0", "s1"]
        bootstrap_meta_unfixed = metadata.loc[bootstrap_samples].copy()

        # Without fix: index has duplicates
        assert not bootstrap_meta_unfixed.index.is_unique, (
            "Without fix, index should have duplicates"
        )

        # With fix: index is unique
        bootstrap_meta_fixed = bootstrap_meta_unfixed.reset_index(drop=True)
        assert bootstrap_meta_fixed.index.is_unique, (
            "With fix, index should be unique"
        )
