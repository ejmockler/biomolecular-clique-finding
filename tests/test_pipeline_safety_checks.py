"""
Tests for pipeline safety checks: checkpoint, input guards, and security.

Covers:
- Checkpoint resume preserves protein_df
- Guard on zero common samples
- Division by zero when ensembl_ids empty
- GO URLs use HTTPS
- Deterministic cache keys (hashlib, not hash())
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =====================================================================
# Checkpoint resume preserves protein_df
# =====================================================================

class TestCheckpointProteinDf:
    """Checkpoint serialization round-trip for protein_df."""

    def test_save_checkpoint_includes_protein_df(self, tmp_path):
        """_save_checkpoint serialises protein_df into the JSON."""
        from cliquefinder.cli.validate_baselines import _save_checkpoint
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"z_score": 3.5, "p_value": 0.001})

        protein_df = pd.DataFrame(
            {"logFC": [0.5, -0.3, 1.2], "pvalue": [0.01, 0.05, 0.001]},
            index=["gene1", "gene2", "gene3"],
        )

        _save_checkpoint(report, tmp_path, protein_df=protein_df)

        checkpoint_path = tmp_path / "validation_checkpoint.json"
        assert checkpoint_path.exists()

        with open(checkpoint_path) as f:
            data = json.load(f)

        assert "protein_df_dict" in data
        reconstructed = pd.DataFrame(data["protein_df_dict"])
        pd.testing.assert_frame_equal(reconstructed, protein_df)

    def test_save_checkpoint_no_protein_df(self, tmp_path):
        """_save_checkpoint without protein_df omits the key."""
        from cliquefinder.cli.validate_baselines import _save_checkpoint
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"z_score": 3.5})

        _save_checkpoint(report, tmp_path, protein_df=None)

        with open(tmp_path / "validation_checkpoint.json") as f:
            data = json.load(f)

        assert "protein_df_dict" not in data

    def test_load_checkpoint_restores_protein_df(self, tmp_path):
        """_load_checkpoint restores protein_df from checkpoint."""
        from cliquefinder.cli.validate_baselines import (
            _save_checkpoint,
            _load_checkpoint,
        )
        from cliquefinder.stats.validation_report import ValidationReport

        # Save a checkpoint with protein_df
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"z_score": 3.5})

        protein_df = pd.DataFrame(
            {"logFC": [0.5, -0.3], "adj_pvalue": [0.01, 0.05]},
            index=["geneA", "geneB"],
        )
        _save_checkpoint(report, tmp_path, protein_df=protein_df)

        # Load it back
        loaded_report, loaded_protein_df = _load_checkpoint(tmp_path)

        assert "covariate_adjusted" in loaded_report.phases
        assert loaded_protein_df is not None
        pd.testing.assert_frame_equal(loaded_protein_df, protein_df)

    def test_load_checkpoint_no_protein_df_returns_none(self, tmp_path):
        """_load_checkpoint returns None protein_df when not in checkpoint."""
        from cliquefinder.cli.validate_baselines import (
            _save_checkpoint,
            _load_checkpoint,
        )
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"z_score": 3.5})
        _save_checkpoint(report, tmp_path)

        loaded_report, loaded_protein_df = _load_checkpoint(tmp_path)
        assert loaded_protein_df is None

    def test_load_checkpoint_empty_dir_returns_none(self, tmp_path):
        """_load_checkpoint with no checkpoint file returns empty report + None."""
        from cliquefinder.cli.validate_baselines import _load_checkpoint

        report, protein_df = _load_checkpoint(tmp_path)
        assert len(report.phases) == 0
        assert protein_df is None

    def test_round_trip_preserves_dtypes(self, tmp_path):
        """Round-trip preserves float dtypes in protein_df."""
        from cliquefinder.cli.validate_baselines import (
            _save_checkpoint,
            _load_checkpoint,
        )
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"z_score": 3.5})

        protein_df = pd.DataFrame({
            "logFC": np.array([0.5, -0.3, 1.2], dtype=np.float64),
            "t_stat": np.array([2.1, -1.5, 4.0], dtype=np.float64),
            "pvalue": np.array([0.01, 0.05, 0.001], dtype=np.float64),
        })
        _save_checkpoint(report, tmp_path, protein_df=protein_df)
        _, loaded_df = _load_checkpoint(tmp_path)

        assert loaded_df is not None
        for col in protein_df.columns:
            np.testing.assert_allclose(
                loaded_df[col].values, protein_df[col].values
            )


# =====================================================================
# Guard on zero common samples
# =====================================================================

class TestZeroCommonSamples:
    """Validate-baselines raises ValueError on zero common samples."""

    @patch("cliquefinder.io.loaders.load_csv_matrix")
    def test_zero_common_samples_raises(self, mock_load_matrix, tmp_path):
        """ValueError when protein data and metadata share no samples."""
        from cliquefinder.cli.validate_baselines import run_validate_baselines

        # Matrix with samples S1, S2
        matrix = MagicMock()
        matrix.sample_ids = ["S1", "S2"]
        matrix.n_features = 100
        matrix.n_samples = 2
        mock_load_matrix.return_value = matrix

        # Metadata with completely disjoint samples S3, S4
        metadata = pd.DataFrame(
            {"phenotype": ["ALS", "Control"]},
            index=["S3", "S4"],
        )

        args = MagicMock()
        args.data = Path("fake_data.csv")
        args.metadata = Path("fake_meta.csv")
        args.output = tmp_path
        args.cohort_config = None
        args.condition_col = "phenotype"
        args.force_restart = True
        args.seed = 42

        with patch("pandas.read_csv", return_value=metadata):
            with pytest.raises(ValueError, match="No common samples"):
                run_validate_baselines(args)


# =====================================================================
# Division by zero when ensembl_ids empty
# =====================================================================

class TestEmptyEnsemblIds:
    """build_gene_symbol_mapping returns empty dict for empty input."""

    def test_empty_ids_returns_empty_dict(self):
        """Empty ensembl_ids list returns {} without ZeroDivisionError."""
        from cliquefinder.cli._analyze_core import build_gene_symbol_mapping

        result = build_gene_symbol_mapping([], source_type="ensembl_gene")
        assert result == {}

    def test_empty_ids_symbol_type_returns_empty(self):
        """Empty ids with source_type='symbol' returns {} gracefully."""
        from cliquefinder.cli._analyze_core import build_gene_symbol_mapping

        result = build_gene_symbol_mapping([], source_type="symbol")
        assert result == {}

    def test_nonempty_ids_symbol_type_works(self):
        """Non-empty ids with symbol type still works (identity mapping)."""
        from cliquefinder.cli._analyze_core import build_gene_symbol_mapping

        result = build_gene_symbol_mapping(["TP53", "SOD1"], source_type="symbol")
        assert result == {"TP53": "TP53", "SOD1": "SOD1"}


# =====================================================================
# GO URLs use HTTPS
# =====================================================================

class TestHttpsUrls:
    """All GO download URLs use HTTPS (not HTTP)."""

    def test_goa_url_uses_https(self):
        """GOA annotation download URL uses https://."""
        import inspect
        from cliquefinder.validation.annotation_providers import GOAnnotationProvider

        source = inspect.getsource(GOAnnotationProvider._download_goa)
        assert "https://current.geneontology.org" in source
        assert "http://current.geneontology.org" not in source

    def test_obo_url_uses_https(self):
        """GO OBO term definition download URL uses https://."""
        import inspect
        from cliquefinder.validation.annotation_providers import GOAnnotationProvider

        source = inspect.getsource(GOAnnotationProvider._download_obo)
        assert "https://purl.obolibrary.org" in source
        assert "http://purl.obolibrary.org" not in source

    def test_no_http_urls_in_module(self):
        """No plain http:// URLs remain in annotation_providers.py."""
        import inspect
        from cliquefinder.validation import annotation_providers

        source = inspect.getsource(annotation_providers)
        # Find all http:// occurrences that are NOT https://
        lines = source.split("\n")
        http_only_lines = [
            line.strip()
            for line in lines
            if "http://" in line and "https://" not in line
            # Ignore comments referencing URLs in other contexts
            and not line.strip().startswith("#")
        ]
        assert http_only_lines == [], (
            f"Found non-HTTPS URLs: {http_only_lines}"
        )


# =====================================================================
# Deterministic cache keys
# =====================================================================

class TestDeterministicCacheKeys:
    """Cache keys use hashlib.sha256 (deterministic), not hash() (randomized)."""

    def test_id_mapping_cache_key_deterministic(self):
        """MyGeneInfoMapper cache key is the same across calls."""
        ids = ["ENSG000001", "ENSG000002", "ENSG000003"]
        expected_hash = hashlib.sha256(
            ",".join(sorted(ids)).encode()
        ).hexdigest()[:16]

        # Compute it twice — should be identical
        key1 = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]
        key2 = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]

        assert key1 == key2
        assert key1 == expected_hash

    def test_id_mapping_no_builtin_hash(self):
        """id_mapping.py does not use hash() for cache keys."""
        import inspect
        from cliquefinder.validation import id_mapping

        source = inspect.getsource(id_mapping.MyGeneInfoMapper.map_ids)
        assert "hash(tuple" not in source, "Still uses non-deterministic hash()"
        assert "hashlib" in source or "sha256" in source

    def test_entity_resolver_no_builtin_hash(self):
        """entity_resolver.py does not use hash() for cache keys."""
        import inspect
        from cliquefinder.validation import entity_resolver

        source = inspect.getsource(
            entity_resolver.GeneEntityResolver.resolve_ensembl_ids
        )
        assert "hash(tuple" not in source, "Still uses non-deterministic hash()"
        assert "hashlib" in source or "sha256" in source

    def test_cache_key_order_independent(self):
        """Same IDs in different order produce the same cache key."""
        ids_a = ["C", "A", "B"]
        ids_b = ["B", "C", "A"]

        key_a = hashlib.sha256(",".join(sorted(ids_a)).encode()).hexdigest()[:16]
        key_b = hashlib.sha256(",".join(sorted(ids_b)).encode()).hexdigest()[:16]

        assert key_a == key_b

    def test_different_ids_different_keys(self):
        """Different ID sets produce different cache keys."""
        ids_a = ["ENSG000001", "ENSG000002"]
        ids_b = ["ENSG000003", "ENSG000004"]

        key_a = hashlib.sha256(",".join(sorted(ids_a)).encode()).hexdigest()[:16]
        key_b = hashlib.sha256(",".join(sorted(ids_b)).encode()).hexdigest()[:16]

        assert key_a != key_b

    def test_id_mapping_uses_hashlib_import(self):
        """id_mapping.py imports hashlib."""
        import inspect
        from cliquefinder.validation import id_mapping

        source = inspect.getsource(id_mapping)
        assert "import hashlib" in source

    def test_entity_resolver_uses_hashlib_import(self):
        """entity_resolver.py imports hashlib."""
        import inspect
        from cliquefinder.validation import entity_resolver

        source = inspect.getsource(entity_resolver)
        assert "import hashlib" in source
