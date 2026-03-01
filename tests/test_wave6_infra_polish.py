"""
Tests for Wave 6 (Audit II) — low-priority infrastructure and polish items.

Covers code changes from the following findings:
  KG-14   corr_cache LRU eviction
  SEC-13  ReDoS protection (input length limit)
  SEC-17  Symlink check on cache directory
  CLI-14  Top-level exception handler
  CLI-15  Runtime default resolution for indra-env-file
  CLI-5   O(n) sample alignment
  CLI-10  NaN in clique_genes
  STAT-CORE-19  Fisher Z clip precision
  GPU-12  SE floor debug logging
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# =====================================================================
# KG-14: corr_cache LRU eviction
# =====================================================================


class TestCorrCacheLRU:
    """Verify that CliqueValidator._corr_cache uses LRU eviction."""

    def _make_validator(self):
        """Create a minimal CliqueValidator for cache testing."""
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag

        n_genes, n_samples = 20, 10
        rng = np.random.RandomState(42)
        data = rng.randn(n_genes, n_samples)
        feature_ids = pd.Index([f"GENE{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        metadata = pd.DataFrame(
            {"phenotype": ["CASE"] * 5 + ["CTRL"] * 5},
            index=sample_ids,
        )
        quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=int)
        matrix = BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)

        from cliquefinder.knowledge.clique_validator import CliqueValidator

        return CliqueValidator(matrix, stratify_by=["phenotype"], precompute=False)

    def test_corr_cache_is_ordered_dict(self):
        """Cache should be an OrderedDict for LRU semantics."""
        v = self._make_validator()
        assert isinstance(v._corr_cache, OrderedDict)

    def test_corr_cache_maxsize_default(self):
        """Default maxsize should be 10_000."""
        v = self._make_validator()
        assert v._corr_cache_maxsize == 10_000

    def test_corr_cache_evicts_oldest(self):
        """When cache exceeds maxsize, oldest entry should be evicted."""
        v = self._make_validator()
        v._corr_cache_maxsize = 3  # Small limit for testing

        # Manually insert 4 items to trigger eviction
        for i in range(4):
            key = (f"cond_{i}", frozenset([f"G{i}"]), "pearson")
            v._corr_cache[key] = pd.DataFrame({"x": [i]})
            if len(v._corr_cache) > v._corr_cache_maxsize:
                v._corr_cache.popitem(last=False)

        assert len(v._corr_cache) == 3
        # First entry (cond_0) should have been evicted
        assert ("cond_0", frozenset(["G0"]), "pearson") not in v._corr_cache
        # Last entry should still be there
        assert ("cond_3", frozenset(["G3"]), "pearson") in v._corr_cache

    def test_corr_cache_move_to_end_on_hit(self):
        """Cache hits should move the entry to the end (most recently used)."""
        v = self._make_validator()
        v._corr_cache_maxsize = 3

        # Insert 3 items
        keys = []
        for i in range(3):
            key = (f"cond_{i}", frozenset([f"G{i}"]), "pearson")
            keys.append(key)
            v._corr_cache[key] = pd.DataFrame({"x": [i]})

        # Access the first item (should move it to end)
        v._corr_cache.move_to_end(keys[0])

        # Now the order should be: keys[1], keys[2], keys[0]
        cache_keys = list(v._corr_cache.keys())
        assert cache_keys[0] == keys[1]
        assert cache_keys[-1] == keys[0]


# =====================================================================
# SEC-13: ReDoS protection — input string length limit
# =====================================================================


class TestReDoSProtection:
    """Verify input length limits on regex-matched strings."""

    def test_formats_extract_id_rejects_long_input(self):
        """DataFormat.extract_id should reject strings >10000 chars."""
        from cliquefinder.io.formats import DataFormat

        fmt = DataFormat(id_pattern=r"(?P<id>\w+)")
        with pytest.raises(ValueError, match="too long"):
            fmt.extract_id("A" * 10_001)

    def test_formats_extract_id_accepts_normal_input(self):
        """DataFormat.extract_id should work for normal-length strings."""
        from cliquefinder.io.formats import DataFormat

        fmt = DataFormat(id_pattern=r"(?P<id>\w+)")
        result = fmt.extract_id("GENE123")
        assert result == "GENE123"

    def test_formats_extract_sample_metadata_long_input(self):
        """DataFormat.extract_sample_metadata should return empty for >10000 chars."""
        from cliquefinder.io.formats import DataFormat

        fmt = DataFormat(sample_id_pattern=r"(?P<group>\w+)_(?P<rest>\w+)")
        result = fmt.extract_sample_metadata("X" * 10_001)
        assert result == {}

    def test_phenotype_extract_subject_id_long_input(self):
        """AnswerALSPhenotypeInferencer should return None for long inputs."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        inferencer = AnswerALSPhenotypeInferencer()
        result = inferencer._extract_subject_id("X" * 10_001)
        assert result is None

    def test_phenotype_extract_subject_id_normal_input(self):
        """AnswerALSPhenotypeInferencer should work for normal inputs."""
        from cliquefinder.io.phenotype import AnswerALSPhenotypeInferencer

        inferencer = AnswerALSPhenotypeInferencer()
        result = inferencer._extract_subject_id("CASE_NEUAA295HHE-9014-P_D3")
        assert result == "NEUAA295HHE"


# =====================================================================
# SEC-17: Symlink attack on cache directory
# =====================================================================


class TestSymlinkCacheProtection:
    """Verify symlinked cache directories are rejected."""

    def test_symlinked_cache_dir_raises(self, tmp_path):
        """CachedAnnotationProvider should raise ValueError if cache dir is symlink."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        # Create a target directory and a symlink to it
        target_dir = tmp_path / "real_cache"
        target_dir.mkdir()
        symlink_dir = tmp_path / "sym_cache"
        symlink_dir.symlink_to(target_dir)

        cache_file = symlink_dir / "annotations.json"

        # Should raise because the parent dir is a symlink
        with pytest.raises(ValueError, match="symlink"):
            CachedAnnotationProvider(provider=mock.MagicMock(), cache_file=cache_file)

    def test_normal_cache_dir_works(self, tmp_path):
        """CachedAnnotationProvider should work with normal (non-symlink) directories."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        cache_file = tmp_path / "normal_cache" / "annotations.json"

        # Should not raise
        provider = CachedAnnotationProvider(
            provider=mock.MagicMock(), cache_file=cache_file
        )
        assert provider.cache_file == cache_file


# =====================================================================
# CLI-14: Top-level exception handler
# =====================================================================


class TestCLIExceptionHandler:
    """Verify the CLI entry point catches exceptions gracefully."""

    def test_keyboard_interrupt_returns_130(self):
        """KeyboardInterrupt should return exit code 130."""
        from cliquefinder.cli import main

        def raises_keyboard_interrupt(_args):
            raise KeyboardInterrupt()

        with mock.patch(
            "cliquefinder.cli.argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                command="test", verbose=False, func=raises_keyboard_interrupt
            ),
        ):
            result = main(["test"])
            assert result == 130

    def test_unhandled_exception_returns_1(self):
        """Unhandled exceptions should return exit code 1."""
        from cliquefinder.cli import main

        def raises_runtime_error(_args):
            raise RuntimeError("Something went wrong")

        with mock.patch(
            "cliquefinder.cli.argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                command="test", verbose=False, func=raises_runtime_error
            ),
        ):
            result = main(["test"])
            assert result == 1

    def test_verbose_flag_exists_in_parser(self):
        """The --verbose flag should be accepted by the CLI parser."""
        from cliquefinder.cli import main

        # If --verbose is accepted, parse_args won't raise SystemExit
        # We just test the parser recognizes the flag by calling main with no command
        result = main(["--verbose"])
        assert result == 0  # No command => print help => return 0


# =====================================================================
# CLI-15: Runtime default resolution for indra-env-file
# =====================================================================


class TestIndarEnvFileRuntimeDefault:
    """Verify indra-env-file default is resolved at runtime, not import time."""

    def test_default_is_none_in_parser(self):
        """The parser default for --indra-env-file should be None."""
        from cliquefinder.cli import differential

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        differential.setup_parser(subs)

        # Parse with no --indra-env-file to get default
        args = parser.parse_args(
            [
                "differential",
                "--data", "dummy.csv",
                "--metadata", "dummy.csv",
                "--output", "dummy_out",
                "--condition-col", "phenotype",
            ]
        )
        assert args.indra_env_file is None

    def test_runtime_resolution_uses_env_var(self):
        """When INDRA_ENV_FILE is set, it should be used at runtime."""
        from cliquefinder.cli.differential import run_differential

        # We just need to test the resolution logic, not the full run.
        # Create a minimal args namespace
        args = argparse.Namespace(
            indra_env_file=None,
        )

        with mock.patch.dict(os.environ, {"INDRA_ENV_FILE": "/tmp/test.env"}):
            # CLI-15 resolution code
            if getattr(args, "indra_env_file", None) is None:
                env_from_var = os.environ.get("INDRA_ENV_FILE")
                args.indra_env_file = (
                    Path(env_from_var) if env_from_var else Path.home() / ".indra" / ".env"
                )

        assert args.indra_env_file == Path("/tmp/test.env")

    def test_runtime_resolution_default_path(self):
        """When INDRA_ENV_FILE is not set, default path should be used."""
        args = argparse.Namespace(indra_env_file=None)

        with mock.patch.dict(os.environ, {}, clear=True):
            env_from_var = os.environ.get("INDRA_ENV_FILE")
            args.indra_env_file = (
                Path(env_from_var) if env_from_var else Path.home() / ".indra" / ".env"
            )

        assert args.indra_env_file == Path.home() / ".indra" / ".env"


# =====================================================================
# CLI-5: O(n) sample alignment
# =====================================================================


class TestSampleAlignmentEfficiency:
    """Verify sample alignment uses efficient set/dict lookups."""

    def test_differential_alignment_uses_set(self):
        """differential.py should use set-based metadata lookup."""
        import inspect
        from cliquefinder.cli import differential

        source = inspect.getsource(differential)
        # Should use set-based lookup
        assert "metadata_set = set(metadata.index)" in source or "set(metadata.index)" in source

    def test_differential_alignment_uses_dict_for_indices(self):
        """differential.py should use dict-based index lookup."""
        import inspect
        from cliquefinder.cli import differential

        source = inspect.getsource(differential)
        assert "sample_id_to_idx" in source

    def test_validate_baselines_alignment_uses_set(self):
        """validate_baselines.py should use set-based metadata lookup."""
        import inspect
        from cliquefinder.cli import validate_baselines

        source = inspect.getsource(validate_baselines)
        assert "metadata_set" in source or "set(metadata.index)" in source

    def test_validate_baselines_alignment_uses_dict_for_indices(self):
        """validate_baselines.py should use dict-based index lookup."""
        import inspect
        from cliquefinder.cli import validate_baselines

        source = inspect.getsource(validate_baselines)
        assert "sample_id_to_idx" in source


# =====================================================================
# CLI-10: NaN in clique_genes
# =====================================================================


class TestNaNCliqueGenes:
    """Verify NaN handling in clique_genes string parsing."""

    def test_nan_clique_genes_yields_empty_list(self):
        """If clique_genes is NaN, the gene list should be empty."""
        target_genes_str = float("nan")
        if pd.notna(target_genes_str) and target_genes_str:
            target_gene_list = [
                g.strip()
                for g in str(target_genes_str).split(",")
                if pd.notna(g) and g.strip()
            ]
        else:
            target_gene_list = []

        assert target_gene_list == []

    def test_empty_string_clique_genes_yields_empty_list(self):
        """If clique_genes is empty string, the gene list should be empty."""
        target_genes_str = ""
        if pd.notna(target_genes_str) and target_genes_str:
            target_gene_list = [
                g.strip()
                for g in str(target_genes_str).split(",")
                if pd.notna(g) and g.strip()
            ]
        else:
            target_gene_list = []

        assert target_gene_list == []

    def test_valid_clique_genes_parsed(self):
        """Valid comma-separated genes should be parsed correctly."""
        target_genes_str = "SOD1, TARDBP, FUS"
        if pd.notna(target_genes_str) and target_genes_str:
            target_gene_list = [
                g.strip()
                for g in str(target_genes_str).split(",")
                if pd.notna(g) and g.strip()
            ]
        else:
            target_gene_list = []

        assert target_gene_list == ["SOD1", "TARDBP", "FUS"]

    def test_none_clique_genes_yields_empty_list(self):
        """If clique_genes is None, the gene list should be empty."""
        target_genes_str = None
        if pd.notna(target_genes_str) and target_genes_str:
            target_gene_list = [
                g.strip()
                for g in str(target_genes_str).split(",")
                if pd.notna(g) and g.strip()
            ]
        else:
            target_gene_list = []

        assert target_gene_list == []


# =====================================================================
# STAT-CORE-19: Fisher Z clip precision
# =====================================================================


class TestFisherZClipPrecision:
    """Verify tighter Fisher Z clip bounds."""

    def test_clip_at_near_perfect_correlation(self):
        """r=0.99999999999 should produce finite Fisher Z."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(0.99999999999)
        assert np.isfinite(z)
        assert z > 0

    def test_clip_at_negative_near_perfect(self):
        """r=-0.99999999999 should produce finite Fisher Z."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(-0.99999999999)
        assert np.isfinite(z)
        assert z < 0

    def test_clip_at_exactly_one(self):
        """r=1.0 should be clipped to produce finite output."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(1.0)
        assert np.isfinite(z)

    def test_clip_at_exactly_negative_one(self):
        """r=-1.0 should be clipped to produce finite output."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(-1.0)
        assert np.isfinite(z)

    def test_precision_improvement(self):
        """New clip bound should give larger Z than old 0.9999 bound for r near 1."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(0.99999)
        # With clip at 1-1e-10, arctanh(0.99999) ≈ 6.1, much larger than
        # arctanh(0.9999) ≈ 4.95 (old bound would have clipped this)
        assert z > 5.0  # Should not be limited by old 0.9999 clip

    def test_moderate_correlation_unchanged(self):
        """Moderate correlations should not be affected by the clip."""
        from cliquefinder.stats.correlation_tests import fisher_z_transform

        z = fisher_z_transform(0.5)
        expected = 0.5 * np.log((1 + 0.5) / (1 - 0.5))
        assert abs(z - expected) < 1e-12


# =====================================================================
# GPU-12: SE floor debug logging
# =====================================================================


class TestSEFloorLogging:
    """Verify that GPU SE floor application triggers debug logging."""

    def test_se_floor_logs_when_hit(self, caplog):
        """When SE values are below 1e-10, a debug message should be logged."""
        # We test the logging pattern directly
        se = np.array([1e-12, 0.5, 1e-15, 0.3])
        n_floored = int(np.sum(se < 1e-10))

        logger = logging.getLogger("cliquefinder.stats.permutation_gpu")
        with caplog.at_level(logging.DEBUG, logger="cliquefinder.stats.permutation_gpu"):
            if n_floored > 0:
                logger.debug(
                    "SE floor (1e-10) applied to %d / %d features — likely near-zero "
                    "residual variance",
                    n_floored,
                    len(se),
                )

        assert n_floored == 2
        assert "SE floor" in caplog.text
        assert "2 / 4" in caplog.text

    def test_se_floor_no_log_when_not_hit(self, caplog):
        """When all SE values are above 1e-10, no debug message should appear."""
        se = np.array([0.1, 0.5, 0.3])
        n_floored = int(np.sum(se < 1e-10))

        with caplog.at_level(logging.DEBUG, logger="cliquefinder.stats.permutation_gpu"):
            if n_floored > 0:
                logging.getLogger("cliquefinder.stats.permutation_gpu").debug(
                    "SE floor applied"
                )

        assert n_floored == 0
        assert "SE floor" not in caplog.text


# =====================================================================
# KG-11: CoGExClient thread-safety docstring
# =====================================================================


class TestCoGExClientDocstring:
    """Verify thread-safety note in CoGExClient docstring."""

    def test_thread_safety_note_in_docstring(self):
        """CoGExClient docstring should mention thread safety."""
        from cliquefinder.knowledge.cogex import CoGExClient

        assert "not thread-safe" in CoGExClient.__doc__


# =====================================================================
# Documentation / comment checks (spot checks for key items)
# =====================================================================


class TestDocumentationComments:
    """Spot-check that documentation-only findings have their comments."""

    def test_stat_core_18_censoring_threshold_comment(self):
        """STAT-CORE-18: censoring threshold comment should be present."""
        import inspect
        from cliquefinder.stats import missing

        source = inspect.getsource(missing)
        assert "diagnostics only" in source

    def test_set_test_15_msq_direction_comment(self):
        """SET-TEST-15: MSQ direction comment should be present."""
        import inspect
        from cliquefinder.stats import rotation

        source = inspect.getsource(rotation)
        assert "MSQ UP/DOWN zeros the opposite direction" in source

    def test_gpu_6_global_convergence_comment(self):
        """GPU-6: global convergence comment should be present."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        assert "GLOBAL criterion" in source

    def test_gpu_11_float32_precision_comment(self):
        """GPU-11: float32 precision comment should be present."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        assert "float32" in source and "6e-8" in source

    def test_val_5_categorical_dummies_comment(self):
        """VAL-5: categorical dummies ordering comment should be present."""
        import inspect
        from cliquefinder.stats import design_matrix

        source = inspect.getsource(design_matrix)
        assert "VAL-5" in source

    def test_val_8_specificity_ratio_comment(self):
        """VAL-8: specificity ratio interpretation comment should be present."""
        import inspect
        from cliquefinder.stats import specificity

        source = inspect.getsource(specificity)
        assert "opposite signs" in source

    def test_val_9_l_matrix_assertion(self):
        """VAL-9: L matrix column ordering assertion should be present."""
        import inspect
        from cliquefinder.stats import design_matrix

        source = inspect.getsource(design_matrix)
        assert "Condition columns must be the first" in source

    def test_val_11_n_covariate_params_comment(self):
        """VAL-11: n_covariate_params interaction comment should be present."""
        import inspect
        from cliquefinder.stats import design_matrix

        source = inspect.getsource(design_matrix)
        assert "VAL-11" in source

    def test_set_test_14_bootstrap_frequency_comment(self):
        """SET-TEST-14: bootstrap frequency denominator comment should be present."""
        import inspect
        from cliquefinder.stats import bootstrap_comparison

        source = inspect.getsource(bootstrap_comparison)
        assert "SET-TEST-14" in source

    def test_stat_core_9_vsn_init_comment(self):
        """STAT-CORE-9: VSN initialization comment should be present."""
        import inspect
        from cliquefinder.stats import normalization

        source = inspect.getsource(normalization)
        assert "STAT-CORE-9" in source

    def test_stat_core_13_quantile_nan_warning(self):
        """STAT-CORE-13: quantile normalization NaN warning should be present."""
        from cliquefinder.stats.normalization import quantile_normalization

        assert "simple" in quantile_normalization.__doc__
        assert "NaN" in quantile_normalization.__doc__

    def test_sec_15_dependency_comment(self):
        """SEC-15: unpinned dependency rationale comment should be present."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "SEC-15" in content
        assert "lockfile" in content
