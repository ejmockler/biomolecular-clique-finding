"""
Tests for Wave 4 infrastructure improvements:
  ARCH-10  SeedSequence adoption
  ARCH-11  Atomic file writes
  ARCH-16  Vectorized random sampling
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import textwrap
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# =====================================================================
# ARCH-10: SeedSequence adoption
# =====================================================================


class TestSeedSequenceAdoption:
    """Verify that seed arithmetic (seed + N) is replaced by SeedSequence."""

    def test_validate_baselines_no_seed_arithmetic(self):
        """Confirm ``_base_seed + N`` pattern is gone from validate_baselines."""
        from cliquefinder.cli import validate_baselines as vb_mod

        source = inspect.getsource(vb_mod.run_validate_baselines)
        # The old pattern was: ``_base_seed + 1000``, ``_base_seed + 2000``, etc.
        assert "_base_seed + 1000" not in source
        assert "_base_seed + 2000" not in source
        assert "_base_seed + 3000" not in source
        assert "_base_seed + 4000" not in source
        # SeedSequence should be present
        assert "SeedSequence" in source

    def test_spawned_seeds_are_different(self):
        """Seeds derived via SeedSequence.spawn() + generate_state() are distinct."""
        from numpy.random import SeedSequence

        ss = SeedSequence(42)
        children = ss.spawn(5)
        seeds = [int(c.generate_state(1)[0]) for c in children]
        # All five must be unique
        assert len(set(seeds)) == 5, (
            f"Expected 5 unique seeds, got {len(set(seeds))}: {seeds}"
        )

    def test_spawned_seeds_produce_independent_streams(self):
        """RNG streams from spawned seeds must be statistically independent.

        Generate 1000 values from each of two spawned seeds and check
        that their Pearson correlation is below 0.1 (no linear
        relationship beyond random noise).
        """
        from numpy.random import SeedSequence

        ss = SeedSequence(42)
        child_a, child_b = ss.spawn(2)
        rng_a = np.random.default_rng(int(child_a.generate_state(1)[0]))
        rng_b = np.random.default_rng(int(child_b.generate_state(1)[0]))
        vals_a = rng_a.random(1000)
        vals_b = rng_b.random(1000)
        corr = np.corrcoef(vals_a, vals_b)[0, 1]
        assert abs(corr) < 0.1, f"Correlation too high: {corr:.4f}"

    def test_bootstrap_comparison_no_seed_arithmetic(self):
        """Confirm ``config.seed + b`` is not used as executable code in bootstrap_comparison.

        The comment describing the old pattern is allowed; only actual
        executable usage (``seed=config.seed + b``) must be gone.
        """
        from cliquefinder.stats import bootstrap_comparison as bc_mod

        source = inspect.getsource(bc_mod)
        # ``seed=config.seed + b`` was the old assignment — it should be gone
        assert "seed=config.seed + b" not in source
        assert "SeedSequence" in source


# =====================================================================
# ARCH-11: Atomic file writes
# =====================================================================


class TestAtomicWriteJson:
    """Tests for ``atomic_write_json``."""

    def test_write_succeeds_content_correct(self, tmp_path):
        """File written by atomic_write_json has correct JSON content."""
        from cliquefinder.utils.fileio import atomic_write_json

        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42, "nested": [1, 2, 3]}
        atomic_write_json(path, data)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_no_tmp_file_left_on_success(self, tmp_path):
        """No .tmp files remain after a successful write."""
        from cliquefinder.utils.fileio import atomic_write_json

        path = tmp_path / "clean.json"
        atomic_write_json(path, {"ok": True})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"

    def test_no_file_on_serialization_error(self, tmp_path):
        """If JSON serialization fails, no destination or temp file is left."""
        from cliquefinder.utils.fileio import atomic_write_json

        path = tmp_path / "should_not_exist.json"

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            atomic_write_json(path, {"bad": Unserializable()})

        assert not path.exists(), "Destination file should not exist after failure"
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"

    def test_validation_report_uses_atomic_write(self):
        """ValidationReport.save() delegates to atomic_write_json."""
        from cliquefinder.stats.validation_report import ValidationReport

        source = inspect.getsource(ValidationReport.save)
        assert "atomic_write_json" in source


class TestAtomicWriteText:
    """Tests for ``atomic_write_text``."""

    def test_write_succeeds(self, tmp_path):
        """Text file written atomically has correct content."""
        from cliquefinder.utils.fileio import atomic_write_text

        path = tmp_path / "output.txt"
        atomic_write_text(path, "hello world\n")

        assert path.exists()
        assert path.read_text() == "hello world\n"


# =====================================================================
# ARCH-16: Vectorized random sampling
# =====================================================================


class TestVectorizedSampling:
    """Tests for the argpartition-based vectorized sampling in
    ``precompute_random_indices``."""

    def test_correct_shape(self):
        """Output arrays have expected (n_perms, clique_size) shape."""
        from cliquefinder.stats.permutation_gpu import precompute_random_indices

        clique_sizes = {"C1": 5, "C2": 10}
        pool_size = 2000
        n_perms = 100

        result, unique_sizes = precompute_random_indices(
            clique_sizes, pool_size, n_perms, random_state=42
        )

        assert result["C1"].shape == (100, 5)
        assert result["C2"].shape == (100, 10)
        assert unique_sizes == [5, 10]

    def test_indices_within_bounds(self):
        """All sampled indices are in [0, pool_size)."""
        from cliquefinder.stats.permutation_gpu import precompute_random_indices

        pool_size = 500
        clique_sizes = {"C1": 10, "C2": 20}
        result, _ = precompute_random_indices(
            clique_sizes, pool_size, 200, random_state=7
        )

        for cid, arr in result.items():
            assert arr.min() >= 0, f"{cid}: negative index"
            assert arr.max() < pool_size, f"{cid}: index >= pool_size"

    def test_no_duplicate_indices_per_row(self):
        """Each row (single permutation) has no duplicate indices — sampling
        without replacement."""
        from cliquefinder.stats.permutation_gpu import precompute_random_indices

        clique_sizes = {"C1": 30}
        pool_size = 5000
        n_perms = 500
        result, _ = precompute_random_indices(
            clique_sizes, pool_size, n_perms, random_state=99
        )

        arr = result["C1"]  # (500, 30)
        for i in range(arr.shape[0]):
            row = arr[i]
            assert len(set(row)) == len(row), (
                f"Duplicate indices in permutation {i}: {row}"
            )

    def test_distribution_matches_sequential(self):
        """Vectorized and sequential approaches produce samples from the
        same uniform-without-replacement distribution.

        We check that the marginal frequency of each index is
        approximately ``size / pool_size`` (uniform) across many samples.
        """
        from cliquefinder.stats.permutation_gpu import precompute_random_indices

        pool_size = 200
        size = 10
        n_perms = 5000
        clique_sizes = {"C1": size}

        result, _ = precompute_random_indices(
            clique_sizes, pool_size, n_perms, random_state=123
        )
        arr = result["C1"]  # (5000, 10)
        flat = arr.ravel()

        # Each index should appear roughly n_perms * size / pool_size = 250 times
        counts = np.bincount(flat, minlength=pool_size)
        expected = n_perms * size / pool_size  # 250
        # Chi-squared goodness-of-fit test: should NOT reject uniformity
        from scipy.stats import chisquare

        stat, pvalue = chisquare(counts, f_exp=np.full(pool_size, expected))
        assert pvalue > 0.01, (
            f"Distribution deviates from uniform: chi2={stat:.1f}, p={pvalue:.4f}"
        )

    def test_fallback_for_small_pool(self):
        """When pool_size <= 2 * size, the sequential fallback is used.

        This is a functional test — the result should still be correct.
        """
        from cliquefinder.stats.permutation_gpu import precompute_random_indices

        # pool_size=20, size=15 → pool_size <= 2*size triggers fallback
        clique_sizes = {"C1": 15}
        pool_size = 20
        n_perms = 100

        result, _ = precompute_random_indices(
            clique_sizes, pool_size, n_perms, random_state=42
        )

        arr = result["C1"]
        assert arr.shape == (100, 15)
        # Verify correctness: no duplicates, within bounds
        for i in range(arr.shape[0]):
            row = arr[i]
            assert len(set(row)) == len(row)
            assert row.min() >= 0
            assert row.max() < pool_size

    def test_source_has_argpartition(self):
        """The vectorized path uses np.argpartition (not a for-loop)."""
        from cliquefinder.stats import permutation_gpu as pg_mod

        source = inspect.getsource(pg_mod.precompute_random_indices)
        assert "argpartition" in source
