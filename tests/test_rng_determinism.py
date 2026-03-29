"""Tests for RNG determinism and SeedSequence integration.

Covers:
- Legacy np.random.seed replaced with np.random.default_rng in all call sites.
- Reproducibility of permutation and matched-comparison functions with fixed seeds.
- Phase 2 specificity seed derived from SeedSequence hierarchy in validate_baselines.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# =====================================================================
# Helpers: read source directly from files (worktree-safe)
# =====================================================================

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cliquefinder"


def _read_function_source(filepath: Path, func_name: str) -> str:
    """Extract a function's source from a file by name.

    Reads the file, finds 'def func_name(', then extracts all lines
    until the next top-level def/class or end of file.
    """
    lines = filepath.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def {func_name}\(", line):
            start = i
            break
    if start is None:
        raise ValueError(f"Function {func_name} not found in {filepath}")

    # Collect function body (until next top-level def/class or EOF)
    func_lines = [lines[start]]
    for line in lines[start + 1:]:
        if re.match(r"^(def |class )", line):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def _parse_function(filepath: Path, func_name: str) -> ast.Module:
    """Parse a function's source into an AST."""
    source = _read_function_source(filepath, func_name)
    return ast.parse(textwrap.dedent(source))


def _find_legacy_calls(tree: ast.Module, attr_name: str) -> list[ast.Call]:
    """Find all np.random.<attr_name>(...) calls in an AST."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and func.attr == attr_name
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "random"
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "np"):
                calls.append(node)
    return calls


# =====================================================================
# No legacy np.random.seed in targeted functions
# =====================================================================

_CLIQUE_ANALYSIS = _SRC_ROOT / "stats" / "clique_analysis.py"
_PERMUTATION_GPU = _SRC_ROOT / "stats" / "permutation_gpu.py"


class TestNoLegacySeed:
    """Verify np.random.seed/choice/randn not called in targeted functions."""

    def test_run_permutation_clique_test_no_legacy_seed(self):
        tree = _parse_function(_CLIQUE_ANALYSIS, "run_permutation_clique_test")
        assert len(_find_legacy_calls(tree, "seed")) == 0, \
            "run_permutation_clique_test still uses np.random.seed"

    def test_run_permutation_clique_test_no_legacy_choice(self):
        tree = _parse_function(_CLIQUE_ANALYSIS, "run_permutation_clique_test")
        assert len(_find_legacy_calls(tree, "choice")) == 0, \
            "run_permutation_clique_test still uses np.random.choice"

    def test_run_permutation_clique_test_uses_default_rng(self):
        source = _read_function_source(_CLIQUE_ANALYSIS, "run_permutation_clique_test")
        assert "default_rng" in source, \
            "run_permutation_clique_test should use np.random.default_rng"

    def test_run_matched_single_gene_no_legacy_seed(self):
        tree = _parse_function(_CLIQUE_ANALYSIS, "run_matched_single_gene_comparison")
        assert len(_find_legacy_calls(tree, "seed")) == 0, \
            "run_matched_single_gene_comparison still uses np.random.seed"

    def test_run_matched_single_gene_no_legacy_choice(self):
        tree = _parse_function(_CLIQUE_ANALYSIS, "run_matched_single_gene_comparison")
        assert len(_find_legacy_calls(tree, "choice")) == 0, \
            "run_matched_single_gene_comparison still uses np.random.choice"

    def test_run_matched_single_gene_uses_default_rng(self):
        source = _read_function_source(_CLIQUE_ANALYSIS, "run_matched_single_gene_comparison")
        assert "default_rng" in source, \
            "run_matched_single_gene_comparison should use np.random.default_rng"

    def test_validate_ols_no_legacy_seed(self):
        tree = _parse_function(_PERMUTATION_GPU, "validate_ols_implementation")
        assert len(_find_legacy_calls(tree, "seed")) == 0, \
            "validate_ols_implementation still uses np.random.seed"

    def test_validate_ols_no_legacy_randn(self):
        tree = _parse_function(_PERMUTATION_GPU, "validate_ols_implementation")
        assert len(_find_legacy_calls(tree, "randn")) == 0, \
            "validate_ols_implementation still uses np.random.randn"

    def test_validate_ols_no_legacy_choice(self):
        tree = _parse_function(_PERMUTATION_GPU, "validate_ols_implementation")
        assert len(_find_legacy_calls(tree, "choice")) == 0, \
            "validate_ols_implementation still uses np.random.choice"

    def test_validate_ols_uses_default_rng(self):
        source = _read_function_source(_PERMUTATION_GPU, "validate_ols_implementation")
        assert "default_rng" in source, \
            "validate_ols_implementation should use np.random.default_rng"


# =====================================================================
# Reproducibility with default_rng
# =====================================================================


class TestValidateOlsReproducibility:
    """Verify validate_ols_implementation produces reproducible results."""

    def test_same_seed_same_results(self):
        """Same seed should produce identical validation metrics."""
        from cliquefinder.stats.permutation_gpu import validate_ols_implementation
        r1 = validate_ols_implementation(
            n_samples=30, n_features=10, n_conditions=2, random_state=42
        )
        r2 = validate_ols_implementation(
            n_samples=30, n_features=10, n_conditions=2, random_state=42
        )
        assert r1["max_beta_diff"] == r2["max_beta_diff"]
        assert r1["max_t_diff"] == r2["max_t_diff"]
        assert r1["all_close"] == r2["all_close"]

    def test_different_seed_different_results(self):
        """Different seeds should produce different data (at least usually)."""
        from cliquefinder.stats.permutation_gpu import validate_ols_implementation
        r1 = validate_ols_implementation(
            n_samples=30, n_features=10, n_conditions=2, random_state=42
        )
        r2 = validate_ols_implementation(
            n_samples=30, n_features=10, n_conditions=2, random_state=99
        )
        assert r1["mean_beta_diff"] != r2["mean_beta_diff"] or \
               r1["mean_t_diff"] != r2["mean_t_diff"]

    def test_validation_still_passes(self):
        """Validation should still report all_close=True (numerical correctness)."""
        from cliquefinder.stats.permutation_gpu import validate_ols_implementation
        r = validate_ols_implementation(
            n_samples=50, n_features=20, n_conditions=2, random_state=42
        )
        assert r["all_close"], (
            f"OLS validation failed after RNG migration: "
            f"max_beta_diff={r['max_beta_diff']:.2e}, max_t_diff={r['max_t_diff']:.2e}"
        )


# =====================================================================
# Reproducibility of clique_analysis functions
# =====================================================================


def _make_clique_test_data(n_features=20, n_samples=12):
    """Create minimal synthetic data for clique analysis functions."""
    from cliquefinder.stats.clique_analysis import CliqueDefinition

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_features, n_samples))
    feature_ids = [f"gene_{i}" for i in range(n_features)]

    # Create sample metadata with two conditions, 6 samples each
    conditions = ["ctrl"] * 6 + ["treat"] * 6
    metadata = pd.DataFrame({
        "condition": conditions,
        "subject_id": [f"S{i}" for i in range(n_samples)],
    })

    # Two cliques using subsets of features
    cliques = [
        CliqueDefinition(
            clique_id="clique_A",
            regulator="TF1",
            protein_ids=["gene_0", "gene_1", "gene_2"],
        ),
        CliqueDefinition(
            clique_id="clique_B",
            regulator="TF2",
            protein_ids=["gene_3", "gene_4", "gene_5"],
        ),
    ]

    return data, feature_ids, metadata, cliques


class TestPermutationCliqueTestReproducibility:
    """Verify run_permutation_clique_test is reproducible with same seed."""

    def test_same_seed_same_pvalues(self):
        """Same random_state should produce identical p-values."""
        from cliquefinder.stats.clique_analysis import run_permutation_clique_test

        data, feature_ids, metadata, cliques = _make_clique_test_data()

        results1, _ = run_permutation_clique_test(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            n_permutations=20,
            use_mixed_model=False,
            random_state=42,
            map_ids=False,
            verbose=False,
        )

        results2, _ = run_permutation_clique_test(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            n_permutations=20,
            use_mixed_model=False,
            random_state=42,
            map_ids=False,
            verbose=False,
        )

        # Same seed => same results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.empirical_pvalue == r2.empirical_pvalue, \
                f"P-values differ for {r1.clique_id}: {r1.empirical_pvalue} != {r2.empirical_pvalue}"

    def test_none_seed_still_works(self):
        """random_state=None should not raise (uses entropy from OS)."""
        from cliquefinder.stats.clique_analysis import run_permutation_clique_test

        data, feature_ids, metadata, cliques = _make_clique_test_data()

        results, _ = run_permutation_clique_test(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            n_permutations=5,
            use_mixed_model=False,
            random_state=None,
            map_ids=False,
            verbose=False,
        )
        assert isinstance(results, list)


class TestMatchedSingleGeneReproducibility:
    """Verify run_matched_single_gene_comparison is reproducible with same seed."""

    def test_same_seed_same_results(self):
        """Same random_state should produce identical comparison results."""
        from cliquefinder.stats.clique_analysis import run_matched_single_gene_comparison

        data, feature_ids, metadata, cliques = _make_clique_test_data()

        df1 = run_matched_single_gene_comparison(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            use_mixed_model=False,
            random_state=42,
            verbose=False,
        )

        df2 = run_matched_single_gene_comparison(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            use_mixed_model=False,
            random_state=42,
            verbose=False,
        )

        # Same seed => same results
        pd.testing.assert_frame_equal(df1, df2)

    def test_none_seed_still_works(self):
        """random_state=None should not raise (uses entropy from OS)."""
        from cliquefinder.stats.clique_analysis import run_matched_single_gene_comparison

        data, feature_ids, metadata, cliques = _make_clique_test_data()

        df = run_matched_single_gene_comparison(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col="condition",
            contrast=("ctrl", "treat"),
            subject_col="subject_id",
            use_mixed_model=False,
            random_state=None,
            verbose=False,
        )
        assert isinstance(df, pd.DataFrame)


# =====================================================================
# Phase 2 SeedSequence integration
# =====================================================================

_VALIDATE_BASELINES = _SRC_ROOT / "cli" / "validate_baselines.py"


class TestPhase2SeedSequence:
    """Verify Phase 2 seed is derived from SeedSequence hierarchy."""

    def test_phase2_seed_in_spawn(self):
        """SeedSequence must spawn 6 streams (not 5) to include Phase 2."""
        from numpy.random import SeedSequence

        _ss = SeedSequence(42)
        children = _ss.spawn(6)
        assert len(children) == 6, "SeedSequence should spawn 6 children"

        # Destructure as in the code
        (_ss_boot, _ss_p3s, _ss_p3f, _ss_p4, _ss_p5, _ss_p2) = children

        # All should be SeedSequence instances
        for ss in children:
            assert isinstance(ss, SeedSequence)

    def test_phase2_seed_differs_from_others(self):
        """Phase 2 seed must be distinct from all other phase seeds."""
        from numpy.random import SeedSequence

        _ss = SeedSequence(42)
        (_ss_boot, _ss_p3s, _ss_p3f, _ss_p4, _ss_p5, _ss_p2) = _ss.spawn(6)

        seeds = {
            "bootstrap": int(_ss_boot.generate_state(1)[0]),
            "phase2": int(_ss_p2.generate_state(1)[0]),
            "phase3_strat": int(_ss_p3s.generate_state(1)[0]),
            "phase3_free": int(_ss_p3f.generate_state(1)[0]),
            "phase4": int(_ss_p4.generate_state(1)[0]),
            "phase5": int(_ss_p5.generate_state(1)[0]),
        }

        # All seeds must be unique
        seed_values = list(seeds.values())
        assert len(set(seed_values)) == len(seed_values), \
            f"Duplicate seeds found: {seeds}"

    def test_all_six_streams_unique(self):
        """All 6 SeedSequence-spawned streams must produce unique seeds."""
        from numpy.random import SeedSequence

        _ss = SeedSequence(42)
        children = _ss.spawn(6)
        seeds = [int(c.generate_state(1)[0]) for c in children]
        assert len(set(seeds)) == 6, "All 6 phase seeds must be unique"

    def test_phase2_seed_is_none_when_base_seed_is_none(self):
        """When base seed is None, phase2 seed should also be None."""
        _base_seed = None
        if _base_seed is not None:
            from numpy.random import SeedSequence
            _ss = SeedSequence(_base_seed)
            (*_, _ss_p2) = _ss.spawn(6)
            _seed_phase2 = int(_ss_p2.generate_state(1)[0])
        else:
            _seed_phase2 = None

        assert _seed_phase2 is None

    def test_validate_baselines_source_has_spawn8(self):
        """validate_baselines.py must use spawn(8) to include Phase 2 + graph perm + proximity."""
        source = _VALIDATE_BASELINES.read_text()
        assert ".spawn(8)" in source, \
            "SeedSequence.spawn(8) not found in validate_baselines.py"

    def test_validate_baselines_source_has_phase2_seed(self):
        """validate_baselines.py must define _seed_phase2."""
        source = _VALIDATE_BASELINES.read_text()
        assert "_seed_phase2" in source, \
            "_seed_phase2 not found in validate_baselines.py"

    def test_validate_baselines_source_phase2_uses_derived_seed(self):
        """Phase 2 compute_specificity must use _seed_phase2, not args.seed."""
        source = _VALIDATE_BASELINES.read_text()
        assert "seed=_seed_phase2" in source, \
            "Phase 2 compute_specificity should use seed=_seed_phase2"

    def test_validate_baselines_phase2_section_no_args_seed(self):
        """In the Phase 2 section, seed=args.seed must not appear."""
        source = _VALIDATE_BASELINES.read_text()

        # Find the Phase 2 section boundaries
        phase2_start = source.find("PHASE 2: MULTI-CONTRAST SPECIFICITY")
        phase3_start = source.find("PHASE 3: LABEL PERMUTATION NULL")
        assert phase2_start > 0, "Could not find Phase 2 section"
        assert phase3_start > 0, "Could not find Phase 3 section"

        phase2_section = source[phase2_start:phase3_start]
        assert "seed=args.seed" not in phase2_section, \
            "Phase 2 still uses seed=args.seed instead of _seed_phase2"
        assert "seed=_seed_phase2" in phase2_section, \
            "Phase 2 does not use seed=_seed_phase2"

    def test_phase2_and_graph_seeds_appended_at_end_of_spawn(self):
        """_ss_p2 and _ss_p5g must be the last elements in the spawn tuple.

        This ensures existing seeds for phases 3-5 are not perturbed
        by the addition of Phase 2 and graph permutation seeds.
        """
        source = _VALIDATE_BASELINES.read_text()
        # Find the spawn destructuring line
        match = re.search(r"\(([^)]+)\)\s*=\s*_ss\.spawn\(7\)", source)
        assert match is not None, "Could not find spawn(7) destructuring"
        names = [n.strip() for n in match.group(1).split(",")]
        assert names[-2] == "_ss_p2", \
            f"_ss_p2 must be second-to-last in spawn tuple, got: {names}"
        assert names[-1] == "_ss_p5g", \
            f"_ss_p5g must be last in spawn tuple, got: {names}"
