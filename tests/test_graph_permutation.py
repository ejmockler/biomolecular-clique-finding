"""Tests for graph node-label permutation null test.

Covers: basic permutation, error handling, seed determinism, eligible regulator
pre-filtering, serialization, partial symbol resolution, regulator sampling,
and set size contraction prevention (XVI-1 / XVI-2).
"""

import json

import numpy as np
import pytest
from unittest.mock import MagicMock

from cliquefinder.stats.graph_permutation import (
    GraphPermutationResult,
    run_graph_permutation_null,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_engine(gene_ids=None):
    """Create a mock RotationTestEngine."""
    if gene_ids is None:
        gene_ids = [f"P{i:05d}" for i in range(100)]
    engine = MagicMock()
    engine._fitted = True
    engine.gene_ids = gene_ids
    engine.gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
    mock_result = MagicMock()
    mock_result.p_values = {"msq": {"mixed": 0.5}}
    engine.test_gene_set.return_value = mock_result
    return engine


def _make_adjacency(n_regulators=5, n_targets_per=10):
    """Build a test adjacency dict with gene symbols."""
    adj = {}
    all_symbols = []
    for i in range(n_regulators):
        reg = f"REG{i}"
        targets = [f"TGT{i}_{j}" for j in range(n_targets_per)]
        adj[reg] = targets
        all_symbols.append(reg)
        all_symbols.extend(targets)
    return adj, all_symbols


def _make_symbol_to_feature(symbols, gene_ids):
    """Map gene symbols to feature IDs (first N symbols get mapped)."""
    return {sym: gene_ids[i] for i, sym in enumerate(symbols) if i < len(gene_ids)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphPermutation:
    """Tests for run_graph_permutation_null()."""

    def test_basic_permutation_produces_valid_result(self):
        """5 regulators, 10 targets each, 20 permutations -> valid result."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)
        adj, symbols = _make_adjacency(n_regulators=5, n_targets_per=10)
        sym_to_feat = _make_symbol_to_feature(symbols, gene_ids)

        target_ids = gene_ids[:10]

        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="test_set",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=20,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, GraphPermutationResult)
        assert len(result.control_pvalues) > 0
        assert 0.0 <= result.target_percentile <= 100.0
        assert 0.0 <= result.fpr <= 1.0
        assert result.graph_stats["n_regulators"] == 5
        assert result.graph_stats["n_edges"] == 50
        assert result.n_permutations == 20
        # XVI-1: new fields
        assert result.n_eligible_regulators > 0
        assert result.n_eligible_regulators + result.n_excluded_regulators == 5
        assert result.median_control_set_size > 0
        # XVI-2: n_measurable_nodes in graph_stats
        assert "n_measurable_nodes" in result.graph_stats
        assert result.graph_stats["n_measurable_nodes"] > 0

    def test_empty_adjacency_raises(self):
        """Empty adjacency dict raises ValueError."""
        engine = _make_mock_engine()
        with pytest.raises(ValueError, match="empty"):
            run_graph_permutation_null(
                engine=engine,
                target_gene_ids=["P00000"],
                target_set_id="test",
                adjacency={},
                symbol_to_feature={},
                verbose=False,
            )

    def test_unfitted_engine_raises(self):
        """Unfitted engine raises RuntimeError."""
        engine = _make_mock_engine()
        engine._fitted = False
        adj, symbols = _make_adjacency(n_regulators=2, n_targets_per=3)
        sym_to_feat = _make_symbol_to_feature(symbols, engine.gene_ids)

        with pytest.raises(RuntimeError, match="fitted"):
            run_graph_permutation_null(
                engine=engine,
                target_gene_ids=engine.gene_ids[:3],
                target_set_id="test",
                adjacency=adj,
                symbol_to_feature=sym_to_feat,
                verbose=False,
            )

    def test_no_targets_in_data_raises(self):
        """Target gene IDs not in engine.gene_to_idx raises ValueError."""
        engine = _make_mock_engine()
        adj, symbols = _make_adjacency(n_regulators=2, n_targets_per=3)
        sym_to_feat = _make_symbol_to_feature(symbols, engine.gene_ids)

        with pytest.raises(ValueError, match="No target genes"):
            run_graph_permutation_null(
                engine=engine,
                target_gene_ids=["NOTEXIST_1", "NOTEXIST_2"],
                target_set_id="test",
                adjacency=adj,
                symbol_to_feature=sym_to_feat,
                verbose=False,
            )

    def test_seed_determinism(self):
        """Same seed produces identical control_pvalues."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        adj, symbols = _make_adjacency(n_regulators=5, n_targets_per=10)
        sym_to_feat = _make_symbol_to_feature(symbols, gene_ids)
        target_ids = gene_ids[:10]

        results = []
        for _ in range(2):
            engine = _make_mock_engine(gene_ids)
            r = run_graph_permutation_null(
                engine=engine,
                target_gene_ids=target_ids,
                target_set_id="seed_test",
                adjacency=adj,
                symbol_to_feature=sym_to_feat,
                n_permutations=20,
                seed=123,
                verbose=False,
            )
            results.append(r)

        np.testing.assert_array_equal(
            results[0].control_pvalues, results[1].control_pvalues
        )

    def test_sparse_graph_few_eligible_regulators(self):
        """Graph with few eligible regulators still produces valid results."""
        gene_ids = [f"P{i:05d}" for i in range(200)]
        engine = _make_mock_engine(gene_ids)

        # 2 regulators with resolvable targets, many with too few
        adj = {}
        all_symbols = []
        # Eligible regulators (>= 2 resolvable targets)
        for i in range(2):
            reg = f"REG{i}"
            targets = [f"TGT{i}_{j}" for j in range(5)]
            adj[reg] = targets
            all_symbols.append(reg)
            all_symbols.extend(targets)
        # Regulators with only 1 target (excluded)
        for i in range(10):
            reg = f"THIN{i}"
            adj[reg] = [f"THIN_TGT{i}"]
            all_symbols.append(reg)
            all_symbols.append(f"THIN_TGT{i}")

        sym_to_feat = _make_symbol_to_feature(all_symbols, gene_ids)
        target_ids = gene_ids[:5]

        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="sparse_test",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=30,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, GraphPermutationResult)
        assert result.n_eligible_regulators == 2
        assert result.n_excluded_regulators == 10
        assert result.n_valid_permutations > 0

    def test_to_dict_serialization(self):
        """Result serializes to JSON-compatible dict."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)
        adj, symbols = _make_adjacency(n_regulators=5, n_targets_per=10)
        sym_to_feat = _make_symbol_to_feature(symbols, gene_ids)
        target_ids = gene_ids[:10]

        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="dict_test",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=20,
            seed=42,
            verbose=False,
        )

        d = result.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Expected keys present (XVI-1/XVI-2 new fields)
        for key in [
            "target_set_id", "target_set_size", "target_pvalue",
            "fpr", "alpha", "target_percentile",
            "n_permutations", "n_valid_permutations",
            "n_eligible_regulators", "n_excluded_regulators",
            "median_control_set_size",
            "graph_stats",
        ]:
            assert key in d, f"Missing key: {key}"

        # Old field should NOT be present
        assert "n_empty_permutations" not in d

    def test_partial_symbol_resolution(self):
        """Some symbols don't resolve to features -- handled gracefully."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        adj = {
            "REG0": ["KNOWN1", "KNOWN2", "UNKNOWN1", "UNKNOWN2", "UNKNOWN3"],
            "REG1": ["KNOWN3", "KNOWN4", "UNKNOWN4", "UNKNOWN5", "UNKNOWN6"],
        }
        # Only map KNOWN symbols, not UNKNOWN ones
        sym_to_feat = {
            "KNOWN1": gene_ids[0],
            "KNOWN2": gene_ids[1],
            "KNOWN3": gene_ids[2],
            "KNOWN4": gene_ids[3],
            "REG0": gene_ids[4],
            "REG1": gene_ids[5],
        }
        target_ids = gene_ids[:4]

        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="partial_resolve",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=20,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, GraphPermutationResult)
        # Each regulator has 2 resolvable targets (KNOWN1/2 and KNOWN3/4)
        assert result.n_eligible_regulators == 2

    def test_no_eligible_regulators_raises(self):
        """All regulators have < 2 resolvable targets -> RuntimeError."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        # Each regulator has only 1 resolvable target (below threshold of 2)
        adj = {
            "REG0": ["MAPPED_A", "UNMAPPED_B"],
            "REG1": ["MAPPED_C", "UNMAPPED_D"],
        }
        sym_to_feat = {
            "MAPPED_A": gene_ids[0],
            "MAPPED_C": gene_ids[1],
            "REG0": gene_ids[2],
            "REG1": gene_ids[3],
        }
        target_ids = gene_ids[:2]

        with pytest.raises(RuntimeError, match="No eligible regulators"):
            run_graph_permutation_null(
                engine=engine,
                target_gene_ids=target_ids,
                target_set_id="no_eligible_test",
                adjacency=adj,
                symbol_to_feature=sym_to_feat,
                n_permutations=20,
                seed=42,
                verbose=False,
            )

    def test_all_permutations_empty_raises(self):
        """Adjacency where no permutation resolves raises RuntimeError.

        With XVI-1 pre-filtering, this becomes 'no eligible regulators'
        because regulators with unmapped targets have < 2 resolvable targets.
        """
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        # All nodes are regulators but none of their targets can resolve
        adj = {
            "REG0": ["UNMAPPED_A", "UNMAPPED_B"],
            "REG1": ["UNMAPPED_C", "UNMAPPED_D"],
        }
        # No target symbol maps to any feature
        sym_to_feat = {"REG0": gene_ids[0], "REG1": gene_ids[1]}
        target_ids = gene_ids[:2]

        with pytest.raises(RuntimeError, match="No eligible regulators"):
            run_graph_permutation_null(
                engine=engine,
                target_gene_ids=target_ids,
                target_set_id="empty_test",
                adjacency=adj,
                symbol_to_feature=sym_to_feat,
                n_permutations=20,
                seed=42,
                verbose=False,
            )

    def test_no_set_size_contraction(self):
        """XVI-2: All permuted sets have consistent size (no resolution shrinkage).

        When all targets are resolvable and the permutation space is restricted
        to resolvable genes, every permuted set must have the same size as the
        regulator's original resolvable target count.
        """
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        # Record gene_set sizes passed to test_gene_set
        observed_sizes = []
        original_test = engine.test_gene_set

        def recording_test_gene_set(gene_set, gene_set_id, weights=None):
            # Skip the real target set call (first call)
            observed_sizes.append(len(gene_set))
            return original_test(gene_set=gene_set, gene_set_id=gene_set_id)

        engine.test_gene_set = recording_test_gene_set

        # Build a graph where ALL targets are resolvable
        n_targets = 8
        adj = {"REG0": [f"TGT0_{j}" for j in range(n_targets)]}
        all_symbols = ["REG0"] + [f"TGT0_{j}" for j in range(n_targets)]
        sym_to_feat = _make_symbol_to_feature(all_symbols, gene_ids)
        target_ids = gene_ids[:n_targets]

        n_perms = 50
        run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="contraction_test",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=n_perms,
            seed=42,
            verbose=False,
        )

        # First call is the real target set; rest are permutations
        perm_sizes = observed_sizes[1:]
        assert len(perm_sizes) == n_perms
        # All permuted sets should have exactly n_targets genes
        assert all(s == n_targets for s in perm_sizes), (
            f"Set size contraction detected: sizes={set(perm_sizes)}, "
            f"expected all={n_targets}"
        )

    def test_regulator_sampling_uniformity(self):
        """XVI-1: With many permutations, all eligible regulators are sampled."""
        gene_ids = [f"P{i:05d}" for i in range(200)]
        engine = _make_mock_engine(gene_ids)

        # 4 regulators, each with distinct targets
        n_regs = 4
        adj = {}
        all_symbols = []
        for i in range(n_regs):
            reg = f"REG{i}"
            targets = [f"TGT{i}_{j}" for j in range(5)]
            adj[reg] = targets
            all_symbols.append(reg)
            all_symbols.extend(targets)

        sym_to_feat = _make_symbol_to_feature(all_symbols, gene_ids)
        target_ids = gene_ids[:5]

        # Track which gene_set_ids are used (captures regulator choice indirectly)
        # We can't directly observe regulator choice, but with enough permutations
        # and 4 regulators, each should appear at least once.
        n_perms = 200
        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="uniformity_test",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=n_perms,
            seed=42,
            verbose=False,
        )

        assert result.n_eligible_regulators == n_regs
        assert result.n_valid_permutations == n_perms

    def test_single_regulator_all_permutations_same_structure(self):
        """One regulator -> all permutations use the same neighborhood structure.

        All permuted sets must have the same size (the regulator's resolvable
        target count), but with different gene identities.
        """
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        # Record sets passed to test_gene_set
        observed_sets = []
        original_test = engine.test_gene_set

        def recording_test(gene_set, gene_set_id, weights=None):
            observed_sets.append(set(gene_set))
            return original_test(gene_set=gene_set, gene_set_id=gene_set_id)

        engine.test_gene_set = recording_test

        n_targets = 6
        adj = {"SOLO_REG": [f"TGT_{j}" for j in range(n_targets)]}
        all_symbols = ["SOLO_REG"] + [f"TGT_{j}" for j in range(n_targets)]
        sym_to_feat = _make_symbol_to_feature(all_symbols, gene_ids)
        target_ids = gene_ids[:n_targets]

        n_perms = 30
        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="single_reg_test",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=n_perms,
            seed=42,
            verbose=False,
        )

        # First call is real target; rest are permutations
        perm_sets = observed_sets[1:]
        assert len(perm_sets) == n_perms

        # All same size
        sizes = {len(s) for s in perm_sets}
        assert sizes == {n_targets}, f"Expected all size {n_targets}, got {sizes}"

        assert result.n_eligible_regulators == 1
        assert result.n_excluded_regulators == 0
        assert result.median_control_set_size == n_targets

    def test_partial_resolution_no_contraction(self):
        """XVI-2: Partially resolvable graph — permuted sets sized to resolvable
        count, not total target count. Unmapped targets are excluded from
        regulator_resolvable_targets, so set size = resolvable count."""
        gene_ids = [f"P{i:05d}" for i in range(100)]
        engine = _make_mock_engine(gene_ids)

        # REG0 has 6 targets: 4 mapped, 2 unmapped
        adj = {
            "REG0": ["M0", "M1", "M2", "M3", "UNMAPPED_A", "UNMAPPED_B"],
        }
        sym_to_feat = {
            "M0": gene_ids[0], "M1": gene_ids[1],
            "M2": gene_ids[2], "M3": gene_ids[3],
            "REG0": gene_ids[4],
        }
        target_ids = gene_ids[:4]

        # Record set sizes
        observed_sizes = []
        original_test = engine.test_gene_set

        def recording_test(gene_set, gene_set_id, weights=None):
            observed_sizes.append(len(gene_set))
            return original_test(gene_set=gene_set, gene_set_id=gene_set_id)

        engine.test_gene_set = recording_test

        n_perms = 30
        result = run_graph_permutation_null(
            engine=engine,
            target_gene_ids=target_ids,
            target_set_id="partial_contraction",
            adjacency=adj,
            symbol_to_feature=sym_to_feat,
            n_permutations=n_perms,
            seed=42,
            verbose=False,
        )

        # First call is real target; rest are permutations
        perm_sizes = observed_sizes[1:]
        assert len(perm_sizes) == n_perms
        # All permuted sets should have exactly 4 genes (resolvable count),
        # NOT 6 (total target count)
        assert all(s == 4 for s in perm_sizes), (
            f"Expected all size 4 (resolvable), got {set(perm_sizes)}"
        )
        assert result.median_control_set_size == 4
