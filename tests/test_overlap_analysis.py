"""Tests for gene set overlap quantification and effective independent tests."""

import numpy as np
import pytest

from cliquefinder.stats.overlap_analysis import (
    annotate_discovery_with_overlap,
    compute_jaccard_matrix,
    effective_independent_tests,
    overlap_summary,
)


class TestComputeJaccardMatrix:
    """Tests for compute_jaccard_matrix()."""

    def test_identical_sets_yield_ones(self):
        """Identical sets should have Jaccard = 1.0 everywhere."""
        gene_sets = {
            "A": {"g1", "g2", "g3"},
            "B": {"g1", "g2", "g3"},
            "C": {"g1", "g2", "g3"},
        }
        mat = compute_jaccard_matrix(gene_sets)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(mat, np.ones((3, 3)))

    def test_disjoint_sets_yield_zeros(self):
        """Completely disjoint sets should have Jaccard = 0.0 off-diagonal."""
        gene_sets = {
            "A": {"g1", "g2"},
            "B": {"g3", "g4"},
            "C": {"g5", "g6"},
        }
        mat = compute_jaccard_matrix(gene_sets)
        expected = np.eye(3)
        np.testing.assert_allclose(mat, expected)

    def test_known_overlap(self):
        """50% overlap: |A&B|=1, |A|B|=3 => Jaccard=1/3."""
        gene_sets = {
            "A": {"g1", "g2"},
            "B": {"g2", "g3"},
        }
        mat = compute_jaccard_matrix(gene_sets)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 1.0
        assert mat[1, 1] == 1.0
        np.testing.assert_almost_equal(mat[0, 1], 1.0 / 3.0)
        np.testing.assert_almost_equal(mat[1, 0], 1.0 / 3.0)

    def test_symmetric(self):
        """Jaccard matrix must be symmetric."""
        gene_sets = {
            "A": {"g1", "g2", "g3"},
            "B": {"g2", "g3", "g4"},
            "C": {"g3", "g4", "g5"},
        }
        mat = compute_jaccard_matrix(gene_sets)
        np.testing.assert_allclose(mat, mat.T)

    def test_empty_input(self):
        """Empty dict returns empty matrix."""
        mat = compute_jaccard_matrix({})
        assert mat.shape == (0, 0)

    def test_single_set(self):
        """Single set yields 1x1 matrix with value 1.0."""
        mat = compute_jaccard_matrix({"A": {"g1", "g2"}})
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 1.0

    def test_empty_sets_yield_zero(self):
        """Empty sets should produce 0.0 Jaccard with everything."""
        gene_sets = {
            "A": set(),
            "B": {"g1", "g2"},
        }
        mat = compute_jaccard_matrix(gene_sets)
        assert mat[0, 1] == 0.0
        assert mat[1, 0] == 0.0


class TestEffectiveIndependentTests:
    """Tests for effective_independent_tests()."""

    def test_independent_sets_m_eff_near_nominal(self):
        """Fully disjoint sets should have m_eff close to m_nominal."""
        # Create 10 disjoint gene sets of size 5
        gene_sets = {}
        for i in range(10):
            gene_sets[f"set_{i}"] = {f"gene_{i}_{j}" for j in range(5)}

        result = effective_independent_tests(gene_sets)
        assert result["m_nominal"] == 10
        # For perfectly independent sets, m_eff should be close to 10
        assert result["m_eff"] >= 9.0
        assert result["ratio"] >= 0.9

    def test_identical_sets_m_eff_near_one(self):
        """Identical sets should have m_eff close to 1."""
        shared = {"g1", "g2", "g3", "g4", "g5"}
        gene_sets = {f"set_{i}": shared.copy() for i in range(10)}

        result = effective_independent_tests(gene_sets)
        assert result["m_nominal"] == 10
        # Perfectly correlated sets => m_eff should be close to 1
        assert result["m_eff"] <= 2.0
        assert result["ratio"] <= 0.2

    def test_partial_overlap_intermediate(self):
        """Partially overlapping sets: 1 < m_eff < m_nominal."""
        # 10 sets sharing a large core (8 genes) + 2 unique genes each.
        # This produces strong correlation (Jaccard ~0.67) and m_eff << 10.
        shared_core = {f"core_{j}" for j in range(8)}
        gene_sets = {}
        for i in range(10):
            genes = shared_core.copy()
            genes |= {f"unique_{i}_{j}" for j in range(2)}
            gene_sets[f"set_{i}"] = genes

        result = effective_independent_tests(gene_sets)
        assert result["m_nominal"] == 10
        assert 1.0 < result["m_eff"] < 10.0
        assert 0.0 < result["ratio"] < 1.0

    def test_empty_input(self):
        """Empty input returns zeros."""
        result = effective_independent_tests({})
        assert result["m_nominal"] == 0
        assert result["m_eff"] == 0.0
        assert result["ratio"] == 0.0

    def test_single_set(self):
        """Single set yields m_eff = 1."""
        result = effective_independent_tests({"A": {"g1", "g2", "g3"}})
        assert result["m_nominal"] == 1
        assert result["m_eff"] == 1.0
        assert result["ratio"] == 1.0

    def test_return_keys(self):
        """All expected keys present in returned dict."""
        gene_sets = {"A": {"g1"}, "B": {"g2"}}
        result = effective_independent_tests(gene_sets)
        expected_keys = {
            "m_nominal", "m_eff", "ratio", "median_jaccard",
            "max_jaccard", "pct_pairs_above_50", "eigenvalue_summary",
        }
        assert set(result.keys()) == expected_keys

    def test_m_eff_bounded(self):
        """m_eff is always between 1 and m_nominal."""
        rng = np.random.default_rng(42)
        # Random sets with varying overlap
        all_genes = [f"g{i}" for i in range(50)]
        gene_sets = {}
        for i in range(20):
            size = rng.integers(5, 20)
            genes = set(rng.choice(all_genes, size=size, replace=False))
            gene_sets[f"set_{i}"] = genes

        result = effective_independent_tests(gene_sets)
        assert 1.0 <= result["m_eff"] <= result["m_nominal"]

    def test_jaccard_statistics_consistent(self):
        """Jaccard stats in result match direct computation."""
        gene_sets = {
            "A": {"g1", "g2", "g3"},
            "B": {"g2", "g3", "g4"},
            "C": {"g5", "g6", "g7"},
        }
        result = effective_independent_tests(gene_sets)
        jac_mat = compute_jaccard_matrix(gene_sets)
        triu = jac_mat[np.triu_indices(3, k=1)]

        np.testing.assert_almost_equal(
            result["median_jaccard"], float(np.median(triu)), decimal=4
        )
        np.testing.assert_almost_equal(
            result["max_jaccard"], float(np.max(triu)), decimal=4
        )


class TestOverlapSummary:
    """Tests for overlap_summary()."""

    def test_basic_output(self):
        """Returns expected keys with valid values."""
        gene_sets = {
            "A": {"g1", "g2", "g3"},
            "B": {"g2", "g3", "g4"},
        }
        result = overlap_summary(gene_sets)
        assert "m_nominal" in result
        assert "median_jaccard" in result
        assert "mean_set_size" in result
        assert result["m_nominal"] == 2
        assert result["mean_set_size"] == 3.0

    def test_empty_input(self):
        """Empty input returns zeros."""
        result = overlap_summary({})
        assert result["m_nominal"] == 0
        assert result["mean_set_size"] == 0.0

    def test_set_size_stats(self):
        """Set size statistics are correct."""
        gene_sets = {
            "A": {"g1", "g2"},
            "B": {"g3", "g4", "g5", "g6"},
        }
        result = overlap_summary(gene_sets)
        assert result["min_set_size"] == 2
        assert result["max_set_size"] == 4
        assert result["mean_set_size"] == 3.0


class TestAnnotateDiscoveryWithOverlap:
    """Tests for annotate_discovery_with_overlap()."""

    def test_annotates_with_gene_sets_per_hop(self):
        """Adds overlap stats when gene_sets_per_hop is provided."""
        discovery = {
            "hops": [
                {
                    "hop": 1,
                    "all_arms": [{"intermediary": "A"}],
                },
                {
                    "hop": 2,
                    "all_arms": [
                        {"intermediary": "B"},
                        {"intermediary": "C"},
                        {"intermediary": "D"},
                    ],
                },
            ]
        }
        gene_sets_per_hop = {
            1: {"A": {"g1", "g2", "g3"}},
            2: {
                "B": {"g1", "g2", "g3"},
                "C": {"g2", "g3", "g4"},
                "D": {"g5", "g6", "g7"},
            },
        }

        result = annotate_discovery_with_overlap(
            discovery, gene_sets_per_hop=gene_sets_per_hop
        )

        # Hop 1 has only 1 set
        assert result["hops"][0]["overlap"]["m_nominal"] == 1
        assert result["hops"][0]["overlap"]["m_eff"] == 1.0

        # Hop 2 has 3 sets with some overlap
        hop2 = result["hops"][1]["overlap"]
        assert hop2["m_nominal"] == 3
        assert 1.0 <= hop2["m_eff"] <= 3.0
        assert "adjusted_fdr_threshold" in hop2

    def test_annotates_from_arm_targets(self):
        """Uses arm targets when gene_sets_per_hop not provided."""
        discovery = {
            "hops": [
                {
                    "hop": 1,
                    "all_arms": [
                        {"intermediary": "A", "targets": ["g1", "g2"]},
                        {"intermediary": "B", "targets": ["g2", "g3"]},
                    ],
                },
            ]
        }
        result = annotate_discovery_with_overlap(discovery)
        assert "overlap" in result["hops"][0]
        assert result["hops"][0]["overlap"]["m_nominal"] == 2

    def test_error_when_no_targets_available(self):
        """Reports error when targets cannot be reconstructed."""
        discovery = {
            "hops": [
                {
                    "hop": 1,
                    "all_arms": [
                        {"intermediary": "A"},
                        {"intermediary": "B"},
                    ],
                },
            ]
        }
        result = annotate_discovery_with_overlap(discovery)
        assert "error" in result["hops"][0]["overlap"]

    def test_adjusted_fdr_threshold(self):
        """Adjusted FDR threshold scales with overlap."""
        # Identical sets => high adjustment
        shared = {"g1", "g2", "g3", "g4", "g5"}
        gene_sets_per_hop = {
            1: {f"set_{i}": shared.copy() for i in range(10)}
        }
        discovery = {
            "hops": [{"hop": 1, "all_arms": [{"intermediary": f"set_{i}"} for i in range(10)]}]
        }
        result = annotate_discovery_with_overlap(
            discovery, gene_sets_per_hop=gene_sets_per_hop, fdr_threshold=0.05
        )
        # adjusted_fdr should be much larger than 0.05 since m_eff << m_nominal
        assert result["hops"][0]["overlap"]["adjusted_fdr_threshold"] > 0.05

    def test_does_not_mutate_input(self):
        """Original discovery_result dict is not mutated."""
        discovery = {
            "hops": [
                {
                    "hop": 1,
                    "all_arms": [
                        {"intermediary": "A", "targets": ["g1"]},
                    ],
                },
            ]
        }
        import copy
        original = copy.deepcopy(discovery)
        annotate_discovery_with_overlap(discovery)
        # The function returns a copy, original should be unchanged
        assert "overlap" not in discovery["hops"][0]
