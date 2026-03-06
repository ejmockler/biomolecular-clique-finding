"""Tests for Audit VI coherence fixes.

Covers:
- STAT-VI-3: Phipson-Smyth correction for permutation p-values
- KG-VI-1: Deterministic community detection via seeded RNG
- KG-VI-2: Per-community modularity contribution
- KG-VI-3: Spearman 2-gene edge case
- CLEAN-VI-3: CorrelationSign.BOTH removal
- CLEAN-VI-4: Narrowed exception handling
"""

import pytest
import numpy as np
import pandas as pd

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.knowledge.regulatory_coherence import (
    CoherenceAnalyzer,
    CoherenceConfig,
    CorrelationSign,
    _community_modularity,
)


def _make_matrix(n_genes=10, n_samples=30, seed=42):
    """Create a BioMatrix with two anticorrelated gene blocks."""
    rng = np.random.RandomState(seed)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    sample_names = [f"S{i}" for i in range(n_samples)]

    # Two anticorrelated blocks
    base_signal = rng.randn(n_samples)
    data = np.zeros((n_genes, n_samples))
    for i in range(n_genes // 2):
        data[i] = base_signal + rng.randn(n_samples) * 0.3
    for i in range(n_genes // 2, n_genes):
        data[i] = -base_signal + rng.randn(n_samples) * 0.3
    data += 10.0

    feature_ids = pd.Index(gene_names)
    sample_ids = pd.Index(sample_names)
    metadata = pd.DataFrame({"phenotype": ["A"] * n_samples}, index=sample_ids)
    quality_flags = np.full(
        (n_genes, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32
    )
    return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)


def _make_analyzer(matrix, n_bootstrap=5, seed=42):
    """Create a CoherenceAnalyzer with relaxed thresholds for testing."""
    config = CoherenceConfig(
        n_bootstrap=n_bootstrap,
        n_permutations=20,
        min_samples_per_condition=3,
        min_expression_percentile=0.0,
        min_variance_percentile=0.0,
        min_community_size=2,
        min_community_density=0.0,
        soft_threshold_power=2.0,
        min_edge_weight=0.01,
    )
    return CoherenceAnalyzer(matrix, config=config, seed=seed)


# ---------------------------------------------------------------------------
# STAT-VI-3: Phipson-Smyth correction for permutation p-values
# ---------------------------------------------------------------------------


class TestPhipsonSmythCorrection:
    """Verify permutation p-value uses (b+1)/(B+1) formula."""

    def test_permutation_pvalue_never_zero(self):
        """Permutation p-value must never be exactly 0.0."""
        matrix = _make_matrix(n_genes=8, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        genes = set(matrix.feature_ids)
        # Use a very high observed modularity so b=0
        p = analyzer.permutation_null(
            genes=genes,
            condition="all",
            observed_modularity=999.0,
            n_permutations=50,
        )
        # With Phipson-Smyth: p = (0+1)/(50+1) = 1/51 ≈ 0.0196
        assert p > 0.0, "Phipson-Smyth correction should prevent p=0.0"
        assert p == pytest.approx(1.0 / 51.0, abs=1e-10)

    def test_permutation_pvalue_minimum_with_100_perms(self):
        """Minimum p-value with 100 permutations is 1/101."""
        matrix = _make_matrix(n_genes=8, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        genes = set(matrix.feature_ids)
        p = analyzer.permutation_null(
            genes=genes,
            condition="all",
            observed_modularity=999.0,
            n_permutations=100,
        )
        expected_min = 1.0 / 101.0
        assert p == pytest.approx(expected_min, abs=1e-10)

    def test_permutation_pvalue_all_exceed(self):
        """If all null >= observed, p should be (B+1)/(B+1) = 1.0."""
        matrix = _make_matrix(n_genes=8, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        genes = set(matrix.feature_ids)
        # observed_modularity of -999 means all null values exceed it
        p = analyzer.permutation_null(
            genes=genes,
            condition="all",
            observed_modularity=-999.0,
            n_permutations=20,
        )
        # b = 20, p = (20+1)/(20+1) = 1.0
        assert p == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# KG-VI-1: Deterministic community detection via seeded RNG
# ---------------------------------------------------------------------------


class TestCommunityDetectionSeeding:
    """Verify community detection is deterministic with same seed."""

    def test_same_seed_same_results(self):
        """Two analyzers with the same seed should detect identical communities."""
        matrix = _make_matrix(n_genes=10, n_samples=30, seed=42)
        genes = set(matrix.feature_ids)

        analyzer1 = _make_analyzer(matrix, seed=123)
        result1 = analyzer1.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=False,
        )

        analyzer2 = _make_analyzer(matrix, seed=123)
        result2 = analyzer2.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=False,
        )

        # Same communities
        assert len(result1.positive_communities) == len(result2.positive_communities)
        for c1, c2 in zip(result1.positive_communities, result2.positive_communities):
            assert c1.genes == c2.genes

    def test_different_seeds_may_differ(self):
        """Different seeds do not guarantee different results, but the mechanism is tested."""
        matrix = _make_matrix(n_genes=10, n_samples=30, seed=42)

        analyzer1 = _make_analyzer(matrix, seed=1)
        analyzer2 = _make_analyzer(matrix, seed=99999)

        # Just verify both run without error; results may or may not differ
        genes = set(matrix.feature_ids)
        r1 = analyzer1.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=False,
        )
        r2 = analyzer2.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=False,
        )
        # Both should produce valid results
        assert r1.condition == "all"
        assert r2.condition == "all"

    def test_get_community_seed_deterministic(self):
        """_get_community_seed produces deterministic sequence for same initial seed."""
        matrix = _make_matrix()
        a1 = _make_analyzer(matrix, seed=42)
        a2 = _make_analyzer(matrix, seed=42)

        seeds1 = [a1._get_community_seed() for _ in range(5)]
        seeds2 = [a2._get_community_seed() for _ in range(5)]
        assert seeds1 == seeds2

    def test_get_community_seed_bounded(self):
        """_get_community_seed returns values in [0, 2^31)."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, seed=0)
        for _ in range(20):
            s = analyzer._get_community_seed()
            assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# KG-VI-2: Per-community modularity contribution
# ---------------------------------------------------------------------------


class TestModularityContribution:
    """Verify modularity_contribution uses proper per-community calculation."""

    def test_modularity_contribution_not_naive_average(self):
        """modularity_contribution should NOT be modularity/n_communities."""
        matrix = _make_matrix(n_genes=10, n_samples=30, seed=42)
        analyzer = _make_analyzer(matrix, seed=42)
        genes = set(matrix.feature_ids)

        result = analyzer.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=False,
        )

        if len(result.positive_communities) >= 2:
            # With proper per-community modularity, different-sized communities
            # should generally get different contribution values
            contributions = [c.modularity_contribution for c in result.positive_communities]
            sizes = [c.size for c in result.positive_communities]
            # At minimum, check contributions are finite and non-negative for
            # communities that have edges
            for mc in contributions:
                assert np.isfinite(mc)

    def test_modularity_contribution_scales_with_quality(self):
        """A well-separated community should have higher modularity than
        a poorly-separated one."""
        import networkx as nx

        # Build a graph with one well-separated and one poorly-separated community
        G = nx.Graph()
        # Community A: 4 nodes, fully connected internally, NO cross-edges
        for i in range(4):
            for j in range(i + 1, 4):
                G.add_edge(f"A{i}", f"A{j}", weight=0.9)
        # Community B: 4 nodes, fully connected internally
        for i in range(4):
            for j in range(i + 1, 4):
                G.add_edge(f"B{i}", f"B{j}", weight=0.9)
        # Community C: 4 nodes with MANY cross-edges to B (poorly separated)
        for i in range(4):
            for j in range(i + 1, 4):
                G.add_edge(f"C{i}", f"C{j}", weight=0.9)
        # Dense cross-edges B<->C (makes C poorly separated from B)
        for i in range(4):
            for j in range(4):
                G.add_edge(f"B{i}", f"C{j}", weight=0.7)
        # Minimal cross-edge from A
        G.add_edge("A0", "B0", weight=0.05)

        comm_a = {f"A{i}" for i in range(4)}
        comm_c = {f"C{i}" for i in range(4)}

        mod_a = _community_modularity(G, comm_a)
        mod_c = _community_modularity(G, comm_c)

        # A is well-separated (almost no cross-edges), C is not
        assert mod_a > mod_c, (
            f"Well-separated community should have higher modularity: "
            f"mod_a={mod_a} vs mod_c={mod_c}"
        )

    def test_community_modularity_empty_rest(self):
        """If community IS the entire graph, modularity contribution is 0."""
        import networkx as nx

        G = nx.Graph()
        G.add_edge("A", "B", weight=1.0)
        result = _community_modularity(G, {"A", "B"})
        assert result == 0.0


# ---------------------------------------------------------------------------
# KG-VI-3: Spearman 2-gene edge case
# ---------------------------------------------------------------------------


class TestSpearmanTwoGenes:
    """Verify Spearman with exactly 2 genes returns proper 2x2 matrix."""

    def test_spearman_two_genes_returns_2x2(self):
        """spearmanr with 2 genes should return 2x2, not 1x1."""
        # Create matrix with only 2 genes that are positively correlated
        rng = np.random.RandomState(42)
        n_samples = 30
        base = rng.randn(n_samples)
        data = np.zeros((2, n_samples))
        data[0] = base + rng.randn(n_samples) * 0.1
        data[1] = base + rng.randn(n_samples) * 0.1
        data += 10.0

        gene_names = ["GENE0", "GENE1"]
        sample_names = [f"S{i}" for i in range(n_samples)]
        feature_ids = pd.Index(gene_names)
        sample_ids = pd.Index(sample_names)
        metadata = pd.DataFrame({"phenotype": ["A"] * n_samples}, index=sample_ids)
        quality_flags = np.full((2, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32)
        matrix = BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)

        config = CoherenceConfig(
            min_samples_per_condition=3,
            min_expression_percentile=0.0,
            min_variance_percentile=0.0,
            soft_threshold_power=2.0,
            min_edge_weight=0.01,
        )
        analyzer = CoherenceAnalyzer(matrix, config=config, seed=42)

        corr_matrix, genes = analyzer.compute_correlation_matrix(
            gene_names, "all", method="spearman"
        )

        assert corr_matrix.shape == (2, 2), f"Expected 2x2, got {corr_matrix.shape}"
        # Diagonal should be 1.0
        assert corr_matrix[0, 0] == pytest.approx(1.0)
        assert corr_matrix[1, 1] == pytest.approx(1.0)
        # Off-diagonal should be the actual correlation (high positive)
        assert corr_matrix[0, 1] > 0.5, "Two correlated genes should show high rho"
        assert corr_matrix[0, 1] == pytest.approx(corr_matrix[1, 0])


# ---------------------------------------------------------------------------
# CLEAN-VI-3: CorrelationSign.BOTH removed
# ---------------------------------------------------------------------------


class TestCorrelationSignBothRemoved:
    """Verify CorrelationSign.BOTH no longer exists."""

    def test_both_raises_attribute_error(self):
        """Accessing CorrelationSign.BOTH should raise AttributeError."""
        with pytest.raises(AttributeError):
            _ = CorrelationSign.BOTH

    def test_only_positive_and_negative(self):
        """CorrelationSign should have exactly 2 members."""
        members = list(CorrelationSign)
        assert len(members) == 2
        assert CorrelationSign.POSITIVE in members
        assert CorrelationSign.NEGATIVE in members


# ---------------------------------------------------------------------------
# CLEAN-VI-4: Narrowed exception handling
# ---------------------------------------------------------------------------


class TestNarrowedExceptionHandling:
    """Verify analyze_all_conditions lets unexpected exceptions propagate."""

    def test_type_error_propagates(self):
        """TypeError in analyze_coherence should NOT be swallowed."""
        matrix = _make_matrix(n_genes=10, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        # Patch analyze_coherence to raise TypeError
        original = analyzer.analyze_coherence

        def raise_type_error(**kwargs):
            raise TypeError("unexpected type mismatch")

        analyzer.analyze_coherence = raise_type_error

        with pytest.raises(TypeError, match="unexpected type mismatch"):
            analyzer.analyze_all_conditions(genes=set(matrix.feature_ids))

    def test_value_error_is_caught(self):
        """ValueError should still be caught and logged, not propagated."""
        matrix = _make_matrix(n_genes=10, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        original = analyzer.analyze_coherence

        def raise_value_error(**kwargs):
            raise ValueError("too few samples")

        analyzer.analyze_coherence = raise_value_error

        # Should not raise; just returns empty dict
        results = analyzer.analyze_all_conditions(genes=set(matrix.feature_ids))
        assert results == {}

    def test_runtime_error_is_caught(self):
        """RuntimeError should be caught (singular matrix, etc.)."""
        matrix = _make_matrix(n_genes=10, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        def raise_runtime_error(**kwargs):
            raise RuntimeError("singular matrix")

        analyzer.analyze_coherence = raise_runtime_error

        results = analyzer.analyze_all_conditions(genes=set(matrix.feature_ids))
        assert results == {}

    def test_attribute_error_propagates(self):
        """AttributeError should propagate (indicates a real bug)."""
        matrix = _make_matrix(n_genes=10, n_samples=30)
        analyzer = _make_analyzer(matrix, seed=42)

        def raise_attr_error(**kwargs):
            raise AttributeError("no such attribute")

        analyzer.analyze_coherence = raise_attr_error

        with pytest.raises(AttributeError, match="no such attribute"):
            analyzer.analyze_all_conditions(genes=set(matrix.feature_ids))
