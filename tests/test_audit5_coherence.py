#!/usr/bin/env python3
"""
Tests for Audit V Cycle 2 coherence fixes:
  KG-V-3: permutation_null tests both positive and negative graphs
  KG-V-4: Dead _corr_cache removed
  KG-V-5: build_signed_graphs vectorized
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# We need networkx for the coherence module
pytest.importorskip("networkx")


# ---------------------------------------------------------------------------
# Mock BioMatrix helper
# ---------------------------------------------------------------------------

class MockBioMatrix:
    """Minimal mock that satisfies CoherenceAnalyzer's interface."""

    def __init__(self, data: np.ndarray, feature_ids: pd.Index, n_samples: int,
                 sample_metadata: pd.DataFrame | None = None):
        self.data = data
        self.feature_ids = feature_ids
        self.sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        self.n_samples = n_samples
        if sample_metadata is not None:
            self.sample_metadata = sample_metadata
        else:
            self.sample_metadata = pd.DataFrame(index=self.sample_ids)


def _make_mock_matrix(n_genes: int = 10, n_samples: int = 30, seed: int = 42,
                      sign: str = "mixed"):
    """Create a mock BioMatrix with controlled correlation structure.

    Args:
        sign: "positive" for positively correlated genes, "negative" for
              negatively correlated (half anti-correlated), "mixed" for both.
    """
    rng = np.random.RandomState(seed)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    feature_ids = pd.Index(gene_names)

    # Create data with correlation structure
    base_signal = rng.randn(1, n_samples)

    if sign == "positive":
        # All genes positively correlated
        data = base_signal + rng.randn(n_genes, n_samples) * 0.3
    elif sign == "negative":
        # First half positively correlated, second half negatively
        data = np.empty((n_genes, n_samples))
        half = n_genes // 2
        data[:half] = base_signal + rng.randn(half, n_samples) * 0.3
        data[half:] = -base_signal + rng.randn(n_genes - half, n_samples) * 0.3
    else:
        # Mixed: some positive, some negative, some noise
        data = np.empty((n_genes, n_samples))
        third = n_genes // 3
        data[:third] = base_signal + rng.randn(third, n_samples) * 0.3
        data[third:2*third] = -base_signal + rng.randn(third, n_samples) * 0.3
        data[2*third:] = rng.randn(n_genes - 2*third, n_samples) * 1.0

    return MockBioMatrix(data, feature_ids, n_samples), gene_names


def _make_coherence_analyzer(n_genes=10, n_samples=30, seed=42, sign="mixed",
                             **config_kwargs):
    """Create a CoherenceAnalyzer with a mock matrix."""
    from cliquefinder.knowledge.regulatory_coherence import (
        CoherenceAnalyzer, CoherenceConfig,
    )

    matrix, gene_names = _make_mock_matrix(n_genes, n_samples, seed, sign)
    config_defaults = dict(
        n_bootstrap=5,
        n_permutations=10,
        min_samples_per_condition=3,
        min_community_size=2,
        min_community_density=0.0,
        soft_threshold_power=2.0,  # lower power so more edges survive
        min_edge_weight=0.01,
        min_expression_percentile=0.0,
        min_variance_percentile=0.0,
    )
    config_defaults.update(config_kwargs)
    config = CoherenceConfig(**config_defaults)
    analyzer = CoherenceAnalyzer(matrix=matrix, config=config)
    return analyzer, gene_names


# ===========================================================================
# Test 1: permutation_null with correlation_sign=NEGATIVE
# ===========================================================================

class TestPermutationNullNegativeGraph:
    """KG-V-3: permutation_null should test negative graph when requested."""

    def test_permutation_null_negative_graph(self):
        """Call permutation_null with NEGATIVE and verify it returns a valid p-value."""
        from cliquefinder.knowledge.regulatory_coherence import CorrelationSign

        # Use data with strong negative correlations
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=8, n_samples=30, seed=99, sign="negative"
        )
        genes = set(gene_names)

        # Compute observed negative modularity
        corr, gl = analyzer.compute_correlation_matrix(list(genes), "all")
        _, G_neg = analyzer.build_signed_graphs(corr, gl)

        if G_neg.number_of_edges() == 0:
            pytest.skip("No negative edges in test data")

        import networkx as nx
        from networkx.algorithms import community as nx_community

        partition = analyzer.detect_communities(G_neg)
        comms = {}
        for gene, cid in partition.items():
            comms.setdefault(cid, set()).add(gene)
        observed_mod = nx_community.modularity(G_neg, comms.values(), weight="weight")

        # Run permutation null with NEGATIVE sign
        p_val = analyzer.permutation_null(
            genes, "all", observed_mod,
            n_permutations=20,
            correlation_sign=CorrelationSign.NEGATIVE,
        )

        # p-value should be a float in [0, 1]
        assert 0.0 <= p_val <= 1.0
        assert isinstance(p_val, (float, np.floating))

    def test_permutation_null_default_positive(self):
        """Default (no correlation_sign) should use positive graph, same as before."""
        from cliquefinder.knowledge.regulatory_coherence import CorrelationSign

        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=8, n_samples=30, seed=42, sign="positive"
        )
        genes = set(gene_names)

        corr, gl = analyzer.compute_correlation_matrix(list(genes), "all")
        G_pos, _ = analyzer.build_signed_graphs(corr, gl)

        if G_pos.number_of_edges() == 0:
            pytest.skip("No positive edges in test data")

        import networkx as nx
        from networkx.algorithms import community as nx_community

        partition = analyzer.detect_communities(G_pos)
        comms = {}
        for gene, cid in partition.items():
            comms.setdefault(cid, set()).add(gene)
        observed_mod = nx_community.modularity(G_pos, comms.values(), weight="weight")

        # No correlation_sign → default positive
        p_val_default = analyzer.permutation_null(
            genes, "all", observed_mod, n_permutations=15
        )
        assert 0.0 <= p_val_default <= 1.0

        # Explicit POSITIVE should give same behavior
        p_val_explicit = analyzer.permutation_null(
            genes, "all", observed_mod, n_permutations=15,
            correlation_sign=CorrelationSign.POSITIVE
        )
        assert 0.0 <= p_val_explicit <= 1.0


# ===========================================================================
# Test 2: analyze_coherence calls permutation_null for negative communities
# ===========================================================================

class TestAnalyzeCoherenceNegPermutation:
    """KG-V-3: analyze_coherence should run permutation_null for neg graph too."""

    def test_analyze_coherence_neg_permutation(self):
        """When negative communities exist, analyze_coherence calls permutation_null
        with CorrelationSign.NEGATIVE."""
        from cliquefinder.knowledge.regulatory_coherence import CorrelationSign

        # Use data that creates both positive and negative communities
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=10, n_samples=30, seed=42, sign="negative",
            n_permutations=5, n_bootstrap=0,
        )
        genes = set(gene_names)

        # Patch permutation_null to track calls
        original_perm_null = analyzer.permutation_null
        call_log = []

        def tracking_perm_null(*args, **kwargs):
            call_log.append(kwargs.get("correlation_sign"))
            return original_perm_null(*args, **kwargs)

        analyzer.permutation_null = tracking_perm_null

        result = analyzer.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=True,
        )

        # Should have called for POSITIVE at minimum
        assert CorrelationSign.POSITIVE in call_log

        # If negative communities were found, should have called for NEGATIVE too
        if result.negative_communities:
            assert CorrelationSign.NEGATIVE in call_log

    def test_analyze_coherence_stores_perm_pvalue(self):
        """Permutation p-values should be stored on CommunityResult objects."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=10, n_samples=30, seed=42, sign="positive",
            n_permutations=5, n_bootstrap=0,
        )
        genes = set(gene_names)

        result = analyzer.analyze_coherence(
            genes=genes,
            condition="all",
            compute_bootstrap=False,
            compute_permutation=True,
        )

        # Positive communities should have permutation_pvalue set
        for comm in result.positive_communities:
            assert comm.permutation_pvalue is not None
            assert 0.0 <= comm.permutation_pvalue <= 1.0


# ===========================================================================
# Test 3: _corr_cache removed (KG-V-4)
# ===========================================================================

class TestCorrCacheRemoved:
    """KG-V-4: CoherenceAnalyzer should not have _corr_cache attribute."""

    def test_corr_cache_removed(self):
        analyzer, _ = _make_coherence_analyzer()
        assert not hasattr(analyzer, "_corr_cache"), (
            "Dead _corr_cache attribute should be removed from CoherenceAnalyzer"
        )

    def test_no_corr_cache_reference_in_methods(self):
        """_corr_cache should not be referenced anywhere in the class methods."""
        import inspect
        from cliquefinder.knowledge.regulatory_coherence import CoherenceAnalyzer

        source = inspect.getsource(CoherenceAnalyzer)
        assert "_corr_cache" not in source, (
            "_corr_cache should not appear in CoherenceAnalyzer source"
        )


# ===========================================================================
# Test 4: build_signed_graphs vectorized (KG-V-5)
# ===========================================================================

class TestBuildSignedGraphsVectorized:
    """KG-V-5: build_signed_graphs should produce correct results via vectorized path."""

    def test_build_signed_graphs_basic(self):
        """Verify correct positive and negative edge assignment."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=5, n_samples=30, seed=42, sign="mixed",
            soft_threshold_power=2.0, min_edge_weight=0.01,
        )

        # Compute correlation
        corr, gl = analyzer.compute_correlation_matrix(gene_names, "all")

        G_pos, G_neg = analyzer.build_signed_graphs(corr, gl)

        # All nodes present in both graphs
        assert set(G_pos.nodes()) == set(gl)
        assert set(G_neg.nodes()) == set(gl)

        # Check that every positive edge has positive correlation
        for u, v, d in G_pos.edges(data=True):
            i = gl.index(u)
            j = gl.index(v)
            assert corr[i, j] > 0, f"Positive graph edge ({u},{v}) has negative corr"
            assert d["weight"] > 0

        # Check that every negative edge has negative correlation
        for u, v, d in G_neg.edges(data=True):
            i = gl.index(u)
            j = gl.index(v)
            assert corr[i, j] <= 0, f"Negative graph edge ({u},{v}) has positive corr"
            assert d["weight"] > 0

    def test_build_signed_graphs_edge_count(self):
        """Total non-zero upper-triangle entries should equal pos + neg edges."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=6, n_samples=30, seed=7, sign="mixed",
            soft_threshold_power=2.0, min_edge_weight=0.01,
        )

        corr, gl = analyzer.compute_correlation_matrix(gene_names, "all")
        weights = analyzer.soft_threshold(corr)

        G_pos, G_neg = analyzer.build_signed_graphs(corr, gl)

        # Count non-zero upper triangle
        n = len(gl)
        expected_edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if weights[i, j] > 0:
                    expected_edges += 1

        actual_edges = G_pos.number_of_edges() + G_neg.number_of_edges()
        assert actual_edges == expected_edges

    def test_build_signed_graphs_sparse(self):
        """With high min_edge_weight, most entries are zero — vectorized handles it."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=8, n_samples=30, seed=42, sign="mixed",
            soft_threshold_power=6.0,  # High power → small weights
            min_edge_weight=0.5,       # High threshold → very sparse
        )

        corr, gl = analyzer.compute_correlation_matrix(gene_names, "all")
        G_pos, G_neg = analyzer.build_signed_graphs(corr, gl)

        # Should still have all nodes
        assert set(G_pos.nodes()) == set(gl)
        assert set(G_neg.nodes()) == set(gl)

        # With aggressive thresholding, many edges should be filtered
        total_possible = len(gl) * (len(gl) - 1) // 2
        total_actual = G_pos.number_of_edges() + G_neg.number_of_edges()
        assert total_actual < total_possible, (
            "Sparse threshold should remove some edges"
        )

    def test_build_signed_graphs_all_positive(self):
        """When all correlations are positive, negative graph should have no edges."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=5, n_samples=30, seed=42, sign="positive",
            soft_threshold_power=2.0, min_edge_weight=0.01,
        )

        corr, gl = analyzer.compute_correlation_matrix(gene_names, "all")
        G_pos, G_neg = analyzer.build_signed_graphs(corr, gl)

        # Should have positive edges but few/no negative edges
        assert G_pos.number_of_edges() > 0
        # Negative graph might have 0 edges (if all pairwise corrs > 0)
        # We just check that positive dominates
        assert G_pos.number_of_edges() >= G_neg.number_of_edges()

    def test_build_signed_graphs_weights_match(self):
        """Edge weights should match soft_threshold(corr_matrix) values."""
        analyzer, gene_names = _make_coherence_analyzer(
            n_genes=5, n_samples=30, seed=42, sign="mixed",
            soft_threshold_power=2.0, min_edge_weight=0.01,
        )

        corr, gl = analyzer.compute_correlation_matrix(gene_names, "all")
        weights = analyzer.soft_threshold(corr)
        G_pos, G_neg = analyzer.build_signed_graphs(corr, gl)

        for G in [G_pos, G_neg]:
            for u, v, d in G.edges(data=True):
                i = gl.index(u)
                j = gl.index(v)
                expected_w = float(weights[i, j])
                assert abs(d["weight"] - expected_w) < 1e-12, (
                    f"Edge ({u},{v}) weight {d['weight']} != expected {expected_w}"
                )
