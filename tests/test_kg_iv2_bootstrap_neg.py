"""Tests for KG-IV-2: bootstrap_stability for negative communities.

Verifies that:
1. The `correlation_sign` parameter selects the correct graph (positive vs negative).
2. Negative community bootstrap stability is computed when `comm_neg` is non-empty.
3. Bootstrap stability is still not computed when `compute_bootstrap=False`.
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.knowledge.regulatory_coherence import (
    CoherenceAnalyzer,
    CoherenceConfig,
    CorrelationSign,
)


def _make_matrix(n_genes=10, n_samples=30, seed=42):
    """Create a BioMatrix with structure that produces both positive and negative correlations."""
    rng = np.random.RandomState(seed)

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    sample_names = [f"S{i}" for i in range(n_samples)]

    # Create data with two anticorrelated blocks so we get both pos and neg edges.
    # Block 1 (genes 0-4): positively correlated
    # Block 2 (genes 5-9): positively correlated within, negatively with block 1
    base_signal = rng.randn(n_samples)
    data = np.zeros((n_genes, n_samples))
    for i in range(n_genes // 2):
        data[i] = base_signal + rng.randn(n_samples) * 0.3
    for i in range(n_genes // 2, n_genes):
        data[i] = -base_signal + rng.randn(n_samples) * 0.3

    # Add baseline expression so filter_genes doesn't remove everything
    data += 10.0

    feature_ids = pd.Index(gene_names)
    sample_ids = pd.Index(sample_names)
    metadata = pd.DataFrame({"phenotype": ["A"] * n_samples}, index=sample_ids)
    quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32)

    return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)


def _make_analyzer(matrix, n_bootstrap=5):
    """Create a CoherenceAnalyzer with relaxed config for testing."""
    config = CoherenceConfig(
        n_bootstrap=n_bootstrap,
        min_samples_per_condition=3,
        min_expression_percentile=0.0,
        min_variance_percentile=0.0,
        min_community_size=2,
        min_community_density=0.0,
        soft_threshold_power=2.0,  # low power so edges survive
        min_edge_weight=0.01,
    )
    return CoherenceAnalyzer(matrix, config=config)


class TestBootstrapStabilityCorrelationSign:
    """Tests for the correlation_sign parameter on bootstrap_stability."""

    def test_positive_sign_uses_positive_graph(self):
        """With correlation_sign=POSITIVE, bootstrap_stability should use G_pos."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=3)
        genes = set(matrix.feature_ids)

        # Spy on detect_communities and build_signed_graphs
        original_build = analyzer.build_signed_graphs
        graphs_built = []

        def spy_build(corr, gene_list):
            g_pos, g_neg = original_build(corr, gene_list)
            graphs_built.append((g_pos, g_neg))
            return g_pos, g_neg

        original_detect = analyzer.detect_communities
        detect_calls = []

        def spy_detect(G, method=None):
            detect_calls.append(G)
            return original_detect(G, method)

        with patch.object(analyzer, "build_signed_graphs", side_effect=spy_build):
            with patch.object(analyzer, "detect_communities", side_effect=spy_detect):
                result = analyzer.bootstrap_stability(
                    genes, "all", n_bootstrap=3,
                    correlation_sign=CorrelationSign.POSITIVE,
                )

        assert isinstance(result, dict)
        # Every detect_communities call should have received the positive graph
        for i, (g_pos, g_neg) in enumerate(graphs_built):
            assert detect_calls[i] is g_pos, (
                f"Call {i}: expected positive graph, got negative"
            )

    def test_negative_sign_uses_negative_graph(self):
        """With correlation_sign=NEGATIVE, bootstrap_stability should use G_neg."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=3)
        genes = set(matrix.feature_ids)

        original_build = analyzer.build_signed_graphs
        graphs_built = []

        def spy_build(corr, gene_list):
            g_pos, g_neg = original_build(corr, gene_list)
            graphs_built.append((g_pos, g_neg))
            return g_pos, g_neg

        original_detect = analyzer.detect_communities
        detect_calls = []

        def spy_detect(G, method=None):
            detect_calls.append(G)
            return original_detect(G, method)

        with patch.object(analyzer, "build_signed_graphs", side_effect=spy_build):
            with patch.object(analyzer, "detect_communities", side_effect=spy_detect):
                result = analyzer.bootstrap_stability(
                    genes, "all", n_bootstrap=3,
                    correlation_sign=CorrelationSign.NEGATIVE,
                )

        assert isinstance(result, dict)
        # Every detect_communities call should have received the negative graph
        for i, (g_pos, g_neg) in enumerate(graphs_built):
            assert detect_calls[i] is g_neg, (
                f"Call {i}: expected negative graph, got positive"
            )

    def test_default_sign_uses_positive_graph(self):
        """With no correlation_sign (None default), bootstrap_stability uses G_pos."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=2)
        genes = set(matrix.feature_ids)

        original_build = analyzer.build_signed_graphs
        graphs_built = []

        def spy_build(corr, gene_list):
            g_pos, g_neg = original_build(corr, gene_list)
            graphs_built.append((g_pos, g_neg))
            return g_pos, g_neg

        original_detect = analyzer.detect_communities
        detect_calls = []

        def spy_detect(G, method=None):
            detect_calls.append(G)
            return original_detect(G, method)

        with patch.object(analyzer, "build_signed_graphs", side_effect=spy_build):
            with patch.object(analyzer, "detect_communities", side_effect=spy_detect):
                result = analyzer.bootstrap_stability(
                    genes, "all", n_bootstrap=2,
                )

        assert isinstance(result, dict)
        for i, (g_pos, g_neg) in enumerate(graphs_built):
            assert detect_calls[i] is g_pos


class TestAnalyzeRegulatorNegBootstrap:
    """Tests for negative bootstrap stability in analyze_coherence."""

    def test_neg_bootstrap_computed_when_neg_communities_exist(self):
        """analyze_coherence should call bootstrap_stability for negative communities."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=3)
        genes = set(matrix.feature_ids)

        bootstrap_calls = []
        original_bootstrap = analyzer.bootstrap_stability

        def spy_bootstrap(genes, condition, n_bootstrap=None, correlation_sign=None):
            bootstrap_calls.append(correlation_sign)
            return original_bootstrap(
                genes, condition, n_bootstrap=n_bootstrap,
                correlation_sign=correlation_sign,
            )

        with patch.object(analyzer, "bootstrap_stability", side_effect=spy_bootstrap):
            result = analyzer.analyze_coherence(
                genes, "all", compute_bootstrap=True, compute_permutation=False,
            )

        # We should have at least a POSITIVE call
        assert CorrelationSign.POSITIVE in bootstrap_calls, (
            "Expected bootstrap_stability called with POSITIVE"
        )

        # If negative communities were found, NEGATIVE should also be in calls
        if result.negative_communities:
            assert CorrelationSign.NEGATIVE in bootstrap_calls, (
                "Expected bootstrap_stability called with NEGATIVE when neg communities exist"
            )

    def test_neg_bootstrap_not_computed_when_no_neg_communities(self):
        """If there are no negative communities, skip negative bootstrap."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=3)
        genes = set(matrix.feature_ids)

        bootstrap_calls = []
        original_bootstrap = analyzer.bootstrap_stability

        def spy_bootstrap(genes, condition, n_bootstrap=None, correlation_sign=None):
            bootstrap_calls.append(correlation_sign)
            return original_bootstrap(
                genes, condition, n_bootstrap=n_bootstrap,
                correlation_sign=correlation_sign,
            )

        # Force no negative edges by making all correlations positive:
        # Use only the first 5 genes (the positively correlated block)
        pos_genes = {f"GENE{i}" for i in range(5)}

        with patch.object(analyzer, "bootstrap_stability", side_effect=spy_bootstrap):
            result = analyzer.analyze_coherence(
                pos_genes, "all", compute_bootstrap=True, compute_permutation=False,
            )

        # POSITIVE call should exist
        assert CorrelationSign.POSITIVE in bootstrap_calls

        # If no neg communities, NEGATIVE should NOT be called
        if not result.negative_communities:
            assert CorrelationSign.NEGATIVE not in bootstrap_calls

    def test_bootstrap_not_computed_when_disabled(self):
        """compute_bootstrap=False should skip all bootstrap computation."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=3)
        genes = set(matrix.feature_ids)

        bootstrap_calls = []
        original_bootstrap = analyzer.bootstrap_stability

        def spy_bootstrap(genes, condition, n_bootstrap=None, correlation_sign=None):
            bootstrap_calls.append(correlation_sign)
            return original_bootstrap(
                genes, condition, n_bootstrap=n_bootstrap,
                correlation_sign=correlation_sign,
            )

        with patch.object(analyzer, "bootstrap_stability", side_effect=spy_bootstrap):
            result = analyzer.analyze_coherence(
                genes, "all", compute_bootstrap=False, compute_permutation=False,
            )

        assert len(bootstrap_calls) == 0, (
            "bootstrap_stability should not be called when compute_bootstrap=False"
        )

        # All communities should have bootstrap_stability=None
        for comm in result.positive_communities + result.negative_communities:
            assert comm.bootstrap_stability is None

    def test_neg_communities_have_stability_scores(self):
        """Negative communities should have non-None bootstrap_stability values."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=5)
        genes = set(matrix.feature_ids)

        result = analyzer.analyze_coherence(
            genes, "all", compute_bootstrap=True, compute_permutation=False,
        )

        if result.negative_communities:
            for comm in result.negative_communities:
                assert comm.bootstrap_stability is not None, (
                    f"Negative community {comm.community_id} has None bootstrap_stability"
                )
                assert 0.0 <= comm.bootstrap_stability <= 1.0, (
                    f"bootstrap_stability {comm.bootstrap_stability} out of [0, 1] range"
                )

    def test_pos_communities_still_have_stability_scores(self):
        """Positive communities should still get bootstrap stability (regression check)."""
        matrix = _make_matrix()
        analyzer = _make_analyzer(matrix, n_bootstrap=5)
        genes = set(matrix.feature_ids)

        result = analyzer.analyze_coherence(
            genes, "all", compute_bootstrap=True, compute_permutation=False,
        )

        if result.positive_communities:
            for comm in result.positive_communities:
                assert comm.bootstrap_stability is not None, (
                    f"Positive community {comm.community_id} has None bootstrap_stability"
                )
                assert 0.0 <= comm.bootstrap_stability <= 1.0
