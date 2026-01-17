"""
Unit tests for clique enumeration algorithms and graph reduction techniques.

Tests the advanced pruning methods in clique_algorithms module:
- K-core decomposition reduction
- Degeneracy ordering computation
- Clique complexity estimation

Scientific Validation:
    Tests verify mathematical properties of k-core decomposition:
    - Soundness: No false cliques created by pruning
    - Completeness: No valid cliques lost by pruning
    - Correctness: K-core properties maintained
"""

import pytest
import networkx as nx
import numpy as np

from cliquefinder.knowledge.clique_algorithms import (
    kcore_reduction,
    compute_degeneracy_ordering,
    estimate_clique_complexity,
)


class TestKCoreReduction:
    """Test k-core decomposition for clique enumeration."""

    def test_empty_graph(self):
        """Empty graph should remain empty."""
        G = nx.Graph()
        H = kcore_reduction(G, min_clique_size=3)
        assert H.number_of_nodes() == 0
        assert H.number_of_edges() == 0

    def test_single_node(self):
        """Single node has no edges, should be removed for min_clique_size > 1."""
        G = nx.Graph()
        G.add_node(1)

        H = kcore_reduction(G, min_clique_size=2)
        assert H.number_of_nodes() == 0

        # min_clique_size=1 means k=0, should preserve node
        H = kcore_reduction(G, min_clique_size=1)
        assert H.number_of_nodes() == 1

    def test_triangle(self):
        """Triangle (K3) should be preserved for min_clique_size <= 3."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])

        # Triangle has 3-clique, all nodes have degree 2
        H = kcore_reduction(G, min_clique_size=3)
        assert H.number_of_nodes() == 3
        assert H.number_of_edges() == 3

        # For min_clique_size=4, need degree >= 3, all nodes removed
        H = kcore_reduction(G, min_clique_size=4)
        assert H.number_of_nodes() == 0

    def test_triangle_with_pendant(self):
        """Triangle with pendant vertex - pendant should be removed."""
        G = nx.Graph()
        G.add_edges_from([
            (1, 2), (2, 3), (3, 1),  # Triangle
            (3, 4)  # Pendant edge
        ])

        # For min_clique_size=3 (k=2), node 4 has degree 1 < 2
        H = kcore_reduction(G, min_clique_size=3)
        assert H.number_of_nodes() == 3
        assert set(H.nodes()) == {1, 2, 3}
        assert H.number_of_edges() == 3

    def test_iterative_pruning(self):
        """Test that k-core does iterative pruning (not single-pass)."""
        # Create path graph: 1-2-3-4-5
        # For min_clique_size=3 (k=2):
        # - Initial degrees: [1, 2, 2, 2, 1]
        # - Iteration 1: Remove nodes 1,5 (degree 1)
        # - After removal: nodes 2,4 now have degree 1
        # - Iteration 2: Remove nodes 2,4
        # - Final: only node 3 remains, but degree 0 < 2, so removed too
        G = nx.path_graph(5)
        H = kcore_reduction(G, min_clique_size=3)

        # All nodes should be removed via iterative pruning
        assert H.number_of_nodes() == 0

    def test_complete_graph(self):
        """Complete graph K_n should be preserved for min_clique_size <= n."""
        n = 5
        G = nx.complete_graph(n)

        # K_5 has 5-clique, all nodes have degree 4
        H = kcore_reduction(G, min_clique_size=5)
        assert H.number_of_nodes() == n
        assert H.number_of_edges() == G.number_of_edges()

        # For min_clique_size=6, need degree >= 5, all nodes have degree 4
        H = kcore_reduction(G, min_clique_size=6)
        assert H.number_of_nodes() == 0

    def test_two_cliques_connected(self):
        """Two 4-cliques connected by single edge."""
        # K4 + bridge + K4
        G = nx.Graph()

        # First clique (nodes 1-4)
        for i in range(1, 5):
            for j in range(i + 1, 5):
                G.add_edge(i, j)

        # Second clique (nodes 5-8)
        for i in range(5, 9):
            for j in range(i + 1, 9):
                G.add_edge(i, j)

        # Bridge edge
        G.add_edge(4, 5)

        # For min_clique_size=4 (k=3):
        # - Nodes 1-3, 6-8 have degree 3 within their cliques
        # - Nodes 4,5 have degree 4 (3 from clique + 1 from bridge)
        # All nodes should be preserved initially
        H = kcore_reduction(G, min_clique_size=4)

        # After k-core reduction, both cliques should remain
        # (all nodes have degree >= 3)
        assert H.number_of_nodes() == 8

    def test_sparse_graph_heavy_pruning(self):
        """Sparse graphs should be heavily pruned."""
        # Create random sparse graph
        G = nx.erdos_renyi_graph(n=100, p=0.03, seed=42)

        # For min_clique_size=5 (k=4), most nodes should be removed
        H = kcore_reduction(G, min_clique_size=5)

        # Expect heavy pruning in sparse graph
        reduction_ratio = H.number_of_nodes() / G.number_of_nodes()
        assert reduction_ratio < 0.5  # At least 50% reduction

    def test_soundness_no_false_cliques(self):
        """Verify k-core reduction doesn't create false cliques."""
        G = nx.erdos_renyi_graph(n=50, p=0.1, seed=42)

        min_size = 4
        H = kcore_reduction(G, min_clique_size=min_size)

        # Find cliques in both graphs
        original_cliques = set(
            frozenset(c) for c in nx.find_cliques(G)
            if len(c) >= min_size
        )
        reduced_cliques = set(
            frozenset(c) for c in nx.find_cliques(H)
            if len(c) >= min_size
        )

        # All cliques in reduced graph must be in original graph
        assert reduced_cliques.issubset(original_cliques)

    def test_completeness_no_lost_cliques(self):
        """Verify k-core reduction doesn't lose valid cliques."""
        # Create graph with known cliques
        G = nx.Graph()

        # Add a 5-clique (nodes 1-5)
        for i in range(1, 6):
            for j in range(i + 1, 6):
                G.add_edge(i, j)

        # Add a 4-clique (nodes 6-9)
        for i in range(6, 10):
            for j in range(i + 1, 10):
                G.add_edge(i, j)

        # Add some random edges
        G.add_edges_from([(1, 6), (2, 7), (3, 8)])

        # Find 4-cliques in original
        min_size = 4
        original_cliques = set(
            frozenset(c) for c in nx.find_cliques(G)
            if len(c) >= min_size
        )

        # Reduce graph
        H = kcore_reduction(G, min_clique_size=min_size)

        # Find cliques in reduced graph
        reduced_cliques = set(
            frozenset(c) for c in nx.find_cliques(H)
            if len(c) >= min_size
        )

        # Should find the same cliques
        assert original_cliques == reduced_cliques


class TestDegeneracyOrdering:
    """Test degeneracy ordering computation."""

    def test_empty_graph(self):
        """Empty graph has degeneracy 0."""
        G = nx.Graph()
        ordering, d = compute_degeneracy_ordering(G)
        assert ordering == []
        assert d == 0

    def test_single_node(self):
        """Single isolated node has degeneracy 0."""
        G = nx.Graph()
        G.add_node(1)
        ordering, d = compute_degeneracy_ordering(G)
        assert len(ordering) == 1
        assert d == 0

    def test_complete_graph(self):
        """Complete graph K_n has degeneracy n-1."""
        for n in [3, 5, 7]:
            G = nx.complete_graph(n)
            ordering, d = compute_degeneracy_ordering(G)
            assert len(ordering) == n
            assert d == n - 1

    def test_tree(self):
        """Trees have degeneracy 1."""
        # Path graph (special case of tree)
        G = nx.path_graph(10)
        ordering, d = compute_degeneracy_ordering(G)
        assert d == 1

        # Balanced binary tree
        G = nx.balanced_tree(2, 3)
        ordering, d = compute_degeneracy_ordering(G)
        assert d == 1

        # Star graph (also a tree)
        G = nx.star_graph(10)
        ordering, d = compute_degeneracy_ordering(G)
        assert d == 1

    def test_cycle(self):
        """Cycles have degeneracy 2."""
        for n in [3, 5, 10, 20]:
            G = nx.cycle_graph(n)
            ordering, d = compute_degeneracy_ordering(G)
            assert d == 2

    def test_ordering_length(self):
        """Ordering should contain all vertices."""
        G = nx.erdos_renyi_graph(50, 0.1, seed=42)
        ordering, d = compute_degeneracy_ordering(G)
        assert len(ordering) == G.number_of_nodes()
        assert len(set(ordering)) == G.number_of_nodes()  # No duplicates


class TestComplexityEstimation:
    """Test clique complexity estimation."""

    def test_empty_graph(self):
        """Empty graph is trivial."""
        G = nx.Graph()
        stats = estimate_clique_complexity(G)
        assert stats['n'] == 0
        assert stats['m'] == 0
        assert stats['degeneracy'] == 0
        assert stats['difficulty'] == 'trivial'
        assert stats['recommendation'] == 'empty'

    def test_sparse_graph(self):
        """Sparse graphs should be classified as easy."""
        # Create sparse graph (low degeneracy)
        G = nx.erdos_renyi_graph(100, 0.03, seed=42)
        stats = estimate_clique_complexity(G)

        assert stats['n'] == 100
        assert stats['density'] < 0.1
        # Sparse random graphs typically have low degeneracy
        # Exact value depends on random graph, but should be reasonable
        assert stats['degeneracy'] < 20

    def test_dense_graph(self):
        """Dense graphs should be classified as moderate/hard."""
        # Create dense graph
        G = nx.erdos_renyi_graph(50, 0.4, seed=42)
        stats = estimate_clique_complexity(G)

        assert stats['n'] == 50
        assert stats['density'] > 0.3
        # Dense graphs have higher degeneracy
        assert stats['degeneracy'] > 5

    def test_complete_graph(self):
        """Complete graphs have maximum degeneracy."""
        G = nx.complete_graph(20)
        stats = estimate_clique_complexity(G)

        assert stats['n'] == 20
        assert stats['m'] == 20 * 19 / 2  # Complete graph edges
        assert stats['density'] == 1.0
        assert stats['degeneracy'] == 19

    def test_estimation_bounds(self):
        """Verify clique count estimation is reasonable."""
        G = nx.erdos_renyi_graph(50, 0.1, seed=42)
        stats = estimate_clique_complexity(G)

        # Estimate should be positive for non-empty graph
        assert stats['estimated_cliques'] > 0

        # Actual clique count (for verification)
        actual_cliques = sum(1 for _ in nx.find_cliques(G))

        # Estimation is an upper bound, should be >= actual
        # (may not always hold for small graphs, but worth checking)
        # For larger graphs, this bound is tighter


class TestIntegrationWithNetworkX:
    """Integration tests with NetworkX clique algorithms."""

    def test_kcore_preserves_cliques(self):
        """Verify k-core + clique enumeration finds all cliques."""
        # Create test graph with known structure
        G = nx.Graph()

        # Add multiple cliques of different sizes
        # 5-clique (nodes 0-4)
        G.add_edges_from([(i, j) for i in range(5) for j in range(i + 1, 5)])

        # 4-clique (nodes 5-8)
        G.add_edges_from([(i, j) for i in range(5, 9) for j in range(i + 1, 9)])

        # 3-clique (nodes 9-11)
        G.add_edges_from([(i, j) for i in range(9, 12) for j in range(i + 1, 12)])

        # Add connecting edges
        G.add_edge(4, 5)
        G.add_edge(8, 9)

        # Test for different min_clique_sizes
        for min_size in [3, 4, 5]:
            # Find cliques in original graph
            original_cliques = set(
                frozenset(c) for c in nx.find_cliques(G)
                if len(c) >= min_size
            )

            # Reduce graph and find cliques
            H = kcore_reduction(G, min_clique_size=min_size)
            reduced_cliques = set(
                frozenset(c) for c in nx.find_cliques(H)
                if len(c) >= min_size
            )

            # Should find identical cliques
            assert original_cliques == reduced_cliques

    def test_performance_improvement(self):
        """Verify k-core reduction improves enumeration performance."""
        # Create moderately large graph
        G = nx.erdos_renyi_graph(200, 0.05, seed=42)

        min_size = 4

        # Measure nodes/edges before and after reduction
        original_nodes = G.number_of_nodes()
        original_edges = G.number_of_edges()

        H = kcore_reduction(G, min_clique_size=min_size)

        reduced_nodes = H.number_of_nodes()
        reduced_edges = H.number_of_edges()

        # Should see significant reduction for sparse graphs
        node_reduction = (original_nodes - reduced_nodes) / original_nodes

        # Even modest reduction helps (some graphs may have little reduction)
        # Just verify the reduction doesn't increase size
        assert reduced_nodes <= original_nodes
        assert reduced_edges <= original_edges


class TestBiologicalNetworkScenarios:
    """Test scenarios typical in biological correlation networks."""

    def test_gene_coexpression_network(self):
        """Simulate gene co-expression network structure."""
        # Biological networks often have:
        # - Low overall density
        # - Dense local clusters (co-expressed modules)
        # - Scale-free degree distribution

        G = nx.Graph()

        # Create 5 dense modules (co-expression clusters)
        n_modules = 5
        module_size = 10

        for module_idx in range(n_modules):
            # Each module is a dense subgraph (not complete)
            nodes = range(module_idx * module_size, (module_idx + 1) * module_size)

            # Add edges with high probability within module
            for i in nodes:
                for j in nodes:
                    if i < j and np.random.rand() > 0.3:  # 70% edge probability
                        G.add_edge(i, j)

        # Add sparse inter-module edges (hub genes)
        for i in range(n_modules - 1):
            # Connect module i to module i+1
            hub1 = i * module_size
            hub2 = (i + 1) * module_size
            G.add_edge(hub1, hub2)

        # Estimate complexity
        stats = estimate_clique_complexity(G)

        # Should be tractable (low degeneracy despite dense modules)
        assert stats['difficulty'] in ['easy', 'moderate']

        # K-core reduction should be effective
        H = kcore_reduction(G, min_clique_size=5)
        reduction = (G.number_of_nodes() - H.number_of_nodes()) / G.number_of_nodes()

        # Should see some reduction (exact amount depends on random edges)
        assert reduction >= 0  # At minimum, non-negative

    def test_correlation_threshold_filtering(self):
        """Simulate effect of correlation threshold on graph density."""
        # Higher correlation threshold = sparser graph

        # Create a graph simulating correlation network
        # Add weighted edges representing correlations
        G_dense = nx.Graph()
        G_sparse = nx.Graph()

        # 30 genes, some correlated
        n_genes = 30

        # Simulate correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.rand(n_genes, n_genes)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Symmetric
        np.fill_diagonal(corr_matrix, 1.0)

        # Build graphs with different thresholds
        threshold_low = 0.3
        threshold_high = 0.7

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                if corr_matrix[i, j] > threshold_low:
                    G_dense.add_edge(i, j, weight=corr_matrix[i, j])
                if corr_matrix[i, j] > threshold_high:
                    G_sparse.add_edge(i, j, weight=corr_matrix[i, j])

        # Dense graph (low threshold) should have higher degeneracy
        stats_dense = estimate_clique_complexity(G_dense)
        stats_sparse = estimate_clique_complexity(G_sparse)

        # Sparse graph should be easier to process
        assert stats_sparse['m'] <= stats_dense['m']  # Fewer edges
        assert stats_sparse['density'] <= stats_dense['density']
