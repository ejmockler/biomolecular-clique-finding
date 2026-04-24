"""Tests for graph_rewiring — Maslov-Sneppen, BFS, diagnostics."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from cliquefinder.stats.graph_rewiring import (
    MixingDiagnostic,
    bfs_distances_from,
    bimodality_coefficient,
    compute_undirected_degrees,
    disconnection_rate,
    edges_to_undirected_graph,
    rewire_maslov_sneppen,
    seed_component,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_graph():
    """A 6-node graph: SEED—A—B—C—D—E, plus extra edges for swappability."""
    G = nx.Graph()
    edges = [
        ("SEED", "A"), ("A", "B"), ("B", "C"),
        ("C", "D"), ("D", "E"),
        ("SEED", "B"), ("A", "C"),  # triangle-ish for more mixing
    ]
    G.add_edges_from(edges)
    return G


@pytest.fixture
def scale_free_graph():
    """A 500-node scale-free graph for mixing diagnostics."""
    rng = np.random.default_rng(42)
    G = nx.barabasi_albert_graph(500, m=3, seed=42)
    # Relabel nodes to strings so they look like gene symbols
    mapping = {i: f"G{i}" for i in G.nodes()}
    return nx.relabel_nodes(G, mapping)


# ---------------------------------------------------------------------------
# Undirected degrees
# ---------------------------------------------------------------------------


class TestComputeUndirectedDegrees:
    def test_basic(self, small_graph):
        deg = compute_undirected_degrees(small_graph)
        assert deg["SEED"] == 2  # connected to A, B
        assert deg["A"] == 3      # SEED, B, C
        assert deg["C"] == 3      # B, A, D

    def test_empty_graph(self):
        assert compute_undirected_degrees(nx.Graph()) == {}


# ---------------------------------------------------------------------------
# BFS distances
# ---------------------------------------------------------------------------


class TestBfsDistancesFrom:
    def test_linear_path(self):
        G = nx.path_graph(["A", "B", "C", "D", "E"])
        distances = bfs_distances_from(
            G, "A", {"B", "C", "D", "E"}, max_hops=4,
        )
        assert distances == {"B": 1, "C": 2, "D": 3, "E": 4}

    def test_max_hops_truncates(self):
        G = nx.path_graph(["A", "B", "C", "D", "E"])
        distances = bfs_distances_from(
            G, "A", {"B", "C", "D", "E"}, max_hops=2,
        )
        assert distances == {"B": 1, "C": 2}
        assert "D" not in distances

    def test_unreachable_omitted(self):
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("C", "D")])
        distances = bfs_distances_from(G, "A", {"B", "C", "D"}, max_hops=5)
        assert distances == {"B": 1}

    def test_seed_not_in_graph(self):
        G = nx.path_graph(["A", "B"])
        assert bfs_distances_from(G, "MISSING", {"A", "B"}, max_hops=3) == {}

    def test_seed_in_targets_excluded(self):
        G = nx.path_graph(["A", "B", "C"])
        distances = bfs_distances_from(G, "A", {"A", "B", "C"}, max_hops=3)
        assert "A" not in distances

    def test_early_termination_when_targets_found(self):
        # All targets reachable at hop 1; BFS should stop expanding
        G = nx.star_graph(["CTR", "X", "Y", "Z"])
        distances = bfs_distances_from(G, "CTR", {"X", "Y", "Z"}, max_hops=10)
        assert distances == {"X": 1, "Y": 1, "Z": 1}


# ---------------------------------------------------------------------------
# Maslov-Sneppen rewiring
# ---------------------------------------------------------------------------


class TestRewireMaslovSneppen:
    def test_preserves_degree_sequence_exactly(self, scale_free_graph):
        rng = np.random.default_rng(42)
        original_degrees = compute_undirected_degrees(scale_free_graph)

        rewired, diag = rewire_maslov_sneppen(
            scale_free_graph, rng, max_swaps=50_000, check_every=2000,
        )
        new_degrees = compute_undirected_degrees(rewired)

        # Degree sequence must be preserved exactly per-node
        for node in scale_free_graph.nodes():
            assert new_degrees[node] == original_degrees[node], \
                f"Degree changed for {node}: {original_degrees[node]} -> {new_degrees[node]}"

    def test_produces_simple_graph(self, scale_free_graph):
        rng = np.random.default_rng(7)
        rewired, _ = rewire_maslov_sneppen(
            scale_free_graph, rng, max_swaps=20_000, check_every=2000,
        )
        # No self-loops
        assert all(u != v for u, v in rewired.edges())
        # No multi-edges (networkx.Graph is simple by construction)
        assert not any(
            rewired.number_of_edges(u, v) > 1
            for u, v in rewired.edges()
        )

    def test_deterministic_with_same_seed(self, scale_free_graph):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        G1, _ = rewire_maslov_sneppen(scale_free_graph, rng1, max_swaps=10_000, check_every=1000)
        G2, _ = rewire_maslov_sneppen(scale_free_graph, rng2, max_swaps=10_000, check_every=1000)
        assert set(frozenset(e) for e in G1.edges()) == set(frozenset(e) for e in G2.edges())

    def test_different_seeds_give_different_graphs(self, scale_free_graph):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        G1, _ = rewire_maslov_sneppen(scale_free_graph, rng1, max_swaps=10_000, check_every=1000)
        G2, _ = rewire_maslov_sneppen(scale_free_graph, rng2, max_swaps=10_000, check_every=1000)
        assert set(frozenset(e) for e in G1.edges()) != set(frozenset(e) for e in G2.edges())

    def test_detects_mixing_plateau(self, scale_free_graph):
        """Diagnostic mode: plateau should be detected and distance
        from origin should grow then saturate."""
        rng = np.random.default_rng(42)
        _, diag = rewire_maslov_sneppen(
            scale_free_graph, rng,
            target_nswap=None,  # diagnostic mode
            max_swaps=500_000,
            check_every=1000,
            tolerance_pct=0.05,
            plateau_window=3,
            min_plateau_checkpoints=5,
            safety_margin=0.1,
        )
        assert diag.plateau_swaps is not None, "Should have detected plateau"
        assert diag.plateau_swaps > 0
        assert len(diag.swap_counts) >= 5  # at least min_plateau_checkpoints
        # Distance from origin should grow then saturate
        assert diag.distances_from_origin[-1] >= diag.distances_from_origin[0]

    def test_target_nswap_fixed_mode_respects_budget(self, scale_free_graph):
        """Fixed mode: run exactly target_nswap accepted swaps, no plateau."""
        rng = np.random.default_rng(7)
        _, diag = rewire_maslov_sneppen(
            scale_free_graph, rng,
            target_nswap=3000,
        )
        # plateau_swaps holds accepted_total in fixed mode (so callers can read it)
        assert diag.plateau_swaps == 3000
        # No trajectory recorded in fixed mode
        assert len(diag.swap_counts) == 0
        assert len(diag.distances_from_origin) == 0

    def test_fixed_mode_preserves_degree(self, scale_free_graph):
        """Fixed mode must still preserve degree per node."""
        rng = np.random.default_rng(3)
        original = compute_undirected_degrees(scale_free_graph)
        rewired, _ = rewire_maslov_sneppen(
            scale_free_graph, rng, target_nswap=2000,
        )
        new = compute_undirected_degrees(rewired)
        for node in scale_free_graph.nodes():
            assert new[node] == original[node]

    def test_diagnostic_to_dict(self, scale_free_graph):
        rng = np.random.default_rng(0)
        _, diag = rewire_maslov_sneppen(
            scale_free_graph, rng, max_swaps=10_000, check_every=1000,
        )
        d = diag.to_dict()
        assert "swap_counts" in d
        assert "distances_from_origin" in d
        assert "accepted_fraction" in d
        assert 0.0 <= d["accepted_fraction"] <= 1.0

    def test_tiny_graph_handled_gracefully(self):
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("C", "D")])
        rng = np.random.default_rng(0)
        rewired, diag = rewire_maslov_sneppen(G, rng, max_swaps=1000, check_every=100)
        # Should not error; may not mix but returns a valid graph
        assert rewired.number_of_edges() == 2

    def test_does_not_mutate_input(self, scale_free_graph):
        original_edges = set(frozenset(e) for e in scale_free_graph.edges())
        rng = np.random.default_rng(0)
        _ = rewire_maslov_sneppen(scale_free_graph, rng, max_swaps=5000, check_every=1000)
        post_edges = set(frozenset(e) for e in scale_free_graph.edges())
        assert original_edges == post_edges, "Input graph should not be mutated"


# ---------------------------------------------------------------------------
# Edge-list → graph helper
# ---------------------------------------------------------------------------


class TestEdgesToUndirectedGraph:
    def test_basic(self):
        G = edges_to_undirected_graph([("A", "B"), ("B", "C")])
        assert G.has_edge("A", "B")
        assert G.has_edge("B", "C")
        assert G.number_of_nodes() == 3

    def test_with_metadata_ignored(self):
        G = edges_to_undirected_graph([("A", "B", {"reliability": 0.9})])
        assert G.has_edge("A", "B")

    def test_self_loops_removed(self):
        G = edges_to_undirected_graph([("A", "A"), ("A", "B")])
        assert not G.has_edge("A", "A")
        assert G.has_edge("A", "B")

    def test_duplicate_edges_deduplicated(self):
        G = edges_to_undirected_graph([("A", "B"), ("B", "A"), ("A", "B")])
        assert G.number_of_edges() == 1


# ---------------------------------------------------------------------------
# Seed component
# ---------------------------------------------------------------------------


class TestSeedComponent:
    def test_extracts_connected_component(self):
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("X", "Y"), ("Y", "Z")])
        comp = seed_component(G, "A")
        assert set(comp.nodes()) == {"A", "B", "C"}
        assert "X" not in comp
        assert "Y" not in comp

    def test_seed_missing_returns_empty(self):
        G = nx.path_graph(["A", "B", "C"])
        assert seed_component(G, "MISSING").number_of_nodes() == 0

    def test_singleton_seed(self):
        G = nx.Graph()
        G.add_node("A")
        assert set(seed_component(G, "A").nodes()) == {"A"}


# ---------------------------------------------------------------------------
# Pathology diagnostics
# ---------------------------------------------------------------------------


class TestBimodalityCoefficient:
    """Pfister-style bimodality coefficient:
    BC > 5/9 (≈ 0.555) suggests bimodality; uniform is exactly 5/9."""

    def test_normal_unimodal_below_threshold(self):
        """Gaussian sample has BC well below 5/9."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        bc, p = bimodality_coefficient(x)
        assert bc < 5.0 / 9.0, f"Normal should have BC < 0.555, got {bc:.3f}"
        assert p >= 0.9  # well above threshold, no warning

    def test_bimodal_above_threshold(self):
        """Mixture-of-two-Gaussians sample should exceed BC=5/9."""
        rng = np.random.default_rng(0)
        x = np.concatenate([
            rng.normal(-3.0, 0.5, 500),
            rng.normal(+3.0, 0.5, 500),
        ])
        bc, p = bimodality_coefficient(x)
        assert bc > 5.0 / 9.0, f"Bimodal should have BC > 0.555, got {bc:.3f}"
        assert p < 1.0  # warning triggered

    def test_discriminates_unimodal_vs_bimodal(self):
        """The coefficient should be strictly larger for the bimodal case."""
        rng = np.random.default_rng(99)
        unimodal = rng.normal(0, 1, 1000)
        bimodal = np.concatenate([
            rng.normal(-3, 0.5, 500),
            rng.normal(+3, 0.5, 500),
        ])
        bc_uni, _ = bimodality_coefficient(unimodal)
        bc_bi, _ = bimodality_coefficient(bimodal)
        assert bc_bi > bc_uni
        # Margin should be large
        assert bc_bi - bc_uni > 0.2

    def test_tiny_sample_returns_one(self):
        bc, p = bimodality_coefficient(np.array([1.0, 2.0]))
        assert p == 1.0

    def test_zero_variance_handled(self):
        bc, p = bimodality_coefficient(np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
        assert bc == 0.0
        assert p == 1.0


class TestDisconnectionRate:
    def test_all_connected(self):
        targets = {"A", "B", "C"}
        components = [{"A", "B", "C", "D"}, {"A", "B", "C"}]
        assert disconnection_rate(components, targets) == 0.0

    def test_all_disconnected(self):
        targets = {"A", "B", "C", "D"}
        components = [{"A"}, {"A", "B"}]
        # First covers 1/4=0.25 < 0.5; second covers 2/4=0.5 which is NOT < 0.5
        # So first is disconnected, second is not.
        rate = disconnection_rate(components, targets)
        assert rate == 0.5

    def test_empty_inputs(self):
        assert disconnection_rate([], set()) == 0.0
        assert disconnection_rate([{"A"}], set()) == 0.0
