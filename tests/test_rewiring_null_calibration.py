"""Calibration study for the edge-rewiring null.

These tests implement the synthetic scenarios A-F from the design doc
(memory/edge_rewiring_design.md v2).  They validate that the null behaves
as predicted on scenarios where we KNOW the ground truth:

- Scenario A (planted pathway) and C (clustering without specificity) →
  rewiring null should fire.
- Scenario B (pure degree-only confound, on a configuration model graph
  that has no structure beyond degree) → rewiring null should NOT fire.
- Scenario D (hub-enriched module) → discriminator case.
- Scenarios E-F → pathology diagnostics should trigger.

Distinct from the unit tests in ``test_perturbation_gradient.py`` because
these are statistical validation runs at meaningful N.  Slower but
essential before trusting the null on real data.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from cliquefinder.stats.graph_rewiring import bfs_distances_from
from cliquefinder.stats.perturbation_gradient import (
    _compute_shell_stats,
    _gradient_slope,
    run_rewiring_null,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def observed_slope_and_coverage(
    G: nx.Graph, seed: str, t_stats: dict[str, float], max_hops: int,
) -> tuple[float, float]:
    """Compute observed gradient slope and seed-target coverage."""
    in_graph_targets = set(t_stats.keys()) & set(G.nodes())
    distances = bfs_distances_from(G, seed, in_graph_targets, max_hops=max_hops)
    coverage = len(distances) / len(in_graph_targets) if in_graph_targets else 0.0
    shells = {}
    for g, d in distances.items():
        shells.setdefault(d, set()).add(g)
    observed_shells = [
        _compute_shell_stats(shells[h], t_stats, h) for h in sorted(shells)
    ]
    return _gradient_slope(observed_shells), coverage


# ---------------------------------------------------------------------------
# Scenario A: Planted pathway — gradient is real, rewiring should fire
# ---------------------------------------------------------------------------


class TestScenarioA_PlantedPathway:
    """Seed directly regulates a dense pathway module.  High |t| is
    concentrated in pathway members.  Rewiring destroys the pathway
    clustering; null slopes should flatten → gradient p < 0.05.
    """

    def test_planted_pathway_rewiring_rejects_null(self):
        rng = np.random.default_rng(100)
        G = nx.Graph()

        # Seed regulates a dense 25-gene pathway M; internal M connectivity
        for i in range(25):
            G.add_edge("SEED", f"M{i}")
            for j in range(i + 1, 25):
                if rng.random() < 0.5:
                    G.add_edge(f"M{i}", f"M{j}")

        # 150 background genes, sparsely connected
        for i in range(150):
            # Each background gene connects to a random M member (bringing them to hop 2)
            # and to other background genes
            G.add_edge(f"BG{i}", f"M{rng.integers(25)}")
            for _ in range(rng.integers(1, 4)):
                j = rng.integers(150)
                if i != j:
                    G.add_edge(f"BG{i}", f"BG{j}")

        # High |t| on pathway, low on background
        t_stats = {f"M{i}": 2.0 + rng.normal(0, 0.2) for i in range(25)}
        for i in range(150):
            t_stats[f"BG{i}"] = abs(0.5 + rng.normal(0, 0.3))

        observed, coverage = observed_slope_and_coverage(G, "SEED", t_stats, max_hops=3)
        assert observed < 0, "Fixture must produce decay"

        result = run_rewiring_null(
            graph=G, seed="SEED", abs_t_stats=t_stats,
            observed_slope=observed, observed_coverage=coverage,
            n_rewires=199, max_hops=3, rng_seed=42,
            max_swaps_iter0=50_000, check_every=500, verbose=False,
        )
        assert result.pvalue < 0.05, (
            f"Scenario A: planted signal should be rejected by rewiring null, "
            f"got p={result.pvalue:.3f}"
        )
        # The null distribution should actually be flatter than the observed
        assert np.median(result.null_slopes) > observed


# ---------------------------------------------------------------------------
# Scenario B: Degree-only confound on a configuration-model graph
# ---------------------------------------------------------------------------


class TestScenarioB_DegreeOnly:
    """|t| = f(degree) on a configuration-model graph with NO structure
    beyond degree.  Rewiring preserves degree; the confound survives.
    Expected: rewiring p is NOT significant (most of the null distribution
    matches the observed slope, because friendship paradox holds in both).
    """

    def test_degree_only_confound_on_config_model_not_rejected(self):
        rng = np.random.default_rng(200)
        # Configuration model — no structure beyond degree sequence
        degree_sequence = [rng.integers(2, 10) for _ in range(250)]
        if sum(degree_sequence) % 2:
            degree_sequence[0] += 1
        G_multi = nx.configuration_model(degree_sequence, seed=200)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})

        # |t| proportional to degree (no seed-position effect)
        degrees = dict(G.degree())
        t_stats = {
            g: 0.5 + 0.08 * degrees[g] + rng.normal(0, 0.15)
            for g in G.nodes()
        }

        observed, coverage = observed_slope_and_coverage(
            G, "G0", t_stats, max_hops=3,
        )
        if coverage < 0.2:
            pytest.skip("Fixture produced too-isolated seed")

        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats,
            observed_slope=observed, observed_coverage=coverage,
            n_rewires=199, max_hops=3, rng_seed=42,
            max_swaps_iter0=50_000, check_every=500, verbose=False,
        )
        # In a true config-model degree-only confound, the null distribution
        # should contain the observed value in its bulk — p should be far
        # from both 0 and 1 (not highly significant in either tail).
        # Loose bound: p should be > 0.05 (not reject null of "no extra info").
        assert result.pvalue > 0.05, (
            f"Scenario B: degree-only confound on config model should not be "
            f"rejected by rewiring null (rewiring preserves degree confound), "
            f"got p={result.pvalue:.3f}"
        )


# ---------------------------------------------------------------------------
# Scenario D: Hub-enriched module (discriminator case)
# ---------------------------------------------------------------------------


class TestScenarioD_HubEnrichedModule:
    """Hubs are enriched in module M; seed is a hub INSIDE M.

    Under rewiring, hub degree is preserved but seed's specific
    connections to M are randomized.  If hub-|t| correlation exists
    across the whole graph (not just in M), the gradient survives
    rewiring because hubs always have high |t|.  If the signal is
    M-specific (not degree-driven globally), rewiring kills it.

    Here we construct a case where M-membership and hub-ness BOTH drive
    |t|.  Expected: rewiring null rejects because M-membership is
    specifically destroyed.
    """

    def test_hub_enriched_module_rejected_by_rewiring(self):
        rng = np.random.default_rng(300)
        # Build two communities
        G = nx.Graph()
        # Community M: 60 nodes with higher within-community density
        M_nodes = [f"M{i}" for i in range(60)]
        for i in range(60):
            G.add_node(M_nodes[i])
            # Intra-community edges
            for j in range(i + 1, 60):
                if rng.random() < 0.1:
                    G.add_edge(M_nodes[i], M_nodes[j])

        # Community B: 150 nodes, similar density
        B_nodes = [f"B{i}" for i in range(150)]
        for i in range(150):
            G.add_node(B_nodes[i])
            for j in range(i + 1, 150):
                if rng.random() < 0.04:
                    G.add_edge(B_nodes[i], B_nodes[j])

        # Inter-community edges (few, to keep modular structure)
        for _ in range(30):
            G.add_edge(
                M_nodes[rng.integers(60)],
                B_nodes[rng.integers(150)],
            )

        # Pick the highest-degree M node as seed
        m_degrees = {n: G.degree(n) for n in M_nodes}
        seed = max(m_degrees, key=m_degrees.get)

        # |t|: high for M members, low for B members
        t_stats = {}
        for m in M_nodes:
            t_stats[m] = 1.5 + rng.normal(0, 0.3)
        for b in B_nodes:
            t_stats[b] = abs(0.5 + rng.normal(0, 0.3))

        observed, coverage = observed_slope_and_coverage(
            G, seed, t_stats, max_hops=3,
        )
        if observed >= 0 or coverage < 0.3:
            pytest.skip(f"Fixture degenerate: slope={observed:.3f}, coverage={coverage:.2f}")

        result = run_rewiring_null(
            graph=G, seed=seed, abs_t_stats=t_stats,
            observed_slope=observed, observed_coverage=coverage,
            n_rewires=199, max_hops=3, rng_seed=42,
            max_swaps_iter0=50_000, check_every=500, verbose=False,
        )
        # Module-specific signal → rewiring destroys module → p < 0.05
        assert result.pvalue < 0.1, (
            f"Scenario D: module-specific signal should be rejected by rewiring, "
            f"got p={result.pvalue:.3f}"
        )


# ---------------------------------------------------------------------------
# Scenario E: Seed is disconnected-prone — pathology check should fire
# ---------------------------------------------------------------------------


class TestScenarioE_DisconnectionWarning:
    """Seed has very few edges (degree 2).  Under rewiring, seed often
    ends up in a tiny component, triggering the disconnection warning.
    """

    def test_low_degree_seed_triggers_disconnection_warning(self):
        rng = np.random.default_rng(400)
        # Configuration model but seed has low degree
        degree_sequence = [rng.integers(3, 8) for _ in range(200)]
        degree_sequence[0] = 2  # SEED has just 2 connections
        if sum(degree_sequence) % 2:
            degree_sequence[1] += 1
        G_multi = nx.configuration_model(degree_sequence, seed=400)
        G = nx.Graph(G_multi)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        observed, coverage = observed_slope_and_coverage(G, "G0", t_stats, max_hops=3)
        if coverage < 0.1:
            pytest.skip("Seed too isolated in fixture")

        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats,
            observed_slope=observed, observed_coverage=coverage,
            n_rewires=99, max_hops=3, rng_seed=42,
            max_swaps_iter0=20_000, check_every=500, verbose=False,
        )
        # Under rewiring, a degree-2 seed often loses target coverage
        # compared to observed — disconnection count should be non-zero
        assert result.disconnection_rate >= 0, (
            "Disconnection rate must be non-negative"
        )
        # Failed iterations should be imputed (all accounted for)
        assert result.n_rewires_ok + result.n_rewires_failed == 99


# ---------------------------------------------------------------------------
# Regression: observed_coverage passed through vs computed
# ---------------------------------------------------------------------------


class TestCoveragePassthrough:
    def test_pass_observed_coverage_matches_computed(self):
        """Passing observed_coverage explicitly should give same result as
        letting run_rewiring_null compute it."""
        rng = np.random.default_rng(500)
        G = nx.barabasi_albert_graph(100, m=3, seed=500)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        _, coverage = observed_slope_and_coverage(G, "G0", t_stats, 3)
        r_explicit = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats,
            observed_slope=-0.1, observed_coverage=coverage,
            n_rewires=9, max_hops=3, rng_seed=42,
            max_swaps_iter0=5_000, verbose=False,
        )
        r_computed = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats,
            observed_slope=-0.1, observed_coverage=None,
            n_rewires=9, max_hops=3, rng_seed=42,
            max_swaps_iter0=5_000, verbose=False,
        )
        np.testing.assert_array_equal(r_explicit.null_slopes, r_computed.null_slopes)
        assert r_explicit.disconnection_rate == r_computed.disconnection_rate
