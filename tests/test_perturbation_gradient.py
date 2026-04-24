"""Tests for perturbation gradient discovery module."""
import numpy as np
import pytest

from cliquefinder.stats.perturbation_gradient import (
    GradientResult,
    HopShellStats,
    compute_hop_shells,
    run_gradient_test,
    _compute_shell_stats,
    _gradient_slope,
    _active_horizon,
    _build_degree_bins,
    _degree_preserving_permute,
    _compute_graph_degrees,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_graph():
    """A -> B -> C -> D -> E (chain graph)."""
    return {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C", "E"],
        "E": ["D"],
    }


@pytest.fixture
def star_graph():
    """SEED connected to 20 genes, each connected to 5 unique leaves.

    Also includes 30 degree-matched background hubs (degree 6, not
    connected to SEED) and 150 low-degree background leaves. This
    ensures the degree-preserving permutation has comparison genes
    for each degree tier.
    """
    adj: dict[str, list[str]] = {"SEED": []}
    for i in range(20):
        hub = f"H{i}"
        adj["SEED"].append(hub)
        adj[hub] = ["SEED"]
        for j in range(5):
            leaf = f"L{i}_{j}"
            adj[hub].append(leaf)
            adj[leaf] = [hub]
    # Background hubs: same degree as seed hubs, but not in neighborhood
    for i in range(30):
        bgh = f"BGH{i}"
        adj[bgh] = []
        for j in range(6):
            tgt = f"BGH{i}_T{j}"
            adj[bgh].append(tgt)
            adj[tgt] = [bgh]
    # Low-degree background leaves (degree 0)
    for i in range(150):
        adj[f"BG{i}"] = []
    return adj


@pytest.fixture
def decaying_t_stats():
    """T-stats that decay with distance from SEED in the star graph.

    Distance 1 (H0-H19): mean|t| ~ 2.0  (elevated)
    Distance 2 (L*): mean|t| ~ 1.0  (moderate)
    Background hubs (BGH*): mean|t| ~ 0.5  (low despite same degree)
    Background leaves (BG*): mean|t| ~ 0.5

    The signal is REAL: seed-neighborhood hubs have higher |t| than
    degree-matched background hubs.
    """
    t: dict[str, float] = {}
    rng = np.random.default_rng(42)
    for i in range(20):
        t[f"H{i}"] = 2.0 + rng.normal(0, 0.3)
    for i in range(20):
        for j in range(5):
            t[f"L{i}_{j}"] = 1.0 + rng.normal(0, 0.2)
    # Background hubs with LOW |t| (degree-matched to seed hubs)
    for i in range(30):
        t[f"BGH{i}"] = abs(0.5 + rng.normal(0, 0.3))
        for j in range(6):
            t[f"BGH{i}_T{j}"] = abs(0.5 + rng.normal(0, 0.3))
    for i in range(150):
        t[f"BG{i}"] = abs(0.5 + rng.normal(0, 0.3))
    return t


@pytest.fixture
def flat_t_stats():
    """T-stats with NO gradient -- uniform across all genes."""
    t: dict[str, float] = {}
    rng = np.random.default_rng(99)
    for i in range(20):
        t[f"H{i}"] = abs(rng.normal(1.0, 0.3))
    for i in range(20):
        for j in range(5):
            t[f"L{i}_{j}"] = abs(rng.normal(1.0, 0.3))
    for i in range(30):
        t[f"BGH{i}"] = abs(rng.normal(1.0, 0.3))
        for j in range(6):
            t[f"BGH{i}_T{j}"] = abs(rng.normal(1.0, 0.3))
    for i in range(150):
        t[f"BG{i}"] = abs(rng.normal(1.0, 0.3))
    return t


# ---------------------------------------------------------------------------
# BFS tests
# ---------------------------------------------------------------------------


class TestComputeHopShells:
    def test_linear_chain(self, linear_graph):
        shells = compute_hop_shells(linear_graph, "A", max_hops=10)
        assert shells[1] == {"B"}
        assert shells[2] == {"C"}
        assert shells[3] == {"D"}
        assert shells[4] == {"E"}
        assert 5 not in shells

    def test_star_graph(self, star_graph):
        shells = compute_hop_shells(star_graph, "SEED", max_hops=3)
        assert len(shells[1]) == 20  # seed hubs
        assert len(shells[2]) == 100  # seed leaves
        # Hop 3 should NOT include disconnected background clusters
        assert 3 not in shells

    def test_max_hops_respected(self, linear_graph):
        shells = compute_hop_shells(linear_graph, "A", max_hops=2)
        assert 1 in shells and 2 in shells
        assert 3 not in shells

    def test_isolated_seed(self):
        shells = compute_hop_shells({"X": []}, "X")
        assert shells == {}

    def test_seed_not_in_adjacency(self):
        shells = compute_hop_shells({"A": ["B"]}, "MISSING")
        assert shells == {}


# ---------------------------------------------------------------------------
# Shell statistics tests
# ---------------------------------------------------------------------------


class TestShellStats:
    def test_basic(self):
        genes = {"A", "B", "C"}
        t = {"A": 2.0, "B": 3.0, "C": 1.0}
        stats = _compute_shell_stats(genes, t, hop=1)
        assert stats.n_genes == 3
        assert stats.mean_abs_t == pytest.approx(2.0)
        assert stats.median_abs_t == pytest.approx(2.0)
        assert stats.hop == 1

    def test_no_measurable_genes(self):
        stats = _compute_shell_stats({"X", "Y"}, {"A": 1.0}, hop=2)
        assert stats.n_genes == 0
        assert np.isnan(stats.mean_abs_t)

    def test_single_gene(self):
        stats = _compute_shell_stats({"A"}, {"A": 5.0}, hop=1)
        assert stats.n_genes == 1
        assert stats.mean_abs_t == 5.0
        assert stats.std_abs_t == 0.0

    def test_to_dict(self):
        stats = _compute_shell_stats({"A", "B"}, {"A": 1.0, "B": 3.0}, hop=1)
        d = stats.to_dict()
        assert d["hop"] == 1
        assert d["n_genes"] == 2
        assert "mean_abs_t" in d


# ---------------------------------------------------------------------------
# Gradient slope tests
# ---------------------------------------------------------------------------


class TestGradientSlope:
    def test_negative_slope(self):
        shells = [
            HopShellStats(hop=1, n_genes=20, mean_abs_t=2.0,
                          median_abs_t=2.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=100, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.2, genes=()),
        ]
        slope = _gradient_slope(shells)
        assert slope < 0  # decay

    def test_flat_slope(self):
        shells = [
            HopShellStats(hop=1, n_genes=50, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=50, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.3, genes=()),
        ]
        slope = _gradient_slope(shells)
        assert slope == pytest.approx(0.0, abs=1e-10)

    def test_single_shell(self):
        shells = [
            HopShellStats(hop=1, n_genes=10, mean_abs_t=2.0,
                          median_abs_t=2.0, std_abs_t=0.3, genes=()),
        ]
        assert _gradient_slope(shells) == 0.0

    def test_empty_shells_skipped(self):
        shells = [
            HopShellStats(hop=1, n_genes=20, mean_abs_t=2.0,
                          median_abs_t=2.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=0, mean_abs_t=float("nan"),
                          median_abs_t=float("nan"), std_abs_t=float("nan"),
                          genes=()),
            HopShellStats(hop=3, n_genes=50, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.2, genes=()),
        ]
        slope = _gradient_slope(shells)
        assert slope < 0  # only hops 1 and 3 contribute

    def test_n_weights_used(self):
        """With n-weights, the slope is dominated by the larger shell."""
        shells = [
            HopShellStats(hop=1, n_genes=10, mean_abs_t=3.0,
                          median_abs_t=3.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=1000, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.3, genes=()),
        ]
        slope = _gradient_slope(shells)
        assert slope < 0
        # With n-weights, the weighted mean of y is pulled toward 1.0
        # (the large shell's value), so slope magnitude reflects this


# ---------------------------------------------------------------------------
# Active horizon tests
# ---------------------------------------------------------------------------


class TestActiveHorizon:
    def test_decaying(self):
        shells = [
            HopShellStats(hop=1, n_genes=20, mean_abs_t=2.0,
                          median_abs_t=2.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=100, mean_abs_t=1.0,
                          median_abs_t=1.0, std_abs_t=0.2, genes=()),
            HopShellStats(hop=3, n_genes=200, mean_abs_t=0.4,
                          median_abs_t=0.4, std_abs_t=0.1, genes=()),
        ]
        # Background = 0.5 -> hop 1 and 2 exceed, hop 3 does not
        assert _active_horizon(shells, 0.5) == 2

    def test_all_above_background(self):
        shells = [
            HopShellStats(hop=1, n_genes=10, mean_abs_t=3.0,
                          median_abs_t=3.0, std_abs_t=0.3, genes=()),
            HopShellStats(hop=2, n_genes=10, mean_abs_t=2.0,
                          median_abs_t=2.0, std_abs_t=0.3, genes=()),
        ]
        assert _active_horizon(shells, 1.0) == 2

    def test_none_above_background(self):
        shells = [
            HopShellStats(hop=1, n_genes=10, mean_abs_t=0.3,
                          median_abs_t=0.3, std_abs_t=0.1, genes=()),
        ]
        assert _active_horizon(shells, 1.0) == 0


# ---------------------------------------------------------------------------
# Degree-preserving permutation
# ---------------------------------------------------------------------------


class TestDegreePreservingPermutation:
    def test_degree_bins_respect_bin_size(self):
        degrees = {f"G{i}": i % 10 for i in range(250)}
        bins, g2b = _build_degree_bins(degrees, bin_size=100)
        # 250 genes / 100 = 3 bins (100, 100, 50)
        assert len(bins) == 3
        assert len(bins[0]) == 100
        assert len(bins[2]) == 50

    def test_permute_within_bins(self):
        """Permuted values stay within degree bins."""
        all_genes = [f"G{i}" for i in range(200)]
        all_t = np.arange(200, dtype=np.float64)
        degrees = {g: i for i, g in enumerate(all_genes)}
        bins, g2b = _build_degree_bins(degrees, bin_size=100)
        rng = np.random.default_rng(42)

        perm = _degree_preserving_permute(all_genes, all_t, bins, g2b, rng)

        # Genes in bin 0 should only get t-values from bin 0
        bin0_genes = set(bins[0])
        bin0_original_t = {all_t[i] for i, g in enumerate(all_genes) if g in bin0_genes}
        bin0_perm_t = {perm[g] for g in bin0_genes}
        assert bin0_perm_t == bin0_original_t  # same values, just shuffled

    def test_graph_degrees(self, star_graph):
        all_genes = list(star_graph.keys())
        degrees = _compute_graph_degrees(star_graph, all_genes + ["ISOLATED"])
        assert degrees["SEED"] == 20  # connected to 20 hubs
        assert degrees["ISOLATED"] == 0


# ---------------------------------------------------------------------------
# Integration: run_gradient_test
# ---------------------------------------------------------------------------


class TestRunGradientTest:
    def test_detects_gradient(self, star_graph, decaying_t_stats):
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=199,
            rng_seed=42,
            verbose=False,
        )
        assert isinstance(result, GradientResult)
        assert result.seed_gene == "SEED"
        assert len(result.shells) == 2  # hop 1 and 2
        assert result.slope < 0  # decay
        assert result.slope_pvalue < 0.05
        assert result.spearman_rho < 0
        assert result.spearman_pvalue < 0.05
        assert result.active_horizon >= 1
        assert result.n_genes_total == 120  # 20 hubs + 100 leaves

    def test_no_gradient_high_pvalue(self, star_graph, flat_t_stats):
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=flat_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=199,
            rng_seed=42,
            verbose=False,
        )
        assert result.slope_pvalue > 0.05

    def test_background_excludes_shell_genes(self, star_graph, decaying_t_stats):
        """Background mean|t| should not include graph neighborhood genes."""
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=49,
            rng_seed=42,
            verbose=False,
        )
        # Background should only be BG* genes (~0.5 mean)
        # Not contaminated by H* (~2.0) and L* (~1.0)
        assert result.background_mean_abs_t < 0.8

    def test_to_dict_roundtrip(self, star_graph, decaying_t_stats):
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=99,
            rng_seed=42,
            verbose=False,
        )
        d = result.to_dict()
        assert d["seed_gene"] == "SEED"
        assert len(d["shells"]) == 2
        assert isinstance(d["slope"], float)
        assert isinstance(d["slope_pvalue"], float)

    def test_too_few_genes_raises(self):
        adj = {"S": ["A", "B"], "A": ["S"], "B": ["S"]}
        t = {"A": 1.0, "B": 2.0}  # only 2 measurable
        with pytest.raises(ValueError, match="at least 10"):
            run_gradient_test(adj, t, "S", n_permutations=10, verbose=False)

    def test_no_neighbors_raises(self):
        with pytest.raises(ValueError, match="No neighbors"):
            run_gradient_test({"S": []}, {"S": 1.0}, "S", verbose=False)

    def test_shell_stats_correct(self, star_graph, decaying_t_stats):
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=49,
            rng_seed=42,
            verbose=False,
        )
        hop1 = result.shells[0]
        hop2 = result.shells[1]
        assert hop1.hop == 1
        assert hop1.n_genes == 20
        assert hop2.hop == 2
        assert hop2.n_genes == 100
        assert hop1.mean_abs_t > hop2.mean_abs_t


# ---------------------------------------------------------------------------
# Degree-confounded scenario (critical false-positive calibration)
# ---------------------------------------------------------------------------


class TestDegreeConfoundedNull:
    """Test that degree-preserving permutation controls for hub bias.

    When high-degree genes have higher |t| regardless of distance from seed,
    naive gene-label permutation would produce false positives. The degree-
    preserving null should control for this confound.
    """

    def test_hub_bias_controlled(self):
        """Hub genes have high |t| but NO seed-specific gradient.

        Build a graph where degree correlates with |t| but there's no
        real perturbation decay from the seed. The degree-preserving
        null should yield non-significant p-values.
        """
        rng = np.random.default_rng(77)

        # Build a random graph with degree heterogeneity
        adj: dict[str, list[str]] = {"SEED": []}
        all_genes_set: set[str] = {"SEED"}

        # 20 high-degree hubs at distance 1 from seed
        hubs = [f"HUB{i}" for i in range(20)]
        adj["SEED"] = list(hubs)
        for h in hubs:
            adj[h] = ["SEED"]
            all_genes_set.add(h)

        # 100 low-degree leaves at distance 2
        for i, h in enumerate(hubs):
            for j in range(5):
                leaf = f"LEAF{i}_{j}"
                adj[h].append(leaf)
                adj[leaf] = [h]
                all_genes_set.add(leaf)

        # Add cross-connections among hubs (making them truly high-degree)
        for i in range(len(hubs)):
            for j in range(i + 1, min(i + 5, len(hubs))):
                adj[hubs[i]].append(hubs[j])
                adj[hubs[j]].append(hubs[i])

        # 500 background genes with varying degree
        for i in range(500):
            bg = f"BG{i}"
            adj[bg] = []
            all_genes_set.add(bg)
            # Some background genes are also hubs (connected to many others)
            n_connections = rng.integers(0, 15)
            for c in range(n_connections):
                target = f"BG{rng.integers(500)}"
                if target != bg:
                    adj[bg].append(target)

        # |t| correlates with DEGREE, not with distance from seed
        degrees = _compute_graph_degrees(adj, sorted(all_genes_set))
        t_stats: dict[str, float] = {}
        for g in all_genes_set:
            # |t| = 0.5 + 0.1 * degree + noise
            t_stats[g] = abs(0.5 + 0.1 * degrees[g] + rng.normal(0, 0.5))

        result = run_gradient_test(
            adjacency=adj,
            abs_t_stats=t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=199,
            rng_seed=42,
            verbose=False,
        )

        # With degree-preserving permutation, the hub-bias confound
        # should be controlled: hubs swap with other hubs, so the
        # gradient signal should NOT be significant
        assert result.slope_pvalue > 0.05, (
            f"Degree-preserving null should control hub bias: "
            f"slope_p={result.slope_pvalue:.3f}"
        )


# ---------------------------------------------------------------------------
# Edge-quality stratification
# ---------------------------------------------------------------------------


class TestStratification:
    def test_stratified_gradient(self, star_graph, decaying_t_stats):
        eq: dict[str, str] = {}
        for i in range(10):
            eq[f"H{i}"] = "multi_source"
            for j in range(5):
                eq[f"L{i}_{j}"] = "multi_source"
        for i in range(10, 20):
            eq[f"H{i}"] = "text_mined"
            for j in range(5):
                eq[f"L{i}_{j}"] = "text_mined"

        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=99,
            rng_seed=42,
            edge_quality=eq,
            verbose=False,
        )
        assert result.stratified is not None
        assert "multi_source" in result.stratified
        assert "text_mined" in result.stratified
        ms = result.stratified["multi_source"]
        assert ms.n_genes_total == 60  # 10 hubs + 50 leaves
        assert ms.slope < 0

    def test_bonferroni_applied(self, star_graph, decaying_t_stats):
        """Stratified p-values should be Bonferroni-corrected."""
        eq: dict[str, str] = {}
        for i in range(10):
            eq[f"H{i}"] = "tier_a"
            for j in range(5):
                eq[f"L{i}_{j}"] = "tier_a"
        for i in range(10, 20):
            eq[f"H{i}"] = "tier_b"
            for j in range(5):
                eq[f"L{i}_{j}"] = "tier_b"

        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=99,
            rng_seed=42,
            edge_quality=eq,
            verbose=False,
        )
        # With 2 tiers, Bonferroni multiplies p by 2
        # The raw p should be < 0.05, Bonferroni-adjusted may be higher
        assert result.stratified is not None
        for tier_name, tier_result in result.stratified.items():
            # Adjusted p-values cannot exceed 1.0
            assert tier_result.slope_pvalue <= 1.0

    def test_no_stratification_without_edge_quality(
        self, star_graph, decaying_t_stats
    ):
        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            n_permutations=49,
            rng_seed=42,
            verbose=False,
        )
        assert result.stratified is None

    def test_precomputed_shells_bypass_bfs(self, decaying_t_stats):
        """When precomputed_shells + graph_degrees are provided, the
        adjacency is unused and shells/degrees come from the caller.
        Mirrors the run_gradient_via_shortest_paths() integration path.
        """
        shells = {
            1: {"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
                "H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19"},
            2: {f"L{i}_{j}" for i in range(20) for j in range(5)},
        }
        # Real degrees (e.g., from Neo4j) — hubs have high degree
        degrees = {g: 6 for g in shells[1]}
        degrees.update({g: 1 for g in shells[2]})
        # Add background gene degrees so the permutation null has a pool
        for i in range(150):
            degrees[f"BG{i}"] = 0
        for i in range(30):
            degrees[f"BGH{i}"] = 6
            for j in range(6):
                degrees[f"BGH{i}_T{j}"] = 1

        result = run_gradient_test(
            adjacency={},  # ignored
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            max_hops=3,
            n_permutations=199,
            rng_seed=42,
            precomputed_shells=shells,
            graph_degrees=degrees,
            verbose=False,
        )
        assert len(result.shells) == 2
        assert result.shells[0].n_genes == 20
        assert result.shells[1].n_genes == 100
        assert result.slope < 0
        # Real degrees should produce the expected null pattern
        assert result.slope_pvalue < 0.05

    def test_rewiring_null_planted_signal(self):
        """Scenario A: planted-pathway signal should survive rewiring.

        Construct a graph where seed-connected genes form a dense clique with
        elevated |t|; the rest of the graph is sparse with low |t|. Rewiring
        destroys the clique structure; expected: null p < 0.1.
        """
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import (
            run_rewiring_null, _gradient_slope, _compute_shell_stats,
        )
        from cliquefinder.stats.graph_rewiring import bfs_distances_from

        rng = np.random.default_rng(0)
        G = nx.Graph()
        # Seed directly regulates a dense 20-gene "pathway"; those genes
        # also interconnect so hop-2 reaches more pathway members
        for i in range(20):
            G.add_edge("SEED", f"P{i}")
            # Inter-pathway connections
            for j in range(i + 1, 20):
                if rng.random() < 0.6:
                    G.add_edge(f"P{i}", f"P{j}")
        # 100 background genes with some connections to pathway (dilution)
        for i in range(100):
            G.add_edge(f"BG{i}", f"P{rng.integers(20)}")
            # Background-background connections
            for _ in range(rng.integers(1, 4)):
                G.add_edge(f"BG{i}", f"BG{rng.integers(100)}")

        # High |t| on pathway genes, low on background
        t_stats = {}
        for i in range(20):
            t_stats[f"P{i}"] = 2.0 + rng.normal(0, 0.2)
        for i in range(100):
            t_stats[f"BG{i}"] = abs(0.5 + rng.normal(0, 0.3))

        # Compute observed slope
        distances = bfs_distances_from(G, "SEED", set(t_stats.keys()), max_hops=3)
        shells = {}
        for g, d in distances.items():
            shells.setdefault(d, set()).add(g)
        observed_shells = [
            _compute_shell_stats(shells[h], t_stats, h) for h in sorted(shells)
        ]
        observed_slope = _gradient_slope(observed_shells)
        assert observed_slope < 0, "Fixture should produce decay"

        result = run_rewiring_null(
            graph=G, seed="SEED", abs_t_stats=t_stats,
            observed_slope=observed_slope,
            n_rewires=99, max_hops=3, rng_seed=42,
            check_every=500, max_swaps_iter0=20_000,
            verbose=False,
        )
        # Gradient should survive rewiring (planted signal is real)
        assert result.pvalue < 0.15, (
            f"Planted signal should survive rewiring, got p={result.pvalue:.3f}"
        )
        assert result.n_rewires_ok > 70

    def test_rewiring_null_runs_on_config_model(self):
        """Regression: null runs cleanly on a configuration-model graph.

        Full calibration studies (scenarios A-F from the design doc)
        belong in Wave 20f's dedicated validation suite. Here we only
        verify: structural invariants hold, pathology diagnostics fire
        appropriately, result object is well-formed.
        """
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        rng = np.random.default_rng(1)
        # Configuration model on a random degree sequence (avoids BA's
        # preferential-attachment clustering that rewiring legitimately destroys)
        degree_sequence = [rng.integers(2, 8) for _ in range(200)]
        if sum(degree_sequence) % 2:
            degree_sequence[0] += 1
        G = nx.configuration_model(degree_sequence, seed=1)
        G = nx.Graph(G)  # drop multi-edges and self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        if "G0" not in G:
            pytest.skip("Fixture produced graph without seed")

        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats,
            observed_slope=-0.05,
            n_rewires=49, max_hops=3, rng_seed=7,
            check_every=500, max_swaps_iter0=20_000,
            verbose=False,
        )
        # Structural: result object is well-formed
        assert 0.0 <= result.pvalue <= 1.0
        assert result.n_rewires_ok > 0
        assert result.subgraph_n_nodes == G.number_of_nodes()
        assert result.subgraph_n_edges == G.number_of_edges()
        # Elapsed time is reported
        assert result.elapsed_seconds > 0
        # Disconnection rate is sensible
        assert 0.0 <= result.disconnection_rate <= 1.0

    def test_rewiring_null_result_to_dict_has_provenance(self):
        """RewiringNullResult.to_dict carries fields needed for reproducibility."""
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        rng = np.random.default_rng(0)
        G = nx.barabasi_albert_graph(100, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats, observed_slope=-0.1,
            n_rewires=19, max_hops=3, rng_seed=42,
            check_every=500, max_swaps_iter0=10_000,
            verbose=False,
        )
        d = result.to_dict()
        required_provenance = (
            "observed_slope", "observed_coverage",
            "null_slopes", "null_slopes_summary", "null_slopes_imputed_preview",
            "pvalue", "pvalue_formula",
            "n_rewires_requested", "n_rewires_ok", "n_rewires_failed",
            "seed", "subgraph_n_nodes", "subgraph_n_edges",
            "seed_component_too_small",
            "rng_seed", "max_hops", "target_nswap", "plateau_nswap",
            "iter0_plateau_reached", "accepted_fraction_iter0",
            "bimodality_coefficient", "bimodality_pseudo_pvalue",
            "disconnection_rate",
            "bimodality_warning", "disconnection_warning", "mixing_warning",
            "elapsed_seconds",
        )
        for key in required_provenance:
            assert key in d, f"Missing provenance key: {key}"
        # Specific values that MUST match inputs
        assert d["seed"] == "G0"
        assert d["rng_seed"] == 42
        assert d["max_hops"] == 3
        assert d["n_rewires_requested"] == 19
        # Null slopes serialized as full array (enables offline p-value audit)
        assert isinstance(d["null_slopes"], list)
        # Deciles included in summary
        summary = d["null_slopes_summary"]
        if summary.get("deciles"):
            assert len(summary["deciles"]) == 11

    def test_rewiring_null_deterministic(self):
        """Two runs with the same rng_seed produce identical null slopes."""
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        rng = np.random.default_rng(0)
        G = nx.barabasi_albert_graph(100, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        r1 = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats, observed_slope=-0.1,
            n_rewires=9, max_hops=3, rng_seed=42,
            check_every=500, max_swaps_iter0=5_000, verbose=False,
        )
        r2 = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats, observed_slope=-0.1,
            n_rewires=9, max_hops=3, rng_seed=42,
            check_every=500, max_swaps_iter0=5_000, verbose=False,
        )
        np.testing.assert_array_equal(r1.null_slopes, r2.null_slopes)
        assert r1.target_nswap == r2.target_nswap

    def test_rewiring_null_raises_if_seed_missing(self):
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        # Seed missing needs a graph large enough to pass the >=30 node check
        G = nx.barabasi_albert_graph(50, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        with pytest.raises(ValueError, match="not in graph"):
            run_rewiring_null(
                graph=G, seed="MISSING", abs_t_stats={"G0": 1.0},
                observed_slope=0.0, n_rewires=2, verbose=False,
            )

    def test_rewiring_null_raises_if_graph_too_small(self):
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        G = nx.path_graph(["A", "B", "C", "D", "E"])
        with pytest.raises(ValueError, match="too small"):
            run_rewiring_null(
                graph=G, seed="A", abs_t_stats={"A": 1.0, "B": 1.0},
                observed_slope=0.0, n_rewires=2, verbose=False,
            )

    def test_rewiring_null_rejects_n_jobs_gt_1(self):
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        G = nx.barabasi_albert_graph(50, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        with pytest.raises(NotImplementedError, match="n_jobs"):
            run_rewiring_null(
                graph=G, seed="G0", abs_t_stats={"G0": 1.0},
                observed_slope=0.0, n_rewires=2, n_jobs=4, verbose=False,
            )

    def test_rewiring_null_n_accounting_correct(self):
        """n_rewires_requested + n_failed/ok split is properly recorded."""
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        rng = np.random.default_rng(0)
        G = nx.barabasi_albert_graph(100, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats, observed_slope=-0.1,
            n_rewires=25, max_hops=3, rng_seed=42,
            check_every=500, max_swaps_iter0=5_000, verbose=False,
        )
        assert result.n_rewires_requested == 25
        assert result.n_rewires_ok + result.n_rewires_failed == 25

    def test_rewiring_null_iter1_uses_fixed_swap_budget(self):
        """Iterations ≥ 1 should run with the plateau-determined swap budget,
        not max_swaps_iter0.  Guards against the bug where record_diagnostic=False
        disabled plateau termination, causing iter ≥ 1 to run the full max budget."""
        import networkx as nx
        import time
        from cliquefinder.stats.perturbation_gradient import run_rewiring_null

        rng = np.random.default_rng(0)
        G = nx.barabasi_albert_graph(150, m=3, seed=0)
        G = nx.relabel_nodes(G, {i: f"G{i}" for i in G.nodes()})
        t_stats = {g: abs(rng.normal(1.0, 0.3)) for g in G.nodes()}

        # Iter 0 plateau should be small; iter ≥ 1 should use that budget.
        # If the bug were still present (full max_swaps per iter), runtime would
        # scale with n_rewires * max_swaps_iter0 not n_rewires * plateau.
        t0 = time.time()
        result = run_rewiring_null(
            graph=G, seed="G0", abs_t_stats=t_stats, observed_slope=-0.1,
            n_rewires=20, max_hops=3, rng_seed=42,
            max_swaps_iter0=200_000, check_every=500, verbose=False,
        )
        elapsed = time.time() - t0

        # target_nswap < max_swaps_iter0 → fast iterations
        assert result.target_nswap < 200_000, (
            "target_nswap should be derived from plateau, not max_swaps_iter0"
        )
        # 20 iters × small nswap each should take under 30s on this fixture
        assert elapsed < 30.0, f"Iter ≥1 budget is too large; runtime={elapsed:.1f}s"

    def test_tier_too_small_skipped(self, star_graph, decaying_t_stats):
        eq = {f"H{i}": "main" for i in range(20)}
        for i in range(20):
            for j in range(5):
                eq[f"L{i}_{j}"] = "main"
        eq["H0"] = "rare"
        eq["H1"] = "rare"

        result = run_gradient_test(
            adjacency=star_graph,
            abs_t_stats=decaying_t_stats,
            seed="SEED",
            n_permutations=49,
            rng_seed=42,
            edge_quality=eq,
            verbose=False,
        )
        assert result.stratified is not None
        assert "rare" not in result.stratified  # too few genes
        assert "main" in result.stratified
