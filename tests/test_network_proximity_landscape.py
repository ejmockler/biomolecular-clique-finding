"""Landscape-scale extraction primitives in network_proximity.py.

Covers extract_subgraph_induced_by_features (Cypher-shape verification
via mocked Neo4j client) and compute_all_pairs_shortest_paths_bounded
(pure-Python BFS correctness, edge cases, target_filter).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES
from cliquefinder.stats.network_proximity import (
    compute_all_pairs_shortest_paths_bounded,
    extract_subgraph_induced_by_features,
)


# --- extract_subgraph_induced_by_features ----------------------------------


class TestExtractSubgraphCypherShape:
    """Cypher shape + parameter behavior verified via fake _execute_query
    that captures every query/params and returns synthetic rows.

    The implementation is iterative regulatory-only BFS (no APOC):
    one query per hop for frontier expansion, then one query per
    batch for final edge extraction.
    """

    def test_pins_regulatory_scope_in_edge_queries(self):
        """Every Cypher that touches edges must bind ALL_REGULATORY_TYPES.
        The Phase-0 BioEntity-existence probe is exempt — it queries
        nodes only, not edges.
        """
        captured: list[dict] = []

        def fake_execute(query, **params):
            captured.append({"query": query, "params": params})
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A", "B"], max_hops=2,
        )
        # Every edge-touching Cypher call must bind the scope.
        edge_calls = [c for c in captured if "indra_rel" in c["query"]]
        assert edge_calls, "expected at least one edge-touching call"
        for call in edge_calls:
            assert "stmt_types" in call["params"]
            assert set(call["params"]["stmt_types"]) == ALL_REGULATORY_TYPES

    def test_does_not_use_apoc(self):
        """APOC subgraphNodes is unaware of edge properties and walked
        every indra_rel edge — OOMed Neo4j at 200 seeds.  The new
        implementation must not regress to that pattern.
        """
        captured: list[str] = []

        def fake_execute(query, **params):
            captured.append(query)
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        for q in captured:
            assert "apoc" not in q.lower(), (
                "APOC reintroduced — see Wave 24g: APOC traversal "
                "ignores stmt_type and OOMs at proteome scale"
            )

    def test_edge_queries_filter_by_stmt_type_inline(self):
        """Edge-touching queries (expansion + extraction) must have
        the inline regulatory filter.  Phase-0 probe is exempt.
        """
        captured: list[str] = []

        def fake_execute(query, **params):
            captured.append(query)
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        edge_qs = [q for q in captured if "indra_rel" in q]
        assert edge_qs
        for q in edge_qs:
            assert "r.stmt_type IN $stmt_types" in q

    def test_edge_queries_have_server_side_limit(self):
        """Edge-touching queries must carry LIMIT $max_edges (KG-1/KG-2)."""
        captured: list[dict] = []

        def fake_execute(query, **params):
            captured.append({"query": query, "params": params})
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        edge_calls = [c for c in captured if "indra_rel" in c["query"]]
        for call in edge_calls:
            assert "LIMIT $max_edges" in call["query"]
            assert call["params"]["max_edges"] == 5_000_000

    def test_max_edges_per_batch_overridable(self):
        captured: list[dict] = []

        def fake_execute(query, **params):
            captured.append({"query": query, "params": params})
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
            max_edges_per_batch=1000,
        )
        edge_calls = [c for c in captured if "indra_rel" in c["query"]]
        for call in edge_calls:
            assert call["params"]["max_edges"] == 1000

    def test_iterative_bfs_issues_probe_plus_expansion_plus_extraction(self):
        """For an isolated seed (no edges), expansion stops after hop 1
        and we still issue the final extraction query.  Plus the
        Phase-0 BioEntity-existence probe.
        """
        call_log: list[dict] = []

        def fake_execute(query, **params):
            call_log.append({"query": query, "batch": list(params.get("batch", []))})
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        # 1 BioEntity probe + 1 hop-1 expansion (returns nothing so
        # no hop 2) + 1 edge extraction = 3 queries.
        assert len(call_log) == 3

    def test_max_hops_validated(self):
        client = MagicMock()
        client._execute_query = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            extract_subgraph_induced_by_features(
                cogex_client=client, features=["A"], max_hops=0,
            )
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            extract_subgraph_induced_by_features(
                cogex_client=client, features=["A"], max_hops=-1,
            )

    def test_seed_batch_size_validated(self):
        client = MagicMock()
        client._execute_query = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="seed_batch_size must be >= 1"):
            extract_subgraph_induced_by_features(
                cogex_client=client, features=["A"], seed_batch_size=0,
            )

    def test_min_evidence_passed_to_edge_queries(self):
        captured: list[dict] = []

        def fake_execute(query, **params):
            captured.append({"query": query, "params": params})
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], min_evidence=5,
        )
        edge_calls = [c for c in captured if "indra_rel" in c["query"]]
        for call in edge_calls:
            assert call["params"]["min_evidence"] == 5

    def test_empty_features_returns_empty_without_query(self):
        client = MagicMock()
        client._execute_query = MagicMock(side_effect=AssertionError(
            "must not call _execute_query for empty features",
        ))
        edges, matched = extract_subgraph_induced_by_features(
            cogex_client=client, features=[], max_hops=2,
        )
        assert edges == []
        assert matched == set()

    def test_matched_features_from_bioentity_probe_not_edges(self):
        """matched_features should reflect BioEntity existence, NOT
        whether the entity has regulatory edges.  A matched-but-isolated
        feature must be in matched_features.
        """
        call_idx = [0]
        # Probe returns A and C (both exist as BioEntity nodes).
        # Expansion: A has an edge A→B; C has no edges.
        # No hop 2 (frontier = {B}, B has no edges).
        # Extraction: A→B edge.
        responses = [
            [["A"], ["C"]],          # Phase 0 probe: A and C resolve
            [["A", "B"]],            # Phase 1 hop 1: A→B
            [],                       # Phase 1 hop 2: from B → nothing
            [["A", "B", 5, "Activation"]],  # Phase 2 extraction
        ]

        def fake_execute(q, **kw):
            i = call_idx[0]
            call_idx[0] += 1
            return responses[i] if i < len(responses) else []

        client = MagicMock()
        client._execute_query = fake_execute
        edges, matched = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A", "C"],
            max_hops=2,
            seed_batch_size=10,
        )
        # Both A and C exist as BioEntity nodes → both matched, even
        # though C has no regulatory edges.
        assert matched == {"A", "C"}
        assert ("A", "B", {"evidence_count": 5, "stmt_type": "Activation"}) in edges

    def test_unmatched_feature_truly_absent_from_matched(self):
        """A feature whose name doesn't resolve to a BioEntity must
        NOT be in matched_features.
        """
        call_idx = [0]
        responses = [
            [["A"]],                 # Probe: only A resolves; TYPO does not
            [],                       # No edges from A
            [],                       # Extraction: nothing
        ]

        def fake_execute(q, **kw):
            i = call_idx[0]
            call_idx[0] += 1
            return responses[i] if i < len(responses) else []

        client = MagicMock()
        client._execute_query = fake_execute
        _, matched = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A", "TYPO_NOT_IN_INDRA"],
            max_hops=2,
        )
        assert matched == {"A"}

    def test_edge_extraction_post_filters_to_collected_nodes(self):
        """Final edge extraction must drop edges whose target is NOT
        in the collected node set (those edges leave the subgraph).
        """
        responses = iter([
            # Phase 0 probe: A resolves.
            [["A"]],
            # Hop 1 expansion: A→B.
            [["A", "B"]],
            # Hop 2 expansion: B has no new neighbors.
            [],
            # Edge extraction over {A, B}: returns A→B (in set) AND
            # B→Z (Z is OUTSIDE the collected node set).
            [
                ["A", "B", 1, "Activation"],
                ["B", "Z", 1, "Inhibition"],
            ],
        ])
        client = MagicMock()
        client._execute_query = MagicMock(side_effect=lambda q, **kw: next(responses))
        edges, _ = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A"],
            max_hops=2,
            seed_batch_size=10,
        )
        edge_pairs = {(s, t) for s, t, _ in edges}
        assert ("A", "B") in edge_pairs
        assert ("B", "Z") not in edge_pairs  # Z not in nodes_seen

    def test_raises_when_limit_hit_during_expansion(self):
        """A truncated subgraph would silently corrupt downstream
        analysis — refuse to continue rather than ship a broken result.
        """
        responses = iter([
            # Phase 0 probe: S resolves.
            [["S"]],
            # Expansion query returns exactly max_edges rows → limit hit.
            [[f"S", f"T{i}"] for i in range(10)],
        ])
        client = MagicMock()
        client._execute_query = MagicMock(side_effect=lambda q, **kw: next(responses))
        with pytest.raises(RuntimeError, match="frontier expansion"):
            extract_subgraph_induced_by_features(
                cogex_client=client,
                features=["S"],
                max_hops=1,
                max_edges_per_batch=10,
            )

    def test_raises_when_limit_hit_during_edge_extraction(self):
        responses = iter([
            # Phase 0: S resolves.
            [["S"]],
            # Phase 1: S has 1 edge to T (1 row, not at limit).
            [["S", "T"]],
            # Phase 2 extraction returns max_edges rows.
            [[f"S{i}", "X", 1, "Activation"] for i in range(10)],
        ])
        client = MagicMock()
        client._execute_query = MagicMock(side_effect=lambda q, **kw: next(responses))
        with pytest.raises(RuntimeError, match="edge extraction"):
            extract_subgraph_induced_by_features(
                cogex_client=client,
                features=["S"],
                max_hops=1,
                max_edges_per_batch=10,
            )


class TestRestrictEndpointsToFeatures:
    """Wave 24l: fast-path extraction skips frontier expansion when
    paths can only route through measured proteins."""

    def test_no_frontier_expansion_queries(self):
        """Under restrict_endpoints_to_features=True, no Phase-1
        expansion queries (which use undirected ``-[r:indra_rel]-``)
        are issued — only the BioEntity probe + Phase-2 measured-pair
        extraction (directed ``->``)."""
        captured: list[str] = []

        def fake_execute(query, **params):
            captured.append(query)
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A", "B"],
            max_hops=2, restrict_endpoints_to_features=True,
        )
        edge_qs = [q for q in captured if "indra_rel" in q]
        # No undirected expansion query should appear.
        for q in edge_qs:
            assert "-[r:indra_rel]-" not in q or "->" in q, (
                "frontier expansion query (undirected) leaked into "
                "measured-only extraction path"
            )

    def test_measured_target_filter_present_in_cypher(self):
        """The fast-path Cypher must filter targets to the measured
        feature set server-side via ``b.name IN $feature_set``."""
        captured: list[dict] = []

        def fake_execute(query, **params):
            captured.append({"query": query, "params": params})
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A", "B", "C"],
            max_hops=2, restrict_endpoints_to_features=True,
        )
        edge_calls = [c for c in captured if "indra_rel" in c["query"]]
        assert edge_calls, "no edge-touching queries issued"
        for call in edge_calls:
            assert "b.name IN $feature_set" in call["query"]
            assert call["params"]["feature_set"] == ["A", "B", "C"]

    def test_returns_only_measured_pair_edges(self):
        """If Neo4j returns an edge with an unmeasured endpoint (which
        the Cypher filter should prevent — but verify defensively),
        the returned edge list is whatever Cypher returned.  Under
        the fast path, the Cypher's b.name IN $feature_set guards
        this; this test verifies the function trusts the Cypher.
        """
        rows = [
            ("A", "B", 5, "Activation"),
            ("B", "C", 3, "Inhibition"),
        ]
        client = MagicMock()
        # Phase-0 probe returns measured set; Phase-2 returns rows.
        responses = iter([
            [("A",), ("B",), ("C",)],  # BioEntity probe
            rows,                       # measured-pair edge query
        ])
        client._execute_query = MagicMock(
            side_effect=lambda q, **kw: next(responses)
        )
        edges, matched = extract_subgraph_induced_by_features(
            cogex_client=client, features=["A", "B", "C"],
            max_hops=2, restrict_endpoints_to_features=True,
        )
        assert matched == {"A", "B", "C"}
        assert {(s, t) for s, t, _ in edges} == {("A", "B"), ("B", "C")}


# --- compute_all_pairs_shortest_paths_bounded ------------------------------


def _attrs() -> dict:
    return {"evidence_count": 1, "stmt_type": "Activation"}


class TestBFSCorrectness:
    def test_simple_chain_distances(self):
        """A — B — C — D, sources from each, h=3."""
        edges = [
            ("A", "B", _attrs()),
            ("B", "C", _attrs()),
            ("C", "D", _attrs()),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges,
            source_nodes=["A", "B", "C", "D"],
            max_hops=3,
        )
        assert distances["A"] == {"A": 0, "B": 1, "C": 2, "D": 3}
        assert distances["B"] == {"B": 0, "A": 1, "C": 1, "D": 2}
        assert distances["C"] == {"C": 0, "B": 1, "D": 1, "A": 2}
        assert distances["D"] == {"D": 0, "C": 1, "B": 2, "A": 3}

    def test_max_hops_bound_truncates_distances(self):
        edges = [
            ("A", "B", _attrs()),
            ("B", "C", _attrs()),
            ("C", "D", _attrs()),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
        )
        # D is at distance 3 from A; should NOT appear at max_hops=2.
        assert distances["A"] == {"A": 0, "B": 1, "C": 2}

    def test_undirected_treatment(self):
        """Edge stored as (A, B) → BFS from B must reach A at dist 1."""
        edges = [("A", "B", _attrs())]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["B"], max_hops=1,
        )
        assert distances["B"] == {"B": 0, "A": 1}

    def test_disconnected_components(self):
        """Two independent components; BFS in each respects boundaries."""
        edges = [
            ("A", "B", _attrs()),  # component 1
            ("C", "D", _attrs()),  # component 2
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A", "C"], max_hops=2,
        )
        assert distances["A"] == {"A": 0, "B": 1}
        assert distances["C"] == {"C": 0, "D": 1}

    def test_source_with_no_edges(self):
        edges = [("A", "B", _attrs())]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A", "Z"], max_hops=2,
        )
        assert distances["A"] == {"A": 0, "B": 1}
        # Z has no edges; only itself reachable.
        assert distances["Z"] == {"Z": 0}

    def test_self_loops_dropped(self):
        """An A → A edge is meaningless undirected and should not affect BFS."""
        edges = [
            ("A", "A", _attrs()),
            ("A", "B", _attrs()),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
        )
        assert distances["A"] == {"A": 0, "B": 1}

    def test_duplicate_edges_dropped(self):
        """Multiple edges (a, b) with different attrs collapse to one
        adjacency entry — BFS distance is unaffected.
        """
        edges = [
            ("A", "B", {"evidence_count": 1, "stmt_type": "Activation"}),
            ("A", "B", {"evidence_count": 5, "stmt_type": "Inhibition"}),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=1,
        )
        assert distances["A"] == {"A": 0, "B": 1}

    def test_diamond_shortest_path_chosen(self):
        """A → B → D and A → C → D both length 2; BFS picks min."""
        edges = [
            ("A", "B", _attrs()), ("B", "D", _attrs()),
            ("A", "C", _attrs()), ("C", "D", _attrs()),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=3,
        )
        assert distances["A"]["D"] == 2


class TestTargetFilter:
    def test_filter_restricts_output_targets(self):
        edges = [
            ("A", "B", _attrs()),
            ("B", "C", _attrs()),
            ("C", "D", _attrs()),
        ]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=3,
            target_filter={"C", "D"},  # only keep distances to these
        )
        # Source itself is always included; B is filtered out.
        assert distances["A"] == {"A": 0, "C": 2, "D": 3}

    def test_empty_filter_keeps_only_source(self):
        edges = [("A", "B", _attrs())]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            target_filter=set(),
        )
        # Only source remains.
        assert distances["A"] == {"A": 0}

    def test_filter_does_not_alter_disconnected_source(self):
        edges = [("A", "B", _attrs())]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["Z"], max_hops=2,
            target_filter={"A", "B"},
        )
        # Z is not in the filter and has no edges — empty result.
        assert distances["Z"] == {}


class TestErrors:
    def test_zero_max_hops_rejected(self):
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            compute_all_pairs_shortest_paths_bounded(
                edges=[], source_nodes=["A"], max_hops=0,
            )

    def test_negative_max_hops_rejected(self):
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            compute_all_pairs_shortest_paths_bounded(
                edges=[], source_nodes=["A"], max_hops=-1,
            )

    def test_empty_sources_returns_empty_dict(self):
        edges = [("A", "B", _attrs())]
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=[], max_hops=2,
        )
        assert distances == {}

    def test_empty_edges_with_sources(self):
        distances = compute_all_pairs_shortest_paths_bounded(
            edges=[], source_nodes=["A", "B"], max_hops=2,
        )
        assert distances == {"A": {"A": 0}, "B": {"B": 0}}


class TestNodeFilterMeasuredOnlyPaths:
    """Wave 24l: BFS must NOT route through unmeasured intermediates.

    The ``node_filter`` parameter restricts adjacency to edges where
    both endpoints are in the filter — preventing paths from being
    completed via an unmeasured intermediate at any hop.
    """

    def test_blocks_unmeasured_intermediate_at_hop2(self):
        """A — U — C where U is unmeasured: with node_filter, hop-2
        unreachable; without, hop-2 = C."""
        edges = [
            ("A", "U", _attrs()),
            ("U", "C", _attrs()),
        ]
        # Baseline: U participates, C reachable at hop 2.
        baseline = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
        )
        assert baseline["A"] == {"A": 0, "U": 1, "C": 2}

        # With node_filter excluding U: C unreachable.
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            node_filter={"A", "C"},
        )
        assert restricted["A"] == {"A": 0}

    def test_allows_path_through_measured_intermediate(self):
        """A — M — C where M IS measured: hop-2 = C in both modes."""
        edges = [
            ("A", "M", _attrs()),
            ("M", "C", _attrs()),
        ]
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            node_filter={"A", "M", "C"},
        )
        assert restricted["A"] == {"A": 0, "M": 1, "C": 2}

    def test_unmeasured_neighbor_dropped_even_at_hop1(self):
        """A — U where U is unmeasured: not in adjacency at all."""
        edges = [("A", "U", _attrs()), ("A", "M", _attrs())]
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            node_filter={"A", "M"},
        )
        assert restricted["A"] == {"A": 0, "M": 1}

    def test_mixed_path_only_measured_route_survives(self):
        """A reaches D via A-M1-M2-D (all measured) AND via A-U-D
        (unmeasured U).  Measured-only path is length 3; mixed path is
        length 2.  With node_filter, distance is 3 (measured path only)."""
        edges = [
            ("A", "M1", _attrs()),
            ("M1", "M2", _attrs()),
            ("M2", "D", _attrs()),
            ("A", "U", _attrs()),
            ("U", "D", _attrs()),
        ]
        baseline = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=3,
        )
        # Mixed path A-U-D = 2; chosen as min.
        assert baseline["A"]["D"] == 2

        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=3,
            node_filter={"A", "M1", "M2", "D"},
        )
        assert restricted["A"]["D"] == 3  # forced through measured chain

    def test_source_outside_filter_yields_only_self(self):
        """If the source itself isn't in node_filter, its adjacency
        is empty after filtering — only itself reachable."""
        edges = [("A", "B", _attrs())]
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            node_filter={"B"},  # A excluded
        )
        # A is the source so it appears at distance 0; B unreachable
        # because A has no surviving edges.
        assert restricted["A"] == {"A": 0}

    def test_measured_source_with_only_unmeasured_neighbors(self):
        """Measured source whose only neighbors are unmeasured —
        common in production for sparsely-measured regulators.  All
        edges drop after node_filter; source yields only {source: 0}.
        Branch covered: ``source not in adjacency`` after filtering.
        """
        edges = [
            ("A", "U1", _attrs()),
            ("A", "U2", _attrs()),
            ("U1", "U2", _attrs()),
        ]
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=2,
            node_filter={"A"},
        )
        assert restricted["A"] == {"A": 0}

    def test_target_filter_and_node_filter_compose(self):
        """node_filter restricts traversal; target_filter restricts
        the returned dict.  Use both together."""
        edges = [
            ("A", "M1", _attrs()),
            ("M1", "M2", _attrs()),
            ("M2", "M3", _attrs()),
        ]
        restricted = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=3,
            node_filter={"A", "M1", "M2", "M3"},
            target_filter={"M2"},  # only keep distance to M2
        )
        # Source always included; only M2 retained from targets.
        assert restricted["A"] == {"A": 0, "M2": 2}


class TestUnboundedBFS:
    """Wave 24l: max_hops=None lets BFS run to connected-component
    completion (anchor-adaptive depth)."""

    def test_reaches_cc_completion(self):
        """A — B — C — D — E chain: with max_hops=None, all are
        reachable from A at their true graph distances."""
        edges = [
            ("A", "B", _attrs()),
            ("B", "C", _attrs()),
            ("C", "D", _attrs()),
            ("D", "E", _attrs()),
        ]
        result = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=None,
        )
        assert result["A"] == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def test_unbounded_stops_at_cc_boundary(self):
        """Two disconnected components: BFS terminates at the boundary
        of source's CC, not at the other."""
        edges = [
            ("A", "B", _attrs()),
            ("B", "C", _attrs()),
            ("X", "Y", _attrs()),
        ]
        result = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=None,
        )
        assert result["A"] == {"A": 0, "B": 1, "C": 2}
        assert "X" not in result["A"]
        assert "Y" not in result["A"]

    def test_unbounded_with_node_filter(self):
        """node_filter still gates traversal under max_hops=None;
        BFS goes to CC-completion within the filtered subgraph."""
        edges = [
            ("A", "M1", _attrs()),
            ("M1", "M2", _attrs()),
            ("M2", "U", _attrs()),    # blocked: U unmeasured
            ("U", "M3", _attrs()),    # not reachable via measured-only
            ("M2", "M3", _attrs()),   # reachable: A-M1-M2-M3
        ]
        result = compute_all_pairs_shortest_paths_bounded(
            edges=edges, source_nodes=["A"], max_hops=None,
            node_filter={"A", "M1", "M2", "M3"},
        )
        assert result["A"] == {"A": 0, "M1": 1, "M2": 2, "M3": 3}

    def test_zero_max_hops_still_rejected(self):
        """max_hops=0 is invalid (negative-depth nonsense)."""
        with pytest.raises(ValueError, match="max_hops must be >= 1 or None"):
            compute_all_pairs_shortest_paths_bounded(
                edges=[], source_nodes=["A"], max_hops=0,
            )

    def test_unbounded_isolated_source(self):
        """Source not in any edge: only itself at distance 0."""
        result = compute_all_pairs_shortest_paths_bounded(
            edges=[("A", "B", _attrs())],
            source_nodes=["Z"], max_hops=None,
        )
        assert result["Z"] == {"Z": 0}
