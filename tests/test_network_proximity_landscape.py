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
