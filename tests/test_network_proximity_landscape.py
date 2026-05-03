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
    """Cypher binding + parameter behavior verified via fake _execute_query
    that captures the query/params and returns synthetic rows.
    """

    def test_pins_regulatory_scope_in_cypher_params(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured["query"] = query
            captured["params"] = params
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A", "B", "C"],
            max_hops=2,
            min_evidence=1,
        )
        assert "stmt_types" in captured["params"]
        assert set(captured["params"]["stmt_types"]) == ALL_REGULATORY_TYPES

    def test_seed_names_passed_as_list(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A", "B"],
            max_hops=2,
        )
        assert captured["seed_names"] == ["A", "B"]

    def test_uses_apoc_subgraph_nodes_for_traversal(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured["query"] = query
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        assert "apoc.path.subgraphNodes" in captured["query"]
        assert "indra_rel" in captured["query"]
        assert "r.stmt_type IN $stmt_types" in captured["query"]

    def test_cypher_has_server_side_limit(self):
        """KG-1/KG-2 pattern: every extraction Cypher must have LIMIT."""
        captured: dict = {}

        def fake_execute(query, **params):
            captured["query"] = query
            captured["params"] = params
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
        )
        assert "LIMIT $max_edges" in captured["query"]
        assert captured["params"]["max_edges"] == 5_000_000  # default

    def test_max_edges_per_batch_overridable(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=2,
            max_edges_per_batch=1000,
        )
        assert captured["max_edges"] == 1000

    def test_max_hops_inlined_in_cypher(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured["query"] = query
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], max_hops=3,
        )
        assert "maxLevel: 3" in captured["query"]

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

    def test_min_evidence_passed_as_param(self):
        captured: dict = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_subgraph_induced_by_features(
            cogex_client=client, features=["A"], min_evidence=5,
        )
        assert captured["min_evidence"] == 5

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

    def test_returns_edges_and_matched_features(self):
        client = MagicMock()
        client._execute_query = MagicMock(return_value=[
            ["A", "B", 5, "Activation", ["A", "B"]],
            ["B", "C", 2, "Inhibition", ["A", "B"]],
        ])
        edges, matched = extract_subgraph_induced_by_features(
            cogex_client=client, features=["A", "B", "C"], max_hops=2,
        )
        assert len(edges) == 2
        assert ("A", "B", {"evidence_count": 5, "stmt_type": "Activation"}) in edges
        assert ("B", "C", {"evidence_count": 2, "stmt_type": "Inhibition"}) in edges
        # Cypher reported A and B as matched; C did not match a BioEntity.
        assert matched == {"A", "B"}

    def test_unmatched_features_distinguishable(self):
        """A feature that doesn't resolve to a BioEntity must NOT be in
        matched_features — distinguishable from a matched-but-isolated one.
        """
        client = MagicMock()
        client._execute_query = MagicMock(return_value=[
            ["A", "B", 1, "Activation", ["A"]],  # B was reachable but not a seed match
        ])
        edges, matched = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["A", "TYPO_NOT_IN_INDRA"],
            max_hops=2,
        )
        assert "A" in matched
        assert "TYPO_NOT_IN_INDRA" not in matched

    def test_dedupes_edges_across_batches(self):
        """Two batches that both surface the same edge collapse it."""
        edge_attr_row = ["X", "Y", 3, "Activation", ["X", "Y"]]
        client = MagicMock()
        client._execute_query = MagicMock(return_value=[edge_attr_row])
        edges, _ = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["X", "Y", "Z"],
            seed_batch_size=2,  # forces 2 batches
            max_hops=2,
        )
        # Even though the fake execute returns the same row twice (once
        # per batch), the dedup keeps one entry.
        assert len(edges) == 1
        assert edges[0] == ("X", "Y", {"evidence_count": 3, "stmt_type": "Activation"})

    def test_dedup_keeps_max_evidence_count(self):
        """If two batches report the same edge with different evidence,
        keep the larger count (deterministic).
        """
        client = MagicMock()
        # Batch 1 returns evidence_count=2; batch 2 returns 5.
        client._execute_query = MagicMock(side_effect=[
            [["X", "Y", 2, "Activation", ["X", "Y"]]],
            [["X", "Y", 5, "Activation", ["X", "Y"]]],
        ])
        edges, _ = extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["X", "Y", "Z"],
            seed_batch_size=2,  # forces 2 batches
            max_hops=2,
        )
        assert len(edges) == 1
        assert edges[0][2]["evidence_count"] == 5

    def test_warns_when_limit_hit(self):
        """When a batch returns exactly max_edges_per_batch rows, warn."""
        client = MagicMock()
        rows = [
            [f"S{i}", f"T{i}", 1, "Activation", [f"S{i}"]]
            for i in range(10)
        ]
        client._execute_query = MagicMock(return_value=rows)
        with pytest.warns(RuntimeWarning, match="max_edges_per_batch"):
            extract_subgraph_induced_by_features(
                cogex_client=client,
                features=[f"S{i}" for i in range(10)],
                max_hops=2,
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
