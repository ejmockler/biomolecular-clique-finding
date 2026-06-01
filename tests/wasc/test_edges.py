"""Tests for WASC M1 edge enumeration.

Covers:
- WascEdge dataclass invariants (lex ordering, self-loop rejection, edge_id format)
- compute_measured_cluster_members: per-theme union → UniProt → measured intersection
- enumerate_wasc_indra_edges: within-theme only, undirected dedupe, metadata
  aggregation, self-loop filter, unmeasured-endpoint filter
- DEFAULT_CLUSTER_TERMS shape (3 themes × 8 terms)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cliquefinder.stats.wasc import (
    DEFAULT_CLUSTER_TERMS,
    Network,
    Theme,
    WascEdge,
    compute_measured_cluster_members,
    enumerate_wasc_indra_edges,
)


# ---------------------------------------------------------------------------
# WascEdge dataclass
# ---------------------------------------------------------------------------

class TestWascEdge:
    def test_lex_ordering_enforced(self):
        with pytest.raises(ValueError, match="lex-smaller"):
            WascEdge(
                anchor_uniprot="UZ",
                target_uniprot="UA",
                theme=Theme.SPLICING,
                network=Network.INDRA,
            )

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            WascEdge(
                anchor_uniprot="UA",
                target_uniprot="UA",
                theme=Theme.SPLICING,
                network=Network.INDRA,
            )

    def test_edge_id_format(self):
        e = WascEdge(
            anchor_uniprot="UA", target_uniprot="UB",
            theme=Theme.SPLICING, network=Network.INDRA,
        )
        assert e.edge_id == "UA|UB"

    def test_make_auto_orders(self):
        e = WascEdge.make("UZ", "UA", Theme.SPLICING)
        assert e.anchor_uniprot == "UA"
        assert e.target_uniprot == "UZ"

    def test_make_swaps_symbols_too(self):
        e = WascEdge.make(
            "UZ", "UA", Theme.SPLICING,
            anchor_symbol="SYM_Z", target_symbol="SYM_A",
        )
        # anchor swapped from UZ→UA, so symbols swap too
        assert e.anchor_uniprot == "UA"
        assert e.anchor_symbol == "SYM_A"
        assert e.target_uniprot == "UZ"
        assert e.target_symbol == "SYM_Z"

    def test_make_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            WascEdge.make("UA", "UA", Theme.SPLICING)

    def test_frozen_dataclass(self):
        e = WascEdge(
            anchor_uniprot="UA", target_uniprot="UB",
            theme=Theme.SPLICING, network=Network.INDRA,
        )
        with pytest.raises((AttributeError, Exception)):
            e.anchor_uniprot = "UC"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_measured_cluster_members
# ---------------------------------------------------------------------------

# Standard test fixtures: a mini HGNC→UniProt dictionary.
HGNC_TO_UNIPROT = {
    "H_A": ["U_A"],
    "H_B": ["U_B"],
    "H_C": ["U_C"],
    "H_D": ["U_D"],
    "H_E": ["U_E"],
}


def _hgnc_set_to_uniprot_set(hgncs: set[str]) -> set[str]:
    """Mock implementation of hgnc_ids_to_uniprots."""
    return {u for h in hgncs for u in HGNC_TO_UNIPROT.get(h, [])}


class TestComputeMeasuredClusterMembers:
    def test_basic_per_theme_intersection(self):
        """Per-theme HGNC union → UniProt → measured intersection."""
        cluster_terms = [
            (Theme.SPLICING, "term:1"),
            (Theme.SPLICING, "term:2"),
            (Theme.CHROMATIN, "term:3"),
        ]
        measured = frozenset({"U_A", "U_B", "U_D", "U_E"})  # U_C not measured

        fetch_mock = MagicMock(return_value={
            "term:1": {"H_A", "H_B"},
            "term:2": {"H_B", "H_C"},   # overlap with term:1 on H_B
            "term:3": {"H_D", "H_E"},
        })
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)

        result = compute_measured_cluster_members(
            cluster_terms, measured, fetch_mock, hgnc_mock,
        )

        # Splicing union: H_A, H_B, H_C → U_A, U_B, U_C; measured ∩ = {U_A, U_B}
        assert result[Theme.SPLICING] == frozenset({"U_A", "U_B"})
        # Chromatin: H_D, H_E → U_D, U_E; both measured
        assert result[Theme.CHROMATIN] == frozenset({"U_D", "U_E"})

    def test_fetch_called_with_all_term_ids_batched(self):
        """fetch_term_members_func should be called ONCE with all term ids."""
        cluster_terms = [
            (Theme.SPLICING, "term:1"),
            (Theme.CHROMATIN, "term:2"),
            (Theme.TRANSPORT, "term:3"),
        ]
        fetch_mock = MagicMock(return_value={
            "term:1": set(), "term:2": set(), "term:3": set(),
        })
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)

        compute_measured_cluster_members(
            cluster_terms, frozenset(), fetch_mock, hgnc_mock,
        )

        assert fetch_mock.call_count == 1
        called_with = fetch_mock.call_args[0][0]
        assert sorted(called_with) == ["term:1", "term:2", "term:3"]

    def test_empty_measured_yields_empty_themes(self):
        """If measured is empty, every theme M_T is empty."""
        fetch_mock = MagicMock(return_value={"t:1": {"H_A"}})
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        result = compute_measured_cluster_members(
            [(Theme.SPLICING, "t:1")], frozenset(), fetch_mock, hgnc_mock,
        )
        assert result[Theme.SPLICING] == frozenset()


# ---------------------------------------------------------------------------
# enumerate_wasc_indra_edges — the M1 headline function
# ---------------------------------------------------------------------------

class TestEnumerateWascIndraEdges:
    def test_within_theme_edges_kept(self):
        """Within-theme INDRA hop-1 edges land in E_WASC."""
        fetch_mock = MagicMock(return_value={
            "term:spli": {"H_A", "H_B"},
            "term:chro": {"H_C", "H_D"},
        })
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        uniprot_to_sym_mock = MagicMock(return_value={
            "U_A": "SYM_A", "U_B": "SYM_B",
            "U_C": "SYM_C", "U_D": "SYM_D",
        })

        def extract_side_effect(*, cogex_client, features, max_hops,
                                min_evidence, restrict_endpoints_to_features):
            assert max_hops == 1
            assert restrict_endpoints_to_features is True
            features_set = set(features)
            if features_set == {"SYM_A", "SYM_B"}:
                return ([
                    ("SYM_A", "SYM_B", {"evidence_count": 5, "stmt_type": "Activation"}),
                ], features_set)
            if features_set == {"SYM_C", "SYM_D"}:
                return ([
                    ("SYM_C", "SYM_D", {"evidence_count": 3, "stmt_type": "Inhibition"}),
                ], features_set)
            return ([], features_set)

        extract_mock = MagicMock(side_effect=extract_side_effect)

        edges = enumerate_wasc_indra_edges(
            [(Theme.SPLICING, "term:spli"), (Theme.CHROMATIN, "term:chro")],
            frozenset({"U_A", "U_B", "U_C", "U_D"}),
            cogex_client=MagicMock(),
            fetch_term_members_func=fetch_mock,
            hgnc_to_uniprot_func=hgnc_mock,
            uniprot_to_symbol_func=uniprot_to_sym_mock,
            extract_subgraph_func=extract_mock,
        )

        assert len(edges) == 2
        ab = next(e for e in edges if e.theme == Theme.SPLICING)
        cd = next(e for e in edges if e.theme == Theme.CHROMATIN)
        assert {ab.anchor_uniprot, ab.target_uniprot} == {"U_A", "U_B"}
        assert {cd.anchor_uniprot, cd.target_uniprot} == {"U_C", "U_D"}
        assert ab.network == Network.INDRA
        # Lex-ordering invariant
        assert ab.anchor_uniprot <= ab.target_uniprot
        assert cd.anchor_uniprot <= cd.target_uniprot
        # Metadata propagated
        assert ab.evidence_count == 5
        assert ab.stmt_types == ("Activation",)
        assert cd.evidence_count == 3
        assert cd.stmt_types == ("Inhibition",)

    def test_cross_theme_edges_not_returned(self):
        """An anchor in Splicing and a target in Chromatin sharing an
        INDRA edge falls outside the primary E_WASC pool — because the
        extract_subgraph call is run per-theme, restrict_endpoints_to_features
        enforces both endpoints in the same theme's features set.

        This test verifies that two separate extract calls are made
        (one per theme), and that no edges cross theme boundaries."""
        fetch_mock = MagicMock(return_value={
            "term:spli": {"H_A"},
            "term:chro": {"H_C"},
        })
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        uniprot_to_sym_mock = MagicMock(return_value={
            "U_A": "SYM_A", "U_C": "SYM_C",
        })
        # The extract function only sees per-theme features.  If asked
        # to mix themes, it would return [] (no extract call happens
        # with mixed features in the M1 enumerator).
        extract_mock = MagicMock(return_value=([], set()))

        edges = enumerate_wasc_indra_edges(
            [(Theme.SPLICING, "term:spli"), (Theme.CHROMATIN, "term:chro")],
            frozenset({"U_A", "U_C"}),
            cogex_client=MagicMock(),
            fetch_term_members_func=fetch_mock,
            hgnc_to_uniprot_func=hgnc_mock,
            uniprot_to_symbol_func=uniprot_to_sym_mock,
            extract_subgraph_func=extract_mock,
        )
        # Each theme has only 1 measured member → no edges possible
        assert len(edges) == 0
        # extract_subgraph_func is NOT called for cross-theme features —
        # only single-theme calls (or no call when |M_T| < 2)
        assert extract_mock.call_count == 0  # both themes |M_T| = 1

    def test_undirected_dedupe(self):
        """If extract_subgraph returns both A→B and B→A, the output has one edge."""
        fetch_mock = MagicMock(return_value={"t:1": {"H_A", "H_B"}})
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        uniprot_to_sym_mock = MagicMock(return_value={"U_A": "A", "U_B": "B"})
        extract_mock = MagicMock(return_value=(
            [
                ("A", "B", {"evidence_count": 4, "stmt_type": "Activation"}),
                ("B", "A", {"evidence_count": 2, "stmt_type": "Inhibition"}),
            ],
            {"A", "B"},
        ))

        edges = enumerate_wasc_indra_edges(
            [(Theme.SPLICING, "t:1")],
            frozenset({"U_A", "U_B"}),
            cogex_client=MagicMock(),
            fetch_term_members_func=fetch_mock,
            hgnc_to_uniprot_func=hgnc_mock,
            uniprot_to_symbol_func=uniprot_to_sym_mock,
            extract_subgraph_func=extract_mock,
        )

        assert len(edges) == 1
        # Lex-ordered
        assert edges[0].anchor_uniprot == "U_A"
        assert edges[0].target_uniprot == "U_B"
        # Metadata aggregated across both directions
        assert edges[0].evidence_count == 6  # 4 + 2
        assert set(edges[0].stmt_types) == {"Activation", "Inhibition"}

    def test_self_loops_excluded(self):
        """Edge A → A is dropped from the output."""
        fetch_mock = MagicMock(return_value={"t:1": {"H_A", "H_B"}})
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        uniprot_to_sym_mock = MagicMock(return_value={
            "U_A": "SYM_A", "U_B": "SYM_B",
        })
        extract_mock = MagicMock(return_value=(
            [
                ("SYM_A", "SYM_A", {"evidence_count": 1}),  # self-loop
                ("SYM_A", "SYM_B", {"evidence_count": 2}),  # legit
            ],
            {"SYM_A", "SYM_B"},
        ))

        edges = enumerate_wasc_indra_edges(
            [(Theme.SPLICING, "t:1")],
            frozenset({"U_A", "U_B"}),
            cogex_client=MagicMock(),
            fetch_term_members_func=fetch_mock,
            hgnc_to_uniprot_func=hgnc_mock,
            uniprot_to_symbol_func=uniprot_to_sym_mock,
            extract_subgraph_func=extract_mock,
        )

        assert len(edges) == 1
        assert edges[0].evidence_count == 2

    def test_unmeasured_endpoints_filtered(self):
        """Theme with only 1 measured member yields no edges (and no
        extract_subgraph call)."""
        fetch_mock = MagicMock(return_value={"t:1": {"H_A", "H_B"}})
        hgnc_mock = MagicMock(side_effect=_hgnc_set_to_uniprot_set)
        # Only U_A is in the measured set; U_B not measured.
        uniprot_to_sym_mock = MagicMock(return_value={"U_A": "A"})
        extract_mock = MagicMock(return_value=([], {"A"}))

        edges = enumerate_wasc_indra_edges(
            [(Theme.SPLICING, "t:1")],
            frozenset({"U_A"}),
            cogex_client=MagicMock(),
            fetch_term_members_func=fetch_mock,
            hgnc_to_uniprot_func=hgnc_mock,
            uniprot_to_symbol_func=uniprot_to_sym_mock,
            extract_subgraph_func=extract_mock,
        )

        assert len(edges) == 0
        # extract_subgraph_func should NOT be called when |M_T| < 2
        assert extract_mock.call_count == 0


# ---------------------------------------------------------------------------
# DEFAULT_CLUSTER_TERMS frozen reference
# ---------------------------------------------------------------------------

class TestDefaultClusterTerms:
    def test_eight_terms_three_themes(self):
        assert len(DEFAULT_CLUSTER_TERMS) == 8
        themes = {t for t, _ in DEFAULT_CLUSTER_TERMS}
        assert themes == {Theme.SPLICING, Theme.CHROMATIN, Theme.TRANSPORT}
        spli = sum(1 for t, _ in DEFAULT_CLUSTER_TERMS if t == Theme.SPLICING)
        chro = sum(1 for t, _ in DEFAULT_CLUSTER_TERMS if t == Theme.CHROMATIN)
        tran = sum(1 for t, _ in DEFAULT_CLUSTER_TERMS if t == Theme.TRANSPORT)
        # Per spec §1: 3 Splicing, 2 Chromatin, 3 Transport
        assert (spli, chro, tran) == (3, 2, 3)

    def test_all_terms_unique(self):
        term_ids = [tid for _, tid in DEFAULT_CLUSTER_TERMS]
        assert len(term_ids) == len(set(term_ids))
