"""Graph-node identity: CURIE keying vs legacy symbol keying.

The legacy graph keyed INDRA ``BioEntity`` nodes on their bare ``name``
and represented each measured protein by every symbol/alias MyGene
returned for it.  Name strings are not entity-preserving, so an alias of
a measured protein that happens to be another entity's official name
admitted that entity — and all of its edges — under the measured
protein's identity.  The canonical proteome run acquired the androgen
receptor (``hgnc:644``, unmeasured, ~14.8k regulatory edges) as AKR1B1
(P15121) because ``AR`` is an AKR1B1 alias, which gave P15121 a hop-1
shell of 821 against a dataset median of 17.

These tests pin the fix: one namespaced CURIE per feature, matched on
``BioEntity.id``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cliquefinder.panels._graph_key import (
    CURIE_GRAPH_KEY,
    SYMBOL_GRAPH_KEY,
    VALID_GRAPH_KEYS,
    node_property_for,
    resolve_feature_curies,
)
from cliquefinder.panels.landscape import LandscapeDesign, _build_distance_matrix
from cliquefinder.stats.network_proximity import (
    _validated_key_property,
    extract_subgraph_induced_by_features,
    query_gene_degrees_batched,
)


class TestKeySpaceMapping:
    def test_maps_to_node_properties(self):
        assert node_property_for(CURIE_GRAPH_KEY) == "id"
        assert node_property_for(SYMBOL_GRAPH_KEY) == "name"

    def test_rejects_unknown_key_space(self):
        with pytest.raises(ValueError, match="graph_key must be one of"):
            node_property_for("uniprot")

    def test_key_property_guard_blocks_cypher_injection(self):
        """``key_property`` is interpolated into query TEXT (Cypher has no
        parameter slot for a property name), so the allowlist is the only
        thing standing between a caller and arbitrary Cypher."""
        for hostile in ("name}) DETACH DELETE (n", "id RETURN 1 //", ""):
            with pytest.raises(ValueError, match="key_property must be one of"):
                _validated_key_property(hostile)

    def test_valid_key_spaces_are_exactly_two(self):
        assert VALID_GRAPH_KEYS == {CURIE_GRAPH_KEY, SYMBOL_GRAPH_KEY}


class TestCurieResolutionIsEntityPreserving:
    """Offline resolution against INDRA's bundled UniProt/HGNC tables."""

    def test_collision_prone_features_resolve_to_their_own_gene(self):
        """The regression that motivated the fix.

        Each of these measured proteins carries an alias that is another,
        UNMEASURED gene's official symbol.  Under CURIE resolution each
        must land on its own gene, never the homonym.
        """
        # measured feature -> (its own HGNC CURIE, the homonym it must NOT get)
        cases = {
            "P15121": ("hgnc:381", "hgnc:644"),    # AKR1B1, alias "AR"   vs androgen receptor
            "P49327": ("hgnc:3594", "hgnc:11920"),  # FASN,   alias "FAS" vs FAS receptor
            "Q07889": ("hgnc:11187", "hgnc:4893"),  # SOS1,   alias "HGF" vs HGF
            "P21291": ("hgnc:2469", "hgnc:2367"),   # CSRP1,  alias "CRP" vs CRP
        }
        feat_to_curie, _, _ = resolve_feature_curies(list(cases))
        for fid, (own, homonym) in cases.items():
            assert feat_to_curie[fid] == own
            assert feat_to_curie[fid] != homonym

    def test_one_node_per_feature(self):
        """The collision can only happen when a feature owns many node
        keys.  CURIE resolution must yield exactly one."""
        feat_to_curie, _, _ = resolve_feature_curies(["P15121", "P49327"])
        assert all(
            isinstance(c, str) and c.startswith("hgnc:")
            for c in feat_to_curie.values()
        )

    def test_reports_unresolvable_features_instead_of_dropping_them(self):
        """A synthetic standard has no gene identity.  It must come back
        in ``unresolved`` rather than silently vanishing."""
        _, _, unresolved = resolve_feature_curies(
            ["P15121", "1/iRT_protein", "NOT_AN_ACCESSION"]
        )
        assert "1/iRT_protein" in unresolved
        assert "NOT_AN_ACCESSION" in unresolved
        assert "P15121" not in unresolved

    def test_features_sharing_a_gene_share_one_node(self):
        """Distinct rows can be the same gene (the two TMPO rows).  They
        are genuinely one INDRA node and the inverse map must say so."""
        feat_to_curie, curie_to_feats, _ = resolve_feature_curies(
            ["P42166", "P42167"]
        )
        assert feat_to_curie["P42166"] == feat_to_curie["P42167"]
        shared = feat_to_curie["P42166"]
        assert sorted(curie_to_feats[shared]) == ["P42166", "P42167"]


class TestCypherKeysOnRequestedProperty:
    """The key space must reach every Cypher that identifies a node."""

    @staticmethod
    def _capture():
        captured: list[str] = []

        def fake_execute(query, **params):
            captured.append(query)
            return []

        client = MagicMock()
        client._execute_query = fake_execute
        return client, captured

    @pytest.mark.parametrize(
        "key_property,expected,forbidden",
        [("id", "b.id", "b.name"), ("name", "b.name", "b.id")],
    )
    def test_extraction_keys_on_property(self, key_property, expected, forbidden):
        client, captured = self._capture()
        extract_subgraph_induced_by_features(
            cogex_client=client,
            features=["hgnc:381"],
            max_hops=2,
            restrict_endpoints_to_features=True,
            key_property=key_property,
        )
        joined = "\n".join(captured)
        assert expected in joined
        assert forbidden not in joined

    @pytest.mark.parametrize(
        "key_property,expected,forbidden",
        [("id", "g.id", "g.name"), ("name", "g.name", "g.id")],
    )
    def test_degree_query_keys_on_property(
        self, key_property, expected, forbidden
    ):
        client, captured = self._capture()
        query_gene_degrees_batched(
            cogex_client=client,
            gene_names=["hgnc:381"],
            key_property=key_property,
        )
        joined = "\n".join(captured)
        assert expected in joined
        assert forbidden not in joined


class TestDesignRecordsKeySpace:
    @staticmethod
    def _design(**kw):
        base = dict(
            contrast=("A", "B"), max_hops=2,
            n_permutations=999, covariates=("Sex",),
        )
        base.update(kw)
        return LandscapeDesign(**base)

    def test_defaults_to_curie(self):
        assert self._design().graph_key == CURIE_GRAPH_KEY

    def test_round_trips(self):
        d = self._design()
        assert LandscapeDesign.from_dict(d.to_dict()).graph_key == CURIE_GRAPH_KEY

    def test_missing_key_means_legacy_symbol_run(self):
        """A manifest written before the field existed came from the
        symbol-keyed graph.  Defaulting it to the constructor's "curie"
        would let the resume guard mix two different graphs in one
        output_dir."""
        payload = self._design().to_dict()
        payload.pop("graph_key")
        assert LandscapeDesign.from_dict(payload).graph_key == SYMBOL_GRAPH_KEY

    def test_rejects_unknown_key_space(self):
        with pytest.raises(ValueError, match="graph_key must be one of"):
            self._design(graph_key="ensembl")


class _FakeGraphClient:
    """Minimal CoGEx stand-in driven by an explicit node/edge spec."""

    def __init__(self, nodes: set[str], edges: list[tuple[str, str]],
                 degrees: dict[str, int] | None = None):
        self.nodes = nodes
        self.edges = edges
        self.degrees = degrees or {}

    def _execute_query(self, query, **params):
        if "count(r)" in query:  # degree query
            return [[n, self.degrees.get(n, 0)]
                    for n in params["gene_list"] if n in self.nodes]
        if "indra_rel" in query:  # edge query
            batch, fset = set(params["batch"]), set(params["feature_set"])
            return [[s, t, 5, "Activation"] for s, t in self.edges
                    if s in batch and t in fset]
        return [[n] for n in params["batch"] if n in self.nodes]  # probe


class TestAggregationAccounting:
    def test_feature_with_no_graph_node_is_reported_unmatched(self):
        """The legacy loop iterated the feature->node mapping, so a
        feature that resolved to NOTHING was never visited and never
        marked unmatched — it became indistinguishable downstream from a
        feature that resolved but was graph-isolated."""
        client = _FakeGraphClient(nodes={"n1", "n2"}, edges=[("n1", "n2")])
        _matrix, unmatched, degrees, _edges = _build_distance_matrix(
            cogex_client=client,
            node_keys=["n1", "n2"],
            measured_feature_ids=["F1", "F2", "F_UNRESOLVED"],
            feat_to_nodes={"F1": ["n1"], "F2": ["n2"]},
            node_to_feats={"n1": ["F1"], "n2": ["F2"]},
            max_hops=2,
            key_property="id",
        )
        assert "F_UNRESOLVED" in unmatched
        assert degrees["F_UNRESOLVED"] == 0
        assert "F1" not in unmatched and "F2" not in unmatched

    def test_features_sharing_a_node_both_inherit_its_distances(self):
        """Two rows of the same gene are one INDRA node; each must see
        that node's neighbours."""
        client = _FakeGraphClient(
            nodes={"shared", "other"}, edges=[("shared", "other")],
        )
        matrix, _unmatched, _degrees, _edges = _build_distance_matrix(
            cogex_client=client,
            node_keys=["shared", "other"],
            measured_feature_ids=["A", "B", "C"],
            feat_to_nodes={"A": ["shared"], "B": ["shared"], "C": ["other"]},
            node_to_feats={"shared": ["A", "B"], "other": ["C"]},
            max_hops=2,
            key_property="id",
        )
        assert matrix.distances_from("A").get("C") == 1
        assert matrix.distances_from("B").get("C") == 1

    def test_degree_is_the_features_own_node_degree(self):
        """Under CURIE keying a feature owns one node, so its degree is
        that node's — it cannot inherit a homonym's larger count."""
        client = _FakeGraphClient(
            nodes={"hgnc:381"}, edges=[], degrees={"hgnc:381": 2120},
        )
        _matrix, _unmatched, degrees, _edges = _build_distance_matrix(
            cogex_client=client,
            node_keys=["hgnc:381"],
            measured_feature_ids=["P15121"],
            feat_to_nodes={"P15121": ["hgnc:381"]},
            node_to_feats={"hgnc:381": ["P15121"]},
            max_hops=2,
            key_property="id",
        )
        assert degrees["P15121"] == 2120
