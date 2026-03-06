"""Tests for statement type presets, resolve_stmt_types(), and phosphorylation edge parsing.

KG-IV-1: Ensures --stmt-types phosphorylation produces actual results by verifying
that Phosphorylation edges are parsed into INDRAEdge objects (not silently dropped).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from cliquefinder.knowledge.cogex import (
    resolve_stmt_types,
    ALL_REGULATORY_TYPES,
    ACTIVATION_TYPES,
    REPRESSION_TYPES,
    PHOSPHORYLATION_TYPES,
    STMT_TYPE_PRESETS,
    INDRAEdge,
    INDRAModule,
    CoGExClient,
)


class TestResolveStmtTypes:
    def test_none_returns_regulatory(self):
        result = resolve_stmt_types(None)
        assert set(result) == ALL_REGULATORY_TYPES

    def test_preset_regulatory(self):
        assert set(resolve_stmt_types("regulatory")) == ALL_REGULATORY_TYPES

    def test_preset_activation(self):
        assert set(resolve_stmt_types("activation")) == ACTIVATION_TYPES

    def test_preset_repression(self):
        assert set(resolve_stmt_types("repression")) == REPRESSION_TYPES

    def test_preset_phosphorylation(self):
        assert set(resolve_stmt_types("phosphorylation")) == PHOSPHORYLATION_TYPES

    def test_preset_case_insensitive(self):
        assert set(resolve_stmt_types("ACTIVATION")) == ACTIVATION_TYPES
        assert set(resolve_stmt_types("Regulatory")) == ALL_REGULATORY_TYPES
        assert set(resolve_stmt_types("PHOSPHORYLATION")) == PHOSPHORYLATION_TYPES

    def test_raw_comma_separated(self):
        result = resolve_stmt_types("IncreaseAmount,Phosphorylation")
        assert set(result) == {"IncreaseAmount", "Phosphorylation"}

    def test_raw_with_spaces(self):
        result = resolve_stmt_types(" IncreaseAmount , Phosphorylation ")
        assert set(result) == {"IncreaseAmount", "Phosphorylation"}

    def test_single_raw_type(self):
        result = resolve_stmt_types("Phosphorylation")
        # "phosphorylation" matches the preset (case-insensitive)
        assert set(result) == PHOSPHORYLATION_TYPES

    def test_raw_non_preset(self):
        result = resolve_stmt_types("IncreaseAmount")
        # "increaseamount" doesn't match any preset, treated as raw
        assert result == ["IncreaseAmount"]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            resolve_stmt_types("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            resolve_stmt_types("   ")


class TestStmtTypePresets:
    def test_all_presets_are_non_empty(self):
        for name, types in STMT_TYPE_PRESETS.items():
            assert len(types) > 0, f"Preset '{name}' is empty"

    def test_regulatory_is_union(self):
        assert ALL_REGULATORY_TYPES == ACTIVATION_TYPES | REPRESSION_TYPES

    def test_phosphorylation_type(self):
        assert PHOSPHORYLATION_TYPES == {"Phosphorylation"}

    def test_preset_keys(self):
        assert set(STMT_TYPE_PRESETS.keys()) == {
            "regulatory", "activation", "repression", "phosphorylation"
        }


# ===========================================================================
# Fixtures for INDRA mocking (required for CoGExClient tests)
# ===========================================================================

MOCK_TFS = ["TP53", "MYC"]
MOCK_KINASES = ["AKT1", "MAPK1"]
MOCK_PHOSPHATASES = ["PTEN"]


@pytest.fixture()
def _mock_indra():
    """Patch hgnc_client on the already-imported cogex module."""
    import cliquefinder.knowledge.cogex as cogex_mod

    mock_hgnc = MagicMock()
    mock_hgnc.tfs = MOCK_TFS
    mock_hgnc.kinases = MOCK_KINASES
    mock_hgnc.phosphatases = MOCK_PHOSPHATASES
    mock_hgnc.get_current_hgnc_id = MagicMock(return_value=None)

    orig_hgnc = cogex_mod.hgnc_client
    orig_available = cogex_mod.INDRA_AVAILABLE
    cogex_mod.hgnc_client = mock_hgnc
    cogex_mod.INDRA_AVAILABLE = True
    yield mock_hgnc
    cogex_mod.hgnc_client = orig_hgnc
    cogex_mod.INDRA_AVAILABLE = orig_available


def _make_query_row(
    reg_id="hgnc:391", reg_name="AKT1",
    target_id="hgnc:1234", target_name="BAD",
    stmt_type="Phosphorylation", evidence=5,
    stmt_hash=99999, source_counts='{"reach": 3}',
):
    """Produce a row tuple as returned by Neo4j query."""
    return (reg_id, reg_name, target_id, target_name,
            stmt_type, evidence, stmt_hash, source_counts)


# ===========================================================================
# KG-IV-1: Phosphorylation edge parsing in get_downstream_targets
# ===========================================================================


class TestPhosphorylationEdgeParsing:
    """KG-IV-1: Phosphorylation edges must be parsed, not silently dropped."""

    def test_phosphorylation_edge_parsed_in_get_downstream_targets(self, _mock_indra):
        """Phosphorylation stmt_type produces an INDRAEdge with reg_type='phosphorylation'."""
        import cliquefinder.knowledge.cogex as cogex_mod

        row = _make_query_row(stmt_type="Phosphorylation")
        mock_client = MagicMock()
        mock_client.query_tx.return_value = [row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:391"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                edges = client.get_downstream_targets(
                    regulator=("HGNC", "391"),
                    stmt_types=["Phosphorylation"],
                    min_evidence=1,
                )

        assert len(edges) == 1
        assert edges[0].regulation_type == "phosphorylation"
        assert edges[0].target_name == "BAD"

    def test_activation_edge_still_works(self, _mock_indra):
        """Activation edges are unaffected by the phosphorylation fix."""
        import cliquefinder.knowledge.cogex as cogex_mod

        row = _make_query_row(stmt_type="IncreaseAmount", target_id="hgnc:5555", target_name="MDM2")
        mock_client = MagicMock()
        mock_client.query_tx.return_value = [row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:391"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                edges = client.get_downstream_targets(
                    regulator=("HGNC", "391"),
                    stmt_types=["IncreaseAmount"],
                    min_evidence=1,
                )

        assert len(edges) == 1
        assert edges[0].regulation_type == "activation"

    def test_repression_edge_still_works(self, _mock_indra):
        """Repression edges are unaffected by the phosphorylation fix."""
        import cliquefinder.knowledge.cogex as cogex_mod

        row = _make_query_row(stmt_type="Inhibition", target_id="hgnc:6666", target_name="PTEN")
        mock_client = MagicMock()
        mock_client.query_tx.return_value = [row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:391"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                edges = client.get_downstream_targets(
                    regulator=("HGNC", "391"),
                    stmt_types=["Inhibition"],
                    min_evidence=1,
                )

        assert len(edges) == 1
        assert edges[0].regulation_type == "repression"

    def test_unknown_stmt_type_skipped_with_warning(self, _mock_indra, caplog):
        """Unknown statement types are skipped and produce a warning log."""
        import cliquefinder.knowledge.cogex as cogex_mod

        unknown_row = _make_query_row(stmt_type="ComplexFormation", target_id="hgnc:7777", target_name="BRCA1")
        good_row = _make_query_row(stmt_type="Phosphorylation", target_id="hgnc:8888", target_name="BAX")
        mock_client = MagicMock()
        mock_client.query_tx.return_value = [unknown_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:391"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                with caplog.at_level(logging.WARNING):
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "391"),
                        stmt_types=["ComplexFormation", "Phosphorylation"],
                        min_evidence=1,
                    )

        # Only the phosphorylation edge survives
        assert len(edges) == 1
        assert edges[0].regulation_type == "phosphorylation"
        assert edges[0].target_name == "BAX"
        assert "Unknown statement type: ComplexFormation" in caplog.text

    def test_mixed_stmt_types_all_parsed(self, _mock_indra):
        """Activation, repression, and phosphorylation edges can coexist."""
        import cliquefinder.knowledge.cogex as cogex_mod

        rows = [
            _make_query_row(stmt_type="IncreaseAmount", target_id="hgnc:1001", target_name="GENE_A"),
            _make_query_row(stmt_type="Inhibition", target_id="hgnc:1002", target_name="GENE_B"),
            _make_query_row(stmt_type="Phosphorylation", target_id="hgnc:1003", target_name="GENE_C"),
        ]
        mock_client = MagicMock()
        mock_client.query_tx.return_value = rows

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:391"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                edges = client.get_downstream_targets(
                    regulator=("HGNC", "391"),
                    stmt_types=["IncreaseAmount", "Inhibition", "Phosphorylation"],
                    min_evidence=1,
                )

        assert len(edges) == 3
        reg_types = {e.regulation_type for e in edges}
        assert reg_types == {"activation", "repression", "phosphorylation"}


# ===========================================================================
# KG-IV-1: Phosphorylation edge parsing in get_regulator_modules
# ===========================================================================


class TestPhosphorylationInDiscoverRegulators:
    """KG-IV-1: discover_regulators must also handle Phosphorylation edges."""

    def test_phosphorylation_edges_in_discover_regulators(self, _mock_indra):
        """Phosphorylation edges in discover_regulators are not silently dropped."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Mock hgnc_client.get_current_hgnc_id to resolve gene names to IDs
        gene_id_map = {f"TARGET{i}": str(2000 + i) for i in range(5)}
        _mock_indra.get_current_hgnc_id = MagicMock(
            side_effect=lambda name: gene_id_map.get(name)
        )

        # Build rows for a single regulator with phosphorylation targets
        rows = [
            _make_query_row(
                reg_id="hgnc:391", reg_name="AKT1",
                target_id=f"hgnc:{2000 + i}", target_name=f"TARGET{i}",
                stmt_type="Phosphorylation", evidence=3, stmt_hash=50000 + i,
            )
            for i in range(5)
        ]

        mock_client = MagicMock()
        mock_client.query_tx.return_value = rows

        # norm_id needs to return the right CURIE for each call
        def mock_norm_id(ns, hid):
            return f"hgnc:{hid}"

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", side_effect=mock_norm_id):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="pw")
                # discover_regulators returns Dict[str, List[INDRAEdge]]
                reg_dict = client.discover_regulators(
                    gene_universe=[f"TARGET{i}" for i in range(5)],
                    stmt_types=["Phosphorylation"],
                    min_evidence=1,
                    min_targets=1,
                )

        # Should produce edges grouped by regulator
        assert len(reg_dict) >= 1
        # All edges for AKT1 should have phosphorylation type
        akt1_edges = reg_dict.get("AKT1", [])
        assert len(akt1_edges) == 5
        assert all(e.regulation_type == "phosphorylation" for e in akt1_edges)


# ===========================================================================
# KG-IV-1: INDRAModule.phosphorylated_targets property
# ===========================================================================


class TestINDRAModulePhosphorylatedTargets:
    """INDRAModule should expose phosphorylated_targets alongside activated/repressed."""

    def test_phosphorylated_targets_property(self):
        """phosphorylated_targets returns only edges with regulation_type='phosphorylation'."""
        edges = [
            INDRAEdge(
                regulator_id=("HGNC", "391"), regulator_name="AKT1",
                target_id=("HGNC", "1001"), target_name="BAD",
                regulation_type="phosphorylation", evidence_count=5,
                stmt_hash=100, source_counts="{}",
            ),
            INDRAEdge(
                regulator_id=("HGNC", "391"), regulator_name="AKT1",
                target_id=("HGNC", "1002"), target_name="GSK3B",
                regulation_type="phosphorylation", evidence_count=3,
                stmt_hash=101, source_counts="{}",
            ),
            INDRAEdge(
                regulator_id=("HGNC", "391"), regulator_name="AKT1",
                target_id=("HGNC", "1003"), target_name="MTOR",
                regulation_type="activation", evidence_count=4,
                stmt_hash=102, source_counts="{}",
            ),
        ]
        module = INDRAModule(
            regulator_id=("HGNC", "391"),
            regulator_name="AKT1",
            targets=edges,
        )
        assert module.phosphorylated_targets == {("HGNC", "1001"), ("HGNC", "1002")}
        assert module.activated_targets == {("HGNC", "1003")}

    def test_phosphorylated_targets_empty_when_none(self):
        """phosphorylated_targets is empty when no phosphorylation edges exist."""
        edges = [
            INDRAEdge(
                regulator_id=("HGNC", "11998"), regulator_name="TP53",
                target_id=("HGNC", "1234"), target_name="MDM2",
                regulation_type="activation", evidence_count=10,
                stmt_hash=200, source_counts="{}",
            ),
        ]
        module = INDRAModule(
            regulator_id=("HGNC", "11998"),
            regulator_name="TP53",
            targets=edges,
        )
        assert module.phosphorylated_targets == set()


# ===========================================================================
# KG-IV-1: indra_source._map_relationship handles phosphorylation
# ===========================================================================


class TestIndraSourcePhosphorylationMapping:
    """_map_relationship must map 'phosphorylation' to PHOSPHORYLATES."""

    def test_phosphorylation_maps_to_phosphorylates(self):
        from cliquefinder.knowledge.base import RelationshipType
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        # Access the static method via the class
        source = INDRAKnowledgeSource.__new__(INDRAKnowledgeSource)
        result = source._map_relationship("phosphorylation")
        assert result == RelationshipType.PHOSPHORYLATES

    def test_activation_still_maps_correctly(self):
        from cliquefinder.knowledge.base import RelationshipType
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        source = INDRAKnowledgeSource.__new__(INDRAKnowledgeSource)
        assert source._map_relationship("activation") == RelationshipType.INCREASES_EXPRESSION

    def test_repression_still_maps_correctly(self):
        from cliquefinder.knowledge.base import RelationshipType
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        source = INDRAKnowledgeSource.__new__(INDRAKnowledgeSource)
        assert source._map_relationship("repression") == RelationshipType.DECREASES_EXPRESSION

    def test_unknown_maps_to_regulates(self):
        from cliquefinder.knowledge.base import RelationshipType
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        source = INDRAKnowledgeSource.__new__(INDRAKnowledgeSource)
        assert source._map_relationship("something_else") == RelationshipType.REGULATES
