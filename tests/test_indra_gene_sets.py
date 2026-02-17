"""Tests for INDRA module → CliqueDefinition conversion."""

import pytest
from unittest.mock import MagicMock

from cliquefinder.knowledge.cogex import INDRAModule, INDRAEdge
from cliquefinder.stats.clique_analysis import (
    modules_to_clique_definitions,
    CliqueDefinition,
)


# ---------------------------------------------------------------------------
# Fixtures: mock INDRA's hgnc_client so tests don't require INDRA
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_indra():
    """Patch hgnc_client on the already-imported cogex module."""
    mock_hgnc = MagicMock()
    mock_hgnc.tfs = ["TP53", "MYC"]
    mock_hgnc.kinases = ["AKT1"]
    mock_hgnc.phosphatases = ["PTEN"]
    mock_hgnc.get_current_hgnc_id = MagicMock(return_value=None)

    import cliquefinder.knowledge.cogex as cogex_mod
    orig_hgnc = cogex_mod.hgnc_client
    orig_available = cogex_mod.INDRA_AVAILABLE
    cogex_mod.hgnc_client = mock_hgnc
    cogex_mod.INDRA_AVAILABLE = True
    yield mock_hgnc
    cogex_mod.hgnc_client = orig_hgnc
    cogex_mod.INDRA_AVAILABLE = orig_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_ID_COUNTER = 0

def _make_edge(reg_name, target_name, reg_type="activation"):
    global _TARGET_ID_COUNTER
    _TARGET_ID_COUNTER += 1
    return INDRAEdge(
        regulator_id=("HGNC", "1"),
        regulator_name=reg_name,
        target_id=("HGNC", str(_TARGET_ID_COUNTER)),
        target_name=target_name,
        regulation_type=reg_type,
        evidence_count=5,
        stmt_hash=12345,
        source_counts="{}",
    )


def _make_module(name, targets_with_types):
    """targets_with_types: list of (target_name, regulation_type)"""
    edges = [_make_edge(name, t, rt) for t, rt in targets_with_types]
    return INDRAModule(
        regulator_id=("HGNC", "1"),
        regulator_name=name,
        targets=edges,
    )


# ---------------------------------------------------------------------------
# modules_to_clique_definitions tests
# ---------------------------------------------------------------------------

class TestModulesToCliqueDefinitions:
    def test_basic_conversion(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "UP001", "BAX": "UP002", "BBC3": "UP003"}

        result = modules_to_clique_definitions([module], symbol_to_feature, verbose=False)
        assert len(result) == 1
        assert result[0].clique_id == "TP53"
        assert result[0].regulator == "TP53"
        assert len(result[0].protein_ids) == 3
        assert set(result[0].protein_ids) == {"UP001", "UP002", "UP003"}

    def test_direction_positive(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}, verbose=False
        )
        assert result[0].direction == "positive"
        assert result[0].n_positive_edges == 3
        assert result[0].n_negative_edges == 0

    def test_direction_negative(self):
        module = _make_module("TP53", [
            ("MDM2", "repression"),
            ("BAX", "repression"),
            ("BBC3", "repression"),
        ])
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}, verbose=False
        )
        assert result[0].direction == "negative"
        assert result[0].n_positive_edges == 0
        assert result[0].n_negative_edges == 3

    def test_direction_mixed(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "repression"),
            ("BBC3", "activation"),
        ])
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}, verbose=False
        )
        assert result[0].direction == "mixed"
        assert result[0].n_positive_edges == 2
        assert result[0].n_negative_edges == 1

    def test_min_genes_filter(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
        ])
        # Only 1 gene maps
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1"}, min_genes_found=3, verbose=False
        )
        assert len(result) == 0  # Filtered out

    def test_partial_mapping(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("UNMAPPED", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2"}
        result = modules_to_clique_definitions(
            [module], symbol_to_feature, min_genes_found=2, verbose=False
        )
        assert len(result) == 1
        assert len(result[0].protein_ids) == 2  # Only mapped ones

    def test_n_indra_targets_field(self):
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}, verbose=False
        )
        assert result[0].n_indra_targets == 3

    def test_multiple_modules(self):
        m1 = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        m2 = _make_module("MYC", [
            ("CDK2", "activation"),
            ("CCND1", "repression"),
            ("E2F1", "activation"),
        ])
        symbol_to_feature = {
            "MDM2": "U1", "BAX": "U2", "BBC3": "U3",
            "CDK2": "U4", "CCND1": "U5", "E2F1": "U6",
        }
        result = modules_to_clique_definitions(
            [m1, m2], symbol_to_feature, verbose=False
        )
        assert len(result) == 2
        assert result[0].clique_id == "TP53"
        assert result[1].clique_id == "MYC"
        assert result[0].direction == "positive"
        assert result[1].direction == "mixed"

    def test_empty_modules_list(self):
        result = modules_to_clique_definitions([], {}, verbose=False)
        assert result == []

    def test_clique_definition_fields(self):
        """Verify all expected CliqueDefinition fields are populated."""
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "repression"),
            ("BBC3", "activation"),
        ])
        result = modules_to_clique_definitions(
            [module], {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}, verbose=False
        )
        cd = result[0]
        assert cd.clique_id == "TP53"
        assert cd.regulator == "TP53"
        assert cd.condition is None  # Not condition-specific
        assert cd.coherence is None  # No correlation from INDRA
        assert cd.n_indra_targets == 3
        assert cd.direction == "mixed"
        assert cd.n_positive_edges == 2
        assert cd.n_negative_edges == 1
        assert isinstance(cd.protein_ids, list)
