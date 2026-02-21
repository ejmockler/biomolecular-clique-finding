"""Tests for INDRA module → CliqueDefinition conversion."""

import logging
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


# ---------------------------------------------------------------------------
# Case-insensitive gene name mapping tests
# ---------------------------------------------------------------------------

class TestCaseInsensitiveMapping:
    """Mixed-case gene names should all resolve to the same feature."""

    def test_lowercase_gene_names_resolve(self):
        """'tp53' style target names should map via upper-case fallback."""
        # INDRA edge has lowercase target names
        module = _make_module("TP53", [
            ("mdm2", "activation"),
            ("bax", "activation"),
            ("bbc3", "activation"),
        ])
        # symbol_to_feature uses canonical uppercase
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}
        result = modules_to_clique_definitions(
            [module], symbol_to_feature, verbose=False
        )
        assert len(result) == 1
        assert set(result[0].protein_ids) == {"U1", "U2", "U3"}

    def test_mixed_case_gene_names_resolve(self):
        """'Tp53' style target names should map via upper-case fallback."""
        module = _make_module("TP53", [
            ("Mdm2", "activation"),
            ("Bax", "activation"),
            ("Bbc3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}
        result = modules_to_clique_definitions(
            [module], symbol_to_feature, verbose=False
        )
        assert len(result) == 1
        assert set(result[0].protein_ids) == {"U1", "U2", "U3"}

    def test_uppercase_gene_names_still_work(self):
        """Standard upper-case names continue to resolve directly."""
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}
        result = modules_to_clique_definitions(
            [module], symbol_to_feature, verbose=False
        )
        assert len(result) == 1
        assert set(result[0].protein_ids) == {"U1", "U2", "U3"}

    def test_all_case_variants_produce_same_feature(self):
        """'tp53', 'Tp53', and 'TP53' all resolve to same feature ID."""
        for variant in ["tp53", "Tp53", "TP53"]:
            module = _make_module("REG1", [
                (variant, "activation"),
                ("BAX", "activation"),
                ("BBC3", "activation"),
            ])
            symbol_to_feature = {"TP53": "FEAT_TP53", "BAX": "U2", "BBC3": "U3"}
            result = modules_to_clique_definitions(
                [module], symbol_to_feature, verbose=False
            )
            assert len(result) == 1, f"Failed for variant: {variant}"
            assert "FEAT_TP53" in result[0].protein_ids, (
                f"Variant '{variant}' did not resolve to FEAT_TP53"
            )


# ---------------------------------------------------------------------------
# Unmapped gene warning tests
# ---------------------------------------------------------------------------

class TestUnmappedGeneLogging:
    """Verify that unmapped genes produce appropriate log warnings."""

    def test_unmapped_genes_logged_as_warning(self, caplog):
        """Unmapped genes should produce a WARNING log message."""
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("FAKE_GENE_1", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2"}
        with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.clique_analysis"):
            modules_to_clique_definitions(
                [module], symbol_to_feature, min_genes_found=2, verbose=False
            )
        # Should mention the module name and the unmapped gene
        assert any("TP53" in rec.message and "unmapped" in rec.message for rec in caplog.records), (
            f"Expected unmapped warning for TP53. Records: {[r.message for r in caplog.records]}"
        )
        assert any("FAKE_GENE_1" in rec.message for rec in caplog.records)

    def test_majority_unmapped_produces_louder_warning(self, caplog):
        """>50% unmapped targets should produce an extra loud warning."""
        # 4 targets, only 1 maps => 75% unmapped
        module = _make_module("MYC", [
            ("MDM2", "activation"),
            ("NOPE1", "activation"),
            ("NOPE2", "activation"),
            ("NOPE3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1"}
        with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.clique_analysis"):
            modules_to_clique_definitions(
                [module], symbol_to_feature, min_genes_found=1, verbose=False
            )
        # Should mention >50% unmapped
        loud_warnings = [
            r for r in caplog.records
            if "MYC" in r.message and ">50%" in r.message
        ]
        assert len(loud_warnings) >= 1, (
            f"Expected >50% unmapped warning. Records: {[r.message for r in caplog.records]}"
        )

    def test_mapping_summary_logged(self, caplog):
        """Gene mapping summary should be logged at INFO level."""
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}
        with caplog.at_level(logging.INFO, logger="cliquefinder.stats.clique_analysis"):
            modules_to_clique_definitions(
                [module], symbol_to_feature, verbose=False
            )
        summary_msgs = [
            r for r in caplog.records if "Gene mapping summary" in r.message
        ]
        assert len(summary_msgs) == 1
        assert "1/1 modules retained" in summary_msgs[0].message
        assert "3 total genes mapped" in summary_msgs[0].message
        assert "0 genes unmapped" in summary_msgs[0].message

    def test_no_warning_when_all_genes_map(self, caplog):
        """No unmapped-gene warnings when all targets resolve."""
        module = _make_module("TP53", [
            ("MDM2", "activation"),
            ("BAX", "activation"),
            ("BBC3", "activation"),
        ])
        symbol_to_feature = {"MDM2": "U1", "BAX": "U2", "BBC3": "U3"}
        with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.clique_analysis"):
            modules_to_clique_definitions(
                [module], symbol_to_feature, verbose=False
            )
        unmapped_warnings = [
            r for r in caplog.records if "unmapped" in r.message
        ]
        assert len(unmapped_warnings) == 0


# ---------------------------------------------------------------------------
# CoGEx resolve_gene_name case normalization tests
# ---------------------------------------------------------------------------

class TestResolveGeneNameCaseNormalization:
    """Verify resolve_gene_name handles case-insensitive gene symbols."""

    def test_uppercase_resolves(self, _mock_indra):
        """Standard upper-case gene symbol resolves via HGNC client."""
        from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor

        _mock_indra.get_current_hgnc_id = MagicMock(
            side_effect=lambda name: "11998" if name == "TP53" else None
        )
        client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(client)

        result = extractor.resolve_gene_name("TP53")
        assert result == ("HGNC", "11998")

    def test_lowercase_resolves_via_upper_fallback(self, _mock_indra):
        """Lowercase 'tp53' should resolve via upper-case fallback to 'TP53'."""
        from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor

        _mock_indra.get_current_hgnc_id = MagicMock(
            side_effect=lambda name: "11998" if name == "TP53" else None
        )
        client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(client)

        result = extractor.resolve_gene_name("tp53")
        assert result == ("HGNC", "11998")

    def test_mixedcase_resolves_via_upper_fallback(self, _mock_indra):
        """Mixed-case 'Tp53' should resolve via upper-case fallback."""
        from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor

        _mock_indra.get_current_hgnc_id = MagicMock(
            side_effect=lambda name: "11998" if name == "TP53" else None
        )
        client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(client)

        result = extractor.resolve_gene_name("Tp53")
        assert result == ("HGNC", "11998")


# ---------------------------------------------------------------------------
# RegulatorClass E3_LIGASE and RECEPTOR_KINASE tests
# ---------------------------------------------------------------------------

class TestE3LigaseClass:
    """Verify E3_LIGASE regulator class returns the curated gene set."""

    def test_e3_ligase_class(self):
        from cliquefinder.knowledge.cogex import (
            RegulatorClass,
            get_regulator_class_genes,
            _E3_LIGASE_GENES,
        )
        result = get_regulator_class_genes({RegulatorClass.E3_LIGASE})
        assert result == _E3_LIGASE_GENES
        assert len(result) == 10
        # Spot-check known members
        assert "MDM2" in result
        assert "PARKIN" in result
        assert "VHL" in result


class TestReceptorKinaseClass:
    """Verify RECEPTOR_KINASE regulator class returns the curated gene set."""

    def test_receptor_kinase_class(self):
        from cliquefinder.knowledge.cogex import (
            RegulatorClass,
            get_regulator_class_genes,
            _RECEPTOR_KINASE_GENES,
        )
        result = get_regulator_class_genes({RegulatorClass.RECEPTOR_KINASE})
        assert result == _RECEPTOR_KINASE_GENES
        assert len(result) == 20
        # Spot-check known members
        assert "EGFR" in result
        assert "MET" in result
        assert "ALK" in result
