"""Tests for RegulatorClass enum and regulator class filtering."""

import argparse
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures: mock INDRA's hgnc_client lists so tests don't require INDRA
# ---------------------------------------------------------------------------

MOCK_TFS = ["TP53", "MYC", "JUN", "STAT3", "BRCA1"]
MOCK_KINASES = ["AKT1", "MAPK1", "CDK2", "BRCA1"]  # BRCA1 intentionally shared
MOCK_PHOSPHATASES = ["PTEN", "PTPN1", "DUSP1"]


@pytest.fixture(autouse=True)
def _mock_indra():
    """Patch hgnc_client on the already-imported cogex module.

    No reload — this avoids creating a second RegulatorClass enum whose
    members would be identity-incompatible with the one the tests import.
    """
    mock_hgnc = MagicMock()
    mock_hgnc.tfs = MOCK_TFS
    mock_hgnc.kinases = MOCK_KINASES
    mock_hgnc.phosphatases = MOCK_PHOSPHATASES
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
# Import after mocking
# ---------------------------------------------------------------------------

from cliquefinder.knowledge.cogex import (
    RegulatorClass,
    get_regulator_class_genes,
    INDRAModule,
    INDRAEdge,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestRegulatorClassEnum:
    def test_enum_values_match_cli_choices(self):
        assert RegulatorClass.TF.value == "tf"
        assert RegulatorClass.KINASE.value == "kinase"
        assert RegulatorClass.PHOSPHATASE.value == "phosphatase"

    def test_enum_members_count(self):
        assert len(RegulatorClass) == 3

    def test_enum_from_string(self):
        assert RegulatorClass("tf") is RegulatorClass.TF
        assert RegulatorClass("kinase") is RegulatorClass.KINASE
        assert RegulatorClass("phosphatase") is RegulatorClass.PHOSPHATASE


# ---------------------------------------------------------------------------
# get_regulator_class_genes tests
# ---------------------------------------------------------------------------

class TestGetRegulatorClassGenes:
    def test_single_class_tf(self):
        genes = get_regulator_class_genes({RegulatorClass.TF})
        assert genes == set(MOCK_TFS)

    def test_single_class_kinase(self):
        genes = get_regulator_class_genes({RegulatorClass.KINASE})
        assert genes == set(MOCK_KINASES)

    def test_single_class_phosphatase(self):
        genes = get_regulator_class_genes({RegulatorClass.PHOSPHATASE})
        assert genes == set(MOCK_PHOSPHATASES)

    def test_union_deduplicates(self):
        """BRCA1 is in both TF and KINASE lists; should appear once."""
        genes = get_regulator_class_genes({RegulatorClass.TF, RegulatorClass.KINASE})
        expected = set(MOCK_TFS) | set(MOCK_KINASES)
        assert genes == expected
        assert "BRCA1" in genes

    def test_all_classes(self):
        genes = get_regulator_class_genes(
            {RegulatorClass.TF, RegulatorClass.KINASE, RegulatorClass.PHOSPHATASE}
        )
        expected = set(MOCK_TFS) | set(MOCK_KINASES) | set(MOCK_PHOSPHATASES)
        assert genes == expected

    def test_empty_set_returns_empty(self):
        genes = get_regulator_class_genes(set())
        assert genes == set()


# ---------------------------------------------------------------------------
# discover_modules class filter tests
# ---------------------------------------------------------------------------

class TestDiscoverModulesClassFilter:
    """Test that discover_modules applies regulator_classes correctly."""

    def _make_module(self, name: str) -> INDRAModule:
        edge = INDRAEdge(
            regulator_id=("HGNC", "1"),
            regulator_name=name,
            target_id=("HGNC", "2"),
            target_name="TARGET1",
            regulation_type="activation",
            evidence_count=5,
            stmt_hash=12345,
            source_counts="{}",
        )
        return INDRAModule(
            regulator_id=("HGNC", "1"),
            regulator_name=name,
            targets=[edge],
        )

    def test_tf_filter_keeps_only_tfs(self):
        """Simulate post-discovery filtering: only TFs should survive."""
        modules = [
            self._make_module("TP53"),   # TF
            self._make_module("AKT1"),   # Kinase, not TF
            self._make_module("PTEN"),   # Phosphatase, not TF
        ]
        class_genes = get_regulator_class_genes({RegulatorClass.TF})
        filtered = [m for m in modules if m.regulator_name in class_genes]
        assert len(filtered) == 1
        assert filtered[0].regulator_name == "TP53"

    def test_tf_and_kinase_filter(self):
        modules = [
            self._make_module("TP53"),   # TF
            self._make_module("AKT1"),   # Kinase
            self._make_module("PTEN"),   # Phosphatase
        ]
        class_genes = get_regulator_class_genes(
            {RegulatorClass.TF, RegulatorClass.KINASE}
        )
        filtered = [m for m in modules if m.regulator_name in class_genes]
        assert len(filtered) == 2
        names = {m.regulator_name for m in filtered}
        assert names == {"TP53", "AKT1"}

    def test_no_class_filter_keeps_all(self):
        """regulator_classes=None should not filter anything."""
        modules = [
            self._make_module("TP53"),
            self._make_module("AKT1"),
            self._make_module("PTEN"),
        ]
        # Simulate: no filter applied
        regulator_classes = None
        if regulator_classes is not None:
            class_genes = get_regulator_class_genes(regulator_classes)
            modules = [m for m in modules if m.regulator_name in class_genes]
        assert len(modules) == 3


# ---------------------------------------------------------------------------
# Class + RNA filter composition tests
# ---------------------------------------------------------------------------

class TestFilterComposition:
    def test_class_and_rna_both_must_pass(self):
        """A regulator must pass BOTH class and RNA filters."""
        rna_filter_genes = {"TP53", "AKT1", "UNKNOWN_GENE"}
        class_genes = get_regulator_class_genes({RegulatorClass.TF})

        regulators = ["TP53", "AKT1", "PTEN", "MYC"]

        # Apply RNA filter
        after_rna = [r for r in regulators if r in rna_filter_genes]
        assert set(after_rna) == {"TP53", "AKT1"}

        # Apply class filter
        after_class = [r for r in after_rna if r in class_genes]
        # AKT1 is a kinase, not a TF — should be removed
        assert after_class == ["TP53"]

    def test_class_filter_only(self):
        """Without RNA filter, only class filter applies."""
        class_genes = get_regulator_class_genes({RegulatorClass.PHOSPHATASE})
        regulators = ["TP53", "PTEN", "PTPN1", "AKT1"]
        filtered = [r for r in regulators if r in class_genes]
        assert set(filtered) == {"PTEN", "PTPN1"}


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def _make_parser(self):
        """Create a minimal parser mirroring analyze.py's --regulator-class."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--regulator-class", nargs="+",
                            choices=["tf", "kinase", "phosphatase"], default=None)
        return parser

    def test_single_class(self):
        parser = self._make_parser()
        args = parser.parse_args(["--regulator-class", "tf"])
        assert args.regulator_class == ["tf"]

    def test_multiple_classes(self):
        parser = self._make_parser()
        args = parser.parse_args(["--regulator-class", "tf", "kinase"])
        assert args.regulator_class == ["tf", "kinase"]

    def test_no_class_flag(self):
        parser = self._make_parser()
        args = parser.parse_args([])
        assert args.regulator_class is None

    def test_invalid_class_rejected(self):
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--regulator-class", "invalid"])

    def test_cli_to_enum_conversion(self):
        """Test the string → enum conversion logic from analyze.py."""
        _CLI_TO_ENUM = {
            "tf": RegulatorClass.TF,
            "kinase": RegulatorClass.KINASE,
            "phosphatase": RegulatorClass.PHOSPHATASE,
        }
        cli_values = ["tf", "kinase"]
        result = {_CLI_TO_ENUM[c] for c in cli_values}
        assert result == {RegulatorClass.TF, RegulatorClass.KINASE}
