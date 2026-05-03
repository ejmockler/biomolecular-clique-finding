"""PanelDesign + PanelStratum: validation, serialization, round-trip."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cliquefinder.panels import PanelDesign, PanelStratum


def _good_design() -> PanelDesign:
    return PanelDesign(
        target_seed="C9orf72",
        strata=(
            PanelStratum(name="RNA_RBP", members=("HNRNPK", "G3BP1")),
            PanelStratum(name="Cytoskeletal", members=("VCL", "MSN")),
        ),
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=2,
        n_permutations=999,
        covariates=("Sex",),
        selection_rng_seed=42,
        description="test",
    )


# --- PanelStratum validation ------------------------------------------------


class TestPanelStratumValidation:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            PanelStratum(name="", members=("A",))

    def test_rejects_no_members(self):
        with pytest.raises(ValueError, match="at least one member"):
            PanelStratum(name="X", members=())

    def test_rejects_duplicate_members(self):
        with pytest.raises(ValueError, match="duplicate members"):
            PanelStratum(name="X", members=("A", "A"))


# --- PanelDesign validation -------------------------------------------------


class TestPanelDesignValidation:
    def test_rejects_empty_target(self):
        with pytest.raises(ValueError, match="target_seed must be non-empty"):
            PanelDesign(
                target_seed="",
                strata=(PanelStratum("X", ("A",)),),
                contrast=("a", "b"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_no_strata(self):
        with pytest.raises(ValueError, match="at least one stratum"):
            PanelDesign(
                target_seed="C9orf72", strata=(),
                contrast=("a", "b"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_degenerate_contrast(self):
        with pytest.raises(ValueError, match="2-tuple of distinct"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(PanelStratum("X", ("A",)),),
                contrast=("a", "a"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_bad_max_hops(self):
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(PanelStratum("X", ("A",)),),
                contrast=("a", "b"), max_hops=0, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_bad_n_permutations(self):
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(PanelStratum("X", ("A",)),),
                contrast=("a", "b"), max_hops=2, n_permutations=0,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_duplicate_stratum_names(self):
        with pytest.raises(ValueError, match="duplicate stratum names"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(
                    PanelStratum("X", ("A",)),
                    PanelStratum("X", ("B",)),
                ),
                contrast=("a", "b"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_seed_in_two_strata(self):
        with pytest.raises(ValueError, match="multiple strata"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(
                    PanelStratum("X", ("A", "B")),
                    PanelStratum("Y", ("B", "C")),
                ),
                contrast=("a", "b"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )

    def test_rejects_target_in_stratum(self):
        with pytest.raises(ValueError, match="target_seed.*must not appear"):
            PanelDesign(
                target_seed="C9orf72",
                strata=(PanelStratum("X", ("C9orf72", "VCL")),),
                contrast=("a", "b"), max_hops=2, n_permutations=999,
                covariates=(), selection_rng_seed=0,
            )


# --- PanelDesign API --------------------------------------------------------


class TestPanelDesignAPI:
    def test_selected_seeds_preserves_stratum_order(self):
        d = _good_design()
        assert d.selected_seeds() == ("HNRNPK", "G3BP1", "VCL", "MSN")

    def test_stratum_for_known_seed(self):
        d = _good_design()
        assert d.stratum_for("HNRNPK") == "RNA_RBP"
        assert d.stratum_for("MSN") == "Cytoskeletal"

    def test_stratum_for_unknown_seed_raises(self):
        d = _good_design()
        with pytest.raises(KeyError, match="not a panel member"):
            d.stratum_for("NOTAPANEL")

    def test_stratum_for_target_raises(self):
        """Target is implicit; not a stratum member."""
        d = _good_design()
        with pytest.raises(KeyError, match="not a panel member"):
            d.stratum_for("C9orf72")


# --- Serialization round-trip ----------------------------------------------


class TestPanelDesignSerialization:
    def test_to_dict_then_from_dict_round_trips(self):
        d = _good_design()
        recovered = PanelDesign.from_dict(d.to_dict())
        assert recovered == d

    def test_yaml_round_trip(self, tmp_path: Path):
        d = _good_design()
        manifest_path = tmp_path / "manifest.yaml"
        d.save_yaml(manifest_path)
        recovered = PanelDesign.load_yaml(manifest_path)
        assert recovered == d

    def test_yaml_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PanelDesign.load_yaml(tmp_path / "nonexistent.yaml")

    def test_yaml_load_non_mapping_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- not a mapping\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            PanelDesign.load_yaml(bad)

    def test_yaml_format_is_block_style(self, tmp_path: Path):
        """Block-style YAML (no flow style) for human readability of the manifest."""
        d = _good_design()
        manifest_path = tmp_path / "manifest.yaml"
        d.save_yaml(manifest_path)
        text = manifest_path.read_text()
        # Block style: stratum names on their own indented lines, not {name: X, members: [...]}
        assert "name: RNA_RBP" in text
        assert "{" not in text  # no flow-style mappings

    def test_from_dict_rejects_bad_contrast(self):
        d = _good_design().to_dict()
        d["contrast"] = ["only_one"]
        with pytest.raises(ValueError, match="2-element sequence"):
            PanelDesign.from_dict(d)


# --- Frozen-ness ------------------------------------------------------------


class TestFrozen:
    def test_panel_design_is_immutable(self):
        d = _good_design()
        with pytest.raises(Exception):  # FrozenInstanceError
            d.target_seed = "OTHER"  # type: ignore[misc]

    def test_panel_stratum_is_immutable(self):
        s = PanelStratum("X", ("A",))
        with pytest.raises(Exception):
            s.name = "Y"  # type: ignore[misc]

    def test_strata_is_tuple_not_list(self):
        d = _good_design()
        assert isinstance(d.strata, tuple)
        assert isinstance(d.strata[0].members, tuple)
        assert isinstance(d.covariates, tuple)
