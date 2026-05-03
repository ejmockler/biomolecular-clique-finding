"""select_panel: determinism, stratification, exclusion of target."""
from __future__ import annotations

import pytest

from cliquefinder.panels import select_panel


# Synthetic candidates pool (mimics the find_matched_seed output).
POOL = {g: {"degree": 3700} for g in [
    "HNRNPK", "G3BP1", "HNRNPH1", "HNRNPA0",      # RNA
    "VCL", "MSN", "ACTN1", "FLNA",                # Cytoskeletal
    "SMAD1", "TP53BP1", "WDR5", "XRCC6",          # Transcription
    "ACLY", "AIFM1", "XDH", "HUWE1",              # Metabolic
]}

STRATA_DEFS = {
    "RNA_RBP":       ["HNRNPK", "G3BP1", "HNRNPH1", "HNRNPA0"],
    "Cytoskeletal":  ["VCL", "MSN", "ACTN1", "FLNA"],
    "Transcription": ["SMAD1", "TP53BP1", "WDR5", "XRCC6"],
    "Metabolic":     ["ACLY", "AIFM1", "XDH", "HUWE1"],
}


class TestSelection:
    def test_returns_correct_strata_count(self):
        design = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("C9ORF72", "SPORADIC"),
        )
        assert len(design.strata) == 4
        for s in design.strata:
            assert len(s.members) == 2

    def test_strata_named_correctly(self):
        design = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("C9ORF72", "SPORADIC"),
        )
        names = [s.name for s in design.strata]
        assert names == ["RNA_RBP", "Cytoskeletal", "Transcription", "Metabolic"]

    def test_members_only_from_intersection(self):
        """Selected members must be in BOTH the pool AND the stratum definition."""
        # Drop some genes from the pool — selection must respect the intersection.
        small_pool = {g: {} for g in ["HNRNPK", "G3BP1", "VCL", "ACLY"]}
        small_strata = {
            "RNA_RBP": ["HNRNPK", "G3BP1", "HNRNPH1"],
            "Cytoskeletal": ["VCL", "MSN", "ACTN1"],
            "Metabolic": ["ACLY", "AIFM1"],
        }
        design = select_panel(
            candidates_pool=small_pool,
            strata_definitions=small_strata,
            n_per_stratum=1,
            target_seed="C9orf72",
            contrast=("a", "b"),
        )
        for s in design.strata:
            for m in s.members:
                assert m in small_pool

    def test_target_excluded_even_if_in_definition(self):
        """If target appears in a stratum definition or pool, it must NOT be selected."""
        pool = {**POOL, "C9orf72": {}}
        strata = {**STRATA_DEFS, "RNA_RBP": ["C9orf72", "HNRNPK", "G3BP1"]}
        design = select_panel(
            candidates_pool=pool,
            strata_definitions=strata,
            n_per_stratum=1,
            target_seed="C9orf72",
            contrast=("a", "b"),
        )
        for s in design.strata:
            assert "C9orf72" not in s.members


class TestDeterminism:
    def test_same_seed_same_output(self):
        a = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=42,
        )
        b = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=42,
        )
        assert a == b
        assert a.selected_seeds() == b.selected_seeds()

    def test_different_seed_different_output(self):
        """Distinct rng seeds should usually produce distinct selections.

        Not guaranteed for tiny pools, but with our 4-from-4-eligible
        case the selection is fixed; use a 2-from-4 case where there
        are multiple possible outcomes.
        """
        a = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=1,
        )
        b = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=999,
        )
        # At least one stratum's members should differ.
        assert a.selected_seeds() != b.selected_seeds()

    def test_dict_ordering_does_not_affect_output(self):
        """Strata insertion order is honored, but member order within a
        stratum definition must not influence the selected set.
        """
        strata_a = {
            "RNA_RBP": ["HNRNPK", "G3BP1", "HNRNPH1", "HNRNPA0"],
        }
        strata_b = {
            "RNA_RBP": ["HNRNPA0", "HNRNPH1", "G3BP1", "HNRNPK"],
        }
        a = select_panel(
            candidates_pool=POOL,
            strata_definitions=strata_a,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=42,
        )
        b = select_panel(
            candidates_pool=POOL,
            strata_definitions=strata_b,
            n_per_stratum=2,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=42,
        )
        assert a.selected_seeds() == b.selected_seeds()


class TestSelectionErrors:
    def test_too_few_eligible_raises(self):
        with pytest.raises(ValueError, match="< 5 required"):
            select_panel(
                candidates_pool=POOL,
                strata_definitions={"RNA_RBP": ["HNRNPK", "G3BP1"]},
                n_per_stratum=5,
                target_seed="C9orf72",
                contrast=("a", "b"),
            )

    def test_zero_per_stratum_raises(self):
        with pytest.raises(ValueError, match="n_per_stratum must be >= 1"):
            select_panel(
                candidates_pool=POOL,
                strata_definitions=STRATA_DEFS,
                n_per_stratum=0,
                target_seed="C9orf72",
                contrast=("a", "b"),
            )

    def test_no_strata_raises(self):
        with pytest.raises(ValueError, match="strata_definitions must be non-empty"):
            select_panel(
                candidates_pool=POOL,
                strata_definitions={},
                n_per_stratum=1,
                target_seed="C9orf72",
                contrast=("a", "b"),
            )


class TestMembersAreSorted:
    def test_each_stratum_members_alphabetical(self):
        design = select_panel(
            candidates_pool=POOL,
            strata_definitions=STRATA_DEFS,
            n_per_stratum=3,
            target_seed="C9orf72",
            contrast=("a", "b"),
            selection_rng_seed=7,
        )
        for s in design.strata:
            assert list(s.members) == sorted(s.members)
