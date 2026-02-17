"""Tests for statement type presets and resolve_stmt_types()."""

import pytest

from cliquefinder.knowledge.cogex import (
    resolve_stmt_types,
    ALL_REGULATORY_TYPES,
    ACTIVATION_TYPES,
    REPRESSION_TYPES,
    PHOSPHORYLATION_TYPES,
    STMT_TYPE_PRESETS,
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
