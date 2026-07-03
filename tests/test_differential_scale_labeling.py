"""
Scale-labeling integrity tests for run_protein_differential.

The contrast effect (beta @ c) is a mean difference on whatever scale the input
`data` is on. On log2 intensities it is a genuine log2 fold change; on RAW linear
intensities (AnswerALS spans 0..5.3e10) it is a raw mean-difference that must NOT
be silently reported as a log2FC.

These tests pin the input-scale contract added to run_protein_differential:
    (a) raw-scale input (max ~1e6) -> UserWarning + effect_scale=='raw'
        (NOT a silent log2FC),
    (b) genuinely log2-scaled input (values ~10) -> effect_scale=='log2',
        no scale warning, log2fc/effect_size correct,
    (c) effect sign/direction is unchanged by the label-only fix, and
        effect_size is byte-identical to the legacy log2fc column,
    (d) declaring input_scale='log2' on obviously-raw data raises ValueError.

The statistical math (beta @ c, EB squeeze_var, SE, t, p) is untouched; only the
scale label is made honest.
"""

import warnings

import numpy as np
import pytest

from cliquefinder.stats.differential import (
    run_protein_differential,
    _infer_intensity_scale,
)


N_FEATURES = 60
N_PER_GROUP = 12


def _make_condition():
    return np.array(["CTRL"] * N_PER_GROUP + ["CASE"] * N_PER_GROUP)


def _feature_ids():
    return [f"P{i:03d}" for i in range(N_FEATURES)]


def _log2_data(seed=0):
    """Log2-scale matrix (values ~10) with injected up/down structure in CASE."""
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 1.0, (N_FEATURES, 2 * N_PER_GROUP))
    data[0:5, N_PER_GROUP:] += 2.0   # up in CASE
    data[5:10, N_PER_GROUP:] -= 2.0  # down in CASE
    return data


def _raw_data(seed=0):
    """Raw linear matrix (max ~1e6) with the SAME up/down sign structure."""
    rng = np.random.default_rng(seed)
    data = rng.normal(5.0e5, 5.0e4, (N_FEATURES, 2 * N_PER_GROUP))
    data[0:5, N_PER_GROUP:] += 2.0e5   # up in CASE
    data[5:10, N_PER_GROUP:] -= 2.0e5  # down in CASE
    return data


# ── Unit: the pure scale detector ───────────────────────────────────────────

class TestInferIntensityScale:
    def test_log2_valued_matrix_is_log2(self):
        assert _infer_intensity_scale(_log2_data()) == "log2"

    def test_raw_valued_matrix_is_raw(self):
        assert _infer_intensity_scale(_raw_data()) == "raw"

    def test_all_nan_defaults_to_log2(self):
        data = np.full((4, 4), np.nan)
        assert _infer_intensity_scale(data) == "log2"

    def test_threshold_boundary(self):
        # max just below 50 -> log2; just above -> raw
        low = np.array([[1.0, 49.0], [2.0, 3.0]])
        high = np.array([[1.0, 51.0], [2.0, 3.0]])
        assert _infer_intensity_scale(low) == "log2"
        assert _infer_intensity_scale(high) == "raw"

    def test_infinite_entries_ignored(self):
        data = np.array([[10.0, np.inf], [9.0, 11.0]])
        assert _infer_intensity_scale(data) == "log2"


# ── (a) raw input warns + effect_scale=='raw', not silent log2FC ────────────

class TestRawInputNotSilentlyLog2FC:
    def test_auto_raw_warns_and_flags(self):
        data = _raw_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                eb_moderation=True,
            )
        # effect_scale column present and honest
        assert "effect_scale" in result.columns
        assert (result["effect_scale"] == "raw").all()
        # A raw-scale warning was emitted (not a silent log2FC)
        raw_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "RAW" in str(w.message)
        ]
        assert raw_warnings, "raw-scale input must warn, not be silently labeled log2FC"

    def test_explicit_raw_declaration_flags(self):
        data = _raw_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                input_scale="raw",
            )
        assert (result["effect_scale"] == "raw").all()


# ── (b) log2 input accepted, labeled, no scale warning ──────────────────────

class TestLog2InputAcceptedAndLabeled:
    def test_auto_log2_no_warning_and_flagged(self):
        data = _log2_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                eb_moderation=True,
            )
        assert (result["effect_scale"] == "log2").all()
        raw_warnings = [w for w in caught if "RAW" in str(w.message)]
        assert not raw_warnings, "log2-scale input must not raise a raw-scale warning"

    def test_explicit_log2_declaration_accepted(self):
        data = _log2_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail here
            result = run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                eb_moderation=True,
                input_scale="log2",
            )
        assert (result["effect_scale"] == "log2").all()

    def test_log2fc_magnitude_recovers_injected_effect(self):
        data = _log2_data()
        result = run_protein_differential(
            data=data,
            feature_ids=_feature_ids(),
            sample_condition=_make_condition(),
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            input_scale="log2",
        )
        up = result.iloc[0]      # injected +2.0
        assert up["log2fc"] == pytest.approx(2.0, abs=0.8)


# ── (c) sign/direction unchanged; effect_size mirrors log2fc byte-for-byte ──

class TestEffectSignAndValueUnchanged:
    def test_up_and_down_signs(self):
        for scale_data, declared in [(_log2_data(), "log2"), (_raw_data(), "raw")]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_protein_differential(
                    data=scale_data,
                    feature_ids=_feature_ids(),
                    sample_condition=_make_condition(),
                    contrast=("CASE", "CTRL"),
                    input_scale=declared,
                )
            # Features 0-4 up in CASE -> positive; 5-9 down -> negative
            assert (result.iloc[0:5]["log2fc"] > 0).all()
            assert (result.iloc[5:10]["log2fc"] < 0).all()

    def test_effect_size_is_byte_identical_to_log2fc(self):
        data = _log2_data()
        result = run_protein_differential(
            data=data,
            feature_ids=_feature_ids(),
            sample_condition=_make_condition(),
            contrast=("CASE", "CTRL"),
            input_scale="log2",
        )
        # Label-only change: effect_size mirrors the legacy log2fc column exactly.
        assert np.array_equal(
            result["effect_size"].to_numpy(),
            result["log2fc"].to_numpy(),
            equal_nan=True,
        )

    def test_input_scale_param_does_not_change_math(self):
        """The scale label must not perturb beta@c, t, p, or df."""
        data = _log2_data()
        common = dict(
            feature_ids=_feature_ids(),
            sample_condition=_make_condition(),
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
        )
        auto = run_protein_differential(data=data, input_scale="auto", **common)
        declared = run_protein_differential(data=data, input_scale="log2", **common)
        for col in ("log2fc", "effect_size", "t_statistic", "p_value", "df"):
            assert np.array_equal(
                auto[col].to_numpy(), declared[col].to_numpy(), equal_nan=True
            ), f"column {col} changed with input_scale"


# ── (d) declared log2 on obviously-raw data raises ──────────────────────────

class TestDeclaredLog2OnRawRaises:
    def test_raises_value_error(self):
        data = _raw_data()
        with pytest.raises(ValueError, match="RAW linear scale"):
            run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                input_scale="log2",
            )

    def test_invalid_input_scale_rejected(self):
        data = _log2_data()
        with pytest.raises(ValueError, match="input_scale must be one of"):
            run_protein_differential(
                data=data,
                feature_ids=_feature_ids(),
                sample_condition=_make_condition(),
                contrast=("CASE", "CTRL"),
                input_scale="ln",
            )
