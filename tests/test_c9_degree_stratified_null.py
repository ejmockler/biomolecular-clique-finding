"""Contract tests for the canonical F5a degree-matched anchor null."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_c9_degree_stratified_null.py"
SPEC = importlib.util.spec_from_file_location("c9_degree_stratified_null", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

DEFAULT_OUTPUT_JSON = MODULE.DEFAULT_OUTPUT_JSON
DEFAULT_OUTPUT_MD = MODULE.DEFAULT_OUTPUT_MD
DEFAULT_TERM_SNAPSHOT = MODULE.DEFAULT_TERM_SNAPSHOT
N_TERMS = MODULE.N_TERMS
PRIMARY_ANCHOR_SCOPE = MODULE.PRIMARY_ANCHOR_SCOPE
PRIMARY_DEGREE_METRIC = MODULE.PRIMARY_DEGREE_METRIC
SENSITIVITY_DEGREE_METRIC = MODULE.SENSITIVITY_DEGREE_METRIC
TERMS = MODULE.TERMS
_add_eightfold_threshold = MODULE._add_eightfold_threshold
_sha256 = MODULE._sha256
candidate_scores_for_members = MODULE.candidate_scores_for_members
monte_carlo_matched_null = MODULE.monte_carlo_matched_null


def test_reciprocal_degree_window_is_the_actual_may_rule() -> None:
    scores = {name: float(index) for index, name in enumerate("mabcd")}
    degrees = {"m": 100, "a": 79, "b": 80, "c": 125, "d": 126}

    members, candidates, counts = candidate_scores_for_members(
        ["m"], ["a", "b", "c", "d"], scores, degrees
    )

    assert members == ["m"]
    assert counts == [2]
    assert candidates[0].tolist() == [scores["b"], scores["c"]]


def test_zero_degree_members_match_only_zero_degree_controls() -> None:
    scores = {"member": 3.0, "zero": 1.0, "one": 2.0}
    degrees = {"member": 0, "zero": 0, "one": 1}

    members, candidates, counts = candidate_scores_for_members(
        ["member"], ["zero", "one"], scores, degrees
    )

    assert members == ["member"]
    assert counts == [1]
    assert candidates[0].tolist() == [1.0]


def test_monte_carlo_p_value_uses_plus_one_correction() -> None:
    result = monte_carlo_matched_null(
        observed_scores=np.asarray([5.0, 6.0]),
        candidate_scores=[np.asarray([0.0]), np.asarray([1.0])],
        n_replicates=999,
        rng=np.random.default_rng(42),
    )

    assert result["mean_score"]["observed"] == 5.5
    assert result["mean_score"]["null_mean"] == 0.5
    assert result["mean_score"]["empirical_p_greater"] == pytest.approx(0.001)
    assert result["median_score"]["empirical_p_greater"] == pytest.approx(0.001)


def test_fixed_list_threshold_multiplies_p_by_eight() -> None:
    rows = [
        {
            "mean_score": {
                "empirical_p_greater": 0.006,
                "difference": 1.0,
            },
            "median_score": {
                "empirical_p_greater": 0.00625,
                "difference": 1.0,
            },
        }
    ]

    _add_eightfold_threshold(rows)

    assert rows[0]["mean_score"]["eightfold_p"] == pytest.approx(0.048)
    assert rows[0]["mean_score"]["meets_eightfold_threshold"] is True
    assert rows[0]["median_score"]["eightfold_p"] == pytest.approx(0.05)
    assert rows[0]["median_score"]["meets_eightfold_threshold"] is False


def test_frozen_term_snapshot_matches_fixed_discovery_derived_list() -> None:
    snapshot = json.loads(DEFAULT_TERM_SNAPSHOT.read_text(encoding="utf-8"))
    expected_order = [term_id for _, term_id, _, _ in TERMS]

    assert snapshot["term_order"] == expected_order
    assert snapshot["selection_status"].startswith("terms discovered on this cohort")
    assert len(snapshot["terms"]) == N_TERMS == 8
    assert [row["term_id"] for row in snapshot["terms"]] == expected_order
    assert all(row["hgnc_ids"] for row in snapshot["terms"])
    assert all(row["uniprot_ids"] for row in snapshot["terms"])
    endpoint = snapshot["source"]["endpoint"]
    assert endpoint == "configured via environment; endpoint not serialized"
    assert not any(token in endpoint for token in ("://", "@", "?", "password"))


def test_published_degree_null_is_canonical_and_internally_consistent() -> None:
    artifact = json.loads(DEFAULT_OUTPUT_JSON.read_text(encoding="utf-8"))
    analysis = artifact["analysis"]

    assert "generated_at_utc" not in artifact
    assert not any(key.endswith("elapsed_seconds") for key in artifact["runtime"])
    assert analysis["input_scale"].startswith("log2(x+1)")
    assert analysis["path_traversal"] == "measured_only"
    assert analysis["max_hops"] == 2
    assert analysis["n_valid_anchors"] == 3117
    assert analysis["fixed_canonical_anchor_scope"] == PRIMARY_ANCHOR_SCOPE
    assert analysis["fixed_canonical_degree_metric"] == PRIMARY_DEGREE_METRIC
    assert analysis["sensitivity_degree_metric"] == SENSITIVITY_DEGREE_METRIC
    assert analysis["n_fixed_terms"] == 8
    assert analysis["term_selection_status"].startswith("discovery-derived")
    assert analysis["post_selection_fwer_guarantee"] is False
    assert analysis["selective_inference_correction"] is False
    assert "per-anchor slope_pvalue" in analysis["explicit_non_inputs"]
    assert "GSEA NES" in analysis["explicit_non_inputs"]
    assert artifact["inputs"]["analysis_script_sha256"] == _sha256(SCRIPT_PATH)
    assert artifact["inputs"]["term_snapshot_sha256"] == _sha256(DEFAULT_TERM_SNAPSHOT)

    expected_primary = {
        "C9 vs Sporadic": 7,
        "C9 vs Control": 7,
        "Sporadic vs Control": 0,
    }
    expected_may_legacy = {
        "C9 vs Sporadic": 8,
        "C9 vs Control": 8,
        "Sporadic vs Control": 1,
    }
    expected_robust_term_sizes = [60, 74, 55, 185, 106, 37, 10, 8]
    for label, block in artifact["contrasts"].items():
        assert block["design"]["transform"] == "log2"
        assert block["design"]["max_hops"] == 2
        assert block["n_valid_anchors"] == 3117
        assert block["n_scope_anchors"][PRIMARY_ANCHOR_SCOPE] == 1407
        assert len(block["results"]) == 4 * N_TERMS
        assert all(row["match_coverage"] == 1.0 for row in block["results"])
        robust_primary_rows = [
            row
            for row in block["results"]
            if row["anchor_scope"] == PRIMARY_ANCHOR_SCOPE
            and row["degree_metric"] == PRIMARY_DEGREE_METRIC
        ]
        assert [row["n_term_eligible"] for row in robust_primary_rows] == (
            expected_robust_term_sizes
        )
        assert (
            block["summary"][PRIMARY_ANCHOR_SCOPE][PRIMARY_DEGREE_METRIC][
                "mean_eightfold_threshold_count"
            ]
            == expected_primary[label]
        )
        assert (
            block["summary"][MODULE.LEGACY_ANCHOR_SCOPE][PRIMARY_DEGREE_METRIC][
                "may_legacy_mwu_eightfold_threshold_count"
            ]
            == expected_may_legacy[label]
        )
        if label.startswith("C9"):
            assert (
                block["summary"][MODULE.LEGACY_ANCHOR_SCOPE][PRIMARY_DEGREE_METRIC][
                    "mean_eightfold_threshold_count"
                ]
                == 8
            )
            vpr = next(
                row
                for row in block["results"]
                if row["anchor_scope"] == PRIMARY_ANCHOR_SCOPE
                and row["degree_metric"] == PRIMARY_DEGREE_METRIC
                and row["term_id"] == "reactome:R-HSA-180910"
            )
            assert vpr["n_term_matched"] == 8
            assert vpr["mean_score"]["meets_eightfold_threshold"] is False


def test_markdown_states_the_inferential_boundary() -> None:
    markdown = DEFAULT_OUTPUT_MD.read_text(encoding="utf-8")

    assert "This is not an analysis of `slope_pvalue`" in markdown
    assert "it is not GSEA" in markdown
    assert "discovery on the same cohort" in markdown
    assert "not a post-selection FWER guarantee" in markdown
    assert "does not rule that explanation out" in markdown
    assert "cannot establish network independence" in markdown
    assert "C9 vs Sporadic" in markdown
    assert "Sporadic vs Control" in markdown


def test_noncanonical_run_cannot_overwrite_publication_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alternate --output-json"):
        MODULE.main(["--n-replicates", "999"])
    with pytest.raises(ValueError, match="alternate --output-json"):
        MODULE.main(["--degree-meta", str(tmp_path / "distances.meta.json")])


def test_degree_artifact_is_deterministic_when_local_inputs_exist() -> None:
    local_inputs = [
        *MODULE.DEFAULT_RESULTS.values(),
        MODULE.DEFAULT_DEGREE_META,
    ]
    if not all(path.exists() for path in local_inputs):
        pytest.skip("local canonical landscape inputs are absent")

    first = MODULE.build_artifact(
        MODULE.DEFAULT_RESULTS,
        MODULE.DEFAULT_DEGREE_META,
        DEFAULT_TERM_SNAPSHOT,
        999,
        MODULE.BASE_SEED,
    )
    second = MODULE.build_artifact(
        MODULE.DEFAULT_RESULTS,
        MODULE.DEFAULT_DEGREE_META,
        DEFAULT_TERM_SNAPSHOT,
        999,
        MODULE.BASE_SEED,
    )

    assert json.dumps(first, indent=2) == json.dumps(second, indent=2)


def test_external_paths_are_portable(tmp_path: Path) -> None:
    external = tmp_path / "result.json"
    assert MODULE._portable_path(external) == str(external.resolve())
