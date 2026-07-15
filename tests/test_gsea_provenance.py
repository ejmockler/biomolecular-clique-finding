"""Tests for the offline canonical GSEA provenance snapshot."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "snapshot_gsea_provenance.py"
SNAPSHOT_PATH = ROOT / "data" / "publication" / "c9_gsea_provenance.json"
SPEC = importlib.util.spec_from_file_location("snapshot_gsea_provenance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)

CONTRAST_ORDER = ["C9 vs Sporadic", "C9 vs Control", "Sporadic vs Control"]
EXPECTED_FDR_COUNTS = {
    "bounded_h2": {
        "C9 vs Sporadic": {"go": 161, "reactome": 114, "wikipathways": 9, "phenotype": 0},
        "C9 vs Control": {"go": 157, "reactome": 64, "wikipathways": 13, "phenotype": 26},
        "Sporadic vs Control": {"go": 0, "reactome": 0, "wikipathways": 0, "phenotype": 0},
    },
    "unbounded": {
        "C9 vs Sporadic": {"go": 365, "reactome": 106, "wikipathways": 20, "phenotype": 47},
        "C9 vs Control": {"go": 38, "reactome": 42, "wikipathways": 19, "phenotype": 0},
        "Sporadic vs Control": {"go": 58, "reactome": 32, "wikipathways": 13, "phenotype": 0},
    },
}
EXPECTED_ROW_COUNTS = {
    "go": 7165,
    "reactome": 1952,
    "wikipathways": 731,
    "phenotype": 5474,
}


def load_snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_frozen_snapshot_records_complete_canonical_contract():
    snapshot = load_snapshot()
    assert snapshot["schema_version"] == 1
    assert snapshot["contrast_order"] == CONTRAST_ORDER
    assert snapshot["configuration"]["intensity_transform"] == "log2(x+1)"
    assert snapshot["configuration"]["path_traversal"] == "measured_only"
    assert snapshot["configuration"]["gsea"] == {
        "scope": "robust",
        "scope_definition": "hop-1 measured neighborhood size >= 20",
        "score": "negative landscape slope",
        "permutation_num": 1000,
        "weighted_score_type": 1.0,
        "drop_top_pct": 0.0,
        "min_size": 1,
        "databases": ["go", "reactome", "wikipathways", "phenotype"],
        "fdr_rule": "FDR q-val < 0.05 (strict)",
        "count_unit": (
            "database-term rows; overlapping terms across databases are not "
            "unique or independent pathways"
        ),
        "seed_argument": None,
        "reference_gseapy_1_2_1_default_seed": 123,
        "historical_effective_seed_boundary": (
            "No seed argument was passed. The effective producer default and package "
            "revision were not embedded in the retained July outputs."
        ),
    }

    assert snapshot["regimes"]["bounded_h2"][
        "fdr_q_lt_0_05_summed_database_term_row_pattern"
    ] == [284, 260, 0]
    assert snapshot["regimes"]["bounded_h2"][
        "fixed_eight_term_pass_pattern"
    ] == [8, 6, 0]
    assert snapshot["regimes"]["unbounded"][
        "fixed_eight_term_pass_pattern"
    ] == [6, 0, 0]

    sha256 = re.compile(r"[0-9a-f]{64}")
    artifact_paths: set[str] = set()
    for regime_name, expected_contrasts in EXPECTED_FDR_COUNTS.items():
        regime = snapshot["regimes"][regime_name]
        for contrast in CONTRAST_ORDER:
            record = regime["contrasts"][contrast]
            observed_counts = {
                database: record["robust_gsea"][database]["fdr_q_lt_0_05"]
                for database in provenance.DATABASES
            }
            assert observed_counts == expected_contrasts[contrast]
            assert sum(observed_counts.values()) == record[
                "fdr_q_lt_0_05_summed_database_term_rows"
            ]
            for database, artifact in record["robust_gsea"].items():
                assert artifact["rows"] == EXPECTED_ROW_COUNTS[database]
                assert sha256.fullmatch(artifact["sha256"])
                artifact_paths.add(artifact["path"])

            fixed = record["fixed_eight_term_panel"]
            assert fixed["rows"] == fixed["found_rows"] == 8
            assert len(fixed["passing_term_ids"]) == fixed["pass_count"]
            assert sha256.fullmatch(fixed["sha256"])
            artifact_paths.add(fixed["path"])

            landscape = record["landscape"]
            assert landscape["design"]["transform"] == "log2"
            assert landscape["distance_path_traversal"] == "measured_only"
            for key in ("result", "distance_metadata", "input_manifest", "run_manifest"):
                assert sha256.fullmatch(landscape[key]["sha256"])
                artifact_paths.add(landscape[key]["path"])

    # 24 robust database CSVs + 6 fixed panels + 24 landscape support files.
    assert len(artifact_paths) == 54

    reference = snapshot["reference_implementation"]
    assert "cannot prove the exact historical producer revision" in reference[
        "relationship_to_outputs"
    ]
    assert len(reference["scripts"]) >= 8
    assert all(sha256.fullmatch(item["sha256"]) for item in reference["scripts"])
    assert [item["path"] for item in reference["environment_files"]] == [
        "pyproject.toml",
        "uv.lock",
    ]
    assert all(
        sha256.fullmatch(item["sha256"])
        for item in reference["environment_files"]
    )

    boundary = snapshot["reproducibility_boundary"]
    assert boundary["exact_upstream_regeneration_guaranteed"] is False
    assert "live CoGEx corpus/version" in boundary["limitation"]
    assert "effective GSEA seed/default" in boundary["limitation"]
    assert "producer package revisions" in boundary["limitation"]
    assert "without network access" in boundary["offline_guarantee"]


def test_frozen_snapshot_uses_canonical_serialization():
    snapshot = load_snapshot()
    assert SNAPSHOT_PATH.read_text(encoding="utf-8") == provenance.serialized(snapshot)


def test_csv_summarizers_recompute_strict_thresholds(tmp_path: Path):
    gsea_path = tmp_path / "gsea.csv"
    with gsea_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Term", "FDR q-val"])
        writer.writeheader()
        writer.writerows(
            [
                {"Term": "a", "FDR q-val": "0.049999"},
                {"Term": "b", "FDR q-val": "0.05"},
                {"Term": "c", "FDR q-val": "0.5"},
            ]
        )
    gsea = provenance.summarize_gsea_csv(tmp_path, "gsea.csv")
    assert gsea["rows"] == 3
    assert gsea["fdr_q_lt_0_05"] == 1

    fixed_path = tmp_path / "fixed.csv"
    fieldnames = ["term_id", "found", "NES", "raw_p", "bonferroni_pass"]
    with fixed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(8):
            passes = index < 3
            writer.writerow(
                {
                    "term_id": f"term:{index}",
                    "found": "True",
                    "NES": "1.0",
                    "raw_p": "0.001" if passes else "0.00625",
                    "bonferroni_pass": str(passes),
                }
            )
    fixed = provenance.summarize_fixed_term_csv(tmp_path, "fixed.csv")
    assert fixed["rows"] == fixed["found_rows"] == 8
    assert fixed["pass_count"] == 3
    assert fixed["passing_term_ids"] == ["term:0", "term:1", "term:2"]


def test_fixed_term_summarizer_rejects_stored_decision_drift(tmp_path: Path):
    path = tmp_path / "fixed.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["term_id", "found", "NES", "raw_p", "bonferroni_pass"],
        )
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "term_id": f"term:{index}",
                    "found": "True",
                    "NES": "1.0",
                    "raw_p": "0.001",
                    "bonferroni_pass": "False" if index == 0 else "True",
                }
            )
    with pytest.raises(provenance.ProvenanceError, match="decisions differ"):
        provenance.summarize_fixed_term_csv(tmp_path, "fixed.csv")


def test_local_retained_artifacts_rebuild_the_frozen_snapshot():
    first_local_artifact = (
        ROOT
        / "output/landscape_gsea_c9spor_measured_only_log2/robust_go.csv"
    )
    if not first_local_artifact.exists():
        pytest.skip("local ignored GSEA artifacts are absent")
    assert provenance.build_snapshot(ROOT) == load_snapshot()
