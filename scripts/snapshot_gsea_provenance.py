#!/usr/bin/env python3
"""Freeze or verify the canonical C9 robust-GSEA provenance snapshot.

This command is deliberately offline: it only reads retained repository files.
It hashes the bounded and unbounded landscape inputs, every robust GSEA CSV,
and every fixed-eight-term CSV, then recomputes the reported row counts and
threshold patterns.

Usage::

    python scripts/snapshot_gsea_provenance.py --verify
    python scripts/snapshot_gsea_provenance.py --write
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = Path("data/publication/c9_gsea_provenance.json")

DATABASES = ("go", "reactome", "wikipathways", "phenotype")
CONTRASTS = (
    ("C9 vs Sporadic", "c9spor", "landscape_proteome", ("C9ORF72", "SPORADIC")),
    ("C9 vs Control", "c9ctrl", "landscape_c9_vs_control", ("C9ORF72", "CONTROL")),
    (
        "Sporadic vs Control",
        "spctrl",
        "landscape_sporadic_vs_control",
        ("SPORADIC", "CONTROL"),
    ),
)
REGIMES = {
    "bounded_h2": {
        "suffix": "measured_only_log2",
        "max_hops": 2,
        "valid_gradients": 3117,
        "undersized": 10,
    },
    "unbounded": {
        "suffix": "measured_only_unbounded",
        "max_hops": None,
        "valid_gradients": 3125,
        "undersized": 2,
    },
}
EXPECTED_FDR_TOTALS = {
    "bounded_h2": [284, 260, 0],
    "unbounded": [538, 99, 103],
}
EXPECTED_FIXED_TERM_PATTERNS = {
    "bounded_h2": [8, 6, 0],
    "unbounded": [6, 0, 0],
}
REFERENCE_SCRIPTS = (
    ("scripts/run_landscape_log2.py", "bounded log2 landscape driver"),
    ("scripts/run_landscape_proteome.py", "unbounded C9-vs-Sporadic landscape driver"),
    (
        "scripts/run_landscape_c9_vs_control.py",
        "unbounded C9-vs-Control landscape driver",
    ),
    (
        "scripts/run_landscape_sporadic_vs_control.py",
        "unbounded Sporadic-vs-Control landscape driver",
    ),
    ("scripts/run_landscape_gsea.py", "robust GSEA producer"),
    ("scripts/analyze_landscape_confirmatory.py", "fixed-term CSV producer"),
    ("scripts/run_log2_downstream.sh", "bounded downstream orchestration"),
    ("scripts/run_unbounded_downstream.sh", "unbounded downstream orchestration"),
    ("scripts/snapshot_gsea_provenance.py", "offline snapshot generator and verifier"),
)
REFERENCE_ENVIRONMENT_FILES = (
    ("pyproject.toml", "declared direct and optional dependency contract"),
    ("uv.lock", "frozen uv dependency graph and Git revisions"),
)


class ProvenanceError(RuntimeError):
    """Raised when retained artifacts violate the canonical contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, relative_path: str) -> dict[str, Any]:
    """Describe one repository-relative artifact."""
    path = root / relative_path
    if not path.is_file():
        raise ProvenanceError(f"required artifact is absent: {relative_path}")
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries, rejecting a missing header."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProvenanceError(f"CSV has no header: {path}")
        return list(reader)


def summarize_gsea_csv(root: Path, relative_path: str) -> dict[str, Any]:
    """Hash a GSEA CSV and recompute its strict FDR-q threshold count."""
    path = root / relative_path
    rows = read_csv_rows(path)
    if not rows or "FDR q-val" not in rows[0]:
        raise ProvenanceError(f"GSEA CSV lacks FDR q-val rows: {relative_path}")
    record = file_record(root, relative_path)
    record.update(
        {
            "rows": len(rows),
            "fdr_q_lt_0_05": sum(float(row["FDR q-val"]) < 0.05 for row in rows),
        }
    )
    return record


def parse_bool(value: str) -> bool:
    """Parse the explicit booleans written by pandas CSV output."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ProvenanceError(f"expected True/False, got {value!r}")


def summarize_fixed_term_csv(root: Path, relative_path: str) -> dict[str, Any]:
    """Hash an eight-term CSV and independently recompute its pass decisions."""
    path = root / relative_path
    rows = read_csv_rows(path)
    required = {"term_id", "found", "NES", "raw_p", "bonferroni_pass"}
    if not rows or not required.issubset(rows[0]):
        raise ProvenanceError(f"fixed-term CSV lacks required columns: {relative_path}")
    if len(rows) != 8 or len({row["term_id"] for row in rows}) != 8:
        raise ProvenanceError(f"fixed-term CSV is not an eight-unique-term panel: {relative_path}")

    passing_term_ids: list[str] = []
    for row in rows:
        recomputed = float(row["raw_p"]) < 0.00625 and float(row["NES"]) > 0
        if parse_bool(row["bonferroni_pass"]) is not recomputed:
            raise ProvenanceError(
                f"stored and recomputed fixed-term decisions differ in {relative_path}: "
                f"{row['term_id']}"
            )
        if recomputed:
            passing_term_ids.append(row["term_id"])

    record = file_record(root, relative_path)
    record.update(
        {
            "rows": len(rows),
            "found_rows": sum(parse_bool(row["found"]) for row in rows),
            "pass_count": len(passing_term_ids),
            "passing_term_ids": passing_term_ids,
        }
    )
    return record


def summarize_landscape(
    root: Path,
    directory: str,
    expected_contrast: tuple[str, str],
    expected_max_hops: int | None,
    expected_valid: int,
    expected_undersized: int,
) -> dict[str, Any]:
    """Bind a GSEA run to its local landscape result and distance metadata."""
    result_relative = f"output/{directory}/result.json"
    metadata_relative = f"output/{directory}/distances.meta.json"
    inputs_relative = f"output/{directory}/inputs.json"
    manifest_relative = f"output/{directory}/manifest.yaml"

    result = json.loads((root / result_relative).read_text(encoding="utf-8"))
    metadata = json.loads((root / metadata_relative).read_text(encoding="utf-8"))
    inputs = json.loads((root / inputs_relative).read_text(encoding="utf-8"))
    design = result.get("design", {})

    expected_design = {
        "contrast": list(expected_contrast),
        "max_hops": expected_max_hops,
        "n_permutations": 999,
        "covariates": ["Sex"],
        "transform": "log2",
    }
    observed_design = {key: design.get(key) for key in expected_design}
    if observed_design != expected_design:
        raise ProvenanceError(
            f"landscape design drift in {result_relative}: "
            f"expected {expected_design}, got {observed_design}"
        )
    accounting = {
        "attempted": int(result["n_features_input"]),
        "valid_gradients": len(result["per_feature"]),
        "disconnected": len(result["degenerate_features"]),
        "undersized": len(result["error_features"]),
    }
    expected_accounting = {
        "attempted": 3264,
        "valid_gradients": expected_valid,
        "disconnected": 137,
        "undersized": expected_undersized,
    }
    if accounting != expected_accounting:
        raise ProvenanceError(
            f"landscape accounting drift in {result_relative}: "
            f"expected {expected_accounting}, got {accounting}"
        )
    if metadata.get("path_traversal") != "measured_only":
        raise ProvenanceError(f"non-measured-only distance metadata: {metadata_relative}")
    if metadata.get("max_hops") != expected_max_hops:
        raise ProvenanceError(f"distance-depth drift in {metadata_relative}")
    if len(metadata.get("feature_names", [])) != 3264:
        raise ProvenanceError(f"distance feature-count drift in {metadata_relative}")

    recorded_inputs: dict[str, dict[str, Any]] = {}
    for name in ("data", "metadata"):
        item = inputs.get(name, {})
        recorded_inputs[name] = {
            "bytes": int(item["size"]),
            "sha256": item["sha256"],
        }

    return {
        "result": file_record(root, result_relative),
        "distance_metadata": file_record(root, metadata_relative),
        "input_manifest": file_record(root, inputs_relative),
        "run_manifest": file_record(root, manifest_relative),
        "design": observed_design,
        "distance_path_traversal": metadata["path_traversal"],
        "feature_accounting": accounting,
        "recorded_source_inputs": recorded_inputs,
    }


def build_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Build the deterministic snapshot from retained local artifacts."""
    source_inputs = {
        "proteomics_matrix": file_record(root, "output/proteomics/all_als.data.csv"),
        "sample_metadata": file_record(root, "output/proteomics/all_als.metadata.csv"),
    }
    reference_scripts = []
    for path, role in REFERENCE_SCRIPTS:
        record = file_record(root, path)
        record["role"] = role
        reference_scripts.append(record)
    reference_environment = []
    for path, role in REFERENCE_ENVIRONMENT_FILES:
        record = file_record(root, path)
        record["role"] = role
        reference_environment.append(record)

    regimes: dict[str, Any] = {}
    for regime_name, regime_config in REGIMES.items():
        contrast_records: dict[str, Any] = {}
        fdr_total_pattern: list[int] = []
        fixed_term_pattern: list[int] = []
        suffix = regime_config["suffix"]

        for label, tag, landscape_stem, groups in CONTRASTS:
            directory = f"{landscape_stem}_{suffix}"
            gsea_directory = f"output/landscape_gsea_{tag}_{suffix}"
            fixed_relative = (
                f"output/landscape_confirmatory_{tag}_{suffix}/"
                "confirmatory_8terms_robust.csv"
            )
            gsea = {
                database: summarize_gsea_csv(
                    root, f"{gsea_directory}/robust_{database}.csv"
                )
                for database in DATABASES
            }
            fdr_total = sum(item["fdr_q_lt_0_05"] for item in gsea.values())
            fixed_terms = summarize_fixed_term_csv(root, fixed_relative)
            fdr_total_pattern.append(fdr_total)
            fixed_term_pattern.append(fixed_terms["pass_count"])

            landscape = summarize_landscape(
                root,
                directory,
                groups,
                regime_config["max_hops"],
                regime_config["valid_gradients"],
                regime_config["undersized"],
            )
            for name, expected in landscape["recorded_source_inputs"].items():
                canonical_key = "proteomics_matrix" if name == "data" else "sample_metadata"
                canonical = source_inputs[canonical_key]
                if expected != {"bytes": canonical["bytes"], "sha256": canonical["sha256"]}:
                    raise ProvenanceError(
                        f"recorded input drift in output/{directory}/inputs.json: {name}"
                    )

            contrast_records[label] = {
                "landscape": landscape,
                "robust_gsea": gsea,
                "fdr_q_lt_0_05_summed_database_term_rows": fdr_total,
                "fixed_eight_term_panel": fixed_terms,
            }

        if fdr_total_pattern != EXPECTED_FDR_TOTALS[regime_name]:
            raise ProvenanceError(
                f"{regime_name} robust-GSEA FDR pattern drift: {fdr_total_pattern}"
            )
        if fixed_term_pattern != EXPECTED_FIXED_TERM_PATTERNS[regime_name]:
            raise ProvenanceError(
                f"{regime_name} fixed-term pattern drift: {fixed_term_pattern}"
            )
        regimes[regime_name] = {
            "max_hops": regime_config["max_hops"],
            "contrasts": contrast_records,
            "fdr_q_lt_0_05_summed_database_term_row_pattern": fdr_total_pattern,
            "fixed_eight_term_pass_pattern": fixed_term_pattern,
        }

    return {
        "schema_version": 1,
        "snapshot_id": "c9-log2-measured-only-robust-gsea-2026-07",
        "purpose": (
            "Offline byte-integrity and count verification for the retained canonical "
            "bounded/unbounded robust GSEA and discovery-derived fixed-eight-term outputs."
        ),
        "contrast_order": [label for label, *_ in CONTRASTS],
        "configuration": {
            "intensity_transform": "log2(x+1)",
            "path_traversal": "measured_only",
            "landscape_permutations": 999,
            "landscape_rng_seed": 42,
            "covariates": ["Sex"],
            "gsea": {
                "scope": "robust",
                "scope_definition": "hop-1 measured neighborhood size >= 20",
                "score": "negative landscape slope",
                "permutation_num": 1000,
                "weighted_score_type": 1.0,
                "drop_top_pct": 0.0,
                "min_size": 1,
                "databases": list(DATABASES),
                "fdr_rule": "FDR q-val < 0.05 (strict)",
                "count_unit": (
                    "database-term rows; overlapping terms across databases are not "
                    "unique or independent pathways"
                ),
                "seed_argument": None,
                "reference_gseapy_1_2_1_default_seed": 123,
                "historical_effective_seed_boundary": (
                    "No seed argument was passed. The effective producer default and "
                    "package revision were not embedded in the retained July outputs."
                ),
            },
            "fixed_eight_term_panel": {
                "term_count": 8,
                "decision_rule": "raw_p < 0.00625 and NES > 0 (strict)",
                "family_alpha_divided_by_eight": 0.00625,
                "selection_status": (
                    "discovery-derived on the same cohort; not a prospective "
                    "preregistration, selective-inference result, or independent confirmation"
                ),
            },
            "derivation": (
                "Landscape settings come from retained result.json and distance metadata; "
                "GSEA and fixed-term settings come from the reference scripts and wrapper "
                "invocations hashed below."
            ),
        },
        "source_inputs": source_inputs,
        "reference_implementation": {
            "relationship_to_outputs": (
                "Reference scripts at snapshot time. Their hashes were not embedded in the "
                "July outputs, so they cannot prove the exact historical producer revision."
            ),
            "scripts": reference_scripts,
            "environment_files": reference_environment,
        },
        "regimes": regimes,
        "reproducibility_boundary": {
            "offline_guarantee": (
                "Given the retained local files, this snapshot makes exact artifact-byte "
                "integrity, row counts, FDR counts, and fixed-term decisions reproducibly "
                "verifiable without network access."
            ),
            "exact_upstream_regeneration_guaranteed": False,
            "limitation": (
                "The original live CoGEx corpus/version, producer package revisions, and "
                "effective GSEA seed/default were not embedded in the July outputs. "
                "Reference-script hashes also were not embedded in those outputs. Exact "
                "regeneration from upstream network resources is therefore not guaranteed, "
                "even when retained artifact integrity verifies."
            ),
        },
    }


def serialized(snapshot: dict[str, Any]) -> str:
    """Return the canonical on-disk JSON representation."""
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="replace the snapshot")
    mode.add_argument("--verify", action="store_true", help="verify it (default)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot path, relative to --root unless absolute",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    snapshot_path = args.snapshot
    if not snapshot_path.is_absolute():
        snapshot_path = root / snapshot_path

    try:
        observed = build_snapshot(root)
        observed_text = serialized(observed)
        if args.write:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(observed_text, encoding="utf-8")
            print(f"wrote {snapshot_path}")
            return 0

        if not snapshot_path.is_file():
            raise ProvenanceError(f"snapshot is absent: {snapshot_path}")
        expected_text = snapshot_path.read_text(encoding="utf-8")
        if expected_text != observed_text:
            diff = difflib.unified_diff(
                expected_text.splitlines(),
                observed_text.splitlines(),
                fromfile=str(snapshot_path),
                tofile="recomputed",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
            return 1
        print(f"verified {snapshot_path}")
        return 0
    except (KeyError, OSError, ValueError, json.JSONDecodeError, ProvenanceError) as error:
        print(f"provenance error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
