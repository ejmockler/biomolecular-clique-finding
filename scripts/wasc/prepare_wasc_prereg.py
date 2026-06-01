"""Freeze the WASC pre-registration artifacts (M6a).

Writes the three frozen inputs that get tagged at `wasc-prereg-v1.0`:

  data/wasc/cluster_members_v1.json   — per-theme measured cluster member sets
  data/wasc/E_WASC_v1.json            — the 944 within-cluster INDRA hop-1 edges
  data/wasc/manifest_v1.json          — SHA-256 fingerprints + provenance

Per memory/wasc_spec.md §1, these artifacts must be frozen before any
anchor-local null Q is computed on real data (M2+).

Re-running this script regenerates the artifacts deterministically given
the same INDRA snapshot and proteomics matrix. Inputs are SHA-pinned in
the manifest so any drift is detected at M2 launch.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))

import pandas as pd  # noqa: E402

from cliquefinder.knowledge.cogex import CoGExClient  # noqa: E402
from cliquefinder.stats.wasc import (  # noqa: E402
    DEFAULT_CLUSTER_TERMS,
    Theme,
    enumerate_wasc_indra_edges,
)
from cliquefinder.stats.wasc.edges import compute_measured_cluster_members  # noqa: E402

logger = logging.getLogger("wasc-prereg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

OUT_DIR = ROOT / "data" / "wasc"
PROTEOMICS_CSV = ROOT / "output" / "proteomics" / "all_als.data.csv"
METADATA_CSV = ROOT / "output" / "proteomics" / "all_als.metadata.csv"
INDRA_ENDPOINT = "bolt://indra-cogex-lb-b954b684556c373c.elb.us-east-1.amazonaws.com:7687"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load measured proteome (just the row index — UniProt accessions)
    logger.info("Loading measured proteome from %s", PROTEOMICS_CSV)
    full = pd.read_csv(PROTEOMICS_CSV, index_col=0, usecols=[0])
    measured_uniprots = frozenset(full.index.astype(str))
    logger.info("Measured UniProts: %d", len(measured_uniprots))

    proteomics_sha = _sha256_of_file(PROTEOMICS_CSV)
    metadata_sha = _sha256_of_file(METADATA_CSV) if METADATA_CSV.exists() else None

    # ----- 2. Query INDRA + enumerate
    timestamp = dt.datetime.now(dt.UTC).isoformat()
    logger.info("Connecting to INDRA CoGEx (%s)", INDRA_ENDPOINT)
    # Import here so common.py (used by the default fetch_term_members_func) is
    # available with the project-level sys.path tweak above.
    from common import (  # type: ignore[import-not-found]
        fetch_term_members_via_indra,
        hgnc_ids_to_uniprots,
    )

    with CoGExClient(env_file=ROOT / ".env") as client:
        # Capture per-theme members separately for the cluster_members artifact
        members_by_theme = compute_measured_cluster_members(
            list(DEFAULT_CLUSTER_TERMS),
            measured_uniprots,
            fetch_term_members_via_indra,
            hgnc_ids_to_uniprots,
        )
        # Run full enumeration (re-queries INDRA but the cluster-member step is fast)
        edges = enumerate_wasc_indra_edges(
            DEFAULT_CLUSTER_TERMS,
            measured_uniprots,
            cogex_client=client,
        )

    # ----- 3. Write cluster_members_v1.json
    # Build term grouping
    terms_by_theme: dict[Theme, list[str]] = defaultdict(list)
    for theme, tid in DEFAULT_CLUSTER_TERMS:
        terms_by_theme[theme].append(tid)

    cluster_members_doc = {
        "version": "v1.0",
        "frozen_at_git_tag": "wasc-prereg-v1.0",
        "indra_query_timestamp_utc": timestamp,
        "indra_cogex_endpoint": INDRA_ENDPOINT,
        "proteomics_csv_sha256": proteomics_sha,
        "metadata_csv_sha256": metadata_sha,
        "total_measured_uniprots": len(measured_uniprots),
        "total_cluster_members_measured": sum(len(m) for m in members_by_theme.values()),
        "themes": {
            theme.value: {
                "term_ids": sorted(terms_by_theme[theme]),
                "measured_uniprots": sorted(members_by_theme[theme]),
                "measured_count": len(members_by_theme[theme]),
            }
            for theme in (Theme.SPLICING, Theme.CHROMATIN, Theme.TRANSPORT)
        },
    }
    cluster_members_path = OUT_DIR / "cluster_members_v1.json"
    cluster_members_text = json.dumps(cluster_members_doc, indent=2, sort_keys=False)
    cluster_members_path.write_text(cluster_members_text + "\n")
    logger.info("Wrote %s (%d bytes)", cluster_members_path, len(cluster_members_text))

    # ----- 4. Write E_WASC_v1.json
    per_theme_counts = {theme.value: 0 for theme in Theme}
    for e in edges:
        per_theme_counts[e.theme.value] += 1

    e_wasc_doc = {
        "version": "v1.0",
        "frozen_at_git_tag": "wasc-prereg-v1.0",
        "indra_query_timestamp_utc": timestamp,
        "indra_cogex_endpoint": INDRA_ENDPOINT,
        "proteomics_csv_sha256": proteomics_sha,
        "edge_count_total": len(edges),
        "edge_count_per_theme": per_theme_counts,
        "edges": [
            {
                "edge_id": e.edge_id,
                "anchor_uniprot": e.anchor_uniprot,
                "target_uniprot": e.target_uniprot,
                "theme": e.theme.value,
                "network": e.network.value,
                "anchor_symbol": e.anchor_symbol,
                "target_symbol": e.target_symbol,
                "evidence_count": e.evidence_count,
                "stmt_types": list(e.stmt_types) if e.stmt_types else None,
            }
            for e in edges
        ],
    }
    e_wasc_path = OUT_DIR / "E_WASC_v1.json"
    e_wasc_text = json.dumps(e_wasc_doc, indent=2, sort_keys=False)
    e_wasc_path.write_text(e_wasc_text + "\n")
    logger.info("Wrote %s (%d bytes)", e_wasc_path, len(e_wasc_text))

    # ----- 5. Write manifest_v1.json
    manifest = {
        "version": "v1.0",
        "frozen_at_git_tag": "wasc-prereg-v1.0",
        "frozen_at_utc": timestamp,
        "artifacts": {
            "cluster_members_v1.json": {
                "path": "data/wasc/cluster_members_v1.json",
                "sha256": _sha256_of_text(cluster_members_text),
                "size_bytes": len(cluster_members_text),
            },
            "E_WASC_v1.json": {
                "path": "data/wasc/E_WASC_v1.json",
                "sha256": _sha256_of_text(e_wasc_text),
                "size_bytes": len(e_wasc_text),
            },
        },
        "inputs": {
            "proteomics_csv": {
                "path": str(PROTEOMICS_CSV.relative_to(ROOT)),
                "sha256": proteomics_sha,
            },
            "metadata_csv": {
                "path": str(METADATA_CSV.relative_to(ROOT)) if METADATA_CSV.exists() else None,
                "sha256": metadata_sha,
            },
            "indra_cogex_endpoint": INDRA_ENDPOINT,
            "indra_query_timestamp_utc": timestamp,
        },
        "summary": {
            "total_measured_uniprots": len(measured_uniprots),
            "total_cluster_members_measured": sum(len(m) for m in members_by_theme.values()),
            "edge_count_total": len(edges),
            "edge_count_per_theme": per_theme_counts,
        },
    }
    manifest_path = OUT_DIR / "manifest_v1.json"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=False)
    manifest_path.write_text(manifest_text + "\n")
    logger.info("Wrote %s (%d bytes)", manifest_path, len(manifest_text))

    # ----- Final report
    print()
    print("=" * 70)
    print("WASC pre-registration artifacts written.")
    print("=" * 70)
    print(f"  data/wasc/cluster_members_v1.json   ({len(cluster_members_text):>7,} bytes)")
    print(f"  data/wasc/E_WASC_v1.json            ({len(e_wasc_text):>7,} bytes)")
    print(f"  data/wasc/manifest_v1.json          ({len(manifest_text):>7,} bytes)")
    print()
    print(f"  |E_WASC|       = {len(edges):>5,}")
    for theme in (Theme.SPLICING, Theme.CHROMATIN, Theme.TRANSPORT):
        print(f"    {theme.value:<10}     {per_theme_counts[theme.value]:>5,}")
    print()
    print(f"Next step: git add data/wasc/ && git commit && git tag wasc-prereg-v1.0")


if __name__ == "__main__":
    main()
