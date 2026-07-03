"""GSEA over the proteome landscape — GO + Reactome + WikiPathways + HPO.

Tests whether the gradient signal (slope of mean |t| vs hop distance) is
concentrated in pathways/phenotypes curated *independently* of INDRA's
literature evidence.  Cross-source concordance is the robustness criterion.

Score per anchor (depending on --score-type):
- slope (default): score = -slope, so larger values = stronger gradient near anchor
- spearman:        score = -spearman_rho, partly degree-corrected via rank correlation

Scopes (--scope-set):
- both (default): runs both `robust` (hop1 >= 20) and `all` (every measured)
- robust:        only the robust subset
- all:           only the full set
- peripheral_only: only hop1 < 20 anchors  (the (all - robust) split for self-correlation tests)

Sensitivity-test knobs:
- --permutation-num   (default 1000; the Wave 24i baseline. Set 10000 for FDR-resolution tests)
- --weighted-score-type {0|1|1.5|2}  (gseapy.prerank weight parameter; default 1)
- --drop-top-pct      (default 0.0; e.g. 0.01 drops the most-extreme 1% of anchors before scoring)
- --min-size          (default 1; the indra_cogex default. Set 15 to match original GSEA paper)

Outputs: <out-dir>/{scope}_{db}.csv  (one row per gene set) + summary.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from indra.databases import uniprot_client
from indra_cogex.client.enrichment.continuous import (
    go_gsea, reactome_gsea, wikipathways_gsea, phenotype_gsea,
)
from indra_cogex.client.neo4j_client import Neo4jClient


GSEA_FNS = {
    "go": go_gsea,
    "reactome": reactome_gsea,
    "wikipathways": wikipathways_gsea,
    "phenotype": phenotype_gsea,
}


def _score_for(record: dict, score_type: str) -> float:
    if score_type == "slope":
        return -float(record["slope"])
    if score_type == "spearman":
        # Spearman rho is bounded [-1, 1]; negate so anti-gradient (positive rho) ranks low.
        return -float(record.get("spearman_rho") or 0.0)
    raise ValueError(f"Unknown score_type: {score_type}")


def build_scores(
    per_feature: list[dict],
    scope: str,
    score_type: str = "slope",
    drop_top_pct: float = 0.0,
) -> dict[str, float]:
    """Build {hgnc_id_str: score}, optionally trimming the most-extreme anchors.

    scope ∈ {robust, all, peripheral_only}
    score_type ∈ {slope, spearman}
    drop_top_pct: e.g. 0.01 drops the most-extreme 1% of anchors (largest score) before
                  building the dict — sensitivity test for outlier-driven GSEA hits.
    """
    # Phase 1: filter by scope (hop1 size)
    if scope == "robust":
        records = [r for r in per_feature if r.get("shells") and r["shells"][0]["n_genes"] >= 20]
    elif scope == "peripheral_only":
        records = [r for r in per_feature if r.get("shells") and r["shells"][0]["n_genes"] < 20]
    elif scope == "all":
        records = list(per_feature)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    # Phase 2: compute scores; drop_top_pct removes the most-extreme N before HGNC mapping
    pairs: list[tuple[str, float]] = []
    skipped_unmapped = 0
    for r in records:
        hgnc_id = uniprot_client.get_hgnc_id(r["seed"])
        if not hgnc_id:
            skipped_unmapped += 1
            continue
        pairs.append((str(hgnc_id), _score_for(r, score_type)))

    if drop_top_pct and pairs:
        n_drop = max(1, int(round(drop_top_pct * len(pairs))))
        pairs.sort(key=lambda x: -x[1])  # largest score first
        pairs = pairs[n_drop:]

    return dict(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Landscape output dir containing result.json. Required "
                             "(no default) so a bare invocation cannot silently consume "
                             "a stale artifact on the wrong intensity scale. The "
                             "result.json's embedded design records the transform.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/landscape_gsea")
    parser.add_argument("--permutation-num", type=int, default=1000,
                        help="gseapy permutation count. p_floor = 1/N. Default 1000.")
    parser.add_argument("--weighted-score-type", type=float, default=1.0,
                        help="gseapy weighted_score_type. 0 = pure rank ES; 1 = magnitude-weighted (default).")
    parser.add_argument("--drop-top-pct", type=float, default=0.0,
                        help="Drop the most-extreme this fraction of anchors before scoring.")
    parser.add_argument("--score-type", choices=("slope", "spearman"), default="slope")
    parser.add_argument("--scope-set", choices=("both", "robust", "all", "peripheral_only"),
                        default="both", help="Which scopes to run.")
    parser.add_argument("--min-size", type=int, default=1,
                        help="gseapy min_size. Default 1; original GSEA paper used 15.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("landscape-gsea")

    result_path = args.result_dir / "result.json"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", result_path)
    log.info("Config: perm=%d  weighted=%g  drop_top=%g  score=%s  scope=%s  min_size=%d",
             args.permutation_num, args.weighted_score_type, args.drop_top_pct,
             args.score_type, args.scope_set, args.min_size)
    data = json.loads(result_path.read_text())
    pf = data["per_feature"]
    log.info("per_feature: %d", len(pf))

    # Provenance: surface the intensity scale these slopes were computed on,
    # so a raw-vs-log2 mixup is loud, not silent.  result.json from
    # compute_landscape embeds the design (with its transform); the bolt-on
    # log2 emitter writes a top-level "transform".  A legacy file with neither
    # was a raw run (mirrors LandscapeDesign.from_dict's back-compat default).
    _design = data.get("design", {}) if isinstance(data, dict) else {}
    _transform = (
        (_design.get("transform") if isinstance(_design, dict) else None)
        or data.get("transform")
        or "raw (legacy: no transform recorded)"
    )
    log.info("Intensity transform of input slopes: %s", _transform)

    if args.scope_set == "both":
        scope_names = ["robust", "all"]
    else:
        scope_names = [args.scope_set]

    scopes = {
        s: build_scores(pf, s, score_type=args.score_type, drop_top_pct=args.drop_top_pct)
        for s in scope_names
    }
    for name, sc in scopes.items():
        log.info("%s scope: %d HGNC scores", name, len(sc))

    client = Neo4jClient()
    summary_rows: list[dict] = []

    for scope_name, scores in scopes.items():
        for db_name, fn in GSEA_FNS.items():
            tag = f"{scope_name}_{db_name}"
            out_csv = out_dir / f"{tag}.csv"
            log.info("[%s] running GSEA (n=%d scores)", tag, len(scores))
            t0 = time.time()
            df = fn(
                scores=scores,
                client=client,
                directory=None,
                permutation_num=args.permutation_num,
                weighted_score_type=args.weighted_score_type,
                min_size=args.min_size,
            )
            elapsed = time.time() - t0
            df.to_csv(out_csv, index=False)
            n_total = len(df)
            n_sig = (df["FDR q-val"] < 0.05).sum() if "FDR q-val" in df.columns else 0
            log.info("[%s] %d gene sets, %d FDR q<0.05, %.1fs -> %s",
                     tag, n_total, n_sig, elapsed, out_csv.name)
            summary_rows.append({
                "scope": scope_name,
                "db": db_name,
                "n_gene_sets": n_total,
                "n_fdr_lt_05": int(n_sig),
                "elapsed_s": round(elapsed, 1),
            })

    summary = pd.DataFrame(summary_rows)
    log.info("\n=== Summary ===\n%s", summary.to_string(index=False))
    summary.to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
