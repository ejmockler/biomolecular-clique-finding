"""Bounded h=2, log2(x+1) landscape for ONE contrast — the published-primary
re-run on the corrected intensity scale.

Mirrors run_landscape_proteome.py exactly (same resolver, data paths,
n_permutations=999, covariates=Sex, measured-only graph) but pins
max_hops=2 (the decision-rule primary, NOT the unbounded sensitivity) and
transform="log2".  Writes to output/landscape_<dir>_measured_only_log2 so
it sits beside the raw canonical without clobbering it.

Usage:
    .venv/bin/python scripts/run_landscape_log2.py --contrast c9spor
    .venv/bin/python scripts/run_landscape_log2.py --contrast c9ctrl
    .venv/bin/python scripts/run_landscape_log2.py --contrast spctrl
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.panels import (
    LandscapeDesign,
    analyze_landscape,
    compute_landscape,
)

# contrast tag -> (case, control), output dir stem
CONTRASTS = {
    "c9spor": (("C9ORF72", "SPORADIC"), "landscape_proteome"),
    "c9ctrl": (("C9ORF72", "CONTROL"), "landscape_c9_vs_control"),
    "spctrl": (("SPORADIC", "CONTROL"), "landscape_sporadic_vs_control"),
}


def resolve_als_groups(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    c9 = metadata[
        (metadata["ClinReport_Mutations_Details"] == "C9orf72")
        | (metadata["C9orf72_repeat_length"] >= 30)
    ]
    known_muts = [
        "C9orf72", "SOD1", "FUS", "TARDBP", "TARDBP (TDP43)",
        "SETX", "Multiple", "Other",
    ]
    sporadic = metadata[
        (metadata["phenotype"] == "CASE")
        & (~metadata["ClinReport_Mutations_Details"].isin(known_muts))
        & (
            (metadata["C9orf72_repeat_length"] < 30)
            | metadata["C9orf72_repeat_length"].isna()
        )
    ]
    control = metadata[metadata["phenotype"] == "CTRL"]
    return {
        "C9ORF72": c9.index,
        "SPORADIC": sporadic.index,
        "CONTROL": control.index,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    ap.add_argument("--max-hops", type=int, default=2,
                    help="Bounded depth (2 = published primary).")
    ap.add_argument(
        "--graph-key", default="curie", choices=("curie", "symbol"),
        help=(
            "How a measured feature is identified as a graph node. "
            "'curie' gives each feature one namespaced gene id, so a "
            "traversal cannot inherit a homonymous entity's edges. "
            "'symbol' is the legacy name-matched space that produced the "
            "original artifacts. Written into the output dir name so the "
            "two key spaces can never share a directory."
        ),
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(f"landscape-log2-{args.contrast}")

    (case, control), stem = CONTRASTS[args.contrast]
    # The legacy symbol-keyed artifacts own the unsuffixed directory name;
    # a curie-keyed run is a DIFFERENT graph and must never resume into or
    # overwrite them.
    suffix = "" if args.graph_key == "symbol" else f"_{args.graph_key}"
    out_dir = ROOT / f"output/{stem}_measured_only_log2{suffix}"
    data_path = ROOT / "output/proteomics/all_als.data.csv"
    metadata_path = ROOT / "output/proteomics/all_als.metadata.csv"

    design = LandscapeDesign(
        contrast=(case, control),
        max_hops=args.max_hops,
        n_permutations=999,
        covariates=("Sex",),
        transform="log2",  # log2(x+1) — the corrected production scale
        graph_key=args.graph_key,
        description=(
            f"log2 published-primary re-run — {case} vs {control}, "
            f"measured-only regulatory graph, max_hops={args.max_hops}, "
            f"log2(x+1) intensities, {args.graph_key}-keyed graph nodes"
        ),
    )

    log.info("=== LOG2 LANDSCAPE START: %s vs %s (h=%d, key=%s) ===",
             case, control, args.max_hops, args.graph_key)
    log.info("Output dir: %s", out_dir)

    t0 = time.time()
    result = compute_landscape(
        design,
        data_path=data_path,
        metadata_path=metadata_path,
        group_resolver=resolve_als_groups,
        indra_env_file=ROOT / ".env",
        output_dir=out_dir,
        rng_seed=42,
        seed_batch_size=500,
        checkpoint=True,   # resumable if interrupted
    )
    elapsed = time.time() - t0

    log.info("=== LOG2 LANDSCAPE DONE: %s (%.1f min) ===",
             args.contrast, elapsed / 60)
    log.info("%d completed, %d degenerate, %d errored (of %d input features)",
             len(result.per_feature), len(result.degenerate_features),
             len(result.error_features), result.n_features_input)

    if result.per_feature:
        slopes = sorted(r.slope for r in result.per_feature)
        n = len(slopes)
        log.info("Slope: min=%.3f p10=%.3f median=%.3f p90=%.3f max=%.3f",
                 slopes[0], slopes[n // 10], slopes[n // 2],
                 slopes[9 * n // 10], slopes[-1])

    analysis = analyze_landscape(result, q_threshold=0.05)
    disc = [a for a in analysis.feature_results_adjusted if a.discovery]
    log.info("BH-FDR discoveries (q<0.05): %d", len(disc))
    log.info("LOG2_LANDSCAPE_DONE_%s", args.contrast.upper())


if __name__ == "__main__":
    main()
