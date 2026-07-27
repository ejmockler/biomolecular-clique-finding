"""200-feature landscape measurement — refine the proteome-scale extrapolation.

The 50-feature smoke took 48 minutes (almost entirely Neo4j subgraph
extraction).  Linear extrapolation to 3,256 features would be ~50 hours,
but the extraction cost may be sublinear due to neighborhood overlap
(many seeds reach the same regulatory hubs).

This script runs 200 features with explicit phase timing.  If extraction
takes ~3-4× the smoke (~150-200 min), scaling is roughly linear and the
full proteome is an overnight commitment.  If it takes ~1-1.5× (60-90
min), scaling is sublinear and the full proteome is ~3-5 hours.

Run:
    .venv/bin/python scripts/run_landscape_200_measure.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.panels import LandscapeDesign, compute_landscape


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("landscape-200")

    n_features = 200
    full_data = pd.read_csv(
        ROOT / "output/proteomics/all_als.data.csv", index_col=0,
    )
    out_dir = ROOT / "output/validation/landscape_200_measure"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_data = full_data.iloc[:n_features, :]
    subset_data_path = out_dir / "subset_data.csv"
    subset_data.to_csv(subset_data_path)
    log.info(
        "Subset proteomics: %d × %d → %s",
        subset_data.shape[0], subset_data.shape[1], subset_data_path,
    )

    design = LandscapeDesign(
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=2,
        n_permutations=49,
        covariates=("Sex",),
        transform="log2",  # explicit: matches the production default
        # Pinned: this driver reproduces the historical symbol-keyed
        # artifacts in its output dir. The pipeline default is now
        # "curie" (one namespaced gene id per feature); leaving this
        # implicit would silently write a different graph here.
        graph_key="symbol",
        description=f"Wave 24f extrapolation — {n_features} features × 49 perms, log2(x+1)",
    )

    log.info("=== TIMING START: %d features ===", n_features)
    t_start = time.time()
    result = compute_landscape(
        design,
        data_path=subset_data_path,
        metadata_path=ROOT / "output/proteomics/all_als.metadata.csv",
        group_resolver=resolve_als_groups,
        indra_env_file=ROOT / ".env",
        output_dir=out_dir,
        rng_seed=42,
        seed_batch_size=200,  # one batch
    )
    total_elapsed = time.time() - t_start
    log.info("=== TIMING END: %.1fs total (%.1f min) ===",
             total_elapsed, total_elapsed / 60)

    log.info("Result: %d completed, %d degenerate, %d errored",
             len(result.per_feature),
             len(result.degenerate_features),
             len(result.error_features))

    # Print extrapolations to common targets.
    per_feature_cost_min = (total_elapsed / 60) / n_features
    log.info(
        "Per-feature cost: %.2f min/feature",
        per_feature_cost_min,
    )
    for n in [500, 1000, 2000, 3256]:
        log.info(
            "Linear extrapolation to %d features: %.1f min (%.1f hr)",
            n,
            per_feature_cost_min * n,
            per_feature_cost_min * n / 60,
        )
    log.info("(NOTE: linear extrapolation ignores neighborhood overlap, "
             "which makes per-feature cost decrease as N grows.)")

    log.info("MEASURE_DONE")


if __name__ == "__main__":
    main()
