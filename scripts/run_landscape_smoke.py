"""Smoke test: 50-feature landscape × 49 perms via real Neo4j.

Verifies the end-to-end compute_landscape path that the unit tests
deliberately bypass:
- Real Neo4j extraction
- Bridge-delegated |t|, alias collapse, and degree aggregation
- All-pairs shortest-paths over a real subgraph
- Per-feature gradient with dynamic per-feature RNG

Run:
    .venv/bin/python scripts/run_landscape_smoke.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.panels import (
    LandscapeDesign,
    analyze_landscape,
    compute_landscape,
)


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
    log = logging.getLogger("landscape-smoke")

    # Subset to first 50 measured proteins for the smoke test by
    # creating a temporary trimmed CSV.  This avoids running the full
    # 3,256-feature landscape during smoke (~hours); we only need to
    # verify the pipeline shape end-to-end.
    full_data = pd.read_csv(
        ROOT / "output/proteomics/all_als.data.csv", index_col=0,
    )
    smoke_dir = ROOT / "output/validation/landscape_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    subset_data = full_data.iloc[:50, :]
    subset_data_path = smoke_dir / "smoke_data.csv"
    subset_data.to_csv(subset_data_path)
    log.info("Subset proteomics: %d × %d → %s",
             subset_data.shape[0], subset_data.shape[1], subset_data_path)

    design = LandscapeDesign(
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=2,
        n_permutations=49,
        covariates=("Sex",),
        description="Wave 24f smoke — 50 features × 49 perms",
    )

    log.info("Running landscape on 50-feature subset")
    result = compute_landscape(
        design,
        data_path=subset_data_path,
        metadata_path=ROOT / "output/proteomics/all_als.metadata.csv",
        group_resolver=resolve_als_groups,
        indra_env_file=ROOT / ".env",
        output_dir=smoke_dir,
        rng_seed=42,
        seed_batch_size=50,  # one batch given the subset size
    )

    log.info(
        "Landscape result: %d completed, %d degenerate, %d errored "
        "(of %d input features)",
        len(result.per_feature),
        len(result.degenerate_features),
        len(result.error_features),
        result.n_features_input,
    )

    if result.per_feature:
        slopes = sorted(r.slope for r in result.per_feature)
        log.info("Slope distribution: min=%.3f median=%.3f max=%.3f",
                 slopes[0], slopes[len(slopes) // 2], slopes[-1])

    analysis = analyze_landscape(result, q_threshold=0.05)
    log.info("BH-FDR discoveries (q<0.05): %d",
             sum(1 for adj in analysis.feature_results_adjusted if adj.discovery))

    log.info("SMOKE LANDSCAPE OK")


if __name__ == "__main__":
    main()
