"""SPORADIC-vs-CONTROL proteome landscape — third leg of the triangulation.

Together with C9-vs-SPORADIC and C9-vs-CONTROL, completes the 3-contrast
panel for testing whether the splicing/NPC/chromatin gradient signature is:
- C9-specific (only appears in C9-vs-SPORADIC and C9-vs-CONTROL, not here), or
- Shared ALS pathology (appears in this contrast too), or
- A topological property of the regulatory graph (appears in all three).

Outputs:
- output/landscape_sporadic_vs_control_measured_only/manifest.yaml
- output/landscape_sporadic_vs_control_measured_only/distances.npz + distances.meta.json
- output/landscape_sporadic_vs_control_measured_only/checkpoint.jsonl
- output/landscape_sporadic_vs_control_measured_only/inputs.json
- output/landscape_sporadic_vs_control_measured_only/result.json
"""
from __future__ import annotations

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
    log = logging.getLogger("landscape-sporadic-vs-control")

    # log2(x+1) is the production scale (LandscapeDesign.transform), recorded
    # in this run's manifest.yaml and surfaced by the downstream GSEA log.
    # To reproduce the historical raw-linear fit, set transform="raw" below
    # (writes to this same dir; the manifest records which scale was used).
    out_dir = ROOT / "output/landscape_sporadic_vs_control_measured_only_unbounded"
    data_path = ROOT / "output/proteomics/all_als.data.csv"
    metadata_path = ROOT / "output/proteomics/all_als.metadata.csv"

    design = LandscapeDesign(
        contrast=("SPORADIC", "CONTROL"),
        max_hops=None,  # wave_24l unbounded
        n_permutations=999,
        covariates=("Sex",),
        transform="log2",  # log2(x+1) intensities
        description=(
            "Wave 24l SPORADIC-vs-CONTROL — every measured protein as seed, "
            "regulatory edges, max_hops=None (BFS to CC completion), "
            "log2(x+1) intensities"
        ),
    )

    log.info("=== SPORADIC-vs-CONTROL PROTEOME LANDSCAPE START ===")
    log.info("Output dir: %s", out_dir)
    log.info("Contrast: %s vs %s", *design.contrast)
    log.info("max_hops=%s, n_permutations=%d, covariates=%s",
             design.max_hops, design.n_permutations, design.covariates)
    log.info("checkpoint=True (Wave 24h streaming JSONL)")

    t_start = time.time()
    result = compute_landscape(
        design,
        data_path=data_path,
        metadata_path=metadata_path,
        group_resolver=resolve_als_groups,
        indra_env_file=ROOT / ".env",
        output_dir=out_dir,
        rng_seed=42,
        seed_batch_size=500,
        checkpoint=True,
    )
    total_elapsed = time.time() - t_start

    log.info("=== SPORADIC-vs-CONTROL PROTEOME LANDSCAPE DONE ===")
    log.info("Wall time: %.1f min (%.1f hr)", total_elapsed / 60, total_elapsed / 3600)
    log.info(
        "Results: %d completed, %d degenerate, %d errored (of %d input features)",
        len(result.per_feature),
        len(result.degenerate_features),
        len(result.error_features),
        result.n_features_input,
    )

    if result.per_feature:
        slopes = sorted(r.slope for r in result.per_feature)
        n = len(slopes)
        log.info(
            "Slope distribution: min=%.3f  p10=%.3f  median=%.3f  p90=%.3f  max=%.3f",
            slopes[0], slopes[n // 10], slopes[n // 2],
            slopes[9 * n // 10], slopes[-1],
        )

    analysis = analyze_landscape(result, q_threshold=0.05)
    discoveries = [
        adj for adj in analysis.feature_results_adjusted if adj.discovery
    ]
    log.info("BH-FDR discoveries (q<0.05): %d", len(discoveries))

    log.info("SPORADIC_VS_CONTROL_LANDSCAPE_DONE")


if __name__ == "__main__":
    main()
