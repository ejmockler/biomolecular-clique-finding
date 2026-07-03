"""Full proteome landscape — every measured protein as a seed.

The decisive run: ~3,256 measured proteins, max_hops=2, n_permutations=999,
contrast=C9_vs_SPORADIC.  Outputs per-feature slope, p-value, shells +
the (3,256 × 3,256) sparse distance matrix.

Estimated wall time (extrapolating from the 200-feature smoke at 37.6 min):
~2-5 hours.  Sublinear scaling expected because frontier overlap means
later seed batches contribute few new nodes to the regulatory subgraph.

Outputs:
- output/landscape_proteome_measured_only/manifest.yaml
- output/landscape_proteome_measured_only/distances.npz + distances.meta.json
- output/landscape_proteome_measured_only/result.json

After this completes, analyze with:
    from cliquefinder.panels import LandscapeResult, analyze_landscape
    result = LandscapeResult.load_json("output/landscape_proteome_measured_only/result.json")
    analysis = analyze_landscape(result, q_threshold=0.05)
    # C9orf72's UniProt accession → look up its rank in
    # analysis.feature_results_adjusted (sorted by slope ascending)

Run:
    .venv/bin/python scripts/run_landscape_proteome.py
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
    LandscapeResult,
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
    log = logging.getLogger("landscape-proteome")

    # log2(x+1) is the production scale (LandscapeDesign.transform), recorded
    # in this run's manifest.yaml and surfaced by the downstream GSEA log.
    # To reproduce the historical raw-linear fit, set transform="raw" below
    # (writes to this same dir; the manifest records which scale was used).
    out_dir = ROOT / "output/landscape_proteome_measured_only_unbounded"
    data_path = ROOT / "output/proteomics/all_als.data.csv"
    metadata_path = ROOT / "output/proteomics/all_als.metadata.csv"

    design = LandscapeDesign(
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=None,  # wave_24l unbounded — BFS to CC completion
        n_permutations=999,
        covariates=("Sex",),
        transform="log2",  # log2(x+1) intensities
        description=(
            "Wave 24l full proteome — every measured protein as seed, "
            "C9_vs_SPORADIC, regulatory edges, max_hops=None (BFS to "
            "CC completion, anchor-adaptive depth), log2(x+1) intensities"
        ),
    )

    log.info("=== FULL PROTEOME LANDSCAPE START ===")
    log.info("Output dir: %s", out_dir)
    log.info("Contrast: %s vs %s", *design.contrast)
    log.info("max_hops=%s, n_permutations=%d, covariates=%s",
             design.max_hops, design.n_permutations, design.covariates)

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
    )
    total_elapsed = time.time() - t_start

    log.info("=== FULL PROTEOME LANDSCAPE DONE ===")
    log.info("Wall time: %.1f min (%.1f hr)", total_elapsed / 60, total_elapsed / 3600)
    log.info(
        "Results: %d completed, %d degenerate, %d errored (of %d input features)",
        len(result.per_feature),
        len(result.degenerate_features),
        len(result.error_features),
        result.n_features_input,
    )

    # Slope distribution summary.
    if result.per_feature:
        slopes = sorted(r.slope for r in result.per_feature)
        n = len(slopes)
        log.info(
            "Slope distribution: min=%.3f  p10=%.3f  median=%.3f  p90=%.3f  max=%.3f",
            slopes[0], slopes[n // 10], slopes[n // 2],
            slopes[9 * n // 10], slopes[-1],
        )

    # BH-FDR analysis.
    analysis = analyze_landscape(result, q_threshold=0.05)
    discoveries = [
        adj for adj in analysis.feature_results_adjusted if adj.discovery
    ]
    log.info("BH-FDR discoveries (q<0.05): %d", len(discoveries))
    if discoveries:
        log.info("Top 10 discoveries by slope (most negative first):")
        for adj in discoveries[:10]:
            log.info(
                "  %-12s  slope=%+.4f  raw_p=%.4f  bh_q=%.4f  rank=%d",
                adj.seed, adj.slope, adj.slope_pvalue,
                adj.bh_qvalue, adj.rank_left_tail,
            )

    # Look up C9orf72's empirical position.
    c9_uniprot = "Q96LT7"  # C9orf72 canonical UniProt accession
    c9_alt = "P0DPL3"  # alt isoform (just in case)
    c9_seeds_in_result = [
        adj for adj in analysis.feature_results_adjusted
        if adj.seed in (c9_uniprot, c9_alt)
    ]
    if c9_seeds_in_result:
        log.info("C9orf72 in landscape:")
        for adj in c9_seeds_in_result:
            log.info(
                "  %-12s  slope=%+.4f  raw_p=%.4f  bh_q=%.4f  "
                "rank=%d / %d (left tail)",
                adj.seed, adj.slope, adj.slope_pvalue, adj.bh_qvalue,
                adj.rank_left_tail, len(analysis.feature_results_adjusted),
            )
    else:
        log.warning(
            "C9orf72 (UniProt %s/%s) NOT found in landscape — was it "
            "matched in INDRA? Check distance_matrix.unmatched.",
            c9_uniprot, c9_alt,
        )

    log.info("PROTEOME_LANDSCAPE_DONE")


if __name__ == "__main__":
    main()
