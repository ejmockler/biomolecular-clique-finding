"""C9-vs-CONTROL proteome landscape — control contrast for the C9-vs-SPORADIC run.

Same data, same group resolver, same regulatory subgraph (re-extracted because
manifest mismatch prevents reusing distances.npz from the C9-vs-SPORADIC run).
Differs only in the differential statistic |t|: now contrasts C9-ALS against
healthy controls instead of sporadic-ALS.

Purpose: confirm that the C9-vs-SPORADIC gradient's *negative*-NES on
ALS-clinical HPO terms is because those proteins are shared with sporadic.
Under C9-vs-CONTROL, both motor-neuron-degeneration AND C9-specific machinery
should be perturbed; ALS-clinical HPO terms should now show *positive* NES.

Outputs:
- output/landscape_c9_vs_control_measured_only/manifest.yaml
- output/landscape_c9_vs_control_measured_only/distances.npz + distances.meta.json
- output/landscape_c9_vs_control_measured_only/checkpoint.jsonl   (Wave 24h streaming)
- output/landscape_c9_vs_control_measured_only/inputs.json        (Wave 24h fingerprint)
- output/landscape_c9_vs_control_measured_only/result.json
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
    log = logging.getLogger("landscape-c9-vs-control")

    out_dir = ROOT / "output/landscape_c9_vs_control_measured_only_unbounded"
    data_path = ROOT / "output/proteomics/all_als.data.csv"
    metadata_path = ROOT / "output/proteomics/all_als.metadata.csv"

    design = LandscapeDesign(
        contrast=("C9ORF72", "CONTROL"),
        max_hops=None,  # wave_24l unbounded
        n_permutations=999,
        covariates=("Sex",),
        description=(
            "Wave 24l C9-vs-CONTROL — every measured protein as seed, "
            "regulatory edges, max_hops=None (BFS to CC completion)"
        ),
    )

    log.info("=== C9-vs-CONTROL PROTEOME LANDSCAPE START ===")
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

    log.info("=== C9-vs-CONTROL PROTEOME LANDSCAPE DONE ===")
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

    log.info("C9_VS_CONTROL_LANDSCAPE_DONE")


if __name__ == "__main__":
    main()
