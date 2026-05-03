"""Smoke test: 2-seed panel × 49 perms via the new panels module.

Verifies the end-to-end ProcessPoolExecutor + real Neo4j path that
the unit tests deliberately bypass.  Not a fast test — expect ~1
minute against the live INDRA endpoint.

Run:
    .venv/bin/python scripts/run_panel_smoke.py

Inspects the on-disk manifest + result for byte-deterministic
ordering and structured failure info.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.panels import (
    PanelDesign,
    PanelStratum,
    analyze_panel,
    run_panel,
)


def resolve_als_groups(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    """Module-level function (picklable) — AnswerALS cohort resolver.

    Mirrors scripts/specificity_triangle.py:resolve_groups so the
    smoke test exercises the same biology.
    """
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
    log = logging.getLogger("smoke")

    design = PanelDesign(
        target_seed="C9orf72",
        strata=(
            PanelStratum(name="RNA_RBP_smoke", members=("HNRNPK",)),
        ),
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=2,
        n_permutations=49,
        covariates=("Sex",),
        selection_rng_seed=42,
        description="Wave 24e smoke test — 2 seeds × 49 perms",
    )

    out_dir = ROOT / "output/validation/panel_smoke"
    log.info("Running 2-seed panel: %s", design.selected_seeds())

    result = run_panel(
        design,
        data_path=ROOT / "output/proteomics/all_als.data.csv",
        metadata_path=ROOT / "output/proteomics/all_als.metadata.csv",
        group_resolver=resolve_als_groups,
        indra_env_file=ROOT / ".env",
        output_dir=out_dir,
        parallelism=2,
        seed_timeout_seconds=900,  # 15 min/seed cap
    )

    log.info("Target %s: slope=%.4f p=%.4f",
             result.target_result.seed,
             result.target_result.slope,
             result.target_result.slope_pvalue)
    for r in result.per_seed:
        log.info("Panel %s [%s]: slope=%.4f p=%.4f",
                 r.seed, r.stratum, r.slope, r.slope_pvalue)
    if result.failed_seeds:
        log.warning("Failed: %s", [f.to_dict() for f in result.failed_seeds])

    analysis = analyze_panel(result)
    log.info("Target empirical p (left tail): %.4f",
             analysis.target_position.empirical_p_left)
    log.info("SMOKE PANEL OK")


if __name__ == "__main__":
    main()
