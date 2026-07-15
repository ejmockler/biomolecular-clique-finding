"""Wave 24l fixed-term same-cohort consistency analysis.

The eight cluster terms were selected on this cohort using the old
with-intermediates slope-GSEA in Wave 24i, then fixed before the measured-only
method-transfer reruns. Re-running full GSEA does not make them a new discovery.
This script extracts an explicitly same-cohort consistency readout:

  FIXED TERM PANEL (n=8): report NES, raw p-value, and FDR q-value from the
  method-transfer GSEA, and apply an eightfold raw-p threshold. Because term
  selection used the same cohort, this is not a post-selection FWER or
  selective-inference guarantee and does not establish graph invariance.

  Run after:
    1. Landscape compute for the contrast → result.json
    2. Full discovery GSEA (run_landscape_gsea.py) → CSV files per scope/db

Outputs: <out-dir>/confirmatory_8terms_{scope}.csv  +  summary.csv

8 discovery-derived fixed terms (Wave 24i selection):
  NPC: go:0005643 nuclear pore, go:0006913 nucleocytoplasmic transport,
       reactome:R-HSA-180910 Vpr-mediated nuclear import of PICs
  Splicing: reactome:R-HSA-72172 mRNA Splicing,
            reactome:R-HSA-72203 Processing of Capped Intron-Containing Pre-mRNA,
            go:0000398 mRNA splicing, via spliceosome
  Chromatin: go:0005694 chromosome, go:0000785 chromatin

Run:
    .venv/bin/python scripts/analyze_landscape_confirmatory.py \
        --gsea-dir output/landscape_gsea_c9spor_measured_only \
        --out-dir output/landscape_confirmatory_c9spor_measured_only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Discovery-derived cluster term IDs, fixed before method transfer. Each tuple is
# (db, term_id, display_name, cluster).
FIXED_TERMS: list[tuple[str, str, str, str]] = [
    ("go",       "go:0005643",            "nuclear pore",                                    "NPC"),
    ("go",       "go:0006913",            "nucleocytoplasmic transport",                     "NPC"),
    ("reactome", "reactome:R-HSA-180910", "Vpr-mediated nuclear import of PICs",             "NPC"),
    ("reactome", "reactome:R-HSA-72172",  "mRNA Splicing",                                   "Splicing"),
    ("reactome", "reactome:R-HSA-72203",  "Processing of Capped Intron-Containing Pre-mRNA", "Splicing"),
    ("go",       "go:0000398",            "mRNA splicing, via spliceosome",                  "Splicing"),
    ("go",       "go:0005694",            "chromosome",                                      "Chromatin"),
    ("go",       "go:0000785",            "chromatin",                                       "Chromatin"),
]
N_TERMS = len(FIXED_TERMS)
BONFERRONI_ALPHA_FAMILY = 0.05
BONFERRONI_ALPHA_PER_TEST = BONFERRONI_ALPHA_FAMILY / N_TERMS  # 0.00625


def load_gsea_results(
    gsea_dir: Path, scope: str,
) -> dict[str, pd.DataFrame]:
    """Load <scope>_{db}.csv for each db in {go, reactome}.  Returns
    {db_name: DataFrame indexed by Term}."""
    out: dict[str, pd.DataFrame] = {}
    for db in ("go", "reactome"):
        path = gsea_dir / f"{scope}_{db}.csv"
        if not path.exists():
            logging.warning("Missing %s — skipping db=%s", path, db)
            continue
        df = pd.read_csv(path)
        # GSEA output uses 'Term' as the ID column.
        df = df.set_index("Term", drop=False)
        out[db] = df
    return out


def confirmatory_table(
    gsea_by_db: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the eightfold-threshold fixed-term table from GSEA outputs."""
    rows: list[dict] = []
    for db, term_id, display_name, cluster in FIXED_TERMS:
        gsea = gsea_by_db.get(db)
        if gsea is None or term_id not in gsea.index:
            rows.append({
                "db": db,
                "term_id": term_id,
                "term": display_name,
                "cluster": cluster,
                "found": False,
                "NES": None,
                "raw_p": None,
                "fdr_q_full": None,
                "bonferroni_p": None,
                "bonferroni_pass": False,
                "matched_size": None,
                "geneset_size": None,
            })
            continue
        row = gsea.loc[term_id]
        raw_p = float(row["NOM p-val"])
        bonferroni_p = min(raw_p * N_TERMS, 1.0)
        rows.append({
            "db": db,
            "term_id": term_id,
            "term": display_name,
            "cluster": cluster,
            "found": True,
            "NES": float(row["NES"]),
            "raw_p": raw_p,
            "fdr_q_full": float(row["FDR q-val"]),
            "bonferroni_p": bonferroni_p,
            "bonferroni_pass": bool(
                raw_p < BONFERRONI_ALPHA_PER_TEST
                and float(row["NES"]) > 0
            ),
            "matched_size": int(row["matched_size"]),
            "geneset_size": int(row["geneset_size"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gsea-dir", type=Path, required=True,
        help="Directory containing <scope>_{go,reactome}.csv from run_landscape_gsea.py.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
    )
    parser.add_argument(
        "--scope", choices=("robust", "all"),
        default="robust",
        help="Which anchor scope to use.  Default: robust (hop1 >= 20).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("confirmatory")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Eightfold-threshold same-cohort check over %d discovery-derived "
        "fixed terms (nominal α=%.3f, per-test threshold=%.5f; no "
        "post-selection FWER guarantee)",
        N_TERMS, BONFERRONI_ALPHA_FAMILY, BONFERRONI_ALPHA_PER_TEST,
    )

    for scope in ([args.scope] if args.scope != "both" else ["robust", "all"]):
        gsea_by_db = load_gsea_results(args.gsea_dir, scope)
        if not gsea_by_db:
            log.error("No GSEA results in %s; skipping scope=%s",
                      args.gsea_dir, scope)
            continue
        table = confirmatory_table(gsea_by_db)
        out_csv = args.out_dir / f"confirmatory_8terms_{scope}.csv"
        table.to_csv(out_csv, index=False)
        log.info(
            "Wrote %s — %d/%d terms found, %d pass Bonferroni-8 "
            "(raw_p < %.5f and NES > 0)",
            out_csv, table["found"].sum(), N_TERMS,
            table["bonferroni_pass"].sum(),
            BONFERRONI_ALPHA_PER_TEST,
        )
        # Pretty summary to stdout.
        log.info(
            "\n%s",
            table[
                ["cluster", "db", "term", "found", "NES", "raw_p",
                 "fdr_q_full", "bonferroni_p", "bonferroni_pass"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        )

    log.info("CONFIRMATORY_DONE")


if __name__ == "__main__":
    main()
