"""Wave 24l confirmatory analysis — 8 pre-registered cluster terms.

The 8 cluster terms were SELECTED on the old (with-intermediates) slope-GSEA
in Wave 24i.  Under the new (measured-only-paths) regime, re-running the
full GSEA and "discovering" the same terms is data-snooping.  This script
implements the H5 confirmatory-vs-discovery separation:

  CONFIRMATORY (n=8, Bonferroni-corrected): take exactly the 8 pre-registered
  cluster terms.  Report their NES, raw p-value, FDR q-value from the new
  full discovery, AND apply Bonferroni-8 on the raw p-values.  This is the
  falsifiable test of whether the cluster claim is graph-invariant.

  Run after:
    1. Landscape compute for the contrast → result.json
    2. Full discovery GSEA (run_landscape_gsea.py) → CSV files per scope/db

Outputs: <out-dir>/confirmatory_8terms_{scope}.csv  +  summary.csv

8 pre-registered terms (Wave 24i selection):
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


# Pre-registered cluster term IDs (Wave 24i selection).  Each tuple is
# (db, term_id, display_name, cluster).
PREREGISTERED_TERMS: list[tuple[str, str, str, str]] = [
    ("go",       "go:0005643",            "nuclear pore",                                    "NPC"),
    ("go",       "go:0006913",            "nucleocytoplasmic transport",                     "NPC"),
    ("reactome", "reactome:R-HSA-180910", "Vpr-mediated nuclear import of PICs",             "NPC"),
    ("reactome", "reactome:R-HSA-72172",  "mRNA Splicing",                                   "Splicing"),
    ("reactome", "reactome:R-HSA-72203",  "Processing of Capped Intron-Containing Pre-mRNA", "Splicing"),
    ("go",       "go:0000398",            "mRNA splicing, via spliceosome",                  "Splicing"),
    ("go",       "go:0005694",            "chromosome",                                      "Chromatin"),
    ("go",       "go:0000785",            "chromatin",                                       "Chromatin"),
]
N_TERMS = len(PREREGISTERED_TERMS)
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
    """Build the Bonferroni-8 confirmatory table from the GSEA outputs."""
    rows: list[dict] = []
    for db, term_id, display_name, cluster in PREREGISTERED_TERMS:
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
        "Bonferroni-corrected confirmatory test over %d pre-registered "
        "cluster terms (α_family=%.3f, α_per_test=%.5f)",
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
