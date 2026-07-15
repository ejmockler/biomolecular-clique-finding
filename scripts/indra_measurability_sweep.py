#!/usr/bin/env python
"""
Sweep min_evidence thresholds for C9orf72 INDRA targets and report
how many are measurable in the AnswerALS proteomics dataset.

Usage:
    .venv/bin/python scripts/indra_measurability_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ensure src is importable
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from cliquefinder.io.loaders import load_csv_matrix
from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

# --- Configuration ---
DATA_CSV = ROOT / "output/proteomics/all_als.data.csv"
ENV_FILE = ROOT / ".env"
GENE_SYMBOL = "C9orf72"
THRESHOLDS = [1, 2, 3, 4, 5, 6, 7, 8]


def main():
    # 1. Load proteomics data to get feature_ids
    print(f"Loading data: {DATA_CSV}")
    matrix = load_csv_matrix(DATA_CSV)
    feature_ids = list(matrix.feature_ids)
    print(f"  {len(feature_ids)} features (UniProt IDs)")

    # 2. Map feature IDs to gene symbols (done once — expensive API calls)
    print("\nMapping feature IDs to gene symbols...")
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=True)
    measurable_symbols = set(symbol_to_feature.keys())
    print(f"  {len(measurable_symbols)} total symbol entries in mapping "
          f"(includes aliases + case variants)")

    # 3. Initialize INDRA source (one connection, reused for all thresholds)
    print(f"\nConnecting to INDRA CoGEx...")
    indra_source = INDRAKnowledgeSource(env_file=str(ENV_FILE))

    # 4. Sweep thresholds
    print(f"\n{'='*80}")
    print(f"{'Threshold':>10} | {'INDRA targets':>14} | {'HGNC genes':>11} | "
          f"{'Measurable':>11} | {'Coverage':>9}")
    print(f"{'-'*80}")

    results = []
    for min_ev in THRESHOLDS:
        # Query INDRA for downstream targets at this threshold
        edges = indra_source.get_edges(
            source_entity=GENE_SYMBOL,
            relationship_types=None,
            min_evidence=min_ev,
            min_confidence=0.0,
        )

        # All unique target gene symbols from INDRA
        all_target_symbols = {edge.target for edge in edges}
        n_indra_targets = len(all_target_symbols)

        # Filter to HGNC genes: check which ones are in the symbol mapping
        # (same logic as query_network_targets)
        measurable_targets = {}
        for target_symbol in all_target_symbols:
            if target_symbol in symbol_to_feature:
                measurable_targets[target_symbol] = symbol_to_feature[target_symbol]

        n_measurable = len(measurable_targets)
        coverage = (n_measurable / n_indra_targets * 100) if n_indra_targets > 0 else 0

        print(f"{min_ev:>10} | {n_indra_targets:>14} | {n_indra_targets:>11} | "
              f"{n_measurable:>11} | {coverage:>8.1f}%")

        results.append({
            "min_evidence": min_ev,
            "n_indra_targets": n_indra_targets,
            "n_measurable": n_measurable,
            "coverage_pct": round(coverage, 1),
            "measurable_genes": sorted(measurable_targets.keys()),
        })

    indra_source.close()

    # 5. Print detailed gene lists
    print(f"\n{'='*80}")
    print("Detailed measurable gene lists by threshold:")
    print(f"{'='*80}")

    for r in results:
        genes = r["measurable_genes"]
        print(f"\nmin_evidence={r['min_evidence']}: "
              f"{r['n_measurable']} measurable / {r['n_indra_targets']} INDRA targets")
        if genes:
            # Print in columns
            for i in range(0, len(genes), 8):
                chunk = genes[i:i+8]
                print(f"  {', '.join(chunk)}")

    # 6. Show genes lost at each threshold step
    print(f"\n{'='*80}")
    print("Genes LOST at each threshold step:")
    print(f"{'='*80}")

    for i in range(1, len(results)):
        prev_genes = set(results[i-1]["measurable_genes"])
        curr_genes = set(results[i]["measurable_genes"])
        lost = sorted(prev_genes - curr_genes)
        gained = sorted(curr_genes - prev_genes)
        t_prev = results[i-1]["min_evidence"]
        t_curr = results[i]["min_evidence"]
        if lost:
            print(f"\n  {t_prev} -> {t_curr}: lost {len(lost)} genes: {', '.join(lost)}")
        if gained:
            print(f"  {t_prev} -> {t_curr}: gained {len(gained)} genes: {', '.join(gained)}")
        if not lost and not gained:
            print(f"\n  {t_prev} -> {t_curr}: no change")


if __name__ == "__main__":
    main()
