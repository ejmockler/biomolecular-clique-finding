#!/usr/bin/env python
"""
Sweep min_intermediaries thresholds for C9orf72 2-hop network.

Queries INDRA CoGEx once, then applies different min_intermediaries
thresholds locally to avoid redundant API calls.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print(f"Loaded .env from {env_file}")

import pandas as pd

# ── Load feature IDs from proteomics data ────────────────────────────
data_path = ROOT / "output" / "proteomics" / "c9orf72_als.data.csv"
print(f"\nLoading feature IDs from {data_path}...")
data = pd.read_csv(data_path, index_col=0, usecols=[0])  # just need row index
feature_ids = list(data.index)
print(f"  {len(feature_ids)} features in dataset")

# ── Map feature IDs to gene symbols ──────────────────────────────────
from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

print("\nMapping feature IDs to gene symbols...")
symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=True)
print(f"  {len(symbol_to_feature)} symbols mapped")

# ── Query INDRA (once) ───────────────────────────────────────────────
from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource
from indra.databases import hgnc_client

GENE = "C9orf72"
MIN_EVIDENCE = 3

print(f"\nQuerying INDRA CoGEx for {GENE} 2-hop network (min_evidence={MIN_EVIDENCE})...")
indra_source = INDRAKnowledgeSource(env_file=str(env_file))

# Hop 1
hop1_edges = indra_source.get_edges(source_entity=GENE, min_evidence=MIN_EVIDENCE)
hop1_genes: set[str] = set()
for e in hop1_edges:
    if hgnc_client.get_current_hgnc_id(e.target):
        hop1_genes.add(e.target)
hop1_genes.discard(GENE)
print(f"  Hop 1: {GENE} -> {len(hop1_genes)} gene targets "
      f"(filtered from {len({e.target for e in hop1_edges})} raw)")

# Hop 2 — collect ALL intermediaries
hop2_intermediaries: dict[str, set[str]] = {}
for i, h1 in enumerate(sorted(hop1_genes)):
    edges = indra_source.get_edges(h1, min_evidence=MIN_EVIDENCE)
    for e in edges:
        t = e.target
        if t == h1 or t == GENE:
            continue
        if not hgnc_client.get_current_hgnc_id(t):
            continue
        if t not in hop2_intermediaries:
            hop2_intermediaries[t] = set()
        hop2_intermediaries[t].add(h1)
    if (i + 1) % 10 == 0 or (i + 1) == len(hop1_genes):
        print(f"    Queried {i+1}/{len(hop1_genes)} hop-1 genes...")

indra_source.close()

print(f"\n  Raw hop-2 pool: {len(hop2_intermediaries)} unique genes")

# ── Map hop-1 genes to data ─────────────────────────────────────────
hop1_in_data = {sym for sym in hop1_genes if sym in symbol_to_feature}
print(f"  Hop-1 in data: {len(hop1_in_data)}/{len(hop1_genes)}")

# ── Sweep min_intermediaries ─────────────────────────────────────────
thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 10]

print("\n" + "=" * 80)
print(f"{'min_inter':>10} | {'Total':>6} | {'Hop-1':>6} | {'Hop-2':>6} | Hop-2 gene symbols")
print("-" * 80)

from cliquefinder.stats.target_set import TargetSet

sweep_out = ROOT / "output" / "sweep_min_intermediaries"
sweep_out.mkdir(parents=True, exist_ok=True)

for thresh in thresholds:
    # Apply threshold
    hop2_genes = {
        t for t, ints in hop2_intermediaries.items()
        if len(ints) >= thresh
    } - hop1_genes  # exclude hop-1 genes

    # Filter to genes in data
    hop2_in_data = {sym for sym in hop2_genes if sym in symbol_to_feature}

    total = len(hop1_in_data) + len(hop2_in_data)

    # Sort hop-2 by intermediary count for display
    hop2_sorted = sorted(
        hop2_in_data,
        key=lambda g: -len(hop2_intermediaries[g])
    )
    symbols_str = ", ".join(
        f"{g}({len(hop2_intermediaries[g])})" for g in hop2_sorted
    )

    print(f"{thresh:>10} | {total:>6} | {len(hop1_in_data):>6} | {len(hop2_in_data):>6} | {symbols_str}")

    # Serialize canonical target set for this threshold
    all_in_data = {sym: symbol_to_feature[sym] for sym in hop1_in_data}
    for sym in hop2_in_data:
        all_in_data[sym] = symbol_to_feature[sym]

    ts = TargetSet.from_query(
        targets_in_data=all_in_data,
        gene_symbol=GENE,
        min_evidence=MIN_EVIDENCE,
        n_hops=2,
        min_intermediaries=thresh,
        n_indra_edges_raw=len(hop1_edges),
    )
    ts.save(sweep_out / f"indra_targets_minint{thresh}.json")
print(f"\nTarget set files saved to {sweep_out}/")

print("=" * 80)

# Also show hop-1 gene symbols for completeness
print(f"\nHop-1 genes in data ({len(hop1_in_data)}):")
print("  " + ", ".join(sorted(hop1_in_data)))

# Show max intermediary counts for context
print(f"\nTop hop-2 genes by intermediary count (all, not just in data):")
top_all = sorted(hop2_intermediaries.items(), key=lambda x: -len(x[1]))[:20]
for gene, ints in top_all:
    in_data = "Y" if gene in symbol_to_feature else "N"
    in_hop1 = " [hop-1]" if gene in hop1_genes else ""
    print(f"  {gene:>12}: {len(ints):>3} intermediaries (in data: {in_data}){in_hop1}")
