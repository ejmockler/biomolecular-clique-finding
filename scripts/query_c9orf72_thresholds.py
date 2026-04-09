#!/usr/bin/env python
"""Query INDRA for C9orf72 downstream targets at different min_evidence thresholds."""

import os
import sys
from pathlib import Path

# Load .env file
env_file = Path("/Users/noot/Documents/biomolecular-clique-finding/.env")
if env_file.exists():
    for line in env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]

print("=" * 70)
print("C9orf72 Downstream Targets by min_evidence Threshold")
print("=" * 70)

indra_source = INDRAKnowledgeSource(env_file=str(env_file))

for min_ev in thresholds:
    edges = indra_source.get_edges(
        source_entity="C9orf72",
        relationship_types=None,  # All relationship types
        min_evidence=min_ev,
        min_confidence=0.0,
    )

    target_symbols = sorted({e.target for e in edges})
    n_targets = len(target_symbols)

    print(f"\nmin_evidence={min_ev:>2d}: {n_targets} targets")
    if target_symbols:
        # Show evidence counts per target
        target_ev = {}
        for e in edges:
            if e.target not in target_ev:
                target_ev[e.target] = 0
            target_ev[e.target] = max(target_ev[e.target], e.evidence_count)
        # Sort by evidence count descending
        sorted_targets = sorted(target_ev.items(), key=lambda x: -x[1])
        for sym, ev in sorted_targets:
            print(f"    {sym:<15s} (max evidence: {ev})")

indra_source.close()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"{'Threshold':>10s} | {'Targets':>8s}")
print("-" * 25)

# Re-run summary (use cached results approach)
indra_source2 = INDRAKnowledgeSource(env_file=str(env_file))
# Get all edges at threshold 1 (the loosest) and filter client-side for summary
all_edges = indra_source2.get_edges(
    source_entity="C9orf72",
    relationship_types=None,
    min_evidence=1,
    min_confidence=0.0,
)
indra_source2.close()

# Build per-target max evidence
target_max_ev = {}
for e in all_edges:
    if e.target not in target_max_ev:
        target_max_ev[e.target] = 0
    target_max_ev[e.target] = max(target_max_ev[e.target], e.evidence_count)

for min_ev in thresholds:
    # Count targets with at least one edge >= min_ev
    # Note: the server filters edges, so a target may have multiple edges
    # with different evidence counts. We use max evidence per target.
    n = sum(1 for ev in target_max_ev.values() if ev >= min_ev)
    print(f"{min_ev:>10d} | {n:>8d}")

# Serialize canonical target sets if proteomics data is available
data_path = Path(__file__).resolve().parent.parent / "output" / "proteomics" / "c9orf72_als.data.csv"
if data_path.exists():
    import csv
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.target_set import TargetSet

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)
        feature_ids = [row[0] for row in reader if row[0]]
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=False)

    sweep_out = Path(__file__).resolve().parent.parent / "output" / "sweep_min_evidence"
    sweep_out.mkdir(parents=True, exist_ok=True)

    for min_ev in thresholds:
        passing = {t for t, ev in target_max_ev.items() if ev >= min_ev}
        targets_in_data = {
            sym: symbol_to_feature[sym]
            for sym in passing if sym in symbol_to_feature
        }
        ts = TargetSet.from_query(
            targets_in_data=targets_in_data,
            gene_symbol="C9orf72",
            min_evidence=min_ev,
            n_hops=1,
            n_indra_edges_raw=len(all_edges),
        )
        ts.save(sweep_out / f"indra_targets_minev{min_ev}.json")

    print(f"\nTarget set files saved to {sweep_out}/")
