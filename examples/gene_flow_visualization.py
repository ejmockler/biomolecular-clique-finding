#!/usr/bin/env python
"""
Example: Gene Flow Visualization for Regulatory Rewiring Analysis

This script demonstrates how to use the plot_gene_flow() method to visualize
how regulatory cliques change between healthy (CTRL) and disease (CASE) conditions.

The gene flow diagram reveals:
- Which genes are LOST in disease (present in CTRL, absent in CASE)
- Which genes are GAINED in disease (absent in CTRL, present in CASE)
- Which genes are STABLE (present in both conditions)
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz import CliqueVisualizer

# Load stratified cliques data
data_path = Path("results/cliques/stratified_cliques.csv")
df = pd.read_csv(data_path)

# Initialize visualizer with paper style for publication quality
viz = CliqueVisualizer(style="paper")

# Create output directory
output_dir = Path("figures/gene_flow")
output_dir.mkdir(parents=True, exist_ok=True)

# Example 1: MAPT (Tau protein) regulatory rewiring
print("Creating gene flow diagram for MAPT...")
fig = viz.plot_gene_flow("MAPT", df)
fig.save(output_dir / "mapt_gene_flow.pdf")

# Access metadata to see the changes
meta = fig.metadata
print(f"\nMAPT Regulatory Rewiring Summary:")
print(f"  Lost in disease: {meta['n_lost']} genes")
print(f"    {', '.join(meta['lost_genes'][:5])}...")
print(f"  Gained in disease: {meta['n_gained']} genes")
print(f"    {', '.join(meta['gained_genes'])}")
print(f"  Stable (both): {meta['n_stable']} genes")

fig.close()

# Example 2: SIRT3 (Mitochondrial deacetylase) regulatory rewiring
print("\n\nCreating gene flow diagram for SIRT3...")
fig = viz.plot_gene_flow("SIRT3", df)
fig.save(output_dir / "sirt3_gene_flow.pdf")

meta = fig.metadata
print(f"\nSIRT3 Regulatory Rewiring Summary:")
print(f"  Lost: {meta['n_lost']}, Gained: {meta['n_gained']}, Stable: {meta['n_stable']}")

fig.close()

# Example 3: Batch process multiple regulators
print("\n\nBatch processing regulators...")
regulators_of_interest = ["HMGB1", "CAT", "CDC42", "ELAVL1"]

for regulator in regulators_of_interest:
    try:
        fig = viz.plot_gene_flow(regulator, df)
        output_path = output_dir / f"{regulator.lower()}_gene_flow.pdf"
        fig.save(output_path)

        meta = fig.metadata
        print(f"  {regulator}: Lost={meta['n_lost']}, Gained={meta['n_gained']}, Stable={meta['n_stable']}")

        fig.close()
    except Exception as e:
        print(f"  {regulator}: Error - {e}")

print(f"\n✓ All gene flow diagrams saved to {output_dir}/")
