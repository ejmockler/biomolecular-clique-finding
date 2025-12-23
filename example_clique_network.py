#!/usr/bin/env python
"""
Example: Clique Network Visualization

This script demonstrates how to create network graphs showing the correlation
structure within regulatory cliques.

The network reveals WHY genes form a clique by showing their membership patterns
across conditions. Nodes are colored by condition membership and sized by connectivity.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz.cliques import CliqueVisualizer

def main():
    print("=" * 70)
    print("CLIQUE NETWORK VISUALIZATION EXAMPLES")
    print("=" * 70)

    # Load clique data
    print("\nLoading data...")
    df = pd.read_csv("results/cliques/stratified_cliques.csv")
    print(f"  Loaded {len(df)} clique entries")
    print(f"  Regulators: {df['regulator'].nunique()}")

    # Create output directory
    output_dir = Path("figures/clique_networks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = CliqueVisualizer(style="paper")

    print("\n" + "=" * 70)
    print("Example 1: MAPT Regulatory Clique Network")
    print("=" * 70)
    print("\nMAPT (Microtubule-Associated Protein Tau) is a key ALS-related protein.")
    print("This network shows which genes co-occur in MAPT regulatory cliques.\n")

    fig1 = viz.plot_clique_network(
        regulator="MAPT",
        df=df,
        correlation_data=None,  # Membership-only network
        figsize=(12, 12)
    )

    fig1.save(output_dir / "mapt_network.pdf")
    fig1.save(output_dir / "mapt_network.png", dpi=300)

    print(f"Network Statistics:")
    print(f"  Total genes in union: {fig1.metadata['n_genes']}")
    print(f"  Edges (connections): {fig1.metadata['n_edges']}")
    print(f"\nGene Categories:")
    cats = fig1.metadata['gene_categories']
    print(f"  Stable (all conditions): {cats['stable']} genes")
    print(f"  CTRL-only (lost in disease): {cats['ctrl_only']} genes")
    print(f"  CASE-only (gained in disease): {cats['case_only']} genes")
    print(f"  Partial (some conditions): {cats['partial']} genes")

    print(f"\nSaved to: {output_dir / 'mapt_network.pdf'}")

    print("\n" + "=" * 70)
    print("Example 2: Condition-Specific Networks")
    print("=" * 70)
    print("\nComparing CASE_F vs CTRL_F networks for MAPT...")

    # CASE_F network
    fig2 = viz.plot_clique_network(
        regulator="MAPT",
        df=df,
        condition_focus="CASE_F",
        figsize=(10, 10)
    )
    fig2.save(output_dir / "mapt_case_f.pdf")

    # CTRL_F network
    fig3 = viz.plot_clique_network(
        regulator="MAPT",
        df=df,
        condition_focus="CTRL_F",
        figsize=(10, 10)
    )
    fig3.save(output_dir / "mapt_ctrl_f.pdf")

    print(f"  CASE_F: {fig2.metadata['n_genes']} genes")
    print(f"  CTRL_F: {fig3.metadata['n_genes']} genes")
    print(f"  Saved condition-specific networks")

    print("\n" + "=" * 70)
    print("Example 3: Multiple Regulators Comparison")
    print("=" * 70)
    print("\nCreating networks for key ALS-related regulators...")

    regulators = ["MAPT", "SLC2A1", "CDK1", "HMGB1", "CDC42"]

    for reg in regulators:
        try:
            fig = viz.plot_clique_network(
                regulator=reg,
                df=df,
                correlation_data=None,
                figsize=(10, 10)
            )
            fig.save(output_dir / f"{reg.lower()}_network.pdf")

            n_genes = fig.metadata['n_genes']
            n_stable = fig.metadata['gene_categories']['stable']
            print(f"  {reg:10s}: {n_genes:2d} genes ({n_stable} stable)")

        except ValueError as e:
            print(f"  {reg:10s}: No data")

    print(f"\nAll figures saved to: {output_dir.absolute()}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Visual Encoding:
  Node Colors:
    - Emerald: Gene in ALL conditions (stable across disease)
    - Blue:    Gene in CASE only (disease-specific)
    - Orange:  Gene in CTRL only (lost in disease)
    - Gray:    Gene in SOME conditions (partial membership)

  Node Size:
    - Large nodes = many connections (hub genes)
    - Small nodes = few connections (peripheral genes)

  Edge Thickness:
    - All edges have uniform weight in membership-only mode
    - With correlation data, thickness shows correlation strength

Key Insights:
  - Hub genes (large) are central to regulatory network
  - Stable genes (emerald) maintain function across conditions
  - Disease-specific genes (blue) are potential biomarkers
  - Lost genes (orange) indicate disrupted pathways

Recommended Workflow:
  1. Start with full network to see overall structure
  2. Compare condition-specific networks (CASE vs CTRL)
  3. Identify hub genes and condition-specific genes
  4. Investigate biological function of key genes
    """)

    print("\n" + "=" * 70)
    print("SUCCESS! All example networks generated.")
    print("=" * 70)
    print(f"\nView PDFs in: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Open PDFs to examine network structures")
    print("  2. Look for genes that appear in multiple conditions")
    print("  3. Identify hub genes (large nodes)")
    print("  4. Compare CASE vs CTRL networks")
    print("\n")

if __name__ == "__main__":
    main()
