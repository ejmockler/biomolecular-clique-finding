#!/usr/bin/env python3
"""
Full workflow test: Regulator Overview → Detailed Analysis

This script demonstrates the complete analysis workflow:
1. Start with regulator overview (scanning)
2. Identify interesting regulators
3. Drill down with detailed visualizations
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz.cliques import CliqueVisualizer

print("=" * 80)
print("FULL WORKFLOW: REGULATOR OVERVIEW → DETAILED ANALYSIS")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data_dir = Path("results/cliques")
rewiring_df = pd.read_csv(data_dir / "regulator_rewiring_stats.csv")
stratified_df = pd.read_csv(data_dir / "cliques.csv")
print(f"   ✓ Loaded {len(rewiring_df)} rewiring records")
print(f"   ✓ Loaded {len(stratified_df)} stratified records")

# Create output directory
output_dir = Path("figures/full_workflow")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Create regulator overview
print("\n2. Creating regulator overview (entry point)...")
viz = CliqueVisualizer(style="paper")
fig_overview = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.25,  # Filter to moderate-strong signals
    label_top_n=15
)
fig_overview.save(output_dir / "step1_regulator_overview.pdf")
print(f"   ✓ Generated overview plot")
print(f"   ✓ Showing {fig_overview.metadata['n_regulators']} regulators")
print(f"   ✓ Both sexes: {fig_overview.metadata['n_both_sexes']}")
print(f"   ✓ Male-specific: {fig_overview.metadata['n_male_specific']}")
print(f"   ✓ Female-specific: {fig_overview.metadata['n_female_specific']}")

# Step 2: Identify top regulators programmatically
print("\n3. Identifying top regulators by rewiring score...")
top_regulators = []
for reg in rewiring_df['regulator'].unique():
    reg_data = rewiring_df[rewiring_df['regulator'] == reg]
    max_rewiring = reg_data['rewiring_score'].abs().max()
    if max_rewiring >= 0.35:  # High threshold
        top_regulators.append((reg, max_rewiring))

top_regulators.sort(key=lambda x: x[1], reverse=True)
top_regulators = top_regulators[:5]  # Top 5
print(f"   ✓ Identified {len(top_regulators)} high-rewiring regulators:")
for reg, score in top_regulators:
    print(f"     - {reg}: {score:.3f}")

# Step 3: Generate detailed plots for top regulators
print("\n4. Generating detailed visualizations for top regulators...")
for i, (regulator, score) in enumerate(top_regulators, 1):
    print(f"\n   Processing {regulator} ({i}/{len(top_regulators)})...")

    # Check if regulator has clique data
    reg_strata = stratified_df[stratified_df['regulator'] == regulator]
    if len(reg_strata) == 0:
        print(f"     ⚠ No stratified data for {regulator}, skipping")
        continue

    try:
        # Stratum heatmap
        fig_heatmap = viz.plot_stratum_heatmap(
            regulator=regulator,
            df=stratified_df
        )
        fig_heatmap.save(output_dir / f"step2_{i}_{regulator}_heatmap.pdf")
        print(f"     ✓ Generated heatmap")

        # Gene flow diagram
        fig_flow = viz.plot_gene_flow(
            regulator=regulator,
            df=stratified_df
        )
        fig_flow.save(output_dir / f"step2_{i}_{regulator}_flow.pdf")
        print(f"     ✓ Generated flow diagram")

    except Exception as e:
        print(f"     ✗ Error processing {regulator}: {e}")

# Step 4: Summary
print("\n" + "=" * 80)
print("WORKFLOW COMPLETE")
print("=" * 80)
all_files = sorted(output_dir.glob("*.pdf"))
print(f"\nGenerated {len(all_files)} visualizations:")
for f in all_files:
    print(f"  - {f.name}")

print(f"\nOutput directory: {output_dir}")
print("\nWorkflow summary:")
print("  1. Overview plot: Scan all regulators for rewiring patterns")
print("  2. Top regulators: Identify strongest signals programmatically")
print("  3. Detailed plots: Heatmaps and flow diagrams for top hits")
print("\nNext steps:")
print("  - Review overview plot to understand global patterns")
print("  - Examine heatmaps to see gene membership changes")
print("  - Use flow diagrams to understand gained/lost/stable genes")
