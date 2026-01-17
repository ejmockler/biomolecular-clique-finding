#!/usr/bin/env python3
"""
Examples for using the plot_regulator_overview() visualization.

This script demonstrates different ways to use the regulator overview plot
to identify interesting rewiring patterns in the ALS clique-finding analysis.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz.cliques import CliqueVisualizer

# Load data
data_dir = Path("results/cliques")
rewiring_df = pd.read_csv(data_dir / "regulator_rewiring_stats.csv")
stratified_df = pd.read_csv(data_dir / "cliques.csv")

# Create output directory
output_dir = Path("figures/regulator_overview_examples")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("REGULATOR OVERVIEW PLOT EXAMPLES")
print("=" * 80)

# Example 1: Default overview - show all regulators
print("\n1. Default Overview (all regulators)")
print("-" * 80)
viz = CliqueVisualizer(style="paper")
fig1 = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.0,
    label_top_n=20
)
fig1.save(output_dir / "01_default_overview.pdf")
print(f"Created: {output_dir / '01_default_overview.pdf'}")
print(f"  - Total regulators: {fig1.metadata['n_regulators']}")
print(f"  - Both sexes: {fig1.metadata['n_both_sexes']}")
print(f"  - Male-specific: {fig1.metadata['n_male_specific']}")
print(f"  - Female-specific: {fig1.metadata['n_female_specific']}")
print(f"  - Weak rewiring: {fig1.metadata['n_weak']}")

# Example 2: Filter for strong rewiring signals
print("\n2. Strong Rewiring Only (|score| >= 0.3)")
print("-" * 80)
viz2 = CliqueVisualizer(style="paper")
fig2 = viz2.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.3,
    label_top_n=15
)
fig2.save(output_dir / "02_strong_rewiring.pdf")
print(f"Created: {output_dir / '02_strong_rewiring.pdf'}")
print(f"  - Filtered to {fig2.metadata['n_regulators']} regulators with strong rewiring")
print(f"  - Both sexes: {fig2.metadata['n_both_sexes']}")
print(f"  - Male-specific: {fig2.metadata['n_male_specific']}")
print(f"  - Female-specific: {fig2.metadata['n_female_specific']}")

# Example 3: Presentation style for talks
print("\n3. Presentation Style (larger fonts, high contrast)")
print("-" * 80)
viz3 = CliqueVisualizer(style="presentation")
fig3 = viz3.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.35,
    label_top_n=10,
    point_alpha=0.8,
    figsize=(16, 12)
)
fig3.save(output_dir / "03_presentation_style.png", dpi=150)
print(f"Created: {output_dir / '03_presentation_style.png'}")
print(f"  - Optimized for presentations")
print(f"  - Showing top {fig3.metadata['n_regulators']} regulators")

# Example 4: Notebook style for interactive exploration
print("\n4. Notebook Style (interactive exploration)")
print("-" * 80)
viz4 = CliqueVisualizer(style="notebook")
fig4 = viz4.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.25,
    label_top_n=30,
    figsize=(12, 9)
)
fig4.save(output_dir / "04_notebook_style.png", dpi=100)
print(f"Created: {output_dir / '04_notebook_style.png'}")
print(f"  - Optimized for Jupyter notebooks")

# Example 5: Identify top rewiring regulators programmatically
print("\n5. Programmatic Analysis - Top Rewiring Regulators")
print("-" * 80)

# Aggregate metrics manually to identify top regulators
reg_metrics = []
for regulator in rewiring_df['regulator'].unique():
    reg_rewiring = rewiring_df[rewiring_df['regulator'] == regulator]
    reg_strata = stratified_df[stratified_df['regulator'] == regulator]

    if len(reg_strata) == 0:
        continue

    # Get max rewiring score
    max_rewiring = reg_rewiring['rewiring_score'].abs().max()
    max_coherence = reg_strata['coherence_ratio'].max()

    reg_metrics.append({
        'regulator': regulator,
        'max_abs_rewiring': max_rewiring,
        'max_coherence': max_coherence
    })

# Convert to DataFrame and sort
metrics_df = pd.DataFrame(reg_metrics)
metrics_df = metrics_df.sort_values('max_abs_rewiring', ascending=False)

print("\nTop 10 regulators by rewiring score:")
print(metrics_df.head(10).to_string(index=False))

# Find regulators with both high rewiring AND high coherence
high_quality = metrics_df[
    (metrics_df['max_abs_rewiring'] >= 0.4) &
    (metrics_df['max_coherence'] >= 0.15)
]
print(f"\nHigh-quality rewiring (rewiring >= 0.4, coherence >= 0.15): {len(high_quality)} regulators")
if len(high_quality) > 0:
    print(high_quality.to_string(index=False))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Generated {len(list(output_dir.glob('*.pdf'))) + len(list(output_dir.glob('*.png')))} example plots")
print(f"Output directory: {output_dir}")
print("\nUsage tips:")
print("  - Start with default overview to scan all regulators")
print("  - Filter by min_rewiring_score to focus on strong signals")
print("  - Look for top-right quadrant: high-quality gains in ALS")
print("  - Look for top-left quadrant: high-quality losses in ALS")
print("  - Color indicates sex-specific vs pan-sex effects")
print("  - Point size shows clique size (biological impact)")
