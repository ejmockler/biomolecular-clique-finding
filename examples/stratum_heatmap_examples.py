#!/usr/bin/env python3
"""
Example usage of the Stratum Heatmap Grid visualization.

This script demonstrates how to create publication-quality heatmaps
for comparing regulatory clique membership across demographic strata.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz import CliqueVisualizer, FigureCollection

# Configuration
DATA_PATH = Path("results/cliques/cliques.csv")
OUTPUT_DIR = Path("figures/stratum_heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Stratum Heatmap Grid Examples")
print("=" * 60)
print()

# Load data
print("Loading stratified cliques data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total regulators: {df['regulator'].nunique()}")
print(f"  Conditions: {df['condition'].unique().tolist()}")
print()

# =========================================================================
# Example 1: Single regulator (paper style)
# =========================================================================
print("Example 1: Single regulator visualization (paper style)")
print("-" * 60)

regulator = "MAPT"
viz_paper = CliqueVisualizer(style="paper")
fig = viz_paper.plot_stratum_heatmap(regulator, df)

output_path = OUTPUT_DIR / f"{regulator}_paper.pdf"
fig.save(output_path, dpi=300)
print(f"  Saved: {output_path}")
print(f"  Metadata: {fig.metadata['n_stable']} stable, "
      f"{fig.metadata['n_ctrl_only']} CTRL-only, "
      f"{fig.metadata['n_case_only']} CASE-only")
print()

# =========================================================================
# Example 2: Presentation style (larger fonts, bold colors)
# =========================================================================
print("Example 2: Same regulator with presentation style")
print("-" * 60)

viz_presentation = CliqueVisualizer(style="presentation")
fig = viz_presentation.plot_stratum_heatmap(regulator, df, figsize=(12, 14))

output_path = OUTPUT_DIR / f"{regulator}_presentation.png"
fig.save(output_path, dpi=150)
print(f"  Saved: {output_path}")
print()

# =========================================================================
# Example 3: Multiple regulators in batch
# =========================================================================
print("Example 3: Batch processing multiple regulators")
print("-" * 60)

# Select regulators of interest (top 5 by total clique size)
regulator_stats = []
for regulator in df['regulator'].unique():
    reg_data = df[df['regulator'] == regulator]
    total_genes = set()
    for genes_str in reg_data['clique_genes']:
        genes = [g.strip() for g in genes_str.split(',')]
        total_genes.update(genes)
    regulator_stats.append({
        'regulator': regulator,
        'n_genes': len(total_genes)
    })

stats_df = pd.DataFrame(regulator_stats).sort_values('n_genes', ascending=False)
top_regulators = stats_df.head(5)['regulator'].tolist()

print(f"  Processing top {len(top_regulators)} regulators by gene count:")
for reg in top_regulators:
    print(f"    - {reg}")
print()

# Create collection
collection = FigureCollection()
viz = CliqueVisualizer(style="paper")

for regulator in top_regulators:
    try:
        fig = viz.plot_stratum_heatmap(regulator, df)
        collection.add(regulator, fig)
        print(f"  ✓ {regulator}")
    except Exception as e:
        print(f"  ✗ {regulator}: {e}")

# Save all as PDFs
saved_paths = collection.save_all(OUTPUT_DIR / "batch", format="pdf", dpi=300)
print(f"\n  Saved {len(saved_paths)} figures to {OUTPUT_DIR / 'batch'}")
print()

# =========================================================================
# Example 4: HTML report with embedded figures
# =========================================================================
print("Example 4: Generate HTML report with top regulators")
print("-" * 60)

report_path = OUTPUT_DIR / "stratum_heatmap_report.html"
collection.to_html_report(
    report_path,
    title="Regulatory Clique Stratum Analysis",
    description=(
        "Comparison of gene membership in regulatory cliques across "
        "demographic strata (CTRL/CASE × Female/Male). "
        "Top regulators selected by total unique gene count."
    )
)
print(f"  Saved: {report_path}")
print(f"  Open in browser to view interactive report")
print()

# =========================================================================
# Example 5: Finding interesting patterns
# =========================================================================
print("Example 5: Finding regulators with disease-specific patterns")
print("-" * 60)

# Calculate disease specificity metrics
specificity_stats = []

for regulator in df['regulator'].unique():
    reg_data = df[df['regulator'] == regulator]

    # Extract gene sets
    ctrl_genes = set()
    case_genes = set()

    for _, row in reg_data.iterrows():
        genes = set(g.strip() for g in row['clique_genes'].split(','))
        if row['condition'].startswith('CTRL'):
            ctrl_genes.update(genes)
        else:
            case_genes.update(genes)

    # Calculate metrics
    ctrl_only = ctrl_genes - case_genes
    case_only = case_genes - ctrl_genes
    stable = ctrl_genes & case_genes

    total = len(ctrl_genes | case_genes)
    if total > 0:
        disease_specificity = len(case_only) / total
        health_specificity = len(ctrl_only) / total
        stability = len(stable) / total

        specificity_stats.append({
            'regulator': regulator,
            'total_genes': total,
            'stable': len(stable),
            'ctrl_only': len(ctrl_only),
            'case_only': len(case_only),
            'disease_specificity': disease_specificity,
            'health_specificity': health_specificity,
            'stability': stability
        })

spec_df = pd.DataFrame(specificity_stats)

# Find most disease-specific regulators
disease_specific = spec_df.nlargest(3, 'disease_specificity')
print("\n  Most disease-specific regulators (genes gained in ALS):")
for _, row in disease_specific.iterrows():
    print(f"    {row['regulator']:10s} - {row['disease_specificity']:.1%} disease-specific "
          f"({row['case_only']}/{row['total_genes']} genes)")

# Find most health-specific regulators
health_specific = spec_df.nlargest(3, 'health_specificity')
print("\n  Most health-specific regulators (genes lost in ALS):")
for _, row in health_specific.iterrows():
    print(f"    {row['regulator']:10s} - {row['health_specificity']:.1%} health-specific "
          f"({row['ctrl_only']}/{row['total_genes']} genes)")

# Find most stable regulators
stable_regulators = spec_df.nlargest(3, 'stability')
print("\n  Most stable regulators (consistent across conditions):")
for _, row in stable_regulators.iterrows():
    print(f"    {row['regulator']:10s} - {row['stability']:.1%} stable "
          f"({row['stable']}/{row['total_genes']} genes)")

print()

# =========================================================================
# Summary
# =========================================================================
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"Generated visualizations in: {OUTPUT_DIR}")
print()
print("Key patterns to look for in the heatmaps:")
print("  1. Top-row-only genes (orange) → Lost in disease")
print("  2. Bottom-row-only genes (blue) → Gained in disease")
print("  3. Full-row genes → Stable across conditions")
print("  4. Sex-specific patterns → Left vs right columns")
print()
print("The 2×2 layout builds spatial memory:")
print("  - Top = Healthy, Bottom = Disease")
print("  - Left = Female, Right = Male")
print()
