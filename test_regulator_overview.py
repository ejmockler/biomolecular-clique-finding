#!/usr/bin/env python3
"""
Test script for the regulator overview plot.

This script verifies that the plot_regulator_overview() method works correctly
with the actual data files.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz.cliques import CliqueVisualizer

# Load data
data_dir = Path("results/cliques")
rewiring_df = pd.read_csv(data_dir / "regulator_rewiring_stats.csv")
stratified_df = pd.read_csv(data_dir / "cliques.csv")

print(f"Loaded {len(rewiring_df)} rewiring records")
print(f"Loaded {len(stratified_df)} stratified clique records")
print(f"Unique regulators in rewiring_df: {rewiring_df['regulator'].nunique()}")
print(f"Unique regulators in stratified_df: {stratified_df['regulator'].nunique()}")

# Create visualizer
viz = CliqueVisualizer(style="paper")

# Generate overview plot
print("\nGenerating regulator overview plot...")
fig = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.0,
    label_top_n=20
)

# Display metadata
print("\nPlot metadata:")
for key, value in fig.metadata.items():
    print(f"  {key}: {value}")

# Save the figure
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "regulator_overview.png"
fig.save(output_path, format="png", dpi=150)
print(f"\nSaved plot to: {output_path}")

# Also save as PDF for publication quality
output_path_pdf = output_dir / "regulator_overview.pdf"
fig.save(output_path_pdf, format="pdf")
print(f"Saved PDF to: {output_path_pdf}")

print("\nTest completed successfully!")
