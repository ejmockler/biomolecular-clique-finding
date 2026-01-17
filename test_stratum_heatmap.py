#!/usr/bin/env python3
"""
Test script for Stratum Heatmap Grid visualization.

This validates the implementation works with the actual data.
"""

import pandas as pd
from pathlib import Path
from cliquefinder.viz import CliqueVisualizer

# Load data
data_path = Path("/Users/noot/Documents/biomolecular-clique-finding/results/cliques/cliques.csv")
df = pd.read_csv(data_path)

print("Data loaded successfully!")
print(f"Total regulators: {df['regulator'].nunique()}")
print(f"Conditions: {df['condition'].unique()}")
print()

# Test with MAPT regulator (from the example data)
regulator = "MAPT"
print(f"Creating stratum heatmap for {regulator}...")

# Create visualizer
viz = CliqueVisualizer(style="paper")

# Generate heatmap
fig = viz.plot_stratum_heatmap(regulator, df)

# Save to file
output_path = Path("/Users/noot/Documents/biomolecular-clique-finding/test_mapt_heatmap.pdf")
fig.save(output_path)

print(f"Saved to: {output_path}")
print()

# Print metadata
print("Figure metadata:")
for key, value in fig.metadata.items():
    print(f"  {key}: {value}")
print()

print("Success! The visualization has been created.")
print()
print("Visual checks to perform:")
print("  1. Layout is 2×2 with CTRL (top) and CASE (bottom)")
print("  2. Gene labels are italicized on the left")
print("  3. Color tints: orange for CTRL, blue for CASE")
print("  4. Margin annotations show CTRL-only, CASE-only, Stable genes")
print("  5. Genes are clustered to group co-occurring patterns")
