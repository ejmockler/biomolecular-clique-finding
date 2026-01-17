#!/usr/bin/env python3
"""
Example usage of run_protein_differential() for genome-wide EB-moderated differential expression.

This demonstrates the protein-level analogue of clique-level permutation testing,
using Empirical Bayes variance shrinkage (limma-style) for improved statistical power.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from cliquefinder.stats import run_protein_differential

# Simulate proteomics data
np.random.seed(42)
n_proteins = 200
n_case = 30
n_ctrl = 30

# Background expression in log2 space
data = np.random.randn(n_proteins, n_case + n_ctrl) * 0.8 + 10.0

# Add differential expression
# Strong upregulation (10 proteins): log2FC ~ 2.0
data[0:10, :n_case] += 2.0
# Moderate upregulation (20 proteins): log2FC ~ 1.0
data[10:30, :n_case] += 1.0
# Weak upregulation (30 proteins): log2FC ~ 0.5
data[30:60, :n_case] += 0.5

# Add missing values (MNAR pattern)
for i in range(n_proteins):
    for j in range(n_case + n_ctrl):
        if data[i, j] < 9.0 and np.random.rand() < 0.2:
            data[i, j] = np.nan

# Metadata
protein_ids = [f'Protein_{i:03d}' for i in range(n_proteins)]
conditions = ['CASE'] * n_case + ['CTRL'] * n_ctrl

print("EXAMPLE: Protein-level Differential Expression with EB Moderation")
print("=" * 70)
print(f"Dataset: {n_proteins} proteins × {n_case + n_ctrl} samples")
print(f"Missing values: {np.isnan(data).sum()} ({100*np.isnan(data).mean():.1f}%)")
print(f"Expected differential: 60 proteins (10 strong, 20 moderate, 30 weak)")
print()

# Run EB-moderated differential analysis
result = run_protein_differential(
    data=data,
    feature_ids=protein_ids,
    sample_condition=pd.Series(conditions),
    contrast=('CASE', 'CTRL'),
    eb_moderation=True,
    target_genes=['Protein_005', 'Protein_015', 'Protein_035'],  # Example targets
    verbose=True
)

# Display results
print("\n\nRESULTS")
print("=" * 70)
print(f"Proteins tested: {result.p_value.notna().sum()}")
print(f"Significant (FDR < 0.05): {(result.p_value < 0.05).sum()}")
print(f"Significant (FDR < 0.01): {(result.p_value < 0.01).sum()}")

print("\n\nTop 10 Differential Proteins:")
print("-" * 70)
top10 = result.nsmallest(10, 'p_value')[
    ['feature_id', 'log2fc', 't_statistic', 'p_value', 'is_target']
]
print(top10.to_string(index=False))

# Enrichment in expected differential proteins
expected_diff = set([f'Protein_{i:03d}' for i in range(60)])
sig = set(result[result.p_value < 0.05]['feature_id'])
true_pos = len(sig & expected_diff)

print(f"\n\nEnrichment Analysis:")
print("-" * 70)
print(f"True positives: {true_pos} / {len(expected_diff)} ({100*true_pos/60:.1f}%)")
print(f"False positives: {len(sig - expected_diff)}")
print(f"Precision: {100*true_pos/len(sig):.1f}%" if len(sig) > 0 else "N/A")

# Variance shrinkage statistics
shrinkage = (result['sigma2'] - result['sigma2_post']).abs() / result['sigma2']
print(f"\n\nVariance Shrinkage (EB Moderation):")
print("-" * 70)
print(f"Mean shrinkage: {100*shrinkage.mean():.1f}%")
print(f"Median shrinkage: {100*shrinkage.median():.1f}%")
print(f"Max shrinkage: {100*shrinkage.max():.1f}%")

# Save results
result.to_csv('protein_differential_results.csv', index=False)
print(f"\n\n✓ Results saved to protein_differential_results.csv")
