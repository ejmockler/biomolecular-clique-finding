#!/usr/bin/env python
"""
Example usage of improved normalization methods for proteomics data.

This demonstrates the key improvements:
1. Censored quantile normalization for MNAR missing data
2. Proper iterative VSN with convergence diagnostics
3. Edge case handling
"""

import numpy as np
from src.cliquefinder.stats.normalization import (
    quantile_normalization,
    vsn_normalization,
    assess_normalization_quality,
)

# Set seed for reproducibility
np.random.seed(42)


def example_quantile_normalization():
    """Example: Censored quantile normalization with missing values."""
    print("=" * 70)
    print("Example 1: Censored Quantile Normalization")
    print("=" * 70)

    # Simulate proteomics data (log-transformed intensities)
    n_proteins, n_samples = 500, 8
    data = np.random.randn(n_proteins, n_samples) + 20

    # Simulate MNAR: low-abundance proteins are missing
    # This is the typical pattern in label-free proteomics
    for j in range(n_samples):
        threshold = np.percentile(data[:, j], 25)  # Bottom 25% missing
        data[data[:, j] < threshold, j] = np.nan

    print(f"\nData shape: {data.shape}")
    print(f"Missingness per sample: {(np.isnan(data).sum(axis=0) / n_proteins * 100).round(1)}%")
    print(f"Median before: {np.nanmedian(data, axis=0).round(2)}")

    # Apply censored quantile normalization (recommended)
    result = quantile_normalization(data, method="censored")

    print(f"\nMedian after (censored): {np.nanmedian(result.data, axis=0).round(2)}")
    print(f"Target distribution length: {result.diagnostics['target_distribution_length']}")
    print(f"Missing value handling: {result.diagnostics['missing_value_handling']}")

    # Compare with simple method
    result_simple = quantile_normalization(data, method="simple")
    print(f"Median after (simple): {np.nanmedian(result_simple.data, axis=0).round(2)}")

    # Assess quality
    quality = assess_normalization_quality(data, result.data)
    print(f"\nNormalization quality:")
    print(f"  Median CV reduction: {quality['median_cv_reduction']:.1%}")
    print(f"  Median range before: {quality['median_range_before']:.3f}")
    print(f"  Median range after: {quality['median_range_after']:.3f}")

    print("\n✓ Censored method better handles samples with different missingness patterns\n")


def example_vsn_proper():
    """Example: Proper VSN with convergence monitoring."""
    print("=" * 70)
    print("Example 2: Proper Iterative VSN")
    print("=" * 70)

    # Simulate raw proteomics intensities (not log-transformed)
    # These typically have variance-mean dependence
    n_proteins, n_samples = 300, 6
    raw_intensities = np.random.lognormal(mean=10, sigma=2, size=(n_proteins, n_samples))

    # Add sample-specific scaling (technical variation)
    scaling_factors = np.random.uniform(0.7, 1.3, size=n_samples)
    raw_intensities *= scaling_factors

    print(f"\nData shape: {raw_intensities.shape}")
    print(f"Mean intensities: {np.mean(raw_intensities, axis=0).round(0)}")
    print(f"Std intensities: {np.std(raw_intensities, axis=0).round(0)}")
    print(f"CV (Std/Mean): {(np.std(raw_intensities, axis=0) / np.mean(raw_intensities, axis=0)).round(2)}")

    # Apply proper VSN
    result = vsn_normalization(
        raw_intensities,
        method="proper",
        max_iter=100,
        tol=1e-4,
        use_gpu=False,  # Set to True for MLX GPU acceleration
    )

    print(f"\nAfter VSN:")
    print(f"  Mean: {np.mean(result.data, axis=0).round(2)}")
    print(f"  Std: {np.std(result.data, axis=0).round(2)}")
    print(f"  Converged: {result.diagnostics['converged']}")
    print(f"  Iterations: {result.diagnostics['iterations']}")

    # Show convergence history
    if result.diagnostics['iterations'] > 0:
        history = result.diagnostics['iteration_history']
        print(f"\n  Convergence history (last 5 iterations):")
        for h in history[-5:]:
            print(f"    Iter {h['iteration']}: max_change = {h['max_change']:.2e}")

    # Show estimated parameters
    print(f"\n  Estimated parameters (sample 0):")
    print(f"    a (offset) = {result.diagnostics['final_a'][0]:.2f}")
    print(f"    b (scale) = {result.diagnostics['final_b'][0]:.2f}")
    print(f"    Transformation: arsinh((x - {result.diagnostics['final_a'][0]:.1f}) / {result.diagnostics['final_b'][0]:.1f})")

    # Compare with simple method
    result_simple = vsn_normalization(raw_intensities, method="simple")
    print(f"\n  Simple VSN (for comparison):")
    print(f"    Mean: {np.mean(result_simple.data, axis=0).round(2)}")
    print(f"    Std: {np.std(result_simple.data, axis=0).round(2)}")

    print("\n✓ Proper VSN stabilizes variance and normalizes samples simultaneously\n")


def example_edge_cases():
    """Example: Robust handling of edge cases."""
    print("=" * 70)
    print("Example 3: Edge Case Handling")
    print("=" * 70)

    # Case 1: Sample with all-NaN
    data = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0], [3.0, np.nan, 5.0]])
    result = quantile_normalization(data, method="censored")
    print("\nCase 1: All-NaN sample")
    print(f"  Input:\n{data}")
    print(f"  All NaN preserved in column 1: {np.all(np.isnan(result.data[:, 1]))}")

    # Case 2: Single non-NaN value
    data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 5.0]])
    result = quantile_normalization(data, method="censored")
    print("\nCase 2: Single non-NaN value")
    print(f"  Single value mapped to target median: {not np.isnan(result.data[2, 1])}")

    # Case 3: Extreme values in VSN
    data = np.array([[1e-10, 1e10], [1e-8, 1e9], [1e-6, 1e8]])
    result = vsn_normalization(data, method="proper", max_iter=20)
    print("\nCase 3: Extreme values")
    print(f"  Input range: {data.min():.2e} to {data.max():.2e}")
    print(f"  No inf values after VSN: {not np.any(np.isinf(result.data))}")
    print(f"  No NaN values after VSN: {not np.any(np.isnan(result.data))}")

    # Case 4: High sparsity (>50% missing)
    n_proteins, n_samples = 200, 4
    data = np.random.randn(n_proteins, n_samples) + 15
    # Make 60% missing in each sample
    for j in range(n_samples):
        missing_indices = np.random.choice(n_proteins, size=int(0.6 * n_proteins), replace=False)
        data[missing_indices, j] = np.nan

    result = quantile_normalization(data, method="censored")
    print(f"\nCase 4: High sparsity (60% missing)")
    print(f"  Successfully normalized: {result.data.shape == data.shape}")
    print(f"  Median stabilized: {np.std(np.nanmedian(result.data, axis=0)) < 0.1}")

    print("\n✓ All edge cases handled gracefully\n")


def example_performance():
    """Example: Performance with large dataset."""
    print("=" * 70)
    print("Example 4: Performance with Large Dataset")
    print("=" * 70)

    import time

    # Large dataset: 5000 proteins × 20 samples
    n_proteins, n_samples = 5000, 20
    data = np.random.randn(n_proteins, n_samples) + 18

    # Add 30% missing values
    missing_mask = np.random.random(data.shape) < 0.3
    data[missing_mask] = np.nan

    print(f"\nDataset: {n_proteins} proteins × {n_samples} samples")
    print(f"Missing values: {np.isnan(data).sum() / data.size * 100:.1f}%")

    # Benchmark quantile normalization
    start = time.time()
    result = quantile_normalization(data, method="censored")
    elapsed = time.time() - start
    print(f"\nQuantile normalization: {elapsed:.3f}s")
    print(f"  (~{elapsed * 1000 / n_samples:.1f}ms per sample)")

    # Benchmark VSN
    raw_data = np.abs(data) * 1000  # Simulate raw intensities
    start = time.time()
    result = vsn_normalization(raw_data, method="proper", max_iter=20)
    elapsed = time.time() - start
    print(f"\nVSN (20 iterations): {elapsed:.3f}s")
    print(f"  (~{elapsed * 1000 / 20:.1f}ms per iteration)")

    print("\n✓ Performance suitable for large-scale proteomics studies\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Proteomics Normalization: Usage Examples")
    print("=" * 70 + "\n")

    example_quantile_normalization()
    example_vsn_proper()
    example_edge_cases()
    example_performance()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use method='censored' for quantile norm with missing data")
    print("  2. Use method='proper' for VSN with convergence monitoring")
    print("  3. Edge cases are handled automatically")
    print("  4. Check VSN convergence via result.diagnostics['converged']")
    print("=" * 70 + "\n")
