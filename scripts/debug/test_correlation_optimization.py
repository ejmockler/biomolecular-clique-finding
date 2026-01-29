#!/usr/bin/env python
"""
Verification script for correlation matrix precomputation optimization.

This demonstrates that the optimization:
1. Produces identical results to the original implementation
2. Significantly reduces computation time for multi-regulator analysis
"""

import numpy as np
import pandas as pd
import time
from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.knowledge.clique_validator import CliqueValidator


def create_test_matrix(n_genes=3000, n_samples=100):
    """Create a test BioMatrix for benchmarking."""
    np.random.seed(42)
    data = np.random.randn(n_genes, n_samples)

    feature_ids = pd.Index([f"GENE{i}" for i in range(n_genes)])
    sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])

    # Create stratified metadata (2 phenotypes × 2 sexes = 4 conditions)
    metadata = pd.DataFrame({
        'phenotype': ['CASE'] * 50 + ['CTRL'] * 50,
        'Sex': ['Male', 'Female'] * 50
    }, index=sample_ids)

    quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=int)

    return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)


def simulate_regulator_modules(n_regulators=100, avg_targets=50, gene_universe=None):
    """Simulate regulator modules with overlapping target genes."""
    np.random.seed(42)
    modules = []

    for i in range(n_regulators):
        # Each regulator has 30-70 target genes
        n_targets = np.random.randint(30, 71)
        targets = set(np.random.choice(gene_universe, size=n_targets, replace=False))

        modules.append({
            'regulator_name': f"TF{i}",
            'target_genes': targets
        })

    return modules


def benchmark_without_precomputation(validator, modules, conditions, method="max"):
    """Benchmark original approach: compute correlation for each regulator."""
    start = time.time()

    results = []
    for i, module in enumerate(modules):
        for condition in conditions:
            # This computes correlation matrix from scratch each time
            corr = validator.compute_correlation_matrix(
                list(module['target_genes']),
                condition=condition,
                method=method
            )
            results.append(corr)

    elapsed = time.time() - start
    return elapsed, results


def benchmark_with_precomputation(validator, modules, conditions, method="max"):
    """Benchmark optimized approach: precompute once, then subset."""
    # Collect union of all target genes
    all_targets = set()
    for module in modules:
        all_targets.update(module['target_genes'])

    # Precompute correlation matrices
    start_precompute = time.time()
    validator.precompute_correlation_matrices(
        genes=all_targets,
        conditions=conditions,
        method=method
    )
    precompute_time = time.time() - start_precompute

    # Now subset for each regulator (should be very fast)
    start_subset = time.time()
    results = []
    for module in modules:
        for condition in conditions:
            # This extracts from precomputed matrix (O(k^2) indexing)
            corr = validator.compute_correlation_matrix(
                list(module['target_genes']),
                condition=condition,
                method=method
            )
            results.append(corr)

    subset_time = time.time() - start_subset
    total_time = precompute_time + subset_time

    return total_time, results, precompute_time, subset_time


def verify_results_identical(results1, results2):
    """Verify that two sets of correlation matrices are identical."""
    assert len(results1) == len(results2), "Different number of results"

    max_diff = 0.0
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        # Check that indices match
        assert list(r1.index) == list(r2.index), f"Index mismatch at result {i}"
        assert list(r1.columns) == list(r2.columns), f"Column mismatch at result {i}"

        # Check that values are identical (within floating point precision)
        diff = np.abs(r1.values - r2.values).max()
        max_diff = max(max_diff, diff)

        np.testing.assert_allclose(
            r1.values, r2.values,
            rtol=1e-10, atol=1e-12,
            err_msg=f"Correlation matrices differ at result {i}"
        )

    return max_diff


def main():
    print("=" * 80)
    print("CORRELATION MATRIX PRECOMPUTATION OPTIMIZATION TEST")
    print("=" * 80)
    print()

    # Create test data
    print("Creating test data...")
    matrix = create_test_matrix(n_genes=3000, n_samples=100)
    print(f"  Matrix: {matrix.n_features} genes × {matrix.n_samples} samples")

    # Simulate regulator modules
    n_regulators = 100
    gene_universe = list(matrix.feature_ids)
    modules = simulate_regulator_modules(n_regulators, avg_targets=50, gene_universe=gene_universe)
    print(f"  Modules: {n_regulators} regulators")

    # Count total target genes
    all_targets = set()
    for module in modules:
        all_targets.update(module['target_genes'])
    print(f"  Unique targets: {len(all_targets)} genes")

    # Create validator
    validator = CliqueValidator(
        matrix=matrix,
        stratify_by=['phenotype', 'Sex'],
        min_samples=10,
        precompute=True  # Precompute condition data
    )

    conditions = validator.get_available_conditions()
    print(f"  Conditions: {conditions}")
    print()

    # Calculate expected number of correlation computations
    n_computations = n_regulators * len(conditions)
    print(f"Total correlation computations: {n_computations} ({n_regulators} regulators × {len(conditions)} conditions)")
    print()

    # Benchmark WITHOUT precomputation (original approach)
    print("Benchmarking WITHOUT precomputation (original approach)...")
    # Clear any cached correlations
    validator.clear_cache(correlation_only=True)

    time_without, results_without = benchmark_without_precomputation(
        validator, modules, conditions, method="max"
    )
    print(f"  Time: {time_without:.2f} seconds")
    print(f"  Avg per computation: {time_without / n_computations * 1000:.1f} ms")
    print()

    # Benchmark WITH precomputation (optimized approach)
    print("Benchmarking WITH precomputation (optimized approach)...")
    # Create fresh validator to avoid caching effects
    validator2 = CliqueValidator(
        matrix=matrix,
        stratify_by=['phenotype', 'Sex'],
        min_samples=10,
        precompute=True
    )

    time_with, results_with, precompute_time, subset_time = benchmark_with_precomputation(
        validator2, modules, conditions, method="max"
    )

    print(f"  Precomputation time: {precompute_time:.2f} seconds")
    print(f"  Subsetting time: {subset_time:.2f} seconds")
    print(f"  Total time: {time_with:.2f} seconds")
    print(f"  Avg per subset: {subset_time / n_computations * 1000:.1f} ms")
    print()

    # Verify results are identical
    print("Verifying results are identical...")
    max_diff = verify_results_identical(results_without, results_with)
    print(f"  ✓ All {len(results_without)} correlation matrices are identical")
    print(f"  Max difference: {max_diff:.2e} (within floating point precision)")
    print()

    # Report speedup
    speedup = time_without / time_with
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  Original approach: {time_without:.2f} seconds")
    print(f"  Optimized approach: {time_with:.2f} seconds")
    print(f"  Speedup: {speedup:.1f}x faster")
    print(f"  Time saved: {time_without - time_with:.2f} seconds ({(1 - time_with/time_without)*100:.1f}% reduction)")
    print()

    # Cache statistics
    stats = validator2.get_cache_stats()
    print(f"  Precomputed matrices: {stats['n_precomputed_corr']}")
    print(f"  Precomputed genes: {stats['n_precomputed_genes']}")
    print(f"  Memory usage: {stats['precomputed_corr_mb']:.1f} MB")
    print()

    # Extrapolate to real analysis
    print("EXTRAPOLATION TO REAL ANALYSIS")
    print("=" * 80)
    real_n_regulators = 8209
    real_n_conditions = 4
    real_computations = real_n_regulators * real_n_conditions

    estimated_original = (time_without / n_computations) * real_computations
    estimated_optimized = precompute_time * (real_n_conditions / len(conditions)) + (subset_time / n_computations) * real_computations

    print(f"  Real analysis: {real_n_regulators} regulators × {real_n_conditions} conditions = {real_computations} computations")
    print(f"  Estimated time (original): {estimated_original:.1f} seconds ({estimated_original/60:.1f} minutes)")
    print(f"  Estimated time (optimized): {estimated_optimized:.1f} seconds ({estimated_optimized/60:.1f} minutes)")
    print(f"  Estimated speedup: {estimated_original/estimated_optimized:.1f}x")
    print()
    print("✓ Optimization verified and working correctly!")
    print()


if __name__ == "__main__":
    main()
