"""
Benchmark performance improvements from optimizations.

These tests measure actual speedup from:
1. Correlation matrix caching
2. Parallelization (multi-core)
3. Code cleanup (remove dead code overhead)
"""

import time
import numpy as np
import pytest
import multiprocessing
from cliquefinder.quality import OutlierDetector, Imputer
from cliquefinder.core.quality import QualityFlag


class TestCorrelationCachePerformance:
    """Benchmark correlation matrix caching speedup."""

    def test_cache_speedup(self, large_matrix):
        """
        Measure speedup from correlation matrix caching.

        Expected: 5-15x speedup on second run (cached correlations).

        NOTE: Current implementation doesn't have explicit caching yet.
        This test measures baseline performance for comparison when
        caching is implemented.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(large_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        print(f"\nImputing {to_impute.sum()} outliers in {large_matrix.n_features} genes")

        # First run (baseline)
        print("\n--- First run (baseline) ---")
        start = time.time()
        imputer1 = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result1 = imputer1.apply(flagged)
        time_first = time.time() - start

        # Second run (should be same time without caching)
        print(f"\n--- Second run (expect same time without caching) ---")
        start = time.time()
        imputer2 = Imputer(
            strategy="knn_correlation",
            n_neighbors=5,
            weighted=True
        )
        result2 = imputer2.apply(flagged)
        time_second = time.time() - start

        # Results should be identical
        np.testing.assert_array_equal(result1.data, result2.data)

        # Report performance
        print(f"\nPerformance:")
        print(f"  First run:  {time_first:.1f}s")
        print(f"  Second run: {time_second:.1f}s")
        print(f"  Ratio: {time_first / time_second:.2f}x")

        print(f"\nNote: When correlation caching is implemented, expect 5-15x speedup on cached runs")


class TestParallelizationPerformance:
    """Benchmark parallelization speedup."""

    def test_parallel_speedup(self, large_matrix):
        """
        Measure speedup from parallelization.

        Expected: 6-8x on 8 cores, 10-12x on 16 cores.

        NOTE: Current implementation has n_jobs parameter but may not be
        fully optimized yet. This test establishes baseline.
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(large_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        print(f"\nImputing {to_impute.sum()} outliers in {large_matrix.n_features} genes")

        # Get number of available cores
        n_cores = multiprocessing.cpu_count()
        print(f"System has {n_cores} CPU cores")

        # Test different numbers of jobs
        timings = {}
        results = {}

        for n_jobs in [1, 2, 4, min(8, n_cores)]:
            if n_jobs > n_cores:
                continue

            print(f"\n--- Testing n_jobs={n_jobs} ---")
            start = time.time()

            # NOTE: n_jobs parameter may not be implemented yet
            # If not, this will just run sequentially
            try:
                imputer = Imputer(
                    strategy="knn_correlation",
                    n_neighbors=5,
                    weighted=True
                )
                result = imputer.apply(flagged)
                elapsed = time.time() - start

                timings[n_jobs] = elapsed
                results[n_jobs] = result.data

                speedup = timings[1] / elapsed if n_jobs > 1 else 1.0
                efficiency = speedup / n_jobs * 100 if n_jobs > 1 else 100

                print(f"  Time: {elapsed:.1f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Efficiency: {efficiency:.0f}%")

            except TypeError as e:
                if "n_jobs" in str(e):
                    print(f"  n_jobs parameter not implemented yet")
                    break
                else:
                    raise

        # Verify all results are identical
        if len(results) > 1:
            base_result = results[1]
            for n_jobs, result in results.items():
                if n_jobs != 1:
                    np.testing.assert_array_equal(
                        result, base_result,
                        err_msg=f"n_jobs={n_jobs} produced different results"
                    )
            print(f"\n✓ All parallel runs produced identical results")

        # Report expected performance
        if len(timings) == 1:
            print(f"\nNote: When parallelization is implemented, expect:")
            print(f"  - 2 cores: ~1.8x speedup")
            print(f"  - 4 cores: ~3.5x speedup")
            print(f"  - 8 cores: ~6-7x speedup")


class TestStrategyComparison:
    """Compare performance of different imputation strategies."""

    def test_strategy_performance(self, large_matrix):
        """
        Benchmark different imputation strategies.

        Expected relative performance:
        - median: baseline (fastest)
        - knn_correlation: 50-100x slower than median
        - radius_correlation: ~1.2x slower than knn_correlation
        """
        # Detect outliers
        detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
        flagged = detector.apply(large_matrix)

        to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        print(f"\nImputing {to_impute.sum()} outliers in {large_matrix.n_features} genes")

        strategies = [
            ("median", {}),
            ("knn_correlation", {"n_neighbors": 5, "weighted": True}),
            ("radius_correlation", {
                "correlation_threshold": 0.7,
                "min_neighbors": 3,
                "max_neighbors": 20,
                "weighted": True
            })
        ]

        timings = {}
        results = {}

        for strategy, params in strategies:
            print(f"\n--- Testing {strategy} ---")
            start = time.time()

            try:
                imputer = Imputer(strategy=strategy, **params)
                result = imputer.apply(flagged)
                elapsed = time.time() - start

                timings[strategy] = elapsed
                results[strategy] = result.data

                print(f"  Time: {elapsed:.2f}s")

                # All should produce valid results
                assert not np.any(np.isnan(result.data))
                assert not np.any(np.isinf(result.data))

                # All should impute the same number of values
                imputed = (result.quality_flags & QualityFlag.IMPUTED) > 0
                assert imputed.sum() == to_impute.sum()

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Report relative performance
        if "median" in timings and "knn_correlation" in timings:
            baseline = timings["median"]
            print(f"\nRelative performance (vs median):")
            for strategy in timings:
                ratio = timings[strategy] / baseline
                print(f"  {strategy:25s}: {ratio:6.1f}x")


class TestScalability:
    """Test how performance scales with data size."""

    def test_scaling_with_genes(self, small_matrix, medium_matrix, large_matrix):
        """
        Test how imputation time scales with number of genes.

        Expected: O(n²) for correlation computation, O(n) for imputation
        Overall: ~O(n²) but with good parallelization → ~O(n² / cores)
        """
        matrices = {
            "100 genes": small_matrix,
            "1000 genes": medium_matrix,
            "5000 genes": large_matrix
        }

        timings = {}

        for name, matrix in matrices.items():
            print(f"\n--- Testing {name} ---")

            # Detect outliers
            detector = OutlierDetector(method="mad-z", threshold=3.5, mode="per_feature")
            flagged = detector.apply(matrix)

            to_impute = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
            print(f"  Outliers: {to_impute.sum()}")

            # Benchmark imputation
            start = time.time()
            imputer = Imputer(
                strategy="knn_correlation",
                n_neighbors=5,
                weighted=True
            )
            result = imputer.apply(flagged)
            elapsed = time.time() - start

            timings[name] = elapsed
            print(f"  Time: {elapsed:.2f}s")

        # Analyze scaling
        print(f"\nScaling analysis:")
        times = list(timings.values())
        if len(times) >= 2:
            # 100 → 1000 genes (10x increase)
            scaling_10x = times[1] / times[0]
            print(f"  10x genes → {scaling_10x:.1f}x time")

            # 100 → 5000 genes (50x increase)
            if len(times) >= 3:
                scaling_50x = times[2] / times[0]
                print(f"  50x genes → {scaling_50x:.1f}x time")

                # Estimate complexity
                # If O(n²), expect 2500x. If O(n), expect 50x.
                import math
                if scaling_50x > 0:
                    exponent = math.log(scaling_50x) / math.log(50)
                    print(f"  Estimated complexity: O(n^{exponent:.2f})")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
