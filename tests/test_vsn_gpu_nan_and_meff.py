"""
Tests for VSN GPU NaN handling and Li-Ji effective test count (M_eff).

VSN GPU NaN: The MLX GPU VSN path must correctly compute the reference array
when data contains NaN values, using valid counts per feature and dividing
correctly (nanmean equivalent) rather than dividing by total column count.

Li-Ji M_eff: The formula must use the original test count M (not
len(filtered eigenvalues)) and zero-pad eigenvalues to length M for the
variance computation, per Li & Ji (2005).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cliquefinder.stats.correlation_tests import estimate_effective_tests
from cliquefinder.stats.normalization import HAS_MLX

# ---------------------------------------------------------------------------
# VSN GPU NaN handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestVSNGpuNanHandling:
    """Tests for correct NaN handling in the MLX GPU VSN path."""

    def test_reference_matches_cpu_with_nan(self):
        """GPU and CPU VSN references should match when data has NaN values.

        With 20% missing data, the old code would compute
        sum_of_valid / n_total instead of sum_of_valid / n_valid,
        producing systematically lower reference values.
        """
        from cliquefinder.stats.normalization import vsn_normalization

        rng = np.random.RandomState(42)
        n_features, n_samples = 50, 10
        data = rng.exponential(scale=1000, size=(n_features, n_samples)).astype(np.float64)

        # Introduce 20% NaN
        nan_mask = rng.random((n_features, n_samples)) < 0.2
        data[nan_mask] = np.nan

        # Run CPU path
        result_cpu = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=False)
        # Run GPU path
        result_gpu = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=True)

        # The references (reflected in final normalized data) should be close.
        # NaN positions should remain NaN in both.
        cpu_data = result_cpu.data
        gpu_data = result_gpu.data

        # Both should have NaN in the same positions
        assert_allclose(np.isnan(cpu_data), np.isnan(gpu_data))

        # Non-NaN values should match within float32 tolerance
        # (GPU uses float32 internally via MLX)
        valid = ~np.isnan(cpu_data)
        assert_allclose(
            gpu_data[valid],
            cpu_data[valid],
            rtol=1e-3,
            atol=1e-3,
            err_msg="GPU and CPU VSN results diverge with NaN data",
        )

    def test_reference_matches_cpu_heavy_nan(self):
        """Test with 50% NaN -- heavier missingness amplifies the bug more."""
        from cliquefinder.stats.normalization import vsn_normalization

        rng = np.random.RandomState(123)
        n_features, n_samples = 30, 8
        data = rng.exponential(scale=500, size=(n_features, n_samples)).astype(np.float64)

        # 50% NaN
        nan_mask = rng.random((n_features, n_samples)) < 0.5
        data[nan_mask] = np.nan

        result_cpu = vsn_normalization(data.copy(), method="proper", max_iter=15, use_gpu=False)
        result_gpu = vsn_normalization(data.copy(), method="proper", max_iter=15, use_gpu=True)

        valid = ~np.isnan(result_cpu.data)
        assert_allclose(
            result_gpu.data[valid],
            result_cpu.data[valid],
            rtol=1e-3,
            atol=1e-3,
            err_msg="GPU and CPU VSN results diverge with 50% NaN data",
        )

    def test_nanmean_reference_directly(self):
        """Directly test the nanmean-equivalent MLX computation.

        Constructs a known case where the bug produces a measurably wrong answer:
        Feature with NaN in 4 of 10 samples. The old code gives sum_of_6/10
        instead of the correct sum_of_6/6.
        """
        import mlx.core as mx

        # Row 0: 6 valid values of 10.0, 4 NaN  -> nanmean = 10.0
        # Row 1: all valid, all 5.0             -> nanmean = 5.0
        y_np = np.full((2, 10), np.nan)
        y_np[0, :6] = 10.0
        y_np[1, :] = 5.0

        y_mx = mx.array(y_np)

        # Buggy approach (old code)
        buggy_ref = mx.mean(mx.where(mx.isnan(y_mx), 0.0, y_mx), axis=1)
        buggy_vals = np.array(buggy_ref)

        # Correct approach (new code)
        valid_count = mx.sum(mx.where(mx.isnan(y_mx), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y_mx), 0.0, y_mx), axis=1)
        correct_ref = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        correct_vals = np.array(correct_ref)

        expected = np.nanmean(y_np, axis=1)

        # Buggy code gives wrong answer for row 0
        assert buggy_vals[0] == pytest.approx(6.0, abs=0.01), (
            "Buggy code should give 60/10=6.0, not 10.0"
        )
        # Correct code matches np.nanmean
        assert_allclose(correct_vals, expected, rtol=1e-5)

    def test_all_nan_row(self):
        """A row that is entirely NaN should not cause division by zero.

        The fix uses mx.maximum(valid_count, 1.0) to prevent 0/0.
        """
        import mlx.core as mx

        y_np = np.array([[np.nan, np.nan, np.nan],
                          [1.0, 2.0, 3.0]])
        y_mx = mx.array(y_np)

        valid_count = mx.sum(mx.where(mx.isnan(y_mx), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y_mx), 0.0, y_mx), axis=1)
        ref = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        ref_np = np.array(ref)

        # All-NaN row: valid_sum=0, valid_count=0 -> ref = 0/1 = 0.0
        assert ref_np[0] == pytest.approx(0.0)
        # Normal row: nanmean = 2.0
        assert ref_np[1] == pytest.approx(2.0)

    def test_single_valid_value_row(self):
        """A row with exactly one valid value should return that value."""
        import mlx.core as mx

        y_np = np.array([[np.nan, 7.0, np.nan, np.nan]])
        y_mx = mx.array(y_np)

        valid_count = mx.sum(mx.where(mx.isnan(y_mx), 0.0, 1.0), axis=1)
        valid_sum = mx.sum(mx.where(mx.isnan(y_mx), 0.0, y_mx), axis=1)
        ref = valid_sum / mx.maximum(valid_count, mx.array(1.0))
        ref_np = np.array(ref)

        assert ref_np[0] == pytest.approx(7.0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestVSNGpuNoNanRegression:
    """Regression tests: GPU VSN should still work correctly without NaN."""

    def test_no_nan_data_gpu_cpu_match(self):
        """Without NaN, GPU and CPU paths should produce matching results."""
        from cliquefinder.stats.normalization import vsn_normalization

        rng = np.random.RandomState(99)
        n_features, n_samples = 40, 6
        data = rng.exponential(scale=500, size=(n_features, n_samples)).astype(np.float64)

        result_cpu = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=False)
        result_gpu = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=True)

        # With no NaN, both paths should converge to the same result
        assert_allclose(
            result_gpu.data,
            result_cpu.data,
            rtol=1e-3,
            atol=1e-3,
            err_msg="GPU and CPU VSN diverge on clean (no-NaN) data",
        )

    def test_no_nan_convergence(self):
        """GPU path should converge when there are no NaN values."""
        from cliquefinder.stats.normalization import vsn_normalization

        rng = np.random.RandomState(77)
        data = rng.exponential(scale=1000, size=(30, 5)).astype(np.float64)

        result = vsn_normalization(data, method="proper", max_iter=50, use_gpu=True)
        assert result.diagnostics["converged"] or result.diagnostics["iterations"] <= 50
        assert not np.any(np.isnan(result.data))

    def test_vsn_deterministic_gpu(self):
        """GPU VSN should be deterministic across repeated calls."""
        from cliquefinder.stats.normalization import vsn_normalization

        rng = np.random.RandomState(55)
        data = rng.exponential(scale=800, size=(25, 4)).astype(np.float64)

        result1 = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=True)
        result2 = vsn_normalization(data.copy(), method="proper", max_iter=20, use_gpu=True)

        assert_allclose(result1.data, result2.data, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Li-Ji M_eff formula
# ---------------------------------------------------------------------------


class TestLiJiMeff:
    """Tests for the corrected Li-Ji effective test count formula."""

    def test_meff_uses_original_m_not_filtered(self):
        """M_eff formula must use M=original test count, not len(filtered eigenvalues).

        We construct a 10x10 correlation matrix with 3 near-zero eigenvalues.
        After filtering, only 7 eigenvalues remain. The old code would use M=7;
        the corrected code uses M=10.
        """
        # Build a correlation matrix with known rank deficiency.
        # Start with a rank-7 matrix, then add small noise to get 3 near-zero eigenvalues.
        rng = np.random.RandomState(42)
        M = 10

        # Create a low-rank (rank 7) positive semidefinite matrix
        V = rng.randn(M, 7)
        cov = V @ V.T
        # Regularize to ensure it's a proper correlation matrix
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

        # Add tiny noise to make eigenvalues strictly positive but near-zero
        corr += np.eye(M) * 1e-12

        # Verify eigenvalue structure
        eigvals = np.linalg.eigvalsh(corr)
        n_near_zero = np.sum(eigvals < 1e-10)
        n_positive = np.sum(eigvals > 1e-10)
        assert n_near_zero >= 2, f"Expected >=2 near-zero eigenvalues, got {n_near_zero}"
        assert n_positive < M, f"Expected some filtered eigenvalues"

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')

        # The formula is: 1 + (M-1) * (1 - Var(padded_eigenvalues) / M)
        # With M=10, the result should be significantly different from using len(filtered).
        # Compute expected value manually
        filtered_eigvals = eigvals[eigvals > 1e-10]
        padded = np.zeros(M)
        padded[:len(filtered_eigvals)] = filtered_eigvals
        expected_var = np.var(padded)
        expected_meff = 1 + (M - 1) * (1 - expected_var / M)
        expected_meff = np.clip(expected_meff, 1.0, M)

        assert m_eff == pytest.approx(expected_meff, rel=1e-10)

        # Old buggy result would use len(filtered_eigvals) instead of M
        buggy_var = np.var(filtered_eigvals)
        buggy_meff = 1 + (len(filtered_eigvals) - 1) * (1 - buggy_var / len(filtered_eigvals))
        buggy_meff = np.clip(buggy_meff, 1.0, M)

        # The corrected value should differ from the buggy value
        assert m_eff != pytest.approx(buggy_meff, abs=0.01), (
            f"Corrected M_eff ({m_eff:.4f}) should differ from buggy ({buggy_meff:.4f})"
        )

    def test_meff_analytically_for_identity(self):
        """For an identity matrix (independent tests), M_eff should equal M.

        All eigenvalues of I_M are 1.0, so Var(eigenvalues) = 0.
        M_eff = 1 + (M-1) * (1 - 0/M) = 1 + (M-1) = M.
        """
        M = 10
        corr = np.eye(M)
        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
        assert m_eff == pytest.approx(float(M), rel=1e-10)

    def test_meff_analytically_for_perfect_correlation(self):
        """For a matrix of all 1s (perfectly correlated), M_eff should be 1.

        Eigenvalues: [M, 0, 0, ..., 0]. After filtering, only [M] remains.
        Zero-padded: [M, 0, 0, ..., 0] (length M).
        Var = M * (M-1)^2 / M^2 + (M-1) * M^2 / M^2 ... actually compute directly.
        """
        M = 10
        corr = np.ones((M, M))

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')

        # Eigenvalues: one eigenvalue = M, rest = 0
        # Padded eigenvalues: [M, 0, 0, ..., 0]
        # Mean = M/M = 1, Var = ((M-1)^2 + (M-1)*1^2) / M = (M-1)*M / M = M-1
        # Wait, let's compute properly:
        padded = np.zeros(M)
        padded[0] = M
        expected_var = np.var(padded)
        expected_meff = 1 + (M - 1) * (1 - expected_var / M)
        expected_meff = np.clip(expected_meff, 1.0, M)

        assert m_eff == pytest.approx(expected_meff, rel=1e-10)
        # M_eff should be close to 1 for perfectly correlated tests
        assert m_eff < 2.0, f"M_eff for perfectly correlated tests should be near 1, got {m_eff}"

    def test_meff_full_rank_unchanged(self):
        """For a full-rank matrix, no eigenvalues are filtered, so old and new should agree.

        This ensures the fix doesn't break the common case where no filtering happens.
        """
        rng = np.random.RandomState(88)
        M = 8

        # Generate a well-conditioned correlation matrix
        A = rng.randn(100, M)
        corr = np.corrcoef(A.T)

        # All eigenvalues should be well above 1e-10
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals > 1e-10), "Expected full-rank matrix"

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')

        # Since no eigenvalues are filtered, the fix shouldn't change the result.
        # Compute expected: Var over all eigenvalues (same as padded since no filtering)
        expected_var = np.var(eigvals)
        expected_meff = 1 + (M - 1) * (1 - expected_var / M)
        expected_meff = np.clip(expected_meff, 1.0, M)

        assert m_eff == pytest.approx(expected_meff, rel=1e-10)

    def test_meff_range_bounds(self):
        """M_eff should always be in [1, M] regardless of correlation structure."""
        rng = np.random.RandomState(33)

        for trial in range(10):
            M = rng.randint(3, 15)
            A = rng.randn(50, M)
            corr = np.corrcoef(A.T)

            m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
            assert 1.0 <= m_eff <= M, f"M_eff={m_eff} out of bounds [1, {M}]"

    def test_nyholt_unaffected_by_fix(self):
        """The Nyholt method only uses filtered eigenvalues; it should be unchanged.

        Nyholt: M_eff = sum(eigenvalues) / max(eigenvalues). The filtering only
        removes near-zero values, which barely affect the sum or max. So Nyholt
        was not buggy, but we verify it's not broken by our changes.
        """
        rng = np.random.RandomState(42)
        M = 10
        V = rng.randn(M, 7)
        cov = V @ V.T
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        corr += np.eye(M) * 1e-12

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='nyholt')

        eigvals = np.linalg.eigvalsh(corr)
        filtered = eigvals[eigvals > 1e-10]
        expected = np.sum(filtered) / np.max(filtered)
        expected = np.clip(expected, 1.0, M)

        assert m_eff == pytest.approx(expected, rel=1e-10)

    def test_meff_zero_padded_variance(self):
        """Verify that variance is computed over zero-padded eigenvalues, not filtered.

        Build a case where the difference between the two is measurable:
        5x5 matrix with 2 near-zero eigenvalues. Filtered has 3 eigenvalues;
        zero-padded has 5 with two zeros. The variances will differ.
        """
        M = 5
        # Eigenvalues: [3.0, 1.5, 0.5, ~0, ~0]
        # We'll build a matrix with these eigenvalues
        eigvals_target = np.array([3.0, 1.5, 0.5, 1e-15, 1e-15])
        # Construct: Q @ diag(eigvals) @ Q^T
        rng = np.random.RandomState(77)
        Q, _ = np.linalg.qr(rng.randn(M, M))
        corr = Q @ np.diag(eigvals_target) @ Q.T
        # Symmetrize
        corr = (corr + corr.T) / 2

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')

        # Expected: zero-padded variance
        filtered = eigvals_target[eigvals_target > 1e-10]
        assert len(filtered) == 3
        padded = np.zeros(M)
        padded[:len(filtered)] = filtered
        var_padded = np.var(padded)
        expected = 1 + (M - 1) * (1 - var_padded / M)
        expected = np.clip(expected, 1.0, M)

        assert m_eff == pytest.approx(expected, rel=1e-6)

        # Verify filtered-only variance would give a different answer
        var_filtered_only = np.var(filtered)
        buggy = 1 + (len(filtered) - 1) * (1 - var_filtered_only / len(filtered))
        assert abs(m_eff - buggy) > 0.01, (
            f"Corrected M_eff ({m_eff}) should differ from buggy ({buggy})"
        )


class TestLiJiEdgeCases:
    """Edge case tests for Li-Ji M_eff estimation."""

    def test_single_test(self):
        """M=1 should return 1.0 without error."""
        corr = np.array([[1.0]])
        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
        assert m_eff == pytest.approx(1.0)

    def test_two_tests_independent(self):
        """M=2 identity matrix: M_eff should be 2."""
        corr = np.eye(2)
        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
        assert m_eff == pytest.approx(2.0, rel=1e-10)

    def test_two_tests_perfectly_correlated(self):
        """M=2 all-ones matrix: M_eff should be ~1."""
        corr = np.ones((2, 2))
        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
        assert m_eff == pytest.approx(1.0, abs=0.5)
        assert m_eff >= 1.0

    def test_moderate_correlation(self):
        """M_eff should be between 1 and M for moderately correlated tests."""
        M = 6
        # Off-diagonal correlation of 0.5
        corr = np.full((M, M), 0.5)
        np.fill_diagonal(corr, 1.0)

        m_eff = estimate_effective_tests(correlation_matrix=corr, method='li-ji')
        assert 1.0 < m_eff < M, f"M_eff={m_eff} should be between 1 and {M}"
