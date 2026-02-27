"""
Tests for STAT-1: Per-pattern OLS NaN handling in batched_ols_gpu and run_protein_differential.

Validates that:
1. Features with different NaN patterns get correct per-pattern coefficients
2. Features with no NaN match the original (full-data) computation
3. Features with all NaN get NaN results
4. GPU and CPU paths agree
5. Synthetic data demonstrates measurable bias from the old (shared-inverse) approach
6. The bare-except fix (S-3d) is correct
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from cliquefinder.stats.differential import (
    MLX_AVAILABLE,
    batched_ols_gpu,
    build_contrast_matrix,
    run_protein_differential,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_design_matrix(n_ctrl: int, n_case: int) -> np.ndarray:
    """Create a simple dummy-coded design matrix [intercept, treatment]."""
    n = n_ctrl + n_case
    X = np.zeros((n, 2), dtype=np.float64)
    X[:, 0] = 1.0  # intercept
    X[n_ctrl:, 1] = 1.0  # treatment indicator
    return X


@pytest.fixture
def simple_no_nan_data():
    """Dataset with no NaN values -- baseline correctness check."""
    rng = np.random.default_rng(42)
    n_ctrl, n_case = 10, 10
    n_samples = n_ctrl + n_case
    n_features = 50

    X = _make_design_matrix(n_ctrl, n_case)
    conditions = ["CTRL", "CASE"]
    sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)

    # True log2FC = 2.0 for features 0-9, 0.0 for features 10+
    Y = rng.normal(10.0, 1.0, (n_samples, n_features))
    for j in range(10):
        Y[n_ctrl:, j] += 2.0

    feature_ids = [f"P{i:03d}" for i in range(n_features)]

    return {
        "Y": Y,
        "X": X,
        "conditions": conditions,
        "feature_ids": feature_ids,
        "sample_condition": sample_condition,
        "n_ctrl": n_ctrl,
        "n_case": n_case,
    }


@pytest.fixture
def mixed_nan_data():
    """Dataset where different features have different NaN patterns.

    Feature 0: No NaN (full data)
    Feature 1: 3 NaN in CTRL only
    Feature 2: 3 NaN in CASE only
    Feature 3: NaN in both groups (different counts)
    Feature 4: All NaN (should produce NaN results)
    Feature 5: Only 2 valid samples (insufficient data)
    """
    rng = np.random.default_rng(123)
    n_ctrl, n_case = 8, 8
    n_samples = n_ctrl + n_case

    X = _make_design_matrix(n_ctrl, n_case)
    conditions = ["CTRL", "CASE"]
    sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)

    # True parameters: intercept=10, treatment effect=3.0
    true_intercept = 10.0
    true_effect = 3.0
    Y = np.full((n_samples, 6), true_intercept, dtype=np.float64)
    Y[n_ctrl:, :] += true_effect
    Y += rng.normal(0, 0.3, (n_samples, 6))

    # Introduce NaN patterns
    # Feature 1: remove 3 CTRL samples
    Y[:3, 1] = np.nan
    # Feature 2: remove 3 CASE samples
    Y[n_ctrl:n_ctrl + 3, 2] = np.nan
    # Feature 3: remove 2 CTRL + 2 CASE
    Y[:2, 3] = np.nan
    Y[n_ctrl:n_ctrl + 2, 3] = np.nan
    # Feature 4: all NaN
    Y[:, 4] = np.nan
    # Feature 5: only 2 valid (one per group -- insufficient for 2-param model)
    Y[:, 5] = np.nan
    Y[0, 5] = 10.0
    Y[n_ctrl, 5] = 13.0

    feature_ids = [f"P{i:03d}" for i in range(6)]

    return {
        "Y": Y,
        "X": X,
        "conditions": conditions,
        "feature_ids": feature_ids,
        "sample_condition": sample_condition,
        "n_ctrl": n_ctrl,
        "n_case": n_case,
        "true_intercept": true_intercept,
        "true_effect": true_effect,
    }


# ── Reference OLS implementation ────────────────────────────────────────


def _reference_ols(Y_col: np.ndarray, X: np.ndarray) -> dict | None:
    """Compute OLS for a single feature, dropping NaN rows.

    Returns dict with beta, residual_var, df, n_valid, XtX_inv, or None if rank-deficient.
    """
    valid = ~np.isnan(Y_col)
    n_valid = int(np.sum(valid))
    n_params = X.shape[1]

    if n_valid <= n_params:
        return None

    X_v = X[valid, :]
    y_v = Y_col[valid]

    beta, residuals, rank, sv = np.linalg.lstsq(X_v, y_v, rcond=None)

    y_pred = X_v @ beta
    rss = float(np.sum((y_v - y_pred) ** 2))
    df = n_valid - n_params
    res_var = rss / max(df, 1)

    XtX_inv = np.linalg.inv(X_v.T @ X_v)

    return {
        "beta": beta,
        "residual_var": res_var,
        "df": df,
        "n_valid": n_valid,
        "XtX_inv": XtX_inv,
    }


# ── Tests for GPU path (batched_ols_gpu) ────────────────────────────────


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
class TestBatchedOlsGpuNaN:
    """Tests for pattern-grouped OLS in the GPU path."""

    def test_no_nan_matches_reference(self, simple_no_nan_data):
        """When there are no NaN values, results match reference OLS exactly."""
        d = simple_no_nan_data
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        results = batched_ols_gpu(
            Y=d["Y"],
            X=d["X"],
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        for j, pr in enumerate(results):
            ref = _reference_ols(d["Y"][:, j], d["X"])
            assert ref is not None
            assert pr.convergence, f"Feature {j} should converge"
            assert len(pr.contrasts) == 1

            # Check log2FC matches reference
            # L transforms beta to condition means, then contrast computes difference
            L = np.zeros((2, 2))
            L[0, 0] = 1.0
            L[1, 0] = 1.0
            L[1, 1] = 1.0
            condition_means = L @ ref["beta"]
            c_vec = contrast_matrix[0]
            expected_fc = np.dot(c_vec, condition_means)

            # rtol=5e-4 accommodates MLX float32 intermediate precision
            # when the STAT-1-OPT fast path is active (complete data)
            np.testing.assert_allclose(
                pr.contrasts[0].log2_fc, expected_fc, rtol=5e-4, atol=1e-6,
                err_msg=f"Feature {j}: log2FC mismatch",
            )

    def test_per_pattern_nan_correctness(self, mixed_nan_data):
        """Features with different NaN patterns each get correct per-pattern OLS."""
        d = mixed_nan_data
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        results = batched_ols_gpu(
            Y=d["Y"],
            X=d["X"],
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        # Feature 0: full data -- should recover ~3.0 effect
        assert results[0].convergence
        np.testing.assert_allclose(
            results[0].contrasts[0].log2_fc, d["true_effect"], atol=0.5,
        )

        # Features 1-3: partial NaN -- should still recover effect with correct per-pattern OLS
        for idx in [1, 2, 3]:
            assert results[idx].convergence, f"Feature {idx} should converge"
            ref = _reference_ols(d["Y"][:, idx], d["X"])
            assert ref is not None

            # Check that per-pattern log2FC matches single-feature reference
            L = np.zeros((2, 2))
            L[0, 0] = 1.0
            L[1, 0] = 1.0
            L[1, 1] = 1.0
            cond_means = L @ ref["beta"]
            expected_fc = np.dot(contrast_matrix[0], cond_means)

            np.testing.assert_allclose(
                results[idx].contrasts[0].log2_fc,
                expected_fc,
                atol=1e-8,
                err_msg=f"Feature {idx}: log2FC does not match per-feature reference OLS",
            )

            # Also check residual variance
            np.testing.assert_allclose(
                results[idx].residual_variance,
                ref["residual_var"],
                atol=1e-8,
                err_msg=f"Feature {idx}: residual variance mismatch",
            )

    def test_all_nan_feature_not_converged(self, mixed_nan_data):
        """Feature with all NaN should not converge."""
        d = mixed_nan_data
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        results = batched_ols_gpu(
            Y=d["Y"],
            X=d["X"],
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        # Feature 4: all NaN
        assert not results[4].convergence
        assert len(results[4].contrasts) == 0

    def test_insufficient_data_feature(self, mixed_nan_data):
        """Feature with only 2 valid samples (= n_params) should not converge."""
        d = mixed_nan_data
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        results = batched_ols_gpu(
            Y=d["Y"],
            X=d["X"],
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        # Feature 5: only 2 valid samples with 2 parameters
        assert not results[5].convergence

    def test_df_is_per_feature(self, mixed_nan_data):
        """Each feature should have its own degrees of freedom based on valid samples."""
        d = mixed_nan_data
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        results = batched_ols_gpu(
            Y=d["Y"],
            X=d["X"],
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        # Feature 0: 16 samples - 2 params = 14 df
        assert results[0].contrasts[0].df == 14.0

        # Feature 1: 13 valid - 2 params = 11 df
        assert results[1].contrasts[0].df == 11.0

        # Feature 2: 13 valid - 2 params = 11 df
        assert results[2].contrasts[0].df == 11.0

        # Feature 3: 12 valid - 2 params = 10 df
        assert results[3].contrasts[0].df == 10.0


# ── Tests for CPU path (run_protein_differential) ───────────────────────


class TestRunProteinDifferentialNaN:
    """Tests for pattern-grouped OLS in the CPU path.

    NOTE: run_protein_differential builds its own design matrix via
    precompute_ols_matrices(), which may include regularization and uses
    its own dummy-coding. We validate correctness by comparing against
    a reference that uses the SAME design matrix, extracted from the
    precomputed matrices.
    """

    def _get_cpu_design_matrix(self, sample_condition, conditions, contrast):
        """Get the design matrix that run_protein_differential would use."""
        from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

        matrices = precompute_ols_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
            regularization=1e-8,
        )
        # Build valid mask to match what run_protein_differential does
        condition_cat = pd.Categorical(sample_condition, categories=conditions)
        valid_mask = ~pd.isna(condition_cat)
        return matrices, valid_mask

    def test_no_nan_matches_reference(self, simple_no_nan_data):
        """CPU path with no NaN should produce consistent coefficients."""
        d = simple_no_nan_data
        # data shape for CPU path: (n_features, n_samples)
        data = d["Y"].T

        results_df = run_protein_differential(
            data=data,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # Get the actual design matrix used internally
        matrices, valid_mask = self._get_cpu_design_matrix(
            d["sample_condition"], d["conditions"], ("CASE", "CTRL")
        )
        X = matrices.X

        # Check features against reference using the SAME design matrix
        for j in range(10):
            y = data[j, valid_mask]
            ref = _reference_ols(y, X)
            assert ref is not None

            # Expected log2FC = beta @ c
            expected_fc = ref["beta"] @ matrices.c

            row = results_df[results_df["feature_id"] == f"P{j:03d}"].iloc[0]
            np.testing.assert_allclose(
                row["log2fc"], expected_fc, atol=1e-6,
                err_msg=f"Feature {j}: log2FC mismatch in CPU path",
            )

    def test_per_pattern_nan_correctness_cpu(self, mixed_nan_data):
        """CPU path: features with different NaN patterns get correct per-pattern OLS."""
        d = mixed_nan_data
        data = d["Y"].T  # (n_features, n_samples)

        results_df = run_protein_differential(
            data=data,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # Get the design matrix
        matrices, valid_mask = self._get_cpu_design_matrix(
            d["sample_condition"], d["conditions"], ("CASE", "CTRL")
        )
        X = matrices.X

        # Check features 0-3 against reference using the same design matrix
        for idx in [0, 1, 2, 3]:
            y = data[idx, valid_mask]
            ref = _reference_ols(y, X)
            assert ref is not None, f"Feature {idx} should have a valid reference"

            expected_fc = ref["beta"] @ matrices.c
            row = results_df[results_df["feature_id"] == f"P{idx:03d}"].iloc[0]
            np.testing.assert_allclose(
                row["log2fc"], expected_fc, atol=1e-6,
                err_msg=f"Feature {idx}: log2FC mismatch in CPU path",
            )

            # Check variance
            np.testing.assert_allclose(
                row["sigma2"], ref["residual_var"], atol=1e-6,
                err_msg=f"Feature {idx}: sigma2 mismatch in CPU path",
            )

    def test_all_nan_feature_cpu(self, mixed_nan_data):
        """CPU path: feature with all NaN should have NaN p-value."""
        d = mixed_nan_data
        data = d["Y"].T

        results_df = run_protein_differential(
            data=data,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        row = results_df[results_df["feature_id"] == "P004"].iloc[0]
        assert np.isnan(row["p_value"]), "All-NaN feature should have NaN p-value"

    def test_per_feature_df_in_cpu_path(self, mixed_nan_data):
        """CPU path: per-feature df should reflect different sample counts."""
        d = mixed_nan_data
        data = d["Y"].T

        results_df = run_protein_differential(
            data=data,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # Get n_params from the internal design matrix
        matrices, valid_mask = self._get_cpu_design_matrix(
            d["sample_condition"], d["conditions"], ("CASE", "CTRL")
        )
        n_params = matrices.X.shape[1]

        # Feature 0: 16 valid - n_params df
        row0 = results_df[results_df["feature_id"] == "P000"].iloc[0]
        np.testing.assert_allclose(row0["df"], 16 - n_params, atol=0.5)

        # Feature 1: 13 valid - n_params df
        row1 = results_df[results_df["feature_id"] == "P001"].iloc[0]
        np.testing.assert_allclose(row1["df"], 13 - n_params, atol=0.5)

    def test_eb_moderation_with_per_feature_df(self):
        """EB moderation should work correctly with per-feature df."""
        rng = np.random.default_rng(456)
        n_ctrl, n_case = 10, 10
        n_features = 100

        sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)

        # Create heteroskedastic data so EB has something to shrink.
        # Different features have different noise levels (some low, some high).
        data = np.zeros((n_features, n_ctrl + n_case))
        for j in range(n_features):
            noise_sd = rng.uniform(0.1, 3.0)  # wide range of variances
            data[j, :] = rng.normal(10.0, noise_sd, n_ctrl + n_case)
        # Add effect to first 20 features
        data[:20, n_ctrl:] += 2.0
        # Introduce NaN in 30% of features, with different patterns
        for j in range(30, 60):
            nan_idx = rng.choice(n_ctrl + n_case, size=3, replace=False)
            data[j, nan_idx] = np.nan

        feature_ids = [f"P{i:03d}" for i in range(n_features)]

        results_df = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=sample_condition,
            contrast=("CASE", "CTRL"),
            eb_moderation=True,
            verbose=False,
        )

        # EB moderation should produce finite results for most features
        valid_results = results_df[results_df["p_value"].notna()]
        assert len(valid_results) > 80, "Most features should have valid results"

        # EB-moderated variance should be shrunk (not identical to sample variance)
        non_nan = valid_results["sigma2"].notna() & valid_results["sigma2_post"].notna()
        if non_nan.any():
            diff = np.abs(
                valid_results.loc[non_nan, "sigma2_post"].values
                - valid_results.loc[non_nan, "sigma2"].values
            )
            assert np.mean(diff > 1e-10) > 0.5, "EB should shrink most variances"


# ── GPU/CPU agreement test ──────────────────────────────────────────────


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
class TestGpuCpuAgreement:
    """GPU and CPU paths should produce statistically equivalent results.

    IMPORTANT: The GPU path (batched_ols_gpu) takes an explicit X matrix,
    while the CPU path (run_protein_differential) builds X internally via
    precompute_ols_matrices with regularization. To compare them, we must
    provide the same X matrix to the GPU path.
    """

    def _get_cpu_design_matrix(self, sample_condition, conditions, contrast):
        from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

        matrices = precompute_ols_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
            regularization=1e-8,
        )
        condition_cat = pd.Categorical(sample_condition, categories=conditions)
        valid_mask = ~pd.isna(condition_cat)
        return matrices, valid_mask

    def test_agreement_no_nan(self, simple_no_nan_data):
        """With no NaN, GPU and CPU paths should agree when using same X."""
        d = simple_no_nan_data

        # Get the design matrix that the CPU path uses
        matrices, valid_mask = self._get_cpu_design_matrix(
            d["sample_condition"], d["conditions"], ("CASE", "CTRL")
        )

        # Build contrast for GPU path
        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        # Use the same X matrix from precompute_ols_matrices
        gpu_results = batched_ols_gpu(
            Y=d["Y"][valid_mask, :],
            X=matrices.X,
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        cpu_results = run_protein_differential(
            data=d["Y"].T,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        for j, gpu_pr in enumerate(gpu_results):
            if not gpu_pr.convergence:
                continue
            fid = gpu_pr.feature_id
            cpu_row = cpu_results[cpu_results["feature_id"] == fid].iloc[0]

            # GPU uses L-transform + contrast vector on condition means,
            # CPU uses beta @ c directly. Both should give same result.
            np.testing.assert_allclose(
                gpu_pr.contrasts[0].log2_fc,
                cpu_row["log2fc"],
                atol=1e-4,
                err_msg=f"Feature {fid}: GPU/CPU log2FC disagree",
            )

    def test_agreement_with_nan(self, mixed_nan_data):
        """With mixed NaN patterns, GPU and CPU should produce equivalent results."""
        d = mixed_nan_data

        matrices, valid_mask = self._get_cpu_design_matrix(
            d["sample_condition"], d["conditions"], ("CASE", "CTRL")
        )

        contrast_matrix, contrast_names = build_contrast_matrix(
            d["conditions"], {"CASE_vs_CTRL": ("CASE", "CTRL")}
        )

        gpu_results = batched_ols_gpu(
            Y=d["Y"][valid_mask, :],
            X=matrices.X,
            conditions=d["conditions"],
            feature_ids=d["feature_ids"],
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
        )

        cpu_results = run_protein_differential(
            data=d["Y"].T,
            feature_ids=d["feature_ids"],
            sample_condition=d["sample_condition"],
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        for j, gpu_pr in enumerate(gpu_results):
            fid = gpu_pr.feature_id
            cpu_row = cpu_results[cpu_results["feature_id"] == fid].iloc[0]

            if not gpu_pr.convergence:
                # CPU should also flag as invalid
                assert np.isnan(cpu_row["p_value"]) or np.isnan(cpu_row["log2fc"]), (
                    f"Feature {fid}: GPU non-converged but CPU has valid result"
                )
                continue

            np.testing.assert_allclose(
                gpu_pr.contrasts[0].log2_fc,
                cpu_row["log2fc"],
                atol=1e-4,
                err_msg=f"Feature {fid}: GPU/CPU log2FC disagree with NaN",
            )

            np.testing.assert_allclose(
                gpu_pr.residual_variance,
                cpu_row["sigma2"],
                atol=1e-4,
                err_msg=f"Feature {fid}: GPU/CPU residual variance disagree",
            )


# ── Bias detection test ─────────────────────────────────────────────────


class TestSharedInverseBias:
    """Demonstrate that the old (shared-inverse) approach produces biased results.

    The old code did:
        Y_clean = np.where(nan_mask, 0.0, Y)
        beta = (X'X)^-1 X' Y_clean

    This adds phantom zero observations, biasing beta toward zero for features
    with NaN in one group.
    """

    def test_bias_from_asymmetric_nan(self):
        """When one group has many NaN, shared inverse biases log2FC toward zero.

        Setup: 10 CTRL + 10 CASE. Feature 0 has 5 CASE samples as NaN.
        The correct per-pattern OLS should still recover the true effect.
        """
        rng = np.random.default_rng(789)
        n_ctrl, n_case = 10, 10
        n_samples = n_ctrl + n_case

        sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)

        # Feature 0: partial NaN in CASE (5 out of 10 missing)
        # Feature 1: no NaN (control)
        data = np.full((2, n_samples), 10.0)
        data[:, n_ctrl:] += 5.0  # True effect = 5.0
        data += rng.normal(0, 0.5, (2, n_samples))

        # Make 5 CASE samples NaN for feature 0
        data[0, n_ctrl:n_ctrl + 5] = np.nan

        feature_ids = ["feat_partial_nan", "feat_no_nan"]

        results_df = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=sample_condition,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        row_partial = results_df[results_df["feature_id"] == "feat_partial_nan"].iloc[0]
        row_full = results_df[results_df["feature_id"] == "feat_no_nan"].iloc[0]

        # Both should recover the true effect (~5.0) within reasonable tolerance
        np.testing.assert_allclose(row_partial["log2fc"], 5.0, atol=1.5)
        np.testing.assert_allclose(row_full["log2fc"], 5.0, atol=1.5)

        # The partial-NaN feature should have fewer df
        assert row_partial["df"] < row_full["df"]

    def test_old_approach_would_bias(self):
        """Explicitly show the shared-inverse approach produces wrong beta.

        We compute beta both ways and show they differ for features with NaN.
        """
        rng = np.random.default_rng(101)
        n_ctrl, n_case = 8, 8
        n_samples = n_ctrl + n_case

        X = _make_design_matrix(n_ctrl, n_case)

        # True parameters: intercept=10, effect=4
        Y = np.full((n_samples, 1), 10.0)
        Y[n_ctrl:, 0] += 4.0
        Y += rng.normal(0, 0.3, (n_samples, 1))

        # Remove 4 CTRL samples
        Y[:4, 0] = np.nan

        # -- Old (incorrect) approach --
        nan_mask = np.isnan(Y[:, 0])
        Y_old = np.where(np.isnan(Y), 0.0, Y)
        XtX_inv_shared = np.linalg.inv(X.T @ X)
        beta_old = XtX_inv_shared @ (X.T @ Y_old[:, 0])

        # -- New (correct) approach --
        valid = ~nan_mask
        X_v = X[valid, :]
        Y_v = Y[valid, 0]
        XtX_inv_correct = np.linalg.inv(X_v.T @ X_v)
        beta_correct = XtX_inv_correct @ (X_v.T @ Y_v)

        # The old beta[0] (intercept) is biased because phantom zeros in CTRL
        # pull the CTRL mean down from ~10 toward ~(10*4 + 0*4)/8 = 5
        # The correct intercept should be ~10
        assert abs(beta_correct[0] - 10.0) < 1.0, "Correct intercept should be near 10"
        assert abs(beta_old[0] - 10.0) > 1.0, "Old intercept should be biased away from 10"

        # The old beta[1] (effect) is also biased
        assert abs(beta_correct[1] - 4.0) < 1.0, "Correct effect should be near 4"
        assert abs(beta_old[1] - 4.0) > 2.0, "Old effect estimate should be substantially biased"


# ── Edge case tests ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases in the pattern-grouped OLS."""

    def test_single_feature_no_nan(self):
        """Single feature with no NaN should work correctly."""
        rng = np.random.default_rng(42)
        n_ctrl, n_case = 5, 5
        sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)
        data = rng.normal(10.0, 1.0, (1, n_ctrl + n_case))
        data[0, n_ctrl:] += 2.0

        results_df = run_protein_differential(
            data=data,
            feature_ids=["single"],
            sample_condition=sample_condition,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        assert len(results_df) == 1
        assert results_df.iloc[0]["p_value"] < 0.05  # Should detect 2.0 effect

    def test_all_features_same_nan_pattern(self):
        """When all features share the same NaN pattern, all get same df."""
        rng = np.random.default_rng(42)
        n_ctrl, n_case = 8, 8
        n_features = 20
        sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)
        data = rng.normal(10.0, 1.0, (n_features, n_ctrl + n_case))
        data[:10, n_ctrl:] += 2.0

        # All features: same 2 NaN positions
        data[:, 0] = np.nan
        data[:, n_ctrl] = np.nan

        results_df = run_protein_differential(
            data=data,
            feature_ids=[f"P{i}" for i in range(n_features)],
            sample_condition=sample_condition,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        valid = results_df[results_df["p_value"].notna()]
        assert len(valid) == n_features

        # All should have same df (df values should be consistent)
        unique_df = valid["df"].unique()
        assert len(unique_df) == 1, f"Expected 1 unique df, got {len(unique_df)}: {unique_df}"

    def test_many_unique_patterns(self):
        """Stress test: many features with unique NaN patterns, validated against reference."""
        rng = np.random.default_rng(42)
        n_ctrl, n_case = 15, 15
        n_features = 50
        sample_condition = np.array(["CTRL"] * n_ctrl + ["CASE"] * n_case)
        data = rng.normal(10.0, 1.0, (n_features, n_ctrl + n_case))
        data[:20, n_ctrl:] += 2.0

        # Each feature gets a different random NaN pattern
        for j in range(n_features):
            n_nan = rng.integers(0, 5)
            if n_nan > 0:
                nan_idx = rng.choice(n_ctrl + n_case, size=n_nan, replace=False)
                data[j, nan_idx] = np.nan

        results_df = run_protein_differential(
            data=data,
            feature_ids=[f"P{i}" for i in range(n_features)],
            sample_condition=sample_condition,
            contrast=("CASE", "CTRL"),
            eb_moderation=False,
            verbose=False,
        )

        # Get the actual design matrix used internally
        from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

        matrices = precompute_ols_matrices(
            sample_condition=sample_condition,
            conditions=["CTRL", "CASE"],
            contrast=("CASE", "CTRL"),
            regularization=1e-8,
        )
        condition_cat = pd.Categorical(sample_condition, categories=["CTRL", "CASE"])
        valid_mask = ~pd.isna(condition_cat)
        X = matrices.X
        data_valid = data[:, valid_mask]

        # Verify against reference for each feature
        for j in range(n_features):
            y = data_valid[j, :]
            ref = _reference_ols(y, X)
            row = results_df[results_df["feature_id"] == f"P{j}"].iloc[0]

            if ref is None:
                assert np.isnan(row["p_value"]) or np.isnan(row["log2fc"])
                continue

            expected_fc = ref["beta"] @ matrices.c
            np.testing.assert_allclose(
                row["log2fc"], expected_fc, atol=1e-6,
                err_msg=f"Feature {j}: CPU path log2FC vs reference",
            )


# ── S-3d: Bare except fix test ──────────────────────────────────────────


class TestBareExceptFix:
    """Test that the bare except in differential.py has been replaced."""

    def test_no_bare_except_in_source_file(self):
        """The source file should not contain any bare 'except:' clauses."""
        import pathlib

        # Read the source file directly from the filesystem
        # This works regardless of which copy Python loaded
        src_file = pathlib.Path(__file__).parent.parent / "src" / "cliquefinder" / "stats" / "differential.py"
        source = src_file.read_text()
        lines = source.split("\n")

        bare_excepts = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "except:" or stripped.startswith("except: "):
                bare_excepts.append((i, line.rstrip()))

        assert len(bare_excepts) == 0, (
            f"Found bare except clause(s): {bare_excepts}"
        )

    def test_intercept_lookup_uses_typed_except(self):
        """The intercept lookup should catch KeyError and IndexError specifically."""
        import pathlib

        src_file = pathlib.Path(__file__).parent.parent / "src" / "cliquefinder" / "stats" / "differential.py"
        source = src_file.read_text()

        # Find the intercept lookup block
        assert "except (KeyError, IndexError)" in source, (
            "Expected 'except (KeyError, IndexError)' in differential.py"
        )
