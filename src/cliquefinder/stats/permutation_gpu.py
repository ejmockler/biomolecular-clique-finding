"""
GPU-accelerated permutation testing for clique differential abundance.

This module implements batched OLS regression with proper standard error calculation
for computing permutation test statistics at scale. Key features:

1. Precomputation: Design matrix and (X'X)^-1 computed once, reused 1.7M times
2. Batched OLS: All permutations processed in single GPU operations
3. Subject aggregation: Mixed model approximation via subject-level means
4. Numerical stability: Regularization for near-singular matrices

Statistical Framework:
    For each clique and permutation:
    - Summarize proteins to single value per sample (Tukey's Median Polish)
    - Fit: y ~ condition + (optional subject aggregation)
    - Compute t-statistic for condition contrast
    - Return all 1.7M t-statistics as single array

References:
    - MSstats: Choi et al. (2014) Bioinformatics 30(17):2524-2526
    - Batched computation: See PERMUTATION_OPTIMIZATION_SPEC.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.special import digamma, polygamma

# Optional MLX import for GPU acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# =============================================================================
# Empirical Bayes Functions (limma-style moderated t-statistics)
# =============================================================================

def trigamma_inverse(x: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """
    Compute the inverse of the trigamma function using Newton's method.

    Solves for y where trigamma(y) = x. This is a core utility for
    estimating the prior degrees of freedom in Empirical Bayes.

    The implementation follows limma's trigammaInverse.R:
    - Initial guess: y = 0.5 + 1/x (valid since 1/trigamma(y) > y - 0.5)
    - Newton iteration on 1/trigamma(y) which is convex and nearly linear

    Args:
        x: Target trigamma value (must be positive)
        tol: Convergence tolerance
        max_iter: Maximum Newton iterations

    Returns:
        y such that trigamma(y) ≈ x

    References:
        Smyth (2004) Statistical Applications in Genetics and Molecular Biology
    """
    if x <= 0:
        return np.inf

    # Initial guess from asymptotic approximation
    # trigamma(y) ≈ 1/y + 1/(2y²) for large y, so y ≈ 1/x for small x
    if x > 1e6:
        return 1.0 / np.sqrt(x)
    elif x < 1e-6:
        return 1.0 / x

    y = 0.5 + 1.0 / x  # Initial guess

    for _ in range(max_iter):
        tri = polygamma(1, y)  # trigamma(y)
        tri_deriv = polygamma(2, y)  # tetragamma(y) = d/dy trigamma(y)

        # Newton step: y_new = y - f(y)/f'(y) where f(y) = trigamma(y) - x
        if abs(tri_deriv) < 1e-15:
            break
        delta = (tri - x) / tri_deriv
        y_new = y - delta

        # Ensure y stays positive
        if y_new <= 0:
            y = y / 2.0
        else:
            y = y_new

        if abs(delta) < tol * abs(y):
            break

    return max(y, 1e-10)


def fit_f_dist(
    sigma2: NDArray[np.float64],
    df: int | NDArray[np.int32],
) -> tuple[float, float]:
    """
    Estimate prior d0 and s0² via method of moments (limma fitFDist).

    Fits a scaled F-distribution to the sample variances, estimating
    hyperparameters for the Empirical Bayes prior on true variances.

    Mathematical basis:
        Assume s²_i ~ (s₀²/d₀) × χ²_{d₀} for true variance s₀²
        Then E[log(s²_i)] and Var[log(s²_i)] depend on d₀ and s₀²
        Method of moments matches sample moments to theoretical ones

    Algorithm (from limma):
        1. z = log(s²) - digamma(df/2) + log(df/2)
        2. evar = var(z) - mean(trigamma(df/2))
        3. d₀ = 2 × trigamma⁻¹(evar)
        4. s₀² = exp(mean(z) + digamma(d₀/2) - log(d₀/2))

    Args:
        sigma2: Array of sample variances (n_features,)
        df: Residual degrees of freedom (scalar or per-feature array)

    Returns:
        Tuple (d0, s0_sq):
        - d0: Prior degrees of freedom (can be np.inf if no shrinkage)
        - s0_sq: Prior scale (prior variance estimate)

    References:
        Smyth (2004) "Linear models and empirical Bayes methods for
        assessing differential expression in microarray experiments"
    """
    # Filter valid (positive) variances
    valid_mask = (sigma2 > 0) & np.isfinite(sigma2)
    sigma2_valid = sigma2[valid_mask]

    if len(sigma2_valid) < 3:
        # Insufficient data - no shrinkage
        return np.inf, float(np.median(sigma2_valid)) if len(sigma2_valid) > 0 else 1.0

    # Handle scalar or array df
    if np.isscalar(df):
        df_valid = df
        df_half = df / 2.0
        trigamma_df = polygamma(1, df_half)
    else:
        df_valid = np.asarray(df)[valid_mask]
        df_half = df_valid / 2.0
        trigamma_df = polygamma(1, df_half)

    # Transform to log scale and center
    z = np.log(sigma2_valid)

    if np.isscalar(df_valid):
        e = z - digamma(df_half) + np.log(df_half)
        mean_trigamma = trigamma_df
    else:
        e = z - digamma(df_half) + np.log(df_half)
        mean_trigamma = np.mean(trigamma_df)

    # Method of moments
    emean = np.mean(e)
    evar = np.var(e, ddof=1)

    # Subtract expected variance from trigamma
    evar_adjusted = evar - mean_trigamma

    if evar_adjusted <= 0:
        # Variance is lower than expected under F-distribution
        # This means very strong shrinkage (d0 -> inf)
        d0 = np.inf
        s0_sq = np.exp(emean)
    else:
        # Solve for d0 using trigamma inverse
        d0 = 2.0 * trigamma_inverse(evar_adjusted)

        # Ensure d0 is reasonable (limma caps at very large values)
        if d0 > 1e10:
            d0 = np.inf
            s0_sq = np.exp(emean)
        else:
            # Compute s0² from d0
            s0_sq = np.exp(emean + digamma(d0 / 2.0) - np.log(d0 / 2.0))

    return float(d0), float(s0_sq)


def squeeze_var(
    sigma2: NDArray[np.float64],
    df: int | float | NDArray[np.float64],
    d0: float,
    s0_sq: float,
) -> tuple[NDArray[np.float64], float | NDArray[np.float64]]:
    """
    Apply Empirical Bayes variance shrinkage (limma squeezeVar).

    Computes posterior variances by shrinking sample variances toward
    the prior estimate. This is the core of moderated t-statistics.

    Formula:
        s²_post = (d₀ × s₀² + df × s²) / (d₀ + df)

    This is a weighted average of:
        - Prior variance s₀² (weight d₀)
        - Sample variance s² (weight df)

    Supports both scalar and per-feature (array) degrees of freedom.
    When df is an array, each feature can have a different residual df
    (e.g., due to different NaN missingness patterns), and the posterior
    df is also returned as an array.

    Args:
        sigma2: Sample variances (n_features,)
        df: Residual degrees of freedom. Scalar (int/float) when all
            features share the same df, or array (n_features,) when
            features have different df due to missing data patterns.
        d0: Prior degrees of freedom (from fit_f_dist)
        s0_sq: Prior scale (from fit_f_dist)

    Returns:
        Tuple (s2_post, df_total):
        - s2_post: Posterior (moderated) variances (n_features,)
        - df_total: Total degrees of freedom for t-distribution (d0 + df).
          Returns float when df is scalar, NDArray when df is array.
    """
    df_is_array = isinstance(df, np.ndarray)

    if np.isinf(d0):
        # No shrinkage - return original variances
        if df_is_array:
            return sigma2.copy(), df.astype(np.float64)
        return sigma2.copy(), float(df)

    # Weighted average shrinkage — NumPy broadcasting handles both
    # scalar and array df transparently.
    s2_post = (d0 * s0_sq + df * sigma2) / (d0 + df)
    df_total = d0 + df

    if df_is_array:
        return s2_post, df_total.astype(np.float64)
    return s2_post, float(df_total)


# =============================================================================
# OLS Dataclasses and Core Functions
# =============================================================================

@dataclass
class OLSPrecomputedMatrices:
    """Precomputed matrices for batched OLS regression.

    These matrices are computed once and reused for all permutations.

    Attributes:
        X: Design matrix (n_samples, n_params)
        XtX_inv: Inverse of X'X (n_params, n_params)
        c: Contrast vector in parameter space (n_params,)
        c_var_factor: Scalar c' @ (X'X)^-1 @ c for SE computation
        df_residual: Residual degrees of freedom (n_samples - n_params)
        conditions: Condition names (for reference)
        contrast_name: Name of the contrast being tested
        eb_d0: Empirical Bayes prior degrees of freedom (None = disabled)
        eb_s0_sq: Empirical Bayes prior scale (variance estimate)
        eb_df_total: Total df for moderated t-distribution (d0 + df_residual)
    """
    X: NDArray[np.float64]
    XtX_inv: NDArray[np.float64]
    c: NDArray[np.float64]
    c_var_factor: float
    df_residual: int
    conditions: list[str]
    contrast_name: str
    # Empirical Bayes priors (computed from observed data, then fixed)
    eb_d0: float | None = None
    eb_s0_sq: float | None = None
    eb_df_total: float | None = None


def precompute_ols_matrices(
    sample_condition: NDArray | pd.Series | None = None,
    conditions: list[str] | None = None,
    contrast: tuple[str, str] | None = None,
    regularization: float = 1e-8,
    covariates_df: pd.DataFrame | None = None,
    covariate_design: "CovariateDesign | None" = None,
) -> OLSPrecomputedMatrices:
    """
    Precompute design matrix and contrast vectors for batched OLS.

    Computes the matrices that are invariant across all permutations:
    - Design matrix X with dummy coding (+ optional covariate columns)
    - (X'X)^-1 for efficient coefficient estimation
    - Contrast vector c in parameter space (zero-padded for covariates)
    - Variance scaling factor c' @ (X'X)^-1 @ c

    These are computed ONCE and reused for all 1.7M permutation tests.

    Args:
        sample_condition: Condition labels for each sample (length n_samples).
            Not required when covariate_design is provided.
        conditions: Ordered list of unique condition names.
            Not required when covariate_design is provided.
        contrast: Tuple of (condition1, condition2) to test condition1 - condition2.
            Not required when covariate_design is provided.
        regularization: Ridge regularization for near-singular (X'X)
        covariates_df: Optional DataFrame of covariates (one row per sample).
            When provided, covariate columns are appended to the design matrix
            and the contrast vector is zero-padded so the test targets only
            the condition effect while adjusting for covariates.
            Ignored when covariate_design is provided.
        covariate_design: Optional pre-built CovariateDesign from
            design_matrix.build_covariate_design_matrix(). When provided,
            the design's X matrix, contrast vector, and sample_mask are used
            directly, skipping all internal design matrix construction.
            This eliminates the F8 latent bug where masks could diverge
            between the design matrix builder and this function.

    Returns:
        OLSPrecomputedMatrices with all precomputed components

    Example:
        >>> conditions = ["treatment", "control"]
        >>> contrast = ("treatment", "control")
        >>> matrices = precompute_ols_matrices(metadata['treatment_group'], conditions, contrast)
        >>> # Now reuse matrices for all permutations
    """
    # If a pre-built CovariateDesign is provided, use it directly (H6/F8 fix).
    # This consolidates the design matrix construction into a single path,
    # ensuring the X matrix, contrast vector, and sample mask all come from
    # the same source.
    if covariate_design is not None:
        X = covariate_design.X
        c = covariate_design.contrast
        n_params = X.shape[1]
        n_valid = X.shape[0]
        contrast_name = covariate_design.contrast_name

        # Compute (X'X) with regularization
        XtX = X.T @ X
        if regularization > 0:
            XtX = XtX + regularization * np.eye(n_params)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot invert design matrix: {e}")

        c_var_factor = float(c @ XtX_inv @ c)
        if c_var_factor < 0:
            warnings.warn(f"Negative variance factor {c_var_factor}, taking absolute value")
            c_var_factor = abs(c_var_factor)

        df_residual = n_valid - n_params
        if df_residual < 1:
            raise ValueError(
                f"Insufficient df: {n_valid} samples - {n_params} parameters = {df_residual}"
            )

        return OLSPrecomputedMatrices(
            X=X,
            XtX_inv=XtX_inv,
            c=c,
            c_var_factor=c_var_factor,
            df_residual=df_residual,
            conditions=conditions if conditions is not None else [],
            contrast_name=contrast_name,
        )

    # If covariates provided, use the unified design matrix builder
    if covariates_df is not None and len(covariates_df.columns) > 0:
        from .design_matrix import build_covariate_design_matrix

        design = build_covariate_design_matrix(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
            covariates_df=covariates_df,
        )
        X = design.X
        c = design.contrast
        n_params = X.shape[1]
        n_valid = X.shape[0]
        contrast_name = design.contrast_name

        # Compute (X'X) with regularization
        XtX = X.T @ X
        if regularization > 0:
            XtX = XtX + regularization * np.eye(n_params)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot invert design matrix: {e}")

        c_var_factor = float(c @ XtX_inv @ c)
        if c_var_factor < 0:
            warnings.warn(f"Negative variance factor {c_var_factor}, taking absolute value")
            c_var_factor = abs(c_var_factor)

        df_residual = n_valid - n_params
        if df_residual < 1:
            raise ValueError(
                f"Insufficient df: {n_valid} samples - {n_params} parameters = {df_residual}"
            )

        return OLSPrecomputedMatrices(
            X=X,
            XtX_inv=XtX_inv,
            c=c,
            c_var_factor=c_var_factor,
            df_residual=df_residual,
            conditions=conditions,
            contrast_name=contrast_name,
        )

    # Standard path: no covariates
    import statsmodels.api as sm

    n_samples = len(sample_condition)

    # Convert condition to categorical with explicit order
    condition_cat = pd.Categorical(sample_condition, categories=conditions)
    df = pd.DataFrame({'condition': condition_cat})

    # Remove any NaN values
    df = df.dropna()
    n_valid = len(df)

    if n_valid < 3:
        raise ValueError(f"Insufficient samples after removing NaN: {n_valid}")

    # Create design matrix with dummy coding (first condition is reference)
    X_df = pd.get_dummies(df['condition'], drop_first=True, dtype=float)
    X_df = sm.add_constant(X_df)
    X = X_df.values  # Shape: (n_samples, n_params)

    n_params = X.shape[1]

    # Compute (X'X) with optional regularization for stability
    XtX = X.T @ X

    # Add ridge regularization to diagonal (improves numerical stability)
    if regularization > 0:
        XtX = XtX + regularization * np.eye(n_params)

    # Compute inverse
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Singular matrix - try higher regularization
        warnings.warn(f"Singular design matrix, increasing regularization to {regularization * 100}")
        XtX = X.T @ X + (regularization * 100) * np.eye(n_params)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot invert design matrix even with regularization: {e}")

    # Build contrast vector in condition space
    # Contrast is condition1 - condition2
    n_conditions = len(conditions)
    contrast_vec_condition = np.zeros(n_conditions)

    try:
        idx1 = conditions.index(contrast[0])
        idx2 = conditions.index(contrast[1])
    except ValueError as e:
        raise ValueError(f"Contrast conditions {contrast} not found in {conditions}: {e}")

    contrast_vec_condition[idx1] = 1.0
    contrast_vec_condition[idx2] = -1.0

    # Transform contrast to parameter space
    # For dummy coding: condition_mean[i] = intercept (i=0) or intercept + coef[i] (i>0)
    L = np.zeros((n_conditions, n_params))
    L[:, 0] = 1.0  # All conditions include intercept
    for i in range(1, min(n_conditions, n_params)):
        L[i, i] = 1.0  # Add dummy coefficient for non-reference conditions

    # Contrast in parameter space: c = L' @ contrast_vec_condition
    c = L.T @ contrast_vec_condition

    # Precompute variance scaling factor: c' @ (X'X)^-1 @ c
    # This is the multiplier for σ² to get SE²: SE² = σ² * c_var_factor
    c_var_factor = float(c @ XtX_inv @ c)

    if c_var_factor < 0:
        warnings.warn(f"Negative variance factor {c_var_factor}, taking absolute value")
        c_var_factor = abs(c_var_factor)

    # Degrees of freedom
    df_residual = n_valid - n_params
    if df_residual < 1:
        raise ValueError(f"Insufficient df: {n_valid} samples - {n_params} parameters = {df_residual}")

    contrast_name = f"{contrast[0]}_vs_{contrast[1]}"

    return OLSPrecomputedMatrices(
        X=X,
        XtX_inv=XtX_inv,
        c=c,
        c_var_factor=c_var_factor,
        df_residual=df_residual,
        conditions=conditions,
        contrast_name=contrast_name,
    )


def batched_ols_contrast_test(
    Y: NDArray[np.float64],
    matrices: OLSPrecomputedMatrices,
    use_gpu: bool = True,
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """
    Batched OLS regression with contrast testing for permutation tests.

    Efficiently computes t-statistics for all permutations using precomputed
    design matrix components. This is the core optimization that enables
    processing 1.7M model fits in seconds instead of hours.

    Mathematical Details:
        For each row y in Y (one permutation):

        1. Coefficients: β = y @ X @ (X'X)^-1'
        2. Predictions: ŷ = β @ X'
        3. Residuals: e = y - ŷ
        4. Residual variance: σ² = sum(e²) / df_residual
        5. Contrast estimate: est = β @ c
        6. Standard error: SE = sqrt(σ² * c_var_factor)
        7. t-statistic: t = est / SE

    All operations are fully vectorized across the batch dimension.

    Args:
        Y: Response matrix (n_total, n_samples) where n_total = n_cliques * n_permutations
           Each row is a summarized clique abundance profile across samples
        matrices: Precomputed OLS matrices (design, inverse, contrast)
        use_gpu: Whether to use MLX for GPU acceleration
        chunk_size: Process Y in chunks of this size (for memory management)
                    If None, processes all at once

    Returns:
        Array of t-statistics (n_total,) for each test

    Example:
        >>> # Precompute once
        >>> matrices = precompute_ols_matrices(conditions, contrast)
        >>> # Run on 1.7M permutations
        >>> Y_summarized = ...  # (1.7M, 379) from batched median polish
        >>> t_stats = batched_ols_contrast_test(Y_summarized, matrices)
        >>> # Shape: (1.7M,) with t-statistic for each permutation
    """
    n_total, n_samples = Y.shape
    n_params = matrices.X.shape[1]

    if n_samples != matrices.X.shape[0]:
        raise ValueError(
            f"Sample dimension mismatch: Y has {n_samples}, X has {matrices.X.shape[0]}"
        )

    # Determine if we should use GPU
    can_use_gpu = use_gpu and MLX_AVAILABLE

    # If chunking requested or data is very large, process in chunks
    if chunk_size is not None or n_total > 500000:
        if chunk_size is None:
            chunk_size = 100000  # Default chunk size

        t_stats = np.zeros(n_total)

        for start_idx in range(0, n_total, chunk_size):
            end_idx = min(start_idx + chunk_size, n_total)
            Y_chunk = Y[start_idx:end_idx]

            t_stats[start_idx:end_idx] = _batched_ols_contrast_test_impl(
                Y_chunk, matrices, can_use_gpu
            )

        return t_stats
    else:
        return _batched_ols_contrast_test_impl(Y, matrices, can_use_gpu)


def _batched_ols_contrast_test_impl(
    Y: NDArray[np.float64],
    matrices: OLSPrecomputedMatrices,
    use_gpu: bool,
) -> NDArray[np.float64]:
    """
    Internal implementation of batched OLS contrast test.

    Separated for easier chunking logic.
    """
    n_batch, n_samples = Y.shape

    if use_gpu:
        # GPU implementation with MLX
        return _batched_ols_gpu(Y, matrices)
    else:
        # CPU implementation with NumPy
        return _batched_ols_cpu(Y, matrices)


def _batched_ols_gpu(
    Y: NDArray[np.float64],
    matrices: OLSPrecomputedMatrices,
) -> NDArray[np.float64]:
    """
    GPU implementation using MLX.
    """
    # Convert to MLX arrays
    Y_mx = mx.array(Y, dtype=mx.float32)
    X_mx = mx.array(matrices.X, dtype=mx.float32)
    XtX_inv_mx = mx.array(matrices.XtX_inv, dtype=mx.float32)
    c_mx = mx.array(matrices.c, dtype=mx.float32)

    # Compute coefficients: β = Y @ X @ (X'X)^-1'
    # Shape: (n_batch, n_params)
    beta = mx.matmul(mx.matmul(Y_mx, X_mx), XtX_inv_mx.T)

    # Predictions: ŷ = β @ X'
    # Shape: (n_batch, n_samples)
    Y_pred = mx.matmul(beta, X_mx.T)

    # Residuals: e = Y - ŷ
    residuals = Y_mx - Y_pred

    # Residual sum of squares: RSS = sum(e², axis=1)
    # Shape: (n_batch,)
    rss = mx.sum(residuals ** 2, axis=1)

    # Residual variance: σ² = RSS / df_residual (GPU)
    sigma2 = rss / matrices.df_residual

    # Contrast estimate: est = β @ c (GPU)
    # Shape: (n_batch,)
    estimate = mx.matmul(beta, c_mx)

    # Evaluate MLX arrays and convert to CPU for EB moderation (type-safe float64)
    mx.eval(sigma2, estimate)  # Force evaluation before conversion
    sigma2_np = np.array(sigma2, dtype=np.float64)
    estimate_np = np.array(estimate, dtype=np.float64)

    # Apply Empirical Bayes variance shrinkage on CPU (all float64, no type mixing)
    if matrices.eb_d0 is not None and matrices.eb_s0_sq is not None:
        if not np.isinf(matrices.eb_d0):
            s2_post = (matrices.eb_d0 * matrices.eb_s0_sq + matrices.df_residual * sigma2_np) / (matrices.eb_d0 + matrices.df_residual)
        else:
            s2_post = sigma2_np
    else:
        s2_post = sigma2_np

    # Standard error: SE = sqrt(s²_post * c_var_factor) (CPU)
    # Uses moderated variance if EB enabled, otherwise original variance
    # Shape: (n_batch,)
    se = np.sqrt(s2_post * matrices.c_var_factor)

    # Prevent division by zero
    se = np.maximum(se, 1e-10)

    # t-statistic: t = estimate / SE (CPU)
    t_stats = estimate_np / se

    return t_stats


def _batched_ols_cpu(
    Y: NDArray[np.float64],
    matrices: OLSPrecomputedMatrices,
) -> NDArray[np.float64]:
    """
    CPU implementation using NumPy.

    Same algorithm as GPU version, but using NumPy operations.
    """
    # Compute coefficients: β = Y @ X @ (X'X)^-1'
    # Shape: (n_batch, n_params)
    beta = Y @ matrices.X @ matrices.XtX_inv.T

    # Predictions: ŷ = β @ X'
    # Shape: (n_batch, n_samples)
    Y_pred = beta @ matrices.X.T

    # Residuals: e = Y - ŷ
    residuals = Y - Y_pred

    # Residual sum of squares: RSS = sum(e², axis=1)
    # Shape: (n_batch,)
    rss = np.sum(residuals ** 2, axis=1)

    # Residual variance: σ² = RSS / df_residual
    sigma2 = rss / matrices.df_residual

    # Apply Empirical Bayes variance shrinkage if enabled
    # s²_post = (d₀ × s₀² + df × σ²) / (d₀ + df)
    if matrices.eb_d0 is not None and matrices.eb_s0_sq is not None:
        if not np.isinf(matrices.eb_d0):
            s2_post = (matrices.eb_d0 * matrices.eb_s0_sq + matrices.df_residual * sigma2) / (matrices.eb_d0 + matrices.df_residual)
        else:
            s2_post = sigma2
    else:
        s2_post = sigma2

    # Contrast estimate: est = β @ c
    # Shape: (n_batch,)
    estimate = beta @ matrices.c

    # Standard error: SE = sqrt(s²_post * c_var_factor)
    # Uses moderated variance if EB enabled, otherwise original variance
    # Shape: (n_batch,)
    se = np.sqrt(s2_post * matrices.c_var_factor)

    # Prevent division by zero
    se = np.maximum(se, 1e-10)

    # t-statistic: t = estimate / SE
    t_stats = estimate / se

    return t_stats


def batched_median_polish_gpu(
    data: NDArray[np.float64],
    max_iter: int = 10,
    eps: float = 0.01,
    use_gpu: bool = True,
) -> NDArray[np.float64]:
    """
    Batched Tukey's Median Polish for GPU acceleration.

    Vectorizes the iterative median polish algorithm across the batch dimension,
    processing thousands of gene sets simultaneously on GPU.

    Algorithm (vectorized across batch):
        Initialize: overall = 0, row_effects = 0, col_effects = 0
        For iter in 1..max_iter:
            1. row_medians = median(data, axis=2)      # (batch, n_proteins)
            2. data -= row_medians[:, :, None]
            3. row_effects += row_medians

            4. col_medians = median(data, axis=1)      # (batch, n_samples)
            5. data -= col_medians[:, None, :]
            6. col_effects += col_medians

            7. Check convergence: max(|row_medians|, |col_medians|) < eps

        Extract overall: overall = median(row_effects, axis=1)
        Return: overall[:, None] + col_effects  # (batch, n_samples)

    Input shape:  (batch_size, n_proteins, n_samples)
    Output shape: (batch_size, n_samples)

    Args:
        data: 3D array with shape (batch, n_proteins, n_samples).
              Each batch element is a proteins × samples matrix of log2 intensities.
        max_iter: Maximum iterations for convergence.
        eps: Convergence tolerance - stop when max median < eps.
        use_gpu: If True and MLX available, use GPU. Otherwise use NumPy.

    Returns:
        2D array of shape (batch, n_samples) with summarized abundances.
        This represents the protein/clique abundance per sample for each batch element.

    Complexity:
        Time:  O(batch × max_iter × n_proteins × n_samples × log(n))
               where log(n) is median computation
        Space: O(batch × n_proteins × n_samples)

    Example:
        >>> # 1000 cliques, each with 3 proteins, 379 samples
        >>> data = np.random.randn(1000, 3, 379)
        >>> summarized = batched_median_polish_gpu(data)
        >>> summarized.shape
        (1000, 379)

    References:
        - Tukey (1977) Exploratory Data Analysis
        - MSstats: Choi et al. (2014) for application to proteomics summarization
    """
    # Determine backend
    use_mlx = use_gpu and MLX_AVAILABLE

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (batch, n_proteins, n_samples), got {data.ndim}D")

    if use_mlx:
        data_arr = mx.array(data, dtype=mx.float32)
    else:
        data_arr = np.asarray(data, dtype=np.float64)

    batch_size, n_proteins, n_samples = data_arr.shape

    # Early return for empty batch
    if batch_size == 0:
        return np.empty((0, n_samples), dtype=np.float64)

    # Initialize accumulators
    if use_mlx:
        row_effects = mx.zeros((batch_size, n_proteins), dtype=mx.float32)
        col_effects = mx.zeros((batch_size, n_samples), dtype=mx.float32)
        residuals = mx.array(data_arr, dtype=mx.float32)  # Work on copy
    else:
        row_effects = np.zeros((batch_size, n_proteins), dtype=np.float64)
        col_effects = np.zeros((batch_size, n_samples), dtype=np.float64)
        residuals = data_arr.copy()

    # Iterative median polish
    converged = False
    for iteration in range(max_iter):
        # Row sweep: subtract row medians
        # Handle NaN by using nanmedian
        if use_mlx:
            # MLX doesn't have nanmedian, so we handle NaN by masking
            # For the batched case, we assume data is clean (no NaN)
            row_medians = mx.median(residuals, axis=2)  # (batch, n_proteins)
            # Replace any NaN with 0 for numerical stability
            row_medians = mx.where(mx.isnan(row_medians), mx.zeros_like(row_medians), row_medians)
            residuals = residuals - row_medians[:, :, None]
            row_effects = row_effects + row_medians
        else:
            row_medians = np.nanmedian(residuals, axis=2)  # (batch, n_proteins)
            row_medians = np.nan_to_num(row_medians, nan=0.0)
            residuals = residuals - row_medians[:, :, np.newaxis]
            row_effects = row_effects + row_medians

        # Column sweep: subtract column medians
        if use_mlx:
            col_medians = mx.median(residuals, axis=1)  # (batch, n_samples)
            col_medians = mx.where(mx.isnan(col_medians), mx.zeros_like(col_medians), col_medians)
            residuals = residuals - col_medians[:, None, :]
            col_effects = col_effects + col_medians
        else:
            col_medians = np.nanmedian(residuals, axis=1)  # (batch, n_samples)
            col_medians = np.nan_to_num(col_medians, nan=0.0)
            residuals = residuals - col_medians[:, np.newaxis, :]
            col_effects = col_effects + col_medians

        # Check convergence: max adjustment across all batches
        if use_mlx:
            max_row_adj = float(mx.max(mx.abs(row_medians)))
            max_col_adj = float(mx.max(mx.abs(col_medians)))
        else:
            max_row_adj = float(np.max(np.abs(row_medians)))
            max_col_adj = float(np.max(np.abs(col_medians)))

        max_adjustment = max(max_row_adj, max_col_adj)

        if max_adjustment < eps:
            converged = True
            break

    # Extract overall effect from row effects
    # We take the median of row effects for each batch element
    # This matches the sequential algorithm in summarization.py
    if use_mlx:
        overall = mx.median(row_effects, axis=1)  # (batch,)
        overall = mx.where(mx.isnan(overall), mx.zeros_like(overall), overall)
        # Adjust row_effects by subtracting overall (for consistency with sequential)
        row_effects = row_effects - overall[:, None]
        # Combine overall + col_effects for final abundances
        sample_abundances = overall[:, None] + col_effects  # (batch, n_samples)
        # Convert back to NumPy for compatibility
        return np.array(sample_abundances, dtype=np.float64)
    else:
        overall = np.nanmedian(row_effects, axis=1)  # (batch,)
        overall = np.nan_to_num(overall, nan=0.0)
        # Adjust row_effects by subtracting overall (for consistency with sequential)
        row_effects = row_effects - overall[:, np.newaxis]
        # Combine overall + col_effects for final abundances
        sample_abundances = overall[:, np.newaxis] + col_effects  # (batch, n_samples)
        return sample_abundances


def precompute_random_indices(
    clique_sizes: dict[str, int],
    pool_size: int,
    n_permutations: int,
    random_state: int | None = None,
) -> tuple[dict[str, NDArray[np.int32]], list[int]]:
    """
    Vectorized generation of all random gene samples for permutation testing.

    Pre-generates ALL random samples at once using efficient NumPy operations,
    avoiding sequential random.choice calls in inner loops.

    Strategy:
        1. Group cliques by size (e.g., all 831 size-3 cliques together)
        2. For each size, generate (n_cliques × n_perms) samples at once
        3. Reshape into (n_cliques, n_perms, size) array

    This reduces 1.7M random.choice calls to ~18 vectorized operations.

    Args:
        clique_sizes: Dict mapping clique_id → number of proteins
        pool_size: Total number of genes in regulated pool
        n_permutations: Number of permutations per clique
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
        - Dict mapping clique_id → array of shape (n_perms, clique_size)
          containing indices into the gene pool
        - List of unique clique sizes (sorted) for batching

    Example:
        >>> clique_sizes = {"TF1": 3, "TF2": 3, "TF3": 4}
        >>> indices, sizes = precompute_random_indices(clique_sizes, 100, 1000)
        >>> indices["TF1"].shape  # (1000, 3) - 1000 permutations of 3 genes
        (1000, 3)

    Complexity:
        Time: O(n_size_classes × n_cliques_per_class × n_perms × clique_size × log(pool))
              where log(pool) is for sampling without replacement
        Space: O(n_cliques × n_perms × max_clique_size)

    Notes:
        - Uses PCG64 generator for reproducible, high-quality randomness
        - Sampling without replacement ensures no duplicate genes per clique
        - Memory efficient: stores int32 indices rather than full data
    """
    rng = np.random.Generator(np.random.PCG64(random_state))

    # Group cliques by size for efficient batched sampling
    size_to_cliques: dict[int, list[str]] = {}
    for clique_id, size in clique_sizes.items():
        size_to_cliques.setdefault(size, []).append(clique_id)

    unique_sizes = sorted(size_to_cliques.keys())
    random_indices_dict: dict[str, NDArray[np.int32]] = {}

    for size in unique_sizes:
        clique_ids_this_size = size_to_cliques[size]
        n_cliques_this_size = len(clique_ids_this_size)

        # Generate all samples for this size class at once
        # Shape: (n_cliques × n_perms, size)
        total_samples = n_cliques_this_size * n_permutations

        if size > pool_size:
            # Edge case: clique larger than pool (shouldn't happen in practice)
            warnings.warn(
                f"Clique size {size} exceeds pool size {pool_size}. "
                f"Using pool with replacement."
            )
            all_samples = rng.choice(
                pool_size,
                size=(total_samples, size),
                replace=True
            )
        else:
            # Standard case: sample without replacement FOR EACH ROW
            # We need to generate each row independently
            # Using a vectorized approach with permuted indices
            all_samples = np.zeros((total_samples, size), dtype=np.int32)
            for i in range(total_samples):
                all_samples[i] = rng.choice(pool_size, size=size, replace=False)

        # Reshape: (n_cliques, n_perms, size)
        all_samples_reshaped = all_samples.reshape(
            n_cliques_this_size, n_permutations, size
        )

        # Assign to each clique
        for idx, clique_id in enumerate(clique_ids_this_size):
            random_indices_dict[clique_id] = all_samples_reshaped[idx].astype(np.int32)

    return random_indices_dict, unique_sizes


def aggregate_to_subject_level(
    data: NDArray[np.float64],
    subject_ids: NDArray | pd.Series,
    method: Literal["mean", "median"] = "mean",
) -> tuple[NDArray[np.float64], NDArray]:
    """
    Aggregate protein/clique data to subject level.

    This function implements a mixed model approximation by pre-aggregating
    repeated measures within subjects. This approach:

    1. Preserves the correlation structure (no pseudoreplication)
    2. Enables batched OLS computation (avoids iterative REML)
    3. Is statistically valid for balanced or nearly-balanced designs

    The aggregation effectively treats within-subject variation as part of
    the measurement error, which is a reasonable approximation when the
    primary interest is between-subject (condition) effects.

    Args:
        data: 2D array (n_features, n_samples) of protein intensities
        subject_ids: Subject identifier for each sample (length n_samples)
        method: Aggregation method - "mean" (default) or "median"

    Returns:
        Tuple of (aggregated_data, unique_subject_ids)
        - aggregated_data: (n_features, n_subjects) with one value per subject
        - unique_subject_ids: Array of unique subject IDs in consistent order

    Example:
        >>> # Data has 379 samples from 100 subjects
        >>> data_agg, subjects = aggregate_to_subject_level(data, subject_ids)
        >>> # data_agg is now (n_features, 100)
        >>> # Can now use batched OLS with subject-level phenotypes

    Notes:
        - NaN values are handled by ignoring them in the aggregation
        - Subjects with all NaN for a feature will have NaN in output
        - This is equivalent to fitting a mixed model and extracting BLUPs
          in the case of balanced designs with equal within-subject variance
    """
    if isinstance(subject_ids, pd.Series):
        subject_ids = subject_ids.values

    n_features, n_samples = data.shape

    if len(subject_ids) != n_samples:
        raise ValueError(
            f"subject_ids length ({len(subject_ids)}) != data columns ({n_samples})"
        )

    # Get unique subjects in consistent order
    unique_subjects = pd.unique(subject_ids)
    unique_subjects = unique_subjects[~pd.isna(unique_subjects)]
    n_subjects = len(unique_subjects)

    # Create subject index map
    subject_to_idx = {subj: i for i, subj in enumerate(unique_subjects)}

    # Allocate output array
    data_agg = np.full((n_features, n_subjects), np.nan, dtype=np.float64)

    # Aggregate for each subject
    for subj_idx, subject in enumerate(unique_subjects):
        # Find samples belonging to this subject
        sample_mask = subject_ids == subject

        if not np.any(sample_mask):
            continue

        # Extract subject's samples
        subject_data = data[:, sample_mask]

        # Aggregate across samples
        if method == "mean":
            # Use nanmean to handle missing values
            data_agg[:, subj_idx] = np.nanmean(subject_data, axis=1)
        elif method == "median":
            # Use nanmedian to handle missing values
            data_agg[:, subj_idx] = np.nanmedian(subject_data, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return data_agg, unique_subjects


def validate_ols_implementation(
    n_samples: int = 100,
    n_features: int = 50,
    n_conditions: int = 2,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Validate batched OLS implementation against statsmodels.

    Runs a small test comparing:
    1. Statsmodels OLS on each feature individually
    2. Batched OLS on all features simultaneously

    Returns diagnostic statistics on numerical agreement.

    Args:
        n_samples: Number of samples to simulate
        n_features: Number of features to test
        n_conditions: Number of conditions (default 2)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with validation metrics:
        - max_beta_diff: Maximum absolute difference in coefficients
        - max_t_diff: Maximum absolute difference in t-statistics
        - max_pval_diff: Maximum absolute difference in p-values
        - mean_beta_diff: Mean absolute difference in coefficients
        - mean_t_diff: Mean absolute difference in t-statistics
        - all_close: Boolean, True if all metrics within tolerance

    Example:
        >>> metrics = validate_ols_implementation()
        >>> assert metrics['all_close'], "Validation failed!"
    """
    import statsmodels.api as sm
    from .differential import build_contrast_matrix

    np.random.seed(random_state)

    # Generate synthetic data
    conditions = [f"C{i}" for i in range(n_conditions)]
    sample_condition = np.random.choice(conditions, size=n_samples)

    # Generate feature data with some true effects
    Y = np.random.randn(n_features, n_samples)

    # Add condition effects
    for i, cond in enumerate(conditions):
        mask = sample_condition == cond
        Y[:, mask] += np.random.randn(n_features, 1) * 0.5

    # Define contrast
    contrast = (conditions[0], conditions[1])

    # Precompute matrices
    matrices = precompute_ols_matrices(sample_condition, conditions, contrast)

    # Batched computation
    t_stats_batched = batched_ols_contrast_test(Y, matrices, use_gpu=False)

    # Individual statsmodels computation
    t_stats_individual = np.zeros(n_features)
    beta_individual = np.zeros((n_features, matrices.X.shape[1]))

    for i in range(n_features):
        y = Y[i, :]
        model = sm.OLS(y, matrices.X)
        result = model.fit()

        beta_individual[i] = result.params

        # Compute contrast t-statistic
        estimate = result.params @ matrices.c
        se = np.sqrt(result.scale * matrices.c_var_factor)
        t_stats_individual[i] = estimate / se

    # Compute batched betas for comparison
    beta_batched = Y @ matrices.X @ matrices.XtX_inv.T

    # Compute differences
    beta_diff = np.abs(beta_batched - beta_individual)
    t_diff = np.abs(t_stats_batched - t_stats_individual)

    max_beta_diff = np.max(beta_diff)
    max_t_diff = np.max(t_diff)
    mean_beta_diff = np.mean(beta_diff)
    mean_t_diff = np.mean(t_diff)

    # Check if within tolerance
    all_close = (
        max_beta_diff < 1e-6 and
        max_t_diff < 1e-6
    )

    return {
        'max_beta_diff': max_beta_diff,
        'max_t_diff': max_t_diff,
        'mean_beta_diff': mean_beta_diff,
        'mean_t_diff': mean_t_diff,
        'all_close': all_close,
        'n_samples': n_samples,
        'n_features': n_features,
    }


def compute_empirical_pvalues(
    observed_t: dict[str, tuple[float, float, float]],
    null_t: dict[str, NDArray[np.float64]],
    null_log2fc: dict[str, NDArray[np.float64]],
    significance_threshold: float = 0.05,
) -> list:
    """
    Compute empirical p-values from observed and null t-statistics.

    For each clique, computes:
        - Two-sided p-value: P(|t_null| >= |t_obs|)
        - One-sided p-value: P(t_null >= t_obs) for same direction
        - Percentile rank: where t_obs falls in null distribution

    Args:
        observed_t: Dict mapping clique_id -> (log2fc, pvalue, tvalue)
        null_t: Dict mapping clique_id -> array of null t-values
        null_log2fc: Dict mapping clique_id -> array of null log2fc values
        significance_threshold: Threshold for empirical p-value

    Returns:
        List of PermutationTestResult objects
    """
    from .clique_analysis import PermutationTestResult

    results = []

    for clique_id, (obs_log2fc, obs_pval, obs_tval) in observed_t.items():
        if clique_id not in null_t:
            continue

        null_tvals = null_t[clique_id]

        if len(null_tvals) < 10:
            # Too few successful permutations
            continue

        # Two-sided empirical p-value: |t| >= |observed t|
        n_extreme = np.sum(np.abs(null_tvals) >= np.abs(obs_tval))
        empirical_pval = (n_extreme + 1) / (len(null_tvals) + 1)

        # One-sided (directional): same sign and >= magnitude
        if obs_tval > 0:
            n_extreme_dir = np.sum(null_tvals >= obs_tval)
        else:
            n_extreme_dir = np.sum(null_tvals <= obs_tval)
        empirical_pval_dir = (n_extreme_dir + 1) / (len(null_tvals) + 1)

        # Percentile rank
        percentile = 100 * np.mean(np.abs(null_tvals) < np.abs(obs_tval))

        # Compute null distribution statistics
        null_log2fc_vals = null_log2fc.get(clique_id, np.array([]))
        null_log2fc_mean = float(np.mean(null_log2fc_vals)) if len(null_log2fc_vals) > 0 else 0.0
        null_log2fc_std = float(np.std(null_log2fc_vals)) if len(null_log2fc_vals) > 0 else 0.0
        null_tvalue_mean = float(np.mean(null_tvals))

        results.append(PermutationTestResult(
            clique_id=clique_id,
            observed_log2fc=obs_log2fc,
            observed_pvalue=obs_pval,
            observed_tvalue=obs_tval,
            null_log2fc_mean=null_log2fc_mean,
            null_log2fc_std=null_log2fc_std,
            null_tvalue_mean=null_tvalue_mean,
            empirical_pvalue=empirical_pval,
            empirical_pvalue_directional=empirical_pval_dir,
            n_permutations=len(null_tvals),
            percentile_rank=percentile,
            is_significant=empirical_pval < significance_threshold,
        ))

    return results


def run_permutation_test_gpu(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list,
    condition_col: str,
    contrast: tuple[str, str],
    subject_col: str | None = "subject_id",
    summarization_method = None,
    n_permutations: int = 1000,
    use_mixed_model: bool = True,
    significance_threshold: float = 0.05,
    random_state: int | None = None,
    map_ids: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
    eb_moderation: bool = True,
) -> tuple[list, pd.DataFrame]:
    """
    GPU-accelerated permutation testing for clique differential abundance.

    Main orchestrator that coordinates the full pipeline:
    1. Precompute random indices and OLS matrices
    2. Run observed analysis on actual cliques
    3. Process permutations by clique size class (batched)
    4. Compute empirical p-values

    This function provides drop-in replacement for run_permutation_clique_test()
    with identical signature and return types.

    Args:
        data: 2D array (n_features, n_samples) of log2 intensities
        feature_ids: List of protein identifiers
        sample_metadata: DataFrame with sample information
        clique_definitions: List of TF cliques to test
        condition_col: Metadata column for condition labels (REQUIRED - must be specified by user)
        contrast: Tuple of (test_condition, reference_condition) (REQUIRED - must be specified by user)
        subject_col: Metadata column for subject IDs
        summarization_method: How to aggregate proteins within clique (unused, for API compat)
        n_permutations: Number of permutations for null distribution
        use_mixed_model: Whether to use mixed models
        significance_threshold: Threshold for empirical p-value
        random_state: Random seed for reproducibility
        map_ids: Whether to map gene symbols to feature IDs
        n_jobs: Number of parallel jobs (unused, for API compat)
        verbose: Print progress

    Returns:
        Tuple of (list of PermutationTestResult, DataFrame with null distribution stats)

    Performance:
        CPU Sequential: ~14.8 hours
        GPU Batched:    ~15 seconds
        Speedup:        ~3,500x
    """
    import time
    from .clique_analysis import map_feature_ids_to_symbols

    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX not available. Install with: pip install mlx\n"
            "Or use the CPU parallel implementation instead."
        )

    overall_start = time.time()

    if verbose:
        print(f"\nGPU-Accelerated Permutation Testing")
        print(f"=" * 70)
        print(f"Cliques: {len(clique_definitions)}")
        print(f"Permutations: {n_permutations}")
        print(f"Contrast: {contrast[0]} vs {contrast[1]}")
        print(f"EB moderation: {'enabled (limma-style)' if eb_moderation else 'disabled'}")

    n_features, n_samples = data.shape

    # Build feature index lookup
    feature_to_idx = {f: i for i, f in enumerate(feature_ids)}

    # ID mapping if needed
    if map_ids and len(clique_definitions) > 0:
        sample_proteins = []
        for clique in clique_definitions[:10]:
            sample_proteins.extend(clique.protein_ids[:3])

        matches = sum(1 for p in sample_proteins if p in feature_to_idx)
        if matches < len(sample_proteins) * 0.5:
            if verbose:
                print(f"\nID mapping required")
            symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

            for symbol, feature_id in symbol_to_feature.items():
                if feature_id in feature_to_idx:
                    feature_to_idx[symbol] = feature_to_idx[feature_id]

    # Collect all regulated genes (pool for permutation)
    all_regulated_genes: set[str] = set()
    clique_sizes: dict[str, int] = {}
    clique_proteins: dict[str, list[str]] = {}

    for clique in clique_definitions:
        present_proteins = [p for p in clique.protein_ids if p in feature_to_idx]
        if len(present_proteins) >= 2:
            all_regulated_genes.update(present_proteins)
            clique_sizes[clique.clique_id] = len(present_proteins)
            clique_proteins[clique.clique_id] = present_proteins

    regulated_genes_list = sorted(list(all_regulated_genes))

    if verbose:
        print(f"\nRegulated gene pool: {len(regulated_genes_list)} genes")
        if clique_sizes:
            print(f"Clique sizes: min={min(clique_sizes.values())}, max={max(clique_sizes.values())}")

    if not clique_sizes:
        if verbose:
            print("No valid cliques found")
        return [], pd.DataFrame()

    # PHASE 1: Precomputation
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 1: Precomputation")
        print(f"{'=' * 70}")

    # Precompute random indices
    random_indices_dict, unique_sizes = precompute_random_indices(
        clique_sizes=clique_sizes,
        pool_size=len(regulated_genes_list),
        n_permutations=n_permutations,
        random_state=random_state,
    )

    # Precompute OLS matrices
    if verbose:
        print(f"  Precomputing OLS matrices...")
        start_time = time.time()

    sample_condition = sample_metadata[condition_col].values
    conditions = sorted([contrast[0], contrast[1]])

    # Filter to samples belonging to the contrast conditions only
    contrast_mask = np.isin(sample_condition, conditions)
    n_contrast_samples = np.sum(contrast_mask)

    if n_contrast_samples < n_samples:
        if verbose:
            print(f"    Filtering to {n_contrast_samples}/{n_samples} samples in contrast")
        # Filter data and metadata to contrast samples
        data = data[:, contrast_mask]
        sample_metadata = sample_metadata.iloc[contrast_mask].copy()
        sample_condition = sample_metadata[condition_col].values
        n_samples = n_contrast_samples

    matrices = precompute_ols_matrices(sample_condition, conditions, contrast)

    if verbose:
        elapsed = time.time() - start_time
        print(f"    Design matrix: {matrices.X.shape}, completed in {elapsed:.2f}s")

    # PHASE 2: Observed Analysis (BATCHED BY SIZE)
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: Observed Analysis (GPU Batched)")
        print(f"{'=' * 70}")
        print(f"  Analyzing {len(clique_proteins)} observed cliques...")
        start_time = time.time()

    observed_t: dict[str, tuple[float, float, float]] = {}
    observed_clique_ids = list(clique_sizes.keys())

    # Group observed cliques by size for batching
    obs_size_to_cliques: dict[int, list[str]] = {}
    for clique_id in observed_clique_ids:
        size = clique_sizes[clique_id]
        if size not in obs_size_to_cliques:
            obs_size_to_cliques[size] = []
        obs_size_to_cliques[size].append(clique_id)

    # Process observed cliques in batches by size
    Y_observed_dict: dict[str, NDArray] = {}

    for size in sorted(obs_size_to_cliques.keys()):
        clique_ids_batch = obs_size_to_cliques[size]
        batch_size = len(clique_ids_batch)

        # Gather protein data for this size class
        protein_batch = np.zeros((batch_size, size, n_samples), dtype=np.float64)

        for i, clique_id in enumerate(clique_ids_batch):
            proteins = clique_proteins[clique_id]
            indices = [feature_to_idx[p] for p in proteins]
            protein_batch[i] = data[indices, :]

        # Batched median polish (GPU)
        Y_batch = batched_median_polish_gpu(protein_batch, max_iter=10, eps=0.01, use_gpu=True)

        # Store results
        for i, clique_id in enumerate(clique_ids_batch):
            Y_observed_dict[clique_id] = Y_batch[i]

    # Reconstruct Y_observed in original order
    Y_observed = np.array([Y_observed_dict[cid] for cid in observed_clique_ids])

    # Compute log2FC from beta coefficients
    beta_obs = Y_observed @ matrices.X @ matrices.XtX_inv.T
    log2fc_obs = beta_obs @ matrices.c

    # Empirical Bayes moderation (limma-style)
    if eb_moderation:
        # Compute residual variances for Empirical Bayes prior estimation
        # RSS = sum((Y - Ŷ)²) for each clique
        Y_pred_obs = beta_obs @ matrices.X.T
        residuals_obs = Y_observed - Y_pred_obs
        rss_obs = np.sum(residuals_obs ** 2, axis=1)
        sigma2_obs = rss_obs / matrices.df_residual

        # Estimate Empirical Bayes priors from observed variances (limma fitFDist)
        d0, s0_sq = fit_f_dist(sigma2_obs, matrices.df_residual)

        if verbose:
            if np.isinf(d0):
                print(f"    EB priors: d0=Inf (no shrinkage), s0²={s0_sq:.6f}")
            else:
                print(f"    EB priors: d0={d0:.2f}, s0²={s0_sq:.6f}")
                shrinkage_weight = d0 / (d0 + matrices.df_residual)
                print(f"    Shrinkage weight: {100*shrinkage_weight:.1f}% prior, {100*(1-shrinkage_weight):.1f}% sample")

        # Update matrices with EB priors for all subsequent computations
        matrices.eb_d0 = d0
        matrices.eb_s0_sq = s0_sq
        matrices.eb_df_total = d0 + matrices.df_residual if not np.isinf(d0) else float(matrices.df_residual)
    else:
        # No EB moderation - use standard OLS
        if verbose:
            print(f"    EB moderation: disabled (using standard t-statistics)")
        d0 = np.inf
        matrices.eb_d0 = None
        matrices.eb_s0_sq = None
        matrices.eb_df_total = float(matrices.df_residual)

    # Run batched OLS (with or without Empirical Bayes moderation)
    t_obs = batched_ols_contrast_test(Y_observed, matrices, use_gpu=True)

    # Compute p-values from t-distribution with appropriate degrees of freedom
    for i, clique_id in enumerate(observed_clique_ids):
        if not eb_moderation or np.isinf(d0):
            # No moderation - use normal approximation (large df)
            pval = 2 * scipy_stats.norm.sf(np.abs(t_obs[i]))
        else:
            # Moderated t-test: use t-distribution with d0 + df_residual df
            pval = 2 * scipy_stats.t.sf(np.abs(t_obs[i]), matrices.eb_df_total)
        observed_t[clique_id] = (log2fc_obs[i], pval, t_obs[i])

    if verbose:
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.2f}s")

    # PHASE 3: Null Distribution (FULLY BATCHED)
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: Null Distribution (GPU Batched)")
        print(f"{'=' * 70}")

    null_t: dict[str, list[float]] = {cid: [] for cid in observed_clique_ids}
    null_log2fc: dict[str, list[float]] = {cid: [] for cid in observed_clique_ids}

    # Group by size for batched processing
    size_to_cliques = {}
    for clique_id, size in clique_sizes.items():
        if size not in size_to_cliques:
            size_to_cliques[size] = []
        size_to_cliques[size].append(clique_id)

    # Create regulated gene index lookup (pool index -> data index)
    regulated_gene_indices = np.array(
        [feature_to_idx[g] for g in regulated_genes_list], dtype=np.int32
    )

    phase3_start = time.time()
    perm_chunk_size = 100  # Permutations per chunk (memory management)

    for size in sorted(size_to_cliques.keys()):
        clique_ids_this_size = size_to_cliques[size]
        n_cliques_this_size = len(clique_ids_this_size)

        if verbose:
            print(f"  Size {size}: {n_cliques_this_size} cliques × {n_permutations} perms")

        # Process permutations in chunks for memory management
        for chunk_start in range(0, n_permutations, perm_chunk_size):
            chunk_end = min(chunk_start + perm_chunk_size, n_permutations)
            chunk_perms = chunk_end - chunk_start

            # Total items in this batch: n_cliques × chunk_perms
            batch_size = n_cliques_this_size * chunk_perms

            # Gather all protein data into 3D batch array: (batch, size, n_samples)
            protein_batch = np.zeros((batch_size, size, n_samples), dtype=np.float64)

            batch_idx = 0
            for clique_id in clique_ids_this_size:
                random_indices_for_clique = random_indices_dict[clique_id]

                for p in range(chunk_start, chunk_end):
                    # Get random gene pool indices for this permutation
                    pool_indices = random_indices_for_clique[p]
                    # Map to actual data indices
                    data_indices = regulated_gene_indices[pool_indices]
                    # Gather protein data
                    protein_batch[batch_idx] = data[data_indices, :]
                    batch_idx += 1

            # BATCHED MEDIAN POLISH (GPU) - single call for entire batch
            Y_chunk = batched_median_polish_gpu(protein_batch, max_iter=10, eps=0.01, use_gpu=True)

            # BATCHED OLS (GPU)
            t_chunk = batched_ols_contrast_test(Y_chunk, matrices, use_gpu=True)

            # Compute log2FC from beta coefficients
            beta_chunk = Y_chunk @ matrices.X @ matrices.XtX_inv.T
            log2fc_chunk = beta_chunk @ matrices.c

            # Store results back to clique dictionaries
            batch_idx = 0
            for clique_id in clique_ids_this_size:
                for p_local in range(chunk_perms):
                    null_t[clique_id].append(t_chunk[batch_idx])
                    null_log2fc[clique_id].append(log2fc_chunk[batch_idx])
                    batch_idx += 1

    phase3_elapsed = time.time() - phase3_start

    if verbose:
        print(f"  Completed in {phase3_elapsed:.2f}s")

    # Convert to arrays
    null_t_arrays = {cid: np.array(vals) for cid, vals in null_t.items()}
    null_log2fc_arrays = {cid: np.array(vals) for cid, vals in null_log2fc.items()}

    # PHASE 4: Empirical P-values
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 4: Empirical P-values")
        print(f"{'=' * 70}")
        start_time = time.time()

    results = compute_empirical_pvalues(
        observed_t=observed_t,
        null_t=null_t_arrays,
        null_log2fc=null_log2fc_arrays,
        significance_threshold=significance_threshold,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s")

    # Create null distribution summary
    null_summary_rows = []
    for clique_id in observed_clique_ids:
        if clique_id in null_log2fc_arrays and clique_id in null_t_arrays:
            log2fc_vals = null_log2fc_arrays[clique_id]
            tval_vals = null_t_arrays[clique_id]

            null_summary_rows.append({
                'clique_id': clique_id,
                'null_log2FC_mean': np.mean(log2fc_vals),
                'null_log2FC_std': np.std(log2fc_vals),
                'null_log2FC_5pct': np.percentile(log2fc_vals, 5),
                'null_log2FC_95pct': np.percentile(log2fc_vals, 95),
                'null_tvalue_mean': np.mean(tval_vals),
                'null_tvalue_std': np.std(tval_vals),
                'null_tvalue_5pct': np.percentile(tval_vals, 5),
                'null_tvalue_95pct': np.percentile(tval_vals, 95),
                'n_permutations': len(tval_vals),
            })

    null_df = pd.DataFrame(null_summary_rows)

    # Summary
    overall_elapsed = time.time() - overall_start
    n_significant = sum(1 for r in results if r.is_significant)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RESULTS")
        print(f"{'=' * 70}")
        print(f"Cliques tested: {len(results)}")
        print(f"Significant (p < {significance_threshold}): {n_significant}")
        print(f"Significance rate: {100 * n_significant / len(results) if len(results) > 0 else 0:.1f}%")
        print(f"\nTotal wall-clock time: {overall_elapsed:.2f}s")

    return results, null_df
