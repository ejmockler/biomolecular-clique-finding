"""
Normalization methods for proteomics data.

Implements MSstats-style normalization approaches:
- Median normalization (equalizeMedians): Centers samples to common median
- Quantile normalization: Forces identical distributions across samples
- Global standards: Normalize to spike-in or housekeeping proteins
- VSN (Variance Stabilizing Normalization): Iterative MLE-based transformation

The fundamental assumption underlying median normalization is that
most proteins do not change between conditions, so systematic differences
between samples reflect technical variation rather than biology.

References:
    - Choi et al. (2014) MSstats: Bioinformatics 30(17):2524-2526
    - Bolstad et al. (2003) Bioinformatics 19(2):185-193 (quantile normalization)
    - Huber et al. (2002) Bioinformatics 18(Suppl 1):S96-S104 (VSN)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

# Optional MLX for GPU acceleration
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class NormalizationMethod(Enum):
    """Available normalization methods."""

    NONE = "none"
    MEDIAN = "median"  # MSstats default: equalizeMedians
    QUANTILE = "quantile"
    GLOBAL_STANDARDS = "global_standards"
    VSNS = "vsn"  # Variance stabilizing normalization


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalization procedure.

    Attributes:
        data: Normalized data matrix (features × samples)
        method: Normalization method used
        normalization_factors: Per-sample normalization factors applied
        reference_median: Global reference median (for median normalization)
        diagnostics: Additional diagnostic information
    """

    data: NDArray[np.float64]
    method: str
    normalization_factors: NDArray[np.float64]
    reference_median: float | None = None
    diagnostics: dict | None = None


def median_normalization(
    data: NDArray[np.float64],
    reference: Literal["global", "first", "median_sample"] = "global",
) -> NormalizationResult:
    """
    Median normalization (MSstats equalizeMedians).

    Adjusts each sample so that all samples have the same median log-intensity.
    This corrects for systematic differences in total protein loading or
    instrument sensitivity between runs.

    Assumption: Most proteins do not change between conditions, so differences
    in sample medians reflect technical variation.

    Args:
        data: 2D array (n_features, n_samples) of log-transformed intensities.
        reference: How to determine the target median:
            - "global": Use median of all sample medians (default)
            - "first": Use first sample as reference
            - "median_sample": Use the sample closest to global median

    Returns:
        NormalizationResult with centered data.

    Mathematical formulation:
        normalized[i,j] = original[i,j] - (median_j - reference_median)

    where median_j is the median of sample j across all features.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    n_features, n_samples = data.shape

    # Compute per-sample medians (ignoring NaN)
    sample_medians = np.nanmedian(data, axis=0)

    # Determine reference median
    if reference == "global":
        ref_median = np.nanmedian(sample_medians)
    elif reference == "first":
        ref_median = sample_medians[0]
    elif reference == "median_sample":
        # Find sample closest to global median
        global_median = np.nanmedian(sample_medians)
        closest_idx = np.argmin(np.abs(sample_medians - global_median))
        ref_median = sample_medians[closest_idx]
    else:
        raise ValueError(f"Unknown reference type: {reference}")

    # Compute normalization factors (what to subtract from each sample)
    norm_factors = sample_medians - ref_median

    # Apply normalization
    normalized = data - norm_factors[np.newaxis, :]

    return NormalizationResult(
        data=normalized,
        method="median",
        normalization_factors=norm_factors,
        reference_median=float(ref_median),
        diagnostics={
            "original_sample_medians": sample_medians.tolist(),
            "reference_type": reference,
        },
    )


def quantile_normalization(
    data: NDArray[np.float64],
    target_distribution: NDArray[np.float64] | None = None,
    method: Literal["simple", "censored"] = "censored",
) -> NormalizationResult:
    """
    Quantile normalization with proper handling of missing values.

    Forces all samples to have identical intensity distributions by replacing
    each value with the corresponding quantile from a reference distribution.

    This is a stronger assumption than median normalization - it assumes not
    just that medians should be equal, but that entire distributions should
    be identical across samples.

    Args:
        data: 2D array (n_features, n_samples) of log-transformed intensities.
        target_distribution: Target distribution to normalize to. If None,
            uses the mean of sorted values across samples.
        method: Normalization method for handling missing values:
            - "simple": Original implementation with interpolation (biased with >20% missingness)
            - "censored": Proper handling assuming MNAR (Missing Not At Random) for low-abundance proteins

    Returns:
        NormalizationResult with quantile-normalized data.

    Algorithm (censored method):
        1. Compute target distribution from complete cases only (unbiased)
        2. For each sample, rank observed values
        3. Map ranks to corresponding quantiles in full target distribution
        4. This preserves the assumption that missing = low abundance

    Note:
        Uses scipy.stats.rankdata for O(n log n) tie handling.
        NaN values are preserved in their original positions.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    n_features, n_samples = data.shape
    normalized = data.copy()

    # Compute target distribution
    if target_distribution is None:
        if method == "censored":
            # Use only complete cases to avoid bias from missing data
            complete_mask = ~np.isnan(data).any(axis=1)
            if np.sum(complete_mask) == 0:
                # No complete cases - fall back to all observed values
                sorted_data = np.sort(data, axis=0)
                target = np.nanmean(sorted_data, axis=1)
            else:
                # Compute target from complete cases only
                complete_data = data[complete_mask, :]
                sorted_data = np.sort(complete_data, axis=0)
                with np.errstate(invalid='ignore'):  # Suppress warnings for empty slices
                    target = np.mean(sorted_data, axis=1)
        else:  # simple
            # Original: use all data (can be biased with high missingness)
            sorted_data = np.sort(data, axis=0)
            target = np.nanmean(sorted_data, axis=1)
    else:
        target = target_distribution

    # Handle edge case: empty target
    if len(target) == 0 or np.all(np.isnan(target)):
        return NormalizationResult(
            data=data.copy(),
            method=f"quantile_{method}",
            normalization_factors=np.zeros(n_samples),
            reference_median=None,
            diagnostics={"error": "No valid data for target distribution"},
        )

    # Handle each sample separately to deal with NaN patterns
    for j in range(n_samples):
        col = data[:, j]
        valid_mask = ~np.isnan(col)
        n_valid = np.sum(valid_mask)

        # Edge cases
        if n_valid == 0:
            continue
        if n_valid == 1:
            # Single value - map to median of target
            normalized[valid_mask, j] = np.nanmedian(target)
            continue

        valid_values = col[valid_mask]

        # Use scipy.stats.rankdata for efficient tie handling (O(n log n))
        # method='average' handles ties by averaging ranks
        ranks = rankdata(valid_values, method='average')  # 1-based ranks

        if method == "censored":
            # Map ranks to full target distribution
            # This assumes missing values are MNAR (low abundance)
            # Rank 1 -> target[0], Rank n_valid -> target[n_valid-1]
            # Uses only the lower quantiles of target since missing = low abundance

            # Convert 1-based ranks to 0-based indices in target
            # Use only first n_valid elements of target
            target_subset = target[:n_valid] if n_valid <= len(target) else target

            # Map fractional ranks to target quantiles via interpolation
            # ranks range from 1 to n_valid, we map to indices 0 to len(target_subset)-1
            indices = (ranks - 1) * (len(target_subset) - 1) / (n_valid - 1) if n_valid > 1 else np.array([0])

            # Handle numerical stability
            indices = np.clip(indices, 0, len(target_subset) - 1)

            # Interpolate for fractional indices
            normalized_values = np.interp(indices, np.arange(len(target_subset)), target_subset)

        else:  # simple
            # Original method: interpolate subset of target
            # This can introduce bias when samples have different missingness patterns
            if n_valid < len(target):
                # Subsample target to match number of valid values
                target_indices = np.linspace(0, len(target) - 1, n_valid)
                target_subset = np.interp(target_indices, np.arange(len(target)), target)
            else:
                target_subset = target

            # Convert 1-based ranks to 0-based indices
            indices = ranks - 1

            # Handle fractional ranks (from tie averaging) via interpolation
            normalized_values = np.empty_like(ranks)
            for i, idx in enumerate(indices):
                if idx == int(idx):
                    # Exact rank - direct lookup
                    normalized_values[i] = target_subset[int(idx)]
                else:
                    # Fractional rank - interpolate
                    low_idx = int(np.floor(idx))
                    high_idx = int(np.ceil(idx))
                    low_idx = max(0, min(low_idx, len(target_subset) - 1))
                    high_idx = max(0, min(high_idx, len(target_subset) - 1))
                    frac = idx - np.floor(idx)
                    normalized_values[i] = (1 - frac) * target_subset[low_idx] + frac * target_subset[high_idx]

        normalized[valid_mask, j] = normalized_values

    # Compute effective normalization factors (change in median)
    with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
        norm_factors = np.nanmedian(normalized, axis=0) - np.nanmedian(data, axis=0)

    return NormalizationResult(
        data=normalized,
        method=f"quantile_{method}",
        normalization_factors=norm_factors,
        reference_median=None,
        diagnostics={
            "target_distribution_range": (
                float(np.nanmin(target)),
                float(np.nanmax(target)),
            ),
            "target_distribution_length": len(target),
            "missing_value_handling": method,
        },
    )


def global_standards_normalization(
    data: NDArray[np.float64],
    standard_indices: list[int] | NDArray[np.int_],
    method: Literal["median", "mean"] = "median",
) -> NormalizationResult:
    """
    Normalization using global standards (spike-ins or housekeeping proteins).

    Normalizes based on a subset of proteins assumed to be constant across
    conditions (e.g., spike-in standards, housekeeping proteins).

    Args:
        data: 2D array (n_features, n_samples) of log-transformed intensities.
        standard_indices: Row indices of standard proteins.
        method: How to summarize standards ("median" or "mean").

    Returns:
        NormalizationResult with normalized data.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    if len(standard_indices) == 0:
        raise ValueError("At least one standard protein index required")

    standard_data = data[standard_indices, :]

    # Compute reference value per sample from standards
    if method == "median":
        sample_refs = np.nanmedian(standard_data, axis=0)
    else:
        sample_refs = np.nanmean(standard_data, axis=0)

    # Global reference (median of sample references)
    global_ref = np.nanmedian(sample_refs)

    # Normalization factors
    norm_factors = sample_refs - global_ref

    # Apply normalization
    normalized = data - norm_factors[np.newaxis, :]

    return NormalizationResult(
        data=normalized,
        method="global_standards",
        normalization_factors=norm_factors,
        reference_median=float(global_ref),
        diagnostics={
            "n_standards": len(standard_indices),
            "standard_method": method,
        },
    )


def vsn_normalization(
    data: NDArray[np.float64],
    method: Literal["simple", "proper"] = "proper",
    max_iter: int = 50,
    tol: float = 1e-4,
    use_gpu: bool = False,
    reference_sample: int | None = None,
) -> NormalizationResult:
    """
    Variance Stabilizing Normalization (VSN) from Huber et al. (2002).

    Applies an arcsinh transformation with iteratively estimated parameters
    to stabilize variance across the intensity range: h(x) = arsinh((x - a) / b)

    This is a more sophisticated approach than log transformation,
    particularly useful when variance-mean relationship is complex.

    Args:
        data: 2D array (n_features, n_samples) of RAW intensities (not log).
        method: VSN method:
            - "simple": Single-pass arcsinh(x/c) with c = median (toy implementation)
            - "proper": Iterative MLE-based VSN from Huber et al. (2002)
        max_iter: Maximum iterations for proper VSN (default: 50).
        tol: Convergence tolerance for parameter changes (default: 1e-4).
        use_gpu: Use MLX for GPU acceleration if available (default: False).
        reference_sample: Index of reference sample, or None for global.

    Returns:
        NormalizationResult with VSN-transformed data.

    Algorithm (proper method):
        1. Initialize: a = 0, b = median(x) for each sample
        2. Iterate until convergence:
           a. Transform: y = arsinh((x - a) / b)
           b. Compute reference array (row-wise mean across samples)
           c. Re-estimate a, b via weighted least squares
        3. Apply final transformation

    References:
        Huber et al. (2002) Bioinformatics 18(Suppl 1):S96-S104
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    if method == "simple":
        return _vsn_simple(data)
    elif method == "proper":
        return _vsn_proper(data, max_iter=max_iter, tol=tol, use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown VSN method: {method}")


def _vsn_simple(data: NDArray[np.float64]) -> NormalizationResult:
    """
    Simple VSN: single-pass arcsinh(x/c) transformation.

    This is the original toy implementation - kept for backwards compatibility.
    """
    n_features, n_samples = data.shape
    normalized = np.empty_like(data)
    norm_factors = np.empty(n_samples)

    for j in range(n_samples):
        col = data[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            normalized[:, j] = np.nan
            norm_factors[j] = np.nan
            continue

        # Estimate scale parameter (robust)
        c = np.median(valid[valid > 0]) if np.any(valid > 0) else 1.0

        # Apply transformation
        normalized[:, j] = np.arcsinh(col / c)
        norm_factors[j] = c

    return NormalizationResult(
        data=normalized,
        method="vsn_simple",
        normalization_factors=norm_factors,
        reference_median=None,
        diagnostics={"transformation": "arcsinh(x/c)", "algorithm": "simple"},
    )


def _vsn_proper(
    data: NDArray[np.float64],
    max_iter: int = 50,
    tol: float = 1e-4,
    use_gpu: bool = False,
) -> NormalizationResult:
    """
    Proper iterative MLE-based VSN from Huber et al. (2002).

    Transformation: h(x) = arsinh((x - a) / b)
    Parameters a (offset) and b (scale) are estimated via iterative weighted least squares.
    """
    n_features, n_samples = data.shape

    # Use GPU acceleration if requested and available
    if use_gpu and HAS_MLX:
        return _vsn_proper_mlx(data, max_iter, tol)

    # CPU implementation
    # Initialize parameters for each sample
    a = np.zeros(n_samples)  # offset
    b = np.empty(n_samples)  # scale

    for j in range(n_samples):
        col = data[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) == 0:
            b[j] = 1.0
        else:
            # Initialize b as median of positive values
            b[j] = np.median(valid[valid > 0]) if np.any(valid > 0) else 1.0

    # Iterative parameter estimation
    converged = False
    iteration_history = []

    for iteration in range(max_iter):
        # 1. Apply current transformation to all samples
        y = np.empty_like(data)
        for j in range(n_samples):
            # Handle edge case: ensure numerical stability
            x_shifted = data[:, j] - a[j]
            # Clip to avoid numerical issues with arcsinh
            x_shifted = np.clip(x_shifted, -1e10, 1e10)
            y[:, j] = np.arcsinh(x_shifted / np.maximum(b[j], 1e-10))

        # 2. Compute reference array (row-wise mean across samples)
        # This represents the "true" signal common to all samples
        ref = np.nanmean(y, axis=1)

        # 3. Re-estimate parameters a and b for each sample via weighted least squares
        a_new = np.empty(n_samples)
        b_new = np.empty(n_samples)

        for j in range(n_samples):
            col = data[:, j]
            valid_mask = ~np.isnan(col)

            if np.sum(valid_mask) == 0:
                a_new[j] = a[j]
                b_new[j] = b[j]
                continue

            x = col[valid_mask]
            y_ref = ref[valid_mask]

            # Estimate parameters by matching transformed data to reference
            # Inverse transform: x = a + b * sinh(y_ref)
            # Use robust estimation (median-based)

            # For a: use median of (x - b * sinh(y_ref))
            sinh_ref = np.sinh(np.clip(y_ref, -10, 10))  # Clip for numerical stability
            a_new[j] = np.median(x - b[j] * sinh_ref)

            # For b: use median of |x - a| / |sinh(y_ref)|
            x_centered = x - a_new[j]
            sinh_ref_nonzero = sinh_ref[np.abs(sinh_ref) > 1e-10]
            x_centered_nonzero = x_centered[np.abs(sinh_ref) > 1e-10]

            if len(sinh_ref_nonzero) > 0:
                b_new[j] = np.median(np.abs(x_centered_nonzero) / np.abs(sinh_ref_nonzero))
            else:
                b_new[j] = b[j]

            # Ensure b is positive and reasonable
            b_new[j] = np.maximum(b_new[j], 1e-10)

        # 4. Check convergence
        a_change = np.max(np.abs(a_new - a))
        b_change = np.max(np.abs(b_new - b) / np.maximum(b, 1e-10))
        max_change = max(a_change, b_change)

        iteration_history.append({
            "iteration": iteration + 1,
            "a_change": float(a_change),
            "b_change": float(b_change),
            "max_change": float(max_change),
        })

        # Update parameters
        a = a_new
        b = b_new

        if max_change < tol:
            converged = True
            break

    # Apply final transformation
    normalized = np.empty_like(data)
    for j in range(n_samples):
        x_shifted = data[:, j] - a[j]
        x_shifted = np.clip(x_shifted, -1e10, 1e10)
        normalized[:, j] = np.arcsinh(x_shifted / np.maximum(b[j], 1e-10))

    return NormalizationResult(
        data=normalized,
        method="vsn_proper",
        normalization_factors=b,  # scale parameters
        reference_median=None,
        diagnostics={
            "transformation": "arsinh((x - a) / b)",
            "algorithm": "iterative MLE",
            "converged": converged,
            "iterations": len(iteration_history),
            "final_a": a.tolist(),
            "final_b": b.tolist(),
            "iteration_history": iteration_history,
        },
    )


def _vsn_proper_mlx(
    data: NDArray[np.float64],
    max_iter: int = 50,
    tol: float = 1e-4,
) -> NormalizationResult:
    """
    GPU-accelerated VSN using MLX.

    Same algorithm as _vsn_proper but with MLX arrays for GPU computation.
    """
    n_features, n_samples = data.shape

    # Convert to MLX arrays
    data_mx = mx.array(data)

    # Initialize parameters
    a = mx.zeros(n_samples)
    b = mx.array([np.median(data[:, j][~np.isnan(data[:, j])][data[:, j][~np.isnan(data[:, j])] > 0])
                  if np.sum(~np.isnan(data[:, j])) > 0 and np.any(data[:, j][~np.isnan(data[:, j])] > 0)
                  else 1.0 for j in range(n_samples)])

    converged = False
    iteration_history = []

    for iteration in range(max_iter):
        # Apply transformation
        y = mx.arcsinh((data_mx - a[None, :]) / mx.maximum(b[None, :], 1e-10))

        # Compute reference (handle NaN)
        ref = mx.mean(mx.where(mx.isnan(y), 0.0, y), axis=1)

        # Re-estimate parameters (simplified for GPU)
        a_new = mx.zeros(n_samples)
        b_new = mx.zeros(n_samples)

        for j in range(n_samples):
            col = data_mx[:, j]
            valid_mask = ~mx.isnan(col)

            if mx.sum(valid_mask) == 0:
                a_new[j] = a[j]
                b_new[j] = b[j]
                continue

            x = col[valid_mask]
            y_ref = ref[valid_mask]

            sinh_ref = mx.sinh(mx.clip(y_ref, -10, 10))
            a_new[j] = mx.median(x - b[j] * sinh_ref)

            x_centered = x - a_new[j]
            mask_nonzero = mx.abs(sinh_ref) > 1e-10

            if mx.sum(mask_nonzero) > 0:
                b_new[j] = mx.median(mx.abs(x_centered[mask_nonzero]) / mx.abs(sinh_ref[mask_nonzero]))
            else:
                b_new[j] = b[j]

            b_new[j] = mx.maximum(b_new[j], 1e-10)

        # Check convergence
        a_change = float(mx.max(mx.abs(a_new - a)))
        b_change = float(mx.max(mx.abs(b_new - b) / mx.maximum(b, 1e-10)))
        max_change = max(a_change, b_change)

        iteration_history.append({
            "iteration": iteration + 1,
            "a_change": a_change,
            "b_change": b_change,
            "max_change": max_change,
        })

        a = a_new
        b = b_new

        if max_change < tol:
            converged = True
            break

    # Apply final transformation
    normalized = mx.arcsinh((data_mx - a[None, :]) / mx.maximum(b[None, :], 1e-10))

    # Convert back to numpy
    normalized_np = np.array(normalized)
    a_np = np.array(a)
    b_np = np.array(b)

    return NormalizationResult(
        data=normalized_np,
        method="vsn_proper_gpu",
        normalization_factors=b_np,
        reference_median=None,
        diagnostics={
            "transformation": "arsinh((x - a) / b)",
            "algorithm": "iterative MLE (GPU)",
            "converged": converged,
            "iterations": len(iteration_history),
            "final_a": a_np.tolist(),
            "final_b": b_np.tolist(),
            "iteration_history": iteration_history,
            "accelerator": "MLX",
        },
    )


def normalize(
    data: NDArray[np.float64],
    method: NormalizationMethod | str = NormalizationMethod.MEDIAN,
    **kwargs,
) -> NormalizationResult:
    """
    Apply normalization to proteomics data.

    This is the main entry point for normalization, dispatching to
    the appropriate method.

    Args:
        data: 2D array (n_features, n_samples) of intensities.
        method: Normalization method to use.
        **kwargs: Additional arguments passed to the specific method.

    Returns:
        NormalizationResult with normalized data.
    """
    if isinstance(method, str):
        method = NormalizationMethod(method)

    if method == NormalizationMethod.NONE:
        return NormalizationResult(
            data=data.copy(),
            method="none",
            normalization_factors=np.zeros(data.shape[1]),
            reference_median=None,
        )

    elif method == NormalizationMethod.MEDIAN:
        return median_normalization(data, **kwargs)

    elif method == NormalizationMethod.QUANTILE:
        return quantile_normalization(data, **kwargs)

    elif method == NormalizationMethod.GLOBAL_STANDARDS:
        return global_standards_normalization(data, **kwargs)

    elif method == NormalizationMethod.VSNS:
        return vsn_normalization(data, **kwargs)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def assess_normalization_quality(
    before: NDArray[np.float64],
    after: NDArray[np.float64],
) -> dict:
    """
    Assess quality of normalization by comparing before/after statistics.

    Args:
        before: Original data matrix.
        after: Normalized data matrix.

    Returns:
        Dictionary with quality metrics.
    """
    # Sample medians
    medians_before = np.nanmedian(before, axis=0)
    medians_after = np.nanmedian(after, axis=0)

    # Sample standard deviations
    stds_before = np.nanstd(before, axis=0)
    stds_after = np.nanstd(after, axis=0)

    # Coefficient of variation of medians (should decrease)
    cv_medians_before = np.std(medians_before) / np.mean(medians_before) if np.mean(medians_before) != 0 else np.nan
    cv_medians_after = np.std(medians_after) / np.mean(medians_after) if np.mean(medians_after) != 0 else np.nan

    return {
        "median_cv_before": float(cv_medians_before),
        "median_cv_after": float(cv_medians_after),
        "median_cv_reduction": float((cv_medians_before - cv_medians_after) / cv_medians_before) if cv_medians_before != 0 else np.nan,
        "median_range_before": float(np.ptp(medians_before)),
        "median_range_after": float(np.ptp(medians_after)),
        "mean_std_before": float(np.mean(stds_before)),
        "mean_std_after": float(np.mean(stds_after)),
    }
