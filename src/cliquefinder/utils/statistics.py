"""
Shared statistical utilities for biomarker analysis.

This module provides common statistical functions used across the quality
control and biomarker discovery modules, eliminating code duplication.

Functions:
    otsu_threshold: Find optimal bimodal threshold using Otsu's method
    cohens_d: Compute Cohen's d effect size between two groups
"""

from __future__ import annotations

import numpy as np


__all__ = [
    'otsu_threshold',
    'cohens_d',
]


def otsu_threshold(values: np.ndarray) -> float:
    """
    Compute optimal bimodal threshold using Otsu's method.

    Otsu's method finds the threshold that minimizes within-class variance
    (equivalently, maximizes between-class variance). This is ideal for
    bimodal distributions like Y-linked gene expression in sex determination.

    Args:
        values: 1D array of values to threshold. NaN values are excluded.

    Returns:
        Optimal threshold value that maximizes between-class variance.

    Note:
        For small samples (< 10), returns median as a robust fallback.

    Example:
        >>> # Find threshold separating male/female DDX3Y expression
        >>> ddx3y_values = matrix.data[ddx3y_idx, :]
        >>> threshold = otsu_threshold(ddx3y_values)
        >>> is_male = ddx3y_values > threshold

    References:
        Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms"
        IEEE Trans. Sys. Man. Cyber. 9 (1): 62-66.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) < 10:
        return float(np.median(values))

    # Adaptive bin count based on sample size
    n_bins = min(100, len(values) // 5)
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float) / hist.sum()

    best_threshold = bin_centers[0]
    best_variance = 0.0

    for i in range(1, len(hist)):
        # Class probabilities
        w0 = hist[:i].sum()
        w1 = hist[i:].sum()

        if w0 < 1e-10 or w1 < 1e-10:
            continue

        # Class means
        mu0 = (hist[:i] * bin_centers[:i]).sum() / w0
        mu1 = (hist[i:] * bin_centers[i:]).sum() / w1

        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = bin_centers[i]

    return float(best_threshold)


def cohens_d(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two group means,
    providing a scale-independent measure of separation quality.

    Args:
        values: 1D array of measurements
        labels: 1D array of binary group labels (0/1 or boolean)

    Returns:
        Absolute Cohen's d effect size (always non-negative)

    Interpretation:
        d < 0.2:  negligible effect
        d = 0.2:  small effect
        d = 0.5:  medium effect
        d = 0.8:  large effect
        d > 1.5:  very large effect
        d > 2.0:  excellent separation (ideal for biomarkers)

    Note:
        Returns 0.0 if either group has fewer than 2 samples, or if
        pooled standard deviation is near zero.

    Example:
        >>> # Measure sex separation by DDX3Y expression
        >>> is_male = ddx3y_values > threshold
        >>> d = cohens_d(ddx3y_values, is_male)
        >>> print(f"Effect size: {d:.2f}")  # e.g., "Effect size: 2.39"

    References:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    values = np.asarray(values)
    labels = np.asarray(labels)

    # Handle boolean labels
    if labels.dtype == bool:
        labels = labels.astype(int)

    g0 = values[labels == 0]
    g1 = values[labels == 1]

    if len(g0) < 2 or len(g1) < 2:
        return 0.0

    n0, n1 = len(g0), len(g1)
    var0 = np.var(g0, ddof=1)
    var1 = np.var(g1, ddof=1)

    # Pooled standard deviation (Welch-Satterthwaite)
    pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float(abs(np.mean(g1) - np.mean(g0)) / pooled_std)
