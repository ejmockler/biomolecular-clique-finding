"""
Reusable competitive z-score test for gene set enrichment.

Computes a competitive enrichment statistic by comparing the mean absolute
t-statistic of a target gene set against the background, standardized by
the standard error of the mean with optional Camera VIF correction.

This module wraps the core logic from enrichment_z.py into a convenient
function signature that accepts target indices (rather than boolean masks),
making it suitable for direct use by DiscoveryBridge where gene indices
are already resolved.

References:
    Wu & Smyth (2012) "Camera: a competitive gene set test accounting
    for inter-gene correlation", NAR 40(17):e133.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


def competitive_z_test(
    t_statistics: NDArray[np.float64],
    target_indices: NDArray[np.intp],
    correlation_matrix: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """
    Compute a competitive z-score and two-sided p-value for a gene set.

    Tests whether target genes have systematically higher |t-statistics|
    than the background (all other genes), using the Camera VIF to account
    for inter-gene correlation.

    Formula:
        z = (mean|t_target| - mean|t_background|) / SE
        SE = std|t_background| / sqrt(k) * sqrt(VIF)
        VIF = 1 + (k-1) * rho_bar

    where rho_bar is the mean pairwise correlation among target genes.

    Args:
        t_statistics: Array of t-statistics for ALL genes (n_genes,).
            These are typically moderated t-statistics from the fitted
            ROAST engine.
        target_indices: Integer indices of genes in the target set
            within the t_statistics array.
        correlation_matrix: Optional pairwise correlation matrix among
            target genes (k x k). If provided, VIF is estimated from
            the mean off-diagonal correlation. If None, VIF = 1.

    Returns:
        Tuple of (z_score, p_value) where p_value is two-sided.
        Returns (0.0, 1.0) for degenerate cases (empty set, all genes
        in set, zero-variance background).
    """
    n_total = len(t_statistics)
    k = len(target_indices)

    if k == 0 or k >= n_total:
        return 0.0, 1.0

    abs_t = np.abs(t_statistics)

    # Target and background statistics
    target_abs_t = abs_t[target_indices]
    background_mask = np.ones(n_total, dtype=bool)
    background_mask[target_indices] = False
    background_abs_t = abs_t[background_mask]

    if len(background_abs_t) < 2:
        return 0.0, 1.0

    target_mean = float(np.mean(target_abs_t))
    bg_mean = float(np.mean(background_abs_t))
    bg_std = float(np.std(background_abs_t, ddof=1))

    if bg_std < 1e-10:
        return 0.0, 1.0

    # Compute VIF from correlation matrix
    vif = 1.0
    if correlation_matrix is not None and k > 1:
        rho_bar = _mean_off_diagonal(correlation_matrix)
        if rho_bar > 0:
            vif = 1.0 + (k - 1) * rho_bar

    # Standard error of the mean with VIF
    se = bg_std / np.sqrt(k) * np.sqrt(vif)

    if se < 1e-10:
        return 0.0, 1.0

    z_score = (target_mean - bg_mean) / se

    # Two-sided p-value from standard normal
    p_value = 2.0 * scipy_stats.norm.sf(abs(z_score))

    return float(z_score), float(p_value)


def _mean_off_diagonal(corr_matrix: NDArray[np.float64]) -> float:
    """Compute mean off-diagonal correlation, floored at 0.

    Handles NaN entries (from constant-expression genes) by excluding
    them from the mean.

    Args:
        corr_matrix: Square correlation matrix (k x k).

    Returns:
        Mean off-diagonal correlation, floored at 0.0.
    """
    k = corr_matrix.shape[0]
    if k < 2:
        return 0.0

    # Zero the diagonal, then compute mean of finite off-diagonal entries
    mat = corr_matrix.copy()
    np.fill_diagonal(mat, 0.0)

    n_valid = int(np.sum(np.isfinite(mat))) - k  # subtract zeroed diagonal
    if n_valid < 1:
        return 0.0

    rho_bar = float(np.nansum(mat)) / n_valid
    return max(rho_bar, 0.0)
