"""
Competitive enrichment z-score computation.

Shared utility for computing the competitive enrichment z-score
(mean|t| of targets vs background, standardized by the standard error
of the mean). Used by:
- label_permutation.py (Phase 3)
- negative_controls.py (Phase 5)
- specificity.py (Phase 2 interaction test)

Statistical corrections (Camera VIF):
    The denominator uses SE = sigma / sqrt(k) rather than sigma alone
    (STAT-3 fix), and optionally applies the Camera variance inflation
    factor VIF = 1 + (k-1)*rho_bar when inter-gene correlation is
    provided (STAT-2 fix). See Wu & Smyth, NAR 2012.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_competitive_z(
    t_statistics: NDArray[np.float64],
    is_target: NDArray[np.bool_],
    *,
    robust: bool = False,
    inter_gene_correlation: float | None = None,
) -> float:
    """
    Compute competitive enrichment z-score from t-statistics.

    Tests whether target genes have systematically higher |t-statistics|
    than background genes, returning a standardized z-score.

    Standard mode (``robust=False``):

        z = (mean|t_target| - mean|t_background|) / SE

    where SE = std|t_background| / sqrt(k) * sqrt(VIF) and
    VIF = 1 + (k-1)*rho_bar (Camera variance inflation factor).

    Robust mode (``robust=True``):

        z = (median|t_target| - median|t_background|) / SE_robust

    where SE_robust = MAD|t_background| / sqrt(k) * sqrt(VIF) and
    MAD = 1.4826 * median(|x - median(x)|).

    When ``inter_gene_correlation`` is None, VIF defaults to 1.0
    (no correlation adjustment). This preserves backward compatibility
    while still applying the SE-of-the-mean correction (STAT-3).

    The Camera VIF accounts for positive inter-gene correlation that
    inflates the variance of the set mean |t| beyond what independent
    sampling would predict. For TF target sets this is expected because
    co-regulated genes share upstream signal.

    References:
        Wu & Smyth (2012) "Camera: a competitive gene set test
        accounting for inter-gene correlation", NAR 40(17):e133.

    Args:
        t_statistics: Array of t-statistics for all features (n_features,).
        is_target: Boolean mask identifying target features (n_features,).
            Must be same length as t_statistics.
        robust: If True, use median + MAD instead of mean + std.
            This is resistant to outlier t-statistics that can inflate
            or deflate the standard z-score.
        inter_gene_correlation: Mean pairwise correlation (rho_bar)
            among target genes. If provided and > 0, applies the Camera
            variance inflation factor VIF = 1 + (k-1)*rho_bar.
            If None (default), no VIF adjustment is applied.

    Returns:
        Competitive z-score. Returns 0.0 for degenerate cases
        (no targets, all targets, zero-variance background, or
        zero-MAD background in robust mode).
    """
    n_total = len(t_statistics)
    n_targets = int(np.sum(is_target))

    if n_targets == 0 or n_targets >= n_total:
        return 0.0

    abs_t = np.abs(t_statistics)
    background = abs_t[~is_target]
    k = n_targets

    # Compute Camera VIF
    vif = 1.0
    if inter_gene_correlation is not None and inter_gene_correlation > 0:
        vif = 1.0 + (k - 1) * inter_gene_correlation

    if robust:
        target_center = float(np.median(abs_t[is_target]))
        bg_center = float(np.median(background))
        # MAD with consistency constant for normal distribution
        bg_mad = 1.4826 * float(np.median(np.abs(background - np.median(background))))
        if bg_mad < 1e-10:
            return 0.0
        # SE of the median (using MAD as robust scale), with VIF
        se = bg_mad / np.sqrt(k) * np.sqrt(vif) if k > 1 else bg_mad
        if se < 1e-10:
            return 0.0
        return (target_center - bg_center) / se
    else:
        target_mean = float(np.mean(abs_t[is_target]))
        bg_mean = float(np.mean(background))
        bg_std = float(np.std(background, ddof=1))
        if bg_std < 1e-10:
            return 0.0
        # SE of the mean, with Camera VIF
        se = bg_std / np.sqrt(k) * np.sqrt(vif) if k > 1 else bg_std
        if se < 1e-10:
            return 0.0
        return (target_mean - bg_mean) / se


def compute_robust_competitive_z(
    t_statistics: NDArray[np.float64],
    is_target: NDArray[np.bool_],
    *,
    inter_gene_correlation: float | None = None,
) -> float:
    """
    Convenience wrapper: compute competitive z using median + MAD.

    Equivalent to ``compute_competitive_z(t_statistics, is_target, robust=True)``.
    See :func:`compute_competitive_z` for full documentation.
    """
    return compute_competitive_z(
        t_statistics,
        is_target,
        robust=True,
        inter_gene_correlation=inter_gene_correlation,
    )


def estimate_inter_gene_correlation(
    expression_data: NDArray[np.float64],
    is_target: NDArray[np.bool_],
) -> float:
    """
    Estimate mean pairwise correlation among target genes.

    Uses the raw expression matrix (not residuals/t-statistics) to avoid
    double-dipping. Returns rho_bar floored at 0 (negative average
    correlation is conservative and needs no correction).

    This implements the inter-gene correlation estimate from Camera
    (Wu & Smyth, NAR 2012).

    Args:
        expression_data: Expression matrix of shape (n_genes, n_samples).
            Rows are genes, columns are samples.
        is_target: Boolean mask identifying target genes (n_genes,).

    Returns:
        Mean pairwise correlation (rho_bar), floored at 0.0.
        Returns 0.0 if fewer than 2 target genes.
    """
    target_data = expression_data[is_target, :]
    k = target_data.shape[0]
    if k < 2:
        return 0.0

    # Compute pairwise correlation matrix
    corr_matrix = np.corrcoef(target_data)

    # Mean off-diagonal correlation
    # Sum of all entries minus diagonal (k ones), divided by k*(k-1) off-diag entries
    rho_bar = (corr_matrix.sum() - k) / (k * (k - 1))

    # Floor at 0 -- negative average correlation is conservative
    return max(float(rho_bar), 0.0)
