"""
Competitive enrichment z-score computation.

Shared utility for computing the competitive enrichment z-score
(mean|t| of targets vs background, standardized by background
distribution). Used by:
- label_permutation.py (Phase 3)
- negative_controls.py (Phase 5)
- specificity.py (Phase 2 interaction test)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_competitive_z(
    t_statistics: NDArray[np.float64],
    is_target: NDArray[np.bool_],
    *,
    robust: bool = False,
) -> float:
    """
    Compute competitive enrichment z-score from t-statistics.

    Tests whether target genes have systematically higher |t-statistics|
    than background genes, returning a standardized z-score.

    Standard mode (``robust=False``):

        z = (mean|t_target| - mean|t_background|) / std|t_background|

    Robust mode (``robust=True``):

        z = (median|t_target| - median|t_background|) / MAD|t_background|

    where MAD = 1.4826 * median(|x - median(x)|) is the median absolute
    deviation scaled for asymptotic normal consistency.

    This is the same statistic used by run_network_enrichment_test()
    but computed directly without the 10,000 inner competitive
    permutations. Suitable for use inside permutation loops where
    the outer loop provides the null distribution.

    Args:
        t_statistics: Array of t-statistics for all features (n_features,).
        is_target: Boolean mask identifying target features (n_features,).
            Must be same length as t_statistics.
        robust: If True, use median + MAD instead of mean + std.
            This is resistant to outlier t-statistics that can inflate
            or deflate the standard z-score.

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

    if robust:
        target_center = float(np.median(abs_t[is_target]))
        bg_center = float(np.median(background))
        # MAD with consistency constant for normal distribution
        bg_mad = 1.4826 * float(np.median(np.abs(background - np.median(background))))
        if bg_mad < 1e-10:
            return 0.0
        return (target_center - bg_center) / bg_mad
    else:
        target_mean = float(np.mean(abs_t[is_target]))
        bg_mean = float(np.mean(background))
        bg_std = float(np.std(background, ddof=1))
        if bg_std < 1e-10:
            return 0.0
        return (target_mean - bg_mean) / bg_std


def compute_robust_competitive_z(
    t_statistics: NDArray[np.float64],
    is_target: NDArray[np.bool_],
) -> float:
    """
    Convenience wrapper: compute competitive z using median + MAD.

    Equivalent to ``compute_competitive_z(t_statistics, is_target, robust=True)``.
    See :func:`compute_competitive_z` for full documentation.
    """
    return compute_competitive_z(t_statistics, is_target, robust=True)
