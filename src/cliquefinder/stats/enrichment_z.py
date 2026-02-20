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
) -> float:
    """
    Compute competitive enrichment z-score from t-statistics.

    Tests whether target genes have systematically higher |t-statistics|
    than background genes, returning a standardized z-score:

        z = (mean|t_target| - mean|t_background|) / std|t_background|

    This is the same statistic used by run_network_enrichment_test()
    but computed directly without the 10,000 inner competitive
    permutations. Suitable for use inside permutation loops where
    the outer loop provides the null distribution.

    Args:
        t_statistics: Array of t-statistics for all features (n_features,).
        is_target: Boolean mask identifying target features (n_features,).
            Must be same length as t_statistics.

    Returns:
        Competitive z-score. Returns 0.0 for degenerate cases
        (no targets, all targets, or zero-variance background).
    """
    n_total = len(t_statistics)
    n_targets = int(np.sum(is_target))

    if n_targets == 0 or n_targets >= n_total:
        return 0.0

    abs_t = np.abs(t_statistics)
    target_mean = float(np.mean(abs_t[is_target]))

    background = abs_t[~is_target]
    bg_mean = float(np.mean(background))
    bg_std = float(np.std(background, ddof=1))

    if bg_std < 1e-10:
        return 0.0

    return (target_mean - bg_mean) / bg_std
