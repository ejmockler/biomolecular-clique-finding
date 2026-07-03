"""Intensity-scale transform for the per-protein moderated-t fit.

Both production gradient paths — the exhaustive landscape
(``landscape._fit_engine_for_contrast``) and the per-seed panel runner
(``seed_runner.run_seed_gradient``) — map the abundance matrix onto a
modeling scale *before* the ROAST engine sees it.  This module is the
single source of truth for that transform so the two paths cannot
drift, and lives in its own module because ``landscape`` imports
``seed_runner`` (a shared helper in either would be a circular import).

``"log2"`` (the default) is ``log2(x+1)`` — the correct scale for
proteomic intensities spanning ~10 orders of magnitude and strongly
right-skewed, and the scale the report and engine docstrings assume.
The ``+1`` pseudocount maps the AnswerALS matrix's 120 exact zeros
(0.008% of cells; min positive ≈ 13.6) to 0 instead of ``-inf``, with
negligible distortion above the ~1st percentile (≈ 290).  ``"raw"``
fits on the delivered linear intensities — the historical pre-2026-07
production default, kept byte-identically reproducible.

The choice is recorded on the design (``LandscapeDesign.transform`` /
``PanelDesign.transform``) so every manifest is self-describing and the
resume design-equality guard refuses to mix scales on one output_dir.
"""
from __future__ import annotations

import numpy as np

LOG2_TRANSFORM = "log2"
RAW_TRANSFORM = "raw"
VALID_TRANSFORMS = frozenset({LOG2_TRANSFORM, RAW_TRANSFORM})


def apply_intensity_transform(data: np.ndarray, transform: str) -> np.ndarray:
    """Map a raw abundance matrix onto the modeling scale.

    Parameters
    ----------
    data
        Raw intensity matrix ``(n_features, n_samples)``.
    transform
        ``"raw"`` returns the matrix unchanged (same object).  ``"log2"``
        returns ``log2(data + 1)``.

    Negative inputs are rejected under ``"log2"``: the log is undefined
    there, and a negative entry means the matrix is *not* raw intensities
    (e.g. already log-transformed) — silently log-transforming it twice
    would corrupt the analysis, so we fail loud instead.  NaNs are
    preserved (the downstream engine handles missingness).
    """
    if transform == RAW_TRANSFORM:
        return data
    if transform == LOG2_TRANSFORM:
        finite = data[np.isfinite(data)]
        if finite.size and float(finite.min()) < 0.0:
            raise ValueError(
                f"log2 transform requires non-negative intensities, but the "
                f"matrix has min={float(finite.min()):.4g}.  Refusing to "
                f"log-transform a matrix that is not raw abundances (already "
                f"log-scaled? wrong input?)."
            )
        return np.log2(data + 1.0)
    raise ValueError(
        f"unknown transform {transform!r}; expected one of "
        f"{sorted(VALID_TRANSFORMS)}"
    )
