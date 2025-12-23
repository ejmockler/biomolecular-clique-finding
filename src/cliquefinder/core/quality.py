"""
Quality flag system for tracking data provenance and quality attributes.

This module provides a bitwise flag system for marking individual values in expression
matrices with quality annotations. Critical for scientific reproducibility - enables
answering reviewer questions like "which values were imputed?" or "what proportion
of the dataset was batch-corrected?"

Biological Context:
    In proteomics/transcriptomics, not all measurements are equal. Some may be:
    - Statistical outliers due to technical artifacts
    - Imputed to handle missingness
    - Low confidence due to poor signal-to-noise
    - Batch corrected to remove systematic technical variation

    Tracking provenance is essential for publication and reanalysis.

Engineering Design:
    IntFlag enables efficient bitwise operations:
    - Multiple flags per value: OUTLIER_DETECTED | IMPUTED
    - Fast bitwise checks: if flags & QualityFlag.IMPUTED
    - Memory efficient: single int per value
    - Composable: flags combine naturally with | operator

Examples:
    >>> from cliquefinder.core.quality import QualityFlag
    >>>
    >>> # Mark a value as both outlier and imputed
    >>> flag = QualityFlag.OUTLIER_DETECTED | QualityFlag.IMPUTED
    >>>
    >>> # Check if value was imputed
    >>> if flag & QualityFlag.IMPUTED:
    ...     print("This value was imputed")
    >>>
    >>> # Count imputed values in array
    >>> import numpy as np
    >>> flags = np.array([0, 1, 2, 3, 2], dtype=int)
    >>> n_imputed = np.sum(flags & QualityFlag.IMPUTED != 0)
"""

from __future__ import annotations

from enum import IntFlag

__all__ = ['QualityFlag']


class QualityFlag(IntFlag):
    """
    Bitwise flags for per-value quality tracking in expression matrices.

    Multiple flags can be combined using the bitwise OR operator (|).
    Each flag represents a specific quality attribute or transformation
    applied to a value.

    Attributes:
        ORIGINAL: Untouched original value from raw data (0)
        OUTLIER_DETECTED: Flagged as statistical outlier (1)
        IMPUTED: Value was imputed to replace missing/outlier (2)
        MISSING_ORIGINAL: Originally missing (NaN) in raw data (4)
        BATCH_CORRECTED: Underwent batch effect correction (8)
        LOW_CONFIDENCE: Low quality measurement (low counts, poor S/N) (16)
        MANUAL_REVIEW: Flagged for manual inspection by analyst (32)

    Scientific Rationale:
        - OUTLIER_DETECTED: Statistical outliers (e.g., >3 SD from mean) may
          represent technical artifacts rather than biological signal
        - IMPUTED: Imputation methods (KNN, MICE) introduce synthetic data that
          may affect downstream statistics
        - MISSING_ORIGINAL: Complete case analysis vs imputation decisions depend
          on missingness patterns (MCAR vs MAR vs MNAR)
        - BATCH_CORRECTED: Batch effects are confounding variables that must be
          removed, but correction may introduce artifacts
        - LOW_CONFIDENCE: Low-abundance proteins/transcripts have higher measurement
          error and may need different statistical treatment
        - MANUAL_REVIEW: Complex cases may need domain expert review before
          including in analysis

    Examples:
        >>> # Original value, no transformations
        >>> flag = QualityFlag.ORIGINAL
        >>>
        >>> # Value was originally missing and then imputed
        >>> flag = QualityFlag.MISSING_ORIGINAL | QualityFlag.IMPUTED
        >>>
        >>> # Check if value was imputed
        >>> if flag & QualityFlag.IMPUTED:
        ...     print("Value was imputed")
        >>>
        >>> # Check multiple conditions
        >>> if (flag & QualityFlag.OUTLIER_DETECTED) and (flag & QualityFlag.IMPUTED):
        ...     print("Outlier was imputed")
        >>>
        >>> # All transformations for a heavily processed value
        >>> flag = (QualityFlag.OUTLIER_DETECTED |
        ...         QualityFlag.IMPUTED |
        ...         QualityFlag.BATCH_CORRECTED)
    """

    ORIGINAL = 0
    """Untouched original value - no quality issues or transformations."""

    OUTLIER_DETECTED = 1
    """
    Flagged as statistical outlier (e.g., Grubbs test, IQR method, MAD).
    May indicate technical artifact, contamination, or true biological extreme.
    """

    IMPUTED = 2
    """
    Value was imputed using statistical method (KNN, MICE, SVD, etc.).
    Imputed values have different statistical properties than measured values.
    """

    MISSING_ORIGINAL = 4
    """
    Originally missing (NaN) in raw data.
    Missingness may be informative (MNAR: missing not at random).
    """

    BATCH_CORRECTED = 8
    """
    Underwent batch effect correction (ComBat, limma, etc.).
    Batch correction removes systematic technical variation but may introduce artifacts.
    """

    LOW_CONFIDENCE = 16
    """
    Low quality measurement.
    Examples: low read counts, poor signal-to-noise ratio, high CV in replicates.
    """

    MANUAL_REVIEW = 32
    """
    Flagged for manual inspection by domain expert.
    Use for complex cases requiring biological interpretation.
    """
