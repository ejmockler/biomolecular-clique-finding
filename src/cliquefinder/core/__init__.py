"""
Core data structures and abstractions for the biomolecular analysis platform.

This module provides the foundational types that all other modules build upon:

1. BioMatrix: Expression/count matrix with biological metadata and quality tracking
2. QualityFlag: Bitwise flags for tracking data provenance and quality
3. Transform: Abstract base class for immutable matrix transformations

Design Philosophy:
    - Immutability: All operations return new instances (functional style)
    - Type Safety: Full type hints for IDE support and mypy validation
    - Composability: Small operations chain into complex pipelines
    - Domain Expertise: Biological best practices built into the API

Examples:
    >>> from cliquefinder.core import BioMatrix, QualityFlag, Transform
    >>>
    >>> # Create matrix
    >>> matrix = BioMatrix(...)
    >>>
    >>> # Check quality flags
    >>> n_imputed = np.sum(matrix.quality_flags & QualityFlag.IMPUTED != 0)
    >>>
    >>> # Apply transformation
    >>> class LogTransform(Transform):
    ...     def apply(self, matrix: BioMatrix) -> BioMatrix:
    ...         result = matrix.copy()
    ...         result._data = np.log2(matrix.data + 1)
    ...         return result
"""

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.core.transform import Transform

__all__ = [
    'BioMatrix',
    'QualityFlag',
    'Transform',
]
