"""
Base transformation framework for immutable matrix operations.

This module provides the abstract base class for all data transformations in the
platform. Follows functional programming principles: transformations are pure
functions that return new instances rather than modifying data in place.

Biological Context:
    Proteomics/transcriptomics pipelines involve many sequential transformations:
    1. Quality control (outlier detection, filtering)
    2. Normalization (TPM, RPKM, TMM, quantile)
    3. Batch correction (ComBat, limma)
    4. Imputation (KNN, MICE, SVD)
    5. Feature selection (variance, differential expression)

    Each step must be:
    - Reproducible (same input → same output)
    - Auditable (parameters logged)
    - Reversible (can rollback to previous state)

Engineering Design:
    Pure Functions:
        - No side effects (don't modify inputs)
        - Deterministic (same input + params → same output)
        - Composable (chain transformations)
        - Testable (easy to unit test)

    Immutability Benefits:
        - Thread-safe parallel processing
        - Easy rollback to previous state
        - Clear data flow (no hidden state changes)
        - Facilitates debugging (inspect intermediate states)

Examples:
    >>> from cliquefinder.core.transform import Transform
    >>> from cliquefinder.core.biomatrix import BioMatrix
    >>>
    >>> class LogTransform(Transform):
    ...     def __init__(self, base: float = 2.0, pseudocount: float = 1.0):
    ...         super().__init__(
    ...             name="LogTransform",
    ...             params={"base": base, "pseudocount": pseudocount}
    ...         )
    ...         self.base = base
    ...         self.pseudocount = pseudocount
    ...
    ...     def apply(self, matrix: BioMatrix) -> BioMatrix:
    ...         import numpy as np
    ...         new_matrix = matrix.copy()
    ...         new_matrix.data = np.log(matrix.data + self.pseudocount) / np.log(self.base)
    ...         return new_matrix
    >>>
    >>> # Apply transformation (returns new matrix)
    >>> transform = LogTransform(base=2.0)
    >>> transformed = transform.apply(original_matrix)
    >>> # original_matrix is unchanged
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from cliquefinder.core.biomatrix import BioMatrix

__all__ = ['Transform']


class Transform(ABC):
    """
    Abstract base class for all matrix transformations.

    Design Philosophy:
        Pure functions - transformations take a matrix and parameters,
        return a new matrix. Input matrix is never modified.

    This enables:
        - Reproducibility: Same input + params → same output
        - Testing: Easy to unit test pure functions
        - Composition: Chain transformations into pipelines
        - Rollback: Keep original matrix, can always go back
        - Parallelism: Thread-safe (no shared mutable state)

    Attributes:
        name: Human-readable transformation name (e.g., "LogTransform")
        params: Dictionary of parameters used for this transformation
        timestamp: When this transform instance was created (for audit trail)

    Scientific Best Practices:
        1. Validate preconditions (check data properties before transform)
        2. Log parameters (essential for methods sections in papers)
        3. Preserve provenance (track what was done to the data)
        4. Return new instances (enable rollback and A/B testing)

    Examples:
        >>> class ZScoreNormalization(Transform):
        ...     def __init__(self, axis: int = 0):
        ...         super().__init__(
        ...             name="ZScoreNormalization",
        ...             params={"axis": axis}
        ...         )
        ...         self.axis = axis
        ...
        ...     def apply(self, matrix: BioMatrix) -> BioMatrix:
        ...         import numpy as np
        ...         new_matrix = matrix.copy()
        ...         mean = np.mean(matrix.data, axis=self.axis, keepdims=True)
        ...         std = np.std(matrix.data, axis=self.axis, keepdims=True)
        ...         new_matrix.data = (matrix.data - mean) / (std + 1e-8)
        ...         return new_matrix
        ...
        ...     def validate(self, matrix: BioMatrix) -> list[str]:
        ...         errors = super().validate(matrix)
        ...         if np.any(np.isnan(matrix.data)):
        ...             errors.append("Matrix contains NaN values - remove or impute first")
        ...         return errors
        >>>
        >>> transform = ZScoreNormalization(axis=0)
        >>> errors = transform.validate(matrix)
        >>> if not errors:
        ...     normalized = transform.apply(matrix)
    """

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        """
        Initialize transformation with name and parameters.

        Args:
            name: Human-readable transformation name (e.g., "LogTransform", "ComBatCorrection")
            params: Dictionary of parameters. Must be JSON-serializable for provenance tracking.
                   Example: {"method": "knn", "k": 5, "metric": "euclidean"}

        Note:
            Timestamp is automatically set to current time for audit trail.
        """
        self.name = name
        self.params = params
        self.timestamp = datetime.now()

    @abstractmethod
    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Execute transformation and return new matrix.

        Critical: Must never modify input matrix (immutability principle).
        Always call matrix.copy() to create new instance before modifications.

        Args:
            matrix: Input BioMatrix to transform

        Returns:
            New BioMatrix with transformation applied (input unchanged)

        Raises:
            ValueError: If transformation cannot be applied (check validate() first)

        Implementation Pattern:
            ```python
            def apply(self, matrix: BioMatrix) -> BioMatrix:
                # Create new instance (preserve immutability)
                new_matrix = matrix.copy()

                # Apply transformation to new_matrix.data
                new_matrix.data = transform_function(matrix.data, self.params)

                # Update quality flags if needed
                # e.g., mark imputed values: new_matrix.quality_flags[mask] |= QualityFlag.IMPUTED

                return new_matrix
            ```

        Scientific Considerations:
            - Document biological assumptions in docstring
            - Validate preconditions (call validate() first)
            - Update quality_flags to track changes
            - Preserve sample/feature metadata
            - Log warnings for edge cases
        """
        pass

    def validate(self, matrix: BioMatrix) -> list[str]:
        """
        Check preconditions before applying transformation.

        Call this before apply() to verify transformation is safe.
        Subclasses should override and call super().validate() first.

        Args:
            matrix: BioMatrix to validate

        Returns:
            List of error messages (empty list = valid, transformation can proceed)

        Examples:
            >>> transform = SomeTransform()
            >>> errors = transform.validate(matrix)
            >>> if errors:
            ...     print("Cannot apply transformation:")
            ...     for error in errors:
            ...         print(f"  - {error}")
            ... else:
            ...     result = transform.apply(matrix)

        Common Validations to Override:
            - Check for NaN/Inf values
            - Verify data range (e.g., log transform needs positive values)
            - Check minimum sample size
            - Validate metadata presence (e.g., batch correction needs batch labels)
            - Ensure appropriate data distribution
        """
        errors: list[str] = []

        if matrix.data.size == 0:
            errors.append("Cannot process empty matrix")

        return errors

    def __repr__(self) -> str:
        """
        String representation for logging and debugging.

        Returns:
            String like "LogTransform(base=2.0, pseudocount=1.0)"

        Examples:
            >>> transform = LogTransform(base=2.0, pseudocount=1.0)
            >>> print(transform)
            LogTransform(base=2.0, pseudocount=1.0)
            >>>
            >>> # Useful for logging pipelines
            >>> transforms = [LogTransform(), ZScoreNormalization(axis=0)]
            >>> print(" -> ".join(str(t) for t in transforms))
            LogTransform(base=2.0, pseudocount=1.0) -> ZScoreNormalization(axis=0)
        """
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
