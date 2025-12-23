"""
Core data structure for biomolecular expression matrices.

BioMatrix unifies numerical data (counts/intensities) with biological metadata
(sample phenotypes, gene annotations) and quality provenance (which values were
imputed, batch corrected, etc.).

Biological Context:
    Expression matrices are the fundamental data structure in genomics/proteomics:
    - Rows = features (genes, proteins, transcripts)
    - Columns = samples (patients, cell lines, time points)
    - Values = measurements (counts, intensities, abundances)

    Unlike generic dataframes, expression matrices require:
    - Tight coupling between data and biological metadata
    - Quality tracking for scientific reproducibility
    - Efficient subsetting preserving all annotations
    - Immutability for pipeline composition

Engineering Design:
    - Immutable: Operations return new instances (functional style)
    - Type-safe: NumPy arrays for data, Pandas for metadata
    - Memory-efficient: Share arrays where possible via copy()
    - Validated: Constructor checks shape consistency

Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cliquefinder.core.biomatrix import BioMatrix
    >>> from cliquefinder.core.quality import QualityFlag
    >>>
    >>> # Create a simple matrix
    >>> data = np.array([[10, 20], [30, 40]])
    >>> feature_ids = pd.Index(["ENSG001", "ENSG002"])
    >>> sample_ids = pd.Index(["CTRL_001", "CASE_002"])
    >>> sample_metadata = pd.DataFrame({
    ...     'phenotype': ['CTRL', 'CASE']
    ... }, index=sample_ids)
    >>> quality_flags = np.full((2, 2), QualityFlag.ORIGINAL, dtype=int)
    >>>
    >>> matrix = BioMatrix(
    ...     data=data,
    ...     feature_ids=feature_ids,
    ...     sample_ids=sample_ids,
    ...     sample_metadata=sample_metadata,
    ...     quality_flags=quality_flags
    ... )
    >>>
    >>> # Subset samples
    >>> case_matrix = matrix.select_samples(matrix.sample_metadata['phenotype'] == 'CASE')
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from cliquefinder.core.quality import QualityFlag

__all__ = ['BioMatrix']


class BioMatrix:
    """
    Immutable container for expression matrix + biological metadata + quality flags.

    The fundamental data structure for genomic/proteomic analysis. Maintains tight
    coupling between numerical measurements and their biological/technical context.

    Attributes:
        data: Numerical expression matrix (features × samples)
        feature_ids: Row identifiers (e.g., Ensembl gene IDs)
        sample_ids: Column identifiers (e.g., patient/sample IDs)
        sample_metadata: Biological annotations (phenotype, cohort, etc.)
        quality_flags: Per-value quality tracking (outliers, imputation, etc.)

    Shape Invariants:
        - data.shape[0] == len(feature_ids)
        - data.shape[1] == len(sample_ids)
        - quality_flags.shape == data.shape
        - sample_metadata.index equals sample_ids

    Design Principles:
        1. Immutability: All operations return new instances
        2. Validation: Constructor ensures consistency
        3. Composability: Works seamlessly with Transform pipeline
        4. Memory efficiency: Uses views/references where safe
    """

    def __init__(
        self,
        data: np.ndarray,
        feature_ids: pd.Index,
        sample_ids: pd.Index,
        sample_metadata: pd.DataFrame,
        quality_flags: np.ndarray,
    ):
        """
        Initialize BioMatrix with validation.

        Args:
            data: Expression matrix (features × samples)
            feature_ids: Row identifiers (genes, proteins, etc.)
            sample_ids: Column identifiers (samples, patients, etc.)
            sample_metadata: DataFrame with biological annotations
                Must have index matching sample_ids
            quality_flags: Quality tracking matrix (same shape as data)
                Each value is a QualityFlag (int)

        Raises:
            ValueError: If shapes are inconsistent or indices don't match
            TypeError: If data types are incorrect

        Examples:
            >>> data = np.array([[1, 2], [3, 4]])
            >>> feature_ids = pd.Index(["gene1", "gene2"])
            >>> sample_ids = pd.Index(["sample1", "sample2"])
            >>> sample_metadata = pd.DataFrame(index=sample_ids)
            >>> quality_flags = np.zeros((2, 2), dtype=int)
            >>> matrix = BioMatrix(data, feature_ids, sample_ids, sample_metadata, quality_flags)
        """
        # Type validation
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be np.ndarray, got {type(data)}")
        if not isinstance(feature_ids, pd.Index):
            raise TypeError(f"feature_ids must be pd.Index, got {type(feature_ids)}")
        if not isinstance(sample_ids, pd.Index):
            raise TypeError(f"sample_ids must be pd.Index, got {type(sample_ids)}")
        if not isinstance(sample_metadata, pd.DataFrame):
            raise TypeError(f"sample_metadata must be pd.DataFrame, got {type(sample_metadata)}")
        if not isinstance(quality_flags, np.ndarray):
            raise TypeError(f"quality_flags must be np.ndarray, got {type(quality_flags)}")

        # Shape validation
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {data.shape}")
        if quality_flags.ndim != 2:
            raise ValueError(f"quality_flags must be 2D, got shape {quality_flags.shape}")

        n_features, n_samples = data.shape

        if len(feature_ids) != n_features:
            raise ValueError(
                f"feature_ids length ({len(feature_ids)}) must match data rows ({n_features})"
            )
        if len(sample_ids) != n_samples:
            raise ValueError(
                f"sample_ids length ({len(sample_ids)}) must match data columns ({n_samples})"
            )
        if quality_flags.shape != data.shape:
            raise ValueError(
                f"quality_flags shape {quality_flags.shape} must match data shape {data.shape}"
            )

        # Index validation
        if not sample_metadata.index.equals(sample_ids):
            raise ValueError(
                "sample_metadata.index must match sample_ids exactly. "
                f"Got {len(sample_metadata.index)} metadata rows for {len(sample_ids)} samples."
            )

        # Store as private attributes (immutability by convention)
        self._data = data
        self._feature_ids = feature_ids
        self._sample_ids = sample_ids
        self._sample_metadata = sample_metadata
        self._quality_flags = quality_flags

    @property
    def data(self) -> np.ndarray:
        """Expression matrix (features × samples)."""
        return self._data

    @property
    def feature_ids(self) -> pd.Index:
        """Row identifiers (genes, proteins, etc.)."""
        return self._feature_ids

    @property
    def sample_ids(self) -> pd.Index:
        """Column identifiers (samples, patients, etc.)."""
        return self._sample_ids

    @property
    def sample_metadata(self) -> pd.DataFrame:
        """Biological annotations for samples."""
        return self._sample_metadata

    @property
    def quality_flags(self) -> np.ndarray:
        """Quality tracking matrix (same shape as data)."""
        return self._quality_flags

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions (n_features, n_samples)."""
        return self._data.shape

    @property
    def n_features(self) -> int:
        """Number of features (genes, proteins, etc.)."""
        return self._data.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self._data.shape[1]

    def select_samples(self, mask: np.ndarray | pd.Series) -> BioMatrix:
        """
        Subset matrix by samples (columns).

        Returns new BioMatrix with selected samples, preserving all metadata.

        Args:
            mask: Boolean array/Series indicating which samples to keep
                If Series, uses values and ignores index

        Returns:
            New BioMatrix with selected samples

        Raises:
            ValueError: If mask length doesn't match n_samples

        Examples:
            >>> # Select CASE samples
            >>> case_mask = matrix.sample_metadata['phenotype'] == 'CASE'
            >>> case_matrix = matrix.select_samples(case_mask)
            >>>
            >>> # Select first 10 samples
            >>> subset = matrix.select_samples(np.arange(10) < matrix.n_samples)
        """
        # Convert Series to array
        if isinstance(mask, pd.Series):
            mask = mask.values

        # Validate
        if len(mask) != self.n_samples:
            raise ValueError(
                f"mask length ({len(mask)}) must match n_samples ({self.n_samples})"
            )

        # Subset all components
        return BioMatrix(
            data=self._data[:, mask],
            feature_ids=self._feature_ids,
            sample_ids=self._sample_ids[mask],
            sample_metadata=self._sample_metadata.loc[self._sample_ids[mask]],
            quality_flags=self._quality_flags[:, mask],
        )

    def select_features(self, mask: np.ndarray | pd.Series) -> BioMatrix:
        """
        Subset matrix by features (rows).

        Returns new BioMatrix with selected features, preserving all metadata.

        Args:
            mask: Boolean array/Series indicating which features to keep
                If Series, uses values and ignores index

        Returns:
            New BioMatrix with selected features

        Raises:
            ValueError: If mask length doesn't match n_features

        Examples:
            >>> # Select protein-coding genes
            >>> coding_mask = matrix.feature_ids.str.startswith('ENSG')
            >>> coding_matrix = matrix.select_features(coding_mask)
            >>>
            >>> # Select high-variance genes
            >>> variances = np.var(matrix.data, axis=1)
            >>> high_var_mask = variances > np.percentile(variances, 90)
            >>> variable_matrix = matrix.select_features(high_var_mask)
        """
        # Convert Series to array
        if isinstance(mask, pd.Series):
            mask = mask.values

        # Validate
        if len(mask) != self.n_features:
            raise ValueError(
                f"mask length ({len(mask)}) must match n_features ({self.n_features})"
            )

        # Subset all components
        return BioMatrix(
            data=self._data[mask, :],
            feature_ids=self._feature_ids[mask],
            sample_ids=self._sample_ids,
            sample_metadata=self._sample_metadata,
            quality_flags=self._quality_flags[mask, :],
        )

    def copy(self, deep: bool = True) -> BioMatrix:
        """
        Create a copy of this matrix.

        Args:
            deep: If True, copy all arrays. If False, share arrays (faster but mutable)

        Returns:
            New BioMatrix instance

        Examples:
            >>> # Deep copy for mutation
            >>> modified = matrix.copy(deep=True)
            >>>
            >>> # Shallow copy for performance
            >>> view = matrix.copy(deep=False)
        """
        if deep:
            return BioMatrix(
                data=self._data.copy(),
                feature_ids=self._feature_ids.copy(),
                sample_ids=self._sample_ids.copy(),
                sample_metadata=self._sample_metadata.copy(),
                quality_flags=self._quality_flags.copy(),
            )
        else:
            return BioMatrix(
                data=self._data,
                feature_ids=self._feature_ids,
                sample_ids=self._sample_ids,
                sample_metadata=self._sample_metadata,
                quality_flags=self._quality_flags,
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BioMatrix({self.n_features} features × {self.n_samples} samples)\n"
            f"  Features: {self.feature_ids[0]}...{self.feature_ids[-1]}\n"
            f"  Samples: {self.sample_ids[0]}...{self.sample_ids[-1]}\n"
            f"  Metadata columns: {list(self.sample_metadata.columns)}"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
