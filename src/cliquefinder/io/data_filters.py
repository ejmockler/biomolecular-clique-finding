"""
Data filters for biomolecular datasets.

This module provides filters for removing technical artifacts and metadata rows
from expression matrices. Common in proteomics and other omics data where
feature IDs may contain quality control metrics.

Classes:
    MetadataRowFilter: Filter features based on ID patterns

Design Philosophy:
    - Study-specific artifacts are configurable
    - Generic pipelines can swap out filter configurations
    - Filters are composable and transparent
    - Filtering logic is separated from data loading

Example:
    >>> import pandas as pd
    >>> from cliquefinder.io.data_filters import MetadataRowFilter
    >>>
    >>> # Filter out proteomics QC metrics
    >>> filter = MetadataRowFilter(patterns=["nFragment", "nPeptide", "iRT_protein"])
    >>>
    >>> # Load data
    >>> df = pd.read_csv("proteomics_data.txt", sep='\t', index_col=0)
    >>> print(f"Before filtering: {len(df)} features")
    >>>
    >>> # Apply filter
    >>> filtered_ids = filter.filter(df.index)
    >>> df_filtered = df.loc[filtered_ids]
    >>> print(f"After filtering: {len(df_filtered)} features")
"""

from typing import Sequence

import pandas as pd


class MetadataRowFilter:
    """
    Filter features/rows based on ID patterns.

    Removes features whose IDs match specified patterns. Commonly used to
    exclude quality control metrics, technical artifacts, or metadata rows
    from expression matrices.

    Args:
        patterns: List of substring patterns to match for exclusion
        case_sensitive: Whether pattern matching is case-sensitive (default: False)

    Attributes:
        patterns: Configured exclusion patterns
        case_sensitive: Case sensitivity setting
        n_filtered_: Number of features filtered in last apply() call

    Example:
        >>> # Proteomics: Filter QC metrics
        >>> filter = MetadataRowFilter(["nFragment", "nPeptide", "iRT_protein"])
        >>> filtered_ids = filter.filter(feature_ids)
        >>>
        >>> # RNA-seq: Filter spike-in controls
        >>> filter = MetadataRowFilter(["ERCC-", "SIRV-"])
        >>> filtered_ids = filter.filter(feature_ids)
    """

    def __init__(
        self,
        patterns: Sequence[str],
        case_sensitive: bool = False,
    ):
        if not patterns:
            raise ValueError("At least one pattern must be provided")

        self.patterns = list(patterns)
        self.case_sensitive = case_sensitive
        self.n_filtered_: int | None = None

    def filter(self, feature_ids: pd.Index) -> pd.Index:
        """
        Filter feature IDs, excluding those matching configured patterns.

        Args:
            feature_ids: Feature identifiers to filter

        Returns:
            Filtered feature IDs (excluding matches)

        Example:
            >>> feature_ids = pd.Index(["PROTEIN1", "PROTEIN2", "nFragment", "nPeptide"])
            >>> filter = MetadataRowFilter(["nFragment", "nPeptide"])
            >>> filtered = filter.filter(feature_ids)
            >>> print(filtered)
            Index(['PROTEIN1', 'PROTEIN2'], dtype='object')
        """
        # Create boolean mask for rows to keep (NOT matching patterns)
        mask = pd.Series(True, index=feature_ids)

        # Combine all patterns into a single regex
        pattern = '|'.join(self.patterns)

        # Apply pattern matching
        matches = feature_ids.str.contains(
            pattern,
            case=self.case_sensitive,
            na=False,
            regex=False,  # Treat patterns as literal substrings, not regex
        )

        mask = ~matches  # Invert: keep rows that DON'T match

        # Track filtering statistics
        self.n_filtered_ = (~mask).sum()

        return feature_ids[mask]

    def get_filtered_ids(self, feature_ids: pd.Index) -> pd.Index:
        """
        Get feature IDs that would be EXCLUDED by this filter.

        Useful for quality control and validation.

        Args:
            feature_ids: Feature identifiers to check

        Returns:
            Feature IDs that match exclusion patterns

        Example:
            >>> feature_ids = pd.Index(["PROTEIN1", "nFragment", "nPeptide"])
            >>> filter = MetadataRowFilter(["nFragment", "nPeptide"])
            >>> excluded = filter.get_filtered_ids(feature_ids)
            >>> print(excluded)
            Index(['nFragment', 'nPeptide'], dtype='object')
        """
        # Combine all patterns into a single regex
        pattern = '|'.join(self.patterns)

        # Apply pattern matching
        matches = feature_ids.str.contains(
            pattern,
            case=self.case_sensitive,
            na=False,
            regex=False,
        )

        return feature_ids[matches]

    def __repr__(self) -> str:
        """String representation of filter configuration."""
        return (
            f"MetadataRowFilter(patterns={self.patterns}, "
            f"case_sensitive={self.case_sensitive})"
        )


class RegexMetadataRowFilter(MetadataRowFilter):
    """
    Filter features using regex patterns instead of literal substrings.

    Extends MetadataRowFilter to support full regex syntax for more complex
    pattern matching scenarios.

    Args:
        patterns: List of regex patterns for exclusion
        case_sensitive: Whether pattern matching is case-sensitive (default: False)

    Example:
        >>> # Filter features starting with "iRT" or ending with "_QC"
        >>> filter = RegexMetadataRowFilter([r"^iRT", r"_QC$"])
        >>> filtered_ids = filter.filter(feature_ids)
    """

    def filter(self, feature_ids: pd.Index) -> pd.Index:
        """
        Filter feature IDs using regex patterns.

        Args:
            feature_ids: Feature identifiers to filter

        Returns:
            Filtered feature IDs (excluding matches)
        """
        # Create boolean mask for rows to keep (NOT matching patterns)
        mask = pd.Series(True, index=feature_ids)

        # Combine all patterns into a single regex
        pattern = '|'.join(self.patterns)

        # Apply regex pattern matching
        matches = feature_ids.str.contains(
            pattern,
            case=self.case_sensitive,
            na=False,
            regex=True,  # Use full regex syntax
        )

        mask = ~matches  # Invert: keep rows that DON'T match

        # Track filtering statistics
        self.n_filtered_ = (~mask).sum()

        return feature_ids[mask]

    def get_filtered_ids(self, feature_ids: pd.Index) -> pd.Index:
        """
        Get feature IDs that would be EXCLUDED by regex patterns.

        Args:
            feature_ids: Feature identifiers to check

        Returns:
            Feature IDs that match exclusion patterns
        """
        # Combine all patterns into a single regex
        pattern = '|'.join(self.patterns)

        # Apply regex pattern matching
        matches = feature_ids.str.contains(
            pattern,
            case=self.case_sensitive,
            na=False,
            regex=True,
        )

        return feature_ids[matches]

    def __repr__(self) -> str:
        """String representation of filter configuration."""
        return (
            f"RegexMetadataRowFilter(patterns={self.patterns}, "
            f"case_sensitive={self.case_sensitive})"
        )
