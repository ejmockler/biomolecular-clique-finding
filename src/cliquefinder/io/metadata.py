"""
Clinical metadata enrichment for expression matrices.

This module provides principled integration of external clinical metadata with
BioMatrix sample annotations. Handles the common bioinformatics challenge of
linking expression data (keyed by sample IDs) to clinical registries (keyed
by subject IDs).

Biological Context:
    Expression data and clinical metadata often use different identifiers:
    - Expression sample ID: CASE-NEUVM674HUA-5257-T_P003 (sample-level)
    - Clinical subject ID: NEUVM674HUA (subject-level, may have multiple samples)

    This module bridges that gap through:
    1. Subject ID extraction from sample IDs (configurable patterns)
    2. Left-join enrichment (preserves all samples, adds clinical columns)
    3. Explicit handling of missing metadata (flagged, not silently dropped)

Engineering Design:
    - Declarative column selection (explicit > implicit)
    - Immutable operations (returns new BioMatrix)
    - Comprehensive logging (match rates, missing data)
    - Type-safe with validation
    - Compatible with Transform pipeline

Clinical Variables:
    The module distinguishes between variable types:
    - Categorical: Sex, Subject Group, Site of Motor Onset
    - Ordinal: El Escorial Criteria, C9orf72 status
    - Continuous: Age at Onset, Disease Duration
    - Genetic: C9orf72 expansion, ATXN2 status

    Each type may require different downstream handling (one-hot encoding,
    normalization, etc.).

Examples:
    >>> from cliquefinder.io.metadata import ClinicalMetadataEnricher
    >>> from cliquefinder.io.loaders import load_csv_matrix
    >>>
    >>> # Load expression data
    >>> matrix = load_csv_matrix("expression.csv")
    >>>
    >>> # Enrich with clinical metadata
    >>> enricher = ClinicalMetadataEnricher.from_csv("clinical.csv")
    >>> enriched = enricher.enrich(matrix)
    >>>
    >>> # Check enrichment summary
    >>> print(enricher.summary())
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Union, List
from dataclasses import dataclass, field
import warnings
import re
import numpy as np
import pandas as pd

from cliquefinder.core.biomatrix import BioMatrix

__all__ = ['ClinicalMetadataEnricher', 'SubjectIdExtractor', 'CLINICAL_COLUMNS']


# Curated clinical columns for ALS studies
# Organized by biological/clinical relevance
CLINICAL_COLUMNS = {
    # Demographics
    'demographics': [
        'Sex',
        'Ethnicity',
    ],
    # Disease classification
    'disease': [
        'Subject Group',
        'Subject Group Subcategory',
        'Revised El Escorial Criteria',
    ],
    # Disease progression
    'progression': [
        'Site of Motor Onset',
        'Age at Symptom Onset',
        'Age at Death',
        'Disease Duration in Months',
        'Age at Sample Collection (Biopsy/Blood)',
    ],
    # Genetic factors
    'genetic': [
        'C9orf72 Repeat Expansion (Data from CUMC)',
        'ATXN2 Repeat Expansion (Data from CUMC)',
        'C9 repeat size',
        'Expanded, >=30',
        'ATXN2 repeat size',
        'Intermediate, 30-33',
    ],
    # Sample information
    'sample': [
        'Sample Tissue Source',
        'ExternalSampleId',
    ],
    # Ancestry (for population stratification)
    'ancestry': [
        'pct_european',
        'pct_african',
        'pct_east_asian',
        'pct_south_asian',
        'pct_americas',
    ],
}

# Flatten for convenience
ALL_CLINICAL_COLUMNS = [col for cols in CLINICAL_COLUMNS.values() for col in cols]


@dataclass
class SubjectIdExtractor:
    """
    Extract subject IDs from sample IDs using configurable patterns.

    Sample ID formats vary by study. This class encapsulates the extraction
    logic and provides common patterns.

    Default pattern for this study:
        CASE-NEUVM674HUA-5257-T_P003 -> NEUVM674HUA
        Component index 1 (0-indexed), split by '-'

    Supports both regex-based and delimiter-based extraction:
        - If pattern is provided, uses regex to extract participant ID
        - Falls back to delimiter-based extraction if pattern doesn't match
        - Default pattern r'(NEU[A-Z0-9]+)' handles both RNA and proteomics formats

    Attributes:
        delimiter: Character(s) to split sample ID on
        component_index: Which component contains subject ID (0-indexed)
        pattern: Optional regex pattern for extraction (default: r'(NEU[A-Z0-9]+)')

    Examples:
        >>> extractor = SubjectIdExtractor(delimiter='-', component_index=1)
        >>> extractor.extract('CASE-NEUVM674HUA-5257-T_P003')
        'NEUVM674HUA'
        >>> extractor.extract('CASE_NEUAA295HHE-9014-P_D3')
        'NEUAA295HHE'
    """
    delimiter: str = '-'
    component_index: int = 1
    pattern: Optional[str] = r'(NEU[A-Z0-9]+)'

    def __post_init__(self):
        """Compile regex pattern if provided."""
        self._compiled_pattern = re.compile(self.pattern) if self.pattern else None

    def extract(self, sample_id: str) -> Optional[str]:
        """
        Extract subject/participant ID from sample ID.

        Tries regex pattern first if available, falls back to delimiter-based extraction.

        Args:
            sample_id: Sample identifier string

        Returns:
            Extracted subject ID or None if extraction fails
        """
        sid = str(sample_id)

        # Try regex pattern first if available
        if self._compiled_pattern:
            match = self._compiled_pattern.search(sid)
            if match:
                return match.group(1)

        # Fall back to delimiter-based extraction
        parts = sid.split(self.delimiter)
        if len(parts) > self.component_index:
            return parts[self.component_index]
        return None

    def extract_series(self, sample_ids: Union[pd.Index, pd.Series, List[str]]) -> pd.Series:
        """
        Extract subject IDs from multiple sample IDs.

        Args:
            sample_ids: Collection of sample identifiers (Index, Series, or List)

        Returns:
            Series of extracted subject IDs with original index preserved
        """
        if isinstance(sample_ids, pd.Index):
            sample_ids = sample_ids.to_series()
        elif isinstance(sample_ids, list):
            sample_ids = pd.Series(sample_ids)
        return sample_ids.apply(self.extract)


@dataclass
class EnrichmentSummary:
    """Summary of metadata enrichment operation."""
    n_samples_input: int
    n_samples_output: int
    n_matched: int
    n_unmatched: int
    n_dropped: int
    match_rate: float
    columns_added: list[str]
    unmatched_subjects: list[str]

    def __repr__(self) -> str:
        dropped_str = f", dropped: {self.n_dropped}" if self.n_dropped > 0 else ""
        return (
            f"EnrichmentSummary(\n"
            f"  input_samples: {self.n_samples_input}\n"
            f"  output_samples: {self.n_samples_output}\n"
            f"  matched: {self.n_matched} ({100*self.match_rate:.1f}%){dropped_str}\n"
            f"  columns: {len(self.columns_added)}\n"
            f")"
        )


class ClinicalMetadataEnricher:
    """
    Enrich BioMatrix sample metadata with external clinical data.

    This class handles the complete workflow of:
    1. Loading clinical metadata from CSV/DataFrame
    2. Extracting subject IDs from sample IDs
    3. Joining clinical data to expression samples
    4. Tracking match statistics

    The enrichment is a left join: all expression samples are preserved,
    and clinical columns are added where matches exist (NaN otherwise).

    Attributes:
        clinical_df: Clinical metadata DataFrame
        subject_col: Column name containing subject IDs
        extractor: SubjectIdExtractor for sample ID parsing
        columns: Clinical columns to include (None = all available)

    Examples:
        >>> # From CSV file
        >>> enricher = ClinicalMetadataEnricher.from_csv(
        ...     "clinical.csv",
        ...     subject_col="ExternalSubjectId"
        ... )
        >>> enriched = enricher.enrich(matrix)
        >>>
        >>> # With custom column selection
        >>> enricher = ClinicalMetadataEnricher.from_csv(
        ...     "clinical.csv",
        ...     columns=['Sex', 'Age at Symptom Onset', 'C9orf72 Repeat Expansion']
        ... )
        >>>
        >>> # With column groups
        >>> enricher = ClinicalMetadataEnricher.from_csv(
        ...     "clinical.csv",
        ...     column_groups=['demographics', 'disease', 'genetic']
        ... )
    """

    def __init__(
        self,
        clinical_df: pd.DataFrame,
        subject_col: str = 'ExternalSubjectId',
        extractor: Optional[SubjectIdExtractor] = None,
        columns: Optional[list[str]] = None,
        column_groups: Optional[list[str]] = None,
    ):
        """
        Initialize enricher with clinical data.

        Args:
            clinical_df: Clinical metadata DataFrame
            subject_col: Column containing subject IDs
            extractor: Subject ID extractor (default: standard ALS format)
            columns: Specific columns to include
            column_groups: Column groups to include (from CLINICAL_COLUMNS)
        """
        if subject_col not in clinical_df.columns:
            raise ValueError(f"Subject column '{subject_col}' not in clinical data")

        self.clinical_df = clinical_df.copy()
        self.subject_col = subject_col
        self.extractor = extractor or SubjectIdExtractor()

        # Determine columns to use
        if columns is not None:
            self.columns = [c for c in columns if c in clinical_df.columns]
            missing = [c for c in columns if c not in clinical_df.columns]
            if missing:
                warnings.warn(f"Requested columns not in data: {missing}")
        elif column_groups is not None:
            self.columns = []
            for group in column_groups:
                if group in CLINICAL_COLUMNS:
                    self.columns.extend([
                        c for c in CLINICAL_COLUMNS[group]
                        if c in clinical_df.columns
                    ])
                else:
                    warnings.warn(f"Unknown column group: {group}")
            self.columns = list(dict.fromkeys(self.columns))  # Remove duplicates
        else:
            # Default: all available curated columns
            self.columns = [c for c in ALL_CLINICAL_COLUMNS if c in clinical_df.columns]

        self._summary: Optional[EnrichmentSummary] = None

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        subject_col: str = 'ExternalSubjectId',
        extractor: Optional[SubjectIdExtractor] = None,
        columns: Optional[list[str]] = None,
        column_groups: Optional[list[str]] = None,
    ) -> 'ClinicalMetadataEnricher':
        """
        Create enricher from CSV file.

        Args:
            path: Path to clinical metadata CSV
            subject_col: Column containing subject IDs
            extractor: Subject ID extractor
            columns: Specific columns to include
            column_groups: Column groups to include

        Returns:
            Configured ClinicalMetadataEnricher
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Clinical metadata not found: {path}")

        df = pd.read_csv(path)
        return cls(
            clinical_df=df,
            subject_col=subject_col,
            extractor=extractor,
            columns=columns,
            column_groups=column_groups,
        )

    def enrich(self, matrix: BioMatrix, drop_unmatched: bool = False) -> BioMatrix:
        """
        Enrich BioMatrix with clinical metadata.

        Performs left join: clinical data added where subject IDs match.
        Optionally drops samples without clinical metadata.

        Args:
            matrix: Input BioMatrix
            drop_unmatched: If True, remove samples without clinical metadata.
                           If False (default), keep all samples (NaN for missing).

        Returns:
            New BioMatrix with enriched sample_metadata
        """
        # Extract subject IDs from sample IDs
        subject_ids = self.extractor.extract_series(matrix.sample_ids)

        # Build lookup from subject ID to clinical data
        clinical_indexed = self.clinical_df.set_index(self.subject_col)[self.columns]

        # Handle duplicate subjects (take first occurrence)
        if clinical_indexed.index.duplicated().any():
            n_dup = clinical_indexed.index.duplicated().sum()
            warnings.warn(f"Clinical data has {n_dup} duplicate subjects, using first")
            clinical_indexed = clinical_indexed[~clinical_indexed.index.duplicated(keep='first')]

        # Compute match mask before any filtering
        matched_mask = subject_ids.isin(clinical_indexed.index)
        n_matched = matched_mask.sum()
        n_unmatched = (~matched_mask).sum()
        unmatched_subjects = subject_ids[~matched_mask].unique().tolist()

        # Filter matrix if dropping unmatched
        if drop_unmatched and n_unmatched > 0:
            # Use BioMatrix.select_samples to properly filter all components
            keep_mask = matched_mask.values
            matrix = matrix.select_samples(keep_mask)
            subject_ids = subject_ids[keep_mask]
            n_dropped = n_unmatched
        else:
            n_dropped = 0

        # Join: map subject_ids to clinical data
        enriched_data = subject_ids.map(
            lambda sid: clinical_indexed.loc[sid] if sid in clinical_indexed.index else pd.Series(dtype=float)
        )

        # Convert to DataFrame
        enriched_df = pd.DataFrame(
            [row if isinstance(row, pd.Series) else pd.Series(dtype=float) for row in enriched_data],
            index=matrix.sample_ids
        )

        # Add subject_id column
        enriched_df['subject_id'] = subject_ids.values

        # Combine with existing metadata
        new_metadata = matrix.sample_metadata.copy()
        for col in enriched_df.columns:
            if col not in new_metadata.columns:
                new_metadata[col] = enriched_df[col].values

        self._summary = EnrichmentSummary(
            n_samples_input=n_matched + n_unmatched,  # Original count
            n_samples_output=len(matrix.sample_ids),
            n_matched=n_matched,
            n_unmatched=n_unmatched,
            n_dropped=n_dropped,
            match_rate=n_matched / (n_matched + n_unmatched) if (n_matched + n_unmatched) > 0 else 0,
            columns_added=list(enriched_df.columns),
            unmatched_subjects=unmatched_subjects[:20],  # Limit for display
        )

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=new_metadata,
            quality_flags=matrix.quality_flags,
        )

    @property
    def summary(self) -> Optional[EnrichmentSummary]:
        """Get enrichment summary (available after enrich() called)."""
        return self._summary

    def __repr__(self) -> str:
        return (
            f"ClinicalMetadataEnricher(\n"
            f"  clinical_samples: {len(self.clinical_df)}\n"
            f"  subject_col: {self.subject_col}\n"
            f"  columns: {len(self.columns)}\n"
            f")"
        )


def get_column_groups() -> dict[str, list[str]]:
    """Return available clinical column groups."""
    return CLINICAL_COLUMNS.copy()


def list_available_columns(clinical_path: Path | str) -> dict[str, list[str]]:
    """
    List available clinical columns organized by group.

    Args:
        clinical_path: Path to clinical CSV

    Returns:
        Dict mapping group name to available columns
    """
    df = pd.read_csv(clinical_path)
    available = {}

    for group, cols in CLINICAL_COLUMNS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            available[group] = present

    return available
