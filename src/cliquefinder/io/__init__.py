"""
I/O module for loading and writing biomolecular data.

This module provides functions for reading expression matrices from various file formats
and writing processed results back to disk. Maintains scientific reproducibility through
transparent handling of data provenance and quality metadata.

Key Functions:
    - load_csv_matrix: Load expression matrix from CSV
    - write_csv_matrix: Write BioMatrix to CSV (data + quality flags)
    - write_sample_metadata: Export sample annotations

Design Philosophy:
    - Robust error handling for malformed data
    - Clear validation messages for data quality issues
    - Efficient handling of large files (90MB+)
    - Biological metadata parsing from sample IDs
    - Quality flag initialization for downstream tracking

Supported Formats:
    - CSV: Universal format for expression matrices
    - Future: HDF5, AnnData, Parquet for large-scale data

Examples:
    >>> from cliquefinder.io import load_csv_matrix, write_csv_matrix
    >>> from pathlib import Path
    >>>
    >>> # Load data
    >>> matrix = load_csv_matrix(Path("raw_data.csv"))
    >>> print(f"Loaded {matrix.n_features} features x {matrix.n_samples} samples")
    >>>
    >>> # Process data (apply transforms...)
    >>> processed = some_transform.apply(matrix)
    >>>
    >>> # Write results
    >>> write_csv_matrix(processed, Path("processed_data"))
"""

from cliquefinder.io.loaders import load_csv_matrix
from cliquefinder.io.writers import write_csv_matrix, write_sample_metadata
from cliquefinder.io.metadata import (
    ClinicalMetadataEnricher,
    SubjectIdExtractor,
    CLINICAL_COLUMNS,
    get_column_groups,
    list_available_columns,
)

__all__ = [
    'load_csv_matrix',
    'write_csv_matrix',
    'write_sample_metadata',
    'ClinicalMetadataEnricher',
    'SubjectIdExtractor',
    'CLINICAL_COLUMNS',
    'get_column_groups',
    'list_available_columns',
]
