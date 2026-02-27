"""Utility modules for biomolecular data processing."""

from cliquefinder.utils.correlation_matrix import (
    get_correlation_matrix,
    clear_cache,
    get_cache_info,
    compute_correlation_matrix_chunked,
    weighted_pearson_correlation,
    compute_weighted_correlation_matrix_chunked,
    get_weighted_correlation_matrix
)
from cliquefinder.utils.fileio import (
    atomic_write_json,
    atomic_write_text,
)
from cliquefinder.utils.statistics import (
    otsu_threshold,
    cohens_d,
)

__all__ = [
    # Correlation matrix utilities
    'get_correlation_matrix',
    'clear_cache',
    'get_cache_info',
    'compute_correlation_matrix_chunked',
    # Weighted correlation utilities
    'weighted_pearson_correlation',
    'compute_weighted_correlation_matrix_chunked',
    'get_weighted_correlation_matrix',
    # Atomic file-write utilities
    'atomic_write_json',
    'atomic_write_text',
    # Statistical utilities
    'otsu_threshold',
    'cohens_d',
]
