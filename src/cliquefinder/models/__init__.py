"""
Statistical models for proteomics data analysis.

This module provides high-performance vectorized computation for proteomics:
- Vectorized residual computation for outlier detection
- Model diagnostics and validation
"""

from cliquefinder.models.vectorized_residuals import (
    VectorizedResidualComputer,
    ResidualDiagnostics,
    detect_outliers_vectorized,
)

__all__ = [
    'VectorizedResidualComputer',
    'ResidualDiagnostics',
    'detect_outliers_vectorized',
]
