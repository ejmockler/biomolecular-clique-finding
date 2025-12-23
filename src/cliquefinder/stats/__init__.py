"""
Statistical testing module for correlation-based regulatory analysis.

Exports core functions for:
- Differential correlation testing (CASE vs CTRL)
- Multiple testing correction (FDR)
- Confidence intervals and significance thresholds
"""

from .correlation_tests import (
    CorrelationTestResult,
    fisher_z_transform,
    inverse_fisher_z,
    fisher_z_standard_error,
    correlation_confidence_interval,
    test_correlation_difference,
    compute_significance_threshold,
    estimate_effective_tests,
    permutation_fdr,
    apply_fdr_correction,
)

__all__ = [
    "CorrelationTestResult",
    "fisher_z_transform",
    "inverse_fisher_z",
    "fisher_z_standard_error",
    "correlation_confidence_interval",
    "test_correlation_difference",
    "compute_significance_threshold",
    "estimate_effective_tests",
    "permutation_fdr",
    "apply_fdr_correction",
]
