"""
Statistical method implementations for clique differential testing.

This package provides four method adapters that conform to the
``CliqueTestMethod`` protocol defined in ``method_comparison_types``:

* :class:`OLSMethod`  -- fixed-effects OLS
* :class:`LMMMethod`  -- linear mixed model with random subject effects
* :class:`ROASTMethod` -- rotation-based gene-set test (ROAST)
* :class:`PermutationMethod` -- competitive permutation test

All four classes are re-exported from ``method_comparison`` for backward
compatibility.
"""

from __future__ import annotations

from .ols import OLSMethod
from .lmm import LMMMethod
from .roast import ROASTMethod
from .permutation import PermutationMethod

__all__ = [
    "OLSMethod",
    "LMMMethod",
    "ROASTMethod",
    "PermutationMethod",
]
