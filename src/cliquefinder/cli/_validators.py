"""Shared argparse type validators for CLI parameter bounds checking.

These validators produce clear error messages when users pass invalid
values (e.g., ``--alpha 2.0``, ``--n-rotations -5``).  They are
intended to be used as the ``type=`` argument in ``add_argument()``.
"""

from __future__ import annotations

import argparse


def _positive_int(value: str) -> int:
    """argparse type for positive integers (> 0)."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def _probability(value: str) -> float:
    """argparse type for values in the open interval (0, 1)."""
    fvalue = float(value)
    if not (0 < fvalue < 1):
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid probability (must be in (0, 1))"
        )
    return fvalue


def _positive_float(value: str) -> float:
    """argparse type for positive floats (> 0)."""
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    return fvalue


def _percentile(value: str) -> float:
    """argparse type for percentile values in the open interval (0, 100)."""
    fvalue = float(value)
    if not (0 < fvalue < 100):
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid percentile (must be in (0, 100))"
        )
    return fvalue
