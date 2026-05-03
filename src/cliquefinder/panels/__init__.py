"""Multi-seed gradient panels — design and analysis.

A panel is a stratified collection of degree-matched control seeds
run alongside one target seed under a single contrast.  The design
is a frozen, YAML-serializable manifest; the analysis applies
BH-FDR + Bonferroni and a stratum-vs-stratum Mann–Whitney U.

Public API
----------

>>> from cliquefinder.panels import (
...     PanelDesign, PanelStratum, PanelResult, PanelAnalysis,
...     select_panel, analyze_panel,
... )
"""
from __future__ import annotations

from .analysis import (
    AdjustedSeedResult,
    FailedSeed,
    PanelAnalysis,
    PanelResult,
    PerSeedResult,
    ShellSummary,
    StratumComparison,
    TARGET_STRATUM_LABEL,
    TargetPosition,
    analyze_panel,
)
from .design import PanelDesign, PanelStratum
from .runner import run_panel
from .seed_runner import GroupResolver, run_seed_gradient
from .selection import select_panel

__all__ = [
    "AdjustedSeedResult",
    "FailedSeed",
    "GroupResolver",
    "PanelAnalysis",
    "PanelDesign",
    "PanelResult",
    "PanelStratum",
    "PerSeedResult",
    "ShellSummary",
    "StratumComparison",
    "TARGET_STRATUM_LABEL",
    "TargetPosition",
    "analyze_panel",
    "run_panel",
    "run_seed_gradient",
    "select_panel",
]
