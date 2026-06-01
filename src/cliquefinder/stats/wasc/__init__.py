"""WASC — Within-cluster Anchor-Slope Concordance.

Per-edge inverse-variance-weighted Cochran-Q invariance test on partial
regression slopes within the 8 pre-registered C9-ALS cluster terms across
{C9, SPOR, CTRL} donor groups.

See memory/wasc_spec.md and memory/wasc_build_plan.md.

M1 scope (current): edge enumeration only. fit / null / FDR / STRING
control / three-contrast decomposition come in M2-M5.
"""
from __future__ import annotations

from .edges import (
    DEFAULT_CLUSTER_TERMS,
    compute_measured_cluster_members,
    enumerate_wasc_indra_edges,
)
from .types import Network, Theme, WascEdge

__all__ = [
    # M1 — edges
    "DEFAULT_CLUSTER_TERMS",
    "compute_measured_cluster_members",
    "enumerate_wasc_indra_edges",
    # types
    "Network",
    "Theme",
    "WascEdge",
]
