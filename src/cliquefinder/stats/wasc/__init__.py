"""WASC — Within-cluster Anchor-Slope Concordance.

Per-edge inverse-variance-weighted Cochran-Q invariance test on partial
regression slopes within eight discovery-derived C9-ALS cluster terms that
were fixed before WASC, across {C9, SPOR, CTRL} donor groups.

See memory/wasc_spec.md and memory/wasc_build_plan.md.

Implemented scope: M1 edge enumeration; M2 preprocessing, FWL fits,
Cochran-Q concordance, and matched substitute-target nulls that exclude each
work unit's recorded true-target list; M2.5 label-shuffle and downsampling
calibration; and M3 empirical Brown combination with BY-FDR.
WASC's frozen calibration/stability gate failed when the SPOR group was reduced
from 294 to 25 (mean selected-set Jaccard 0.285 versus the required 0.70), so the
B=9999 production run and every edge-level verdict remain blocked. That result
is specific to this selector/pipeline and is not a general theorem about n=25.
"""
from __future__ import annotations

from .bins import (
    AnchorBins,
    assign_decile,
    build_anchor_bins,
    compute_marginal_correlation_with_anchor,
    compute_missingness_per_protein,
    load_measured_degrees,
    sample_matched_non_neighbors,
)
from .combination import (
    BrownResult,
    BrownTable,
    by_fdr,
    compute_brown_per_anchor,
    empirical_brown_per_anchor,
)
from .concordance import (
    CochranQResult,
    ConcordanceTable,
    cochran_q,
    compute_concordance_per_edge,
)
from .edges import (
    DEFAULT_CLUSTER_TERMS,
    compute_measured_cluster_members,
    enumerate_wasc_indra_edges,
)
from .fit import (
    EdgeBetaTable,
    FwlFit,
    fit_edges_per_group,
    fit_fwl_per_pair,
)
from .null import (
    AnchorNullResult,
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    append_checkpoint,
    compute_anchor_null,
    load_completed_anchors,
    run_null_serial,
)
from .preprocess import (
    GroupDesign,
    WascDataBundle,
    build_group_design,
    build_wasc_data_bundle,
    load_enriched_metadata,
    load_proteomics,
)
from .sanity import (
    LabelShuffleResult,
    downsample_group,
    run_label_shuffle_calibration,
    shuffle_group_labels,
)
from .types import Network, Theme, WascEdge

__all__ = [
    # M2.3 — bins
    "AnchorBins",
    "assign_decile",
    "build_anchor_bins",
    "compute_marginal_correlation_with_anchor",
    "compute_missingness_per_protein",
    "load_measured_degrees",
    "sample_matched_non_neighbors",
    # M2.4 — null loop
    "AnchorNullResult",
    "AnchorWork",
    "NullLoopContext",
    "anchor_seed",
    "append_checkpoint",
    "compute_anchor_null",
    "load_completed_anchors",
    "run_null_serial",
    # M2.5 — calibration tripwire
    "LabelShuffleResult",
    "downsample_group",
    "run_label_shuffle_calibration",
    "shuffle_group_labels",
    # M3 — empirical Brown's + BY-FDR
    "BrownResult",
    "BrownTable",
    "by_fdr",
    "compute_brown_per_anchor",
    "empirical_brown_per_anchor",
    # M1 — edges
    "DEFAULT_CLUSTER_TERMS",
    "compute_measured_cluster_members",
    "enumerate_wasc_indra_edges",
    # M2 — preprocess
    "GroupDesign",
    "WascDataBundle",
    "build_group_design",
    "build_wasc_data_bundle",
    "load_enriched_metadata",
    "load_proteomics",
    # M2 — fit
    "EdgeBetaTable",
    "FwlFit",
    "fit_edges_per_group",
    "fit_fwl_per_pair",
    # M2.2 — concordance
    "CochranQResult",
    "ConcordanceTable",
    "cochran_q",
    "compute_concordance_per_edge",
    # types
    "Network",
    "Theme",
    "WascEdge",
]
