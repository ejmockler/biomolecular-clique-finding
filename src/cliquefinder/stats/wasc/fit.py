"""WASC F-W (Frisch-Waugh-Lovell) per-(edge, group) regression kernel.

Per spec §2:
    y_{j,s} = β_0 + β_a · anchor_{a,s} + γ · covariates_s + ε_s   for s ∈ donors(g)

The estimate of interest is β̂_{j|a,g} and its standard error.  The kernel
uses the Frisch-Waugh-Lovell theorem: residualize y_j and y_a against the
covariates X_cov once per (group, protein) pair, then per-edge β̂ is a
single dot-product computation (the coefficient from regressing
y_j_resid on y_a_resid with no intercept).

This is equivalent to the full OLS coefficient on the anchor (FWL theorem)
but is O(n_anchors · n_targets) dot products instead of O(n_edges · n)
OLS solves.  The numerical-identity gate (test_fit_fwl_vs_ols_identity)
validates this to within 1e-8 against statsmodels.OLS on real triples.

Missing-data policy: per-pair, drop rows where any of {y_target, y_anchor,
any covariate} is NaN.  Residualization is recomputed against the
per-pair restricted X_cov subset — strict OLS-equivalence (not a fast-path
approximation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FwlFit:
    """Result of one (target, anchor, group) F-W fit.

    Attributes
    ----------
    beta : float
        OLS coefficient on the anchor; equivalent to the anchor coefficient
        in the full model `y_target ~ y_anchor + X_cov`.
    se   : float
        Standard error of `beta`, with degrees of freedom adjusted for the
        full design (n − p_cov − 1).
    df   : int
        Residual degrees of freedom of the full model fit.
    n    : int
        Number of donors retained after per-pair NaN masking.
    """
    beta: float
    se: float
    df: int
    n: int

    @property
    def converged(self) -> bool:
        return (
            self.df > 0
            and self.n > 0
            and np.isfinite(self.beta)
            and np.isfinite(self.se)
            and self.se > 0
        )


def fit_fwl_per_pair(
    target_y: np.ndarray,
    anchor_y: np.ndarray,
    X_cov: np.ndarray,
    *,
    min_n: int = 10,
) -> FwlFit:
    """Fit one ``y_target = β·y_anchor + X_cov·γ`` regression via FWL.

    Parameters
    ----------
    target_y, anchor_y
        ``(n_samples,)`` arrays of log2 abundances for the target and anchor proteins
        in the donor group of interest.
    X_cov
        ``(n_samples, p_cov)`` covariate matrix.  Must include an intercept column;
        :func:`build_group_design` returns one with intercept in column 0.
    min_n
        Hard floor on the number of complete-case donors.  Below this, returns
        a non-converged FwlFit.

    Returns
    -------
    FwlFit
    """
    # Per-pair complete-case mask.
    nan_mask = (
        np.isnan(target_y)
        | np.isnan(anchor_y)
        | np.isnan(X_cov).any(axis=1)
    )
    keep = ~nan_mask
    n = int(keep.sum())
    p_cov = X_cov.shape[1]
    p_total = p_cov + 1  # +1 for the anchor coefficient
    df = n - p_total
    if n < min_n or df <= 0:
        return FwlFit(beta=np.nan, se=np.nan, df=0, n=n)

    y = target_y[keep]
    a = anchor_y[keep]
    X = X_cov[keep]

    # Hat matrix for X_cov (projection onto its column space).
    # Use pinv for numerical stability on near-rank-deficient designs.
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    y_resid = y - X @ (XtX_inv @ (X.T @ y))
    a_resid = a - X @ (XtX_inv @ (X.T @ a))

    aa = float(np.dot(a_resid, a_resid))
    if aa < 1e-15:
        # Anchor is fully explained by covariates (collinearity); β undefined.
        return FwlFit(beta=np.nan, se=np.nan, df=df, n=n)

    beta = float(np.dot(y_resid, a_resid) / aa)
    final_resid = y_resid - beta * a_resid
    rss = float(np.dot(final_resid, final_resid))
    sigma2 = rss / df
    if sigma2 < 0 or not np.isfinite(sigma2):
        return FwlFit(beta=beta, se=np.nan, df=df, n=n)
    se = float(np.sqrt(sigma2 / aa))
    return FwlFit(beta=beta, se=se, df=df, n=n)


class EdgeBetaTable(NamedTuple):
    """Compact per-edge β/SE table across one or more groups.

    ``beta[g][i]`` is β̂ for edge i in group g; analogous for ``se`` and ``df``.
    ``edge_ids[i]`` is the canonical edge identifier (anchor|target).
    """
    edge_ids: list[str]
    beta: dict[str, np.ndarray]
    se:   dict[str, np.ndarray]
    df:   dict[str, np.ndarray]
    n:    dict[str, np.ndarray]


def fit_edges_per_group(
    edges: list,                        # tuple[WascEdge, ...] from M1
    abundance,                          # pd.DataFrame proteins × samples
    designs: dict,                      # group_name -> GroupDesign
    *,
    min_n_per_group: dict[str, int] | None = None,
    verbose: bool = True,
) -> EdgeBetaTable:
    """Run the F-W per-(edge, group) fit across all edges.

    Parameters
    ----------
    edges
        Tuple of :class:`WascEdge` (M1 output).
    abundance
        Proteomics matrix; columns indexed by sample_id.
    designs
        ``{group_name: GroupDesign}`` from :func:`build_wasc_data_bundle`.
    min_n_per_group
        Per-group lower bound on retained donors (after per-pair NaN drop).
        Defaults from spec §2.3: C9ORF72: 10, others: 15.

    Returns
    -------
    EdgeBetaTable
    """
    if min_n_per_group is None:
        min_n_per_group = {"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15}

    n_edges = len(edges)
    groups = list(designs.keys())
    edge_ids = [e.edge_id for e in edges]
    beta = {g: np.full(n_edges, np.nan) for g in groups}
    se = {g: np.full(n_edges, np.nan) for g in groups}
    df = {g: np.zeros(n_edges, dtype=int) for g in groups}
    n_used = {g: np.zeros(n_edges, dtype=int) for g in groups}

    # Pre-extract per-group sample alignment: (n_g,) array indices into
    # `abundance.columns`.  Done once per group.
    sample_index = {
        g: [abundance.columns.get_loc(s) for s in d.sample_ids]
        for g, d in designs.items()
    }
    A = abundance.values  # numpy view; rows = proteins, cols = samples

    # Pre-extract per-group X_cov.
    X_per_group = {g: d.X_cov for g, d in designs.items()}

    # Edge loop.  Each anchor / target lookup is one pandas .get_loc per
    # protein per edge; cache once per unique UniProt to avoid repeat lookup.
    uniprot_to_row: dict[str, int] = {}
    for e in edges:
        for up in (e.anchor_uniprot, e.target_uniprot):
            if up not in uniprot_to_row:
                try:
                    uniprot_to_row[up] = abundance.index.get_loc(up)
                except KeyError:
                    uniprot_to_row[up] = -1

    for i, e in enumerate(edges):
        a_row = uniprot_to_row[e.anchor_uniprot]
        j_row = uniprot_to_row[e.target_uniprot]
        if a_row < 0 or j_row < 0:
            continue
        for g in groups:
            cols = sample_index[g]
            target_y = A[j_row, cols]
            anchor_y = A[a_row, cols]
            X = X_per_group[g]
            fit = fit_fwl_per_pair(
                target_y, anchor_y, X,
                min_n=min_n_per_group.get(g, 10),
            )
            beta[g][i] = fit.beta
            se[g][i] = fit.se
            df[g][i] = fit.df
            n_used[g][i] = fit.n
        if verbose and (i + 1) % 200 == 0:
            logger.info("Fit edges %d/%d", i + 1, n_edges)

    return EdgeBetaTable(
        edge_ids=edge_ids,
        beta=beta,
        se=se,
        df=df,
        n=n_used,
    )
