"""WASC M3 — empirical Brown's combination per spec §5.

For each anchor a with n_a within-theme edges, combine the per-edge
permutation p-values into a single anchor-level p-value that accounts for
the EMPIRICAL DEPENDENCE between -2 log p values (Poole et al. 2016).

Under independence Brown reduces to Fisher:
    chi2_a = -2 sum_j log p(j, a)   ~   chi2(2 n_a)

Under dependence (positive correlation through shared anchor expression
vector, shared donors, shared covariate design):

    E[chi2_a]   = 2 n_a
    Var[chi2_a] = 4 n_a  +  2 * Sum_{i < j} cov_emp(-2 log p_i, -2 log p_j)
                = 4 n_a  +  sum_{i != j} cov_emp(-2 log p_i, -2 log p_j)
                (the two forms are algebraically equivalent — the second
                 is the "ordered-pair" sum over the off-diagonal of the
                 full empirical covariance matrix)

    c_a  = Var[chi2_a] / (2 E[chi2_a])
    df_a = 2 E[chi2_a]^2 / Var[chi2_a]
    p_a  = P(chi2(df_a) >= chi2_a / c_a)

The empirical covariance is computed from the null distribution of
(-2 log p^(b)_i, -2 log p^(b)_j) pairs across the B permutation
iterations.  p^(b)_i is the rank-based empirical p-value of the i-th
edge's null Q at iteration b, computed against the column distribution
Q_null[i, :].

This module also wires per-edge BY-FDR via the existing primitive in
differential.py (no new code; we re-export for convenience).

Spec §5 notes:
  - Brown is chosen over Fisher because within-anchor edge tests are
    positively correlated through the shared anchor expression vector.
    Fisher would over-state significance under positive dependence.
  - Fisher (c = 1, df = 2 n_a) is also reported as a SECONDARY for
    contrast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import rankdata

from .null import AnchorNullResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrownResult:
    """Per-anchor empirical Brown's combination output.

    Attributes
    ----------
    chi2_obs : float
        Observed combined statistic: ``-2 sum_j log p(j, a)``.
    df : float
        Effective degrees of freedom after empirical-dependence adjustment.
    c : float
        Scaling factor (Var/2E).  c > 1 under positive dependence.
    p_brown : float
        Anchor-level p-value.  NaN if too few finite edge p-values.
    p_fisher : float
        Secondary: Fisher's combination (c = 1, df = 2 n_eff), reported
        alongside per spec.
    n_edges_combined : int
        Number of edges with finite p-values that entered the combination.
    n_iter_used : int
        Number of permutation iterations that contributed finite log-p
        vectors to the empirical covariance estimate.  NaN if the
        covariance was not estimable (returns the Fisher analytic
        distribution as the fallback).
    """
    chi2_obs: float
    df: float
    c: float
    p_brown: float
    p_fisher: float
    n_edges_combined: int
    n_iter_used: int


def empirical_brown_per_anchor(
    null_result: AnchorNullResult,
    *,
    p_floor: float | None = None,
    eps: float = 1e-300,
) -> BrownResult:
    """Empirical Brown's combination on one anchor's null-loop result.

    Parameters
    ----------
    null_result
        Output of :func:`compute_anchor_null` — provides ``p_values``
        (per-edge permutation p-values) and ``null_Q`` (``(n_edges, B)``
        null Cochran-Q matrix used for empirical covariance estimation).
    p_floor
        Lower bound for any p-value used in the chi2 statistic.  Default
        ``1 / (B + 1)`` (Phipson-Smyth floor — the smallest p-value any
        per-edge result can ever return at permutation depth B).  This
        guards against log(0) when Q_obs exceeds every null draw.
    eps
        Numerical floor on the log argument as a final defense against
        underflow.

    Returns
    -------
    BrownResult
    """
    p_obs = null_result.p_values
    null_Q = null_result.null_Q
    n_edges, B = null_Q.shape
    if p_floor is None:
        p_floor = 1.0 / (B + 1)

    finite_mask = np.isfinite(p_obs)
    n_eff = int(finite_mask.sum())
    if n_eff < 1:
        return BrownResult(
            chi2_obs=np.nan, df=np.nan, c=np.nan,
            p_brown=np.nan, p_fisher=np.nan,
            n_edges_combined=0, n_iter_used=0,
        )

    p_obs_finite = np.maximum(p_obs[finite_mask], max(p_floor, eps))
    # Observed chi2
    chi2_obs = float(-2.0 * np.sum(np.log(p_obs_finite)))

    # Fisher (secondary)
    p_fisher = float(chi2_dist.sf(chi2_obs, 2 * n_eff))

    # Single-edge anchor: Brown == Fisher exactly (no off-diagonal cov)
    if n_eff == 1:
        return BrownResult(
            chi2_obs=chi2_obs, df=2.0 * n_eff, c=1.0,
            p_brown=p_fisher, p_fisher=p_fisher,
            n_edges_combined=n_eff, n_iter_used=B,
        )

    # Build per-edge (-2 log p_emp) matrix across the B iterations.
    null_Q_finite = null_Q[finite_mask]   # (n_eff, B)
    minus2log = np.full(null_Q_finite.shape, np.nan, dtype=np.float64)

    for i in range(n_eff):
        row = null_Q_finite[i]
        valid_mask = np.isfinite(row)
        n_valid = int(valid_mask.sum())
        if n_valid < 2:
            continue
        # LOWER-TAIL per spec §4: small Q ⇒ small p ⇒ WASC-positive.
        # Rank ASCENDING so smallest Q gets rank 1 → smallest empirical p.
        row_valid = row[valid_mask]
        ranks = rankdata(row_valid, method="average")
        emp_p = ranks / n_valid
        emp_p = np.maximum(emp_p, max(p_floor, eps))
        minus2log[i, valid_mask] = -2.0 * np.log(emp_p)

    # Use only iterations where ALL finite edges have a finite log-p value.
    iter_all_finite = np.isfinite(minus2log).all(axis=0)
    n_iter_used = int(iter_all_finite.sum())
    if n_iter_used < 2:
        # Cannot estimate covariance — fall back to Fisher
        logger.debug(
            f"Anchor {null_result.anchor_uniprot}: only {n_iter_used} iter "
            f"with full finite log-p; falling back to Fisher."
        )
        return BrownResult(
            chi2_obs=chi2_obs, df=2.0 * n_eff, c=1.0,
            p_brown=p_fisher, p_fisher=p_fisher,
            n_edges_combined=n_eff, n_iter_used=n_iter_used,
        )

    M = minus2log[:, iter_all_finite]   # (n_eff, n_iter_used)
    # Empirical covariance: ddof=1 ("sample" covariance)
    M_centered = M - M.mean(axis=1, keepdims=True)
    cov_mat = (M_centered @ M_centered.T) / (n_iter_used - 1)

    # Brown variance: 4 n + sum_off-diagonal of full empirical cov matrix.
    # The off-diagonal sum is symmetric (cov[i,j] = cov[j,i]) so this
    # equals 2 · Sum_{i<j} cov[i,j], matching the spec §5 form.
    E_chi2 = 2.0 * n_eff
    sum_off = float(cov_mat.sum() - np.trace(cov_mat))
    Var_chi2 = 4.0 * n_eff + sum_off

    # Degenerate guard: if Var_chi2 ≤ 0 (large negative correlations make
    # the variance estimate non-positive), fall back to Fisher rather than
    # producing a bogus df.
    if Var_chi2 <= 0:
        logger.warning(
            f"Anchor {null_result.anchor_uniprot}: empirical Var ≤ 0 "
            f"({Var_chi2:.3g}); falling back to Fisher."
        )
        return BrownResult(
            chi2_obs=chi2_obs, df=2.0 * n_eff, c=1.0,
            p_brown=p_fisher, p_fisher=p_fisher,
            n_edges_combined=n_eff, n_iter_used=n_iter_used,
        )

    c = Var_chi2 / (2.0 * E_chi2)
    df = 2.0 * E_chi2 ** 2 / Var_chi2
    p_brown = float(chi2_dist.sf(chi2_obs / c, df))

    return BrownResult(
        chi2_obs=chi2_obs, df=df, c=c,
        p_brown=p_brown, p_fisher=p_fisher,
        n_edges_combined=n_eff, n_iter_used=n_iter_used,
    )


class BrownTable(NamedTuple):
    """Per-anchor Brown's table across many anchors."""
    anchors: list[str]
    n_edges: np.ndarray
    chi2_obs: np.ndarray
    df: np.ndarray
    c: np.ndarray
    p_brown: np.ndarray
    p_fisher: np.ndarray


def compute_brown_per_anchor(
    null_results: list[AnchorNullResult],
    **kwargs,
) -> BrownTable:
    """Apply :func:`empirical_brown_per_anchor` to every anchor."""
    anchors: list[str] = []
    n_edges = np.zeros(len(null_results), dtype=np.int64)
    chi2 = np.full(len(null_results), np.nan)
    df = np.full(len(null_results), np.nan)
    c = np.full(len(null_results), np.nan)
    p_brown = np.full(len(null_results), np.nan)
    p_fisher = np.full(len(null_results), np.nan)
    for i, r in enumerate(null_results):
        b = empirical_brown_per_anchor(r, **kwargs)
        anchors.append(r.anchor_uniprot)
        n_edges[i] = b.n_edges_combined
        chi2[i] = b.chi2_obs
        df[i] = b.df
        c[i] = b.c
        p_brown[i] = b.p_brown
        p_fisher[i] = b.p_fisher
    return BrownTable(
        anchors=anchors, n_edges=n_edges, chi2_obs=chi2,
        df=df, c=c, p_brown=p_brown, p_fisher=p_fisher,
    )


# ---------------------------------------------------------------------------
# BY-FDR re-export (spec §6: existing primitive in differential.py)
# ---------------------------------------------------------------------------

def by_fdr(p_values: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Yekutieli FDR adjustment over p_values.

    Returns ``(rejected, q_values)`` where ``q_values[i]`` is the
    BY-adjusted p-value and ``rejected[i]`` is True iff
    ``q_values[i] <= alpha``.  NaN p-values pass through as NaN and are
    NOT counted as rejected.

    Spec §6: BY (not BH) because per-edge p-values are positively
    dependent through (a) shared anchors, (b) shared targets,
    (c) shared null draws within an anchor.  BY controls FDR under
    arbitrary dependence at the cost of an inflation factor
    H_n = sum_{i=1}^{n} 1/i.
    """
    from cliquefinder.stats.differential import fdr_correction
    q = fdr_correction(p_values, method="BY", alpha=alpha)
    rejected = np.zeros_like(q, dtype=bool)
    finite = np.isfinite(q)
    rejected[finite] = q[finite] <= alpha
    return rejected, q
