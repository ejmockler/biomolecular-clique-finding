"""WASC concordance statistic — inverse-variance-weighted Cochran Q.

Per spec §3:

    w_g = 1 / SE(β̂_g)²
    β̄  = Σ_g w_g · β̂_g  /  Σ_g w_g
    Q   = Σ_g w_g · (β̂_g − β̄)²

Lower-tail Q ⇒ small dispersion ⇒ invariant slopes ⇒ WASC-positive
(after permutation null in M3).  Q ~ χ²(G-1) under the null of common β.

Companion statistics from meta-analysis:
  I²  — heterogeneity percentage:        max(0, (Q − df)/Q) × 100
  τ²  — DerSimonian-Laird between-study variance estimate
        max(0, (Q − df) / (Σw − Σw² / Σw))

These are reported as descriptive/secondary outcomes per spec §3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CochranQResult:
    """Per-edge Cochran-Q + auxiliary heterogeneity stats."""
    Q: float
    beta_bar: float
    I_squared: float          # in [0, 100]
    tau2: float               # DerSimonian-Laird between-group variance
    n_groups: int             # number of groups with finite β̂_g, SE>0


def cochran_q(
    betas: np.ndarray,
    ses: np.ndarray,
) -> CochranQResult:
    """Inverse-variance-weighted Cochran Q on one edge across groups.

    Parameters
    ----------
    betas, ses
        ``(G,)`` arrays of β̂_g and SE(β̂_g) for one edge.  Entries with
        non-finite values OR SE ≤ 0 are dropped (treated as group-missing
        for this edge).

    Returns
    -------
    CochranQResult
        With ``Q = nan`` and ``n_groups < 2`` if fewer than 2 valid groups.
    """
    betas = np.asarray(betas, dtype=np.float64)
    ses = np.asarray(ses, dtype=np.float64)
    valid = np.isfinite(betas) & np.isfinite(ses) & (ses > 0)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return CochranQResult(
            Q=np.nan, beta_bar=np.nan,
            I_squared=np.nan, tau2=np.nan, n_groups=n_valid,
        )

    b = betas[valid]
    s = ses[valid]
    w = 1.0 / (s * s)
    sum_w = w.sum()
    beta_bar = float((w * b).sum() / sum_w)
    Q = float((w * (b - beta_bar) ** 2).sum())
    df = n_valid - 1

    # I² heterogeneity
    if Q <= 0 or df <= 0:
        I2 = 0.0
    else:
        I2 = float(100.0 * max(0.0, (Q - df) / Q))

    # DerSimonian-Laird τ²
    sum_w2 = float((w * w).sum())
    denom = sum_w - sum_w2 / sum_w
    if denom <= 0 or Q < df:
        tau2 = 0.0
    else:
        tau2 = float(max(0.0, (Q - df) / denom))

    return CochranQResult(
        Q=Q, beta_bar=beta_bar,
        I_squared=I2, tau2=tau2, n_groups=n_valid,
    )


class ConcordanceTable(NamedTuple):
    """Per-edge concordance table across N edges."""
    edge_ids: list[str]
    Q: np.ndarray
    beta_bar: np.ndarray
    I_squared: np.ndarray
    tau2: np.ndarray
    n_groups: np.ndarray


def compute_concordance_per_edge(
    bet,                       # EdgeBetaTable
    group_order: tuple[str, ...] = ("C9ORF72", "SPORADIC", "CONTROL"),
) -> ConcordanceTable:
    """Apply :func:`cochran_q` to every row of an :class:`EdgeBetaTable`.

    Parameters
    ----------
    bet
        EdgeBetaTable from :func:`fit_edges_per_group`.
    group_order
        Group keys in `bet.beta` / `bet.se`. Default matches the spec.

    Returns
    -------
    ConcordanceTable
    """
    n_edges = len(bet.edge_ids)
    Q = np.full(n_edges, np.nan)
    beta_bar = np.full(n_edges, np.nan)
    I2 = np.full(n_edges, np.nan)
    Tau2 = np.full(n_edges, np.nan)
    NG = np.zeros(n_edges, dtype=int)

    for i in range(n_edges):
        betas = np.array([bet.beta[g][i] for g in group_order])
        ses = np.array([bet.se[g][i] for g in group_order])
        r = cochran_q(betas, ses)
        Q[i] = r.Q
        beta_bar[i] = r.beta_bar
        I2[i] = r.I_squared
        Tau2[i] = r.tau2
        NG[i] = r.n_groups

    return ConcordanceTable(
        edge_ids=bet.edge_ids,
        Q=Q,
        beta_bar=beta_bar,
        I_squared=I2,
        tau2=Tau2,
        n_groups=NG,
    )
