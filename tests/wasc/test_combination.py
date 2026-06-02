"""Tests for the WASC M3 Brown's combination + BY-FDR.

Covers:
- empirical_brown_per_anchor:
    * Independence: when null Q rows are statistically independent,
      Brown's df ≈ 2·n_eff and c ≈ 1 (recovers Fisher's identity).
    * Strong positive dependence: when null Q rows are perfectly
      correlated, df shrinks toward 2 and c grows toward n_eff.
    * Single-edge anchor: Brown == Fisher exactly.
    * NaN handling: Q_obs NaN → p NaN → edge dropped from n_eff;
      empty anchor → all-NaN result.
    * p_floor: caps log(0) blow-up when Q_obs > every null draw.
    * Reproducibility: same inputs → identical output.
- compute_brown_per_anchor: per-anchor table shape; preserves anchor order.
- by_fdr wrapper:
    * Delegates to differential.fdr_correction(method='BY')
    * Returns (rejected, q) tuple with NaN-propagation.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

from cliquefinder.stats.wasc.combination import (
    BrownResult,
    BrownTable,
    by_fdr,
    compute_brown_per_anchor,
    empirical_brown_per_anchor,
)
from cliquefinder.stats.wasc.null import AnchorNullResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_null_result(
    n_edges: int = 3,
    B: int = 500,
    *,
    Q_obs: np.ndarray | None = None,
    p_values: np.ndarray | None = None,
    null_Q: np.ndarray | None = None,
) -> AnchorNullResult:
    """Build a synthetic AnchorNullResult for testing."""
    if null_Q is None:
        rng = np.random.default_rng(0)
        null_Q = rng.chisquare(df=2, size=(n_edges, B))
    if Q_obs is None:
        Q_obs = np.median(null_Q, axis=1)  # ~ p=0.5 each
    if p_values is None:
        # Compute the per-edge p-value the same way compute_anchor_null does
        p_values = np.array([
            (1 + np.sum(null_Q[i] >= Q_obs[i])) / (B + 1)
            for i in range(n_edges)
        ])
    return AnchorNullResult(
        anchor_uniprot="TEST",
        edge_ids=tuple(f"e{i}" for i in range(n_edges)),
        Q_obs=Q_obs,
        null_Q=null_Q,
        p_values=p_values,
        n_degenerate_per_edge=np.zeros(n_edges, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Independence: Brown ≈ Fisher
# ---------------------------------------------------------------------------

class TestIndependence:
    def test_independent_rows_recover_fisher_df(self):
        """With STATISTICALLY INDEPENDENT null Q rows, the empirical
        covariance off-diagonal is sample noise; df ≈ 2·n_eff and c ≈ 1.
        """
        rng = np.random.default_rng(42)
        n_edges, B = 5, 2000
        # Independent chi2(2) per row
        null_Q = rng.chisquare(df=2, size=(n_edges, B))
        Q_obs = np.median(null_Q, axis=1)
        nr = _make_null_result(n_edges, B, null_Q=null_Q, Q_obs=Q_obs)
        b = empirical_brown_per_anchor(nr)
        # Independence ⇒ Var ≈ 4n, c ≈ 1, df ≈ 2n
        assert abs(b.c - 1.0) < 0.15, f"c={b.c} not ≈ 1 under independence"
        assert abs(b.df - 2 * n_edges) < 1.5, f"df={b.df} not ≈ {2*n_edges}"

    def test_brown_equals_fisher_for_single_edge(self):
        nr = _make_null_result(n_edges=1, B=500)
        b = empirical_brown_per_anchor(nr)
        assert b.p_brown == b.p_fisher
        assert b.c == 1.0
        assert b.df == 2.0
        assert b.n_edges_combined == 1


# ---------------------------------------------------------------------------
# Dependence: Brown ≠ Fisher
# ---------------------------------------------------------------------------

class TestDependence:
    def test_perfectly_correlated_rows_shrink_df(self):
        """When all null Q rows are IDENTICAL (perfect positive dependence),
        the effective df should shrink toward 2 (one degree of freedom's
        worth) and c should grow to ~n_eff."""
        rng = np.random.default_rng(0)
        B = 1000
        # All rows are the same column draw
        base = rng.chisquare(df=2, size=B)
        n_edges = 4
        null_Q = np.tile(base, (n_edges, 1))
        Q_obs = np.full(n_edges, float(np.median(base)))
        nr = _make_null_result(n_edges, B, null_Q=null_Q, Q_obs=Q_obs)
        b = empirical_brown_per_anchor(nr)
        # Heavy positive dependence ⇒ Var >> 4·n ⇒ c > 1, df < 2·n
        assert b.c > 1.5, f"c={b.c} should be much larger than 1 under perfect dependence"
        assert b.df < 2 * n_edges, f"df={b.df} should shrink below 2n={2*n_edges}"

    def test_brown_more_conservative_than_fisher_under_positive_dep(self):
        """With positive dependence, p_brown >= p_fisher (Brown rejects less)."""
        rng = np.random.default_rng(7)
        B = 800
        n_edges = 3
        # Construct correlated null draws via shared latent
        latent = rng.chisquare(df=2, size=B)
        noise = rng.chisquare(df=2, size=(n_edges, B))
        null_Q = 0.7 * latent + 0.3 * noise  # broadcasts (B,) over rows
        # Add a moderately extreme Q_obs to make Fisher reject:
        Q_obs = np.array([float(np.quantile(null_Q[i], 0.95)) for i in range(n_edges)])
        nr = _make_null_result(n_edges, B, null_Q=null_Q, Q_obs=Q_obs)
        b = empirical_brown_per_anchor(nr)
        assert b.p_brown >= b.p_fisher - 1e-9, (
            f"Brown p {b.p_brown} not >= Fisher p {b.p_fisher} under positive dep"
        )


# ---------------------------------------------------------------------------
# NaN / edge-case handling
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_all_nan_p_returns_all_nan_result(self):
        nr = AnchorNullResult(
            anchor_uniprot="X", edge_ids=("a",),
            Q_obs=np.array([np.nan]),
            null_Q=np.full((1, 100), np.nan),
            p_values=np.array([np.nan]),
            n_degenerate_per_edge=np.array([100]),
        )
        b = empirical_brown_per_anchor(nr)
        assert np.isnan(b.chi2_obs)
        assert np.isnan(b.p_brown)
        assert b.n_edges_combined == 0

    def test_partial_nan_p_drops_those_edges(self):
        rng = np.random.default_rng(0)
        n_edges, B = 4, 500
        null_Q = rng.chisquare(df=2, size=(n_edges, B))
        p_values = np.array([0.5, np.nan, 0.3, 0.1])  # only 3 finite
        Q_obs = np.median(null_Q, axis=1)
        nr = AnchorNullResult(
            anchor_uniprot="X",
            edge_ids=tuple(f"e{i}" for i in range(n_edges)),
            Q_obs=Q_obs, null_Q=null_Q, p_values=p_values,
            n_degenerate_per_edge=np.array([0, B, 0, 0]),
        )
        b = empirical_brown_per_anchor(nr)
        assert b.n_edges_combined == 3
        assert np.isfinite(b.p_brown)


class TestPFloor:
    def test_pfloor_default_avoids_log_zero(self):
        """If Q_obs > every null draw, raw p = 0 (without floor).  The
        default p_floor = 1/(B+1) prevents log(0) → -inf chi2."""
        B = 100
        null_Q = np.full((2, B), 1.0)
        Q_obs = np.array([10.0, 10.0])  # exceeds all null draws
        p_values = np.array([1.0 / (B + 1), 1.0 / (B + 1)])  # what compute_anchor_null returns
        nr = AnchorNullResult(
            anchor_uniprot="X", edge_ids=("a", "b"),
            Q_obs=Q_obs, null_Q=null_Q, p_values=p_values,
            n_degenerate_per_edge=np.zeros(2, dtype=np.int64),
        )
        b = empirical_brown_per_anchor(nr)
        assert np.isfinite(b.chi2_obs)
        assert np.isfinite(b.p_brown)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_inputs_same_outputs(self):
        nr = _make_null_result()
        b1 = empirical_brown_per_anchor(nr)
        b2 = empirical_brown_per_anchor(nr)
        assert b1.chi2_obs == b2.chi2_obs
        assert b1.df == b2.df
        assert b1.p_brown == b2.p_brown


# ---------------------------------------------------------------------------
# compute_brown_per_anchor
# ---------------------------------------------------------------------------

class TestComputeBrownPerAnchor:
    def test_table_shape_and_order(self):
        results = [
            _make_null_result(n_edges=2),
            _make_null_result(n_edges=3),
            _make_null_result(n_edges=1),
        ]
        for i, r in enumerate(results):
            object.__setattr__(r, "anchor_uniprot", f"A{i}")
        tab = compute_brown_per_anchor(results)
        assert isinstance(tab, BrownTable)
        assert tab.anchors == ["A0", "A1", "A2"]
        assert tab.n_edges.tolist() == [2, 3, 1]
        assert tab.p_brown.shape == (3,)


# ---------------------------------------------------------------------------
# by_fdr
# ---------------------------------------------------------------------------

class TestByFdr:
    def test_basic_wiring(self):
        # 10 p-values, half tiny, half large
        p = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
        rejected, q = by_fdr(p, alpha=0.10)
        assert rejected.shape == q.shape == (10,)
        # Tiny p's should be rejected, large should not
        assert rejected[0]    # p=0.001 rejected
        assert not rejected[-1]  # p=0.9 not rejected
        # BY correction inflates q vs raw p
        assert all(q >= p - 1e-12)

    def test_nan_passes_through(self):
        p = np.array([0.01, np.nan, 0.05, np.nan])
        rejected, q = by_fdr(p, alpha=0.10)
        assert np.isnan(q[1]) and np.isnan(q[3])
        assert not rejected[1] and not rejected[3]

    def test_all_nan_returns_all_nan(self):
        p = np.full(5, np.nan)
        rejected, q = by_fdr(p, alpha=0.10)
        assert np.isnan(q).all()
        assert not rejected.any()
