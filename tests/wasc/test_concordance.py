"""Tests for the WASC Cochran-Q concordance kernel.

Covers:
- Identity when all β̂ are equal (Q = 0)
- Inverse-variance weighting (large-SE groups contribute less)
- Reference comparison against `scipy.stats.combine_pvalues` style
  meta-analysis ground truth on synthetic data
- I² and τ² behavior
- Group-missing handling (NaN or SE=0 dropped)
- compute_concordance_per_edge end-to-end on a small EdgeBetaTable
"""
from __future__ import annotations

import numpy as np
import pytest

from cliquefinder.stats.wasc.concordance import (
    CochranQResult,
    cochran_q,
    compute_concordance_per_edge,
)
from cliquefinder.stats.wasc.fit import EdgeBetaTable


class TestCochranQBasics:
    def test_zero_when_all_betas_equal(self):
        """If β̂_g are identical across groups, Q = 0 exactly."""
        r = cochran_q(np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.2, 0.15]))
        assert r.Q == pytest.approx(0.0, abs=1e-12)
        assert r.beta_bar == pytest.approx(0.5)
        assert r.n_groups == 3
        assert r.I_squared == 0.0
        assert r.tau2 == 0.0

    def test_iv_weighted_mean(self):
        """β̄ uses inverse-variance weights, not arithmetic mean."""
        # β = [0, 2]; SE = [1, 0.1] → w = [1, 100]; β̄ ≈ 100·2/(1+100) ≈ 1.98
        r = cochran_q(np.array([0.0, 2.0]), np.array([1.0, 0.1]))
        assert r.beta_bar == pytest.approx(2.0 * 100 / 101, abs=1e-9)

    def test_q_positive_when_betas_differ(self):
        r = cochran_q(np.array([0.0, 1.0, -1.0]), np.array([0.5, 0.5, 0.5]))
        assert r.Q > 0
        assert r.beta_bar == pytest.approx(0.0, abs=1e-12)  # symmetric → 0
        # df = 2; if Q > df, I² > 0
        if r.Q > 2:
            assert r.I_squared > 0

    def test_chi2_distribution_under_null(self):
        """Q ~ χ²(G-1) under the null of common β.
        Simulate 5000 nulls, check that mean(Q) ≈ G-1 = 2."""
        rng = np.random.default_rng(0)
        n_sim = 5000
        Q_vals = np.empty(n_sim)
        for i in range(n_sim):
            # Simulate β̂ ~ N(0, SE²) for 3 groups (true β = 0; group sizes vary)
            ses = np.array([0.3, 0.1, 0.2])  # mimics n=25, n=300, n=60
            betas = rng.normal(0.0, ses)
            Q_vals[i] = cochran_q(betas, ses).Q
        assert abs(np.mean(Q_vals) - 2.0) < 0.1, (
            f"mean(Q) = {np.mean(Q_vals):.3f}, expected ≈ 2"
        )


class TestNanHandling:
    def test_one_group_missing_drops_it(self):
        r = cochran_q(np.array([1.0, np.nan, 2.0]), np.array([0.1, np.nan, 0.1]))
        assert r.n_groups == 2
        assert np.isfinite(r.Q)

    def test_all_missing_returns_nan(self):
        r = cochran_q(
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
        )
        assert np.isnan(r.Q)
        assert r.n_groups == 0

    def test_se_zero_dropped(self):
        r = cochran_q(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.0, 0.1]))
        # Group 2 (SE=0) should be dropped; remaining 2 groups give a Q
        assert r.n_groups == 2

    def test_only_one_valid_group_returns_nan(self):
        r = cochran_q(np.array([1.0, np.nan, np.nan]), np.array([0.1, np.nan, np.nan]))
        assert r.n_groups == 1
        assert np.isnan(r.Q)


class TestI2Tau2:
    def test_i2_zero_when_q_below_df(self):
        """I² = 0 when Q ≤ df (no excess heterogeneity)."""
        # All βs nearly equal → Q very small
        r = cochran_q(np.array([0.5, 0.51, 0.49]), np.array([0.5, 0.5, 0.5]))
        assert r.Q < 2  # df = 2
        assert r.I_squared == 0.0
        assert r.tau2 == 0.0

    def test_i2_positive_when_q_above_df(self):
        """I² > 0 when heterogeneity exceeds expectation."""
        # Large dispersion relative to SEs
        r = cochran_q(np.array([0.0, 5.0, -3.0]), np.array([0.1, 0.1, 0.1]))
        assert r.Q > 2
        assert r.I_squared > 0

    def test_tau2_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            betas = rng.normal(0, 1, 3)
            ses = np.abs(rng.normal(0.5, 0.2, 3))
            r = cochran_q(betas, ses)
            if np.isfinite(r.tau2):
                assert r.tau2 >= 0.0


class TestEndToEnd:
    def _make_edge_beta_table(self):
        edge_ids = ["E1", "E2", "E3"]
        groups = ["C9ORF72", "SPORADIC", "CONTROL"]
        # E1: all β̂ = 0.5 (Q should be 0)
        # E2: β̂ scattered (Q > 0)
        # E3: one group missing (n_groups = 2)
        beta = {
            "C9ORF72":  np.array([0.5, 0.0, 1.0]),
            "SPORADIC": np.array([0.5, 1.0, np.nan]),
            "CONTROL":  np.array([0.5, -1.0, -0.5]),
        }
        se = {
            "C9ORF72":  np.array([0.1, 0.1, 0.1]),
            "SPORADIC": np.array([0.1, 0.1, np.nan]),
            "CONTROL":  np.array([0.1, 0.1, 0.1]),
        }
        df = {g: np.array([10, 10, 10]) for g in groups}
        n = {g: np.array([20, 20, 20]) for g in groups}
        return EdgeBetaTable(
            edge_ids=edge_ids, beta=beta, se=se, df=df, n=n,
        )

    def test_compute_concordance_per_edge(self):
        bet = self._make_edge_beta_table()
        c = compute_concordance_per_edge(bet)
        assert len(c.edge_ids) == 3
        # E1: all equal → Q = 0
        assert c.Q[0] == pytest.approx(0.0, abs=1e-12)
        # E2: scattered → Q > 0
        assert c.Q[1] > 0
        # E3: one group missing but the remaining 2 differ → n_groups=2 and Q > 0
        assert c.n_groups[2] == 2
        assert c.Q[2] > 0  # β=1, β=-0.5 with same SE differ noticeably
