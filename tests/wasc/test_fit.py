"""Tests for the WASC F-W (Frisch-Waugh-Lovell) per-pair regression kernel.

Includes the M1 numerical-identity gate: F-W fits agree with
statsmodels.OLS on the anchor coefficient to within 1e-8 on synthetic
triples (covers 50 random configurations). The brutalist-required
real-data validation is in `scripts/wasc/run_m1_gate.py`.

Also covers:
- swap-invariance check (j~a vs a~j): Q should be insensitive to the
  choice of which protein is "anchor" vs "target" up to the slope-sign
  flip; documented if drift > 5%
- min_n threshold behavior
- NaN handling (per-pair drop)
- Collinear anchor-with-covariates → β undefined
"""
from __future__ import annotations

import numpy as np
import pytest

from cliquefinder.stats.wasc.fit import FwlFit, fit_fwl_per_pair


# ---------------------------------------------------------------------------
# Numerical-identity gate vs statsmodels.OLS — the M1 hard requirement
# ---------------------------------------------------------------------------

class TestFwlVsOlsIdentity:
    """The M1 gate: F-W must agree with statsmodels.OLS on the anchor
    coefficient and SE to within 1e-8 on 50 synthetic triples spanning
    a range of n, p_cov, noise levels, and design conditions."""

    def _make_triple(self, n, p_cov, beta_true, noise_sigma, rng):
        """One synthetic (target, anchor, X_cov) triple with known beta_true."""
        # Intercept + random correlated covariates.
        intercept = np.ones(n)
        cov_pre = rng.standard_normal((n, p_cov - 1))  # p_cov - 1 non-intercept
        # Add some correlation between covariates and anchor for realism.
        anchor = rng.standard_normal(n) + 0.3 * cov_pre[:, 0]
        # True model: target = β · anchor + γ · X_cov + ε
        gamma = rng.standard_normal(p_cov - 1) * 0.5
        target = (
            beta_true * anchor
            + cov_pre @ gamma
            + noise_sigma * rng.standard_normal(n)
        )
        X_cov = np.column_stack([intercept, cov_pre])
        return target, anchor, X_cov

    @pytest.mark.parametrize("seed", range(50))
    def test_50_random_triples_agree_to_1e8(self, seed):
        """50 synthetic configurations; F-W β and SE must match statsmodels
        to 1e-8 absolute tolerance on β, 1e-8 relative on SE."""
        import statsmodels.api as sm

        rng = np.random.default_rng(seed)
        # Vary n, p_cov, beta_true, noise.
        n = int(rng.choice([20, 40, 80, 150, 300]))
        p_cov = int(rng.choice([2, 4, 6]))    # intercept + 1..5 covariates
        beta_true = float(rng.uniform(-2.0, 2.0))
        noise = float(rng.uniform(0.1, 1.5))
        target, anchor, X_cov = self._make_triple(n, p_cov, beta_true, noise, rng)

        # F-W fit
        fit = fit_fwl_per_pair(target, anchor, X_cov, min_n=5)
        assert fit.converged, f"F-W didn't converge for seed {seed}"

        # statsmodels OLS with [X_cov, anchor] as the design
        X_full = np.column_stack([X_cov, anchor])
        ols = sm.OLS(target, X_full).fit()
        ols_beta = ols.params[-1]
        ols_se = ols.bse[-1]
        ols_df = int(ols.df_resid)

        # M1 gate: identity to 1e-8
        assert abs(fit.beta - ols_beta) < 1e-8, (
            f"β mismatch seed={seed}: FW={fit.beta:.12f}, OLS={ols_beta:.12f}"
        )
        assert abs(fit.se - ols_se) / max(ols_se, 1e-12) < 1e-8, (
            f"SE mismatch seed={seed}: FW={fit.se:.12f}, OLS={ols_se:.12f}"
        )
        assert fit.df == ols_df, f"df mismatch seed={seed}: FW={fit.df}, OLS={ols_df}"


# ---------------------------------------------------------------------------
# Swap-invariance — informational; documents direction-symmetry behaviour
# ---------------------------------------------------------------------------

class TestSwapInvariance:
    """When (anchor, target) are swapped, β_a flips sign-relationship and
    magnitude may differ (OLS of y~x ≠ OLS of x~y in general). This test
    documents the slope-symmetry behavior; the spec's "Q invariant up to
    5%" caveat (§12) is verified by the production sweep, not here."""

    def test_swap_alters_beta_magnitude_in_general(self):
        rng = np.random.default_rng(0)
        n = 100
        x = rng.standard_normal(n)
        y = 1.5 * x + 0.3 * rng.standard_normal(n)
        X_cov = np.column_stack([np.ones(n)])

        fit_yx = fit_fwl_per_pair(y, x, X_cov, min_n=5)
        fit_xy = fit_fwl_per_pair(x, y, X_cov, min_n=5)

        # In OLS y~x, β ≈ Cov(x,y)/Var(x). In OLS x~y, β ≈ Cov(x,y)/Var(y).
        # These differ unless Var(x) = Var(y). For x with low noise, β_yx >> β_xy.
        assert fit_yx.converged and fit_xy.converged
        # Product of the two slopes equals R² (population identity).
        r2_approx = fit_yx.beta * fit_xy.beta
        assert 0 < r2_approx < 1, f"Expected 0 < β_yx·β_xy < 1, got {r2_approx}"


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

class TestMinNFloor:
    def test_below_min_n_returns_unconverged(self):
        n = 8
        rng = np.random.default_rng(0)
        target = rng.standard_normal(n)
        anchor = rng.standard_normal(n)
        X_cov = np.column_stack([np.ones(n), rng.standard_normal(n)])
        fit = fit_fwl_per_pair(target, anchor, X_cov, min_n=10)
        assert not fit.converged
        assert np.isnan(fit.beta)
        assert fit.df == 0


class TestNanHandling:
    def test_nan_in_target_drops_those_rows(self):
        rng = np.random.default_rng(0)
        n = 50
        anchor = rng.standard_normal(n)
        target = 0.5 * anchor + 0.1 * rng.standard_normal(n)
        X_cov = np.column_stack([np.ones(n)])
        # Inject NaNs in target
        target[3:6] = np.nan
        target[15] = np.nan
        fit = fit_fwl_per_pair(target, anchor, X_cov, min_n=5)
        # Should converge on the remaining 46 rows
        assert fit.converged
        assert fit.n == 46
        # Beta should be near 0.5 (true effect), give or take noise
        assert abs(fit.beta - 0.5) < 0.1

    def test_nan_in_covariate_drops_row(self):
        rng = np.random.default_rng(0)
        n = 50
        anchor = rng.standard_normal(n)
        target = 0.5 * anchor + 0.1 * rng.standard_normal(n)
        cov = rng.standard_normal(n)
        cov[10:15] = np.nan
        X_cov = np.column_stack([np.ones(n), cov])
        fit = fit_fwl_per_pair(target, anchor, X_cov, min_n=5)
        assert fit.converged
        assert fit.n == 45


class TestCollinearity:
    def test_anchor_collinear_with_covariate_yields_nan_beta(self):
        """If anchor is a deterministic function of an existing covariate,
        β_a is undefined (collinearity)."""
        rng = np.random.default_rng(0)
        n = 50
        cov = rng.standard_normal(n)
        anchor = 2.0 * cov + 1.0  # perfectly collinear
        target = rng.standard_normal(n)
        X_cov = np.column_stack([np.ones(n), cov])
        fit = fit_fwl_per_pair(target, anchor, X_cov, min_n=5)
        # F-W projection of anchor onto X_cov gives ≈ 0 residual → β undefined
        assert np.isnan(fit.beta) or abs(fit.beta) > 1e6


class TestFwlFitProperties:
    def test_converged_property(self):
        fit = FwlFit(beta=0.5, se=0.1, df=10, n=15)
        assert fit.converged

    def test_unconverged_when_nan(self):
        fit = FwlFit(beta=np.nan, se=np.nan, df=0, n=0)
        assert not fit.converged

    def test_unconverged_when_se_zero(self):
        fit = FwlFit(beta=0.5, se=0.0, df=10, n=15)
        assert not fit.converged
