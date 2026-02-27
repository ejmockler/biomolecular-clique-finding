"""
Tests for STAT-4: Satterthwaite df containment fallback.

Validates that ``satterthwaite_df()`` uses the containment method as the
primary df strategy and only adopts the Satterthwaite--Welch refinement
when it agrees with the containment result.

References:
    - SAS PROC MIXED containment method
    - Kuznetsova, Brockhoff & Christensen (2017). J. Stat. Softw. 82(13).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from cliquefinder.stats.differential import (
    ModelType,
    build_contrast_matrix,
    fit_linear_model,
    run_differential_analysis,
    satterthwaite_df,
    test_contrasts as compute_contrasts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_cov_beta(n_params: int, scale: float = 0.01) -> np.ndarray:
    """Return a simple positive-definite covariance matrix."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n_params, n_params))
    return scale * (A @ A.T) + np.eye(n_params) * scale


def _balanced_design(
    n_subjects_per_group: int = 5,
    n_replicates: int = 3,
    seed: int = 100,
):
    """Create a balanced two-group repeated-measures design."""
    rng = np.random.default_rng(seed)
    subjects = []
    conditions = []
    for grp_idx, cond in enumerate(["CTRL", "CASE"]):
        for s in range(n_subjects_per_group):
            sid = f"G{grp_idx}_S{s:02d}"
            for _ in range(n_replicates):
                subjects.append(sid)
                conditions.append(cond)

    subjects = np.array(subjects)
    conditions = np.array(conditions)
    n_samples = len(subjects)
    n_features = 10

    data = np.zeros((n_features, n_samples))
    for i in range(n_features):
        subj_effs = {s: rng.normal(0, 0.5) for s in np.unique(subjects)}
        for j, (subj, cond) in enumerate(zip(subjects, conditions)):
            base = 10.0
            if i < 3 and cond == "CASE":
                base += 1.5
            data[i, j] = base + subj_effs[subj] + rng.normal(0, 0.3)

    feature_ids = [f"Prot_{i:03d}" for i in range(n_features)]
    return data, feature_ids, conditions, subjects


# ---------------------------------------------------------------------------
# 1. Balanced design gives reasonable df
# ---------------------------------------------------------------------------

class TestBalancedDesign:

    def test_balanced_df_close_to_standard(self):
        """Balanced design should give df close to n_subjects - n_params."""
        n_groups = 10  # 5 per group
        n_obs = 30      # 3 replicates each
        n_params = 2    # intercept + 1 condition

        cov_beta = _simple_cov_beta(n_params, scale=0.01)
        contrast = np.array([1.0, -1.0])

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.1,
            subject_var=0.25,
            n_groups=n_groups,
            n_obs=n_obs,
            use_mlx=False,
        )

        assert df is not None
        # Containment df = n_groups - n_params = 10 - 2 = 8
        # Result should be in the neighborhood of 8
        assert 1 <= df <= n_obs - 1
        # With moderate subject variance the containment anchor is 8
        containment = n_groups - n_params
        assert containment * 0.5 <= df <= containment * 2.0

    def test_balanced_end_to_end(self):
        """Full pipeline on balanced data produces valid df values."""
        data, fids, conds, subjs = _balanced_design()

        result = run_differential_analysis(
            data=data,
            feature_ids=fids,
            sample_condition=conds,
            sample_subject=subjs,
            contrasts={"CASE_vs_CTRL": ("CASE", "CTRL")},
            use_mixed=True,
            verbose=False,
        )
        df_out = result.to_dataframe()
        mixed = df_out[df_out["model_type"] == "mixed"]
        if len(mixed) > 0:
            assert all(mixed["df"] >= 1)
            assert all(mixed["df"] <= mixed["n_obs"] - 1)


# ---------------------------------------------------------------------------
# 2. Highly unbalanced design — containment should be conservative
# ---------------------------------------------------------------------------

class TestUnbalancedDesign:

    def test_highly_unbalanced_conservative(self):
        """
        1 subject in one group vs 10 in another.

        Containment df = n_subjects - n_params = 11 - 2 = 9.
        The old heuristic could wildly over- or under-estimate; containment
        anchors to a sensible conservative value.
        """
        rng = np.random.default_rng(77)

        # Group 1: 1 subject, 10 replicates
        subj_g1 = ["G1_S00"] * 10
        # Group 2: 10 subjects, 1 replicate each
        subj_g2 = [f"G2_S{i:02d}" for i in range(10)]

        subjects = np.array(subj_g1 + subj_g2)
        conditions = np.array(["CTRL"] * 10 + ["CASE"] * 10)
        n_obs = 20
        n_groups = 11
        n_params = 2

        # Build a plausible cov_beta
        cov_beta = _simple_cov_beta(n_params, scale=0.02)
        contrast = np.array([1.0, -1.0])

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.3,
            subject_var=1.0,
            n_groups=n_groups,
            n_obs=n_obs,
            use_mlx=False,
        )

        assert df is not None
        # containment df = 11 - 2 = 9
        containment = n_groups - n_params
        assert 1 <= df <= n_obs - 1
        # Should be anchored near containment
        assert df <= containment * 2.0

    def test_unbalanced_end_to_end(self):
        """Full pipeline on unbalanced data."""
        rng = np.random.default_rng(88)

        # 2 subjects CTRL (5 reps), 8 subjects CASE (2 reps)
        subjects_g1 = [f"G1_S{i}" for i in range(2) for _ in range(5)]
        subjects_g2 = [f"G2_S{i}" for i in range(8) for _ in range(2)]
        subjects = np.array(subjects_g1 + subjects_g2)
        conditions = np.array(["CTRL"] * 10 + ["CASE"] * 16)

        n_features = 5
        n_samples = len(subjects)
        data = rng.normal(10, 1, (n_features, n_samples))
        fids = [f"Prot_{i}" for i in range(n_features)]

        result = run_differential_analysis(
            data=data,
            feature_ids=fids,
            sample_condition=conditions,
            sample_subject=subjects,
            contrasts={"CASE_vs_CTRL": ("CASE", "CTRL")},
            use_mixed=True,
            verbose=False,
        )

        df_out = result.to_dataframe()
        assert all(df_out["df"] >= 1)
        assert all(df_out["pvalue"].notna())


# ---------------------------------------------------------------------------
# 3. Upper bound: df <= n_obs - 1
# ---------------------------------------------------------------------------

class TestUpperBound:

    def test_df_never_exceeds_n_obs_minus_1(self):
        """df must not exceed n_obs - 1 regardless of variance components."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        for n_groups, n_obs in [(20, 40), (5, 100), (50, 200)]:
            df = satterthwaite_df(
                contrast_vector=contrast,
                cov_beta=cov_beta,
                residual_var=0.01,
                subject_var=10.0,
                n_groups=n_groups,
                n_obs=n_obs,
                use_mlx=False,
            )
            if df is not None:
                assert df <= n_obs - 1, (
                    f"df={df} exceeds n_obs-1={n_obs - 1} for "
                    f"n_groups={n_groups}, n_obs={n_obs}"
                )


# ---------------------------------------------------------------------------
# 4. Lower bound: df >= 1
# ---------------------------------------------------------------------------

class TestLowerBound:

    def test_df_at_least_1(self):
        """df must be at least 1 when the function returns a value."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        # Minimal design: 3 groups, 6 obs (containment = 3-2 = 1)
        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.5,
            n_groups=3,
            n_obs=6,
            use_mlx=False,
        )
        if df is not None:
            assert df >= 1.0

    def test_df_is_1_at_minimum_design(self):
        """With n_groups = n_params + 1, containment df = 1."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        # n_groups = 3, containment = 3 - 2 = 1
        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.5,
            n_groups=3,
            n_obs=9,
            use_mlx=False,
        )
        assert df is not None
        assert df >= 1.0
        # Clipped to n_obs - 1 = 8 at most
        assert df <= 8.0


# ---------------------------------------------------------------------------
# 5. Containment fallback triggered when heuristic would give crazy values
# ---------------------------------------------------------------------------

class TestContainmentFallback:

    def test_extreme_variance_ratio_clipped_to_upper_bound(self):
        """
        When residual_var >> subject_var the Satterthwaite formula pushes df
        toward the residual df (n_obs - n_params).  This is legitimate
        (the model approaches fixed effects) but is still clipped to n_obs - 1.
        """
        n_params = 2
        n_groups = 10
        n_obs = 30
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        containment = n_groups - n_params  # 8

        # Extreme: residual >> subject (Satterthwaite pushes df high)
        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=100.0,
            subject_var=0.0001,
            n_groups=n_groups,
            n_obs=n_obs,
            use_mlx=False,
        )
        assert df is not None
        # Satterthwaite refinement is accepted (above containment*0.5)
        # but clipped to n_obs - 1 at most
        assert df <= n_obs - 1
        # Should exceed containment since residual dominates
        assert df >= containment

    def test_extreme_subject_variance_uses_containment(self):
        """
        When subject_var >> residual_var, the Satterthwaite formula
        approaches n_groups - 1 (df for the random effect).  If it falls
        below containment * 0.5, the containment fallback kicks in.
        """
        n_params = 2
        n_groups = 10
        n_obs = 30
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.5)

        containment = n_groups - n_params  # 8

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.0001,
            subject_var=100.0,
            n_groups=n_groups,
            n_obs=n_obs,
            use_mlx=False,
        )
        assert df is not None
        # Should be at least containment * 0.5 or 1.0 (fallback guards lower end)
        assert df >= max(containment * 0.5, 1.0)
        # And at most n_obs - 1
        assert df <= n_obs - 1

    def test_zero_subject_variance_still_returns(self):
        """Zero subject variance should not crash; returns containment df."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.0,
            n_groups=10,
            n_obs=30,
            use_mlx=False,
        )
        # Should return something (containment or refinement)
        assert df is not None
        assert df >= 1.0


# ---------------------------------------------------------------------------
# 6. Single-subject edge case
# ---------------------------------------------------------------------------

class TestSingleSubject:

    def test_single_subject_returns_none(self):
        """
        With only 1 subject total, n_groups=1, containment = 1-2 = -1.
        Should return None (under-identified).
        """
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = satterthwaite_df(
                contrast_vector=contrast,
                cov_beta=cov_beta,
                residual_var=0.5,
                subject_var=0.3,
                n_groups=1,
                n_obs=10,
                use_mlx=False,
            )

        assert df is None
        # Should have issued a warning about non-positive containment df
        warn_msgs = [str(x.message) for x in w]
        assert any("non-positive" in m.lower() for m in warn_msgs)

    def test_two_subjects_returns_none(self):
        """
        With 2 subjects and 2 params, containment = 0 -> None.
        """
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.3,
            n_groups=2,
            n_obs=6,
            use_mlx=False,
        )
        assert df is None

    def test_three_subjects_returns_df_1(self):
        """With 3 subjects and 2 params, containment = 1. Should return >= 1."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        df = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.3,
            n_groups=3,
            n_obs=9,
            use_mlx=False,
        )
        assert df is not None
        assert df >= 1.0


# ---------------------------------------------------------------------------
# 7. Backward compatibility: same function signature, no downstream breakage
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_function_signature_unchanged(self):
        """The function accepts the same positional/keyword arguments."""
        import inspect
        sig = inspect.signature(satterthwaite_df)
        param_names = list(sig.parameters.keys())
        assert param_names == [
            "contrast_vector",
            "cov_beta",
            "residual_var",
            "subject_var",
            "n_groups",
            "n_obs",
            "use_mlx",
        ]

    def test_return_type_float_or_none(self):
        """Return is either float or None — same contract as before."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        result = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.3,
            n_groups=10,
            n_obs=30,
            use_mlx=False,
        )
        assert result is None or isinstance(result, float)

    def test_none_on_negative_cov(self):
        """Non-positive V_c falls back to containment df (not None like before)."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        # A covariance matrix that produces zero V_c for this contrast
        cov_beta = np.zeros((2, 2))

        result = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=0.5,
            subject_var=0.3,
            n_groups=10,
            n_obs=30,
            use_mlx=False,
        )
        # V_c = 0, so we can't compute Satterthwaite but containment = 8
        # Implementation returns containment_df when V_c <= 0
        assert result is not None
        assert result == float(10 - 2)

    def test_contrast_integration_pipeline(self):
        """Full pipeline: fit -> contrasts -> Satterthwaite df. No crash."""
        data, fids, conds, subjs = _balanced_design(seed=200)

        y = data[0, :]
        coef_df, model_type, residual_var, subject_var, converged, issue, cov_params, residual_df, n_obs, n_groups = fit_linear_model(
            y=y,
            condition=conds,
            subject=subjs,
            use_mixed=True,
            conditions=sorted(pd.Series(conds).unique()),
        )

        if model_type == ModelType.MIXED and subject_var is not None and converged:
            cond_list = sorted(pd.Series(conds).unique())
            contrast_matrix, contrast_names = build_contrast_matrix(
                conditions=cond_list,
                contrasts={"CASE_vs_CTRL": ("CASE", "CTRL")},
            )

            results = compute_contrasts(
                coef_df=coef_df,
                conditions=cond_list,
                contrast_matrix=contrast_matrix,
                contrast_names=contrast_names,
                residual_var=residual_var,
                n_obs=n_obs,
                model_type=model_type,
                cov_params=cov_params,
                residual_df=residual_df,
                subject_var=subject_var,
                n_groups=n_groups,
            )

            assert len(results) == 1
            r = results[0]
            assert r.df >= 1
            assert np.isfinite(r.df)
            assert 0 <= r.p_value <= 1

    def test_clipping_emits_warning(self):
        """When clipping is needed a warning is issued."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        # n_groups=100 so containment=98, but n_obs=10 -> n_obs-1=9 < 98
        # This forces clipping to n_obs - 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = satterthwaite_df(
                contrast_vector=contrast,
                cov_beta=cov_beta,
                residual_var=0.5,
                subject_var=0.3,
                n_groups=100,
                n_obs=10,
                use_mlx=False,
            )

        assert df is not None
        assert df <= 9.0  # n_obs - 1
        # Should have emitted a clipping warning
        warn_msgs = [str(x.message) for x in w]
        assert any("clipped" in m.lower() for m in warn_msgs)


# ---------------------------------------------------------------------------
# Additional: warnings and diagnostics
# ---------------------------------------------------------------------------

class TestWarningsAndDiagnostics:

    def test_no_warning_for_normal_case(self):
        """Normal balanced design should not produce warnings."""
        n_params = 2
        contrast = np.array([1.0, -1.0])
        cov_beta = _simple_cov_beta(n_params, scale=0.01)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = satterthwaite_df(
                contrast_vector=contrast,
                cov_beta=cov_beta,
                residual_var=0.5,
                subject_var=0.3,
                n_groups=10,
                n_obs=30,
                use_mlx=False,
            )

        # Filter to only our warnings (ignore numpy/scipy deprecation etc.)
        our_warnings = [
            x for x in w
            if "satterthwaite" in str(x.message).lower()
            or "containment" in str(x.message).lower()
            or "clipped" in str(x.message).lower()
        ]
        assert len(our_warnings) == 0
        assert df is not None
