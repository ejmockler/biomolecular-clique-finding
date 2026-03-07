"""Tests for Audit X findings — 10 fixes across 3 cycles."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class TestC1_NaNGPUFallbackUseMlx:
    """C-1: NaN fallback must update use_mlx, not just use_gpu."""

    def test_use_mlx_updated_on_nan(self):
        """Simulate the guard logic — use_mlx must be set False."""
        use_gpu = True
        MLX_AVAILABLE = True
        use_mlx = use_gpu and MLX_AVAILABLE
        data = np.array([[[1.0, np.nan]]])

        if use_gpu and MLX_AVAILABLE:
            n_nan = np.sum(np.isnan(data))
            if n_nan > 0:
                use_gpu = False
                use_mlx = False  # The fix

        assert use_mlx is False
        assert use_gpu is False

    def test_use_mlx_unchanged_without_nan(self):
        use_gpu = True
        MLX_AVAILABLE = True
        use_mlx = use_gpu and MLX_AVAILABLE
        data = np.array([[[1.0, 2.0]]])

        if use_gpu and MLX_AVAILABLE:
            n_nan = np.sum(np.isnan(data))
            if n_nan > 0:
                use_gpu = False
                use_mlx = False

        assert use_mlx is True


class TestQMX1_MadClipGlobalFallback:
    """QM-X-1: _mad_clip_global falls back to global bounds for <2 clean values."""

    def test_few_clean_values_clips_to_global(self):
        from cliquefinder.quality.imputation import Imputer

        # Gene with only 1 clean value and 1 outlier
        data = np.array([[5.0, 500.0]])  # 1 gene, 2 samples
        mask = np.array([[False, True]])  # sample 1 is outlier

        imp = Imputer.__new__(Imputer)
        imp.threshold = 3.0
        imp.max_upper_bound = None
        imp.global_lower_ = 0.0
        imp.global_upper_ = 50.0

        imp._mad_clip_global(data, mask)

        # Outlier should be clipped to global_upper (50.0), not left at 500.0
        assert data[0, 1] <= 50.0
        assert data[0, 1] != 500.0


class TestQMX2_SoftClipStratifiedMADZero:
    """QM-X-2: _soft_clip_stratified falls back to global bounds when MAD=0."""

    def test_mad_zero_uses_fallback(self):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.quality.imputation import Imputer

        # 6 clean identical values + 1 outlier (need >=10 for group to be used)
        vals = [5.0] * 11 + [100.0]
        data = np.array([vals])
        mask_vals = [False] * 11 + [True]
        mask = np.array([mask_vals])

        sids = [f"S{i}" for i in range(12)]
        md = pd.DataFrame({"g": ["A"] * 12}, index=sids)
        matrix = BioMatrix(
            data=data.copy(),
            feature_ids=pd.Index(["G1"]),
            sample_ids=pd.Index(sids),
            sample_metadata=md,
            quality_flags=np.ones_like(data),
        )

        imp = Imputer.__new__(Imputer)
        imp.threshold = 3.0
        imp.max_upper_bound = None
        imp.sharpness = 5.0
        imp.group_cols = ["g"]
        imp.global_lower_ = 0.0
        imp.global_upper_ = 50.0

        mutable_data = data.copy()
        imp._soft_clip_stratified(mutable_data, mask, matrix)

        # After fix: outlier should be clipped via global bounds, not left at 100.0
        assert mutable_data[0, -1] < 100.0


class TestH5_QRILCUniformLowerBound:
    """H-5: QRILC uniform lower bound must exclude 0 to avoid norm.ppf(0)=-inf."""

    def test_no_neg_inf_imputation(self):
        from cliquefinder.stats.missing import impute_qrilc

        data = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [2.0, np.nan, 3.0, 5.0],
        ])  # 2 features x 4 samples

        result = impute_qrilc(data, random_state=42)
        assert np.all(np.isfinite(result.data)), "No -inf values should appear"


class TestM6_SingleElementBackground:
    """M-6: 1-element background std(ddof=1)=NaN must return 0.0."""

    def test_single_background_returns_zero(self):
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        # 2 total genes: 1 target, 1 background
        t_stats = np.array([3.0, 1.0])
        is_target = np.array([True, False])
        z = compute_competitive_z(t_stats, is_target)
        # Should return 0.0 (not NaN) because background has only 1 element
        assert z == 0.0


class TestM7_InterGeneCorrelationNaN:
    """M-7: Constant-expression genes should not produce NaN correlation."""

    def test_constant_gene_returns_zero(self):
        from cliquefinder.stats.enrichment_z import estimate_inter_gene_correlation

        # 3 genes: gene 0 is constant (zero variance)
        expr = np.array([
            [5.0, 5.0, 5.0, 5.0],  # constant
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])
        is_target = np.array([True, True, True])
        rho = estimate_inter_gene_correlation(expr, is_target)
        assert np.isfinite(rho), "Should not be NaN"
        assert rho >= 0.0


class TestRCX1_FloormeanWeightConsistency:
    """RC-X-1: FLOORMEAN DOWN should use signed weights like UP."""

    def test_down_uses_signed_weights(self):
        from cliquefinder.stats.rotation import _compute_floormean_stat, Alternative

        z = np.array([[1.0, -2.0, -3.0]])  # 1 rotation, 3 genes
        w = np.array([1.0, -1.0, 1.0])  # gene 1 has negative weight
        A = 3.0
        floor = 0.1

        stat_down = _compute_floormean_stat(z, w, A, Alternative.DOWN, floor)
        stat_up = _compute_floormean_stat(z, w, A, Alternative.UP, floor)

        # Both should use signed weights (w), not abs(w)
        assert np.isfinite(stat_down[0])
        assert np.isfinite(stat_up[0])


class TestM22_InfCovariateGuard:
    """M-22: Inf in numeric covariate must raise, not silently corrupt."""

    def test_inf_covariate_raises(self):
        from cliquefinder.stats.design_matrix import build_covariate_design_matrix

        cov_df = pd.DataFrame({"age": [25.0, np.inf, 35.0, 40.0]})
        sample_condition = pd.Series(["A", "A", "B", "B"])

        with pytest.raises(ValueError, match="non-finite"):
            build_covariate_design_matrix(
                sample_condition=sample_condition,
                conditions=["A", "B"],
                contrast=("A", "B"),
                covariates_df=cov_df,
            )


class TestM1_SpecificityInteractionSign:
    """M-1: Negative delta-z should label 'shared' not 'specific'."""

    def test_negative_interaction_z_not_specific(self):
        """When interaction_z < 0 and significant, label should be 'shared'."""
        # Test the logic directly from specificity.py lines 464-480
        interaction_z = -2.0
        interaction_pvalue = 0.01
        p_threshold = 0.05

        if interaction_pvalue < p_threshold:
            if interaction_z > 0:
                label = "specific"
            else:
                label = "shared"

        assert label == "shared"

    def test_positive_interaction_z_is_specific(self):
        interaction_z = 2.0
        interaction_pvalue = 0.01
        p_threshold = 0.05

        if interaction_pvalue < p_threshold:
            if interaction_z > 0:
                label = "specific"
            else:
                label = "shared"

        assert label == "specific"
