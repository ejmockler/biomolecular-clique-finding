"""
Tests for ROAST rotation set statistic correctness.

Validates:
- MEAN statistic for MIXED alternative uses signed z, not |z|.
- FLOORMEAN for UP/DOWN applies floor to small z-scores.
- MEAN50 selects top 50% by |z| (unweighted), not by w*z.
- Sample weights applied to expression data Y before QR projection.

References:
    Wu et al. (2010) "ROAST: rotation gene set tests"
    limma R package, roast() implementation
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats as scipy_stats

from cliquefinder.stats.rotation import (
    Alternative,
    RotationPrecomputed,
    RotationTestEngine,
    SetStatistic,
    _compute_floormean_stat,
    _compute_mean50_stat,
    _compute_mean_stat,
    compute_rotation_matrices,
    compute_rotation_pvalues,
    compute_set_statistics,
    extract_gene_effects,
)


# =============================================================================
# MEAN statistic for MIXED alternative
# =============================================================================


class TestMeanMixedSignedZ:
    """MEAN+MIXED must use signed z-scores, with two-sided p-value."""

    def test_mean_mixed_returns_signed_mean(self):
        """MEAN-MIXED should return sum(w*z)/A, which can be negative."""
        # Gene set with mixed up/down regulation:
        # z1 = +3.0 (strong up), z2 = -2.0 (moderate down), z3 = -1.0 (weak down)
        z = np.array([[3.0, -2.0, -1.0]])  # (1, 3) — one "rotation"
        w = np.ones(3)
        A = np.sum(np.abs(w))

        result = _compute_mean_stat(z, w, A, Alternative.MIXED)

        # Signed mean: (3.0 + (-2.0) + (-1.0)) / 3 = 0.0
        expected = (3.0 - 2.0 - 1.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean_mixed_can_be_negative(self):
        """With more down-regulated genes, MEAN-MIXED should be negative."""
        z = np.array([[1.0, -3.0, -2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_mean_stat(z, w, A, Alternative.MIXED)

        # (1 - 3 - 2) / 3 = -4/3 < 0
        expected = (1.0 - 3.0 - 2.0) / 3.0
        assert result[0] < 0
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean_mixed_not_absolute(self):
        """MEAN-MIXED must differ from the absolute-value formula."""
        z = np.array([[1.0, -3.0, -2.0]])
        w = np.ones(3)
        A = 3.0

        signed_result = _compute_mean_stat(z, w, A, Alternative.MIXED)
        abs_result = np.sum(np.abs(w) * np.abs(z), axis=1) / A  # old wrong formula

        # Signed: (1-3-2)/3 = -4/3
        # Absolute: (1+3+2)/3 = 2
        assert signed_result[0] != abs_result[0]
        assert signed_result[0] < 0
        assert abs_result[0] > 0

    def test_mean_mixed_pvalue_uses_twosided_comparison(self):
        """P-value for MEAN+MIXED must use |null| >= |obs| comparison."""
        # Observed stat is small positive
        observed = {"mean": {"mixed": 0.5}}

        # Null distribution: some positive, some negative, some with |null| >= |obs|
        # null values: [-1.0, -0.6, -0.3, 0.2, 0.4, 0.8]
        # |null|:       1.0,  0.6,  0.3, 0.2, 0.4, 0.8
        # |null| >= |0.5|: -1.0 (yes), -0.6 (yes), 0.8 (yes) => 3 out of 6
        null = {"mean": {"mixed": np.array([-1.0, -0.6, -0.3, 0.2, 0.4, 0.8])}}

        p_values = compute_rotation_pvalues(observed, null)

        # p = (b + 1) / (B + 1) = (3 + 1) / (6 + 1) = 4/7
        expected_p = 4.0 / 7.0
        assert_allclose(p_values["mean"]["mixed"], expected_p, atol=1e-12)

    def test_mean_mixed_pvalue_differs_from_onesided(self):
        """Two-sided p-value should differ from one-sided for signed stats."""
        observed = {"mean": {"mixed": 0.5}}
        null = {"mean": {"mixed": np.array([-1.0, -0.6, -0.3, 0.2, 0.4, 0.8])}}

        p_values = compute_rotation_pvalues(observed, null)

        # One-sided would count null >= 0.5: only 0.8 => p = 2/7
        one_sided_p = 2.0 / 7.0
        assert p_values["mean"]["mixed"] != one_sided_p

    def test_mean_up_still_onesided(self):
        """MEAN+UP should still use standard one-sided comparison."""
        observed = {"mean": {"up": 0.5}}
        null = {"mean": {"up": np.array([-1.0, -0.6, 0.3, 0.2, 0.4, 0.8])}}

        p_values = compute_rotation_pvalues(observed, null)

        # One-sided: null >= 0.5: only 0.8 => p = (1+1)/(6+1) = 2/7
        expected_p = 2.0 / 7.0
        assert_allclose(p_values["mean"]["up"], expected_p, atol=1e-12)

    def test_mean_down_still_onesided(self):
        """MEAN+DOWN should still use standard one-sided comparison."""
        observed = {"mean": {"down": 1.5}}
        null = {"mean": {"down": np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])}}

        p_values = compute_rotation_pvalues(observed, null)

        # One-sided: null >= 1.5: 1.5, 2.0, 2.5, 3.0 => b=4 => p = 5/7
        expected_p = 5.0 / 7.0
        assert_allclose(p_values["mean"]["down"], expected_p, atol=1e-12)

    def test_mean_mixed_via_compute_set_statistics(self):
        """End-to-end: compute_set_statistics MEAN+MIXED returns signed mean."""
        z = np.array([2.0, -3.0, 1.0, -0.5])
        stats = compute_set_statistics(
            z,
            statistics=[SetStatistic.MEAN],
            alternatives=[Alternative.MIXED],
        )

        expected = np.mean(z)  # signed mean with equal weights
        assert_allclose(stats["mean"]["mixed"], expected, atol=1e-12)


# =============================================================================
# FLOORMEAN for UP/DOWN applies the floor
# =============================================================================


class TestFloormeanFloor:
    """FLOORMEAN UP/DOWN must floor small z at sqrt(median chi2_1) ~ 0.6745."""

    @pytest.fixture
    def floor_value(self):
        """The standard ROAST floor: sqrt(median(chi2_1))."""
        return np.sqrt(scipy_stats.chi2.ppf(0.5, df=1))

    def test_floormean_up_floors_small_positive_z(self, floor_value):
        """z=0.1 and z=0.3 (both < floor ~0.6745) should be floored up."""
        z = np.array([[0.1, 0.3, 2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.UP, floor_value)

        # Expected: (max(0.1, floor) + max(0.3, floor) + max(2.0, floor)) / 3
        # = (floor + floor + 2.0) / 3
        expected = (floor_value + floor_value + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_up_old_formula_wrong(self, floor_value):
        """Old formula max(z, 0) would give (0.1 + 0.3 + 2.0)/3, which is wrong."""
        z = np.array([[0.1, 0.3, 2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.UP, floor_value)
        old_wrong = (0.1 + 0.3 + 2.0) / 3.0  # No flooring

        assert result[0] != old_wrong
        assert result[0] > old_wrong  # Flooring raises small values

    def test_floormean_up_negative_z_zeroed(self, floor_value):
        """Negative z-scores contribute zero in UP direction (not floored)."""
        z = np.array([[-0.5, 0.1, 2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.UP, floor_value)

        # z=-0.5 contributes 0, z=0.1 floored to floor_value, z=2.0 stays
        expected = (0 + floor_value + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_down_floors_small_negative_z(self, floor_value):
        """Small negative z should be floored at sqrt(q) in DOWN direction."""
        z = np.array([[-0.1, -0.3, -2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.DOWN, floor_value)

        # For DOWN: f = where(z<0, max(|z|, floor), 0)
        # |-0.1|=0.1 < floor -> floor; |-0.3|=0.3 < floor -> floor; |-2.0|=2.0 > floor -> 2.0
        expected = (floor_value + floor_value + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_down_positive_z_zeroed(self, floor_value):
        """Positive z-scores contribute zero in DOWN direction."""
        z = np.array([[0.5, -0.1, -2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.DOWN, floor_value)

        # z=0.5 zeroed, z=-0.1 floored to floor_value, z=-2.0 stays at 2.0
        expected = (0 + floor_value + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_mixed_unchanged(self, floor_value):
        """MIXED floormean was already correct: max(|z|, floor) for all genes."""
        z = np.array([[0.1, -0.3, 2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.MIXED, floor_value)

        # max(|0.1|, floor) + max(|-0.3|, floor) + max(|2.0|, floor)
        expected = (floor_value + floor_value + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_up_large_z_not_floored(self, floor_value):
        """z-scores already above the floor should not be modified."""
        z = np.array([[1.0, 1.5, 2.0]])
        w = np.ones(3)
        A = 3.0

        result = _compute_floormean_stat(z, w, A, Alternative.UP, floor_value)

        # All z > floor, so no flooring needed
        expected = (1.0 + 1.5 + 2.0) / 3.0
        assert_allclose(result[0], expected, atol=1e-10)

    def test_floormean_up_via_compute_set_statistics(self, floor_value):
        """End-to-end test through compute_set_statistics API."""
        z = np.array([0.1, 0.3, 2.0])
        stats = compute_set_statistics(
            z,
            statistics=[SetStatistic.FLOORMEAN],
            alternatives=[Alternative.UP],
        )

        expected = (floor_value + floor_value + 2.0) / 3.0
        assert_allclose(stats["floormean"]["up"], expected, atol=1e-10)

    def test_floormean_down_via_compute_set_statistics(self, floor_value):
        """End-to-end test through compute_set_statistics API for DOWN."""
        z = np.array([-0.1, -0.3, -2.0])
        stats = compute_set_statistics(
            z,
            statistics=[SetStatistic.FLOORMEAN],
            alternatives=[Alternative.DOWN],
        )

        expected = (floor_value + floor_value + 2.0) / 3.0
        assert_allclose(stats["floormean"]["down"], expected, atol=1e-10)


# =============================================================================
# MEAN50 selects top 50% by |z|, not w*z
# =============================================================================


class TestMean50Selection:
    """MEAN50 must select top 50% genes by |z| (unweighted), not w*z."""

    def test_mean50_selects_by_abs_z_not_wz(self):
        """Top 50% selection by |z| vs w*z gives different results when
        weights and z-magnitudes conflict."""
        # 4 genes. We want top 2 by |z|.
        # z = [0.5, -3.0, 2.0, -0.1]
        # |z| = [0.5, 3.0, 2.0, 0.1]
        # Top 2 by |z|: genes 1 (|z|=3.0) and 2 (|z|=2.0)
        #
        # Weights deliberately chosen so w*z ordering differs:
        # w = [10.0, 0.1, 0.1, 10.0]
        # w*z = [5.0, -0.3, 0.2, -1.0]
        # Top 2 by w*z: genes 0 (5.0) and 2 (0.2) or 3 (-1.0) depending on UP/DOWN
        #
        # With correct |z| selection (genes 1,2):
        # mean(w*z for selected) = mean(0.1*(-3.0), 0.1*2.0) = mean(-0.3, 0.2) = -0.05
        #
        # With wrong w*z selection for UP (genes 0,2):
        # mean(w*z for selected) = mean(5.0, 0.2) = 2.6

        z = np.array([[0.5, -3.0, 2.0, -0.1]])
        w = np.array([10.0, 0.1, 0.1, 10.0])
        A = np.sum(np.abs(w))

        result = _compute_mean50_stat(z, w, A, Alternative.MIXED)

        # MIXED uses abs(z) before averaging.
        # Top 2 by |z|: genes 1 and 2
        # selected w*|z|: [0.1*3.0, 0.1*2.0] = [0.3, 0.2]
        # mean: (0.3 + 0.2) / 2 = 0.25
        expected = (0.3 + 0.2) / 2.0
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean50_up_selects_by_abs_z(self):
        """For UP, selection is still by |z|, then compute mean of w*z for selected."""
        z = np.array([[0.5, -3.0, 2.0, -0.1]])
        w = np.array([10.0, 0.1, 0.1, 10.0])
        A = np.sum(np.abs(w))

        result = _compute_mean50_stat(z, w, A, Alternative.UP)

        # Top 2 by |z|: genes 1 (|z|=3.0) and 2 (|z|=2.0)
        # selected w*z: [0.1*(-3.0), 0.1*2.0] = [-0.3, 0.2]
        # mean: (-0.3 + 0.2) / 2 = -0.05
        expected = np.mean([-0.3, 0.2])
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean50_down_selects_by_abs_z(self):
        """For DOWN, selection is still by |z|, negated."""
        z = np.array([[0.5, -3.0, 2.0, -0.1]])
        w = np.array([10.0, 0.1, 0.1, 10.0])
        A = np.sum(np.abs(w))

        result = _compute_mean50_stat(z, w, A, Alternative.DOWN)

        # Top 2 by |z|: genes 1 and 2
        # selected w*z: [-0.3, 0.2], mean = -0.05
        # DOWN negates: -(-0.05) = 0.05
        expected = -np.mean([-0.3, 0.2])
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean50_equal_weights_consistent(self):
        """With equal weights, MIXED uses abs(z) for averaging."""
        z = np.array([[1.0, -3.0, 2.0, -0.5]])
        w = np.ones(4)
        A = 4.0

        result = _compute_mean50_stat(z, w, A, Alternative.MIXED)

        # MIXED uses abs(z). Top 2 by |z|: genes 1 (3.0) and 2 (2.0)
        # w*|z| for selected: [1.0*3.0, 1.0*2.0] = [3.0, 2.0]
        # mean = (3.0 + 2.0) / 2.0 = 2.5
        expected = (3.0 + 2.0) / 2.0
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean50_h_is_half_floor(self):
        """For 4 genes, h = max(1, 4//2) = 2 (selects top 2)."""
        z = np.array([[4.0, 3.0, 2.0, 1.0]])
        w = np.ones(4)
        A = 4.0

        result = _compute_mean50_stat(z, w, A, Alternative.MIXED)

        # Top 2 by |z|: genes 0 (4.0) and 1 (3.0)
        expected = (4.0 + 3.0) / 2.0
        assert_allclose(result[0], expected, atol=1e-12)

    def test_mean50_h_at_least_1(self):
        """For 1 gene, h = max(1, 1//2) = max(1, 0) = 1."""
        z = np.array([[5.0]])
        w = np.ones(1)
        A = 1.0

        result = _compute_mean50_stat(z, w, A, Alternative.MIXED)
        assert_allclose(result[0], 5.0, atol=1e-12)

    def test_mean50_via_compute_set_statistics(self):
        """End-to-end through compute_set_statistics."""
        z = np.array([0.5, -3.0, 2.0, -0.1])
        w = np.array([10.0, 0.1, 0.1, 10.0])
        stats = compute_set_statistics(
            z,
            weights=w,
            statistics=[SetStatistic.MEAN50],
            alternatives=[Alternative.MIXED],
        )

        # MIXED uses abs(z).
        # Top 2 by |z|: genes 1 (3.0) and 2 (2.0)
        # w*|z| for selected: 0.1*3.0=0.3, 0.1*2.0=0.2
        expected = (0.3 + 0.2) / 2.0
        assert_allclose(stats["mean50"]["mixed"], expected, atol=1e-12)

    def test_mean50_batched(self):
        """Test with multiple rotations (batch dimension)."""
        z = np.array([
            [0.5, -3.0, 2.0, -0.1],
            [1.0, -0.5, 0.3, -4.0],
        ])
        w = np.ones(4)
        A = 4.0

        result = _compute_mean50_stat(z, w, A, Alternative.MIXED)

        # MIXED uses abs(z).
        # Rotation 0: top 2 by |z| = genes 1 (3.0), 2 (2.0)
        #   mean(|z|) = mean(3.0, 2.0) = 2.5
        # Rotation 1: top 2 by |z| = genes 3 (4.0), 0 (1.0)
        #   mean(|z|) = mean(4.0, 1.0) = 2.5
        assert_allclose(result[0], 2.5, atol=1e-12)
        assert_allclose(result[1], 2.5, atol=1e-12)


# =============================================================================
# Sample weights applied to expression data Y
# =============================================================================


class TestSampleWeightsOnY:
    """Sample weights must be applied to Y as well as X for QR consistency."""

    def test_w_sqrt_vec_stored_in_precomputed(self):
        """RotationPrecomputed stores w_sqrt_vec when weights are provided."""
        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')
        weights = np.array([1.0, 2.0, 1.5, 1.0, 2.5, 1.0, 1.5, 2.0, 1.0, 1.5])

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )

        assert precomputed.w_sqrt_vec is not None
        assert_allclose(precomputed.w_sqrt_vec, np.sqrt(weights), atol=1e-12)

    def test_w_sqrt_vec_none_without_weights(self):
        """RotationPrecomputed.w_sqrt_vec is None when no weights provided."""
        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast,
        )

        assert precomputed.w_sqrt_vec is None

    def test_w_sqrt_vec_immutable(self):
        """w_sqrt_vec should be read-only (immutable via frozen dataclass)."""
        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')
        weights = np.ones(10) * 2.0

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )

        with pytest.raises((ValueError, TypeError)):
            precomputed.w_sqrt_vec[0] = 999.0

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce the same gene effects as no weights."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 20, 10
        Y = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE{i}" for i in range(n_genes)]

        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')

        # Unweighted
        precomp_unw = compute_rotation_matrices(
            sample_condition, conditions, contrast,
        )
        effects_unw = extract_gene_effects(Y, gene_ids, precomp_unw)

        # Uniform weights (all 1.0)
        weights = np.ones(n_samples)
        precomp_w = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )
        effects_w = extract_gene_effects(Y, gene_ids, precomp_w)

        # U matrices should be the same (up to sign convention)
        # Since sqrt(1.0) = 1.0, weighting Y by 1.0 changes nothing
        assert_allclose(
            np.abs(effects_w.U[:, 0]),
            np.abs(effects_unw.U[:, 0]),
            atol=1e-10,
        )
        assert_allclose(
            effects_w.sample_variances,
            effects_unw.sample_variances,
            atol=1e-10,
        )

    def test_nonuniform_weights_differ_from_unweighted(self):
        """Non-uniform weights must produce different gene effects than unweighted."""
        rng = np.random.default_rng(123)
        n_genes, n_samples = 20, 10
        Y = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE{i}" for i in range(n_genes)]

        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')

        # Unweighted
        precomp_unw = compute_rotation_matrices(
            sample_condition, conditions, contrast,
        )
        effects_unw = extract_gene_effects(Y, gene_ids, precomp_unw)

        # Non-uniform weights (heavily weighted toward first sample)
        weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        precomp_w = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )
        effects_w = extract_gene_effects(Y, gene_ids, precomp_w)

        # The first elements of U (contrast effects) should differ
        # because Y is weighted differently
        contrast_diff = np.mean(np.abs(effects_w.U[:, 0] - effects_unw.U[:, 0]))
        assert contrast_diff > 0.01, (
            f"Non-uniform weights should change gene effects, but diff = {contrast_diff}"
        )

    def test_weighted_y_projection_is_consistent(self):
        """Verify that Y is weighted by w_sqrt_vec before Q2 projection."""
        rng = np.random.default_rng(77)
        n_genes, n_samples = 5, 6
        Y = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"G{i}" for i in range(n_genes)]

        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 3 + ['CTRL'] * 3)
        contrast = ('CASE', 'CTRL')
        weights = np.array([2.0, 1.0, 3.0, 1.5, 2.5, 1.0])

        precomp = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )

        # Manually compute expected U
        w_sqrt = np.sqrt(weights)
        Y_weighted = Y * w_sqrt[np.newaxis, :]
        U_expected = Y_weighted @ precomp.Q2

        effects = extract_gene_effects(Y, gene_ids, precomp)

        assert_allclose(effects.U, U_expected, atol=1e-10)

    def test_weighted_rho_sq_consistency(self):
        """rho_sq should be computed from the weighted projection."""
        rng = np.random.default_rng(99)
        n_genes, n_samples = 10, 8
        Y = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"G{i}" for i in range(n_genes)]

        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 4 + ['CTRL'] * 4)
        contrast = ('CASE', 'CTRL')
        weights = np.array([3.0, 1.0, 2.0, 1.5, 1.0, 2.0, 1.5, 3.0])

        precomp = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )
        effects = extract_gene_effects(Y, gene_ids, precomp)

        # rho_sq should be sum of U^2
        expected_rho_sq = np.sum(effects.U ** 2, axis=1)
        assert_allclose(effects.rho_sq, expected_rho_sq, atol=1e-10)

    def test_w_sqrt_vec_preserved_after_eb_update(self):
        """w_sqrt_vec should be preserved when RotationPrecomputed is re-created
        with EB parameters in the engine's fit() method."""
        conditions = ['CASE', 'CTRL']
        sample_condition = np.array(['CASE'] * 5 + ['CTRL'] * 5)
        contrast = ('CASE', 'CTRL')
        weights = np.array([1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5])

        precomp = compute_rotation_matrices(
            sample_condition, conditions, contrast, sample_weights=weights,
        )

        # Simulate the EB update that happens in fit()
        precomp_updated = RotationPrecomputed(
            Q2=precomp.Q2,
            df_residual=precomp.df_residual,
            contrast_name=precomp.contrast_name,
            eb_d0=5.0,
            eb_s0_sq=0.1,
            eb_df_total=5.0 + precomp.df_residual,
            design_rank=precomp.design_rank,
            n_samples=precomp.n_samples,
            w_sqrt_vec=precomp.w_sqrt_vec,
        )

        assert precomp_updated.w_sqrt_vec is not None
        assert_allclose(
            precomp_updated.w_sqrt_vec,
            np.sqrt(weights),
            atol=1e-12,
        )


# =============================================================================
# Integration tests across multiple fixes
# =============================================================================


class TestIntegration:
    """Integration tests that exercise multiple fixes together."""

    def test_all_statistics_computed_without_error(self):
        """All statistics and alternatives should compute without exceptions."""
        rng = np.random.default_rng(42)
        z = rng.standard_normal(10)

        stats = compute_set_statistics(z)

        for stat in SetStatistic:
            for alt in Alternative:
                assert stat.value in stats
                assert alt.value in stats[stat.value]
                assert np.isfinite(stats[stat.value][alt.value])

    def test_pvalues_for_all_stat_alt_combinations(self):
        """p-values should be computed for all stat+alt combos."""
        rng = np.random.default_rng(42)

        # Observed
        z_obs = rng.standard_normal(10)
        obs_stats = compute_set_statistics(z_obs)

        # Null
        z_null = rng.standard_normal((100, 10))
        null_stats = compute_set_statistics(z_null)

        p_values = compute_rotation_pvalues(obs_stats, null_stats)

        for stat in SetStatistic:
            assert stat.value in p_values
            for alt in Alternative:
                assert alt.value in p_values[stat.value]
                p = p_values[stat.value][alt.value]
                assert 0 < p <= 1, f"p-value for {stat.value}+{alt.value} out of range: {p}"

    def test_mean_mixed_symmetry(self):
        """For a symmetric null distribution, MEAN+MIXED p-value should be ~0.5 for obs=0."""
        rng = np.random.default_rng(42)

        observed = {"mean": {"mixed": 0.0}}
        null = {"mean": {"mixed": rng.standard_normal(10000)}}

        p_values = compute_rotation_pvalues(observed, null)

        # |null| >= |0| = |null| >= 0: all nulls pass => p ~ 1.0
        # Actually |null| >= 0 is always true for abs values
        assert_allclose(p_values["mean"]["mixed"], 1.0, atol=0.01)
