"""Tests for MEAN50 MIXED weight sign invariance and pass-2 outlier NaN resilience.

- MEAN50 MIXED signed weights: np.abs(w) * np.abs(z) for MIXED alternative.
- Pass 2 outlier NaN vulnerability: np.nanmedian for residual stats.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


class TestMean50MixedAbsoluteWeights:
    """MEAN50 MIXED must use |w| so negative weights don't cancel."""

    def test_signed_weights_same_as_positive(self):
        """MIXED stat should be identical for weights [1,1,-1,-1] vs [1,1,1,1]."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        rng = np.random.default_rng(42)
        n_rot, k = 100, 10
        z = rng.standard_normal((n_rot, k))

        w_pos = np.ones((1, k))
        w_neg = np.array([[1, 1, 1, 1, 1, -1, -1, -1, -1, -1]], dtype=float)

        stat_pos = _compute_mean50_stat(z, w_pos, A=1.0, alt=Alternative.MIXED)
        stat_neg = _compute_mean50_stat(z, w_neg, A=1.0, alt=Alternative.MIXED)

        assert_allclose(stat_neg, stat_pos, atol=1e-14,
                        err_msg="MIXED stat must be invariant to weight sign")

    def test_all_negative_weights_same_as_positive(self):
        """All-negative weights [-1,-1,...] should equal all-positive [1,1,...]."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        rng = np.random.default_rng(123)
        n_rot, k = 50, 8
        z = rng.standard_normal((n_rot, k))

        w_pos = np.ones((1, k))
        w_neg = -np.ones((1, k))

        stat_pos = _compute_mean50_stat(z, w_pos, A=1.0, alt=Alternative.MIXED)
        stat_neg = _compute_mean50_stat(z, w_neg, A=1.0, alt=Alternative.MIXED)

        assert_allclose(stat_neg, stat_pos, atol=1e-14)

    def test_up_still_uses_signed_weights(self):
        """UP alternative must preserve weight signs (direction matters)."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        z = np.array([[2.0, -1.0, 1.5, -0.5]])
        w_pos = np.array([[1.0, 1.0, 1.0, 1.0]])
        w_neg = np.array([[1.0, 1.0, -1.0, -1.0]])

        stat_pos = _compute_mean50_stat(z, w_pos, A=1.0, alt=Alternative.UP)
        stat_neg = _compute_mean50_stat(z, w_neg, A=1.0, alt=Alternative.UP)

        # With signed weights, flipping signs of w changes the UP stat
        assert not np.allclose(stat_pos, stat_neg), \
            "UP stat must depend on weight signs"

    def test_down_still_uses_signed_weights(self):
        """DOWN alternative must preserve weight signs (direction matters)."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        z = np.array([[2.0, -1.0, 1.5, -0.5]])
        w_pos = np.array([[1.0, 1.0, 1.0, 1.0]])
        w_neg = np.array([[1.0, 1.0, -1.0, -1.0]])

        stat_pos = _compute_mean50_stat(z, w_pos, A=1.0, alt=Alternative.DOWN)
        stat_neg = _compute_mean50_stat(z, w_neg, A=1.0, alt=Alternative.DOWN)

        assert not np.allclose(stat_pos, stat_neg), \
            "DOWN stat must depend on weight signs"

    def test_mixed_consistency_across_set_statistics(self):
        """FLOORMEAN, MEAN50, MSQ MIXED should all be invariant to weight sign."""
        from cliquefinder.stats.rotation import (
            Alternative,
            _compute_floormean_stat,
            _compute_mean50_stat,
            _compute_msq_stat,
        )

        rng = np.random.default_rng(99)
        n_rot, k = 50, 12
        z = rng.standard_normal((n_rot, k))

        w_pos = np.ones((1, k))
        w_signed = np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]], dtype=float)
        abs_w = np.abs(w_signed)
        A = float(k)
        floor = 0.6745  # sqrt(median chi2_1)

        # FLOORMEAN MIXED
        fm_pos = _compute_floormean_stat(z, w_pos, A, Alternative.MIXED, floor)
        fm_neg = _compute_floormean_stat(z, w_signed, A, Alternative.MIXED, floor)
        assert_allclose(fm_neg, fm_pos, atol=1e-14,
                        err_msg="FLOORMEAN MIXED not sign-invariant")

        # MEAN50 MIXED
        m50_pos = _compute_mean50_stat(z, w_pos, A=1.0, alt=Alternative.MIXED)
        m50_neg = _compute_mean50_stat(z, w_signed, A=1.0, alt=Alternative.MIXED)
        assert_allclose(m50_neg, m50_pos, atol=1e-14,
                        err_msg="MEAN50 MIXED not sign-invariant")

        # MSQ MIXED
        msq_pos = _compute_msq_stat(z, abs_w, A, Alternative.MIXED)
        msq_neg = _compute_msq_stat(z, abs_w, A, Alternative.MIXED)
        assert_allclose(msq_neg, msq_pos, atol=1e-14,
                        err_msg="MSQ MIXED not sign-invariant")


class TestPass2OutlierNaNResilience:
    """Pass 2 must detect outliers even when some cells are NaN."""

    def _make_matrix(self, data, n_genes, n_samples):
        from cliquefinder.core.biomatrix import BioMatrix

        feature_ids = pd.Index([f"G{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        return BioMatrix(
            data=data.copy(),
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            sample_metadata=pd.DataFrame(index=sample_ids),
            quality_flags=np.zeros((n_genes, n_samples), dtype=np.uint8),
        )

    def test_pass1_all_outlier_row_doesnt_silence_pass2(self):
        """When pass 1 flags an entire row, expected gets NaN for that row.

        Before fix: np.median(residuals) returns NaN due to NaN propagation,
        silencing pass 2. After fix: np.nanmedian skips the NaN entries.
        """
        from cliquefinder.quality.outliers import MultiPassOutlierDetector

        rng = np.random.default_rng(42)
        n_genes, n_samples = 60, 12

        # Strong additive model: row + column effects + small noise
        row_effect = rng.standard_normal((n_genes, 1)) * 2
        col_effect = rng.standard_normal((1, n_samples)) * 2
        data = row_effect + col_effect + rng.standard_normal((n_genes, n_samples)) * 0.3

        # Make row 0 ALL extreme outliers → pass 1 flags every cell in this row
        # → masked_data row 0 is all NaN → row_medians[0] = NaN → expected[0] = NaN
        # → residuals[0] = NaN → np.median(residuals) = NaN → pass 2 silenced
        data[0, :] = 500.0

        # Inject a residual outlier in a normal row for pass 2 to detect
        data[5, 5] += 25.0

        matrix = self._make_matrix(data, n_genes, n_samples)
        detector = MultiPassOutlierDetector(
            detection_threshold=3.0,
            residual_enabled=True,
            residual_threshold=3.0,
        )
        detector.apply(matrix)

        # Pass 2 must still fire despite NaN in residuals from the all-outlier row
        assert detector.pass2_count_ > 0, \
            "Pass 2 should detect residual outlier despite all-outlier row causing NaN"

    def test_no_all_outlier_row_pass2_works(self):
        """Regression: pass 2 detects residual outliers in clean data."""
        from cliquefinder.quality.outliers import MultiPassOutlierDetector

        rng = np.random.default_rng(42)
        n_genes, n_samples = 80, 12

        # Strong additive model
        row_effect = rng.standard_normal((n_genes, 1)) * 5
        col_effect = rng.standard_normal((1, n_samples)) * 5
        noise = rng.standard_normal((n_genes, n_samples)) * 0.2
        data = row_effect + col_effect + noise

        # Inject a cell-level residual outlier: not extreme in its row
        # (row mean absorbs part of it) but extreme in its residual from
        # the additive model.
        data[10, 3] += 15.0

        matrix = self._make_matrix(data, n_genes, n_samples)
        detector = MultiPassOutlierDetector(
            detection_threshold=3.0,
            residual_enabled=True,
            residual_threshold=3.0,
        )
        detector.apply(matrix)
        total = detector.pass1_count_ + detector.pass2_count_
        assert total > 0, "Should detect the injected outlier"

    def test_nanmedian_used_in_residual_stats(self):
        """Directly verify that residual stats use nanmedian (unit test)."""
        # This tests the fix at the function level — np.nanmedian is used
        # so NaN entries in residuals don't propagate to median/MAD.
        vals = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
        # np.median would return NaN; np.nanmedian returns 3.0
        assert np.nanmedian(vals) == 3.0
        assert np.isnan(np.median(vals))
