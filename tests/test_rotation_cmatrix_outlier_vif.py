"""
Tests for ROAST C-matrix reparameterization, multi-pass outlier detection,
and competitive z-score VIF consistency.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cliquefinder.stats.rotation import (
    _construct_c_matrix,
    compute_rotation_matrices_general,
)


class TestCMatrixReparameterization:
    """X* = X @ C must place the contrast direction in Q2[:,0].

    The QR decomposition of the reparameterized design matrix X* assigns
    Q2[:,0] to the direction of X*'s last column. Since C[:,-1] = c_unit,
    we need X*[:,-1] = X @ c_unit (the contrast in sample space). This
    requires X* = X @ C, not X @ C⁻¹.

    For p=2, C is symmetric (2×2 orthogonal), so XC = XC⁻¹ and both are
    correct. The distinction matters only for p >= 3.
    """

    def test_two_group_smoke(self):
        """Two-group design produces Q2 with correct shape."""
        rng = np.random.default_rng(42)
        n_samples, p = 20, 2
        X = np.column_stack([np.ones(n_samples), rng.choice([0, 1], n_samples)])
        contrast = np.array([0.0, 1.0])

        precomp = compute_rotation_matrices_general(X, contrast)
        assert precomp.Q2.shape == (n_samples, n_samples - p + 1)

    def test_three_group_contrast_alignment(self):
        """For p=3, Q2[:,0] must project strongly onto X @ contrast."""
        rng = np.random.default_rng(42)
        n_samples = 30
        group = np.repeat([0, 1, 2], 10)
        X = np.column_stack([
            np.ones(n_samples),
            (group == 1).astype(float),
            (group == 2).astype(float),
        ])
        contrast = np.array([0.0, 1.0, -1.0])

        precomp = compute_rotation_matrices_general(X, contrast)
        Xc = X @ contrast
        Xc_unit = Xc / np.linalg.norm(Xc)

        proj = abs(precomp.Q2[:, 0] @ Xc_unit)
        assert proj > 0.9, (
            f"Q2[:,0] projection onto X@c = {proj:.4f}, expected > 0.9"
        )

    def test_four_group_contrast_alignment(self):
        """Generalizes to p=4 with a composite contrast."""
        rng = np.random.default_rng(42)
        n_samples = 40
        group = np.repeat([0, 1, 2, 3], 10)
        X = np.column_stack([
            np.ones(n_samples),
            (group == 1).astype(float),
            (group == 2).astype(float),
            (group == 3).astype(float),
        ])
        contrast = np.array([0.0, 0.5, 0.5, -1.0])

        precomp = compute_rotation_matrices_general(X, contrast)
        Xc = X @ contrast
        Xc_unit = Xc / np.linalg.norm(Xc)
        proj = abs(precomp.Q2[:, 0] @ Xc_unit)
        assert proj > 0.9, f"p=4 projection {proj:.4f} too low"

    def test_c_matrix_last_column_is_normalized_contrast(self):
        """C[:,-1] must equal c / ||c||."""
        contrast = np.array([0.0, 1.0, -1.0])
        C = _construct_c_matrix(contrast)
        c_unit = contrast / np.linalg.norm(contrast)
        assert_allclose(C[:, -1], c_unit, atol=1e-12)

    def test_c_matrix_is_orthogonal(self):
        """C'C = I for any contrast."""
        contrast = np.array([0.0, 1.0, -1.0])
        C = _construct_c_matrix(contrast)
        assert_allclose(C.T @ C, np.eye(3), atol=1e-12)

    def test_generic_c_matrix_not_symmetric(self):
        """For generic p>=3 contrast, C ≠ C' — so XC ≠ XC⁻¹."""
        contrast = np.array([1.0, 1.0, -2.0])
        C = _construct_c_matrix(contrast)
        diff = np.max(np.abs(C - C.T))
        assert diff > 0.1, (
            f"C appears symmetric (max|C-C'|={diff:.6f}), "
            f"but generic p>=3 contrasts should produce asymmetric C"
        )


class TestMultiPassOutlierDetectorResidualPass:
    """Multi-pass outlier detection: residual-based (pass 2) functionality."""

    def test_residual_pass_executes_without_error(self):
        """Pass 2 uses pass-1 outlier mask to exclude flagged values."""
        from cliquefinder.quality.outliers import MultiPassOutlierDetector
        from cliquefinder.core.biomatrix import BioMatrix

        rng = np.random.default_rng(42)
        n_genes, n_samples = 50, 10
        data = rng.standard_normal((n_genes, n_samples))
        data[0, 0] = 100.0
        data[1, 1] = -100.0

        feature_ids = pd.Index([f"G{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        matrix = BioMatrix(
            data=data.copy(),
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            sample_metadata=pd.DataFrame(index=sample_ids),
            quality_flags=np.zeros((n_genes, n_samples), dtype=np.uint8),
        )

        detector = MultiPassOutlierDetector(
            detection_threshold=3.0,
            residual_enabled=True,
            residual_threshold=4.0,
        )
        result = detector.apply(matrix)
        assert result is not None
        assert detector.pass1_count_ >= 0
        assert detector.pass2_count_ >= 0

    def test_residual_pass_detects_additive_model_outliers(self):
        """Residual pass catches deviations from row+column additive model."""
        from cliquefinder.quality.outliers import MultiPassOutlierDetector
        from cliquefinder.core.biomatrix import BioMatrix

        rng = np.random.default_rng(42)
        n_genes, n_samples = 50, 10
        row_effect = rng.standard_normal((n_genes, 1)) * 2
        col_effect = rng.standard_normal((1, n_samples)) * 2
        data = row_effect + col_effect + rng.standard_normal((n_genes, n_samples)) * 0.5
        data[5, 5] += 20.0  # Large additive-model residual

        feature_ids = pd.Index([f"G{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        matrix = BioMatrix(
            data=data.copy(),
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            sample_metadata=pd.DataFrame(index=sample_ids),
            quality_flags=np.zeros((n_genes, n_samples), dtype=np.uint8),
        )

        detector = MultiPassOutlierDetector(
            detection_threshold=3.0,
            residual_enabled=True,
            residual_threshold=3.0,
        )
        detector.apply(matrix)
        total = detector.pass1_count_ + detector.pass2_count_
        assert total > 0, "Should detect at least one outlier"


class TestCompetitiveZVIFConsistency:
    """Competitive z-score VIF behavior: deflation, symmetry, identity at zero."""

    def test_vif_deflates_z(self):
        """Positive inter-gene correlation inflates SE, reducing |z|."""
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        rng = np.random.default_rng(42)
        t_stats = rng.standard_normal(200)
        t_stats[:20] += 1.5
        mask = np.zeros(200, dtype=bool)
        mask[:20] = True

        z_raw = compute_competitive_z(t_stats, mask)
        z_vif = compute_competitive_z(t_stats, mask, inter_gene_correlation=0.3)

        assert abs(z_vif) < abs(z_raw), (
            f"VIF should reduce |z|: raw={z_raw:.3f}, vif={z_vif:.3f}"
        )

    def test_asymmetric_vif_creates_conservative_bias(self):
        """Applying VIF to target but not controls deflates target z,
        creating systematic false negatives under the null.
        """
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        rng = np.random.default_rng(42)
        n_features, k, n_trials = 500, 20, 200

        biased_count = 0
        for _ in range(n_trials):
            t_stats = rng.standard_normal(n_features)
            mask = np.zeros(n_features, dtype=bool)
            mask[rng.choice(n_features, k, replace=False)] = True

            z_fair = compute_competitive_z(t_stats, mask)
            z_biased = compute_competitive_z(
                t_stats, mask, inter_gene_correlation=0.3
            )
            if abs(z_biased) > abs(z_fair):
                biased_count += 1

        assert biased_count < n_trials * 0.05, (
            f"VIF-biased z more extreme in {biased_count}/{n_trials} trials"
        )


class TestWeightedSignCorrection:
    """Sign correction in QR decomposition with sample weights."""

    def test_weighted_q2_aligns_with_weighted_contrast(self):
        """Q2[:,0] should align with W^{1/2} @ X @ c when weights are present."""
        rng = np.random.default_rng(42)
        n_samples = 30
        group = np.repeat([0, 1, 2], 10)
        X = np.column_stack([
            np.ones(n_samples),
            (group == 1).astype(float),
            (group == 2).astype(float),
        ])
        contrast = np.array([0.0, 1.0, -1.0])
        weights = rng.uniform(0.5, 2.0, n_samples)

        precomp = compute_rotation_matrices_general(
            X, contrast, sample_weights=weights
        )
        W_sqrt = np.sqrt(np.diag(weights))
        Xc_weighted = W_sqrt @ X @ contrast
        Xc_unit = Xc_weighted / np.linalg.norm(Xc_weighted)

        proj = abs(precomp.Q2[:, 0] @ Xc_unit)
        assert proj > 0.8, f"Weighted projection {proj:.4f} too low"
