"""
Tests for STAT-5: Q2 sign convention uses contrast projection.

Validates that the Q2 sign correction in ROAST's QR decomposition uses
a contrast-projection method instead of the fragile correlation heuristic.
The projection method is mathematically robust even for balanced designs
where the correlation between Q2[:,0] and the contrast pattern can be
near zero.

Test coverage:
1. Well-separated groups: sign is deterministic
2. Perfectly balanced groups: sign is still deterministic (key improvement)
3. UP/DOWN p-values are consistent across sign flips
4. MSQ (direction-agnostic) is invariant to Q2[:,0] sign
5. General path (compute_rotation_matrices_general) sign correction
6. Near-degenerate case: warning issued when projection is near-zero
7. Regression: p-values match for standard two-group comparison
"""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cliquefinder.stats.rotation import (
    RotationPrecomputed,
    RotationTestConfig,
    RotationTestEngine,
    compute_rotation_matrices,
    compute_rotation_matrices_general,
    compute_set_statistics,
    extract_gene_effects,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def well_separated_design():
    """Unbalanced two-group design: 15 CASE, 5 CTRL."""
    n_case, n_ctrl = 15, 5
    sample_condition = np.array(["CASE"] * n_case + ["CTRL"] * n_ctrl)
    conditions = ["CASE", "CTRL"]
    contrast = ("CASE", "CTRL")
    return sample_condition, conditions, contrast


@pytest.fixture
def balanced_design():
    """Perfectly balanced two-group design: 10 CASE, 10 CTRL."""
    n_per_group = 10
    sample_condition = np.array(["CASE"] * n_per_group + ["CTRL"] * n_per_group)
    conditions = ["CASE", "CTRL"]
    contrast = ("CASE", "CTRL")
    return sample_condition, conditions, contrast


@pytest.fixture
def large_balanced_design():
    """Larger balanced design: 50 per group (correlation heuristic is most
    fragile at large n with perfect balance)."""
    n_per_group = 50
    sample_condition = np.array(["CASE"] * n_per_group + ["CTRL"] * n_per_group)
    conditions = ["CASE", "CTRL"]
    contrast = ("CASE", "CTRL")
    return sample_condition, conditions, contrast


# =============================================================================
# Test 1: Well-separated groups -- sign is deterministic
# =============================================================================


class TestWellSeparatedGroups:
    """Sign should be deterministic for unbalanced designs."""

    def test_sign_deterministic_simple_path(self, well_separated_design):
        """Q2[:,0] sign is deterministic across multiple QR decompositions."""
        sample_condition, conditions, contrast = well_separated_design

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        # The contrast direction in sample space: CASE=+1, CTRL=-1
        contrast_pattern = np.zeros(len(sample_condition))
        contrast_pattern[sample_condition == "CASE"] = 1.0
        contrast_pattern[sample_condition == "CTRL"] = -1.0

        # Q2[:,0] should have positive dot product with contrast pattern
        dot = precomp.Q2[:, 0] @ contrast_pattern
        assert dot > 0, (
            f"Q2[:,0] should align with contrast direction, got dot={dot:.4f}"
        )

    def test_sign_deterministic_general_path(self, well_separated_design):
        """General path also produces deterministic sign."""
        sample_condition, conditions, contrast = well_separated_design

        import statsmodels.api as sm

        # Build design matrix manually (mirroring simple path internals)
        condition_cat = pd.Categorical(sample_condition, categories=conditions)
        df = pd.DataFrame({"condition": condition_cat})
        X_df = pd.get_dummies(df["condition"], drop_first=True, dtype=float)
        X_df = sm.add_constant(X_df)
        X = X_df.values.astype(np.float64)

        # Contrast vector in parameter space
        n_conditions = len(conditions)
        n_params = X.shape[1]
        contrast_vec = np.zeros(n_conditions)
        contrast_vec[conditions.index(contrast[0])] = 1.0
        contrast_vec[conditions.index(contrast[1])] = -1.0

        L = np.zeros((n_conditions, n_params))
        L[:, 0] = 1.0
        for i in range(1, min(n_conditions, n_params)):
            L[i, i] = 1.0
        c = L.T @ contrast_vec

        precomp = compute_rotation_matrices_general(
            design_matrix=X,
            contrast=c,
            contrast_name="CASE_vs_CTRL",
        )

        # Q2[:,0] should align with X @ c
        Xc = X @ c
        dot = precomp.Q2[:, 0] @ Xc
        assert dot > 0, (
            f"General path Q2[:,0] should align with contrast, got dot={dot:.4f}"
        )


# =============================================================================
# Test 2: Perfectly balanced groups -- sign is still deterministic
# =============================================================================


class TestBalancedGroups:
    """This is the KEY improvement: balanced designs should have deterministic
    sign even though the correlation heuristic would yield near-zero correlation."""

    def test_sign_deterministic_balanced_simple(self, balanced_design):
        """Balanced design: Q2[:,0] sign is deterministic."""
        sample_condition, conditions, contrast = balanced_design

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        # With contrast projection, the sign is well-defined
        contrast_pattern = np.zeros(len(sample_condition))
        contrast_pattern[sample_condition == "CASE"] = 1.0
        contrast_pattern[sample_condition == "CTRL"] = -1.0

        dot = precomp.Q2[:, 0] @ contrast_pattern
        # Should be clearly positive (not near zero)
        assert dot > 0.01, (
            f"Balanced design: Q2[:,0] should align with contrast, got dot={dot:.6f}"
        )

    def test_sign_deterministic_large_balanced(self, large_balanced_design):
        """Large balanced design: sign is still deterministic.

        This is the case where the old correlation heuristic was most fragile.
        """
        sample_condition, conditions, contrast = large_balanced_design

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        contrast_pattern = np.zeros(len(sample_condition))
        contrast_pattern[sample_condition == "CASE"] = 1.0
        contrast_pattern[sample_condition == "CTRL"] = -1.0

        dot = precomp.Q2[:, 0] @ contrast_pattern
        assert dot > 0.01, (
            f"Large balanced design: Q2[:,0] should align with contrast, "
            f"got dot={dot:.6f}"
        )

    def test_reproducible_across_runs(self, balanced_design):
        """Sign should be identical across multiple calls (no randomness)."""
        sample_condition, conditions, contrast = balanced_design

        precomp1 = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )
        precomp2 = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        assert_allclose(
            precomp1.Q2[:, 0],
            precomp2.Q2[:, 0],
            atol=1e-12,
            err_msg="Q2 should be identical across calls",
        )


# =============================================================================
# Test 3: UP/DOWN p-values are consistent with known signal direction
# =============================================================================


class TestUpDownConsistency:
    """With correct sign, UP/DOWN p-values should be consistent with the
    known signal direction.  Uses the full RotationTestEngine pipeline."""

    def test_up_pvalue_lower_for_upregulated_set(self, balanced_design):
        """UP-regulated gene set should have lower UP p-value than DOWN."""
        sample_condition, conditions, contrast = balanced_design
        n_samples = len(sample_condition)
        n_genes = 100

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[:10, is_case] += 2.0  # UP signal in first 10 genes

        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": sample_condition})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=conditions,
            contrast=contrast,
            condition_column="condition",
        )

        config = RotationTestConfig(
            n_rotations=999, use_eb=False, use_gpu=False, seed=123,
        )

        up_genes = [f"GENE_{i}" for i in range(10)]
        result = engine.test_gene_set(
            gene_set=up_genes, gene_set_id="up_set", config=config,
        )

        p_up = result.p_values.get("mean", {}).get("up", 1.0)
        p_down = result.p_values.get("mean", {}).get("down", 1.0)

        assert p_up < 0.05, f"UP p-value should be significant, got {p_up:.4f}"
        assert p_down > 0.1, (
            f"DOWN p-value should be non-significant, got {p_down:.4f}"
        )

    def test_down_pvalue_lower_for_downregulated_set(self, balanced_design):
        """DOWN-regulated gene set should have lower DOWN p-value than UP."""
        sample_condition, conditions, contrast = balanced_design
        n_samples = len(sample_condition)
        n_genes = 100

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[10:20, is_case] -= 2.0  # DOWN signal in genes 10-19

        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": sample_condition})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=conditions,
            contrast=contrast,
            condition_column="condition",
        )

        config = RotationTestConfig(
            n_rotations=999, use_eb=False, use_gpu=False, seed=123,
        )

        down_genes = [f"GENE_{i}" for i in range(10, 20)]
        result = engine.test_gene_set(
            gene_set=down_genes, gene_set_id="down_set", config=config,
        )

        p_up = result.p_values.get("mean", {}).get("up", 1.0)
        p_down = result.p_values.get("mean", {}).get("down", 1.0)

        assert p_down < 0.05, (
            f"DOWN p-value should be significant, got {p_down:.4f}"
        )
        assert p_up > 0.1, (
            f"UP p-value should be non-significant, got {p_up:.4f}"
        )


# =============================================================================
# Test 4: MSQ is invariant to Q2[:,0] sign flip
# =============================================================================


class TestMSQInvariance:
    """MSQ (mean-square) statistic should be unaffected by Q2[:,0] sign."""

    def _make_flipped_precomputed(self, precomp):
        """Create a RotationPrecomputed with Q2[:,0] flipped."""
        Q2_flipped = precomp.Q2.copy()
        Q2_flipped[:, 0] = -Q2_flipped[:, 0]
        # RotationPrecomputed is frozen, so we use dataclasses.replace
        return replace(precomp, Q2=Q2_flipped)

    def test_msq_invariant_to_sign_flip(self, balanced_design):
        """Manually flip Q2[:,0] sign and verify MSQ is unchanged."""
        sample_condition, conditions, contrast = balanced_design
        n_samples = len(sample_condition)
        n_genes = 50

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        effects_orig = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp,
        )

        precomp_flipped = self._make_flipped_precomputed(precomp)

        effects_flipped = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp_flipped,
        )

        # Compute z-scores for a subset (first 10 genes) from U[:, 0] / sqrt(var)
        gene_idx = slice(0, 10)
        se_orig = np.sqrt(np.maximum(effects_orig.sample_variances[gene_idx], 1e-20))
        z_orig = effects_orig.U[gene_idx, 0] / se_orig

        se_flip = np.sqrt(np.maximum(effects_flipped.sample_variances[gene_idx], 1e-20))
        z_flip = effects_flipped.U[gene_idx, 0] / se_flip

        # MSQ is computed from z-scores: mean(z^2) -- should be identical
        obs_orig = compute_set_statistics(z_orig)
        obs_flip = compute_set_statistics(z_flip)

        msq_orig = obs_orig["msq"]["mixed"]
        msq_flip = obs_flip["msq"]["mixed"]

        assert_allclose(
            msq_orig,
            msq_flip,
            rtol=1e-10,
            err_msg="MSQ should be invariant to Q2[:,0] sign flip",
        )

    def test_first_column_of_U_negates_on_flip(self, balanced_design):
        """U[:, 0] should negate when Q2[:,0] is flipped."""
        sample_condition, conditions, contrast = balanced_design
        n_samples = len(sample_condition)
        n_genes = 50

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[:10, is_case] += 3.0  # Add signal for a nonzero effect
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        effects_orig = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp,
        )

        precomp_flipped = self._make_flipped_precomputed(precomp)

        effects_flipped = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp_flipped,
        )

        # U[:, 0] is Q2[:,0].T @ y_g -- flipping Q2[:,0] negates this
        assert_allclose(
            effects_orig.U[:, 0],
            -effects_flipped.U[:, 0],
            rtol=1e-10,
            err_msg="U[:, 0] should negate on Q2[:,0] sign flip",
        )

    def test_mean_stat_negates_on_flip(self, balanced_design):
        """The mean UP statistic should negate when Q2[:,0] is flipped."""
        sample_condition, conditions, contrast = balanced_design
        n_samples = len(sample_condition)
        n_genes = 50

        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[:10, is_case] += 3.0
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]

        precomp = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=conditions,
            contrast=contrast,
        )

        effects_orig = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp,
        )

        precomp_flipped = self._make_flipped_precomputed(precomp)

        effects_flipped = extract_gene_effects(
            Y=data, gene_ids=gene_ids, precomputed=precomp_flipped,
        )

        # Get z-scores for the first 10 genes
        gene_idx = slice(0, 10)
        se_orig = np.sqrt(np.maximum(effects_orig.sample_variances[gene_idx], 1e-20))
        z_orig = effects_orig.U[gene_idx, 0] / se_orig

        se_flip = np.sqrt(np.maximum(effects_flipped.sample_variances[gene_idx], 1e-20))
        z_flip = effects_flipped.U[gene_idx, 0] / se_flip

        obs_orig = compute_set_statistics(z_orig)
        obs_flip = compute_set_statistics(z_flip)

        # Mean UP statistic is mean(z) for original, should negate
        mean_up_orig = obs_orig["mean"]["up"]
        mean_up_flip = obs_flip["mean"]["up"]

        assert_allclose(
            mean_up_orig,
            -mean_up_flip,
            rtol=1e-10,
            err_msg="Mean UP statistic should negate on Q2[:,0] sign flip",
        )


# =============================================================================
# Test 5: General path with multi-group design
# =============================================================================


class TestGeneralPathMultiGroup:
    """Test the general path (compute_rotation_matrices_general) with
    designs that go beyond simple two-group."""

    def test_cell_means_interaction_contrast(self):
        """4-group cell-means design with interaction contrast.

        Design: 2 factors (sex x disease), 4 cells.
        Contrast: interaction (M_CASE - M_CTRL) - (F_CASE - F_CTRL).
        """
        n_per_cell = 5
        groups = (
            ["M_CASE"] * n_per_cell
            + ["M_CTRL"] * n_per_cell
            + ["F_CASE"] * n_per_cell
            + ["F_CTRL"] * n_per_cell
        )
        n_samples = len(groups)

        # Cell-means design matrix (4 columns, no intercept)
        X = np.zeros((n_samples, 4))
        for i, g in enumerate(groups):
            col = ["M_CASE", "M_CTRL", "F_CASE", "F_CTRL"].index(g)
            X[i, col] = 1.0

        # Interaction contrast
        c = np.array([1.0, -1.0, -1.0, 1.0])

        precomp = compute_rotation_matrices_general(
            design_matrix=X,
            contrast=c,
            contrast_name="sex_x_disease",
        )

        Xc = X @ c
        dot = precomp.Q2[:, 0] @ Xc
        assert dot > 0, (
            f"Interaction contrast: Q2[:,0] should align with X@c, got dot={dot:.4f}"
        )

    def test_three_group_pairwise_contrast(self):
        """Three-group design with pairwise contrast (A vs C)."""
        n_per_group = 8
        n_samples = 3 * n_per_group

        X = np.zeros((n_samples, 3))
        X[:n_per_group, 0] = 1.0
        X[n_per_group : 2 * n_per_group, 1] = 1.0
        X[2 * n_per_group :, 2] = 1.0

        c = np.array([1.0, 0.0, -1.0])

        precomp = compute_rotation_matrices_general(
            design_matrix=X,
            contrast=c,
            contrast_name="A_vs_C",
        )

        Xc = X @ c
        dot = precomp.Q2[:, 0] @ Xc
        assert dot > 0, (
            f"Three-group contrast: Q2[:,0] should align with X@c, got dot={dot:.4f}"
        )

    def test_general_path_balanced_cell_means(self):
        """Perfectly balanced cell-means should still have deterministic sign."""
        n_per_cell = 10
        groups = (
            ["A"] * n_per_cell
            + ["B"] * n_per_cell
            + ["C"] * n_per_cell
            + ["D"] * n_per_cell
        )
        n_samples = len(groups)

        X = np.zeros((n_samples, 4))
        for i, g in enumerate(groups):
            col = ["A", "B", "C", "D"].index(g)
            X[i, col] = 1.0

        c = np.array([0.5, 0.5, -0.5, -0.5])

        precomp = compute_rotation_matrices_general(
            design_matrix=X,
            contrast=c,
            contrast_name="AB_vs_CD",
        )

        Xc = X @ c
        dot = precomp.Q2[:, 0] @ Xc
        assert dot > 0.01, (
            f"Balanced 4-group: Q2[:,0] should align with X@c, got dot={dot:.6f}"
        )


# =============================================================================
# Test 6: Near-degenerate case -- warning issued
# =============================================================================


class TestNearDegenerateWarning:
    """When the contrast projection is near-zero, a warning should be issued."""

    def test_general_path_degenerate_contrast_warns(self):
        """A contrast nearly in the reduced model's column space should warn.

        We construct a scenario where X @ c is almost entirely within the
        column space of the reparameterized reduced model, making the
        projection onto Q2[:,0] near-zero.
        """
        # Strategy: make the C-matrix reparameterization place Xc almost
        # entirely within Q1.  We do this with a design where the contrast
        # column is nearly a linear combination of other columns.
        n = 20
        rng = np.random.default_rng(99)

        # Intercept + covariate + near-duplicate of covariate
        X = np.ones((n, 3))
        X[:, 1] = np.linspace(0, 1, n)
        # Column 2 is column 1 with tiny noise, making them nearly collinear
        X[:, 2] = X[:, 1] + rng.normal(0, 1e-14, n)

        # Contrast on the nearly-redundant third parameter
        c = np.array([0.0, 0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_rotation_matrices_general(
                design_matrix=X,
                contrast=c,
                contrast_name="degenerate",
            )

            relevant = [
                x for x in w if "near-zero" in str(x.message).lower()
            ]
            assert len(relevant) >= 1, (
                f"Expected near-zero projection warning, got warnings: "
                f"{[str(x.message) for x in w]}"
            )

    def test_no_warning_for_valid_contrast(self, balanced_design):
        """A valid contrast should NOT trigger the near-zero warning."""
        sample_condition, conditions, contrast = balanced_design

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_rotation_matrices(
                sample_condition=sample_condition,
                conditions=conditions,
                contrast=contrast,
            )

            relevant = [
                x for x in w if "near-zero" in str(x.message).lower()
            ]
            assert len(relevant) == 0, (
                f"Valid contrast should not trigger warning, got: "
                f"{[str(x.message) for x in relevant]}"
            )


# =============================================================================
# Test 7: Regression -- standard two-group p-values are consistent
# =============================================================================


class TestRegressionTwoGroup:
    """Ensure the fix does not change p-values for standard two-group designs
    where the old correlation heuristic gave the correct answer."""

    def test_pvalues_match_expected_direction(self):
        """For a clear signal, UP/DOWN p-values have the expected ordering."""
        rng = np.random.default_rng(42)
        n_per_group = 10
        n_samples = 2 * n_per_group
        n_genes = 80

        sample_condition = np.array(
            ["CASE"] * n_per_group + ["CTRL"] * n_per_group
        )
        conditions = ["CASE", "CTRL"]
        contrast = ("CASE", "CTRL")

        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[:10, is_case] += 3.0  # Strong UP signal

        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": sample_condition})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=conditions,
            contrast=contrast,
            condition_column="condition",
        )

        config = RotationTestConfig(
            n_rotations=999, use_eb=False, use_gpu=False, seed=123,
        )

        up_genes = [f"GENE_{i}" for i in range(10)]
        result = engine.test_gene_set(
            gene_set=up_genes, gene_set_id="up_set", config=config,
        )

        p_up = result.p_values.get("mean", {}).get("up", 1.0)
        p_down = result.p_values.get("mean", {}).get("down", 1.0)
        p_mixed = result.p_values.get("msq", {}).get("mixed", 1.0)

        assert p_up < p_down, (
            f"UP p-value ({p_up:.4f}) should be smaller than DOWN ({p_down:.4f}) "
            "for upregulated genes"
        )
        assert p_up < 0.05, f"UP p-value should be significant, got {p_up:.4f}"
        assert p_mixed < 0.05, f"MSQ p-value should be significant, got {p_mixed:.4f}"

    def test_null_genes_not_significant(self):
        """Gene set with null genes should not be significant."""
        rng = np.random.default_rng(42)
        n_per_group = 10
        n_samples = 2 * n_per_group
        n_genes = 80

        sample_condition = np.array(
            ["CASE"] * n_per_group + ["CTRL"] * n_per_group
        )
        conditions = ["CASE", "CTRL"]
        contrast = ("CASE", "CTRL")

        data = rng.standard_normal((n_genes, n_samples))  # Pure noise
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": sample_condition})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=conditions,
            contrast=contrast,
            condition_column="condition",
        )

        config = RotationTestConfig(
            n_rotations=999, use_eb=False, use_gpu=False, seed=456,
        )

        null_genes = [f"GENE_{i}" for i in range(10)]
        result = engine.test_gene_set(
            gene_set=null_genes, gene_set_id="null_set", config=config,
        )

        p_mixed = result.p_values.get("msq", {}).get("mixed", 1.0)
        assert p_mixed > 0.01, (
            f"Null genes should not be significant, got p_mixed={p_mixed:.4f}"
        )


# =============================================================================
# Test: Contrast projection is mathematically non-zero for valid contrasts
# =============================================================================


class TestProjectionNonZero:
    """Verify that the projection Q2[:,0] @ (X @ c) is non-trivially nonzero
    for valid contrasts, confirming the method is mathematically sound."""

    def test_projection_magnitude_simple_path(self, balanced_design):
        """The projection should have substantial magnitude (not near-zero)
        for a standard two-group contrast."""
        sample_condition, conditions, contrast = balanced_design

        import statsmodels.api as sm

        condition_cat = pd.Categorical(sample_condition, categories=conditions)
        df = pd.DataFrame({"condition": condition_cat})
        X_df = pd.get_dummies(df["condition"], drop_first=True, dtype=float)
        X_df = sm.add_constant(X_df)
        X = X_df.values.astype(np.float64)

        n_conditions = len(conditions)
        n_params = X.shape[1]
        contrast_vec = np.zeros(n_conditions)
        contrast_vec[conditions.index(contrast[0])] = 1.0
        contrast_vec[conditions.index(contrast[1])] = -1.0

        L = np.zeros((n_conditions, n_params))
        L[:, 0] = 1.0
        for i in range(1, min(n_conditions, n_params)):
            L[i, i] = 1.0
        c = L.T @ contrast_vec

        Q, R = np.linalg.qr(X, mode="complete")
        Q2_col0 = Q[:, n_params - 1]
        Xc = X @ c
        proj = Q2_col0 @ Xc

        assert abs(proj) > 0.1, (
            f"Projection should be substantially nonzero for valid contrast, "
            f"got |proj|={abs(proj):.6f}"
        )

    def test_projection_magnitude_general_path(self):
        """General path projection is nonzero for a 3-group contrast."""
        from cliquefinder.stats.rotation import _construct_c_matrix

        n_per_group = 10
        n = 3 * n_per_group
        X = np.zeros((n, 3))
        X[:n_per_group, 0] = 1.0
        X[n_per_group : 2 * n_per_group, 1] = 1.0
        X[2 * n_per_group :, 2] = 1.0

        c = np.array([1.0, -0.5, -0.5])

        C = _construct_c_matrix(c)
        C_inv = np.linalg.inv(C)
        X_reparam = X @ C_inv
        Q, R = np.linalg.qr(X_reparam, mode="complete")
        n_params = X.shape[1]
        Q2_col0 = Q[:, n_params - 1]

        Xc = X @ c
        proj = Q2_col0 @ Xc

        assert abs(proj) > 0.1, (
            f"General path projection should be nonzero, got |proj|={abs(proj):.6f}"
        )


# =============================================================================
# Test: Engine-level integration
# =============================================================================


class TestEngineIntegration:
    """End-to-end tests through RotationTestEngine to verify the fix
    works in the actual pipeline."""

    def test_engine_balanced_design_produces_valid_pvalues(self):
        """Run full engine pipeline on balanced design with known signal."""
        rng = np.random.default_rng(42)
        n_per_group = 15
        n_samples = 2 * n_per_group
        n_genes = 60

        sample_condition = np.array(
            ["CASE"] * n_per_group + ["CTRL"] * n_per_group
        )
        conditions = ["CASE", "CTRL"]
        contrast = ("CASE", "CTRL")

        data = rng.standard_normal((n_genes, n_samples))
        is_case = sample_condition == "CASE"
        data[:10, is_case] += 2.5

        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": sample_condition})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=conditions,
            contrast=contrast,
            condition_column="condition",
        )

        config = RotationTestConfig(
            n_rotations=499, use_eb=False, use_gpu=False, seed=42,
        )

        up_genes = [f"GENE_{i}" for i in range(10)]
        result = engine.test_gene_set(
            gene_set=up_genes, gene_set_id="up_set", config=config,
        )

        p_up = result.p_values.get("mean", {}).get("up", 1.0)
        p_down = result.p_values.get("mean", {}).get("down", 1.0)

        assert p_up < p_down, (
            f"Engine: UP p-value ({p_up:.4f}) should be < DOWN ({p_down:.4f}) "
            "for upregulated gene set"
        )

    def test_engine_general_path_interaction(self):
        """Test engine with general path (fit_general) for interaction contrast."""
        rng = np.random.default_rng(42)
        n_per_cell = 8
        n_samples = 4 * n_per_cell
        n_genes = 60

        groups = (
            ["M_CASE"] * n_per_cell
            + ["M_CTRL"] * n_per_cell
            + ["F_CASE"] * n_per_cell
            + ["F_CTRL"] * n_per_cell
        )

        # Cell-means design
        X = np.zeros((n_samples, 4))
        for i, g in enumerate(groups):
            col = ["M_CASE", "M_CTRL", "F_CASE", "F_CTRL"].index(g)
            X[i, col] = 1.0

        # Interaction contrast
        c = np.array([1.0, -1.0, -1.0, 1.0])

        data = rng.standard_normal((n_genes, n_samples))
        # Add interaction signal to first 10 genes
        for i in range(10):
            data[i, :n_per_cell] += 1.5       # M_CASE up
            data[i, n_per_cell:2*n_per_cell] -= 1.5  # M_CTRL down
            data[i, 2*n_per_cell:3*n_per_cell] -= 1.5  # F_CASE down
            data[i, 3*n_per_cell:] += 1.5     # F_CTRL up

        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"group": groups})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit_general(
            design_matrix=X,
            contrast=c,
            contrast_name="sex_x_disease",
        )

        config = RotationTestConfig(
            n_rotations=499, use_eb=False, use_gpu=False, seed=42,
        )

        signal_genes = [f"GENE_{i}" for i in range(10)]
        result = engine.test_gene_set(
            gene_set=signal_genes, gene_set_id="interaction_set", config=config,
        )

        p_mixed = result.p_values.get("msq", {}).get("mixed", 1.0)
        assert p_mixed < 0.05, (
            f"Interaction signal should be detected, got p_mixed={p_mixed:.4f}"
        )
