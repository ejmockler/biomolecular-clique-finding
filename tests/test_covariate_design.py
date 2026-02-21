"""Tests for covariate-aware design matrix construction."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.design_matrix import (
    CovariateDesign,
    build_covariate_design_matrix,
    pad_contrast_for_covariates,
)


class TestBuildCovariateDesignMatrix:
    """Tests for build_covariate_design_matrix."""

    def test_no_covariates_backward_compatible(self):
        """Without covariates, should produce standard dummy-coded design."""
        condition = np.array(["A", "A", "A", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=None
        )

        # Should have intercept + 1 dummy = 2 columns
        assert design.X.shape == (6, 2)
        assert design.n_condition_params == 2
        assert design.n_covariate_params == 0
        assert len(design.covariate_cols) == 0
        assert design.contrast_name == "B_vs_A"
        assert np.all(design.sample_mask)
        assert design.df_residual == 4

    def test_categorical_covariate(self):
        """Categorical covariate should be dummy-coded."""
        condition = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({"Sex": ["M", "F", "M", "F", "M", "F", "M", "F"]})

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # intercept + 1 condition dummy + 1 sex dummy = 3 columns
        assert design.X.shape[1] == 3
        assert design.n_condition_params == 2
        assert design.n_covariate_params == 1
        assert len(design.covariate_cols) == 1
        assert design.covariate_cols[0] == 2  # After condition columns

    def test_numeric_covariate_standardized(self):
        """Numeric covariate should be standardized."""
        condition = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        age_values = np.array([20.0, 30.0, 40.0, 50.0, 25.0, 35.0, 45.0, 55.0])
        covariates = pd.DataFrame({"Age": age_values})

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # intercept + 1 condition dummy + 1 numeric = 3 columns
        assert design.X.shape[1] == 3
        assert design.n_covariate_params == 1

        # Covariate column should be approximately standardized
        cov_col = design.X[:, design.covariate_cols[0]]
        assert abs(np.mean(cov_col)) < 0.01
        assert abs(np.std(cov_col, ddof=1) - 1.0) < 0.01

    def test_contrast_zero_padded_for_covariates(self):
        """Contrast vector should have zeros for covariate columns."""
        condition = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({"Sex": ["M", "F", "M", "F", "M", "F", "M", "F"]})

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # Contrast should be zero for covariate columns
        for idx in design.covariate_cols:
            assert design.contrast[idx] == 0.0

        # Contrast should be non-zero for condition columns
        condition_part = design.contrast[: design.n_condition_params]
        assert np.any(condition_part != 0)

    def test_full_rank_validation(self):
        """Should raise if design matrix is rank-deficient."""
        # Create a covariate perfectly correlated with condition
        condition = np.array(["A", "A", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        # Covariate has same grouping as condition — collinear
        covariates = pd.DataFrame({"bad_cov": ["X", "X", "Y", "Y"]})

        with pytest.raises(ValueError, match="rank-deficient"):
            build_covariate_design_matrix(
                condition, conditions, contrast, covariates_df=covariates
            )

    def test_nan_in_covariates_excluded(self):
        """Samples with NaN covariates should be excluded."""
        condition = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame(
            {"Sex": ["M", "F", "M", np.nan, "M", "F", "M", "F"]}
        )

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # One sample excluded
        assert design.X.shape[0] == 7
        assert np.sum(design.sample_mask) == 7
        assert not design.sample_mask[3]  # NaN sample excluded

    def test_nan_in_condition_excluded(self):
        """Samples with NaN condition should be excluded."""
        condition = np.array(["A", "A", np.nan, "A", "B", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=None
        )

        assert design.X.shape[0] == 7
        assert not design.sample_mask[2]

    def test_multiple_covariates(self):
        """Multiple covariates should all be included."""
        n = 20
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame(
            {
                "Sex": np.random.choice(["M", "F"], n),
                "Batch": np.random.choice(["B1", "B2", "B3"], n),
                "Age": np.random.randn(n) * 10 + 50,
            }
        )

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # intercept + 1 condition + 1 sex + 2 batch + 1 age = 6
        assert design.X.shape[1] == 6
        assert design.n_covariate_params == 4  # 1 sex + 2 batch + 1 age
        assert len(design.contrast) == 6
        # All covariate positions should be zero
        for idx in design.covariate_cols:
            assert design.contrast[idx] == 0.0

    def test_three_conditions(self):
        """Should work with >2 conditions."""
        condition = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
        conditions = ["A", "B", "C"]
        contrast = ("B", "A")

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=None
        )

        # intercept + 2 dummies = 3 columns
        assert design.X.shape == (15, 3)
        assert design.n_condition_params == 3

    def test_three_conditions_with_covariates(self):
        """Three conditions + covariates should work."""
        n = 30
        condition = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        conditions = ["A", "B", "C"]
        contrast = ("B", "A")
        covariates = pd.DataFrame(
            {"Sex": np.tile(["M", "F"], n // 2)}
        )

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=covariates
        )

        # intercept + 2 condition dummies + 1 sex dummy = 4 columns
        assert design.X.shape[1] == 4
        assert design.n_condition_params == 3
        assert design.n_covariate_params == 1

    def test_insufficient_samples_raises(self):
        """Should raise with too few samples."""
        condition = np.array(["A", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")

        with pytest.raises(ValueError, match="Insufficient"):
            build_covariate_design_matrix(
                condition, conditions, contrast, covariates_df=None
            )

    def test_zero_variance_covariate_raises(self):
        """Constant numeric covariate should raise."""
        condition = np.array(["A", "A", "A", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]})

        with pytest.raises(ValueError, match="zero variance"):
            build_covariate_design_matrix(
                condition, conditions, contrast, covariates_df=covariates
            )

    def test_contrast_invalid_condition_raises(self):
        """Should raise if contrast references unknown condition."""
        condition = np.array(["A", "A", "A", "B", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("C", "A")

        with pytest.raises(ValueError, match="not found"):
            build_covariate_design_matrix(
                condition, conditions, contrast, covariates_df=None
            )

    def test_covariates_df_length_mismatch_raises(self):
        """Should raise if covariates_df has wrong number of rows."""
        condition = np.array(["A", "A", "B", "B"])
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({"Sex": ["M", "F", "M"]})  # 3 != 4

        with pytest.raises(ValueError, match="rows"):
            build_covariate_design_matrix(
                condition, conditions, contrast, covariates_df=covariates
            )


class TestInteractionTerms:
    """Tests for interaction_terms parameter (M-7)."""

    def test_interaction_correct_column_count(self):
        """2x2 interaction design: correct column count and names."""
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({
            "Sex": np.tile(["M", "F"], 10),
        })

        design = build_covariate_design_matrix(
            condition, conditions, contrast,
            covariates_df=covariates, interaction_terms=True,
        )

        # intercept + 1 cond dummy + 1 sex dummy + 1 interaction = 4 columns
        assert design.X.shape[1] == 4
        assert any(":" in name for name in design.col_names)

    def test_interaction_contrast_zero_padded(self):
        """Contrast vector has zeros for interaction columns."""
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({
            "Sex": np.tile(["M", "F"], 10),
        })

        design = build_covariate_design_matrix(
            condition, conditions, contrast,
            covariates_df=covariates, interaction_terms=True,
        )

        # All non-condition columns should be zero in contrast
        for idx in range(design.n_condition_params, design.n_params):
            assert design.contrast[idx] == 0.0

    def test_df_guard_error_on_low_df(self):
        """ValueError when residual df < 10 with interaction terms."""
        # 12 samples - intercept(1) - cond(1) - sex(1) - interaction(1) = 8 df
        condition = np.array(["A"] * 6 + ["B"] * 6)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({
            "Sex": np.tile(["M", "F"], 6),
        })

        with pytest.raises(ValueError, match="Residual df too low"):
            build_covariate_design_matrix(
                condition, conditions, contrast,
                covariates_df=covariates, interaction_terms=True,
            )

    def test_backward_compat_no_interaction(self):
        """interaction_terms=False → identical to pre-change."""
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({
            "Sex": np.tile(["M", "F"], 10),
        })

        design_without = build_covariate_design_matrix(
            condition, conditions, contrast,
            covariates_df=covariates, interaction_terms=False,
        )
        design_default = build_covariate_design_matrix(
            condition, conditions, contrast,
            covariates_df=covariates,
        )

        np.testing.assert_array_equal(design_without.X, design_default.X)
        np.testing.assert_array_equal(design_without.contrast, design_default.contrast)

    def test_three_conditions_interaction(self):
        """Three conditions + covariates + interaction terms."""
        n = 45
        condition = np.array(["A"] * 15 + ["B"] * 15 + ["C"] * 15)
        conditions = ["A", "B", "C"]
        contrast = ("B", "A")
        covariates = pd.DataFrame({
            "Sex": np.tile(["M", "F", "M"], 15),
        })

        design = build_covariate_design_matrix(
            condition, conditions, contrast,
            covariates_df=covariates, interaction_terms=True,
        )

        # intercept + 2 cond dummies + 1 sex + 2 interactions = 6 columns
        assert design.X.shape[1] == 6
        interaction_names = [n for n in design.col_names if ":" in n]
        assert len(interaction_names) == 2


class TestCovariateDesignWithDifferential:
    """Tests for M-6: CovariateDesign passed to run_protein_differential."""

    def test_covariate_design_matches_covariates_df(self):
        """run_protein_differential with CovariateDesign matches covariates_df path."""
        rng = np.random.default_rng(42)
        n_features, n_samples = 50, 20
        data = rng.normal(0, 1, size=(n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        cov_df = pd.DataFrame({"Sex": np.tile(["M", "F"], 10)})

        from cliquefinder.stats.differential import run_protein_differential

        # Without CovariateDesign
        result_without = run_protein_differential(
            data=data, feature_ids=feature_ids,
            sample_condition=condition, contrast=contrast,
            covariates_df=cov_df, verbose=False,
        )

        # With CovariateDesign
        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=cov_df,
        )
        result_with = run_protein_differential(
            data=data, feature_ids=feature_ids,
            sample_condition=condition, contrast=contrast,
            covariates_df=cov_df, covariate_design=design, verbose=False,
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result_without["t_statistic"].values,
            result_with["t_statistic"].values,
        )

    def test_covariate_design_nan_mask_used(self):
        """CovariateDesign.sample_mask is used as authoritative NaN mask."""
        rng = np.random.default_rng(42)
        n_features, n_samples = 50, 20
        data = rng.normal(0, 1, size=(n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")

        # Introduce NaN in covariate
        cov_df = pd.DataFrame({"Sex": ["M", "F"] * 9 + [np.nan, "F"]})

        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=cov_df,
        )
        # Verify NaN sample is excluded
        assert not design.sample_mask[18]
        assert design.n_samples == 19

        from cliquefinder.stats.differential import run_protein_differential

        result = run_protein_differential(
            data=data, feature_ids=feature_ids,
            sample_condition=condition, contrast=contrast,
            covariates_df=cov_df, covariate_design=design, verbose=False,
        )

        # Should succeed and use 19 samples
        assert result["n_samples"].iloc[0] == 19


class TestPrecomputeWithCovariateDesign:
    """Tests for H6: precompute_ols_matrices() with CovariateDesign pass-through."""

    def test_precompute_with_covariate_design(self):
        """Pass CovariateDesign to precompute_ols_matrices(), verify it uses X directly."""
        from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        cov_df = pd.DataFrame({"Sex": np.tile(["M", "F"], 10)})

        # Build CovariateDesign
        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=cov_df,
        )

        # Pass design directly to precompute_ols_matrices
        matrices = precompute_ols_matrices(
            covariate_design=design,
            conditions=conditions,
        )

        # Verify it used the design's X matrix (shape should match exactly)
        assert matrices.X.shape == design.X.shape
        np.testing.assert_array_equal(matrices.X, design.X)

        # Verify it used the design's contrast vector
        np.testing.assert_array_equal(matrices.c, design.contrast)

        # Verify contrast name comes from design
        assert matrices.contrast_name == design.contrast_name

        # Verify df_residual is consistent
        assert matrices.df_residual == design.df_residual

        # Verify XtX_inv and c_var_factor are properly computed
        assert matrices.XtX_inv.shape == (design.n_params, design.n_params)
        assert matrices.c_var_factor > 0

    def test_precompute_without_covariate_design_unchanged(self):
        """No covariate_design: identical behavior to before (backward compat)."""
        from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

        condition = np.array(["A"] * 10 + ["B"] * 10)
        conditions = ["A", "B"]
        contrast = ("B", "A")
        cov_df = pd.DataFrame({"Sex": np.tile(["M", "F"], 10)})

        # Without CovariateDesign (old path via covariates_df)
        matrices_old = precompute_ols_matrices(
            sample_condition=condition,
            conditions=conditions,
            contrast=contrast,
            covariates_df=cov_df,
        )

        # With CovariateDesign (new consolidated path)
        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=cov_df,
        )
        matrices_new = precompute_ols_matrices(
            covariate_design=design,
            conditions=conditions,
        )

        # Both paths should produce identical matrices
        np.testing.assert_array_almost_equal(matrices_old.X, matrices_new.X)
        np.testing.assert_array_almost_equal(matrices_old.c, matrices_new.c)
        np.testing.assert_array_almost_equal(matrices_old.XtX_inv, matrices_new.XtX_inv)
        assert abs(matrices_old.c_var_factor - matrices_new.c_var_factor) < 1e-10
        assert matrices_old.df_residual == matrices_new.df_residual
        assert matrices_old.contrast_name == matrices_new.contrast_name

    def test_differential_with_covariate_design_consistent(self):
        """run_protein_differential with and without CovariateDesign yield same results."""
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.default_rng(123)
        n_features, n_samples = 80, 24
        data = rng.normal(0, 1, size=(n_features, n_samples))
        # Add a condition effect for some features
        data[:10, :12] += 0.5
        feature_ids = [f"protein_{i}" for i in range(n_features)]
        condition = np.array(["ctrl"] * 12 + ["treat"] * 12)
        conditions = ["ctrl", "treat"]
        contrast = ("treat", "ctrl")
        cov_df = pd.DataFrame({"Sex": np.tile(["M", "F"], 12)})

        # Path 1: covariates_df only (no CovariateDesign)
        result_without = run_protein_differential(
            data=data, feature_ids=feature_ids,
            sample_condition=condition, contrast=contrast,
            covariates_df=cov_df, verbose=False,
        )

        # Path 2: pre-built CovariateDesign passed through
        design = build_covariate_design_matrix(
            condition, conditions, contrast, covariates_df=cov_df,
        )
        result_with = run_protein_differential(
            data=data, feature_ids=feature_ids,
            sample_condition=condition, contrast=contrast,
            covariates_df=cov_df, covariate_design=design, verbose=False,
        )

        # t-statistics should be identical (same design matrix, same data)
        np.testing.assert_array_almost_equal(
            result_without["t_statistic"].values,
            result_with["t_statistic"].values,
            decimal=10,
        )

        # log2fc should also match
        np.testing.assert_array_almost_equal(
            result_without["log2fc"].values,
            result_with["log2fc"].values,
            decimal=10,
        )

        # p-values should match
        np.testing.assert_array_almost_equal(
            result_without["p_value"].values,
            result_with["p_value"].values,
            decimal=10,
        )


class TestImbalancedGroupsWarning:
    """Tests for Satterthwaite df imbalance warning (M4)."""

    def test_imbalanced_groups_warning(self):
        """Highly imbalanced groups (5 vs 35) should emit UserWarning."""
        import warnings
        from cliquefinder.stats.differential import run_protein_differential

        rng = np.random.default_rng(42)
        n_a, n_b = 5, 35
        n_samples = n_a + n_b
        n_features = 20
        data = rng.normal(0, 1, size=(n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        condition = np.array(["A"] * n_a + ["B"] * n_b)
        contrast = ("B", "A")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_protein_differential(
                data=data,
                feature_ids=feature_ids,
                sample_condition=condition,
                contrast=contrast,
                eb_moderation=True,
                verbose=False,
            )
            imbalance_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning)
                and "imbalanced" in str(x.message).lower()
            ]
            assert len(imbalance_warnings) >= 1, (
                f"Expected UserWarning about imbalanced groups, got: "
                f"{[str(x.message) for x in w]}"
            )
            msg = str(imbalance_warnings[0].message)
            assert "5" in msg and "35" in msg
            assert "Satterthwaite" in msg


class TestPadContrastForCovariates:
    """Tests for pad_contrast_for_covariates."""

    def test_no_padding(self):
        """Zero covariate cols should return original."""
        c = np.array([1.0, -1.0])
        result = pad_contrast_for_covariates(c, 0)
        np.testing.assert_array_equal(result, c)

    def test_padding_appends_zeros(self):
        """Should append zeros for covariate columns."""
        c = np.array([1.0, -1.0])
        result = pad_contrast_for_covariates(c, 3)
        expected = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)
