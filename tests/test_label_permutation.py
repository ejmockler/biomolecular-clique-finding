"""Tests for label permutation null analysis."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.label_permutation import (
    LabelPermutationResult,
    generate_free_permutation,
    generate_stratified_permutation,
    run_label_permutation_null,
)


class TestGenerateStratifiedPermutation:
    """Tests for generate_stratified_permutation()."""

    def test_preserves_within_stratum_counts(self):
        """Stratified permutation preserves label counts within each stratum."""
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "A", "B", "B", "B", "A", "B", "A", "B"])
        strata = np.array(["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"])

        for _ in range(20):
            perm = generate_stratified_permutation(labels, strata, rng)

            # Check within-stratum label counts are preserved
            for stratum in ["M", "F"]:
                mask = strata == stratum
                orig_counts = pd.Series(labels[mask]).value_counts().sort_index()
                perm_counts = pd.Series(perm[mask]).value_counts().sort_index()
                pd.testing.assert_series_equal(orig_counts, perm_counts)

    def test_actually_permutes(self):
        """Stratified permutation actually changes label positions."""
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
        strata = np.array(["M", "M", "M", "M", "F", "F", "F", "F"])

        # Run enough times that at least one should differ
        any_different = False
        for _ in range(50):
            perm = generate_stratified_permutation(labels, strata, rng)
            if not np.array_equal(perm, labels):
                any_different = True
                break

        assert any_different, "Permutation never changed labels"

    def test_single_stratum_equals_free(self):
        """Single stratum reduces to free permutation of that group."""
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "B", "B", "A", "B"])
        strata = np.array(["X", "X", "X", "X", "X", "X"])

        perm = generate_stratified_permutation(labels, strata, rng)
        # Should still have same label counts
        assert sorted(perm) == sorted(labels)


class TestGenerateFreePermutation:
    """Tests for generate_free_permutation()."""

    def test_preserves_counts(self):
        """Free permutation preserves overall label counts."""
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "A", "B", "B"])

        perm = generate_free_permutation(labels, rng)
        assert sorted(perm) == sorted(labels)

    def test_actually_permutes(self):
        """Free permutation actually changes positions."""
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])

        any_different = False
        for _ in range(50):
            perm = generate_free_permutation(labels, rng)
            if not np.array_equal(perm, labels):
                any_different = True
                break

        assert any_different


class TestRunLabelPermutationNull:
    """Tests for run_label_permutation_null()."""

    @pytest.fixture
    def null_data(self):
        """Create data with no true signal (null hypothesis true)."""
        rng = np.random.default_rng(42)
        n_features = 100
        n_samples = 40

        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        conditions = np.array(["A"] * 20 + ["B"] * 20)

        # Mark some genes as "targets" (but no real signal)
        target_ids = feature_ids[:10]

        return data, feature_ids, conditions, target_ids

    @pytest.fixture
    def signal_data(self):
        """Create data with a real signal in targets."""
        rng = np.random.default_rng(42)
        n_features = 100
        n_samples = 40

        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        conditions = np.array(["A"] * 20 + ["B"] * 20)

        # Add strong signal to first 10 genes
        data[:10, :20] += 2.0  # Group A upregulated

        target_ids = feature_ids[:10]

        return data, feature_ids, conditions, target_ids

    def test_null_data_nonsignificant(self, null_data):
        """Null data should produce non-significant permutation p-value."""
        data, feature_ids, conditions, target_ids = null_data

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=50,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, LabelPermutationResult)
        assert result.permutation_pvalue > 0.05
        assert result.n_permutations > 0
        assert not result.stratified

    def test_signal_data_significant(self, signal_data):
        """Signal data should produce significant permutation p-value."""
        data, feature_ids, conditions, target_ids = signal_data

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=99,
            seed=42,
            verbose=False,
        )

        assert result.permutation_pvalue < 0.1  # Should be quite low
        assert result.observed_z > result.null_mean

    def test_stratified_permutation(self, null_data):
        """Stratified permutation runs and reports correctly."""
        data, feature_ids, conditions, target_ids = null_data

        # Create strata: half M, half F in each group
        strata = np.array(
            ["M"] * 10 + ["F"] * 10 + ["M"] * 10 + ["F"] * 10
        )

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=30,
            stratify_by=strata,
            seed=42,
            verbose=False,
        )

        assert result.stratified is True

    def test_mismatched_strata_length(self, null_data):
        """Mismatched strata length → ValueError."""
        data, feature_ids, conditions, target_ids = null_data

        with pytest.raises(ValueError, match="stratify_by length"):
            run_label_permutation_null(
                data=data,
                feature_ids=feature_ids,
                sample_condition=conditions,
                contrast=("A", "B"),
                target_gene_ids=target_ids,
                n_permutations=10,
                stratify_by=np.array(["M", "F"]),  # Wrong length
                verbose=False,
            )

    def test_to_dict(self, null_data):
        """Result serializes to dict."""
        data, feature_ids, conditions, target_ids = null_data

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=20,
            seed=42,
            verbose=False,
        )

        d = result.to_dict()
        assert "observed_z" in d
        assert "permutation_pvalue" in d
        assert "null_z_quantiles" in d

    def test_with_covariates(self, null_data):
        """Covariates are passed through to protein differential."""
        data, feature_ids, conditions, target_ids = null_data

        covariates_df = pd.DataFrame({
            "Sex": (["M"] * 10 + ["F"] * 10) * 2,
        })

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=20,
            covariates_df=covariates_df,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, LabelPermutationResult)

    def test_covariate_design_passed_through(self, null_data):
        """covariate_design is passed to every run_protein_differential call."""
        from unittest.mock import patch, MagicMock
        from cliquefinder.stats.design_matrix import CovariateDesign

        data, feature_ids, conditions, target_ids = null_data
        n_samples = len(conditions)  # 40

        # Build a CovariateDesign with a restrictive mask (drop last 10)
        mask = np.ones(n_samples, dtype=bool)
        mask[-10:] = False
        n_valid = int(mask.sum())  # 30

        import statsmodels.api as sm
        valid_conditions = conditions[mask]
        X_cond = pd.get_dummies(
            pd.Categorical(valid_conditions, categories=["A", "B"]),
            drop_first=True,
            dtype=float,
        )
        X_cond = sm.add_constant(X_cond)
        X_np = X_cond.values.astype(np.float64)
        contrast_vec = np.array([-1.0, 1.0])

        design = CovariateDesign(
            X=X_np,
            condition_cols=[0, 1],
            covariate_cols=[],
            col_names=list(X_cond.columns),
            contrast=contrast_vec,
            contrast_name="A_vs_B",
            sample_mask=mask,
            n_condition_params=2,
            n_covariate_params=0,
        )

        # Build a fake protein results DataFrame that _extract_enrichment_z needs
        rng = np.random.default_rng(99)
        n_features = len(feature_ids)
        target_set = set(target_ids)
        fake_results = pd.DataFrame({
            "feature_id": feature_ids,
            "t_statistic": rng.normal(0, 1, n_features),
            "is_target": [fid in target_set for fid in feature_ids],
        })

        # Track covariate_design arguments in each call
        received_designs = []

        def mock_rpd(**kwargs):
            received_designs.append(kwargs.get("covariate_design"))
            return fake_results

        # Patch at the source module (label_permutation imports it locally)
        with patch(
            "cliquefinder.stats.differential.run_protein_differential",
            side_effect=mock_rpd,
        ):
            result = run_label_permutation_null(
                data=data,
                feature_ids=feature_ids,
                sample_condition=conditions,
                contrast=("A", "B"),
                target_gene_ids=target_ids,
                n_permutations=5,
                covariate_design=design,
                seed=42,
                verbose=False,
            )

        assert isinstance(result, LabelPermutationResult)
        # 1 observed + 5 permutations = 6 calls
        assert len(received_designs) == 6
        for cd in received_designs:
            assert cd is design, (
                "covariate_design should be the same object passed to "
                "run_label_permutation_null"
            )
            assert int(cd.sample_mask.sum()) == n_valid

    def test_covariate_design_not_passed_when_none(self, null_data):
        """Without covariate_design, None is passed to run_protein_differential."""
        from unittest.mock import patch

        data, feature_ids, conditions, target_ids = null_data

        rng = np.random.default_rng(99)
        n_features = len(feature_ids)
        target_set = set(target_ids)
        fake_results = pd.DataFrame({
            "feature_id": feature_ids,
            "t_statistic": rng.normal(0, 1, n_features),
            "is_target": [fid in target_set for fid in feature_ids],
        })

        received_designs = []

        def mock_rpd(**kwargs):
            received_designs.append(kwargs.get("covariate_design"))
            return fake_results

        with patch(
            "cliquefinder.stats.differential.run_protein_differential",
            side_effect=mock_rpd,
        ):
            result = run_label_permutation_null(
                data=data,
                feature_ids=feature_ids,
                sample_condition=conditions,
                contrast=("A", "B"),
                target_gene_ids=target_ids,
                n_permutations=3,
                covariate_design=None,
                seed=42,
                verbose=False,
            )

        assert isinstance(result, LabelPermutationResult)
        # 1 observed + 3 permutations = 4 calls, all should have None
        assert len(received_designs) == 4
        for cd in received_designs:
            assert cd is None

    def test_covariate_design_none_default(self, null_data):
        """Without covariate_design, function still works (backward compat)."""
        data, feature_ids, conditions, target_ids = null_data

        result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=conditions,
            contrast=("A", "B"),
            target_gene_ids=target_ids,
            n_permutations=10,
            covariate_design=None,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, LabelPermutationResult)
        assert result.n_permutations > 0
