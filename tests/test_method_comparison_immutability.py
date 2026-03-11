"""
Tests for method comparison result immutability, unclamped p-value storage,
O(1) dict-based clique lookup, protein count consistency via
clique_to_feature_indices, and defensive metadata copying.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.concordance import MethodComparisonResult
from cliquefinder.stats.method_comparison_types import (
    ConcordanceMetrics,
    MethodName,
    UnifiedCliqueResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    clique_id: str,
    method: MethodName = MethodName.OLS,
    p_value: float = 0.01,
    effect_size: float = 1.5,
) -> UnifiedCliqueResult:
    """Create a minimal valid UnifiedCliqueResult."""
    return UnifiedCliqueResult(
        clique_id=clique_id,
        method=method,
        effect_size=effect_size,
        effect_size_se=0.3,
        p_value=p_value,
        statistic_value=5.0,
        statistic_type="t",
        degrees_of_freedom=10.0,
        n_proteins=10,
        n_proteins_found=8,
    )


def _make_mcr(**overrides) -> MethodComparisonResult:
    """Create a minimal MethodComparisonResult with sensible defaults."""
    defaults = dict(
        results_by_method={
            MethodName.OLS: [_make_result("X", MethodName.OLS)],
        },
        pairwise_concordance=[],
        mean_spearman_rho=0.8,
        mean_cohen_kappa=0.6,
        disagreement_cases=pd.DataFrame(),
        preprocessing_params={"normalization": "median"},
        methods_run=[MethodName.OLS],
        n_cliques_tested=1,
        failed_methods={},
    )
    defaults.update(overrides)
    return MethodComparisonResult(**defaults)


# ===========================================================================
# MethodComparisonResult is frozen
# ===========================================================================


class TestMethodComparisonFrozen:
    """MethodComparisonResult must be frozen (immutable after construction)."""

    def test_frozen_attribute_assignment_raises(self):
        """Assigning to any field on a frozen dataclass raises FrozenInstanceError."""
        mcr = _make_mcr()
        with pytest.raises(FrozenInstanceError):
            mcr.mean_spearman_rho = 0.5

    def test_frozen_n_cliques_tested_raises(self):
        """Cannot reassign n_cliques_tested."""
        mcr = _make_mcr()
        with pytest.raises(FrozenInstanceError):
            mcr.n_cliques_tested = 999

    def test_results_by_method_is_mappingproxy(self):
        """results_by_method should be wrapped in MappingProxyType."""
        mcr = _make_mcr()
        assert isinstance(mcr.results_by_method, MappingProxyType)

    def test_results_by_method_values_are_tuples(self):
        """The list values inside results_by_method should be converted to tuples."""
        mcr = _make_mcr()
        for v in mcr.results_by_method.values():
            assert isinstance(v, tuple), f"Expected tuple, got {type(v)}"

    def test_pairwise_concordance_is_tuple(self):
        """pairwise_concordance should be converted from list to tuple."""
        mcr = _make_mcr()
        assert isinstance(mcr.pairwise_concordance, tuple)

    def test_methods_run_is_tuple(self):
        """methods_run should be converted from list to tuple."""
        mcr = _make_mcr()
        assert isinstance(mcr.methods_run, tuple)

    def test_preprocessing_params_is_mappingproxy(self):
        """preprocessing_params should be wrapped in MappingProxyType."""
        mcr = _make_mcr()
        assert isinstance(mcr.preprocessing_params, MappingProxyType)

    def test_failed_methods_is_mappingproxy(self):
        """failed_methods should be wrapped in MappingProxyType."""
        mcr = _make_mcr()
        assert isinstance(mcr.failed_methods, MappingProxyType)

    def test_results_by_method_mutation_raises(self):
        """Mutating results_by_method should raise TypeError."""
        mcr = _make_mcr()
        with pytest.raises(TypeError):
            mcr.results_by_method[MethodName.LMM] = []

    def test_preprocessing_params_mutation_raises(self):
        """Mutating preprocessing_params should raise TypeError."""
        mcr = _make_mcr()
        with pytest.raises(TypeError):
            mcr.preprocessing_params["new_key"] = "value"

    def test_failed_methods_mutation_raises(self):
        """Mutating failed_methods should raise TypeError."""
        mcr = _make_mcr()
        with pytest.raises(TypeError):
            mcr.failed_methods["ols"] = "error"

    def test_wide_format_still_works_after_freezing(self):
        """Ensure wide_format() still works after freezing (regression)."""
        mcr = _make_mcr()
        df = mcr.wide_format()
        assert len(df) == 1
        assert "clique_id" in df.columns

    def test_summary_still_works_after_freezing(self):
        """Ensure summary() still works after freezing (regression)."""
        mcr = _make_mcr()
        text = mcr.summary()
        assert "Method Comparison Results" in text

    def test_concordance_matrix_still_works_after_freezing(self):
        """Ensure concordance_matrix() still works after freezing."""
        mcr = _make_mcr()
        mat = mcr.concordance_matrix()
        assert mat.shape == (1, 1)  # one method

    def test_robust_hits_still_works_after_freezing(self):
        """Ensure robust_hits() still works after freezing."""
        mcr = _make_mcr()
        hits = mcr.robust_hits(threshold=0.05)
        assert isinstance(hits, list)

    def test_iteration_over_frozen_results(self):
        """Can iterate over frozen results_by_method and methods_run."""
        mcr = _make_mcr()
        for method in mcr.methods_run:
            results = mcr.results_by_method[method]
            assert len(results) > 0

    def test_default_factory_still_works(self):
        """The default_factory for failed_methods should still work."""
        mcr = MethodComparisonResult(
            results_by_method={},
            pairwise_concordance=[],
            mean_spearman_rho=np.nan,
            mean_cohen_kappa=np.nan,
            disagreement_cases=pd.DataFrame(),
            preprocessing_params={},
            methods_run=[],
            n_cliques_tested=0,
            # Not passing failed_methods -- testing default
        )
        assert len(mcr.failed_methods) == 0


# ===========================================================================
# Permutation adapter stores unclamped p-values
# ===========================================================================


def _make_perm_mock_experiment(clique_id="TP53", protein_ids=None, n_features=5):
    """Helper: create a mock experiment for permutation tests."""
    if protein_ids is None:
        protein_ids = ["P1", "P2"]

    feature_ids = tuple(f"P{i+1}" for i in range(n_features))
    clique = MagicMock(clique_id=clique_id, protein_ids=protein_ids)

    feature_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    clique_indices = tuple(
        feature_to_idx[p] for p in protein_ids if p in feature_to_idx
    )

    mock_experiment = MagicMock()
    mock_experiment.sample_metadata = pd.DataFrame({"cond": ["A", "B"]})
    mock_experiment.data = np.zeros((n_features, 2))
    mock_experiment.feature_ids = feature_ids
    mock_experiment.cliques = (clique,)
    mock_experiment.condition_column = "cond"
    mock_experiment.contrast = ("A", "B")
    mock_experiment.subject_column = None
    mock_experiment.clique_to_feature_indices = {clique_id: clique_indices}
    mock_experiment.feature_to_idx = feature_to_idx
    return mock_experiment


def _make_perm_result_mock(clique_id, empirical_pvalue, observed_tvalue=2.0):
    """Helper: create a mock PermutationTestResult."""
    pr = MagicMock()
    pr.clique_id = clique_id
    pr.observed_tvalue = observed_tvalue
    pr.empirical_pvalue = empirical_pvalue
    pr.percentile_rank = 95.0
    pr.observed_log2fc = 0.5
    pr.is_significant = empirical_pvalue < 0.05
    return pr


def _make_null_df(clique_ids):
    """Helper: create a null distribution DataFrame."""
    return pd.DataFrame({
        "clique_id": clique_ids,
        "null_tvalue_mean": [0.0] * len(clique_ids),
        "null_tvalue_std": [1.0] * len(clique_ids),
    })


class TestUnclampedPValues:
    """
    The permutation adapter must store the original empirical p-value,
    not a clamped version. Clamping should only happen for z-score computation.
    """

    def test_zero_pvalue_stored_as_zero(self):
        """A p-value of exactly 0.0 must be stored as 0.0, not clamped."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        mock_exp = _make_perm_mock_experiment("TP53", ["P1", "P2"])
        perm_result = _make_perm_result_mock("TP53", 0.0, observed_tvalue=5.0)
        null_df = _make_null_df(["TP53"])

        method = PermutationMethod(n_permutations=1000)
        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_exp)

        assert len(results) == 1
        assert results[0].p_value == 0.0

    def test_one_pvalue_stored_as_one(self):
        """A p-value of exactly 1.0 must be stored as 1.0, not clamped."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        mock_exp = _make_perm_mock_experiment("MYC", ["P1"])
        perm_result = _make_perm_result_mock("MYC", 1.0, observed_tvalue=0.1)
        null_df = _make_null_df(["MYC"])

        method = PermutationMethod(n_permutations=1000)
        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_exp)

        assert len(results) == 1
        assert results[0].p_value == 1.0

    def test_normal_pvalue_stored_unchanged(self):
        """A normal p-value (e.g. 0.034) should be stored exactly as-is."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        mock_exp = _make_perm_mock_experiment("BRCA1", ["P1", "P2"])
        perm_result = _make_perm_result_mock("BRCA1", 0.034, observed_tvalue=2.5)
        null_df = _make_null_df(["BRCA1"])

        method = PermutationMethod(n_permutations=1000)
        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_exp)

        assert len(results) == 1
        assert results[0].p_value == 0.034

    def test_z_score_still_finite_for_edge_pvalues(self):
        """z-score computation should clamp internally but still produce finite values."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        mock_exp = _make_perm_mock_experiment("TP53", ["P1"])
        perm_result = _make_perm_result_mock("TP53", 0.0, observed_tvalue=5.0)
        null_df = _make_null_df(["TP53"])

        method = PermutationMethod(n_permutations=1000)
        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_exp)

        assert len(results) == 1
        # p_value stored as 0.0, but z-score is finite (clamped for ppf)
        assert results[0].p_value == 0.0
        assert np.isfinite(results[0].statistic_value)


# ===========================================================================
# O(1) clique lookup via dict
# ===========================================================================


class TestCliqueDictLookup:
    """
    Verify that the permutation adapter builds a dict for clique lookup
    instead of doing a linear scan inside the result loop.
    """

    def test_source_uses_dict_lookup(self):
        """
        Parse the source of PermutationMethod.test to verify it builds a
        clique_lookup dict and uses .get() instead of a linear scan.
        """
        from cliquefinder.stats.methods.permutation import PermutationMethod

        source = inspect.getsource(PermutationMethod.test)

        # Should have a dict comprehension building clique_lookup
        assert "clique_lookup" in source, (
            "Expected 'clique_lookup' dict in PermutationMethod.test"
        )

        # Should use .get() for O(1) lookup
        assert "clique_lookup.get(" in source, (
            "Expected 'clique_lookup.get(' for O(1) dict lookup"
        )

        # Should NOT have the old linear scan pattern
        assert "for c in experiment.cliques:" not in source, (
            "Old O(N*M) linear scan pattern 'for c in experiment.cliques:' "
            "should be removed"
        )

    def test_lookup_correctness_with_multiple_cliques(self):
        """Verify the dict lookup finds the correct clique definitions."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        # Create multiple cliques with distinct sizes
        clique_a = MagicMock(clique_id="A", protein_ids=["P1", "P2"])
        clique_b = MagicMock(clique_id="B", protein_ids=["P3"])
        clique_c = MagicMock(clique_id="C", protein_ids=["P4", "P5", "P6"])

        perm_results = []
        for cid, t_val in [("A", 3.0), ("B", 1.5), ("C", 4.2)]:
            perm_results.append(_make_perm_result_mock(cid, 0.05, t_val))

        null_df = _make_null_df(["A", "B", "C"])

        mock_experiment = MagicMock()
        mock_experiment.sample_metadata = pd.DataFrame({"cond": ["A", "B"]})
        mock_experiment.data = np.zeros((6, 2))
        mock_experiment.feature_ids = ("P1", "P2", "P3", "P4", "P5", "P6")
        mock_experiment.cliques = (clique_a, clique_b, clique_c)
        mock_experiment.condition_column = "cond"
        mock_experiment.contrast = ("A", "B")
        mock_experiment.subject_column = None
        mock_experiment.clique_to_feature_indices = {
            "A": (0, 1), "B": (2,), "C": (3, 4, 5)
        }
        mock_experiment.feature_to_idx = {f"P{i+1}": i for i in range(6)}

        method = PermutationMethod(n_permutations=1000)

        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=(perm_results, null_df),
        ):
            results = method.test(mock_experiment)

        assert len(results) == 3
        result_map = {r.clique_id: r for r in results}

        # Verify correct n_proteins from each clique definition
        assert result_map["A"].n_proteins == 2
        assert result_map["B"].n_proteins == 1
        assert result_map["C"].n_proteins == 3


# ===========================================================================
# n_proteins_found uses clique_to_feature_indices
# ===========================================================================


class TestProteinFoundConsistency:
    """
    n_proteins_found must use experiment.clique_to_feature_indices,
    which reflects the actual mapping used during testing, not
    a manual scan of feature_to_idx.
    """

    def test_n_proteins_found_uses_feature_indices(self):
        """
        When clique_to_feature_indices differs from a naive feature_to_idx
        scan, n_proteins_found should match clique_to_feature_indices.
        """
        from cliquefinder.stats.methods.permutation import PermutationMethod

        # Clique has 3 protein_ids but only 2 are in clique_to_feature_indices
        clique = MagicMock(
            clique_id="TEST", protein_ids=["P1", "P2", "UNMAPPED"]
        )

        perm_result = _make_perm_result_mock("TEST", 0.03)
        null_df = _make_null_df(["TEST"])

        mock_experiment = MagicMock()
        mock_experiment.sample_metadata = pd.DataFrame({"cond": ["A", "B"]})
        mock_experiment.data = np.zeros((3, 2))
        mock_experiment.feature_ids = ("P1", "P2", "P3")
        mock_experiment.cliques = (clique,)
        mock_experiment.condition_column = "cond"
        mock_experiment.contrast = ("A", "B")
        mock_experiment.subject_column = None
        # Only 2 indices -- the precomputed source of truth
        mock_experiment.clique_to_feature_indices = {"TEST": (0, 1)}
        mock_experiment.feature_to_idx = {"P1": 0, "P2": 1, "P3": 2}

        method = PermutationMethod(n_permutations=1000)

        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_experiment)

        assert len(results) == 1
        # n_proteins is the full clique definition size
        assert results[0].n_proteins == 3
        # n_proteins_found should come from clique_to_feature_indices
        assert results[0].n_proteins_found == 2

    def test_n_proteins_found_zero_when_not_in_indices(self):
        """When a clique has no entry in clique_to_feature_indices, n_found=0."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        clique = MagicMock(clique_id="MISSING", protein_ids=["X1", "X2"])
        perm_result = _make_perm_result_mock("MISSING", 0.5)
        null_df = _make_null_df(["MISSING"])

        mock_experiment = MagicMock()
        mock_experiment.sample_metadata = pd.DataFrame({"cond": ["A", "B"]})
        mock_experiment.data = np.zeros((2, 2))
        mock_experiment.feature_ids = ("P1", "P2")
        mock_experiment.cliques = (clique,)
        mock_experiment.condition_column = "cond"
        mock_experiment.contrast = ("A", "B")
        mock_experiment.subject_column = None
        mock_experiment.clique_to_feature_indices = {}  # not present
        mock_experiment.feature_to_idx = {"P1": 0, "P2": 1}

        method = PermutationMethod(n_permutations=1000)

        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([perm_result], null_df),
        ):
            results = method.test(mock_experiment)

        assert len(results) == 1
        assert results[0].n_proteins_found == 0

    def test_source_references_clique_to_feature_indices(self):
        """Verify source code references clique_to_feature_indices."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        source = inspect.getsource(PermutationMethod.test)
        assert "clique_to_feature_indices" in source, (
            "Expected 'clique_to_feature_indices' in PermutationMethod.test"
        )


# ===========================================================================
# Metadata defensively copied
# ===========================================================================


class TestMetadataDefensiveCopy:
    """
    The permutation adapter must pass a copy of sample_metadata to
    run_permutation_test_gpu to prevent mutation of the experiment.
    """

    def test_source_calls_copy_on_metadata(self):
        """Source code should call .copy() on sample_metadata before passing."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        source = inspect.getsource(PermutationMethod.test)
        assert (
            "sample_metadata.copy()" in source
            or "sample_metadata_copy" in source
        ), (
            "Expected defensive copy of sample_metadata before passing to "
            "run_permutation_test_gpu"
        )

    def test_original_metadata_not_mutated(self):
        """
        Simulate a permutation engine that mutates the passed DataFrame
        and verify the experiment's metadata is unchanged.
        """
        from cliquefinder.stats.methods.permutation import PermutationMethod

        original_metadata = pd.DataFrame({
            "cond": ["A", "A", "B", "B"],
            "subject": ["s1", "s2", "s3", "s4"],
        })
        original_copy = original_metadata.copy()

        def mutating_gpu_fn(
            data, feature_ids, sample_metadata, clique_definitions,
            condition_col, contrast, subject_col, n_permutations,
            random_state, verbose, **kwargs
        ):
            """Simulate a GPU function that mutates the metadata."""
            sample_metadata["new_col"] = 999
            sample_metadata.iloc[0, 0] = "MUTATED"
            return [], pd.DataFrame()

        mock_experiment = MagicMock()
        mock_experiment.sample_metadata = original_metadata
        mock_experiment.data = np.zeros((5, 4))
        mock_experiment.feature_ids = ("P1", "P2", "P3", "P4", "P5")
        mock_experiment.cliques = ()
        mock_experiment.condition_column = "cond"
        mock_experiment.contrast = ("A", "B")
        mock_experiment.subject_column = "subject"
        mock_experiment.clique_to_feature_indices = {}
        mock_experiment.feature_to_idx = {}

        method = PermutationMethod(n_permutations=100)

        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            side_effect=mutating_gpu_fn,
        ):
            method.test(mock_experiment)

        # Original metadata should be unchanged
        pd.testing.assert_frame_equal(
            original_metadata, original_copy,
            check_exact=True,
        )
        assert "new_col" not in original_metadata.columns
