"""Tests for Audit VIII findings — 12 fixes across 3 cycles."""

import warnings

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Cycle 1: Semantic/Statistical
# =============================================================================


class TestRCVIII4_IsSignificant:
    """RC-VIII-4: is_significant requires BOTH graph-level p-value AND community quality."""

    def _make_community(self, *, size, density, pvalue):
        from cliquefinder.knowledge.regulatory_coherence import CommunityResult
        genes = {f"G{i}" for i in range(size)}
        return CommunityResult(
            community_id=0,
            genes=genes,
            mean_correlation=0.8,
            min_correlation=0.5,
            max_correlation=0.95,
            density=density,
            modularity_contribution=0.1,
            permutation_pvalue=pvalue,
        )

    def test_both_pass(self):
        c = self._make_community(size=5, density=0.7, pvalue=0.01)
        assert c.is_significant is True

    def test_good_pvalue_low_density(self):
        """Small noisy community should NOT be significant even with good graph p-value."""
        c = self._make_community(size=5, density=0.3, pvalue=0.01)
        assert c.is_significant is False

    def test_good_pvalue_too_small(self):
        c = self._make_community(size=2, density=0.8, pvalue=0.01)
        assert c.is_significant is False

    def test_high_pvalue_good_quality(self):
        c = self._make_community(size=5, density=0.7, pvalue=0.2)
        assert c.is_significant is False

    def test_no_pvalue_fallback_quality(self):
        """Without permutation test, fallback to quality check."""
        c = self._make_community(size=5, density=0.7, pvalue=None)
        assert c.is_significant is True

    def test_no_pvalue_bad_quality(self):
        c = self._make_community(size=2, density=0.3, pvalue=None)
        assert c.is_significant is False


class TestVALVIII6_FailedPhaseVerdict:
    """VAL-VIII-6: Phase 1 status='failed' should yield 'inconclusive', not 'refuted'."""

    def test_phase1_failed_no_phase3(self):
        from cliquefinder.stats.validation_report import ValidationReport
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"status": "failed", "error": "crash"})
        report.compute_verdict()
        assert report.verdict == "inconclusive"
        assert "runtime error" in report.summary.lower()

    def test_phase3_failed_phase1_pass(self):
        from cliquefinder.stats.validation_report import ValidationReport
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {"status": "failed", "error": "crash"})
        report.compute_verdict()
        assert report.verdict == "inconclusive"

    def test_both_failed(self):
        from cliquefinder.stats.validation_report import ValidationReport
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"status": "failed"})
        report.add_phase("label_permutation", {"status": "failed"})
        report.compute_verdict()
        assert report.verdict == "inconclusive"

    def test_normal_refuted_still_works(self):
        """Non-error failure (high p-values) should still produce 'refuted'."""
        from cliquefinder.stats.validation_report import ValidationReport
        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.5})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.5},
            "permutation_pvalue": 0.5,
        })
        report.compute_verdict()
        assert report.verdict == "refuted"


class TestVALVIII3_FrozenFraction:
    """VAL-VIII-3: frozen_fraction field on LabelPermutationResult."""

    def test_frozen_fraction_in_result(self):
        from cliquefinder.stats.label_permutation import LabelPermutationResult
        result = LabelPermutationResult(
            observed_z=2.0,
            null_z_scores=np.array([0.1, 0.2]),
            permutation_pvalue=0.5,
            n_permutations=2,
            stratified=True,
            frozen_fraction=0.3,
        )
        assert result.frozen_fraction == 0.3
        d = result.to_dict()
        assert d["frozen_fraction"] == 0.3

    def test_frozen_fraction_default_zero(self):
        from cliquefinder.stats.label_permutation import LabelPermutationResult
        result = LabelPermutationResult(
            observed_z=2.0,
            null_z_scores=np.array([0.1]),
            permutation_pvalue=0.5,
            n_permutations=1,
            stratified=False,
        )
        assert result.frozen_fraction == 0.0

    def test_generate_stratified_returns_frozen_count(self):
        from cliquefinder.stats.label_permutation import generate_stratified_permutation
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "B", "B", "A", "A"])
        # All in stratum "X" => non-degenerate (has A and B)
        strata = np.array(["X", "X", "X", "X", "X", "X"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, n_frozen = generate_stratified_permutation(labels, strata, rng)
        assert n_frozen == 0

    def test_generate_stratified_degenerate_frozen(self):
        from cliquefinder.stats.label_permutation import generate_stratified_permutation
        rng = np.random.default_rng(42)
        labels = np.array(["A", "A", "B", "B"])
        # Stratum Y has only A, stratum Z has only B => both degenerate
        strata = np.array(["Y", "Y", "Z", "Z"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, n_frozen = generate_stratified_permutation(labels, strata, rng)
        assert n_frozen == 4  # All samples frozen


class TestRCVIII1_ComputeCorr:
    """RC-VIII-1: _compute_corr handles NaN and diagonal internally."""

    def test_pearson_zero_variance(self):
        from unittest.mock import MagicMock
        from cliquefinder.knowledge.regulatory_coherence import CoherenceAnalyzer

        # Create a mock matrix
        matrix = MagicMock()
        matrix.feature_ids = pd.Index(["G1", "G2", "G3"])

        analyzer = CoherenceAnalyzer.__new__(CoherenceAnalyzer)
        analyzer.matrix = matrix
        analyzer.config = MagicMock()
        analyzer._rng = np.random.default_rng(42)

        # Zero-variance gene (row 0 all same)
        expr = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        corr, n_nan = analyzer._compute_corr(expr, method='pearson')

        assert corr.shape == (3, 3)
        assert np.all(np.diag(corr) == 1.0), "Diagonal should be 1.0"
        assert not np.any(np.isnan(corr)), "No NaN should remain"
        assert n_nan > 0, "Should report NaN substitutions"

    def test_spearman_clean(self):
        from unittest.mock import MagicMock
        from cliquefinder.knowledge.regulatory_coherence import CoherenceAnalyzer

        matrix = MagicMock()
        analyzer = CoherenceAnalyzer.__new__(CoherenceAnalyzer)
        analyzer.matrix = matrix
        analyzer._rng = np.random.default_rng(42)

        expr = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        corr, n_nan = analyzer._compute_corr(expr, method='spearman')
        assert corr.shape == (2, 2)
        assert np.all(np.diag(corr) == 1.0)
        assert n_nan == 0


# =============================================================================
# Cycle 2: Immutability & Safety
# =============================================================================


class TestARCHVIII1_GetCliqueDataWriteable:
    """ARCH-VIII-1: get_clique_data returns read-only array."""

    def test_returned_array_is_readonly(self):
        from types import MappingProxyType
        from cliquefinder.stats.experiment import PreparedCliqueExperiment
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        exp = PreparedCliqueExperiment(
            data=data,
            feature_ids=("A", "B"),
            feature_to_idx={"A": 0, "B": 1},
            sample_metadata=pd.DataFrame({"cond": ["X", "Y"]}),
            condition_column="cond",
            subject_column=None,
            conditions=("X", "Y"),
            n_samples=2,
            cliques=(),
            clique_to_feature_indices={"c1": (0, 1)},
            symbol_to_feature={},
            contrast=("X", "Y"),
            contrast_name="X_vs_Y",
            preprocessing_params={},
            creation_timestamp="2026-01-01",
        )
        subset, ids = exp.get_clique_data("c1")
        with pytest.raises(ValueError):
            subset[0, 0] = 999.0


class TestARCHVIII3_BioMatrixImmutable:
    """ARCH-VIII-3: BioMatrix.data is read-only."""

    def test_data_is_readonly(self):
        from cliquefinder.core.biomatrix import BioMatrix
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = BioMatrix(
            data=data,
            feature_ids=pd.Index(["G1", "G2"]),
            sample_ids=pd.Index(["S1", "S2"]),
            sample_metadata=pd.DataFrame({"cond": ["A", "B"]}, index=["S1", "S2"]),
            quality_flags=np.ones((2, 2)),
        )
        with pytest.raises(ValueError):
            m.data[0, 0] = 999.0


class TestNEGVIII1_ToDictGuard:
    """NEG-VIII-1: to_dict doesn't crash on empty competitive z scores."""

    def test_to_dict_with_none_control_z(self):
        from cliquefinder.stats.negative_controls import NegativeControlResult
        result = NegativeControlResult(
            target_pvalue=0.01,
            target_set_id="test",
            target_set_size=10,
            control_pvalues=np.array([0.1, 0.2]),
            fpr=0.0,
            alpha=0.05,
            target_percentile=5.0,
            median_control_pvalue=0.15,
            mean_control_pvalue=0.15,
            n_control_sets=2,
            n_significant_controls=0,
            n_valid_controls=2,
            target_competitive_z=2.5,
            control_competitive_z_scores=None,  # <-- None
        )
        d = result.to_dict()
        assert "competitive_z" not in d  # Should skip, not crash

    def test_to_dict_with_empty_control_z(self):
        from cliquefinder.stats.negative_controls import NegativeControlResult
        result = NegativeControlResult(
            target_pvalue=0.01,
            target_set_id="test",
            target_set_size=10,
            control_pvalues=np.array([0.1]),
            fpr=0.0,
            alpha=0.05,
            target_percentile=5.0,
            median_control_pvalue=0.1,
            mean_control_pvalue=0.1,
            n_control_sets=1,
            n_significant_controls=0,
            n_valid_controls=1,
            target_competitive_z=2.5,
            control_competitive_z_scores=np.array([]),  # <-- empty
        )
        d = result.to_dict()
        assert "competitive_z" not in d


class TestRCVIII5_MatchWithinSign:
    """RC-VIII-5: Cross-condition matching only matches within same sign."""

    def test_pos_matches_pos_not_neg(self):
        from cliquefinder.knowledge.regulatory_coherence import (
            CommunityResult, CorrelationSign, StratifiedCoherenceResult,
            CoherenceAnalyzer,
        )
        # Positive community with genes {A, B, C}
        pos_a = CommunityResult(
            community_id=0, genes={"A", "B", "C"},
            mean_correlation=0.8, min_correlation=0.6, max_correlation=0.9,
            density=0.8, modularity_contribution=0.1,
            correlation_sign=CorrelationSign.POSITIVE,
        )
        # In condition B, same genes appear as NEGATIVE community
        neg_b = CommunityResult(
            community_id=0, genes={"A", "B", "C"},
            mean_correlation=0.8, min_correlation=0.6, max_correlation=0.9,
            density=0.8, modularity_contribution=0.1,
            correlation_sign=CorrelationSign.NEGATIVE,
        )
        result_a = StratifiedCoherenceResult(
            condition="A", n_samples=10,
            positive_communities=[pos_a], negative_communities=[],
            modularity_positive=0.3, modularity_negative=0.0,
            genes_in_positive={"A", "B", "C"}, genes_in_negative=set(),
            genes_unclustered=set(),
        )
        result_b = StratifiedCoherenceResult(
            condition="B", n_samples=10,
            positive_communities=[], negative_communities=[neg_b],
            modularity_positive=0.0, modularity_negative=0.3,
            genes_in_positive=set(), genes_in_negative={"A", "B", "C"},
            genes_unclustered=set(),
        )
        analyzer = CoherenceAnalyzer.__new__(CoherenceAnalyzer)
        only_a, only_b, shared = analyzer._match_communities(result_a, result_b)
        # Should NOT match pos to neg — both should appear as unmatched
        assert len(shared) == 0
        assert len(only_a) == 1
        assert len(only_b) == 1


# =============================================================================
# Cycle 3: Minor Polish
# =============================================================================


class TestARCHVIII4_EmptyBioMatrixRepr:
    """ARCH-VIII-4: __repr__ doesn't crash on empty BioMatrix."""

    def test_empty_features(self):
        from cliquefinder.core.biomatrix import BioMatrix
        data = np.empty((0, 2))
        m = BioMatrix(
            data=data,
            feature_ids=pd.Index([]),
            sample_ids=pd.Index(["S1", "S2"]),
            sample_metadata=pd.DataFrame({"cond": ["A", "B"]}, index=["S1", "S2"]),
            quality_flags=np.empty((0, 2)),
        )
        r = repr(m)
        assert "0 features" in r
        assert "(empty)" in r

    def test_empty_samples(self):
        from cliquefinder.core.biomatrix import BioMatrix
        data = np.empty((2, 0))
        m = BioMatrix(
            data=data,
            feature_ids=pd.Index(["G1", "G2"]),
            sample_ids=pd.Index([]),
            sample_metadata=pd.DataFrame(index=pd.Index([])),
            quality_flags=np.empty((2, 0)),
        )
        r = repr(m)
        assert "0 samples" in r
