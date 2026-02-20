"""Tests for negative control gene sets analysis."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.negative_controls import (
    NegativeControlResult,
    run_negative_control_sets,
)


class MockRotationResult:
    """Minimal mock of RotationResult dataclass."""

    def __init__(self, set_id, n_genes, pvalue):
        self.feature_set_id = set_id
        self.n_genes = n_genes
        self.p_values = {"msq": {"mixed": pvalue}}


class MockRotationEngine:
    """Minimal mock of RotationTestEngine for testing."""

    def __init__(self, gene_ids, pvalue_func=None):
        self.gene_ids = gene_ids
        self.gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        self._fitted = True
        self._pvalue_func = pvalue_func or (lambda _: 0.5)

    def test_gene_set(self, gene_set, gene_set_id):
        """Return mock RotationResult (matches real API)."""
        pval = self._pvalue_func(gene_set)
        return MockRotationResult(gene_set_id, len(gene_set), pval)


class TestRunNegativeControlSets:
    """Tests for run_negative_control_sets()."""

    def test_basic_run(self):
        """Basic negative control run completes."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="test_targets",
            n_control_sets=20,
            seed=42,
            verbose=False,
        )

        assert isinstance(result, NegativeControlResult)
        assert result.n_control_sets == 20
        assert result.target_set_size == 10
        assert result.target_set_id == "test_targets"

    def test_fpr_with_uniform_pvalues(self):
        """With uniform p-values, FPR should approximate alpha."""
        rng = np.random.default_rng(42)
        gene_ids = [f"gene_{i}" for i in range(200)]

        # Return random p-values for all sets (null scenario)
        counter = {"i": 0}
        random_pvals = rng.uniform(0, 1, size=300)

        def uniform_pval(genes):
            idx = counter["i"]
            counter["i"] += 1
            return float(random_pvals[idx % len(random_pvals)])

        engine = MockRotationEngine(gene_ids, pvalue_func=uniform_pval)
        target_genes = gene_ids[:15]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="null_target",
            n_control_sets=100,
            alpha=0.05,
            seed=42,
            verbose=False,
        )

        # FPR should be roughly 0.05 (with some variance)
        assert 0.0 <= result.fpr <= 0.30  # Wide tolerance for randomness

    def test_strong_target_low_percentile(self):
        """Target with very low p-value should have low percentile."""
        gene_ids = [f"gene_{i}" for i in range(100)]

        # Target gets p=0.001, controls get p=0.5
        call_count = {"n": 0}

        def target_vs_control(genes):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return 0.001  # Target
            return 0.5  # Controls

        engine = MockRotationEngine(gene_ids, pvalue_func=target_vs_control)
        target_genes = gene_ids[:10]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="strong_target",
            n_control_sets=50,
            seed=42,
            verbose=False,
        )

        assert result.target_pvalue == pytest.approx(0.001)
        assert result.target_percentile < 10.0  # Should be in bottom decile

    def test_not_fitted_raises(self):
        """Unfitted engine → RuntimeError."""
        gene_ids = [f"gene_{i}" for i in range(50)]
        engine = MockRotationEngine(gene_ids)
        engine._fitted = False

        with pytest.raises(RuntimeError, match="must be fitted"):
            run_negative_control_sets(
                engine=engine,
                target_gene_ids=gene_ids[:5],
                target_set_id="test",
                verbose=False,
            )

    def test_no_targets_found_raises(self):
        """No target genes in universe → ValueError."""
        gene_ids = [f"gene_{i}" for i in range(50)]
        engine = MockRotationEngine(gene_ids)

        with pytest.raises(ValueError, match="No target genes"):
            run_negative_control_sets(
                engine=engine,
                target_gene_ids=["missing_1", "missing_2"],
                target_set_id="test",
                verbose=False,
            )

    def test_all_genes_as_targets_raises(self):
        """All genes as targets → ValueError."""
        gene_ids = [f"gene_{i}" for i in range(10)]
        engine = MockRotationEngine(gene_ids)

        with pytest.raises(ValueError, match="covers all"):
            run_negative_control_sets(
                engine=engine,
                target_gene_ids=gene_ids,
                target_set_id="test",
                verbose=False,
            )

    def test_to_dict(self):
        """Result serializes to dict."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=gene_ids[:10],
            target_set_id="test",
            n_control_sets=10,
            seed=42,
            verbose=False,
        )

        d = result.to_dict()
        assert "fpr" in d
        assert "target_percentile" in d
        assert "control_pvalue_quantiles" in d

    def test_partial_target_overlap(self):
        """Only targets that exist in gene universe are used."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)

        # Mix of existing and non-existing targets
        target_genes = gene_ids[:5] + ["missing_1", "missing_2"]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="partial",
            n_control_sets=10,
            seed=42,
            verbose=False,
        )

        assert result.target_set_size == 5  # Only the 5 that exist


class TestCompetitiveZIntegration:
    """Tests for competitive z-score integration in negative controls."""

    def _make_protein_results(self, gene_ids, target_genes, target_t=3.0, bg_t=0.0):
        """Build a protein_results DataFrame with known t-statistics."""
        rng = np.random.default_rng(42)
        n = len(gene_ids)
        target_set = set(target_genes)
        t_stats = rng.normal(bg_t, 0.5, size=n)
        is_target = np.zeros(n, dtype=bool)
        for i, g in enumerate(gene_ids):
            if g in target_set:
                t_stats[i] = target_t + rng.normal(0, 0.3)
                is_target[i] = True
        return pd.DataFrame({
            "feature_id": gene_ids,
            "t_statistic": t_stats,
            "is_target": is_target,
        })

    def test_competitive_z_with_protein_results(self):
        """Competitive z computed when protein_results provided."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]

        protein_df = self._make_protein_results(gene_ids, target_genes, target_t=4.0)

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="z_test",
            n_control_sets=20,
            seed=42,
            protein_results=protein_df,
            verbose=False,
        )

        assert result.target_competitive_z is not None
        assert result.control_competitive_z_scores is not None
        assert result.competitive_z_fpr is not None
        assert result.competitive_z_percentile is not None
        assert result.target_competitive_z > 0  # Targets have higher |t|

    def test_competitive_z_without_protein_results(self):
        """Without protein_results, all competitive z fields are None."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="no_z_test",
            n_control_sets=10,
            seed=42,
            verbose=False,
        )

        assert result.target_competitive_z is None
        assert result.control_competitive_z_scores is None
        assert result.competitive_z_fpr is None
        assert result.competitive_z_percentile is None

    def test_competitive_z_strong_signal(self):
        """Targets with high |t| → low competitive z FPR."""
        gene_ids = [f"gene_{i}" for i in range(200)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:15]

        # Targets: very high |t|, background: near zero
        protein_df = self._make_protein_results(
            gene_ids, target_genes, target_t=5.0, bg_t=0.0,
        )

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="strong_signal",
            n_control_sets=50,
            seed=42,
            protein_results=protein_df,
            verbose=False,
        )

        # With strong signal, most random sets should have lower z
        assert result.competitive_z_fpr < 0.2
        assert result.competitive_z_percentile < 20.0

    def test_to_dict_includes_competitive_z(self):
        """Serialization includes competitive_z sub-dict when populated."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]
        protein_df = self._make_protein_results(gene_ids, target_genes)

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="dict_test",
            n_control_sets=10,
            seed=42,
            protein_results=protein_df,
            verbose=False,
        )

        d = result.to_dict()
        assert "competitive_z" in d
        assert "target_z" in d["competitive_z"]
        assert "fpr" in d["competitive_z"]
        assert "percentile" in d["competitive_z"]
        assert "control_z_quantiles" in d["competitive_z"]


class TestExpressionMatchedControls:
    """Tests for expression-matched negative control sampling (M-4)."""

    def test_expression_matched_sampling(self):
        """Matched controls have similar mean expression to targets."""
        rng = np.random.default_rng(42)
        n_features = 200
        n_samples = 30
        gene_ids = [f"gene_{i}" for i in range(n_features)]

        # Create data where targets have higher mean expression
        data = rng.normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 3.0  # Targets: elevated expression

        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:15]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="matched_test",
            n_control_sets=20,
            seed=42,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        assert result.matched_control_pvalues is not None
        assert result.matched_fpr is not None
        assert result.matched_target_percentile is not None
        assert len(result.matched_control_pvalues) > 0

    def test_matching_both_mode(self):
        """Both uniform and matched results populated in 'both' mode."""
        rng = np.random.default_rng(42)
        n_features = 100
        n_samples = 20
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        data = rng.normal(0, 1, size=(n_features, n_samples))

        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="both_test",
            n_control_sets=10,
            seed=42,
            data=data,
            matching="both",
            verbose=False,
        )

        # Uniform results always populated
        assert result.control_pvalues is not None
        assert result.fpr is not None

        # Matched results also populated
        assert result.matched_control_pvalues is not None
        assert result.matched_fpr is not None

    def test_matching_uniform_default(self):
        """Without data, identical to pre-change (no matched results)."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        engine = MockRotationEngine(gene_ids)
        target_genes = gene_ids[:10]

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="uniform_default",
            n_control_sets=10,
            seed=42,
            verbose=False,
        )

        assert result.matched_control_pvalues is None
        assert result.matched_fpr is None
        assert result.matched_target_percentile is None

    def test_matched_controls_similar_expression(self):
        """Matched controls have more similar mean expression than uniform."""
        from cliquefinder.stats.negative_controls import _sample_expression_matched_set

        rng = np.random.default_rng(42)
        n_features = 200
        n_samples = 30

        # Create data where targets (indices 0-14) have mean ~5
        # and most background has mean ~0
        data = rng.normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 5.0  # Targets: much higher expression

        gene_means = np.nanmean(data, axis=1)
        gene_variances = np.nanvar(data, axis=1, ddof=1)

        target_indices = list(range(15))
        non_target_indices = np.arange(15, n_features, dtype=np.intp)

        matched = _sample_expression_matched_set(
            target_indices, non_target_indices,
            gene_means, gene_variances, rng,
        )

        # Matched genes should have higher mean than random sample
        matched_mean = np.mean(gene_means[matched])
        random_indices = rng.choice(non_target_indices, size=15, replace=False)
        random_mean = np.mean(gene_means[random_indices])
        target_mean = np.mean(gene_means[target_indices])

        # Matched controls should be closer to target mean than random
        assert abs(matched_mean - target_mean) < abs(random_mean - target_mean)
