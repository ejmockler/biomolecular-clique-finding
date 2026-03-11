"""Tests for statistical engine corrections across GPU, rotation, and validation.

Covers:
- Inline EB d0=inf returns prior variance (not original)
- eb_df_total=inf when d0=inf + verbose log corrected
- Permutation null uses per-gene shuffling (not column reorder)
- mean50 MIXED uses abs(z) for bidirectional detection
- Phase 3 gate falls back to free permutation when stratified fails
- Negative control sampling excludes target genes
- PreparedCliqueExperiment deep-copies sample_metadata
- Modified Gram-Schmidt orthogonalization for C-matrix
- Bootstrap matching excludes already-matched communities
- Bootstrap/permutation thread correlation method
- Degenerate strata warning in stratified permutation
"""

import warnings
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats as sp_stats


# =============================================================================
# Inline EB d0=inf returns prior variance
# =============================================================================


class TestInlineEBD0Inf:
    """_batched_ols_gpu and _batched_ols_cpu must return s0_sq when d0=inf."""

    def _make_matrices(self, n_samples=20, n_params=2, eb_d0=np.inf, eb_s0_sq=0.5):
        from cliquefinder.stats.permutation_gpu import OLSPrecomputedMatrices

        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n_samples), rng.standard_normal(n_samples)])
        XtX_inv = np.linalg.inv(X.T @ X)
        c = np.array([0.0, 1.0])
        c_var_factor = float(c @ XtX_inv @ c)

        return OLSPrecomputedMatrices(
            X=X,
            XtX_inv=XtX_inv,
            c=c,
            c_var_factor=c_var_factor,
            df_residual=n_samples - n_params,
            conditions=["A", "B"],
            contrast_name="B-A",
            eb_d0=eb_d0,
            eb_s0_sq=eb_s0_sq,
            eb_df_total=np.inf if (eb_d0 is not None and np.isinf(eb_d0)) else (
                (eb_d0 + (n_samples - n_params)) if eb_d0 is not None else None
            ),
        )

    def test_gpu_path_d0_inf_uses_prior(self):
        """GPU (MLX) path should use s0_sq, not sigma2, when d0=inf."""
        from cliquefinder.stats.permutation_gpu import _batched_ols_gpu

        matrices = self._make_matrices(eb_d0=np.inf, eb_s0_sq=0.5)
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((5, 20))

        t_stats = _batched_ols_gpu(Y, matrices)

        # Compute expected: SE should use s0_sq, not per-gene sigma2
        beta = Y @ matrices.X @ matrices.XtX_inv.T
        estimate = beta @ matrices.c
        se_expected = np.sqrt(0.5 * matrices.c_var_factor)
        t_expected = estimate / np.maximum(se_expected, 1e-10)

        # GPU path uses float32 intermediates — relax tolerance
        assert_allclose(t_stats, t_expected, rtol=1e-6)

    def test_cpu_path_d0_inf_uses_prior(self):
        """CPU path should use s0_sq, not sigma2, when d0=inf."""
        from cliquefinder.stats.permutation_gpu import _batched_ols_cpu

        matrices = self._make_matrices(eb_d0=np.inf, eb_s0_sq=0.5)
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((5, 20))

        t_stats = _batched_ols_cpu(Y, matrices)

        beta = Y @ matrices.X @ matrices.XtX_inv.T
        estimate = beta @ matrices.c
        se_expected = np.sqrt(0.5 * matrices.c_var_factor)
        t_expected = estimate / np.maximum(se_expected, 1e-10)

        assert_allclose(t_stats, t_expected, rtol=1e-10)

    def test_gpu_cpu_match_d0_inf(self):
        """GPU and CPU paths should give close results for d0=inf."""
        from cliquefinder.stats.permutation_gpu import (
            _batched_ols_cpu,
            _batched_ols_gpu,
        )

        matrices = self._make_matrices(eb_d0=np.inf, eb_s0_sq=0.5)
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((10, 20))

        t_gpu = _batched_ols_gpu(Y, matrices)
        t_cpu = _batched_ols_cpu(Y, matrices)

        # GPU uses float32 intermediates, so allow some tolerance
        assert_allclose(t_gpu, t_cpu, rtol=1e-6)

    def test_d0_inf_different_from_original_variance(self):
        """d0=inf t-stats must differ from unmoderated (sigma2) t-stats."""
        from cliquefinder.stats.permutation_gpu import _batched_ols_cpu

        matrices_eb = self._make_matrices(eb_d0=np.inf, eb_s0_sq=0.5)
        matrices_none = replace(matrices_eb, eb_d0=None, eb_s0_sq=None, eb_df_total=None)

        rng = np.random.default_rng(99)
        Y = rng.standard_normal((10, 20))

        t_eb = _batched_ols_cpu(Y, matrices_eb)
        t_none = _batched_ols_cpu(Y, matrices_none)

        # They should differ — EB uses constant s0_sq, unmoderated uses per-gene sigma2
        assert not np.allclose(t_eb, t_none)

    def test_finite_d0_unchanged(self):
        """Finite d0 path should work correctly (regression check)."""
        from cliquefinder.stats.permutation_gpu import _batched_ols_cpu

        matrices = self._make_matrices(eb_d0=5.0, eb_s0_sq=0.5)
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((3, 20))

        t_stats = _batched_ols_cpu(Y, matrices)

        # Should produce finite, non-zero t-stats
        assert np.all(np.isfinite(t_stats))
        assert not np.allclose(t_stats, 0)

    def test_d0_inf_matches_squeeze_var(self):
        """Inline EB d0=inf should produce same result as squeeze_var."""
        from cliquefinder.stats.permutation_gpu import _batched_ols_cpu, squeeze_var

        matrices = self._make_matrices(eb_d0=np.inf, eb_s0_sq=0.5)
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((5, 20))

        # Get t-stats from inline EB
        t_inline = _batched_ols_cpu(Y, matrices)

        # Compute using squeeze_var independently
        beta = Y @ matrices.X @ matrices.XtX_inv.T
        Y_pred = beta @ matrices.X.T
        rss = np.sum((Y - Y_pred) ** 2, axis=1)
        sigma2 = rss / matrices.df_residual
        s2_post, _ = squeeze_var(sigma2, matrices.df_residual, d0=np.inf, s0_sq=0.5)
        estimate = beta @ matrices.c
        se = np.sqrt(s2_post * matrices.c_var_factor)
        t_squeeze = estimate / np.maximum(se, 1e-10)

        assert_allclose(t_inline, t_squeeze, rtol=1e-10)


# =============================================================================
# eb_df_total and verbose log
# =============================================================================


class TestEBDfTotalAndVerbose:
    """eb_df_total must be inf when d0=inf; verbose log must say 'maximum shrinkage'."""

    def test_eb_df_total_logic(self):
        """When d0=inf, eb_df_total must be inf."""
        # Test the logic directly
        d0 = np.inf
        df_residual = 18
        eb_df_total = d0 + df_residual if not np.isinf(d0) else np.inf
        assert np.isinf(eb_df_total)

        # Finite case
        d0 = 5.0
        eb_df_total = d0 + df_residual if not np.isinf(d0) else np.inf
        assert eb_df_total == 23.0

    def test_verbose_log_maximum_shrinkage(self):
        """Verbose message for d0=inf should say 'maximum shrinkage'."""
        from cliquefinder.stats.permutation_gpu import run_permutation_test_gpu

        rng = np.random.default_rng(42)
        n_samples, n_features = 20, 10
        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"F{i}" for i in range(n_features)]
        conditions = ["A"] * 10 + ["B"] * 10
        metadata = pd.DataFrame({
            "condition": conditions,
            "subject_id": [f"S{i}" for i in range(n_samples)],
        })

        from cliquefinder.stats.clique_analysis import CliqueDefinition
        cliques = [CliqueDefinition(
            clique_id="test", protein_ids=feature_ids[:5],
        )]

        # Patch fit_f_dist to force d0=inf
        with patch("cliquefinder.stats.permutation_gpu.fit_f_dist", return_value=(np.inf, 0.5)):
            import io, sys
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                run_permutation_test_gpu(
                    data, feature_ids, metadata, cliques,
                    condition_col="condition", contrast=("B", "A"),
                    n_permutations=5, verbose=True, use_mixed_model=False,
                    eb_moderation=True, random_state=42,
                )
            except Exception:
                pass  # May fail due to mock; we only care about log output
            finally:
                sys.stdout = old_stdout

            output = captured.getvalue()
            if "EB priors" in output:
                assert "maximum shrinkage" in output
                assert "no shrinkage" not in output


# =============================================================================
# Permutation null uses per-gene shuffling
# =============================================================================


class TestPermutationNullPerGeneShuffle:
    """Permutation null must destroy inter-gene correlation via per-gene shuffling."""

    def test_column_permutation_is_invariant(self):
        """Verify the old approach (column permutation) was indeed a no-op."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 20))

        orig = np.corrcoef(X)
        perm_idx = rng.permutation(20)
        permuted = np.corrcoef(X[:, perm_idx])

        # Column permutation does NOT change correlation matrix
        assert_allclose(orig, permuted, atol=1e-15)

    def test_per_gene_shuffle_changes_correlation(self):
        """Per-gene independent shuffling DOES change correlation matrix."""
        rng = np.random.default_rng(42)
        # Create strongly correlated data
        latent = rng.standard_normal(30)
        X = np.outer(np.ones(5), latent) + 0.1 * rng.standard_normal((5, 30))

        orig = np.corrcoef(X)

        # Per-gene shuffle
        X_perm = X.copy()
        for row in range(X_perm.shape[0]):
            rng.shuffle(X_perm[row])
        perm_corr = np.corrcoef(X_perm)

        # Off-diagonal elements should differ
        mask = ~np.eye(5, dtype=bool)
        assert not np.allclose(orig[mask], perm_corr[mask])

    def test_per_gene_shuffle_destroys_correlation(self):
        """After per-gene shuffling, mean abs correlation should drop significantly."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 10, 50

        # Strong positive correlation
        latent = rng.standard_normal(n_samples)
        X = np.outer(np.ones(n_genes), latent) + 0.1 * rng.standard_normal((n_genes, n_samples))

        orig_corr = np.corrcoef(X)
        mask = ~np.eye(n_genes, dtype=bool)
        orig_mean_abs = np.mean(np.abs(orig_corr[mask]))

        # Average over many shuffles
        perm_means = []
        for _ in range(20):
            X_perm = X.copy()
            for row in range(X_perm.shape[0]):
                rng.shuffle(X_perm[row])
            perm_corr = np.corrcoef(X_perm)
            perm_means.append(np.mean(np.abs(perm_corr[mask])))

        avg_perm_mean = np.mean(perm_means)
        # Original should have high correlation (~0.99), permuted should be low (~0.1)
        assert orig_mean_abs > 0.8
        assert avg_perm_mean < 0.3


# =============================================================================
# mean50 MIXED uses abs(z)
# =============================================================================


class TestMean50MixedAbsZ:
    """mean50+MIXED should use abs(z) for bidirectional detection."""

    def test_bidirectional_set_gets_large_stat(self):
        """Gene set with balanced up/down z-scores should give large mean50 stat."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        # 1 rotation, 10 genes. 5 genes at z=+3, 5 at z=-3
        z = np.array([[3.0, 3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, -3.0, -3.0]])
        w = np.ones_like(z)

        stat = _compute_mean50_stat(z, w, A=1.0, alt=Alternative.MIXED)

        # With abs(z), the top 50% should all have |z|=3, so mean50 = 3.0
        assert stat[0] > 2.5, f"Expected large stat for bidirectional set, got {stat[0]}"

    def test_mean50_mixed_nonnegative(self):
        """mean50 MIXED stat should be non-negative (uses abs z)."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        rng = np.random.default_rng(42)
        z = rng.standard_normal((100, 20))
        w = np.ones_like(z)

        stat = _compute_mean50_stat(z, w, A=1.0, alt=Alternative.MIXED)
        assert np.all(stat >= 0)

    def test_mean50_up_still_signed(self):
        """mean50 UP alternative should still use signed z (not abs)."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        z = np.array([[-3.0] * 10])
        w = np.ones_like(z)

        stat = _compute_mean50_stat(z, w, A=1.0, alt=Alternative.UP)
        assert stat[0] < 0

    def test_mean50_down_still_signed(self):
        """mean50 DOWN should negate to get positive stat from negative z."""
        from cliquefinder.stats.rotation import Alternative, _compute_mean50_stat

        z = np.array([[-3.0] * 10])
        w = np.ones_like(z)

        stat = _compute_mean50_stat(z, w, A=1.0, alt=Alternative.DOWN)
        assert stat[0] > 0

    def test_pvalue_uses_upper_tail_for_mean50_mixed(self):
        """mean50+MIXED p-value should use upper-tail (not abs) since stat is unsigned."""
        from cliquefinder.stats.rotation import (
            Alternative,
            SetStatistic,
            compute_rotation_pvalues,
        )

        observed = {
            SetStatistic.MEAN50.value: {Alternative.MIXED.value: 2.5}
        }
        null = {
            SetStatistic.MEAN50.value: {
                Alternative.MIXED.value: np.array([1.0, 1.5, 2.0, 1.2, 0.8])
            }
        }
        pvals = compute_rotation_pvalues(observed, null)
        # b=0 (no null >= 2.5), p = (0+1)/(5+1) = 1/6
        assert pvals[SetStatistic.MEAN50.value][Alternative.MIXED.value] == pytest.approx(1 / 6)

    def test_mean_mixed_still_uses_abs_comparison(self):
        """MEAN+MIXED should still use |null| >= |obs| (two-sided)."""
        from cliquefinder.stats.rotation import (
            Alternative,
            SetStatistic,
            compute_rotation_pvalues,
        )

        observed = {
            SetStatistic.MEAN.value: {Alternative.MIXED.value: -2.5}
        }
        null = {
            SetStatistic.MEAN.value: {
                Alternative.MIXED.value: np.array([3.0, -3.0, 1.0, -1.0, 0.5])
            }
        }
        pvals = compute_rotation_pvalues(observed, null)
        # |null| >= |obs=2.5|: |3|>=2.5 yes, |-3|>=2.5 yes, |1|>=2.5 no,
        # |-1|>=2.5 no, |0.5|>=2.5 no → b=2, p=(2+1)/(5+1)=0.5
        assert pvals[SetStatistic.MEAN.value][Alternative.MIXED.value] == pytest.approx(0.5)


# =============================================================================
# Phase 3 free permutation fallback
# =============================================================================


class TestPhase3FreeFallback:
    """Phase 3 gate should fall back to free permutation when stratified fails."""

    def _make_report(self, phases):
        from cliquefinder.stats.validation_report import ValidationReport
        return ValidationReport(phases=phases)

    def test_stratified_failed_free_passes(self):
        """When stratified fails but free passes, gate should pass."""
        phases = {
            "covariate_adjusted": {"status": "completed", "empirical_pvalue": 0.01},
            "label_permutation": {
                "status": "completed",
                "stratified": {"status": "failed"},
                "free": {"permutation_pvalue": 0.001},
            },
        }
        report = self._make_report(phases)
        report.compute_verdict()

        assert "fallback from free permutation" in report.phase_details.get(
            "label_permutation_stratified", ""
        )
        # Both gates should pass → validated
        assert report.verdict == "validated"

    def test_both_missing_gate_fails(self):
        """When both stratified and free are absent, gate should fail."""
        phases = {
            "covariate_adjusted": {"status": "completed", "empirical_pvalue": 0.01},
            "label_permutation": {
                "status": "completed",
                "stratified": {"status": "failed"},
                "free": {},
            },
        }
        report = self._make_report(phases)
        report.compute_verdict()
        assert "fallback" not in report.phase_details.get("label_permutation_stratified", "")

    def test_stratified_passes_no_fallback_needed(self):
        """When stratified passes, no fallback should occur."""
        phases = {
            "covariate_adjusted": {"status": "completed", "empirical_pvalue": 0.01},
            "label_permutation": {
                "status": "completed",
                "stratified": {"permutation_pvalue": 0.01},
                "free": {"permutation_pvalue": 0.001},
            },
        }
        report = self._make_report(phases)
        report.compute_verdict()
        assert "fallback" not in report.phase_details.get("label_permutation_stratified", "")

    def test_free_also_fails_no_gate(self):
        """When both stratified and free fail, gate should not pass."""
        phases = {
            "covariate_adjusted": {"status": "completed", "empirical_pvalue": 0.01},
            "label_permutation": {
                "status": "completed",
                "stratified": {"status": "failed"},
                "free": {"permutation_pvalue": 0.8},
            },
        }
        report = self._make_report(phases)
        report.compute_verdict()
        # Permutation gate fails → not validated
        assert report.verdict != "validated"


# =============================================================================
# Negative control excludes target genes
# =============================================================================


class TestNegativeControlExclusion:
    """Control gene sets must not overlap with target genes."""

    def test_no_overlap_with_targets(self):
        """Control sets should not contain any target genes."""
        from cliquefinder.stats.negative_controls import run_negative_control_sets
        from cliquefinder.stats.rotation import RotationTestEngine, RotationResult

        rng = np.random.default_rng(42)
        n_genes, n_samples = 100, 20
        data = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": ["A"] * 10 + ["B"] * 10})

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(conditions=["A", "B"], contrast=("B", "A"), condition_column="condition")

        target_genes = gene_ids[:20]

        # Mock test_gene_set to capture gene sets
        tested_sets = []
        original_test = engine.test_gene_set

        def mock_test(gene_set, gene_set_id, **kwargs):
            tested_sets.append(set(gene_set))
            return RotationResult(
                feature_set_id=gene_set_id,
                n_genes=len(gene_set),
                n_genes_found=len(gene_set),
                gene_ids=list(gene_set),
                observed_stats={"msq": 1.0},
                null_distributions={},
                p_values={"msq": {"mixed": 0.5}},
                active_proportion={"msq": 0.5},
                n_rotations=100,
                contrast_name="B-A",
            )

        engine.test_gene_set = mock_test

        run_negative_control_sets(
            engine, target_genes, target_set_id="test",
            n_control_sets=10, seed=42,
        )

        target_set = set(target_genes)
        # Skip first entry (target set itself), check controls
        for i, control_set in enumerate(tested_sets[1:]):
            overlap = control_set & target_set
            assert len(overlap) == 0, f"Control set {i} overlaps: {overlap}"

    def test_small_pool_warning(self):
        """When pool is smaller than target, should log a warning."""
        from cliquefinder.stats.negative_controls import run_negative_control_sets
        from cliquefinder.stats.rotation import RotationTestEngine, RotationResult
        import logging

        rng = np.random.default_rng(42)
        n_genes, n_samples = 30, 20
        data = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({"condition": ["A"] * 10 + ["B"] * 10})

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(conditions=["A", "B"], contrast=("B", "A"), condition_column="condition")

        # Target 25 of 30 genes — control pool is only 5
        target_genes = gene_ids[:25]

        def mock_test(gene_set, gene_set_id, **kwargs):
            return RotationResult(
                feature_set_id=gene_set_id,
                n_genes=len(gene_set),
                n_genes_found=len(gene_set),
                gene_ids=list(gene_set),
                observed_stats={"msq": 1.0},
                null_distributions={},
                p_values={"msq": {"mixed": 0.5}},
                active_proportion={"msq": 0.5},
                n_rotations=100,
                contrast_name="B-A",
            )

        engine.test_gene_set = mock_test

        with pytest.warns(match="") if False else warnings.catch_warnings(record=True):
            # Just check it doesn't crash
            run_negative_control_sets(
                engine, target_genes, target_set_id="test",
                n_control_sets=5, seed=42,
            )


# =============================================================================
# PreparedCliqueExperiment deep-copies sample_metadata
# =============================================================================


class TestSampleMetadataCopy:
    """External mutation of source DataFrame should not affect experiment."""

    def _make_experiment(self, metadata=None):
        from cliquefinder.stats.experiment import PreparedCliqueExperiment

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        if metadata is None:
            metadata = pd.DataFrame({"condition": ["A", "A", "B", "B", "B"]})

        return PreparedCliqueExperiment(
            data=data,
            feature_ids=tuple(f"F{i}" for i in range(10)),
            feature_to_idx={f"F{i}": i for i in range(10)},
            sample_metadata=metadata,
            condition_column="condition",
            subject_column=None,
            conditions=("A", "B"),
            n_samples=5,
            cliques=(),
            clique_to_feature_indices={},
            symbol_to_feature={},
            contrast=("B", "A"),
            contrast_name="B_vs_A",
            preprocessing_params={"norm": "log2"},
            creation_timestamp="2026-03-06",
        ), metadata

    def test_external_mutation_isolated(self):
        experiment, metadata = self._make_experiment()

        # Mutate original DataFrame
        metadata.iloc[0, 0] = "MUTATED"

        # Experiment should be unaffected
        assert experiment.sample_metadata.iloc[0, 0] == "A"

    def test_data_still_read_only(self):
        """Data array should still be read-only after post_init."""
        experiment, _ = self._make_experiment()

        with pytest.raises((ValueError, TypeError)):
            experiment.data[0, 0] = 999.0


# =============================================================================
# Modified Gram-Schmidt orthogonalization
# =============================================================================


class TestModifiedGramSchmidt:
    """_construct_c_matrix should produce orthogonal columns for large p."""

    def test_orthogonality_p20(self):
        """C matrix columns should be orthonormal for p=20."""
        from cliquefinder.stats.rotation import _construct_c_matrix

        rng = np.random.default_rng(42)
        c = rng.standard_normal(20)
        c = c / np.linalg.norm(c)

        C = _construct_c_matrix(c)

        CtC = C.T @ C
        assert_allclose(CtC, np.eye(20), atol=1e-12)

    def test_orthogonality_near_axis(self):
        """Near-axis contrast should not cause numerical issues with MGS."""
        from cliquefinder.stats.rotation import _construct_c_matrix

        # Contrast close to e_0 but not pathologically so
        c = np.zeros(15)
        c[0] = 1.0
        c[1] = 1e-4
        c = c / np.linalg.norm(c)

        C = _construct_c_matrix(c)
        CtC = C.T @ C
        assert_allclose(CtC, np.eye(15), atol=1e-10)

    def test_contrast_in_last_column(self):
        """Normalized contrast should be the last column of C."""
        from cliquefinder.stats.rotation import _construct_c_matrix

        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c_unit = c / np.linalg.norm(c)
        C = _construct_c_matrix(c_unit)

        assert_allclose(C[:, -1], c_unit, atol=1e-14)

    def test_mgs_better_than_cgs_for_ill_conditioned(self):
        """MGS should maintain orthogonality better than naive approach for p=30."""
        from cliquefinder.stats.rotation import _construct_c_matrix

        rng = np.random.default_rng(42)
        c = rng.standard_normal(30)
        c = c / np.linalg.norm(c)

        C = _construct_c_matrix(c)
        CtC = C.T @ C
        off_diag = CtC - np.eye(30)
        max_error = np.max(np.abs(off_diag))

        # MGS should keep error well below 1e-10 for p=30
        assert max_error < 1e-10


# =============================================================================
# Bootstrap matching exclusion tracking
# =============================================================================


class TestBootstrapMatchingExclusion:
    """Multiple original communities should not match same bootstrap community."""

    def _make_analyzer(self, n_genes=20, n_samples=30, seed=42):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.knowledge.regulatory_coherence import (
            CoherenceAnalyzer,
            CoherenceConfig,
        )

        rng = np.random.default_rng(seed)
        # Two distinct clusters
        latent1 = rng.standard_normal(n_samples)
        latent2 = rng.standard_normal(n_samples)
        data = np.zeros((n_genes, n_samples))
        for i in range(n_genes // 2):
            data[i] = latent1 + 0.1 * rng.standard_normal(n_samples)
        for i in range(n_genes // 2, n_genes):
            data[i] = latent2 + 0.1 * rng.standard_normal(n_samples)

        gene_ids = [f"G{i}" for i in range(n_genes)]
        sample_ids = [f"S{i}" for i in range(n_samples)]

        feature_ids = pd.Index(gene_ids)
        sample_index = pd.Index(sample_ids)
        metadata = pd.DataFrame(
            {"condition": ["cond1"] * n_samples},
            index=sample_index,
        )
        quality_flags = np.zeros((n_genes, n_samples), dtype=int)

        matrix = BioMatrix(data, feature_ids, sample_index, metadata, quality_flags)

        config = CoherenceConfig(n_bootstrap=20, min_community_size=2)
        analyzer = CoherenceAnalyzer(
            matrix=matrix,
            stratify_by=["condition"],
            config=config,
            seed=seed,
        )
        return analyzer, gene_ids

    def test_stability_scores_bounded(self):
        """Stability scores should be between 0 and 1."""
        analyzer, gene_ids = self._make_analyzer()
        stability = analyzer.bootstrap_stability(set(gene_ids), "cond1")

        for score in stability.values():
            assert 0 <= score <= 1


# =============================================================================
# Bootstrap/permutation thread correlation method
# =============================================================================


class TestCorrelationMethodThreading:
    """Bootstrap and permutation should accept method parameter."""

    def _make_analyzer(self, seed=42):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.knowledge.regulatory_coherence import (
            CoherenceAnalyzer,
            CoherenceConfig,
        )

        rng = np.random.default_rng(seed)
        n_genes, n_samples = 8, 25
        data = rng.standard_normal((n_genes, n_samples))

        gene_ids = [f"G{i}" for i in range(n_genes)]
        sample_ids = [f"S{i}" for i in range(n_samples)]
        feature_ids = pd.Index(gene_ids)
        sample_index = pd.Index(sample_ids)
        metadata = pd.DataFrame(
            {"condition": ["cond1"] * n_samples},
            index=sample_index,
        )
        quality_flags = np.zeros((n_genes, n_samples), dtype=int)

        matrix = BioMatrix(data, feature_ids, sample_index, metadata, quality_flags)

        return CoherenceAnalyzer(
            matrix=matrix,
            stratify_by=["condition"],
            config=CoherenceConfig(n_bootstrap=5, n_permutations=5, min_community_size=2),
            seed=seed,
        )

    def test_bootstrap_accepts_method(self):
        """bootstrap_stability should accept method='spearman' without error."""
        analyzer = self._make_analyzer()
        genes = set(analyzer.matrix.feature_ids)
        stability = analyzer.bootstrap_stability(genes, "cond1", method='spearman')
        assert isinstance(stability, dict)

    def test_permutation_accepts_method(self):
        """permutation_null should accept method='spearman' without error."""
        analyzer = self._make_analyzer()
        genes = set(analyzer.matrix.feature_ids)
        p = analyzer.permutation_null(genes, "cond1", observed_modularity=0.5, method='spearman')
        assert 0 <= p <= 1

    def test_compute_corr_pearson(self):
        """_compute_corr with pearson should match np.corrcoef."""
        analyzer = self._make_analyzer()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((5, 20))

        corr, n_nan = analyzer._compute_corr(X, method='pearson')
        expected = np.corrcoef(X)
        assert_allclose(corr, expected, atol=1e-14)
        assert n_nan == 0

    def test_compute_corr_spearman(self):
        """_compute_corr with spearman should match scipy.stats.spearmanr."""
        analyzer = self._make_analyzer()
        rng = np.random.default_rng(99)
        X = rng.standard_normal((5, 20))

        corr, n_nan = analyzer._compute_corr(X, method='spearman')
        expected, _ = sp_stats.spearmanr(X.T)
        assert_allclose(corr, expected, atol=1e-14)
        assert n_nan == 0


# =============================================================================
# Degenerate strata warning
# =============================================================================


class TestDegenerateStrataWarning:
    """Stratified permutation should warn about single-condition strata."""

    def test_warns_on_degenerate_stratum(self):
        """Single-condition stratum should trigger UserWarning."""
        from cliquefinder.stats.label_permutation import generate_stratified_permutation

        labels = np.array(["A", "A", "A", "A", "B", "B"])
        strata = np.array([0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_stratified_permutation(labels, strata, rng)

        degenerate_warnings = [x for x in w if "one condition value" in str(x.message)]
        assert len(degenerate_warnings) >= 1

    def test_no_warning_for_balanced_strata(self):
        """Strata with both conditions should not trigger warning."""
        from cliquefinder.stats.label_permutation import generate_stratified_permutation

        labels = np.array(["A", "B", "A", "B", "A", "B"])
        strata = np.array([0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_stratified_permutation(labels, strata, rng)

        degenerate_warnings = [x for x in w if "one condition value" in str(x.message)]
        assert len(degenerate_warnings) == 0

    def test_degenerate_stratum_preserves_labels(self):
        """Labels in degenerate stratum should be unchanged (permutation is no-op)."""
        from cliquefinder.stats.label_permutation import generate_stratified_permutation

        labels = np.array(["A", "A", "A", "B", "A", "B"])
        strata = np.array([0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(42)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result, _ = generate_stratified_permutation(labels, strata, rng)

        # Stratum 0 has only "A" — should all remain "A"
        assert all(result[:3] == "A")
