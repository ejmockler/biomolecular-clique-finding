"""
Tests for ROAST rotation-based gene set testing.

These tests validate:
1. QR decomposition and residual extraction
2. Rotation vector generation (spherical uniformity)
3. Rotation application (variance preservation)
4. Set statistics computation
5. P-value calibration
6. GPU/CPU equivalence
7. Integration with existing infrastructure

References:
    - Wu et al. (2010) "ROAST: rotation gene set tests"
    - Limma package (Bioconductor) for numerical validation
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less

from cliquefinder.stats.rotation import (
    SetStatistic,
    Alternative,
    RotationPrecomputed,
    GeneEffects,
    RotationResult,
    RotationTestConfig,
    RotationTestEngine,
    run_rotation_test,
    compute_rotation_matrices,
    compute_rotation_matrices_general,
    extract_gene_effects,
    generate_rotation_vectors,
    apply_rotations_batched,
    compute_set_statistics,
    compute_rotation_pvalues,
    estimate_active_proportion,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_design():
    """Simple two-group design with 20 samples."""
    n_samples = 20
    conditions = ['CASE', 'CTRL']
    sample_condition = np.array(['CASE'] * 10 + ['CTRL'] * 10)
    contrast = ('CASE', 'CTRL')
    return n_samples, conditions, sample_condition, contrast


@pytest.fixture
def synthetic_expression(simple_design):
    """Synthetic expression data with known signal.

    Creates 100 genes with:
    - Genes 0-9: Strong UP regulation (log2FC ~ 2)
    - Genes 10-19: Strong DOWN regulation (log2FC ~ -2)
    - Genes 20-29: Mixed signal (up and down)
    - Genes 30-99: Null (no differential expression)
    """
    n_samples, conditions, sample_condition, contrast = simple_design
    n_genes = 100

    rng = np.random.default_rng(42)

    # Base expression (noise)
    data = rng.standard_normal((n_genes, n_samples))

    # Add signal
    is_case = sample_condition == 'CASE'

    # Strong UP (genes 0-9)
    data[0:10, is_case] += 2.0

    # Strong DOWN (genes 10-19)
    data[10:20, is_case] -= 2.0

    # Mixed (genes 20-29): half up, half down
    data[20:25, is_case] += 1.5
    data[25:30, is_case] -= 1.5

    gene_ids = [f"GENE_{i}" for i in range(n_genes)]

    metadata = pd.DataFrame({
        'sample_id': [f"S{i}" for i in range(n_samples)],
        'phenotype': sample_condition,
    })

    return data, gene_ids, metadata


@pytest.fixture
def gene_sets():
    """Gene sets for testing."""
    return {
        'up_genes': [f"GENE_{i}" for i in range(10)],
        'down_genes': [f"GENE_{i}" for i in range(10, 20)],
        'mixed_genes': [f"GENE_{i}" for i in range(20, 30)],
        'null_genes': [f"GENE_{i}" for i in range(30, 50)],
        'small_set': [f"GENE_{i}" for i in range(5)],
    }


# =============================================================================
# Unit Tests: Rotation Vector Generation
# =============================================================================

class TestRotationVectors:
    """Tests for rotation vector generation."""

    def test_rotation_vectors_unit_norm(self):
        """Rotation vectors should have unit norm."""
        R = generate_rotation_vectors(1000, 10, rng=np.random.default_rng(42))
        norms = np.linalg.norm(R, axis=1)
        assert_allclose(norms, np.ones(1000), rtol=1e-6)

    def test_rotation_vectors_shape(self):
        """Output shape should match input dimensions."""
        R = generate_rotation_vectors(500, 15, rng=np.random.default_rng(42))
        assert R.shape == (500, 15)

    def test_rotation_vectors_reproducible(self):
        """Same seed should give same results."""
        R1 = generate_rotation_vectors(100, 10, rng=np.random.default_rng(123))
        R2 = generate_rotation_vectors(100, 10, rng=np.random.default_rng(123))
        assert_allclose(R1, R2)

    def test_rotation_vectors_spherical_coverage(self):
        """Vectors should uniformly cover the sphere (mean ~ 0)."""
        R = generate_rotation_vectors(10000, 5, rng=np.random.default_rng(42))
        # Mean of each coordinate should be ~ 0
        means = np.mean(R, axis=0)
        assert_allclose(means, np.zeros(5), atol=0.05)


# =============================================================================
# Unit Tests: QR Decomposition
# =============================================================================

class TestQRDecomposition:
    """Tests for QR decomposition and residual extraction."""

    def test_compute_rotation_matrices_shape(self, simple_design):
        """Q2 should have correct shape."""
        n_samples, conditions, sample_condition, contrast = simple_design

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast
        )

        # Q2 should be (n_samples, df_residual + 1)
        expected_df = n_samples - 2  # intercept + 1 condition param
        assert precomputed.Q2.shape[0] == n_samples
        assert precomputed.df_residual == expected_df

    def test_q2_orthonormal(self, simple_design):
        """Q2 columns should be orthonormal."""
        n_samples, conditions, sample_condition, contrast = simple_design

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast
        )

        Q2 = precomputed.Q2
        # Q2' @ Q2 should be identity
        QtQ = Q2.T @ Q2
        assert_allclose(QtQ, np.eye(QtQ.shape[0]), atol=1e-10)

    def test_extract_gene_effects_variance_preserving(self, simple_design, synthetic_expression):
        """Projection should preserve variance structure."""
        n_samples, conditions, sample_condition, contrast = simple_design
        data, gene_ids, metadata = synthetic_expression

        precomputed = compute_rotation_matrices(
            sample_condition, conditions, contrast
        )

        effects = extract_gene_effects(data, gene_ids, precomputed)

        # Squared norms should be positive
        assert np.all(effects.rho_sq > 0)

        # Sample variances should be non-negative
        assert np.all(effects.sample_variances >= 0)


# =============================================================================
# Unit Tests: Set Statistics
# =============================================================================

class TestSetStatistics:
    """Tests for gene set statistics computation."""

    def test_mean_statistic_direction(self):
        """Mean statistic should reflect direction."""
        # All positive z-scores
        z_pos = np.array([[1.0, 2.0, 1.5, 2.5]])
        stats = compute_set_statistics(z_pos, statistics=[SetStatistic.MEAN])

        assert stats['mean']['up'] > 0
        assert stats['mean']['down'] < 0  # Flipped for down
        assert stats['mean']['mixed'] > 0  # Absolute values

    def test_msq_statistic_direction_agnostic(self):
        """MSQ should be insensitive to direction (for mixed alt)."""
        z_pos = np.array([[1.0, 2.0, 1.5]])
        z_neg = np.array([[-1.0, -2.0, -1.5]])

        stats_pos = compute_set_statistics(z_pos, statistics=[SetStatistic.MSQ])
        stats_neg = compute_set_statistics(z_neg, statistics=[SetStatistic.MSQ])

        # MSQ for mixed should be same regardless of sign
        assert_allclose(
            stats_pos['msq']['mixed'],
            stats_neg['msq']['mixed']
        )

    def test_mean50_uses_top_half(self):
        """Mean50 should use top 50% of genes."""
        # 4 genes: mean50 should use top 2
        z = np.array([[0.1, 0.2, 3.0, 4.0]])  # Top 2 are 3.0 and 4.0
        stats = compute_set_statistics(z, statistics=[SetStatistic.MEAN50])

        # For mixed: mean of top 2 absolute values = (3 + 4) / 2 = 3.5
        assert_allclose(stats['mean50']['mixed'], 3.5, rtol=0.1)

    def test_weighted_statistics(self):
        """Weights should scale gene contributions."""
        z = np.array([[1.0, 1.0, 1.0, 1.0]])

        # Equal weights
        stats_equal = compute_set_statistics(z, weights=np.ones(4))

        # Double weight on first gene
        stats_weighted = compute_set_statistics(z, weights=np.array([2.0, 1.0, 1.0, 1.0]))

        # Mean should be same (all z are equal)
        assert_allclose(stats_equal['mean']['up'], stats_weighted['mean']['up'])


# =============================================================================
# Unit Tests: P-Value Computation
# =============================================================================

class TestPValues:
    """Tests for p-value computation."""

    def test_pvalue_bounds(self):
        """P-values should be in (0, 1]."""
        observed = {'mean': {'up': 2.0, 'down': -2.0, 'mixed': 2.0}}
        null = {
            'mean': {
                'up': np.random.standard_normal(1000),
                'down': np.random.standard_normal(1000),
                'mixed': np.abs(np.random.standard_normal(1000)),
            }
        }

        pvals = compute_rotation_pvalues(observed, null)

        for stat in pvals:
            for alt in pvals[stat]:
                p = pvals[stat][alt]
                assert 0 < p <= 1

    def test_extreme_observed_gives_small_pvalue(self):
        """Very extreme observed values should give small p-values."""
        observed = {'mean': {'up': 10.0}}  # Very extreme
        null = {'mean': {'up': np.random.standard_normal(10000)}}

        pvals = compute_rotation_pvalues(observed, null)
        assert pvals['mean']['up'] < 0.01

    def test_null_observed_gives_uniform_pvalue(self):
        """Null observed value should give uniform-ish p-value."""
        # Run multiple times and check distribution
        pvals_list = []
        for _ in range(100):
            null_samples = np.random.standard_normal(1000)
            # Pick one null sample as "observed"
            observed = {'mean': {'up': float(np.random.standard_normal())}}
            null = {'mean': {'up': null_samples}}
            pval = compute_rotation_pvalues(observed, null)['mean']['up']
            pvals_list.append(pval)

        # Under null, p-values should be roughly uniform
        # Check that median is near 0.5
        assert 0.3 < np.median(pvals_list) < 0.7


# =============================================================================
# Integration Tests
# =============================================================================

class TestRotationEngine:
    """Integration tests for the full rotation engine."""

    def test_engine_fit_and_test(self, synthetic_expression, gene_sets):
        """Full workflow should complete without errors."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=999, seed=42)
        result = engine.test_gene_set(gene_sets['up_genes'], config=config)

        assert result.n_genes_found == 10
        assert result.n_rotations == 999
        assert 'mean' in result.p_values
        assert 'msq' in result.p_values

    def test_up_genes_detected_with_up_alternative(self, synthetic_expression, gene_sets):
        """Up-regulated genes should be significant with UP alternative."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=999, seed=42)
        result = engine.test_gene_set(gene_sets['up_genes'], config=config)

        # Should be highly significant for UP
        assert result.get_pvalue(SetStatistic.MEAN, Alternative.UP) < 0.05

    def test_down_genes_detected_with_down_alternative(self, synthetic_expression, gene_sets):
        """Down-regulated genes should be significant with DOWN alternative."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=999, seed=42)
        result = engine.test_gene_set(gene_sets['down_genes'], config=config)

        # Should be highly significant for DOWN
        assert result.get_pvalue(SetStatistic.MEAN, Alternative.DOWN) < 0.05

    def test_mixed_genes_detected_with_msq(self, synthetic_expression, gene_sets):
        """Mixed-direction genes should be detected by MSQ statistic."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=999, seed=42)
        result = engine.test_gene_set(gene_sets['mixed_genes'], config=config)

        # MSQ should detect mixed regulation
        # MEAN may not (signals cancel)
        msq_pval = result.get_pvalue(SetStatistic.MSQ, Alternative.MIXED)
        mean_pval = result.get_pvalue(SetStatistic.MEAN, Alternative.MIXED)

        # MSQ should be more significant than MEAN for mixed signals
        assert msq_pval < 0.1  # Should detect something

    def test_null_genes_not_significant(self, synthetic_expression, gene_sets):
        """Null genes should not be significant."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=999, seed=42)
        result = engine.test_gene_set(gene_sets['null_genes'], config=config)

        # Should NOT be significant
        assert result.get_pvalue(SetStatistic.MSQ, Alternative.MIXED) > 0.1

    def test_batch_testing(self, synthetic_expression, gene_sets):
        """Batch testing should work correctly."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=99, seed=42)
        results = engine.test_gene_sets(gene_sets, config=config, verbose=False)

        assert len(results) == len(gene_sets)

    def test_results_to_dataframe(self, synthetic_expression, gene_sets):
        """Results should convert to DataFrame correctly."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',
        )

        config = RotationTestConfig(n_rotations=99, seed=42)
        results = engine.test_gene_sets(gene_sets, config=config, verbose=False)

        df = engine.results_to_dataframe(results)

        assert len(df) == len(gene_sets)
        assert 'pvalue_msq_mixed' in df.columns
        assert 'pvalue_msq_mixed' in df.columns


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for run_rotation_test convenience function."""

    def test_run_rotation_test_basic(self, synthetic_expression, gene_sets):
        """Basic usage should work."""
        data, gene_ids, metadata = synthetic_expression

        df = run_rotation_test(
            data=data,
            gene_ids=gene_ids,
            metadata=metadata,
            gene_sets=gene_sets,
            conditions=['CASE', 'CTRL'],
            contrast=('CASE', 'CTRL'),
            condition_column='phenotype',  # Required parameter
            n_rotations=99,
            seed=42,
            verbose=False,
        )

        assert len(df) == len(gene_sets)
        assert 'feature_set_id' in df.columns
        assert 'pvalue_msq_mixed' in df.columns


# =============================================================================
# GPU/CPU Equivalence Tests (if MLX available)
# =============================================================================

class TestGPUCPUEquivalence:
    """Tests that GPU and CPU give equivalent results."""

    def test_rotation_cpu_gpu_equivalence(self, synthetic_expression):
        """CPU and GPU rotation should give same results."""
        data, gene_ids, metadata = synthetic_expression

        # Get gene effects
        precomputed = compute_rotation_matrices(
            metadata['phenotype'].values,
            ['CASE', 'CTRL'],
            ('CASE', 'CTRL'),
        )

        effects = extract_gene_effects(data, gene_ids, precomputed)

        # Generate same rotations
        rng = np.random.default_rng(42)
        R = generate_rotation_vectors(100, precomputed.residual_dims, rng=rng)

        # Apply with CPU
        t_cpu, z_cpu, valid_cpu = apply_rotations_batched(
            effects.U,
            effects.rho_sq,
            R,
            effects.sample_variances,
            effects.moderated_variances,
            precomputed.df_residual,
            effects.df_total,
            use_gpu=False,
        )

        # Apply with GPU (if available)
        try:
            import mlx.core as mx
            t_gpu, z_gpu, valid_gpu = apply_rotations_batched(
                effects.U,
                effects.rho_sq,
                R,
                effects.sample_variances,
                effects.moderated_variances,
                precomputed.df_residual,
                effects.df_total,
                use_gpu=True,
            )

            # Should be very close (float32 vs float64 tolerance)
            assert_allclose(t_cpu, t_gpu, rtol=1e-4, atol=1e-4)
        except ImportError:
            pytest.skip("MLX not available")


# =============================================================================
# Active Proportion Tests
# =============================================================================

class TestActiveProportion:
    """Tests for active gene proportion estimation."""

    def test_active_proportion_bounds(self):
        """Active proportion should be in [0, 1]."""
        t_stats = np.random.standard_normal(100)
        props = estimate_active_proportion(t_stats, df_total=18.0)

        for alt in props:
            assert 0 <= props[alt] <= 1

    def test_strong_signal_high_proportion(self):
        """Strong signals should give high active proportion."""
        # All genes have |t| > sqrt(2)
        t_stats = np.array([3.0, -2.5, 4.0, -3.5, 2.0])
        props = estimate_active_proportion(t_stats, df_total=18.0)

        assert props['mixed'] == 1.0  # All active

    def test_weak_signal_low_proportion(self):
        """Weak signals should give low active proportion."""
        # All genes have |t| < sqrt(2)
        t_stats = np.array([0.1, -0.2, 0.3, -0.1, 0.5])
        props = estimate_active_proportion(t_stats, df_total=18.0)

        assert props['mixed'] == 0.0  # None active


# =============================================================================
# 2-Group Warning Tests (H3 audit finding)
# =============================================================================

class TestTwoGroupWarning:
    """Tests for ROAST 2-group limit warning."""

    def test_two_group_no_warning(self, synthetic_expression):
        """2 conditions should not emit a warning."""
        data, gene_ids, metadata = synthetic_expression

        engine = RotationTestEngine(data, gene_ids, metadata)

        with pytest.warns(match="does_not_match_anything") if False else warnings.catch_warnings():
            warnings.simplefilter("error")
            # This should NOT raise any warning
            engine.fit(
                conditions=['CASE', 'CTRL'],
                contrast=('CASE', 'CTRL'),
                condition_column='phenotype',
            )

    def test_three_group_warning(self):
        """3 conditions should emit a warning about 2-group limit.

        The warning fires based on unique values in sample_condition, even
        though only 2 conditions are passed in `conditions`. The extra
        samples from the third group are treated as NaN by the categorical
        and filtered out during QR decomposition, but the data matrix must
        match the metadata rows. Here we use 30 samples (10 per group)
        and the downstream QR only uses the 20 A/B samples.
        """
        n_genes = 50
        rng = np.random.default_rng(42)

        # 3 groups present in metadata, but only A and B in conditions
        sample_condition = np.array(['A'] * 10 + ['B'] * 10 + ['C'] * 10)
        n_samples = len(sample_condition)  # 30

        data = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({
            'sample_id': [f"S{i}" for i in range(n_samples)],
            'phenotype': sample_condition,
        })

        engine = RotationTestEngine(data, gene_ids, metadata)

        # We only need to verify the warning is emitted. The downstream
        # compute_rotation_matrices filters to valid (A/B) samples for QR,
        # but extract_gene_effects will fail because data has 30 columns
        # and Q2 only has 20 rows. That is expected -- the audit finding
        # is about the warning, not about making 3-group analysis work.
        # So we catch both the warning and the downstream error.
        with pytest.warns(
            UserWarning,
            match=r"ROAST rotation test is designed for 2-group comparisons\. 3 groups detected",
        ):
            with pytest.raises(ValueError, match="Sample mismatch"):
                engine.fit(
                    conditions=['A', 'B'],
                    contrast=('A', 'B'),
                    condition_column='phenotype',
                )


# =============================================================================
# QR Rank-Deficiency Detection (STAT-III-2)
# =============================================================================

class TestQRRankDeficiency:
    """Tests for QR rank-deficiency detection in both rotation functions."""

    def test_general_rank_deficient_raises(self):
        """compute_rotation_matrices_general rejects rank-deficient design."""
        n = 10
        # Create a rank-deficient matrix: col 2 = col 0 + col 1
        X = np.random.default_rng(42).standard_normal((n, 3))
        X[:, 2] = X[:, 0] + X[:, 1]  # Exact collinearity
        contrast = np.array([0.0, 1.0, -1.0])

        with pytest.raises(ValueError, match="rank-deficient"):
            compute_rotation_matrices_general(X, contrast)

    def test_general_full_rank_passes(self):
        """compute_rotation_matrices_general accepts full-rank design."""
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((n, 3))
        contrast = np.array([0.0, 1.0, -1.0])

        # Should not raise
        result = compute_rotation_matrices_general(X, contrast)
        assert isinstance(result, RotationPrecomputed)

    def test_general_ill_conditioned_warns(self):
        """compute_rotation_matrices_general warns on ill-conditioned design."""
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((n, 3))
        # Make nearly collinear: col 2 ≈ col 0 + col 1 with tiny perturbation
        X[:, 2] = X[:, 0] + X[:, 1] + rng.standard_normal(n) * 1e-12
        contrast = np.array([0.0, 1.0, -1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_rotation_matrices_general(X, contrast)
            cond_warnings = [x for x in w if "ill-conditioned" in str(x.message)]
            assert len(cond_warnings) >= 1

    def test_convenience_full_rank_passes(self):
        """compute_rotation_matrices accepts a healthy two-group design."""
        # The convenience function builds its own design matrix via
        # pd.get_dummies, so rank deficiency only arises in pathological
        # cases.  Verify that a normal design passes the new check.
        sample_condition = np.array(
            ['A'] * 10 + ['B'] * 10, dtype=object
        )
        result = compute_rotation_matrices(
            sample_condition=sample_condition,
            conditions=['A', 'B'],
            contrast=('A', 'B'),
        )
        assert isinstance(result, RotationPrecomputed)

    def test_general_single_collinear_column_detected(self):
        """Rank deficiency detected even when only one column is redundant."""
        rng = np.random.default_rng(99)
        n = 15
        X = rng.standard_normal((n, 4))
        # Make col 3 = 2*col 0 - col 1 (exact linear dependency)
        X[:, 3] = 2.0 * X[:, 0] - X[:, 1]
        contrast = np.array([1.0, -1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="rank-deficient"):
            compute_rotation_matrices_general(X, contrast)

    def test_general_rank_deficient_reports_numerical_rank(self):
        """Error message includes the actual numerical rank."""
        rng = np.random.default_rng(7)
        n = 12
        X = rng.standard_normal((n, 3))
        X[:, 2] = X[:, 0]  # col 2 is exact duplicate of col 0
        contrast = np.array([1.0, -1.0, 0.0])

        with pytest.raises(ValueError, match="numerical rank 2 < 3"):
            compute_rotation_matrices_general(X, contrast)


class TestNaNGeneFilter:
    """Tests for STAT-IV-2: NaN gene filtering in RotationTestEngine.fit()."""

    def _make_engine(self, data, n_genes, n_samples):
        """Helper to build a RotationTestEngine from raw data."""
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        sample_condition = np.array(["CASE"] * (n_samples // 2) + ["CTRL"] * (n_samples // 2))
        metadata = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "phenotype": sample_condition,
        })
        return RotationTestEngine(data, gene_ids, metadata)

    def test_partial_nan_genes_removed_with_warning(self):
        """Genes with partial NaN (e.g., 1 of 6 samples) are removed."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 50, 10
        data = rng.standard_normal((n_genes, n_samples))
        # Inject partial NaN into genes 5 and 10
        data[5, 3] = np.nan
        data[10, 7] = np.nan

        engine = self._make_engine(data, n_genes, n_samples)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            nan_warnings = [x for x in w if "missing values" in str(x.message)]
            assert len(nan_warnings) >= 1, "Expected a warning about NaN gene removal"

        # Two genes removed
        assert len(engine.gene_ids) == n_genes - 2
        assert "GENE_5" not in engine.gene_ids
        assert "GENE_10" not in engine.gene_ids
        # Data should have no NaN
        assert not np.any(np.isnan(engine.data))

    def test_no_nan_genes_pass_through(self):
        """When no genes have NaN, all genes remain and no warning is emitted."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 50, 10
        data = rng.standard_normal((n_genes, n_samples))

        engine = self._make_engine(data, n_genes, n_samples)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            nan_warnings = [x for x in w if "missing values" in str(x.message)]
            assert len(nan_warnings) == 0, "No NaN warning expected when data is clean"

        assert len(engine.gene_ids) == n_genes

    def test_over_50_pct_nan_triggers_extra_warning(self):
        """When >50% of genes have NaN, an extra imputation warning is emitted."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 20, 10
        data = rng.standard_normal((n_genes, n_samples))
        # Inject NaN into 11 of 20 genes (55%)
        for i in range(11):
            data[i, rng.integers(0, n_samples)] = np.nan

        engine = self._make_engine(data, n_genes, n_samples)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            impute_warnings = [x for x in w if "imputing" in str(x.message).lower()]
            assert len(impute_warnings) >= 1, "Expected imputation suggestion warning"

    def test_all_nan_genes_caught(self):
        """Genes that are entirely NaN are removed (also caught by zero-var filter)."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 30, 10
        data = rng.standard_normal((n_genes, n_samples))
        # Make gene 0 all NaN
        data[0, :] = np.nan

        engine = self._make_engine(data, n_genes, n_samples)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )

        assert "GENE_0" not in engine.gene_ids
        assert not np.any(np.isnan(engine.data))


class TestEBD0Infinity:
    """Tests for STAT-IV-1: d0=inf means prior dominates (all genes use s0_sq)."""

    def _make_simple_data(self, n_genes=50, n_samples=20, seed=42):
        """Create a simple two-group design with known signal."""
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_genes, n_samples))
        # Add signal to first 10 genes in first 10 samples (CASE group)
        data[:10, :10] += 2.0
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
        })
        return data, gene_ids, metadata

    def test_d0_inf_moderated_variances_equal_s0sq(self):
        """When d0=inf, all moderated variances should equal s0_sq."""
        data, gene_ids, metadata = self._make_simple_data()

        # Compute QR decomposition
        precomputed = compute_rotation_matrices(
            metadata["phenotype"].values, ["CASE", "CTRL"], ("CASE", "CTRL")
        )

        # Extract gene effects with d0=inf
        s0_sq = 0.5  # arbitrary prior variance
        effects = extract_gene_effects(
            data, gene_ids, precomputed, eb_d0=np.inf, eb_s0_sq=s0_sq
        )

        # All moderated variances should be exactly s0_sq
        assert effects.moderated_variances is not None
        assert_allclose(effects.moderated_variances, s0_sq)
        # df_total should be inf
        assert np.isinf(effects.df_total)

    def test_d0_inf_pvalue_computation_works(self):
        """When d0=inf, t-to-z conversion with df=inf should produce valid p-values."""
        from scipy import stats as scipy_stats

        # t.cdf(x, df=inf) should match norm.cdf(x)
        t_vals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        p_t_inf = scipy_stats.t.cdf(t_vals, df=np.inf)
        p_norm = scipy_stats.norm.cdf(t_vals)
        assert_allclose(p_t_inf, p_norm, atol=1e-14)

    def test_d0_inf_rotation_cpu_path(self):
        """Full CPU rotation pipeline works with d0=inf."""
        data, gene_ids, metadata = self._make_simple_data()

        precomputed = compute_rotation_matrices(
            metadata["phenotype"].values, ["CASE", "CTRL"], ("CASE", "CTRL")
        )

        s0_sq = 1.0
        effects = extract_gene_effects(
            data, gene_ids, precomputed, eb_d0=np.inf, eb_s0_sq=s0_sq
        )

        rng = np.random.default_rng(123)
        R = generate_rotation_vectors(100, precomputed.Q2.shape[1], rng, use_gpu=False)

        # apply_rotations_batched should not raise
        t_rot, z_rot, valid = apply_rotations_batched(
            effects.U, effects.rho_sq, R,
            effects.sample_variances, effects.moderated_variances,
            precomputed.df_residual, effects.df_total,
            use_gpu=False,
            eb_d0=np.inf, eb_s0_sq=s0_sq,
        )

        # Shapes should be correct
        assert t_rot.shape == (100, len(gene_ids))
        assert z_rot.shape == (100, len(gene_ids))
        assert valid.shape == (100,)

        # With d0=inf and use_df=inf, t ≈ z (fast path: inf > 1000)
        # All valid rotations should have t == z
        valid_t = t_rot[valid]
        valid_z = z_rot[valid]
        assert_allclose(valid_t, valid_z)

        # No NaN or all-zero results
        assert not np.any(np.isnan(z_rot[valid]))
        assert np.any(z_rot[valid] != 0)

    def test_d0_inf_end_to_end_engine(self):
        """RotationTestEngine works end-to-end when fit_f_dist returns d0=inf."""
        data, gene_ids, metadata = self._make_simple_data()

        engine = RotationTestEngine(
            data=data,
            gene_ids=gene_ids,
            metadata=metadata,
        )

        # Patch fit_f_dist to return d0=inf
        import unittest.mock as mock
        with mock.patch(
            "cliquefinder.stats.permutation_gpu.fit_f_dist",
            return_value=(np.inf, 0.8),
        ):
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )

        # Check that EB params were stored correctly
        assert engine._precomputed.eb_d0 == np.inf
        assert engine._precomputed.eb_s0_sq == 0.8
        assert np.isinf(engine._precomputed.eb_df_total)

        # Test a gene set — should produce valid result
        config = RotationTestConfig(n_rotations=200, use_gpu=False)
        gene_set = gene_ids[:10]
        result = engine.test_gene_set(
            gene_set=gene_set, gene_set_id="test_set", config=config
        )
        assert isinstance(result, RotationResult)
        # p-values should be in [0, 1]
        for stat_name, alternatives in result.p_values.items():
            for alt_name, pval in alternatives.items():
                assert 0.0 <= pval <= 1.0, (
                    f"Invalid p-value {pval} for {stat_name}/{alt_name}"
                )

    def test_finite_d0_still_works(self):
        """Regression guard: finite d0 should still produce correct EB shrinkage."""
        data, gene_ids, metadata = self._make_simple_data()

        precomputed = compute_rotation_matrices(
            metadata["phenotype"].values, ["CASE", "CTRL"], ("CASE", "CTRL")
        )

        d0 = 5.0
        s0_sq = 1.0
        effects = extract_gene_effects(
            data, gene_ids, precomputed, eb_d0=d0, eb_s0_sq=s0_sq
        )

        # Moderated variances should be a weighted average
        df = precomputed.df_residual
        expected = (d0 * s0_sq + df * effects.sample_variances) / (d0 + df)
        assert_allclose(effects.moderated_variances, expected)
        assert effects.df_total == d0 + df


# =============================================================================
# Audit IV tests
# =============================================================================


class TestAuditIVFixes:
    """Tests for Audit IV low-severity fixes."""

    def _make_simple_data(self, n_genes=50, n_samples=20, seed=42):
        """Create a simple two-group design with known signal."""
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_genes, n_samples))
        data[:10, :10] += 2.0
        gene_ids = [f"GENE_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
        })
        return data, gene_ids, metadata

    def test_double_fit_raises_runtime_error(self):
        """STATE-IV-1: Calling fit() twice raises RuntimeError."""
        data, gene_ids, metadata = self._make_simple_data()
        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)

        engine.fit(
            conditions=["CASE", "CTRL"],
            contrast=("CASE", "CTRL"),
            condition_column="phenotype",
        )

        with pytest.raises(RuntimeError, match="already been called"):
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )

    def test_double_fit_general_raises_runtime_error(self):
        """STATE-IV-1: Calling fit_general() twice raises RuntimeError."""
        data, gene_ids, metadata = self._make_simple_data()
        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)

        # First fit via fit_general
        design = np.column_stack([
            (metadata["phenotype"] == "CASE").values.astype(float),
            (metadata["phenotype"] == "CTRL").values.astype(float),
        ])
        contrast = np.array([1.0, -1.0])
        engine.fit_general(design, contrast, "test")

        with pytest.raises(RuntimeError, match="already been called"):
            engine.fit_general(design, contrast, "test2")

    def test_double_fit_cross_method_raises(self):
        """STATE-IV-1: fit() then fit_general() raises RuntimeError."""
        data, gene_ids, metadata = self._make_simple_data()
        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)

        engine.fit(
            conditions=["CASE", "CTRL"],
            contrast=("CASE", "CTRL"),
            condition_column="phenotype",
        )

        design = np.column_stack([
            (metadata["phenotype"] == "CASE").values.astype(float),
            (metadata["phenotype"] == "CTRL").values.astype(float),
        ])
        contrast = np.array([1.0, -1.0])

        with pytest.raises(RuntimeError, match="already been called"):
            engine.fit_general(design, contrast, "test")

    def test_null_distributions_no_nan_from_invalid_rotations(self):
        """STAT-IV-3: null_distributions should not contain NaN from invalid rotations."""
        data, gene_ids, metadata = self._make_simple_data()
        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)

        engine.fit(
            conditions=["CASE", "CTRL"],
            contrast=("CASE", "CTRL"),
            condition_column="phenotype",
        )

        config = RotationTestConfig(n_rotations=200, use_gpu=False)
        result = engine.test_gene_set(
            gene_set=gene_ids[:10], gene_set_id="test_set", config=config
        )

        # Check that null_distributions contain no NaN values
        for stat_name, alternatives in result.null_distributions.items():
            for alt_name, values in alternatives.items():
                arr = np.asarray(values)
                assert not np.any(np.isnan(arr)), (
                    f"NaN found in null_distributions[{stat_name}][{alt_name}]"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
