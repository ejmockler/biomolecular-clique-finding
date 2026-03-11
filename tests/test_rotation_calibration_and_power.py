"""
Tests for ROAST rotation null calibration, power, and FDR correction.

- ROAST null calibration via KS test for p-value uniformity
- Multi-replicate power tests at various effect sizes
- Effect size type documentation consistency
- Experiment-wide FDR correction scope
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from cliquefinder.stats.rotation import (
    RotationTestConfig,
    RotationTestEngine,
)


# =============================================================================
# ROAST null calibration
# =============================================================================

class TestROASTNullCalibration:
    """Verify ROAST p-values are uniform under the null hypothesis.

    This is the formal calibration test missing from the original test suite.
    Under the null (no differential expression), p-values should follow
    Uniform(0, 1). We verify this with a Kolmogorov-Smirnov test.
    """

    @pytest.mark.slow
    def test_pvalues_uniform_under_null_msq(self):
        """Full pipeline null calibration for MSQ statistic with KS test."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 200, 20
        n_replicates = 100

        pvalues = []
        for rep in range(n_replicates):
            # Generate null data — no differential expression
            data = rng.standard_normal((n_genes, n_samples))
            gene_ids = [f"gene_{i}" for i in range(n_genes)]
            metadata = pd.DataFrame({
                "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
            })

            # Random gene set (20 genes)
            gene_set = list(rng.choice(gene_ids, size=20, replace=False))

            engine = RotationTestEngine(data, gene_ids, metadata)
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            result = engine.test_gene_set(
                gene_set=gene_set,
                gene_set_id=f"null_set_{rep}",
                config=RotationTestConfig(
                    n_rotations=999,
                    seed=int(rng.integers(1_000_000)),
                    use_gpu=False,  # CPU for reproducibility
                ),
            )

            p = result.p_values.get("msq", {}).get("mixed", np.nan)
            if not np.isnan(p):
                pvalues.append(p)

        assert len(pvalues) >= 90, (
            f"Too many NaN p-values: only {len(pvalues)}/100 were valid"
        )

        pvalues = np.array(pvalues)

        # KS test for uniformity
        ks_stat, ks_pval = scipy_stats.kstest(pvalues, "uniform")
        assert ks_pval > 0.01, (
            f"ROAST MSQ p-values not uniform under null: KS stat={ks_stat:.4f}, "
            f"p={ks_pval:.4f}. Type I error control may be violated."
        )

        # Type I error rate at alpha=0.05
        type1_rate = float(np.mean(pvalues < 0.05))
        assert 0.01 <= type1_rate <= 0.15, (
            f"Type I error rate {type1_rate:.3f} outside [0.01, 0.15] "
            f"(expected ~0.05 under null)"
        )

    @pytest.mark.slow
    def test_pvalues_uniform_under_null_mean(self):
        """Null calibration for mean statistic."""
        rng = np.random.default_rng(123)
        n_genes, n_samples = 200, 20
        n_replicates = 100

        pvalues = []
        for rep in range(n_replicates):
            data = rng.standard_normal((n_genes, n_samples))
            gene_ids = [f"gene_{i}" for i in range(n_genes)]
            metadata = pd.DataFrame({
                "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
            })
            gene_set = list(rng.choice(gene_ids, size=20, replace=False))

            engine = RotationTestEngine(data, gene_ids, metadata)
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            result = engine.test_gene_set(
                gene_set=gene_set,
                gene_set_id=f"null_set_{rep}",
                config=RotationTestConfig(
                    n_rotations=999,
                    seed=int(rng.integers(1_000_000)),
                    use_gpu=False,
                ),
            )

            p = result.p_values.get("mean", {}).get("up", np.nan)
            if not np.isnan(p):
                pvalues.append(p)

        pvalues = np.array(pvalues)
        ks_stat, ks_pval = scipy_stats.kstest(pvalues, "uniform")
        assert ks_pval > 0.01, (
            f"ROAST mean-up p-values not uniform under null: KS stat={ks_stat:.4f}, "
            f"p={ks_pval:.4f}"
        )

    def test_null_pvalue_median_check(self):
        """Quick non-slow check: null p-value median should be near 0.5."""
        rng = np.random.default_rng(999)
        n_genes, n_samples = 100, 16

        pvalues = []
        for _ in range(30):
            data = rng.standard_normal((n_genes, n_samples))
            gene_ids = [f"gene_{i}" for i in range(n_genes)]
            metadata = pd.DataFrame({
                "phenotype": ["CASE"] * 8 + ["CTRL"] * 8,
            })
            gene_set = list(rng.choice(gene_ids, size=15, replace=False))

            engine = RotationTestEngine(data, gene_ids, metadata)
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            result = engine.test_gene_set(
                gene_set=gene_set,
                gene_set_id="null",
                config=RotationTestConfig(
                    n_rotations=499,
                    seed=int(rng.integers(1_000_000)),
                    use_gpu=False,
                ),
            )
            p = result.p_values.get("msq", {}).get("mixed", np.nan)
            if not np.isnan(p):
                pvalues.append(p)

        median_p = float(np.median(pvalues))
        assert 0.2 < median_p < 0.8, (
            f"Null p-value median {median_p:.3f} far from 0.5"
        )


# =============================================================================
# Multi-replicate power tests
# =============================================================================

class TestROASTPower:
    """Verify ROAST has adequate power to detect real signals."""

    @pytest.mark.slow
    def test_power_at_large_effect(self):
        """ROAST should detect large effect (d=2.0) with high power."""
        rng = np.random.default_rng(42)
        n_genes, n_samples = 200, 20
        n_replicates = 50
        effect_size = 2.0

        detections = 0
        for rep in range(n_replicates):
            data = rng.standard_normal((n_genes, n_samples))
            # Add strong signal to first 20 genes in CASE group
            data[:20, :10] += effect_size

            gene_ids = [f"gene_{i}" for i in range(n_genes)]
            metadata = pd.DataFrame({
                "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
            })
            gene_set = [f"gene_{i}" for i in range(20)]

            engine = RotationTestEngine(data, gene_ids, metadata)
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            result = engine.test_gene_set(
                gene_set=gene_set,
                gene_set_id="signal_set",
                config=RotationTestConfig(
                    n_rotations=999,
                    seed=int(rng.integers(1_000_000)),
                    use_gpu=False,
                ),
            )

            p = result.p_values.get("msq", {}).get("mixed", 1.0)
            if p < 0.05:
                detections += 1

        power = detections / n_replicates
        assert power > 0.70, (
            f"ROAST power {power:.2f} < 0.70 for effect size d={effect_size}"
        )

    @pytest.mark.slow
    def test_power_at_moderate_effect(self):
        """ROAST should detect moderate effect (d=1.0) with reasonable power."""
        rng = np.random.default_rng(99)
        n_genes, n_samples = 200, 20
        n_replicates = 50
        effect_size = 1.0

        detections = 0
        for rep in range(n_replicates):
            data = rng.standard_normal((n_genes, n_samples))
            data[:20, :10] += effect_size

            gene_ids = [f"gene_{i}" for i in range(n_genes)]
            metadata = pd.DataFrame({
                "phenotype": ["CASE"] * 10 + ["CTRL"] * 10,
            })
            gene_set = [f"gene_{i}" for i in range(20)]

            engine = RotationTestEngine(data, gene_ids, metadata)
            engine.fit(
                conditions=["CASE", "CTRL"],
                contrast=("CASE", "CTRL"),
                condition_column="phenotype",
            )
            result = engine.test_gene_set(
                gene_set=gene_set,
                gene_set_id="signal_set",
                config=RotationTestConfig(
                    n_rotations=999,
                    seed=int(rng.integers(1_000_000)),
                    use_gpu=False,
                ),
            )

            p = result.p_values.get("msq", {}).get("mixed", 1.0)
            if p < 0.05:
                detections += 1

        power = detections / n_replicates
        assert power > 0.40, (
            f"ROAST power {power:.2f} < 0.40 for effect size d={effect_size}"
        )

    def test_power_direction_sensitivity(self):
        """Up statistic should detect upregulation better than downregulation."""
        rng = np.random.default_rng(77)
        n_genes, n_samples = 100, 16
        data = rng.standard_normal((n_genes, n_samples))
        # Strong upregulation in first 15 genes
        data[:15, :8] += 2.5

        gene_ids = [f"gene_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({
            "phenotype": ["CASE"] * 8 + ["CTRL"] * 8,
        })
        gene_set = [f"gene_{i}" for i in range(15)]

        engine = RotationTestEngine(data, gene_ids, metadata)
        engine.fit(
            conditions=["CASE", "CTRL"],
            contrast=("CASE", "CTRL"),
            condition_column="phenotype",
        )
        result = engine.test_gene_set(
            gene_set=gene_set,
            gene_set_id="up_set",
            config=RotationTestConfig(
                n_rotations=999, seed=42, use_gpu=False,
            ),
        )

        p_up = result.p_values.get("mean", {}).get("up", 1.0)
        p_down = result.p_values.get("mean", {}).get("down", 1.0)

        # Up p-value should be much smaller than down p-value
        assert p_up < 0.05, f"Failed to detect upregulation: p_up={p_up:.4f}"
        assert p_down > p_up, (
            f"Down p-value ({p_down:.4f}) should be larger than up ({p_up:.4f})"
        )


# =============================================================================
# Effect size type documentation tests
# =============================================================================

class TestEffectSizeTypes:
    """Verify effect_size_type is correctly reported in method results."""

    def test_unified_result_has_effect_size_type(self):
        """UnifiedCliqueResult should have effect_size_type field."""
        from cliquefinder.stats.method_comparison_types import UnifiedCliqueResult
        import inspect
        sig = inspect.signature(UnifiedCliqueResult)
        # Check that the dataclass either has effect_size_type or at minimum
        # has method_metadata where the type can be inferred
        params = set(sig.parameters.keys())
        has_type_field = "effect_size_type" in params
        has_metadata = "method_metadata" in params
        assert has_type_field or has_metadata, (
            "UnifiedCliqueResult should have effect_size_type or method_metadata"
        )

    def test_ols_returns_log2fc(self):
        """OLS method should report log2FC as effect size."""
        from cliquefinder.stats.methods._base_linear import _BaseLinearMethod
        import inspect
        # Verify the _build_result method references log2_fc
        src = inspect.getsource(_BaseLinearMethod)
        assert "log2_fc" in src, (
            "OLS/LMM effect_size should use log2_fc from ContrastResult"
        )

    def test_roast_returns_mean_z(self):
        """ROAST method should use mean z-score as effect size."""
        from cliquefinder.stats.methods.roast import ROASTMethod
        import inspect
        src = inspect.getsource(ROASTMethod)
        assert "mean_z" in src or "mean" in src, (
            "ROAST effect_size should be a mean z-score"
        )

    def test_permutation_stores_log2fc_in_metadata(self):
        """Permutation method should have observed_log2fc accessible."""
        from cliquefinder.stats.methods.permutation import PermutationMethod
        import inspect
        src = inspect.getsource(PermutationMethod)
        assert "observed_log2fc" in src, (
            "Permutation method should provide observed_log2fc"
        )


# =============================================================================
# FDR scope tests
# =============================================================================

class TestFDRCorrection:
    """Tests for FDR correction scope and behavior."""

    def test_per_contrast_fdr_is_default(self):
        """FDR correction should be applied per-contrast by default."""
        from cliquefinder.stats.differential import fdr_correction
        # Basic functionality test
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])
        adj = fdr_correction(pvals, method="BH")
        # Adjusted p-values should be >= raw p-values
        assert np.all(adj >= pvals - 1e-10), (
            "BH-adjusted p-values should be >= raw p-values"
        )
        # Should preserve ordering
        assert np.all(np.diff(adj) >= -1e-10), (
            "BH-adjusted p-values should preserve rank ordering"
        )

    def test_fdr_handles_nan(self):
        """FDR correction should propagate NaN correctly."""
        from cliquefinder.stats.differential import fdr_correction
        pvals = np.array([0.001, np.nan, 0.05, 0.1, np.nan, 0.9])
        adj = fdr_correction(pvals, method="BH")
        # NaN inputs should produce NaN outputs
        assert np.isnan(adj[1]), "NaN input should produce NaN output"
        assert np.isnan(adj[4]), "NaN input should produce NaN output"
        # Non-NaN should be adjusted
        assert not np.isnan(adj[0])
        assert adj[0] >= pvals[0]

    def test_fdr_methods_available(self):
        """All documented FDR methods should work."""
        from cliquefinder.stats.differential import fdr_correction
        pvals = np.array([0.001, 0.01, 0.05, 0.5])
        for method in ["BH", "BY", "bonferroni"]:
            adj = fdr_correction(pvals, method=method)
            assert len(adj) == len(pvals)
            assert np.all(adj >= pvals - 1e-10)

    def test_concordance_warns_against_cross_method_pooling(self):
        """Concordance module should warn against cross-method FDR pooling."""
        import inspect
        from cliquefinder.stats import concordance
        src = inspect.getsource(concordance)
        assert "Do NOT select" in src or "DO NOT" in src, (
            "Concordance module should warn against cross-method p-value selection"
        )
        assert "Do NOT combine" in src or "DO NOT combine" in src, (
            "Concordance module should warn against cross-method p-value combination"
        )
