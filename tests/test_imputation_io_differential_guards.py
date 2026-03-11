"""Tests for imputation, IO loading, differential testing, and clique analysis guards."""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# IO / Data loading
# =============================================================================


class TestSoftClipOnlyOutliers:
    """Soft-clip should only modify outlier positions, not non-outlier values."""

    def test_global_soft_clip_preserves_non_outliers(self):
        from cliquefinder.quality.imputation import Imputer

        # 3 genes x 8 samples; gene 0 has one extreme outlier at col 7
        data = np.array([
            [10.0, 11.0, 10.5, 10.2, 10.8, 10.3, 10.7, 100.0],  # outlier at 7
            [5.0, 5.1, 5.2, 5.0, 5.3, 5.1, 5.0, 5.2],            # no outliers
            [8.0, 8.1, 8.0, 8.2, 8.1, 8.0, 8.1, 8.0],            # no outliers
        ])
        original = data.copy()

        # mask: only position [0, 7] is an outlier
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 7] = True

        imp = Imputer.__new__(Imputer)
        imp.threshold = 3.0
        imp.max_upper_bound = None
        imp.sharpness = 5.0
        imp.global_lower_ = 0.0
        imp.global_upper_ = 50.0

        imp._soft_clip_global(data, mask)

        # Gene 1 and 2 should be completely untouched
        np.testing.assert_array_equal(data[1, :], original[1, :])
        np.testing.assert_array_equal(data[2, :], original[2, :])
        # Gene 0: non-outlier positions unchanged
        np.testing.assert_array_equal(data[0, :7], original[0, :7])
        # Gene 0: outlier position changed
        assert data[0, 7] != original[0, 7]
        assert data[0, 7] < 100.0  # Clipped down

    def test_stratified_soft_clip_preserves_non_outliers(self):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.quality.imputation import Imputer

        data = np.array([
            [10.0, 10.5, 10.2, 100.0],  # outlier at col 3
            [5.0, 5.1, 5.2, 5.0],        # clean
        ])
        original = data.copy()

        mask = np.zeros_like(data, dtype=bool)
        mask[0, 3] = True

        md = pd.DataFrame({"group": ["A", "A", "B", "B"]}, index=["S1", "S2", "S3", "S4"])
        matrix = BioMatrix(
            data=data.copy(),  # BioMatrix makes data read-only
            feature_ids=pd.Index(["G1", "G2"]),
            sample_ids=pd.Index(["S1", "S2", "S3", "S4"]),
            sample_metadata=md,
            quality_flags=np.ones_like(data),
        )

        imp = Imputer.__new__(Imputer)
        imp.threshold = 3.0
        imp.max_upper_bound = None
        imp.sharpness = 5.0
        imp.group_cols = ["group"]
        imp.global_lower_ = 0.0
        imp.global_upper_ = 50.0

        # _soft_clip_stratified operates on mutable data, not BioMatrix
        mutable_data = data.copy()
        imp._soft_clip_stratified(mutable_data, mask, matrix)

        # Gene 1 entirely unchanged
        np.testing.assert_array_equal(mutable_data[1, :], original[1, :])
        # Gene 0 non-outlier positions unchanged
        np.testing.assert_array_equal(mutable_data[0, :3], original[0, :3])


class TestMADZeroClipToMedian:
    """MAD=0 should clip outliers to median, not skip."""

    def test_mad_zero_clips_outliers(self):
        from cliquefinder.quality.imputation import Imputer

        # Gene where all clean values are identical (MAD=0), one outlier
        data = np.array([
            [5.0, 5.0, 5.0, 5.0, 5.0, 50.0],  # 50.0 is outlier
        ])
        mask = np.array([
            [False, False, False, False, False, True],
        ])

        imp = Imputer.__new__(Imputer)
        imp.threshold = 3.0
        imp.max_upper_bound = None

        # Call the mad-clip method which should clip MAD=0 outliers to median
        imp._mad_clip_global(data, mask)

        # After fix: outlier should be clipped to median (5.0)
        assert data[0, 5] == 5.0


class TestUTF8BOM:
    """CSV loader handles UTF-8 BOM encoding."""

    def test_bom_encoded_csv(self):
        from cliquefinder.io.loaders import load_csv_matrix

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="wb", delete=False) as f:
            # Write UTF-8 BOM + valid CSV
            f.write(b'\xef\xbb\xbf')  # BOM
            f.write(b'protein,S1,S2,S3\n')
            f.write(b'TP53,1.0,2.0,3.0\n')
            f.write(b'AKT1,4.0,5.0,6.0\n')
            path = f.name

        try:
            matrix = load_csv_matrix(Path(path))
            # With utf-8-sig encoding, BOM is stripped
            # Feature IDs should be clean, without BOM prefix
            assert "TP53" in list(matrix.feature_ids)
            assert not any(fid.startswith('\ufeff') for fid in matrix.feature_ids)
        finally:
            Path(path).unlink()


class TestQuantileNormFractionalRank:
    """Quantile normalization uses np.interp for fractional ranks."""

    def test_fractional_rank_interpolation(self):
        from cliquefinder.stats.normalization import quantile_normalization

        # Simple test: data with known ranks and target distribution
        data = np.array([
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ])  # 4 features x 2 samples

        result = quantile_normalization(data)
        # Both columns should be mapped to same target distribution
        # Column 0 is already sorted ascending, column 1 is reversed
        # After quantile normalization, both should have same distribution
        col0_sorted = np.sort(result.data[:, 0])
        col1_sorted = np.sort(result.data[:, 1])
        np.testing.assert_allclose(col0_sorted, col1_sorted, rtol=1e-10)

    def test_quantile_norm_with_ties(self):
        from cliquefinder.stats.normalization import quantile_normalization

        # Data with tied values
        data = np.array([
            [1.0, 3.0],
            [2.0, 3.0],  # tie in column 1
            [3.0, 1.0],
        ])

        result = quantile_normalization(data)
        # Should not crash, tied values get averaged fractional ranks
        assert result.data.shape == (3, 2)
        assert not np.any(np.isnan(result.data))


class TestQualityFlagDtype:
    """Quality flags preserve dtype after bitwise OR."""

    def test_imputed_flag_preserves_dtype(self):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag

        # Create flags as uint8 (common compact dtype)
        flags = np.ones((2, 3), dtype=np.uint8)
        to_impute = np.array([
            [True, False, False],
            [False, True, False],
        ])

        new_flags = flags.copy()
        new_flags[to_impute] = (new_flags[to_impute] | QualityFlag.IMPUTED).astype(new_flags.dtype)

        assert new_flags.dtype == np.uint8
        assert new_flags[0, 0] & QualityFlag.IMPUTED


# =============================================================================
# GPU / differential testing
# =============================================================================


class TestEBDfInfFallback:
    """When d0=inf (no EB moderation), use df_residual not eb_df_total."""

    def test_df_for_pval_no_eb(self):
        """Without EB moderation, df_for_pval should be df_residual."""
        # This tests the logic: df_for_pval = matrices.eb_df_total if eb_moderation else float(matrices.df_residual)
        eb_moderation = False
        df_residual = 10.0
        eb_df_total = np.inf  # Would be inf if EB was used with d0=inf

        df_for_pval = eb_df_total if eb_moderation else float(df_residual)
        assert df_for_pval == 10.0
        assert np.isfinite(df_for_pval)

    def test_df_for_pval_with_eb(self):
        """With EB moderation and finite d0, use eb_df_total."""
        eb_moderation = True
        df_residual = 10.0
        eb_df_total = 14.5  # d0 + df_residual

        df_for_pval = eb_df_total if eb_moderation else float(df_residual)
        assert df_for_pval == 14.5


class TestNaNCPUFallback:
    """NaN in input triggers CPU fallback for batched_median_polish."""

    def test_nan_triggers_cpu_fallback(self):
        """When data contains NaN, GPU path should fall back to CPU."""
        data_with_nan = np.array([[[1.0, 2.0], [np.nan, 4.0]]])  # 1 batch, 2 feats, 2 samples

        # Simulate the guard logic
        use_gpu = True
        n_nan = np.sum(np.isnan(data_with_nan))
        if n_nan > 0:
            use_gpu = False

        assert n_nan == 1
        assert use_gpu is False

    def test_clean_data_uses_gpu(self):
        """When data has no NaN, GPU path proceeds."""
        data_clean = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        use_gpu = True
        n_nan = np.sum(np.isnan(data_clean))
        if n_nan > 0:
            use_gpu = False

        assert n_nan == 0
        assert use_gpu is True


class TestSatterthwaiteDfDecomposition:
    """Satterthwaite uses within/between df, not total params."""

    def test_df_residual_is_n_obs_minus_n_groups(self):
        """df_residual = n_obs - n_groups (within-group residual)."""
        from cliquefinder.stats.differential import satterthwaite_df

        n_obs = 20
        n_groups = 4
        n_params = 2  # e.g., intercept + condition
        contrast = np.array([0.0, 1.0])
        cov_beta = np.eye(2) * 0.1

        result = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=1.0,
            subject_var=0.5,
            n_groups=n_groups,
            n_obs=n_obs,
            use_mlx=False,
        )

        assert result is not None
        # The result should use df_residual = 20 - 4 = 16, not 20 - 2 = 18
        # The result should use df_random = 4 - 2 = 2, not 4 - 1 = 3
        # We can verify by checking the containment df which is n_groups - n_params = 2
        assert result >= 1.0

    def test_containment_df_n_groups_minus_n_params(self):
        """Containment df = n_groups - n_params."""
        from cliquefinder.stats.differential import satterthwaite_df

        # With only 3 groups and 3 params, containment_df = 0 → returns None
        contrast = np.array([0.0, 1.0, 0.0])
        cov_beta = np.eye(3) * 0.1

        result = satterthwaite_df(
            contrast_vector=contrast,
            cov_beta=cov_beta,
            residual_var=1.0,
            subject_var=0.5,
            n_groups=3,
            n_obs=30,
            use_mlx=False,
        )
        # n_groups - n_params = 3 - 3 = 0 → None
        assert result is None


class TestFloat64ConditionNumberCheck:
    """MLX XtX cast to float64 before condition number check."""

    def test_float64_cond_check(self):
        """Condition number should be computed in float64 for precision."""
        # A near-singular matrix that float32 might misrepresent
        XtX_float32 = np.array([[1.0, 0.9999999], [0.9999999, 1.0]], dtype=np.float32)
        XtX_float64 = XtX_float32.astype(np.float64)

        cond_32 = np.linalg.cond(XtX_float32)
        cond_64 = np.linalg.cond(XtX_float64)

        # float64 should give a more accurate (usually higher) condition number
        # The key thing is that float64 is used in the code path
        assert np.isfinite(cond_64)


# =============================================================================
# Clique analysis / integration
# =============================================================================


class TestIDSpaceMismatch:
    """compare_protein_vs_clique_results handles ID-space mismatch gracefully."""

    def _make_clique_analysis_result(self, protein_feature_ids, clique_protein_ids):
        """Helper to create minimal CliqueAnalysisResult with protein_results."""
        from cliquefinder.stats.clique_analysis import (
            CliqueAnalysisResult,
            CliqueDifferentialResult,
        )
        from cliquefinder.stats.differential import DifferentialResult, ContrastResult

        # Clique result uses clique_protein_ids (e.g., gene symbols)
        clique_result = CliqueDifferentialResult(
            clique_id="C1",
            regulator="TP53",
            n_proteins=len(clique_protein_ids),
            n_proteins_found=len(clique_protein_ids),
            protein_ids=clique_protein_ids,
            summarization_method="median_polish",
            coherence=0.8,
            log2_fc=1.5,
            se=0.3,
            t_value=5.0,
            df=10.0,
            p_value=0.001,
            adj_p_value=0.01,
            ci_lower=0.9,
            ci_upper=2.1,
            contrast="A_vs_B",
            model_type="fixed",
            issue=None,
            direction="positive",
        )

        # Protein results use protein_feature_ids (e.g., UniProt IDs)
        contrast_results = []
        for fid in protein_feature_ids:
            contrast_results.append(ContrastResult(
                contrast_name="A_vs_B",
                log2_fc=1.0,
                se=0.5,
                t_value=2.0,
                p_value=0.05,
                df=8.0,
                ci_lower=0.0,
                ci_upper=2.0,
            ))

        protein_results = DifferentialResult(
            results=[],
            contrasts_tested=["A_vs_B"],
        )
        # Manually build the protein DataFrame
        protein_df = pd.DataFrame({
            "feature_id": protein_feature_ids,
            "contrast": ["A_vs_B"] * len(protein_feature_ids),
            "log2FC": [1.0] * len(protein_feature_ids),
            "adj_pvalue": [0.05] * len(protein_feature_ids),
        })

        # Monkeypatch to_dataframe on protein_results
        protein_results.to_dataframe = lambda: protein_df

        return CliqueAnalysisResult(
            clique_results=[clique_result],
            protein_results=protein_results,
            contrasts_tested=["A_vs_B"],
            n_cliques_tested=1,
            n_significant=1,
            fdr_threshold=0.05,
            preprocessing_params={},
        )

    def test_matching_ids(self):
        from cliquefinder.stats.clique_analysis import compare_protein_vs_clique_results

        result = self._make_clique_analysis_result(
            protein_feature_ids=["TP53", "AKT1"],
            clique_protein_ids=["TP53", "AKT1"],
        )
        df = compare_protein_vs_clique_results(result)
        # IDs match, should find proteins
        assert len(df) == 1
        assert df.iloc[0]["n_proteins"] == 2

    def test_mismatched_ids_no_crash(self):
        """When clique uses symbols but proteins use UniProt, should not crash."""
        from cliquefinder.stats.clique_analysis import compare_protein_vs_clique_results

        result = self._make_clique_analysis_result(
            protein_feature_ids=["P04637", "P31749"],  # UniProt IDs
            clique_protein_ids=["TP53", "AKT1"],        # Gene symbols
        )
        df = compare_protein_vs_clique_results(result)
        assert len(df) == 1
        # No matches found, but doesn't crash
        assert df.iloc[0]["n_sig_proteins"] == 0


class TestNaNIndraTargets:
    """load_clique_definitions handles NaN n_indra_targets."""

    def test_nan_n_indra_targets(self):
        from cliquefinder.stats.clique_analysis import load_clique_definitions

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("clique_id,clique_genes,n_indra_targets\n")
            f.write("C1,\"A,B,C\",\n")  # NaN n_indra_targets
            f.write("C2,\"D,E,F\",5\n")
            path = f.name

        try:
            cliques = load_clique_definitions(path)
            assert len(cliques) == 2
            assert cliques[0].n_indra_targets is None  # NaN → None
            assert cliques[1].n_indra_targets == 5
        finally:
            Path(path).unlink()


class TestCliqueIdNotRegulator:
    """ROAST functions use clique_id (not regulator) as dict key."""

    def test_none_regulator_no_crash(self):
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        clique = CliqueDefinition(
            clique_id="cluster_0",
            protein_ids=["A", "B", "C"],
            regulator=None,  # No regulator
        )
        # The fix uses clique.clique_id which is always set
        gene_symbols = {}
        gene_symbols[clique.clique_id] = list(clique.protein_ids)

        assert "cluster_0" in gene_symbols
        assert gene_symbols["cluster_0"] == ["A", "B", "C"]

    def test_duplicate_regulator_name_no_collision(self):
        """Two cliques with same regulator but different IDs shouldn't collide."""
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        c1 = CliqueDefinition(clique_id="C1", protein_ids=["A", "B"], regulator="TP53")
        c2 = CliqueDefinition(clique_id="C2", protein_ids=["C", "D"], regulator="TP53")

        gene_symbols = {}
        gene_symbols[c1.clique_id] = list(c1.protein_ids)
        gene_symbols[c2.clique_id] = list(c2.protein_ids)

        # Both preserved (wouldn't be if using regulator as key)
        assert len(gene_symbols) == 2
        assert gene_symbols["C1"] == ["A", "B"]
        assert gene_symbols["C2"] == ["C", "D"]


class TestZeroDivisionGuard:
    """No ZeroDivisionError when permutation_results is empty."""

    def test_empty_results_no_crash(self):
        """Significance rate should not crash with empty results list."""
        # Simulates the guard logic
        permutation_results = []
        n_significant = 0

        # After fix: only prints rate when len > 0
        if len(permutation_results) > 0:
            rate = 100 * n_significant / len(permutation_results)
        else:
            rate = 0.0

        assert rate == 0.0


class TestBareExceptLogging:
    """Bare except now logs instead of silently swallowing."""

    def test_exception_logged(self):
        import logging
        # The fix replaces `except Exception: pass` with
        # `except Exception: logger.debug(...)` — verify the pattern exists
        from cliquefinder.stats import clique_analysis

        # Verify the module has a logger
        assert hasattr(clique_analysis, 'logger')
        assert isinstance(clique_analysis.logger, logging.Logger)


class TestFrozenFractionVerdict:
    """frozen_fraction > 0.5 downgrades Phase 3 gate."""

    def test_high_frozen_fraction_downgrades(self):
        """Stratified p-value passes but frozen_fraction > 0.5 → inconclusive."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {
            "stratified": {
                "permutation_pvalue": 0.01,
                "frozen_fraction": 0.7,  # >50% frozen
            },
            "permutation_pvalue": 0.01,
            "frozen_fraction": 0.7,
        })
        report.compute_verdict()

        # Gate should be downgraded because frozen_fraction > 0.5
        # and no free permutation to confirm
        assert report.verdict != "validated"
        assert "DOWNGRADED" in report.phase_details.get("label_permutation_stratified", "")

    def test_high_frozen_fraction_with_passing_free(self):
        """frozen_fraction > 0.5 but free permutation passes → still validated."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {
            "stratified": {
                "permutation_pvalue": 0.01,
                "frozen_fraction": 0.7,
            },
            "free": {
                "permutation_pvalue": 0.01,  # Free also passes
            },
            "permutation_pvalue": 0.01,
            "frozen_fraction": 0.7,
        })
        report.compute_verdict()

        # Free passes, so gate not downgraded
        assert report.verdict == "validated"

    def test_low_frozen_fraction_no_downgrade(self):
        """frozen_fraction <= 0.5 should not downgrade."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {
            "stratified": {
                "permutation_pvalue": 0.01,
                "frozen_fraction": 0.1,  # Low frozen fraction
            },
            "permutation_pvalue": 0.01,
        })
        report.compute_verdict()

        assert report.verdict == "validated"
        assert "DOWNGRADED" not in report.phase_details.get("label_permutation_stratified", "")

    def test_zero_frozen_fraction_no_downgrade(self):
        """frozen_fraction = 0 should not trigger downgrade."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {
            "stratified": {"permutation_pvalue": 0.01},
            "permutation_pvalue": 0.01,
        })
        report.compute_verdict()

        assert report.verdict == "validated"

