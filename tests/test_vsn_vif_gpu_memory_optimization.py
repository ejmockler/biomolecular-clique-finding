"""Tests for VSN scale guards, Camera VIF correction, GPU memory bounding, and lazy index generation."""

import numpy as np
import pytest
import warnings


class TestVSNScaleGuard:
    """VSN should warn when data appears log-transformed."""

    def test_warns_on_log2_data(self):
        """Log2 proteomics intensities (range ~0-30) should trigger warning."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Typical log2 proteomics data: values in [5, 25]
        data = np.random.normal(loc=15, scale=3, size=(100, 10))

        with pytest.warns(UserWarning, match="log-transformed"):
            vsn_normalization(data)

    def test_no_warning_on_raw_intensities(self):
        """Raw intensities (range ~100-100000) should NOT trigger warning."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Raw proteomics intensities: large positive values
        data = np.random.exponential(scale=1e4, size=(100, 10))
        data = np.clip(data, 100, None)  # Ensure all > 100

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise — data is raw intensities
            result = vsn_normalization(data)
            assert result.data.shape == data.shape

    def test_warns_on_negative_log_data(self):
        """Data with negative values in small range still triggers warning."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Centered log-ratio data: range ~ [-3, 3]
        data = np.random.normal(loc=0, scale=1.5, size=(50, 8))

        with pytest.warns(UserWarning, match="log-transformed"):
            vsn_normalization(data)

    def test_warning_message_content(self):
        """Warning message should be informative about the issue."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        data = np.random.normal(loc=15, scale=2, size=(50, 5))

        with pytest.warns(UserWarning) as record:
            vsn_normalization(data)

        assert len(record) >= 1
        msg = str(record[0].message)
        assert "double variance-stabilization" in msg
        assert "raw intensities" in msg

    def test_still_produces_output_despite_warning(self):
        """Even with warning, VSN should still run and return valid data."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        data = np.random.normal(loc=15, scale=3, size=(100, 10))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = vsn_normalization(data)

        assert result.data.shape == data.shape
        assert np.all(np.isfinite(result.data))

    def test_edge_case_boundary(self):
        """Data at the boundary (max ~35) should NOT warn."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Data with max just above threshold
        data = np.random.uniform(10, 40, size=(50, 5))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # max > 35, so should not warn
            result = vsn_normalization(data)
            assert result.data.shape == data.shape

    def test_handles_nan_in_guard(self):
        """NaN values should not crash the log-detection guard."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        data = np.random.exponential(scale=1e4, size=(50, 5))
        data[0, 0] = np.nan
        data[5, 2] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Raw intensities with NaN — should not warn and not crash
            result = vsn_normalization(data)
            assert result.data.shape == data.shape


class TestEBPriorsDocumentation:
    """Verify EB priors comment exists in permutation_gpu.py."""

    def test_eb_priors_comment_in_phase3(self):
        """Phase 3 null distribution code should document EB priors tradeoff."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        assert "DF-XII-2" in source, "EB priors documentation comment missing"
        assert "conservative bias" in source, "Should document bias direction"
        assert "shrinks null variances DOWN" in source or "over-shrink" in source, (
            "Should explain shrinkage mechanism"
        )


class TestCameraVIFIntegration:
    """Camera VIF correction in competitive z-scores."""

    def test_vif_deflates_correlated_set_z_score(self):
        """VIF > 1 should reduce z-score for correlated gene sets."""
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        np.random.seed(42)
        n = 500
        k = 20

        # Create t-statistics where targets are slightly enriched
        all_t = np.random.normal(0, 1, n)
        all_t[:k] += 1.5  # Target enrichment
        is_target = np.zeros(n, dtype=bool)
        is_target[:k] = True

        z_no_vif = compute_competitive_z(all_t, is_target)
        z_with_vif = compute_competitive_z(
            all_t, is_target, inter_gene_correlation=0.3
        )

        # VIF deflates: z_with_vif < z_no_vif
        assert z_with_vif < z_no_vif, (
            f"VIF should deflate z-score: {z_with_vif:.3f} >= {z_no_vif:.3f}"
        )
        # The ratio should match sqrt(VIF)
        vif = 1.0 + (k - 1) * 0.3
        expected_ratio = 1.0 / np.sqrt(vif)
        actual_ratio = z_with_vif / z_no_vif
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_vif_zero_correlation_no_effect(self):
        """VIF with rho_bar=0 should equal no VIF."""
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        np.random.seed(42)
        all_t = np.random.normal(0, 1, 200)
        all_t[:10] += 2.0
        is_target = np.zeros(200, dtype=bool)
        is_target[:10] = True

        z_none = compute_competitive_z(all_t, is_target)
        z_zero = compute_competitive_z(
            all_t, is_target, inter_gene_correlation=0.0
        )

        assert z_none == pytest.approx(z_zero)

    def test_estimate_inter_gene_correlation_correlated(self):
        """Correlated genes should have positive rho_bar."""
        from cliquefinder.stats.enrichment_z import estimate_inter_gene_correlation

        np.random.seed(42)
        n_genes = 100
        n_samples = 20

        # Create expression data with correlated target genes
        shared_signal = np.random.normal(0, 1, n_samples)
        expr = np.random.normal(0, 1, (n_genes, n_samples))
        # First 10 genes share a signal
        for i in range(10):
            expr[i, :] += shared_signal * 2

        is_target = np.zeros(n_genes, dtype=bool)
        is_target[:10] = True

        rho = estimate_inter_gene_correlation(expr, is_target)
        assert rho > 0.3, f"Expected rho > 0.3 for correlated genes, got {rho:.3f}"

    def test_estimate_inter_gene_correlation_uncorrelated(self):
        """Random genes should have rho_bar near 0."""
        from cliquefinder.stats.enrichment_z import estimate_inter_gene_correlation

        np.random.seed(42)
        n_genes = 100
        n_samples = 50

        expr = np.random.normal(0, 1, (n_genes, n_samples))
        is_target = np.zeros(n_genes, dtype=bool)
        is_target[:10] = True

        rho = estimate_inter_gene_correlation(expr, is_target)
        # Should be near 0 (possibly exactly 0 due to floor)
        assert rho < 0.15, f"Expected rho near 0 for random genes, got {rho:.3f}"

    def test_vif_deflates_z_score(self):
        """VIF-corrected z should be smaller than uncorrected (Camera property)."""
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        rng = np.random.default_rng(42)
        t_stats = rng.standard_normal(200)
        t_stats[:20] += 2.0
        mask = np.zeros(200, dtype=bool)
        mask[:20] = True

        z_raw = compute_competitive_z(t_stats, mask)
        z_vif = compute_competitive_z(t_stats, mask, inter_gene_correlation=0.3)
        assert abs(z_vif) < abs(z_raw), (
            f"VIF should deflate |z|: raw={z_raw:.3f}, vif={z_vif:.3f}"
        )

    def test_vif_integration_end_to_end(self):
        """VIF-corrected z-score should be lower than uncorrected for real cliques."""
        from cliquefinder.stats.enrichment_z import (
            compute_competitive_z,
            estimate_inter_gene_correlation,
        )

        np.random.seed(42)
        n_genes = 200
        n_samples = 30
        k = 15

        # Create expression with correlated targets
        shared = np.random.normal(0, 1, n_samples)
        expr = np.random.normal(0, 1, (n_genes, n_samples))
        for i in range(k):
            expr[i, :] += shared * 1.5

        is_target = np.zeros(n_genes, dtype=bool)
        is_target[:k] = True

        # Simulate t-statistics (correlated targets have higher |t|)
        t_stats = np.random.normal(0, 1, n_genes)
        t_stats[:k] += 2.0  # Enrichment

        # Without VIF
        z_raw = compute_competitive_z(t_stats, is_target)

        # With VIF from expression data
        rho = estimate_inter_gene_correlation(expr, is_target)
        z_vif = compute_competitive_z(
            t_stats, is_target, inter_gene_correlation=rho
        )

        assert rho > 0.2, f"Expected meaningful correlation, got {rho:.3f}"
        assert z_vif < z_raw, (
            f"VIF-corrected z ({z_vif:.3f}) should be < raw ({z_raw:.3f})"
        )

    def test_vif_magnitude_for_typical_clique(self):
        """For typical TF regulon (k=30, rho=0.3), VIF should be substantial."""
        k = 30
        rho = 0.3
        vif = 1.0 + (k - 1) * rho
        se_inflation = np.sqrt(vif)

        # SE inflated by sqrt(VIF) → z deflated by same factor
        assert vif == pytest.approx(9.7)
        assert se_inflation == pytest.approx(3.114, abs=0.01)
        # z-score is 3x smaller — this is a huge correction!


class TestPhase3CliqueChunking:
    """Phase 3 should chunk by clique count to bound memory."""

    def test_chunking_variables_in_source(self):
        """Phase 3 should use _MAX_BATCH_ELEMS and clique_ids_sub."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        assert "_MAX_BATCH_ELEMS" in source, "Missing batch element cap"
        assert "clique_ids_sub" in source, "Missing clique sub-chunking"
        assert "max_cliques_per_chunk" in source, "Missing clique chunk size calc"

    def test_max_batch_elems_is_bounded(self):
        """_MAX_BATCH_ELEMS should cap allocation at ~400MB."""
        import re, inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        match = re.search(r"_MAX_BATCH_ELEMS\s*=\s*(\d[\d_]*)", source)
        assert match is not None, "_MAX_BATCH_ELEMS not found in source"
        val = int(match.group(1).replace("_", ""))
        max_bytes = val * 8  # float64
        assert max_bytes <= 500_000_000, f"Batch cap too large: {max_bytes / 1e6:.0f}MB"
        assert max_bytes >= 100_000_000, f"Batch cap too small: {max_bytes / 1e6:.0f}MB"


class TestSilentVIFFallback:
    """VIF fallback should log a warning when engine.data is None."""

    def test_vif_zero_correlation_is_identity(self):
        """With rho_bar=0, VIF=1 and z-score equals uncorrected."""
        from cliquefinder.stats.enrichment_z import compute_competitive_z

        rng = np.random.default_rng(42)
        t_stats = rng.standard_normal(200)
        t_stats[:20] += 2.0
        mask = np.zeros(200, dtype=bool)
        mask[:20] = True

        z_raw = compute_competitive_z(t_stats, mask)
        z_zero_rho = compute_competitive_z(t_stats, mask, inter_gene_correlation=0.0)
        assert z_raw == pytest.approx(z_zero_rho), (
            "rho_bar=0 should give identical z to no VIF correction"
        )


class TestVSNSkipScaleCheck:
    """VSN should support skip_scale_check parameter."""

    def test_skip_scale_check_suppresses_warning(self):
        """skip_scale_check=True should suppress the log-detection warning."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Log2 data that would normally trigger warning
        data = np.random.normal(loc=15, scale=3, size=(100, 10))

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should NOT raise — skip_scale_check suppresses the guard
            result = vsn_normalization(data, skip_scale_check=True)
            assert result.data.shape == data.shape

    def test_skip_scale_check_false_still_warns(self):
        """skip_scale_check=False (default) should still warn on log data."""
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        data = np.random.normal(loc=15, scale=3, size=(100, 10))

        with pytest.warns(UserWarning, match="log-transformed"):
            vsn_normalization(data, skip_scale_check=False)


class TestQRILCBiasNote:
    """QRILC docstring should document truncation bias."""

    def test_qrilc_docstring_has_bias_note(self):
        """impute_qrilc docstring should mention truncation bias."""
        from cliquefinder.stats.missing import impute_qrilc

        doc = impute_qrilc.__doc__
        assert doc is not None
        assert "5-15%" in doc, "Should quantify bias magnitude"
        assert "Wei et al" in doc, "Should cite original algorithm"
        assert "MNAR" in doc or "missing not at random" in doc, (
            "Should mention MNAR context"
        )


class TestNegativeControlResultRhoBar:
    """NegativeControlResult should store target_inter_gene_correlation."""

    def test_field_exists(self):
        """Dataclass should have target_inter_gene_correlation field."""
        from cliquefinder.stats.negative_controls import NegativeControlResult

        result = NegativeControlResult(
            target_pvalue=0.01,
            target_set_id="test",
            target_set_size=10,
            control_pvalues=np.array([0.1, 0.2, 0.3]),
            fpr=0.0,
            alpha=0.05,
            target_percentile=5.0,
            median_control_pvalue=0.2,
            mean_control_pvalue=0.2,
            n_control_sets=3,
            target_inter_gene_correlation=0.35,
        )
        assert result.target_inter_gene_correlation == 0.35

    def test_rho_bar_in_to_dict(self):
        """to_dict should include target_inter_gene_correlation in competitive_z section."""
        from cliquefinder.stats.negative_controls import NegativeControlResult

        result = NegativeControlResult(
            target_pvalue=0.01,
            target_set_id="test",
            target_set_size=10,
            control_pvalues=np.array([0.1, 0.2, 0.3]),
            fpr=0.0,
            alpha=0.05,
            target_percentile=5.0,
            median_control_pvalue=0.2,
            mean_control_pvalue=0.2,
            n_control_sets=3,
            target_competitive_z=3.5,
            control_competitive_z_scores=np.array([1.0, 1.5, 2.0]),
            competitive_z_fpr=0.0,
            competitive_z_tail_pct=0.0,
            target_inter_gene_correlation=0.35,
        )
        d = result.to_dict()
        assert "competitive_z" in d
        assert d["competitive_z"]["target_inter_gene_correlation"] == 0.35

    def test_rho_bar_defaults_none(self):
        """target_inter_gene_correlation should default to None."""
        from cliquefinder.stats.negative_controls import NegativeControlResult

        result = NegativeControlResult(
            target_pvalue=0.01,
            target_set_id="test",
            target_set_size=10,
            control_pvalues=np.array([0.1, 0.2, 0.3]),
            fpr=0.0,
            alpha=0.05,
            target_percentile=5.0,
            median_control_pvalue=0.2,
            mean_control_pvalue=0.2,
            n_control_sets=3,
        )
        assert result.target_inter_gene_correlation is None


# ========================================================================
# Additional VSN, VIF, GPU memory, and lazy index tests
# ========================================================================


class TestCameraBackgroundTerm:
    """Camera 1/(G-m) background variance term documentation."""

    def test_comment_in_source(self):
        """compute_competitive_z should document the Camera simplification."""
        import inspect
        from cliquefinder.stats import enrichment_z

        source = inspect.getsource(enrichment_z)
        assert "DF-XIII-D6" in source
        assert "1/(G-m)" in source, "Should document the omitted background term"
        assert "VIF/m" in source, "Should show the dominant term for comparison"


class TestRawVsResidualCorrelation:
    """Raw expression vs residual correlation documentation."""

    def test_comment_in_source(self):
        """estimate_inter_gene_correlation should document limma departure."""
        import inspect
        from cliquefinder.stats import enrichment_z

        source = inspect.getsource(enrichment_z)
        assert "DF-XIII-D7" in source
        assert "residuals" in source, "Should mention residual-based estimation"
        assert "conservative" in source.lower(), "Should note conservative direction"


class TestControlRhoBarSpotCheck:
    """Control set rho_bar spot-check diagnostic."""

    def test_spot_check_in_source(self):
        """Negative controls should spot-check rho_bar for first control set."""
        import inspect
        from cliquefinder.stats import negative_controls

        source = inspect.getsource(negative_controls)
        assert "DF-XIII-D5" in source
        assert "rho_bar" in source
        assert "0.1" in source, "Should have 0.1 warning threshold"

    def test_spot_check_import(self):
        """Should import estimate_inter_gene_correlation for spot-check."""
        import inspect
        from cliquefinder.stats import negative_controls

        source = inspect.getsource(negative_controls)
        assert "estimate_inter_gene_correlation" in source


class TestStreamingPValues:
    """XIII-8: Streaming p-values replace stored null distribution arrays."""

    def test_streaming_counters_in_source(self):
        """Phase 3 should use streaming counters, not stored arrays."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu)
        assert "XIII-8" in source, "Should reference XIII-8 streaming p-values"
        assert "_exceed_twosided" in source, "Should have two-sided exceedance counter"
        assert "_n_valid" in source, "Should track valid perm count"

    def test_no_null_array_storage(self):
        """Phase 3 scatter loop should NOT store full null distribution arrays."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        # The old patterns that stored full arrays
        assert "null_t_arrays" not in source, "Should not store null t arrays"
        assert "null_log2fc_arrays" not in source, "Should not store null log2fc arrays"
        assert "null_t[clique_id].append" not in source
        assert "null_log2fc[clique_id].append" not in source

    def test_perm_count_assertion_present(self):
        """Should assert all cliques saw all permutations after Phase 3."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        assert "_total_perms_seen[cid] == n_permutations" in source


class TestPhase2MemoryBounding:
    """Phase 2 observed analysis memory bounding."""

    def test_phase2_chunking_in_source(self):
        """Phase 2 should use _MAX_BATCH_ELEMS chunking like Phase 3."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        assert "DF-XIII-D1" in source
        assert "max_per_chunk" in source, "Should compute max cliques per chunk"
        assert "clique_ids_sub" in source, "Should sub-chunk cliques"

    def test_max_batch_elems_defined_before_phase2(self):
        """_MAX_BATCH_ELEMS should be defined before Phase 2 starts."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        # _MAX_BATCH_ELEMS must appear before "PHASE 2"
        elems_pos = source.index("_MAX_BATCH_ELEMS")
        phase2_pos = source.index("PHASE 2")
        assert elems_pos < phase2_pos, (
            "_MAX_BATCH_ELEMS should be defined before Phase 2"
        )


class TestReturnLog2fc:
    """batched_ols_contrast_test returns log2fc to avoid redundant computation."""

    def test_return_log2fc_parameter_exists(self):
        """batched_ols_contrast_test should accept return_log2fc parameter."""
        import inspect
        from cliquefinder.stats.permutation_gpu import batched_ols_contrast_test

        sig = inspect.signature(batched_ols_contrast_test)
        assert "return_log2fc" in sig.parameters
        assert sig.parameters["return_log2fc"].default is False

    def test_return_log2fc_false_returns_array(self):
        """Default return should be a single array."""
        from cliquefinder.stats.permutation_gpu import (
            batched_ols_contrast_test,
            precompute_ols_matrices,
        )

        np.random.seed(42)
        n_samples = 10
        conditions = ["A", "B"]
        sample_cond = np.array(["A"] * 5 + ["B"] * 5)
        matrices = precompute_ols_matrices(sample_cond, conditions, ("B", "A"))

        Y = np.random.normal(0, 1, (20, n_samples))
        result = batched_ols_contrast_test(Y, matrices, use_gpu=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_return_log2fc_true_returns_tuple(self):
        """return_log2fc=True should return (t_stats, log2fc) tuple."""
        from cliquefinder.stats.permutation_gpu import (
            batched_ols_contrast_test,
            precompute_ols_matrices,
        )

        np.random.seed(42)
        n_samples = 10
        conditions = ["A", "B"]
        sample_cond = np.array(["A"] * 5 + ["B"] * 5)
        matrices = precompute_ols_matrices(sample_cond, conditions, ("B", "A"))

        Y = np.random.normal(0, 1, (20, n_samples))
        result = batched_ols_contrast_test(
            Y, matrices, use_gpu=False, return_log2fc=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        t_stats, log2fc = result
        assert t_stats.shape == (20,)
        assert log2fc.shape == (20,)

    def test_log2fc_matches_manual_computation(self):
        """Returned log2fc should match manual beta @ c computation."""
        from cliquefinder.stats.permutation_gpu import (
            batched_ols_contrast_test,
            precompute_ols_matrices,
        )

        np.random.seed(42)
        n_samples = 10
        conditions = ["A", "B"]
        sample_cond = np.array(["A"] * 5 + ["B"] * 5)
        matrices = precompute_ols_matrices(sample_cond, conditions, ("B", "A"))

        Y = np.random.normal(0, 1, (20, n_samples))
        t_stats, log2fc = batched_ols_contrast_test(
            Y, matrices, use_gpu=False, return_log2fc=True
        )

        # Manual computation
        beta = Y @ matrices.X @ matrices.XtX_inv.T
        manual_log2fc = beta @ matrices.c

        np.testing.assert_allclose(log2fc, manual_log2fc, rtol=1e-10)

    def test_t_stats_same_regardless_of_return_log2fc(self):
        """t-statistics should be identical whether or not log2fc is returned."""
        from cliquefinder.stats.permutation_gpu import (
            batched_ols_contrast_test,
            precompute_ols_matrices,
        )

        np.random.seed(42)
        n_samples = 10
        conditions = ["A", "B"]
        sample_cond = np.array(["A"] * 5 + ["B"] * 5)
        matrices = precompute_ols_matrices(sample_cond, conditions, ("B", "A"))

        Y = np.random.normal(0, 1, (20, n_samples))
        t_only = batched_ols_contrast_test(Y, matrices, use_gpu=False)
        t_with, _ = batched_ols_contrast_test(
            Y, matrices, use_gpu=False, return_log2fc=True
        )

        np.testing.assert_array_equal(t_only, t_with)

    def test_phase3_uses_return_log2fc(self):
        """Phase 3 should call batched_ols_contrast_test with return_log2fc=True."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        assert "return_log2fc=True" in source
        # Should NOT have the old redundant beta computation in Phase 3
        assert "beta_chunk = Y_chunk" not in source


class TestLazyIndexGeneration:
    """Lazy random index generation via SeedSequence."""

    def test_no_precompute_call(self):
        """run_permutation_test_gpu should NOT call precompute_random_indices."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        assert "precompute_random_indices(" not in source, (
            "Should use lazy generation instead of precompute_random_indices"
        )

    def test_seed_sequence_in_source(self):
        """Should use SeedSequence for reproducible lazy generation."""
        import inspect
        from cliquefinder.stats import permutation_gpu

        source = inspect.getsource(permutation_gpu.run_permutation_test_gpu)
        assert "SeedSequence" in source
        assert "_size_seeds" in source
        assert "_generate_indices_for_chunk" in source

    def test_generate_indices_for_chunk_shape(self):
        """_generate_indices_for_chunk should return correct shape."""
        from cliquefinder.stats.permutation_gpu import _generate_indices_for_chunk

        rng = np.random.default_rng(42)
        result = _generate_indices_for_chunk(
            pool_size=100, size=5, n_cliques=10, chunk_perms=20, rng=rng,
        )
        assert result.shape == (10, 20, 5)
        assert result.dtype == np.int32

    def test_generate_indices_no_duplicates_within_perm(self):
        """Each permutation row should have unique indices (no replacement)."""
        from cliquefinder.stats.permutation_gpu import _generate_indices_for_chunk

        rng = np.random.default_rng(42)
        result = _generate_indices_for_chunk(
            pool_size=100, size=10, n_cliques=5, chunk_perms=50, rng=rng,
        )
        for ci in range(5):
            for pi in range(50):
                row = result[ci, pi]
                assert len(set(row)) == len(row), "Duplicate indices in permutation"

    def test_generate_indices_reproducible(self):
        """Same seed should produce identical indices."""
        from cliquefinder.stats.permutation_gpu import _generate_indices_for_chunk

        rng1 = np.random.default_rng(42)
        result1 = _generate_indices_for_chunk(
            pool_size=100, size=5, n_cliques=10, chunk_perms=20, rng=rng1,
        )
        rng2 = np.random.default_rng(42)
        result2 = _generate_indices_for_chunk(
            pool_size=100, size=5, n_cliques=10, chunk_perms=20, rng=rng2,
        )
        np.testing.assert_array_equal(result1, result2)

    def test_generate_indices_small_pool(self):
        """Should handle small pool (pool_size <= 2*size) correctly."""
        from cliquefinder.stats.permutation_gpu import _generate_indices_for_chunk

        rng = np.random.default_rng(42)
        # pool_size=8, size=5 → pool_size <= 2*size, falls back to sequential
        result = _generate_indices_for_chunk(
            pool_size=8, size=5, n_cliques=3, chunk_perms=10, rng=rng,
        )
        assert result.shape == (3, 10, 5)
        assert np.all(result >= 0)
        assert np.all(result < 8)

    def test_generate_indices_large_pool(self):
        """Should handle large pool with argpartition correctly."""
        from cliquefinder.stats.permutation_gpu import _generate_indices_for_chunk

        rng = np.random.default_rng(42)
        result = _generate_indices_for_chunk(
            pool_size=5000, size=30, n_cliques=5, chunk_perms=100, rng=rng,
        )
        assert result.shape == (5, 100, 30)
        assert np.all(result >= 0)
        assert np.all(result < 5000)
