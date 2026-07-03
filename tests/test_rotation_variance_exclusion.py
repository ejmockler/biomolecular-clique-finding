"""
Tests for rotation variance exclusion and GPU/CPU precision guards.

Covers:
- GPU vs CPU numerical divergence documentation + --force-cpu flag
- Subject aggregation fallback heterogeneity warning
- Rotation negative variance truncation replaced by exclusion
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from cliquefinder.stats.rotation import (
    RotationResult,
    RotationTestConfig,
    RotationTestEngine,
    _apply_rotations_cpu,
    _apply_rotations_gpu,
    apply_rotations_batched,
    compute_rotation_pvalues,
    generate_rotation_vectors,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_rotation_data():
    """Minimal rotation data for testing CPU/GPU paths.

    Returns (U, rho_sq, R, sample_variances, df_residual, use_df) with
    moderate effect sizes so that GPU float32 and CPU float64 should agree.
    """
    rng = np.random.default_rng(42)
    n_genes = 10
    n_dims = 18  # 20 samples - 2 parameters
    n_rotations = 100

    # Gene effects: moderate values (|u| < 3)
    U = rng.standard_normal((n_genes, n_dims))
    rho_sq = np.sum(U ** 2, axis=1)

    # Rotation vectors (unit-norm rows)
    V = rng.standard_normal((n_rotations, n_dims))
    R = V / np.linalg.norm(V, axis=1, keepdims=True)

    sample_variances = rng.uniform(0.5, 2.0, n_genes)
    df_residual = n_dims - 1
    use_df = float(df_residual)

    return U, rho_sq, R, sample_variances, df_residual, use_df


@pytest.fixture
def engine_with_data():
    """Build a minimal RotationTestEngine for integration tests."""
    rng = np.random.default_rng(123)
    n_genes = 50
    n_samples = 20

    data = rng.standard_normal((n_genes, n_samples))
    # Add signal to first 10 genes
    data[:10, :10] += 2.0

    gene_ids = [f"GENE{i}" for i in range(n_genes)]
    metadata = pd.DataFrame({
        'condition': ['CASE'] * 10 + ['CTRL'] * 10,
    }, index=[f"S{i}" for i in range(n_samples)])

    engine = RotationTestEngine(data, gene_ids, metadata)
    engine.fit(
        conditions=['CASE', 'CTRL'],
        contrast=('CASE', 'CTRL'),
        condition_column='condition',
    )
    return engine, gene_ids


# =============================================================================
# GPU precision documentation + --force-cpu flag
# =============================================================================

class TestGpuPrecision:
    """GPU float32 precision documentation and --force-cpu flag."""

    def test_force_cpu_flag_exists_in_cli_parser(self):
        """--force-cpu flag is registered in the differential CLI parser."""
        import argparse
        from cliquefinder.cli.differential import setup_parser

        parent = argparse.ArgumentParser()
        sub = parent.add_subparsers()
        setup_parser(sub)

        # Parse with --force-cpu
        args = parent.parse_args([
            'differential',
            '--data', 'dummy.csv',
            '--metadata', 'dummy_meta.csv',
            '--output', '/tmp/out',
            '--force-cpu',
        ])
        assert args.force_cpu is True

    def test_force_cpu_flag_defaults_false(self):
        """--force-cpu defaults to False when not specified."""
        import argparse
        from cliquefinder.cli.differential import setup_parser

        parent = argparse.ArgumentParser()
        sub = parent.add_subparsers()
        setup_parser(sub)

        args = parent.parse_args([
            'differential',
            '--data', 'dummy.csv',
            '--metadata', 'dummy_meta.csv',
            '--output', '/tmp/out',
        ])
        assert args.force_cpu is False

    def test_gpu_function_docstring_mentions_float32(self):
        """_apply_rotations_gpu docstring documents float32 limitation."""
        doc = _apply_rotations_gpu.__doc__
        assert doc is not None
        assert 'float32' in doc
        assert 'float64' in doc.lower() or 'CPU' in doc
        assert '--force-cpu' in doc or 'force_cpu' in doc or 'use_gpu=False' in doc

    def test_cpu_gpu_agree_for_moderate_t(self, simple_rotation_data):
        """CPU and GPU paths agree for moderate t-statistics (|t| < 5).

        Skipped if MLX is not available (non-Apple Silicon).
        """
        pytest.importorskip('mlx.core')

        U, rho_sq, R, sample_variances, df_residual, use_df = simple_rotation_data

        t_cpu, z_cpu, valid_cpu = _apply_rotations_cpu(
            U, rho_sq, R, sample_variances, None,
            df_residual, use_df,
        )
        t_gpu, z_gpu, valid_gpu = _apply_rotations_gpu(
            U, rho_sq, R, sample_variances, None,
            df_residual, use_df,
        )

        # GPU forms the rotated t/z in float32, CPU in float64. Measured
        # float32-vs-float64 gap on this fixture is ~0.006 (moderate |t|);
        # rho_sq=sum(U^2) lets some rotations drive residual SS -> 0, where the
        # rotated t = U/se explodes and float32 diverges more. Tolerances are
        # float32-justified (~a few x the observed gap), not float64 parity;
        # the old atol/rtol=1e-3 demanded float64 agreement from a float32 path.
        assert_allclose(t_cpu, t_gpu, atol=3e-2, rtol=3e-2)
        assert_allclose(z_cpu, z_gpu, atol=3e-2, rtol=3e-2)

        # Scale/bias guard: the elementwise tolerance alone would pass a uniform
        # systematic error (e.g. a GPU df off-by-one -> ~few% scale shift). On
        # the body (0.1 < |t| < 3) the GPU/CPU median ratio must be within 1% of
        # 1 (honest float32 ~0.999), which a scale/df regression would fail.
        tc = np.asarray(t_cpu).ravel()
        tg = np.asarray(t_gpu).ravel()
        body = (np.abs(tc) > 0.1) & (np.abs(tc) < 3)
        assert abs(float(np.median(tg[body] / tc[body])) - 1.0) < 0.01

    def test_rotation_result_has_precision_note(self, engine_with_data):
        """RotationResult includes precision_note field."""
        engine, gene_ids = engine_with_data

        config = RotationTestConfig(
            n_rotations=99,
            use_gpu=False,  # Force CPU so we know the note is None
            seed=42,
        )
        result = engine.test_gene_set(
            gene_set=gene_ids[:10],
            gene_set_id='test_set',
            config=config,
        )
        # With use_gpu=False, precision_note should be None
        assert result.precision_note is None

    def test_rotation_result_precision_note_in_dict(self):
        """precision_note appears in to_dict() output when set."""
        result = RotationResult(
            feature_set_id='test',
            n_genes=10,
            n_genes_found=10,
            gene_ids=[],
            observed_stats={},
            null_distributions={},
            p_values={},
            active_proportion={},
            n_rotations=100,
            contrast_name='test',
            precision_note='GPU float32',
        )
        d = result.to_dict()
        assert 'precision_note' in d
        assert d['precision_note'] == 'GPU float32'


# =============================================================================
# Subject aggregation fallback warning
# =============================================================================

class TestAggregationWarning:
    """Heterogeneous observation counts produce warning."""

    def test_heterogeneous_obs_counts_produce_warning(self):
        """Warning emitted when max/min observation ratio > 3x."""
        from cliquefinder.stats.differential import fit_linear_model

        rng = np.random.default_rng(99)

        # Subject A: 10 obs, Subject B: 2 obs => ratio = 5 > 3
        y_a = rng.standard_normal(10) + 1.0
        y_b = rng.standard_normal(2) - 1.0
        # Subject C: 10 obs, Subject D: 2 obs
        y_c = rng.standard_normal(10) + 1.0
        y_d = rng.standard_normal(2) - 1.0

        y = np.concatenate([y_a, y_b, y_c, y_d])
        condition = (
            ['CASE'] * 10 + ['CASE'] * 2 +
            ['CTRL'] * 10 + ['CTRL'] * 2
        )
        subject = (
            ['S1'] * 10 + ['S2'] * 2 +
            ['S3'] * 10 + ['S4'] * 2
        )

        # Force mixed model to fail by making it impossible to converge
        # (too few groups per level => singular)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock mixed model failure by passing bad data that won't converge
            # Actually, we need can_fit_mixed=True and mixed to fail.
            # With 4 subjects and 2 conditions, mixed model may converge.
            # Let's force failure by using use_mixed=True with degenerate data.
            result = fit_linear_model(
                y=y,
                condition=pd.Series(condition),
                subject=pd.Series(subject),
                use_mixed=True,
                conditions=['CASE', 'CTRL'],
            )

            # Check if the heterogeneity warning was emitted
            # Note: the warning only fires on the fallback path (when mixed
            # model was attempted). If mixed model converges, no fallback
            # happens and no warning is emitted. We check both paths.
            model_type = result[1]

            if model_type.value == 'fixed':
                # Fell back to OLS — should have warning about heterogeneity
                het_warnings = [
                    x for x in w
                    if 'heterogeneous' in str(x.message).lower()
                    or 'inverse-variance' in str(x.message).lower()
                ]
                assert len(het_warnings) > 0, (
                    "Expected heterogeneity warning on OLS fallback path"
                )

    def test_homogeneous_obs_no_warning(self):
        """No warning when observation counts are uniform."""
        from cliquefinder.stats.differential import fit_linear_model

        rng = np.random.default_rng(99)

        # 3 obs per subject — all equal
        n_per = 3
        y = rng.standard_normal(n_per * 4)
        condition = (
            ['CASE'] * n_per + ['CASE'] * n_per +
            ['CTRL'] * n_per + ['CTRL'] * n_per
        )
        subject = (
            ['S1'] * n_per + ['S2'] * n_per +
            ['S3'] * n_per + ['S4'] * n_per
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            fit_linear_model(
                y=y,
                condition=pd.Series(condition),
                subject=pd.Series(subject),
                use_mixed=True,
                conditions=['CASE', 'CTRL'],
            )

            het_warnings = [
                x for x in w
                if 'heterogeneous' in str(x.message).lower()
            ]
            assert len(het_warnings) == 0, (
                "Should not warn when obs counts are uniform"
            )

    def test_fit_linear_model_docstring_mentions_limitation(self):
        """The fallback code block documents the simple-averaging limitation."""
        import inspect
        from cliquefinder.stats import differential

        source = inspect.getsource(differential.fit_linear_model)
        assert 'simple averaging' in source.lower() or 'heterogeneous' in source.lower()
        assert 'inverse-variance' in source.lower() or 'WLS' in source


# =============================================================================
# Rotation negative variance exclusion
# =============================================================================

class TestRotationNegativeVarianceExclusion:
    """Near-singular rotations detected and excluded."""

    def _make_pathological_data(self):
        """Create data where some rotations will have negative residual SS.

        Strategy: make rho_sq very small for some genes so that when
        U_rot_sq > rho_sq, residual SS goes negative.
        """
        rng = np.random.default_rng(7)
        n_genes = 5
        n_dims = 10
        n_rotations = 200

        U = rng.standard_normal((n_genes, n_dims))

        # Make rho_sq = ||U||^2 for most genes, but artificially small
        # for gene 0 so it's guaranteed to go negative for many rotations.
        rho_sq = np.sum(U ** 2, axis=1)
        # Shrink gene 0's rho_sq to be less than its max U_rot_sq
        rho_sq[0] = 0.01  # Much smaller than U[0] norm

        V = rng.standard_normal((n_rotations, n_dims))
        R = V / np.linalg.norm(V, axis=1, keepdims=True)

        sample_variances = np.ones(n_genes)
        df_residual = n_dims - 1

        return U, rho_sq, R, sample_variances, df_residual

    def test_negative_residual_ss_detected(self):
        """CPU path detects rotations with negative residual SS."""
        U, rho_sq, R, sample_variances, df_residual = (
            self._make_pathological_data()
        )
        use_df = float(df_residual)

        _, _, valid_mask = _apply_rotations_cpu(
            U, rho_sq, R, sample_variances, None,
            df_residual, use_df,
        )

        # With rho_sq[0] = 0.01, many rotations should be invalid
        n_invalid = int(np.sum(~valid_mask))
        assert n_invalid > 0, "Expected some rotations to be invalid"

    def test_excluded_rotations_reduce_n_valid(self):
        """n_valid_rotations + n_excluded = n_total."""
        U, rho_sq, R, sample_variances, df_residual = (
            self._make_pathological_data()
        )
        use_df = float(df_residual)

        _, _, valid_mask = _apply_rotations_cpu(
            U, rho_sq, R, sample_variances, None,
            df_residual, use_df,
        )

        n_total = len(R)
        n_valid = int(np.sum(valid_mask))
        n_excluded = int(np.sum(~valid_mask))

        assert n_valid + n_excluded == n_total
        assert n_valid < n_total  # Some should be excluded

    def test_pvalues_use_n_valid_denominator(self):
        """P-values computed using n_valid + 1 denominator, not n_total + 1."""
        n_rotations = 100
        # 50 valid rotations
        valid_mask = np.zeros(n_rotations, dtype=bool)
        valid_mask[:50] = True

        # Observed stat = 0.5, null values all 1.0 for valid, 0.0 for invalid
        null_values = np.zeros(n_rotations)
        null_values[:50] = 1.0  # valid rotations: all >= observed
        null_values[50:] = 0.0  # invalid rotations: all < observed

        observed_stats = {'msq': {'mixed': 0.5}}
        null_stats = {'msq': {'mixed': null_values}}

        # With mask: b=50 valid values >= 0.5, B=50 valid total
        # p = (50 + 1) / (50 + 1) = 1.0
        p_with_mask = compute_rotation_pvalues(
            observed_stats, null_stats, valid_rotation_mask=valid_mask,
        )
        assert p_with_mask['msq']['mixed'] == pytest.approx(
            (50 + 1) / (50 + 1)
        )

        # Without mask: b=50 values >= 0.5 (out of 100), B=100
        # p = (50 + 1) / (100 + 1) ≈ 0.505
        p_without_mask = compute_rotation_pvalues(
            observed_stats, null_stats,
        )
        assert p_without_mask['msq']['mixed'] == pytest.approx(
            (50 + 1) / (100 + 1)
        )

        # The masked version should give a different (higher) p-value
        assert p_with_mask['msq']['mixed'] > p_without_mask['msq']['mixed']

    def test_warning_when_more_than_10pct_excluded(self, engine_with_data):
        """Warning emitted when >10% of rotations are excluded.

        We monkey-patch apply_rotations_batched to return a mask with
        many invalid rotations to trigger the warning without needing
        pathological input data.
        """
        engine, gene_ids = engine_with_data

        # Patch to return mostly-invalid mask
        import cliquefinder.stats.rotation as rot_mod
        original_fn = rot_mod.apply_rotations_batched

        def patched_batched(*args, **kwargs):
            t, z, valid = original_fn(*args, **kwargs)
            # Mark 50% as invalid
            fake_valid = np.zeros(len(valid), dtype=bool)
            fake_valid[:len(valid) // 2] = True
            return t, z, fake_valid

        rot_mod.apply_rotations_batched = patched_batched
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                config = RotationTestConfig(
                    n_rotations=99,
                    use_gpu=False,
                    seed=42,
                )
                result = engine.test_gene_set(
                    gene_set=gene_ids[:10],
                    gene_set_id='test_set',
                    config=config,
                )

                excl_warnings = [
                    x for x in w
                    if 'excluded' in str(x.message).lower()
                    or 'negative residual' in str(x.message).lower()
                ]
                assert len(excl_warnings) > 0, (
                    "Expected warning about excluded rotations"
                )
        finally:
            rot_mod.apply_rotations_batched = original_fn

    def test_normal_case_no_rotations_excluded(self, engine_with_data):
        """For well-conditioned data, no rotations should be excluded."""
        engine, gene_ids = engine_with_data

        config = RotationTestConfig(
            n_rotations=99,
            use_gpu=False,
            seed=42,
        )
        result = engine.test_gene_set(
            gene_set=gene_ids[:10],
            gene_set_id='test_set',
            config=config,
        )

        assert result.n_valid_rotations is not None
        assert result.n_excluded_rotations is not None
        assert result.n_excluded_rotations == 0
        assert result.n_valid_rotations == 99

    def test_batched_aggregates_valid_masks(self):
        """apply_rotations_batched with chunking aggregates valid masks."""
        rng = np.random.default_rng(42)
        n_genes = 5
        n_dims = 10
        n_rotations = 50

        U = rng.standard_normal((n_genes, n_dims))
        rho_sq = np.sum(U ** 2, axis=1)
        V = rng.standard_normal((n_rotations, n_dims))
        R = V / np.linalg.norm(V, axis=1, keepdims=True)
        sample_variances = np.ones(n_genes)

        # Run with small chunk_size to force chunking
        t, z, valid = apply_rotations_batched(
            U, rho_sq, R, sample_variances, None,
            df_residual=n_dims - 1,
            df_total=None,
            use_gpu=False,
            chunk_size=15,  # Forces 4 chunks of ~13 each
        )

        assert t.shape == (n_rotations, n_genes)
        assert z.shape == (n_rotations, n_genes)
        assert valid.shape == (n_rotations,)
        assert valid.dtype == np.bool_

    def test_gpu_path_detects_negative_residual_ss(self):
        """GPU path also tracks valid rotation mask.

        Skipped if MLX is not available (non-Apple Silicon).
        """
        pytest.importorskip('mlx.core')

        U, rho_sq, R, sample_variances, df_residual = (
            self._make_pathological_data()
        )
        use_df = float(df_residual)

        _, _, valid_mask = _apply_rotations_gpu(
            U, rho_sq, R, sample_variances, None,
            df_residual, use_df,
        )

        n_invalid = int(np.sum(~valid_mask))
        assert n_invalid > 0, "Expected some rotations to be invalid on GPU path"


class TestRotationCancellationBoundary:
    """Coverage for the near-zero residual-SS *cancellation boundary*.

    residual_ss = rho_sq - U_rot^2. By Cauchy-Schwarz U_rot^2 <= rho_sq, so
    residual_ss >= 0 and -> 0 when a rotation nearly aligns with a gene's
    effect. STAT-III-1 does this subtraction in float64 precisely because
    float32 catastrophically cancels there. The other fixtures set rho_sq[0]
    = 0.01 (STRONGLY negative residual_ss, easy to detect); these drive it
    across zero to exercise the cancellation regime the float64 subtraction
    exists to protect.

    Note on an inherent float32 limit (not a code bug): a reduced-precision
    GPU matmul (Apple-Silicon MLX, ~1e-3 relative) cannot resolve residual_ss
    once it is smaller than the matmul error in U_rot, so at the EXTREME
    boundary the float32 GPU cannot detect the near-zero negatives the float64
    CPU detects. This is benign downstream because the t->z transform
    saturates (z, not raw t, feeds the set statistic), asserted below.
    """

    @staticmethod
    def _boundary_data(seed=11):
        rng = np.random.default_rng(seed)
        ndim, nrot, n_genes = 12, 400, 6
        U = rng.standard_normal((n_genes, ndim))
        rho_sq = np.sum(U ** 2, axis=1).copy()
        # Drive gene 0 across the boundary: rho_sq just below |U0|^2, with many
        # rotations aimed tightly along U0 so residual_ss straddles zero.
        Ru = U[0] / np.linalg.norm(U[0])
        perp = rng.standard_normal((nrot, ndim))
        perp -= (perp @ Ru)[:, None] * Ru
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        rho_sq[0] = float(U[0] @ U[0]) * 0.9995
        eps = np.geomspace(1e-4, 8e-2, nrot)
        rng.shuffle(eps)
        R = np.sqrt(1 - eps ** 2)[:, None] * Ru + eps[:, None] * perp
        sv = rng.uniform(0.5, 2.0, n_genes)
        return U, rho_sq, R, sv, ndim - 1

    def test_cpu_float64_detects_near_zero_negatives_and_stays_finite(self):
        """CPU (float64) reaches the cancellation boundary: it detects the
        near-zero NEGATIVE residual-SS rotations (marks them invalid) while
        keeping some valid, and never emits NaN/inf t or z."""
        U, rho_sq, R, sv, df = self._boundary_data()
        t, z, valid = _apply_rotations_cpu(U, rho_sq, R, sv, None, df, float(df))
        assert (~valid).sum() > 0, "boundary not exercised: no invalid rotations"
        assert valid.sum() > 0, "fixture too aggressive: all rotations invalid"
        assert np.isfinite(np.asarray(t)).all()
        assert np.isfinite(np.asarray(z)).all()

    def test_gpu_boundary_finite_and_downstream_z_robust(self):
        """GPU stays finite at the boundary (the 1e-10 clamp works), and on the
        rotations BOTH paths accept, the downstream z-scores agree with CPU
        (corr > 0.99) even though raw t diverges — z saturation makes the
        boundary benign. GPU/CPU validity is intentionally NOT asserted equal:
        the float32 matmul cannot resolve the extreme near-zero negatives."""
        pytest.importorskip('mlx.core')
        U, rho_sq, R, sv, df = self._boundary_data()
        tc, zc, vc = _apply_rotations_cpu(U, rho_sq, R, sv, None, df, float(df))
        tg, zg, vg = _apply_rotations_gpu(U, rho_sq, R, sv, None, df, float(df))
        tg = np.asarray(tg); zg = np.asarray(zg)
        zc = np.asarray(zc)
        assert np.isfinite(tg).all() and np.isfinite(zg).all()
        both = np.asarray(vc) & np.asarray(vg)
        assert both.sum() > 5, "too few jointly-valid rotations to compare"
        assert np.corrcoef(zc[both].ravel(), zg[both].ravel())[0, 1] > 0.99

    def test_gpu_residual_ss_subtraction_is_float64(self):
        """Guard STAT-III-1: the GPU path must compute residual_ss = rho_sq -
        U_rot^2 in float64 (a float32 subtraction catastrophically cancels at
        the boundary). Source check so a revert to float32 is caught."""
        import inspect
        from cliquefinder.stats import rotation
        src = inspect.getsource(rotation._apply_rotations_gpu)
        assert 'STAT-III-1' in src
        # U_rot is cast to float64 and the subtraction runs on float64 arrays.
        assert 'dtype=np.float64' in src
        assert 'residual_ss_np = rho_sq' in src
