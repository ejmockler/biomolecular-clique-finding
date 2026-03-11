"""Tests for rotation engine gene filtering and squeeze_var d0=inf handling.

- squeeze_var d0=inf returns prior variance (not original).
- fit_general() filters zero-variance and NaN genes like fit().
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.permutation_gpu import squeeze_var
from cliquefinder.stats.rotation import RotationTestEngine


# ---------------------------------------------------------------------------
# squeeze_var d0=inf
# ---------------------------------------------------------------------------


class TestSqueezeVarD0Inf:
    """When d0=inf the prior dominates: all posteriors should equal s0_sq."""

    def test_squeeze_var_d0_inf_returns_prior(self):
        """Verify squeeze_var(d0=inf) returns np.full_like(sigma2, s0_sq)
        and df_total=inf for scalar df."""
        sigma2 = np.array([1.0, 2.0, 3.0, 4.0])
        df = 10
        s0_sq = 0.5

        s2_post, df_total = squeeze_var(sigma2, df, d0=np.inf, s0_sq=s0_sq)

        np.testing.assert_array_equal(s2_post, np.full_like(sigma2, s0_sq))
        assert df_total == np.inf

    def test_squeeze_var_d0_inf_array_df(self):
        """Same but with array df input: df_total should be array of inf."""
        sigma2 = np.array([1.0, 2.0, 3.0])
        df = np.array([8.0, 10.0, 12.0])
        s0_sq = 0.75

        s2_post, df_total = squeeze_var(sigma2, df, d0=np.inf, s0_sq=s0_sq)

        np.testing.assert_array_equal(s2_post, np.full_like(sigma2, s0_sq))
        assert isinstance(df_total, np.ndarray)
        assert np.all(np.isinf(df_total))
        assert df_total.shape == df.shape

    def test_squeeze_var_d0_inf_preserves_shape(self):
        """Verify output shape matches input shape for various sizes."""
        for n in [1, 10, 100]:
            sigma2 = np.random.default_rng(42).standard_normal(n) ** 2 + 0.1
            df = 5
            s0_sq = 1.0

            s2_post, df_total = squeeze_var(sigma2, df, d0=np.inf, s0_sq=s0_sq)

            assert s2_post.shape == sigma2.shape
            assert df_total == np.inf

    def test_squeeze_var_d0_inf_does_not_return_original(self):
        """Regression: d0=inf must NOT return original variances."""
        sigma2 = np.array([10.0, 20.0, 30.0])
        s0_sq = 1.0

        s2_post, _ = squeeze_var(sigma2, df=5, d0=np.inf, s0_sq=s0_sq)

        # Old bug: s2_post would equal sigma2. Now it must equal s0_sq.
        assert not np.allclose(s2_post, sigma2)
        np.testing.assert_array_equal(s2_post, s0_sq)

    def test_squeeze_var_finite_d0_still_works(self):
        """Sanity check: finite d0 path is unaffected."""
        sigma2 = np.array([1.0, 2.0, 3.0])
        df = 10
        d0 = 5.0
        s0_sq = 0.5

        s2_post, df_total = squeeze_var(sigma2, df, d0=d0, s0_sq=s0_sq)

        expected = (d0 * s0_sq + df * sigma2) / (d0 + df)
        np.testing.assert_allclose(s2_post, expected)
        assert df_total == pytest.approx(d0 + df)


# ---------------------------------------------------------------------------
# fit_general() gene filtering
# ---------------------------------------------------------------------------


def _make_engine(
    n_genes: int = 50,
    n_samples: int = 20,
    seed: int = 42,
    add_zero_var: bool = False,
    add_nan: bool = False,
):
    """Helper: create a RotationTestEngine with optional degenerate genes.

    Gene 0 is zero-variance if add_zero_var is True.
    Gene 1 has a NaN if add_nan is True.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_genes, n_samples))

    if add_zero_var:
        data[0, :] = 5.0  # constant row

    if add_nan:
        data[1, 3] = np.nan

    gene_ids = [f"GENE{i}" for i in range(n_genes)]
    conditions = ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2)
    metadata = pd.DataFrame({"condition": conditions})

    return RotationTestEngine(data, gene_ids, metadata)


class TestFitGeneralFiltersZeroVariance:
    """fit_general() should remove zero-variance genes before fitting."""

    def test_fit_general_filters_zero_variance(self):
        engine = _make_engine(add_zero_var=True, add_nan=False)
        n_samples = engine.data.shape[1]

        design = np.column_stack([
            np.ones(n_samples),
            (np.array(["A"] * 10 + ["B"] * 10) == "B").astype(float),
        ])
        contrast = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit_general(design, contrast, "test")

        # GENE0 (zero-variance) should be removed
        assert "GENE0" not in engine.gene_ids
        assert engine.data.shape[0] == 49
        assert len(engine.gene_ids) == 49
        # Check a warning was emitted
        zero_var_warnings = [
            x for x in w if "zero-variance" in str(x.message)
        ]
        assert len(zero_var_warnings) >= 1

    def test_fit_general_filters_nan_genes(self):
        engine = _make_engine(add_zero_var=False, add_nan=True)
        n_samples = engine.data.shape[1]

        design = np.column_stack([
            np.ones(n_samples),
            (np.array(["A"] * 10 + ["B"] * 10) == "B").astype(float),
        ])
        contrast = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.fit_general(design, contrast, "test")

        # GENE1 (NaN) should be removed
        assert "GENE1" not in engine.gene_ids
        assert engine.data.shape[0] == 49
        nan_warnings = [x for x in w if "NaN" in str(x.message)]
        assert len(nan_warnings) >= 1


class TestFitGeneralSameFiltersAsFit:
    """fit() and fit_general() should produce identical gene filtering."""

    def test_fit_general_same_filters_as_fit(self):
        # Build two engines with the same degenerate data
        engine_fit = _make_engine(add_zero_var=True, add_nan=True)
        engine_general = _make_engine(add_zero_var=True, add_nan=True)

        n_samples = engine_fit.data.shape[1]

        # fit() path
        engine_fit.fit(
            conditions=["A", "B"],
            contrast=("B", "A"),
            condition_column="condition",
        )

        # fit_general() path
        design = np.column_stack([
            np.ones(n_samples),
            (np.array(["A"] * 10 + ["B"] * 10) == "B").astype(float),
        ])
        contrast = np.array([0.0, 1.0])
        engine_general.fit_general(design, contrast, "test")

        # Both should have the same gene list after filtering
        assert engine_fit.gene_ids == engine_general.gene_ids
        assert engine_fit.data.shape == engine_general.data.shape


class TestFitStillWorksAfterRefactor:
    """Regression test: fit() should still work correctly after refactor."""

    def test_fit_still_works_after_refactor(self):
        engine = _make_engine(add_zero_var=False, add_nan=False)

        engine.fit(
            conditions=["A", "B"],
            contrast=("B", "A"),
            condition_column="condition",
        )

        # Engine should be fitted
        assert engine._fitted
        # All 50 genes should remain (no degenerate genes)
        assert len(engine.gene_ids) == 50
        assert engine.data.shape[0] == 50
        # Precomputed and effects should be populated
        assert engine._precomputed is not None
        assert engine._effects is not None

    def test_fit_with_degenerate_genes_still_produces_results(self):
        """fit() with degenerate genes should still complete successfully."""
        engine = _make_engine(add_zero_var=True, add_nan=True)

        engine.fit(
            conditions=["A", "B"],
            contrast=("B", "A"),
            condition_column="condition",
        )

        assert engine._fitted
        # GENE0 and GENE1 should be removed
        assert len(engine.gene_ids) == 48
        assert "GENE0" not in engine.gene_ids
        assert "GENE1" not in engine.gene_ids
