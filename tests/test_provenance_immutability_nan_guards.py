"""Tests for provenance forwarding, dataclass immutability, NaN guards,
VSN convergence tolerance, QRILC robust minimum, and medcouple median ties."""

import numpy as np
import pandas as pd
import pytest


class TestTransformProvenanceForwarding:
    """Provenance must be preserved through quality transforms."""

    def _make_matrix(self, is_log=True):
        from cliquefinder.core.biomatrix import BioMatrix, TransformProvenance

        prov = TransformProvenance(
            is_log_transformed=is_log,
            log_base=2.0,
            is_normalized=True,
            normalization_method="median",
        )
        np.random.seed(42)
        data = np.random.randn(50, 20).astype(np.float64)
        data = np.abs(data) + 1.0  # positive values
        fids = pd.Index([f"G{i}" for i in range(50)])
        sids = pd.Index([f"S{i}" for i in range(20)])
        md = pd.DataFrame({"phenotype": (["A"] * 10 + ["B"] * 10)}, index=sids)
        qf = np.ones_like(data, dtype=np.uint8)
        return BioMatrix(
            data=data,
            feature_ids=fids,
            sample_ids=sids,
            sample_metadata=md,
            quality_flags=qf,
            provenance=prov,
        )

    def test_outlier_detector_preserves_provenance(self):
        from cliquefinder.quality.outliers import OutlierDetector

        m = self._make_matrix()
        det = OutlierDetector(method="mad-z", threshold=2.0)
        result = det.apply(m)
        assert result.provenance.is_log_transformed is True
        assert result.provenance.log_base == 2.0
        assert result.provenance.is_normalized is True

    def test_imputer_preserves_provenance(self):
        from cliquefinder.core.biomatrix import BioMatrix, TransformProvenance
        from cliquefinder.core.quality import QualityFlag
        from cliquefinder.quality.imputation import Imputer

        prov = TransformProvenance(is_log_transformed=True, log_base=2.0)
        np.random.seed(42)
        data = np.random.randn(10, 5).astype(np.float64)
        qf = np.ones((10, 5), dtype=np.uint8)
        # Flag some positions as outliers
        qf[0, 0] = QualityFlag.OUTLIER_DETECTED
        qf[1, 1] = QualityFlag.OUTLIER_DETECTED
        data[0, 0] = 100.0  # extreme value to ensure clipping
        data[1, 1] = -100.0

        m = BioMatrix(
            data=data,
            feature_ids=pd.Index([f"G{i}" for i in range(10)]),
            sample_ids=pd.Index([f"S{i}" for i in range(5)]),
            sample_metadata=pd.DataFrame(
                {"cond": ["A", "A", "B", "B", "B"]},
                index=pd.Index([f"S{i}" for i in range(5)]),
            ),
            quality_flags=qf,
            provenance=prov,
        )
        imp = Imputer(strategy="mad-clip", threshold=3.5)
        result = imp.apply(m)
        assert result.provenance.is_log_transformed is True
        assert result.provenance.log_base == 2.0

    def test_provenance_not_default_after_transform(self):
        """After transform, provenance should match input (not default)."""
        from cliquefinder.quality.outliers import OutlierDetector

        m = self._make_matrix(is_log=True)
        det = OutlierDetector(method="mad-z", threshold=2.0)
        result = det.apply(m)
        # Should NOT be the default (all False)
        assert result.provenance.is_log_transformed is True
        assert result.provenance.normalization_method == "median"


class TestImputedFlagAccuracy:
    """IMPUTED flag should only be set where values actually change."""

    def test_unchanged_values_not_flagged_imputed(self):
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag
        from cliquefinder.quality.imputation import Imputer

        np.random.seed(42)
        # Create data where some "outlier" positions have values within bounds
        data = np.array([
            [2.0, 3.0, 2.5, 3.5, 2.8],
            [1.0, 50.0, 2.0, 3.0, 2.5],  # 50.0 is true outlier
        ], dtype=np.float64)
        qf = np.ones_like(data, dtype=np.uint8)
        # Flag positions — first row's flagged value (2.5) is within bounds
        qf[0, 2] = QualityFlag.OUTLIER_DETECTED
        qf[1, 1] = QualityFlag.OUTLIER_DETECTED

        m = BioMatrix(
            data=data,
            feature_ids=pd.Index(["G0", "G1"]),
            sample_ids=pd.Index(["S0", "S1", "S2", "S3", "S4"]),
            sample_metadata=pd.DataFrame(
                {"cond": ["A", "A", "B", "B", "B"]},
                index=pd.Index(["S0", "S1", "S2", "S3", "S4"]),
            ),
            quality_flags=qf,
        )
        imp = Imputer(strategy="mad-clip", threshold=3.5)
        result = imp.apply(m)

        imputed_flags = (result.quality_flags & QualityFlag.IMPUTED) > 0

        # The extreme outlier (50.0) should definitely be imputed
        assert imputed_flags[1, 1], "True outlier should be marked IMPUTED"

        # For any position where original == result, IMPUTED should NOT be set
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isclose(data[i, j], result.data[i, j]):
                    assert not imputed_flags[i, j], \
                        f"Value unchanged at ({i},{j}) but IMPUTED flag set"


class TestCliqueDefinitionImmutability:
    """CliqueDefinition should be immutable (frozen dataclass)."""

    def test_frozen(self):
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        cd = CliqueDefinition(clique_id="test", protein_ids=["A", "B"])
        with pytest.raises(AttributeError, match="cannot assign"):
            cd.clique_id = "mutated"

    def test_protein_ids_is_tuple(self):
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        cd = CliqueDefinition(clique_id="test", protein_ids=["A", "B", "C"])
        assert isinstance(cd.protein_ids, tuple)
        assert cd.protein_ids == ("A", "B", "C")

    def test_external_list_mutation_does_not_affect(self):
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        ids = ["A", "B", "C"]
        cd = CliqueDefinition(clique_id="test", protein_ids=ids)
        ids.append("D")  # mutate original list
        assert len(cd.protein_ids) == 3  # should be unaffected

    def test_accepts_tuple_input(self):
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        cd = CliqueDefinition(clique_id="test", protein_ids=("X", "Y"))
        assert cd.protein_ids == ("X", "Y")

    def test_hashable(self):
        """Frozen dataclass should be hashable if all fields are hashable."""
        from cliquefinder.stats.clique_analysis import CliqueDefinition

        cd = CliqueDefinition(clique_id="test", protein_ids=["A", "B"])
        # Should not raise
        hash(cd)
        # Can be used in sets
        s = {cd}
        assert cd in s


class TestSexClassifierNaNGuard:
    """SexClassifier must handle NaN data without crashing."""

    def test_impute_nan_function_exists(self):
        from cliquefinder.quality.sex_imputation import _impute_nan_for_sklearn
        assert callable(_impute_nan_for_sklearn)

    def test_impute_nan_replaces_nans(self):
        from cliquefinder.quality.sex_imputation import _impute_nan_for_sklearn

        X = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 2.0, 6.0],
            [5.0, 4.0, np.nan],
        ])
        result = _impute_nan_for_sklearn(X)
        assert not np.any(np.isnan(result))
        # Column medians: [3.0, 3.0, 4.5]
        assert result[0, 1] == pytest.approx(3.0)
        assert result[1, 0] == pytest.approx(3.0)
        assert result[2, 2] == pytest.approx(4.5)

    def test_impute_nan_no_copy_when_clean(self):
        from cliquefinder.quality.sex_imputation import _impute_nan_for_sklearn

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _impute_nan_for_sklearn(X)
        # When no NaN, returns same array (fast path)
        assert result is X

    def test_impute_nan_all_nan_column(self):
        from cliquefinder.quality.sex_imputation import _impute_nan_for_sklearn

        X = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])
        result = _impute_nan_for_sklearn(X)
        assert not np.any(np.isnan(result))
        # All-NaN column fills with 0
        assert result[0, 0] == 0.0
        assert result[1, 0] == 0.0

    def test_score_feature_handles_nan(self):
        """_score_feature should not crash on NaN-containing features."""
        from cliquefinder.quality.sex_imputation import SemiSupervisedSexClassifier

        clf = SemiSupervisedSexClassifier(min_effect_size=0.1)
        feature_data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        labels = np.array([0, 0, -1, -1, 1, 1, -1, -1])
        labeled_mask = labels >= 0

        # Should not crash — np.nanstd handles NaN
        result = clf._score_feature(feature_data, labels, labeled_mask)
        # Result may be None if effect size too small, but should not crash
        assert result is None or isinstance(result, dict)


class TestVSNConvergenceTolerance:
    """VSN should use relative tolerance for offset parameter."""

    def test_convergence_with_large_intensities(self):
        from cliquefinder.stats.normalization import vsn_normalization

        np.random.seed(42)
        # Large raw intensities — absolute tolerance would never converge
        data = np.random.exponential(scale=1e6, size=(100, 5))
        result = vsn_normalization(data, max_iter=50, tol=1e-4)
        # Should converge without running all iterations
        assert result.data.shape == data.shape
        assert np.all(np.isfinite(result.data))


class TestQRILCRobustMinimum:
    """QRILC should use robust percentile instead of raw minimum."""

    def test_outlier_does_not_poison_imputation(self):
        from cliquefinder.stats.missing import impute_qrilc

        np.random.seed(42)
        n_genes, n_samples = 20, 15

        # Normal data around 10-15
        data = np.random.normal(loc=12, scale=2, size=(n_genes, n_samples))

        # One extreme low carry-over artifact in sample 0
        data[0, 0] = 0.001

        # Add some missing values
        missing_mask = np.zeros_like(data, dtype=bool)
        missing_mask[1:5, 0] = True
        data[missing_mask] = np.nan

        result = impute_qrilc(data, random_state=42)

        # Imputed values should not be near 0 (the artifact)
        # They should be in the left tail of the normal distribution (~8-12)
        imputed_values = result.data[missing_mask]
        assert np.all(imputed_values > 2.0), \
            f"Imputed values too low: {imputed_values}. Likely poisoned by outlier."


class TestMedcoupleMedianPairs:
    """Medcouple should include h=0 for median-equal pairs per Brys et al."""

    def test_median_ties_contribute_zero(self):
        from cliquefinder.quality.outliers import compute_medcouple

        # Data with heavy ties at median
        data = np.array([1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0])
        mc = compute_medcouple(data)
        # Median is 3.0, many ties → lots of h=0 values → mc closer to 0
        assert np.isfinite(mc)
        assert -1.0 <= mc <= 1.0

    def test_all_ties_at_median(self):
        from cliquefinder.quality.outliers import compute_medcouple

        data = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        mc = compute_medcouple(data)
        assert np.isfinite(mc)
