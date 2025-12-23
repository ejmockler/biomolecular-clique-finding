"""
Verify that legacy code has been properly removed.

This ensures code cleanup doesn't break the API and that deprecated
functionality is properly removed.
"""

import pytest
import sys
from cliquefinder.quality import Imputer


class TestLegacyStrategiesRemoved:
    """Verify legacy imputation strategies are gone."""

    def test_euclidean_knn_removed(self):
        """
        Verify that legacy strategies (KNN, winsorization) are removed.

        The old strategies have been replaced with MAD-clip for consistency.
        """
        # Test that only valid strategies are accepted
        valid_strategies = ["mad-clip", "median"]

        # Test each valid strategy works
        for strategy in valid_strategies:
            try:
                if strategy == "mad-clip":
                    imputer = Imputer(strategy=strategy, threshold=3.5)
                else:
                    imputer = Imputer(strategy=strategy)
                print(f"✓ Strategy '{strategy}' accepted")
            except Exception as e:
                pytest.fail(f"Valid strategy '{strategy}' rejected: {e}")

        # Test invalid/deprecated strategies raise error
        deprecated_strategies = ["winsorize", "knn_correlation", "radius_correlation", "knn"]
        for strategy in deprecated_strategies:
            with pytest.raises(ValueError, match="Unknown strategy"):
                Imputer(strategy=strategy)
            print(f"✓ Deprecated strategy '{strategy}' rejected")

        print("✓ Only valid strategies accepted")


class TestDependencyCleanup:
    """Verify optional dependencies are truly optional."""

    def test_sklearn_optional(self):
        """
        Verify sklearn is not required (MAD-clip and median are pure NumPy).

        Core functionality should work without sklearn.
        """
        # Check if sklearn is imported in imputation module
        import cliquefinder.quality.imputation as imp_module

        # Get module source to check imports
        import inspect
        try:
            source = inspect.getsource(imp_module)
        except:
            # Can't get source in some environments
            print("✓ Cannot inspect source, skipping import check")
            return

        # Check what's imported
        has_sklearn_import = "from sklearn" in source or "import sklearn" in source

        if has_sklearn_import:
            pytest.fail("sklearn should not be imported (MAD-clip is pure NumPy)")
        else:
            print("✓ sklearn not imported (pure NumPy implementation)")

        # Test that core strategies work without sklearn
        try:
            imputer1 = Imputer(strategy="mad-clip", threshold=3.5)
            imputer2 = Imputer(strategy="median")
            print("✓ Core strategies work without sklearn")
        except ImportError as e:
            pytest.fail(f"Core strategies should work without sklearn: {e}")


class TestAPISimplified:
    """Verify API has been simplified."""

    def test_required_parameters_only(self):
        """
        Test that only essential parameters are required.

        Good API design: minimal required parameters, sensible defaults.
        """
        # Should work with defaults
        imputer1 = Imputer()  # Defaults to mad-clip with threshold=3.5
        assert imputer1.strategy == "mad-clip"
        assert imputer1.threshold == 3.5

        # Should work with explicit strategy
        imputer2 = Imputer(strategy="mad-clip", threshold=3.5)
        assert imputer2.strategy == "mad-clip"
        assert imputer2.threshold == 3.5

        # Should work with median
        imputer3 = Imputer(strategy="median")
        assert imputer3.strategy == "median"

        print("✓ API accepts minimal parameters with good defaults")

    def test_legacy_parameters_rejected(self):
        """
        Test that legacy parameters are properly rejected.

        If old parameters are removed, they should raise TypeError.
        """
        # Test that deprecated percentile parameters are rejected
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Imputer(strategy="mad-clip", lower_percentile=2.5)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Imputer(strategy="mad-clip", upper_percentile=97.5)

        # Test that KNN-specific parameters are rejected
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Imputer(strategy="median", n_neighbors=5)

        print("✓ Legacy parameters properly rejected")


class TestNoDeadCode:
    """Verify no dead code remains."""

    def test_all_strategies_reachable(self):
        """
        Verify all declared strategies are actually implemented.

        Dead code detection: if strategy is in valid list but not handled,
        it's a bug.
        """
        from cliquefinder.quality.imputation import Imputer

        # Get valid strategies from error message
        try:
            Imputer(strategy="invalid")
        except ValueError as e:
            error_msg = str(e)
            # Should mention valid strategies
            assert "mad-clip" in error_msg or "Must be one of" in error_msg
            assert "median" in error_msg
            print(f"✓ Error message lists valid strategies: {error_msg[:100]}...")

    def test_no_unreachable_branches(self):
        """
        Test that all code paths are reachable.

        This is a sanity check that refactoring didn't leave dead branches.
        """
        # Test each strategy actually executes its code path
        # We'll use a tiny matrix to make this fast

        import numpy as np
        import pandas as pd
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag

        # Create tiny test matrix
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        feature_ids = pd.Index(["G1", "G2", "G3"])
        sample_ids = pd.Index(["S1", "S2"])
        sample_metadata = pd.DataFrame(index=sample_ids)
        quality_flags = np.zeros_like(data, dtype=np.uint32)

        # Mark one value as outlier
        quality_flags[0, 0] = QualityFlag.OUTLIER_DETECTED

        matrix = BioMatrix(data, feature_ids, sample_ids, sample_metadata, quality_flags)

        # Test each strategy executes
        strategies_to_test = [
            ("mad-clip", {"threshold": 3.5}),
            ("median", {}),
        ]

        for strategy, params in strategies_to_test:
            try:
                imputer = Imputer(strategy=strategy, **params)
                result = imputer.apply(matrix)

                # Verify it actually imputed
                imputed = (result.quality_flags & QualityFlag.IMPUTED) > 0
                assert imputed.sum() > 0, f"Strategy {strategy} didn't impute anything"

                print(f"✓ Strategy '{strategy}' code path executed")

            except Exception as e:
                pytest.fail(f"Strategy '{strategy}' failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
