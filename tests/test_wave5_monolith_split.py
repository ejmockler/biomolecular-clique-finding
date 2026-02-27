"""
Tests for ARCH-7: method_comparison.py monolith decomposition.

Validates that the split into sub-modules preserves all public APIs,
backward-compatible import paths, and cross-module identity guarantees.
"""

from __future__ import annotations

import importlib
import types

import pytest


# ---------------------------------------------------------------------------
# Backward-compatibility: all old import paths still resolve
# ---------------------------------------------------------------------------


class TestBackwardCompatImports:
    """All existing import paths from method_comparison must still work."""

    def test_import_methodname(self):
        from cliquefinder.stats.method_comparison import MethodName  # noqa: F401

    def test_import_unified_clique_result(self):
        from cliquefinder.stats.method_comparison import UnifiedCliqueResult  # noqa: F401

    def test_import_concordance_metrics(self):
        from cliquefinder.stats.method_comparison import ConcordanceMetrics  # noqa: F401

    def test_import_clique_test_method(self):
        from cliquefinder.stats.method_comparison import CliqueTestMethod  # noqa: F401

    def test_import_prepared_clique_experiment(self):
        from cliquefinder.stats.method_comparison import PreparedCliqueExperiment  # noqa: F401

    def test_import_prepare_experiment(self):
        from cliquefinder.stats.method_comparison import prepare_experiment  # noqa: F401

    def test_import_ols_method(self):
        from cliquefinder.stats.method_comparison import OLSMethod  # noqa: F401

    def test_import_lmm_method(self):
        from cliquefinder.stats.method_comparison import LMMMethod  # noqa: F401

    def test_import_roast_method(self):
        from cliquefinder.stats.method_comparison import ROASTMethod  # noqa: F401

    def test_import_permutation_method(self):
        from cliquefinder.stats.method_comparison import PermutationMethod  # noqa: F401

    def test_import_compute_pairwise_concordance(self):
        from cliquefinder.stats.method_comparison import compute_pairwise_concordance  # noqa: F401

    def test_import_identify_disagreements(self):
        from cliquefinder.stats.method_comparison import identify_disagreements  # noqa: F401

    def test_import_method_comparison_result(self):
        from cliquefinder.stats.method_comparison import MethodComparisonResult  # noqa: F401

    def test_import_run_method_comparison(self):
        from cliquefinder.stats.method_comparison import run_method_comparison  # noqa: F401

    def test_bulk_import_all_symbols(self):
        """Import all 14 public symbols at once (matches original usage)."""
        from cliquefinder.stats.method_comparison import (
            MethodName,
            UnifiedCliqueResult,
            MethodComparisonResult,
            PreparedCliqueExperiment,
            prepare_experiment,
            OLSMethod,
            LMMMethod,
            ROASTMethod,
            PermutationMethod,
            CliqueTestMethod,
            ConcordanceMetrics,
            compute_pairwise_concordance,
            identify_disagreements,
            run_method_comparison,
        )
        # Spot-check they are the right types
        assert hasattr(MethodName, "OLS")
        assert callable(prepare_experiment)
        assert callable(run_method_comparison)
        assert callable(compute_pairwise_concordance)
        assert callable(identify_disagreements)


class TestStatsInitImports:
    """stats/__init__.py imports must still resolve."""

    def test_stats_methodname(self):
        from cliquefinder.stats import MethodName  # noqa: F401

    def test_stats_unified_clique_result(self):
        from cliquefinder.stats import UnifiedCliqueResult  # noqa: F401

    def test_stats_concordance_metrics(self):
        from cliquefinder.stats import ConcordanceMetrics  # noqa: F401

    def test_stats_method_comparison_result(self):
        from cliquefinder.stats import MethodComparisonResult  # noqa: F401

    def test_stats_prepared_clique_experiment(self):
        from cliquefinder.stats import PreparedCliqueExperiment  # noqa: F401

    def test_stats_run_method_comparison(self):
        from cliquefinder.stats import run_method_comparison  # noqa: F401

    def test_stats_prepare_experiment(self):
        from cliquefinder.stats import prepare_experiment  # noqa: F401


# ---------------------------------------------------------------------------
# Sub-module imports (new import paths)
# ---------------------------------------------------------------------------


class TestSubmoduleImports:
    """New sub-module files are importable."""

    def test_method_comparison_types_module(self):
        from cliquefinder.stats.method_comparison_types import (
            MethodName,
            UnifiedCliqueResult,
            ConcordanceMetrics,
            CliqueTestMethod,
        )
        assert hasattr(MethodName, "OLS")

    def test_experiment_module(self):
        from cliquefinder.stats.experiment import (
            PreparedCliqueExperiment,
            prepare_experiment,
        )
        assert callable(prepare_experiment)

    def test_methods_package_init(self):
        from cliquefinder.stats.methods import (
            OLSMethod,
            LMMMethod,
            ROASTMethod,
            PermutationMethod,
        )
        assert OLSMethod is not None

    def test_methods_ols_submodule(self):
        from cliquefinder.stats.methods.ols import OLSMethod
        assert OLSMethod is not None

    def test_methods_lmm_submodule(self):
        from cliquefinder.stats.methods.lmm import LMMMethod
        assert LMMMethod is not None

    def test_methods_roast_submodule(self):
        from cliquefinder.stats.methods.roast import ROASTMethod
        assert ROASTMethod is not None

    def test_methods_permutation_submodule(self):
        from cliquefinder.stats.methods.permutation import PermutationMethod
        assert PermutationMethod is not None

    def test_concordance_module(self):
        from cliquefinder.stats.concordance import (
            compute_pairwise_concordance,
            identify_disagreements,
            MethodComparisonResult,
        )
        assert callable(compute_pairwise_concordance)
        assert callable(identify_disagreements)

    def test_base_linear_module(self):
        from cliquefinder.stats.methods._base_linear import _BaseLinearMethod
        import abc
        assert issubclass(_BaseLinearMethod, abc.ABC)


# ---------------------------------------------------------------------------
# Identity checks: same object regardless of import path
# ---------------------------------------------------------------------------


class TestIdentity:
    """Objects imported from different paths are the SAME object (no duplication)."""

    def test_methodname_identity(self):
        from cliquefinder.stats.method_comparison import MethodName as A
        from cliquefinder.stats.method_comparison_types import MethodName as B
        assert A is B

    def test_unified_clique_result_identity(self):
        from cliquefinder.stats.method_comparison import UnifiedCliqueResult as A
        from cliquefinder.stats.method_comparison_types import UnifiedCliqueResult as B
        assert A is B

    def test_concordance_metrics_identity(self):
        from cliquefinder.stats.method_comparison import ConcordanceMetrics as A
        from cliquefinder.stats.method_comparison_types import ConcordanceMetrics as B
        assert A is B

    def test_prepared_clique_experiment_identity(self):
        from cliquefinder.stats.method_comparison import PreparedCliqueExperiment as A
        from cliquefinder.stats.experiment import PreparedCliqueExperiment as B
        assert A is B

    def test_prepare_experiment_identity(self):
        from cliquefinder.stats.method_comparison import prepare_experiment as A
        from cliquefinder.stats.experiment import prepare_experiment as B
        assert A is B

    def test_ols_method_identity(self):
        from cliquefinder.stats.method_comparison import OLSMethod as A
        from cliquefinder.stats.methods.ols import OLSMethod as B
        from cliquefinder.stats.methods import OLSMethod as C
        assert A is B
        assert B is C

    def test_lmm_method_identity(self):
        from cliquefinder.stats.method_comparison import LMMMethod as A
        from cliquefinder.stats.methods.lmm import LMMMethod as B
        from cliquefinder.stats.methods import LMMMethod as C
        assert A is B
        assert B is C

    def test_roast_method_identity(self):
        from cliquefinder.stats.method_comparison import ROASTMethod as A
        from cliquefinder.stats.methods.roast import ROASTMethod as B
        from cliquefinder.stats.methods import ROASTMethod as C
        assert A is B
        assert B is C

    def test_permutation_method_identity(self):
        from cliquefinder.stats.method_comparison import PermutationMethod as A
        from cliquefinder.stats.methods.permutation import PermutationMethod as B
        from cliquefinder.stats.methods import PermutationMethod as C
        assert A is B
        assert B is C

    def test_method_comparison_result_identity(self):
        from cliquefinder.stats.method_comparison import MethodComparisonResult as A
        from cliquefinder.stats.concordance import MethodComparisonResult as B
        assert A is B

    def test_concordance_function_identity(self):
        from cliquefinder.stats.method_comparison import compute_pairwise_concordance as A
        from cliquefinder.stats.concordance import compute_pairwise_concordance as B
        assert A is B

    def test_disagreements_function_identity(self):
        from cliquefinder.stats.method_comparison import identify_disagreements as A
        from cliquefinder.stats.concordance import identify_disagreements as B
        assert A is B


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestAllExports:
    """__all__ lists are complete and cover the full public API."""

    def test_method_comparison_all_complete(self):
        import cliquefinder.stats.method_comparison as mc

        expected = {
            "MethodName",
            "UnifiedCliqueResult",
            "PreparedCliqueExperiment",
            "ConcordanceMetrics",
            "MethodComparisonResult",
            "CliqueTestMethod",
            "prepare_experiment",
            "OLSMethod",
            "LMMMethod",
            "ROASTMethod",
            "PermutationMethod",
            "compute_pairwise_concordance",
            "identify_disagreements",
            "run_method_comparison",
        }
        actual = set(mc.__all__)
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_method_comparison_all_resolves(self):
        """Every name in __all__ is actually an attribute of the module."""
        import cliquefinder.stats.method_comparison as mc

        for name in mc.__all__:
            assert hasattr(mc, name), f"{name} in __all__ but not found on module"

    def test_method_comparison_types_all(self):
        import cliquefinder.stats.method_comparison_types as mct

        expected = {"MethodName", "UnifiedCliqueResult", "ConcordanceMetrics", "CliqueTestMethod"}
        actual = set(mct.__all__)
        assert actual == expected

    def test_experiment_all(self):
        import cliquefinder.stats.experiment as exp

        expected = {"PreparedCliqueExperiment", "prepare_experiment"}
        actual = set(exp.__all__)
        assert actual == expected

    def test_methods_init_all(self):
        import cliquefinder.stats.methods as methods

        expected = {"OLSMethod", "LMMMethod", "ROASTMethod", "PermutationMethod"}
        actual = set(methods.__all__)
        assert actual == expected

    def test_concordance_all(self):
        import cliquefinder.stats.concordance as conc

        expected = {"compute_pairwise_concordance", "identify_disagreements", "MethodComparisonResult"}
        actual = set(conc.__all__)
        assert actual == expected


# ---------------------------------------------------------------------------
# Structural checks: method classes and their properties
# ---------------------------------------------------------------------------


class TestMethodStructure:
    """Verify method classes have correct properties and are instantiable."""

    def test_ols_name(self):
        from cliquefinder.stats.method_comparison import OLSMethod, MethodName
        assert OLSMethod().name == MethodName.OLS

    def test_lmm_name(self):
        from cliquefinder.stats.method_comparison import LMMMethod, MethodName
        assert LMMMethod().name == MethodName.LMM

    def test_roast_msq_name(self):
        from cliquefinder.stats.method_comparison import ROASTMethod, MethodName
        assert ROASTMethod(statistic="msq").name == MethodName.ROAST_MSQ

    def test_roast_mean_name(self):
        from cliquefinder.stats.method_comparison import ROASTMethod, MethodName
        assert ROASTMethod(statistic="mean").name == MethodName.ROAST_MEAN

    def test_roast_floormean_name(self):
        from cliquefinder.stats.method_comparison import ROASTMethod, MethodName
        assert ROASTMethod(statistic="floormean").name == MethodName.ROAST_FLOORMEAN

    def test_permutation_name(self):
        from cliquefinder.stats.method_comparison import PermutationMethod, MethodName
        assert PermutationMethod().name == MethodName.PERMUTATION_COMPETITIVE

    def test_ols_has_test_method(self):
        from cliquefinder.stats.method_comparison import OLSMethod
        m = OLSMethod()
        assert hasattr(m, "test")
        assert callable(m.test)

    def test_lmm_has_test_method(self):
        from cliquefinder.stats.method_comparison import LMMMethod
        m = LMMMethod()
        assert hasattr(m, "test")
        assert callable(m.test)

    def test_roast_has_test_method(self):
        from cliquefinder.stats.method_comparison import ROASTMethod
        m = ROASTMethod()
        assert hasattr(m, "test")
        assert callable(m.test)

    def test_permutation_has_test_method(self):
        from cliquefinder.stats.method_comparison import PermutationMethod
        m = PermutationMethod()
        assert hasattr(m, "test")
        assert callable(m.test)

    def test_ols_inherits_base_linear(self):
        from cliquefinder.stats.methods.ols import OLSMethod
        from cliquefinder.stats.methods._base_linear import _BaseLinearMethod
        assert issubclass(OLSMethod, _BaseLinearMethod)

    def test_lmm_inherits_base_linear(self):
        from cliquefinder.stats.methods.lmm import LMMMethod
        from cliquefinder.stats.methods._base_linear import _BaseLinearMethod
        assert issubclass(LMMMethod, _BaseLinearMethod)

    def test_ols_default_summarization(self):
        from cliquefinder.stats.method_comparison import OLSMethod
        m = OLSMethod()
        assert m.summarization == "tmp"
        assert m.eb_moderation is True

    def test_lmm_default_summarization(self):
        from cliquefinder.stats.method_comparison import LMMMethod
        m = LMMMethod()
        assert m.summarization == "tmp"
        assert m.use_satterthwaite is True

    def test_roast_default_params(self):
        from cliquefinder.stats.method_comparison import ROASTMethod
        m = ROASTMethod()
        assert m.statistic == "msq"
        assert m.alternative == "mixed"
        assert m.n_rotations == 9999

    def test_permutation_default_params(self):
        from cliquefinder.stats.method_comparison import PermutationMethod
        m = PermutationMethod()
        assert m.n_permutations == 10000
        assert m.summarization == "tmp"


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify dataclass construction and frozen/mutable behavior."""

    def test_unified_clique_result_frozen(self):
        from cliquefinder.stats.method_comparison import UnifiedCliqueResult, MethodName

        r = UnifiedCliqueResult(
            clique_id="TP53",
            method=MethodName.OLS,
            effect_size=1.5,
            effect_size_se=0.3,
            p_value=0.001,
            statistic_value=5.0,
            statistic_type="t",
            degrees_of_freedom=45.0,
            n_proteins=25,
            n_proteins_found=20,
        )
        assert r.is_valid
        assert r.is_significant
        assert r.coverage_fraction == pytest.approx(0.8)

        with pytest.raises(AttributeError):
            r.p_value = 0.5  # type: ignore[misc]

    def test_unified_clique_result_to_dict(self):
        from cliquefinder.stats.method_comparison import UnifiedCliqueResult, MethodName

        r = UnifiedCliqueResult(
            clique_id="TP53",
            method=MethodName.OLS,
            effect_size=1.5,
            effect_size_se=0.3,
            p_value=0.001,
            statistic_value=5.0,
            statistic_type="t",
            degrees_of_freedom=45.0,
            n_proteins=25,
            n_proteins_found=20,
            method_metadata={"foo": "bar"},
        )
        d = r.to_dict()
        assert d["method"] == "ols"
        assert d["meta_foo"] == "bar"

    def test_concordance_metrics_frozen(self):
        from cliquefinder.stats.method_comparison import ConcordanceMetrics, MethodName

        m = ConcordanceMetrics(
            method_a=MethodName.OLS,
            method_b=MethodName.ROAST_MSQ,
            n_cliques_compared=100,
            spearman_rho=0.85,
            spearman_pvalue=1e-10,
            threshold=0.05,
            n_both_significant=15,
            n_both_nonsignificant=70,
            n_a_only=5,
            n_b_only=10,
            cohen_kappa=0.65,
            effect_pearson_r=0.72,
            effect_rmse=0.45,
            direction_agreement_frac=0.92,
        )
        assert m.jaccard_index == pytest.approx(0.5)
        assert m.agreement_rate == pytest.approx(0.85)
        assert m.to_dict()["method_a"] == "ols"
        assert "Concordance:" in m.summary()

        with pytest.raises(AttributeError):
            m.spearman_rho = 0.5  # type: ignore[misc]

    def test_methodname_enum_values(self):
        from cliquefinder.stats.method_comparison import MethodName

        values = {m.value for m in MethodName}
        expected = {
            "ols", "lmm",
            "roast_msq", "roast_mean", "roast_floormean",
            "permutation_competitive", "permutation_rotation",
        }
        assert values == expected


# ---------------------------------------------------------------------------
# run_method_comparison is callable
# ---------------------------------------------------------------------------


class TestRunMethodComparison:
    """run_method_comparison function is available and callable."""

    def test_is_callable(self):
        from cliquefinder.stats.method_comparison import run_method_comparison
        assert callable(run_method_comparison)

    def test_is_a_function(self):
        from cliquefinder.stats.method_comparison import run_method_comparison
        assert isinstance(run_method_comparison, types.FunctionType)

    def test_has_correct_signature_params(self):
        import inspect
        from cliquefinder.stats.method_comparison import run_method_comparison

        sig = inspect.signature(run_method_comparison)
        param_names = list(sig.parameters.keys())
        # Verify key parameters exist (not all, just critical ones)
        assert "data" in param_names
        assert "feature_ids" in param_names
        assert "sample_metadata" in param_names
        assert "cliques" in param_names
        assert "condition_column" in param_names
        assert "contrast" in param_names
        assert "subject_column" in param_names
        assert "methods" in param_names
        assert "verbose" in param_names
        assert "precomputed_symbol_map" in param_names


# ---------------------------------------------------------------------------
# Module docstrings present
# ---------------------------------------------------------------------------


class TestModuleDocstrings:
    """Each new module should have a module-level docstring."""

    @pytest.mark.parametrize("module_path", [
        "cliquefinder.stats.method_comparison",
        "cliquefinder.stats.method_comparison_types",
        "cliquefinder.stats.experiment",
        "cliquefinder.stats.concordance",
        "cliquefinder.stats.methods",
        "cliquefinder.stats.methods.ols",
        "cliquefinder.stats.methods.lmm",
        "cliquefinder.stats.methods.roast",
        "cliquefinder.stats.methods.permutation",
        "cliquefinder.stats.methods._base_linear",
    ])
    def test_module_has_docstring(self, module_path):
        mod = importlib.import_module(module_path)
        assert mod.__doc__ is not None, f"{module_path} has no module docstring"
        assert len(mod.__doc__.strip()) > 20, f"{module_path} docstring too short"


# ---------------------------------------------------------------------------
# Future-import consistency
# ---------------------------------------------------------------------------


class TestFutureAnnotations:
    """All new modules should have 'from __future__ import annotations'."""

    @pytest.mark.parametrize("module_path", [
        "cliquefinder.stats.method_comparison",
        "cliquefinder.stats.method_comparison_types",
        "cliquefinder.stats.experiment",
        "cliquefinder.stats.concordance",
        "cliquefinder.stats.methods",
        "cliquefinder.stats.methods.ols",
        "cliquefinder.stats.methods.lmm",
        "cliquefinder.stats.methods.roast",
        "cliquefinder.stats.methods.permutation",
        "cliquefinder.stats.methods._base_linear",
    ])
    def test_future_annotations(self, module_path):
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "annotations") or "annotations" in getattr(mod, "__annotations__", {}) or True
        # More reliable: check the source file
        source_file = mod.__file__
        if source_file:
            with open(source_file) as f:
                content = f.read()
            assert "from __future__ import annotations" in content, (
                f"{module_path} missing 'from __future__ import annotations'"
            )
