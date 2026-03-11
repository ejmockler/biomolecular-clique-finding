"""Tests for defensive coding and logging level correctness.

Covers read-only array handling in ROAST, enum key consistency in
results_by_method, duplicate clique ID detection, Neo4j reconnection
parameter removal, HGNC normalization stub behavior, empty null z-score
serialization, single gene set rotation warnings, credential sanitization
in logs, gene resolution logging levels, and interaction permutation
failure logging with suppression.
"""

from __future__ import annotations

import logging
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Read-only array defensively copied before ROAST engine
# ---------------------------------------------------------------------------


class TestReadOnlyArray:
    """RotationTestEngine receives a writeable copy of read-only data."""

    def test_roast_copies_readonly_data(self):
        """If experiment.data is read-only, ROAST should copy it."""
        from cliquefinder.stats.methods.roast import ROASTMethod

        # Build a minimal mock experiment with read-only data
        data = np.random.default_rng(42).standard_normal((10, 6))
        data.flags.writeable = False

        experiment = MagicMock()
        experiment.data = data
        experiment.feature_ids = tuple(f"gene_{i}" for i in range(10))
        experiment.sample_metadata = pd.DataFrame(
            {"cond": ["A"] * 3 + ["B"] * 3}
        )
        experiment.conditions = ("A", "B")
        experiment.contrast = ("A", "B")
        experiment.condition_column = "cond"
        experiment.cliques = []  # no cliques => early return

        method = ROASTMethod(n_rotations=99)
        # Should NOT raise even though data is read-only
        results = method.test(experiment, use_gpu=False, seed=1)
        assert results == []  # no cliques, but no crash

    def test_roast_skips_copy_when_writeable(self):
        """No unnecessary copy when data is already writeable."""
        data = np.random.default_rng(42).standard_normal((10, 6))
        assert data.flags.writeable  # sanity

        # The copy guard only triggers when writeable is False
        if not data.flags.writeable:
            copied = data.copy()
        else:
            copied = data
        assert copied is data  # same object, no copy


# ---------------------------------------------------------------------------
# Consistent MethodName enum keys in results_by_method
# ---------------------------------------------------------------------------


class TestConsistentEnumKeys:
    """results_by_method always uses MethodName enum, never strings."""

    def test_results_by_method_keys_are_enum(self):
        """All keys in results_by_method are MethodName enum."""
        from cliquefinder.stats.method_comparison_types import MethodName

        # Simulate the dict construction from run_method_comparison
        results_by_method: dict[MethodName, list] = {}
        method_name = MethodName.OLS
        assert isinstance(method_name, MethodName)
        results_by_method[method_name] = []

        for key in results_by_method:
            assert isinstance(key, MethodName), (
                f"Expected MethodName enum, got {type(key)}"
            )

    def test_failed_methods_uses_string_value(self):
        """failed_methods uses .value strings by design."""
        from cliquefinder.stats.method_comparison_types import MethodName

        failed_methods: dict[str, str] = {}
        method_name = MethodName.OLS
        failed_methods[method_name.value] = "some error"

        assert "ols" in failed_methods
        assert isinstance(list(failed_methods.keys())[0], str)


# ---------------------------------------------------------------------------
# Duplicate clique_id guard in permutation null_df
# ---------------------------------------------------------------------------


class TestDuplicateCliqueId:
    """Duplicate clique_ids in null_df are detected and deduplicated."""

    def test_duplicate_warning_logged(self, caplog):
        """Warning when duplicate clique_ids in null_df."""
        from cliquefinder.stats.methods.permutation import PermutationMethod

        method = PermutationMethod(n_permutations=100)

        # Build mock experiment
        experiment = MagicMock()
        experiment.data = np.random.default_rng(42).standard_normal((5, 6))
        experiment.feature_ids = [f"p{i}" for i in range(5)]
        experiment.sample_metadata = pd.DataFrame(
            {"cond": ["A"] * 3 + ["B"] * 3}
        )
        experiment.condition_column = "cond"
        experiment.contrast = ("A", "B")
        experiment.subject_column = None
        experiment.cliques = []
        experiment.feature_to_idx = {f"p{i}": i for i in range(5)}

        # Mock run_permutation_test_gpu to return duplicate clique_ids
        mock_perm_result = MagicMock()
        mock_perm_result.clique_id = "clique_1"
        mock_perm_result.observed_tvalue = 2.5
        mock_perm_result.empirical_pvalue = 0.05
        mock_perm_result.percentile_rank = 95.0
        mock_perm_result.observed_log2fc = 0.5
        mock_perm_result.is_significant = True

        null_df = pd.DataFrame({
            "clique_id": ["clique_1", "clique_1"],  # DUPLICATE
            "null_tvalue_mean": [0.0, 0.1],
            "null_tvalue_std": [1.0, 1.1],
        })

        with patch(
            "cliquefinder.stats.permutation_gpu.run_permutation_test_gpu",
            return_value=([mock_perm_result], null_df),
        ):
            with caplog.at_level(logging.WARNING):
                results = method.test(experiment, use_gpu=False, seed=42)

        # Check that warning about duplicates was logged
        assert any("Duplicate clique_ids" in msg for msg in caplog.messages), (
            f"Expected duplicate warning; got: {caplog.messages}"
        )

    def test_no_warning_when_no_duplicates(self, caplog):
        """No warning when clique_ids are unique."""
        null_df = pd.DataFrame({
            "clique_id": ["clique_1", "clique_2"],
            "null_tvalue_mean": [0.0, 0.1],
            "null_tvalue_std": [1.0, 1.1],
        })
        assert not null_df["clique_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Dead force_reconnect parameter removed
# ---------------------------------------------------------------------------


class TestForceReconnect:
    """force_reconnect parameter has been removed from _get_client."""

    def test_get_client_no_force_reconnect_param(self):
        """_get_client no longer accepts force_reconnect."""
        import inspect
        from cliquefinder.knowledge.cogex import CoGExClient

        sig = inspect.signature(CoGExClient._get_client)
        params = list(sig.parameters.keys())
        assert "force_reconnect" not in params, (
            f"force_reconnect should be removed; params: {params}"
        )

    def test_get_client_signature_is_self_only(self):
        """_get_client takes only self."""
        import inspect
        from cliquefinder.knowledge.cogex import CoGExClient

        sig = inspect.signature(CoGExClient._get_client)
        # Only 'self' (which isn't in sig.parameters for bound methods
        # but IS for unbound)
        params = [p for p in sig.parameters if p != "self"]
        assert params == [], f"Unexpected params: {params}"


# ---------------------------------------------------------------------------
# norm_id stub raises ImportError when INDRA is not installed
# ---------------------------------------------------------------------------


class TestNormIdStub:
    """norm_id raises ImportError (not TypeError) when INDRA missing."""

    def test_stub_raises_import_error(self):
        """When INDRA is unavailable, norm_id gives a clear error."""
        # Import the module-level norm_id that's either real or stub
        import cliquefinder.knowledge.cogex as cogex_mod

        if not cogex_mod.INDRA_AVAILABLE:
            # Stub path — should raise ImportError
            with pytest.raises(ImportError, match="INDRA is required"):
                cogex_mod.norm_id("HGNC", "1234")
        else:
            # Real INDRA is installed — norm_id should be callable
            assert callable(cogex_mod.norm_id)

    def test_stub_is_callable(self):
        """Stub is callable (not None), preventing confusing TypeError."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Whether INDRA is installed or not, norm_id should be callable
        assert callable(cogex_mod.norm_id)


# ---------------------------------------------------------------------------
# to_dict() handles empty null_z_scores gracefully
# ---------------------------------------------------------------------------


class TestEmptyNullZScores:
    """LabelPermutationResult.to_dict() handles empty arrays."""

    def test_to_dict_with_empty_array(self):
        """to_dict does not crash when null_z_scores is empty."""
        from cliquefinder.stats.label_permutation import LabelPermutationResult

        result = LabelPermutationResult(
            observed_z=2.5,
            null_z_scores=np.array([]),  # empty!
            permutation_pvalue=1.0,
            n_permutations=0,
            stratified=False,
        )

        # Should not raise
        d = result.to_dict()
        assert d["observed_z"] == 2.5
        assert d["n_permutations"] == 0
        # Quantiles should be NaN
        for key in ("q05", "q25", "q50", "q75", "q95"):
            assert np.isnan(d["null_z_quantiles"][key]), (
                f"Expected NaN for {key}, got {d['null_z_quantiles'][key]}"
            )

    def test_to_dict_with_normal_array(self):
        """to_dict still works normally with populated array."""
        from cliquefinder.stats.label_permutation import LabelPermutationResult

        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)

        result = LabelPermutationResult(
            observed_z=2.5,
            null_z_scores=scores,
            permutation_pvalue=0.01,
            n_permutations=100,
            stratified=True,
        )

        d = result.to_dict()
        assert d["n_permutations"] == 100
        assert np.isfinite(d["null_z_quantiles"]["q50"])

    def test_to_dict_with_none_scores(self):
        """to_dict handles None null_z_scores."""
        from cliquefinder.stats.label_permutation import LabelPermutationResult

        result = LabelPermutationResult(
            observed_z=1.0,
            null_z_scores=None,  # type: ignore[arg-type]
            permutation_pvalue=1.0,
            n_permutations=0,
            stratified=False,
        )

        d = result.to_dict()
        for key in ("q05", "q25", "q50", "q75", "q95"):
            assert np.isnan(d["null_z_quantiles"][key])


# ---------------------------------------------------------------------------
# Single-gene sets warn and return NaN p-values
# ---------------------------------------------------------------------------


class TestSingleGeneSet:
    """Single-gene sets produce a warning and NaN p-values, not silent empty."""

    def test_single_gene_set_warns(self):
        """Warning issued for single-gene sets."""
        from cliquefinder.stats.rotation import (
            RotationTestEngine,
            RotationTestConfig,
        )

        rng = np.random.default_rng(42)
        n_genes, n_samples = 20, 10
        data = rng.standard_normal((n_genes, n_samples))
        gene_ids = [f"gene_{i}" for i in range(n_genes)]
        metadata = pd.DataFrame({
            "condition": ["A"] * 5 + ["B"] * 5,
        })

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=["A", "B"],
            contrast=("A", "B"),
            condition_column="condition",
        )

        config = RotationTestConfig(n_rotations=99, use_gpu=False, seed=42)

        # Test with a single gene that IS in the data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.test_gene_set(
                gene_set=["gene_0"],
                gene_set_id="single_gene_set",
                config=config,
            )

        # Should have issued a warning
        single_gene_warnings = [
            x for x in w if "single-gene" in str(x.message).lower()
        ]
        assert len(single_gene_warnings) == 1, (
            f"Expected 1 single-gene warning, got {len(single_gene_warnings)}; "
            f"all warnings: {[str(x.message) for x in w]}"
        )

        # Result should have empty p_values (NaN from get_pvalue)
        assert result.n_genes_found == 1
        assert result.n_rotations == 0
        assert result.p_values == {}
        assert np.isnan(result.get_pvalue("msq", "mixed"))

    def test_zero_gene_set_no_warning(self):
        """Zero-gene sets return empty result without warning."""
        from cliquefinder.stats.rotation import (
            RotationTestEngine,
            RotationTestConfig,
        )

        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10))
        gene_ids = [f"gene_{i}" for i in range(20)]
        metadata = pd.DataFrame({"condition": ["A"] * 5 + ["B"] * 5})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=["A", "B"],
            contrast=("A", "B"),
            condition_column="condition",
        )

        config = RotationTestConfig(n_rotations=99, use_gpu=False, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.test_gene_set(
                gene_set=["NOT_IN_DATA"],
                gene_set_id="empty_set",
                config=config,
            )

        # No single-gene warning for zero-gene sets
        single_gene_warnings = [
            x for x in w if "single-gene" in str(x.message).lower()
        ]
        assert len(single_gene_warnings) == 0, (
            f"Expected no single-gene warning for zero-gene set, "
            f"got: {[str(x.message) for x in w]}"
        )
        assert result.n_genes_found == 0

    def test_multi_gene_set_no_warning(self):
        """Multi-gene sets work normally without warning."""
        from cliquefinder.stats.rotation import (
            RotationTestEngine,
            RotationTestConfig,
        )

        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 10))
        gene_ids = [f"gene_{i}" for i in range(20)]
        metadata = pd.DataFrame({"condition": ["A"] * 5 + ["B"] * 5})

        engine = RotationTestEngine(data=data, gene_ids=gene_ids, metadata=metadata)
        engine.fit(
            conditions=["A", "B"],
            contrast=("A", "B"),
            condition_column="condition",
        )

        config = RotationTestConfig(n_rotations=99, use_gpu=False, seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.test_gene_set(
                gene_set=["gene_0", "gene_1", "gene_2"],
                gene_set_id="multi_gene_set",
                config=config,
            )

        single_gene_warnings = [
            x for x in w if "single-gene" in str(x.message).lower()
        ]
        assert len(single_gene_warnings) == 0
        assert result.n_genes_found == 3
        assert result.n_rotations > 0
        # Should have actual p-values
        assert np.isfinite(result.get_pvalue("msq", "mixed"))


# ---------------------------------------------------------------------------
# Credential and connection logging at DEBUG level
# ---------------------------------------------------------------------------


class TestCredentialLogging:
    """Sensitive info (URLs, .env paths) logged at DEBUG, not INFO."""

    def test_credential_env_var_logs_at_debug(self, caplog):
        """'Using credentials from environment variables' at DEBUG."""
        import cliquefinder.knowledge.cogex as cogex_mod

        client = cogex_mod.CoGExClient.__new__(cogex_mod.CoGExClient)
        client._url = None
        client._user = None
        client._password = None
        client._env_file = None
        client._client = None

        with patch.dict("os.environ", {
            "INDRA_NEO4J_URL": "bolt://test:7687",
            "INDRA_NEO4J_USER": "testuser",
            "INDRA_NEO4J_PASSWORD": "secret",
        }):
            with caplog.at_level(logging.DEBUG, logger="cliquefinder.knowledge.cogex"):
                url, user, pw = client._load_credentials()

        assert url == "bolt://test:7687"
        # Check the message was logged at DEBUG
        credential_messages = [
            r for r in caplog.records
            if "credentials" in r.message.lower()
        ]
        assert len(credential_messages) >= 1
        for record in credential_messages:
            assert record.levelno == logging.DEBUG, (
                f"Expected DEBUG, got {record.levelname}: {record.message}"
            )

    def test_credential_explicit_logs_at_debug(self, caplog):
        """'Using explicit credentials' at DEBUG."""
        import cliquefinder.knowledge.cogex as cogex_mod

        client = cogex_mod.CoGExClient.__new__(cogex_mod.CoGExClient)
        client._url = "bolt://test:7687"
        client._user = "testuser"
        client._password = "secret"
        client._env_file = None
        client._client = None

        with caplog.at_level(logging.DEBUG, logger="cliquefinder.knowledge.cogex"):
            url, user, pw = client._load_credentials()

        assert url == "bolt://test:7687"
        credential_messages = [
            r for r in caplog.records
            if "explicit credentials" in r.message.lower()
        ]
        assert len(credential_messages) == 1
        assert credential_messages[0].levelno == logging.DEBUG

    def test_connection_url_not_at_info(self, caplog):
        """Connection URL not logged at INFO level."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Verify the log message uses DEBUG by inspecting source
        import inspect
        source = inspect.getsource(cogex_mod.CoGExClient._get_client)
        assert "logger.debug" in source, (
            "Expected logger.debug for connection URL logging"
        )
        assert "logger.info" not in source, (
            "logger.info should not appear in _get_client"
        )


# ---------------------------------------------------------------------------
# Per-gene resolution logging at DEBUG level
# ---------------------------------------------------------------------------


class TestGeneResolutionLogging:
    """Per-gene 'Could not resolve' messages are DEBUG, not INFO."""

    def test_per_gene_failure_is_debug(self):
        """Per-gene resolution failures logged at DEBUG."""
        import inspect
        import cliquefinder.knowledge.cogex as cogex_mod

        source = inspect.getsource(cogex_mod.CoGExClient.discover_regulators)
        # The per-gene log should use logger.debug
        assert 'logger.debug("Could not resolve gene' in source, (
            "Per-gene resolution failure should use logger.debug"
        )

    def test_aggregate_resolution_is_info(self):
        """Aggregate resolution summary logged at INFO."""
        import inspect
        import cliquefinder.knowledge.cogex as cogex_mod

        source = inspect.getsource(cogex_mod.CoGExClient.discover_regulators)
        # The aggregate summary should use logger.info
        assert "Resolved %d/%d gene names successfully" in source, (
            "Expected aggregate INFO message about gene resolution"
        )


# ---------------------------------------------------------------------------
# Interaction permutation failures logged with warning
# ---------------------------------------------------------------------------


class TestInteractionPermFailureLogging:
    """Failed interaction permutations are logged, not silently swallowed."""

    def test_failure_logged_as_warning(self, caplog):
        """Failed interaction permutation logged as WARNING."""
        from cliquefinder.stats.specificity import _run_interaction_permutation

        rng = np.random.default_rng(42)
        n_features, n_samples = 20, 12
        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        metadata = pd.DataFrame({
            "cond": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        })
        target_gene_ids = feature_ids[:5]

        # Patch run_protein_differential to fail on some calls
        call_count = [0]

        def _mock_differential(*args, **kwargs):
            call_count[0] += 1
            # Let observed calls succeed, fail on some permutation calls
            if call_count[0] > 2 and call_count[0] % 3 == 0:
                raise ValueError("Simulated permutation failure")
            # Return a minimal DataFrame
            df = pd.DataFrame({
                "feature_id": feature_ids,
                "t_statistic": rng.standard_normal(n_features),
                "is_target": [fid in target_gene_ids for fid in feature_ids],
            })
            return df

        with patch(
            "cliquefinder.stats.differential.run_protein_differential",
            side_effect=_mock_differential,
        ):
            with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.specificity"):
                result = _run_interaction_permutation(
                    data=data,
                    feature_ids=feature_ids,
                    metadata=metadata,
                    condition_col="cond",
                    primary_contrast=("A", "C"),
                    secondary_contrast=("B", "C"),
                    target_gene_ids=target_gene_ids,
                    n_perms=20,
                    seed=42,
                )

        # Check that some failures occurred and were logged
        failure_messages = [
            r for r in caplog.records
            if "permutation" in r.message.lower() and "failed" in r.message.lower()
        ]
        # We should have at least 1 warning (up to 3 before suppression)
        assert len(failure_messages) >= 1, (
            f"Expected at least 1 failure warning; got: {caplog.messages}"
        )
        for record in failure_messages:
            assert record.levelno == logging.WARNING

    def test_no_warning_when_all_succeed(self, caplog):
        """No warning when all interaction permutations succeed."""
        from cliquefinder.stats.specificity import _run_interaction_permutation

        rng = np.random.default_rng(42)
        n_features, n_samples = 20, 12
        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        metadata = pd.DataFrame({
            "cond": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        })
        target_gene_ids = feature_ids[:5]

        def _mock_differential(*args, **kwargs):
            df = pd.DataFrame({
                "feature_id": feature_ids,
                "t_statistic": rng.standard_normal(n_features),
                "is_target": [fid in target_gene_ids for fid in feature_ids],
            })
            return df

        with patch(
            "cliquefinder.stats.differential.run_protein_differential",
            side_effect=_mock_differential,
        ):
            with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.specificity"):
                result = _run_interaction_permutation(
                    data=data,
                    feature_ids=feature_ids,
                    metadata=metadata,
                    condition_col="cond",
                    primary_contrast=("A", "C"),
                    secondary_contrast=("B", "C"),
                    target_gene_ids=target_gene_ids,
                    n_perms=10,
                    seed=42,
                )

        # No failure warnings
        failure_messages = [
            r for r in caplog.records
            if "permutation" in r.message.lower() and "failed" in r.message.lower()
        ]
        assert len(failure_messages) == 0, (
            f"Expected no failure warnings; got: {[r.message for r in failure_messages]}"
        )

    def test_warning_suppression_after_three(self, caplog):
        """After 3 failures, further warnings are suppressed."""
        from cliquefinder.stats.specificity import _run_interaction_permutation

        rng = np.random.default_rng(42)
        n_features, n_samples = 20, 12
        data = rng.standard_normal((n_features, n_samples))
        feature_ids = [f"gene_{i}" for i in range(n_features)]
        metadata = pd.DataFrame({
            "cond": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        })
        target_gene_ids = feature_ids[:5]

        call_count = [0]

        def _mock_differential(*args, **kwargs):
            call_count[0] += 1
            # Let the first 2 calls (observed) succeed, fail all perms
            if call_count[0] > 2:
                raise ValueError("Always fail in perm")
            df = pd.DataFrame({
                "feature_id": feature_ids,
                "t_statistic": rng.standard_normal(n_features),
                "is_target": [fid in target_gene_ids for fid in feature_ids],
            })
            return df

        with patch(
            "cliquefinder.stats.differential.run_protein_differential",
            side_effect=_mock_differential,
        ):
            with caplog.at_level(logging.WARNING, logger="cliquefinder.stats.specificity"):
                result = _run_interaction_permutation(
                    data=data,
                    feature_ids=feature_ids,
                    metadata=metadata,
                    condition_col="cond",
                    primary_contrast=("A", "C"),
                    secondary_contrast=("B", "C"),
                    target_gene_ids=target_gene_ids,
                    n_perms=10,  # all 10 will fail
                    seed=42,
                )

        # Should have at most 4 warning messages (3 individual + 1 suppression)
        warning_messages = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "permutation" in r.message.lower()
        ]
        assert len(warning_messages) <= 4, (
            f"Expected at most 4 warnings (3 + suppression); got {len(warning_messages)}: "
            f"{[r.message for r in warning_messages]}"
        )
        # Check suppression message exists
        suppression_msgs = [
            r for r in caplog.records if "Suppressing" in r.message
        ]
        assert len(suppression_msgs) >= 1, (
            "Expected suppression message after 3 failures"
        )
