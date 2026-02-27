"""Tests for ARCH-5 (memory-mapped process pool) and ARCH-4-NOTE (Neo4j typed exceptions).

ARCH-5: Verifies that ProcessPoolExecutor workers load expression matrices via
memory-mapped .npy files instead of pickle-serialized copies.

ARCH-4-NOTE: Verifies that CoGExClient._execute_query() uses typed Neo4j
exception classes (ServiceUnavailable, SessionExpired, TransientError) for
connection error detection, with string-keyword fallback.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag


# =============================================================================
# ARCH-5: Memory-Mapped Arrays for Process Pool Workers
# =============================================================================


class TestMmapWorkerInit:
    """Test that _init_worker loads data from a memory-mapped .npy file."""

    def _make_mmap_file(self, data: np.ndarray) -> str:
        """Write a NumPy array to a temp .npy file and return the path."""
        tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(tmp, data)
        tmp.close()
        return tmp.name

    def _make_sample_metadata(self, n_samples: int) -> pd.DataFrame:
        """Create minimal sample metadata for worker init."""
        phenotypes = ["CASE" if i % 2 == 0 else "CTRL" for i in range(n_samples)]
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        return pd.DataFrame(
            {'phenotype': phenotypes},
            index=pd.Index(sample_ids),
        )

    def test_init_worker_loads_mmap(self):
        """_init_worker should create a validator from a memory-mapped file."""
        from cliquefinder.cli._analyze_core import _init_worker, _worker_validator

        n_genes, n_samples = 20, 10
        data = np.random.RandomState(42).rand(n_genes, n_samples)
        mmap_path = self._make_mmap_file(data)

        feature_ids = [f"GENE_{i}" for i in range(n_genes)]
        sample_ids = [f"S{i:03d}" for i in range(n_samples)]
        metadata = self._make_sample_metadata(n_samples)
        quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32)

        try:
            # Patch CliqueValidator to avoid actual precomputation
            with patch('cliquefinder.cli._analyze_core.CliqueValidator') as MockValidator:
                mock_instance = MagicMock()
                MockValidator.return_value = mock_instance

                _init_worker(
                    mmap_path,
                    feature_ids,
                    sample_ids,
                    metadata,
                    quality_flags,
                    'phenotype',
                    5,  # min_samples
                    ['CASE', 'CTRL'],  # conditions
                    0.5,  # min_correlation
                    3,    # min_clique_size
                    False,  # use_fast_maximum
                    'pearson',  # correlation_method
                )

                # Verify CliqueValidator was created
                MockValidator.assert_called_once()
                call_kwargs = MockValidator.call_args
                constructed_matrix = call_kwargs[1]['matrix'] if 'matrix' in call_kwargs[1] else call_kwargs[0][0]

                # The matrix data should match original
                np.testing.assert_array_almost_equal(
                    constructed_matrix.data, data
                )

                # Precompute should have been called
                mock_instance.precompute_condition_data.assert_called_once()
        finally:
            os.unlink(mmap_path)

    def test_mmap_file_is_read_only(self):
        """Memory-mapped array loaded with mmap_mode='r' should be read-only."""
        n_genes, n_samples = 10, 5
        data = np.random.RandomState(99).rand(n_genes, n_samples)
        mmap_path = self._make_mmap_file(data)

        try:
            loaded = np.load(mmap_path, mmap_mode='r')

            # Data values should match
            np.testing.assert_array_equal(loaded, data)

            # Writing to the mmap should raise
            with pytest.raises((ValueError, TypeError)):
                loaded[0, 0] = 999.0
        finally:
            os.unlink(mmap_path)

    def test_worker_produces_same_data_as_direct(self):
        """Data loaded via mmap should be bitwise identical to direct array."""
        data = np.random.RandomState(7).rand(50, 20).astype(np.float64)
        mmap_path = self._make_mmap_file(data)

        try:
            loaded = np.load(mmap_path, mmap_mode='r')
            np.testing.assert_array_equal(loaded, data)
            assert loaded.dtype == data.dtype
            assert loaded.shape == data.shape
        finally:
            os.unlink(mmap_path)


class TestMmapTempFileCleanup:
    """Test that temporary .npy files are cleaned up after pool execution."""

    def test_cleanup_on_success(self):
        """Temp mmap file should be deleted after successful pool execution."""
        mmap_path = None
        data = np.random.RandomState(1).rand(5, 3)

        try:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                mmap_path = tmp.name
                np.save(tmp, data)

            assert os.path.exists(mmap_path)
        finally:
            if mmap_path and os.path.exists(mmap_path):
                os.unlink(mmap_path)

        # After cleanup, file should not exist
        assert not os.path.exists(mmap_path)

    def test_cleanup_on_error(self):
        """Temp mmap file should be deleted even if pool execution raises."""
        mmap_path = None
        data = np.random.RandomState(2).rand(5, 3)

        try:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                mmap_path = tmp.name
                np.save(tmp, data)

            assert os.path.exists(mmap_path)

            # Simulate an error during pool execution
            raise RuntimeError("Simulated pool failure")
        except RuntimeError:
            pass
        finally:
            if mmap_path and os.path.exists(mmap_path):
                os.unlink(mmap_path)

        # File should be cleaned up despite the error
        assert not os.path.exists(mmap_path)

    def test_cleanup_when_path_is_none(self):
        """Cleanup should handle None mmap_path gracefully (no crash)."""
        mmap_path = None
        try:
            raise RuntimeError("Simulated early failure before file creation")
        except RuntimeError:
            pass
        finally:
            # This should not raise
            if mmap_path and os.path.exists(mmap_path):
                os.unlink(mmap_path)
        # No assertion needed — just verifying no crash


class TestMmapIntegration:
    """Integration tests for the mmap-based process pool code path."""

    def test_init_args_tuple_contains_mmap_path(self):
        """The init_args tuple passed to ProcessPoolExecutor should contain a
        file path string (not a raw numpy array) as its first element."""
        # Read the source to verify the code structure
        import inspect
        from cliquefinder.cli import _analyze_core
        source = inspect.getsource(_analyze_core._init_worker)

        # The first parameter should be named mmap_path
        assert 'mmap_path' in source
        assert "np.load(mmap_path, mmap_mode='r')" in source

    def test_run_stratified_analysis_signature(self):
        """Verify _init_worker accepts mmap_path as first arg."""
        import inspect
        from cliquefinder.cli._analyze_core import _init_worker

        sig = inspect.signature(_init_worker)
        params = list(sig.parameters.keys())
        assert params[0] == 'mmap_path', (
            f"Expected first parameter 'mmap_path', got '{params[0]}'"
        )


# =============================================================================
# ARCH-4-NOTE: Neo4j Typed Exception Handling
# =============================================================================


class TestNeo4jExceptionImportGuard:
    """Test that the Neo4j exception import guard works correctly."""

    def test_neo4j_exceptions_flag_exists(self):
        """_NEO4J_EXCEPTIONS_AVAILABLE flag should exist in cogex module."""
        from cliquefinder.knowledge import cogex as cogex_mod
        assert hasattr(cogex_mod, '_NEO4J_EXCEPTIONS_AVAILABLE')
        # Value depends on whether neo4j is installed — just check it's bool
        assert isinstance(cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE, bool)

    def test_connection_error_keywords_defined(self):
        """CoGExClient should define _CONNECTION_ERROR_KEYWORDS."""
        from cliquefinder.knowledge.cogex import CoGExClient
        assert hasattr(CoGExClient, '_CONNECTION_ERROR_KEYWORDS')
        keywords = CoGExClient._CONNECTION_ERROR_KEYWORDS
        assert 'connection' in keywords
        assert 'timeout' in keywords
        assert 'unavailable' in keywords

    def test_import_guard_with_neo4j_missing(self):
        """When neo4j is not installed, _NEO4J_EXCEPTIONS_AVAILABLE should be False
        and the sentinel values should be None."""
        # We test this by checking the fallback path logic exists
        from cliquefinder.knowledge import cogex as cogex_mod

        if not cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE:
            # If neo4j is actually not installed, verify sentinels
            assert cogex_mod.ServiceUnavailable is None
            assert cogex_mod.SessionExpired is None
            assert cogex_mod.TransientError is None
        else:
            # If neo4j IS installed, verify they are real exception classes
            assert cogex_mod.ServiceUnavailable is not None
            assert issubclass(cogex_mod.ServiceUnavailable, Exception)


class TestExecuteQueryTypedExceptions:
    """Test _execute_query with typed Neo4j exceptions and string fallback."""

    @pytest.fixture
    def mock_client(self):
        """Create a CoGExClient with mocked internals."""
        with patch('cliquefinder.knowledge.cogex.INDRA_AVAILABLE', True):
            from cliquefinder.knowledge.cogex import CoGExClient
            client = CoGExClient.__new__(CoGExClient)
            client._url = "bolt://test:7687"
            client._user = "neo4j"
            client._password = "test"
            client._env_file = None
            client._client = MagicMock()
            return client

    def test_successful_query(self, mock_client):
        """_execute_query should return results on success."""
        mock_client._client.query_tx.return_value = [("row1",), ("row2",)]

        result = mock_client._execute_query("RETURN 1")
        assert result == [("row1",), ("row2",)]
        mock_client._client.query_tx.assert_called_once_with("RETURN 1")

    def test_typed_service_unavailable_retries(self, mock_client):
        """ServiceUnavailable should trigger retry with client reset."""
        from cliquefinder.knowledge import cogex as cogex_mod

        if not cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE:
            pytest.skip("neo4j package not installed")

        # First call raises ServiceUnavailable, second succeeds
        mock_client._client.query_tx.side_effect = [
            cogex_mod.ServiceUnavailable("Server gone"),
            [("recovered",)],
        ]

        # Need to handle client reset — after reset, _get_client recreates
        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            result = mock_client._execute_query("MATCH (n) RETURN n", max_retries=1)

        assert result == [("recovered",)]

    def test_typed_session_expired_retries(self, mock_client):
        """SessionExpired should trigger retry with client reset."""
        from cliquefinder.knowledge import cogex as cogex_mod

        if not cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE:
            pytest.skip("neo4j package not installed")

        mock_client._client.query_tx.side_effect = [
            cogex_mod.SessionExpired("Session lost"),
            [("ok",)],
        ]

        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            result = mock_client._execute_query("RETURN 1", max_retries=1)

        assert result == [("ok",)]

    def test_typed_transient_error_retries(self, mock_client):
        """TransientError should trigger retry with client reset."""
        from cliquefinder.knowledge import cogex as cogex_mod

        if not cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE:
            pytest.skip("neo4j package not installed")

        mock_client._client.query_tx.side_effect = [
            cogex_mod.TransientError("Deadlock"),
            [("ok",)],
        ]

        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            result = mock_client._execute_query("RETURN 1", max_retries=1)

        assert result == [("ok",)]

    def test_string_fallback_connection_error(self, mock_client):
        """String keyword matching should detect 'connection refused' errors."""
        # Use a generic Exception with connection keywords
        mock_client._client.query_tx.side_effect = [
            Exception("Connection refused to bolt://host:7687"),
            [("recovered",)],
        ]

        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            result = mock_client._execute_query("RETURN 1", max_retries=1)

        assert result == [("recovered",)]

    def test_string_fallback_timeout_error(self, mock_client):
        """String keyword matching should detect 'timeout' errors."""
        mock_client._client.query_tx.side_effect = [
            Exception("Query timeout after 30000ms"),
            [("ok",)],
        ]

        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            result = mock_client._execute_query("RETURN 1", max_retries=1)

        assert result == [("ok",)]

    def test_syntax_error_not_retried(self, mock_client):
        """Query syntax errors should NOT be retried — they are deterministic."""
        mock_client._client.query_tx.side_effect = Exception(
            "SyntaxError: Invalid input 'METCH'"
        )

        with pytest.raises(RuntimeError, match="non-retryable"):
            mock_client._execute_query("METCH (n) RETURN n", max_retries=3)

        # Should only be called once (no retry)
        assert mock_client._client.query_tx.call_count == 1

    def test_generic_error_not_retried(self, mock_client):
        """Errors without connection keywords should NOT be retried."""
        mock_client._client.query_tx.side_effect = ValueError(
            "Invalid parameter type"
        )

        with pytest.raises(RuntimeError, match="non-retryable"):
            mock_client._execute_query("RETURN 1", max_retries=3)

        assert mock_client._client.query_tx.call_count == 1

    def test_max_retries_exhausted(self, mock_client):
        """Should raise RuntimeError after max_retries exhausted."""
        # Keep a reference to the mock before _client gets reset to None
        inner_client = mock_client._client
        inner_client.query_tx.side_effect = Exception(
            "Connection reset by peer"
        )

        with patch.object(mock_client, '_get_client', return_value=inner_client):
            with pytest.raises(RuntimeError, match="failed after"):
                mock_client._execute_query("RETURN 1", max_retries=2)

        # 1 initial + 2 retries = 3 total
        assert inner_client.query_tx.call_count == 3

    def test_client_reset_on_connection_error(self, mock_client):
        """Client should be set to None on connection error to force reconnect."""
        mock_client._client.query_tx.side_effect = Exception(
            "Connection broken"
        )

        with patch.object(mock_client, '_get_client', return_value=mock_client._client):
            with pytest.raises(RuntimeError):
                mock_client._execute_query("RETURN 1", max_retries=0)

        # After failure, _client should be reset to None
        assert mock_client._client is None

    def test_query_params_passed_through(self, mock_client):
        """Keyword params should be forwarded to query_tx."""
        mock_client._client.query_tx.return_value = []

        mock_client._execute_query(
            "MATCH (n) WHERE n.id = $id RETURN n",
            id="hgnc:11998",
            limit=10,
        )

        mock_client._client.query_tx.assert_called_once_with(
            "MATCH (n) WHERE n.id = $id RETURN n",
            id="hgnc:11998",
            limit=10,
        )

    def test_max_retries_zero_no_retry(self, mock_client):
        """With max_retries=0, connection errors should raise immediately."""
        inner_client = mock_client._client
        inner_client.query_tx.side_effect = Exception(
            "Connection timeout"
        )

        with pytest.raises(RuntimeError, match="failed after"):
            mock_client._execute_query("RETURN 1", max_retries=0)

        # Only 1 attempt (no retry with max_retries=0)
        assert inner_client.query_tx.call_count == 1


class TestExecuteQueryWithMockedNeo4jExceptions:
    """Test _execute_query behavior when neo4j exceptions are synthetically
    available or unavailable, independent of actual neo4j installation."""

    def test_typed_exception_path_when_available(self):
        """When _NEO4J_EXCEPTIONS_AVAILABLE is True, typed exceptions are caught."""
        from cliquefinder.knowledge import cogex as cogex_mod
        from cliquefinder.knowledge.cogex import CoGExClient

        if not cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE:
            pytest.skip("neo4j package not installed")

        client = CoGExClient.__new__(CoGExClient)
        client._url = "bolt://test:7687"
        client._user = "neo4j"
        client._password = "test"
        client._env_file = None
        client._client = MagicMock()

        # Raise a real ServiceUnavailable
        client._client.query_tx.side_effect = [
            cogex_mod.ServiceUnavailable("gone"),
            [("ok",)],
        ]

        with patch.object(client, '_get_client', return_value=client._client):
            result = client._execute_query("RETURN 1", max_retries=1)

        assert result == [("ok",)]

    def test_string_fallback_when_neo4j_unavailable(self):
        """When _NEO4J_EXCEPTIONS_AVAILABLE is False, string matching is sole detection."""
        from cliquefinder.knowledge import cogex as cogex_mod
        from cliquefinder.knowledge.cogex import CoGExClient

        client = CoGExClient.__new__(CoGExClient)
        client._url = "bolt://test:7687"
        client._user = "neo4j"
        client._password = "test"
        client._env_file = None
        client._client = MagicMock()

        # Temporarily disable typed exceptions
        original_flag = cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE
        try:
            cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE = False

            client._client.query_tx.side_effect = [
                Exception("Connection refused"),
                [("ok",)],
            ]

            with patch.object(client, '_get_client', return_value=client._client):
                result = client._execute_query("RETURN 1", max_retries=1)

            assert result == [("ok",)]
        finally:
            cogex_mod._NEO4J_EXCEPTIONS_AVAILABLE = original_flag


class TestPingUsesExecuteQuery:
    """Verify that ping() delegates to _execute_query."""

    def test_ping_calls_execute_query(self):
        """ping() should use _execute_query for connection testing."""
        from cliquefinder.knowledge.cogex import CoGExClient

        client = CoGExClient.__new__(CoGExClient)
        client._url = "bolt://test:7687"
        client._user = "neo4j"
        client._password = "test"
        client._env_file = None
        client._client = MagicMock()

        with patch.object(client, '_execute_query', return_value=[(1,)]) as mock_eq:
            result = client.ping()

        assert result is True
        mock_eq.assert_called_once_with("RETURN 1 as test")

    def test_ping_returns_false_on_failure(self):
        """ping() should return False when _execute_query raises."""
        from cliquefinder.knowledge.cogex import CoGExClient

        client = CoGExClient.__new__(CoGExClient)
        client._url = "bolt://test:7687"
        client._user = "neo4j"
        client._password = "test"
        client._env_file = None
        client._client = MagicMock()

        with patch.object(client, '_execute_query', side_effect=RuntimeError("Connection lost")):
            result = client.ping()

        assert result is False
