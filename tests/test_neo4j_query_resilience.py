"""Tests for Neo4j/CoGEx query resilience and resource management.

Covers:
- get_downstream_targets Cypher has server-side LIMIT via max_results.
- discover_regulators passes remaining budget as per-chunk LIMIT.
- Exponential backoff with jitter on retryable connection errors.
- CoGExClient supports context manager protocol (with ... as client).
"""

from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Fixtures: mock INDRA so tests don't require a live Neo4j connection
# ---------------------------------------------------------------------------

MOCK_TFS = ["TP53", "MYC", "JUN"]
MOCK_KINASES = ["AKT1", "MAPK1"]
MOCK_PHOSPHATASES = ["PTEN"]


@pytest.fixture(autouse=True)
def _mock_indra():
    """Patch hgnc_client on the already-imported cogex module."""
    mock_hgnc = MagicMock()
    mock_hgnc.tfs = MOCK_TFS
    mock_hgnc.kinases = MOCK_KINASES
    mock_hgnc.phosphatases = MOCK_PHOSPHATASES
    mock_hgnc.get_current_hgnc_id = MagicMock(return_value=None)

    import cliquefinder.knowledge.cogex as cogex_mod
    orig_hgnc = cogex_mod.hgnc_client
    orig_available = cogex_mod.INDRA_AVAILABLE
    cogex_mod.hgnc_client = mock_hgnc
    cogex_mod.INDRA_AVAILABLE = True
    yield mock_hgnc
    cogex_mod.hgnc_client = orig_hgnc
    cogex_mod.INDRA_AVAILABLE = orig_available


from cliquefinder.knowledge.cogex import (
    CoGExClient,
    INDRAEdge,
)


# ===========================================================================
# get_downstream_targets server-side LIMIT
# ===========================================================================

class TestDownstreamTargetsLimit:
    """get_downstream_targets should pass max_results to Cypher LIMIT."""

    def test_default_max_results_passed_to_query(self):
        """Default max_results=50000 is forwarded as a Cypher parameter."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.return_value = []

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client.get_downstream_targets(
                    regulator=("HGNC", "11998"),
                    stmt_types=["IncreaseAmount"],
                    min_evidence=2,
                )

                # Verify query_tx was called with max_results parameter
                assert mock_client.query_tx.call_count == 1
                _, kwargs = mock_client.query_tx.call_args
                assert kwargs["max_results"] == 50_000

    def test_custom_max_results_forwarded(self):
        """Custom max_results is forwarded to the Cypher query."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.return_value = []

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client.get_downstream_targets(
                    regulator=("HGNC", "11998"),
                    stmt_types=["IncreaseAmount"],
                    min_evidence=2,
                    max_results=100,
                )

                _, kwargs = mock_client.query_tx.call_args
                assert kwargs["max_results"] == 100

    def test_cypher_query_contains_limit(self):
        """The Cypher query string includes LIMIT $max_results."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.return_value = []

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client.get_downstream_targets(
                    regulator=("HGNC", "11998"),
                )

                query_arg = mock_client.query_tx.call_args[0][0]
                assert "LIMIT $max_results" in query_arg

    def test_warning_emitted_when_limit_reached(self):
        """A warning is emitted when results equal max_results."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Simulate 5 rows returned for max_results=5
        mock_rows = [
            (
                "hgnc:11998", "TP53",
                f"hgnc:{1000 + i}", f"TARGET{i}",
                "IncreaseAmount", 3, 10000 + i, '{"reach": 2}'
            )
            for i in range(5)
        ]
        mock_client = MagicMock()
        mock_client.query_tx.return_value = mock_rows

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                import logging
                with patch.object(
                    logging.getLogger("cliquefinder.knowledge.cogex"),
                    "warning",
                ) as mock_warn:
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        max_results=5,
                    )
                    # Should have returned all 5 edges and warned
                    assert len(edges) == 5
                    # Check that a truncation warning was issued
                    warn_calls = [
                        c for c in mock_warn.call_args_list
                        if "max_results" in str(c)
                    ]
                    assert len(warn_calls) >= 1

    def test_no_warning_when_under_limit(self):
        """No warning when results are below max_results."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_rows = [
            (
                "hgnc:11998", "TP53",
                "hgnc:1000", "TARGET0",
                "IncreaseAmount", 3, 10000, '{"reach": 2}'
            )
        ]
        mock_client = MagicMock()
        mock_client.query_tx.return_value = mock_rows

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                import logging
                with patch.object(
                    logging.getLogger("cliquefinder.knowledge.cogex"),
                    "warning",
                ) as mock_warn:
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        max_results=1000,
                    )
                    assert len(edges) == 1
                    # No truncation warning
                    warn_calls = [
                        c for c in mock_warn.call_args_list
                        if "max_results" in str(c)
                    ]
                    assert len(warn_calls) == 0


# ===========================================================================
# discover_regulators server-side LIMIT per chunk
# ===========================================================================

class TestDiscoverRegulatorsChunkLimit:
    """discover_regulators should pass remaining budget as per-chunk LIMIT."""

    def test_cypher_query_contains_chunk_limit(self, _mock_indra):
        """The Cypher query string includes LIMIT $chunk_limit."""
        import cliquefinder.knowledge.cogex as cogex_mod

        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_client = MagicMock()
        mock_client.query_tx.return_value = []

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client.discover_regulators(
                    gene_universe=["TP53"],
                    min_targets=1,
                )

                query_arg = mock_client.query_tx.call_args[0][0]
                assert "LIMIT $chunk_limit" in query_arg

    def test_chunk_limit_equals_max_results_on_first_chunk(self, _mock_indra):
        """First chunk should use max_results as chunk_limit."""
        import cliquefinder.knowledge.cogex as cogex_mod

        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_client = MagicMock()
        mock_client.query_tx.return_value = []

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client.discover_regulators(
                    gene_universe=["TP53"],
                    min_targets=1,
                    max_results=500,
                )

                _, kwargs = mock_client.query_tx.call_args
                assert kwargs["chunk_limit"] == 500

    def test_remaining_budget_decreases_across_chunks(self, _mock_indra):
        """When chunking, subsequent chunks use reduced budget."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Make hgnc_client resolve genes to IDs
        id_counter = [0]
        def fake_hgnc_id(name):
            id_counter[0] += 1
            return str(id_counter[0])
        _mock_indra.get_current_hgnc_id.side_effect = fake_hgnc_id

        # Create enough genes to force 2 chunks (chunk_size=5000)
        # Instead, we'll shrink chunk_size for the test
        mock_client = MagicMock()

        # First chunk returns 3 rows, second returns 0
        chunk1_rows = [
            (
                "hgnc:99", "REGULATOR",
                f"hgnc:{i}", f"TARGET{i}",
                "IncreaseAmount", 5, 10000 + i, '{"reach": 2}'
            )
            for i in range(3)
        ]
        mock_client.query_tx.side_effect = [chunk1_rows, []]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", side_effect=lambda ns, id: f"hgnc:{id}"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                # Use a very small chunk size to force multiple chunks
                orig_chunk = client.CURIE_CHUNK_SIZE
                client.CURIE_CHUNK_SIZE = 1

                gene_names = ["GENE1", "GENE2"]
                client.discover_regulators(
                    gene_universe=gene_names,
                    min_targets=1,
                    max_results=100,
                )

                client.CURIE_CHUNK_SIZE = orig_chunk

                # Should have called query_tx twice (2 chunks)
                assert mock_client.query_tx.call_count == 2

                # First chunk: chunk_limit = 100 (full budget)
                _, kwargs1 = mock_client.query_tx.call_args_list[0]
                assert kwargs1["chunk_limit"] == 100

                # Second chunk: chunk_limit = 100 - 3 = 97 (remaining)
                _, kwargs2 = mock_client.query_tx.call_args_list[1]
                assert kwargs2["chunk_limit"] == 97

    def test_stops_chunking_when_budget_exhausted(self, _mock_indra):
        """When max_results is reached, further chunks are skipped."""
        import cliquefinder.knowledge.cogex as cogex_mod

        id_counter = [0]
        def fake_hgnc_id(name):
            id_counter[0] += 1
            return str(id_counter[0])
        _mock_indra.get_current_hgnc_id.side_effect = fake_hgnc_id

        mock_client = MagicMock()

        # First chunk returns 3 rows — that meets the max_results=3 budget
        chunk1_rows = [
            (
                "hgnc:99", "REGULATOR",
                f"hgnc:{i}", f"TARGET{i}",
                "IncreaseAmount", 5, 10000 + i, '{"reach": 2}'
            )
            for i in range(3)
        ]
        mock_client.query_tx.side_effect = [chunk1_rows, []]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", side_effect=lambda ns, id: f"hgnc:{id}"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                orig_chunk = client.CURIE_CHUNK_SIZE
                client.CURIE_CHUNK_SIZE = 1

                gene_names = ["GENE1", "GENE2", "GENE3"]
                client.discover_regulators(
                    gene_universe=gene_names,
                    min_targets=1,
                    max_results=3,
                )

                client.CURIE_CHUNK_SIZE = orig_chunk

                # Should have called query_tx only once since budget is exhausted
                assert mock_client.query_tx.call_count == 1


# ===========================================================================
# Exponential backoff with jitter
# ===========================================================================

class TestExponentialBackoff:
    """_execute_query should apply exponential backoff on connection errors."""

    def test_backoff_called_on_connection_error(self):
        """time.sleep is called with exponential delay on connection error retry."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client_bad = MagicMock()
        mock_client_bad.query_tx.side_effect = Exception("Connection refused by host")

        mock_client_good = MagicMock()
        mock_client_good.query_tx.return_value = [("ok",)]

        call_count = [0]
        def fake_neo4j(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client_bad
            return mock_client_good

        with patch.object(cogex_mod, "Neo4jClient", side_effect=fake_neo4j):
            with patch.object(cogex_mod, "time") as mock_time:
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                client._get_client()

                result = client._execute_query("RETURN 1")
                assert result == [("ok",)]

                # time.sleep should have been called once (before first retry)
                assert mock_time.sleep.call_count == 1
                delay = mock_time.sleep.call_args[0][0]
                # delay = 2^0 + jitter(0,1) => between 1.0 and 2.0
                assert 1.0 <= delay <= 2.0

    def test_backoff_increases_with_retry_count(self):
        """Backoff delay increases exponentially with attempt number."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        # Fail 3 times, then succeed on 4th attempt
        mock_client.query_tx.side_effect = [
            Exception("Connection timeout"),
            Exception("Connection timeout"),
            Exception("Connection timeout"),
            [("ok",)],
        ]

        client_call_count = [0]
        def fake_neo4j(*args, **kwargs):
            client_call_count[0] += 1
            return mock_client

        with patch.object(cogex_mod, "Neo4jClient", side_effect=fake_neo4j):
            with patch.object(cogex_mod, "time") as mock_time:
                with patch.object(cogex_mod, "random") as mock_random:
                    # Fix jitter to 0.5 for predictable delays
                    mock_random.uniform.return_value = 0.5

                    client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                    result = client._execute_query("RETURN 1", max_retries=3)
                    assert result == [("ok",)]

                    # Should have 3 sleep calls (attempts 0, 1, 2)
                    assert mock_time.sleep.call_count == 3

                    delays = [c[0][0] for c in mock_time.sleep.call_args_list]
                    # attempt 0: 2^0 + 0.5 = 1.5
                    assert delays[0] == pytest.approx(1.5)
                    # attempt 1: 2^1 + 0.5 = 2.5
                    assert delays[1] == pytest.approx(2.5)
                    # attempt 2: 2^2 + 0.5 = 4.5
                    assert delays[2] == pytest.approx(4.5)

    def test_no_backoff_on_non_connection_error(self):
        """Syntax errors should not trigger backoff or sleep."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = ValueError("Invalid Cypher syntax")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "time") as mock_time:
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(ValueError, match="Invalid Cypher syntax"):
                    client._execute_query("BAD CYPHER")

                # No sleep calls for non-connection errors
                assert mock_time.sleep.call_count == 0

    def test_no_backoff_on_final_attempt(self):
        """When all retries are exhausted, sleep is NOT called after the last failure."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Connection timeout")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "time") as mock_time:
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError, match="Query failed after"):
                    client._execute_query("RETURN 1", max_retries=2)

                # With max_retries=2: 3 attempts total, 2 sleeps (before retry 1 and 2)
                # No sleep after the final failed attempt
                assert mock_time.sleep.call_count == 2

    def test_backoff_uses_random_jitter(self):
        """Jitter is applied using random.uniform(0, 1)."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = [
            Exception("Connection refused"),
            [("ok",)],
        ]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "time") as mock_time:
                with patch.object(cogex_mod, "random") as mock_random:
                    mock_random.uniform.return_value = 0.42

                    client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
                    client._execute_query("RETURN 1")

                    # random.uniform(0, 1) was called
                    mock_random.uniform.assert_called_with(0, 1)

                    # delay = 2^0 + 0.42 = 1.42
                    delay = mock_time.sleep.call_args[0][0]
                    assert delay == pytest.approx(1.42)


# ===========================================================================
# Context manager protocol
# ===========================================================================

class TestContextManager:
    """CoGExClient should support `with` statement for resource cleanup."""

    def test_enter_returns_self(self):
        """__enter__ returns the client instance itself."""
        import cliquefinder.knowledge.cogex as cogex_mod

        with patch.object(cogex_mod, "Neo4jClient"):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            result = client.__enter__()
            assert result is client

    def test_exit_calls_close(self):
        """__exit__ calls self.close()."""
        import cliquefinder.knowledge.cogex as cogex_mod

        with patch.object(cogex_mod, "Neo4jClient"):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            with patch.object(client, "close") as mock_close:
                client.__exit__(None, None, None)
                mock_close.assert_called_once()

    def test_exit_returns_false(self):
        """__exit__ returns False (does not suppress exceptions)."""
        import cliquefinder.knowledge.cogex as cogex_mod

        with patch.object(cogex_mod, "Neo4jClient"):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            result = client.__exit__(None, None, None)
            assert result is False

    def test_with_statement_calls_close(self):
        """Using `with` statement calls close on exit."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_neo4j = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_neo4j):
            with CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret") as client:
                # Force a connection so we can verify close behavior
                client._get_client()
                assert client._client is not None

            # After exiting the with block, client should be closed
            assert client._client is None

    def test_with_statement_closes_on_exception(self):
        """Using `with` statement calls close even when exception occurs."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_neo4j = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_neo4j):
            with pytest.raises(ValueError, match="test error"):
                with CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret") as client:
                    client._get_client()
                    raise ValueError("test error")

            # Client should still be closed after exception
            assert client._client is None

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_neo4j = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_neo4j):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            client._get_client()
            client.close()
            client.close()  # Should not raise

    def test_close_calls_underlying_close(self):
        """close() calls the underlying Neo4jClient.close() if available."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_neo4j = MagicMock()
        mock_neo4j.close = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_neo4j):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            client._get_client()
            client.close()
            mock_neo4j.close.assert_called_once()

    def test_close_without_connection_is_noop(self):
        """close() is a no-op if no connection was ever made."""
        import cliquefinder.knowledge.cogex as cogex_mod

        with patch.object(cogex_mod, "Neo4jClient"):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            # Never called _get_client(), so _client is None
            client.close()  # Should not raise
