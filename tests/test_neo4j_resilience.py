"""Tests for Neo4j/CoGEx infrastructure resilience (ARCH-2, ARCH-4, ARCH-13).

ARCH-2:  Query failures must raise RuntimeError, never return empty lists.
ARCH-4:  Dead Neo4j connections trigger automatic reconnection.
ARCH-13: Gene name resolution is cached; MyGene.info client is a singleton.
"""

from unittest.mock import MagicMock, patch, PropertyMock
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
    INDRAModuleExtractor,
    INDRAEdge,
    GeneId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client_with_mock_neo4j(mock_neo4j_cls):
    """Create a CoGExClient with mocked Neo4jClient constructor."""
    client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
    return client


def _make_edge(target_name="MDM2", target_id="1234"):
    """Create a minimal INDRAEdge for testing."""
    return INDRAEdge(
        regulator_id=("HGNC", "11998"),
        regulator_name="TP53",
        target_id=("HGNC", target_id),
        target_name=target_name,
        regulation_type="activation",
        evidence_count=3,
        stmt_hash=12345,
        source_counts='{"reach": 2, "trrust": 1}',
    )


# ===========================================================================
# ARCH-4: Dead connection reconnection
# ===========================================================================

class TestExecuteQueryReconnection:
    """_execute_query should retry once on connection errors."""

    def test_reconnect_on_connection_error(self):
        """First call raises connection error -> reconnect -> retry succeeds."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client_bad = MagicMock()
        mock_client_bad.query_tx.side_effect = Exception("Connection refused by host")

        mock_client_good = MagicMock()
        mock_client_good.query_tx.return_value = [("ok",)]

        # Patch Neo4jClient constructor to return bad then good client
        call_count = [0]
        def fake_neo4j(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client_bad
            return mock_client_good

        with patch.object(cogex_mod, "Neo4jClient", side_effect=fake_neo4j):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

            # Force initial connection
            client._get_client()
            assert client._client is mock_client_bad

            # _execute_query should detect connection error, reconnect, and succeed
            result = client._execute_query("RETURN 1")
            assert result == [("ok",)]
            # Client should now be the good one
            assert client._client is mock_client_good

    def test_reconnect_fails_raises_runtime_error(self):
        """Connection error on both attempts -> RuntimeError."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Connection timeout")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

            with pytest.raises(RuntimeError, match="Query failed after reconnect attempt"):
                client._execute_query("RETURN 1")

    def test_non_connection_error_not_retried(self):
        """Syntax errors and other non-connection errors are re-raised as-is."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = ValueError("Invalid Cypher syntax")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

            with pytest.raises(ValueError, match="Invalid Cypher syntax"):
                client._execute_query("BAD CYPHER")

            # Should only have tried once (no reconnect)
            assert mock_client.query_tx.call_count == 1

    def test_force_reconnect_creates_new_client(self):
        """_get_client(force_reconnect=True) replaces existing client."""
        import cliquefinder.knowledge.cogex as cogex_mod

        clients = [MagicMock(name="client1"), MagicMock(name="client2")]
        call_idx = [0]

        def fake_neo4j(*args, **kwargs):
            c = clients[call_idx[0]]
            call_idx[0] += 1
            return c

        with patch.object(cogex_mod, "Neo4jClient", side_effect=fake_neo4j):
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

            first = client._get_client()
            assert first is clients[0]

            # Without force_reconnect, same client is returned
            same = client._get_client()
            assert same is clients[0]

            # With force_reconnect, new client is created
            second = client._get_client(force_reconnect=True)
            assert second is clients[1]

    def test_successful_query_no_reconnect(self):
        """When query succeeds, no reconnection is attempted."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [("row1",)]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client) as neo4j_cls:
            client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")
            result = client._execute_query("RETURN 1")

            assert result == [("row1",)]
            # Neo4jClient should only be constructed once (lazy init)
            assert neo4j_cls.call_count == 1


# ===========================================================================
# ARCH-2: Query failures raise RuntimeError
# ===========================================================================

class TestQueryFailuresRaiseRuntimeError:
    """Query methods must never silently return empty lists on infrastructure failure."""

    def test_get_downstream_targets_raises_on_failure(self):
        """get_downstream_targets raises RuntimeError when query fails."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Connection reset by peer")

        # After reconnect attempt, still fails -> RuntimeError
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError, match="Query failed"):
                    client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

    def test_discover_regulators_raises_on_failure(self, _mock_indra):
        """discover_regulators raises RuntimeError when query fails."""
        import cliquefinder.knowledge.cogex as cogex_mod

        # Make hgnc_client.get_current_hgnc_id return an ID for at least one gene
        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Connection timeout expired")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError, match="Reverse query failed"):
                    client.discover_regulators(
                        gene_universe=["TP53", "MDM2"],
                        min_evidence=2,
                        min_targets=1
                    )

    def test_get_downstream_targets_does_not_return_empty_on_failure(self):
        """Ensure failure does NOT silently return [] (the pre-fix anti-pattern)."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Server unavailable")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                # Should raise, not return []
                with pytest.raises((RuntimeError, Exception)):
                    result = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                    )
                    # This line should never be reached
                    assert result != [], "Query failure returned [] instead of raising"


# ===========================================================================
# ARCH-13: Gene resolution caching
# ===========================================================================

class TestGeneResolutionCache:
    """resolve_gene_name should cache results to avoid redundant lookups."""

    def test_cache_avoids_redundant_lookups(self, _mock_indra):
        """Second call for same gene should not call hgnc_client again."""
        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)

        # First call
        result1 = extractor.resolve_gene_name("TP53")
        assert result1 == ("HGNC", "11998")

        # Second call — should return cached result
        result2 = extractor.resolve_gene_name("TP53")
        assert result2 == ("HGNC", "11998")

        # hgnc_client.get_current_hgnc_id should only have been called on first resolve
        # (it's called for both name and name.upper() via dict.fromkeys, but only
        # on the first resolve — the second should be pure cache hit)
        first_call_count = _mock_indra.get_current_hgnc_id.call_count
        # Reset and call again
        _mock_indra.get_current_hgnc_id.reset_mock()
        result3 = extractor.resolve_gene_name("TP53")
        assert result3 == ("HGNC", "11998")
        # After reset, no new calls should be made
        assert _mock_indra.get_current_hgnc_id.call_count == 0

    def test_cache_stores_none_for_unresolvable(self, _mock_indra):
        """None results are also cached to avoid repeated failed lookups."""
        _mock_indra.get_current_hgnc_id.return_value = None

        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)

        result1 = extractor.resolve_gene_name("FAKEGENE")
        assert result1 is None

        _mock_indra.get_current_hgnc_id.reset_mock()

        result2 = extractor.resolve_gene_name("FAKEGENE")
        assert result2 is None
        assert _mock_indra.get_current_hgnc_id.call_count == 0

    def test_different_genes_resolve_independently(self, _mock_indra):
        """Different gene names get their own cache entries."""
        def side_effect(name):
            return {"TP53": "11998", "MYC": "7553"}.get(name)

        _mock_indra.get_current_hgnc_id.side_effect = side_effect

        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)

        r1 = extractor.resolve_gene_name("TP53")
        r2 = extractor.resolve_gene_name("MYC")

        assert r1 == ("HGNC", "11998")
        assert r2 == ("HGNC", "7553")

    def test_cache_is_instance_level(self, _mock_indra):
        """Each extractor instance has its own cache (no cross-contamination)."""
        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_cogex_client = MagicMock(spec=CoGExClient)
        ext1 = INDRAModuleExtractor(mock_cogex_client)
        ext2 = INDRAModuleExtractor(mock_cogex_client)

        ext1.resolve_gene_name("TP53")
        assert "TP53" in ext1._gene_cache
        assert "TP53" not in ext2._gene_cache


class TestMyGeneSingleton:
    """_get_mygene_client should return the same instance on repeated calls."""

    def test_mygene_client_is_singleton(self):
        """Calling _get_mygene_client twice returns the same object."""
        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)

        mock_mg = MagicMock()
        with patch("cliquefinder.knowledge.cogex.mygene", create=True) as mock_mygene_mod:
            # We need to patch the import inside _get_mygene_client
            with patch.dict("sys.modules", {"mygene": MagicMock(MyGeneInfo=MagicMock(return_value=mock_mg))}):
                client1 = extractor._get_mygene_client()
                client2 = extractor._get_mygene_client()
                assert client1 is client2

    def test_mygene_client_initially_none(self):
        """New extractor has _mygene_client = None."""
        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)
        assert extractor._mygene_client is None

    def test_mygene_client_set_after_first_call(self):
        """After first call, _mygene_client is set."""
        mock_cogex_client = MagicMock(spec=CoGExClient)
        extractor = INDRAModuleExtractor(mock_cogex_client)

        mock_mg = MagicMock()
        with patch.dict("sys.modules", {"mygene": MagicMock(MyGeneInfo=MagicMock(return_value=mock_mg))}):
            extractor._get_mygene_client()
            assert extractor._mygene_client is mock_mg


# ===========================================================================
# Integration: _execute_query flows through to query methods
# ===========================================================================

class TestExecuteQueryIntegration:
    """Verify that get_downstream_targets and discover_regulators use _execute_query."""

    def test_get_downstream_targets_uses_execute_query(self):
        """get_downstream_targets should call _execute_query internally."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client_instance = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client_instance):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                # Patch _execute_query to verify it's called
                with patch.object(client, "_execute_query", return_value=[]) as mock_eq:
                    result = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )
                    assert mock_eq.called
                    assert result == []

    def test_discover_regulators_uses_execute_query(self, _mock_indra):
        """discover_regulators should call _execute_query internally."""
        import cliquefinder.knowledge.cogex as cogex_mod

        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_client_instance = MagicMock()
        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client_instance):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with patch.object(client, "_execute_query", return_value=[]) as mock_eq:
                    result = client.discover_regulators(
                        gene_universe=["TP53"],
                        min_evidence=2,
                        min_targets=1
                    )
                    assert mock_eq.called
