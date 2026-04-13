"""Tests for CoGExClient.fetch_evidence_for_edges and INDRAKnowledgeSource.fetch_evidence_text.

Verifies that:
- Evidence text is fetched via get_statements_by_hash and parsed correctly
- Results are cached so repeated calls avoid redundant API requests
- Empty input returns empty output
- INDRA DB unavailability is handled gracefully (warning, empty result)
"""

from unittest.mock import MagicMock, patch

import pytest

from cliquefinder.knowledge.cogex import CoGExClient, INDRAEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edge(stmt_hash: int, target: str = "GeneA") -> INDRAEdge:
    """Build a minimal INDRAEdge for testing."""
    return INDRAEdge(
        regulator_id=("HGNC", "0"),
        regulator_name="REG",
        target_id=("HGNC", "1"),
        target_name=target,
        regulation_type="activation",
        evidence_count=3,
        stmt_hash=stmt_hash,
        source_counts='{"reach": 2, "sparser": 1}',
    )


def _make_mock_statement(stmt_hash: int, texts: list[str]) -> MagicMock:
    """Build a mock INDRA Statement with evidence list."""
    stmt = MagicMock()
    stmt.get_hash.return_value = stmt_hash

    evidences = []
    for i, text in enumerate(texts):
        ev = MagicMock()
        ev.text = text
        ev.source_api = "reach"
        ev.pmid = f"1234{i}"
        evidences.append(ev)

    stmt.evidence = evidences
    return stmt


# ---------------------------------------------------------------------------
# CoGExClient.fetch_evidence_for_edges
# ---------------------------------------------------------------------------

class TestFetchEvidenceForEdges:
    """Tests for CoGExClient.fetch_evidence_for_edges."""

    @patch("cliquefinder.knowledge.cogex.INDRA_AVAILABLE", True)
    def _make_client(self) -> CoGExClient:
        """Create a CoGExClient with INDRA availability mocked."""
        with patch("cliquefinder.knowledge.cogex.Neo4jClient"):
            return CoGExClient(url="bolt://x", user="u", password="p")

    def test_basic_fetch(self):
        """get_statements_by_hash is called and evidence is extracted."""
        client = self._make_client()
        edges = [_make_edge(111), _make_edge(222)]

        mock_processor = MagicMock()
        mock_processor.statements = [
            _make_mock_statement(111, ["Text A", "Text B"]),
            _make_mock_statement(222, ["Text C"]),
        ]

        with patch(
            "cliquefinder.knowledge.cogex.get_statements_by_hash",
            create=True,
        ) as mock_fetch:
            # Patch the import inside the method
            with patch.dict(
                "sys.modules",
                {"indra.sources.indra_db_rest": MagicMock(
                    get_statements_by_hash=MagicMock(return_value=mock_processor)
                )},
            ):
                result = client.fetch_evidence_for_edges(edges)

        assert 111 in result
        assert 222 in result
        assert len(result[111]) == 2
        assert result[111][0]["text"] == "Text A"
        assert result[111][1]["source_api"] == "reach"
        assert result[222][0]["pmid"] == "12340"

    def test_caching_prevents_refetch(self):
        """Second call with same hashes does not invoke the API again."""
        client = self._make_client()
        edges = [_make_edge(333)]

        mock_processor = MagicMock()
        mock_processor.statements = [
            _make_mock_statement(333, ["Cached text"]),
        ]

        mock_api = MagicMock(return_value=mock_processor)

        with patch.dict(
            "sys.modules",
            {"indra.sources.indra_db_rest": MagicMock(
                get_statements_by_hash=mock_api
            )},
        ):
            result1 = client.fetch_evidence_for_edges(edges)
            result2 = client.fetch_evidence_for_edges(edges)

        # API should have been called only once
        assert mock_api.call_count == 1
        assert result1 == result2
        assert result2[333][0]["text"] == "Cached text"

    def test_empty_edges_returns_empty(self):
        """Empty edge list returns empty dict without API call."""
        client = self._make_client()
        result = client.fetch_evidence_for_edges([])
        assert result == {}

    def test_graceful_on_import_error(self):
        """When indra.sources.indra_db_rest is not importable, return empty and warn."""
        client = self._make_client()
        edges = [_make_edge(444)]

        # Remove the module so the import inside the method fails
        with patch.dict("sys.modules", {"indra.sources.indra_db_rest": None}):
            result = client.fetch_evidence_for_edges(edges)

        assert result == {}

    def test_graceful_on_api_exception(self):
        """When the INDRA DB REST API raises, log warning and return cached (empty)."""
        client = self._make_client()
        edges = [_make_edge(555)]

        mock_api = MagicMock(side_effect=ConnectionError("DB down"))

        with patch.dict(
            "sys.modules",
            {"indra.sources.indra_db_rest": MagicMock(
                get_statements_by_hash=mock_api
            )},
        ):
            result = client.fetch_evidence_for_edges(edges)

        # No crash, returns empty for the hash
        assert result == {}

    def test_batching_large_hash_lists(self):
        """Hashes are batched in chunks of 100."""
        client = self._make_client()
        # 250 unique edges -> 3 batches (100, 100, 50)
        edges = [_make_edge(i) for i in range(250)]

        call_args_list = []

        def mock_api_fn(hash_list, ev_limit=10):
            call_args_list.append(len(hash_list))
            proc = MagicMock()
            proc.statements = [
                _make_mock_statement(h, [f"text for {h}"]) for h in hash_list
            ]
            return proc

        with patch.dict(
            "sys.modules",
            {"indra.sources.indra_db_rest": MagicMock(
                get_statements_by_hash=mock_api_fn
            )},
        ):
            result = client.fetch_evidence_for_edges(edges)

        assert call_args_list == [100, 100, 50]
        assert len(result) == 250

    def test_missing_hashes_cached_as_empty(self):
        """Hashes not returned by the API are cached as empty lists."""
        client = self._make_client()
        edges = [_make_edge(600), _make_edge(601)]

        mock_processor = MagicMock()
        # Only hash 600 comes back; 601 is missing
        mock_processor.statements = [
            _make_mock_statement(600, ["Found"]),
        ]

        mock_api = MagicMock(return_value=mock_processor)

        with patch.dict(
            "sys.modules",
            {"indra.sources.indra_db_rest": MagicMock(
                get_statements_by_hash=mock_api
            )},
        ):
            result = client.fetch_evidence_for_edges(edges)

        assert result[600][0]["text"] == "Found"
        assert result[601] == []

        # Second call should not re-fetch 601
        with patch.dict(
            "sys.modules",
            {"indra.sources.indra_db_rest": MagicMock(
                get_statements_by_hash=mock_api
            )},
        ):
            result2 = client.fetch_evidence_for_edges(edges)

        # Still only one call total
        assert mock_api.call_count == 1
        assert result2[601] == []
