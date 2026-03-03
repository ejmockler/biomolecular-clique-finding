"""Tests for SEC-III security/performance fixes (Audit III).

SEC-III-1: Cache integrity checksums for correlation matrix caches
SEC-III-2: Neo4j credential sanitization in error messages
SEC-III-3: Batch gene resolution using local HGNC symbol map
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SEC-III-1: Cache integrity checksums
# ---------------------------------------------------------------------------


class TestCacheIntegrityChecksums:
    """Tests for _write_cache_checksum / _verify_cache_checksum."""

    def test_write_creates_sidecar_file(self, tmp_path):
        """_write_cache_checksum creates a .sha256 sidecar file."""
        from cliquefinder.utils.correlation_matrix import _write_cache_checksum

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"test data content")

        _write_cache_checksum(cache_file)

        sha_path = Path(f"{cache_file}.sha256")
        assert sha_path.exists()

    def test_sidecar_contains_correct_sha256(self, tmp_path):
        """Sidecar file contains SHA256 hex digest of the cache file."""
        from cliquefinder.utils.correlation_matrix import _write_cache_checksum

        cache_file = tmp_path / "test_cache.mmap"
        content = b"deterministic test content"
        cache_file.write_bytes(content)

        _write_cache_checksum(cache_file)

        sha_path = Path(f"{cache_file}.sha256")
        expected_hash = hashlib.sha256(content).hexdigest()
        actual_hash = sha_path.read_text().strip()
        assert actual_hash == expected_hash

    def test_verify_returns_true_for_valid_cache(self, tmp_path):
        """_verify_cache_checksum returns True when digest matches."""
        from cliquefinder.utils.correlation_matrix import (
            _verify_cache_checksum,
            _write_cache_checksum,
        )

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"valid data")
        _write_cache_checksum(cache_file)

        assert _verify_cache_checksum(cache_file) is True

    def test_verify_returns_false_for_tampered_cache(self, tmp_path):
        """_verify_cache_checksum returns False when file has been modified."""
        from cliquefinder.utils.correlation_matrix import (
            _verify_cache_checksum,
            _write_cache_checksum,
        )

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"original data")
        _write_cache_checksum(cache_file)

        # Tamper with the cache file
        cache_file.write_bytes(b"tampered data")

        assert _verify_cache_checksum(cache_file) is False

    def test_verify_returns_true_for_legacy_cache_no_sidecar(self, tmp_path):
        """Graceful upgrade: returns True if no sidecar exists."""
        from cliquefinder.utils.correlation_matrix import _verify_cache_checksum

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"legacy cache without sidecar")

        # No sidecar file exists
        assert _verify_cache_checksum(cache_file) is True

    def test_verify_returns_false_for_corrupted_sidecar(self, tmp_path):
        """Returns False when sidecar hash is garbage."""
        from cliquefinder.utils.correlation_matrix import _verify_cache_checksum

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"data")

        sha_path = Path(f"{cache_file}.sha256")
        sha_path.write_text("0000000000000000000000000000000000000000000000000000000000000000")

        assert _verify_cache_checksum(cache_file) is False

    def test_verify_logs_warning_on_mismatch(self, tmp_path, caplog):
        """Logs a warning when integrity check fails."""
        from cliquefinder.utils.correlation_matrix import _verify_cache_checksum

        cache_file = tmp_path / "test_cache.mmap"
        cache_file.write_bytes(b"data")

        sha_path = Path(f"{cache_file}.sha256")
        sha_path.write_text("deadbeef" * 8)

        with caplog.at_level(logging.WARNING, logger="cliquefinder.utils.correlation_matrix"):
            result = _verify_cache_checksum(cache_file)

        assert result is False
        assert "Cache integrity check failed" in caplog.text

    def test_roundtrip_write_then_verify(self, tmp_path):
        """Full roundtrip: write checksum then verify."""
        from cliquefinder.utils.correlation_matrix import (
            _verify_cache_checksum,
            _write_cache_checksum,
        )

        cache_file = tmp_path / "roundtrip.mmap"
        # Create a realistic-sized file
        data = np.random.default_rng(42).random((100, 100), dtype=np.float32)
        data.tofile(str(cache_file))

        _write_cache_checksum(cache_file)
        assert _verify_cache_checksum(cache_file) is True

    def test_overwrite_updates_sidecar(self, tmp_path):
        """Re-writing checksum updates the sidecar file."""
        from cliquefinder.utils.correlation_matrix import (
            _verify_cache_checksum,
            _write_cache_checksum,
        )

        cache_file = tmp_path / "updated.mmap"
        cache_file.write_bytes(b"version1")
        _write_cache_checksum(cache_file)

        # Overwrite cache and update checksum
        cache_file.write_bytes(b"version2")
        _write_cache_checksum(cache_file)

        assert _verify_cache_checksum(cache_file) is True


# ---------------------------------------------------------------------------
# SEC-III-2: Neo4j credential sanitization
# ---------------------------------------------------------------------------


class TestNeo4jCredentialSanitization:
    """Tests for _sanitize_connection_error."""

    def test_strips_userinfo_from_neo4j_uri(self):
        """Removes user:password from neo4j:// URIs."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        msg = "Failed to connect to neo4j://admin:secretpassword@db.example.com:7687"
        result = _sanitize_connection_error(msg)
        assert "secretpassword" not in result
        assert "admin" not in result
        assert "neo4j://***@db.example.com:7687" in result

    def test_strips_userinfo_from_bolt_uri(self):
        """Removes user:password from bolt:// URIs."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        msg = "Connection refused: bolt://neo4j:mypassword@localhost:7687"
        result = _sanitize_connection_error(msg)
        assert "mypassword" not in result
        assert "neo4j" not in result
        assert "bolt://***@localhost:7687" in result

    def test_strips_userinfo_from_neo4js_uri(self):
        """Removes user:password from neo4j+s:// URIs."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        msg = "neo4j+s://user:pass123@host.com/db timeout"
        result = _sanitize_connection_error(msg)
        assert "pass123" not in result
        assert "neo4j+s://***@host.com/db timeout" in result

    def test_preserves_message_without_credentials(self):
        """Messages without URIs are returned unchanged."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        msg = "Connection timed out after 30 seconds"
        result = _sanitize_connection_error(msg)
        assert result == msg

    def test_handles_empty_string(self):
        """Empty string input returns empty string."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        assert _sanitize_connection_error("") == ""

    def test_handles_non_string_input(self):
        """Non-string input is converted via str()."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        result = _sanitize_connection_error(42)
        assert result == "42"

    def test_multiple_uris_in_message(self):
        """Multiple URIs in a message are all sanitized."""
        from cliquefinder.knowledge.cogex import _sanitize_connection_error

        msg = "Tried neo4j://u1:p1@host1 then bolt://u2:p2@host2"
        result = _sanitize_connection_error(msg)
        assert "p1" not in result
        assert "p2" not in result
        assert "u1" not in result
        assert "u2" not in result

    def test_execute_query_sanitizes_connection_error(self):
        """_execute_query uses sanitized messages when logging/raising."""
        from cliquefinder.knowledge import cogex as cogex_mod

        # Create a client with mock credentials
        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            client = cogex_mod.CoGExClient.__new__(cogex_mod.CoGExClient)
            client._url = "neo4j://admin:secret@db.example.com"
            client._user = "admin"
            client._password = "secret"
            client._env_file = None
            client._client = None

            # Mock _get_client to raise a connection error with credentials.
            # Use a message containing one of the _CONNECTION_ERROR_KEYWORDS
            # ("connection") so the retry logic classifies it correctly.
            def raise_connection_error():
                raise Exception(
                    "connection refused to neo4j://admin:secret@db.example.com"
                )

            with mock.patch.object(client, '_get_client', side_effect=raise_connection_error):
                with pytest.raises(RuntimeError) as exc_info:
                    client._execute_query("RETURN 1", max_retries=0)

                error_msg = str(exc_info.value)
                assert "secret" not in error_msg
                assert "***" in error_msg

    def test_ping_sanitizes_error(self):
        """ping() logs sanitized error messages."""
        from cliquefinder.knowledge import cogex as cogex_mod

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            client = cogex_mod.CoGExClient.__new__(cogex_mod.CoGExClient)
            client._url = "neo4j://admin:secret@db.example.com"
            client._user = "admin"
            client._password = "secret"
            client._env_file = None
            client._client = None

            with mock.patch.object(
                client, '_execute_query',
                side_effect=ConnectionError("bolt://user:pass@host failed")
            ):
                with mock.patch.object(cogex_mod.logger, 'error') as mock_log:
                    result = client.ping()

                    assert result is False
                    log_msg = mock_log.call_args[0][1]
                    assert "pass" not in log_msg
                    assert "***" in log_msg


# ---------------------------------------------------------------------------
# SEC-III-3: Batch gene resolution
# ---------------------------------------------------------------------------


class TestHGNCSymbolMap:
    """Tests for _get_hgnc_symbol_map."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        from cliquefinder.knowledge import cogex as cogex_mod

        # Reset the cached map
        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = {"11998": "TP53", "6407": "LRRK2"}

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                result = cogex_mod._get_hgnc_symbol_map()

        assert isinstance(result, dict)
        assert result["TP53"] == "11998"
        assert result["LRRK2"] == "6407"

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_caches_result(self):
        """Only builds the map once (cached globally)."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = {"1": "A"}

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                result1 = cogex_mod._get_hgnc_symbol_map()
                result2 = cogex_mod._get_hgnc_symbol_map()

        assert result1 is result2  # Same object, not rebuilt

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_returns_empty_dict_on_error(self):
        """Returns empty dict if hgnc_client is not available."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = mock.PropertyMock(side_effect=AttributeError)
            type(mock_hgnc).hgnc_names = mock.PropertyMock(side_effect=AttributeError)

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                result = cogex_mod._get_hgnc_symbol_map()

        assert result == {}

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None


class TestBatchGeneResolution:
    """Tests for INDRAModuleExtractor.resolve_gene_names_batch."""

    def _make_extractor(self):
        """Create a minimal INDRAModuleExtractor for testing."""
        from cliquefinder.knowledge import cogex as cogex_mod

        extractor = cogex_mod.INDRAModuleExtractor.__new__(
            cogex_mod.INDRAModuleExtractor
        )
        extractor.client = mock.MagicMock()
        extractor.id_mapper = None
        extractor._gene_cache = OrderedDict()
        extractor._gene_cache_maxsize = 50_000
        extractor._mygene_client = None
        return extractor

    def test_resolves_known_symbols_from_map(self):
        """Batch resolution uses local symbol map for known genes."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = {"11998": "TP53", "391": "AKT1"}

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                extractor = self._make_extractor()
                results = extractor.resolve_gene_names_batch(["TP53", "AKT1"])

        assert results["TP53"] == ("HGNC", "11998")
        assert results["AKT1"] == ("HGNC", "391")

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_populates_instance_cache(self):
        """Batch resolution populates the instance gene cache."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = {"11998": "TP53"}

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                extractor = self._make_extractor()
                extractor.resolve_gene_names_batch(["TP53"])

        assert "TP53" in extractor._gene_cache
        assert extractor._gene_cache["TP53"] == ("HGNC", "11998")

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_uses_instance_cache_for_known_entries(self):
        """Already-cached entries are returned without map lookup."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        extractor = self._make_extractor()
        # Pre-populate cache
        extractor._gene_cache["CACHED"] = ("HGNC", "9999")

        with mock.patch.object(cogex_mod, '_get_hgnc_symbol_map', return_value={}):
            results = extractor.resolve_gene_names_batch(["CACHED"])

        assert results["CACHED"] == ("HGNC", "9999")

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_falls_back_to_resolve_gene_name(self):
        """Unknown symbols fall back to per-gene resolve_gene_name."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        extractor = self._make_extractor()

        with mock.patch.object(cogex_mod, '_get_hgnc_symbol_map', return_value={}):
            with mock.patch.object(
                extractor, 'resolve_gene_name',
                return_value=("HGNC", "12345")
            ) as mock_resolve:
                results = extractor.resolve_gene_names_batch(["UNKNOWN"])

        assert results["UNKNOWN"] == ("HGNC", "12345")
        mock_resolve.assert_called_once_with("UNKNOWN")

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_case_insensitive_lookup(self):
        """Uppercase fallback works when original case doesn't match."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, 'INDRA_AVAILABLE', True):
            mock_hgnc = mock.MagicMock()
            mock_hgnc.hgnc_names = {"11998": "TP53"}

            with mock.patch.object(cogex_mod, 'hgnc_client', mock_hgnc):
                extractor = self._make_extractor()
                results = extractor.resolve_gene_names_batch(["tp53"])

        assert results["tp53"] == ("HGNC", "11998")

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_empty_input_returns_empty_dict(self):
        """Empty input list returns empty dict."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        with mock.patch.object(cogex_mod, '_get_hgnc_symbol_map', return_value={}):
            extractor = self._make_extractor()
            results = extractor.resolve_gene_names_batch([])

        assert results == {}

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_cache_eviction_on_overflow(self):
        """Cache evicts oldest entries when maxsize is exceeded."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        extractor = self._make_extractor()
        extractor._gene_cache_maxsize = 3

        # Pre-populate with 2 entries
        extractor._gene_cache["OLD1"] = ("HGNC", "1")
        extractor._gene_cache["OLD2"] = ("HGNC", "2")

        symbol_map = {"NEW1": "100", "NEW2": "200"}
        with mock.patch.object(cogex_mod, '_get_hgnc_symbol_map', return_value=symbol_map):
            extractor.resolve_gene_names_batch(["NEW1", "NEW2"])

        # Cache should have evicted oldest to maintain maxsize=3
        assert len(extractor._gene_cache) <= 3

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None

    def test_get_regulator_modules_uses_batch_resolution(self):
        """get_regulator_modules uses batch resolution (SEC-III-3 integration)."""
        from cliquefinder.knowledge import cogex as cogex_mod

        cogex_mod._HGNC_SYMBOL_TO_ID = None

        extractor = self._make_extractor()

        # Mock batch resolution
        batch_results = {
            "GENE1": ("HGNC", "1"),
            "GENE2": ("HGNC", "2"),
            "REG1": ("HGNC", "100"),
        }

        with mock.patch.object(
            extractor, 'resolve_gene_names_batch', return_value=batch_results
        ) as mock_batch:
            # Mock the downstream target query to return empty
            extractor.client.get_downstream_targets.return_value = []

            extractor.get_regulator_modules(
                regulators=["REG1"],
                gene_universe=["GENE1", "GENE2"],
            )

        # Should have been called twice: once for universe, once for regulators
        assert mock_batch.call_count == 2

        # Cleanup
        cogex_mod._HGNC_SYMBOL_TO_ID = None
