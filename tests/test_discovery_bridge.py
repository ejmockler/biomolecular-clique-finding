"""Tests for DiscoveryBridge pipeline integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from cliquefinder.stats.discovery_bridge import DiscoveryBridge


class TestDiscoveryBridgeTargets:
    """Test get_targets resolution chain."""

    def _make_bridge(self):
        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0, "P35637": 1, "P10636": 2}
        sym_to_feat = {"SOD1": "P00441", "FUS": "P35637", "MAPT": "P10636", "MISSING": "PXXXXX"}
        return DiscoveryBridge(engine, sym_to_feat, env_file=None, min_evidence=2)

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_get_targets_filters_to_engine(self, mock_ensure):
        bridge = self._make_bridge()
        # Mock INDRA edges
        mock_edge1 = MagicMock()
        mock_edge1.target = "SOD1"
        mock_edge2 = MagicMock()
        mock_edge2.target = "MISSING"  # not in engine.gene_to_idx
        mock_edge3 = MagicMock()
        mock_edge3.target = "NONHGNC"  # will fail HGNC check

        bridge._indra_source = MagicMock()
        bridge._indra_source.get_edges.return_value = [mock_edge1, mock_edge2, mock_edge3]
        bridge._hgnc_client = MagicMock()
        bridge._hgnc_client.get_current_hgnc_id.side_effect = lambda x: "123" if x != "NONHGNC" else None

        targets = bridge.get_targets("VPS4A")
        assert "P00441" in targets  # SOD1 resolves
        assert "PXXXXX" not in targets  # MISSING not in engine
        assert len(targets) == 1

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_caching(self, mock_ensure):
        bridge = self._make_bridge()
        bridge._indra_source = MagicMock()
        bridge._indra_source.get_edges.return_value = []
        bridge._hgnc_client = MagicMock()

        bridge.get_targets("GENE1")
        bridge.get_targets("GENE1")
        # Should only call get_edges once (cached)
        assert bridge._indra_source.get_edges.call_count == 1

    def test_cache_clears_on_close(self):
        bridge = self._make_bridge()
        bridge._target_cache["GENE1"] = ["P00441"]
        bridge.close()
        assert bridge._target_cache == {}


class TestDiscoveryBridgeROAST:
    """Test test_gene_set callback."""

    def test_accepts_feature_ids(self):
        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0, "P35637": 1}
        mock_result = MagicMock()
        mock_result.p_values = {"msq": {"mixed": 0.05}}
        engine.test_gene_set.return_value = mock_result

        bridge = DiscoveryBridge(engine, {})
        p = bridge.test_gene_set(["P00441", "P35637"], "test")
        assert p == 0.05

    def test_filters_unknown_ids(self):
        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0}
        bridge = DiscoveryBridge(engine, {})

        # Only 1 valid ID → returns 1.0 (need ≥2)
        p = bridge.test_gene_set(["P00441", "UNKNOWN"], "test")
        assert p == 1.0


class TestContextManager:
    def test_enter_exit(self):
        engine = MagicMock()
        bridge = DiscoveryBridge(engine, {})
        with bridge as b:
            assert b is bridge
        assert bridge._target_cache == {}

    def test_default_min_evidence(self):
        engine = MagicMock()
        bridge = DiscoveryBridge(engine, {})
        assert bridge.min_evidence == 2  # matches pipeline default
