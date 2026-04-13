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


class TestComposedScorerWiring:
    """Test ComposedBeliefScorer integration in DiscoveryBridge."""

    def test_default_composed_scorer_is_none(self):
        """Backwards compatibility: composed_scorer defaults to None."""
        engine = MagicMock()
        bridge = DiscoveryBridge(engine, {})
        assert bridge.composed_scorer is None

    def test_accepts_mock_composed_scorer(self):
        """DiscoveryBridge stores a provided composed_scorer."""
        engine = MagicMock()
        mock_scorer = MagicMock()
        bridge = DiscoveryBridge(engine, {}, composed_scorer=mock_scorer)
        assert bridge.composed_scorer is mock_scorer

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_composed_scoring_called_on_get_targets(self, mock_ensure):
        """When scorer is provided, _compute_composed_scores is called."""
        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0, "P35637": 1}
        sym_to_feat = {"SOD1": "P00441", "FUS": "P35637"}

        mock_scorer = MagicMock()
        bridge = DiscoveryBridge(
            engine, sym_to_feat, env_file=None,
            composed_scorer=mock_scorer,
        )

        # Mock INDRA source returning edges with metadata
        mock_edge = MagicMock()
        mock_edge.target = "SOD1"
        mock_edge.sources = ["reach", "sparser"]
        mock_edge.evidence_count = 3
        mock_edge.metadata = {
            "source_counts": {"reach": 2, "sparser": 1},
            "regulation_type": "activation",
            "stmt_hash": 12345,
        }

        bridge._indra_source = MagicMock()
        bridge._indra_source.get_edges.return_value = [mock_edge]
        bridge._indra_source.fetch_evidence_text.return_value = {
            12345: [
                {"text": "SOD1 is activated by VPS4A", "source_api": "reach", "pmid": "123"},
            ],
        }
        bridge._hgnc_client = MagicMock()
        bridge._hgnc_client.get_current_hgnc_id.return_value = "999"

        # Mock composed score result
        mock_composed = MagicMock()
        mock_composed.belief = 0.85
        mock_composed.parametric_only = 0.80
        mock_composed.n_total = 1
        mock_composed.has_llm_scores = False
        mock_scorer.score_edge.return_value = mock_composed

        # Mock the causal_path_scoring dependency used inside get_targets
        mock_reliability_module = MagicMock()
        mock_reliability_module.compute_edge_reliability = MagicMock(return_value=0.7)
        with patch.dict("sys.modules", {
            "causal_path_scoring": MagicMock(),
            "causal_path_scoring.core": MagicMock(),
            "causal_path_scoring.core.edge_reliability": mock_reliability_module,
        }):
            targets = bridge.get_targets("VPS4A")

        assert "P00441" in targets
        # Verify scorer was called
        mock_scorer.score_edge.assert_called_once()
        # Verify composed_belief was stored in edge metadata
        edge_meta = bridge._edge_metadata_cache["VPS4A"]
        assert len(edge_meta) == 1
        assert edge_meta[0]["composed_belief"] == 0.85
        assert edge_meta[0]["composed_parametric_only"] == 0.80
        assert edge_meta[0]["composed_has_llm"] is False

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_no_scoring_without_scorer(self, mock_ensure):
        """When composed_scorer is None, no scoring happens."""
        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0}
        sym_to_feat = {"SOD1": "P00441"}

        bridge = DiscoveryBridge(
            engine, sym_to_feat, env_file=None,
            composed_scorer=None,
        )

        mock_edge = MagicMock()
        mock_edge.target = "SOD1"
        mock_edge.sources = ["reach"]
        mock_edge.evidence_count = 1
        mock_edge.metadata = {
            "source_counts": {"reach": 1},
            "regulation_type": "activation",
            "stmt_hash": 99,
        }

        bridge._indra_source = MagicMock()
        bridge._indra_source.get_edges.return_value = [mock_edge]
        bridge._hgnc_client = MagicMock()
        bridge._hgnc_client.get_current_hgnc_id.return_value = "999"

        # Mock the causal_path_scoring dependency used inside get_targets
        mock_reliability_module = MagicMock()
        mock_reliability_module.compute_edge_reliability = MagicMock(return_value=0.7)
        with patch.dict("sys.modules", {
            "causal_path_scoring": MagicMock(),
            "causal_path_scoring.core": MagicMock(),
            "causal_path_scoring.core.edge_reliability": mock_reliability_module,
        }):
            targets = bridge.get_targets("VPS4A")

        assert "P00441" in targets
        # fetch_evidence_text should NOT be called
        bridge._indra_source.fetch_evidence_text.assert_not_called()
        # No composed_belief key in metadata
        edge_meta = bridge._edge_metadata_cache["VPS4A"]
        assert "composed_belief" not in edge_meta[0]

    def test_score_edges_with_llm_stub_returns_empty(self):
        """The LLM scoring stub returns an empty dict."""
        engine = MagicMock()
        mock_scorer = MagicMock()
        bridge = DiscoveryBridge(engine, {}, composed_scorer=mock_scorer)
        result = bridge.score_edges_with_llm("GENE1", llm_client=MagicMock())
        assert result == {}

    def test_score_edges_with_llm_no_client_returns_empty(self):
        """score_edges_with_llm returns empty when llm_client is None."""
        engine = MagicMock()
        bridge = DiscoveryBridge(engine, {}, composed_scorer=MagicMock())
        result = bridge.score_edges_with_llm("GENE1", llm_client=None)
        assert result == {}

    def test_close_clears_evidence_text_cache(self):
        """close() clears the evidence text cache."""
        engine = MagicMock()
        bridge = DiscoveryBridge(engine, {})
        bridge._evidence_text_cache["GENE1"] = {123: [{"text": "foo"}]}
        bridge.close()
        assert bridge._evidence_text_cache == {}


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
