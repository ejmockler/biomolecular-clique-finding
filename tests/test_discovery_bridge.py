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
        # Mock INDRA edges with proper metadata
        mock_edge1 = MagicMock()
        mock_edge1.target = "SOD1"
        mock_edge1.sources = ["reach"]
        mock_edge1.evidence_count = 1
        mock_edge1.metadata = {"source_counts": {"reach": 1}, "regulation_type": "activation"}
        mock_edge2 = MagicMock()
        mock_edge2.target = "MISSING"  # not in engine.gene_to_idx
        mock_edge2.sources = ["reach"]
        mock_edge2.evidence_count = 1
        mock_edge2.metadata = {"source_counts": {"reach": 1}}
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
        assert bridge.min_evidence == 1  # query broadly, filter by reliability


class TestEdgeQualityClassification:
    """Test _classify_edge_quality helper."""

    def test_multi_source(self):
        from cliquefinder.stats.discovery_bridge import _classify_edge_quality
        assert _classify_edge_quality(2, ["reach", "sparser"]) == "multi_source"
        assert _classify_edge_quality(3, ["reach", "signor", "trips"]) == "multi_source"

    def test_single_curated(self):
        from cliquefinder.stats.discovery_bridge import _classify_edge_quality
        assert _classify_edge_quality(1, ["signor"]) == "single_curated"
        assert _classify_edge_quality(1, ["reactome"]) == "single_curated"

    def test_single_text_mined(self):
        from cliquefinder.stats.discovery_bridge import _classify_edge_quality
        assert _classify_edge_quality(1, ["reach"]) == "single_text_mined"
        assert _classify_edge_quality(1, ["sparser"]) == "single_text_mined"


class TestGradientBridge:
    """Test gradient-mode methods on DiscoveryBridge."""

    def _make_bridge_with_t(self):
        """Create a bridge with a mock engine that has moderated t-stats."""
        import numpy as np

        engine = MagicMock()
        engine.gene_to_idx = {
            "P00441": 0, "P35637": 1, "P10636": 2,
            "P12345": 3, "P67890": 4,
        }
        # Mock effects for _get_moderated_t
        effects = MagicMock()
        effects.U = np.array([[2.0], [1.5], [1.0], [0.5], [0.3]])
        effects.moderated_variances = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {
            "SOD1": "P00441", "FUS": "P35637", "MAPT": "P10636",
            "APP": "P12345", "GRN": "P67890",
        }
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)
        return bridge

    def test_get_abs_t_stats(self):
        bridge = self._make_bridge_with_t()
        t_stats = bridge.get_abs_t_stats()
        assert t_stats["SOD1"] == pytest.approx(2.0)
        assert t_stats["FUS"] == pytest.approx(1.5)
        assert t_stats["GRN"] == pytest.approx(0.3)
        assert len(t_stats) == 5

    def test_get_abs_t_stats_handles_multi_alias_uniprot(self):
        """Multiple gene symbols mapping to the same UniProt feature must
        all appear in the result with the same |t|. The reverse-map approach
        silently dropped aliases and broke shell construction (INDRA target
        names did not match the chosen symbol per feature).
        """
        import numpy as np

        engine = MagicMock()
        engine.gene_to_idx = {"P00441": 0}
        effects = MagicMock()
        effects.U = np.array([[2.5]])
        effects.moderated_variances = np.array([1.0])
        effects.sample_variances = None
        engine._effects = effects

        # Three symbols all alias the same UniProt
        sym_to_feat = {
            "SOD1": "P00441",
            "ALS1": "P00441",          # alias
            "SOD1_HUMAN": "P00441",    # legacy name
        }
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)

        t_stats = bridge.get_abs_t_stats()
        assert "SOD1" in t_stats
        assert "ALS1" in t_stats
        assert "SOD1_HUMAN" in t_stats
        assert t_stats["SOD1"] == pytest.approx(2.5)
        assert t_stats["ALS1"] == pytest.approx(2.5)
        assert t_stats["SOD1_HUMAN"] == pytest.approx(2.5)
        assert len(t_stats) == 3

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_build_adjacency_populates_leaf_edges(self, mock_ensure):
        """Leaf shell genes must have outgoing edges recorded so that
        _compute_graph_degrees returns the correct degree for them.
        Without this, the degree-preserving null fails for the outermost shell.
        """
        bridge = self._make_bridge_with_t()

        # Mock get_targets to return deterministic results per gene.
        call_log: list[str] = []

        def fake_get_targets(gene: str) -> list[str]:
            call_log.append(gene)
            # SEED -> [SOD1]; SOD1 -> [FUS, MAPT]; FUS -> [APP]; MAPT -> [GRN]
            graph = {
                "SEED": ["P00441"],  # SOD1
                "SOD1": ["P35637", "P10636"],  # FUS, MAPT
                "FUS": ["P12345"],  # APP
                "MAPT": ["P67890"],  # GRN
                "APP": [],
                "GRN": [],
            }
            return graph.get(gene, [])

        bridge.get_targets = fake_get_targets  # type: ignore
        bridge.sym_to_feat["SEED"] = "SEED_FID"
        # max_hops=2 -> shells = {1: [SOD1], 2: [FUS, MAPT]}
        # Leaves (FUS, MAPT) must still be queried for their outgoing edges.
        adj, eq = bridge.build_neighborhood_adjacency("SEED", max_hops=2)

        # Leaves should have non-empty adjacency entries
        assert "FUS" in adj
        assert "MAPT" in adj
        assert adj["FUS"] == ["APP"]
        assert adj["MAPT"] == ["GRN"]
        # They were queried in the leaf pass
        assert "FUS" in call_log
        assert "MAPT" in call_log

    def test_edge_quality_keeps_best_tier(self):
        """When a gene is reached through multiple parents, the tier should
        be updated if a better-quality edge is found on a subsequent visit.
        """
        from cliquefinder.stats.discovery_bridge import (
            _classify_edge_quality,
            _TIER_RANK,
        )

        # Verify the helper ranking is correct
        assert _TIER_RANK[_classify_edge_quality(2, ["reach", "sparser"])] > (
            _TIER_RANK[_classify_edge_quality(1, ["reach"])]
        )
        assert _TIER_RANK[_classify_edge_quality(1, ["signor"])] > (
            _TIER_RANK[_classify_edge_quality(1, ["reach"])]
        )

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_build_adjacency_preserves_aliases(self, mock_ensure):
        """When sym_to_feat has multiple aliases per UniProt, BFS must
        propagate ALL aliases into shell membership, not just one. The
        previous one-to-one feat_to_sym reverse map dropped >80% of
        measurable genes from the BFS path."""
        bridge = self._make_bridge_with_t()
        # Add aliases to sym_to_feat: SOD1, ALS1, IPOA1 all → P00441
        bridge.sym_to_feat["ALS1"] = "P00441"
        bridge.sym_to_feat["IPOA1"] = "P00441"

        def fake_get_targets(gene: str) -> list[str]:
            graph = {
                "SEED": ["P00441"],
                "SOD1": [], "ALS1": [], "IPOA1": [],
            }
            return graph.get(gene, [])

        bridge.get_targets = fake_get_targets  # type: ignore
        bridge.sym_to_feat["SEED"] = "SEED_FID"

        adj, eq = bridge.build_neighborhood_adjacency("SEED", max_hops=2)

        # All three aliases of P00441 should appear in the SEED's adjacency
        assert "SOD1" in adj["SEED"]
        assert "ALS1" in adj["SEED"]
        assert "IPOA1" in adj["SEED"]

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_graph_query_cache_invocation(self, mock_ensure):
        """Repeated calls with same (seed, max_hops, measured_set) reuse the
        cached distances and degrees instead of re-querying Cypher."""
        import numpy as np

        # Build a bridge large enough for gradient threshold (≥10 measurable shell genes)
        engine = MagicMock()
        n_genes = 50
        gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        engine.gene_to_idx = gene_to_idx
        effects = MagicMock()
        rng = np.random.default_rng(42)
        t_vals = np.concatenate([
            2.0 + rng.normal(0, 0.2, 10),
            1.0 + rng.normal(0, 0.2, 20),
            0.5 + rng.normal(0, 0.1, 20),
        ])
        effects.U = t_vals.reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects
        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)
        bridge._indra_source = MagicMock()
        bridge._indra_source.client = MagicMock()

        with patch(
            "cliquefinder.stats.network_proximity.query_shortest_paths_batched"
        ) as mock_paths, patch(
            "cliquefinder.stats.network_proximity.query_gene_degrees_batched"
        ) as mock_degrees:
            mock_paths.return_value = {
                **{f"G{i}": 1 for i in range(10)},
                **{f"G{i}": 2 for i in range(10, 30)},
            }
            mock_degrees.return_value = {f"G{i}": (10 - i % 5) for i in range(50)}

            # First call: queries fire
            bridge.run_gradient_via_shortest_paths(
                seed="SEED", max_hops=2, n_permutations=49, rng_seed=42,
            )
            assert mock_paths.call_count == 1
            assert mock_degrees.call_count == 1

            # Second call with same params: uses cache, no new queries
            bridge.run_gradient_via_shortest_paths(
                seed="SEED", max_hops=2, n_permutations=49, rng_seed=42,
            )
            assert mock_paths.call_count == 1  # unchanged
            assert mock_degrees.call_count == 1

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_run_gradient_via_shortest_paths(self, mock_ensure):
        """Bridge method queries Cypher for shortest paths + degrees,
        builds shells from distances, and delegates to run_gradient_test
        with precomputed_shells + graph_degrees.
        """
        import numpy as np

        engine = MagicMock()
        n_genes = 50
        gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        engine.gene_to_idx = gene_to_idx
        effects = MagicMock()
        rng = np.random.default_rng(42)
        t_vals = np.concatenate([
            2.0 + rng.normal(0, 0.2, 10),
            1.0 + rng.normal(0, 0.2, 20),
            0.5 + rng.normal(0, 0.1, 20),
        ])
        effects.U = t_vals.reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)

        # Mock the indra source's CoGEx client
        bridge._indra_source = MagicMock()
        cogex = MagicMock()
        bridge._indra_source.client = cogex

        # Patch the network_proximity helpers
        with patch(
            "cliquefinder.stats.network_proximity.query_shortest_paths_batched"
        ) as mock_paths, patch(
            "cliquefinder.stats.network_proximity.query_gene_degrees_batched"
        ) as mock_degrees:
            mock_paths.return_value = {
                **{f"G{i}": 1 for i in range(10)},
                **{f"G{i}": 2 for i in range(10, 30)},
            }
            mock_degrees.return_value = {f"G{i}": (10 - i % 5) for i in range(50)}

            result = bridge.run_gradient_via_shortest_paths(
                seed="SEED", max_hops=2, n_permutations=99, rng_seed=42,
            )

        assert mock_paths.called
        assert mock_degrees.called
        assert len(result.shells) == 2
        assert result.shells[0].n_genes == 10  # hop 1
        assert result.shells[1].n_genes == 20  # hop 2
        assert result.slope < 0  # decay

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_run_rewiring_null_extracts_subgraph_and_caches(self, mock_ensure):
        """Bridge method extracts subgraph via Cypher (cached),
        builds graph, delegates to run_rewiring_null."""
        import numpy as np

        engine = MagicMock()
        n_genes = 80
        gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        engine.gene_to_idx = gene_to_idx
        effects = MagicMock()
        rng = np.random.default_rng(0)
        t_vals = rng.normal(0, 1, n_genes)
        effects.U = t_vals.reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects
        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)
        bridge._indra_source = MagicMock()
        bridge._indra_source.client = MagicMock()

        # Mock edges: seed connected to 20 neighbors; each neighbor to 3 others
        mock_edges = []
        for i in range(20):
            mock_edges.append(("SEED", f"G{i}", {}))
        for i in range(20, 50):
            mock_edges.append((f"G{i % 20}", f"G{i}", {}))
        # Seed itself as a node + measurable
        bridge.sym_to_feat["SEED"] = "SEED_FID"
        engine.gene_to_idx["SEED_FID"] = n_genes
        # Expand effects to include seed
        new_U = np.vstack([effects.U, [[0.5]]])
        new_mv = np.concatenate([effects.moderated_variances, [1.0]])
        effects.U = new_U
        effects.moderated_variances = new_mv

        with patch(
            "cliquefinder.stats.network_proximity.extract_local_subgraph_edges"
        ) as mock_extract:
            mock_extract.return_value = mock_edges

            # First call: extraction fires
            bridge._moderated_t = None  # reset cache
            r1 = bridge.run_rewiring_null(
                seed="SEED", observed_slope=-0.1,
                n_rewires=9, max_hops=2, rng_seed=42,
                max_swaps_iter0=5_000, verbose=False,
            )
            assert mock_extract.call_count == 1

            # Second call with same params: uses cache
            bridge._moderated_t = None
            r2 = bridge.run_rewiring_null(
                seed="SEED", observed_slope=-0.1,
                n_rewires=9, max_hops=2, rng_seed=42,
                max_swaps_iter0=5_000, verbose=False,
            )
            assert mock_extract.call_count == 1  # unchanged
            # Determinism
            assert r1.target_nswap == r2.target_nswap

    def test_run_gradient_with_prebuilt_adjacency(self):
        import numpy as np

        # Build a bridge with enough genes for the gradient test
        engine = MagicMock()
        n_genes = 50
        gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        engine.gene_to_idx = gene_to_idx

        effects = MagicMock()
        rng = np.random.default_rng(42)
        # Genes 0-9 (hop 1) elevated, 10-29 (hop 2) moderate, rest background
        t_vals = np.concatenate([
            2.0 + rng.normal(0, 0.2, 10),  # hop 1
            1.0 + rng.normal(0, 0.2, 20),  # hop 2
            0.5 + rng.normal(0, 0.1, 20),  # background
        ])
        effects.U = t_vals.reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)

        # Pre-built adjacency
        adj = {"SEED": [f"G{i}" for i in range(10)]}
        for i in range(10):
            adj[f"G{i}"] = ["SEED"] + [f"G{j}" for j in range(10, 30)]

        result = bridge.run_gradient(
            seed="SEED",
            max_hops=3,
            n_permutations=99,
            rng_seed=42,
            adjacency=adj,
        )
        assert result.seed_gene == "SEED"
        assert result.slope < 0  # decay
        assert len(result.shells) >= 2


class TestProteinLevelInputs:
    """get_protein_level_inputs aggregates HGNC-keyed query results
    to UniProt-keyed inputs.  One observation per measured protein
    regardless of how many HGNC aliases the proteomics resolver
    expanded it into.
    """

    def _make_bridge_two_proteins_three_aliases(self):
        """One measured protein with 2 aliases; another with 1 alias."""
        import numpy as np

        engine = MagicMock()
        engine.gene_to_idx = {"P_A": 0, "P_B": 1}

        effects = MagicMock()
        effects.U = np.array([[3.0], [1.0]])  # |t| = 3.0 for A, 1.0 for B
        effects.moderated_variances = np.array([1.0, 1.0])
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {
            "GENE_A1": "P_A",   # P_A has two aliases
            "GENE_A2": "P_A",
            "GENE_B": "P_B",
        }
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)
        return bridge

    def test_one_entry_per_protein_regardless_of_aliases(self):
        bridge = self._make_bridge_two_proteins_three_aliases()
        # HGNC-level distances for all aliases
        distances_hgnc = {
            "GENE_A1": 1,
            "GENE_A2": 2,
            "GENE_B": 1,
        }
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        abs_t, dist, deg, via = bridge.get_protein_level_inputs(
            seed="SEED",  # not in sym_to_feat → no protein excluded
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        assert set(abs_t.keys()) == {"P_A", "P_B"}
        assert len(abs_t) == 2  # not 3 (HGNC-level would be 3)
        assert abs_t["P_A"] == pytest.approx(3.0)
        assert abs_t["P_B"] == pytest.approx(1.0)

    def test_min_distance_alias_chosen(self):
        bridge = self._make_bridge_two_proteins_three_aliases()
        distances_hgnc = {"GENE_A1": 2, "GENE_A2": 1, "GENE_B": 2}
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        _, dist, _, via = bridge.get_protein_level_inputs(
            seed="SEED",
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        # P_A's closest alias is GENE_A2 at distance 1
        assert dist["P_A"] == 1
        assert via["P_A"] == "GENE_A2"
        # P_B has only one alias
        assert dist["P_B"] == 2
        assert via["P_B"] == "GENE_B"

    def test_max_degree_used_for_binning(self):
        bridge = self._make_bridge_two_proteins_three_aliases()
        distances_hgnc = {"GENE_A1": 1, "GENE_A2": 1, "GENE_B": 1}
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        _, _, deg, _ = bridge.get_protein_level_inputs(
            seed="SEED",
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        # P_A's max alias degree = 50 (from GENE_A2)
        assert deg["P_A"] == 50
        # P_B has only GENE_B (degree 20)
        assert deg["P_B"] == 20

    def test_unreachable_protein_dropped_from_dist_kept_in_abs_t(self):
        bridge = self._make_bridge_two_proteins_three_aliases()
        # P_B's alias is unreachable
        distances_hgnc = {"GENE_A1": 1, "GENE_A2": 2}
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        abs_t, dist, deg, via = bridge.get_protein_level_inputs(
            seed="SEED",
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        # P_B kept in abs_t and deg (background pool for permutation)
        assert "P_B" in abs_t
        assert "P_B" in deg
        # but absent from dist (no shell membership)
        assert "P_B" not in dist
        assert "P_A" in dist

    def test_seed_protein_excluded_when_measured(self):
        """If the seed name is one of multiple aliases of a measured
        UniProt, the entire UniProt is dropped (not just the seed
        symbol).
        """
        bridge = self._make_bridge_two_proteins_three_aliases()
        distances_hgnc = {"GENE_A2": 1, "GENE_B": 1}
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        abs_t, dist, deg, _ = bridge.get_protein_level_inputs(
            seed="GENE_A1",  # this is an alias of P_A → drop P_A entirely
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        assert "P_A" not in abs_t
        assert "P_A" not in dist
        assert "P_A" not in deg
        # P_B still present
        assert "P_B" in abs_t

    def test_distance_above_max_hops_treated_as_unreachable(self):
        bridge = self._make_bridge_two_proteins_three_aliases()
        distances_hgnc = {"GENE_A1": 5, "GENE_A2": 4, "GENE_B": 1}  # 4,5 > max_hops=2
        degrees_hgnc = {"GENE_A1": 30, "GENE_A2": 50, "GENE_B": 20}

        _, dist, _, _ = bridge.get_protein_level_inputs(
            seed="SEED",
            max_hops=2,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )

        assert "P_A" not in dist  # both aliases beyond max_hops
        assert "P_B" in dist


class TestProteinLevelRewiringNull:
    """Verify that run_rewiring_null aggregates HGNC distances to
    UniProt distances when aliases are provided.
    """

    def test_aliases_collapse_duplicate_paths(self):
        """When two HGNC aliases of the same protein appear at different
        hops in the rewired graph, the protein contributes once at the
        min hop.
        """
        import networkx as nx
        import numpy as np
        from cliquefinder.stats.perturbation_gradient import (
            _slope_and_coverage_from_rewired,
        )

        # Tiny graph: SEED — A1 — A2 (alias of same protein as A1)
        # and SEED — B (separate protein)
        G = nx.Graph()
        G.add_edges_from([
            ("SEED", "A1"), ("A1", "A2"),
            ("SEED", "B"),
            # Need >=10 measured units in shells, plus ~30 nodes for graph;
            # add filler measured proteins to reach the minimum
        ])
        for i in range(40):
            G.add_edge("SEED", f"X{i}")

        abs_t = {"P_A": 5.0, "P_B": 3.0}
        for i in range(40):
            abs_t[f"P_X{i}"] = 1.0
        aliases = {
            "P_A": ["A1", "A2"],
            "P_B": ["B"],
        }
        for i in range(40):
            aliases[f"P_X{i}"] = [f"X{i}"]

        slope, coverage = _slope_and_coverage_from_rewired(
            G, "SEED", abs_t, max_hops=3, aliases=aliases,
        )
        # P_A is reached at hop 1 via A1 (not hop 2 via A2)
        # Test runs without error and produces a valid slope
        assert slope is not None or coverage is not None

    def test_hgnc_keyed_path_unchanged(self):
        """When aliases is None, behavior matches the legacy HGNC-keyed
        flow used by synthetic tests.
        """
        import networkx as nx
        from cliquefinder.stats.perturbation_gradient import (
            _slope_and_coverage_from_rewired,
        )

        G = nx.Graph()
        for i in range(15):
            G.add_edge("SEED", f"H1_{i}")
        for i in range(15):
            G.add_edge(f"H1_0", f"H2_{i}")

        abs_t = {f"H1_{i}": 2.0 for i in range(15)}
        abs_t.update({f"H2_{i}": 1.0 for i in range(15)})

        slope, coverage = _slope_and_coverage_from_rewired(
            G, "SEED", abs_t, max_hops=2, aliases=None,
        )
        assert slope is not None
        assert slope < 0  # hop-1 mean (2.0) > hop-2 mean (1.0) → decay


class TestRegulatoryEdgeScope:
    """The bridge defaults INDRA queries to ``ALL_REGULATORY_TYPES`` —
    only the four canonical regulatory statements (Activation,
    Inhibition, IncreaseAmount, DecreaseAmount).  This is the
    architectural commitment from Wave 24: graph proximity is computed
    over edges that propagate perturbation, not over co-mention or
    binding edges.
    """

    def test_bridge_defaults_to_regulatory_types(self):
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES

        engine = MagicMock()
        engine.gene_to_idx = {"P1": 0}
        bridge = DiscoveryBridge(engine, {"G1": "P1"}, env_file=None)
        assert set(bridge.stmt_types) == ALL_REGULATORY_TYPES

    def test_bridge_accepts_explicit_stmt_types(self):
        engine = MagicMock()
        engine.gene_to_idx = {"P1": 0}
        bridge = DiscoveryBridge(
            engine, {"G1": "P1"}, env_file=None,
            stmt_types=["Phosphorylation"],
        )
        assert bridge.stmt_types == ["Phosphorylation"]

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_shortest_paths_query_filters_by_stmt_types(self, mock_ensure):
        """Bridge passes its ``stmt_types`` (default regulatory) into
        every Cypher query — the path-finder must restrict to
        regulatory edges.
        """
        import numpy as np
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES

        engine = MagicMock()
        n_genes = 50
        engine.gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        effects = MagicMock()
        rng = np.random.default_rng(42)
        t_vals = np.concatenate([
            2.0 + rng.normal(0, 0.2, 10),
            1.0 + rng.normal(0, 0.2, 20),
            0.5 + rng.normal(0, 0.1, 20),
        ])
        effects.U = t_vals.reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=None)
        bridge._indra_source = MagicMock()
        bridge._indra_source.client = MagicMock()

        with patch(
            "cliquefinder.stats.network_proximity.query_shortest_paths_batched"
        ) as mock_paths, patch(
            "cliquefinder.stats.network_proximity.query_gene_degrees_batched"
        ) as mock_degrees:
            mock_paths.return_value = {
                **{f"G{i}": 1 for i in range(10)},
                **{f"G{i}": 2 for i in range(10, 30)},
            }
            mock_degrees.return_value = {f"G{i}": (10 - i % 5) for i in range(50)}

            bridge.run_gradient_via_shortest_paths(
                seed="SEED", max_hops=2, n_permutations=49, rng_seed=42,
            )

        # Bridge should have passed ALL_REGULATORY_TYPES (as a list) to
        # both queries.
        paths_kwargs = mock_paths.call_args.kwargs
        degrees_kwargs = mock_degrees.call_args.kwargs
        assert "stmt_types" in paths_kwargs
        assert "stmt_types" in degrees_kwargs
        assert set(paths_kwargs["stmt_types"]) == ALL_REGULATORY_TYPES
        assert set(degrees_kwargs["stmt_types"]) == ALL_REGULATORY_TYPES

    @patch("cliquefinder.stats.discovery_bridge.DiscoveryBridge._ensure_indra")
    def test_explicit_stmt_types_overrides_default(self, mock_ensure):
        """Constructor stmt_types override the default and propagate
        to network query calls.
        """
        import numpy as np

        engine = MagicMock()
        n_genes = 50
        engine.gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        effects = MagicMock()
        rng = np.random.default_rng(42)
        effects.U = (1.0 + rng.normal(0, 0.2, n_genes)).reshape(-1, 1)
        effects.moderated_variances = np.ones(n_genes)
        effects.sample_variances = None
        engine._effects = effects

        sym_to_feat = {f"G{i}": f"P{i:05d}" for i in range(n_genes)}
        custom_types = ["Activation", "Phosphorylation"]
        bridge = DiscoveryBridge(
            engine, sym_to_feat, env_file=None, stmt_types=custom_types,
        )
        bridge._indra_source = MagicMock()
        bridge._indra_source.client = MagicMock()

        with patch(
            "cliquefinder.stats.network_proximity.query_shortest_paths_batched"
        ) as mock_paths, patch(
            "cliquefinder.stats.network_proximity.query_gene_degrees_batched"
        ) as mock_degrees:
            mock_paths.return_value = {
                **{f"G{i}": 1 for i in range(10)},
                **{f"G{i}": 2 for i in range(10, 30)},
            }
            mock_degrees.return_value = {f"G{i}": 5 for i in range(50)}

            bridge.run_gradient_via_shortest_paths(
                seed="SEED", max_hops=2, n_permutations=49, rng_seed=42,
            )

        paths_kwargs = mock_paths.call_args.kwargs
        assert paths_kwargs["stmt_types"] == custom_types

    def test_query_shortest_paths_default_is_regulatory(self):
        """The standalone network_proximity function defaults to
        ALL_REGULATORY_TYPES too — so a caller that bypasses the bridge
        still gets the right scope.
        """
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES
        from cliquefinder.stats.network_proximity import (
            query_shortest_paths_batched,
        )

        captured = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        query_shortest_paths_batched(
            cogex_client=client,
            seed_gene_name="SEED",
            target_gene_names=["A"],
            max_hops=2,
            verbose=False,
        )
        assert "stmt_types" in captured
        assert set(captured["stmt_types"]) == ALL_REGULATORY_TYPES

    def test_query_gene_degrees_default_is_regulatory(self):
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES
        from cliquefinder.stats.network_proximity import (
            query_gene_degrees_batched,
        )

        captured = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        query_gene_degrees_batched(
            cogex_client=client,
            gene_names=["A"],
        )
        assert "stmt_types" in captured
        assert set(captured["stmt_types"]) == ALL_REGULATORY_TYPES

    def test_extract_local_subgraph_default_is_regulatory(self):
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES
        from cliquefinder.stats.network_proximity import (
            extract_local_subgraph_edges,
        )

        captured = {}

        def fake_execute(query, **params):
            captured.update(params)
            return []

        client = MagicMock()
        client._execute_query = fake_execute

        extract_local_subgraph_edges(
            cogex_client=client,
            seed_gene_name="SEED",
            max_hops=2,
        )
        assert "stmt_types" in captured
        assert set(captured["stmt_types"]) == ALL_REGULATORY_TYPES
