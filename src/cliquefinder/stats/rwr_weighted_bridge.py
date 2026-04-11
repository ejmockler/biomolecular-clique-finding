"""RWR-weighted bridge for discovery — injects RWR proximity as ROAST weights.

Wraps a ``DiscoveryBridge`` to add topology-aware weights to gene set tests.
The gene sets (which intermediary regulates which targets) come from the
inner bridge's INDRA queries. The weights come from signed RWR scores,
encoding how topologically close each target is to the disease seed.

Usage::

    bridge = DiscoveryBridge(engine, sym_to_feat, env_file=".env")
    signed_rwr = compute_signed_rwr_scores(edges, seed_gene="C9orf72")
    rwr_bridge = RWRWeightedBridge(bridge, signed_rwr, feat_to_sym)
    result = run_discovery(
        ..., test_gene_set=rwr_bridge.test_gene_set,
        get_targets=rwr_bridge.get_targets, ...
    )
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cliquefinder.stats.network_proximity import SignedRWRResult

logger = logging.getLogger(__name__)


class RWRWeightedBridge:
    """Wraps a DiscoveryBridge to inject RWR proximity as ROAST weights.

    The inner bridge handles INDRA queries and gene set membership.
    This wrapper adds per-gene weights derived from signed RWR scores
    before passing to the ROAST engine.

    Parameters
    ----------
    inner_bridge
        A ``DiscoveryBridge`` instance (already initialized with engine,
        sym_to_feat, INDRA source, etc.).
    signed_rwr
        Signed RWR result from ``compute_signed_rwr_scores()``.
    feat_to_sym
        Reverse mapping from feature ID (UniProt) to gene symbol.
        Used to look up RWR scores for features in gene sets.
    weight_mode
        Which RWR score to use as weight:
        - "combined": act + rep (total proximity, default)
        - "activation": activation subgraph only
        - "repression": repression subgraph only
    """

    def __init__(
        self,
        inner_bridge: Any,
        signed_rwr: SignedRWRResult,
        feat_to_sym: dict[str, str],
        weight_mode: str = "combined",
    ):
        self.inner = inner_bridge
        self.signed_rwr = signed_rwr
        self.feat_to_sym = feat_to_sym
        self.weight_mode = weight_mode

        # Select the score dict based on mode
        if weight_mode == "combined":
            self._scores = signed_rwr.combined_scores
        elif weight_mode == "activation":
            self._scores = signed_rwr.act_scores
        elif weight_mode == "repression":
            self._scores = signed_rwr.rep_scores
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        # Delegate engine access
        self.engine = inner_bridge.engine
        self.config = inner_bridge.config

    def get_targets(self, intermediary: str) -> list[str]:
        """Delegate to inner bridge — gene set membership is unchanged."""
        return self.inner.get_targets(intermediary)

    def test_gene_set(self, gene_ids: list[str], set_id: str) -> float:
        """Run ROAST on a gene set with RWR proximity weights.

        Gene set membership comes from the inner bridge (INDRA edges).
        Weights come from signed RWR scores (topology-aware proximity
        to the disease seed gene).
        """
        fids = [g for g in gene_ids if g in self.engine.gene_to_idx]

        if len(fids) < 2:
            return 1.0

        # Build per-gene weights from rank-normalized RWR scores.
        # Raw RWR probabilities have ~10,000:1 dynamic range across a 25K-node
        # graph, which would collapse ROAST into a single-gene test dominated
        # by the 1-2 highest-RWR genes. Rank normalization compresses to [0,1]
        # while preserving ordering.
        raw_scores = np.zeros(len(fids), dtype=np.float64)
        reliabilities = np.ones(len(fids), dtype=np.float64)
        for i, fid in enumerate(fids):
            sym = self.feat_to_sym.get(fid, "")
            raw_scores[i] = self._scores.get(sym, 0.0)
            reliabilities[i] = self._get_reliability(set_id, fid)

        # Rank-normalize within the gene set: rank / n maps to [1/n, 1]
        n = len(fids)
        order = np.argsort(raw_scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        weights = (ranks / n) * reliabilities
        # Floor to avoid zero weights
        weights = np.maximum(weights, 1.0 / (2 * n))

        result = self.engine.test_gene_set(
            gene_set=fids,
            gene_set_id=set_id,
            weights=weights,
            config=self.config,
        )
        return result.p_values.get("msq", {}).get("mixed", 1.0)

    def _get_reliability(self, intermediary: str, target_fid: str) -> float:
        """Look up edge reliability from inner bridge's metadata cache.

        Precondition: get_targets(intermediary) must have been called first
        to populate the cache. This is guaranteed by run_discovery()'s
        calling convention (get_targets before test_gene_set per intermediary).
        """
        # Lazy-build O(1) lookup index from the inner bridge's list-based cache
        if not hasattr(self, '_reliability_index'):
            self._reliability_index: dict[str, dict[str, float]] = {}

        if intermediary not in self._reliability_index:
            edge_meta = self.inner._edge_metadata_cache.get(intermediary, [])
            self._reliability_index[intermediary] = {
                m.get("target_fid", ""): m.get("reliability", 1.0)
                for m in edge_meta
            }

        return self._reliability_index.get(intermediary, {}).get(target_fid, 1.0)

    def close(self):
        """Delegate cleanup to inner bridge."""
        self.inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
