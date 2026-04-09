"""Bridge between the causal-path-scoring discovery module and the
biomolecular-clique-finding validation pipeline.

Provides get_targets() and test_gene_set() callbacks that route through
the pipeline's data path: INDRA → HGNC filtering → sym_to_feat → ROAST.
This ensures the discovery module sees the SAME targets and produces
the SAME p-values as the manual validation analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class DiscoveryBridge:
    """Wraps the pipeline's INDRA + ROAST infrastructure for discovery.

    Usage:
        bridge = DiscoveryBridge(engine, sym_to_feat, env_file=".env")
        result = run_discovery(
            seed="C9orf72",
            adjacency=graph,
            test_gene_set=bridge.test_gene_set,
            get_targets=bridge.get_targets,
            ...
        )
    """

    def __init__(
        self,
        engine,  # RotationTestEngine — already fitted
        sym_to_feat: dict[str, str],
        env_file: Path | str | None = None,
        min_evidence: int = 1,  # query broadly, filter by reliability
        min_reliability: float = 0.0,  # per-edge reliability threshold (0 = no filter)
        min_sources: int = 1,  # minimum unique source APIs per edge
        roast_config=None,
        llm_scorer=None,  # Optional EvidenceScorer for LLM-based edge filtering
        min_llm_score: float = 0.0,  # Threshold when llm_scorer is provided
    ):
        self.engine = engine
        self.sym_to_feat = sym_to_feat
        self.env_file = str(env_file) if env_file else None
        self.min_evidence = min_evidence
        self.min_reliability = min_reliability
        self.min_sources = min_sources
        self.llm_scorer = llm_scorer  # None = disabled
        self.min_llm_score = min_llm_score
        self._llm_score_cache: dict[tuple, float] = {}  # (src, type, tgt, text) → score

        if roast_config is None:
            from cliquefinder.stats.rotation import RotationTestConfig, SetStatistic
            self.config = RotationTestConfig(
                statistics=[SetStatistic.MSQ], n_rotations=999, seed=42,
            )
        else:
            self.config = roast_config

        self._indra_source = None
        self._hgnc_client = None
        self._target_cache: dict[str, list[str]] = {}
        self._edge_metadata_cache: dict[str, list[dict]] = {}  # intermediary → [{sources, evidence_count, regulation_type}]

    def _ensure_indra(self):
        if self._indra_source is None:
            from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource
            self._indra_source = INDRAKnowledgeSource(
                env_file=self.env_file
            )
        if self._hgnc_client is None:
            from indra.databases import hgnc_client
            self._hgnc_client = hgnc_client

    def get_targets(self, intermediary: str) -> list[str]:
        """Resolve an intermediary's measurable targets through the pipeline.

        Routes through: INDRA get_edges → HGNC filter → sym_to_feat → engine.
        Caches per intermediary to avoid redundant INDRA queries.

        Returns feature IDs (UniProt) compatible with the ROAST engine.
        """
        if intermediary in self._target_cache:
            return self._target_cache[intermediary]

        self._ensure_indra()

        edges = self._indra_source.get_edges(
            intermediary, min_evidence=self.min_evidence,
        )

        fids = set()
        edge_meta = []
        for e in edges:
            if not self._hgnc_client.get_current_hgnc_id(e.target):
                continue
            fid = self.sym_to_feat.get(e.target)
            if fid and fid in self.engine.gene_to_idx:
                # Compute per-edge reliability from source diversity
                meta = e.metadata or {}
                _src_counts = meta.get("source_counts", {})
                if _src_counts:
                    _sources = []
                    _total_ev = 0
                    for src_name, cnt in _src_counts.items():
                        _sources.extend([src_name] * cnt)
                        _total_ev += cnt
                else:
                    _sources = list(e.sources)
                    _total_ev = e.evidence_count

                from causal_path_scoring.core.edge_reliability import compute_edge_reliability
                edge_reliability = compute_edge_reliability(_sources, _total_ev)

                n_unique_sources = len(set(s.lower() for s in _sources))
                if edge_reliability < self.min_reliability:
                    continue
                if n_unique_sources < self.min_sources:
                    continue

                fids.add(fid)
                edge_meta.append({
                    "target_fid": fid,
                    "target_symbol": e.target,
                    "regulation_type": meta.get("regulation_type", "unknown"),
                    "sources": list(e.sources),
                    "evidence_count": e.evidence_count,
                    "source_counts": _src_counts,
                    "reliability": edge_reliability,
                    "n_unique_sources": n_unique_sources,
                })

        result = sorted(fids)
        self._target_cache[intermediary] = result
        self._edge_metadata_cache[intermediary] = edge_meta
        return result

    def test_gene_set(self, gene_ids: list[str], set_id: str) -> float:
        """Run ROAST on a gene set.

        Accepts feature IDs (UniProt) as returned by get_targets().
        """
        fids = [g for g in gene_ids if g in self.engine.gene_to_idx]

        if len(fids) < 2:
            return 1.0

        result = self.engine.test_gene_set(
            gene_set=fids, gene_set_id=set_id, config=self.config,
        )
        return result.p_values.get("msq", {}).get("mixed", 1.0)

    def llm_score_edges(
        self,
        intermediary: str,
        fetch_evidence_text: bool = True,
    ) -> dict[str, dict]:
        """Score each edge for an intermediary using the LLM (optional).

        Returns a dict mapping target symbol → {
          "edge_score": float in [0, 1],
          "per_evidence": [...],
          "n_evidence": int,
        }

        This is an OPT-IN post-discovery analysis. Requires:
        - self.llm_scorer to be set (EvidenceScorer instance)
        - get_targets() to have been called for this intermediary

        Note: fetches evidence text from INDRA, which adds I/O cost.
        Caches scores by (source, type, target, text) tuple.
        """
        if self.llm_scorer is None:
            return {}

        edge_meta = self._edge_metadata_cache.get(intermediary, [])
        if not edge_meta:
            return {}

        if fetch_evidence_text:
            self._ensure_indra()

        results = {}
        for em in edge_meta:
            target = em["target_symbol"]
            stmt_type = em.get("regulation_type", "unknown")

            # Fetch evidence text if needed
            evidences = em.get("evidences", [])
            if not evidences and fetch_evidence_text:
                # The current INDRAKnowledgeSource doesn't return evidence text;
                # we'd need to extend it or query CoGEx directly.
                # For now, skip LLM scoring if no text available.
                results[target] = {
                    "edge_score": None,
                    "note": "evidence text not available",
                    "n_evidence": 0,
                }
                continue

            edge_result = self.llm_scorer.score_edge(
                source=intermediary,
                target=target,
                stmt_type=stmt_type,
                evidences=evidences,
            )
            results[target] = edge_result

        return results

    def filter_targets_by_llm(
        self,
        intermediary: str,
        min_edge_score: float | None = None,
    ) -> list[str]:
        """Return target feature IDs filtered by LLM edge score.

        Args:
            intermediary: the upstream regulator
            min_edge_score: threshold (uses self.min_llm_score if None)

        Returns:
            List of target feature IDs where edge_score >= threshold.
            Targets without scorable evidence are INCLUDED (benefit of doubt).
        """
        threshold = min_edge_score if min_edge_score is not None else self.min_llm_score
        if self.llm_scorer is None or threshold <= 0:
            # No filtering — return full target list
            return self._target_cache.get(intermediary, [])

        llm_results = self.llm_score_edges(intermediary)
        edge_meta = self._edge_metadata_cache.get(intermediary, [])

        filtered_fids = []
        for em in edge_meta:
            target = em["target_symbol"]
            fid = em["target_fid"]
            score_info = llm_results.get(target, {})
            score = score_info.get("edge_score")
            if score is None or score >= threshold:
                filtered_fids.append(fid)
        return sorted(filtered_fids)

    def close(self):
        """Release INDRA connection and clear cache."""
        if self._indra_source is not None:
            self._indra_source.close()
            self._indra_source = None
        self._target_cache.clear()
        self._edge_metadata_cache.clear()
        self._llm_score_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
