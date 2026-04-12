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
        use_recalibrated_priors: bool = True,  # use benchmark-calibrated priors
    ):
        self.engine = engine
        self.sym_to_feat = sym_to_feat
        self.env_file = str(env_file) if env_file else None
        self.min_evidence = min_evidence
        self.min_reliability = min_reliability
        self.min_sources = min_sources
        if use_recalibrated_priors:
            from indra_belief.noise_model import RECALIBRATED_PRIORS
            self._priors = RECALIBRATED_PRIORS
        else:
            self._priors = None  # uses INDRA defaults

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

                from indra_belief.noise_model import compute_edge_reliability
                edge_reliability = compute_edge_reliability(
                    _sources, _total_ev, priors=self._priors,
                )

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

    def close(self):
        """Release INDRA connection and clear cache."""
        if self._indra_source is not None:
            self._indra_source.close()
            self._indra_source = None
        self._target_cache.clear()
        self._edge_metadata_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
