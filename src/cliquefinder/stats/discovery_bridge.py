"""Bridge between the causal-path-scoring discovery module and the
biomolecular-clique-finding validation pipeline.

Provides get_targets() and test_gene_set() callbacks that route through
the pipeline's data path: INDRA → HGNC filtering → sym_to_feat → ROAST.
This ensures the discovery module sees the SAME targets and produces
the SAME p-values as the manual validation analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


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

    Optionally accepts a ``composed_scorer``
    (:class:`indra_belief.composed_scorer.ComposedBeliefScorer`) to compute
    composed belief scores for each INDRA edge.  When provided, parametric
    belief scores (source-diversity only, no LLM verdicts) are attached to
    every edge in ``_edge_metadata_cache`` under the key
    ``"composed_belief"``.  Full LLM-augmented scoring can be triggered
    later via :meth:`score_edges_with_llm`.
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
        composed_scorer: Any | None = None,
    ):
        self.engine = engine
        self.sym_to_feat = sym_to_feat
        self.env_file = str(env_file) if env_file else None
        self.min_evidence = min_evidence
        self.min_reliability = min_reliability
        self.min_sources = min_sources
        self.composed_scorer = composed_scorer
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
        self._evidence_text_cache: dict[str, dict[int, list[dict]]] = {}  # intermediary → {stmt_hash: [{text, source_api, pmid}]}

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

        # --- Composed belief scoring (parametric only, no LLM) ---
        if self.composed_scorer is not None and edges:
            self._compute_composed_scores(intermediary, edges, edge_meta)

        return result

    def _compute_composed_scores(
        self,
        intermediary: str,
        edges: list,
        edge_meta: list[dict],
    ) -> None:
        """Compute parametric composed belief scores for cached edges.

        Fetches evidence text via INDRA DB REST, builds
        :class:`EvidenceRecord` objects (verdict=None since no LLM is
        available at this stage), and calls
        ``composed_scorer.score_edge()`` for each edge.

        Results are stored as ``"composed_belief"`` in each entry of
        ``_edge_metadata_cache[intermediary]``.
        """
        try:
            from indra_belief.composed_scorer import EvidenceRecord
        except ImportError:
            logger.warning(
                "indra_belief package not available; "
                "skipping composed belief scoring"
            )
            return

        # Fetch evidence text for all edges of this intermediary
        evidence_by_hash = self._indra_source.fetch_evidence_text(edges)
        self._evidence_text_cache[intermediary] = evidence_by_hash

        # Build a lookup from (target_symbol) to edge_meta entry
        # Each edge carries a stmt_hash in its metadata
        for meta_entry in edge_meta:
            target_sym = meta_entry["target_symbol"]
            # Find the matching KnowledgeEdge to get stmt_hash
            matching_edges = [
                e for e in edges
                if e.target == target_sym
                and (e.metadata or {}).get("stmt_hash") is not None
            ]
            if not matching_edges:
                continue

            stmt_hash = matching_edges[0].metadata["stmt_hash"]
            ev_records_raw = evidence_by_hash.get(stmt_hash, [])

            # Build EvidenceRecord objects (no LLM verdict yet)
            ev_records = []
            for ev in ev_records_raw:
                ev_records.append(EvidenceRecord(
                    source_api=ev.get("source_api", "unknown"),
                    verdict=None,  # No LLM scoring at this stage
                    regulation_type=meta_entry.get("regulation_type"),
                    stmt_hash=stmt_hash,
                ))

            # Fall back to source_counts if no evidence text was fetched
            if not ev_records:
                for src in meta_entry.get("sources", []):
                    ev_records.append(EvidenceRecord(
                        source_api=src,
                        verdict=None,
                        regulation_type=meta_entry.get("regulation_type"),
                        stmt_hash=stmt_hash,
                    ))

            if ev_records:
                try:
                    composed = self.composed_scorer.score_edge(ev_records)
                    meta_entry["composed_belief"] = composed.belief
                    meta_entry["composed_parametric_only"] = composed.parametric_only
                    meta_entry["composed_n_total"] = composed.n_total
                    meta_entry["composed_has_llm"] = composed.has_llm_scores
                except Exception:
                    logger.debug(
                        "Composed scoring failed for %s -> %s",
                        intermediary, target_sym,
                        exc_info=True,
                    )

    def score_edges_with_llm(
        self,
        intermediary: str,
        llm_client: Any = None,
    ) -> dict[str, Any]:
        """Score cached edges with LLM verdicts (stub).

        This method defines the interface for full LLM-augmented composed
        belief scoring.  The pipeline is:

        1. Retrieve cached evidence text from ``_evidence_text_cache``
        2. For each evidence sentence, call ``llm_client.score_record(text)``
           to get a verdict (``"correct"``, ``"incorrect"``, or ``"neutral"``)
        3. Build ``EvidenceRecord`` objects with actual LLM verdicts
        4. Call ``composed_scorer.score_edge(records)`` to get ``ComposedScore``

        Args:
            intermediary: Gene symbol whose edges should be scored.
            llm_client: An object with a ``score_record(text) -> str``
                method that returns an LLM verdict for a given evidence
                sentence.  Not yet implemented.

        Returns:
            ``{target_symbol: ComposedScore}`` mapping for each edge
            of the intermediary.  Currently returns an empty dict.

        .. note::
            This is a stub.  The LLM scoring pipeline involves prompt
            engineering, rate limiting, and caching that are better
            handled in a dedicated module.  This method documents the
            intended interface so the wiring is in place.
        """
        if llm_client is None or self.composed_scorer is None:
            return {}

        evidence_by_hash = self._evidence_text_cache.get(intermediary, {})
        if not evidence_by_hash:
            logger.debug(
                "No cached evidence text for %s; "
                "call get_targets() first",
                intermediary,
            )
            return {}

        # Stub: return empty — actual implementation requires LLM client
        logger.info(
            "LLM scoring stub called for %s (%d cached hashes); "
            "returning empty (not yet implemented)",
            intermediary,
            len(evidence_by_hash),
        )
        return {}

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
        self._evidence_text_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
