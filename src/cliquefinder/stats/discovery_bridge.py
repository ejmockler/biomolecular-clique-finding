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
    composed belief scores for each INDRA edge.

    When use_competitive=True, test_gene_set returns a competitive z-score
    p-value instead of the raw ROAST MSQ p-value. The competitive z adjusts
    for inter-gene correlation via the Camera VIF, providing better FPR
    calibration on anticonservative ROAST datasets.
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
        use_competitive: bool = False,
    ):
        self.engine = engine
        self.sym_to_feat = sym_to_feat
        self.env_file = str(env_file) if env_file else None
        self.min_evidence = min_evidence
        self.min_reliability = min_reliability
        self.min_sources = min_sources
        self.composed_scorer = composed_scorer
        self.use_competitive = use_competitive
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

        # Cache moderated t-statistics (computed lazily on first competitive test)
        self._moderated_t: np.ndarray | None = None

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

    def _get_moderated_t(self) -> np.ndarray:
        """Compute moderated t-statistics for all genes from the fitted engine.

        The moderated t for gene g is: t_g = U[g, 0] / sqrt(moderated_var[g])
        where U[:, 0] contains the contrast effect and moderated_var is the
        EB-shrunk variance.

        Returns:
            Array of moderated t-statistics (n_genes,) in engine gene order.
        """
        if self._moderated_t is not None:
            return self._moderated_t

        effects = self.engine._effects
        if effects is None:
            raise RuntimeError(
                "Engine effects not available. Ensure engine.fit() has been called."
            )

        # Use moderated variances if available, otherwise sample variances
        if effects.moderated_variances is not None:
            se = np.sqrt(effects.moderated_variances)
        else:
            se = np.sqrt(effects.sample_variances)

        # Avoid division by zero
        se_safe = np.maximum(se, 1e-10)
        self._moderated_t = effects.U[:, 0] / se_safe
        return self._moderated_t

    def test_gene_set(
        self, gene_ids: list[str], set_id: str, use_competitive: bool | None = None,
    ) -> float:
        """Run gene set test and return p-value.

        Accepts feature IDs (UniProt) as returned by get_targets().

        When use_competitive is True (or self.use_competitive is True and
        use_competitive is not explicitly False), uses the competitive z-score
        with Camera VIF correction instead of the raw ROAST p-value.

        Args:
            gene_ids: Feature IDs (e.g., UniProt accessions) for the gene set.
            set_id: Identifier for this gene set.
            use_competitive: Override the instance-level use_competitive flag.
                If None, uses self.use_competitive.

        Returns:
            P-value (competitive z two-sided p-value or ROAST MSQ mixed p-value).
        """
        competitive = use_competitive if use_competitive is not None else self.use_competitive

        fids = [g for g in gene_ids if g in self.engine.gene_to_idx]

        if len(fids) < 2:
            return 1.0

        if competitive:
            return self._test_competitive(fids, set_id)

        result = self.engine.test_gene_set(
            gene_set=fids, gene_set_id=set_id, config=self.config,
        )
        return result.p_values.get("msq", {}).get("mixed", 1.0)

    def _test_competitive(self, fids: list[str], set_id: str) -> float:
        """Compute competitive z-score p-value for a gene set.

        Uses the moderated t-statistics from the fitted ROAST engine and
        optionally estimates VIF from the inter-gene correlation matrix.

        Args:
            fids: Filtered feature IDs (already verified in engine).
            set_id: Gene set identifier (for logging).

        Returns:
            Two-sided p-value from the competitive z-score.
        """
        from .competitive_z import competitive_z_test

        mod_t = self._get_moderated_t()
        target_indices = np.array(
            [self.engine.gene_to_idx[g] for g in fids], dtype=np.intp
        )

        # Estimate inter-gene correlation from MODEL RESIDUALS, not raw expression.
        # Camera (Wu & Smyth 2012) uses residuals to exclude shared treatment signal.
        # Using raw expression inflates rho_bar when targets are co-regulated,
        # causing VIF overcorrection. Residuals = data projected into the space
        # orthogonal to the design matrix (Q2 from the QR decomposition).
        corr_matrix = None
        if hasattr(self.engine, 'data') and self.engine.data is not None:
            k = len(target_indices)
            if k >= 2 and k <= 500:
                target_data = self.engine.data[target_indices, :]  # (k, n_samples)
                # Project into residual space using Q2 from the fitted model
                if (self.engine._precomputed is not None
                        and self.engine._precomputed.Q2 is not None):
                    Q2 = self.engine._precomputed.Q2
                    # Residuals = Y @ Q2 (project data into residual space)
                    target_residuals = target_data @ Q2  # (k, df_residual+1)
                    corr_matrix = np.corrcoef(target_residuals)
                else:
                    # Fallback: raw expression (less accurate but functional)
                    corr_matrix = np.corrcoef(target_data)

        z_score, p_value = competitive_z_test(mod_t, target_indices, corr_matrix)

        logger.debug(
            "Competitive z for %s: z=%.3f, p=%.4f (k=%d)",
            set_id, z_score, p_value, len(fids),
        )

        return p_value

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
