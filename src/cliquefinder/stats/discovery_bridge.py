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

# Edge quality tiers (highest to lowest)
_CURATED_SOURCES = frozenset({
    "signor", "biogrid", "hprd", "pid", "kegg", "reactome", "pc",
})
_TIER_RANK = {"multi_source": 3, "single_curated": 2, "single_text_mined": 1}


def _classify_edge_quality(n_unique_sources: int, sources: list[str]) -> str:
    """Classify an INDRA edge into a quality tier."""
    if n_unique_sources >= 2:
        return "multi_source"
    src_lower = {s.lower() for s in sources}
    if src_lower & _CURATED_SOURCES:
        return "single_curated"
    return "single_text_mined"


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
        # (seed, max_hops, hash(measured_symbols)) → (distances, degrees)
        # Caches contrast-invariant graph queries so triangle runs don't repeat them.
        self._graph_query_cache: dict[tuple, tuple[dict, dict]] = {}

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

    # ------------------------------------------------------------------
    # Gradient-based discovery
    # ------------------------------------------------------------------

    def get_abs_t_stats(self) -> dict[str, float]:
        """Return ``{gene_symbol: |t|}`` for every measurable gene.

        Uses moderated t-statistics from the fitted ROAST engine.

        The mapping ``self.sym_to_feat`` is many-to-one (multiple gene
        symbols can map to the same UniProt feature ID — isoforms, gene
        aliases, ambiguous mappings).  A reverse map ``feat_to_sym`` would
        collapse this and silently drop >80% of measurable symbols, which
        breaks INDRA-side lookups for adjacency/path queries that need
        every alias to resolve.

        We instead build ``fid_to_t`` once from the engine, then key the
        result by *every* symbol that maps to a measured feature.

        **Note for downstream gradient/null users:** this output is
        gene-symbol-keyed and inflates each protein measurement across
        its aliases.  For protein-level analysis (the correct unit, since
        the proteomics measures one value per UniProt), use
        :meth:`get_abs_t_per_feature` and
        :meth:`get_protein_level_inputs` instead.
        """
        mod_t = self._get_moderated_t()
        idx_to_feat = {idx: fid for fid, idx in self.engine.gene_to_idx.items()}

        fid_to_t: dict[str, float] = {}
        for idx in range(len(mod_t)):
            fid = idx_to_feat.get(idx)
            if fid is not None:
                fid_to_t[fid] = float(abs(mod_t[idx]))

        return {
            sym: fid_to_t[fid]
            for sym, fid in self.sym_to_feat.items()
            if fid in fid_to_t
        }

    def get_abs_t_per_feature(self) -> dict[str, float]:
        """Return ``{feature_id: |t|}`` — one entry per measured protein.

        Unlike :meth:`get_abs_t_stats`, this does NOT expand across HGNC
        aliases.  The proteomics measures one |t| per UniProt accession;
        this method returns exactly that, keyed by the engine's feature
        ID.  Used by protein-level gradient analyses.
        """
        mod_t = self._get_moderated_t()
        idx_to_feat = {idx: fid for fid, idx in self.engine.gene_to_idx.items()}

        fid_to_t: dict[str, float] = {}
        for idx in range(len(mod_t)):
            fid = idx_to_feat.get(idx)
            if fid is not None:
                fid_to_t[fid] = float(abs(mod_t[idx]))
        return fid_to_t

    def _build_feat_to_syms(self) -> dict[str, list[str]]:
        """Reverse ``sym_to_feat``: ``{feature_id: [hgnc_symbol, ...]}``."""
        feat_to_syms: dict[str, list[str]] = {}
        for sym, fid in self.sym_to_feat.items():
            feat_to_syms.setdefault(fid, []).append(sym)
        return feat_to_syms

    def get_protein_level_inputs(
        self,
        seed: str,
        max_hops: int,
        distances_hgnc: dict[str, int],
        degrees_hgnc: dict[str, int],
    ) -> tuple[
        dict[str, float],
        dict[str, int],
        dict[str, int],
        dict[str, str],
    ]:
        """Aggregate HGNC-keyed INDRA query results to UniProt-level.

        The proteomics measures one |t| per UniProt accession (~3,264 in
        the AnswerALS data).  UniProt → HGNC is many-to-many; the naive
        HGNC-keyed gradient treats each alias as an independent
        observation, which (a) inflates shell n by ~5×, (b) creates
        within-shell correlation in the observed graph that the null
        decorrelates by shuffling aliases independently, biasing toward
        rejection.

        This method aggregates aliases so each protein contributes once.
        For each measured UniProt:

        - ``|t|`` is the engine's protein-level statistic (no alias
          duplication).
        - **Distance** to seed = min over aliases that have a finite
          distance entry within ``max_hops``.  Proteins with no
          reachable alias are dropped from ``dist_prot`` but kept in
          ``abs_t_prot`` / ``deg_prot`` so they participate in
          degree-binned permutation as background.
        - **Degree** (for binning) = max over all aliases.  Captures
          the protein's hub-status ceiling — the relevant confound is
          "hubs are well-studied / well-curated", and the most-connected
          alias is what makes a protein hub-like in INDRA.

        The seed's UniProt (if the seed is itself measured) is excluded
        from all outputs.  Looking up via ``self.sym_to_feat[seed]``
        handles the case where the seed name is one of multiple aliases
        of a single measured protein (e.g., VCL → P18206; we drop
        P18206 and any other aliases of P18206).

        Parameters
        ----------
        seed
            Seed gene symbol (HGNC).  May or may not be in
            ``sym_to_feat``; if absent, no protein is excluded.
        max_hops
            Distances strictly greater than this are treated as
            unreachable (clipped, not used for shell membership).
        distances_hgnc
            ``{hgnc_symbol: distance}`` from
            :func:`query_shortest_paths_batched`.  Only contains
            reachable symbols (within max_hops on the query side).
        degrees_hgnc
            ``{hgnc_symbol: degree}`` from
            :func:`query_gene_degrees_batched`.  Should cover every
            HGNC alias used as a target.

        Returns
        -------
        abs_t_prot : dict[feature_id, float]
            UniProt-keyed |t| for every measured protein except seed's.
        dist_prot : dict[feature_id, int]
            UniProt-keyed min-distance via best alias.  Subset of
            ``abs_t_prot`` keys: only reachable proteins.
        deg_prot : dict[feature_id, int]
            UniProt-keyed max-degree across aliases.  Same key set as
            ``abs_t_prot``.
        via_alias : dict[feature_id, str]
            UniProt → HGNC alias used for the min-distance assignment.
            Diagnostic only; subset of ``dist_prot`` keys.
        """
        feat_to_syms = self._build_feat_to_syms()
        fid_to_t = self.get_abs_t_per_feature()

        # Seed protein: if seed is measured, drop its UniProt entirely
        seed_fid = self.sym_to_feat.get(seed)

        abs_t_prot: dict[str, float] = {}
        dist_prot: dict[str, int] = {}
        deg_prot: dict[str, int] = {}
        via_alias: dict[str, str] = {}

        for fid, aliases in feat_to_syms.items():
            if seed_fid is not None and fid == seed_fid:
                continue
            if fid not in fid_to_t:
                continue  # protein not in the engine (no t-stat)

            abs_t_prot[fid] = fid_to_t[fid]

            # Degree: max across aliases (missing alias → 0)
            alias_degs = [degrees_hgnc.get(a, 0) for a in aliases]
            deg_prot[fid] = max(alias_degs) if alias_degs else 0

            # Distance: min over reachable aliases within max_hops
            reachable = [
                (a, distances_hgnc[a])
                for a in aliases
                if a in distances_hgnc and 1 <= distances_hgnc[a] <= max_hops
            ]
            if reachable:
                best_alias, best_dist = min(reachable, key=lambda ad: ad[1])
                dist_prot[fid] = best_dist
                via_alias[fid] = best_alias

        return abs_t_prot, dist_prot, deg_prot, via_alias

    def build_neighborhood_adjacency(
        self,
        seed: str,
        max_hops: int = 3,
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Build adjacency by querying INDRA hop-by-hop from seed.

        Queries one extra hop past ``max_hops`` solely to populate
        leaf-node outgoing edges.  The gradient module still honors
        ``max_hops`` for shell construction, but the adjacency now
        contains enough edges to compute true graph degrees for every
        shell gene (including leaves).  Without this extra pass, leaf
        shell genes would have zero outgoing edges in the local
        adjacency and fall into the same degree bin as fully
        disconnected background genes, breaking the degree-preserving
        null for the outermost shell.

        Returns
        -------
        adjacency
            ``{gene: [neighbor, ...]}`` suitable for
            :func:`~perturbation_gradient.compute_hop_shells`.
        edge_quality
            ``{gene: tier}`` where tier is one of
            ``"multi_source"``, ``"single_curated"``, ``"single_text_mined"``.
            Assigned by the best incoming edge across all visits.
        """
        self._ensure_indra()

        # Precompute fid -> [all aliases] (NOT a one-to-one collapse).
        # ``sym_to_feat`` is many-to-one (multiple symbols can map to the
        # same UniProt feature ID).  Reversing with a dict comprehension
        # silently drops aliases and breaks shell membership for any
        # measured gene whose INDRA target name happens not to be the
        # surviving alias.  We must keep ALL aliases per feature.
        feat_to_syms: dict[str, list[str]] = {}
        for sym, fid in self.sym_to_feat.items():
            feat_to_syms.setdefault(fid, []).append(sym)

        adjacency: dict[str, list[str]] = {}
        edge_quality: dict[str, str] = {}
        visited: set[str] = {seed}
        frontier: set[str] = {seed}

        # Main BFS: query frontier genes, expand shells
        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for gene in frontier:
                targets = self.get_targets(gene)
                neighbor_syms: list[str] = []
                for fid in targets:
                    neighbor_syms.extend(feat_to_syms.get(fid, []))

                adjacency.setdefault(gene, []).extend(neighbor_syms)

                meta_list = self._edge_metadata_cache.get(gene, [])
                meta_by_fid = {m["target_fid"]: m for m in meta_list}
                for fid in targets:
                    syms = feat_to_syms.get(fid, [])
                    if not syms:
                        continue

                    # Update edge quality on every visit to keep best tier;
                    # apply to every alias of this feature so any of them
                    # appearing in a shell carries the right tier.
                    meta = meta_by_fid.get(fid, {})
                    n_src = meta.get("n_unique_sources", 1)
                    sources = meta.get("sources", [])
                    tier = _classify_edge_quality(n_src, sources)
                    for sym in syms:
                        existing = edge_quality.get(sym)
                        if (
                            existing is None
                            or _TIER_RANK.get(tier, 0) > _TIER_RANK.get(existing, 0)
                        ):
                            edge_quality[sym] = tier

                        # Add to frontier only on first visit (per-symbol)
                        if sym not in visited:
                            next_frontier.add(sym)
                            visited.add(sym)

            if not next_frontier:
                break
            frontier = next_frontier
            logger.info(
                "Gradient BFS hop %d: %d new genes (total visited: %d)",
                hop, len(next_frontier), len(visited),
            )

        # Extra pass: populate leaf outgoing edges for degree computation.
        # These edges do NOT extend the shells (compute_hop_shells stops at
        # max_hops) but allow _compute_graph_degrees to return the true
        # local degree for leaf nodes.
        leaf_genes = [g for g in frontier if g not in adjacency]
        if leaf_genes:
            logger.info(
                "Querying %d leaf genes for degree-preservation coverage",
                len(leaf_genes),
            )
            for gene in leaf_genes:
                targets = self.get_targets(gene)
                neighbor_syms_leaf: list[str] = []
                for fid in targets:
                    neighbor_syms_leaf.extend(feat_to_syms.get(fid, []))
                adjacency[gene] = neighbor_syms_leaf

        return adjacency, edge_quality

    def run_gradient(
        self,
        seed: str,
        max_hops: int = 3,
        n_permutations: int = 1000,
        rng_seed: int | None = 42,
        adjacency: dict[str, list[str]] | None = None,
        edge_quality: dict[str, str] | None = None,
    ):
        """Run perturbation gradient test using directed BFS through measured genes.

        **Semantics: DIRECTED, MEASURED-ONLY.**  The walk follows
        ``regulator → target`` edges from INDRA (``get_downstream_targets``)
        and only traverses through proteins present in the proteomics data.
        This matches the original "causal cascade" framing but can shrink
        shells dramatically when many INDRA intermediaries are not detected.

        For shells defined over the FULL INDRA graph using **undirected**
        shortest paths (mirroring the proximity test methodology), use
        :meth:`run_gradient_via_shortest_paths` instead.

        The two methods compute different graphs and the results are not
        directly comparable.  Pick the one that matches the biological
        question:
        - Directed-measured: "does perturbation cascade downstream from
          seed through proteins we can see?"
        - Undirected-full: "is perturbation magnitude correlated with
          knowledge-graph proximity to seed?"

        If ``adjacency`` is not provided, builds it by querying INDRA
        hop-by-hop (expensive for large neighborhoods).

        Returns
        -------
        :class:`~perturbation_gradient.GradientResult`
        """
        from .perturbation_gradient import run_gradient_test

        logger.info(
            "Running gradient (mode=DIRECTED-BFS-through-measured) "
            "from seed=%s with max_hops=%d",
            seed, max_hops,
        )

        abs_t = self.get_abs_t_stats()

        if adjacency is None:
            adjacency, edge_quality = self.build_neighborhood_adjacency(
                seed, max_hops=max_hops,
            )

        return run_gradient_test(
            adjacency=adjacency,
            abs_t_stats=abs_t,
            seed=seed,
            max_hops=max_hops,
            n_permutations=n_permutations,
            rng_seed=rng_seed,
            edge_quality=edge_quality,
        )

    def run_gradient_via_shortest_paths(
        self,
        seed: str,
        max_hops: int = 3,
        n_permutations: int = 1000,
        rng_seed: int | None = 42,
    ):
        """Run gradient test with shells from full-INDRA shortest paths.

        **Semantics: UNDIRECTED, FULL-GRAPH.**  Neo4j ``shortestPath``
        queries find paths through the entire INDRA graph from ``seed``
        to each measured gene without regard to edge direction (the
        Cypher uses ``-[:indra_rel*..N]-`` with no arrow).  Unmeasured
        intermediaries participate in path-finding; only the |t|
        computation is restricted to measured genes.

        Mirrors :func:`network_proximity.run_proximity_decay_test`.
        Recommended over :meth:`run_gradient` when many INDRA
        intermediaries may be unmeasured (typical for proteomics).

        **Important caveats:**

        - Undirected traversal counts ``A ← B ← seed`` and
          ``seed → B → A`` both as distance 2.  This measures graph
          proximity, not directed regulatory cascade.
        - Cypher does not filter by ``min_evidence`` /
          ``min_reliability`` — every ``indra_rel`` edge participates,
          including low-quality text-mining edges.  This matches the
          proximity test for comparability.
        - Edge-quality stratification is not supported in this mode
          (would require per-path quality bookkeeping); a WARN is
          logged when this method runs.
        - Distances and degrees are cached on the bridge instance keyed
          by (seed, max_hops, measured-symbol-set).  Repeated calls with
          the same parameters return cached results.

        Parameters
        ----------
        seed
            Seed gene symbol.
        max_hops
            Maximum hop distance for shortest-path queries.
        n_permutations
            Degree-preserving permutations for the null distribution.
        rng_seed
            Random seed.

        Returns
        -------
        :class:`~perturbation_gradient.GradientResult`
        """
        from .network_proximity import (
            query_gene_degrees_batched,
            query_shortest_paths_batched,
        )
        from .perturbation_gradient import run_gradient_test

        self._ensure_indra()

        logger.info(
            "Running gradient (mode=UNDIRECTED-shortest-paths-full-graph) "
            "from seed=%s with max_hops=%d",
            seed, max_hops,
        )
        logger.warning(
            "Edge-quality stratification is not supported in shortest-paths mode; "
            "result.stratified will be None."
        )

        # We query INDRA at HGNC-symbol level (the graph's native
        # vertex identity) but aggregate to UniProt-level for the
        # gradient itself.  Aliases of the same protein are NOT
        # independent observations.
        all_hgnc = sorted(self.sym_to_feat.keys())
        # Exclude the seed gene from the target list — Neo4j's
        # shortestPath refuses paths between identical start and end
        # nodes (DatabaseError 51N23), and self-distance is 0 anyway
        # (outside our 1 ≤ d ≤ max_hops shell range).  We also exclude
        # any other HGNC alias of the seed's UniProt: if the seed is
        # measured (e.g., VCL → P18206), other aliases of P18206 would
        # both have valid Neo4j distances but would all be the seed
        # protein, which we drop from the analysis entirely.
        seed_fid = self.sym_to_feat.get(seed)
        if seed_fid is not None:
            seed_aliases = {
                s for s, fid in self.sym_to_feat.items() if fid == seed_fid
            }
        else:
            seed_aliases = {seed}
        measured_symbols = [s for s in all_hgnc if s not in seed_aliases]

        # Cache key: HGNC-level queries are invariant under contrast and
        # under the protein-level aggregation (which is deterministic
        # given the alias map).  Edge scope is the constant
        # ALL_REGULATORY_TYPES, so it doesn't enter the key.
        cache_key = (
            "hgnc_dist_deg",
            seed_fid or seed,
            max_hops,
            hash(tuple(measured_symbols)),
        )
        cached = self._graph_query_cache.get(cache_key)
        if cached is not None:
            distances_hgnc, degrees_hgnc = cached
            logger.info(
                "Using cached shortest paths and degrees (seed=%s, max_hops=%d)",
                seed, max_hops,
            )
        else:
            cogex = self._indra_source.client
            logger.info(
                "Querying server-side shortest paths from %s to %d HGNC aliases "
                "(measured proteins, alias-expanded; edge scope: regulatory — "
                "Activation/Inhibition/IncreaseAmount/DecreaseAmount)",
                seed, len(measured_symbols),
            )
            distances_hgnc = query_shortest_paths_batched(
                cogex_client=cogex,
                seed_gene_name=seed,
                target_gene_names=measured_symbols,
                max_hops=max_hops,
                batch_size=500,
                verbose=True,
            )

            logger.info(
                "Querying graph degrees for %d HGNC aliases",
                len(measured_symbols),
            )
            degrees_hgnc = query_gene_degrees_batched(
                cogex_client=cogex,
                gene_names=measured_symbols,
                batch_size=500,
            )
            self._graph_query_cache[cache_key] = (distances_hgnc, degrees_hgnc)

        # Aggregate HGNC-keyed query results to UniProt-keyed inputs.
        # One observation per protein measurement.
        abs_t_prot, dist_prot, deg_prot, via_alias = self.get_protein_level_inputs(
            seed=seed,
            max_hops=max_hops,
            distances_hgnc=distances_hgnc,
            degrees_hgnc=degrees_hgnc,
        )
        n_proteins_measured = len(abs_t_prot)
        n_proteins_reachable = len(dist_prot)
        logger.info(
            "Protein-level inputs: %d measured proteins (%d reachable within %d hops)",
            n_proteins_measured, n_proteins_reachable, max_hops,
        )

        # Build shells keyed by UniProt feature_id
        shells: dict[int, set[str]] = {}
        for fid, d in dist_prot.items():
            shells.setdefault(d, set()).add(fid)

        if not shells:
            raise ValueError(
                f"No measured proteins within {max_hops} hops of '{seed}'."
            )

        return run_gradient_test(
            adjacency={},  # unused when precomputed_shells is provided
            abs_t_stats=abs_t_prot,
            seed=seed,
            max_hops=max_hops,
            n_permutations=n_permutations,
            rng_seed=rng_seed,
            precomputed_shells=shells,
            graph_degrees=deg_prot,
        )

    def run_rewiring_null(
        self,
        seed: str,
        observed_slope: float,
        observed_coverage: float | None = None,
        n_rewires: int = 999,
        max_hops: int = 3,
        rng_seed: int = 42,
        subgraph_max_hops: int | None = None,
        max_swaps_iter0: int = 500_000,
        verbose: bool = True,
    ):
        """Run the degree-preserving edge-rewiring null on the seed's component.

        Extracts the INDRA subgraph around ``seed`` via Cypher (cached on the
        bridge per (seed, subgraph_max_hops)), restricts to the connected
        component containing the seed, and runs
        :func:`~perturbation_gradient.run_rewiring_null`.

        **Semantics: UNDIRECTED, FULL-GRAPH within the extracted component.**
        This mirrors the shortest-paths gradient's methodology.  The rewiring
        null tests whether the observed gradient is reproducible when the
        component is randomly rewired while preserving degree sequence.

        Parameters
        ----------
        seed
            Seed gene symbol.
        observed_slope
            Slope from the observed (unrewired) gradient.
        observed_coverage
            Fraction of in-graph |t|-stat keys reachable from seed within
            ``max_hops``.  Optional; recomputed if None.
        n_rewires
            Number of rewiring permutations.
        max_hops
            BFS depth for gradient shell construction INSIDE each rewire.
        rng_seed
            Base random seed for reproducibility.
        subgraph_max_hops
            Hops from seed for Cypher subgraph extraction.  Defaults to
            ``max_hops`` (no buffer) because INDRA is sufficiently dense
            around typical biomedical seeds that any buffer blows up the
            subgraph — ``max_hops + 1`` around C9orf72 already pulls
            ~9M edges, making per-iter rewiring infeasible at N=999.
            The no-buffer choice means rewirings which push targets past
            ``max_hops`` show up as disconnections (tracked in
            ``disconnection_rate``) rather than as extended shells; this
            is a conservative tradeoff that we accept rather than silently
            running for hours.  For global rewiring guarantees on sparser
            graphs, pass a larger value explicitly.
        max_swaps_iter0
            Hard ceiling on iter-0 mixing diagnostic.
        verbose
            Log progress.

        Returns
        -------
        :class:`~perturbation_gradient.RewiringNullResult`
        """
        from .graph_rewiring import edges_to_undirected_graph, seed_component
        from .network_proximity import extract_local_subgraph_edges
        from .perturbation_gradient import run_rewiring_null

        self._ensure_indra()

        # Protein-level inputs: |t| keyed by UniProt feature_id (no
        # alias inflation), with an aliases map so the per-iter BFS in
        # the inner loop can aggregate HGNC distances back to UniProts.
        abs_t_prot = self.get_abs_t_per_feature()
        feat_to_syms = self._build_feat_to_syms()
        # Drop the seed protein (if measured) — same logic as the
        # gradient path: VCL → P18206 means we exclude P18206 entirely.
        seed_fid = self.sym_to_feat.get(seed)
        if seed_fid is not None:
            abs_t_prot = {fid: t for fid, t in abs_t_prot.items() if fid != seed_fid}
            feat_to_syms = {
                fid: aliases for fid, aliases in feat_to_syms.items()
                if fid != seed_fid
            }
        # The aliases map is restricted to measured proteins so the
        # rewiring null only ever does BFS on units we can score.
        aliases = {fid: feat_to_syms.get(fid, []) for fid in abs_t_prot}

        if subgraph_max_hops is None:
            subgraph_max_hops = max_hops

        cache_key = ("subgraph_edges", seed, subgraph_max_hops)
        cached = self._graph_query_cache.get(cache_key)
        if cached is not None:
            edge_list = cached[0]  # reuse existing tuple slot
            logger.info(
                "Using cached subgraph edges for %s (subgraph_max_hops=%d)",
                seed, subgraph_max_hops,
            )
        else:
            cogex = self._indra_source.client
            logger.info(
                "Extracting INDRA subgraph around %s (max_hops=%d) for rewiring "
                "(edge scope: regulatory — "
                "Activation/Inhibition/IncreaseAmount/DecreaseAmount)...",
                seed, subgraph_max_hops,
            )
            edge_list = extract_local_subgraph_edges(
                cogex_client=cogex,
                seed_gene_name=seed,
                max_hops=subgraph_max_hops,
                min_evidence=1,
            )
            self._graph_query_cache[cache_key] = (edge_list, None)

        if not edge_list:
            raise ValueError(
                f"No edges extracted from INDRA for seed '{seed}' at "
                f"subgraph_max_hops={subgraph_max_hops}. Check seed name "
                f"resolution and Neo4j connectivity."
            )

        G_full = edges_to_undirected_graph(edge_list)
        G = seed_component(G_full, seed)
        if seed not in G:
            raise ValueError(
                f"Seed '{seed}' is in the extracted subgraph ({G_full.number_of_nodes()} "
                f"nodes) but not in its largest connected component "
                f"(|V|={G.number_of_nodes()}). Extraction may have returned only "
                f"outgoing edges from non-seed nodes; verify Cypher."
            )
        if G.number_of_nodes() < 100:
            logger.warning(
                "Seed component is small (|V|=%d, |E|=%d); rewiring null may "
                "have low power and is sensitive to disconnection artifacts.",
                G.number_of_nodes(), G.number_of_edges(),
            )

        return run_rewiring_null(
            graph=G,
            seed=seed,
            abs_t_stats=abs_t_prot,
            observed_slope=observed_slope,
            observed_coverage=observed_coverage,
            n_rewires=n_rewires,
            max_hops=max_hops,
            rng_seed=rng_seed,
            max_swaps_iter0=max_swaps_iter0,
            verbose=verbose,
            aliases=aliases,
        )

    def close(self):
        """Release INDRA connection and clear caches."""
        if self._indra_source is not None:
            self._indra_source.close()
            self._indra_source = None
        self._target_cache.clear()
        self._edge_metadata_cache.clear()
        self._evidence_text_cache.clear()
        self._graph_query_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
