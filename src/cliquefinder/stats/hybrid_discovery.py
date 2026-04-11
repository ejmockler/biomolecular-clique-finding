"""Hybrid RWR + discrete discovery — wraps run_discovery with RWR annotations.

Preserves the discrete discovery pipeline's FDR control (Storey BH,
hierarchical Yekutieli, knockoff filter) while adding:

1. Per-intermediary RWR proximity annotations
2. Discovery gain metric per hop (are significant arms RWR-enriched?)
3. Beyond-hop candidates (top genes by RWR not reached by discrete pipeline)

The wrapper does NOT modify run_discovery() — it calls it unchanged and
post-processes the result.

Usage::

    signed_rwr = compute_signed_rwr_scores(edges, seed_gene="C9orf72")
    result = run_hybrid_discovery(
        seed="C9orf72", adjacency=graph, signed_rwr=signed_rwr,
        test_gene_set=bridge.test_gene_set, get_targets=bridge.get_targets,
    )
    print(result.rwr_only_candidates)  # genes beyond hop frontier
    print(result.discovery_gain_per_hop)  # RWR enrichment per hop
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cliquefinder.stats.network_proximity import SignedRWRResult

logger = logging.getLogger(__name__)


@dataclass
class RWRCandidate:
    """A gene identified by RWR but not reached by discrete discovery."""

    gene: str
    ranking_score: float   # RDPN z-score if available, else raw RWR combined
    rwr_signed: float      # z-normalized act - rep (net direction)
    ranking_rank: int      # 1-indexed rank by ranking_score
    score_type: str = ""   # "rdpn_zscore" or "rwr_combined"


@dataclass
class HybridDiscoveryResult:
    """Discrete discovery result augmented with RWR annotations.

    Attributes
    ----------
    discovery_result
        The underlying ``DiscoveryResult`` from ``run_discovery()``.
    rwr_annotations
        Per-intermediary RWR combined score: ``{gene: score}``.
    discovery_gain_per_hop
        Ratio of median RWR score of significant arms to median of all arms,
        per hop. >1.0 means significant arms are RWR-enriched.
    rwr_only_candidates
        Top genes by RWR combined score not reached by the discrete pipeline.
    signed_rwr
        The full signed RWR result for downstream use.
    """

    discovery_result: Any  # DiscoveryResult from causal_path_scoring
    rwr_annotations: dict[str, float]
    discovery_gain_per_hop: list[float]
    rwr_only_candidates: list[RWRCandidate]
    signed_rwr: SignedRWRResult

    @property
    def seed_gene(self) -> str:
        return self.discovery_result.seed_gene

    @property
    def hops(self):
        return self.discovery_result.hops

    def summary(self) -> str:
        lines = [self.discovery_result.summary()]
        lines.append(f"\nRWR annotations: {len(self.rwr_annotations)} intermediaries scored")
        for i, gain in enumerate(self.discovery_gain_per_hop):
            lines.append(f"  Hop {i+1} discovery gain: {gain:.2f}")
        lines.append(f"RWR-only candidates: {len(self.rwr_only_candidates)}")
        for c in self.rwr_only_candidates[:10]:
            lines.append(
                f"  {c.gene:>12s}  score={c.ranking_score:.4f} ({c.score_type})  "
                f"signed={c.rwr_signed:.2f}  rank={c.ranking_rank}"
            )
        return "\n".join(lines)


def run_hybrid_discovery(
    seed: str,
    adjacency: dict[str, list],
    signed_rwr: SignedRWRResult,
    test_gene_set: Callable[[list[str], str], float],
    get_targets: Callable[[str], list[str]],
    measurable_genes: set[str] | None = None,
    top_k_candidates: int = 50,
    rdpn_zscores: dict[str, float] | None = None,
    target_to_effect: dict[str, float] | None = None,
    target_to_direction: dict[str, str] | None = None,
    **discovery_kwargs: Any,
) -> HybridDiscoveryResult:
    """Run discrete discovery with RWR annotations.

    Parameters
    ----------
    seed, adjacency, test_gene_set, get_targets, discovery_kwargs
        Passed directly to ``run_discovery()`` (unchanged).
    signed_rwr
        Signed RWR result from ``compute_signed_rwr_scores()``.
    measurable_genes
        Set of gene symbols measurable in the expression dataset.
        Used to filter RWR-only candidates to actionable genes.
    top_k_candidates
        Number of beyond-hop RWR candidates to report.
    rdpn_zscores
        If provided, RDPN degree-deconfounded z-scores per gene.
        Used to rank RWR-only candidates (controls for hub bias).
        Without this, candidates are ranked by raw RWR combined score.

    Returns
    -------
    HybridDiscoveryResult with discrete result + RWR annotations.
    """
    from causal_path_scoring.core.discovery import run_discovery

    # Run discrete discovery unchanged — forward all required args explicitly
    disc_result = run_discovery(
        seed=seed,
        adjacency=adjacency,
        test_gene_set=test_gene_set,
        target_to_effect=target_to_effect or {},
        target_to_direction=target_to_direction or {},
        measurable_genes=measurable_genes or set(),
        get_targets=get_targets,
        **discovery_kwargs,
    )

    # Collect all intermediaries tested across all hops
    all_intermediaries: set[str] = set()
    for hop_result in disc_result.hops:
        for arm in hop_result.all_arms:
            all_intermediaries.add(arm.intermediary)

    # Annotate each intermediary with RWR combined score
    rwr_annotations = {}
    for gene in all_intermediaries:
        rwr_annotations[gene] = signed_rwr.combined_scores.get(gene, 0.0)

    # Compute discovery gain per hop
    discovery_gain_per_hop = []
    for hop_result in disc_result.hops:
        all_scores = [
            signed_rwr.combined_scores.get(a.intermediary, 0.0)
            for a in hop_result.all_arms
        ]
        sig_scores = [
            signed_rwr.combined_scores.get(a.intermediary, 0.0)
            for a in hop_result.significant_arms
        ]
        if all_scores and sig_scores:
            med_all = float(np.median(all_scores))
            med_sig = float(np.median(sig_scores))
            gain = med_sig / med_all if med_all > 0 else float("inf")
        else:
            gain = float("nan")
        discovery_gain_per_hop.append(gain)

    # Find RWR-only candidates: high RWR score but not tested by discrete pipeline.
    # Use RDPN z-scores if available (controls for hub bias), otherwise raw RWR.
    if rdpn_zscores is not None:
        ranking_scores = rdpn_zscores
        score_type = "rdpn_zscore"
        logger.info("Ranking RWR-only candidates by RDPN z-scores (hub-deconfounded)")
    else:
        ranking_scores = signed_rwr.combined_scores
        score_type = "rwr_combined"
        logger.warning("No RDPN z-scores — ranking by raw RWR (hub-biased)")
    ranked = sorted(
        ranking_scores.items(),
        key=lambda x: -x[1],
    )

    rwr_only = []
    for rank, (gene, score) in enumerate(ranked, 1):
        if gene == seed:
            continue
        if gene in all_intermediaries:
            continue
        if measurable_genes is not None and gene not in measurable_genes:
            continue
        rwr_only.append(RWRCandidate(
            gene=gene,
            ranking_score=score,
            rwr_signed=signed_rwr.signed_scores.get(gene, 0.0),
            ranking_rank=rank,
            score_type=score_type,
        ))
        if len(rwr_only) >= top_k_candidates:
            break

    return HybridDiscoveryResult(
        discovery_result=disc_result,
        rwr_annotations=rwr_annotations,
        discovery_gain_per_hop=discovery_gain_per_hop,
        rwr_only_candidates=rwr_only,
        signed_rwr=signed_rwr,
    )
