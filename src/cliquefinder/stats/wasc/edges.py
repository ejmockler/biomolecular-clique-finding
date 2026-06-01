"""WASC edge enumeration — Within-cluster INDRA hop-1 pairs (M1).

This module implements the M1 vertical slice from memory/wasc_build_plan.md:
enumerate ``E_WASC`` per theme using only existing utilities. No per-pair
regression, no null, no FDR — those are M2+.

The output is a frozen UniProt-pair list that gates the rest of the build.
If |E_WASC| comes back substantially outside the spec's ±30% of 220±70
sanity gate, the build pauses for review before M6a tagging.

See memory/wasc_spec.md §1 for the formal edge definition.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, TYPE_CHECKING

from .types import Network, Theme, WascEdge

if TYPE_CHECKING:
    from cliquefinder.knowledge.cogex import CoGExClient

logger = logging.getLogger(__name__)


# Default cluster terms per theme — frozen reference list matching
# scripts/viz/common.py::TERMS.  Kept in this module as a self-contained
# default; callers may override with custom term lists for sensitivity runs.
DEFAULT_CLUSTER_TERMS: tuple[tuple[Theme, str], ...] = (
    (Theme.SPLICING, "reactome:R-HSA-72172"),
    (Theme.SPLICING, "reactome:R-HSA-72203"),
    (Theme.SPLICING, "go:0000398"),
    (Theme.CHROMATIN, "go:0005694"),
    (Theme.CHROMATIN, "go:0000785"),
    (Theme.TRANSPORT, "go:0006913"),
    (Theme.TRANSPORT, "go:0005643"),
    (Theme.TRANSPORT, "reactome:R-HSA-180910"),
)


# ---------------------------------------------------------------------------
# Step 1 — per-theme measured cluster members
# ---------------------------------------------------------------------------

def compute_measured_cluster_members(
    cluster_terms: list[tuple[Theme, str]] | tuple[tuple[Theme, str], ...],
    measured_uniprots: frozenset[str],
    fetch_term_members_func: Callable[[list[str]], dict[str, set[str]]],
    hgnc_to_uniprot_func: Callable[[set[str]], set[str]],
) -> dict[Theme, frozenset[str]]:
    """For each theme T, compute ``M_T = measured ∩ union(cluster members)``.

    Parameters
    ----------
    cluster_terms
        Iterable of ``(theme, term_id)`` pairs (e.g. ``DEFAULT_CLUSTER_TERMS``).
    measured_uniprots
        The proteomics-measured UniProt set ``M`` (3,257 for the Wave-22
        protein-level matrix in this project).
    fetch_term_members_func
        ``list[term_id] -> {term_id: {hgnc_id}}``. Typically
        ``scripts/viz/common.py::fetch_term_members_via_indra``.
    hgnc_to_uniprot_func
        ``{hgnc_id} -> {uniprot}``. Typically
        ``scripts/viz/common.py::hgnc_ids_to_uniprots``.

    Returns
    -------
    dict[Theme, frozenset[str]]
        Measured cluster members per theme.
    """
    by_theme: dict[Theme, list[str]] = defaultdict(list)
    for theme, term_id in cluster_terms:
        by_theme[theme].append(term_id)

    # Batched fetch across all term_ids (efficiency).
    all_term_ids = [tid for ids in by_theme.values() for tid in ids]
    hgnc_per_term = fetch_term_members_func(all_term_ids)

    result: dict[Theme, frozenset[str]] = {}
    for theme, term_ids in by_theme.items():
        hgnc_union: set[str] = set()
        for tid in term_ids:
            hgnc_union |= hgnc_per_term.get(tid, set())
        uniprot_union = hgnc_to_uniprot_func(hgnc_union)
        measured_intersection = uniprot_union & measured_uniprots
        result[theme] = frozenset(measured_intersection)
        logger.info(
            "Theme %s: %d HGNC members → %d UniProt → %d measured",
            theme.value,
            len(hgnc_union),
            len(uniprot_union),
            len(measured_intersection),
        )
    return result


# ---------------------------------------------------------------------------
# Step 2 — INDRA hop-1 edge enumeration within each theme
# ---------------------------------------------------------------------------

def _resolve_default_dependencies() -> tuple[Callable, Callable, Callable, Callable]:
    """Return the four real implementations of the dependency-injection hooks.

    Imports are lazy + path-massaged because the cluster-member fetch and
    the symbol mapping live in ``scripts/viz/common.py`` (a project-level
    utility module that pre-dates the ``src/`` layout).  Tests pass mocks
    directly and never reach this function.
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[4]
    common_path = project_root / "scripts" / "viz"
    if str(common_path) not in sys.path:
        sys.path.insert(0, str(common_path))
    # `common` is intentionally a shared name in scripts/viz/.
    from common import (  # type: ignore[import-not-found]
        fetch_term_members_via_indra,
        hgnc_ids_to_uniprots,
        uniprot_to_hgnc_symbol,
    )
    from cliquefinder.stats.network_proximity import (
        extract_subgraph_induced_by_features,
    )
    return (
        fetch_term_members_via_indra,
        hgnc_ids_to_uniprots,
        uniprot_to_hgnc_symbol,
        extract_subgraph_induced_by_features,
    )


def enumerate_wasc_indra_edges(
    cluster_terms: list[tuple[Theme, str]] | tuple[tuple[Theme, str], ...],
    measured_uniprots: frozenset[str],
    cogex_client: "CoGExClient",
    *,
    fetch_term_members_func: Callable | None = None,
    hgnc_to_uniprot_func: Callable | None = None,
    uniprot_to_symbol_func: Callable | None = None,
    extract_subgraph_func: Callable | None = None,
) -> tuple[WascEdge, ...]:
    """Enumerate within-cluster INDRA hop-1 edges per spec §1.

    For each theme T, returns all unordered pairs ``{a, j}`` where:

    - ``a, j`` are both measured cluster members of theme T (i.e. ``a, j ∈ M_T``)
    - ``dist_INDRA(a, j) = 1`` in the measured-only regulatory subgraph
      restricted to ALL_REGULATORY_TYPES
      (``Activation``/``Inhibition``/``IncreaseAmount``/``DecreaseAmount``).

    Edges are returned with anchor lexicographically smaller than target.
    **Cross-theme edges are NOT included** (a splicing-member anchor and
    a chromatin-member target sharing an INDRA edge falls outside the
    primary BY-FDR pool; reported only in a secondary exploratory
    module not yet implemented).

    Self-loops are dropped.  Bidirectional pairs (``A→B`` and ``B→A``
    both present in INDRA) are deduplicated into a single unordered edge.

    Parameters
    ----------
    cluster_terms
        Iterable of ``(theme, term_id)`` pairs.  Pass
        :data:`DEFAULT_CLUSTER_TERMS` for the pre-registered set.
    measured_uniprots
        The proteomics-measured UniProt set.
    cogex_client
        Active CoGExClient for INDRA Neo4j queries.
    fetch_term_members_func, hgnc_to_uniprot_func, uniprot_to_symbol_func, extract_subgraph_func
        Dependency-injection hooks for testing.  If ``None``, defaults to
        the real implementations.

    Returns
    -------
    tuple[WascEdge, ...]
        Frozen edge list.  ``|E_WASC|`` is checked against the spec's
        ±30% sanity gate by the orchestrator script, not here.
    """
    if (fetch_term_members_func is None
            or hgnc_to_uniprot_func is None
            or uniprot_to_symbol_func is None
            or extract_subgraph_func is None):
        defaults = _resolve_default_dependencies()
        fetch_term_members_func = fetch_term_members_func or defaults[0]
        hgnc_to_uniprot_func = hgnc_to_uniprot_func or defaults[1]
        uniprot_to_symbol_func = uniprot_to_symbol_func or defaults[2]
        extract_subgraph_func = extract_subgraph_func or defaults[3]

    # Step 1 — per-theme measured cluster members.
    members_by_theme = compute_measured_cluster_members(
        cluster_terms,
        measured_uniprots,
        fetch_term_members_func,
        hgnc_to_uniprot_func,
    )

    # Step 2 — UniProt ↔ HGNC-symbol bidirectional mapping for the
    # union of all M_T (the subgraph extractor expects HGNC symbols).
    all_members_uniprot = set().union(*members_by_theme.values()) if members_by_theme else set()
    uniprot_to_symbol = uniprot_to_symbol_func(sorted(all_members_uniprot))
    symbol_to_uniprots: dict[str, set[str]] = defaultdict(set)
    for up, sym in uniprot_to_symbol.items():
        symbol_to_uniprots[sym].add(up)

    # Step 3 — per theme, query hop-1 edges with both endpoints in M_T.
    edges: list[WascEdge] = []
    per_theme_counts: dict[Theme, int] = {}
    for theme, m_t in members_by_theme.items():
        if len(m_t) < 2:
            logger.warning(
                "Theme %s has only %d measured members — skipping (no edges possible)",
                theme.value, len(m_t),
            )
            per_theme_counts[theme] = 0
            continue

        m_t_symbols = sorted({
            uniprot_to_symbol.get(up, up) for up in m_t
        })

        # restrict_endpoints_to_features=True ensures both endpoints
        # are in `features`, i.e. both endpoints ∈ M_T.
        raw_edges, _matched_features = extract_subgraph_func(
            cogex_client=cogex_client,
            features=m_t_symbols,
            max_hops=1,
            min_evidence=1,
            restrict_endpoints_to_features=True,
        )
        logger.info(
            "Theme %s: %d raw hop-1 edges from INDRA (pre-dedupe)",
            theme.value, len(raw_edges),
        )

        # Step 4 — convert symbol-edges back to UniProt pairs, dedupe unordered.
        # Aggregate metadata (evidence_count, stmt_types) across both
        # directions when the same unordered pair appears twice.
        pair_meta: dict[tuple[str, str], dict[str, Any]] = {}
        for src_sym, tgt_sym, meta in raw_edges:
            # A symbol may expand to multiple UniProts (rare; isoforms).
            src_uniprots = symbol_to_uniprots.get(src_sym, set()) & m_t
            tgt_uniprots = symbol_to_uniprots.get(tgt_sym, set()) & m_t
            for su in src_uniprots:
                for tu in tgt_uniprots:
                    if su == tu:
                        continue  # self-loop
                    pair = tuple(sorted([su, tu]))
                    accum = pair_meta.setdefault(
                        pair, {"evidence_count": 0, "stmt_types": []}
                    )
                    if meta:
                        ev = meta.get("evidence_count")
                        if ev is not None:
                            try:
                                accum["evidence_count"] += int(ev)
                            except (TypeError, ValueError):
                                pass
                        stmt = meta.get("stmt_type")
                        if stmt and stmt not in accum["stmt_types"]:
                            accum["stmt_types"].append(stmt)

        theme_edges = []
        for (anchor_up, target_up), accum in pair_meta.items():
            theme_edges.append(WascEdge(
                anchor_uniprot=anchor_up,
                target_uniprot=target_up,
                theme=theme,
                network=Network.INDRA,
                anchor_symbol=uniprot_to_symbol.get(anchor_up, ""),
                target_symbol=uniprot_to_symbol.get(target_up, ""),
                evidence_count=(accum["evidence_count"] or None),
                stmt_types=(tuple(accum["stmt_types"]) or None),
            ))
        edges.extend(theme_edges)
        per_theme_counts[theme] = len(theme_edges)
        logger.info(
            "Theme %s: %d unique within-theme UniProt-pair edges",
            theme.value, len(theme_edges),
        )

    edges_tuple = tuple(edges)
    logger.info(
        "|E_WASC| total = %d "
        "(Splicing: %d, Chromatin: %d, Transport: %d)",
        len(edges_tuple),
        per_theme_counts.get(Theme.SPLICING, 0),
        per_theme_counts.get(Theme.CHROMATIN, 0),
        per_theme_counts.get(Theme.TRANSPORT, 0),
    )
    return edges_tuple
