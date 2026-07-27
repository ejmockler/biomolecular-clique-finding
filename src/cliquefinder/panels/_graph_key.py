"""How a measured feature is identified as a node in the INDRA graph.

Two key spaces exist, and the difference is not cosmetic.

``symbol`` (legacy)
    A feature is represented by *every* HGNC symbol and alias that a live
    MyGene lookup returns for it, and INDRA ``BioEntity`` nodes are matched
    on their bare ``name``.  Name strings are not entity-preserving: one
    string can name several entities across namespaces, so an alias of a
    measured protein may be the official name of a completely different,
    *unmeasured* entity.  That entity's node and all of its edges then enter
    the graph under the measured feature's identity.  Observed on the
    3,264-row AnswerALS proteome: ``AR`` is an alias of AKR1B1 (P15121), so
    the androgen receptor (``hgnc:644``, 14,818 regulatory edges, not
    measured) was admitted as P15121 and gave it a hop-1 shell of 821
    against a dataset median of 17.  Protein-family and chemical nodes leak
    the same way (``AKT`` -> ``fplx:AKT``, ``NADPH`` -> ``chebi:16474``).

``curie`` (default)
    A feature is represented by exactly one namespaced identifier — its
    canonical HGNC CURIE, e.g. ``hgnc:391`` — and ``BioEntity`` nodes are
    matched on ``id``.  One key, one entity, so a traversal cannot acquire
    another gene's edges, and a "measured-only" path restriction is enforced
    over measured *proteins* rather than over name strings.

``curie`` is also resolved offline from INDRA's bundled UniProt tables
rather than from a live MyGene call, which removes a source of run-to-run
drift: features have been observed to resolve differently between runs
under ``symbol``.

The choice is recorded on the design (``LandscapeDesign.graph_key``) so
every manifest is self-describing and the resume design-equality guard
refuses to mix key spaces within one output_dir.
"""
from __future__ import annotations

CURIE_GRAPH_KEY = "curie"
SYMBOL_GRAPH_KEY = "symbol"
VALID_GRAPH_KEYS = frozenset({CURIE_GRAPH_KEY, SYMBOL_GRAPH_KEY})

#: Which ``BioEntity`` property each key space matches on.  Consumed by
#: ``network_proximity``'s ``key_property`` argument.
GRAPH_KEY_TO_NODE_PROPERTY = {
    CURIE_GRAPH_KEY: "id",
    SYMBOL_GRAPH_KEY: "name",
}


def node_property_for(graph_key: str) -> str:
    """Map a design-level ``graph_key`` to its Cypher node property."""
    if graph_key not in VALID_GRAPH_KEYS:
        raise ValueError(
            f"graph_key must be one of {sorted(VALID_GRAPH_KEYS)}, "
            f"got {graph_key!r}"
        )
    return GRAPH_KEY_TO_NODE_PROPERTY[graph_key]


def resolve_feature_curies(
    feature_ids: list[str],
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    """Resolve measured feature ids to canonical HGNC CURIEs.

    Resolution is offline — it reads INDRA's bundled UniProt/HGNC tables,
    so it is deterministic across runs and needs no network call.

    Parameters
    ----------
    feature_ids
        Measured feature row labels (UniProt accessions in production).

    Returns
    -------
    feat_to_curie
        ``{feature_id: "hgnc:NNNN"}`` for every feature that resolved.
    curie_to_feats
        Inverse, ``{curie: [feature_id, ...]}``.  The list is normally a
        singleton, but distinct rows can share a gene (e.g. the two TMPO
        rows P42166/P42167 both map to ``hgnc:11875``); such features are
        genuinely one node in INDRA and must share that node's distances.
    unresolved
        Features with no HGNC identity — a synthetic standard such as
        ``1/iRT_protein``, or an accession INDRA's tables do not carry.
        These are excluded from the graph, and — unlike the legacy
        ``symbol`` path — they are reported here rather than silently
        becoming indistinguishable from graph-isolated features.
    """
    from indra.databases import uniprot_client

    feat_to_curie: dict[str, str] = {}
    curie_to_feats: dict[str, list[str]] = {}
    unresolved: list[str] = []

    for fid in feature_ids:
        hgnc_id = uniprot_client.get_hgnc_id(str(fid))
        if not hgnc_id:
            unresolved.append(str(fid))
            continue
        curie = f"hgnc:{hgnc_id}"
        feat_to_curie[str(fid)] = curie
        curie_to_feats.setdefault(curie, []).append(str(fid))

    return feat_to_curie, curie_to_feats, unresolved
