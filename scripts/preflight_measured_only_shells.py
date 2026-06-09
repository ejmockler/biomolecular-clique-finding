"""Wave 24l pre-flight: measured-only-paths hop-2 shell-size diagnostic.

Brutalist G12 / M4 finding: before committing 2-5 hr × 3 contrasts of
landscape compute under the new measured-only-paths regime, we need
to know whether hop-2 collapses to a shell so small that the slope
statistic (mean|t|_hop1 - mean|t|_hop2) is shot-noise.

Cheap diagnostic — directly queries (measured, measured) regulatory
edges from Neo4j without frontier expansion through unmeasured nodes.
Should complete in minutes, not the hour-scale that the full
extract_subgraph_induced_by_features takes.

  1. Query INDRA for edges where both endpoints are measured.
  2. Locally BFS with node_filter for a sample of anchors (C9orf72 +
     cluster anchors + 50 random measured).
  3. Tabulate hop-1 and hop-2 sizes.

Decision rule (informational, surfaces to user):
  median hop-2 < 10  : STOP — slope statistic is shot-noise
  median hop-2 < 30  : ESCALATE — fragile; user decision needed
  median hop-2 < 50  : EXPLORATORY — proceed with wide CIs
  median hop-2 >= 50 : WORKABLE — proceed with multi-hour compute

Run:
    .venv/bin/python scripts/preflight_measured_only_shells.py
"""
from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CLUSTER_ANCHOR_PROBES = [
    "NUP98", "NUP155", "RANBP2",          # NPC / transport
    "SF3B1", "PRPF8", "SRSF1",            # splicing
    "MCM3", "RBBP4",                       # chromatin
]
SEED_PROBES = ["C9ORF72"] + CLUSTER_ANCHOR_PROBES


def query_measured_pair_edges(
    client,
    measured_symbols: list[str],
    min_evidence: int = 1,
    batch_size: int = 500,
):
    """Query INDRA for regulatory edges where BOTH endpoints are in
    ``measured_symbols``.  One Cypher pass per source-batch; target
    is filtered server-side via ``b.name IN $measured_set``.

    Returns a list of ``(source, target, attrs)`` tuples.
    """
    from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES

    measured_set = list(measured_symbols)
    edge_index: dict[tuple[str, str, str], int] = {}
    edge_query = """
    MATCH (a:BioEntity)-[r:indra_rel]->(b:BioEntity)
    WHERE a.name IN $batch
      AND b.name IN $measured_set
      AND r.evidence_count >= $min_evidence
      AND r.stmt_type IN $stmt_types
    RETURN a.name AS source, b.name AS target,
           r.evidence_count AS evidence_count,
           r.stmt_type AS stmt_type
    """
    n_batches = (len(measured_set) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(measured_set), batch_size)):
        batch = measured_set[i:i + batch_size]
        t0 = time.time()
        rows = client._execute_query(
            edge_query,
            batch=batch,
            measured_set=measured_set,
            min_evidence=min_evidence,
            stmt_types=list(ALL_REGULATORY_TYPES),
        )
        for row in rows:
            src, tgt, ec, st = row[0], row[1], row[2], row[3]
            key = (src, tgt, st)
            cur = edge_index.get(key)
            if cur is None or cur < ec:
                edge_index[key] = ec
        logging.getLogger("preflight-shells").info(
            "  batch %d/%d: %d rows (%.1fs)",
            batch_idx + 1, n_batches, len(rows), time.time() - t0,
        )

    edges = [
        (src, tgt, {"evidence_count": ec, "stmt_type": st})
        for (src, tgt, st), ec in edge_index.items()
    ]
    return edges


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("preflight-shells")

    from cliquefinder.knowledge.cogex import CoGExClient
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.network_proximity import (
        compute_all_pairs_shortest_paths_bounded,
    )

    data_path = ROOT / "output/proteomics/all_als.data.csv"
    env_path = ROOT / ".env"

    log.info("Loading proteomics feature IDs from %s", data_path)
    data_df = pd.read_csv(data_path, index_col=0)
    feature_ids = list(data_df.index)
    sym_to_feat = map_feature_ids_to_symbols(feature_ids, verbose=False)
    measured_symbols = sorted(sym_to_feat.keys())
    measured_symbol_set = set(measured_symbols)
    log.info(
        "Measured: %d UniProt features → %d HGNC aliases",
        len(feature_ids), len(measured_symbols),
    )

    seed_probes_present = [
        s for s in SEED_PROBES if s in measured_symbol_set
    ]
    missing = [s for s in SEED_PROBES if s not in measured_symbol_set]
    if missing:
        log.warning("Probe symbols not in measured set: %s", missing)

    log.info(
        "Querying (measured, measured) regulatory edges across %d "
        "measured symbols…",
        len(measured_symbols),
    )
    t0 = time.time()
    with CoGExClient(env_file=env_path) as client:
        edges = query_measured_pair_edges(
            client=client,
            measured_symbols=measured_symbols,
            min_evidence=1,
            batch_size=500,
        )
    log.info(
        "Got %d measured-pair regulatory edges (%.1fs total)",
        len(edges), time.time() - t0,
    )

    # Sample 50 random anchors deterministically, ONLY from symbols
    # that appear as edge endpoints (i.e., have ≥1 measured regulatory
    # neighbor).  Sampling from the full sym_to_feat keyset would
    # over-represent obscure aliases that don't appear in INDRA at
    # all, producing a misleading "median 0" verdict.
    in_graph = {s for s, _, _ in edges} | {t for _, t, _ in edges}
    in_graph_measured = sorted(in_graph & measured_symbol_set)
    pool = sorted(set(in_graph_measured) - set(seed_probes_present))
    rng = random.Random(42)
    random_probes = rng.sample(pool, k=min(50, len(pool)))
    log.info(
        "Pool for random sampling: %d in-graph measured symbols",
        len(in_graph_measured),
    )
    all_probes = seed_probes_present + random_probes

    log.info(
        "Computing measured-only BFS for %d probes (%d named + %d random)",
        len(all_probes), len(seed_probes_present), len(random_probes),
    )
    t0 = time.time()
    distances = compute_all_pairs_shortest_paths_bounded(
        edges=edges,
        source_nodes=all_probes,
        max_hops=2,
        target_filter=measured_symbol_set,
        node_filter=measured_symbol_set,
    )
    log.info("BFS done (%.1fs)", time.time() - t0)

    log.info("=" * 64)
    log.info("%-12s  %6s  %6s  [%s]", "anchor", "hop1", "hop2", "kind")
    log.info("-" * 64)
    named_hop2: list[int] = []
    random_hop2: list[int] = []
    named_hop1: list[int] = []
    random_hop1: list[int] = []
    for probe in all_probes:
        d = distances.get(probe, {})
        hop1 = sum(1 for v in d.values() if v == 1)
        hop2 = sum(1 for v in d.values() if v == 2)
        is_named = probe in seed_probes_present
        kind = "named" if is_named else "rand"
        log.info("%-12s  %6d  %6d  [%s]", probe, hop1, hop2, kind)
        if is_named:
            named_hop1.append(hop1); named_hop2.append(hop2)
        else:
            random_hop1.append(hop1); random_hop2.append(hop2)
    log.info("=" * 64)

    def _summary(name: str, vals: list[int]) -> None:
        if not vals:
            log.info("%s: no values", name)
            return
        v = sorted(vals)
        n = len(v)
        log.info(
            "%s (n=%d): min=%d  p25=%d  median=%d  p75=%d  max=%d  "
            "frac_<30=%.2f  frac_<10=%.2f",
            name, n, v[0], v[n // 4], v[n // 2], v[3 * n // 4], v[-1],
            sum(1 for x in v if x < 30) / n,
            sum(1 for x in v if x < 10) / n,
        )

    log.info("---- Hop-1 size distribution ----")
    _summary("Named anchors", named_hop1)
    _summary("Random anchors", random_hop1)
    _summary("All probes",     named_hop1 + random_hop1)
    log.info("---- Hop-2 size distribution ----")
    _summary("Named anchors", named_hop2)
    _summary("Random anchors", random_hop2)
    _summary("All probes",     named_hop2 + random_hop2)

    all_hop2 = sorted(named_hop2 + random_hop2)
    if not all_hop2:
        log.warning("No hop-2 values; cannot decide.")
        return
    median_hop2 = all_hop2[len(all_hop2) // 2]
    if median_hop2 < 10:
        verdict = "STOP — slope statistic is shot-noise"
    elif median_hop2 < 30:
        verdict = "ESCALATE — slope statistic fragile; user decision"
    elif median_hop2 < 50:
        verdict = "EXPLORATORY — proceed with wide CIs"
    else:
        verdict = "WORKABLE — proceed with multi-hour landscape compute"
    log.info("---- DECISION ----")
    log.info("Median hop-2: %d → %s", median_hop2, verdict)
    log.info("PREFLIGHT_DONE")


if __name__ == "__main__":
    main()
