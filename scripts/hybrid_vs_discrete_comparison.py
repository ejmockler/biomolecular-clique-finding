#!/usr/bin/env python3
"""Compare hybrid RWR + discrete discovery vs discrete-only on ALS C9orf72.

Loads the same data as validate_baselines.py, runs both pipelines, and
reports concordance, power delta, RWR-only candidates, and RDPN z-score
enrichment.

Usage:
    .venv/bin/python scripts/hybrid_vs_discrete_comparison.py \
        --output-dir results/hybrid_comparison \
        --indra-env-file .env \
        --network-query C9orf72 \
        --max-hops 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Hybrid vs discrete discovery comparison")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indra-env-file", type=str, default=".env")
    parser.add_argument("--network-query", type=str, default="C9orf72")
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--n-rotations", type=int, default=999)
    parser.add_argument("--rwr-hops", type=int, default=2,
                        help="Hops for INDRA subgraph extraction (for RWR)")
    parser.add_argument("--rdpn-rewirings", type=int, default=100,
                        help="Number of RDPN rewirings (100=fast, 500=production)")
    parser.add_argument("--validation-dir", type=Path, default=None,
                        help="Path to existing validation output (protein_differential_results.csv, indra_targets.json)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load existing validation data
    # -------------------------------------------------------------------------
    val_dir = args.validation_dir
    if val_dir is None:
        # Try common locations
        for candidate in [
            Path("results/validation"),
            Path("data/validation"),
            Path("results"),
        ]:
            if (candidate / "protein_differential_results.csv").exists():
                val_dir = candidate
                break
    if val_dir is None:
        logger.error("Cannot find validation data. Pass --validation-dir")
        sys.exit(1)

    print(f"Loading data from {val_dir}")

    protein_csv = val_dir / "protein_differential_results.csv"
    targets_json = val_dir / "indra_targets.json"

    if not protein_csv.exists():
        logger.error(f"Missing {protein_csv}")
        sys.exit(1)

    protein_df = pd.read_csv(protein_csv)
    print(f"  {len(protein_df)} proteins loaded")

    # Build effect maps
    disc_effects = {}
    disc_directions = {}
    abs_t_stats = {}
    for _, row in protein_df.iterrows():
        if pd.notna(row.get('t_statistic')):
            fid = row['feature_id']
            t = float(row['t_statistic'])
            disc_effects[fid] = abs(t)
            disc_directions[fid] = 'down' if t < 0 else 'up'
            # For RWR correlation, map by symbol if available
            sym = row.get('gene_symbol') or row.get('symbol') or ''
            if sym:
                abs_t_stats[sym] = abs(t)

    # -------------------------------------------------------------------------
    # Step 2: Set up ROAST engine
    # -------------------------------------------------------------------------
    print("\nSetting up ROAST engine...")
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.rotation import RotationTestConfig, SetStatistic

    # We need the fitted engine — load from validation checkpoint if possible
    engine_ckpt = val_dir / "rotation_engine.pkl"
    if engine_ckpt.exists():
        import pickle
        with open(engine_ckpt, 'rb') as f:
            eng = pickle.load(f)
        print(f"  Engine loaded from checkpoint")
    else:
        logger.error("No rotation engine checkpoint found. Run validate_baselines.py first.")
        sys.exit(1)

    feature_ids = list(eng.gene_to_idx.keys())
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=False)
    feature_to_symbol = {v: k for k, v in symbol_to_feature.items()}
    print(f"  {len(symbol_to_feature)} genes mapped")

    roast_config = RotationTestConfig(
        statistics=[SetStatistic.MSQ],
        n_rotations=args.n_rotations,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # Step 3: Build adjacency from target set
    # -------------------------------------------------------------------------
    print("\nBuilding adjacency...")
    from causal_path_scoring.core.reliability import Edge as CPSEdge
    from causal_path_scoring.core.edge_reliability import compute_edge_reliability_with_contradiction

    disc_adjacency = {args.network_query: []}

    if targets_json.exists():
        from cliquefinder.stats.target_set import TargetSet
        ts = TargetSet.load(targets_json)
        for sym, edges in ts.edge_metadata.items():
            reliability, direction, contradictory = compute_edge_reliability_with_contradiction(edges)
            disc_adjacency[args.network_query].append(CPSEdge(
                source=args.network_query, target=sym,
                reliability=reliability, edge_type=direction,
            ))
        print(f"  {len(disc_adjacency[args.network_query])} edges from target set")
    else:
        logger.error(f"Missing {targets_json}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4: Extract INDRA subgraph and compute signed RWR
    # -------------------------------------------------------------------------
    print(f"\nExtracting {args.rwr_hops}-hop INDRA subgraph for RWR...")
    from cliquefinder.knowledge.cogex import CoGExClient
    from cliquefinder.stats.network_proximity import (
        extract_local_subgraph_edges,
        compute_signed_rwr_scores,
        compute_rwr_scores,
        run_rwr_correlation_test,
    )

    cogex = CoGExClient(env_file=args.indra_env_file)
    try:
        rwr_edges = extract_local_subgraph_edges(
            cogex_client=cogex,
            seed_gene_name=args.network_query,
            max_hops=args.rwr_hops,
            min_evidence=1,
        )
        print(f"  {len(rwr_edges)} edges extracted")
    finally:
        cogex.close()

    # Signed RWR
    t0 = time.time()
    signed_rwr = compute_signed_rwr_scores(
        rwr_edges, seed_gene=args.network_query,
    )
    rwr_time = time.time() - t0
    print(f"  Signed RWR: {signed_rwr.n_act_edges} act + {signed_rwr.n_rep_edges} rep edges")
    print(f"  {len(signed_rwr.node_names)} nodes, computed in {rwr_time:.1f}s")
    print(f"  Act convergence: delta={signed_rwr.act_convergence[0]:.2e}, iters={signed_rwr.act_convergence[1]}")
    print(f"  Rep convergence: delta={signed_rwr.rep_convergence[0]:.2e}, iters={signed_rwr.rep_convergence[1]}")

    # -------------------------------------------------------------------------
    # Step 5+6: Run both pipelines with SHARED bridge (same INDRA cache)
    # -------------------------------------------------------------------------
    from causal_path_scoring.core.discovery import run_discovery
    from cliquefinder.stats.discovery_bridge import DiscoveryBridge
    from cliquefinder.stats.rwr_weighted_bridge import RWRWeightedBridge
    from cliquefinder.stats.hybrid_discovery import run_hybrid_discovery

    with DiscoveryBridge(
        eng, symbol_to_feature,
        env_file=args.indra_env_file,
        min_evidence=1, min_reliability=0.0, min_sources=1,
        roast_config=roast_config,
    ) as bridge:
        # --- Discrete-only ---
        print(f"\n{'='*60}")
        print("DISCRETE-ONLY DISCOVERY")
        print(f"{'='*60}")
        disc_result = run_discovery(
            seed=args.network_query,
            adjacency=disc_adjacency,
            test_gene_set=bridge.test_gene_set,
            target_to_effect=disc_effects,
            target_to_direction=disc_directions,
            measurable_genes=set(),
            max_hops=args.max_hops,
            min_targets_per_arm=5,
            fdr_threshold=args.alpha,
            get_targets=bridge.get_targets,
            verbose=True,
        )
        print(disc_result.summary())

        # Clear target cache so hybrid run re-queries (same bridge, fresh cache)
        bridge._target_cache.clear()
        bridge._edge_metadata_cache.clear()

        # --- RDPN null model (hub deconfounding) ---
        print(f"\n{'='*60}")
        print(f"RDPN NULL MODEL ({args.rdpn_rewirings} rewirings)")
        print(f"{'='*60}")
        from cliquefinder.stats.rdpn import compute_rdpn_null, compute_rdpn_zscores

        # Build unweighted adjacency for RDPN (same nodes as signed RWR)
        node_list = list(signed_rwr.node_names)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)
        _r, _c = [], []
        for src, tgt, _ in rwr_edges:
            if src in node_to_idx and tgt in node_to_idx:
                _r.append(node_to_idx[src])
                _c.append(node_to_idx[tgt])
        import scipy.sparse as sp_sparse
        rdpn_adj = sp_sparse.csr_matrix(
            (np.ones(len(_r), dtype=np.float64), (np.array(_r), np.array(_c))),
            shape=(n_nodes, n_nodes),
        )
        seed_idx = node_to_idx[args.network_query]

        # Compute unweighted observed RWR for fair RDPN comparison
        from cliquefinder.stats.network_proximity import compute_rwr_scores
        observed_unweighted, _, _ = compute_rwr_scores(rdpn_adj, seed_idx)

        rdpn_null = compute_rdpn_null(
            rdpn_adj, seed_idx,
            n_rewirings=args.rdpn_rewirings,
            rng=np.random.default_rng(42),
        )
        rdpn_z_arr = compute_rdpn_zscores(observed_unweighted, rdpn_null)
        rdpn_zscores = {node_list[i]: float(rdpn_z_arr[i]) for i in range(n_nodes)}
        print(f"  RDPN: {rdpn_null.n_successful}/{rdpn_null.n_rewirings} converged")
        print(f"  Z-scores: mean={rdpn_z_arr.mean():.2f}, std={rdpn_z_arr.std():.2f}")

        # --- Hybrid (RWR-weighted) ---
        print(f"\n{'='*60}")
        print("HYBRID DISCOVERY (RWR-weighted ROAST)")
        print(f"{'='*60}")
        rwr_bridge = RWRWeightedBridge(
            inner_bridge=bridge,
            signed_rwr=signed_rwr,
            feat_to_sym=feature_to_symbol,
            weight_mode="combined",
        )
        measurable = set(symbol_to_feature.keys())
        hybrid_result = run_hybrid_discovery(
            seed=args.network_query,
            adjacency=disc_adjacency,
            signed_rwr=signed_rwr,
            test_gene_set=rwr_bridge.test_gene_set,
            get_targets=rwr_bridge.get_targets,
            target_to_effect=disc_effects,
            target_to_direction=disc_directions,
            measurable_genes=measurable,
            max_hops=args.max_hops,
            min_targets_per_arm=5,
            fdr_threshold=args.alpha,
            top_k_candidates=50,
            rdpn_zscores=rdpn_zscores,
            verbose=True,
        )
        print(hybrid_result.summary())

    # -------------------------------------------------------------------------
    # Step 7: Concordance analysis
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CONCORDANCE ANALYSIS")
    print(f"{'='*60}")

    disc_sig = set()
    for hop in disc_result.hops:
        for arm in hop.significant_arms:
            disc_sig.add(arm.intermediary)

    hybrid_sig = set()
    for hop in hybrid_result.hops:
        for arm in hop.significant_arms:
            hybrid_sig.add(arm.intermediary)

    overlap = disc_sig & hybrid_sig
    disc_only = disc_sig - hybrid_sig
    hybrid_only = hybrid_sig - disc_sig

    print(f"  Discrete significant:  {len(disc_sig)}")
    print(f"  Hybrid significant:    {len(hybrid_sig)}")
    print(f"  Overlap:               {len(overlap)}")
    print(f"  Discrete-only:         {len(disc_only)} — {sorted(disc_only)}")
    print(f"  Hybrid-only:           {len(hybrid_only)} — {sorted(hybrid_only)}")
    if disc_sig:
        concordance = len(overlap) / len(disc_sig)
        print(f"  Concordance:           {concordance:.1%}")

    # -------------------------------------------------------------------------
    # Step 8: Power delta (paired p-value comparison)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("POWER DELTA (paired p-values)")
    print(f"{'='*60}")

    # Collect p-values for intermediaries tested by both
    disc_pvals = {}
    for hop in disc_result.hops:
        for arm in hop.all_arms:
            disc_pvals[arm.intermediary] = arm.p_value

    hybrid_pvals = {}
    for hop in hybrid_result.hops:
        for arm in hop.all_arms:
            hybrid_pvals[arm.intermediary] = arm.p_value

    common_arms = sorted(set(disc_pvals) & set(hybrid_pvals))
    if common_arms:
        disc_p = np.array([disc_pvals[a] for a in common_arms])
        hybrid_p = np.array([hybrid_pvals[a] for a in common_arms])
        log_ratio = np.log10(hybrid_p + 1e-300) - np.log10(disc_p + 1e-300)

        print(f"  Common intermediaries: {len(common_arms)}")
        print(f"  Mean log10(p_hybrid/p_discrete): {log_ratio.mean():.3f}")
        print(f"  Median: {np.median(log_ratio):.3f}")
        more_sig = (hybrid_p < disc_p).sum()
        less_sig = (hybrid_p > disc_p).sum()
        print(f"  RWR makes more significant: {more_sig}")
        print(f"  RWR makes less significant: {less_sig}")

    # -------------------------------------------------------------------------
    # Step 9: RWR-only candidates
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RWR-ONLY CANDIDATES (beyond hop frontier)")
    print(f"{'='*60}")

    rwr_only = hybrid_result.rwr_only_candidates
    print(f"  {len(rwr_only)} candidates not reached by discrete discovery")
    for c in rwr_only[:20]:
        t_stat = abs_t_stats.get(c.gene, 0.0)
        print(f"    {c.gene:>12s}  score={c.ranking_score:.4f} ({c.score_type})  "
              f"signed={c.rwr_signed:+.2f}  rank={c.ranking_rank:4d}  "
              f"|t|={t_stat:.2f}")

    # -------------------------------------------------------------------------
    # Step 10: Save results
    # -------------------------------------------------------------------------
    report = {
        "seed": args.network_query,
        "max_hops": args.max_hops,
        "rwr_hops": args.rwr_hops,
        "n_rwr_edges": len(rwr_edges),
        "n_rwr_act_edges": signed_rwr.n_act_edges,
        "n_rwr_rep_edges": signed_rwr.n_rep_edges,
        "n_rwr_nodes": len(signed_rwr.node_names),
        "discrete_significant": len(disc_sig),
        "hybrid_significant": len(hybrid_sig),
        "overlap": len(overlap),
        "concordance": len(overlap) / max(len(disc_sig), 1),
        "disc_only_arms": sorted(disc_only),
        "hybrid_only_arms": sorted(hybrid_only),
        "n_rwr_only_candidates": len(rwr_only),
        "rwr_only_top10": [
            {"gene": c.gene, "score": c.ranking_score, "score_type": c.score_type,
             "signed": c.rwr_signed, "rank": c.ranking_rank}
            for c in rwr_only[:10]
        ],
        "discovery_gain_per_hop": hybrid_result.discovery_gain_per_hop,
    }

    out_path = args.output_dir / "hybrid_comparison.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
