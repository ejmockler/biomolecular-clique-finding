#!/usr/bin/env python3
"""Run the recursive discovery pipeline on three contrasts for C9orf72 specificity analysis.

Contrasts:
  1. C9ORF72 vs SPORADIC  — what's unique to C9orf72 carriers (already done)
  2. C9ORF72 vs CONTROL   — total C9 disease signal vs healthy
  3. SPORADIC vs CONTROL   — general ALS signal vs healthy

Comparing hop-by-hop results across all three tells us which regulatory
cascades are C9-specific vs shared ALS vs general neurodegeneration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_groups(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    """Resolve three groups from AnswerALS metadata."""
    c9 = metadata[
        (metadata["ClinReport_Mutations_Details"] == "C9orf72")
        | (metadata["C9orf72_repeat_length"] >= 30)
    ]
    known_muts = [
        "C9orf72", "SOD1", "FUS", "TARDBP", "TARDBP (TDP43)",
        "SETX", "Multiple", "Other",
    ]
    sporadic = metadata[
        (metadata["phenotype"] == "CASE")
        & (~metadata["ClinReport_Mutations_Details"].isin(known_muts))
        & ((metadata["C9orf72_repeat_length"] < 30) | metadata["C9orf72_repeat_length"].isna())
    ]
    control = metadata[metadata["phenotype"] == "CTRL"]
    return {
        "C9ORF72": c9.index,
        "SPORADIC": sporadic.index,
        "CONTROL": control.index,
    }


def run_discovery_for_contrast(
    contrast_name: str,
    contrast: tuple[str, str],
    data: np.ndarray,
    feature_ids: list[str],
    metadata: pd.DataFrame,
    groups: dict[str, pd.Index],
    target_set_path: Path,
    output_dir: Path,
    indra_env_file: Path = Path(".env"),
    n_rotations: int = 9999,
    max_hops: int = 6,
    seed_null_b: int = 30,
    covariates: list[str] | None = None,
    compute_overlap: bool = False,
    use_competitive: bool = False,
):
    """Fit engine for a contrast and run recursive discovery."""
    from cliquefinder.stats.rotation import (
        RotationTestEngine, RotationTestConfig, SetStatistic,
    )
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.target_set import TargetSet
    from cliquefinder.stats.discovery_bridge import DiscoveryBridge
    from causal_path_scoring.core.reliability import Edge as CPSEdge
    from causal_path_scoring.core.discovery import run_discovery
    from indra_belief.noise_model import (
        compute_edge_reliability_with_contradiction,
        compute_edge_reliability,
    )

    cond1, cond2 = contrast
    print(f"\n{'=' * 70}")
    print(f"CONTRAST: {contrast_name}  ({cond1} vs {cond2})")
    print("=" * 70)

    # Subset metadata to the two groups
    keep_samples = groups[cond1].union(groups[cond2])
    sub_meta = metadata.loc[metadata.index.intersection(keep_samples)].copy()

    # Assign condition labels
    sub_meta["_condition"] = None
    sub_meta.loc[sub_meta.index.isin(groups[cond1]), "_condition"] = cond1
    sub_meta.loc[sub_meta.index.isin(groups[cond2]), "_condition"] = cond2
    sub_meta = sub_meta.dropna(subset=["_condition"])

    # Align data columns
    sample_id_to_idx = {s: i for i, s in enumerate(metadata.index)}
    # Data columns are in metadata.index order from the caller
    # We need to re-index
    aligned_indices = [sample_id_to_idx[s] for s in sub_meta.index if s in sample_id_to_idx]
    sub_meta = sub_meta.loc[[s for s in sub_meta.index if s in sample_id_to_idx]]
    sub_data = data[:, aligned_indices]

    n1 = (sub_meta["_condition"] == cond1).sum()
    n2 = (sub_meta["_condition"] == cond2).sum()
    print(f"  {cond1}: n={n1}")
    print(f"  {cond2}: n={n2}")
    print(f"  Total: {len(sub_meta)} samples, {sub_data.shape[0]} proteins")

    # Fit ROAST engine for this contrast
    print("  Fitting ROAST engine...")
    engine = RotationTestEngine(sub_data, feature_ids, sub_meta)
    # Use caller-specified covariates (already validated for existence)
    fit_covariates = [c for c in (covariates or []) if c in sub_meta.columns]
    engine.fit(
        conditions=[cond1, cond2],
        contrast=(cond1, cond2),
        condition_column="_condition",
        covariates=fit_covariates,
    )
    print(f"  Engine fitted (covariates: {fit_covariates or 'none'})")

    # Load target set
    ts = TargetSet.load(target_set_path)
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=False)

    # Build adjacency from edge metadata
    seed = ts.gene_symbol  # "C9orf72"
    disc_adjacency = {seed: []}
    for sym, edges in ts.edge_metadata.items():
        reliability, direction, contradictory = compute_edge_reliability_with_contradiction(edges)
        disc_adjacency[seed].append(CPSEdge(
            source=seed, target=sym,
            reliability=reliability,
            edge_type=direction,
        ))
    print(f"  Adjacency: {seed} → {len(disc_adjacency[seed])} direct targets")

    # Differential effects from this contrast
    # Run per-protein EB-moderated differential to get t-statistics
    print("  Computing per-protein differential statistics...")
    from cliquefinder.stats.differential import run_protein_differential
    cov_df = sub_meta[fit_covariates] if fit_covariates else None
    protein_df = run_protein_differential(
        data=sub_data,
        feature_ids=feature_ids,
        sample_condition=sub_meta["_condition"],
        contrast=(cond1, cond2),
        eb_moderation=True,
        covariates_df=cov_df,
    )
    disc_effects = {}
    disc_directions = {}
    for _, row in protein_df.iterrows():
        if pd.notna(row.get("t_statistic")):
            fid = row["feature_id"]
            disc_effects[fid] = abs(float(row["t_statistic"]))
            disc_directions[fid] = "down" if float(row["t_statistic"]) < 0 else "up"

    # Seed null pool
    seed_neighbors = set(ts.edge_metadata.keys())
    seed_null_pool = sorted(
        sym for sym, fid in symbol_to_feature.items()
        if sym != seed
        and sym not in seed_neighbors
        and fid in engine.gene_to_idx
    )
    print(f"  Seed null pool: {len(seed_null_pool)} candidates")

    # ROAST config matching pipeline rotations
    roast_config = RotationTestConfig(
        statistics=[SetStatistic.MSQ],
        n_rotations=n_rotations,
        seed=42,
    )

    # Run discovery
    print(f"  Running {max_hops}-hop discovery...")
    t0 = time.time()
    target_cache_snapshot = {}
    with DiscoveryBridge(
        engine, symbol_to_feature,
        env_file=indra_env_file,
        min_evidence=1,
        min_reliability=0.0,
        min_sources=1,
        roast_config=roast_config,
        use_competitive=use_competitive,
    ) as bridge:
        disc_result = run_discovery(
            seed=seed,
            adjacency=disc_adjacency,
            test_gene_set=bridge.test_gene_set,
            target_to_effect=disc_effects,
            target_to_direction=disc_directions,
            measurable_genes=set(),
            max_hops=max_hops,
            min_targets_per_arm=5,
            fdr_threshold=0.05,
            effect_threshold=1.5,
            get_targets=bridge.get_targets,
            verbose=True,
            hierarchical_fdr=True,
            seed_null_stop=True,
            seed_null_b=seed_null_b,
            seed_null_threshold=0.1,
            seed_null_pool=seed_null_pool,
            seed_null_rng=np.random.default_rng(42),
            knockoff_filter=True,
            knockoff_rng=np.random.default_rng(42),
        )
        # Snapshot the target cache before bridge.close() clears it
        if compute_overlap:
            target_cache_snapshot = {
                k: list(v) for k, v in bridge._target_cache.items()
            }

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")
    print(disc_result.summary())

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    disc_dict = disc_result.to_dict()
    disc_dict["contrast"] = contrast_name
    disc_dict["contrast_groups"] = {cond1: n1, cond2: n2}
    disc_dict["elapsed_seconds"] = round(elapsed, 1)

    # Overlap quantification
    if compute_overlap and target_cache_snapshot:
        from cliquefinder.stats.overlap_analysis import (
            annotate_discovery_with_overlap,
        )
        # Build gene_sets_per_hop from the cached targets
        # Each hop's arms reference intermediaries whose targets are in the cache
        gene_sets_per_hop = {}
        for hop_data in disc_dict.get("hops", []):
            hop_num = hop_data.get("hop")
            arms = hop_data.get("all_arms", [])
            hop_sets = {}
            for arm in arms:
                intermediary = arm.get("intermediary", "")
                targets = target_cache_snapshot.get(intermediary, [])
                if targets:
                    hop_sets[intermediary] = set(targets)
            if hop_sets:
                gene_sets_per_hop[hop_num] = hop_sets

        disc_dict = annotate_discovery_with_overlap(
            disc_dict, gene_sets_per_hop=gene_sets_per_hop
        )
        # Print overlap summary per hop
        print("  Overlap analysis:")
        for hop_data in disc_dict.get("hops", []):
            ov = hop_data.get("overlap", {})
            if "m_eff" in ov:
                print(
                    f"    Hop {hop_data['hop']}: "
                    f"M_eff={ov['m_eff']:.1f}/{ov['m_nominal']} "
                    f"(ratio={ov['ratio']:.3f}, "
                    f"median_J={ov['median_jaccard']:.3f})"
                )

    out_path = output_dir / f"discovery_{contrast_name}.json"
    with open(out_path, "w") as f:
        json.dump(disc_dict, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return disc_dict


def run_gradient_for_contrast(
    contrast_name: str,
    contrast: tuple[str, str],
    data: np.ndarray,
    feature_ids: list[str],
    metadata: pd.DataFrame,
    groups: dict[str, pd.Index],
    output_dir: Path,
    indra_env_file: Path = Path(".env"),
    covariates: list[str] | None = None,
    max_hops: int = 2,
    n_permutations: int = 999,
    seed: str = "C9orf72",
):
    """Fit engine for a contrast and run gradient-based discovery."""
    from cliquefinder.stats.rotation import RotationTestEngine
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.discovery_bridge import DiscoveryBridge

    cond1, cond2 = contrast
    print(f"\n{'=' * 70}")
    print(f"GRADIENT: {contrast_name}  ({cond1} vs {cond2})")
    print("=" * 70)

    # Subset and align (same as per-arm)
    keep_samples = groups[cond1].union(groups[cond2])
    sub_meta = metadata.loc[metadata.index.intersection(keep_samples)].copy()
    sub_meta["_condition"] = None
    sub_meta.loc[sub_meta.index.isin(groups[cond1]), "_condition"] = cond1
    sub_meta.loc[sub_meta.index.isin(groups[cond2]), "_condition"] = cond2
    sub_meta = sub_meta.dropna(subset=["_condition"])

    sample_id_to_idx = {s: i for i, s in enumerate(metadata.index)}
    aligned_indices = [sample_id_to_idx[s] for s in sub_meta.index if s in sample_id_to_idx]
    sub_meta = sub_meta.loc[[s for s in sub_meta.index if s in sample_id_to_idx]]
    sub_data = data[:, aligned_indices]

    n1 = (sub_meta["_condition"] == cond1).sum()
    n2 = (sub_meta["_condition"] == cond2).sum()
    print(f"  {cond1}: n={n1},  {cond2}: n={n2}")
    print(f"  {sub_data.shape[0]} proteins")

    # Fit engine
    engine = RotationTestEngine(sub_data, feature_ids, sub_meta)
    fit_covariates = [c for c in (covariates or []) if c in sub_meta.columns]
    engine.fit(
        conditions=[cond1, cond2],
        contrast=(cond1, cond2),
        condition_column="_condition",
        covariates=fit_covariates,
    )
    print(f"  Engine fitted (covariates: {fit_covariates or 'none'})")

    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=False)

    # Run gradient via bridge
    print(f"  Building {max_hops}-hop neighborhood via INDRA...")
    t0 = time.time()
    with DiscoveryBridge(
        engine, symbol_to_feature,
        env_file=indra_env_file,
        min_evidence=1,
        min_reliability=0.0,
        min_sources=1,
    ) as bridge:
        result = bridge.run_gradient(
            seed=seed,
            max_hops=max_hops,
            n_permutations=n_permutations,
            rng_seed=42,
        )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Print shell summary
    print(f"\n  Gradient: slope={result.slope:.4f}, p={result.slope_pvalue:.4f}")
    print(f"  Spearman: rho={result.spearman_rho:.4f}, p={result.spearman_pvalue:.4f}")
    print(f"  Active horizon: {result.active_horizon}")
    print(f"  Background mean|t|: {result.background_mean_abs_t:.4f}")
    print(f"  {'Hop':>4}  {'n_genes':>8}  {'mean|t|':>10}  {'median|t|':>10}")
    for s in result.shells:
        print(f"  {s.hop:>4}  {s.n_genes:>8}  {s.mean_abs_t:>10.4f}  {s.median_abs_t:>10.4f}")

    if result.stratified:
        print("\n  Edge-quality stratification (Bonferroni-corrected):")
        for tier, tier_r in result.stratified.items():
            print(f"    {tier}: slope={tier_r.slope:.4f}, p={tier_r.slope_pvalue:.4f}, "
                  f"n={tier_r.n_genes_total}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dict = result.to_dict()
    out_dict["contrast"] = contrast_name
    out_dict["contrast_groups"] = {cond1: n1, cond2: n2}
    out_dict["elapsed_seconds"] = round(elapsed, 1)

    out_path = output_dir / f"gradient_{contrast_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_dict, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return out_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("output/proteomics/all_als.data.csv"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("output/proteomics/all_als.metadata.csv"))
    parser.add_argument("--target-set", type=Path,
                        default=Path("output/validation/c9orf72_phase2/indra_targets.json"))
    parser.add_argument("--indra-env-file", type=Path,
                        default=Path(".env"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/validation/specificity_triangle"))
    parser.add_argument("--n-rotations", type=int, default=9999)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--seed-null-b", type=int, default=30)
    parser.add_argument("--contrasts", nargs="+",
                        choices=["C9_vs_SPORADIC", "C9_vs_CTRL", "SPORADIC_vs_CTRL"],
                        default=["C9_vs_SPORADIC", "C9_vs_CTRL", "SPORADIC_vs_CTRL"],
                        help="Which contrasts to run (default: all three)")
    parser.add_argument("--covariates", nargs="*", metavar="COL", default=["Sex"],
                        help="Covariate columns from metadata CSV for design matrix "
                             "adjustment (default: Sex). Pass no arguments to disable.")
    parser.add_argument("--covariate-report", action="store_true", default=False,
                        help="Print a covariate confounding assessment table before "
                             "running the analysis.")
    parser.add_argument("--compute-overlap", action="store_true", default=False,
                        help="Compute gene set overlap statistics per hop using "
                             "Li & Ji (2005) effective independent tests. Adds "
                             "'overlap' field to each hop in the output JSON.")
    parser.add_argument("--use-competitive", action="store_true", default=False,
                        help="Use competitive z-score with Camera VIF correction "
                             "instead of raw ROAST MSQ for extension decisions.")
    parser.add_argument("--gradient", action="store_true", default=False,
                        help="Use gradient-based discovery (perturbation decay) "
                             "instead of per-arm binary gate discovery.")
    parser.add_argument("--gradient-hops", type=int, default=2,
                        help="Max BFS depth for gradient mode (default: 2). "
                             "Each hop requires INDRA queries for frontier genes.")
    parser.add_argument("--gradient-perms", type=int, default=999,
                        help="Number of degree-preserving permutations for "
                             "gradient null distribution (default: 999).")
    parser.add_argument("--seed", type=str, default="C9orf72",
                        help="Seed gene symbol for gradient mode (default: C9orf72)")
    args = parser.parse_args()

    # Reject incompatible flag combinations
    if args.gradient and args.use_competitive:
        parser.error(
            "--use-competitive is incompatible with --gradient. "
            "The gradient path aggregates over shells, not per-set competitive z."
        )
    if args.gradient and args.compute_overlap:
        parser.error(
            "--compute-overlap is a binary-gate concern (arm overlap at each hop). "
            "Gradient mode has no arm structure."
        )

    # Load data
    print(f"Loading data: {args.data}")
    data_df = pd.read_csv(args.data, index_col=0)
    feature_ids = list(data_df.index)
    print(f"  {data_df.shape[0]} proteins × {data_df.shape[1]} samples")

    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata, index_col=0)

    # Align
    common = [s for s in data_df.columns if s in metadata.index]
    metadata = metadata.loc[common]
    data = data_df[common].values  # (n_features, n_samples)

    # Use Sex column (handle case variations)
    if "Sex" not in metadata.columns and "SEX" in metadata.columns:
        metadata["Sex"] = metadata["SEX"]

    # Validate and apply covariates
    covariates = args.covariates if args.covariates else []
    if covariates:
        missing_cols = [c for c in covariates if c not in metadata.columns]
        if missing_cols:
            print(f"  WARNING: Covariate columns not found in metadata: {missing_cols}")
            covariates = [c for c in covariates if c in metadata.columns]
        if covariates:
            cov_nan_mask = metadata[covariates].isna().any(axis=1)
            n_dropped = int(cov_nan_mask.sum())
            if n_dropped > 0:
                nan_cols = [c for c in covariates if metadata[c].isna().any()]
                print(f"  Dropped {n_dropped} samples with missing covariates: {nan_cols}")
                keep_mask = ~cov_nan_mask
                metadata = metadata.loc[keep_mask]
                data = data[:, keep_mask.values]

    # Resolve groups
    groups = resolve_groups(metadata)
    print(f"\nCohort breakdown:")
    for name, idx in groups.items():
        print(f"  {name}: n={len(idx)}")

    # Covariate confounding report
    if args.covariate_report and covariates:
        from cliquefinder.stats.covariate_diagnostics import assess_covariate_confounding
        # Build a temporary condition column for the report
        temp_meta = metadata.copy()
        temp_meta["_group"] = None
        for gname, gidx in groups.items():
            temp_meta.loc[temp_meta.index.isin(gidx), "_group"] = gname
        temp_meta = temp_meta.dropna(subset=["_group"])

        report = assess_covariate_confounding(
            metadata=temp_meta,
            group_column="_group",
            covariates=covariates,
            groups=list(groups.keys()),
        )
        print(f"\n{report.format_table()}\n")
        if report.has_confounded():
            print("  WARNING: Confounded covariates detected (p < 0.05). "
                  "Interpret results with caution.\n")

    # Define contrasts
    contrast_map = {
        "C9_vs_SPORADIC": ("C9ORF72", "SPORADIC"),
        "C9_vs_CTRL": ("C9ORF72", "CONTROL"),
        "SPORADIC_vs_CTRL": ("SPORADIC", "CONTROL"),
    }

    results = {}
    for cname in args.contrasts:
        contrast = contrast_map[cname]
        if args.gradient:
            result = run_gradient_for_contrast(
                contrast_name=cname,
                contrast=contrast,
                data=data,
                feature_ids=feature_ids,
                metadata=metadata,
                groups=groups,
                output_dir=args.output,
                indra_env_file=args.indra_env_file,
                covariates=covariates,
                max_hops=args.gradient_hops,
                n_permutations=args.gradient_perms,
                seed=args.seed,
            )
        else:
            result = run_discovery_for_contrast(
                contrast_name=cname,
                contrast=contrast,
                data=data,
                feature_ids=feature_ids,
                metadata=metadata,
                groups=groups,
                target_set_path=args.target_set,
                output_dir=args.output,
                indra_env_file=args.indra_env_file,
                n_rotations=args.n_rotations,
                max_hops=args.max_hops,
                seed_null_b=args.seed_null_b,
                covariates=covariates,
                compute_overlap=args.compute_overlap,
                use_competitive=args.use_competitive,
            )
        results[cname] = result

    # Print comparison summary
    print(f"\n{'=' * 70}")
    if args.gradient:
        print("GRADIENT SPECIFICITY SUMMARY")
        print("=" * 70)
        print(f"{'Contrast':<20} {'slope':>8} {'slope_p':>8} {'rho':>8} {'rho_p':>8} {'horizon':>8}")
        print("-" * 70)
        for cname in args.contrasts:
            r = results[cname]
            print(f"{cname:<20} {r['slope']:>8.4f} {r['slope_pvalue']:>8.4f} "
                  f"{r['spearman_rho']:>8.4f} {r['spearman_pvalue']:>8.4f} "
                  f"{r['active_horizon']:>8}")
    else:
        print("SPECIFICITY TRIANGLE SUMMARY")
        print("=" * 70)
        print(f"{'Contrast':<20} {'Hop':>4} {'Sig/Total':>12} {'π̂₀':>8} {'Stop':>16}")
        print("-" * 70)
        for cname in args.contrasts:
            r = results[cname]
            for hop in r.get("hops", []):
                n_arms = len(hop.get("all_arms", []))
                n_sig = hop["n_significant"]
                pi0 = hop.get("pi0", float("nan"))
                stop = hop.get("stop_reason", "")
                snp = hop.get("seed_null_pvalue")
                label = f"{cname}" if hop["hop"] == 1 else ""
                snp_str = f"  snp={snp:.3f}" if snp else ""
                print(f"{label:<20} {hop['hop']:>4} {n_sig:>5}/{n_arms:<5} {pi0:>8.3f} {stop:>16}{snp_str}")
            print()

    # Save combined summary (mode-specific filename to avoid schema collision)
    summary_filename = (
        "gradient_summary.json" if args.gradient else "specificity_summary.json"
    )
    summary_path = args.output / summary_filename
    if args.gradient:
        summary = {}
        for cname in args.contrasts:
            r = results[cname]
            summary[cname] = {
                "groups": r.get("contrast_groups", {}),
                "slope": r["slope"],
                "slope_pvalue": r["slope_pvalue"],
                "spearman_rho": r["spearman_rho"],
                "spearman_pvalue": r["spearman_pvalue"],
                "active_horizon": r["active_horizon"],
                "n_genes_total": r["n_genes_total"],
                "shells": r["shells"],
            }
    else:
        summary = {}
        for cname in args.contrasts:
            r = results[cname]
            summary[cname] = {
                "groups": r.get("contrast_groups", {}),
                "hops": [
                    {
                        "hop": h["hop"],
                        "n_significant": h["n_significant"],
                        "n_tested": len(h.get("all_arms", [])),
                        "pi0": h.get("pi0"),
                        "stop_reason": h.get("stop_reason", ""),
                        "seed_null_pvalue": h.get("seed_null_pvalue"),
                    }
                    for h in r.get("hops", [])
                ],
            }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
