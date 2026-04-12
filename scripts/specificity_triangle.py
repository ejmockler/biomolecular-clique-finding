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
    sex_col = "Sex" if "Sex" in sub_meta.columns else ("SEX" if "SEX" in sub_meta.columns else None)
    engine.fit(
        conditions=[cond1, cond2],
        contrast=(cond1, cond2),
        condition_column="_condition",
        covariates=[sex_col] if sex_col else [],
    )
    print(f"  Engine fitted")

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
    cov_df = sub_meta[[sex_col]] if sex_col else None
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
    with DiscoveryBridge(
        engine, symbol_to_feature,
        env_file=indra_env_file,
        min_evidence=1,
        min_reliability=0.0,
        min_sources=1,
        roast_config=roast_config,
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

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")
    print(disc_result.summary())

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    disc_dict = disc_result.to_dict()
    disc_dict["contrast"] = contrast_name
    disc_dict["contrast_groups"] = {cond1: n1, cond2: n2}
    disc_dict["elapsed_seconds"] = round(elapsed, 1)

    out_path = output_dir / f"discovery_{contrast_name}.json"
    with open(out_path, "w") as f:
        json.dump(disc_dict, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return disc_dict


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
    args = parser.parse_args()

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

    # Resolve groups
    groups = resolve_groups(metadata)
    print(f"\nCohort breakdown:")
    for name, idx in groups.items():
        print(f"  {name}: n={len(idx)}")

    # Define contrasts
    contrast_map = {
        "C9_vs_SPORADIC": ("C9ORF72", "SPORADIC"),
        "C9_vs_CTRL": ("C9ORF72", "CONTROL"),
        "SPORADIC_vs_CTRL": ("SPORADIC", "CONTROL"),
    }

    results = {}
    for cname in args.contrasts:
        contrast = contrast_map[cname]
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
        )
        results[cname] = result

    # Print comparison summary
    print(f"\n{'=' * 70}")
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

    # Save combined summary
    summary_path = args.output / "specificity_summary.json"
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
