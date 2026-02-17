"""
CLI for cross-method differential abundance comparison.

Runs multiple statistical methods on cliques and computes concordance:
- OLS: Fixed effects ordinary least squares
- LMM: Linear mixed model with random subject effects
- ROAST_MSQ: Rotation-based test (bidirectional, direction-agnostic)
- ROAST_MEAN: Rotation-based test (directional)
- PERMUTATION: Competitive permutation test

Usage:
    cliquefinder compare \
        --data output/proteomics/sporadic.data.csv \
        --metadata output/proteomics/sporadic.metadata.csv \
        --cliques output/proteomics/discovery_exact_unstratified/cliques.csv \
        --output output/proteomics/method_comparison \
        --condition-col phenotype \
        --subject-col subject_id \
        --contrast CASE_vs_CTRL CASE CTRL
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cliquefinder.cohort import resolve_cohort_from_args


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the compare subcommand to the parser."""
    parser = subparsers.add_parser(
        "compare",
        help="Cross-method differential abundance comparison",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input files
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Protein abundance data CSV (features x samples)",
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        required=True,
        help="Sample metadata CSV",
    )
    parser.add_argument(
        "--cliques", "-c",
        type=Path,
        required=True,
        help="Clique definitions CSV (from analyze command)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results",
    )

    # Experimental design
    parser.add_argument(
        "--condition-col",
        type=str,
        default="phenotype",
        help="Metadata column for condition labels (default: phenotype)",
    )
    parser.add_argument(
        "--subject-col",
        type=str,
        default=None,
        help="Metadata column for subject IDs (enables LMM)",
    )
    parser.add_argument(
        "--contrast",
        type=str,
        nargs=3,
        metavar=("NAME", "COND1", "COND2"),
        default=None,
        help="Contrast to test: NAME CONDITION1 CONDITION2. "
             "Required unless --cohort-config or --genetic-contrast is used.",
    )

    # Genetic subtype analysis
    parser.add_argument(
        "--genetic-contrast",
        type=str,
        metavar="MUTATION",
        help="Genetic subtype contrast (e.g., 'C9orf72' for carriers vs sporadic ALS). "
             "Auto-detects genomic columns for biologically accurate carrier definition.",
    )
    parser.add_argument(
        "--cohort-config",
        type=Path,
        metavar="PATH",
        help="YAML/JSON cohort definition file for declarative cohort assignment. "
             "See cohorts/ directory for examples. Overrides --genetic-contrast.",
    )

    # Preprocessing options
    parser.add_argument(
        "--summarization",
        type=str,
        choices=["tmp", "median", "mean", "pca"],
        default="tmp",
        help="Method to summarize proteins to clique (default: tmp = Tukey's Median Polish)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "median", "quantile"],
        default="none",
        help="Normalization method (default: none, assumes pre-normalized)",
    )
    parser.add_argument(
        "--imputation",
        type=str,
        choices=["none", "min_feature", "knn"],
        default="none",
        help="Missing value imputation (default: none)",
    )

    # Method-specific options
    parser.add_argument(
        "--n-rotations",
        type=int,
        default=499,
        help="Number of rotations for ROAST methods (default: 499, sufficient for p<0.05)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for permutation method (default: 1000)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        dest="gpu",
        help="Use GPU acceleration for rotation/permutation testing",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU acceleration, use CPU",
    )
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Concordance options
    parser.add_argument(
        "--concordance-threshold",
        type=float,
        default=0.05,
        help="P-value threshold for agreement classification (default: 0.05)",
    )

    # Interaction testing (optional)
    parser.add_argument(
        "--interaction",
        nargs=2,
        metavar=("FACTOR1_COL", "FACTOR2_COL"),
        help="Add 2x2 factorial interaction testing (e.g., --interaction sex phenotype)",
    )

    # Filtering
    parser.add_argument(
        "--min-proteins",
        type=int,
        default=3,
        help="Minimum proteins required per clique (default: 3)",
    )

    # Stratification
    parser.add_argument(
        "--stratify-by",
        type=str,
        default=None,
        help="Run separate comparisons for each level of this metadata column (e.g., --stratify-by Sex)",
    )

    # Bootstrap subsampling for imbalanced designs
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable bootstrap subsampling for imbalanced designs (case:control ratio > 3)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=100,
        help="Number of bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--bootstrap-ratio",
        type=float,
        default=2.0,
        help="Target case:control ratio for subsampling (default: 2.0)",
    )
    parser.add_argument(
        "--bootstrap-stability",
        type=float,
        default=0.80,
        help="Fraction of bootstraps for 'stable' hit classification (default: 0.80)",
    )
    parser.add_argument(
        "--bootstrap-controls",
        action="store_true",
        dest="bootstrap_controls_flag",
        help="Force bootstrap controls WITH replacement (default: auto-detect based on n_controls >= 50)",
    )
    parser.add_argument(
        "--no-bootstrap-controls",
        action="store_true",
        dest="no_bootstrap_controls_flag",
        help="Force fixed controls mode (all controls used every iteration)",
    )
    parser.add_argument(
        "--mixed-criterion",
        type=str,
        choices=["roast_only", "exclude"],
        default="roast_only",
        help="Stability criterion for mixed-direction cliques: "
             "'roast_only' (default) requires ROAST stable, "
             "'exclude' excludes mixed cliques from stable hits",
    )

    parser.set_defaults(func=run_compare)


def _run_bootstrap_comparison(
    data: np.ndarray,
    feature_ids: list[str],
    metadata: pd.DataFrame,
    cliques: list,
    output_dir: Path,
    args: argparse.Namespace,
    stratum_name: str | None = None,
) -> dict:
    """
    Run bootstrap subsampling analysis for imbalanced designs.

    Returns dict with summary statistics.
    """
    from cliquefinder.stats import run_bootstrap_comparison, BootstrapConfig
    import time

    contrast_name, cond1, cond2 = args.contrast
    contrast = (cond1, cond2)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine bootstrap_controls setting from CLI flags
    # Priority: --no-bootstrap-controls > --bootstrap-controls > auto-detect (None)
    bootstrap_controls = None  # Auto-detect by default
    if getattr(args, 'no_bootstrap_controls_flag', False):
        bootstrap_controls = False
    elif getattr(args, 'bootstrap_controls_flag', False):
        bootstrap_controls = True

    # Configure bootstrap
    config = BootstrapConfig(
        n_bootstraps=args.bootstrap_n,
        target_ratio=args.bootstrap_ratio,
        significance_threshold=args.concordance_threshold,
        stability_threshold=args.bootstrap_stability,
        concordance_threshold=0.50,
        n_rotations=min(args.n_rotations, 499),  # Reduce for speed
        use_gpu=args.gpu,
        seed=args.seed,
        verbose=True,
        bootstrap_controls=bootstrap_controls,
    )

    print(f"\n{'=' * 70}")
    if stratum_name:
        print(f"BOOTSTRAP ANALYSIS for stratum: {stratum_name}")
    else:
        print("BOOTSTRAP ANALYSIS")
    print("=" * 70)

    start_time = time.time()

    # Run bootstrap
    bootstrap_df = run_bootstrap_comparison(
        data=data,
        feature_ids=feature_ids,
        metadata=metadata,
        cliques=cliques,
        condition_column=args.condition_col,
        contrast=contrast,
        config=config,
        subject_column=None,  # Skip LMM for speed in bootstrap
        output_dir=output_dir,
    )

    elapsed_time = time.time() - start_time
    print(f"\nBootstrap analysis completed in {elapsed_time:.1f}s")

    # Summary statistics with direction-aware counts
    if len(bootstrap_df) > 0:
        # Count by direction
        n_coherent = bootstrap_df['direction'].isin(['positive', 'negative']).sum()
        n_mixed = (bootstrap_df['direction'] == 'mixed').sum()
        n_unknown = (bootstrap_df['direction'] == 'unknown').sum()

        # Stable counts
        coherent_mask = bootstrap_df['direction'].isin(['positive', 'negative'])
        n_stable_coherent = int(bootstrap_df.loc[coherent_mask, 'is_robust'].sum()) if coherent_mask.any() else 0

        mixed_mask = bootstrap_df['direction'] == 'mixed'
        n_stable_mixed = int(bootstrap_df.loc[mixed_mask, 'is_robust'].sum()) if mixed_mask.any() else 0

        n_robust_total = int(bootstrap_df['is_robust'].sum())

        # Concordance for coherent cliques only
        coherent_df = bootstrap_df[coherent_mask]
        mean_concordance = float(coherent_df['method_concordance'].mean()) if len(coherent_df) > 0 else np.nan
    else:
        n_coherent = 0
        n_mixed = 0
        n_unknown = 0
        n_stable_coherent = 0
        n_stable_mixed = 0
        n_robust_total = 0
        mean_concordance = np.nan

    params = {
        "timestamp": datetime.now().isoformat(),
        "stratum": stratum_name,
        "n_samples": len(metadata),
        "analysis_type": "bootstrap",
        "n_bootstraps": config.n_bootstraps,
        "target_ratio": config.target_ratio,
        "stability_threshold": config.stability_threshold,
        "n_cliques_tested": len(bootstrap_df),
        # Direction-aware stats
        "n_coherent_cliques": int(n_coherent),
        "n_mixed_cliques": int(n_mixed),
        "n_unknown_cliques": int(n_unknown),
        "n_stable_coherent": int(n_stable_coherent),
        "n_stable_mixed": int(n_stable_mixed),
        "n_robust": int(n_robust_total),
        "mean_method_concordance_coherent": float(mean_concordance) if not np.isnan(mean_concordance) else None,
        # Not computed in bootstrap mode, but included for schema consistency
        "mean_spearman_rho": None,
        "elapsed_seconds": elapsed_time,
    }

    params_output = output_dir / "bootstrap_analysis_parameters.json"
    with open(params_output, "w") as f:
        json.dump(params, f, indent=2)

    # Print top stable cliques by direction
    if n_robust_total > 0:
        print(f"\n{'=' * 70}")
        print(f"TOP BOOTSTRAP-STABLE CLIQUES ({n_robust_total} total)")
        print("=" * 70)

        # Show coherent cliques
        if n_stable_coherent > 0:
            print(f"\nCoherent (both OLS & ROAST): {n_stable_coherent}")
            coherent_stable = bootstrap_df[
                bootstrap_df['direction'].isin(['positive', 'negative']) &
                bootstrap_df['is_robust']
            ].head(10)
            for _, row in coherent_stable.iterrows():
                direction_label = row['direction'].upper()
                concordance_str = f"concordance={row['method_concordance']:.2f}" if row['method_concordance'] is not None else ""
                print(f"  [{direction_label}] {row['clique_id']}: "
                      f"OLS={row['selection_freq_ols']:.2f}, "
                      f"ROAST={row['selection_freq_roast']:.2f}, "
                      f"{concordance_str}, "
                      f"effect={row['median_effect']:.3f}")

        # Show mixed cliques
        if n_stable_mixed > 0:
            print(f"\nMixed (ROAST only): {n_stable_mixed}")
            mixed_stable = bootstrap_df[
                bootstrap_df['direction'].isin(['mixed', 'unknown']) &
                bootstrap_df['is_robust']
            ].head(10)
            for _, row in mixed_stable.iterrows():
                print(f"  [MIXED] {row['clique_id']}: "
                      f"ROAST={row['selection_freq_roast']:.2f}, "
                      f"OLS={row['selection_freq_ols']:.2f}, "
                      f"effect={row['median_effect']:.3f}")

    return params


def _run_single_comparison(
    data: np.ndarray,
    feature_ids: list[str],
    metadata: pd.DataFrame,
    common_samples: list[str],
    cliques: list,
    output_dir: Path,
    args: argparse.Namespace,
    stratum_name: str | None = None,
) -> dict:
    """
    Run method comparison for a single stratum.

    Returns dict with summary statistics for aggregation.
    """
    from cliquefinder.stats import run_method_comparison
    from cliquefinder.stats.clique_analysis import run_clique_roast_interaction_analysis
    import time

    contrast_name, cond1, cond2 = args.contrast
    contrast = (cond1, cond2)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run method comparison
    print(f"\n{'=' * 70}")
    if stratum_name:
        print(f"Running method comparison for stratum: {stratum_name}")
    else:
        print("Running method comparison...")
    print("=" * 70)

    start_time = time.time()

    comparison = run_method_comparison(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=metadata,
        cliques=cliques,
        condition_column=args.condition_col,
        contrast=contrast,
        subject_column=args.subject_col,
        concordance_threshold=args.concordance_threshold,
        normalization_method=args.normalization,
        imputation_method=args.imputation,
        n_rotations=args.n_rotations,
        n_permutations=args.n_permutations,
        use_gpu=args.gpu,
        seed=args.seed,
        verbose=True,
    )

    elapsed_time = time.time() - start_time
    print(f"\nMethod comparison completed in {elapsed_time:.2f}s")

    # Save comparison results
    print(f"\n{'=' * 70}")
    print("Saving results...")
    print("=" * 70)

    # 1. Wide-format results
    wide_df = comparison.wide_format()
    wide_output = output_dir / "comparison_results.csv"
    wide_df.to_csv(wide_output, index=False)
    print(f"  Comparison results: {wide_output}")

    # 2. Concordance matrix
    conc_matrix = comparison.concordance_matrix()
    conc_output = output_dir / "concordance_matrix.csv"
    conc_matrix.to_csv(conc_output)
    print(f"  Concordance matrix: {conc_output}")

    # 3. Robust hits
    robust_hits = comparison.robust_hits(threshold=args.concordance_threshold)
    if robust_hits:
        robust_df = wide_df[wide_df['clique_id'].isin(robust_hits)]
        robust_output = output_dir / f"robust_hits_p{int(args.concordance_threshold * 100):02d}.csv"
        robust_df.to_csv(robust_output, index=False)
        print(f"  Robust hits (p < {args.concordance_threshold}): {robust_output} ({len(robust_hits)} cliques)")
    else:
        print(f"  Robust hits: None at p < {args.concordance_threshold}")

    # 4. Method-specific hits
    for method in comparison.methods_run:
        specific_hits = comparison.method_specific_hits(method, threshold=args.concordance_threshold)
        if specific_hits:
            specific_df = wide_df[wide_df['clique_id'].isin(specific_hits)]
            specific_output = output_dir / f"method_specific_{method.value}.csv"
            specific_df.to_csv(specific_output, index=False)
            print(f"  {method.value}-specific hits: {specific_output} ({len(specific_hits)} cliques)")

    # 5. Disagreements
    if comparison.disagreement_cases is not None and isinstance(comparison.disagreement_cases, pd.DataFrame):
        if len(comparison.disagreement_cases) > 0:
            disagree_output = output_dir / "disagreements.csv"
            comparison.disagreement_cases.to_csv(disagree_output, index=False)
            print(f"  Disagreements: {disagree_output} ({len(comparison.disagreement_cases)} cases)")

    # 6. Analysis parameters
    params = {
        "timestamp": datetime.now().isoformat(),
        "stratum": stratum_name,
        "n_samples": len(metadata),
        "analysis_type": "single",
        "data": str(args.data),
        "metadata": str(args.metadata),
        "cliques": str(args.cliques),
        "condition_col": args.condition_col,
        "subject_col": args.subject_col,
        "contrast": {
            "name": contrast_name,
            "test_condition": cond1,
            "reference_condition": cond2,
        },
        "summarization": args.summarization,
        "normalization": args.normalization,
        "imputation": args.imputation,
        "n_rotations": args.n_rotations,
        "n_permutations": args.n_permutations,
        "use_gpu": args.gpu,
        "seed": args.seed,
        "concordance_threshold": args.concordance_threshold,
        "min_proteins": args.min_proteins,
        "n_cliques_tested": comparison.n_cliques_tested,
        "methods_run": [m.value for m in comparison.methods_run],
        "mean_spearman_rho": float(comparison.mean_spearman_rho) if not np.isnan(comparison.mean_spearman_rho) else None,
        "mean_cohen_kappa": float(comparison.mean_cohen_kappa) if not np.isnan(comparison.mean_cohen_kappa) else None,
        "n_robust": len(robust_hits) if robust_hits else 0,
        "elapsed_seconds": elapsed_time,
    }
    params_output = output_dir / "analysis_parameters.json"
    with open(params_output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Parameters: {params_output}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(comparison.summary())

    # Print top results per method
    print(f"\nTop cliques by method:")
    for method in comparison.methods_run:
        pval_col = f"{method.value}_pvalue"
        if pval_col in wide_df.columns:
            top = wide_df.nsmallest(5, pval_col)[['clique_id', pval_col]]
            print(f"\n  {method.value}:")
            for _, row in top.iterrows():
                print(f"    {row['clique_id']}: p = {row[pval_col]:.4f}")

    return params


def run_compare(args: argparse.Namespace) -> int:
    """Execute cross-method comparison analysis."""
    from cliquefinder.io.loaders import load_csv_matrix
    from cliquefinder.stats import load_clique_definitions
    from cliquefinder.stats.clique_analysis import run_clique_roast_interaction_analysis

    print("=" * 70)
    print("  Cross-Method Differential Abundance Comparison")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print(f"Loading data: {args.data}")
    matrix = load_csv_matrix(args.data)
    print(f"  {matrix.n_features} features x {matrix.n_samples} samples")

    # Load metadata
    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata, index_col=0)

    # Align metadata with data
    common_samples = [s for s in matrix.sample_ids if s in metadata.index]
    if len(common_samples) < len(matrix.sample_ids):
        print(f"  Warning: {len(matrix.sample_ids) - len(common_samples)} samples missing from metadata")

    metadata = metadata.loc[common_samples]

    # Get data as array
    sample_indices = [list(matrix.sample_ids).index(s) for s in common_samples]
    data = matrix.data[:, sample_indices]
    feature_ids = list(matrix.feature_ids)

    print(f"  Aligned: {len(common_samples)} samples")

    # Handle cohort-based contrast (--cohort-config or --genetic-contrast)
    cohort_config = getattr(args, 'cohort_config', None)
    genetic_contrast = getattr(args, 'genetic_contrast', None)

    metadata, resolved_condition_col, cohort_contrasts = resolve_cohort_from_args(
        metadata=metadata,
        cohort_config=cohort_config,
        genetic_contrast=genetic_contrast,
        condition_col=args.condition_col,
    )

    if cohort_contrasts is not None:
        if args.contrast:
            print(f"  Warning: Ignoring --contrast (overridden by cohort resolution)")
        args.contrast = cohort_contrasts[0]  # compare expects a single triple
        args.condition_col = resolved_condition_col

        # Re-align after cohort filtering (metadata may have been filtered)
        common_samples = [s for s in common_samples if s in metadata.index]
        sample_indices = [list(matrix.sample_ids).index(s) for s in common_samples]
        data = matrix.data[:, sample_indices]
        metadata = metadata.loc[common_samples]
        print(f"  After cohort resolution: {len(common_samples)} samples")

    # Validate contrast is available
    if args.contrast is None:
        print("Error: --contrast is required (or use --cohort-config / --genetic-contrast)")
        return 1

    # Load clique definitions
    print(f"\nLoading cliques: {args.cliques}")
    cliques = load_clique_definitions(args.cliques, min_proteins=args.min_proteins)
    print(f"  {len(cliques)} cliques loaded (min {args.min_proteins} proteins)")

    # Parse contrast
    contrast_name, cond1, cond2 = args.contrast
    print(f"\nContrast: {contrast_name}: {cond1} vs {cond2}")

    # Print settings
    print(f"\nMethod comparison settings:")
    print(f"  Summarization: {args.summarization}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Imputation: {args.imputation}")
    print(f"  Subject column: {args.subject_col or 'None (no LMM)'}")
    print(f"  GPU acceleration: {args.gpu}")
    print(f"  Rotations: {args.n_rotations}")
    print(f"  Permutations: {args.n_permutations}")
    print(f"  Concordance threshold: {args.concordance_threshold}")
    if args.seed:
        print(f"  Random seed: {args.seed}")
    if args.stratify_by:
        print(f"  Stratify by: {args.stratify_by}")

    # Determine strata
    if args.stratify_by:
        stratify_col = args.stratify_by
        if stratify_col not in metadata.columns:
            print(f"ERROR: Stratification column '{stratify_col}' not found in metadata")
            print(f"  Available columns: {list(metadata.columns)}")
            return 1

        # Get valid strata (non-null, exclude "Unknown")
        strata_values = metadata[stratify_col].dropna().unique()
        strata_values = [s for s in strata_values if str(s).lower() != "unknown"]
        strata_values = sorted(strata_values)

        print(f"\nStratification levels: {strata_values}")

        # Show sample counts per stratum × condition
        condition_col = args.condition_col
        print(f"\nSample distribution:")
        for stratum in strata_values:
            stratum_mask = metadata[stratify_col] == stratum
            stratum_meta = metadata[stratum_mask]
            counts = stratum_meta[condition_col].value_counts().to_dict()
            print(f"  {stratum}: {counts}")
    else:
        strata_values = [None]  # Single run, no stratification

    # Create base output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run comparisons
    all_params = []

    for stratum in strata_values:
        if stratum is not None:
            # Filter to this stratum
            stratum_mask = metadata[args.stratify_by] == stratum
            stratum_metadata = metadata[stratum_mask].copy()
            stratum_samples = stratum_metadata.index.tolist()

            # Filter data to match
            sample_mask = [i for i, s in enumerate(common_samples) if s in stratum_samples]
            stratum_data = data[:, sample_mask]
            stratum_common = [common_samples[i] for i in sample_mask]

            # Output to subdirectory
            output_dir = args.output / stratum
            stratum_name = stratum
        else:
            stratum_metadata = metadata
            stratum_data = data
            stratum_common = common_samples
            output_dir = args.output
            stratum_name = None

        # Check we have both conditions in this stratum
        condition_counts = stratum_metadata[args.condition_col].value_counts()
        _, cond1, cond2 = args.contrast
        if cond1 not in condition_counts or cond2 not in condition_counts:
            print(f"\nWARNING: Skipping stratum '{stratum}' - missing condition levels")
            print(f"  Found: {condition_counts.to_dict()}")
            print(f"  Need: {cond1}, {cond2}")
            continue

        if condition_counts.get(cond1, 0) < 3 or condition_counts.get(cond2, 0) < 3:
            print(f"\nWARNING: Skipping stratum '{stratum}' - insufficient samples")
            print(f"  {cond1}: {condition_counts.get(cond1, 0)}, {cond2}: {condition_counts.get(cond2, 0)}")
            continue

        # Check for imbalanced design
        n_cond1 = condition_counts.get(cond1, 0)
        n_cond2 = condition_counts.get(cond2, 0)
        imbalance_ratio = max(n_cond1, n_cond2) / min(n_cond1, n_cond2) if min(n_cond1, n_cond2) > 0 else float('inf')

        use_bootstrap = args.bootstrap
        if imbalance_ratio > 3 and not args.bootstrap:
            print(f"\n  NOTE: Detected imbalanced design ({imbalance_ratio:.1f}:1 ratio)")
            print(f"  Consider using --bootstrap flag for more stable results")

        if use_bootstrap:
            print(f"\n  Running BOOTSTRAP mode (ratio={imbalance_ratio:.1f}:1)")
            params = _run_bootstrap_comparison(
                data=stratum_data,
                feature_ids=feature_ids,
                metadata=stratum_metadata,
                cliques=cliques,
                output_dir=output_dir,
                args=args,
                stratum_name=stratum_name,
            )
        else:
            params = _run_single_comparison(
                data=stratum_data,
                feature_ids=feature_ids,
                metadata=stratum_metadata,
                common_samples=stratum_common,
                cliques=cliques,
                output_dir=output_dir,
                args=args,
                stratum_name=stratum_name,
            )
        all_params.append(params)

    # Save aggregated summary if stratified
    if args.stratify_by and len(all_params) > 1:
        summary_output = args.output / "stratified_summary.json"
        with open(summary_output, "w") as f:
            json.dump(all_params, f, indent=2)
        print(f"\n{'=' * 70}")
        print("STRATIFIED SUMMARY")
        print("=" * 70)
        for p in all_params:
            print(f"\n  {p['stratum']}:")
            print(f"    Samples: {p.get('n_samples', 'N/A')}")
            print(f"    Robust cliques: {p.get('n_robust', 'N/A')}")
            # mean_spearman_rho only exists in regular (non-bootstrap) mode
            if 'mean_spearman_rho' in p and p['mean_spearman_rho'] is not None:
                print(f"    Mean Spearman ρ: {p['mean_spearman_rho']:.3f}")
            # Bootstrap mode has concordance metric instead
            elif 'mean_method_concordance_coherent' in p and p['mean_method_concordance_coherent'] is not None:
                print(f"    Mean concordance: {p['mean_method_concordance_coherent']:.3f}")
        print(f"\n  Summary saved: {summary_output}")

    # Run interaction analysis if requested (on full dataset, not stratified)
    if args.interaction and not args.stratify_by:
        factor1_col, factor2_col = args.interaction
        print(f"\n{'=' * 70}")
        print(f"Running 2x2 factorial interaction analysis: {factor1_col} x {factor2_col}")
        print("=" * 70)

        # Filter to samples with valid values for both factors
        interaction_metadata = metadata.copy()
        valid_mask = (
            interaction_metadata[factor1_col].notna() &
            interaction_metadata[factor2_col].notna()
        )
        interaction_metadata = interaction_metadata[valid_mask]

        # Filter out "Unknown" or similar values if present
        levels1 = interaction_metadata[factor1_col].unique()
        levels2 = interaction_metadata[factor2_col].unique()

        if len(levels1) > 2:
            top_levels = interaction_metadata[factor1_col].value_counts().head(2).index.tolist()
            interaction_metadata = interaction_metadata[interaction_metadata[factor1_col].isin(top_levels)]
            print(f"  Filtered {factor1_col} to: {top_levels}")

        if len(levels2) > 2:
            top_levels = interaction_metadata[factor2_col].value_counts().head(2).index.tolist()
            interaction_metadata = interaction_metadata[interaction_metadata[factor2_col].isin(top_levels)]
            print(f"  Filtered {factor2_col} to: {top_levels}")

        # Get sample indices for filtered data
        valid_samples = interaction_metadata.index.tolist()
        sample_mask = [i for i, s in enumerate(common_samples) if s in valid_samples]
        interaction_data = data[:, sample_mask]
        interaction_sample_ids = [common_samples[i] for i in sample_mask]

        # Reindex metadata to match
        interaction_metadata = interaction_metadata.loc[interaction_sample_ids]

        print(f"  Samples for interaction analysis: {len(interaction_sample_ids)}")
        print(f"  {factor1_col}: {sorted(interaction_metadata[factor1_col].unique())}")
        print(f"  {factor2_col}: {sorted(interaction_metadata[factor2_col].unique())}")

        interaction_df = run_clique_roast_interaction_analysis(
            data=interaction_data,
            feature_ids=feature_ids,
            sample_metadata=interaction_metadata.reset_index(),
            clique_definitions=cliques,
            factor1_column=factor1_col,
            factor2_column=factor2_col,
            n_rotations=args.n_rotations,
            seed=args.seed,
            use_gpu=args.gpu,
            map_ids=True,
            verbose=True,
        )

        interaction_output = args.output / "interaction_results.csv"
        interaction_df.to_csv(interaction_output, index=False)
        print(f"  Interaction results: {interaction_output}")

        # Top interaction hits
        if 'pvalue_msq_mixed' in interaction_df.columns:
            top_interaction = interaction_df.nsmallest(10, 'pvalue_msq_mixed')
            top_output = args.output / "interaction_top_hits.csv"
            top_interaction.to_csv(top_output, index=False)
            print(f"  Top interaction hits: {top_output}")

    print(f"\n{'=' * 70}")
    print(f"Complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    return 0
