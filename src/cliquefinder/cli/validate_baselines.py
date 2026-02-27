"""
Validate-baselines orchestrator command.

Runs the full validation suite for network differential enrichment:
1. Covariate-adjusted ROAST + enrichment (Phase 1)
2. Multi-contrast specificity (Phase 2)
3. Label permutation null — stratified + free (Phase 3)
4. Sex-matched subsampling reanalysis (Phase 4)
5. Negative control gene sets (Phase 5)
6. Aggregate into validation_report.json

Usage:
    cliquefinder validate-baselines \\
        --data data.csv --metadata metadata.csv \\
        --network-query C9ORF72 --cohort-config cohorts/three_group_als.yaml \\
        --output output/validation/ \\
        --covariates Sex
"""

# Warning convention:
#   warnings.warn() -- user-facing (convergence, deprecated, sample size)
#   logger.warning() -- operator-facing (fallback, retry, missing data, phase failure)

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cliquefinder.cli._validators import _positive_int, _probability

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the validate-baselines subcommand."""
    parser = subparsers.add_parser(
        "validate-baselines",
        help="Run comprehensive baseline validation suite for network enrichment",
        description=(
            "Validates network differential enrichment through multiple "
            "complementary analyses: covariate adjustment, multi-contrast "
            "specificity, label permutation null, matched subsampling, "
            "and negative control gene sets."
        ),
    )

    # Required arguments
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Path to expression/proteomics data matrix (CSV)",
    )
    parser.add_argument(
        "--metadata", type=Path, required=True,
        help="Path to sample metadata (CSV)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for validation results",
    )
    parser.add_argument(
        "--network-query", type=str, required=True, metavar="GENE",
        help="Gene to query INDRA network for (e.g., C9ORF72)",
    )

    # Cohort / contrast
    parser.add_argument(
        "--cohort-config", type=Path, default=None,
        help="YAML cohort configuration for multi-group analysis",
    )
    parser.add_argument(
        "--condition-col", type=str, default="phenotype",
        help="Metadata column with condition labels (default: phenotype)",
    )
    parser.add_argument(
        "--contrast", nargs=3, action="append", metavar=("NAME", "COND1", "COND2"),
        help="Contrast to test: NAME COND1 COND2 (can specify multiple)",
    )

    # Covariate and matching
    parser.add_argument(
        "--covariates", nargs="+", metavar="COL", default=["Sex"],
        help="Covariates for design matrix adjustment (default: Sex)",
    )
    parser.add_argument(
        "--match-covariates", nargs="+", metavar="COL", default=["Sex"],
        dest="match_vars",
        help="Variables for exact covariate matching (default: Sex)",
    )

    # Permutation settings
    parser.add_argument(
        "--label-permutations", type=_positive_int, default=500,
        help="Number of label permutations (default: 500)",
    )
    parser.add_argument(
        "--permutation-stratify", type=str, default="Sex", metavar="COL",
        dest="stratify_col",
        help="Column for stratified permutation (default: Sex)",
    )

    # Negative controls
    parser.add_argument(
        "--negative-control-sets", type=_positive_int, default=200,
        dest="n_neg_controls",
        help="Number of random gene sets for FPR calibration (default: 200)",
    )

    # INDRA settings
    parser.add_argument(
        "--min-evidence", type=_positive_int, default=1,
        help="Minimum INDRA evidence count (default: 1)",
    )
    parser.add_argument(
        "--indra-env-file", type=Path,
        default=Path(os.environ.get("INDRA_ENV_FILE", Path.home() / ".indra" / ".env")),
        help="Path to .env file with INDRA CoGEx credentials "
             "(default: $INDRA_ENV_FILE or ~/.indra/.env)",
    )

    # General settings
    parser.add_argument(
        "--n-rotations", type=_positive_int, default=9999,
        help="ROAST rotations (default: 9999)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")

    # Bootstrap stability (M-5)
    parser.add_argument(
        "--bootstrap-stability", action="store_true", default=False,
        help="Run bootstrap stability analysis (annotation, not a gate)",
    )
    parser.add_argument(
        "--n-bootstraps", type=_positive_int, default=200,
        help="Number of bootstrap resamples for stability (default: 200)",
    )

    # Interaction terms (M-7)
    parser.add_argument(
        "--interaction", action="store_true", default=False,
        help="Include condition × covariate interaction terms in design matrix",
    )

    # Verdict threshold (N-1)
    parser.add_argument(
        "--alpha", type=_probability, default=0.05,
        help="Significance threshold for phase gates (default: 0.05)",
    )

    parser.set_defaults(func=run_validate_baselines)


def run_validate_baselines(args: argparse.Namespace) -> int:
    """Execute the full baseline validation suite."""
    from cliquefinder.io.loaders import load_csv_matrix
    from cliquefinder.stats.validation_report import ValidationReport

    print("=" * 70)
    print("  BASELINE VALIDATION SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    report = ValidationReport()

    # -----------------------------------------------------------------
    # Seed propagation strategy (M3 audit finding)
    # -----------------------------------------------------------------
    # Each phase that uses random state gets a distinct offset from
    # args.seed, preventing subtle correlations between permutation-
    # based phases that would arise from sharing the same RNG entry
    # point. When args.seed is None, all phase seeds are None (fully
    # random).
    #
    # Phase offsets:
    #   Bootstrap stability:       seed + 0    (runs first, no conflict)
    #   Phase 3 stratified perm:   seed + 1000
    #   Phase 3 free perm:         seed + 2000
    #   Phase 4 matching:          seed + 3000
    #   Phase 5 negative controls: seed + 4000
    # -----------------------------------------------------------------
    _base_seed = args.seed
    _seed_bootstrap = _base_seed if _base_seed is None else _base_seed + 0
    _seed_phase3_strat = _base_seed if _base_seed is None else _base_seed + 1000
    _seed_phase3_free = _base_seed if _base_seed is None else _base_seed + 2000
    _seed_phase4 = _base_seed if _base_seed is None else _base_seed + 3000
    _seed_phase5 = _base_seed if _base_seed is None else _base_seed + 4000

    # --- Load data ---
    print(f"Loading data: {args.data}")
    matrix = load_csv_matrix(args.data)
    print(f"  {matrix.n_features} features × {matrix.n_samples} samples")

    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata, index_col=0)

    # Handle cohort config
    condition_col = args.condition_col
    if args.cohort_config:
        from cliquefinder.cli.differential import resolve_cohort_from_args
        metadata, condition_col, cohort_contrasts = resolve_cohort_from_args(
            metadata=metadata,
            cohort_config=args.cohort_config,
            genetic_contrast=None,
            condition_col=condition_col,
        )
        if cohort_contrasts:
            args.contrast = cohort_contrasts

    # Align
    common_samples = [s for s in matrix.sample_ids if s in metadata.index]
    metadata = metadata.loc[common_samples]
    sample_indices = [list(matrix.sample_ids).index(s) for s in common_samples]
    data = matrix.data[:, sample_indices]
    feature_ids = list(matrix.feature_ids)
    print(f"  Aligned: {len(common_samples)} samples")

    # Parse contrasts
    contrasts = {}
    if args.contrast:
        for name, c1, c2 in args.contrast:
            contrasts[name] = (c1, c2)
    else:
        conditions = sorted(metadata[condition_col].dropna().unique())
        if len(conditions) >= 2:
            contrasts[f"{conditions[0]}_vs_{conditions[1]}"] = (conditions[0], conditions[1])

    primary_contrast_name = list(contrasts.keys())[0]
    primary_contrast = contrasts[primary_contrast_name]

    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Query INDRA network ---
    from cliquefinder.cli.differential import query_network_targets
    print(f"\nQuerying INDRA network for {args.network_query}...")
    network_targets = query_network_targets(
        gene_symbol=args.network_query,
        feature_ids=feature_ids,
        min_evidence=args.min_evidence,
        env_file=args.indra_env_file,
        verbose=True,
    )
    target_gene_ids = list(network_targets.values())
    print(f"  {len(target_gene_ids)} targets found in data")

    # Build covariates DataFrame
    covariates_df = None
    if args.covariates:
        covariates_df = metadata[args.covariates]

    # Build CovariateDesign once for NaN mask consolidation (M-6)
    from cliquefinder.stats.design_matrix import build_covariate_design_matrix
    covariate_design = build_covariate_design_matrix(
        sample_condition=metadata[condition_col],
        conditions=sorted(metadata[condition_col].dropna().unique().tolist()),
        contrast=primary_contrast,
        covariates_df=covariates_df,
        interaction_terms=getattr(args, "interaction", False),
    )

    # =====================================================================
    # OPTIONAL: Bootstrap stability annotation (M-5)
    # =====================================================================
    if args.bootstrap_stability:
        print(f"\n{'=' * 70}")
        print("BOOTSTRAP STABILITY ANALYSIS")
        print("=" * 70)

        from cliquefinder.stats.bootstrap_stability import run_bootstrap_stability

        boot_result = run_bootstrap_stability(
            data=data,
            feature_ids=feature_ids,
            sample_condition=metadata[condition_col],
            contrast=primary_contrast,
            target_gene_ids=target_gene_ids,
            covariates_df=covariates_df,
            covariate_design=covariate_design,
            n_bootstraps=args.n_bootstraps,
            seed=_seed_bootstrap,
            verbose=True,
        )
        report.bootstrap_stability = boot_result["stability"]
        report.bootstrap_ci = boot_result["z_ci"]

    # =====================================================================
    # PHASE 1: Covariate-adjusted enrichment
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: COVARIATE-ADJUSTED ENRICHMENT")
    print("=" * 70)

    from cliquefinder.stats.differential import (
        run_protein_differential,
        run_network_enrichment_test,
    )

    protein_df = None  # Initialize; downstream phases (e.g., Phase 5) check this
    try:
        protein_df = run_protein_differential(
            data=data,
            feature_ids=feature_ids,
            sample_condition=metadata[condition_col],
            contrast=primary_contrast,
            eb_moderation=True,
            target_gene_ids=target_gene_ids,
            verbose=True,
            covariates_df=covariates_df,
            covariate_design=covariate_design,
        )

        enrichment = run_network_enrichment_test(protein_df, verbose=True)
        report.add_phase("covariate_adjusted", enrichment.to_dict())

        # Save phase-specific output
        enrichment_out = args.output / "phase1_covariate_enrichment.json"
        with open(enrichment_out, "w") as f:
            json.dump(enrichment.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning("Phase 1 (covariate_adjusted) failed: %s", e)
        report.add_phase("covariate_adjusted", {"status": "failed", "error": str(e)})
        # protein_df remains None from initialization above; no reassignment needed
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 2: Multi-contrast specificity
    # =====================================================================
    if len(contrasts) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 2: MULTI-CONTRAST SPECIFICITY")
        print("=" * 70)

        try:
            from cliquefinder.stats.specificity import compute_specificity

            enrichment_by_contrast = {}
            for name, contrast_tuple in contrasts.items():
                print(f"\n  Running contrast: {name} ({contrast_tuple[0]} vs {contrast_tuple[1]})")

                # Filter to samples in this contrast's groups.
                # M-6 note: each sub-contrast uses a different sample subset
                # (only samples belonging to the two conditions in that contrast),
                # so the main covariate_design (built from the primary contrast's
                # full sample set) cannot be reused here. Each sub-contrast
                # correctly recomputes its own NaN mask from the subsetted
                # metadata and covariates.
                mask = metadata[condition_col].isin(contrast_tuple)
                sub_data = data[:, mask.values]
                sub_meta = metadata[mask]
                sub_cov = covariates_df[mask] if covariates_df is not None else None

                try:
                    sub_results = run_protein_differential(
                        data=sub_data,
                        feature_ids=feature_ids,
                        sample_condition=sub_meta[condition_col],
                        contrast=contrast_tuple,
                        eb_moderation=True,
                        target_gene_ids=target_gene_ids,
                        verbose=False,
                        covariates_df=sub_cov,
                    )
                    sub_enrichment = run_network_enrichment_test(sub_results, verbose=False)
                    enrichment_by_contrast[name] = sub_enrichment.to_dict()
                    print(f"    z={sub_enrichment.z_score:.2f}, "
                          f"p={sub_enrichment.empirical_pvalue:.4f}")
                except Exception as e:
                    print(f"    Error: {e}")

            if len(enrichment_by_contrast) > 1:
                specificity = compute_specificity(
                    enrichment_by_contrast,
                    primary_contrast=primary_contrast_name,
                    data=data,
                    feature_ids=feature_ids,
                    metadata=metadata,
                    condition_col=condition_col,
                    contrast_tuples=contrasts,
                    target_gene_ids=target_gene_ids,
                    covariates_df=covariates_df,
                    n_interaction_perms=200,
                    seed=args.seed,
                )
                report.add_phase("specificity", specificity.to_dict())
                print(f"\n  Specificity: {specificity.specificity_label}")
                print(f"  {specificity.summary}")

                spec_out = args.output / "phase2_specificity.json"
                with open(spec_out, "w") as f:
                    json.dump(specificity.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("Phase 2 (specificity) failed: %s", e)
            report.add_phase("specificity", {"status": "failed", "error": str(e)})
        report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 3: Label permutation null (stratified + free)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: LABEL PERMUTATION NULL")
    print("=" * 70)

    try:
        from cliquefinder.stats.label_permutation import run_label_permutation_null

        # Stratified permutation
        stratify_by = None
        if args.stratify_col and args.stratify_col in metadata.columns:
            stratify_by = metadata[args.stratify_col].values
            print(f"  Stratification: {args.stratify_col}")

        # M-6: Pass covariate_design to ensure the same NaN mask is used
        # across all permutations. Covariates do not change when labels are
        # permuted, so the same design (and sample_mask) applies throughout.
        print(f"\n  Running stratified permutation ({args.label_permutations} permutations)...")
        strat_result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=metadata[condition_col],
            contrast=primary_contrast,
            target_gene_ids=target_gene_ids,
            n_permutations=args.label_permutations,
            stratify_by=stratify_by,
            covariates_df=covariates_df,
            covariate_design=covariate_design,
            seed=_seed_phase3_strat,
            verbose=True,
        )
        strat_dict = strat_result.to_dict()
        strat_dict["mode"] = "stratified"

        # Free permutation
        print(f"\n  Running free permutation ({args.label_permutations} permutations)...")
        free_result = run_label_permutation_null(
            data=data,
            feature_ids=feature_ids,
            sample_condition=metadata[condition_col],
            contrast=primary_contrast,
            target_gene_ids=target_gene_ids,
            n_permutations=args.label_permutations,
            stratify_by=None,
            covariates_df=covariates_df,
            covariate_design=covariate_design,
            seed=_seed_phase3_free,
            verbose=True,
        )
        free_dict = free_result.to_dict()
        free_dict["mode"] = "free"

        report.add_phase("label_permutation", {
            "stratified": strat_dict,
            "free": free_dict,
            "permutation_pvalue": strat_result.permutation_pvalue,
        })

        perm_out = args.output / "phase3_label_permutation.json"
        with open(perm_out, "w") as f:
            json.dump({"stratified": strat_dict, "free": free_dict}, f, indent=2)
    except Exception as e:
        logger.warning("Phase 3 (label_permutation) failed: %s", e)
        report.add_phase("label_permutation", {"status": "failed", "error": str(e)})
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 4: Sex-matched subsampling reanalysis
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 4: MATCHED SUBSAMPLING REANALYSIS")
    print("=" * 70)

    try:
        from cliquefinder.stats.matching import exact_match_covariates

        match_result = exact_match_covariates(
            metadata=metadata,
            group_col=condition_col,
            match_vars=args.match_vars,
            groups=list(primary_contrast),
            seed=_seed_phase4,
        )

        print(f"  Original: {match_result.n_original} → Matched: {match_result.n_matched}")

        # M-6 note: matched subsampling produces a different sample subset
        # than the primary analysis, so the main covariate_design (built from
        # the full sample set) does not apply. The matched subset correctly
        # recomputes its own NaN mask from the subsetted covariates.
        matched_data = data[:, match_result.matched_indices]
        matched_meta = metadata.iloc[match_result.matched_indices]
        matched_cov = covariates_df.iloc[match_result.matched_indices] if covariates_df is not None else None

        matched_protein_df = run_protein_differential(
            data=matched_data,
            feature_ids=feature_ids,
            sample_condition=matched_meta[condition_col],
            contrast=primary_contrast,
            eb_moderation=True,
            target_gene_ids=target_gene_ids,
            verbose=True,
            covariates_df=matched_cov,
        )

        matched_enrichment = run_network_enrichment_test(matched_protein_df, verbose=True)
        report.add_phase("matched_reanalysis", {
            **matched_enrichment.to_dict(),
            "n_original": match_result.n_original,
            "n_matched": match_result.n_matched,
            "match_vars": match_result.match_vars,
        })

        matched_out = args.output / "phase4_matched_enrichment.json"
        with open(matched_out, "w") as f:
            json.dump(matched_enrichment.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning("Phase 4 (matched_reanalysis) failed: %s", e)
        report.add_phase("matched_reanalysis", {"status": "failed", "error": str(e)})
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 5: Negative control gene sets
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 5: NEGATIVE CONTROL GENE SETS")
    print("=" * 70)

    try:
        from cliquefinder.stats.negative_controls import run_negative_control_sets
        from cliquefinder.stats.rotation import RotationTestEngine

        # M-6 note: RotationTestEngine.fit() builds its own design matrix
        # from the full data + metadata + covariates. It uses the same
        # covariate columns listed in args.covariates, so its internal NaN
        # mask is consistent with the covariate_design built above. The
        # engine operates on the full sample set (not a subset), matching
        # Phase 1's scope.
        conditions_list = list(primary_contrast)
        engine = RotationTestEngine(data, feature_ids, metadata)
        engine.fit(
            conditions=conditions_list,
            contrast=primary_contrast,
            condition_column=condition_col,
            covariates=args.covariates,
        )

        # protein_df may be None if Phase 1 failed. Pass it through;
        # run_negative_control_sets() handles None gracefully (skips
        # competitive z-score computation).
        neg_result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_gene_ids,
            target_set_id=f"{args.network_query}_targets",
            n_control_sets=args.n_neg_controls,
            seed=_seed_phase5,
            protein_results=protein_df,
            data=data,
            matching="both",
            verbose=True,
        )

        report.add_phase("negative_controls", neg_result.to_dict())

        neg_out = args.output / "phase5_negative_controls.json"
        with open(neg_out, "w") as f:
            json.dump(neg_result.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning("Phase 5 (negative_controls) failed: %s", e)
        report.add_phase("negative_controls", {"status": "failed", "error": str(e)})
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # AGGREGATE REPORT
    # =====================================================================
    report.compute_verdict(alpha=args.alpha)
    report.save(args.output / "validation_report.json")
    report.print_summary()

    print(f"\nAll results saved to: {args.output}")
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0
