"""
Validate-baselines orchestrator command.

Runs the full validation suite for network differential enrichment:
1. Covariate-adjusted ROAST + enrichment (Phase 1)
2. Multi-contrast specificity (Phase 2)
3. Label permutation null — stratified + free (Phase 3)
4. Sex-matched subsampling reanalysis (Phase 4)
5. Negative control gene sets + graph permutation (Phase 5a/5b)
6. Network proximity tests — parameter-free (Phase 6a/6b/6c)
7. Aggregate into validation_report.json

Usage:
    cliquefinder validate-baselines \\
        --data data.csv --metadata metadata.csv \\
        --network-query C9ORF72 --cohort-config cohorts/three_group_als.yaml \\
        --output output/validation/ \\
        --covariates Sex
"""

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
from cliquefinder.utils.fileio import atomic_write_json

# Warning convention:
#   warnings.warn() — user-facing (convergence, deprecated, sample size)
#   logger.warning() — operator-facing (fallback, retry, missing data)
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

    # Graph permutation
    parser.add_argument(
        "--graph-permutations", type=_positive_int, default=100,
        dest="n_graph_perms",
        help="Number of graph node-label permutations (default: 100)",
    )
    parser.add_argument(
        "--graph-size-match", type=float, default=0.5,
        dest="graph_size_match_tolerance",
        help="Size-match tolerance for graph permutation null. "
             "Restricts sampling to regulators with target counts within "
             "±N%% of the target set size (default: 0.5 = ±50%%). "
             "Set to 0 to disable size matching.",
    )

    # INDRA settings
    parser.add_argument(
        "--target-set", type=Path, default=None, dest="target_set_file",
        help="Path to indra_targets.json from a prior analysis run. "
             "When provided, the exact gene set and adjacency are loaded "
             "from this file instead of re-querying INDRA live. This "
             "guarantees experiment/validation use the same gene set.",
    )
    parser.add_argument(
        "--min-evidence", type=_positive_int, default=1,
        help="Minimum INDRA evidence count (default: 1)",
    )
    parser.add_argument(
        "--network-hops", type=_positive_int, default=1, dest="network_hops",
        help="Path length from seed gene (1=direct targets, 2=targets of "
             "targets). Default: 1.",
    )
    parser.add_argument(
        "--min-intermediaries", type=_positive_int, default=1,
        dest="min_intermediaries",
        help="For 2-hop queries, minimum number of independent hop-1 "
             "intermediaries required for a hop-2 target to be included. "
             "Higher values select more convergently supported genes. "
             "Default: 1.",
    )
    parser.add_argument(
        "--min-sources", type=_positive_int, default=None,
        dest="min_sources",
        help="Minimum distinct evidence sources (reading systems/databases) "
             "per INDRA edge. Filters by source diversity, not raw evidence "
             "count. E.g., --min-sources 2 requires corroboration from at "
             "least 2 independent systems.",
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
    parser.add_argument(
        "--rcr-fdr-threshold", type=float, default=0.05,
        dest="rcr_fdr_threshold",
        help="FDR threshold for defining DE genes in reverse causal "
             "reasoning (Phase 6b). Lower values = fewer DE genes = fewer "
             "testable regulators. Use 0.10 or nominal p-value threshold "
             "for exploratory analyses with small effect sizes. Default: 0.05.",
    )
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

    # Checkpoint / resume (ARCH-6)
    parser.add_argument(
        "--force-restart", action="store_true", default=False,
        help="Ignore any existing checkpoint and re-run all phases from scratch",
    )

    # Tunable thresholds (STAT-15)
    parser.add_argument(
        "--specificity-z-threshold", type=float, default=1.5,
        dest="specificity_z_threshold",
        help=(
            "Minimum z-score for a contrast to be considered 'present' in "
            "Phase 2 specificity analysis (default: 1.5). Lower values make "
            "the specificity call more sensitive; higher values more conservative."
        ),
    )
    parser.add_argument(
        "--negative-control-percentile", type=float, default=10.0,
        dest="neg_ctrl_percentile",
        help=(
            "Percentile threshold for Phase 5 negative control pass/fail "
            "(default: 10.0). The target gene set must rank below this "
            "percentile among random control sets. Lower values are stricter."
        ),
    )
    parser.add_argument(
        "--interaction-n-perms", type=int, default=200,
        dest="interaction_n_perms",
        help=(
            "Number of permutations for Phase 2 interaction z-test "
            "(default: 200). Higher values give more precise p-values "
            "but increase runtime linearly."
        ),
    )

    parser.set_defaults(func=run_validate_baselines)


def _compute_params_fingerprint(args: argparse.Namespace) -> str:
    """Hash key analysis parameters for checkpoint integrity checking."""
    import hashlib
    params = {
        "network_query": getattr(args, "network_query", None),
        "target_set_file": str(getattr(args, "target_set_file", None)),
        "min_evidence": getattr(args, "min_evidence", None),
        "min_sources": getattr(args, "min_sources", None),
        "covariates": getattr(args, "covariates", None),
        "contrast": getattr(args, "contrast", None),
        "alpha": getattr(args, "alpha", None),
        "data": str(getattr(args, "data", None)),
    }
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_checkpoint(output_dir: Path):
    """Load existing checkpoint if resuming a previous run.

    Returns a tuple of (ValidationReport, protein_df_or_None) populated
    with any previously completed phase results, allowing the orchestrator
    to skip those phases.  When Phase 1 was previously completed, the
    serialised ``protein_df`` is restored so downstream phases (e.g.
    Phase 5 negative controls) can use it.
    """
    from cliquefinder.stats.validation_report import ValidationReport

    checkpoint_path = output_dir / "validation_checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            report = ValidationReport()
            report.phases = data.get("phases", {})
            report._params_fingerprint = data.get("params_fingerprint")
            n_phases = len(report.phases)
            logger.info(
                f"Loaded checkpoint with {n_phases} completed phase(s) "
                f"from {checkpoint_path}"
            )
            # VAL-3: Restore protein_df from checkpoint if available
            protein_df = None
            protein_df_dict = data.get("protein_df_dict")
            if protein_df_dict:
                protein_df = pd.DataFrame(protein_df_dict)
                logger.info(
                    f"Restored protein_df from checkpoint "
                    f"({protein_df.shape[0]} rows x {protein_df.shape[1]} cols)"
                )
            return report, protein_df
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt checkpoint file, starting fresh: {e}")
    return ValidationReport(), None


def _save_checkpoint(
    report: "ValidationReport",
    output_dir: Path,
    protein_df: "pd.DataFrame | None" = None,
) -> None:
    """Persist current report state so the run can resume later.

    When *protein_df* is not None it is serialised into the checkpoint
    JSON under the ``protein_df_dict`` key so that a resumed run can
    supply it to downstream phases (VAL-3).
    """
    checkpoint_path = output_dir / "validation_checkpoint.json"
    data = report.to_dict()
    # VAL-3: Persist protein_df so Phase 5 can use it on resume
    if protein_df is not None:
        data["protein_df_dict"] = protein_df.to_dict()
    # Store parameter fingerprint for integrity checking on resume
    if hasattr(report, '_params_fingerprint'):
        data["params_fingerprint"] = report._params_fingerprint
    atomic_write_json(checkpoint_path, data)
    logger.debug(f"Checkpoint saved to {checkpoint_path}")


def run_validate_baselines(args: argparse.Namespace) -> int:
    """Execute the full baseline validation suite."""
    from cliquefinder.io.loaders import load_csv_matrix
    from cliquefinder.stats.validation_report import ValidationReport

    print("=" * 70)
    print("  BASELINE VALIDATION SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # --- ARCH-6: Resume from checkpoint if available ---
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_protein_df = None  # VAL-3: may be restored from checkpoint
    if getattr(args, "force_restart", False):
        report = ValidationReport()
        report._params_fingerprint = _compute_params_fingerprint(args)
        # Remove stale checkpoint so subsequent phases don't load it
        checkpoint_path = args.output / "validation_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("  Force restart: removed existing checkpoint")
    else:
        report, checkpoint_protein_df = _load_checkpoint(args.output)
        if report.phases:
            print(f"  Resuming from checkpoint: {list(report.phases.keys())} already complete")
            # Check parameter fingerprint for integrity
            current_fp = _compute_params_fingerprint(args)
            old_fp = getattr(report, '_params_fingerprint', None)
            if old_fp and old_fp != current_fp:
                import warnings
                warnings.warn(
                    f"Checkpoint parameter mismatch: checkpoint was created with "
                    f"different parameters (fingerprint {old_fp} vs current "
                    f"{current_fp}). Resumed phases may be inconsistent. "
                    f"Use --force-restart for a clean run.",
                    UserWarning,
                    stacklevel=1,
                )
            report._params_fingerprint = current_fp
        else:
            report._params_fingerprint = _compute_params_fingerprint(args)

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
    # ARCH-10: Use SeedSequence for statistically independent phase seeds
    if _base_seed is not None:
        from numpy.random import SeedSequence
        _ss = SeedSequence(_base_seed)
        # VAL-6: Phase 2 appended at END to preserve existing seed streams
        # Graph permutation appended at END to preserve all existing streams
        # Phase 6 (network proximity) appended at END to preserve streams
        (_ss_boot, _ss_p3s, _ss_p3f, _ss_p4, _ss_p5, _ss_p2, _ss_p5g, _ss_p6) = _ss.spawn(8)
        _seed_bootstrap = int(_ss_boot.generate_state(1)[0])
        _seed_phase2 = int(_ss_p2.generate_state(1)[0])
        _seed_phase3_strat = int(_ss_p3s.generate_state(1)[0])
        _seed_phase3_free = int(_ss_p3f.generate_state(1)[0])
        _seed_phase4 = int(_ss_p4.generate_state(1)[0])
        _seed_phase5 = int(_ss_p5.generate_state(1)[0])
        _seed_phase5_graph = int(_ss_p5g.generate_state(1)[0])
        # Phase 5c: derive from Phase 5b's stream to avoid changing top-level spawn count
        _ss_p5c = _ss_p5g.spawn(1)[0]
        _seed_phase5c = int(_ss_p5c.generate_state(1)[0])
        _seed_phase6 = int(_ss_p6.generate_state(1)[0])
    else:
        _seed_bootstrap = None
        _seed_phase2 = None
        _seed_phase3_strat = None
        _seed_phase3_free = None
        _seed_phase4 = None
        _seed_phase5 = None
        _seed_phase5_graph = None
        _seed_phase5c = None
        _seed_phase6 = None

    # --- Load data ---
    print(f"Loading data: {args.data}")
    matrix = load_csv_matrix(args.data)
    print(f"  {matrix.n_features} features × {matrix.n_samples} samples")

    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata, index_col=0)

    # Handle cohort config
    condition_col = args.condition_col
    cli_contrasts = args.contrast  # Save CLI --contrast before cohort may overwrite
    if args.cohort_config:
        from cliquefinder.cli.differential import resolve_cohort_from_args
        metadata, condition_col, cohort_contrasts = resolve_cohort_from_args(
            metadata=metadata,
            cohort_config=args.cohort_config,
            genetic_contrast=None,
            condition_col=condition_col,
        )
        # CLI --contrast takes priority over YAML contrast when both are provided
        if cli_contrasts:
            args.contrast = cli_contrasts
        elif cohort_contrasts:
            args.contrast = cohort_contrasts

    # Align (CLI-5: use set + dict for O(n) alignment instead of O(n^2))
    metadata_set = set(metadata.index)
    common_samples = [s for s in matrix.sample_ids if s in metadata_set]
    # CLI-6: Guard against zero common samples
    if len(common_samples) == 0:
        raise ValueError("No common samples between protein data and sample metadata")
    metadata = metadata.loc[common_samples]
    sample_id_to_idx = {s: i for i, s in enumerate(matrix.sample_ids)}
    sample_indices = [sample_id_to_idx[s] for s in common_samples]
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

    # CLI-1: Guard against empty contrasts dict
    if not contrasts:
        print("Error: No contrasts defined. Check --contrast arguments or ensure "
              "metadata has at least 2 distinct conditions in the condition column.")
        return 1

    primary_contrast_name = list(contrasts.keys())[0]
    primary_contrast = contrasts[primary_contrast_name]

    # --- Resolve target gene set ---
    _loaded_target_set = None  # TargetSet if loaded from file
    if args.target_set_file is not None:
        from cliquefinder.stats.target_set import TargetSet
        print(f"\nLoading pinned target set from {args.target_set_file}")
        _loaded_target_set = TargetSet.load(args.target_set_file)
        network_targets = _loaded_target_set.targets

        # Validate feature IDs exist in proteomics data
        missing = [fid for fid in network_targets.values()
                   if fid not in set(feature_ids)]
        if missing:
            print(f"  WARNING: {len(missing)} target feature IDs from "
                  f"target set file not found in proteomics data: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

        print(f"  Loaded {len(network_targets)} targets "
              f"(gene={_loaded_target_set.gene_symbol}, "
              f"{_loaded_target_set.n_hops}-hop, "
              f"min_evidence={_loaded_target_set.min_evidence}, "
              f"query_time={_loaded_target_set.query_timestamp})")
        if _loaded_target_set.adjacency:
            print(f"  Adjacency: {len(_loaded_target_set.adjacency)} regulators")
        else:
            print("  WARNING: No adjacency in target set file — "
                  "Phase 5b will query INDRA live")

        # Apply --min-sources post-load filtering
        _min_src = getattr(args, "min_sources", None)
        if _min_src is not None and _loaded_target_set.edge_metadata:
            n_before = len(network_targets)
            network_targets = _loaded_target_set.filter_by_min_sources(_min_src)
            n_dropped = n_before - len(network_targets)
            if n_dropped:
                print(f"  min_sources={_min_src}: {n_dropped} targets dropped, "
                      f"{len(network_targets)} remain")
            if len(network_targets) < 5:
                print(f"  WARNING: Only {len(network_targets)} targets survive "
                      f"min_sources={_min_src}")
    else:
        import warnings
        warnings.warn(
            "No --target-set file provided. Target gene set will be "
            "re-derived from a live INDRA query, which may diverge from "
            "the experimental analysis. For reproducible validation, "
            "pass --target-set <analysis_output>/indra_targets.json.",
            UserWarning,
            stacklevel=1,
        )
        from cliquefinder.cli.differential import query_network_targets_multihop
        n_hops = getattr(args, "network_hops", 1)
        min_intermediaries = getattr(args, "min_intermediaries", 1)
        print(f"\nQuerying INDRA network for {args.network_query} "
              f"({n_hops}-hop, min_evidence={args.min_evidence}"
              f"{f', min_intermediaries={min_intermediaries}' if n_hops > 1 else ''})...")
        network_targets = query_network_targets_multihop(
            gene_symbol=args.network_query,
            feature_ids=feature_ids,
            n_hops=n_hops,
            min_evidence=args.min_evidence,
            min_sources=getattr(args, "min_sources", None),
            min_intermediaries=min_intermediaries,
            env_file=args.indra_env_file,
            verbose=True,
            output_dir=args.output,
        )

    target_gene_ids = list(network_targets.values())
    print(f"  {len(target_gene_ids)} targets found in data")

    # CLI-8: Guard against empty or tiny target gene sets
    if len(target_gene_ids) < 2:
        print("Error: Gene set needs at least 2 genes for meaningful enrichment "
              f"analysis, but got {len(target_gene_ids)}. Check network query "
              f"'{args.network_query}' or --min-evidence threshold.")
        return 1
    elif len(target_gene_ids) < 5:
        import warnings
        warnings.warn(
            f"Gene set has only {len(target_gene_ids)} genes. Results may be "
            f"unreliable with fewer than 5 genes. Consider a broader network query.",
            UserWarning,
            stacklevel=1,
        )

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

    # Report covariate-induced sample exclusions (transparency)
    n_total_samples = len(covariate_design.sample_mask)
    n_valid = int(covariate_design.sample_mask.sum())
    n_dropped = n_total_samples - n_valid
    if n_dropped > 0:
        print(f"\n  Covariate NaN exclusion: {n_dropped}/{n_total_samples} samples dropped "
              f"({n_valid} remain)")
        # Report per-group impact
        cond_series = metadata[condition_col]
        for grp in sorted(cond_series.dropna().unique()):
            grp_mask = (cond_series == grp).values
            grp_total = int(grp_mask.sum())
            grp_valid = int((grp_mask & covariate_design.sample_mask).sum())
            grp_lost = grp_total - grp_valid
            if grp_lost > 0:
                print(f"    {grp}: {grp_lost}/{grp_total} dropped → {grp_valid} remain")

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
    from cliquefinder.stats.differential import (
        run_protein_differential,
        run_network_enrichment_test,
    )

    protein_df = None  # Initialize; downstream phases (e.g., Phase 5) check this

    if "covariate_adjusted" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 1: COVARIATE-ADJUSTED ENRICHMENT  [SKIPPED — checkpoint]")
        print("=" * 70)
        # VAL-3: Restore protein_df from checkpoint so Phase 5 can use it
        if checkpoint_protein_df is not None:
            protein_df = checkpoint_protein_df
            print(f"  Restored protein_df from checkpoint ({protein_df.shape[0]} rows)")
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 1: COVARIATE-ADJUSTED ENRICHMENT")
        print("=" * 70)

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
            enrichment_out = args.output / "covariate_enrichment.json"
            atomic_write_json(enrichment_out, enrichment.to_dict())

            # Save per-gene differential results CSV
            # Add gene symbols for target genes
            target_symbols = {v: k for k, v in network_targets.items()}
            protein_df_out = protein_df.copy()
            protein_df_out["gene_symbol"] = protein_df_out["feature_id"].map(
                target_symbols
            ).fillna("")
            protein_df_out.to_csv(
                args.output / "protein_differential_results.csv", index=False,
            )
        except Exception as e:
            logger.error("MANDATORY Phase 1 (covariate_adjusted) failed: %s", e)
            report.add_phase("covariate_adjusted", {"status": "failed", "error": str(e)})
            # protein_df remains None from initialization above; no reassignment needed
            # VALID-IV-1 (Audit IV): Mandatory gate failure — abort pipeline.
            # Phase 1 is a required gate; continuing without it produces misleading verdicts.
            if isinstance(e, (ValueError,)) or 'LinAlgError' in type(e).__name__:
                _save_checkpoint(report, args.output)
                report.save(args.output / "validation_report.json")
                print(f"\nABORTED: Mandatory Phase 1 failed with {type(e).__name__}: {e}")
                print("Fix the data issue and re-run. Remaining phases skipped.")
                return 0
        _save_checkpoint(report, args.output, protein_df=protein_df)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 2: Multi-contrast specificity
    # =====================================================================
    if len(contrasts) > 1:
        if "specificity" in report.phases:
            print(f"\n{'=' * 70}")
            print("PHASE 2: MULTI-CONTRAST SPECIFICITY  [SKIPPED — checkpoint]")
            print("=" * 70)
        else:
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
                    # VAL-6: Use SeedSequence-derived seed for Phase 2
                    specificity = compute_specificity(
                        enrichment_by_contrast,
                        primary_contrast=primary_contrast_name,
                        z_threshold=getattr(args, "specificity_z_threshold", 1.5),
                        data=data,
                        feature_ids=feature_ids,
                        metadata=metadata,
                        condition_col=condition_col,
                        contrast_tuples=contrasts,
                        target_gene_ids=target_gene_ids,
                        covariates_df=covariates_df,
                        n_interaction_perms=getattr(args, "interaction_n_perms", 200),
                        seed=_seed_phase2,
                    )
                    report.add_phase("specificity", specificity.to_dict())
                    print(f"\n  Specificity: {specificity.specificity_label}")
                    print(f"  {specificity.summary}")

                    spec_out = args.output / "specificity.json"
                    atomic_write_json(spec_out, specificity.to_dict())
            except Exception as e:
                logger.warning("Phase 2 (specificity) failed: %s", e)
                report.add_phase("specificity", {"status": "failed", "error": str(e)})
            _save_checkpoint(report, args.output)
        report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 3: Label permutation null (stratified + free)
    # =====================================================================
    if "label_permutation" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 3: LABEL PERMUTATION NULL  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
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

            perm_out = args.output / "label_permutation.json"
            atomic_write_json(perm_out, {"stratified": strat_dict, "free": free_dict})

            # Save full null distributions as CSV for plotting
            perm_rows = [{"permutation_id": "observed", "mode": "observed",
                          "competitive_z": strat_result.observed_z}]
            for i, z in enumerate(strat_result.null_z_scores):
                perm_rows.append({"permutation_id": f"stratified_{i:03d}",
                                  "mode": "stratified", "competitive_z": z})
            for i, z in enumerate(free_result.null_z_scores):
                perm_rows.append({"permutation_id": f"free_{i:03d}",
                                  "mode": "free", "competitive_z": z})
            pd.DataFrame(perm_rows).to_csv(
                args.output / "label_permutation_distributions.csv", index=False,
            )
        except Exception as e:
            logger.error("MANDATORY Phase 3 (label_permutation) failed: %s", e)
            report.add_phase("label_permutation", {"status": "failed", "error": str(e)})
            # VALID-IV-1 (Audit IV): Mandatory gate failure — abort pipeline.
            # Phase 3 is a required gate; continuing without it produces misleading verdicts.
            if isinstance(e, (ValueError,)) or 'LinAlgError' in type(e).__name__:
                _save_checkpoint(report, args.output)
                report.save(args.output / "validation_report.json")
                print(f"\nABORTED: Mandatory Phase 3 failed with {type(e).__name__}: {e}")
                print("Fix the data issue and re-run. Remaining phases skipped.")
                return 0
        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 4: Sex-matched subsampling reanalysis
    # =====================================================================
    if "matched_reanalysis" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 4: MATCHED SUBSAMPLING REANALYSIS  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
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

            matched_out = args.output / "matched_enrichment.json"
            atomic_write_json(matched_out, matched_enrichment.to_dict())
        except Exception as e:
            logger.warning("Phase 4 (matched_reanalysis) failed: %s", e)
            report.add_phase("matched_reanalysis", {"status": "failed", "error": str(e)})
        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 5a: Negative control gene sets (uniform random)
    # =====================================================================
    # XVI-3: Phase 5a and 5b have independent checkpoint keys so each can
    # be retried independently (e.g., if INDRA times out for 5b but 5a
    # succeeded, a resume will only re-run 5b).

    # Shared engine for both 5a and 5b — built once if either sub-phase runs.
    engine = None

    def _ensure_engine():
        nonlocal engine
        if engine is not None:
            return engine
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
        return engine

    if "negative_controls" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 5a: NEGATIVE CONTROLS  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 5a: NEGATIVE CONTROLS")
        print("=" * 70)

        try:
            from cliquefinder.stats.negative_controls import run_negative_control_sets

            eng = _ensure_engine()

            # protein_df may be None if Phase 1 failed. Pass it through;
            # run_negative_control_sets() handles None gracefully (skips
            # competitive z-score computation).
            # Compute evidence weights aligned to target_gene_ids (which may
            # be min_sources-filtered).  Build weights from the symbol→weight
            # map keyed by feature_id to guarantee alignment.
            _target_weights = None
            if _loaded_target_set is not None and _loaded_target_set.edge_metadata:
                _ew = _loaded_target_set.evidence_weights()  # {symbol: weight}
                # Build feature_id → weight lookup
                _fid_to_w = {}
                for sym, fid in _loaded_target_set.targets.items():
                    _fid_to_w[fid] = _ew.get(sym, 0.2)
                # Align to target_gene_ids order
                _target_weights = [_fid_to_w.get(fid, 0.2) for fid in target_gene_ids]

            neg_result = run_negative_control_sets(
                engine=eng,
                target_gene_ids=target_gene_ids,
                target_set_id=f"{args.network_query}_targets",
                n_control_sets=args.n_neg_controls,
                seed=_seed_phase5,
                protein_results=protein_df,
                verbose=True,
                target_weights=_target_weights,
            )

            report.add_phase("negative_controls", neg_result.to_dict())

            neg_out = args.output / "negative_controls.json"
            atomic_write_json(neg_out, neg_result.to_dict())

            # Save per-control-set distributions as CSV
            neg_rows = [{"set_id": f"{args.network_query}_targets",
                         "type": "target",
                         "roast_pvalue": neg_result.target_pvalue,
                         "competitive_z": (neg_result.target_competitive_z
                                           if neg_result.target_competitive_z is not None
                                           else np.nan)}]
            for i, p in enumerate(neg_result.control_pvalues):
                row = {"set_id": f"random_{i:03d}", "type": "random_gene_set",
                       "roast_pvalue": p, "competitive_z": np.nan}
                if (neg_result.control_competitive_z_scores is not None
                        and i < len(neg_result.control_competitive_z_scores)):
                    row["competitive_z"] = neg_result.control_competitive_z_scores[i]
                neg_rows.append(row)
            pd.DataFrame(neg_rows).to_csv(
                args.output / "negative_control_distributions.csv", index=False,
            )
        except Exception as e:
            logger.warning("Phase 5a (negative_controls) failed: %s", e)
            report.add_phase("negative_controls", {"status": "failed", "error": str(e)})

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 5b: Graph permutation null (node-label permutation on INDRA)
    # =====================================================================
    # XVI-4 note: This queries INDRA for ALL regulators in the data universe,
    # not just the query gene's targets. The broader scope is intentional —
    # the null distribution requires multiple regulator neighborhoods to
    # sample from. The Phase 1 single-gene query (query_network_targets)
    # returns only one regulator, which would produce a degenerate null.

    if "graph_permutation" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 5b: GRAPH PERMUTATION  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 5b: GRAPH PERMUTATION")
        print("=" * 70)

        try:
            from cliquefinder.stats.graph_permutation import run_graph_permutation_null
            from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

            eng = _ensure_engine()

            # Build symbol_to_feature mapping (gene_symbol -> feature_id)
            symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=False)

            # Use adjacency from pinned target set if available
            if _loaded_target_set is not None and _loaded_target_set.adjacency:
                adjacency = _loaded_target_set.adjacency
                print(f"  Using pinned adjacency: "
                      f"{len(adjacency)} regulators from target set file")
            else:
                # Pull INDRA subgraph: all regulatory edges for genes in our data
                if _loaded_target_set is not None:
                    print("  WARNING: Target set file has no adjacency — "
                          "querying INDRA live for Phase 5b")
                from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource
                indra_source = INDRAKnowledgeSource(env_file=str(args.indra_env_file))
                gene_universe = set(symbol_to_feature.keys())
                indra_modules = indra_source.discover_regulators(
                    target_universe=gene_universe,
                    min_targets=2,
                    min_evidence=args.min_evidence,
                )
                indra_source.close()

                # Build adjacency dict: regulator_name -> [target_name, ...]
                adjacency = {}
                for module in indra_modules:
                    adjacency[module.regulator] = sorted(module.targets)

                # Save adjacency to target set file for future runs
                target_set_path = args.output / "indra_targets.json"
                if target_set_path.exists():
                    from cliquefinder.stats.target_set import TargetSet as _TS
                    ts = _TS.load(target_set_path)
                    ts.attach_adjacency(
                        adjacency=adjacency,
                        min_evidence=args.min_evidence,
                        min_targets=2,
                    )
                    ts.save(target_set_path)
                    print(f"  Adjacency ({len(adjacency)} regulators) "
                          f"saved to {target_set_path}")

            if adjacency:
                graph_result = run_graph_permutation_null(
                    engine=eng,
                    target_gene_ids=target_gene_ids,
                    target_set_id=f"{args.network_query}_targets",
                    adjacency=adjacency,
                    symbol_to_feature=symbol_to_feature,
                    n_permutations=args.n_graph_perms,
                    seed=_seed_phase5_graph,
                    verbose=True,
                    size_match_tolerance=getattr(
                        args, "graph_size_match_tolerance", 0.5),
                    target_weights=_target_weights,
                )
                report.add_phase("graph_permutation", graph_result.to_dict())
                graph_out = args.output / "graph_permutation.json"
                atomic_write_json(graph_out, graph_result.to_dict())

                # Save per-permutation distributions as CSV
                graph_rows = [{"set_id": f"{args.network_query}_targets",
                               "type": "target",
                               "roast_pvalue": graph_result.target_pvalue}]
                for i, p in enumerate(graph_result.control_pvalues):
                    graph_rows.append({"set_id": f"graph_perm_{i:03d}",
                                       "type": "graph_permutation",
                                       "roast_pvalue": p})
                pd.DataFrame(graph_rows).to_csv(
                    args.output / "graph_permutation_distributions.csv",
                    index=False,
                )
            else:
                logger.warning("No INDRA modules found for graph permutation -- skipping")
        except ImportError:
            logger.warning("INDRA not available for graph permutation -- skipping")
        except Exception as e:
            logger.warning("Graph permutation failed: %s", e)
            report.add_phase("graph_permutation", {"status": "failed", "error": str(e)})

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 5c: SIGNED CONCORDANCE TEST
    # =====================================================================
    # Tests whether the DIRECTION of each target's differential expression
    # matches the prediction from the INDRA edge type (activation → down
    # for loss-of-function, repression → up).

    if "signed_concordance" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 5c: SIGNED CONCORDANCE  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 5c: SIGNED CONCORDANCE")
        print("=" * 70)

        try:
            from cliquefinder.stats.signed_concordance import compute_signed_concordance

            # Need protein_df and a TargetSet with edge_metadata.
            # Use the filtered target set if --min-sources was applied,
            # so Phase 5c tests the same gene set as other phases.
            _ts_for_concordance = _loaded_target_set
            _min_src = getattr(args, "min_sources", None)
            if (_ts_for_concordance is not None
                    and _min_src is not None
                    and _ts_for_concordance.edge_metadata):
                from cliquefinder.stats.target_set import TargetSet as _TSf
                filtered_targets = _ts_for_concordance.filter_by_min_sources(_min_src)
                _ts_for_concordance = _TSf.from_query(
                    targets_in_data=filtered_targets,
                    gene_symbol=_ts_for_concordance.gene_symbol,
                    min_evidence=_ts_for_concordance.min_evidence,
                    n_hops=_ts_for_concordance.n_hops,
                    edge_metadata={
                        s: _ts_for_concordance.edge_metadata[s]
                        for s in filtered_targets
                        if s in _ts_for_concordance.edge_metadata
                    },
                    min_sources=_min_src,
                )
            if _ts_for_concordance is None:
                # Try loading from the output dir (created during this run)
                _ts_path = args.output / "indra_targets.json"
                if _ts_path.exists():
                    from cliquefinder.stats.target_set import TargetSet as _TSc
                    _ts_for_concordance = _TSc.load(_ts_path)

            if protein_df is None:
                csv_path = args.output / "protein_differential_results.csv"
                if csv_path.exists():
                    protein_df = pd.read_csv(csv_path)

            if (protein_df is not None
                    and _ts_for_concordance is not None
                    and _ts_for_concordance.edge_metadata):
                concordance_result = compute_signed_concordance(
                    protein_df=protein_df,
                    target_set=_ts_for_concordance,
                    n_permutations=1000,
                    seed=_seed_phase5c,
                )
                report.add_phase("signed_concordance", concordance_result.to_dict())
                conc_out = args.output / "signed_concordance.json"
                atomic_write_json(conc_out, concordance_result.to_dict())

                cr = concordance_result
                print(f"  Unambiguous targets: {cr.n_unambiguous} "
                      f"({cr.n_predicted_down} act→down, "
                      f"{cr.n_predicted_up} rep→up)")
                print(f"  Mixed excluded: {cr.n_mixed_excluded}")
                print(f"  Concordance: {cr.n_concordant}/{cr.n_unambiguous} "
                      f"({cr.concordance_rate:.1%})")
                print(f"  Background rate: {cr.background_concordance_rate:.1%}")
                if cr.activation_subgroup:
                    a = cr.activation_subgroup
                    print(f"    Activation: {a['n_concordant']}/{a['n']} "
                          f"({a['concordance_rate']:.1%}, bg={a['background_rate']:.1%}, "
                          f"p={a['binomial_pvalue']:.4f})")
                if cr.repression_subgroup:
                    r = cr.repression_subgroup
                    print(f"    Repression: {r['n_concordant']}/{r['n']} "
                          f"({r['concordance_rate']:.1%}, bg={r['background_rate']:.1%}, "
                          f"p={r['binomial_pvalue']:.4f})")
                print(f"  Permutation p-value: {cr.permutation_pvalue:.4f} "
                      f"({cr.n_permutations} permutations)")
                print(f"  Binomial p-value: {cr.binomial_pvalue:.4f}")
                print(f"  GoF sensitivity: {cr.gof_concordance_rate:.1%} "
                      f"(p={cr.gof_binomial_pvalue:.4f})")
                print(f"  Best model: {cr.best_model}")
            else:
                reason = []
                if protein_df is None:
                    reason.append("no protein_df")
                if _ts_for_concordance is None:
                    reason.append("no target set")
                elif not _ts_for_concordance.edge_metadata:
                    reason.append("no edge metadata in target set")
                skip_reason = ", ".join(reason) or "unknown"
                logger.warning("Phase 5c skipped — %s", skip_reason)
                report.add_phase("signed_concordance", {
                    "status": "skipped", "reason": skip_reason,
                })
        except Exception as e:
            logger.warning("Phase 5c (signed concordance) failed: %s", e)
            report.add_phase("signed_concordance", {"status": "failed", "error": str(e)})

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 5d: SUBGROUP ROAST (directional, activation vs repression)
    # =====================================================================
    # Decomposes the target set by INDRA edge type and tests each subgroup
    # with a directional statistic (MEAN-down for activation, MEAN-up for
    # repression).  More powerful than MSQ-mixed when the LoF hypothesis
    # predicts a specific direction.

    if "subgroup_enrichment" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 5d: SUBGROUP ROAST  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 5d: SUBGROUP ROAST (directional)")
        print("=" * 70)

        try:
            _ts_for_subgroup = _loaded_target_set
            _min_src = getattr(args, "min_sources", None)
            if (_ts_for_subgroup is not None
                    and _min_src is not None
                    and _ts_for_subgroup.edge_metadata):
                from cliquefinder.stats.target_set import TargetSet as _TSs
                filtered = _ts_for_subgroup.filter_by_min_sources(_min_src)
                _ts_for_subgroup = _TSs.from_query(
                    targets_in_data=filtered,
                    gene_symbol=_ts_for_subgroup.gene_symbol,
                    min_evidence=_ts_for_subgroup.min_evidence,
                    n_hops=_ts_for_subgroup.n_hops,
                    edge_metadata={
                        s: _ts_for_subgroup.edge_metadata[s]
                        for s in filtered
                        if s in _ts_for_subgroup.edge_metadata
                    },
                    min_sources=_min_src,
                )

            if (_ts_for_subgroup is not None
                    and _ts_for_subgroup.edge_metadata):
                eng = _ensure_engine()
                predictions = _ts_for_subgroup.get_unambiguous_targets()
                ev_weights = _ts_for_subgroup.evidence_weights()

                # Split into activation and repression subgroups
                act_fids, act_weights = [], []
                rep_fids, rep_weights = [], []
                for sym, direction in predictions.items():
                    fid = _ts_for_subgroup.targets.get(sym)
                    if fid is None or fid not in eng.gene_to_idx:
                        continue
                    w = ev_weights.get(sym, 0.2)
                    if direction == "predicted_down":
                        act_fids.append(fid)
                        act_weights.append(w)
                    elif direction == "predicted_up":
                        rep_fids.append(fid)
                        rep_weights.append(w)

                print(f"  Activation targets (predicted DOWN): {len(act_fids)}")
                print(f"  Repression targets (predicted UP):   {len(rep_fids)}")

                from cliquefinder.stats.rotation import (
                    RotationTestConfig, SetStatistic, Alternative,
                )

                subgroup_results = {}

                # Activation subgroup: MEAN-down (directional)
                if len(act_fids) >= 2:
                    act_config = RotationTestConfig(
                        statistics=[SetStatistic.MEAN, SetStatistic.MSQ],
                        n_rotations=getattr(args, "n_rotations", 9999),
                        seed=getattr(args, "seed", None),
                    )
                    act_result = eng.test_gene_set(
                        gene_set=act_fids,
                        gene_set_id="activation_targets",
                        weights=np.array(act_weights),
                        config=act_config,
                    )
                    p_mean_down = act_result.p_values.get("mean", {}).get("down", 1.0)
                    p_msq_mixed = act_result.p_values.get("msq", {}).get("mixed", 1.0)
                    subgroup_results["activation"] = {
                        "n_genes": len(act_fids),
                        "p_mean_down": p_mean_down,
                        "p_msq_mixed": p_msq_mixed,
                        "n_genes_found": act_result.n_genes_found,
                    }
                    print(f"  Activation MEAN-down: p={p_mean_down:.4f} "
                          f"(MSQ-mixed: p={p_msq_mixed:.4f})")

                # Repression subgroup: MEAN-up (directional)
                if len(rep_fids) >= 2:
                    rep_config = RotationTestConfig(
                        statistics=[SetStatistic.MEAN, SetStatistic.MSQ],
                        n_rotations=getattr(args, "n_rotations", 9999),
                        seed=getattr(args, "seed", None),
                    )
                    rep_result = eng.test_gene_set(
                        gene_set=rep_fids,
                        gene_set_id="repression_targets",
                        weights=np.array(rep_weights),
                        config=rep_config,
                    )
                    p_mean_up = rep_result.p_values.get("mean", {}).get("up", 1.0)
                    p_msq_mixed = rep_result.p_values.get("msq", {}).get("mixed", 1.0)
                    subgroup_results["repression"] = {
                        "n_genes": len(rep_fids),
                        "p_mean_up": p_mean_up,
                        "p_msq_mixed": p_msq_mixed,
                        "n_genes_found": rep_result.n_genes_found,
                    }
                    print(f"  Repression MEAN-up:   p={p_mean_up:.4f} "
                          f"(MSQ-mixed: p={p_msq_mixed:.4f})")

                # Bonferroni correction across 2 subgroups
                n_tests = sum(1 for k in ["activation", "repression"]
                              if k in subgroup_results)
                for key in subgroup_results:
                    sr = subgroup_results[key]
                    primary_p = sr.get("p_mean_down", sr.get("p_mean_up", 1.0))
                    sr["bonferroni_p"] = min(primary_p * n_tests, 1.0)

                if subgroup_results:
                    report.add_phase("subgroup_enrichment", subgroup_results)
                    atomic_write_json(
                        args.output / "subgroup_enrichment.json",
                        subgroup_results,
                    )
                    # Summary
                    for key, sr in subgroup_results.items():
                        print(f"  {key} Bonferroni-corrected: p={sr['bonferroni_p']:.4f}")
            else:
                logger.warning("Phase 5d skipped — no target set with edge metadata")
                report.add_phase("subgroup_enrichment", {
                    "status": "skipped", "reason": "no edge metadata",
                })

        except Exception as e:
            logger.warning("Phase 5d (subgroup ROAST) failed: %s", e)
            report.add_phase("subgroup_enrichment", {
                "status": "failed", "error": str(e),
            })

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 6: NETWORK PROXIMITY TESTS (parameter-free)
    # =====================================================================
    # Three continuous-score tests that avoid the set-size / null-width
    # problem. Each produces a single p-value; Bonferroni across 3 tests.
    #   6a. Proximity decay: shortest-path distance predicts |t-stat|
    #   6b. Reverse causal reasoning: DE genes point back to query gene
    #   6c. RWR correlation: diffusion proximity predicts |t-stat|

    if "network_proximity" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 6: NETWORK PROXIMITY TESTS  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 6: NETWORK PROXIMITY TESTS")
        print("=" * 70)

        try:
            from cliquefinder.stats.network_proximity import (
                compute_rwr_scores,
                extract_local_subgraph_edges,
                query_gene_degrees_batched,
                query_shortest_paths_batched,
                run_proximity_decay_test,
                run_reverse_causal_reasoning,
                run_rwr_correlation_test,
            )
            from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

            # -- Build feature_to_symbol mapping (canonical HGNC symbols) --
            import mygene
            mg = mygene.MyGeneInfo()
            mg_results = mg.querymany(
                feature_ids, scopes="uniprot", fields="symbol",
                species="human", verbose=False,
            )
            feature_to_symbol_p6: dict[str, str] = {}
            for r in mg_results:
                if isinstance(r, dict) and "symbol" in r:
                    feature_to_symbol_p6[r["query"]] = r["symbol"]
            print(f"  Resolved {len(feature_to_symbol_p6)} UniProt → HGNC symbols")

            # -- Compute |t-statistics| from Phase 1 protein_df --
            if protein_df is None:
                # Restore from saved CSV when checkpoint lacks protein_df_dict
                csv_path = args.output / "protein_differential_results.csv"
                if csv_path.exists():
                    protein_df = pd.read_csv(csv_path)
                    print(f"  Restored protein_df from CSV ({protein_df.shape[0]} rows)")
                else:
                    raise ValueError("Phase 1 protein_df not available for proximity tests")

            abs_t_stats: dict[str, float] = {}
            for _, row in protein_df.iterrows():
                sym = feature_to_symbol_p6.get(row["feature_id"])
                if sym and np.isfinite(row["t_statistic"]):
                    abs_t_stats[sym] = abs(float(row["t_statistic"]))

            print(f"  {len(abs_t_stats)} genes with valid |t-statistics|")

            # -- Extract INDRA subgraph around query gene --
            print(f"\n  Extracting INDRA subgraph around {args.network_query}...")
            from cliquefinder.knowledge.cogex import CoGExClient
            cogex = CoGExClient(env_file=args.indra_env_file)

            # Resolve query gene to CURIE
            from cliquefinder.knowledge.cogex import INDRAModuleExtractor
            extractor = INDRAModuleExtractor(client=cogex)
            seed_gene_id = extractor.resolve_gene_name(args.network_query)
            if seed_gene_id is None:
                raise ValueError(f"Cannot resolve '{args.network_query}' to HGNC ID")
            # GeneId is Tuple[str, str] e.g. ("HGNC", "11998")
            seed_curie = f"hgnc:{seed_gene_id[1]}"

            # -- Server-side shortest paths (78s vs 30-70 min for APOC) --
            measured_symbols = sorted(set(abs_t_stats.keys()))
            print(f"\n  Querying server-side shortest paths to {len(measured_symbols)} genes...")
            distances = query_shortest_paths_batched(
                cogex_client=cogex,
                seed_gene_name=args.network_query,
                target_gene_names=measured_symbols,
                max_hops=8,
                batch_size=500,
                verbose=True,
            )
            print(f"  {len(distances)} genes reachable")

            # -- Server-side degrees for permutation null --
            reachable = sorted(set(distances.keys()) & set(abs_t_stats.keys()))
            print(f"  Querying degrees for {len(reachable)} reachable genes...")
            graph_degrees = query_gene_degrees_batched(
                cogex_client=cogex,
                gene_names=reachable,
                batch_size=500,
            )

            # -- Phase 6a: Proximity decay --
            print(f"\n  Phase 6a: Proximity decay test...")

            # Derive sub-seeds for the three tests
            if _seed_phase6 is not None:
                _ss6 = np.random.SeedSequence(_seed_phase6)
                _ss6a, _ss6b, _ss6c = _ss6.spawn(3)
                _seed_6a = int(_ss6a.generate_state(1)[0])
                _seed_6c = int(_ss6c.generate_state(1)[0])
            else:
                _seed_6a = None
                _seed_6c = None

            decay_result = run_proximity_decay_test(
                distances=distances,
                abs_t_stats=abs_t_stats,
                graph_degrees=graph_degrees,
                seed_gene=args.network_query,
                n_permutations=1000,
                seed=_seed_6a,
                verbose=True,
            )
            print(f"    rho={decay_result.spearman_rho:.4f}, "
                  f"perm_p={decay_result.permutation_pvalue:.4f}")

            # -- Phase 6b: Reverse causal reasoning --
            print(f"\n  Phase 6b: Reverse causal reasoning...")
            rcr_result = run_reverse_causal_reasoning(
                protein_results=protein_df,
                query_gene=args.network_query,
                feature_to_symbol=feature_to_symbol_p6,
                env_file=args.indra_env_file,
                fdr_threshold=getattr(args, "rcr_fdr_threshold", 0.05),
                min_evidence=2,
                min_targets=3,
                verbose=True,
            )
            print(f"    {args.network_query} rank={rcr_result.query_gene_rank}/"
                  f"{rcr_result.n_regulators_tested}, "
                  f"z={rcr_result.query_gene_zscore:.3f}, "
                  f"p={rcr_result.query_gene_pvalue:.4f}")

            # -- Phase 6c: RWR correlation --
            print(f"\n  Phase 6c: RWR correlation test...")
            import scipy.sparse as sp_sparse

            # Extract small 2-hop subgraph for RWR (minutes, not hours)
            print(f"  Extracting 2-hop subgraph for RWR...")
            rwr_edges = extract_local_subgraph_edges(
                cogex_client=cogex,
                seed_gene_name=args.network_query,
                max_hops=2,
                min_evidence=1,
            )
            print(f"  {len(rwr_edges)} edges in 2-hop subgraph")

            # Build sparse adjacency from edge list
            _rwr_nodes_set: set[str] = set()
            for src, tgt, _ in rwr_edges:
                _rwr_nodes_set.add(src)
                _rwr_nodes_set.add(tgt)
            rwr_node_list = sorted(_rwr_nodes_set)
            rwr_node_to_idx = {n: i for i, n in enumerate(rwr_node_list)}
            _r, _c, _d = [], [], []
            for src, tgt, attrs in rwr_edges:
                _r.append(rwr_node_to_idx[src])
                _c.append(rwr_node_to_idx[tgt])
                _d.append(attrs.get("evidence_count", 1))
            rwr_adj = sp_sparse.csr_matrix(
                (np.array(_d, dtype=np.float64),
                 (np.array(_r), np.array(_c))),
                shape=(len(rwr_node_list), len(rwr_node_list)),
            )
            print(f"  RWR graph: {len(rwr_node_list)} nodes, {rwr_adj.nnz} edges")

            if args.network_query not in rwr_node_to_idx:
                raise ValueError(f"Seed '{args.network_query}' not in 2-hop subgraph")
            rwr_seed_idx = rwr_node_to_idx[args.network_query]

            rwr_scores_arr, conv_delta, n_iter = compute_rwr_scores(
                adjacency=rwr_adj,
                seed_index=rwr_seed_idx,
                restart_prob=0.15,
            )
            rwr_scores = {
                rwr_node_list[i]: float(rwr_scores_arr[i])
                for i in range(len(rwr_node_list))
            }

            rwr_result = run_rwr_correlation_test(
                rwr_scores=rwr_scores,
                abs_t_stats=abs_t_stats,
                seed_gene=args.network_query,
                n_graph_nodes=len(rwr_node_list),
                n_graph_edges=rwr_adj.nnz,
                restart_probability=0.15,
                convergence_delta=conv_delta,
                n_rwr_iterations=n_iter,
                n_permutations=1000,
                seed=_seed_6c,
                verbose=True,
            )
            print(f"    rho={rwr_result.spearman_rho:.4f}, "
                  f"perm_p={rwr_result.permutation_pvalue:.4f}")

            # cogex.close() handled in finally block below

            # -- Combine and save --
            from cliquefinder.stats.network_proximity import NetworkProximityReport

            proximity_report = NetworkProximityReport(
                proximity_decay=decay_result,
                reverse_causal=rcr_result,
                rwr_correlation=rwr_result,
            )

            report.add_phase("network_proximity", proximity_report.to_dict())

            prox_out = args.output / "network_proximity.json"
            atomic_write_json(prox_out, proximity_report.to_dict())

            # Export proximity decay curve as CSV
            decay_rows = []
            for dist, stats in sorted(decay_result.distance_bins.items()):
                decay_rows.append({
                    "distance": dist,
                    "n_genes": int(stats["n_genes"]),
                    "mean_abs_t": stats["mean_abs_t"],
                    "median_abs_t": stats["median_abs_t"],
                    "std_abs_t": stats["std_abs_t"],
                })
            pd.DataFrame(decay_rows).to_csv(
                args.output / "proximity_decay_curve.csv", index=False,
            )

            # Export top regulators from reverse causal as CSV
            pd.DataFrame(rcr_result.top_regulators).to_csv(
                args.output / "reverse_causal_top_regulators.csv", index=False,
            )

            bonf = 0.05 / 3
            print(f"\n  Bonferroni alpha = {bonf:.4f}")
            print(f"  6a proximity decay:    p={decay_result.permutation_pvalue:.4f} "
                  f"{'PASS' if decay_result.permutation_pvalue < bonf else 'FAIL'}")
            print(f"  6b reverse causal:     p={rcr_result.query_gene_pvalue:.4f} "
                  f"{'PASS' if rcr_result.query_gene_pvalue < bonf else 'FAIL'}")
            print(f"  6c RWR correlation:    p={rwr_result.permutation_pvalue:.4f} "
                  f"{'PASS' if rwr_result.permutation_pvalue < bonf else 'FAIL'}")
            print(f"  Any significant:       {proximity_report.any_significant}")

        except ImportError as e:
            logger.warning("Phase 6 (network proximity) skipped — missing dependency: %s", e)
        except Exception as e:
            logger.warning("Phase 6 (network proximity) failed: %s", e, exc_info=True)
            report.add_phase("network_proximity", {"status": "failed", "error": str(e)})
        finally:
            # Ensure CoGEx connection is closed even on exception
            try:
                cogex.close()  # type: ignore[possibly-undefined]
            except (NameError, Exception):
                pass

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # PHASE 7: HIERARCHICAL DISCOVERY (n-hop pathway decomposition)
    # =====================================================================
    # Uses the causal-path-scoring framework to decompose the regulatory
    # network by intermediary (hop 2) and find convergence points (hop 3).
    # Informational — not a validation gate.

    if "discovery" in report.phases:
        print(f"\n{'=' * 70}")
        print("PHASE 7: HIERARCHICAL DISCOVERY  [SKIPPED — checkpoint]")
        print("=" * 70)
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 7: HIERARCHICAL DISCOVERY")
        print("=" * 70)

        try:
            from causal_path_scoring.core.reliability import Edge as CPSEdge
            from causal_path_scoring.core.discovery import run_discovery
            from causal_path_scoring.core.belief import compute_belief_with_contradiction
            from cliquefinder.stats.discovery_bridge import DiscoveryBridge

            # Build adjacency from TargetSet edge metadata
            _ts_disc = _loaded_target_set
            if _ts_disc is None:
                _ts_path = args.output / "indra_targets.json"
                if _ts_path.exists():
                    from cliquefinder.stats.target_set import TargetSet as _TSD
                    _ts_disc = _TSD.load(_ts_path)

            if (_ts_disc is not None and _ts_disc.edge_metadata
                    and hasattr(args, 'indra_env_file')):
                eng = _ensure_engine()
                from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols as _map_syms
                symbol_to_feature = _map_syms(feature_ids, verbose=False)

                # Build adjacency for graph structure with computed beliefs
                disc_adjacency = {args.network_query: []}
                for sym, edges in _ts_disc.edge_metadata.items():
                    belief, direction, contradictory = compute_belief_with_contradiction(edges)
                    disc_adjacency[args.network_query].append(CPSEdge(
                        source=args.network_query, target=sym,
                        belief=belief,
                        edge_type=direction,
                    ))

                # Effect maps from protein_df (restore from CSV if needed)
                if protein_df is None:
                    csv_path = args.output / "protein_differential_results.csv"
                    if csv_path.exists():
                        protein_df = pd.read_csv(csv_path)
                disc_effects = {}
                disc_directions = {}
                if protein_df is not None:
                    for _, row in protein_df.iterrows():
                        if pd.notna(row.get('t_statistic')):
                            fid = row['feature_id']
                            disc_effects[fid] = abs(float(row['t_statistic']))
                            disc_directions[fid] = 'down' if float(row['t_statistic']) < 0 else 'up'

                # Build seed null pool from all genes measurable in the
                # expression data that are NOT the seed's direct neighbors.
                # These genes can be queried via INDRA for their own targets.
                _seed_neighbors = set(_ts_disc.edge_metadata.keys())
                _seed_null_pool = sorted(
                    sym for sym, fid in symbol_to_feature.items()
                    if sym != args.network_query
                    and sym not in _seed_neighbors
                    and fid in eng.gene_to_idx
                )
                print(f"  Seed null pool: {len(_seed_null_pool)} candidate genes "
                      f"(excluding {len(_seed_neighbors)} seed neighbors)")

                # Pass the same rotation count as the main pipeline
                from cliquefinder.stats.rotation import RotationTestConfig, SetStatistic
                _disc_roast_config = RotationTestConfig(
                    statistics=[SetStatistic.MSQ],
                    n_rotations=args.n_rotations,
                    seed=42,
                )
                with DiscoveryBridge(
                    eng, symbol_to_feature,
                    env_file=args.indra_env_file,
                    min_evidence=args.min_evidence,
                    roast_config=_disc_roast_config,
                ) as bridge:
                    disc_result = run_discovery(
                        seed=args.network_query,
                        adjacency=disc_adjacency,
                        test_gene_set=bridge.test_gene_set,
                        target_to_effect=disc_effects,
                        target_to_direction=disc_directions,
                        measurable_genes=set(),  # empty: get_targets callback handles filtering
                        max_hops=3,
                        min_targets_per_arm=5,
                        fdr_threshold=args.alpha,
                        effect_threshold=1.5,
                        get_targets=bridge.get_targets,
                        verbose=True,
                        # Phase 2: Three-layer inferential boundary
                        hierarchical_fdr=True,
                        seed_null_stop=True,
                        seed_null_b=100,
                        seed_null_threshold=0.1,
                        seed_null_pool=_seed_null_pool,
                        seed_null_rng=np.random.default_rng(42),
                        knockoff_filter=True,
                        knockoff_rng=np.random.default_rng(42),
                    )

                    # --- Soft posterior propagation ---
                    # Must be inside `with` block: bridge._target_cache
                    # is cleared on __exit__
                    from causal_path_scoring.core.posterior_propagation import (
                        compute_posterior_target_scores as _compute_pts,
                    )
                    # Extract intermediary posteriors from all hop results
                    _inter_posteriors = {}
                    for hop_res in disc_result.hops:
                        for arm in hop_res.all_arms:
                            if not np.isnan(arm.posterior):
                                _inter_posteriors[arm.intermediary] = arm.posterior

                    # Build full adjacency including intermediary→target edges
                    # from the bridge's cached target lookups + edge metadata
                    _full_adj = dict(disc_adjacency)
                    for intermediary, edge_metas in bridge._edge_metadata_cache.items():
                        if intermediary not in _full_adj:
                            _full_adj[intermediary] = []
                        for em in edge_metas:
                            # Compute per-edge belief from INDRA noise model
                            from causal_path_scoring.core.belief import compute_belief as _cb
                            _src_counts = em.get("source_counts", {})
                            if _src_counts:
                                _sources = []
                                _total_ev = 0
                                for src_name, cnt in _src_counts.items():
                                    _sources.extend([src_name] * cnt)
                                    _total_ev += cnt
                                _edge_belief = _cb(_sources, _total_ev)
                            else:
                                _edge_belief = _cb(em.get("sources", []), em.get("evidence_count", 1))
                            _full_adj[intermediary].append(CPSEdge(
                                source=intermediary, target=em["target_fid"],
                                belief=_edge_belief,
                                edge_type=em.get("regulation_type", "unknown"),
                            ))

                    _target_scores = {}
                    if _inter_posteriors and _full_adj:
                        _target_scores = _compute_pts(
                            source=args.network_query,
                            adjacency=_full_adj,
                            intermediary_posteriors=_inter_posteriors,
                            max_hops=2,
                        )

                # Build serializable output (outside with — bridge no longer needed)
                _pts_list = sorted(
                    [
                        {
                            "target": ts.target,
                            "posterior": round(ts.posterior, 4),
                            "belief_only": round(ts.belief_only, 4),
                            "n_paths": ts.n_paths,
                            "net_direction": ts.net_direction,
                        }
                        for ts in _target_scores.values()
                    ],
                    key=lambda x: x["posterior"],
                    reverse=True,
                )

                disc_dict = disc_result.to_dict()
                disc_dict["posterior_target_scores"] = _pts_list
                disc_dict["n_intermediary_posteriors"] = len(_inter_posteriors)

                report.add_phase("discovery", disc_dict)
                disc_out = args.output / "discovery_results.json"
                atomic_write_json(disc_out, disc_dict)
                print(disc_result.summary())

                if _pts_list:
                    print(f"\n  Posterior target scores: {len(_pts_list)} targets")
                    print(f"  Top 10 by posterior:")
                    for t in _pts_list[:10]:
                        delta = t["posterior"] - t["belief_only"]
                        print(f"    {t['target']:12s}  post={t['posterior']:.3f}  "
                              f"belief={t['belief_only']:.3f}  Δ={delta:+.3f}  "
                              f"dir={t['net_direction']}")
            else:
                reason = []
                if _ts_disc is None:
                    reason.append("no target set")
                elif not _ts_disc.edge_metadata:
                    reason.append("no edge metadata")
                logger.warning("Phase 7 skipped — %s", ", ".join(reason) or "unknown")
                report.add_phase("discovery", {
                    "status": "skipped", "reason": ", ".join(reason) or "unknown",
                })

        except ImportError as e:
            logger.warning("Phase 7 skipped — causal-path-scoring not available: %s", e)
            report.add_phase("discovery", {"status": "skipped", "reason": str(e)})
        except Exception as e:
            logger.warning("Phase 7 (discovery) failed: %s", e, exc_info=True)
            report.add_phase("discovery", {"status": "failed", "error": str(e)})

        _save_checkpoint(report, args.output)
    report.save(args.output / "validation_report.json")

    # =====================================================================
    # AGGREGATE REPORT
    # =====================================================================
    report.compute_verdict(
        alpha=args.alpha,
        neg_ctrl_percentile=getattr(args, "neg_ctrl_percentile", 10.0),
    )
    report.save(args.output / "validation_report.json")
    report.print_summary()

    print(f"\nAll results saved to: {args.output}")
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0
