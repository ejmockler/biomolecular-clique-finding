"""
CLI for clique-level differential abundance analysis.

Implements MSstats-inspired statistical testing at the clique level:
- Aggregates proteins within cliques using Tukey's Median Polish
- Tests for differential abundance using linear mixed models
- Applies FDR correction across cliques

Usage:
    cliquefinder differential \\
        --data output/proteomics/imputed.data.csv \\
        --metadata output/proteomics/imputed.metadata.csv \\
        --cliques output/proteomics/cliques/cliques.csv \\
        --output output/proteomics/differential \\
        --condition-col phenotype \\
        --subject-col subject_id \\
        --contrast CASE_vs_CTRL CASE CTRL
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cliquefinder.cli._validators import _positive_int
from cliquefinder.cohort import resolve_cohort_from_args


def query_network_targets(
    gene_symbol: str,
    feature_ids: list[str],
    min_evidence: int = 1,
    env_file: Path = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Query INDRA CoGEx for regulatory targets and map to UniProt IDs in data.

    Args:
        gene_symbol: Gene symbol to query (e.g., "C9ORF72")
        feature_ids: List of UniProt or Ensembl IDs in the dataset
        min_evidence: Minimum INDRA evidence count (default: 1)
        env_file: Path to .env file with INDRA credentials
        verbose: Print progress information

    Returns:
        Dict mapping {gene_symbol: uniprot_id} for targets found in data

    Raises:
        ImportError: If INDRA packages not available
        ValueError: If INDRA credentials not available
    """
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols

    if verbose:
        print(f"\nQuerying INDRA CoGEx for {gene_symbol} regulatory targets...")

    # Initialize INDRA knowledge source
    try:
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource
    except ImportError as e:
        raise ImportError(
            "INDRA packages required for network queries. "
            "Install with: pip install git+https://github.com/indralab/indra_cogex.git"
        ) from e

    # Initialize with env file
    try:
        if env_file and env_file.exists():
            indra_source = INDRAKnowledgeSource(env_file=str(env_file))
        else:
            indra_source = INDRAKnowledgeSource()
    except ValueError as e:
        raise ValueError(
            f"INDRA credentials not available. Please ensure credentials are in:\n"
            f"  - Environment variables (INDRA_NEO4J_URL, INDRA_NEO4J_USER, INDRA_NEO4J_PASSWORD)\n"
            f"  - .env file at {env_file}\n"
            f"Original error: {e}"
        ) from e

    # Query INDRA for regulatory targets
    edges = indra_source.get_edges(
        source_entity=gene_symbol,
        relationship_types=None,  # All relationship types
        min_evidence=min_evidence,
        min_confidence=0.0,
    )

    if not edges:
        if verbose:
            print(f"  No targets found for {gene_symbol} (min_evidence={min_evidence})")
        indra_source.close()
        return {}

    # Extract target gene symbols
    target_symbols = {edge.target for edge in edges}
    if verbose:
        print(f"  Found {len(target_symbols)} INDRA targets for {gene_symbol}")
        print(f"  Evidence counts: min={min([e.evidence_count for e in edges])}, "
              f"max={max([e.evidence_count for e in edges])}, "
              f"median={np.median([e.evidence_count for e in edges]):.1f}")

    # Map feature IDs to gene symbols (returns {symbol: feature_id})
    if verbose:
        print(f"\nMapping {len(feature_ids)} feature IDs to gene symbols...")
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

    # Find targets present in the data
    targets_in_data = {}
    for target_symbol in target_symbols:
        if target_symbol in symbol_to_feature:
            targets_in_data[target_symbol] = symbol_to_feature[target_symbol]

    if verbose:
        print(f"\nNetwork query results:")
        print(f"  {gene_symbol} -> {len(target_symbols)} INDRA targets")
        print(f"  {len(targets_in_data)} targets found in dataset ({len(targets_in_data)/len(target_symbols)*100:.1f}%)")
        if targets_in_data:
            print(f"  Example targets: {', '.join(list(targets_in_data.keys())[:5])}")

    indra_source.close()
    return targets_in_data


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the differential subcommand to the parser."""
    parser = subparsers.add_parser(
        "differential",
        help="Clique-level differential abundance analysis (MSstats-inspired)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input files
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Protein abundance data CSV (features × samples)",
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
        required=False,
        default=None,
        help="Clique definitions CSV (from analyze command). "
             "Mutually exclusive with --discover-gene-sets.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for results",
    )

    # INDRA gene set discovery (alternative to --cliques)
    parser.add_argument(
        "--discover-gene-sets",
        action="store_true",
        help="Discover INDRA regulatory modules as gene sets (alternative to --cliques). "
             "Uses single batch reverse query to find regulators targeting genes in data.",
    )
    parser.add_argument(
        "--regulator-class",
        nargs="+",
        choices=["tf", "kinase", "phosphatase", "e3_ligase", "receptor_kinase"],
        default=None,
        help="Filter discovered regulators by functional class "
             "(e.g., --regulator-class tf kinase). Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--stmt-types",
        type=str,
        default=None,
        help="INDRA statement types: preset (regulatory, activation, repression, "
             "phosphorylation) or comma-separated raw types. Default: regulatory. "
             "Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--strict-stmt-types",
        action="store_true",
        default=False,
        help="Warn when using the mixed 'regulatory' preset, which conflates "
             "activators and repressors. Suggests using --stmt-types activation "
             "or --stmt-types repression for directional analysis. "
             "Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--exclude-mixed-cliques",
        action="store_true",
        default=False,
        help="Exclude gene sets with mixed activation/repression directions. "
             "Only retains coherent cliques whose direction is 'positive' or "
             "'negative'. Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--min-targets",
        type=_positive_int,
        default=10,
        help="Minimum INDRA targets in data for a regulator (default: 10). "
             "Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Maximum targets per regulator, excludes hub regulators. "
             "Only with --discover-gene-sets.",
    )
    parser.add_argument(
        "--max-regulators",
        type=int,
        default=None,
        help="Maximum number of regulators to discover. "
             "Only with --discover-gene-sets.",
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
        help="Metadata column for subject IDs (enables mixed model)",
    )
    parser.add_argument(
        "--contrast",
        type=str,
        nargs=3,
        action="append",
        metavar=("NAME", "COND1", "COND2"),
        help="Contrast to test: NAME CONDITION1 CONDITION2 (can specify multiple)",
    )

    # Statistical options
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
        choices=["none", "min_feature", "aft", "qrilc", "knn"],
        default="none",
        help="Missing value imputation (default: none)",
    )
    parser.add_argument(
        "--no-mixed-model",
        action="store_true",
        help="Use fixed effects model only (ignore subject info)",
    )
    parser.add_argument(
        "--fdr-method",
        type=str,
        choices=["BH", "BY", "bonferroni"],
        default="BH",
        help="FDR correction method (default: BH = Benjamini-Hochberg)",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR significance threshold (default: 0.05)",
    )

    # Filtering
    parser.add_argument(
        "--min-proteins",
        type=_positive_int,
        default=3,
        help="Minimum proteins required per clique (default: 3)",
    )
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=None,
        help="Minimum clique coherence to include (default: no filter)",
    )

    # Additional analysis
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clique", "protein"],
        default="clique",
        help="Analysis mode: clique-level or protein-level (default: clique)",
    )
    parser.add_argument(
        "--also-protein-level",
        action="store_true",
        help="Also run protein-level differential analysis (only for clique mode)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=_positive_int,
        default=1,
        help="Parallel workers (default: 1)",
    )

    # Genetic subtype analysis
    parser.add_argument(
        "--genetic-contrast",
        type=str,
        metavar="MUTATION",
        help="Genetic subtype contrast (e.g., 'C9orf72' for carriers vs sporadic ALS). "
             "Auto-detects genomic columns (repeat length, ExpansionHunter) for "
             "biologically accurate carrier definition. "
             "Known mutations: C9orf72, SOD1, TARDBP, FUS, SETX, Multiple, Other",
    )
    parser.add_argument(
        "--cohort-config",
        type=Path,
        metavar="PATH",
        help="YAML/JSON cohort definition file for declarative cohort assignment. "
             "See cohorts/ directory for examples. Overrides --genetic-contrast.",
    )

    # Permutation testing (competitive null)
    parser.add_argument(
        "--permutation-test",
        action="store_true",
        help="Use competitive permutation testing instead of BH FDR correction",
    )
    parser.add_argument(
        "--n-permutations",
        type=_positive_int,
        default=1000,
        help="Number of permutations for null distribution (default: 1000)",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Random seed for reproducible permutation testing",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for permutation testing (requires MLX)",
    )
    parser.add_argument(
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="Disable GPU acceleration, use CPU parallelization",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=False,
        help="Force CPU computation for exact float64 reproducibility. "
             "The GPU (MLX) path uses float32 for rotation inner loops, "
             "which can shift p-values by ~1e-4 for extreme t-statistics "
             "(|t| > 10). Use this flag for cross-platform benchmarking "
             "or regression testing.",
    )
    parser.set_defaults(gpu=True)  # GPU is default if available

    # ROAST rotation-based gene set testing
    parser.add_argument(
        "--roast",
        action="store_true",
        help="Use ROAST rotation-based gene set test (detects bidirectional regulation, no FDR)",
    )
    parser.add_argument(
        "--n-rotations",
        type=_positive_int,
        default=9999,
        help="Number of rotations for ROAST (default: 9999)",
    )
    parser.add_argument(
        "--interaction",
        nargs=2,
        metavar=("FACTOR1_COL", "FACTOR2_COL"),
        help="Test interaction between two factors (e.g., --interaction sex phenotype). "
             "Requires --roast. Tests: (F1_L1×F2_L1 - F1_L1×F2_L2) - (F1_L2×F2_L1 - F1_L2×F2_L2)",
    )

    # Empirical Bayes moderation (limma-style)
    parser.add_argument(
        "--eb-moderation",
        action="store_true",
        default=True,
        dest="eb_moderation",
        help="Use Empirical Bayes moderated t-statistics (default: enabled)",
    )
    parser.add_argument(
        "--no-eb-moderation",
        action="store_false",
        dest="eb_moderation",
        help="Disable Empirical Bayes moderation (use standard t-statistics)",
    )

    # Covariate adjustment
    parser.add_argument(
        "--covariates",
        nargs="+",
        metavar="COL",
        help="Metadata columns to include as fixed covariates in the design "
             "matrix (e.g., --covariates Sex Batch). Adjusts for confounders "
             "while testing the condition effect. Categorical columns are "
             "dummy-coded; numeric columns are standardized.",
    )

    # Validation / baseline controls
    parser.add_argument(
        "--label-permutation-null",
        type=int,
        metavar="N",
        default=None,
        dest="label_perm_n",
        help="Run label permutation null with N permutations. Permutes "
             "condition labels and re-runs protein differential + enrichment "
             "to compute a permutation p-value for the enrichment z-score.",
    )
    parser.add_argument(
        "--permutation-stratify",
        type=str,
        metavar="COL",
        default=None,
        dest="perm_stratify_col",
        help="Stratification column for label permutation (e.g., Sex). "
             "Permutes labels within each stratum to preserve covariate balance.",
    )
    parser.add_argument(
        "--match-covariates",
        nargs="+",
        metavar="COL",
        default=None,
        help="Create sex/covariate-matched subset before analysis. "
             "Exact-matches groups on specified columns by downsampling "
             "larger groups (e.g., --match-covariates Sex).",
    )
    parser.add_argument(
        "--negative-control-sets",
        type=int,
        metavar="N",
        default=None,
        dest="neg_control_n",
        help="Run ROAST on N random gene sets of the same size as the "
             "target set to estimate false positive rate and calibrate "
             "significance.",
    )

    # Network query integration
    parser.add_argument(
        "--network-query",
        type=str,
        metavar="GENE",
        help="Query INDRA CoGEx for regulatory targets of this gene (e.g., C9ORF72)",
    )
    parser.add_argument(
        "--min-evidence",
        type=_positive_int,
        default=1,
        help="Minimum INDRA evidence count for network edges (default: 1)",
    )
    parser.add_argument(
        "--indra-env-file",
        type=Path,
        default=Path(os.environ.get("INDRA_ENV_FILE", Path.home() / ".indra" / ".env")),
        help="Path to .env file with INDRA CoGEx credentials "
             "(default: $INDRA_ENV_FILE or ~/.indra/.env)",
    )

    # Network enrichment testing
    parser.add_argument(
        "--enrichment-test",
        action="store_true",
        help="Run competitive permutation enrichment test on network targets. "
             "Requires --network-query. Tests if network targets have higher |t| "
             "than random protein sets of same size.",
    )

    parser.set_defaults(func=run_differential)


def run_differential(args: argparse.Namespace) -> int:
    """Execute clique differential analysis."""
    from cliquefinder.io.loaders import load_csv_matrix
    from cliquefinder.stats import (
        SummarizationMethod,
        NormalizationMethod,
        ImputationMethod,
        load_clique_definitions,
        run_clique_differential_analysis,
    )
    from cliquefinder.stats.clique_analysis import run_permutation_clique_test

    # Validate argument dependencies
    if getattr(args, 'enrichment_test', False) and not args.network_query:
        print("Error: --enrichment-test requires --network-query")
        return 1

    discover_gene_sets = getattr(args, 'discover_gene_sets', False)
    if discover_gene_sets and args.cliques:
        print("Error: --discover-gene-sets and --cliques are mutually exclusive")
        return 1

    if args.mode == "clique" and not discover_gene_sets and not args.cliques:
        print("Error: clique mode requires either --cliques or --discover-gene-sets")
        return 1

    print("=" * 70)
    print("  Clique-Level Differential Abundance Analysis")
    print("  (MSstats-inspired methodology)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print(f"Loading data: {args.data}")
    matrix = load_csv_matrix(args.data)
    print(f"  {matrix.n_features} features × {matrix.n_samples} samples")

    # Load metadata
    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata, index_col=0)

    # Handle cohort-based contrast (--cohort-config or --genetic-contrast)
    condition_col = args.condition_col
    cohort_config = getattr(args, 'cohort_config', None)
    genetic_contrast = getattr(args, 'genetic_contrast', None)

    metadata, condition_col, cohort_contrasts = resolve_cohort_from_args(
        metadata=metadata,
        cohort_config=cohort_config,
        genetic_contrast=genetic_contrast,
        condition_col=condition_col,
    )

    if cohort_contrasts is not None:
        if args.contrast:
            print(f"  Warning: Ignoring --contrast (overridden by cohort resolution)")
        args.contrast = cohort_contrasts

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

    # Covariate matching (creates balanced subset before analysis)
    match_result = None
    if getattr(args, 'match_covariates', None):
        from cliquefinder.stats.matching import exact_match_covariates

        print(f"\nCreating covariate-matched subset...")
        print(f"  Match variables: {args.match_covariates}")

        # Determine groups from contrast or condition column
        match_groups = None
        if args.contrast:
            # Use groups from all contrasts
            match_groups = list({c for _, c1, c2 in args.contrast for c in (c1, c2)})

        match_result = exact_match_covariates(
            metadata=metadata,
            group_col=condition_col,
            match_vars=args.match_covariates,
            groups=match_groups,
            seed=args.permutation_seed,
        )

        print(f"  Original: {match_result.n_original} → Matched: {match_result.n_matched}")
        for group, dropped in match_result.dropped_per_group.items():
            print(f"    {group}: dropped {dropped}")

        # Apply matching: subset data and metadata
        matched_idx_in_meta = match_result.matched_indices
        metadata = metadata.iloc[matched_idx_in_meta].copy()
        data = data[:, matched_idx_in_meta]
        common_samples = [common_samples[i] for i in matched_idx_in_meta]
        print(f"  Proceeding with {len(common_samples)} matched samples")

    # Load clique definitions (skip if protein-only mode)

    # Network query integration
    network_targets = None
    if args.network_query:
        try:
            network_targets = query_network_targets(
                gene_symbol=args.network_query,
                feature_ids=feature_ids,
                min_evidence=args.min_evidence,
                env_file=args.indra_env_file,
                verbose=True,
            )
        except (ImportError, ValueError) as e:
            print(f"\nError: Network query failed: {e}")
            print("Continuing without network filtering...")
            network_targets = None
    cliques = None
    if args.mode == "clique":
        if discover_gene_sets:
            # Discover INDRA modules as gene sets
            print(f"\nDiscovering INDRA gene sets for differential testing...")
            from cliquefinder.stats.clique_analysis import discover_gene_sets_from_indra
            from cliquefinder.knowledge.cogex import RegulatorClass, resolve_stmt_types

            # Parse regulator classes
            regulator_classes = None
            if args.regulator_class:
                _CLI_TO_ENUM = {
                    "tf": RegulatorClass.TF,
                    "kinase": RegulatorClass.KINASE,
                    "phosphatase": RegulatorClass.PHOSPHATASE,
                    "e3_ligase": RegulatorClass.E3_LIGASE,
                    "receptor_kinase": RegulatorClass.RECEPTOR_KINASE,
                }
                regulator_classes = {_CLI_TO_ENUM[c] for c in args.regulator_class}
                print(f"  Regulator class filter: {args.regulator_class}")

            # Resolve stmt_types
            stmt_types = resolve_stmt_types(args.stmt_types) if args.stmt_types else None
            if stmt_types:
                print(f"  Statement types: {stmt_types}")

            # Strict stmt-types warning (H5 audit finding)
            if getattr(args, 'strict_stmt_types', False) and (args.stmt_types is None or args.stmt_types == "regulatory"):
                import warnings
                warnings.warn(
                    "The default --stmt-types 'regulatory' preset conflates activators "
                    "(IncreaseAmount, Activation) and repressors (DecreaseAmount, Inhibition) "
                    "into a single gene set, which can dilute directional enrichment signals. "
                    "Consider using --stmt-types activation or --stmt-types repression for "
                    "directional analysis. To silence this warning, choose a specific preset.",
                    UserWarning,
                    stacklevel=1,
                )

            cliques = discover_gene_sets_from_indra(
                feature_ids=feature_ids,
                min_evidence=args.min_evidence,
                min_targets=getattr(args, 'min_targets', 10),
                max_targets=getattr(args, 'max_targets', None),
                max_regulators=getattr(args, 'max_regulators', None),
                regulator_classes=regulator_classes,
                stmt_types=stmt_types,
                min_genes_found=args.min_proteins,
                env_file=str(args.indra_env_file) if args.indra_env_file else None,
                verbose=True,
            )
            print(f"  {len(cliques)} gene sets discovered")

            # Filter out mixed-direction cliques if requested (M7 audit finding)
            if getattr(args, 'exclude_mixed_cliques', False):
                pre_filter = len(cliques)
                cliques = [c for c in cliques if c.direction in ("positive", "negative")]
                print(f"  After excluding mixed cliques: {len(cliques)} gene sets "
                      f"(removed {pre_filter - len(cliques)} mixed-direction sets)")

        else:
            # Load from CSV (existing path)
            print(f"\nLoading cliques: {args.cliques}")
            cliques = load_clique_definitions(args.cliques, min_proteins=args.min_proteins)
            print(f"  {len(cliques)} cliques loaded (min {args.min_proteins} proteins)")

            # Filter by coherence if specified
            if args.min_coherence:
                original_count = len(cliques)
                cliques = [c for c in cliques if c.coherence is None or c.coherence >= args.min_coherence]
                print(f"  Filtered by coherence >= {args.min_coherence}: {len(cliques)} remaining")
    else:
        print(f"\nMode: protein-level analysis (skipping clique loading)")

    # Parse contrasts
    contrasts = None
    if args.contrast:
        contrasts = {}
        for name, cond1, cond2 in args.contrast:
            contrasts[name] = (cond1, cond2)
        print(f"\nContrasts to test:")
        for name, (c1, c2) in contrasts.items():
            print(f"  {name}: {c1} vs {c2}")
    else:
        # Auto-detect from condition column
        conditions = sorted(metadata[condition_col].dropna().unique())
        if len(conditions) >= 2:
            contrasts = {f"{conditions[0]}_vs_{conditions[1]}": (conditions[0], conditions[1])}
            print(f"\nAuto-detected contrast: {conditions[0]} vs {conditions[1]}")

    # Map method strings to enums
    summarization = SummarizationMethod(args.summarization)
    normalization = NormalizationMethod(args.normalization)
    imputation = ImputationMethod(args.imputation)

    print(f"\nStatistical settings:")
    print(f"  Summarization: {summarization.value}")
    print(f"  Normalization: {normalization.value}")
    print(f"  Imputation: {imputation.value}")
    print(f"  Model: {'Mixed (with subject random effects)' if not args.no_mixed_model and args.subject_col else 'Fixed effects'}")
    if args.roast:
        if args.interaction:
            print(f"  Method: ROAST interaction test ({args.n_rotations} rotations)")
            print(f"  Interaction: {args.interaction[0]} × {args.interaction[1]}")
        else:
            print(f"  Method: ROAST rotation-based gene set test ({args.n_rotations} rotations)")
        print(f"  Significance: Raw p-values (no FDR - statistically valid for ROAST)")
    elif args.permutation_test:
        print(f"  Significance testing: Competitive permutation ({args.n_permutations} permutations)")
        print(f"  Permutation seed: {args.permutation_seed or 'random'}")
    else:
        print(f"  FDR method: {args.fdr_method}")
    if not args.roast:
        print(f"  Significance threshold: {args.fdr_threshold}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run analysis
    print(f"\n{'=' * 70}")

    # Branch based on mode
    if args.mode == "protein":
        # Protein-level differential analysis
        from cliquefinder.stats.differential import run_differential_analysis

        print("Running protein-level differential analysis...")
        print("=" * 70)

        # Handle enrichment test flow
        enrichment = None
        if args.network_query and getattr(args, 'enrichment_test', False) and network_targets:
            from cliquefinder.stats.differential import (
                run_protein_differential,
                run_network_enrichment_test,
            )

            # Get target UniProt IDs from network query
            target_gene_ids = list(network_targets.values())

            # Extract contrast tuple for the genetic contrast case
            contrast_tuple = list(contrasts.values())[0]

            print(f"\nRunning genome-wide EB differential with target flagging...")
            print(f"  Network targets: {len(target_gene_ids)}")

            # Build covariates DataFrame if specified
            covariates_df = None
            if getattr(args, 'covariates', None):
                covariates_df = metadata[args.covariates]

            # Run genome-wide EB differential with target flagging
            protein_df = run_protein_differential(
                data=data,
                feature_ids=feature_ids,
                sample_condition=metadata[condition_col],
                contrast=contrast_tuple,
                eb_moderation=args.eb_moderation,
                target_gene_ids=target_gene_ids,
                verbose=True,
                covariates_df=covariates_df,
            )

            # Run enrichment test
            print(f"\nRunning competitive permutation enrichment test...")
            print(f"  Permutations: {args.n_permutations}")
            if args.permutation_seed:
                print(f"  Seed: {args.permutation_seed}")

            enrichment = run_network_enrichment_test(
                protein_results=protein_df,
                n_permutations=args.n_permutations,
                seed=args.permutation_seed,
                verbose=True,
            )

            # Add gene symbols for readability
            # network_targets is {symbol: uniprot}, invert for uniprot -> symbol
            uniprot_to_symbol = {v: k for k, v in network_targets.items()}
            # For non-target proteins, use the symbol mapping we already computed
            from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
            symbol_to_uniprot = map_feature_ids_to_symbols(feature_ids, verbose=False)
            # Invert: uniprot -> symbol (use first symbol if multiple)
            for sym, uid in symbol_to_uniprot.items():
                if uid not in uniprot_to_symbol:
                    uniprot_to_symbol[uid] = sym
            # Add gene_symbol column
            protein_df['gene_symbol'] = protein_df['feature_id'].map(uniprot_to_symbol)
            # Reorder columns: gene_symbol first for readability
            cols = protein_df.columns.tolist()
            cols.remove('gene_symbol')
            protein_df = protein_df[['gene_symbol'] + cols]

            # Save genome-wide results
            all_proteins_output = args.output / "all_proteins.csv"
            protein_df.to_csv(all_proteins_output, index=False)
            print(f"\nAll proteins: {all_proteins_output}")

            # Save network targets only
            targets_df = protein_df[protein_df['is_target']]
            targets_output = args.output / "network_targets.csv"
            targets_df.to_csv(targets_output, index=False)
            print(f"Network targets: {targets_output}")

            # Save enrichment results
            enrichment_output = args.output / "enrichment_results.json"
            with open(enrichment_output, "w") as f:
                json.dump(enrichment.to_dict(), f, indent=2)
            print(f"Enrichment results: {enrichment_output}")

            # Label permutation null (if requested)
            if getattr(args, 'label_perm_n', None):
                from cliquefinder.stats.label_permutation import run_label_permutation_null

                stratify_by = None
                if getattr(args, 'perm_stratify_col', None):
                    stratify_by = metadata[args.perm_stratify_col].values

                print(f"\n{'=' * 70}")
                print("LABEL PERMUTATION NULL")
                print("=" * 70)

                perm_result = run_label_permutation_null(
                    data=data,
                    feature_ids=feature_ids,
                    sample_condition=metadata[condition_col],
                    contrast=contrast_tuple,
                    target_gene_ids=target_gene_ids,
                    n_permutations=args.label_perm_n,
                    stratify_by=stratify_by,
                    covariates_df=covariates_df,
                    eb_moderation=args.eb_moderation,
                    seed=args.permutation_seed,
                    verbose=True,
                )

                perm_output = args.output / "label_permutation_null.json"
                with open(perm_output, "w") as f:
                    json.dump(perm_result.to_dict(), f, indent=2)
                print(f"Label permutation results: {perm_output}")

            # Create result-like object for downstream compatibility
            class EnrichmentResult:
                def __init__(self, df):
                    self.n_features_tested = len(df)
                    self.n_significant = df['significant'].sum() if 'significant' in df.columns else 0

                def to_dataframe(self):
                    return protein_df

            result = EnrichmentResult(protein_df)
            sig_proteins = targets_df  # For summary, show targets

        else:
            # Standard protein-level analysis (no enrichment test)
            result = run_differential_analysis(
                data=data,
                feature_ids=feature_ids,
                sample_condition=metadata[condition_col],
                sample_subject=metadata[args.subject_col] if args.subject_col else None,
                contrasts=contrasts,
                use_mixed=not args.no_mixed_model,
                fdr_method=args.fdr_method,
                fdr_threshold=args.fdr_threshold,
                n_jobs=args.workers,
                verbose=True,
            )

            # Save results
            protein_df = result.to_dataframe()
            protein_output = args.output / "protein_differential.csv"
            protein_df.to_csv(protein_output, index=False)
            print(f"\nProtein results: {protein_output}")

            # Save significant proteins
            sig_proteins = protein_df[protein_df['significant']]
            if len(sig_proteins) > 0:
                sig_output = args.output / "significant_proteins.csv"
                sig_proteins.to_csv(sig_output, index=False)
                print(f"Significant proteins: {sig_output}")

            # Normalize result interface for downstream params/summary
            result.n_features_tested = len(result.results)
            result.n_significant = len(sig_proteins)

        # Save parameters
        params = {
            "timestamp": datetime.now().isoformat(),
            "mode": "protein",
            "data": str(args.data),
            "metadata": str(args.metadata),
            "condition_col": condition_col,
            "subject_col": args.subject_col,
            "contrasts": contrasts,
            "normalization": normalization.value,
            "imputation": imputation.value,
            "use_mixed_model": not args.no_mixed_model,
            "fdr_method": args.fdr_method,
            "fdr_threshold": args.fdr_threshold,
            "n_features_tested": result.n_features_tested,
            "n_significant": result.n_significant,
        }
        if args.genetic_contrast:
            params["genetic_contrast"] = args.genetic_contrast
        if args.network_query:
            params["network_query"] = args.network_query
            params["min_evidence"] = args.min_evidence
            params["n_network_targets"] = len(network_targets) if network_targets else 0
        if getattr(args, 'enrichment_test', False):
            params["enrichment_test"] = True
            params["n_permutations"] = args.n_permutations
            params["permutation_seed"] = args.permutation_seed

        params_output = args.output / "analysis_parameters.json"
        with open(params_output, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters: {params_output}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("SUMMARY (Protein-Level)")
        print("=" * 70)
        print(f"Proteins tested: {result.n_features_tested}")
        print(f"Significant (FDR < {args.fdr_threshold}): {result.n_significant}")

        if not getattr(args, 'enrichment_test', False) and len(sig_proteins) > 0:
            print(f"\nTop significant proteins:")
            top = sig_proteins.nsmallest(10, 'adj_pvalue')[['feature_id', 'log2FC', 'adj_pvalue', 'contrast']]
            for _, row in top.iterrows():
                direction = "↑" if row['log2FC'] > 0 else "↓"
                print(f"  {row['feature_id']}: log2FC={row['log2FC']:.3f} {direction}, q={row['adj_pvalue']:.2e}")

        # Print enrichment test summary
        if getattr(args, 'enrichment_test', False) and enrichment:
            print(f"\n{'=' * 70}")
            print("NETWORK ENRICHMENT TEST")
            print("=" * 70)
            print(f"Network targets: {enrichment.n_targets}")
            print(f"Background proteins: {enrichment.n_background}")
            print(f"Observed mean |t|: {enrichment.observed_mean_abs_t:.4f}")
            print(f"Null mean |t|: {enrichment.null_mean:.4f}")
            print(f"Z-score: {enrichment.z_score:.2f}")
            print(f"Empirical p-value: {enrichment.empirical_pvalue:.4f}")
            print(f"Direction: {enrichment.pct_down:.0f}% targets downregulated")

        print(f"\nComplete! Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    # STAT-11: --force-cpu overrides GPU for exact float64 reproducibility
    use_gpu = args.gpu
    if getattr(args, 'force_cpu', False):
        use_gpu = False
        print("  Note: --force-cpu active, using float64 CPU path for exact reproducibility")

    # Clique-level analysis
    if args.roast:
        # ROAST rotation-based gene set testing
        import time
        start_time = time.time()

        if args.interaction:
            # Interaction analysis (2×2 factorial)
            from cliquefinder.stats.clique_analysis import run_clique_roast_interaction_analysis

            factor1_col, factor2_col = args.interaction

            print(f"Running ROAST interaction analysis: {factor1_col} × {factor2_col}")
            print("  (Tests whether effect of factor2 differs across levels of factor1)")
            print("=" * 70)

            clique_df = run_clique_roast_interaction_analysis(
                data=data,
                feature_ids=feature_ids,
                sample_metadata=metadata,
                clique_definitions=cliques,
                factor1_column=factor1_col,
                factor2_column=factor2_col,
                n_rotations=args.n_rotations,
                seed=args.permutation_seed,
                use_gpu=use_gpu,
                map_ids=True,
                verbose=True,
            )

            contrast_tuple = (factor1_col, factor2_col)  # For params
        else:
            # Standard two-group comparison
            from cliquefinder.stats.clique_analysis import run_clique_roast_analysis

            print("Running ROAST rotation-based clique analysis...")
            print("  (Exact p-values via rotation, detects bidirectional regulation)")
            print("=" * 70)

            # Parse the first contrast for ROAST
            contrast_tuple = list(contrasts.values())[0]
            conditions = list(contrast_tuple)

            clique_df = run_clique_roast_analysis(
                data=data,
                feature_ids=feature_ids,
                sample_metadata=metadata,
                clique_definitions=cliques,
                condition_column=condition_col,
                conditions=conditions,
                contrast=contrast_tuple,
                n_rotations=args.n_rotations,
                seed=args.permutation_seed,  # Reuse seed arg
                use_gpu=use_gpu,
                map_ids=True,
                verbose=True,
                covariates=getattr(args, 'covariates', None),
            )

        elapsed_time = time.time() - start_time
        print(f"\nROAST analysis completed in {elapsed_time:.2f}s")

        # Save results
        clique_output = args.output / "roast_clique_results.csv"
        clique_df.to_csv(clique_output, index=False)
        print(f"\nROAST results: {clique_output}")

        # Save top hits (p < 0.05)
        top_hits = clique_df[clique_df['pvalue_msq_mixed'] < 0.05]
        n_top_hits = len(top_hits)
        if n_top_hits > 0:
            top_output = args.output / "roast_top_hits.csv"
            top_hits.to_csv(top_output, index=False)
            print(f"Top hits (p < 0.05): {top_output}")

        # Save bidirectional candidates (MSQ p < 0.05, MEAN not significant)
        bidir = clique_df[
            (clique_df['pvalue_msq_mixed'] < 0.05) &
            (clique_df['pvalue_mean_up'] >= 0.05) &
            (clique_df['pvalue_mean_down'] >= 0.05)
        ]
        if len(bidir) > 0:
            bidir_output = args.output / "roast_bidirectional_candidates.csv"
            bidir.to_csv(bidir_output, index=False)
            print(f"Bidirectional candidates: {bidir_output}")

        # Negative control gene sets (if requested, ROAST path only)
        if getattr(args, 'neg_control_n', None) and not args.interaction:
            from cliquefinder.stats.negative_controls import run_negative_control_sets
            from cliquefinder.stats.rotation import RotationTestEngine, RotationTestConfig

            print(f"\n{'=' * 70}")
            print("NEGATIVE CONTROL GENE SETS")
            print("=" * 70)

            # Build a fresh engine (fast — just QR decomposition)
            neg_engine = RotationTestEngine(data, feature_ids, metadata)
            neg_engine.fit(
                conditions=conditions,
                contrast=contrast_tuple,
                condition_column=condition_col,
                covariates=getattr(args, 'covariates', None),
            )

            # Use the first (most significant) clique as the target
            best_clique = clique_df.iloc[0]
            target_id = best_clique['feature_set_id']
            target_genes_str = best_clique.get('clique_genes', '')
            if target_genes_str:
                target_gene_list = [g.strip() for g in target_genes_str.split(',')]
            else:
                target_gene_list = []

            if target_gene_list:
                neg_result = run_negative_control_sets(
                    engine=neg_engine,
                    target_gene_ids=target_gene_list,
                    target_set_id=target_id,
                    n_control_sets=args.neg_control_n,
                    seed=args.permutation_seed,
                    verbose=True,
                )

                neg_output = args.output / "negative_control_sets.json"
                with open(neg_output, "w") as f:
                    json.dump(neg_result.to_dict(), f, indent=2)
                print(f"Negative control results: {neg_output}")

        # Save parameters
        n_cliques_tested = len(clique_df)
        params = {
            "timestamp": datetime.now().isoformat(),
            "mode": "clique",
            "method": "ROAST" if not args.interaction else "ROAST_interaction",
            "data": str(args.data),
            "metadata": str(args.metadata),
            "cliques": str(args.cliques) if args.cliques else None,
            "n_rotations": args.n_rotations,
            "seed": args.permutation_seed,
            "use_gpu": use_gpu,
            "n_cliques_tested": n_cliques_tested,
            "n_top_hits_p05": n_top_hits,
            "n_bidirectional": len(bidir),
            "min_pvalue_msq": float(clique_df['pvalue_msq_mixed'].min()) if 'pvalue_msq_mixed' in clique_df.columns else None,
        }
        if args.interaction:
            params["interaction"] = {
                "factor1": args.interaction[0],
                "factor2": args.interaction[1],
            }
        else:
            params["condition_col"] = condition_col
            params["contrast"] = contrast_tuple
        params_output = args.output / "analysis_parameters.json"
        with open(params_output, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters: {params_output}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("SUMMARY (ROAST Rotation Gene Set Test)")
        print("=" * 70)
        print(f"Cliques tested: {n_cliques_tested}")
        print(f"Top hits (p < 0.05): {n_top_hits}")
        print(f"Bidirectional candidates: {len(bidir)}")
        if 'pvalue_msq_mixed' in clique_df.columns:
            print(f"Minimum p-value (MSQ): {clique_df['pvalue_msq_mixed'].min():.4f}")

        if n_top_hits > 0:
            print(f"\nTop cliques by MSQ p-value:")
            top = clique_df.nsmallest(10, 'pvalue_msq_mixed')[['feature_set_id', 'clique_genes', 'n_genes_found', 'pvalue_msq_mixed', 'pvalue_mean_down']]
            for _, row in top.iterrows():
                genes = row['clique_genes'][:40] + '...' if len(str(row['clique_genes'])) > 40 else row['clique_genes']
                print(f"  {row['feature_set_id']}: p_msq={row['pvalue_msq_mixed']:.4f}, p_down={row['pvalue_mean_down']:.4f}, genes={genes}")

    elif args.permutation_test:
        # Competitive permutation testing
        print("Running competitive permutation clique analysis...")
        print("  (Random protein sets of same cardinality as null)")
        print("=" * 70)

        # Parse the first contrast for permutation test
        contrast_tuple = list(contrasts.values())[0]

        # Try GPU implementation first if requested (respects --force-cpu)
        gpu_available = False

        if use_gpu:
            try:
                from cliquefinder.stats.permutation_gpu import run_permutation_test_gpu
                import mlx.core as mx
                gpu_available = True
                print(f"\nUsing GPU acceleration (MLX)")
            except ImportError:
                print(f"\nWarning: MLX not available, falling back to CPU parallelization")
                use_gpu = False

        import time
        start_time = time.time()

        if use_gpu and gpu_available:
            # GPU-accelerated implementation
            perm_results, null_df = run_permutation_test_gpu(
                data=data,
                feature_ids=feature_ids,
                sample_metadata=metadata,
                clique_definitions=cliques,
                condition_col=condition_col,
                subject_col=args.subject_col,
                contrast=contrast_tuple,
                summarization_method=summarization,
                n_permutations=args.n_permutations,
                use_mixed_model=not args.no_mixed_model,
                significance_threshold=args.fdr_threshold,
                random_state=args.permutation_seed,
                n_jobs=args.workers,
                verbose=True,
                eb_moderation=args.eb_moderation,
            )
        else:
            # CPU parallel implementation
            print(f"\nUsing CPU parallelization ({args.workers} workers)")
            perm_results, null_df = run_permutation_clique_test(
                data=data,
                feature_ids=feature_ids,
                sample_metadata=metadata,
                clique_definitions=cliques,
                condition_col=condition_col,
                subject_col=args.subject_col,
                contrast=contrast_tuple,
                summarization_method=summarization,
                n_permutations=args.n_permutations,
                use_mixed_model=not args.no_mixed_model,
                significance_threshold=args.fdr_threshold,
                random_state=args.permutation_seed,
                n_jobs=args.workers,
                verbose=True,
            )

        elapsed_time = time.time() - start_time
        print(f"\nPermutation testing completed in {elapsed_time:.2f}s")

        # Convert permutation results to DataFrame
        clique_df = pd.DataFrame([r.to_dict() for r in perm_results])

        # Rename columns for consistency
        clique_df = clique_df.rename(columns={
            'observed_log2FC': 'log2FC',
            'observed_pvalue': 'pvalue',
            'observed_tvalue': 'tvalue',
            'empirical_pvalue': 'perm_pvalue',
        })

        # Merge clique member information
        if args.cliques and args.cliques.exists():
            cliques_source = pd.read_csv(args.cliques)
            if 'clique_genes' in cliques_source.columns:
                clique_info = cliques_source.groupby('regulator').agg({
                    'clique_genes': 'first',
                }).reset_index()
                clique_info = clique_info.rename(columns={'regulator': 'clique_id'})
                clique_info['n_proteins'] = clique_info['clique_genes'].str.count(',') + 1
                clique_df = clique_df.merge(clique_info, on='clique_id', how='left')
        elif cliques:
            # Build from in-memory definitions (--discover-gene-sets)
            clique_info_rows = []
            for cd in cliques:
                clique_info_rows.append({
                    'clique_id': cd.clique_id,
                    'clique_genes': ','.join(cd.protein_ids),
                    'n_proteins': len(cd.protein_ids),
                })
            clique_info = pd.DataFrame(clique_info_rows)
            clique_df = clique_df.merge(clique_info, on='clique_id', how='left')

        if 'clique_genes' in clique_df.columns:
            # Reorder columns to put clique_genes early
            cols = clique_df.columns.tolist()
            priority_cols = ['clique_id', 'clique_genes', 'n_proteins', 'log2FC', 'perm_pvalue']
            other_cols = [c for c in cols if c not in priority_cols]
            clique_df = clique_df[[c for c in priority_cols if c in cols] + other_cols]

        # Save permutation results
        clique_output = args.output / "clique_differential_permutation.csv"
        clique_df.to_csv(clique_output, index=False)
        print(f"\nPermutation results: {clique_output}")

        # Save null distribution summary
        null_output = args.output / "null_distribution_summary.csv"
        null_df.to_csv(null_output, index=False)
        print(f"Null distribution: {null_output}")

        # Save significant cliques
        sig_cliques = clique_df[clique_df['is_significant']]
        n_significant = len(sig_cliques)
        n_cliques_tested = len(clique_df)

        if len(sig_cliques) > 0:
            sig_output = args.output / "significant_cliques.csv"
            sig_cliques.to_csv(sig_output, index=False)
            print(f"Significant cliques: {sig_output}")

        # Save parameters
        params = {
            "timestamp": datetime.now().isoformat(),
            "mode": "clique",
            "data": str(args.data),
            "metadata": str(args.metadata),
            "cliques": str(args.cliques) if args.cliques else None,
            "condition_col": condition_col,
            "subject_col": args.subject_col,
            "contrasts": contrasts,
            "summarization": summarization.value,
            "normalization": normalization.value,
            "imputation": imputation.value,
            "use_mixed_model": not args.no_mixed_model,
            "significance_method": "permutation",
            "n_permutations": args.n_permutations,
            "permutation_seed": args.permutation_seed,
            "significance_threshold": args.fdr_threshold,
            "min_proteins": args.min_proteins,
            "n_cliques_tested": n_cliques_tested,
            "n_significant": n_significant,
            "network_query": args.network_query,
            "min_evidence": args.min_evidence if args.network_query else None,
            "n_network_targets": len(network_targets) if network_targets else 0,
        }
        if discover_gene_sets:
            params["gene_set_source"] = "indra_discovery"
            params["regulator_classes"] = args.regulator_class
            params["stmt_types"] = args.stmt_types
            params["min_targets_discovery"] = getattr(args, 'min_targets', 10)
            params["max_targets_discovery"] = getattr(args, 'max_targets', None)
            params["max_regulators_discovery"] = getattr(args, 'max_regulators', None)
        if args.genetic_contrast:
            params["genetic_contrast"] = args.genetic_contrast
        params_output = args.output / "analysis_parameters.json"
        with open(params_output, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters: {params_output}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("SUMMARY (Competitive Permutation Testing)")
        print("=" * 70)
        print(f"Cliques tested: {n_cliques_tested}")
        print(f"Significant (empirical p < {args.fdr_threshold}): {n_significant}")

        if len(sig_cliques) > 0:
            print(f"\nTop significant cliques:")
            top = sig_cliques.nsmallest(10, 'perm_pvalue')[['clique_id', 'log2FC', 'perm_pvalue', 'percentile_rank']]
            for _, row in top.iterrows():
                direction = "↑" if row['log2FC'] > 0 else "↓"
                print(f"  {row['clique_id']}: log2FC={row['log2FC']:.3f} {direction}, p_emp={row['perm_pvalue']:.3f}, percentile={row['percentile_rank']:.1f}")

    else:
        # Standard BH FDR correction
        print("Running clique differential analysis...")
        print("=" * 70)

        result = run_clique_differential_analysis(
            data=data,
            feature_ids=feature_ids,
            sample_metadata=metadata,
            clique_definitions=cliques,
            condition_col=args.condition_col,
            subject_col=args.subject_col,
            contrasts=contrasts,
            summarization_method=summarization,
            normalization_method=normalization,
            imputation_method=imputation,
            use_mixed_model=not args.no_mixed_model,
            fdr_method=args.fdr_method,
            fdr_threshold=args.fdr_threshold,
            min_proteins_found=2,
            also_run_protein_level=args.also_protein_level,
            n_jobs=args.workers,
            verbose=True,
        )

        # Save clique results
        clique_df = result.to_dataframe()
        clique_output = args.output / "clique_differential.csv"
        clique_df.to_csv(clique_output, index=False)
        print(f"\nClique results: {clique_output}")

        # Save protein results if available
        if result.protein_results is not None:
            protein_df = result.protein_results.to_dataframe()
            protein_output = args.output / "protein_differential.csv"
            protein_df.to_csv(protein_output, index=False)
            print(f"Protein results: {protein_output}")

        # Save significant cliques
        if 'adj_pvalue' in clique_df.columns:
            sig_cliques = clique_df[clique_df['adj_pvalue'] < args.fdr_threshold]
        else:
            sig_cliques = pd.DataFrame()

        if len(sig_cliques) > 0:
            sig_output = args.output / "significant_cliques.csv"
            sig_cliques.to_csv(sig_output, index=False)
            print(f"Significant cliques: {sig_output}")

        # Save parameters
        params = {
            "timestamp": datetime.now().isoformat(),
            "data": str(args.data),
            "metadata": str(args.metadata),
            "cliques": str(args.cliques) if args.cliques else None,
            "condition_col": args.condition_col,
            "subject_col": args.subject_col,
            "contrasts": contrasts,
            "summarization": summarization.value,
            "normalization": normalization.value,
            "imputation": imputation.value,
            "use_mixed_model": not args.no_mixed_model,
            "fdr_method": args.fdr_method,
            "fdr_threshold": args.fdr_threshold,
            "min_proteins": args.min_proteins,
            "n_cliques_tested": result.n_cliques_tested,
            "n_significant": result.n_significant,
            "network_query": args.network_query,
            "min_evidence": args.min_evidence if args.network_query else None,
            "n_network_targets": len(network_targets) if network_targets else 0,
        }
        params_output = args.output / "analysis_parameters.json"
        with open(params_output, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters: {params_output}")

        # Print summary
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)
        print(f"Cliques tested: {result.n_cliques_tested}")
        print(f"Significant (FDR < {args.fdr_threshold}): {result.n_significant}")

        if len(sig_cliques) > 0:
            print(f"\nTop significant cliques:")
            top = sig_cliques.nsmallest(10, 'adj_pvalue')[['clique_id', 'log2FC', 'adj_pvalue', 'n_proteins']]
            for _, row in top.iterrows():
                direction = "↑" if row['log2FC'] > 0 else "↓"
                print(f"  {row['clique_id']}: log2FC={row['log2FC']:.3f} {direction}, q={row['adj_pvalue']:.2e}, n={row['n_proteins']}")

    print(f"\nComplete! Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0
