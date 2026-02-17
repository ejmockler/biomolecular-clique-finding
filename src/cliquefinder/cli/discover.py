"""
CliqueFinder discover command - De novo co-expression module discovery.

Data-driven discovery without prior knowledge constraints.
Uses the same clique-finding machinery as regulatory validation,
but with gene universe derived from expression variance.

Usage:
    cliquefinder discover --input data.csv --n-genes 5000 --min-correlation 0.8
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the discover subcommand."""
    parser = subparsers.add_parser(
        "discover",
        help="De novo co-expression module discovery (data-driven)",
        description=(
            "Discover co-expression modules from high-variance genes. "
            "No prior knowledge required - purely data-driven discovery "
            "using the same clique-finding machinery as regulatory validation."
        )
    )

    # Input/output
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Expression data CSV (genes x samples)")
    parser.add_argument("--metadata", "-m", type=Path,
                        help="Sample metadata CSV (optional, enables stratification)")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/denovo"),
                        help="Output directory for results")

    # Gene universe selection
    parser.add_argument("--n-genes", type=int, default=5000,
                        help="Number of high-variance genes to analyze (default: 5000)")
    parser.add_argument("--percentile", type=float, default=None,
                        help="Alternative to --n-genes: top X percentile by variance")

    # Clique parameters (same as analyze)
    parser.add_argument("--min-correlation", type=float, default=0.7,
                        help="Minimum correlation for co-expression edge (default: 0.7)")
    parser.add_argument("--min-module-size", type=int, default=5,
                        help="Minimum genes per module (default: 5)")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson",
                        help="Correlation method (default: pearson)")
    parser.add_argument("--exact-cliques", action="store_true",
                        help="Use exact clique enumeration (slow) vs greedy (default)")

    # Stratification
    parser.add_argument("--stratify-by", nargs="+", default=None,
                        help="Metadata columns for condition stratification")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Specific conditions to analyze (default: all)")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Minimum samples per stratum (default: 20)")

    # Efficiency options
    parser.add_argument("--partition-components", action="store_true", default=True,
                        help="Partition by connected components for efficiency (default: True)")
    parser.add_argument("--no-partition", action="store_true",
                        help="Disable component partitioning")

    # Preprocessing
    parser.add_argument("--log-transform", action="store_true", default=True,
                        help="Apply log1p transformation (default: True)")
    parser.add_argument("--no-log-transform", action="store_true",
                        help="Disable log transformation")

    # Cross-modal RNA filtering (NEW)
    parser.add_argument("--rna-filter", type=Path, default=None,
                        help="RNA-seq data CSV for cross-modal filtering (optional)")
    parser.add_argument("--rna-annotation", type=Path, default=None,
                        help="Gene annotation CSV for numeric RNA indices (optional)")

    parser.set_defaults(func=run_discover)


def run_discover(args: argparse.Namespace) -> int:
    """Execute the discover command."""
    import logging
    from cliquefinder import BioMatrix
    from cliquefinder.io.loaders import load_matrix  # UPDATED: Use smart loader
    from cliquefinder.knowledge import ModuleDiscovery

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Handle flags
    if args.no_log_transform:
        args.log_transform = False
    if args.no_partition:
        args.partition_components = False

    print(f"\n{'='*70}")
    print("  De Novo Co-expression Module Discovery")
    print(f"{'='*70}\n")

    # Load expression data (using smart loader)
    logger.info(f"Loading: {args.input}")
    matrix = load_matrix(args.input)  # UPDATED: Auto-detects format

    # Load metadata if provided (optional override)
    if args.metadata and args.metadata.exists():
        metadata = pd.read_csv(args.metadata, index_col=0)
        # Update matrix with external metadata
        matrix = BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=metadata.loc[matrix.sample_ids],
            quality_flags=matrix.quality_flags
        )
        logger.info(f"Loaded metadata with columns: {list(metadata.columns)}")

    logger.info(f"Matrix: {matrix.n_features} genes x {matrix.n_samples} samples")

    # Log transform
    if args.log_transform:
        logger.info("Applying log1p transformation...")
        matrix = BioMatrix(
            data=np.log1p(matrix.data),
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags
        )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load RNA filter if provided (NEW - cross-modal integration)
    rna_genes = None
    if args.rna_filter and args.rna_filter.exists():
        from cliquefinder.knowledge.rna_loader import RNADataLoader
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        logger.info(f"Loading RNA filter: {args.rna_filter}")
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(
            rna_path=args.rna_filter,
            annotation_path=args.rna_annotation if args.rna_annotation else None
        )
        logger.info(f"RNA data: {rna_dataset.n_genes} genes, type={rna_dataset.id_type}")

        # Map IDs to common namespace
        mapper = CrossModalIDMapper()
        mapping = mapper.unify_ids(
            protein_ids=list(matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type=rna_dataset.id_type,
            species='human'
        )
        rna_genes = mapping.common_genes
        logger.info(
            f"Cross-modal mapping: {len(mapping.common_genes)} common genes "
            f"({len(mapping.protein_only)} proteomics-only, {len(mapping.rna_only)} RNA-only)"
        )

    # Initialize ModuleDiscovery
    logger.info("Initializing ModuleDiscovery...")
    discovery = ModuleDiscovery.from_matrix(
        matrix,
        stratify_by=args.stratify_by,
        min_samples=args.min_samples,
        rna_filter_genes=rna_genes,  # Pass RNA filter (NEW)
    )

    # Run de novo discovery
    logger.info(f"Starting de novo discovery: top {args.n_genes} variance genes, r >= {args.min_correlation}")
    modules = discovery.discover_de_novo(
        n_genes=args.n_genes,
        min_correlation=args.min_correlation,
        min_module_size=args.min_module_size,
        method=args.method,
        conditions=args.conditions,
        use_greedy=not args.exact_cliques,
        partition_by_components=args.partition_components,
    )

    logger.info(f"Found {len(modules)} co-expression modules")

    # Save results
    if modules:
        # Modules summary
        modules_df = pd.DataFrame([m.to_dict() for m in modules])
        modules_df.to_csv(args.output / "modules.csv", index=False)
        logger.info(f"Saved: {args.output / 'modules.csv'}")

        # Per-condition summary
        condition_summary = modules_df.groupby('condition').agg({
            'size': ['count', 'mean', 'max'],
            'mean_correlation': 'mean',
        }).round(3)
        condition_summary.columns = ['n_modules', 'mean_size', 'max_size', 'mean_correlation']
        condition_summary.to_csv(args.output / "conditions_summary.csv")
        logger.info(f"Saved: {args.output / 'conditions_summary.csv'}")

        # Print summary
        print(f"\n{'='*70}")
        print("  Discovery Results")
        print(f"{'='*70}")
        print(f"  Total modules found: {len(modules)}")
        print(f"  Conditions analyzed: {modules_df['condition'].nunique()}")
        print(f"\n  Size distribution:")
        print(f"    Min:  {modules_df['size'].min()} genes")
        print(f"    Mean: {modules_df['size'].mean():.1f} genes")
        print(f"    Max:  {modules_df['size'].max()} genes")
        print(f"\n  Correlation:")
        print(f"    Mean within-module: {modules_df['mean_correlation'].mean():.3f}")
        print(f"    Min edge:           {modules_df['min_correlation'].min():.3f}")

        # Top modules by size
        print(f"\n  Top 5 modules by size:")
        top5 = modules_df.nlargest(5, 'size')[['condition', 'size', 'mean_correlation', 'genes']]
        for i, row in top5.iterrows():
            genes_preview = row['genes'][:60] + '...' if len(row['genes']) > 60 else row['genes']
            print(f"    {row['condition']}: {row['size']} genes (r={row['mean_correlation']:.3f})")
            print(f"      {genes_preview}")
    else:
        print("\nNo modules found meeting criteria.")

    # Save run config
    config = {
        'timestamp': datetime.now().isoformat(),
        'paradigm': 'de_novo_discovery',
        'input': str(args.input),
        'n_genes': args.n_genes,
        'min_correlation': args.min_correlation,
        'min_module_size': args.min_module_size,
        'method': args.method,
        'stratify_by': args.stratify_by,
        'conditions': args.conditions,
        'log_transform': args.log_transform,
        'partition_components': args.partition_components,
        'exact_cliques': args.exact_cliques,
        'n_modules_found': len(modules),
        'rna_filter': str(args.rna_filter) if args.rna_filter else None,  # NEW
        'rna_annotation': str(args.rna_annotation) if args.rna_annotation else None,  # NEW
        'rna_filtered_genes': len(rna_genes) if rna_genes else None,  # NEW
    }
    with open(args.output / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")
    return 0
