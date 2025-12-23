"""
Visualization CLI subcommand.

Generates QC visualizations, network graphs, and HTML reports.

Usage:
    cliquefinder viz qc --input results/imputed --output figures/qc
    cliquefinder viz network --edges network.csv --output figures/network.html
    cliquefinder viz report --input results/ --output report.html
"""

import argparse
from pathlib import Path
import sys


def register_parser(subparsers):
    """Register viz subcommand and its subcommands."""
    parser = subparsers.add_parser(
        "viz",
        help="Generate visualizations and reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Generate visualizations for QC, networks, and analysis reports.

Modes:
  qc       Quality control plots (outliers, imputation, sex classification)
  network  Correlation network visualization (static or interactive)
  report   Full HTML report with embedded figures

Examples:
  cliquefinder viz qc --input results/imputed --output figures/qc
  cliquefinder viz network --edges results/network.csv --output network.html
  cliquefinder viz report --input results/ --output analysis_report.html
        """
    )

    subsubparsers = parser.add_subparsers(dest="viz_mode", help="Visualization mode")

    # QC mode
    qc_parser = subsubparsers.add_parser("qc", help="Quality control visualizations")
    qc_parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Imputation results directory (containing .data.csv and .metadata.csv)"
    )
    qc_parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for figures"
    )
    qc_parser.add_argument(
        "--format", "-f", choices=["png", "pdf", "svg"], default="png",
        help="Figure format (default: png)"
    )
    qc_parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for raster formats (default: 300)"
    )
    qc_parser.add_argument(
        "--style", choices=["paper", "presentation", "notebook"], default="paper",
        help="Visual style (default: paper)"
    )
    qc_parser.add_argument(
        "--original", type=Path,
        help="Original data file (before imputation) for comparison plots"
    )
    qc_parser.set_defaults(func=run_qc)

    # Network mode
    network_parser = subsubparsers.add_parser("network", help="Network visualizations")
    network_parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Expression data CSV (for computing correlations)"
    )
    network_parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output file (HTML for interactive, PNG/PDF for static)"
    )
    network_parser.add_argument(
        "--edges", type=Path,
        help="Pre-computed edge list CSV (columns: gene1, gene2, correlation)"
    )
    network_parser.add_argument(
        "--threshold", "-t", type=float, default=0.8,
        help="Correlation threshold for edges (default: 0.8)"
    )
    network_parser.add_argument(
        "--max-genes", type=int, default=500,
        help="Maximum genes to include (by variance, default: 500)"
    )
    network_parser.add_argument(
        "--max-edges", type=int, default=5000,
        help="Maximum edges to plot (default: 5000)"
    )
    network_parser.add_argument(
        "--layout", choices=["fr", "spring", "kamada_kawai", "circle"], default="fr",
        help="Layout algorithm (default: fr)"
    )
    network_parser.add_argument(
        "--static", action="store_true",
        help="Generate static matplotlib figure instead of interactive plotly"
    )
    network_parser.set_defaults(func=run_network)

    # Report mode
    report_parser = subsubparsers.add_parser("report", help="Full HTML report")
    report_parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Analysis results directory"
    )
    report_parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output HTML report file"
    )
    report_parser.add_argument(
        "--title", default="CliqueFinder Analysis Report",
        help="Report title"
    )
    report_parser.add_argument(
        "--style", choices=["paper", "presentation", "notebook"], default="paper",
        help="Visual style (default: paper)"
    )
    report_parser.set_defaults(func=run_report)

    parser.set_defaults(func=run_viz_help)


def run_viz_help(args):
    """Show viz help if no mode specified."""
    print("Error: must specify viz mode (qc, network, or report)")
    print("Run 'cliquefinder viz --help' for options")
    return 1


def run_qc(args):
    """Generate narrative-focused QC visualizations."""
    import numpy as np
    import pandas as pd
    from cliquefinder.io.loaders import load_csv_matrix, load_matrix
    from cliquefinder.viz import QCVisualizer, FigureCollection
    from cliquefinder.viz.styles import configure_style
    from cliquefinder.core.biomatrix import BioMatrix
    from cliquefinder.core.quality import QualityFlag

    print(f"Loading imputation results from {args.input}")

    # Find data files in directory or use file directly
    if args.input.is_dir():
        data_file = None
        for pattern in ["*.data.csv", "*_imputed.csv"]:
            matches = list(args.input.glob(pattern))
            if matches:
                data_file = matches[0]
                break

        metadata_files = list(args.input.glob("*.metadata.csv"))
        metadata_file = metadata_files[0] if metadata_files else None

        flags_files = list(args.input.glob("*.flags.csv"))
        flags_file = flags_files[0] if flags_files else None
    else:
        data_file = args.input
        metadata_file = None
        flags_file = None
        # Try to find sibling files
        stem = args.input.stem.replace('.data', '')
        parent = args.input.parent
        for pattern in [f"{stem}.metadata.csv", f"{stem}_metadata.csv"]:
            candidate = parent / pattern
            if candidate.exists():
                metadata_file = candidate
                break
        for pattern in [f"{stem}.flags.csv", f"{stem}_flags.csv"]:
            candidate = parent / pattern
            if candidate.exists():
                flags_file = candidate
                break

    if data_file is None:
        print(f"Error: No data file found")
        return 1

    print(f"  Data: {data_file}")
    if metadata_file:
        print(f"  Metadata: {metadata_file}")
    if flags_file:
        print(f"  Flags: {flags_file}")

    # Load data
    matrix = load_csv_matrix(data_file)

    # Load metadata
    if metadata_file:
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        matrix = BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=metadata_df,
            quality_flags=matrix.quality_flags
        )

    # Load flags
    if flags_file:
        flags_df = pd.read_csv(flags_file, index_col=0)
        flags_array = flags_df.values.astype(np.uint8)
        matrix = BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=flags_array
        )

    # Load original if provided
    matrix_before = None
    if args.original:
        print(f"  Original: {args.original}")
        # Use load_matrix for flexible format handling (TSV, proteomics formats, etc.)
        raw_original = load_matrix(args.original)

        # Align to imputed samples and features
        # The imputed data may have dropped samples/features during QC
        common_samples = matrix.sample_ids.intersection(raw_original.sample_ids)
        common_features = matrix.feature_ids.intersection(raw_original.feature_ids)

        print(f"    Aligning: {len(common_samples)} samples, {len(common_features)} features")

        if len(common_samples) < 10 or len(common_features) < 10:
            print(f"    Warning: Too few common elements. Skipping before/after comparison.")
            matrix_before = None
        else:
            # Extract aligned subsets
            sample_mask_orig = np.isin(raw_original.sample_ids, common_samples)
            feature_mask_orig = np.isin(raw_original.feature_ids, common_features)

            sample_mask_imputed = np.isin(matrix.sample_ids, common_samples)
            feature_mask_imputed = np.isin(matrix.feature_ids, common_features)

            # Get sample/feature order from imputed to maintain consistency
            sample_order = [s for s in matrix.sample_ids if s in common_samples]
            feature_order = [f for f in matrix.feature_ids if f in common_features]

            # Build aligned original data
            orig_sample_idx = {s: i for i, s in enumerate(raw_original.sample_ids)}
            orig_feature_idx = {f: i for i, f in enumerate(raw_original.feature_ids)}

            aligned_data = np.array([
                [raw_original.data[orig_feature_idx[f], orig_sample_idx[s]]
                 for s in sample_order]
                for f in feature_order
            ])

            # Get aligned metadata from imputed matrix
            aligned_metadata = matrix.sample_metadata.loc[sample_order] if matrix.sample_metadata is not None else None

            matrix_before = BioMatrix(
                data=aligned_data,
                feature_ids=pd.Index(feature_order),
                sample_ids=pd.Index(sample_order),
                sample_metadata=aligned_metadata,
                quality_flags=np.full(aligned_data.shape, QualityFlag.ORIGINAL, dtype=int)
            )

            # Also align the imputed matrix for comparison
            imputed_sample_idx = {s: i for i, s in enumerate(matrix.sample_ids)}
            imputed_feature_idx = {f: i for i, f in enumerate(matrix.feature_ids)}

            aligned_imputed_data = np.array([
                [matrix.data[imputed_feature_idx[f], imputed_sample_idx[s]]
                 for s in sample_order]
                for f in feature_order
            ])

            # Get aligned flags
            aligned_flags = np.array([
                [matrix.quality_flags[imputed_feature_idx[f], imputed_sample_idx[s]]
                 for s in sample_order]
                for f in feature_order
            ]) if matrix.quality_flags is not None else None

            matrix = BioMatrix(
                data=aligned_imputed_data,
                feature_ids=pd.Index(feature_order),
                sample_ids=pd.Index(sample_order),
                sample_metadata=aligned_metadata,
                quality_flags=aligned_flags
            )

    # Configure style
    palette = configure_style(style=args.style)
    viz = QCVisualizer(palette=palette, style=args.style)
    collection = FigureCollection()

    print(f"\nGenerating narrative QC figures...")

    # === OUTLIER IMPUTATION NARRATIVE ===
    if matrix_before is not None:
        print("\n--- Outlier Imputation Narrative ---")

        # 1. Summary card (4-panel overview)
        print("  - Outlier summary card...")
        fig = viz.plot_outlier_summary_card(matrix_before, matrix)
        collection.add("01_outlier_summary", fig)

        # 2. Distribution preservation by stratum
        print("  - Distribution preservation...")
        try:
            fig = viz.plot_distribution_preservation(matrix_before, matrix)
            collection.add("02_distribution_preservation", fig)
        except Exception as e:
            print(f"    Warning: {e}")

        # 3. Protein vulnerability
        print("  - Protein vulnerability...")
        if matrix.quality_flags is not None:
            outlier_mask = (matrix.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        else:
            outlier_mask = matrix_before.data != matrix.data
        fig = viz.plot_protein_vulnerability(outlier_mask, matrix.feature_ids)
        collection.add("03_protein_vulnerability", fig)

        # 4. Outlier distribution by stratum (data-driven, no arbitrary thresholds)
        print("  - Outlier distribution by stratum...")
        try:
            fig = viz.plot_outlier_distribution_by_stratum(matrix_before, matrix)
            collection.add("04_outlier_by_stratum", fig)
        except Exception as e:
            print(f"    Warning: {e}")

    # === SEX CLASSIFICATION NARRATIVE ===
    if matrix.sample_metadata is not None:
        sex_pred_col = None
        sex_gt_col = None

        for col in ["Sex_predicted"]:
            if col in matrix.sample_metadata.columns:
                sex_pred_col = col
                break

        for col in ["SEX", "sex", "Sex"]:
            if col in matrix.sample_metadata.columns:
                sex_gt_col = col
                break

        if sex_pred_col:
            print("\n--- Sex Classification Narrative ---")

            sex_labels = matrix.sample_metadata[sex_pred_col].values

            # Get ground truth and normalize to M/F format
            if sex_gt_col:
                raw_gt = matrix.sample_metadata[sex_gt_col].values
                # Normalize: "Male"/"Female" -> "M"/"F"
                ground_truth = np.array([
                    'M' if str(v).lower().startswith('m') else
                    'F' if str(v).lower().startswith('f') else
                    np.nan
                    for v in raw_gt
                ], dtype=object)
            else:
                ground_truth = np.array([np.nan] * len(sex_labels))

            # Top features from report
            top_features = ["P22314", "P51784", "O43602"]

            # 5. Summary card
            print("  - Sex classification summary...")
            fig = viz.plot_sex_summary_card(
                matrix, sex_labels, ground_truth, top_features
            )
            collection.add("05_sex_summary", fig)

            # 6. Discriminative signal
            print("  - Discriminative signal...")
            fig = viz.plot_sex_discriminative_signal(matrix, sex_labels, top_features)
            collection.add("06_sex_discriminative", fig)

            # 7. Imputed placement
            print("  - Imputed sample placement...")
            fig = viz.plot_imputed_sex_placement(
                matrix, sex_labels, ground_truth, top_features
            )
            collection.add("07_sex_imputed_placement", fig)

            # 8. Sex × phenotype interaction
            if "phenotype" in matrix.sample_metadata.columns and len(top_features) > 0:
                print("  - Sex × phenotype interaction...")
                try:
                    fig = viz.plot_sex_phenotype_interaction(
                        matrix, sex_labels, top_features[0]
                    )
                    collection.add("08_sex_phenotype_interaction", fig)
                except Exception as e:
                    print(f"    Warning: {e}")

    # Save all figures
    args.output.mkdir(parents=True, exist_ok=True)
    saved = collection.save_all(args.output, format=args.format, dpi=args.dpi)

    print(f"\n✓ Saved {len(saved)} figures to {args.output}")
    for path in saved:
        print(f"  - {path.name}")

    # Generate HTML report
    report_path = args.output / "qc_report.html"
    collection.to_html_report(
        report_path,
        title="QC Narrative Report",
        description="Aggregate visualizations answering: What intervention did we make? Did we preserve biology?"
    )
    print(f"  - qc_report.html")

    collection.close_all()
    return 0


def run_network(args):
    """Generate network visualization."""
    import numpy as np
    import pandas as pd
    from cliquefinder.io.loaders import load_csv_matrix, load_matrix
    from cliquefinder.viz.network import NetworkVisualizer

    print(f"Loading data from {args.input}")

    if args.edges:
        # Load pre-computed edges
        print(f"Using pre-computed edges from {args.edges}")
        edges_df = pd.read_csv(args.edges)

        # Get unique genes
        gene_ids = list(set(edges_df.iloc[:, 0].tolist() + edges_df.iloc[:, 1].tolist()))
        n_genes = len(gene_ids)

        # Build correlation matrix
        gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        corr_matrix = np.zeros((n_genes, n_genes))
        np.fill_diagonal(corr_matrix, 1.0)

        for _, row in edges_df.iterrows():
            g1, g2, corr = row.iloc[0], row.iloc[1], row.iloc[2]
            if g1 in gene_to_idx and g2 in gene_to_idx:
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    else:
        # Compute correlations from expression data
        matrix = load_csv_matrix(args.input)
        print(f"  {matrix.n_features} features × {matrix.n_samples} samples")

        # Select top genes by variance
        variances = np.var(matrix.data, axis=1)
        n_select = min(args.max_genes, matrix.n_features)
        top_idx = np.argsort(variances)[-n_select:]

        gene_ids = matrix.feature_ids[top_idx].tolist()
        data_subset = matrix.data[top_idx, :]

        print(f"  Computing correlations for top {n_select} genes by variance...")
        corr_matrix = np.corrcoef(data_subset)

    # Create visualizer
    viz = NetworkVisualizer(layout_algorithm=args.layout)

    # Generate plot
    print(f"Generating network (threshold={args.threshold}, max_edges={args.max_edges})...")

    interactive = not args.static
    fig = viz.plot_correlation_network(
        gene_ids=gene_ids,
        correlation_matrix=corr_matrix,
        correlation_threshold=args.threshold,
        max_edges=args.max_edges,
        interactive=interactive
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.save(args.output)

    print(f"\n✓ Network saved to {args.output}")
    print(f"  {fig.metadata.get('n_nodes', '?')} nodes, {fig.metadata.get('n_edges', '?')} edges")

    return 0


def run_report(args):
    """Generate full HTML report."""
    import pandas as pd
    from pathlib import Path
    from cliquefinder.io.loaders import load_csv_matrix, load_matrix
    from cliquefinder.viz import QCVisualizer, FigureCollection
    from cliquefinder.viz.styles import configure_style

    print(f"Generating report from {args.input}")

    # Configure style
    palette = configure_style(style=args.style)

    # Create collection
    viz = QCVisualizer(palette=palette, style=args.style)
    collection = FigureCollection()

    # Look for imputation results
    impute_dir = args.input / "imputed" if (args.input / "imputed").exists() else args.input

    data_files = list(impute_dir.glob("*.data.csv"))
    if data_files:
        print(f"  Found imputation data: {data_files[0].name}")
        matrix = load_csv_matrix(data_files[0])

        # Load metadata
        metadata_files = list(impute_dir.glob("*.metadata.csv"))
        if metadata_files:
            metadata_df = pd.read_csv(metadata_files[0], index_col=0)
            from cliquefinder.core.biomatrix import BioMatrix
            matrix = BioMatrix(
                data=matrix.data,
                feature_ids=matrix.feature_ids,
                sample_ids=matrix.sample_ids,
                sample_metadata=metadata_df,
                quality_flags=matrix.quality_flags
            )

        # Add QC figures
        print("  Adding QC visualizations...")
        collection.add("sample_correlation", viz.plot_sample_correlation(matrix, max_samples=80))

        if matrix.sample_metadata is not None and "phenotype" in matrix.sample_metadata.columns:
            collection.add("pca", viz.plot_pca(matrix, color_by="phenotype"))

    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    collection.to_html_report(
        args.output,
        title=args.title,
        description=f"Analysis results from {args.input}"
    )

    print(f"\n✓ Report saved to {args.output}")
    collection.close_all()

    return 0
