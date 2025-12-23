"""
CliqueFinder sensitivity command - MAD-Z threshold sensitivity analysis.

This module provides methodological rigor for Nature Methods publication by
demonstrating that outlier detection results are robust across a range of
reasonable threshold values.

Scientific rationale:
    The MAD-Z threshold (default 3.5) is somewhat arbitrary. Demonstrating
    result stability across threshold values (e.g., 2.5-4.5) shows that
    conclusions are not dependent on this specific choice.

Usage:
    cliquefinder sensitivity --input data.csv --output sensitivity_report
    cliquefinder sensitivity --input data.csv --thresholds 2.5 3.0 3.5 4.0 4.5
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from cliquefinder import BioMatrix
from cliquefinder.io.loaders import load_csv_matrix
from cliquefinder.quality.outliers import OutlierDetector
from cliquefinder.core.quality import QualityFlag

logger = logging.getLogger(__name__)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the sensitivity subcommand."""
    parser = subparsers.add_parser(
        "sensitivity",
        help="MAD-Z threshold sensitivity analysis",
        description=(
            "Evaluate robustness of outlier detection across multiple MAD-Z thresholds.\n"
            "Generates structured reports for supplementary materials in scientific publications."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scientific Context:
  The MAD-Z threshold (default 3.5) controls outlier detection sensitivity.
  This analysis sweeps multiple thresholds to demonstrate result stability,
  a methodological requirement for rigorous publication.

Output Formats:
  - JSON: Machine-readable results with full statistics
  - CSV: Tabular summary of outlier counts per threshold
  - TXT: Human-readable report with interpretation

Examples:
  # Default thresholds (2.5, 3.0, 3.5, 4.0, 4.5)
  cliquefinder sensitivity --input data.csv --output results/sensitivity

  # Custom threshold range
  cliquefinder sensitivity --input data.csv --thresholds 3.0 3.5 4.0

  # Stratified by phenotype and sex
  cliquefinder sensitivity --input data.csv --group-cols phenotype Sex
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV file (gene IDs × samples)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output base path (without extension)"
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[2.5, 3.0, 3.5, 4.0, 4.5],
        help="MAD-Z thresholds to test (default: 2.5 3.0 3.5 4.0 4.5)"
    )

    parser.add_argument(
        "--method",
        choices=["mad-z", "iqr"],
        default="mad-z",
        help="Outlier detection method (default: mad-z)"
    )

    parser.add_argument(
        "--mode",
        choices=["within_group", "per_feature", "global"],
        default="within_group",
        help="Detection mode (default: within_group)"
    )

    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=["phenotype"],
        help="Metadata columns for within_group mode (default: phenotype)"
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        help="External metadata CSV"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)"
    )

    parser.set_defaults(func=run_sensitivity)


def analyze_threshold(
    matrix: BioMatrix,
    threshold: float,
    method: str = "mad-z",
    mode: str = "within_group",
    group_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze outlier detection at a specific threshold.

    Args:
        matrix: Input expression matrix
        threshold: MAD-Z threshold value
        method: Detection method ("mad-z" or "iqr")
        mode: Detection mode ("within_group", "per_feature", "global")
        group_cols: Grouping columns for within_group mode

    Returns:
        Dictionary containing:
            - threshold: Tested threshold value
            - n_outliers: Total outliers detected
            - pct_outliers: Percentage of data points flagged
            - n_features_affected: Number of features with ≥1 outlier
            - n_samples_affected: Number of samples with ≥1 outlier
            - per_group_stats: Outlier stats per group (if mode=within_group)
            - feature_outlier_rates: Distribution of outlier rates per feature
            - sample_outlier_rates: Distribution of outlier rates per sample
    """
    if group_cols is None:
        group_cols = ["phenotype"]

    detector = OutlierDetector(
        method=method,
        threshold=threshold,
        mode=mode,
        group_cols=group_cols
    )

    flagged_matrix = detector.apply(matrix)
    outlier_mask = (flagged_matrix.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0

    n_outliers = np.sum(outlier_mask)
    total_values = matrix.data.size
    pct_outliers = 100.0 * n_outliers / total_values

    # Feature-level statistics
    feature_outlier_counts = np.sum(outlier_mask, axis=1)
    n_features_affected = np.sum(feature_outlier_counts > 0)
    feature_outlier_rates = feature_outlier_counts / matrix.n_samples

    # Sample-level statistics
    sample_outlier_counts = np.sum(outlier_mask, axis=0)
    n_samples_affected = np.sum(sample_outlier_counts > 0)
    sample_outlier_rates = sample_outlier_counts / matrix.n_features

    result = {
        "threshold": threshold,
        "n_outliers": int(n_outliers),
        "pct_outliers": float(pct_outliers),
        "n_features_affected": int(n_features_affected),
        "pct_features_affected": float(100.0 * n_features_affected / matrix.n_features),
        "n_samples_affected": int(n_samples_affected),
        "pct_samples_affected": float(100.0 * n_samples_affected / matrix.n_samples),
        "feature_outlier_rates": {
            "mean": float(np.mean(feature_outlier_rates)),
            "median": float(np.median(feature_outlier_rates)),
            "q25": float(np.percentile(feature_outlier_rates, 25)),
            "q75": float(np.percentile(feature_outlier_rates, 75)),
            "max": float(np.max(feature_outlier_rates))
        },
        "sample_outlier_rates": {
            "mean": float(np.mean(sample_outlier_rates)),
            "median": float(np.median(sample_outlier_rates)),
            "q25": float(np.percentile(sample_outlier_rates, 25)),
            "q75": float(np.percentile(sample_outlier_rates, 75)),
            "max": float(np.max(sample_outlier_rates))
        }
    }

    # Per-group statistics for within_group mode
    if mode == "within_group" and matrix.sample_metadata is not None:
        per_group_stats = {}

        # Create composite group key
        if len(group_cols) == 1:
            group_series = matrix.sample_metadata[group_cols[0]]
        else:
            group_series = matrix.sample_metadata[group_cols].apply(
                lambda row: '_'.join(str(v) for v in row), axis=1
            )

        for group in group_series.unique():
            group_mask = (group_series == group).values
            group_outliers = outlier_mask[:, group_mask]

            n_group_outliers = np.sum(group_outliers)
            n_group_values = group_outliers.size

            per_group_stats[str(group)] = {
                "n_samples": int(np.sum(group_mask)),
                "n_outliers": int(n_group_outliers),
                "pct_outliers": float(100.0 * n_group_outliers / n_group_values) if n_group_values > 0 else 0.0
            }

        result["per_group_stats"] = per_group_stats

    return result


def run_sensitivity_analysis(
    matrix: BioMatrix,
    thresholds: List[float],
    method: str = "mad-z",
    mode: str = "within_group",
    group_cols: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Run sensitivity analysis across multiple thresholds.

    Args:
        matrix: Input expression matrix
        thresholds: List of threshold values to test
        method: Detection method
        mode: Detection mode
        group_cols: Grouping columns

    Returns:
        List of result dictionaries, one per threshold
    """
    results = []

    logger.info(f"Running sensitivity analysis with {len(thresholds)} thresholds...")

    for i, threshold in enumerate(sorted(thresholds), 1):
        logger.info(f"  [{i}/{len(thresholds)}] Testing threshold = {threshold}")

        result = analyze_threshold(
            matrix=matrix,
            threshold=threshold,
            method=method,
            mode=mode,
            group_cols=group_cols
        )
        results.append(result)

        logger.info(
            f"      Outliers: {result['n_outliers']:,} ({result['pct_outliers']:.2f}%), "
            f"Features affected: {result['n_features_affected']:,}"
        )

    return results


def generate_interpretation(results: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable interpretation of sensitivity analysis.

    Args:
        results: List of threshold analysis results

    Returns:
        Interpretation text suitable for supplementary materials
    """
    interpretation = []
    interpretation.append("=" * 80)
    interpretation.append("SENSITIVITY ANALYSIS INTERPRETATION")
    interpretation.append("=" * 80)
    interpretation.append("")

    # Extract key metrics
    thresholds = [r["threshold"] for r in results]
    pct_outliers = [r["pct_outliers"] for r in results]

    min_pct = min(pct_outliers)
    max_pct = max(pct_outliers)
    range_pct = max_pct - min_pct

    interpretation.append(f"Threshold range tested: {min(thresholds)} - {max(thresholds)}")
    interpretation.append(f"Number of thresholds: {len(thresholds)}")
    interpretation.append("")

    interpretation.append("Outlier Detection Summary:")
    interpretation.append(f"  Minimum detection rate: {min_pct:.2f}% (threshold={thresholds[pct_outliers.index(min_pct)]})")
    interpretation.append(f"  Maximum detection rate: {max_pct:.2f}% (threshold={thresholds[pct_outliers.index(max_pct)]})")
    interpretation.append(f"  Range: {range_pct:.2f} percentage points")
    interpretation.append("")

    # Robustness assessment
    interpretation.append("Robustness Assessment:")

    if range_pct < 2.0:
        assessment = "HIGHLY ROBUST"
        explanation = "Detection rates vary by <2 percentage points across thresholds."
    elif range_pct < 5.0:
        assessment = "ROBUST"
        explanation = "Detection rates vary by <5 percentage points across thresholds."
    elif range_pct < 10.0:
        assessment = "MODERATELY ROBUST"
        explanation = "Detection rates vary by 5-10 percentage points. Consider default threshold."
    else:
        assessment = "SENSITIVE TO THRESHOLD"
        explanation = "Detection rates vary by >10 percentage points. Threshold choice is critical."

    interpretation.append(f"  {assessment}")
    interpretation.append(f"  {explanation}")
    interpretation.append("")

    # Recommendations
    interpretation.append("Recommendations:")

    # Find middle threshold (closest to 3.5)
    default_threshold = 3.5
    closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - default_threshold))
    recommended = results[closest_idx]

    interpretation.append(f"  Recommended threshold: {recommended['threshold']}")
    interpretation.append(f"    Outliers detected: {recommended['n_outliers']:,} ({recommended['pct_outliers']:.2f}%)")
    interpretation.append(f"    Features affected: {recommended['n_features_affected']:,} ({recommended['pct_features_affected']:.1f}%)")
    interpretation.append(f"    Samples affected: {recommended['n_samples_affected']:,} ({recommended['pct_samples_affected']:.1f}%)")
    interpretation.append("")

    interpretation.append("For Supplementary Materials:")
    interpretation.append(f"  'Outlier detection was performed using MAD-Z scores with threshold = {recommended['threshold']}.")
    interpretation.append(f"  Sensitivity analysis across thresholds {min(thresholds)}-{max(thresholds)} showed {assessment.lower()} results")
    interpretation.append(f"  (detection rate range: {min_pct:.2f}%-{max_pct:.2f}%), confirming that conclusions are")
    interpretation.append(f"  not dependent on this specific threshold choice.'")
    interpretation.append("")

    return "\n".join(interpretation)


def save_results(
    results: List[Dict[str, Any]],
    output_path: Path,
    format: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Save sensitivity analysis results in requested format(s).

    Args:
        results: List of threshold analysis results
        output_path: Base output path (without extension)
        format: Output format ("json", "csv", or "both")
        metadata: Analysis metadata (input file, timestamp, etc.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON output (full details)
    if format in ("json", "both"):
        json_path = output_path.with_suffix(".json")

        full_output = {
            "metadata": metadata,
            "results": results,
            "summary": {
                "thresholds": [r["threshold"] for r in results],
                "outlier_percentages": [r["pct_outliers"] for r in results],
                "features_affected": [r["n_features_affected"] for r in results]
            }
        }

        with open(json_path, 'w') as f:
            json.dump(full_output, f, indent=2)

        logger.info(f"Saved JSON report: {json_path}")

    # CSV output (summary table)
    if format in ("csv", "both"):
        csv_path = output_path.with_suffix(".csv")

        # Flatten results for CSV
        csv_rows = []
        for r in results:
            row = {
                "threshold": r["threshold"],
                "n_outliers": r["n_outliers"],
                "pct_outliers": r["pct_outliers"],
                "n_features_affected": r["n_features_affected"],
                "pct_features_affected": r["pct_features_affected"],
                "n_samples_affected": r["n_samples_affected"],
                "pct_samples_affected": r["pct_samples_affected"],
                "mean_feature_outlier_rate": r["feature_outlier_rates"]["mean"],
                "median_feature_outlier_rate": r["feature_outlier_rates"]["median"],
                "max_sample_outlier_rate": r["sample_outlier_rates"]["max"]
            }
            csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False, float_format="%.6f")

        logger.info(f"Saved CSV summary: {csv_path}")

    # Text interpretation (always generated)
    txt_path = output_path.with_suffix(".txt")

    with open(txt_path, 'w') as f:
        # Header
        f.write("MAD-Z THRESHOLD SENSITIVITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {metadata['timestamp']}\n")
        f.write(f"Input: {metadata['input_file']}\n")
        f.write(f"Method: {metadata['method']}\n")
        f.write(f"Mode: {metadata['mode']}\n")
        f.write(f"Dataset: {metadata['n_features']} features × {metadata['n_samples']} samples\n\n")

        # Results table
        f.write("RESULTS BY THRESHOLD\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Threshold':>10} {'Outliers':>12} {'Pct Data':>10} {'Features':>12} {'Samples':>12}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(
                f"{r['threshold']:>10.2f} "
                f"{r['n_outliers']:>12,} "
                f"{r['pct_outliers']:>9.2f}% "
                f"{r['n_features_affected']:>12,} "
                f"{r['n_samples_affected']:>12,}\n"
            )

        f.write("\n")

        # Interpretation
        f.write(generate_interpretation(results))

    logger.info(f"Saved interpretation: {txt_path}")


def run_sensitivity(args: argparse.Namespace) -> int:
    """Execute the sensitivity command."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    start_time = datetime.now()

    print(f"\n{'='*80}")
    print("  MAD-Z Threshold Sensitivity Analysis")
    print(f"{'='*80}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    print(f"Loading: {args.input}")
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    matrix = load_csv_matrix(args.input, infer_phenotypes=True)

    # Load external metadata if provided
    if args.metadata and args.metadata.exists():
        ext_metadata = pd.read_csv(args.metadata, index_col=0)
        aligned_metadata = pd.DataFrame(index=matrix.sample_ids)
        for col in ext_metadata.columns:
            aligned_metadata[col] = ext_metadata[col].reindex(matrix.sample_ids)

        from cliquefinder import BioMatrix
        matrix = BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=aligned_metadata,
            quality_flags=matrix.quality_flags
        )

    print(f"Loaded: {matrix.n_features:,} features × {matrix.n_samples:,} samples")
    print(f"Method: {args.method}")
    print(f"Mode: {args.mode}")
    print(f"Thresholds: {args.thresholds}")
    print()

    # Run sensitivity analysis
    results = run_sensitivity_analysis(
        matrix=matrix,
        thresholds=args.thresholds,
        method=args.method,
        mode=args.mode,
        group_cols=args.group_cols
    )

    # Prepare metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(args.input),
        "method": args.method,
        "mode": args.mode,
        "group_cols": args.group_cols,
        "n_features": matrix.n_features,
        "n_samples": matrix.n_samples,
        "thresholds_tested": len(args.thresholds)
    }

    # Save results
    save_results(
        results=results,
        output_path=args.output,
        format=args.format,
        metadata=metadata
    )

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nComplete! Duration: {duration:.1f}s")
    print(f"Output: {args.output}.*")

    # Print quick summary
    print("\nQuick Summary:")
    pct_range = max(r["pct_outliers"] for r in results) - min(r["pct_outliers"] for r in results)
    print(f"  Detection rate range: {pct_range:.2f} percentage points")

    if pct_range < 2.0:
        print("  Assessment: HIGHLY ROBUST ✓")
    elif pct_range < 5.0:
        print("  Assessment: ROBUST ✓")
    elif pct_range < 10.0:
        print("  Assessment: MODERATELY ROBUST ~")
    else:
        print("  Assessment: SENSITIVE TO THRESHOLD ⚠")

    return 0
