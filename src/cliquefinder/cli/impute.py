"""
CliqueFinder impute command - Outlier detection and imputation.

Usage:
    cliquefinder impute --input data.csv --output results/imputed
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from cliquefinder import BioMatrix
from cliquefinder.io.loaders import load_csv_matrix, load_matrix
from cliquefinder.io.formats import PRESETS
from cliquefinder.io.writers import write_csv_matrix, write_sample_metadata
from cliquefinder.quality.outliers import OutlierDetector, AdaptiveOutlierDetector, MultiPassOutlierDetector, KDEAdaptiveOutlierDetector
from cliquefinder.quality.imputation import Imputer
from cliquefinder.core.quality import QualityFlag


def _extract_phenotype_from_sample_id(sample_id: str) -> str | None:
    """
    Extract phenotype from sample ID prefix for unharmonized datasets.

    Handles common patterns:
    - CASE_... -> 'CASE'
    - CTRL_... -> 'CTRL'
    - CONTROL_... -> 'CTRL'

    This is a fallback for datasets where phenotype is encoded in sample ID
    rather than in external metadata. Treats CTRL-prefixed samples as controls.

    Parameters:
        sample_id: Sample identifier string

    Returns:
        Phenotype string ('CASE', 'CTRL') or None if not detected
    """
    if sample_id is None:
        return None

    sid_upper = str(sample_id).upper()

    # Check for common phenotype prefixes
    if sid_upper.startswith('CASE_') or sid_upper.startswith('CASE-'):
        return 'CASE'
    elif sid_upper.startswith('CTRL_') or sid_upper.startswith('CTRL-'):
        return 'CTRL'
    elif sid_upper.startswith('CONTROL_') or sid_upper.startswith('CONTROL-'):
        return 'CTRL'

    return None


def _enrich_with_clinical_metadata(
    matrix: BioMatrix,
    clinical_path: Path,
    clinical_id_col: str,
    phenotype_source_col: str,
    case_values: list[str],
    ctrl_values: list[str],
    phenotype_col: str = 'phenotype',
    sample_id_fallback: bool = False,
) -> BioMatrix:
    """
    Enrich matrix with clinical metadata and derive phenotype from clinical column.

    This function:
    1. Loads clinical metadata from CSV
    2. Extracts participant IDs from sample IDs using SubjectIdExtractor
    3. Joins clinical data to samples by participant ID
    4. Maps values from phenotype_source_col to CASE/CTRL:
       - case_values -> 'CASE'
       - ctrl_values -> 'CTRL'
       - Other values -> sample EXCLUDED (unless sample_id_fallback is True)
    5. Adds SEX column from clinical metadata if available

    Parameters:
        matrix: BioMatrix to enrich
        clinical_path: Path to clinical metadata CSV
        clinical_id_col: Column name for participant IDs in clinical data
        phenotype_source_col: Clinical column to derive phenotype from (e.g., SUBJECT_GROUP)
        case_values: Values that map to CASE phenotype (e.g., ['ALS'])
        ctrl_values: Values that map to CTRL phenotype (e.g., ['Healthy Control'])
        phenotype_col: Output column name for phenotype (default: 'phenotype')
        sample_id_fallback: If True, try extracting phenotype from sample ID prefix
            (CASE_/CTRL_) when clinical metadata is missing (default: False)

    Returns:
        BioMatrix with clinical metadata added and phenotype derived from clinical data.
        Only samples with valid phenotype mapping (CASE or CTRL) are retained.
    """
    import pandas as pd
    from cliquefinder.io.metadata import ClinicalMetadataEnricher, SubjectIdExtractor

    print(f"\nEnriching with clinical metadata: {clinical_path}")

    # Check if file exists
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical metadata not found: {clinical_path}")

    # Load clinical CSV to get all column names
    clinical_df = pd.read_csv(clinical_path)

    # Create SubjectIdExtractor for consistent ID extraction
    # The extractor uses pattern r'(NEU[A-Z0-9]+)' which extracts:
    # - CASE_NEUAA295HHE-9014-P_D3 -> NEUAA295HHE
    # - CASE-NEUAA295HHE -> NEUAA295HHE
    extractor = SubjectIdExtractor()

    # CRITICAL: Extract participant IDs from clinical data too
    # Clinical Participant_ID is like "CASE-NEUAA295HHE" but we extract "NEUAA295HHE"
    # to match against sample-derived IDs
    clinical_df['_extracted_id'] = clinical_df[clinical_id_col].apply(extractor.extract)
    n_extracted = clinical_df['_extracted_id'].notna().sum()
    print(f"  Extracted participant IDs from clinical data: {n_extracted}/{len(clinical_df)}")

    # Pass all columns explicitly to override the curated column list
    all_columns = [col for col in clinical_df.columns if col not in [clinical_id_col, '_extracted_id']]

    enricher = ClinicalMetadataEnricher(
        clinical_df=clinical_df,
        subject_col='_extracted_id',  # Use extracted ID for matching
        extractor=extractor,
        columns=all_columns,  # Use ALL columns from CSV
    )

    # Enrich matrix (left join - all samples preserved initially)
    enriched_matrix = enricher.enrich(matrix, drop_unmatched=False)

    # Report match statistics
    if enricher.summary:
        summary = enricher.summary
        print(f"  Clinical data match rate: {summary.n_matched}/{summary.n_samples_input} ({100*summary.match_rate:.1f}%)")
        print(f"  Added {len(summary.columns_added)} clinical columns")
        if summary.n_unmatched > 0:
            print(f"  WARNING: {summary.n_unmatched} samples have no clinical metadata")
            if summary.unmatched_subjects:
                print(f"  Example unmatched subjects: {summary.unmatched_subjects[:5]}")

    # Check if phenotype source column exists
    if phenotype_source_col not in enriched_matrix.sample_metadata.columns:
        raise ValueError(
            f"Phenotype source column '{phenotype_source_col}' not found in clinical metadata.\n"
            f"Available columns: {list(enriched_matrix.sample_metadata.columns)}"
        )

    # Derive phenotype from clinical column
    print(f"\nMapping phenotype from clinical column: {phenotype_source_col}")
    print(f"  CASE values: {case_values}")
    print(f"  CTRL values: {ctrl_values}")
    if sample_id_fallback:
        print(f"  Sample ID fallback: ENABLED (will extract from CASE_/CTRL_ prefixes if clinical missing)")

    clinical_values = enriched_matrix.sample_metadata[phenotype_source_col].values
    sample_ids = enriched_matrix.sample_ids
    phenotypes = []
    keep_mask = []
    n_fallback_case = 0
    n_fallback_ctrl = 0

    for i, val in enumerate(clinical_values):
        if pd.isna(val):
            # No clinical data for this sample - try fallback if enabled
            if sample_id_fallback:
                fallback_pheno = _extract_phenotype_from_sample_id(sample_ids[i])
                if fallback_pheno:
                    phenotypes.append(fallback_pheno)
                    keep_mask.append(True)
                    if fallback_pheno == 'CASE':
                        n_fallback_case += 1
                    else:
                        n_fallback_ctrl += 1
                    continue
            # No fallback or fallback failed
            phenotypes.append(None)
            keep_mask.append(False)
        elif val in case_values:
            phenotypes.append('CASE')
            keep_mask.append(True)
        elif val in ctrl_values:
            phenotypes.append('CTRL')
            keep_mask.append(True)
        else:
            # Value doesn't match either CASE or CTRL mapping
            phenotypes.append(None)
            keep_mask.append(False)

    # Report phenotype distribution BEFORE filtering
    n_case = sum(1 for p in phenotypes if p == 'CASE')
    n_ctrl = sum(1 for p in phenotypes if p == 'CTRL')
    n_excluded = sum(1 for p in phenotypes if p is None)

    print(f"  Phenotype mapping results:")
    print(f"    CASE: {n_case}" + (f" ({n_fallback_case} from sample ID fallback)" if n_fallback_case > 0 else ""))
    print(f"    CTRL: {n_ctrl}" + (f" ({n_fallback_ctrl} from sample ID fallback)" if n_fallback_ctrl > 0 else ""))
    print(f"    Excluded (no match or missing): {n_excluded}")

    if sample_id_fallback and (n_fallback_case + n_fallback_ctrl) > 0:
        print(f"  Sample ID fallback recovered: {n_fallback_case + n_fallback_ctrl} samples")

    # Show distribution of excluded values
    if n_excluded > 0:
        excluded_values = [val for val, keep in zip(clinical_values, keep_mask) if not keep]
        excluded_counts = pd.Series(excluded_values).value_counts()
        print(f"  Excluded value distribution:")
        for val, count in excluded_counts.items():
            val_str = f"'{val}'" if pd.notna(val) else "NaN (no clinical data)"
            print(f"    {val_str}: {count}")

    # Filter matrix to only keep samples with valid phenotype mapping
    keep_mask = np.array(keep_mask, dtype=bool)
    n_kept = keep_mask.sum()

    if n_kept == 0:
        raise ValueError(
            f"No samples matched CASE or CTRL values.\n"
            f"CASE values: {case_values}\n"
            f"CTRL values: {ctrl_values}\n"
            f"Available values in {phenotype_source_col}: {pd.Series(clinical_values).dropna().unique()}"
        )

    # Use BioMatrix.select_samples to properly filter
    filtered_matrix = enriched_matrix.select_samples(keep_mask)

    # Add phenotype column to filtered matrix
    filtered_phenotypes = [p for p, keep in zip(phenotypes, keep_mask) if keep]
    updated_metadata = filtered_matrix.sample_metadata.copy()
    updated_metadata[phenotype_col] = filtered_phenotypes

    print(f"\n  Final dataset after clinical filtering: {filtered_matrix.n_samples} samples")
    print(f"  Phenotype distribution: CASE={n_case}, CTRL={n_ctrl}")

    # Check if SEX column is available for sex classification
    if 'SEX' in updated_metadata.columns:
        sex_counts = updated_metadata['SEX'].value_counts()
        print(f"  SEX column available for classification:")
        for sex, count in sex_counts.items():
            print(f"    {sex}: {count}")

    return BioMatrix(
        data=filtered_matrix.data,
        feature_ids=filtered_matrix.feature_ids,
        sample_ids=filtered_matrix.sample_ids,
        sample_metadata=updated_metadata,
        quality_flags=filtered_matrix.quality_flags
    )


def _ensure_phenotype_metadata(matrix: BioMatrix, phenotype_col: str = 'phenotype') -> BioMatrix:
    """
    Ensure phenotype metadata exists, extracting from sample IDs if needed.

    For unharmonized datasets where phenotype is encoded in sample ID prefix
    (e.g., CASE_..., CTRL_...), this function populates the phenotype column
    in sample_metadata from the sample IDs.

    Parameters:
        matrix: BioMatrix to check/update
        phenotype_col: Name of phenotype column (default: 'phenotype')

    Returns:
        BioMatrix with phenotype metadata populated
    """
    import pandas as pd

    # Check if phenotype column already exists and has valid data
    if phenotype_col in matrix.sample_metadata.columns:
        existing_values = matrix.sample_metadata[phenotype_col].dropna()
        if len(existing_values) > 0:
            # Check if values are meaningful (not all NaN/empty)
            valid_count = sum(1 for v in existing_values if v and str(v).strip())
            if valid_count > 0:
                return matrix  # Already has phenotype data

    # Try to extract phenotype from sample IDs
    phenotypes = []
    extracted_count = 0
    for sid in matrix.sample_ids:
        pheno = _extract_phenotype_from_sample_id(sid)
        phenotypes.append(pheno)
        if pheno is not None:
            extracted_count += 1

    if extracted_count > 0:
        print(f"  Extracted phenotype from {extracted_count} sample IDs (CASE_/CTRL_ prefixes)")

        # Update metadata
        updated_metadata = matrix.sample_metadata.copy()
        updated_metadata[phenotype_col] = phenotypes

        # Count by phenotype
        pheno_counts = {}
        for p in phenotypes:
            if p is not None:
                pheno_counts[p] = pheno_counts.get(p, 0) + 1
        print(f"  Phenotype distribution: {pheno_counts}")

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=updated_metadata,
            quality_flags=matrix.quality_flags
        )

    return matrix


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the impute subcommand."""
    parser = subparsers.add_parser(
        "impute",
        help="Detect and impute outliers in expression data",
        description="Phase 1: Outlier detection and MAD-clip imputation for proteomic/transcriptomic data"
    )

    # Configuration file support
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Path to YAML/JSON config file (optional, CLI args override config values)")

    parser.add_argument("--input", "-i", type=Path, required=False,
                        help="Input data file (gene/protein IDs x samples)")
    parser.add_argument("--output", "-o", type=Path, required=False,
                        help="Output base path (without extension)")
    parser.add_argument("--format", "-f", choices=list(PRESETS.keys()) + ['auto'],
                        default='auto',
                        help="Input format preset (default: auto-detect)")
    parser.add_argument("--method", choices=["mad-z", "iqr", "adjusted-boxplot", "kde-adaptive"], default="mad-z",
                        help="Outlier detection method: mad-z (symmetric), iqr (symmetric), "
                             "adjusted-boxplot (asymmetric), kde-adaptive (finds optimal threshold per stratum). Default: mad-z")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Outlier detection threshold. For mad-z: MAD-Z score (default: 5.0). "
                             "For iqr/adjusted-boxplot: IQR whisker multiplier (default: 1.5)")
    parser.add_argument("--upper-threshold", type=float, default=None,
                        help="Upper tail threshold (default: same as --threshold). "
                             "Use higher values to be more lenient on upper tail.")
    parser.add_argument("--lower-threshold", type=float, default=None,
                        help="Lower tail threshold (default: same as --threshold). "
                             "Use lower values to be stricter on lower tail.")
    parser.add_argument("--upper-only", action="store_true", default=False,
                        help="Only detect upper-tail outliers (disable lower-tail detection). "
                             "Recommended for proteomics where low values are often real biology, "
                             "not technical artifacts.")
    parser.add_argument("--mode", choices=["within_group", "per_feature", "global"],
                        default="within_group", help="Detection mode (default: within_group)")
    parser.add_argument("--group-cols", nargs="+", default=["phenotype"],
                        help="Metadata columns for within_group mode")
    parser.add_argument("--metadata", type=Path, help="External metadata CSV")
    parser.add_argument("--scoring-method", choices=["binary", "student_t"], default="binary",
                        help="Scoring method: binary (classic threshold) or student_t "
                             "(probabilistic, better for heavy-tailed proteomics). Default: binary")
    parser.add_argument("--impute-strategy",
                        choices=["mad-clip", "median", "soft-clip", "knn"],
                        default="mad-clip",
                        help="Imputation strategy: mad-clip (hard threshold), median (group median), "
                             "soft-clip (tanh-based, preserves rank order), knn (k-nearest neighbors). Default: mad-clip")
    parser.add_argument("--soft-clip-sharpness", type=float, default=None,
                        help="Sharpness for soft-clip (auto-computed if not specified). "
                             "Higher values → closer to hard clipping")
    parser.add_argument("--clip-threshold", type=float, default=5.0,
                        help="MAD-Z threshold for clipping (default: 5.0, should match --threshold)")

    # Multi-pass detection options (Pass 2: Residual-based detection)
    parser.add_argument("--residual-threshold", type=float, default=4.25,
                        help="MAD-Z threshold for residual-based outlier detection in Pass 2 (default: 4.25)")
    parser.add_argument("--no-residual-pass", dest="residual_enabled", action="store_false", default=True,
                        help="Disable residual-based outlier detection (Pass 2)")
    parser.add_argument("--residual-high-end-only", action="store_true", default=True,
                        help="Only flag high-end residuals in Pass 2 (default: True)")
    parser.add_argument("--residual-both-tails", dest="residual_high_end_only", action="store_false",
                        help="Flag both high and low residuals in Pass 2")

    # Multi-pass detection options (Pass 3: Global percentile cap)
    parser.add_argument("--global-cap-percentile", type=float, default=99.95,
                        help="Global percentile threshold for Pass 3 capping (default: 99.95)")
    parser.add_argument("--no-global-cap", dest="global_cap_enabled", action="store_false", default=True,
                        help="Disable global percentile capping (Pass 3)")

    # Log transformation options
    parser.add_argument("--log-transform", action="store_true", default=True,
                        help="Apply log1p transformation BEFORE outlier detection (default: True)")
    parser.add_argument("--no-log-transform", dest="log_transform", action="store_false",
                        help="Disable log1p transformation (use raw values)")

    parser.add_argument("--no-quality-flags", action="store_true",
                        help="Don't write quality flags file")
    parser.add_argument("--gene-symbols", action="store_true",
                        help="Output with gene symbols instead of Ensembl IDs")
    parser.add_argument("--symbol-cache-dir", type=Path, default=None,
                        help="Cache directory for gene symbol mappings")

    # Phenotype filtering options
    parser.add_argument("--phenotype-filter", type=str, nargs="+", default=None,
                        help="Only process samples matching these phenotype values (e.g., --phenotype-filter CASE or --phenotype-filter CTRL)")
    parser.add_argument("--phenotype-col", type=str, default="phenotype",
                        help="Metadata column containing phenotype labels (default: phenotype)")

    # Sex classification options (data-driven approach)
    parser.add_argument("--classify-sex", action="store_true",
                        help="Classify biological sex from expression data (data-driven)")
    parser.add_argument("--sex-labels-col", type=str, default=None,
                        help="Metadata column with known sex labels for supervised classification")
    parser.add_argument("--sex-output-col", type=str, default="Sex_predicted",
                        help="Column name for predicted sex (default: Sex_predicted)")
    parser.add_argument("--sex-phenotype-filter", type=str, nargs="+", default=None,
                        help="Only IMPUTE sex for samples matching these phenotypes (training uses ALL labeled samples)")
    parser.add_argument("--sex-within-phenotype", action="store_true",
                        help="Train separate sex classifiers within each phenotype group (avoids disease confounding)")
    parser.add_argument("--sex-impute-only", action="store_true", default=True,
                        help="Only impute unknown sex labels; preserve ground truth (default: True)")
    parser.add_argument("--sex-overwrite", action="store_true", default=False,
                        help="Overwrite ALL sex labels with predictions (not recommended)")

    # Clinical metadata integration options
    parser.add_argument("--clinical-metadata", type=Path, default=None,
                        help="Path to clinical metadata CSV for phenotype mapping")
    parser.add_argument("--clinical-id-col", type=str, default="Participant_ID",
                        help="Column in clinical metadata containing participant IDs (default: Participant_ID)")
    parser.add_argument("--phenotype-source-col", type=str, default=None,
                        help="Clinical column to derive phenotype from (required if --clinical-metadata provided)")
    parser.add_argument("--case-values", type=str, nargs="+", default=None,
                        help="Values in phenotype-source-col that map to CASE (required if --phenotype-source-col provided)")
    parser.add_argument("--ctrl-values", type=str, nargs="+", default=None,
                        help="Values in phenotype-source-col that map to CTRL (required if --phenotype-source-col provided)")
    parser.add_argument("--sample-id-fallback", action="store_true", default=False,
                        help="For samples without clinical metadata, fall back to extracting phenotype from sample ID prefix (CASE_/CTRL_)")

    parser.set_defaults(func=run_impute)


def _classify_sex_unified(matrix, args, impute_mask, logger):
    """
    Unified sex classification - train on ALL labeled samples, impute only unknowns.

    Key behavior:
    1. Train on ALL samples with known sex labels (maximizes training data)
    2. Preserve ground truth labels (never overwrite known values)
    3. Only impute/predict for samples with UNKNOWN sex
    4. impute_mask further filters which unknowns to impute (e.g., by phenotype)

    Parameters:
        matrix: BioMatrix with expression data
        args: CLI arguments
        impute_mask: Boolean array - which samples are CANDIDATES for imputation
                     (combined with unknown status to determine final imputation set)
        logger: Logger instance
    """
    import pandas as pd
    from cliquefinder.quality.sex_imputation import (
        SupervisedSexClassifier,
        SemiSupervisedSexClassifier,
    )

    n_samples = matrix.n_samples
    results_dict = None

    try:
        # Get known labels
        known_sex = None
        if args.sex_labels_col and args.sex_labels_col in matrix.sample_metadata.columns:
            known_sex = matrix.sample_metadata[args.sex_labels_col].values
            print(f"  Using labels from: {args.sex_labels_col}")

            # Identify samples with KNOWN vs UNKNOWN sex
            known_mask = np.zeros(n_samples, dtype=bool)
            for i, s in enumerate(known_sex):
                if s is not None and str(s).upper() in ['M', 'F', 'MALE', 'FEMALE']:
                    known_mask[i] = True

            n_known = known_mask.sum()
            n_unknown = n_samples - n_known
            print(f"  Known sex labels: {n_known}")
            print(f"  Unknown sex labels: {n_unknown}")

            # Determine which samples to actually impute:
            # - Must have unknown sex (unless --sex-overwrite)
            # - Must pass the impute_mask filter (e.g., phenotype filter)
            if args.sex_overwrite:
                # Overwrite mode: impute all samples matching the mask
                to_impute_mask = impute_mask.copy()
                print(f"  WARNING: --sex-overwrite mode - will overwrite known labels")
            else:
                # Default: only impute unknowns that match the filter
                to_impute_mask = ~known_mask & impute_mask

            n_to_impute = to_impute_mask.sum()
            print(f"  Samples to impute: {n_to_impute}")

            if n_known >= 50:
                # Supervised classification - train on ALL known labels
                print("  Using SupervisedSexClassifier (sufficient labels)")
                print(f"  Training on ALL {n_known} labeled samples")
                classifier = SupervisedSexClassifier(min_training_samples=50)
                results = classifier.fit_predict(matrix, known_sex)

                # Build final predictions:
                # - Known samples: preserve ground truth
                # - Unknown samples matching filter: use predictions
                # - Others: mark as 'U' (unclassified)
                predictions = ['U'] * n_samples
                confidence = [0.0] * n_samples

                n_preserved = 0
                n_imputed = 0

                for i in range(n_samples):
                    if known_mask[i] and not args.sex_overwrite:
                        # Preserve ground truth
                        sex_val = str(known_sex[i]).upper()
                        if sex_val in ['M', 'MALE', '1']:
                            predictions[i] = 'M'
                        elif sex_val in ['F', 'FEMALE', '0']:
                            predictions[i] = 'F'
                        confidence[i] = 1.0  # Ground truth = 100% confidence
                        n_preserved += 1
                    elif to_impute_mask[i]:
                        # Impute this sample
                        predictions[i] = results.sex_labels[i]
                        confidence[i] = results.confidence[i]
                        n_imputed += 1
                    # else: leave as 'U' with confidence 0.0

                results_dict = {
                    'method': 'supervised',
                    'cv_accuracy': results.cv_accuracy,
                    'cv_auc': results.cv_auc,
                    'n_training_samples': results.n_training_samples,
                    'top_features': results.selected_features[:5] if results.selected_features else [],
                    'predictions': predictions,
                    'confidence': confidence,
                    'n_preserved': n_preserved,
                    'n_imputed': n_imputed,
                }

                male_count = sum(1 for s in predictions if s == 'M')
                female_count = sum(1 for s in predictions if s == 'F')
                high_conf = sum(1 for i, c in enumerate(confidence) if c >= 0.8 and to_impute_mask[i])

                print(f"  CV Accuracy: {results.cv_accuracy:.1%}")
                print(f"  Top features: {', '.join(results.selected_features[:3]) if results.selected_features else 'N/A'}")
                print(f"  Ground truth preserved: {n_preserved}")
                print(f"  Imputed: {n_imputed}")
                print(f"  Final counts: Male={male_count}, Female={female_count}")
                if n_imputed > 0:
                    print(f"  Imputed high confidence (>=0.8): {high_conf}/{n_imputed}")

                results_dict['male_count'] = male_count
                results_dict['female_count'] = female_count
                results_dict['high_confidence_count'] = high_conf

            elif n_known >= 10:
                # Semi-supervised classification
                print("  Using SemiSupervisedSexClassifier (partial labels)")
                print(f"  Training on ALL {n_known} labeled samples")

                labels = np.full(n_samples, -1)
                for i, s in enumerate(known_sex):
                    if s is not None:
                        s_upper = str(s).upper()
                        if s_upper in ['M', 'MALE', '1']:
                            labels[i] = 1
                        elif s_upper in ['F', 'FEMALE', '0']:
                            labels[i] = 0

                classifier = SemiSupervisedSexClassifier()
                results = classifier.fit_predict(matrix, labels)

                # Build final predictions with ground truth preservation
                predictions = ['U'] * n_samples
                confidence = [0.0] * n_samples
                n_preserved = 0
                n_imputed = 0

                for i in range(n_samples):
                    if known_mask[i] and not args.sex_overwrite:
                        sex_val = str(known_sex[i]).upper()
                        if sex_val in ['M', 'MALE', '1']:
                            predictions[i] = 'M'
                        elif sex_val in ['F', 'FEMALE', '0']:
                            predictions[i] = 'F'
                        confidence[i] = 1.0
                        n_preserved += 1
                    elif to_impute_mask[i]:
                        predictions[i] = results.sex_labels[i]
                        confidence[i] = results.confidence[i]
                        n_imputed += 1

                results_dict = {
                    'method': 'semi-supervised',
                    'discovered_marker': results.discovered_marker,
                    'cv_accuracy': results.cv_accuracy,
                    'cv_auc': results.cv_auc,
                    'n_labeled': results.n_labeled,
                    'label_fraction': results.label_fraction,
                    'predictions': predictions,
                    'confidence': confidence,
                    'n_preserved': n_preserved,
                    'n_imputed': n_imputed,
                }

                male_count = sum(1 for s in predictions if s == 'M')
                female_count = sum(1 for s in predictions if s == 'F')

                print(f"  Discovered marker: {results.discovered_marker}")
                print(f"  CV Accuracy: {results.cv_accuracy:.1%}")
                print(f"  Ground truth preserved: {n_preserved}")
                print(f"  Imputed: {n_imputed}")
                print(f"  Final counts: Male={male_count}, Female={female_count}")

                results_dict['male_count'] = male_count
                results_dict['female_count'] = female_count
            else:
                print(f"  WARNING: Only {n_known} labels - need at least 10 for classification")
        else:
            print("  ERROR: --sex-labels-col required for sex classification")
            print("  Use --sex-labels-col to specify a metadata column with 'M'/'F' labels.")

    except Exception as e:
        logger.error(f"Sex classification failed: {e}")
        print(f"  WARNING: Sex classification failed - {e}")

    return results_dict


def _classify_sex_within_phenotypes(matrix, args, logger):
    """
    Within-phenotype sex classification - train separate models per phenotype.

    This approach:
    1. Splits samples by phenotype (e.g., CASE vs CTRL)
    2. Trains separate sex classifiers within each group
    3. Avoids confounding from disease-related expression changes

    Scientific rationale:
    - Disease state affects protein expression
    - Training on mixed populations may learn disease features instead of sex features
    - CTRL samples often have better sex classification accuracy (97% vs 82% for CASE)
    - Separate models capture phenotype-specific sex signatures
    """
    import pandas as pd
    from cliquefinder.quality.sex_imputation import (
        SupervisedSexClassifier,
        SemiSupervisedSexClassifier,
    )

    n_samples = matrix.n_samples
    phenotype_col = args.phenotype_col

    # Initialize result arrays
    predictions = ['U'] * n_samples
    confidence = [0.0] * n_samples

    results_dict = {
        'method': 'within-phenotype',
        'phenotype_results': {},
        'predictions': predictions,
        'confidence': confidence,
    }

    if phenotype_col not in matrix.sample_metadata.columns:
        print(f"  ERROR: Phenotype column '{phenotype_col}' not found")
        print(f"  Within-phenotype classification requires phenotype metadata.")
        return None

    phenotype_values = matrix.sample_metadata[phenotype_col].values
    unique_phenotypes = np.unique([p for p in phenotype_values if pd.notna(p)])

    print(f"  Phenotypes found: {list(unique_phenotypes)}")

    # Check for known sex labels
    if not args.sex_labels_col or args.sex_labels_col not in matrix.sample_metadata.columns:
        print("  ERROR: --sex-labels-col required for within-phenotype classification")
        return None

    known_sex = matrix.sample_metadata[args.sex_labels_col].values

    total_male = 0
    total_female = 0
    total_classified = 0

    # Train separate classifier for each phenotype
    for pheno in unique_phenotypes:
        pheno_mask = phenotype_values == pheno
        n_pheno = pheno_mask.sum()

        print(f"\n  --- {pheno} ({n_pheno} samples) ---")

        # Get labels for this phenotype
        pheno_labels = []
        for i, (is_pheno, sex) in enumerate(zip(pheno_mask, known_sex)):
            if is_pheno and sex is not None and str(sex).upper() in ['M', 'F', 'MALE', 'FEMALE']:
                pheno_labels.append(sex)

        print(f"    Labeled samples: {len(pheno_labels)}")

        if len(pheno_labels) < 10:
            print(f"    SKIP: Insufficient labels (need >= 10)")
            continue

        # Create subset matrix for this phenotype
        pheno_indices = np.where(pheno_mask)[0]
        pheno_data = matrix.data[:, pheno_mask]
        pheno_sample_ids = pd.Index([matrix.sample_ids[i] for i in pheno_indices])
        pheno_metadata = matrix.sample_metadata.iloc[pheno_mask].copy()
        pheno_flags = matrix.quality_flags[:, pheno_mask] if matrix.quality_flags is not None else None

        from cliquefinder.core.biomatrix import BioMatrix
        pheno_matrix = BioMatrix(
            data=pheno_data,
            feature_ids=matrix.feature_ids,
            sample_ids=pheno_sample_ids,
            sample_metadata=pheno_metadata,
            quality_flags=pheno_flags
        )

        try:
            if len(pheno_labels) >= 50:
                # Supervised classification
                print(f"    Using SupervisedSexClassifier")
                classifier = SupervisedSexClassifier(min_training_samples=min(50, len(pheno_labels)))
                pheno_known_sex = known_sex[pheno_mask]
                results = classifier.fit_predict(pheno_matrix, pheno_known_sex)

                # Store results back to full arrays
                for local_i, global_i in enumerate(pheno_indices):
                    predictions[global_i] = results.sex_labels[local_i]
                    confidence[global_i] = results.confidence[local_i]

                male_count = sum(1 for s in results.sex_labels if s == 'M')
                female_count = sum(1 for s in results.sex_labels if s == 'F')

                print(f"    CV Accuracy: {results.cv_accuracy:.1%}")
                print(f"    Top features: {', '.join(results.selected_features[:3]) if results.selected_features else 'N/A'}")
                print(f"    Predicted: Male={male_count}, Female={female_count}")

                results_dict['phenotype_results'][pheno] = {
                    'cv_accuracy': results.cv_accuracy,
                    'cv_auc': results.cv_auc,
                    'n_samples': n_pheno,
                    'male_count': male_count,
                    'female_count': female_count,
                    'top_features': results.selected_features[:5] if results.selected_features else [],
                }

                total_male += male_count
                total_female += female_count
                total_classified += n_pheno

            else:
                # Semi-supervised for smaller groups
                print(f"    Using SemiSupervisedSexClassifier")

                labels = np.full(n_pheno, -1)
                pheno_known = known_sex[pheno_mask]
                for i, s in enumerate(pheno_known):
                    if s is not None:
                        s_upper = str(s).upper()
                        if s_upper in ['M', 'MALE', '1']:
                            labels[i] = 1
                        elif s_upper in ['F', 'FEMALE', '0']:
                            labels[i] = 0

                classifier = SemiSupervisedSexClassifier()
                results = classifier.fit_predict(pheno_matrix, labels)

                for local_i, global_i in enumerate(pheno_indices):
                    predictions[global_i] = results.sex_labels[local_i]
                    confidence[global_i] = results.confidence[local_i]

                male_count = sum(1 for s in results.sex_labels if s == 'M')
                female_count = sum(1 for s in results.sex_labels if s == 'F')

                print(f"    Discovered marker: {results.discovered_marker}")
                print(f"    CV Accuracy: {results.cv_accuracy:.1%}")
                print(f"    Predicted: Male={male_count}, Female={female_count}")

                results_dict['phenotype_results'][pheno] = {
                    'cv_accuracy': results.cv_accuracy,
                    'cv_auc': results.cv_auc,
                    'n_samples': n_pheno,
                    'male_count': male_count,
                    'female_count': female_count,
                    'discovered_marker': results.discovered_marker,
                }

                total_male += male_count
                total_female += female_count
                total_classified += n_pheno

        except Exception as e:
            logger.warning(f"Sex classification failed for {pheno}: {e}")
            print(f"    WARNING: Failed - {e}")

    # Update results dict with final counts
    results_dict['predictions'] = predictions
    results_dict['confidence'] = confidence
    results_dict['male_count'] = total_male
    results_dict['female_count'] = total_female
    results_dict['total_classified'] = total_classified

    # Summary
    print(f"\n  === Within-Phenotype Summary ===")
    print(f"  Total classified: {total_classified}")
    print(f"  Male: {total_male}, Female: {total_female}")

    if results_dict['phenotype_results']:
        avg_acc = np.mean([r['cv_accuracy'] for r in results_dict['phenotype_results'].values()])
        results_dict['cv_accuracy'] = avg_acc
        print(f"  Average CV accuracy: {avg_acc:.1%}")

    return results_dict


def run_impute(args: argparse.Namespace) -> int:
    """Execute the impute command."""
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load and merge config file if provided
    if args.config:
        from cliquefinder.cli.config import load_config, merge_config_with_args, validate_config

        print(f"Loading configuration from: {args.config}")
        try:
            config = load_config(args.config)
            validate_config(config)

            # Get raw CLI args to detect explicit overrides
            # Note: sys.argv includes script name, need to skip it
            cli_args = sys.argv[2:]  # Skip 'cliquefinder impute'

            # Merge config with CLI args (CLI overrides)
            args = merge_config_with_args(config, args, cli_args)
            print(f"  Configuration loaded successfully")
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: Config file error: {e}")
            return 1

    # Validate required arguments (after config merge)
    if not args.input:
        print("ERROR: --input is required (via CLI or config file)")
        return 1
    if not args.output:
        print("ERROR: --output is required (via CLI or config file)")
        return 1

    start_time = datetime.now()
    print(f"\n{'='*70}")
    print("  Phase 1: Outlier Detection and Imputation")
    print(f"{'='*70}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    print(f"Loading: {args.input}")
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Use format-aware loader
    format_spec = None if args.format == 'auto' else args.format
    matrix = load_matrix(args.input, format=format_spec)
    print(f"Format: {PRESETS[args.format].name if args.format != 'auto' else 'auto-detected'}")

    # Load external metadata if provided
    if args.metadata and args.metadata.exists():
        ext_metadata = pd.read_csv(args.metadata, index_col=0)
        aligned_metadata = pd.DataFrame(index=matrix.sample_ids)
        for col in ext_metadata.columns:
            aligned_metadata[col] = ext_metadata[col].reindex(matrix.sample_ids)
        matrix = BioMatrix(
            data=matrix.data, feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids, sample_metadata=aligned_metadata,
            quality_flags=matrix.quality_flags
        )

    print(f"Loaded: {matrix.n_features:,} features x {matrix.n_samples:,} samples")

    # Clinical metadata enrichment (if provided)
    # This REPLACES sample ID-derived phenotype with clinical-derived phenotype
    if args.clinical_metadata:
        matrix = _enrich_with_clinical_metadata(
            matrix,
            clinical_path=args.clinical_metadata,
            clinical_id_col=args.clinical_id_col,
            phenotype_source_col=args.phenotype_source_col,
            case_values=args.case_values,
            ctrl_values=args.ctrl_values,
            phenotype_col=args.phenotype_col,
            sample_id_fallback=args.sample_id_fallback,
        )
    else:
        # Fallback: Ensure phenotype metadata exists (extract from sample IDs if needed)
        # This handles unharmonized datasets where phenotype is encoded as CASE_/CTRL_ prefix
        matrix = _ensure_phenotype_metadata(matrix, args.phenotype_col)

    # Apply phenotype filter if specified
    original_n_samples = matrix.n_samples
    filtered_sample_ids = None
    if args.phenotype_filter:
        phenotype_col = args.phenotype_col
        if phenotype_col not in matrix.sample_metadata.columns:
            print(f"WARNING: Phenotype column '{phenotype_col}' not found in metadata")
            print(f"  Available columns: {list(matrix.sample_metadata.columns)}")
        else:
            # Get mask for samples matching the filter
            phenotype_values = matrix.sample_metadata[phenotype_col].values
            mask = np.isin(phenotype_values, args.phenotype_filter)
            n_matching = mask.sum()

            if n_matching == 0:
                print(f"ERROR: No samples match phenotype filter: {args.phenotype_filter}")
                print(f"  Available values: {np.unique(phenotype_values)}")
                return 1

            # Filter the matrix
            filtered_sample_ids = matrix.sample_ids[mask]
            filtered_metadata = matrix.sample_metadata.iloc[mask].copy()
            filtered_data = matrix.data[:, mask]
            filtered_flags = matrix.quality_flags[:, mask] if matrix.quality_flags is not None else None

            matrix = BioMatrix(
                data=filtered_data,
                feature_ids=matrix.feature_ids,
                sample_ids=pd.Index(filtered_sample_ids),
                sample_metadata=filtered_metadata,
                quality_flags=filtered_flags
            )

            print(f"Phenotype filter: {args.phenotype_filter}")
            print(f"  Filtered to {matrix.n_samples:,} samples (from {original_n_samples:,})")

    # Apply log transformation if requested (BEFORE outlier detection)
    if args.log_transform:
        print(f"\nApplying log1p transformation (pre-imputation)...")
        original_range = (matrix.data.min(), matrix.data.max())
        matrix = BioMatrix(
            data=np.log1p(matrix.data),
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags
        )
        transformed_range = (matrix.data.min(), matrix.data.max())
        print(f"  Original range: [{original_range[0]:.2f}, {original_range[1]:.2f}]")
        print(f"  Transformed range: [{transformed_range[0]:.2f}, {transformed_range[1]:.2f}]")
    else:
        print(f"\nLog transformation disabled (--no-log-transform)")

    # Determine if two-pass cleaning is needed (when sex classification is enabled)
    use_two_pass = args.classify_sex

    # Save original matrix for Pass 2 (if needed)
    matrix_original = matrix

    if use_two_pass:
        print(f"\n{'='*70}")
        print(f"  Two-Pass Outlier Detection/Imputation")
        print(f"  (Required for sex-stratified cleaning)")
        print(f"{'='*70}")
        print(f"\n=== PASS 1: Phenotype-only stratification ===")
        print(f"  Purpose: Clean data for sex classification")

    # Pass 1: Detect outliers (phenotype-only if two-pass, otherwise user-specified)
    pass1_group_cols = ["phenotype"] if use_two_pass else (args.group_cols if args.mode == "within_group" else None)

    # Determine detection threshold (adjust default for IQR-based methods)
    detection_threshold = args.threshold
    if args.method in ("iqr", "adjusted-boxplot") and args.threshold == 5.0:
        # User didn't override threshold, use IQR default (1.5)
        detection_threshold = 1.5
        print(f"\nNote: Using default IQR threshold (1.5) for {args.method} method")

    # Handle --upper-only flag: disable lower-tail detection
    if args.upper_only:
        args.lower_threshold = float('inf')  # Effectively disable lower tail
        print(f"\nUpper-tail only mode: Lower-tail detection disabled")
        print(f"  Rationale: Low values in proteomics are often real biology, not artifacts")

    print(f"\nDetecting outliers: method={args.method}, threshold={detection_threshold}")
    print(f"  Scoring: {args.scoring_method}")
    print(f"  Stratification: {pass1_group_cols}")

    # Determine if multi-pass detection is enabled
    use_multipass = args.residual_enabled or args.global_cap_enabled

    # Determine if adaptive detector should be used (for Pass 2 sex-stratified detection)
    use_adaptive = args.method == "adjusted-boxplot" or args.scoring_method == "student_t"

    # Use KDE Adaptive detector if specified (standalone, doesn't combine with multi-pass)
    if args.method == "kde-adaptive":
        print(f"\nUsing KDE Adaptive outlier detection:")
        print(f"  Finds optimal threshold per stratum based on distribution")

        # For KDE-adaptive, use 5.0 as default min_threshold (only extreme outliers)
        kde_min_threshold = 5.0

        detector = KDEAdaptiveOutlierDetector(
            group_cols=pass1_group_cols,
            upper_only=args.upper_only,
            min_threshold=kde_min_threshold,
            max_threshold=7.0,
        )
        matrix_with_outliers = detector.apply(matrix)

        # Report per-stratum thresholds
        print(f"\nAdaptive thresholds per stratum:")
        for stratum, threshold in detector.thresholds_.items():
            count = detector.stratum_counts_[stratum]
            print(f"  {stratum}: threshold={threshold:.2f}, outliers={count:,}")

    # Use MultiPassOutlierDetector if any pass is enabled beyond Pass 1
    elif use_multipass:
        print(f"\nUsing multi-pass outlier detection:")
        print(f"  Pass 1: Within-group {args.method} detection")
        if args.residual_enabled:
            print(f"  Pass 2: Residual-based detection (threshold={args.residual_threshold}, high-end-only={args.residual_high_end_only})")
        if args.global_cap_enabled:
            print(f"  Pass 3: Global cap at {args.global_cap_percentile}th percentile")

        detector = MultiPassOutlierDetector(
            detection_method=args.method,
            detection_threshold=detection_threshold,
            scoring_method=args.scoring_method,
            group_cols=pass1_group_cols,
            upper_threshold=args.upper_threshold,
            lower_threshold=args.lower_threshold,
            residual_enabled=args.residual_enabled,
            residual_threshold=args.residual_threshold,
            residual_high_end_only=args.residual_high_end_only,
            global_cap_enabled=args.global_cap_enabled,
            global_cap_percentile=args.global_cap_percentile,
        )
        matrix_with_outliers = detector.apply(matrix)

        # Report multi-pass diagnostics
        pass_summary = detector.get_pass_summary()
        print(f"\nMulti-pass detection results:")
        print(f"  Pass 1 (within-group): {pass_summary['pass1_within_group']:,} outliers")
        if args.residual_enabled:
            print(f"  Pass 2 (residual): +{pass_summary['pass2_residual']:,} additional outliers")
        if args.global_cap_enabled:
            print(f"  Pass 3 (global cap): +{pass_summary['pass3_global_cap']:,} additional outliers")
            if pass_summary['global_threshold'] is not None:
                print(f"  Global threshold: {pass_summary['global_threshold']:.4f}")

        # Report adjusted-boxplot diagnostics if applicable
        if args.method == "adjusted-boxplot" and detector.detector_.medcouples_ is not None:
            mean_mc = detector.detector_.medcouples_.mean()
            print(f"  Mean medcouple: {mean_mc:.4f} ({'right-skewed' if mean_mc > 0.1 else 'left-skewed' if mean_mc < -0.1 else 'symmetric'})")
    else:
        # Use AdaptiveOutlierDetector for new methods (adjusted-boxplot or student_t scoring)
        use_adaptive = args.method == "adjusted-boxplot" or args.scoring_method == "student_t"

        if use_adaptive:
            detector = AdaptiveOutlierDetector(
                method=args.method, threshold=detection_threshold,
                mode=args.mode, group_cols=pass1_group_cols,
                scoring_method=args.scoring_method,
                upper_threshold=args.upper_threshold,
                lower_threshold=args.lower_threshold
            )
            matrix_with_outliers = detector.apply(matrix)

            # Report adaptive-specific diagnostics
            if args.method == "adjusted-boxplot" and detector.medcouples_ is not None:
                mean_mc = detector.medcouples_.mean()
                print(f"  Mean medcouple: {mean_mc:.4f} ({'right-skewed' if mean_mc > 0.1 else 'left-skewed' if mean_mc < -0.1 else 'symmetric'})")
            if args.scoring_method == "student_t" and detector.df_shared_ is not None:
                print(f"  Fitted degrees of freedom: {detector.df_shared_:.2f}")
        else:
            detector = OutlierDetector(
                method=args.method, threshold=detection_threshold,
                mode=args.mode, group_cols=pass1_group_cols,
                upper_threshold=args.upper_threshold,
                lower_threshold=args.lower_threshold
            )
            matrix_with_outliers = detector.apply(matrix)

    outlier_mask = (matrix_with_outliers.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
    n_outliers = np.sum(outlier_mask)
    print(f"Detected: {n_outliers:,} outliers ({100*n_outliers/matrix.data.size:.2f}%)")

    # Pass 1: Impute outliers (phenotype-only if two-pass)
    print(f"\nImputing: strategy={args.impute_strategy}")
    print(f"  Stratification: {pass1_group_cols}")

    # Build imputer kwargs based on strategy
    imputer_kwargs = {
        "strategy": args.impute_strategy,
        "group_cols": pass1_group_cols,
    }

    if args.impute_strategy == "mad-clip":
        imputer_kwargs["threshold"] = args.clip_threshold
    elif args.impute_strategy == "soft-clip":
        # CRITICAL: Pass detection threshold so soft-clip bounds match detection bounds
        imputer_kwargs["threshold"] = args.threshold  # Use detection threshold, not clip_threshold
        if args.soft_clip_sharpness:
            imputer_kwargs["sharpness"] = args.soft_clip_sharpness
            print(f"  Sharpness: {args.soft_clip_sharpness}")
        else:
            print(f"  Sharpness: auto-computed from data bounds")
        print(f"  Threshold: {args.threshold} (matching detection)")
    elif args.impute_strategy == "knn":
        print(f"  Using k-nearest neighbors imputation")

    # Enforce global cap as max upper bound for imputation
    if use_multipass and args.global_cap_enabled and pass_summary.get('global_threshold') is not None:
        imputer_kwargs["max_upper_bound"] = pass_summary['global_threshold']
        print(f"  Max upper bound: {pass_summary['global_threshold']:.4f} (from global cap)")

    imputer = Imputer(**imputer_kwargs)
    matrix_imputed = imputer.apply(matrix_with_outliers)

    n_imputed = np.sum((matrix_imputed.quality_flags & QualityFlag.IMPUTED) > 0)
    print(f"Imputed: {n_imputed:,} values")

    # Track Pass 1 statistics for report
    pass1_stats = {
        'n_outliers': n_outliers,
        'n_imputed': n_imputed,
        'group_cols': pass1_group_cols,
    }

    # Convert to gene symbols if requested
    symbol_conversion_info = None
    if args.gene_symbols:
        from cliquefinder.cli._analyze_core import (
            build_gene_symbol_mapping,
            create_symbol_indexed_matrix,
        )

        print(f"\nConverting Ensembl IDs to gene symbols...")
        cache_dir = args.symbol_cache_dir or args.output.parent / ".symbol_cache"

        ensembl_to_symbol = build_gene_symbol_mapping(
            list(matrix_imputed.feature_ids),
            cache_dir=cache_dir
        )

        matrix_imputed, symbol_to_ensembl = create_symbol_indexed_matrix(
            matrix_imputed, ensembl_to_symbol
        )

        n_mapped = len([s for s in matrix_imputed.feature_ids if not s.startswith("ENSG")])
        n_unmapped = matrix_imputed.n_features - n_mapped
        print(f"Mapped: {n_mapped:,} genes to symbols, {n_unmapped:,} retained as Ensembl IDs")

        symbol_conversion_info = {
            'n_mapped': n_mapped,
            'n_unmapped': n_unmapped,
            'symbol_to_ensembl': symbol_to_ensembl
        }

    # Sex classification (optional, data-driven)
    sex_classification_info = None
    if args.classify_sex:
        from cliquefinder.quality.sex_imputation import (
            SupervisedSexClassifier,
            SemiSupervisedSexClassifier,
        )

        print(f"\nClassifying biological sex (data-driven approach)")
        print(f"  Strategy: Train on ALL labeled samples, impute only unknowns")

        # Determine which samples are CANDIDATES for imputation
        # (will be further filtered to only unknown sex samples)
        impute_candidate_mask = np.ones(matrix_imputed.n_samples, dtype=bool)  # Default: all samples
        if args.sex_phenotype_filter:
            phenotype_col = args.phenotype_col
            if phenotype_col in matrix_imputed.sample_metadata.columns:
                phenotype_values = matrix_imputed.sample_metadata[phenotype_col].values
                impute_candidate_mask = np.isin(phenotype_values, args.sex_phenotype_filter)
                n_candidates = impute_candidate_mask.sum()
                print(f"  Phenotype filter for imputation: {args.sex_phenotype_filter}")
                print(f"  Candidate samples: {n_candidates} of {matrix_imputed.n_samples}")
            else:
                print(f"  WARNING: Phenotype column '{phenotype_col}' not found, considering all samples")

        # Decide between within-phenotype vs unified classification
        if args.sex_within_phenotype:
            print(f"  Mode: Within-phenotype classification (avoids disease confounding)")
            sex_classification_info = _classify_sex_within_phenotypes(
                matrix_imputed, args, logger
            )
        else:
            print(f"  Mode: Unified classification (train on all, impute unknowns)")
            sex_classification_info = _classify_sex_unified(
                matrix_imputed, args, impute_candidate_mask, logger
            )

        # Update matrix with predictions if successful
        if sex_classification_info and 'predictions' in sex_classification_info:
            updated_metadata = matrix_imputed.sample_metadata.copy()
            updated_metadata[args.sex_output_col] = sex_classification_info['predictions']
            updated_metadata[f"{args.sex_output_col}_confidence"] = sex_classification_info['confidence']

            matrix_imputed = BioMatrix(
                data=matrix_imputed.data,
                feature_ids=matrix_imputed.feature_ids,
                sample_ids=matrix_imputed.sample_ids,
                sample_metadata=updated_metadata,
                quality_flags=matrix_imputed.quality_flags
            )

            # === PASS 2: Re-clean with [phenotype, sex] stratification ===
            if use_two_pass:
                print(f"\n=== PASS 2: [Phenotype, Sex] stratification ===")
                print(f"  Purpose: Final cleaning with sex-aware stratification")

                pass2_group_cols = ["phenotype", args.sex_output_col]

                # Reset to ORIGINAL data but with sex labels added to metadata
                original_metadata_with_sex = matrix_original.sample_metadata.copy()
                original_metadata_with_sex[args.sex_output_col] = sex_classification_info['predictions']
                original_metadata_with_sex[f"{args.sex_output_col}_confidence"] = sex_classification_info['confidence']

                matrix_for_pass2 = BioMatrix(
                    data=matrix_original.data,  # ORIGINAL data (before any cleaning)
                    feature_ids=matrix_original.feature_ids,
                    sample_ids=matrix_original.sample_ids,
                    sample_metadata=original_metadata_with_sex,
                    quality_flags=matrix_original.quality_flags.copy()  # Reset flags
                )

                # Pass 2: Detect outliers with [phenotype, sex] stratification
                print(f"\nDetecting outliers: method={args.method}, threshold={detection_threshold}")
                print(f"  Scoring: {args.scoring_method}")
                print(f"  Stratification: {pass2_group_cols}")

                # Use same detection method as Pass 1 for consistency
                if args.method == "kde-adaptive":
                    print(f"\nUsing KDE Adaptive outlier detection (Pass 2):")
                    kde_min_threshold = 5.0
                    detector_pass2 = KDEAdaptiveOutlierDetector(
                        group_cols=pass2_group_cols,
                        upper_only=args.upper_only,
                        min_threshold=kde_min_threshold,
                        max_threshold=7.0,
                    )
                elif use_multipass:
                    print(f"\nUsing multi-pass outlier detection (same as Pass 1):")
                    print(f"  Pass 1: Within-group {args.method} detection")
                    if args.residual_enabled:
                        print(f"  Pass 2: Residual-based detection (threshold={args.residual_threshold}, high-end-only={args.residual_high_end_only})")
                    if args.global_cap_enabled:
                        print(f"  Pass 3: Global cap at {args.global_cap_percentile}th percentile")

                    detector_pass2 = MultiPassOutlierDetector(
                        detection_method=args.method,
                        detection_threshold=detection_threshold,
                        group_cols=pass2_group_cols,
                        upper_threshold=args.upper_threshold,
                        lower_threshold=args.lower_threshold,
                        residual_enabled=args.residual_enabled,
                        residual_threshold=args.residual_threshold,
                        residual_high_end_only=args.residual_high_end_only,
                        global_cap_enabled=args.global_cap_enabled,
                        global_cap_percentile=args.global_cap_percentile,
                    )
                elif use_adaptive:
                    detector_pass2 = AdaptiveOutlierDetector(
                        method=args.method, threshold=detection_threshold,
                        mode=args.mode, group_cols=pass2_group_cols,
                        scoring_method=args.scoring_method,
                        upper_threshold=args.upper_threshold,
                        lower_threshold=args.lower_threshold
                    )
                else:
                    detector_pass2 = OutlierDetector(
                        method=args.method, threshold=detection_threshold,
                        mode=args.mode, group_cols=pass2_group_cols,
                        upper_threshold=args.upper_threshold,
                        lower_threshold=args.lower_threshold
                    )
                matrix_with_outliers_pass2 = detector_pass2.apply(matrix_for_pass2)

                outlier_mask_pass2 = (matrix_with_outliers_pass2.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
                n_outliers_pass2 = np.sum(outlier_mask_pass2)
                print(f"Detected: {n_outliers_pass2:,} outliers ({100*n_outliers_pass2/matrix_original.data.size:.2f}%)")

                # Pass 2: Impute outliers with [phenotype, sex] stratification
                print(f"\nImputing: strategy={args.impute_strategy}")
                print(f"  Stratification: {pass2_group_cols}")

                # Build imputer kwargs for pass2
                imputer_kwargs_pass2 = {
                    "strategy": args.impute_strategy,
                    "group_cols": pass2_group_cols,
                }
                if args.impute_strategy == "mad-clip":
                    imputer_kwargs_pass2["threshold"] = args.clip_threshold
                elif args.impute_strategy == "soft-clip" and args.soft_clip_sharpness:
                    imputer_kwargs_pass2["sharpness"] = args.soft_clip_sharpness

                # Enforce global cap as max upper bound for imputation (Pass 2)
                if use_multipass and args.global_cap_enabled:
                    pass2_summary = detector_pass2.get_pass_summary()
                    if pass2_summary.get('global_threshold') is not None:
                        imputer_kwargs_pass2["max_upper_bound"] = pass2_summary['global_threshold']
                        print(f"  Max upper bound: {pass2_summary['global_threshold']:.4f} (from global cap)")

                imputer_pass2 = Imputer(**imputer_kwargs_pass2)
                matrix_imputed = imputer_pass2.apply(matrix_with_outliers_pass2)

                n_imputed_pass2 = np.sum((matrix_imputed.quality_flags & QualityFlag.IMPUTED) > 0)
                print(f"Imputed: {n_imputed_pass2:,} values")

                # Update pass1_stats with pass2 info for report
                pass1_stats['pass2'] = {
                    'n_outliers': n_outliers_pass2,
                    'n_imputed': n_imputed_pass2,
                    'group_cols': pass2_group_cols,
                }

                print(f"\n  Pass 2 complete: Final data uses {pass2_group_cols} stratification")

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_matrix(matrix_imputed, args.output, write_quality_flags=not args.no_quality_flags)
    write_sample_metadata(matrix_imputed, Path(str(args.output) + ".metadata.csv"))

    # Write symbol-to-ensembl mapping if gene symbols were used
    import json
    if symbol_conversion_info:
        mapping_path = Path(str(args.output) + ".symbol_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(symbol_conversion_info['symbol_to_ensembl'], f, indent=2)
        print(f"Symbol mapping: {mapping_path}")

    # Write preprocessing parameters for downstream steps
    params_path = Path(str(args.output) + ".params.json")

    # For kde-adaptive, record the actual min_threshold used
    effective_threshold = 5.0 if args.method == "kde-adaptive" else args.threshold

    preprocessing_params = {
        'timestamp': datetime.now().isoformat(),
        'input': str(args.input),
        'output': str(args.output),
        'is_log_transformed': args.log_transform,
        'log_transform_method': 'log1p' if args.log_transform else None,
        'outlier_detection': {
            'method': args.method,
            'threshold': effective_threshold,
            'upper_only': args.upper_only,
            'upper_threshold': args.upper_threshold,
            'lower_threshold': None if args.upper_only else args.lower_threshold,
            'mode': args.mode,
            'group_cols': args.group_cols if args.mode == 'within_group' else None,
            'multi_pass_enabled': use_multipass,
            'residual_enabled': args.residual_enabled if use_multipass else False,
            'residual_threshold': args.residual_threshold if use_multipass and args.residual_enabled else None,
            'residual_high_end_only': args.residual_high_end_only if use_multipass and args.residual_enabled else None,
            'global_cap_enabled': args.global_cap_enabled if use_multipass else False,
            'global_cap_percentile': args.global_cap_percentile if use_multipass and args.global_cap_enabled else None,
        },
        'imputation': {
            'strategy': args.impute_strategy,
            'threshold': args.clip_threshold if args.impute_strategy == 'mad-clip' else None,
        },
        'n_features': matrix_imputed.n_features,
        'n_samples': matrix_imputed.n_samples,
    }
    with open(params_path, 'w') as f:
        json.dump(preprocessing_params, f, indent=2)
    print(f"Preprocessing parameters: {params_path}")

    # Write report
    report_path = Path(str(args.output) + ".report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Phase 1: Outlier Detection and Imputation Report\n{'='*70}\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input: {args.input}\nOutput: {args.output}\n\n")

        # Log transformation info
        f.write(f"Log Transformation:\n")
        f.write(f"  Applied: {'Yes' if args.log_transform else 'No'}\n")
        f.write(f"  Method: {'log1p' if args.log_transform else 'None'}\n")
        f.write(f"  Timing: Pre-imputation (BEFORE outlier detection)\n\n")

        # Clinical metadata enrichment info
        if args.clinical_metadata:
            f.write(f"Clinical Metadata Integration:\n")
            f.write(f"  Source: {args.clinical_metadata}\n")
            f.write(f"  Participant ID column: {args.clinical_id_col}\n")
            f.write(f"  Phenotype source: {args.phenotype_source_col}\n")
            f.write(f"  CASE mapping: {', '.join(args.case_values)}\n")
            f.write(f"  CTRL mapping: {', '.join(args.ctrl_values)}\n")
            f.write(f"  Sample ID fallback: {'Yes' if args.sample_id_fallback else 'No'}\n\n")

        f.write(f"Dataset: {matrix_original.n_features:,} features x {matrix_original.n_samples:,} samples\n")

        # Report multi-pass outlier detection if used
        if use_multipass:
            f.write(f"\nMulti-Pass Outlier Detection:\n")
            pass_summary = detector.get_pass_summary()
            f.write(f"  Pass 1 (within-group {args.method}): {pass_summary['pass1_within_group']:,} outliers\n")
            if args.residual_enabled:
                f.write(f"  Pass 2 (residual-based): +{pass_summary['pass2_residual']:,} additional outliers\n")
                f.write(f"    Threshold: {args.residual_threshold}\n")
                f.write(f"    High-end only: {args.residual_high_end_only}\n")
            if args.global_cap_enabled:
                f.write(f"  Pass 3 (global percentile cap): +{pass_summary['pass3_global_cap']:,} additional outliers\n")
                f.write(f"    Percentile: {args.global_cap_percentile}\n")
                if pass_summary['global_threshold'] is not None:
                    f.write(f"    Threshold value: {pass_summary['global_threshold']:.4f}\n")
            f.write(f"  Total outliers: {pass_summary['total']:,} ({100*pass_summary['total']/matrix_original.data.size:.2f}%)\n")

        # Report outlier detection/imputation stats (two-pass aware)
        if use_two_pass and 'pass2' in pass1_stats:
            f.write(f"\nTwo-Pass Outlier Detection/Imputation:\n")
            f.write(f"  Pass 1 (phenotype-only stratification):\n")
            f.write(f"    Stratification: {pass1_stats['group_cols']}\n")
            f.write(f"    Outliers detected: {pass1_stats['n_outliers']:,} ({100*pass1_stats['n_outliers']/matrix_original.data.size:.2f}%)\n")
            f.write(f"    Values imputed: {pass1_stats['n_imputed']:,}\n")
            f.write(f"  Pass 2 ([phenotype, sex] stratification):\n")
            f.write(f"    Stratification: {pass1_stats['pass2']['group_cols']}\n")
            f.write(f"    Outliers detected: {pass1_stats['pass2']['n_outliers']:,} ({100*pass1_stats['pass2']['n_outliers']/matrix_original.data.size:.2f}%)\n")
            f.write(f"    Values imputed: {pass1_stats['pass2']['n_imputed']:,}\n")
            f.write(f"  Final output uses Pass 2 (sex-stratified) cleaning\n")
        else:
            f.write(f"Outliers detected: {pass1_stats['n_outliers']:,} ({100*pass1_stats['n_outliers']/matrix_original.data.size:.2f}%)\n")
            f.write(f"Values imputed: {pass1_stats['n_imputed']:,}\n")
            f.write(f"Stratification: {pass1_stats['group_cols']}\n")

        f.write(f"Imputation method: {args.impute_strategy}\n")
        if args.impute_strategy == "mad-clip":
            f.write(f"  Clipping threshold: {args.clip_threshold}\n")
        if symbol_conversion_info:
            f.write(f"\nGene Symbol Conversion:\n")
            f.write(f"  Mapped to symbols: {symbol_conversion_info['n_mapped']:,}\n")
            f.write(f"  Retained as Ensembl: {symbol_conversion_info['n_unmapped']:,}\n")
        if sex_classification_info:
            f.write(f"\nSex Classification (data-driven):\n")
            f.write(f"  Method: {sex_classification_info['method']}\n")
            f.write(f"  Strategy: Train on ALL labeled, impute only unknowns\n")
            if 'n_preserved' in sex_classification_info:
                f.write(f"  Ground truth preserved: {sex_classification_info['n_preserved']}\n")
            if 'n_imputed' in sex_classification_info:
                f.write(f"  Imputed: {sex_classification_info['n_imputed']}\n")
            f.write(f"  Final counts - Male: {sex_classification_info.get('male_count', 'N/A')}\n")
            f.write(f"  Final counts - Female: {sex_classification_info.get('female_count', 'N/A')}\n")
            if 'high_confidence_count' in sex_classification_info:
                f.write(f"  Imputed high confidence (>=0.8): {sex_classification_info.get('high_confidence_count', 'N/A')}\n")
            if 'cv_accuracy' in sex_classification_info:
                f.write(f"  CV Accuracy: {sex_classification_info['cv_accuracy']:.1%}\n")
            if 'cv_auc' in sex_classification_info:
                f.write(f"  CV AUC: {sex_classification_info['cv_auc']:.3f}\n")
            if 'discovered_marker' in sex_classification_info:
                f.write(f"  Discovered marker: {sex_classification_info['discovered_marker']}\n")
            if 'top_features' in sex_classification_info:
                f.write(f"  Top features: {', '.join(sex_classification_info['top_features'])}\n")
            # Within-phenotype specific results
            if 'phenotype_results' in sex_classification_info:
                f.write(f"\n  Per-Phenotype Results:\n")
                for pheno, pheno_result in sex_classification_info['phenotype_results'].items():
                    f.write(f"    {pheno}:\n")
                    f.write(f"      Samples: {pheno_result.get('n_samples', 'N/A')}\n")
                    f.write(f"      Male: {pheno_result.get('male_count', 'N/A')}, Female: {pheno_result.get('female_count', 'N/A')}\n")
                    f.write(f"      CV Accuracy: {pheno_result['cv_accuracy']:.1%}\n")
                    if 'top_features' in pheno_result and pheno_result['top_features']:
                        f.write(f"      Top features: {', '.join(pheno_result['top_features'][:3])}\n")
                    if 'discovered_marker' in pheno_result:
                        f.write(f"      Discovered marker: {pheno_result['discovered_marker']}\n")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nComplete! Duration: {duration:.1f}s")
    print(f"Output: {args.output}.data.csv")
    return 0
