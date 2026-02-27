"""
CliqueFinder analyze command - Regulatory clique discovery.

Usage:
    cliquefinder analyze --input data.csv --discover --workers 6
"""

import argparse
import re
import sys
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from cliquefinder.cli._validators import _positive_int


def _detect_log_transform_status(input_path: Path, data: np.ndarray) -> tuple[bool, str]:
    """
    Detect if input data is already log-transformed.

    Returns:
        (is_log_transformed, detection_method)
        detection_method is 'metadata', 'heuristic', or 'unknown'
    """
    import json

    # Check for params.json sidecar file
    base_path = str(input_path)
    if base_path.endswith('.data.csv'):
        params_path = Path(base_path.replace('.data.csv', '.params.json'))
    else:
        params_path = Path(base_path.rsplit('.', 1)[0] + '.params.json')

    if params_path.exists():
        try:
            with open(params_path) as f:
                params = json.load(f)
            if 'is_log_transformed' in params:
                return params['is_log_transformed'], 'metadata'
        except (json.JSONDecodeError, KeyError):
            pass

    # Heuristic fallback: log1p data has max < 25, range < 25
    max_val = np.nanmax(data)
    min_val = np.nanmin(data)

    if max_val < 25 and (max_val - min_val) < 25:
        return True, 'heuristic'
    elif max_val > 100:
        return False, 'heuristic'

    return False, 'unknown'


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the analyze subcommand."""
    parser = subparsers.add_parser(
        "analyze",
        help="Discover regulatory cliques using INDRA CoGEx",
        description="Phase 2: Stratified regulatory module analysis for ALS transcriptomics"
    )

    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Expression data CSV (Ensembl IDs as rows)")
    parser.add_argument("--metadata", "-m", type=Path,
                        help="Sample metadata CSV")
    parser.add_argument("--output", "-o", type=Path, default=Path("results/analysis"),
                        help="Output directory for results")
    parser.add_argument("--regulators", nargs="+",
                        default=None,
                        help="Regulator gene symbols to analyze (required unless --discover)")
    parser.add_argument("--stratify-by", nargs="+", default=None,
                        help="Metadata columns for stratification (default: none)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratification (analyze all samples as one cohort)")
    parser.add_argument("--min-evidence", type=_positive_int, default=2,
                        help="Minimum evidence for CoGEx relationships")
    parser.add_argument("--min-correlation", type=float, default=0.7,
                        help="Minimum correlation for cliques")
    parser.add_argument("--min-clique-size", type=int, default=3,
                        help="Minimum genes in a clique")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Minimum samples per stratum")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="Path to .env file with CoGEx credentials")

    # Cross-modal RNA filtering
    parser.add_argument("--rna-filter", type=Path, default=None,
                        help="RNA-seq data CSV for cross-modal filtering (optional)")
    parser.add_argument("--rna-annotation", type=Path, default=None,
                        help="Gene annotation CSV for numeric RNA indices (optional)")

    # Expression filtering arguments
    parser.add_argument(
        "--min-cpm",
        type=float,
        default=1.0,
        help="Minimum CPM for gene to be considered expressed (default: 1.0)"
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.10,
        help="Fraction of samples in stratum where gene must be expressed (default: 0.10)"
    )
    parser.add_argument(
        "--expression-stratify-by",
        nargs="+",
        default=None,
        help="Metadata columns for expression stratification (default: uses --stratify-by)"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=10,
        help="Minimum samples per stratum for expression filtering (default: 10)"
    )
    parser.add_argument(
        "--require-sample-alignment",
        action="store_true",
        help="Only analyze samples present in both RNA and proteomics"
    )
    parser.add_argument(
        "--skip-expression-filter",
        action="store_true",
        help="Skip expression-based filtering (use ID overlap only)"
    )

    # Discovery mode
    parser.add_argument("--discover", action="store_true",
                        help="Discovery mode: auto-find regulators from INDRA")
    parser.add_argument("--min-targets", type=_positive_int, default=10,
                        help="Discovery: min INDRA targets in dataset")
    parser.add_argument("--max-targets", type=int, default=100,
                        help="Discovery: max INDRA targets (exclude hub regulators, default: 100)")
    parser.add_argument("--max-regulators", type=int, default=None,
                        help="Discovery: max regulators to analyze")
    parser.add_argument("--regulator-class", nargs="+",
                        choices=["tf", "kinase", "phosphatase", "e3_ligase", "receptor_kinase"],
                        default=None,
                        help="Filter regulators by functional class (e.g., --regulator-class tf kinase). "
                             "Omit for all classes.")
    parser.add_argument("--stmt-types", type=str, default=None,
                        help="INDRA statement types: preset (regulatory, activation, repression, "
                             "phosphorylation) or comma-separated raw types "
                             "(e.g., 'IncreaseAmount,Phosphorylation'). Default: regulatory")
    parser.add_argument("--strict-stmt-types", action="store_true", default=False,
                        help="Warn when using the mixed 'regulatory' preset, which conflates "
                             "activators and repressors. Suggests using --stmt-types activation "
                             "or --stmt-types repression for directional analysis.")
    parser.add_argument("--exact-cliques", action="store_true",
                        help="Use exact clique enumeration vs greedy")
    parser.add_argument("--correlation-method", choices=["pearson", "spearman", "max"],
                        default="max",
                        help="Correlation method: 'max' (default) uses max(Pearson, Spearman) "
                             "for each gene pair to capture both linear and monotonic relationships, "
                             "'pearson' uses linear correlation only, 'spearman' uses rank correlation only")

    # Parallelism
    parser.add_argument("--workers", "-j", type=_positive_int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--parallel-mode", choices=["threads", "processes", "hybrid"],
                        default="threads", help="Parallelism strategy")

    # Cohort filtering (ensures cliques are discovered only on target cohort)
    parser.add_argument("--cohort-config", type=Path, default=None,
                        help="YAML cohort definition file (filters samples before clique finding)")
    parser.add_argument("--genetic-contrast", type=str, default=None,
                        help="Quick cohort setup: mutation name (e.g., 'C9orf72') vs sporadic")

    # Preprocessing
    parser.add_argument("--log-transform", action="store_true", default=False,
                        help="Force log1p transformation (default: False, auto-detects if needed)")
    parser.add_argument("--no-log-transform", action="store_true",
                        help="Explicitly disable log transformation")
    parser.add_argument("--auto-detect-log", action="store_true", default=True,
                        help="Auto-detect if data is already log-transformed (default: True)")
    parser.add_argument("--no-auto-detect-log", dest="auto_detect_log", action="store_false",
                        help="Disable auto-detection of log transformation")
    parser.add_argument("--filter-low-expression", type=float, default=None,
                        dest="filter_low_expression",
                        help="Filter genes with mean expression below threshold (after log transform)")

    parser.set_defaults(func=run_analyze)


def run_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command."""
    import logging
    from datetime import datetime
    from cliquefinder import BioMatrix
    from cliquefinder.io.loaders import load_matrix  # UPDATED: Use smart loader
    from cliquefinder.knowledge.cogex import CoGExClient
    from cliquefinder.cli._analyze_core import (
        build_gene_symbol_mapping,
        create_symbol_indexed_matrix,
        run_stratified_analysis,
        save_results,
        print_summary,
    )
    from cliquefinder.knowledge.rna_loader import RNADataLoader
    from cliquefinder.knowledge.cross_modal_mapper import (
        CrossModalIDMapper,
        SampleAlignedCrossModalMapper,
    )
    from cliquefinder.quality import StratifiedExpressionFilter
    from cliquefinder.cohort import resolve_cohort_from_args

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Handle log transform flag
    if args.no_log_transform:
        args.log_transform = False

    # Validate required arguments
    if not args.discover and not args.regulators:
        logger.error("--regulators is required unless --discover is set")
        return 1

    # Handle stratification flag
    if args.no_stratify:
        args.stratify_by = []
        logger.info("Stratification disabled - analyzing all samples as one cohort")
    elif args.stratify_by is None:
        args.stratify_by = []  # Default to no stratification

    print(f"\n{'='*70}")
    print("  Phase 2: Regulatory Clique Discovery")
    print(f"{'='*70}\n")

    # Load expression data (using smart loader for ID cleaning)
    logger.info(f"Loading: {args.input}")
    matrix = load_matrix(args.input)  # UPDATED: Auto-detects Answer ALS format

    # If external metadata provided, merge it (optional override)
    if args.metadata and args.metadata.exists():
        metadata = pd.read_csv(args.metadata, index_col=0)

        # Try direct match first
        common_samples = matrix.sample_ids.intersection(metadata.index)

        if len(common_samples) == 0:
            # No direct match - try extracting participant IDs from sample IDs
            # Pattern: CASE_NEUAA295HHE-9014-P_D3 -> CASE-NEUAA295HHE
            # Also: CTRL_W14C179-7929-P_B2 -> CTRL-W14C179
            participant_pattern = re.compile(r'^(CASE|CTRL)_?([A-Z0-9]+?)[-_]\d+')

            sample_to_participant = {}
            for sample_id in matrix.sample_ids:
                match = participant_pattern.match(str(sample_id))
                if match:
                    phenotype, neu_id = match.groups()
                    participant_id = f"{phenotype}-{neu_id}"
                    sample_to_participant[sample_id] = participant_id

            if sample_to_participant:
                # Create merged metadata using participant ID lookup
                merged_meta = matrix.sample_metadata.copy()
                for sample_id in matrix.sample_ids:
                    participant_id = sample_to_participant.get(sample_id)
                    if participant_id and participant_id in metadata.index:
                        for col in metadata.columns:
                            if col not in merged_meta.columns:
                                merged_meta.loc[sample_id, col] = metadata.loc[participant_id, col]
                            elif pd.isna(merged_meta.loc[sample_id, col]):
                                merged_meta.loc[sample_id, col] = metadata.loc[participant_id, col]

                # Map Sex column if present
                if 'SEX' in metadata.columns and 'Sex' not in merged_meta.columns:
                    merged_meta['Sex'] = merged_meta.get('SEX', pd.NA)

                n_matched = sum(1 for s in matrix.sample_ids if sample_to_participant.get(s) in metadata.index)
                logger.info(f"Merged external metadata via participant ID: {n_matched}/{len(matrix.sample_ids)} samples matched")

                matrix = BioMatrix(
                    data=matrix.data,
                    feature_ids=matrix.feature_ids,
                    sample_ids=matrix.sample_ids,
                    sample_metadata=merged_meta,
                    quality_flags=matrix.quality_flags
                )
        else:
            # Direct match worked
            if len(common_samples) < len(matrix.sample_ids):
                logger.warning(
                    f"Metadata missing for {len(matrix.sample_ids) - len(common_samples)} samples. "
                    "Keeping original matrix samples."
                )
            matrix = BioMatrix(
                data=matrix.data,
                feature_ids=matrix.feature_ids,
                sample_ids=matrix.sample_ids,
                sample_metadata=metadata.loc[common_samples].reindex(matrix.sample_ids),
                quality_flags=matrix.quality_flags
            )

    logger.info(f"Loaded: {matrix.n_features} genes x {matrix.n_samples} samples")
    logger.info(f"Metadata columns: {list(matrix.sample_metadata.columns)}")

    # =============================================================================
    # Cohort Filtering (optional - ensures cliques discovered only on target cohort)
    # =============================================================================
    cohort_info = None
    if args.cohort_config or args.genetic_contrast:
        cohort_metadata, condition_col, cohort_contrasts = resolve_cohort_from_args(
            metadata=matrix.sample_metadata,
            cohort_config=args.cohort_config,
            genetic_contrast=args.genetic_contrast,
        )
        # Filter matrix to only include samples in the cohort
        cohort_samples = list(cohort_metadata.index)
        sample_mask = np.array([s in cohort_samples for s in matrix.sample_ids])
        sample_indices = np.where(sample_mask)[0]

        # Slice quality_flags if it's 2D (per-sample), otherwise keep as-is (per-feature only)
        if matrix.quality_flags.ndim == 2:
            sliced_flags = matrix.quality_flags[:, sample_indices]
        else:
            sliced_flags = matrix.quality_flags

        matrix = BioMatrix(
            data=matrix.data[:, sample_indices],
            feature_ids=matrix.feature_ids,
            sample_ids=pd.Index([matrix.sample_ids[i] for i in sample_indices]),
            sample_metadata=cohort_metadata,
            quality_flags=sliced_flags
        )
        cohort_info = {
            'cohort_config': str(args.cohort_config) if args.cohort_config else None,
            'genetic_contrast': args.genetic_contrast,
            'original_samples': int(sample_mask.sum() + (~sample_mask).sum()),
            'cohort_samples': len(cohort_samples),
            'groups': {g: int((cohort_metadata[condition_col] == g).sum())
                       for g in cohort_metadata[condition_col].unique()},
        }
        logger.info(f"Cohort filter applied: {cohort_info['original_samples']} → {cohort_info['cohort_samples']} samples")
        for group, count in cohort_info['groups'].items():
            logger.info(f"  {group}: n={count}")

    # Determine log transform status
    should_log_transform = args.log_transform
    detected_log_status = None
    detection_method = None

    if args.auto_detect_log and not args.no_log_transform:
        detected_log_status, detection_method = _detect_log_transform_status(args.input, matrix.data)

        if detected_log_status:
            logger.info(f"Data appears to be already log-transformed (detected via {detection_method})")
            if not args.log_transform:
                should_log_transform = False
        else:
            if detection_method == 'heuristic':
                logger.info(f"Data appears to be raw (linear scale), applying log1p transformation")
                should_log_transform = True
            elif detection_method == 'unknown':
                logger.warning("Could not determine if data is log-transformed. Use --log-transform or --no-log-transform explicitly.")

    if args.no_log_transform:
        should_log_transform = False
        logger.info("Log transformation explicitly disabled")

    if should_log_transform:
        logger.info("Applying log1p transformation...")
        original_range = (matrix.data.min(), matrix.data.max())
        matrix = BioMatrix(
            data=np.log1p(matrix.data), feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids, sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags
        )
        transformed_range = (matrix.data.min(), matrix.data.max())
        logger.info(f"  Original range: [{original_range[0]:.2f}, {original_range[1]:.2f}]")
        logger.info(f"  Transformed range: [{transformed_range[0]:.2f}, {transformed_range[1]:.2f}]")
    else:
        logger.info("Log transformation skipped (data already log-transformed or explicitly disabled)")

    # Filter low-expression genes (optional)
    if args.filter_low_expression is not None:
        mean_expr = np.mean(matrix.data, axis=1)
        keep_mask = mean_expr >= args.filter_low_expression
        n_filtered = (~keep_mask).sum()
        logger.info(f"Filtering {n_filtered} genes with mean < {args.filter_low_expression}")
        matrix = BioMatrix(
            data=matrix.data[keep_mask],
            feature_ids=matrix.feature_ids[keep_mask],
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags[keep_mask]
        )
        logger.info(f"  Remaining genes: {matrix.n_features}")

    # =============================================================================
    # UniProt → Gene Symbol Transformation
    # =============================================================================
    # CRITICAL: INDRA returns gene symbols, but proteomics data uses UniProt IDs.
    # We must transform feature IDs to gene symbols for clique analysis to work.
    # This is detected automatically by checking if feature IDs match UniProt pattern.

    def detect_id_type(feature_ids: pd.Index) -> str:
        """Detect ID type from feature IDs."""
        import re
        uniprot_pattern = re.compile(r'^[A-Z][A-Z0-9]{5,9}$')
        ensembl_pattern = re.compile(r'^ENSG[0-9]+$')

        sample = list(feature_ids[:min(100, len(feature_ids))])
        uniprot_count = sum(1 for fid in sample if uniprot_pattern.match(str(fid)))
        ensembl_count = sum(1 for fid in sample if ensembl_pattern.match(str(fid)))

        if uniprot_count > len(sample) * 0.5:
            return 'uniprot'
        elif ensembl_count > len(sample) * 0.5:
            return 'ensembl'
        else:
            return 'symbol'  # Assume already gene symbols

    def transform_uniprot_to_symbols(
        matrix: 'BioMatrix',
        cache_dir: Optional[Path] = None
    ) -> 'BioMatrix':
        """
        Transform UniProt IDs to gene symbols using INDRA uniprot_client.

        This is essential for clique analysis because INDRA returns gene symbols
        but proteomics data uses UniProt accessions.

        Args:
            matrix: BioMatrix with UniProt IDs as feature_ids
            cache_dir: Cache directory for ID mapping

        Returns:
            BioMatrix with gene symbols as feature_ids (unmapped IDs dropped)
        """
        from indra.databases import uniprot_client

        feature_ids = list(matrix.feature_ids)
        mapping = {}  # UniProt → gene symbol
        failed = []

        logger.info(f"Transforming {len(feature_ids)} UniProt IDs to gene symbols...")

        for i, uniprot_id in enumerate(feature_ids):
            if i % 500 == 0 and i > 0:
                logger.debug(f"  Processed {i}/{len(feature_ids)} IDs...")

            try:
                gene_name = uniprot_client.get_gene_name(str(uniprot_id))
                if gene_name:
                    mapping[uniprot_id] = gene_name
                else:
                    failed.append(uniprot_id)
            except Exception:
                failed.append(uniprot_id)

        success_rate = len(mapping) / len(feature_ids) * 100 if feature_ids else 0
        logger.info(f"UniProt→Symbol mapping: {len(mapping)}/{len(feature_ids)} ({success_rate:.1f}%)")

        if len(failed) <= 10:
            logger.debug(f"  Failed IDs: {failed}")
        elif failed:
            logger.debug(f"  Failed IDs (first 10): {failed[:10]}...")

        if not mapping:
            logger.error("No UniProt IDs could be mapped to gene symbols!")
            raise ValueError("UniProt→Symbol mapping failed completely")

        # Build new matrix with only mapped genes
        # Keep only rows that have successful mappings
        keep_indices = [i for i, fid in enumerate(feature_ids) if fid in mapping]
        new_feature_ids = pd.Index([mapping[feature_ids[i]] for i in keep_indices])

        # Handle duplicate gene symbols (keep first occurrence)
        seen_symbols = set()
        unique_indices = []
        unique_feature_ids = []
        for idx, symbol in zip(keep_indices, new_feature_ids):
            if symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_indices.append(idx)
                unique_feature_ids.append(symbol)

        n_duplicates = len(keep_indices) - len(unique_indices)
        if n_duplicates > 0:
            logger.info(f"  Removed {n_duplicates} duplicate gene symbols (keeping first occurrence)")

        new_matrix = BioMatrix(
            data=matrix.data[unique_indices],
            feature_ids=pd.Index(unique_feature_ids),
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags[unique_indices]
        )

        logger.info(f"  Final matrix: {new_matrix.n_features} genes × {new_matrix.n_samples} samples")
        return new_matrix

    # Detect ID type and transform if needed
    detected_id_type = detect_id_type(matrix.feature_ids)
    logger.info(f"Detected feature ID type: {detected_id_type}")

    if detected_id_type == 'uniprot':
        logger.info("UniProt IDs detected - transforming to gene symbols for INDRA compatibility...")
        matrix = transform_uniprot_to_symbols(matrix)
    elif detected_id_type == 'ensembl':
        # Ensembl IDs also need transformation
        logger.info("Ensembl IDs detected - transforming to gene symbols for INDRA compatibility...")
        from cliquefinder.validation.id_mapping import MyGeneInfoMapper
        id_mapper = MyGeneInfoMapper(cache_dir=Path.home() / '.cache/biocore/id_mapping')
        mapping = id_mapper.map_ids(
            source_ids=list(matrix.feature_ids),
            source_type='ensembl_gene',
            target_type='symbol'
        )
        logger.info(f"Ensembl→Symbol mapping: {len(mapping)}/{matrix.n_features} ({100*len(mapping)/matrix.n_features:.1f}%)")

        # Build new matrix with mapped genes
        keep_indices = [i for i, fid in enumerate(matrix.feature_ids) if str(fid) in mapping]
        new_feature_ids = pd.Index([mapping[str(matrix.feature_ids[i])] for i in keep_indices])

        # Handle duplicates
        seen_symbols = set()
        unique_indices = []
        unique_feature_ids = []
        for idx, symbol in zip(keep_indices, new_feature_ids):
            if symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_indices.append(idx)
                unique_feature_ids.append(symbol)

        matrix = BioMatrix(
            data=matrix.data[unique_indices],
            feature_ids=pd.Index(unique_feature_ids),
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=matrix.quality_flags[unique_indices]
        )
        logger.info(f"  Final matrix: {matrix.n_features} genes × {matrix.n_samples} samples")
    else:
        logger.info("Gene symbols detected - no transformation needed")

    # Load RNA filter if provided (cross-modal integration with sample alignment)
    rna_filter_genes = None
    sample_alignment_stats = None
    expression_filter_stats = None

    if args.rna_filter and args.rna_filter.exists():
        logger.info(f"Loading RNA filter: {args.rna_filter}")
        rna_loader = RNADataLoader()

        # Load with full matrix for expression filtering (unless skipped)
        include_matrix = not args.skip_expression_filter
        rna_dataset = rna_loader.load(
            rna_path=args.rna_filter,
            annotation_path=args.rna_annotation if args.rna_annotation else None,
            include_matrix=include_matrix
        )
        logger.info(f"RNA data: {rna_dataset.n_genes} genes x {rna_dataset.n_samples} samples, type={rna_dataset.id_type}")

        # Propagate metadata from proteomics to RNA dataset for stratified expression filtering
        # RNA sample IDs may be like: CASE-NEUVM674HUA-5257-T_P003
        # Proteomics sample IDs: CASE_NEUAA295HHE-9014-P_D3
        # Strategy: Match by participant ID (NEU...) and propagate phenotype + Sex
        if rna_dataset.matrix is not None:
            logger.info("Propagating metadata from proteomics to RNA samples...")

            # Helper function to propagate metadata from proteomics to RNA
            def propagate_metadata_to_rna(
                proteomics_sample_ids: List[str],
                proteomics_metadata: pd.DataFrame,
                rna_sample_ids: List[str],
                participant_pattern: re.Pattern = re.compile(r'(NEU[A-Z0-9]+)')
            ) -> pd.DataFrame:
                """
                Propagate metadata from proteomics samples to RNA samples via participant ID matching.

                Args:
                    proteomics_sample_ids: Sample IDs from proteomics matrix
                    proteomics_metadata: Metadata DataFrame from proteomics matrix
                    rna_sample_ids: Sample IDs from RNA matrix
                    participant_pattern: Regex pattern to extract participant ID

                Returns:
                    DataFrame indexed by RNA sample IDs with propagated metadata
                """
                # Extract participant IDs from both datasets
                prot_participant_map = {}  # participant_id -> proteomics sample_id
                for sample_id in proteomics_sample_ids:
                    match = participant_pattern.search(str(sample_id))
                    if match:
                        participant_id = match.group(1)
                        # Keep first match for each participant
                        if participant_id not in prot_participant_map:
                            prot_participant_map[participant_id] = sample_id

                # Build RNA metadata by matching to proteomics
                rna_metadata_rows = []
                n_matched = 0

                for rna_sample_id in rna_sample_ids:
                    # Extract participant ID from RNA sample
                    match = participant_pattern.search(str(rna_sample_id))

                    if match:
                        participant_id = match.group(1)
                        prot_sample_id = prot_participant_map.get(participant_id)

                        if prot_sample_id:
                            # Found matching proteomics sample - propagate metadata
                            n_matched += 1

                            # Extract phenotype from RNA sample ID prefix (more reliable than proteomics)
                            phenotype = 'CASE' if str(rna_sample_id).startswith('CASE-') else \
                                       'CTRL' if str(rna_sample_id).startswith('CTRL-') else pd.NA

                            # Get Sex from proteomics metadata if available
                            sex_value = pd.NA
                            if prot_sample_id in proteomics_metadata.index:
                                prot_meta = proteomics_metadata.loc[prot_sample_id]
                                # Check for Sex or Sex_predicted columns
                                if 'Sex_predicted' in prot_meta.index and pd.notna(prot_meta['Sex_predicted']):
                                    sex_value = prot_meta['Sex_predicted']
                                elif 'Sex' in prot_meta.index and pd.notna(prot_meta['Sex']):
                                    sex_value = prot_meta['Sex']

                            row = pd.Series({
                                'phenotype': phenotype,
                                'Sex': sex_value,
                                'participant_id': participant_id,
                                'matched_proteomics_sample': prot_sample_id
                            }, name=rna_sample_id)
                            rna_metadata_rows.append(row)
                            continue

                    # No match - create row with inferred phenotype only
                    phenotype = 'CASE' if str(rna_sample_id).startswith('CASE-') else \
                               'CTRL' if str(rna_sample_id).startswith('CTRL-') else pd.NA

                    row = pd.Series({
                        'phenotype': phenotype,
                        'Sex': pd.NA,
                        'participant_id': pd.NA,
                        'matched_proteomics_sample': pd.NA
                    }, name=rna_sample_id)
                    rna_metadata_rows.append(row)

                result_df = pd.DataFrame(rna_metadata_rows)
                logger.info(
                    f"Metadata propagation: {n_matched}/{len(rna_sample_ids)} RNA samples matched to proteomics "
                    f"({100*n_matched/len(rna_sample_ids):.1f}%)"
                )

                return result_df

            # Propagate metadata from proteomics to RNA
            rna_sample_metadata = propagate_metadata_to_rna(
                proteomics_sample_ids=list(matrix.sample_ids),
                proteomics_metadata=matrix.sample_metadata,
                rna_sample_ids=list(rna_dataset.sample_ids)
            )

            # Log summary statistics
            n_with_phenotype = rna_sample_metadata['phenotype'].notna().sum()
            n_with_sex = rna_sample_metadata['Sex'].notna().sum()
            logger.info(
                f"RNA metadata summary: {n_with_phenotype} samples with phenotype, "
                f"{n_with_sex} samples with Sex"
            )

            # Update RNA dataset with propagated metadata
            rna_dataset = rna_dataset.with_metadata(rna_sample_metadata)
            logger.info(f"RNA matrix metadata columns: {list(rna_dataset.matrix.sample_metadata.columns)}")

        # Merge external metadata if provided (optional override/enrichment)
        if rna_dataset.matrix is not None and args.metadata and args.metadata.exists():
            logger.info("Merging external metadata into RNA dataset...")

            # Load metadata
            ext_metadata = pd.read_csv(args.metadata, index_col=0)

            # Build participant ID -> metadata sample ID mapping from ext_metadata
            # Metadata index format: CASE_NEUAA295HHE-9014-P_D3 or CTRL_W14C179-7929-P_B2
            # We need to extract participant IDs and map them to full sample IDs
            participant_to_meta_sample = {}
            meta_participant_pattern = re.compile(r'^(CASE|CTRL)[_-]([A-Z0-9]+?)[-_]\d+')
            for meta_sample_id in ext_metadata.index:
                meta_match = meta_participant_pattern.match(str(meta_sample_id))
                if meta_match:
                    phenotype, part_id = meta_match.groups()
                    participant_id = f"{phenotype}-{part_id}"
                    # Keep first match for each participant (arbitrary but consistent)
                    if participant_id not in participant_to_meta_sample:
                        participant_to_meta_sample[participant_id] = meta_sample_id

            logger.info(f"Built participant lookup: {len(participant_to_meta_sample)} unique participants in metadata")

            # Build RNA sample ID -> participant ID mapping
            # RNA format: CASE-NEUVM674HUA-5257-T_P003 -> CASE-NEUVM674HUA
            rna_sample_to_participant = {}

            # Pattern for Answer ALS RNA: (CASE|CTRL)-(NEU[A-Z0-9]+)-\d+
            aals_rna_pattern = re.compile(r'^(CASE|CTRL)-(NEU[A-Z0-9]+)')
            cgnd_pattern = re.compile(r'(CGND-HDA-\d+)')

            for sample_id in rna_dataset.sample_ids:
                sid = str(sample_id)

                # Try Answer ALS RNA pattern first (most common)
                aals_match = aals_rna_pattern.match(sid)
                if aals_match:
                    phenotype, neu_id = aals_match.groups()
                    participant_id = f"{phenotype}-{neu_id}"
                    # Check if this participant exists in our lookup
                    if participant_id in participant_to_meta_sample:
                        rna_sample_to_participant[sample_id] = participant_id
                    continue

                # Try CGND pattern (e.g., CGND-HDA-03247)
                cgnd_match = cgnd_pattern.search(sid)
                if cgnd_match:
                    cgnd_id = cgnd_match.group(1)
                    # NYGC_CGND_ID column might have this
                    if 'NYGC_CGND_ID' in ext_metadata.columns:
                        matches = ext_metadata[ext_metadata['NYGC_CGND_ID'] == cgnd_id].index
                        if len(matches) > 0:
                            rna_sample_to_participant[sample_id] = matches[0]
                    continue

                # Direct match attempt
                if sid in ext_metadata.index:
                    rna_sample_to_participant[sample_id] = sid

            n_rna_matched = len(rna_sample_to_participant)
            logger.info(f"External metadata mapping: {n_rna_matched}/{len(rna_dataset.sample_ids)} RNA samples matched")

            if n_rna_matched > 0:
                # Build aligned metadata DataFrame for RNA samples
                rna_meta_rows = []
                for sample_id in rna_dataset.sample_ids:
                    participant_id = rna_sample_to_participant.get(sample_id)
                    if participant_id and participant_id in participant_to_meta_sample:
                        meta_sample_id = participant_to_meta_sample[participant_id]
                        row = ext_metadata.loc[meta_sample_id].copy()
                        row.name = sample_id
                        rna_meta_rows.append(row)
                    else:
                        # Create empty row for unmatched samples
                        row = pd.Series(index=ext_metadata.columns, name=sample_id, dtype=object)
                        rna_meta_rows.append(row)

                ext_rna_metadata = pd.DataFrame(rna_meta_rows)

                # Add phenotype column from SUBJECT_GROUP if not already present
                if 'phenotype' not in ext_rna_metadata.columns:
                    if 'SUBJECT_GROUP' in ext_rna_metadata.columns:
                        # Map ALS -> CASE, others -> CTRL
                        ext_rna_metadata['phenotype'] = ext_rna_metadata['SUBJECT_GROUP'].apply(
                            lambda x: 'CASE' if str(x).upper() == 'ALS' else 'CTRL' if pd.notna(x) else pd.NA
                        )

                # Add Sex column (normalize from SEX) if not already present
                if 'Sex' not in ext_rna_metadata.columns and 'SEX' in ext_rna_metadata.columns:
                    ext_rna_metadata['Sex'] = ext_rna_metadata['SEX']

                # Merge external metadata with existing propagated metadata (external takes precedence)
                current_meta = rna_dataset.matrix.sample_metadata.copy()
                for col in ext_rna_metadata.columns:
                    if col not in current_meta.columns:
                        current_meta[col] = ext_rna_metadata[col]
                    else:
                        # Override with external metadata where available
                        current_meta[col] = ext_rna_metadata[col].combine_first(current_meta[col])

                # Update RNA dataset with merged metadata
                from cliquefinder.core.biomatrix import BioMatrix
                new_matrix = BioMatrix(
                    data=rna_dataset.matrix.data,
                    feature_ids=rna_dataset.matrix.feature_ids,
                    sample_ids=rna_dataset.matrix.sample_ids,
                    sample_metadata=current_meta,
                    quality_flags=rna_dataset.matrix.quality_flags
                )
                rna_dataset = rna_dataset.__class__(
                    gene_ids=rna_dataset.gene_ids,
                    id_type=rna_dataset.id_type,
                    n_genes=rna_dataset.n_genes,
                    n_samples=rna_dataset.n_samples,
                    sample_ids=rna_dataset.sample_ids,
                    matrix=new_matrix
                )
                logger.info(f"Merged external metadata. Final columns: {list(rna_dataset.matrix.sample_metadata.columns)}")

        # Expression filtering (stratified by phenotype/sex)
        expressed_gene_symbols = None
        if not args.skip_expression_filter and rna_dataset.matrix is not None:
            logger.info("Applying stratified expression filter...")
            # Use expression stratification columns, fall back to main stratify_by
            expr_stratify = args.expression_stratify_by or args.stratify_by

            # Verify required columns exist
            missing_cols = [c for c in expr_stratify if c not in rna_dataset.matrix.sample_metadata.columns]
            if missing_cols:
                logger.warning(f"Missing stratification columns in RNA metadata: {missing_cols}. Skipping expression filter.")
            else:
                expr_filter = StratifiedExpressionFilter(
                    min_cpm=args.min_cpm,
                    min_prevalence=args.min_prevalence,
                    stratify_by=expr_stratify,
                    min_group_size=args.min_group_size
                )

                try:
                    filter_result = expr_filter.get_passing_genes(rna_dataset.matrix)
                    expressed_ensembl_ids = filter_result.passed_genes
                    expression_filter_stats = {
                        'n_passed': filter_result.n_passed,
                        'n_failed': filter_result.n_failed,
                        'pass_rate': filter_result.pass_rate,
                        'stratum_stats': filter_result.stratum_stats,
                        'parameters': filter_result.parameters
                    }
                    logger.info(
                        f"Expression filter: {filter_result.n_passed}/{filter_result.n_passed + filter_result.n_failed} "
                        f"genes passed ({filter_result.pass_rate*100:.1f}%)"
                    )

                    # Convert Ensembl IDs to gene symbols for cross-modal matching
                    # (expression filter returns Ensembl IDs, mapper expects symbols)
                    if rna_dataset.id_type == 'ensembl_gene':
                        from cliquefinder.validation.id_mapping import MyGeneInfoMapper
                        logger.info("Converting expressed gene Ensembl IDs to symbols...")
                        id_mapper = MyGeneInfoMapper(
                            cache_dir=args.output / ".cache" / "id_mapping"
                        )
                        ensembl_to_symbol = id_mapper.map_ids(
                            list(expressed_ensembl_ids),
                            source_type='ensembl_gene',
                            target_type='symbol',
                            species='human'
                        )
                        expressed_gene_symbols = {
                            ensembl_to_symbol.get(eid)
                            for eid in expressed_ensembl_ids
                            if ensembl_to_symbol.get(eid) is not None
                        }
                        logger.info(
                            f"Converted {len(expressed_gene_symbols)}/{len(expressed_ensembl_ids)} "
                            f"expressed genes to symbols"
                        )
                    else:
                        expressed_gene_symbols = expressed_ensembl_ids

                except Exception as e:
                    logger.warning(f"Expression filtering failed: {e}. Falling back to ID overlap only.")
        elif args.skip_expression_filter:
            logger.info("Expression filtering skipped (--skip-expression-filter)")
        else:
            logger.info("RNA matrix not available for expression filtering, using ID overlap only")

        # Cross-modal mapping with sample alignment
        mapper = SampleAlignedCrossModalMapper(cache_dir=args.output / ".cache")

        mapping = mapper.unify_with_expression_filter(
            protein_ids=list(matrix.feature_ids),
            proteomics_sample_ids=list(matrix.sample_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_sample_ids=rna_dataset.sample_ids if hasattr(rna_dataset, 'sample_ids') else [],
            rna_id_type=rna_dataset.id_type,
            expressed_genes=expressed_gene_symbols,
            species='human'
        )

        # Store sample alignment stats
        if mapping.sample_alignment:
            sample_alignment_stats = {
                'n_matched_participants': mapping.sample_alignment.n_matched,
                'n_proteomics_only': len(mapping.sample_alignment.proteomics_only),
                'n_rna_only': len(mapping.sample_alignment.rna_only),
                'n_proteomics_samples': mapping.sample_alignment.n_proteomics_samples,
                'n_rna_samples': mapping.sample_alignment.n_rna_samples,
            }
            logger.info(
                f"Sample alignment: {mapping.sample_alignment.n_matched} participants matched "
                f"({sample_alignment_stats['n_proteomics_only']} prot-only, "
                f"{sample_alignment_stats['n_rna_only']} rna-only)"
            )

        # RNA filter genes for REGULATOR selection
        # Use ALL RNA-expressed genes (not just proteomics intersection) to filter regulators
        # Regulators must be expressed in RNA, but don't need to be in proteomics dataset
        if expressed_gene_symbols is not None:
            rna_filter_genes = expressed_gene_symbols
            logger.info(
                f"RNA regulator filter: {len(rna_filter_genes)} expressed genes "
                f"(regulators must be RNA-expressed, targets from proteomics)"
            )
        elif mapping.common_genes:
            # Fallback to cross-modal overlap if expression filtering was skipped
            rna_filter_genes = mapping.common_genes
            logger.info(
                f"RNA regulator filter (fallback): {len(mapping.common_genes)} common genes "
                f"(expression filter skipped, using ID overlap)"
            )
    else:
        logger.info("No RNA annotation specified. All proteomics proteins used for analysis.")

    # Detect source ID type
    first_id = str(matrix.feature_ids[0])
    source_type = 'ensembl_gene' if first_id.startswith('ENSG') else 'symbol'
    logger.info(f"Detected source ID type: {source_type} (Example: {first_id})")

    # Map IDs to symbols
    ensembl_to_symbol = build_gene_symbol_mapping(
        list(matrix.feature_ids),
        cache_dir=args.output / ".cache",
        source_type=source_type
    )

    # Create symbol-indexed matrix
    symbol_matrix, _ = create_symbol_indexed_matrix(matrix, ensembl_to_symbol)

    # Parse regulator class filter
    from cliquefinder.knowledge.cogex import RegulatorClass
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
        logger.info(f"Regulator class filter: {[c.value for c in regulator_classes]}")

    # Parse statement types
    from cliquefinder.knowledge.cogex import resolve_stmt_types
    stmt_types = resolve_stmt_types(args.stmt_types) if args.stmt_types else None
    if stmt_types:
        logger.info(f"Statement types: {stmt_types}")

    # Strict stmt-types warning (H5 audit finding)
    if args.strict_stmt_types and (args.stmt_types is None or args.stmt_types == "regulatory"):
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

    # Connect to CoGEx
    logger.info("Connecting to INDRA CoGEx...")
    cogex_client = CoGExClient(env_file=args.env_file)
    if not cogex_client.ping():
        logger.error("Failed to connect to INDRA CoGEx")
        return 1

    # Run analysis
    try:
        results = run_stratified_analysis(
            matrix=symbol_matrix,
            regulators=args.regulators if not args.discover else None,
            cogex_client=cogex_client,
            stratify_by=args.stratify_by,
            min_evidence=args.min_evidence,
            min_correlation=args.min_correlation,
            min_clique_size=args.min_clique_size,
            min_samples=args.min_samples,
            discover_mode=args.discover,
            min_targets=args.min_targets,
            max_targets=args.max_targets,
            max_regulators=args.max_regulators,
            use_fast_maximum=not args.exact_cliques,
            n_workers=args.workers,
            parallel_mode=args.parallel_mode,
            correlation_method=args.correlation_method,
            rna_filter_genes=rna_filter_genes,
            regulator_classes=regulator_classes,
            stmt_types=stmt_types,
        )

        # Save results
        parameters = {
            'timestamp': datetime.now().isoformat(),
            'input': str(args.input),
            'regulators': args.regulators if not args.discover else 'discovered',
            'discover_mode': args.discover,
            'stratify_by': args.stratify_by,
            'min_evidence': args.min_evidence,
            'min_correlation': args.min_correlation,
            'min_clique_size': args.min_clique_size,
            'min_samples': args.min_samples,
            'min_targets': args.min_targets if args.discover else None,
            'max_targets': args.max_targets if args.discover else None,
            'max_regulators': args.max_regulators if args.discover else None,
            'regulator_classes': [c.value for c in regulator_classes] if regulator_classes else None,
            'use_fast_maximum': not args.exact_cliques,
            'correlation_method': args.correlation_method,
            'log_transform_applied': should_log_transform,
            'log_transform_detected_status': detected_log_status,
            'log_transform_detection_method': detection_method,
            'filter_low_expression': args.filter_low_expression,
            'n_genes_in_universe': symbol_matrix.n_features,
            'n_samples': symbol_matrix.n_samples,
            'n_workers': args.workers,
            'parallel_mode': args.parallel_mode,
            # RNA cross-modal filtering
            'rna_filter': str(args.rna_filter) if args.rna_filter else None,
            'rna_annotation': str(args.rna_annotation) if args.rna_annotation else None,
            'rna_filtered_genes_count': len(rna_filter_genes) if rna_filter_genes else 0,
            # Expression filtering parameters
            'expression_filter': {
                'min_cpm': args.min_cpm,
                'min_prevalence': args.min_prevalence,
                'stratify_by': args.expression_stratify_by or args.stratify_by,
                'min_group_size': args.min_group_size,
                'skipped': args.skip_expression_filter,
            } if not args.skip_expression_filter else None,
            'expression_filter_stats': expression_filter_stats,
            # Sample alignment
            'sample_alignment': sample_alignment_stats,
            'require_sample_alignment': args.require_sample_alignment,
            # Cohort filtering
            'cohort_filter': cohort_info,
        }

        save_results(results, args.output, parameters)
        print_summary(results)
        logger.info("Analysis complete!")
        return 0

    finally:
        cogex_client.close()
