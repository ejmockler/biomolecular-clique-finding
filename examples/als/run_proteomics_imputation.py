#!/usr/bin/env python3
"""
AnswerALS Proteomics Imputation Example
========================================

This is an AnswerALS-specific example demonstrating how to use the cliquefinder
imputation pipeline on real proteomics data.

Dataset: AnswerALS Cohort 1-6 Proteomics (436 participants)
- Data file: Data_AnswerALS-436-P_proteomics-protein-matrix_correctedImputed.txt
- Metadata: aals_dataportal_datatable.csv

This example shows:
1. Loading AnswerALS-specific data formats
2. Phenotype inference from GUID and sample IDs
3. Multi-pass outlier detection with adaptive boxplots
4. Soft-clip imputation with global bounds fallback
5. Quality control visualization generation

For generic imputation utilities, see:
- cliquefinder.quality.MultiPassOutlierDetector
- cliquefinder.quality.Imputer
- cliquefinder.viz.qc.QCVisualizer

Run adaptive outlier imputation on ALS proteomics data.

Uses the cliquefinder package API for multi-pass detection and imputation:
- MultiPassOutlierDetector: Three-pass detection (within-group, residual, global cap)
- Imputer: Soft-clip imputation with global bounds fallback
- QCVisualizer: Quality control visualizations

Phenotype inference:
1. SUBJECT_GROUP == "ALS" → CASE
2. SUBJECT_GROUP == "Healthy Control" → CTRL
3. Fallback: sample_id starts with "CTRL_" → CTRL, "CASE_" → CASE
4. Exclude: Non-ALS MND, Asymptomatic, metadata rows
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.quality import MultiPassOutlierDetector, Imputer
from cliquefinder.viz.qc import QCVisualizer

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'input': {
        'data': 'Data_AnswerALS-436-P_proteomics-protein-matrix_correctedImputed.txt',
        'metadata': 'aals_dataportal_datatable.csv',
    },
    'output': {
        'dir': Path('output/proteomics_imputation'),
    },
    'detection': {
        'method': 'adjusted-boxplot',
        'threshold': 1.5,
        'scoring_method': 'binary',
        'group_cols': ['phenotype'],
        'residual_enabled': True,
        'residual_threshold': 4.25,
        'residual_high_end_only': True,
        'global_cap_enabled': True,
        'global_cap_percentile': 99.95,
    },
    'imputation': {
        'strategy': 'soft-clip',
        'sharpness': 5.0,
    },
}

# =============================================================================
# PHENOTYPE INFERENCE (STUDY-SPECIFIC)
# =============================================================================

def extract_guid(sample_id: str) -> str:
    """Extract GUID from sample ID like CASE_NEUAA295HHE-9014-P_D3."""
    parts = sample_id.split('_')
    if len(parts) >= 2:
        return parts[1].split('-')[0]
    return None

def infer_phenotype(sample_id: str, guid_to_group: dict) -> tuple[str, str]:
    """
    Infer phenotype with fallback to sample ID.

    Returns (phenotype, source) tuple.
    """
    guid = extract_guid(sample_id)

    # Primary: metadata lookup
    if guid and guid in guid_to_group:
        group = guid_to_group[guid]
        if group == 'ALS':
            return 'CASE', 'metadata'
        elif group == 'Healthy Control':
            return 'CTRL', 'metadata'
        else:
            return group, 'metadata'

    # Fallback: sample ID prefix
    if sample_id.startswith('CTRL_'):
        return 'CTRL', 'sample_id_fallback'
    elif sample_id.startswith('CASE_'):
        return 'CASE', 'sample_id_fallback'

    return 'UNKNOWN', 'no_match'

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("ALS PROTEOMICS ADAPTIVE OUTLIER IMPUTATION")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading data...")
    data = pd.read_csv(CONFIG['input']['data'], sep='\t', index_col=0)
    print(f"  Proteomics: {data.shape[0]:,} proteins × {data.shape[1]:,} samples")

    # Filter metadata rows
    metadata_rows = ['nFragment', 'nPeptide', 'iRT_protein']
    data = data[~data.index.str.contains('|'.join(metadata_rows), case=False, na=False)]
    print(f"  After filtering metadata rows: {data.shape[0]:,} proteins")

    # Load portal metadata
    portal = pd.read_csv(CONFIG['input']['metadata'])
    guid_to_group = dict(zip(portal['GUID'], portal['SUBJECT_GROUP']))
    guid_to_sex = dict(zip(portal['GUID'], portal['SEX']))
    print(f"  Portal metadata: {len(portal):,} participants")

    # --- Phenotype inference ---
    print("\n[2/6] Inferring phenotypes...")
    sample_phenotypes = {}
    sample_sources = {}
    for sample_id in data.columns:
        pheno, source = infer_phenotype(sample_id, guid_to_group)
        sample_phenotypes[sample_id] = pheno
        sample_sources[sample_id] = source

    # Filter to CASE/CTRL only
    valid_samples = [s for s, p in sample_phenotypes.items() if p in ('CASE', 'CTRL')]
    excluded = [s for s, p in sample_phenotypes.items() if p not in ('CASE', 'CTRL')]

    print(f"  CASE samples: {sum(1 for p in sample_phenotypes.values() if p == 'CASE')}")
    print(f"  CTRL samples: {sum(1 for p in sample_phenotypes.values() if p == 'CTRL')}")
    print(f"  Excluded: {len(excluded)} (Non-ALS MND, Asymptomatic, Unknown)")

    data = data[valid_samples]
    print(f"  Final matrix: {data.shape[0]:,} proteins × {data.shape[1]:,} samples")

    # --- Build BioMatrix ---
    print("\n[3/6] Building BioMatrix with log2 transform...")
    log2_data = np.log2(data.values + 1)
    print(f"  log2 range: [{log2_data.min():.2f}, {log2_data.max():.2f}]")

    sample_metadata = pd.DataFrame({
        'sample_id': valid_samples,
        'phenotype': [sample_phenotypes[s] for s in valid_samples],
        'sex': [guid_to_sex.get(extract_guid(s), 'Unknown') for s in valid_samples],
        'inference_source': [sample_sources[s] for s in valid_samples],
    })
    sample_metadata.set_index('sample_id', inplace=True)

    matrix = BioMatrix(
        data=log2_data,
        feature_ids=pd.Index(data.index),
        sample_ids=pd.Index(valid_samples),
        sample_metadata=sample_metadata,
        quality_flags=np.zeros(log2_data.shape, dtype=np.uint8),
    )

    print(f"  BioMatrix: {matrix.n_features:,} features × {matrix.n_samples:,} samples")
    print(f"  Phenotype distribution: {dict(sample_metadata['phenotype'].value_counts())}")

    # --- Multi-pass outlier detection ---
    print("\n[4/6] Running multi-pass outlier detection...")
    print(f"  Detection method: {CONFIG['detection']['method']}")
    print(f"  Detection mode: within_group")
    print(f"  Scoring: {CONFIG['detection']['scoring_method']}")
    print(f"  Two-pass (residual): {CONFIG['detection']['residual_enabled']}")
    print(f"  Imputation: {CONFIG['imputation']['strategy']}")

    detector = MultiPassOutlierDetector(
        detection_method=CONFIG['detection']['method'],
        detection_threshold=CONFIG['detection']['threshold'],
        scoring_method=CONFIG['detection']['scoring_method'],
        group_cols=CONFIG['detection']['group_cols'],
        residual_enabled=CONFIG['detection']['residual_enabled'],
        residual_threshold=CONFIG['detection']['residual_threshold'],
        residual_high_end_only=CONFIG['detection']['residual_high_end_only'],
        global_cap_enabled=CONFIG['detection']['global_cap_enabled'],
        global_cap_percentile=CONFIG['detection']['global_cap_percentile'],
    )

    matrix_flagged = detector.apply(matrix)

    # Get outlier counts
    outlier_mask = (matrix_flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
    n_outliers = outlier_mask.sum()
    pct_outliers = 100 * n_outliers / matrix.data.size

    # Print pass-by-pass summary
    summary = detector.get_pass_summary()
    print(f"\n  Pass 1 (within-group) results:")
    print(f"    Outliers detected: {summary['pass1_within_group']:,} ({100 * summary['pass1_within_group'] / matrix.data.size:.2f}%)")

    if hasattr(detector.detector_, 'medcouples_') and detector.detector_.medcouples_ is not None:
        mc = detector.detector_.medcouples_
        print(f"    Medcouple range: [{mc.min():.3f}, {mc.max():.3f}]")
        print(f"    Left-skewed features (MC < -0.1): {(mc < -0.1).sum()}")
        print(f"    Right-skewed features (MC > 0.1): {(mc > 0.1).sum()}")

    if CONFIG['detection']['residual_enabled']:
        print(f"\n  Pass 2 (residual-based) for high-end outliers...")
        print(f"    New high-end outliers: {summary['pass2_residual']:,}")

    if CONFIG['detection']['global_cap_enabled']:
        print(f"\n  Pass 3 (global percentile cap)...")
        print(f"    Global {CONFIG['detection']['global_cap_percentile']}th percentile: {summary['global_threshold']:.2f}")
        print(f"    New outliers from global cap: {summary['pass3_global_cap']:,}")

    print(f"\n  Final results:")
    print(f"    Total outliers: {n_outliers:,} ({pct_outliers:.2f}%)")

    # --- Imputation ---
    print("\n  Running imputation...")
    imputer = Imputer(
        strategy=CONFIG['imputation']['strategy'],
        sharpness=CONFIG['imputation']['sharpness'],
    )

    matrix_imputed = imputer.apply(matrix_flagged)

    # Report fully-flagged features
    if hasattr(imputer, 'fully_flagged_features_') and imputer.fully_flagged_features_:
        print(f"\n  Fully-flagged features (using global bounds): {len(imputer.fully_flagged_features_)}")
        for feat_id, reason in imputer.fully_flagged_features_:
            print(f"    {feat_id}")
        print(f"    Global bounds: [{imputer.global_lower_:.2f}, {imputer.global_upper_:.2f}]")

    # Verify imputation
    imputed_mask = matrix_flagged.data != matrix_imputed.data
    n_imputed = imputed_mask.sum()
    print(f"    Values imputed: {n_imputed:,}")

    # --- Visualizations ---
    print("\n[5/6] Generating visualizations...")
    output_dir = CONFIG['output']['dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = QCVisualizer(style="paper")

    # Standard summary card
    fig1 = viz.plot_outlier_summary_card(matrix, matrix_imputed)
    fig1.fig.savefig(output_dir / "outlier_summary_card.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: outlier_summary_card.png")

    # Adaptive summary card
    fig2 = viz.plot_adaptive_summary_card(
        matrix, matrix_imputed,
        medcouples=detector.detector_.medcouples_ if hasattr(detector.detector_, 'medcouples_') else None,
        outlier_pvalues=detector.detector_.pvalues_ if hasattr(detector.detector_, 'pvalues_') else None,
        degrees_of_freedom=detector.detector_.degrees_of_freedom_ if hasattr(detector.detector_, 'degrees_of_freedom_') else None,
    )
    fig2.fig.savefig(output_dir / "adaptive_summary_card.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: adaptive_summary_card.png")

    # Skewness adaptation (if medcouple was used)
    if hasattr(detector.detector_, 'medcouples_') and detector.detector_.medcouples_ is not None:
        fig3 = viz.plot_skewness_adaptation(
            detector.detector_.medcouples_,
            detector.detector_.lower_fences_,
            detector.detector_.upper_fences_,
            matrix.feature_ids.values,
        )
        fig3.fig.savefig(output_dir / "skewness_adaptation.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: skewness_adaptation.png")

    # Soft clip transformation
    fig4 = viz.plot_soft_clip_transformation(
        matrix.data, matrix_imputed.data, outlier_mask,
        sharpness=CONFIG['imputation']['sharpness'],
    )
    fig4.fig.savefig(output_dir / "soft_clip_transformation.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: soft_clip_transformation.png")

    # Distribution by stratum
    fig5 = viz.plot_outlier_distribution_by_stratum(matrix, matrix_imputed)
    fig5.fig.savefig(output_dir / "outlier_distribution_by_stratum.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: outlier_distribution_by_stratum.png")

    # Protein vulnerability
    fig6 = viz.plot_protein_vulnerability(outlier_mask, matrix.feature_ids.values)
    fig6.fig.savefig(output_dir / "protein_vulnerability.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: protein_vulnerability.png")

    # --- Save results ---
    print("\n[6/6] Saving imputed data...")

    # Save log2 imputed data
    imputed_df = pd.DataFrame(
        matrix_imputed.data,
        index=matrix.feature_ids,
        columns=matrix.sample_ids,
    )
    imputed_df.to_csv(output_dir / "proteomics_log2_imputed.csv")
    print(f"  Saved: proteomics_log2_imputed.csv")

    # Save metadata
    sample_metadata.to_csv(output_dir / "sample_metadata.csv")
    print(f"  Saved: sample_metadata.csv")

    # Save outlier mask
    outlier_df = pd.DataFrame(
        outlier_mask,
        index=matrix.feature_ids,
        columns=matrix.sample_ids,
    )
    outlier_df.to_csv(output_dir / "outlier_mask.csv")
    print(f"  Saved: outlier_mask.csv")

    # Summary statistics
    fully_flagged = imputer.fully_flagged_features_ if hasattr(imputer, 'fully_flagged_features_') else []
    summary_stats = {
        'n_proteins': matrix.n_features,
        'n_samples': matrix.n_samples,
        'n_case': (sample_metadata['phenotype'] == 'CASE').sum(),
        'n_ctrl': (sample_metadata['phenotype'] == 'CTRL').sum(),
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers,
        'method': CONFIG['detection']['method'],
        'scoring_method': CONFIG['detection']['scoring_method'],
        'impute_strategy': CONFIG['imputation']['strategy'],
        'degrees_of_freedom': detector.detector_.degrees_of_freedom_ if hasattr(detector.detector_, 'degrees_of_freedom_') else None,
        'n_fully_flagged_features': len(fully_flagged),
        'fully_flagged_features': ';'.join([f[0] for f in fully_flagged]) if fully_flagged else '',
        'global_lower_bound': imputer.global_lower_ if hasattr(imputer, 'global_lower_') else None,
        'global_upper_bound': imputer.global_upper_ if hasattr(imputer, 'global_upper_') else None,
    }

    pd.Series(summary_stats).to_csv(output_dir / "imputation_summary.csv")
    print(f"  Saved: imputation_summary.csv")

    # Save fully-flagged features details (if any)
    if fully_flagged:
        flagged_df = pd.DataFrame(fully_flagged, columns=['feature_id', 'note'])
        flagged_df.to_csv(output_dir / "fully_flagged_features.csv", index=False)
        print(f"  Saved: fully_flagged_features.csv")

    print("\n" + "=" * 60)
    print("IMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total outliers: {n_outliers:,} ({pct_outliers:.2f}%)")

if __name__ == '__main__':
    main()
