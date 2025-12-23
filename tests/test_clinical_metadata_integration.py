#!/usr/bin/env python
"""
Test script for clinical metadata integration in impute.py

This script validates:
1. Argument parsing for clinical metadata options
2. Clinical enrichment function behavior
3. Integration with existing phenotype extraction
"""

import argparse
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Test data creation
def create_test_clinical_metadata(path: Path):
    """Create test clinical metadata CSV matching the expected format."""
    # NOTE: Participant IDs should match what SubjectIdExtractor extracts from sample IDs
    # SubjectIdExtractor uses pattern r'(NEU[A-Z0-9]+)' which extracts:
    # - CASE_NEUAA001-9001-P_A1 -> NEUAA001
    # - CTRL_NEUBS001-1001-P_E5 -> NEUBS001
    data = {
        'Participant_ID': [
            'NEUAA001',  # Extracted from CASE_NEUAA001-9001-P_A1
            'NEUAA002',
            'NEUAA003',
            'NEUAA004',
            'NEUBS001',  # Extracted from CTRL_NEUBS001-1001-P_E5
            'NEUBS002',
            'NEUBS003',
        ],
        'SUBJECT_GROUP': [
            'ALS',
            'ALS',
            'Non-ALS MND',  # Should be excluded
            'Asymptomatic ALS Gene carrier',  # Should be excluded
            'Healthy Control',
            'Healthy Control',
            'Healthy Control',
        ],
        'SEX': [
            'Male',
            'Female',
            'Male',
            'Female',
            'Male',
            'Female',
            'Male',
        ],
        'Age at Symptom Onset': [55, 62, 48, 71, None, None, None],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Created test clinical metadata: {path}")
    return df


def create_test_expression_matrix(path: Path):
    """Create test expression matrix with sample IDs matching clinical data."""
    # Sample IDs that should match clinical metadata
    # SubjectIdExtractor will extract participant IDs using pattern r'(NEU[A-Z0-9]+)'
    sample_ids = [
        'CASE_NEUAA001-9001-P_A1',   # -> NEUAA001 (ALS, keep)
        'CASE_NEUAA002-9002-P_B2',   # -> NEUAA002 (ALS, keep)
        'CASE_NEUAA003-9003-P_C3',   # -> NEUAA003 (Non-ALS MND, exclude)
        'CASE_NEUAA004-9004-P_D4',   # -> NEUAA004 (Asymptomatic, exclude)
        'CTRL_NEUBS001-1001-P_E5',   # -> NEUBS001 (Healthy Control, keep)
        'CTRL_NEUBS002-1002-P_F6',   # -> NEUBS002 (Healthy Control, keep)
        'CTRL_NEUBS003-1003-P_G7',   # -> NEUBS003 (Healthy Control, keep)
        'CASE_NEUAA999-9999-P_Z9',   # -> NEUAA999 (No clinical data, exclude)
    ]

    # Create random expression data
    n_features = 100
    n_samples = len(sample_ids)
    data = np.random.randn(n_features, n_samples)

    # Create DataFrame
    feature_ids = [f'ENSG0000{i:06d}' for i in range(n_features)]
    df = pd.DataFrame(data, index=feature_ids, columns=sample_ids)

    df.to_csv(path)
    print(f"Created test expression matrix: {path}")
    print(f"  Features: {n_features}, Samples: {n_samples}")
    return df


def test_clinical_metadata_integration():
    """Test the clinical metadata integration feature."""
    print("\n" + "="*70)
    print("Testing Clinical Metadata Integration")
    print("="*70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test data
        clinical_path = tmppath / "clinical.csv"
        expression_path = tmppath / "expression.csv"
        output_path = tmppath / "output"

        create_test_clinical_metadata(clinical_path)
        create_test_expression_matrix(expression_path)

        # Test 1: Check default arguments
        print("\nTest 1: Verify CLI argument defaults")
        print("-" * 70)
        from cliquefinder.cli.impute import register_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register_parser(subparsers)

        # Parse with minimal args
        args = parser.parse_args([
            'impute',
            '--input', str(expression_path),
            '--output', str(output_path),
        ])

        assert args.clinical_metadata is None, "Default clinical_metadata should be None"
        assert args.clinical_id_col == "Participant_ID", "Default clinical_id_col incorrect"
        assert args.phenotype_source_col == "SUBJECT_GROUP", "Default phenotype_source_col incorrect"
        assert args.case_values == ["ALS"], "Default case_values incorrect"
        assert args.ctrl_values == ["Healthy Control"], "Default ctrl_values incorrect"
        print("  ✓ All default arguments correct")

        # Test 2: Check custom arguments
        print("\nTest 2: Verify custom CLI arguments")
        print("-" * 70)
        args = parser.parse_args([
            'impute',
            '--input', str(expression_path),
            '--output', str(output_path),
            '--clinical-metadata', str(clinical_path),
            '--clinical-id-col', 'Participant_ID',
            '--phenotype-source-col', 'SUBJECT_GROUP',
            '--case-values', 'ALS', 'Progressive Muscular Atrophy',
            '--ctrl-values', 'Healthy Control', 'Non-diseased Control',
        ])

        assert args.clinical_metadata == clinical_path, "Clinical metadata path incorrect"
        assert args.case_values == ['ALS', 'Progressive Muscular Atrophy'], "Case values incorrect"
        assert args.ctrl_values == ['Healthy Control', 'Non-diseased Control'], "Ctrl values incorrect"
        print("  ✓ All custom arguments parsed correctly")

        # Test 3: Test the enrichment function
        print("\nTest 3: Test _enrich_with_clinical_metadata function")
        print("-" * 70)

        from cliquefinder.io.loaders import load_csv_matrix
        from cliquefinder.cli.impute import _enrich_with_clinical_metadata

        # Load expression matrix
        matrix = load_csv_matrix(expression_path)
        print(f"  Loaded matrix: {matrix.n_features} features x {matrix.n_samples} samples")

        # Enrich with clinical metadata
        enriched = _enrich_with_clinical_metadata(
            matrix,
            clinical_path=clinical_path,
            clinical_id_col='Participant_ID',
            phenotype_source_col='SUBJECT_GROUP',
            case_values=['ALS'],
            ctrl_values=['Healthy Control'],
            phenotype_col='phenotype',
        )

        print(f"\n  After enrichment: {enriched.n_features} features x {enriched.n_samples} samples")

        # Verify results
        assert 'phenotype' in enriched.sample_metadata.columns, "Phenotype column missing"
        assert 'SEX' in enriched.sample_metadata.columns, "SEX column missing"

        # Count phenotypes
        phenotype_counts = enriched.sample_metadata['phenotype'].value_counts()
        n_case = phenotype_counts.get('CASE', 0)
        n_ctrl = phenotype_counts.get('CTRL', 0)

        print(f"  Phenotype distribution: CASE={n_case}, CTRL={n_ctrl}")

        # Expected: 2 CASE (ALS only), 3 CTRL (Healthy Control)
        # Excluded: 2 CASE (Non-ALS MND, Asymptomatic), 1 no clinical data
        assert n_case == 2, f"Expected 2 CASE samples, got {n_case}"
        assert n_ctrl == 3, f"Expected 3 CTRL samples, got {n_ctrl}"
        assert enriched.n_samples == 5, f"Expected 5 total samples, got {enriched.n_samples}"

        print("  ✓ Phenotype mapping correct (2 ALS, 3 Healthy Control)")
        print("  ✓ Excluded samples: 2 Non-ALS MND/Asymptomatic, 1 no clinical data")

        # Verify SEX column
        sex_counts = enriched.sample_metadata['SEX'].value_counts()
        print(f"  SEX distribution: {dict(sex_counts)}")

        # Test 4: Test backward compatibility (no clinical metadata)
        print("\nTest 4: Test backward compatibility (no clinical metadata)")
        print("-" * 70)

        from cliquefinder.cli.impute import _ensure_phenotype_metadata, _extract_phenotype_from_sample_id

        # Test the extraction function directly (most reliable test)
        test_samples = ['CASE_TEST', 'CTRL_TEST', 'OTHER']
        for sample_id in test_samples:
            extracted = _extract_phenotype_from_sample_id(sample_id)
            expected = 'CASE' if sample_id.startswith('CASE') else ('CTRL' if sample_id.startswith('CTRL') else None)
            assert extracted == expected, f"Extraction failed for {sample_id}: got {extracted}, expected {expected}"

        print("  ✓ Phenotype extraction function works correctly")

        # Test _ensure_phenotype_metadata by creating a clean matrix without loader interference
        from cliquefinder import BioMatrix
        import pandas as pd

        clean_sample_ids = ['CASE_A', 'CASE_B', 'CTRL_C']
        clean_data = np.random.randn(10, 3)
        clean_metadata = pd.DataFrame(index=pd.Index(clean_sample_ids, name='sample_id'))
        clean_flags = np.zeros((10, 3), dtype=np.uint32)  # No quality flags

        clean_matrix = BioMatrix(
            data=clean_data,
            feature_ids=pd.Index([f'Gene{i}' for i in range(10)]),
            sample_ids=pd.Index(clean_sample_ids),
            sample_metadata=clean_metadata,
            quality_flags=clean_flags,
        )

        enriched_clean = _ensure_phenotype_metadata(clean_matrix, 'phenotype')
        pheno_counts = enriched_clean.sample_metadata['phenotype'].value_counts()

        assert pheno_counts.get('CASE', 0) == 2, f"Expected 2 CASE, got {pheno_counts.get('CASE', 0)}"
        assert pheno_counts.get('CTRL', 0) == 1, f"Expected 1 CTRL, got {pheno_counts.get('CTRL', 0)}"

        print("  ✓ Backward compatibility maintained - phenotype extraction works without clinical metadata")

        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70 + "\n")


if __name__ == '__main__':
    test_clinical_metadata_integration()
