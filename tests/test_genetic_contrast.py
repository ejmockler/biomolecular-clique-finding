#!/usr/bin/env python3
"""
Test script for genetic subtype contrast functionality.

Tests the derive_genetic_phenotype function with sample metadata.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cliquefinder.cli.differential import derive_genetic_phenotype


def test_c9orf72_contrast():
    """Test C9orf72 vs sporadic ALS contrast."""

    # Create sample metadata
    metadata = pd.DataFrame({
        'phenotype': ['CASE'] * 10 + ['CTRL'] * 5,
        'ClinReport_Mutations_Details': [
            'C9orf72', 'C9orf72', 'C9orf72',  # 3 C9orf72 carriers
            'SOD1', 'SOD1',  # 2 SOD1 carriers
            None, None, None, None, None,  # 5 sporadic ALS
            None, None, None, None, None,  # 5 healthy controls (will be filtered)
        ],
        'subject_id': [f'S{i:03d}' for i in range(15)],
    })
    metadata.index = metadata['subject_id']

    print("Input metadata:")
    print(metadata[['phenotype', 'ClinReport_Mutations_Details']].to_string())
    print()

    # Test C9orf72 contrast
    print("=" * 70)
    print("Testing C9orf72 vs sporadic ALS contrast")
    print("=" * 70)

    filtered_meta, carrier_label, sporadic_label = derive_genetic_phenotype(
        metadata=metadata,
        mutation='C9orf72',
    )

    print("\nFiltered metadata:")
    print(filtered_meta[['phenotype', 'ClinReport_Mutations_Details', 'genetic_phenotype']].to_string())
    print()

    # Validate results
    assert carrier_label == 'C9ORF72', f"Expected carrier_label='C9ORF72', got '{carrier_label}'"
    assert sporadic_label == 'SPORADIC', f"Expected sporadic_label='SPORADIC', got '{sporadic_label}'"

    n_carriers = (filtered_meta['genetic_phenotype'] == carrier_label).sum()
    n_sporadic = (filtered_meta['genetic_phenotype'] == sporadic_label).sum()

    assert n_carriers == 3, f"Expected 3 C9orf72 carriers, got {n_carriers}"
    assert n_sporadic == 5, f"Expected 5 sporadic ALS, got {n_sporadic}"
    assert len(filtered_meta) == 8, f"Expected 8 total samples, got {len(filtered_meta)}"

    # Verify controls were filtered out
    assert 'CTRL' not in filtered_meta['phenotype'].values, "Controls should be filtered out"

    # Verify SOD1 carriers are excluded from both groups
    assert 'SOD1' not in filtered_meta['ClinReport_Mutations_Details'].values, \
        "SOD1 carriers should be filtered out (not in either contrast group)"

    print("\nValidation passed:")
    print(f"  ✓ {n_carriers} C9orf72 carriers")
    print(f"  ✓ {n_sporadic} sporadic ALS")
    print(f"  ✓ Controls excluded")
    print(f"  ✓ Other mutation carriers excluded")
    print()


def test_underpowered_warning():
    """Test warning messages for small sample sizes."""

    # Small sample metadata
    metadata = pd.DataFrame({
        'phenotype': ['CASE'] * 6 + ['CTRL'] * 2,
        'ClinReport_Mutations_Details': [
            'C9orf72', 'C9orf72',  # Only 2 C9orf72 carriers
            None, None, None, None,  # 4 sporadic
            None, None,  # 2 controls
        ],
        'subject_id': [f'S{i:03d}' for i in range(8)],
    })
    metadata.index = metadata['subject_id']

    print("=" * 70)
    print("Testing underpowered sample size warnings")
    print("=" * 70)

    # Should print warnings about small sample size
    filtered_meta, _, _ = derive_genetic_phenotype(
        metadata=metadata,
        mutation='C9orf72',
    )

    print("\nExpected warnings about n<10 and n<30")
    print()


def test_invalid_mutation():
    """Test error handling for invalid mutation names."""

    metadata = pd.DataFrame({
        'phenotype': ['CASE'] * 5,
        'ClinReport_Mutations_Details': ['C9orf72', None, None, None, None],
        'subject_id': [f'S{i:03d}' for i in range(5)],
    })
    metadata.index = metadata['subject_id']

    print("=" * 70)
    print("Testing invalid mutation name")
    print("=" * 70)

    try:
        derive_genetic_phenotype(metadata=metadata, mutation='INVALID_MUTATION')
        assert False, "Should have raised ValueError for invalid mutation"
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        print()


if __name__ == '__main__':
    print("\n")
    print("=" * 70)
    print("  Testing Genetic Subtype Contrast Functionality")
    print("=" * 70)
    print()

    test_c9orf72_contrast()
    test_underpowered_warning()
    test_invalid_mutation()

    print("=" * 70)
    print("  All tests passed!")
    print("=" * 70)
    print()
