"""
Pytest configuration and shared fixtures for integration tests.

This module provides test data generators and shared fixtures for all test suites.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag


@pytest.fixture
def test_data_dir():
    """Directory for test data files."""
    path = Path(__file__).parent / "data"
    path.mkdir(exist_ok=True)
    return path


def generate_synthetic_expression_matrix(
    n_genes: int,
    n_samples: int,
    correlation_structure: bool = True,
    outlier_fraction: float = 0.02,
    seed: int = 42
) -> BioMatrix:
    """
    Generate synthetic gene expression matrix with realistic properties.

    Args:
        n_genes: Number of genes (features)
        n_samples: Number of samples
        correlation_structure: If True, create correlated gene modules
        outlier_fraction: Fraction of values to make outliers
        seed: Random seed for reproducibility

    Returns:
        BioMatrix with synthetic expression data

    Design:
        - Generates log-normal distribution (realistic for RNA-seq)
        - Creates correlated gene modules (5-10 genes per module)
        - Injects outliers at specified fraction
        - Adds realistic sample metadata
    """
    rng = np.random.RandomState(seed)

    # Generate base expression (log-normal distribution)
    # Shape: n_genes x n_samples
    data = rng.lognormal(mean=5, sigma=2, size=(n_genes, n_samples))

    # Add correlation structure (gene modules)
    if correlation_structure and n_genes >= 10:
        # Create 5-10 correlated gene modules
        n_modules = min(n_genes // 10, 20)
        module_size = n_genes // n_modules

        for module_idx in range(n_modules):
            # Generate module pattern (shared across genes in module)
            module_pattern = rng.randn(n_samples)

            # Apply pattern to genes in this module
            start_idx = module_idx * module_size
            end_idx = min(start_idx + module_size, n_genes)

            for gene_idx in range(start_idx, end_idx):
                # Add correlated noise to create variation within module
                noise_weight = rng.uniform(0.5, 1.0)
                data[gene_idx, :] = (
                    data[gene_idx, :] * (1 - noise_weight) +
                    module_pattern * noise_weight * data[gene_idx, :].mean()
                )

    # Inject outliers
    n_outliers = int(n_genes * n_samples * outlier_fraction)
    outlier_positions = rng.choice(n_genes * n_samples, size=n_outliers, replace=False)
    outlier_rows = outlier_positions // n_samples
    outlier_cols = outlier_positions % n_samples

    # Make outliers: multiply by 10 or divide by 10 randomly
    for i, j in zip(outlier_rows, outlier_cols):
        if rng.rand() > 0.5:
            data[i, j] *= 10
        else:
            data[i, j] /= 10

    # Create feature IDs (gene names)
    feature_ids = pd.Index([f"GENE_{i:05d}" for i in range(n_genes)])

    # Create sample IDs with phenotype encoding
    sample_ids = []
    phenotypes = []
    for i in range(n_samples):
        phenotype = "CASE" if i % 2 == 0 else "CTRL"
        sample_ids.append(f"{phenotype}-SAMPLE_{i:04d}")
        phenotypes.append(phenotype)

    sample_ids = pd.Index(sample_ids)

    # Create sample metadata
    sample_metadata = pd.DataFrame({
        'phenotype': phenotypes,
        'batch': [i % 5 for i in range(n_samples)]  # 5 batches
    }, index=sample_ids)

    # Initialize quality flags (all ORIGINAL)
    quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=np.uint32)

    return BioMatrix(
        data=data,
        feature_ids=feature_ids,
        sample_ids=sample_ids,
        sample_metadata=sample_metadata,
        quality_flags=quality_flags
    )


@pytest.fixture
def small_matrix():
    """Small test matrix (100 genes x 20 samples) for fast unit tests."""
    return generate_synthetic_expression_matrix(
        n_genes=100,
        n_samples=20,
        correlation_structure=True,
        outlier_fraction=0.02,
        seed=42
    )


@pytest.fixture
def medium_matrix():
    """Medium test matrix (1000 genes x 50 samples) for integration tests."""
    return generate_synthetic_expression_matrix(
        n_genes=1000,
        n_samples=50,
        correlation_structure=True,
        outlier_fraction=0.02,
        seed=42
    )


@pytest.fixture
def large_matrix():
    """Large test matrix (5000 genes x 100 samples) for performance tests."""
    return generate_synthetic_expression_matrix(
        n_genes=5000,
        n_samples=100,
        correlation_structure=True,
        outlier_fraction=0.02,
        seed=42
    )


def save_test_matrix_csv(matrix: BioMatrix, path: Path):
    """
    Save BioMatrix to CSV file for testing file I/O.

    Args:
        matrix: BioMatrix to save
        path: Output CSV path
    """
    # Create DataFrame with feature IDs as index
    df = pd.DataFrame(
        matrix.data,
        index=matrix.feature_ids,
        columns=matrix.sample_ids
    )

    # Save to CSV
    df.to_csv(path)


@pytest.fixture(scope="session")
def test_csv_files(tmp_path_factory):
    """
    Generate test CSV files for all test suites.

    Returns:
        Dictionary with paths to test CSV files
    """
    tmpdir = tmp_path_factory.mktemp("test_data")

    # Generate and save test matrices
    files = {}

    # Small matrix (100 genes)
    small = generate_synthetic_expression_matrix(100, 20, seed=42)
    files['small'] = tmpdir / "test_100genes.csv"
    save_test_matrix_csv(small, files['small'])

    # Medium matrix (1000 genes)
    medium = generate_synthetic_expression_matrix(1000, 50, seed=42)
    files['medium'] = tmpdir / "test_1000genes.csv"
    save_test_matrix_csv(medium, files['medium'])

    # Large matrix (5000 genes)
    large = generate_synthetic_expression_matrix(5000, 100, seed=42)
    files['large'] = tmpdir / "test_5000genes.csv"
    save_test_matrix_csv(large, files['large'])

    return files
