"""Tests for correlation sign preservation in clique finding."""

import pytest
import numpy as np
import pandas as pd
from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.knowledge.clique_validator import (
    CliqueValidator,
    CorrelationClique,
    CorrelationDirection,
)


def create_test_matrix(correlations: dict, n_samples: int = 200, seed: int = 42) -> BioMatrix:
    """
    Create a test BioMatrix with specified inter-gene correlations.

    Args:
        correlations: Dict mapping (gene1, gene2) tuples to desired correlation values
        n_samples: Number of samples to generate (default: 200 for stable correlations)
        seed: Random seed for reproducibility

    Returns:
        BioMatrix with genes having approximately specified correlations

    Notes:
        Uses an iterative approach that works for both positive and negative correlations.
        With n_samples=200, achieves correlations within ~0.05 of target values.
        For perfectly reproducible results, uses fixed random seed.
    """
    np.random.seed(seed)

    # Extract unique genes
    genes = sorted(set(g for pair in correlations.keys() for g in pair))
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Start with independent random data
    data = np.random.randn(n_genes, n_samples)

    # Iteratively adjust to achieve target correlations
    # Use multiple iterations to converge
    for iteration in range(100):
        # Compute current correlations
        current_corr = np.corrcoef(data)

        # For each target correlation, adjust the data
        converged = True
        for (g1, g2), target_r in correlations.items():
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            current_r = current_corr[i, j]
            error = target_r - current_r

            if abs(error) > 0.01:  # Tolerance
                converged = False
                # Adjust gene j towards/away from gene i
                alpha = 0.05  # Learning rate
                data[j, :] = data[j, :] + alpha * error * data[i, :]
                # Re-standardize
                data[j, :] = (data[j, :] - data[j, :].mean()) / data[j, :].std()

        if converged:
            break

    # Create BioMatrix
    feature_ids = pd.Index(genes)
    sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
    metadata = pd.DataFrame({'condition': ['A'] * n_samples}, index=sample_ids)
    quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=int)

    return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)


class TestCorrelationSignPreservation:
    """Test suite for correlation sign preservation."""

    def test_positive_clique_direction(self):
        """Test that a clique with all positive correlations is classified as POSITIVE."""
        # Create genes with high positive correlations
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,
            ('GENE_A', 'GENE_C'): 0.80,
            ('GENE_B', 'GENE_C'): 0.90,
        }
        matrix = create_test_matrix(correlations, n_samples=100)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        cliques = validator.find_cliques(
            genes={'GENE_A', 'GENE_B', 'GENE_C'},
            condition='all',
            min_correlation=0.7,
            min_clique_size=3,
        )

        assert len(cliques) >= 1, "Expected at least one clique with positive correlations"
        clique = cliques[0]

        # Check direction
        assert clique.direction == CorrelationDirection.POSITIVE, \
            f"Expected POSITIVE direction, got {clique.direction}"

        # Check signed stats are positive
        assert clique.signed_mean_correlation > 0, \
            f"Expected positive signed_mean_correlation, got {clique.signed_mean_correlation}"
        assert clique.signed_min_correlation > 0, \
            f"Expected positive signed_min_correlation, got {clique.signed_min_correlation}"

        # Check edge counts
        assert clique.n_positive_edges == 3, \
            f"Expected 3 positive edges in triangle, got {clique.n_positive_edges}"
        assert clique.n_negative_edges == 0, \
            f"Expected 0 negative edges, got {clique.n_negative_edges}"

    def test_negative_clique_direction(self):
        """Test that a clique with all negative correlations is classified as NEGATIVE.

        Note: We use a 2-node clique here since having 3+ variables with all pairwise
        negative correlations is mathematically impossible (would require negative
        definite correlation matrix, which doesn't exist).
        """
        # Create genes with high negative correlation (2 nodes only)
        correlations = {
            ('GENE_A', 'GENE_B'): -0.85,
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B'},
            condition='all',
            min_correlation=0.7,  # Uses absolute value threshold
        )

        assert clique is not None, "Expected to find a clique with negative correlation"

        # Check direction
        assert clique.direction == CorrelationDirection.NEGATIVE, \
            f"Expected NEGATIVE direction, got {clique.direction}"

        # Check signed stats are negative
        assert clique.signed_mean_correlation < 0, \
            f"Expected negative signed_mean_correlation, got {clique.signed_mean_correlation}"
        assert clique.signed_max_correlation < 0, \
            f"Expected negative signed_max_correlation, got {clique.signed_max_correlation}"

        # Check edge counts
        assert clique.n_positive_edges == 0, \
            f"Expected 0 positive edges, got {clique.n_positive_edges}"
        assert clique.n_negative_edges == 1, \
            f"Expected 1 negative edge, got {clique.n_negative_edges}"

    def test_mixed_clique_direction(self):
        """Test that a clique with mixed correlations is classified as MIXED."""
        # Create genes with mixed positive/negative correlations
        # Use a 4-node pattern that's mathematically feasible
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,   # positive
            ('GENE_A', 'GENE_C'): 0.80,   # positive
            ('GENE_A', 'GENE_D'): -0.75,  # negative
            ('GENE_B', 'GENE_C'): 0.88,   # positive
            ('GENE_B', 'GENE_D'): -0.78,  # negative
            ('GENE_C', 'GENE_D'): -0.82,  # negative
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a clique"
        assert clique.size == 4, f"Expected 4 nodes, got {clique.size}"

        # Check direction
        assert clique.direction == CorrelationDirection.MIXED, \
            f"Expected MIXED direction, got {clique.direction}"

        # Check edge counts - should have both positive and negative
        assert clique.n_positive_edges > 0, \
            f"Expected some positive edges, got {clique.n_positive_edges}"
        assert clique.n_negative_edges > 0, \
            f"Expected some negative edges, got {clique.n_negative_edges}"
        assert clique.n_positive_edges + clique.n_negative_edges == 6, \
            f"Expected 6 total edges in 4-node clique, got {clique.n_positive_edges + clique.n_negative_edges}"

    def test_backward_compatibility_absolute_values(self):
        """Test that mean_correlation and min_correlation remain absolute values."""
        # Use a single negative edge for simplicity
        correlations = {
            ('GENE_A', 'GENE_B'): -0.85,
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a clique"

        # Backward compatible fields should be POSITIVE (absolute values)
        assert clique.mean_correlation > 0, \
            f"Expected positive mean_correlation (absolute), got {clique.mean_correlation}"
        assert clique.min_correlation > 0, \
            f"Expected positive min_correlation (absolute), got {clique.min_correlation}"

        # But signed fields should be NEGATIVE
        assert clique.signed_mean_correlation < 0, \
            f"Expected negative signed_mean_correlation, got {clique.signed_mean_correlation}"

    def test_find_maximum_clique_preserves_sign(self):
        """Test that find_maximum_clique also preserves correlation signs."""
        # Test with positive clique (feasible for 3+ nodes)
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,
            ('GENE_A', 'GENE_C'): 0.80,
            ('GENE_B', 'GENE_C'): 0.90,
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a maximum clique"
        assert clique.direction == CorrelationDirection.POSITIVE, \
            f"Expected POSITIVE direction, got {clique.direction}"
        assert clique.signed_mean_correlation > 0, \
            f"Expected positive signed_mean_correlation, got {clique.signed_mean_correlation}"


class TestEdgeCases:
    """Test edge cases for sign preservation."""

    def test_single_edge_positive(self):
        """Test a 2-node clique with positive correlation."""
        correlations = {('GENE_A', 'GENE_B'): 0.85}
        matrix = create_test_matrix(correlations, n_samples=100)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a 2-node clique"
        assert clique.n_positive_edges == 1, \
            f"Expected 1 positive edge, got {clique.n_positive_edges}"
        assert clique.n_negative_edges == 0, \
            f"Expected 0 negative edges, got {clique.n_negative_edges}"
        assert clique.direction == CorrelationDirection.POSITIVE, \
            f"Expected POSITIVE direction, got {clique.direction}"

    def test_single_edge_negative(self):
        """Test a 2-node clique with negative correlation."""
        correlations = {('GENE_A', 'GENE_B'): -0.85}
        matrix = create_test_matrix(correlations, n_samples=100)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a 2-node clique"
        assert clique.n_positive_edges == 0, \
            f"Expected 0 positive edges, got {clique.n_positive_edges}"
        assert clique.n_negative_edges == 1, \
            f"Expected 1 negative edge, got {clique.n_negative_edges}"
        assert clique.direction == CorrelationDirection.NEGATIVE, \
            f"Expected NEGATIVE direction, got {clique.direction}"

    def test_larger_positive_clique(self):
        """Test a larger clique (4 nodes) with all positive correlations."""
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,
            ('GENE_A', 'GENE_C'): 0.80,
            ('GENE_A', 'GENE_D'): 0.82,
            ('GENE_B', 'GENE_C'): 0.88,
            ('GENE_B', 'GENE_D'): 0.83,
            ('GENE_C', 'GENE_D'): 0.86,
        }
        matrix = create_test_matrix(correlations, n_samples=100)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a 4-node clique"
        assert clique.size == 4, f"Expected 4 nodes, got {clique.size}"
        # 4 nodes form 6 edges: C(4,2) = 6
        assert clique.n_positive_edges == 6, \
            f"Expected 6 positive edges, got {clique.n_positive_edges}"
        assert clique.n_negative_edges == 0, \
            f"Expected 0 negative edges, got {clique.n_negative_edges}"
        assert clique.direction == CorrelationDirection.POSITIVE, \
            f"Expected POSITIVE direction, got {clique.direction}"

    def test_larger_mixed_clique(self):
        """Test a larger clique (4 nodes) with mixed positive/negative correlations."""
        # A feasible mixed correlation pattern
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,   # positive
            ('GENE_A', 'GENE_C'): 0.80,   # positive
            ('GENE_A', 'GENE_D'): -0.75,  # negative
            ('GENE_B', 'GENE_C'): 0.88,   # positive
            ('GENE_B', 'GENE_D'): -0.78,  # negative
            ('GENE_C', 'GENE_D'): -0.82,  # negative
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a 4-node clique"
        assert clique.size == 4, f"Expected 4 nodes, got {clique.size}"
        # 4 nodes form 6 edges: C(4,2) = 6
        # We should have mix of positive and negative
        assert clique.n_positive_edges > 0, \
            f"Expected some positive edges, got {clique.n_positive_edges}"
        assert clique.n_negative_edges > 0, \
            f"Expected some negative edges, got {clique.n_negative_edges}"
        assert clique.n_positive_edges + clique.n_negative_edges == 6, \
            f"Expected 6 total edges, got {clique.n_positive_edges + clique.n_negative_edges}"
        assert clique.direction == CorrelationDirection.MIXED, \
            f"Expected MIXED direction, got {clique.direction}"


class TestChildSetType2SignPreservation:
    """Test correlation sign preservation in ChildSetType2 (coherent modules)."""

    def test_child_set_type2_positive(self):
        """Test that ChildSetType2 preserves positive correlation signs."""
        correlations = {
            ('TARGET_1', 'TARGET_2'): 0.85,
            ('TARGET_1', 'TARGET_3'): 0.80,
            ('TARGET_2', 'TARGET_3'): 0.88,
        }
        matrix = create_test_matrix(correlations, n_samples=100)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        coherent = validator.get_child_set_type2(
            regulator_name='REG1',
            indra_targets={'TARGET_1', 'TARGET_2', 'TARGET_3'},
            condition='all',
            min_correlation=0.7,
        )

        assert coherent is not None, "Expected to find a coherent module"
        assert coherent.direction == CorrelationDirection.POSITIVE, \
            f"Expected POSITIVE direction, got {coherent.direction}"
        assert coherent.signed_mean_correlation > 0, \
            f"Expected positive signed_mean_correlation, got {coherent.signed_mean_correlation}"
        assert coherent.n_positive_edges == 3, \
            f"Expected 3 positive edges, got {coherent.n_positive_edges}"
        assert coherent.n_negative_edges == 0, \
            f"Expected 0 negative edges, got {coherent.n_negative_edges}"

    def test_child_set_type2_negative(self):
        """Test that ChildSetType2 preserves negative correlation signs."""
        # Use a 2-node clique since all-negative 3+ nodes is impossible
        correlations = {
            ('TARGET_1', 'TARGET_2'): -0.85,
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        coherent = validator.get_child_set_type2(
            regulator_name='REG1',
            indra_targets={'TARGET_1', 'TARGET_2'},
            condition='all',
            min_correlation=0.7,
        )

        assert coherent is not None, "Expected to find a coherent module"
        assert coherent.direction == CorrelationDirection.NEGATIVE, \
            f"Expected NEGATIVE direction, got {coherent.direction}"
        assert coherent.signed_mean_correlation < 0, \
            f"Expected negative signed_mean_correlation, got {coherent.signed_mean_correlation}"
        assert coherent.n_positive_edges == 0, \
            f"Expected 0 positive edges, got {coherent.n_positive_edges}"
        assert coherent.n_negative_edges == 1, \
            f"Expected 1 negative edge, got {coherent.n_negative_edges}"


class TestSignedStatisticsConsistency:
    """Test consistency between signed and absolute statistics."""

    def test_signed_stats_magnitude_matches_absolute(self):
        """Test that |signed_mean_correlation| ≈ mean_correlation for uniform sign."""
        # Use all positive for uniform sign test (feasible for 3+ nodes)
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,
            ('GENE_A', 'GENE_C'): 0.80,
            ('GENE_B', 'GENE_C'): 0.90,
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a clique"

        # For uniform sign (all positive), magnitude should match
        assert abs(clique.signed_mean_correlation - clique.mean_correlation) < 0.05, \
            f"Expected signed_mean ≈ mean_correlation for uniform positive sign, got {clique.signed_mean_correlation} vs {clique.mean_correlation}"

    def test_mixed_clique_signed_mean_range(self):
        """Test that signed_mean is in reasonable range for mixed correlations."""
        # Use a feasible mixed pattern
        correlations = {
            ('GENE_A', 'GENE_B'): 0.85,   # positive
            ('GENE_A', 'GENE_C'): 0.80,   # positive
            ('GENE_A', 'GENE_D'): -0.75,  # negative
            ('GENE_B', 'GENE_C'): 0.88,   # positive
            ('GENE_B', 'GENE_D'): -0.78,  # negative
            ('GENE_C', 'GENE_D'): -0.82,  # negative
        }
        matrix = create_test_matrix(correlations, n_samples=200)

        validator = CliqueValidator(matrix, stratify_by=None, min_samples=5, precompute=False)
        clique = validator.find_maximum_clique(
            genes={'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'},
            condition='all',
            min_correlation=0.7,
        )

        assert clique is not None, "Expected to find a clique"
        assert clique.size == 4, f"Expected 4 nodes, got {clique.size}"
        assert clique.direction == CorrelationDirection.MIXED, \
            f"Expected MIXED direction, got {clique.direction}"

        # For mixed correlations, signed_mean should be between signed_min and signed_max
        assert clique.signed_min_correlation <= clique.signed_mean_correlation <= clique.signed_max_correlation, \
            f"Expected signed_min ({clique.signed_min_correlation}) <= signed_mean ({clique.signed_mean_correlation}) <= signed_max ({clique.signed_max_correlation})"

        # Signed mean should have smaller magnitude than absolute mean (cancellation effect)
        assert abs(clique.signed_mean_correlation) < clique.mean_correlation, \
            f"Expected |signed_mean| < mean_correlation for mixed signs, got {abs(clique.signed_mean_correlation)} vs {clique.mean_correlation}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
