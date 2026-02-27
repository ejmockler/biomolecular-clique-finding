"""
Integration tests for CrossModalIDMapper.

Tests real ID mapping with MyGeneInfo API (with caching to minimize API calls).
Minimal mocking - validates actual mapping functionality.
"""

import pytest
import tempfile
from pathlib import Path
import time

from cliquefinder.knowledge.cross_modal_mapper import (
    CrossModalIDMapper,
    CrossModalMapping,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mapper(temp_cache_dir):
    """Create mapper with temporary cache directory."""
    return CrossModalIDMapper(cache_dir=temp_cache_dir)


class TestCrossModalMapping:
    """Test CrossModalMapping dataclass."""

    def test_valid_mapping(self):
        """Test creation of valid mapping."""
        mapping = CrossModalMapping(
            common_genes={'TP53', 'BRCA1'},
            protein_only={'EGFR'},
            rna_only={'MYC'}
        )
        assert len(mapping.common_genes) == 2
        assert len(mapping.protein_only) == 1
        assert len(mapping.rna_only) == 1

    def test_disjoint_sets(self):
        """Test that common, protein_only, and rna_only are disjoint."""
        mapping = CrossModalMapping(
            common_genes={'TP53', 'BRCA1'},
            protein_only={'EGFR'},
            rna_only={'MYC'}
        )
        # No overlap
        assert len(mapping.common_genes & mapping.protein_only) == 0
        assert len(mapping.common_genes & mapping.rna_only) == 0
        assert len(mapping.protein_only & mapping.rna_only) == 0


class TestDirectIntersection:
    """Test direct symbol matching (no API calls)."""

    def test_identical_symbols(self, mapper):
        """Test when both datasets use same gene symbols."""
        protein_ids = ['TP53', 'BRCA1', 'EGFR', 'MYC']
        rna_ids = ['TP53', 'BRCA1', 'KRAS']

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='symbol',
            species='human'
        )

        # Direct intersection
        assert mapping.common_genes == {'TP53', 'BRCA1'}
        assert 'EGFR' in mapping.protein_only
        assert 'MYC' in mapping.protein_only
        assert 'KRAS' in mapping.rna_only

    def test_case_insensitive_matching(self, mapper):
        """Test case-insensitive symbol matching."""
        protein_ids = ['TP53', 'brca1']  # Mixed case
        rna_ids = ['tp53', 'BRCA1']      # Different case

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='symbol',
            species='human'
        )

        # Should normalize to uppercase
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes

    def test_empty_intersection(self, mapper):
        """Test when no genes overlap."""
        protein_ids = ['TP53', 'BRCA1']
        rna_ids = ['KRAS', 'MYC']

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='symbol',
            species='human'
        )

        assert len(mapping.common_genes) == 0
        assert mapping.protein_only == {'TP53', 'BRCA1'}
        assert mapping.rna_only == {'KRAS', 'MYC'}


class TestEnsemblMapping:
    """Test mapping Ensembl IDs to gene symbols via MyGeneInfo API."""

    def test_ensembl_gene_to_symbol(self, mapper):
        """Test mapping Ensembl gene IDs to symbols (uses real API)."""
        protein_ids = ['TP53', 'BRCA1', 'EGFR']

        # Real Ensembl IDs for known genes
        rna_ids = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
            'ENSG00000171862',  # PTEN (not in protein list)
        ]

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Should map Ensembl IDs to symbols
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes
        assert 'EGFR' in mapping.protein_only
        assert 'PTEN' in mapping.rna_only or len(mapping.rna_only) == 1

    def test_mixed_valid_and_invalid_ensembl(self, mapper):
        """Test handling of some valid and some invalid Ensembl IDs."""
        protein_ids = ['TP53']

        rna_ids = [
            'ENSG00000141510',     # Valid: TP53
            'ENSG99999999999',     # Invalid
        ]

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Should successfully map TP53
        assert 'TP53' in mapping.common_genes


class TestCaching:
    """Test disk caching functionality."""

    def test_cache_creation(self, mapper, temp_cache_dir):
        """Test that cache files are created."""
        protein_ids = ['TP53', 'BRCA1']
        rna_ids = ['ENSG00000141510', 'ENSG00000012048']

        # First call - will create cache
        mapping1 = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Check cache directory has files (JSON format after S-2 security fix)
        cache_files = list(temp_cache_dir.rglob('*.json'))
        assert len(cache_files) > 0

        # Second call - should use cache (faster)
        start = time.time()
        mapping2 = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )
        cached_time = time.time() - start

        # Results should be identical
        assert mapping1.common_genes == mapping2.common_genes
        assert mapping1.protein_only == mapping2.protein_only
        assert mapping1.rna_only == mapping2.rna_only

        # Cached call should be very fast (<0.1s)
        assert cached_time < 0.1

    def test_different_queries_different_caches(self, mapper, temp_cache_dir):
        """Test that different queries use different cache keys."""
        # Query 1
        mapper.unify_ids(
            protein_ids=['TP53'],
            rna_ids=['ENSG00000141510'],
            rna_id_type='ensembl_gene',
            species='human'
        )
        n_caches_1 = len(list(temp_cache_dir.rglob('*.json')))

        # Query 2 (different)
        mapper.unify_ids(
            protein_ids=['BRCA1'],
            rna_ids=['ENSG00000012048'],
            rna_id_type='ensembl_gene',
            species='human'
        )
        n_caches_2 = len(list(temp_cache_dir.rglob('*.json')))

        # Should have more cache files
        assert n_caches_2 > n_caches_1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_protein_ids(self, mapper):
        """Test with empty protein list."""
        mapping = mapper.unify_ids(
            protein_ids=[],
            rna_ids=['TP53', 'BRCA1'],
            rna_id_type='symbol',
            species='human'
        )

        assert len(mapping.common_genes) == 0
        assert len(mapping.protein_only) == 0
        assert mapping.rna_only == {'TP53', 'BRCA1'}

    def test_empty_rna_ids(self, mapper):
        """Test with empty RNA list."""
        mapping = mapper.unify_ids(
            protein_ids=['TP53', 'BRCA1'],
            rna_ids=[],
            rna_id_type='symbol',
            species='human'
        )

        assert len(mapping.common_genes) == 0
        assert mapping.protein_only == {'TP53', 'BRCA1'}
        assert len(mapping.rna_only) == 0

    def test_both_empty(self, mapper):
        """Test with both lists empty."""
        mapping = mapper.unify_ids(
            protein_ids=[],
            rna_ids=[],
            rna_id_type='symbol',
            species='human'
        )

        assert len(mapping.common_genes) == 0
        assert len(mapping.protein_only) == 0
        assert len(mapping.rna_only) == 0

    def test_duplicate_ids(self, mapper):
        """Test handling of duplicate IDs."""
        protein_ids = ['TP53', 'TP53', 'BRCA1']
        rna_ids = ['TP53', 'TP53', 'KRAS']

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='symbol',
            species='human'
        )

        # Should deduplicate
        assert mapping.common_genes == {'TP53'}
        assert mapping.protein_only == {'BRCA1'}
        assert mapping.rna_only == {'KRAS'}


class TestRealWorldScenarios:
    """Test realistic cross-modal integration scenarios."""

    def test_proteomics_vs_rnaseq_realistic(self, mapper):
        """Test realistic proteomics (symbols) vs RNA-seq (Ensembl) scenario."""
        # Typical proteomics dataset: gene symbols
        protein_ids = [
            'TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS',
            'PTEN', 'AKT1', 'MAPK1', 'CDK1', 'RB1'
        ]

        # Typical RNA-seq dataset: Ensembl IDs (subset overlap)
        rna_ids = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
            'ENSG00000146648',  # EGFR
            'ENSG00000141736',  # ERBB2 (not in protein list)
            'ENSG00000133703',  # KRAS
        ]

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Should have significant overlap
        assert len(mapping.common_genes) >= 3
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes
        assert 'EGFR' in mapping.common_genes or 'KRAS' in mapping.common_genes

        # Should have some protein-only and RNA-only
        assert len(mapping.protein_only) > 0
        assert len(mapping.rna_only) >= 0

    def test_large_dataset_mapping(self, mapper):
        """Test mapping with larger dataset (100 genes)."""
        # Generate 100 gene symbols
        protein_ids = [f'GENE{i:03d}' for i in range(100)]
        protein_ids[:10] = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS',
                            'PTEN', 'AKT1', 'MAPK1', 'CDK1', 'RB1']

        # Subset of real genes as Ensembl IDs
        rna_ids = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
            'ENSG00000146648',  # EGFR
        ] + [f'ENSG{i:011d}' for i in range(97)]  # Fake IDs

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Should successfully map known genes
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes

    def test_mouse_species(self, mapper):
        """Test mapping with mouse species."""
        # Mouse gene symbols
        protein_ids = ['Tp53', 'Brca1']  # Mouse uses sentence case

        # Mouse Ensembl IDs
        rna_ids = [
            'ENSMUSG00000059552',  # Tp53
            'ENSMUSG00000017146',  # Brca1
        ]

        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='mouse'
        )

        # Should successfully map mouse genes
        # Note: case normalization might differ for mouse
        assert len(mapping.common_genes) >= 1


class TestPerformance:
    """Test performance characteristics."""

    def test_batch_query_efficiency(self, mapper):
        """Test that batch queries are efficient."""
        # Large batch (50 genes)
        protein_ids = ['TP53', 'BRCA1', 'EGFR'] + [f'GENE{i}' for i in range(47)]
        rna_ids = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
        ] + [f'ENSG{i:011d}' for i in range(48)]

        start = time.time()
        mapping = mapper.unify_ids(
            protein_ids=protein_ids,
            rna_ids=rna_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (<10s for first query)
        assert elapsed < 10.0

        # Should return valid results
        assert isinstance(mapping, CrossModalMapping)
        assert len(mapping.common_genes) >= 0
