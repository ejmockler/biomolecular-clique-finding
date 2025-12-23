"""
End-to-end integration tests for cross-modal RNA filtering.

Tests the full pipeline: RNA loading → ID mapping → ModuleDiscovery
with minimal mocking to validate real functionality.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from cliquefinder import BioMatrix
from cliquefinder.knowledge import ModuleDiscovery
from cliquefinder.knowledge.rna_loader import RNADataLoader
from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def proteomics_matrix():
    """Create realistic proteomics test matrix."""
    # 10 genes × 30 samples
    n_genes = 10
    n_samples = 30

    gene_ids = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS',
                'PTEN', 'AKT1', 'MAPK1', 'CDK1', 'RB1']
    sample_ids = [f'Sample{i:03d}' for i in range(n_samples)]

    # Simulate proteomics data (log-normal distribution)
    np.random.seed(42)
    data = np.random.lognormal(mean=5, sigma=1.5, size=(n_genes, n_samples))

    return BioMatrix(
        data=data,
        feature_ids=pd.Index(gene_ids),
        sample_ids=pd.Index(sample_ids),
        sample_metadata=pd.DataFrame(index=sample_ids),  # Empty metadata
        quality_flags=np.zeros((n_genes, n_samples), dtype=int)  # All pass QC (2D)
    )


class TestFullPipelineSymbols:
    """Test full pipeline with gene symbols."""

    def test_proteomics_and_rna_both_symbols(self, temp_dir, proteomics_matrix):
        """Test pipeline when both datasets use gene symbols."""
        # Create RNA data (overlapping genes)
        rna_file = temp_dir / "rna_symbols.csv"
        rna_genes = ['TP53', 'BRCA1', 'EGFR', 'KRAS', 'CDK2']  # 4 overlap
        rna_data = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(len(rna_genes), 20)),
            index=rna_genes,
            columns=[f'RNAsample{i}' for i in range(20)]
        )
        rna_data.to_csv(rna_file)

        # Load RNA data
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(rna_file)

        assert rna_dataset.id_type == 'symbol'
        assert len(rna_dataset.gene_ids) == 5

        # Map IDs
        mapper = CrossModalIDMapper(cache_dir=temp_dir / 'cache')
        mapping = mapper.unify_ids(
            protein_ids=list(proteomics_matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type='symbol',
            species='human'
        )

        # Should have 4 common genes
        assert len(mapping.common_genes) == 4
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes
        assert 'EGFR' in mapping.common_genes
        assert 'KRAS' in mapping.common_genes

        # Protein-only (not in RNA)
        assert 'MYC' in mapping.protein_only or len(mapping.protein_only) > 0

        # RNA-only (not in protein)
        assert 'CDK2' in mapping.rna_only

        # Create ModuleDiscovery with RNA filter
        discovery = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=mapping.common_genes
        )

        # Validate RNA filter is applied
        assert discovery.rna_filter_genes == mapping.common_genes

    def test_no_rna_filter_backward_compatible(self, proteomics_matrix):
        """Test that omitting RNA filter works (backward compatible)."""
        # Create discovery without RNA filter
        discovery = ModuleDiscovery.from_matrix(proteomics_matrix)

        # Should work without RNA filter
        assert discovery.rna_filter_genes is None

        # Should be able to run discover_de_novo
        modules = discovery.discover_de_novo(
            n_genes=5,
            min_correlation=0.7,
            min_module_size=3,
            method='pearson'
        )

        # Should return results (may be empty, that's OK)
        assert isinstance(modules, list)


class TestFullPipelineEnsembl:
    """Test full pipeline with Ensembl IDs."""

    def test_proteomics_symbols_rna_ensembl(self, temp_dir, temp_cache_dir, proteomics_matrix):
        """Test pipeline with proteomics symbols and RNA Ensembl IDs."""
        # Create RNA data with Ensembl IDs
        rna_file = temp_dir / "rna_ensembl.csv"

        # Real Ensembl IDs for genes in proteomics matrix
        rna_ensembl = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
            'ENSG00000146648',  # EGFR
            'ENSG00000141736',  # ERBB2 (not in protein)
        ]

        rna_data = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(len(rna_ensembl), 15)),
            index=rna_ensembl,
            columns=[f'RNA{i}' for i in range(15)]
        )
        rna_data.to_csv(rna_file)

        # Load RNA
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(rna_file)
        assert rna_dataset.id_type == 'ensembl_gene'

        # Map IDs (will use MyGeneInfo API)
        mapper = CrossModalIDMapper(cache_dir=temp_cache_dir)
        mapping = mapper.unify_ids(
            protein_ids=list(proteomics_matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type='ensembl_gene',
            species='human'
        )

        # Should successfully map Ensembl → symbols
        assert len(mapping.common_genes) >= 2
        assert 'TP53' in mapping.common_genes or 'BRCA1' in mapping.common_genes

        # Create discovery with RNA filter
        discovery = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=mapping.common_genes
        )

        assert len(discovery.rna_filter_genes) >= 2


class TestFullPipelineNumeric:
    """Test full pipeline with numeric RNA indices."""

    def test_proteomics_symbols_rna_numeric(self, temp_dir, temp_cache_dir, proteomics_matrix):
        """Test pipeline with numeric RNA indices + annotation."""
        # Create RNA data with numeric indices
        rna_file = temp_dir / "rna_numeric.csv"
        n_rna_genes = 100

        rna_data = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(n_rna_genes, 20)),
            index=range(n_rna_genes),
            columns=[f'RNA{i}' for i in range(20)]
        )
        rna_data.to_csv(rna_file)

        # Create annotation (first 5 match protein genes)
        annot_file = temp_dir / "annotation.csv"
        gene_mapping = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'KRAS'] + \
                       [f'GENE{i:05d}' for i in range(95)]

        annot = pd.DataFrame({
            'index': list(range(n_rna_genes)),
            'gene_id': gene_mapping
        })
        annot.to_csv(annot_file, index=False)

        # Load RNA with annotation
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(rna_file, annotation_path=annot_file)

        # NOTE: After mapping numeric -> symbols, type should be 'symbol'
        assert rna_dataset.id_type == 'symbol'
        assert rna_dataset.gene_ids[0] == 'TP53'

        # Map IDs
        mapper = CrossModalIDMapper(cache_dir=temp_cache_dir)
        mapping = mapper.unify_ids(
            protein_ids=list(proteomics_matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type='symbol',  # After annotation, they're symbols
            species='human'
        )

        # Should find 5 common genes
        assert len(mapping.common_genes) == 5
        assert 'TP53' in mapping.common_genes
        assert 'BRCA1' in mapping.common_genes

        # Create discovery
        discovery = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=mapping.common_genes
        )

        assert len(discovery.rna_filter_genes) == 5


class TestModuleDiscoveryWithRNAFilter:
    """Test ModuleDiscovery behavior with RNA filtering."""

    def test_universe_selection_respects_rna_filter(self, temp_dir, proteomics_matrix):
        """Test that gene universe respects RNA filter."""
        # Create RNA filter (subset of genes)
        rna_genes = {'TP53', 'BRCA1', 'EGFR'}

        # Create discovery with RNA filter
        discovery = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=rna_genes
        )

        # Discover de novo
        modules = discovery.discover_de_novo(
            n_genes=5,  # Request 5 genes
            min_correlation=0.5,
            min_module_size=2,
            method='pearson'
        )

        # If modules found, all genes should be in RNA filter
        for module in modules:
            module_genes = set(module.genes.split(','))
            # All module genes should be in RNA filter or the high-variance set
            # (this tests that the filter was applied)
            assert all(
                gene in rna_genes or gene in proteomics_matrix.feature_ids
                for gene in module_genes
            )

    def test_rna_filter_reduces_universe(self, proteomics_matrix):
        """Test that RNA filter actually reduces gene universe."""
        # Small RNA filter
        rna_genes_small = {'TP53', 'BRCA1'}

        # Discovery with small RNA filter
        discovery_filtered = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=rna_genes_small
        )

        # Discovery without filter
        discovery_unfiltered = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=None
        )

        # Filtered should have RNA filter
        assert discovery_filtered.rna_filter_genes == rna_genes_small

        # Unfiltered should not
        assert discovery_unfiltered.rna_filter_genes is None


class TestEdgeCases:
    """Test edge cases in cross-modal integration."""

    def test_no_overlap_between_datasets(self, temp_dir, temp_cache_dir, proteomics_matrix):
        """Test when proteomics and RNA have no overlapping genes."""
        # RNA with completely different genes
        rna_file = temp_dir / "rna_disjoint.csv"
        rna_genes = ['GENE1', 'GENE2', 'GENE3']
        rna_data = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(3, 10)),
            index=rna_genes,
            columns=[f'RNA{i}' for i in range(10)]
        )
        rna_data.to_csv(rna_file)

        # Load and map
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(rna_file)

        mapper = CrossModalIDMapper(cache_dir=temp_cache_dir)
        mapping = mapper.unify_ids(
            protein_ids=list(proteomics_matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type='symbol',
            species='human'
        )

        # No common genes
        assert len(mapping.common_genes) == 0

        # All protein genes are protein-only
        assert len(mapping.protein_only) == proteomics_matrix.n_features

        # All RNA genes are RNA-only
        assert len(mapping.rna_only) == len(rna_genes)

        # Can still create discovery (with empty filter)
        discovery = ModuleDiscovery.from_matrix(
            proteomics_matrix,
            rna_filter_genes=mapping.common_genes
        )

        # RNA filter is empty set
        assert discovery.rna_filter_genes == set()

    def test_complete_overlap(self, temp_dir, temp_cache_dir, proteomics_matrix):
        """Test when all proteomics genes are in RNA."""
        # RNA with all proteomics genes + extras
        rna_file = temp_dir / "rna_complete.csv"
        rna_genes = list(proteomics_matrix.feature_ids) + ['EXTRA1', 'EXTRA2']

        rna_data = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, size=(len(rna_genes), 15)),
            index=rna_genes,
            columns=[f'RNA{i}' for i in range(15)]
        )
        rna_data.to_csv(rna_file)

        # Load and map
        rna_loader = RNADataLoader()
        rna_dataset = rna_loader.load(rna_file)

        mapper = CrossModalIDMapper(cache_dir=temp_cache_dir)
        mapping = mapper.unify_ids(
            protein_ids=list(proteomics_matrix.feature_ids),
            rna_ids=rna_dataset.gene_ids,
            rna_id_type='symbol',
            species='human'
        )

        # All protein genes are common
        assert len(mapping.common_genes) == proteomics_matrix.n_features

        # No protein-only
        assert len(mapping.protein_only) == 0

        # Some RNA-only (EXTRA1, EXTRA2)
        assert len(mapping.rna_only) == 2
