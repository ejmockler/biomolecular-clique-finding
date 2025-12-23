"""
Integration tests for RNADataLoader.

Tests real file loading, ID format detection, and annotation mapping
with minimal mocking. Uses temporary files with realistic data structures.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from cliquefinder.knowledge.rna_loader import RNADataLoader, RNADataset


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def rna_loader():
    """Create RNA data loader instance."""
    return RNADataLoader()


class TestRNADataset:
    """Test RNADataset dataclass validation."""

    def test_valid_dataset(self):
        """Test creation of valid dataset."""
        dataset = RNADataset(
            gene_ids=['TP53', 'BRCA1', 'EGFR'],
            id_type='symbol',
            n_genes=3,
            n_samples=10,
            sample_ids=[f'S{i:03d}' for i in range(10)]
        )
        assert len(dataset.gene_ids) == 3
        assert dataset.id_type == 'symbol'
        assert dataset.n_genes == 3
        assert dataset.n_samples == 10

    def test_gene_count_mismatch(self):
        """Test validation catches gene count mismatch."""
        with pytest.raises(ValueError, match="n_genes.*must match gene_ids length"):
            RNADataset(
                gene_ids=['TP53', 'BRCA1'],
                id_type='symbol',
                n_genes=3,  # Mismatch!
                n_samples=10
            )

    def test_sample_count_mismatch(self):
        """Test validation catches sample count mismatch."""
        with pytest.raises(ValueError, match="n_samples.*must match sample_ids length"):
            RNADataset(
                gene_ids=['TP53', 'BRCA1'],
                id_type='symbol',
                n_genes=2,
                n_samples=5,
                sample_ids=['S1', 'S2', 'S3']  # Only 3 samples!
            )


class TestRNADataLoaderSymbols:
    """Test loading RNA data with gene symbols."""

    def test_load_gene_symbols_csv(self, temp_dir, rna_loader):
        """Test loading CSV with gene symbols as indices."""
        # Create realistic RNA counts file
        rna_file = temp_dir / "rna_symbols.csv"
        data = pd.DataFrame({
            'Sample1': [100, 250, 80, 150],
            'Sample2': [120, 230, 75, 160],
            'Sample3': [95, 260, 85, 140],
        }, index=['TP53', 'BRCA1', 'EGFR', 'MYC'])
        data.index.name = 'gene_symbol'
        data.to_csv(rna_file)

        # Load
        dataset = rna_loader.load(rna_file)

        # Validate
        assert dataset.id_type == 'symbol'
        assert dataset.n_genes == 4
        assert dataset.n_samples == 3
        assert dataset.gene_ids == ['TP53', 'BRCA1', 'EGFR', 'MYC']
        assert dataset.sample_ids == ['Sample1', 'Sample2', 'Sample3']

    def test_load_gene_symbols_tsv(self, temp_dir, rna_loader):
        """Test loading TSV with gene symbols."""
        rna_file = temp_dir / "rna_symbols.tsv"
        data = pd.DataFrame({
            'S1': [100, 250],
            'S2': [120, 230],
        }, index=['TP53', 'BRCA1'])
        data.index.name = 'gene'
        data.to_csv(rna_file, sep='\t')

        dataset = rna_loader.load(rna_file)

        assert dataset.id_type == 'symbol'
        assert dataset.n_genes == 2
        assert dataset.gene_ids == ['TP53', 'BRCA1']


class TestRNADataLoaderEnsembl:
    """Test loading RNA data with Ensembl IDs."""

    def test_load_ensembl_gene_ids(self, temp_dir, rna_loader):
        """Test loading with Ensembl gene IDs (ENSG...)."""
        rna_file = temp_dir / "rna_ensembl.csv"

        # Realistic Ensembl gene IDs
        ensembl_ids = [
            'ENSG00000141510',  # TP53
            'ENSG00000012048',  # BRCA1
            'ENSG00000146648',  # EGFR
        ]

        data = pd.DataFrame({
            'Sample1': [100, 250, 80],
            'Sample2': [120, 230, 75],
        }, index=ensembl_ids)
        data.to_csv(rna_file)

        dataset = rna_loader.load(rna_file)

        assert dataset.id_type == 'ensembl_gene'
        assert dataset.n_genes == 3
        assert dataset.gene_ids == ensembl_ids

    def test_load_ensembl_transcript_ids(self, temp_dir, rna_loader):
        """Test loading with Ensembl transcript IDs (ENST...)."""
        rna_file = temp_dir / "rna_transcripts.csv"

        transcript_ids = [
            'ENST00000269305',
            'ENST00000471181',
            'ENST00000455263',
        ]

        data = pd.DataFrame({
            'S1': [100, 250, 80],
            'S2': [120, 230, 75],
        }, index=transcript_ids)
        data.to_csv(rna_file)

        dataset = rna_loader.load(rna_file)

        assert dataset.id_type == 'ensembl_transcript'
        assert dataset.n_genes == 3


class TestRNADataLoaderNumeric:
    """Test loading RNA data with numeric indices + annotation."""

    def test_load_numeric_with_annotation(self, temp_dir, rna_loader):
        """Test loading numeric indices with annotation file."""
        # Create RNA counts with numeric indices
        rna_file = temp_dir / "rna_numeric.csv"
        data = pd.DataFrame({
            'Sample1': [100, 250, 80, 150],
            'Sample2': [120, 230, 75, 160],
            'Sample3': [95, 260, 85, 140],
        }, index=[0, 1, 2, 3])
        data.to_csv(rna_file)

        # Create annotation file
        annot_file = temp_dir / "annotation.csv"
        annot = pd.DataFrame({
            'index': [0, 1, 2, 3],
            'gene_id': ['TP53', 'BRCA1', 'EGFR', 'MYC']
        })
        annot.to_csv(annot_file, index=False)

        # Load with annotation
        dataset = rna_loader.load(rna_file, annotation_path=annot_file)

        # Validate mapping worked
        # NOTE: After mapping, the IDs are symbols, so type should be 'symbol'
        assert dataset.id_type == 'symbol'
        assert dataset.n_genes == 4
        assert dataset.gene_ids == ['TP53', 'BRCA1', 'EGFR', 'MYC']

    def test_numeric_without_annotation_raises(self, temp_dir, rna_loader):
        """Test that numeric IDs without annotation raises error."""
        rna_file = temp_dir / "rna_numeric.csv"
        data = pd.DataFrame({
            'S1': [100, 250],
            'S2': [120, 230],
        }, index=[0, 1])
        data.to_csv(rna_file)

        with pytest.raises(ValueError, match="Numeric gene IDs.*detected.*no annotation file"):
            rna_loader.load(rna_file)

    def test_annotation_tsv_format(self, temp_dir, rna_loader):
        """Test annotation file in TSV format."""
        rna_file = temp_dir / "rna_numeric.csv"
        data = pd.DataFrame({
            'S1': [100, 250],
        }, index=[0, 1])
        data.to_csv(rna_file)

        # TSV annotation
        annot_file = temp_dir / "annotation.tsv"
        annot = pd.DataFrame({
            'index': [0, 1],
            'gene': ['TP53', 'BRCA1']
        })
        annot.to_csv(annot_file, sep='\t', index=False)

        dataset = rna_loader.load(rna_file, annotation_path=annot_file)
        assert dataset.gene_ids == ['TP53', 'BRCA1']

    def test_incomplete_annotation_mapping(self, temp_dir, rna_loader):
        """Test partial annotation coverage (should warn, not error)."""
        rna_file = temp_dir / "rna_numeric.csv"
        data = pd.DataFrame({
            'S1': [100, 250, 80],
        }, index=[0, 1, 2])
        data.to_csv(rna_file)

        # Annotation missing index 2
        annot_file = temp_dir / "annotation.csv"
        annot = pd.DataFrame({
            'index': [0, 1],
            'gene_id': ['TP53', 'BRCA1']
        })
        annot.to_csv(annot_file, index=False)

        # Should load with warning, using raw index as fallback
        with pytest.warns(UserWarning, match="Annotation mapping incomplete"):
            dataset = rna_loader.load(rna_file, annotation_path=annot_file)

        # Index 2 should remain as '2'
        assert dataset.gene_ids == ['TP53', 'BRCA1', '2']


class TestRNADataLoaderEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self, rna_loader):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="RNA data file not found"):
            rna_loader.load(Path("/nonexistent/file.csv"))

    def test_empty_file(self, temp_dir, rna_loader):
        """Test loading empty file raises error."""
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            rna_loader.load(empty_file)

    def test_no_samples(self, temp_dir, rna_loader):
        """Test file with genes but no samples."""
        rna_file = temp_dir / "no_samples.csv"
        # Only index column, no data columns
        rna_file.write_text("gene_id\nTP53\nBRCA1\n")

        with pytest.raises(ValueError, match="no samples"):
            rna_loader.load(rna_file)

    def test_duplicate_gene_ids(self, temp_dir, rna_loader):
        """Test duplicate gene IDs triggers warning."""
        rna_file = temp_dir / "duplicates.csv"
        data = pd.DataFrame({
            'S1': [100, 250, 80],
            'S2': [120, 230, 75],
        }, index=['TP53', 'BRCA1', 'TP53'])  # Duplicate TP53
        data.to_csv(rna_file)

        with pytest.warns(UserWarning, match="duplicate gene IDs"):
            dataset = rna_loader.load(rna_file)

        assert dataset.n_genes == 3


class TestIDFormatDetection:
    """Test automatic ID format detection."""

    def test_mixed_ids_defaults_to_symbol(self, temp_dir, rna_loader):
        """Test mixed ID types defaults to symbol."""
        rna_file = temp_dir / "mixed.csv"

        # Mix of symbols and one Ensembl ID (but <90% threshold)
        mixed_ids = ['TP53', 'BRCA1', 'ENSG00000141510', 'MYC', 'EGFR', 'KRAS']
        data = pd.DataFrame({
            'S1': np.random.randint(50, 200, len(mixed_ids)),
        }, index=mixed_ids)
        data.to_csv(rna_file)

        dataset = rna_loader.load(rna_file)

        # Should default to 'symbol' since no pattern hits 90% threshold
        assert dataset.id_type == 'symbol'

    def test_mostly_ensembl_gene_ids(self, temp_dir, rna_loader):
        """Test ≥90% Ensembl gene IDs triggers ensembl_gene type."""
        rna_file = temp_dir / "mostly_ensembl.csv"

        # 9 Ensembl + 1 symbol = 90% Ensembl
        ensembl_ids = [f'ENSG{i:011d}' for i in range(9)]
        all_ids = ensembl_ids + ['TP53']

        data = pd.DataFrame({
            'S1': np.random.randint(50, 200, len(all_ids)),
        }, index=all_ids)
        data.to_csv(rna_file)

        dataset = rna_loader.load(rna_file)
        assert dataset.id_type == 'ensembl_gene'


class TestRealWorldScenarios:
    """Test realistic RNA-seq data scenarios."""

    def test_large_rna_matrix(self, temp_dir, rna_loader):
        """Test loading large RNA matrix (10K genes × 100 samples)."""
        rna_file = temp_dir / "large_rna.csv"

        n_genes = 10000
        n_samples = 100

        # Generate realistic gene symbols
        gene_ids = [f'GENE{i:05d}' for i in range(n_genes)]
        sample_ids = [f'Sample{i:03d}' for i in range(n_samples)]

        # Create sparse count matrix (most counts are low)
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))

        data = pd.DataFrame(counts, index=gene_ids, columns=sample_ids)
        data.to_csv(rna_file)

        # Load
        dataset = rna_loader.load(rna_file)

        # Validate
        assert dataset.n_genes == n_genes
        assert dataset.n_samples == n_samples
        assert len(dataset.gene_ids) == n_genes
        assert len(dataset.sample_ids) == n_samples

    def test_typical_numeric_rna_workflow(self, temp_dir, rna_loader):
        """Test typical workflow: numeric RNA data + annotation."""
        # Scenario: User has RNA counts from pipeline with numeric indices
        # and separate gene annotation file (common in many RNA-seq workflows)

        n_genes = 1000
        n_samples = 50

        # RNA counts with numeric indices
        rna_file = temp_dir / "numeric_counts.csv"
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        sample_ids = [f'S{i:03d}' for i in range(n_samples)]
        rna_df = pd.DataFrame(counts, index=range(n_genes), columns=sample_ids)
        rna_df.to_csv(rna_file)

        # Annotation with Ensembl IDs
        annot_file = temp_dir / "gene_annotation.csv"
        annot = pd.DataFrame({
            'index': list(range(n_genes)),
            'gene_id': [f'ENSG{i:011d}' for i in range(n_genes)]
        })
        annot.to_csv(annot_file, index=False)

        # Load
        dataset = rna_loader.load(rna_file, annotation_path=annot_file)

        # Validate
        assert dataset.n_genes == n_genes
        assert dataset.n_samples == n_samples
        # NOTE: After mapping numeric -> Ensembl, type should be 'ensembl_gene'
        assert dataset.id_type == 'ensembl_gene'
        assert all(gid.startswith('ENSG') for gid in dataset.gene_ids)
