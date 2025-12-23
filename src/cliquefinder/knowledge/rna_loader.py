"""
RNA data loader for multi-omics integration.

Provides flexible loading of RNA-seq count matrices with automatic ID format detection.
Supports multiple identifier types (gene symbols, Ensembl IDs, numeric indices) and
handles annotation mapping for numeric datasets.

Biological Context:
    RNA-seq data represents transcript abundance across samples. Common formats:
    - Gene symbols as row indices (TP53, BRCA1, etc.)
    - Ensembl gene IDs (ENSG00000141510, ENSG00000012048, etc.)
    - Ensembl transcript IDs (ENST00000269305, etc.)
    - Numeric indices (0, 1, 2, ...) requiring separate annotation file

    Used for cross-modal filtering: Only consider proteins/genes that are
    transcriptomally active (present in RNA-seq data).

Engineering Design:
    - Auto-detection of ID format via pattern matching
    - Support for CSV and TSV formats
    - Robust error handling for missing/malformed data
    - Validation of annotation mappings
    - Memory-efficient loading of large matrices (100M+)

Examples:
    >>> from pathlib import Path
    >>> from cliquefinder.knowledge.rna_loader import RNADataLoader
    >>>
    >>> # Load with gene symbols
    >>> loader = RNADataLoader()
    >>> dataset = loader.load(Path("rna_counts_symbols.csv"))
    >>> print(dataset.id_type)  # 'symbol'
    >>> print(dataset.n_genes)  # 15000
    >>>
    >>> # Load with numeric indices + annotation
    >>> dataset = loader.load(
    ...     rna_path=Path("numeric__AALS-RNAcountsMatrix.csv"),
    ...     annotation_path=Path("gene_annotations.csv")
    ... )
    >>> print(dataset.id_type)  # 'numeric' (mapped via annotation)
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cliquefinder.core.biomatrix import BioMatrix

__all__ = ['RNADataset', 'RNADataLoader']

logger = logging.getLogger(__name__)


@dataclass
class RNADataset:
    """
    RNA-seq dataset with metadata and optional expression matrix.

    Supports two modes:
    1. Lightweight (default): Only gene IDs for cross-modal ID mapping
    2. Full matrix: Includes BioMatrix for expression-based filtering (CPM, prevalence)

    Attributes:
        gene_ids: List of gene identifiers (symbols, Ensembl IDs, etc.)
        id_type: Detected identifier format
            - 'symbol': Gene symbols (TP53, BRCA1, etc.)
            - 'ensembl_gene': Ensembl gene IDs (ENSG00000...)
            - 'ensembl_transcript': Ensembl transcript IDs (ENST00000...)
            - 'numeric': Numeric indices (requires annotation file)
        n_genes: Number of genes
        n_samples: Number of samples
        sample_ids: Sample identifiers if available (optional)
        matrix: Optional BioMatrix with full expression data for filtering.
                When present, enables CPM-based stratified expression filtering.

    Examples:
        >>> # Lightweight mode (ID mapping only)
        >>> dataset = RNADataset(
        ...     gene_ids=['TP53', 'BRCA1', 'EGFR'],
        ...     id_type='symbol',
        ...     n_genes=3,
        ...     n_samples=100,
        ...     sample_ids=['S001', 'S002', ...]
        ... )
        >>>
        >>> # Full matrix mode (expression filtering)
        >>> dataset = loader.load(rna_path, include_matrix=True)
        >>> assert dataset.matrix is not None
    """
    gene_ids: List[str]
    id_type: str
    n_genes: int
    n_samples: int
    sample_ids: Optional[List[str]] = None
    matrix: Optional['BioMatrix'] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate dataset consistency."""
        if self.n_genes != len(self.gene_ids):
            raise ValueError(
                f"n_genes ({self.n_genes}) must match gene_ids length ({len(self.gene_ids)})"
            )
        if self.sample_ids is not None and self.n_samples != len(self.sample_ids):
            raise ValueError(
                f"n_samples ({self.n_samples}) must match sample_ids length ({len(self.sample_ids)})"
            )
        if self.matrix is not None:
            if self.matrix.n_features != self.n_genes:
                raise ValueError(
                    f"matrix.n_features ({self.matrix.n_features}) must match n_genes ({self.n_genes})"
                )
            if self.matrix.n_samples != self.n_samples:
                raise ValueError(
                    f"matrix.n_samples ({self.matrix.n_samples}) must match n_samples ({self.n_samples})"
                )

    def __repr__(self) -> str:
        """String representation for debugging."""
        sample_info = f", {self.n_samples} samples" if self.sample_ids else ""
        matrix_info = ", with matrix" if self.matrix is not None else ""
        return (
            f"RNADataset({self.n_genes} genes ({self.id_type}){sample_info}{matrix_info})\n"
            f"  First gene: {self.gene_ids[0] if self.gene_ids else 'N/A'}\n"
            f"  Last gene: {self.gene_ids[-1] if self.gene_ids else 'N/A'}"
        )

    def with_metadata(self, sample_metadata: pd.DataFrame) -> 'RNADataset':
        """
        Return a new RNADataset with updated sample metadata on the matrix.

        This enables stratified expression filtering by merging external
        metadata (phenotype, sex, etc.) into the BioMatrix.

        Args:
            sample_metadata: DataFrame indexed by sample ID with metadata columns

        Returns:
            New RNADataset with metadata-enriched matrix

        Raises:
            ValueError: If matrix is None or sample IDs don't align
        """
        if self.matrix is None:
            raise ValueError("Cannot add metadata: matrix is None. Load with include_matrix=True.")

        from cliquefinder.core.biomatrix import BioMatrix

        # Align metadata to matrix samples
        aligned_meta = sample_metadata.reindex(self.matrix.sample_ids)
        n_matched = aligned_meta.notna().any(axis=1).sum()

        if n_matched == 0:
            logger.warning("No sample IDs matched between RNA matrix and metadata")

        # Merge with existing metadata
        merged_meta = self.matrix.sample_metadata.copy()
        for col in aligned_meta.columns:
            if col not in merged_meta.columns:
                merged_meta[col] = aligned_meta[col]

        new_matrix = BioMatrix(
            data=self.matrix.data,
            feature_ids=self.matrix.feature_ids,
            sample_ids=self.matrix.sample_ids,
            sample_metadata=merged_meta,
            quality_flags=self.matrix.quality_flags
        )

        return RNADataset(
            gene_ids=self.gene_ids,
            id_type=self.id_type,
            n_genes=self.n_genes,
            n_samples=self.n_samples,
            sample_ids=self.sample_ids,
            matrix=new_matrix
        )


class RNADataLoader:
    """
    Flexible RNA data loader with automatic ID format detection.

    Handles multiple RNA-seq data formats:
    1. Gene symbols as row indices
    2. Ensembl gene IDs as row indices
    3. Ensembl transcript IDs as row indices
    4. Numeric indices with separate annotation file

    Auto-detects ID format by analyzing patterns in the first 100 IDs.

    Usage:
        >>> loader = RNADataLoader()
        >>> dataset = loader.load(Path("rna_counts.csv"))
        >>> print(f"Detected {dataset.id_type} IDs")
        >>> print(f"Loaded {dataset.n_genes} genes")
    """

    # ID format detection patterns
    ENSEMBL_GENE_PATTERN = re.compile(r'^ENS[A-Z]*G\d{11}')
    ENSEMBL_TRANSCRIPT_PATTERN = re.compile(r'^ENS[A-Z]*T\d{11}')
    ENSEMBL_PROTEIN_PATTERN = re.compile(r'^ENS[A-Z]*P\d{11}')
    REFSEQ_PATTERN = re.compile(r'^[NX][M_]\d+')
    UNIPROT_PATTERN = re.compile(r'^[P,Q,O][0-9][A-Z0-9]{3}[0-9]|[A-N,R-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$')
    INTEGER_PATTERN = re.compile(r'^\d+$')

    def __init__(self):
        """Initialize RNA data loader."""
        pass

    def load(
        self,
        rna_path: Path,
        annotation_path: Optional[Path] = None,
        detect_sample_size: int = 100,
        include_matrix: bool = False
    ) -> RNADataset:
        """
        Load RNA-seq count matrix with automatic ID format detection.

        Expected CSV/TSV format:
        - First column: gene/feature identifiers (or numeric indices)
        - Remaining columns: sample IDs with count/expression values
        - Header row with sample identifiers

        Example (gene symbols):
        ```
        gene_id,Sample1,Sample2,Sample3
        TP53,100,150,120
        BRCA1,50,75,60
        ```

        Example (numeric indices):
        ```
        ,Sample1,Sample2,Sample3
        0,100,150,120
        1,50,75,60
        ```

        ID Format Detection:
        - Analyzes first 100 IDs to determine format
        - ENSG00000... → 'ensembl_gene'
        - ENST00000... → 'ensembl_transcript'
        - RefSeq (NM_...) → 'refseq'
        - Uniprot → 'uniprot'
        - Integers (sparse/random) → 'entrez'
        - Integers (sequential 0..N) → 'numeric' (requires annotation_path)
        - Otherwise → 'symbol'

        Args:
            rna_path: Path to RNA counts matrix (CSV or TSV)
            annotation_path: Path to annotation file (required if numeric IDs)
                Expected format:
                - First column: numeric index (0, 1, 2, ...)
                - Second column: gene ID (symbol, Ensembl ID, etc.)
            detect_sample_size: Number of IDs to analyze for format detection
            include_matrix: If True, include full expression matrix as BioMatrix.
                Required for CPM-based stratified expression filtering.
                Default False for memory efficiency in ID-only workflows.

        Returns:
            RNADataset with gene IDs, format type, and dimensions.
            If include_matrix=True, also includes .matrix BioMatrix.

        Raises:
            FileNotFoundError: If rna_path does not exist
            ValueError: If data is malformed or numeric IDs without annotation
            pd.errors.EmptyDataError: If file is empty

        Examples:
            >>> # Load with gene symbols (lightweight, ID-only)
            >>> dataset = loader.load(Path("rna_symbols.csv"))
            >>> assert dataset.id_type == 'symbol'
            >>> assert dataset.matrix is None
            >>>
            >>> # Load with full matrix for expression filtering
            >>> dataset = loader.load(Path("rna_counts.csv"), include_matrix=True)
            >>> assert dataset.matrix is not None
            >>> # Can now apply CPM-based filtering
        """
        # Validate paths
        if not isinstance(rna_path, Path):
            rna_path = Path(rna_path)

        if not rna_path.exists():
            raise FileNotFoundError(f"RNA data file not found: {rna_path}")

        if not rna_path.is_file():
            raise ValueError(f"Path is not a file: {rna_path}")

        logger.info(f"Loading RNA data from {rna_path}")

        # Detect delimiter
        delimiter = self._detect_delimiter(rna_path)
        logger.debug(f"Detected delimiter: {repr(delimiter)}")

        # Read RNA counts matrix
        try:
            df = pd.read_csv(rna_path, sep=delimiter, index_col=0)
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"RNA data file is empty: {rna_path}") from e
        except Exception as e:
            raise ValueError(f"Failed to read RNA data file {rna_path}: {e}") from e

        # Validate
        if df.shape[1] == 0:
            raise ValueError(f"RNA data contains no samples (columns): {rna_path}")
        if df.empty:
            raise ValueError(f"RNA data contains no data (empty): {rna_path}")

        logger.info(f"Loaded {df.shape[0]:,} features × {df.shape[1]:,} samples")

        # Extract raw gene IDs
        raw_gene_ids = [str(idx) for idx in df.index]

        # Detect ID type
        id_type = self._detect_id_type(raw_gene_ids, detect_sample_size)
        logger.info(f"Detected ID type: {id_type}")

        # Handle numeric IDs (require annotation)
        if id_type == 'numeric':
            if annotation_path is None:
                # NEW: Try to find annotation file in same directory
                potential_files = [
                    rna_path.parent / "genes.csv",
                    rna_path.parent / "features.tsv",
                    rna_path.parent / "features.csv",
                    rna_path.parent / "gene_ids.csv"
                ]
                for p in potential_files:
                    if p.exists():
                        logger.info(f"Auto-detected annotation file: {p}")
                        annotation_path = p
                        break
            
            if annotation_path is None:
                raise ValueError(
                    "Numeric gene IDs (sequential indices) detected, but no annotation file provided. "
                    "Cannot resolve 0, 1, 2... to gene symbols without a map. "
                    "Please provide --rna-annotation."
                )

            gene_ids = self._load_annotation_mapping(
                annotation_path,
                raw_gene_ids,
                delimiter
            )
            logger.info(f"Mapped {len(gene_ids)} numeric IDs to gene identifiers")
            # Re-detect type of mapped IDs
            id_type = self._detect_id_type(gene_ids, detect_sample_size)
            logger.info(f"Resolved mapped ID type: {id_type}")
        else:
            gene_ids = raw_gene_ids

        # Handle duplicates
        if len(gene_ids) != len(set(gene_ids)):
            n_duplicates = len(gene_ids) - len(set(gene_ids))
            warnings.warn(
                f"Found {n_duplicates} duplicate gene IDs. "
                "This may affect downstream analysis.",
                UserWarning
            )

        # Extract sample IDs
        sample_ids = [str(sid) for sid in df.columns]

        # Optionally create BioMatrix for expression filtering
        matrix = None
        if include_matrix:
            from cliquefinder.core.biomatrix import BioMatrix

            # Convert DataFrame to numpy array (features x samples)
            data = df.values.astype(np.float64)

            # Create minimal sample metadata (can be enriched later via with_metadata())
            sample_metadata = pd.DataFrame(index=pd.Index(sample_ids, name='sample_id'))

            # Create feature and sample IDs as pd.Index (required by BioMatrix)
            feature_ids_index = pd.Index(gene_ids, name='feature_id')
            sample_ids_index = pd.Index(sample_ids, name='sample_id')

            # Quality flags must be 2D with shape (n_features, n_samples)
            quality_flags = np.ones(data.shape, dtype=np.int8)

            matrix = BioMatrix(
                data=data,
                feature_ids=feature_ids_index,
                sample_ids=sample_ids_index,
                sample_metadata=sample_metadata,
                quality_flags=quality_flags
            )
            logger.info(f"Created BioMatrix: {matrix.n_features} features × {matrix.n_samples} samples")

        # Create dataset
        dataset = RNADataset(
            gene_ids=gene_ids,
            id_type=id_type,
            n_genes=len(gene_ids),
            n_samples=len(sample_ids),
            sample_ids=sample_ids,
            matrix=matrix
        )

        matrix_status = "with matrix" if include_matrix else "ID-only"
        logger.info(f"Successfully loaded RNA dataset: {dataset.n_genes} genes, {dataset.n_samples} samples ({matrix_status})")

        return dataset

    def _detect_delimiter(self, path: Path) -> str:
        """
        Detect CSV or TSV delimiter.

        Args:
            path: Path to file

        Returns:
            Delimiter character (',' or '\t')
        """
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()

        # Count delimiters in first line
        comma_count = first_line.count(',')
        tab_count = first_line.count('\t')

        if tab_count > comma_count:
            return '\t'
        else:
            return ','

    def _detect_id_type(self, gene_ids: List[str], sample_size: int = 100) -> str:
        """
        Analyze gene IDs to detect format.

        Examines first `sample_size` IDs and uses majority voting:
        - If ≥90% match ENSG pattern → 'ensembl_gene'
        - If ≥90% match ENST pattern → 'ensembl_transcript'
        - If ≥90% match RefSeq pattern → 'refseq'
        - If ≥90% match Uniprot pattern → 'uniprot'
        - If ≥90% match Integers:
            - If sequential/dense (0,1,2...) → 'numeric' (Index)
            - If sparse/random → 'entrez' (Gene ID)
        - Otherwise → 'symbol'

        Args:
            gene_ids: List of gene identifier strings
            sample_size: Number of IDs to analyze (default: 100)

        Returns:
            ID type string
        """
        # Sample IDs for analysis
        sample = gene_ids[:min(sample_size, len(gene_ids))]

        if not sample:
            raise ValueError("Cannot detect ID type from empty gene list")

        total = len(sample)
        threshold = 0.9  # 90% threshold for pattern match

        # Check regex patterns
        ensembl_gene_count = sum(1 for gid in sample if self.ENSEMBL_GENE_PATTERN.match(gid))
        if ensembl_gene_count / total >= threshold:
            return 'ensembl_gene'

        ensembl_transcript_count = sum(1 for gid in sample if self.ENSEMBL_TRANSCRIPT_PATTERN.match(gid))
        if ensembl_transcript_count / total >= threshold:
            return 'ensembl_transcript'
            
        ensembl_protein_count = sum(1 for gid in sample if self.ENSEMBL_PROTEIN_PATTERN.match(gid))
        if ensembl_protein_count / total >= threshold:
            return 'ensembl_protein'

        refseq_count = sum(1 for gid in sample if self.REFSEQ_PATTERN.match(gid))
        if refseq_count / total >= threshold:
            return 'refseq'

        uniprot_count = sum(1 for gid in sample if self.UNIPROT_PATTERN.match(gid))
        if uniprot_count / total >= threshold:
            return 'uniprot'

        # Check numeric/entrez
        numeric_count = sum(1 for gid in sample if self.INTEGER_PATTERN.match(gid))
        if numeric_count / total >= threshold:
            # Distinguish between sequential index and Entrez ID
            # Convert sample to integers
            try:
                int_ids = [int(gid) for gid in sample]
                
                # Check for 0 (Entrez IDs are positive integers, 0 is invalid)
                if 0 in int_ids:
                    return 'numeric'
                
                # Check for sequentiality (index-like)
                # If sorted sample is exactly 0..N or 1..N, it's likely an index
                sorted_ids = sorted(int_ids)
                is_sequential_0 = sorted_ids == list(range(len(sorted_ids)))
                is_sequential_1 = sorted_ids == list(range(1, len(sorted_ids) + 1))
                
                if is_sequential_0 or is_sequential_1:
                    return 'numeric'
                
                # Heuristic: Entrez IDs are usually large (>1000) or sparse
                # If average ID is small (<len), likely index
                if np.mean(int_ids) < 2 * len(gene_ids) and len(gene_ids) > 100:
                     return 'numeric'
                     
                return 'entrez'
            except ValueError:
                pass # Fallback to symbol

        # Default
        return 'symbol'

    def _load_annotation_mapping(
        self,
        annotation_path: Path,
        raw_ids: List[str],
        delimiter: str
    ) -> List[str]:
        """
        Load annotation file and map numeric indices to gene IDs.

        Expected annotation format:
        ```
        index,gene_id
        0,ENSG00000141510
        1,ENSG00000012048
        ```

        Or tab-delimited:
        ```
        0    TP53
        1    BRCA1
        ```

        Args:
            annotation_path: Path to annotation CSV/TSV
            raw_ids: List of raw numeric indices from RNA matrix
            delimiter: Delimiter used in annotation file

        Returns:
            List of mapped gene IDs (same order as raw_ids)

        Raises:
            FileNotFoundError: If annotation file doesn't exist
            ValueError: If annotation is malformed or mapping incomplete
        """
        if not isinstance(annotation_path, Path):
            annotation_path = Path(annotation_path)

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        logger.info(f"Loading annotation mapping from {annotation_path}")

        # Detect delimiter for annotation file
        annot_delimiter = self._detect_delimiter(annotation_path)

        # Read annotation file
        try:
            # Try with header first
            annot_df = pd.read_csv(annotation_path, sep=annot_delimiter)

            # Check if it has 2 columns
            if annot_df.shape[1] < 2:
                # Try without header (index in first column, gene in second)
                annot_df = pd.read_csv(
                    annotation_path,
                    sep=annot_delimiter,
                    header=None,
                    names=['index', 'gene_id']
                )
            else:
                # Use first two columns
                cols = annot_df.columns[:2]
                annot_df = annot_df[cols]
                annot_df.columns = ['index', 'gene_id']

        except Exception as e:
            raise ValueError(f"Failed to read annotation file {annotation_path}: {e}") from e

        # Validate structure
        if annot_df.empty:
            raise ValueError(f"Annotation file is empty: {annotation_path}")

        logger.info(f"Loaded {len(annot_df)} annotation entries")

        # Convert index column to string for matching
        annot_df['index'] = annot_df['index'].astype(str)
        annot_df['gene_id'] = annot_df['gene_id'].astype(str)

        # Create mapping dictionary
        annot_map = dict(zip(annot_df['index'], annot_df['gene_id']))

        # Map raw IDs to gene IDs
        mapped_ids = []
        missing_count = 0

        for raw_id in raw_ids:
            if raw_id in annot_map:
                mapped_ids.append(annot_map[raw_id])
            else:
                # Missing annotation - use raw ID as fallback
                mapped_ids.append(raw_id)
                missing_count += 1

        if missing_count > 0:
            warnings.warn(
                f"Annotation mapping incomplete: {missing_count}/{len(raw_ids)} IDs "
                f"not found in annotation file. Using numeric IDs as fallback.",
                UserWarning
            )

        return mapped_ids
