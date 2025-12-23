"""
Format configuration for multi-omic data loading.

This module provides explicit configuration for loading diverse biomolecular data formats.
The design philosophy: **auto-detect what's safe, require explicit config for ambiguous cases**.

Design Principles:
    1. Auto-detection for unambiguous properties (delimiter sniffing, numeric validation)
    2. Explicit configuration for semantic extraction (ID patterns, non-data columns)
    3. Preset profiles for common formats (transcriptomics, proteomics, metabolomics)
    4. Clear error messages when detection fails

Why Not Full Auto-Detection?
    Gene symbols can contain delimiters: HLA-A, MT-CO1, C10orf2
    Compound keys vary by database: sp|P12345|GENE_HUMAN vs ENSG00000000003
    Sample metadata encoding is institution-specific

Threading the Needle:
    - Zero-config for standard formats (Ensembl CSV, gene symbol CSV)
    - Explicit patterns for complex formats (UniProt proteomics TSV)
    - Fail-fast with helpful messages rather than silent misinterpretation

Examples:
    >>> from cliquefinder.io.formats import DataFormat, PRESETS
    >>>
    >>> # Use preset for standard proteomics
    >>> fmt = PRESETS['uniprot_tsv']
    >>> matrix = load_matrix("data.txt", format=fmt)
    >>>
    >>> # Custom format with explicit ID extraction
    >>> fmt = DataFormat(
    ...     delimiter='\t',
    ...     id_pattern=r'(?P<gene_symbol>[A-Z][A-Z0-9]+)_HUMAN$',
    ...     drop_columns=['nFragment', 'nPeptide'],
    ... )
"""

from __future__ import annotations

import re
import csv
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Optional, Callable, Pattern
from pathlib import Path


__all__ = [
    'OmicType',
    'IdNamespace',
    'DataFormat',
    'PRESETS',
    'sniff_delimiter',
    'extract_id',
]


class OmicType(Enum):
    """Biomolecular data type for downstream processing hints."""
    TRANSCRIPTOMIC = auto()  # RNA-seq, microarray
    PROTEOMIC = auto()       # Mass spec, antibody arrays
    METABOLOMIC = auto()     # LC-MS, NMR
    GENOMIC = auto()         # Variants, CNV
    EPIGENOMIC = auto()      # Methylation, ATAC-seq
    UNKNOWN = auto()


class IdNamespace(Enum):
    """
    Biological identifier namespace.

    Used for:
    - Validation (check ID format matches expected pattern)
    - Normalization (map to canonical identifiers)
    - Database lookups (query appropriate APIs)
    """
    ENSEMBL_GENE = auto()      # ENSG00000000003
    ENSEMBL_TRANSCRIPT = auto() # ENST00000000003
    ENSEMBL_PROTEIN = auto()    # ENSP00000000003
    UNIPROT = auto()            # P12345, A0AVT1
    GENE_SYMBOL = auto()        # TP53, BRCA1, HLA-A
    REFSEQ = auto()             # NM_000546, NP_000537
    HGNC = auto()               # HGNC:11998
    ENTREZ = auto()             # 7157
    HMDB = auto()               # HMDB0000001
    CHEBI = auto()              # CHEBI:15377
    UNKNOWN = auto()


@dataclass
class DataFormat:
    """
    Configuration for loading biomolecular expression/abundance matrices.

    This class captures all format-specific knowledge needed to reliably
    extract biomolecular identifiers from diverse file formats.

    Attributes:
        name: Human-readable format name (e.g., "UniProt Proteomics TSV")
        omic_type: Type of omic data (transcriptomic, proteomic, etc.)
        id_namespace: Expected identifier namespace (Ensembl, UniProt, symbol)

        delimiter: Column delimiter (None = auto-detect from '\t', ',', ';')
        has_header: First row is header (None = auto-detect)
        index_col: Column to use as row index (default: 0, first column)

        id_pattern: Regex with named group 'id' to extract biomolecular identifier
                   Example: r'sp\\|(?P<uniprot>[A-Z0-9]+)\\|(?P<id>[A-Z0-9]+)_HUMAN'
                   If None, uses entire index value as ID

        drop_columns: Column names to exclude from data (e.g., ['nFragment', 'nPeptide'])
        drop_columns_pattern: Regex pattern for columns to drop

        sample_id_pattern: Regex with named groups to parse sample metadata
                          Example: r'(?P<phenotype>CASE|CTRL)_(?P<code>[A-Z0-9]+)'

        encoding: File encoding (default: utf-8)
        skip_rows: Number of header rows to skip before data
        na_values: Values to treat as missing (default: ['', 'NA', 'NaN', 'nan'])

    Examples:
        >>> # Simple CSV with Ensembl IDs
        >>> fmt = DataFormat(
        ...     name="Ensembl CSV",
        ...     omic_type=OmicType.TRANSCRIPTOMIC,
        ...     id_namespace=IdNamespace.ENSEMBL_GENE,
        ... )

        >>> # Complex proteomics TSV
        >>> fmt = DataFormat(
        ...     name="UniProt Proteomics TSV",
        ...     omic_type=OmicType.PROTEOMIC,
        ...     id_namespace=IdNamespace.GENE_SYMBOL,
        ...     delimiter='\\t',
        ...     id_pattern=r'\\d+/sp\\|[A-Z0-9]+\\|(?P<id>[A-Z0-9]+)_HUMAN',
        ...     drop_columns=['nFragment', 'nPeptide'],
        ...     sample_id_pattern=r'(?P<phenotype>CASE|CTRL)_(?P<rest>.+)',
        ... )
    """

    # Identity
    name: str = "Unknown Format"
    omic_type: OmicType = OmicType.UNKNOWN
    id_namespace: IdNamespace = IdNamespace.UNKNOWN

    # Structural
    delimiter: Optional[str] = None  # None = auto-detect
    has_header: Optional[bool] = None  # None = assume True
    index_col: int = 0
    encoding: str = 'utf-8'
    skip_rows: int = 0

    # ID extraction
    id_pattern: Optional[str] = None  # Regex with named group 'id'
    _compiled_id_pattern: Optional[Pattern] = field(default=None, repr=False, compare=False)

    # Column filtering
    drop_columns: list[str] = field(default_factory=list)
    drop_columns_pattern: Optional[str] = None

    # Sample metadata parsing
    sample_id_pattern: Optional[str] = None
    _compiled_sample_pattern: Optional[Pattern] = field(default=None, repr=False, compare=False)

    # Missing value handling
    na_values: list[str] = field(default_factory=lambda: ['', 'NA', 'NaN', 'nan', 'NULL', 'null'])

    def __post_init__(self):
        """Compile regex patterns for efficiency."""
        if self.id_pattern:
            try:
                self._compiled_id_pattern = re.compile(self.id_pattern)
                if 'id' not in self._compiled_id_pattern.groupindex:
                    raise ValueError(
                        f"id_pattern must contain named group 'id': {self.id_pattern}"
                    )
            except re.error as e:
                raise ValueError(f"Invalid id_pattern regex: {e}")

        if self.sample_id_pattern:
            try:
                self._compiled_sample_pattern = re.compile(self.sample_id_pattern)
            except re.error as e:
                raise ValueError(f"Invalid sample_id_pattern regex: {e}")

    def extract_id(self, raw_index: str) -> str:
        """
        Extract biomolecular ID from raw index value.

        Args:
            raw_index: Raw value from index column (e.g., "1/sp|A0AVT1|UBA6_HUMAN")

        Returns:
            Extracted ID (e.g., "UBA6") or original value if no pattern

        Raises:
            ValueError: If pattern doesn't match
        """
        if not self._compiled_id_pattern:
            return str(raw_index).strip()

        match = self._compiled_id_pattern.search(str(raw_index))
        if not match:
            raise ValueError(
                f"ID pattern '{self.id_pattern}' did not match: '{raw_index}'"
            )
        return match.group('id')

    def extract_sample_metadata(self, sample_id: str) -> dict[str, str]:
        """
        Extract metadata from sample ID using pattern.

        Args:
            sample_id: Sample column header (e.g., "CASE_NEUAA295HHE-9014-P_D3")

        Returns:
            Dict of extracted groups (e.g., {'phenotype': 'CASE', 'rest': '...'})
        """
        if not self._compiled_sample_pattern:
            return {}

        match = self._compiled_sample_pattern.search(str(sample_id))
        if not match:
            return {}
        return match.groupdict()

    def should_drop_column(self, col_name: str) -> bool:
        """Check if column should be excluded from data matrix."""
        if col_name in self.drop_columns:
            return True
        if self.drop_columns_pattern:
            if re.search(self.drop_columns_pattern, col_name):
                return True
        return False


# =============================================================================
# Format Presets
# =============================================================================

PRESETS: dict[str, DataFormat] = {
    # Standard RNA-seq with Ensembl gene IDs
    'ensembl_csv': DataFormat(
        name="Ensembl Gene CSV",
        omic_type=OmicType.TRANSCRIPTOMIC,
        id_namespace=IdNamespace.ENSEMBL_GENE,
        delimiter=',',
        sample_id_pattern=r'^(?P<phenotype>CASE|CTRL|case|ctrl|Case|Ctrl)[-_]',
    ),

    # Gene symbol CSV (simple format)
    'gene_symbol_csv': DataFormat(
        name="Gene Symbol CSV",
        omic_type=OmicType.TRANSCRIPTOMIC,
        id_namespace=IdNamespace.GENE_SYMBOL,
        delimiter=',',
        sample_id_pattern=r'^(?P<phenotype>CASE|CTRL|case|ctrl|Case|Ctrl)[-_]',
    ),

    # Answer ALS proteomics format
    # Index: "1/sp|A0AVT1|UBA6_HUMAN"
    # Samples: "CASE_NEUAA295HHE-9014-P_D3"
    'answerals_proteomics': DataFormat(
        name="Answer ALS Proteomics TSV",
        omic_type=OmicType.PROTEOMIC,
        id_namespace=IdNamespace.UNIPROT,
        delimiter='\t',
        # Extract UniProt accession: 1/sp|A0AVT1|UBA6_HUMAN -> A0AVT1
        # (entry names like UBA6 often don't match HGNC symbols; accessions resolve better)
        id_pattern=r'\d+/[a-z]+\|(?P<id>[A-Z0-9]+)\|[A-Z0-9]+_[A-Z]+$',
        drop_columns=['nFragment', 'nPeptide'],
        sample_id_pattern=r'^(?P<phenotype>CASE|CTRL)_',
    ),

    # Generic UniProt proteomics
    # Index: "sp|P12345|GENE_HUMAN" or "tr|A0A0A0|GENE_HUMAN"
    'uniprot_tsv': DataFormat(
        name="UniProt Proteomics TSV",
        omic_type=OmicType.PROTEOMIC,
        id_namespace=IdNamespace.GENE_SYMBOL,
        delimiter='\t',
        # Extract gene symbol: sp|A0AVT1|UBA6_HUMAN -> UBA6
        id_pattern=r'(?:sp|tr)\|[A-Z0-9]+\|(?P<id>[A-Z0-9]+)_[A-Z]+',
    ),

    # MaxQuant proteomics output
    'maxquant': DataFormat(
        name="MaxQuant Proteomics",
        omic_type=OmicType.PROTEOMIC,
        id_namespace=IdNamespace.GENE_SYMBOL,
        delimiter='\t',
        id_pattern=r'^(?P<id>[A-Za-z0-9]+)(?:;|$)',  # First symbol before semicolon
        drop_columns_pattern=r'^(Reverse|Potential contaminant|Only identified by site)',
    ),

    # HMDB metabolomics
    'hmdb_csv': DataFormat(
        name="HMDB Metabolomics CSV",
        omic_type=OmicType.METABOLOMIC,
        id_namespace=IdNamespace.HMDB,
        delimiter=',',
    ),

    # Generic tab-separated with simple IDs
    'generic_tsv': DataFormat(
        name="Generic TSV",
        omic_type=OmicType.UNKNOWN,
        id_namespace=IdNamespace.UNKNOWN,
        delimiter='\t',
    ),

    # Generic comma-separated with simple IDs
    'generic_csv': DataFormat(
        name="Generic CSV",
        omic_type=OmicType.UNKNOWN,
        id_namespace=IdNamespace.UNKNOWN,
        delimiter=',',
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================

def sniff_delimiter(path: Path, sample_size: int = 8192) -> str:
    """
    Auto-detect delimiter from file content.

    Uses Python's csv.Sniffer with fallback heuristics.

    Args:
        path: Path to data file
        sample_size: Bytes to sample for detection

    Returns:
        Detected delimiter character ('\t', ',', ';', or ' ')

    Raises:
        ValueError: If delimiter cannot be determined
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(sample_size)

    # Try csv.Sniffer first
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters='\t,;|')
        return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: count delimiter occurrences in first line
    first_line = sample.split('\n')[0]
    counts = {
        '\t': first_line.count('\t'),
        ',': first_line.count(','),
        ';': first_line.count(';'),
    }

    if max(counts.values()) == 0:
        raise ValueError(
            f"Could not detect delimiter in {path}. "
            "Please specify explicitly with format.delimiter"
        )

    return max(counts, key=counts.get)


def extract_id(raw_value: str, pattern: Optional[str] = None) -> str:
    """
    Extract biomolecular ID from raw index value.

    Convenience function for one-off extraction without DataFormat.

    Args:
        raw_value: Raw index value (e.g., "1/sp|A0AVT1|UBA6_HUMAN")
        pattern: Regex with named group 'id' (e.g., r'\\|(?P<id>[A-Z0-9]+)_')

    Returns:
        Extracted ID or original value if no pattern

    Examples:
        >>> extract_id("1/sp|A0AVT1|UBA6_HUMAN", r"\\|(?P<id>[A-Z0-9]+)_[A-Z]+$")
        'UBA6'
        >>> extract_id("ENSG00000000003")
        'ENSG00000000003'
    """
    if not pattern:
        return str(raw_value).strip()

    compiled = re.compile(pattern)
    match = compiled.search(str(raw_value))
    if not match or 'id' not in match.groupdict():
        return str(raw_value).strip()
    return match.group('id')


def suggest_format(path: Path) -> tuple[str, DataFormat]:
    """
    Suggest appropriate format preset based on file inspection.

    Performs heuristic analysis to recommend a format. Always verify
    the suggestion before using in production.

    Args:
        path: Path to data file

    Returns:
        Tuple of (preset_name, DataFormat)

    Note:
        This is a convenience for exploration. For production pipelines,
        explicitly specify the format configuration.
    """
    # Read sample of file
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()
        first_data = f.readline().strip()

    delimiter = sniff_delimiter(path)

    # Check index column pattern
    if delimiter == '\t':
        parts = first_data.split('\t')
    else:
        parts = first_data.split(delimiter)

    if not parts:
        return ('generic_csv', PRESETS['generic_csv'])

    index_val = parts[0]

    # Check for Answer ALS proteomics format: "1/sp|...|..._HUMAN" or "1/iRT_protein"
    # Also check subsequent rows if first is internal standard
    if re.match(r'\d+/(?:[a-z]+\|[A-Z0-9]+\|[A-Z0-9]+_[A-Z]+|iRT_)', index_val):
        fmt = PRESETS['answerals_proteomics']
        if delimiter != fmt.delimiter:
            fmt = replace(
                fmt, 
                delimiter=delimiter, 
                name=f"Answer ALS Proteomics ({'CSV' if delimiter == ',' else 'Custom'})"
            )
        return ('answerals_proteomics', fmt)

    # Check for UniProt format: "sp|...|..."
    if re.match(r'(?:sp|tr)\|[A-Z0-9]+\|', index_val):
        fmt = PRESETS['uniprot_tsv']
        if delimiter != fmt.delimiter:
            fmt = replace(
                fmt, 
                delimiter=delimiter, 
                name=f"UniProt Proteomics ({'CSV' if delimiter == ',' else 'Custom'})"
            )
        return ('uniprot_tsv', fmt)

    # Check for Ensembl gene ID
    if re.match(r'ENSG\d{11}', index_val):
        return ('ensembl_csv', PRESETS['ensembl_csv'])

    # Check for HMDB
    if re.match(r'HMDB\d+', index_val):
        return ('hmdb_csv', PRESETS['hmdb_csv'])

    # Default based on delimiter
    if delimiter == '\t':
        return ('generic_tsv', PRESETS['generic_tsv'])
    return ('generic_csv', PRESETS['generic_csv'])
