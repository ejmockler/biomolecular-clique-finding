"""
CSV loader for expression matrices.

Provides robust loading of expression/count matrices from CSV files into BioMatrix
objects. Handles biological metadata parsing, validation, and quality flag initialization.

Biological Context:
    Expression data typically comes in CSV format:
    - Rows = genes/proteins (identified by Ensembl IDs, gene symbols, etc.)
    - Columns = samples (patients, cell lines, conditions)
    - Sample IDs often encode metadata (phenotype, batch, cohort)

    Example sample ID: CASE-NEUVM674HUA-5257-T_P003
    - CASE/CTRL = disease status (phenotype)
    - 5257 = cohort number (for batch correction)
    - P003 = plate identifier (technical replicate tracking)

Engineering Design:
    - Robust error handling for malformed CSVs
    - Efficient loading of large files (90MB+)
    - Optional metadata inference from sample IDs
    - Clear validation messages
    - Memory-efficient: single pass through data

Examples:
    >>> from pathlib import Path
    >>> from cliquefinder.io.loaders import load_csv_matrix
    >>>
    >>> # Load with automatic phenotype parsing
    >>> matrix = load_csv_matrix(Path("data.csv"), infer_phenotypes=True)
    >>> print(matrix.sample_metadata['phenotype'].value_counts())
    CASE    289
    CTRL    289
    >>>
    >>> # Load without metadata inference
    >>> matrix = load_csv_matrix(Path("data.csv"), infer_phenotypes=False)
    >>> print(matrix.sample_metadata.columns)
    Index([], dtype='object')
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.quality import QualityFlag
from cliquefinder.io.formats import DataFormat, PRESETS, sniff_delimiter, suggest_format

__all__ = ['load_csv_matrix', 'load_matrix']


def load_csv_matrix(path: Path, infer_phenotypes: bool = True) -> BioMatrix:
    """
    Load CSV expression/count matrix into BioMatrix.

    Expected CSV format:
    - First column: feature IDs (genes/proteins)
        - Column header may be empty ("") or "feature_id" or similar
        - Values should be unique identifiers (e.g., ENSG00000000003)
    - Remaining columns: sample IDs (headers) with numerical values
    - No missing values in sample IDs (column headers)
    - Numerical values may contain zeros (e.g., no expression)

    Example:
    ```
    "","CTRL-NEUEU392AE8-5190-T_P001","CASE-NEUVM674HUA-5257-T_P003"
    "ENSG00000000003",612,1056
    "ENSG00000000005",0,1
    ```

    Biological Metadata Parsing:
    If infer_phenotypes=True, attempts to parse sample IDs with format:
        {PHENOTYPE}-{CODE}-{COHORT}-{SUFFIX}_{PLATE}

    Example: CASE-NEUVM674HUA-5257-T_P003
        -> phenotype: CASE
        -> cohort: 5257

    Extracted metadata:
    - phenotype: Disease status (CASE/CTRL) or experimental condition
    - cohort: Numeric cohort identifier (important for batch effects)

    Args:
        path: Path to CSV file
        infer_phenotypes: Parse biological metadata from sample IDs
            If True, attempts to extract phenotype and cohort from sample names
            If False, creates empty sample_metadata DataFrame

    Returns:
        BioMatrix with:
        - data: Numerical expression matrix (features × samples)
        - feature_ids: Row identifiers from first column
        - sample_ids: Column identifiers from header
        - sample_metadata: Parsed metadata (if infer_phenotypes=True)
        - quality_flags: All values initialized to ORIGINAL

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If CSV is malformed (wrong structure, non-numeric data, etc.)
        pd.errors.EmptyDataError: If CSV is empty

    Examples:
        >>> # Load ALS proteomic data
        >>> matrix = load_csv_matrix(Path("aals_cohort1-6_counts_merged.csv"))
        >>> print(f"Loaded {matrix.n_features} genes × {matrix.n_samples} samples")
        Loaded 60665 genes × 578 samples
        >>>
        >>> # Check phenotype distribution
        >>> print(matrix.sample_metadata['phenotype'].value_counts())
        CASE    289
        CTRL    289
        >>>
        >>> # Verify all flags are ORIGINAL
        >>> assert np.all(matrix.quality_flags == QualityFlag.ORIGINAL)
    """
    # Validate path
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Read CSV with first column as index
    try:
        df = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {path}") from e
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {path}: {e}") from e

    # Validate structure
    if df.empty:
        raise ValueError(f"CSV contains no data: {path}")

    if df.shape[0] == 0:
        raise ValueError(f"CSV contains no features (rows): {path}")

    if df.shape[1] == 0:
        raise ValueError(f"CSV contains no samples (columns): {path}")

    # Check for duplicate feature IDs
    if df.index.duplicated().any():
        n_duplicates = df.index.duplicated().sum()
        warnings.warn(
            f"Found {n_duplicates} duplicate feature IDs. "
            "Using first occurrence of each.",
            UserWarning
        )
        df = df[~df.index.duplicated(keep='first')]

    # Check for duplicate sample IDs
    if df.columns.duplicated().any():
        n_duplicates = df.columns.duplicated().sum()
        warnings.warn(
            f"Found {n_duplicates} duplicate sample IDs. "
            "Using first occurrence of each.",
            UserWarning
        )
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Extract components
    feature_ids = pd.Index(df.index)
    sample_ids = pd.Index(df.columns)

    # Convert to numerical matrix
    try:
        data = df.values.astype(float)
    except ValueError as e:
        # Find non-numeric values for better error message
        non_numeric = []
        for i, row in enumerate(df.values):
            for j, val in enumerate(row):
                try:
                    float(val)
                except (ValueError, TypeError):
                    non_numeric.append(f"row {i} ('{df.index[i]}'), col {j} ('{df.columns[j]}'): {val}")
                    if len(non_numeric) >= 5:  # Limit examples
                        break
            if len(non_numeric) >= 5:
                break

        raise ValueError(
            f"CSV contains non-numeric values:\n" +
            "\n".join(f"  - {x}" for x in non_numeric) +
            ("\n  ..." if len(non_numeric) >= 5 else "")
        ) from e

    # Check for NaN values
    if np.isnan(data).any():
        n_nan = np.isnan(data).sum()
        total = data.size
        warnings.warn(
            f"Found {n_nan:,} NaN values ({100*n_nan/total:.2f}% of data). "
            "These will need to be imputed before analysis.",
            UserWarning
        )

    # Check for infinite values
    if np.isinf(data).any():
        n_inf = np.isinf(data).sum()
        raise ValueError(
            f"CSV contains {n_inf} infinite values. "
            "Please clean data before loading."
        )

    # Parse sample metadata
    if infer_phenotypes:
        try:
            sample_metadata = _parse_sample_metadata(sample_ids)
        except Exception as e:
            warnings.warn(
                f"Failed to parse sample metadata: {e}. "
                "Continuing with empty metadata.",
                UserWarning
            )
            sample_metadata = pd.DataFrame(index=sample_ids)
    else:
        sample_metadata = pd.DataFrame(index=sample_ids)

    # Initialize quality flags (all ORIGINAL)
    quality_flags = np.full(data.shape, QualityFlag.ORIGINAL, dtype=int)

    # Create and return BioMatrix
    try:
        matrix = BioMatrix(
            data=data,
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            quality_flags=quality_flags,
        )
    except Exception as e:
        raise ValueError(f"Failed to create BioMatrix: {e}") from e

    return matrix


def _parse_sample_metadata(sample_ids: pd.Index) -> pd.DataFrame:
    """
    Parse biological metadata from sample IDs.

    Expected format: {PHENOTYPE}-{CODE}-{COHORT}-{SUFFIX}_{PLATE}
    Example: CASE-NEUVM674HUA-5257-T_P003

    Extracts:
    - phenotype: First component (CASE, CTRL, etc.)
    - participant_id: Second component (e.g., NEUVM674HUA)
    - cohort: Third component if numeric (e.g., 5257)

    Robust parsing:
    - Handles missing components gracefully
    - Non-numeric cohorts are set to None
    - Malformed IDs get default values

    Args:
        sample_ids: Index of sample identifiers

    Returns:
        DataFrame with columns:
        - phenotype: str (CASE, CTRL, or UNKNOWN)
        - participant_id: Optional[str] (participant identifier or None)
        - cohort: Optional[int] (cohort number or None)

    Examples:
        >>> sample_ids = pd.Index([
        ...     "CASE-NEUVM674HUA-5257-T_P003",
        ...     "CTRL-NEUEU392AE8-5190-T_P001",
        ...     "MALFORMED"
        ... ])
        >>> metadata = _parse_sample_metadata(sample_ids)
        >>> print(metadata)
                                     phenotype participant_id  cohort
        CASE-NEUVM674HUA-5257-T_P003      CASE  NEUVM674HUA  5257.0
        CTRL-NEUEU392AE8-5190-T_P001      CTRL  NEUEU392AE8  5190.0
        MALFORMED                      UNKNOWN           None     NaN
    """
    metadata = []

    for sample_id in sample_ids:
        # Convert to string and split by delimiter
        parts = str(sample_id).split('-')

        # Default values
        phenotype = "UNKNOWN"
        cohort = None

        # Parse phenotype (first part)
        if len(parts) > 0 and parts[0]:
            phenotype = parts[0].strip()

        # Extract participant_id (component 1)
        participant_id = None
        if len(parts) > 1 and parts[1]:
            participant_id = parts[1].strip()

        # Parse cohort (third part, should be numeric)
        if len(parts) > 2:
            cohort_str = parts[2].strip()
            try:
                cohort = int(cohort_str)
            except ValueError:
                # Not a valid integer, leave as None
                pass

        metadata.append({
            'sample_id': sample_id,
            'phenotype': phenotype,
            'participant_id': participant_id,
            'cohort': cohort,
        })

    # Create DataFrame
    df = pd.DataFrame(metadata).set_index('sample_id')

    # Validate index matches
    if not df.index.equals(sample_ids):
        raise ValueError("Internal error: metadata index doesn't match sample_ids")

    return df


def load_matrix(
    path: Path,
    format: str | DataFormat | None = None,
    infer_format: bool = True,
) -> BioMatrix:
    """
    Load biomolecular expression/abundance matrix with explicit format configuration.

    This is the recommended entry point for loading diverse omic data formats.
    It handles delimiter detection, ID extraction from compound keys, column filtering,
    and sample metadata parsing.

    Design Philosophy:
        - Auto-detect what's safe (delimiter, basic structure)
        - Require explicit config for semantic extraction (ID patterns)
        - Provide presets for common formats
        - Fail fast with helpful messages

    Args:
        path: Path to data file (CSV, TSV, or tab-delimited text)
        format: One of:
            - None: Auto-detect (safe for simple formats)
            - str: Preset name (e.g., 'ensembl_csv', 'answerals_proteomics')
            - DataFormat: Custom format configuration
        infer_format: If True and format is None, attempt to suggest format

    Returns:
        BioMatrix with:
        - data: Numerical expression matrix (features x samples)
        - feature_ids: Extracted biomolecular IDs
        - sample_ids: Sample identifiers
        - sample_metadata: Parsed from sample IDs if pattern provided
        - quality_flags: Initialized to ORIGINAL

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If format detection fails or data is malformed
        KeyError: If preset name is not recognized

    Examples:
        >>> # Auto-detect format (for simple CSVs)
        >>> matrix = load_matrix("expression.csv")

        >>> # Use preset for Answer ALS proteomics
        >>> matrix = load_matrix(
        ...     "Data_AnswerALS-436-P_proteomics.txt",
        ...     format='answerals_proteomics'
        ... )

        >>> # Custom format with explicit ID extraction
        >>> from cliquefinder.io.formats import DataFormat
        >>> fmt = DataFormat(
        ...     delimiter='\\t',
        ...     id_pattern=r'\\|(?P<id>[A-Z0-9]+)_HUMAN$',
        ...     drop_columns=['nFragment', 'nPeptide'],
        ... )
        >>> matrix = load_matrix("proteomics.tsv", format=fmt)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Validate path
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Resolve format configuration
    if format is None:
        if infer_format:
            preset_name, fmt = suggest_format(path)
            logger.info(f"Auto-detected format: {preset_name} ({fmt.name})")
        else:
            # Default to generic CSV
            fmt = PRESETS['generic_csv']
    elif isinstance(format, str):
        if format not in PRESETS:
            raise KeyError(
                f"Unknown format preset: '{format}'. "
                f"Available: {list(PRESETS.keys())}"
            )
        fmt = PRESETS[format]
    else:
        fmt = format

    # Determine delimiter
    delimiter = fmt.delimiter
    if delimiter is None:
        delimiter = sniff_delimiter(path)
        logger.debug(f"Sniffed delimiter: {repr(delimiter)}")

    # Read raw data
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            index_col=fmt.index_col,
            skiprows=fmt.skip_rows,
            encoding=fmt.encoding,
            na_values=fmt.na_values,
        )
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Data file is empty: {path}") from e
    except Exception as e:
        raise ValueError(f"Failed to read data file {path}: {e}") from e

    # Validate structure
    if df.empty:
        raise ValueError(f"Data file contains no data: {path}")

    logger.info(f"Raw data: {df.shape[0]} rows x {df.shape[1]} columns")

    # Drop non-data columns
    cols_to_drop = [c for c in df.columns if fmt.should_drop_column(c)]
    if cols_to_drop:
        logger.info(f"Dropping non-data columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Extract biomolecular IDs from index
    raw_index = df.index.tolist()
    if fmt._compiled_id_pattern:
        extracted_ids = []
        extraction_errors = []
        for raw_id in raw_index:
            try:
                extracted_ids.append(fmt.extract_id(raw_id))
            except ValueError as e:
                extraction_errors.append(str(e))
                extracted_ids.append(str(raw_id))  # Fallback to raw

        if extraction_errors:
            n_errors = len(extraction_errors)
            sample_errors = extraction_errors[:3]
            warnings.warn(
                f"ID extraction failed for {n_errors} rows. "
                f"Examples: {sample_errors}. "
                f"Using raw index values for failed rows.",
                UserWarning
            )

        feature_ids = pd.Index(extracted_ids)
        logger.info(f"Extracted {len(feature_ids)} biomolecular IDs")
    else:
        feature_ids = pd.Index(raw_index)

    # Handle duplicate feature IDs
    if feature_ids.duplicated().any():
        n_duplicates = feature_ids.duplicated().sum()
        warnings.warn(
            f"Found {n_duplicates} duplicate feature IDs after extraction. "
            "Using first occurrence of each.",
            UserWarning
        )
        # Need to filter both df and feature_ids
        keep_mask = ~feature_ids.duplicated(keep='first')
        df = df[keep_mask]
        feature_ids = feature_ids[keep_mask]

    # Get sample IDs
    sample_ids = pd.Index(df.columns)

    # Handle duplicate sample IDs
    if sample_ids.duplicated().any():
        n_duplicates = sample_ids.duplicated().sum()
        warnings.warn(
            f"Found {n_duplicates} duplicate sample IDs. "
            "Using first occurrence of each.",
            UserWarning
        )
        df = df.loc[:, ~sample_ids.duplicated(keep='first')]
        sample_ids = pd.Index(df.columns)

    # Convert to numerical matrix
    try:
        data = df.values.astype(float)
    except ValueError as e:
        # Find non-numeric values for better error message
        non_numeric = []
        for i, row in enumerate(df.values):
            for j, val in enumerate(row):
                try:
                    float(val)
                except (ValueError, TypeError):
                    non_numeric.append(
                        f"row {i} ('{feature_ids[i]}'), "
                        f"col {j} ('{sample_ids[j]}'): {val}"
                    )
                    if len(non_numeric) >= 5:
                        break
            if len(non_numeric) >= 5:
                break

        raise ValueError(
            f"Data contains non-numeric values:\n" +
            "\n".join(f"  - {x}" for x in non_numeric) +
            ("\n  ..." if len(non_numeric) >= 5 else "")
        ) from e

    # Check for NaN values
    if np.isnan(data).any():
        n_nan = np.isnan(data).sum()
        total = data.size
        warnings.warn(
            f"Found {n_nan:,} NaN values ({100*n_nan/total:.2f}% of data). "
            "These will need to be imputed before analysis.",
            UserWarning
        )

    # Check for infinite values
    if np.isinf(data).any():
        n_inf = np.isinf(data).sum()
        raise ValueError(
            f"Data contains {n_inf} infinite values. "
            "Please clean data before loading."
        )

    # Parse sample metadata
    if fmt._compiled_sample_pattern:
        metadata_records = []
        for sample_id in sample_ids:
            extracted = fmt.extract_sample_metadata(sample_id)
            extracted['sample_id'] = sample_id
            metadata_records.append(extracted)

        sample_metadata = pd.DataFrame(metadata_records).set_index('sample_id')
        logger.info(f"Extracted metadata columns: {list(sample_metadata.columns)}")
    else:
        sample_metadata = pd.DataFrame(index=sample_ids)

    # Initialize quality flags
    quality_flags = np.full(data.shape, QualityFlag.ORIGINAL, dtype=int)

    # Create BioMatrix
    try:
        matrix = BioMatrix(
            data=data,
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            quality_flags=quality_flags,
        )
    except Exception as e:
        raise ValueError(f"Failed to create BioMatrix: {e}") from e

    logger.info(
        f"Loaded {matrix.n_features:,} features x {matrix.n_samples:,} samples "
        f"({fmt.name})"
    )

    return matrix
