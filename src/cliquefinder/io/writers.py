"""
CSV writer for expression matrices.

Writes BioMatrix objects to CSV files for downstream analysis and publication.
Maintains scientific transparency by separating data from quality metadata.

Biological Context:
    After processing (outlier detection, imputation, normalization), analysts need:
    1. The final processed data for analysis
    2. Quality flags showing which values were modified
    3. Metadata for sample annotations

    Two-file approach enables:
    - Analysts: Use processed data directly
    - Reviewers: Verify which values were imputed/corrected
    - Reproducibility: Full audit trail of transformations

Engineering Design:
    - Two-file output: .data.csv and .flags.csv
    - Maintains CSV structure compatible with R/Excel/Python
    - Clear file naming conventions
    - Memory-efficient: single pass writing via pandas
    - Handles large files (90MB+) efficiently

Examples:
    >>> from pathlib import Path
    >>> from cliquefinder.io.writers import write_csv_matrix
    >>>
    >>> # Write both data and quality flags
    >>> write_csv_matrix(matrix, Path("output"), write_quality_flags=True)
    Wrote data matrix to output.data.csv
    Wrote quality flags to output.flags.csv
    >>>
    >>> # Write data only (faster for large files)
    >>> write_csv_matrix(matrix, Path("output"), write_quality_flags=False)
    Wrote data matrix to output.data.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from cliquefinder.core.biomatrix import BioMatrix

__all__ = ['write_csv_matrix']


def write_csv_matrix(
    matrix: BioMatrix,
    path: Path,
    write_quality_flags: bool = True
) -> None:
    """
    Write BioMatrix to CSV file(s).

    Output Files:
    1. {path}.data.csv - Expression matrix with processed values
       - First column: feature IDs
       - Remaining columns: sample IDs with numerical values
       - Contains imputed/corrected values (final state)

    2. {path}.flags.csv - Quality flags (if write_quality_flags=True)
       - Same structure as data file
       - Integer values represent QualityFlag combinations
       - 0 = ORIGINAL (untouched)
       - 1 = OUTLIER_DETECTED
       - 2 = IMPUTED
       - 3 = OUTLIER_DETECTED | IMPUTED
       - etc.

    Scientific Rationale:
        Two-file approach maintains transparency:
        - Methods section: "Outliers were detected and imputed (see .flags.csv)"
        - Reviewers can verify: "Only 2.3% of values were imputed"
        - Reanalysis: Load flags to weight/exclude modified values

    Args:
        matrix: BioMatrix to write
        path: Base path (without extension)
            Example: Path("output") → output.data.csv, output.flags.csv
        write_quality_flags: Whether to write quality flags separately
            Set to False for faster writing of large files if flags not needed

    Raises:
        TypeError: If matrix is not a BioMatrix
        OSError: If path is not writable
        ValueError: If matrix is empty

    Examples:
        >>> from cliquefinder.io.loaders import load_csv_matrix
        >>> from cliquefinder.io.writers import write_csv_matrix
        >>>
        >>> # Load, process, and write
        >>> matrix = load_csv_matrix(Path("raw_data.csv"))
        >>> # ... apply transformations ...
        >>> write_csv_matrix(matrix, Path("processed_data"))
        >>>
        >>> # Round-trip test
        >>> matrix_reloaded = load_csv_matrix(Path("processed_data.data.csv"))
        >>> assert matrix.data.shape == matrix_reloaded.data.shape

    Notes:
        - Creates parent directories if they don't exist
        - Overwrites existing files without warning
        - Uses pandas.DataFrame.to_csv() for efficient writing
        - CSV uses UTF-8 encoding
        - Feature/sample IDs are written as strings (preserves leading zeros)
    """
    # Type validation
    if not isinstance(matrix, BioMatrix):
        raise TypeError(f"matrix must be BioMatrix, got {type(matrix)}")

    # Validate matrix has data
    if matrix.data.size == 0:
        raise ValueError("Cannot write empty matrix")

    # Ensure path is Path object
    if not isinstance(path, Path):
        path = Path(path)

    # Create parent directory if needed
    if path.parent != Path('.') and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(
        matrix.data,
        index=matrix.feature_ids,
        columns=matrix.sample_ids
    )

    # Write data matrix
    data_path = Path(str(path) + ".data.csv")
    try:
        df.to_csv(data_path)
        print(f"Wrote data matrix to {data_path}")
    except Exception as e:
        raise OSError(f"Failed to write data file {data_path}: {e}") from e

    # Write quality flags if requested
    if write_quality_flags:
        flags_df = pd.DataFrame(
            matrix.quality_flags,
            index=matrix.feature_ids,
            columns=matrix.sample_ids
        )
        flags_path = Path(str(path) + ".flags.csv")
        try:
            flags_df.to_csv(flags_path)
            print(f"Wrote quality flags to {flags_path}")
        except Exception as e:
            raise OSError(f"Failed to write flags file {flags_path}: {e}") from e


def write_sample_metadata(
    matrix: BioMatrix,
    path: Path
) -> None:
    """
    Write sample metadata to CSV file.

    Useful for exporting phenotype information, batch labels, and other
    sample-level annotations for downstream analysis or plotting.

    Output:
    - CSV with sample_id as first column
    - Remaining columns: metadata fields (phenotype, cohort, etc.)

    Args:
        matrix: BioMatrix with sample metadata
        path: Output path (will be used exactly as provided)

    Raises:
        TypeError: If matrix is not a BioMatrix
        OSError: If path is not writable

    Examples:
        >>> write_sample_metadata(matrix, Path("sample_annotations.csv"))
        Wrote sample metadata to sample_annotations.csv
        >>>
        >>> # Load in R for plotting
        >>> # metadata <- read.csv("sample_annotations.csv")
        >>> # ggplot(metadata, aes(x=phenotype, y=cohort)) + geom_point()
    """
    # Type validation
    if not isinstance(matrix, BioMatrix):
        raise TypeError(f"matrix must be BioMatrix, got {type(matrix)}")

    # Ensure path is Path object
    if not isinstance(path, Path):
        path = Path(path)

    # Create parent directory if needed
    if path.parent != Path('.') and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Write metadata
    try:
        # Reset index to include sample_id as column
        metadata_with_id = matrix.sample_metadata.reset_index()
        metadata_with_id.to_csv(path, index=False)
        print(f"Wrote sample metadata to {path}")
    except Exception as e:
        raise OSError(f"Failed to write metadata file {path}: {e}") from e
