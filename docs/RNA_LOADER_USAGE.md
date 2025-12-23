# RNA Data Loader Usage Guide

## Overview

The `RNADataLoader` module provides flexible loading of RNA-seq count matrices with automatic ID format detection. This enables cross-modal integration between proteomics and transcriptomics data.

## Installation

The loader is part of the `cliquefinder.knowledge` package:

```python
from cliquefinder.knowledge.rna_loader import RNADataLoader, RNADataset
```

## Basic Usage

### Loading RNA Data with Gene Symbols

```python
from pathlib import Path
from cliquefinder.knowledge.rna_loader import RNADataLoader

# Initialize loader
loader = RNADataLoader()

# Load RNA counts with gene symbols
dataset = loader.load(Path("rna_counts_symbols.csv"))

print(f"Detected ID type: {dataset.id_type}")  # 'symbol'
print(f"Loaded {dataset.n_genes} genes × {dataset.n_samples} samples")
print(f"First 5 genes: {dataset.gene_ids[:5]}")
```

**Expected CSV format:**
```csv
gene_id,Sample1,Sample2,Sample3
TP53,100,150,120
BRCA1,50,75,60
EGFR,200,180,190
```

### Loading RNA Data with Ensembl IDs

```python
# Automatically detects Ensembl gene IDs
dataset = loader.load(Path("rna_counts_ensembl.csv"))

print(f"Detected ID type: {dataset.id_type}")  # 'ensembl_gene'
```

**Expected CSV format:**
```csv
gene_id,Sample1,Sample2,Sample3
ENSG00000141510,100,150,120
ENSG00000012048,50,75,60
ENSG00000146648,200,180,190
```

### Loading RNA Data with Numeric Indices + Annotation

For datasets using numeric row indices (like `numeric__AALS-RNAcountsMatrix.csv`), you must provide an annotation file:

```python
# Load with annotation mapping
dataset = loader.load(
    rna_path=Path("numeric__AALS-RNAcountsMatrix.csv"),
    annotation_path=Path("gene_annotations.csv")
)

print(f"Detected ID type: {dataset.id_type}")  # 'numeric'
print(f"Mapped {dataset.n_genes} numeric IDs to gene identifiers")
```

**RNA counts format (numeric__AALS-RNAcountsMatrix.csv):**
```csv
,Sample1,Sample2,Sample3
0,100,150,120
1,50,75,60
2,200,180,190
```

**Annotation file format (gene_annotations.csv):**
```csv
index,gene_id
0,ENSG00000141510
1,ENSG00000012048
2,ENSG00000146648
```

Or tab-delimited without header:
```tsv
0	TP53
1	BRCA1
2	EGFR
```

## ID Format Detection

The loader automatically detects ID format by analyzing the first 100 row identifiers:

| Pattern | Detected Type | Example |
|---------|---------------|---------|
| `ENSG\d{11}` | `ensembl_gene` | ENSG00000141510 |
| `ENST\d{11}` | `ensembl_transcript` | ENST00000269305 |
| `^\d+$` | `numeric` | 0, 1, 2, 3, ... |
| Other | `symbol` | TP53, BRCA1, EGFR |

### Manual Detection Testing

```python
loader = RNADataLoader()

# Test different ID formats
ensembl_genes = ['ENSG00000141510', 'ENSG00000012048']
print(loader._detect_id_type(ensembl_genes))  # 'ensembl_gene'

symbols = ['TP53', 'BRCA1', 'EGFR']
print(loader._detect_id_type(symbols))  # 'symbol'

numeric = ['0', '1', '2', '3']
print(loader._detect_id_type(numeric))  # 'numeric'
```

## RNADataset Object

The `load()` method returns an `RNADataset` dataclass:

```python
@dataclass
class RNADataset:
    gene_ids: List[str]           # List of gene identifiers
    id_type: str                   # Detected format ('symbol', 'ensembl_gene', etc.)
    n_genes: int                   # Number of genes
    n_samples: int                 # Number of samples
    sample_ids: Optional[List[str]] # Sample identifiers
```

### Accessing Dataset Information

```python
dataset = loader.load(Path("rna_counts.csv"))

# Gene information
print(f"Total genes: {dataset.n_genes}")
print(f"ID type: {dataset.id_type}")
print(f"First gene: {dataset.gene_ids[0]}")
print(f"Last gene: {dataset.gene_ids[-1]}")

# Sample information
print(f"Total samples: {dataset.n_samples}")
if dataset.sample_ids:
    print(f"First sample: {dataset.sample_ids[0]}")
    print(f"All samples: {dataset.sample_ids}")

# String representation
print(dataset)
# Output:
# RNADataset(15000 genes (symbol), 100 samples)
#   First gene: TP53
#   Last gene: ZNF750
```

## Cross-Modal Integration Workflow

Typical usage for proteomics + RNA-seq integration:

```python
from pathlib import Path
from cliquefinder.knowledge.rna_loader import RNADataLoader
from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper
from cliquefinder.io.loaders import load_matrix

# 1. Load proteomics data
proteomics_matrix = load_matrix(Path("proteomics.csv"))
protein_ids = list(proteomics_matrix.feature_ids)

# 2. Load RNA data
rna_loader = RNADataLoader()
rna_dataset = rna_loader.load(
    rna_path=Path("numeric__AALS-RNAcountsMatrix.csv"),
    annotation_path=Path("gene_annotations.csv")
)

# 3. Unify ID spaces
mapper = CrossModalIDMapper()
common_genes = mapper.unify_ids(
    protein_ids=protein_ids,
    rna_ids=rna_dataset.gene_ids,
    rna_id_type=rna_dataset.id_type
)

print(f"Common genes between proteomics and RNA: {len(common_genes)}")
```

## File Format Requirements

### Supported Delimiters
- CSV: Comma-separated (`,`)
- TSV: Tab-separated (`\t`)

Auto-detected by analyzing the first line.

### Required Structure
- **First column**: Gene/feature identifiers (or numeric indices)
- **Remaining columns**: Sample IDs with numerical expression values
- **Header row**: Must include sample identifiers

### Edge Cases

#### Missing Values
```python
# NaN values are allowed but will generate a warning
dataset = loader.load(Path("rna_with_nans.csv"))
# Warning: Found 150 NaN values (0.5% of data)
```

#### Duplicate Gene IDs
```python
# Duplicates generate a warning
dataset = loader.load(Path("rna_with_duplicates.csv"))
# Warning: Found 25 duplicate gene IDs. This may affect downstream analysis.
```

#### Missing Annotation Entries
```python
# Missing annotation entries fall back to numeric IDs
dataset = loader.load(
    rna_path=Path("numeric_counts.csv"),
    annotation_path=Path("partial_annotations.csv")
)
# Warning: Annotation mapping incomplete: 100/60665 IDs not found
```

## Error Handling

### FileNotFoundError
```python
try:
    dataset = loader.load(Path("nonexistent.csv"))
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### ValueError for Numeric IDs without Annotation
```python
try:
    # This will fail if IDs are numeric
    dataset = loader.load(Path("numeric__AALS-RNAcountsMatrix.csv"))
except ValueError as e:
    print(f"Error: {e}")
    # Error: Numeric gene IDs detected, but no annotation file provided.
```

### Malformed Data
```python
try:
    dataset = loader.load(Path("malformed.csv"))
except ValueError as e:
    print(f"Data validation failed: {e}")
```

## Advanced Options

### Custom Detection Sample Size

Control how many IDs are analyzed for format detection:

```python
# Analyze first 50 IDs instead of default 100
dataset = loader.load(
    rna_path=Path("rna_counts.csv"),
    detect_sample_size=50
)
```

### TSV Format

Works automatically with tab-delimited files:

```python
# Automatically detects TSV format
dataset = loader.load(Path("rna_counts.tsv"))
```

## Integration with Existing Pipeline

The RNA loader integrates with the existing cliquefinder architecture:

```python
from cliquefinder.knowledge import ModuleDiscovery, RNAValidatedUniverse
from cliquefinder.knowledge.rna_loader import RNADataLoader
from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

# Load RNA data
rna_loader = RNADataLoader()
rna_data = rna_loader.load(
    rna_path=Path("numeric__AALS-RNAcountsMatrix.csv"),
    annotation_path=Path("gene_annotations.csv")
)

# Create cross-modal mapper
mapper = CrossModalIDMapper()
common_genes = mapper.unify_ids(
    protein_ids=list(proteomics_matrix.feature_ids),
    rna_ids=rna_data.gene_ids,
    rna_id_type=rna_data.id_type
)

# Create RNA-filtered universe
rna_universe = RNAValidatedUniverse(rna_genes=common_genes, mapper=mapper)

# Run module discovery with RNA filtering
discovery = ModuleDiscovery()
result = discovery.discover(
    matrix=proteomics_matrix,
    universe_selector=rna_universe,  # Only consider RNA-measured genes
)
```

## Logging

The loader uses Python's standard logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see detailed messages:
# INFO: Loading RNA data from numeric__AALS-RNAcountsMatrix.csv
# DEBUG: Detected delimiter: ','
# INFO: Loaded 60,665 features × 774 samples
# INFO: Detected ID type: numeric
# INFO: Loading annotation mapping from gene_annotations.csv
# INFO: Loaded 60665 annotation entries
# INFO: Mapped 60665 numeric IDs to gene identifiers
# INFO: Successfully loaded RNA dataset: 60665 genes, 774 samples
```

## Performance Considerations

- **Memory-efficient**: Loads only gene IDs and sample IDs, not the full expression matrix
- **Large files**: Tested with 120MB CSV (60,665 genes × 774 samples)
- **Fast detection**: Only analyzes first 100 IDs for format detection
- **Optimized I/O**: Single-pass reading with pandas

## Testing

Run the test suite:

```bash
cd /Users/noot/Documents/biomolecular-clique-finding
python test_rna_loader.py
```

## Troubleshooting

### "Numeric gene IDs detected, but no annotation file provided"
**Solution**: Provide `annotation_path` parameter with index-to-gene mapping.

### "Annotation mapping incomplete"
**Solution**: Check that annotation file covers all numeric indices in your RNA counts matrix.

### "Found N duplicate gene IDs"
**Solution**: This is a warning. Check your input data for duplicate row identifiers. Duplicates may indicate data quality issues.

### Format not detected correctly
**Solution**: Check that at least 90% of the first 100 IDs match the expected pattern. You may need to preprocess your data.

## See Also

- [CROSS_MODAL_DESIGN.md](CROSS_MODAL_DESIGN.md) - Architecture overview
- `cliquefinder.knowledge.cross_modal_mapper` - ID mapping utilities
- `cliquefinder.knowledge.gene_universes` - RNA-validated universe selectors
