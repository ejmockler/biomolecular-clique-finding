# Phenotype Inference Module - Developer Guide

## Overview

The phenotype inference module (`cliquefinder.io.phenotype`) and data filters module (`cliquefinder.io.data_filters`) provide reusable, study-specific logic for processing biomolecular datasets. These modules enable clean separation of concerns between generic data processing pipelines and study-specific business rules.

## Quick Start

### Basic Usage

```python
from cliquefinder.io import (
    AnswerALSPhenotypeInferencer,
    MetadataRowFilter,
)
import pandas as pd

# 1. Filter out technical metadata rows
metadata_filter = MetadataRowFilter(
    patterns=['nFragment', 'nPeptide', 'iRT_protein']
)
filtered_ids = metadata_filter.filter(data.index)
data_clean = data.loc[filtered_ids]

# 2. Load clinical metadata
clinical_df = pd.read_csv("portal_metadata.csv")

# 3. Infer phenotypes
inferencer = AnswerALSPhenotypeInferencer()
phenotypes = inferencer.infer(data_clean.columns, clinical_df)

# 4. Get detailed provenance
provenance = inferencer.get_inference_provenance(data_clean.columns, clinical_df)
```

## Module Components

### 1. Data Filters (`cliquefinder.io.data_filters`)

#### MetadataRowFilter

Filters features based on substring patterns.

```python
from cliquefinder.io import MetadataRowFilter

# Proteomics: Remove QC metrics
filter = MetadataRowFilter(
    patterns=['nFragment', 'nPeptide', 'iRT_protein']
)

# RNA-seq: Remove spike-in controls
filter = MetadataRowFilter(
    patterns=['ERCC-', 'SIRV-']
)

# Apply filter
filtered_ids = filter.filter(feature_ids)
print(f"Filtered {filter.n_filtered_} features")

# Get excluded IDs for QC
excluded_ids = filter.get_filtered_ids(feature_ids)
```

#### RegexMetadataRowFilter

Filters features using full regex patterns.

```python
from cliquefinder.io import RegexMetadataRowFilter

# Filter features starting with "iRT" or ending with "_QC"
filter = RegexMetadataRowFilter(
    patterns=[r"^iRT", r"_QC$"]
)
```

### 2. Phenotype Inference (`cliquefinder.io.phenotype`)

#### PhenotypeInferencer (Abstract Base Class)

Defines the interface for all phenotype inferencer implementations.

```python
from cliquefinder.io.phenotype import PhenotypeInferencer

class MyStudyInferencer(PhenotypeInferencer):
    def infer(self, sample_ids, clinical_df=None):
        # Custom inference logic
        pass
```

#### AnswerALSPhenotypeInferencer

ALS AnswerALS study-specific implementation.

```python
from cliquefinder.io import AnswerALSPhenotypeInferencer

inferencer = AnswerALSPhenotypeInferencer(
    subject_group_col="SUBJECT_GROUP",
    case_values=["ALS"],
    ctrl_values=["Healthy Control"],
    exclude_values=["Non-ALS MND", "Asymptomatic"],
    sample_id_pattern=r"^(?:CASE|CTRL)_([A-Z0-9]+)",
    subject_id_col="GUID",
)

# Infer phenotypes
phenotypes = inferencer.infer(sample_ids, clinical_df)

# Get provenance
provenance = inferencer.get_inference_provenance(sample_ids, clinical_df)
```

**Logic flow:**
1. Extract participant GUID from sample ID using regex pattern
2. Look up SUBJECT_GROUP in clinical metadata
3. Map SUBJECT_GROUP to CASE/CTRL/EXCLUDE
4. Fallback to sample ID prefix (CASE_/CTRL_) if metadata missing
5. Mark as EXCLUDE if no match

**Provenance columns:**
- `sample_id`: Sample identifier
- `phenotype`: CASE/CTRL/EXCLUDE
- `source`: metadata/sample_id_fallback/no_match
- `subject_id`: Extracted GUID
- `subject_group`: Raw SUBJECT_GROUP value

#### GenericPhenotypeInferencer

Simple metadata column-based inference.

```python
from cliquefinder.io import GenericPhenotypeInferencer

# TCGA example
inferencer = GenericPhenotypeInferencer(
    phenotype_col="sample_type",
    case_values=["Primary Tumor", "Metastatic"],
    ctrl_values=["Solid Tissue Normal"],
    sample_id_col="barcode",
)

phenotypes = inferencer.infer(sample_ids, clinical_df)
```

**Provenance columns:**
- `sample_id`: Sample identifier
- `phenotype`: CASE/CTRL/EXCLUDE
- `source`: Always 'metadata'
- `raw_value`: Raw value from phenotype column

## Common Use Cases

### Use Case 1: ALS Proteomics Pipeline

```python
from cliquefinder.io import (
    AnswerALSPhenotypeInferencer,
    MetadataRowFilter,
)

# Filter technical rows
metadata_filter = MetadataRowFilter(['nFragment', 'nPeptide', 'iRT_protein'])
data = data.loc[metadata_filter.filter(data.index)]

# Infer phenotypes
inferencer = AnswerALSPhenotypeInferencer()
provenance = inferencer.get_inference_provenance(data.columns, clinical_df)

# Filter to CASE/CTRL only
valid = provenance[provenance['phenotype'].isin(['CASE', 'CTRL'])]
data = data[valid['sample_id']]
```

### Use Case 2: TCGA RNA-seq Pipeline

```python
from cliquefinder.io import (
    GenericPhenotypeInferencer,
    RegexMetadataRowFilter,
)

# Filter spike-ins
spike_filter = RegexMetadataRowFilter([r'^ERCC-', r'^SIRV-'])
data = data.loc[spike_filter.filter(data.index)]

# Infer phenotypes
inferencer = GenericPhenotypeInferencer(
    phenotype_col="sample_type",
    case_values=["Primary Tumor", "Recurrent Tumor"],
    ctrl_values=["Solid Tissue Normal"],
    sample_id_col="barcode",
)
phenotypes = inferencer.infer(data.columns, clinical_df)
```

### Use Case 3: Custom Study Implementation

```python
from cliquefinder.io.phenotype import PhenotypeInferencer
import pandas as pd

class MyCustomInferencer(PhenotypeInferencer):
    def __init__(self, age_threshold=50):
        self.age_threshold = age_threshold

    def infer(self, sample_ids, clinical_df=None):
        phenotypes = pd.Series(index=sample_ids, dtype=object)

        # Custom logic: CASE if age > threshold
        for sample_id in sample_ids:
            row = clinical_df[clinical_df['sample_id'] == sample_id]
            if not row.empty:
                age = row.iloc[0]['age']
                phenotypes[sample_id] = 'CASE' if age > self.age_threshold else 'CTRL'
            else:
                phenotypes[sample_id] = 'EXCLUDE'

        return phenotypes

# Use custom inferencer
inferencer = MyCustomInferencer(age_threshold=60)
phenotypes = inferencer.infer(sample_ids, clinical_df)
```

## Design Patterns

### 1. Strategy Pattern

Swap inferencer implementations without changing pipeline code:

```python
def run_pipeline(data, clinical_df, inferencer):
    """Generic pipeline accepting any PhenotypeInferencer."""
    phenotypes = inferencer.infer(data.columns, clinical_df)
    # ... rest of pipeline
    return processed_data

# Use with different studies
als_result = run_pipeline(data, clinical, AnswerALSPhenotypeInferencer())
tcga_result = run_pipeline(data, clinical, GenericPhenotypeInferencer(...))
```

### 2. Configuration Objects

Serialize inferencer configuration to YAML/JSON:

```python
import yaml

config = {
    'inferencer': 'AnswerALSPhenotypeInferencer',
    'params': {
        'subject_group_col': 'SUBJECT_GROUP',
        'case_values': ['ALS'],
        'ctrl_values': ['Healthy Control'],
        'exclude_values': ['Non-ALS MND', 'Asymptomatic'],
    },
    'filters': {
        'metadata_patterns': ['nFragment', 'nPeptide', 'iRT_protein'],
    },
}

# Save configuration
with open('pipeline_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Load and apply
with open('pipeline_config.yaml') as f:
    config = yaml.safe_load(f)

inferencer_class = globals()[config['inferencer']]
inferencer = inferencer_class(**config['params'])
```

### 3. Provenance Tracking

Always track how phenotypes were inferred:

```python
# Get detailed provenance
provenance = inferencer.get_inference_provenance(sample_ids, clinical_df)

# Save for reproducibility
provenance.to_csv('phenotype_provenance.csv', index=False)

# Audit inference sources
print(provenance['source'].value_counts())
# metadata                 150
# sample_id_fallback        10
# no_match                   5
```

## Testing

See `tests/test_phenotype_inference.py` for comprehensive test examples.

```python
import pytest
from cliquefinder.io import AnswerALSPhenotypeInferencer

def test_basic_inference():
    sample_ids = pd.Index(["CASE_NEUAA295HHE-9014-P_D3"])
    clinical_df = pd.DataFrame({
        "GUID": ["NEUAA295HHE"],
        "SUBJECT_GROUP": ["ALS"],
    })

    inferencer = AnswerALSPhenotypeInferencer()
    phenotypes = inferencer.infer(sample_ids, clinical_df)

    assert phenotypes["CASE_NEUAA295HHE-9014-P_D3"] == "CASE"
```

## Migration Guide

### Migrating Existing Scripts

**Before:**
```python
# Hardcoded study-specific logic
metadata_rows = ['nFragment', 'nPeptide']
data = data[~data.index.str.contains('|'.join(metadata_rows))]

def extract_guid(sample_id):
    return sample_id.split('_')[1].split('-')[0]

def infer_phenotype(sample_id):
    guid = extract_guid(sample_id)
    if guid in guid_to_group:
        group = guid_to_group[guid]
        if group == 'ALS':
            return 'CASE'
    return 'EXCLUDE'
```

**After:**
```python
# Reusable modules
from cliquefinder.io import (
    AnswerALSPhenotypeInferencer,
    MetadataRowFilter,
)

filter = MetadataRowFilter(['nFragment', 'nPeptide'])
data = data.loc[filter.filter(data.index)]

inferencer = AnswerALSPhenotypeInferencer()
phenotypes = inferencer.infer(data.columns, clinical_df)
```

## Best Practices

1. **Always use provenance tracking** - Helps debugging and quality assurance
2. **Configure via constructors** - Avoid hardcoded values
3. **Test with edge cases** - Missing metadata, malformed IDs, unknown groups
4. **Document study-specific assumptions** - What do CASE/CTRL mean for your study?
5. **Version your configurations** - Track changes to inference logic
6. **Validate outputs** - Compare with manual annotations

## API Reference

### MetadataRowFilter

**Constructor:**
```python
MetadataRowFilter(patterns: Sequence[str], case_sensitive: bool = False)
```

**Methods:**
- `filter(feature_ids: pd.Index) -> pd.Index`: Apply filter
- `get_filtered_ids(feature_ids: pd.Index) -> pd.Index`: Get excluded IDs

**Attributes:**
- `n_filtered_`: Number of features filtered in last call

### AnswerALSPhenotypeInferencer

**Constructor:**
```python
AnswerALSPhenotypeInferencer(
    subject_group_col: str = "SUBJECT_GROUP",
    case_values: list[str] = ["ALS"],
    ctrl_values: list[str] = ["Healthy Control"],
    exclude_values: list[str] = ["Non-ALS MND", "Asymptomatic"],
    sample_id_pattern: str | None = None,
    subject_id_col: str = "GUID",
)
```

**Methods:**
- `infer(sample_ids: pd.Index, clinical_df: pd.DataFrame | None) -> pd.Series`
- `get_inference_provenance(sample_ids: pd.Index, clinical_df: pd.DataFrame | None) -> pd.DataFrame`

### GenericPhenotypeInferencer

**Constructor:**
```python
GenericPhenotypeInferencer(
    phenotype_col: str,
    case_values: list[str],
    ctrl_values: list[str],
    exclude_values: list[str] | None = None,
    sample_id_col: str = "sample_id",
)
```

**Methods:**
- `infer(sample_ids: pd.Index, clinical_df: pd.DataFrame | None) -> pd.Series`
- `get_inference_provenance(sample_ids: pd.Index, clinical_df: pd.DataFrame | None) -> pd.DataFrame`

## Further Reading

- `docs/refactoring_example.md` - Detailed refactoring example
- `tests/test_phenotype_inference.py` - Comprehensive test suite
- `scripts/run_proteomics_imputation.py` - Original implementation (for comparison)
