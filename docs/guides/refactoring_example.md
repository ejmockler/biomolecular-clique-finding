# Phenotype Inference Refactoring Example

This document demonstrates how the original study-specific logic in `scripts/run_proteomics_imputation.py` can be refactored using the new reusable modules.

## Original Code (Lines 86-154)

```python
# Filter out metadata rows (nFragment, nPeptide)
metadata_rows = ['nFragment', 'nPeptide', 'iRT_protein']
data = data[~data.index.str.contains('|'.join(metadata_rows), case=False, na=False)]

# Load portal metadata
portal = pd.read_csv(METADATA_FILE)
guid_to_group = dict(zip(portal['GUID'], portal['SUBJECT_GROUP']))

def extract_guid(sample_id: str) -> str:
    """Extract GUID from sample ID like CASE_NEUAA295HHE-9014-P_D3."""
    parts = sample_id.split('_')
    if len(parts) >= 2:
        return parts[1].split('-')[0]
    return None

def infer_phenotype(sample_id: str) -> tuple[str, str]:
    """Infer phenotype with fallback to sample ID."""
    guid = extract_guid(sample_id)

    # Primary: metadata lookup
    if guid and guid in guid_to_group:
        group = guid_to_group[guid]
        if group == 'ALS':
            return 'CASE', 'metadata'
        elif group == 'Healthy Control':
            return 'CTRL', 'metadata'
        else:
            return group, 'metadata'  # Non-ALS MND, Asymptomatic

    # Fallback: sample ID prefix
    if sample_id.startswith('CTRL_'):
        return 'CTRL', 'sample_id_fallback'
    elif sample_id.startswith('CASE_'):
        return 'CASE', 'sample_id_fallback'

    return 'UNKNOWN', 'no_match'

# Apply inference
sample_phenotypes = {}
sample_sources = {}
for sample_id in data.columns:
    pheno, source = infer_phenotype(sample_id)
    sample_phenotypes[sample_id] = pheno
    sample_sources[sample_id] = source

# Filter to CASE/CTRL only
valid_samples = [s for s, p in sample_phenotypes.items() if p in ('CASE', 'CTRL')]
excluded = [s for s, p in sample_phenotypes.items() if p not in ('CASE', 'CTRL')]

# Filter data
data = data[valid_samples]
```

## Refactored Code (Using New Modules)

```python
from cliquefinder.io import (
    MetadataRowFilter,
    AnswerALSPhenotypeInferencer,
)

# Filter out metadata rows using reusable filter
metadata_filter = MetadataRowFilter(
    patterns=['nFragment', 'nPeptide', 'iRT_protein']
)
filtered_ids = metadata_filter.filter(data.index)
data = data.loc[filtered_ids]

print(f"  Filtered {metadata_filter.n_filtered_} metadata rows")

# Load portal metadata
portal = pd.read_csv(METADATA_FILE)

# Create study-specific phenotype inferencer
phenotype_inferencer = AnswerALSPhenotypeInferencer(
    subject_group_col="SUBJECT_GROUP",
    case_values=["ALS"],
    ctrl_values=["Healthy Control"],
    exclude_values=["Non-ALS MND", "Asymptomatic"],
    subject_id_col="GUID",
)

# Infer phenotypes with provenance tracking
provenance_df = phenotype_inferencer.get_inference_provenance(
    sample_ids=data.columns,
    clinical_df=portal,
)

# Filter to CASE/CTRL only
valid_samples = provenance_df[provenance_df['phenotype'].isin(['CASE', 'CTRL'])]['sample_id']
excluded_samples = provenance_df[~provenance_df['phenotype'].isin(['CASE', 'CTRL'])]['sample_id']

print(f"  CASE samples: {(provenance_df['phenotype'] == 'CASE').sum()}")
print(f"  CTRL samples: {(provenance_df['phenotype'] == 'CTRL').sum()}")
print(f"  Excluded: {len(excluded_samples)} (Non-ALS MND, Asymptomatic, Unknown)")

# Filter data
data = data[valid_samples]
```

## Benefits of Refactoring

### 1. Separation of Concerns
- **Before**: Study-specific logic mixed with data processing
- **After**: Study logic in dedicated `AnswerALSPhenotypeInferencer` class

### 2. Reusability
- **Before**: Can only process ALS AnswerALS data
- **After**: Swap in `GenericPhenotypeInferencer` for other studies (TCGA, GTEx, etc.)

### 3. Testability
- **Before**: Functions embedded in script, hard to test in isolation
- **After**: Classes can be unit tested independently

### 4. Configurability
- **Before**: Hardcoded patterns and mappings
- **After**: Configurable patterns, values, and column names

### 5. Transparency
- **Before**: Manual tracking of sources in separate dictionaries
- **After**: Built-in provenance tracking via `get_inference_provenance()`

## Example: Generic Study Adaptation

To adapt the same pipeline for a different study:

```python
from cliquefinder.io import (
    MetadataRowFilter,
    GenericPhenotypeInferencer,
)

# Different study-specific metadata rows
metadata_filter = MetadataRowFilter(
    patterns=['ERCC-', 'SIRV-']  # RNA-seq spike-ins
)

# Simple phenotype inference for TCGA data
phenotype_inferencer = GenericPhenotypeInferencer(
    phenotype_col="sample_type",
    case_values=["Primary Tumor", "Metastatic"],
    ctrl_values=["Solid Tissue Normal"],
    sample_id_col="barcode",
)

# Same downstream processing...
```

## Migration Path

For scripts like `run_proteomics_imputation.py`:

1. Keep original script for backward compatibility
2. Create new `run_proteomics_imputation_v2.py` using refactored modules
3. Validate outputs match between old and new versions
4. Deprecate original script after validation period

## Class Design Decisions

### Abstract Base Class Pattern
- `PhenotypeInferencer` defines interface contract
- Study-specific implementations provide concrete logic
- Enables polymorphism and dependency injection

### Configuration via Constructor
- All study-specific parameters are constructor arguments
- No hardcoded magic values
- Easy to serialize configurations to YAML/JSON

### Provenance Tracking
- `get_inference_provenance()` returns detailed metadata
- Enables quality assurance and debugging
- Documents how each phenotype was determined
