# Cross-Modal Integration Design

## Overview

This document outlines the strategy for integrating RNA-seq data with proteomics clique-finding via INDRA knowledge graph filtering.

## User Request

"When we discover cliques through the INDRA-resolved path, we need to only consider regulators that also appear in a complementary RNA dataset."

## Key Requirements

1. **Optional filtering**: Must be configurable (not always-on)
2. **ID mapping**: Auto-detect biomolecular ID types and unify via INDRA/MyGeneInfo
3. **Cross-modal validation**: Filter INDRA-discovered regulators to those present in RNA data
4. **Flexibility**: Handle multiple RNA data formats (gene symbols, Ensembl IDs, numeric indices + annotation)

## Architecture

### 1. RNA Data Loading (src/cliquefinder/knowledge/rna_loader.py)

```python
class RNADataLoader:
    """
    Flexible RNA data loader supporting:
    - Gene symbols as row indices
    - Ensembl IDs as row indices
    - Numeric indices with separate annotation file
    - Auto-detection of ID format
    """

    def load(self, rna_path: Path, annotation_path: Optional[Path] = None) -> RNADataset:
        """
        Returns:
            RNADataset with gene_ids (list of identifiers) and id_type ('symbol' | 'ensembl' | etc.)
        """
```

### 2. Cross-Modal ID Mapper (src/cliquefinder/knowledge/cross_modal_mapper.py)

```python
class CrossModalIDMapper:
    """
    Maps between proteomics and RNA ID spaces using:
    1. Direct INDRA gene name resolution (via CoGExClient.resolve_gene_name())
    2. Fallback to MyGeneInfo for complex mappings
    """

    def unify_ids(
        self,
        protein_ids: List[str],  # From proteomics (gene symbols)
        rna_ids: List[str],       # From RNA data
        rna_id_type: str          # 'symbol', 'ensembl_gene', etc.
    ) -> Set[str]:
        """
        Returns: Set of unified gene symbols present in both datasets
        """
```

### 3. RNA-Validated Gene Universe (src/cliquefinder/knowledge/gene_universes.py)

```python
class RNAValidatedUniverse(GeneUniverseSelector):
    """
    Filters gene candidates to those present in RNA dataset.

    Use case: When discovering regulators via INDRA, only consider genes
    that also have RNA-seq measurements (transcriptomally active).
    """

    def __init__(self, rna_genes: Set[str], mapper: CrossModalIDMapper):
        self.rna_genes = rna_genes
        self.mapper = mapper

    def select_universe(
        self,
        matrix: BioMatrix,
        num_proteins: Optional[int] = None
    ) -> Set[str]:
        """
        Returns: Intersection of (top variable proteins) AND (RNA-measured genes)
        """
```

### 4. Extended CliqueValidator (src/cliquefinder/knowledge/clique_validator.py)

```python
class CliqueValidator:
    def find_cliques(
        self,
        matrix: BioMatrix,
        universe_selector: GeneUniverseSelector,
        rna_filter: Optional[RNAValidatedUniverse] = None,  # NEW PARAMETER
        ...
    ):
        """
        If rna_filter is provided:
          1. Discover regulators via INDRA (as usual)
          2. Filter regulators to rna_filter.rna_genes before clique finding
          3. Only validate cliques using RNA-measured regulators
        """
```

### 5. CLI Integration (src/cliquefinder/cli/commands.py)

```bash
cliquefinder discover \
  --data results/proteomics/imputed.data.csv \
  --rna-filter numeric__AALS-RNAcountsMatrix.csv \  # NEW OPTION
  --rna-annotation genes.csv \                       # OPTIONAL (if numeric indices)
  --output results/rna_filtered_cliques/
```

## ID Mapping Strategy

### Scenario 1: Both use gene symbols
```
Proteomics: PGK1, TMM11, etc.
RNA:        PGK1, TMM11, etc.
→ Direct string intersection (fast)
```

### Scenario 2: Proteomics uses symbols, RNA uses Ensembl
```
Proteomics: PGK1, BRCA1
RNA:        ENSG00000102144, ENSG00000012048

Mapping via INDRA/MyGeneInfo:
  1. CoGExClient.resolve_gene_name("PGK1") → check HGNC/Ensembl mappings
  2. Fallback: MyGeneInfoMapper.map_ids(['PGK1'], 'symbol', 'ensembl_gene')

→ Unified to gene symbols: {PGK1, BRCA1} ∩ {PGK1, BRCA1} = {PGK1, BRCA1}
```

### Scenario 3: RNA has numeric indices + annotation file
```
RNA counts:     0, 1, 2, ..., 60664
Annotation:     0 → ENSG00000102144
                1 → ENSG00000012048
                ...

→ Load annotation, map Ensembl → symbols, then unify
```

## Implementation Workflow

1. **RNADataLoader**: Detect ID format, load gene list
2. **CrossModalIDMapper**: Unify proteomics and RNA IDs to common namespace (gene symbols)
3. **RNAValidatedUniverse**: Wrap existing variance filtering with RNA intersection
4. **CliqueValidator**: Add optional `rna_filter` parameter
5. **ModuleDiscovery**: Pass `rna_filter` to clique_validator.find_cliques()
6. **CLI**: Add `--rna-filter` and `--rna-annotation` options

## Data Flow

```
User input:
  --data proteomics.csv (features: PGK1, BRCA1, TP53, ...)
  --rna-filter rna_counts.csv (features: ENSG..., or gene symbols)

↓

RNADataLoader:
  rna_genes = ['ENSG00000102144', 'ENSG00000012048', ...]
  rna_id_type = 'ensembl_gene'

↓

CrossModalIDMapper:
  protein_symbols = ['PGK1', 'BRCA1', 'TP53']
  rna_symbols = map_ids(rna_genes, 'ensembl_gene', 'symbol')
             = ['PGK1', 'BRCA1', ...]

  common_genes = protein_symbols ∩ rna_symbols
               = {'PGK1', 'BRCA1'}

↓

RNAValidatedUniverse:
  select_universe(matrix) → VarianceFiltered(matrix) ∩ common_genes
                          → Top 5000 variable proteins that are also in RNA

↓

CliqueValidator:
  discover_regulators(genes) → INDRA query
  filter regulators to RNAValidatedUniverse  # NEW STEP
  find_cliques(filtered_regulators)

↓

Output: Cliques where ALL regulators are transcriptomally active (measured in RNA-seq)
```

## Benefits

1. **Biological validity**: Only consider regulators with RNA evidence (reduces false positives from protein-only artifacts)
2. **Multi-omics integration**: Connects proteomics discoveries to transcriptomics
3. **Flexible**: Works with any ID type via INDRA/MyGeneInfo
4. **Optional**: Doesn't break existing workflows (backward compatible)

## Example Usage

```python
from cliquefinder.knowledge import ModuleDiscovery, RNAValidatedUniverse
from cliquefinder.knowledge.rna_loader import RNADataLoader
from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

# Load RNA data
rna_loader = RNADataLoader()
rna_data = rna_loader.load(
    rna_path="numeric__AALS-RNAcountsMatrix.csv",
    annotation_path="rna_gene_annotations.csv"  # if needed
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
    universe_selector=rna_universe,  # Automatically filters via RNA
    ...
)
```

## Testing Strategy

1. **Unit tests**: RNADataLoader with different formats
2. **Integration tests**: Full pipeline with mock RNA + proteomics data
3. **ID mapping tests**: Verify INDRA → MyGeneInfo fallback
4. **Regression tests**: Ensure no RNA filter = same results as before

## Next Steps

1. ✅ Analyze RNA data structure → DONE (numeric indices, need annotation)
2. ⏳ Implement RNADataLoader
3. ⏳ Implement CrossModalIDMapper
4. ⏳ Extend CliqueValidator
5. ⏳ Create RNAValidatedUniverse
6. ⏳ Update CLI
7. ⏳ Write tests
