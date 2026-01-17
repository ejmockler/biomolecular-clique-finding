# Network Query Integration for Differential Analysis

## Overview

The differential analysis CLI now supports querying INDRA CoGEx to identify and filter results based on known regulatory relationships. This enables hypothesis-driven analysis where you can focus on proteins that are known targets of a specific gene of interest.

## Features

- **INDRA CoGEx Integration**: Query mechanistic relationships from literature and pathway databases
- **Automatic ID Mapping**: Maps gene symbols to UniProt IDs in your proteomics data
- **Result Annotation**: Adds network membership annotations to all differential results
- **Network Filtering**: Outputs separate files containing only cliques with network targets
- **Graceful Degradation**: Continues analysis even if INDRA credentials unavailable

## Usage

### Basic Usage

```bash
cliquefinder differential \
  --data proteomics_data.csv \
  --metadata metadata.csv \
  --cliques cliques.csv \
  --output results/ \
  --network-query C9ORF72 \
  --min-evidence 2
```

### Complete Example

```bash
cliquefinder differential \
  --data output/proteomics/imputed.data.csv \
  --metadata output/proteomics/imputed.metadata.csv \
  --cliques output/proteomics/cliques/cliques.csv \
  --output output/differential/c9orf72_targets/ \
  --condition-col phenotype \
  --subject-col subject_id \
  --contrast CASE_vs_CTRL CASE CTRL \
  --network-query C9ORF72 \
  --min-evidence 2 \
  --indra-env-file /path/to/indra-cogex/.env \
  --permutation-test \
  --n-permutations 1000
```

## CLI Arguments

### Network Query Arguments

- `--network-query GENE`: Gene symbol to query for regulatory targets (e.g., "C9ORF72", "TP53")
- `--min-evidence N`: Minimum INDRA evidence count for network edges (default: 1)
  - Evidence count represents number of supporting sources/papers
  - Recommended: 2+ for higher confidence relationships
- `--indra-env-file PATH`: Path to .env file with INDRA CoGEx credentials
  - Default: `/Users/noot/workspace/indra-cogex/.env`

### INDRA Credentials

The system looks for credentials in this order:

1. **Environment variables**:
   ```bash
   export INDRA_NEO4J_URL="bolt://hostname:7687"
   export INDRA_NEO4J_USER="neo4j"
   export INDRA_NEO4J_PASSWORD="password"
   ```

2. **.env file** (specified via `--indra-env-file`):
   ```
   INDRA_NEO4J_URL=bolt://hostname:7687
   INDRA_NEO4J_USER=neo4j
   INDRA_NEO4J_PASSWORD=password
   ```

3. **Explicit parameters** (not supported via CLI, only programmatic API)

## Output Files

When `--network-query` is specified, the CLI produces additional outputs:

### Standard Output Files
- `clique_differential.csv` or `clique_differential_permutation.csv`: All differential results
- `significant_cliques.csv`: Significant cliques (FDR < threshold)
- `analysis_parameters.json`: Analysis parameters and metadata

### Network-Filtered Output Files
- `network_filtered_{GENE}.csv`: Cliques containing targets of the query gene
- `network_filtered_{GENE}_permutation.csv`: (if using permutation testing)

### Added Columns in Results

The following columns are added to differential results:

- `n_network_targets`: Count of proteins in the clique that are INDRA targets
- `network_targets`: Comma-separated list of gene symbols for targets found

## Example Workflow

### 1. Query C9ORF72 targets and run differential analysis

```bash
cliquefinder differential \
  --data als_proteomics.csv \
  --metadata als_metadata.csv \
  --cliques als_cliques.csv \
  --output results/c9orf72_analysis/ \
  --network-query C9ORF72 \
  --min-evidence 2 \
  --permutation-test \
  --n-permutations 1000
```

**Expected output**:
```
Querying INDRA CoGEx for C9ORF72 regulatory targets...
  Found 87 INDRA targets for C9ORF72
  Evidence counts: min=2, max=15, median=3.0

Mapping 5432 feature IDs to gene symbols...
  Mapped 5218 symbols/aliases

Network query results:
  C9ORF72 -> 87 INDRA targets
  61 targets found in dataset (70.1%)
  Example targets: TARDBP, FUS, SQSTM1, OPTN, TBK1

Running competitive permutation clique analysis...
[... analysis output ...]

Network filtering: 61 INDRA targets of C9ORF72

Network-filtered results: results/c9orf72_analysis/network_filtered_C9ORF72_permutation.csv
  23 cliques contain targets of C9ORF72
  8 are significant
```

### 2. Examine network-filtered results

```python
import pandas as pd

# Load network-filtered results
df = pd.read_csv('results/c9orf72_analysis/network_filtered_C9ORF72_permutation.csv')

# Show cliques with most network targets
top_cliques = df.nsmallest(10, 'perm_pvalue')[
    ['clique_id', 'n_network_targets', 'network_targets', 'log2FC', 'perm_pvalue']
]
print(top_cliques)
```

### 3. Compare with and without network filtering

```python
# Load all results
all_results = pd.read_csv('results/c9orf72_analysis/clique_differential_permutation.csv')

# Load network-filtered results
network_results = pd.read_csv('results/c9orf72_analysis/network_filtered_C9ORF72_permutation.csv')

print(f"Total cliques: {len(all_results)}")
print(f"Cliques with C9ORF72 targets: {len(network_results)}")
print(f"Significant (all): {sum(all_results['is_significant'])}")
print(f"Significant (network): {sum(network_results['is_significant'])}")
```

## Implementation Details

### Query Flow

1. **Load Data**: Load proteomics data and feature IDs
2. **Query INDRA**:
   - Initialize `INDRAKnowledgeSource` with credentials
   - Query for targets of specified gene
   - Filter by minimum evidence count
3. **Map IDs**:
   - Use `map_feature_ids_to_symbols()` to convert UniProt/Ensembl IDs to gene symbols
   - Match INDRA targets against dataset features
4. **Run Analysis**: Standard differential analysis proceeds
5. **Annotate Results**:
   - Add `n_network_targets` and `network_targets` columns
   - Filter to cliques containing network targets
   - Save network-filtered results

### Helper Function

```python
def query_network_targets(
    gene_symbol: str,
    feature_ids: list[str],
    min_evidence: int = 1,
    env_file: Path = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Query INDRA CoGEx for regulatory targets and map to UniProt IDs in data.

    Returns:
        Dict mapping {gene_symbol: uniprot_id} for targets found in data
    """
```

### Key Components

- **INDRA Source**: `cliquefinder.knowledge.indra_source.INDRAKnowledgeSource`
- **ID Mapping**: `cliquefinder.stats.clique_analysis.map_feature_ids_to_symbols()`
- **CoGEx Client**: `cliquefinder.knowledge.cogex.CoGExClient`

## Use Cases

### 1. ALS Genetic Subtypes

Focus on known targets of ALS-associated genes:

```bash
# C9ORF72 expansion carriers
cliquefinder differential ... --network-query C9ORF72

# SOD1 mutation carriers
cliquefinder differential ... --network-query SOD1

# TARDBP mutation carriers
cliquefinder differential ... --network-query TARDBP
```

### 2. Master Regulator Analysis

Identify dysregulated targets of key transcription factors:

```bash
# TP53 targets
cliquefinder differential ... --network-query TP53 --min-evidence 3

# MYC targets
cliquefinder differential ... --network-query MYC --min-evidence 3
```

### 3. Pathway-Focused Analysis

Narrow analysis to specific signaling pathways:

```bash
# mTOR pathway
cliquefinder differential ... --network-query MTOR --min-evidence 2

# NF-kB pathway
cliquefinder differential ... --network-query NFKB1 --min-evidence 2
```

## Troubleshooting

### INDRA Credentials Error

```
Error: Network query failed: INDRA credentials not available
```

**Solution**: Ensure credentials are set in environment or .env file:
```bash
export INDRA_NEO4J_URL="bolt://hostname:7687"
export INDRA_NEO4J_USER="neo4j"
export INDRA_NEO4J_PASSWORD="password"
```

### No Targets Found

```
No targets found for GENE (min_evidence=2)
```

**Possible causes**:
- Gene name not recognized by INDRA (try alternate names)
- Evidence threshold too high (try `--min-evidence 1`)
- Gene has no known regulatory targets in INDRA

### Few Targets in Dataset

```
Network query results:
  GENE -> 100 INDRA targets
  5 targets found in dataset (5.0%)
```

**Possible causes**:
- ID mapping mismatch (check feature ID format)
- Limited proteome coverage
- Targets not expressed in your experimental system

## Best Practices

1. **Evidence Thresholds**: Start with `--min-evidence 2` for balance between coverage and confidence
2. **Check Coverage**: Review the "targets found in dataset" percentage - low coverage may indicate ID mapping issues
3. **Compare with All Results**: Always compare network-filtered results with unfiltered results for context
4. **Document Parameters**: The `analysis_parameters.json` file records all network query settings
5. **Validate Gene Names**: INDRA uses HGNC gene symbols - check your gene name is correct

## References

- **INDRA**: Automated assembly of causal knowledge from literature (https://indra.bio/)
- **CoGEx**: INDRA's knowledge graph interface (https://github.com/indralab/indra_cogex)
- **HGNC**: HUGO Gene Nomenclature Committee (https://www.genenames.org/)

## Citation

If you use this network query functionality, please cite:

1. **INDRA**: Gyori BM, et al. (2017). From word models to executable models of signaling networks using automated assembly. *Molecular Systems Biology*.

2. **CoGEx**: Hoyt CT, et al. (2022). INDRA CoGEx: A database for exploring mechanism and causality in COVID-19 and other diseases. *JAMA Network Open*.
