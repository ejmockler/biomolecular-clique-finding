# Network Query - Quick Reference

## One-Line Usage

```bash
cliquefinder differential --data DATA.csv --metadata META.csv --cliques CLIQUES.csv \
  --output OUTDIR/ --network-query GENE --min-evidence 2
```

## Common Examples

### C9ORF72 ALS Analysis
```bash
cliquefinder differential \
  --data als_proteomics.csv --metadata als_metadata.csv --cliques als_cliques.csv \
  --output results/c9orf72/ --network-query C9ORF72 --min-evidence 2 \
  --permutation-test --n-permutations 1000
```

### TP53 Cancer Analysis
```bash
cliquefinder differential \
  --data cancer_proteomics.csv --metadata cancer_metadata.csv --cliques cancer_cliques.csv \
  --output results/tp53/ --network-query TP53 --min-evidence 3
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--network-query GENE` | Gene symbol to query | None (disabled) |
| `--min-evidence N` | Minimum INDRA evidence count | 1 |
| `--indra-env-file PATH` | Path to INDRA credentials | `/Users/noot/workspace/indra-cogex/.env` |

## INDRA Credentials Setup

### Option 1: Environment Variables
```bash
export INDRA_NEO4J_URL="bolt://hostname:7687"
export INDRA_NEO4J_USER="neo4j"
export INDRA_NEO4J_PASSWORD="password"
```

### Option 2: .env File
```bash
# Create ~/.env or specify via --indra-env-file
cat > ~/.env << EOF
INDRA_NEO4J_URL=bolt://hostname:7687
INDRA_NEO4J_USER=neo4j
INDRA_NEO4J_PASSWORD=password
EOF
```

## Output Files

When `--network-query GENE` is used:

- **`clique_differential.csv`** - All results (with network annotations)
- **`network_filtered_GENE.csv`** - Only cliques containing GENE targets ← **This is the key file**
- **`significant_cliques.csv`** - All significant cliques
- **`analysis_parameters.json`** - Includes network query parameters

### New Columns in Results

- `n_network_targets` - Count of targets in clique
- `network_targets` - Comma-separated list of target gene symbols

## Troubleshooting

### No targets found
```
No targets found for GENE (min_evidence=2)
```
→ Try `--min-evidence 1` or check gene name spelling

### Few targets in dataset
```
5 targets found in dataset (5.0%)
```
→ Normal if gene has tissue-specific targets or limited proteome coverage

### Credentials error
```
Error: INDRA credentials not available
```
→ Set environment variables or create .env file (see above)

## Quick Analysis Pattern

```bash
# 1. Run with network query
cliquefinder differential --network-query GENE ... --output results/

# 2. Check network-filtered results
head results/network_filtered_GENE.csv

# 3. Compare with all results
wc -l results/clique_differential.csv results/network_filtered_GENE.csv
```

## Evidence Threshold Recommendations

| Use Case | `--min-evidence` | Rationale |
|----------|------------------|-----------|
| Exploratory | 1 | Maximum coverage |
| Standard | 2 | Balance coverage/confidence |
| High confidence | 3-5 | Well-established relationships |
| Literature-backed | 5+ | Multiple independent sources |

## Common Genes for ALS Research

```bash
--network-query C9ORF72    # C9orf72 expansion (most common familial ALS)
--network-query SOD1       # SOD1 mutations
--network-query TARDBP     # TDP-43 (also TDP43)
--network-query FUS        # FUS mutations
--network-query OPTN       # Autophagy pathway
--network-query SQSTM1     # p62, autophagy
--network-query TBK1       # Innate immunity
```

## See Full Documentation

- **`NETWORK_QUERY_USAGE.md`** - Complete usage guide
- **`IMPLEMENTATION_SUMMARY_NETWORK_QUERY.md`** - Implementation details
