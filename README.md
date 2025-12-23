# Biomolecular Clique Finding

INDRA-driven regulatory coherence analysis for ALS transcriptomics data.

## Overview

This project analyzes transcriptional regulatory networks in ALS (Amyotrophic Lateral Sclerosis) patient-derived motor neurons. The pipeline:

1. **Imputes** missing/outlier values in RNA-seq count data using k-NN with quality tracking
2. **Queries** INDRA CoGEx knowledge graph for regulator-target relationships
3. **Finds** coherent regulatory modules via correlation-based community detection
4. **Compares** disease vs control to identify regulatory rewiring

## Data Sources

- **Expression**: ALS cohort 1-6 RNA-seq counts (60k+ genes x 3000+ samples)
- **Metadata**: Sample-Mapping-File-Feb-2024.csv + acwm.csv clinical data
- **Knowledge**: INDRA CoGEx Neo4j graph (causal regulator-target relationships)

## Pipeline

```bash
# 1. Prepare metadata (links samples to phenotype + sex)
python scripts/prepare_metadata.py \
    --counts aals_cohort1-6_counts_merged.csv \
    --mapping Sample-Mapping-File-Feb-2024.csv \
    --clinical acwm.csv \
    --output data/prepared_metadata.csv

# 2. Impute outliers with quality tracking
python scripts/impute_outliers.py \
    --input aals_cohort1-6_counts_merged.csv \
    --metadata data/prepared_metadata.csv \
    --output results/imputed \
    --k 5

# 3. Run INDRA-driven coherence analysis
python scripts/analyze_tf_coherence.py \
    --input results/imputed/imputed.data.csv \
    --metadata results/imputed/imputed.metadata.csv \
    --output results/coherence \
    --regulators TP53 SOD1 TARDBP FUS \
    --stratify-by phenotype Sex \
    --min-evidence 3
```

## Core Modules

### biocore/
- `BioMatrix`: Immutable expression matrix with quality flags
- `io/`: Data loading and export
- `transforms/`: Imputation, normalization, enrichment
- `knowledge/`: INDRA CoGEx client, coherence analysis

### Key Classes
- `CoGExClient`: Query INDRA Neo4j for regulator targets
- `INDRAModuleExtractor`: Extract regulatory modules from CoGEx
- `CoherenceAnalyzer`: Leiden community detection on correlation networks
- `CliqueValidator`: Validate regulatory coherence via maximal cliques

## Requirements

```bash
pip install -r requirements.txt
```

Requires `.env` file with INDRA CoGEx credentials:
```
INDRA_NEO4J_URL=bolt://...
INDRA_NEO4J_USER=...
INDRA_NEO4J_PASSWORD=...
```

## Project Structure

```
biocore/           # Core library
scripts/           # Analysis pipelines
tests/             # Unit tests
data/              # Input data and caches
results/           # Analysis outputs
docs/              # Specifications
```
