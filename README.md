# Biomolecular Clique Finding

Multi-modal biomolecular data analysis with outlier detection, imputation, and INDRA CoGEx knowledge graph integration.

## Overview

`cliquefinder` is a Python package for analyzing proteomics and transcriptomics data with robust quality control. The pipeline:

1. **Detects outliers** using multi-pass strategies (adjusted-boxplot, residual-based, global cap)
2. **Imputes** missing/outlier values with soft-clip strategy and quality flag tracking
3. **Infers phenotypes** from clinical data (AnswerALS-specific)
4. **Integrates** INDRA CoGEx knowledge graph for regulatory network analysis
5. **Validates** results via enrichment tests and cross-modal analysis

## Installation

```bash
pip install -r requirements.txt
```

For INDRA CoGEx integration, create a `.env` file with credentials:
```
INDRA_NEO4J_URL=bolt://...
INDRA_NEO4J_USER=...
INDRA_NEO4J_PASSWORD=...
```

## Quick Start

### CLI Usage

```bash
# Imputation with configuration file
cliquefinder impute --config configs/proteomics_als.yaml

# Or with CLI arguments
cliquefinder impute \
  --input data.csv \
  --output results/imputed \
  --method adjusted-boxplot \
  --threshold 1.5 \
  --impute-strategy soft-clip

# Analysis
cliquefinder analyze --input results/imputed.csv --output results/analysis
```

### Configuration Files

See `configs/` for YAML/JSON examples:
- `proteomics_als.yaml`: ALS proteomics imputation
- `transcriptomics_als.yaml`: ALS transcriptomics imputation
- Custom configurations supported

## Core Components

### Key Classes

- **BioMatrix**: Core data structure with quality flags and metadata tracking
- **MultiPassOutlierDetector**: Three-pass outlier detection pipeline
  - Adjusted-boxplot method
  - Residual-based detection
  - Global cap enforcement
- **Imputer**: Multiple imputation strategies (soft-clip, knn, mean, median)
- **AnswerALSPhenotypeInferencer**: ALS-specific phenotype inference from clinical data
- **QCVisualizer**: Quality control visualizations
- **CoGExClient**: INDRA CoGEx knowledge graph integration

### Package Modules

```
src/cliquefinder/
  core/              # BioMatrix, transforms, quality flags
  quality/           # Outlier detection, imputation
  io/                # Data loading, phenotype inference
  cli/               # Command-line interface
  viz/               # Visualizations
  knowledge/         # INDRA CoGEx integration
  validation/        # Enrichment tests, ID mapping
```

## Features

- **Multi-pass outlier detection**: Combines statistical methods for robust outlier identification
- **Quality flag tracking**: Every data point tagged with quality status (original, imputed, outlier)
- **Soft-clip imputation**: Preserves data structure while handling outliers
- **Phenotype inference**: Automated clinical phenotype extraction
- **Cross-modal analysis**: Proteomics + transcriptomics integration
- **Knowledge graph queries**: INDRA CoGEx regulatory network analysis
- **Flexible configuration**: YAML/JSON config files or CLI arguments

## Documentation

See `docs/` for detailed documentation:
- Architecture and design decisions
- API reference
- Analysis workflows
- Quality control procedures

## Project Structure

```
src/cliquefinder/    # Main package
  core/              # BioMatrix, transforms, quality flags
  quality/           # Outlier detection, imputation
  io/                # Data loading, phenotype inference
  cli/               # Command-line interface
  viz/               # Visualizations
  knowledge/         # INDRA CoGEx integration
  validation/        # Enrichment tests, ID mapping
scripts/             # Analysis scripts
configs/             # YAML/JSON configuration files
tests/               # Unit tests
docs/                # Documentation
data/                # Input data and caches
results/             # Analysis outputs
```

## Development

Run tests:
```bash
pytest tests/
```

See `docs/` for development guidelines and contribution instructions.
