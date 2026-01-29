# AnswerALS-Specific Analysis Examples

This directory contains examples demonstrating how to use the generic permutation testing framework for **AnswerALS-specific research questions**.

## Overview

The examples in this directory show how to apply the framework from `cliquefinder.stats.permutation_framework` to real AnswerALS data scenarios. They demonstrate:

1. **Custom experimental designs** derived from AnswerALS metadata
2. **Genetic subtype comparisons** (C9orf72 vs Sporadic, Familial vs Sporadic)
3. **Proper metadata handling** for complex cohort definitions

These are **not general-purpose utilities** but rather **experiment-specific demonstrations** of how to use the generic framework.

## Files

### Data Processing and Imputation

#### `run_proteomics_imputation.py`

Complete example of AnswerALS proteomics imputation pipeline using the cliquefinder API:

- **Load**: AnswerALS proteomics matrix (436 participants)
- **Phenotype inference**: GUID-based lookup with sample ID fallback
- **Multi-pass outlier detection**: Adaptive boxplot with residual and global cap
- **Soft-clip imputation**: Gradual winsorization with global bounds fallback
- **Quality control**: Comprehensive visualization suite

**Usage**:
```bash
python examples/als/run_proteomics_imputation.py
```

**Required data files**:
- `Data_AnswerALS-436-P_proteomics-protein-matrix_correctedImputed.txt`
- `aals_dataportal_datatable.csv`

#### `run_sporadic_imputation.sh`

Shell script demonstrating the cliquefinder CLI for sporadic ALS analysis:

- **Dataset**: 357 samples (170 sporadic ALS, 187 controls)
- **Exclusion**: C9orf72 carriers, familial ALS, other mutations
- **Method**: MAD-Z outlier detection with soft-clip imputation
- **Sex classification**: Within-phenotype sex prediction

**IMPORTANT**: Contains absolute paths that must be customized before running.

**Usage**:
```bash
# Update PROJECT_DIR in the script first
bash examples/als/run_sporadic_imputation.sh
```

#### `create_sporadic_metadata.py`

AnswerALS-specific metadata filtering to identify sporadic vs familial ALS:

- **Input**: AnswerALS portal metadata (`aals_dataportal_datatable.csv`)
- **Logic**: Identifies familial cases based on mutation columns
- **Output**: Filtered metadata with relabeled familial samples

**Usage**:
```bash
python examples/als/create_sporadic_metadata.py
```

### Experimental Design

#### `genetic_contrasts.py`

Demonstrates creating experimental designs for genetic subtype comparisons:

- **`create_c9_vs_sporadic_design()`**: Compare C9orf72 carriers vs sporadic ALS
- **`create_sod1_vs_sporadic_design()`**: Compare SOD1 carriers vs sporadic ALS
- **`create_familial_vs_sporadic_design()`**: Compare all familial vs sporadic ALS

Each function returns a `MetadataDerivedDesign` that can be used directly with `PermutationTestEngine`.

**Key Pattern**: These functions show how to derive complex condition labels from existing metadata columns using the `MetadataDerivedDesign` class.

### Knowledge Graph Queries

#### `graph_queries.py`

Generic and ALS-specific functions for querying biological knowledge graphs:

- **`get_gene_neighbor_sets(gene_name, ...)`**: Generic function to get 1-hop neighbors for any gene
- **`get_c9orf72_neighbor_sets(...)`**: Convenience wrapper specifically for C9orf72
- **`get_als_gene_neighbor_sets(...)`**: Query neighbors for all major ALS genes (C9orf72, SOD1, TARDBP, FUS)
- **`get_gene_neighbors_custom_categories(...)`**: Query neighbors with custom relationship type groupings

Each function returns a dictionary with keys: `"activated"`, `"inhibited"`, `"all"` containing `QueryResult` objects that can be directly converted to `FeatureSet` for permutation testing.

**Key Pattern**: These functions demonstrate how to use the generic `GraphQuery` API from the core library to build gene-specific queries, split results by relationship type, and integrate with statistical analysis.

**Example**:
```python
from examples.als.graph_queries import get_gene_neighbor_sets
from cliquefinder.knowledge import INDRAKnowledgeSource

source = INDRAKnowledgeSource(env_file=".env")
gene_universe = set(matrix.feature_ids)

# Get C9orf72 neighbors
neighbors = get_gene_neighbor_sets("C9orf72", source, gene_universe)

# Convert to feature sets for permutation testing
activated_fs = neighbors["activated"].to_feature_set("C9_activated")
inhibited_fs = neighbors["inhibited"].to_feature_set("C9_inhibited")

# Run competitive test
results = engine.run_competitive_test([activated_fs, inhibited_fs], design, pool)
```

#### `example_graph_queries.py`

Comprehensive examples showing how to use the knowledge graph query functions:

1. **Single gene query**: Query neighbors for one gene (e.g., C9orf72)
2. **Multiple ALS genes**: Comparative analysis across C9orf72, SOD1, TARDBP, FUS
3. **Custom relationship categories**: Define custom groupings (transcriptional, post-translational, etc.)
4. **Integration with permutation testing**: Full workflow from query to statistical test
5. **Comparative analysis pattern**: Compare regulation across multiple genes

**Usage**:
```bash
python examples/als/example_graph_queries.py
```

## Usage Example

```python
from examples.als.genetic_contrasts import create_c9_vs_sporadic_design
from cliquefinder.stats.permutation_framework import (
    PermutationTestEngine,
    MedianPolishSummarizer,
    MixedModelStatistic,
)

# Create experimental design
design = create_c9_vs_sporadic_design(blocking_column="subject_id")

# Setup permutation testing engine
engine = PermutationTestEngine(
    data=proteomics_matrix,  # features × samples (numpy array)
    feature_ids=protein_ids,  # list of protein/gene IDs
    metadata=sample_metadata,  # pandas DataFrame with metadata
    summarizer=MedianPolishSummarizer(),
    test=MixedModelStatistic(),
)

# Run competitive permutation test
results = engine.run_competitive_test(
    feature_sets=protein_cliques,  # list of FeatureSet objects
    design=design,
    feature_pool=all_regulated_proteins,  # background for null distribution
    n_permutations=1000,
)

# Examine results
for result in results:
    if result.is_significant:
        print(f"{result.feature_set_id}: p={result.empirical_pvalue:.4f}")
```

## Running the Examples

The examples can be run standalone to demonstrate the design patterns:

```bash
cd /Users/noot/Documents/biomolecular-clique-finding
python examples/als/genetic_contrasts.py
```

This will create sample metadata and show how the derivation functions assign samples to genetic subtypes.

## Relationship to Generic Framework

The generic framework lives in `cliquefinder.stats.permutation_framework` and provides:

- **Protocol-based abstractions** (FeatureSet, ExperimentalDesign, Summarizer, StatisticalTest)
- **Core engine** (PermutationTestEngine)
- **Reusable implementations** (TwoGroupDesign, MetadataDerivedDesign, etc.)

These examples demonstrate **how to apply** those generic tools to AnswerALS-specific questions. They are:

- ✅ **Good examples** of how to use the framework
- ✅ **Reusable patterns** for other AnswerALS analyses
- ❌ **Not general-purpose** utilities (experiment-specific)
- ❌ **Not part of core library** (application layer, not framework layer)

## Adding New Examples

When adding AnswerALS-specific examples:

1. **Create standalone, runnable scripts** with clear docstrings
2. **Show complete usage patterns** from data loading to results
3. **Document metadata requirements** (required columns, expected values)
4. **Explain biological context** (what comparison, why it matters)
5. **Keep experiment-specific logic here**, not in core framework

## AnswerALS Metadata Reference

Common metadata columns used in these examples:

- `phenotype`: CASE (ALS) or CTRL (control)
- `ClinReport_Mutations_Details`: Genetic mutation information
- `subject_id`: Subject identifier for blocking/random effects
- `sample_id`: Unique sample identifier
- `age_at_collection`: Age at sample collection
- `sex`: Biological sex

For full metadata schema, see AnswerALS documentation.
