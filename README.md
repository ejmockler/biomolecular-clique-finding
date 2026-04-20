# Cliquefinder

Regulatory gene module discovery in ALS proteomics, integrating INDRA causal knowledge graphs with rotation-based gene set testing.

## What it does

Given quantitative proteomics data and the [INDRA](https://github.com/gyorilab/indra) knowledge base, `cliquefinder` identifies upstream regulatory genes whose downstream targets are collectively disrupted in disease. Three pipelines address this from different angles:

1. **Discovery** (`analyze`) — finds regulators whose INDRA targets form correlated cliques, and identifies where co-regulation changes between disease and control
2. **Differential testing** (`differential`) — tests whether specific gene sets are more differentially expressed than expected, using ROAST rotation and GPU-accelerated permutation
3. **Validation** (`validate-baselines`) — assesses whether an observed enrichment signal is robust to confounders, permutation nulls, and alternative explanations across 7 phases

These pipelines can run independently but form a logical progression: discover, test, validate.

## Installation

```bash
pip install -e .
```

For INDRA CoGEx integration, create a `.env` file:
```
INDRA_NEO4J_URL=bolt://...
INDRA_NEO4J_USER=...
INDRA_NEO4J_PASSWORD=...
```

For belief-scored edge reliability (optional):
```bash
pip install -e ".[belief]"
```

## CLI

```bash
# Regulatory module discovery (knowledge-guided)
cliquefinder analyze --input data.csv --regulators TP53 TARDBP --discover

# Differential gene set testing (ROAST + permutation)
cliquefinder differential --input data.csv --gene-sets targets.json

# Multi-phase validation
cliquefinder validate-baselines --input data.csv --gene C9orf72 --target-set targets.json

# De novo module discovery (data-driven, variance-filtered)
cliquefinder discover --input data.csv --n-genes 5000

# Quality control
cliquefinder impute --input data.csv --method adjusted-boxplot
```

## Architecture

```
src/cliquefinder/
  cli/               # Pipeline entry points (analyze, differential, validate-baselines, discover)
  stats/             # Statistical engines
    rotation.py          — ROAST rotation (Wu et al. 2010): QR, EB moderation, set statistics
    permutation_gpu.py   — Batched OLS with MLX GPU acceleration
    discovery_bridge.py  — Adapts causal-path-scoring to INDRA + ROAST
    hybrid_discovery.py  — Discrete discovery + RWR proximity annotations
    differential.py      — Protein-level differential testing
    clique_analysis.py   — CliqueDefinition, gene set discovery
    graph_permutation.py — Node-label permutation null on INDRA graph
    normalization.py     — Median, quantile, VSN normalization
    target_set.py        — Serializable INDRA target snapshots (v2 with edge metadata)
  knowledge/         # INDRA CoGEx integration
    cogex.py             — Neo4j client, INDRAModule, RegulatorClass
    clique_validator.py  — Correlation-based maximum clique finding
    module_discovery.py  — Orchestrator for both discovery paradigms
    indra_source.py      — Edge reliability via indra-belief noise model
  quality/           # Outlier detection (adjusted-boxplot, MAD-Z), imputation (soft-clip)
  core/              # BioMatrix data structure, transforms, quality flags
  io/                # Data loading, phenotype inference, cohort metadata
  validation/        # Enrichment tests, ID mapping, annotation providers
  viz/               # Plotting (clique networks, heatmaps, QC)
```

## Related packages

- **[causal-path-scoring](../causal-path-scoring/)** — recursive n-hop discovery engine (Storey FDR, knockoff filter, posterior propagation)
- **[indra-belief-model](../indra-belief-model/)** — LLM-based evidence quality scoring for INDRA edges (hard gate composition with recalibrated priors)

## Key statistical methods

- **ROAST** (Wu et al. 2010) — rotation-based gene set testing with EB-moderated t-statistics
- **Storey's π₀-adaptive BH** — FDR control that recovers power when most hypotheses are non-null
- **Seed permutation null** — empirical p-value confirming enrichment is seed-specific
- **Graph-structural extension** — hop extension determined by INDRA topology, not expression data (avoids selective inference)
- **Signed concordance** — INDRA edge direction vs observed t-statistic direction

## Documentation

- **[analysis_reference.md](analysis_reference.md)** — complete description of the scientific question, all three pipelines, output formats, and statistical methodology
- **[docs/](docs/)** — architecture specs, implementation notes, visualization guides

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

Python >= 3.10. ~1700 tests across 16 audit cycles.
