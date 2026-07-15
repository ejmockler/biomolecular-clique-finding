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
uv sync --extra dev
# or, in a pip-managed environment:
pip install -e ".[dev]"
```

The core/dev environment supports local data and statistical workflows. The
knowledge-graph commands and GSEA producers additionally require the
commit-pinned CoGEx integration extra on Python 3.11 or newer:

```bash
uv sync --extra dev --extra cogex
```

Use the uv-managed path for CoGEx: the project override and `uv.lock` pin the
otherwise unversioned transitive INDRA Git dependency.

Then create a `.env` file for the live CoGEx service:

```
INDRA_NEO4J_URL=bolt://...
INDRA_NEO4J_USER=...
INDRA_NEO4J_PASSWORD=...
```

The client code is pinned by the lockfile, but the remote corpus is an external
input. Publication-facing GSEA artifacts therefore use the offline integrity
receipt in `data/publication/c9_gsea_provenance.json`; exact upstream
regeneration is not claimed unless the corpus state is also frozen.

For belief-scored edge reliability (optional):
```bash
uv sync --extra belief
# or, in a pip-managed environment:
pip install -e ".[belief]"
```

On Python 3.11 or newer, the belief extra is fetched from an immutable commit of the
[`indra-belief-model`](https://github.com/gyorilab/indra-belief-model)
repository because `indra-belief` is not published on PyPI. Installing this
extra therefore requires Git and network access. The core package still
supports Python 3.10, but the optional belief stack does not: its final
Python-3.10-compatible PySB release fails under modern isolated builds.

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
  cli/               # Pipeline entry points and argument validation
  core/              # BioMatrix, transforms, and quality flags
  io/                # Data loading, phenotype inference, and metadata alignment
  knowledge/         # INDRA CoGEx queries, graph construction, and module discovery
  models/            # Reusable numerical model components
  panels/            # Multi-contrast network-landscape analysis
  quality/           # Outlier detection, filtering, and imputation
  stats/             # Statistical engines
    rotation.py          — ROAST rotation (Wu et al. 2010): QR, EB moderation, set statistics
    permutation_gpu.py   — Batched OLS with MLX GPU acceleration
    discovery_bridge.py  — Adapts causal-path-scoring to INDRA + ROAST
    differential.py      — Protein-level differential testing
    clique_analysis.py   — CliqueDefinition, gene set discovery
    network_proximity.py — Network-distance enrichment tests
    perturbation_gradient.py — Per-anchor distance-gradient estimation
    graph_permutation.py — Node-label permutation null on INDRA graph
    normalization.py     — Median, quantile, VSN normalization
    target_set.py        — Serializable INDRA target snapshots (v2 with edge metadata)
    wasc/                 — Within-cluster anchor-slope concordance kernels
  utils/             # Atomic file I/O and numerical helpers
  validation/        # Enrichment tests, ID mapping, annotation providers
  viz/               # Plotting (clique networks, heatmaps, QC)
```

## Related packages

- **`causal-path-scoring`** — optional recursive n-hop discovery engine used in
  some research workspaces; it is not required by the core install.
- **[indra-belief-model](https://github.com/gyorilab/indra-belief-model)** —
  evidence-quality scoring for INDRA edges, available through the pinned
  `belief` extra.

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
# Reproducible locked environment
uv sync --extra dev
uv run --no-sync pytest

# Or an editable pip environment
pip install -e ".[dev]"
pytest
```

Python >= 3.10. The suite contains 2,200+ tests across 100+ test modules.
Run `pytest --collect-only -q` for the exact current count.
