# Biomolecular Clique Finding - Documentation

This directory contains all documentation for the biomolecular clique finding codebase.

## Architecture Documentation

System design specifications and implementation plans for core features.

- [**validation_framework.md**](architecture/validation_framework.md) - Control baseline validation framework: 6-phase architecture (covariate adjustment, specificity, label permutation, matching, negative controls, orchestrator), verdict logic, dependency graph, and file map
- [**validation_methodology_notes.md**](architecture/validation_methodology_notes.md) - Statistical methodology notes for the validation framework: 7 open issues with problem statements, code references, proposed solutions, pitfalls, and priority ranking
- [**outlier_detection.md**](architecture/outlier_detection.md) - Adaptive outlier handling plan with adjusted-boxplot detection and MultiPassOutlierDetector implementation
- [**cross_modal_design.md**](architecture/cross_modal_design.md) - Cross-modal analysis design for integrating multi-omics data
- [**cogex_integration_spec.md**](architecture/cogex_integration_spec.md) - CoGEx knowledge graph integration specification

## User Guides

Step-by-step guides for using key features of the codebase.

- [**phenotype_inference_guide.md**](guides/phenotype_inference_guide.md) - Guide for using the AnswerALSPhenotypeInferencer for phenotype inference
- [**rna_loader_usage.md**](guides/rna_loader_usage.md) - RNA data loader usage and configuration
- [**refactoring_example.md**](guides/refactoring_example.md) - Refactoring patterns and best practices

## Visualization Documentation

Documentation for visualization components and plotting utilities.

- [**regulator_overview_plot.md**](viz/regulator_overview_plot.md) - Regulator overview plot design and usage
- [**regulator_overview_quick_ref.md**](viz/regulator_overview_quick_ref.md) - Quick reference for regulator overview plots
- [**stratum_heatmap_design.md**](viz/stratum_heatmap_design.md) - Stratum heatmap visualization design

## Statistical Methods

Documentation for statistical testing infrastructure.

- [**ROAST_IMPLEMENTATION.md**](ROAST_IMPLEMENTATION.md) - ROAST rotation gene set tests: mathematical foundation, QR decomposition, EB moderation
- [**BOOTSTRAP_IMPLEMENTATION.md**](BOOTSTRAP_IMPLEMENTATION.md) - Direction-aware bootstrap framework
- [**BOOTSTRAP_QUICK_REFERENCE.md**](BOOTSTRAP_QUICK_REFERENCE.md) - Bootstrap quick reference
- [**BATCHED_OLS_IMPLEMENTATION.md**](BATCHED_OLS_IMPLEMENTATION.md) - GPU-accelerated batched OLS implementation
- [**RESULTS_GLOSSARY.md**](RESULTS_GLOSSARY.md) - Glossary of result fields and metrics

## Core Package Structure

The main codebase is organized in `src/cliquefinder/` with the following modules:

- **quality** - Outlier detection (MultiPassOutlierDetector, adjusted-boxplot) and imputation (soft-clip)
- **io** - Phenotype inference (AnswerALSPhenotypeInferencer), data loading
- **cli** - Command-line interface with YAML configuration support
- **stats** - Statistical testing (ROAST, differential, permutation, validation framework)
- **viz** - Visualization utilities
- **knowledge** - Knowledge graph integration

## Archived Documentation

Older documentation and historical design documents.

- [**archive/**](archive/) - Previous versions of documentation and deprecated specs

## Recent Additions

- Control baseline validation framework (6 phases + orchestrator) with hierarchical verdict logic
- Covariate-aware design matrix infrastructure for ROAST, OLS, and GPU paths
- Multi-contrast specificity scoring
- Label permutation null (stratified + free) with lightweight z-score extraction
- Exact covariate matching for sensitivity analysis
- Negative control gene sets for FPR calibration
- MultiPassOutlierDetector with adjusted-boxplot method
- AnswerALSPhenotypeInferencer for phenotype classification
- YAML configuration support in CLI
- Soft-clip imputation strategy
- Cross-modal analysis capabilities
