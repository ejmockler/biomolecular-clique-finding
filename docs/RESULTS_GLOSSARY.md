# Results Glossary

Provenance-layered reference for every result artifact produced by CliqueFinder.
Intended for paper authors, methods-section writers, and reviewers.

Each entry is documented at four depths:

| Layer | Audience | Answers |
|-------|----------|---------|
| **Artifact** | First author | What is this file? |
| **Semantics** | Methods writer | What does each column mean? |
| **Method** | Reviewer | How was it computed? (with citation) |
| **Provenance** | Auditor | Where did the inputs come from? |

---

## Table of Contents

- [Foundational Concepts](#foundational-concepts)
- [Workflow 1: Knowledge-Guided Coherence (`analyze`)](#workflow-1-knowledge-guided-coherence-analysis)
- [Workflow 2: Differential Abundance (`differential`)](#workflow-2-differential-abundance-testing)
- [Workflow 3: Method Comparison (`compare`)](#workflow-3-method-comparison)
- [Workflow 4: De Novo Discovery (`discover`)](#workflow-4-de-novo-module-discovery)
- [Workflow 5: Preprocessing (`impute`)](#workflow-5-preprocessing-and-imputation)
- [Workflow 6: Sensitivity Analysis (`sensitivity`)](#workflow-6-sensitivity-analysis)
- [Statistical Methods Reference](#statistical-methods-reference)
- [Knowledge Layer Reference](#knowledge-layer-reference)
- [Column Dictionary](#column-dictionary)

---

## Foundational Concepts

### Feature

A molecular entity measured in the expression/abundance matrix. Rows of the
input matrix. Identified by Ensembl gene IDs (e.g., `ENSG00000000003`) or
UniProt accessions (e.g., `A0AVT1`). In proteomics workflows, a feature is a
protein; in transcriptomics, a transcript or gene.

### Sample

A biological specimen. Columns of the input matrix. Sample IDs encode metadata
in the format `{PHENOTYPE}_{PARTICIPANT}-{COHORT}-{PLATE}` (e.g.,
`CASE_NEUVM674HUA-5257-P_P003`). Parsed automatically to extract:

- **phenotype**: Disease status (`CASE`, `CTRL`) or experimental condition
- **participant_id**: Unique donor identifier (e.g., `NEUVM674HUA`)
- **cohort**: Numeric batch identifier (e.g., `5257`), critical for batch effect correction

### BioMatrix

The core data structure (`src/cliquefinder/core/biomatrix.py`). An immutable
container holding:

- `data`: Numerical matrix (features x samples), dtype float64
- `feature_ids`: Row identifiers (pd.Index)
- `sample_ids`: Column identifiers (pd.Index)
- `sample_metadata`: Parsed sample annotations (pd.DataFrame)
- `quality_flags`: Per-cell bitmask tracking data provenance (0=ORIGINAL, 1=OUTLIER_DETECTED, 2=IMPUTED, 3=BOTH)

### Regulator

A gene whose protein product controls expression of other genes. Three classes:

| Class | Source | Count (HGNC) | Example |
|-------|--------|------|---------|
| **TF** (Transcription Factor) | `hgnc_client.tfs` | ~1,672 | TP53, STAT3 |
| **KINASE** | `hgnc_client.kinases` | ~539 | AKT1, LRRK2 |
| **PHOSPHATASE** | `hgnc_client.phosphatases` | ~184 | PTEN, PTPN11 |

Defined in `src/cliquefinder/knowledge/cogex.py:RegulatorClass`.

### Module (INDRAModule)

A regulator and its set of known downstream targets, as reported by INDRA. Each
module records:

- `regulator`: Gene symbol of the upstream regulator
- `targets`: Set of downstream gene symbols
- `edges`: List of `INDRAEdge` objects with statement type and evidence count
- `activated_targets` / `repressed_targets`: Targets split by regulatory direction

Source: INDRA CoGEx knowledge graph via `INDRAModuleExtractor.discover_modules()`.

### Clique

A subset of a module's targets that are **co-expressed** (correlated) in a
specific biological condition. A module may contain 50 targets, but only 12 of
them form a coherent co-expression clique in disease samples. The clique is the
biologically active, condition-specific signal.

Detected via correlation-based community detection in `clique_validator.py`.

### Clique Definition (CliqueDefinition)

The bridge between the knowledge layer and the differential testing layer
(`src/cliquefinder/stats/clique_analysis.py`). Contains:

- `clique_id`: Human-readable identifier (e.g., `TP53_activated`)
- `gene_ids`: Set of feature IDs (mapped from gene symbols)
- `regulator`: Upstream regulator name
- `direction`: `"positive"` (all activation), `"negative"` (all repression), or `"mixed"`

Created by `modules_to_clique_definitions()` which maps gene symbols to feature
IDs and determines direction from INDRA statement types.

### Statement Type

The kind of regulatory relationship reported by INDRA:

| Preset | Statement Types | Biological Meaning |
|--------|----------------|-------------------|
| `activation` | IncreaseAmount, Activation | Regulator increases target expression/activity |
| `repression` | DecreaseAmount, Inhibition | Regulator decreases target expression/activity |
| `regulatory` | All of the above (default) | Any regulatory relationship |
| `phosphorylation` | Phosphorylation | Post-translational modification |

Resolved by `resolve_stmt_types()` in `cogex.py`. Threaded through all INDRA
queries via the `--stmt-types` CLI flag.

### Condition / Stratum

A subgroup of samples defined by metadata columns. In the `analyze` workflow,
samples are stratified by phenotype (and optionally sex, cohort) to produce
condition-specific correlation matrices. Common conditions:

- `CASE_Male`, `CASE_Female`, `CTRL_Male`, `CTRL_Female`
- Or simply `CASE`, `CTRL` when not sex-stratified

### Coherence

The degree to which a set of genes are co-expressed. Quantified as the mean
absolute pairwise Pearson correlation among genes within a clique, computed
within a single condition's samples. Range: [0, 1]. Values > 0.3 typically
indicate biologically meaningful co-regulation.

### Rewiring Score

A summary statistic for how much a regulator's target correlation structure
changes between conditions (e.g., CASE vs CTRL). Computed as:

```
rewiring_score = gained_cliques + lost_cliques
```

where gained/lost cliques are communities that appear in one condition but not
the other (Jaccard similarity < 0.3).

---

## Workflow 1: Knowledge-Guided Coherence Analysis

**Command**: `cliquefinder analyze`
**Orchestration**: `src/cliquefinder/cli/_analyze_core.py:run_stratified_analysis()`
**Output directory**: `--output` (default: `results/analysis`)

### `analysis_parameters.json`

| Layer | Description |
|-------|-------------|
| **Artifact** | JSON file recording all run parameters for reproducibility. |
| **Semantics** | Contains: timestamp, input file paths, all CLI flags, gene universe size, number of regulators tested, filter statistics. |
| **Method** | Direct serialization of `argparse` namespace + runtime metadata. |
| **Provenance** | Self-referential: documents the provenance of all other files in this directory. |

### `regulators_summary.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per regulator that produced at least one clique. The primary results table for reporting. |
| **Semantics** | See column dictionary below. |
| **Method** | Aggregation of per-condition coherence analysis results. Each regulator's INDRA targets are intersected with the expression matrix features, then community detection is run per condition. |
| **Provenance** | Regulator-target edges from INDRA CoGEx, filtered by `--regulator-class` and `--stmt-types`. Expression data from the preprocessed BioMatrix. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `regulator` | str | Gene symbol of upstream regulator (e.g., TP53) |
| `n_indra_targets` | int | Number of targets reported by INDRA for this regulator |
| `n_rna_validated_targets` | int | Subset of INDRA targets present in the expression matrix |
| `max_clique_size` | int | Largest coherent community found across all conditions |
| `max_coherence` | float | Highest mean |r| among all cliques for this regulator |
| `best_condition` | str | Condition where the largest/most coherent clique was found |
| `conditions_with_cliques` | int | Number of conditions producing at least one clique |
| `max_rewiring_score` | float | Maximum rewiring score across all condition comparisons |
| `avg_rewiring_score` | float | Mean rewiring score across comparisons |
| `total_gained_cliques` | int | Cliques present in CASE but not CTRL (disease-gained) |
| `total_lost_cliques` | int | Cliques present in CTRL but not CASE (disease-lost) |

### `cliques.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per (regulator, condition) clique. The core discovery table. |
| **Semantics** | Each row is a coherent co-expression community detected within a specific condition. |
| **Method** | Pairwise Pearson correlations computed among INDRA-validated targets within condition samples. Correlation matrix thresholded to build a signed graph (positive edges: r > threshold, negative edges: r < -threshold). Louvain community detection applied. Communities filtered by minimum size and density. |
| **Provenance** | Gene membership from INDRA CoGEx. Correlations from condition-specific expression data. Community structure from networkx Louvain algorithm (Blondel et al., 2008). |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `regulator` | str | Upstream regulator gene symbol |
| `condition` | str | Biological condition (e.g., `CASE_Male`) |
| `n_samples` | int | Number of samples in this condition |
| `n_indra_targets` | int | Total INDRA targets for this regulator |
| `n_rna_validated_targets` | int | Targets present in expression data |
| `n_coherent_genes` | int | Genes in this clique (community size) |
| `coherence_ratio` | float | `n_coherent_genes / n_rna_validated_targets` |
| `rna_validation_ratio` | float | `n_rna_validated_targets / n_indra_targets` |
| `direction` | str | `"positive"` or `"negative"` (correlation sign of graph) |
| `signed_mean_correlation` | float | Mean pairwise |r| within the clique |
| `signed_min_correlation` | float | Minimum pairwise |r| (weakest link) |
| `signed_max_correlation` | float | Maximum pairwise |r| |
| `n_positive_edges` | int | Edges with r > 0 in the clique subgraph |
| `n_negative_edges` | int | Edges with r < 0 |
| `clique_genes` | str | Semicolon-delimited list of gene symbols in the clique |

### `regulator_rewiring_stats.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per (regulator, comparison) pair. Quantifies disease-associated regulatory rewiring. |
| **Semantics** | Measures how the co-expression structure of a regulator's targets changes between conditions. |
| **Method** | Fisher Z-transformation applied to all gene pairs shared between conditions. Z-test for difference: `z = (Z_a - Z_b) / sqrt(1/(n_a-3) + 1/(n_b-3))`. FDR correction via Benjamini-Hochberg. Effective number of tests estimated via Nyholt (2004) eigenvalue method. |
| **Provenance** | Condition-specific correlation matrices. Sample sizes from metadata stratification. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `regulator` | str | Upstream regulator |
| `comparison` | str | E.g., `CASE_Male_vs_CTRL_Male` |
| `n_case_samples` | int | Samples in condition A |
| `n_ctrl_samples` | int | Samples in condition B |
| `gained_cliques` | int | Communities in A but not B |
| `lost_cliques` | int | Communities in B but not A |
| `case_coherence` | float | Mean coherence of cliques in condition A |
| `ctrl_coherence` | float | Mean coherence of cliques in condition B |
| `rewiring_score` | float | `gained_cliques + lost_cliques` |
| `n_gene_pairs_tested` | int | Total gene pairs evaluated (n*(n-1)/2) |
| `n_significant_pairs` | int | Pairs passing FDR < threshold AND \|delta_r\| >= min_change |
| `fdr_support_ratio` | float | `n_significant_pairs / n_gene_pairs_tested` |
| `fdr_threshold` | float | FDR cutoff used (default 0.05) |
| `correlation_threshold` | float | Minimum \|delta_r\| for biological relevance (default 0.3) |
| `nominal_tests` | int | Total statistical tests performed |
| `effective_tests` | float | M_eff: equivalent independent tests (Nyholt 2004) |
| `effective_test_reduction` | float | `effective_tests / nominal_tests` (typically 0.3-0.6 for co-regulated genes) |

### `clique_genes.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Long-form table: one row per (regulator, condition, gene) membership. |
| **Semantics** | Enumerates every gene's clique membership with RNA validation status. |
| **Method** | Flattened from community detection results. |
| **Provenance** | Gene membership from Louvain communities. RNA validation from expression matrix feature intersection. |

**Columns**: `regulator`, `condition`, `gene`, `rna_validated` (bool)

### `clique_edges.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Edge list with pairwise correlations within each clique. |
| **Semantics** | Every gene-gene correlation that forms the clique's internal structure. |
| **Method** | Pearson correlation coefficients computed on condition-stratified expression data. |
| **Provenance** | Expression values from BioMatrix, restricted to condition samples. |

**Columns**: `regulator`, `condition`, `gene1`, `gene2`, `correlation` (float, [-1, 1])

### `gene_pair_stats.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Per-gene-pair differential correlation statistics. Created only when differential comparisons have significant results. |
| **Semantics** | Full statistical detail for every tested gene pair across two conditions. |
| **Method** | Fisher Z-transformation: `Z = arctanh(r)`, `SE(Z) = 1/sqrt(n-3)`. Test statistic: `z = (Z_a - Z_b) / sqrt(SE_a^2 + SE_b^2)`. P-values from standard normal. FDR via Benjamini-Hochberg (1995). Confidence intervals: `tanh(Z +/- 1.96*SE)`. |
| **Provenance** | Correlations from condition-specific expression data. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `gene1`, `gene2` | str | Gene pair tested |
| `r_case`, `r_ctrl` | float | Pearson r in each condition |
| `delta_r` | float | `r_case - r_ctrl` |
| `z_score` | float | Fisher Z-test statistic |
| `p_value` | float | Two-tailed nominal p-value |
| `q_value` | float | BH-adjusted p-value |
| `is_significant` | bool | `q_value < fdr_threshold AND abs(delta_r) >= min_change` |
| `ci_case_lower`, `ci_case_upper` | float | 95% CI for r_case |
| `ci_ctrl_lower`, `ci_ctrl_upper` | float | 95% CI for r_ctrl |

### `multiple_testing_report.json`

| Layer | Description |
|-------|-------------|
| **Artifact** | Aggregated multiple testing correction statistics. |
| **Semantics** | Reports nominal vs effective test counts to demonstrate that the multiple testing burden is appropriately handled. |
| **Method** | M_eff estimated via eigenvalue decomposition of the gene-gene correlation matrix. Nyholt (2004): `M_eff = sum(lambda) / max(lambda)`. |
| **Provenance** | Correlation matrices from expression data. |

---

## Workflow 2: Differential Abundance Testing

**Command**: `cliquefinder differential`
**Orchestration**: `src/cliquefinder/cli/differential.py`
**Output directory**: `--output`

Three modes produce distinct file sets:

### Mode A: Standard Linear Model (default)

#### `clique_differential.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per clique. Primary differential abundance results. |
| **Semantics** | Tests whether each clique's summarized abundance differs between conditions. |
| **Method** | Per-clique: (1) Tukey median polish summarizes member proteins to one abundance per sample. (2) OLS or mixed model (`log2(abundance) ~ condition`) tests the contrast. (3) Satterthwaite degrees of freedom for mixed models. (4) BH FDR correction across cliques. |
| **Provenance** | Clique definitions from `--cliques` CSV or `--discover-gene-sets` (INDRA). Expression data from BioMatrix. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `clique_id` | str | Identifier from CliqueDefinition |
| `regulator` | str | Upstream regulator (if INDRA-derived) |
| `n_proteins` | int | Proteins in clique definition |
| `n_proteins_found` | int | Proteins present in expression data |
| `summarization_method` | str | `"median_polish"` (Tukey 1977) |
| `coherence` | float | Mean pairwise Spearman rho among member proteins |
| `log2FC` | float | Log2 fold change (condition A vs B) |
| `SE` | float | Standard error of log2FC |
| `tvalue` | float | t-statistic: `log2FC / SE` |
| `df` | float | Degrees of freedom (residual for OLS, Satterthwaite for LMM) |
| `pvalue` | float | Two-tailed p-value from t-distribution |
| `adj_pvalue` | float | BH-adjusted p-value |
| `CI_lower`, `CI_upper` | float | 95% confidence interval for log2FC |
| `contrast` | str | E.g., `CASE-CTRL` |
| `model_type` | str | `"fixed"` (OLS) or `"mixed"` (LMM) |
| `issue` | str | Warnings (e.g., convergence issues) |
| `direction` | str | Clique direction from CliqueDefinition |
| `signed_mean_correlation` | float | Internal coherence |

#### `significant_cliques.csv`

Subset of `clique_differential.csv` where `adj_pvalue < threshold` (default 0.05).

#### `protein_differential.csv` (with `--also-protein-level`)

Per-protein differential results (same columns as clique_differential but at individual protein level).

### Mode B: ROAST Rotation Testing (`--roast`)

#### `roast_clique_results.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per clique. Rotation-based gene set test results. |
| **Semantics** | Tests whether the genes in each clique are enriched for differential expression, preserving inter-gene correlation structure. |
| **Method** | ROAST (Wu et al., 2010): QR decomposition of design matrix, then rotation of residual vectors on a hypersphere. P-values computed as `(b+1)/(B+1)` where b = count of null statistics >= observed (Phipson & Smyth 2010). Multiple test statistics capture different signal types. |
| **Provenance** | Gene set membership from CliqueDefinition. Expression data in linear model framework. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `clique_id` | str | Clique identifier |
| `clique_genes` | str | Semicolon-delimited gene list |
| `n_genes_found` | int | Genes present in expression data |
| `pvalue_mean_up` | float | P(genes upregulated together). Sensitive when majority change in same direction. |
| `pvalue_mean_down` | float | P(genes downregulated together). |
| `pvalue_mean_mixed` | float | P(genes DE in either direction, absolute). `T = sum(a_i * abs(z_i)) / A`. |
| `pvalue_msq_mixed` | float | **Bidirectional regulation test.** `T = sum(abs(a_i) * z_i^2) / A`. Detects regulation regardless of direction. Essential for TFs that both activate and repress targets. |
| `pvalue_floormean_mixed` | float | Noise-dampened: floors small effects to `sqrt(median(chi2_1))` before averaging. |
| `observed_mean` | float | Observed MEAN test statistic |
| `observed_msq` | float | Observed MSQ test statistic |
| `active_proportion_mixed` | float | Fraction of genes with \|moderated t\| > sqrt(2) (AIC-based activity criterion) |
| `n_rotations` | int | Number of rotations (default 9999) |

#### `roast_top_hits.csv`

Subset where any p-value < 0.05. Quick reference for significant results.

#### `roast_bidirectional_candidates.csv`

Cliques where `pvalue_msq_mixed < 0.05` but `pvalue_mean_*` are not significant.
These represent **bidirectional regulation**: the TF's targets change, but not all
in the same direction. Biologically important for TFs with dual activator/repressor roles.

### Mode C: Competitive Permutation Testing (`--permutation-test`)

#### `clique_differential_permutation.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per clique. Competitive enrichment results. |
| **Semantics** | Tests whether clique abundance changes more than expected from random gene sets of equal size drawn from the regulated gene pool. |
| **Method** | Competitive test: (1) Observe t-statistic for each clique via median polish + OLS. (2) For N permutations, sample random gene sets from the regulated gene pool and compute t-statistics. (3) Empirical p-value: `(count(abs(t_null) >= abs(t_obs)) + 1) / (N + 1)`. Optionally with empirical Bayes variance moderation (Smyth 2004). |
| **Provenance** | Null distribution from the union of all TF target pools. GPU-accelerated via MLX (Apple Silicon). |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `clique_id` | str | Clique identifier |
| `observed_log2fc` | float | Observed fold change |
| `observed_tvalue` | float | Observed t-statistic |
| `empirical_pvalue` | float | Two-sided: `P(abs(t_null) >= abs(t_obs))` |
| `empirical_pvalue_directional` | float | One-sided: same direction as observed |
| `percentile_rank` | float | Where observed falls in null (0-100) |
| `is_significant` | bool | `empirical_pvalue < threshold` |
| `null_log2fc_mean`, `null_log2fc_std` | float | Null distribution summary |

#### `null_distribution_summary.csv`

| Column | Type | Definition |
|--------|------|-----------|
| `null_log2FC_5pct` | float | 5th percentile of null fold changes |
| `null_log2FC_95pct` | float | 95th percentile (defines 90% null interval) |
| `null_tvalue_mean`, `null_tvalue_std` | float | Null t-statistic distribution |

### Mode D: Protein-Level (`--mode protein`)

#### `all_proteins.csv` (with `--enrichment-test`)

Per-protein results with a binary `is_target` column indicating whether each
protein is a known target of the queried regulator.

#### `enrichment_results.json` (with `--enrichment-test`)

Competitive enrichment test: are the regulator's targets more differentially
abundant than non-target proteins? Reports enrichment p-value, odds ratio, and
mean log2FC for targets vs non-targets.

---

## Workflow 3: Method Comparison

**Command**: `cliquefinder compare`
**Orchestration**: `src/cliquefinder/cli/compare.py`
**Output directory**: `--output`

### `comparison_results.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Wide-format table with all methods' results for each clique. |
| **Semantics** | Enables direct comparison of p-values, effect sizes, and significance calls across methods. |
| **Method** | Runs OLS, ROAST (all statistics), and optionally permutation test on every clique, then merges into one table. |
| **Provenance** | Same expression data and clique definitions fed to each method independently. |

### `concordance_matrix.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Symmetric matrix of pairwise method agreement. |
| **Semantics** | Quantifies how often methods agree on significance. |
| **Method** | For each method pair: Spearman rho of p-value ranks, Cohen's kappa of binary calls, Jaccard index of significant sets, direction agreement fraction. |
| **Provenance** | Derived from `comparison_results.csv`. |

**Metrics** (per cell):

| Metric | Definition | Interpretation |
|--------|-----------|---------------|
| `spearman_rho` | Rank correlation of p-values | > 0.8 excellent, < 0.5 poor |
| `cohen_kappa` | Chance-corrected binary agreement (Cohen 1960) | > 0.6 substantial, < 0.2 slight |
| `jaccard_index` | \|A intersect B\| / \|A union B\| for significant sets | Set overlap |
| `direction_agreement_frac` | Fraction with same sign of effect | Should be > 0.95 |

### `robust_hits_p{threshold}.csv`

Cliques called significant by **all methods** at the given threshold. These are
the highest-confidence findings.

### `method_specific_{method}.csv`

Cliques significant by one method but not others. Useful for understanding
method-specific sensitivity.

### `disagreements.csv`

Cases where methods disagree on significance. Columns include both methods'
p-values and effect sizes for manual inspection.

### Bootstrap Sub-Workflow (`--bootstrap`)

#### `bootstrap_results.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Per-clique stability metrics across bootstrap iterations. |
| **Semantics** | Quantifies robustness to sampling variability, especially critical for imbalanced designs. |
| **Method** | Balanced bootstrap resampling (Efron & Tibshirani 1993). Both cases and controls resampled WITH replacement (when n_controls >= 50) or controls fixed (Davison & Hinkley 1997, when n_controls < 50). Each iteration runs OLS + ROAST. Selection frequency = fraction of iterations calling the clique significant. |
| **Provenance** | Same data, resampled. |

**Columns**:

| Column | Type | Definition |
|--------|------|-----------|
| `clique_id` | str | Clique identifier |
| `direction` | str | From CliqueDefinition |
| `selection_freq_ols` | float | Fraction significant via OLS across bootstraps |
| `selection_freq_roast` | float | Fraction significant via ROAST |
| `is_stable_ols` | bool | `selection_freq_ols >= 0.80` |
| `is_stable_roast` | bool | `selection_freq_roast >= 0.80` |
| `is_robust` | bool | Direction-aware combined stability (see below) |
| `stability_criterion` | str | `"both_methods"` or `"roast_only"` |
| `median_pvalue_ols` | float | Central tendency of OLS p-values |
| `median_pvalue_roast` | float | Central tendency of ROAST p-values |
| `pvalue_ci_low_ols`, `pvalue_ci_high_ols` | float | 95% CI (2.5th-97.5th percentile) |
| `mean_effect`, `median_effect` | float | Log2FC summary across bootstraps |
| `effect_ci_low`, `effect_ci_high` | float | 95% CI for effect size |
| `method_concordance` | float | Fraction where OLS and ROAST agree (coherent cliques only; None for mixed) |

**Direction-aware robustness**:
- **Coherent cliques** (direction = positive/negative): `is_robust = is_stable_ols AND is_stable_roast`. Both methods must be stable because coherent cliques satisfy both methods' assumptions.
- **Mixed cliques** (direction = mixed): `is_robust = is_stable_roast`. Only ROAST is valid because its MSQ statistic handles bidirectional regulation; OLS assumes additive structure which mixed cliques violate.

### Stratified Comparison (`--stratify-by`)

Creates a subdirectory per stratum with the same file structure, plus:

#### `stratified_summary.json`

Aggregated cross-stratum statistics: which cliques are significant in all
strata, which are stratum-specific.

### Interaction Testing (`--interaction`)

#### `interaction_results.csv`

Tests for Factor1 x Factor2 interaction (e.g., Sex x Disease). The contrast
tested is `(A_L1 - A_L2) - (B_L1 - B_L2)`. Significant results indicate
that the treatment effect differs by the stratification factor.

---

## Workflow 4: De Novo Module Discovery

**Command**: `cliquefinder discover`
**Orchestration**: `src/cliquefinder/cli/discover.py`
**Output directory**: `--output` (default: `results/denovo`)

### `modules.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | One row per discovered co-expression module. |
| **Semantics** | Data-driven (non-knowledge-guided) gene modules found by correlation-based community detection. |
| **Method** | Gene-gene correlation matrix computed per condition. Thresholded to signed graph. Louvain community detection. No INDRA knowledge used. |
| **Provenance** | Expression data only; no external knowledge graph input. |

**Columns**: `condition`, `size`, `mean_correlation`, `genes`

### `conditions_summary.csv`

Per-condition aggregates: `n_modules`, `mean_size`, `max_size`, `mean_correlation`.

### `config.json`

Discovery parameters for reproducibility.

---

## Workflow 5: Preprocessing and Imputation

**Command**: `cliquefinder impute`
**Orchestration**: `src/cliquefinder/cli/impute.py`
**Output directory**: `--output`

### `{output}.data.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Cleaned, imputed expression matrix. Same structure as input (features x samples). |
| **Semantics** | Missing values imputed, outliers handled. Ready for downstream analysis. |
| **Method** | Pipeline: (1) Load raw data. (2) Detect outliers via MAD-Z scoring within groups. (3) Replace outliers with NaN. (4) Impute NaN values (method depends on `--imputation-method`). (5) Optional log2 transformation. |
| **Provenance** | Raw expression data. Quality flags track every modification. |

### `{output}.flags.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Quality flag matrix, same dimensions as data. Each cell is a bitmask. |
| **Semantics** | Complete provenance for every value in the data matrix. |
| **Method** | Bitmask values: `0` = ORIGINAL (untouched), `1` = OUTLIER_DETECTED (flagged by MAD-Z), `2` = IMPUTED (value replaced), `3` = BOTH (outlier that was then imputed). |
| **Provenance** | Generated during the imputation pipeline. |

### `{output}.metadata.csv`

Sample metadata extracted from sample IDs: `phenotype`, `participant_id`, `cohort`.

### `{output}.symbol_mapping.json` (with `--gene-symbols`)

Maps gene symbols to Ensembl IDs (or vice versa). Used for cross-referencing
with INDRA knowledge graph which uses gene symbols.

### `{output}.params.json`

Preprocessing parameters: `is_log_transformed`, outlier detection config
(method, threshold, mode), imputation config. Required for reproducibility
and for feeding into downstream analysis.

### `{output}.report.txt`

Human-readable summary: input dimensions, number of outliers detected,
number of values imputed, distribution statistics before/after.

---

## Workflow 6: Sensitivity Analysis

**Command**: `cliquefinder sensitivity`
**Orchestration**: `src/cliquefinder/cli/sensitivity.py`
**Output directory**: `--output`

### `{output}.json`

Machine-readable results with full per-threshold statistics including feature-
and sample-level outlier rate distributions.

### `{output}.csv`

| Layer | Description |
|-------|-------------|
| **Artifact** | Tabular summary: one row per MAD-Z threshold tested. |
| **Semantics** | Demonstrates result stability across threshold choices. |
| **Method** | For each threshold in [4.0, 4.5, 5.0, 5.5, 6.0] (configurable): apply MAD-Z outlier detection, count flagged values, compute feature/sample-level rates. |
| **Provenance** | Same input matrix; only the threshold parameter varies. |

**Columns**: `threshold`, `n_outliers`, `pct_outliers`, `n_features_affected`,
`pct_features_affected`, `n_samples_affected`, `pct_samples_affected`,
`per_group_stats` (JSON)

### `{output}.txt`

Narrative interpretation suitable for supplementary materials. Includes
assessment of whether results are threshold-dependent.

---

## Statistical Methods Reference

### Tukey Median Polish

**Purpose**: Summarize multiple protein abundances to a single clique-level value per sample.

**Algorithm**: Iteratively subtracts row and column medians until convergence.
Returns `overall_effect + column_effects` as per-sample clique abundances.

**Assumption**: Additive structure (all proteins move in the same direction).
A warning is issued for mixed-direction cliques.

**Reference**: Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

### Ordinary Least Squares (OLS)

**Model**: `log2(abundance) ~ condition`
**Test**: t-test for condition contrast coefficient.
**df**: `n - p` (residual degrees of freedom).

### Linear Mixed Model (LMM)

**Model**: `log2(abundance) ~ condition + (1 | subject)`
**Estimation**: REML via `statsmodels.mixedlm`.
**df**: Satterthwaite approximation: `df = 2 * V^2 / Var(V)` where V = contrast variance.

**References**:
- Satterthwaite, F.E. (1946). *Biometrics Bulletin*, 2(6), 110-114.
- Choi, M. et al. (2014). MSstats. *Bioinformatics*, 30(17), 2524-2526.

### Empirical Bayes Variance Moderation

**Purpose**: Stabilize variance estimates by borrowing strength across genes.

**Algorithm**: Fit scaled F-distribution to sample variances (method of moments).
Posterior variance: `s2_post = (d0*s0^2 + df*s^2) / (d0 + df)`.
Moderated t follows t-distribution with `d0 + df` degrees of freedom.

**Reference**: Smyth, G.K. (2004). *Statistical Applications in Genetics and Molecular Biology*, 3(1), Article 3.

### ROAST Rotation Testing

**Purpose**: Self-contained gene set test preserving inter-gene correlation.

**Procedure**: QR decomposition of design matrix. Rotate residual vectors on
hypersphere to generate null distribution. P-value: `(b+1)/(B+1)`.

**Test statistics**:
- **MEAN**: Directional co-regulation (`T = sum(a_i * z_i) / A`)
- **MSQ**: Bidirectional regulation (`T = sum(|a_i| * z_i^2) / A`)
- **FLOORMEAN**: Noise-dampened (`floors |z| to sqrt(0.67)`)
- **MEAN50**: Robust to outliers (top 50% of |z|)

**Reference**: Wu, D. et al. (2010). ROAST. *Bioinformatics*, 26(17), 2176-2182.

### Permutation-Based FDR

**Purpose**: Control FDR under arbitrary dependence (no PRDS assumption).

**Algorithm**: For each threshold t: R(t) = observed rejections, V(t) = null
rejections scaled by m/n_null. FDR(t) = pi0 * V(t) / R(t). Pi0 estimated
via Storey's method.

**Reference**: Storey, J.D. & Tibshirani, R. (2003). *PNAS*, 100(16), 9440-9445.

### Benjamini-Hochberg FDR

**Purpose**: Control expected proportion of false discoveries among rejections.

**Assumption**: Independence or positive regression dependence (PRDS).

**Reference**: Benjamini, Y. & Hochberg, Y. (1995). *JRSS-B*, 57(1), 289-300.

### Fisher Z-Transformation

**Purpose**: Compare correlations between independent samples.

**Transform**: `Z = arctanh(r)`, `SE = 1/sqrt(n-3)`.
**Test**: `z = (Z1 - Z2) / sqrt(SE1^2 + SE2^2)`.

**Reference**: Fisher, R.A. (1921). *Biometrika*, 10(4), 507-521.

### Effective Number of Tests (M_eff)

**Purpose**: Adjust multiple testing burden for correlated tests.

**Nyholt method**: `M_eff = sum(lambda) / max(lambda)` where lambda are
eigenvalues of correlation matrix. Conservative for strong correlation.

**Reference**: Nyholt, D.R. (2004). *AJHG*, 74(4), 765-769.

### Bootstrap Resampling

**Purpose**: Quantify robustness to sampling variability.

**True bootstrap** (n_controls >= 50): Sample WITH replacement from both groups.
**Fixed controls** (n_controls < 50): Resample cases only, fix small control group.

**Reference**: Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

---

## Knowledge Layer Reference

### INDRA (Integrated Network and Dynamical Reasoning Assembler)

A machine-reading system that extracts biological relationships from published
literature (PubMed/PMC full text). Statements are structured assertions like
"TP53 increases the amount of BAX" with evidence counts.

### CoGEx (Causal Ontology Graph Extension)

INDRA's knowledge graph, queryable via the `indra_cogex` API. Contains
regulator-target edges with statement types, evidence counts, and source
databases.

### Statement Types in INDRA

| Type | Direction | Example |
|------|-----------|---------|
| IncreaseAmount | Activation | "TP53 increases BAX expression" |
| Activation | Activation | "EGF activates MAPK signaling" |
| DecreaseAmount | Repression | "MYC decreases p21 levels" |
| Inhibition | Repression | "PTEN inhibits AKT" |
| Phosphorylation | PTM | "CDK2 phosphorylates RB1" |

### Evidence Count

Number of independent literature mentions supporting a statement. Higher
evidence counts indicate more reliable relationships. Used as a confidence
filter (default: >= 1).

### Cross-Modal Mapping

`src/cliquefinder/knowledge/cross_modal_mapper.py` bridges identifier systems:
UniProt accessions (proteomics) to HGNC gene symbols (INDRA) and back.
Required because INDRA uses gene symbols while proteomics data uses UniProt IDs.

---

## Column Dictionary

Quick-reference alphabetical index of all output columns.

| Column | Files | Type | Definition |
|--------|-------|------|-----------|
| `active_proportion_mixed` | roast_clique_results | float | Fraction of genes with \|t\| > sqrt(2) |
| `adj_pvalue` | clique_differential, protein_differential | float | BH-adjusted p-value |
| `avg_rewiring_score` | regulators_summary | float | Mean rewiring across comparisons |
| `best_condition` | regulators_summary | str | Condition with highest coherence |
| `bootstrap_stability` | cliques (analyze) | float | Fraction of bootstrap iterations recovering community (Jaccard > 0.5) |
| `case_coherence` | regulator_rewiring_stats | float | Mean coherence in case condition |
| `CI_lower`, `CI_upper` | clique_differential | float | 95% CI for log2FC |
| `clique_genes` | cliques, roast_clique_results | str | Semicolon-delimited gene list |
| `clique_id` | all differential outputs | str | Unique clique identifier |
| `coherence` | clique_differential | float | Mean pairwise Spearman rho |
| `coherence_ratio` | cliques | float | Clique size / validated targets |
| `comparison` | regulator_rewiring_stats | str | Condition pair compared |
| `condition` | cliques, clique_genes | str | Biological condition |
| `correlation` | clique_edges | float | Pearson r between gene pair |
| `ctrl_coherence` | regulator_rewiring_stats | float | Mean coherence in control |
| `delta_r` | gene_pair_stats | float | r_case - r_ctrl |
| `density` | cliques (internal) | float | Edge density of clique subgraph |
| `df` | clique_differential | float | Degrees of freedom |
| `direction` | cliques, CliqueDefinition | str | positive/negative/mixed |
| `effective_test_reduction` | regulator_rewiring_stats | float | M_eff / M_nominal |
| `effective_tests` | regulator_rewiring_stats | float | Nyholt M_eff |
| `effect_ci_low`, `effect_ci_high` | bootstrap_results | float | 95% CI for effect size |
| `empirical_pvalue` | permutation results | float | Two-sided empirical p |
| `gained_cliques` | regulator_rewiring_stats | int | Disease-gained communities |
| `gene` | clique_genes | str | Gene symbol |
| `is_robust` | bootstrap_results | bool | Direction-aware combined stability |
| `is_significant` | gene_pair_stats, permutation | bool | Passes significance threshold |
| `log2FC` | clique_differential, permutation | float | Log2 fold change |
| `lost_cliques` | regulator_rewiring_stats | int | Disease-lost communities |
| `max_clique_size` | regulators_summary | int | Largest community found |
| `max_coherence` | regulators_summary | float | Highest mean \|r\| |
| `max_rewiring_score` | regulators_summary | float | Maximum rewiring |
| `method_concordance` | bootstrap_results | float | OLS-ROAST agreement rate |
| `model_type` | clique_differential | str | fixed or mixed |
| `modularity` | analyze (internal) | float | Newman modularity of community partition |
| `n_coherent_genes` | cliques | int | Community size |
| `n_indra_targets` | regulators_summary, cliques | int | INDRA-reported targets |
| `n_proteins_found` | clique_differential | int | Proteins in data |
| `n_rna_validated_targets` | regulators_summary, cliques | int | Targets in expression data |
| `n_rotations` | roast_clique_results | int | Rotations performed |
| `n_samples` | cliques | int | Samples in condition |
| `observed_msq` | roast_clique_results | float | Observed MSQ statistic |
| `p_value` | gene_pair_stats | float | Nominal p-value |
| `pct_outliers` | sensitivity | float | Percentage flagged |
| `percentile_rank` | permutation results | float | Position in null (0-100) |
| `pvalue` | clique_differential | float | Nominal p-value |
| `pvalue_mean_up` | roast_clique_results | float | Genes upregulated together |
| `pvalue_msq_mixed` | roast_clique_results | float | Bidirectional regulation |
| `q_value` | gene_pair_stats | float | BH-adjusted p-value |
| `r_case`, `r_ctrl` | gene_pair_stats | float | Condition-specific correlations |
| `regulator` | all analyze/differential outputs | str | Upstream regulator symbol |
| `rewiring_score` | regulator_rewiring_stats | float | gained + lost cliques |
| `rna_validated` | clique_genes | bool | Gene in expression matrix |
| `SE` | clique_differential | float | Standard error |
| `selection_freq_ols` | bootstrap_results | float | OLS stability (0-1) |
| `selection_freq_roast` | bootstrap_results | float | ROAST stability (0-1) |
| `signed_mean_correlation` | cliques | float | Mean \|r\| within clique |
| `stability_criterion` | bootstrap_results | str | both_methods or roast_only |
| `summarization_method` | clique_differential | str | median_polish |
| `threshold` | sensitivity | float | MAD-Z threshold tested |
| `tvalue` | clique_differential | float | t-statistic |
| `z_score` | gene_pair_stats | float | Fisher Z-test statistic |
