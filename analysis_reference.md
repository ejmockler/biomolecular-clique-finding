# Analysis Reference

This document describes the scientific question, the three analytical pipelines that address it, every output file they produce, and the statistical and biological concepts underlying the analysis. It is written for engineers and collaborators who need to understand what the framework does, what its outputs mean, and how to interpret them.

---

## I. The Scientific Question

ALS (amyotrophic lateral sclerosis) is a neurodegenerative disease with no cure. We have mass-spectrometry-based quantitative proteomics data from the Answer ALS consortium: thousands of proteins measured across hundreds of patient and control samples, including both C9orf72-linked (a specific genetic mutation) and sporadic (no known genetic cause) disease subtypes.

The question is: **which upstream regulatory genes have downstream targets that are collectively disrupted in ALS?** Not just individual proteins changing --- that is standard differential expression --- but entire regulatory circuits shifting together in a coordinated way that implicates a specific upstream cause.

We operationalize this through the INDRA knowledge base (Integrated Network and Dynamical Reasoning Assembler), a causal graph of biological relationships assembled from text mining, curated databases (Reactome, TRRUST, RegNetwork, PhosphoSitePlus), and pathway resources. INDRA tells us which genes regulate which other genes, through what mechanisms (activation, repression, phosphorylation), and with how much supporting evidence. The graph lives in a Neo4j database called CoGEx (Causal Ontology Graph Extension), queried via Cypher.

A **regulator** in this context is any gene whose protein product influences expression or activity of other genes --- transcription factors, kinases, phosphatases, E3 ligases, receptor kinases. Not limited to transcription factors. A **regulatory module** is a regulator together with all its known downstream targets in INDRA.

Three pipelines address the question from different angles:
1. **Discovery** (`analyze`): which regulators have targets that form correlated cliques, and where has that co-regulation changed between disease and control?
2. **Differential testing** (`differential`): are specific gene sets (cliques or network targets) statistically more differentially expressed than expected?
3. **Validation** (`validate-baselines`): is an observed network enrichment signal robust to confounders, permutation nulls, and alternative explanations?

They can run independently but form a logical progression: discover, test, validate.

---

## II. Discovery Pipeline (`cliquefinder analyze`)

### What It Does

Given proteomics data and access to INDRA, this pipeline asks: for each candidate regulator, do its downstream targets show coordinated co-expression in disease samples? And does that co-regulation differ between disease and control?

It works in two modes:
- **Hand-picked** (`--regulators TP53 MYC`): tests specific regulators you already suspect.
- **Discovery** (`--discover`): reverse-queries INDRA to find all regulators whose targets overlap the measured gene universe. Hypothesis-generating.

### Workflow

The pipeline loads the expression matrix, maps feature IDs to canonical HGNC gene symbols (via MyGene.info; Wu et al., 2013), optionally filters regulators by RNA-seq expression, and connects to INDRA CoGEx. For each regulator, in each condition stratum (e.g., ALS male, ALS female, control male, control female), it computes pairwise correlations among the regulator's INDRA targets and finds the **maximum clique** --- the largest subset where every gene is correlated with every other gene above a threshold (default 0.7). This clique is the "coherent module," the subset of known targets that actually behave as co-regulated in the data.

For matched case/control pairs within each sex stratum, the pipeline performs differential correlation analysis: Fisher's Z-test on each gene pair's correlation change, with Benjamini-Hochberg FDR correction and effective-tests adjustment (M_eff) to account for correlated tests. This identifies "gained" cliques (co-regulated in disease but not control) and "lost" cliques (co-regulated in control but disrupted in disease), quantified as a **rewiring score** (absolute difference in coherence between conditions).

### Output Files

All files are written to the directory specified by `--output`.

**analysis_parameters.json** --- Reproducibility record. Contains every parameter that governs the run: which regulators were queried, discovery mode settings, INDRA parameters (min_evidence, statement types), correlation threshold, clique-finding algorithm, and runtime configuration. This file exists so any result can be reproduced exactly.

**regulators_summary.csv** --- The top-level results table. One row per regulator that produced at least one clique. This is typically the first file a collaborator opens to answer "which regulators look interesting?"

| Column | Meaning |
|--------|---------|
| regulator | Gene symbol of the upstream regulator |
| n_indra_targets | Number of downstream targets in INDRA for this regulator |
| n_rna_validated_targets | Subset confirmed expressed in RNA-seq (annotation, not filter) |
| max_clique_size | Largest correlated clique found across all conditions |
| max_coherence | Highest coherence ratio (clique size / INDRA target count) across conditions |
| best_condition | Condition stratum where max_coherence was observed |
| conditions_with_cliques | Number of strata where at least one clique was found |
| max_rewiring_score | Largest |case_coherence - ctrl_coherence| across comparisons |
| avg_rewiring_score | Mean rewiring score across comparisons |
| total_gained_cliques | Cliques present in disease but absent in control |
| total_lost_cliques | Cliques present in control but absent in disease |

**cliques.csv** --- The main analytical output. One row per (regulator, condition) pair. Each row describes the coherent module discovered in that stratum --- which genes form the clique, how tightly correlated they are, and in what direction.

| Column | Meaning |
|--------|---------|
| regulator | Upstream regulator |
| condition | Stratum (e.g., CASE_Male, CTRL_Female) |
| n_samples | Sample count in this stratum |
| n_indra_targets | INDRA targets measurable in the data |
| n_rna_validated_targets | Subset confirmed expressed in RNA-seq |
| n_coherent_genes | Number of genes in the maximum clique |
| coherence_ratio | n_coherent_genes / n_indra_targets --- fraction of known targets that co-regulate |
| rna_validation_ratio | Fraction of clique genes confirmed in RNA-seq |
| direction | POSITIVE (all positively correlated), NEGATIVE, or MIXED |
| signed_mean_correlation | Mean pairwise correlation with sign preserved |
| signed_min_correlation | Minimum pairwise correlation in the clique |
| signed_max_correlation | Maximum pairwise correlation in the clique |
| n_positive_edges | Edges with r > 0 |
| n_negative_edges | Edges with r < 0 |
| clique_genes | Comma-separated list of gene symbols in the clique |

A coherence_ratio of 0.5 means half the regulator's known targets form a correlated clique. Values above 0.3 are typically noteworthy. Direction tells you whether the targets move together (POSITIVE) or include opposed regulation (MIXED), which is biologically meaningful for regulators that both activate and repress.

**regulator_rewiring_stats.csv** --- Differential results between case and control. One row per (regulator, comparison) pair. Answers "how much did this regulator's co-regulation change in disease?"

| Column | Meaning |
|--------|---------|
| regulator | Upstream regulator |
| comparison | Which case/control contrast (e.g., CASE_Male_vs_CTRL_Male) |
| n_case_samples, n_ctrl_samples | Sample counts |
| gained_cliques | Cliques present in case but not control |
| lost_cliques | Cliques present in control but not case |
| case_coherence, ctrl_coherence | Coherence ratios in each condition |
| rewiring_score | |case_coherence - ctrl_coherence| |
| n_gene_pairs_tested | Number of target gene pairs tested for differential correlation |
| n_significant_pairs | Pairs passing FDR threshold |
| fdr_support_ratio | n_significant_pairs / n_gene_pairs_tested |
| fdr_threshold | Alpha used for BH FDR |
| correlation_threshold | Threshold for clique edge inclusion |
| nominal_tests | Total pairwise tests before M_eff correction |
| effective_tests | Effective independent tests (M_eff) |
| effective_test_reduction | nominal_tests / effective_tests --- the multiplicity discount |

**clique_genes.csv** --- Long-form gene membership. One row per (regulator, condition, gene). Enables gene-centric queries: "which cliques does TP53 participate in?"

| Column | Meaning |
|--------|---------|
| regulator | Upstream regulator |
| condition | Stratum |
| gene | Gene symbol |
| rna_validated | Whether this gene was confirmed expressed in RNA-seq |

**clique_edges.csv** --- Per-edge correlation values within each clique. Enables reconstruction of the co-expression subgraph for visualization.

| Column | Meaning |
|--------|---------|
| regulator | Upstream regulator |
| condition | Stratum |
| gene1, gene2 | Gene pair |
| correlation | Pairwise correlation value |

**gene_pair_stats.csv** --- FDR-corrected differential correlation statistics for individual gene pairs. The evidence base for rewiring claims. One row per (regulator, comparison, gene1, gene2) where the pair showed significant correlation change.

| Column | Meaning |
|--------|---------|
| regulator | Upstream regulator |
| comparison | Case/control contrast |
| gene1, gene2 | Gene pair |
| r_case | Correlation in case samples |
| r_ctrl | Correlation in control samples |
| delta_r | r_case - r_ctrl |
| z_score | Fisher Z-test statistic for the difference |
| p_value | Raw p-value |
| q_value | BH FDR-adjusted p-value |
| is_significant | Whether q_value < threshold |
| ci_case_lower, ci_case_upper | 95% CI for case correlation (via Fisher Z) |
| ci_ctrl_lower, ci_ctrl_upper | 95% CI for control correlation |

**multiple_testing_report.json** --- Transparency about multiple testing burden. Reports the number of regulators analyzed, total nominal tests, effective tests (M_eff), and the reduction factor. M_eff accounts for the fact that correlated gene pairs do not constitute fully independent tests, using eigenvalue decomposition of the correlation matrix.

---

## III. Differential Testing Pipeline (`cliquefinder differential`)

### What It Does

Tests whether specific gene sets are differentially abundant between conditions, with rigorous statistical controls. Three distinct analysis paths address different questions.

### Path 1: Standard Differential (BH FDR)

Summarizes proteins within each clique to a single abundance per sample using Tukey's Median Polish (Tukey, 1977; Choi et al., 2014) --- a robust two-way decomposition that extracts row and column effects through iterative median subtraction. Fits a linear model (optionally a mixed model with subject random effects) for each clique, then applies Benjamini-Hochberg FDR correction across all tested cliques.

**When to use:** Quick screening of many cliques. Conservative; treats each clique independently.

**protein_differential.csv** --- Protein-level results used as input to clique summarization.

| Column | Meaning |
|--------|---------|
| contrast | Which comparison was tested |
| feature_id | Protein identifier |
| log2FC | Log2 fold change between conditions |
| SE | Standard error of the fold change estimate |
| tvalue | t-statistic |
| df | Degrees of freedom |
| pvalue | Raw p-value |
| CI_lower, CI_upper | 95% confidence interval for log2FC |
| model_type | LMM or fixed-effects |
| n_obs | Number of observations |
| residual_var | Residual variance from the model |
| subject_var | Subject-level variance (LMM only) |
| converged | Whether the model converged |
| issue | Any fitting warnings |
| adj_pvalue | BH FDR-adjusted p-value |
| significant | Whether adj_pvalue < threshold |

**significant_cliques.csv** --- Cliques passing the FDR threshold. Same schema as clique_differential_permutation.csv (below).

### Path 2: ROAST Rotation (`--roast`)

Wu et al. (2010) rotation-based gene set testing. The gold standard for self-contained gene set tests because it preserves the inter-gene correlation structure through random rotations in the residual space, rather than breaking it by permuting genes independently. See Section V for full methodology.

**When to use:** Validating specific cliques with proper correlation handling. Detects both unidirectional regulation (all targets move the same direction) and bidirectional regulation (some targets up, some down, but all perturbed). No FDR correction needed because each rotation test is self-calibrated.

**roast_clique_results.csv** --- All cliques tested. The column structure encodes p-values for every combination of set statistic and alternative hypothesis:

| Column | Meaning |
|--------|---------|
| feature_set_id | Clique identifier |
| clique_genes | Comma-separated member genes |
| n_genes | Number of genes in the set |
| n_genes_found | Number found in the expression data |
| n_rotations | Number of random rotations performed |
| contrast | Which comparison was tested |
| pvalue_{stat}_{alt} | P-value for set statistic {stat} under alternative {alt} |

The {stat} values are: mean, floormean, mean50, msq, mixed. The {alt} values are: up (targets upregulated), down (targets downregulated), mixed (two-sided). So pvalue_msq_mixed tests whether the targets are collectively perturbed in either direction, while pvalue_mean_down tests whether they are coherently downregulated. Additional columns record the observed set statistic values and active proportions.

**roast_top_hits.csv** --- Subset of roast_clique_results.csv where pvalue_msq_mixed < 0.05. These are cliques with significant collective perturbation regardless of direction.

**roast_bidirectional_candidates.csv** --- Subset where MSQ is significant (pvalue_msq_mixed < 0.05) but MEAN is not (pvalue_mean_up >= 0.05 and pvalue_mean_down >= 0.05). These are candidates for bidirectional regulation: the clique members are collectively perturbed, but some increase while others decrease, canceling the directional signal. Biologically, this occurs when a regulator both activates and represses different targets.

**negative_control_sets.json** --- When requested (`--negative-control-sets N`), ROAST p-values from N random gene sets of the same size as each tested clique. Calibrates the expected false positive rate: if random gene sets are just as "significant" as the real cliques, the ROAST results are untrustworthy.

### Path 3: Competitive Permutation (`--permutation-test`)

GPU-accelerated batched OLS with label permutation. For each clique, precomputes (X'X)^-1 once and applies it across all permutations simultaneously, computing t-statistics in float32 (MLX on Apple Silicon) with RSS in float64 to prevent catastrophic cancellation. Empirical Bayes moderation (Smyth, 2004) stabilizes variance estimates.

**When to use:** When you want a competitive null --- "is this gene set more differentially expressed than a random set of the same size?" --- rather than a self-contained null.

**clique_differential_permutation.csv** --- Per-clique permutation results.

| Column | Meaning |
|--------|---------|
| clique_id | Clique identifier |
| clique_genes | Member genes |
| n_proteins | Number of proteins in the clique |
| log2FC | Observed log2 fold change |
| perm_pvalue | Empirical permutation p-value |
| pvalue | Parametric p-value from the model |
| tvalue | Observed t-statistic |
| null_log2FC_mean | Mean of the null log2FC distribution |
| null_log2FC_std | Standard deviation of the null |
| null_tvalue_mean | Mean of the null t-statistic distribution |
| empirical_pvalue_directional | One-sided permutation p-value |
| n_permutations | Number of permutations performed |
| percentile_rank | Where the observed statistic falls in the null |
| is_significant | Whether perm_pvalue < threshold |

**null_distribution_summary.csv** --- Summary statistics (mean, std, 5th/95th percentiles) of the null distributions from permutation, per clique. For diagnostic inspection: a well-calibrated null should be centered near zero with reasonable spread.

| Column | Meaning |
|--------|---------|
| clique_id | Clique identifier |
| null_log2FC_mean, null_log2FC_std | Null distribution moments for fold change |
| null_log2FC_5pct, null_log2FC_95pct | Null distribution quantiles |
| null_tvalue_mean, null_tvalue_std | Null distribution moments for t-statistic |
| null_tvalue_5pct, null_tvalue_95pct | Null distribution quantiles |
| n_permutations | Number of permutations |

### Enrichment Analysis Path

When run with `--enrichment-test --network-query GENE`, the differential pipeline tests whether a regulator's INDRA targets have systematically larger effect sizes than the proteome background.

**all_proteins.csv** --- Genome-wide protein-level differential results. Every measured protein, not just network targets. The background against which enrichment is measured.

| Column | Meaning |
|--------|---------|
| gene_symbol | HGNC symbol |
| feature_id | Original protein identifier |
| log2fc | Log2 fold change |
| t_statistic | Moderated t-statistic |
| p_value | Raw p-value |
| df | Degrees of freedom |
| sigma2_post | Posterior variance (EB-moderated) |
| sigma2 | Raw sample variance |
| n_samples | Number of observations |
| is_target | Whether this protein is in the INDRA target set |

**network_targets.csv** --- Same schema, filtered to is_target = True. Enables inspection of individual target gene behavior: which specific targets are driving the enrichment?

**enrichment_results.json** --- The enrichment test result. See the validation pipeline's covariate_enrichment.json below for the full field schema, which is identical.

**analysis_parameters.json** --- Written by every analysis path. Records method, contrast, covariates, number of rotations or permutations, FDR method, and all configuration.

---

## IV. Validation Pipeline (`cliquefinder validate-baselines`)

### What It Does

Stress-tests a network enrichment signal through a hierarchy of increasingly stringent challenges. Each phase attempts to explain away the signal through a specific alternative hypothesis. The pipeline is designed to be skeptical: it tries to break the finding, and a "validated" verdict means the signal survived every attempt.

### Mandatory Gates and Supplementary Tests

Two phases are **mandatory gates** that determine the verdict:
- Phase 1 (covariate-adjusted enrichment) and Phase 3 (label permutation null) must both pass for "validated."
- Both must fail for "refuted."
- Anything else is "inconclusive."

The remaining phases are supplementary: they strengthen or weaken confidence but do not override the gates.

### Phase 1: Covariate-Adjusted Enrichment

**What it tests:** Are the regulatory module's targets more differentially expressed than the proteome background, after adjusting for known confounders (sex, cohort)?

**What it rules out:** The enrichment being driven by confounders rather than disease biology. If sex imbalance between groups drives differential expression, adjusting for sex in the linear model will absorb that effect and the enrichment will vanish.

The pipeline fits an Empirical Bayes-moderated linear model (Smyth, 2004) with covariates in the design matrix, producing a moderated t-statistic for every measured protein. It then asks whether the INDRA target set has a higher mean |t| than the background, using a competitive permutation test with variance inflation factor correction (Wu & Smyth, 2012) to account for inter-gene correlation within the target set.

**covariate_enrichment.json** --- The primary enrichment result.

| Field | Meaning |
|-------|---------|
| observed_mean_abs_t | Mean |t-statistic| for the target gene set |
| null_mean | Mean of the null distribution (competitive permutation) |
| null_std | Standard deviation of the null |
| z_score | (observed - null_mean) / null_std --- the competitive z-score |
| empirical_pvalue | Fraction of null permutations with z >= observed |
| n_targets | Number of genes in the target set |
| n_background | Number of genes in the background |
| pct_down | Percentage of target genes with negative t-statistics |
| direction_pvalue | Binomial test of whether pct_down deviates from 50% |
| mannwhitney_pvalue | Non-parametric rank-sum test (targets vs background) |
| variance_inflation_factor | VIF = 1 + (k-1) * rho_bar, inflates SE for correlated genes |
| mean_pairwise_correlation | Average inter-gene correlation (rho_bar) within the target set |

The z_score is the headline number. A z of 2.0 means the target set's effect sizes are 2 standard deviations above what random gene sets produce. The VIF correction is critical: without it, a set of correlated genes moving together would appear "enriched" simply because correlation inflates their collective signal. The direction_pvalue tells you whether the targets are coherently moving in one direction (small p) or mixed up and down (p near 1).

**protein_differential_results.csv** --- Full protein-level results from the covariate-adjusted model. One row per measured protein. This file serves double duty: it is a scientific output (every protein's differential statistics) and a checkpoint artifact (downstream phases need the t-statistics).

| Column | Meaning |
|--------|---------|
| feature_id | Protein identifier |
| log2fc | Log2 fold change |
| t_statistic | EB-moderated t-statistic |
| p_value | Raw p-value |
| df | Degrees of freedom (d0 + df_residual) |
| sigma2_post | Posterior variance (EB-moderated) |
| sigma2 | Raw sample variance |
| n_samples | Number of observations |
| is_target | Whether this protein is in the target set |
| gene_symbol | HGNC symbol (for target genes only) |

### Phase 2: Multi-Contrast Specificity

**What it tests:** Is the enrichment specific to the primary contrast (e.g., C9orf72 vs sporadic) or shared across multiple contrasts (C9orf72 vs control, sporadic vs control)?

**What it rules out:** Nothing --- this is characterization, not gatekeeping. A "shared" signal is valid biology (the regulatory module may be dysregulated in all ALS subtypes). A "specific" signal is more mechanistically informative. The interaction test quantifies the contrast difference with a z-test corrected for the correlation between overlapping null distributions.

**specificity.json**

| Field | Meaning |
|-------|---------|
| primary_contrast | The contrast with the strongest signal |
| specificity_ratio | Ratio of primary to secondary z-score |
| specificity_label | "specific", "shared", or "inconclusive" |
| summary | Narrative interpretation |
| contrasts.{name}.z_score | Competitive z-score for each contrast |
| contrasts.{name}.empirical_pvalue | Enrichment p-value for each contrast |
| contrasts.{name}.pct_down | Direction bias for each contrast |
| contrasts.{name}.direction_pvalue | Binomial direction test for each contrast |
| null_correlation | Correlation between null distributions of different contrasts |
| interaction_test.z_difference | Difference in z-scores between primary and secondary contrast |
| interaction_test.interaction_pvalue | P-value for whether the contrast difference is significant |
| interaction_test.z_difference_ci | 95% CI for the z-score difference |

### Phase 3: Label Permutation Null

**What it tests:** Does the enrichment depend on the true case/control labeling?

**What it rules out:** The signal being an artifact of data structure (batch effects, sample processing order, or any systematic feature that correlates with condition but is not condition itself). This is the most fundamental test of whether the signal is real. If random label shuffles produce equally strong enrichment, the "signal" has nothing to do with disease.

The pipeline shuffles condition labels and re-runs the full differential + enrichment pipeline for each permutation. Two modes: **stratified** (shuffles within strata, e.g., within each sex, preserving covariate balance) and **free** (unrestricted shuffling). Stratified is the primary gate; free is a robustness check. If stratification freezes more than 50% of samples (degenerate strata), the result is flagged.

**label_permutation.json** --- Contains both stratified and free results.

| Field | Meaning |
|-------|---------|
| {mode}.observed_z | The real competitive z-score |
| {mode}.permutation_pvalue | Fraction of null z-scores >= observed |
| {mode}.n_permutations | Number of label permutations |
| {mode}.null_mean | Mean of the null z-score distribution |
| {mode}.null_std | Standard deviation of the null |
| {mode}.frozen_fraction | Fraction of samples that could not be permuted within their stratum |
| {mode}.null_z_quantiles | 5th, 25th, 50th, 75th, 95th percentiles of the null |

**label_permutation_distributions.csv** --- The full null distribution for visualization. One row per permutation per mode, plus the observed value.

| Column | Meaning |
|--------|---------|
| permutation_id | "observed" or permutation number |
| mode | "stratified", "free", or "observed" |
| competitive_z | The competitive z-score for this permutation |

This CSV enables plotting the null distribution as a histogram with the observed z-score marked, the standard way to visualize permutation test results.

### Phase 4: Matched Reanalysis

**What it tests:** Does the enrichment survive exact covariate matching?

**What it rules out:** Residual confounding from covariate imbalance. Phase 1 adjusts for covariates statistically (in the model); Phase 4 adjusts by physical sample selection (propensity-score matching). If the signal vanishes with perfectly balanced groups, Phase 1's adjustment was insufficient.

Phase 4 typically has lower power because matching discards samples. Failure here often reflects power loss rather than signal absence.

**matched_enrichment.json** --- Same schema as covariate_enrichment.json, with additional fields in the validation report: n_original (pre-matching sample count), n_matched (post-matching count), and match_vars (which variables were matched on, e.g., ["SEX"]).

### Phase 5a: Negative Control Gene Sets

**What it tests:** Is the target gene set more enriched than random gene sets of the same size?

**What it rules out:** The enrichment being a statistical artifact of set size, or the ROAST test being miscalibrated. If 200 random gene sets produce p-values as small as the target set's, the target's "significance" is meaningless --- any gene set of that size would look significant.

**negative_controls.json**

| Field | Meaning |
|-------|---------|
| target_set_id | Identifier for the target gene set |
| target_set_size | Number of genes in the target set |
| target_pvalue | ROAST p-value for the target set |
| fpr | False positive rate: fraction of control sets with p < alpha |
| target_percentile | Where the target ranks among controls (0 = most significant) |
| median_control_pvalue | Median p-value of random gene sets |
| mean_control_pvalue | Mean p-value of random gene sets |
| n_control_sets | Number of random sets tested |
| n_significant_controls | Number of control sets with p < alpha |
| control_pvalue_quantiles | Distribution of control p-values (q05--q95) |
| competitive_z.target_z | Competitive z-score for the target set |
| competitive_z.target_inter_gene_correlation | rho_bar for the target set (diagnostic) |
| competitive_z.fpr | FPR at the competitive z-score level |
| competitive_z.percentile | Percentile rank of the target's z-score |
| competitive_z.control_z_quantiles | Distribution of control z-scores |

A target_percentile of 9 means the target set is more significant than 91% of random gene sets its size. An FPR of 0.49 means 49% of random sets are nominally significant at alpha = 0.05, which is expected inflation when genes are correlated --- this is why the percentile rank matters more than the raw FPR.

**negative_control_distributions.csv** --- Per-control-set results for plotting.

| Column | Meaning |
|--------|---------|
| set_id | Target set ID or "control_N" |
| type | "target" or "control" |
| roast_pvalue | ROAST p-value |
| competitive_z | Competitive z-score |

### Phase 5b: Graph Permutation Null

**What it tests:** Is the enrichment specific to this regulator's targets, or would any comparably-structured gene set from the INDRA graph show enrichment?

**What it rules out:** The signal being an artifact of network topology. Network neighborhoods share properties (pathway membership, expression level, genomic location). If any regulator's INDRA targets show enrichment, the result reflects graph structure, not the specific biology of the query gene.

On each permutation, a random eligible regulator is sampled (one with at least 2 resolvable targets among measured genes), its targets are tested via ROAST, and the p-value enters the null distribution. The target set's p-value is compared as a percentile rank.

**graph_permutation.json**

| Field | Meaning |
|-------|---------|
| target_set_id | Identifier for the target gene set |
| target_set_size | Number of genes |
| target_pvalue | ROAST p-value for the real targets |
| fpr | Fraction of random regulators with p < alpha |
| target_percentile | Percentile rank (0 = most significant regulator) |
| median_control_pvalue | Median p-value across random regulators |
| n_permutations | Number of random regulators tested |
| n_eligible_regulators | Total eligible regulators in the INDRA subgraph |
| median_control_set_size | Median target set size of random regulators |
| graph_stats.n_nodes | Nodes in the INDRA subgraph |
| graph_stats.n_edges | Edges in the INDRA subgraph |
| graph_stats.n_regulators | Regulators in the subgraph |
| graph_stats.mean_degree | Mean node degree |
| graph_stats.n_measurable_nodes | Nodes with measured expression data |

**graph_permutation_distributions.csv** --- Per-permutation results.

| Column | Meaning |
|--------|---------|
| set_id | Target set ID or "perm_N" |
| type | "target" or "permutation" |
| roast_pvalue | ROAST p-value |

### Phase 6: Network Proximity Tests

**What they test:** Does the query gene's position in the INDRA causal graph specifically predict the observed differential expression pattern?

**What they rule out:** The set-membership framing itself. Phases 1--5 treat "target" vs "non-target" as a binary classification. But network relationships are continuous: genes one hop from the query are more directly regulated than genes two hops away. Phase 6 tests three continuous, parameter-free hypotheses that avoid arbitrary set boundaries.

All three tests are Bonferroni-corrected at alpha = 0.05/3 = 0.0167 because they are pre-specified independent hypotheses.

**Test 6a --- Proximity Decay:** Genes closer to the query gene in the INDRA graph should have larger differential expression effects. Shortest-path distances are computed via server-side Cypher BFS queries. The test statistic is Spearman's rho between distance and |t-statistic| across all reachable genes. A significantly negative rho means effect sizes decay with distance --- the hallmark of a network-mediated signal. The null is degree-preserving permutation (Guney et al., 2016): genes are binned by degree and permuted within bins, controlling for the confound that hub genes tend to have larger effects.

**Test 6b --- Reverse Causal Reasoning:** Starts from all significantly differentially expressed genes (FDR < 0.05) and asks: which upstream regulators in INDRA best explain the observed pattern? For each regulator with known targets among the DE genes, a signed concordance z-score is computed (activated targets upregulated and repressed targets downregulated count as concordant). The query gene should rank highly if it is truly the upstream cause.

**Test 6c --- Random Walk with Restart (RWR):** Network diffusion from the query gene through the INDRA graph (Cowen et al., 2017). Starting from a unit vector at the seed node, probability mass is iteratively distributed to neighbors via the row-normalized adjacency matrix while returning to the seed with restart probability 0.15. The stationary distribution assigns each gene a proximity score reflecting effective network distance through all paths. The test statistic is Spearman's rho between RWR proximity scores and |t-statistics|. A positive rho means genes the random walk visits frequently from the query are more differentially expressed.

**network_proximity.json** --- Nested results from all three tests.

| Field | Meaning |
|-------|---------|
| proximity_decay.spearman_rho | Correlation between distance and |t| (negative = decay) |
| proximity_decay.permutation_pvalue | Degree-preserving permutation p-value |
| proximity_decay.n_genes_reachable | Genes with a path to the seed |
| proximity_decay.n_genes_unreachable | Measured genes with no path |
| proximity_decay.distance_bins.{d}.n_genes | Gene count at distance d |
| proximity_decay.distance_bins.{d}.mean_abs_t | Mean |t| at distance d |
| proximity_decay.distance_bins.{d}.median_abs_t | Median |t| at distance d |
| reverse_causal.query_gene_rank | Where the query ranks among all regulators |
| reverse_causal.query_gene_zscore | Concordance z-score for the query |
| reverse_causal.n_regulators_tested | Number of regulators with targets among DE genes |
| reverse_causal.n_up_submitted, n_down_submitted | DE genes by direction |
| reverse_causal.top_regulators | Ranked list of regulators with z-scores |
| rwr_correlation.spearman_rho | Correlation between RWR scores and |t| (positive = proximity predicts effect) |
| rwr_correlation.permutation_pvalue | Gene-label permutation p-value |
| rwr_correlation.restart_probability | RWR damping factor (0.15) |
| rwr_correlation.n_graph_nodes, n_graph_edges | Local subgraph size |
| rwr_correlation.convergence_delta | L1 norm change at convergence |
| rwr_correlation.n_iterations | Power iterations to convergence |
| bonferroni_alpha | 0.0167 (0.05 / 3) |
| any_significant | At least one test passed |
| all_significant | All three tests passed |

**proximity_decay_curve.csv** --- For plotting the distance gradient.

| Column | Meaning |
|--------|---------|
| distance | Shortest-path hop count from the seed gene |
| n_genes | Number of genes at this distance |
| mean_abs_t | Mean |t-statistic| at this distance |
| median_abs_t | Median |t-statistic| at this distance |
| std_abs_t | Standard deviation |

A well-behaved proximity decay shows monotonically decreasing mean_abs_t with increasing distance.

**reverse_causal_top_regulators.csv** --- Top upstream regulators ranked by concordance.

| Column | Meaning |
|--------|---------|
| regulator | Gene symbol |
| zscore | Signed concordance z-score |
| rank | Rank among all tested regulators |

### Aggregate Outputs

**validation_report.json** --- The final aggregated report. Contains the verdict, a narrative summary, all phase results nested under their phase names, and a compact phase_details section with one-line summaries per phase.

| Field | Meaning |
|-------|---------|
| verdict | "validated", "inconclusive", or "refuted" |
| summary | Narrative explanation of the verdict |
| phases.{name} | Full result object for each phase (same schemas as above) |
| phase_details.{name} | One-line summary string (e.g., "p=0.031 (pass)") |

**validation_checkpoint.json** --- Persistence for interrupted runs. Same structure as validation_report.json but represents partial state. Stores all completed phase results so the pipeline can resume from the last completed phase rather than restarting. Also persists protein_differential_results needed by downstream phases.

---

## V. Statistical Methods Reference

### Empirical Bayes Moderation

The central problem in omics differential testing is that per-gene variance estimates from small samples are unstable. A gene with two outlier samples might appear to have tiny variance, producing an enormous t-statistic that is actually noise. Empirical Bayes moderation (Smyth, 2004, the engine behind limma) addresses this by shrinking each gene's variance estimate toward a common prior learned from all genes.

Each gene's sample variance s_g^2 is modeled as drawn from a scaled inverse chi-squared prior with hyperparameters d0 (prior degrees of freedom, controlling shrinkage strength) and s0^2 (prior variance, the "typical" variance across all genes). The posterior variance is:

    s_post^2 = (d0 * s0^2 + df * s_g^2) / (d0 + df)

When d0 is large, most genes share a common variance (strong shrinkage). When d0 is small, genes retain individual estimates (weak shrinkage). The moderated t-statistic replaces s_g with s_post and follows a t-distribution with d0 + df degrees of freedom, providing more reliable inference especially with few samples.

### ROAST Rotation Testing

ROAST (Wu et al., 2010) is a self-contained gene set test that asks "is this gene set differentially expressed?" while properly accounting for correlation between genes. Standard permutation-based gene set tests break the correlation structure by permuting gene labels; ROAST instead rotates the data in the residual space, preserving correlations exactly.

The procedure: (1) reparameterize the design matrix via a contrast matrix C so the last coefficient captures the contrast of interest; (2) QR-decompose to extract the residual subspace; (3) for each gene, project expression into this subspace, yielding a point on a hypersphere; (4) generate random rotations on the sphere, producing null t-statistics that respect the original correlations; (5) summarize gene-level z-scores into a set statistic; (6) compare observed to null.

The five set statistics capture different biological scenarios:

- **MEAN**: detects coherent unidirectional regulation (all targets up or all down).
- **FLOORMEAN**: like MEAN but floors individual |z| at ~0.67, dampening noise genes.
- **MEAN50**: mean of the top 50% of |z|-scores, detecting sparse signals where only a fraction of targets respond.
- **MSQ**: mean of squared z-scores, direction-agnostic. Detects bidirectional regulation where some targets increase and others decrease --- critical for regulators that both activate and repress.
- **MIXED**: (MEAN^2 + MSQ) / 2, balancing coherence and magnitude.

P-values are exact Monte Carlo (Phipson & Smyth, 2010): p = (b + 1) / (B + 1), where b counts null statistics >= observed. This formula avoids zero p-values.

### Competitive Enrichment and the Variance Inflation Factor

Competitive testing asks "is this gene set *more* differentially expressed than genes outside the set?" The competitive z-score is:

    z = (mean|t_target| - mean|t_background|) / SE

The VIF (Wu & Smyth, 2012, camera) corrects the standard error for inter-gene correlation: VIF = 1 + (k - 1) * rho_bar, where k is the set size and rho_bar is the mean pairwise correlation within the set. The SE is multiplied by sqrt(VIF). Without this correction, a set of 12 correlated genes moving together would falsely appear enriched because their collective signal exceeds what independent genes would produce --- but they are not independent.

### Fisher Z Transform

Converts a correlation r to z = arctanh(r), which is approximately normal with SE = 1/sqrt(n-3) for Pearson and 1.06/sqrt(n-3) for Spearman. The difference between two Fisher-Z values (z_case - z_ctrl) is approximately normal with SE = sqrt(1/(n_case-3) + 1/(n_ctrl-3)), enabling z-tests and confidence intervals for differential correlation.

### Degree-Preserving Permutation

For network proximity tests, naive gene-label permutation would confound the test with degree: high-degree hub genes tend to have larger effects simply because they participate in more pathways. Degree-preserving permutation (Guney et al., 2016) bins genes by degree and permutes only within the same bin, so the null distribution matches the degree structure of the real data.

---

## VI. Biological Vocabulary

**Expression matrix**: features-by-samples numeric matrix. Features are genes or proteins; values are abundances.

**Log2 fold change (log2FC)**: log2(mean_case / mean_control). Positive means higher in disease; negative means lower. A log2FC of 1.0 means the protein is twice as abundant in disease.

**t-statistic**: effect size (log2FC) divided by its standard error. Larger |t| means more confident the effect is real, accounting for sample-to-sample variability.

**FDR (false discovery rate)**: expected proportion of false positives among all rejected hypotheses. An FDR of 0.05 means ~5% of "significant" findings are expected to be false.

**Coherence ratio**: fraction of a regulator's INDRA targets that form a correlated clique. A ratio of 0.5 means half the known targets co-regulate in the data.

**Rewiring score**: |coherence_case - coherence_ctrl|. Quantifies how much a regulator's co-regulation pattern changed between disease and control.

**HGNC symbol**: canonical human gene name (e.g., TP53, C9orf72) maintained by the HUGO Gene Nomenclature Committee. The standard for cross-database integration.

**Statement type (INDRA)**: the kind of causal relationship. IncreaseAmount and Activation mean the regulator increases the target; DecreaseAmount and Inhibition mean it decreases the target. Phosphorylation indicates post-translational modification.

**Evidence count**: number of independent sources (papers, databases) supporting an INDRA statement. Higher evidence means more confidence in the relationship.

---

## VII. References

Bolstad, B. M., Irizarry, R. A., Astrand, M., & Speed, T. P. (2003). A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics*, 19(2), 185--193.

Choi, M., Chang, C. Y., Clough, T., Brouber, D., Killeen, T., MacLean, B., & Vitek, O. (2014). MSstats: an R package for statistical analysis of quantitative mass spectrometry-based proteomic experiments. *Bioinformatics*, 30(17), 2524--2526.

Cowen, L., Ideker, T., Raphael, B. J., & Sharan, R. (2017). Network propagation: a universal amplifier of genetic associations. *Nature Reviews Genetics*, 18, 551--562.

Guney, E., Menche, J., Vidal, M., & Barabasi, A. L. (2016). Network-based in silico drug efficacy screening. *Nature Communications*, 7, 10331.

Huber, W., von Heydebreck, A., Sultmann, H., Poustka, A., & Vingron, M. (2002). Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*, 18(Suppl 1), S96--S104.

Iglewicz, B. & Hoaglin, D. C. (1993). *Volume 16: How to Detect and Handle Outliers*. ASQ Quality Press.

Kohler, D., Kall, L., Strutz, J., Tsai, T. H., Beltran, P. M. J., Ramaswamy, R., & Choi, M. (2023). MSstats version 4.0: statistical analyses of quantitative mass spectrometry-based proteomic experiments with chromatography-based quantification at scale. *Journal of Proteome Research*, 22(5), 1466--1482.

Langsrud, O. (2005). Rotation tests. *Statistics and Computing*, 15, 53--60.

Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764--766.

Phipson, B. & Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

Smyth, G. K. (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1), Article 3.

Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

Wei, R., Wang, J., Su, M., Jia, E., Chen, S., Chen, T., & Ni, Y. (2018). Missing value imputation approach for mass spectrometry-based metabolomics data. *Scientific Reports*, 8, 663.

Wu, C., Jin, X., Tsueng, G., Afrasiabi, C., & Su, A. I. (2013). BioGPS: building your own mash-up of gene annotations and expression profiles. *Nucleic Acids Research*, 41(D1), D1144--D1150.

Wu, D., Lim, E., Vaillant, F., Asselin-Labat, M. L., Visvader, J. E., & Smyth, G. K. (2010). ROAST: rotation gene set tests for complex microarray experiments. *Bioinformatics*, 26(17), 2176--2182.

Wu, D. & Smyth, G. K. (2012). Camera: a competitive gene set test accounting for inter-gene correlation. *Nucleic Acids Research*, 40(17), e133.
