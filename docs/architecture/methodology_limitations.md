# Methodology Limitations

**Document version:** 2026-03-02
**Finding IDs:** SCI-III-1 through SCI-III-5 (Audit III)
**Status:** Active — these are inherent limitations, not bugs

This document catalogues known methodological limitations of the
biomolecular clique-finding framework. Each limitation is stated precisely,
its impact on inference is assessed, and available mitigations are described.
Researchers using results from this pipeline should cite the relevant
subsection when reporting findings.

---

## Table of Contents

1. [Knowledge Graph Circularity](#1-knowledge-graph-circularity-sci-iii-1)
2. [Ascertainment Bias in INDRA Edge Counts](#2-ascertainment-bias-in-indra-edge-counts-sci-iii-2)
3. [Validation Phase Non-Independence](#3-validation-phase-non-independence-sci-iii-3)
4. [ROAST Proteomics Validity](#4-roast-proteomics-validity-sci-iii-4)
5. [Statement Type Filtering as Researcher Degree of Freedom](#5-statement-type-filtering-as-researcher-degree-of-freedom-sci-iii-5)

---

## 1. Knowledge Graph Circularity (SCI-III-1)

### The problem

The knowledge-guided pipeline queries the INDRA CoGEx knowledge graph for
regulatory relationships (e.g., "TP53 activates CDKN1A"). INDRA aggregates
statements from literature, pathway databases, and high-throughput screens.
When the pipeline is applied to a disease (e.g., ALS), the INDRA graph may
already contain relationships *derived from ALS studies*. This creates
partial circularity: the gene sets being tested for enrichment in ALS data
were themselves defined, in part, by prior ALS observations.

Concretely, if an INDRA module for regulator R includes target genes
T1, T2, T3 because of ALS-specific publications, then finding enrichment
of {T1, T2, T3} in an ALS proteomics dataset is partially tautological.

### Impact on inference

- **Upward bias on enrichment p-values.** Knowledge-guided gene sets will
  appear more enriched than they would under an unbiased null, because the
  set definition is informed by the same biological system.
- **Inflated discovery rate.** Some "discoveries" may reflect the knowledge
  graph's prior exposure to the disease rather than genuine regulatory
  signal in the new data.

### Mitigations

1. **Data-driven validation (primary).** The pipeline includes a
   variance-driven paradigm (`analyze --paradigm variance`) that constructs
   gene sets entirely from the expression data, with no knowledge graph
   input. Concordance between knowledge-guided and data-driven results
   strengthens confidence that findings are not circular.

2. **Source filtering (potential).** INDRA statements carry provenance
   metadata (source API, PMIDs). A future enhancement could filter
   statements originating from disease-specific publications before
   constructing gene sets, breaking the circularity chain. This is not
   yet implemented but is architecturally feasible via the `stmt_types`
   and source-filtering capabilities of `INDRAModuleExtractor`.

3. **Permutation baseline (existing).** The competitive permutation method
   (`PermutationMethod`) tests whether a gene set is more enriched than
   *random gene sets of the same size*. This null model is agnostic to the
   knowledge graph and provides a distribution-free reference.

### Recommendation

Report knowledge-guided and data-driven results side by side. When
publishing, disclose that INDRA may contain disease-derived relationships
and cite the data-driven concordance as evidence against circularity.

---

## 2. Ascertainment Bias in INDRA Edge Counts (SCI-III-2)

### The problem

Well-studied genes (TP53, MYC, AKT1, EGFR, etc.) have disproportionately
more INDRA edges than less-studied genes. This is a known property of
literature-derived knowledge graphs: "popular" genes accumulate more
publications, more pathway annotations, and therefore more extracted
relationships.

In the knowledge-guided pipeline, `INDRAModuleExtractor.get_indra_targets()`
returns all targets for a regulator. Well-studied regulators yield larger
gene sets. Larger gene sets have higher statistical power in both
self-contained tests (ROAST) and competitive tests (permutation), simply
because they aggregate more features.

### Impact on inference

- **Power asymmetry.** TP53 with 200 targets will have higher power to
  detect enrichment than FOXO3 with 15 targets, even if the true effect
  sizes are identical.
- **Biased ranking.** The `concordance_rank` output (see `concordance.py`)
  ranks gene sets by cross-method significance. This ranking conflates
  biological signal with ascertainment bias: well-studied regulators
  are more likely to rank highly regardless of disease relevance.

### Mitigations

1. **Concordance rank (existing).** The `compute_concordance_rank()`
   function ranks by cross-method agreement rather than raw p-value,
   which partially mitigates single-method power artifacts.

2. **Degree-corrected null model (potential).** A future enhancement
   could implement a null model where random gene sets are sampled with
   the same size distribution as the INDRA-derived sets, rather than
   uniformly. This would control for the power advantage of larger sets.

3. **Matched subsampling (existing, Phase 4).** The validation framework's
   Phase 4 (`stats/matching.py`) performs matched subsampling that accounts
   for gene set size, partially addressing the power asymmetry.

4. **Report gene set sizes.** The `UnifiedCliqueResult` dataclass includes
   `n_proteins` and `n_proteins_found` fields. These should always be
   reported alongside p-values to allow readers to assess whether
   significance is driven by set size.

### Recommendation

When interpreting ranked results, consider gene set size as a confound.
For high-confidence findings, verify that smaller gene sets (n < 20) also
show enrichment, or apply a degree-corrected null.

---

## 3. Validation Phase Non-Independence (SCI-III-3)

### The problem

The 5-phase validation framework (`validate-baselines`) applies multiple
statistical tests to the same underlying data:

| Phase | Test | Shared data |
|-------|------|-------------|
| 1 | Covariate-adjusted enrichment | Expression matrix, design matrix |
| 2 | Multi-contrast specificity | Expression matrix, contrast definitions |
| 3 | Label permutation null | Expression matrix, sample labels |
| 4 | Matched subsampling | Expression matrix, sample metadata |
| 5 | Negative control gene sets | Expression matrix, rotation engine |

All five phases operate on the same expression matrix and share the same
gene set definitions. The underlying test statistics are correlated because
they are computed from overlapping data subsets and share the rotation
engine's precomputed QR decomposition.

### Impact on inference

The estimated inter-phase correlation is rho = 0.6-0.9, depending on the
dataset and phase pair. Under this correlation structure, the effective
number of independent tests is:

```
n_eff = n_phases / (1 + (n_phases - 1) * mean_rho)
```

For n_phases = 5 and mean_rho = 0.7:

```
n_eff = 5 / (1 + 4 * 0.7) = 5 / 3.8 ≈ 1.3
```

This means the five phases provide roughly 1.3 independent pieces of
evidence, not five. The hierarchical verdict system
(`validation_report.py`) treats Phase 1 and Phase 3 as mandatory gates
and uses the remaining phases as supplementary evidence. Because the
verdict requires *all* gates to pass, non-independence makes the verdict
**conservative**: the probability of all phases passing by chance is
*higher* under positive correlation than under independence, meaning the
effective alpha is larger than the nominal product of per-phase alphas.

### Mitigations

1. **Conservative verdict (existing, by design).** The hierarchical gate
   structure already accounts for non-independence implicitly by not
   treating all phases as independent hypothesis tests. The verdict is
   "validated" only if both mandatory gates pass, making it robust to
   the correlation structure.

2. **Do not multiply p-values.** The framework intentionally avoids
   combining per-phase p-values (e.g., via Fisher's method), which would
   require independence. Each phase produces a separate assessment.

3. **Report n_eff.** When citing the validation framework, report that
   n_eff is approximately 1-2 independent tests, not 5.

### Recommendation

Interpret the multi-phase validation as a robustness check with built-in
redundancy, not as five independent replications. The strength of the
framework lies in *qualitative* agreement across complementary perspectives,
not in the multiplicity of tests.

---

## 4. ROAST Proteomics Validity (SCI-III-4)

### The problem

ROAST (Wu et al., 2010) was designed for microarray and RNA-seq data.
Its statistical framework assumes:

1. **Low missingness.** Microarray data has near-complete observations.
   Proteomics (especially DIA/DDA mass spectrometry) routinely has
   20-50% missing values, often not at random (MNAR: low-abundance
   proteins are preferentially missing).

2. **Many features.** RNA-seq measures ~20,000 genes. Proteomics datasets
   typically quantify 3,000-8,000 proteins, reducing the degrees of
   freedom available for the rotation null distribution.

3. **Known variance structure.** ROAST uses empirical Bayes (EB) variance
   shrinkage (Smyth, 2004) that borrows strength across genes. The EB
   prior assumes a scaled inverse chi-squared distribution for gene-level
   variances. Proteomics intensity distributions may violate this
   assumption due to different noise sources (ion suppression, peptide
   identification variability, chromatographic drift).

### Impact on inference

- **Imputation artifacts.** Missing values are typically imputed before
  ROAST analysis. The imputation method (KNN, QRILC, AFT) introduces
  correlation structure that is not accounted for by the rotation null.
  This can inflate or deflate gene set statistics depending on whether
  missingness is informative for group membership.

- **EB shrinkage mismatch.** If the true variance distribution is
  heavier-tailed or multimodal (common in proteomics), EB shrinkage
  will over-regularize high-variance proteins and under-regularize
  low-variance ones, distorting the moderated t-statistics used by ROAST.

- **Reduced rotation space.** With fewer features, the Q2 subspace
  (residual space after projecting out the design matrix) has fewer
  dimensions. The rotation null distribution is less smooth, potentially
  affecting p-value calibration for extreme statistics.

### Why ROAST is still used

The rotation framework is fundamentally **domain-agnostic**: it tests
whether the observed gene set statistic is extreme relative to statistics
computed under random rotations that preserve the correlation structure.
This property holds regardless of the data-generating distribution. The
concern is specifically about the EB variance shrinkage step, which is
applied *before* computing gene-level statistics that feed into ROAST.

### Mitigations

1. **Permutation reference (existing).** The `PermutationMethod` provides
   a distribution-free competitive test that makes no parametric
   assumptions. Concordance between ROAST and permutation results
   validates that EB shrinkage distortions are not driving findings.

2. **Multiple ROAST statistics (existing).** The framework computes MSQ,
   mean, and floormean statistics. MSQ is robust to bidirectional effects;
   mean and floormean are sensitive to directional signal. Disagreement
   across statistics can flag cases where variance model misspecification
   matters.

3. **Imputation sensitivity (potential).** A future enhancement could run
   ROAST under multiple imputation strategies and report the range of
   p-values, quantifying sensitivity to missing data handling.

### References

- Wu D, Lim E, Vaillant F, Asselin-Labat ML, Visvader JE, Smyth GK
  (2010). "ROAST: rotation gene set tests for complex microarray
  experiments." Bioinformatics 26(17):2176-2182.
- Smyth GK (2004). "Linear models and empirical Bayes methods for
  assessing differential expression in microarray experiments."
  Statistical Applications in Genetics and Molecular Biology 3(1):Article 3.

---

## 5. Statement Type Filtering as Researcher Degree of Freedom (SCI-III-5)

### The problem

The `--stmt-types` CLI flag allows users to filter INDRA statements by
type (e.g., `--stmt-types activation`, `--stmt-types phosphorylation`).
The available presets are defined in `cogex.py`:

| Preset | Statement types |
|--------|----------------|
| `regulatory` (default) | IncreaseAmount, Activation, DecreaseAmount, Inhibition |
| `activation` | IncreaseAmount, Activation |
| `repression` | DecreaseAmount, Inhibition |
| `phosphorylation` | Phosphorylation |

Different choices of statement type filter produce different gene sets,
which in turn produce different enrichment results. A researcher who tries
multiple filters and reports only the most significant result inflates the
false discovery rate — a classic "garden of forking paths" problem
(Gelman and Loken, 2013).

### Impact on inference

- **Multiplied hypothesis space.** Testing 4 statement type filters on
  100 gene sets effectively tests 400 hypotheses, but only 100 are
  corrected for multiple testing (within each filter run).
- **Selection bias.** Reporting "the kinase phosphorylation network is
  enriched (p = 0.003)" after trying all four filters is misleading if
  the other three filters showed p > 0.05.

### Mitigations

1. **Pre-registration (recommended).** The default filter
   (`ALL_REGULATORY_TYPES`) should be declared as the primary analysis
   in any pre-registration or analysis plan. Other filters are explicitly
   **sensitivity analyses** and should be labeled as such.

2. **Report all filters tested (recommended).** When multiple statement
   type filters are run, report results for all of them, not just the
   most significant. The `MethodComparisonResult.summary()` output
   facilitates this.

3. **Cross-filter concordance (potential).** A future enhancement could
   compute concordance across statement type filters (analogous to
   cross-method concordance), flagging gene sets that are sensitive to
   filter choice.

### Recommendation

Always run the default `regulatory` filter as the primary analysis.
Document any alternative filters as exploratory sensitivity analyses.
Apply a Bonferroni or Holm correction across filters if results from
non-default filters are used for inference.

### References

- Gelman A, Loken E (2013). "The garden of forking paths: Why multiple
  comparisons can be a problem, even when there is no 'fishing
  expedition' or 'p-hacking' and the research hypothesis was posited
  ahead of time." Department of Statistics, Columbia University.

---

## Summary Table

| ID | Limitation | Impact | Primary mitigation | Status |
|----|-----------|--------|-------------------|--------|
| SCI-III-1 | KG circularity | Upward enrichment bias | Data-driven validation | Documented |
| SCI-III-2 | Ascertainment bias | Power asymmetry by gene set size | Concordance rank, report sizes | Documented |
| SCI-III-3 | Phase non-independence | n_eff << n_phases | Conservative verdict, do not multiply p-values | Documented |
| SCI-III-4 | ROAST proteomics validity | EB shrinkage mismatch, imputation artifacts | Permutation reference, multiple statistics | Documented |
| SCI-III-5 | Statement type filtering | Researcher degree of freedom | Pre-register default, label alternatives | Documented |
