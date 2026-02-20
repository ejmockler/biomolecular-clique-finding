# Control Baseline Validation Framework

## Overview

The validation framework tests whether observed network differential enrichment signals survive rigorous controls for confounding, calibration, and specificity. It was developed in response to the C9orf72 ALS proteomics analysis, where the INDRA-derived gene set showed coordinated downregulation (ROAST p=0.004) but borderline competitive enrichment (z=1.34, p=0.091), bootstrap stability of 0.62, and severe sex imbalance (cases 73% male, controls 35% male — a 37.7 percentage point gap).

The framework implements five complementary validation phases, each testing a distinct aspect of the signal, plus an orchestrator that runs all phases and produces an aggregate verdict.

### Design Principles

1. **Phases test different hypotheses.** Covariate adjustment asks "does the signal survive confound correction?" Label permutation asks "is the signal robust to label reassignment?" Negative controls ask "is the false positive rate calibrated?" These are not interchangeable — a signal can pass one and fail another for legitimate reasons.

2. **Hierarchical verdict, not voting.** Phases are not equally informative. The covariate-adjusted enrichment and label permutation null are mandatory gates (both must pass). Specificity characterizes the signal without gatekeeping. Matched reanalysis and negative controls provide supplementary evidence.

3. **Same test statistic across phases.** Phases 1, 3, and 4 all use the competitive enrichment z-score (mean|t| of targets vs random same-size sets). This ensures comparability across validation results. Phase 5 (negative controls) uses ROAST's self-contained p-value — a known limitation documented in the methodology notes.

4. **Covariate infrastructure is foundational.** All downstream phases benefit from covariate-adjusted t-statistics. The design matrix infrastructure (Phase 1) feeds into every subsequent analysis.

---

## Phase Architecture

### Phase 1: Covariate-Adjusted Enrichment

**Hypothesis:** The enrichment signal persists after adjusting for known confounders (e.g., sex, batch).

**Implementation:** Extends the linear model design matrix with covariate columns, producing adjusted t-statistics that flow through to both ROAST and competitive enrichment tests.

| Component | File | Key Function |
|-----------|------|-------------|
| Design matrix builder | `stats/design_matrix.py` | `build_covariate_design_matrix()` |
| Covariate design dataclass | `stats/design_matrix.py` | `CovariateDesign` |
| Contrast zero-padding | `stats/design_matrix.py` | `pad_contrast_for_covariates()` |
| ROAST integration | `stats/rotation.py` | `RotationTestEngine.fit(covariates=)` |
| Protein differential | `stats/differential.py` | `run_protein_differential(covariates_df=)` |
| GPU OLS path | `stats/permutation_gpu.py` | `precompute_ols_matrices(covariates_df=)` |

**Design matrix structure:**

```
X = [intercept | condition_dummies | covariate_columns]
```

- Categorical covariates: dummy-coded with `drop_first=True`
- Numeric covariates: standardized to zero mean, unit variance
- Contrast vector: zero-padded for covariate columns (covariates are nuisance, not tested)
- Full column rank validation with informative error on collinearity

**Gate criterion:** Covariate-adjusted competitive enrichment p-value < 0.05.

### Phase 2: Multi-Contrast Specificity

**Hypothesis:** The enrichment is specific to the primary contrast (e.g., C9ORF72 vs CTRL) rather than shared across all contrasts (e.g., also present in Sporadic vs CTRL).

**Implementation:** Runs competitive enrichment independently for each contrast, then compares z-scores and significance.

| Component | File | Key Function |
|-----------|------|-------------|
| Specificity scoring | `stats/specificity.py` | `compute_specificity()` |
| Per-contrast results | `stats/specificity.py` | `ContrastEnrichment` |
| Result dataclass | `stats/specificity.py` | `SpecificityResult` |
| Cohort definition | `cohorts/three_group_als.yaml` | — |

**Classification logic:**

| Condition | Label |
|-----------|-------|
| Primary significant, no secondary significant | `specific` |
| Primary significant, secondary also significant | `shared` |
| Primary not significant | `inconclusive` |

**Verdict role:** Characterization, not gate. A "shared" signal is valid biology — it means the gene set responds to ALS broadly, not just C9ORF72. Only "inconclusive" is a concern.

**Known limitation:** Comparing significance across contrasts with different sample sizes commits the Gelman & Stern (2006) error. See methodology notes for proposed solution.

### Phase 3: Label Permutation Null

**Hypothesis:** The enrichment signal is robust to random reassignment of condition labels (the signal is not an artifact of the specific label configuration).

**Implementation:** Permutes condition labels and re-runs the full protein differential + enrichment pipeline. Both stratified (within covariate strata) and free permutation modes are run and reported separately.

| Component | File | Key Function |
|-----------|------|-------------|
| Stratified permutation | `stats/label_permutation.py` | `generate_stratified_permutation()` |
| Free permutation | `stats/label_permutation.py` | `generate_free_permutation()` |
| Permutation loop | `stats/label_permutation.py` | `run_label_permutation_null()` |
| Lightweight z-score | `stats/label_permutation.py` | `_extract_enrichment_z()` |
| Result dataclass | `stats/label_permutation.py` | `LabelPermutationResult` |

**Per-permutation loop:**

1. Permute condition labels (stratified within Sex strata, or free)
2. `run_protein_differential()` with permuted labels + covariates
3. `_extract_enrichment_z()` — lightweight mean|t| z-score (NO inner competitive permutations)
4. Collect null z-score

**Performance:** `_extract_enrichment_z()` replaces `run_network_enrichment_test()` in the permutation loop. The latter runs 10,000 inner competitive permutations per call — with 500 outer permutations, that would be 5,000,000 unnecessary inner permutations. The lightweight function computes the same z-score statistic directly from the t-statistic array in microseconds.

**Gate criterion:** Stratified permutation p-value < 0.05. Free permutation p-value reported alongside. If stratified passes but free fails, a warning is raised indicating the signal may partly reflect covariate structure.

### Phase 4: Matched Subsampling Reanalysis

**Hypothesis:** The enrichment signal persists in a covariate-balanced subset (non-parametric sensitivity check complementing Phase 1's parametric adjustment).

**Implementation:** Exact covariate matching creates balanced subsets by downsampling. For each stratum of the match variables, finds the minimum group size across conditions and randomly samples to match.

| Component | File | Key Function |
|-----------|------|-------------|
| Matching algorithm | `stats/matching.py` | `exact_match_covariates()` |
| Result dataclass | `stats/matching.py` | `MatchResult` |

**Example:** C9ORF72 has 17M/8F, Sporadic has 207M/77F. Matched subset: 17M + 8F from each group. Total matched samples: (17+8) x 2 = 50 from original ~309.

**Positional indexing:** Uses `np.where(group_mask.values)[0]` for positional indices, avoiding crashes when metadata has string-based index (e.g., sample IDs).

**Power warning:** When matching retains < 30% of samples, a `UserWarning` is raised noting potential loss of statistical power.

**Verdict role:** Supplementary evidence. Evaluated against same competitive enrichment p < 0.05 threshold as Phase 1.

### Phase 5: Negative Control Gene Sets

**Hypothesis:** The observed gene set significance is not an artifact of the gene set size or the statistical test — random gene sets of the same size produce appropriately calibrated p-values.

**Implementation:** Reuses the already-fitted ROAST engine (QR decomposition computed once). Only gene set membership changes per iteration.

| Component | File | Key Function |
|-----------|------|-------------|
| Negative control loop | `stats/negative_controls.py` | `run_negative_control_sets()` |
| Result dataclass | `stats/negative_controls.py` | `NegativeControlResult` |

**Output metrics:**
- **FPR:** Fraction of control sets with p < alpha (should approximate alpha)
- **Target percentile:** Where the target p-value falls among controls (lower = more significant than random sets)
- **Control p-value distribution:** Quantiles for calibration assessment

**ROAST API usage:** Calls `engine.test_gene_set(gene_set=..., gene_set_id=...)` which returns a `RotationResult` dataclass. P-value extracted via `result.p_values.get("msq", {}).get("mixed", 1.0)`.

**Verdict role:** Supplementary. Target percentile < 10% indicates the gene set is in the bottom decile compared to random sets.

**Known limitation:** Uses ROAST self-contained p-value while other phases use competitive z-score. See methodology notes.

### Phase 6: Orchestrator

**Implementation:** Single CLI command that runs all phases sequentially and produces an aggregate report.

| Component | File | Key Function |
|-----------|------|-------------|
| CLI registration | `cli/validate_baselines.py` | `register_parser()` |
| Orchestrator | `cli/validate_baselines.py` | `run_validate_baselines()` |
| Report aggregation | `stats/validation_report.py` | `ValidationReport` |
| Verdict computation | `stats/validation_report.py` | `ValidationReport.compute_verdict()` |

**CLI usage:**

```bash
cliquefinder validate-baselines \
    --data data.csv --metadata metadata.csv \
    --network-query C9ORF72 --output output/validation/ \
    --cohort-config cohorts/three_group_als.yaml \
    --covariates Sex \
    --match-covariates Sex \
    --label-permutations 500 \
    --negative-control-sets 200
```

**Individual phase arguments** are also available on the `differential` subcommand:

```bash
cliquefinder differential ... \
    --covariates Sex \
    --label-permutation-null 500 --permutation-stratify Sex \
    --match-covariates Sex \
    --negative-control-sets 200
```

---

## Verdict Logic

The verdict uses hierarchical logic with mandatory gates and supplementary evidence.

### Mandatory Gates

Both must pass for a "validated" verdict:

| Gate | Source | Criterion |
|------|--------|-----------|
| Covariate-adjusted enrichment (Phase 1) | `covariate_adjusted.empirical_pvalue` | p < 0.05 |
| Label permutation null (Phase 3) | `label_permutation.stratified.permutation_pvalue` | p < 0.05 |

### Supplementary Evidence

Counted but not gatekeeping:

| Phase | Criterion | Interpretation |
|-------|-----------|----------------|
| Phase 2 (Specificity) | `specificity_label` in {"specific", "shared"} | Characterizes the signal; "shared" is valid biology |
| Phase 4 (Matched) | `empirical_pvalue` < 0.05 | Non-parametric confirmation |
| Phase 5 (Neg. controls) | `target_percentile` < 10.0 | Calibration check |

### Verdict Outcomes

| Both gates pass | Supplementary | Verdict |
|-----------------|---------------|---------|
| Yes | >= 1 pass | **validated** (with specificity qualifier) |
| Yes | All fail | **inconclusive** — calibration concerning |
| Adjusted only | — | **inconclusive** — not robust to label reassignment |
| Permutation only | — | **inconclusive** — may reflect confounding |
| Neither | — | **refuted** — likely spurious |

### Permutation Divergence Warning

When stratified permutation passes but free permutation fails, the report includes:

> "Stratified passes but free fails — signal may partly reflect covariate structure."

This indicates the signal is robust within covariate strata but not when covariate balance is disrupted, suggesting partial confounding.

---

## Dependency Graph

```
Phase 1 (Covariate Design Matrix) ← foundation for all phases
    |
    ├── Phase 2 (Multi-Contrast Specificity) ← needs covariate-adjusted models
    ├── Phase 3 (Label Permutation) ← needs covariate-adjusted differential
    ├── Phase 4 (Matched Subsampling) ← independent, but benefits from covariates
    └── Phase 5 (Negative Controls) ← needs fitted ROAST engine
                    |
Phase 6 (Orchestrator) ← composes Phases 1-5, produces ValidationReport
```

Phases 2-5 are operationally independent after Phase 1 completes.

---

## File Map

### Source Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/cliquefinder/stats/design_matrix.py` | Covariate-aware design matrix builder | ~150 |
| `src/cliquefinder/stats/specificity.py` | Multi-contrast specificity scoring | ~207 |
| `src/cliquefinder/stats/label_permutation.py` | Stratified + free label permutation null | ~300 |
| `src/cliquefinder/stats/matching.py` | Exact covariate matching | ~200 |
| `src/cliquefinder/stats/negative_controls.py` | Random gene set FPR calibration | ~205 |
| `src/cliquefinder/stats/validation_report.py` | Report aggregation + hierarchical verdict | ~196 |
| `src/cliquefinder/cli/validate_baselines.py` | Orchestrator CLI command | ~435 |
| `cohorts/three_group_als.yaml` | 3-group cohort (C9ORF72, SPORADIC, CONTROL) | ~20 |

### Modified Files

| File | Changes |
|------|---------|
| `src/cliquefinder/stats/rotation.py` | `fit()` gains `covariates` kwarg, delegates to `fit_general()` |
| `src/cliquefinder/stats/differential.py` | `run_protein_differential()` and `build_contrast_matrix()` gain covariate support |
| `src/cliquefinder/stats/permutation_gpu.py` | `precompute_ols_matrices()` gains `covariates_df` |
| `src/cliquefinder/cli/differential.py` | 5 new CLI args; matching, label permutation, negative controls wiring |
| `src/cliquefinder/cli/__init__.py` | Register `validate-baselines` subcommand |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_covariate_design.py` | 16 | Design matrix construction, rank validation, NaN handling |
| `tests/test_specificity.py` | 10 | Specificity classification, thresholds, serialization |
| `tests/test_label_permutation.py` | 11 | Stratified preservation, signal detection, covariates |
| `tests/test_matching.py` | 10 | Balance verification, edge cases, reproducibility |
| `tests/test_negative_controls.py` | 8 | FPR calibration, API correctness, error handling |

---

## References

- Wu, D. et al. (2010). "ROAST: rotation gene set tests for complex microarray experiments." *Bioinformatics*, 26(17), 2176-2182.
- Subramanian, A. et al. (2005). "Gene set enrichment analysis." *PNAS*, 102(43), 15545-15550.
- Wu, D. & Smyth, G.K. (2012). "Camera: a competitive gene set test." *NAR*, 40(17), e133.
- Gelman, A. & Stern, H. (2006). "The difference between 'significant' and 'not significant' is not itself statistically significant." *The American Statistician*, 60(4), 328-331.
