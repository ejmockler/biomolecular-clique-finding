# Validation Framework: Statistical Methodology Notes

## Purpose

This document catalogues known statistical limitations in the control baseline validation framework, with proposed solutions, implementation pitfalls, and priority assessments. Each issue is grounded in the actual codebase with file and line references.

These are not bugs — the current implementation produces correct results within its stated assumptions. These are methodological refinements that would improve the analytical rigor of the framework.

**Last updated:** 2026-02-20

---

## Issue Index

| # | Issue | Priority | Effort | Status |
|---|-------|----------|--------|--------|
| M-1 | [Test statistic mismatch across phases](#m-1-test-statistic-mismatch-across-phases) | P1 | Low | Open |
| M-2 | [Specificity comparison across unequal power](#m-2-specificity-comparison-across-unequal-power) | P2 | Medium | Open |
| M-3 | [Orchestrator lacks checkpointing](#m-3-orchestrator-lacks-checkpointing) | P2 | Low | Open |
| M-4 | [Uniform random negative control gene sets](#m-4-uniform-random-negative-control-gene-sets) | P3 | Medium | Open |
| M-5 | [Bootstrap stability not integrated](#m-5-bootstrap-stability-not-integrated) | P3 | Medium | Open |
| M-6 | [NaN valid mask duplication](#m-6-nan-valid-mask-duplication) | P4 | Low | Open |
| M-7 | [Additive-only covariate adjustment](#m-7-additive-only-covariate-adjustment) | P4 | Medium | Open |
| M-8 | [Covariate documentation and cell-type composition gap](#m-8-covariate-documentation-and-cell-type-composition) | P2 | Low | Open |

---

## M-1: Test Statistic Mismatch Across Phases

### Problem

Phases 1, 3, and 4 use the **competitive** enrichment z-score (mean|t| of targets vs random same-size sets), while Phase 5 uses the **ROAST self-contained** p-value (MSQ mixed statistic). These test different null hypotheses:

| Test | Null Hypothesis | Detects |
|------|----------------|---------|
| Competitive (Phases 1/3/4) | Targets are not more differentially expressed than random same-size sets | Gene-set-specific signal |
| Self-contained / ROAST (Phase 5) | Genes in the set are not differentially expressed at all | Any systematic shift, including genome-wide effects |

**Consequence:** Phase 5's negative control FPR and target percentile are measured on a different scale from the test statistic used in the verdict's mandatory gates. A gene set with strong coordinated regulation (high ROAST significance) but modest enrichment relative to background (low competitive z) will appear well-calibrated in Phase 5 while failing Phases 1/3/4. Conversely, genome-wide differential expression inflates ROAST p-values for all gene sets (including controls), making Phase 5 FPR appear well-calibrated even when it shouldn't be.

### Code References

- Phase 1 competitive z-score: `stats/differential.py:1310` — `z_score = (observed_mean_abs_t - null_mean) / null_std`
- Phase 3 lightweight z-score: `stats/label_permutation.py:62` — `_extract_enrichment_z()` uses identical formula
- Phase 4 reuses Phase 1's `run_network_enrichment_test()`: `cli/validate_baselines.py:378`
- Phase 5 ROAST p-value: `stats/negative_controls.py:142` — `target_result.p_values.get("msq", {}).get("mixed", 1.0)`

### Proposed Solution

Add a competitive z-score alongside the ROAST p-value in Phase 5's negative control loop. For each control gene set, compute both:

1. **ROAST self-contained p-value** (already done via `engine.test_gene_set()`)
2. **Competitive mean|t| z-score** using the same formula as `_extract_enrichment_z()`

This requires access to the protein-level t-statistics (available from Phase 1's `protein_df`). For each control set, compute `mean(|t[control_set]|)` and compare to `mean(|t[~control_set]|)` — this is an $O(n)$ array operation per control set, adding negligible runtime.

Extend `NegativeControlResult` with:

```python
competitive_z_percentile: float      # where target competitive z ranks among controls
competitive_z_fpr: float             # fraction of controls with competitive z > target
control_competitive_z: NDArray       # competitive z for each control set
```

Update the verdict logic to evaluate Phase 5 against the competitive z percentile (consistent with Phases 1/3/4).

### Implementation Pitfall

The t-statistics in `protein_df` are from the OLS/EB fit, while ROAST internally computes its own moderated t-statistics via the rotation framework. These can differ slightly due to different EB shrinkage (ROAST's moderation is applied per-rotation). Using the OLS t-statistics for competitive z is consistent with what `run_network_enrichment_test()` does, so this is the correct choice for cross-phase consistency.

### Verification

After implementation, run on the C9ORF72 dataset and verify:
- ROAST target percentile and competitive z target percentile can diverge (expected for sets with directional coherence but modest magnitude)
- Competitive z FPR approximates alpha (0.05) when using uniform random controls

---

## M-2: Specificity Comparison Across Unequal Power

### Problem

`specificity.py:164-172` classifies a signal as "specific" when the primary contrast is significant and no secondary contrast reaches significance:

```python
if primary_sig and not any_secondary_sig:
    label = "specific"
```

This commits the Gelman & Stern (2006) error: **the difference between "significant" and "not significant" is not itself statistically significant.** In our dataset, C9ORF72 (n=25) vs Sporadic (n=284) have an 11x sample size difference. The secondary contrast may fail to reach significance purely due to power differences, not because the effect is absent.

The competitive enrichment z-score itself (`differential.py:1310`) depends on sample size through the t-statistic distribution. With n=25, t-statistics have fatter tails (df $\approx$ 23), affecting both the observed mean|t| and the null distribution in ways that don't cancel cleanly.

### Code References

- Significance comparison: `stats/specificity.py:164-172`
- Z-score ratio: `stats/specificity.py:157-162`
- Z-score computation (sample-size dependent): `stats/differential.py:1297-1312`

### Proposed Solution: Interaction Z-Test with Correlation Correction

Replace the binary significance comparison with a direct test of whether the effect sizes differ:

$$z_{\text{diff}} = \frac{z_{\text{primary}} - z_{\text{secondary}}}{\sqrt{\text{SE}_1^2 + \text{SE}_2^2 - 2r \cdot \text{SE}_1 \cdot \text{SE}_2}}$$

where:
- $z_{\text{primary}}$, $z_{\text{secondary}}$ are the competitive enrichment z-scores
- $\text{SE}_i = 1$ for standardized z-scores from their respective null distributions
- $r$ is the correlation between the null z-scores (non-zero because contrasts share CTRL samples)

**Estimating $r$:** Run paired permutations — in each permutation, permute labels once across all samples, compute competitive z for both contrasts simultaneously, and compute the empirical correlation across permutation replicates.

**Output:** Replace the binary "specific"/"shared" with:
- `specificity_z_diff`: the interaction z-statistic
- `specificity_p_interaction`: p-value for the difference
- `specificity_ci`: 95% CI for the z-score difference
- `specificity_label`: "specific" if p_interaction < 0.05, "shared" if both contrasts significant and p_interaction > 0.05, "inconclusive" if neither is significant

### Implementation Pitfalls

1. **Shared control samples.** C9 vs CTRL and Sporadic vs CTRL share the same CTRL group. The z-scores are positively correlated because the same control samples affect both. Ignoring this correlation makes the interaction test anti-conservative (SE too small, z_diff inflated). **Mitigation:** The paired permutation approach estimates $r$ empirically, naturally capturing the shared-sample correlation.

2. **Paired permutation cost.** Each permutation requires running `run_protein_differential()` twice (once per contrast) + two calls to `_extract_enrichment_z()`. With ~100ms per differential and ~1μs per z-score: 200 permutations $\approx$ 40 seconds. Acceptable.

3. **Interaction test is underpowered at n=25.** Even if C9ORF72's true effect is 2x larger, the interaction z-test may not detect a significant difference. **Mitigation:** Report the point estimate (z-score ratio) alongside the CI. An inconclusive interaction test with ratio=3.0 and wide CI is more informative than a binary label. The current `specificity_ratio` field already captures this — supplement it with a proper SE.

4. **Directionality.** The competitive z-score is always positive (mean|t| is non-negative). The interaction test correctly handles this — a positive z_diff means the primary has higher enrichment than the secondary.

### Alternative Considered: Shared-Null Permutation

Pool all samples across both contrasts, permute labels jointly, and generate a null distribution for the z-score difference directly. This avoids normality assumptions but requires re-running the pipeline $2 \times N_{\text{perm}}$ times. With the lightweight `_extract_enrichment_z()`, this is feasible (~200 paired permutations in 40 seconds). This is equivalent to Approach A when $N_{\text{perm}}$ is large enough, but more robust for small sample sizes.

**Verdict:** Either approach works. The paired permutation (shared-null) approach is preferable because it avoids distributional assumptions and naturally handles the shared-control correlation.

---

## M-3: Orchestrator Lacks Checkpointing

### Problem

`validate_baselines.py` runs 5 phases sequentially (lines 206-422). If any phase crashes (e.g., matching produces empty result, OOM in label permutation), the `ValidationReport` object in memory is lost — `compute_verdict()` never runs, no `validation_report.json` is produced. Prior phases' JSON files exist on disk but are not aggregated.

Additionally, re-running the pipeline after fixing a crash repeats all phases from scratch, including the expensive INDRA network query.

### Code References

- Sequential phase execution: `cli/validate_baselines.py:206-422`
- Single report save at end: `cli/validate_baselines.py:427-429`

### Proposed Solution: Partial Saves + Per-Phase Error Handling

**Partial saves:** After each phase completes, immediately save the report:

```python
# After Phase 1
report.add_phase("covariate_adjusted", enrichment)
report.save(args.output / "validation_report.json")  # partial

# After Phase 2
report.add_phase("specificity", specificity.to_dict())
report.save(args.output / "validation_report.json")  # updated
# ... etc
```

**Per-phase error handling:** Wrap each phase in try/except. On failure, record the error and continue:

```python
try:
    # ... Phase 4 code ...
    report.add_phase("matched_reanalysis", {...})
except Exception as e:
    import warnings
    warnings.warn(f"Phase 4 (matched reanalysis) failed: {e}")
    report.add_phase("matched_reanalysis", {"status": "failed", "error": str(e)})
report.save(args.output / "validation_report.json")
```

The verdict's `.get()` calls with defaults already handle missing phase data gracefully (`cov.get("empirical_pvalue", 1.0)` defaults to 1.0 which fails the gate). Add a check for `status: "failed"` to skip failed phases in the verdict logic.

### Implementation Pitfalls

1. **Stale checkpoints.** If the user changes the data file or INDRA query but reuses the same output directory, old phase JSON files could be loaded inadvertently if resume-from-checkpoint is added later. **Mitigation:** For now, don't implement resume — just do partial saves. The full pipeline is fast enough (<15 min) that re-running from scratch is acceptable. If resume is added later, hash the input parameters and validate hashes on load.

2. **Phase dependencies for error handling.** Phase 5 needs the fitted ROAST engine. If Phase 1 fails, can we still run Phase 5? The current implementation creates a fresh engine in Phase 5 (`cli/validate_baselines.py:401-407`), so it's independent of Phase 1's engine. The data loading and INDRA query happen before any phases — if those fail, the entire pipeline should abort (not per-phase recoverable).

3. **Verdict with failed phases.** If Phase 3 fails, both gates cannot pass, so the verdict will be "inconclusive" — this is the correct behavior. The summary should mention which phases failed.

### Scope

Partial saves + try/except per phase solve 90% of the problem. Resume-from-checkpoint adds significant complexity (input hashing, cache invalidation, dependency tracking) and can be deferred.

---

## M-4: Uniform Random Negative Control Gene Sets

### Problem

`negative_controls.py:155` samples control gene sets uniformly at random:

```python
control_genes = rng.choice(all_gene_ids, size=target_size, replace=False)
```

INDRA-derived target gene sets are not random genes. Regulatory targets of transcription factors tend to be:
- **More highly expressed** — regulated genes must be expressed to show regulation
- **More variable** — signal-responsive genes show higher variance
- **Co-regulated** — sharing regulatory elements induces inter-gene correlation

Uniform random gene sets will have lower inter-gene correlation and lower mean expression. ROAST's self-contained p-value is valid for any gene set (the rotation null preserves within-set correlation), so individual ROAST tests are unbiased. However, comparing the target gene set's ROAST p-value against the distribution of random-set ROAST p-values produces an overly generous baseline: if 5% of random sets are significant but 15% of pathway-structured sets would be significant, the FPR estimate understates the true rate by 3x.

### Code References

- Uniform sampling: `stats/negative_controls.py:155`
- FPR computation: `stats/negative_controls.py:179` — `fpr = sum(controls < alpha) / n_valid`
- Target percentile: `stats/negative_controls.py:183`

### Proposed Solution: Expression-Matched Control Gene Sets

Sample control genes while matching on marginal expression properties:

1. **Bin target genes by mean expression** — divide the non-target gene pool into deciles based on row-wise mean log2 intensity
2. **Match targets to controls by bin** — for each target gene, select a control gene from the same mean-expression bin
3. **Variance as tiebreaker** — within each bin, prefer control genes with similar row-wise variance

This can be implemented as a bipartite matching problem using `scipy.optimize.linear_sum_assignment` on a distance matrix:

$$d_{ij} = w_1 \cdot |\bar{x}_{\text{target}_i} - \bar{x}_{\text{control}_j}| + w_2 \cdot |\text{var}_{\text{target}_i} - \text{var}_{\text{control}_j}|$$

with $w_1 = 1$, $w_2 = 0.5$ (mean expression is the more important confounder).

**Dual reporting:** Run both uniform-random and expression-matched controls. The gap between their FPR estimates is itself informative — a large gap indicates that expression-level properties confound the naive FPR estimate.

### Implementation Pitfalls

1. **Pool exhaustion.** With ~1,200 measured proteins and ~50 targets, the non-target pool is ~1,150. Matching on two properties across 200 control sets could exhaust good candidates. **Mitigation:** Allow replacement *across* control sets (each set is independently constructed) but not *within* a set. This gives 200 x 50 = 10,000 matchings from a pool of 1,150, which is fine because each set only needs 50 unique genes.

2. **Over-matching.** If control genes are matched to be identical to targets in every way, the target set will never appear special — the very signal being tested gets eliminated. **Mitigation:** Only match on *marginal* properties (mean, variance) that are confounds for set-level tests. Do NOT match on differential expression (the biology) or inter-gene correlation (which ROAST handles internally).

3. **Computational cost.** `linear_sum_assignment` on a 50x1150 cost matrix is O(n^3) but with n=50, this is ~125K operations per control set — negligible. Total: 200 sets x ~1ms = 0.2 seconds.

4. **Bin edge effects.** Targets at the extreme high or low end of the expression distribution may have few matched candidates. **Mitigation:** Use overlapping bins or fallback to nearest-available gene when a bin is exhausted.

### Verification

After implementation:
- Matched controls should have similar mean expression and variance distributions to the target set (verifiable with a KS test)
- Matched FPR should be >= uniform FPR (matched controls are harder to beat)
- If matched and uniform FPR are similar, expression matching is unnecessary for this dataset

---

## M-5: Bootstrap Stability Not Integrated

### Problem

The original C9orf72 analysis showed bootstrap stability of 0.62 — only 62% of bootstrap resamples produce the same qualitative enrichment result (p < 0.05). This reliability metric is not captured in the validation framework. The verdict can report "validated" while the signal is unstable under resampling.

### Relationship to Other Phases

Bootstrap stability answers a different question from the 5 existing phases: **"How sensitive is the result to which specific samples are included?"** This is a precision/reliability measure, not a validity measure. A true but noisy signal (small n) will have low bootstrap stability — that doesn't make it confounded, just uncertain.

| Phase | Question |
|-------|----------|
| Phase 1 | Does the signal survive confound correction? |
| Phase 3 | Is the signal robust to label reassignment? |
| Phase 4 | Does the signal survive covariate balancing? |
| Phase 5 | Is the test calibrated? |
| Bootstrap | How reproducible is the result? |

### Code References

- Existing bootstrap implementation: `stats/bootstrap.py` (direction-aware bootstrap, ~500 lines)
- Bootstrap result docs: `docs/BOOTSTRAP_IMPLEMENTATION.md`, `docs/BOOTSTRAP_QUICK_REFERENCE.md`

### Proposed Solution: Annotation, Not Gate

Add bootstrap stability as a diagnostic annotation in the validation report, not a mandatory gate:

1. **Run bootstrap resampling as a preliminary step** (before Phase 1). Resample patients with replacement (stratified within disease groups to maintain balance), re-run covariate-adjusted competitive enrichment, and check if p < 0.05.

2. **Annotate the report** with `bootstrap_stability` (fraction of resamples where p < 0.05) and `bootstrap_ci` (percentile CI for the z-score).

3. **Qualify the verdict text** when stability is low:

```python
if bootstrap_stability < 0.7:
    qualifier += " [low stability — result may be sensitive to sample composition]"
```

This preserves the verdict's hierarchical logic (gates + supplementary) while surfacing reliability information.

### Implementation Pitfalls

1. **What to bootstrap.** The full pipeline (resample → OLS/EB → competitive enrichment) should be bootstrapped, not just the ROAST test. This ensures the bootstrap probes sensitivity to sample composition through the same statistical pathway used in Phase 1.

2. **Stratified bootstrap.** Resample within disease groups (e.g., resample 25 C9ORF72 patients with replacement from the 25 available, independently resample 284 Sporadic patients). This prevents bootstrap samples that accidentally under-represent one condition.

3. **Covariate consistency.** Each bootstrap resample preserves the covariate values attached to each patient. The covariate-adjusted design matrix is rebuilt per resample. This is automatic if we resample rows of the metadata (preserving all columns).

4. **Runtime.** 1000 bootstrap resamples x (OLS fit ~100ms + z-score ~1μs) $\approx$ 100 seconds. Acceptable.

5. **Bootstrap with small n.** With n=25 C9ORF72, many bootstrap samples will contain only 15-20 unique subjects (expected unique $= n(1 - (1-1/n)^n) \approx 0.632n \approx 16$). This is inherent to bootstrap with small n and produces conservative stability estimates. The low observed stability (0.62) is partly attributable to this — it doesn't necessarily indicate a false signal, just insufficient power.

### Verdict Role

**Not a gate.** A true signal with n=25 will often have stability < 0.7 simply due to sampling variability. Using it as a gate would reject many valid findings. Instead:
- Stability >= 0.8: "highly stable"
- 0.6 <= stability < 0.8: "moderately stable" (annotate in summary)
- Stability < 0.6: "unstable" (strong warning in summary)

---

## M-6: NaN Valid Mask Duplication

### Problem

`design_matrix.py` computes a `sample_mask` (which samples have non-NaN covariate values) as part of the `CovariateDesign` dataclass. Independently, `differential.py` (in `run_protein_differential`) drops NaN samples from the condition array. If these masks disagree — e.g., because NaN patterns differ between the condition column and the covariate columns — the design matrix and data matrix can have misaligned rows.

In practice, both modules receive the same input data and compute compatible masks. But the logic is duplicated without an explicit contract, creating a fragile coupling.

### Code References

- Design matrix NaN handling: `stats/design_matrix.py` — `sample_mask` attribute of `CovariateDesign`
- Differential NaN handling: `stats/differential.py` — within `run_protein_differential()`, condition NaN filter

### Proposed Solution: Single Source of Truth

Have `run_protein_differential()` accept either a raw `covariates_df: pd.DataFrame` (current behavior) or a pre-built `CovariateDesign` object. When a `CovariateDesign` is provided, use its `sample_mask` as the authoritative subset — no independent NaN filtering.

```python
def run_protein_differential(
    data, feature_ids, sample_condition, contrast,
    covariates_df=None,
    covariate_design=None,  # NEW: pre-built CovariateDesign
    ...
):
    if covariate_design is not None:
        # Use pre-computed mask
        valid_mask = covariate_design.sample_mask
        data = data[:, valid_mask]
        sample_condition = np.asarray(sample_condition)[valid_mask]
        # Design matrix already built
    elif covariates_df is not None:
        # Build design matrix (existing path), use its mask
        design = build_covariate_design_matrix(...)
        valid_mask = design.sample_mask
        data = data[:, valid_mask]
        sample_condition = np.asarray(sample_condition)[valid_mask]
```

### Implementation Pitfalls

1. **Interface change.** Adding `covariate_design` to `run_protein_differential()` changes the function signature. Existing callers pass `covariates_df` and continue to work unchanged. The new parameter is opt-in.

2. **Orchestrator usage.** In `validate_baselines.py`, build the `CovariateDesign` once at the top and pass it to all phases. This avoids rebuilding the design matrix per phase and guarantees all phases use the same valid-sample mask.

3. **Backward compatibility.** When neither `covariate_design` nor `covariates_df` is provided, behavior is identical to the current implementation.

### Priority

Low. The current code works correctly because both code paths see the same NaN patterns. This is a code-quality improvement, not a correctness fix.

---

## M-7: Additive-Only Covariate Adjustment

### Problem

The design matrix models covariates as additive effects:

$$Y_g = \beta_0 + \beta_1 \cdot \text{Condition} + \beta_2 \cdot \text{Sex} + \varepsilon_g$$

This assumes the Sex effect is constant across conditions. If male C9ORF72 patients have a distinctive proteomics signature not shared by male Sporadic patients, the interaction term $\text{Condition} \times \text{Sex}$ captures biology that additive adjustment misses. Without interaction terms, this biology is either absorbed into the condition effect (inflating it) or the residual (masking it), depending on the imbalance direction.

### Assessment: Likely Not Critical

In proteomics, interaction effects are typically small relative to main effects. The severe sex imbalance (37.7pp gap) creates a large main-effect confound that additive adjustment handles well. Adding interaction terms reduces degrees of freedom from ~23 to ~22 (with n=25 for C9ORF72) — a marginal cost that doesn't justify the added complexity unless there's evidence of interaction.

Moreover, Phase 4 (matched subsampling) provides a non-parametric sensitivity check that is inherently robust to interaction effects: by creating a sex-balanced subset, the interaction term is equalized between groups by construction.

### Proposed Solution: Optional Diagnostic Flag

Add `--interaction` to the CLI, extending the design matrix:

```python
# In build_covariate_design_matrix():
if interaction_terms:
    for cond_idx, cond_name in enumerate(condition_names[1:], 1):
        for cov_name, cov_col in covariate_map.items():
            interaction_col = X[:, cond_idx] * X[:, cov_col]
            X = np.column_stack([X, interaction_col])
            col_names.append(f"{cond_name}:{cov_name}")
    # Re-validate rank
```

Run with and without interaction. If the competitive enrichment z-score changes by more than 0.5 standard deviations, the interaction effect is meaningful.

### Implementation Pitfalls

1. **Degrees of freedom.** Each interaction term costs 1 df. With n=25, 2 conditions, and 1 binary covariate:
   - Additive: 3 parameters, df=22
   - With interaction: 4 parameters, df=21
   - With 2 covariates + interactions: 6 parameters, df=19

   At df < 20, EB moderation becomes critical (there's not enough data to estimate variance per gene). Our implementation already uses EB moderation, so this is handled.

   **Guard:** Warn when residual df < 15. Refuse to fit when df < 10.

2. **Near-collinearity.** When Sex is nearly confounded with Condition (73% vs 35%), the interaction term $\text{Condition} \times \text{Sex}$ is nearly collinear with the main effects. The existing rank check in `build_covariate_design_matrix()` will catch exact collinearity, but near-collinearity inflates standard errors without triggering the rank check. **Mitigation:** Check the condition number of $X^TX$ and warn if > 30.

3. **Contrast interpretation.** Without interaction: the condition contrast estimates the average condition effect, marginalizing over Sex. With interaction: the condition contrast estimates the condition effect at the reference level of Sex (typically Female, since it's coded 0). These are different quantities. **Mitigation:** Document the contrast interpretation in the report output.

### Priority

Low. Additive adjustment handles the primary confound. Phase 4 (matched subsampling) already provides interaction-robust sensitivity analysis. Implement this as a future diagnostic when there's specific evidence of interaction effects.

---

## M-8: Covariate Documentation and Cell-Type Composition

### Required Covariates for ALS Proteomics
- **Sex**: Binary (M/F). ALS has ~1.5:1 male:female ratio. Must be adjusted.
- **Age at Collection**: Continuous. ALS onset age varies; affects protein expression.
- **Site of Onset**: Categorical (bulbar/limb/respiratory). Different ALS subtypes.
- **Post-Mortem Interval (PMI)**: Continuous. Affects protein degradation.
- **Batch**: Categorical. TMT/iTRAQ batch effects.

### Cell-Type Composition Gap
Bulk tissue proteomics measures a mixture of cell types. ALS involves selective
motor neuron degeneration, meaning disease vs. control differences may partly
reflect cell-type proportion changes rather than per-cell regulatory changes.

Current framework does NOT adjust for cell-type composition. Options:
1. **CIBERSORTx deconvolution** — requires reference single-cell proteomics
2. **Marker gene regression** — include known cell-type markers as covariates
3. **Document as limitation** — note that enrichment may reflect cell composition

### Code References

- Phase 1 covariate adjustment: `stats/design_matrix.py` — `build_covariate_design_matrix()`
- CLI covariate argument: `cli/validate_baselines.py:78` — `--covariates` (default: Sex only)

### Impact on Current Analysis

The current framework adjusts for Sex only. Age, PMI, site of onset, and batch
are available in the Answer ALS metadata but are not included in the default
covariate set. Cell-type composition is not available.

Adding covariates reduces residual degrees of freedom (see M-7). With n=25 for
C9ORF72, each additional covariate costs 1 df. Recommendation: include Age and
PMI as continuous covariates (2 df cost), defer site of onset (sparse categories
may be rank-deficient) and batch (may be confounded with condition).

---

## Cross-Cutting Concern: Statistical Power at n=25

Several issues above are exacerbated by the small C9ORF72 sample size (n=25):

- **M-2:** Specificity interaction test is underpowered
- **M-5:** Bootstrap stability is inherently low
- **M-7:** Interaction terms consume scarce degrees of freedom

This is a dataset limitation, not a framework limitation. The framework correctly surfaces these uncertainties through:
- Wide confidence intervals (when reported)
- Low bootstrap stability values
- Inconclusive verdicts when power is insufficient

The appropriate response is to interpret results with appropriate caution, not to change the statistical methodology. The framework's hierarchical verdict logic already degrades gracefully — with n=25, the gates may not pass, producing "inconclusive" rather than a false "validated."

### Recommendations for Small-n Analysis

1. **Report effect size alongside p-value.** The competitive z-score is an effect size measure. Even if p > 0.05, a z=1.34 indicates moderate enrichment.
2. **Use one-sided tests where direction is pre-specified.** If INDRA predicts downregulation and 89% of targets are downregulated, a one-sided test doubles power.
3. **Consider Bayesian alternatives.** A Bayes factor comparing the enrichment model to the null provides evidence quantification without the sharp significance threshold.

---

## Implementation Roadmap

### Immediate (next implementation cycle)

- **M-1:** Add competitive z-score to Phase 5 negative controls — low effort, high impact on verdict consistency
- **M-3:** Add partial saves + try/except per phase — low effort, prevents data loss

### Near-term (subsequent cycle)

- **M-2:** Implement interaction z-test with paired permutation for shared-control correlation — medium effort, fixes a methodological error
- **M-4:** Add expression-matched control gene sets alongside uniform — medium effort, improves FPR calibration

### Deferred

- **M-5:** Bootstrap stability annotation — medium effort, requires integration with existing bootstrap code
- **M-6:** NaN mask consolidation — low effort, code-quality refactor
- **M-7:** Interaction terms — medium effort, unlikely to matter with current sample sizes
