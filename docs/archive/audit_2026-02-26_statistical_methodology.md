# Statistical Methodology Audit Findings

**Source:** Brutalist multi-agent review (Claude + Gemini), 2026-02-26
**Scope:** `src/cliquefinder/stats/` — all statistical modules
**Reviewer background:** Computational biology, biostatistics, gene set enrichment analysis

---

## Finding Index

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| STAT-1 | [NaN handling in batched OLS is mathematically incorrect](#stat-1-nan-handling-in-batched-ols) | CRITICAL | Valid |
| STAT-2 | [Competitive permutation ignores inter-gene correlation](#stat-2-competitive-permutation-ignores-inter-gene-correlation) | HIGH | Valid |
| STAT-3 | [Competitive z-score is not a proper test statistic](#stat-3-competitive-z-score-is-not-a-proper-test-statistic) | HIGH | Valid |
| STAT-4 | [Satterthwaite df uses non-standard scaling heuristic](#stat-4-satterthwaite-df-uses-non-standard-heuristic) | HIGH | Valid |
| STAT-5 | [ROAST Q2 sign convention is fragile](#stat-5-roast-q2-sign-convention-is-fragile) | HIGH | Valid |
| STAT-6 | [Specificity interaction p-value ignores shared-sample correlation](#stat-6-specificity-interaction-p-value-ignores-shared-sample-correlation) | HIGH | Valid |
| STAT-7 | [Covariate standardization before complete NaN mask](#stat-7-covariate-standardization-before-complete-nan-mask) | HIGH | Valid |
| STAT-8 | [FWER not controlled as claimed in verdict logic](#stat-8-fwer-not-controlled-as-claimed) | HIGH | Valid |
| STAT-9 | [Mismatched control sets in negative controls](#stat-9-mismatched-control-sets-in-negative-controls) | MEDIUM | Valid |
| STAT-10 | [Inconsistent p-value sidedness across modules](#stat-10-inconsistent-p-value-sidedness) | MEDIUM | Valid |
| STAT-11 | [GPU vs CPU numerical divergence (float32 vs float64)](#stat-11-gpu-vs-cpu-numerical-divergence) | MEDIUM | Valid |
| STAT-12 | [Pseudoreplication fix is ad hoc](#stat-12-pseudoreplication-fix-is-ad-hoc) | MEDIUM | Partially valid |
| STAT-13 | [No FDR correction across ROAST gene sets](#stat-13-no-fdr-across-roast-gene-sets) | MEDIUM | Partially valid |
| STAT-14 | [Rotation negative variance truncation](#stat-14-rotation-negative-variance-truncation) | MEDIUM | Valid |
| STAT-15 | [Hardcoded constants without sensitivity analysis](#stat-15-hardcoded-constants) | HIGH | Valid |

---

## STAT-1: NaN Handling in Batched OLS is Mathematically Incorrect

### Status: VALID — CRITICAL

### Location
`stats/differential.py:480-535` (batched_ols_gpu), `stats/differential.py:1164-1171` (run_protein_differential)

### Problem
The batched OLS function replaces NaN with 0 and computes a **shared** `(X'X)^-1` from the full design matrix, then applies it to all features:

```python
Y_clean = np.where(nan_mask, 0.0, Y)
XtX_inv = np.linalg.inv(XtX_np)
beta = XtX_inv @ (X_mx.T @ Y_mx_clean)
```

When different features (proteins) have different missing value patterns — which is the norm in proteomics — the correct OLS solution requires **per-feature** `(X_i'X_i)^-1` where `X_i` is the design matrix restricted to non-missing samples for feature `i`. Using a shared inverse with zero-imputed values:

1. **Biases coefficient estimates**: Zeros drag group means toward zero for features with missing data
2. **Inflates/deflates t-statistics**: The denominator (residual variance) is computed from zero-imputed residuals, which systematically underestimates variance for features with many NaN values
3. **Corrupts p-values**: Both numerator (effect size) and denominator (SE) are wrong

### Mathematical Detail
For feature `g` with observations at sample indices `S_g ⊂ {1,...,n}`:

**Correct:** `β̂_g = (X[S_g]'X[S_g])^-1 X[S_g]' y_g[S_g]`
**Current:** `β̂_g = (X'X)^-1 X' ỹ_g` where `ỹ_g[i] = 0` for `i ∉ S_g`

The current approach adds `|S_g^c|` phantom zero observations, pulling the OLS fit toward predicting zero in all groups.

### Impact
Every feature with missing data has biased log2FC and invalid p-values. In proteomics, missingness rates of 10-40% per protein are typical, meaning this affects a substantial fraction of all test results.

### Solution

**Option A: Per-feature OLS loop (correct but slower)**
```python
for g in range(n_features):
    valid = ~np.isnan(Y[g, :])
    X_g = X[valid, :]
    y_g = Y[g, valid]
    if np.sum(valid) <= X.shape[1]:
        # Insufficient observations
        beta[g, :] = np.nan
        continue
    beta[g, :] = np.linalg.lstsq(X_g, y_g, rcond=None)[0]
```

**Option B: Grouped OLS by missingness pattern (efficient)**
Group features by their missingness pattern (which samples are present). Features sharing the same pattern share a single `(X_g'X_g)^-1`. Typically the number of unique patterns is much smaller than the number of features:

```python
from collections import defaultdict
pattern_groups = defaultdict(list)
for g in range(n_features):
    pattern = tuple(~np.isnan(Y[g, :]))
    pattern_groups[pattern].append(g)

for pattern, gene_indices in pattern_groups.items():
    valid = np.array(pattern)
    X_g = X[valid, :]
    if X_g.shape[0] <= X_g.shape[1]:
        # Rank-deficient for this pattern
        for g in gene_indices:
            beta[g, :] = np.nan
        continue
    XtX_inv_g = np.linalg.inv(X_g.T @ X_g)
    for g in gene_indices:
        y_g = Y[g, valid]
        beta[g, :] = XtX_inv_g @ (X_g.T @ y_g)
```

**Option C: Weighted least squares with zero-weight masking**
Use weights `W[g]` where `W[g][i] = 0` for missing samples, `W[g][i] = 1` otherwise. This is mathematically equivalent to Option A but can be vectorized more easily on GPU:

```python
W = (~nan_mask).astype(np.float64)  # 1 for observed, 0 for missing
# For each feature: beta_g = (X' W_g X)^-1 X' W_g y_g
```

### Pitfalls
- **Option A** is O(n_features × n_samples^2 × n_params) — may be too slow for >10k features without batching
- **Option B** is the best tradeoff — but requires careful handling of near-singular `X_g'X_g` when many samples are missing for a pattern (check rank before inverting)
- **Option C** on GPU requires per-feature matrix inversions in float32, which may lose precision for ill-conditioned designs
- All options change the residual degrees of freedom per feature — downstream Empirical Bayes moderation (`fit_f_dist`) must receive per-feature df, not a single global df
- The EB prior estimation (`s0^2`, `d0`) should ideally be fit on features with similar missingness to avoid heterogeneous prior estimation; in practice, using all features is a reasonable approximation
- Existing tests that mock OLS results may break — update test fixtures

### Priority: P0 — Fix this week

---

## STAT-2: Competitive Permutation Ignores Inter-Gene Correlation

### Status: VALID — HIGH

### Location
`stats/differential.py:1282-1450` (`run_network_enrichment_test`)
`stats/permutation_framework.py:419-511` (`run_competitive_test`)

### Problem
The competitive permutation test samples random gene sets **uniformly** from the measured proteome to build a null distribution of mean |t|. This assumes that inter-gene correlation in random sets matches that of the biological target sets.

For TF target sets and co-regulated modules, genes are **positively correlated** by construction (they share an upstream regulator). Positively correlated genes have:
- Higher variance of the set mean |t| than independent genes
- A wider distribution of the set-level test statistic

The null distribution from random (weakly correlated) gene sets is therefore **too narrow**, producing:
- Inflated z-scores
- Anticonservative (too small) p-values

This is exactly the problem that Camera (Wu & Smyth, NAR 2012) was designed to correct. The framework cites Camera in `method_comparison.py:29-30` but does not implement its inter-gene correlation adjustment.

### Mathematical Detail
For a gene set of size `k` with mean pairwise correlation `ρ̄`:

```
Var(mean|t|) = σ² / k · [1 + (k-1) · ρ̄]     (VIF adjustment)
```

The variance inflation factor `VIF = 1 + (k-1) · ρ̄` can be substantial. For k=50 and ρ̄=0.2, VIF = 10.8 — the null distribution should be ~3.3× wider.

### Impact
The primary enrichment z-score (used in Phase 1, 3, 4 verdict) is anticonservative for correlated gene sets. This is the central statistical claim of the framework.

### Solution

**Option A: Camera-style VIF correction (recommended)**
After computing the null distribution, adjust the z-score:

```python
# Estimate mean pairwise correlation among target genes
target_data = data[target_indices, :]  # (k, n_samples)
corr_matrix = np.corrcoef(target_data)
k = len(target_indices)
# Mean off-diagonal correlation
rho_bar = (corr_matrix.sum() - k) / (k * (k - 1))
rho_bar = max(rho_bar, 0)  # Floor at 0 (negative correlation is conservative)

vif = 1 + (k - 1) * rho_bar
adjusted_z = z_score / np.sqrt(vif)
```

**Option B: Correlation-preserving null sampling**
Sample null gene sets from genes with similar correlation structure. Group all measured genes into correlation neighborhoods and sample within neighborhoods to approximate the target set's correlation.

**Option C: Report both adjusted and unadjusted z-scores**
For transparency, compute both and let the user assess the impact:

```python
@dataclass
class NetworkEnrichmentResult:
    z_score: float                    # Original (may be anticonservative)
    z_score_camera_adjusted: float    # Camera VIF-adjusted
    variance_inflation_factor: float  # VIF for transparency
    mean_pairwise_correlation: float  # ρ̄ for the target set
```

### Pitfalls
- **Estimating ρ̄ from data** uses the same data that generated the t-statistics — slight positive bias. Use the raw expression matrix (pre-differential), not the residuals.
- **Negative ρ̄**: For gene sets with negative average correlation (rare for TF targets), the VIF < 1 and the adjustment makes the z-score MORE significant. Floor ρ̄ at 0 to be conservative.
- **Small gene sets (k < 10)**: The VIF is small regardless of ρ̄, so the correction is minor. The impact is largest for medium-to-large sets (k=30-200).
- **Camera's exact formula** also accounts for df in the correlation estimate; the simplified version above is adequate for proteomics sample sizes (n > 30).
- **Cross-phase consistency**: If Camera-adjusted z is adopted for Phase 1, it must also be used in Phase 3 (_extract_enrichment_z) and Phase 5 (competitive z in negative controls).

### Priority: P1 — Fix this month

---

## STAT-3: Competitive Z-Score is Not a Proper Test Statistic

### Status: VALID — HIGH

### Location
`stats/enrichment_z.py:18-82` (`compute_competitive_z`)

### Problem
The competitive z-score formula is:

```
z = (mean|t_target| - mean|t_background|) / std(|t_background|)
```

This standardizes by the **standard deviation of individual** background |t|-values, not by the **standard error of the mean** of a random set of size k. The correct standardization for a set of size k is:

```
z_correct = (mean|t_target| - mean|t_background|) / (std(|t_background|) / sqrt(k))
```

The current formula therefore computes:

```
z_current = z_correct / sqrt(k)
```

Wait — this is inverted. Let me re-derive:

```
z_current = (diff) / std_pop
z_correct = (diff) / (std_pop / sqrt(k)) = z_current * sqrt(k)
```

So the current z-score is **sqrt(k) times too small** compared to what it should be if interpreted as "how many standard errors above the null mean."

However, this issue is **mitigated** in Phases 1 and 4, where the z-score is computed from the **permutation null distribution** (`differential.py:1396-1404`):

```python
z_score = (observed_mean_abs_t - null_mean) / null_std
```

Here `null_std` is the standard deviation of the permuted set means (which correctly captures the sampling distribution width, including set-size effects). So Phase 1/4 z-scores are correctly calibrated.

The problem is in `enrichment_z.py:compute_competitive_z()`, used by:
- Phase 3 (`label_permutation.py:_extract_enrichment_z`)
- Phase 5 (`negative_controls.py:337-361`)

These use the population-std formula, not the permutation-null-std formula. This creates **cross-phase inconsistency**: Phase 1/4 z-scores are on a different scale from Phase 3/5 z-scores.

### Impact
- Phase 3 and Phase 5 z-scores are not directly comparable to Phase 1/4 z-scores
- The Phase 3 label permutation null compares `observed_z` (population-std) against null `z_scores` (also population-std) from permuted labels, so the **p-value is still valid** (same scale on both sides)
- Phase 5 competitive z percentile (target z rank among control z-scores) is valid for the same reason (same formula applied to both target and controls)
- But cross-phase numerical comparisons (e.g., "Phase 1 z=3.5, Phase 3 z=2.1") are **not meaningful** because they use different formulas

### Solution

**Option A: Standardize all phases to use permutation-null-based z (most correct)**
This requires the inner competitive permutation for every z-score computation, which is expensive. Not practical for Phase 3's 500-permutation outer loop.

**Option B: Use SE-of-mean formula in `compute_competitive_z` (simple fix)**
```python
def compute_competitive_z(all_t, is_target):
    target_t = np.abs(all_t[is_target])
    background_t = np.abs(all_t[~is_target])
    k = len(target_t)
    diff = np.mean(target_t) - np.mean(background_t)
    se = np.std(background_t) / np.sqrt(k)  # SE of the mean, not population std
    return diff / se if se > 1e-10 else 0.0
```

This makes `compute_competitive_z` scale-consistent with the permutation-based z from Phase 1 (both now measure "standard errors above null mean").

**Option C: Document the inconsistency and use z-scores only within-phase**
Label all z-scores with their formula source and only compare z-scores within the same phase, not across phases.

### Pitfalls
- **Option B introduces the inter-gene correlation issue from STAT-2**: the SE formula `std/sqrt(k)` assumes independent genes. For correlated gene sets, the true SE is `std/sqrt(k) * sqrt(VIF)`. If STAT-2 is also fixed, both corrections should be applied together:
  ```python
  se = np.std(background_t) / np.sqrt(k) * np.sqrt(1 + (k-1) * rho_bar)
  ```
- Changing `compute_competitive_z` affects Phase 3 and Phase 5 z-scores numerically, which changes permutation p-values (slightly, because both observed and null use the same formula)
- The validation_report.py verdict logic does NOT use z-score magnitudes for gating (only p-values), so the cross-phase z-score inconsistency does not affect verdicts directly

### Priority: P1 — Fix this month (coordinate with STAT-2)

---

## STAT-4: Satterthwaite DF Uses Non-Standard Scaling Heuristic

### Status: VALID — HIGH

### Location
`stats/differential.py:393-433`

### Problem
The Satterthwaite df computation uses a "balanced approximation" heuristic:

```python
avg_group_size = n_obs / n_groups
v_within = residual_var / avg_group_size
v_between = subject_var
v_theoretical = v_within + v_between
scale = V_c / v_theoretical
```

The standard Satterthwaite df for linear mixed models (as in lmerTest, Kuznetsova et al. 2017) uses the gradient of the contrast variance with respect to each variance component:

```
df_satt = 2 * V_c^2 / sum_i (dV_c/dθ_i)^2 * Var(θ̂_i)
```

The current heuristic:
1. Assumes balanced groups (uses `avg_group_size`)
2. Assumes a specific relationship between contrast variance and variance components
3. Uses an ad hoc scaling factor rather than the proper gradient-based formula
4. Clips to `[1, n_obs - 1]` without diagnosing the underlying issue

### Impact
- Incorrect df → incorrect p-values, especially for **unbalanced designs with few subjects**
- Clipping to `n_obs - 1` can be overly liberal (too many df → too-small p-values)
- For balanced designs with moderate sample sizes, the heuristic is a reasonable approximation

### Solution

**Option A: Implement proper Satterthwaite via lmerTest formula (gold standard)**
Requires computing the Hessian of the log-likelihood at the variance component estimates, which is available from `statsmodels` mixed model fits. Substantial implementation effort.

**Option B: Use Kenward-Roger approximation**
More conservative than Satterthwaite, corrects for small-sample bias in both df and the F-statistic. Available in some R packages but not directly in Python.

**Option C: Fall back to conservative df estimates**
Use `min(n_groups - 1, n_obs - n_params)` as a conservative lower bound on df. This is the "between-within" method that is always valid but may be overly conservative for balanced designs.

**Option D: Use containment df (pragmatic)**
For the simple random-intercept model `y ~ condition + (1|subject)`:
- df for condition effect ≈ n_subjects - n_conditions (when condition varies between subjects)
- This is exact for balanced one-way designs and approximately correct for moderate imbalance

### Pitfalls
- **Option A** requires access to the variance-covariance matrix of the variance component estimates, which statsmodels' `MixedLM` provides via `cov_re` but the Hessian requires additional computation
- **Option D** is the most pragmatic for the current use case (proteomics with subject replicates) and is what MSstats effectively uses
- If implementing Option A, test against R's `lmerTest::contest()` for known designs to validate
- The current clipping `[1, n_obs - 1]` should be replaced with a warning when the computed df is outside reasonable bounds, rather than silent truncation

### Priority: P2 — Fix this month

---

## STAT-5: ROAST Q2 Sign Convention is Fragile

### Status: VALID — HIGH

### Location
`stats/rotation.py:747-789` (simple path), `stats/rotation.py:521-525` (general path)

### Problem
After QR decomposition, the sign of Q2[:,0] is ambiguous (QR decomposition is unique only up to sign of R's diagonal). The code uses a correlation heuristic:

```python
correlation = np.corrcoef(Q2_full[:, 0], contrast_pattern)[0, 1]
if correlation < 0:
    Q2_full[:, 0] = -Q2_full[:, 0]
```

For marginally balanced designs (nearly equal group sizes), the correlation between Q2[:,0] and the contrast direction can be near zero, making the sign flip essentially random. The docstring explicitly warns: "sign correction heuristic that works correctly for two-group designs but would fail silently for complex contrasts."

### Impact
- If sign is assigned incorrectly: UP and DOWN alternative p-values are **swapped**
- MSQ statistic (direction-agnostic) is **unaffected** — only directional tests are at risk
- For well-separated groups (typical in proteomics case/control), the heuristic works reliably
- Risk is elevated for multi-condition designs with >2 groups or continuous covariates

### Solution

**Option A: Use the contrast vector directly (recommended)**
Instead of correlating Q2[:,0] with a derived contrast pattern, use the mathematical relationship between the QR decomposition and the contrast:

```python
# The last column of Q (corresponding to the contrast direction in the
# reparameterized model) should have positive inner product with the
# contrast vector c'X:
contrast_projection = X @ contrast_vector  # n_samples vector
sign = np.sign(Q2_full[:, 0] @ contrast_projection)
if sign < 0:
    Q2_full[:, 0] = -Q2_full[:, 0]
elif sign == 0:
    warnings.warn("Q2 sign is indeterminate — directional p-values unreliable")
```

**Option B: Add a reliability check and warning**
```python
if abs(correlation) < 0.3:
    warnings.warn(
        f"Q2 sign determination unreliable (r={correlation:.3f}). "
        "Directional p-values (UP/DOWN) may be inaccurate. "
        "Use MSQ (direction-agnostic) statistic instead."
    )
```

**Option C: Skip directional tests for ambiguous designs**
Return `NaN` for UP/DOWN p-values when the sign determination is unreliable, forcing users to use MSQ.

### Pitfalls
- **Option A** requires the contrast vector to be available in `compute_rotation_matrices()` — currently only the design matrix and contrast are passed to the general path
- Changing the sign convention affects all existing results that used directional p-values — flag as a breaking change
- The existing MSQ-based verdict logic in the validation framework is unaffected (uses `msq_mixed`)
- Test with synthetic data where ground truth direction is known

### Priority: P2 — Fix this month

---

## STAT-6: Specificity Interaction P-Value Ignores Shared-Sample Correlation

### Status: VALID — HIGH

### Location
`stats/specificity.py:226-228`

### Problem
The interaction z-test computes `Δz = z_primary - z_secondary` where both contrasts share control samples (e.g., C9ORF72-vs-CTRL and Sporadic-vs-CTRL both use CTRL). The null correlation between `z_primary` and `z_secondary` is **computed** (line 238: `null_corr = np.corrcoef(valid_p, valid_s)[0, 1]`) but only reported as metadata — the p-value calculation does not correct for this correlation.

The paired permutation approach (permuting all labels at once, then subsetting) **partially** accounts for this correlation because the same permuted labels generate both z-scores. However, the p-value formula:

```python
interaction_pvalue = (np.sum(np.abs(valid_null) >= abs(observed_dz)) + 1) / (n_valid + 1)
```

computes the fraction of null |Δz| exceeding observed |Δz|. This IS actually the correct permutation p-value for the paired null, since the pairing preserves the correlation structure.

### Re-Assessment After Deeper Analysis
On reflection, the paired permutation approach is **methodologically correct**: by permuting labels once and computing both z-scores from the same permuted data, the null distribution of Δz naturally inherits the shared-sample correlation. The computed `null_corr` is diagnostic metadata, not a correction that needs to be applied.

The finding is **partially valid** in that:
1. The `null_corr` should be prominently reported (not just metadata) to help users interpret the interaction test
2. With only 200 permutations (default), the resolution of the interaction p-value is limited to 1/201 ≈ 0.005
3. The test may be underpowered for detecting moderate specificity differences

### Revised Impact
- The p-value itself is correct for the paired permutation null
- Low permutation count limits resolution
- Users may misinterpret the null correlation as a quality issue when it's actually expected

### Solution
1. Increase default `n_interaction_perms` from 200 to 1000 (improves resolution to 0.001)
2. Add the null correlation to the printed output and specificity summary
3. Add a power note: "With N permutations, the minimum detectable interaction p-value is 1/(N+1)"

### Priority: P3 — Improvement, not a bug

---

## STAT-7: Covariate Standardization Before Complete NaN Mask

### Status: VALID — HIGH

### Location
`stats/design_matrix.py:147-157`

### Problem
Numeric covariate standardization uses `valid_mask` from line 116 (condition-only NaNs). Covariate NaN rows are added to `valid_mask` at line 130-131, **after** the standardization statistics are computed:

```python
# Line 116: valid_mask = ~cond_df["condition"].isna()  (condition NaN only)
# Line 130-131: cov_valid = ~covariates_df.isna().any(axis=1).values
#               valid_mask = valid_mask & cov_valid   (adds covariate NaN)
# Line 148-149: (BEFORE line 130 in the code flow for numeric covariates)
#   valid_vals = vals[valid_mask]  # Uses condition-only mask
#   mu = np.mean(valid_vals)       # Mean includes samples with NaN in OTHER covariates
```

Wait — let me re-read the code flow more carefully. Lines 130-131 update `valid_mask` BEFORE the covariate processing loop at line 134-160. So `valid_mask` at line 148 already includes covariate NaNs from lines 130-131.

### Re-Assessment
Actually, looking at the code structure again:

```python
# Line 116
valid_mask = ~cond_df["condition"].isna()

# Line 122-131 (BEFORE the per-column loop)
if covariates_df is not None:
    cov_valid = ~covariates_df.isna().any(axis=1).values  # NaN in ANY covariate
    valid_mask = valid_mask & cov_valid                      # Updated here

# Line 135-160 (per-column loop, uses already-updated valid_mask)
for col in covariates_df.columns:
    ...
    valid_vals = vals[valid_mask]  # Uses the COMBINED mask
    mu = np.mean(valid_vals)
```

The `valid_mask` IS updated before standardization. The NaN from ALL covariates (`.any(axis=1)`) is incorporated before any numeric covariate is standardized.

### Revised Status: **INVALID** — The code is correct. The NaN mask is consolidated before standardization.

The reviewer misread the code flow. The `cov_valid` computation at line 130 uses `.isna().any(axis=1)` which catches NaN in ANY covariate column, and this is applied to `valid_mask` at line 131, BEFORE the per-column standardization loop starts at line 135.

### Priority: None — no fix needed

---

## STAT-8: FWER Not Controlled as Claimed in Verdict Logic

### Status: VALID — HIGH

### Location
`stats/validation_report.py:73-98` (docstring rationale)

### Problem
The docstring argues that the hierarchical gating structure provides implicit multiplicity control because "joint probability under the global null is alpha^2 when tests are independent." Two issues:

1. **Phase 1 and Phase 3 are not independent**: Both test the same gene set on the same data. If the gene set has large |t|-values by chance, both phases detect it. The joint null probability is substantially larger than alpha^2.

2. **Phase 4 uses the same alpha**: While Phase 4 cannot override a "refuted" verdict, it CAN modulate between "validated" and "inconclusive" (line 185-191). This adds a third test at the same alpha without penalty.

### Quantitative Assessment
Under the global null with correlation ρ between Phase 1 and Phase 3 p-values:

```
P(both pass) = P(Z1 > z_α) × P(Z2 > z_α | Z1 > z_α)
             ≈ α × Φ((z_α - ρ·z_α) / sqrt(1-ρ^2))
```

For α = 0.05 and ρ = 0.5: P(both pass) ≈ 0.012 (vs α^2 = 0.0025 claimed)
For α = 0.05 and ρ = 0.8: P(both pass) ≈ 0.030 (12× the claimed rate)

The actual correlation between Phase 1 (competitive permutation z) and Phase 3 (label permutation z) depends on the data, but is likely 0.3-0.7 given they share the same data and gene set.

### Impact
- The FWER claim in the docstring is overstated
- The actual FWER is controlled but at a higher level than claimed
- For α = 0.05, the effective FWER is likely 0.01-0.03 (still reasonable, but not α^2)

### Solution

**Option A: Correct the docstring (minimal)**
Replace the alpha^2 claim with an honest statement:

```python
"""
Multiple-testing rationale
--------------------------
The joint gating structure provides multiplicity control more stringent
than either test alone, but the exact level depends on the correlation
between Phase 1 and Phase 3 test statistics (which share the same data
and gene set). Under positive correlation, the joint false positive rate
is bounded above by alpha and below by alpha^2, with the actual rate
depending on the test correlation.

For conservative family-wise error control, set alpha=0.01.
"""
```

**Option B: Estimate the Phase 1–3 correlation and report it**
During Phase 3, compute the correlation between the observed z-score and the Phase 1 z-score across permutations. Report this as diagnostic metadata.

**Option C: Apply Bonferroni-Holm to mandatory gates**
Use α/2 for each mandatory gate. This is overly conservative (the gates are positively correlated) but provides a guaranteed FWER ≤ α.

### Pitfalls
- **Option C** reduces power unnecessarily — the gates ARE more stringent than either alone, just not as stringent as claimed
- **Option A** is the most honest approach — the current framework's effective FWER is still quite good, just not α^2
- Users who want formal FWER control can set `--alpha 0.01`

### Priority: P2 — Fix the docstring this sprint, consider Option B for the next version

---

## STAT-9: Mismatched Control Sets in Negative Controls

### Status: VALID — MEDIUM

### Location
`stats/negative_controls.py:432-453`

### Problem
When computing matched competitive z-scores, the function calls `_sample_expression_matched_set()` AGAIN with the same RNG that was already advanced during the matched p-value computation (lines 401-420). This means:

1. Matched p-values (lines 401-420): control set 0 = genes A, B, C
2. Matched z-scores (lines 436-446): control set 0 = genes D, E, F (DIFFERENT genes!)

The `matched_competitive_z_percentile` is computed from different gene sets than the `matched_fpr` and `matched_target_percentile`.

### Impact
- The matched z-score percentile and matched FPR are not paired — they describe different control sets
- Each individual metric is still valid (uniform sampling from same distribution), but they cannot be interpreted as "for the same control sets, both FPR and z-percentile are X"
- The unpaired comparison is still a valid statistical summary of the null distribution

### Solution
Cache the matched gene sets from the first pass and reuse them:

```python
# First pass: generate and cache matched sets
matched_gene_sets = []
for i in range(n_control_sets):
    matched_indices = _sample_expression_matched_set(...)
    matched_genes = [all_gene_ids[idx] for idx in matched_indices]
    matched_gene_sets.append(matched_genes)

# Use cached sets for both p-value and z-score computation
for i, matched_genes in enumerate(matched_gene_sets):
    # Compute ROAST p-value
    result = engine.test_gene_set(gene_set=matched_genes, ...)
    matched_pvalues_arr[i] = ...

    # Compute competitive z-score (SAME genes)
    ctrl_mask = ...
    matched_comp_z[i] = compute_competitive_z(all_t, ctrl_mask)
```

### Pitfalls
- Caching all gene sets requires storing n_control_sets × target_size gene IDs — negligible memory
- The cached approach also avoids the double call to the Hungarian algorithm, saving computation
- Ensure the RNG state is consistent — the fix should produce different overall results because the second set of matched controls was consuming RNG entropy that would otherwise go to other operations

### Priority: P2 — Fix this sprint

---

## STAT-10: Inconsistent P-Value Sidedness Across Modules

### Status: VALID — MEDIUM

### Location
- One-sided: `label_permutation.py:280` — `np.sum(valid_null >= observed_z)`
- Two-sided: `permutation_framework.py:490` — `np.sum(np.abs(null_array) >= np.abs(obs_t))`
- ROAST: `rotation.py:1377-1381` — one-sided UP, DOWN, and two-sided MIXED

### Problem
Different modules use different sidedness conventions:
- Label permutation (Phase 3): one-sided (enrichment = higher z)
- Competitive permutation (framework): two-sided
- ROAST: provides all three alternatives

When these p-values are compared or combined in the validation report, the implicit sidedness affects interpretation. A gene set with strong DOWN-regulation might have a large two-sided p-value but a non-significant one-sided (UP) p-value.

### Impact
- Interpretive confusion when comparing Phase 3 (one-sided) vs competitive permutation (two-sided)
- The validation report's verdict logic uses p-values directly, so sidedness inconsistency can affect verdicts
- In practice, enrichment is typically one-sided (targets more DE than background), so one-sided is the correct default

### Solution
1. Document the sidedness convention for each module in its docstring
2. Ensure the validation report compares like with like (all one-sided or all two-sided)
3. Add a `sidedness` field to all result dataclasses:
   ```python
   sidedness: Literal["one-sided-greater", "one-sided-less", "two-sided"]
   ```

### Pitfalls
- Changing sidedness retroactively changes all p-values — this is a breaking change
- The permutation framework's two-sided test is correct for its use case (testing whether a clique is differentially abundant in either direction)
- Only change sidedness where it's clearly wrong, not for consistency's sake

### Priority: P3 — Document, then harmonize in next major version

---

## STAT-11: GPU vs CPU Numerical Divergence (float32 vs float64)

### Status: VALID — MEDIUM

### Location
`stats/rotation.py:1026-1099` (GPU path) vs `stats/rotation.py:1103-1161` (CPU path)

### Problem
The GPU path uses float32 (MLX on Apple Silicon), while the CPU path uses float64 (NumPy):

```python
U_mx = mx.array(U, dtype=mx.float32)  # GPU: float32
```

For the t→z conversion, the GPU path uses a normal approximation for df > 100 and transfers to CPU for proper t-distribution CDF for small df. The float32 precision loss can affect tail probabilities for extreme t-statistics.

### Impact
- Results differ between GPU and CPU runs — not bitwise reproducible across hardware
- For typical t-statistics (|t| < 5), the difference is negligible (<1e-6 in p-values)
- For extreme t-statistics (|t| > 10), float32 truncation can affect p-values at the 1e-4 level
- The normal approximation for df > 100 is standard (identical to limma's approach)

### Solution

**Option A: Use float64 on GPU where available**
MLX supports float64 on newer Apple Silicon. Check availability:
```python
if mx.default_device() == mx.gpu and hasattr(mx, 'float64'):
    dtype = mx.float64
else:
    dtype = mx.float32
```

**Option B: Compute critical statistics in float64 on CPU**
Keep GPU for bulk matrix operations (rotation, projection) but transfer to CPU for t-statistics and p-values:
```python
# GPU: bulk rotation (float32 is fine for matrix multiplication)
U_rot = U_mx @ R_mx.T  # float32 on GPU

# CPU: statistics (float64 for numerical stability)
t_stats = np.array(U_rot[:, 0], dtype=np.float64) / np.sqrt(var_cpu)
```

**Option C: Document the discrepancy and add a `--no-gpu` flag for reproducibility-critical runs**

### Pitfalls
- MLX float64 support varies by hardware generation
- Transferring every iteration's results to CPU negates GPU speedup — batch the transfer
- The forced CPU conversion at `rotation.py:1086` (`np.array(t_rot.T, dtype=np.float64)`) is already the correct approach; the issue is upstream float32 accumulation

### Priority: P3 — Document, add --exact-precision flag

---

## STAT-12: Pseudoreplication Fix is Ad Hoc

### Status: PARTIALLY VALID — MEDIUM

### Location
`stats/differential.py:740-747`

### Problem
When the mixed model fails to converge, the code falls back to OLS with subject-level aggregation:

```python
df_agg = df.groupby(['subject', 'condition'], ...).agg({'y': 'mean'})
```

Simple averaging ignores within-subject variance heterogeneity (some subjects may have 2 observations, others 10). This inflates the weight of poorly estimated subject means.

### Assessment
This is a reasonable defensive measure (better than OLS on raw data, which has pseudoreplication). The critique that it should use inverse-variance weighting is theoretically correct but:
1. Within-subject variance estimates are noisy with few observations
2. MSstats (the reference implementation) also uses simple averaging as a fallback
3. The primary analysis path (mixed model) handles this correctly

### Solution
Add inverse-variance weighting as a refinement:
```python
# Within each subject × condition, compute mean and weight
grouped = df.groupby(['subject', 'condition'])
agg = grouped.agg(
    y_mean=('y', 'mean'),
    y_var=('y', 'var'),
    n_obs=('y', 'count'),
)
agg['weight'] = agg['n_obs'] / agg['y_var'].clip(lower=1e-10)
```

### Priority: P3 — Nice to have, current approach is adequate

---

## STAT-13: No FDR Correction Across ROAST Gene Sets

### Status: PARTIALLY VALID — MEDIUM

### Location
`stats/rotation.py:2000-2018` (docstring)

### Problem
The docstring states: "ROAST produces exact p-values per gene set via rotation tests. These raw p-values are statistically valid without FDR correction."

This is correct per-test but misleading when testing multiple gene sets. Testing 50 gene sets at p < 0.01 expects ~0.5 false positives.

### Assessment
The statement is not wrong — each individual ROAST p-value IS exact. The issue is the implication that no correction is needed. In the context of this framework:
- The method_comparison module tests multiple cliques, each with multiple methods
- The validation framework tests a single gene set through multiple phases (different question)
- For exploratory use (which the docstring acknowledges: "rank by p-value"), FDR is the user's responsibility

### Solution
Revise the docstring to be less misleading:
```python
"""
ROAST produces exact p-values per gene set. When testing multiple gene sets,
apply FDR correction (e.g., Benjamini-Hochberg) to the collection of p-values.
For exploratory analysis of a single gene set, the raw p-value is valid.
"""
```

### Priority: P3 — Documentation fix

---

## STAT-14: Rotation Negative Variance Truncation

### Status: VALID — MEDIUM

### Location
`stats/rotation.py:1053-1056`

### Problem
When `rho_sq < U_rot_sq` (near-singular rotation), residual SS goes negative. Truncating to 1e-10 creates artificial minimal variance:

```python
residual_ss_rot = rho_sq_mx[:, None] - U_rot_sq
residual_ss_rot = mx.maximum(residual_ss_rot, mx.array(1e-10))
```

This inflates t-statistics for near-singular rotations. The EB shrinkage partially mitigates but doesn't fully correct this.

### Impact
- Affects a small fraction of rotations (those near the numerical boundary)
- For well-conditioned data, this rarely triggers
- For poorly conditioned data (near-zero variance genes), can produce spuriously significant gene sets

### Solution
Instead of truncating, exclude these rotations:
```python
residual_ss_rot = rho_sq - U_rot_sq
valid_rotation = residual_ss_rot > 1e-10
# Only compute t-statistics for valid rotations
# For invalid rotations, set t = 0 (conservative)
t_rot = np.where(valid_rotation, U_rot_0 / np.sqrt(var), 0.0)
```

### Pitfalls
- Setting t=0 for invalid rotations is conservative (deflates the gene set statistic)
- An alternative is to skip invalid rotations entirely and reduce the effective B — but this changes the p-value denominator per gene, complicating the set-level statistic
- The current truncation approach is used by some ROAST implementations (limma uses a similar floor); the critique is valid but the impact is small

### Priority: P3 — Low impact, consider for next major version

---

## STAT-15: Hardcoded Constants Without Sensitivity Analysis

### Status: VALID — HIGH

### Locations

| Constant | Location | Value | Justification | Impact |
|----------|----------|-------|---------------|--------|
| EB fallback threshold | `rotation.py:1615` | 10 valid variances | None stated | Below threshold, EB disabled entirely |
| Active gene threshold | `rotation.py:1408` | sqrt(2) | "AIC-motivated" | Determines gene-level significance |
| Specificity z_threshold | `specificity.py:280` | 1.5 | None stated | **Gates specificity verdict** |
| Neg control percentile | `validation_report.py:172` | 10% | None stated | **Gates supplementary verdict** |
| Min residual df (interaction) | `design_matrix.py:241` | 10 | None stated | Prevents interaction model fitting |
| Condition number warning | `design_matrix.py:256` | 30 | None stated | Warning only |
| Cost matrix variance weight | `negative_controls.py:183` | 0.5 | None stated | Matching quality |
| Cost matrix noise | `negative_controls.py:187` | ±10% | None stated | Diversity of matched sets |
| Min permutations threshold | `permutation_framework.py:482` | 10 | None stated | Sets excluded with <10 valid perms |

The two most impactful are **specificity z_threshold=1.5** and **negative control percentile=10%**, which directly gate the final verdict.

### Solution
1. **Document all constants** with justification (literature reference or empirical rationale)
2. **Expose key thresholds as CLI parameters** with documented defaults:
   ```
   --specificity-z-threshold 1.5
   --negative-control-percentile 10
   --eb-min-variances 10
   ```
3. **Add sensitivity analysis** to the validation report: re-run verdict computation at multiple thresholds and report stability

### Priority: P1 — Document and expose as CLI parameters this month
