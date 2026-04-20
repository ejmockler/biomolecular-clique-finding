# Audit II — Detailed Findings Reference

**Date**: 2026-02-27
**Companion to**: `audit_2026-02-27_remediation_plan.md`
**Purpose**: Full technical detail, code context, and impact analysis for each finding.
Cross-reference the remediation plan for solution approaches and wave assignments.

---

## Domain 1: Statistical Core Engine

### STAT-CORE-1 — CRITICAL
**VSN MLX GPU path computes wrong reference array when data contains NaN**
**File**: `normalization.py:601`

The GPU VSN path computes the reference as:
```python
ref = mx.mean(mx.where(mx.isnan(y), 0.0, y), axis=1)
```
This replaces NaN with 0.0 before averaging. The denominator includes the count of
NaN entries, making this NOT equivalent to `np.nanmean`. For a feature with NaN in
4 of 10 samples, the reference is `sum_of_6_valid / 10` instead of `sum_of_6_valid / 6`.

The CPU path at line 489 correctly uses `np.nanmean(y, axis=1)`.

**Impact**: Systematic underestimation of feature-level signal in the GPU path.
Bias scales linearly with the missing data rate. At 30% missingness, the reference
is biased by ~30%, causing parameter estimation errors that compound over VSN iterations.

### STAT-CORE-2 — HIGH
**UnboundLocalError when max_iter=0 in tukey_median_polish**
**File**: `summarization.py:156`

When `max_iter=0`, the `for iteration in range(max_iter)` loop never executes,
leaving `iteration` unbound. The return uses `iterations=iteration + 1`.

### STAT-CORE-3 — HIGH
**AFT imputation uses population std (ddof=0) instead of sample std**
**File**: `missing.py:328`

`np.std(observed)` defaults to ddof=0. For n=3: underestimates sigma by ~18%.
The truncated normal draw `norm.ppf(u, loc=mu, scale=sigma)` then clusters
imputed values too tightly, creating artificially low variance that inflates
downstream t-statistics. Also appears in QRILC at line 398.

### STAT-CORE-4 — HIGH
**impute_min_value crashes on all-NaN input**
**File**: `missing.py:238`

`np.nanmin(data)` raises `ValueError: zero-size reduction operation fmin`
when the entire data matrix is NaN. Occurs in `feature`, `global`, and `sample` paths.

### STAT-CORE-5 — HIGH
**Deprecated scipy.stats.binom_test (removed SciPy ≥1.12)**
**File**: `differential.py:1598`

Uses `hasattr` fallback pattern. The two APIs (`binom_test` vs `binomtest`) use
different exact methods for two-sided tests, producing slightly different p-values.

### STAT-CORE-6 — HIGH
**Anti-conservative p-values when EB enabled but d0=inf**
**File**: `differential.py:1424`

When d0=inf (no shrinkage needed), the code uses Normal instead of t(df_residual).
For df=5: `t.sf(2.3, 5)=0.035` vs `norm.sf(2.3)=0.011` — a 3x difference.
Uses t-statistic with original (unmoderated) variance but wrong reference distribution.

### STAT-CORE-7 — MEDIUM
**logsumexp does not handle NaN — silent data loss**
**File**: `summarization.py:198`

`scipy.special.logsumexp` propagates NaN to entire column. Every other
summarization method (MEDIAN, MEAN, TMP) uses nan-aware functions.
LOGSUM silently drops samples with any missing features.

### STAT-CORE-8 — MEDIUM
**OLS formula asymmetry with pinv fallback**
**File**: `differential.py:1347`

Formula `beta = Y @ X @ XtX_inv.T` relies on symmetry of `(X'X)^{-1}`.
When `pinv` is used (near-singular matrices), pseudo-inverse may not be symmetric.
GPU path at line 615 uses standard `XtX_inv @ (X' @ Y)` which is correct for pinv.

### STAT-CORE-9 — MEDIUM
**VSN MLX initialization fragile for all-non-positive columns**
**File**: `normalization.py:589`

Double-indexing pattern `data[:,j][~np.isnan(data[:,j])][data[:,j][~np.isnan(data[:,j])] > 0]`
is fragile and hard to verify. CPU version handles same case cleanly with explicit checks.

### STAT-CORE-10 — MEDIUM
**Li-Ji M_eff uses filtered eigenvalue count**
**File**: `correlation_tests.py:386`

After `eigenvalues = eigenvalues[eigenvalues > 1e-10]`, `len(eigenvalues)` is less
than M (original test count). The Li & Ji formula should use M, not the filtered count.
Over-estimates effective tests → insufficient multiple testing correction.

### STAT-CORE-11 — MEDIUM
**AFT draws u=0 → norm.ppf(0)=-inf**
**File**: `missing.py:344`

`rng.uniform(0, phi_threshold)` can produce exactly 0 (probability ~2^-53 per draw).
`norm.ppf(0) = -inf` propagates silently through downstream computation.

### STAT-CORE-12 — MEDIUM (= GPU-8)
**Median df for EB hyperparameter estimation discards per-feature heterogeneity**
**File**: `differential.py:1389`

`int(np.median(df_valid))` collapses per-feature df array to scalar before `fit_f_dist`.
`fit_f_dist` already supports array df (lines 151-154). R limma uses per-feature df.

### STAT-CORE-13 — MEDIUM
**Quantile normalization "simple" method sorts NaN, corrupting target distribution**
**File**: `normalization.py:184`

`np.sort` pushes NaN to end. Columns with different NaN counts become misaligned.
`np.nanmean` across misaligned rows biases upper quantiles of target distribution.
The "censored" method avoids this. Known limitation of "simple" method.

### STAT-CORE-14 — MEDIUM
**QRILC global parameters use ddof=0**
**File**: `missing.py:398`

Same class as STAT-CORE-3. `np.std(all_observed)` uses population std.

### STAT-CORE-15 — MEDIUM
**Mixed model fallback residual df off-by-one**
**File**: `differential.py:821`

`residual_df = max(n_groups - n_fixed, len(df) - n_fixed - 1)` — the extra `-1`
may be an erroneous accounting for the random effect. Standard within-subject df
is `n_obs - n_groups`, not `n_obs - n_fixed - 1`.

### STAT-CORE-16 — LOW
**Population std in CV diagnostic**
**File**: `normalization.py:746`

`np.std(medians_before)` uses ddof=0 for normalization quality CV.

### STAT-CORE-17 — LOW
**Population std in CliqueSummary.to_dict()**
**File**: `summarization.py:253`

`np.nanstd(self.sample_abundances)` uses ddof=0.

### STAT-CORE-18 — LOW
**Censoring threshold at 0.1th percentile**
**File**: `missing.py:156`

`np.percentile(observed, 0.1)` — estimated by ~10 data points for typical datasets.
Used for diagnostics only, not imputation decisions.

### STAT-CORE-19 — LOW
**Fisher Z clip at ±0.9999**
**File**: `correlation_tests.py:44`

Tighter bound like `1 - 1e-10` preserves more precision for near-perfect correlations.

### STAT-CORE-20 — INFO
**Dead n_draws parameter in impute_aft_model**
**File**: `missing.py:267`

Parameter accepted but never used for multiple imputation. API contract violation.

---

## Domain 2: Set-Level Testing

### SET-TEST-1 — HIGH
**MEAN statistic for MIXED alternative uses |z| instead of signed z**
**File**: `rotation.py:1394`

ROAST paper (Wu et al. 2010) defines MEAN for mixed/two-sided as the signed mean,
with directionality tested by comparing `|observed| >= |null|`. The code uses
`np.sum(np.abs(w) * np.abs(z), axis=1) / A` — a mean-absolute-deviation test.
This conflates MEAN-MIXED with FLOORMEAN(floor=0).

### SET-TEST-2 — HIGH
**FLOORMEAN for UP/DOWN alternatives omits floor**
**File**: `rotation.py:1399`

ROAST floormean: `T = sum(a_g * max(|z_g|, sqrt(q))) / A`. The floor `sqrt(q)` ≈ 0.6745
is applied in MIXED but omitted in UP/DOWN, where `max(z, 0)` (ReLU) is used instead
of `max(z, floor)`. Noisy small positive z-scores contribute 0 instead of `floor`.

### SET-TEST-3 — HIGH
**NaN conditions cause Q2/data dimension mismatch**
**File**: `rotation.py:748`

`compute_rotation_matrices` filters NaN conditions, producing Q2 with `n_valid` rows.
`RotationTestEngine.fit()` passes `self.data` (all n_samples columns) to
`extract_gene_effects`, causing shape mismatch. Multi-group datasets crash.

### SET-TEST-4 — HIGH
**Bootstrap with-replacement creates duplicate DataFrame indices**
**File**: `bootstrap_comparison.py:297`

`rng.choice(case_samples, ...)` with `replace=True` returns duplicate IDs.
`metadata.loc[bootstrap_samples]` creates a DataFrame with duplicate index.
Downstream `run_method_comparison` may incorrectly handle duplicate indices.

### SET-TEST-5 — MEDIUM
**Legacy np.random.seed() usage (3 call sites)**
**Files**: `clique_analysis.py:1237,1533`, `permutation_gpu.py:1127`

Mutates global NumPy RNG state. Not thread-safe. Deprecated since NumPy 1.17.
The S-7 remediation fixed `bootstrap_comparison.py` and `precompute_random_indices`
but missed these three sites.

### SET-TEST-6 — MEDIUM
**Hard-coded df>100 threshold for t→z approximation**
**File**: `rotation.py:1212`

At df=100, the t-to-normal approximation has ~2% error for |t|=3, ~4% for |t|=5.
With EB moderation (d0 ~50-80), df easily exceeds 100 even with small samples.
MSQ statistic amplifies error quadratically.

### SET-TEST-7 — MEDIUM
**MEAN50 selection criterion uses w*z not |z|**
**File**: `rotation.py:1418`

limma selects top 50% by |z| (unweighted), then computes weighted mean.
The code selects by w*z, conflating weight with signal rank.

### SET-TEST-8 — MEDIUM
**Independent rotation vectors per gene set prevent FWER correction**
**File**: `rotation.py:2059`

Each `test_gene_set` call generates independent rotations. limma's `mroast`
generates once and applies to all sets, enabling tighter FDR control.

### SET-TEST-9 — MEDIUM
**GPU OLS divides by integer df in float32**
**File**: `permutation_gpu.py:636`

`sigma2 = rss / matrices.df_residual` — float32 division for small df (3-5).

### SET-TEST-10 — MEDIUM
**Normal approximation for unmoderated permutation p-values**
**File**: `permutation_gpu.py:1515`

When EB disabled: `norm.sf(|t|)` instead of `t.sf(|t|, df_residual)`.
For df=8: 2.3x error in p-value at t=2.3.

### SET-TEST-11 — MEDIUM
**Sample weights not applied to expression data Y**
**File**: `rotation.py:794`

Design matrix weighted by W_sqrt but Y is not. The QR decomposition produces
Q2 in weighted space but it's applied to unweighted data, breaking orthogonality.

### SET-TEST-12 — MEDIUM
**GPU rotation normalization in float32**
**File**: `rotation.py:984`

For high-dimensional residual spaces (n_dims > 100), float32 sum-of-squares
loses precision. Normalized "unit" vectors deviate from ||R||=1 by ~1e-6.

### SET-TEST-13 — LOW
**PermutationResult zscore uses ddof=0**
**File**: `permutation_framework.py:218`

### SET-TEST-14 — LOW
**Bootstrap selection frequency denominator inconsistency**
**File**: `bootstrap_comparison.py:393`

### SET-TEST-15 — LOW
**MSQ UP/DOWN zeros opposite direction**
**File**: `rotation.py:1457`

Design choice, not bug. Document that MEAN is preferred for directional tests.

### SET-TEST-16 — LOW
**fit_f_dist heterogeneous df weighting**
**File**: `permutation_gpu.py:104`

Simple `np.var(e, ddof=1)` treats all features equally regardless of df.
R limma uses weighted moments. Impact small when df is uniform.

### SET-TEST-17 — INFO
**Single-gene sets return empty RotationResult**
**File**: `rotation.py:1998`

---

## Domain 3: Method Comparison Framework

### MCOMP-1 — HIGH
**robust_hits() returns empty when any method fails**
**File**: `concordance.py:562`

Failed methods produce NaN columns in wide format. `NaN < threshold` = False.
`.all(axis=1)` returns False for every row when ANY column is NaN.
Comment says "dropna behavior" but `.all()` does NOT drop NaN.

### MCOMP-2 — HIGH
**wide_format() includes invalid results (NaN/inf p-values)**
**File**: `concordance.py:401`

No `is_valid` filter. A result with `p_value=-inf` passes `< threshold`.
`identify_disagreements()` filters by `is_valid` but `wide_format()` does not.

### MCOMP-3 — MEDIUM
**Permutation adapter clamps p-value before storing**
**File**: `methods/permutation.py:157`

True empirical p=0.0 replaced with 1e-15. Stored in UnifiedCliqueResult.

### MCOMP-4 — MEDIUM
**O(N*M) clique lookup in permutation result conversion**
**File**: `methods/permutation.py:168`

### MCOMP-5 — MEDIUM
**n_proteins_found uses wrong lookup domain**
**File**: `methods/permutation.py:179`

### MCOMP-6 — MEDIUM
**MethodComparisonResult is not frozen**
**File**: `concordance.py:318`

Mutable dict fields. Only output type in framework lacking immutability.

### MCOMP-7 — MEDIUM
**Mutable metadata passed to permutation engine**
**File**: `methods/permutation.py:128`

### MCOMP-8 — LOW
**Misleading "dropna" comment**
**File**: `concordance.py:570`

### MCOMP-9 — LOW
**Read-only array passed to ROAST engine**
**File**: `methods/roast.py:141`

### MCOMP-10 — LOW
**Inconsistent key types (MethodName enum vs str)**
**File**: `method_comparison.py:253`

### MCOMP-11 — LOW
**Duplicate clique_id in null_df not guarded**
**File**: `methods/permutation.py:187`

### MCOMP-12 — INFO
**Hardcoded "up" direction for effect_size**
**File**: `methods/roast.py:220`

---

## Domain 4: Knowledge Graph & INDRA

### KG-1 — HIGH
**get_downstream_targets Cypher lacks LIMIT clause**
**File**: `cogex.py:668`

Hub regulators (TP53, MYC) with min_evidence=1 can return 50k+ edges.
No pagination. Called in loop by `get_regulator_modules`.

### KG-2 — HIGH
**discover_regulators Cypher lacks server-side LIMIT**
**File**: `cogex.py:836`

Python-side truncation at line 861 happens AFTER full result materialization
on Neo4j server. For 5000-gene universe: 100k+ rows transferred then truncated.

### KG-3 — MEDIUM
**No retry backoff in _execute_query**
**File**: `cogex.py:542`

Immediate retry on connection failure. During Neo4j transient outages,
rapid retries worsen the situation.

### KG-4 — MEDIUM
**No context manager protocol**
**File**: `cogex.py:944`

No `__enter__`/`__exit__`. Multiple call sites lack `finally: client.close()`.

### KG-5 — MEDIUM
**Gene cache grows without bound**
**File**: `cogex.py:1004`

Plain dict, never evicted. 45k entries for typical proteomics study.
Prevents detecting HGNC ID updates across analyses.

### KG-6 — MEDIUM
**CURIE parsing crashes on malformed records**
**File**: `cogex.py:706`

`split(":", 1)` with no try/except. One bad row crashes entire query.

### KG-7 — MEDIUM
**Double-wrapped RuntimeError**
**File**: `cogex.py:726`

`_execute_query` wraps in RuntimeError. `get_downstream_targets` wraps again.
Original exception type buried two levels deep.

### KG-8 — MEDIUM
**condition.split('_') breaks with underscore metadata values**
**File**: `clique_validator.py:689`

"Early_Onset" → 3 parts for 2 stratification columns. Lossy join/split round-trip.

### KG-9 — MEDIUM
**discover_regulators signature breaks LSP**
**File**: `indra_source.py:116`

Extra `max_targets` parameter inserted between `min_targets` and `relationship_types`,
disrupting positional argument contract.

### KG-10 — LOW: Dead force_reconnect parameter
### KG-11 — LOW: CoGExClient not thread-safe (document)
### KG-12 — LOW: Credentials path logged at INFO
### KG-13 — LOW: Per-gene resolution spams INFO
### KG-14 — LOW: corr_cache unbounded
### KG-15 — INFO: norm_id=None placeholder

---

## Domain 5: CLI & Pipeline

### CLI-1 — HIGH
**Empty contrasts dict crashes with IndexError**
**File**: `validate_baselines.py:328`

### CLI-2 — HIGH
**contrasts=None flows into .values() call**
**File**: `differential.py:769`

### CLI-3 — MEDIUM: Checkpoint non-atomic write
### CLI-4 — MEDIUM: 7 non-atomic JSON writes in differential.py
### CLI-5 — MEDIUM: O(n²) sample alignment
### CLI-6 — MEDIUM: No guard on zero common samples
### CLI-7 — MEDIUM: Division by zero on empty ensembl_ids
### CLI-8 — MEDIUM: Empty target gene set runs full pipeline silently
### CLI-9 — MEDIUM: Checkpoint resume loses Phase 1 intermediate state (= VAL-3)
### CLI-10 — LOW: NaN in clique_genes crashes string split
### CLI-11 — LOW: Non-atomic writes in _analyze_core
### CLI-12 — LOW: Temp mmap permissions (= SEC-14)
### CLI-13 — LOW: Phase 2 bypasses SeedSequence (= VAL-6)
### CLI-14 — LOW: Raw tracebacks reach users
### CLI-15 — INFO: Default path evaluated at import time

---

## Domain 6: Validation Framework

### VAL-1 — HIGH
**Verdict returns "refuted" when Phase 3 was absent (not failed)**
**File**: `validation_report.py:299`

When Phase 1 passes but Phase 3 was never added, the logic falls through all
conditionals to the `else` branch. Summary incorrectly says "Neither enrichment
nor permutation reaches significance."

### VAL-2 — HIGH
**competitive_z_percentile docstring inverted**
**File**: `negative_controls.py:60`

Docstring: "100 = most enriched." Computation: `sum(valid >= target) / len(valid) * 100`.
Most enriched target → few controls beat it → percentile ≈ 0, not 100.

### VAL-3 — HIGH
**Checkpoint resume loses Phase 5 competitive z-scores**
**File**: `validate_baselines.py:392`

`protein_df` initialized to None, set only in Phase 1 block. On resume,
Phase 1 skipped → protein_df stays None → Phase 5 degrades silently.

### VAL-4 — MEDIUM: details dict built but never stored
### VAL-5 — MEDIUM: Categorical dummies before NaN filter
### VAL-6 — MEDIUM: Phase 2 seed not from SeedSequence
### VAL-7 — MEDIUM: Non-atomic checkpoint write
### VAL-8 — MEDIUM: Negative specificity ratio undefined
### VAL-9 — LOW: L matrix ordering unvalidated
### VAL-10 — LOW: Silent interaction permutation failures
### VAL-11 — LOW: n_covariate_params includes interactions
### VAL-12 — INFO: to_dict() crash on empty null_z_scores
### VAL-13 — INFO: "No supplementary phases ran" conflation

---

## Domain 7: GPU & Numerical Computing

### GPU-1 — HIGH
**fit_f_dist s0_sq diverges from R limma when d0=Inf**
**File**: `permutation_gpu.py:173`

Python: `s0_sq = exp(emean)` (geometric mean of adjusted variances).
R limma: `s20 = mean(x)` (arithmetic mean of raw variances).
Geometric mean < arithmetic mean for skewed distributions.

### GPU-2 — HIGH
**trigamma_inverse Newton formula differs from R limma**
**File**: `permutation_gpu.py:82`

R limma iterates on reciprocal trigamma (nearly linear, convex).
Python uses standard Newton on `trigamma(y) - x`.
Different convergence properties at boundaries. Asymptotic threshold
differs (1e6 vs 1e7).

### GPU-3 — HIGH
**Float32 catastrophic cancellation in OLS residuals**
**File**: `permutation_gpu.py:607`

`Y_mx - Y_pred` in float32 loses precision for well-fitting models.
For Y ≈ 10 (log2 intensity), residuals accurate to ~6 digits.
RSS from squared imprecise residuals can have 100%+ relative error.
Creates asymmetry: observed (good fit) has worse precision than
permuted null (poor fit).

### GPU-4 — MEDIUM: Satterthwaite df in float32 (no benefit for small matrices)
### GPU-5 — MEDIUM: Median polish float32 accumulation over iterations
### GPU-6 — MEDIUM: Global convergence in batched median polish
### GPU-7 — MEDIUM: Hard MLX requirement despite CPU paths
### GPU-8 — MEDIUM: Median df for EB (= STAT-CORE-12)
### GPU-9 — MEDIUM: Mutable OLSPrecomputedMatrices
### GPU-10 — MEDIUM: np.random.seed in validation function
### GPU-11 — LOW: Float32 int division precision
### GPU-12 — LOW: Hard-coded SE floor without warning
### GPU-13 — LOW: Deprecated binom_test preference order
### GPU-14 — LOW: Normal approx for small-df p-values (= STAT-CORE-6)
### GPU-15 — INFO: Temp file cleanup lacks SIGKILL handling

---

## Domain 8: Security

### SEC-9 — MEDIUM
**HTTP (not HTTPS) for GO annotation downloads**
**File**: `annotation_providers.py:265,277`

MitM can substitute malicious annotation file on shared infrastructure.

### SEC-10 — MEDIUM
**Neo4j URL and .env path logged at INFO**
**File**: `cogex.py:504,532`

### SEC-11 — LOW
**hash() cache keys non-deterministic across processes**
**Files**: `id_mapping.py:197`, `entity_resolver.py:133`

PYTHONHASHSEED randomization → cache never hits across restarts.
Orphaned cache files accumulate without bound.

### SEC-12 — LOW: Non-atomic checkpoint (= CLI-3)
### SEC-13 — LOW: ReDoS in user regex
### SEC-14 — LOW: Temp mmap permissions
### SEC-15 — LOW: Unpinned dependencies
### SEC-16 — INFO: Global np.random.seed remnants (= SET-TEST-5)
### SEC-17 — INFO: Symlink attack on cache dir
