# Audit II Remediation Plan — 2026-02-27

**Audit type**: Brutalist adversarial review, 8 domain-specific agents
**Scope**: Full codebase (~35k LOC across `src/cliquefinder/`)
**Raw findings**: 116 across 8 domains
**Post-dedup unique**: ~101 findings (15 cross-domain duplicates)
**Prior audit**: 36 findings (35 fixed + 1 revised-invalid), Feb 26 2026

## Severity Distribution

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 1 | VSN GPU/CPU divergence with NaN |
| HIGH | 18 | limma formula deviations, ROAST stat errors, crash-on-edge-case, concordance logic |
| MEDIUM | ~42 | float32 precision, non-atomic writes, missing guards, resource leaks |
| LOW | ~28 | ddof=0 in diagnostics, deprecation, documentation |
| INFO | ~10 | dead code, defensive coding suggestions |

## Wave Structure Overview

| Wave | Priority | Theme | Finding Count | Estimated Files |
|------|----------|-------|---------------|-----------------|
| 1 | P0 | Statistical Formula Correctness | 10 | 5 |
| 2 | P0–P1 | Crash/Edge-Case Fixes + Concordance Logic | 12 | 8 |
| 3 | P1 | GPU Precision & Numerical Safety | 8 | 4 |
| 4 | P2 | Infrastructure & Resource Management | 14 | 12 |
| 5 | P3 | Architecture, API Contracts & Defensive Coding | 16 | 14 |
| 6 | P4 | Polish, Documentation & Low-Severity Cleanup | ~28 | misc |

---

## Wave 1: Statistical Formula Correctness (P0)

**Rationale**: These findings produce **wrong statistical results** — incorrect p-values,
biased test statistics, or divergence from the published reference algorithms (R limma,
ROAST/Wu et al. 2010). They must be fixed first because all downstream analyses
(method comparison, validation framework, bootstrap) inherit these errors.

**Scope**: 10 findings, ~5 files, ~400 lines changed

### Findings

| ID | File(s) | Description | Ref Algorithm |
|----|---------|-------------|---------------|
| STAT-CORE-1 | `normalization.py:601` | VSN MLX NaN reference: `mx.mean(mx.where(isnan,0,y))` ≠ `np.nanmean` — denominator includes NaN count | Huber et al. 2002 VSN |
| STAT-CORE-6 | `differential.py:1424` | When EB enabled but d0=inf, uses Normal instead of t(df_residual) — anti-conservative for small n | Smyth 2004 limma |
| GPU-1 | `permutation_gpu.py:173` | `fit_f_dist` returns `exp(emean)` for s0_sq when d0=inf; R limma uses `mean(sigma2)` | Smyth 2004 §4 |
| GPU-2 | `permutation_gpu.py:82` | `trigamma_inverse` Newton step differs from R limma's reciprocal formulation | limma `fitFDist.R` |
| GPU-8 | `differential.py:1389` | `fit_f_dist` called with `int(median(df))` not per-feature df array — discards heterogeneity | limma `squeezeVar` |
| SET-TEST-1 | `rotation.py:1394` | MEAN stat for MIXED alt uses \|z\| not signed z — non-standard test, inflated significance | Wu et al. 2010 ROAST |
| SET-TEST-2 | `rotation.py:1399` | FLOORMEAN for UP/DOWN omits floor (uses ReLU) — reduced power | Wu et al. 2010 §2.3 |
| SET-TEST-11 | `rotation.py:794` | Sample weights applied to X but not Y — invalidates weighted ROAST entirely | Langsrud 2005 |
| SET-TEST-7 | `rotation.py:1418` | MEAN50 selects top 50% by w*z not \|z\| — conflates weight with signal rank | limma `mroast` |
| STAT-CORE-10 | `correlation_tests.py:386` | Li-Ji M_eff uses `len(filtered_eigenvalues)` not original M — inflates effective test count | Li & Ji 2005 |

### Solution Approaches

#### STAT-CORE-1: VSN MLX NaN Reference Fix
**Approach**: Compute valid count per feature, sum only valid entries, divide correctly.
```python
valid_count = mx.sum(~mx.isnan(y), axis=1)
valid_sum = mx.sum(mx.where(mx.isnan(y), 0.0, y), axis=1)
ref = valid_sum / mx.maximum(valid_count, 1)
```
**Pitfall**: MLX may not support `~mx.isnan()` (bitwise NOT on boolean). Verify MLX API.
**Mitigation**: If unsupported, use `mx.sum(mx.where(mx.isnan(y), 0.0, 1.0), axis=1)` for count.
**Validation**: Compare GPU vs CPU VSN output on dataset with 20% MNAR missingness; require max |delta| < 1e-6 (float32 epsilon).

#### STAT-CORE-6 + GPU-1: EB d0=inf p-value and s0_sq
**Approach**: Two coordinated fixes:
1. `differential.py:1424`: Always use `t.sf(|t|, df_total)` regardless of d0. The normal approximation is never needed — `t(df)` converges to normal as df→∞, so using t() universally is correct and eliminates the branching.
2. `permutation_gpu.py:173`: When `evar_adjusted <= 0`, set `s0_sq = float(np.mean(sigma2_valid))` to match R limma's `mean(x)` path.
**Pitfall**: Changing p-value distribution affects ALL downstream FDR corrections. Existing test baselines may shift.
**Mitigation**: Re-derive expected test values from the corrected formula. Add a regression test comparing Python output against R limma on a reference dataset (10 features, 6 samples, known p-values).
**Pitfall 2**: The `d0=inf` path in `squeeze_var` returns original variances unchanged (line 234-237), so the corrected `s0_sq` doesn't actually flow into variance shrinkage. But it IS returned to callers and could be inspected/used.
**Mitigation**: Add a comment documenting that `s0_sq` is informational when `d0=inf`.

#### GPU-2: trigamma_inverse Newton Alignment
**Approach**: Port the exact R limma formulation:
```python
# R limma reciprocal formulation (converges faster, better curvature)
dif = tri * (1.0 - tri / x) / tri_deriv
y = y + dif
converged = abs(-dif / y) < tol
```
Also raise asymptotic threshold from `1e6` to `1e7` to match R.
**Pitfall**: The current implementation works for typical inputs. Changing could introduce regressions for edge cases we haven't tested.
**Mitigation**: Add comprehensive edge-case tests: x=1e-10, x=1e7, x=1e20, x=0.001. Compare each against R `trigammaInverse()` output.

#### GPU-8: Per-Feature df for EB Hyperparameters
**Approach**: Pass `df_valid` array directly to `fit_f_dist` instead of `int(np.median(df_valid))`.
`fit_f_dist` already supports array df (lines 151-154 compute `digamma(df_half)` and
`trigamma(df_half)` element-wise). This is a one-line call-site change.
**Pitfall**: Per-feature df increases sensitivity to outlier features with very small df (e.g., df=1). The median was acting as a robust estimator.
**Mitigation**: In `fit_f_dist`, clip `df` to `>= 2` before `digamma`/`trigamma` computation. df=1 produces `trigamma(0.5) = π²/2 ≈ 4.93` which is finite but large, causing the variance of `e` to be dominated by low-df features.
**Validation**: Compare d0 and s0_sq estimates with and without the fix on a dataset with heterogeneous missingness. Verify that the moderated t-statistics change by < 5% for most features.

#### SET-TEST-1, SET-TEST-2, SET-TEST-7: ROAST Set Statistic Corrections
**Approach**: Three coordinated fixes in `rotation.py`:
1. **MEAN-MIXED** (line 1394): Use signed `z`, test with `|null| >= |obs|` in p-value step.
   ```python
   # MIXED: two-sided signed mean
   return np.sum(w * z, axis=1) / A
   ```
   Then in `compute_rotation_pvalues`, detect MEAN+MIXED and use `|null| >= |obs|`.
2. **FLOORMEAN UP/DOWN** (line 1399): Apply floor before zeroing:
   ```python
   # UP: floor positive z, zero negative
   f = np.where(z > 0, np.maximum(z, floor), 0)
   ```
3. **MEAN50** (line 1418): Select by `|z|` not `w*z`:
   ```python
   top_h_indices = np.argsort(np.abs(z), axis=1)[:, ::-1][:, :h]
   ```
**Pitfall**: Changing set statistics changes ALL ROAST p-values. Existing results become non-reproducible.
**Mitigation**: Version the statistic definitions. Add `statistic_version` field to `RotationResult`. Document in CHANGELOG that ROAST statistics now match the limma reference implementation (Wu et al. 2010). Add a "compatibility mode" flag if backward compatibility is critical.
**Pitfall 2**: The MEAN-MIXED two-sided test requires modifying `compute_rotation_pvalues` to handle `|null| >= |obs|` for specific stat+alternative combinations. This adds branching complexity.
**Mitigation**: Refactor `compute_rotation_pvalues` to accept a `tail` parameter per statistic, computed at statistic definition time, not at p-value time.

#### SET-TEST-11: Sample Weights in Rotation
**Approach**: Store `W_sqrt` in `RotationPrecomputed` and apply to Y in `extract_gene_effects`:
```python
# In extract_gene_effects:
if precomputed.W_sqrt is not None:
    Y_weighted = (precomputed.W_sqrt @ Y.T).T
else:
    Y_weighted = Y
U = Y_weighted @ precomputed.Q2
```
**Pitfall**: `W_sqrt` is `(n_samples, n_samples)` diagonal. Storing the full matrix wastes memory.
**Mitigation**: Store only the diagonal vector, apply via broadcasting: `Y * w_sqrt[None, :]`.
**Pitfall 2**: If no one currently uses sample weights, this code path is untested.
**Mitigation**: Add explicit test with known weights, verify against R limma's `mroast(weights=...)`.

#### STAT-CORE-10: Li-Ji M_eff Formula
**Approach**: Use `M` (original test count, `len(eigenvalues_original)`) instead of `len(eigenvalues)` after filtering:
```python
M = len(eigenvalues_original)  # before filtering
eigenvalues = eigenvalues_original[eigenvalues_original > 1e-10]
lambda_var = np.var(eigenvalues)
m_eff = 1 + (M - 1) * (1 - lambda_var / M)
```
**Pitfall**: The filtered eigenvalues' variance is computed on a subset, but the formula expects variance over all M eigenvalues (treating filtered ones as zero). Need to decide whether `np.var` should be over filtered or original (zero-padded) eigenvalues.
**Mitigation**: Verify against Li & Ji 2005 original paper. The formula uses the sample variance of ALL eigenvalues, so pad to M with zeros before computing variance.

### Wave 1 Execution Plan

**Agents (3 parallel, isolated worktrees):**
- `limma-alignment`: STAT-CORE-6, GPU-1, GPU-2, GPU-8 — all R limma formula alignments
- `roast-stats`: SET-TEST-1, SET-TEST-2, SET-TEST-7, SET-TEST-11 — ROAST statistic corrections
- `vsn-meff`: STAT-CORE-1, STAT-CORE-10 — VSN NaN fix + Li-Ji correction

**Validation protocol**: Each agent must include a test that compares Python output against R limma reference values for at least one non-trivial input.

---

## Wave 2: Crash/Edge-Case Fixes + Concordance Logic (P0–P1)

**Rationale**: These findings cause **crashes on valid input** or produce **silently wrong
concordance/validation results**. They affect user-facing workflows and data integrity
of the method comparison and validation frameworks.

**Scope**: 12 findings, ~8 files, ~300 lines changed

### Findings

| ID | File(s) | Description |
|----|---------|-------------|
| STAT-CORE-2 | `summarization.py:156` | `UnboundLocalError` when `max_iter=0` — `iteration` never bound |
| STAT-CORE-3 | `missing.py:328` | AFT uses ddof=0 → 18% sigma underestimate at n=3, inflates t-stats |
| STAT-CORE-4 | `missing.py:238` | `impute_min_value` crashes on all-NaN input |
| SET-TEST-3 | `rotation.py:748` | NaN conditions → Q2/data dimension mismatch crash |
| SET-TEST-4 | `bootstrap_comparison.py:297` | Duplicate DataFrame indices from bootstrap with-replacement |
| MCOMP-1 | `concordance.py:562` | `robust_hits()` empty when ANY method fails (NaN poisons `.all()`) |
| MCOMP-2 | `concordance.py:401` | `wide_format()` includes invalid results (NaN/inf p-values) |
| VAL-1 | `validation_report.py:299` | "refuted" verdict when Phase 3 absent — should be "inconclusive" |
| VAL-2 | `negative_controls.py:60` | Docstring says "100 = most enriched" but 0 = most enriched |
| CLI-1 | `validate_baselines.py:328` | Empty contrasts dict → bare `IndexError` |
| CLI-2 | `differential.py:769` | `contrasts=None` → `AttributeError` on `.values()` |
| CLI-8 | `validate_baselines.py:341` | Empty target gene set runs entire pipeline silently |

### Solution Approaches

#### STAT-CORE-2: UnboundLocalError at max_iter=0
**Approach**: Initialize `iteration = -1` before the loop. Then `iterations = iteration + 1` correctly returns 0 when `max_iter=0`.
**Pitfall**: None — straightforward defensive initialization.

#### STAT-CORE-3: AFT ddof=0 Bias
**Approach**: Change `np.std(observed)` → `np.std(observed, ddof=1)` in `impute_aft_model` (line 328) and `impute_qrilc` global fallback (line 398). Also fix `np.nanstd` calls at the global fallback paths.
**Pitfall**: When `len(observed) == 1`, `np.std(observed, ddof=1)` returns `nan` (division by zero). Need a guard.
**Mitigation**: `sigma = np.std(observed, ddof=1) if len(observed) > 1 else np.nanstd(all_observed, ddof=1)`. The global fallback already requires `len(all_observed) > 1`.
**Validation**: Unit test: 3 observations with known std. Verify ddof=1 output matches `scipy.stats.tstd`.

#### STAT-CORE-4: All-NaN Crash
**Approach**: Early return with empty imputation result when `np.all(np.isnan(data))`.
**Pitfall**: Need to handle both `method="feature"` and `method="global"` and `method="sample"` paths.
**Mitigation**: Single guard at function entry, before method dispatch.

#### SET-TEST-3: NaN Conditions Dimension Mismatch
**Approach**: In `RotationTestEngine.fit()`, filter data and metadata to samples with valid conditions before calling `compute_rotation_matrices`:
```python
mask = np.isin(sample_conditions, conditions)
if not mask.all():
    self.data = self.data[:, mask]
    self.metadata = self.metadata[mask].reset_index(drop=True)
```
**Pitfall**: Mutating `self.data` and `self.metadata` in `fit()` changes the engine state. If `fit()` is called multiple times, the data shrinks each time.
**Mitigation**: Store original data separately: `self._original_data = data`, `self.data = data[:, mask]`. Or compute the mask in `fit()` without mutating.

#### SET-TEST-4: Bootstrap Duplicate Index
**Approach**: Reset index after bootstrap sampling: `bootstrap_meta.index = range(len(bootstrap_meta))`.
**Pitfall**: If downstream code uses the index for sample identification (e.g., matching back to original samples), resetting to integer index breaks that mapping.
**Mitigation**: Store original sample IDs as a column before resetting: `bootstrap_meta['original_sample_id'] = bootstrap_meta.index`.

#### MCOMP-1 + MCOMP-2: Concordance Wide-Format Fixes
**Approach**: Two fixes in `concordance.py`:
1. `wide_format()`: Filter by `r.is_valid` when building wide table.
2. `robust_hits()`: Only include `pval_cols` from `self.methods_run` (excludes failed methods).
**Pitfall**: Changing `wide_format()` to filter by `is_valid` changes the output for ALL consumers of this method. Some may want to see invalid results for diagnostic purposes.
**Mitigation**: Add `include_invalid: bool = False` parameter to `wide_format()`. Default to filtered; explicit opt-in for unfiltered.

#### VAL-1: Verdict Logic Gap
**Approach**: Add explicit branch for `gate_adjusted and not gate_permutation and perm is None`:
```python
elif gate_adjusted and not gate_permutation and perm is None:
    self.verdict = "inconclusive"
    self.summary = "Covariate-adjusted enrichment passes but label permutation was not run."
```
**Pitfall**: This changes the verdict for programmatic callers who construct partial reports. Need to verify no existing tests depend on the "refuted" verdict for this case.
**Mitigation**: Search tests for `verdict == "refuted"` assertions and verify they don't exercise this specific path.

#### VAL-2: Docstring Inversion
**Approach**: Fix docstring to match computation: "0 = most enriched, 100 = least enriched."
**Pitfall**: None — pure documentation fix. The computation and verdict logic are already correct.

#### CLI-1 + CLI-2 + CLI-8: Missing Input Guards
**Approach**: Three guard clauses:
1. `validate_baselines.py:328`: Check `if not contrasts:` → print error, return 1.
2. `differential.py:703-769`: Check `if contrasts is None:` → print error with available conditions, return 1.
3. `validate_baselines.py:341`: Check `if len(target_gene_ids) < 3:` → print error, return 1.
**Pitfall**: The `< 3` threshold in CLI-8 is arbitrary. A single-gene set IS statistically testable.
**Mitigation**: Use `< 2` as the hard minimum (need at least 2 for variance), warn at `< 5`.

### Wave 2 Execution Plan

**Agents (3 parallel, isolated worktrees):**
- `edge-crash`: STAT-CORE-2, STAT-CORE-3, STAT-CORE-4, SET-TEST-3 — input edge case crashes
- `concordance-fix`: MCOMP-1, MCOMP-2, VAL-1, VAL-2 — concordance and verdict logic
- `cli-guards`: CLI-1, CLI-2, CLI-8, SET-TEST-4 — input validation and bootstrap index

---

## Wave 3: GPU Precision & Numerical Safety (P1)

**Rationale**: Float32 intermediate computations create **systematic precision loss** that
can bias statistical results, particularly for well-fitting models where residuals are
small. These don't produce outright wrong results but create GPU/CPU divergence that
undermines reproducibility claims.

**Scope**: 8 findings, ~4 files, ~200 lines changed

### Findings

| ID | File(s) | Description |
|----|---------|-------------|
| GPU-3 | `permutation_gpu.py:607` | Float32 residual `Y - Y_pred` has catastrophic cancellation for well-fitting models |
| GPU-4 | `differential.py:402` | Satterthwaite df uses float32 for tiny covariance matrices (no GPU benefit) |
| GPU-5 | `permutation_gpu.py:789` | Median polish accumulates float32 rounding over 10 iterations |
| GPU-7 | `permutation_gpu.py:1328` | `run_permutation_test_gpu` hard-requires MLX despite CPU-capable code paths |
| STAT-CORE-5 | `differential.py:1598` | Deprecated `binom_test` (removed SciPy ≥1.12) |
| STAT-CORE-7 | `summarization.py:198` | `logsumexp` does not handle NaN — silent data loss |
| STAT-CORE-11 | `missing.py:344` | AFT draws `u=0` → `ppf(0)=-inf` — silent downstream corruption |
| STAT-CORE-15 | `differential.py:821` | Mixed model fallback df off-by-one (subtracts extra 1) |

### Solution Approaches

#### GPU-3: Float32 Catastrophic Cancellation in Residuals
**Approach**: Use the algebraic identity `RSS = Y'Y - β'X'Y` which avoids explicit residual computation and the catastrophic cancellation inherent in `Y - Ŷ`:
```python
# Precompute Y'Y on CPU in float64 (one-time, O(n_features))
YtY = np.sum(Y ** 2, axis=1)
# After GPU beta computation, compute RSS on CPU
beta_np = np.array(beta, dtype=np.float64)
XtY = Y @ matrices.X  # float64 on CPU
rss = YtY - np.sum(beta_np * XtY, axis=1)
rss = np.maximum(rss, 0.0)  # Guard against floating point negativity
```
**Pitfall**: The algebraic RSS identity `||Y||² - β'X'Y` can produce negative values due to floating-point arithmetic when R² ≈ 1. This is well-known in computational statistics.
**Mitigation**: Floor RSS at 0.0 and log a warning when negative RSS is encountered. In R limma, this case produces `sigma2 = 0` which is then shrunk toward `s0_sq` by EB.
**Pitfall 2**: Moving RSS to CPU defeats the purpose of GPU acceleration.
**Mitigation**: The GPU is used for the expensive `(X'X)^{-1}X'Y` computation (O(n_features × n_params²)). The RSS computation is O(n_features × n_samples) which is fast on CPU. The bottleneck is the matrix inverse, not the residual sum. Profile to confirm.

#### GPU-4: Satterthwaite Float32 for Small Matrices
**Approach**: Remove the MLX path entirely for `satterthwaite_df`. The covariance matrix is at most ~10×10 — GPU provides zero benefit and float32 loses precision in the quadratic form cancellation.
```python
V_c = float(contrast_vector @ cov_beta @ contrast_vector)  # Always float64
```
**Pitfall**: None — this is a pure simplification. The `cov_beta.size > 16` threshold was overly aggressive.

#### GPU-5: Median Polish Float32 Accumulation
**Approach**: Keep GPU for median computation (the expensive step) but accumulate effects on CPU in float64:
```python
row_med_np = np.array(row_medians, dtype=np.float64)
row_effects_np += row_med_np
residuals = residuals - row_medians  # Still float32 on GPU for next iteration
```
Final result combines float64 accumulators: `summary = overall_np + col_effects_np`.
**Pitfall**: Converting between MLX and NumPy every iteration adds overhead.
**Mitigation**: Only convert the accumulated effects (small arrays: n_proteins × n_samples). The residuals stay on GPU. Profile to confirm overhead is < 10% of total median polish time.

#### GPU-7: Hard MLX Requirement
**Approach**: Replace `raise ImportError` with a warning and `use_gpu = False` fallback. All internal functions (`batched_median_polish_gpu`, `batched_ols_contrast_test`) already have CPU paths.
**Pitfall**: The function name is `run_permutation_test_gpu` — the "gpu" suffix implies MLX is required.
**Mitigation**: Rename to `run_permutation_test_batched` and add a deprecation alias. Or keep the name but document that "gpu" means "GPU-accelerated when available, CPU fallback otherwise."

#### STAT-CORE-5: Deprecated binom_test
**Approach**: Remove the deprecated path entirely, use only `binomtest`:
```python
direction_pvalue = float(
    scipy_stats.binomtest(n_negative, n_targets, p=0.5, alternative='two-sided').pvalue
)
```
**Pitfall**: Requires SciPy ≥ 1.7.0. Check minimum SciPy version in `pyproject.toml`.
**Mitigation**: Current minimum is `scipy>=1.10.0` (already above 1.7), so this is safe.

#### STAT-CORE-7: logsumexp NaN Propagation
**Approach**: Mask NaN as -inf before logsumexp (since exp(-inf) = 0, contributing nothing to the sum):
```python
masked = np.where(np.isnan(feature_data), -np.inf, feature_data)
result = logsumexp(masked, axis=0)
all_nan_cols = np.all(np.isnan(feature_data), axis=0)
result[all_nan_cols] = np.nan  # Restore NaN for all-missing columns
```
**Pitfall**: If ALL features are NaN for a sample, `logsumexp([-inf, -inf, ...])` returns `-inf`, which is then overwritten to NaN by the guard. This is correct.

#### STAT-CORE-11: AFT -inf Draw
**Approach**: Use `np.finfo(np.float64).tiny` (~2.2e-308) as lower bound for uniform draw:
```python
u = rng.uniform(np.finfo(np.float64).tiny, phi_threshold, size=n_missing_feature)
```
**Pitfall**: `tiny` is extremely small; `norm.ppf(tiny)` ≈ -37.5, which is still a valid (extreme) imputation. This is acceptable — it's better than -inf.
**Mitigation**: Alternatively, clip the imputed values: `np.clip(imputed, mu - 10*sigma, mu + 10*sigma)`.

#### STAT-CORE-15: Mixed Model Fallback df
**Approach**: Remove the extra `-1` from the within-subject df computation:
```python
residual_df = max(n_groups - n_fixed, len(df) - n_groups)  # was len(df) - n_fixed - 1
```
**Pitfall**: The `-1` may have been intentionally accounting for the random effect variance parameter. Need to verify against SAS PROC MIXED documentation.
**Mitigation**: Compare with Satterthwaite df output for the same model on test data. The fallback should be close to (but not necessarily equal to) the Satterthwaite df.

### Wave 3 Execution Plan

**Agents (2 parallel, isolated worktrees):**
- `gpu-precision`: GPU-3, GPU-4, GPU-5, GPU-7 — all GPU float32 fixes
- `numerical-safety`: STAT-CORE-5, STAT-CORE-7, STAT-CORE-11, STAT-CORE-15 — edge case numerics

---

## Wave 4: Infrastructure & Resource Management (P2)

**Rationale**: These findings affect **operational reliability** — non-atomic writes that
corrupt checkpoints, unbounded queries that exhaust memory, missing input guards that
let invalid data flow through the pipeline, and legacy RNG that breaks reproducibility.

**Scope**: 14 findings, ~12 files, ~400 lines changed

### Findings

| ID | File(s) | Description |
|----|---------|-------------|
| CLI-3 | `validate_baselines.py:225` | Checkpoint write non-atomic (S-5 fix not applied) |
| CLI-4 | `differential.py:834+` | 7 non-atomic JSON writes |
| CLI-11 | `_analyze_core.py:1037` | Non-atomic JSON writes in analysis output |
| SET-TEST-5 | `clique_analysis.py:1237,1533`, `permutation_gpu.py:1127` | Legacy `np.random.seed()` (3 call sites) |
| VAL-3 | `validate_baselines.py:392` | Checkpoint resume loses protein_df → degraded Phase 5 |
| VAL-6 | `validate_baselines.py:492` | Phase 2 seed not derived from SeedSequence |
| KG-1 | `cogex.py:668` | `get_downstream_targets` Cypher has no LIMIT clause |
| KG-2 | `cogex.py:836` | `discover_regulators` Cypher has no server-side LIMIT |
| KG-3 | `cogex.py:542` | No retry backoff — hammers recovering Neo4j server |
| KG-4 | `cogex.py:944` | No context manager protocol for connection cleanup |
| CLI-6 | `validate_baselines.py:310` | No guard on zero common samples |
| CLI-7 | `_analyze_core.py:296` | Division by zero when `ensembl_ids` empty |
| SEC-9 | `annotation_providers.py:265,277` | HTTP (not HTTPS) for GO downloads |
| SEC-11 | `id_mapping.py:197`, `entity_resolver.py:133` | `hash()` cache keys non-deterministic across processes |

### Solution Approaches

#### CLI-3 + CLI-4 + CLI-11: Atomic Write Adoption (class fix)
**Approach**: Replace all `open(path, "w") + json.dump()` with `atomic_write_json()` from `utils/fileio.py`. Grep for `json.dump` across all CLI modules and replace.
**Pitfall**: `atomic_write_json` may not handle custom `default=` serializers (e.g., `default=str` used in `_analyze_core.py:1037`).
**Mitigation**: Check `atomic_write_json` signature. If it doesn't accept `default`, add the parameter. Or use `json.dumps(data, default=str)` then `atomic_write_text()`.

#### SET-TEST-5: Legacy np.random.seed Elimination (class fix)
**Approach**: Replace all `np.random.seed(x); np.random.choice(...)` with `rng = np.random.default_rng(x); rng.choice(...)`.
Three call sites:
1. `clique_analysis.py:1237` — `run_permutation_clique_test`
2. `clique_analysis.py:1533` — `run_matched_single_gene_comparison`
3. `permutation_gpu.py:1127` — `validate_ols_implementation`
**Pitfall**: `np.random.default_rng` uses PCG64, which produces different sequences than legacy MT19937. Any test that checks specific permutation indices will break.
**Mitigation**: Tests should verify statistical properties (p-value distribution under null), not specific permutation sequences.

#### VAL-3: Checkpoint Resume Protein_df Reconstruction
**Approach**: Re-run `run_protein_differential` when Phase 1 is skipped (checkpoint resume) to reconstruct `protein_df` for Phase 5:
```python
if "covariate_adjusted" in report.phases:
    print("PHASE 1: [SKIPPED — checkpoint]")
    try:
        protein_df = run_protein_differential(...)
    except Exception:
        protein_df = None
```
**Pitfall**: This re-computation adds ~30-60 seconds to resume time. For fast-resume scenarios this may be undesirable.
**Mitigation**: Alternatively, serialize `protein_df` to checkpoint JSON (it's a DataFrame → dict → JSON). On resume, deserialize. This avoids re-computation.
**Preferred approach**: Serialize to checkpoint. Add `protein_df_dict` key to checkpoint JSON, reconstruct via `pd.DataFrame(checkpoint["protein_df_dict"])`.

#### VAL-6: Phase 2 SeedSequence
**Approach**: Add Phase 2 to SeedSequence spawn: `(_ss_boot, _ss_p2, _ss_p3s, _ss_p3f, _ss_p4, _ss_p5) = _ss.spawn(6)`.
**Pitfall**: Changing spawn count from 5 to 6 changes ALL downstream seed sequences (Phase 3-5 seeds shift). This breaks checkpoint compatibility.
**Mitigation**: Insert Phase 2 at the END of the spawn to preserve existing seeds: `(_ss_boot, _ss_p3s, _ss_p3f, _ss_p4, _ss_p5, _ss_p2) = _ss.spawn(6)`. This preserves positions 0-4 and adds position 5.

#### KG-1 + KG-2: Cypher LIMIT Clauses
**Approach**: Add `max_results` parameter to both functions with sensible defaults:
- `get_downstream_targets(max_results=50_000)`: Add `LIMIT $max_results` to Cypher
- `discover_regulators`: Move `max_results` enforcement into the Cypher query per chunk (not just Python-side truncation)
**Pitfall**: A LIMIT on `discover_regulators` per-chunk could miss regulators that appear in later chunks. Need per-chunk budget tracking.
**Mitigation**: Track `remaining = max_results - len(all_results)` and pass as chunk LIMIT.

#### KG-3: Retry Backoff
**Approach**: Add exponential backoff with jitter to `_execute_query`:
```python
delay = (2 ** attempt) + random.uniform(0, 1)
time.sleep(delay)
```
**Pitfall**: `time.sleep` blocks the thread. In async contexts this would freeze the event loop.
**Mitigation**: The codebase is synchronous. Document that `CoGExClient` is not async-safe.

#### KG-4: Context Manager Protocol
**Approach**: Add `__enter__`/`__exit__` to `CoGExClient`.
**Pitfall**: Existing call sites don't use `with` blocks. Adding the protocol doesn't fix existing resource leaks.
**Mitigation**: Add the protocol AND update the most critical call sites (`clique_analysis.py:744`, `_analyze_core.py:730`).

#### CLI-6 + CLI-7: Input Guards
**Approach**: Guard clauses at pipeline entry points for zero common samples and empty ID lists.
**Pitfall**: None — defensive guards with clear error messages.

#### SEC-9: HTTPS for GO Downloads
**Approach**: Change `http://` → `https://` for both GO URLs. Both servers support HTTPS.
**Pitfall**: Some institutional proxies may not support HTTPS for these domains.
**Mitigation**: Add a `--no-ssl-verify` flag or document the requirement.

#### SEC-11: Deterministic Cache Keys
**Approach**: Replace `hash(tuple(sorted(ids)))` with `hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]`.
**Pitfall**: This invalidates all existing cache files (different hash values).
**Mitigation**: On first run with new code, caches miss and are rebuilt. Old orphans remain. Add a `--clear-cache` CLI flag. Document the migration.

### Wave 4 Execution Plan

**Agents (4 parallel, isolated worktrees):**
- `atomic-writes`: CLI-3, CLI-4, CLI-11 — uniform atomic_write_json adoption
- `rng-seeds`: SET-TEST-5, VAL-6 — legacy RNG elimination + SeedSequence completeness
- `neo4j-hardening`: KG-1, KG-2, KG-3, KG-4 — query limits, backoff, context manager
- `pipeline-guards`: VAL-3, CLI-6, CLI-7, SEC-9, SEC-11 — input guards, HTTPS, cache keys

---

## Wave 5: Architecture, API Contracts & Defensive Coding (P3)

**Rationale**: These findings improve **API correctness, type safety, and maintainability**.
They don't produce wrong results today but create fragility, inconsistency, or
maintenance hazards.

**Scope**: 16 findings, ~14 files, ~500 lines changed

### Findings

| ID | File(s) | Description |
|----|---------|-------------|
| MCOMP-6 | `concordance.py` | `MethodComparisonResult` not frozen — mutable contract violation |
| MCOMP-3 | `methods/permutation.py:157` | Permutation adapter clamps p-value before storing (data fidelity loss) |
| MCOMP-4 | `methods/permutation.py:168` | O(N*M) clique lookup — build dict instead |
| MCOMP-5 | `methods/permutation.py:179` | `n_proteins_found` uses wrong lookup domain |
| MCOMP-7 | `methods/permutation.py:128` | Mutable metadata passed to permutation engine |
| GPU-9 | `permutation_gpu.py:253` | `OLSPrecomputedMatrices` mutated after construction |
| KG-5 | `cogex.py:1004` | Gene cache grows without bound |
| KG-6 | `cogex.py:706` | CURIE parsing crashes on malformed records |
| KG-7 | `cogex.py:726` | Double-wrapped RuntimeError obscures exception types |
| KG-8 | `clique_validator.py:689` | `condition.split('_')` breaks on underscore metadata |
| KG-9 | `indra_source.py:116` | `discover_regulators` signature breaks LSP |
| SET-TEST-6 | `rotation.py:1212` | df>100 threshold too low for t→z approximation |
| SET-TEST-8 | `rotation.py:2059` | Independent rotations per gene set prevent FWER correction |
| SET-TEST-12 | `rotation.py:984` | GPU rotation normalization in float32 |
| STAT-CORE-8 | `differential.py:1347` | OLS formula with `.T` on pinv (asymmetric) |
| VAL-4 | `validation_report.py:148` | `details` dict built but never stored/used |

### Solution Approaches (abbreviated — lower priority)

#### MCOMP-6: Freeze MethodComparisonResult
Use `@dataclass(frozen=True)` + `MappingProxyType` in `__post_init__`, matching `UnifiedCliqueResult` pattern.

#### MCOMP-3: Preserve Original p-value
Separate z-score clamping from stored p-value. Store `perm_result.empirical_pvalue` as-is.

#### MCOMP-4: Clique Lookup Dict
Build `{clique_id: clique_def}` dict once before loop. O(N+M) instead of O(N*M).

#### MCOMP-5: Consistent n_proteins_found
Use `experiment.clique_to_feature_indices` for protein counting consistency.

#### MCOMP-7: Copy Metadata
Pass `experiment.sample_metadata.copy()` to permutation engine.

#### GPU-9: Freeze OLSPrecomputedMatrices
Use `dataclasses.replace()` for EB prior assignment instead of mutation.

#### KG-5: Bounded Gene Cache
Use `functools.lru_cache(maxsize=50_000)` or add `clear_cache()` method.

#### KG-6: Defensive CURIE Parsing
Wrap parsing in try/except, skip malformed records with warning.

#### KG-7: Simplified Exception Wrapping
Remove outer try/except in `get_downstream_targets`. Let `_execute_query` errors propagate.

#### KG-8: Condition Delimiter
Replace `_` join/split with `||` delimiter, or use tuple keys internally.

#### KG-9: LSP Fix
Move `max_targets` parameter after `min_evidence` to preserve positional arg contract.

#### SET-TEST-6: Raise t→z Threshold
Change `df > 100` to `df > 1000` (max error < 0.1% for |t|=5).

#### SET-TEST-8: Shared Rotations for FWER
Pre-generate rotation vectors once in `test_gene_sets`, pass to each `test_gene_set` call.

#### SET-TEST-12: Rotation Normalization in Float64
Generate and normalize rotation vectors on CPU in float64. Convert to GPU only for matrix multiply.

#### STAT-CORE-8: OLS Formula with pinv
Use standard `(XtX_inv @ X' @ Y')'` formulation for safety with pinv.

#### VAL-4: Store Details Dict
Assign `self.phase_details = details` and include in `to_dict()`.

### Wave 5 Execution Plan

**Agents (4 parallel, isolated worktrees):**
- `mcomp-cleanup`: MCOMP-3, MCOMP-4, MCOMP-5, MCOMP-6, MCOMP-7 — method comparison API
- `kg-hardening`: KG-5, KG-6, KG-7, KG-8, KG-9 — knowledge graph defensive coding
- `rotation-precision`: SET-TEST-6, SET-TEST-8, SET-TEST-12 — rotation framework precision
- `misc-arch`: GPU-9, STAT-CORE-8, VAL-4 — miscellaneous architecture fixes

---

## Wave 6: Polish, Documentation & Low-Severity Cleanup (P4)

**Rationale**: These findings are **cosmetic, diagnostic, or defensive coding improvements**
that don't affect correctness but improve code quality and maintainability.

**Scope**: ~28 findings across all domains

### Findings — Grouped by Type

#### Population std (ddof=0) in Diagnostics — 5 instances
| ID | File:Line | Context |
|----|-----------|---------|
| STAT-CORE-14 | `missing.py:398` | QRILC global fallback |
| STAT-CORE-16 | `normalization.py:746` | CV computation |
| STAT-CORE-17 | `summarization.py:253` | CliqueSummary.to_dict() |
| SET-TEST-13 | `permutation_framework.py:218` | PermutationResult z-score |
| SET-TEST-16 | `permutation_gpu.py:104` | fit_f_dist heterogeneous df weighting |

**Approach**: Batch change `np.std(x)` → `np.std(x, ddof=1)` and `np.nanstd(x)` → `np.nanstd(x, ddof=1)` across all diagnostic contexts. Guard `len(x) > 1` before `ddof=1`.

#### Documentation Fixes — 4 instances
| ID | File:Line | Context |
|----|-----------|---------|
| MCOMP-8 | `concordance.py:570` | Misleading "dropna" comment |
| MCOMP-12 | `methods/roast.py:220` | Hardcoded "up" direction needs rationale comment |
| STAT-CORE-20 | `missing.py:267` | Dead `n_draws` parameter — either implement MI or remove |
| VAL-13 | `validation_report.py:250` | "No supplementary phases ran" conflation |

#### Defensive Coding — 7 instances
| ID | File:Line | Context |
|----|-----------|---------|
| MCOMP-9 | `methods/roast.py:141` | Read-only array passed to engine (defensive copy) |
| MCOMP-10 | `method_comparison.py:253` | Inconsistent key types (MethodName vs str) |
| MCOMP-11 | `methods/permutation.py:187` | Duplicate clique_id in null_df (add warning) |
| KG-10 | `cogex.py:516` | Dead `force_reconnect` parameter |
| KG-15 | `cogex.py:84` | `norm_id = None` → stub with clear error |
| VAL-12 | `label_permutation.py:80` | to_dict() crash on empty null_z_scores |
| SET-TEST-17 | `rotation.py:1998` | Single-gene sets return empty result |

#### Logging & Observability — 3 instances
| ID | File:Line | Context |
|----|-----------|---------|
| SEC-10 | `cogex.py:532` | Neo4j URL at INFO → DEBUG |
| KG-13 | `cogex.py:810` | Per-gene resolution spams INFO → aggregate + DEBUG |
| VAL-10 | `specificity.py:230` | Silent interaction permutation failures (add warning) |

#### Low-Priority Infrastructure — 7 instances
| ID | File:Line | Context |
|----|-----------|---------|
| KG-11 | `cogex.py:389` | CoGExClient not thread-safe (document) |
| KG-14 | `clique_validator.py:426` | corr_cache unbounded (add LRU bound) |
| SEC-13 | `formats.py:170`, `phenotype.py:195` | ReDoS in user regex (length limit) |
| SEC-14 | `_analyze_core.py:857` | Temp mmap file permissions (0o600) |
| SEC-15 | `pyproject.toml:22` | Unpinned dependency versions |
| SEC-17 | `annotation_providers.py:506` | Symlink attack on cache dir |
| CLI-14 | `__init__.py:69` | Raw tracebacks reach users |

#### Remaining LOW/INFO — 7 instances
| ID | File:Line | Context |
|----|-----------|---------|
| STAT-CORE-9 | `normalization.py:589` | VSN MLX init fragile (shared helper) |
| STAT-CORE-13 | `normalization.py:184` | Quantile norm NaN alignment (document limitation) |
| STAT-CORE-18 | `missing.py:156` | Censoring threshold at 0.1th percentile |
| STAT-CORE-19 | `correlation_tests.py:44` | Fisher Z clip at 0.9999 → tighter bound |
| SET-TEST-14 | `bootstrap_comparison.py:393` | Bootstrap denominator inconsistency |
| SET-TEST-15 | `rotation.py:1457` | MSQ UP/DOWN zeros opposite direction (document) |
| VAL-5 | `design_matrix.py:137` | Categorical dummies before NaN filter |
| VAL-8 | `specificity.py:376` | Negative specificity ratio undefined |
| VAL-9 | `design_matrix.py:283` | L matrix ordering assumption lacks validation |
| VAL-11 | `design_matrix.py:290` | n_covariate_params includes interaction columns |
| CLI-5 | `differential.py:572`, `validate_baselines.py:313` | O(n²) sample alignment |
| CLI-10 | `differential.py:1087` | NaN in clique_genes crashes split |
| CLI-15 | `differential.py:488` | Default path evaluated at import time |
| GPU-6 | `permutation_gpu.py:840` | Global convergence in batched median polish (document) |
| GPU-11 | `permutation_gpu.py:636` | Float32 int division (document) |
| GPU-12 | `permutation_gpu.py:662` | SE floor without warning |
| GPU-15 | `_analyze_core.py:855` | Temp file cleanup lacks SIGKILL handling |

### Wave 6 Execution Plan

**Agents (3 parallel, isolated worktrees):**
- `ddof-docs`: All ddof=0 fixes + documentation fixes — mechanical batch changes
- `defensive-logging`: Defensive coding fixes + logging level adjustments
- `infra-polish`: Low-priority infrastructure + remaining LOW/INFO items

---

## Execution Protocol

1. **Agent execution** with isolated worktrees per agent
2. **Manual review** between waves — verify mathematical correctness against reference
   implementations, run full test suite, check for regressions
3. **R limma validation** — Wave 1 requires comparison against R output for:
   - `fitFDist()` / `squeezeVar()` — GPU-1, GPU-2, GPU-8
   - `mroast()` set statistics — SET-TEST-1, SET-TEST-2, SET-TEST-7
4. **Document** — update this plan with completion status, capture new findings
5. **Structure next wave** — adjust scope based on manual review discoveries

## Cross-Wave Dependencies

```
Wave 1 (formulas) ─┐
                    ├─→ Wave 3 (GPU precision) ─→ Wave 5 (architecture)
Wave 2 (crashes)  ──┘                            ↗
                                Wave 4 (infra) ──┘
                                                  ↘
                                                    Wave 6 (polish)
```

Waves 1 and 2 are independent and can run in parallel.
Wave 3 depends on Wave 1 (formula fixes must land before precision fixes).
Wave 4 is independent of Waves 1-3 but should follow for clean test baselines.
Wave 5 depends on Waves 1-4 (API changes build on corrected implementations).
Wave 6 is lowest priority and can run any time after Wave 2.
