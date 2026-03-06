# Audit IV — Findings, Solutions, and Pitfalls

**Date:** 2026-03-05
**Critics:** Claude, Gemini (via brutalist MCP)
**Domains reviewed:** Statistics, Architecture, Knowledge Graph, Data Integrity, Concordance
**Verification method:** Every finding below was cross-referenced against actual codebase at commit `712a4e0`

---

## Table of Contents

1. [Statistical Correctness](#1-statistical-correctness)
2. [Knowledge Graph Integration](#2-knowledge-graph-integration)
3. [Data Integrity & Caching](#3-data-integrity--caching)
4. [Concordance & Effect Size Coherence](#4-concordance--effect-size-coherence)
5. [Rotation Engine State Management](#5-rotation-engine-state-management)
6. [Validation Pipeline](#6-validation-pipeline)

---

## 1. Statistical Correctness

### 1.1 EB d0=inf Skips Shrinkage Instead of Using Prior Variance

**Finding ID:** STAT-IV-1
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:1013, 1261, 1349`

**The problem:**
```python
if eb_d0 is not None and eb_s0_sq is not None and not np.isinf(eb_d0):
    moderated_variances = (eb_d0 * eb_s0_sq + df * sample_variances) / (eb_d0 + df)
```

When `fit_f_dist` returns `d0 = inf`, this means the prior is infinitely strong — all genes should use the global prior variance `s0_sq`. The posterior formula reduces to:

```
lim d0→∞: s²_post = (d0 × s0² + df × s²_gene) / (d0 + df) → s0²
```

The current code skips moderation entirely, falling back to raw sample variances — the exact opposite of what infinite prior degrees of freedom means.

**Why practical impact is MEDIUM not CRITICAL:** `d0 = inf` only occurs when `fit_f_dist` detects zero heterogeneity across gene variances (all genes have essentially the same variance). In this degenerate case, `s0_sq ≈ mean(s²_gene)`, so the moderated and unmoderated variances are nearly identical. The statistical error exists but the numerical error is small.

**Solution:**
Handle `d0 = inf` as the limiting case: use `s0_sq` directly.

```python
if eb_d0 is not None and eb_s0_sq is not None:
    if np.isinf(eb_d0):
        # d0=inf: prior dominates completely → all genes get prior variance
        moderated_variances = np.full_like(sample_variances, eb_s0_sq)
        df_total = np.inf  # or a large sentinel; downstream uses df_total for t-dist
    else:
        moderated_variances = (eb_d0 * eb_s0_sq + df * sample_variances) / (eb_d0 + df)
        df_total = eb_d0 + df
```

**Pitfall:** `df_total = inf` complicates downstream t-to-z conversion. With infinite df, the t-distribution converges to normal, so `scipy.stats.t.cdf(t, df=inf)` works correctly (returns `norm.cdf(t)`). Verify scipy handles `df=inf` gracefully before assuming this works.

**Approach out:** Test `scipy.stats.t.cdf(2.0, df=np.inf)` vs `scipy.stats.norm.cdf(2.0)`. If they differ, use a sentinel `df_total = 1e15` instead of `np.inf`.

---

### 1.2 NaN Genes Propagate Through Rotation Projection

**Finding ID:** STAT-IV-2
**Severity:** HIGH
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:989-1007`

**The problem:**
The zero-variance filter (VALID-III-2, Audit III) uses `np.nanvar(data, axis=1)` — genes with partial NaN (e.g., 1 missing out of 6 samples) pass this filter because `nanvar` ignores NaN and returns a finite variance.

When these genes reach the projection `U = Y_weighted @ Q2` (line 995), `NaN × Q2_ij = NaN`, producing an entire NaN row in U. This NaN row then:
1. Corrupts `rho_sq` (line 998) → NaN
2. Corrupts `sample_variances` (line 1007) → NaN
3. Gets passed to `fit_f_dist` for EB prior estimation → corrupts `s0_sq` and `d0` for ALL genes
4. Makes EB moderation wrong for every gene, not just the NaN gene

**Solution:**
Add a NaN-row filter alongside the zero-variance filter in `fit()`:

```python
nan_row_mask = np.any(np.isnan(self.data), axis=1)
if nan_row_mask.any():
    n_nan = int(nan_row_mask.sum())
    warnings.warn(
        f"Removing {n_nan}/{len(self.gene_ids)} genes with missing values "
        f"before rotation testing (NaN propagates through QR projection).",
        stacklevel=2,
    )
    keep_mask = ~nan_row_mask
    self.data = self.data[keep_mask]
    self.gene_ids = [g for g, keep in zip(self.gene_ids, keep_mask) if keep]
    self.gene_to_idx = {g: i for i, g in enumerate(self.gene_ids)}
```

**Pitfall:** Proteomics data can have 10-30% missingness. Removing all genes with any NaN could discard a large fraction of the data. This is acceptable for rotation testing (which needs complete-case data for the QR projection), but users should be warned about the extent of data loss.

**Approach out:** Log the percentage removed. If >50%, emit a stronger warning suggesting imputation before rotation testing. The user can impute upstream (e.g., via `BioMatrix` transforms) and re-run.

---

### 1.3 Null Distributions Contain Invalid-Rotation Statistics

**Finding ID:** STAT-IV-3
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:2264-2298`

**The problem:**
`compute_set_statistics` at line 2264 is called with the full `z_rot` array (including invalid rotations). The returned `null_stats` is stored in `RotationResult.null_distributions` at line 2298. While `compute_rotation_pvalues` correctly filters by `valid_mask` (line 1626-1627), users who inspect `result.null_distributions` directly see contaminated values.

**Solution:**
Filter `z_rot` by `valid_mask` before computing set statistics:

```python
z_rot_valid = z_rot[:, valid_mask] if valid_mask is not None else z_rot
null_stats = compute_set_statistics(
    z_rot_valid, weights=final_weights,
    statistics=config.statistics, alternatives=config.alternatives,
)
```

**Pitfall:** None — this is a clean filter. The valid_mask is already computed by this point.

---

### 1.4 Concordance Ranking: Documentation Says "Dense" but Implementation is Competition

**Finding ID:** STAT-IV-4
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/concordance.py:404-412`

**The problem:**
The comment says "Assign dense ranks (ties get same rank)" but the implementation uses `current_rank = i + 1` which is competition ranking (ranks 1, 1, 3 instead of 1, 1, 2 for two tied items).

Dense ranking: [1, 1, 2, 3]
Competition ranking: [1, 1, 3, 4]

**Solution:**
Either fix the implementation to match the documentation (dense) or fix the documentation to match the implementation (competition). Dense ranking is more intuitive for users:

```python
# True dense ranking
for i, cid in enumerate(sorted_cliques):
    key = ranking_keys[cid]
    if prev_key is not None and key != prev_key:
        current_rank += 1  # increment by 1, not jump to position
    ranks[cid] = current_rank
    prev_key = key
```

**Pitfall:** Changing ranking semantics could affect downstream consumers. Since this API is new (Audit III), there are no known downstream consumers beyond tests. Fix now before it becomes load-bearing.

---

## 2. Knowledge Graph Integration

### 2.1 Phosphorylation Statement Type Silently Dropped

**Finding ID:** KG-IV-1
**Severity:** HIGH
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/cogex.py:754-760, 966-971`

**The problem:**
The CLI advertises `--stmt-types phosphorylation` (via `STMT_TYPE_PRESETS`), but the downstream edge parsing only handles `ACTIVATION_TYPES` and `REPRESSION_TYPES`:

```python
if stmt_type in ACTIVATION_TYPES:
    reg_type = "activation"
elif stmt_type in REPRESSION_TYPES:
    reg_type = "repression"
else:
    continue  # Silently drops Phosphorylation!
```

Users who run `--stmt-types phosphorylation` get zero results with no error message. The Cypher query correctly fetches Phosphorylation edges, but they're all discarded during parsing.

There are two independent sites with this issue:
- `get_downstream_targets` (line 754-760): logs a warning
- `get_regulator_modules` (line 966-971): silently continues

**Solution:**
Add `PHOSPHORYLATION_TYPES` to the parsing logic. Since phosphorylation doesn't have a natural "activation" or "repression" direction, add a third regulation type:

```python
if stmt_type in ACTIVATION_TYPES:
    reg_type = "activation"
elif stmt_type in REPRESSION_TYPES:
    reg_type = "repression"
elif stmt_type in PHOSPHORYLATION_TYPES:
    reg_type = "phosphorylation"
else:
    logger.warning(f"Unknown statement type: {stmt_type}")
    continue
```

**Pitfall:** Downstream consumers (community detection, clique analysis) may not expect `reg_type="phosphorylation"`. The `INDRAEdge.regulation_type` is a string field — adding a new value won't break construction, but code that branches on `regulation_type == "activation"` or `"repression"` will skip phosphorylation edges unless explicitly handled.

**Approach out:** Audit all downstream consumers of `INDRAEdge.regulation_type`. The key consumer is `regulatory_coherence.py` which builds separate positive/negative correlation graphs. Phosphorylation edges should go into the positive graph by default (phosphorylation typically activates downstream signaling), with a configuration option to place them in neither (unsigned) if the user wants conservative interpretation.

---

### 2.2 Negative Community Bootstrap Stability Never Computed

**Finding ID:** KG-IV-2
**Severity:** MEDIUM
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:1148-1152`

**The problem:**
```python
bootstrap_stability_pos = {}
bootstrap_stability_neg = {}
if compute_bootstrap and self.config.n_bootstrap > 0:
    bootstrap_stability_pos = self.bootstrap_stability(filtered_genes, condition)
```

`bootstrap_stability_neg` is initialized as empty dict and never populated. All negative communities get `bootstrap_stability=None` in their `CommunityResult` (line 1195).

**Solution:**
Compute bootstrap stability for both positive and negative communities. The `bootstrap_stability` method needs a `sign` parameter:

```python
if compute_bootstrap and self.config.n_bootstrap > 0:
    logger.info("Computing bootstrap stability...")
    bootstrap_stability_pos = self.bootstrap_stability(
        filtered_genes, condition, correlation_sign=CorrelationSign.POSITIVE
    )
    bootstrap_stability_neg = self.bootstrap_stability(
        filtered_genes, condition, correlation_sign=CorrelationSign.NEGATIVE
    )
```

**Pitfall:** Doubles the bootstrap computation time. If negative communities are rarely used, this is wasted work.

**Approach out:** Only compute negative bootstrap if any negative communities survived the density filter. Check `len(comm_neg) > 0` before running the bootstrap:

```python
if compute_bootstrap and self.config.n_bootstrap > 0 and comm_neg:
    bootstrap_stability_neg = self.bootstrap_stability(...)
```

---

## 3. Data Integrity & Caching

### 3.1 Cache Key Uses 1% Data Sampling — Collision Risk

**Finding ID:** CACHE-IV-1
**Severity:** MEDIUM
**Convergence:** 3/3 critics (architecture, statistics, security), code-verified
**Location:** `src/cliquefinder/utils/correlation_matrix.py:130-143`

**The problem:**
```python
sample_size = max(1000, int(0.01 * n_features * n_samples))
np.random.seed(42)
indices = np.random.choice(n_features * n_samples, ...)
sample_data = flat_data[indices]
hasher.update(sample_data.tobytes())
return hasher.hexdigest()[:16]
```

Three compounding risks:
1. **1% sampling:** Two matrices differing only in the unsampled 99% get the same cache key. Common when the only difference is imputation of sparse values.
2. **Global RNG pollution:** `np.random.seed(42)` sets the global legacy RNG state, affecting any downstream code using `np.random.*`.
3. **Hash truncation:** 64-bit hash (16 hex chars) has birthday-bound collision at ~4 billion entries — fine in practice, but combined with 1% sampling makes the effective collision bound much lower.

**Solution:**
Hash the full data using a streaming approach. For a 10k × 100 matrix (float64), full data is ~8MB — hashing takes <10ms:

```python
rng = np.random.default_rng(42)  # Local RNG, no global state pollution

# For matrices under 100MB, hash everything
data_bytes = matrix.data.tobytes()
if len(data_bytes) <= 100_000_000:  # 100 MB
    hasher.update(data_bytes)
else:
    # For very large matrices, sample 10% with local RNG
    n_elements = matrix.data.size
    sample_size = max(10000, n_elements // 10)
    indices = rng.choice(n_elements, sample_size, replace=False)
    indices.sort()  # Sequential access for cache efficiency
    hasher.update(matrix.data.ravel()[indices].tobytes())

return hasher.hexdigest()[:32]  # 128-bit hash
```

**Pitfall:** Full-data hashing for very large matrices (60k genes × 600 samples = 288 MB float64) could take ~300ms. The 100 MB threshold handles this gracefully.

**Approach out:** Profile hashing time for the actual dataset sizes in use. If full hashing is always <500ms, remove the sampling path entirely and always hash everything.

---

### 3.2 Second `np.random.seed(42)` in Correlation Matrix

**Finding ID:** CACHE-IV-2
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/utils/correlation_matrix.py:1035`

**The problem:**
A second `np.random.seed(42)` call at line 1035 in a different function. Same global RNG pollution issue as CACHE-IV-1.

**Solution:**
Replace with `np.random.default_rng(42)` as part of the CACHE-IV-1 fix. Both sites should be fixed together.

---

## 4. Concordance & Effect Size Coherence

### 4.1 RMSE Across Methods Compares Incompatible Units

**Finding ID:** CONC-IV-1
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/stats/concordance.py:148-150`

**The problem:**
```python
eff_rmse = np.sqrt(np.mean((eff_a[valid_eff_mask] - eff_b[valid_eff_mask]) ** 2))
```

When comparing OLS (log2FC) vs ROAST (mean z-score) vs Permutation (observed t-stat), the RMSE is meaningless — it computes the root-mean-square difference between quantities in different units. A z-score of 3.0 minus a log2FC of 0.5 is not a meaningful quantity.

The `effect_size_type` field was added in Audit III (MT-III-3), but there's no guard preventing RMSE computation across incompatible types.

**Solution:**
Check `effect_size_type` before computing RMSE. If types differ, set `eff_rmse = np.nan` and log a warning:

```python
# Check effect size compatibility
types_a = {r.effect_size_type for r in results_a if r.effect_size_type}
types_b = {r.effect_size_type for r in results_b if r.effect_size_type}

if types_a and types_b and types_a != types_b:
    eff_rmse = np.nan  # Incomparable units
    logger.debug(
        "Skipping RMSE: effect size types differ (%s vs %s). "
        "Pearson r (rank association) is still valid.",
        types_a, types_b,
    )
else:
    eff_rmse = np.sqrt(np.mean((eff_a[valid_eff_mask] - eff_b[valid_eff_mask]) ** 2))
```

**Pitfall:** Need access to the original `UnifiedCliqueResult` objects to read `effect_size_type`, but `compute_pairwise_concordance` currently receives pre-extracted arrays. Need to thread the type information through.

**Approach out:** The simplest approach is to extract `effect_size_type` from the first valid result in each method's list, since all results from the same method have the same type. Pass this as a parameter to the concordance function.

---

### 4.2 `mean_spearman_rho` Aggregation Uses `np.mean` Not `np.nanmean`

**Finding ID:** CONC-IV-2
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/concordance.py` (aggregation in `MethodComparisonResult` construction)

**The problem:**
If any pairwise Spearman rho is NaN (e.g., a method failed or produced identical p-values), `np.mean` of an array containing NaN returns NaN, poisoning the aggregate metric.

**Solution:**
Use `np.nanmean` with a warning if any values are NaN. This is a one-line fix at the aggregation site.

---

## 5. Rotation Engine State Management

### 5.1 `fit()` Mutates Engine State Non-Idempotently

**Finding ID:** STATE-IV-1
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:1797-1878`

**The problem:**
`fit()` modifies `self.data`, `self.gene_ids`, and `self.gene_to_idx` in place (zero-variance filter at L1864-1878, NaN condition filter at L1855-1862). Calling `fit()` a second time would:
1. Filter already-filtered data
2. Potentially remove additional genes if data has changed
3. Leave the engine in an inconsistent state

**Solution:**
Guard against double-fit with a `_fitted` flag:

```python
def fit(self, ...):
    if hasattr(self, '_fitted') and self._fitted:
        raise RuntimeError(
            "RotationTestEngine.fit() has already been called. "
            "Create a new engine instance for different parameters."
        )
    # ... existing fit logic ...
    self._fitted = True
```

**Pitfall:** None — this is a simple guard. Users who genuinely need to re-fit with different parameters should create a new engine, which is the intended API.

---

### 5.2 Median Polish Convergence Flag Unused

**Finding ID:** STATE-IV-2
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/permutation_gpu.py` (batched median polish)

**The problem:**
The `converged` flag is set but never checked or propagated. If median polish fails to converge within `max_iter`, the function silently returns potentially unconverged results.

**Solution:**
Add a warning when convergence is not reached:

```python
if not converged:
    logger.warning(
        "Batched median polish did not converge in %d iterations "
        "(max adjustment %.2e > eps %.2e). Results may be approximate.",
        max_iter, max_adjustment, eps,
    )
```

**Pitfall:** None — this is purely informational. Median polish typically converges within 10 iterations for well-conditioned data.

---

## 6. Validation Pipeline

### 6.1 Mandatory Gate Failures Don't Abort Pipeline

**Finding ID:** VALID-IV-1
**Severity:** MEDIUM
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/cli/validate_baselines.py` (phase execution try/except blocks)

**The problem:**
Phases 1 and 3 are documented as "mandatory gates" in the validation framework. But all phase executions are wrapped in `try/except` with `continue`, so a Phase 1 crash lets the pipeline proceed through Phases 2-5 and compute a verdict. The `compute_verdict()` function would see Phase 1 as missing, which produces a misleading verdict.

**Solution:**
After catching a mandatory gate exception, mark the result as `FAIL` instead of missing, and abort remaining phases:

```python
mandatory_phases = {1, 3}

try:
    result_phase1 = run_phase_1(...)
except Exception as e:
    logger.error("Mandatory Phase 1 failed: %s", e)
    if phase_num in mandatory_phases:
        # Abort: mandatory gate failed
        return ValidationVerdict(
            overall="FAIL",
            reason=f"Mandatory Phase {phase_num} failed: {e}",
            phase_results=completed_results,
        )
```

**Pitfall:** Some Phase 1 failures are recoverable (e.g., insufficient covariates — just skip covariate adjustment). Only truly fatal errors (bad data dimensions, zero genes) should trigger abort.

**Approach out:** Distinguish between `ValueError` (data quality → abort) and `RuntimeWarning` (configuration → warn and continue without adjustment). Only abort on `ValueError` or `LinAlgError`.

---

## Appendix: Findings Assessed as Invalid or Overstated

| Finding | Assessment | Reason |
|---------|-----------|--------|
| "60k gene 28.8 GB correlation matrix" | **Overstated** | Target scale is 10k genes (~800 MB float64). Valid concern at 60k but not the design target. |
| "18% false confidence from source-inspection tests" | **Inflammatory** | Source-inspection tests serve as regression guards. Not behavioral, but catch pattern-removal regressions. |
| "8k regulators multiple testing explosion" | **Partially valid** | Each regulator is a biologically motivated hypothesis; cross-regulator FDR is legitimate concern in discover mode but analogy to "p-hacking" is misleading. Noted as future methodology improvement. |
| "TOCTOU in cache verification" | **Impractical** | Single-user CLI tool; theoretical race window for file modification between hash check and load. |
| "GPU/CPU ping-pong bottleneck" | **By design** | Precision-correctness tradeoff documented in STAT-III-1 (Audit III). |
| "NamedTemporaryFile permissions" | **Not actionable** | OS/umask dependent; standard Python pattern. |
| "BioMatrix returns numpy views" | **By design** | Performance optimization; mutation by stats module would be a caller bug, not a BioMatrix bug. |

---

## Remediation Status

**All 13 findings remediated across 2 implementation cycles.**
**Test suite: 1393 passed, 1 skipped, 0 failures** (1 pre-existing unrelated skip)

### Cycle 1: Statistical Correctness + Knowledge Graph + Cache Integrity
| Finding | Status | Tests |
|---------|--------|-------|
| STAT-IV-1: EB d0=inf bypass | COMPLETE | 5 tests |
| STAT-IV-2: NaN gene propagation | COMPLETE | 4 tests |
| KG-IV-1: Phosphorylation stmt_type | COMPLETE | 12 tests |
| CACHE-IV-1+2: Full-data hashing + local RNG | COMPLETE | 13 tests |

### Cycle 2: Concordance + Low-Severity + Validation Pipeline
| Finding | Status | Tests |
|---------|--------|-------|
| CONC-IV-1: RMSE across incompatible units | COMPLETE | 4 tests |
| CONC-IV-2: mean_spearman_rho NaN poisoning | COMPLETE | 1 test |
| STAT-IV-3: Null distribution filtering | COMPLETE | 1 test |
| STAT-IV-4: Dense vs competition ranking | COMPLETE | 2 tests |
| STATE-IV-1: fit() non-idempotent | COMPLETE | 3 tests |
| STATE-IV-2: Median polish convergence warning | COMPLETE | (inline) |
| KG-IV-2: Negative bootstrap stability | COMPLETE | 8 tests |
| VALID-IV-1: Mandatory gate abort | COMPLETE | 7 tests |

### New Tests Added
| Test File | Count | Coverage |
|-----------|-------|----------|
| test_rotation.py (additions) | 13 | EB d0=inf, NaN filter, fit() guard, null dist filter |
| test_stmt_types.py (additions) | 12 | Phosphorylation parsing, reg_type mapping |
| test_cache_iv_fixes.py (new) | 13 | Cache key determinism, collision resistance, no RNG pollution |
| test_concordance_rank.py (additions) | 7 | Dense ranking, RMSE guard, nanmean |
| test_kg_iv2_bootstrap_neg.py (new) | 8 | Negative community bootstrap stability |
| test_valid_iv1_mandatory_gates.py (new) | 7 | Phase 1/3 abort, checkpoint persistence |
| **Total new tests** | **59** | |

### Surfaced Regressions (None)
No regressions detected. All existing 1334 tests continue to pass.|
