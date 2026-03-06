# Audit VII — Findings and Solutions

**Date:** 2026-03-06
**Critics:** 10 reports (Claude + Gemini) across 5 domains
**Raw findings:** ~50 → **12 valid actionable** after dedup and critical assessment

## Status: COMPLETE — 1506 passed, 1 skipped, 1 pre-existing failure, 37 new tests

---

## Cycle 1: Critical Statistical Fixes

### GPU-VII-1: Inline EB d0=inf returns wrong variance (CRITICAL)
- **File:** `src/cliquefinder/stats/permutation_gpu.py` lines 686-689 (GPU), 747-751 (CPU)
- **Problem:** `_batched_ols_gpu` and `_batched_ols_cpu` return `sigma2` (NO shrinkage) when `d0=inf`, but `squeeze_var()` at line 246 correctly returns `s0_sq` (MAX shrinkage). d0=inf means "prior dominates completely" per Smyth (2004). The inline code does the exact opposite.
- **Root cause:** squeeze_var was fixed in Audit VI but the inline copies in the batched OLS functions were not updated.
- **Fix:** Replace `s2_post = sigma2_np` with `s2_post = np.full_like(sigma2_np, matrices.eb_s0_sq)` in both GPU and CPU paths.
- **Pitfall:** Must also update the GPU MLX path if it has separate logic.
- **Tests:** Verify moderated t-stats match squeeze_var output for d0=inf case.
- **Status:** [x] DONE

### GPU-VII-2: eb_df_total wrong when d0=inf + verbose log inverted (HIGH)
- **File:** `src/cliquefinder/stats/permutation_gpu.py` line 1613, 1605
- **Problem:** `eb_df_total = d0 + df_residual if not np.isinf(d0) else float(df_residual)` — should be `np.inf` when d0=inf. Verbose log says "no shrinkage" for d0=inf — semantically inverted (d0=inf = MAX shrinkage).
- **Fix:** Set `eb_df_total = np.inf` when d0=inf. Change log to "d0=Inf (maximum shrinkage — prior dominates)".
- **Tests:** Verify eb_df_total is inf when d0=inf.
- **Status:** [x] DONE

### RC-VII-7: Permutation null is a no-op — column reorder is correlation-invariant (CRITICAL)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` lines 948-954
- **Problem:** `perm_idx = self._rng.permutation(sample_idx)` permutes column indices. `np.corrcoef(expr_data)` is computed on `self.matrix.data[np.ix_(gene_idx, perm_idx)]`. But correlation is invariant to column reordering — `np.corrcoef(X[:, perm]) == np.corrcoef(X)` for any permutation. The null distribution only reflects community detection stochasticity (from random seed), NOT the null hypothesis that gene-gene correlations are absent.
- **Correct null:** To destroy inter-gene correlation while preserving marginals, each gene's expression values must be shuffled INDEPENDENTLY (per-row permutation). This is the standard "gene-label permutation" null for co-expression networks.
- **Fix:** Replace `perm_idx = self._rng.permutation(sample_idx)` + single matrix indexing with per-gene independent permutation:
  ```python
  expr_data = self.matrix.data[np.ix_(gene_idx, sample_idx)].copy()
  for row in range(expr_data.shape[0]):
      self._rng.shuffle(expr_data[row])
  perm_corr = np.corrcoef(expr_data)
  ```
- **Pitfall:** The bootstrap_stability method (line 862) uses `self._rng.choice(sample_idx, ...)` for bootstrap resampling — this is CORRECT (resampling samples preserves within-sample correlation, which is what bootstrap should do). Only the permutation null needs per-gene shuffling.
- **Tests:** Verify null modularity distribution differs from observed; verify correlation matrix changes across permutations.
- **Status:** [x] DONE

### STAT-VII-1: mean50 MIXED returns signed mean — bidirectional sets get p≈1 (MEDIUM)
- **File:** `src/cliquefinder/stats/rotation.py` lines 1546-1549
- **Problem:** For MIXED alternative, `_compute_mean50_stat` returns `np.mean(selected_wz)` — a signed mean. A gene set with 5 genes at z=+3 and 5 at z=-3 yields mean50≈0. The `|null|>=|obs|` two-sided correction partially mitigates but `|mean(z)|` has much lower power than `mean(|z|)` for bidirectional signal. limma's mean50 uses absolute values before averaging for the mixed case.
- **Fix:** For MIXED alternative, use `np.mean(np.abs(selected_wz))` (or equivalently, take absolute z before selection and averaging). Keep UP/DOWN paths unchanged.
- **Pitfall:** The p-value comparison must change from `|null|>=|obs|` to `null>=obs` for mean50+MIXED since the statistic is now unsigned.
- **Tests:** Verify bidirectional gene set with balanced up/down gets low p-value with mean50+MIXED.
- **Status:** [x] DONE

---

## Cycle 2: Validation & Architecture Fixes

### VAL-VII-9: Phase 3 gate ignores free permutation when stratified fails (MEDIUM)
- **File:** `src/cliquefinder/stats/validation_report.py` lines 178-181
- **Problem:** When Phase 3 dict has `{"stratified": {"status": "failed"}, "free": {"permutation_pvalue": 0.001}}`, the code reads `strat.get("permutation_pvalue")` → None, falls back to `perm.get("permutation_pvalue", 1.0)` → 1.0 (no top-level key). Gate fails even though free permutation succeeded.
- **Fix:** After stratified lookup, also check free permutation as fallback:
  ```python
  if strat_p is None or strat_p == 1.0:
      free = perm.get("free", {})
      free_p = free.get("permutation_pvalue")
      if free_p is not None:
          strat_p = free_p  # Use free as fallback
  ```
- **Tests:** Test with stratified-failed + free-passed structure → gate should pass.
- **Status:** [x] DONE

### VAL-VII-4: Negative control gene sets can overlap with target genes (MEDIUM)
- **File:** `src/cliquefinder/stats/negative_controls.py` line ~287
- **Problem:** `rng.choice(all_gene_ids, size=target_size, replace=False)` draws from full universe including target genes. For target=50, universe=500, expected overlap=5 genes (10%).
- **Fix:** Exclude target gene IDs from sampling pool:
  ```python
  pool = [g for g in all_gene_ids if g not in target_set]
  genes = rng.choice(pool, size=min(target_size, len(pool)), replace=False)
  ```
- **Pitfall:** If pool < target_size after exclusion, sample with smaller size and log warning.
- **Tests:** Verify no overlap between control sets and target genes.
- **Status:** [x] DONE

### ARCH-VII-1: PreparedCliqueExperiment leaves sample_metadata mutable (HIGH)
- **File:** `src/cliquefinder/stats/experiment.py` line 99-111
- **Problem:** `__post_init__` wraps dicts in MappingProxyType and sets arrays read-only, but `sample_metadata` (pd.DataFrame) is untouched. `frozen=True` prevents reassignment but not in-place mutation.
- **Fix:** Deep-copy the DataFrame in `__post_init__` and store it. While we can't make a DataFrame truly immutable, copying prevents external references from mutating the experiment's data.
  ```python
  if isinstance(self.sample_metadata, pd.DataFrame):
      object.__setattr__(self, 'sample_metadata', self.sample_metadata.copy())
  ```
- **Pitfall:** pd.DataFrame has no `flags.writeable` equivalent. Copy is best available defense.
- **Tests:** Verify external DataFrame mutation doesn't affect experiment's copy.
- **Status:** [x] DONE

### STAT-VII-4: _construct_c_matrix uses Classical Gram-Schmidt (MEDIUM)
- **File:** `src/cliquefinder/stats/rotation.py` lines 456-478
- **Problem:** Docstring says "modified Gram-Schmidt" but implementation is Classical GS with a pre-step (subtract c_unit first, then subtract all previous in one pass). For p>10, CGS accumulates rounding errors.
- **Fix:** Implement true Modified Gram-Schmidt: subtract projections one at a time, re-reading the updated v each time.
  ```python
  for j in range(col_idx):
      v = v - np.dot(v, C[:, j]) * C[:, j]  # CGS: uses original v implicitly
  # MGS: update v after EACH projection
  for j in range(col_idx):
      proj = np.dot(v, C[:, j])
      v = v - proj * C[:, j]
  ```
- **Tests:** Verify orthogonality of C matrix columns for p=20 near-axis contrast.
- **Status:** [x] DONE

---

## Cycle 3: Minor Fixes

### RC-VII-1: Bootstrap matching allows double-counting communities (MEDIUM→LOW)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` lines 892-902
- **Problem:** Multiple original communities can match to the same bootstrap community, inflating stability scores. No exclusion tracking like `_match_communities()` has.
- **Fix:** Track matched bootstrap communities and exclude from subsequent matches.
- **Status:** [x] DONE

### RC-VII-5: Bootstrap/permutation hardcode Pearson correlation (MEDIUM→LOW)
- **File:** `regulatory_coherence.py` lines 863, 954
- **Problem:** Uses `np.corrcoef` (Pearson) regardless of what `compute_correlation_matrix` used. If user requested Spearman, the null/bootstrap uses a different correlation measure.
- **Fix:** Add `method` parameter threaded through to bootstrap and permutation, default to 'pearson'.
- **Status:** [x] DONE

### GPU-VII-3: Verbose log "no shrinkage" semantic inversion (LOW — covered by GPU-VII-2)
- Rolled into GPU-VII-2 fix.

### VAL-VII-2: Degenerate strata leak real signal in stratified permutation (MEDIUM)
- **File:** `src/cliquefinder/stats/label_permutation.py`
- **Problem:** Strata with only one condition value are preserved unchanged, leaking real signal into every null permutation.
- **Fix:** Detect single-condition strata and warn. Option: exclude those samples from the permutation test or fall back to free permutation.
- **Status:** [x] DONE

---

## Rejected Findings (with rationale)

| Finding | Reason |
|---|---|
| Gemini: "RSS algebraic identity is unstable" | Invalid — Y'Y - β'X'Y is the *standard* stable approach for GPU float32. CPU uses float64 where cancellation is negligible. Floor to 0 is a safety net. |
| Gemini: "rtol=1.0 validation" | That specific test is a rough GPU/CPU shape-matching sanity check, not the primary numerical validation. Other tests verify math precisely. |
| STAT-VII-6 (stdev.unscaled missing) | Reviewer self-corrected: p-values are invariant to multiplicative scaling since both observed and null use same formula. Active proportion threshold (√2) may differ from limma but is documented. |
| STAT-VII-7 (rotation not full vector) | Reviewer confirmed CORRECT: only first element and residual SS needed for t-stat. Matches limma optimization. |
| STAT-VII-5 (two-group path fragility) | Valid observation but the path is explicitly scoped to two-group designs with a clear ValueError for >2 conditions. Low risk of misuse. |
| STAT-VII-2 (floormean weight inconsistency) | Only triggers with signed weights, which are not used in this codebase. Documented as latent. |
| STAT-VII-3 (mean50 normalization) | p-values still valid (same formula for obs and null). Only affects relative power characteristics. |
| Gemini: "pseudoreplication-lite in subject aggregation" | This is the documented design choice — WLS would be better but the MSstats-inspired pipeline deliberately uses simple aggregation as a tractable approximation. |
| RC-VII-3 (null_std ddof=0) | Not used in any downstream computation. Property only for reporting. |
| RC-VII-9 (2-partition modularity ≠ community contribution) | Correct observation but the metric is internally consistent (used for both observed and comparisons). Rename might help but not a bug. |
| VAL-VII-1 (Phase 2 can downgrade verdict) | Correct by design — "inconclusive" specificity with no other supplementary evidence is genuinely inconclusive. The docstring could be clearer but the logic is intentional. |
| STAT-VII-10 (GPU float32 EB precision) | Bounded error for typical EB parameters. Already documented in precision_note. |
| STAT-VII-12 (dense diagonal weight matrix) | Performance only, not correctness. O(n²) vs O(n) for typically small n. |

---

## Implementation Plan

### Cycle 1: Critical Statistical (4 findings)
Files: `permutation_gpu.py`, `regulatory_coherence.py`, `rotation.py`
- GPU-VII-1 + GPU-VII-2: Fix inline EB d0=inf + eb_df_total + verbose log
- RC-VII-7: Fix permutation null to use per-gene shuffling
- STAT-VII-1: Fix mean50 MIXED to use absolute z-scores

### Cycle 2: Validation & Architecture (4 findings)
Files: `validation_report.py`, `negative_controls.py`, `experiment.py`, `rotation.py`
- VAL-VII-9: Phase 3 free permutation fallback
- VAL-VII-4: Exclude target genes from negative control sampling
- ARCH-VII-1: Deep-copy sample_metadata in PreparedCliqueExperiment
- STAT-VII-4: Modified Gram-Schmidt in _construct_c_matrix

### Cycle 3: Minor (4 findings)
Files: `regulatory_coherence.py`, `label_permutation.py`
- RC-VII-1: Bootstrap matching exclusion tracking
- RC-VII-5: Thread correlation method to bootstrap/permutation
- VAL-VII-2: Degenerate strata detection
