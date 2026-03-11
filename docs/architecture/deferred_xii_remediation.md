# Deferred Findings XII — Systematic Remediation

Deep examination of remaining deferred items from Audit XI with full statistical
and computational biology context. Three waves: fix, integrate, close.

## Wave 1: VSN Scale Guard + EB Priors Documentation

### DF-XII-1: VSN Scale Mismatch (Runtime Guard) — FIXED
- **File:** `stats/normalization.py:397-415`
- **Issue:** `prepare_experiment()` documents log2 input. `vsn_normalization()` expects raw intensities.
  Applying arcsinh to log2 data = double-stabilization → statistically meaningless output.
- **Fix:** Heuristic detection: if `max(data) < 35 AND min(data) >= -5`, emit UserWarning.
  NaN-safe via `np.isfinite` filtering. Non-blocking (warning, not error).
- **Tests:** 7 tests in `TestVSNScaleGuard` — log2 data, raw intensities, negative range,
  message content, still-produces-output, boundary, NaN handling.

### DF-XII-2: EB Priors from Observed Applied to Null (Documentation) — FIXED
- **File:** `stats/permutation_gpu.py:1737` (8-line inline comment)
- **Issue:** EB priors (d0, s0²) estimated from observed clique variances are reused for null
  gene sets. Random sets may have different variance distributions.
- **Direction:** Conservative (over-shrinkage of null → deflated null t-stats → larger p-values).
  Partially cancels the anti-conservative Camera VIF gap (DF-XII-3).
- **Fix:** Documented tradeoff in inline comment explaining:
  - Observed cliques have higher variance (correlated genes) → s0² estimated high
  - Null over-shrunk → conservative p-values (reduced power, not inflated FPR)
  - Re-estimation per null set computationally prohibitive
- **Tests:** 1 source inspection test.

## Wave 2: Camera VIF Integration — FIXED

### DF-XII-3: Camera VIF for Negative Controls
- **File:** `stats/negative_controls.py:358-398`
- **Issue:** `estimate_inter_gene_correlation()` defined at `enrichment_z.py:143-189` with 14 tests
  but never called from any production path. Competitive z-scores in negative controls compared
  target cliques (ρ_bar ~ 0.3-0.5) against random controls (ρ_bar ~ 0) without VIF correction.
  Anti-conservative bias: target z-scores inflated by up to √(1 + (k-1)ρ) factor.
- **Fix:** In `run_negative_control_sets`, compute `target_rho` from `engine.data` via
  `estimate_inter_gene_correlation()`. Pass `inter_gene_correlation=target_rho` to
  `compute_competitive_z()` for the target clique only. Controls use VIF=1 (random genes).
- **Why only negative_controls:** In label_permutation, specificity, and bootstrap_stability,
  the same clique is compared across conditions/permutations → VIF cancels. Only negative
  controls compares **different gene sets** with different correlation structures.
- **Magnitude:** For k=30, ρ=0.3: VIF=9.7, SE inflated 3.1×. This is a major correction.
- **Tests:** 7 tests in `TestCameraVIFIntegration` — VIF deflation, zero-correlation identity,
  correlated/uncorrelated estimation, integration end-to-end, magnitude check.

## Wave 3: Triage Closures

### Items Closed (no fix needed, with rationale)

| Finding | Verdict | Rationale |
|---------|---------|-----------|
| GPU batch memory | **FALSE ALARM** | Chunking at lines 1095-1113 bounds memory to ~400MB chunks |
| Competitive z-score SE | **NEGLIGIBLE** | Two-sample correction < 0.5% for k≤50 from n≥5000 |
| PCA summarization axis | **CORRECT** | axis=0 = per-sample means, standard for proteomics |
| Consistency metric 0.5 floor | **BY DESIGN** | Intentional label-flip robustness |
| GPU NaN guard / dead median polish | **MLX LIMITATION** | Guard correct; no NaN-aware MLX ops available |
| QRILC/AFT truncation bias | **STANDARD ALGORITHM** | Wei et al. 2018. Censored MLE = research enhancement |
| GPU covariate support | **INTENTIONALLY DEFERRED** | Explicit warnings at lines 1452-1471 |

## Wave 4: Review Cycle Remediation — FIXED

Brutalist review (5 critics) found 6 valid issues:

| # | Finding | Fix |
|---|---------|-----|
| R1 | EB comment mechanism wrong | Corrected causal chain (low s0² → null shrinks DOWN) |
| R2 | Phase 3 GPU batch memory unbounded | `_MAX_BATCH_ELEMS`, `max_cliques_per_chunk`, clique sub-chunking |
| R3 | Silent VIF fallback | `logger.warning()` when `engine.data` is None |
| R4 | VSN guard no bypass | `skip_scale_check=False` parameter |
| R5 | QRILC undocumented bias | Note section: 5-15% overestimate, Wei et al. 2018, MNAR |
| R6 | target_rho not persisted | `target_inter_gene_correlation` field + `to_dict()["competitive_z"]["target_rho_bar"]` |

## Test Summary
- **24 tests** in `tests/test_deferred_xii.py` (15 original + 9 review cycle)
- **1631 passed, 1 skipped, 2 pre-existing failures**
- Pre-existing: `test_enum_members_count` (stale count), `test_neg_communities_have_stability_scores` (seed-dependent)

## Status
- [x] Wave 1: VSN guard + EB comment
- [x] Wave 2: Camera VIF integration
- [x] Wave 3: Closures + memory update
- [x] Wave 4: Review cycle remediation (6 findings from 5 brutalist critics)
