# Audit XI — Findings and Solutions

**Date:** 2026-03-06
**Critics:** 10 (5 Claude + 5 Gemini) via brutalist MCP across 5 domains
**Domains:** Stats methods, quality/imputation, GPU permutation, negative controls, normalization

## Summary

- **4 deferred findings confirmed** (H-1, H-3, H-4, M-2) — all independently verified by multiple critics
- **4 novel findings fixed** (NORM-XI-2, QM-XI-3, XI-4, NC-XI-2)
- **8 total fixes**, 13 new tests
- **1577 passed, 1 skipped, 2 pre-existing failures**

## Confirmed Deferred Findings

### H-4: Censored Quantile Normalization Maps to Wrong Quantiles
- **File:** `stats/normalization.py:237-245`
- **Severity:** HIGH
- **Bug:** Under MNAR, missing values are low-abundance so observed values should map to UPPER quantiles. Code used `target[:n_valid]` (lower quantiles) — exactly backwards.
- **Fix:** Changed to `target[-n_valid:]` (upper quantiles).

### H-1: Negative Controls Index Space Mismatch
- **File:** `stats/negative_controls.py:408-433`
- **Severity:** HIGH
- **Bug:** `gene_means`/`gene_variances` were computed from the caller-supplied `data` (unfiltered), but `engine.gene_to_idx` maps into the filtered index space after `_filter_degenerate_genes()`. Expression-matched controls were sampling from the wrong gene distributions.
- **Fix:** Use `engine.data` (filtered) for computing gene statistics, with `getattr(engine, "data", data)` fallback for test mocks.

### H-3: subject_col and use_mixed_model Silently Ignored
- **File:** `stats/permutation_gpu.py:1401,1404`
- **Severity:** HIGH
- **Bug:** GPU permutation path accepted `subject_col` and `use_mixed_model` but never used them. Users with repeated measures got simple OLS instead of subject-aggregated analysis.
- **Fix:** Added `UserWarning` when either parameter is set, directing users to the CPU path for repeated-measures support.

### M-2: use_eb Flag in RotationTestConfig is Dead Code
- **File:** `stats/rotation.py:1749`
- **Severity:** MEDIUM
- **Bug:** `RotationTestConfig.use_eb` was accepted but never checked — EB moderation always applied during `fit()`.
- **Fix:** Deprecated the flag with `DeprecationWarning` when `use_eb=False` is passed. Updated docstring to document that EB is always applied (integral to ROAST methodology).

## Novel Findings

### NORM-XI-2: Negative CV When Medians Are Negative
- **File:** `stats/normalization.py:763-772`
- **Severity:** MEDIUM
- **Bug:** `assess_normalization_quality` computed CV as `std/mean` but for log-transformed data with negative medians, the mean is negative, producing a negative CV.
- **Fix:** Changed denominator to `np.abs(np.mean(...))`.

### QM-XI-3: Zero-IQR Adjusted Boxplot Flags Everything
- **File:** `quality/outliers.py:610-613`
- **Severity:** MEDIUM
- **Bug:** When IQR=0, fences collapsed to `median ± 1e-10`, flagging nearly every value as an outlier.
- **Fix:** Return `(-inf, inf)` when IQR=0 — no meaningful outlier fences can be defined.

### XI-4: iloc with Boolean Mask (Pandas Deprecation)
- **File:** `stats/permutation_gpu.py:1550`
- **Severity:** MEDIUM
- **Bug:** `sample_metadata.iloc[contrast_mask]` where `contrast_mask` is a boolean array — deprecated in pandas 2.x.
- **Fix:** Changed to `.loc[contrast_mask]`.

### NC-XI-2: NaN Cost Matrix from nanvar(ddof=1)
- **File:** `stats/negative_controls.py:409`
- **Severity:** MEDIUM
- **Bug:** `np.nanvar(data, axis=1, ddof=1)` produces NaN for genes with only 1 valid observation, poisoning the Hungarian algorithm cost matrix.
- **Fix:** Replace NaN variances with 0.0 after computation.

## Rejected Findings (from critics)

| Claim | Verdict | Reason |
|---|---|---|
| GPU OLS `Y @ X` wrong matrix algebra | INVALID | `Y @ X` correctly computes `X'y_i` for batched Y. Math verified. |
| mean50 weight normalization | INVALID | Self-consistent between observed/null — p-value valid. |
| bootstrap_comparison denominator mismatch | INVALID | Documented as SET-TEST-14 intentional design. |
| frozen_fraction non-constant | INVALID | Depends only on fixed strata structure. |
| experiment_wide_fdr biased NaN exclusion | INVALID | Standard FDR practice — correct only for tests conducted. |
| Competitive z-score SE missing `1/(n_total-k)` term | INVALID | One-sample-vs-pool test, not two-sample. Negligible for k << n_total. |
| QRILC truncation bias | INVALID | Standard algorithm per Wei et al. (2018). Enhancement, not bug. |
| NaN propagation in compute_competitive_z | INVALID | Contract assumes finite inputs (degenerate genes filtered upstream). |
| PCA summarization axis=0 vs axis=1 | DEBATABLE | Both are valid choices for PCA imputation; current is defensible. |
| VSN on log data (scale mismatch) | LOW | Methodology issue; VSN docstring warns about raw intensities. |
| Non-monotonic target in simple quantile norm | INVALID | Standard algorithm (used by limma, preprocessCore). |
| Provenance black hole (TransformProvenance reset) | ARCHITECTURE | Valid observation but not a correctness bug. Deferred. |
| sample_metadata immutability gap | ARCHITECTURE | Valid but partial fix (data already immutable from Audit VIII). Deferred. |
| Unbounded GPU batch memory | PERFORMANCE | Valid but not a correctness bug. |
| float32 catastrophic cancellation | LOW | GPU path uses algebraic identity; concern valid for extreme R² but unlikely in practice. |
