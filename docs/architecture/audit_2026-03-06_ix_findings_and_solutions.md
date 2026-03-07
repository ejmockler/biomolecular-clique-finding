# Audit IX — Findings and Solutions (2026-03-06)

## Overview
- **5 critic agents** (Claude) across: IO/data loading, GPU permutation, differential stats, clique analysis, integration/validation
- **~37 raw findings** → 20 rejected (already fixed, invalid, or too low priority), 15 valid actionable
- **15 fixes applied** across 3 cycles, **25 new tests**
- **1552 passed, 1 skipped, 3 pre-existing failures**

## Assessment Summary

| ID | Domain | Verdict | Description |
|----|--------|---------|-------------|
| IO-1 | Imputation | VALID | `_soft_clip_global/stratified` applied soft-clip to ALL values, not just outlier positions |
| IO-2 | Imputation | VALID | MAD=0 case skipped gene entirely, leaving outliers unmodified |
| IO-4 | Loaders | VALID | UTF-8 BOM in CSV files silently corrupts first column ID |
| IO-5 | Normalization | VALID | Buggy manual fractional rank interpolation in quantile normalization |
| IO-8 | Imputation | VALID | Quality flags dtype widened by bitwise OR with IntFlag |
| GPU-IX-1 | Permutation GPU | VALID | `df_for_pval` used `eb_df_total` (inf) when EB disabled |
| GPU-IX-2 | Permutation GPU | VALID | `mx.median` propagates NaN while CPU `np.nanmedian` ignores it |
| GPU-IX-3 | Permutation GPU | LOW | MLX float32 may accumulate rounding differently than CPU float64 |
| DIFF-IX-1 | Differential | VALID | Satterthwaite `df_residual = n_obs - n_params` wrong for mixed model |
| DIFF-IX-3 | Differential | VALID | MLX XtX in float32 can misrepresent condition numbers |
| CA-1 | Clique Analysis | VALID | `compare_protein_vs_clique_results` fails silently on ID-space mismatch |
| CA-2 | Clique Analysis | VALID | NaN in `n_indra_targets` column crashes `int()` conversion |
| CA-3 | Clique Analysis | VALID | ROAST functions use `clique.regulator` (can be None) as dict key |
| CA-4 | Clique Analysis | VALID | ZeroDivisionError when all permutation results filtered out |
| CA-7 | Clique Analysis | VALID LOW | Bare `except: pass` silently swallows failures |
| INT-3 | Validation | VALID | `frozen_fraction > 0.5` invalidates Phase 3 but verdict ignores it |

## Cycle 1: IO / Data Loading (5 fixes)

### IO-1: Soft-clip applied to non-outlier values
- **File**: `src/cliquefinder/quality/imputation.py` lines 906-914, 997-1004
- **Bug**: `_soft_clip_global` and `_soft_clip_stratified` applied `soft_clip()` to entire gene rows, modifying non-outlier values. The tanh compression distorted values that were within normal range.
- **Fix**: Changed to apply soft-clip ONLY to `outlier_cols = np.where(mask[gene_idx, :])[0]`, preserving non-outlier values exactly.

### IO-2: MAD=0 skips gene (outliers left intact)
- **File**: `src/cliquefinder/quality/imputation.py` lines 695-699
- **Bug**: When all clean values are identical (MAD=0), the code `continue`d, leaving outliers at their original extreme values.
- **Fix**: Clip outliers to the median value when MAD=0.

### IO-4: UTF-8 BOM corruption
- **File**: `src/cliquefinder/io/loaders.py` line 131
- **Bug**: `pd.read_csv(..., encoding='utf-8')` doesn't strip BOM (byte order mark), causing the first column name to have an invisible `\ufeff` prefix. This breaks downstream ID matching.
- **Fix**: Changed to `encoding='utf-8-sig'` which automatically strips BOM.

### IO-5: Quantile normalization fractional rank interpolation
- **File**: `src/cliquefinder/stats/normalization.py` lines 267-273
- **Bug**: Manual fractional rank interpolation used incorrect index arithmetic that could produce wrong values.
- **Fix**: Replaced with `np.interp(indices, np.arange(len(target_subset)), target_subset)`.

### IO-8: Quality flag dtype widening
- **File**: `src/cliquefinder/quality/imputation.py` line 589
- **Bug**: `new_flags[to_impute] | QualityFlag.IMPUTED` widened dtype from uint8 to int64 due to Python IntFlag.
- **Fix**: Added `.astype(new_flags.dtype)` to preserve original dtype.

## Cycle 2: GPU / Differential (4 fixes)

### GPU-IX-1: EB df infinity when moderation disabled
- **File**: `src/cliquefinder/stats/permutation_gpu.py` line 1647
- **Bug**: When `eb_moderation=False`, the code still used `matrices.eb_df_total` which could be `inf`. This made all p-values 0 (t-distribution with inf df).
- **Fix**: `df_for_pval = matrices.eb_df_total if eb_moderation else float(matrices.df_residual)`

### GPU-IX-2: NaN divergence between GPU/CPU median polish
- **File**: `src/cliquefinder/stats/permutation_gpu.py` lines 848-858
- **Bug**: `mx.median` propagates NaN while `np.nanmedian` ignores them, producing different results on same data.
- **Fix**: Check for NaN before GPU path; fall back to CPU with warning if NaN found.

### DIFF-IX-1: Satterthwaite df decomposition error
- **File**: `src/cliquefinder/stats/differential.py` lines 461-462
- **Bug**: `df_residual = n_obs - n_params` should be `n_obs - n_groups` (within-group residual df). `df_random = n_groups - 1` should be `n_groups - n_params` (between-group df after fixed effects).
- **Fix**: Corrected to `df_residual = max(n_obs - n_groups, 1)` and `df_random = max(n_groups - n_params, 1)`.

### DIFF-IX-3: Float32 condition number check
- **File**: `src/cliquefinder/stats/differential.py` line 601
- **Bug**: `np.array(XtX_mx)` from MLX preserves float32, which can misrepresent condition numbers for near-singular matrices.
- **Fix**: Added explicit `dtype=np.float64` cast.

## Cycle 3: Clique Analysis / Integration (7 fixes)

### CA-1: ID-space mismatch in protein comparison
- **File**: `src/cliquefinder/stats/clique_analysis.py` lines 1082-1097
- **Bug**: `compare_protein_vs_clique_results` uses `protein_df['feature_id'].isin(proteins)` but clique `proteins` are gene symbols while protein results may use UniProt IDs.
- **Fix**: Added diagnostic logging when zero matches found despite non-empty protein list.

### CA-2: NaN n_indra_targets crash
- **File**: `src/cliquefinder/stats/clique_analysis.py` line 378
- **Bug**: `int(group['n_indra_targets'].iloc[0])` crashes when value is NaN.
- **Fix**: Added `pd.notna()` guard: `int(...) if ('n_indra_targets' in group.columns and pd.notna(...)) else None`.

### CA-3: None regulator as dict key
- **File**: `src/cliquefinder/stats/clique_analysis.py` lines 1795, 1983
- **Bug**: `clique_gene_symbols[clique.regulator]` uses `clique.regulator` which can be None (data-driven cliques), causing dict key collisions.
- **Fix**: Changed to `clique.clique_id` which is always set and unique.

### CA-4: ZeroDivisionError in significance rate
- **File**: `src/cliquefinder/stats/clique_analysis.py` line 1472
- **Bug**: `100 * n_significant / len(permutation_results)` crashes when `permutation_results` is empty.
- **Fix**: Guard with `if len(permutation_results) > 0`.

### CA-7: Bare except swallows errors
- **File**: `src/cliquefinder/stats/clique_analysis.py` lines 1346-1347
- **Bug**: `except Exception: pass` silently swallows all errors in permutation inner loop.
- **Fix**: Changed to `logger.debug(...)` with `exc_info=True`.

### INT-3: frozen_fraction not used in verdict
- **File**: `src/cliquefinder/stats/validation_report.py` lines 196-207
- **Bug**: When `frozen_fraction > 0.5` (most samples in degenerate strata), the Phase 3 stratified permutation test is unreliable, but the verdict treated it as valid.
- **Fix**: If `frozen_fraction > 0.5` and no passing free permutation, downgrade `gate_permutation` to False with diagnostic annotation.
