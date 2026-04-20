# Deferred Findings Remediation — Audit XI+

Systematic remediation of deferred findings from Audit XI and prior audits.

## Findings Addressed (7 fixes)

### 1. TransformProvenance Passthrough (Architecture)
- **Files:** `quality/outliers.py` (6 locations), `quality/imputation.py` (1 location)
- **Issue:** Quality transforms created new BioMatrix without forwarding input provenance, losing `is_log_transformed` tracking through `OutlierDetector → Imputer` pipeline
- **Fix:** Added `provenance=matrix.provenance` to all 7 BioMatrix constructor calls
- **Tests:** `test_outlier_detector_preserves_provenance`, `test_imputer_preserves_provenance`, `test_provenance_not_default_after_transform`

### 2. QM-XI-5: IMPUTED Flag Provenance Accuracy (Integrity)
- **File:** `quality/imputation.py:587-591`
- **Issue:** IMPUTED quality flag set for ALL outlier-flagged positions, including those whose values didn't change (already within clipping bounds)
- **Fix:** Compare original vs. new data with `np.isclose(equal_nan=True)`, only set IMPUTED where values actually differ
- **Tests:** `test_unchanged_values_not_flagged_imputed`
- **Cascading fix:** Updated `test_full_pipeline.py` assertions to expect `n_imputed <= n_outliers`

### 3. SexClassifier NaN Crash Guard (Robustness)
- **File:** `quality/sex_imputation.py`
- **Issue:** `StandardScaler` and sklearn classifiers crash on NaN data; proteomics data commonly has NaN
- **Fix:** Added `_impute_nan_for_sklearn()` function (column-median imputation) called at 4 entry points. Changed `_score_feature` to use `np.nanstd`
- **Tests:** `test_impute_nan_replaces_nans`, `test_impute_nan_no_copy_when_clean`, `test_impute_nan_all_nan_column`, `test_score_feature_handles_nan`

### 4. CliqueDefinition Immutability (Architecture)
- **File:** `stats/clique_analysis.py:64-103`
- **Issue:** `CliqueDefinition` was a mutable dataclass; external mutation of `protein_ids` list would break `PreparedCliqueExperiment` reproducibility
- **Fix:** Made `@dataclass(frozen=True)` with `__post_init__` converting `protein_ids` list to tuple
- **Tests:** `test_frozen`, `test_protein_ids_is_tuple`, `test_external_list_mutation_does_not_affect`, `test_hashable`

### 5. VSN Convergence Tolerance (Performance)
- **File:** `stats/normalization.py:532`
- **Issue:** `a_change` (offset parameter) used absolute tolerance on raw-intensity scale; for large intensities (>1e6), convergence might never be reached
- **Fix:** Changed to relative tolerance: `np.abs(a_new - a) / (np.abs(a) + 1e-10)`
- **Tests:** `test_convergence_with_large_intensities`

### 6. QRILC Sample-Minimum Robustness (Statistical)
- **File:** `stats/missing.py:441-447`
- **Issue:** `min_observed = np.min(observed)` vulnerable to single extreme low outlier (carry-over artifact) that poisons all imputed values in that sample
- **Fix:** Use `np.percentile(observed, 1)` when `len(observed) >= 10`, fall back to `np.min` for small samples
- **Tests:** `test_outlier_does_not_poison_imputation`

### 7. Medcouple Median-Equal Pairs (Correctness)
- **File:** `quality/outliers.py:506-515`
- **Issue:** When `xi == xj`, kernel is 0/0 → code skipped these pairs. Per Brys et al. (2004), they should be included as h=0
- **Fix:** Changed `continue` to `h_values.append(0.0)` for tied pairs
- **Tests:** `test_median_ties_contribute_zero`, `test_all_ties_at_median`

## Test Summary
- **18 new tests** in `tests/test_deferred_fixes.py`
- **1606 passed, 1 skipped, 2 pre-existing failures**
- Pre-existing: `test_vectorized_faster_large` (benchmark), `test_enum_members_count` (stale count)

## Remaining Deferred Items
See `memory/deferred_findings.md` for items not yet addressed (statistical methodology choices, performance optimizations, GPU covariate support).
