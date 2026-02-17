# Test Failure Analysis: 26 Failures + 2 Collection Errors

**Date**: 2025-02-12
**Baseline**: 220 passing, 26 failed, 2 errors (1 collection + 1 fixture)
**Scope**: All failures are pre-existing; none introduced by statement-types or INDRA gene-sets work.

---

## Taxonomy of Root Causes

| ID | Category | Count | Severity | Root Cause Class |
|----|----------|-------|----------|------------------|
| F1 | MetadataRowFilter `regex=False` bug | 2 | **High** | Logic error — filter is a no-op |
| F2 | GPU batched OLS: float32 precision + EB type mixing | 4 | **Critical** | Numerical correctness |
| F3 | GPU batched OLS: API signature drift | 3 | Medium | Test–source contract |
| F4 | Imputer API: removed KNN/radius strategies | 9 | Low | Dead test code |
| F5 | Rotation: removed FDR-adjusted p-values | 2 | Low | Intentional simplification, stale tests |
| F6 | Stale CLI default assertions | 2 | Low | Test–source drift |
| F7 | Cross-modal mapper: case normalization gap | 1 | **High** | Logic error in ID resolution |
| F8 | CliqueValidator: empty gene list unhandled | 1 | Low | Missing guard clause |
| F9 | Satterthwaite df: numerical edge cases | 2 | Medium | Tolerance / approximation |
| F10 | Accidental test collection from prod code | 1 (error) | Low | Naming convention |
| F11 | `derive_genetic_phenotype` import error | 1 (error) | Low | Refactored to `cohort.py` |

---

## F1: MetadataRowFilter — Substring Matching is Broken (Critical Logic Bug)

**Files**: `src/cliquefinder/io/data_filters.py:78-114`, `src/cliquefinder/io/data_filters.py:116-145`
**Tests**: `test_phenotype_inference.py::test_basic_filtering`, `test_phenotype_inference.py::test_get_filtered_ids`

### Diagnosis

Both `filter()` and `get_filtered_ids()` join multiple patterns with `|` into a single
string, then pass that to `pd.Series.str.contains(..., regex=False)`:

```python
pattern = '|'.join(self.patterns)  # "nFragment|nPeptide|iRT_protein"
matches = feature_ids.str.contains(pattern, ..., regex=False)
```

With `regex=False`, pandas treats `|` as a **literal character**, not an OR operator.
It searches for the substring `"nFragment|nPeptide|iRT_protein"` verbatim.
No feature ID contains that literal string, so `matches` is all-False.

**Result**: The filter is a complete no-op. Every feature passes through unfiltered.
This affects any downstream pipeline using `MetadataRowFilter` for QC row removal.

### Impact Assessment

- **Silent data contamination**: QC rows (nFragment, nPeptide, iRT spike-ins) flow into
  differential analysis, inflating feature counts and corrupting statistical tests.
- Any analysis run with this filter active has included technical artifacts.

### Solution

**Option A (Preferred)**: Iterate patterns individually — preserves the `regex=False`
safety guarantee while correctly matching multiple substrings:

```python
def filter(self, feature_ids: pd.Index) -> pd.Index:
    mask = pd.Series(False, index=feature_ids)
    for pat in self.patterns:
        mask |= feature_ids.str.contains(
            pat, case=self.case_sensitive, na=False, regex=False
        )
    self.n_filtered_ = mask.sum()
    return feature_ids[~mask]
```

**Option B**: Use `regex=True` with proper escaping:
```python
import re
escaped = [re.escape(p) for p in self.patterns]
pattern = '|'.join(escaped)
matches = feature_ids.str.contains(pattern, case=self.case_sensitive, na=False, regex=True)
```

**Trade-off**: Option A is O(n×k) where k = number of patterns but avoids regex engine
overhead and injection risks. For typical k<10 patterns, negligible difference. Option B
is a single vectorized pass but introduces regex escaping complexity.

**Recommendation**: Option A. The filter operates on feature IDs (typically <10k rows)
with <10 patterns. Simplicity and correctness over micro-optimization. Apply the same
fix to `get_filtered_ids()`.

### Pitfalls
- Don't use `regex=True` without escaping — patterns like `iRT_protein` contain `.`
  and `_` which have regex meaning in some contexts.
- The case-insensitive test (`test_case_insensitive_default`) currently passes by
  accident — a single pattern doesn't trigger the `|`-join bug.

---

## F2: GPU Batched OLS — Numerical Correctness (Critical)

**Files**: `src/cliquefinder/stats/permutation_gpu.py:488-545` (GPU), `:548-600` (CPU)
**Tests**: `test_permutation_gpu.py::test_ols_matches_statsmodels`,
`test_permutation_gpu.py::test_gpu_matches_cpu_small`,
`test_differential_improvements.py::TestGPUBatchedOLS` (3 tests)

### Diagnosis

Four interacting issues in the GPU OLS path:

#### Issue 2a: float32 Precision Loss

The GPU path casts everything to `mx.float32` (line 496-499):
```python
Y_mx = mx.array(Y, dtype=mx.float32)
X_mx = mx.array(matrices.X, dtype=mx.float32)
```

For small datasets (n_samples~20, n_features~5), the design matrix `X'X` can have
condition numbers where float32 introduces ~1e-3 relative error in the inverse.
This propagates quadratically through the OLS formula.

**Evidence**: `test_batched_ols_performance` shows 42.1% of values exceed `rtol=1e-5`,
but the max relative difference is only 0.006 — consistent with float32 vs float64 gap.

#### Issue 2b: Empirical Bayes Type Mixing

Lines 521-523 mix Python `float` with MLX arrays:
```python
s2_post = (matrices.eb_d0 * matrices.eb_s0_sq + matrices.df_residual * sigma2) / (...)
#          ^Python float     ^Python float       ^Python int           ^MLX array
```

MLX may or may not auto-promote correctly. When EB parameters are extreme (d0→∞ or
d0→0), this can produce NaN through 0×∞ or ∞/∞ edge cases.

**Evidence**: `test_ols_matches_statsmodels` returns ALL NaN — a systematic type/overflow
failure, not a precision issue.

#### Issue 2c: Empty Batch Broadcasting Failure

`batched_median_polish_gpu()` at line 717:
```python
max_row_adj = float(mx.max(mx.abs(row_medians)))  # row_medians shape: (0, n_proteins)
```

`mx.max()` on a zero-element array raises `ValueError: Cannot max reduce zero size array`.

#### Issue 2d: CPU–GPU Observed Statistic Divergence

`test_gpu_matches_cpu_small` shows t-stat divergence (1.30 vs 0.97) — beyond float32
error. This suggests the CPU and GPU paths use different summarization: the CPU path
uses sequential `tukey_median_polish` per clique, while the GPU path uses
`batched_median_polish_gpu` which may differ in iteration/convergence behavior for
edge-case clique sizes.

### Solution Architecture

**Principle**: The OLS kernel must be bit-exact between CPU and GPU for the same
input precision. Empirical Bayes moderation is a post-hoc variance shrinkage — it
should be computed on CPU and applied to GPU results.

```
┌────────────────────────────────────────────────────┐
│  Precompute Phase (CPU, float64)                    │
│  ┌──────────────────────────────────────────┐       │
│  │ Design matrix X, (X'X)^-1, contrast c   │       │
│  │ Empirical Bayes: d0, s0², df_total       │       │
│  └──────────────────────────────────────────┘       │
├────────────────────────────────────────────────────┤
│  Batch OLS Phase (GPU, float32)                     │
│  ┌──────────────────────────────────────────┐       │
│  │ β = Y @ X @ (X'X)^-1                     │       │
│  │ σ² = RSS / df_residual                    │       │
│  │ Return: σ² array back to CPU              │       │
│  └──────────────────────────────────────────┘       │
├────────────────────────────────────────────────────┤
│  Moderation Phase (CPU, float64)                    │
│  ┌──────────────────────────────────────────┐       │
│  │ s²_post = (d0·s0² + df·σ²) / (d0 + df)  │       │
│  │ t = estimate / sqrt(s²_post × c_var)     │       │
│  └──────────────────────────────────────────┘       │
└────────────────────────────────────────────────────┘
```

**Specific fixes**:

1. **Float32 guard**: For the hot path (>100 features), float32 is fine — the
   aggregation across many features averages out error. For small validation tests,
   add a `precision='float64'` option or accept wider tolerance in tests.

2. **EB on CPU**: Move lines 521-527 (GPU) to a post-GPU CPU step:
   ```python
   # GPU: return raw sigma2 as numpy
   sigma2_np = np.array(sigma2, dtype=np.float64)
   # CPU: apply EB moderation
   if matrices.eb_d0 is not None and not np.isinf(matrices.eb_d0):
       s2_post = (matrices.eb_d0 * matrices.eb_s0_sq + matrices.df_residual * sigma2_np) / (...)
   ```

3. **Empty batch guard**: Add at top of `batched_median_polish_gpu()`:
   ```python
   if batch_size == 0:
       return np.empty((0, n_samples), dtype=np.float64)
   ```

4. **Summarization alignment**: Ensure `batched_median_polish_gpu` uses identical
   convergence criteria as the sequential `tukey_median_polish` in `summarization.py`.
   Currently both use `max_iter=10, eps=0.01`, but the batched version checks
   convergence globally (max across all batches) while sequential checks per-clique.
   This means the batched version may over- or under-iterate for individual cliques.

### Pitfalls
- **Don't just widen test tolerances** — that masks real disagreement in the
  summarization step. The t-stat divergence (0.97 vs 1.30) is 34%, well beyond
  float32 error (~0.01%). This is a real algorithmic difference.
- **Don't remove EB from GPU** entirely — the EB shrinkage per-feature is the
  efficiency win (limma's core insight). Just do the EB arithmetic on CPU.
- **MLX `mx.median` behavior**: Verify MLX handles NaN identically to `np.nanmedian`.
  If MLX propagates NaN (no `nanmedian`), all-NaN slices will corrupt the entire batch.

---

## F3: GPU OLS API Signature Drift

**Files**: `src/cliquefinder/stats/permutation_gpu.py:1113-1130`
**Tests**: `test_permutation_gpu.py::test_benchmark_full_pipeline`

### Diagnosis

`run_permutation_test_gpu()` now requires `condition_col` and `contrast` as positional
arguments (no defaults). The benchmark test at line 347 omits both:

```python
results, null_df = run_permutation_test_gpu(
    data=data, feature_ids=feature_ids, sample_metadata=metadata,
    clique_definitions=cliques, n_permutations=n_perms, verbose=False,
    # Missing: condition_col='condition', contrast=('CASE', 'CTRL')
)
```

### Solution

Update the test to pass the required arguments. This is a 2-line fix.

---

## F4: Imputer API — Removed KNN/Radius Strategies (Dead Tests)

**Files**: `src/cliquefinder/quality/imputation.py`
**Tests**: `test_optimization_equivalence.py` (6), `test_optimization_performance.py` (3)

### Diagnosis

The `Imputer` class was simplified to three strategies: `mad-clip`, `soft-clip`, `median`.
Nine tests still reference removed strategies (`knn_correlation`, `radius_correlation`)
and removed parameters (`n_neighbors`, `correlation_threshold`, `weighted`, `weight_power`).

Current signature:
```python
class Imputer:
    def __init__(self, strategy="mad-clip", threshold=5.0, sharpness=None,
                 group_cols=None, max_upper_bound=None):
```

### Solution

**Decision required**: Were KNN/radius strategies intentionally removed, or are they
planned future work?

- **If intentionally removed**: Delete `test_optimization_equivalence.py` and
  `test_optimization_performance.py` entirely, or rewrite them to test the current
  `mad-clip`/`soft-clip`/`median` strategies.

- **If planned**: Keep tests as aspirational, mark with `@pytest.mark.skip(reason="KNN strategies not yet implemented")`.

**Recommendation**: Delete. The `test_legacy_removed.py` already validates that these
strategies correctly raise `ValueError`. The current strategies (MAD-clip with
group-stratified bounds, soft sigmoid clipping) are statistically sound for proteomics
outlier handling. KNN-based imputation introduces correlation structure that can inflate
downstream test statistics — a known problem in the field (Webb-Robertson et al., 2015).

---

## F5: Rotation — Removed FDR-Adjusted P-Values (Intentional)

**Files**: `src/cliquefinder/stats/rotation.py:1940-1965`
**Tests**: `test_rotation.py::test_results_to_dataframe`, `test_rotation.py::test_run_rotation_test_basic`

### Diagnosis

Tests assert `'adj_pvalue_msq_mixed' in df.columns`, but `results_to_dataframe()` now
returns only raw p-values (`pvalue_msq_mixed`). The `adj_` prefix columns were removed.

The documentation explicitly states the rationale:

> *"ROAST produces exact p-values per gene set via rotation tests. These raw p-values
> are statistically valid without FDR correction."*

This is statistically correct. ROAST p-values are self-calibrated — each gene set's
p-value is computed from its own null distribution (via rotation). Unlike enrichment
analyses that test thousands of gene sets from a database, ROAST tests researcher-specified
sets where multiple testing correction may be overly conservative.

### Solution

Update tests to assert `pvalue_msq_mixed` instead of `adj_pvalue_msq_mixed`.
If FDR is needed for exploratory analyses (many gene sets), the user can apply
`statsmodels.stats.multitest.multipletests()` externally. Don't bake it into the engine.

---

## F6: Stale CLI Default Assertions

**Tests**: `test_legacy_removed.py::test_required_parameters_only`,
`test_clinical_metadata_integration.py::test_clinical_metadata_integration`

### Diagnosis

1. **Imputer threshold**: Test asserts `threshold == 3.5`, actual default is `5.0`.
   The default was changed (5.0 MAD-Z is the common convention for proteomics).

2. **phenotype_source_col**: Test asserts default `'SUBJECT_GROUP'`, actual is `None`.
   The CLI was refactored to auto-detect phenotype columns.

### Solution

Update assertions to match current defaults. Verify the defaults are scientifically
appropriate:
- MAD-Z threshold 5.0: standard in DIA proteomics (corresponds to ~1 in 3.5 million
  under normality). Conservative but appropriate for untargeted discovery.
- `phenotype_source_col=None`: auto-detection is correct behavior for a general tool.

---

## F7: Cross-Modal Mapper — Case Normalization Gap (Logic Bug)

**File**: `src/cliquefinder/knowledge/cross_modal_mapper.py:455-590`
**Test**: `test_cross_modal_mapper.py::TestRealWorldScenarios::test_mouse_species`

### Diagnosis

Protein symbols and RNA symbols are not normalized to the same case before intersection:

- **Protein path** (line 530-532): On successful API resolution, returns symbols in
  their canonical form (`Trp53`, `Brca1` — mouse sentence case). Only the exception
  fallback does `.upper()`.

- **RNA path** (line 554): Explicitly uppercases: `rna_mapping = {k: v.upper() for k, v in rna_mapping.items()}`

- **Intersection** (line 558): `common_genes = protein_symbols & rna_symbols`

Result: `{'Brca1', 'Trp53'} & {'BRCA1', 'TRP53'} == set()`.

### Impact Assessment

This silently produces zero overlap for any non-human species (and potentially for
human aliases that resolve to mixed-case symbols). The `mapping_stats['overlap_rate']`
reports 0.0, but no error is raised.

### Solution

Normalize protein symbols to uppercase at the same point as RNA symbols. Add
normalization after successful API resolution (around line 527):

```python
# After protein resolution
protein_symbols = {s.upper() for s in protein_symbols}
```

Alternatively, normalize at the intersection point for clarity:

```python
# Normalize both to uppercase for comparison
protein_upper = {s.upper() for s in protein_symbols}
rna_upper = {s.upper() for s in rna_symbols}
common_genes = protein_upper & rna_upper
```

Keep the original-case symbols in the mapping for downstream use (gene names in reports
should preserve the canonical species-appropriate form). Use the uppercase intersection
only for set membership testing.

### Pitfalls
- Don't lowercase — gene symbols are conventionally uppercase (human) or sentence case
  (mouse). Upper is the safe normalization direction.
- Ensure the `protein_only` and `rna_only` sets also use the same normalization,
  otherwise the partition `common + protein_only + rna_only` won't equal the universe.

---

## F8: CliqueValidator — Empty Gene List

**File**: `src/cliquefinder/knowledge/clique_validator.py:770`
**Test**: `test_graph_construction_benchmark.py::test_edge_cases`

### Diagnosis

`_get_gene_indices([])` raises `GeneNotFoundError("None of the 0 genes found...")`.
The test expects an empty graph to be returned instead.

### Solution

Add early return in `build_correlation_graph()`:

```python
def build_correlation_graph(self, genes, condition, ...):
    if not genes:
        return nx.Graph()  # Empty graph for empty input
```

This is the principle of least surprise — vacuous input produces vacuous output.

---

## F9: Satterthwaite df — Numerical Edge Cases

**File**: `src/cliquefinder/stats/differential.py`
**Tests**: `test_satterthwaite_df.py::test_satterthwaite_edge_cases`,
`test_satterthwaite_df.py::test_satterthwaite_improves_accuracy`

### Diagnosis

Satterthwaite-Welch df approximation:

```
df_sw = (σ_between² + σ_within²/n)² / [(σ_between²)²/(n_groups-1) + (σ_within²/n)²/(n_total-n_groups)]
```

Edge cases where variance components are near-zero or very unbalanced produce df
estimates that diverge from test expectations. The test uses `rtol=1e-4` which is too
tight for the numerical approximation.

### Solution

1. Widen tolerance to `rtol=1e-2` for the edge case test — the Satterthwaite
   approximation is itself an approximation, not an exact formula.
2. Add bounds clamping: `df_sw = max(1.0, min(df_sw, n_total - 1))` to prevent
   degenerate df values.
3. For the "improves accuracy" test, compare against simulation-based reference
   values rather than analytic expectations.

---

## F10: Accidental Test Collection

**File**: `src/cliquefinder/stats/differential.py:715`
**Error**: `fixture 'coef_df' not found`

### Diagnosis

Production function `test_contrasts()` starts with `test_` prefix. Pytest's
`testpaths = ["tests", "src"]` configuration (from `pyproject.toml`) causes it to be
collected as a test.

### Solution

**Option A**: Rename to `evaluate_contrasts()` or `compute_contrasts()`.

**Option B**: Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]  # Don't collect from src/
```

**Recommendation**: Option B first (immediate fix), then Option A during next
refactoring pass. The function name `test_contrasts` is misleading in a production
module regardless of pytest collection.

**Also**: Check if `testpaths` includes `src` intentionally for doctest collection.
If so, use `collect_ignore` patterns instead.

---

## F11: `derive_genetic_phenotype` Import Error

**File**: `tests/test_genetic_contrast.py:15`
**Error**: `ImportError: cannot import name 'derive_genetic_phenotype'`

### Diagnosis

`derive_genetic_phenotype()` was moved from `differential.py` to `cohort.py` as part
of the cohort refactoring (visible in git diff). The test file still imports from the
old location.

### Solution

Update import:
```python
from cliquefinder.cohort import derive_genetic_phenotype  # or resolve_cohort_from_args
```

Verify the function signature hasn't changed in the move.

---

## Prioritized Fix Order

### Tier 1: Data Correctness (blocks scientific validity)

| Priority | Issue | Effort | Risk if Unfixed |
|----------|-------|--------|-----------------|
| **P0** | F1: MetadataRowFilter no-op | 30 min | QC rows contaminate all downstream analysis |
| **P0** | F7: Cross-modal case mismatch | 15 min | Zero overlap for non-human species |
| **P1** | F2: GPU OLS NaN + precision | 2-4 hr | Permutation test results unreliable |

### Tier 2: API Hygiene (blocks clean test suite)

| Priority | Issue | Effort | Risk if Unfixed |
|----------|-------|--------|-----------------|
| **P2** | F4: Delete dead Imputer tests | 15 min | Noise in CI |
| **P2** | F5: Update rotation assertions | 10 min | Noise in CI |
| **P2** | F6: Update stale defaults | 10 min | Noise in CI |
| **P2** | F3: Fix GPU test args | 5 min | Noise in CI |
| **P2** | F8: Empty gene list guard | 10 min | Minor edge case |
| **P2** | F10: Rename test_contrasts | 5 min | Pytest confusion |
| **P2** | F11: Fix import path | 5 min | Collection error |

### Tier 3: Statistical Refinement

| Priority | Issue | Effort | Risk if Unfixed |
|----------|-------|--------|-----------------|
| **P3** | F9: Satterthwaite tolerances | 30 min | Minor test noise |

---

## Design Pattern Notes

### Defensive Numerics Pattern

For GPU code that mixes Python scalars with array types, use an explicit
**type-barrier** at the GPU–CPU boundary:

```python
# Pattern: GPU compute → CPU barrier → CPU moderation
raw_stats = gpu_kernel(Y_gpu, X_gpu)          # All MLX arrays
raw_stats_np = np.asarray(raw_stats, dtype=np.float64)  # Barrier
moderated = eb_moderate(raw_stats_np, d0, s0_sq)         # All numpy
```

This eliminates mixed-type arithmetic bugs and makes the precision boundary explicit.

### Filter Composition Pattern

For ID filtering with multiple patterns, prefer **pattern-per-pass** over
**single-regex**:

```python
# Anti-pattern: join + regex
pattern = '|'.join(patterns)
mask = ids.str.contains(pattern, regex=True)  # Regex escaping bugs

# Pattern: reduce over masks
mask = functools.reduce(
    operator.or_,
    (ids.str.contains(p, case=False, regex=False) for p in patterns),
    pd.Series(False, index=ids)
)
```

Each pass is simple, testable, and immune to regex injection.

### Vacuous Input Pattern

For methods operating on collections, handle the empty case at the boundary:

```python
def build_graph(self, genes: list[str], ...) -> nx.Graph:
    if not genes:
        return nx.Graph()
    # ... main logic
```

This follows the mathematical convention: the empty graph has 0 vertices and 0 edges.
