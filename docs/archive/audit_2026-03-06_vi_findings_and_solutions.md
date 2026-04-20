# Audit VI — Findings, Solutions, and Pitfalls

**Date:** 2026-03-06
**Critics:** Claude, Gemini (via brutalist MCP) — 10 critic reports across 5 domains
**Domains reviewed:** Statistics/Rotation, Knowledge Graph/Coherence, Permutation/GPU, Security, Test Coverage
**Verification method:** Every finding cross-referenced against actual codebase at commit `94f0b5b` (post-Audit V)

---

## Critical Assessment Summary

**Total raw findings from critics:** ~50
**Already fixed in prior audits (I-V):** ~20
**Invalid / misread code:** ~8
**Novel valid findings:** 14

---

## Table of Contents

1. [Statistical Correctness](#1-statistical-correctness)
2. [Knowledge Graph / Coherence](#2-knowledge-graph--coherence)
3. [Code Quality / Dead Code](#3-code-quality--dead-code)

---

## Rejected Findings (with rationale)

### STAT-VI-R1: "mean50 MIXED uses signed w*z, causes cancellation"
**Verdict: BY DESIGN** — The mean50 MIXED stat uses `w * z` (signed) just like the `mean` stat. This is consistent with limma. For MIXED alternative, the p-value comparison uses `|null| >= |obs|`, so sign cancellation reduces power but doesn't invalidate the test. The `msq` stat exists for direction-agnostic testing.

### STAT-VI-R2: "EB prior estimated once, applied to all nulls"
**Verdict: ALREADY ASSESSED** — Audit V (STAT-V-3) reviewed this and confirmed it follows Wu et al. (2010) and limma exactly. Rotation preserves variance structure by construction.

### KG-VI-R1: "CoGExClient credential logging"
**Verdict: ALREADY FIXED** — Audit III (SEC-III-3) sanitized credential logging. Current code only logs "Using explicit credentials" (no values) and logs the URL but NOT password at connection time.

### KG-VI-R2: "_HGNC_SYMBOL_TO_ID mutable singleton"
**Verdict: ALREADY ASSESSED** — Audit V confirmed this is read-only after initialization with no mutating code paths.

### SEC-VI-R1: "HTML XSS / pickle / eval injection"
**Verdict: ALREADY FIXED** — All addressed in Audit I (SEC-1 through SEC-7).

### GPU-VI-R1: "float32 catastrophic cancellation in RSS"
**Verdict: ALREADY FIXED** — Audit II (GPU-3) switched to algebraic identity `RSS = Y'Y - β'X'Y` with Y'Y on CPU in float64.

### GPU-VI-R2: "SeedSequence not used for ThreadPoolExecutor"
**Verdict: ALREADY FIXED** — Audit V (KG-V-2) added SeedSequence to stability.py.

### TEST-VI-R1: "Source inspection tests are ghost tests"
**Verdict: ALREADY ASSESSED** — Audit V confirmed these are intentional architecture enforcement tests.

---

## 1. Statistical Correctness

### 1.1 `squeeze_var` Returns Original Variances When d0=inf (Should Return Prior)

**Finding ID:** STAT-VI-1
**Severity:** HIGH
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/stats/permutation_gpu.py:246-250`

**The problem:**
When `d0=inf`, `squeeze_var` returns original variances with the comment "No shrinkage":

```python
if np.isinf(d0):
    # No shrinkage - return original variances
    return sigma2.copy(), float(df)
```

But mathematically, `s2_post = (d0 * s0_sq + df * sigma2) / (d0 + df)` → `s0_sq` as d0→∞. This means d0=inf indicates **maximum shrinkage** (prior dominates completely), NOT "no shrinkage."

The rotation engine (`rotation.py:1014-1016`) correctly handles this:
```python
if np.isinf(eb_d0):
    # d0=inf: prior dominates completely — use prior variance for all genes
    moderated_variances = np.full_like(sample_variances, eb_s0_sq)
```

**Impact:** `squeeze_var` is called from `differential.py:1473` for EB-moderated differential analysis. When `fit_f_dist` estimates d0=inf (which happens with very homogeneous datasets), the differential pipeline returns unshrunken variances instead of maximally shrunken ones.

**Solution:**
```python
if np.isinf(d0):
    # d0=inf: prior dominates completely — use prior variance for all genes
    s2_post = np.full_like(sigma2, s0_sq)
    if df_is_array:
        return s2_post, np.full_like(df, np.inf, dtype=np.float64)
    return s2_post, np.inf
```

**Pitfall:** Must also update `df_total` to `inf` (not original `df`), since d0=inf means infinite prior degrees of freedom.

---

### 1.2 `fit_general()` Missing Zero-Variance and NaN Gene Filters

**Finding ID:** STAT-VI-2
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:2034-2091`

**The problem:**
`fit()` (lines 1888-1925) filters out zero-variance genes and NaN-containing genes before QR projection and EB estimation. `fit_general()` skips these filters entirely:

- **Zero-variance genes** produce NaN in t-statistics and contaminate EB prior estimation
- **NaN genes** propagate through `Y @ Q2`, corrupting `rho_sq` for ALL genes

The `fit_general()` path is used for interaction contrasts (Sex × Disease), multi-group comparisons, and arbitrary design matrices — exactly the complex scenarios most likely to have edge-case data issues.

**Solution:**
Extract the zero-variance and NaN filters from `fit()` into a shared `_filter_genes()` method and call it from both `fit()` and `fit_general()`.

**Pitfall:** `fit()` also handles condition-based sample filtering and >2 group warnings — those should NOT be shared. Only the gene-level filters should be extracted.

---

### 1.3 Permutation Null p-value Missing Phipson-Smyth +1 Correction

**Finding ID:** STAT-VI-3
**Severity:** MEDIUM
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:1073`

**The problem:**
```python
p_value = np.mean(null_modularities >= observed_modularity)
```

This uses `b/B` instead of the Phipson-Smyth formula `(b+1)/(B+1)` used everywhere else in the codebase (e.g., `rotation.py:1663`). The `b/B` formula:
1. Can return exactly 0.0 (impossible under proper Monte Carlo)
2. Anti-conservative for small permutation counts (n_permutations=100 default)

**Solution:**
```python
b = np.sum(null_modularities >= observed_modularity)
p_value = (b + 1) / (n_permutations + 1)
```

---

## 2. Knowledge Graph / Coherence

### 2.1 Community Detection Missing Random Seed (Non-deterministic)

**Finding ID:** KG-VI-1
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:651-693`

**The problem:**
`CoherenceAnalyzer` has `self._rng` (added in Audit V), but community detection calls don't use it:

```python
# Line 651 — python-louvain
community_louvain.best_partition(G, weight='weight', resolution=...)
# Missing: random_state parameter

# Line 658 — networkx.community
nx_community.louvain_communities(G, weight='weight', resolution=...)
# Missing: seed parameter

# Line 688 — leidenalg
leidenalg.find_partition(ig_graph, ..., resolution_parameter=...)
# Missing: seed parameter
```

This causes non-deterministic community assignments across runs, even with the same data and parameters.

**Solution:**
Thread `self._rng` seed through to all three backends:
- `community_louvain.best_partition(..., random_state=seed)`
- `nx_community.louvain_communities(..., seed=seed)`
- `leidenalg.find_partition(..., seed=seed)`

**Pitfall:** Leiden's `seed` parameter must be a Python int, not a numpy int. Use `int(seed)`.

---

### 2.2 `modularity_contribution` Divides Total by Community Count

**Finding ID:** KG-VI-2
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:1212, 1236`

**The problem:**
```python
modularity_contribution=modularity_pos / len(comm_pos) if comm_pos else 0.0
```

This divides the graph's total modularity by the number of communities, assigning each community an equal share. This is nonsensical — a 3-gene community and a 50-gene community get the same "contribution." The field name suggests per-community modularity contribution, but the value is just `Q / k`.

**Solution:**
Compute actual per-community modularity contribution using NetworkX:
```python
# Per-community modularity contribution
community_nodes = set(comm_genes)
modularity_contribution = nx_community.modularity(G, [community_nodes, G.nodes() - community_nodes])
```

Or more simply, rename the field and document what it actually represents.

**Approach:** Use `nx.community.modularity(G, [{comm_genes}, {rest}])` for per-community measure. This gives the actual contribution of this partition to overall modularity.

---

### 2.3 Spearman 2-Gene Edge Case Returns 1×1 Identity

**Finding ID:** KG-VI-3
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:527-529`

**The problem:**
When computing Spearman correlation with exactly 2 genes, `scipy.stats.spearmanr` returns a scalar (the single correlation coefficient). The current handling:
```python
if corr_matrix.ndim == 0:
    corr_matrix = np.array([[1.0]])
```

This creates a 1×1 matrix with just 1.0 (the diagonal), losing the actual correlation value between the two genes. The correct 2×2 matrix should be:
```
[[1.0, rho],
 [rho, 1.0]]
```

But `spearmanr` with 2 variables returns a scalar correlation, so `ndim == 0` is True. We need the 2×2 matrix.

**Solution:**
```python
if corr_matrix.ndim == 0:
    # spearmanr returns scalar for 2 variables — reconstruct 2×2 matrix
    rho = float(corr_matrix)
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
```

**Pitfall:** Also need to check for `expr_data.T` shape — with 2 genes, it's (n_samples, 2) which is correct for `spearmanr`.

---

## 3. Code Quality / Dead Code

### 3.1 Dead Imports: ProcessPoolExecutor, as_completed

**Finding ID:** CLEAN-VI-1
**Severity:** INFO
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:33`

**The problem:**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

Neither `ProcessPoolExecutor` nor `as_completed` is used anywhere in the file. Dead import.

**Solution:** Remove the import line.

---

### 3.2 Dead Code: `_leiden_multiplex` Method

**Finding ID:** CLEAN-VI-2
**Severity:** INFO
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:697-800`

**The problem:**
`_leiden_multiplex()` is defined but never called from `detect_communities()` or any other method. The method switch at line 634 routes to `_louvain_communities`, `_leiden_communities`, or `_greedy_communities` — multiplex Leiden is not an option.

**Assessment:** This may be intentional future code, but it's ~100 lines of untested dead code that could rot. Since it's not in the `CommunityMethod` enum and has no tests, it should be removed.

**Solution:** Remove `_leiden_multiplex()` method entirely. If needed later, it can be retrieved from git history.

---

### 3.3 Unused `CorrelationSign.BOTH` Enum Value

**Finding ID:** CLEAN-VI-3
**Severity:** INFO
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:80`

**The problem:**
```python
class CorrelationSign(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOTH = "both"  # Combined (use with caution)
```

`CorrelationSign.BOTH` is never referenced anywhere in the codebase (grep confirms 0 matches). It's defined with a warning comment "use with caution" but has no implementation.

**Solution:** Remove the `BOTH` variant. If multiplex analysis is added later, it can be re-added with actual implementation.

---

### 3.4 `analyze_all_conditions` Bare Exception Swallows Errors

**Finding ID:** CLEAN-VI-4
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:1294`

**The problem:**
```python
except Exception as e:
    logger.error(f"Error analyzing {condition}: {e}")
    continue
```

This catches ALL exceptions including `TypeError`, `AttributeError`, `KeyError` — real bugs that should propagate. Only expected failures (e.g., too few samples for a condition) should be caught.

**Solution:**
Catch only the expected exception type:
```python
except ValueError as e:
    logger.error(f"Error analyzing {condition}: {e}")
    continue
```

---

## Implementation Plan

### Cycle 1: Statistical Fixes (STAT-VI-1, STAT-VI-2, STAT-VI-3)

**Files:** `permutation_gpu.py`, `rotation.py`, `regulatory_coherence.py`

| Finding | Fix | Tests |
|---------|-----|-------|
| STAT-VI-1 | Fix `squeeze_var` d0=inf to return s0_sq | 3 tests |
| STAT-VI-2 | Extract `_filter_genes()`, call from `fit_general()` | 4 tests |
| STAT-VI-3 | Phipson-Smyth p-value in `permutation_null` | 2 tests |

### Cycle 2: Knowledge Graph Fixes (KG-VI-1, KG-VI-2, KG-VI-3)

**Files:** `regulatory_coherence.py`

| Finding | Fix | Tests |
|---------|-----|-------|
| KG-VI-1 | Thread seed to Louvain/Leiden/NetworkX community detection | 3 tests |
| KG-VI-2 | Compute per-community modularity properly | 2 tests |
| KG-VI-3 | Fix Spearman 2-gene edge case | 2 tests |

### Cycle 3: Code Cleanup (CLEAN-VI-1 through CLEAN-VI-4)

**Files:** `regulatory_coherence.py`

| Finding | Fix | Tests |
|---------|-----|-------|
| CLEAN-VI-1 | Remove dead imports | 0 tests |
| CLEAN-VI-2 | Remove `_leiden_multiplex` dead code | 0 tests |
| CLEAN-VI-3 | Remove `CorrelationSign.BOTH` | 1 test |
| CLEAN-VI-4 | Narrow exception catch | 1 test |

---

## Status

- [x] Cycle 1: Statistical Fixes — 10 tests, 3 files modified
- [x] Cycle 2: Knowledge Graph Fixes — 10 tests, 1 file modified
- [x] Cycle 3: Code Cleanup — 7 tests, 1 file modified
- **Total: 27 new tests, 857 passed, 1 skipped, 1 pre-existing failure**
