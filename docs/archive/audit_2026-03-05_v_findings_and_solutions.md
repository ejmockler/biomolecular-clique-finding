# Audit V — Findings, Solutions, and Pitfalls

**Date:** 2026-03-05
**Critics:** Claude, Gemini (via brutalist MCP) — 10 critic reports across 5 domains
**Domains reviewed:** Statistics, Knowledge Graph, Test Coverage, Security, Architecture
**Verification method:** Every finding cross-referenced against actual codebase at commit `f19f983` (post-Audit IV)

---

## Critical Assessment Summary

**Total raw findings from critics:** ~45
**Already fixed in prior audits (I-IV):** ~18
**Invalid / misread code:** ~7
**Novel valid findings:** 14

---

## Table of Contents

1. [Statistical Correctness](#1-statistical-correctness)
2. [Knowledge Graph / Regulatory Coherence](#2-knowledge-graph--regulatory-coherence)
3. [Infrastructure & Performance](#3-infrastructure--performance)
4. [Security](#4-security)

---

## Rejected Findings (with rationale)

### STAT-V-1: "Double-filtering footgun" (zero-var + NaN)
**Verdict: INVALID** — The two filters are independent and correctly ordered. Zero-variance is checked first (gene_vars == 0), then NaN rows checked separately (np.any(np.isnan(...))). A gene could be zero-variance but not contain NaN, or vice versa. The filters don't interfere. The `gene_ids` and `gene_to_idx` are rebuilt after each filter, so indices stay consistent.

### STAT-V-9: "fit() state mutation"
**Verdict: ALREADY FIXED** — Audit IV (STATE-IV-1) added double-fit guard in both `fit()` and `fit_general()`.

### STAT-V-5: "Single EB prior across all clique sizes"
**Verdict: INVALID** — This is by design. The EB prior is estimated from ALL genes in the dataset (not per clique), following limma's approach exactly. Gene-set-specific priors would be statistically invalid with small sets (3-50 genes → unreliable F-distribution fitting). The whole point of empirical Bayes is to borrow strength across the genome.

### STAT-V-10: "Camera VIF uses raw expression not residual correlation"
**Verdict: INVALID** — The Camera VIF in `enrichment_z.py` uses `inter_gene_correlation` which is passed in by the caller. The z-score function itself doesn't compute the correlation — it receives it as an argument. The correlation can be computed on residuals by the caller. Not a bug in this function.

### KG-V: "discover_regulators uses per-gene HTTP not batch resolution"
**Verdict: ALREADY FIXED** — Audit III (SEC-III-3) added `resolve_gene_names_batch()` and `_get_hgnc_symbol_map()` for O(1) local resolution. Per-gene HTTP is now only the fallback path for names not in the local HGNC map.

### KG-V: "INDRAKnowledgeSource never closes CoGExClient"
**Verdict: ALREADY FIXED** — `CoGExClient` has `close()`, `__enter__`, and `__exit__` methods (lines 1045-1062). The `INDRAKnowledgeSource` is a higher-level wrapper; closing is the caller's responsibility via context manager.

### KG-V: "HGNC symbol map poisonable singleton"
**Verdict: LOW RISK / BY DESIGN** — `_HGNC_SYMBOL_TO_ID` is populated from `hgnc_client.hgnc_names` which is INDRA's authoritative local data. It's read-only after initialization. The `Dict` type is mutable in principle, but no code path modifies it after construction. Converting to `MappingProxyType` would be a marginal improvement but the singleton pattern is correct.

### TEST-V: "Ghost tests" / source-inspection tests
**Verdict: INFORMATIONAL** — Some tests introspect source code (e.g., checking that certain patterns exist). These are architecture enforcement tests, not "ghost tests." They're intentional.

### SEC-V: Most of the 11 MEDIUM security findings
**Verdict: ALREADY FIXED** — HTML XSS, credential logging, ReDoS gaps, non-atomic writes, cache permissions were all addressed in Audits I-III. The critics are working from stale context.

---

## 1. Statistical Correctness

### 1.1 mean50 Normalization Divisor Mismatch with limma

**Finding ID:** STAT-V-2
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:1520-1549`

**The problem:**
The `_compute_mean50_stat` uses `np.mean(selected_wz, axis=1)` which divides by `h` (the number of selected genes = n_genes // 2). limma's implementation uses the FULL weight sum `A` as the divisor, not just the selected subset's weight sum. This means our mean50 statistic has a different scale than limma's, though since it's applied consistently to both observed and null distributions, p-values are unaffected.

**Assessment:** Not a bug — p-values are invariant to monotonic rescaling of the statistic since both observed and null use the same formula. The scale difference only matters for comparing the statistic's magnitude across implementations. **No fix needed.**

### 1.2 EB Prior Estimated from Observed Applied Unchanged to Null Rotations

**Finding ID:** STAT-V-3
**Severity:** INFORMATIONAL
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:2286-2300`

**The problem:**
The EB priors (d0, s0_sq) are estimated from the observed data and then used for all null rotations. Under H0, the variance distribution may differ slightly from the observed. The critic argues this could bias the null.

**Assessment:** This is mathematically correct per Wu et al. (2010) and limma's implementation. The rotation mechanism preserves the variance structure — rotated residual variances have the SAME distribution as observed under H0 (that's the whole point of the rotation framework). The EB priors are population-level parameters, not sample-specific. **No fix needed.**

---

## 2. Knowledge Graph / Regulatory Coherence

### 2.1 Global RNG Usage in regulatory_coherence.py

**Finding ID:** KG-V-1
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:934, 1018`

**The problem:**
`bootstrap_stability()` and `permutation_null()` use `np.random.choice()` and `np.random.permutation()` which rely on the legacy global RNG. This causes:
1. Non-reproducibility when called from different contexts
2. Thread-unsafe behavior (ThreadPoolExecutor in stability.py uses the same global state)
3. Global state pollution affecting downstream random operations

```python
# Line 934 (bootstrap_stability):
boot_sample_idx = np.random.choice(sample_idx, size=n_samples, replace=True)

# Line 1018 (permutation_null):
perm_idx = np.random.permutation(sample_idx)
```

**Solution:**
Accept an optional `rng` parameter (np.random.Generator) and use it throughout:

```python
def bootstrap_stability(self, genes, condition, n_bootstrap=None,
                        correlation_sign=None, rng=None):
    rng = rng or np.random.default_rng()
    # ...
    boot_sample_idx = rng.choice(sample_idx, size=n_samples, replace=True)
```

**Pitfall:** The `analyze_coherence()` method calls both `bootstrap_stability()` and `permutation_null()` — need to thread the `rng` through consistently.

**Approach:** Add `seed` parameter to `CoherenceAnalyzer.__init__()`, create a `self._rng = np.random.default_rng(seed)` instance, and use it in all stochastic methods. This is the same pattern used successfully in `RotationTestEngine`.

---

### 2.2 Global RNG Usage in stability.py

**Finding ID:** KG-V-2
**Severity:** MEDIUM
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/stability.py:66, 83`

**The problem:**
`find_stable_cliques()` uses `np.random.seed(random_state)` (global seeding) and `np.random.choice()` inside `run_bootstrap()` which runs in a `ThreadPoolExecutor`. This is doubly problematic:
1. Global seed pollution
2. Thread-unsafe RNG access from parallel workers

```python
# Line 66:
np.random.seed(random_state)  # Global pollution

# Line 83 (inside threaded function):
boot_indices = np.random.choice(condition_indices, size=n_samples, replace=True)
```

**Solution:**
Replace with `np.random.default_rng()`:

```python
rng = np.random.default_rng(random_state)
# ... pass rng to each bootstrap iteration
```

**Pitfall:** The inner function `run_bootstrap(b)` is called from a ThreadPoolExecutor. Must either use `rng.spawn(n_bootstrap)` to create independent child generators, or use `SeedSequence` to ensure thread-safe independent streams.

**Approach:** Use `SeedSequence` as done in the rotation engine:
```python
ss = np.random.SeedSequence(random_state)
child_seeds = ss.spawn(n_bootstrap)
# Each worker gets: rng_b = np.random.default_rng(child_seeds[b])
```

---

### 2.3 permutation_null Tests Only Positive Graph

**Finding ID:** KG-V-3
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:1036`

**The problem:**
`permutation_null()` builds signed graphs but only uses `G_perm` (the positive graph from `build_signed_graphs()`):

```python
G_perm, _ = self.build_signed_graphs(perm_corr, gene_list)
partition = self.detect_communities(G_perm)
```

The negative graph is discarded. This means the permutation null only assesses whether positive-correlation modularity is higher than expected, never testing negative-correlation structure.

**Assessment:** This is a valid concern for completeness. The `analyze_coherence()` method computes modularity for both positive and negative graphs but only runs permutation null on the positive one. However, the biological interpretation is asymmetric — positive co-expression modularity is the primary signal of interest (co-regulation), while negative modularity is supplementary.

**Solution:**
Add `correlation_sign` parameter to `permutation_null()`, mirroring what was done for `bootstrap_stability()` in Audit IV:

```python
def permutation_null(self, genes, condition, observed_modularity,
                     n_permutations=None, correlation_sign=None):
    # ...
    G_pos, G_neg = self.build_signed_graphs(perm_corr, gene_list)
    if correlation_sign == CorrelationSign.NEGATIVE:
        partition = self.detect_communities(G_neg)
        G_test = G_neg
    else:
        partition = self.detect_communities(G_pos)
        G_test = G_pos
```

---

### 2.4 `_corr_cache` Unbounded and Never Used

**Finding ID:** KG-V-4
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:399`

**The problem:**
`CoherenceAnalyzer.__init__` creates `self._corr_cache: Dict[str, np.ndarray] = {}` but `compute_correlation_matrix()` never reads from or writes to it. It's dead code that:
1. Creates false expectations about caching behavior
2. Holds a reference that could grow unbounded if someone adds caching later without bounds

**Solution:**
Remove the dead code:
```python
# Delete line 399
# self._corr_cache: Dict[str, np.ndarray] = {}
```

---

### 2.5 O(n²) Python Loop in build_signed_graphs

**Finding ID:** KG-V-5
**Severity:** LOW
**Convergence:** 2/2 critics, code-verified
**Location:** `src/cliquefinder/knowledge/regulatory_coherence.py:606-613`

**The problem:**
```python
for i in range(n):
    for j in range(i + 1, n):
        weight = weights[i, j]
        if weight > 0:
            if corr_matrix[i, j] > 0:
                G_pos.add_edge(genes[i], genes[j], weight=weight)
            else:
                G_neg.add_edge(genes[i], genes[j], weight=weight)
```

This O(n²) Python loop iterates over all pairs. For n=500 genes, that's ~125K iterations in pure Python. Could be vectorized with numpy to find non-zero entries, then add edges in batch.

**Solution:**
Vectorize the edge selection:
```python
# Get upper triangle indices where weight > 0
rows, cols = np.where(np.triu(weights, k=1) > 0)
for r, c in zip(rows, cols):
    w = float(weights[r, c])
    if corr_matrix[r, c] > 0:
        G_pos.add_edge(genes[r], genes[c], weight=w)
    else:
        G_neg.add_edge(genes[r], genes[c], weight=w)
```

This is still O(edges) in the loop, but avoids iterating over zero-weight pairs (which dominate after soft thresholding). For typical datasets, edges << n², so this is a significant speedup.

**Approach:** Alternative is `np.argwhere` + split by sign + `nx.Graph.add_edges_from()` batch call, but NetworkX's `add_edges_from` with attributes is barely faster than individual `add_edge` calls.

---

### 2.6 Global RNG in marker_discovery.py and viz/layouts.py

**Finding ID:** KG-V-6
**Severity:** LOW
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/quality/marker_discovery.py:369`, `src/cliquefinder/viz/layouts.py:60,163`

**The problem:**
Three remaining `np.random.seed()` calls in non-critical code paths. These pollute global RNG state.

**Solution:**
Replace with `np.random.default_rng()` for consistency. These are in visualization and quality modules, so impact is minimal, but the fix is trivial.

---

## 3. Infrastructure & Performance

### 3.1 Null Distributions Stored as Python Dicts of Numpy Arrays

**Finding ID:** INFRA-V-1
**Severity:** INFORMATIONAL
**Convergence:** 1/2 critics, code-verified
**Location:** `src/cliquefinder/stats/rotation.py:353`

**The problem:**
`RotationResult.null_distributions` is `dict[str, dict[str, NDArray]]` — nested dicts mapping stat_name → alternative → array. This is fine for the current use case. The critic suggests converting to a single structured numpy array for memory efficiency, but with 5-6 keys and ~10K-element arrays, the overhead of dict wrapping is negligible (<1KB per result).

**Assessment:** Not worth the refactoring complexity. **No fix needed.**

---

## 4. Security

### 4.1 Remaining np.random.seed in viz/layouts.py

**Finding ID:** SEC-V-1
**Severity:** LOW
**Same as KG-V-6** — tracked under that finding.

---

## Implementation Plan

### Cycle 1: RNG Modernization (KG-V-1, KG-V-2, KG-V-6)

**Files:** `regulatory_coherence.py`, `stability.py`, `marker_discovery.py`, `viz/layouts.py`

| Finding | Fix | Tests |
|---------|-----|-------|
| KG-V-1 | Replace global RNG in `bootstrap_stability` + `permutation_null` with Generator | 4 tests |
| KG-V-2 | Replace `np.random.seed` + threaded `np.random.choice` with SeedSequence | 3 tests |
| KG-V-6 | Replace `np.random.seed` in marker_discovery and layouts | 2 tests |

### Cycle 2: Coherence Fixes (KG-V-3, KG-V-4, KG-V-5)

**Files:** `regulatory_coherence.py`

| Finding | Fix | Tests |
|---------|-----|-------|
| KG-V-3 | Add `correlation_sign` to `permutation_null()` + call for neg in analyze_coherence | 3 tests |
| KG-V-4 | Remove dead `_corr_cache` dict | 1 test |
| KG-V-5 | Vectorize edge selection in `build_signed_graphs` | 2 tests |

---

## Status

- [x] Cycle 1: RNG Modernization — 14 tests, 4 files modified
- [x] Cycle 2: Coherence Fixes — 11 tests, 1 file modified
- **Total: 25 new tests, 1442 passed, 1 skipped, 1 pre-existing failure**
