# Permutation Testing Optimization Specification

## Executive Summary

**Current State**: 1,777,000 sequential model fits → **~14.8 hours** estimated
**Target State**: GPU-batched computation → **~30 seconds** (300x speedup)

## Problem Analysis

### Data Dimensions
```
Proteins:     3,264
Samples:      379
Cliques:      1,777
Permutations: 1,000
─────────────────────
Total fits:   1,777,000
```

### Clique Size Distribution (Critical for Batching)
```
Size  Count   %     Cumulative
───────────────────────────────
3     831     47%   47%
4     393     22%   69%
5     193     11%   80%
6     124     7%    87%
7     75      4%    91%
8-24  161     9%    100%
```

**Key Insight**: 80% of cliques have size ≤5. Only 18 unique sizes total.
This enables efficient batching by size class.

### Current Implementation Bottlenecks

```python
# Location: src/cliquefinder/stats/clique_analysis.py:966-985
for perm_idx in range(n_permutations):        # 1000 iterations
    for clique_id, size in clique_sizes.items():  # 1777 cliques
        random_genes = np.random.choice(...)      # CPU random
        result = analyze_gene_set(random_genes)   # Full pipeline:
            # 1. Extract protein data
            # 2. Tukey's Median Polish (iterative)
            # 3. Build design matrix
            # 4. Fit linear model
            # 5. Compute t-statistic
```

**Issues**:
1. Sequential outer loops (no parallelism)
2. `n_jobs` parameter accepted but never used
3. Redundant design matrix construction (1.7M times)
4. No GPU utilization despite MLX available
5. No batching despite identical operations

---

## Optimization Architecture

### Phase 1: Precomputation (One-Time, ~1 second)

```
┌─────────────────────────────────────────────────────────────┐
│ PRECOMPUTATION                                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Design Matrix X          (379, n_params)     → GPU      │
│ 2. (X'X)⁻¹                   (n_params, n_params) → GPU    │
│ 3. Contrast Vector c         (n_params,)         → GPU      │
│ 4. Random Gene Indices       (1777, 1000, max_k) → CPU      │
│ 5. Protein-to-Index Map      dict                → CPU      │
└─────────────────────────────────────────────────────────────┘
```

**Random Sampling Strategy**:
```python
# Vectorized pre-generation of ALL random samples
# Shape: (n_cliques, n_permutations, max_clique_size)
rng = np.random.Generator(np.random.PCG64(seed))
all_random_indices = np.zeros((n_cliques, n_perms, max_size), dtype=np.int32)

for size_idx, size in enumerate(unique_sizes):
    clique_mask = clique_sizes == size
    n_cliques_this_size = clique_mask.sum()
    # Batch generate: (n_cliques * n_perms) samples of size k
    samples = rng.choice(pool_size, size=(n_cliques_this_size * n_perms, size), replace=False)
    all_random_indices[clique_mask, :, :size] = samples.reshape(n_cliques_this_size, n_perms, size)
```

### Phase 2: Batched Summarization (GPU, ~10 seconds)

**Tukey's Median Polish** can be vectorized across the batch dimension:

```
┌──────────────────────────────────────────────────────────────┐
│ BATCHED MEDIAN POLISH (per size class)                       │
├──────────────────────────────────────────────────────────────┤
│ Input:  protein_data[batch, k, n_samples]  (MLX array)       │
│ Output: summarized[batch, n_samples]                          │
│                                                               │
│ Algorithm (vectorized):                                       │
│   grand_effect = 0                                            │
│   for iter in range(max_iter):                               │
│       row_medians = mx.median(data, axis=2, keepdims=True)   │
│       data = data - row_medians                               │
│       col_medians = mx.median(data, axis=1, keepdims=True)   │
│       data = data - col_medians                               │
│       grand = mx.median(data.reshape(batch, -1), axis=1)     │
│       grand_effect += grand                                   │
│       data = data - grand[:, None, None]                     │
│   return grand_effect[:, None] + col_medians.squeeze(1)      │
└──────────────────────────────────────────────────────────────┘
```

**Processing Order** (by size, largest batches first):
```
Size 3: 831 cliques × 1000 perms = 831,000 → batch of 831K
Size 4: 393 cliques × 1000 perms = 393,000 → batch of 393K
Size 5: 193 cliques × 1000 perms = 193,000 → batch of 193K
...
```

**Memory Management**: Process in chunks of 100 permutations:
- Per chunk: 177,700 × 379 × 4 bytes = 269 MB (fits GPU)

### Phase 3: Batched OLS (GPU, ~5 seconds)

Once summarized, ALL models share the same structure:

```
┌──────────────────────────────────────────────────────────────┐
│ BATCHED OLS (single GPU operation)                           │
├──────────────────────────────────────────────────────────────┤
│ Y: (n_total, n_samples)  where n_total = 1,777,000          │
│ X: (n_samples, n_params) SAME FOR ALL                        │
│                                                               │
│ β = Y @ X @ (X'X)⁻¹'     # (n_total, n_params)              │
│                                                               │
│ Residuals = Y - β @ X'   # (n_total, n_samples)             │
│ RSS = sum(Residuals², axis=1)                                │
│ σ² = RSS / df_residual                                       │
│                                                               │
│ # Contrast testing (vectorized)                              │
│ estimate = β @ c                    # (n_total,)             │
│ se² = σ² × (c' @ (X'X)⁻¹ @ c)      # (n_total,)             │
│ t = estimate / sqrt(se²)            # (n_total,)             │
└──────────────────────────────────────────────────────────────┘
```

**Key Optimization**: The matrix `(X'X)⁻¹` and `c' @ (X'X)⁻¹ @ c` are computed ONCE and reused 1.7M times.

### Phase 4: Empirical P-values (CPU, <1 second)

Already vectorized in current implementation:
```python
# Reshape t-statistics: (n_cliques, n_permutations)
t_observed = t_all[:n_cliques]  # First entry per clique is observed
t_null = t_all.reshape(n_cliques, n_permutations + 1)[:, 1:]

# Vectorized comparison
n_extreme = np.sum(np.abs(t_null) >= np.abs(t_observed[:, None]), axis=1)
empirical_pvalue = (n_extreme + 1) / (n_permutations + 1)
```

---

## Memory Layout & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY FLOW DIAGRAM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Protein Data │    │   Random     │    │   Design     │              │
│  │ (3264, 379)  │    │   Indices    │    │   Matrix X   │              │
│  │   49 MB      │    │   (precomp)  │    │  (379, p)    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   │                       │
│  ┌──────────────────────────────────┐          │                       │
│  │  Batched Gather (per size)       │          │                       │
│  │  protein_data[indices]           │          │                       │
│  │  → (batch, k, 379)               │          │                       │
│  └──────────────┬───────────────────┘          │                       │
│                 │                               │                       │
│                 ▼                               │                       │
│  ┌──────────────────────────────────┐          │                       │
│  │  Batched Median Polish (GPU)     │          │                       │
│  │  (batch, k, 379) → (batch, 379)  │          │                       │
│  └──────────────┬───────────────────┘          │                       │
│                 │                               │                       │
│                 ▼                               ▼                       │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │  Batched OLS (GPU)                                      │           │
│  │  Y @ X @ (X'X)⁻¹' → β                                   │           │
│  │  → t-statistics (1.7M values)                           │           │
│  └──────────────┬──────────────────────────────────────────┘           │
│                 │                                                       │
│                 ▼                                                       │
│  ┌──────────────────────────────────┐                                  │
│  │  Reshape & Empirical P-values    │                                  │
│  │  (n_cliques, n_perms) → p-vals   │                                  │
│  └──────────────────────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Wall Clock Time Estimates

### Current Sequential Implementation
```
Operation                    Time per op    Total ops     Total time
─────────────────────────────────────────────────────────────────────
Random sampling              0.01 ms        1,777,000     17.8 sec
Data extraction              0.1 ms         1,777,000     177.7 sec
Median Polish                5 ms           1,777,000     8,885 sec (2.5 hr)
Design matrix build          0.5 ms         1,777,000     888.5 sec
Model fit (statsmodels)      20 ms          1,777,000     35,540 sec (9.9 hr)
t-statistic extraction       0.1 ms         1,777,000     177.7 sec
─────────────────────────────────────────────────────────────────────
TOTAL                                                     ~14.8 hours
```

### Optimized GPU Implementation
```
Operation                    Strategy            Estimated time
─────────────────────────────────────────────────────────────────────
Precomputation              One-time            0.5 sec
Random index generation     Vectorized NumPy    0.2 sec
Batched data gather         GPU indexing        2 sec
Batched Median Polish       GPU (18 batches)    8 sec
Batched OLS                 GPU matmul          3 sec
Empirical p-values          Vectorized NumPy    0.3 sec
─────────────────────────────────────────────────────────────────────
TOTAL                                           ~15 seconds
```

### Speedup: **~3,500x** (14.8 hours → 15 seconds)

---

## Implementation Modules

### New Module: `src/cliquefinder/stats/permutation_gpu.py`

```python
"""
GPU-accelerated permutation testing for clique differential abundance.

Uses MLX (Metal) for batched:
- Tukey's Median Polish summarization
- OLS regression
- t-statistic computation

Falls back to parallel CPU if GPU unavailable.
"""

# Core functions to implement:
def precompute_permutation_data(...)
def batched_median_polish_gpu(...)
def batched_ols_gpu(...)
def run_permutation_test_gpu(...)
```

### Integration Points

1. **CLI** (`differential.py`):
   ```python
   if args.permutation_test:
       if MLX_AVAILABLE and not args.force_cpu:
           results = run_permutation_test_gpu(...)
       else:
           results = run_permutation_clique_test(...)  # existing
   ```

2. **Fallback** (parallel CPU):
   - Use joblib to parallelize across permutations
   - Pre-aggregate subjects for pseudo-mixed-model
   - Expected time: ~2-3 minutes with 6 workers

---

## Mixed Model Considerations

**Problem**: True mixed models require iterative REML, killing batching.

**Solutions**:

1. **Subject Aggregation** (Recommended):
   ```python
   # Pre-aggregate to subject level before permutation
   # This is statistically valid and enables batched OLS
   subject_means = data.groupby(subject_id).mean()
   ```

2. **Satterthwaite Approximation**:
   - Compute variance components from observed data
   - Apply correction to df in batched t-test

3. **Permutation of Residuals**:
   - Fit mixed model once on observed
   - Permute residuals for null (preserves correlation structure)

For this implementation, we use **Option 1** with subject aggregation.

---

## Testing & Validation

### Numerical Equivalence Test
```python
def test_gpu_cpu_equivalence():
    """Verify GPU implementation matches CPU within tolerance."""
    # Run on small subset (10 cliques, 10 permutations)
    cpu_results = run_permutation_clique_test(n_permutations=10, ...)
    gpu_results = run_permutation_test_gpu(n_permutations=10, ...)

    # Compare t-statistics
    assert np.allclose(cpu_t, gpu_t, rtol=1e-5)

    # Compare empirical p-values (may differ due to random sampling)
    # Use same seed and verify exact match
```

### Performance Benchmark
```python
def benchmark_permutation_test():
    """Measure wall-clock time for each component."""
    # Time precomputation
    # Time summarization (per size class)
    # Time OLS
    # Time p-value computation
    # Report total and breakdown
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MLX not available | Fallback to joblib parallel CPU |
| GPU memory overflow | Chunk processing (100 perms/chunk) |
| Numerical instability | Use float64, add regularization |
| Different results | Validation tests with tolerance |
| Mixed model loss | Subject aggregation preserves validity |

---

## Deliverables

1. **`permutation_gpu.py`**: Core GPU implementation
2. **`test_permutation_gpu.py`**: Validation and benchmarks
3. **CLI integration**: `--gpu/--no-gpu` flags
4. **Documentation**: Usage examples

## Delegation Notes

**For Sonnet Implementation Agents**:

1. **Agent 1: GPU Kernel Specialist**
   - Implement `batched_median_polish_gpu()`
   - Handle MLX array operations
   - Memory chunking logic

2. **Agent 2: Statistical Methods Specialist**
   - Implement `batched_ols_gpu()` with proper SE calculation
   - Subject aggregation for mixed model approximation
   - Satterthwaite df correction

3. **Agent 3: Integration Specialist**
   - Wire up CLI options
   - Fallback logic
   - Result format compatibility

Each agent should read this spec and the existing implementation at:
- `src/cliquefinder/stats/clique_analysis.py:750-1062` (current permutation test)
- `src/cliquefinder/stats/differential.py:374-568` (batched OLS reference)
