# Batched OLS Quick Reference

## Three Core Functions

### 1. Precompute OLS Matrices

```python
from cliquefinder.stats.permutation_gpu import precompute_ols_matrices

matrices = precompute_ols_matrices(
    sample_condition=metadata['phenotype'],    # Array of condition labels
    conditions=['CASE', 'CTRL'],               # List of unique conditions
    contrast=('CASE', 'CTRL'),                 # Tuple: (test, reference)
    regularization=1e-8                        # Ridge regularization
)
```

**Returns**: `OLSPrecomputedMatrices` with:
- `X`: Design matrix (n_samples, n_params)
- `XtX_inv`: (X'X)^-1 matrix
- `c`: Contrast vector
- `c_var_factor`: SE scaling factor
- `df_residual`: Degrees of freedom

### 2. Batched OLS Contrast Test

```python
from cliquefinder.stats.permutation_gpu import batched_ols_contrast_test

t_stats = batched_ols_contrast_test(
    Y=Y_summarized,           # (n_total, n_samples) response matrix
    matrices=matrices,        # Precomputed matrices
    use_gpu=True,            # Use MLX if available
    chunk_size=100000        # Optional: process in chunks
)
```

**Input**:
- `Y` shape: (n_cliques * n_permutations, n_samples)
- Each row is one statistical test

**Output**:
- Array of t-statistics (n_total,)

### 3. Aggregate to Subject Level

```python
from cliquefinder.stats.permutation_gpu import aggregate_to_subject_level

data_agg, subjects = aggregate_to_subject_level(
    data=protein_data,       # (n_features, n_samples)
    subject_ids=subject_ids, # Array of subject IDs
    method='mean'            # 'mean' or 'median'
)
```

**Output**:
- `data_agg`: (n_features, n_subjects)
- `subjects`: Array of unique subject IDs

## Complete Workflow

```python
# Step 1: Aggregate to subject level (if repeated measures)
data_subject, subjects = aggregate_to_subject_level(
    data_sample, subject_ids, method='mean'
)

# Step 2: Map subjects to conditions
subject_to_condition = dict(zip(subject_ids, sample_conditions))
subject_conditions = np.array([subject_to_condition[s] for s in subjects])

# Step 3: Precompute OLS matrices (once)
conditions = ['CASE', 'CTRL']
contrast = ('CASE', 'CTRL')
matrices = precompute_ols_matrices(subject_conditions, conditions, contrast)

# Step 4: Create Y matrix (from permutations/cliques)
# Each row is one test: (n_cliques * n_permutations, n_subjects)
Y = ...  # From batched median polish or sampling

# Step 5: Compute t-statistics
t_stats = batched_ols_contrast_test(Y, matrices)

# Step 6: Compute empirical p-values
t_stats_reshaped = t_stats.reshape(n_cliques, n_permutations)
t_observed = t_stats_reshaped[:, 0]  # First permutation is observed
t_null = t_stats_reshaped[:, 1:]     # Rest are null

n_extreme = np.sum(np.abs(t_null) >= np.abs(t_observed[:, None]), axis=1)
empirical_pvalue = (n_extreme + 1) / (n_permutations + 1)
```

## Key Formulas

### Batched OLS
```
β = Y @ X @ (X'X)^-1'         # Coefficients (n_batch, n_params)
ŷ = β @ X'                    # Predictions (n_batch, n_samples)
e = Y - ŷ                     # Residuals
σ² = sum(e², axis=1) / df     # Residual variance (n_batch,)
```

### Contrast Test
```
estimate = β @ c              # (n_batch,)
SE = sqrt(σ² * c_var_factor)  # (n_batch,)
t = estimate / SE             # (n_batch,)
```

where `c_var_factor = c' @ (X'X)^-1 @ c` is precomputed once.

## Performance

**Benchmark** (5,000 tests):
- Time: 0.002s
- Throughput: 2,048,480 tests/sec
- Memory: 3.8 MB

**Full Scale** (1,777,000 tests):
- Estimated time: 0.9s
- Sequential baseline: 12.3 hours
- **Speedup: 51,212x**

## Validation

Built-in validation:
```python
from cliquefinder.stats.permutation_gpu import validate_ols_implementation

metrics = validate_ols_implementation()
print(metrics)
# {'max_beta_diff': 9.44e-10, 'max_t_diff': 4.62e-09, 'all_close': True}
```

## Common Patterns

### With Repeated Measures
```python
# Aggregate first
data_agg, subjects = aggregate_to_subject_level(data, subject_ids)
# Map subjects to conditions
subject_conditions = map_subjects_to_conditions(subjects, metadata)
# Precompute with subject-level data
matrices = precompute_ols_matrices(subject_conditions, conditions, contrast)
```

### Without Repeated Measures
```python
# Use sample-level data directly
matrices = precompute_ols_matrices(
    sample_conditions, conditions, contrast
)
```

### GPU vs CPU
```python
# Auto-select (GPU if available)
t_stats = batched_ols_contrast_test(Y, matrices, use_gpu=True)

# Force CPU
t_stats = batched_ols_contrast_test(Y, matrices, use_gpu=False)
```

### Memory Management
```python
# Process in chunks for large datasets
t_stats = batched_ols_contrast_test(
    Y, matrices,
    chunk_size=100000  # Process 100K tests at a time
)
```

## Error Messages

```python
# Dimension mismatch
ValueError: Sample dimension mismatch: Y has 379, X has 100

# Invalid contrast
ValueError: Contrast conditions ('X', 'Y') not found in ['CASE', 'CTRL']

# Insufficient samples
ValueError: Insufficient samples after removing NaN: 2

# Insufficient df
ValueError: Insufficient df: 10 samples - 10 parameters = 0
```

## Files

- **Implementation**: `src/cliquefinder/stats/permutation_gpu.py` (812 lines)
- **Documentation**: `docs/BATCHED_OLS_IMPLEMENTATION.md`
- **Tests**: `tests/test_permutation_gpu.py`

## See Also

- `PERMUTATION_OPTIMIZATION_SPEC.md` - Full optimization architecture
- `differential.py` - Reference implementation for single-feature OLS
- `summarization.py` - Tukey's Median Polish for clique summarization
