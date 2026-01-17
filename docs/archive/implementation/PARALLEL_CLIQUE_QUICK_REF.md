# Parallel Clique Finding - Quick Reference

## TL;DR
The `find_cliques()` method now supports parallel processing. Default behavior uses 4 worker threads. No code changes required for existing users.

## Quick Examples

### Default (Parallel with 4 workers)
```python
cliques = validator.find_cliques(genes, 'CASE')  # Uses n_workers=4 by default
```

### Custom Worker Count
```python
cliques = validator.find_cliques(genes, 'CASE', n_workers=8)  # 8 threads
```

### Sequential (Old Behavior)
```python
cliques = validator.find_cliques(genes, 'CASE', n_workers=1)  # No parallelism
```

## When to Adjust `n_workers`

| Scenario | Recommended `n_workers` |
|----------|------------------------|
| **Many small gene sets** | 4-8 (default 4 is good) |
| **Few large gene sets** | 2-4 (diminishing returns) |
| **Single-threaded environment** | 1 (force sequential) |
| **High-core-count machine** | 8-16 (if many components) |
| **Debugging** | 1 (easier to trace) |

## Performance Expectations

### Good Speedup Scenarios
- ✓ Many independent connected components (10+)
- ✓ Components of similar size
- ✓ Large gene sets (500+ genes)
- ✓ High min_correlation (sparse graphs)

### Limited Speedup Scenarios
- ✗ Single large component (use `n_workers=1`)
- ✗ Very small gene sets (<50 genes)
- ✗ Low min_correlation (dense graphs)
- ✗ Already very fast (<100ms total)

## API Changes

### `find_cliques()`
```python
def find_cliques(
    genes,
    condition,
    min_correlation=0.7,
    min_clique_size=3,
    method="max",
    max_cliques=10000,
    timeout_seconds=None,
    exact=False,
    n_workers=4,  # NEW: Number of parallel workers
)
```

### `get_child_set_type2()`
```python
def get_child_set_type2(
    regulator_name,
    indra_targets,
    condition,
    min_correlation=0.7,
    method="pearson",
    use_fast_maximum=True,
    n_workers=4,  # NEW: Used when use_fast_maximum=False
)
```

### `find_differential_cliques()`
```python
def find_differential_cliques(
    genes,
    case_condition="CASE",
    ctrl_condition="CTRL",
    min_correlation=0.7,
    min_clique_size=3,
    method="pearson",
    n_workers=4,  # NEW: Applies to both case and control
)
```

### `find_differential_cliques_with_stats()`
```python
def find_differential_cliques_with_stats(
    genes,
    case_condition="CASE",
    ctrl_condition="CTRL",
    min_correlation=None,
    min_clique_size=3,
    method="pearson",
    fdr_threshold=0.05,
    use_adaptive_threshold=True,
    n_workers=4,  # NEW: Applies to clique finding
)
```

## Monitoring Performance

### Enable Debug Logging
```python
import logging
logging.getLogger('cliquefinder.knowledge.clique_validator').setLevel(logging.DEBUG)

# Now you'll see:
# DEBUG: Processing 15 components with 4 workers
# DEBUG: Parallel processing completed in 2.34s: 1247 cliques from 15 components
```

### Measure Speedup
```python
import time

# Sequential
start = time.time()
cliques_seq = validator.find_cliques(genes, 'CASE', n_workers=1)
time_seq = time.time() - start

# Parallel
start = time.time()
cliques_par = validator.find_cliques(genes, 'CASE', n_workers=4)
time_par = time.time() - start

print(f"Speedup: {time_seq / time_par:.2f}x")
```

## Troubleshooting

### "No speedup observed"
- Check number of components: May have only 1-2 components
- Try increasing gene set size
- Increase `min_correlation` to create sparser graph
- Monitor with debug logging

### "Results differ between sequential and parallel"
- This should NEVER happen - file a bug report!
- Verify with:
```python
seq = {frozenset(c.genes) for c in cliques_seq}
par = {frozenset(c.genes) for c in cliques_par}
assert seq == par
```

### "High memory usage"
- Reduce `n_workers` (each thread processes components)
- Increase `min_correlation` (fewer edges, smaller subgraphs)
- Use `max_cliques` limit

## Backward Compatibility

✓ **100% Backward Compatible**
- Existing code works without changes
- Default `n_workers=4` provides automatic speedup
- Sequential behavior available with `n_workers=1`

## Implementation Details

### Algorithm
1. Build correlation graph (sequential)
2. Degree pruning (sequential)
3. **Extract connected components** (sequential)
4. **Process components in parallel** (NEW!)
   - Each component is independent
   - `nx.find_cliques()` releases GIL
   - Results collected thread-safely
5. Sort and return (sequential)

### Thread Safety
- ✓ No locks required (independent components)
- ✓ Graph is read-only after construction
- ✓ `list.extend()` is atomic in CPython
- ✓ Timeout uses shared read-only `start_time`

### Load Balancing
- Components sorted by size **descending** (largest first)
- Prevents end-of-execution stragglers
- ThreadPoolExecutor work-stealing handles remainder

## Related Documentation
- `PARALLEL_CLIQUE_IMPLEMENTATION.md` - Full implementation details
- `docs/CLIQUE_OPTIMIZATION_PLAN.md` - Full optimization roadmap
