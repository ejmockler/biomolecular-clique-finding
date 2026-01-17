# Exact Clique Enumeration Optimization Plan

## Current State Analysis

### Bottlenecks Identified

1. **Sequential Component Processing** (Lines 1003-1040 in clique_validator.py)
   - Connected components processed in serial loop
   - Each component's `nx.find_cliques()` blocks until complete
   - No work distribution across cores

2. **Naive Graph Construction** (Lines 903-909)
   - O(n²) Python loop with DataFrame indexing
   - No vectorization or numpy broadcasting
   - Edge-by-edge insertion into NetworkX graph

3. **Single-Threaded Bron-Kerbosch** (Line 1016)
   - NetworkX's `find_cliques()` is pure Python + C backend
   - No parallelization within the algorithm
   - Pivot selection is sequential

4. **Insufficient Pruning** (Lines 980-985)
   - Only degree-based pruning implemented
   - Missing k-core decomposition
   - No vertex ordering optimization

5. **Suboptimal Data Structures**
   - NetworkX graph has Python object overhead
   - Edge weights stored redundantly
   - No adjacency matrix caching for dense graphs

---

## Optimization Strategies

### OPT-1: Parallel Connected Component Processing
**Complexity:** Low | **Impact:** High | **Risk:** Low

**Current Code:**
```python
for component in components:
    for clique_nodes in nx.find_cliques(subgraph):
        # sequential enumeration
```

**Optimized Approach:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def enumerate_component(component, G, min_clique_size):
    subgraph = G.subgraph(component)
    cliques = []
    for clique_nodes in nx.find_cliques(subgraph):
        if len(clique_nodes) >= min_clique_size:
            cliques.append(clique_nodes)
    return cliques

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {
        executor.submit(enumerate_component, comp, G, min_clique_size): comp
        for comp in components if len(comp) >= min_clique_size
    }
    for future in as_completed(futures):
        cliques.extend(future.result())
```

**Implementation Notes:**
- Use ThreadPoolExecutor (GIL released during NetworkX C operations)
- Sort components by size descending for better load balancing
- Add per-component timeout to prevent single component blocking
- Aggregate results with thread-safe collection

---

### OPT-2: K-Core Decomposition Pruning
**Complexity:** Medium | **Impact:** High | **Risk:** Low

**Theory:**
A k-clique requires all vertices to have degree ≥ k-1. The k-core of a graph is the maximal subgraph where all vertices have degree ≥ k. Iteratively computing the (min_clique_size-1)-core removes vertices that cannot participate in any valid clique.

**Implementation:**
```python
def kcore_pruning(G: nx.Graph, min_clique_size: int) -> nx.Graph:
    """Iteratively remove vertices with degree < min_clique_size - 1."""
    k = min_clique_size - 1
    # NetworkX has efficient k-core implementation
    core = nx.k_core(G, k=k)
    return core
```

**Expected Speedup:**
- For sparse graphs: 2-5x reduction in graph size
- For dense graphs: Minimal effect (most nodes already high-degree)
- Always beneficial, never harmful

---

### OPT-3: Degeneracy Ordering (Vertex Ordering Optimization)
**Complexity:** Medium | **Impact:** Medium-High | **Risk:** Low

**Theory:**
The degeneracy of a graph is the smallest k such that every subgraph has a vertex of degree ≤ k. Processing vertices in degeneracy order reduces the branching factor of Bron-Kerbosch.

**Implementation:**
```python
def degeneracy_ordering(G: nx.Graph) -> List[str]:
    """Compute degeneracy ordering for Bron-Kerbosch optimization."""
    ordering = []
    H = G.copy()
    while H.nodes():
        # Find minimum degree vertex
        min_node = min(H.nodes(), key=lambda n: H.degree(n))
        ordering.append(min_node)
        H.remove_node(min_node)
    return ordering
```

**Integration with Bron-Kerbosch:**
- Pass ordering to modified algorithm
- Process vertices in this order for R-extension
- Proven to give O(d * n * 3^(d/3)) time complexity where d = degeneracy

---

### OPT-4: Tomita's Algorithm for Dense Graphs
**Complexity:** High | **Impact:** High | **Risk:** Medium

**Theory:**
Tomita et al. (2006) algorithm with pivot selection achieves optimal O(3^(n/3)) worst-case for maximal clique enumeration. Key insight: choose pivot to maximize |P ∩ N(pivot)| to minimize branching.

**Implementation Sketch:**
```python
def tomita_bron_kerbosch(R: set, P: set, X: set, G: nx.Graph, cliques: list):
    """Bron-Kerbosch with Tomita pivot selection."""
    if not P and not X:
        cliques.append(R.copy())
        return

    # Tomita pivot: maximize |P ∩ N(u)| over u in P ∪ X
    pivot = max(P | X, key=lambda u: len(P & set(G.neighbors(u))))

    # Only branch on P - N(pivot)
    for v in P - set(G.neighbors(pivot)):
        neighbors_v = set(G.neighbors(v))
        tomita_bron_kerbosch(
            R | {v},
            P & neighbors_v,
            X & neighbors_v,
            G, cliques
        )
        P.remove(v)
        X.add(v)
```

**Trade-offs:**
- Better for dense graphs (our case with min_correlation=0.4)
- Requires custom implementation (can't use nx.find_cliques)
- More complex to parallelize

---

### OPT-5: Vectorized Graph Construction
**Complexity:** Low | **Impact:** Medium | **Risk:** Low

**Current (Slow):**
```python
for i, g1 in enumerate(corr.index):
    for j, g2 in enumerate(corr.columns):
        if i < j and abs(corr.iloc[i, j]) >= min_correlation:
            G.add_edge(g1, g2, weight=corr.iloc[i, j])
```

**Optimized:**
```python
def build_correlation_graph_vectorized(corr: pd.DataFrame, min_correlation: float) -> nx.Graph:
    """Vectorized graph construction using numpy."""
    corr_values = corr.values
    genes = list(corr.index)
    n = len(genes)

    # Get upper triangle indices where |corr| >= threshold
    i_upper, j_upper = np.triu_indices(n, k=1)
    mask = np.abs(corr_values[i_upper, j_upper]) >= min_correlation

    # Extract edges
    i_edges = i_upper[mask]
    j_edges = j_upper[mask]
    weights = corr_values[i_edges, j_edges]

    # Build graph from edge list (much faster than add_edge loop)
    G = nx.Graph()
    G.add_nodes_from(genes)
    edges = [(genes[i], genes[j], {'weight': w})
             for i, j, w in zip(i_edges, j_edges, weights)]
    G.add_edges_from(edges)

    return G
```

**Expected Speedup:** 10-50x for graph construction phase

---

### OPT-6: Numba JIT Compilation of Core Loops
**Complexity:** High | **Impact:** Very High | **Risk:** Medium

**Target Functions:**
1. Adjacency checking in clique extension
2. Pivot selection loop
3. Correlation statistics computation

**Example:**
```python
from numba import njit
import numpy as np

@njit(cache=True)
def find_cliques_numba(adj_matrix: np.ndarray, min_size: int) -> List[np.ndarray]:
    """Numba-accelerated Bron-Kerbosch implementation."""
    n = adj_matrix.shape[0]
    cliques = []

    # ... Bron-Kerbosch implementation using integer arrays
    # ... instead of Python sets

    return cliques
```

**Considerations:**
- Requires converting graph to adjacency matrix (dense representation)
- Set operations replaced with bitmask operations
- 10-100x speedup potential for hot loops
- More complex debugging

---

### OPT-7: Work-Stealing Parallel Bron-Kerbosch
**Complexity:** Very High | **Impact:** Very High | **Risk:** High

**Theory:**
Parallelize at the recursion tree level. Each recursive branch can be processed independently. Use work-stealing scheduler for load balancing.

**Implementation Strategy:**
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_bron_kerbosch(G: nx.Graph, n_workers: int, min_clique_size: int):
    """Parallel Bron-Kerbosch with work-stealing."""

    # Generate initial work items (top-level branches)
    initial_branches = []
    P = set(G.nodes())
    X = set()

    for v in list(P):
        neighbors_v = set(G.neighbors(v))
        initial_branches.append({
            'R': {v},
            'P': P & neighbors_v,
            'X': X & neighbors_v
        })
        P.remove(v)
        X.add(v)

    # Process branches in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(bk_worker, branch, G, min_clique_size)
            for branch in initial_branches
        ]

        all_cliques = []
        for future in as_completed(futures):
            all_cliques.extend(future.result())

    return all_cliques
```

**Challenges:**
- Graph must be serializable (pickle) for ProcessPoolExecutor
- Work imbalance if branches have very different sizes
- Need adaptive granularity control

---

### OPT-8: Sparse Adjacency Matrix with CSR Format
**Complexity:** Medium | **Impact:** Medium | **Risk:** Low

**For sparse graphs (high min_correlation):**
```python
from scipy.sparse import csr_matrix

def build_sparse_adjacency(corr: pd.DataFrame, min_correlation: float):
    """Build CSR sparse adjacency matrix."""
    mask = np.abs(corr.values) >= min_correlation
    np.fill_diagonal(mask, False)  # No self-loops

    adj = csr_matrix(mask.astype(np.int8))
    return adj, list(corr.index)
```

**Benefits:**
- O(1) neighbor lookup via CSR indices
- Memory efficient for sparse graphs
- Compatible with scipy.sparse.csgraph algorithms

---

## Implementation Priority Matrix

| Optimization | Complexity | Impact | Dependencies | Priority |
|-------------|------------|--------|--------------|----------|
| OPT-1: Parallel Components | Low | High | None | **P0** |
| OPT-5: Vectorized Graph Build | Low | Medium | None | **P0** |
| OPT-2: K-Core Pruning | Medium | High | None | **P1** |
| OPT-3: Degeneracy Ordering | Medium | Medium-High | OPT-4 | **P1** |
| OPT-8: Sparse CSR | Medium | Medium | None | **P2** |
| OPT-4: Tomita Algorithm | High | High | OPT-3 | **P2** |
| OPT-6: Numba JIT | High | Very High | OPT-4 | **P3** |
| OPT-7: Parallel BK | Very High | Very High | OPT-4, OPT-6 | **P3** |

---

## Subagent Delegation Plan

### Agent 1: Graph Construction Optimizer
**Background Context:**
- NumPy vectorization expert
- Sparse matrix specialist (scipy.sparse)
- NetworkX internals knowledge

**Tasks:**
1. Implement OPT-5 (vectorized graph construction)
2. Implement OPT-8 (CSR sparse adjacency)
3. Add benchmarking harness

**Files to Modify:**
- `src/cliquefinder/knowledge/clique_validator.py` (build_correlation_graph)

---

### Agent 2: Parallel Execution Specialist
**Background Context:**
- concurrent.futures expert
- Work-stealing scheduler design
- GIL-aware parallelism patterns
- Thread-safe data structures

**Tasks:**
1. Implement OPT-1 (parallel component processing)
2. Add per-component timeout handling
3. Implement progress reporting with thread-safe counters
4. Design work-stealing for unbalanced components

**Files to Modify:**
- `src/cliquefinder/knowledge/clique_validator.py` (find_cliques)

---

### Agent 3: Algorithm Specialist
**Background Context:**
- Graph theory algorithms (Bron-Kerbosch variants)
- Combinatorial optimization
- K-core decomposition
- Degeneracy theory

**Tasks:**
1. Implement OPT-2 (k-core pruning)
2. Implement OPT-3 (degeneracy ordering)
3. Implement OPT-4 (Tomita pivot selection)
4. Validate correctness against nx.find_cliques

**Files to Modify:**
- `src/cliquefinder/knowledge/clique_validator.py`
- New file: `src/cliquefinder/knowledge/clique_algorithms.py`

---

### Agent 4: Performance Engineer (Numba/Cython)
**Background Context:**
- Numba JIT compilation
- Low-level optimization
- Bitmask operations for set membership
- Cache-aware algorithms

**Tasks:**
1. Implement OPT-6 (Numba-accelerated core)
2. Design adjacency matrix representation for Numba
3. Benchmark against pure Python implementation

**Files to Create:**
- `src/cliquefinder/knowledge/clique_numba.py`

---

## Testing Strategy

### Correctness Tests
```python
def test_clique_correctness():
    """Verify optimized implementation matches NetworkX baseline."""
    G = generate_random_graph(n=50, p=0.3)

    baseline = set(frozenset(c) for c in nx.find_cliques(G))
    optimized = set(frozenset(c) for c in optimized_find_cliques(G))

    assert baseline == optimized, "Clique sets must match exactly"
```

### Performance Benchmarks
```python
@pytest.mark.benchmark
def test_performance_scaling():
    """Benchmark across graph sizes and densities."""
    for n in [50, 100, 200, 500]:
        for p in [0.1, 0.3, 0.5, 0.7]:
            G = nx.erdos_renyi_graph(n, p)

            t_baseline = timeit(lambda: list(nx.find_cliques(G)), number=3)
            t_optimized = timeit(lambda: optimized_find_cliques(G), number=3)

            print(f"n={n}, p={p}: {t_baseline/t_optimized:.2f}x speedup")
```

---

## Success Metrics

1. **Correctness:** 100% match with NetworkX baseline on all test cases
2. **Speedup:** ≥5x on typical regulator target graphs (70-100 genes, ρ≥0.4)
3. **Scalability:** Linear speedup with worker count up to 8 cores
4. **Memory:** No more than 2x baseline memory usage
5. **Timeout Reduction:** ≥80% fewer timeouts on exact enumeration

---

## Risk Mitigation

1. **Correctness Bugs:** Comprehensive test suite against nx.find_cliques
2. **Race Conditions:** Use thread-safe collections, no shared mutable state
3. **Memory Blowup:** Monitor memory during parallel execution, add limits
4. **Numba Complexity:** Keep pure Python fallback for debugging
5. **Integration Issues:** Feature flag to enable/disable each optimization
