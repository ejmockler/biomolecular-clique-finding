# Platform Architecture

## Conceptual Model

```
┌──────────────────────────────────────────────────────────────┐
│                    USER / ANALYST                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Pipeline / Scripts   │  ← High-level workflows
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │      Transforms        │  ← Composable operations
        │  - OutlierDetector     │
        │  - Imputer             │
        │  - Normalizer          │
        │  - CliqueDetector      │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │      BioMatrix         │  ← Core data structure
        │  data + metadata +     │
        │  quality flags +       │
        │  provenance            │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │    I/O Layer           │  ← Format adapters
        │  CSV, HDF5, AnnData    │
        └────────────────────────┘
```

## Data Flow: Outlier Imputation Pipeline

```
Raw CSV File
    │
    │ MatrixLoader.from_csv()
    ▼
BioMatrix
│ data: [60665 x 578]
│ quality_flags: all ORIGINAL
│
    │ OutlierDetector.apply()
    │   (MAD-Z threshold = 3.5)
    ▼
BioMatrix
│ data: [60665 x 578] (unchanged)
│ quality_flags: some OUTLIER_DETECTED
│
    │ Imputer.apply()
    │   (KNN strategy, k=5)
    ▼
BioMatrix
│ data: [60665 x 578] (outliers replaced)
│ quality_flags: OUTLIER_DETECTED | IMPUTED
│
    │ MatrixWriter.to_csv()
    ▼
Two CSV Files:
  - imputed.data.csv (corrected values)
  - imputed.flags.csv (0=original, 3=outlier+imputed)
```

## Module Dependencies

```
scripts/
  │
  ├─→ biocore.io (load/save)
  │
  └─→ biocore.quality (transforms)
        │
        └─→ biocore.core (BioMatrix, Transform)
```

No circular dependencies. Clean layering.

## Transform Composition

Transforms are composable through two mechanisms:

### 1. Sequential Application
```python
matrix = load_csv_matrix("data.csv")
matrix = OutlierDetector(threshold=3.5).apply(matrix)
matrix = Imputer(strategy="knn").apply(matrix)
matrix = Normalizer(method="tmm").apply(matrix)
write_csv_matrix(matrix, "processed.csv")
```

### 2. Pipeline (Phase 3)
```python
pipeline = Pipeline("preprocess")
pipeline.add(OutlierDetector(threshold=3.5))
pipeline.add(Imputer(strategy="knn"))
pipeline.add(Normalizer(method="tmm"))

matrix = load_csv_matrix("data.csv")
result = pipeline.execute(matrix)
```

## Quality Flag Propagation

Quality flags are **accumulated bitwise** through pipeline:

```
Initial state:        0b000000 (ORIGINAL)
After OutlierDetector: 0b000001 (OUTLIER_DETECTED)
After Imputer:         0b000011 (OUTLIER_DETECTED | IMPUTED)
After BatchCorrect:    0b001011 (... | BATCH_CORRECTED)
```

Any downstream analysis can check:
```python
# Was this value modified?
if matrix.quality_flags[i, j] != QualityFlag.ORIGINAL:
    # Handle differently or exclude

# Was this value imputed specifically?
if matrix.quality_flags[i, j] & QualityFlag.IMPUTED:
    # Reduce weight in statistical tests
```

## Graph Analysis Integration (Phase 2)

```
BioMatrix (imputed counts)
    │
    │ select_features(tf_genes)
    ▼
BioMatrix (TFs only)
    │
    │ CliqueDetector.from_matrix()
    │   - Compute correlation matrix
    │   - Build graph (edges = high correlation)
    ▼
NetworkX Graph
    │
    │ CliqueDetector.find_cliques()
    ▼
List[Set[gene_id]]  # Each set is a clique

    │ For each TF, find cliques in target genes
    ▼
Dict[tf_id, List[clique]]
```

**Key Insight:** Graph operations work on BioMatrix but produce different output types (graphs, cliques). They don't need to be Transforms - they're **analyzers** rather than **transformers**.

## Type System

```python
# Core types
BioMatrix         # The fundamental container
Transform         # Abstract base for data transformations
Pipeline          # Ordered sequence of Transforms
ProvenanceGraph   # DAG of Transform applications

# Quality types
QualityFlag       # IntFlag enum (bitwise operations)

# Graph types (Phase 2)
BioGraph          # NetworkX graph with biological metadata
Clique            # Set[feature_id]

# Statistical types (Phase 3+)
DifferentialResult   # DESeq2-style output
TestResult           # Hypothesis test results
```

## Extensibility Points

### 1. New Transform Types
```python
class CustomTransform(Transform):
    def apply(self, matrix: BioMatrix) -> BioMatrix:
        # Your logic here
        return modified_matrix
```

### 2. New I/O Formats
```python
class MatrixLoader:
    @staticmethod
    def from_custom_format(path: Path) -> BioMatrix:
        # Parse format, return BioMatrix
        pass
```

### 3. New Quality Flags
```python
class QualityFlag(IntFlag):
    # Existing flags...
    CUSTOM_FLAG = 64  # Add new flag
```

### 4. New Outlier Detection Methods
```python
class OutlierDetector(Transform):
    def __init__(self, method: str = "mad-z"):
        # Add new method to _detect() dispatcher
```

## Performance Considerations

### Current Scale
- **Dataset:** 60,665 × 578 ≈ 35M values
- **Memory:** ~280 MB for float64 data + ~280 MB for flags = ~560 MB
- **Computation:** MAD-Z on 35M values ≈ 1-2 seconds
- **KNN Imputation:** Scales as O(n_features² × n_samples) ≈ 10-30 seconds

### Future Optimizations (>10M features or >10K samples)

1. **Chunked Processing**
   - Process matrix in row/column chunks
   - Reduce memory footprint
   - Parallelize with joblib

2. **Sparse Matrices**
   - Many RNA-seq datasets are sparse (lots of zeros)
   - Use scipy.sparse for storage
   - Specialized operations that maintain sparsity

3. **Rust Kernels**
   - Outlier detection loop in Rust via PyO3
   - 10-100x speedup for tight loops
   - Still expose Python API

4. **Dask for Distributed**
   - Split matrix across workers
   - For cohorts with 10K+ samples

## Error Handling Strategy

### Validation Levels

**1. Constructor Validation**
```python
class OutlierDetector(Transform):
    def __init__(self, method: str, threshold: float):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if method not in ["mad-z", "iqr"]:
            raise ValueError(f"Unknown method: {method}")
```

**2. Apply-Time Validation**
```python
def apply(self, matrix: BioMatrix) -> BioMatrix:
    if matrix.data.size == 0:
        raise ValueError("Cannot process empty matrix")
    if np.any(np.isnan(matrix.data)):
        raise ValueError("Input contains NaN (use Imputer first)")
```

**3. Runtime Warnings**
```python
import warnings

if n_outliers / total_values > 0.5:
    warnings.warn(
        f"Detected {n_outliers} outliers (>50% of data). "
        "Consider adjusting threshold or checking data quality."
    )
```

### Error Recovery

- **Immutability helps:** Original data never corrupted
- **Checkpointing:** Save intermediate BioMatrix objects
- **Provenance:** Always know what transforms were applied

```python
try:
    matrix = detector.apply(matrix)
except OutlierDetectionError as e:
    # Fall back to previous checkpoint
    matrix = checkpoint["after_normalization"]
    # Try different parameters
    detector = OutlierDetector(threshold=4.0)
    matrix = detector.apply(matrix)
```

## Future Directions

### Short Term (Phases 1-3)
- Complete Phase 1 (outlier imputation) ✓
- Implement graph analysis (Phase 2)
- Build pipeline engine (Phase 3)

### Medium Term
- Differential expression module
- Batch correction (ComBat, Harmony)
- Dimensionality reduction (PCA, UMAP)
- Statistical testing framework

### Long Term
- Web UI for interactive exploration
- GPU acceleration for large matrices
- Integration with public databases (GEO, TCGA)
- Automated report generation
- Cloud deployment (AWS Batch, Google Cloud Life Sciences)

## Design Trade-offs

### Immutability vs. Memory
**Choice:** Immutable BioMatrix instances
**Trade-off:** More memory copies
**Rationale:** Safety, reproducibility, parallelization outweigh memory cost at current scale

### Pure Python vs. Rust/C++
**Choice:** Start with pure Python
**Trade-off:** Slower for tight loops
**Rationale:** Faster development, easier debugging. Optimize bottlenecks later with Rust.

### Type Hints vs. Duck Typing
**Choice:** Strong type hints everywhere
**Trade-off:** More verbose code
**Rationale:** Catches bugs early, better IDE support, easier collaboration.

### Pandas vs. NumPy
**Choice:** NumPy for data, Pandas for metadata
**Trade-off:** Less convenient API for data
**Rationale:** NumPy is faster, lighter, more predictable for numerical ops. Pandas for string/categorical metadata.

### NetworkX vs. graph-tool vs. igraph
**Choice:** NetworkX
**Trade-off:** Slower than graph-tool/igraph
**Rationale:** Pure Python, easier to install, sufficient for current scale. Can swap later if needed.
