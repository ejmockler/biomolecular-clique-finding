# Biomolecular Analysis Platform Specification

## Overview
A composable, type-safe platform for high-dimensional biomolecular data analysis. Built around the core insight that most computational biology workflows follow similar analytical patterns: **transform → track → analyze → interpret**.

**Current Dataset Context:**
- 60,665 genes × 578 samples (ALS proteomic counts)
- Immediate need: outlier imputation → clique detection in TF networks

---

## Core Abstractions

### 1. BioMatrix
The fundamental data structure representing high-dimensional biological measurements with rich metadata.

```python
@dataclass
class BioMatrix:
    """
    Expression/count matrix with full provenance tracking.

    Dimensions: features × samples
    """
    data: np.ndarray                    # Core numerical matrix
    feature_ids: pd.Index               # Gene IDs (ENSG...), protein IDs, etc.
    sample_ids: pd.Index                # Sample identifiers
    feature_metadata: pd.DataFrame      # Gene annotations, biotypes, etc.
    sample_metadata: pd.DataFrame       # Phenotypes (CASE/CTRL), batch, cohort
    quality_flags: np.ndarray           # Per-value quality indicators
    provenance: ProvenanceGraph         # Transformation history

    def select_samples(self, mask: Callable) -> 'BioMatrix'
    def select_features(self, mask: Callable) -> 'BioMatrix'
    def apply(self, transform: Transform) -> 'BioMatrix'
```

**Key Design Principle:** Immutable transformations. Each operation returns a new BioMatrix with updated provenance.

### 2. Transform (Abstract Operation)
All data transformations inherit from this base. Enables composition, testing, and reproducibility.

```python
@dataclass
class Transform(ABC):
    """Base class for all matrix transformations."""
    name: str
    params: Dict[str, Any]
    timestamp: datetime

    @abstractmethod
    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """Execute transformation, return new matrix."""
        pass

    @abstractmethod
    def inverse(self, matrix: BioMatrix) -> Optional[BioMatrix]:
        """Reverse transformation if possible."""
        pass

    def validate(self, matrix: BioMatrix) -> List[str]:
        """Check preconditions. Return list of errors."""
        pass
```

### 3. Pipeline
Chains transforms into reproducible workflows.

```python
class Pipeline:
    """Composable analysis workflow."""

    def __init__(self, name: str):
        self.transforms: List[Transform] = []
        self.name = name
        self.checkpoints: Dict[str, BioMatrix] = {}

    def add(self, transform: Transform) -> 'Pipeline':
        """Add transform to pipeline (builder pattern)."""
        self.transforms.append(transform)
        return self

    def execute(self, matrix: BioMatrix,
                checkpoint_strategy: str = 'final') -> BioMatrix:
        """Run pipeline, optionally saving intermediate results."""
        result = matrix
        for i, transform in enumerate(self.transforms):
            result = result.apply(transform)
            if checkpoint_strategy == 'all':
                self.checkpoints[f"step_{i}_{transform.name}"] = result
        return result

    def to_script(self) -> str:
        """Generate executable Python script for pipeline."""
        pass
```

### 4. ProvenanceGraph
Tracks complete transformation history for reproducibility and debugging.

```python
class ProvenanceGraph:
    """
    DAG of transformations applied to data.
    Enables auditing, rollback, and reproducibility.
    """
    nodes: List[TransformNode]
    edges: List[Tuple[TransformNode, TransformNode]]

    def add_transform(self, transform: Transform,
                      parent: Optional[TransformNode]) -> TransformNode:
        """Record new transformation."""
        pass

    def get_lineage(self) -> List[Transform]:
        """Return ordered list of all transforms."""
        pass

    def export_notebook(self, path: Path):
        """Generate Jupyter notebook documenting analysis."""
        pass

    def export_methods(self, path: Path):
        """Generate methods section for publication."""
        pass
```

---

## Module Architecture

### Module 1: Quality Control (`biocore.quality`)

**OutlierDetector**
```python
class OutlierDetector(Transform):
    """
    Detect outliers using configurable strategies.

    Strategies:
    - IQR (interquartile range)
    - Z-score (modified, MAD-based)
    - Isolation Forest
    - Domain-specific (e.g., count-based for RNA-seq)
    """
    method: OutlierMethod
    threshold: float
    per_feature: bool = True  # Detect per gene or globally

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        outlier_mask = self._detect(matrix.data)

        # Mark outliers in quality_flags
        new_flags = matrix.quality_flags.copy()
        new_flags[outlier_mask] = QualityFlag.OUTLIER_DETECTED

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            feature_metadata=matrix.feature_metadata,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
            provenance=matrix.provenance.add_transform(self)
        )
```

**Imputer**
```python
class Imputer(Transform):
    """
    Impute outlier/missing values with tracking.

    Strategies:
    - KNN (k-nearest neighbors in feature space)
    - Mean/median (global or per-sample-group)
    - Matrix completion (iterative SVD)
    - Model-based (regression on correlated features)
    """
    strategy: ImputationStrategy
    target_flags: Set[QualityFlag]  # Which flags to impute

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        impute_mask = np.isin(matrix.quality_flags, list(self.target_flags))

        new_data = matrix.data.copy()
        new_data[impute_mask] = self._impute(matrix, impute_mask)

        # Update quality flags to mark imputed values
        new_flags = matrix.quality_flags.copy()
        new_flags[impute_mask] = QualityFlag.IMPUTED

        return BioMatrix(
            data=new_data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            feature_metadata=matrix.feature_metadata,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
            provenance=matrix.provenance.add_transform(self)
        )
```

### Module 2: Graph Analysis (`biocore.graph`)

**CliqueDetector**
```python
class CliqueDetector:
    """
    Find cliques in biological networks.

    Application: TF regulatory networks, protein-protein interactions,
    co-expression modules.
    """

    def __init__(self,
                 correlation_threshold: float = 0.7,
                 min_clique_size: int = 3,
                 max_clique_size: Optional[int] = None):
        self.correlation_threshold = correlation_threshold
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size

    def from_matrix(self, matrix: BioMatrix,
                    feature_subset: Optional[List[str]] = None) -> nx.Graph:
        """
        Build correlation network from expression matrix.
        Optionally focus on feature subset (e.g., TFs only).
        """
        if feature_subset:
            matrix = matrix.select_features(
                lambda m: m.feature_ids.isin(feature_subset)
            )

        # Compute correlation matrix
        corr = np.corrcoef(matrix.data)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(matrix.feature_ids)

        # Add edges for significant correlations
        for i in range(len(corr)):
            for j in range(i+1, len(corr)):
                if abs(corr[i, j]) >= self.correlation_threshold:
                    G.add_edge(
                        matrix.feature_ids[i],
                        matrix.feature_ids[j],
                        weight=corr[i, j]
                    )

        return G

    def find_cliques(self, G: nx.Graph) -> List[Set[str]]:
        """Enumerate maximal cliques within size constraints."""
        cliques = list(nx.find_cliques(G))

        # Filter by size
        filtered = [
            set(c) for c in cliques
            if len(c) >= self.min_clique_size and
               (self.max_clique_size is None or len(c) <= self.max_clique_size)
        ]

        return filtered

    def clique_subsets(self,
                       parent_features: Set[str],
                       child_features: Set[str],
                       matrix: BioMatrix) -> Dict[str, Set[str]]:
        """
        Find child cliques regulated by each parent.

        Example: For each TF, find cliques in its target genes.
        """
        G = self.from_matrix(matrix, feature_subset=list(child_features))
        cliques = self.find_cliques(G)

        # Map parents to their child cliques
        parent_cliques = {}
        for parent in parent_features:
            # Find which cliques are connected to this parent
            # (Implementation depends on how TF->target relationships are defined)
            parent_cliques[parent] = [
                clique for clique in cliques
                if self._is_regulated_by(clique, parent, matrix)
            ]

        return parent_cliques
```

### Module 3: Normalization (`biocore.normalize`)

```python
class Normalizer(Transform):
    """
    Standard normalization strategies for omics data.

    Methods:
    - TPM/FPKM (transcriptomics)
    - TMM, DESeq2 size factors (differential expression)
    - Z-score, quantile normalization
    - Batch correction (ComBat, Harmony)
    """
    method: NormalizationMethod

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        # Implementation specific to method
        pass
```

### Module 4: I/O (`biocore.io`)

```python
class MatrixLoader:
    """Load various omics formats into BioMatrix."""

    @staticmethod
    def from_csv(path: Path,
                 sample_metadata_path: Optional[Path] = None,
                 feature_col: str = "feature_id") -> BioMatrix:
        """Load from CSV, infer structure."""
        pass

    @staticmethod
    def from_h5ad(path: Path) -> BioMatrix:
        """Load from AnnData (scanpy format)."""
        pass

    @staticmethod
    def from_geo(accession: str) -> BioMatrix:
        """Download from GEO and parse."""
        pass

class MatrixWriter:
    """Export BioMatrix in various formats."""

    @staticmethod
    def to_csv(matrix: BioMatrix,
               path: Path,
               include_quality_flags: bool = True):
        """
        Export to CSV with optional quality flag annotation.

        If include_quality_flags:
            - Writes two files: {path} and {path}.flags.csv
        """
        pass

    @staticmethod
    def to_h5ad(matrix: BioMatrix, path: Path):
        """Export to AnnData for use with scanpy/seurat."""
        pass
```

---

## Quality Flag System

Enum-based tracking of data quality at per-value resolution:

```python
class QualityFlag(IntEnum):
    """
    Bitwise flags for data quality tracking.
    Multiple flags can be combined.
    """
    ORIGINAL = 0              # Untouched original value
    OUTLIER_DETECTED = 1      # Flagged as statistical outlier
    IMPUTED = 2               # Value was imputed
    MISSING_ORIGINAL = 4      # Originally missing (NaN)
    BATCH_CORRECTED = 8       # Underwent batch correction
    LOW_CONFIDENCE = 16       # Low quality measurement (platform-specific)
    MANUAL_REVIEW = 32        # Flagged for manual inspection

    # Convenience methods
    def is_modified(self) -> bool:
        return self > QualityFlag.ORIGINAL

    def is_trustworthy(self) -> bool:
        return not (self & QualityFlag.LOW_CONFIDENCE)
```

**Usage:**
```python
# Mark a value as both outlier-detected and imputed
matrix.quality_flags[i, j] = QualityFlag.OUTLIER_DETECTED | QualityFlag.IMPUTED

# Check if value is imputed
if matrix.quality_flags[i, j] & QualityFlag.IMPUTED:
    # Handle imputed value differently
    pass
```

---

## Implementation Priorities

### Phase 1: Immediate Needs (Current Sprint)
1. **BioMatrix core** with CSV I/O
2. **OutlierDetector** (IQR and MAD-Z methods)
3. **Imputer** (KNN and median strategies)
4. **Quality flag export** (two-file CSV format)

**Deliverable:** Imputed dataset + quality flags for ALS analysis

### Phase 2: Graph Analysis (Next Sprint)
1. **CliqueDetector** with correlation network builder
2. **TF child set analysis** (load TF annotations, find cliques in targets)
3. **Network visualization** utilities

**Deliverable:** TF clique analysis for ALS cohort

### Phase 3: Pipeline & Provenance (Following Sprint)
1. **Pipeline execution engine**
2. **ProvenanceGraph** with notebook export
3. **Transform validation framework**

**Deliverable:** Reproducible pipeline scripts

### Phase 4: Advanced Operations (Ongoing)
- Differential expression module
- Batch correction
- Dimensionality reduction
- Statistical testing framework

---

## Design Principles

1. **Immutability**: Transformations return new BioMatrix instances. Original data is never modified in place.

2. **Explicit over Implicit**: All parameters are explicit. No hidden global state.

3. **Composition over Inheritance**: Combine simple transforms rather than building complex hierarchies.

4. **Fail Fast**: Validate preconditions early. Clear error messages.

5. **Type Safety**: Leverage Python type hints. Consider migrating to Rust for performance-critical kernels.

6. **Testability**: Each transform is a pure function (given same input matrix + params → same output).

7. **Domain Expertise Built In**: Encode computational biology best practices as defaults (e.g., log-transform count data before correlation).

---

## Technology Stack

**Core:**
- Python 3.11+
- NumPy/SciPy (numerical operations)
- Pandas (metadata management)
- NetworkX (graph algorithms)

**Quality & Testing:**
- pytest (unit tests)
- hypothesis (property-based testing)
- mypy (static type checking)

**Performance (future):**
- Rust via PyO3 (performance-critical kernels)
- Polars (faster dataframe operations)
- Dask (distributed computing for large cohorts)

**Visualization:**
- Matplotlib/Seaborn (publication figures)
- Plotly (interactive exploration)

---

## File Structure

```
biocore/
├── __init__.py
├── core/
│   ├── biomatrix.py          # BioMatrix class
│   ├── transform.py           # Transform base class
│   ├── pipeline.py            # Pipeline execution
│   ├── provenance.py          # ProvenanceGraph
│   └── quality.py             # QualityFlag enum
├── quality/
│   ├── outliers.py            # OutlierDetector
│   ├── imputation.py          # Imputer
│   └── normalization.py       # Normalizer
├── graph/
│   ├── cliques.py             # CliqueDetector
│   ├── networks.py            # Network construction utilities
│   └── communities.py         # Community detection
├── io/
│   ├── loaders.py             # MatrixLoader
│   └── writers.py             # MatrixWriter
├── stats/
│   ├── differential.py        # Differential expression
│   └── testing.py             # Statistical tests
└── utils/
    ├── validation.py          # Input validation
    └── plotting.py            # Visualization helpers

scripts/
├── impute_outliers.py         # Phase 1 deliverable script
├── analyze_tf_cliques.py      # Phase 2 deliverable script
└── example_pipeline.py        # Full pipeline example

tests/
├── test_biomatrix.py
├── test_outliers.py
├── test_imputation.py
└── test_cliques.py

notebooks/
├── 01_data_exploration.ipynb
├── 02_outlier_analysis.ipynb
└── 03_tf_cliques.ipynb
```

---

## Next Steps

1. **Review this spec** with computational biologist collaborator
2. **Prototype BioMatrix + OutlierDetector** (3-4 hours)
3. **Validate on ALS dataset** - confirm outlier detection makes sense
4. **Implement Imputer + export** (2-3 hours)
5. **Deliver Phase 1** - imputed dataset for downstream TF clique analysis

**Timeline:** Phase 1 deliverable in 1-2 days.
