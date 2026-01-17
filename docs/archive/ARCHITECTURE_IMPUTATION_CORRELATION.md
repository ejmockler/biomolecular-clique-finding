# Biomolecular Clique Finding - Imputation & Correlation Architecture

## Executive Summary

This document maps the code structure for imputation and correlation analysis, identifying:
1. Where imputation strategies are defined and implemented
2. Where correlation computation happens
3. How CLI entry points wire these together
4. Extension points for new imputation strategies

**Target Audience:** Engineers implementing new imputation or correlation methods.

---

## 1. IMPUTATION SYSTEM ARCHITECTURE

### 1.1 Core Imputation Files

#### File: `/src/cliquefinder/quality/imputation.py`
**Purpose:** High-level imputation orchestration and strategy selection

**Key Classes:**
- `Imputer(Transform)` - Main imputation class

**Key Methods:**
- `__init__(strategy, ...)` - Initialize with strategy
- `apply(matrix)` - Apply imputation to BioMatrix
- `validate(matrix)` - Pre-flight checks
- `_impute_winsorize(matrix, mask)` - Delegate to winsorize imputation
- `_impute_median(matrix, mask)` - Delegate to median imputation

**Supported Strategies:**
```python
valid_strategies = ["winsorize", "median"]
```

**Quality Flag Integration:**
- Imputed values marked with `QualityFlag.IMPUTED` for provenance
- Outliers marked with `QualityFlag.OUTLIER_DETECTED` before imputation
- Missing values marked with `QualityFlag.MISSING_ORIGINAL`

Examples:
```python
# In __init__, add to valid_strategies list:
valid_strategies = ["winsorize", "median", "YOUR_STRATEGY_HERE"]

# In apply(), add dispatch case:
elif self.strategy == "YOUR_STRATEGY_HERE":
    new_data = self._impute_your_strategy(matrix, to_impute)

# Implement the worker method:
def _impute_your_strategy(self, matrix: BioMatrix, mask: np.ndarray) -> np.ndarray:
    """Your imputation logic here."""
    data = matrix.data.astype(float).copy()
    # ... imputation logic ...
    return data
```



---

### 1.2 Quality Flagging System

#### File: `/src/cliquefinder/core/quality.py`
**Purpose:** Bitwise quality flag system for tracking data provenance

**Key Enum:**
```python
class QualityFlag(IntFlag):
    ORIGINAL = 0              # Untouched value
    OUTLIER_DETECTED = 1      # Flagged as statistical outlier
    IMPUTED = 2               # Value was imputed
    MISSING_ORIGINAL = 4      # Originally missing (NaN)
    BATCH_CORRECTED = 8       # Underwent batch correction
    LOW_CONFIDENCE = 16       # Low quality measurement
    MANUAL_REVIEW = 32        # Flagged for manual inspection
```

**Usage in Imputation Pipeline:**
```python
# 1. Outlier detection marks values:
new_flags[outlier_mask] |= QualityFlag.OUTLIER_DETECTED

# 2. Imputation then marks them:
new_flags[to_impute] |= QualityFlag.IMPUTED

# 3. Downstream code can query provenance:
imputed_mask = (matrix.quality_flags & QualityFlag.IMPUTED) > 0
n_imputed = np.sum(imputed_mask)
```

**Adding New Quality Flags:**
Simply add to the IntFlag enum with next power of 2:
```python
class QualityFlag(IntFlag):
    # ... existing flags ...
    MY_NEW_FLAG = 64  # Next available bit (2^6)
```

---

### 1.3 Outlier Detection

#### File: `/src/cliquefinder/quality/outliers.py`
**Purpose:** Detect statistical outliers before imputation

**Key Classes:**
- `OutlierDetector(Transform)` - Main outlier detection
- `ResidualOutlierDetector(Transform)` - Model-based residual detection

**Detection Methods:**
1. **MAD-Z (Median Absolute Deviation):** threshold=3.5 (default)
   - More robust than standard deviation
   - 50% breakdown point
   
2. **IQR (Interquartile Range, Tukey):** threshold=1.5
   - Simpler but less robust

**Detection Modes:**
1. **within_group** (default, recommended)
   - Detects outliers within each group separately
   - Preserves biological differences between groups
   - Requires sample_metadata
   
2. **per_feature** (legacy)
   - Detects per-gene across all samples
   - Conflates biology with artifacts (not recommended)
   
3. **global**
   - Detects across entire matrix
   - Use with pre-normalized data only

**Quality Flag Output:**
- Marks detected outliers with `QualityFlag.OUTLIER_DETECTED`
- Data unchanged (only flags updated)

---

## 2. CORRELATION COMPUTATION SYSTEM

### 2.1 Correlation Matrix Computation

#### File: `/src/cliquefinder/utils/correlation_matrix.py`
**Purpose:** Memory-efficient correlation matrix computation with caching

**Key Functions:**

1. **`get_correlation_matrix(matrix, method='pearson', cache=True, ...)`**
   - Main entry point
   - Handles caching transparently
   - Returns float32 for memory efficiency
   
2. **`compute_correlation_matrix_chunked(data, chunk_size=500, verbose=True, output=None)`**
   - Core computation engine
   - Processes in chunks to avoid memory explosion
   - Can write to memory-mapped file (output parameter)
   - Uses standardized vectors for numerical stability

**Performance:**
- First call: 1-2 hours (60K genes, 500 samples)
- Subsequent calls: 2-3 seconds (cached)
- Memory: 20GB RAM for computation, uses memmap for storage

**Cache Structure:**
```
~/.cache/biocore/correlation_matrices/
├── corr_{cache_key}.mmap    (binary, ~14GB for 60K genes)
└── corr_{cache_key}.meta    (JSON metadata with provenance)
```

**Cache Key:** SHA256(gene_ids + sample_ids + data_sample_hash)[:16]

**Extension Point for New Correlation Methods:**
```python
def get_correlation_matrix(matrix, method='pearson', ...):
    if method != 'pearson':
        raise NotImplementedError(f"Only 'pearson' method supported, got '{method}'")
    
    # TO ADD SPEARMAN:
    # 1. Add 'spearman' to allowed methods
    # 2. Implement correlation computation in compute_correlation_matrix_chunked()
    # 3. Update cache key to include method name
```

**Current Limitation:** Only Pearson correlation implemented

---

### 2.2 Correlation Usage in Clique Finding

#### File: `/src/cliquefinder/knowledge/clique_validator.py`
**Purpose:** Find regulatory cliques based on correlation structure

**Key Methods:**
1. **`compute_correlation_matrix(condition, method='pearson')`**
   - Computes correlation for subset of samples matching condition
   - Cached per-condition for efficiency
   
2. **`build_correlation_graph(genes, condition, min_correlation, method='pearson')`**
   - Builds NetworkX graph with correlation >= threshold
   - Edges represent co-expression relationships
   
3. **`find_cliques(genes, condition, min_correlation, min_clique_size)`**
   - Finds all maximal cliques meeting threshold
   - Returns CorrelationClique objects
   
4. **`find_maximum_clique(genes, condition, min_correlation)`**
   - Greedy maximum clique algorithm (O(n·d²))
   - Faster than exhaustive enumeration

**Correlation Type:**
- **Fixed:** Pearson correlation (L2-normalized)
- **Threshold:** min_correlation parameter (typically 0.7)
- **Graph Type:** Undirected (uses absolute correlation)

**Querying Correlations:**
```python
# Get correlation matrix for a condition:
corr_matrix = validator.compute_correlation_matrix(condition='CASE_Male')

# Extract pairwise correlation:
corr_val = corr_matrix[gene_i_idx, gene_j_idx]

# Build graph with threshold:
G = validator.build_correlation_graph(
    genes=['TP53', 'BRCA1', 'MDM2'],
    condition='CASE',
    min_correlation=0.7
)
```

---

## 3. CLI INTEGRATION

### 3.1 Impute Command

#### File: `/src/cliquefinder/cli/impute.py`
**Purpose:** CLI entry point for Phase 1 (outlier detection + imputation)

**Command Line Arguments:**
```bash
cliquefinder impute \
  --input data.csv \
  --output results/imputed \
  --method mad-z \
  --threshold 3.5 \
  --mode within_group \
  --group-cols phenotype \
  --impute-strategy knn_correlation \
  --n-neighbors 5 \
  --weighted \
  --weight-power 1.0 \
  --correlation-threshold 0.7 \
  --min-neighbors 3 \
  --max-neighbors 20
```

**Key Arguments for Imputation:**
- `--impute-strategy`: winsorize (default) or median
- `--lower-percentile`, `--upper-percentile`: Percentile bounds for winsorize

**Workflow:**
1. Load expression data
2. Detect outliers (OutlierDetector)
3. Impute flagged values (Imputer)
4. Write imputed matrix + quality flags + metadata

**Output Files:**
- `{output}.data.csv` - Imputed expression matrix
- `{output}.quality_flags.csv` - Quality flags (same shape)
- `{output}.metadata.csv` - Sample metadata
- `{output}.report.txt` - Summary statistics

---

### 3.2 Analyze Command

#### File: `/src/cliquefinder/cli/analyze.py`
**Purpose:** CLI entry point for Phase 2 (regulatory clique discovery)

**Correlation-Related Arguments:**
- `--min-correlation`: Minimum correlation for cliques (default 0.7)
  - Passed directly to CliqueValidator
  - Uses default Pearson correlation (currently hardcoded)

**Command Line:**
```bash
cliquefinder analyze \
  --input data.csv \
  --output results/analysis \
  --discover \
  --min-correlation 0.7 \
  --min-clique-size 3
```

**Correlation Type Selection:**
Currently **not configurable via CLI** - hardcoded to Pearson:
- Line 407-408 in `_analyze_core.py`: `build_correlation_graph(..., method='pearson')`
- Would require adding `--correlation-method` argument and passing through pipeline

---

### 3.3 Core Analysis Functions

#### File: `/src/cliquefinder/cli/_analyze_core.py`
**Purpose:** Refactored analysis logic shared between CLI and library use

**Key Functions:**

1. **`run_stratified_analysis(..., min_correlation=0.7, ...)`**
   - Orchestrates regulator analysis across conditions
   - Parallelization: threads, processes, or hybrid
   - Calls CliqueValidator with min_correlation threshold

2. **`analyze_regulator_module(regulator, module, validator, conditions, min_correlation, ...)`**
   - Analyzes single regulator across conditions
   - Uses validator.find_cliques() or find_maximum_clique()
   - Returns RegulatorAnalysisResult with correlation cliques

3. **`_analyze_single_condition(regulator, targets, validator, condition, min_correlation, ...)`**
   - Analyzes one condition for one regulator
   - Internally uses Pearson correlation via validator

**Correlation Parameters:**
- `min_correlation` (default 0.7): Minimum correlation for clique membership
- Hardcoded to Pearson (no Spearman support)

---

## 4. CORRELATION TYPE SPECIFICATIONS

### 4.1 Current Implementation (Pearson Only)

**Where Pearson is Specified:**
1. `/src/cliquefinder/quality/correlation_knn.py:176`
   ```python
   corr_matrix = np.corrcoef(x, y)  # Pearson by default
   ```

2. `/src/cliquefinder/utils/correlation_matrix.py:318`
   ```python
   chunk_corr = (chunk_standardized @ data_standardized.T) / n_samples
   # This is Pearson (standardized dot product)
   ```

3. `/src/cliquefinder/knowledge/clique_validator.py`
   ```python
   def compute_correlation_matrix(self, condition, method='pearson'):
       # method parameter exists but only 'pearson' implemented
   ```

4. `/src/cliquefinder/cli/_analyze_core.py:407`
   ```python
   G = validator.build_correlation_graph(..., method='pearson')
   ```

### 4.2 Adding Spearman Support

**Required Changes:**

1. **In `correlation_matrix.py`:**
   ```python
   def compute_correlation_matrix_chunked(data, method='pearson', ...):
       if method == 'spearman':
           from scipy.stats import rankdata
           # Rank transform data
           ranked_data = np.apply_along_axis(rankdata, axis=1, arr=data)
           # Then compute Pearson on ranks
       else:
           ranked_data = data
       # ... rest of computation ...
   ```

2. **In `clique_validator.py`:**
   ```python
   def compute_correlation_matrix(self, condition, method='pearson'):
       # Pass method through to correlation computation
   ```

3. **In CLI `/src/cliquefinder/cli/analyze.py`:**
   ```python
   parser.add_argument('--correlation-method', 
                       choices=['pearson', 'spearman'],
                       default='pearson')
   # Pass to run_stratified_analysis()
   ```

---

## 5. STRATEGY PATTERN & ENUM ANALYSIS

### 5.1 Imputation Strategy Pattern

**Design:** Strategy pattern with runtime selection

**Location:** `/src/cliquefinder/quality/imputation.py:283`

```python
class Imputer:
    def __init__(self, strategy: str = "knn_correlation", ...):
        valid_strategies = ["knn_correlation", "radius_correlation", "median"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: '{strategy}'")
        self.strategy = strategy
    
    def apply(self, matrix):
        if self.strategy == "knn_correlation":
            return self._impute_knn_correlation(matrix, to_impute)
        elif self.strategy == "radius_correlation":
            return self._impute_radius_correlation(matrix, to_impute)
        elif self.strategy == "median":
            return self._impute_median(matrix, to_impute)
```

**To Add New Strategy:**
1. Add to `valid_strategies` list
2. Add dispatch case in `apply()`
3. Implement `_impute_your_strategy()` method
4. Update CLI argument choices in `impute.py:45`

### 5.2 Quality Flag Enum

**Design:** Bitwise IntFlag enum (not strategy pattern)

**Location:** `/src/cliquefinder/core/quality.py:48`

```python
class QualityFlag(IntFlag):
    ORIGINAL = 0                  # 0b000000
    OUTLIER_DETECTED = 1          # 0b000001
    IMPUTED = 2                   # 0b000010
    MISSING_ORIGINAL = 4          # 0b000100
    BATCH_CORRECTED = 8           # 0b001000
    LOW_CONFIDENCE = 16           # 0b010000
    MANUAL_REVIEW = 32            # 0b100000
```

**Why IntFlag, Not Enum?**
- Bitwise operations: `flag & QualityFlag.IMPUTED`
- Multiple flags per value: `OUTLIER_DETECTED | IMPUTED`
- Memory efficient: single int per value

---

## 6. DATA FLOW DIAGRAMS

### 6.1 Imputation Pipeline

```
Raw Data (CSV)
    ↓
load_csv_matrix() → BioMatrix(data, feature_ids, sample_ids, quality_flags)
    ↓
OutlierDetector.apply() 
    ├─ Detects outliers (within_group/per_feature/global)
    └─ Updates quality_flags with OUTLIER_DETECTED
    ↓
Imputer.apply()
    ├─ Identifies to_impute = OUTLIER_DETECTED | MISSING_ORIGINAL
    ├─ Selects strategy (knn_correlation/radius_correlation/median)
    │   ├─ knn_correlation → correlation_knn.correlation_knn_impute()
    │   │   ├─ Computes correlation matrix
    │   │   ├─ For each gene: find k neighbors by correlation
    │   │   └─ Aggregates with weighted median
    │   ├─ radius_correlation → correlation_knn.correlation_radius_impute()
    │   │   ├─ Computes correlations
    │   │   ├─ For each gene: find neighbors by threshold
    │   │   └─ Aggregates with weighted median
    │   └─ median → Direct median per gene (or per group)
    ├─ Returns new_data with imputed values
    └─ Updates quality_flags with IMPUTED
    ↓
write_csv_matrix() → {output}.data.csv, {output}.quality_flags.csv, etc.
```

### 6.2 Correlation Pipeline (Clique Finding)

```
Imputed Data (CSV)
    ↓
load_csv_matrix() → BioMatrix
    ↓
CliqueValidator.__init__(matrix, stratify_by=['phenotype', 'Sex'])
    ├─ Stores matrix reference
    └─ Initializes condition caches
    ↓
validator.precompute_condition_data()
    ├─ For each condition (e.g., 'CASE_Male', 'CTRL_Female'):
    │   └─ Slices matrix for that condition
    └─ Caches condition-specific submatrices
    ↓
For each regulator:
    ├─ Get INDRA targets from CoGEx
    ├─ For each condition:
    │   ├─ validator.find_maximum_clique() or find_cliques()
    │   │   ├─ compute_correlation_matrix(condition)
    │   │   │   ├─ Uses utils.correlation_matrix.get_correlation_matrix()
    │   │   │   │   └─ Pearson correlation (cached)
    │   │   │   └─ Subsets to condition's samples
    │   │   ├─ build_correlation_graph(genes, condition, min_correlation=0.7)
    │   │   │   └─ Creates NetworkX graph with edges where |r| >= threshold
    │   │   └─ Returns CorrelationClique objects
    │   └─ [Parallel across conditions if n_threads > 1]
    └─ Differential analysis (CASE vs CTRL)
        └─ Uses Fisher's Z-test from stats.correlation_tests
    ↓
save_results() → CSV files with cliques, differential rewiring, etc.
```

---

## 7. IMPLEMENTATION TASK BREAKDOWN

### Task 1: Add New Imputation Strategy (e.g., SVD-based)

**Files to Modify:**
1. `/src/cliquefinder/quality/imputation.py`
   - Add "svd" to valid_strategies (line 283)
   - Add elif branch in apply() (line 414)
   - Add _impute_svd() method (new, ~50 lines)
   - Update __init__ docstring

2. `/src/cliquefinder/cli/impute.py`
   - Update --impute-strategy choices (line 45)

**Estimated Effort:** 2-3 hours

---

### Task 2: Add Spearman Correlation Support

**Files to Modify:**
1. `/src/cliquefinder/utils/correlation_matrix.py`
   - Modify compute_correlation_matrix_chunked() to accept method parameter
   - Implement rank transformation for Spearman
   - Update get_correlation_matrix() to pass method through
   - Update cache key to include method name

2. `/src/cliquefinder/knowledge/clique_validator.py`
   - Update compute_correlation_matrix() to pass method through

3. `/src/cliquefinder/cli/analyze.py`
   - Add --correlation-method argument
   - Pass through to run_stratified_analysis()

4. `/src/cliquefinder/cli/_analyze_core.py`
   - Update run_stratified_analysis() signature

**Estimated Effort:** 4-6 hours

---

### Task 3: Add Distance Metric Options for KNN Imputation

**Files to Modify:**
1. `/src/cliquefinder/quality/correlation_knn.py`
   - Refactor _find_k_nearest_correlation() to use pluggable distance metric
   - Create distance metric interface/enum
   - Implement alternative metrics (e.g., mutual information, dynamic time warping)

2. `/src/cliquefinder/quality/imputation.py`
   - Add distance_metric parameter to Imputer
   - Pass through to correlation_knn_impute()

3. `/src/cliquefinder/cli/impute.py`
   - Add --distance-metric argument

**Estimated Effort:** 5-8 hours

---

## 8. TESTING CONSIDERATIONS

### Unit Test Targets

1. **Imputation Tests** (existing: `tests/test_imputation.py`)
   - New strategy produces valid numeric output
   - Quality flags updated correctly
   - No NaN values remain after imputation

2. **Correlation Tests** (to create: `tests/test_correlation_methods.py`)
   - Pearson vs Spearman produce expected differences
   - Correlation caching works correctly
   - Different correlation types affect clique results differently

3. **CLI Tests** (to enhance: `tests/test_cli.py`)
   - New imputation strategy accessible via CLI
   - New correlation method flows through pipeline
   - Output files match expected format

### Integration Test

Full pipeline with different imputation strategies and verify downstream analyses:
```python
for strategy in ["knn_correlation", "radius_correlation", "median"]:
    result = run_impute_analyze_pipeline(data, strategy)
    assert result.n_cliques > 0
    assert all(gene in matrix.feature_ids for clique in result for gene in clique)
```

---

## 9. QUICK REFERENCE: WHERE THINGS HAPPEN

| Component | File | Lines | Key Function |
|-----------|------|-------|--------------|
| **Imputation Strategies** | imputation.py | 283-420 | Strategy selection & dispatch |
| **KNN Imputation (Pearson)** | correlation_knn.py | 489-739 | correlation_knn_impute() |
| **Radius Imputation** | correlation_knn.py | 742-1034 | correlation_radius_impute() |
| **Quality Flags** | core/quality.py | 48-138 | QualityFlag IntFlag enum |
| **Outlier Detection** | quality/outliers.py | 55-267 | OutlierDetector class |
| **Correlation Computation** | utils/correlation_matrix.py | 195-340 | compute_correlation_matrix_chunked() |
| **Correlation Caching** | utils/correlation_matrix.py | 410-562 | get_correlation_matrix() |
| **Clique Finding** | knowledge/clique_validator.py | ~400+ | build_correlation_graph(), find_cliques() |
| **Impute CLI** | cli/impute.py | 23-145 | register_parser(), run_impute() |
| **Analyze CLI** | cli/analyze.py | 18-207 | register_parser(), run_analyze() |
| **Core Analysis** | cli/_analyze_core.py | 225-779 | run_stratified_analysis(), analyze_regulator_module() |

---

## 10. ARCHITECTURE DECISIONS & RATIONALE

### Design Decision 1: Strategy Pattern for Imputation

**Decision:** Runtime strategy selection via string parameter

**Rationale:**
- User-friendly CLI interface
- Easy to add new strategies without modifying calling code
- Clear separation of concerns

**Alternative Considered:** Inheritance hierarchy (Imputation base class)
- Would be more OOP-pure
- But harder to add new strategies from plugins
- Current approach allows monkeypatching if needed

### Design Decision 2: Pearson Only in Correlation Computation

**Decision:** Hardcoded Pearson correlation (no Spearman)

**Rationale:**
- Pearson optimal for gene expression after log-transformation
- Faster than Spearman (no ranking required)
- Spearman useful only for non-normal distributions (rare in log-space)

**How to Change:** See Task 2 above

### Design Decision 3: Bitwise IntFlag for Quality Flags

**Decision:** Use bitwise flags instead of separate columns

**Rationale:**
- Memory efficient: 1 int per value vs multiple boolean columns
- Enables composable flags: `OUTLIER_DETECTED | IMPUTED`
- Fast queries: `mask = (flags & QualityFlag.IMPUTED) > 0`

**Alternative:** DataFrame columns
- Would be more readable but 8x more memory
- Slower bitwise operations

### Design Decision 4: Correlation Caching via Memory Mapping

**Decision:** Use numpy.memmap for large matrices

**Rationale:**
- 60K × 60K × 8 bytes = 28GB (exceeds most RAM)
- memmap allows OS to manage paging
- First computation expensive, but subsequent runs fast

**Limitations:**
- Cache key based on data hash (exact data match required)
- Not portable across systems (absolute file paths)
- Requires filesystem with fast I/O

---

## 11. KNOWN LIMITATIONS & FUTURE WORK

### Current Limitations

1. **Correlation Type:** Only Pearson implemented
   - Spearman would help for count-based data
   - Mutual information for non-linear relationships

3. **Missing Configuration:**
   - Cannot specify correlation method via CLI (hardcoded Pearson)
   - Cannot override correlation caching strategy
   - No option for exact vs approximate correlation

4. **Parallelization:**
   - Imputation uses joblib (multiprocessing)
   - Clique finding uses ThreadPoolExecutor
   - Mixing threading levels can cause memory issues

### Recommended Improvements

1. Make correlation method configurable end-to-end
2. Add alternative distance metrics with benchmark suite
3. Expose caching configuration via CLI
4. Better error messages when imputation fails
5. Sensitivity analysis (vary k, compare strategies)

---

## 12. CONTACT & SUPPORT

For questions about this architecture:

- **Imputation:** See `/src/cliquefinder/quality/imputation.py` docstrings
- **Correlation:** See `/src/cliquefinder/utils/correlation_matrix.py` docstrings
- **CLI:** See individual `register_parser()` functions
- **Quality Flags:** See `/src/cliquefinder/core/quality.py` module docstring

