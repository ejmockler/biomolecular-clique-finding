# Method Comparison Framework: Multi-Method Clique Differential Testing

## Executive Summary

**Goal**: Run multiple statistical methods (OLS, LMM, ROAST, Permutation) on the same cliques and quantify their concordance, enabling robust discovery and method-specific insights.

**Key Principle**: Methods answer different questions—disagreement is informative, not a bug.

| Method | Question Answered | Test Type |
|--------|------------------|-----------|
| **OLS/LMM** | "Is aggregated clique abundance different?" | Parametric contrast |
| **ROAST** | "Are genes in this clique enriched for DE?" | Self-contained rotation |
| **Permutation** | "Is this clique more DE than random sets?" | Competitive enrichment |

**Deliverable**: `src/cliquefinder/stats/method_comparison.py` with unified abstractions, method adapters, and concordance analysis.

---

## Statistical Foundations

### Why Multiple Methods?

Each method makes different assumptions and has different sensitivities:

1. **OLS/LMM (Aggregation-based)**
   - Summarizes proteins → single clique value via Tukey's Median Polish
   - Tests if summary differs between conditions
   - **Weakness**: Misses bidirectional regulation (up/down signals cancel)
   - **Strength**: Interpretable effect sizes (log2FC)

2. **ROAST (Rotation-based)**
   - Tests gene-level effects within clique
   - Preserves inter-gene correlation structure
   - **Strength**: Detects bidirectional regulation via MSQ statistic
   - **Weakness**: No single "effect size" interpretation

3. **Permutation (Competitive)**
   - Compares clique to random gene sets of same size
   - **Strength**: Non-parametric, accounts for gene-gene correlation
   - **Weakness**: Tests enrichment relative to background, not absolute effect

### Multiple Testing Considerations

**Critical Design Decision**: This framework is for **descriptive comparison**, not inference.

- We do NOT select the "best" p-value per clique (would inflate FDR)
- We do NOT combine p-values across methods (requires strong assumptions)
- We DO report concordance metrics to characterize agreement
- We DO flag disagreement cases for biological investigation

### Concordance Metrics

| Metric | What It Measures | Interpretation |
|--------|-----------------|----------------|
| **Spearman ρ** | Rank agreement of p-values | > 0.8 = excellent, < 0.5 = poor |
| **Cohen's κ** | Classification agreement (sig/non-sig) | > 0.6 = substantial |
| **Jaccard Index** | Overlap of significant sets | Intersection / Union |
| **Effect Size r** | Correlation of effect estimates | Log2FC vs z-score |
| **RMSE** | Effect size divergence | Lower = more similar |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PreparedCliqueExperiment                           │   │
│  │  (Immutable snapshot: normalized, imputed, ID-mapped)                 │   │
│  │                                                                       │   │
│  │  • data: ndarray (n_features, n_samples)                             │   │
│  │  • feature_ids: list[str]                                            │   │
│  │  • cliques: list[CliqueDefinition]                                   │   │
│  │  • design: ExperimentalDesign                                        │   │
│  │  • preprocessing_params: dict (provenance)                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            METHOD LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │   OLSMethod    │  │   LMMMethod    │  │  ROASTMethod   │  │ PermMethod│  │
│  │                │  │                │  │                │  │           │  │
│  │ .name          │  │ .name          │  │ .name          │  │ .name     │  │
│  │ .test(exp)     │  │ .test(exp)     │  │ .test(exp)     │  │ .test(exp)│  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └─────┬─────┘  │
│          │                   │                   │                 │         │
│          └───────────────────┼───────────────────┼─────────────────┘         │
│                              │                   │                           │
│                              ▼                   ▼                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      UnifiedCliqueResult                              │   │
│  │  (Standardized output across all methods)                            │   │
│  │                                                                       │   │
│  │  • clique_id, method: MethodName                                     │   │
│  │  • effect_size, effect_size_se, p_value                              │   │
│  │  • statistic_value, statistic_type, degrees_of_freedom               │   │
│  │  • n_proteins, n_proteins_found                                       │   │
│  │  • method_metadata: dict                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    MethodComparisonResult                             │   │
│  │                                                                       │   │
│  │  • results_by_method: dict[MethodName, list[UnifiedCliqueResult]]    │   │
│  │  • pairwise_concordance: list[ConcordanceMetrics]                    │   │
│  │  • disagreement_cases: DataFrame                                      │   │
│  │                                                                       │   │
│  │  Methods:                                                             │   │
│  │  • .wide_format() → DataFrame (one row per clique)                   │   │
│  │  • .concordance_heatmap_data() → DataFrame (matrix)                  │   │
│  │  • .robust_hits(threshold) → list[str] (sig in all methods)          │   │
│  │  • .method_specific_hits(method) → list[str] (sig only in one)       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/cliquefinder/stats/method_comparison.py
├── Enums
│   └── MethodName                    # OLS, LMM, ROAST_MSQ, ROAST_MEAN, PERMUTATION_*
│
├── Core Dataclasses (frozen=True)
│   ├── UnifiedCliqueResult           # Standardized per-clique output
│   ├── PreparedCliqueExperiment      # Immutable preprocessed data
│   ├── ConcordanceMetrics            # Pairwise method agreement
│   └── MethodComparisonResult        # Complete comparison output
│
├── Protocol
│   └── CliqueTestMethod              # Interface all methods implement
│
├── Method Implementations
│   ├── OLSMethod                     # Wraps run_clique_differential (fixed)
│   ├── LMMMethod                     # Wraps run_clique_differential (mixed)
│   ├── ROASTMethod                   # Wraps RotationTestEngine
│   └── PermutationMethod             # Wraps run_permutation_clique_test
│
├── Concordance Functions
│   ├── compute_pairwise_concordance  # Single pair analysis
│   ├── compute_all_concordance       # All pairs
│   └── identify_disagreements        # Flag divergent cliques
│
└── Entry Points
    ├── prepare_experiment            # Shared preprocessing
    └── run_method_comparison         # Main orchestrator
```

---

## Detailed API Specification

### 1. Enums

```python
class MethodName(Enum):
    """
    Registered differential testing methods.

    Naming convention: METHOD_VARIANT for methods with multiple modes.
    """
    OLS = "ols"                                    # Fixed effects OLS
    LMM = "lmm"                                    # Linear mixed model
    ROAST_MSQ = "roast_msq"                        # ROAST with MSQ statistic (bidirectional)
    ROAST_MEAN = "roast_mean"                      # ROAST with mean statistic (directional)
    ROAST_FLOORMEAN = "roast_floormean"            # ROAST with floormean
    PERMUTATION_COMPETITIVE = "permutation_competitive"  # Random gene set null
    PERMUTATION_ROTATION = "permutation_rotation"  # Rotation null (future)
```

### 2. UnifiedCliqueResult

```python
@dataclass(frozen=True)
class UnifiedCliqueResult:
    """
    Standardized result format across all methods.

    Design principles:
    1. Frozen (immutable) for reproducibility
    2. Core fields are semantically equivalent across methods
    3. method_metadata captures method-specific details

    Semantic equivalence notes:
    - effect_size: log2FC for OLS/LMM, mean z-score for ROAST
    - p_value: parametric for OLS/LMM, empirical for permutation, exact for ROAST
    - statistic_value: t-statistic, rotation statistic, or empirical z-score
    """

    # Identity
    clique_id: str
    method: MethodName

    # Core statistics (semantically equivalent)
    effect_size: float              # Magnitude of differential expression
    effect_size_se: float | None    # Standard error (where available)
    p_value: float                  # Method-appropriate p-value

    # Test statistic details
    statistic_value: float          # Raw test statistic
    statistic_type: str             # "t", "z", "msq", "empirical_z"
    degrees_of_freedom: float | None  # df for parametric tests

    # Clique metadata
    n_proteins: int                 # Size of clique definition
    n_proteins_found: int           # Proteins found in data

    # Method-specific (for deep analysis)
    method_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flatten for DataFrame construction."""
        base = {
            'clique_id': self.clique_id,
            'method': self.method.value,
            'effect_size': self.effect_size,
            'effect_size_se': self.effect_size_se,
            'p_value': self.p_value,
            'statistic_value': self.statistic_value,
            'statistic_type': self.statistic_type,
            'df': self.degrees_of_freedom,
            'n_proteins': self.n_proteins,
            'n_proteins_found': self.n_proteins_found,
        }
        # Flatten metadata with prefix
        for k, v in self.method_metadata.items():
            base[f'meta_{k}'] = v
        return base

    @property
    def is_valid(self) -> bool:
        """Check if result has valid statistics."""
        return (
            np.isfinite(self.p_value) and
            0 <= self.p_value <= 1 and
            np.isfinite(self.effect_size)
        )
```

### 3. PreparedCliqueExperiment

```python
@dataclass(frozen=True)
class PreparedCliqueExperiment:
    """
    Immutable snapshot of preprocessed data ready for testing.

    ALL methods receive the same PreparedCliqueExperiment, ensuring:
    1. No preprocessing-induced differences between methods
    2. Fair comparison (same input data)
    3. Reproducibility (frozen state)

    This class should be created via prepare_experiment() factory.
    """

    # Expression data (preprocessed)
    data: np.ndarray                          # (n_features, n_samples), float64
    feature_ids: tuple[str, ...]              # Immutable tuple
    feature_to_idx: dict[str, int]            # Fast lookup (note: dict is mutable but frozen class prevents reassignment)

    # Sample information
    sample_metadata: pd.DataFrame             # Full metadata
    condition_column: str
    subject_column: str | None
    conditions: tuple[str, ...]               # Ordered, immutable
    n_samples: int

    # Clique definitions
    cliques: tuple[CliqueDefinition, ...]     # Immutable tuple
    clique_to_feature_indices: dict[str, tuple[int, ...]]  # clique_id -> feature indices

    # Contrast
    contrast: tuple[str, str]                 # (test, reference)
    contrast_name: str

    # Provenance
    preprocessing_params: dict
    creation_timestamp: str

    @property
    def n_features(self) -> int:
        return self.data.shape[0]

    @property
    def n_cliques(self) -> int:
        return len(self.cliques)

    def get_clique_data(self, clique_id: str) -> tuple[np.ndarray, list[str]]:
        """
        Extract expression data for a specific clique.

        Returns:
            Tuple of (data_subset, feature_ids_subset)
            data_subset shape: (n_proteins_found, n_samples)
        """
        indices = self.clique_to_feature_indices.get(clique_id, ())
        if not indices:
            return np.array([]).reshape(0, self.n_samples), []

        data_subset = self.data[list(indices), :]
        ids_subset = [self.feature_ids[i] for i in indices]
        return data_subset, ids_subset

    def get_condition_mask(self, condition: str) -> np.ndarray:
        """Boolean mask for samples in given condition."""
        return (self.sample_metadata[self.condition_column] == condition).values

    def get_design_matrix(self) -> np.ndarray:
        """Build design matrix for OLS/LMM."""
        import statsmodels.api as sm
        condition_cat = pd.Categorical(
            self.sample_metadata[self.condition_column],
            categories=list(self.conditions)
        )
        X = pd.get_dummies(pd.DataFrame({'condition': condition_cat}), drop_first=True, dtype=float)
        X = sm.add_constant(X)
        return X.values
```

### 4. Factory Function: prepare_experiment

```python
def prepare_experiment(
    data: np.ndarray,
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    cliques: list[CliqueDefinition],
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    normalization_method: str | NormalizationMethod = "median",
    imputation_method: str | ImputationMethod = "min_feature",
    map_ids: bool = True,
    verbose: bool = True,
) -> PreparedCliqueExperiment:
    """
    Prepare data for multi-method comparison.

    This is the SINGLE preprocessing entry point. All methods receive
    the same prepared data, ensuring fair comparison.

    Preprocessing pipeline:
    1. Copy data (never modify input)
    2. Normalize (median centering, quantile, etc.)
    3. Impute missing values
    4. Map feature IDs (UniProt → Symbol if needed)
    5. Build clique → feature index mapping
    6. Validate contrast

    Args:
        data: Expression matrix (n_features, n_samples)
        feature_ids: Feature identifiers (UniProt, Ensembl, or Symbol)
        sample_metadata: Sample annotations
        cliques: Clique definitions
        condition_column: Metadata column for condition
        contrast: (test_condition, reference_condition)
        subject_column: Optional column for biological replicates
        normalization_method: Normalization to apply
        imputation_method: Imputation for missing values
        map_ids: Whether to map IDs to gene symbols
        verbose: Print progress

    Returns:
        PreparedCliqueExperiment (frozen, immutable)

    Raises:
        ValueError: If contrast conditions not found in data
        ValueError: If no cliques have proteins in data
    """
    from datetime import datetime
    from .normalization import normalize, NormalizationMethod
    from .missing import impute_missing_values, ImputationMethod
    from ..utils.id_mapping import map_feature_ids_to_symbols

    # Convert string to enum if needed
    if isinstance(normalization_method, str):
        normalization_method = NormalizationMethod(normalization_method)
    if isinstance(imputation_method, str):
        imputation_method = ImputationMethod(imputation_method)

    if verbose:
        print(f"Preparing experiment for method comparison")
        print(f"  Data: {data.shape[0]} features × {data.shape[1]} samples")
        print(f"  Cliques: {len(cliques)}")
        print(f"  Contrast: {contrast[0]} vs {contrast[1]}")

    # 1. Copy and convert to float64
    work_data = data.astype(np.float64).copy()
    work_ids = list(feature_ids)

    # 2. Normalize
    if normalization_method != NormalizationMethod.NONE:
        if verbose:
            print(f"  Normalizing: {normalization_method.value}")
        norm_result = normalize(work_data, method=normalization_method)
        work_data = norm_result.data

    # 3. Impute
    if imputation_method != ImputationMethod.NONE:
        if verbose:
            print(f"  Imputing: {imputation_method.value}")
        imp_result = impute_missing_values(work_data, method=imputation_method)
        work_data = imp_result.data

    # 4. ID mapping
    if map_ids:
        # Check if mapping is needed
        sample_proteins = [p for c in cliques[:10] for p in c.protein_ids[:3]]
        matches = sum(1 for p in sample_proteins if p in work_ids)
        if matches < len(sample_proteins) * 0.5:
            if verbose:
                print(f"  Mapping feature IDs to symbols...")
            symbol_to_feature = map_feature_ids_to_symbols(work_ids, verbose=verbose)
            # Build reverse mapping
            feature_to_symbol = {v: k for k, v in symbol_to_feature.items()}
        else:
            feature_to_symbol = {f: f for f in work_ids}
    else:
        feature_to_symbol = {f: f for f in work_ids}

    # 5. Build feature index map
    feature_to_idx = {fid: i for i, fid in enumerate(work_ids)}

    # 6. Build clique → feature indices
    clique_to_indices = {}
    n_cliques_with_data = 0

    for clique in cliques:
        indices = []
        for pid in clique.protein_ids:
            # Try direct match first
            if pid in feature_to_idx:
                indices.append(feature_to_idx[pid])
            # Try symbol mapping
            elif pid in symbol_to_feature:
                mapped = symbol_to_feature[pid]
                if mapped in feature_to_idx:
                    indices.append(feature_to_idx[mapped])

        if indices:
            clique_to_indices[clique.clique_id] = tuple(indices)
            n_cliques_with_data += 1

    if n_cliques_with_data == 0:
        raise ValueError("No cliques have any proteins in the data")

    if verbose:
        print(f"  Cliques with data: {n_cliques_with_data}/{len(cliques)}")

    # 7. Validate conditions
    conditions = sorted(sample_metadata[condition_column].dropna().unique())
    if contrast[0] not in conditions or contrast[1] not in conditions:
        raise ValueError(f"Contrast {contrast} not found in conditions {conditions}")

    # 8. Build preprocessing params
    preprocessing_params = {
        'normalization': normalization_method.value if hasattr(normalization_method, 'value') else str(normalization_method),
        'imputation': imputation_method.value if hasattr(imputation_method, 'value') else str(imputation_method),
        'map_ids': map_ids,
        'n_features_original': len(feature_ids),
        'n_samples': work_data.shape[1],
        'n_cliques_with_data': n_cliques_with_data,
    }

    return PreparedCliqueExperiment(
        data=work_data,
        feature_ids=tuple(work_ids),
        feature_to_idx=feature_to_idx,
        sample_metadata=sample_metadata.copy(),
        condition_column=condition_column,
        subject_column=subject_column,
        conditions=tuple(conditions),
        n_samples=work_data.shape[1],
        cliques=tuple(cliques),
        clique_to_feature_indices=clique_to_indices,
        contrast=contrast,
        contrast_name=f"{contrast[0]}_vs_{contrast[1]}",
        preprocessing_params=preprocessing_params,
        creation_timestamp=datetime.now().isoformat(),
    )
```

### 5. CliqueTestMethod Protocol

```python
@runtime_checkable
class CliqueTestMethod(Protocol):
    """
    Protocol for clique differential testing methods.

    All methods must:
    1. Have a `name` property returning MethodName
    2. Have a `test` method accepting PreparedCliqueExperiment
    3. Return list[UnifiedCliqueResult]
    4. Be stateless (no side effects on experiment)

    Methods MAY accept additional kwargs for method-specific configuration.
    """

    @property
    def name(self) -> MethodName:
        """Unique identifier for this method."""
        ...

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        **kwargs,
    ) -> list[UnifiedCliqueResult]:
        """
        Run differential test on all cliques.

        Args:
            experiment: Prepared data (immutable)
            **kwargs: Method-specific options

        Returns:
            List of UnifiedCliqueResult, one per clique with data
        """
        ...
```

### 6. ConcordanceMetrics

```python
@dataclass(frozen=True)
class ConcordanceMetrics:
    """
    Pairwise concordance between two methods.

    All metrics are computed on the intersection of cliques
    that both methods successfully tested.
    """

    method_a: MethodName
    method_b: MethodName
    n_cliques_compared: int

    # Rank-based agreement
    spearman_rho: float               # Correlation of p-value ranks
    spearman_pvalue: float            # Significance of correlation

    # Classification agreement at threshold
    threshold: float                  # P-value threshold used
    n_both_significant: int           # Both methods call significant
    n_both_nonsignificant: int        # Both methods call non-significant
    n_a_only: int                     # Only method A significant
    n_b_only: int                     # Only method B significant
    cohen_kappa: float                # Agreement beyond chance (-1 to 1)

    # Effect size agreement
    effect_pearson_r: float           # Correlation of effect sizes
    effect_rmse: float                # Root mean squared difference

    # Direction agreement (for signed effects)
    direction_agreement_frac: float   # Fraction with same sign

    @property
    def jaccard_index(self) -> float:
        """Overlap of significant calls: |A ∩ B| / |A ∪ B|"""
        union = self.n_both_significant + self.n_a_only + self.n_b_only
        return self.n_both_significant / union if union > 0 else 0.0

    @property
    def agreement_rate(self) -> float:
        """Fraction of cliques where methods agree."""
        agreed = self.n_both_significant + self.n_both_nonsignificant
        return agreed / self.n_cliques_compared if self.n_cliques_compared > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'method_a': self.method_a.value,
            'method_b': self.method_b.value,
            'n_compared': self.n_cliques_compared,
            'spearman_rho': self.spearman_rho,
            'spearman_pvalue': self.spearman_pvalue,
            'threshold': self.threshold,
            'n_both_sig': self.n_both_significant,
            'n_both_nonsig': self.n_both_nonsignificant,
            'n_a_only': self.n_a_only,
            'n_b_only': self.n_b_only,
            'cohen_kappa': self.cohen_kappa,
            'jaccard': self.jaccard_index,
            'agreement_rate': self.agreement_rate,
            'effect_r': self.effect_pearson_r,
            'effect_rmse': self.effect_rmse,
            'direction_agreement': self.direction_agreement_frac,
        }
```

### 7. MethodComparisonResult

```python
@dataclass
class MethodComparisonResult:
    """
    Complete comparison across all methods.

    This is the main output of run_method_comparison().
    Provides multiple views into the results for different analyses.
    """

    # Raw results
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]]

    # Pairwise concordance
    pairwise_concordance: list[ConcordanceMetrics]

    # Aggregate statistics
    mean_spearman_rho: float
    mean_cohen_kappa: float

    # Disagreement analysis
    disagreement_cases: pd.DataFrame  # Cliques where methods diverge

    # Provenance
    preprocessing_params: dict
    methods_run: list[MethodName]
    n_cliques_tested: int

    def wide_format(self) -> pd.DataFrame:
        """
        Pivot to wide format: one row per clique, columns for each method.

        Columns:
        - clique_id
        - n_proteins, n_proteins_found
        - {method}_pvalue, {method}_effect_size, {method}_statistic
        - ...for each method

        Returns:
            DataFrame with one row per clique
        """
        # Collect all clique IDs
        all_ids = set()
        for results in self.results_by_method.values():
            all_ids.update(r.clique_id for r in results)

        # Build wide format
        rows = []
        for cid in sorted(all_ids):
            row = {'clique_id': cid}

            for method, results in self.results_by_method.items():
                result = next((r for r in results if r.clique_id == cid), None)
                prefix = method.value

                if result:
                    row[f'{prefix}_pvalue'] = result.p_value
                    row[f'{prefix}_effect_size'] = result.effect_size
                    row[f'{prefix}_statistic'] = result.statistic_value
                    row['n_proteins'] = result.n_proteins
                    row['n_proteins_found'] = result.n_proteins_found
                else:
                    row[f'{prefix}_pvalue'] = np.nan
                    row[f'{prefix}_effect_size'] = np.nan
                    row[f'{prefix}_statistic'] = np.nan

            rows.append(row)

        return pd.DataFrame(rows)

    def concordance_matrix(self) -> pd.DataFrame:
        """
        Square matrix of pairwise Spearman correlations.

        Use for heatmap visualization.
        """
        methods = [m.value for m in self.methods_run]
        n = len(methods)
        matrix = np.eye(n)  # Diagonal = 1

        for conc in self.pairwise_concordance:
            i = methods.index(conc.method_a.value)
            j = methods.index(conc.method_b.value)
            matrix[i, j] = conc.spearman_rho
            matrix[j, i] = conc.spearman_rho

        return pd.DataFrame(matrix, index=methods, columns=methods)

    def robust_hits(self, threshold: float = 0.05) -> list[str]:
        """
        Cliques significant in ALL methods.

        These are high-confidence discoveries.
        """
        wide = self.wide_format()
        pval_cols = [c for c in wide.columns if c.endswith('_pvalue')]

        # All p-values below threshold
        mask = (wide[pval_cols] < threshold).all(axis=1)
        return wide.loc[mask, 'clique_id'].tolist()

    def method_specific_hits(
        self,
        method: MethodName,
        threshold: float = 0.05
    ) -> list[str]:
        """
        Cliques significant ONLY in the specified method.

        These may indicate method-specific sensitivity (e.g., ROAST
        detecting bidirectional regulation that OLS misses).
        """
        wide = self.wide_format()
        method_col = f'{method.value}_pvalue'
        other_cols = [c for c in wide.columns if c.endswith('_pvalue') and c != method_col]

        # Significant in this method, not in others
        sig_in_method = wide[method_col] < threshold
        not_sig_elsewhere = (wide[other_cols] >= threshold).all(axis=1)

        mask = sig_in_method & not_sig_elsewhere
        return wide.loc[mask, 'clique_id'].tolist()

    def get_concordance(self, method_a: MethodName, method_b: MethodName) -> ConcordanceMetrics | None:
        """Get concordance metrics for a specific pair."""
        for conc in self.pairwise_concordance:
            if (conc.method_a == method_a and conc.method_b == method_b) or \
               (conc.method_a == method_b and conc.method_b == method_a):
                return conc
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Method Comparison Results",
            "=" * 40,
            f"Cliques tested: {self.n_cliques_tested}",
            f"Methods: {', '.join(m.value for m in self.methods_run)}",
            "",
            "Concordance Summary:",
            f"  Mean Spearman ρ: {self.mean_spearman_rho:.3f}",
            f"  Mean Cohen's κ: {self.mean_cohen_kappa:.3f}",
            "",
            "Robust hits (significant in all methods):",
        ]

        for threshold in [0.05, 0.01, 0.001]:
            n_robust = len(self.robust_hits(threshold))
            lines.append(f"  p < {threshold}: {n_robust}")

        return "\n".join(lines)
```

---

## Method Adapter Implementations

### OLSMethod

```python
class OLSMethod:
    """
    OLS-based differential testing with optional Empirical Bayes moderation.

    Wraps the existing run_clique_differential_analysis internals but
    skips preprocessing (already done) and returns UnifiedCliqueResult.

    Statistical approach:
    1. Summarize proteins within clique via Tukey's Median Polish
    2. Fit OLS: summary ~ condition
    3. Apply EB variance shrinkage (optional)
    4. Test contrast with moderated t-statistic
    """

    def __init__(
        self,
        summarization: str = "tmp",
        eb_moderation: bool = True,
    ):
        self.summarization = summarization
        self.eb_moderation = eb_moderation

    @property
    def name(self) -> MethodName:
        return MethodName.OLS

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs,
    ) -> list[UnifiedCliqueResult]:
        """Run OLS differential test on all cliques."""
        from .summarization import summarize_clique, SummarizationMethod
        from .differential import differential_analysis_single, build_contrast_matrix

        results = []
        conditions = list(experiment.conditions)
        contrast_matrix, contrast_names = build_contrast_matrix(
            conditions,
            {experiment.contrast_name: experiment.contrast}
        )

        sample_condition = experiment.sample_metadata[experiment.condition_column].values

        for clique in experiment.cliques:
            clique_data, clique_features = experiment.get_clique_data(clique.clique_id)

            if len(clique_features) < 2:
                # Not enough proteins
                continue

            # Summarize
            summary = summarize_clique(
                clique_data,
                method=SummarizationMethod(self.summarization)
            )

            # Run differential
            protein_result = differential_analysis_single(
                intensities=summary.summarized_values,
                condition=sample_condition,
                subject=None,  # OLS = no random effects
                feature_id=clique.clique_id,
                contrast_matrix=contrast_matrix,
                contrast_names=contrast_names,
                conditions=conditions,
                use_mixed=False,
            )

            if protein_result.contrasts:
                contrast = protein_result.contrasts[0]
                results.append(UnifiedCliqueResult(
                    clique_id=clique.clique_id,
                    method=self.name,
                    effect_size=contrast.log2_fc,
                    effect_size_se=contrast.se,
                    p_value=contrast.p_value,
                    statistic_value=contrast.t_value,
                    statistic_type="t",
                    degrees_of_freedom=contrast.df,
                    n_proteins=len(clique.protein_ids),
                    n_proteins_found=len(clique_features),
                    method_metadata={
                        'summarization': self.summarization,
                        'eb_moderation': self.eb_moderation,
                        'ci_lower': contrast.ci_lower,
                        'ci_upper': contrast.ci_upper,
                        'coherence': summary.coherence,
                    },
                ))

        return results
```

### LMMMethod

```python
class LMMMethod:
    """
    Linear Mixed Model with random subject effects.

    Identical to OLSMethod but uses mixed model fitting.
    Requires subject_column in experiment.

    Model: summary ~ condition + (1 | subject)
    """

    def __init__(
        self,
        summarization: str = "tmp",
        use_satterthwaite: bool = True,
    ):
        self.summarization = summarization
        self.use_satterthwaite = use_satterthwaite

    @property
    def name(self) -> MethodName:
        return MethodName.LMM

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs,
    ) -> list[UnifiedCliqueResult]:
        """Run LMM differential test on all cliques."""
        if experiment.subject_column is None:
            raise ValueError("LMMMethod requires subject_column in experiment")

        # Similar implementation to OLSMethod but with use_mixed=True
        # ... (implementation follows same pattern)
        pass
```

### ROASTMethod

```python
class ROASTMethod:
    """
    Rotation-based gene set testing (ROAST).

    Tests whether genes within clique show coordinated differential
    expression while preserving inter-gene correlation structure.

    Key advantage: MSQ statistic detects bidirectional regulation.
    """

    def __init__(
        self,
        statistic: str = "msq",
        alternative: str = "mixed",
        n_rotations: int = 9999,
    ):
        self.statistic = statistic
        self.alternative = alternative
        self.n_rotations = n_rotations

    @property
    def name(self) -> MethodName:
        if self.statistic == "msq":
            return MethodName.ROAST_MSQ
        elif self.statistic == "mean":
            return MethodName.ROAST_MEAN
        else:
            return MethodName.ROAST_FLOORMEAN

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[UnifiedCliqueResult]:
        """Run ROAST on all cliques."""
        from .rotation import (
            RotationTestEngine,
            RotationTestConfig,
            SetStatistic,
            Alternative,
        )

        # Initialize engine with full expression data
        engine = RotationTestEngine(
            data=experiment.data,
            gene_ids=list(experiment.feature_ids),
            metadata=experiment.sample_metadata,
        )

        # Fit model (QR decomposition, EB priors)
        engine.fit(
            conditions=list(experiment.conditions),
            contrast=experiment.contrast,
            condition_column=experiment.condition_column,
        )

        # Configure test
        config = RotationTestConfig(
            n_rotations=self.n_rotations,
            use_gpu=use_gpu,
            seed=seed,
        )

        # Build gene sets from cliques
        gene_sets = {
            clique.clique_id: clique.protein_ids
            for clique in experiment.cliques
        }

        # Run tests
        rotation_results = engine.test_gene_sets(gene_sets, config=config, verbose=verbose)

        # Convert to UnifiedCliqueResult
        stat_key = self.statistic
        alt_key = self.alternative

        results = []
        for rot_result in rotation_results:
            p_value = rot_result.get_pvalue(stat_key, alt_key)
            observed = rot_result.observed_stats.get(stat_key, np.nan)

            # For effect size, use mean z-score if available
            mean_z = rot_result.observed_stats.get('mean', 0.0)

            results.append(UnifiedCliqueResult(
                clique_id=rot_result.feature_set_id,
                method=self.name,
                effect_size=mean_z,  # Mean z-score as effect size proxy
                effect_size_se=None,  # ROAST doesn't provide SE
                p_value=p_value,
                statistic_value=observed,
                statistic_type=stat_key,
                degrees_of_freedom=None,  # Rotation test = exact
                n_proteins=rot_result.n_genes,
                n_proteins_found=rot_result.n_genes_found,
                method_metadata={
                    'alternative': alt_key,
                    'n_rotations': rot_result.n_rotations,
                    'active_proportion': rot_result.active_proportion.get(alt_key, np.nan),
                    'observed_msq': rot_result.observed_stats.get('msq', np.nan),
                    'observed_mean': rot_result.observed_stats.get('mean', np.nan),
                },
            ))

        return results
```

### PermutationMethod

```python
class PermutationMethod:
    """
    Competitive permutation-based testing.

    Tests whether clique has stronger differential expression than
    random gene sets of the same size from the measured proteome.

    Null: Random gene sets show equal or greater effect
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        summarization: str = "tmp",
    ):
        self.n_permutations = n_permutations
        self.summarization = summarization

    @property
    def name(self) -> MethodName:
        return MethodName.PERMUTATION_COMPETITIVE

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[UnifiedCliqueResult]:
        """Run competitive permutation test on all cliques."""
        from .permutation_gpu import run_permutation_test_gpu

        # Run permutation test
        perm_results = run_permutation_test_gpu(
            data=experiment.data,
            feature_ids=list(experiment.feature_ids),
            sample_metadata=experiment.sample_metadata,
            cliques=list(experiment.cliques),
            condition_column=experiment.condition_column,
            contrast=experiment.contrast,
            n_permutations=self.n_permutations,
            summarization=self.summarization,
            use_gpu=use_gpu,
            seed=seed,
            verbose=verbose,
        )

        # Convert to UnifiedCliqueResult
        results = []
        for _, row in perm_results.iterrows():
            # Compute z-score from empirical p-value
            from scipy.stats import norm
            p = row['empirical_pvalue']
            z = -norm.ppf(p) if p < 0.5 else norm.ppf(1 - p)

            results.append(UnifiedCliqueResult(
                clique_id=row['clique_id'],
                method=self.name,
                effect_size=row['observed_t'],  # Use observed t as effect
                effect_size_se=None,
                p_value=p,
                statistic_value=z,
                statistic_type="empirical_z",
                degrees_of_freedom=None,
                n_proteins=row['n_proteins'],
                n_proteins_found=row['n_proteins_found'],
                method_metadata={
                    'n_permutations': self.n_permutations,
                    'percentile': row.get('percentile', np.nan),
                    'null_mean': row.get('null_mean', np.nan),
                    'null_std': row.get('null_std', np.nan),
                },
            ))

        return results
```

---

## Concordance Computation

```python
def compute_pairwise_concordance(
    results_a: list[UnifiedCliqueResult],
    results_b: list[UnifiedCliqueResult],
    threshold: float = 0.05,
) -> ConcordanceMetrics:
    """
    Compute concordance metrics between two methods.

    Args:
        results_a: Results from method A
        results_b: Results from method B
        threshold: P-value threshold for classification

    Returns:
        ConcordanceMetrics with all agreement measures
    """
    from scipy import stats as scipy_stats

    # Build lookup by clique_id
    a_by_id = {r.clique_id: r for r in results_a if r.is_valid}
    b_by_id = {r.clique_id: r for r in results_b if r.is_valid}

    # Find common cliques
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    n = len(common_ids)

    if n < 3:
        raise ValueError(f"Need ≥3 common cliques, got {n}")

    # Extract aligned vectors
    p_a = np.array([a_by_id[cid].p_value for cid in common_ids])
    p_b = np.array([b_by_id[cid].p_value for cid in common_ids])
    eff_a = np.array([a_by_id[cid].effect_size for cid in common_ids])
    eff_b = np.array([b_by_id[cid].effect_size for cid in common_ids])

    # 1. Rank correlation of p-values
    rho, rho_pval = scipy_stats.spearmanr(p_a, p_b)

    # 2. Classification agreement
    sig_a = p_a < threshold
    sig_b = p_b < threshold

    n_both_sig = int(np.sum(sig_a & sig_b))
    n_both_nonsig = int(np.sum(~sig_a & ~sig_b))
    n_a_only = int(np.sum(sig_a & ~sig_b))
    n_b_only = int(np.sum(~sig_a & sig_b))

    # Cohen's kappa
    p_o = (n_both_sig + n_both_nonsig) / n  # Observed agreement
    p_a_rate = np.sum(sig_a) / n
    p_b_rate = np.sum(sig_b) / n
    p_e = p_a_rate * p_b_rate + (1 - p_a_rate) * (1 - p_b_rate)  # Expected by chance

    if p_e < 1:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0 if p_o == 1.0 else 0.0

    # 3. Effect size agreement
    valid_eff = np.isfinite(eff_a) & np.isfinite(eff_b)
    if np.sum(valid_eff) >= 3:
        eff_r, _ = scipy_stats.pearsonr(eff_a[valid_eff], eff_b[valid_eff])
        eff_rmse = np.sqrt(np.mean((eff_a[valid_eff] - eff_b[valid_eff]) ** 2))
    else:
        eff_r, eff_rmse = np.nan, np.nan

    # 4. Direction agreement
    if np.sum(valid_eff) >= 3:
        same_sign = np.sign(eff_a[valid_eff]) == np.sign(eff_b[valid_eff])
        dir_agree = float(np.mean(same_sign))
    else:
        dir_agree = np.nan

    return ConcordanceMetrics(
        method_a=results_a[0].method,
        method_b=results_b[0].method,
        n_cliques_compared=n,
        spearman_rho=float(rho),
        spearman_pvalue=float(rho_pval),
        threshold=threshold,
        n_both_significant=n_both_sig,
        n_both_nonsignificant=n_both_nonsig,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        cohen_kappa=float(kappa),
        effect_pearson_r=float(eff_r) if np.isfinite(eff_r) else np.nan,
        effect_rmse=float(eff_rmse) if np.isfinite(eff_rmse) else np.nan,
        direction_agreement_frac=dir_agree,
    )


def identify_disagreements(
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]],
    threshold: float = 0.05,
    min_disagreement_count: int = 1,
) -> pd.DataFrame:
    """
    Identify cliques where methods disagree on significance.

    A clique is flagged if:
    - At least one method calls it significant (p < threshold)
    - At least one method calls it non-significant (p >= threshold)

    Returns:
        DataFrame with disagreement cases and method-by-method p-values
    """
    # Build wide format
    all_ids = set()
    for results in results_by_method.values():
        all_ids.update(r.clique_id for r in results if r.is_valid)

    rows = []
    for cid in sorted(all_ids):
        row = {'clique_id': cid}
        sig_count = 0
        nonsig_count = 0

        for method, results in results_by_method.items():
            result = next((r for r in results if r.clique_id == cid and r.is_valid), None)
            if result:
                row[f'{method.value}_pvalue'] = result.p_value
                row[f'{method.value}_effect'] = result.effect_size
                if result.p_value < threshold:
                    sig_count += 1
                else:
                    nonsig_count += 1
            else:
                row[f'{method.value}_pvalue'] = np.nan
                row[f'{method.value}_effect'] = np.nan

        row['n_methods_significant'] = sig_count
        row['n_methods_nonsignificant'] = nonsig_count
        row['is_disagreement'] = sig_count > 0 and nonsig_count > 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Filter to disagreements
    disagreements = df[df['is_disagreement']].copy()
    disagreements = disagreements.sort_values('n_methods_significant', ascending=False)

    return disagreements
```

---

## Main Entry Point

```python
def run_method_comparison(
    data: np.ndarray,
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    cliques: list[CliqueDefinition],
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    methods: list[CliqueTestMethod] | None = None,
    concordance_threshold: float = 0.05,
    normalization_method: str = "median",
    imputation_method: str = "min_feature",
    n_rotations: int = 9999,
    n_permutations: int = 10000,
    use_gpu: bool = True,
    seed: int | None = None,
    verbose: bool = True,
) -> MethodComparisonResult:
    """
    Run multiple differential testing methods and compute concordance.

    This is the main entry point for method comparison. Pipeline:
    1. Prepare data ONCE (shared preprocessing)
    2. Run each method on the prepared data
    3. Compute pairwise concordance metrics
    4. Identify disagreement cases
    5. Return structured comparison results

    Statistical Note:
        Results are DESCRIPTIVE. Do NOT:
        - Select the "best" p-value per clique (inflates FDR)
        - Combine p-values (requires careful assumptions)
        - Declare a "winner" method

        Instead, use concordance to:
        - Validate findings (high agreement = robust)
        - Understand method behavior
        - Flag interesting disagreements

    Args:
        data: Expression matrix (n_features, n_samples)
        feature_ids: Gene/protein identifiers
        sample_metadata: Sample annotations
        cliques: Clique definitions
        condition_column: Metadata column for condition
        contrast: (test_condition, reference_condition)
        subject_column: Optional column for biological replicates
        methods: List of methods (default: OLS, ROAST_MSQ, ROAST_MEAN, PERMUTATION)
        concordance_threshold: P-value threshold for classification agreement
        normalization_method: Preprocessing normalization
        imputation_method: Preprocessing imputation
        n_rotations: Rotations for ROAST
        n_permutations: Permutations for permutation test
        use_gpu: GPU acceleration
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        MethodComparisonResult with:
        - Per-method results
        - Pairwise concordance metrics
        - Disagreement analysis
        - Wide-format combined table

    Example:
        >>> comparison = run_method_comparison(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ... )
        >>>
        >>> # Check overall agreement
        >>> print(f"Mean Spearman ρ: {comparison.mean_spearman_rho:.3f}")
        >>>
        >>> # Get robust hits (significant in all methods)
        >>> robust = comparison.robust_hits(threshold=0.05)
        >>>
        >>> # Get ROAST-specific hits (bidirectional regulation?)
        >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
        >>>
        >>> # Export wide format for downstream analysis
        >>> wide_df = comparison.wide_format()
        >>> wide_df.to_csv("method_comparison.csv")
    """
    # 1. Prepare experiment (shared preprocessing)
    if verbose:
        print("=" * 60)
        print("METHOD COMPARISON FRAMEWORK")
        print("=" * 60)
        print()

    experiment = prepare_experiment(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=sample_metadata,
        cliques=cliques,
        condition_column=condition_column,
        contrast=contrast,
        subject_column=subject_column,
        normalization_method=normalization_method,
        imputation_method=imputation_method,
        verbose=verbose,
    )

    # 2. Default methods if not specified
    if methods is None:
        methods = [
            OLSMethod(),
            ROASTMethod(statistic="msq"),
            ROASTMethod(statistic="mean"),
            PermutationMethod(n_permutations=n_permutations),
        ]
        if subject_column is not None:
            methods.insert(1, LMMMethod())

    if verbose:
        print()
        print(f"Methods to run: {[m.name.value for m in methods]}")
        print()

    # 3. Run each method
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]] = {}

    for method in methods:
        if verbose:
            print(f"Running {method.name.value}...", end=" ", flush=True)

        try:
            results = method.test(
                experiment,
                use_gpu=use_gpu,
                seed=seed,
                verbose=False,  # Method-level verbosity off
            )
            results_by_method[method.name] = results

            if verbose:
                n_sig = sum(1 for r in results if r.p_value < concordance_threshold)
                print(f"done ({len(results)} cliques, {n_sig} significant)")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results_by_method[method.name] = []

    # 4. Compute pairwise concordance
    if verbose:
        print()
        print("Computing concordance metrics...")

    method_names = [m for m in results_by_method.keys() if len(results_by_method[m]) > 0]
    pairwise = []

    for i, name_a in enumerate(method_names):
        for name_b in method_names[i + 1:]:
            try:
                conc = compute_pairwise_concordance(
                    results_by_method[name_a],
                    results_by_method[name_b],
                    threshold=concordance_threshold,
                )
                pairwise.append(conc)
            except ValueError as e:
                if verbose:
                    print(f"  Skipping {name_a.value} vs {name_b.value}: {e}")

    # 5. Aggregate metrics
    if pairwise:
        mean_rho = float(np.mean([c.spearman_rho for c in pairwise]))
        mean_kappa = float(np.mean([c.cohen_kappa for c in pairwise]))
    else:
        mean_rho, mean_kappa = np.nan, np.nan

    # 6. Identify disagreements
    disagreements = identify_disagreements(
        results_by_method,
        threshold=concordance_threshold
    )

    # 7. Count cliques tested
    all_tested = set()
    for results in results_by_method.values():
        all_tested.update(r.clique_id for r in results if r.is_valid)

    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Cliques tested: {len(all_tested)}")
        print(f"Mean Spearman ρ: {mean_rho:.3f}")
        print(f"Mean Cohen's κ: {mean_kappa:.3f}")
        print(f"Disagreement cases: {len(disagreements)}")
        print()

        # Quick concordance table
        if pairwise:
            print("Pairwise concordance (Spearman ρ):")
            for conc in pairwise:
                print(f"  {conc.method_a.value} vs {conc.method_b.value}: "
                      f"ρ={conc.spearman_rho:.3f}, κ={conc.cohen_kappa:.3f}")

    return MethodComparisonResult(
        results_by_method=results_by_method,
        pairwise_concordance=pairwise,
        mean_spearman_rho=mean_rho,
        mean_cohen_kappa=mean_kappa,
        disagreement_cases=disagreements,
        preprocessing_params=experiment.preprocessing_params,
        methods_run=list(method_names),
        n_cliques_tested=len(all_tested),
    )
```

---

## Implementation Waves

### Wave 1: Core Abstractions
**Agent Profile**: Software architect with deep Python typing and dataclass expertise

**Deliverables**:
1. `MethodName` enum
2. `UnifiedCliqueResult` dataclass (frozen)
3. `PreparedCliqueExperiment` dataclass (frozen)
4. `CliqueTestMethod` protocol
5. `prepare_experiment()` factory function
6. Unit tests for all dataclasses

**Files to read**:
- `src/cliquefinder/stats/clique_analysis.py` (existing patterns)
- `src/cliquefinder/stats/permutation_framework.py` (protocol patterns)
- This spec document

**Acceptance criteria**:
- All dataclasses pass mypy strict mode
- `prepare_experiment()` produces identical output to existing preprocessing
- 100% test coverage on new code

---

### Wave 2: Method Adapters
**Agent Profile**: Statistical computing specialist with OLS/LMM/rotation expertise

**Deliverables**:
1. `OLSMethod` class (wrap existing differential)
2. `LMMMethod` class (wrap existing differential with mixed=True)
3. `ROASTMethod` class (wrap RotationTestEngine)
4. `PermutationMethod` class (wrap permutation_gpu)
5. Integration tests comparing adapter output to direct method calls

**Files to read**:
- `src/cliquefinder/stats/differential.py` (OLS/LMM)
- `src/cliquefinder/stats/rotation.py` (ROAST)
- `src/cliquefinder/stats/permutation_gpu.py` (Permutation)
- Wave 1 output (core abstractions)

**Acceptance criteria**:
- Each adapter produces correct `UnifiedCliqueResult`
- Adapter results match direct method calls within tolerance
- Memory-efficient (no unnecessary copies)

---

### Wave 3: Concordance Analysis
**Agent Profile**: Biostatistician with expertise in method comparison and meta-analysis

**Deliverables**:
1. `ConcordanceMetrics` dataclass
2. `compute_pairwise_concordance()` function
3. `identify_disagreements()` function
4. `MethodComparisonResult` dataclass with all helper methods
5. Visualization helpers (concordance heatmap data)

**Files to read**:
- Wave 1 & 2 output
- `src/cliquefinder/stats/correlation_tests.py` (related patterns)

**Acceptance criteria**:
- Concordance metrics match manual calculation on test data
- Disagreement identification correct for edge cases
- All `MethodComparisonResult` methods tested

---

### Wave 4: Integration and CLI
**Agent Profile**: Full-stack engineer with CLI and API design experience

**Deliverables**:
1. `run_method_comparison()` main entry point
2. CLI integration in `src/cliquefinder/cli/differential.py`
3. Example script in `examples/`
4. Integration tests with real data subset

**Files to read**:
- Waves 1-3 output
- `src/cliquefinder/cli/differential.py` (CLI patterns)
- `examples/als/` (example patterns)

**Acceptance criteria**:
- CLI works end-to-end: `cliquefinder compare-methods --help`
- Example script runs without error
- Results reproducible with seed

---

## Testing Strategy

### Unit Tests
```python
# tests/test_method_comparison.py

class TestUnifiedCliqueResult:
    def test_frozen(self):
        """Result is immutable."""
        result = UnifiedCliqueResult(...)
        with pytest.raises(FrozenInstanceError):
            result.clique_id = "new_id"

    def test_is_valid(self):
        """is_valid correctly identifies valid results."""
        valid = UnifiedCliqueResult(p_value=0.05, effect_size=1.0, ...)
        assert valid.is_valid

        invalid = UnifiedCliqueResult(p_value=np.nan, ...)
        assert not invalid.is_valid

class TestPreparedExperiment:
    def test_immutable_data(self):
        """Data cannot be modified after preparation."""
        exp = prepare_experiment(...)
        with pytest.raises(ValueError):
            exp.data[0, 0] = 999

class TestConcordance:
    def test_perfect_agreement(self):
        """Identical results have rho=1, kappa=1."""
        results = [UnifiedCliqueResult(...) for _ in range(10)]
        conc = compute_pairwise_concordance(results, results)
        assert conc.spearman_rho == pytest.approx(1.0)
        assert conc.cohen_kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        """Opposite rankings have negative rho."""
        # ... test with reversed p-values
```

### Integration Tests
```python
class TestMethodComparison:
    @pytest.fixture
    def small_dataset(self):
        """Minimal dataset for fast testing."""
        return load_test_data("small_proteomics.h5")

    def test_full_pipeline(self, small_dataset):
        """End-to-end test with all methods."""
        result = run_method_comparison(
            data=small_dataset.data,
            feature_ids=small_dataset.features,
            sample_metadata=small_dataset.metadata,
            cliques=small_dataset.cliques[:10],
            condition_column='phenotype',
            contrast=('CASE', 'CTRL'),
            n_rotations=99,  # Fast for testing
            n_permutations=99,
        )

        assert result.n_cliques_tested == 10
        assert len(result.methods_run) >= 3
        assert result.wide_format().shape[0] == 10
```

---

## Usage Examples

### Basic Usage
```python
from cliquefinder.stats import run_method_comparison, MethodName

# Run comparison
comparison = run_method_comparison(
    data=expression_matrix,
    feature_ids=protein_ids,
    sample_metadata=metadata,
    cliques=clique_definitions,
    condition_column='phenotype',
    contrast=('ALS', 'Control'),
    subject_column='subject_id',
)

# Get robust hits (significant in all methods)
robust_cliques = comparison.robust_hits(threshold=0.05)
print(f"Robust discoveries: {len(robust_cliques)}")

# Get ROAST-specific (bidirectional regulation?)
roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
print(f"ROAST-only discoveries: {len(roast_only)}")

# Export for downstream analysis
wide_df = comparison.wide_format()
wide_df.to_csv("method_comparison.csv", index=False)
```

### Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Concordance heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    comparison.concordance_matrix(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    vmin=-1, vmax=1,
    ax=ax,
)
ax.set_title("Method Concordance (Spearman ρ)")
plt.tight_layout()
plt.savefig("concordance_heatmap.png")

# P-value scatter
wide = comparison.wide_format()
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (m1, m2) in zip(axes, [
    ('ols', 'roast_msq'),
    ('ols', 'permutation_competitive'),
    ('roast_msq', 'permutation_competitive'),
]):
    ax.scatter(
        -np.log10(wide[f'{m1}_pvalue']),
        -np.log10(wide[f'{m2}_pvalue']),
        alpha=0.5,
    )
    ax.set_xlabel(f'-log10(p) {m1}')
    ax.set_ylabel(f'-log10(p) {m2}')
    ax.plot([0, 5], [0, 5], 'k--', alpha=0.3)

plt.tight_layout()
plt.savefig("pvalue_scatter.png")
```

---

## References

1. Wu D, Smyth GK (2012). "Camera: a competitive gene set test accounting for inter-gene correlation." *Nucleic Acids Research* 40(17):e133.

2. Subramanian A, et al. (2005). "Gene set enrichment analysis: A knowledge-based approach." *PNAS* 102(43):15545-50.

3. Benjamini Y, Hochberg Y (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *JRSS-B* 57(1):289-300.

4. Cohen J (1960). "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement* 20(1):37-46.

5. Ritchie ME, et al. (2015). "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research* 43(7):e47.
