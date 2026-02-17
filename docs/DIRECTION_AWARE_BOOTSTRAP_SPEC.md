# Direction-Aware Bootstrap Stability Assessment

## Engineering Specification v1.0

**Author**: Distinguished Engineering Review
**Date**: 2026-02-01
**Status**: Ready for Implementation

---

## 1. Problem Statement

### Current Limitations

The current bootstrap stability assessment (`bootstrap_comparison.py`) treats all cliques identically, requiring concordance between OLS and ROAST_MSQ methods for a clique to be classified as "stable". This is methodologically flawed for **mixed-sign cliques**.

**The Core Issue**: OLS+TMP (Tukey's Median Polish) aggregation assumes directional coherence—all features moving together. For MIXED cliques (both positive and negative correlations), effects cancel during aggregation, making OLS results **invalid/meaningless**. However, ROAST_MSQ (mean squared t-statistics) is **direction-agnostic** and remains valid.

### Data Flow Analysis

```
Upstream (clique_validator.py)    →    Bootstrap (bootstrap_comparison.py)
─────────────────────────────────────────────────────────────────────────
ChildSetType2 contains:           →    Current: Direction info NOT used
  - direction: POSITIVE/NEGATIVE/MIXED
  - signed_mean_correlation
  - n_positive_edges / n_negative_edges
  - clique_genes

CliqueDefinition (CSV load)       →    Direction info NOT in cliques.csv
  - regulator
  - clique_genes (only)
```

### Requirements

1. **Direction-Aware Stability**: Different criteria for coherent (POSITIVE/NEGATIVE) vs MIXED cliques
2. **Per-Method Stability**: Report stability metrics for each method independently
3. **Principled Cross-Method Comparison**: Only compare methods where both are valid
4. **Backward Compatibility**: Existing workflows must not break
5. **Computational Efficiency**: No additional method runs; only change aggregation logic

---

## 2. Method Validity Matrix

| Clique Direction | OLS+TMP Valid? | ROAST_MSQ Valid? | Stability Criterion |
|------------------|----------------|------------------|---------------------|
| POSITIVE         | ✓ Yes          | ✓ Yes            | Both methods ≥80%   |
| NEGATIVE         | ✓ Yes          | ✓ Yes            | Both methods ≥80%   |
| MIXED            | ✗ Invalid      | ✓ Yes            | ROAST only ≥80%     |

### Statistical Rationale

**OLS+TMP for Coherent Cliques**:
- TMP assumes additive structure: `y_ij = μ + α_i + β_j + ε_ij`
- All features contribute with same sign to aggregate
- Valid when all correlations are positive (co-activation) or all negative (anti-correlation)

**OLS+TMP for Mixed Cliques (INVALID)**:
- Effects cancel: +1 and -1 average to 0
- Loses signal completely for balanced mixed cliques
- May produce spurious "significant" results for unbalanced mixed cliques
- **Should NOT be used for stability assessment**

**ROAST_MSQ for All Cliques**:
- Uses mean squared t-statistics: `MSQ = (1/k) Σ t_i²`
- Direction-agnostic: both up and down contribute positively
- Valid regardless of correlation structure

---

## 3. Data Model Changes

### 3.1 CliqueDefinition Enhancement

**File**: `src/cliquefinder/stats/clique_analysis.py`

```python
@dataclass
class CliqueDefinition:
    """Definition of a clique for differential testing."""
    regulator: str
    proteins: list[str]
    # NEW: Direction information
    direction: str = "unknown"  # "positive", "negative", "mixed", "unknown"
    n_positive_edges: int = 0
    n_negative_edges: int = 0
    signed_mean_correlation: float | None = None

    @property
    def clique_id(self) -> str:
        return self.regulator

    @property
    def is_coherent(self) -> bool:
        """True if POSITIVE or NEGATIVE (not MIXED or unknown)."""
        return self.direction in ("positive", "negative")

    @property
    def is_mixed(self) -> bool:
        """True if clique has mixed correlation signs."""
        return self.direction == "mixed"
```

### 3.2 Enhanced Bootstrap Result

**File**: `src/cliquefinder/stats/bootstrap_comparison.py`

```python
@dataclass
class BootstrapCliqueResult:
    """Aggregated results for a single clique across bootstrap iterations."""

    clique_id: str
    n_bootstraps: int

    # Direction information (from input clique)
    direction: str  # "positive", "negative", "mixed", "unknown"
    n_positive_edges: int
    n_negative_edges: int

    # Per-method selection frequencies
    selection_freq_ols: float
    selection_freq_roast: float
    selection_freq_any: float   # Either method significant
    selection_freq_both: float  # Both methods significant (only for coherent)

    # Per-method stability flags (NEW)
    is_stable_ols: bool      # OLS ≥ stability_threshold
    is_stable_roast: bool    # ROAST ≥ stability_threshold

    # Direction-aware combined stability (NEW)
    is_robust: bool  # Direction-appropriate stability criterion met
    stability_criterion: str  # "both_methods" or "roast_only"

    # P-value summaries
    median_pvalue_ols: float
    median_pvalue_roast: float
    pvalue_ci_low_ols: float
    pvalue_ci_high_ols: float
    pvalue_ci_low_roast: float   # NEW
    pvalue_ci_high_roast: float  # NEW

    # Effect size summaries (OLS)
    mean_effect_ols: float
    median_effect_ols: float
    effect_ci_low_ols: float
    effect_ci_high_ols: float
    effect_std_ols: float

    # Effect size summaries (ROAST) - NEW
    mean_msq_roast: float      # Mean of MSQ statistics across bootstraps
    median_msq_roast: float

    # Method concordance (only meaningful for coherent cliques)
    method_concordance: float | None  # None for mixed cliques

    # Legacy compatibility
    @property
    def is_stable(self) -> bool:
        """Legacy compatibility: alias for is_robust."""
        return self.is_robust
```

### 3.3 BootstrapConfig Enhancement

```python
@dataclass
class BootstrapConfig:
    """Configuration for bootstrap subsampling analysis."""

    # ... existing fields ...

    # Direction-aware stability (NEW)
    require_direction_info: bool = True  # If True, error if direction unknown
    mixed_clique_criterion: str = "roast_only"  # or "exclude"
    coherent_clique_criterion: str = "both_methods"  # or "either_method"
```

---

## 4. Algorithm: Direction-Aware Aggregation

### 4.1 Pseudocode

```python
def aggregate_bootstrap_results(
    clique_id: str,
    direction: str,
    bootstrap_results: list[tuple[float, float, float, float]],  # (ols_p, roast_p, ols_effect, roast_msq)
    config: BootstrapConfig,
) -> BootstrapCliqueResult:
    """
    Aggregate per-bootstrap results with direction awareness.

    Args:
        clique_id: Clique identifier
        direction: "positive", "negative", "mixed", or "unknown"
        bootstrap_results: List of (ols_pvalue, roast_pvalue, ols_effect, roast_msq) tuples
        config: Bootstrap configuration

    Returns:
        BootstrapCliqueResult with direction-appropriate stability flags
    """
    # Extract arrays
    ols_pvals = np.array([r[0] for r in bootstrap_results])
    roast_pvals = np.array([r[1] for r in bootstrap_results])
    ols_effects = np.array([r[2] for r in bootstrap_results])
    roast_msqs = np.array([r[3] for r in bootstrap_results])

    # Clean NaN
    valid_ols = ~np.isnan(ols_pvals)
    valid_roast = ~np.isnan(roast_pvals)

    # Per-method selection frequencies
    sig_ols = ols_pvals[valid_ols] < config.significance_threshold
    sig_roast = roast_pvals[valid_roast] < config.significance_threshold

    selection_freq_ols = sig_ols.mean()
    selection_freq_roast = sig_roast.mean()

    # Per-method stability
    is_stable_ols = selection_freq_ols >= config.stability_threshold
    is_stable_roast = selection_freq_roast >= config.stability_threshold

    # Direction-aware combined stability
    is_coherent = direction in ("positive", "negative")

    if is_coherent:
        # Both methods must be stable
        is_robust = is_stable_ols and is_stable_roast
        stability_criterion = "both_methods"

        # Concordance is meaningful
        valid_both = valid_ols & valid_roast
        agree = (ols_pvals[valid_both] < config.significance_threshold) == \
                (roast_pvals[valid_both] < config.significance_threshold)
        method_concordance = agree.mean()
    else:
        # Mixed or unknown: only ROAST is valid
        is_robust = is_stable_roast
        stability_criterion = "roast_only"
        method_concordance = None  # Not meaningful

    # ... compute p-value and effect summaries ...

    return BootstrapCliqueResult(
        clique_id=clique_id,
        direction=direction,
        # ... all fields ...
        is_stable_ols=is_stable_ols,
        is_stable_roast=is_stable_roast,
        is_robust=is_robust,
        stability_criterion=stability_criterion,
        method_concordance=method_concordance,
    )
```

### 4.2 Vectorized Implementation

For computational efficiency with 1672 cliques × 100 bootstraps:

```python
def aggregate_all_cliques_vectorized(
    clique_directions: np.ndarray,  # (n_cliques,) dtype=str
    ols_pvals: np.ndarray,          # (n_cliques, n_bootstraps)
    roast_pvals: np.ndarray,        # (n_cliques, n_bootstraps)
    config: BootstrapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized stability computation.

    Returns:
        is_stable_ols: (n_cliques,) bool array
        is_stable_roast: (n_cliques,) bool array
        is_robust: (n_cliques,) bool array with direction-aware logic
    """
    threshold = config.significance_threshold
    stability = config.stability_threshold

    # Selection frequencies (vectorized across cliques)
    sig_ols = ols_pvals < threshold  # (n_cliques, n_bootstraps) bool
    sig_roast = roast_pvals < threshold

    freq_ols = np.nanmean(sig_ols, axis=1)  # (n_cliques,)
    freq_roast = np.nanmean(sig_roast, axis=1)

    # Per-method stability
    is_stable_ols = freq_ols >= stability
    is_stable_roast = freq_roast >= stability

    # Direction-aware combined stability
    is_coherent = np.isin(clique_directions, ["positive", "negative"])

    # Coherent: both methods must be stable
    # Mixed: only ROAST must be stable
    is_robust = np.where(
        is_coherent,
        is_stable_ols & is_stable_roast,  # Both required
        is_stable_roast                    # ROAST only
    )

    return is_stable_ols, is_stable_roast, is_robust
```

---

## 5. CSV Input/Output Schema

### 5.1 Input: cliques.csv (Enhanced)

Must include direction columns. If missing, warn and treat as "unknown".

```csv
regulator,condition,n_samples,n_indra_targets,n_coherent_genes,coherence_ratio,direction,signed_mean_correlation,n_positive_edges,n_negative_edges,clique_genes
GCG,all,379,100,11,0.11,positive,0.782,55,0,"ACAT1,CASK,..."
SIRT3,all,379,80,20,0.25,mixed,0.123,150,40,"ACO2,AIFM1,..."
```

### 5.2 Output: bootstrap_summary.csv (Enhanced)

```csv
clique_id,direction,n_positive_edges,n_negative_edges,n_bootstraps,selection_freq_ols,selection_freq_roast,is_stable_ols,is_stable_roast,is_robust,stability_criterion,method_concordance,median_pvalue_ols,median_pvalue_roast,...
GCG,positive,55,0,100,0.92,0.88,True,True,True,both_methods,0.85,0.012,0.018,...
SIRT3,mixed,150,40,100,0.45,0.91,False,True,True,roast_only,,0.234,0.008,...
```

### 5.3 Output: bootstrap_stable_hits.csv

Split into two files for clarity:

- `bootstrap_stable_coherent.csv` - POSITIVE/NEGATIVE cliques stable by both methods
- `bootstrap_stable_mixed.csv` - MIXED cliques stable by ROAST

---

## 6. Implementation Tasks

### Task 1: Enhance CliqueDefinition and CSV Loading

**Expert**: Data Engineering / IO Specialist
**Files**:
- `src/cliquefinder/stats/clique_analysis.py`
- `src/cliquefinder/io/loaders.py`

**Subtasks**:
1. Add direction fields to `CliqueDefinition` dataclass
2. Update `load_clique_definitions()` to parse direction columns
3. Handle missing direction columns gracefully (warn, default to "unknown")
4. Add validation: if direction present, verify n_pos + n_neg > 0

**Test Cases**:
- Load CSV with direction columns → direction populated
- Load CSV without direction columns → direction = "unknown", warning logged
- Load CSV with malformed direction → error with clear message

### Task 2: Modify Bootstrap Result Dataclass and Aggregation

**Expert**: Statistical Methods / Bootstrap Specialist
**Files**:
- `src/cliquefinder/stats/bootstrap_comparison.py`

**Subtasks**:
1. Enhance `BootstrapCliqueResult` with direction-aware fields
2. Add per-method stability flags (`is_stable_ols`, `is_stable_roast`)
3. Implement `is_robust` with direction-aware logic
4. Add `stability_criterion` field for transparency
5. Make `method_concordance` nullable for mixed cliques
6. Implement vectorized aggregation for efficiency

**Test Cases**:
- Coherent clique with both methods stable → is_robust = True
- Coherent clique with only OLS stable → is_robust = False
- Mixed clique with ROAST stable → is_robust = True
- Mixed clique with only OLS stable → is_robust = False
- Unknown direction → treated as mixed (conservative)

### Task 3: Update Bootstrap Runner and Output

**Expert**: Pipeline Engineering / CLI Specialist
**Files**:
- `src/cliquefinder/stats/bootstrap_comparison.py` (run_bootstrap_comparison)
- `src/cliquefinder/cli/compare.py`

**Subtasks**:
1. Accept direction info from CliqueDefinition in run_bootstrap_comparison
2. Pass direction to aggregation function
3. Update CSV output schema with new columns
4. Generate separate stable hits files by clique type
5. Update verbose logging to report by direction category
6. Add `--mixed-criterion` CLI flag (roast_only | exclude)

**Output Summary Format**:
```
BOOTSTRAP STABILITY SUMMARY
═══════════════════════════
Coherent cliques (POSITIVE/NEGATIVE): 1200 tested
  Stable (both OLS & ROAST ≥80%): 12 (1.0%)

Mixed cliques: 472 tested
  Stable (ROAST ≥80%): 8 (1.7%)

Total robust cliques: 20
```

### Task 4: Re-generate cliques.csv with Direction Info

**Expert**: Pipeline Integration
**Files**:
- `src/cliquefinder/cli/_analyze_core.py` (already modified per summary)
- Run analyze command on proteomics data

**Note**: This was reportedly done in previous session. Verify by checking if output CSV has direction columns. If not, re-run analysis.

---

## 7. Testing Strategy

### Unit Tests

```python
def test_coherent_clique_requires_both_methods():
    """Coherent clique must have both OLS and ROAST stable."""
    result = aggregate_bootstrap_results(
        clique_id="TEST",
        direction="positive",
        bootstrap_results=[...],  # OLS stable, ROAST unstable
        config=BootstrapConfig(stability_threshold=0.8),
    )
    assert result.is_stable_ols == True
    assert result.is_stable_roast == False
    assert result.is_robust == False  # Coherent requires BOTH
    assert result.stability_criterion == "both_methods"

def test_mixed_clique_only_needs_roast():
    """Mixed clique only needs ROAST stable."""
    result = aggregate_bootstrap_results(
        clique_id="TEST",
        direction="mixed",
        bootstrap_results=[...],  # OLS unstable, ROAST stable
        config=BootstrapConfig(stability_threshold=0.8),
    )
    assert result.is_stable_ols == False
    assert result.is_stable_roast == True
    assert result.is_robust == True  # Mixed only needs ROAST
    assert result.stability_criterion == "roast_only"
    assert result.method_concordance is None  # Not meaningful for mixed

def test_unknown_direction_treated_as_mixed():
    """Unknown direction uses conservative mixed criterion."""
    result = aggregate_bootstrap_results(
        clique_id="TEST",
        direction="unknown",
        bootstrap_results=[...],
        config=BootstrapConfig(),
    )
    assert result.stability_criterion == "roast_only"
```

### Integration Tests

1. Full pipeline with cliques.csv containing direction info
2. Verify output CSV has all new columns
3. Verify stable hits files are split correctly
4. Performance benchmark: 1672 cliques × 100 bootstraps < 10 minutes

---

## 8. Confounding Factors Addressed

| Concern | Mitigation |
|---------|------------|
| Bootstrap correlation between proteins | ROAST rotations preserve correlation structure; bootstrap captures sample-level uncertainty |
| Multiple testing (1672 cliques) | FDR correction applied; stability threshold (80%) is already conservative |
| TMP homogeneity assumption violated for mixed | Now excluded from combined stability for mixed cliques |
| Effect heterogeneity | Per-method CI reported; flagged in output |
| Small control samples (28) | Hybrid bootstrap (fixed controls) already implemented |

---

## 9. Backward Compatibility

1. **`is_stable` property**: Aliased to `is_robust` for existing code
2. **Legacy CSV loading**: Works with old format (direction defaults to "unknown")
3. **CLI unchanged**: New flags are optional with sensible defaults
4. **Existing test scripts**: Will continue to work

---

## 10. Implementation Order

1. **Task 1**: CliqueDefinition enhancement (prerequisite for all)
2. **Task 4**: Re-generate cliques.csv if needed
3. **Task 2**: Bootstrap result dataclass and aggregation logic
4. **Task 3**: CLI and output formatting
5. **Testing**: Unit tests, integration tests, validation run

---

## Appendix A: Method Comparison Table

| Aspect | OLS+TMP | ROAST_MSQ |
|--------|---------|-----------|
| Test type | Parametric contrast | Self-contained rotation |
| Null hypothesis | μ_case = μ_ref (aggregate) | Genes not enriched for DE |
| Effect direction | Directional (up/down) | Direction-agnostic |
| Handles mixed signs | ✗ No (effects cancel) | ✓ Yes (squared terms) |
| Sample correlation | Assumed independent | Rotations preserve structure |
| Output | log2FC, t-stat, p-value | MSQ stat, p-value |

## Appendix B: Expected Output Distribution

Based on current data (1672 cliques, ~60% coherent, ~40% mixed estimated):

| Category | N (est.) | Expected Stable (1%) |
|----------|----------|----------------------|
| POSITIVE | 800 | 8 |
| NEGATIVE | 200 | 2 |
| MIXED | 672 | 6-10 (ROAST only) |
| **Total** | 1672 | 16-20 |

Note: Current 0 stable hits may increase slightly due to more lenient criterion for mixed cliques.
