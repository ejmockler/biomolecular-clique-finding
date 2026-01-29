# Correlation Sign Preservation Engineering Specification

**Status**: Implemented
**Author**: Distinguished Engineer Review
**Date**: 2025-01-28
**Implementation Date**: 2025-01-28
**Priority**: High - Data Integrity Issue

---

## Executive Summary

The cliquefinder pipeline has an inconsistency in correlation sign handling that loses biologically meaningful information. Positively-correlated cliques (co-activation) and negatively-correlated cliques (anti-correlation/repression) are currently indistinguishable in output, despite representing fundamentally different regulatory biology.

**Implementation Summary (2025-01-28):**
All work packages (WP-1 through WP-7) have been completed. The feature now preserves signed correlation values throughout the pipeline while maintaining backward compatibility. Key additions include:
- `CorrelationDirection` enum for classifying clique types (POSITIVE, NEGATIVE, MIXED)
- Signed correlation statistics fields in all dataclasses (CorrelationClique, ChildSetType2, CoexpressionModule, CliqueDefinition, CliqueDifferentialResult)
- Updated `find_cliques()` and `find_maximum_clique()` to compute and populate signed statistics
- Complete documentation updates across all affected modules
- Runtime warnings for mixed-sign cliques in differential analysis (see Section 8.1)

---

## 1. Current State Analysis

### 1.1 Correlation Flow Through Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORRELATION DATA FLOW AUDIT                              │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Correlation Matrix Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/cliquefinder/knowledge/clique_validator.py
Method: compute_correlation_matrix() [lines 720-860]

  Input:  Gene expression data (n_genes × n_samples)
  Output: Correlation matrix with SIGNED values [-1, +1]

  ✅ CORRECT: Returns signed Pearson/Spearman correlations

  For "max" method (lines 842-851):
    - Computes both Pearson and Spearman
    - Takes element-wise max of |r|, PRESERVING SIGN of stronger
    - Example: Pearson=+0.7, Spearman=-0.8 → returns -0.8


STEP 2: Graph Construction
━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/cliquefinder/knowledge/clique_validator.py
Method: build_correlation_graph() [lines 862-1066]

  Threshold Check (line 1004):
    mask = np.abs(upper_values) >= min_correlation

    ✅ CORRECT: Uses |r| for threshold (r=-0.8 passes 0.7 threshold)

  Edge Weight Storage (lines 1009, 1018):
    weights = upper_values[mask]  # Preserves sign
    G.add_edge(g1, g2, weight=corr_val)

    ✅ CORRECT: Graph edges store SIGNED correlation values


STEP 3: Clique Statistics Computation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/cliquefinder/knowledge/clique_validator.py
Method: find_cliques() [lines 1069-1295]

  Edge correlation extraction (line 1193):
    edge_corrs = [abs(G[u][v]['weight'])
                  for u, v in itertools.combinations(clique_nodes, 2)]

    ⚠️ BUG: Takes absolute value - LOSES SIGN INFORMATION

  Statistics computation (lines 1199-1200):
    mean_correlation=float(np.mean(edge_corrs))   # Always positive
    min_correlation=float(np.min(edge_corrs))     # Always positive

    ⚠️ BUG: Both metrics are absolute - cannot distinguish +/- cliques


STEP 4: Clique Summarization (Differential Analysis Path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/cliquefinder/stats/summarization.py
Method: summarize_clique() [lines 256-305]

  Coherence computation (lines 292-296):
    corr_matrix = np.corrcoef(valid_data)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    coherence = float(np.mean(upper_tri))

    ✅ CORRECT: Uses signed correlations

  BUT: This is RE-COMPUTED from expression data, not from graph edges
       So it may differ from clique discovery statistics


STEP 5: Results Output
━━━━━━━━━━━━━━━━━━━━━━
File: src/cliquefinder/knowledge/clique_validator.py
Dataclass: CorrelationClique [lines 112-142]

  @dataclass
  class CorrelationClique:
      genes: Set[str]
      condition: str
      mean_correlation: float   # ⚠️ Absolute value (always positive)
      min_correlation: float    # ⚠️ Absolute value (always positive)
      size: int

  MISSING FIELDS:
    - correlation_direction: Literal["positive", "negative", "mixed"]
    - signed_mean_correlation: float
    - n_positive_edges: int
    - n_negative_edges: int

File: src/cliquefinder/knowledge/module_discovery.py
Dataclass: CoexpressionModule [lines 355-391]

  Same issue - inherits absolute-value semantics from CorrelationClique
```

### 1.2 Affected Code Locations

| File | Lines | Function/Class | Issue |
|------|-------|----------------|-------|
| `knowledge/clique_validator.py` | 1193 | `find_cliques()` | `abs()` on edge weights |
| `knowledge/clique_validator.py` | 1393 | `find_maximum_clique()` | `abs()` on edge weights |
| `knowledge/clique_validator.py` | 112-142 | `CorrelationClique` | Missing sign fields |
| `knowledge/module_discovery.py` | 355-391 | `CoexpressionModule` | Missing sign fields |
| `stats/clique_analysis.py` | 62-80 | `CliqueDefinition` | Missing sign fields |

### 1.3 Biological Significance

```
POSITIVE CORRELATION (r > 0):
  Gene A ↑  →  Gene B ↑  (co-activation)

  Biological meaning:
    - Shared activating TF
    - Same pathway activation
    - Co-regulated module

NEGATIVE CORRELATION (r < 0):
  Gene A ↑  →  Gene B ↓  (anti-correlation)

  Biological meaning:
    - Repressor relationship
    - Mutual exclusion (cell state switching)
    - Feedback inhibition
    - Opposing pathway arms

MIXED CORRELATION CLIQUE:
  Some edges positive, some negative

  Biological meaning:
    - Complex regulatory logic
    - Feed-forward loops with both arms
    - May indicate clique is artificial (graph artifact)
```

---

## 2. Requirements

### 2.1 Functional Requirements

**FR-1**: Preserve signed correlation values through entire pipeline
**FR-2**: Add correlation direction classification to clique outputs
**FR-3**: Report both signed and absolute correlation statistics
**FR-4**: Flag mixed-sign cliques for review
**FR-5**: Maintain backward compatibility (existing fields keep same semantics)

### 2.2 Non-Functional Requirements

**NFR-1**: No performance regression in clique enumeration
**NFR-2**: Minimal memory overhead for additional fields
**NFR-3**: Clear documentation of new vs legacy field semantics

---

## 3. Design Specification

### 3.1 New Data Structures

```python
# New enum for correlation direction
class CorrelationDirection(Enum):
    POSITIVE = "positive"      # All edges r > 0
    NEGATIVE = "negative"      # All edges r < 0
    MIXED = "mixed"           # Both positive and negative edges
    UNKNOWN = "unknown"       # Not computed


# Extended CorrelationClique (backward compatible)
@dataclass
class CorrelationClique:
    # Existing fields (semantics unchanged for backward compat)
    genes: Set[str]
    condition: str
    mean_correlation: float      # Mean of |r| (existing behavior)
    min_correlation: float       # Min of |r| (existing behavior)
    size: int

    # NEW fields for sign preservation
    direction: CorrelationDirection = CorrelationDirection.UNKNOWN
    signed_mean_correlation: float | None = None   # Mean of r (with sign)
    signed_min_correlation: float | None = None    # Min of r (with sign)
    signed_max_correlation: float | None = None    # Max of r (with sign)
    n_positive_edges: int = 0
    n_negative_edges: int = 0
    edge_correlations: list[float] | None = None   # Optional: all edge r values
```

### 3.2 Classification Logic

```python
def classify_correlation_direction(edge_correlations: list[float]) -> CorrelationDirection:
    """
    Classify clique correlation direction from edge correlations.

    Args:
        edge_correlations: List of signed correlation values for all edges

    Returns:
        CorrelationDirection enum value
    """
    if not edge_correlations:
        return CorrelationDirection.UNKNOWN

    n_positive = sum(1 for r in edge_correlations if r > 0)
    n_negative = sum(1 for r in edge_correlations if r < 0)

    if n_negative == 0:
        return CorrelationDirection.POSITIVE
    elif n_positive == 0:
        return CorrelationDirection.NEGATIVE
    else:
        return CorrelationDirection.MIXED
```

### 3.3 Modified Clique Statistics Computation

```python
# In find_cliques(), replace lines 1192-1202 with:

# Extract SIGNED correlations from graph edges
signed_edge_corrs = [
    G[u][v]['weight']  # No abs() - preserve sign
    for u, v in itertools.combinations(clique_nodes, 2)
]

# Compute absolute values for backward-compatible fields
abs_edge_corrs = [abs(r) for r in signed_edge_corrs]

# Classify direction
n_positive = sum(1 for r in signed_edge_corrs if r > 0)
n_negative = sum(1 for r in signed_edge_corrs if r < 0)

if n_negative == 0:
    direction = CorrelationDirection.POSITIVE
elif n_positive == 0:
    direction = CorrelationDirection.NEGATIVE
else:
    direction = CorrelationDirection.MIXED

# Build clique with both signed and absolute stats
clique = CorrelationClique(
    genes=clique_genes,
    condition=condition,
    # Backward-compatible fields (absolute values)
    mean_correlation=float(np.mean(abs_edge_corrs)),
    min_correlation=float(np.min(abs_edge_corrs)),
    size=len(clique_genes),
    # New signed fields
    direction=direction,
    signed_mean_correlation=float(np.mean(signed_edge_corrs)),
    signed_min_correlation=float(np.min(signed_edge_corrs)),
    signed_max_correlation=float(np.max(signed_edge_corrs)),
    n_positive_edges=n_positive,
    n_negative_edges=n_negative,
)
```

---

## 4. Implementation Work Packages

### WP-1: Core Data Structure Updates
**Scope**: Update dataclasses with new fields
**Files**:
  - `src/cliquefinder/knowledge/clique_validator.py` (CorrelationClique, ChildSetType2)
  - `src/cliquefinder/knowledge/module_discovery.py` (CoexpressionModule)
  - `src/cliquefinder/stats/clique_analysis.py` (CliqueDefinition, CliqueDifferentialResult)
**Complexity**: Low
**Dependencies**: None

### WP-2: Clique Statistics Computation
**Scope**: Modify find_cliques() and find_maximum_clique() to compute signed stats
**Files**:
  - `src/cliquefinder/knowledge/clique_validator.py`
**Complexity**: Medium
**Dependencies**: WP-1

### WP-3: Module Discovery Integration
**Scope**: Propagate signed stats through ModuleDiscovery
**Files**:
  - `src/cliquefinder/knowledge/module_discovery.py`
**Complexity**: Low
**Dependencies**: WP-1, WP-2

### WP-4: Serialization Updates
**Scope**: Update to_dict(), to_dataframe() methods for new fields
**Files**:
  - `src/cliquefinder/knowledge/clique_validator.py`
  - `src/cliquefinder/knowledge/module_discovery.py`
  - `src/cliquefinder/stats/clique_analysis.py`
**Complexity**: Low
**Dependencies**: WP-1

### WP-5: Test Coverage
**Scope**: Add tests for sign preservation and direction classification
**Files**:
  - `tests/test_clique_validator_integration.py` (extend)
  - `tests/test_correlation_sign_preservation.py` (new)
**Complexity**: Medium
**Dependencies**: WP-1, WP-2, WP-3, WP-4

### WP-6: Documentation Updates
**Scope**: Update docstrings and user documentation
**Files**:
  - All modified files (docstrings)
  - `docs/` (if user docs exist)
**Complexity**: Low
**Dependencies**: All other WPs

---

## 5. Delegation Strategy

### Phase 1: Foundation (WP-1)
Single agent to update all dataclasses consistently.
- Add CorrelationDirection enum
- Add new fields to all affected dataclasses
- Ensure default values for backward compatibility

### Phase 2: Core Logic (WP-2)
Single agent with deep focus on clique_validator.py.
- Modify find_cliques()
- Modify find_maximum_clique()
- Ensure consistency between both methods

### Phase 3: Integration (WP-3, WP-4)
Can parallelize:
- Agent A: Module discovery integration
- Agent B: Serialization updates

### Phase 4: Validation (WP-5, WP-6)
- Test agent: Write comprehensive tests
- Doc agent: Update all documentation

---

## 6. Acceptance Criteria

- [ ] All cliques include `direction` field in output
- [ ] `signed_mean_correlation` matches manual computation from edge weights
- [ ] Mixed-sign cliques are correctly identified
- [ ] Existing tests pass (backward compatibility)
- [ ] New tests cover positive, negative, and mixed cliques
- [ ] Performance benchmarks show no regression

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking downstream code | High | Keep existing field semantics, add new fields |
| Performance regression | Medium | Profile before/after, edge_correlations field is optional |
| Inconsistent coherence values | Medium | Document that `coherence` (from summarization) may differ from `signed_mean_correlation` (from graph) |

---

## Appendix A: Test Cases

```python
# Test Case 1: Pure positive clique
# Genes A, B, C with r(A,B)=0.85, r(A,C)=0.80, r(B,C)=0.90
# Expected: direction=POSITIVE, signed_mean=0.85, mean_correlation=0.85

# Test Case 2: Pure negative clique
# Genes A, B, C with r(A,B)=-0.85, r(A,C)=-0.80, r(B,C)=-0.90
# Expected: direction=NEGATIVE, signed_mean=-0.85, mean_correlation=0.85

# Test Case 3: Mixed clique
# Genes A, B, C with r(A,B)=0.85, r(A,C)=-0.80, r(B,C)=0.75
# Expected: direction=MIXED, signed_mean≈0.27, mean_correlation≈0.80

# Test Case 4: Threshold edge case
# Gene pair with r=-0.70 should be included (|r| >= 0.7)
# But should be marked as negative edge
```

---

## 8. Known Limitations and Mitigations

### 8.1 Mixed-Sign Cliques in Differential Analysis

**Limitation**: Tukey's Median Polish (TMP) assumes an additive model where all features (proteins) move in the same direction. For mixed-sign cliques (some positive edges, some negative edges), this assumption is violated:

- If Gene A increases and Gene B decreases in disease (negative correlation), TMP tries to find a common "clique abundance"
- The resulting summarized value is mathematically unstable
- Differential test results may be unreliable or show wrong direction

**Mitigation (WP-7, implemented 2025-01-28)**:

Runtime warnings are issued when mixed-sign cliques are processed:

```python
# In run_clique_differential_analysis() and run_matched_comparison()
if clique.direction == "mixed":
    warnings.warn(
        f"Clique '{clique.clique_id}' has mixed correlation signs "
        f"({clique.n_positive_edges} positive, {clique.n_negative_edges} negative edges). "
        f"Tukey Median Polish summarization assumes co-directional features. "
        f"Results for this clique may be unreliable.",
        UserWarning
    )
```

**Recommended User Action**:
- Filter results by `direction == 'positive'` or `direction == 'negative'` for robust analysis
- Treat mixed-sign clique results as exploratory, not confirmatory
- Consider separate analysis of positive and negative subgraphs for mixed cliques

### 8.2 Future Enhancements (Not Implemented)

The following enhancements could improve handling of mixed-sign cliques but are deferred:

1. **Sign-aware summarization**: Separate TMP for positive and negative subgraphs
2. **Directional permutation tests**: Test for same-direction vs any-direction effects
3. **Visualization encoding**: Edge color/style based on correlation sign
4. **Coherence decomposition**: Report `mean_positive_corr` and `mean_negative_corr` separately

---

*Document generated for engineering review and delegation planning.*
