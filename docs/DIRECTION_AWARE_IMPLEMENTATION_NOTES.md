# Direction-Aware Bootstrap Implementation Notes

**Date**: 2026-02-01
**Status**: Implementation Complete, Pending Validation Run

---

## Summary of Agent Work

Three sonnet experts implemented the direction-aware bootstrap stability assessment:

### Expert 1: Data Engineering (Agent a40c03c)

**Files Modified**: `src/cliquefinder/stats/clique_analysis.py`

**Changes**:
1. Added `is_coherent` property to `CliqueDefinition` - returns `True` for POSITIVE/NEGATIVE cliques
2. Added `is_mixed` property to `CliqueDefinition` - returns `True` for MIXED cliques
3. Updated `load_clique_definitions()` with:
   - Backward-compatible direction column parsing
   - Warning when direction columns missing (defaults to "unknown")
   - Case-insensitive direction normalization
4. Updated two usage sites to use `clique.is_mixed` property

**Key Implementation**:
```python
@property
def is_coherent(self) -> bool:
    """True if POSITIVE or NEGATIVE (not MIXED or unknown)."""
    return self.direction in ("positive", "negative")

@property
def is_mixed(self) -> bool:
    """True if clique has mixed correlation signs."""
    return self.direction == "mixed"
```

### Expert 2: Statistical Methods (Agent a499927)

**Files Modified**: `src/cliquefinder/stats/bootstrap_comparison.py`

**Changes**:
1. Enhanced `BootstrapCliqueResult` with:
   - `direction`, `n_positive_edges`, `n_negative_edges` fields
   - `is_stable_ols`, `is_stable_roast` per-method stability flags
   - `is_robust` direction-aware combined stability
   - `stability_criterion` ("both_methods" or "roast_only")
   - `pvalue_ci_low_roast`, `pvalue_ci_high_roast`
   - Nullable `method_concordance` for mixed cliques
   - `is_stable` property alias for backward compatibility

2. Updated aggregation logic with direction-aware stability:
```python
is_coherent = direction in ("positive", "negative")
if is_coherent:
    is_robust = is_stable_ols and is_stable_roast
    stability_criterion = "both_methods"
else:
    is_robust = is_stable_roast
    stability_criterion = "roast_only"
    method_concordance = None
```

### Expert 3: Pipeline Engineering (Agent ae64a3d)

**Files Modified**:
- `src/cliquefinder/stats/bootstrap_comparison.py` (output section)
- `src/cliquefinder/cli/compare.py`

**Changes**:
1. Enhanced verbose logging with direction-aware summary:
```
DIRECTION-AWARE BOOTSTRAP STABILITY SUMMARY
Coherent cliques (POSITIVE/NEGATIVE): X tested
  Stable (both OLS & ROAST ≥80%): Y
Mixed cliques: X tested
  Stable (ROAST ≥80%): Y
Total robust cliques: Z
```

2. Split output files:
   - `bootstrap_stable_coherent.csv` - POSITIVE/NEGATIVE hits (both methods)
   - `bootstrap_stable_mixed.csv` - MIXED hits (ROAST only)
   - `bootstrap_stable_hits.csv` - Combined (legacy)

3. Added CLI flag:
```
--mixed-criterion {roast_only,exclude}
```

4. Enhanced `bootstrap_parameters.json` with direction stats

---

## Verification Status

### Code Verified ✓
- `bootstrap_comparison.py`: Full direction-aware logic implemented
- `clique_analysis.py`: `is_coherent`, `is_mixed` properties confirmed
- `compare.py`: `--mixed-criterion` flag confirmed

### Pending Actions

1. **Re-run analyze command** to generate cliques.csv with direction columns
   - Current cliques.csv was generated before the direction output code
   - Bootstrap will work but all cliques treated as "unknown"

2. **Write unit tests** for direction-aware stability logic

3. **Run bootstrap validation** on both strata

---

## Usage

### To re-generate cliques with direction info:
```bash
cliquefinder analyze \
    --data output/proteomics/sporadic.data.csv \
    --metadata output/proteomics/sporadic.metadata.csv \
    --output output/proteomics/discovery_exact_unstratified_v2 \
    --stratify-by phenotype \
    ...
```

### To run bootstrap with direction awareness:
```bash
cliquefinder compare \
    --data output/proteomics/sporadic.data.csv \
    --metadata output/proteomics/sporadic.metadata.csv \
    --cliques output/proteomics/discovery_exact_unstratified_v2/cliques.csv \
    --output output/proteomics/method_comparison_bootstrap_v2 \
    --condition-col phenotype \
    --contrast CASE_vs_CTRL CASE CTRL \
    --stratify-by Sex \
    --bootstrap \
    --bootstrap-n 100 \
    --mixed-criterion roast_only
```

---

## Statistical Logic

| Direction | OLS Valid? | ROAST Valid? | `is_robust` Criterion |
|-----------|------------|--------------|----------------------|
| positive  | Yes        | Yes          | `is_stable_ols AND is_stable_roast` |
| negative  | Yes        | Yes          | `is_stable_ols AND is_stable_roast` |
| mixed     | **NO**     | Yes          | `is_stable_roast` only |
| unknown   | **NO** (conservative) | Yes | `is_stable_roast` only |

**Rationale**: OLS+TMP aggregation assumes directional coherence. For mixed cliques, positive and negative effects cancel during aggregation, making OLS results invalid. ROAST_MSQ (mean squared t) is direction-agnostic and remains valid.

---

## Expected Output

With direction info available, expect output like:
```
DIRECTION-AWARE BOOTSTRAP STABILITY SUMMARY
═══════════════════════════════════════════
Coherent cliques (POSITIVE/NEGATIVE): 1200 tested
  Stable (both OLS & ROAST ≥80%): 8

Mixed cliques: 472 tested
  Stable (ROAST ≥80%): 12

Total robust cliques: 20
```

If direction info missing (current state):
```
DIRECTION-AWARE BOOTSTRAP STABILITY SUMMARY
═══════════════════════════════════════════
Unknown direction: 1672 (treated as mixed)

Total robust cliques: 15  # ROAST-only criterion applied to all
```
