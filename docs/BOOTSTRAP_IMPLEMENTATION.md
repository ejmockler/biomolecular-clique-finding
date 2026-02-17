# Bootstrap Implementation Update

## Summary

Updated `/Users/noot/Documents/biomolecular-clique-finding/src/cliquefinder/stats/bootstrap_comparison.py` to implement proper balanced bootstrap sampling instead of repeated subsampling.

## Changes Made

### 1. Added `bootstrap_controls` Parameter to `BootstrapConfig`

```python
@dataclass
class BootstrapConfig:
    ...
    bootstrap_controls: bool = True  # Sample controls WITH replacement (true bootstrap)
```

**Default: `True`** - Enables proper bootstrap sampling by default.

### 2. Updated Module Documentation

Changed from "Bootstrap subsampling" to "Bootstrap resampling" and clarified the two modes:

- **True bootstrap** (`replace=True`): Captures uncertainty in both populations
- **Subsampling mode** (`replace=False` for cases, all controls): Original behavior for cleaner interpretation

### 3. Modified Sampling Logic (Lines 192-208)

**NEW IMPLEMENTATION:**

```python
if config.bootstrap_controls:
    # TRUE BALANCED BOOTSTRAP:
    # Sample WITH replacement from both groups to capture uncertainty
    # in both populations. This allows:
    # - Same sample to appear multiple times (bootstrap property)
    # - Variance estimation for both case and control populations
    # - Proper confidence interval construction
    selected_cases = rng.choice(case_samples, size=target_n_cases, replace=True)
    selected_ctrls = rng.choice(ctrl_samples, size=n_ctrls, replace=True)
    bootstrap_samples = list(selected_cases) + list(selected_ctrls)
else:
    # SUBSAMPLING MODE:
    # Sample cases WITHOUT replacement, use all controls
    # This is the original behavior - cleaner interpretation but
    # doesn't capture control population uncertainty
    selected_cases = rng.choice(case_samples, size=target_n_cases, replace=False)
    bootstrap_samples = list(selected_cases) + ctrl_samples
```

**OLD IMPLEMENTATION:**

```python
# Subsample cases (with replacement for bootstrap, without for subsampling)
# Using without replacement for cleaner interpretation
selected_cases = rng.choice(case_samples, size=target_n_cases, replace=False)

# Combine with all controls
bootstrap_samples = list(selected_cases) + ctrl_samples
```

### 4. Enhanced Verbose Output

Added sampling strategy information to the output:

```python
print(f"Sampling strategy: {'True bootstrap (WITH replacement)' if config.bootstrap_controls else 'Subsampling (cases only, all controls)'}")
```

## Statistical Rationale

### Problem with Previous Implementation

1. **Cases**: Sampled WITHOUT replacement → different 56 from 173 each time
2. **Controls**: ALL 28 controls used every iteration → IDENTICAL across bootstraps

This fails to capture uncertainty in the control population, leading to:
- Underestimated variance
- Overly narrow confidence intervals
- Invalid statistical inference

### True Bootstrap Properties

With `bootstrap_controls=True`:

1. **Sample WITH replacement** from both groups
2. Same sample can appear multiple times (key bootstrap property)
3. Captures sampling variability in BOTH populations
4. Enables proper confidence interval construction
5. More conservative (wider CIs) but statistically correct

### When to Use Each Mode

**Use `bootstrap_controls=True` (default) when:**
- Control sample size is small (n < 50)
- Control population is heterogeneous
- You need conservative, statistically valid confidence intervals
- Proper uncertainty quantification is critical

**Use `bootstrap_controls=False` when:**
- Controls are numerous and well-characterized
- You want cleaner interpretation (same controls, varying cases)
- Narrower confidence intervals are acceptable
- Control uncertainty is not of primary interest

## Verification

### Syntax Check

```bash
python3 -m py_compile src/cliquefinder/stats/bootstrap_comparison.py
# ✓ No errors
```

### Import Test

```python
from cliquefinder.stats.bootstrap_comparison import BootstrapConfig, run_bootstrap_comparison
config = BootstrapConfig()
print(config.bootstrap_controls)  # True
```

### Demonstration Script

Created `/Users/noot/Documents/biomolecular-clique-finding/tests/test_bootstrap_modes.py` to demonstrate the difference between modes:

```bash
.venv/bin/python tests/test_bootstrap_modes.py
```

Output shows:
- **True Bootstrap**: Samples appear multiple times, both groups vary
- **Subsampling**: No reuse in cases, identical controls every iteration

## Backward Compatibility

The implementation maintains backward compatibility:

1. **Default behavior changes**: The new default (`bootstrap_controls=True`) is the statistically correct approach
2. **Old behavior available**: Set `bootstrap_controls=False` to use the original subsampling mode
3. **API unchanged**: All other parameters and return values remain the same

## Example Usage

```python
from cliquefinder.stats.bootstrap_comparison import BootstrapConfig, run_bootstrap_comparison

# True bootstrap (default, recommended)
config_bootstrap = BootstrapConfig(
    n_bootstraps=100,
    target_ratio=2.0,
    bootstrap_controls=True  # Default
)

# Subsampling mode (original behavior)
config_subsample = BootstrapConfig(
    n_bootstraps=100,
    target_ratio=2.0,
    bootstrap_controls=False
)

# Run analysis
results = run_bootstrap_comparison(
    data=data,
    feature_ids=feature_ids,
    metadata=metadata,
    cliques=cliques,
    condition_column='condition',
    contrast=('case', 'control'),
    config=config_bootstrap
)
```

## References

- Efron B, Tibshirani RJ (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Davison AC, Hinkley DV (1997). Bootstrap Methods and their Application. Cambridge University Press.
- Good PI (2005). Resampling Methods: A Practical Guide to Data Analysis. Birkhäuser.

## Implementation Date

2026-01-29

## Files Modified

1. `/Users/noot/Documents/biomolecular-clique-finding/src/cliquefinder/stats/bootstrap_comparison.py`
   - Added `bootstrap_controls` parameter to `BootstrapConfig`
   - Updated sampling logic to support both modes
   - Enhanced documentation and comments

## Files Created

1. `/Users/noot/Documents/biomolecular-clique-finding/tests/test_bootstrap_modes.py`
   - Demonstration script showing difference between modes
   - Educational output explaining statistical implications

2. `/Users/noot/Documents/biomolecular-clique-finding/docs/BOOTSTRAP_IMPLEMENTATION.md`
   - This documentation file
