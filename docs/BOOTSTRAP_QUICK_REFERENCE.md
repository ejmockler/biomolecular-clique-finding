# Bootstrap Sampling Quick Reference

## TL;DR

The bootstrap implementation now supports **two modes**:

| Mode | Cases | Controls | Use When |
|------|-------|----------|----------|
| **True Bootstrap** (default) | WITH replacement | WITH replacement | You need proper uncertainty quantification |
| **Subsampling** | WITHOUT replacement | All (identical) | You want cleaner interpretation, controls are numerous |

## Quick Start

### Recommended: True Bootstrap (Default)

```python
from cliquefinder.stats.bootstrap_comparison import BootstrapConfig, run_bootstrap_comparison

# Use default configuration (bootstrap_controls=True)
config = BootstrapConfig(n_bootstraps=100, target_ratio=2.0)

results = run_bootstrap_comparison(
    data=data,
    feature_ids=feature_ids,
    metadata=metadata,
    cliques=cliques,
    condition_column='condition',
    contrast=('disease', 'healthy'),
    config=config
)
```

### Alternative: Subsampling Mode

```python
# Explicitly disable control bootstrapping
config = BootstrapConfig(
    n_bootstraps=100,
    target_ratio=2.0,
    bootstrap_controls=False  # Use original subsampling approach
)
```

## What Changed?

### Before (Subsampling)

```python
# Cases: Sample 56 from 173 WITHOUT replacement
selected_cases = rng.choice(case_samples, size=target_n_cases, replace=False)

# Controls: Use ALL 28 controls (identical every iteration)
bootstrap_samples = list(selected_cases) + ctrl_samples
```

**Problem**: Control population uncertainty is NOT captured

### After (True Bootstrap)

```python
# Cases: Sample WITH replacement
selected_cases = rng.choice(case_samples, size=target_n_cases, replace=True)

# Controls: Sample WITH replacement
selected_ctrls = rng.choice(ctrl_samples, size=n_ctrls, replace=True)

bootstrap_samples = list(selected_cases) + list(selected_ctrls)
```

**Benefit**: Captures uncertainty in BOTH populations

## Visual Comparison

### True Bootstrap (bootstrap_controls=True)

```
Bootstrap 1: [case_0, case_7, case_7, case_4, ...] + [ctrl_0, ctrl_3, ctrl_0, ctrl_2, ...]
                     ↑          ↑                            ↑          ↑
                     Same sample appears twice               Same control appears twice

Bootstrap 2: [case_9, case_7, case_7, case_5, ...] + [ctrl_0, ctrl_4, ctrl_2, ctrl_1, ...]
                     Different samples, some repeated        Different controls each time

→ Captures variability in BOTH groups
```

### Subsampling (bootstrap_controls=False)

```
Bootstrap 1: [case_6, case_4, case_8, case_9, ...] + [ctrl_0, ctrl_1, ctrl_2, ctrl_3, ctrl_4]
                     No repeats                            All controls

Bootstrap 2: [case_1, case_7, case_3, case_5, ...] + [ctrl_0, ctrl_1, ctrl_2, ctrl_3, ctrl_4]
                     Different cases                       SAME controls

→ Only captures case variability
```

## Statistical Implications

### True Bootstrap (bootstrap_controls=True)

**Advantages:**
- Statistically correct uncertainty quantification
- Confidence intervals reflect sampling variability in both groups
- More conservative (wider CIs)
- Follows Efron & Tibshirani methodology

**Disadvantages:**
- Slightly more computational cost
- Wider confidence intervals (more conservative)

**Recommended when:**
- n_controls < 50
- Controls are heterogeneous
- Regulatory submission or publication
- Conservative inference desired

### Subsampling (bootstrap_controls=False)

**Advantages:**
- Cleaner interpretation (controls are "fixed")
- Narrower confidence intervals
- Faster computation
- Useful for exploratory analysis

**Disadvantages:**
- Underestimates uncertainty
- Assumes controls have zero sampling error
- May lead to anti-conservative inference

**Recommended when:**
- n_controls > 100 and homogeneous
- Exploratory analysis
- Controls are well-characterized reference population

## Configuration Parameters

```python
@dataclass
class BootstrapConfig:
    n_bootstraps: int = 100              # Number of bootstrap iterations
    target_ratio: float = 2.0            # Desired case:control ratio
    bootstrap_controls: bool = True      # NEW: Enable true bootstrap
    significance_threshold: float = 0.05
    stability_threshold: float = 0.80    # % bootstraps for "stable"
    concordance_threshold: float = 0.50  # Method agreement threshold
    seed: int | None = 42
    n_rotations: int = 499               # ROAST rotations
    use_gpu: bool = True
    verbose: bool = True
```

## Example Output Interpretation

### Selection Frequency

```python
# From results DataFrame
selection_freq_both = 0.85  # Significant in 85% of bootstraps

# Interpretation:
# - bootstrap_controls=True:  Robust finding, accounts for all uncertainty
# - bootstrap_controls=False: May overestimate stability (controls fixed)
```

### Confidence Intervals

```python
# Effect size CI from results
effect_ci_low = 0.45
effect_ci_high = 1.23

# Interpretation:
# - bootstrap_controls=True:  Proper 95% CI with both sources of uncertainty
# - bootstrap_controls=False: Narrower CI (conditional on observed controls)
```

## Decision Tree

```
Do you need statistically rigorous uncertainty quantification?
├─ YES → Use bootstrap_controls=True (default)
│  └─ Is n_controls < 50?
│     ├─ YES → MUST use bootstrap_controls=True
│     └─ NO  → Still recommended, but bootstrap_controls=False is defensible
│
└─ NO (exploratory analysis)
   └─ Use bootstrap_controls=False for cleaner interpretation
```

## Common Scenarios

### Scenario 1: ALS Study (173 cases, 28 controls)

```python
# n_controls is small → MUST use true bootstrap
config = BootstrapConfig(
    n_bootstraps=100,
    target_ratio=2.0,
    bootstrap_controls=True  # Essential for valid inference
)
```

### Scenario 2: Large Cohort (5000 cases, 500 controls)

```python
# Controls are numerous → Either mode is reasonable
config_conservative = BootstrapConfig(bootstrap_controls=True)   # More conservative
config_efficient = BootstrapConfig(bootstrap_controls=False)     # Faster, cleaner
```

### Scenario 3: Exploratory Analysis

```python
# Quick exploration → Subsampling is fine
config = BootstrapConfig(
    n_bootstraps=50,           # Fewer iterations
    bootstrap_controls=False,  # Faster
    verbose=True
)
```

## Verification

Test that your installation has the correct implementation:

```python
from cliquefinder.stats.bootstrap_comparison import BootstrapConfig

config = BootstrapConfig()
assert hasattr(config, 'bootstrap_controls'), "Parameter not found!"
assert config.bootstrap_controls == True, "Default should be True!"
print("✓ Implementation verified!")
```

## References

1. Efron B, Tibshirani RJ (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
   - Chapter 6: Bootstrap confidence intervals
   - Chapter 11: Balanced resampling

2. Davison AC, Hinkley DV (1997). *Bootstrap Methods and their Application*. Cambridge.
   - Section 3.3: Variance estimation
   - Section 5.2: Balanced sampling

3. Good PI (2005). *Resampling Methods*. Birkhäuser.
   - Chapter 3: Bootstrap vs. permutation

## Support

For questions or issues:
1. Check `/docs/BOOTSTRAP_IMPLEMENTATION.md` for detailed technical documentation
2. Run `/tests/test_bootstrap_modes.py` to see demonstration
3. Review source: `/src/cliquefinder/stats/bootstrap_comparison.py`
