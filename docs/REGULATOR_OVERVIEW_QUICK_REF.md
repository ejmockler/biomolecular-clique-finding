# Regulator Overview Plot - Quick Reference

## One-Liner

```python
viz.plot_regulator_overview(rewiring_df, stratified_df).save("overview.pdf")
```

## Visual Encoding Cheat Sheet

| Element | Meaning | Values |
|---------|---------|--------|
| **X-axis** | Rewiring score | ← Lost in ALS \| Gained in ALS → |
| **Y-axis** | Regulatory quality | 0.0 (low) to 1.0 (high coherence) |
| **Point size** | Clique size | Small (few genes) to Large (many genes) |
| **Point color** | Sex pattern | Purple=Both, Teal=Male, Violet=Female, Gray=Weak |

## Quadrants

```
        │ HIGH-QUALITY GAINS
  HIGH  │ (target for investigation)
QUALITY ├─────────────────────────
        │ LOW-QUALITY GAINS

  ← LOST IN ALS │ GAINED IN ALS →
```

## Common Patterns

```python
# Default: scan all regulators
viz.plot_regulator_overview(rewiring_df, stratified_df)

# Strong signals only
viz.plot_regulator_overview(rewiring_df, stratified_df, min_rewiring_score=0.3)

# For presentations
viz = CliqueVisualizer(style="presentation")
viz.plot_regulator_overview(rewiring_df, stratified_df, figsize=(16, 12))
```

## Metadata

```python
fig.metadata['n_regulators']      # Total shown
fig.metadata['n_both_sexes']      # Purple points
fig.metadata['n_male_specific']   # Teal points
fig.metadata['n_female_specific'] # Violet points
```

## Color Palette

| Pattern | Color | Hex |
|---------|-------|-----|
| Both sexes | Purple | `#7c3aed` |
| Male-specific | Teal | `#0d9488` |
| Female-specific | Violet | `#9333ea` |
| Weak rewiring | Gray | `#9ca3af` |

## Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_rewiring_score` | 0.0 | Filter threshold for \|rewiring\| |
| `label_top_n` | 20 | Number of regulators to label |
| `point_alpha` | 0.7 | Point transparency (0-1) |
| `figsize` | (14, 10) | Figure dimensions (inches) |

## Workflow

1. **Scan**: Use overview to identify interesting regulators
2. **Filter**: Apply `min_rewiring_score` to focus on strong signals
3. **Investigate**: Use regulator names to drill down with other plots
4. **Report**: Save as PDF for publication

## Pro Tips

- Install `adjusttext` for better label placement: `pip install adjusttext`
- Start with default view, then filter progressively
- Top-right quadrant = highest priority regulators
- Look for clusters of same-color points (sex-specific effects)
- Large purple points in corners = strong, generalizable findings
