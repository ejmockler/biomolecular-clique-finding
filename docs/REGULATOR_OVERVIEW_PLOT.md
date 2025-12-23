# Regulator Overview Plot

## Purpose

The **Regulator Overview Plot** is the **entry point visualization** for the biomolecular clique-finding analysis. It enables users to scan all 538 regulators at once and identify those with strong rewiring patterns in ALS disease.

This is a **perceptual scanning tool** - designed to support rapid visual triage of regulators based on:
- Rewiring strength (gained vs lost regulatory relationships)
- Regulatory quality (coherence)
- Biological impact (clique size)
- Sex-specific patterns

## Visual Encoding

### Axes
- **X-axis**: `rewiring_score` (signed)
  - Positive values: Regulatory cliques **gained** in ALS (CASE)
  - Negative values: Regulatory cliques **lost** in ALS
  - Magnitude: Strength of rewiring

- **Y-axis**: `max_coherence` across all conditions
  - Higher = better quality regulatory relationships
  - Measures how well-correlated genes are within cliques
  - Range: 0.0 to 1.0

### Point Attributes

#### Size
Point size encodes **total clique size** (number of unique genes regulated):
- Larger points = more genes affected
- Uses `sqrt` scaling for better visual discrimination
- Size legend in bottom-left

#### Color
Point color encodes **sex interaction pattern**:
- **Purple** (#7c3aed): Both sexes affected
- **Teal** (#0d9488): Male-specific rewiring
- **Violet** (#9333ea): Female-specific rewiring
- **Gray** (#9ca3af): Weak rewiring (below threshold)

Sex pattern is determined by comparing rewiring scores:
- Threshold: |rewiring_score| ≥ 0.25
- "Both sexes": Both male and female comparisons exceed threshold
- "Male-specific": Only male comparisons exceed threshold
- "Female-specific": Only female comparisons exceed threshold

### Quadrants

The plot is divided into four conceptual quadrants:

```
           High Coherence
                 |
   High-quality  |  High-quality
   losses        |  gains
   (lost in ALS) |  (gained in ALS)
─────────────────┼─────────────────
                 |
   Low-quality   |  Low-quality
   losses        |  gains
                 |
           Low Coherence
```

**Top-right quadrant** (most interesting): High-quality regulatory relationships gained in disease
**Top-left quadrant**: High-quality regulatory relationships lost in disease
**Bottom quadrants**: Lower quality changes

## Data Sources

### Input DataFrames

#### 1. `regulator_rewiring_stats.csv`
Per-comparison statistics:
- `regulator`: Gene symbol
- `comparison`: e.g., "CASE_M_vs_CTRL_M", "CASE_F_vs_CTRL_F"
- `rewiring_score`: Signed measure of rewiring
- `gained_cliques`: Number of cliques gained
- `lost_cliques`: Number of cliques lost
- `case_coherence`: Coherence in CASE condition
- `ctrl_coherence`: Coherence in CTRL condition

#### 2. `stratified_cliques.csv`
Per-condition clique composition:
- `regulator`: Gene symbol
- `condition`: CASE_F, CASE_M, CTRL_F, CTRL_M
- `coherence_ratio`: Quality of regulatory relationships
- `clique_genes`: Comma-separated list of regulated genes
- `n_samples`: Sample size

### Aggregation Logic

For each regulator, the plot computes:

1. **Overall rewiring score**: Max absolute rewiring across male/female comparisons
   - Preserves sign (positive = gained, negative = lost)
   - Takes the sex with stronger signal

2. **Max coherence**: Highest coherence_ratio across all 4 conditions
   - Represents best-quality regulatory state

3. **Total genes**: Union of all clique_genes across all conditions
   - Measures total biological impact

4. **Sex pattern**: Categorizes based on which sexes show significant rewiring
   - Enables identification of sex-specific effects

## Usage

### Basic Usage

```python
import pandas as pd
from cliquefinder.viz.cliques import CliqueVisualizer

# Load data
rewiring_df = pd.read_csv("results/cliques/regulator_rewiring_stats.csv")
stratified_df = pd.read_csv("results/cliques/stratified_cliques.csv")

# Create visualizer
viz = CliqueVisualizer(style="paper")

# Generate overview
fig = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df
)

# Save
fig.save("figures/regulator_overview.pdf")
```

### Parameters

- **`min_rewiring_score`** (float, default=0.0): Filter weak signals
  - Set to 0.3 for strong rewiring only
  - Set to 0.0 to see all regulators

- **`label_top_n`** (int, default=20): Number of regulators to label
  - Uses `adjustText` package if available for smart label placement
  - Falls back to simple labels otherwise

- **`point_alpha`** (float, default=0.7): Point transparency
  - Lower values for dense plots
  - Higher values for sparse plots

- **`figsize`** (tuple, default=(14, 10)): Figure dimensions in inches

### Filtering Examples

```python
# Show only strong rewiring
fig = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.3,  # |score| >= 0.3
    label_top_n=15
)

# Focus on top signals
fig = viz.plot_regulator_overview(
    rewiring_df=rewiring_df,
    stratified_df=stratified_df,
    min_rewiring_score=0.35,
    label_top_n=10
)
```

### Style Variants

```python
# Paper style (default): high DPI, publication-quality
viz_paper = CliqueVisualizer(style="paper")

# Presentation style: larger fonts, higher contrast
viz_pres = CliqueVisualizer(style="presentation")

# Notebook style: interactive-friendly
viz_nb = CliqueVisualizer(style="notebook")
```

## Interpretation Guide

### Visual Workflow

1. **Scan quadrants**: Identify overall distribution
   - Where are most points? (usually centered around 0)
   - Are there outliers in the top quadrants?

2. **Check colors**: Identify sex patterns
   - Purple points = generalizable findings (both sexes)
   - Teal/violet = sex-specific biology

3. **Look at size**: Assess biological impact
   - Large points affect many genes
   - Small points are focused effects

4. **Read labels**: Top regulators are automatically labeled
   - Start investigation with labeled regulators
   - Use labels as entry points for deeper analysis

### Finding Interesting Regulators

**High-priority regulators** have:
- High |rewiring_score| (far from x=0)
- High coherence (high on y-axis)
- Large size (many genes affected)
- Color = purple (both sexes) or teal/violet (sex-specific)

**Example questions**:
- Which regulators show the strongest gains in ALS?
  → Look at far-right, high points

- Which high-quality relationships are lost in disease?
  → Look at far-left, high points

- Are there sex-specific rewiring patterns?
  → Compare teal vs violet point distributions

- Which regulators affect the most genes?
  → Look at largest points

## Perceptual Design Rationale

### Why scatter plot?
- Enables simultaneous encoding of 4 dimensions (x, y, size, color)
- Supports rapid visual scanning (eye can process ~100 points/second)
- Quadrant structure provides natural grouping

### Why signed rewiring score?
- Preserves biological direction (gained vs lost)
- Enables symmetry detection
- Natural mapping to spatial position (left/right)

### Why max coherence?
- Single quality metric across conditions
- Enables y-axis sorting by quality
- Highlights high-confidence findings

### Why sqrt size scaling?
- Better discrimination than linear scaling
- Prevents large points from dominating
- Standard practice for area encoding

### Why sex-pattern colors?
- Biological significance (personalized medicine)
- Categorical encoding (discrete colors)
- Colorblind-safe palette

## Output Metadata

The Figure object includes metadata:

```python
fig.metadata
{
    'n_regulators': 538,              # Total regulators shown
    'min_rewiring_score': 0.0,        # Filter threshold
    'n_both_sexes': 15,               # Purple points
    'n_male_specific': 41,            # Teal points
    'n_female_specific': 24,          # Violet points
    'n_weak': 458,                    # Gray points
    'created_at': '2025-12-17T...'    # Timestamp
}
```

## Integration with Other Visualizations

The regulator overview is the **entry point**. After identifying interesting regulators, use:

1. **Stratum Heatmap** (`plot_stratum_heatmap`):
   - Shows gene membership patterns across 4 strata
   - Reveals which specific genes are gained/lost/stable
   - Use for: Understanding composition of regulatory changes

2. **Gene Flow Diagram** (`plot_gene_flow`):
   - Shows flow of genes from CTRL → CASE
   - Visualizes lost, gained, and stable genes
   - Use for: Understanding directionality of change

3. **Network Graph** (`plot_clique_network`):
   - Shows correlation structure within cliques
   - Reveals why genes form coherent groups
   - Use for: Understanding regulatory mechanisms

### Recommended Workflow

```python
# 1. Start with overview
fig_overview = viz.plot_regulator_overview(rewiring_df, stratified_df)

# 2. Identify interesting regulators (e.g., "MAPT", "SOD1")
interesting = ["MAPT", "SOD1", "FUS"]

# 3. Drill down into each regulator
for reg in interesting:
    # Show membership patterns
    fig_heatmap = viz.plot_stratum_heatmap(reg, stratified_df)

    # Show gene flow
    fig_flow = viz.plot_gene_flow(reg, stratified_df)

    # Save for report
    fig_heatmap.save(f"figures/{reg}_heatmap.pdf")
    fig_flow.save(f"figures/{reg}_flow.pdf")
```

## Dependencies

Required packages:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `seaborn`: Styling (via configure_style)

Optional packages:
- `adjustText`: Smart label placement (highly recommended)
  - Install: `pip install adjusttext`
  - Without it, labels may overlap

## Examples

See `examples/plot_regulator_overview_examples.py` for:
- Default overview (all regulators)
- Filtered overview (strong rewiring only)
- Different styles (paper/presentation/notebook)
- Programmatic analysis of top regulators

Generated example plots are in `figures/regulator_overview_examples/`.

## Performance

- Processes 538 regulators in ~1-2 seconds
- Generates publication-quality PDF in ~3-4 seconds
- Memory usage: ~50MB for full dataset

## Limitations

1. **Aggregation**: Each regulator shown as single point
   - Per-sex details are aggregated
   - Use heatmap for detailed sex-specific patterns

2. **Label overlap**: Without `adjustText`, labels may overlap
   - Reduce `label_top_n` if overlapping
   - Or install `adjusttext` package

3. **Static view**: Not interactive
   - Consider Plotly version for interactivity
   - Use for print/PDF output

## Future Enhancements

Potential improvements:
- Interactive version with hover tooltips
- Brushing/linking to other views
- Color by pathway/function
- Filter by gene ontology terms
- Export selected regulators to table
