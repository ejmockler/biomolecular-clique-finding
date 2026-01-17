# Clique Network Visualization

## Overview

The `plot_clique_network()` method creates network graphs that reveal the **correlation structure** within regulatory cliques. This complements the stratum heatmap by showing WHY genes cluster together.

## Location

```python
from cliquefinder.viz.cliques import CliqueVisualizer

viz = CliqueVisualizer(style="paper")
fig = viz.plot_clique_network(regulator="MAPT", df=cliques_df)
```

**File**: `/Users/noot/Documents/biomolecular-clique-finding/src/cliquefinder/viz/cliques.py`
**Line**: 410

## Visual Encoding (Perceptual Engineering)

The network uses multiple visual channels to encode information:

### Node Properties
- **Node Color**: Condition membership pattern
  - **Emerald (#059669)**: Gene present in ALL conditions (stable across disease states)
  - **Blue (#2563eb)**: Gene present only in CASE conditions (disease-specific)
  - **Orange (#f97316)**: Gene present only in CTRL conditions (healthy-specific)
  - **Gray (#6b7280)**: Gene present in SOME but not all conditions (partial membership)

- **Node Size**: Degree centrality (proportional to number of connections)
  - Large nodes = hub genes with many correlations
  - Small nodes = peripheral genes with fewer connections

### Edge Properties
- **Edge Thickness**: Correlation strength (absolute value)
  - Thicker edges = stronger correlations
  - Thinner edges = weaker correlations (but still above threshold)

- **Edge Color/Alpha**: Gray (#9ca3af) with opacity proportional to correlation strength
  - More opaque = stronger correlation
  - More transparent = weaker correlation

### Layout
- **Force-directed (spring) layout**: Physically simulates nodes as charged particles
  - Highly correlated genes are pulled together
  - Weakly correlated genes are pushed apart
  - Results in natural clustering of co-expressed genes

## Parameters

```python
def plot_clique_network(
    self,
    regulator: str,                        # Required
    df: pd.DataFrame,                      # Required
    correlation_data: Optional[...] = None,  # Optional
    condition_focus: Optional[str] = None,   # Optional
    correlation_threshold: float = 0.4,      # Default: 0.4
    figsize: tuple[float, float] = (10, 10), # Default: (10, 10)
    node_size_scale: float = 300,            # Default: 300
    edge_width_scale: float = 3.0,           # Default: 3.0
    layout_iterations: int = 50,             # Default: 50
    layout_k: Optional[float] = None         # Default: auto
) -> Figure:
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `regulator` | str | Regulator gene symbol (e.g., "MAPT", "SLC2A1") |
| `df` | DataFrame | Stratified cliques data with columns: `regulator`, `condition`, `clique_genes` |
| `correlation_data` | DataFrame/BioMatrix/None | Expression data for computing correlations (see below) |
| `condition_focus` | str | If provided, show only genes from this condition (e.g., "CASE_F") |
| `correlation_threshold` | float | Minimum absolute correlation to draw an edge (0-1) |

### Correlation Data

The `correlation_data` parameter accepts:

1. **`None` (recommended for membership visualization)**:
   - Shows all genes in the clique with uniform edge weights
   - No correlation computation needed
   - Fast and always works
   - Good for understanding membership patterns

2. **`pd.DataFrame`** (for correlation-weighted edges):
   - Rows = genes (MUST be gene symbols matching clique_genes)
   - Columns = samples
   - Values = expression levels
   - Computes Pearson correlation between genes
   - **Important**: Row indices must match gene symbols in cliques

3. **`BioMatrix`**:
   - Uses `data` array and `feature_ids` index
   - Feature IDs must be gene symbols

**Note on Current Data**: The proteomics data (`proteomics_imputed.data.csv`) uses UniProt IDs as row indices, not gene symbols. Therefore, correlation computation will fall back to membership-only mode. To use correlation-weighted edges, you would need expression data with gene symbols as row names.

## Usage Examples

### Example 1: Basic Membership Network (No Correlations)

This is the recommended approach for understanding clique membership patterns.

```python
import pandas as pd
from cliquefinder.viz.cliques import CliqueVisualizer

# Load clique data
df = pd.read_csv("results/cliques/cliques.csv")

# Create visualizer
viz = CliqueVisualizer(style="paper")

# Create network (membership only, no correlations)
fig = viz.plot_clique_network(
    regulator="MAPT",
    df=df,
    correlation_data=None  # This creates membership-only network
)

# Save figure
fig.save("figures/mapt_network.pdf")
fig.save("figures/mapt_network.png", dpi=300)

# Check metadata
print(fig.metadata)
# {'regulator': 'MAPT',
#  'n_genes': 25,
#  'n_edges': 300,
#  'gene_categories': {'stable': 5, 'ctrl_only': 7, 'case_only': 3, 'partial': 10}}
```

### Example 2: Condition-Specific Network

View only genes from a specific condition (e.g., CASE females).

```python
fig = viz.plot_clique_network(
    regulator="MAPT",
    df=df,
    condition_focus="CASE_F",  # Only show CASE_F genes
    correlation_data=None
)
fig.save("figures/mapt_case_f_network.pdf")
```

### Example 3: Multiple Regulators

Create networks for several regulators to compare their clique structures.

```python
regulators = ["MAPT", "SLC2A1", "CDK1", "HMGB1"]

for reg in regulators:
    fig = viz.plot_clique_network(
        regulator=reg,
        df=df,
        correlation_data=None,
        figsize=(12, 12)  # Larger figure for more genes
    )
    fig.save(f"figures/{reg.lower()}_network.pdf")
```

### Example 4: Adjust Visual Encoding

Customize node sizes and edge widths for publication.

```python
fig = viz.plot_clique_network(
    regulator="MAPT",
    df=df,
    node_size_scale=500,      # Larger nodes
    edge_width_scale=5.0,     # Thicker edges
    layout_iterations=100,    # More layout refinement
    figsize=(14, 14)          # Larger canvas
)
fig.save("figures/mapt_network_large.pdf")
```

### Example 5: With Correlation Data (If Available)

If you have expression data with gene symbols as row indices:

```python
# Load expression data with gene symbols
expr = pd.read_csv("results/gene_expression.csv", index_col=0)
# expr.index should be gene symbols: ['MAPT', 'SOD1', 'CYCS', ...]

fig = viz.plot_clique_network(
    regulator="MAPT",
    df=df,
    correlation_data=expr,
    correlation_threshold=0.5  # Only show correlations ≥ 0.5
)
fig.save("figures/mapt_network_corr.pdf")
```

## Interpretation Guide

### What to Look For

1. **Hub Genes (Large Nodes)**
   - Genes with many connections
   - Central to the regulatory network
   - Likely key players in the regulatory mechanism

2. **Peripheral Genes (Small Nodes)**
   - Fewer connections
   - May be less central to regulation
   - Could be downstream targets

3. **Color Patterns**
   - **Many emerald nodes**: Stable regulatory network across conditions
   - **Many blue/orange nodes**: Condition-specific rewiring
   - **Clusters of same color**: Genes that appear/disappear together

4. **Network Density**
   - **Dense network**: Tightly co-regulated genes
   - **Sparse network**: More independent regulation
   - **Disconnected components**: Potential subgroups

5. **Spatial Clustering**
   - **Tight clusters**: Strongly correlated genes
   - **Dispersed layout**: Weaker correlations overall

### Biological Insights

- **Stable genes (emerald)**: Core regulatory machinery maintained across disease states
- **Disease-specific genes (blue)**: Potential disease biomarkers or therapeutic targets
- **Lost genes (orange)**: Functions lost in disease progression
- **Hub genes**: Master regulators or key network connectors

## Comparison with Other Visualizations

| Visualization | Shows | Best For |
|---------------|-------|----------|
| **Stratum Heatmap** (`plot_stratum_heatmap`) | Membership patterns across conditions | Understanding what changes |
| **Clique Network** (`plot_clique_network`) | Correlation structure within clique | Understanding why genes cluster |
| **Gene Flow** (`plot_gene_flow`) | Gene transitions between conditions | Tracking regulatory rewiring |

### Recommended Workflow

1. **Start with Stratum Heatmap**: Identify interesting regulators with clear condition differences
2. **Use Clique Network**: Understand the internal structure of selected cliques
3. **Use Gene Flow**: See how genes transition between healthy and disease states

## Output Files

The method returns a `Figure` object that can be saved in multiple formats:

```python
fig = viz.plot_clique_network("MAPT", df)

# Vector formats (best for publication)
fig.save("figures/mapt_network.pdf")
fig.save("figures/mapt_network.svg")

# Raster formats (for presentations/web)
fig.save("figures/mapt_network.png", dpi=300)
fig.save("figures/mapt_network.jpg", dpi=150)
```

## Implementation Details

- **Network Library**: NetworkX for graph construction and layout
- **Layout Algorithm**: Spring layout (force-directed) with reproducible seed
- **Correlation Method**: Pearson correlation (if data provided)
- **Edge Filtering**: Only edges with |correlation| ≥ threshold are drawn
- **Fallback Behavior**: If gene symbols don't match, creates uniform-weight network

## Testing

A test script is provided at `/Users/noot/Documents/biomolecular-clique-finding/test_clique_network.py`:

```bash
cd /Users/noot/Documents/biomolecular-clique-finding
source .venv/bin/activate
python test_clique_network.py
```

This generates example networks for MAPT and SLC2A1 in `figures/test/`.

## Troubleshooting

### "No data found for regulator"
- Check that the regulator name exactly matches the `regulator` column in the cliques DataFrame
- Regulator names are case-sensitive (use "MAPT" not "mapt")

### "Only 0 genes found in expression data"
- This is expected if using `proteomics_imputed.data.csv` (uses UniProt IDs)
- The network will automatically fall back to membership-only mode
- To use correlations, provide expression data with gene symbols as row indices

### Network looks too dense/sparse
- Adjust `correlation_threshold` parameter (lower = more edges, higher = fewer edges)
- Default is 0.4, try values between 0.3-0.7

### Nodes are too small/large
- Adjust `node_size_scale` parameter (default: 300)
- Try values between 200-500

### Layout looks chaotic
- Increase `layout_iterations` (default: 50, try 100-200)
- Adjust `layout_k` for different spacing (smaller k = tighter clusters)

## Future Enhancements

Potential additions to consider:

1. **UniProt ID Mapping**: Automatically convert between UniProt IDs and gene symbols
2. **Edge Color by Sign**: Color edges by positive (blue) vs negative (red) correlation
3. **Community Detection**: Highlight subnetworks/modules within the clique
4. **Interactive Version**: Plotly-based interactive network with hover tooltips
5. **Differential Networks**: Show how correlation structure changes between conditions

## Citations

If using this visualization in publications, cite the appropriate network analysis papers:

- NetworkX: Hagberg, A., Swart, P., & Schult, D. (2008). Exploring network structure, dynamics, and function using NetworkX.
- Force-directed layout: Fruchterman, T. M., & Reingold, E. M. (1991). Graph drawing by force-directed placement.
