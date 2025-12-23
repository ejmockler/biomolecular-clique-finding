# Stratum Heatmap Grid: Perceptual Engineering Design

## Overview

The Stratum Heatmap Grid is a specialized visualization for comparing regulatory clique membership across demographic strata. It implements perceptual engineering principles to support spatial comparison and pattern recognition.

## Cognitive Operations

### Primary Task
**Spatial comparison across conditions** - Users need to quickly identify:
- Genes present only in healthy (CTRL) samples
- Genes present only in disease (CASE) samples
- Genes stable across both conditions
- Sex-specific regulatory differences

### Perceptual Channel
**Spatial memory through invariant layout** - The 2×2 grid position encodes condition information:

```
        Female     Male
       ┌────────┬────────┐
 CTRL  │ CTRL_F │ CTRL_M │  ← Top row = healthy (orange tint)
       ├────────┼────────┤
 CASE  │ CASE_F │ CASE_M │  ← Bottom row = disease (blue tint)
       └────────┴────────┘
```

This builds **spatial memory** - after viewing 2-3 heatmaps, users unconsciously learn "top = healthy, bottom = disease" without reading labels.

## Visual Encoding

### Position
- **Rows**: Genes (union across all strata)
- **Columns**: Fixed order [CTRL_F, CTRL_M, CASE_F, CASE_M]
- **Grid layout**: Always 2×2 with consistent quadrant meanings

### Color
- **Cell fill**:
  - Dark = gene present in clique
  - Light/white = gene absent
- **Background tint**:
  - Orange gradient for CTRL (healthy)
  - Blue gradient for CASE (disease)
  - Darker shade for Male, lighter for Female

### Typography
- **Gene symbols**: Italicized per biology convention (`$\mathit{GENE}$`)
- **Sample counts**: Shown in subplot titles (n=XX)

### Ordering
- **Hierarchical clustering**: Rows ordered to group co-occurring genes
- Uses Jaccard distance for binary presence/absence data
- Visual groups emerge without conscious effort

## Margin Annotations

Right margin shows gene categories:
- **○ (orange)**: CTRL-only genes (lost in disease)
- **● (blue)**: CASE-only genes (gained in disease)
- **■ (green)**: Stable genes (present in both)

This provides instant quantification without table lookup.

## Key Insights Surfaced

1. **Disease progression**: Genes only in top row → lost regulatory relationships
2. **Disease emergence**: Genes only in bottom row → gained regulatory relationships
3. **Robustness**: Full-row genes → stable core regulatory network
4. **Sex differences**: Left vs right column patterns

## Design Rationale

### Why 2×2 instead of 1×4?
The 2×2 layout creates **perceptual chunks**:
- Horizontal comparison = sex differences
- Vertical comparison = disease differences
- Diagonal patterns reveal interaction effects

A 1×4 layout would require serial scanning and working memory to track position meanings.

### Why clustering?
Unsupervised clustering reveals **co-occurrence patterns** without prior hypotheses:
- Genes that cluster together likely share regulatory mechanisms
- Visual gaps suggest functional modules
- Dendrogram (implicit) shows hierarchical relationships

### Why fixed column order?
**Consistency builds expertise**. After seeing multiple heatmaps:
1. Users develop spatial memory for positions
2. Scanning becomes automatic (preattentive)
3. Anomalies pop out immediately
4. Comparison across regulators is trivial

Random ordering would reset learning on every figure.

## Perceptual Hierarchy

```
Title (regulator name)
    ↓
2×2 Grid (phenotype × sex)
    ↓
Color tint (CTRL vs CASE)
    ↓
Cell patterns (presence/absence)
    ↓
Gene labels (left margin)
    ↓
Margin annotations (category counts)
    ↓
Caption (summary statistics)
```

Top-down reading: 3-5 seconds for gist, 30-60 seconds for details.

## Usage Patterns

### Quick Screening
Scan multiple regulators → identify interesting patterns → dive deep

### Hypothesis Generation
"Why are these genes co-occurring only in CASE?" → design follow-up experiments

### Validation
"We expect regulator X to lose gene Y in disease" → visual confirmation

### Presentation
Invariant layout allows side-by-side comparison in slides or papers

## Implementation Notes

### Data Requirements
- Input: `stratified_cliques.csv` with columns:
  - `regulator`: Gene symbol
  - `condition`: One of [CTRL_F, CTRL_M, CASE_F, CASE_M]
  - `n_samples`: Sample count for this stratum
  - `clique_genes`: Comma-separated gene list

### Computational Steps
1. Parse gene lists from comma-separated strings
2. Build presence/absence matrix (genes × strata)
3. Cluster rows using hierarchical clustering (Jaccard distance)
4. Create 2×2 subplot grid with GridSpec
5. Plot heatmaps with custom colormaps (orange/blue tints)
6. Add margin annotations for gene categories
7. Style gene labels with matplotlib italics

### Edge Cases Handled
- Missing strata (filled with empty sets)
- Single-gene cliques (no clustering)
- Clustering failures (fallback to original order)
- Empty cliques (all-white column)

## Comparison to Alternatives

### Alternative 1: Single heatmap with 4 columns
❌ **Problem**: No perceptual grouping, harder to compare CTRL vs CASE

### Alternative 2: Separate figures for each stratum
❌ **Problem**: Requires side-by-side windows, spatial memory lost

### Alternative 3: Venn diagrams
❌ **Problem**: Only 2-3 sets, doesn't show which genes

### Alternative 4: Network layouts
❌ **Problem**: Poor for membership comparison, spatial positions arbitrary

### Our solution
✅ **Wins**: Spatial consistency + perceptual grouping + scalability

## Future Enhancements

### Potential additions (only if needed):
1. **Dendrograms**: Show clustering tree on left margin
2. **Quantitative overlay**: Cell color intensity = correlation strength
3. **Interactive version**: Plotly with gene hover tooltips
4. **Statistical annotations**: Fisher exact test for enrichment
5. **Comparison mode**: Two regulators side-by-side

### NOT recommended:
- Animation (destroys spatial memory)
- 3D views (perceptual nightmare)
- Circular layouts (arbitrary orientation)
- Spring-force positioning (non-reproducible)

## References

### Perceptual principles applied:
- Tufte's "Small multiples" (consistent structure)
- Cleveland & McGill (position > color > size)
- Ware's "4±1 working memory chunks"
- Munzner's "Nested model" (task → encoding → algorithm)

### Biology conventions:
- Italicized gene symbols (standard across journals)
- CASE=blue, CTRL=orange (color-blind safe, semantic)
- Clustering for exploratory analysis (standard in genomics)
