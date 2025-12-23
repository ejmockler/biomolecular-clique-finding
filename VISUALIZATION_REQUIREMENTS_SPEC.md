# Visualization Requirements Specification
## Biomolecular Clique-Finding Pipeline for ALS Proteomics

**Document Purpose**: Define what ALS researchers NEED to see to trust, understand, and interpret results from the imputation and clique-finding pipeline.

**Audience**: Domain scientists (proteomics/transcriptomics researchers), not software engineers.

**Date**: 2025-12-17

---

## Executive Summary

This document specifies visualization requirements for a two-phase analysis pipeline:
1. **Imputation Phase**: MAD-Z outlier detection + winsorization/median imputation with sex stratification
2. **Clique-Finding Phase**: Correlation network construction + maximal clique enumeration + INDRA CoGEx enrichment

Each visualization answers a specific scientific question and follows domain conventions from proteomics/transcriptomics literature.

---

## Part I: Imputation & QC Visualizations

### 1. OUTLIER DETECTION SUMMARY PLOT

**Scientific Question**: "Did outlier detection work correctly? Is my threshold appropriate?"

**Visualization Type**: Multi-panel figure (2x2 layout)

**Panels**:
1. **Distribution of MAD-Z scores** (Top-left)
   - X-axis: MAD-Z score bins (0-10+)
   - Y-axis: Frequency (log scale)
   - Visual elements:
     - Histogram of all MAD-Z scores across all values
     - Vertical line at threshold (default: 3.5)
     - Shaded region beyond threshold (outlier zone, red/orange)
     - Text annotation: "Outliers: N (X.XX%)"
   - Domain convention: MAD-Z > 3.5 is standard cutoff (Leys et al. 2013)

2. **Outlier rate per protein** (Top-right)
   - X-axis: Proteins (can be binned into percentiles)
   - Y-axis: % of samples flagged as outliers
   - Visual elements:
     - Sorted bar plot or violin plot
     - Horizontal line at 1% (expected outlier rate for biological data)
     - Color code: green (<5%), yellow (5-10%), red (>10%)
   - Question answered: "Are outliers concentrated in specific proteins?"

3. **Outlier rate per sample** (Bottom-left)
   - X-axis: Samples (grouped by phenotype × sex)
   - Y-axis: % of proteins flagged as outliers
   - Visual elements:
     - Box plots per group (CASE Male, CASE Female, CTRL Male, CTRL Female)
     - Overlaid strip plot showing individual samples
     - Horizontal line at 1%
   - Question answered: "Are there problematic samples with excessive outliers?"

4. **Outlier pattern heatmap** (Bottom-right)
   - X-axis: Samples (clustered by similarity)
   - Y-axis: Top 50 proteins with highest outlier counts
   - Visual elements:
     - Binary heatmap: Outlier (red) vs Clean (white)
     - Row/column dendrograms from hierarchical clustering
     - Sample annotation bar: phenotype (blue=CASE, gray=CTRL), sex (pink=F, blue=M)
   - Question answered: "Do outliers show batch structure or biological patterns?"

**Essential Interactions**:
- Hover: Show protein ID, sample ID, actual value, MAD-Z score
- Click protein: Highlight across all panels
- Click sample: Highlight across all panels
- Export: High-resolution PNG/PDF for publication

**Domain Conventions**:
- Use colorblind-safe palettes (viridis, ColorBrewer Set2)
- Log scale for heavily skewed distributions
- Always show N and percentages
- Red = warning/problem, green = good

---

### 2. STRATIFICATION EFFECTIVENESS PLOT

**Scientific Question**: "Did sex stratification improve outlier detection? Are groups comparable?"

**Visualization Type**: Before/after comparison + group statistics

**Panels**:
1. **Outlier counts: Global vs Stratified** (Left)
   - X-axis: Detection mode (Global, Phenotype-only, Phenotype×Sex)
   - Y-axis: Number of outliers detected
   - Visual elements:
     - Bar chart with error bars (if multiple runs)
     - Connecting lines showing per-protein changes
     - Color: consistent across modes
   - Question answered: "Does stratification catch different outliers?"

2. **Group sample sizes** (Top-right)
   - Stacked bar chart showing sample counts per stratification group
   - Groups: CASE×Male, CASE×Female, CTRL×Male, CTRL×Female
   - Annotation: Minimum group size (must be ≥10 for robust MAD-Z)
   - Question answered: "Are groups large enough for robust statistics?"

3. **MAD-Z score distributions by group** (Bottom-right)
   - Overlaid density plots per stratification group
   - X-axis: MAD-Z score
   - Y-axis: Density
   - Visual elements:
     - One curve per group (4 total)
     - Vertical line at threshold (3.5)
     - Legend with group labels
   - Question answered: "Do groups have similar variance structure?"

**Essential Interactions**:
- Toggle: Show/hide individual groups
- Hover: Sample count, mean/median MAD-Z
- Export: Combined figure for methods section

**Domain Conventions**:
- Groups must have N ≥ 10 for valid MAD-Z (statistical requirement)
- Warn if groups are imbalanced (largest/smallest > 3:1 ratio)
- Use consistent group color scheme across all plots

---

### 3. IMPUTATION QUALITY ASSESSMENT

**Scientific Question**: "Did imputation preserve biological signal without introducing artifacts?"

**Visualization Type**: Diagnostic multi-panel figure

**Panels**:
1. **Before/After Value Distributions** (Top-left)
   - X-axis: Log2 expression value
   - Y-axis: Density
   - Visual elements:
     - Overlaid density curves: Original (gray), Imputed values only (red), Final (blue)
     - Rug plot showing outlier positions before imputation
   - Question answered: "Are imputed values in reasonable range?"

2. **Imputation Impact Per Gene** (Top-right)
   - X-axis: Genes (sorted by imputation rate)
   - Y-axis: % values imputed
   - Visual elements:
     - Bar plot colored by imputation rate: <5% (green), 5-20% (yellow), >20% (red)
     - Horizontal line at 20% (warning threshold)
     - Text: "N genes >20% imputed"
   - Question answered: "Are too many values imputed in any gene?"

3. **Correlation Preservation Check** (Bottom-left)
   - X-axis: Correlation before imputation (pairwise, excluding outliers)
   - Y-axis: Correlation after imputation (pairwise, including imputed)
   - Visual elements:
     - Scatter plot with density coloring
     - Diagonal reference line (y=x)
     - R² and RMSE annotations
     - Color by point density (many points expected)
   - Question answered: "Did imputation preserve correlation structure?"

4. **PCA: Before vs After** (Bottom-right)
   - Side-by-side PC1 vs PC2 plots
   - Left: Before imputation (outliers excluded)
   - Right: After imputation (all values)
   - Visual elements:
     - Points colored by phenotype (CASE/CTRL)
     - Point shape by sex (circle/triangle)
     - Ellipses: 95% confidence per group
     - Variance explained in axis labels: "PC1 (X.X%)"
   - Question answered: "Did imputation shift sample clustering?"

**Essential Interactions**:
- Hover on correlation plot: Show gene pair IDs
- Hover on PCA: Sample ID, phenotype, sex, imputation rate
- Click gene in panel 2: Highlight in other panels
- Zoom: All scatter plots

**Domain Conventions**:
- Log2 scale for expression (standard in proteomics/RNA-seq)
- PCA plots always show variance explained
- Correlation plots use density shading for overplotting
- Flag genes with >50% imputed values (unreliable)

---

### 4. SEX CLASSIFICATION PERFORMANCE

**Scientific Question**: "How accurate was sex prediction? Can I trust the stratification?"

**Visualization Type**: Classification performance report

**Panels**:
1. **Confusion Matrix** (Left)
   - 2×2 grid: Predicted Sex (cols) × True Sex (rows)
   - Cell values: Sample counts
   - Cell colors: Green diagonal (correct), red off-diagonal (errors)
   - Annotations: Per-cell percentages
   - Question answered: "What's the classification accuracy?"

2. **ROC Curve** (Top-right)
   - X-axis: False positive rate
   - Y-axis: True positive rate
   - Visual elements:
     - ROC curve (blue line)
     - Diagonal reference (gray dashed, random classifier)
     - AUC annotation: "AUC = 0.XX"
     - Operating point marker (chosen threshold)
   - Question answered: "How well separated are the sexes?"

3. **Sex Marker Expression** (Bottom-right)
   - Box plots for top sex-discriminating proteins
   - X-axis: Sex (Male, Female)
   - Y-axis: Log2 expression
   - Visual elements:
     - One box plot per marker protein (e.g., XIST, UTY, RPS4Y1)
     - Overlay strip plot of individual samples
     - P-values annotated (t-test or Mann-Whitney)
   - Question answered: "Which proteins drive sex classification?"

**Essential Interactions**:
- Hover on confusion matrix: Sample list
- Click ROC point: Show samples at that threshold
- Export: Classification report as table + plots

**Domain Conventions**:
- CV accuracy > 90% is acceptable for sex classification
- Known sex markers: XIST (female), UTY/RPS4Y1/KDM5D (male)
- Report both accuracy and balanced accuracy (if classes imbalanced)

---

### 5. QUALITY FLAG PROVENANCE HEATMAP

**Scientific Question**: "Which values were modified? Can reviewers trace data provenance?"

**Visualization Type**: Interactive data quality matrix

**Visual Design**:
- X-axis: Samples (grouped by phenotype × sex)
- Y-axis: Proteins (user-selectable subset or all)
- Cells colored by quality flag:
  - White: ORIGINAL (untouched)
  - Orange: OUTLIER_DETECTED
  - Red: OUTLIER_DETECTED | IMPUTED
  - Gray: MISSING_ORIGINAL | IMPUTED
  - (Future: Blue for BATCH_CORRECTED, etc.)
- Annotations:
  - Row totals: "N flagged per protein"
  - Column totals: "N flagged per sample"
  - Overall stats: "X.X% original, Y.Y% imputed"

**Essential Interactions**:
- Zoom/pan: Handle 3,264 proteins × 423 samples
- Filter: Show only flagged values
- Hover: Show actual value, flag details, imputation method
- Click cell: Show detailed provenance (transforms applied, parameters)
- Export: CSV with quality flags for reviewers

**Domain Conventions**:
- This is critical for reproducibility and reviewer trust
- Must be exportable for supplementary materials
- Support filtering to "show me all imputed values"

---

## Part II: Correlation Network & Clique Visualizations

### 6. CORRELATION MATRIX HEATMAP (Stratified)

**Scientific Question**: "How do protein correlations differ between CASE and CTRL?"

**Visualization Type**: Side-by-side correlation heatmaps

**Panels**:
1. **CASE correlation matrix** (Left)
2. **CTRL correlation matrix** (Right)
3. **Differential correlation** (Optional third panel)

**Visual Design**:
- X/Y-axes: Proteins (same ordering across panels)
- Cell color: Correlation coefficient
  - Color scale: Blue (negative) → White (zero) → Red (positive)
  - Use diverging colormap (RdBu_r or coolwarm)
  - Range: -1 to +1
- Annotations:
  - Row/column labels: Gene symbols (if <100 proteins) or hidden (if >100)
  - Dendrograms: Hierarchical clustering of proteins
  - Title: "CASE (N=348)" / "CTRL (N=163)"

**Essential Interactions**:
- Hover: Show gene pair, correlation value, p-value, N samples
- Click row/col: Highlight that protein's correlations across both panels
- Reorder: By clustering, by correlation strength, alphabetically
- Filter: FDR-controlled threshold (show only significant correlations)
- Zoom: Box zoom to inspect specific protein clusters
- Export: High-res figure + correlation matrices as CSV

**Domain Conventions**:
- Always show same protein ordering in CASE vs CTRL for comparison
- Hierarchical clustering uses average linkage by default
- Correlation method: Pearson (if log-normal) or Spearman (if non-parametric)
- Annotate N samples used (critical for power interpretation)

---

### 7. CORRELATION NETWORK GRAPH

**Scientific Question**: "What are the major protein co-expression modules?"

**Visualization Type**: Force-directed network graph

**Visual Design**:
- **Nodes**: Proteins
  - Size: Degree (number of connections)
  - Color: Community assignment (Leiden/Louvain clustering)
  - Label: Gene symbol (toggle on/off)
  - Border: TF status (thick border if transcription factor)
- **Edges**: Significant correlations (FDR < 0.05)
  - Width: Correlation strength (thicker = stronger)
  - Color: Positive (red) vs Negative (blue)
  - Opacity: Edge weight
- **Layout**: Force-directed (Fruchterman-Reingold or ForceAtlas2)
  - Alternative: Circular layout by community

**Panels**:
- Main network view (center)
- Statistics sidebar (right):
  - N nodes, N edges
  - Network density
  - N communities detected
  - Community size distribution
- Legend (bottom):
  - Node color = community
  - Edge color = correlation sign
  - Node size = degree

**Essential Interactions**:
- Hover node: Show gene symbol, degree, community, expression summary
- Click node: Highlight first-degree neighbors
- Hover edge: Show gene pair, correlation, p-value
- Filter edges: By correlation threshold (slider)
- Filter nodes: By degree (show only hubs)
- Community selection: Click community to isolate
- Export: Network as GraphML, plot as SVG/PNG

**Domain Conventions**:
- Only show FDR-corrected significant edges (avoid hairball)
- Use perceptually uniform color palette for communities
- Hub nodes (high degree) often represent key regulators
- Negative correlations are biologically meaningful (show them)

---

### 8. CLIQUE ENUMERATION SUMMARY

**Scientific Question**: "What cliques were found? How do they differ between CASE and CTRL?"

**Visualization Type**: Comparative clique statistics

**Panels**:
1. **Clique Size Distribution** (Top-left)
   - X-axis: Clique size (number of proteins)
   - Y-axis: Count of cliques
   - Visual elements:
     - Side-by-side bars: CASE (blue), CTRL (orange)
     - Annotations: Total cliques per condition
   - Question answered: "Are cliques larger in disease?"

2. **Shared vs Unique Cliques** (Top-right)
   - Venn diagram or UpSet plot
   - Categories: CASE-only, CTRL-only, Shared
   - Visual elements:
     - Counts and percentages
     - Color: CASE (blue), CTRL (orange), Shared (purple)
   - Question answered: "Are there disease-specific modules?"

3. **Top Cliques Table** (Bottom)
   - Sortable table showing largest/most significant cliques
   - Columns:
     - Clique ID
     - Size (N proteins)
     - Condition (CASE/CTRL/Shared)
     - Mean correlation
     - Proteins (gene symbols, truncated with "...")
     - Enrichment (GO term / pathway, if computed)
   - Question answered: "What are the key modules to investigate?"

**Essential Interactions**:
- Click clique in table: Highlight in network graph
- Sort table: By size, correlation, enrichment p-value
- Export: Clique membership as CSV
- Filter: Min clique size (default: 3)

**Domain Conventions**:
- Maximal cliques only (remove redundant subsets)
- Report both count and total protein coverage
- Cliques of size 2 are edges (not interesting)
- Cliques > 20 are rare and biologically significant

---

### 9. CLIQUE MEMBERSHIP HEATMAP

**Scientific Question**: "Which proteins participate in multiple cliques? Are there overlapping modules?"

**Visualization Type**: Bipartite heatmap

**Visual Design**:
- X-axis: Cliques (sorted by size)
- Y-axis: Proteins (sorted by clique participation count)
- Cells:
  - Filled (black): Protein is in clique
  - Empty (white): Protein not in clique
- Annotations:
  - Row totals: "N cliques per protein"
  - Column totals: "Clique size"
  - Color sidebar: Protein function (if available, e.g., TF vs target)

**Essential Interactions**:
- Hover: Show protein-clique membership
- Click protein: Highlight all cliques containing it
- Click clique: Show member proteins in detail
- Filter: Show only cliques >size N
- Export: Membership matrix as CSV

**Domain Conventions**:
- High-participation proteins are "hub" proteins (biologically important)
- Overlapping cliques suggest hierarchical module structure
- Can be supplemented with side-by-side CASE vs CTRL views

---

### 10. DIFFERENTIAL CLIQUE ANALYSIS

**Scientific Question**: "Which cliques are gained or lost in ALS?"

**Visualization Type**: Comparative analysis figure

**Panels**:
1. **Gained Cliques (CASE-specific)** (Left)
   - List of cliques present in CASE but not CTRL
   - Visualized as mini-network graphs
   - Colored red (disease-associated)
   - Annotated with:
     - Clique size
     - Mean correlation in CASE
     - Mean correlation in CTRL (if partial overlap)

2. **Lost Cliques (CTRL-specific)** (Right)
   - List of cliques present in CTRL but not CASE
   - Visualized as mini-network graphs
   - Colored blue (healthy-associated)
   - Same annotations as gained cliques

3. **Comparison Metrics** (Bottom)
   - Table comparing CASE vs CTRL:
     - N total cliques
     - N unique cliques
     - N shared cliques
     - Mean clique size
     - Network density

**Essential Interactions**:
- Click clique: Expand to show full network context
- Hover: Correlation values, member proteins
- Export: Gained/lost lists for functional enrichment

**Domain Conventions**:
- Use permutation test to assess statistical significance of gains/losses
- Control for network size differences (CASE has more samples)
- Report effect sizes, not just counts

---

## Part III: Biological Interpretation Visualizations

### 11. INDRA COGEX REGULATORY MODULE VIEW

**Scientific Question**: "Do my cliques match known regulatory relationships?"

**Visualization Type**: Regulatory network overlay

**Visual Design**:
- **Two-layer network**:
  - Layer 1 (top): Transcription factors (TFs)
    - Nodes: TFs from INDRA CoGEx
    - Color: By TF family or activity (activator vs repressor)
    - Size: By number of targets in our data
  - Layer 2 (bottom): Target genes
    - Nodes: Genes in expression data
    - Color: By clique membership
    - Border: Highlighted if in detected clique
  - **Edges**: TF → target relationships
    - From INDRA CoGEx (IncreaseAmount/DecreaseAmount)
    - Color: Green (activation), Red (repression)
    - Width: Evidence count from INDRA

- **Annotations**:
  - Legend: Edge types, node types
  - Statistics: "N TFs with targets in data", "N INDRA edges overlapping cliques"

**Essential Interactions**:
- Click TF: Highlight all its targets
- Click target: Show which TFs regulate it
- Filter: Min evidence count (e.g., ≥3 INDRA statements)
- Toggle: Show only INDRA-validated cliques
- Export: TF-target pairs as CSV

**Domain Conventions**:
- INDRA edges are literature-curated (high confidence)
- Evidence count matters (≥3 is robust)
- Distinguish activation vs repression (opposite effects)
- This validates that cliques are not just statistical noise

---

### 12. ENRICHMENT ANALYSIS PLOT

**Scientific Question**: "What biological processes are enriched in my cliques?"

**Visualization Type**: GO term / pathway enrichment dot plot

**Visual Design**:
- X-axis: -log10(FDR-adjusted p-value)
  - Vertical line at significance threshold (FDR < 0.05)
- Y-axis: GO terms / pathways (top 20, sorted by p-value)
- Points:
  - Color: Enrichment fold change (gene ratio)
  - Size: Number of genes in term
- Panels:
  - Separate plots for CASE-specific vs CTRL-specific cliques
  - Side-by-side for comparison

**Essential Interactions**:
- Hover: Full GO term name, gene list, statistics
- Click term: Show which clique(s) contributed
- Filter: By GO category (BP, MF, CC) or pathway database (KEGG, Reactome)
- Export: Enrichment table as CSV

**Domain Conventions**:
- Use Benjamini-Hochberg FDR correction (standard in genomics)
- Report both p-value and gene ratio
- GO terms are hierarchical (avoid redundant parent terms)
- KEGG/Reactome for pathway-level interpretation

---

### 13. CLIQUE EXPRESSION PROFILE PLOT

**Scientific Question**: "How are clique members expressed across samples?"

**Visualization Type**: Heatmap with expression profiles

**Visual Design**:
- **Main heatmap**:
  - X-axis: Samples (grouped by phenotype × sex)
  - Y-axis: Proteins in clique (clustered by expression similarity)
  - Cells: Log2 expression value
  - Color scale: Low (blue) → High (red), diverging around median
- **Annotations**:
  - Sample annotation bars (top):
    - Phenotype (CASE/CTRL)
    - Sex (M/F)
    - (Optional: Clinical covariates if available)
  - Protein annotation bars (left):
    - TF status
    - Clique membership (if protein in multiple cliques)
  - Row/column dendrograms from hierarchical clustering

**Essential Interactions**:
- Hover cell: Protein, sample, value, quality flag
- Click protein: Show expression trajectory across samples
- Toggle: Show/hide sample groups
- Export: Expression matrix as CSV

**Domain Conventions**:
- Z-score transformation for visualization (not analysis!)
- Cluster both rows and columns for pattern discovery
- Show quality flags (mark imputed values with overlay symbol)
- Include biological replicates if available

---

### 14. VOLCANO PLOT (Differential Expression Context)

**Scientific Question**: "Are clique members differentially expressed in ALS?"

**Visualization Type**: Volcano plot with clique overlay

**Visual Design**:
- X-axis: Log2 fold change (CASE vs CTRL)
  - Vertical lines at ±log2(1.5) (common threshold)
- Y-axis: -log10(adjusted p-value)
  - Horizontal line at FDR < 0.05
- Points: All proteins
  - Color:
    - Gray: Not significant
    - Blue: Downregulated in CASE
    - Red: Upregulated in CASE
    - Gold: In detected cliques
  - Size: Larger if in clique
- Annotations:
  - Label top 10 most significant clique members
  - Quadrant counts: "N upreg, N downreg"

**Essential Interactions**:
- Hover: Gene symbol, fold change, p-value, clique membership
- Click point: Highlight in network graph
- Toggle: Show only clique members
- Export: DE results with clique annotations

**Domain Conventions**:
- Use robust DE method (limma, DESeq2, or t-test with Welch correction)
- Control for covariates (sex, batch) if present
- Report both raw and adjusted p-values
- This contextualizes cliques within overall dysregulation

---

### 15. MULTI-OMICS INTEGRATION VIEW (Future)

**Scientific Question**: "How do protein cliques relate to RNA expression?"

**Visualization Type**: Cross-modal correlation plot

**Visual Design**:
- X-axis: Protein expression (log2)
- Y-axis: RNA expression (log2)
- Points: Matched protein-RNA pairs
  - Color: By clique membership (clique vs non-clique)
  - Shape: By correlation sign
- Regression line:
  - Separate for clique vs non-clique
  - Annotate R² and slope
- Panels:
  - Separate plots per clique (if few) or combined with color coding

**Essential Interactions**:
- Hover: Gene symbol, protein value, RNA value, correlation
- Click point: Show detailed profiles
- Toggle: Show/hide non-clique genes
- Export: Cross-modal correlation table

**Domain Conventions**:
- Protein-RNA correlation is often modest (0.3-0.5 typical)
- Cliques may show stronger cross-modal coherence
- Control for time lag (RNA precedes protein)

---

## Part IV: Interactive Dashboard Requirements

### 16. DASHBOARD OVERVIEW

**Scientific Question**: "What's the summary of my entire analysis?"

**Visual Design**: Single-page summary dashboard

**Components**:
1. **Sample Overview** (Top-left)
   - Bar chart: Sample counts by phenotype × sex
   - Text summary: Total samples, features, imputation rate

2. **Outlier Summary** (Top-right)
   - Donut chart: Original vs Imputed values
   - Text: % outliers detected, % imputed

3. **Network Statistics** (Middle-left)
   - Summary cards:
     - N proteins in network
     - N significant edges
     - Network density
     - N cliques detected
   - Sparkline: Clique size distribution

4. **Top Cliques Preview** (Middle-right)
   - Table: Top 5 cliques by size or enrichment
   - Mini-network thumbnails

5. **Differential Analysis** (Bottom)
   - Bar chart: CASE vs CTRL clique counts
   - Text: N gained, N lost, N shared

**Essential Interactions**:
- Click any component: Navigate to detailed view
- Refresh: Re-run analysis with new parameters
- Export: Dashboard as PDF report

**Domain Conventions**:
- One-page summary for presentations
- All numbers linkable to underlying data
- Timestamp and parameters displayed

---

## Design Principles & Domain Conventions

### Color Palettes

**Phenotype**:
- CASE (disease): Blue (#3182BD)
- CTRL (healthy): Orange (#FD8D3C) or Gray (#969696)

**Sex**:
- Male: Light blue (#6BAED6)
- Female: Pink (#FC9272) or Purple (#9E9AC8)

**Quality Flags**:
- Original: White or light gray
- Outlier: Orange (#FEB24C)
- Imputed: Red (#E31A1C)
- Missing: Gray (#BDBDBD)

**Correlation/Expression**:
- Diverging: Blue-White-Red (RdBu_r)
- Sequential: Viridis or Blues for single-direction

**Statistical Significance**:
- Significant: Red or gold
- Not significant: Gray

**Accessibility**:
- All palettes must be colorblind-safe (use ColorBrewer)
- Provide texture/shape alternatives to color where possible

---

### Typography & Labeling

**Gene Symbols**:
- Always italicized (per biology convention): *TP53*, *SOD1*
- Uppercase for human genes (HUGO convention)

**Axis Labels**:
- Always include units: "Log2 Expression", "Correlation (Pearson r)"
- Font size: 12pt minimum for readability

**Statistical Annotations**:
- P-values: Scientific notation for p < 0.001 (e.g., "p = 2.3×10⁻⁵")
- Effect sizes: "FC = 1.5×" (fold change), "r = 0.75" (correlation)
- Sample sizes: "N = 348" or "n = 10 genes"

**Titles**:
- Informative, not decorative: "Outlier Detection: CASE Male (N=219)"
- Include key parameters: "Correlation Network (r > 0.7, FDR < 0.05)"

---

### Statistical Rigor

**Multiple Testing Correction**:
- Always use FDR (Benjamini-Hochberg) for genomics-scale comparisons
- Report both raw and adjusted p-values
- Threshold: FDR < 0.05 (standard)

**Effect Sizes**:
- Report correlation coefficients (r), not just p-values
- Report fold changes (log2FC) for differential expression
- Report N samples used (critical for power)

**Confidence Intervals**:
- Show 95% CI on mean differences (error bars or shaded regions)
- Use bootstrapping if distribution unknown

---

### Interactivity Essentials

**All Plots Must Support**:
1. **Hover tooltips**: Show data values, labels, statistics
2. **Export**: PNG (high-res, 300 dpi), SVG (vector), PDF (publication)
3. **Zoom/pan**: For scatter plots, heatmaps, networks
4. **Linked views**: Click in one plot highlights across all plots

**Advanced Interactions** (priority):
1. **Filter sliders**: Correlation threshold, p-value cutoff, clique size
2. **Toggle groups**: Show/hide phenotypes, sexes, cliques
3. **Reordering**: Heatmap rows/columns by clustering, alphabetical, custom
4. **Search**: Find gene by symbol in any plot

**Performance Requirements**:
- Render plots with <2 second latency for datasets up to:
  - 5,000 proteins
  - 1,000 samples
  - 10,000 edges
- Use WebGL for networks >1,000 nodes
- Downsample scatter plots >10,000 points (with datashader)

---

### Publication-Ready Output

**Figure Formats**:
- **Main figures**: PDF (vector) or TIFF (300 dpi, CMYK)
- **Supplementary figures**: High-res PNG (300 dpi)
- **Web figures**: SVG (interactive) or PNG (static)

**Figure Composition**:
- Multi-panel figures use consistent font sizes
- Panels labeled A, B, C (top-left corner, bold)
- Shared legends placed once (not per panel)
- Consistent color schemes across panels

**Data Export**:
- All underlying data exportable as CSV or Excel
- Include metadata: date, parameters, software version
- README file explaining column headers

---

### Responsive Design

**Desktop** (primary):
- Layout: Multi-panel grids (2×2, 3×2)
- Font size: 12-14pt
- Interactive features: Full

**Tablet**:
- Layout: Single-column scrolling
- Font size: 14-16pt
- Interactive features: Hover → tap

**Print**:
- Layout: Static, high-resolution
- Font size: 10-12pt
- Interactive features: None (export snapshots)

---

## Implementation Priorities

### Phase 1: Imputation QC (Immediate Need)
1. Outlier Detection Summary Plot (#1) - **CRITICAL**
2. Imputation Quality Assessment (#3) - **CRITICAL**
3. Quality Flag Provenance Heatmap (#5) - **CRITICAL**
4. Stratification Effectiveness Plot (#2)
5. Sex Classification Performance (#4)

### Phase 2: Network Exploration
6. Correlation Matrix Heatmap (#6) - **CRITICAL**
7. Correlation Network Graph (#7) - **CRITICAL**
8. Clique Enumeration Summary (#8)
9. Clique Membership Heatmap (#9)
10. Differential Clique Analysis (#10)

### Phase 3: Biological Interpretation
11. INDRA CoGEx Regulatory Module View (#11)
12. Enrichment Analysis Plot (#12)
13. Clique Expression Profile Plot (#13)
14. Volcano Plot (#14)

### Phase 4: Integration & Dashboards
15. Multi-Omics Integration View (#15) - **Future**
16. Dashboard Overview (#16)

---

## Success Criteria

**Scientist can answer**:
- ✓ "Are my outliers real or artifacts?"
- ✓ "Did imputation distort biological signal?"
- ✓ "Which samples are problematic?"
- ✓ "What proteins co-vary together?"
- ✓ "How do networks differ in disease vs control?"
- ✓ "Do my cliques match known biology?"
- ✓ "What pathways are enriched?"
- ✓ "Can I trust these results for publication?"

**Reviewer can verify**:
- ✓ "Which values were imputed?"
- ✓ "What parameters were used?"
- ✓ "Are statistics correctly applied?"
- ✓ "Is multiple testing corrected?"

**Journal accepts**:
- ✓ High-resolution publication-quality figures
- ✓ Supplementary data tables
- ✓ Methods section fully specified

---

## References & Standards

**Visualization Guidelines**:
- Tufte, E. (2001). *The Visual Display of Quantitative Information*
- Wilkinson, L. (2005). *The Grammar of Graphics*
- Rougier, N.P. et al. (2014). "Ten Simple Rules for Better Figures". *PLoS Comp Bio*.

**Statistical Methods**:
- Benjamini, Y. & Hochberg, Y. (1995). Controlling FDR. *J Royal Stat Soc B*.
- Leys, C. et al. (2013). Detecting outliers: MAD-based approach. *J Exp Soc Psych*.

**Bioinformatics Conventions**:
- HUGO Gene Nomenclature Committee (gene symbols)
- Gene Ontology Consortium (GO terms)
- Proteomics Standards Initiative (MS data reporting)

---

## Document Metadata

**Version**: 1.0
**Date**: 2025-12-17
**Author**: Scientific Visualization Expert
**Status**: Specification (awaiting implementation)
**Target Audience**: ALS researchers, bioinformaticians
**Implementation Framework**: To be determined (plotly/dash, matplotlib/seaborn, or custom)

---

### 17. GENE FLOW DIAGRAM (REGULATORY REWIRING)

**Scientific Question**: "How does regulatory control change from healthy to disease? Which genes are gained, lost, or stable?"

**Visualization Type**: Sankey-style flow diagram with gene lists

**Visual Design**:
- **Layout**: Two-column flow diagram
  - Left column: CTRL (Healthy) genes
  - Right column: CASE (ALS) genes
- **Flow ribbons**:
  - **Emerald (#059669)**: Genes GAINED in disease (appear only in CASE)
  - **Red (#ef4444)**: Genes LOST in disease (appear only in CTRL)
  - **Gray (#6b7280)**: STABLE genes (present in both conditions)
- **Flow width**: Proportional to number of genes in that category
- **Gene labels**:
  - Displayed within boxes (up to 15 genes per category)
  - Italicized per biology convention
  - Alphabetically sorted for easy scanning
- **Bezier curves**: Smooth flow ribbons connecting source to destination

**Panels**:
1. **CTRL column** (Left):
   - Box for "CTRL-only" genes (lost in disease) - red border
   - Box for "Stable" genes (present in both) - gray border
2. **CASE column** (Right):
   - Box for "Stable" genes (present in both) - gray border
   - Box for "CASE-only" genes (gained in disease) - emerald border
3. **Flows** (Middle):
   - Lost flow: CTRL-only → fades out (red, semi-transparent)
   - Stable flow: CTRL stable → CASE stable (gray, semi-transparent)
   - Gained flow: fades in → CASE-only (emerald, semi-transparent)

**Key Features**:
- **Immediate interpretability**: Color coding enables instant recognition of regulatory direction
- **Change detection**: Primary cognitive operation is detecting what changed
- **Proportional encoding**: Flow width provides magnitude information at a glance
- **Gene-level detail**: Individual genes visible for small cliques (<15 genes per category)
- **Scalability**: For large cliques, shows gene count instead of individual labels

**Essential Interactions**:
- Static visualization (matplotlib-based, no interactions needed)
- Export: PDF (vector) for publication
- Metadata: Full gene lists stored in Figure.metadata for programmatic access

**Domain Conventions**:
- Gene symbols italicized (*MAPT*, *SOD1*, etc.)
- CTRL (Healthy) = Orange header (#f97316)
- CASE (ALS) = Blue header (#2563eb)
- Left-to-right spatial layout implies temporal progression (healthy → disease)
- Flow direction encodes biological interpretation (gained/lost)

**Use Cases**:
1. **Regulatory rewiring analysis**: Understand how a regulator's targets change in disease
2. **Sex-specific effects**: Compare F vs M flows (optional parameter)
3. **Publication figures**: Clear visual for papers explaining regulatory changes
4. **Hypothesis generation**: Identify gained/lost genes for follow-up experiments

**Example Interpretation** (MAPT regulator):
- Lost in ALS: 7 genes (FUS, BASP1, GSK3B, etc.) → tau loses regulatory control
- Gained in ALS: 3 genes (NFASC, PARP1, SYP) → tau gains new inappropriate targets
- Stable: 15 genes → core tau regulatory network preserved

**Implementation**:
- Method: `CliqueVisualizer.plot_gene_flow(regulator, df)`
- Location: `/src/cliquefinder/viz/cliques.py`
- Dependencies: matplotlib, numpy, pandas
- Performance: <1s for typical clique sizes (10-30 genes)

**Statistical Context**:
- Gene categorization based on set operations (union, intersection, difference)
- Aggregates across sex (M+F) by default for simplicity
- Optional `show_sex_split=True` for sex-specific analysis
- No statistical testing (descriptive visualization only)

**Quality Metrics** (stored in metadata):
- `n_lost`: Number of genes lost in disease
- `n_gained`: Number of genes gained in disease
- `n_stable`: Number of stable genes
- `lost_genes`: List of specific genes lost
- `gained_genes`: List of specific genes gained
- `stable_genes`: List of specific stable genes

**Perceptual Engineering Design**:
- **Spatial invariance**: CTRL always left, CASE always right (builds spatial memory)
- **Color semantics**: Red = loss/warning, Emerald = gain/new, Gray = stable/neutral
- **Flow metaphor**: Curved ribbons suggest transformation/transition (not static comparison)
- **Width encoding**: Preattentive visual variable for magnitude (faster than counting labels)
- **Gestalt grouping**: Boxes group related genes, flows connect related states

---

**END OF SPECIFICATION**
