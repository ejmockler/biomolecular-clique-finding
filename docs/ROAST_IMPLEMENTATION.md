# ROAST Implementation: Rotation Gene Set Tests

## Overview

This document describes the implementation of ROAST (Rotation Gene Set Tests) in the cliquefinder package, providing a self-contained alternative to competitive permutation tests that preserves inter-gene correlation structure.

**Key Achievement**: Detect bidirectional TF regulation (genes going UP and DOWN simultaneously) which the OLS-based approach cannot detect due to signal cancellation.

### Scope and Limitations

> **Important**: This implementation supports **simple two-group pairwise comparisons** (e.g., CASE vs CTRL, C9 vs Sporadic). For designs with >2 groups, continuous covariates, or interaction terms, the full C-matrix reparameterization from Wu et al. (2010) would be required.

The implementation is mathematically valid for:
- Binary comparisons (exactly 2 conditions)
- No covariates in the model
- Simple group effects (no interactions)

For the cliquefinder use case (comparing disease groups), this scope is sufficient.

## Mathematical Foundation

### Linear Model Specification

For each gene $g$ in the dataset:

$$\mathbf{y}_g = X\boldsymbol{\alpha}_g + \boldsymbol{\varepsilon}_g$$

where:
- $\mathbf{y}_g \in \mathbb{R}^n$ is the expression vector across $n$ samples
- $X \in \mathbb{R}^{n \times p}$ is the design matrix (full column rank $p$)
- $\boldsymbol{\alpha}_g \in \mathbb{R}^p$ are the regression coefficients
- $\boldsymbol{\varepsilon}_g \sim N(0, \sigma_g^2 W^{-1})$

The contrast of interest is $\beta_g = \mathbf{c}^\top \boldsymbol{\alpha}_g$.

### QR Decomposition for Residual Extraction

To isolate the degrees of freedom relevant to the hypothesis test:

1. **Reparameterize** so the null hypothesis concerns the last coefficient
2. **Full QR decomposition**: $X = QR$ where $Q \in \mathbb{R}^{n \times n}$
3. **Extract** $Q_2 = Q[:, p-1:]$ (last $d+1$ columns, where $d = n - p$)
4. **Project** each expression vector: $\mathbf{u}_g = Q_2^\top \mathbf{y}_g \in \mathbb{R}^{d+1}$

Under $H_0$, elements $u_{g,0}, \ldots, u_{g,d}$ are independent $N(0, \sigma_g^2)$.

### The Rotation Mechanism

The key insight: under the null hypothesis, the residual vector $\mathbf{u}_g$ lies on a $(d+1)$-sphere of radius $\rho_g = \|\mathbf{u}_g\|$. Rotating $\mathbf{u}_g$ to ANY other point on this sphere produces an equally valid null sample.

For each rotation $r \in \{1, \ldots, B\}$:

1. Generate random direction: $\mathbf{v} \sim N(0, I_{d+1})$
2. Normalize: $\mathbf{r} = \mathbf{v} / \|\mathbf{v}\|$
3. Compute rotated first element: $u^*_{g,0} = \langle \mathbf{u}_g, \mathbf{r} \rangle$
4. Compute rotated variance: $s^{*2}_g = (\rho_g^2 - u^{*2}_{g,0}) / d$
5. Recompute moderated t-statistics and gene set statistic

### Empirical Bayes Moderation

Following Smyth (2004), gene-wise variances are shrunk toward a common prior:

$$\sigma_g^2 \sim \text{Inverse-}\chi^2(d_0, s_0^2)$$

Posterior variance:
$$s^2_{\text{post}} = \frac{d_0 \cdot s_0^2 + d \cdot s_g^2}{d_0 + d}$$

Moderated t-statistic:
$$\tilde{t}_g = \frac{\hat{\beta}_g}{s_{\text{post}} \sqrt{v}} \sim t_{d_0+d}$$

**Per-Rotation Shrinkage**: The implementation applies proper EB shrinkage to each rotated variance using the formula above, rather than a ratio approximation. The prior parameters ($d_0$, $s_0^2$) are estimated once from all genes and applied consistently across rotations.

### Gene Set Statistics

Let $z_g = \Phi^{-1}(F_{t_{d_0+d}}(\tilde{t}_g))$ be the z-score transformation. With optional weights $a_g$ and $A = \sum|a_g|$:

| Statistic | Formula | Best For |
|-----------|---------|----------|
| **mean** | $T = (\sum a_g z_g) / A$ | Coherent, same-direction regulation |
| **floormean** | $T = (\sum a_g \max(\|z_g\|, \sqrt{q})) / A$ | General purpose (dampens noise) |
| **mean50** | Mean of top 50% $\|z_g\|$ | Half of genes are DE |
| **msq** | $T = (\sum \|a_g\| z_g^2) / A$ | **Bidirectional regulation** |
| **mixed** | $(mean^2 + msq) / 2$ | Balanced detection |

**Critical**: The `msq` statistic is direction-agnostic, making it essential for detecting TFs that both activate AND repress targets.

### P-Value Computation

Exact Monte Carlo p-value:

$$p = \frac{b + 1}{B + 1}$$

where $b$ = count of rotations with $T^* \geq T_{\text{observed}}$.

## Comparison: ROAST vs OLS Permutation

| Aspect | OLS Permutation (Current) | ROAST (New) |
|--------|---------------------------|-------------|
| **Test Type** | Competitive | Self-contained |
| **Question** | "More DE than random genes?" | "Is this set DE at all?" |
| **Null Generation** | Permute gene membership | Rotate residuals |
| **Mixed Direction** | ❌ Signals cancel | ✅ MSQ detects |
| **Gene Correlation** | Disrupted by permutation | **Preserved** |
| **Computational** | O(B × k × summarization) | O(B × m × d) |

### When to Use Each

**Use OLS Permutation when:**
- Testing competitive enrichment ("special vs random")
- Cliques have coherent, same-direction regulation
- Gene-gene correlations are modest

**Use ROAST when:**
- Testing self-contained enrichment ("is there any effect?")
- TFs both activate AND repress targets
- Gene-gene correlations must be preserved
- Small sample sizes (rotation gives exact p-values)

## Implementation Architecture

### Core Components

```
src/cliquefinder/stats/rotation.py
├── Dataclasses
│   ├── RotationPrecomputed    # QR matrices (compute once)
│   ├── GeneEffects            # Projected gene vectors
│   ├── RotationResult         # Test results
│   └── RotationTestConfig     # Configuration
│
├── Core Functions
│   ├── compute_rotation_matrices()    # QR decomposition
│   ├── extract_gene_effects()         # Project genes to residual space
│   ├── generate_rotation_vectors()    # Random unit vectors
│   ├── apply_rotations_batched()      # GPU-accelerated rotation
│   └── compute_set_statistics()       # All 5 statistics
│
└── High-Level API
    ├── RotationTestEngine             # Main class
    └── run_rotation_test()            # Convenience function
```

### GPU Acceleration Strategy

1. **Precomputation** (once): QR decomposition, EB priors
2. **Batch Rotation** (vectorized): Generate all rotation vectors at once
3. **Batch Application** (GPU): Apply rotations via matrix multiplication
4. **Batch Statistics** (vectorized): Compute all statistics simultaneously

### MLX Integration

```python
# CPU fallback when MLX unavailable
if use_gpu and MLX_AVAILABLE:
    return _apply_rotations_gpu(U, rho_sq, R, ...)
else:
    return _apply_rotations_cpu(U, rho_sq, R, ...)
```

GPU path uses MLX for:
- Rotation vector generation
- Matrix multiplication (U @ R.T)
- Element-wise variance computation
- Large-batch t-statistic calculation

## Usage Examples

### Basic Usage

```python
from cliquefinder.stats import run_rotation_test

results = run_rotation_test(
    data=expression_matrix,        # (n_genes, n_samples)
    gene_ids=gene_names,
    metadata=sample_metadata,
    gene_sets={'TF1': ['GENE1', 'GENE2', ...], ...},
    conditions=['CASE', 'CTRL'],
    contrast=('CASE', 'CTRL'),
    n_rotations=9999,
    use_gpu=True,
)

# Get significant bidirectional regulation
sig_mixed = results[results['adj_pvalue_msq_mixed'] < 0.05]
```

### Engine-Based Usage (Multiple Contrasts)

```python
from cliquefinder.stats import RotationTestEngine, RotationTestConfig

engine = RotationTestEngine(data, gene_ids, metadata)

# Fit once per contrast
engine.fit(
    conditions=['CASE', 'CTRL'],
    contrast=('CASE', 'CTRL'),
    condition_column='phenotype',
)

# Test many gene sets efficiently
config = RotationTestConfig(n_rotations=9999, use_gpu=True, seed=42)
results = engine.test_gene_sets(gene_sets, config=config)

# Convert to DataFrame with FDR
df = engine.results_to_dataframe(results)
```

### Interpreting Results

```python
# The result contains p-values for all statistic/alternative combinations
result = engine.test_gene_set(gene_list)

# Directional tests (coherent regulation)
p_up = result.get_pvalue(SetStatistic.MEAN, Alternative.UP)
p_down = result.get_pvalue(SetStatistic.MEAN, Alternative.DOWN)

# Bidirectional test (critical for mixed regulation)
p_mixed = result.get_pvalue(SetStatistic.MSQ, Alternative.MIXED)

# Active gene proportion
active_frac = result.active_proportion['mixed']  # Fraction with |t| > sqrt(2)
```

## Performance Characteristics

### Benchmark (100 genes, 20 samples, 9999 rotations)

| Component | Time | Notes |
|-----------|------|-------|
| QR decomposition | ~1ms | One-time per contrast |
| Gene projection | ~0.1ms | O(n_genes × n_samples × d) |
| Rotation generation | ~5ms | Fully vectorized |
| Rotation application | ~50ms (GPU) | Batched matrix multiply |
| Statistics computation | ~10ms | Vectorized across rotations |
| **Total** | ~70ms | Per gene set |

### Scaling

- **Gene sets**: Linear in number of sets (can parallelize)
- **Rotations**: Linear, but amortized via batching
- **Genes per set**: Linear in rotation application
- **Samples**: Affects QR (one-time) and residual dims

## Validation

### Statistical Properties

1. **Type I Error**: Exact p-values for any sample size
2. **Correlation Preservation**: Rotation maintains gene-gene correlations
3. **Uniformity Under Null**: P-values are uniform when no signal present

### Test Coverage

```
tests/test_rotation.py
├── TestRotationVectors       # Unit norm, reproducibility, spherical coverage
├── TestQRDecomposition       # Shape, orthonormality, variance preservation
├── TestSetStatistics         # Direction, MSQ agnostic, weights
├── TestPValues               # Bounds, calibration
├── TestRotationEngine        # Integration tests with known signal
├── TestGPUCPUEquivalence     # Numerical agreement
└── TestActiveProportion      # Estimation accuracy
```

All 26 tests pass.

## References

1. Wu D, Lim E, Vaillant F, Asselin-Labat M-L, Visvader JE, Smyth GK (2010).
   "ROAST: rotation gene set tests for complex microarray experiments."
   *Bioinformatics* 26(17):2176-82.
   [PMC2922896](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922896/)

2. Smyth GK (2004). "Linear models and empirical Bayes methods for assessing
   differential expression in microarray experiments."
   *Statistical Applications in Genetics and Molecular Biology* 3(1):Article 3.

3. Langsrud O (2005). "Rotation tests."
   *Statistics and Computing* 15:53-60.

4. [limma Bioconductor package](https://bioconductor.org/packages/limma) - Reference implementation

5. [MLX Framework](https://github.com/ml-explore/mlx) - Apple Silicon GPU acceleration

## Future Enhancements

1. **General Contrasts**: Implement full C-matrix reparameterization for >2 groups and covariates
2. **Camera Integration**: Add inter-gene correlation adjustment for competitive tests
3. **FRY Approximation**: Fast closed-form approximation when df is large
4. **Multi-Contrast**: Batch multiple contrasts in single QR decomposition
5. **Streaming**: Process very large gene set collections via chunking
6. **CUDA Backend**: CuPy support for NVIDIA GPUs
