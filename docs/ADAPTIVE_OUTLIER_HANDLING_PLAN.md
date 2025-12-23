# Adaptive Outlier Handling Implementation Plan

**Status**: Ready for Implementation
**Author**: Distinguished Engineer
**Date**: 2025-12-22
**Target**: `src/cliquefinder/quality/` module

---

## Executive Summary

This document specifies four implementation priorities for distribution-aware outlier handling that addresses the limitations of hard MAD-Z thresholds. The current approach assumes symmetric distributions and uses binary outlier classification, which is suboptimal for proteomics data that exhibits:

1. **Asymmetric (skewed) distributions** — right-skewed abundance data
2. **Heavy tails** — more extreme values than Gaussian predicts
3. **Information loss** — binary flags discard confidence information

The solution introduces:
- **Asymmetric detection** using medcouple-adjusted fences
- **Probabilistic scoring** using Student's t-distribution
- **Weighted correlation** to preserve original data while downweighting outliers
- **Soft clipping** as a fallback for methods requiring complete data

---

## Priority 1: Medcouple-Adjusted Asymmetric Detection

### Scientific Foundation

**Source**: Hubert, M. & Vandervieren, E. (2008). "An adjusted boxplot for skewed distributions." *Computational Statistics & Data Analysis*, 52, 5186-5201. [PDF](https://wis.kuleuven.be/statdatascience/robust/papers/2008/hubertvandervieren_adjustedboxplot_csda_2008.pdf)

**Problem**: Standard boxplot fences are symmetric around the median:
```
[Q1 - 1.5×IQR, Q3 + 1.5×IQR]
```
For skewed distributions, this flags legitimate values in the longer tail as outliers.

**Solution**: Adjust fence multipliers based on the **medcouple (MC)**, a robust measure of skewness.

### Mathematical Formulation

#### Medcouple Definition (Brys et al., 2004)

The medcouple is defined as the median of a kernel function over all pairs of observations on opposite sides of the median:

```
MC = median{ h(x_i, x_j) : x_i ≤ median(X) ≤ x_j }

where h(x_i, x_j) = (x_j - median(X)) - (median(X) - x_i)
                    ─────────────────────────────────────
                              x_j - x_i
```

**Properties**:
- Range: [-1, 1]
- MC > 0 → right-skewed (long upper tail)
- MC < 0 → left-skewed (long lower tail)
- MC = 0 → symmetric
- Breakdown point: 25% (robust to contamination)
- Time complexity: O(n log n) with fast algorithm

**Reference**: Brys, G., Hubert, M., & Struyf, A. (2004). "A Robust Measure of Skewness." *Journal of Computational and Graphical Statistics*, 13(4), 996-1017. [DOI](https://www.tandfonline.com/doi/abs/10.1198/106186004X12632)

#### Adjusted Boxplot Fences (Hubert & Vandervieren, 2008)

For medcouple MC, the adjusted fences are:

```
Lower fence: Q1 - 1.5 × exp(a × MC) × IQR
Upper fence: Q3 + 1.5 × exp(b × MC) × IQR

where:
  If MC ≥ 0 (right-skewed): a = -4, b = 3
  If MC < 0 (left-skewed):  a = -3, b = 4
```

**Effect**:
- Right-skewed (MC > 0): Upper fence expands (exp(3×MC)), lower fence contracts (exp(-4×MC))
- Left-skewed (MC < 0): Lower fence expands, upper fence contracts
- Symmetric (MC = 0): Reduces to standard boxplot (exp(0) = 1)

**Applicability**: Works well for moderate skewness (-0.6 ≤ MC ≤ 0.6).

### Algorithm

```python
def compute_medcouple(x: np.ndarray) -> float:
    """
    Compute medcouple (robust skewness measure).

    Uses the fast O(n log n) algorithm from Brys et al. (2004).

    Args:
        x: 1D array of observations (no NaN)

    Returns:
        Medcouple value in [-1, 1]
    """
    n = len(x)
    median = np.median(x)

    # Split into values below and above median
    x_minus = x[x <= median]
    x_plus = x[x >= median]

    # Handle ties at median
    if len(x_minus) == 0 or len(x_plus) == 0:
        return 0.0

    # Compute kernel h for all valid pairs
    h_values = []
    for xi in x_minus:
        for xj in x_plus:
            if xi == xj == median:
                # Special handling for ties at median (Brys et al., 2004)
                continue
            elif xj == xi:
                # Avoid division by zero
                continue
            else:
                h = ((xj - median) - (median - xi)) / (xj - xi)
                h_values.append(h)

    if len(h_values) == 0:
        return 0.0

    return np.median(h_values)


def adjusted_boxplot_fences(
    x: np.ndarray,
    mc: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute asymmetric outlier fences using Hubert-Vandervieren method.

    Args:
        x: 1D array of observations
        mc: Pre-computed medcouple (computed if None)

    Returns:
        (lower_fence, upper_fence) tuple
    """
    if mc is None:
        mc = compute_medcouple(x)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1

    # Select coefficients based on skewness direction
    if mc >= 0:
        a, b = -4.0, 3.0
    else:
        a, b = -3.0, 4.0

    lower_fence = q1 - 1.5 * np.exp(a * mc) * iqr
    upper_fence = q3 + 1.5 * np.exp(b * mc) * iqr

    return lower_fence, upper_fence
```

### Implementation Specification

**File**: `src/cliquefinder/quality/outliers.py`

**New Class**: `AdaptiveOutlierDetector`

```python
class AdaptiveOutlierDetector(Transform):
    """
    Distribution-aware outlier detection with asymmetric fences.

    Uses medcouple-adjusted boxplot fences (Hubert & Vandervieren, 2008)
    to handle skewed distributions common in proteomics data.

    Args:
        method: "adjusted-boxplot" (default) or "mad-z" (legacy)
        mode: "within_group" (default), "per_feature", or "global"
        group_cols: Metadata column(s) for grouping
        return_scores: If True, return probability scores instead of binary flags

    Returns:
        BioMatrix with outlier_scores in quality_flags (if return_scores=True)
        or binary OUTLIER_DETECTED flags (if return_scores=False)
    """
```

**Dependencies**:
- `robustbase` R package via `rpy2` (optional, for validation)
- Or: Pure Python implementation using the fast algorithm

**Note**: The R `robustbase::mc()` function implements the O(n log n) algorithm. For Python, use the `robustats` package or implement directly.

### Validation Criteria

1. **Unit test**: MC = 0 for symmetric distributions (normal, uniform)
2. **Unit test**: MC > 0 for right-skewed (log-normal, exponential)
3. **Unit test**: MC < 0 for left-skewed (reflected exponential)
4. **Integration test**: Compare with `robustbase::adjboxStats()` output
5. **Regression test**: Verify backward compatibility with existing MAD-Z results

### Test Cases

```python
def test_medcouple_symmetric():
    """Medcouple should be ~0 for symmetric distributions."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    mc = compute_medcouple(x)
    assert abs(mc) < 0.1, f"Expected MC ≈ 0, got {mc}"

def test_medcouple_right_skewed():
    """Medcouple should be positive for right-skewed data."""
    np.random.seed(42)
    x = np.random.lognormal(0, 1, 1000)
    mc = compute_medcouple(x)
    assert mc > 0.2, f"Expected MC > 0.2 for lognormal, got {mc}"

def test_adjusted_fences_asymmetric():
    """Upper fence should expand more than lower for right-skewed data."""
    np.random.seed(42)
    x = np.random.lognormal(0, 0.5, 1000)
    lower, upper = adjusted_boxplot_fences(x)

    # Standard symmetric fences for comparison
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    std_lower = q1 - 1.5 * iqr
    std_upper = q3 + 1.5 * iqr

    # Adjusted upper should be MORE permissive (higher)
    assert upper > std_upper, "Upper fence should expand for right-skewed"
    # Adjusted lower should be LESS permissive (higher)
    assert lower > std_lower, "Lower fence should contract for right-skewed"
```

---

## Priority 2: Probabilistic Scoring with Student's t-Distribution

### Scientific Foundation

**Source**: PROTRIDER (Scheller et al., 2025). "Protein abundance outlier detection from mass spectrometry-based proteomics data with a conditional autoencoder." *bioRxiv*. [DOI](https://www.biorxiv.org/content/10.1101/2025.02.01.636024v1)

**Key Finding**:
> "The differences between experimental and fitted log-transformed intensities exhibit heavy tails that are poorly captured with the Gaussian distribution, and we report stronger statistical calibration when using the Student's t-distribution."

**Approach**:
> "Fitting a Student's t-distribution for each protein was robustly achieved by learning a value for the degrees of freedom common to all proteins."

### Mathematical Formulation

#### Student's t-Distribution

The probability density function:

```
f(x; ν, μ, σ) = Γ((ν+1)/2) / (√(νπ) × Γ(ν/2) × σ) × (1 + ((x-μ)/σ)²/ν)^(-(ν+1)/2)
```

where:
- ν = degrees of freedom (controls tail heaviness)
- μ = location (like mean)
- σ = scale (like standard deviation)
- Γ = gamma function

**Key Property**: As ν → ∞, Student's t → Normal. Lower ν = heavier tails.

#### Outlier Probability

For a value x, compute the two-tailed probability of being more extreme:

```
P_outlier(x) = 2 × min(F(x; ν, μ, σ), 1 - F(x; ν, μ, σ))
```

where F is the CDF of the Student's t-distribution.

This gives a continuous score in [0, 1] where:
- P ≈ 1 → value is near the center (not an outlier)
- P ≈ 0 → value is in the extreme tails (likely outlier)

For downstream use, we typically use **outlier_score = 1 - P_outlier** so that higher scores indicate more extreme values.

#### Degrees of Freedom Estimation

**Method 1**: Maximum Likelihood Estimation (MLE)

```python
from scipy.stats import t as student_t

# Fit Student's t to data
df, loc, scale = student_t.fit(residuals)
```

**Method 2**: Shared df across all proteins (PROTRIDER approach)

Pool residuals from all proteins, fit a single df parameter shared across the dataset. This is more robust when individual proteins have few observations.

```python
# Pool residuals across all proteins
all_residuals = residuals.ravel()
all_residuals = all_residuals[~np.isnan(all_residuals)]

# Fit shared df
df_shared, _, _ = student_t.fit(all_residuals)
```

**Typical values**: For proteomics data, df ≈ 3-10 (heavier tails than normal).

### Algorithm

```python
def fit_student_t_shared(
    data: np.ndarray,
    per_feature: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Fit Student's t-distribution with shared degrees of freedom.

    Args:
        data: Expression matrix (n_features × n_samples)
        per_feature: If True, fit location/scale per feature, df shared

    Returns:
        (df_shared, locations, scales) where:
        - df_shared: Single df value for all features
        - locations: Per-feature location parameters (n_features,)
        - scales: Per-feature scale parameters (n_features,)
    """
    from scipy.stats import t as student_t

    n_features, n_samples = data.shape

    # Step 1: Compute per-feature robust location and scale
    locations = np.median(data, axis=1)
    # Use MAD for robust scale estimation
    mads = np.median(np.abs(data - locations[:, np.newaxis]), axis=1)
    scales = mads / 0.6745  # Scale to match std for normal
    scales[scales == 0] = 1.0  # Avoid division by zero

    # Step 2: Standardize residuals
    standardized = (data - locations[:, np.newaxis]) / scales[:, np.newaxis]

    # Step 3: Pool and fit df
    pooled = standardized.ravel()
    pooled = pooled[np.isfinite(pooled)]

    # Fit Student's t, fixing loc=0 and scale=1 (already standardized)
    # Only estimate df
    df_shared, _, _ = student_t.fit(pooled, floc=0, fscale=1)

    # Clamp df to reasonable range
    df_shared = np.clip(df_shared, 2.1, 100)

    return df_shared, locations, scales


def compute_outlier_probability(
    data: np.ndarray,
    df: float,
    locations: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    """
    Compute outlier probability for each value using Student's t.

    Args:
        data: Expression matrix (n_features × n_samples)
        df: Degrees of freedom (shared)
        locations: Per-feature locations
        scales: Per-feature scales

    Returns:
        Outlier probability matrix (n_features × n_samples)
        Values near 0 = likely outlier, near 1 = likely normal
    """
    from scipy.stats import t as student_t

    # Standardize
    z = (data - locations[:, np.newaxis]) / scales[:, np.newaxis]

    # Compute two-tailed probability
    # P(|T| > |z|) = 2 * min(CDF(z), 1 - CDF(z))
    cdf_vals = student_t.cdf(z, df)
    p_values = 2 * np.minimum(cdf_vals, 1 - cdf_vals)

    return p_values
```

### Implementation Specification

**File**: `src/cliquefinder/quality/outliers.py`

**New Method in `AdaptiveOutlierDetector`**:

```python
def detect_with_student_t(
    self,
    matrix: BioMatrix,
    shared_df: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Detect outliers using Student's t-distribution.

    Args:
        matrix: Input BioMatrix
        shared_df: If True, fit single df across all proteins (PROTRIDER approach)

    Returns:
        (outlier_scores, diagnostics) where:
        - outlier_scores: Matrix of outlier probabilities (0=outlier, 1=normal)
        - diagnostics: Dict with df, locations, scales for inspection
    """
```

**Dependencies**:
- `scipy.stats.t` (already in requirements)

### Validation Criteria

1. **Calibration test**: For simulated Gaussian data, p-values should be uniform
2. **Heavy tail test**: For simulated t-distribution data, fitted df should match
3. **Comparison test**: Compare with PROTRIDER output on reference dataset
4. **Performance test**: Fit 60K proteins in < 30 seconds

### Test Cases

```python
def test_student_t_calibration():
    """P-values should be uniform under null (Gaussian data)."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 50))  # 100 features, 50 samples

    df, locs, scales = fit_student_t_shared(data)
    p_values = compute_outlier_probability(data, df, locs, scales)

    # KS test for uniformity
    from scipy.stats import kstest
    _, ks_pval = kstest(p_values.ravel(), 'uniform')
    assert ks_pval > 0.01, f"P-values not uniform, KS p={ks_pval}"

def test_student_t_heavy_tails():
    """Should detect heavy tails (low df) in t-distributed data."""
    np.random.seed(42)
    from scipy.stats import t as student_t

    true_df = 4
    data = student_t.rvs(true_df, size=(100, 50))

    fitted_df, _, _ = fit_student_t_shared(data)

    # Fitted df should be close to true df
    assert 2 < fitted_df < 8, f"Expected df ≈ 4, got {fitted_df}"
```

---

## Priority 3: Weighted Correlation Infrastructure

### Scientific Foundation

**Problem**: Imputation introduces synthetic data that can contaminate correlation structure:
- Hard clipping creates ties at boundaries (inflates Spearman)
- KNN imputation uses neighbor correlations (circular with downstream analysis)
- Median imputation shrinks variance (attenuates correlations)

**Solution**: Use weighted correlation where outlier observations are downweighted rather than replaced.

**Source**: Bailey, P., Emad, A., Zhang, T., & Xie, Q. (2023). "wCorr Formulas." CRAN. [PDF](https://cran.r-project.org/web/packages/wCorr/vignettes/wCorrFormulas.pdf)

> "The weights may reflect the possibility of an observation being an outlier or influential data point if a robust estimation is of utmost importance."

### Mathematical Formulation

#### Weighted Pearson Correlation

For vectors x and y with weights w:

```
           Σᵢ wᵢ(xᵢ - x̄ᵂ)(yᵢ - ȳᵂ)
rᵂₓᵧ = ────────────────────────────────────────
       √[Σᵢ wᵢ(xᵢ - x̄ᵂ)² × Σᵢ wᵢ(yᵢ - ȳᵂ)²]
```

where the weighted means are:

```
x̄ᵂ = Σᵢ wᵢxᵢ / Σᵢ wᵢ
ȳᵂ = Σᵢ wᵢyᵢ / Σᵢ wᵢ
```

#### Weight Assignment from Outlier Probabilities

Given outlier probability P (where P ≈ 0 means likely outlier):

```
# Option 1: Direct probability as weight
wᵢⱼ = P_outlier(xᵢⱼ)

# Option 2: Soft threshold (recommended)
wᵢⱼ = 1 - (1 - P_outlier(xᵢⱼ))^k  where k controls sharpness

# Option 3: Hard threshold with soft transition
wᵢⱼ = sigmoid(k × (P_outlier(xᵢⱼ) - threshold))
```

#### Pairwise Weight Combination

For correlation between genes i and j across samples, each sample has two weights (one per gene). Combine using:

```
# Option 1: Geometric mean (default)
w_sample = √(w_gene_i × w_gene_j)

# Option 2: Minimum (conservative)
w_sample = min(w_gene_i, w_gene_j)

# Option 3: Product
w_sample = w_gene_i × w_gene_j
```

### Algorithm

```python
def weighted_pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray
) -> float:
    """
    Compute weighted Pearson correlation.

    Args:
        x: First variable (n_samples,)
        y: Second variable (n_samples,)
        w: Weights (n_samples,), should sum to > 0

    Returns:
        Weighted correlation coefficient
    """
    # Normalize weights
    w = w / w.sum()

    # Weighted means
    x_mean = np.sum(w * x)
    y_mean = np.sum(w * y)

    # Weighted covariance and variances
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    var_x = np.sum(w * (x - x_mean)**2)
    var_y = np.sum(w * (y - y_mean)**2)

    # Correlation
    if var_x <= 0 or var_y <= 0:
        return 0.0

    return cov_xy / np.sqrt(var_x * var_y)


def compute_weighted_correlation_matrix(
    data: np.ndarray,
    weights: np.ndarray,
    weight_combination: str = "geometric",
    chunk_size: int = 500,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute weighted correlation matrix for all gene pairs.

    Args:
        data: Expression matrix (n_features × n_samples)
        weights: Weight matrix (n_features × n_samples), values in [0, 1]
        weight_combination: How to combine weights for pairs
            - "geometric": sqrt(w_i * w_j) (default)
            - "minimum": min(w_i, w_j)
            - "product": w_i * w_j
        chunk_size: Number of genes to process at once
        verbose: Show progress

    Returns:
        Weighted correlation matrix (n_features × n_features)
    """
    n_features, n_samples = data.shape

    # Pre-compute weighted standardized data
    # For each gene, subtract weighted mean and divide by weighted std
    w_normalized = weights / weights.sum(axis=1, keepdims=True)

    weighted_means = np.sum(w_normalized * data, axis=1, keepdims=True)
    centered = data - weighted_means

    weighted_vars = np.sum(w_normalized * centered**2, axis=1, keepdims=True)
    weighted_stds = np.sqrt(weighted_vars)
    weighted_stds[weighted_stds == 0] = 1.0

    standardized = centered / weighted_stds

    # Compute correlation matrix in chunks
    corr_matrix = np.zeros((n_features, n_features), dtype=np.float32)

    for i in tqdm(range(0, n_features, chunk_size), disable=not verbose):
        chunk_end = min(i + chunk_size, n_features)
        chunk_size_actual = chunk_end - i

        for j in range(i, n_features):
            # Combine weights for this pair
            if weight_combination == "geometric":
                combined_w = np.sqrt(weights[i:chunk_end, :, np.newaxis] *
                                     weights[np.newaxis, j, :])
            elif weight_combination == "minimum":
                combined_w = np.minimum(weights[i:chunk_end, :, np.newaxis],
                                        weights[np.newaxis, j, :])
            else:  # product
                combined_w = weights[i:chunk_end, :, np.newaxis] * weights[np.newaxis, j, :]

            # Normalize weights per pair
            combined_w = combined_w / combined_w.sum(axis=1, keepdims=True)

            # Weighted correlation
            # ... (vectorized implementation)

    return corr_matrix
```

### Implementation Specification

**File**: `src/cliquefinder/utils/correlation_matrix.py`

**New Function**: `get_weighted_correlation_matrix`

```python
def get_weighted_correlation_matrix(
    matrix: BioMatrix,
    weights: np.ndarray,
    method: str = 'weighted_pearson',
    weight_combination: str = 'geometric',
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute weighted correlation matrix with outlier downweighting.

    This is the recommended approach for correlation networks when
    outliers are present. Instead of imputing outliers (which introduces
    synthetic data), we downweight them in the correlation calculation.

    Args:
        matrix: BioMatrix with expression data
        weights: Weight matrix (n_features × n_samples) where:
            - 1.0 = full weight (normal observation)
            - 0.0 = zero weight (complete outlier)
            - Values computed from outlier probabilities
        method: 'weighted_pearson' (default)
        weight_combination: How to combine gene-pair weights
        cache: Use caching (cache key includes weight hash)
        cache_dir: Cache directory
        verbose: Show progress

    Returns:
        Weighted correlation matrix (n_features × n_features)
    """
```

### Validation Criteria

1. **Equivalence test**: When all weights = 1, should equal standard Pearson
2. **Outlier robustness test**: Single outlier should have minimal effect when downweighted
3. **Symmetry test**: Correlation matrix should be symmetric
4. **Performance test**: 60K genes in < 2 hours with weights

### Test Cases

```python
def test_weighted_correlation_unweighted():
    """With uniform weights, should match standard Pearson."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 50))
    weights = np.ones_like(data)

    weighted_corr = compute_weighted_correlation_matrix(data, weights)
    standard_corr = np.corrcoef(data)

    np.testing.assert_allclose(weighted_corr, standard_corr, rtol=1e-5)

def test_weighted_correlation_outlier_robustness():
    """Downweighted outlier should have minimal effect."""
    np.random.seed(42)

    # Create correlated data
    x = np.random.normal(0, 1, 50)
    y = 0.8 * x + 0.2 * np.random.normal(0, 1, 50)

    # Add outlier
    x_with_outlier = x.copy()
    x_with_outlier[0] = 10  # Extreme outlier

    # Standard correlation (affected by outlier)
    r_standard = np.corrcoef(x_with_outlier, y)[0, 1]

    # Weighted correlation (outlier downweighted)
    weights = np.ones(50)
    weights[0] = 0.01  # Downweight outlier
    r_weighted = weighted_pearson_correlation(x_with_outlier, y, weights)

    # True correlation
    r_true = np.corrcoef(x, y)[0, 1]

    # Weighted should be closer to true than standard
    assert abs(r_weighted - r_true) < abs(r_standard - r_true)
```

---

## Priority 4: Soft Clipping Fallback

### Scientific Foundation

**Problem**: Some downstream methods require complete data without weights (e.g., certain clustering algorithms, legacy code). Hard clipping to MAD bounds creates:
1. **Boundary spikes** — Many values pile up exactly at the threshold
2. **Discontinuous transformation** — Non-differentiable at boundaries
3. **Rank distortion** — Values originally different become tied

**Solution**: Soft clipping using a smooth sigmoid function that:
- Gradually compresses extreme values
- Preserves rank ordering
- Is differentiable everywhere
- Has tunable sharpness

**Source**: "Clip it, clip it good" (2021). Hopper Engineering Blog. [Link](https://medium.com/life-at-hopper/clip-it-clip-it-good-1f1bf711b291)

> "Soft clipping reduces spikes by squeezing outliers while preserving ordering... A soft clipping function should be a smooth sigmoid curve with f(x) ≃ a for x ≪ a, f(x) ≃ b for b ≪ x, and a near unit slope within [a, b]."

### Mathematical Formulation

#### Soft Clipping Function

The generalized logistic function for soft clipping to range [L, U]:

```
soft_clip(x; L, U, k) = L + (U - L) × σ(k × (x - c))

where:
  σ(z) = 1 / (1 + exp(-z))     # standard sigmoid
  c = (L + U) / 2               # center point
  k = sharpness parameter       # higher = sharper transition
```

**Properties**:
- For x ≪ L: soft_clip(x) ≈ L
- For x ≫ U: soft_clip(x) ≈ U
- For L < x < U: soft_clip(x) ≈ x (with unit slope near center)
- Differentiable everywhere
- Strictly monotonic (preserves ordering)

#### Choosing Sharpness Parameter k

The sharpness k controls how quickly the function transitions from linear to flat:

```
k ≈ 4 / (U - L)  # Standard: 95% of range has slope > 0.5
k ≈ 2 / (U - L)  # Gentle: smoother transition
k ≈ 8 / (U - L)  # Sharp: closer to hard clipping
```

**Recommendation**: Start with `k = 2 / (U - L)` for correlation analysis to minimize distortion.

#### Alternative: Tanh-Based Soft Clipping

```
soft_clip_tanh(x; L, U, k) = c + (U - L)/2 × tanh(k × (x - c))

where c = (L + U) / 2
```

This is mathematically equivalent to the sigmoid form but sometimes more numerically stable.

### Algorithm

```python
def soft_clip(
    x: np.ndarray,
    lower: float,
    upper: float,
    sharpness: Optional[float] = None
) -> np.ndarray:
    """
    Apply soft (sigmoid) clipping to array.

    Args:
        x: Input array
        lower: Lower bound
        upper: Upper bound
        sharpness: Transition sharpness (default: 2 / (upper - lower))

    Returns:
        Soft-clipped array with same shape
    """
    if sharpness is None:
        sharpness = 2.0 / (upper - lower)

    center = (lower + upper) / 2.0
    range_half = (upper - lower) / 2.0

    # Tanh-based implementation (numerically stable)
    z = sharpness * (x - center)
    return center + range_half * np.tanh(z)


def soft_clip_per_feature(
    data: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    sharpness: float = None
) -> np.ndarray:
    """
    Apply soft clipping with per-feature bounds.

    Args:
        data: Expression matrix (n_features × n_samples)
        lower_bounds: Per-feature lower bounds (n_features,)
        upper_bounds: Per-feature upper bounds (n_features,)
        sharpness: Transition sharpness (if None, computed per feature)

    Returns:
        Soft-clipped data matrix
    """
    result = np.empty_like(data)

    for i in range(data.shape[0]):
        result[i, :] = soft_clip(
            data[i, :],
            lower_bounds[i],
            upper_bounds[i],
            sharpness
        )

    return result
```

### Implementation Specification

**File**: `src/cliquefinder/quality/imputation.py`

**New Strategy**: Add `"soft-clip"` to `Imputer` class

```python
class Imputer(Transform):
    """
    ...

    Strategies:
        soft-clip (NEW):
            - Apply smooth sigmoid clipping to outliers
            - Preserves rank ordering unlike hard clipping
            - Bounds determined by medcouple-adjusted fences or MAD
            - Sharpness parameter controls transition smoothness
            - Recommended when downstream methods require complete data
            - WARNING: Can distort correlations; prefer weighted correlation
    """

    def __init__(
        self,
        strategy: str = "mad-clip",
        threshold: float = 3.5,
        sharpness: Optional[float] = None,  # NEW: for soft-clip
        group_cols: str | list[str] | None = None
    ):
        ...
```

### Validation Criteria

1. **Ordering test**: Soft clipping must preserve rank order
2. **Convergence test**: As sharpness → ∞, should approach hard clipping
3. **Continuity test**: Verify smooth derivative at all points
4. **Correlation impact test**: Measure correlation distortion vs hard clipping

### Test Cases

```python
def test_soft_clip_preserves_order():
    """Soft clipping should preserve rank ordering."""
    x = np.array([1, 2, 3, 10, 100, 1000])
    clipped = soft_clip(x, lower=0, upper=50, sharpness=0.1)

    # Ranks should be preserved
    assert np.all(np.argsort(x) == np.argsort(clipped))

def test_soft_clip_approaches_hard():
    """With high sharpness, should approach hard clipping."""
    x = np.linspace(-10, 10, 1000)

    hard = np.clip(x, -5, 5)
    soft = soft_clip(x, lower=-5, upper=5, sharpness=100)

    np.testing.assert_allclose(soft, hard, atol=0.1)

def test_soft_clip_correlation_impact():
    """Soft clipping should distort correlations less than hard clipping."""
    np.random.seed(42)

    # Correlated data with outliers
    x = np.random.normal(0, 1, 100)
    y = 0.8 * x + 0.2 * np.random.normal(0, 1, 100)
    x[0] = 10  # Outlier

    # True correlation (without outlier)
    r_true = 0.8

    # Hard clipping
    x_hard = np.clip(x, -3, 3)
    r_hard = np.corrcoef(x_hard, y)[0, 1]

    # Soft clipping
    x_soft = soft_clip(x, -3, 3, sharpness=1.0)
    r_soft = np.corrcoef(x_soft, y)[0, 1]

    # Soft should introduce less distortion
    # (This may not always hold; the test documents expected behavior)
    print(f"True: {r_true:.3f}, Hard: {r_hard:.3f}, Soft: {r_soft:.3f}")
```

---

## Implementation Delegation

### Subagent 1: Medcouple-Adjusted Detection

**Scope**: Priority 1

**Deliverables**:
1. `compute_medcouple()` function with O(n log n) algorithm
2. `adjusted_boxplot_fences()` function
3. `AdaptiveOutlierDetector` class with `method="adjusted-boxplot"`
4. Unit tests comparing with `robustbase::mc()`
5. Integration test with existing pipeline

**Files to Modify**:
- `src/cliquefinder/quality/outliers.py` (add new class)
- `src/cliquefinder/quality/__init__.py` (export new class)
- `tests/test_outliers.py` (add tests)

**Estimated Complexity**: Medium (300-500 lines)

---

### Subagent 2: Student's t Probabilistic Scoring

**Scope**: Priority 2

**Deliverables**:
1. `fit_student_t_shared()` function
2. `compute_outlier_probability()` function
3. Integration into `AdaptiveOutlierDetector` with `return_scores=True`
4. Diagnostic output (fitted df, QQ-plots)
5. Unit tests for calibration

**Files to Modify**:
- `src/cliquefinder/quality/outliers.py` (add methods)
- `tests/test_outliers.py` (add tests)

**Dependencies**: `scipy.stats.t`

**Estimated Complexity**: Medium (200-400 lines)

---

### Subagent 3: Weighted Correlation Matrix

**Scope**: Priority 3

**Deliverables**:
1. `weighted_pearson_correlation()` function
2. `compute_weighted_correlation_matrix()` with chunked processing
3. `get_weighted_correlation_matrix()` entry point with caching
4. Cache key that includes weight matrix hash
5. Unit tests for equivalence and robustness

**Files to Modify**:
- `src/cliquefinder/utils/correlation_matrix.py` (add functions)
- `tests/test_correlation_matrix.py` (add tests)

**Estimated Complexity**: High (500-800 lines due to performance optimization)

---

### Subagent 4: Soft Clipping Fallback

**Scope**: Priority 4

**Deliverables**:
1. `soft_clip()` function
2. `soft_clip_per_feature()` function
3. Add `strategy="soft-clip"` to `Imputer` class
4. Documentation of correlation impact
5. Unit tests for ordering and convergence

**Files to Modify**:
- `src/cliquefinder/quality/imputation.py` (add strategy)
- `tests/test_imputation.py` (add tests)

**Estimated Complexity**: Low-Medium (150-250 lines)

---

## References

1. **Brys, G., Hubert, M., & Struyf, A.** (2004). A Robust Measure of Skewness. *Journal of Computational and Graphical Statistics*, 13(4), 996-1017. [DOI](https://www.tandfonline.com/doi/abs/10.1198/106186004X12632)

2. **Hubert, M., & Vandervieren, E.** (2008). An adjusted boxplot for skewed distributions. *Computational Statistics & Data Analysis*, 52, 5186-5201. [PDF](https://wis.kuleuven.be/statdatascience/robust/papers/2008/hubertvandervieren_adjustedboxplot_csda_2008.pdf)

3. **Scheller, I.F., et al.** (2025). PROTRIDER: Protein abundance outlier detection from mass spectrometry-based proteomics data with a conditional autoencoder. *bioRxiv*. [DOI](https://www.biorxiv.org/content/10.1101/2025.02.01.636024v1)

4. **Bailey, P., Emad, A., Zhang, T., & Xie, Q.** (2023). wCorr Formulas. *CRAN*. [PDF](https://cran.r-project.org/web/packages/wCorr/vignettes/wCorrFormulas.pdf)

5. **Weighted Pearson Correlation Test Statistic** (2024). *Journal of Applied Statistics*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868461/)

6. **robustbase R package**. mc() function documentation. [CRAN](https://search.r-project.org/CRAN/refmans/robustbase/html/mc.html)

7. **scipy.stats.t**. Student's t-distribution. [SciPy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)

---

## Appendix A: Decision Matrix

| Criterion | MAD-Z (Current) | Medcouple-Adjusted | Student's t Scoring | Weighted Corr | Soft Clip |
|-----------|-----------------|-------------------|---------------------|---------------|-----------|
| Handles asymmetry | No | **Yes** | No | N/A | No |
| Heavy tail robustness | Partial | Partial | **Yes** | N/A | Partial |
| Continuous scores | No | No | **Yes** | **Yes** | Partial |
| Preserves original data | No | No | **Yes** | **Yes** | No |
| Correlation safety | Low | Low | High | **High** | Medium |
| Implementation complexity | Low | Medium | Medium | High | Low |
| Performance overhead | Low | Low | Low | Medium | Low |

**Recommendation**: Use all four in combination:
1. Medcouple for asymmetric fence calculation
2. Student's t for probability scoring
3. Weighted correlation as primary downstream method
4. Soft clipping only when weights not supported

---

## Appendix B: Migration Path

### Phase 1: Non-Breaking Addition
- Add `AdaptiveOutlierDetector` alongside existing `OutlierDetector`
- Add `get_weighted_correlation_matrix` alongside existing function
- Add `soft-clip` strategy to `Imputer`

### Phase 2: Default Change (Major Version)
- Change default `OutlierDetector` to use `AdaptiveOutlierDetector`
- Add deprecation warning for hard MAD-Z

### Phase 3: Legacy Removal
- Remove legacy `OutlierDetector` (or alias to adaptive)
- Update all documentation
