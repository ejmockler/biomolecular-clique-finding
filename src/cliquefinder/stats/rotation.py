"""
ROAST: Rotation Gene Set Tests for Complex Experiments.

This module implements the ROAST methodology (Wu et al., 2010) for rotation-based
gene set testing, providing a self-contained alternative to competitive permutation
tests that preserves inter-gene correlation structure.

Mathematical Foundation
=======================

Linear Model Specification
--------------------------
For each gene g in the dataset:

    y_g = X·α_g + ε_g

where:
    - y_g ∈ ℝⁿ is the expression vector across n samples
    - X ∈ ℝⁿˣᵖ is the design matrix (full column rank p)
    - α_g ∈ ℝᵖ are the regression coefficients for gene g
    - ε_g ~ N(0, σ²_g·W⁻¹) with W as a positive-definite weight matrix

The contrast of interest is β_g = c'·α_g where c ∈ ℝᵖ is the contrast vector.

QR Decomposition for Residual Extraction
----------------------------------------
To isolate the degrees of freedom relevant to the hypothesis test:

1. Reparameterize so the null hypothesis concerns the last coefficient:
   Construct invertible C ∈ ℝᵖˣᵖ with last column = c

2. Perform FULL QR decomposition: X = Q·R where Q ∈ ℝⁿˣⁿ

3. Extract Q₂ = Q[:, p-1:] (last d+1 columns, where d = n - p)

4. Project each expression vector: u_g = Q₂'·y_g ∈ ℝᵈ⁺¹

Under H₀, elements u_g[0], ..., u_g[d] are independent N(0, σ²_g).

Rotation Mechanism
------------------
The key insight: under the null hypothesis, the residual vector u_g lies on a
(d+1)-sphere of radius ρ_g = ||u_g||. Rotating u_g to ANY other point on this
sphere produces an equally valid null sample.

For each rotation r ∈ {1, ..., B}:

1. Generate random direction vector: v ~ N(0, I_{d+1})
2. Normalize: r = v / ||v||
3. For all genes in set S simultaneously, compute rotated first element:
   u*_g[0] = ⟨u_g, r⟩ = u_g' · r
4. Compute rotated residual variance:
   s*²_g = (ρ²_g - u*_g[0]²) / d
5. Recompute moderated t-statistics and gene set statistic

Empirical Bayes Moderation
--------------------------
Following Smyth (2004), gene-wise variances are shrunk toward a common prior:

    σ²_g ~ Inverse-χ²(d₀, s₀²)

Posterior variance: s²_post = (d₀·s₀² + d·s²_g) / (d₀ + d)
Moderated t: t̃_g = β̂_g / (s_post · √v) ~ t_{d₀+d}

Gene Set Statistics
-------------------
Let z_g = Φ⁻¹(F_{t_{d₀+d}}(t̃_g)) be the z-score transformation of moderated t.
With optional weights a_g and A = Σ|a_g|:

- mean:      T = (Σ a_g·z_g) / A           [all genes move together]
- floormean: T = (Σ a_g·max(|z_g|, √q)) / A  [dampens noise, q = median(χ²₁)]
- mean50:    T = mean of top 50% |z_g|     [half of genes are DE]
- msq:       T = (Σ |a_g|·z²_g) / A        [sparse signal, any direction]

The msq statistic is CRITICAL for detecting bidirectional regulation.

P-Value Computation
-------------------
Exact Monte Carlo p-value:

    p = (b + 1) / (B + 1)

where b = count of rotations with T* ≥ T_observed.

Computational Optimization
--------------------------
1. QR decomposition: O(np²) - computed ONCE per contrast
2. Rotation generation: O(B·d) - vectorized on GPU
3. Rotation application: O(B·m·d) where m = genes in set - fully batched
4. Set statistics: O(B·m) - vectorized

Total complexity: O(np² + B·m·d) instead of O(B·n·p²) for naive approach.

References
----------
- Wu et al. (2010) "ROAST: rotation gene set tests for complex microarray
  experiments" Bioinformatics 26(17):2176-82
- Smyth (2004) "Linear models and empirical Bayes methods for assessing
  differential expression in microarray experiments" Stat Appl Genet Mol Biol
- Langsrud (2005) "Rotation tests" Statistics and Computing 15:53-60

Hardware Acceleration
---------------------
This implementation supports:
- CPU: NumPy with BLAS/LAPACK (Apple Accelerate on macOS)
- GPU: MLX for Apple Silicon (Metal Performance Shaders)
- Future: CUDA via CuPy (architecture supports drop-in replacement)

The design follows the existing permutation_gpu.py patterns for seamless
integration with the cliquefinder infrastructure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.special import ndtri  # Inverse normal CDF (Φ⁻¹)

# Optional MLX import for GPU acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore


# =============================================================================
# Type Definitions and Protocols
# =============================================================================

@runtime_checkable
class WeightedFeatureSet(Protocol):
    """Protocol for feature sets with optional per-gene weights."""

    @property
    def id(self) -> str:
        """Unique identifier for this feature set."""
        ...

    @property
    def feature_ids(self) -> list[str]:
        """List of feature identifiers."""
        ...

    @property
    def weights(self) -> NDArray[np.float64] | None:
        """Optional weights for each feature (same order as feature_ids)."""
        ...

    @property
    def size(self) -> int:
        """Number of features."""
        ...


# =============================================================================
# Enums and Configuration
# =============================================================================

class SetStatistic(Enum):
    """Gene set statistics for aggregating gene-level scores.

    The choice of statistic affects sensitivity to different patterns:

    - MEAN: Sensitive when majority of genes change in same direction.
            Best for coherent, coordinated regulation.

    - FLOORMEAN: Compromise between mean and msq. Floors small effects
                 at sqrt(median χ²₁) ≈ 0.67 to dampen noise while
                 preserving signal. Good general-purpose choice.

    - MEAN50: Mean of top 50% most significant genes. Robust when
              approximately half the genes are truly DE.

    - MSQ: Mean of squared z-scores. Direction-agnostic - detects
           regulation regardless of up/down direction. CRITICAL for
           TFs that both activate AND repress different targets.

    - MIXED: Combines MEAN and MSQ: (mean² + msq) / 2. Balances
             directional coherence with magnitude detection.
    """

    MEAN = "mean"
    FLOORMEAN = "floormean"
    MEAN50 = "mean50"
    MSQ = "msq"
    MIXED = "mixed"


class Alternative(Enum):
    """Alternative hypothesis specification.

    - UP: Genes in set are up-regulated (positive effect)
    - DOWN: Genes in set are down-regulated (negative effect)
    - MIXED: Genes in set are DE in either direction (two-sided)
    """

    UP = "up"
    DOWN = "down"
    MIXED = "mixed"


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass(frozen=True)
class RotationPrecomputed:
    """Precomputed matrices for efficient rotation testing.

    These matrices are computed ONCE per contrast and reused for all
    gene sets and all rotations. This is the key to computational efficiency.

    Attributes:
        Q2: Residual space projection matrix (n_samples, df_residual + 1)
            Columns span the space orthogonal to nuisance parameters.

        df_residual: Residual degrees of freedom (n - p)

        contrast_name: Human-readable name for the contrast

        eb_d0: Empirical Bayes prior degrees of freedom (None = no shrinkage)

        eb_s0_sq: Empirical Bayes prior scale (prior variance)

        eb_df_total: Total df for moderated t-distribution (d0 + df_residual)

        design_rank: Rank of the design matrix (= p)

        n_samples: Number of samples in the analysis
    """

    Q2: NDArray[np.float64]
    df_residual: int
    contrast_name: str
    eb_d0: float | None = None
    eb_s0_sq: float | None = None
    eb_df_total: float | None = None
    design_rank: int = 0
    n_samples: int = 0

    def __post_init__(self):
        """Enforce true immutability for mutable fields."""
        # Make NDArray read-only: copy then set flags
        q2 = self.Q2.copy()
        q2.flags.writeable = False
        object.__setattr__(self, 'Q2', q2)

    @property
    def residual_dims(self) -> int:
        """Dimensionality of residual space (df_residual + 1)."""
        return self.df_residual + 1


@dataclass(frozen=True)
class GeneEffects:
    """Per-gene effects extracted via QR projection.

    For each gene g, contains the projected vector u_g = Q2' @ y_g.
    The first element u_g[0] relates to the contrast of interest,
    while remaining elements are residual effects.

    Attributes:
        U: Effects matrix (n_genes, df_residual + 1)
           Row g contains the projected effects for gene g.

        rho_sq: Squared norms ||u_g||² for each gene (n_genes,)
                Used for variance computation after rotation.

        gene_ids: Gene identifiers in same order as rows of U

        sample_variances: Per-gene residual variances (n_genes,)
                         s²_g = (ρ²_g - u_g[0]²) / df_residual

        moderated_variances: EB-shrunk variances (n_genes,) or None

        df_total: Total df for moderated t (scalar)
    """

    U: NDArray[np.float64]
    rho_sq: NDArray[np.float64]
    gene_ids: list[str]
    sample_variances: NDArray[np.float64]
    moderated_variances: NDArray[np.float64] | None = None
    df_total: float | None = None

    def __post_init__(self):
        """Enforce true immutability for mutable fields."""
        # Make NDArrays read-only
        for attr in ('U', 'rho_sq', 'sample_variances'):
            arr = getattr(self, attr).copy()
            arr.flags.writeable = False
            object.__setattr__(self, attr, arr)

        if self.moderated_variances is not None:
            mv = self.moderated_variances.copy()
            mv.flags.writeable = False
            object.__setattr__(self, 'moderated_variances', mv)

        # Convert list to tuple for true immutability
        if isinstance(self.gene_ids, list):
            object.__setattr__(self, 'gene_ids', tuple(self.gene_ids))

    @property
    def n_genes(self) -> int:
        return self.U.shape[0]

    @property
    def residual_dims(self) -> int:
        return self.U.shape[1]


@dataclass
class RotationResult:
    """Result of rotation-based gene set test.

    Contains observed statistics, null distributions, and p-values
    for multiple set statistics and alternative hypotheses.
    """

    feature_set_id: str
    n_genes: int
    n_genes_found: int
    gene_ids: list[str]

    # Observed statistics (one per set statistic type)
    observed_stats: dict[str, float]

    # Null distributions (set_stat -> alternatives -> array)
    null_distributions: dict[str, dict[str, NDArray[np.float64]]]

    # P-values (set_stat -> alternatives -> p-value)
    p_values: dict[str, dict[str, float]]

    # Active gene proportion (estimated fraction contributing to signal)
    active_proportion: dict[str, float]

    # Metadata
    n_rotations: int
    contrast_name: str

    def get_pvalue(
        self,
        statistic: SetStatistic | str = SetStatistic.MSQ,
        alternative: Alternative | str = Alternative.MIXED,
    ) -> float:
        """Get p-value for specific statistic and alternative."""
        stat_key = statistic.value if isinstance(statistic, SetStatistic) else statistic
        alt_key = alternative.value if isinstance(alternative, Alternative) else alternative
        return self.p_values.get(stat_key, {}).get(alt_key, np.nan)

    def to_dict(self) -> dict:
        """Convert to flat dictionary for DataFrame construction."""
        result = {
            'feature_set_id': self.feature_set_id,
            'n_genes': self.n_genes,
            'n_genes_found': self.n_genes_found,
            'n_rotations': self.n_rotations,
            'contrast': self.contrast_name,
        }

        # Add p-values for each combination
        for stat in SetStatistic:
            for alt in Alternative:
                key = f"pvalue_{stat.value}_{alt.value}"
                result[key] = self.get_pvalue(stat, alt)

        # Add observed statistics
        for stat, val in self.observed_stats.items():
            result[f"observed_{stat}"] = val

        # Add active proportions
        for alt, prop in self.active_proportion.items():
            result[f"active_proportion_{alt}"] = prop

        return result


# =============================================================================
# QR Decomposition and Residual Extraction
# =============================================================================

def _construct_c_matrix(contrast: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Construct the C reparameterization matrix with contrast as last column.

    The C matrix is an invertible p × p matrix where the last column equals
    the contrast vector c. The other columns form an orthonormal basis for
    the space orthogonal to c.

    This is constructed via modified Gram-Schmidt orthogonalization:
    1. Start with [I | c] where I is the (p-1) × (p-1) identity
    2. Orthogonalize each basis vector against c
    3. Normalize and place c as the last column

    Args:
        contrast: Contrast vector (p,) - must have at least one non-zero element

    Returns:
        C: Invertible matrix (p, p) with C[:, -1] = c / ||c||
    """
    p = len(contrast)

    # Normalize contrast
    c_norm = np.linalg.norm(contrast)
    if c_norm < 1e-10:
        raise ValueError("Contrast vector is zero or near-zero")
    c_unit = contrast / c_norm

    # Build orthonormal basis for the orthogonal complement of c
    # Start with standard basis vectors and use Gram-Schmidt
    C = np.zeros((p, p), dtype=np.float64)

    # Place normalized contrast as last column
    C[:, -1] = c_unit

    # Build orthogonal complement using Gram-Schmidt
    # We need p-1 vectors orthogonal to c
    col_idx = 0
    for i in range(p):
        # Start with i-th standard basis vector
        v = np.zeros(p)
        v[i] = 1.0

        # Subtract projection onto c
        v = v - np.dot(v, c_unit) * c_unit

        # Subtract projections onto previously found orthogonal vectors
        for j in range(col_idx):
            v = v - np.dot(v, C[:, j]) * C[:, j]

        # Check if we have a non-trivial vector
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            C[:, col_idx] = v / v_norm
            col_idx += 1

            if col_idx == p - 1:
                break

    if col_idx < p - 1:
        raise ValueError(f"Failed to construct orthogonal basis: got {col_idx}, need {p-1}")

    return C


def compute_rotation_matrices_general(
    design_matrix: NDArray[np.float64],
    contrast: NDArray[np.float64],
    sample_weights: NDArray[np.float64] | None = None,
    contrast_name: str = "custom_contrast",
) -> RotationPrecomputed:
    """
    Compute rotation matrices for arbitrary contrasts via C-matrix reparameterization.

    This implements the full ROAST methodology from Wu et al. (2010) for
    arbitrary linear contrasts in any experimental design.

    Mathematical Details:
    ---------------------
    1. Given design matrix X (n × p) and contrast vector c (p × 1)
    2. Construct C (p × p) invertible with last column = c
    3. Reparameterize: X* = X @ C^(-1), so the last coefficient = c'·α
    4. QR decompose X* = Q @ R (full, Q is n × n)
    5. Extract Q2 = Q[:, p-1:] which has d+1 columns (d = n - p)
       - First column of Q2 corresponds to the contrast direction
       - Remaining columns span the residual space

    This approach handles:
    - Simple two-group comparisons
    - Multi-group comparisons (ANOVA-style)
    - Interaction contrasts (2×2 factorial, etc.)
    - Complex contrasts with continuous covariates
    - Any hypothesis expressible as c'·α = 0

    Args:
        design_matrix: Full-rank design matrix X (n_samples, n_params)
        contrast: Contrast vector c (n_params,) defining c'·α
        sample_weights: Optional precision weights (n_samples,)
        contrast_name: Human-readable name for the contrast

    Returns:
        RotationPrecomputed with Q2 matrix and metadata

    Example:
        # 2×2 factorial: Sex × Disease interaction
        # Design: [Male_Case, Male_Ctrl, Female_Case, Female_Ctrl]
        # Contrast: (Male_Case - Male_Ctrl) - (Female_Case - Female_Ctrl)

        design = np.array([...])  # Cell-means design matrix
        contrast = np.array([1, -1, -1, 1])  # Interaction contrast

        precomputed = compute_rotation_matrices_general(design, contrast)
    """
    n_samples, n_params = design_matrix.shape

    if len(contrast) != n_params:
        raise ValueError(
            f"Contrast length ({len(contrast)}) must match design columns ({n_params})"
        )

    if n_samples <= n_params:
        raise ValueError(
            f"Insufficient samples: {n_samples} samples, {n_params} parameters"
        )

    df_residual = n_samples - n_params

    # Construct C matrix for reparameterization
    C = _construct_c_matrix(contrast)

    # Reparameterize design matrix: X* = X @ C^(-1)
    # The last column of X* corresponds to the contrast
    C_inv = np.linalg.inv(C)
    X_reparam = design_matrix @ C_inv

    # Apply weights if provided
    if sample_weights is not None:
        W_sqrt = np.sqrt(np.diag(sample_weights))
        X_reparam = W_sqrt @ X_reparam

    # Full QR decomposition: X* = Q @ R
    # Q is n × n orthogonal, R is n × p upper triangular
    Q, R = np.linalg.qr(X_reparam, mode='complete')

    # Extract Q2 = Q[:, p-1:] which has d+1 columns
    # - Q[:, p-1] is the direction corresponding to the contrast
    # - Q[:, p:] spans the residual space
    Q2 = Q[:, (n_params - 1):]  # Shape: (n_samples, df_residual + 1)

    # Sign correction: ensure Q2[:, 0] aligns with the contrast direction.
    #
    # QR decomposition is unique only up to sign flips of Q columns (the sign
    # of each Q column can be flipped along with the corresponding R row).
    # We resolve this ambiguity by projecting the contrast direction in sample
    # space (Xc = X @ c) onto Q2[:, 0].
    #
    # Mathematical justification: Q2 spans the complement of the first p-1
    # reparameterized columns (the reduced model without the contrast).  The
    # contrast signal Xc = X @ c is NOT in that reduced model's column space,
    # so Q2.T @ Xc is nonzero.  The sign of Q2[:, 0] @ Xc tells us whether
    # Q2[:, 0] points "with" or "against" the contrast — we canonicalize it
    # to point "with" the contrast so that positive test statistics correspond
    # to upregulation in the contrast direction.
    #
    # This replaces a correlation heuristic that was fragile for balanced
    # designs where the correlation could be near zero.
    Xc = design_matrix @ contrast
    proj = Q2[:, 0] @ Xc
    if abs(proj) < 1e-10:
        # Near-degenerate: contrast is almost entirely in the reduced model's
        # column space.  This should not happen for a valid contrast but we
        # warn and keep the current sign rather than making an arbitrary flip.
        warnings.warn(
            "Q2 sign correction: contrast projection onto Q2[:,0] is near-zero "
            f"({proj:.2e}). The contrast may be nearly redundant with the "
            "reduced model. Directional (UP/DOWN) p-values may be unreliable.",
            stacklevel=2,
        )
    elif proj < 0:
        Q2 = Q2.copy()
        Q2[:, 0] = -Q2[:, 0]

    return RotationPrecomputed(
        Q2=Q2,
        df_residual=df_residual,
        contrast_name=contrast_name,
        design_rank=n_params,
        n_samples=n_samples,
    )


def build_interaction_design(
    factor1_labels: NDArray | pd.Series,
    factor2_labels: NDArray | pd.Series,
    factor1_name: str = "factor1",
    factor2_name: str = "factor2",
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Build cell-means design matrix and interaction contrast for 2×2 factorial.

    Creates a design matrix with one column per group (no intercept) and
    a contrast vector for testing the interaction effect.

    Args:
        factor1_labels: Labels for first factor (e.g., sex: 'Male', 'Female')
        factor2_labels: Labels for second factor (e.g., disease: 'CASE', 'CTRL')
        factor1_name: Name of first factor (for contrast naming)
        factor2_name: Name of second factor (for contrast naming)

    Returns:
        Tuple of:
        - design_matrix: (n_samples, 4) cell-means design
        - group_names: List of group names matching columns
        - interaction_contrast: (4,) interaction contrast vector

    Example:
        >>> sex = np.array(['M', 'M', 'F', 'F', ...])
        >>> disease = np.array(['CASE', 'CTRL', 'CASE', 'CTRL', ...])
        >>> X, groups, contrast = build_interaction_design(sex, disease, 'sex', 'disease')
        >>> # groups = ['M_CASE', 'M_CTRL', 'F_CASE', 'F_CTRL']
        >>> # contrast = [1, -1, -1, 1]  # (M_CASE - M_CTRL) - (F_CASE - F_CTRL)
    """
    factor1 = np.asarray(factor1_labels)
    factor2 = np.asarray(factor2_labels)

    if len(factor1) != len(factor2):
        raise ValueError("Factor labels must have same length")

    n_samples = len(factor1)

    # Get unique levels
    levels1 = sorted(set(factor1))
    levels2 = sorted(set(factor2))

    if len(levels1) != 2:
        raise ValueError(f"Factor 1 must have exactly 2 levels, got {len(levels1)}: {levels1}")
    if len(levels2) != 2:
        raise ValueError(f"Factor 2 must have exactly 2 levels, got {len(levels2)}: {levels2}")

    # Create group labels
    # Convention: factor1_level1 + factor2_level1, factor1_level1 + factor2_level2, ...
    group_names = [
        f"{levels1[0]}_{levels2[0]}",  # e.g., Male_CASE
        f"{levels1[0]}_{levels2[1]}",  # e.g., Male_CTRL
        f"{levels1[1]}_{levels2[0]}",  # e.g., Female_CASE
        f"{levels1[1]}_{levels2[1]}",  # e.g., Female_CTRL
    ]

    # Build cell-means design matrix (no intercept)
    design = np.zeros((n_samples, 4), dtype=np.float64)
    for i in range(n_samples):
        f1, f2 = factor1[i], factor2[i]
        if f1 == levels1[0] and f2 == levels2[0]:
            design[i, 0] = 1.0
        elif f1 == levels1[0] and f2 == levels2[1]:
            design[i, 1] = 1.0
        elif f1 == levels1[1] and f2 == levels2[0]:
            design[i, 2] = 1.0
        elif f1 == levels1[1] and f2 == levels2[1]:
            design[i, 3] = 1.0
        else:
            raise ValueError(f"Unexpected factor combination: {f1}, {f2}")

    # Interaction contrast: (A1_B1 - A1_B2) - (A2_B1 - A2_B2)
    # = A1_B1 - A1_B2 - A2_B1 + A2_B2
    interaction_contrast = np.array([1.0, -1.0, -1.0, 1.0])

    return design, group_names, interaction_contrast


def compute_rotation_matrices(
    sample_condition: NDArray | pd.Series,
    conditions: list[str],
    contrast: tuple[str, str],
    sample_weights: NDArray[np.float64] | None = None,
) -> RotationPrecomputed:
    """
    Compute QR decomposition and extract residual projection matrix.

    This is the CRITICAL precomputation step. The resulting Q2 matrix
    projects gene expression vectors onto the residual subspace where
    rotations can be applied.

    .. warning::

        **Scope Limitation**: This implementation is designed for **simple
        two-group pairwise comparisons** (e.g., treatment vs control). For designs
        with >2 groups, continuous covariates, or interaction terms, the
        full C-matrix reparameterization from Wu et al. (2010) would be
        required. The current approach relies on dummy coding alignment
        and a sign correction heuristic that works correctly for two-group
        designs but would fail silently for complex contrasts.

    Mathematical Details:

    1. Construct design matrix X with dummy coding (reference = first level)

    2. Perform FULL QR decomposition: X = Q @ R
       - Q is n × n orthogonal matrix
       - R is n × p upper triangular

    3. Extract Q2 = Q[:, p-1:] which has d+1 columns (d = n - p)
       - These columns span the residual + contrast space
       - First column of Q2 relates to contrast of interest (for two-group)

    4. Apply sign correction to ensure Q2[:,0] aligns with contrast direction

    Args:
        sample_condition: Condition labels for each sample
        conditions: Ordered list of condition names (must be exactly 2)
        contrast: Tuple (test_condition, reference_condition)
        sample_weights: Optional precision weights for each sample

    Returns:
        RotationPrecomputed with Q2 matrix and metadata

    Raises:
        ValueError: If more than 2 conditions provided (not supported)

    Complexity: O(n × p²) - computed once per analysis
    """
    import statsmodels.api as sm

    n_samples = len(sample_condition)

    # Validate: exactly 2 conditions for this implementation
    if len(conditions) != 2:
        raise ValueError(
            f"This ROAST implementation supports exactly 2 conditions for "
            f"pairwise comparisons. Got {len(conditions)}: {conditions}. "
            f"For complex designs (>2 groups, covariates), implement the full "
            f"C-matrix reparameterization from Wu et al. (2010)."
        )

    # Build design matrix
    condition_cat = pd.Categorical(sample_condition, categories=conditions)
    df = pd.DataFrame({'condition': condition_cat})

    # Remove NaN
    valid_mask = ~df['condition'].isna()
    df = df[valid_mask]
    n_valid = len(df)

    if n_valid < 3:
        raise ValueError(f"Insufficient samples: {n_valid}")

    # Create design matrix with dummy coding
    X_df = pd.get_dummies(df['condition'], drop_first=True, dtype=float)
    X_df = sm.add_constant(X_df)
    X = X_df.values.astype(np.float64)

    n_samples_valid, n_params = X.shape
    df_residual = n_samples_valid - n_params

    if df_residual < 1:
        raise ValueError(f"Insufficient df: {n_samples_valid} - {n_params} = {df_residual}")

    # Build contrast vector in parameter space
    n_conditions = len(conditions)
    contrast_vec = np.zeros(n_conditions)

    idx1 = conditions.index(contrast[0])
    idx2 = conditions.index(contrast[1])
    contrast_vec[idx1] = 1.0
    contrast_vec[idx2] = -1.0

    # Transform to parameter space (dummy coding)
    # L[i, :] gives coefficients to compute mean of condition i
    L = np.zeros((n_conditions, n_params))
    L[:, 0] = 1.0  # Intercept
    for i in range(1, min(n_conditions, n_params)):
        L[i, i] = 1.0

    c = L.T @ contrast_vec  # Contrast in parameter space

    # Reparameterize so contrast is LAST coefficient
    # Construct C such that X @ C has c as last column
    # This is done via column pivoting in the QR

    # Simpler approach: augment X with contrast direction, then extract
    # We use the standard QR and extract the residual space

    # Apply weights if provided
    if sample_weights is not None:
        W_sqrt = np.sqrt(np.diag(sample_weights[valid_mask]))
        X_weighted = W_sqrt @ X
    else:
        X_weighted = X

    # Full QR decomposition
    # X = Q @ R where Q is n × n, R is n × p
    Q, R = np.linalg.qr(X_weighted, mode='complete')

    # Q2 contains columns p-1 through n-1 (d+1 columns total)
    # But we need to handle the contrast properly
    #
    # The standard approach: Q2 = Q[:, p:] gives the null space
    # For ROAST, we want Q2[:, 0] to correspond to the contrast direction
    #
    # Following limma's .lmEffects: the effects matrix has d+1 columns
    # where first column is the effect of interest

    # Extract null space (residual space)
    Q2 = Q[:, n_params:]  # Shape: (n_samples, df_residual)

    # For ROAST, we actually want df_residual + 1 dimensions
    # The "+1" comes from the contrast effect dimension
    # Recompute to include contrast dimension

    # Effects approach (like limma .lmEffects):
    # Compute coefficients and residuals together
    # effects_matrix columns: [contrast_effect, residual_1, ..., residual_d]

    # Actually, the projection we need is onto the space containing:
    # 1. The contrast direction (1 dim)
    # 2. The residual space (d dims)
    # Total: d + 1 dims

    # Q[:, n_params-1:] would give us these d+1 dimensions
    # But the ordering depends on how X is constructed

    # Safer approach: compute the projection matrix explicitly
    # P_contrast = c @ c' / (c' @ (X'X)^-1 @ c) projected onto column space

    # For now, use the residual approach with +1 adjustment
    # This matches limma's behavior
    Q2_full = Q[:, (n_params - 1):]  # d + 1 columns

    # Sign correction: ensure first column of Q2 aligns with contrast direction.
    #
    # QR decomposition is unique only up to sign flips of Q columns.  We
    # resolve this by projecting the contrast signal in sample space
    # (Xc = X_weighted @ c) onto Q2_full[:, 0].
    #
    # Mathematical justification: Q2_full spans the complement of the first
    # p-1 columns of Q (the reduced model without the contrast).  The contrast
    # signal Xc is NOT in that reduced model's column space, so Q2.T @ Xc is
    # nonzero and its sign tells us whether Q2[:, 0] points "with" or "against"
    # the contrast.  We canonicalize it to point "with" the contrast so that
    # positive test statistics correspond to upregulation.
    #
    # This replaces a correlation heuristic that was fragile for balanced
    # designs where the correlation could be near zero.
    Xc = X_weighted @ c
    proj = Q2_full[:, 0] @ Xc
    if abs(proj) < 1e-10:
        # Near-degenerate: contrast projection is near-zero.  This should not
        # happen for a valid two-group contrast, but we warn and keep the
        # current sign rather than making an arbitrary flip.
        warnings.warn(
            "Q2 sign correction: contrast projection onto Q2[:,0] is near-zero "
            f"({proj:.2e}). Directional (UP/DOWN) p-values may be unreliable.",
            stacklevel=2,
        )
    elif proj < 0:
        Q2_full = Q2_full.copy()
        Q2_full[:, 0] = -Q2_full[:, 0]

    contrast_name = f"{contrast[0]}_vs_{contrast[1]}"

    return RotationPrecomputed(
        Q2=Q2_full,
        df_residual=df_residual,
        contrast_name=contrast_name,
        design_rank=n_params,
        n_samples=n_samples_valid,
    )


def extract_gene_effects(
    Y: NDArray[np.float64],
    gene_ids: list[str],
    precomputed: RotationPrecomputed,
    eb_d0: float | None = None,
    eb_s0_sq: float | None = None,
) -> GeneEffects:
    """
    Project gene expression data onto residual space.

    For each gene g, computes:
        u_g = Q2' @ y_g

    This projection removes nuisance parameters while preserving the
    information needed for hypothesis testing and rotation.

    Args:
        Y: Expression matrix (n_genes, n_samples)
        gene_ids: Gene identifiers matching rows of Y
        precomputed: Rotation matrices from compute_rotation_matrices()
        eb_d0: Empirical Bayes prior df (None = no shrinkage)
        eb_s0_sq: Empirical Bayes prior scale

    Returns:
        GeneEffects with projected vectors and variance estimates

    Complexity: O(n_genes × n_samples × d) - can be GPU accelerated
    """
    n_genes, n_samples = Y.shape

    if n_samples != precomputed.Q2.shape[0]:
        raise ValueError(
            f"Sample mismatch: Y has {n_samples}, Q2 has {precomputed.Q2.shape[0]}"
        )

    # Project: U = Y @ Q2 (each row is u_g = Q2' @ y_g, transposed)
    # Shape: (n_genes, d+1)
    U = Y @ precomputed.Q2

    # Compute squared norms: ρ²_g = ||u_g||²
    rho_sq = np.sum(U ** 2, axis=1)

    # Sample variances: s²_g = (ρ²_g - u_g[0]²) / df
    # First column of U is the contrast effect
    u_first_sq = U[:, 0] ** 2
    df = precomputed.df_residual

    # Protect against numerical issues
    residual_ss = np.maximum(rho_sq - u_first_sq, 0)
    sample_variances = residual_ss / max(df, 1)

    # Apply empirical Bayes shrinkage if requested
    moderated_variances = None
    df_total = None

    if eb_d0 is not None and eb_s0_sq is not None and not np.isinf(eb_d0):
        # Posterior variance: weighted average of prior and sample
        moderated_variances = (eb_d0 * eb_s0_sq + df * sample_variances) / (eb_d0 + df)
        df_total = eb_d0 + df

    return GeneEffects(
        U=U,
        rho_sq=rho_sq,
        gene_ids=gene_ids,
        sample_variances=sample_variances,
        moderated_variances=moderated_variances,
        df_total=df_total,
    )


# =============================================================================
# Rotation Generation and Application (GPU-Accelerated)
# =============================================================================

def generate_rotation_vectors(
    n_rotations: int,
    n_dims: int,
    rng: np.random.Generator | None = None,
    use_gpu: bool = True,
) -> NDArray[np.float64]:
    """
    Generate random rotation vectors on the unit (n_dims - 1)-sphere.

    Each rotation vector r satisfies ||r|| = 1 and is uniformly
    distributed on the sphere surface.

    Method: Sample v ~ N(0, I), normalize: r = v / ||v||

    Args:
        n_rotations: Number of rotation vectors to generate
        n_dims: Dimensionality (d + 1 for ROAST)
        rng: Random number generator (for reproducibility)
        use_gpu: Whether to use GPU acceleration

    Returns:
        R: Rotation matrix (n_rotations, n_dims) with unit-norm rows

    Complexity: O(n_rotations × n_dims)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate standard normal vectors
    V = rng.standard_normal((n_rotations, n_dims))

    if use_gpu and MLX_AVAILABLE:
        V_mx = mx.array(V, dtype=mx.float32)
        norms = mx.sqrt(mx.sum(V_mx ** 2, axis=1, keepdims=True))
        R_mx = V_mx / mx.maximum(norms, mx.array(1e-10))
        R = np.array(R_mx, dtype=np.float64)
    else:
        # CPU: normalize rows
        norms = np.sqrt(np.sum(V ** 2, axis=1, keepdims=True))
        R = V / np.maximum(norms, 1e-10)

    return R


def apply_rotations_batched(
    U: NDArray[np.float64],
    rho_sq: NDArray[np.float64],
    R: NDArray[np.float64],
    sample_variances: NDArray[np.float64],
    moderated_variances: NDArray[np.float64] | None,
    df_residual: int,
    df_total: float | None,
    use_gpu: bool = True,
    chunk_size: int = 10000,
    eb_d0: float | None = None,
    eb_s0_sq: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply rotation vectors to gene effects and compute rotated t-statistics.

    For each rotation r and each gene g:

    1. Rotated first element: u*_g = u_g' @ r  (inner product)
    2. Rotated variance: s*²_g = (ρ²_g - u*²_g) / df
    3. If EB: moderated variance = (d0·s0² + df·s*²_g) / (d0 + df)
    4. Rotated t-statistic: t*_g = u*_g / sqrt(s*²_mod)

    Args:
        U: Gene effects matrix (n_genes, n_dims)
        rho_sq: Squared norms (n_genes,)
        R: Rotation vectors (n_rotations, n_dims)
        sample_variances: Original sample variances (n_genes,)
        moderated_variances: EB-shrunk variances or None (used for fallback)
        df_residual: Residual degrees of freedom
        df_total: Total df for moderated t (if using EB)
        use_gpu: Whether to use GPU
        chunk_size: Process rotations in chunks for memory management
        eb_d0: Empirical Bayes prior df (for proper per-rotation shrinkage)
        eb_s0_sq: Empirical Bayes prior scale (for proper per-rotation shrinkage)

    Returns:
        Tuple of:
        - rotated_t: Rotated t-statistics (n_rotations, n_genes)
        - rotated_z: Rotated z-scores (n_rotations, n_genes)

    Complexity: O(n_rotations × n_genes × n_dims)
    """
    n_genes, n_dims = U.shape
    n_rotations = R.shape[0]

    # Determine df for p-value/z-score conversion
    if df_total is not None and not np.isinf(df_total):
        use_df = df_total
    else:
        use_df = df_residual

    # Process in chunks if needed
    if n_rotations > chunk_size:
        rotated_t_chunks = []
        rotated_z_chunks = []

        for start in range(0, n_rotations, chunk_size):
            end = min(start + chunk_size, n_rotations)
            R_chunk = R[start:end]

            t_chunk, z_chunk = _apply_rotations_impl(
                U, rho_sq, R_chunk, sample_variances, moderated_variances,
                df_residual, use_df, use_gpu, eb_d0, eb_s0_sq
            )

            rotated_t_chunks.append(t_chunk)
            rotated_z_chunks.append(z_chunk)

        return np.vstack(rotated_t_chunks), np.vstack(rotated_z_chunks)
    else:
        return _apply_rotations_impl(
            U, rho_sq, R, sample_variances, moderated_variances,
            df_residual, use_df, use_gpu, eb_d0, eb_s0_sq
        )


def _apply_rotations_impl(
    U: NDArray[np.float64],
    rho_sq: NDArray[np.float64],
    R: NDArray[np.float64],
    sample_variances: NDArray[np.float64],
    moderated_variances: NDArray[np.float64] | None,
    df_residual: int,
    use_df: float,
    use_gpu: bool,
    eb_d0: float | None = None,
    eb_s0_sq: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Internal implementation of rotation application."""

    n_genes = U.shape[0]
    n_rotations = R.shape[0]

    if use_gpu and MLX_AVAILABLE:
        return _apply_rotations_gpu(
            U, rho_sq, R, sample_variances, moderated_variances,
            df_residual, use_df, eb_d0, eb_s0_sq
        )
    else:
        return _apply_rotations_cpu(
            U, rho_sq, R, sample_variances, moderated_variances,
            df_residual, use_df, eb_d0, eb_s0_sq
        )


def _apply_rotations_gpu(
    U: NDArray[np.float64],
    rho_sq: NDArray[np.float64],
    R: NDArray[np.float64],
    sample_variances: NDArray[np.float64],
    moderated_variances: NDArray[np.float64] | None,
    df_residual: int,
    use_df: float,
    eb_d0: float | None = None,
    eb_s0_sq: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GPU implementation using MLX."""
    # Note: moderated_variances is accepted for backward-compatible call signature
    # but is no longer used. EB shrinkage is applied directly via eb_d0/eb_s0_sq.

    # Convert to MLX arrays
    U_mx = mx.array(U, dtype=mx.float32)
    rho_sq_mx = mx.array(rho_sq, dtype=mx.float32)
    R_mx = mx.array(R, dtype=mx.float32)

    # Rotated first elements: U_rot = U @ R'
    # Shape: (n_genes, n_rotations)
    U_rot = mx.matmul(U_mx, R_mx.T)

    # Rotated squared elements
    U_rot_sq = U_rot ** 2

    # Rotated residual SS: ρ² - u*²
    # Broadcasting: (n_genes, 1) - (n_genes, n_rotations)
    residual_ss_rot = rho_sq_mx[:, None] - U_rot_sq
    residual_ss_rot = mx.maximum(residual_ss_rot, mx.array(1e-10))

    # Rotated sample variances
    var_rot = residual_ss_rot / max(df_residual, 1)

    # Apply EB shrinkage using proper formula if priors are available
    if eb_d0 is not None and eb_s0_sq is not None and not np.isinf(eb_d0):
        # Proper per-rotation EB shrinkage:
        # s²_post = (d0 × s0² + df × s²_rot) / (d0 + df)
        d0_mx = mx.array(eb_d0, dtype=mx.float32)
        s0_sq_mx = mx.array(eb_s0_sq, dtype=mx.float32)
        df_mx = mx.array(float(df_residual), dtype=mx.float32)
        var_rot = (d0_mx * s0_sq_mx + df_mx * var_rot) / (d0_mx + df_mx)
    # When EB priors (d0, s0_sq) are unavailable, use unmoderated rotated
    # variances directly.  A previous version applied a ratio approximation
    # (var_rot * moderated_var / sample_var), but that is mathematically
    # incorrect: rotated variances follow a different distribution than the
    # originals, so the per-gene shrinkage ratio does not transfer.  Using
    # unmoderated variances is conservative but correct.

    # Rotated t-statistics
    se_rot = mx.sqrt(var_rot)
    t_rot = U_rot / mx.maximum(se_rot, mx.array(1e-10))

    # Convert t to z-scores via probability integral transform
    # z = Φ⁻¹(F_t(t)) where F_t is t-distribution CDF
    # For large df, t ≈ z, so we use normal approximation for GPU efficiency
    # and correct on CPU for small df

    # Transfer back to CPU for scipy operations
    t_rot_np = np.array(t_rot.T, dtype=np.float64)  # Shape: (n_rotations, n_genes)

    # Convert t to z via t-distribution (this is the bottleneck for large df)
    if use_df > 100:
        # Large df: t ≈ z
        z_rot_np = t_rot_np.copy()
    else:
        # Small df: proper transformation
        # F_t(t) gives p-value, Φ⁻¹ converts to z
        p_values = scipy_stats.t.cdf(t_rot_np, df=use_df)
        # Clip to avoid inf
        p_values = np.clip(p_values, 1e-15, 1 - 1e-15)
        z_rot_np = ndtri(p_values)

    return t_rot_np, z_rot_np


def _apply_rotations_cpu(
    U: NDArray[np.float64],
    rho_sq: NDArray[np.float64],
    R: NDArray[np.float64],
    sample_variances: NDArray[np.float64],
    moderated_variances: NDArray[np.float64] | None,
    df_residual: int,
    use_df: float,
    eb_d0: float | None = None,
    eb_s0_sq: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """CPU implementation using NumPy."""
    # Note: moderated_variances is accepted for backward-compatible call signature
    # but is no longer used. EB shrinkage is applied directly via eb_d0/eb_s0_sq.

    n_genes = U.shape[0]
    n_rotations = R.shape[0]

    # Rotated first elements: (n_genes, n_rotations)
    U_rot = U @ R.T

    # Rotated squared elements
    U_rot_sq = U_rot ** 2

    # Rotated residual SS
    residual_ss_rot = rho_sq[:, None] - U_rot_sq
    residual_ss_rot = np.maximum(residual_ss_rot, 1e-10)

    # Rotated sample variances
    var_rot = residual_ss_rot / max(df_residual, 1)

    # Apply EB shrinkage using proper formula if priors are available
    if eb_d0 is not None and eb_s0_sq is not None and not np.isinf(eb_d0):
        # Proper per-rotation EB shrinkage:
        # s²_post = (d0 × s0² + df × s²_rot) / (d0 + df)
        var_rot = (eb_d0 * eb_s0_sq + df_residual * var_rot) / (eb_d0 + df_residual)
    # When EB priors (d0, s0_sq) are unavailable, use unmoderated rotated
    # variances directly.  A previous version applied a ratio approximation
    # (var_rot * moderated_var / sample_var), but that is mathematically
    # incorrect: rotated variances follow a different distribution than the
    # originals, so the per-gene shrinkage ratio does not transfer.  Using
    # unmoderated variances is conservative but correct.

    # t-statistics
    se_rot = np.sqrt(var_rot)
    t_rot = U_rot / np.maximum(se_rot, 1e-10)

    # Transpose to (n_rotations, n_genes) for consistency
    t_rot = t_rot.T

    # Convert to z-scores
    if use_df > 100:
        z_rot = t_rot.copy()
    else:
        p_values = scipy_stats.t.cdf(t_rot, df=use_df)
        p_values = np.clip(p_values, 1e-15, 1 - 1e-15)
        z_rot = ndtri(p_values)

    return t_rot, z_rot


# =============================================================================
# Set Statistics Computation
# =============================================================================

def compute_set_statistics(
    z_scores: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    statistics: list[SetStatistic] | None = None,
    alternatives: list[Alternative] | None = None,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """
    Compute gene set statistics from z-scores.

    For each statistic type and alternative hypothesis, aggregates
    gene-level z-scores into a single set-level statistic.

    Args:
        z_scores: Z-scores matrix (n_rotations, n_genes) or (n_genes,) for observed
        weights: Optional gene weights (n_genes,). Default: equal weights.
        statistics: Which statistics to compute. Default: all.
        alternatives: Which alternatives to test. Default: all.

    Returns:
        Nested dict: statistic -> alternative -> array of set statistics
        For observed data, arrays are scalars (0-d arrays).

    Complexity: O(n_rotations × n_genes) per statistic
    """
    if statistics is None:
        statistics = list(SetStatistic)
    if alternatives is None:
        alternatives = list(Alternative)

    # Handle 1D input (observed)
    if z_scores.ndim == 1:
        z_scores = z_scores[np.newaxis, :]  # (1, n_genes)
        squeeze_output = True
    else:
        squeeze_output = False

    n_rotations, n_genes = z_scores.shape

    # Default: equal weights
    if weights is None:
        weights = np.ones(n_genes)

    abs_weights = np.abs(weights)
    A = np.sum(abs_weights)  # Normalizing factor

    # Floor value for floormean (sqrt of median chi-squared with 1 df)
    floor_value = np.sqrt(scipy_stats.chi2.ppf(0.5, df=1))  # ≈ 0.6745

    results: dict[str, dict[str, NDArray[np.float64]]] = {}

    for stat in statistics:
        results[stat.value] = {}

        for alt in alternatives:
            if stat == SetStatistic.MEAN:
                vals = _compute_mean_stat(z_scores, weights, A, alt)
            elif stat == SetStatistic.FLOORMEAN:
                vals = _compute_floormean_stat(z_scores, weights, A, alt, floor_value)
            elif stat == SetStatistic.MEAN50:
                vals = _compute_mean50_stat(z_scores, weights, A, alt)
            elif stat == SetStatistic.MSQ:
                vals = _compute_msq_stat(z_scores, abs_weights, A, alt)
            elif stat == SetStatistic.MIXED:
                mean_vals = _compute_mean_stat(z_scores, weights, A, alt)
                msq_vals = _compute_msq_stat(z_scores, abs_weights, A, alt)
                vals = (mean_vals ** 2 + msq_vals) / 2
            else:
                raise ValueError(f"Unknown statistic: {stat}")

            if squeeze_output:
                vals = vals[0]  # Scalar for observed

            results[stat.value][alt.value] = vals

    return results


def _compute_mean_stat(
    z: NDArray[np.float64],
    w: NDArray[np.float64],
    A: float,
    alt: Alternative,
) -> NDArray[np.float64]:
    """Mean statistic: weighted average of z-scores."""
    if alt == Alternative.UP:
        # Positive direction
        return np.sum(w * z, axis=1) / A
    elif alt == Alternative.DOWN:
        # Negative direction
        return -np.sum(w * z, axis=1) / A
    else:  # MIXED
        # Absolute values for two-sided
        return np.sum(np.abs(w) * np.abs(z), axis=1) / A


def _compute_floormean_stat(
    z: NDArray[np.float64],
    w: NDArray[np.float64],
    A: float,
    alt: Alternative,
    floor: float,
) -> NDArray[np.float64]:
    """Floormean statistic: mean with floored absolute values."""
    if alt == Alternative.UP:
        f = np.maximum(z, 0)
        return np.sum(w * f, axis=1) / A
    elif alt == Alternative.DOWN:
        f = np.minimum(z, 0)
        return -np.sum(w * f, axis=1) / A
    else:  # MIXED
        f = np.maximum(np.abs(z), floor)
        return np.sum(np.abs(w) * f, axis=1) / A


def _compute_mean50_stat(
    z: NDArray[np.float64],
    w: NDArray[np.float64],
    A: float,
    alt: Alternative,
) -> NDArray[np.float64]:
    """Mean50 statistic: mean of top 50% genes."""
    n_genes = z.shape[1]
    h = (n_genes + 1) // 2  # Top half

    if alt == Alternative.UP:
        # Top h largest weighted z
        wz = w * z
        sorted_wz = np.sort(wz, axis=1)[:, ::-1]  # Descending
        return np.mean(sorted_wz[:, :h], axis=1)
    elif alt == Alternative.DOWN:
        # Top h smallest (most negative) weighted z
        wz = w * z
        sorted_wz = np.sort(wz, axis=1)  # Ascending
        return -np.mean(sorted_wz[:, :h], axis=1)
    else:  # MIXED
        # Top h largest |weighted z|
        abs_wz = np.abs(w) * np.abs(z)
        sorted_abs = np.sort(abs_wz, axis=1)[:, ::-1]
        return np.mean(sorted_abs[:, :h], axis=1)


def _compute_msq_stat(
    z: NDArray[np.float64],
    abs_w: NDArray[np.float64],
    A: float,
    alt: Alternative,
) -> NDArray[np.float64]:
    """Mean-squared statistic: mean of squared z-scores.

    This is DIRECTION-AGNOSTIC - critical for bidirectional TF regulation.
    """
    z_sq = z ** 2

    if alt == Alternative.UP:
        # Only positive z contribute
        mask = z > 0
        masked_sq = np.where(mask, z_sq, 0)
        return np.sum(abs_w * masked_sq, axis=1) / A
    elif alt == Alternative.DOWN:
        # Only negative z contribute
        mask = z < 0
        masked_sq = np.where(mask, z_sq, 0)
        return np.sum(abs_w * masked_sq, axis=1) / A
    else:  # MIXED
        # All z contribute (direction-agnostic)
        return np.sum(abs_w * z_sq, axis=1) / A


# =============================================================================
# P-Value Computation
# =============================================================================

def compute_rotation_pvalues(
    observed_stats: dict[str, dict[str, float]],
    null_stats: dict[str, dict[str, NDArray[np.float64]]],
) -> dict[str, dict[str, float]]:
    """
    Compute empirical p-values from observed vs null statistics.

    P-value formula (Phipson & Smyth, 2010):

        p = (b + 1) / (B + 1)

    where b = count of null statistics >= observed.

    This "+1" adjustment ensures:
    - P-values are never exactly 0
    - Conservative for finite B
    - Exact in the limit B → ∞

    Args:
        observed_stats: stat -> alt -> observed value
        null_stats: stat -> alt -> array of null values

    Returns:
        stat -> alt -> p-value
    """
    p_values: dict[str, dict[str, float]] = {}

    for stat, alt_obs in observed_stats.items():
        p_values[stat] = {}

        for alt, obs in alt_obs.items():
            null = null_stats.get(stat, {}).get(alt)

            if null is None or len(null) == 0:
                p_values[stat][alt] = np.nan
                continue

            # Count null values >= observed (upper tail)
            b = np.sum(null >= obs)
            B = len(null)

            p_values[stat][alt] = (b + 1) / (B + 1)

    return p_values


def estimate_active_proportion(
    t_statistics: NDArray[np.float64],
    df_total: float,
    alternatives: list[Alternative] | None = None,
) -> dict[str, float]:
    """
    Estimate proportion of genes actively contributing to the signal.

    A gene is considered "active" if |t̃| > √2, motivated by the
    AIC criterion for model selection (Wu et al., 2010).

    Args:
        t_statistics: Gene-level moderated t-statistics (n_genes,)
        df_total: Degrees of freedom for t-distribution
        alternatives: Which alternatives to compute

    Returns:
        Dict mapping alternative to proportion (0 to 1)
    """
    if alternatives is None:
        alternatives = list(Alternative)

    threshold = np.sqrt(2)  # AIC-motivated threshold

    proportions = {}
    n_genes = len(t_statistics)

    for alt in alternatives:
        if alt == Alternative.UP:
            active = np.sum(t_statistics > threshold)
        elif alt == Alternative.DOWN:
            active = np.sum(t_statistics < -threshold)
        else:  # MIXED
            active = np.sum(np.abs(t_statistics) > threshold)

        proportions[alt.value] = active / max(n_genes, 1)

    return proportions


# =============================================================================
# High-Level API: Rotation Test Engine
# =============================================================================

@dataclass
class RotationTestConfig:
    """Configuration for rotation gene set testing.

    Attributes:
        n_rotations: Number of random rotations (default: 9999)
                    Higher = more precise p-values, more compute

        statistics: Which set statistics to compute
                   Default: all (MEAN, FLOORMEAN, MEAN50, MSQ, MIXED)

        alternatives: Which alternative hypotheses to test
                     Default: all (UP, DOWN, MIXED)

        use_eb: Whether to use Empirical Bayes variance shrinkage
               Strongly recommended for small samples.

        use_gpu: Whether to use GPU acceleration (MLX)

        chunk_size: Process rotations in chunks of this size
                   Larger = faster, more memory

        seed: Random seed for reproducibility
    """

    n_rotations: int = 9999
    statistics: list[SetStatistic] = field(default_factory=lambda: list(SetStatistic))
    alternatives: list[Alternative] = field(default_factory=lambda: list(Alternative))
    use_eb: bool = True
    use_gpu: bool = True
    chunk_size: int = 10000
    seed: int | None = None


class RotationTestEngine:
    """
    Engine for rotation-based gene set testing.

    This class implements the complete ROAST methodology with GPU acceleration
    and integration into the cliquefinder infrastructure.

    Usage:
        # Initialize with expression data
        engine = RotationTestEngine(
            expression_data,      # (n_genes, n_samples)
            gene_ids,             # list of gene identifiers
            sample_metadata,      # DataFrame with sample annotations
        )

        # Precompute design matrices (once per contrast)
        engine.fit(
            conditions=['treatment', 'control'],
            contrast=('treatment', 'control'),
            condition_column='treatment_group',
        )

        # Run tests on gene sets
        results = engine.test_gene_sets(
            gene_sets,            # list of feature sets
            config=RotationTestConfig(n_rotations=9999),
        )

    The engine maintains state for efficient batch testing of multiple
    gene sets against the same contrast.
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        gene_ids: list[str],
        metadata: pd.DataFrame,
    ):
        """
        Initialize the rotation test engine.

        Args:
            data: Expression matrix (n_genes, n_samples), log-transformed
            gene_ids: Gene identifiers matching rows of data
            metadata: Sample metadata with condition and optional subject columns
        """
        self.data = data
        self.gene_ids = gene_ids
        self.gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        self.metadata = metadata

        # State set by fit()
        self._precomputed: RotationPrecomputed | None = None
        self._effects: GeneEffects | None = None
        self._fitted = False

    def fit(
        self,
        conditions: list[str],
        contrast: tuple[str, str],
        condition_column: str,
        subject_column: str | None = None,
        covariates: list[str] | None = None,
    ) -> "RotationTestEngine":
        """
        Fit the rotation model: compute QR decomposition and gene effects.

        This is the expensive precomputation step. After calling fit(),
        you can test many gene sets efficiently.

        Args:
            conditions: List of condition labels (e.g., ['treatment', 'control'])
            contrast: Tuple (test, reference) for comparison
            condition_column: Metadata column containing condition labels (REQUIRED - must be specified by user)
            subject_column: Optional column for subject IDs (for future LMM support)
            covariates: Optional list of metadata column names to include
                as fixed covariates in the design matrix (e.g., ['Sex', 'Batch']).
                When provided, delegates to fit_general() with a covariate-aware
                design matrix. Covariates are nuisance parameters — the contrast
                tests only the condition effect.

        Returns:
            self (for method chaining)
        """
        # If covariates are specified, build a full design matrix and
        # delegate to fit_general() which handles arbitrary designs.
        if covariates:
            from .design_matrix import build_covariate_design_matrix

            covariates_df = self.metadata[covariates]
            design = build_covariate_design_matrix(
                sample_condition=self.metadata[condition_column].values,
                conditions=conditions,
                contrast=contrast,
                covariates_df=covariates_df,
            )

            # Filter data and metadata to valid samples
            self.data = self.data[:, design.sample_mask]
            self.metadata = self.metadata[design.sample_mask].reset_index(drop=True)

            return self.fit_general(
                design_matrix=design.X,
                contrast=design.contrast,
                contrast_name=design.contrast_name,
            )

        # Standard path: no covariates, simple two-group comparison
        # Get condition labels
        sample_conditions = self.metadata[condition_column].values

        # Check for >2 groups: ROAST is designed for 2-group comparisons
        if contrast is not None:
            unique_conditions = set(sample_conditions)
            # Remove NaN-like values
            unique_conditions = {
                c for c in unique_conditions
                if c is not None and (not isinstance(c, float) or not np.isnan(c))
            }
            n_groups = len(unique_conditions)
            if n_groups > 2:
                warnings.warn(
                    f"ROAST rotation test is designed for 2-group comparisons. "
                    f"{n_groups} groups detected; results may not be valid. "
                    f"Consider subsetting to the 2 groups in your contrast."
                )

        # Compute rotation matrices (QR decomposition)
        self._precomputed = compute_rotation_matrices(
            sample_conditions,
            conditions,
            contrast,
        )

        # Compute Empirical Bayes priors from ALL genes
        # This borrows strength across the genome
        from .permutation_gpu import fit_f_dist

        # Extract gene effects for ALL genes
        self._effects = extract_gene_effects(
            self.data,
            self.gene_ids,
            self._precomputed,
        )

        # Fit EB priors
        valid_var = self._effects.sample_variances[
            (self._effects.sample_variances > 0) &
            np.isfinite(self._effects.sample_variances)
        ]

        if len(valid_var) > 10:
            eb_d0, eb_s0_sq = fit_f_dist(valid_var, self._precomputed.df_residual)
        else:
            eb_d0, eb_s0_sq = np.inf, float(np.median(valid_var))

        # Re-extract with EB priors
        self._effects = extract_gene_effects(
            self.data,
            self.gene_ids,
            self._precomputed,
            eb_d0=eb_d0,
            eb_s0_sq=eb_s0_sq,
        )

        # Store EB params in precomputed for reference
        self._precomputed = RotationPrecomputed(
            Q2=self._precomputed.Q2,
            df_residual=self._precomputed.df_residual,
            contrast_name=self._precomputed.contrast_name,
            eb_d0=eb_d0,
            eb_s0_sq=eb_s0_sq,
            eb_df_total=eb_d0 + self._precomputed.df_residual if not np.isinf(eb_d0) else None,
            design_rank=self._precomputed.design_rank,
            n_samples=self._precomputed.n_samples,
        )

        self._fitted = True
        return self

    def fit_general(
        self,
        design_matrix: NDArray[np.float64],
        contrast: NDArray[np.float64],
        contrast_name: str = "custom_contrast",
    ) -> "RotationTestEngine":
        """
        Fit the rotation model with a custom design matrix and contrast.

        This is the most flexible fitting method, accepting arbitrary
        design matrices and contrast vectors. Use this for:
        - Interaction contrasts (Sex × Disease)
        - Multi-group comparisons with complex contrasts
        - Designs with continuous covariates

        Args:
            design_matrix: Design matrix X (n_samples, n_params)
            contrast: Contrast vector c (n_params,)
            contrast_name: Human-readable name for the contrast

        Returns:
            self (for method chaining)

        Example:
            # Cell-means parameterization for 2×2 factorial
            design = np.column_stack([
                (sex == 'M') & (disease == 'CASE'),
                (sex == 'M') & (disease == 'CTRL'),
                (sex == 'F') & (disease == 'CASE'),
                (sex == 'F') & (disease == 'CTRL'),
            ]).astype(float)

            # Interaction: (M_CASE - M_CTRL) - (F_CASE - F_CTRL)
            contrast = np.array([1, -1, -1, 1])

            engine.fit_general(design, contrast, "sex_x_disease")
        """
        from .permutation_gpu import fit_f_dist

        # Compute rotation matrices via C-matrix reparameterization
        self._precomputed = compute_rotation_matrices_general(
            design_matrix,
            contrast,
            contrast_name=contrast_name,
        )

        # Extract gene effects for ALL genes
        self._effects = extract_gene_effects(
            self.data,
            self.gene_ids,
            self._precomputed,
        )

        # Fit EB priors
        valid_var = self._effects.sample_variances[
            (self._effects.sample_variances > 0) &
            np.isfinite(self._effects.sample_variances)
        ]

        if len(valid_var) > 10:
            eb_d0, eb_s0_sq = fit_f_dist(valid_var, self._precomputed.df_residual)
        else:
            eb_d0, eb_s0_sq = np.inf, float(np.median(valid_var))

        # Re-extract with EB priors
        self._effects = extract_gene_effects(
            self.data,
            self.gene_ids,
            self._precomputed,
            eb_d0=eb_d0,
            eb_s0_sq=eb_s0_sq,
        )

        # Update precomputed with EB params
        self._precomputed = RotationPrecomputed(
            Q2=self._precomputed.Q2,
            df_residual=self._precomputed.df_residual,
            contrast_name=self._precomputed.contrast_name,
            eb_d0=eb_d0,
            eb_s0_sq=eb_s0_sq,
            eb_df_total=eb_d0 + self._precomputed.df_residual if not np.isinf(eb_d0) else None,
            design_rank=self._precomputed.design_rank,
            n_samples=self._precomputed.n_samples,
        )

        self._fitted = True
        return self

    def fit_interaction(
        self,
        factor1_column: str,
        factor2_column: str,
        factor1_name: str | None = None,
        factor2_name: str | None = None,
    ) -> "RotationTestEngine":
        """
        Fit the rotation model for a 2×2 factorial interaction test.

        This is a convenience method for testing Sex × Disease interactions
        or similar 2-factor designs. It automatically constructs the
        cell-means design matrix and interaction contrast.

        The interaction tests whether the effect of factor2 differs
        across levels of factor1:
            (F1_L1 × F2_L1 - F1_L1 × F2_L2) - (F1_L2 × F2_L1 - F1_L2 × F2_L2)

        For Sex × Disease:
            (Male_CASE - Male_CTRL) - (Female_CASE - Female_CTRL)

        Args:
            factor1_column: Metadata column for first factor (e.g., 'sex')
            factor2_column: Metadata column for second factor (e.g., 'phenotype')
            factor1_name: Display name for factor 1 (default: column name)
            factor2_name: Display name for factor 2 (default: column name)

        Returns:
            self (for method chaining)

        Example:
            engine.fit_interaction(
                factor1_column='sex',
                factor2_column='phenotype',
            )
            # Tests: (Male_CASE - Male_CTRL) - (Female_CASE - Female_CTRL)
        """
        f1_name = factor1_name or factor1_column
        f2_name = factor2_name or factor2_column

        # Get factor labels from metadata
        factor1_labels = self.metadata[factor1_column].values
        factor2_labels = self.metadata[factor2_column].values

        # Build design matrix and interaction contrast
        design, group_names, contrast = build_interaction_design(
            factor1_labels,
            factor2_labels,
            factor1_name=f1_name,
            factor2_name=f2_name,
        )

        contrast_name = f"{f1_name}_x_{f2_name}_interaction"

        # Use general fit method
        return self.fit_general(design, contrast, contrast_name)

    def test_gene_set(
        self,
        gene_set: WeightedFeatureSet | list[str],
        gene_set_id: str | None = None,
        weights: NDArray[np.float64] | None = None,
        config: RotationTestConfig | None = None,
    ) -> RotationResult:
        """
        Test a single gene set using rotation.

        Args:
            gene_set: Either a WeightedFeatureSet or list of gene IDs
            gene_set_id: Identifier for the set (required if gene_set is a list)
            weights: Optional per-gene weights (if gene_set is a list)
            config: Test configuration

        Returns:
            RotationResult with p-values for all statistic/alternative combinations
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before test_gene_set()")

        if config is None:
            config = RotationTestConfig()

        # Extract gene IDs and weights
        if isinstance(gene_set, list):
            feature_ids = gene_set
            set_id = gene_set_id or "unnamed_set"
            set_weights = weights
        else:
            feature_ids = gene_set.feature_ids
            set_id = gene_set.id
            set_weights = gene_set.weights if hasattr(gene_set, 'weights') else None

        # Find genes in our data
        gene_indices = []
        found_ids = []
        weight_list = []

        for i, gid in enumerate(feature_ids):
            if gid in self.gene_to_idx:
                gene_indices.append(self.gene_to_idx[gid])
                found_ids.append(gid)
                if set_weights is not None:
                    weight_list.append(set_weights[i])

        if len(gene_indices) < 2:
            # Not enough genes found
            return RotationResult(
                feature_set_id=set_id,
                n_genes=len(feature_ids),
                n_genes_found=len(gene_indices),
                gene_ids=found_ids,
                observed_stats={},
                null_distributions={},
                p_values={},
                active_proportion={},
                n_rotations=0,
                contrast_name=self._precomputed.contrast_name,
            )

        gene_indices = np.array(gene_indices)
        final_weights = np.array(weight_list) if weight_list else None

        # Extract subset of effects for this gene set
        U_subset = self._effects.U[gene_indices]
        rho_sq_subset = self._effects.rho_sq[gene_indices]
        var_subset = self._effects.sample_variances[gene_indices]
        mod_var_subset = (
            self._effects.moderated_variances[gene_indices]
            if self._effects.moderated_variances is not None
            else None
        )

        # Compute OBSERVED t-statistics and z-scores
        # t_g = u_g[0] / sqrt(var_g * v) where v = 1 for projected data
        if mod_var_subset is not None:
            se_obs = np.sqrt(mod_var_subset)
        else:
            se_obs = np.sqrt(var_subset)

        t_obs = U_subset[:, 0] / np.maximum(se_obs, 1e-10)

        # Convert to z-scores
        df_total = self._effects.df_total or self._precomputed.df_residual
        if df_total > 100:
            z_obs = t_obs
        else:
            p_obs = scipy_stats.t.cdf(t_obs, df=df_total)
            p_obs = np.clip(p_obs, 1e-15, 1 - 1e-15)
            z_obs = ndtri(p_obs)

        # Compute observed set statistics
        observed_stats = compute_set_statistics(
            z_obs,
            weights=final_weights,
            statistics=config.statistics,
            alternatives=config.alternatives,
        )

        # Flatten observed (they're scalars)
        observed_flat = {
            stat: {alt: float(val) for alt, val in alts.items()}
            for stat, alts in observed_stats.items()
        }

        # Generate rotations
        rng = np.random.default_rng(config.seed)
        R = generate_rotation_vectors(
            config.n_rotations,
            self._precomputed.residual_dims,
            rng=rng,
            use_gpu=config.use_gpu and MLX_AVAILABLE,
        )

        # Apply rotations to get null t-statistics
        _, z_rot = apply_rotations_batched(
            U_subset,
            rho_sq_subset,
            R,
            var_subset,
            mod_var_subset,
            self._precomputed.df_residual,
            df_total,
            use_gpu=config.use_gpu and MLX_AVAILABLE,
            chunk_size=config.chunk_size,
            eb_d0=self._precomputed.eb_d0,
            eb_s0_sq=self._precomputed.eb_s0_sq,
        )

        # Compute null set statistics
        null_stats = compute_set_statistics(
            z_rot,
            weights=final_weights,
            statistics=config.statistics,
            alternatives=config.alternatives,
        )

        # Compute p-values
        p_values = compute_rotation_pvalues(observed_flat, null_stats)

        # Estimate active gene proportions
        active_prop = estimate_active_proportion(
            t_obs,
            df_total,
            config.alternatives,
        )

        return RotationResult(
            feature_set_id=set_id,
            n_genes=len(feature_ids),
            n_genes_found=len(gene_indices),
            gene_ids=found_ids,
            observed_stats=observed_flat,
            null_distributions=null_stats,
            p_values=p_values,
            active_proportion=active_prop,
            n_rotations=config.n_rotations,
            contrast_name=self._precomputed.contrast_name,
        )

    def test_gene_sets(
        self,
        gene_sets: list[WeightedFeatureSet] | dict[str, list[str]],
        config: RotationTestConfig | None = None,
        verbose: bool = True,
    ) -> list[RotationResult]:
        """
        Test multiple gene sets.

        Args:
            gene_sets: Either list of WeightedFeatureSet or dict mapping ID -> genes
            config: Test configuration (shared across all sets)
            verbose: Print progress

        Returns:
            List of RotationResult, one per gene set
        """
        if config is None:
            config = RotationTestConfig()

        # Convert dict format if needed
        if isinstance(gene_sets, dict):
            from .permutation_framework import SimpleFeatureSet
            sets = [
                SimpleFeatureSet(_id=sid, _feature_ids=genes)
                for sid, genes in gene_sets.items()
            ]
        else:
            sets = gene_sets

        results = []
        n_sets = len(sets)

        for i, gset in enumerate(sets):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Testing set {i + 1}/{n_sets}")

            result = self.test_gene_set(gset, config=config)
            results.append(result)

        return results

    def results_to_dataframe(
        self,
        results: list[RotationResult],
    ) -> pd.DataFrame:
        """
        Convert results to DataFrame with raw p-values.

        ROAST produces exact p-values per gene set via rotation tests.
        These raw p-values are statistically valid without FDR correction.

        For exploratory analysis, rank by p-value and use thresholds like
        p < 0.01 or p < 0.001 to identify candidates for follow-up.

        Args:
            results: List of RotationResult from test_gene_sets()

        Returns:
            DataFrame with one row per gene set, including:
            - Raw p-values for all statistic/alternative combinations
            - Observed test statistics
            - Gene set metadata (size, genes found)
        """
        records = [r.to_dict() for r in results]
        df = pd.DataFrame(records)

        return df


# =============================================================================
# Convenience Functions
# =============================================================================

def run_rotation_test(
    data: NDArray[np.float64],
    gene_ids: list[str],
    metadata: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    conditions: list[str],
    contrast: tuple[str, str],
    condition_column: str,
    n_rotations: int = 9999,
    use_eb: bool = True,
    use_gpu: bool = True,
    seed: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function for rotation-based gene set testing.

    This is the main entry point for ROAST analysis. It:
    1. Initializes the engine with expression data
    2. Fits the rotation model (QR decomposition, EB priors)
    3. Tests all gene sets
    4. Returns results as a DataFrame with raw p-values

    ROAST produces exact p-values per gene set. These are statistically
    valid without FDR correction. For exploratory analysis, use thresholds
    like p < 0.01 to identify candidates for follow-up validation.

    Args:
        data: Expression matrix (n_genes, n_samples), log2-transformed
        gene_ids: Gene identifiers
        metadata: Sample metadata
        gene_sets: Dict mapping set ID to list of gene IDs
        conditions: Condition labels (e.g., ['treatment', 'control'])
        contrast: Contrast to test (e.g., ('treatment', 'control'))
        condition_column: Metadata column with condition labels (REQUIRED)
        n_rotations: Number of rotations (higher = more precise p-values)
        use_eb: Use Empirical Bayes variance shrinkage
        use_gpu: Use GPU acceleration (requires MLX)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        DataFrame with columns:
        - feature_set_id: Gene set identifier
        - n_genes, n_genes_found: Set size and found genes
        - pvalue_{stat}_{alt}: Raw p-values for each statistic/alternative
        - observed_{stat}: Observed set statistics
        - active_proportion_{alt}: Estimated active gene fraction

    Example:
        >>> results = run_rotation_test(
        ...     data=expression_matrix,
        ...     gene_ids=gene_names,
        ...     metadata=sample_info,
        ...     gene_sets={'pathway_1': ['GENE1', 'GENE2'], ...},
        ...     conditions=['treatment', 'control'],
        ...     contrast=('treatment', 'control'),
        ...     condition_column='treatment_group',
        ... )
        >>> # Get top hits by MSQ p-value (bidirectional regulation)
        >>> top = results.nsmallest(20, 'pvalue_msq_mixed')
        >>> # Filter by nominal threshold for follow-up
        >>> candidates = results[results['pvalue_msq_mixed'] < 0.01]
    """
    if verbose:
        print(f"ROAST Rotation Test")
        print(f"  Genes: {len(gene_ids)}, Samples: {data.shape[1]}")
        print(f"  Gene sets: {len(gene_sets)}")
        print(f"  Rotations: {n_rotations}")
        print(f"  GPU: {'enabled' if use_gpu and MLX_AVAILABLE else 'disabled'}")
        print()

    # Initialize and fit
    engine = RotationTestEngine(data, gene_ids, metadata)

    if verbose:
        print("Fitting rotation model...")

    engine.fit(
        conditions=conditions,
        contrast=contrast,
        condition_column=condition_column,
    )

    if verbose:
        eb_d0 = engine._precomputed.eb_d0
        print(f"  EB prior df: {eb_d0:.1f}" if eb_d0 and not np.isinf(eb_d0) else "  EB: disabled")
        print()

    # Configure and run tests
    config = RotationTestConfig(
        n_rotations=n_rotations,
        use_eb=use_eb,
        use_gpu=use_gpu,
        seed=seed,
    )

    if verbose:
        print(f"Testing {len(gene_sets)} gene sets...")

    results = engine.test_gene_sets(gene_sets, config=config, verbose=verbose)

    # Convert to DataFrame
    df = engine.results_to_dataframe(results)

    if verbose:
        n_msq_01 = (df['pvalue_msq_mixed'] < 0.01).sum() if 'pvalue_msq_mixed' in df.columns else 0
        n_mean_01 = (df['pvalue_mean_mixed'] < 0.01).sum() if 'pvalue_mean_mixed' in df.columns else 0
        min_p = df['pvalue_msq_mixed'].min() if 'pvalue_msq_mixed' in df.columns else float('nan')
        print()
        print(f"Results (raw p-values, no FDR):")
        print(f"  Gene sets with p < 0.01 (MSQ mixed): {n_msq_01}")
        print(f"  Gene sets with p < 0.01 (MEAN mixed): {n_mean_01}")
        print(f"  Minimum MSQ p-value: {min_p:.4f}")

    return df
