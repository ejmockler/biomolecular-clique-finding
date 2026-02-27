"""
Differential abundance analysis using linear mixed effects models.

Implements MSstats-style statistical testing for proteomics data:
- Linear mixed models (LMM) for experiments with biological replicates
- Fixed effects models for simple designs
- Contrast-based hypothesis testing
- Multiple testing correction (FDR)

The statistical model from MSstats:
    log2(Intensity) ~ Condition + (1 | Subject)

where Condition is the fixed effect (what we're testing) and Subject is
a random effect capturing biological variation between individuals.

References:
    - Choi et al. (2014) MSstats: Bioinformatics 30(17):2524-2526
    - Kohler et al. (2023) MSstats v4: J Proteome Res 22(5):1466-1482
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

# Optional MLX import for GPU acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass(frozen=True)
class NetworkEnrichmentResult:
    """Result of competitive permutation enrichment test for network targets.

    Tests whether network targets (is_target=True) have systematically
    higher |t-statistics| than random protein sets of the same size.

    Attributes:
        observed_mean_abs_t: Mean |t| for network targets.
        null_mean: Mean of the null distribution (random sets).
        null_std: Standard deviation of the null distribution.
        z_score: Standardized enrichment score: (observed - null_mean) / null_std.
        empirical_pvalue: One-sided empirical p-value (targets > random).
        n_targets: Number of network targets with valid t-statistics.
        n_background: Number of background proteins.
        pct_down: Percentage of targets with negative t-statistic.
        direction_pvalue: Binomial test p-value for directional bias.
        mannwhitney_pvalue: Mann-Whitney U test p-value (targets vs background |t|).
        variance_inflation_factor: Camera VIF = 1 + (k-1)*rho_bar, where rho_bar
            is mean pairwise inter-gene correlation. Defaults to 1.0 (no
            correlation adjustment). See Wu & Smyth (2012) NAR 40(17):e133.
        mean_pairwise_correlation: Estimated mean pairwise correlation (rho_bar)
            among target genes. Defaults to 0.0 (unknown / not estimated).
    """

    observed_mean_abs_t: float
    null_mean: float
    null_std: float
    z_score: float
    empirical_pvalue: float
    n_targets: int
    n_background: int
    pct_down: float
    direction_pvalue: float
    mannwhitney_pvalue: float
    variance_inflation_factor: float = 1.0
    mean_pairwise_correlation: float = 0.0

    def to_dict(self) -> dict:
        """Convert to plain dict, matching the legacy return format.

        Returns:
            Dict with the same keys as the previous dict-based return.
        """
        return {
            'observed_mean_abs_t': self.observed_mean_abs_t,
            'null_mean': self.null_mean,
            'null_std': self.null_std,
            'z_score': self.z_score,
            'empirical_pvalue': self.empirical_pvalue,
            'n_targets': self.n_targets,
            'n_background': self.n_background,
            'pct_down': self.pct_down,
            'direction_pvalue': self.direction_pvalue,
            'mannwhitney_pvalue': self.mannwhitney_pvalue,
            'variance_inflation_factor': self.variance_inflation_factor,
            'mean_pairwise_correlation': self.mean_pairwise_correlation,
        }

    def __getitem__(self, key: str):
        """Support dict-style access for backward compatibility."""
        return self.to_dict()[key]

    def get(self, key: str, default=None):
        """Support dict-style .get() for backward compatibility."""
        return self.to_dict().get(key, default)


class ModelType(Enum):
    """Type of statistical model."""

    FIXED = "fixed"   # Simple linear model (no random effects)
    MIXED = "mixed"   # Linear mixed model with random subject effects


@dataclass
class ContrastResult:
    """Result of a single contrast test.

    Attributes:
        contrast_name: Name/label of the contrast (e.g., "treatment_vs_control")
        log2_fc: Log2 fold change (effect size)
        se: Standard error of the estimate
        t_value: t-statistic
        df: Degrees of freedom
        p_value: Raw p-value (two-tailed)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
    """

    contrast_name: str
    log2_fc: float
    se: float
    t_value: float
    df: float
    p_value: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict:
        return {
            'contrast': self.contrast_name,
            'log2FC': self.log2_fc,
            'SE': self.se,
            'tvalue': self.t_value,
            'df': self.df,
            'pvalue': self.p_value,
            'CI_lower': self.ci_lower,
            'CI_upper': self.ci_upper,
        }


@dataclass
class ProteinResult:
    """Differential analysis result for a single protein/clique.

    Attributes:
        feature_id: Protein or clique identifier
        contrasts: List of contrast results
        model_type: Type of model fitted
        n_observations: Number of observations used
        residual_variance: Estimated residual variance
        subject_variance: Estimated subject random effect variance (if mixed)
        convergence: Whether model converged (for mixed models)
        issue: Warning or error message if any
    """

    feature_id: str
    contrasts: list[ContrastResult]
    model_type: ModelType
    n_observations: int
    residual_variance: float
    subject_variance: float | None = None
    convergence: bool = True
    issue: str | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per contrast."""
        rows = []
        for contrast in self.contrasts:
            row = contrast.to_dict()
            row['feature_id'] = self.feature_id
            row['model_type'] = self.model_type.value
            row['n_obs'] = self.n_observations
            row['residual_var'] = self.residual_variance
            row['subject_var'] = self.subject_variance
            row['converged'] = self.convergence
            row['issue'] = self.issue
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class DifferentialResult:
    """Complete differential analysis results.

    Attributes:
        results: List of per-protein/clique results
        contrasts_tested: Names of contrasts tested
        fdr_method: FDR correction method used
        fdr_threshold: FDR threshold for significance
    """

    results: list[ProteinResult]
    contrasts_tested: list[str]
    fdr_method: str = "BH"
    fdr_threshold: float = 0.05

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame."""
        dfs = [r.to_dataframe() for r in self.results]
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Apply FDR correction per contrast
        df['adj_pvalue'] = np.nan
        for contrast in self.contrasts_tested:
            mask = df['contrast'] == contrast
            if mask.any():
                pvals = df.loc[mask, 'pvalue'].values
                adj_pvals = fdr_correction(pvals, method=self.fdr_method)
                df.loc[mask, 'adj_pvalue'] = adj_pvals

        df['significant'] = df['adj_pvalue'] < self.fdr_threshold

        return df

    def significant_features(self, contrast: str | None = None) -> list[str]:
        """Get list of significant features."""
        df = self.to_dataframe()
        if contrast:
            df = df[df['contrast'] == contrast]
        return df[df['significant']]['feature_id'].unique().tolist()


def fdr_correction(
    pvalues: NDArray[np.float64],
    method: Literal["BH", "BY", "bonferroni"] = "BH",
    alpha: float = 0.05,
) -> NDArray[np.float64]:
    """
    Apply multiple testing correction.

    Args:
        pvalues: Array of raw p-values.
        method: Correction method:
            - "BH": Benjamini-Hochberg (controls FDR)
            - "BY": Benjamini-Yekutieli (controls FDR under dependence)
            - "bonferroni": Bonferroni (controls FWER)
        alpha: Significance threshold.

    Returns:
        Array of adjusted p-values.
    """
    from statsmodels.stats.multitest import multipletests

    # Handle NaN p-values
    valid_mask = ~np.isnan(pvalues)
    adj_pvals = np.full_like(pvalues, np.nan)

    if not np.any(valid_mask):
        return adj_pvals

    method_map = {"BH": "fdr_bh", "BY": "fdr_by", "bonferroni": "bonferroni"}
    _, adj_pvals[valid_mask], _, _ = multipletests(
        pvalues[valid_mask],
        alpha=alpha,
        method=method_map.get(method, method),
    )

    return adj_pvals


def build_contrast_matrix(
    conditions: list[str],
    contrasts: dict[str, tuple[str, str]] | None = None,
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Build contrast matrix for hypothesis testing.

    Args:
        conditions: List of unique condition names.
        contrasts: Dict mapping contrast name → (condition1, condition2).
            The contrast tests condition1 - condition2.
            If None, creates all pairwise contrasts.

    Returns:
        Tuple of (contrast_matrix, contrast_names).
        Matrix has shape (n_contrasts, n_conditions).
    """
    n_conditions = len(conditions)
    condition_to_idx = {c: i for i, c in enumerate(conditions)}

    if contrasts is None:
        # All pairwise contrasts
        contrasts = {}
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1:]:
                contrasts[f"{c1}_vs_{c2}"] = (c1, c2)

    contrast_matrix = np.zeros((len(contrasts), n_conditions))
    contrast_names = []

    for i, (name, (c1, c2)) in enumerate(contrasts.items()):
        if c1 not in condition_to_idx or c2 not in condition_to_idx:
            raise ValueError(f"Unknown condition in contrast {name}: {c1} or {c2}")
        contrast_matrix[i, condition_to_idx[c1]] = 1
        contrast_matrix[i, condition_to_idx[c2]] = -1
        contrast_names.append(name)

    return contrast_matrix, contrast_names


def satterthwaite_df(
    contrast_vector: NDArray[np.float64],
    cov_beta: NDArray[np.float64],
    residual_var: float,
    subject_var: float,
    n_groups: int,
    n_obs: int,
    use_mlx: bool = True,
) -> float | None:
    """
    Compute Satterthwaite degrees of freedom for a contrast in a mixed model.

    The Satterthwaite approximation estimates the degrees of freedom for a linear
    combination of variance components. For a contrast c'β, the formula is:

        df = 2 * (V_c)² / Var(V_c)

    where V_c = c' Cov(β) c is the variance of the contrast estimate.

    The variance of V_c depends on the variance components (residual and random effects),
    and we use the delta method to approximate it from the uncertainty in the variance
    component estimates.

    For a mixed model with random intercepts:
        y ~ X*β + Z*u + ε
        where u ~ N(0, σ_u²), ε ~ N(0, σ²)

    The variance of the contrast is:
        V_c = c' (X'V⁻¹X)⁻¹ c
        where V = Z*σ_u²*Z' + σ²*I

    The Satterthwaite df accounts for the uncertainty in estimating σ² and σ_u².

    Args:
        contrast_vector: Contrast vector in parameter space (length = n_fixed_effects).
        cov_beta: Covariance matrix of fixed effect estimates.
        residual_var: Estimated residual variance (σ²).
        subject_var: Estimated random effect variance (σ_u²).
        n_groups: Number of groups (subjects).
        n_obs: Total number of observations.
        use_mlx: Whether to use MLX for GPU-accelerated computation.

    Returns:
        Satterthwaite degrees of freedom, or None if computation fails.

    References:
        - Satterthwaite, F.E. (1946). Biometrics Bulletin 2(6):110-114.
        - Giesbrecht & Burns (1985). Commun. Stat. Theory Methods 14(4):989-1001.
        - Fai & Cornelius (1996). J. Am. Stat. Assoc. 91(434):814-821.
    """
    try:
        # Use MLX for GPU acceleration if available and requested
        if use_mlx and MLX_AVAILABLE and cov_beta.size > 16:
            # Convert to MLX arrays for GPU computation
            c_mx = mx.array(contrast_vector, dtype=mx.float32)
            cov_mx = mx.array(cov_beta, dtype=mx.float32)

            # Contrast variance: V_c = c' * Cov(β) * c
            V_c = mx.matmul(mx.matmul(c_mx, cov_mx), c_mx)
            V_c = float(V_c)
        else:
            # CPU computation with numpy
            V_c = float(contrast_vector @ cov_beta @ contrast_vector)

        if V_c <= 0 or not np.isfinite(V_c):
            return None

        # Satterthwaite-Welch degrees of freedom approximation
        # For a mixed model: y ~ X*β + Z*u + ε
        # where u ~ N(0, σ_u²I), ε ~ N(0, σ²I)
        #
        # The variance of a contrast c'β has two sources:
        # 1. Within-group (residual) variance: σ²
        # 2. Between-group (random effect) variance: σ_u²
        #
        # The Satterthwaite formula is:
        #   df = V² / Σ(V_i² / df_i)
        # where V_i are variance components and df_i are their degrees of freedom

        n_params = len(contrast_vector)

        # Degrees of freedom for each variance component
        df_residual = max(n_obs - n_params, 1)
        df_random = max(n_groups - 1, 1)

        # Average group size (for balanced approximation)
        avg_group_size = n_obs / n_groups if n_groups > 0 else 1

        # Decompose the contrast variance into components
        # For a contrast between two condition means in a mixed model:
        # V_c has contributions from:
        # - Within-group variance: σ²/n_per_group (variance of group mean)
        # - Between-group variance: σ_u² (random effect variance)

        # Scale V_c to get the theoretical variance from variance components
        # V_theoretical = σ²/avg_n + σ_u²
        v_within = residual_var / avg_group_size
        v_between = subject_var
        v_theoretical = v_within + v_between

        # Handle edge case where theoretical variance is zero
        if v_theoretical <= 0:
            return None

        # Scale factor to match V_c with theoretical variance
        # (V_c from cov_beta may differ due to estimation, design imbalance, etc.)
        scale = V_c / v_theoretical if v_theoretical > 0 else 1.0

        # Scaled variance components
        v_within_scaled = v_within * scale
        v_between_scaled = v_between * scale

        # Satterthwaite formula: df = V² / Σ(V_i² / df_i)
        denominator = (v_within_scaled ** 2) / df_residual + (v_between_scaled ** 2) / df_random

        if denominator <= 0 or not np.isfinite(denominator):
            return None

        df_satterthwaite = (V_c ** 2) / denominator

        # Sanity bounds: df should be between 1 and n_obs - 1
        # (We use n_obs - 1 as upper bound, not n_obs - n_params, because
        # Satterthwaite can exceed residual df when random effects dominate)
        df_lower = 1.0
        df_upper = float(n_obs - 1)

        df_satterthwaite = np.clip(df_satterthwaite, df_lower, df_upper)

        return float(df_satterthwaite)

    except Exception as e:
        warnings.warn(f"Satterthwaite df computation failed: {e}")
        return None


def batched_ols_gpu(
    Y: NDArray[np.float64],
    X: NDArray[np.float64],
    conditions: list[str],
    feature_ids: list[str],
    contrast_matrix: NDArray[np.float64],
    contrast_names: list[str],
) -> list[ProteinResult]:
    """
    Batched OLS regression for all features using GPU acceleration.

    Solves: Y = X @ β + ε for all features simultaneously.
    Uses MLX for GPU acceleration of matrix operations.

    Args:
        Y: Response matrix (n_samples, n_features) of log2 intensities.
        X: Design matrix (n_samples, n_params) with dummy coding.
        conditions: List of condition names (for reference).
        feature_ids: List of feature identifiers.
        contrast_matrix: Contrast matrix (n_contrasts, n_conditions).
        contrast_names: Names of contrasts.

    Returns:
        List of ProteinResult objects with test statistics.
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx")

    n_samples, n_features = Y.shape
    n_params = X.shape[1]
    n_conditions = len(conditions)

    # Convert to MLX arrays
    Y_mx = mx.array(Y)
    X_mx = mx.array(X)

    # Handle NaN values by creating masks
    # For simplicity, we'll process all features together and mask results
    nan_mask = np.isnan(Y)  # (n_samples, n_features)

    # Replace NaN with 0 for computation (will be masked out)
    Y_clean = np.where(nan_mask, 0.0, Y)
    Y_mx_clean = mx.array(Y_clean)

    # Compute (X'X)^-1
    # Note: MLX linalg.inv may require CPU stream, so we compute on CPU and convert back
    XtX = X_mx.T @ X_mx
    try:
        # Convert to numpy for inversion (MLX inv may not support GPU)
        XtX_np = np.array(XtX)
        XtX_inv_np = np.linalg.inv(XtX_np)
        XtX_inv = mx.array(XtX_inv_np)
    except np.linalg.LinAlgError as e:
        # Singular matrix - fall back to sequential
        raise ValueError(f"Singular design matrix: {e}")

    # Solve for coefficients: β = (X'X)^-1 X'Y
    # Shape: (n_params, n_features)
    beta = XtX_inv @ (X_mx.T @ Y_mx_clean)

    # Predictions and residuals
    Y_pred = X_mx @ beta
    residuals = Y_mx_clean - Y_pred

    # Residual sum of squares (RSS) per feature
    # Mask out NaN positions before computing RSS
    valid_mask_mx = mx.array(~nan_mask)  # (n_samples, n_features)
    residuals_masked = residuals * valid_mask_mx
    rss = mx.sum(residuals_masked ** 2, axis=0)  # (n_features,)

    # Count valid observations per feature
    n_valid = mx.sum(valid_mask_mx, axis=0)  # (n_features,)

    # Residual degrees of freedom: n_valid - n_params
    df_resid = n_valid - n_params

    # Residual variance: σ² = RSS / df_resid
    residual_var = rss / mx.maximum(df_resid, 1.0)  # Avoid division by zero

    # Standard errors: SE(β) = sqrt(σ² * diag((X'X)^-1))
    # Diagonal of (X'X)^-1
    XtX_inv_diag = mx.diagonal(XtX_inv)  # (n_params,)

    # SE matrix: (n_params, n_features)
    # Each feature has variance residual_var[j], so SE[i,j] = sqrt(residual_var[j] * XtX_inv_diag[i])
    se_matrix = mx.sqrt(residual_var[None, :] * XtX_inv_diag[:, None])

    # Convert back to numpy for contrast computation
    beta_np = np.array(beta)  # (n_params, n_features)
    se_np = np.array(se_matrix)  # (n_params, n_features)
    residual_var_np = np.array(residual_var)  # (n_features,)
    df_resid_np = np.array(df_resid)  # (n_features,)
    n_valid_np = np.array(n_valid)  # (n_features,)
    XtX_inv_np = np.array(XtX_inv)  # (n_params, n_params)

    # Build transformation matrix from coefficients to condition means
    # For dummy coding: condition_mean[i] = intercept (i=0) or intercept + coef[i] (i>0)
    L = np.zeros((n_conditions, n_params))
    L[:, 0] = 1.0  # All conditions include intercept
    for i in range(1, min(n_conditions, n_params)):
        L[i, i] = 1.0  # Add dummy coefficient for non-reference conditions

    # Process each feature to create ProteinResult
    results = []
    for j in range(n_features):
        feature_id = feature_ids[j]
        beta_j = beta_np[:, j]  # (n_params,)
        se_j = se_np[:, j]  # (n_params,)
        res_var_j = residual_var_np[j]
        df_j = max(int(df_resid_np[j]), 1)
        n_obs_j = int(n_valid_np[j])

        # Check for insufficient data
        if n_obs_j < 3 or df_j < 1:
            results.append(ProteinResult(
                feature_id=feature_id,
                contrasts=[],
                model_type=ModelType.FIXED,
                n_observations=n_obs_j,
                residual_variance=np.nan,
                subject_variance=None,
                convergence=False,
                issue="Insufficient data after removing NaN",
            ))
            continue

        # Check for numerical issues
        if not np.isfinite(res_var_j) or res_var_j < 1e-20:
            results.append(ProteinResult(
                feature_id=feature_id,
                contrasts=[],
                model_type=ModelType.FIXED,
                n_observations=n_obs_j,
                residual_variance=res_var_j,
                subject_variance=None,
                convergence=False,
                issue="Near-zero or invalid residual variance",
            ))
            continue

        # Compute condition means: means = L @ beta
        condition_means = L @ beta_j  # (n_conditions,)

        # Test each contrast
        contrast_results = []
        for contrast_vec, contrast_name in zip(contrast_matrix, contrast_names):
            # Estimate = contrast' * condition_means
            estimate = np.dot(contrast_vec, condition_means)

            # Transform contrast to parameter space: c = L' * contrast_vec
            c_param = L.T @ contrast_vec  # (n_params,)

            # Standard error: SE = sqrt(c' * Cov(β) * c)
            # where Cov(β) = σ² * (X'X)^-1
            se_squared = res_var_j * (c_param @ XtX_inv_np @ c_param)

            if se_squared < 0:
                warnings.warn(f"Negative variance for {feature_id}, contrast {contrast_name}")
                se_squared = abs(se_squared)

            se = np.sqrt(se_squared)
            se = max(se, 1e-10)  # Prevent division by zero

            # t-statistic and p-value
            t_value = estimate / se
            p_value = 2 * scipy_stats.t.sf(np.abs(t_value), df_j)

            # Confidence interval
            t_crit = scipy_stats.t.ppf(0.975, df_j)
            ci_lower = estimate - t_crit * se
            ci_upper = estimate + t_crit * se

            contrast_results.append(ContrastResult(
                contrast_name=contrast_name,
                log2_fc=float(estimate),
                se=float(se),
                t_value=float(t_value),
                df=float(df_j),
                p_value=float(p_value),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
            ))

        results.append(ProteinResult(
            feature_id=feature_id,
            contrasts=contrast_results,
            model_type=ModelType.FIXED,
            n_observations=n_obs_j,
            residual_variance=float(res_var_j),
            subject_variance=None,
            convergence=True,
            issue=None,
        ))

    return results


def fit_linear_model(
    y: NDArray[np.float64],
    condition: NDArray | pd.Series,
    subject: NDArray | pd.Series | None = None,
    use_mixed: bool = True,
    conditions: list[str] | None = None,
) -> tuple[pd.DataFrame, ModelType, float, float | None, bool, str | None, NDArray[np.float64] | None, int, int, int]:
    """
    Fit linear model (fixed or mixed effects).

    Args:
        y: Response variable (log2 intensities).
        condition: Condition labels for each observation.
        subject: Subject IDs for random effects (optional).
        use_mixed: Whether to attempt mixed model if subject provided.
        conditions: Ordered list of condition names (ensures consistent reference level).

    Returns:
        Tuple of (coefficients_df, model_type, residual_var, subject_var, converged, issue, cov_params, residual_df, n_obs_used, n_groups).
        cov_params is the covariance matrix of the coefficient estimates.
        residual_df is the residual degrees of freedom.
        n_obs_used is the number of observations after dropping NaN.
        n_groups is the number of groups (subjects) for mixed models, 0 for fixed models.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Prepare data - use explicit category order if provided to ensure consistent reference level
    if conditions is not None:
        condition_cat = pd.Categorical(condition, categories=conditions)
    else:
        condition_cat = pd.Categorical(condition)

    df = pd.DataFrame({
        'y': y,
        'condition': condition_cat,
    })

    # Add subject before dropna to track which samples survive
    if subject is not None:
        df['subject'] = pd.Categorical(subject)

    # Remove missing values - this preserves the alignment of all columns
    initial_length = len(df)
    df = df.dropna()

    n_obs_used = len(df)

    if n_obs_used < 3:
        return None, ModelType.FIXED, np.nan, None, False, "Insufficient data", None, 0, n_obs_used, 0

    # Check if we can fit mixed model
    can_fit_mixed = False
    if subject is not None and use_mixed and 'subject' in df.columns:
        # Need multiple observations per subject for random effects
        obs_per_subject = df.groupby('subject', observed=True).size()
        if (obs_per_subject > 1).any() and len(df['subject'].unique()) > 1:
            can_fit_mixed = True

    # Attempt mixed model first
    mixed_failed_reason = None
    if can_fit_mixed:
        try:
            model = smf.mixedlm("y ~ condition", df, groups=df["subject"])
            result = model.fit(reml=True, method='powell')

            if result.converged:
                # Extract coefficients
                coef_df = pd.DataFrame({
                    'coef': result.fe_params,
                    'se': result.bse_fe,
                })

                residual_var = result.scale
                subject_var = float(result.cov_re.iloc[0, 0]) if hasattr(result, 'cov_re') else None

                # Get covariance matrix of fixed effects only (excludes random effect parameters)
                # Mixed model cov_params includes random effect variances, so extract FE portion
                n_fixed = len(result.fe_params)
                cov_params_full = result.cov_params()
                cov_params = cov_params_full.iloc[:n_fixed, :n_fixed].values

                # For mixed models, compute naive between-within approximation as fallback
                # This is used if Satterthwaite df computation fails
                n_groups = len(df['subject'].unique())
                residual_df = max(n_groups - n_fixed, len(df) - n_fixed - 1)

                # NOTE: The actual Satterthwaite df will be computed per-contrast in test_contrasts()
                # We pass n_groups so that Satterthwaite can be computed there
                return coef_df, ModelType.MIXED, residual_var, subject_var, True, None, cov_params, residual_df, n_obs_used, n_groups
            else:
                mixed_failed_reason = "Mixed model did not converge"

        except Exception as e:
            # Fall back to fixed model
            mixed_failed_reason = f"Mixed model failed: {type(e).__name__}"

    # Fit fixed effects model
    try:
        # PSEUDOREPLICATION FIX: If mixed model was attempted but failed,
        # and we have repeated measures (subject column exists with replicates),
        # aggregate to subject level to avoid treating replicates as independent
        aggregated = False
        if can_fit_mixed and 'subject' in df.columns:
            # Aggregate to subject level: mean per subject per condition
            df_agg = df.groupby(['subject', 'condition'], observed=True, as_index=False).agg({'y': 'mean'})
            df = df_agg
            aggregated = True
            n_obs_used = len(df)  # Update observation count after aggregation

        # Create design matrix with dummy coding
        X = pd.get_dummies(df['condition'], drop_first=True, dtype=float)
        X = sm.add_constant(X)

        model = sm.OLS(df['y'], X)
        result = model.fit()

        coef_df = pd.DataFrame({
            'coef': result.params,
            'se': result.bse,
        })

        residual_var = result.scale

        # Get covariance matrix of coefficients as numpy array
        cov_params = result.cov_params().values

        # Residual degrees of freedom for OLS: n - p
        residual_df = result.df_resid

        # Set issue message if we fell back from mixed model
        issue = None
        if mixed_failed_reason is not None:
            issue = f"Fallback to fixed model: {mixed_failed_reason}"
            if aggregated:
                issue += "; Aggregated to subject level due to mixed model failure"

        # For fixed models, n_groups is 0 (no grouping structure)
        return coef_df, ModelType.FIXED, residual_var, None, True, issue, cov_params, residual_df, n_obs_used, 0

    except Exception as e:
        return None, ModelType.FIXED, np.nan, None, False, str(e), None, 0, n_obs_used, 0


def test_contrasts(
    coef_df: pd.DataFrame,
    conditions: list[str],
    contrast_matrix: NDArray[np.float64],
    contrast_names: list[str],
    residual_var: float,
    n_obs: int,
    model_type: ModelType,
    cov_params: NDArray[np.float64],
    residual_df: int,
    subject_var: float | None = None,
    n_groups: int = 0,
) -> list[ContrastResult]:
    """
    Test contrasts using fitted model coefficients.

    Args:
        coef_df: DataFrame with 'coef' and 'se' columns.
        conditions: List of condition names.
        contrast_matrix: Contrast matrix (n_contrasts × n_conditions).
        contrast_names: Names for each contrast.
        residual_var: Residual variance from model.
        n_obs: Number of observations.
        model_type: Type of model fitted.
        cov_params: Covariance matrix of coefficient estimates (from fitted model).
        residual_df: Residual degrees of freedom from model (fallback if Satterthwaite fails).
        subject_var: Subject random effect variance (for mixed models).
        n_groups: Number of groups/subjects (for mixed models).

    Returns:
        List of ContrastResult objects.
    """
    results = []

    # Get condition means from coefficients
    # Model parameterization: intercept + condition effects (dummy coded)
    n_conditions = len(conditions)

    # Reconstruct condition means
    # Intercept = mean of reference condition
    # Other coefficients = difference from reference
    try:
        intercept = coef_df.loc['const', 'coef'] if 'const' in coef_df.index else coef_df.iloc[0]['coef']
    except:
        intercept = coef_df['coef'].iloc[0]

    condition_means = np.zeros(n_conditions)
    condition_ses = np.zeros(n_conditions)

    # Reference condition (first) gets intercept
    condition_means[0] = intercept

    # Other conditions get intercept + their coefficient
    for i, cond in enumerate(conditions[1:], 1):
        coef_name = f"condition_{cond}" if f"condition_{cond}" in coef_df.index else f"condition[T.{cond}]"
        # Try various naming conventions
        found = False
        for name in [coef_name, f"condition_{cond}", f"C(condition)[T.{cond}]", cond]:
            if name in coef_df.index:
                condition_means[i] = intercept + coef_df.loc[name, 'coef']
                condition_ses[i] = coef_df.loc[name, 'se']
                found = True
                break

        if not found:
            # Fallback: use coefficient by position
            if i < len(coef_df):
                condition_means[i] = intercept + coef_df.iloc[i]['coef']
                condition_ses[i] = coef_df.iloc[i]['se']

    # Build transformation matrix from coefficients to condition means
    # For dummy coding: condition_mean[i] = intercept (i=0) or intercept + coef[i] (i>0)
    n_params = len(coef_df)
    L = np.zeros((n_conditions, n_params))
    L[:, 0] = 1.0  # All conditions include intercept
    for i in range(1, n_conditions):
        if i < n_params:
            L[i, i] = 1.0  # Add dummy coefficient for non-reference conditions

    # Test each contrast
    for contrast_vec, contrast_name in zip(contrast_matrix, contrast_names):
        # Estimate = contrast' * condition_means = contrast' * L * beta
        # where beta is the coefficient vector
        estimate = np.dot(contrast_vec, condition_means)

        # Compute contrast vector in parameter space: c = L' * contrast_vec
        # SE(contrast' * means) = SE(c' * beta) = sqrt(c' * Cov(beta) * c)
        c_param = L.T @ contrast_vec  # Transform contrast to parameter space

        # Proper SE calculation using covariance matrix: SE = sqrt(c' * Cov * c)
        se_squared = c_param @ cov_params @ c_param

        if se_squared < 0:
            # This shouldn't happen, but protect against numerical issues
            warnings.warn(f"Negative variance for contrast {contrast_name}, using absolute value")
            se_squared = abs(se_squared)

        se = np.sqrt(se_squared)

        if se < 1e-10:
            se = 1e-10  # Prevent division by zero

        # t-statistic
        t_value = estimate / se

        # Compute degrees of freedom
        # For mixed models, use Satterthwaite approximation; for fixed models, use residual df
        df = residual_df  # Default fallback

        if model_type == ModelType.MIXED and subject_var is not None and n_groups > 0:
            # Attempt Satterthwaite degrees of freedom for mixed models
            df_satt = satterthwaite_df(
                contrast_vector=c_param,
                cov_beta=cov_params,
                residual_var=residual_var,
                subject_var=subject_var,
                n_groups=n_groups,
                n_obs=n_obs,
                use_mlx=True,
            )

            # Use Satterthwaite df if successfully computed, otherwise fall back
            if df_satt is not None and np.isfinite(df_satt):
                df = df_satt

        df = max(df, 1)  # Ensure positive df

        # p-value (two-tailed)
        p_value = 2 * scipy_stats.t.sf(np.abs(t_value), df)

        # 95% confidence interval
        t_crit = scipy_stats.t.ppf(0.975, df)
        ci_lower = estimate - t_crit * se
        ci_upper = estimate + t_crit * se

        results.append(ContrastResult(
            contrast_name=contrast_name,
            log2_fc=float(estimate),
            se=float(se),
            t_value=float(t_value),
            df=float(df),
            p_value=float(p_value),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        ))

    return results


def differential_analysis_single(
    intensities: NDArray[np.float64],
    condition: NDArray | pd.Series,
    subject: NDArray | pd.Series | None,
    feature_id: str,
    contrast_matrix: NDArray[np.float64],
    contrast_names: list[str],
    conditions: list[str],
    use_mixed: bool = True,
) -> ProteinResult:
    """
    Run differential analysis for a single protein/clique.

    Args:
        intensities: Log2 intensity values.
        condition: Condition labels.
        subject: Subject IDs (optional).
        feature_id: Identifier for this feature.
        contrast_matrix: Contrast matrix for testing.
        contrast_names: Names of contrasts.
        conditions: List of condition names (sorted, for consistent reference level).
        use_mixed: Whether to use mixed model.

    Returns:
        ProteinResult with test statistics.
    """
    # Fit model - pass conditions to ensure consistent reference level
    coef_df, model_type, residual_var, subject_var, converged, issue, cov_params, residual_df, n_obs_used, n_groups = fit_linear_model(
        intensities, condition, subject, use_mixed, conditions=conditions
    )

    if coef_df is None or not converged:
        # Return empty result with error
        return ProteinResult(
            feature_id=feature_id,
            contrasts=[],
            model_type=model_type,
            n_observations=n_obs_used,  # Use actual observations modeled
            residual_variance=np.nan,
            subject_variance=None,
            convergence=False,
            issue=issue or "Model fitting failed",
        )

    # Test contrasts
    contrast_results = test_contrasts(
        coef_df=coef_df,
        conditions=conditions,
        contrast_matrix=contrast_matrix,
        contrast_names=contrast_names,
        residual_var=residual_var,
        n_obs=n_obs_used,  # Use actual observations modeled
        model_type=model_type,
        cov_params=cov_params,
        residual_df=residual_df,
        subject_var=subject_var,
        n_groups=n_groups,
    )

    return ProteinResult(
        feature_id=feature_id,
        contrasts=contrast_results,
        model_type=model_type,
        n_observations=n_obs_used,  # Use actual observations modeled
        residual_variance=residual_var,
        subject_variance=subject_var,
        convergence=converged,
        issue=issue,
    )


def run_protein_differential(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_condition: NDArray | pd.Series,
    contrast: tuple[str, str],
    eb_moderation: bool = True,
    target_gene_ids: list[str] | None = None,
    verbose: bool = False,
    covariates_df: pd.DataFrame | None = None,
    covariate_design: "CovariateDesign | None" = None,
) -> pd.DataFrame:
    """
    Genome-wide Empirical Bayes differential expression for individual proteins.

    Runs OLS regression (y ~ condition [+ covariates]) for each protein and
    applies limma-style Empirical Bayes variance shrinkage for moderated
    t-statistics. This is the protein-level analogue of the clique-level
    permutation test.

    Statistical Model:
        For each protein i:
        1. OLS: y_i ~ β₀ + β₁ × condition [+ γ × covariates] + ε
        2. Sample variance: σ²_i = RSS_i / (n - p)
        3. EB shrinkage: σ²_post,i = (d₀ × s₀² + df × σ²_i) / (d₀ + df)
        4. Moderated t: t_i = β₁ / sqrt(σ²_post,i × c_var)
        5. P-value: t-distribution with df_total = d₀ + df

    The EB machinery (fit_f_dist, squeeze_var) is reused from permutation_gpu.py.

    Args:
        data: Protein expression matrix (n_features, n_samples) of log2 intensities.
        feature_ids: List of protein/feature identifiers.
        sample_condition: Condition labels for each sample.
        contrast: Tuple of (condition1, condition2) to test condition1 - condition2.
        eb_moderation: If True, apply Empirical Bayes variance shrinkage (limma-style).
                       If False, use standard OLS t-statistics.
        target_gene_ids: Optional list of target gene IDs to flag in results.
        verbose: Print progress information.
        covariates_df: Optional DataFrame of covariates (one row per sample).
            When provided, covariates are included in the design matrix as
            nuisance parameters. The contrast tests only the condition effect,
            adjusting for covariates (e.g., Sex, Batch).
        covariate_design: Optional pre-built CovariateDesign from
            design_matrix.build_covariate_design_matrix(). When provided,
            its sample_mask is used as the authoritative NaN mask instead
            of recomputing independently (M-6 consolidation).

    Returns:
        DataFrame with columns:
        - feature_id: Protein/feature identifier
        - log2fc: Log2 fold change (condition1 - condition2)
        - t_statistic: Moderated t-statistic (or standard if eb_moderation=False)
        - p_value: Two-tailed p-value
        - df: Degrees of freedom (moderated if EB, residual if standard)
        - sigma2_post: Posterior variance (after EB shrinkage, if enabled)
        - sigma2: Sample variance (before shrinkage)
        - n_samples: Number of valid samples used
        - is_target: Boolean flag (True if feature_id in target_gene_ids)

    Example:
        >>> # Run EB-moderated differential on all proteins
        >>> result = run_protein_differential(
        ...     data=protein_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_condition=metadata['treatment_group'],
        ...     contrast=('treatment', 'control'),
        ...     eb_moderation=True,
        ... )
        >>> # Get significant proteins
        >>> sig = result[result['p_value'] < 0.05].sort_values('p_value')

    References:
        - Smyth (2004) Statistical Applications in Genetics and Molecular Biology
        - Ritchie et al. (2015) Nucleic Acids Research 43(7):e47 (limma)
    """
    from .permutation_gpu import fit_f_dist, squeeze_var

    n_features, n_samples = data.shape

    if len(feature_ids) != n_features:
        raise ValueError(f"feature_ids length ({len(feature_ids)}) != n_features ({n_features})")

    if len(sample_condition) != n_samples:
        raise ValueError(f"sample_condition length ({len(sample_condition)}) != n_samples ({n_samples})")

    # Get conditions and validate contrast
    conditions = sorted(pd.Series(sample_condition).dropna().unique().tolist())

    if len(conditions) < 2:
        raise ValueError(f"Need at least 2 conditions, got {len(conditions)}")

    if contrast[0] not in conditions or contrast[1] not in conditions:
        raise ValueError(f"Contrast {contrast} not found in conditions {conditions}")

    cov_label = ""
    if covariates_df is not None and len(covariates_df.columns) > 0:
        cov_label = f" + covariates: {list(covariates_df.columns)}"

    if verbose:
        print(f"Protein-level differential analysis")
        print(f"  Features: {n_features}")
        print(f"  Samples: {n_samples}")
        print(f"  Contrast: {contrast[0]} vs {contrast[1]}{cov_label}")
        print(f"  EB moderation: {'enabled' if eb_moderation else 'disabled'}")

    # Precompute design matrix and OLS components.
    # When covariate_design is provided, it is passed through so that
    # precompute_ols_matrices() uses the design's X, contrast, and
    # sample_mask directly — eliminating the F8 latent bug where
    # independently built masks could diverge.
    from .permutation_gpu import precompute_ols_matrices

    matrices = precompute_ols_matrices(
        sample_condition=sample_condition,
        conditions=conditions,
        contrast=contrast,
        regularization=1e-8,
        covariates_df=covariates_df,
        covariate_design=covariate_design,
    )

    # Filter data to valid samples (those included in design matrix)
    # When a pre-built CovariateDesign is provided, use its sample_mask
    # as the authoritative subset (M-6: single NaN mask source of truth).
    # Otherwise, recompute to match what precompute_ols_matrices used.
    if covariate_design is not None:
        valid_mask = covariate_design.sample_mask
    else:
        condition_cat = pd.Categorical(sample_condition, categories=conditions)
        valid_mask = ~pd.isna(condition_cat)
        if covariates_df is not None and len(covariates_df.columns) > 0:
            cov_valid = ~covariates_df.isna().any(axis=1).values
            valid_mask = valid_mask & cov_valid
    data_valid = data[:, valid_mask]
    n_valid = np.sum(valid_mask)

    # Check group size imbalance (Satterthwaite df warning)
    valid_conditions = np.asarray(sample_condition)[valid_mask]
    group_a_mask = valid_conditions == contrast[0]
    group_b_mask = valid_conditions == contrast[1]
    n_a = int(np.sum(group_a_mask))
    n_b = int(np.sum(group_b_mask))
    if n_a > 0 and n_b > 0:
        ratio = max(n_a, n_b) / min(n_a, n_b)
        if ratio > 3.0:
            warnings.warn(
                f"Sample group sizes are highly imbalanced ({n_a} vs {n_b}). "
                f"Satterthwaite df correction may be more appropriate than "
                f"pooled variance.",
                UserWarning,
                stacklevel=2,
            )

    if verbose:
        print(f"  Valid samples: {n_valid}")
        print(f"  Residual df: {matrices.df_residual}")

    # Handle NaN values per protein - compute valid sample count
    nan_mask = np.isnan(data_valid)  # (n_features, n_valid)
    n_valid_per_feature = np.sum(~nan_mask, axis=1)  # (n_features,)

    # Replace NaN with 0 for computation (will handle in variance calculation)
    data_clean = np.where(nan_mask, 0.0, data_valid)

    # Compute OLS coefficients for all proteins: β = Y @ X @ (X'X)^-1'
    # Y shape: (n_features, n_samples)
    # X shape: (n_samples, n_params)
    # β shape: (n_features, n_params)
    beta = data_clean @ matrices.X @ matrices.XtX_inv.T

    # Predictions: Ŷ = β @ X'
    Y_pred = beta @ matrices.X.T

    # Residuals: e = Y - Ŷ
    residuals = data_clean - Y_pred

    # Mask out NaN positions for RSS computation
    residuals_masked = np.where(nan_mask, 0.0, residuals)

    # Residual sum of squares per feature
    rss = np.sum(residuals_masked ** 2, axis=1)  # (n_features,)

    # Residual variance: σ² = RSS / df
    # Use per-feature df if some proteins have missing values
    df_per_feature = n_valid_per_feature - matrices.X.shape[1]
    df_per_feature = np.maximum(df_per_feature, 1)  # Avoid division by zero

    sigma2 = rss / df_per_feature

    # Filter to features with valid variance estimates
    valid_features = (sigma2 > 0) & np.isfinite(sigma2) & (n_valid_per_feature >= 3)

    if verbose:
        n_invalid = np.sum(~valid_features)
        if n_invalid > 0:
            print(f"  Excluded {n_invalid} features with invalid variance or insufficient data")

    # Apply Empirical Bayes variance shrinkage
    if eb_moderation:
        # Estimate EB hyperparameters from valid features only
        sigma2_valid = sigma2[valid_features]

        if len(sigma2_valid) < 3:
            warnings.warn("Insufficient features for EB estimation, disabling moderation")
            eb_moderation = False
            d0 = np.inf
            s0_sq = 1.0
            sigma2_post = sigma2.copy()
            df_total = float(matrices.df_residual)
        else:
            d0, s0_sq = fit_f_dist(sigma2_valid, matrices.df_residual)

            if verbose:
                if np.isinf(d0):
                    print(f"  EB priors: d0=Inf (no shrinkage), s0²={s0_sq:.6f}")
                else:
                    print(f"  EB priors: d0={d0:.2f}, s0²={s0_sq:.6f}")
                    shrinkage_weight = d0 / (d0 + matrices.df_residual)
                    print(f"  Shrinkage: {100*shrinkage_weight:.1f}% prior, "
                          f"{100*(1-shrinkage_weight):.1f}% sample")

            # Apply shrinkage to ALL features (including those filtered out)
            sigma2_post, df_total = squeeze_var(sigma2, matrices.df_residual, d0, s0_sq)
    else:
        # No EB moderation - use standard OLS
        sigma2_post = sigma2.copy()
        df_total = float(matrices.df_residual)
        d0 = np.inf
        s0_sq = np.nan

    # Compute log2FC from contrast: log2FC = β @ c
    log2fc = beta @ matrices.c

    # Standard error: SE = sqrt(σ²_post × c_var_factor)
    se = np.sqrt(sigma2_post * matrices.c_var_factor)
    se = np.maximum(se, 1e-10)  # Prevent division by zero

    # Moderated t-statistic: t = log2FC / SE
    t_statistic = log2fc / se

    # P-values from t-distribution with moderated df
    if eb_moderation and not np.isinf(d0):
        p_value = 2 * scipy_stats.t.sf(np.abs(t_statistic), df_total)
    else:
        # Use normal approximation for large df or when EB disabled
        p_value = 2 * scipy_stats.norm.sf(np.abs(t_statistic))

    # Build results DataFrame
    results = pd.DataFrame({
        'feature_id': feature_ids,
        'log2fc': log2fc,
        't_statistic': t_statistic,
        'p_value': p_value,
        'df': df_total if not np.isinf(d0) else matrices.df_residual,
        'sigma2_post': sigma2_post,
        'sigma2': sigma2,
        'n_samples': n_valid_per_feature,
    })

    # Flag target genes if provided
    if target_gene_ids is not None:
        target_set = set(target_gene_ids)
        results['is_target'] = results['feature_id'].isin(target_set)
    else:
        results['is_target'] = False

    # Mark invalid features with NaN p-values
    results.loc[~valid_features, 'p_value'] = np.nan

    if verbose:
        n_tested = np.sum(valid_features)
        print(f"  Tested: {n_tested} features")
        if target_gene_ids is not None:
            n_targets = np.sum(results['is_target'])
            print(f"  Target genes: {n_targets}")

    return results


def run_network_enrichment_test(
    protein_results: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int | None = None,
    verbose: bool = True,
) -> NetworkEnrichmentResult:
    """
    Competitive permutation test for network target enrichment.

    Tests whether network targets (is_target=True) have systematically
    higher |t-statistics| than random protein sets of the same size.

    This implements a competitive gene set enrichment approach where we ask:
    "Are the network targets enriched for differential expression compared
    to random protein sets of the same size from the measured proteome?"

    Statistical Method:
        1. Compute observed mean |t| for network targets
        2. Generate null distribution by sampling random sets of same size
        3. Compute z-score and empirical p-value
        4. Add directional bias test (binomial) and Mann-Whitney U test

    Args:
        protein_results: DataFrame from run_protein_differential() with columns:
            - t_statistic: EB-moderated t-statistics
            - is_target: Boolean flag for network targets
        n_permutations: Number of random sets for null distribution
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        NetworkEnrichmentResult dataclass with attributes:
            - observed_mean_abs_t: Mean |t| for network targets
            - null_mean: Mean of null distribution
            - null_std: Std of null distribution
            - z_score: (observed - null_mean) / null_std
            - empirical_pvalue: One-sided p-value (targets > random)
            - n_targets: Number of network targets
            - n_background: Number of background proteins
            - pct_down: Percentage of targets with negative t-statistic
            - direction_pvalue: Binomial test for directional bias
            - mannwhitney_pvalue: Mann-Whitney U test on |t| distributions

        Supports both attribute access (result.z_score) and dict-style
        access (result["z_score"]) for backward compatibility. Use
        result.to_dict() for JSON serialization.

    Example:
        >>> # Run protein differential first
        >>> protein_results = run_protein_differential(
        ...     data=intensity_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_condition=metadata['phenotype'],
        ...     target_gene_ids=c9orf72_targets,  # From INDRA query
        ... )
        >>> # Test network enrichment
        >>> enrichment = run_network_enrichment_test(protein_results)
        >>> print(f"Z-score: {enrichment.z_score:.2f}")
        >>> print(f"P-value: {enrichment.empirical_pvalue:.2e}")

    References:
        - GSEA: Subramanian et al., PNAS 2005
        - Camera: Wu & Smyth, NAR 2012
    """
    # Validate input
    required_cols = {'t_statistic', 'is_target'}
    if not required_cols.issubset(protein_results.columns):
        missing = required_cols - set(protein_results.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to valid (non-NaN) t-statistics
    valid_mask = ~protein_results['t_statistic'].isna()
    valid_results = protein_results.loc[valid_mask].copy()

    if len(valid_results) == 0:
        raise ValueError("No valid t-statistics in protein_results")

    # Extract arrays for vectorized operations
    all_t_stats = valid_results['t_statistic'].values.astype(np.float64)
    is_target = valid_results['is_target'].values.astype(bool)

    n_total = len(all_t_stats)
    n_targets = int(np.sum(is_target))
    n_background = n_total - n_targets

    if n_targets == 0:
        raise ValueError("No network targets found (is_target=True)")

    if n_targets >= n_total:
        raise ValueError("All proteins are targets; cannot compute enrichment")

    # Observed statistic: mean |t| for network targets
    target_t_stats = all_t_stats[is_target]
    observed_mean_abs_t = float(np.mean(np.abs(target_t_stats)))

    if verbose:
        print(f"Network enrichment test:")
        print(f"  Targets: {n_targets}, Background: {n_background}")
        print(f"  Observed mean |t|: {observed_mean_abs_t:.4f}")

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Vectorized null generation using argsort trick for sampling without replacement
    # Generate random values for each permutation, then argsort to get indices
    # This is faster than looping and memory-efficient
    #
    # Alternative: Use random permutations and take first n_targets
    # Shape: (n_permutations, n_total) -> argsort -> take first n_targets
    random_values = rng.random((n_permutations, n_total))
    random_indices = np.argpartition(random_values, n_targets, axis=1)[:, :n_targets]

    # Compute mean |t| for each permutation in one vectorized operation
    # all_t_stats[random_indices] has shape (n_permutations, n_targets)
    null_means = np.mean(np.abs(all_t_stats[random_indices]), axis=1)

    # Null distribution statistics
    null_mean = float(np.mean(null_means))
    null_std = float(np.std(null_means, ddof=1))

    # Z-score: standardized enrichment score
    if null_std > 1e-10:
        z_score = (observed_mean_abs_t - null_mean) / null_std
    else:
        z_score = np.nan

    # Empirical p-value: one-sided (targets > random)
    n_extreme = np.sum(null_means >= observed_mean_abs_t)
    empirical_pvalue = float((n_extreme + 1) / (n_permutations + 1))

    # Directional bias: what fraction of targets are down-regulated?
    n_negative = int(np.sum(target_t_stats < 0))
    pct_down = 100.0 * n_negative / n_targets

    # Binomial test for directional bias (two-sided test vs 50%)
    # Tests whether the fraction of negative t-stats differs from 50%
    direction_pvalue = float(scipy_stats.binom_test(
        n_negative, n_targets, p=0.5, alternative='two-sided'
    )) if hasattr(scipy_stats, 'binom_test') else float(
        scipy_stats.binomtest(n_negative, n_targets, p=0.5, alternative='two-sided').pvalue
    )

    # Mann-Whitney U test: compare |t| distributions between targets and background
    background_t_stats = all_t_stats[~is_target]
    mannwhitney_stat, mannwhitney_pvalue = scipy_stats.mannwhitneyu(
        np.abs(target_t_stats),
        np.abs(background_t_stats),
        alternative='greater',  # Targets > background
    )

    if verbose:
        print(f"  Null distribution: mean={null_mean:.4f}, std={null_std:.4f}")
        print(f"  Z-score: {z_score:.2f}")
        print(f"  Empirical p-value: {empirical_pvalue:.2e}")
        print(f"  Direction: {pct_down:.1f}% down-regulated")
        print(f"  Mann-Whitney p-value: {mannwhitney_pvalue:.2e}")

    return NetworkEnrichmentResult(
        observed_mean_abs_t=observed_mean_abs_t,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        empirical_pvalue=empirical_pvalue,
        n_targets=n_targets,
        n_background=n_background,
        pct_down=pct_down,
        direction_pvalue=direction_pvalue,
        mannwhitney_pvalue=float(mannwhitney_pvalue),
    )


def run_differential_analysis(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_condition: NDArray | pd.Series,
    sample_subject: NDArray | pd.Series | None = None,
    contrasts: dict[str, tuple[str, str]] | None = None,
    use_mixed: bool = True,
    use_gpu: bool = True,
    fdr_method: str = "BH",
    fdr_threshold: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = True,
) -> DifferentialResult:
    """
    Run differential abundance analysis on all features.

    This is the main entry point for MSstats-style differential analysis.

    Args:
        data: 2D array (n_features, n_samples) of log2 intensities.
        feature_ids: List of feature (protein/clique) identifiers.
        sample_condition: Condition label for each sample.
        sample_subject: Subject ID for each sample (for biological replicates).
        contrasts: Dict of contrasts to test. If None, tests all pairwise.
        use_mixed: Whether to use mixed models when subject info available.
        use_gpu: Whether to use GPU-accelerated batched OLS (when applicable).
            Only used for fixed effects models. Requires MLX library.
        fdr_method: FDR correction method ("BH", "BY", "bonferroni").
        fdr_threshold: FDR threshold for significance.
        n_jobs: Number of parallel jobs (only for sequential mode).
        verbose: Whether to print progress.

    Returns:
        DifferentialResult with all test results.

    Example:
        >>> # Simple two-group comparison with GPU acceleration
        >>> result = run_differential_analysis(
        ...     data=log2_intensities,
        ...     feature_ids=protein_ids,
        ...     sample_condition=metadata['treatment_group'],
        ...     sample_subject=metadata['subject_id'],
        ...     contrasts={'treatment_vs_control': ('treatment', 'control')},
        ...     use_gpu=True,
        ... )
        >>> df = result.to_dataframe()
        >>> significant = df[df['significant']]
    """
    from joblib import Parallel, delayed

    n_features, n_samples = data.shape

    if len(feature_ids) != n_features:
        raise ValueError(f"feature_ids length ({len(feature_ids)}) != data rows ({n_features})")

    if len(sample_condition) != n_samples:
        raise ValueError(f"sample_condition length ({len(sample_condition)}) != data columns ({n_samples})")

    # Get unique conditions
    conditions = sorted(pd.Series(sample_condition).dropna().unique().tolist())

    if len(conditions) < 2:
        raise ValueError(f"Need at least 2 conditions, got {len(conditions)}")

    # Build contrast matrix
    contrast_matrix, contrast_names = build_contrast_matrix(conditions, contrasts)

    if verbose:
        print(f"Differential analysis: {n_features} features, {n_samples} samples")
        print(f"Conditions: {conditions}")
        print(f"Contrasts: {contrast_names}")
        if sample_subject is not None:
            n_subjects = len(pd.Series(sample_subject).dropna().unique())
            print(f"Subjects: {n_subjects} (using {'mixed' if use_mixed else 'fixed'} model)")

    # Determine if we can use GPU batching
    can_use_gpu = (
        use_gpu
        and MLX_AVAILABLE
        and (sample_subject is None or not use_mixed)  # Only for fixed effects
    )

    if can_use_gpu:
        if verbose:
            print("Using GPU-accelerated batched OLS")

        try:
            # Prepare design matrix
            import statsmodels.api as sm

            # Convert condition to categorical with explicit order
            condition_cat = pd.Categorical(sample_condition, categories=conditions)
            df = pd.DataFrame({'condition': condition_cat})

            # Create design matrix with dummy coding
            X = pd.get_dummies(df['condition'], drop_first=True, dtype=float)
            X = sm.add_constant(X)
            X_np = X.values

            # Transpose data to (n_samples, n_features)
            Y = data.T

            # Run batched GPU analysis
            results = batched_ols_gpu(
                Y=Y,
                X=X_np,
                conditions=conditions,
                feature_ids=feature_ids,
                contrast_matrix=contrast_matrix,
                contrast_names=contrast_names,
            )

            if verbose:
                print(f"  GPU batch processing completed for {n_features} features")

        except Exception as e:
            if verbose:
                print(f"  GPU batching failed ({e}), falling back to sequential")
            can_use_gpu = False

    if not can_use_gpu:
        # Sequential or parallel CPU processing
        def process_feature(i: int) -> ProteinResult:
            return differential_analysis_single(
                intensities=data[i, :],
                condition=sample_condition,
                subject=sample_subject,
                feature_id=feature_ids[i],
                contrast_matrix=contrast_matrix,
                contrast_names=contrast_names,
                conditions=conditions,
                use_mixed=use_mixed,
            )

        if n_jobs == 1:
            results = []
            for i in range(n_features):
                results.append(process_feature(i))
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{n_features} features")
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_feature)(i) for i in range(n_features)
            )

    return DifferentialResult(
        results=results,
        contrasts_tested=contrast_names,
        fdr_method=fdr_method,
        fdr_threshold=fdr_threshold,
    )
