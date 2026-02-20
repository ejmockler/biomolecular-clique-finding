"""
Covariate-aware design matrix construction for differential analysis.

Builds design matrices that include both condition indicators and
covariate adjustments (e.g., Sex, Batch). The contrast vector is
zero-padded so that hypothesis tests target the condition effect
while covariates are treated as nuisance parameters.

Design matrix structure:
    X = [intercept | condition_dummies | covariate_columns]

Contrast vector structure:
    c = [condition_contrast... | 0, 0, ..., 0]
         ^^ from build_contrast_matrix   ^^ zero-padded for covariates

This propagates through ROAST (via fit_general) and OLS/GPU paths
identically, so covariate adjustment is consistent across all methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class CovariateDesign:
    """Complete covariate-aware design specification.

    Attributes:
        X: Design matrix (n_valid_samples, n_params), full column rank.
        condition_cols: Column indices for intercept + condition dummies.
        covariate_cols: Column indices for covariate columns.
        col_names: Human-readable names for all columns.
        contrast: Contrast vector (n_params,) zero-padded for covariates.
        contrast_name: Human-readable contrast description.
        sample_mask: Boolean mask (n_original_samples,) — True for valid
            samples (no NaN in condition or covariates).
        n_condition_params: Number of condition-related columns
            (intercept + dummies).
        n_covariate_params: Number of covariate columns.
    """

    X: NDArray[np.float64]
    condition_cols: list[int]
    covariate_cols: list[int]
    col_names: list[str]
    contrast: NDArray[np.float64]
    contrast_name: str
    sample_mask: NDArray[np.bool_]
    n_condition_params: int
    n_covariate_params: int

    @property
    def n_params(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def df_residual(self) -> int:
        return self.n_samples - self.n_params


def build_covariate_design_matrix(
    sample_condition: NDArray | pd.Series,
    conditions: list[str],
    contrast: tuple[str, str],
    covariates_df: pd.DataFrame | None = None,
    interaction_terms: bool = False,
) -> CovariateDesign:
    """
    Build design matrix with optional covariates.

    Constructs X = [intercept | condition_dummies | covariate_columns]
    and a contrast vector zero-padded for covariate columns.

    Categorical covariates are dummy-coded (drop_first=True).
    Numeric covariates are standardized (zero mean, unit variance).

    Args:
        sample_condition: Condition labels per sample (length n_samples).
        conditions: Ordered list of unique condition names.
        contrast: (test_condition, reference_condition) — tests test - ref.
        covariates_df: DataFrame of covariates, one row per sample.
            Categorical/object columns: dummy coded.
            Numeric columns: standardized.
            If None, builds condition-only design (backward compatible).
        interaction_terms: If True, append condition × covariate interaction
            columns to the design matrix. The contrast vector is zero-padded
            for interaction columns (nuisance, not tested). Requires
            covariates_df. Default: False.

    Returns:
        CovariateDesign with full design matrix, contrast, and metadata.

    Raises:
        ValueError: If contrast conditions not found, insufficient samples,
            or design matrix is rank-deficient.
    """
    import statsmodels.api as sm

    n_samples = len(sample_condition)
    sample_condition = np.asarray(sample_condition)

    # --- Step 1: Build condition part ---
    condition_cat = pd.Categorical(sample_condition, categories=conditions)
    cond_df = pd.DataFrame({"condition": condition_cat})

    # Track valid samples (no NaN in condition)
    valid_mask = ~cond_df["condition"].isna()

    # --- Step 2: Build covariate part and update valid mask ---
    covariate_col_names: list[str] = []
    covariate_matrix: NDArray | None = None

    if covariates_df is not None and len(covariates_df.columns) > 0:
        if len(covariates_df) != n_samples:
            raise ValueError(
                f"covariates_df has {len(covariates_df)} rows but "
                f"sample_condition has {n_samples} elements"
            )

        # Drop rows with NaN in any covariate
        cov_valid = ~covariates_df.isna().any(axis=1).values
        valid_mask = valid_mask & cov_valid

        # Build covariate columns from valid rows
        cov_parts: list[NDArray] = []
        for col in covariates_df.columns:
            series = covariates_df[col]
            if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
                # Categorical: dummy code (drop_first=True)
                dummies = pd.get_dummies(
                    series, prefix=col, drop_first=True, dtype=float
                )
                for dummy_col in dummies.columns:
                    covariate_col_names.append(str(dummy_col))
                cov_parts.append(dummies.values)
            else:
                # Numeric: standardize using valid-sample statistics
                vals = series.values.astype(np.float64)
                valid_vals = vals[valid_mask]
                mu = np.mean(valid_vals)
                sigma = np.std(valid_vals, ddof=1)
                if sigma < 1e-10:
                    raise ValueError(
                        f"Covariate '{col}' has zero variance — cannot standardize"
                    )
                standardized = (vals - mu) / sigma
                covariate_col_names.append(str(col))
                cov_parts.append(standardized.reshape(-1, 1))

        if cov_parts:
            covariate_matrix = np.hstack(cov_parts)

    # --- Step 3: Subset to valid samples ---
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid < 3:
        raise ValueError(f"Insufficient valid samples after NaN removal: {n_valid}")

    cond_df_valid = cond_df.iloc[valid_indices]

    # --- Step 4: Assemble design matrix ---
    # Condition dummies (drop_first=True means reference condition is intercept)
    X_cond = pd.get_dummies(
        cond_df_valid["condition"], drop_first=True, dtype=float
    )
    X_cond = sm.add_constant(X_cond)
    condition_col_names = list(X_cond.columns)
    X_np = X_cond.values.astype(np.float64)

    n_condition_params = X_np.shape[1]
    condition_col_indices = list(range(n_condition_params))

    # Append covariates if present
    covariate_col_indices: list[int] = []
    if covariate_matrix is not None:
        cov_valid = covariate_matrix[valid_indices]
        n_cov_cols = cov_valid.shape[1]
        covariate_col_indices = list(
            range(n_condition_params, n_condition_params + n_cov_cols)
        )
        X_np = np.hstack([X_np, cov_valid])

    all_col_names = condition_col_names + covariate_col_names

    # --- Step 4b: Append interaction terms (condition × covariate) ---
    interaction_col_names: list[str] = []
    if interaction_terms and covariate_matrix is not None:
        import warnings

        cov_valid = covariate_matrix[valid_indices]
        # Condition dummies (excluding intercept)
        cond_dummies = X_np[:, 1:n_condition_params]
        cond_dummy_names = condition_col_names[1:]

        interaction_parts: list[NDArray] = []
        for ci, cov_name in enumerate(covariate_col_names):
            for di, cond_name in enumerate(cond_dummy_names):
                interaction_col = cond_dummies[:, di] * cov_valid[:, ci]
                interaction_parts.append(interaction_col.reshape(-1, 1))
                interaction_col_names.append(f"{cond_name}:{cov_name}")

        if interaction_parts:
            interaction_matrix = np.hstack(interaction_parts)
            X_np = np.hstack([X_np, interaction_matrix])
            covariate_col_indices = list(
                range(n_condition_params, X_np.shape[1])
            )

        all_col_names = all_col_names + interaction_col_names

    n_params = X_np.shape[1]

    # --- Step 5: Validate rank ---
    rank = np.linalg.matrix_rank(X_np)
    if rank < n_params:
        raise ValueError(
            f"Design matrix is rank-deficient: rank={rank}, n_params={n_params}. "
            f"Columns: {all_col_names}. A covariate may be collinear with the "
            f"condition or another covariate."
        )

    if n_valid - n_params < 1:
        raise ValueError(
            f"Insufficient residual df: {n_valid} samples - {n_params} params = "
            f"{n_valid - n_params}. Reduce covariates or increase sample size."
        )

    # df guards for interaction models
    residual_df = n_valid - n_params
    if interaction_terms:
        if residual_df < 10:
            raise ValueError(
                f"Residual df too low for interaction model: {residual_df} "
                f"({n_valid} samples - {n_params} params). "
                f"Need at least 10 residual df."
            )
        if residual_df < 15:
            import warnings
            warnings.warn(
                f"Low residual df ({residual_df}) for interaction model — "
                f"estimates may be unstable."
            )

    # Condition number check (on X, not X'X, to avoid squared scaling)
    cond_number = np.linalg.cond(X_np)
    if cond_number > 30:
        import warnings
        warnings.warn(
            f"Design matrix condition number is high ({cond_number:.1f} > 30). "
            f"Near-collinearity may cause unstable estimates."
        )

    # --- Step 6: Build contrast vector ---
    # First build the condition-space contrast
    n_conditions = len(conditions)
    contrast_vec_condition = np.zeros(n_conditions)

    try:
        idx1 = conditions.index(contrast[0])
        idx2 = conditions.index(contrast[1])
    except ValueError as e:
        raise ValueError(
            f"Contrast conditions {contrast} not found in {conditions}: {e}"
        )

    contrast_vec_condition[idx1] = 1.0
    contrast_vec_condition[idx2] = -1.0

    # Transform to parameter space via the L matrix
    # L maps condition means to parameters (dummy coding)
    L = np.zeros((n_conditions, n_condition_params))
    L[:, 0] = 1.0  # All conditions include intercept
    for i in range(1, min(n_conditions, n_condition_params)):
        L[i, i] = 1.0

    # Contrast in condition-parameter space
    c_condition = L.T @ contrast_vec_condition

    # Zero-pad for covariate and interaction columns
    n_covariate_params = len(covariate_col_indices)
    n_pad = n_params - n_condition_params
    if n_pad > 0:
        c_full = np.concatenate([c_condition, np.zeros(n_pad)])
    else:
        c_full = c_condition

    contrast_name = f"{contrast[0]}_vs_{contrast[1]}"

    return CovariateDesign(
        X=X_np,
        condition_cols=condition_col_indices,
        covariate_cols=covariate_col_indices,
        col_names=all_col_names,
        contrast=c_full,
        contrast_name=contrast_name,
        sample_mask=valid_mask.values if isinstance(valid_mask, pd.Series) else valid_mask,
        n_condition_params=n_condition_params,
        n_covariate_params=n_covariate_params,
    )


def pad_contrast_for_covariates(
    condition_contrast: NDArray[np.float64],
    n_covariate_cols: int,
) -> NDArray[np.float64]:
    """
    Zero-pad a condition contrast vector for covariate columns.

    The contrast tests condition differences while controlling for covariates.
    Covariate coefficients are nuisance parameters with zero contrast weight.

    Args:
        condition_contrast: Contrast vector in condition parameter space.
        n_covariate_cols: Number of covariate columns to append zeros for.

    Returns:
        Extended contrast vector with zeros appended.
    """
    if n_covariate_cols == 0:
        return condition_contrast
    return np.concatenate([condition_contrast, np.zeros(n_covariate_cols)])
