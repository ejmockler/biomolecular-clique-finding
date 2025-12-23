"""
Outlier detection for proteomics/transcriptomics data using robust statistics.

This module provides robust outlier detection that identifies statistical outliers
without being influenced by the outliers themselves (the classic bootstrapping problem).

Key Design Decision:
    **Within-group detection is the default** because per-gene detection across all
    samples conflates biological CASE-CTRL differences with technical artifacts.

    Within-group detection computes statistics within each phenotype group separately,
    preserving biological signal while catching true technical artifacts.

Methods:
    - MAD-Z (Median Absolute Deviation): Robust, 50% breakdown point
    - IQR (Interquartile Range): Simpler, 25% breakdown point

Modes:
    - within_group (default): Per-gene within each phenotype group
    - per_feature: Per-gene across all samples (legacy, conflates biology)
    - global: Across entire matrix (use with normalized data)

References:
    - Leys et al. (2013) "Detecting outliers: Do not use standard deviation
      around the mean, use absolute deviation around the median"
    - Iglewicz & Hoaglin (1993) "How to Detect and Handle Outliers"

Examples:
    >>> from cliquefinder.quality.outliers import OutlierDetector
    >>>
    >>> # Default: within-group detection (preserves biology)
    >>> detector = OutlierDetector()
    >>> flagged = detector.apply(matrix)
    >>>
    >>> # For model-based detection (accounts for covariates)
    >>> from cliquefinder.quality.outliers import ResidualOutlierDetector
    >>> detector = ResidualOutlierDetector(formula="expression ~ C(phenotype)")
    >>> flagged = detector.apply(matrix)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Literal, Tuple
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar

from cliquefinder.core.transform import Transform
from cliquefinder.core.quality import QualityFlag
from cliquefinder.core.biomatrix import BioMatrix

__all__ = [
    'OutlierDetector',
    'ResidualOutlierDetector',
    'AdaptiveOutlierDetector',
    'fit_student_t_shared',
    'compute_outlier_probability',
    'compute_medcouple',
    'adjusted_boxplot_fences',
]


class OutlierDetector(Transform):
    """
    Detect outliers using robust statistics with configurable detection modes.

    This transform marks outliers in quality_flags without modifying data values.
    Use Imputer afterward to replace flagged values if desired.

    Modes:
        within_group (default, recommended):
            Detects outliers per-gene within each group separately.
            Preserves biological differences between groups.
            Requires sample_metadata with specified group column(s).

        per_feature:
            Detects outliers per-gene across ALL samples.
            WARNING: Conflates biological differences with artifacts.
            Use only when no group structure exists.

        global:
            Detects outliers across entire matrix.
            Use with pre-normalized data on same scale.

    Methods:
        mad-z (recommended): Median Absolute Deviation, threshold 3.5
        iqr: Interquartile Range (Tukey's method), threshold 1.5

    Args:
        method: "mad-z" (recommended) or "iqr"
        threshold: Detection threshold (3.5 for MAD-Z, 1.5 for IQR)
        mode: "within_group" (default), "per_feature", or "global"
        group_cols: Metadata column(s) for grouping. Can be:
            - str: Single column (e.g., "phenotype")
            - list[str]: Multiple columns for stratification (e.g., ["phenotype", "Sex"])
            When multiple columns, groups are formed by unique combinations.

    Examples:
        >>> # Default: within-group MAD-Z (preserves biology)
        >>> detector = OutlierDetector()
        >>> flagged = detector.apply(matrix)
        >>>
        >>> # Stratify by phenotype AND sex
        >>> detector = OutlierDetector(group_cols=["phenotype", "Sex"])
        >>>
        >>> # Legacy per-feature mode
        >>> detector = OutlierDetector(mode="per_feature")
    """

    def __init__(
        self,
        method: Literal["mad-z", "iqr"] = "mad-z",
        threshold: float = 3.5,
        mode: Literal["within_group", "per_feature", "global"] = "within_group",
        group_cols: str | list[str] = "phenotype"
    ):
        if method not in ("mad-z", "iqr"):
            raise ValueError(f"method must be 'mad-z' or 'iqr', got '{method}'")
        if mode not in ("within_group", "per_feature", "global"):
            raise ValueError(f"mode must be 'within_group', 'per_feature', or 'global'")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        # Normalize group_cols to list
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        super().__init__(
            name="OutlierDetector",
            params={"method": method, "threshold": threshold, "mode": mode, "group_cols": group_cols}
        )

        self.method = method
        self.threshold = threshold
        self.mode = mode
        self.group_cols = group_cols

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Detect outliers and mark in quality flags.

        Args:
            matrix: Input BioMatrix

        Returns:
            New BioMatrix with outliers marked (data unchanged)
        """
        errors = self.validate(matrix)
        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        # Dispatch to mode-specific detection
        if self.mode == "within_group":
            outlier_mask = self._detect_within_group(matrix)
        elif self.mode == "per_feature":
            outlier_mask = self._detect_per_feature(matrix.data)
        else:  # global
            outlier_mask = self._detect_global(matrix.data)

        # Warn if too many outliers
        outlier_pct = 100 * outlier_mask.mean()
        if outlier_pct > 10:
            warnings.warn(f"Detected {outlier_pct:.1f}% outliers (>10% is high)", UserWarning)

        # Update quality flags (explicit cast to preserve dtype)
        new_flags = matrix.quality_flags.copy()
        new_flags[outlier_mask] = (new_flags[outlier_mask] | QualityFlag.OUTLIER_DETECTED).astype(new_flags.dtype)

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
        )

    def validate(self, matrix: BioMatrix) -> list[str]:
        """Validate matrix for outlier detection."""
        errors = super().validate(matrix)

        if np.any(np.isnan(matrix.data)):
            errors.append("Matrix contains NaN values")
        if np.any(np.isinf(matrix.data)):
            errors.append("Matrix contains Inf values")

        if self.mode == "within_group":
            if matrix.sample_metadata is None:
                errors.append(f"within_group mode requires sample_metadata")
            else:
                missing_cols = [c for c in self.group_cols if c not in matrix.sample_metadata.columns]
                if missing_cols:
                    errors.append(f"Group column(s) not in metadata: {missing_cols}")

        return errors

    def _detect_within_group(self, matrix: BioMatrix) -> np.ndarray:
        """Detect outliers per-gene within each group separately.

        Groups are formed by unique combinations of all group_cols.
        For example, with group_cols=["phenotype", "Sex"]:
            - CASE + Male
            - CASE + Female
            - CTRL + Male
            - CTRL + Female
        """
        outliers = np.zeros_like(matrix.data, dtype=bool)

        # Create composite group key from all group columns
        if len(self.group_cols) == 1:
            group_series = matrix.sample_metadata[self.group_cols[0]]
        else:
            # Combine multiple columns into tuple keys
            group_series = matrix.sample_metadata[self.group_cols].apply(tuple, axis=1)

        unique_groups = group_series.unique()

        for group in unique_groups:
            group_mask = (group_series == group).values
            n_samples = group_mask.sum()

            # Skip groups that are too small for robust MAD-Z statistics
            # Minimum 10 samples recommended for reliable MAD estimation
            # (see Croux & Rousseeuw, 1992 on finite-sample corrections)
            if n_samples < 10:
                warnings.warn(
                    f"Group {group} has only {n_samples} samples (minimum 10 recommended for robust MAD-Z). "
                    f"Skipping outlier detection for this group."
                )
                continue

            group_data = matrix.data[:, group_mask]

            if self.method == "mad-z":
                group_outliers = self._mad_z(group_data, per_feature=True)
            else:
                group_outliers = self._iqr(group_data, per_feature=True)

            outliers[:, group_mask] = group_outliers

        return outliers

    def _detect_per_feature(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers per-gene across all samples."""
        if self.method == "mad-z":
            return self._mad_z(data, per_feature=True)
        return self._iqr(data, per_feature=True)

    def _detect_global(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers globally across entire matrix."""
        if self.method == "mad-z":
            return self._mad_z(data, per_feature=False)
        return self._iqr(data, per_feature=False)

    def _mad_z(self, data: np.ndarray, per_feature: bool) -> np.ndarray:
        """MAD-Z outlier detection."""
        if per_feature:
            medians = np.median(data, axis=1, keepdims=True)
            mad = np.median(np.abs(data - medians), axis=1, keepdims=True)
            mad = np.where(mad == 0, 1.0, mad)  # Avoid division by zero
            z_scores = 0.6745 * np.abs(data - medians) / mad
        else:
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            mad = mad if mad != 0 else 1.0
            z_scores = 0.6745 * np.abs(data - median) / mad

        return z_scores > self.threshold

    def _iqr(self, data: np.ndarray, per_feature: bool) -> np.ndarray:
        """IQR (Tukey) outlier detection."""
        if per_feature:
            q1 = np.percentile(data, 25, axis=1, keepdims=True)
            q3 = np.percentile(data, 75, axis=1, keepdims=True)
        else:
            q1, q3 = np.percentile(data, 25), np.percentile(data, 75)

        iqr = q3 - q1
        lower, upper = q1 - self.threshold * iqr, q3 + self.threshold * iqr
        return (data < lower) | (data > upper)


class ResidualOutlierDetector(Transform):
    """
    Detect outliers in residuals after modeling biological/technical structure.

    This detector fits a linear model to each protein, then detects outliers in
    the residuals (unexplained variance). This cleanly separates:
        - Explained variance (group effects, batch) → Preserved
        - Unexplained variance (residuals) → Flagged as artifacts

    Uses vectorized computation for high performance (~5 seconds for 60K proteins).

    Args:
        formula: R-style model formula (e.g., "expression ~ C(phenotype)")
        threshold: MAD-Z threshold for residuals (default: 3.5)

    Examples:
        >>> detector = ResidualOutlierDetector(formula="expression ~ C(phenotype)")
        >>> flagged = detector.apply(matrix)
        >>>
        >>> # Access diagnostics
        >>> print(f"Median R²: {detector.diagnostics_.summary()['median_r_squared']:.4f}")

    Notes:
        - Formula must not have too many categorical levels (causes rank deficiency)
        - Memory: ~300 MB for 60K proteins × 600 samples
        - Uses hat matrix H = X(X'X)⁻¹X' pre-computed once
    """

    def __init__(
        self,
        formula: str = "expression ~ C(phenotype)",
        threshold: float = 3.5
    ):
        super().__init__(
            name="ResidualOutlierDetector",
            params={"formula": formula, "threshold": threshold}
        )
        self.formula = formula
        self.threshold = threshold
        self.diagnostics_ = None
        self.residuals_ = None

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """Detect outliers via residual analysis."""
        from cliquefinder.models.vectorized_residuals import VectorizedResidualComputer
        import time

        errors = self.validate(matrix)
        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        # Check required columns
        required_cols = self._extract_formula_columns(self.formula)
        missing = [c for c in required_cols if c not in matrix.sample_metadata.columns]
        if missing:
            raise ValueError(f"Missing metadata columns: {missing}")

        print(f"Residual outlier detection: {matrix.n_features:,} proteins")
        start = time.perf_counter()

        # Compute residuals
        computer = VectorizedResidualComputer(
            metadata=matrix.sample_metadata,
            formula=self.formula
        )
        residuals, diagnostics = computer.compute_residuals(matrix.data)

        # MAD-Z on residuals per protein
        outliers = np.zeros_like(residuals, dtype=bool)
        for i in range(residuals.shape[0]):
            resid = residuals[i, :]
            valid = ~np.isnan(resid)
            if valid.sum() < 3:
                continue

            valid_resid = resid[valid]
            median = np.median(valid_resid)
            mad = np.median(np.abs(valid_resid - median))
            if mad == 0:
                continue

            z = 0.6745 * np.abs(valid_resid - median) / mad
            outliers[i, valid] = z > self.threshold

        elapsed = time.perf_counter() - start
        self.diagnostics_ = diagnostics
        self.residuals_ = residuals

        summary = diagnostics.summary()
        print(f"✓ {elapsed:.1f}s | R²={summary['median_r_squared']:.4f} | "
              f"Outliers: {outliers.sum():,} ({100*outliers.mean():.2f}%)")

        new_flags = matrix.quality_flags.copy()
        new_flags[outliers] = (new_flags[outliers] | QualityFlag.OUTLIER_DETECTED).astype(new_flags.dtype)

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
        )

    def validate(self, matrix: BioMatrix) -> list[str]:
        """Validate matrix for residual detection."""
        errors = super().validate(matrix)
        if matrix.sample_metadata is None or len(matrix.sample_metadata) == 0:
            errors.append("sample_metadata required for residual detection")
        return errors

    def _extract_formula_columns(self, formula: str) -> list:
        """Extract column names from R-style formula."""
        import re
        rhs = formula.split('~')[-1].strip()
        c_terms = re.findall(r'C\((\w+)\)', rhs)
        plain_terms = [t for t in re.findall(r'\b(\w+)\b', rhs) if t not in ('C', 'expression')]
        return list(set(c_terms + plain_terms))


# =====================================================================
# Medcouple-Adjusted Asymmetric Detection (Hubert & Vandervieren, 2008)
# =====================================================================

def compute_medcouple(x: np.ndarray) -> float:
    """
    Compute the medcouple, a robust measure of skewness.

    The medcouple is defined as the median of a kernel function over all pairs
    of observations on opposite sides of the median (Brys et al., 2004).

    Mathematical Definition:
        MC = median{ h(x_i, x_j) : x_i ≤ median(X) ≤ x_j }

        where h(x_i, x_j) = (x_j - median(X)) - (median(X) - x_i)
                            ─────────────────────────────────────
                                      x_j - x_i

    Properties:
        - Range: [-1, 1]
        - MC > 0 → right-skewed (long upper tail)
        - MC < 0 → left-skewed (long lower tail)
        - MC = 0 → symmetric
        - Breakdown point: 25% (robust to contamination)

    Args:
        x: 1D array of observations (NaN/Inf values are filtered)

    Returns:
        Medcouple value in [-1, 1]

    References:
        Brys, G., Hubert, M., & Struyf, A. (2004). "A Robust Measure of Skewness."
        Journal of Computational and Graphical Statistics, 13(4), 996-1017.

    Notes:
        This implementation uses the naive O(n²) algorithm. For very large arrays
        (n > 10,000), consider using the O(n log n) fast algorithm from robustats
        or the robustbase R package via rpy2.

    Examples:
        >>> import numpy as np
        >>> # Symmetric distribution → MC ≈ 0
        >>> x_sym = np.random.normal(0, 1, 1000)
        >>> mc_sym = compute_medcouple(x_sym)
        >>> abs(mc_sym) < 0.1  # Should be near zero
        True

        >>> # Right-skewed distribution → MC > 0
        >>> x_right = np.random.lognormal(0, 1, 1000)
        >>> mc_right = compute_medcouple(x_right)
        >>> mc_right > 0.2  # Should be positive
        True
    """
    # Filter NaN/Inf
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 3:
        return 0.0

    median_val = np.median(x)

    # Split into values at or below and at or above median
    # Include values equal to median in both sets per Brys et al. (2004)
    x_minus = x[x <= median_val]
    x_plus = x[x >= median_val]

    if len(x_minus) == 0 or len(x_plus) == 0:
        return 0.0

    # Sort for efficiency
    x_minus = np.sort(x_minus)
    x_plus = np.sort(x_plus)

    # Compute kernel h for all valid pairs
    # h(xi, xj) = [(xj - m) - (m - xi)] / (xj - xi)
    #           = (xj + xi - 2m) / (xj - xi)
    h_values = []

    for xi in x_minus:
        for xj in x_plus:
            # Handle ties at median (special case per Brys et al.)
            if xi == xj:
                # Both values equal to median
                # By convention, h = 0 for this case
                continue
            elif xi == median_val and xj == median_val:
                # Both at median exactly
                continue
            elif xi == median_val:
                # xi at median, xj > median
                # h approaches 1 as xi → median from below
                h = 1.0
            elif xj == median_val:
                # xj at median, xi < median
                # h approaches -1 as xj → median from above
                h = -1.0
            else:
                # General case
                numerator = (xj - median_val) - (median_val - xi)
                denominator = xj - xi
                h = numerator / denominator

            h_values.append(h)

    if len(h_values) == 0:
        return 0.0

    return float(np.median(h_values))


def adjusted_boxplot_fences(
    x: np.ndarray,
    mc: Optional[float] = None,
    whis: float = 1.5
) -> Tuple[float, float]:
    """
    Compute asymmetric outlier fences using the Hubert-Vandervieren method.

    For skewed distributions, standard boxplot fences are symmetric and flag
    legitimate values in the longer tail as outliers. This method adjusts the
    fence multipliers based on the medcouple (MC), a robust skewness measure.

    Mathematical Formulation:
        Lower fence: Q1 - whis × exp(a × MC) × IQR
        Upper fence: Q3 + whis × exp(b × MC) × IQR

        where:
            If MC ≥ 0 (right-skewed): a = -4, b = 3
            If MC < 0 (left-skewed):  a = -3, b = 4

    Effect:
        - Right-skewed (MC > 0): Upper fence expands, lower fence contracts
        - Left-skewed (MC < 0): Lower fence expands, upper fence contracts
        - Symmetric (MC = 0): Reduces to standard boxplot (exp(0) = 1)

    Args:
        x: 1D array of observations (NaN/Inf values are filtered)
        mc: Pre-computed medcouple (computed from x if None)
        whis: Whisker multiplier (default 1.5, standard boxplot value)

    Returns:
        (lower_fence, upper_fence) tuple

    References:
        Hubert, M. & Vandervieren, E. (2008). "An adjusted boxplot for skewed
        distributions." Computational Statistics & Data Analysis, 52, 5186-5201.

    Notes:
        - Works well for moderate skewness: -0.6 ≤ MC ≤ 0.6
        - For extreme skewness, consider log transformation first
        - Computationally efficient: O(n²) for medcouple + O(n) for percentiles

    Examples:
        >>> import numpy as np
        >>> # Right-skewed lognormal data
        >>> x = np.random.lognormal(0, 0.5, 1000)
        >>> lower, upper = adjusted_boxplot_fences(x)
        >>>
        >>> # Compare with standard symmetric fences
        >>> q1, q3 = np.percentile(x, [25, 75])
        >>> iqr = q3 - q1
        >>> std_lower, std_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        >>>
        >>> # Adjusted upper should be more permissive (higher) for right-skewed
        >>> upper > std_upper
        True
    """
    # Filter NaN/Inf
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]

    if len(x) < 4:
        # Not enough data for quartiles
        return float('-inf'), float('inf')

    # Compute medcouple if not provided
    if mc is None:
        mc = compute_medcouple(x)

    # Compute quartiles and IQR
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1

    # Handle zero IQR (all values in central 50% are identical)
    if iqr == 0:
        # Fall back to using median ± small epsilon
        median_val = np.median(x)
        return median_val - 1e-10, median_val + 1e-10

    # Select coefficients based on skewness direction
    # Per Hubert & Vandervieren (2008), Table 1
    if mc >= 0:
        # Right-skewed: expand upper, contract lower
        a, b = -4.0, 3.0
    else:
        # Left-skewed: expand lower, contract upper
        a, b = -3.0, 4.0

    # Compute adjusted fences
    # Lower: Q1 - whis * exp(a * MC) * IQR
    # Upper: Q3 + whis * exp(b * MC) * IQR
    lower_fence = q1 - whis * np.exp(a * mc) * iqr
    upper_fence = q3 + whis * np.exp(b * mc) * iqr

    return float(lower_fence), float(upper_fence)


# =====================================================================
# Student's t Probabilistic Scoring (PROTRIDER approach)
# =====================================================================

def fit_student_t_shared(
    data: np.ndarray,
    robust_location: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Fit Student's t-distribution with shared degrees of freedom across features.

    Implements the PROTRIDER approach: fit per-feature location/scale using
    robust estimators (median/MAD), then pool standardized residuals to
    estimate a single shared df parameter.

    Args:
        data: Expression matrix (n_features × n_samples)
        robust_location: Use median/MAD (True) or mean/std (False)

    Returns:
        (df_shared, locations, scales) where:
        - df_shared: Single df value for all features (typically 3-10 for proteomics)
        - locations: Per-feature location parameters (n_features,)
        - scales: Per-feature scale parameters (n_features,)

    References:
        Scheller et al. (2025). PROTRIDER: Protein abundance outlier detection
        from mass spectrometry-based proteomics data. bioRxiv.
    """
    n_features, n_samples = data.shape

    # Step 1: Estimate per-feature location and scale robustly
    locations = np.zeros(n_features)
    scales = np.zeros(n_features)

    for i in range(n_features):
        feature_data = data[i, :]
        # Remove NaN/Inf
        valid_mask = np.isfinite(feature_data)
        valid_data = feature_data[valid_mask]

        if len(valid_data) < 3:
            # Not enough data, use defaults
            locations[i] = 0.0
            scales[i] = 1.0
            continue

        if robust_location:
            # Median and MAD (Median Absolute Deviation)
            locations[i] = np.median(valid_data)
            mad = np.median(np.abs(valid_data - locations[i]))
            # Convert MAD to std-equivalent (multiply by 1.4826 for normal)
            scales[i] = 1.4826 * mad if mad > 0 else 1.0
        else:
            # Mean and standard deviation
            locations[i] = np.mean(valid_data)
            scales[i] = np.std(valid_data, ddof=1)
            if scales[i] == 0:
                scales[i] = 1.0

    # Step 2: Pool standardized residuals across all features
    standardized_residuals = []
    for i in range(n_features):
        feature_data = data[i, :]
        valid_mask = np.isfinite(feature_data)
        valid_data = feature_data[valid_mask]

        if len(valid_data) < 3:
            continue

        # Standardize: (x - μ) / σ
        z = (valid_data - locations[i]) / scales[i]
        standardized_residuals.extend(z)

    standardized_residuals = np.array(standardized_residuals)

    # Step 3: Fit degrees of freedom to pooled standardized residuals
    # Use MLE via negative log-likelihood minimization
    def neg_log_likelihood(df_val):
        """Negative log-likelihood for Student's t with location=0, scale=1."""
        if df_val <= 2 or df_val > 100:
            return np.inf
        try:
            # Use scipy's t distribution (location=0, scale=1, df=df_val)
            ll = np.sum(stats.t.logpdf(standardized_residuals, df=df_val))
            return -ll
        except:
            return np.inf

    # Optimize df in range [2.1, 100]
    # Start from df=5 (typical for proteomics per PROTRIDER)
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(2.1, 100),
        method='bounded'
    )

    df_shared = result.x

    # Clamp to safe range for numerical stability
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

    Returns two-tailed p-values: P(|T| > |observed|).
    Low values indicate likely outliers.

    Args:
        data: Expression matrix (n_features × n_samples)
        df: Degrees of freedom (shared across features)
        locations: Per-feature locations (n_features,)
        scales: Per-feature scales (n_features,)

    Returns:
        P-value matrix (n_features × n_samples) where values near 0 = outlier
    """
    n_features, n_samples = data.shape
    p_values = np.ones((n_features, n_samples))  # Default: not outlier

    for i in range(n_features):
        feature_data = data[i, :]

        # Standardize using fitted parameters
        z = (feature_data - locations[i]) / scales[i]

        # Compute two-tailed p-value for each observation
        # P(|T| > |z|) = 2 * min(P(T < z), P(T > z))
        # = 2 * min(cdf(z), 1 - cdf(z))
        cdf_vals = stats.t.cdf(z, df=df)
        p_values[i, :] = 2 * np.minimum(cdf_vals, 1 - cdf_vals)

    # Handle NaN/Inf in input data
    p_values[~np.isfinite(data)] = 1.0

    return p_values


class AdaptiveOutlierDetector(Transform):
    """
    Adaptive outlier detector with distribution-aware detection and probabilistic scoring.

    This detector extends basic outlier detection with:
    1. Asymmetric detection using medcouple-adjusted fences (Hubert & Vandervieren, 2008)
    2. Probabilistic scoring using Student's t-distribution (PROTRIDER, Scheller et al., 2025)

    Detection Methods:
        mad-z: Symmetric MAD-Z threshold (default, works well for symmetric data)
        iqr: Symmetric IQR/Tukey method (standard boxplot fences)
        adjusted-boxplot: Asymmetric fences based on medcouple (recommended for skewed data)

    Scoring Methods:
        binary: Classic threshold-based detection (outlier/not outlier)
        student_t: Probabilistic scores using Student's t-distribution

    Args:
        method: Detection method
            - "mad-z" (default): Symmetric MAD-Z threshold
            - "iqr": Symmetric IQR (Tukey) method
            - "adjusted-boxplot": Asymmetric medcouple-adjusted fences (Hubert & Vandervieren)
        threshold: Detection threshold
            - For mad-z: MAD-Z threshold (default: 3.5)
            - For iqr/adjusted-boxplot: IQR whisker multiplier (default: 1.5)
            - For student_t: p-value threshold (default: 0.01)
        mode: Detection scope
            - "within_group" (default): Per-gene within each phenotype group
            - "per_feature": Per-gene across all samples
            - "global": Across entire matrix
        group_cols: Metadata column(s) for grouping in within_group mode
        scoring_method: "binary" (classic) or "student_t" (probabilistic)

    Examples:
        >>> # Symmetric MAD-Z (default, for symmetric data)
        >>> detector = AdaptiveOutlierDetector(method="mad-z")
        >>> flagged = detector.apply(matrix)
        >>>
        >>> # Asymmetric adjusted boxplot (for skewed data like proteomics)
        >>> detector = AdaptiveOutlierDetector(method="adjusted-boxplot")
        >>> flagged = detector.apply(matrix)
        >>> print(f"Medcouple (first gene): {detector.medcouples_[0]:.3f}")
        >>>
        >>> # Probabilistic scoring with Student's t
        >>> detector = AdaptiveOutlierDetector(scoring_method="student_t", threshold=0.01)
        >>> flagged = detector.apply(matrix)
        >>> print(f"Fitted df: {detector.df_shared_:.2f}")

    Attributes:
        df_shared_: Fitted degrees of freedom (student_t scoring only)
        outlier_scores_: P-value matrix (student_t scoring only)
        locations_: Per-feature location parameters (student_t scoring only)
        scales_: Per-feature scale parameters (student_t scoring only)
        medcouples_: Per-feature medcouple values (adjusted-boxplot method only)
        lower_fences_: Per-feature lower fences (adjusted-boxplot method only)
        upper_fences_: Per-feature upper fences (adjusted-boxplot method only)

    References:
        - Hubert, M. & Vandervieren, E. (2008). An adjusted boxplot for skewed
          distributions. Computational Statistics & Data Analysis, 52, 5186-5201.
        - Brys, G., Hubert, M., & Struyf, A. (2004). A Robust Measure of Skewness.
          Journal of Computational and Graphical Statistics, 13(4), 996-1017.
        - Scheller et al. (2025). PROTRIDER: Protein abundance outlier detection
          from mass spectrometry-based proteomics data. bioRxiv.
    """

    def __init__(
        self,
        method: Literal["mad-z", "iqr", "adjusted-boxplot"] = "mad-z",
        threshold: float = 3.5,
        mode: Literal["within_group", "per_feature", "global"] = "within_group",
        group_cols: str | list[str] = "phenotype",
        scoring_method: Literal["binary", "student_t"] = "binary"
    ):
        if method not in ("mad-z", "iqr", "adjusted-boxplot"):
            raise ValueError(f"method must be 'mad-z', 'iqr', or 'adjusted-boxplot', got '{method}'")
        if mode not in ("within_group", "per_feature", "global"):
            raise ValueError(f"mode must be 'within_group', 'per_feature', or 'global'")
        if scoring_method not in ("binary", "student_t"):
            raise ValueError(f"scoring_method must be 'binary' or 'student_t', got '{scoring_method}'")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        # Normalize group_cols to list
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        super().__init__(
            name="AdaptiveOutlierDetector",
            params={
                "method": method,
                "threshold": threshold,
                "mode": mode,
                "group_cols": group_cols,
                "scoring_method": scoring_method
            }
        )

        self.method = method
        self.threshold = threshold
        self.mode = mode
        self.group_cols = group_cols
        self.scoring_method = scoring_method

        # Diagnostic attributes (populated after apply)
        self.df_shared_ = None
        self.outlier_scores_ = None
        self.locations_ = None
        self.scales_ = None

        # Medcouple diagnostic attributes (populated when method="adjusted-boxplot")
        self.medcouples_ = None
        self.lower_fences_ = None
        self.upper_fences_ = None

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Detect outliers and mark in quality flags.

        Args:
            matrix: Input BioMatrix

        Returns:
            New BioMatrix with outliers marked (data unchanged)
        """
        errors = self.validate(matrix)
        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        if self.scoring_method == "binary":
            # Use classic binary approach (delegate to OutlierDetector logic)
            outlier_mask = self._detect_binary(matrix)
            self.outlier_scores_ = None  # Not applicable
        else:
            # Use Student's t probabilistic scoring
            outlier_mask, p_values = self._detect_student_t(matrix)
            self.outlier_scores_ = p_values

        # Warn if too many outliers
        outlier_pct = 100 * outlier_mask.mean()
        if outlier_pct > 10:
            warnings.warn(f"Detected {outlier_pct:.1f}% outliers (>10% is high)", UserWarning)

        # Update quality flags
        new_flags = matrix.quality_flags.copy()
        new_flags[outlier_mask] = (new_flags[outlier_mask] | QualityFlag.OUTLIER_DETECTED).astype(new_flags.dtype)

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
        )

    def validate(self, matrix: BioMatrix) -> list[str]:
        """Validate matrix for outlier detection."""
        errors = []

        if np.any(np.isnan(matrix.data)):
            errors.append("Matrix contains NaN values")
        if np.any(np.isinf(matrix.data)):
            errors.append("Matrix contains Inf values")

        if self.mode == "within_group":
            if matrix.sample_metadata is None:
                errors.append("within_group mode requires sample_metadata")
            else:
                missing_cols = [c for c in self.group_cols if c not in matrix.sample_metadata.columns]
                if missing_cols:
                    errors.append(f"Group column(s) not in metadata: {missing_cols}")

        return errors

    def _detect_binary(self, matrix: BioMatrix) -> np.ndarray:
        """Binary detection using classic thresholding."""
        if self.method == "adjusted-boxplot":
            # Use medcouple-adjusted asymmetric detection
            return self._detect_adjusted_boxplot(matrix)

        # Reuse OutlierDetector logic for symmetric methods
        detector = OutlierDetector(
            method=self.method,
            threshold=self.threshold,
            mode=self.mode,
            group_cols=self.group_cols
        )

        # Dispatch to mode-specific detection
        if self.mode == "within_group":
            return detector._detect_within_group(matrix)
        elif self.mode == "per_feature":
            return detector._detect_per_feature(matrix.data)
        else:  # global
            return detector._detect_global(matrix.data)

    def _detect_adjusted_boxplot(self, matrix: BioMatrix) -> np.ndarray:
        """
        Detect outliers using medcouple-adjusted asymmetric boxplot fences.

        Uses the Hubert-Vandervieren (2008) method which adjusts fence
        multipliers based on the medcouple, a robust skewness measure.

        This method is recommended for skewed data like proteomics abundance.

        Returns:
            Boolean outlier mask (n_features × n_samples)
        """
        n_features, n_samples = matrix.shape
        outliers = np.zeros((n_features, n_samples), dtype=bool)

        # Initialize diagnostic arrays
        self.medcouples_ = np.zeros(n_features)
        self.lower_fences_ = np.zeros(n_features)
        self.upper_fences_ = np.zeros(n_features)

        if self.mode == "within_group":
            # Dispatch to within-group detection
            return self._detect_adjusted_boxplot_within_group(matrix)
        elif self.mode == "per_feature":
            # Per-feature detection across all samples
            for i in range(n_features):
                feature_data = matrix.data[i, :]
                mc = compute_medcouple(feature_data)
                lower, upper = adjusted_boxplot_fences(
                    feature_data,
                    mc=mc,
                    whis=self.threshold
                )

                self.medcouples_[i] = mc
                self.lower_fences_[i] = lower
                self.upper_fences_[i] = upper

                outliers[i, :] = (feature_data < lower) | (feature_data > upper)
        else:  # global
            # Global detection across entire matrix
            all_data = matrix.data.ravel()
            mc = compute_medcouple(all_data)
            lower, upper = adjusted_boxplot_fences(
                all_data,
                mc=mc,
                whis=self.threshold
            )

            # Store global values
            self.medcouples_[:] = mc
            self.lower_fences_[:] = lower
            self.upper_fences_[:] = upper

            outliers = (matrix.data < lower) | (matrix.data > upper)

        return outliers

    def _detect_adjusted_boxplot_within_group(self, matrix: BioMatrix) -> np.ndarray:
        """
        Detect outliers using adjusted boxplot fences within each phenotype group.

        For each group and each feature:
        1. Compute medcouple on that group's values
        2. Compute asymmetric fences
        3. Flag values outside fences

        Returns:
            Boolean outlier mask (n_features × n_samples)
        """
        n_features, n_samples = matrix.shape
        outliers = np.zeros((n_features, n_samples), dtype=bool)

        # Initialize diagnostic arrays with per-feature averages
        # (averaged across groups since fences differ per group)
        self.medcouples_ = np.zeros(n_features)
        self.lower_fences_ = np.zeros(n_features)
        self.upper_fences_ = np.zeros(n_features)
        group_counts = np.zeros(n_features)

        # Create composite group key
        if len(self.group_cols) == 1:
            group_series = matrix.sample_metadata[self.group_cols[0]]
        else:
            group_series = matrix.sample_metadata[self.group_cols].apply(tuple, axis=1)

        unique_groups = group_series.unique()

        for group in unique_groups:
            group_mask = (group_series == group).values
            n_in_group = group_mask.sum()

            # Skip groups too small for robust statistics
            if n_in_group < 10:
                warnings.warn(
                    f"Group {group} has only {n_in_group} samples (minimum 10 recommended). "
                    f"Skipping outlier detection for this group."
                )
                continue

            # Detect outliers per feature within this group
            for i in range(n_features):
                group_data = matrix.data[i, group_mask]

                mc = compute_medcouple(group_data)
                lower, upper = adjusted_boxplot_fences(
                    group_data,
                    mc=mc,
                    whis=self.threshold
                )

                # Flag outliers in this group
                group_outliers = (group_data < lower) | (group_data > upper)
                outliers[i, group_mask] = group_outliers

                # Accumulate diagnostics (will average later)
                self.medcouples_[i] += mc
                self.lower_fences_[i] += lower
                self.upper_fences_[i] += upper
                group_counts[i] += 1

        # Average diagnostics across groups
        valid_counts = np.maximum(group_counts, 1)  # Avoid division by zero
        self.medcouples_ /= valid_counts
        self.lower_fences_ /= valid_counts
        self.upper_fences_ /= valid_counts

        return outliers

    def _detect_student_t(self, matrix: BioMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Student's t probabilistic detection.

        Returns:
            (outlier_mask, p_values) where:
            - outlier_mask: Boolean array of outliers (p < threshold)
            - p_values: Matrix of p-values
        """
        # Fit Student's t with shared df
        df_shared, locations, scales = fit_student_t_shared(
            matrix.data,
            robust_location=True
        )

        # Compute p-values
        p_values = compute_outlier_probability(
            matrix.data,
            df=df_shared,
            locations=locations,
            scales=scales
        )

        # Threshold p-values to get binary mask
        outlier_mask = p_values < self.threshold

        # Store diagnostics
        self.df_shared_ = df_shared
        self.locations_ = locations
        self.scales_ = scales

        return outlier_mask, p_values

    def get_qq_plot_data(self, matrix: BioMatrix) -> Dict[str, np.ndarray]:
        """
        Generate QQ-plot data comparing residuals to fitted t-distribution.

        This diagnostic helps verify that the Student's t model fits the data well.

        Args:
            matrix: Input BioMatrix (same as used in apply())

        Returns:
            Dictionary with:
            - theoretical_quantiles: Expected quantiles from t-distribution
            - sample_quantiles: Observed quantiles from standardized data
            - df: Fitted degrees of freedom

        Raises:
            ValueError: If called before apply() or if scoring_method != "student_t"
        """
        if self.scoring_method != "student_t":
            raise ValueError("QQ-plot only available for scoring_method='student_t'")

        if self.df_shared_ is None:
            raise ValueError("Must call apply() before get_qq_plot_data()")

        # Standardize all data
        standardized = (matrix.data - self.locations_[:, np.newaxis]) / self.scales_[:, np.newaxis]
        standardized = standardized[np.isfinite(standardized)]

        # Sort for QQ-plot
        standardized = np.sort(standardized)
        n = len(standardized)

        # Theoretical quantiles from fitted t-distribution
        p = (np.arange(1, n + 1) - 0.5) / n
        theoretical = stats.t.ppf(p, df=self.df_shared_)

        return {
            'theoretical_quantiles': theoretical,
            'sample_quantiles': standardized,
            'df': self.df_shared_
        }
