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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

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
        mad-z (recommended): Median Absolute Deviation, threshold 5.0
        iqr: Interquartile Range (Tukey's method), threshold 1.5

    Args:
        method: "mad-z" (recommended) or "iqr"
        threshold: Detection threshold (5.0 for MAD-Z, 1.5 for IQR)
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
        threshold: float = 5.0,
        mode: Literal["within_group", "per_feature", "global"] = "within_group",
        group_cols: str | list[str] = "phenotype",
        upper_threshold: float | None = None,
        lower_threshold: float | None = None
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

        # Use threshold as default for both upper/lower if not specified
        effective_upper = upper_threshold if upper_threshold is not None else threshold
        effective_lower = lower_threshold if lower_threshold is not None else threshold

        super().__init__(
            name="OutlierDetector",
            params={
                "method": method, "threshold": threshold, "mode": mode,
                "group_cols": group_cols, "upper_threshold": effective_upper,
                "lower_threshold": effective_lower
            }
        )

        self.method = method
        self.threshold = threshold
        self.upper_threshold = effective_upper
        self.lower_threshold = effective_lower
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
        """MAD-Z outlier detection with asymmetric threshold support."""
        if per_feature:
            medians = np.median(data, axis=1, keepdims=True)
            mad = np.median(np.abs(data - medians), axis=1, keepdims=True)
            mad = np.where(mad == 0, 1.0, mad)  # Avoid division by zero
            scaled_mad = mad / 0.6745  # Convert to MAD scale
            # Asymmetric detection: separate thresholds for upper/lower
            upper_bound = medians + self.upper_threshold * scaled_mad
            lower_bound = medians - self.lower_threshold * scaled_mad
        else:
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            mad = mad if mad != 0 else 1.0
            scaled_mad = mad / 0.6745
            upper_bound = median + self.upper_threshold * scaled_mad
            lower_bound = median - self.lower_threshold * scaled_mad

        return (data > upper_bound) | (data < lower_bound)

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
        threshold: MAD-Z threshold for residuals (default: 5.0)

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
        threshold: float = 5.0
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
            - For mad-z: MAD-Z threshold (default: 5.0)
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
        threshold: float = 5.0,
        mode: Literal["within_group", "per_feature", "global"] = "within_group",
        group_cols: str | list[str] = "phenotype",
        scoring_method: Literal["binary", "student_t"] = "binary",
        upper_threshold: float | None = None,
        lower_threshold: float | None = None
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

        # Use threshold as default for both upper/lower if not specified
        effective_upper = upper_threshold if upper_threshold is not None else threshold
        effective_lower = lower_threshold if lower_threshold is not None else threshold

        super().__init__(
            name="AdaptiveOutlierDetector",
            params={
                "method": method,
                "threshold": threshold,
                "mode": mode,
                "group_cols": group_cols,
                "scoring_method": scoring_method,
                "upper_threshold": effective_upper,
                "lower_threshold": effective_lower
            }
        )

        self.method = method
        self.threshold = threshold
        self.upper_threshold = effective_upper
        self.lower_threshold = effective_lower
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
            group_cols=self.group_cols,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold
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


class MultiPassOutlierDetector(Transform):
    """
    Multi-pass outlier detection combining multiple complementary methods.

    This orchestrator applies multiple detection passes sequentially, combining
    their results. Different artifact mechanisms require different detection
    approaches:

    1. **Within-group detection** (adjusted boxplot): Catches statistical outliers
       relative to each feature's distribution within phenotype groups.

    2. **Residual-based detection**: Catches values that are extreme relative to
       expected row + column effects (additive model). Useful for high-end outliers
       missed by within-group detection.

    3. **Global percentile cap**: Catches physically implausible values like
       detector saturation, regardless of within-feature statistics.

    Scientific Rationale:
        - Adjusted boxplot handles skewed distributions common in omics data
        - Residual model detects cross-feature artifacts (batch effects)
        - Global cap catches hardware limits (saturation, detection floor)

    Example:
        >>> from cliquefinder.quality import MultiPassOutlierDetector
        >>>
        >>> detector = MultiPassOutlierDetector(
        ...     # Pass 1: within-group adjusted boxplot
        ...     detection_method="adjusted-boxplot",
        ...     detection_threshold=1.5,
        ...     group_cols=["phenotype"],
        ...     # Pass 2: residual-based
        ...     residual_enabled=True,
        ...     residual_threshold=4.25,
        ...     residual_high_end_only=True,
        ...     # Pass 3: global percentile cap
        ...     global_cap_enabled=True,
        ...     global_cap_percentile=99.95,
        ... )
        >>> flagged = detector.apply(matrix)
        >>> print(f"Total outliers: {detector.n_outliers_}")

    Attributes (set after apply):
        n_outliers_: Total outlier count across all passes
        pass1_count_: Outliers from within-group detection
        pass2_count_: Additional outliers from residual detection
        pass3_count_: Additional outliers from global cap
        global_threshold_: The computed global percentile threshold
        detector_: The underlying AdaptiveOutlierDetector (for medcouple access)
    """

    name = "MultiPassOutlierDetector"

    def __init__(
        self,
        # Pass 1: Within-group detection
        detection_method: str = "adjusted-boxplot",
        detection_threshold: float = 1.5,
        scoring_method: str = "binary",
        group_cols: list[str] | None = None,
        # Asymmetric threshold support (for upper-only mode)
        upper_threshold: float | None = None,
        lower_threshold: float | None = None,
        # Pass 2: Residual-based detection
        residual_enabled: bool = True,
        residual_threshold: float = 4.25,
        residual_high_end_only: bool = True,
        # Pass 3: Global percentile cap
        global_cap_enabled: bool = True,
        global_cap_percentile: float = 99.95,
    ):
        """
        Initialize multi-pass outlier detector.

        Args:
            detection_method: Method for Pass 1 ("adjusted-boxplot", "mad-z", "iqr")
            detection_threshold: Threshold for Pass 1 (1.5 for IQR, 5.0 for MAD-Z)
            scoring_method: Scoring for Pass 1 ("binary" or "student_t")
            group_cols: Columns to stratify by (e.g., ["phenotype"])

            upper_threshold: Override threshold for upper-tail detection (None = use detection_threshold)
            lower_threshold: Override threshold for lower-tail detection (None = use detection_threshold,
                           float('inf') = disable lower-tail for upper-only mode)

            residual_enabled: Whether to run Pass 2
            residual_threshold: MAD-Z threshold for residuals (default 4.25)
            residual_high_end_only: Only flag positive residuals (high values)

            global_cap_enabled: Whether to run Pass 3
            global_cap_percentile: Percentile threshold (default 99.95)
        """
        self.detection_method = detection_method
        self.detection_threshold = detection_threshold
        self.scoring_method = scoring_method
        self.group_cols = group_cols
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.residual_enabled = residual_enabled
        self.residual_threshold = residual_threshold
        self.residual_high_end_only = residual_high_end_only

        self.global_cap_enabled = global_cap_enabled
        self.global_cap_percentile = global_cap_percentile

        # Attributes set after apply()
        self.n_outliers_: int | None = None
        self.pass1_count_: int | None = None
        self.pass2_count_: int | None = None
        self.pass3_count_: int | None = None
        self.global_threshold_: float | None = None
        self.detector_: AdaptiveOutlierDetector | None = None

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Apply multi-pass outlier detection.

        Args:
            matrix: Input BioMatrix

        Returns:
            BioMatrix with OUTLIER_DETECTED flags set
        """
        errors = self.validate(matrix)
        if errors:
            raise ValueError(
                f"Validation failed for {self.name}:\n" +
                "\n".join(f"  - {err}" for err in errors)
            )

        # Pass 1: Within-group detection
        self.detector_ = AdaptiveOutlierDetector(
            method=self.detection_method,
            threshold=self.detection_threshold,
            mode="within_group" if self.group_cols else "per_feature",
            group_cols=self.group_cols,
            scoring_method=self.scoring_method,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold,
        )
        matrix_flagged = self.detector_.apply(matrix)

        # Track outlier mask
        outlier_mask = (matrix_flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
        self.pass1_count_ = int(outlier_mask.sum())

        # Pass 2: Residual-based detection
        self.pass2_count_ = 0
        if self.residual_enabled:
            # Compute additive model residuals
            row_medians = np.median(matrix.data, axis=1, keepdims=True)
            col_medians = np.median(matrix.data, axis=0, keepdims=True)
            grand_median = np.median(matrix.data)
            expected = row_medians + col_medians - grand_median
            residuals = matrix.data - expected

            # MAD-based z-scores on residuals
            residual_median = np.median(residuals)
            residual_mad = np.median(np.abs(residuals - residual_median)) * 1.4826
            if residual_mad > 0:
                z_residuals = (residuals - residual_median) / residual_mad

                # Flag high residuals
                if self.residual_high_end_only:
                    residual_outliers = z_residuals > self.residual_threshold
                else:
                    residual_outliers = np.abs(z_residuals) > self.residual_threshold

                # Count new outliers (not already flagged)
                new_from_residual = residual_outliers & (~outlier_mask)
                self.pass2_count_ = int(new_from_residual.sum())

                # Combine masks
                outlier_mask = outlier_mask | residual_outliers

                # Update quality flags
                new_flags = matrix_flagged.quality_flags.copy()
                new_flags[residual_outliers] = new_flags[residual_outliers] | np.uint8(QualityFlag.OUTLIER_DETECTED)
                matrix_flagged = BioMatrix(
                    data=matrix_flagged.data,
                    feature_ids=matrix_flagged.feature_ids,
                    sample_ids=matrix_flagged.sample_ids,
                    sample_metadata=matrix_flagged.sample_metadata,
                    quality_flags=new_flags,
                )

        # Pass 3: Global percentile cap
        self.pass3_count_ = 0
        if self.global_cap_enabled:
            self.global_threshold_ = np.percentile(matrix.data, self.global_cap_percentile)
            global_outliers = matrix.data > self.global_threshold_

            # Count new outliers
            new_from_global = global_outliers & (~outlier_mask)
            self.pass3_count_ = int(new_from_global.sum())

            # Combine masks
            outlier_mask = outlier_mask | global_outliers

            # Update quality flags
            new_flags = matrix_flagged.quality_flags.copy()
            new_flags[global_outliers] = new_flags[global_outliers] | np.uint8(QualityFlag.OUTLIER_DETECTED)
            matrix_flagged = BioMatrix(
                data=matrix_flagged.data,
                feature_ids=matrix_flagged.feature_ids,
                sample_ids=matrix_flagged.sample_ids,
                sample_metadata=matrix_flagged.sample_metadata,
                quality_flags=new_flags,
            )

        self.n_outliers_ = int(outlier_mask.sum())

        return matrix_flagged

    def validate(self, matrix: BioMatrix) -> list[str]:
        """Validate input matrix."""
        errors = []

        if matrix.data.size == 0:
            errors.append("Matrix has no data")

        if self.group_cols:
            for col in self.group_cols:
                if col not in matrix.sample_metadata.columns:
                    errors.append(f"Group column '{col}' not in sample_metadata")

        if not 0 < self.global_cap_percentile <= 100:
            errors.append(f"global_cap_percentile must be in (0, 100], got {self.global_cap_percentile}")

        return errors

    def get_pass_summary(self) -> dict:
        """
        Get summary of outliers detected by each pass.

        Returns:
            Dict with counts and percentages per pass
        """
        if self.n_outliers_ is None:
            raise ValueError("Must call apply() before get_pass_summary()")

        return {
            'pass1_within_group': self.pass1_count_,
            'pass2_residual': self.pass2_count_,
            'pass3_global_cap': self.pass3_count_,
            'total': self.n_outliers_,
            'global_threshold': self.global_threshold_,
        }


class KDEAdaptiveOutlierDetector(Transform):
    """
    Adaptive outlier detection using KDE-based threshold selection.

    Instead of a fixed MAD-Z threshold, this detector finds the natural
    "elbow" point in the distribution where the tail begins, adapting
    to each stratum's actual distribution.

    Algorithm:
        1. Compute MAD-Z scores for each gene within each stratum
        2. Pool all MAD-Z scores for the stratum
        3. Use adaptive KDE to estimate the density
        4. Find cumulative density tail (e.g., top 0.5%)
        5. Detect inflection point via gradient analysis
        6. Use inflection point as adaptive threshold

    This is more robust than fixed thresholds because:
        - Heavy-tailed distributions get higher thresholds automatically
        - Light-tailed distributions get lower thresholds
        - Each stratum gets its own adaptive threshold

    Parameters:
        group_cols: Columns for stratification (default: ['phenotype'])
        upper_only: Only detect upper-tail outliers (default: True)
        k_neighbors: Neighbors for adaptive KDE bandwidth (default: 10)
        cumulative_density_threshold: Tail start threshold (default: 0.005)
        gradient_sensitivity: Sensitivity for inflection detection (default: 0.005)
        min_threshold: Minimum threshold to prevent over-detection (default: 2.0)
        max_threshold: Maximum threshold to ensure some detection (default: 5.0)
        fallback_threshold: Threshold if KDE fails (default: 3.0)

    Attributes:
        thresholds_: Dict mapping stratum -> detected threshold
        n_outliers_: Total outliers detected
        stratum_counts_: Dict mapping stratum -> outlier count

    References:
        Based on the adaptive KDE thresholding approach from case-control-genomics
        feature selection (detect_beta_tail_threshold).
    """

    def __init__(
        self,
        group_cols: list[str] | None = None,
        upper_only: bool = True,
        k_neighbors: int = 10,
        cumulative_density_threshold: float = 0.005,
        gradient_sensitivity: float = 0.005,
        min_threshold: float = 2.0,
        max_threshold: float = 5.0,
        fallback_threshold: float = 3.0,
    ):
        self.group_cols = group_cols or ['phenotype']
        self.upper_only = upper_only
        self.k_neighbors = k_neighbors
        self.cumulative_density_threshold = cumulative_density_threshold
        self.gradient_sensitivity = gradient_sensitivity
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.fallback_threshold = fallback_threshold

        # Fitted attributes
        self.thresholds_: dict[str, float] | None = None
        self.n_outliers_: int | None = None
        self.stratum_counts_: dict[str, int] | None = None
        self.kde_diagnostics_: dict[str, dict] | None = None

    def _detect_kde_threshold(
        self,
        mad_z_scores: np.ndarray,
        stratum_name: str
    ) -> tuple[float, dict]:
        """
        Detect adaptive threshold using KDE inflection point.

        Returns:
            (threshold, diagnostics_dict)
        """
        diagnostics = {
            'n_values': len(mad_z_scores),
            'method': 'kde_inflection',
        }

        # Filter to positive (upper-tail) values for upper_only mode
        if self.upper_only:
            scores = mad_z_scores[mad_z_scores > 0]
        else:
            scores = np.abs(mad_z_scores)

        if len(scores) < 50:
            # Not enough data for reliable KDE
            diagnostics['method'] = 'fallback_insufficient_data'
            return self.fallback_threshold, diagnostics

        # Create evaluation grid
        x_min, x_max = scores.min(), scores.max()
        x_grid = np.linspace(x_min, x_max, 500)

        # Calculate adaptive bandwidths using k-nearest neighbors
        k = min(self.k_neighbors, len(scores) - 1)
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(scores.reshape(-1, 1))
        distances, _ = nbrs.kneighbors(scores.reshape(-1, 1))
        bandwidths = distances[:, -1]
        bandwidths = np.clip(bandwidths, 0.01, None)  # Avoid zero bandwidth

        # Estimate KDE with adaptive bandwidths
        kde_values = np.zeros_like(x_grid)
        for i, point in enumerate(scores):
            kde_values += norm.pdf(x_grid, loc=point, scale=bandwidths[i])
        kde_values /= len(scores)

        # Smooth the KDE
        kde_smooth = gaussian_filter1d(kde_values, sigma=1)

        # Calculate cumulative density from the right (upper tail)
        dx = x_grid[1] - x_grid[0]
        cumulative_density = np.cumsum(kde_smooth[::-1] * dx)[::-1]

        # Find start of tail region
        tail_indices = np.where(cumulative_density <= self.cumulative_density_threshold)[0]
        if len(tail_indices) > 0:
            tail_start_idx = tail_indices[0]
        else:
            # Fallback: use 95th percentile as tail start
            tail_start_idx = int(0.95 * len(x_grid))

        # Calculate gradient
        gradient = np.gradient(kde_smooth)
        gradient_smooth = gaussian_filter1d(gradient, sigma=1)

        # Find inflection point in tail region
        threshold_idx = None
        window_points = 5

        for i in range(tail_start_idx, len(gradient_smooth) - window_points):
            window = gradient_smooth[i:i + window_points]
            # Check for sustained negative gradient (density falling off)
            if np.mean(window) < -self.gradient_sensitivity and np.all(window < 0):
                threshold_idx = i
                break

        if threshold_idx is None:
            # Fallback: use tail start
            threshold_idx = tail_start_idx
            diagnostics['method'] = 'kde_tail_start'

        threshold = x_grid[threshold_idx]

        # Apply bounds
        threshold = np.clip(threshold, self.min_threshold, self.max_threshold)

        diagnostics.update({
            'raw_threshold': x_grid[threshold_idx] if threshold_idx < len(x_grid) else None,
            'tail_start': x_grid[tail_start_idx] if tail_start_idx < len(x_grid) else None,
            'final_threshold': threshold,
            'kde_peak': x_grid[np.argmax(kde_smooth)],
            'score_range': (float(x_min), float(x_max)),
        })

        return threshold, diagnostics

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Apply adaptive KDE-based outlier detection.

        Returns:
            BioMatrix with quality_flags updated
        """
        # Initialize outlier mask
        outlier_mask = np.zeros((matrix.n_features, matrix.n_samples), dtype=bool)

        # Build stratum labels
        if self.group_cols:
            strata = matrix.sample_metadata[self.group_cols].apply(
                lambda row: '_'.join(str(v) for v in row), axis=1
            )
        else:
            strata = pd.Series(['all'] * matrix.n_samples, index=matrix.sample_ids)

        unique_strata = strata.unique()
        self.thresholds_ = {}
        self.stratum_counts_ = {}
        self.kde_diagnostics_ = {}

        print(f"\nKDE Adaptive Outlier Detection:")
        print(f"  Stratification: {self.group_cols}")
        print(f"  Upper-only: {self.upper_only}")

        for stratum in unique_strata:
            sample_mask = (strata == stratum).values
            stratum_data = matrix.data[:, sample_mask]

            # Compute MAD-Z scores for all genes in this stratum
            all_mad_z = []
            gene_mad_z = []  # Store per-gene MAD-Z for flagging

            for gene_idx in range(stratum_data.shape[0]):
                gene_values = stratum_data[gene_idx, :]

                # Skip genes with insufficient valid values
                valid_mask = np.isfinite(gene_values) & (gene_values > 0)
                if valid_mask.sum() < 3:
                    gene_mad_z.append(np.full(sample_mask.sum(), np.nan))
                    continue

                valid_values = gene_values[valid_mask]
                median = np.median(valid_values)
                mad = np.median(np.abs(valid_values - median))

                if mad < 1e-10:
                    gene_mad_z.append(np.full(sample_mask.sum(), np.nan))
                    continue

                # Compute MAD-Z scores (scaled to be comparable to Z-scores)
                scaled_mad = mad * 1.4826
                z_scores = (gene_values - median) / scaled_mad
                gene_mad_z.append(z_scores)

                # Collect valid scores for KDE
                all_mad_z.extend(z_scores[valid_mask].tolist())

            all_mad_z = np.array(all_mad_z)

            # Detect adaptive threshold for this stratum
            threshold, diagnostics = self._detect_kde_threshold(all_mad_z, stratum)
            self.thresholds_[stratum] = threshold
            self.kde_diagnostics_[stratum] = diagnostics

            # Apply threshold to flag outliers
            stratum_outliers = 0
            sample_indices = np.where(sample_mask)[0]

            for gene_idx, z_scores in enumerate(gene_mad_z):
                if z_scores is None or np.all(np.isnan(z_scores)):
                    continue

                # Flag based on upper_only setting
                if self.upper_only:
                    gene_outliers = z_scores > threshold
                else:
                    gene_outliers = np.abs(z_scores) > threshold

                # Map back to full outlier mask
                for local_idx, global_idx in enumerate(sample_indices):
                    if gene_outliers[local_idx]:
                        outlier_mask[gene_idx, global_idx] = True
                        stratum_outliers += 1

            self.stratum_counts_[stratum] = stratum_outliers

            print(f"\n  {stratum}:")
            print(f"    Samples: {sample_mask.sum()}")
            print(f"    Adaptive threshold: {threshold:.2f}")
            print(f"    Method: {diagnostics['method']}")
            print(f"    Outliers: {stratum_outliers:,}")

        self.n_outliers_ = outlier_mask.sum()

        # Update quality flags
        if matrix.quality_flags is None:
            new_flags = np.zeros_like(outlier_mask, dtype=np.uint8)
        else:
            new_flags = matrix.quality_flags.copy()

        new_flags[outlier_mask] |= QualityFlag.OUTLIER_DETECTED

        return BioMatrix(
            data=matrix.data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags
        )

    def validate(self, matrix: BioMatrix) -> list[str]:
        """Validate detector configuration."""
        errors = []

        if self.group_cols:
            for col in self.group_cols:
                if col not in matrix.sample_metadata.columns:
                    errors.append(f"Group column '{col}' not in sample_metadata")

        if self.k_neighbors < 1:
            errors.append(f"k_neighbors must be >= 1, got {self.k_neighbors}")

        if not 0 < self.cumulative_density_threshold < 1:
            errors.append(f"cumulative_density_threshold must be in (0, 1)")

        return errors

    def get_threshold_summary(self) -> dict:
        """
        Get summary of adaptive thresholds per stratum.

        Returns:
            Dict with threshold info and diagnostics
        """
        if self.thresholds_ is None:
            raise ValueError("Must call apply() before get_threshold_summary()")

        return {
            'thresholds': self.thresholds_,
            'stratum_counts': self.stratum_counts_,
            'total_outliers': self.n_outliers_,
            'diagnostics': self.kde_diagnostics_,
        }
