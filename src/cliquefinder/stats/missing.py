"""
Missing value handling for proteomics data.

Implements MSstats-style missing value handling, distinguishing between:
- MNAR (Missing Not At Random): Censored values below detection limit
- MAR (Missing At Random): Random technical failures

The key insight from MSstats: in proteomics, most missing values are
censored (MNAR) - they are missing because the true abundance is below
the detection threshold. This requires different statistical treatment
than random missingness.

The Accelerated Failure Time (AFT) model treats observed intensities as
right-censored survival data, where the "event" is detection and the
"censoring" is the detection limit.

References:
    - Choi et al. (2014) MSstats: Bioinformatics 30(17):2524-2526
    - Wei et al. (2018) IJMS 19(10):3099 (missing value mechanisms)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize_scalar


class MissingMechanism(Enum):
    """Missing value mechanism."""

    MNAR = "mnar"  # Missing Not At Random (censored)
    MAR = "mar"    # Missing At Random
    MIXED = "mixed"  # Some MNAR, some MAR


class ImputationMethod(Enum):
    """Imputation methods for missing values."""

    NONE = "none"
    MIN_FEATURE = "min_feature"  # Per-feature minimum (MSstats default)
    MIN_GLOBAL = "min_global"    # Global minimum
    MIN_SAMPLE = "min_sample"    # Per-sample minimum
    AFT = "aft"                  # Accelerated Failure Time model
    KNN = "knn"                  # K-nearest neighbors
    QRILC = "qrilc"             # Quantile regression for left-censored


@dataclass(frozen=True)
class MissingValueAnalysis:
    """Analysis of missing value patterns.

    Attributes:
        n_missing: Total number of missing values
        missing_rate: Overall missing rate
        missing_per_feature: Missing count per feature (row)
        missing_per_sample: Missing count per sample (column)
        estimated_mechanism: Estimated missing mechanism
        censoring_threshold: Estimated detection limit
    """

    n_missing: int
    missing_rate: float
    missing_per_feature: NDArray[np.int_]
    missing_per_sample: NDArray[np.int_]
    estimated_mechanism: MissingMechanism
    censoring_threshold: float | None


@dataclass
class ImputationResult:
    """Result of missing value imputation.

    Attributes:
        data: Imputed data matrix
        method: Imputation method used
        n_imputed: Number of values imputed
        imputation_mask: Boolean mask of imputed positions
        censoring_threshold: Detection threshold used (for AFT)
        diagnostics: Additional diagnostic information
    """

    data: NDArray[np.float64]
    method: str
    n_imputed: int
    imputation_mask: NDArray[np.bool_]
    censoring_threshold: float | None = None
    diagnostics: dict | None = None


def analyze_missing_values(
    data: NDArray[np.float64],
    intensity_threshold: float | None = None,
) -> MissingValueAnalysis:
    """
    Analyze missing value patterns to determine mechanism.

    Uses heuristics to distinguish MNAR (censored) from MAR:
    - MNAR: Missing values cluster in low-abundance features/samples
    - MAR: Missing values are randomly distributed

    Args:
        data: 2D array (n_features, n_samples) with NaN for missing.
        intensity_threshold: If provided, values below this are also
            considered missing (for data where 0 = below detection).

    Returns:
        MissingValueAnalysis with pattern characterization.
    """
    # Create missing mask
    missing_mask = np.isnan(data)
    if intensity_threshold is not None:
        missing_mask = missing_mask | (data <= intensity_threshold)

    n_missing = int(np.sum(missing_mask))
    missing_rate = n_missing / data.size

    missing_per_feature = np.sum(missing_mask, axis=1).astype(np.int_)
    missing_per_sample = np.sum(missing_mask, axis=0).astype(np.int_)

    # Estimate mechanism based on correlation with abundance
    # MNAR: features with more missing have lower mean (when observed)
    feature_means = np.nanmean(data, axis=1)
    feature_missing_rates = missing_per_feature / data.shape[1]

    # Correlation between missing rate and mean abundance
    valid_features = ~np.isnan(feature_means) & (feature_missing_rates < 1.0)
    if np.sum(valid_features) > 10:
        corr = np.corrcoef(
            feature_means[valid_features],
            feature_missing_rates[valid_features]
        )[0, 1]

        # Strong negative correlation → MNAR (low abundance → more missing)
        if corr < -0.3:
            mechanism = MissingMechanism.MNAR
        elif corr > 0.3:
            # Unusual: high abundance → more missing (likely MAR or technical)
            mechanism = MissingMechanism.MAR
        else:
            mechanism = MissingMechanism.MIXED
    else:
        mechanism = MissingMechanism.MIXED

    # Estimate censoring threshold (for MNAR)
    censoring_threshold = None
    if mechanism in (MissingMechanism.MNAR, MissingMechanism.MIXED):
        # Use low percentile of observed values as threshold estimate
        observed = data[~missing_mask]
        if len(observed) > 0:
            censoring_threshold = float(np.percentile(observed, 0.1))

    return MissingValueAnalysis(
        n_missing=n_missing,
        missing_rate=missing_rate,
        missing_per_feature=missing_per_feature,
        missing_per_sample=missing_per_sample,
        estimated_mechanism=mechanism,
        censoring_threshold=censoring_threshold,
    )


def estimate_censoring_threshold(
    data: NDArray[np.float64],
    method: Literal["min_feature", "percentile", "learned"] = "min_feature",
    percentile: float = 0.1,
) -> float:
    """
    Estimate the censoring threshold (detection limit).

    Args:
        data: 2D array with NaN for missing values.
        method: Estimation method:
            - "min_feature": Per-feature minimum (MSstats default)
            - "percentile": Global percentile of observed values
            - "learned": MSstats v4 learned threshold (0.1th percentile)

    Returns:
        Estimated censoring threshold.
    """
    observed = data[~np.isnan(data)]

    if len(observed) == 0:
        return 0.0

    if method == "min_feature":
        # This returns different thresholds per feature
        # For a global threshold, take the median of per-feature mins
        feature_mins = np.nanmin(data, axis=1)
        return float(np.nanmedian(feature_mins))

    elif method == "percentile":
        return float(np.percentile(observed, percentile))

    elif method == "learned":
        # MSstats v4 uses 0.1th percentile
        return float(np.percentile(observed, 0.1))

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def impute_min_value(
    data: NDArray[np.float64],
    method: Literal["feature", "global", "sample"] = "feature",
) -> ImputationResult:
    """
    Simple minimum value imputation.

    Replaces missing values with the minimum observed value,
    which represents the detection limit.

    Args:
        data: 2D array with NaN for missing.
        method: How to determine the minimum:
            - "feature": Per-feature minimum (MSstats cutoffCensored="minFeature")
            - "global": Global minimum across all values
            - "sample": Per-sample minimum

    Returns:
        ImputationResult with imputed data.
    """
    imputed = data.copy()
    missing_mask = np.isnan(data)
    n_imputed = int(np.sum(missing_mask))

    if method == "feature":
        # Per-feature minimum
        feature_mins = np.nanmin(data, axis=1)
        for i in range(data.shape[0]):
            if np.isnan(feature_mins[i]):
                # All values missing for this feature - use global min
                feature_mins[i] = np.nanmin(data)
            imputed[i, missing_mask[i, :]] = feature_mins[i]
        threshold = float(np.nanmedian(feature_mins))

    elif method == "global":
        global_min = np.nanmin(data)
        imputed[missing_mask] = global_min
        threshold = float(global_min)

    elif method == "sample":
        sample_mins = np.nanmin(data, axis=0)
        for j in range(data.shape[1]):
            if np.isnan(sample_mins[j]):
                sample_mins[j] = np.nanmin(data)
            imputed[missing_mask[:, j], j] = sample_mins[j]
        threshold = float(np.nanmedian(sample_mins))

    else:
        raise ValueError(f"Unknown method: {method}")

    return ImputationResult(
        data=imputed,
        method=f"min_{method}",
        n_imputed=n_imputed,
        imputation_mask=missing_mask,
        censoring_threshold=threshold,
    )


def impute_aft_model(
    data: NDArray[np.float64],
    censoring_threshold: float | None = None,
    n_draws: int = 1,
    random_state: int | None = None,
) -> ImputationResult:
    """
    Accelerated Failure Time (AFT) model imputation for censored data.

    Models the observed intensities as coming from a truncated normal
    distribution, where values below the censoring threshold are missing.

    For each missing value, draws from the conditional distribution:
        X | X < threshold ~ TruncatedNormal(mu, sigma, -inf, threshold)

    where mu and sigma are estimated from observed values.

    This is the model-based approach used by MSstats (MBimpute=TRUE).

    Args:
        data: 2D array with NaN for missing (censored) values.
        censoring_threshold: Detection limit. If None, estimated from data.
        n_draws: Number of random draws (1 for point estimate, >1 for MI).
        random_state: Random seed for reproducibility.

    Returns:
        ImputationResult with imputed data.

    Note:
        This implementation uses a per-feature AFT model. MSstats also
        considers run effects, which we simplify here.
    """
    rng = np.random.default_rng(random_state)

    imputed = data.copy()
    missing_mask = np.isnan(data)
    n_imputed = int(np.sum(missing_mask))

    # Estimate censoring threshold if not provided
    if censoring_threshold is None:
        censoring_threshold = estimate_censoring_threshold(data, method="learned")

    n_features, n_samples = data.shape

    # Per-feature imputation
    for i in range(n_features):
        feature_data = data[i, :]
        feature_missing = missing_mask[i, :]

        if not np.any(feature_missing):
            continue

        observed = feature_data[~feature_missing]

        if len(observed) < 2:
            # Not enough data - use global parameters
            all_observed = data[~missing_mask]
            mu = np.nanmean(all_observed) if len(all_observed) > 0 else censoring_threshold
            sigma = np.nanstd(all_observed) if len(all_observed) > 1 else 1.0
        else:
            mu = np.mean(observed)
            sigma = np.std(observed)
            if sigma < 1e-10:
                sigma = 1.0  # Prevent division by zero

        # Draw from truncated normal (below threshold)
        # Using inverse CDF method
        n_missing_feature = int(np.sum(feature_missing))

        # CDF at threshold
        phi_threshold = stats.norm.cdf(censoring_threshold, loc=mu, scale=sigma)

        if phi_threshold < 1e-10:
            # Threshold is far below the distribution - use threshold directly
            imputed_values = np.full(n_missing_feature, censoring_threshold)
        else:
            # Draw uniform on (0, phi_threshold) and invert
            u = rng.uniform(0, phi_threshold, size=n_missing_feature)
            imputed_values = stats.norm.ppf(u, loc=mu, scale=sigma)

            # Ensure values are below threshold (numerical safety)
            imputed_values = np.minimum(imputed_values, censoring_threshold)

        imputed[i, feature_missing] = imputed_values

    return ImputationResult(
        data=imputed,
        method="aft",
        n_imputed=n_imputed,
        imputation_mask=missing_mask,
        censoring_threshold=censoring_threshold,
        diagnostics={
            "n_draws": n_draws,
            "random_state": random_state,
        },
    )


def impute_qrilc(
    data: NDArray[np.float64],
    tune_sigma: float = 1.0,
    random_state: int | None = None,
) -> ImputationResult:
    """
    Quantile Regression Imputation of Left-Censored data (QRILC).

    Imputes missing values by:
    1. Estimating distribution parameters from observed values
    2. Drawing from the left tail of the distribution

    This is similar to AFT but uses quantile regression to estimate
    the parameters, making it more robust to outliers.

    Args:
        data: 2D array with NaN for missing values.
        tune_sigma: Scaling factor for imputation variance (default 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        ImputationResult with imputed data.
    """
    rng = np.random.default_rng(random_state)

    imputed = data.copy()
    missing_mask = np.isnan(data)
    n_imputed = int(np.sum(missing_mask))

    n_features, n_samples = data.shape

    # Global parameters as fallback
    all_observed = data[~missing_mask]
    global_mu = np.mean(all_observed) if len(all_observed) > 0 else 0.0
    global_sigma = np.std(all_observed) if len(all_observed) > 1 else 1.0

    # Per-sample imputation (captures sample-level technical variation)
    for j in range(n_samples):
        sample_data = data[:, j]
        sample_missing = missing_mask[:, j]

        if not np.any(sample_missing):
            continue

        observed = sample_data[~sample_missing]

        if len(observed) < 3:
            mu = global_mu
            sigma = global_sigma
        else:
            # Use robust estimates
            mu = np.median(observed)
            sigma = stats.median_abs_deviation(observed, scale='normal')
            if sigma < 1e-10:
                sigma = global_sigma

        # Impute from left tail
        n_missing_sample = int(np.sum(sample_missing))

        # Determine quantile range for imputation
        # Missing values should be below the minimum observed
        min_observed = np.min(observed) if len(observed) > 0 else mu - 2 * sigma

        # Quantile of minimum observed value
        q_min = stats.norm.cdf(min_observed, loc=mu, scale=sigma * tune_sigma)

        # Draw from left tail (below q_min)
        q_min = max(q_min, 0.001)  # Ensure we have some range
        u = rng.uniform(0, q_min, size=n_missing_sample)
        imputed_values = stats.norm.ppf(u, loc=mu, scale=sigma * tune_sigma)

        imputed[sample_missing, j] = imputed_values

    # Estimate effective censoring threshold
    censoring_threshold = float(np.nanpercentile(data, 1))

    return ImputationResult(
        data=imputed,
        method="qrilc",
        n_imputed=n_imputed,
        imputation_mask=missing_mask,
        censoring_threshold=censoring_threshold,
        diagnostics={
            "tune_sigma": tune_sigma,
        },
    )


def impute_knn(
    data: NDArray[np.float64],
    k: int = 10,
    weights: Literal["uniform", "distance"] = "distance",
) -> ImputationResult:
    """
    K-nearest neighbors imputation.

    Imputes missing values based on similar features (proteins).
    Appropriate when missing values are MAR (random technical failures).

    Note: This assumes MAR mechanism. For MNAR (censored), use AFT instead.

    Args:
        data: 2D array with NaN for missing values.
        k: Number of nearest neighbors.
        weights: Weighting scheme for neighbors.

    Returns:
        ImputationResult with imputed data.
    """
    from sklearn.impute import KNNImputer

    missing_mask = np.isnan(data)
    n_imputed = int(np.sum(missing_mask))

    # KNNImputer expects (n_samples, n_features), but our data is (n_features, n_samples)
    # We want to impute based on similar proteins, so transpose
    imputer = KNNImputer(n_neighbors=k, weights=weights)

    # Impute on transposed data (samples as rows, features as columns)
    # Then transpose back
    imputed_t = imputer.fit_transform(data.T)
    imputed = imputed_t.T

    return ImputationResult(
        data=imputed,
        method="knn",
        n_imputed=n_imputed,
        imputation_mask=missing_mask,
        censoring_threshold=None,
        diagnostics={
            "k": k,
            "weights": weights,
        },
    )


def impute_missing_values(
    data: NDArray[np.float64],
    method: ImputationMethod | str = ImputationMethod.AFT,
    censored_indicator: float | str | None = None,
    **kwargs,
) -> ImputationResult:
    """
    Impute missing values using specified method.

    This is the main entry point for missing value imputation.

    Args:
        data: 2D array (n_features, n_samples).
        method: Imputation method to use.
        censored_indicator: How missing values are encoded:
            - None: NaN indicates missing (default)
            - 0 or "0": Zero indicates missing/censored
            - float: Values <= this are considered censored
        **kwargs: Additional arguments passed to the imputation method.

    Returns:
        ImputationResult with imputed data.
    """
    if isinstance(method, str):
        method = ImputationMethod(method)

    # Standardize missing value encoding to NaN
    work_data = data.astype(np.float64).copy()

    if censored_indicator is not None:
        if censored_indicator == "0" or censored_indicator == 0:
            work_data[work_data == 0] = np.nan
        elif isinstance(censored_indicator, (int, float)):
            work_data[work_data <= censored_indicator] = np.nan

    if method == ImputationMethod.NONE:
        missing_mask = np.isnan(work_data)
        return ImputationResult(
            data=work_data,
            method="none",
            n_imputed=0,
            imputation_mask=missing_mask,
        )

    elif method == ImputationMethod.MIN_FEATURE:
        return impute_min_value(work_data, method="feature")

    elif method == ImputationMethod.MIN_GLOBAL:
        return impute_min_value(work_data, method="global")

    elif method == ImputationMethod.MIN_SAMPLE:
        return impute_min_value(work_data, method="sample")

    elif method == ImputationMethod.AFT:
        return impute_aft_model(work_data, **kwargs)

    elif method == ImputationMethod.KNN:
        return impute_knn(work_data, **kwargs)

    elif method == ImputationMethod.QRILC:
        return impute_qrilc(work_data, **kwargs)

    else:
        raise ValueError(f"Unknown imputation method: {method}")
