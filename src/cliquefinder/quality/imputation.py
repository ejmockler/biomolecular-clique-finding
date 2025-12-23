"""
Value imputation for missing and outlier data in proteomics/transcriptomics.

This module provides imputation transforms that replace missing or outlier-flagged
values with statistically estimated values while tracking provenance.

Biological Context:
    Why Imputation is Necessary:

    1. Missing Values in Omics Data:
       - Mass spec: Proteins below detection limit (left-censored data)
       - RNA-seq: Zero counts (biological zeros vs technical dropouts)
       - Proteomics: ~20-30% missingness common

    2. Outlier Handling:
       - Can't just remove outliers → creates new missing values
       - Can't keep outliers → ruins correlations and statistical tests
       - Must impute to maintain matrix structure for downstream analysis

    3. Downstream Requirements:
       - Correlation networks: Need complete data (Pearson breaks with NaN)
       - PCA/clustering: Most algorithms require complete matrices
       - Statistical tests: Different handling for complete vs incomplete data

Why MAD-Clip:
    MAD-Clip (Median Absolute Deviation Clipping):
        - Pro: Consistent with MAD-Z detection threshold
        - Pro: Clips to median ± threshold × MAD (preserves distribution shape)
        - Pro: No circularity with correlation analysis
        - Con: Conservative (clips to bounds, reduces variance)

    Simple Median:
        - Pro: Fast, deterministic
        - Con: Ignores gene relationships (uses same median for all samples)
        - Con: Reduces variance (all imputed values identical per gene)

Statistical Foundation:
    Imputation introduces synthetic data. Always track with QualityFlag.IMPUTED.

Caveats and Warnings:
    1. Imputation introduces synthetic data
       - Always track with QualityFlag.IMPUTED
       - Consider sensitivity analysis (vary k)

    2. Missingness mechanisms matter
       - MCAR (Missing Completely At Random): Safe to impute
       - MAR (Missing At Random): Careful imputation
       - MNAR (Missing Not At Random): Imputation may introduce bias!
         Example: Low-abundance proteins below detection limit

    3. Over-imputation risks
       - >50% missingness: Imputation unreliable
       - Creates false precision

References:
    - Troyanskaya et al. (2001) "Missing value estimation methods for DNA microarrays"
    - Lazar et al. (2016) "Accounting for the Multiple Natures of Missing Values in
      Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies"

Examples:
    >>> import numpy as np
    >>> from cliquefinder.core.biomatrix import BioMatrix
    >>> from cliquefinder.quality.imputation import Imputer
    >>> from cliquefinder.quality.outliers import OutlierDetector
    >>> from cliquefinder.core.quality import QualityFlag
    >>>
    >>> # Typical workflow: detect then impute
    >>> detector = OutlierDetector(method="mad-z", threshold=3.5)
    >>> flagged = detector.apply(matrix)
    >>>
    >>> # Impute outliers using MAD-clip (RECOMMENDED)
    >>> imputer = Imputer(strategy="mad-clip", threshold=3.5)
    >>> imputed = imputer.apply(flagged)
    >>>
    >>> # Verify outliers were imputed
    >>> outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
    >>> imputed_mask = (imputed.quality_flags & QualityFlag.IMPUTED) > 0
    >>> assert np.array_equal(outlier_mask, imputed_mask)
    >>>
    >>> # Original values should be different for outliers
    >>> assert not np.array_equal(matrix.data[outlier_mask], imputed.data[outlier_mask])
"""

from __future__ import annotations

import numpy as np
import warnings
from typing import Optional

from cliquefinder.core.transform import Transform
from cliquefinder.core.quality import QualityFlag
from cliquefinder.core.biomatrix import BioMatrix

__all__ = ["Imputer", "soft_clip", "soft_clip_per_feature"]


def soft_clip(
    x: np.ndarray,
    lower: float,
    upper: float,
    sharpness: Optional[float] = None
) -> np.ndarray:
    """
    Apply soft (sigmoid) clipping to array.

    Uses tanh-based formulation for numerical stability.
    Preserves rank ordering unlike hard clipping.

    Mathematical Formulation:
        soft_clip(x; L, U, k) = c + (U - L)/2 × tanh(k × (x - c))

        where:
            c = (L + U) / 2  (center point)
            k = sharpness parameter (higher = sharper transition)

    Properties:
        - For x ≪ L: soft_clip(x) → L
        - For x ≫ U: soft_clip(x) → U
        - For L < x < U: soft_clip(x) ≈ x (near-linear in middle)
        - Strictly monotonic: preserves rank ordering
        - C∞ continuous: infinitely differentiable

    Args:
        x: Input array (any shape)
        lower: Lower bound
        upper: Upper bound
        sharpness: Transition sharpness. If None, uses 2/(upper-lower)
            - Lower values = gentler transition
            - Higher values = sharper (approaches hard clip)
            - Recommended: k ≈ 2/(U-L) for correlation analysis
            - Standard: k ≈ 4/(U-L) gives 95% of range with slope > 0.5
            - Sharp: k ≈ 8/(U-L) approaches hard clipping

    Returns:
        Soft-clipped array with same shape

    Raises:
        ValueError: If lower >= upper

    Notes:
        - Strictly monotonic: preserves rank ordering
        - Differentiable everywhere (unlike hard clipping)
        - For correlation analysis, use sharpness ≈ 2/(upper-lower)
        - Numerically stable for all finite inputs

    Scientific Rationale:
        Soft clipping addresses three problems with hard clipping:

        1. Boundary spikes: Hard clipping creates discontinuities where many
           values pile up at exact threshold values, distorting distributions.
           Soft clipping gradually compresses extreme values.

        2. Non-differentiability: Hard clipping has discontinuous derivatives
           at boundaries, problematic for optimization and gradient-based methods.
           Soft clipping is C∞ (infinitely differentiable).

        3. Rank distortion: Hard clipping ties originally different values at
           boundaries. Soft clipping is strictly monotonic, preserving ordering.

    Examples:
        >>> import numpy as np
        >>> x = np.array([-10, -5, 0, 5, 10])
        >>>
        >>> # Gentle soft clip (recommended for correlation)
        >>> soft = soft_clip(x, lower=-3, upper=3, sharpness=None)
        >>> # Values gradually compressed, not hard-clipped
        >>>
        >>> # Sharp soft clip (approaches hard)
        >>> sharp = soft_clip(x, lower=-3, upper=3, sharpness=8/6)
        >>>
        >>> # Verify monotonicity: ordering preserved
        >>> assert np.all(np.diff(soft) >= 0)
        >>> assert np.all(np.diff(sharp) >= 0)
    """
    # Validate bounds
    if lower >= upper:
        raise ValueError(
            f"lower must be < upper, got lower={lower}, upper={upper}"
        )

    # Handle edge case: bounds are very close (avoid division by zero)
    if np.abs(upper - lower) < 1e-10:
        # All values map to the midpoint
        return np.full_like(x, (lower + upper) / 2, dtype=float)

    # Compute center and half-range
    center = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0

    # Default sharpness: gentle (recommended for correlation analysis)
    if sharpness is None:
        sharpness = 2.0 / (upper - lower)

    # Apply soft clipping: c + (U-L)/2 * tanh(k * (x - c))
    # tanh is numerically stable for all finite inputs
    result = center + half_range * np.tanh(sharpness * (x - center))

    # Clamp to exact bounds (handles numerical precision edge cases)
    result = np.clip(result, lower, upper)

    return result


def soft_clip_per_feature(
    data: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    sharpness: Optional[float] = None
) -> np.ndarray:
    """
    Apply soft clipping with per-feature bounds.

    Vectorized implementation for efficiency. Applies soft_clip independently
    to each feature (row) using that feature's specific bounds.

    Args:
        data: Expression matrix (n_features × n_samples)
        lower_bounds: Per-feature lower bounds (n_features,)
        upper_bounds: Per-feature upper bounds (n_features,)
        sharpness: Global sharpness parameter. If None, computed per feature as
            2/(upper-lower). Can specify a single value to use for all features.

    Returns:
        Soft-clipped data matrix (same shape as input)

    Raises:
        ValueError: If shapes don't match or bounds are invalid

    Notes:
        - Fully vectorized using broadcasting (no Python loops)
        - Handles variable bounds per feature efficiently
        - If upper ≈ lower for a feature, returns midpoint for that feature
        - Preserves rank ordering within each feature independently

    Examples:
        >>> import numpy as np
        >>>
        >>> # 3 features × 5 samples
        >>> data = np.array([
        ...     [-5, -2, 0, 2, 5],
        ...     [0, 5, 10, 15, 20],
        ...     [-10, -5, 0, 5, 10]
        ... ])
        >>>
        >>> # Per-feature bounds (from MAD-based detection)
        >>> lower = np.array([-3, 2, -5])
        >>> upper = np.array([3, 18, 5])
        >>>
        >>> clipped = soft_clip_per_feature(data, lower, upper)
        >>>
        >>> # Verify bounds respected
        >>> assert np.all(clipped >= lower[:, None])
        >>> assert np.all(clipped <= upper[:, None])
    """
    # Validate shapes
    n_features, n_samples = data.shape

    if lower_bounds.shape != (n_features,):
        raise ValueError(
            f"lower_bounds must have shape ({n_features},), got {lower_bounds.shape}"
        )
    if upper_bounds.shape != (n_features,):
        raise ValueError(
            f"upper_bounds must have shape ({n_features},), got {upper_bounds.shape}"
        )

    # Validate bounds
    if np.any(lower_bounds >= upper_bounds):
        invalid = lower_bounds >= upper_bounds
        raise ValueError(
            f"Found {np.sum(invalid)} features with lower >= upper bounds. "
            f"First invalid: feature {np.where(invalid)[0][0]} has "
            f"lower={lower_bounds[invalid][0]}, upper={upper_bounds[invalid][0]}"
        )

    # Compute per-feature centers and half-ranges
    # Shape: (n_features, 1) for broadcasting
    centers = ((lower_bounds + upper_bounds) / 2.0)[:, None]
    half_ranges = ((upper_bounds - lower_bounds) / 2.0)[:, None]

    # Compute sharpness
    if sharpness is None:
        # Per-feature sharpness: 2 / (upper - lower)
        # Shape: (n_features, 1) for broadcasting
        k = (2.0 / (upper_bounds - lower_bounds))[:, None]
    else:
        # Global sharpness for all features
        k = sharpness

    # Apply vectorized soft clip: c + (U-L)/2 * tanh(k * (x - c))
    # Broadcasting handles per-feature parameters
    result = centers + half_ranges * np.tanh(k * (data - centers))

    # Clamp to exact bounds (handles numerical precision edge cases)
    # Broadcast bounds to matrix shape
    result = np.clip(result, lower_bounds[:, None], upper_bounds[:, None])

    return result


class Imputer(Transform):
    """
    Impute missing and outlier values with quality tracking.

    This transform replaces values flagged as OUTLIER_DETECTED or MISSING_ORIGINAL
    with statistically estimated values, then marks them as IMPUTED for provenance.

    Biological Rationale:
        Proteomics/transcriptomics analysis requires complete matrices for:
        - Correlation network construction (Pearson/Spearman need complete data)
        - PCA and clustering (most algorithms don't handle NaN)
        - Statistical testing (affects degrees of freedom)

        We can't remove outliers (creates missing values) or keep them (distorts
        statistics). Imputation is the standard solution, but we must track which
        values are synthetic vs measured.

    Strategies:
        mad-clip (RECOMMENDED):
            - Clip outliers to median ± threshold × MAD
            - For outlier at (gene_i, sample_j):
                1. Compute median and MAD from non-outlier values
                2. If value > median: replace with median + threshold × MAD
                3. If value < median: replace with median - threshold × MAD
            - Consistent with MAD-Z detection (same threshold for detection and clipping)
            - Preserves distribution shape better than percentile methods
            - Does NOT introduce correlation structure (orthogonal to downstream analysis)
            - Conservative: minimal data modification
            - Pure NumPy implementation (no external dependencies)
            - MAD uses 0.6745 scale factor for consistency with normal distribution

        soft-clip:
            - Apply smooth sigmoid clipping to outliers (tanh-based)
            - Uses same MAD-based bounds as mad-clip, but applies soft compression
            - Advantages over mad-clip:
                * No boundary spikes (continuous compression)
                * Preserves rank ordering (strictly monotonic)
                * Differentiable everywhere (C∞ continuous)
                * No tied values at boundaries
            - Disadvantages:
                * Can still distort correlations (though less than hard clipping)
                * More complex than hard clipping
                * Weighted correlation is preferred for correlation analysis
            - Sharpness parameter controls transition steepness
            - Default sharpness (2/(U-L)) provides gentle compression
            - Recommended when rank preservation is critical

        median:
            - Replace with per-gene median across all samples
            - Fast, deterministic, simple
            - Ignores gene relationships (all imputed values identical per gene)
            - Reduces variance (conservative but may hurt power)
            - No external dependencies (always available)

    Args:
        strategy: Imputation method ("mad-clip", "soft-clip", or "median")
        threshold: MAD-Z threshold for clipping (default 3.5, must match detection threshold)
            Applies to both mad-clip and soft-clip strategies.
        sharpness: Sharpness parameter for soft-clip (default: None = 2/(upper-lower))
            - None: Gentle compression (recommended for correlation analysis)
            - Higher values: Sharper transition (approaches hard clipping)
            - Only applies to soft-clip strategy
        group_cols: Metadata column(s) for group-stratified imputation (default: None).
            Applies to ALL strategies (mad-clip, soft-clip, median).

    Raises:
        ValueError: If strategy unknown or n_neighbors < 1

    Examples:
        >>> # Recommended: MAD-clip (consistent with detection threshold)
        >>> imputer = Imputer(strategy="mad-clip", threshold=3.5)
        >>> imputed = imputer.apply(flagged_matrix)
        >>>
        >>> # Fallback: Simple median imputation
        >>> imputer_median = Imputer(strategy="median")
        >>> imputed_median = imputer_median.apply(flagged_matrix)
        >>>
        >>> # Check what was imputed
        >>> imputed_mask = (imputed.quality_flags & QualityFlag.IMPUTED) > 0
        >>> print(f"Imputed {imputed_mask.sum()} values ({100*imputed_mask.sum()/imputed.data.size:.2f}%)")

    Scientific Notes:
        - Imputation >50% of values is unreliable (creates false precision)
        - Always perform sensitivity analysis (vary k, compare strategies)
        - Consider domain knowledge: low-abundance proteins may need different handling
        - Report imputation strategy and parameters in methods section
    """

    def __init__(
        self,
        strategy: str = "mad-clip",
        threshold: float = 3.5,
        sharpness: Optional[float] = None,
        group_cols: str | list[str] | None = None
    ):
        """
        Initialize imputer.

        Args:
            strategy: Imputation method ("mad-clip", "soft-clip", or "median")
            threshold: MAD-Z threshold for clipping (default 3.5, must match detection threshold)
                Applies to both mad-clip and soft-clip strategies.
            sharpness: Sharpness parameter for soft-clip (default: None = 2/(upper-lower))
                - None: Gentle compression (recommended for correlation analysis)
                - Higher values: Sharper transition (approaches hard clipping)
                - Only applies to soft-clip strategy; ignored for other strategies
            group_cols: Metadata column(s) for group-stratified imputation (default: None).
                Applies to ALL strategies (mad-clip, soft-clip, median).
                - None: Global median/MAD per gene across all samples
                - str: Single column (e.g., "phenotype")
                - list[str]: Multiple columns (e.g., ["phenotype", "Sex"])
                When specified, median and MAD are computed within each group separately.
                This preserves group-specific expression patterns and prevents cross-group
                contamination.

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate strategy
        valid_strategies = ["mad-clip", "soft-clip", "median"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. Must be one of {valid_strategies}.\n"
                f"Note: Winsorization has been replaced with MAD-clip for consistency with MAD-Z detection."
            )

        # Validate threshold
        if strategy in ["mad-clip", "soft-clip"]:
            if threshold <= 0:
                raise ValueError(
                    f"threshold must be positive, got {threshold}"
                )

        # Validate sharpness
        if sharpness is not None and sharpness <= 0:
            raise ValueError(
                f"sharpness must be positive if specified, got {sharpness}"
            )

        # Normalize group_cols to list or None
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        super().__init__(
            name="Imputer",
            params={
                "strategy": strategy,
                "threshold": threshold,
                "sharpness": sharpness,
                "group_cols": group_cols,
            }
        )

        self.strategy = strategy
        self.threshold = threshold
        self.sharpness = sharpness
        self.group_cols = group_cols

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Impute values flagged as outliers or missing.

        Updates both data AND quality flags. Imputed values are marked with
        QualityFlag.IMPUTED for provenance tracking.

        Args:
            matrix: Input BioMatrix (typically after outlier detection)

        Returns:
            New BioMatrix with imputed values and updated flags

        Raises:
            ValueError: If validation fails or no values need imputation
        """
        # Validate preconditions
        errors = self.validate(matrix)
        if errors:
            raise ValueError(
                f"Validation failed for {self.name}:\n" +
                "\n".join(f"  - {err}" for err in errors)
            )

        # Find values to impute (outliers OR originally missing)
        to_impute = (
            ((matrix.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0) |
            ((matrix.quality_flags & QualityFlag.MISSING_ORIGINAL) > 0)
        )

        if not np.any(to_impute):
            # Nothing to impute - return copy
            warnings.warn(
                "No values flagged for imputation (neither OUTLIER_DETECTED nor MISSING_ORIGINAL). "
                "Returning unchanged matrix.",
                UserWarning
            )
            return matrix.copy()

        # Warn if imputing too many values (>50% unreliable)
        impute_fraction = np.mean(to_impute)
        if impute_fraction > 0.5:
            warnings.warn(
                f"Imputing {100*impute_fraction:.1f}% of values (>50% is unreliable). "
                "Consider: "
                "(1) More conservative outlier detection (higher threshold), "
                "(2) Filtering low-quality features before imputation, "
                "(3) Using complete case analysis instead.",
                UserWarning
            )

        # Impute based on strategy
        if self.strategy == "mad-clip":
            new_data = self._impute_mad_clip(matrix, to_impute)
        elif self.strategy == "soft-clip":
            new_data = self._impute_soft_clip(matrix, to_impute)
        elif self.strategy == "median":
            new_data = self._impute_median(matrix, to_impute)
        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Update quality flags (mark imputed values, preserve other flags)
        new_flags = matrix.quality_flags.copy()
        new_flags[to_impute] = (new_flags[to_impute] | QualityFlag.IMPUTED).astype(np.uint32)

        return BioMatrix(
            data=new_data,
            feature_ids=matrix.feature_ids,
            sample_ids=matrix.sample_ids,
            sample_metadata=matrix.sample_metadata,
            quality_flags=new_flags,
        )

    def validate(self, matrix: BioMatrix) -> list[str]:
        """
        Validate matrix is suitable for imputation.

        Args:
            matrix: Matrix to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = super().validate(matrix)

        # Check that there are values to impute
        to_impute = (
            ((matrix.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0) |
            ((matrix.quality_flags & QualityFlag.MISSING_ORIGINAL) > 0)
        )

        # Don't add error if nothing to impute - we'll just warn and return copy
        # This makes it safe to always include imputation in pipelines

        # Check for features with all missing values (can't impute)
        if np.any(to_impute):
            all_missing_per_feature = np.all(to_impute, axis=1)
            n_all_missing = np.sum(all_missing_per_feature)
            if n_all_missing > 0:
                errors.append(
                    f"Found {n_all_missing} features with all values flagged for imputation. "
                    f"Cannot impute (no reference values). Filter these features first."
                )

        return errors

    def _impute_mad_clip(self, matrix: BioMatrix, mask: np.ndarray) -> np.ndarray:
        """
        Clip outliers to median ± threshold × MAD.

        This method clips outliers to bounds defined by MAD-Z statistics, ensuring
        consistency with the outlier detection threshold.

        For each gene independently:
        1. Compute median and MAD from non-outlier values
        2. Calculate bounds: [median - threshold × MAD, median + threshold × MAD]
        3. Clip flagged values to these bounds

        Group Stratification:
        When group_cols is specified, median and MAD are computed within each
        group separately. This preserves group-specific expression patterns:
        - CASE samples may have different expression ranges than CTRL
        - Using global statistics could bias one group toward the other's distribution
        - Group-stratified statistics maintain biological signal integrity

        Scientific Rationale:
        - Consistent with MAD-Z detection (same threshold for detection and clipping)
        - Preserves distribution shape better than percentile methods
        - No circularity with correlation analysis
        - Conservative (minimal data modification)
        - Respects group-specific biology

        MAD Calculation:
        - MAD = median(|x - median(x)|)
        - Scale factor 0.6745 makes MAD consistent with std dev for normal data
        - Modified Z-score = 0.6745 × |x - median| / MAD

        Args:
            matrix: BioMatrix to clip
            mask: Boolean mask of outlier positions (only these are modified)

        Returns:
            New data array with clipped values
        """
        data = matrix.data.astype(float).copy()

        if self.group_cols is None:
            # Global MAD-clip per gene (original behavior)
            self._mad_clip_global(data, mask)
        else:
            # Group-stratified MAD-clip
            self._mad_clip_stratified(data, mask, matrix)

        return data

    def _mad_clip_global(self, data: np.ndarray, mask: np.ndarray) -> None:
        """
        Apply global MAD-clip (median/MAD computed across all samples per gene).

        Modifies data in-place.
        """
        for gene_idx in range(data.shape[0]):
            gene_row = data[gene_idx, :]
            gene_mask = mask[gene_idx, :]

            if not np.any(gene_mask):
                continue

            # Compute median and MAD from non-outlier values
            clean_values = gene_row[~gene_mask]
            if len(clean_values) < 2:
                continue  # Not enough clean values

            median = np.median(clean_values)
            mad = np.median(np.abs(clean_values - median))

            # Avoid division by zero (if MAD is 0, all clean values are identical)
            if mad == 0:
                continue

            # Scale MAD to be consistent with standard deviation for normal data
            scaled_mad = mad / 0.6745

            # Compute bounds: median ± threshold × MAD
            lower_bound = median - self.threshold * scaled_mad
            upper_bound = median + self.threshold * scaled_mad

            # Clip flagged positions to bounds
            for sample_idx in np.where(gene_mask)[0]:
                val = gene_row[sample_idx]
                if val < lower_bound:
                    data[gene_idx, sample_idx] = lower_bound
                elif val > upper_bound:
                    data[gene_idx, sample_idx] = upper_bound

    def _mad_clip_stratified(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        matrix: BioMatrix,
        min_group_size: int = 10
    ) -> None:
        """
        Apply group-stratified MAD-clip (median/MAD computed within each group).

        For each group, computes median and MAD from only that group's
        non-outlier samples, then applies clipping. This prevents cross-group
        contamination of the reference distribution.

        Modifies data in-place.

        Args:
            data: Data array to modify
            mask: Boolean mask of outlier positions
            matrix: BioMatrix with sample_metadata
            min_group_size: Minimum samples in group to compute group-specific statistics.
                Smaller groups fall back to global statistics.
        """
        # Create composite group key from all group columns
        if len(self.group_cols) == 1:
            group_series = matrix.sample_metadata[self.group_cols[0]]
        else:
            group_series = matrix.sample_metadata[self.group_cols].apply(tuple, axis=1)

        unique_groups = group_series.unique()

        # Precompute global statistics as fallback for small groups
        global_medians = np.full(data.shape[0], np.nan)
        global_mads = np.full(data.shape[0], np.nan)

        for gene_idx in range(data.shape[0]):
            clean_values = data[gene_idx, ~mask[gene_idx, :]]
            if len(clean_values) >= 2:
                global_medians[gene_idx] = np.median(clean_values)
                mad = np.median(np.abs(clean_values - global_medians[gene_idx]))
                if mad > 0:
                    global_mads[gene_idx] = mad / 0.6745

        # Process each group
        for group in unique_groups:
            group_mask = (group_series == group).values
            n_samples_in_group = group_mask.sum()

            for gene_idx in range(data.shape[0]):
                # Get outlier mask for this gene within this group
                gene_outliers_in_group = mask[gene_idx, :] & group_mask
                if not np.any(gene_outliers_in_group):
                    continue

                # Compute statistics from clean values within this group
                group_clean_mask = group_mask & ~mask[gene_idx, :]
                clean_values_in_group = data[gene_idx, group_clean_mask]

                # Use group-specific statistics if sufficient samples, else fall back to global
                if len(clean_values_in_group) >= min_group_size:
                    median = np.median(clean_values_in_group)
                    mad = np.median(np.abs(clean_values_in_group - median))
                    if mad == 0:
                        continue  # Skip if all clean values identical
                    scaled_mad = mad / 0.6745
                elif not np.isnan(global_medians[gene_idx]) and not np.isnan(global_mads[gene_idx]):
                    # Fall back to global statistics for small groups
                    median = global_medians[gene_idx]
                    scaled_mad = global_mads[gene_idx]
                else:
                    # No valid statistics available, skip this gene
                    continue

                # Compute bounds: median ± threshold × MAD
                lower_bound = median - self.threshold * scaled_mad
                upper_bound = median + self.threshold * scaled_mad

                # Apply clipping to outliers in this group
                for sample_idx in np.where(gene_outliers_in_group)[0]:
                    val = data[gene_idx, sample_idx]
                    if val < lower_bound:
                        data[gene_idx, sample_idx] = lower_bound
                    elif val > upper_bound:
                        data[gene_idx, sample_idx] = upper_bound

    def _impute_soft_clip(self, matrix: BioMatrix, mask: np.ndarray) -> np.ndarray:
        """
        Apply soft sigmoid clipping to outliers using MAD-based bounds.

        This method uses the same MAD-Z bounds as mad-clip but applies smooth
        compression instead of hard clipping. The soft clipping function is:

            soft_clip(x) = c + (U-L)/2 * tanh(k * (x - c))

        where c = (L+U)/2, k is sharpness parameter.

        Advantages over hard clipping:
            - No boundary spikes (continuous compression)
            - Preserves rank ordering (strictly monotonic)
            - Differentiable everywhere (C∞ continuous)
            - No tied values at boundaries

        Group Stratification:
            When group_cols is specified, bounds are computed per-group
            (same as mad-clip), but soft compression is applied instead
            of hard clipping.

        Args:
            matrix: BioMatrix to clip
            mask: Boolean mask of outlier positions (only these are modified)

        Returns:
            New data array with soft-clipped values
        """
        data = matrix.data.astype(float).copy()

        if self.group_cols is None:
            # Global soft-clip per gene
            self._soft_clip_global(data, mask)
        else:
            # Group-stratified soft-clip
            self._soft_clip_stratified(data, mask, matrix)

        return data

    def _soft_clip_global(self, data: np.ndarray, mask: np.ndarray) -> None:
        """
        Apply global soft-clip (bounds computed across all samples per gene).

        Modifies data in-place. Uses vectorized operations for efficiency.
        """
        # Compute per-gene bounds from non-outlier values
        n_features = data.shape[0]
        lower_bounds = np.full(n_features, np.nan)
        upper_bounds = np.full(n_features, np.nan)

        for gene_idx in range(n_features):
            gene_row = data[gene_idx, :]
            gene_mask = mask[gene_idx, :]

            if not np.any(gene_mask):
                continue  # No outliers for this gene

            # Compute median and MAD from non-outlier values
            clean_values = gene_row[~gene_mask]
            if len(clean_values) < 2:
                continue  # Not enough clean values

            median = np.median(clean_values)
            mad = np.median(np.abs(clean_values - median))

            # Avoid division by zero
            if mad == 0:
                continue

            # Scale MAD to be consistent with std for normal data
            scaled_mad = mad / 0.6745

            # Compute bounds: median ± threshold × MAD
            lower_bounds[gene_idx] = median - self.threshold * scaled_mad
            upper_bounds[gene_idx] = median + self.threshold * scaled_mad

        # Apply soft clipping to rows with valid bounds
        valid_features = ~np.isnan(lower_bounds) & ~np.isnan(upper_bounds)
        if not np.any(valid_features):
            return  # No valid bounds computed

        # Apply vectorized soft clip to features with outliers
        for gene_idx in np.where(valid_features & np.any(mask, axis=1))[0]:
            data[gene_idx, :] = soft_clip(
                data[gene_idx, :],
                lower=lower_bounds[gene_idx],
                upper=upper_bounds[gene_idx],
                sharpness=self.sharpness
            )

    def _soft_clip_stratified(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        matrix: BioMatrix,
        min_group_size: int = 10
    ) -> None:
        """
        Apply group-stratified soft-clip (bounds computed within each group).

        For each group, computes median and MAD from only that group's
        non-outlier samples, then applies soft clipping. This prevents
        cross-group contamination of the reference distribution.

        Modifies data in-place.

        Args:
            data: Data array to modify
            mask: Boolean mask of outlier positions
            matrix: BioMatrix with sample_metadata
            min_group_size: Minimum samples in group to compute group-specific statistics.
                Smaller groups fall back to global statistics.
        """
        # Create composite group key from all group columns
        if len(self.group_cols) == 1:
            group_series = matrix.sample_metadata[self.group_cols[0]]
        else:
            group_series = matrix.sample_metadata[self.group_cols].apply(tuple, axis=1)

        unique_groups = group_series.unique()

        # Precompute global bounds as fallback for small groups
        n_features = data.shape[0]
        global_lower = np.full(n_features, np.nan)
        global_upper = np.full(n_features, np.nan)

        for gene_idx in range(n_features):
            clean_values = data[gene_idx, ~mask[gene_idx, :]]
            if len(clean_values) >= 2:
                median = np.median(clean_values)
                mad = np.median(np.abs(clean_values - median))
                if mad > 0:
                    scaled_mad = mad / 0.6745
                    global_lower[gene_idx] = median - self.threshold * scaled_mad
                    global_upper[gene_idx] = median + self.threshold * scaled_mad

        # Process each group
        for group in unique_groups:
            group_mask = (group_series == group).values

            for gene_idx in range(n_features):
                # Get outlier mask for this gene within this group
                gene_outliers_in_group = mask[gene_idx, :] & group_mask
                if not np.any(gene_outliers_in_group):
                    continue

                # Compute statistics from clean values within this group
                group_clean_mask = group_mask & ~mask[gene_idx, :]
                clean_values_in_group = data[gene_idx, group_clean_mask]

                # Use group-specific bounds if sufficient samples, else fall back
                if len(clean_values_in_group) >= min_group_size:
                    median = np.median(clean_values_in_group)
                    mad = np.median(np.abs(clean_values_in_group - median))
                    if mad == 0:
                        continue  # Skip if all clean values identical
                    scaled_mad = mad / 0.6745
                    lower_bound = median - self.threshold * scaled_mad
                    upper_bound = median + self.threshold * scaled_mad
                elif not np.isnan(global_lower[gene_idx]) and not np.isnan(global_upper[gene_idx]):
                    # Fall back to global bounds for small groups
                    lower_bound = global_lower[gene_idx]
                    upper_bound = global_upper[gene_idx]
                else:
                    # No valid bounds available, skip
                    continue

                # Apply soft clipping to this gene's values in this group
                group_indices = np.where(group_mask)[0]
                data[gene_idx, group_indices] = soft_clip(
                    data[gene_idx, group_indices],
                    lower=lower_bound,
                    upper=upper_bound,
                    sharpness=self.sharpness
                )

    def _impute_median(
        self,
        matrix: BioMatrix,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Replace flagged values with per-gene median, optionally stratified by groups.

        Simple imputation: For each gene, compute median across samples in the same group
        (excluding flagged values), then replace flagged values with that median.

        When group_cols is specified:
        - Groups are formed by unique combinations of group column values
        - Median is computed per-gene within each group
        - Flagged values are replaced with their group's median
        - This preserves group-specific expression patterns

        Advantages:
        - Fast, deterministic, simple
        - No dependencies (pure NumPy)
        - Robust to outliers in reference values
        - Group-stratified version preserves biological signal

        Disadvantages:
        - Ignores gene relationships (context-blind)
        - Reduces variance (all imputed values in a group are identical per gene)
        - May hurt power in downstream statistical tests

        Args:
            matrix: BioMatrix to impute
            mask: Boolean mask of values to impute

        Returns:
            New data array with imputed values

        Notes:
            - Uses np.nanmedian to handle any existing NaNs
            - Computes median excluding flagged values (cleaner estimate)
            - If all values in a gene/group are flagged, uses global median (rare edge case)
        """
        data = matrix.data.astype(float).copy()

        if self.group_cols is None:
            # Global median per gene (original behavior)
            data_for_median = data.copy()
            data_for_median[mask] = np.nan

            medians = np.nanmedian(data_for_median, axis=1, keepdims=True)
            global_median = np.nanmedian(data)
            medians = np.where(np.isnan(medians), global_median, medians)

            data[mask] = np.broadcast_to(medians, data.shape)[mask]
        else:
            # Group-stratified median imputation
            # Create composite group key from all group columns
            if len(self.group_cols) == 1:
                group_series = matrix.sample_metadata[self.group_cols[0]]
            else:
                group_series = matrix.sample_metadata[self.group_cols].apply(tuple, axis=1)

            unique_groups = group_series.unique()
            global_median = np.nanmedian(data)

            for group in unique_groups:
                group_mask = (group_series == group).values
                n_samples_in_group = group_mask.sum()

                if n_samples_in_group < 1:
                    continue

                # Get sample indices for this group
                group_indices = np.where(group_mask)[0]

                # Extract group data
                group_data = data[:, group_mask].copy()
                group_impute_mask = mask[:, group_mask]

                # Set flagged values to NaN for median computation
                group_data_for_median = group_data.copy()
                group_data_for_median[group_impute_mask] = np.nan

                # Compute per-gene median within this group
                group_medians = np.nanmedian(group_data_for_median, axis=1, keepdims=True)

                # Fallback to global median if entire gene is flagged in this group
                group_medians = np.where(np.isnan(group_medians), global_median, group_medians)

                # Replace flagged values in this group
                # Must iterate to avoid the copy-on-slice issue with advanced indexing
                for local_col, global_col in enumerate(group_indices):
                    col_mask = group_impute_mask[:, local_col]
                    if col_mask.any():
                        data[col_mask, global_col] = group_medians[col_mask, 0]

        return data