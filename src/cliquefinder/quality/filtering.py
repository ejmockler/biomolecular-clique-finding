"""
Quality filtering transformations for expression matrices.

Provides robust filtering based on expression levels, prevalence, and other quality metrics.
Implements the Transform interface for composable pipelines.

Engineering Design:
    - Pure functions (Transform): input matrix -> output matrix
    - Stratified filtering: Ensures rare groups (e.g., specific phenotypes) are not filtered out
    - robust logic: Handles CPM normalization internally for thresholds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.transform import Transform

logger = logging.getLogger(__name__)

__all__ = ['StratifiedExpressionFilter', 'ExpressionFilterResult']


@dataclass
class ExpressionFilterResult:
    """Results from expression filtering with full provenance."""
    passed_genes: Set[str]
    failed_genes: Set[str]
    stratum_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # stratum_stats format: {"CASE_Male": {"passed": 15000, "failed": 5000, "n_samples": 240}}
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return len(self.passed_genes)

    @property
    def n_failed(self) -> int:
        return len(self.failed_genes)

    @property
    def pass_rate(self) -> float:
        total = self.n_passed + self.n_failed
        return self.n_passed / total if total > 0 else 0.0


class StratifiedExpressionFilter(Transform):
    """
    Filter features based on expression levels within stratified groups.

    Retains features that are "present" in at least one biological subgroup.
    Presence is defined as having expression > min_cpm in at least min_prevalence
    fraction of samples within a group.

    This prevents filtering out genes that are specific to a condition (e.g., disease-only)
    or a demographic (e.g., sex-specific), which might be lost in a global filter.

    Params:
        min_cpm: Minimum Counts Per Million (CPM) to consider a gene "expressed" in a sample.
        min_prevalence: Fraction of samples in a group required to express the gene.
        stratify_by: List of metadata columns to define groups (e.g., ['phenotype', 'sex']).
        min_group_size: Minimum number of samples in a group to consider it valid.
                        Small groups (< min_group_size) are skipped to avoid noise.

    Examples:
        >>> filter_transform = StratifiedExpressionFilter(
        ...     min_cpm=1.0,
        ...     min_prevalence=0.1,
        ...     stratify_by=['phenotype', 'Sex']
        ... )
        >>> filtered_matrix = filter_transform.apply(matrix)
    """

    def __init__(
        self,
        min_cpm: float = 1.0,
        min_prevalence: float = 0.1,
        stratify_by: Optional[List[str]] = None,
        min_group_size: int = 10
    ):
        super().__init__(
            name="StratifiedExpressionFilter",
            params={
                "min_cpm": min_cpm,
                "min_prevalence": min_prevalence,
                "stratify_by": stratify_by,
                "min_group_size": min_group_size
            }
        )
        self.min_cpm = min_cpm
        self.min_prevalence = min_prevalence
        self.stratify_by = stratify_by or []
        self.min_group_size = min_group_size

    def _compute_keep_mask(self, matrix: BioMatrix) -> tuple[np.ndarray, Dict[str, Dict[str, int]]]:
        """
        Core filtering logic: compute which features to keep.

        Returns:
            keep_mask: Boolean array indicating which features pass filter
            stratum_stats: Dictionary with per-group statistics
        """
        # 1. Calculate CPM (Counts Per Million)
        raw_data = matrix.data
        library_sizes = raw_data.sum(axis=0)

        # Avoid division by zero
        library_sizes[library_sizes == 0] = 1.0

        # Calculate CPM matrix (temporary, just for filtering)
        cpm = (raw_data / library_sizes[None, :]) * 1e6

        # 2. Define Groups
        if self.stratify_by:
            # Check if columns exist
            missing_cols = [c for c in self.stratify_by if c not in matrix.sample_metadata.columns]
            if missing_cols:
                raise ValueError(f"Stratification columns not found in metadata: {missing_cols}")

            # Create composite group labels
            groups = matrix.sample_metadata[self.stratify_by].astype(str).agg('_'.join, axis=1)
        else:
            # Single global group
            groups = pd.Series("Global", index=matrix.sample_ids)

        unique_groups = groups.unique()
        logger.info(f"Identified {len(unique_groups)} groups for stratification: {unique_groups}")

        # 3. Determine Keep Mask
        # A gene is kept if it passes filters in AT LEAST ONE valid group
        n_features = matrix.n_features
        keep_mask = np.zeros(n_features, dtype=bool)
        stratum_stats: Dict[str, Dict[str, int]] = {}

        for group in unique_groups:
            # Identify samples in this group
            group_samples_mask = (groups == group).values
            n_samples = group_samples_mask.sum()

            # Skip small groups
            if n_samples < self.min_group_size:
                logger.info(f"Skipping small group '{group}' (n={n_samples})")
                continue

            # Calculate threshold count
            thresh_samples = max(1, int(n_samples * self.min_prevalence))

            # Subset CPM for group
            group_cpm = cpm[:, group_samples_mask]

            # Count samples with CPM > min_cpm
            expressed_in_samples = (group_cpm > self.min_cpm).sum(axis=1)

            # Features passing in this group
            group_pass_mask = expressed_in_samples >= thresh_samples

            # Update global keep mask
            keep_mask |= group_pass_mask

            # Store statistics for this stratum
            n_passed_in_group = group_pass_mask.sum()
            n_failed_in_group = n_features - n_passed_in_group
            stratum_stats[group] = {
                "passed": int(n_passed_in_group),
                "failed": int(n_failed_in_group),
                "n_samples": int(n_samples)
            }

            logger.info(f"  Group '{group}' (n={n_samples}): {n_passed_in_group} features passed")

        return keep_mask, stratum_stats

    def apply(self, matrix: BioMatrix) -> BioMatrix:
        """
        Apply stratified expression filter.
        """
        logger.info(f"Applying StratifiedExpressionFilter: min_cpm={self.min_cpm}, "
                    f"min_prevalence={self.min_prevalence}, stratify_by={self.stratify_by}")

        keep_mask, _ = self._compute_keep_mask(matrix)

        # Log results
        n_kept = keep_mask.sum()
        n_removed = matrix.n_features - n_kept

        logger.info(f"Filtering complete: Kept {n_kept}/{matrix.n_features} features "
                    f"({100*n_kept/matrix.n_features:.1f}%), Removed {n_removed}")

        return matrix.select_features(keep_mask)

    def get_passing_genes(self, matrix: BioMatrix) -> ExpressionFilterResult:
        """
        Get genes passing expression filter without transforming matrix.

        Useful for cross-modal filtering where we need the gene set
        but don't need to actually subset the matrix.

        Args:
            matrix: BioMatrix with expression data and sample_metadata

        Returns:
            ExpressionFilterResult with passed/failed genes and statistics
        """
        logger.info(f"Computing passing genes: min_cpm={self.min_cpm}, "
                    f"min_prevalence={self.min_prevalence}, stratify_by={self.stratify_by}")

        # Reuse the filtering logic
        keep_mask, stratum_stats = self._compute_keep_mask(matrix)

        # Convert mask to gene ID sets
        feature_ids = matrix.feature_ids
        passed_genes = set(feature_ids[keep_mask])
        failed_genes = set(feature_ids[~keep_mask])

        # Package parameters
        parameters = {
            "min_cpm": self.min_cpm,
            "min_prevalence": self.min_prevalence,
            "stratify_by": self.stratify_by,
            "min_group_size": self.min_group_size
        }

        result = ExpressionFilterResult(
            passed_genes=passed_genes,
            failed_genes=failed_genes,
            stratum_stats=stratum_stats,
            parameters=parameters
        )

        logger.info(f"Gene filtering complete: {result.n_passed}/{result.n_passed + result.n_failed} "
                    f"genes passed ({result.pass_rate*100:.1f}%)")

        return result

    def validate(self, matrix: BioMatrix) -> list[str]:
        errors = super().validate(matrix)
        if np.any(matrix.data < 0):
            errors.append("Matrix contains negative values (expected counts for CPM calculation)")
        return errors
