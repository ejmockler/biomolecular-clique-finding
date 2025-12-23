"""
Semi-supervised biomarker discovery for binary phenotype classification.

This module provides data-driven discovery of optimal discriminating features
for any binary phenotype when a small amount of labeled data is available.

Key Insight:
    With minimal supervision (~5% labeled samples), we can identify the best
    feature for classification without prior biological knowledge. The method:

    1. Scores features by separation on labeled samples (Cohen's d)
    2. Validates bimodal structure across ALL samples (not just labeled)
    3. Measures consistency between bimodal split and known labels
    4. Ranks by combined score to find the optimal discriminator

    This works because the "true" signal features will:
    - Separate known labels well
    - Have consistent bimodal structure in unlabeled data
    - Extend cleanly to the full dataset

Applications:
    - Sex imputation: Discover Y-linked markers without curated gene lists
    - Disease subtyping: Find discriminating biomarkers from partial labels
    - Batch effect detection: Identify features that separate batches

Usage:
    >>> from cliquefinder.quality.marker_discovery import MarkerDiscovery
    >>>
    >>> # Create partial labels (-1 = unlabeled)
    >>> labels = np.full(n_samples, -1)
    >>> labels[known_indices] = known_labels  # 0 or 1
    >>>
    >>> discovery = MarkerDiscovery()
    >>> result = discovery.discover(matrix, labels)
    >>>
    >>> print(f"Best marker: {result.best_feature}")
    >>> print(f"Effect size: {result.effect_size:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.utils.statistics import otsu_threshold, cohens_d


__all__ = [
    'MarkerDiscovery',
    'MarkerDiscoveryResult',
    'FeatureScore',
]


@dataclass
class FeatureScore:
    """Score for a single feature as a potential marker."""
    feature_idx: int
    feature_name: str

    # Separation metrics
    d_labeled: float          # Cohen's d on labeled samples
    d_full: float             # Cohen's d on full bimodal split

    # Quality metrics
    balance: float            # Class balance ratio [0, 1]
    consistency: float        # Agreement between bimodal split and labels

    # Derived
    combined_score: float     # Composite ranking score

    # For prediction
    threshold: float          # Otsu threshold for bimodal split
    bimodal_labels: np.ndarray = field(repr=False)  # Full sample predictions


@dataclass
class MarkerDiscoveryResult:
    """
    Results from semi-supervised marker discovery.

    Attributes:
        best_feature: Name of discovered optimal feature
        best_feature_idx: Index of best feature in matrix
        effect_size: Cohen's d separation achieved
        threshold: Optimal classification threshold
        predictions: Predicted binary labels for all samples
        confidence: Confidence scores for predictions
        feature_scores: Ranked list of all scored features
        n_labeled: Number of labeled samples used
        label_fraction: Fraction of samples that were labeled
    """
    best_feature: str
    best_feature_idx: int
    effect_size: float
    threshold: float
    predictions: np.ndarray
    confidence: np.ndarray
    feature_scores: list[FeatureScore]
    n_labeled: int
    label_fraction: float

    @property
    def n_samples(self) -> int:
        return len(self.predictions)

    @property
    def top_features(self) -> list[str]:
        """Top 10 features by combined score."""
        return [f.feature_name for f in self.feature_scores[:10]]

    def get_predictions_aligned(self, reference_labels: np.ndarray) -> np.ndarray:
        """
        Return predictions aligned with reference labels.

        Handles potential label flipping (0/1 assignment is arbitrary
        in unsupervised bimodal detection).
        """
        ref_valid = reference_labels >= 0
        if ref_valid.sum() == 0:
            return self.predictions

        # Check alignment
        agree = np.mean(self.predictions[ref_valid] == reference_labels[ref_valid])
        if agree < 0.5:
            return 1 - self.predictions
        return self.predictions


class MarkerDiscovery:
    """
    Semi-supervised discovery of optimal discriminating features.

    Given partial labels and a feature matrix, discovers the single best
    feature for binary classification without requiring prior knowledge
    of which features are biologically relevant.

    Args:
        min_labeled_per_class: Minimum labeled samples per class (default: 5)
        min_effect_size: Minimum Cohen's d to consider a feature (default: 0.5)
        random_state: Random seed for reproducibility

    Example:
        >>> # Discover sex marker from 5% labeled data
        >>> labels = np.full(436, -1)  # -1 = unlabeled
        >>> labels[known_male_idx] = 1
        >>> labels[known_female_idx] = 0
        >>>
        >>> discovery = MarkerDiscovery()
        >>> result = discovery.discover(matrix, labels)
        >>>
        >>> # Result contains the best feature (e.g., DDX3Y for sex)
        >>> print(f"Discovered: {result.best_feature}")
    """

    def __init__(
        self,
        min_labeled_per_class: int = 5,
        min_effect_size: float = 0.5,
        random_state: int = 42,
    ):
        self.min_labeled_per_class = min_labeled_per_class
        self.min_effect_size = min_effect_size
        self.random_state = random_state

    def _otsu_threshold(self, values: np.ndarray) -> float:
        """Compute optimal bimodal threshold using Otsu's method."""
        return otsu_threshold(values)

    def _cohens_d(
        self,
        values: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute Cohen's d effect size between two groups."""
        return cohens_d(values, labels)

    def _score_feature(
        self,
        feature_data: np.ndarray,
        labels: np.ndarray,
        labeled_mask: np.ndarray,
    ) -> Optional[FeatureScore]:
        """
        Score a single feature for discriminative power.

        Combines:
        1. Separation on labeled samples (supervised signal)
        2. Bimodal structure on all samples (unsupervised signal)
        3. Consistency between bimodal split and labels
        """
        if np.std(feature_data) < 1e-10:
            return None

        # Labeled sample analysis
        x_labeled = feature_data[labeled_mask]
        y_labeled = labels[labeled_mask]

        d_labeled = self._cohens_d(x_labeled, y_labeled)

        if d_labeled < self.min_effect_size:
            return None

        # Full data bimodal analysis
        threshold = self._otsu_threshold(feature_data)
        bimodal_labels = (feature_data > threshold).astype(int)

        # Balance ratio
        n0, n1 = (bimodal_labels == 0).sum(), (bimodal_labels == 1).sum()
        balance = min(n0, n1) / max(n0, n1) if max(n0, n1) > 0 else 0

        # Cohen's d on full bimodal split
        d_full = self._cohens_d(feature_data, bimodal_labels)

        # Consistency: does bimodal split agree with labeled samples?
        bimodal_on_labeled = bimodal_labels[labeled_mask]
        consistency = max(
            np.mean(bimodal_on_labeled == y_labeled),
            np.mean(bimodal_on_labeled != y_labeled)  # Handle label flip
        )

        # Combined score prioritizes:
        # 1. High separation on labeled data (most important)
        # 2. High consistency with labels
        # 3. Good balance (relevant for ~50/50 phenotypes like sex)
        combined_score = d_labeled * consistency * (0.5 + 0.5 * balance)

        return FeatureScore(
            feature_idx=-1,  # Set by caller
            feature_name="",  # Set by caller
            d_labeled=d_labeled,
            d_full=d_full,
            balance=balance,
            consistency=consistency,
            combined_score=combined_score,
            threshold=threshold,
            bimodal_labels=bimodal_labels,
        )

    def discover(
        self,
        matrix: BioMatrix,
        labels: np.ndarray,
    ) -> MarkerDiscoveryResult:
        """
        Discover optimal discriminating feature from partial labels.

        Args:
            matrix: BioMatrix with features x samples
            labels: Array of shape (n_samples,) with:
                    - 0 or 1 for labeled samples
                    - -1 for unlabeled samples

        Returns:
            MarkerDiscoveryResult with best feature and predictions

        Raises:
            ValueError: If insufficient labeled samples
        """
        labels = np.asarray(labels)

        if len(labels) != matrix.n_samples:
            raise ValueError(
                f"Labels length ({len(labels)}) != n_samples ({matrix.n_samples})"
            )

        # Validate labeled samples
        labeled_mask = labels >= 0
        n_labeled = labeled_mask.sum()

        n_class_0 = (labels == 0).sum()
        n_class_1 = (labels == 1).sum()

        if n_class_0 < self.min_labeled_per_class:
            raise ValueError(
                f"Class 0 has only {n_class_0} labeled samples "
                f"(minimum: {self.min_labeled_per_class})"
            )

        if n_class_1 < self.min_labeled_per_class:
            raise ValueError(
                f"Class 1 has only {n_class_1} labeled samples "
                f"(minimum: {self.min_labeled_per_class})"
            )

        # Score all features
        feature_scores: list[FeatureScore] = []

        for i in range(matrix.n_features):
            feature_data = matrix.data[i, :]

            score = self._score_feature(feature_data, labels, labeled_mask)

            if score is not None:
                score.feature_idx = i
                score.feature_name = str(matrix.feature_ids[i])
                feature_scores.append(score)

        if not feature_scores:
            raise ValueError(
                f"No features found with effect size >= {self.min_effect_size}. "
                "Try lowering min_effect_size or adding more labeled samples."
            )

        # Rank by combined score
        feature_scores.sort(key=lambda x: -x.combined_score)

        # Best feature
        best = feature_scores[0]

        # Align predictions with labels
        bimodal_on_labeled = best.bimodal_labels[labeled_mask]
        y_labeled = labels[labeled_mask]

        if np.mean(bimodal_on_labeled == y_labeled) < 0.5:
            # Flip labels
            predictions = 1 - best.bimodal_labels
        else:
            predictions = best.bimodal_labels.copy()

        # Confidence based on distance from threshold
        feature_data = matrix.data[best.feature_idx, :]
        distances = np.abs(feature_data - best.threshold)

        # Normalize by IQR for robustness
        q75, q25 = np.percentile(feature_data, [75, 25])
        scale = (q75 - q25) / 2 if (q75 - q25) > 0 else np.std(feature_data)
        scale = max(scale, 1e-10)

        # Sigmoid-based confidence
        confidence = 1 / (1 + np.exp(-distances / scale))
        confidence = np.clip(confidence, 0.0, 1.0)

        return MarkerDiscoveryResult(
            best_feature=best.feature_name,
            best_feature_idx=best.feature_idx,
            effect_size=best.d_labeled,
            threshold=best.threshold,
            predictions=predictions,
            confidence=confidence,
            feature_scores=feature_scores,
            n_labeled=n_labeled,
            label_fraction=n_labeled / matrix.n_samples,
        )

    def discover_from_reference(
        self,
        matrix: BioMatrix,
        reference_feature: str,
        label_fraction: float = 0.10,
    ) -> MarkerDiscoveryResult:
        """
        Discover best marker using a reference feature for pseudo-labels.

        This is useful for validation: use a known marker to generate
        partial labels, then verify discovery finds it (or a correlated marker).

        Args:
            matrix: BioMatrix with features x samples
            reference_feature: Feature name to use for generating pseudo-labels
            label_fraction: Fraction of samples to label (default: 0.10)

        Returns:
            MarkerDiscoveryResult from semi-supervised discovery
        """
        np.random.seed(self.random_state)

        # Find reference feature
        feature_to_idx = {fid: i for i, fid in enumerate(matrix.feature_ids)}

        if reference_feature not in feature_to_idx:
            raise ValueError(f"Reference feature '{reference_feature}' not found")

        ref_idx = feature_to_idx[reference_feature]
        ref_data = matrix.data[ref_idx, :]

        # Generate pseudo-labels using bimodal threshold
        threshold = self._otsu_threshold(ref_data)
        full_labels = (ref_data > threshold).astype(int)

        # Sample stratified subset to label
        n_samples = matrix.n_samples
        n_labeled = int(n_samples * label_fraction)

        idx_0 = np.where(full_labels == 0)[0]
        idx_1 = np.where(full_labels == 1)[0]

        n_0 = int(n_labeled * len(idx_0) / n_samples)
        n_1 = n_labeled - n_0

        labeled_idx = np.concatenate([
            np.random.choice(idx_0, min(n_0, len(idx_0)), replace=False),
            np.random.choice(idx_1, min(n_1, len(idx_1)), replace=False),
        ])

        # Create partial labels
        labels = np.full(n_samples, -1)
        labels[labeled_idx] = full_labels[labeled_idx]

        return self.discover(matrix, labels)
