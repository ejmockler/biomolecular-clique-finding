"""
Quality control module for proteomics/transcriptomics data.

This module provides tools for detecting and handling data quality issues in
high-dimensional biological datasets, particularly proteomics and transcriptomics.

Components:
    OutlierDetector: Robust outlier detection using MAD-Z or IQR methods
    Imputer: Context-aware value imputation using KNN or median strategies
    MarkerDiscovery: Semi-supervised discovery of discriminating biomarkers
    SexClassifier: Data-driven biological sex classification (SupervisedSexClassifier,
                   SemiSupervisedSexClassifier) - no hardcoded biomarkers

Biological Context:
    Omics data quality issues arise from:
    - Technical artifacts (sample prep, instrument noise)
    - Batch effects (different processing times/labs)
    - True biological extremes (disease states, stress responses)

    Quality control workflow:
    1. Detect outliers (OutlierDetector) - marks flags, doesn't modify data
    2. Review flagged values (visualization, domain expertise)
    3. Impute if appropriate (Imputer) - replaces values, tracks provenance
    4. Export with quality flags - full transparency for reviewers

Quality Flag System:
    All transformations update quality_flags to track which values were:
    - OUTLIER_DETECTED: Flagged by statistical methods
    - IMPUTED: Replaced with estimated values
    - MISSING_ORIGINAL: Originally missing in raw data

    This enables sensitivity analysis and reviewer transparency.

Semi-Supervised Biomarker Discovery:
    The MarkerDiscovery class enables data-driven identification of optimal
    discriminating features for binary phenotypes with minimal supervision:

    >>> from cliquefinder.quality import MarkerDiscovery
    >>>
    >>> # With ~5% labeled samples, discover best feature
    >>> labels = np.full(n_samples, -1)  # -1 = unlabeled
    >>> labels[known_idx] = known_labels  # 0 or 1
    >>>
    >>> discovery = MarkerDiscovery()
    >>> result = discovery.discover(matrix, labels)
    >>> print(f"Best marker: {result.best_feature}")  # e.g., "DDX3Y"

Examples:
    >>> from cliquefinder.quality import OutlierDetector, Imputer
    >>> from cliquefinder.core.quality import QualityFlag
    >>>
    >>> # Standard quality control pipeline (within-group detection by default)
    >>> detector = OutlierDetector()  # mode="within_group", method="mad-z"
    >>> flagged = detector.apply(matrix)
    >>>
    >>> # Review outliers before imputing
    >>> outlier_mask = (flagged.quality_flags & QualityFlag.OUTLIER_DETECTED) > 0
    >>> print(f"Detected {outlier_mask.sum()} outliers ({100*outlier_mask.sum()/matrix.data.size:.2f}%)")
    >>>
    >>> # Impute with KNN
    >>> imputer = Imputer(strategy="knn", n_neighbors=5)
    >>> clean = imputer.apply(flagged)
    >>>
    >>> # Verify tracking
    >>> imputed_mask = (clean.quality_flags & QualityFlag.IMPUTED) > 0
    >>> assert imputed_mask.sum() == outlier_mask.sum()

Scientific Best Practices:
    1. Always visualize flagged values before imputation (could be real biology!)
    2. Report imputation parameters in methods section
    3. Perform sensitivity analysis (vary thresholds, compare strategies)
    4. Export quality flags with final data for reviewer transparency
    5. Consider domain knowledge: low-abundance proteins may need special handling

Performance Notes:
    - OutlierDetector: O(n log n) for MAD-Z/IQR (median/percentile computation)
    - Imputer KNN: O(n^2 * k) but optimized in sklearn
    - Imputer median: O(n log n) per feature
    - MarkerDiscovery: O(n_features * n_samples) for full feature scan
    - Fast enough for 60k genes x 500 samples (< 1 minute total)

References:
    - Leys et al. (2013) "Detecting outliers: Do not use standard deviation around
      the mean, use absolute deviation around the median"
    - Troyanskaya et al. (2001) "Missing value estimation methods for DNA microarrays"
    - Lazar et al. (2016) "Accounting for the Multiple Natures of Missing Values in
      Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies"
"""

from cliquefinder.quality.outliers import (
    OutlierDetector,
    ResidualOutlierDetector,
    AdaptiveOutlierDetector,
    MultiPassOutlierDetector,
)
from cliquefinder.quality.imputation import Imputer
from cliquefinder.quality.filtering import (
    StratifiedExpressionFilter,
    ExpressionFilterResult,
)
from cliquefinder.quality.marker_discovery import (
    MarkerDiscovery,
    MarkerDiscoveryResult,
    FeatureScore,
)
from cliquefinder.quality.sex_imputation import (
    Sex,
    SupervisedSexClassifier,
    SupervisedClassifierResult,
    SemiSupervisedSexClassifier,
    SemiSupervisedResult,
)

__all__ = [
    # Outlier detection
    'OutlierDetector',
    'ResidualOutlierDetector',
    'AdaptiveOutlierDetector',
    'MultiPassOutlierDetector',
    # Imputation
    'Imputer',
    # Expression filtering
    'StratifiedExpressionFilter',
    'ExpressionFilterResult',
    # Biomarker discovery
    'MarkerDiscovery',
    'MarkerDiscoveryResult',
    'FeatureScore',
    # Sex classification (data-driven, no hardcoded biology)
    'Sex',
    'SupervisedSexClassifier',
    'SupervisedClassifierResult',
    'SemiSupervisedSexClassifier',
    'SemiSupervisedResult',
]
