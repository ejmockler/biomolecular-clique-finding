"""
Biological validation framework for imputation quality assessment.

This module provides comprehensive biological validation tools to assess whether
imputed values are biologically coherent or represent destroyed biological signal.

Key Question:
    Are outliers technical artifacts (safe to impute) or biological signal
    (imputation would destroy)?

Validation Strategy:
    1. Biological Coherence: Do imputed genes cluster in functional pathways?
    2. Strategy Comparison: Which imputation method preserves biology best?
    3. Correlation Preservation: Are known co-expression patterns maintained?
    4. Disease Association: Are imputed genes enriched for disease markers?

Architecture:
    - Modality-agnostic: Works for proteomics, transcriptomics, metabolomics
    - Annotation-agnostic: Pluggable providers for GO, KEGG, DisGeNET, etc.
    - Strategy-agnostic: Compare any imputation method
    - Statistically rigorous: Multiple testing correction, effect sizes, CIs

Modules:
    annotation_providers: Abstract interface + concrete implementations
    enrichment_tests: Statistical enrichment testing with FDR correction
    biological_validation: Main orchestration and coherence testing
    imputation_comparison: Strategy comparison framework
    correlation_preservation: Known structure preservation analysis
    reports: Publication-quality report generation

Examples:
    >>> from cliquefinder.validation import BiologicalValidator
    >>> from cliquefinder.validation.annotation_providers import GOAnnotationProvider
    >>>
    >>> # Validate imputation quality
    >>> validator = BiologicalValidator(
    ...     annotation_provider=GOAnnotationProvider()
    ... )
    >>>
    >>> # Test biological coherence
    >>> results = validator.validate_imputation(
    ...     original_matrix=original,
    ...     imputed_matrix=imputed,
    ...     outlier_mask=outlier_mask
    ... )
    >>>
    >>> # Compare imputation strategies
    >>> from cliquefinder.validation import ImputationStrategyComparator
    >>> comparator = ImputationStrategyComparator(
    ...     strategies=['remove', 'knn_k5', 'knn_k10', 'median', 'winsorize']
    ... )
    >>> comparison = comparator.compare(matrix, outlier_mask, validator)
"""

from cliquefinder.validation.annotation_providers import (
    AnnotationProvider,
    GOAnnotationProvider,
    # KEGGAnnotationProvider,  # TODO: Implement
    # DisGeNETProvider,  # TODO: Implement
    # CachedAnnotationProvider,  # TODO: Implement
)

from cliquefinder.validation.enrichment_tests import (
    # EnrichmentTest,  # TODO: Check if implemented
    HypergeometricTest,
    # FisherExactTest,  # TODO: Check if implemented
    apply_fdr_correction,
)

from cliquefinder.validation.id_mapping import (
    IDMapper,
    MyGeneInfoMapper,
)

# TODO: Implement remaining modules
# from cliquefinder.validation.biological_validation import (
#     BiologicalValidator,
#     BiologicalCoherenceResult,
# )
#
# from cliquefinder.validation.imputation_comparison import (
#     ImputationStrategyComparator,
#     ImputationStrategy,
#     StrategyComparisonResult,
# )
#
# from cliquefinder.validation.correlation_preservation import (
#     CorrelationPreservationAnalyzer,
#     CorrelationPreservationResult,
# )

__all__ = [
    # Annotation providers (implemented)
    'AnnotationProvider',
    'GOAnnotationProvider',

    # Enrichment testing (implemented)
    'HypergeometricTest',
    'apply_fdr_correction',

    # ID mapping (implemented)
    'IDMapper',
    'MyGeneInfoMapper',

    # TODO: Add remaining exports when implemented
    # 'KEGGAnnotationProvider',
    # 'DisGeNETProvider',
    # 'CachedAnnotationProvider',
    # 'EnrichmentTest',
    # 'FisherExactTest',
    # 'BiologicalValidator',
    # 'BiologicalCoherenceResult',
    # 'ImputationStrategyComparator',
    # 'ImputationStrategy',
    # 'StrategyComparisonResult',
    # 'CorrelationPreservationAnalyzer',
    # 'CorrelationPreservationResult',
]
