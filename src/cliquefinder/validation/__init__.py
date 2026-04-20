"""
Biological validation framework for imputation quality assessment.

Provides annotation providers, enrichment testing, and ID mapping
for assessing whether imputed values preserve biological signal.
"""

from cliquefinder.validation.annotation_providers import (
    AnnotationProvider,
    GOAnnotationProvider,
)

from cliquefinder.validation.enrichment_tests import (
    HypergeometricTest,
    apply_fdr_correction,
)

from cliquefinder.validation.id_mapping import (
    IDMapper,
    MyGeneInfoMapper,
)

__all__ = [
    'AnnotationProvider',
    'GOAnnotationProvider',
    'HypergeometricTest',
    'apply_fdr_correction',
    'IDMapper',
    'MyGeneInfoMapper',
]
