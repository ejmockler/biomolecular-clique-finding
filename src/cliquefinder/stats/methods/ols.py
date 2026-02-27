"""
OLS (Ordinary Least Squares) clique differential testing method.

This module implements the OLS-based method for testing clique-level
differential abundance. It uses fixed effects regression without
random subject effects.

Re-exported from ``method_comparison`` for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base_linear import _BaseLinearMethod

if TYPE_CHECKING:
    from ..experiment import PreparedCliqueExperiment
    from ..method_comparison_types import MethodName


class OLSMethod(_BaseLinearMethod):
    """
    OLS-based clique differential testing with optional Empirical Bayes moderation.

    This method wraps the existing differential analysis infrastructure to test
    clique-level differential abundance using fixed effects ordinary least squares.

    Statistical Approach:
        1. Summarize proteins within each clique via Tukey's Median Polish (or other method)
        2. Fit OLS: summarized_abundance ~ condition
        3. Test contrast with t-statistic
        4. Return UnifiedCliqueResult for each clique

    The summarization step aggregates protein-level data into a single clique-level
    abundance per sample, which is then tested for differential abundance between
    conditions.

    Attributes:
        summarization: Method for aggregating proteins within clique.
            Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca"
        eb_moderation: Whether to apply Empirical Bayes variance shrinkage (future).
            Currently not implemented but reserved for limma-style moderation.

    Example:
        >>> from cliquefinder.stats.method_comparison import OLSMethod, prepare_experiment
        >>> method = OLSMethod(summarization="tmp", eb_moderation=True)
        >>> experiment = prepare_experiment(data, feature_ids, metadata, cliques, ...)
        >>> results = method.test(experiment)
        >>> significant = [r for r in results if r.p_value < 0.05]

    See Also:
        - LMMMethod: Linear mixed model variant with random subject effects
        - differential_analysis_single: Underlying statistical function
        - summarize_clique: Summarization function
    """

    def __init__(
        self,
        summarization: str = "tmp",
        eb_moderation: bool = True,
    ) -> None:
        """
        Initialize OLS method adapter.

        Args:
            summarization: Summarization method for aggregating proteins.
                Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca".
                Default is "tmp" which is robust to outliers.
            eb_moderation: Whether to apply Empirical Bayes variance shrinkage.
                Currently reserved for future implementation.
        """
        super().__init__(summarization=summarization)
        self.eb_moderation = eb_moderation

    @property
    def name(self) -> MethodName:
        """Return the method identifier."""
        from ..method_comparison_types import MethodName
        return MethodName.OLS

    def _get_subject_array(
        self, experiment: PreparedCliqueExperiment
    ) -> object | None:
        """OLS does not use subject random effects."""
        return None

    def _use_mixed(self) -> bool:
        """OLS uses fixed effects only."""
        return False

    def _build_metadata(
        self,
        protein_result: object,
        summary: object,
    ) -> dict[str, object]:
        """Build OLS-specific metadata."""
        return {
            'summarization': self.summarization,
            'eb_moderation': self.eb_moderation,
            'ci_lower': protein_result.contrasts[0].ci_lower,
            'ci_upper': protein_result.contrasts[0].ci_upper,
            'coherence': summary.coherence,
            'model_type': protein_result.model_type.value,
        }


__all__ = ["OLSMethod"]
