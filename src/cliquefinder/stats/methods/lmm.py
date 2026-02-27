"""
LMM (Linear Mixed Model) clique differential testing method.

This module implements the LMM-based method for testing clique-level
differential abundance with random subject effects, properly
accounting for repeated measurements from the same individuals.

Re-exported from ``method_comparison`` for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base_linear import _BaseLinearMethod

if TYPE_CHECKING:
    from ..experiment import PreparedCliqueExperiment
    from ..method_comparison_types import MethodName


class LMMMethod(_BaseLinearMethod):
    """
    Linear Mixed Model clique differential testing with random subject effects.

    This method extends OLS by incorporating random effects for biological subjects,
    properly accounting for repeated measurements from the same individuals.

    Statistical Approach:
        1. Summarize proteins within each clique via Tukey's Median Polish (or other method)
        2. Fit LMM: summarized_abundance ~ condition + (1 | subject)
        3. Compute Satterthwaite degrees of freedom for the contrast
        4. Test contrast with t-statistic
        5. Return UnifiedCliqueResult for each clique

    The random intercept for subject captures between-subject variability,
    preventing pseudoreplication when subjects have multiple samples.

    Attributes:
        summarization: Method for aggregating proteins within clique.
            Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca"
        use_satterthwaite: Whether to use Satterthwaite degrees of freedom approximation.
            If True (default), computes more accurate df for mixed models.

    Requires:
        experiment.subject_column must be set to the column name containing
        subject identifiers.

    Example:
        >>> from cliquefinder.stats.method_comparison import LMMMethod, prepare_experiment
        >>> method = LMMMethod(summarization="tmp", use_satterthwaite=True)
        >>> experiment = prepare_experiment(
        ...     data, feature_ids, metadata, cliques,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',  # Required for LMM
        ... )
        >>> results = method.test(experiment)

    See Also:
        - OLSMethod: Fixed effects variant without random subject effects
        - differential_analysis_single: Underlying statistical function
        - satterthwaite_df: Degrees of freedom approximation
    """

    def __init__(
        self,
        summarization: str = "tmp",
        use_satterthwaite: bool = True,
    ) -> None:
        """
        Initialize LMM method adapter.

        Args:
            summarization: Summarization method for aggregating proteins.
                Options: "tmp" (Tukey's Median Polish), "median", "mean", "logsum", "pca".
                Default is "tmp" which is robust to outliers.
            use_satterthwaite: Whether to use Satterthwaite approximation for df.
                Default True. Set to False for faster but less accurate df.
        """
        super().__init__(summarization=summarization)
        self.use_satterthwaite = use_satterthwaite

    @property
    def name(self) -> MethodName:
        """Return the method identifier."""
        from ..method_comparison_types import MethodName
        return MethodName.LMM

    def _get_subject_array(
        self, experiment: PreparedCliqueExperiment
    ) -> object | None:
        """LMM requires subject labels for random effects."""
        import pandas as pd

        if experiment.subject_column is None:
            raise ValueError(
                "LMMMethod requires subject_column in experiment. "
                "Set subject_column when calling prepare_experiment()."
            )
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            raise TypeError("experiment.sample_metadata must be a pandas DataFrame")
        return experiment.sample_metadata[experiment.subject_column].values

    def _use_mixed(self) -> bool:
        """LMM uses mixed model with random subject effects."""
        return True

    def _build_metadata(
        self,
        protein_result: object,
        summary: object,
    ) -> dict[str, object]:
        """Build LMM-specific metadata (includes variance components)."""
        return {
            'summarization': self.summarization,
            'use_satterthwaite': self.use_satterthwaite,
            'ci_lower': protein_result.contrasts[0].ci_lower,
            'ci_upper': protein_result.contrasts[0].ci_upper,
            'coherence': summary.coherence,
            'model_type': protein_result.model_type.value,
            'subject_variance': protein_result.subject_variance,
            'residual_variance': protein_result.residual_variance,
            'convergence': protein_result.convergence,
            'issue': protein_result.issue,
        }

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run LMM differential test on all cliques in the experiment.

        For each clique:
            1. Extract protein expression data using experiment.get_clique_data()
            2. Summarize to clique-level abundance using the specified method
            3. Fit LMM: abundance ~ condition + (1 | subject)
            4. Test the specified contrast with Satterthwaite df
            5. Package results into UnifiedCliqueResult

        Cliques with fewer than 2 proteins found in the data are skipped.

        Args:
            experiment: Prepared and preprocessed experiment data (immutable).
                Must have subject_column set for random effects.
            verbose: If True, print progress messages.
            **kwargs: Additional keyword arguments (ignored for LMM).

        Returns:
            List of UnifiedCliqueResult objects, one per clique with sufficient data.

        Raises:
            ValueError: If experiment.subject_column is None.
        """
        from ..method_comparison_types import UnifiedCliqueResult

        # Validate subject column early (before entering base class loop)
        if experiment.subject_column is None:
            raise ValueError(
                "LMMMethod requires subject_column in experiment. "
                "Set subject_column when calling prepare_experiment()."
            )

        # Track fallback to OLS for verbose reporting
        results = super().test(experiment, verbose=verbose, **kwargs)

        if verbose:
            from ..differential import ModelType
            n_fallback = sum(
                1 for r in results
                if r.method_metadata.get('model_type') == ModelType.FIXED.value
            )
            if n_fallback > 0:
                print(f"     (note: {n_fallback} fell back to OLS due to convergence)")

        return results


__all__ = ["LMMMethod"]
