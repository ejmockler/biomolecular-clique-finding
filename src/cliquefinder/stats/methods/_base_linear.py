"""
Shared base class for OLS and LMM linear method implementations.

OLSMethod and LMMMethod share ~80% of their ``test()`` code structure:
building contrasts, iterating over cliques, summarizing, running
``differential_analysis_single``, and packaging results.  This module
extracts that shared logic into ``_BaseLinearMethod`` so the concrete
subclasses only need to define their unique configuration and the
method-specific metadata.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..experiment import PreparedCliqueExperiment
    from ..method_comparison_types import MethodName, UnifiedCliqueResult


class _BaseLinearMethod(abc.ABC):
    """
    Shared skeleton for OLS / LMM clique differential testing.

    Subclasses implement:
        * ``name`` property  (MethodName)
        * ``_get_subject_array``  (return subject labels or None)
        * ``_use_mixed``  (bool -- whether to fit mixed model)
        * ``_build_metadata``  (method-specific metadata dict per result)
    """

    def __init__(self, summarization: str = "tmp") -> None:
        self.summarization = summarization

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> MethodName:  # pragma: no cover
        ...

    @abc.abstractmethod
    def _get_subject_array(
        self, experiment: PreparedCliqueExperiment
    ) -> object | None:
        """Return subject labels array or None (OLS)."""
        ...

    @abc.abstractmethod
    def _use_mixed(self) -> bool:
        """Whether to pass use_mixed=True to differential_analysis_single."""
        ...

    @abc.abstractmethod
    def _build_metadata(
        self,
        protein_result: object,
        summary: object,
    ) -> dict[str, object]:
        """Return method-specific metadata dict to embed in UnifiedCliqueResult."""
        ...

    # ------------------------------------------------------------------
    # Shared test() implementation
    # ------------------------------------------------------------------

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run linear differential test on all cliques in the experiment.

        Shared workflow for OLS and LMM:
            1. Build contrast matrix from experiment
            2. Resolve summarization method
            3. Iterate over cliques:
               a. Extract protein expression data
               b. Summarize to clique-level abundance
               c. Fit OLS or LMM via differential_analysis_single
               d. Package result into UnifiedCliqueResult
            4. Report summary statistics

        Args:
            experiment: Prepared and preprocessed experiment data (immutable).
            verbose: If True, print progress messages.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of UnifiedCliqueResult objects, one per clique with sufficient data.
        """
        import pandas as pd

        from ..method_comparison_types import UnifiedCliqueResult
        from ..summarization import SummarizationMethod, summarize_clique
        from ..differential import differential_analysis_single, build_contrast_matrix

        results: list[UnifiedCliqueResult] = []

        # Build contrast matrix for the experiment's contrast
        conditions = list(experiment.conditions)
        contrast_dict = {experiment.contrast_name: experiment.contrast}
        contrast_matrix, contrast_names = build_contrast_matrix(conditions, contrast_dict)

        # Get sample condition labels
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            raise TypeError("experiment.sample_metadata must be a pandas DataFrame")

        sample_condition = experiment.sample_metadata[experiment.condition_column].values

        # Get subject labels (None for OLS, array for LMM)
        sample_subject = self._get_subject_array(experiment)

        # Resolve summarization method
        try:
            sum_method = SummarizationMethod(self.summarization)
        except ValueError:
            sum_method = SummarizationMethod.TUKEY_MEDIAN_POLISH

        n_processed = 0
        n_skipped = 0

        for clique in experiment.cliques:
            # Access clique attributes
            clique_id = getattr(clique, 'clique_id', str(clique))
            protein_ids_definition = getattr(clique, 'protein_ids', [])

            # Get clique data from experiment
            clique_data, clique_features = experiment.get_clique_data(clique_id)

            if len(clique_features) < 2:
                # Not enough proteins found in data
                n_skipped += 1
                continue

            # Summarize proteins to clique level
            summary = summarize_clique(
                clique_data,
                clique_features,
                clique_id,
                method=sum_method,
                compute_coherence=True,
            )

            # Run differential analysis
            try:
                protein_result = differential_analysis_single(
                    intensities=summary.sample_abundances,
                    condition=sample_condition,
                    subject=sample_subject,
                    feature_id=clique_id,
                    contrast_matrix=contrast_matrix,
                    contrast_names=contrast_names,
                    conditions=conditions,
                    use_mixed=self._use_mixed(),
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: {self.name.value} failed for {clique_id}: {e}")
                n_skipped += 1
                continue

            # Extract contrast result and convert to UnifiedCliqueResult
            if protein_result.contrasts:
                contrast = protein_result.contrasts[0]

                results.append(UnifiedCliqueResult(
                    clique_id=clique_id,
                    method=self.name,
                    effect_size=contrast.log2_fc,
                    effect_size_se=contrast.se,
                    p_value=contrast.p_value,
                    statistic_value=contrast.t_value,
                    statistic_type="t",
                    degrees_of_freedom=contrast.df,
                    n_proteins=len(protein_ids_definition),
                    n_proteins_found=len(clique_features),
                    method_metadata=self._build_metadata(protein_result, summary),
                ))
                n_processed += 1
            else:
                n_skipped += 1

        if verbose:
            print(f"{self.name.value}: processed {n_processed} cliques, skipped {n_skipped}")

        return results
