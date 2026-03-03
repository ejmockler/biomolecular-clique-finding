"""
Competitive permutation-based clique testing method adapter.

Tests whether a clique has stronger differential expression than
random gene sets of the same size drawn from the measured proteome.

Re-exported from ``method_comparison`` for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..experiment import PreparedCliqueExperiment
    from ..method_comparison_types import MethodName, UnifiedCliqueResult


class PermutationMethod:
    """
    Competitive permutation-based testing method adapter.

    Tests whether a clique has stronger differential expression than
    random gene sets of the same size from the measured proteome.
    This is a competitive test - it asks "Is this clique more DE than
    random sets of the same size?"

    Null Hypothesis:
        Random gene sets show equal or greater effect than the observed clique.

    Key Characteristics:
        - Non-parametric (no distributional assumptions)
        - Accounts for gene-gene correlation via empirical null
        - Competitive enrichment (relative to background)

    Statistical Approach:
        1. Summarize proteins within clique via Tukey's Median Polish
        2. Compute t-statistic for observed clique
        3. Sample random gene sets of same size from regulated pool
        4. Compute t-statistics for random sets
        5. Empirical p-value = fraction of null >= observed

    Attributes:
        n_permutations: Number of permutations for null distribution
        summarization: Method for aggregating proteins ("tmp" = median polish)

    Example:
        >>> method = PermutationMethod(n_permutations=10000)
        >>> results = method.test(experiment)
        >>> # Low p-value means clique is more DE than expected by chance
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        summarization: str = "tmp",
    ) -> None:
        """
        Initialize permutation method adapter.

        Args:
            n_permutations: Number of random gene set permutations.
                10000 is standard (gives minimum p-value of 1/10001 ~ 0.0001).
                Higher = more precise p-values, more compute time.
            summarization: How to aggregate proteins within clique.
                "tmp": Tukey's Median Polish (robust, recommended)
        """
        self.n_permutations = n_permutations
        self.summarization = summarization

    @property
    def name(self) -> MethodName:
        """
        Get the method name.

        Returns:
            MethodName.PERMUTATION_COMPETITIVE
        """
        from ..method_comparison_types import MethodName
        return MethodName.PERMUTATION_COMPETITIVE

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run competitive permutation test on all cliques in the experiment.

        Args:
            experiment: Prepared clique experiment (immutable)
            use_gpu: Whether to use GPU acceleration (requires MLX)
            seed: Random seed for reproducibility
            verbose: Print progress messages
            **kwargs: Additional arguments (ignored)

        Returns:
            List of UnifiedCliqueResult, one per clique with sufficient data
        """
        import logging
        import pandas as pd
        from scipy import stats as scipy_stats

        from ..method_comparison_types import UnifiedCliqueResult

        logger = logging.getLogger(__name__)

        # Import permutation module
        try:
            from ..permutation_gpu import run_permutation_test_gpu
        except ImportError as e:
            logger.warning(f"Permutation method unavailable (GPU): {e}")
            # Could fall back to CPU version here if available
            return []

        results: list[UnifiedCliqueResult] = []

        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            logger.warning("Permutation method requires sample_metadata as DataFrame")
            return []

        # MCOMP-7: Defensive copy of sample_metadata to prevent mutation
        # by the permutation engine.
        sample_metadata_copy = experiment.sample_metadata.copy()

        # Run GPU-accelerated permutation test
        try:
            perm_results, null_df = run_permutation_test_gpu(
                data=experiment.data,
                feature_ids=list(experiment.feature_ids),
                sample_metadata=sample_metadata_copy,
                clique_definitions=list(experiment.cliques),
                condition_col=experiment.condition_column,
                contrast=experiment.contrast,
                subject_col=experiment.subject_column,
                n_permutations=self.n_permutations,
                random_state=seed,
                verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Permutation test failed: {e}")
            return []

        # MCOMP-11: Guard against duplicate clique_ids in null_df
        if not null_df.empty and null_df["clique_id"].duplicated().any():
            n_dupes = int(null_df["clique_id"].duplicated().sum())
            logger.warning(
                "Duplicate clique_ids detected in permutation null_df (%d duplicates); "
                "keeping first occurrence",
                n_dupes,
            )
            null_df = null_df.drop_duplicates(subset=["clique_id"], keep="first")

        # MCOMP-4: Build clique lookup dict once — O(M) — instead of
        # scanning experiment.cliques per result — O(N*M).
        clique_lookup: dict[str, object] = {
            getattr(c, "clique_id", str(c)): c for c in experiment.cliques
        }

        # Convert PermutationTestResult to UnifiedCliqueResult
        # perm_results is a list of PermutationTestResult dataclass objects
        for perm_result in perm_results:
            try:
                # Extract fields from PermutationTestResult
                clique_id = perm_result.clique_id
                observed_t = perm_result.observed_tvalue
                empirical_p = perm_result.empirical_pvalue

                # MCOMP-3: Preserve original empirical p-value for storage.
                # Only clamp for z-score computation (ppf needs (0, 1)).
                p_for_z = empirical_p
                if p_for_z <= 0 or p_for_z >= 1:
                    p_for_z = max(1e-15, min(1 - 1e-15, p_for_z))

                # Convert clamped p-value to z-score (preserving sign from observed t)
                if observed_t >= 0:
                    z_score = -scipy_stats.norm.ppf(p_for_z / 2)  # Two-sided
                else:
                    z_score = scipy_stats.norm.ppf(p_for_z / 2)

                # MCOMP-4: O(1) clique lookup via pre-built dict
                clique_def = clique_lookup.get(clique_id)

                if clique_def is not None:
                    n_proteins = len(getattr(clique_def, "protein_ids", []))
                else:
                    n_proteins = 0

                # MCOMP-5: Use experiment.clique_to_feature_indices for
                # protein counting, consistent with all other methods.
                feature_indices = experiment.clique_to_feature_indices.get(
                    clique_id, ()
                )
                n_found = len(feature_indices)

                # Build method metadata from null distribution info
                null_row = null_df[null_df["clique_id"] == clique_id] if not null_df.empty else None
                null_mean = float(null_row["null_tvalue_mean"].iloc[0]) if null_row is not None and len(null_row) > 0 else np.nan
                null_std = float(null_row["null_tvalue_std"].iloc[0]) if null_row is not None and len(null_row) > 0 else np.nan

                # Extract comparable log2FC if available
                _comparable_log2fc: float | None = None
                if hasattr(perm_result, "observed_log2fc"):
                    _comparable_log2fc = float(perm_result.observed_log2fc)

                results.append(
                    UnifiedCliqueResult(
                        clique_id=clique_id,
                        method=self.name,
                        effect_size=float(observed_t),  # Use observed t as effect size
                        effect_size_type="observed_t",
                        comparable_effect_size=_comparable_log2fc,
                        effect_size_se=None,  # Permutation doesn't provide SE
                        p_value=float(empirical_p),  # MCOMP-3: unclamped original
                        statistic_value=float(z_score),
                        statistic_type="empirical_z",
                        degrees_of_freedom=None,  # Non-parametric
                        n_proteins=n_proteins,
                        n_proteins_found=n_found,
                        method_metadata={
                            "n_permutations": self.n_permutations,
                            "percentile": float(perm_result.percentile_rank),
                            "null_mean": null_mean,
                            "null_std": null_std,
                            "observed_log2fc": float(perm_result.observed_log2fc),
                            "is_significant_empirical": perm_result.is_significant,
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Permutation result conversion failed for clique: {e}")
                continue

        return results


__all__ = ["PermutationMethod"]
