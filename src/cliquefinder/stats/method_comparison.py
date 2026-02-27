"""
Method Comparison Framework for Multi-Method Clique Differential Testing.

This module provides unified abstractions for running multiple statistical methods
(OLS, LMM, ROAST, Permutation) on the same cliques and quantifying their concordance.

Key Principle: Methods answer different questions - disagreement is informative, not a bug.

| Method        | Question Answered                          | Test Type             |
|---------------|--------------------------------------------|-----------------------|
| OLS/LMM       | "Is aggregated clique abundance different?"| Parametric contrast   |
| ROAST         | "Are genes in this clique enriched for DE?"| Self-contained rotation|
| Permutation   | "Is this clique more DE than random sets?" | Competitive enrichment|

Design Principles:
    1. Frozen (immutable) dataclasses for reproducibility
    2. Protocol-based method interface for extensibility
    3. Unified result format across all methods
    4. Concordance metrics for method comparison

Statistical Note:
    This framework is for DESCRIPTIVE comparison, not inference.
    - We do NOT select the "best" p-value per clique (would inflate FDR)
    - We do NOT combine p-values across methods (requires strong assumptions)
    - We DO report concordance metrics to characterize agreement
    - We DO flag disagreement cases for biological investigation

References:
    - Wu D, Smyth GK (2012). "Camera: a competitive gene set test accounting for
      inter-gene correlation." Nucleic Acids Research 40(17):e133.
    - Subramanian A, et al. (2005). "Gene set enrichment analysis: A knowledge-based
      approach." PNAS 102(43):15545-50.
    - Cohen J (1960). "A coefficient of agreement for nominal scales."
      Educational and Psychological Measurement 20(1):37-46.

Module Structure (decomposed into sub-modules for maintainability):
    - method_comparison_types.py : Enums, dataclasses, protocol
    - experiment.py              : PreparedCliqueExperiment + prepare_experiment
    - methods/ols.py             : OLSMethod
    - methods/lmm.py             : LMMMethod
    - methods/roast.py           : ROASTMethod
    - methods/permutation.py     : PermutationMethod
    - methods/_base_linear.py    : Shared base for OLS/LMM
    - concordance.py             : concordance functions + MethodComparisonResult
    - method_comparison.py       : run_method_comparison + re-exports (this file)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# ---- Re-exports for backward compatibility --------------------------------
# All public symbols that were previously defined in this monolith are
# re-exported here so that ``from cliquefinder.stats.method_comparison import X``
# continues to work unchanged.

from .method_comparison_types import (  # noqa: F401
    MethodName,
    UnifiedCliqueResult,
    ConcordanceMetrics,
    CliqueTestMethod,
)

from .experiment import (  # noqa: F401
    PreparedCliqueExperiment,
    prepare_experiment,
)

from .methods import (  # noqa: F401
    OLSMethod,
    LMMMethod,
    ROASTMethod,
    PermutationMethod,
)

from .concordance import (  # noqa: F401
    compute_pairwise_concordance,
    identify_disagreements,
    MethodComparisonResult,
)

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# Main Entry Point
# =============================================================================


def run_method_comparison(
    data: np.ndarray,
    feature_ids: list[str],
    sample_metadata: "pd.DataFrame",
    cliques: list[object],  # list[CliqueDefinition]
    condition_column: str,
    contrast: tuple[str, str],
    subject_column: str | None = None,
    methods: list[CliqueTestMethod] | None = None,
    concordance_threshold: float = 0.05,
    normalization_method: str = "median",
    imputation_method: str = "min_feature",
    n_rotations: int = 9999,
    n_permutations: int = 10000,
    use_gpu: bool = True,
    seed: int | None = None,
    verbose: bool = True,
    precomputed_symbol_map: dict[str, str] | None = None,
) -> MethodComparisonResult:
    """
    Run multiple differential testing methods and compute concordance.

    This is the main entry point for method comparison. Pipeline:
        1. Prepare data ONCE (shared preprocessing)
        2. Run each method on the prepared data
        3. Compute pairwise concordance metrics
        4. Identify disagreement cases
        5. Return structured comparison results

    Statistical Note:
        Results are DESCRIPTIVE. Do NOT:
        - Select the "best" p-value per clique (inflates FDR)
        - Combine p-values (requires careful assumptions)
        - Declare a "winner" method

        Instead, use concordance to:
        - Validate findings (high agreement = robust)
        - Understand method behavior
        - Flag interesting disagreements

    Args:
        data: Expression matrix (n_features, n_samples) of log2 intensities.
        feature_ids: Gene/protein identifiers matching data rows.
        sample_metadata: DataFrame with sample annotations.
        cliques: List of CliqueDefinition objects to test.
        condition_column: Metadata column name for condition labels.
        contrast: Tuple of (test_condition, reference_condition).
        subject_column: Optional metadata column for biological replicates.
            If provided, LMMMethod will be included in the default methods.
        methods: List of CliqueTestMethod instances to run.
            Default is [OLSMethod(), LMMMethod() if subject_column, ROASTMethod(msq),
            ROASTMethod(mean), PermutationMethod()].
        concordance_threshold: P-value threshold for classification agreement
            analysis (default 0.05).
        normalization_method: Preprocessing normalization (default "median").
            Options: "none", "median", "quantile", "global_standards", "vsn".
        imputation_method: Preprocessing imputation (default "min_feature").
            Options: "none", "min_feature", "min_global", "min_sample", "knn".
        n_rotations: Number of rotations for ROAST method (default 9999).
        n_permutations: Number of permutations for permutation method (default 10000).
        use_gpu: Whether to use GPU acceleration (default True). Falls back to
            CPU if GPU is not available.
        seed: Random seed for reproducibility. Applies to ROAST and permutation
            methods.
        verbose: Print progress messages (default True).

    Returns:
        MethodComparisonResult with:
            - Per-method results (results_by_method)
            - Pairwise concordance metrics (pairwise_concordance)
            - Mean concordance statistics (mean_spearman_rho, mean_cohen_kappa)
            - Disagreement analysis (disagreement_cases DataFrame)
            - Wide-format combined table (via wide_format() method)

    Raises:
        ValueError: If contrast conditions not found in data.
        ValueError: If no cliques have proteins in data.

    Example:
        >>> from cliquefinder.stats import run_method_comparison, MethodName
        >>> comparison = run_method_comparison(
        ...     data=expression_matrix,
        ...     feature_ids=protein_ids,
        ...     sample_metadata=metadata,
        ...     cliques=clique_definitions,
        ...     condition_column='phenotype',
        ...     contrast=('ALS', 'Control'),
        ...     subject_column='subject_id',
        ... )
        >>>
        >>> # Check overall agreement
        >>> print(f"Mean Spearman rho: {comparison.mean_spearman_rho:.3f}")
        >>>
        >>> # Get robust hits (significant in all methods)
        >>> robust = comparison.robust_hits(threshold=0.05)
        >>>
        >>> # Get ROAST-specific hits (bidirectional regulation?)
        >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
        >>>
        >>> # Export wide format for downstream analysis
        >>> wide_df = comparison.wide_format()
        >>> wide_df.to_csv("method_comparison.csv")

    See Also:
        - prepare_experiment: Data preprocessing function
        - MethodComparisonResult: Output structure with helper methods
        - ConcordanceMetrics: Pairwise agreement metrics
        - UnifiedCliqueResult: Individual method results
    """
    import pandas as pd

    # 1. Print header (if verbose)
    if verbose:
        print("=" * 60)
        print("METHOD COMPARISON FRAMEWORK")
        print("=" * 60)
        print()

    # 2. Prepare experiment using prepare_experiment()
    # This ensures all methods get the same preprocessed data
    experiment = prepare_experiment(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=sample_metadata,
        cliques=cliques,
        condition_column=condition_column,
        contrast=contrast,
        subject_column=subject_column,
        normalization_method=normalization_method,
        imputation_method=imputation_method,
        verbose=verbose,
        precomputed_symbol_map=precomputed_symbol_map,
    )

    # 3. Default methods if not specified
    if methods is None:
        methods = [
            OLSMethod(),
        ]
        # Add LMMMethod if subject_column is set
        if subject_column is not None:
            methods.append(LMMMethod())
        # Add ROAST methods with different statistics
        methods.extend([
            ROASTMethod(statistic="msq", n_rotations=n_rotations),
            ROASTMethod(statistic="mean", n_rotations=n_rotations),
        ])
        # Add permutation method
        methods.append(PermutationMethod(n_permutations=n_permutations))

    if verbose:
        print()
        print(f"Methods to run: {[m.name.value for m in methods]}")
        print()

    # 4. Run each method
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]] = {}
    failed_methods: dict[str, str] = {}

    for method in methods:
        if verbose:
            print(f"Running {method.name.value}...", end=" ", flush=True)

        try:
            results = method.test(
                experiment,
                use_gpu=use_gpu,
                seed=seed,
                verbose=False,  # Method-level verbosity off
            )
            results_by_method[method.name] = results

            if verbose:
                n_sig = sum(1 for r in results if r.p_value < concordance_threshold)
                print(f"done ({len(results)} cliques, {n_sig} significant)")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            logger.warning("Method %s failed: %s", method.name.value, e)
            results_by_method[method.name] = []
            failed_methods[method.name.value] = str(e)

    # 5. Compute pairwise concordance
    if verbose:
        print()
        print("Computing concordance metrics...")

    # Get method names with non-empty results
    method_names = [
        m for m in results_by_method.keys()
        if len(results_by_method[m]) > 0
    ]
    pairwise: list[ConcordanceMetrics] = []

    for i, name_a in enumerate(method_names):
        for name_b in method_names[i + 1:]:
            try:
                conc = compute_pairwise_concordance(
                    results_by_method[name_a],
                    results_by_method[name_b],
                    threshold=concordance_threshold,
                )
                pairwise.append(conc)
            except ValueError as e:
                if verbose:
                    print(f"  Skipping {name_a.value} vs {name_b.value}: {e}")

    # 6. Aggregate metrics
    if pairwise:
        mean_rho = float(np.mean([c.spearman_rho for c in pairwise]))
        mean_kappa = float(np.mean([c.cohen_kappa for c in pairwise]))
    else:
        mean_rho = np.nan
        mean_kappa = np.nan

    # 7. Identify disagreements
    disagreements = identify_disagreements(
        results_by_method,
        threshold=concordance_threshold,
    )

    # 8. Count cliques tested (union of all clique_ids across all methods)
    all_tested: set[str] = set()
    for results in results_by_method.values():
        all_tested.update(r.clique_id for r in results if r.is_valid)

    # 9. Print summary (if verbose)
    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        n_attempted = len(methods)
        n_succeeded = len(method_names)
        print(f"Methods attempted: {n_attempted}, succeeded: {n_succeeded}")
        print(f"Cliques tested: {len(all_tested)}")
        print(f"Mean Spearman rho: {mean_rho:.3f}")
        print(f"Mean Cohen's kappa: {mean_kappa:.3f}")

        if isinstance(disagreements, pd.DataFrame):
            print(f"Disagreement cases: {len(disagreements)}")
        else:
            print("Disagreement cases: 0")

        if failed_methods:
            print()
            print("FAILED methods:")
            for mname, err in sorted(failed_methods.items()):
                print(f"  {mname}: {err}")

        print()

        # Quick concordance table
        if pairwise:
            print("Pairwise concordance (Spearman rho):")
            for conc in pairwise:
                print(
                    f"  {conc.method_a.value} vs {conc.method_b.value}: "
                    f"rho={conc.spearman_rho:.3f}, kappa={conc.cohen_kappa:.3f}"
                )

    # 10. Return MethodComparisonResult with all fields populated
    return MethodComparisonResult(
        results_by_method=results_by_method,
        pairwise_concordance=pairwise,
        mean_spearman_rho=mean_rho,
        mean_cohen_kappa=mean_kappa,
        disagreement_cases=disagreements,
        preprocessing_params=experiment.preprocessing_params,
        methods_run=list(method_names),
        n_cliques_tested=len(all_tested),
        failed_methods=failed_methods,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MethodName",
    # Core dataclasses
    "UnifiedCliqueResult",
    "PreparedCliqueExperiment",
    "ConcordanceMetrics",
    "MethodComparisonResult",
    # Protocol
    "CliqueTestMethod",
    # Factory function
    "prepare_experiment",
    # Method adapters
    "OLSMethod",
    "LMMMethod",
    "ROASTMethod",
    "PermutationMethod",
    # Concordance functions
    "compute_pairwise_concordance",
    "identify_disagreements",
    # Main entry point
    "run_method_comparison",
]
