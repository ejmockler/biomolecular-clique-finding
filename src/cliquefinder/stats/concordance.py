"""
Concordance computation functions and the ``MethodComparisonResult`` dataclass.

This module provides:

* :func:`compute_pairwise_concordance` -- Spearman rho, Cohen's kappa, Pearson r,
  direction agreement, etc. between two methods' results.
* :func:`identify_disagreements` -- cliques where methods disagree on significance.
* :class:`MethodComparisonResult` -- aggregated output of :func:`run_method_comparison`.

These symbols are re-exported from ``method_comparison`` for backward
compatibility -- prefer importing from there in application code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .method_comparison_types import (
    ConcordanceMetrics,
    MethodName,
    UnifiedCliqueResult,
)

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# Concordance Computation Functions
# =============================================================================


def compute_pairwise_concordance(
    results_a: list[UnifiedCliqueResult],
    results_b: list[UnifiedCliqueResult],
    threshold: float = 0.05,
) -> ConcordanceMetrics:
    """
    Compute concordance metrics between two differential testing methods.

    This function quantifies agreement between two methods across multiple
    dimensions: rank correlation of p-values, classification agreement
    (significant vs non-significant), effect size correlation, and
    direction agreement.

    Statistical Background:
        - Spearman rho measures monotonic relationship between p-value ranks
        - Cohen's kappa adjusts for chance agreement in classification
        - Pearson r measures linear relationship between effect sizes
        - RMSE quantifies average magnitude of effect size differences

    Args:
        results_a: List of UnifiedCliqueResult from method A.
            Must contain at least 3 valid results.
        results_b: List of UnifiedCliqueResult from method B.
            Must contain at least 3 valid results.
        threshold: P-value threshold for significant/non-significant
            classification. Default is 0.05 (nominal significance level).

    Returns:
        ConcordanceMetrics dataclass with all agreement measures computed
        on the intersection of valid cliques tested by both methods.

    Raises:
        ValueError: If fewer than 3 cliques are common to both result sets.
            At least 3 observations are needed for meaningful correlation.

    Example:
        >>> from cliquefinder.stats.method_comparison import compute_pairwise_concordance
        >>> conc = compute_pairwise_concordance(ols_results, roast_results, threshold=0.05)
        >>> print(f"Spearman rho: {conc.spearman_rho:.3f}")
        >>> print(f"Cohen's kappa: {conc.cohen_kappa:.3f}")
        >>> print(f"Agreement rate: {conc.agreement_rate:.1%}")

    Notes:
        - Only valid results (finite p-value in [0,1] and finite effect size)
          are included in the comparison
        - Cliques must have matching clique_id to be compared
        - For effect size comparisons, both results must have finite effect sizes
    """
    from scipy import stats as scipy_stats

    # Build lookup by clique_id for valid results only
    a_by_id: dict[str, UnifiedCliqueResult] = {
        r.clique_id: r for r in results_a if r.is_valid
    }
    b_by_id: dict[str, UnifiedCliqueResult] = {
        r.clique_id: r for r in results_b if r.is_valid
    }

    # Find common cliques (intersection)
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    n = len(common_ids)

    if n < 3:
        raise ValueError(
            f"Need at least 3 common cliques for concordance analysis, got {n}. "
            f"Method A has {len(a_by_id)} valid results, "
            f"Method B has {len(b_by_id)} valid results."
        )

    # Extract aligned vectors for common cliques
    p_a = np.array([a_by_id[cid].p_value for cid in common_ids])
    p_b = np.array([b_by_id[cid].p_value for cid in common_ids])
    eff_a = np.array([a_by_id[cid].effect_size for cid in common_ids])
    eff_b = np.array([b_by_id[cid].effect_size for cid in common_ids])

    # 1. Rank correlation of p-values (Spearman)
    # Spearman rho is robust to outliers and non-linear relationships
    rho, rho_pval = scipy_stats.spearmanr(p_a, p_b)

    # 2. Classification agreement at threshold
    sig_a = p_a < threshold
    sig_b = p_b < threshold

    n_both_sig = int(np.sum(sig_a & sig_b))
    n_both_nonsig = int(np.sum(~sig_a & ~sig_b))
    n_a_only = int(np.sum(sig_a & ~sig_b))
    n_b_only = int(np.sum(~sig_a & sig_b))

    # 3. Cohen's kappa
    # kappa = (p_o - p_e) / (1 - p_e)
    # where p_o = observed agreement, p_e = expected agreement by chance
    p_o = (n_both_sig + n_both_nonsig) / n  # Observed agreement rate

    # Expected agreement: probability that both agree by chance
    # P(both sig) = P(A sig) * P(B sig)
    # P(both nonsig) = P(A nonsig) * P(B nonsig)
    p_a_sig_rate = np.sum(sig_a) / n
    p_b_sig_rate = np.sum(sig_b) / n
    p_e = p_a_sig_rate * p_b_sig_rate + (1 - p_a_sig_rate) * (1 - p_b_sig_rate)

    if p_e < 1:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        # Perfect expected agreement - kappa is 1 if observed is also perfect
        kappa = 1.0 if p_o == 1.0 else 0.0

    # 4. Effect size agreement (Pearson correlation and RMSE)
    # Only use cliques with finite effect sizes in both methods
    valid_eff_mask = np.isfinite(eff_a) & np.isfinite(eff_b)
    n_valid_eff = int(np.sum(valid_eff_mask))

    if n_valid_eff >= 3:
        eff_r, _ = scipy_stats.pearsonr(eff_a[valid_eff_mask], eff_b[valid_eff_mask])
        eff_rmse = np.sqrt(np.mean((eff_a[valid_eff_mask] - eff_b[valid_eff_mask]) ** 2))
    else:
        eff_r = np.nan
        eff_rmse = np.nan

    # 5. Direction agreement (same sign of effect size)
    # Important for detecting consistent biological interpretation
    if n_valid_eff >= 3:
        # Compare signs: positive, negative, or zero
        # Use np.sign which returns -1, 0, or 1
        sign_a = np.sign(eff_a[valid_eff_mask])
        sign_b = np.sign(eff_b[valid_eff_mask])
        same_sign = sign_a == sign_b
        dir_agree = float(np.mean(same_sign))
    else:
        dir_agree = np.nan

    # Get method names from the first valid result in each list
    # (all results in a list should have the same method)
    method_a = results_a[0].method
    method_b = results_b[0].method

    return ConcordanceMetrics(
        method_a=method_a,
        method_b=method_b,
        n_cliques_compared=n,
        spearman_rho=float(rho) if np.isfinite(rho) else np.nan,
        spearman_pvalue=float(rho_pval) if np.isfinite(rho_pval) else np.nan,
        threshold=threshold,
        n_both_significant=n_both_sig,
        n_both_nonsignificant=n_both_nonsig,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        cohen_kappa=float(kappa) if np.isfinite(kappa) else np.nan,
        effect_pearson_r=float(eff_r) if np.isfinite(eff_r) else np.nan,
        effect_rmse=float(eff_rmse) if np.isfinite(eff_rmse) else np.nan,
        direction_agreement_frac=float(dir_agree) if np.isfinite(dir_agree) else np.nan,
    )


def identify_disagreements(
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]],
    threshold: float = 0.05,
) -> "pd.DataFrame":
    """
    Identify cliques where statistical methods disagree on significance.

    A clique is flagged as a "disagreement case" if:
        - At least one method calls it significant (p < threshold)
        - At least one method calls it non-significant (p >= threshold)

    These disagreement cases are particularly interesting for biological
    investigation because they may reveal:
        - Method-specific sensitivities (e.g., ROAST detecting bidirectional
          regulation that OLS misses due to signal cancellation)
        - Edge cases near the significance boundary
        - Cliques where effect size interpretation differs across methods

    Args:
        results_by_method: Dictionary mapping MethodName to list of
            UnifiedCliqueResult for that method. All methods should have
            been run on the same set of cliques.
        threshold: P-value threshold for significance classification.
            Default is 0.05.

    Returns:
        pandas DataFrame with disagreement cases, containing columns:
            - clique_id: Clique identifier
            - {method}_pvalue: P-value from each method (e.g., ols_pvalue)
            - {method}_effect: Effect size from each method (e.g., ols_effect)
            - n_methods_significant: Count of methods calling clique significant
            - n_methods_nonsignificant: Count of methods calling clique non-significant
            - is_disagreement: True for all rows (DataFrame is filtered to disagreements)

        DataFrame is sorted by n_methods_significant descending (cliques where
        most methods agree on significance appear first).

    Example:
        >>> from cliquefinder.stats.method_comparison import identify_disagreements
        >>> disagreements = identify_disagreements(results_by_method, threshold=0.05)
        >>> print(f"Found {len(disagreements)} disagreement cases")
        >>> # Cliques significant in some but not all methods
        >>> for _, row in disagreements.head(5).iterrows():
        ...     print(f"{row['clique_id']}: {row['n_methods_significant']} sig, "
        ...           f"{row['n_methods_nonsignificant']} non-sig")

    Notes:
        - Only valid results (is_valid=True) are included
        - Cliques with NaN p-values in some methods are still included;
          those methods are counted as neither significant nor non-significant
        - Empty DataFrame returned if no disagreements found
    """
    import pandas as pd

    # Collect all unique clique IDs from all methods (valid results only)
    all_ids: set[str] = set()
    for results in results_by_method.values():
        all_ids.update(r.clique_id for r in results if r.is_valid)

    if not all_ids:
        # No valid results from any method
        return pd.DataFrame()

    # Build wide-format table: one row per clique
    rows: list[dict[str, object]] = []

    for cid in sorted(all_ids):
        row: dict[str, object] = {"clique_id": cid}
        sig_count = 0
        nonsig_count = 0

        for method, results in results_by_method.items():
            # Find result for this clique from this method
            result = next(
                (r for r in results if r.clique_id == cid and r.is_valid),
                None
            )

            method_key = method.value  # e.g., "ols", "roast_msq"

            if result is not None:
                row[f"{method_key}_pvalue"] = result.p_value
                row[f"{method_key}_effect"] = result.effect_size

                if result.p_value < threshold:
                    sig_count += 1
                else:
                    nonsig_count += 1
            else:
                # Method didn't produce a valid result for this clique
                row[f"{method_key}_pvalue"] = np.nan
                row[f"{method_key}_effect"] = np.nan

        row["n_methods_significant"] = sig_count
        row["n_methods_nonsignificant"] = nonsig_count

        # A disagreement occurs when some methods call it significant
        # and some call it non-significant
        row["is_disagreement"] = (sig_count > 0) and (nonsig_count > 0)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Filter to only disagreement cases
    disagreements = df[df["is_disagreement"]].copy()

    # Sort by number of methods calling significant (descending)
    # Cliques that are significant in more methods are of higher interest
    disagreements = disagreements.sort_values(
        "n_methods_significant", ascending=False
    )

    # Reset index for clean output
    disagreements = disagreements.reset_index(drop=True)

    return disagreements


# =============================================================================
# MethodComparisonResult Dataclass
# =============================================================================


@dataclass
class MethodComparisonResult:
    """
    Complete comparison results across all differential testing methods.

    This is the main output of run_method_comparison(). It aggregates results
    from multiple statistical methods (OLS, LMM, ROAST, Permutation) and provides
    multiple views into the comparison for different analyses.

    This dataclass is NOT frozen because it contains mutable dict fields.
    However, it should be treated as read-only after creation.

    Design Principles:
        1. Raw results preserved for custom analysis
        2. Pre-computed concordance metrics for efficiency
        3. Helper methods for common queries (robust hits, method-specific hits)
        4. Full provenance for reproducibility

    Statistical Note:
        This framework is for DESCRIPTIVE comparison, not inference.
        - Do NOT select the "best" p-value per clique (would inflate FDR)
        - Do NOT combine p-values across methods (requires strong assumptions)
        - DO use concordance to validate findings (high agreement = robust)
        - DO investigate disagreements for biological insights

    Attributes:
        results_by_method: Raw results from each method, keyed by MethodName.
            Access individual method results with results_by_method[MethodName.OLS].
        pairwise_concordance: List of ConcordanceMetrics for all method pairs.
            Use concordance_matrix() for a convenient matrix view.
        mean_spearman_rho: Average Spearman correlation of p-value ranks across all pairs.
            Interpretation: >0.8 excellent, 0.6-0.8 good, <0.5 poor agreement.
        mean_cohen_kappa: Average Cohen's kappa (classification agreement) across all pairs.
            Interpretation: >0.6 substantial, 0.4-0.6 moderate, <0.2 slight agreement.
        disagreement_cases: DataFrame of cliques where methods disagree on significance.
            Useful for investigating method-specific sensitivities.
        preprocessing_params: Dict capturing normalization, imputation, etc.
            for full reproducibility.
        methods_run: List of MethodName values that were successfully executed.
        n_cliques_tested: Total number of cliques tested by at least one method.

    Example:
        >>> comparison = run_method_comparison(...)
        >>> print(comparison.summary())
        >>>
        >>> # Get robust discoveries (significant in all methods)
        >>> robust = comparison.robust_hits(threshold=0.01)
        >>> print(f"Robust hits: {len(robust)}")
        >>>
        >>> # Get ROAST-specific hits (bidirectional regulation?)
        >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
        >>>
        >>> # Export to CSV for downstream analysis
        >>> wide_df = comparison.wide_format()
        >>> wide_df.to_csv("method_comparison_results.csv")

    See Also:
        - run_method_comparison: Main entry point that creates this object
        - ConcordanceMetrics: Detailed pairwise agreement metrics
        - UnifiedCliqueResult: Individual method results
    """

    # Raw results from each method
    results_by_method: dict[MethodName, list[UnifiedCliqueResult]]

    # Pairwise concordance metrics
    pairwise_concordance: list[ConcordanceMetrics]

    # Aggregate statistics
    mean_spearman_rho: float
    mean_cohen_kappa: float

    # Disagreement analysis
    disagreement_cases: object  # pd.DataFrame - using object for typing flexibility

    # Provenance
    preprocessing_params: dict[str, object]
    methods_run: list[MethodName]
    n_cliques_tested: int

    def wide_format(self) -> "pd.DataFrame":
        """
        Pivot results to wide format: one row per clique.

        Creates a DataFrame with one row per clique and columns for each method's
        statistics, enabling easy comparison, filtering, and export.

        Columns returned:
            - clique_id: Unique clique identifier
            - n_proteins: Number of proteins in clique definition
            - n_proteins_found: Number of proteins found in expression data
            - {method}_pvalue: P-value from method (e.g., ols_pvalue, roast_msq_pvalue)
            - {method}_effect_size: Effect size from method
            - {method}_statistic: Test statistic from method

        Cliques not tested by some methods will have NaN for those method columns.

        Returns:
            DataFrame with one row per unique clique across all methods.
            Rows are sorted alphabetically by clique_id.

        Example:
            >>> wide = comparison.wide_format()
            >>> # Filter to significant in at least one method
            >>> pval_cols = [c for c in wide.columns if c.endswith('_pvalue')]
            >>> sig_any = (wide[pval_cols] < 0.05).any(axis=1)
            >>> significant = wide[sig_any]
            >>> significant.to_csv("significant_cliques.csv", index=False)
        """
        import pandas as pd

        # Collect all unique clique IDs across all methods
        all_ids: set[str] = set()
        for results in self.results_by_method.values():
            all_ids.update(r.clique_id for r in results)

        if not all_ids:
            # No results from any method
            return pd.DataFrame(columns=["clique_id", "n_proteins", "n_proteins_found"])

        rows: list[dict[str, object]] = []

        for cid in sorted(all_ids):
            row: dict[str, object] = {"clique_id": cid}
            n_proteins: int | None = None
            n_proteins_found: int | None = None

            for method, results in self.results_by_method.items():
                # Find result for this clique from this method
                result = next((r for r in results if r.clique_id == cid), None)
                prefix = method.value

                if result is not None:
                    row[f"{prefix}_pvalue"] = result.p_value
                    row[f"{prefix}_effect_size"] = result.effect_size
                    row[f"{prefix}_statistic"] = result.statistic_value
                    # Capture clique size from first method that has it
                    if n_proteins is None:
                        n_proteins = result.n_proteins
                        n_proteins_found = result.n_proteins_found
                else:
                    row[f"{prefix}_pvalue"] = np.nan
                    row[f"{prefix}_effect_size"] = np.nan
                    row[f"{prefix}_statistic"] = np.nan

            # Add clique size columns
            row["n_proteins"] = n_proteins
            row["n_proteins_found"] = n_proteins_found

            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Reorder columns for clarity: clique_id, n_proteins, n_proteins_found first
        base_cols = ["clique_id", "n_proteins", "n_proteins_found"]
        other_cols = sorted([c for c in df.columns if c not in base_cols])
        df = df[base_cols + other_cols]

        return df

    def concordance_matrix(self) -> "pd.DataFrame":
        """
        Create square matrix of pairwise Spearman correlations.

        Returns a symmetric matrix suitable for heatmap visualization where
        each cell (i,j) contains the Spearman rho between methods i and j.
        Diagonal values are 1.0 (self-correlation).

        Returns:
            DataFrame with method names (e.g., "ols", "roast_msq") as both
            index and columns. Values are Spearman rho in range [-1, 1].

        Example:
            >>> import seaborn as sns
            >>> import matplotlib.pyplot as plt
            >>> matrix = comparison.concordance_matrix()
            >>> fig, ax = plt.subplots(figsize=(8, 6))
            >>> sns.heatmap(
            ...     matrix, annot=True, fmt='.2f',
            ...     cmap='RdYlGn', vmin=-1, vmax=1, ax=ax
            ... )
            >>> ax.set_title("Method Concordance (Spearman rho)")
            >>> plt.tight_layout()
        """
        import pandas as pd

        methods = [m.value for m in self.methods_run]
        n = len(methods)

        if n == 0:
            return pd.DataFrame()

        # Initialize with identity matrix (diagonal = 1.0, perfect self-correlation)
        matrix = np.eye(n)

        # Fill in off-diagonal elements from pairwise concordance
        for conc in self.pairwise_concordance:
            try:
                i = methods.index(conc.method_a.value)
                j = methods.index(conc.method_b.value)
                matrix[i, j] = conc.spearman_rho
                matrix[j, i] = conc.spearman_rho  # Symmetric matrix
            except ValueError:
                # Method not in methods_run (shouldn't happen, but be defensive)
                continue

        return pd.DataFrame(matrix, index=methods, columns=methods)

    def robust_hits(self, threshold: float = 0.05) -> list[str]:
        """
        Get cliques significant in ALL methods.

        These are high-confidence discoveries that replicate across different
        statistical approaches. A clique is a "robust hit" only if its p-value
        is below the threshold in EVERY method that tested it.

        This is a conservative criterion that minimizes false positives at the
        cost of potentially missing some true positives.

        Args:
            threshold: P-value threshold for significance (default 0.05).
                Use stricter thresholds (0.01, 0.001) for higher confidence.

        Returns:
            List of clique_id strings for cliques significant in all methods.
            Returns empty list if no cliques meet the criterion or if no
            methods were run.

        Example:
            >>> # Check robust hits at multiple thresholds
            >>> for thresh in [0.05, 0.01, 0.001]:
            ...     robust = comparison.robust_hits(threshold=thresh)
            ...     print(f"p < {thresh}: {len(robust)} robust hits")
            p < 0.05: 42 robust hits
            p < 0.01: 18 robust hits
            p < 0.001: 5 robust hits
        """
        if not self.methods_run:
            return []

        wide = self.wide_format()
        pval_cols = [c for c in wide.columns if c.endswith("_pvalue")]

        if not pval_cols:
            return []

        # A clique is a robust hit if ALL p-values are below threshold
        # Note: This requires the clique to be tested by ALL methods
        # NaN values mean method didn't test it, so we use dropna behavior
        mask = (wide[pval_cols] < threshold).all(axis=1)

        return wide.loc[mask, "clique_id"].tolist()

    def method_specific_hits(
        self, method: MethodName, threshold: float = 0.05
    ) -> list[str]:
        """
        Get cliques significant ONLY in the specified method.

        These hits are detected by one method but not others, which may indicate:
            - Method-specific sensitivity (e.g., ROAST detecting bidirectional
              regulation that OLS misses due to signal cancellation)
            - Potential false positives in that method
            - Unique biological signal that other methods are insensitive to

        Investigating these cases helps understand method behavior and can
        reveal which method is most appropriate for specific biological questions.

        Args:
            method: MethodName to check for unique significance.
            threshold: P-value threshold for significance (default 0.05).

        Returns:
            List of clique_id strings that are:
                - Significant (p < threshold) in the specified method
                - NOT significant (p >= threshold OR NaN) in all other methods
            Returns empty list if the specified method wasn't run or has no results.

        Example:
            >>> # Find cliques only ROAST detects (bidirectional regulation?)
            >>> roast_only = comparison.method_specific_hits(MethodName.ROAST_MSQ)
            >>> print(f"ROAST-specific hits: {len(roast_only)}")
            >>>
            >>> # Compare method-specific counts
            >>> for m in comparison.methods_run:
            ...     specific = comparison.method_specific_hits(m)
            ...     print(f"{m.value}: {len(specific)} unique hits")
        """
        if method not in self.results_by_method:
            return []

        wide = self.wide_format()
        method_col = f"{method.value}_pvalue"

        if method_col not in wide.columns:
            return []

        other_cols = [
            c for c in wide.columns if c.endswith("_pvalue") and c != method_col
        ]

        # Significant in this method (p < threshold)
        sig_in_method = wide[method_col] < threshold

        # Not significant in ANY other method
        # NaN values are treated as "not significant" (method didn't test it)
        not_sig_elsewhere = True
        for col in other_cols:
            # p >= threshold OR p is NaN counts as "not significant"
            col_not_sig = (wide[col] >= threshold) | wide[col].isna()
            not_sig_elsewhere = not_sig_elsewhere & col_not_sig

        mask = sig_in_method & not_sig_elsewhere
        return wide.loc[mask, "clique_id"].tolist()

    def get_concordance(
        self, method_a: MethodName, method_b: MethodName
    ) -> ConcordanceMetrics | None:
        """
        Retrieve concordance metrics for a specific pair of methods.

        Looks up the pre-computed ConcordanceMetrics for the given method pair.
        Order of arguments doesn't matter (will find either direction).

        Args:
            method_a: First method in the pair.
            method_b: Second method in the pair.

        Returns:
            ConcordanceMetrics for the specified pair, containing Spearman rho,
            Cohen's kappa, effect correlation, etc.
            Returns None if the pair wasn't computed (e.g., method not run or
            insufficient common cliques).

        Example:
            >>> conc = comparison.get_concordance(MethodName.OLS, MethodName.ROAST_MSQ)
            >>> if conc:
            ...     print(f"OLS vs ROAST:")
            ...     print(f"  Spearman rho: {conc.spearman_rho:.3f}")
            ...     print(f"  Cohen's kappa: {conc.cohen_kappa:.3f}")
            ...     print(f"  Jaccard index: {conc.jaccard_index:.3f}")
        """
        for conc in self.pairwise_concordance:
            # Check both orderings since pairs are stored in one direction only
            if (conc.method_a == method_a and conc.method_b == method_b) or (
                conc.method_a == method_b and conc.method_b == method_a
            ):
                return conc
        return None

    def summary(self) -> str:
        """
        Generate human-readable summary of comparison results.

        Provides a multi-line text summary including:
            - Number of cliques tested
            - Methods run
            - Mean concordance metrics
            - Robust hit counts at multiple thresholds
            - Number of disagreement cases
            - Pairwise concordance breakdown

        Returns:
            Multi-line string suitable for printing, logging, or saving to file.

        Example:
            >>> print(comparison.summary())
            Method Comparison Results
            ========================================
            Cliques tested: 250
            Methods: ols, roast_msq, permutation_competitive

            Concordance Summary:
              Mean Spearman rho: 0.782
              Mean Cohen's kappa: 0.654

            Robust hits (significant in all methods):
              p < 0.05: 42
              p < 0.01: 18
              p < 0.001: 5

            Disagreement cases: 35

            Pairwise Concordance:
              ols vs roast_msq: rho=0.823, kappa=0.712
              ols vs permutation_competitive: rho=0.756, kappa=0.623
              roast_msq vs permutation_competitive: rho=0.768, kappa=0.627
        """
        lines = [
            "Method Comparison Results",
            "=" * 40,
            f"Cliques tested: {self.n_cliques_tested}",
            f"Methods: {', '.join(m.value for m in self.methods_run)}",
            "",
            "Concordance Summary:",
            f"  Mean Spearman rho: {self.mean_spearman_rho:.3f}",
            f"  Mean Cohen's kappa: {self.mean_cohen_kappa:.3f}",
            "",
            "Robust hits (significant in all methods):",
        ]

        # Show robust hits at standard thresholds
        for thresh in [0.05, 0.01, 0.001]:
            n_robust = len(self.robust_hits(thresh))
            lines.append(f"  p < {thresh}: {n_robust}")

        # Add disagreement count
        import pandas as pd

        if isinstance(self.disagreement_cases, pd.DataFrame):
            n_disagreements = len(self.disagreement_cases)
        else:
            n_disagreements = 0

        lines.extend([
            "",
            f"Disagreement cases: {n_disagreements}",
        ])

        # Add pairwise concordance details if available
        if self.pairwise_concordance:
            lines.extend([
                "",
                "Pairwise Concordance:",
            ])
            for conc in self.pairwise_concordance:
                lines.append(
                    f"  {conc.method_a.value} vs {conc.method_b.value}: "
                    f"rho={conc.spearman_rho:.3f}, kappa={conc.cohen_kappa:.3f}"
                )

        return "\n".join(lines)


__all__ = [
    "compute_pairwise_concordance",
    "identify_disagreements",
    "MethodComparisonResult",
]
