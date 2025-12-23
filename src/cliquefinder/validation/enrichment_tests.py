"""
Statistical enrichment testing with multiple testing correction.

This module provides rigorous statistical tests for determining if a gene set
is enriched for specific biological annotations beyond random chance.

Biological Context:
    Enrichment testing answers: "Are imputed genes enriched for specific pathways/functions?"

    If outliers are technical artifacts → imputed genes should be random
    If outliers are biology → imputed genes may cluster in functional pathways

Statistical Methods:
    Hypergeometric Test (recommended):
        - Tests: "Given N genes total, M in pathway, drew n genes, got k in pathway"
        - Null: Random sampling (no enrichment)
        - One-sided test (enrichment, not depletion)
        - Exact p-values (no approximation)

    Fisher's Exact Test:
        - Alternative formulation (2x2 contingency table)
        - Equivalent to hypergeometric for one-sided test
        - Useful when you need odds ratios

Multiple Testing Correction:
    - Problem: Testing 10,000 pathways → expect 500 false positives at p<0.05!
    - Solution: FDR correction (Benjamini-Hochberg)
    - Controls false discovery rate (fraction of rejected nulls that are false)
    - Less conservative than Bonferroni (better power)

Examples:
    >>> from cliquefinder.validation.enrichment_tests import HypergeometricTest
    >>> from cliquefinder.validation.enrichment_tests import apply_fdr_correction
    >>>
    >>> # Test single pathway
    >>> test = HypergeometricTest()
    >>> result = test.test_enrichment(
    ...     study_genes={'GENE1', 'GENE2', 'GENE3'},
    ...     pathway_genes={'GENE1', 'GENE2', 'GENE10'},
    ...     background_genes=all_genes
    ... )
    >>> print(f"p-value: {result.pvalue:.4f}")
    >>> print(f"Enrichment ratio: {result.enrichment_ratio:.2f}")
    >>>
    >>> # Test multiple pathways with FDR correction
    >>> results = []
    >>> for pathway_id, pathway_genes in pathways.items():
    ...     result = test.test_enrichment(study_genes, pathway_genes, background)
    ...     results.append((pathway_id, result))
    >>>
    >>> # Apply FDR correction
    >>> pvalues = [r.pvalue for _, r in results]
    >>> qvalues = apply_fdr_correction(pvalues, method='fdr_bh')
    >>> for (pathway_id, result), qvalue in zip(results, qvalues):
    ...     if qvalue < 0.05:
    ...         print(f"{pathway_id}: FDR={qvalue:.4f}")
"""

from __future__ import annotations

from typing import List, Tuple, Set, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import hypergeom, fisher_exact
from statsmodels.stats.multitest import multipletests

__all__ = [
    'EnrichmentResult',
    'EnrichmentTest',
    'HypergeometricTest',
    'FisherExactTest',
    'apply_fdr_correction',
]


@dataclass
class EnrichmentResult:
    """
    Results from enrichment test.

    Attributes:
        pvalue: Raw p-value (probability of observing enrichment by chance)
        enrichment_ratio: Fold enrichment vs random expectation
        study_count: Number of study genes in pathway
        pathway_size: Total genes in pathway
        background_size: Total genes in background
        study_size: Total study genes
        confidence_interval: 95% CI for enrichment ratio (lower, upper)
        odds_ratio: Odds ratio (for Fisher's test)

    Interpretation:
        - pvalue < 0.05: Significant enrichment (before multiple testing)
        - enrichment_ratio > 1: More overlap than expected by chance
        - enrichment_ratio < 1: Less overlap (depletion)

    Examples:
        >>> result = test.test_enrichment(study, pathway, background)
        >>> if result.pvalue < 0.05:
        ...     print(f"Enrichment: {result.enrichment_ratio:.2f}x")
        ...     print(f"Found {result.study_count}/{result.pathway_size} pathway genes")
    """
    pvalue: float
    enrichment_ratio: float
    study_count: int
    pathway_size: int
    background_size: int
    study_size: int
    confidence_interval: Tuple[float, float]
    odds_ratio: float = 1.0


class EnrichmentTest:
    """
    Base class for enrichment tests.

    All enrichment tests should inherit from this class and implement
    the test_enrichment method.
    """

    def test_enrichment(
        self,
        study_genes: Set[str],
        pathway_genes: Set[str],
        background_genes: Set[str]
    ) -> EnrichmentResult:
        """
        Test if study genes are enriched for pathway.

        Args:
            study_genes: Genes of interest (e.g., imputed genes)
            pathway_genes: Genes annotated to pathway/term
            background_genes: All genes that could have been selected
                (e.g., all genes on array/sequencing platform)

        Returns:
            EnrichmentResult with p-value and effect sizes

        Raises:
            ValueError: If sets have invalid relationships
                (e.g., study genes not subset of background)

        Examples:
            >>> # Test if imputed genes enriched for apoptosis
            >>> test = HypergeometricTest()
            >>> result = test.test_enrichment(
            ...     study_genes=imputed_gene_set,
            ...     pathway_genes=apoptosis_genes,
            ...     background_genes=all_measured_genes
            ... )
            >>> if result.pvalue < 0.05:
            ...     print(f"Enriched {result.enrichment_ratio:.2f}x for apoptosis")
        """
        raise NotImplementedError


class HypergeometricTest(EnrichmentTest):
    """
    Hypergeometric test for gene set enrichment.

    Statistical Model:
        - Population of N genes (background)
        - M genes in pathway
        - Drew n genes (study set)
        - Found k genes in pathway (overlap)
        - Test: P(X >= k | N, M, n) where X ~ Hypergeometric(N, M, n)

    Assumptions:
        1. Sampling without replacement
        2. All genes equally likely to be selected (null hypothesis)
        3. Pathway membership independent of selection (null hypothesis)

    One-Sided Test:
        We test for enrichment (k >= observed), not depletion.
        Two-sided test would be appropriate for depletion analysis.

    Examples:
        >>> test = HypergeometricTest()
        >>>
        >>> # Example: 20,000 genes total, 500 in pathway
        >>> # Drew 100 genes, found 10 in pathway
        >>> result = test.test_enrichment(
        ...     study_genes=set([f'GENE{i}' for i in range(100)]),
        ...     pathway_genes=set([f'GENE{i}' for i in range(500)]),
        ...     background_genes=set([f'GENE{i}' for i in range(20000)])
        ... )
        >>> # Expected: 100 * 500/20000 = 2.5 genes
        >>> # Observed: 10 genes
        >>> # Enrichment: 10/2.5 = 4.0x
        >>> print(f"Enrichment: {result.enrichment_ratio:.2f}x")
        >>> print(f"p-value: {result.pvalue:.4e}")
    """

    def test_enrichment(
        self,
        study_genes: Set[str],
        pathway_genes: Set[str],
        background_genes: Set[str]
    ) -> EnrichmentResult:
        """Test enrichment using hypergeometric distribution."""

        # Validate inputs
        if not study_genes.issubset(background_genes):
            # Allow this - just use intersection
            study_genes = study_genes.intersection(background_genes)

        if not pathway_genes.issubset(background_genes):
            # Allow this - just use intersection
            pathway_genes = pathway_genes.intersection(background_genes)

        # Compute set sizes
        N = len(background_genes)  # Population size
        M = len(pathway_genes)     # Success states in population
        n = len(study_genes)       # Number of draws
        k = len(study_genes.intersection(pathway_genes))  # Observed successes

        # Handle edge cases
        if N == 0 or n == 0 or M == 0:
            return EnrichmentResult(
                pvalue=1.0,
                enrichment_ratio=0.0,
                study_count=0,
                pathway_size=M,
                background_size=N,
                study_size=n,
                confidence_interval=(0.0, 0.0),
                odds_ratio=0.0
            )

        # Compute expected count
        expected = n * M / N if N > 0 else 0

        # Compute enrichment ratio
        enrichment_ratio = k / expected if expected > 0 else 0.0

        # Hypergeometric test (one-sided, test for enrichment)
        # P(X >= k) = survival function at k-1
        pvalue = hypergeom.sf(k - 1, N, M, n)

        # Compute 95% confidence interval for enrichment ratio
        # Using Wilson score interval for proportions
        if expected > 0:
            ci_lower = max(0, (k - 1.96 * np.sqrt(k)) / expected)
            ci_upper = (k + 1.96 * np.sqrt(k)) / expected
        else:
            ci_lower = 0.0
            ci_upper = 0.0

        # Compute odds ratio (for comparison with Fisher's test)
        # Odds of being in pathway if in study vs if not in study
        a = k  # In study and pathway
        b = n - k  # In study, not in pathway
        c = M - k  # Not in study, in pathway
        d = (N - M) - (n - k)  # Not in study, not in pathway

        if b > 0 and c > 0:
            odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        else:
            odds_ratio = float('inf') if a > 0 else 0.0

        return EnrichmentResult(
            pvalue=float(pvalue),
            enrichment_ratio=float(enrichment_ratio),
            study_count=k,
            pathway_size=M,
            background_size=N,
            study_size=n,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            odds_ratio=float(odds_ratio)
        )


class FisherExactTest(EnrichmentTest):
    """
    Fisher's exact test for gene set enrichment.

    Formulates enrichment as 2x2 contingency table:
                    In Pathway    Not in Pathway
    In Study             a              b
    Not in Study         c              d

    Tests independence: Are study membership and pathway membership independent?

    Equivalent to hypergeometric test for one-sided test, but provides odds ratio.

    Examples:
        >>> test = FisherExactTest()
        >>> result = test.test_enrichment(study, pathway, background)
        >>> print(f"Odds ratio: {result.odds_ratio:.2f}")
        >>> print(f"p-value: {result.pvalue:.4e}")
    """

    def test_enrichment(
        self,
        study_genes: Set[str],
        pathway_genes: Set[str],
        background_genes: Set[str]
    ) -> EnrichmentResult:
        """Test enrichment using Fisher's exact test."""

        # Validate inputs
        if not study_genes.issubset(background_genes):
            study_genes = study_genes.intersection(background_genes)

        if not pathway_genes.issubset(background_genes):
            pathway_genes = pathway_genes.intersection(background_genes)

        # Construct 2x2 contingency table
        a = len(study_genes.intersection(pathway_genes))  # In study AND pathway
        b = len(study_genes - pathway_genes)  # In study, NOT in pathway
        c = len(pathway_genes - study_genes)  # In pathway, NOT in study
        d = len(background_genes - study_genes - pathway_genes)  # Neither

        # Fisher's exact test (one-sided, test for enrichment)
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='greater')

        # Compute enrichment ratio (same as hypergeometric)
        N = len(background_genes)
        M = len(pathway_genes)
        n = len(study_genes)
        expected = n * M / N if N > 0 else 0
        enrichment_ratio = a / expected if expected > 0 else 0.0

        # Confidence interval for odds ratio (approximate)
        # Using log(OR) ± 1.96 * SE(log(OR))
        if a > 0 and b > 0 and c > 0 and d > 0:
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
            ci_lower = np.exp(log_or - 1.96 * se_log_or)
            ci_upper = np.exp(log_or + 1.96 * se_log_or)
        else:
            ci_lower = 0.0
            ci_upper = float('inf')

        return EnrichmentResult(
            pvalue=float(pvalue),
            enrichment_ratio=float(enrichment_ratio),
            study_count=a,
            pathway_size=M,
            background_size=N,
            study_size=n,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            odds_ratio=float(odds_ratio)
        )


def apply_fdr_correction(
    pvalues: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.

    Args:
        pvalues: List of raw p-values
        method: Correction method
            - 'fdr_bh': Benjamini-Hochberg FDR (recommended, less conservative)
            - 'fdr_by': Benjamini-Yekutieli FDR (more conservative, dependent tests)
            - 'bonferroni': Bonferroni correction (very conservative)
            - 'holm': Holm-Bonferroni (less conservative than Bonferroni)
        alpha: Family-wise error rate or FDR threshold

    Returns:
        Tuple of (reject, qvalues):
            reject: Boolean array indicating which hypotheses to reject
            qvalues: Adjusted p-values (q-values for FDR, adjusted p-values for FWER)

    Statistical Notes:
        Benjamini-Hochberg (FDR):
            - Controls False Discovery Rate (fraction of rejected nulls that are false)
            - Less conservative than Bonferroni (better power)
            - Assumes independent or positive dependent tests
            - RECOMMENDED for enrichment testing

        Benjamini-Yekutieli (FDR):
            - More conservative FDR control
            - Works with arbitrary dependence structure
            - Use if tests are strongly dependent

        Bonferroni:
            - Controls Family-Wise Error Rate (probability of any false positive)
            - Very conservative (low power)
            - Use only if need strict FWER control

    Examples:
        >>> # Test 100 pathways, apply FDR correction
        >>> pvalues = [0.001, 0.05, 0.3, 0.02, 0.8, ...]
        >>> reject, qvalues = apply_fdr_correction(pvalues, method='fdr_bh')
        >>>
        >>> # Get significant pathways
        >>> significant_indices = np.where(reject)[0]
        >>> for idx in significant_indices:
        ...     print(f"Pathway {idx}: p={pvalues[idx]:.4f}, FDR={qvalues[idx]:.4f}")

        >>> # Compare different correction methods
        >>> for method in ['fdr_bh', 'fdr_by', 'bonferroni']:
        ...     reject, qvalues = apply_fdr_correction(pvalues, method=method)
        ...     print(f"{method}: {reject.sum()} significant")
    """
    if len(pvalues) == 0:
        return np.array([], dtype=bool), np.array([])

    # Convert to numpy array
    pvalues = np.array(pvalues)

    # Handle NaN/Inf
    if np.any(np.isnan(pvalues)) or np.any(np.isinf(pvalues)):
        raise ValueError("p-values contain NaN or Inf")

    # Handle out-of-range p-values
    if np.any(pvalues < 0) or np.any(pvalues > 1):
        raise ValueError("p-values must be in [0, 1]")

    # Apply correction
    reject, pvals_corrected, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method=method,
        returnsorted=False
    )

    return reject, pvals_corrected
