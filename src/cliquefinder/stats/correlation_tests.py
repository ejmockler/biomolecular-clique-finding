"""
Statistical tests for correlation-based regulatory analysis.

Provides:
- Fisher's Z-test for differential correlations
- FDR correction across multiple comparisons
- Confidence intervals via Fisher Z-transformation
- Sample-size-adaptive significance thresholds
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class CorrelationTestResult:
    """Result of a correlation difference test."""
    r1: float  # Correlation in condition 1
    r2: float  # Correlation in condition 2
    n1: int    # Sample size condition 1
    n2: int    # Sample size condition 2
    z_score: float
    p_value: float
    ci_r1: Tuple[float, float]  # 95% CI for r1
    ci_r2: Tuple[float, float]  # 95% CI for r2


def fisher_z_transform(r: float) -> float:
    """
    Fisher's Z-transformation: arctanh(r).

    Converts correlation coefficient to a normally distributed variable,
    enabling parametric hypothesis testing.

    Args:
        r: Pearson correlation coefficient

    Returns:
        Fisher Z-transformed value
    """
    # Clip to avoid infinity at |r|=1
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """
    Inverse Fisher Z-transformation: tanh(z).

    Converts Fisher Z back to correlation scale.

    Args:
        z: Fisher Z-transformed value

    Returns:
        Correlation coefficient
    """
    return np.tanh(z)


def fisher_z_standard_error(n: int, method: str = 'pearson') -> float:
    """
    Compute standard error of Fisher Z-transformed correlation.

    For Pearson correlation, SE = 1/sqrt(n-3).
    For Spearman correlation, SE = 1.06/sqrt(n-3) to account for
    rank transformation variance inflation.

    The Fisher Z transformation SE formula requires n >= 4 because it uses
    n-3 degrees of freedom in the denominator. This accounts for the estimation
    of the mean, variance, and correlation coefficient itself from the data.

    Args:
        n: Sample size (must be >= 4)
        method: 'pearson', 'spearman', or 'max'. The 'max' method uses the
            conservative Spearman SE since it selects the stronger correlation.

    Returns:
        Standard error of Fisher Z

    Raises:
        ValueError: If n < 4, as the Fisher Z standard error formula
            requires at least 4 samples (uses n-3 degrees of freedom)

    References:
        Fisher, R.A. (1915). Frequency distribution of the values of the
        correlation coefficient in samples from an indefinitely large population.
        Biometrika, 10(4), 507-521.

        Fisher, R.A. (1921). On the probable error of a coefficient of
        correlation deduced from a small sample. Metron, 1, 3-32.

        Fieller, E.C., Hartley, H.O., & Pearson, E.S. (1957). Tests for rank
        correlation coefficients. I. Biometrika, 44(3/4), 470-481.

        Bonett, D.G., & Wright, T.A. (2000). Sample size requirements for
        estimating Pearson, Kendall and Spearman correlations.
        Psychometrika, 65(1), 23-28.
    """
    # Validate sample size requirement for Fisher Z SE formula
    if not isinstance(n, (int, np.integer)):
        raise ValueError(
            f"Sample size n must be an integer, got {type(n).__name__}: {n}"
        )

    if n < 4:
        raise ValueError(
            f"Fisher Z standard error requires n >= 4 (got n={n}). "
            f"The formula SE = 1/sqrt(n-3) uses n-3 degrees of freedom, "
            f"which becomes undefined (n=3), infinite (n=3), or imaginary (n<3) "
            f"for small samples. Minimum 4 samples are required for valid "
            f"correlation standard error estimation."
        )

    base_se = 1.0 / np.sqrt(n - 3)

    if method == 'spearman':
        # Spearman's rho has inflated variance due to rank transformation
        # Factor 1.06 is the asymptotic correction (1 + 6/(n+1) approaches 1.06)
        return 1.06 * base_se
    elif method == 'pearson':
        return base_se
    elif method == 'max':
        # "max" selects the stronger of Pearson or Spearman correlations.
        # Since we don't know which was selected at this point, use the
        # conservative Spearman SE (1.06 factor). This is slightly conservative
        # if Pearson was chosen, but exactly correct if Spearman was chosen.
        return 1.06 * base_se
    else:
        raise ValueError(f"Unknown correlation method '{method}'. Use 'pearson', 'spearman', or 'max'")


def correlation_confidence_interval(
    r: float,
    n: int,
    alpha: float = 0.05,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Compute confidence interval for correlation using Fisher Z.

    The Fisher Z-transformation yields approximate normality, enabling
    construction of symmetric confidence intervals that are then
    back-transformed to the correlation scale.

    Args:
        r: Observed correlation
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)
        method: 'pearson', 'spearman', or 'max'. Spearman and max use SE
            correction factor of 1.06 due to rank transformation variance inflation.

    Returns:
        (lower, upper) bounds of CI

    References:
        Fisher, R.A. (1915). Frequency distribution of the values of the
        correlation coefficient in samples from an indefinitely large population.
        Biometrika, 10(4), 507-521.

        Fieller, E.C., Hartley, H.O., & Pearson, E.S. (1957). Tests for rank
        correlation coefficients. I. Biometrika, 44(3/4), 470-481.
    """
    z = fisher_z_transform(r)
    se_z = fisher_z_standard_error(n, method=method)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se_z
    z_upper = z + z_crit * se_z

    return (inverse_fisher_z(z_lower), inverse_fisher_z(z_upper))


def test_correlation_difference(
    r1: float, n1: int,
    r2: float, n2: int,
    method: str = 'pearson'
) -> CorrelationTestResult:
    """
    Test if two correlations differ significantly (Fisher's Z-test).

    Used for testing CASE vs CTRL differential co-expression.
    Assumes independent samples from bivariate normal distributions.

    H0: ρ1 = ρ2 (population correlations are equal)
    HA: ρ1 ≠ ρ2 (differential correlation)

    Args:
        r1, n1: Correlation and sample size for condition 1
        r2, n2: Correlation and sample size for condition 2
        method: 'pearson', 'spearman', or 'max'. Spearman and max use SE
            correction factor of 1.06 due to rank transformation variance inflation.

    Returns:
        CorrelationTestResult with z-score, p-value, and CIs

    References:
        Cohen, J., Cohen, P., West, S.G., & Aiken, L.S. (2003).
        Applied Multiple Regression/Correlation Analysis for the
        Behavioral Sciences (3rd ed.). Routledge.

        Fieller, E.C., Hartley, H.O., & Pearson, E.S. (1957). Tests for rank
        correlation coefficients. I. Biometrika, 44(3/4), 470-481.
    """
    z1 = fisher_z_transform(r1)
    z2 = fisher_z_transform(r2)

    # Get SE for each sample, accounting for correlation method
    se1 = fisher_z_standard_error(n1, method=method)
    se2 = fisher_z_standard_error(n2, method=method)

    # Combined SE for difference of independent Z-transformed correlations
    se = np.sqrt(se1**2 + se2**2)
    z_score = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    ci1 = correlation_confidence_interval(r1, n1, method=method)
    ci2 = correlation_confidence_interval(r2, n2, method=method)

    return CorrelationTestResult(
        r1=r1, r2=r2, n1=n1, n2=n2,
        z_score=z_score, p_value=p_value,
        ci_r1=ci1, ci_r2=ci2
    )


def compute_significance_threshold(
    n_samples: int,
    alpha: float = 0.05,
    n_tests: int = 1
) -> float:
    """
    Compute correlation threshold for significance at given alpha.

    Adjusts for sample size (smaller n needs higher r for significance).
    Optionally applies Bonferroni correction for multiple tests.

    Based on the t-distribution of r under H0: ρ = 0:
        t = r * sqrt(n-2) / sqrt(1-r^2)  ~  t(n-2)

    Args:
        n_samples: Number of samples
        alpha: Significance level
        n_tests: Number of tests for Bonferroni correction

    Returns:
        Minimum |r| for significance

    Example:
        >>> # For n=30 samples at α=0.05
        >>> r_crit = compute_significance_threshold(30, 0.05)
        >>> # Correlations with |r| > r_crit are significant
    """
    alpha_adj = alpha / n_tests  # Bonferroni
    t_crit = stats.t.ppf(1 - alpha_adj/2, df=n_samples - 2)
    r_crit = t_crit / np.sqrt(n_samples - 2 + t_crit**2)
    return r_crit


def estimate_effective_tests(
    correlation_matrix: Optional[np.ndarray] = None,
    p_values: Optional[np.ndarray] = None,
    method: str = 'nyholt'
) -> float:
    """
    Estimate the effective number of independent tests from correlated data.

    When testing many correlated gene pairs, the nominal test count M overstates
    the multiple testing burden. This function estimates M_eff, the equivalent
    number of independent tests, using eigenvalue decomposition of the
    correlation structure.

    This is critical for proper FDR correction: Benjamini-Hochberg assumes
    independence or positive dependence. Reporting M_eff alongside nominal M
    provides transparency about the true multiple testing burden.

    Args:
        correlation_matrix: Correlation matrix of test statistics (shape: M×M).
            If not provided, will be computed from p-values assuming
            they're derived from correlated normal variates.
        p_values: Array of p-values. Used to determine M if correlation_matrix
            not provided. At least one of correlation_matrix or p_values required.
        method: Method for estimation:
            - 'nyholt': M_eff = sum(λ) / max(λ) where λ are eigenvalues
                More conservative, handles strong correlation
            - 'li-ji': M_eff = 1 + (M-1) * (1 - Var(λ)/M)
                Simpler, better for weak correlation

    Returns:
        M_eff: Estimated effective number of independent tests

    Raises:
        ValueError: If neither correlation_matrix nor p_values provided,
            or if inputs are invalid

    Notes:
        - For independent tests: M_eff ≈ M (nominal test count)
        - For perfectly correlated tests: M_eff ≈ 1
        - Typical gene expression: M_eff ≈ 0.3-0.6 * M due to co-regulation

    References:
        Nyholt, D.R. (2004). A simple correction for multiple testing for SNPs
        in linkage disequilibrium with each other. American Journal of Human
        Genetics, 74(4), 765-769.

        Li, J., & Ji, L. (2005). Adjusting multiple testing in multilocus
        analyses using the eigenvalues of a correlation matrix. Heredity,
        95(3), 221-227.

    Example:
        >>> # From correlation matrix
        >>> corr_mat = np.corrcoef(gene_expression_data.T)
        >>> m_eff = estimate_effective_tests(correlation_matrix=corr_mat)
        >>> print(f"Nominal: {corr_mat.shape[0]}, Effective: {m_eff:.1f}")
        >>>
        >>> # Report in results
        >>> print(f"Tested {M} gene pairs (effective tests: ~{m_eff:.0f})")
    """
    if correlation_matrix is None and p_values is None:
        raise ValueError("Must provide either correlation_matrix or p_values")

    # Determine nominal test count M
    if correlation_matrix is not None:
        if correlation_matrix.ndim != 2:
            raise ValueError(f"correlation_matrix must be 2D, got shape {correlation_matrix.shape}")
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError(f"correlation_matrix must be square, got shape {correlation_matrix.shape}")
        M = correlation_matrix.shape[0]
    else:
        M = len(p_values)
        # For p-values without correlation matrix, conservatively return M
        # (cannot estimate correlation structure from p-values alone)
        import warnings
        warnings.warn(
            "Cannot estimate M_eff from p-values alone without correlation matrix. "
            "Returning nominal M. Provide correlation_matrix for accurate estimation.",
            UserWarning
        )
        return float(M)

    # Edge cases
    if M <= 1:
        return float(M)

    # Compute eigenvalues of correlation matrix
    # Use only positive eigenvalues (numerical stability)
    try:
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter numerical noise
    except np.linalg.LinAlgError as e:
        import warnings
        warnings.warn(
            f"Failed to compute eigenvalues: {e}. Returning nominal M={M}",
            UserWarning
        )
        return float(M)

    if len(eigenvalues) == 0:
        return 1.0

    # Apply selected method
    if method == 'nyholt':
        # Nyholt (2004): M_eff = λ_sum / λ_max
        # More conservative for highly correlated data
        lambda_sum = np.sum(eigenvalues)
        lambda_max = np.max(eigenvalues)
        m_eff = lambda_sum / lambda_max

    elif method == 'li-ji':
        # Li & Ji (2005): M_eff = 1 + (M-1) * (1 - Var(λ)/M)
        # Simpler, works well for moderate correlation
        lambda_var = np.var(eigenvalues)
        m_eff = 1 + (len(eigenvalues) - 1) * (1 - lambda_var / len(eigenvalues))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nyholt' or 'li-ji'")

    # Clamp to reasonable range [1, M]
    m_eff = np.clip(m_eff, 1.0, float(M))

    return float(m_eff)


def permutation_fdr(
    observed_p_values: np.ndarray,
    null_p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute FDR using permutation-based null distribution.

    This is the RECOMMENDED method for dependent tests (e.g., correlated gene pairs).
    Unlike BH/BY which make assumptions about dependence structure, permutation FDR
    empirically estimates the null distribution from label-shuffled data.

    The method:
    1. For each observed p-value threshold t, count:
       - R(t): number of observed p-values ≤ t (rejections)
       - V(t): number of null p-values ≤ t (expected false positives)
    2. FDR(t) = V(t) / R(t)
    3. q-value for test i = min FDR(t) over all t ≥ p_i

    This correctly handles:
    - Arbitrary dependence structures (no PRDS assumption)
    - Multiple testing on correlated data (gene expression)
    - Non-uniform null distributions

    Args:
        observed_p_values: Array of p-values from actual data
        null_p_values: Array of p-values from permuted data (pooled across permutations)
        alpha: FDR threshold for significance (default 0.05)

    Returns:
        (q_values, significant_mask, stats) where:
            - q_values: permutation-adjusted p-values
            - significant_mask: boolean array of tests passing FDR threshold
            - stats: dict with 'n_permutation_tests', 'estimated_pi0', 'method'

    References:
        Storey, J.D., & Tibshirani, R. (2003). Statistical significance for
        genomewide studies. PNAS, 100(16), 9440-9445.

        Tusher, V.G., Tibshirani, R., & Chu, G. (2001). Significance analysis
        of microarrays applied to the ionizing radiation response.
        PNAS, 98(9), 5116-5121.

    Notes:
        - Requires that null_p_values come from proper permutation procedure
          (phenotype label shuffling, maintaining expression correlation structure)
        - More permutations = more stable FDR estimates (recommend ≥1000)
        - pi0 estimation uses Storey's method when estimating proportion of true nulls

    Example:
        >>> # Observed p-values from differential correlation analysis
        >>> observed = compute_differential_correlations(data, labels)
        >>>
        >>> # Generate null by permuting labels B times
        >>> null_pvals = []
        >>> for _ in range(1000):
        ...     shuffled = np.random.permutation(labels)
        ...     null_pvals.extend(compute_differential_correlations(data, shuffled))
        >>> null_pvals = np.array(null_pvals)
        >>>
        >>> # Compute permutation FDR
        >>> q_vals, sig, stats = permutation_fdr(observed, null_pvals, alpha=0.05)
    """
    observed_p = np.asarray(observed_p_values)
    null_p = np.asarray(null_p_values)

    if len(observed_p) == 0:
        return np.array([]), np.array([], dtype=bool), {
            'n_permutation_tests': 0,
            'estimated_pi0': 1.0,
            'method': 'permutation'
        }

    m = len(observed_p)  # Number of observed tests
    n_null = len(null_p)  # Number of null tests (m * n_permutations typically)

    if n_null == 0:
        # No null distribution available - fall back to BH
        import warnings
        warnings.warn(
            "No null p-values provided for permutation FDR. "
            "Falling back to Benjamini-Hochberg.",
            UserWarning
        )
        from scipy.stats import false_discovery_control
        q_values = false_discovery_control(observed_p, method='bh')
        return q_values, q_values < alpha, {
            'n_permutation_tests': 0,
            'estimated_pi0': None,
            'method': 'bh_fallback'
        }

    # Estimate pi0 (proportion of true null hypotheses) using Storey's method
    # This improves power by accounting for true positives
    lambda_vals = np.arange(0.05, 0.95, 0.05)
    pi0_estimates = []
    for lam in lambda_vals:
        # W(λ) = #{p_i > λ}
        w_lambda = np.sum(observed_p > lam)
        # pi0(λ) = W(λ) / (m * (1 - λ))
        pi0_lambda = w_lambda / (m * (1 - lam))
        pi0_estimates.append(min(pi0_lambda, 1.0))

    # Use smoothing spline to estimate pi0 (simplified: take median of stable estimates)
    # In practice, use the estimate at λ = 0.5 (common choice)
    pi0_est = min(1.0, max(0.0, np.median(pi0_estimates)))

    # Compute q-values using step-up procedure
    # Sort observed p-values
    sorted_indices = np.argsort(observed_p)
    sorted_p = observed_p[sorted_indices]

    q_values = np.zeros(m)

    # For each observed p-value, compute FDR
    for i, p_thresh in enumerate(sorted_p):
        # R(t): rejections at this threshold (cumulative rank)
        r_t = i + 1

        # V(t): expected false positives = (# null p-values ≤ t) * m / n_null
        v_t = np.sum(null_p <= p_thresh) * m / n_null

        # FDR estimate, adjusted by pi0
        if r_t > 0:
            fdr_t = pi0_est * v_t / r_t
        else:
            fdr_t = 0.0

        q_values[sorted_indices[i]] = min(fdr_t, 1.0)

    # Enforce monotonicity: q_i = min(q_j) for j >= i (in sorted order)
    # Process from largest to smallest p-value
    reverse_indices = np.argsort(observed_p)[::-1]
    min_q_so_far = 1.0
    for idx in reverse_indices:
        if q_values[idx] < min_q_so_far:
            min_q_so_far = q_values[idx]
        else:
            q_values[idx] = min_q_so_far

    significant = q_values < alpha

    stats = {
        'n_permutation_tests': n_null,
        'n_permutations': n_null // m if m > 0 else 0,
        'estimated_pi0': float(pi0_est),
        'method': 'permutation'
    }

    return q_values, significant, stats


def apply_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh',
    correlation_matrix: Optional[np.ndarray] = None,
    report_effective_tests: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Apply FDR correction to p-values with optional effective test reporting.

    Controls the False Discovery Rate (expected proportion of false positives
    among discoveries) at level alpha. More powerful than Bonferroni when
    testing many hypotheses.

    Args:
        p_values: Array of raw p-values
        alpha: FDR threshold (e.g., 0.05 = 5% FDR)
        method: 'bh' for Benjamini-Hochberg (assumes independence or
                positive dependence), 'by' for Benjamini-Yekutieli
                (valid under arbitrary dependence)
        correlation_matrix: Optional M×M correlation matrix for estimating
            effective number of independent tests. Recommended for correlated
            gene expression data.
        report_effective_tests: If True, compute and return M_eff statistics

    Returns:
        (q_values, significant_mask, m_eff_stats) where:
            - q_values: adjusted p-values
            - significant_mask: boolean array of tests passing FDR threshold
            - m_eff_stats: dict with 'nominal_tests' and 'effective_tests'
                (None if report_effective_tests=False)

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society: Series B, 57(1), 289-300.

        Benjamini, Y., & Yekutieli, D. (2001). The control of the false
        discovery rate in multiple testing under dependency.
        Annals of Statistics, 29(4), 1165-1188.

    Example:
        >>> p_vals = np.array([0.001, 0.04, 0.03, 0.5, 0.2])
        >>> q_vals, sig, stats = apply_fdr_correction(
        ...     p_vals, alpha=0.05, report_effective_tests=True
        ... )
        >>> if stats:
        ...     print(f"Tested {stats['nominal_tests']}, effective ~{stats['effective_tests']:.0f}")
    """
    from scipy.stats import false_discovery_control

    # Handle edge cases
    p_values = np.asarray(p_values)
    if len(p_values) == 0:
        empty_stats = {'nominal_tests': 0, 'effective_tests': 0.0} if report_effective_tests else None
        return np.array([]), np.array([], dtype=bool), empty_stats

    # Compute q-values
    q_values = false_discovery_control(p_values, method=method)
    significant = q_values < alpha

    # Optionally compute effective tests
    m_eff_stats = None
    if report_effective_tests:
        m_nominal = len(p_values)

        if correlation_matrix is not None:
            m_effective = estimate_effective_tests(
                correlation_matrix=correlation_matrix,
                method='nyholt'
            )
        else:
            # No correlation matrix - report nominal
            m_effective = float(m_nominal)

        m_eff_stats = {
            'nominal_tests': m_nominal,
            'effective_tests': m_effective,
            'reduction_factor': m_effective / m_nominal if m_nominal > 0 else 1.0
        }

    return q_values, significant, m_eff_stats
