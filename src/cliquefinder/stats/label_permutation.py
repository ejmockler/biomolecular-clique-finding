"""
Label permutation null for network enrichment validation.

Tests the self-contained significance of network enrichment by permuting
condition labels and re-running the full pipeline (protein differential →
enrichment test). Both stratified (within-stratum) and free permutation
modes are supported.

Stratified permutation preserves covariate balance within strata (e.g.,
permuting labels within Male and Female separately), preventing spurious
effects from covariate imbalance. Free permutation breaks this structure,
providing a complementary test.

Per-permutation loop:
    1. Permute condition labels (stratified or free)
    2. run_protein_differential() with permuted labels + covariates
    3. run_network_enrichment_test() on permuted results
    4. Collect null z-score

The observed z-score is compared to the null distribution to compute
a permutation p-value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _extract_enrichment_z(protein_results: pd.DataFrame) -> float:
    """
    Extract enrichment z-score directly from protein results.

    This is a lightweight alternative to run_network_enrichment_test() that
    skips the 10,000 inner competitive permutations. Instead, it computes
    the mean|t| for targets vs background and returns a simple z-score.

    This is appropriate for the label permutation loop because the *outer*
    permutation already generates the null distribution — we don't need
    the inner competitive null per permutation.

    Delegates to compute_competitive_z() for the core computation.
    """
    from .enrichment_z import compute_competitive_z

    valid = protein_results.dropna(subset=["t_statistic"])
    all_t = valid["t_statistic"].values.astype(np.float64)
    is_target = valid["is_target"].values.astype(bool)
    return compute_competitive_z(all_t, is_target)


@dataclass
class LabelPermutationResult:
    """Result of label permutation null analysis.

    Attributes:
        observed_z: The observed enrichment z-score from real labels.
        null_z_scores: Array of z-scores from permuted labels.
        permutation_pvalue: Fraction of null z-scores >= observed.
        n_permutations: Number of permutations performed.
        stratified: Whether permutation was stratified.
        stratify_column: Column used for stratification (if any).
        null_mean: Mean of null z-score distribution.
        null_std: Std of null z-score distribution.
    """

    observed_z: float
    null_z_scores: NDArray[np.float64]
    permutation_pvalue: float
    n_permutations: int
    stratified: bool
    stratify_column: str | None = None
    null_mean: float = 0.0
    null_std: float = 1.0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "observed_z": self.observed_z,
            "permutation_pvalue": self.permutation_pvalue,
            "n_permutations": self.n_permutations,
            "stratified": self.stratified,
            "stratify_column": self.stratify_column,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_z_quantiles": {
                "q05": float(np.percentile(self.null_z_scores, 5)),
                "q25": float(np.percentile(self.null_z_scores, 25)),
                "q50": float(np.percentile(self.null_z_scores, 50)),
                "q75": float(np.percentile(self.null_z_scores, 75)),
                "q95": float(np.percentile(self.null_z_scores, 95)),
            },
        }


def generate_stratified_permutation(
    labels: NDArray,
    strata: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    """
    Permute labels within each stratum independently.

    This preserves the label distribution within each stratum,
    preventing confounding from covariate imbalance.

    Args:
        labels: Condition labels (n_samples,).
        strata: Stratum assignments (n_samples,). Same length as labels.
        rng: NumPy random generator.

    Returns:
        Permuted labels array with within-stratum permutation.
    """
    permuted = labels.copy()
    for stratum in np.unique(strata):
        mask = strata == stratum
        indices = np.where(mask)[0]
        permuted[indices] = rng.permutation(labels[indices])
    return permuted


def generate_free_permutation(
    labels: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    """
    Permute labels freely (no stratification).

    Args:
        labels: Condition labels (n_samples,).
        rng: NumPy random generator.

    Returns:
        Permuted labels array.
    """
    return rng.permutation(labels)


def run_label_permutation_null(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_condition: NDArray | pd.Series,
    contrast: tuple[str, str],
    target_gene_ids: list[str],
    n_permutations: int = 1000,
    stratify_by: NDArray | pd.Series | None = None,
    covariates_df: pd.DataFrame | None = None,
    covariate_design: "CovariateDesign | None" = None,
    eb_moderation: bool = True,
    seed: int | None = None,
    verbose: bool = True,
) -> LabelPermutationResult:
    """
    Permutation null for network enrichment.

    Permutes condition labels (stratified or free), re-runs protein-level
    differential analysis and enrichment test, and compares the observed
    z-score against the null distribution.

    Args:
        data: Expression matrix (n_features, n_samples), log2-transformed.
        feature_ids: Feature identifiers matching rows of data.
        sample_condition: Condition labels per sample.
        contrast: (test_condition, reference_condition).
        target_gene_ids: Feature IDs of network targets (INDRA gene set).
        n_permutations: Number of label permutations (default: 1000).
        stratify_by: Stratum assignments for stratified permutation (e.g.,
            Sex column values). If None, free permutation is used.
        covariates_df: Optional covariates DataFrame passed through to
            protein differential analysis.
        covariate_design: Optional pre-built CovariateDesign (M-6 NaN mask
            consolidation). When provided, its sample_mask is used as the
            authoritative NaN mask in run_protein_differential() for both
            the observed and all permuted analyses. The covariates themselves
            do not change across label permutations (only condition labels
            are permuted), so the same design applies throughout.
        eb_moderation: Whether to use Empirical Bayes moderation.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        LabelPermutationResult with observed vs null z-scores.
    """
    from .differential import run_protein_differential

    sample_condition = np.asarray(sample_condition)
    target_set = set(target_gene_ids)
    stratified = stratify_by is not None

    if stratify_by is not None:
        stratify_by = np.asarray(stratify_by)
        if len(stratify_by) != len(sample_condition):
            raise ValueError(
                f"stratify_by length ({len(stratify_by)}) != "
                f"sample_condition length ({len(sample_condition)})"
            )

    rng = np.random.default_rng(seed)

    # --- Step 1: Observed enrichment z-score ---
    # Pre-compute target list once (avoid rebuilding per permutation)
    target_genes_list = [fid for fid in feature_ids if fid in target_set]

    if verbose:
        strat_label = getattr(stratify_by, 'name', 'covariate') if stratified else None
        mode_str = f"stratified by {strat_label}" if stratified else "free"
        print(f"Label permutation null ({mode_str}): {n_permutations} permutations")

    observed_results = run_protein_differential(
        data=data,
        feature_ids=feature_ids,
        sample_condition=sample_condition,
        contrast=contrast,
        eb_moderation=eb_moderation,
        target_gene_ids=target_genes_list,
        verbose=False,
        covariates_df=covariates_df,
        covariate_design=covariate_design,
    )

    # Use lightweight z-score extraction (no 10k inner competitive perms)
    observed_z = _extract_enrichment_z(observed_results)

    if verbose:
        print(f"  Observed z-score: {observed_z:.3f}")

    # --- Step 2: Null distribution ---
    null_z_scores = np.full(n_permutations, np.nan)
    n_failures = 0

    for i in range(n_permutations):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}...")

        # Permute labels
        if stratified:
            perm_labels = generate_stratified_permutation(
                sample_condition, stratify_by, rng
            )
        else:
            perm_labels = generate_free_permutation(sample_condition, rng)

        # Run protein differential with permuted labels, extract z directly
        try:
            perm_results = run_protein_differential(
                data=data,
                feature_ids=feature_ids,
                sample_condition=perm_labels,
                contrast=contrast,
                eb_moderation=eb_moderation,
                target_gene_ids=target_genes_list,
                verbose=False,
                covariates_df=covariates_df,
                covariate_design=covariate_design,
            )

            null_z_scores[i] = _extract_enrichment_z(perm_results)
        except Exception as e:
            # Log first failure for debugging
            n_failures += 1
            if n_failures == 1:
                warnings.warn(
                    f"Label permutation {i} failed: {type(e).__name__}: {e}"
                )
            null_z_scores[i] = np.nan

    # Remove failed permutations
    valid_null = null_z_scores[~np.isnan(null_z_scores)]
    n_valid = len(valid_null)

    if n_valid == 0:
        raise RuntimeError("All permutations failed — cannot compute null distribution")

    # Permutation p-value: fraction of null >= observed
    perm_pvalue = float(np.sum(valid_null >= observed_z) + 1) / (n_valid + 1)

    null_mean = float(np.mean(valid_null))
    null_std = float(np.std(valid_null, ddof=1)) if n_valid > 1 else 1.0

    if verbose:
        print(f"  Valid permutations: {n_valid}/{n_permutations}")
        print(f"  Null z distribution: mean={null_mean:.3f}, std={null_std:.3f}")
        print(f"  Permutation p-value: {perm_pvalue:.4f}")

    return LabelPermutationResult(
        observed_z=observed_z,
        null_z_scores=valid_null,
        permutation_pvalue=perm_pvalue,
        n_permutations=n_valid,
        stratified=stratified,
        stratify_column=None,  # Set by caller if needed
        null_mean=null_mean,
        null_std=null_std,
    )
