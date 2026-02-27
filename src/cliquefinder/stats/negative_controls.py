"""
Negative control gene sets for false positive rate calibration.

Runs ROAST on random gene sets of the same size as the target set
to estimate the false positive rate (FPR) and calibrate the significance
of the target gene set result.

The key insight is that we reuse the already-fitted ROAST engine (QR
decomposition computed once). Only the gene set membership changes per
iteration, making this efficient.

Outputs:
    - FPR: fraction of control sets with p < alpha
    - Target percentile: where the target p-value falls in the
      control distribution
    - Median/mean control p-values
    - Competitive z-score (when protein_results provided): cross-phase
      consistent metric matching Phases 1/3/4

Warning convention:
    warnings.warn() -- user-facing (convergence, deprecated, sample size)
    logger.warning() -- operator-facing (fallback, retry, missing data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class NegativeControlResult:
    """Result of negative control gene set analysis.

    Attributes:
        target_pvalue: p-value of the actual target gene set.
        target_set_id: Identifier for the target gene set.
        target_set_size: Number of genes in the target set.
        control_pvalues: Array of p-values from random control sets.
        fpr: False positive rate (fraction of controls with p < alpha).
        alpha: Significance threshold used for FPR.
        target_percentile: Percentile rank of target p-value among controls
            (0 = most significant, 100 = least).
        median_control_pvalue: Median p-value across control sets.
        mean_control_pvalue: Mean p-value across control sets.
        n_control_sets: Number of control sets tested.
        target_competitive_z: Competitive z-score of the target set
            (consistent with Phases 1/3/4). None if protein_results
            not provided.
        control_competitive_z_scores: Competitive z-scores for each
            control set. None if protein_results not provided.
        competitive_z_fpr: Fraction of controls with competitive z >=
            target z. None if protein_results not provided.
        competitive_z_percentile: Percentile rank of target competitive z
            among controls (100 = most enriched). None if not provided.
    """

    target_pvalue: float
    target_set_id: str
    target_set_size: int
    control_pvalues: NDArray[np.float64]
    fpr: float
    alpha: float
    target_percentile: float
    median_control_pvalue: float
    mean_control_pvalue: float
    n_control_sets: int

    # Individual control set significance tracking
    n_significant_controls: int = 0  # How many control sets had p < alpha
    n_valid_controls: int = 0  # How many control sets were successfully tested

    # Competitive z-score metrics (consistent with Phases 1/3/4)
    target_competitive_z: float | None = None
    control_competitive_z_scores: NDArray[np.float64] | None = None
    competitive_z_fpr: float | None = None
    competitive_z_percentile: float | None = None

    # Expression-matched control metrics (M-4)
    matched_control_pvalues: NDArray[np.float64] | None = None
    matched_fpr: float | None = None
    matched_target_percentile: float | None = None
    matched_competitive_z_percentile: float | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "target_set_id": self.target_set_id,
            "target_set_size": self.target_set_size,
            "target_pvalue": self.target_pvalue,
            "fpr": self.fpr,
            "alpha": self.alpha,
            "target_percentile": self.target_percentile,
            "median_control_pvalue": self.median_control_pvalue,
            "mean_control_pvalue": self.mean_control_pvalue,
            "n_control_sets": self.n_control_sets,
            "n_significant_controls": self.n_significant_controls,
            "n_valid_controls": self.n_valid_controls,
            "control_pvalue_quantiles": {
                "q05": float(np.percentile(self.control_pvalues, 5)),
                "q25": float(np.percentile(self.control_pvalues, 25)),
                "q50": float(np.percentile(self.control_pvalues, 50)),
                "q75": float(np.percentile(self.control_pvalues, 75)),
                "q95": float(np.percentile(self.control_pvalues, 95)),
            },
        }

        if self.target_competitive_z is not None:
            d["competitive_z"] = {
                "target_z": self.target_competitive_z,
                "fpr": self.competitive_z_fpr,
                "percentile": self.competitive_z_percentile,
                "control_z_quantiles": {
                    "q05": float(np.percentile(self.control_competitive_z_scores, 5)),
                    "q25": float(np.percentile(self.control_competitive_z_scores, 25)),
                    "q50": float(np.percentile(self.control_competitive_z_scores, 50)),
                    "q75": float(np.percentile(self.control_competitive_z_scores, 75)),
                    "q95": float(np.percentile(self.control_competitive_z_scores, 95)),
                },
            }

        if self.matched_control_pvalues is not None and len(self.matched_control_pvalues) > 0:
            d["matched_controls"] = {
                "matched_fpr": self.matched_fpr,
                "matched_target_percentile": self.matched_target_percentile,
                "matched_competitive_z_percentile": self.matched_competitive_z_percentile,
                "matched_pvalue_quantiles": {
                    "q05": float(np.percentile(self.matched_control_pvalues, 5)),
                    "q25": float(np.percentile(self.matched_control_pvalues, 25)),
                    "q50": float(np.percentile(self.matched_control_pvalues, 50)),
                    "q75": float(np.percentile(self.matched_control_pvalues, 75)),
                    "q95": float(np.percentile(self.matched_control_pvalues, 95)),
                },
            }

        return d


def _sample_expression_matched_set(
    target_indices: list[int],
    non_target_indices: NDArray[np.intp],
    gene_means: NDArray[np.float64],
    gene_variances: NDArray[np.float64],
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample a control gene set matched on mean expression and variance.

    Uses bipartite matching (Hungarian algorithm) to find the set of
    non-target genes that minimizes total distance to target genes in
    (mean, variance) space.

    Small random noise is added to the cost matrix for diversity across
    repeated calls.

    Args:
        target_indices: Row indices of target genes in the data matrix.
        non_target_indices: Row indices of non-target genes.
        gene_means: Mean expression per gene (n_features,).
        gene_variances: Variance per gene (n_features,).
        rng: NumPy random generator.

    Returns:
        List of matched non-target indices (same length as target_indices).
    """
    from scipy.optimize import linear_sum_assignment

    n_targets = len(target_indices)
    n_pool = len(non_target_indices)

    if n_pool < n_targets:
        # Not enough non-targets; fall back to random sample
        return rng.choice(non_target_indices, size=n_targets, replace=False).tolist()

    # Build cost matrix: |mean_i - mean_j| + 0.5 * |var_i - var_j|
    target_means = gene_means[target_indices]
    target_vars = gene_variances[target_indices]
    pool_means = gene_means[non_target_indices]
    pool_vars = gene_variances[non_target_indices]

    # Shape: (n_targets, n_pool)
    cost = (
        np.abs(target_means[:, None] - pool_means[None, :])
        + 0.5 * np.abs(target_vars[:, None] - pool_vars[None, :])
    )

    # Add small noise for diversity across repeated calls
    cost *= 1.0 + rng.uniform(-0.1, 0.1, size=cost.shape)

    # Hungarian algorithm: O(n^2 * m)
    row_idx, col_idx = linear_sum_assignment(cost)

    return [int(non_target_indices[c]) for c in col_idx]


def run_negative_control_sets(
    engine,  # RotationTestEngine — already fitted
    target_gene_ids: list[str],
    target_set_id: str,
    n_control_sets: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    protein_results: pd.DataFrame | None = None,
    data: NDArray[np.float64] | None = None,
    matching: str = "uniform",
    verbose: bool = True,
) -> NegativeControlResult:
    """
    Run ROAST on random gene sets to calibrate false positive rate.

    Reuses the already-fitted ROAST engine so that the expensive QR
    decomposition is computed only once. Each control set has the same
    size as the target set, with genes sampled uniformly from the
    measured gene universe.

    Args:
        engine: RotationTestEngine that has already been fitted (fit() called).
            Must have gene_ids attribute listing all measured genes.
        target_gene_ids: Gene IDs in the target set (e.g., INDRA targets).
        target_set_id: Identifier for the target set (e.g., "C9ORF72_targets").
        n_control_sets: Number of random control sets to test (default: 200).
        alpha: Significance threshold for FPR calculation (default: 0.05).
        seed: Random seed for reproducibility.
        protein_results: DataFrame from run_protein_differential() with
            columns 't_statistic' and 'is_target'. When provided, computes
            competitive z-scores alongside ROAST p-values for cross-phase
            consistency with Phases 1/3/4.
        data: Expression matrix (n_features, n_samples). When provided with
            matching="expression_matched" or "both", samples control gene
            sets matched on mean expression and variance using bipartite
            matching (Hungarian algorithm).
        matching: Sampling strategy for control gene sets. Options:
            - "uniform" (default): Random uniform sampling.
            - "expression_matched": Matched on mean/variance.
            - "both": Run both uniform and expression-matched.
        verbose: Print progress.

    Returns:
        NegativeControlResult with FPR and calibration statistics.

    Raises:
        RuntimeError: If engine is not fitted.
    """
    if not engine._fitted:
        raise RuntimeError(
            "RotationTestEngine must be fitted before running negative controls. "
            "Call engine.fit() first."
        )

    all_gene_ids = list(engine.gene_ids)
    n_total = len(all_gene_ids)

    # Find target genes in the measured universe
    target_in_data = [g for g in target_gene_ids if g in engine.gene_to_idx]
    target_size = len(target_in_data)

    if target_size == 0:
        raise ValueError("No target genes found in the measured gene universe")

    if target_size >= n_total:
        raise ValueError("Target set covers all measured genes — cannot sample controls")

    if verbose:
        print(f"Negative control analysis: {n_control_sets} random sets of size {target_size}")

    rng = np.random.default_rng(seed)

    # Run ROAST on actual target set
    target_result = engine.test_gene_set(
        gene_set=target_in_data,
        gene_set_id=target_set_id,
    )
    # test_gene_set returns a RotationResult dataclass; extract p-value
    target_pvalue = float(target_result.p_values.get("msq", {}).get("mixed", 1.0))

    if verbose:
        print(f"  Target set p-value: {target_pvalue:.4f}")

    # Pre-generate all control gene sets (stored for competitive z reuse)
    control_gene_sets: list[list[str]] = []
    for _ in range(n_control_sets):
        genes = rng.choice(all_gene_ids, size=target_size, replace=False)
        control_gene_sets.append(genes.tolist())

    # Run ROAST on each control set
    control_pvalues = np.full(n_control_sets, np.nan)

    for i, control_genes in enumerate(control_gene_sets):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Control set {i + 1}/{n_control_sets}...")

        try:
            control_result = engine.test_gene_set(
                gene_set=control_genes,
                gene_set_id=f"control_{i}",
            )
            control_pvalues[i] = float(
                control_result.p_values.get("msq", {}).get("mixed", 1.0)
            )
        except Exception as e:
            if i == 0:
                logger.warning("Control set 0 failed: %s", e)
            control_pvalues[i] = np.nan

    # Remove failed controls
    valid_controls = control_pvalues[~np.isnan(control_pvalues)]
    n_valid = len(valid_controls)

    if n_valid == 0:
        raise RuntimeError("All control sets failed")

    # Compute counts of significant and valid controls
    n_significant = int(np.sum(valid_controls < alpha))

    # Compute FPR: fraction of controls with p < alpha
    fpr = float(n_significant) / n_valid

    # Target percentile: where does target p-value rank among controls?
    # Lower percentile = more significant than controls
    target_percentile = float(np.sum(valid_controls <= target_pvalue)) / n_valid * 100

    median_control = float(np.median(valid_controls))
    mean_control = float(np.mean(valid_controls))

    if verbose:
        print(f"  FPR (p < {alpha}): {fpr:.3f}")
        print(f"  Target percentile: {target_percentile:.1f}%")
        print(f"  Median control p-value: {median_control:.4f}")

    # --- Competitive z-score (cross-phase consistency) ---
    target_comp_z = None
    control_comp_z = None
    comp_z_fpr = None
    comp_z_percentile = None

    if protein_results is not None:
        from .enrichment_z import compute_competitive_z

        valid_pr = protein_results.dropna(subset=["t_statistic"])
        all_t = valid_pr["t_statistic"].values.astype(np.float64)
        pr_feature_ids = valid_pr["feature_id"].values
        feature_to_idx = {fid: i for i, fid in enumerate(pr_feature_ids)}
        n_features = len(all_t)

        # Target competitive z
        target_mask = np.zeros(n_features, dtype=bool)
        for g in target_in_data:
            if g in feature_to_idx:
                target_mask[feature_to_idx[g]] = True
        target_comp_z = compute_competitive_z(all_t, target_mask)

        # Control competitive z-scores
        control_comp_z = np.full(n_control_sets, np.nan)
        for i, control_genes in enumerate(control_gene_sets):
            ctrl_mask = np.zeros(n_features, dtype=bool)
            for g in control_genes:
                if g in feature_to_idx:
                    ctrl_mask[feature_to_idx[g]] = True
            if np.sum(ctrl_mask) > 0:
                control_comp_z[i] = compute_competitive_z(all_t, ctrl_mask)

        valid_comp_z = control_comp_z[~np.isnan(control_comp_z)]
        if len(valid_comp_z) > 0:
            # Higher z = more enriched, so FPR is fraction >= target
            comp_z_fpr = float(np.sum(valid_comp_z >= target_comp_z)) / len(valid_comp_z)
            comp_z_percentile = (
                float(np.sum(valid_comp_z >= target_comp_z)) / len(valid_comp_z) * 100
            )
            control_comp_z = valid_comp_z

        if verbose:
            print(f"  Target competitive z: {target_comp_z:.3f}")
            print(f"  Competitive z FPR (z >= target): {comp_z_fpr:.3f}")
            print(f"  Competitive z percentile: {comp_z_percentile:.1f}%")

    # --- Expression-matched controls (M-4) ---
    matched_pvalues = None
    matched_fpr_val = None
    matched_percentile = None
    matched_comp_z_percentile = None

    if matching in ("expression_matched", "both") and data is not None:
        if verbose:
            print(f"  Running expression-matched controls ({n_control_sets} sets)...")

        gene_means = np.nanmean(data, axis=1)
        gene_variances = np.nanvar(data, axis=1, ddof=1)

        target_idx_set = set()
        for g in target_in_data:
            if g in engine.gene_to_idx:
                target_idx_set.add(engine.gene_to_idx[g])
        target_indices = sorted(target_idx_set)
        non_target_indices = np.array(
            [i for i in range(n_total) if i not in target_idx_set],
            dtype=np.intp,
        )

        matched_pvalues_arr = np.full(n_control_sets, np.nan)

        for i in range(n_control_sets):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Matched control set {i + 1}/{n_control_sets}...")

            matched_indices = _sample_expression_matched_set(
                target_indices, non_target_indices,
                gene_means, gene_variances, rng,
            )
            matched_genes = [all_gene_ids[idx] for idx in matched_indices]

            try:
                matched_result = engine.test_gene_set(
                    gene_set=matched_genes,
                    gene_set_id=f"matched_control_{i}",
                )
                matched_pvalues_arr[i] = float(
                    matched_result.p_values.get("msq", {}).get("mixed", 1.0)
                )
            except Exception:
                matched_pvalues_arr[i] = np.nan

        valid_matched = matched_pvalues_arr[~np.isnan(matched_pvalues_arr)]
        n_valid_matched = len(valid_matched)

        if n_valid_matched > 0:
            matched_pvalues = valid_matched
            matched_fpr_val = float(np.sum(valid_matched < alpha)) / n_valid_matched
            matched_percentile = (
                float(np.sum(valid_matched <= target_pvalue)) / n_valid_matched * 100
            )

            # Matched competitive z percentile (if protein_results available)
            if protein_results is not None and target_comp_z is not None:
                matched_comp_z = np.full(n_control_sets, np.nan)
                for i in range(n_control_sets):
                    matched_indices = _sample_expression_matched_set(
                        target_indices, non_target_indices,
                        gene_means, gene_variances, rng,
                    )
                    matched_genes = [all_gene_ids[idx] for idx in matched_indices]
                    ctrl_mask = np.zeros(n_features, dtype=bool)
                    for g in matched_genes:
                        if g in feature_to_idx:
                            ctrl_mask[feature_to_idx[g]] = True
                    if np.sum(ctrl_mask) > 0:
                        matched_comp_z[i] = compute_competitive_z(all_t, ctrl_mask)

                valid_matched_z = matched_comp_z[~np.isnan(matched_comp_z)]
                if len(valid_matched_z) > 0:
                    matched_comp_z_percentile = (
                        float(np.sum(valid_matched_z >= target_comp_z))
                        / len(valid_matched_z) * 100
                    )

            if verbose:
                print(f"  Matched FPR (p < {alpha}): {matched_fpr_val:.3f}")
                print(f"  Matched target percentile: {matched_percentile:.1f}%")

    return NegativeControlResult(
        target_pvalue=target_pvalue,
        target_set_id=target_set_id,
        target_set_size=target_size,
        control_pvalues=valid_controls,
        fpr=fpr,
        alpha=alpha,
        target_percentile=target_percentile,
        median_control_pvalue=median_control,
        mean_control_pvalue=mean_control,
        n_control_sets=n_valid,
        n_significant_controls=n_significant,
        n_valid_controls=n_valid,
        target_competitive_z=target_comp_z,
        control_competitive_z_scores=control_comp_z,
        competitive_z_fpr=comp_z_fpr,
        competitive_z_percentile=comp_z_percentile,
        matched_control_pvalues=matched_pvalues,
        matched_fpr=matched_fpr_val,
        matched_target_percentile=matched_percentile,
        matched_competitive_z_percentile=matched_comp_z_percentile,
    )
