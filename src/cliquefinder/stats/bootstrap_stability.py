"""
Bootstrap stability for validation pipeline.

Resamples within each condition group (with replacement) and re-runs
protein-level differential analysis + competitive z-score extraction.
Reports the fraction of bootstrap resamples that produce a significant
enrichment z-score — a measure of result reliability.

This is a report annotation, NOT a verdict gate. Low stability (< 0.7)
flags sensitivity to sample composition without overriding the verdict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def run_bootstrap_stability(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_condition: NDArray | pd.Series,
    contrast: tuple[str, str],
    target_feature_ids: list[str],
    covariates_df: pd.DataFrame | None = None,
    n_bootstraps: int = 200,
    z_threshold: float = 1.5,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap stability analysis for enrichment z-score.

    Resamples within each condition group (with replacement), re-runs
    protein differential analysis, and extracts the competitive z-score.
    Stability = fraction of bootstraps where z >= z_threshold.

    Args:
        data: Expression matrix (n_features, n_samples), log2-transformed.
        feature_ids: Feature identifiers matching rows of data.
        sample_condition: Condition labels per sample.
        contrast: (test_condition, reference_condition).
        target_feature_ids: Feature IDs of network targets.
        covariates_df: Optional covariates DataFrame.
        n_bootstraps: Number of bootstrap resamples (default: 200).
        z_threshold: Z-score threshold for "significant" (default: 1.5).
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Dict with keys:
            stability: Fraction of bootstraps with z >= z_threshold.
            z_ci: (2.5th, 97.5th) percentile CI for z-scores.
            z_scores: Array of bootstrap z-scores.
            n_bootstraps: Number of valid bootstraps.
    """
    from .differential import run_protein_differential
    from .enrichment_z import compute_competitive_z

    sample_condition = np.asarray(sample_condition)
    target_set = set(target_feature_ids)
    target_list = [fid for fid in feature_ids if fid in target_set]

    rng = np.random.default_rng(seed)

    # Identify sample indices per condition group
    groups = {}
    for cond in contrast:
        groups[cond] = np.where(sample_condition == cond)[0]

    if verbose:
        sizes = {c: len(idx) for c, idx in groups.items()}
        print(f"Bootstrap stability: {n_bootstraps} resamples, z_threshold={z_threshold}")
        print(f"  Group sizes: {sizes}")

    z_scores = np.full(n_bootstraps, np.nan)
    n_failures = 0

    for i in range(n_bootstraps):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Bootstrap {i + 1}/{n_bootstraps}...")

        # Resample within each group (with replacement)
        boot_indices = []
        for cond in contrast:
            group_idx = groups[cond]
            boot_idx = rng.choice(group_idx, size=len(group_idx), replace=True)
            boot_indices.append(boot_idx)
        boot_indices = np.concatenate(boot_indices)

        boot_data = data[:, boot_indices]
        boot_labels = sample_condition[boot_indices]
        boot_cov = None
        if covariates_df is not None:
            boot_cov = covariates_df.iloc[boot_indices].reset_index(drop=True)

        try:
            results = run_protein_differential(
                data=boot_data,
                feature_ids=feature_ids,
                sample_condition=boot_labels,
                contrast=contrast,
                eb_moderation=True,
                target_genes=target_list,
                verbose=False,
                covariates_df=boot_cov,
            )

            valid = results.dropna(subset=["t_statistic"])
            all_t = valid["t_statistic"].values.astype(np.float64)
            is_target = valid["is_target"].values.astype(bool)
            z_scores[i] = compute_competitive_z(all_t, is_target)
        except Exception as e:
            n_failures += 1
            if n_failures == 1:
                warnings.warn(f"Bootstrap {i} failed: {type(e).__name__}: {e}")
            z_scores[i] = np.nan

    valid_z = z_scores[~np.isnan(z_scores)]
    n_valid = len(valid_z)

    if n_valid == 0:
        return {
            "stability": 0.0,
            "z_ci": (float("nan"), float("nan")),
            "z_scores": np.array([]),
            "n_bootstraps": 0,
        }

    stability = float(np.sum(valid_z >= z_threshold)) / n_valid
    z_ci = (
        float(np.percentile(valid_z, 2.5)),
        float(np.percentile(valid_z, 97.5)),
    )

    if verbose:
        print(f"  Valid bootstraps: {n_valid}/{n_bootstraps}")
        print(f"  Stability (z >= {z_threshold}): {stability:.3f}")
        print(f"  Z 95% CI: [{z_ci[0]:.2f}, {z_ci[1]:.2f}]")

    return {
        "stability": stability,
        "z_ci": z_ci,
        "z_scores": valid_z,
        "n_bootstraps": n_valid,
    }
