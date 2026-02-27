"""
Multi-contrast specificity analysis.

Compares enrichment results across multiple contrasts to determine
whether a gene set effect is specific to one comparison or shared
across conditions. For example, comparing C9ORF72 vs CTRL enrichment
against Sporadic vs CTRL enrichment determines whether the signal
is C9orf72-specific or reflects general ALS biology.

Specificity scoring:
    specificity_ratio = primary_z / max(secondary_z_scores)
    - ratio >> 1: effect specific to primary contrast
    - ratio ~ 1: effect shared across contrasts
    - ratio < 1: stronger in secondary contrast
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ContrastEnrichment:
    """Enrichment results for a single contrast.

    Attributes:
        contrast_name: Human-readable contrast label (e.g., "C9ORF72_vs_CTRL").
        z_score: Competitive enrichment z-score (mean |t| vs null).
        empirical_pvalue: One-sided empirical p-value.
        roast_pvalue: ROAST self-contained p-value (MSQ mixed), if available.
        n_targets: Number of target genes found in data.
        pct_down: Percentage of targets with negative t-statistic.
        direction_pvalue: Binomial test p-value for directional bias.
    """

    contrast_name: str
    z_score: float
    empirical_pvalue: float
    roast_pvalue: float | None = None
    n_targets: int = 0
    pct_down: float = 0.0
    direction_pvalue: float = 1.0


@dataclass
class SpecificityResult:
    """Result of multi-contrast specificity analysis.

    Attributes:
        primary_contrast: Name of the primary contrast being assessed.
        contrasts: Dict mapping contrast name → ContrastEnrichment.
        specificity_ratio: primary_z / max(secondary_z), or inf if
            all secondary z-scores are <= 0.
        specificity_label: "specific", "shared", or "inconclusive".
        summary: Human-readable summary of the specificity assessment.
        interaction_z: Observed z-score difference (primary - secondary).
            None if interaction test not run.
        interaction_pvalue: Empirical p-value for Δz from permutation null.
            None if interaction test not run.
        z_difference: Same as interaction_z (alias for clarity).
        z_difference_ci: 95% CI for Δz from null distribution.
            None if interaction test not run.
        null_correlation: Estimated correlation between null z-scores
            for primary and secondary contrasts from the permutation null.
            This reflects the shared-sample correlation structure: when
            two contrasts share a common reference group (e.g., both
            compare against CTRL), their null z-scores are positively
            correlated. The paired permutation approach already accounts
            for this correlation (labels are permuted once, preserving the
            shared-sample structure), so null_corr is diagnostic rather
            than a correction to apply. Values near 1.0 indicate strong
            shared-control coupling; values near 0.0 indicate independent
            contrasts. None if interaction test not run.

    Notes:
        Permutation resolution: With n_perms=200 (the default for the
        interaction test), the minimum achievable p-value is ~0.005.
        For publication-grade specificity claims, use n_perms >= 1000.
    """

    primary_contrast: str
    contrasts: dict[str, ContrastEnrichment] = field(default_factory=dict)
    specificity_ratio: float = float("inf")
    specificity_label: str = "inconclusive"
    summary: str = ""

    # Interaction z-test fields (M-2)
    interaction_z: float | None = None
    interaction_pvalue: float | None = None
    z_difference: float | None = None
    z_difference_ci: tuple[float, float] | None = None
    null_correlation: float | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "primary_contrast": self.primary_contrast,
            "specificity_ratio": self.specificity_ratio,
            "specificity_label": self.specificity_label,
            "summary": self.summary,
            "contrasts": {
                name: {
                    "z_score": c.z_score,
                    "empirical_pvalue": c.empirical_pvalue,
                    "roast_pvalue": c.roast_pvalue,
                    "n_targets": c.n_targets,
                    "pct_down": c.pct_down,
                    "direction_pvalue": c.direction_pvalue,
                }
                for name, c in self.contrasts.items()
            },
        }

        # Promote null_correlation to top level for visibility (STAT-6).
        # This diagnostic reflects shared-sample correlation structure
        # between contrasts (e.g., shared CTRL group).
        if self.null_correlation is not None:
            d["null_correlation"] = self.null_correlation

        if self.interaction_z is not None:
            d["interaction_test"] = {
                "z_difference": self.interaction_z,
                "interaction_pvalue": self.interaction_pvalue,
                "z_difference_ci": list(self.z_difference_ci) if self.z_difference_ci else None,
                "null_correlation": self.null_correlation,
            }

        return d


def _run_interaction_permutation(
    data: NDArray[np.float64],
    feature_ids: list[str],
    metadata: pd.DataFrame,
    condition_col: str,
    primary_contrast: tuple[str, str],
    secondary_contrast: tuple[str, str],
    target_gene_ids: list[str],
    covariates_df: pd.DataFrame | None = None,
    n_perms: int = 200,
    seed: int | None = None,
) -> dict:
    """
    Paired permutation test for z-score difference between contrasts.

    Permutes condition labels once across all samples (pooling all 3+
    groups), then subsets to each contrast's groups and computes the
    competitive z-score for each. Records Δz = z_primary - z_secondary.

    This accounts for the shared-control correlation (e.g., both
    C9ORF72 vs CTRL and Sporadic vs CTRL share the CTRL samples),
    which inflates apparent differences in unpaired comparisons.

    The returned ``null_correlation`` reflects this shared-sample
    correlation structure in the permutation null. It is diagnostic
    (not a correction to apply), since the paired permutation design
    already preserves the correlation by permuting labels once across
    all groups.

    Permutation resolution: With n_perms=200 (the default), the
    minimum achievable p-value is ~0.005. For publication-grade
    specificity claims, use n_perms >= 1000.

    Args:
        data: Expression matrix (n_features, n_samples), log2-transformed.
        feature_ids: Feature identifiers matching rows of data.
        metadata: Sample metadata DataFrame aligned with data columns.
        condition_col: Column in metadata with condition labels.
        primary_contrast: (test, reference) for primary comparison.
        secondary_contrast: (test, reference) for secondary comparison.
        target_gene_ids: Feature IDs of network targets.
        covariates_df: Optional covariates DataFrame.
        n_perms: Number of label permutations (default: 200).
        seed: Random seed for reproducibility.

    Returns:
        Dict with observed_dz, null_dz array, interaction_pvalue,
        z_difference_ci, null_correlation.
    """
    from .differential import run_protein_differential
    from .enrichment_z import compute_competitive_z

    conditions = metadata[condition_col].values
    all_labels = np.asarray(conditions)
    target_set = set(target_gene_ids)
    target_list = [fid for fid in feature_ids if fid in target_set]

    rng = np.random.default_rng(seed)

    def _compute_contrast_z(labels, contrast_tuple):
        """Subset to contrast groups, run differential, extract z."""
        mask = np.isin(labels, contrast_tuple)
        sub_data = data[:, mask]
        sub_labels = labels[mask]
        sub_cov = None
        if covariates_df is not None:
            sub_cov = covariates_df.iloc[mask]

        results = run_protein_differential(
            data=sub_data,
            feature_ids=feature_ids,
            sample_condition=sub_labels,
            contrast=contrast_tuple,
            eb_moderation=True,
            target_gene_ids=target_list,
            verbose=False,
            covariates_df=sub_cov,
        )

        valid = results.dropna(subset=["t_statistic"])
        all_t = valid["t_statistic"].values.astype(np.float64)
        is_target = valid["is_target"].values.astype(bool)
        return compute_competitive_z(all_t, is_target)

    # Observed Δz
    observed_z_primary = _compute_contrast_z(all_labels, primary_contrast)
    observed_z_secondary = _compute_contrast_z(all_labels, secondary_contrast)
    observed_dz = observed_z_primary - observed_z_secondary

    # Null distribution
    null_dz = np.full(n_perms, np.nan)
    null_z_primary = np.full(n_perms, np.nan)
    null_z_secondary = np.full(n_perms, np.nan)
    n_failures = 0

    for i in range(n_perms):
        perm_labels = rng.permutation(all_labels)
        try:
            zp = _compute_contrast_z(perm_labels, primary_contrast)
            zs = _compute_contrast_z(perm_labels, secondary_contrast)
            null_z_primary[i] = zp
            null_z_secondary[i] = zs
            null_dz[i] = zp - zs
        except Exception:
            n_failures += 1
            continue

    valid_null = null_dz[~np.isnan(null_dz)]
    n_valid = len(valid_null)

    if n_valid == 0:
        return {
            "observed_dz": observed_dz,
            "null_dz": np.array([]),
            "interaction_pvalue": 1.0,
            "z_difference_ci": (float("nan"), float("nan")),
            "null_correlation": 0.0,
        }

    # Two-sided p-value: fraction of null |Δz| >= observed |Δz|
    interaction_pvalue = float(
        np.sum(np.abs(valid_null) >= abs(observed_dz)) + 1
    ) / (n_valid + 1)

    # 95% CI from null distribution (2.5th and 97.5th percentiles)
    ci_low = float(np.percentile(valid_null, 2.5))
    ci_high = float(np.percentile(valid_null, 97.5))

    # Null correlation between primary and secondary z-scores
    valid_p = null_z_primary[~np.isnan(null_dz)]
    valid_s = null_z_secondary[~np.isnan(null_dz)]
    if len(valid_p) > 2:
        null_corr = float(np.corrcoef(valid_p, valid_s)[0, 1])
    else:
        null_corr = 0.0

    return {
        "observed_dz": observed_dz,
        "null_dz": valid_null,
        "interaction_pvalue": interaction_pvalue,
        "z_difference_ci": (ci_low, ci_high),
        "null_correlation": null_corr,
    }


def compute_specificity(
    enrichment_by_contrast: dict[str, dict],
    primary_contrast: str,
    roast_by_contrast: dict[str, pd.DataFrame] | None = None,
    z_threshold: float = 1.5,
    p_threshold: float = 0.05,
    # Interaction test parameters (M-2)
    data: NDArray[np.float64] | None = None,
    feature_ids: list[str] | None = None,
    metadata: pd.DataFrame | None = None,
    condition_col: str | None = None,
    contrast_tuples: dict[str, tuple[str, str]] | None = None,
    target_gene_ids: list[str] | None = None,
    covariates_df: pd.DataFrame | None = None,
    n_interaction_perms: int = 200,
    seed: int | None = None,
) -> SpecificityResult:
    """
    Compare enrichment across contrasts to assess specificity.

    Args:
        enrichment_by_contrast: Dict mapping contrast name → enrichment dict
            (as returned by run_network_enrichment_test). Must include
            'z_score', 'empirical_pvalue', 'n_targets', 'pct_down'.
        primary_contrast: Name of the primary contrast to assess specificity for.
        roast_by_contrast: Optional dict mapping contrast name → ROAST results
            DataFrame. If provided, extracts pvalue_msq_mixed for the first
            gene set row.
        z_threshold: Minimum z-score to consider an effect "present" (default: 1.5).
        p_threshold: Significance threshold for empirical p-value (default: 0.05).
        data: Expression matrix for interaction test (n_features, n_samples).
            When provided along with other data parameters, runs a paired
            permutation interaction z-test for each secondary contrast.
        feature_ids: Feature identifiers matching rows of data.
        metadata: Sample metadata DataFrame aligned with data columns.
        condition_col: Column in metadata with condition labels.
        contrast_tuples: Dict mapping contrast name → (test, reference) tuple.
        target_gene_ids: Feature IDs of network targets.
        covariates_df: Optional covariates DataFrame.
        n_interaction_perms: Permutations for interaction test (default: 200).
            With n_perms=200, the minimum achievable p-value is ~0.005.
            For publication-grade specificity claims, use n_perms >= 1000.
        seed: Random seed for reproducibility.

    Returns:
        SpecificityResult with ratio, label, and per-contrast details.
        When interaction test data is provided, includes interaction_z,
        interaction_pvalue, z_difference_ci, and null_correlation.
        The null_correlation is also available as a top-level key in
        ``to_dict()`` output for visibility.

    Raises:
        ValueError: If primary_contrast not found in enrichment_by_contrast.
    """
    if primary_contrast not in enrichment_by_contrast:
        raise ValueError(
            f"Primary contrast '{primary_contrast}' not found. "
            f"Available: {list(enrichment_by_contrast.keys())}"
        )

    # Build ContrastEnrichment for each contrast
    contrast_results: dict[str, ContrastEnrichment] = {}
    for name, enrich in enrichment_by_contrast.items():
        roast_p = None
        if roast_by_contrast is not None and name in roast_by_contrast:
            roast_df = roast_by_contrast[name]
            if len(roast_df) > 0 and "pvalue_msq_mixed" in roast_df.columns:
                roast_p = float(roast_df["pvalue_msq_mixed"].iloc[0])

        contrast_results[name] = ContrastEnrichment(
            contrast_name=name,
            z_score=enrich.get("z_score", 0.0),
            empirical_pvalue=enrich.get("empirical_pvalue", 1.0),
            roast_pvalue=roast_p,
            n_targets=enrich.get("n_targets", 0),
            pct_down=enrich.get("pct_down", 0.0),
            direction_pvalue=enrich.get("direction_pvalue", 1.0),
        )

    # Compute specificity ratio
    primary = contrast_results[primary_contrast]
    secondary_z = [
        c.z_score for name, c in contrast_results.items()
        if name != primary_contrast
    ]

    if not secondary_z:
        # Only one contrast — can't assess specificity
        return SpecificityResult(
            primary_contrast=primary_contrast,
            contrasts=contrast_results,
            specificity_ratio=float("inf"),
            specificity_label="inconclusive",
            summary="Only one contrast provided; specificity cannot be assessed.",
        )

    max_secondary_z = max(secondary_z)

    if max_secondary_z <= 0:
        ratio = float("inf") if primary.z_score > 0 else 0.0
    else:
        ratio = primary.z_score / max_secondary_z

    # --- Interaction test (M-2) ---
    # When data is provided, run paired permutation to test whether the
    # z-score *difference* is significant, accounting for shared-control
    # correlation (Gelman & Stern 2006 correction).
    interaction_z = None
    interaction_pvalue = None
    z_difference = None
    z_difference_ci = None
    null_correlation = None
    has_interaction = False

    if (
        data is not None
        and feature_ids is not None
        and metadata is not None
        and condition_col is not None
        and contrast_tuples is not None
        and target_gene_ids is not None
    ):
        primary_tuple = contrast_tuples.get(primary_contrast)
        secondary_names = [n for n in contrast_tuples if n != primary_contrast]

        if primary_tuple and secondary_names:
            # Run interaction test against the strongest secondary contrast
            # (the one most likely to produce a "shared" label)
            strongest_secondary = max(
                secondary_names,
                key=lambda n: contrast_results[n].z_score
                if n in contrast_results else -float("inf"),
            )
            secondary_tuple = contrast_tuples[strongest_secondary]

            interaction_result = _run_interaction_permutation(
                data=data,
                feature_ids=feature_ids,
                metadata=metadata,
                condition_col=condition_col,
                primary_contrast=primary_tuple,
                secondary_contrast=secondary_tuple,
                target_gene_ids=target_gene_ids,
                covariates_df=covariates_df,
                n_perms=n_interaction_perms,
                seed=seed,
            )

            interaction_z = interaction_result["observed_dz"]
            z_difference = interaction_z
            interaction_pvalue = interaction_result["interaction_pvalue"]
            z_difference_ci = interaction_result["z_difference_ci"]
            null_correlation = interaction_result["null_correlation"]
            has_interaction = True

    # Classify
    primary_sig = (
        primary.z_score >= z_threshold and primary.empirical_pvalue < p_threshold
    )

    if has_interaction and primary_sig:
        # Interaction test overrides binary comparison
        if interaction_pvalue < p_threshold:
            label = "specific"
            summary = (
                f"Effect is specific to {primary_contrast}: "
                f"z={primary.z_score:.2f}, Δz={interaction_z:.2f} "
                f"(interaction p={interaction_pvalue:.4f} < {p_threshold})."
            )
        else:
            label = "shared"
            summary = (
                f"Effect appears shared: {primary_contrast} z={primary.z_score:.2f} "
                f"but Δz={interaction_z:.2f} is not significant "
                f"(interaction p={interaction_pvalue:.4f} >= {p_threshold}). "
                f"Cannot distinguish from shared effect."
            )
    elif primary_sig:
        # Fallback: binary comparison (no interaction test data)
        any_secondary_sig = any(
            c.z_score >= z_threshold and c.empirical_pvalue < p_threshold
            for name, c in contrast_results.items()
            if name != primary_contrast
        )

        if not any_secondary_sig:
            label = "specific"
            summary = (
                f"Effect is specific to {primary_contrast}: "
                f"z={primary.z_score:.2f} (p={primary.empirical_pvalue:.4f}), "
                f"no secondary contrast reaches significance."
            )
        else:
            label = "shared"
            sig_secondary = [
                name for name, c in contrast_results.items()
                if name != primary_contrast
                and c.z_score >= z_threshold
                and c.empirical_pvalue < p_threshold
            ]
            summary = (
                f"Effect is shared: {primary_contrast} z={primary.z_score:.2f} "
                f"but also significant in {', '.join(sig_secondary)}."
            )
    else:
        label = "inconclusive"
        summary = (
            f"Primary contrast {primary_contrast} does not reach significance "
            f"(z={primary.z_score:.2f}, p={primary.empirical_pvalue:.4f})."
        )

    return SpecificityResult(
        primary_contrast=primary_contrast,
        contrasts=contrast_results,
        specificity_ratio=ratio,
        specificity_label=label,
        summary=summary,
        interaction_z=interaction_z,
        interaction_pvalue=interaction_pvalue,
        z_difference=z_difference,
        z_difference_ci=z_difference_ci,
        null_correlation=null_correlation,
    )
