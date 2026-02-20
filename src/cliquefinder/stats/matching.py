"""
Exact covariate matching for sensitivity analysis.

Creates balanced subsets by exact-matching on categorical covariates
(e.g., Sex). For each combination of match variable values, finds the
minimum group size and randomly downsamples larger groups to match.

This complements covariate adjustment (Phase 1): adjustment controls
for confounders parametrically, while matching creates physically
balanced subsets as a sensitivity check.

Example:
    C9ORF72 has 17M/8F, Sporadic has 207M/77F →
    Matched subset: 17M + 8F from each group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class MatchResult:
    """Result of exact covariate matching.

    Attributes:
        matched_indices: Array of sample indices in the matched subset.
        balance_audit: DataFrame showing counts per group × stratum
            before and after matching.
        n_original: Total samples before matching.
        n_matched: Total samples after matching.
        match_vars: Variables used for matching.
        groups: Group labels that were matched.
        dropped_per_group: Dict of group → number of samples dropped.
    """

    matched_indices: NDArray[np.int64]
    balance_audit: pd.DataFrame
    n_original: int
    n_matched: int
    match_vars: list[str]
    groups: list[str]
    dropped_per_group: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "n_original": self.n_original,
            "n_matched": self.n_matched,
            "match_vars": self.match_vars,
            "groups": self.groups,
            "dropped_per_group": self.dropped_per_group,
            "balance_audit": self.balance_audit.to_dict(orient="records"),
        }


def exact_match_covariates(
    metadata: pd.DataFrame,
    group_col: str,
    match_vars: list[str],
    groups: list[str] | None = None,
    seed: int | None = None,
) -> MatchResult:
    """
    Create exact-matched subset by downsampling to balance covariates.

    For each unique combination of match_vars values (stratum), finds the
    minimum group size across all groups in that stratum and randomly
    samples that many from each group.

    Args:
        metadata: Sample metadata DataFrame.
        group_col: Column containing group labels (e.g., 'phenotype').
        match_vars: Columns to match on (e.g., ['Sex']).
        groups: Optional subset of groups to include. If None, uses all
            unique values in group_col.
        seed: Random seed for reproducible sampling.

    Returns:
        MatchResult with matched sample indices and balance audit.

    Raises:
        ValueError: If group_col or match_vars not in metadata, or if
            matching produces empty result.
    """
    # Validate columns
    missing = [c for c in [group_col] + match_vars if c not in metadata.columns]
    if missing:
        raise ValueError(f"Columns not found in metadata: {missing}")

    rng = np.random.default_rng(seed)

    # Filter to requested groups
    if groups is None:
        groups = sorted(metadata[group_col].dropna().unique().tolist())

    group_mask = metadata[group_col].isin(groups)
    working = metadata[group_mask].copy()
    # Use positional indices (iloc-compatible) regardless of index type
    positional_indices = np.where(group_mask.values)[0]
    working = working.reset_index(drop=True)
    working["_pos_index"] = positional_indices

    n_original = len(working)

    # Create stratum key from match variables
    working["_stratum"] = working[match_vars].apply(
        lambda row: tuple(row.values), axis=1
    )

    # For each stratum, find min group size and subsample
    matched_indices: list[int] = []
    audit_rows: list[dict] = []
    dropped_per_group: dict[str, int] = {g: 0 for g in groups}

    for stratum, stratum_df in working.groupby("_stratum"):
        stratum_label = (
            dict(zip(match_vars, stratum))
            if isinstance(stratum, tuple)
            else {match_vars[0]: stratum}
        )

        # Count per group in this stratum
        group_counts = {}
        group_indices: dict[str, NDArray] = {}
        for group in groups:
            group_df = stratum_df[stratum_df[group_col] == group]
            group_counts[group] = len(group_df)
            group_indices[group] = group_df["_pos_index"].values

        # Minimum count across groups in this stratum
        min_count = min(group_counts.values())

        if min_count == 0:
            # If any group has 0 in this stratum, skip it entirely
            for group in groups:
                dropped_per_group[group] += group_counts[group]
                audit_rows.append({
                    **stratum_label,
                    "group": group,
                    "original_count": group_counts[group],
                    "matched_count": 0,
                    "dropped": group_counts[group],
                })
            continue

        # Subsample each group to min_count
        for group in groups:
            idx = group_indices[group]
            if len(idx) > min_count:
                sampled = rng.choice(idx, size=min_count, replace=False)
            else:
                sampled = idx
            matched_indices.extend(sampled.tolist())
            dropped = group_counts[group] - min_count
            dropped_per_group[group] += dropped

            audit_rows.append({
                **stratum_label,
                "group": group,
                "original_count": group_counts[group],
                "matched_count": min_count,
                "dropped": dropped,
            })

    if not matched_indices:
        raise ValueError(
            "Matching produced empty result — no strata have samples in all groups. "
            f"Groups: {groups}, match_vars: {match_vars}"
        )

    matched_arr = np.array(sorted(matched_indices), dtype=np.int64)

    # Warn if matching drops too many samples (power concern)
    retention = len(matched_arr) / n_original if n_original > 0 else 0
    if retention < 0.3:
        import warnings
        warnings.warn(
            f"Matching retained only {retention:.0%} of samples "
            f"({len(matched_arr)}/{n_original}). Results may lack "
            f"statistical power. Consider whether the matched subset "
            f"is large enough for meaningful inference.",
            stacklevel=2,
        )
    balance_audit = pd.DataFrame(audit_rows)
    n_matched = len(matched_arr)

    return MatchResult(
        matched_indices=matched_arr,
        balance_audit=balance_audit,
        n_original=n_original,
        n_matched=n_matched,
        match_vars=match_vars,
        groups=groups,
        dropped_per_group=dropped_per_group,
    )
