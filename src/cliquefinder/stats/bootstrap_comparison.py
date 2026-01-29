"""
Bootstrap subsampling for balanced cross-method comparison.

When case:control ratios are severely imbalanced, bootstrap subsampling
provides stable estimates by:
1. Repeatedly subsampling cases to match a target ratio
2. Running method comparison on each subsample
3. Aggregating results across bootstraps

This preserves statistical validity while utilizing all available data
through repeated sampling.

References:
    - Efron B, Tibshirani RJ (1993). An Introduction to the Bootstrap.
    - Davison AC, Hinkley DV (1997). Bootstrap Methods and their Application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap subsampling analysis."""

    n_bootstraps: int = 100
    target_ratio: float = 2.0  # case:control ratio
    min_group_size: int = 10
    significance_threshold: float = 0.05
    stability_threshold: float = 0.80  # fraction of bootstraps for "stable"
    concordance_threshold: float = 0.50  # method agreement threshold
    seed: int | None = 42
    methods: list[str] | None = None  # None = use defaults (OLS, ROAST_MSQ)
    n_rotations: int = 499  # Reduced for speed
    use_gpu: bool = True
    verbose: bool = True


@dataclass
class BootstrapCliqueResult:
    """Aggregated results for a single clique across bootstrap iterations."""

    clique_id: str
    n_bootstraps: int

    # Selection frequency (fraction significant)
    selection_freq_ols: float
    selection_freq_roast: float
    selection_freq_any: float  # Either method
    selection_freq_both: float  # Both methods

    # P-value summaries
    median_pvalue_ols: float
    median_pvalue_roast: float
    pvalue_ci_low_ols: float
    pvalue_ci_high_ols: float

    # Effect size summaries
    mean_effect: float
    median_effect: float
    effect_ci_low: float
    effect_ci_high: float
    effect_std: float

    # Method concordance
    method_concordance: float  # Fraction where both agree on significance

    # Bootstrap-robust flag
    is_stable: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "clique_id": self.clique_id,
            "n_bootstraps": self.n_bootstraps,
            "selection_freq_ols": self.selection_freq_ols,
            "selection_freq_roast": self.selection_freq_roast,
            "selection_freq_any": self.selection_freq_any,
            "selection_freq_both": self.selection_freq_both,
            "median_pvalue_ols": self.median_pvalue_ols,
            "median_pvalue_roast": self.median_pvalue_roast,
            "pvalue_ci_low_ols": self.pvalue_ci_low_ols,
            "pvalue_ci_high_ols": self.pvalue_ci_high_ols,
            "mean_effect": self.mean_effect,
            "median_effect": self.median_effect,
            "effect_ci_low": self.effect_ci_low,
            "effect_ci_high": self.effect_ci_high,
            "effect_std": self.effect_std,
            "method_concordance": self.method_concordance,
            "is_stable": self.is_stable,
        }


def run_bootstrap_comparison(
    data: NDArray[np.float64],
    feature_ids: list[str],
    metadata: pd.DataFrame,
    cliques: list,
    condition_column: str,
    contrast: tuple[str, str],
    config: BootstrapConfig,
    subject_column: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run bootstrap subsampling analysis for imbalanced designs.

    Args:
        data: Expression matrix (n_features, n_samples)
        feature_ids: Feature identifiers
        metadata: Sample metadata (index = sample IDs)
        cliques: List of CliqueDefinition objects
        condition_column: Metadata column for condition
        contrast: (test_condition, reference_condition)
        config: Bootstrap configuration
        subject_column: Optional subject column for LMM
        output_dir: Optional output directory for intermediate results

    Returns:
        DataFrame with aggregated bootstrap results per clique
    """
    from cliquefinder.stats.method_comparison import (
        run_method_comparison,
        MethodName,
    )

    rng = np.random.default_rng(config.seed)
    test_cond, ref_cond = contrast

    # Identify case and control samples
    case_mask = metadata[condition_column] == test_cond
    ctrl_mask = metadata[condition_column] == ref_cond

    case_samples = metadata.index[case_mask].tolist()
    ctrl_samples = metadata.index[ctrl_mask].tolist()

    n_cases = len(case_samples)
    n_ctrls = len(ctrl_samples)

    # Calculate target case count for balanced design
    target_n_cases = int(n_ctrls * config.target_ratio)

    if target_n_cases >= n_cases:
        print(f"Warning: Target cases ({target_n_cases}) >= available ({n_cases})")
        print("  Using all cases without subsampling")
        target_n_cases = n_cases

    if config.verbose:
        print("=" * 70)
        print("BOOTSTRAP SUBSAMPLING ANALYSIS")
        print("=" * 70)
        print(f"Original design: {n_cases} {test_cond} vs {n_ctrls} {ref_cond}")
        print(f"  Ratio: {n_cases/n_ctrls:.1f}:1")
        print(f"Target design: {target_n_cases} {test_cond} vs {n_ctrls} {ref_cond}")
        print(f"  Ratio: {config.target_ratio:.1f}:1")
        print(f"Bootstrap iterations: {config.n_bootstraps}")
        print(f"Rotations per iteration: {config.n_rotations}")
        print()

    # Storage for per-bootstrap results
    # Dict: clique_id -> list of (ols_p, roast_p, effect) tuples
    bootstrap_results: dict[str, list[tuple[float, float, float]]] = {
        c.clique_id if hasattr(c, 'clique_id') else c.regulator: []
        for c in cliques
    }

    # Get sample-to-index mapping
    all_samples = metadata.index.tolist()
    sample_to_idx = {s: i for i, s in enumerate(all_samples)}

    # Run bootstrap iterations
    for b in range(config.n_bootstraps):
        if config.verbose:
            print(f"Bootstrap {b+1}/{config.n_bootstraps}...", end=" ", flush=True)

        # Subsample cases (with replacement for bootstrap, without for subsampling)
        # Using without replacement for cleaner interpretation
        selected_cases = rng.choice(case_samples, size=target_n_cases, replace=False)

        # Combine with all controls
        bootstrap_samples = list(selected_cases) + ctrl_samples

        # Get indices and subset data
        sample_indices = [sample_to_idx[s] for s in bootstrap_samples]
        bootstrap_data = data[:, sample_indices]
        bootstrap_meta = metadata.loc[bootstrap_samples].copy()

        # Run method comparison (OLS + ROAST only for speed)
        try:
            comparison = run_method_comparison(
                data=bootstrap_data,
                feature_ids=feature_ids,
                sample_metadata=bootstrap_meta,
                cliques=cliques,
                condition_column=condition_column,
                contrast=contrast,
                subject_column=None,  # Skip LMM for speed
                concordance_threshold=config.significance_threshold,
                normalization_method="none",
                imputation_method="none",
                n_rotations=config.n_rotations,
                n_permutations=100,  # Minimal, we don't use permutation results
                use_gpu=config.use_gpu,
                seed=config.seed + b if config.seed else None,
                verbose=False,
            )

            # Extract results
            wide_df = comparison.wide_format()

            for _, row in wide_df.iterrows():
                clique_id = row['clique_id']
                ols_p = row.get('ols_pvalue', np.nan)
                roast_p = row.get('roast_msq_pvalue', np.nan)
                effect = row.get('ols_effect_size', 0.0)

                if clique_id in bootstrap_results:
                    bootstrap_results[clique_id].append((ols_p, roast_p, effect))

            if config.verbose:
                n_sig_ols = (wide_df['ols_pvalue'] < config.significance_threshold).sum()
                n_sig_roast = (wide_df['roast_msq_pvalue'] < config.significance_threshold).sum()
                print(f"OLS sig: {n_sig_ols}, ROAST sig: {n_sig_roast}")

        except Exception as e:
            if config.verbose:
                print(f"FAILED: {e}")
            continue

    # Aggregate results across bootstraps
    if config.verbose:
        print()
        print("Aggregating bootstrap results...")

    aggregated_results = []

    for clique_id, results in bootstrap_results.items():
        if not results:
            continue

        n_boots = len(results)
        ols_pvals = np.array([r[0] for r in results])
        roast_pvals = np.array([r[1] for r in results])
        effects = np.array([r[2] for r in results])

        # Remove NaN
        valid_ols = ~np.isnan(ols_pvals)
        valid_roast = ~np.isnan(roast_pvals)
        valid_both = valid_ols & valid_roast

        if valid_both.sum() < 5:
            continue

        ols_pvals_clean = ols_pvals[valid_ols]
        roast_pvals_clean = roast_pvals[valid_roast]
        effects_clean = effects[valid_both]

        # Selection frequencies
        sig_ols = ols_pvals_clean < config.significance_threshold
        sig_roast = roast_pvals_clean < config.significance_threshold
        sig_both = (ols_pvals[valid_both] < config.significance_threshold) & \
                   (roast_pvals[valid_both] < config.significance_threshold)
        sig_any = (ols_pvals[valid_both] < config.significance_threshold) | \
                  (roast_pvals[valid_both] < config.significance_threshold)

        selection_freq_ols = sig_ols.mean()
        selection_freq_roast = sig_roast.mean()
        selection_freq_both = sig_both.mean()
        selection_freq_any = sig_any.mean()

        # Method concordance (both agree on significance status)
        agree = ((ols_pvals[valid_both] < config.significance_threshold) ==
                 (roast_pvals[valid_both] < config.significance_threshold))
        method_concordance = agree.mean()

        # P-value summaries
        median_pvalue_ols = np.median(ols_pvals_clean)
        median_pvalue_roast = np.median(roast_pvals_clean)
        pvalue_ci_low_ols = np.percentile(ols_pvals_clean, 2.5)
        pvalue_ci_high_ols = np.percentile(ols_pvals_clean, 97.5)

        # Effect size summaries
        mean_effect = np.mean(effects_clean)
        median_effect = np.median(effects_clean)
        effect_ci_low = np.percentile(effects_clean, 2.5)
        effect_ci_high = np.percentile(effects_clean, 97.5)
        effect_std = np.std(effects_clean)

        # Stability criterion
        is_stable = (
            selection_freq_both >= config.stability_threshold and
            method_concordance >= config.concordance_threshold
        )

        result = BootstrapCliqueResult(
            clique_id=clique_id,
            n_bootstraps=n_boots,
            selection_freq_ols=selection_freq_ols,
            selection_freq_roast=selection_freq_roast,
            selection_freq_any=selection_freq_any,
            selection_freq_both=selection_freq_both,
            median_pvalue_ols=median_pvalue_ols,
            median_pvalue_roast=median_pvalue_roast,
            pvalue_ci_low_ols=pvalue_ci_low_ols,
            pvalue_ci_high_ols=pvalue_ci_high_ols,
            mean_effect=mean_effect,
            median_effect=median_effect,
            effect_ci_low=effect_ci_low,
            effect_ci_high=effect_ci_high,
            effect_std=effect_std,
            method_concordance=method_concordance,
            is_stable=is_stable,
        )
        aggregated_results.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame([r.to_dict() for r in aggregated_results])

    # Sort by selection frequency (both methods)
    summary_df = summary_df.sort_values('selection_freq_both', ascending=False)

    if config.verbose:
        n_stable = summary_df['is_stable'].sum()
        print(f"\nBootstrap-stable cliques: {n_stable}")
        print(f"  (significant in >{config.stability_threshold*100:.0f}% of bootstraps,")
        print(f"   methods agree >{config.concordance_threshold*100:.0f}% of time)")

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        summary_path = output_dir / "bootstrap_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Stable hits
        stable_df = summary_df[summary_df['is_stable']]
        if len(stable_df) > 0:
            stable_path = output_dir / "bootstrap_stable_hits.csv"
            stable_df.to_csv(stable_path, index=False)

        # Parameters
        params = {
            "timestamp": datetime.now().isoformat(),
            "n_bootstraps": config.n_bootstraps,
            "target_ratio": config.target_ratio,
            "original_n_cases": n_cases,
            "original_n_ctrls": n_ctrls,
            "subsampled_n_cases": target_n_cases,
            "significance_threshold": config.significance_threshold,
            "stability_threshold": config.stability_threshold,
            "concordance_threshold": config.concordance_threshold,
            "n_rotations": config.n_rotations,
            "seed": config.seed,
            "n_cliques_tested": len(summary_df),
            "n_stable_hits": int(summary_df['is_stable'].sum()),
        }
        params_path = output_dir / "bootstrap_parameters.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        if config.verbose:
            print(f"\nResults saved to {output_dir}")

    return summary_df


__all__ = [
    "BootstrapConfig",
    "BootstrapCliqueResult",
    "run_bootstrap_comparison",
]
