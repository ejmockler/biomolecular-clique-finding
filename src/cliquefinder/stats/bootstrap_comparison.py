"""
Bootstrap resampling for balanced cross-method comparison.

When case:control ratios are severely imbalanced, bootstrap resampling
provides stable estimates by:
1. Repeatedly resampling cases WITH replacement to match a target ratio
2. Resampling controls WITH replacement (optional, for variance estimation)
3. Running method comparison on each bootstrap sample
4. Aggregating results across bootstraps

True bootstrap (replace=True) captures uncertainty in both populations and
enables proper confidence interval estimation. Subsampling mode (replace=False
for cases, all controls) is also available for cleaner interpretation when
control uncertainty is not of interest.

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
    """Configuration for bootstrap subsampling analysis.

    Performance Notes:
        - ID mappings are cached automatically (no repeated API calls)
        - n_rotations=199 is sufficient for stability assessment (p<0.05 threshold)
        - n_rotations=499 gives higher precision if exact p-values needed
        - GPU acceleration provides ~10x speedup for rotation tests
    """

    n_bootstraps: int = 100
    target_ratio: float = 2.0  # case:control ratio
    min_group_size: int = 10
    significance_threshold: float = 0.05
    stability_threshold: float = 0.80  # fraction of bootstraps for "stable"
    concordance_threshold: float = 0.50  # method agreement threshold
    seed: int | None = 42
    methods: list[str] | None = None  # None = use defaults (OLS, ROAST_MSQ)
    # For stability assessment, 199 rotations is sufficient (p precision ~0.5%)
    # Use 499+ if exact p-values are needed. Minimum 99 for any inference.
    n_rotations: int = 199
    use_gpu: bool = True
    verbose: bool = True
    # Bootstrap strategy for controls:
    # - None (default): Auto-detect based on n_controls (True if n >= 50, else False)
    # - True: Sample controls WITH replacement (true bootstrap, needs n >= 50)
    # - False: Use all controls every iteration (recommended for n < 50)
    # Per Davison & Hinkley (1997): "For imbalanced designs, fix the smaller group"
    bootstrap_controls: bool | None = None
    # Number of parallel workers (future feature - currently sequential)
    # Set to 1 for sequential, >1 for parallel bootstrap iterations
    n_workers: int = 1


@dataclass
class BootstrapCliqueResult:
    """Aggregated results for a single clique across bootstrap iterations."""

    clique_id: str
    n_bootstraps: int

    # Direction information (from input clique)
    direction: str  # "positive", "negative", "mixed", "unknown"
    n_positive_edges: int
    n_negative_edges: int

    # Selection frequency (fraction significant)
    selection_freq_ols: float
    selection_freq_roast: float
    selection_freq_any: float  # Either method
    selection_freq_both: float  # Both methods

    # Per-method stability flags
    is_stable_ols: bool  # selection_freq_ols >= stability_threshold
    is_stable_roast: bool  # selection_freq_roast >= stability_threshold

    # Direction-aware combined stability
    is_robust: bool  # Direction-appropriate criterion met
    stability_criterion: str  # "both_methods" or "roast_only"

    # P-value summaries
    median_pvalue_ols: float
    median_pvalue_roast: float
    pvalue_ci_low_ols: float
    pvalue_ci_high_ols: float
    pvalue_ci_low_roast: float
    pvalue_ci_high_roast: float

    # Effect size summaries (OLS-based)
    mean_effect: float
    median_effect: float
    effect_ci_low: float
    effect_ci_high: float
    effect_std: float

    # Method concordance - nullable for mixed cliques
    method_concordance: float | None  # None for mixed cliques (meaningless)

    @property
    def is_stable(self) -> bool:
        """Legacy compatibility: alias for is_robust."""
        return self.is_robust

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "clique_id": self.clique_id,
            "direction": self.direction,
            "n_positive_edges": self.n_positive_edges,
            "n_negative_edges": self.n_negative_edges,
            "n_bootstraps": self.n_bootstraps,
            "selection_freq_ols": self.selection_freq_ols,
            "selection_freq_roast": self.selection_freq_roast,
            "selection_freq_any": self.selection_freq_any,
            "selection_freq_both": self.selection_freq_both,
            "is_stable_ols": self.is_stable_ols,
            "is_stable_roast": self.is_stable_roast,
            "is_robust": self.is_robust,
            "stability_criterion": self.stability_criterion,
            "method_concordance": self.method_concordance,
            "median_pvalue_ols": self.median_pvalue_ols,
            "median_pvalue_roast": self.median_pvalue_roast,
            "pvalue_ci_low_ols": self.pvalue_ci_low_ols,
            "pvalue_ci_high_ols": self.pvalue_ci_high_ols,
            "pvalue_ci_low_roast": self.pvalue_ci_low_roast,
            "pvalue_ci_high_roast": self.pvalue_ci_high_roast,
            "mean_effect": self.mean_effect,
            "median_effect": self.median_effect,
            "effect_ci_low": self.effect_ci_low,
            "effect_ci_high": self.effect_ci_high,
            "effect_std": self.effect_std,
            # Legacy alias
            "is_stable": self.is_robust,
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

    from numpy.random import SeedSequence

    rng = np.random.default_rng(config.seed)
    # Pre-spawn independent seeds for each bootstrap iteration (ARCH-10).
    # Avoids the prior pattern of ``config.seed + b`` which doesn't
    # guarantee independent streams for arbitrary generators.
    if config.seed is not None:
        _bootstrap_ss = SeedSequence(config.seed)
        _bootstrap_child_seeds = [
            int(child.generate_state(1)[0])
            for child in _bootstrap_ss.spawn(config.n_bootstraps)
        ]
    else:
        _bootstrap_child_seeds = [None] * config.n_bootstraps
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

    # Determine bootstrap strategy for controls
    # Per Davison & Hinkley (1997): Fix smaller group when n < 50
    if config.bootstrap_controls is None:
        # Auto-detect based on control sample size
        bootstrap_controls = n_ctrls >= 50
        if config.verbose and not bootstrap_controls:
            print(f"Note: n_controls={n_ctrls} < 50, using fixed controls (recommended)")
            print("  Per Davison & Hinkley (1997): 'For imbalanced designs, fix the smaller group'")
    else:
        bootstrap_controls = config.bootstrap_controls
        if bootstrap_controls and n_ctrls < 50 and config.verbose:
            print(f"Warning: bootstrap_controls=True but n_controls={n_ctrls} < 50")
            print("  This may produce unstable estimates. Consider bootstrap_controls=False")

    if config.verbose:
        print("=" * 70)
        print("BOOTSTRAP RESAMPLING ANALYSIS")
        print("=" * 70)
        print(f"Original design: {n_cases} {test_cond} vs {n_ctrls} {ref_cond}")
        print(f"  Ratio: {n_cases/n_ctrls:.1f}:1")
        print(f"Target design: {target_n_cases} {test_cond} vs {n_ctrls} {ref_cond}")
        print(f"  Ratio: {config.target_ratio:.1f}:1")
        strategy = "Full bootstrap (cases+controls WITH replacement)" if bootstrap_controls else "Hybrid bootstrap (cases WITH replacement, controls FIXED)"
        print(f"Sampling strategy: {strategy}")
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

    # =========================================================================
    # PRECOMPUTATION (ONE-TIME) - Critical for efficiency
    # =========================================================================
    # Precompute symbol map ONCE (avoids N redundant API calls in bootstrap loop)
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    if config.verbose:
        print("Precomputing ID mappings (one-time)...")
    precomputed_symbol_map = map_feature_ids_to_symbols(feature_ids, verbose=config.verbose)
    if config.verbose:
        print()

    # Run bootstrap iterations
    for b in range(config.n_bootstraps):
        if config.verbose:
            print(f"Bootstrap {b+1}/{config.n_bootstraps}...", end=" ", flush=True)

        # Always sample cases WITH replacement (true bootstrap)
        selected_cases = rng.choice(case_samples, size=target_n_cases, replace=True)

        if bootstrap_controls:
            # TRUE BALANCED BOOTSTRAP:
            # Sample WITH replacement from both groups to capture uncertainty
            # in both populations. This allows:
            # - Same sample to appear multiple times (bootstrap property)
            # - Variance estimation for both case and control populations
            # - Proper confidence interval construction
            selected_ctrls = rng.choice(ctrl_samples, size=n_ctrls, replace=True)
            bootstrap_samples = list(selected_cases) + list(selected_ctrls)
        else:
            # HYBRID BOOTSTRAP (recommended for small control groups):
            # - Cases: WITH replacement (true bootstrap, captures case uncertainty)
            # - Controls: ALL used every iteration (fixed, maximizes power)
            # Per Davison & Hinkley (1997): Fix smaller group when n < 50
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
                seed=_bootstrap_child_seeds[b],
                verbose=False,
                precomputed_symbol_map=precomputed_symbol_map,  # Use cached mappings
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

    # Create clique lookup for direction metadata
    clique_lookup = {}
    for c in cliques:
        cid = c.clique_id if hasattr(c, 'clique_id') else c.regulator
        clique_lookup[cid] = c

    aggregated_results = []

    for clique_id, results in bootstrap_results.items():
        if not results:
            continue

        # Get clique direction metadata
        clique = clique_lookup.get(clique_id)
        if clique:
            direction = clique.direction
            n_positive_edges = clique.n_positive_edges
            n_negative_edges = clique.n_negative_edges
        else:
            # Fallback if clique not found (shouldn't happen)
            direction = "unknown"
            n_positive_edges = 0
            n_negative_edges = 0

        n_boots = len(results)
        ols_pvals = np.array([r[0] for r in results])
        roast_pvals = np.array([r[1] for r in results])
        effects = np.array([r[2] for r in results])

        # Remove NaN
        valid_ols = ~np.isnan(ols_pvals)
        valid_roast = ~np.isnan(roast_pvals)
        valid_both = valid_ols & valid_roast

        # Require at least half of bootstraps to have valid results
        min_required = max(2, n_boots // 2)
        if valid_both.sum() < min_required:
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

        # Per-method stability flags
        is_stable_ols = selection_freq_ols >= config.stability_threshold
        is_stable_roast = selection_freq_roast >= config.stability_threshold

        # Direction-aware combined stability
        is_coherent = direction in ("positive", "negative")
        if is_coherent:
            # For directionally coherent cliques: both methods must be stable
            is_robust = is_stable_ols and is_stable_roast
            stability_criterion = "both_methods"
            # Method concordance is meaningful
            agree = ((ols_pvals[valid_both] < config.significance_threshold) ==
                     (roast_pvals[valid_both] < config.significance_threshold))
            method_concordance = agree.mean()
        else:
            # For mixed/unknown cliques: only ROAST is valid
            is_robust = is_stable_roast
            stability_criterion = "roast_only"
            method_concordance = None  # Not meaningful

        # P-value summaries
        median_pvalue_ols = np.median(ols_pvals_clean)
        median_pvalue_roast = np.median(roast_pvals_clean)
        pvalue_ci_low_ols = np.percentile(ols_pvals_clean, 2.5)
        pvalue_ci_high_ols = np.percentile(ols_pvals_clean, 97.5)
        pvalue_ci_low_roast = np.percentile(roast_pvals_clean, 2.5)
        pvalue_ci_high_roast = np.percentile(roast_pvals_clean, 97.5)

        # Effect size summaries
        mean_effect = np.mean(effects_clean)
        median_effect = np.median(effects_clean)
        effect_ci_low = np.percentile(effects_clean, 2.5)
        effect_ci_high = np.percentile(effects_clean, 97.5)
        effect_std = np.std(effects_clean)

        result = BootstrapCliqueResult(
            clique_id=clique_id,
            n_bootstraps=n_boots,
            direction=direction,
            n_positive_edges=n_positive_edges,
            n_negative_edges=n_negative_edges,
            selection_freq_ols=selection_freq_ols,
            selection_freq_roast=selection_freq_roast,
            selection_freq_any=selection_freq_any,
            selection_freq_both=selection_freq_both,
            is_stable_ols=is_stable_ols,
            is_stable_roast=is_stable_roast,
            is_robust=is_robust,
            stability_criterion=stability_criterion,
            median_pvalue_ols=median_pvalue_ols,
            median_pvalue_roast=median_pvalue_roast,
            pvalue_ci_low_ols=pvalue_ci_low_ols,
            pvalue_ci_high_ols=pvalue_ci_high_ols,
            pvalue_ci_low_roast=pvalue_ci_low_roast,
            pvalue_ci_high_roast=pvalue_ci_high_roast,
            mean_effect=mean_effect,
            median_effect=median_effect,
            effect_ci_low=effect_ci_low,
            effect_ci_high=effect_ci_high,
            effect_std=effect_std,
            method_concordance=method_concordance,
        )
        aggregated_results.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame([r.to_dict() for r in aggregated_results])

    # Sort by selection frequency (both methods) if we have results
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values('selection_freq_both', ascending=False)

    if config.verbose:
        # Count by direction and stability
        if len(summary_df) > 0:
            n_coherent = summary_df['direction'].isin(['positive', 'negative']).sum()
            n_mixed = (summary_df['direction'] == 'mixed').sum()
            n_unknown = (summary_df['direction'] == 'unknown').sum()

            # Stable counts
            coherent_mask = summary_df['direction'].isin(['positive', 'negative'])
            n_stable_coherent = (summary_df.loc[coherent_mask, 'is_robust']).sum() if coherent_mask.any() else 0

            mixed_mask = summary_df['direction'] == 'mixed'
            n_stable_mixed = (summary_df.loc[mixed_mask, 'is_robust']).sum() if mixed_mask.any() else 0

            print()
            print("=" * 70)
            print("DIRECTION-AWARE BOOTSTRAP STABILITY SUMMARY")
            print("=" * 70)
            print(f"Coherent cliques (POSITIVE/NEGATIVE): {n_coherent} tested")
            print(f"  Stable (both OLS & ROAST ≥{config.stability_threshold*100:.0f}%): {n_stable_coherent}")
            print()
            print(f"Mixed cliques: {n_mixed} tested")
            print(f"  Stable (ROAST ≥{config.stability_threshold*100:.0f}%): {n_stable_mixed}")
            if n_unknown > 0:
                print()
                print(f"Unknown direction: {n_unknown} (treated as mixed)")
            print()
            print(f"Total robust cliques: {n_stable_coherent + n_stable_mixed}")
        else:
            print("\nNo cliques tested.")

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary (all cliques)
        summary_path = output_dir / "bootstrap_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        if config.verbose:
            print(f"\n  Summary: {summary_path}")

        # Split stable hits by direction type
        if 'is_robust' in summary_df.columns:
            stable_df = summary_df[summary_df['is_robust']]

            # Coherent stable hits (POSITIVE/NEGATIVE, both methods)
            coherent_stable = stable_df[stable_df['direction'].isin(['positive', 'negative'])]
            if len(coherent_stable) > 0:
                coherent_path = output_dir / "bootstrap_stable_coherent.csv"
                coherent_stable.to_csv(coherent_path, index=False)
                if config.verbose:
                    print(f"  Coherent stable hits: {coherent_path}")

            # Mixed stable hits (ROAST only)
            mixed_stable = stable_df[stable_df['direction'].isin(['mixed', 'unknown'])]
            if len(mixed_stable) > 0:
                mixed_path = output_dir / "bootstrap_stable_mixed.csv"
                mixed_stable.to_csv(mixed_path, index=False)
                if config.verbose:
                    print(f"  Mixed stable hits: {mixed_path}")

            # Combined stable hits (legacy compatibility)
            if len(stable_df) > 0:
                stable_path = output_dir / "bootstrap_stable_hits.csv"
                stable_df.to_csv(stable_path, index=False)
                if config.verbose:
                    print(f"  All stable hits (legacy): {stable_path}")

        # Parameters with direction-aware stats
        if len(summary_df) > 0:
            n_coherent_cliques = int(summary_df['direction'].isin(['positive', 'negative']).sum())
            n_mixed_cliques = int((summary_df['direction'] == 'mixed').sum())
            coherent_mask = summary_df['direction'].isin(['positive', 'negative'])
            n_stable_coherent = int(summary_df.loc[coherent_mask, 'is_robust'].sum()) if coherent_mask.any() else 0
            mixed_mask = summary_df['direction'] == 'mixed'
            n_stable_mixed = int(summary_df.loc[mixed_mask, 'is_robust'].sum()) if mixed_mask.any() else 0
        else:
            n_coherent_cliques = 0
            n_mixed_cliques = 0
            n_stable_coherent = 0
            n_stable_mixed = 0

        params = {
            "timestamp": datetime.now().isoformat(),
            "n_bootstraps": config.n_bootstraps,
            "target_ratio": config.target_ratio,
            "original_n_cases": n_cases,
            "original_n_ctrls": n_ctrls,
            "subsampled_n_cases": target_n_cases,
            "bootstrap_controls": bootstrap_controls,
            "sampling_strategy": "true_bootstrap" if bootstrap_controls else "fixed_controls",
            "significance_threshold": config.significance_threshold,
            "stability_threshold": config.stability_threshold,
            "concordance_threshold": config.concordance_threshold,
            "n_rotations": config.n_rotations,
            "seed": config.seed,
            "n_cliques_tested": len(summary_df),
            # Direction-aware stats
            "n_coherent_cliques": n_coherent_cliques,
            "n_mixed_cliques": n_mixed_cliques,
            "n_stable_coherent": n_stable_coherent,
            "n_stable_mixed": n_stable_mixed,
            "n_robust": int(summary_df['is_robust'].sum()) if len(summary_df) > 0 else 0,
        }
        params_path = output_dir / "bootstrap_parameters.json"
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        if config.verbose:
            print(f"  Parameters: {params_path}")
            print(f"\nResults saved to {output_dir}")

    return summary_df


__all__ = [
    "BootstrapConfig",
    "BootstrapCliqueResult",
    "run_bootstrap_comparison",
]
