"""
ROAST (Rotation-based gene set testing) method adapter.

Tests whether genes within a clique show coordinated differential
expression while preserving inter-gene correlation structure.

Re-exported from ``method_comparison`` for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..experiment import PreparedCliqueExperiment
    from ..method_comparison_types import MethodName, UnifiedCliqueResult


class ROASTMethod:
    """
    Rotation-based gene set testing (ROAST) method adapter.

    Tests whether genes within a clique show coordinated differential
    expression while preserving inter-gene correlation structure.
    This is a self-contained test - it asks "Are genes in this clique
    enriched for differential expression?"

    Key Advantages:
        - MSQ statistic detects bidirectional regulation (TFs that both
          activate AND repress different targets)
        - Preserves inter-gene correlation structure via rotation
        - Exact p-values from rotation distribution

    Statistical Approach:
        1. Project gene expression onto residual space (QR decomposition)
        2. Generate random rotations that preserve correlation structure
        3. Compute gene set statistics for observed and rotated data
        4. Compute exact p-value from rotation null distribution

    Attributes:
        statistic: Set statistic to use ("msq", "mean", or "floormean")
        alternative: Alternative hypothesis ("up", "down", or "mixed")
        n_rotations: Number of random rotations for p-value estimation

    Example:
        >>> method = ROASTMethod(statistic="msq", n_rotations=9999)
        >>> results = method.test(experiment)
        >>> # MSQ statistic is direction-agnostic, detects bidirectional regulation
    """

    def __init__(
        self,
        statistic: str = "msq",
        alternative: str = "mixed",
        n_rotations: int = 9999,
    ) -> None:
        """
        Initialize ROAST method adapter.

        Args:
            statistic: Set statistic for aggregating gene-level effects.
                - "msq": Mean of squared z-scores (direction-agnostic, detects
                  bidirectional regulation - RECOMMENDED for TF cliques)
                - "mean": Mean z-score (directional, detects coordinated up/down)
                - "floormean": Mean with floored small effects (robust)
            alternative: Alternative hypothesis specification.
                - "mixed": Genes DE in either direction (two-sided, RECOMMENDED)
                - "up": Genes up-regulated (one-sided)
                - "down": Genes down-regulated (one-sided)
            n_rotations: Number of random rotations. Higher = more precise p-values.
                9999 is standard (gives minimum p-value of 1/10000 = 0.0001).
        """
        self.statistic = statistic
        self.alternative = alternative
        self.n_rotations = n_rotations

    @property
    def name(self) -> MethodName:
        """
        Get the method name based on the configured statistic.

        Returns:
            MethodName.ROAST_MSQ, ROAST_MEAN, or ROAST_FLOORMEAN
        """
        from ..method_comparison_types import MethodName

        if self.statistic == "msq":
            return MethodName.ROAST_MSQ
        elif self.statistic == "mean":
            return MethodName.ROAST_MEAN
        else:
            return MethodName.ROAST_FLOORMEAN

    def test(
        self,
        experiment: PreparedCliqueExperiment,
        use_gpu: bool = True,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs: object,
    ) -> list[UnifiedCliqueResult]:
        """
        Run ROAST rotation test on all cliques in the experiment.

        Args:
            experiment: Prepared clique experiment (immutable)
            use_gpu: Whether to use GPU acceleration (requires MLX)
            seed: Random seed for reproducibility
            verbose: Print progress messages
            **kwargs: Additional arguments (ignored)

        Returns:
            List of UnifiedCliqueResult, one per clique with sufficient data
        """
        import logging
        import pandas as pd

        from ..method_comparison_types import UnifiedCliqueResult

        logger = logging.getLogger(__name__)

        # Import rotation module components
        try:
            from ..rotation import (
                RotationTestEngine,
                RotationTestConfig,
            )
        except ImportError as e:
            logger.warning(f"ROAST method unavailable: {e}")
            return []

        results: list[UnifiedCliqueResult] = []

        # Initialize rotation engine with full expression data
        if not isinstance(experiment.sample_metadata, pd.DataFrame):
            logger.warning("ROAST requires sample_metadata as DataFrame")
            return []

        engine = RotationTestEngine(
            data=experiment.data,
            gene_ids=list(experiment.feature_ids),
            metadata=experiment.sample_metadata,
        )

        # Fit model (QR decomposition, EB priors)
        try:
            engine.fit(
                conditions=list(experiment.conditions),
                contrast=experiment.contrast,
                condition_column=experiment.condition_column,
            )
        except Exception as e:
            logger.warning(f"ROAST fit failed: {e}")
            return []

        # Build gene sets dict from cliques: clique_id -> feature_ids
        # IMPORTANT: Convert symbols to feature IDs (UniProt) for rotation engine
        gene_sets: dict[str, list[str]] = {}
        clique_metadata: dict[str, tuple[int, int]] = {}  # clique_id -> (n_proteins, n_found)

        # Build reverse map: get feature_ids from experiment
        feature_id_set = set(experiment.feature_ids)

        for clique in experiment.cliques:
            clique_id = getattr(clique, "clique_id", str(clique))
            protein_ids = getattr(clique, "protein_ids", [])

            # Convert symbols to feature IDs and filter to those in data
            feature_ids_for_set: list[str] = []
            for pid in protein_ids:
                # If pid is already a feature ID, use it directly
                if pid in feature_id_set:
                    feature_ids_for_set.append(pid)
                # Otherwise, try symbol -> feature mapping
                elif pid in experiment.symbol_to_feature:
                    fid = experiment.symbol_to_feature[pid]
                    if fid in feature_id_set:
                        feature_ids_for_set.append(fid)

            if len(feature_ids_for_set) >= 2:
                gene_sets[clique_id] = feature_ids_for_set
                clique_metadata[clique_id] = (len(protein_ids), len(feature_ids_for_set))

        if not gene_sets:
            if verbose:
                print("  No cliques with sufficient proteins for ROAST")
            return []

        # Configure test
        config = RotationTestConfig(
            n_rotations=self.n_rotations,
            use_gpu=use_gpu,
            seed=seed,
        )

        # Run tests on all gene sets
        try:
            rotation_results = engine.test_gene_sets(
                gene_sets, config=config, verbose=verbose
            )
        except Exception as e:
            logger.warning(f"ROAST test_gene_sets failed: {e}")
            return []

        # Convert RotationResult to UnifiedCliqueResult
        stat_key = self.statistic
        alt_key = self.alternative

        for rot_result in rotation_results:
            try:
                # Get p-value for the configured statistic and alternative
                p_value = rot_result.get_pvalue(stat_key, alt_key)

                # Get observed statistic (observed_stats is nested: {stat: {alt: value}})
                observed = rot_result.observed_stats.get(stat_key, {}).get(alt_key, np.nan)

                # For effect_size, use mean z-score with 'up' direction (interpretable)
                mean_z = rot_result.observed_stats.get("mean", {}).get("up", 0.0)

                # Get clique size metadata
                n_proteins, n_found = clique_metadata.get(
                    rot_result.feature_set_id, (rot_result.n_genes, rot_result.n_genes_found)
                )

                # Get active proportion if available
                active_prop = rot_result.active_proportion.get(alt_key, np.nan)

                # Helper to safely extract nested stat values
                def get_stat_val(stat: str, alt: str = "mixed") -> float | None:
                    val = rot_result.observed_stats.get(stat, {}).get(alt, np.nan)
                    return float(val) if np.isfinite(val) else None

                results.append(
                    UnifiedCliqueResult(
                        clique_id=rot_result.feature_set_id,
                        method=self.name,
                        effect_size=float(mean_z),  # Mean z-score as effect size proxy
                        effect_size_se=None,  # ROAST doesn't provide SE
                        p_value=float(p_value),
                        statistic_value=float(observed) if np.isfinite(observed) else np.nan,
                        statistic_type=stat_key,
                        degrees_of_freedom=None,  # Rotation test = exact, no df
                        n_proteins=n_proteins,
                        n_proteins_found=n_found,
                        method_metadata={
                            "alternative": alt_key,
                            "n_rotations": rot_result.n_rotations,
                            "active_proportion": float(active_prop) if np.isfinite(active_prop) else None,
                            "observed_msq": get_stat_val("msq", alt_key),
                            "observed_mean": get_stat_val("mean", "up"),
                            "observed_floormean": get_stat_val("floormean", alt_key),
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"ROAST result conversion failed for {rot_result.feature_set_id}: {e}")
                continue

        return results


__all__ = ["ROASTMethod"]
