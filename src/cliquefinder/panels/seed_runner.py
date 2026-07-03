"""Per-seed gradient runner — library function (no CLI, no shell).

Extracts the per-seed orchestration that previously lived in
``scripts/specificity_triangle.py:run_gradient_for_contrast``.  Used
by the panel runner (``cliquefinder.panels.runner``) to launch one
ProcessPoolExecutor worker per seed.

Boundary
--------
This function is the panel-layer / gradient-layer interface.  It
takes raw inputs (data array, metadata, condition→sample map), fits
the ROAST engine for the contrast, and runs
``DiscoveryBridge.run_gradient_via_shortest_paths``.  All edge-scope
choices (regulatory only) and statistical defaults are inherited
from the gradient pipeline; the runner contributes only the
orchestration.

Behaviors deliberately NOT carried over from the legacy script
-------------------------------------------------------------
- ``mode="bfs"`` (alternative shell construction): dropped because
  Wave 23 showed the regulatory subgraph saturates at h=2 with
  shortest-paths, so bfs would only matter for a measured-protein-only
  subgraph that has no current consumer.
- Edge-rewiring null (``rewire_null_n``, ``rewire_hops``): dropped
  because Wave 21 rejected it as the wrong test for 100%-coverage
  configurations; the panel's Mann-Whitney + matched-seed framing
  is the replacement.
- ``stratified`` tier breakdown, ``active_horizon``,
  ``background_mean_abs_t``: not consumed by ``analyze_panel``;
  retrievable via the underlying ``DiscoveryBridge`` API if needed.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ._intensity import LOG2_TRANSFORM, apply_intensity_transform
from .analysis import PerSeedResult, ShellSummary


class GroupResolver(Protocol):
    """Callable mapping aligned metadata → ``{condition: sample index}``.

    Decouples the panel runner from dataset-specific cohort logic
    (e.g., AnswerALS C9/sporadic/control resolution).  Implementations
    must be picklable for ProcessPoolExecutor — module-level functions
    or ``functools.partial`` over module-level functions both work;
    lambdas and nested closures do not.
    """

    def __call__(
        self, metadata: pd.DataFrame,
    ) -> dict[str, pd.Index]: ...


def run_seed_gradient(
    *,
    seed: str,
    stratum: str,
    contrast: tuple[str, str],
    data: np.ndarray,
    feature_ids: Sequence[str],
    metadata: pd.DataFrame,
    groups: Mapping[str, pd.Index],
    indra_env_file: Path | str | None = None,
    covariates: Sequence[str] = (),
    max_hops: int = 2,
    n_permutations: int = 999,
    rng_seed: int = 42,
    transform: str = LOG2_TRANSFORM,
) -> PerSeedResult:
    """Fit ROAST engine for one contrast and run the gradient for one seed.

    Parameters
    ----------
    seed
        Seed gene symbol (HGNC).
    stratum
        Stratum label this seed belongs to.  Use
        :data:`cliquefinder.panels.analysis.TARGET_STRATUM_LABEL` for
        the panel's primary target seed.
    contrast
        ``(case, control)`` condition labels — must match keys in
        ``groups``.
    data
        Protein × sample matrix.  Columns are sample IDs in the same
        order as ``metadata.index``.
    feature_ids
        UniProt accessions for ``data`` rows.
    metadata
        Per-sample metadata; ``metadata.index`` aligns with ``data``
        columns.
    groups
        Mapping of condition label → sample IDs in that group.
    indra_env_file
        Path to the ``.env`` containing INDRA Neo4j credentials.
    covariates
        Metadata column names to include as design-matrix covariates
        (filtered to those present in the subset metadata).
    max_hops
        Shell depth for the shortest-path gradient.  Defaults to 2,
        which saturates the regulatory subgraph (Wave 23).
    n_permutations
        Degree-binned label-permutation null sample size.
    rng_seed
        Permutation RNG seed.
    transform
        Intensity scale for the moderated-t fit: ``"log2"`` (default,
        ``log2(x+1)``) or ``"raw"`` (linear).  Applied to the abundance
        matrix before the engine; same helper as the landscape path.

    Returns
    -------
    PerSeedResult
        Frozen dataclass; safe to pickle for ProcessPoolExecutor return.
    """
    # Local imports keep this module importable without INDRA.
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.discovery_bridge import DiscoveryBridge
    from cliquefinder.stats.rotation import RotationTestEngine

    cond1, cond2 = contrast

    # Subset metadata + data to the two condition groups.
    keep_samples = groups[cond1].union(groups[cond2])
    sub_meta = metadata.loc[metadata.index.intersection(keep_samples)].copy()
    sub_meta["_condition"] = None
    sub_meta.loc[sub_meta.index.isin(groups[cond1]), "_condition"] = cond1
    sub_meta.loc[sub_meta.index.isin(groups[cond2]), "_condition"] = cond2
    sub_meta = sub_meta.dropna(subset=["_condition"])

    # sub_meta.index is a subset of metadata.index by construction
    # (filtered via metadata.loc[...] above), so every entry has a
    # column index in the original data array.
    sample_id_to_idx = {s: i for i, s in enumerate(metadata.index)}
    aligned_indices = [sample_id_to_idx[s] for s in sub_meta.index]
    sub_data = data[:, aligned_indices]
    # Map onto the modeling scale (log2(x+1) by default) before the
    # engine — same single source of truth as the landscape path.
    sub_data = apply_intensity_transform(sub_data, transform)

    # Fit ROAST engine.
    engine = RotationTestEngine(sub_data, list(feature_ids), sub_meta)
    fit_covariates = [c for c in covariates if c in sub_meta.columns]
    engine.fit(
        conditions=[cond1, cond2],
        contrast=(cond1, cond2),
        condition_column="_condition",
        covariates=fit_covariates,
    )

    symbol_to_feature = map_feature_ids_to_symbols(
        list(feature_ids), verbose=False,
    )

    # Run gradient via bridge.  DiscoveryBridge is a context manager
    # so the Neo4j connection is closed cleanly even on exception.
    t0 = time.time()
    with DiscoveryBridge(
        engine, symbol_to_feature,
        env_file=indra_env_file,
        min_evidence=1,
        min_reliability=0.0,
        min_sources=1,
    ) as bridge:
        result = bridge.run_gradient_via_shortest_paths(
            seed=seed,
            max_hops=max_hops,
            n_permutations=n_permutations,
            rng_seed=rng_seed,
        )
    elapsed = time.time() - t0

    return PerSeedResult(
        seed=seed,
        stratum=stratum,
        slope=float(result.slope),
        slope_pvalue=float(result.slope_pvalue),
        spearman_rho=float(result.spearman_rho),
        spearman_pvalue=float(result.spearman_pvalue),
        shells=tuple(
            ShellSummary(
                hop=int(s.hop),
                n_genes=int(s.n_genes),
                mean_abs_t=float(s.mean_abs_t),
                median_abs_t=float(s.median_abs_t),
            )
            for s in result.shells
        ),
        n_genes_total=int(result.n_genes_total),
        elapsed_seconds=round(elapsed, 1),
    )


def load_panel_inputs(
    *,
    data_path: Path | str,
    metadata_path: Path | str,
    group_resolver: GroupResolver,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, pd.Index]]:
    """Load and align proteomics + metadata for a panel run.

    Centralizes the loader logic that ``run_seed_gradient`` callers
    (the runner module + integration scripts) share.  The
    ``group_resolver`` is a callable mapping aligned metadata to a
    ``{condition_label: sample_index}`` dict; using a Protocol here
    decouples the runner from the AnswerALS-specific cohort logic.

    Returns
    -------
    data, feature_ids, metadata, groups
        Aligned panel inputs.  ``data`` is ``(n_features, n_samples)``;
        ``metadata.index`` matches ``data`` columns.
    """
    data_df = pd.read_csv(data_path, index_col=0)
    feature_ids = list(data_df.index)
    metadata = pd.read_csv(metadata_path, index_col=0)
    common = [s for s in data_df.columns if s in metadata.index]
    metadata = metadata.loc[common]
    data = data_df[common].values
    groups = group_resolver(metadata)
    return data, feature_ids, metadata, groups
