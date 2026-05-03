"""Stratified random selection of seeds for a gradient panel.

The selection is **deterministic** in two senses:

1. Same ``selection_rng_seed`` → identical seed assignments within
   the same NumPy release (``Generator.choice`` is stable for fixed
   inputs within a NumPy version, but NumPy does not guarantee
   cross-release stability for ``default_rng``; see NumPy's random
   compatibility policy).  In practice, the on-disk manifest stores
   the *resulting* seed assignments — a re-run with the same seed
   and the same NumPy version reproduces the manifest exactly, but
   the manifest itself is the authoritative artifact.
2. Output order is canonicalized: stratum members are sorted
   alphabetically within each stratum, and strata are processed in
   sorted-by-name order to make the RNG draw sequence independent
   of dict insertion order.  Stratum order in the returned design
   preserves the caller's input order for human readability of the
   manifest.

This means a panel manifest fully reconstructs the seed list — the
selection is a pure function of its inputs (under a fixed NumPy).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .design import PanelDesign, PanelStratum


def select_panel(
    *,
    candidates_pool: Mapping[str, Any],
    strata_definitions: Mapping[str, Sequence[str]],
    n_per_stratum: int,
    target_seed: str,
    contrast: tuple[str, str],
    max_hops: int = 2,
    n_permutations: int = 999,
    covariates: Sequence[str] = (),
    selection_rng_seed: int = 42,
    description: str = "",
) -> PanelDesign:
    """Build a ``PanelDesign`` by stratified random sampling.

    Each stratum's members are drawn uniformly without replacement
    from ``candidates_pool ∩ strata_definitions[stratum_name]``.  The
    target seed is excluded from all strata even if it appears in
    candidates or any stratum definition.

    Parameters
    ----------
    candidates_pool
        Mapping whose keys are gene symbols eligible for selection
        (typically the matched-degree pool produced by
        ``scripts/find_matched_seed.py``).  Values are unused — the
        mapping is the canonical "pool" structure for compatibility
        with that script's output.
    strata_definitions
        ``{stratum_name: [candidate genes]}``.  The intersection of
        each stratum's gene list with ``candidates_pool`` gives the
        eligible draw set for that stratum.
    n_per_stratum
        Number of seeds to select per stratum.  Each stratum must
        have at least this many eligible members or selection raises.
    target_seed
        Seed under primary test (e.g., ``"C9orf72"``).  Excluded from
        all panel strata even if present in their definitions.
    contrast
        ``(case, control)`` condition labels.
    max_hops, n_permutations, covariates
        Forwarded to ``PanelDesign``; defaults match Wave 24
        gradient pipeline conventions.
    selection_rng_seed
        Seed for ``numpy.random.default_rng``; recorded on the
        returned design so the selection can be re-derived.
    description
        Optional free-text note.

    Returns
    -------
    PanelDesign
        Frozen, validated, and ready to serialize via ``save_yaml``.

    Raises
    ------
    ValueError
        If a stratum has fewer eligible members than ``n_per_stratum``.
    """
    if n_per_stratum < 1:
        raise ValueError(
            f"n_per_stratum must be >= 1, got {n_per_stratum}"
        )
    if not strata_definitions:
        raise ValueError("strata_definitions must be non-empty")

    rng = np.random.default_rng(selection_rng_seed)
    pool = set(candidates_pool.keys())

    # Process strata in sorted-by-name order so the RNG draw sequence
    # is independent of dict insertion order.  Membership for stratum
    # X depends only on (selection_rng_seed, sorted(stratum_names),
    # eligible_X), not on how the caller happened to spell out the
    # dict.  Strata in the *returned* design preserve the caller's
    # input order (human-readable manifest).
    name_to_chosen: dict[str, tuple[str, ...]] = {}
    for name in sorted(strata_definitions.keys()):
        gene_list = strata_definitions[name]
        eligible = sorted(
            g for g in set(gene_list)
            if g in pool and g != target_seed
        )
        if len(eligible) < n_per_stratum:
            raise ValueError(
                f"Stratum {name!r}: {len(eligible)} eligible candidates "
                f"(< {n_per_stratum} required). "
                f"Eligible: {eligible}"
            )
        chosen_indices = rng.choice(
            len(eligible), size=n_per_stratum, replace=False,
        )
        # Sort the selected members for canonical output order.
        name_to_chosen[name] = tuple(sorted(eligible[i] for i in chosen_indices))

    strata = tuple(
        PanelStratum(name=name, members=name_to_chosen[name])
        for name in strata_definitions.keys()
    )

    return PanelDesign(
        target_seed=target_seed,
        strata=strata,
        contrast=contrast,
        max_hops=max_hops,
        n_permutations=n_permutations,
        covariates=tuple(covariates),
        selection_rng_seed=selection_rng_seed,
        description=description,
    )
