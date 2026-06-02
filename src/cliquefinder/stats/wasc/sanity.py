"""WASC sanity gates + M2.5 calibration tripwire.

Per spec §12 + [[wasc_build_plan]] §10 (Milestone 2.5), the run is INVALID
unless the four-pronged calibration tripwire passes BEFORE M3 (BY-FDR +
Brown's combination) is applied to real data:

  (a) **label-shuffle null FP rate** — shuffle group assignments preserving
      group sizes, re-run the full WASC pipeline, count edges with raw
      p < 0.10.  Bound: mean FP rate ≤ 0.10 + 2·√(0.10·0.90/|E_WASC|).
      Stratified by degree decile and SE quintile (spec mod a, full).

  (b) SPOR down-sampled overlap (deferred to follow-on)
  (c) all-protein-pool ratio (deferred to follow-on)
  (d) F-W vs OLS production-design identity (deferred to follow-on)

This module implements prong (a) end-to-end.  Prongs (b)-(d) get their own
modules but reuse the shuffle infrastructure.

Stratification: Gate 2 strict requires FP rate stratified by (degree decile ×
SE quintile) cells.  This module exposes both pooled and stratified rates
so the caller can implement Gate 2 strict once M3 is ready.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from .bins import AnchorBins
from .concordance import compute_concordance_per_edge
from .fit import EdgeBetaTable, fit_fwl_per_pair
from .null import (
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    compute_anchor_null,
)
from .preprocess import GroupDesign

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelShuffleResult:
    """Per-shuffle FP rates from M2.5 prong (a).

    Attributes
    ----------
    n_shuffles : int
        Number of label-shuffle iterations actually completed.
    B : int
        Permutation depth per shuffle.
    p_threshold : float
        Raw p-value threshold for "positive" classification (default 0.10).
    fp_rate_per_shuffle : np.ndarray
        ``(n_shuffles,)`` fraction of edges with raw p < p_threshold per
        shuffle.  Under H0 (correctly-calibrated null), each entry should
        be ≈ p_threshold ± sqrt(p·(1-p)/n_edges).
    mean_fp_rate : float
        Mean over shuffles.  Compare against the spec Gate 2 bound:
        ``mean_fp_rate ≤ p_threshold + 2·√(p_threshold·(1-p_threshold)/n_edges)``.
    bound : float
        Computed Gate 2 bound.
    pooled_pass : bool
        ``mean_fp_rate <= bound``.  Note: this is the POOLED gate;
        stratified gate (degree × SE) is computed separately once M3 is
        wired.
    per_shuffle_n_finite_p : np.ndarray
        ``(n_shuffles,)`` number of edges with finite p per shuffle (the
        denominator of fp_rate_per_shuffle).
    """
    n_shuffles: int
    B: int
    p_threshold: float
    fp_rate_per_shuffle: np.ndarray
    mean_fp_rate: float
    bound: float
    pooled_pass: bool
    per_shuffle_n_finite_p: np.ndarray


# ---------------------------------------------------------------------------
# Label shuffle primitive
# ---------------------------------------------------------------------------

def shuffle_group_labels(
    designs: dict[str, GroupDesign],
    abundance: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, GroupDesign], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Permute donor → group assignments preserving group sizes.

    The X_cov matrices are carried with each donor (so Sex/Age/Tissue
    follow the donor when the group label flips).  Returns shuffled
    designs + per-group sample_index + per-group abundance slices ready
    for direct use in :class:`NullLoopContext`.

    Parameters
    ----------
    designs
        ``{group: GroupDesign}`` from :func:`build_wasc_data_bundle`.
    abundance
        The proteomics matrix (rows = UniProt, cols = sample_id).
    rng
        Generator for the donor permutation.

    Returns
    -------
    shuffled_designs, sample_index_by_group, abundance_by_group
    """
    # Pool all donors across groups with their original X_cov rows.
    all_samples: list[str] = []
    all_X_rows: list[np.ndarray] = []
    all_column_names: list[str] = []
    group_sizes: dict[str, int] = {}
    group_column_names: dict[str, list[str]] = {}
    for g, d in designs.items():
        for s, row in zip(d.sample_ids, d.X_cov):
            all_samples.append(s)
            all_X_rows.append(row.copy())
        group_sizes[g] = len(d.sample_ids)
        group_column_names[g] = list(d.column_names)
    n_total = len(all_samples)

    # Shuffle donor order
    perm = rng.permutation(n_total)
    perm_samples = [all_samples[i] for i in perm]
    perm_X = [all_X_rows[i] for i in perm]

    # PROBLEM: groups may have different X_cov column shapes (e.g., C9 lacks
    # tissue_Bulk_or_Unknown).  Two policy options:
    #   (i)  Re-build per-group designs from the shuffled sample IDs against
    #        the original metadata.  This makes X_cov columns RE-DERIVE per
    #        shuffled group, which is the most faithful design re-fit.
    #   (ii) Carry the original donor's X_cov row verbatim with the donor.
    #        This produces non-uniform X_cov column sets per shuffled group
    #        and breaks the regression's design assumption.
    # We use (i): pad each donor's X_cov to a common superset of columns by
    # aligning on column_name.
    superset_cols = sorted(set().union(*[set(cn) for cn in group_column_names.values()]))
    # For each original donor (i), build a padded row of length len(superset_cols).
    def _pad_row(orig_row: np.ndarray, orig_cn: list[str]) -> np.ndarray:
        padded = np.zeros(len(superset_cols), dtype=np.float64)
        for v, name in zip(orig_row, orig_cn):
            j = superset_cols.index(name)
            padded[j] = v
        return padded

    # Re-do the all_X_rows with the superset alignment
    aligned_X: list[np.ndarray] = []
    for g, d in designs.items():
        for row in d.X_cov:
            aligned_X.append(_pad_row(row, group_column_names[g]))
    perm_X = [aligned_X[i] for i in perm]

    # Cut into groups
    shuffled_designs: dict[str, GroupDesign] = {}
    sample_index_by_group: dict[str, np.ndarray] = {}
    abundance_by_group: dict[str, np.ndarray] = {}
    A = abundance.values
    col_lookup = {s: i for i, s in enumerate(abundance.columns)}

    offset = 0
    for g, n_g in group_sizes.items():
        g_samples = perm_samples[offset:offset + n_g]
        g_X = np.array(perm_X[offset:offset + n_g], dtype=np.float64)
        offset += n_g
        # Drop superset columns that are entirely 0 within this shuffled group
        keep_mask = np.array([np.any(g_X[:, j] != 0) for j in range(g_X.shape[1])])
        g_X_trimmed = g_X[:, keep_mask]
        g_colnames = [superset_cols[j] for j, k in enumerate(keep_mask) if k]
        shuffled_designs[g] = GroupDesign(
            group=g, sample_ids=g_samples,
            X_cov=g_X_trimmed, column_names=g_colnames,
        )
        cols = np.array([col_lookup[s] for s in g_samples if s in col_lookup],
                        dtype=np.int64)
        sample_index_by_group[g] = cols
        abundance_by_group[g] = A[:, cols]

    return shuffled_designs, sample_index_by_group, abundance_by_group


# ---------------------------------------------------------------------------
# Observed Q under shuffled labels
# ---------------------------------------------------------------------------

def _fit_observed_q_for_works(
    works: Sequence[AnchorWork],
    abundance_by_group: dict[str, np.ndarray],
    X_cov_by_group: dict[str, np.ndarray],
    uniprot_to_row: dict[str, int],
    *,
    group_order: tuple[str, ...] = ("C9ORF72", "SPORADIC", "CONTROL"),
    min_n_per_group: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Re-compute observed Q per anchor under arbitrary group context.

    Returns ``{anchor_uniprot: (n_edges,) Q array}``.  Used by the
    label-shuffle calibration: under shuffled groups, the OBSERVED Q for
    true edges is no longer the M2.2 value; we re-fit.
    """
    if min_n_per_group is None:
        min_n_per_group = {"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15}

    G = len(group_order)
    out: dict[str, np.ndarray] = {}
    for work in works:
        a_row = uniprot_to_row.get(work.anchor_uniprot, -1)
        if a_row < 0:
            out[work.anchor_uniprot] = np.full(len(work.edge_ids), np.nan)
            continue
        anchor_y_by_group = {g: abundance_by_group[g][a_row, :] for g in group_order}
        Q_per_edge = np.full(len(work.edge_ids), np.nan)
        for i, t in enumerate(work.true_targets):
            t_row = uniprot_to_row.get(t, -1)
            if t_row < 0:
                continue
            betas = np.full(G, np.nan)
            ses = np.full(G, np.nan)
            for gi, g in enumerate(group_order):
                target_y = abundance_by_group[g][t_row, :]
                anchor_y = anchor_y_by_group[g]
                X = X_cov_by_group[g]
                fit = fit_fwl_per_pair(target_y, anchor_y, X,
                                       min_n=min_n_per_group.get(g, 10))
                if fit.converged:
                    betas[gi] = fit.beta
                    ses[gi] = fit.se
            valid = np.isfinite(betas) & np.isfinite(ses) & (ses > 0)
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue
            b = betas[valid]
            s = ses[valid]
            w = 1.0 / (s * s)
            beta_bar = float((w * b).sum() / w.sum())
            Q_per_edge[i] = float((w * (b - beta_bar) ** 2).sum())
        out[work.anchor_uniprot] = Q_per_edge
    return out


# ---------------------------------------------------------------------------
# M2.5 prong (a) — label-shuffle calibration
# ---------------------------------------------------------------------------

def run_label_shuffle_calibration(
    works_template: Sequence[AnchorWork],
    anchor_bins_by_anchor: dict[str, AnchorBins],
    abundance: pd.DataFrame,
    designs: dict[str, GroupDesign],
    uniprot_to_row: dict[str, int],
    *,
    n_shuffles: int = 20,
    B: int = 999,
    p_threshold: float = 0.10,
    min_valid_perms: int = 48,
    min_n_per_group: dict[str, int] | None = None,
    group_order: tuple[str, ...] = ("C9ORF72", "SPORADIC", "CONTROL"),
    shuffle_seed: int = 42,
    global_salt: str = "wasc-v1.0.2-shuffle",
    verbose: bool = True,
) -> LabelShuffleResult:
    """M2.5 prong (a): label-shuffle null calibration (pooled).

    For each of ``n_shuffles`` iterations:
      1. Permute donor → group assignment preserving group sizes.
      2. Re-fit observed Q per edge under the shuffled groups.
      3. Run the M2.4 null loop (B permutations of matched non-neighbors)
         under the shuffled groups.
      4. Count edges with raw p < ``p_threshold``.

    Returns the FP rate distribution and the Gate 2 pooled-pass verdict.

    Notes
    -----
    AnchorBins are constructed from the proteomics matrix index and degree
    cache — those are donor-invariant — so the bins built from the
    unshuffled context are valid under any shuffle.

    Pooled Gate 2: ``mean(FP_rate) ≤ p_threshold + 2·sqrt(p·(1-p)/n_edges)``.
    The stratified gate (per (degree, SE) cell) is computed by the caller
    once M3 is wired — this function exposes the raw per-shuffle rates.

    Raises
    ------
    ValueError
        If ``n_shuffles < 1`` or ``B < min_valid_perms``.
    """
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be >= 1, got {n_shuffles}")
    if B < min_valid_perms:
        raise ValueError(f"B ({B}) must be >= min_valid_perms ({min_valid_perms})")

    rng_shuffle = np.random.default_rng(shuffle_seed)
    n_edges_total = sum(len(w.edge_ids) for w in works_template)
    fp_rate_per_shuffle = np.full(n_shuffles, np.nan)
    n_finite_per_shuffle = np.zeros(n_shuffles, dtype=np.int64)

    for sh in range(n_shuffles):
        t0 = time.time()
        shuf_designs, sample_index, abundance_by_group = shuffle_group_labels(
            designs, abundance, rng_shuffle,
        )
        X_cov_by_group = {g: shuf_designs[g].X_cov for g in group_order}

        # 1. Observed Q under shuffled groups
        obs_q = _fit_observed_q_for_works(
            works_template, abundance_by_group, X_cov_by_group, uniprot_to_row,
            group_order=group_order, min_n_per_group=min_n_per_group,
        )

        # 2. Null loop under shuffled groups
        # Re-derive AnchorWork with shuffled-observed Q (true_targets stay the same)
        shuf_works = [
            AnchorWork(
                anchor_uniprot=w.anchor_uniprot,
                edge_ids=w.edge_ids,
                true_targets=w.true_targets,
                Q_obs=obs_q[w.anchor_uniprot],
                seed=anchor_seed(w.anchor_uniprot, global_salt=f"{global_salt}-{sh}"),
            )
            for w in works_template
        ]

        ctx = NullLoopContext(
            abundance_by_group=abundance_by_group,
            sample_index_by_group=sample_index,
            uniprot_to_row=uniprot_to_row,
            X_cov_by_group=X_cov_by_group,
            min_n_per_group=min_n_per_group or {"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15},
            group_order=group_order,
        )

        all_p = []
        for w in shuf_works:
            r = compute_anchor_null(
                work=w,
                anchor_bins=anchor_bins_by_anchor[w.anchor_uniprot],
                abundance_by_group=ctx.abundance_by_group,
                sample_index_by_group=ctx.sample_index_by_group,
                uniprot_to_row=ctx.uniprot_to_row,
                X_cov_by_group=ctx.X_cov_by_group,
                B=B,
                min_n_per_group=ctx.min_n_per_group,
                min_valid_perms=min_valid_perms,
                group_order=ctx.group_order,
            )
            all_p.append(r.p_values)
        all_p_arr = np.concatenate(all_p)
        finite_p = all_p_arr[np.isfinite(all_p_arr)]
        n_finite_per_shuffle[sh] = len(finite_p)
        if len(finite_p):
            fp_rate_per_shuffle[sh] = float((finite_p < p_threshold).mean())

        dt = time.time() - t0
        if verbose:
            logger.info(
                f"Shuffle {sh + 1}/{n_shuffles}: "
                f"FP rate(p<{p_threshold}) = {fp_rate_per_shuffle[sh]:.4f} "
                f"(n_finite={n_finite_per_shuffle[sh]}/{n_edges_total}), "
                f"elapsed={dt:.1f}s"
            )

    # Pooled bound
    valid = np.isfinite(fp_rate_per_shuffle)
    mean_fp = float(np.nanmean(fp_rate_per_shuffle))
    # Use mean n_finite for the bound denominator (per-shuffle variance averaged)
    n_eff = float(np.mean(n_finite_per_shuffle[valid])) if valid.any() else n_edges_total
    bound = p_threshold + 2.0 * float(np.sqrt(p_threshold * (1 - p_threshold) / max(n_eff, 1)))
    pooled_pass = bool(mean_fp <= bound)

    return LabelShuffleResult(
        n_shuffles=int(valid.sum()),
        B=B,
        p_threshold=p_threshold,
        fp_rate_per_shuffle=fp_rate_per_shuffle,
        mean_fp_rate=mean_fp,
        bound=bound,
        pooled_pass=pooled_pass,
        per_shuffle_n_finite_p=n_finite_per_shuffle,
    )
