"""WASC bin builder + matched non-neighbor sampler.

Per spec §4 (v1.0.2 amendment): the permutation null is anchor-local and
matches sampled non-neighbors on a user-selectable subset of axes:

  "degree" — measured-only INDRA hop-1 degree decile per protein
             (cached in ``data/wasc/measured_degrees_v1.json``).
  "miss"   — per-protein missingness decile (fraction of NaN values).
             ON THIS DATASET (AnswerALS RF-imputed input) this axis is
             empirically degenerate: 0/1,423,104 NaN cells → every
             protein lands in bin 9 → axis contributes zero discriminative
             power.  v1.0.2 amendment drops this axis from the default.
  "corr"   — per-anchor |Pearson(anchor, p)| decile, pooled across
             non-external donors.  Per-anchor by construction.

Default axes for v1.0.2: ``("degree", "corr")``.  The 3-axis code path is
preserved as opt-in (``axes=("degree", "miss", "corr")``) so the v1.1
re-derivation against the AnswerALS prebatch matrix can re-activate the
missingness axis without reverting this amendment.

For one anchor:
  - Each candidate non-neighbor protein lands in a single tuple-of-bins
    cell whose shape equals ``len(axes)``.
  - The matched non-neighbor sampler picks, for each true hop-1 neighbor
    of the anchor, one random protein from the SAME cell (excluding the
    anchor + true-neighbor set + already-sampled).

Bin-empty edge cases: cell genuinely empty after exclusion → that
per-pair sample is marked ``None`` (degenerate) and dropped from the
permutation iteration's Q computation by the caller.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_DEGREES_PATH = (
    Path(__file__).resolve().parents[4]
    / "data" / "wasc" / "measured_degrees_v1.json"
)


# ---------------------------------------------------------------------------
# Bin assignment primitives
# ---------------------------------------------------------------------------

def assign_decile(values: np.ndarray) -> np.ndarray:
    """Bin numeric values into 10 quantile-based bins (0-9).

    Tied values get the same bin (right-side convention).
    Non-finite values get bin -1 (caller must skip them).
    Returns ``(n,)`` int array of bin indices.
    """
    out = np.full(len(values), -1, dtype=np.int8)
    finite = np.isfinite(values)
    if finite.sum() < 10:
        return out
    cutoffs = np.quantile(values[finite], np.linspace(0.1, 0.9, 9))
    out[finite] = np.searchsorted(cutoffs, values[finite], side="right")
    np.clip(out, -1, 9, out=out)
    return out


def compute_missingness_per_protein(abundance: pd.DataFrame) -> pd.Series:
    """Fraction of NaN per protein (row).  Returns ``uniprot → missingness``."""
    return abundance.isna().mean(axis=1)


def compute_marginal_correlation_with_anchor(
    abundance: pd.DataFrame,
    anchor_uniprot: str,
    eligible_samples: list[str] | None = None,
) -> pd.Series:
    """``|Pearson(anchor_a, every_other_protein)|`` pooled across donors.

    NaNs are pairwise-dropped (per-protein, against the anchor's observed
    samples).  Returns uniprot → |corr|; ``anchor_uniprot`` itself gets NaN.
    Proteins with fewer than 5 jointly-observed samples get NaN.
    """
    cols = list(eligible_samples) if eligible_samples is not None else list(abundance.columns)
    a_full = abundance.loc[anchor_uniprot, cols].values.astype(np.float64)
    a_valid = ~np.isnan(a_full)
    if a_valid.sum() < 5:
        return pd.Series(np.nan, index=abundance.index, dtype=np.float64)
    a_obs = a_full[a_valid]
    a_centered = a_obs - a_obs.mean()
    a_sd = float(np.sqrt(np.sum(a_centered ** 2)))
    if a_sd < 1e-12:
        return pd.Series(np.nan, index=abundance.index, dtype=np.float64)

    X = abundance.loc[:, cols].values.astype(np.float64)  # (n_proteins, n_samples)
    X_anchor_obs = X[:, a_valid]
    # Pairwise complete-case correlation: for each protein, drop samples
    # where THAT protein is NaN; intersection with a_obs is X_anchor_obs's NaN mask.
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    for i in range(X.shape[0]):
        row = X_anchor_obs[i]
        valid = ~np.isnan(row)
        if valid.sum() < 5:
            continue
        x = row[valid]
        a = a_obs[valid]
        x_c = x - x.mean()
        a_c = a - a.mean()
        x_sd = np.sqrt(np.sum(x_c ** 2))
        a_sd_local = np.sqrt(np.sum(a_c ** 2))
        if x_sd < 1e-12 or a_sd_local < 1e-12:
            continue
        r = float(np.sum(x_c * a_c) / (x_sd * a_sd_local))
        out[i] = abs(r)
    out[abundance.index == anchor_uniprot] = np.nan  # anchor vs itself: NaN
    return pd.Series(out, index=abundance.index)


# ---------------------------------------------------------------------------
# Per-anchor bin index + matched sampler
# ---------------------------------------------------------------------------

DEFAULT_AXES: tuple[str, ...] = ("degree", "corr")  # v1.0.2


@dataclass(frozen=True)
class AnchorBins:
    """Per-anchor matched-bin index over candidate proteins.

    Cell-key shape and content are determined by ``axes`` (length 1-3).

    Attributes
    ----------
    anchor_uniprot : str
        The anchor protein this index is built for.
    protein_ids : tuple[str, ...]
        All candidate proteins (the full row index of the proteomics matrix).
    deg_bin : np.ndarray
        ``(n_proteins,)`` degree-decile bin indices (-1 for NaN).  Always
        populated.
    miss_bin : np.ndarray | None
        ``(n_proteins,)`` missingness-decile bin indices (-1 for NaN), or
        ``None`` when ``"miss"`` is not in ``axes`` (v1.0.2 default).
    corr_bin : np.ndarray
        ``(n_proteins,)`` |Pearson|-decile bin indices (-1 for NaN).  Always
        populated when ``"corr"`` is in ``axes``.
    cells : dict[tuple[int, ...], tuple[str, ...]]
        ``{(bin_for_axis_0, bin_for_axis_1, ...): tuple of UniProts}``;
        tuple length matches ``len(axes)``.  Excludes the anchor itself.
    axes : tuple[str, ...]
        Axis names in cell-key order.  Default ``("degree", "corr")`` for
        v1.0.2.  Use ``("degree", "miss", "corr")`` to opt back into the
        original 3-axis behavior for v1.1 prebatch re-derivation.
    """
    anchor_uniprot: str
    protein_ids: tuple[str, ...]
    deg_bin: np.ndarray
    miss_bin: np.ndarray | None
    corr_bin: np.ndarray
    cells: dict[tuple[int, ...], tuple[str, ...]]
    axes: tuple[str, ...] = DEFAULT_AXES
    _protein_to_idx: dict[str, int] = field(default_factory=dict)

    def get_cell_key(self, protein: str) -> tuple[int, ...] | None:
        """Return the cell key (one bin per axis) for ``protein``.

        Returns ``None`` if the protein is not in the index or if any
        configured axis assigns it bin -1.  Cell-key tuple length always
        matches ``len(self.axes)``.
        """
        idx = self._protein_to_idx.get(protein)
        if idx is None:
            return None
        bin_arrays = {
            "degree": self.deg_bin,
            "miss": self.miss_bin,
            "corr": self.corr_bin,
        }
        parts: list[int] = []
        for axis in self.axes:
            arr = bin_arrays[axis]
            if arr is None:
                return None
            v = int(arr[idx])
            if v < 0:
                return None
            parts.append(v)
        return tuple(parts)


def build_anchor_bins(
    anchor_uniprot: str,
    abundance: pd.DataFrame,
    degrees: dict[str, int],
    missingness: pd.Series | None = None,
    eligible_samples: list[str] | None = None,
    *,
    precomputed_corr: pd.Series | None = None,
    axes: tuple[str, ...] = DEFAULT_AXES,
    eligible_proteins: set[str] | frozenset[str] | None = None,
) -> AnchorBins:
    """Build the matched-bin index for one anchor.

    Parameters
    ----------
    anchor_uniprot
        The anchor protein.
    abundance
        Proteomics matrix, rows = UniProt, cols = sample IDs.
    degrees
        UniProt → measured-only hop-1 degree (from
        data/wasc/measured_degrees_v1.json).
    missingness
        UniProt → missing-value rate; result of
        :func:`compute_missingness_per_protein`.  Required if and only if
        ``"miss"`` is in ``axes``.
    eligible_samples
        Donor IDs to use when computing the marginal correlation axis.
        Defaults to all columns of ``abundance``.
    precomputed_corr
        Override for the per-anchor |Pearson| series (test injection).
    axes
        Tuple of axis names in cell-key order.  Default
        ``("degree", "corr")`` per v1.0.2 amendment.  Use
        ``("degree", "miss", "corr")`` to opt back into 3-axis behavior
        (requires ``missingness`` to be provided).
    eligible_proteins
        Restrict the matched-bin CANDIDATE POOL to this set of UniProts
        (after removing the anchor itself).  ``None`` (default) means
        "use the full proteomics matrix" — this is the all-protein-pool
        variant (build plan prong (c) sensitivity).  The spec §4 canonical
        primary requires this to be ``M_T`` (the theme's cluster members)
        for an (anchor, theme) bin construction.

        Note: deciles are STILL computed on the full proteome's values
        (degree / |Pearson|), so the bin EDGES are global; only the cell
        MEMBERSHIP is restricted.  This makes the same bin definitions
        directly comparable between the theme-restricted primary and
        the all-protein-pool prong (c) variant.

    Raises
    ------
    ValueError
        If ``axes`` includes ``"miss"`` but ``missingness`` is ``None``.
    """
    if "miss" in axes and missingness is None:
        raise ValueError(
            "axes includes 'miss' but missingness is None; "
            "either pass a missingness Series or remove 'miss' from axes."
        )
    unknown = [a for a in axes if a not in ("degree", "miss", "corr")]
    if unknown:
        raise ValueError(f"Unknown axes: {unknown}; expected subset of (degree, miss, corr).")

    proteins = tuple(abundance.index)
    deg = np.array([degrees.get(p, 0) for p in proteins], dtype=np.float64)
    deg_bin = assign_decile(deg)

    if "miss" in axes:
        miss_vals = missingness.reindex(proteins).values.astype(np.float64)
        miss_bin: np.ndarray | None = assign_decile(miss_vals)
    else:
        miss_bin = None

    if "corr" in axes:
        if precomputed_corr is None:
            corr = compute_marginal_correlation_with_anchor(
                abundance, anchor_uniprot, eligible_samples=eligible_samples,
            )
        else:
            corr = precomputed_corr
        corr_vals = corr.reindex(proteins).values.astype(np.float64)
        corr_bin = assign_decile(corr_vals)
    else:
        # Even if 'corr' is not in axes, keep a placeholder of -1 so the
        # AnchorBins schema is uniform.  Future-proofing for degree-only
        # null variants.
        corr_bin = np.full(len(proteins), -1, dtype=np.int8)

    by_axis = {"degree": deg_bin, "miss": miss_bin, "corr": corr_bin}
    pool: set[str] | None = (
        set(eligible_proteins) if eligible_proteins is not None else None
    )

    cells: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for i, p in enumerate(proteins):
        if p == anchor_uniprot:
            continue
        if pool is not None and p not in pool:
            continue
        parts: list[int] = []
        skip = False
        for axis in axes:
            arr = by_axis[axis]
            v = int(arr[i])
            if v < 0:
                skip = True
                break
            parts.append(v)
        if skip:
            continue
        cells[tuple(parts)].append(p)

    cells_frozen = {k: tuple(v) for k, v in cells.items()}
    return AnchorBins(
        anchor_uniprot=anchor_uniprot,
        protein_ids=proteins,
        deg_bin=deg_bin,
        miss_bin=miss_bin,
        corr_bin=corr_bin,
        cells=cells_frozen,
        axes=axes,
        _protein_to_idx={p: i for i, p in enumerate(proteins)},
    )


def sample_matched_non_neighbors(
    anchor_bins: AnchorBins,
    true_neighbors: list[str],
    rng: np.random.Generator,
    *,
    relaxation_axes: tuple[str, ...] = ("corr",),
    max_relaxation: int = 1,
) -> tuple[list[str | None], int, list[int]]:
    """For each true neighbor, sample a non-neighbor from the SAME bin.

    Excluded: anchor itself, all true neighbors, already-sampled this call.

    Per spec §4 (v1.0.2): if the exact cell is empty after exclusion, fall
    back to a widened cell by relaxing ``relaxation_axes`` by ±1 decile.
    Default ``("corr",)`` matches the spec's "±1 decile fallback on |r|".
    Axes NOT in ``relaxation_axes`` (e.g., ``"degree"``) are kept exact.

    The widened cells are searched in order of L1 distance from the original
    (small relaxations preferred).  Cells from all relaxation levels up to
    ``max_relaxation`` are pooled into ONE candidate set; sampling is
    uniform over that pool, NOT cell-by-cell.

    Returns
    -------
    (sampled, n_degenerate, relaxation_level_per_pos)
        ``sampled``: list of UniProts (one per ``true_neighbors`` position,
        or ``None`` if no eligible candidate found at any relaxation level).
        ``n_degenerate``: count of ``None`` entries.
        ``relaxation_level_per_pos``: per-position relaxation distance
        used (0 = exact, 1 = ±1-axis widening, ..., -1 = degenerate).
        Used by Sanity Gate 7 to report per-anchor fallback rate.
    """
    exclude: set[str] = set(true_neighbors) | {anchor_bins.anchor_uniprot}
    out: list[str | None] = []
    relaxation_levels: list[int] = []
    n_degenerate = 0

    # Resolve which axis indices in the bin's `axes` tuple are relaxable.
    axis_to_index = {axis: i for i, axis in enumerate(anchor_bins.axes)}
    relax_axis_indices = [axis_to_index[a] for a in relaxation_axes
                          if a in axis_to_index]

    def _widened_keys(orig_key: tuple[int, ...], level: int) -> list[tuple[int, ...]]:
        """All cell keys reachable from orig_key by widening EXACTLY one
        relax-axis by +level or -level (the canonical "L_inf shell at
        relaxation distance level on relax_axis_indices")."""
        if level == 0:
            return [orig_key]
        results: list[tuple[int, ...]] = []
        for axis_i in relax_axis_indices:
            for sign in (-1, 1):
                new = list(orig_key)
                new[axis_i] += sign * level
                if 0 <= new[axis_i] <= 9:
                    results.append(tuple(new))
        return results

    for tn in true_neighbors:
        cell_key = anchor_bins.get_cell_key(tn)
        if cell_key is None:
            out.append(None)
            relaxation_levels.append(-1)
            n_degenerate += 1
            continue
        chosen: str | None = None
        level_used = -1
        for level in range(0, max_relaxation + 1):
            keys = _widened_keys(cell_key, level)
            # Pool candidates from ALL keys at this level
            pool: list[str] = []
            for k in keys:
                for p in anchor_bins.cells.get(k, ()):
                    if p not in exclude:
                        pool.append(p)
            if pool:
                idx = int(rng.integers(0, len(pool)))
                chosen = pool[idx]
                level_used = level
                break
        if chosen is None:
            out.append(None)
            relaxation_levels.append(-1)
            n_degenerate += 1
        else:
            out.append(chosen)
            relaxation_levels.append(level_used)
            exclude.add(chosen)
    return out, n_degenerate, relaxation_levels


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_measured_degrees(path: Path | str | None = None) -> dict[str, int]:
    """Load the cached measured-only degree map (data/wasc/measured_degrees_v1.json)."""
    p = Path(path) if path else DEFAULT_DEGREES_PATH
    doc = json.loads(p.read_text())
    return {k: int(v) for k, v in doc["degrees"].items()}
