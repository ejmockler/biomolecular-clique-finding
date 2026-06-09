"""WASC permutation null — per-anchor matched-non-neighbor loop.

Per spec §4: for each anchor a, the null is anchor-local:

  Observed Q_obs[e]  — for true edge e = (a, t), cochran_q(β̂_e_g, SE_e_g)
                       across g ∈ {C9, SPOR, CTRL}.

  Null      Q_null[e, b] — for permutation iteration b, sample a matched
                           non-neighbor t'_e for every true target t_e of a
                           (via :func:`bins.sample_matched_non_neighbors`),
                           refit β̂ and SE for the fake edge (a, t'_e) in
                           every group, compute cochran_q.

  Permutation p-value  p[e] = (1 + |{b : Q_null[e,b] ≤ Q_obs[e]}|) / (B + 1)
                       (LOWER-TAIL per spec §4 line 207: small Q = invariant
                        slopes = WASC-positive ⇒ small p)

Bin-empty positions contribute NaN to Q_null and are dropped from the
denominator.  Edges with fewer than ``min_valid_perms`` (spec C2 floor = 48)
finite null draws get p = NaN and fall out of BY-FDR downstream.

Resumability: a JSONL checkpoint file is appended per-anchor.  On restart,
already-completed anchors are skipped.  RNG seeds are md5-derived from
the anchor's UniProt accession + a global salt, so the same anchor produces
the same null sequence across reruns (subject to permutation engine
versioning recorded in the manifest).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

from .bins import AnchorBins, sample_matched_non_neighbors
from .concordance import cochran_q
from .fit import fit_fwl_per_pair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-anchor work unit + result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorWork:
    """Specification of one anchor's null-loop work unit.

    Attributes
    ----------
    anchor_uniprot : str
        The anchor protein.
    edge_ids : tuple[str, ...]
        Canonical edge identifiers for the anchor's true within-theme
        edges, in the order they appear in the global EdgeBetaTable.
    true_targets : tuple[str, ...]
        Target UniProts of the anchor's true edges, parallel to ``edge_ids``.
    Q_obs : np.ndarray
        ``(n_edges,)`` observed Cochran Q per true edge.
    seed : int
        Deterministic permutation seed for this anchor (md5-derived).
    """
    anchor_uniprot: str
    edge_ids: tuple[str, ...]
    true_targets: tuple[str, ...]
    Q_obs: np.ndarray
    seed: int


@dataclass(frozen=True)
class AnchorNullResult:
    """Per-anchor null-loop output.

    Attributes
    ----------
    anchor_uniprot : str
    edge_ids : tuple[str, ...]
    Q_obs : np.ndarray
        ``(n_edges,)`` copied from input AnchorWork.
    null_Q : np.ndarray
        ``(n_edges, B)`` null Cochran-Q draws; NaN entries are degenerate
        permutations (bin-empty, NaN fit) and dropped from the p-value.
    p_values : np.ndarray
        ``(n_edges,)`` permutation p-values.  NaN if fewer than
        ``min_valid_perms`` finite null draws.
    n_degenerate_per_edge : np.ndarray
        ``(n_edges,)`` count of NaN null draws (bin-empty or non-converged
        FWL).  Diagnostic only.
    """
    anchor_uniprot: str
    edge_ids: tuple[str, ...]
    Q_obs: np.ndarray
    null_Q: np.ndarray
    p_values: np.ndarray
    n_degenerate_per_edge: np.ndarray


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def anchor_seed(anchor_uniprot: str, global_salt: str = "wasc-v1.0") -> int:
    """Deterministic uint32 seed derived from anchor UniProt + salt."""
    h = hashlib.md5(f"{global_salt}|{anchor_uniprot}".encode()).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


# ---------------------------------------------------------------------------
# Per-anchor null computation
# ---------------------------------------------------------------------------

def compute_anchor_null(
    work: AnchorWork,
    anchor_bins: AnchorBins,
    abundance_by_group: dict[str, np.ndarray],     # group → (n_proteins, n_g)
    sample_index_by_group: dict[str, np.ndarray],  # group → (n_g,)  (legacy)
    uniprot_to_row: dict[str, int],                # protein UniProt → row index
    X_cov_by_group: dict[str, np.ndarray],         # group → (n_g, p_cov)
    *,
    B: int = 9999,
    min_n_per_group: dict[str, int] | None = None,
    min_valid_perms: int = 48,
    min_unique_q_values: int = 5,
    group_order: tuple[str, ...] = ("C9ORF72", "SPORADIC", "CONTROL"),
) -> AnchorNullResult:
    """Run B permutations for one anchor and return the null + p-values.

    Parameters
    ----------
    work
        AnchorWork describing the anchor + its true edges + observed Q + seed.
    anchor_bins
        Pre-built AnchorBins for the anchor.
    abundance_by_group
        ``{group: (n_proteins, n_g_samples) numpy array}``.  Columns are the
        per-group sample subset; rows align with ``uniprot_to_row``.
    sample_index_by_group
        Unused but kept in the signature for forward-compat (legacy).
    uniprot_to_row
        UniProt accession → row index into ``abundance_by_group``.
        Proteins not in the matrix should map to -1; the null loop skips them.
    X_cov_by_group
        ``{group: (n_g, p_cov) covariate matrix}`` from
        :func:`build_group_design`.
    B
        Number of permutation iterations.
    min_n_per_group
        Per-group complete-case floor for FWL.  Defaults to spec §2.3.
    min_valid_perms
        Edges with fewer than this many finite null draws get p = NaN.
        Spec C2 floor = 48.
    min_unique_q_values
        Edges whose finite null distribution has fewer than this many
        DISTINCT values get p = NaN.  Default 5.  Addresses the
        mathematical pathology where sparse-cell matched-bin sampling
        draws the same fake t' across many iterations → Q_null is
        constant → lower-tail formula `(1+#{Q_null ≤ Q_obs})/(B+1)` is
        deterministic 0.01 or 1.0 (path-debug audit finding).
        Set to 1 to disable the guard.

    Returns
    -------
    AnchorNullResult
    """
    if min_n_per_group is None:
        min_n_per_group = {"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15}

    n_edges = len(work.edge_ids)
    null_Q = np.full((n_edges, B), np.nan, dtype=np.float64)
    n_deg_per_edge = np.zeros(n_edges, dtype=np.int64)

    rng = np.random.default_rng(work.seed)
    a_row = uniprot_to_row.get(work.anchor_uniprot, -1)
    if a_row < 0:
        # Anchor not in the proteomics matrix — entire null is NaN.
        return AnchorNullResult(
            anchor_uniprot=work.anchor_uniprot,
            edge_ids=work.edge_ids,
            Q_obs=work.Q_obs.copy(),
            null_Q=null_Q,
            p_values=np.full(n_edges, np.nan, dtype=np.float64),
            n_degenerate_per_edge=np.full(n_edges, B, dtype=np.int64),
        )

    # Pre-extract per-group anchor row (constant across permutations).
    anchor_y_by_group = {
        g: abundance_by_group[g][a_row, :]
        for g in group_order
    }

    true_targets_list = list(work.true_targets)
    G = len(group_order)
    betas_buf = np.full(G, np.nan)
    ses_buf = np.full(G, np.nan)

    for b in range(B):
        sampled, _, _ = sample_matched_non_neighbors(
            anchor_bins, true_targets_list, rng,
        )
        for i, fake_t in enumerate(sampled):
            if fake_t is None:
                n_deg_per_edge[i] += 1
                continue
            t_row = uniprot_to_row.get(fake_t, -1)
            if t_row < 0:
                n_deg_per_edge[i] += 1
                continue
            # Fit FWL per group for the fake edge (a, fake_t)
            betas_buf[:] = np.nan
            ses_buf[:] = np.nan
            for gi, g in enumerate(group_order):
                target_y = abundance_by_group[g][t_row, :]
                anchor_y = anchor_y_by_group[g]
                X = X_cov_by_group[g]
                fit = fit_fwl_per_pair(
                    target_y, anchor_y, X,
                    min_n=min_n_per_group.get(g, 10),
                )
                if fit.converged:
                    betas_buf[gi] = fit.beta
                    ses_buf[gi] = fit.se
            q_result = cochran_q(betas_buf.copy(), ses_buf.copy())
            if np.isfinite(q_result.Q):
                null_Q[i, b] = q_result.Q
            else:
                n_deg_per_edge[i] += 1

    # Per-edge permutation p-values — LOWER-TAIL per spec §4 line 207.
    # Small Q ⇒ invariant slopes ⇒ WASC-positive ⇒ small p.
    #
    # GUARD (path-debug audit finding): sparse-cell matched-bin
    # sampling can draw the same fake t' across many iterations, producing
    # CONSTANT Q_null.  The lower-tail formula is then mechanically
    # undefined (deterministic 0.01 or 1.0).  Reject edges with fewer than
    # min_unique_q_values distinct finite null draws.
    p_values = np.full(n_edges, np.nan, dtype=np.float64)
    for i in range(n_edges):
        valid_null = null_Q[i][np.isfinite(null_Q[i])]
        if len(valid_null) < min_valid_perms:
            continue
        if not np.isfinite(work.Q_obs[i]):
            continue
        if len(np.unique(valid_null)) < min_unique_q_values:
            continue
        p_values[i] = (1 + np.sum(valid_null <= work.Q_obs[i])) / (len(valid_null) + 1)

    return AnchorNullResult(
        anchor_uniprot=work.anchor_uniprot,
        edge_ids=work.edge_ids,
        Q_obs=work.Q_obs.copy(),
        null_Q=null_Q,
        p_values=p_values,
        n_degenerate_per_edge=n_deg_per_edge,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O — per-anchor JSONL append
# ---------------------------------------------------------------------------

_CHECKPOINT_LOCK = threading.Lock()


def _result_to_jsonl_record(r: AnchorNullResult) -> dict:
    """Compact JSONL representation of an AnchorNullResult.

    Stores Q_obs, p_values, n_degenerate.  The full ``null_Q`` matrix is
    NOT serialized to JSONL (too large for B=9999) — those go to a sibling
    .npz cache if requested.
    """
    return {
        "anchor": r.anchor_uniprot,
        "edge_ids": list(r.edge_ids),
        "Q_obs": [float(x) if np.isfinite(x) else None for x in r.Q_obs],
        "p_values": [float(x) if np.isfinite(x) else None for x in r.p_values],
        "n_degenerate": [int(x) for x in r.n_degenerate_per_edge],
    }


def append_checkpoint(checkpoint_path: Path, result: AnchorNullResult) -> None:
    """Atomically append one AnchorNullResult to the JSONL checkpoint."""
    record = _result_to_jsonl_record(result)
    line = json.dumps(record, sort_keys=True) + "\n"
    with _CHECKPOINT_LOCK:
        with checkpoint_path.open("a") as fh:
            fh.write(line)


def load_completed_anchors(checkpoint_path: Path) -> set[str]:
    """Return UniProts of anchors that have already been written to the JSONL."""
    if not checkpoint_path.exists():
        return set()
    done: set[str] = set()
    with checkpoint_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["anchor"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed checkpoint line")
    return done


# ---------------------------------------------------------------------------
# Top-level orchestrator (single-process; joblib parallel comes via wrapper)
# ---------------------------------------------------------------------------

class NullLoopContext(NamedTuple):
    """Read-only context shared across all anchor work units."""
    abundance_by_group: dict[str, np.ndarray]
    sample_index_by_group: dict[str, np.ndarray]
    uniprot_to_row: dict[str, int]
    X_cov_by_group: dict[str, np.ndarray]
    min_n_per_group: dict[str, int]
    group_order: tuple[str, ...]


def run_null_serial(
    works: Iterable[AnchorWork],
    anchor_bins_by_anchor: dict[str, AnchorBins],
    ctx: NullLoopContext,
    *,
    B: int = 9999,
    min_valid_perms: int = 48,
    checkpoint_path: Path | None = None,
    skip_completed: bool = True,
) -> list[AnchorNullResult]:
    """Single-process driver for the null loop.

    Use this for small smoke runs and tests.  For production B=9999 ×
    ~300 anchors, switch to a joblib wrapper that calls
    :func:`compute_anchor_null` per anchor in parallel.
    """
    completed: set[str] = set()
    if checkpoint_path is not None and skip_completed:
        completed = load_completed_anchors(checkpoint_path)
        if completed:
            logger.info(f"Resuming: skipping {len(completed)} completed anchors")

    results: list[AnchorNullResult] = []
    for work in works:
        if work.anchor_uniprot in completed:
            continue
        t = time.time()
        r = compute_anchor_null(
            work=work,
            anchor_bins=anchor_bins_by_anchor[work.anchor_uniprot],
            abundance_by_group=ctx.abundance_by_group,
            sample_index_by_group=ctx.sample_index_by_group,
            uniprot_to_row=ctx.uniprot_to_row,
            X_cov_by_group=ctx.X_cov_by_group,
            B=B,
            min_n_per_group=ctx.min_n_per_group,
            min_valid_perms=min_valid_perms,
            group_order=ctx.group_order,
        )
        results.append(r)
        if checkpoint_path is not None:
            append_checkpoint(checkpoint_path, r)
        finite_p = r.p_values[np.isfinite(r.p_values)]
        min_p_str = f"{finite_p.min():.4f}" if finite_p.size else "n/a"
        logger.info(
            f"Anchor {work.anchor_uniprot}: B={B}, "
            f"n_edges={len(work.edge_ids)}, "
            f"min(n_valid)={int(B - r.n_degenerate_per_edge.max())}, "
            f"min(p)={min_p_str}, "
            f"elapsed={time.time() - t:.1f}s"
        )
    return results
