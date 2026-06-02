"""M2.4 entry smoke — 2 anchors, B=100, real data, per v1.0.2 axes default.

After commit 4e04b79 / tag `wasc-prereg-v1.0.2`, the brutalist V2
no-Q-exposure gate is released: the null loop may now consume real data.

This is the FIRST real-data null computation in the WASC pipeline.

Acceptance criteria (entry smoke; full-run criteria are in §12):
  - Both anchors complete without exception.
  - At least one edge yields a finite per-edge p-value (>= min_valid_perms
    finite null draws).
  - Per-anchor wall-clock extrapolates to a feasible full run at B=9999
    (~< 1 hour per anchor on single core; joblib parallel target ~10x).
  - Per-anchor checkpoint JSONL is written and resumable.

If any of the above fails, halt and debug before scaling.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, (REPO / "src").as_posix())

from cliquefinder.stats.wasc.bins import (  # noqa: E402
    build_anchor_bins,
    load_measured_degrees,
)
from cliquefinder.stats.wasc.null import (  # noqa: E402
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    load_completed_anchors,
    run_null_serial,
)
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.4.smoke")


SMOKE_OUT = REPO / "output" / "wasc" / "m2_4_smoke"
SMOKE_OUT.mkdir(parents=True, exist_ok=True)


def pick_smoke_anchors(edges_doc: dict, n: int = 2,
                       min_targets: int = 5, max_targets: int = 10) -> list[str]:
    """Pick `n` anchors whose true-target degree lies in [min_targets, max_targets].

    Tests both small and medium per-anchor compute footprints in one pass.
    Deterministic: returns the first matches sorted by UniProt.
    """
    edges = edges_doc["edges"]
    anchor_targets: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        a, t = e["anchor_uniprot"], e["target_uniprot"]
        anchor_targets[a].add(t)
        anchor_targets[t].add(a)
    eligible = [
        a for a, ts in sorted(anchor_targets.items())
        if min_targets <= len(ts) <= max_targets
    ]
    if len(eligible) < n:
        raise RuntimeError(
            f"Insufficient eligible anchors in [{min_targets},{max_targets}]: "
            f"found {len(eligible)}, need {n}."
        )
    return eligible[:n]


def main() -> int:
    t0 = time.time()
    log.info("=== M2.4 entry smoke ===")
    log.info("v1.0.2 axes default = (degree, corr). 3-axis disabled.")

    # Load WASC bundle (abundance + designs)
    log.info("Loading WASC data bundle...")
    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs
    log.info(f"  abundance shape: {abundance.shape}")
    for g, d in designs.items():
        log.info(f"  design {g}: n={len(d.sample_ids)}")

    # Pre-extract per-group abundance views and X_cov
    group_order = ("C9ORF72", "SPORADIC", "CONTROL")
    sample_index_by_group: dict[str, np.ndarray] = {}
    abundance_by_group: dict[str, np.ndarray] = {}
    X_cov_by_group: dict[str, np.ndarray] = {}
    A_full = abundance.values
    col_index = {s: i for i, s in enumerate(abundance.columns)}
    for g in group_order:
        d = designs[g]
        cols = np.array([col_index[s] for s in d.sample_ids if s in col_index],
                        dtype=np.int64)
        sample_index_by_group[g] = cols
        abundance_by_group[g] = A_full[:, cols]
        X_cov_by_group[g] = d.X_cov
        log.info(f"  per-group {g}: abundance slice {abundance_by_group[g].shape}, "
                 f"X_cov {X_cov_by_group[g].shape}")

    uniprot_to_row = {p: i for i, p in enumerate(abundance.index)}

    # Load Q_obs and edges
    log.info("Loading observed Q (from M2.2) + E_WASC...")
    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())

    # Build anchor → list of (edge_id, target) — both directions of each edge
    # CANONICAL-DIRECTION CONVENTION (spec §1/M1, build_plan §3):
    # Each WascEdge has anchor = lex-smaller endpoint and is fit ONCE per
    # M2.2's EdgeBetaTable.  The per-anchor null + Brown's combination
    # therefore process each edge in its canonical direction only.
    # Iterating both directions here would compare an anchor=v null against
    # the anchor=u canonical observed Q — direction-flipped and invalid.
    anchor_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges_doc["edges"]:
        anchor_targets[e["anchor_uniprot"]].append((e["edge_id"], e["target_uniprot"]))

    # Pick anchors
    smoke_anchors = pick_smoke_anchors(edges_doc, n=2, min_targets=5, max_targets=10)
    log.info(f"Smoke anchors picked: {smoke_anchors}")
    for a in smoke_anchors:
        log.info(f"  {a}: {len(anchor_targets[a])} true within-theme edges")

    # Load degrees (v1.0.2 axis 1)
    degrees = load_measured_degrees()

    # Build per-anchor AnchorBins (v1.0.2 default: 2-axis)
    log.info("Building AnchorBins (v1.0.2 2-axis)...")
    bins_by_anchor = {}
    for a in smoke_anchors:
        t0_bin = time.time()
        b = build_anchor_bins(a, abundance, degrees)  # no missingness, no axes arg
        bins_by_anchor[a] = b
        cell_sizes = [len(v) for v in b.cells.values()]
        log.info(f"  {a}: axes={b.axes}, {len(b.cells)} cells, "
                 f"size min/median/max={min(cell_sizes)}/{int(np.median(cell_sizes))}/{max(cell_sizes)}, "
                 f"build_t={time.time() - t0_bin:.2f}s")

    # Construct AnchorWork per anchor
    works = []
    for a in smoke_anchors:
        targets = anchor_targets[a]
        edge_ids = tuple(e for (e, _) in targets)
        true_targets = tuple(t for (_, t) in targets)
        Q_obs = np.array([obs_by_edge.get(e, np.nan) for e in edge_ids], dtype=np.float64)
        works.append(AnchorWork(
            anchor_uniprot=a,
            edge_ids=edge_ids,
            true_targets=true_targets,
            Q_obs=Q_obs,
            seed=anchor_seed(a, global_salt="wasc-v1.0.2-smoke"),
        ))
        log.info(f"  AnchorWork {a}: {len(edge_ids)} edges, Q_obs finite frac = "
                 f"{np.isfinite(Q_obs).mean():.2f}")

    # Build context
    ctx = NullLoopContext(
        abundance_by_group=abundance_by_group,
        sample_index_by_group=sample_index_by_group,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=X_cov_by_group,
        min_n_per_group={"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15},
        group_order=group_order,
    )

    # Run null loop
    ckpt = SMOKE_OUT / "smoke.jsonl"
    if ckpt.exists():
        log.warning(f"Removing pre-existing smoke checkpoint {ckpt}")
        ckpt.unlink()

    B = 100
    log.info(f"Running run_null_serial with B={B}, min_valid_perms=20...")
    t_run = time.time()
    results = run_null_serial(
        works=works,
        anchor_bins_by_anchor=bins_by_anchor,
        ctx=ctx,
        B=B,
        min_valid_perms=20,
        checkpoint_path=ckpt,
    )
    run_dt = time.time() - t_run

    # Acceptance summary
    log.info("\n=== M2.4 smoke results ===")
    log.info(f"  total run time: {run_dt:.1f}s")
    log.info(f"  per-anchor avg: {run_dt / len(works):.1f}s for B={B}")
    extrap_B = 9999
    log.info(f"  extrapolation to B={extrap_B} single-core: "
             f"~{(run_dt / len(works)) * (extrap_B / B) / 60:.1f} min/anchor")

    n_finite_p = 0
    n_total = 0
    for r in results:
        n_total += len(r.p_values)
        finite = np.isfinite(r.p_values)
        n_finite_p += int(finite.sum())
        finite_p = r.p_values[finite]
        log.info(f"  anchor {r.anchor_uniprot}:")
        log.info(f"    edges                 = {len(r.p_values)}")
        log.info(f"    finite p-values       = {int(finite.sum())}/{len(r.p_values)}")
        log.info(f"    min n_degenerate      = {int(r.n_degenerate_per_edge.min())}")
        log.info(f"    max n_degenerate      = {int(r.n_degenerate_per_edge.max())}")
        if finite.sum():
            log.info(f"    p min/median/max      = "
                     f"{finite_p.min():.4f} / {float(np.median(finite_p)):.4f} / {finite_p.max():.4f}")

    # Verify checkpoint round-trip
    done = load_completed_anchors(ckpt)
    log.info(f"\nCheckpoint contains {len(done)} anchors: {sorted(done)}")
    assert len(done) == len(works), "Checkpoint missing anchors"

    # Verdict
    ok = (
        n_finite_p > 0 and
        len(done) == len(works) and
        run_dt < 600  # 10 min wall-clock budget for smoke
    )
    log.info(f"\nSmoke verdict: {'PASS' if ok else 'FAIL'} "
             f"(total elapsed {time.time() - t0:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
