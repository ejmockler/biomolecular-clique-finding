"""M2.5 prong (a) — K=2 sweep via monkey-patch on compute_anchor_null.

Runs the theme-restricted label-shuffle calibration with
``min_unique_q_values=2`` (very lenient — just guards against fully
CONSTANT Q_null cases, allowing edges with 2+ distinct null draws).

WHY K=2 (vs the K=5 default committed in null.py):
  The path-debug audit identified sparse-cell sampling as producing
  constant Q_null → deterministic lower-tail p (0.01 or 1.0).
  K=5 is the spec; K=2 is the minimal sufficient guard if the bias is
  PURELY from the K=1 (single-value) pathology.  If K=2 returns FP to
  ~0.10, the bias is purely the constant-Q_null case.  If K=2 still
  inflates FP while K=5 passes, the bias extends to near-constant
  Q_null (cells with 2-3 candidates draw the same 2-3 values
  repeatedly).

Monkey-patch path: identical to the K=10 agent's approach.
``compute_anchor_null`` is rebound on the ``sanity`` module to force
``min_unique_q_values=2`` regardless of caller default.

Output: output/wasc/m2_5_prong_a_smoke_k2/result.json (separate dir to
preserve the K=5 baseline result.json).
"""
from __future__ import annotations

import argparse
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
from cliquefinder.stats.wasc import null as null_mod  # noqa: E402
from cliquefinder.stats.wasc import sanity as sanity_mod  # noqa: E402
from cliquefinder.stats.wasc.null import AnchorWork, anchor_seed  # noqa: E402
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.5.prong_a.k2")


OUT = REPO / "output" / "wasc" / "m2_5_prong_a_smoke_k2"
OUT.mkdir(parents=True, exist_ok=True)


K_OVERRIDE = 2


def _patched_compute_anchor_null(*args, **kwargs):
    """Force min_unique_q_values=K_OVERRIDE regardless of caller default."""
    kwargs["min_unique_q_values"] = K_OVERRIDE
    return _ORIG_COMPUTE_ANCHOR_NULL(*args, **kwargs)


_ORIG_COMPUTE_ANCHOR_NULL = null_mod.compute_anchor_null


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--B", type=int, default=99)
    parser.add_argument("--min-valid-perms", type=int, default=20)
    parser.add_argument("--p-threshold", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-pool", choices=["theme", "all"], default="theme")
    parser.add_argument("--k", type=int, default=2,
                        help="min_unique_q_values value to inject via monkey-patch.")
    args = parser.parse_args()

    # Reassign module-level override if caller passes --k
    global K_OVERRIDE
    K_OVERRIDE = int(args.k)

    # Patch the symbol that sanity.run_label_shuffle_calibration imported.
    # The function `run_label_shuffle_calibration` calls `compute_anchor_null`
    # via the name bound at module import: `from .null import compute_anchor_null`
    # → we rebind on `sanity_mod`.
    sanity_mod.compute_anchor_null = _patched_compute_anchor_null

    t0 = time.time()
    log.info("=== M2.5 prong (a) K=%d monkey-patch smoke ===", K_OVERRIDE)
    log.info(f"n_shuffles={args.n_shuffles}, B={args.B}, "
             f"p_threshold={args.p_threshold}, candidate_pool={args.candidate_pool}")
    log.info(f"compute_anchor_null is forced to min_unique_q_values={K_OVERRIDE}")

    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs

    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())

    anchor_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    anchor_themes_set: dict[str, set[str]] = defaultdict(set)
    for e in edges_doc["edges"]:
        anchor_targets[e["anchor_uniprot"]].append((e["edge_id"], e["target_uniprot"]))
        anchor_themes_set[e["anchor_uniprot"]].add(e["theme"])

    all_anchors = sorted(anchor_targets.keys())
    if args.limit:
        all_anchors = all_anchors[:args.limit]
    log.info(f"Anchors to process: {len(all_anchors)}")

    degrees = load_measured_degrees()
    uniprot_to_row = {p: i for i, p in enumerate(abundance.index)}

    log.info(f"Building AnchorBins (candidate_pool={args.candidate_pool})...")
    if args.candidate_pool == "theme":
        cluster_doc = json.loads((REPO / "data" / "wasc" / "cluster_members_v1.json").read_text())
        m_t: dict[str, set[str]] = {
            theme: set(tdata["measured_uniprots"])
            for theme, tdata in cluster_doc["themes"].items()
        }
        log.info(f"  M_T sizes: {{ {', '.join(f'{t}={len(s)}' for t, s in m_t.items())} }}")
    t_bins = time.time()
    bins_by_anchor = {}
    for a in all_anchors:
        if args.candidate_pool == "theme":
            eligible = set().union(*[m_t[t] for t in anchor_themes_set[a]])
        else:
            eligible = None
        bins_by_anchor[a] = build_anchor_bins(a, abundance, degrees,
                                              eligible_proteins=eligible)
    log.info(f"  built in {time.time() - t_bins:.1f}s")

    works = []
    for a in all_anchors:
        targets = anchor_targets[a]
        edge_ids = tuple(e for (e, _) in targets)
        true_targets = tuple(t for (_, t) in targets)
        Q_obs = np.array([obs_by_edge.get(e, np.nan) for e in edge_ids], dtype=np.float64)
        works.append(AnchorWork(
            anchor_uniprot=a, edge_ids=edge_ids, true_targets=true_targets,
            Q_obs=Q_obs,
            seed=anchor_seed(a, global_salt="wasc-v1.0.2-shuffle-smoke"),
        ))
    n_edges_total = sum(len(w.edge_ids) for w in works)
    log.info(f"Total true edges across anchors: {n_edges_total}")

    log.info("Starting calibration...")
    t_calib = time.time()
    result = sanity_mod.run_label_shuffle_calibration(
        works_template=works,
        anchor_bins_by_anchor=bins_by_anchor,
        abundance=abundance,
        designs=designs,
        uniprot_to_row=uniprot_to_row,
        n_shuffles=args.n_shuffles,
        B=args.B,
        p_threshold=args.p_threshold,
        min_valid_perms=args.min_valid_perms,
        shuffle_seed=42,
        global_salt="wasc-v1.0.2-shuffle-smoke",
        verbose=True,
    )
    calib_dt = time.time() - t_calib

    log.info("\n=== M2.5 prong (a) K=%d smoke result ===", K_OVERRIDE)
    log.info(f"  n_shuffles completed: {result.n_shuffles}")
    log.info(f"  B per shuffle       : {result.B}")
    log.info(f"  p_threshold         : {result.p_threshold}")
    log.info(f"  per-shuffle FP rate : {result.fp_rate_per_shuffle}")
    log.info(f"  mean FP rate        : {result.mean_fp_rate:.4f}")
    log.info(f"  spec Gate 2 bound   : {result.bound:.4f}")
    log.info(f"  pooled pass         : {result.pooled_pass}")
    log.info(f"  per-shuffle n_finite: {result.per_shuffle_n_finite_p}")
    log.info(f"  total calib time    : {calib_dt:.1f}s "
             f"({calib_dt / args.n_shuffles:.1f}s per shuffle)")

    out_doc = {
        "config": f"theme + K={K_OVERRIDE} (monkey-patched)",
        "min_unique_q_values": K_OVERRIDE,
        "candidate_pool": args.candidate_pool,
        "n_shuffles_requested": args.n_shuffles,
        "n_shuffles_completed": int(result.n_shuffles),
        "B": int(result.B),
        "p_threshold": float(result.p_threshold),
        "fp_rate_per_shuffle": [
            float(x) if np.isfinite(x) else None
            for x in result.fp_rate_per_shuffle
        ],
        "mean_fp_rate": float(result.mean_fp_rate) if np.isfinite(result.mean_fp_rate) else None,
        "bound": float(result.bound),
        "pooled_pass": bool(result.pooled_pass),
        "per_shuffle_n_finite_p": [int(x) for x in result.per_shuffle_n_finite_p],
        "wall_clock_seconds": float(calib_dt),
        "n_anchors": len(works),
        "n_edges_total": int(n_edges_total),
    }
    (OUT / "result.json").write_text(json.dumps(out_doc, indent=2))
    log.info(f"\nWrote {OUT / 'result.json'}")
    log.info(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
