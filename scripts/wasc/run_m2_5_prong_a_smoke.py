"""M2.5 prong (a) — label-shuffle null calibration smoke.

Small-scale wiring + calibration plausibility check.  Default: 5 shuffles
× B=99 across all ~300 anchors.  Estimated wall-clock: ~5 min single-core
based on M2.4 full B=100 timing (~20s for B=100 → ~20s × (99/100) per
shuffle × 5 = ~100s).

This is NOT the production calibration (which is 20 shuffles × B=999,
per spec Gate 2 / M2.5 prong (a)).  Smoke purpose:

  1. Validate the shuffle infrastructure end-to-end on real data.
  2. Read the pooled FP rate at a budget that's tractable for iteration.
  3. Compare against the spec Gate 2 bound:
       mean_FP_rate ≤ 0.10 + 2·√(0.10·0.90/|edges|) ≈ 0.114 at n_edges≈1888

If pooled mean FP rate looks far from 0.10 (e.g., > 0.20), the wiring
or calibration has a problem and the production run is blocked until
debugged.

Output: output/wasc/m2_5_prong_a_smoke/
  result.{candidate_pool}_b{B}_n{n_shuffles}_seed{shuffle_seed}.json

Per-config naming is mandatory (hygiene fix h1): different
(candidate_pool, B, n_shuffles, shuffle_seed) configurations write to
distinct files so a later sensitivity re-stamp can NEVER silently
overwrite a primary production result.  A 'result.latest.json' pointer
file is also written for convenience (records the path of the most
recent run; not authoritative).
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
from cliquefinder.stats.wasc.null import AnchorWork, anchor_seed  # noqa: E402
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402
from cliquefinder.stats.wasc.sanity import run_label_shuffle_calibration  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.5.prong_a.smoke")


OUT = REPO / "output" / "wasc" / "m2_5_prong_a_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def per_config_result_filename(
    *,
    candidate_pool: str,
    B: int,
    n_shuffles: int,
    shuffle_seed: int,
) -> str:
    """Compute per-config result filename (hygiene fix h1).

    Different (candidate_pool, B, n_shuffles, shuffle_seed) tuples MUST
    write to distinct files so a sensitivity re-stamp cannot silently
    overwrite a primary production result.

    Format: result.{candidate_pool}_b{B}_n{n_shuffles}_seed{shuffle_seed}.json
    """
    if candidate_pool not in ("theme", "all"):
        raise ValueError(
            f"candidate_pool must be 'theme' or 'all', got {candidate_pool!r}"
        )
    if B < 1 or n_shuffles < 1:
        raise ValueError(
            f"B and n_shuffles must be >= 1 (got B={B}, n_shuffles={n_shuffles})"
        )
    return (
        f"result.{candidate_pool}"
        f"_b{int(B)}"
        f"_n{int(n_shuffles)}"
        f"_seed{int(shuffle_seed)}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--B", type=int, default=99)
    parser.add_argument("--min-valid-perms", type=int, default=20)
    parser.add_argument("--p-threshold", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N anchors (for ultra-fast iteration)")
    parser.add_argument("--candidate-pool", choices=["theme", "all"], default="all",
                        help="all: v1.0.3 canonical primary (full measured proteome). "
                             "theme: v1.0.2 substrate — RETAINED AS SENSITIVITY "
                             "with known FAILED calibration. "
                             "Default: all (v1.0.3 primary).")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Master seed for label-shuffle calibration. "
                             "Distinct seeds write to distinct output files "
                             "(per-config naming, hygiene fix h1).")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== M2.5 prong (a) — label-shuffle calibration smoke ===")
    log.info(f"n_shuffles={args.n_shuffles}, B={args.B}, p_threshold={args.p_threshold}")

    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs

    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())

    # CANONICAL-DIRECTION CONVENTION (spec §1/M1, build_plan §3):
    # WascEdges are oriented anchor < target.  M2.2 fits each edge ONCE
    # in canonical direction; null/Brown's must match.
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
            # Per-anchor union of M_T across the anchor's themes (8.4% of
            # anchors are multi-theme).  This is a single-AnchorBins
            # approximation to the canonical per-(anchor, theme) pool.
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
    result = run_label_shuffle_calibration(
        works_template=works,
        anchor_bins_by_anchor=bins_by_anchor,
        abundance=abundance,
        designs=designs,
        uniprot_to_row=uniprot_to_row,
        n_shuffles=args.n_shuffles,
        B=args.B,
        p_threshold=args.p_threshold,
        min_valid_perms=args.min_valid_perms,
        shuffle_seed=args.shuffle_seed,
        global_salt="wasc-v1.0.2-shuffle-smoke",
        verbose=True,
    )
    calib_dt = time.time() - t_calib

    log.info("\n=== M2.5 prong (a) smoke result ===")
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

    # Persist (per-config path — hygiene fix h1)
    result_filename = per_config_result_filename(
        candidate_pool=args.candidate_pool,
        B=args.B,
        n_shuffles=args.n_shuffles,
        shuffle_seed=args.shuffle_seed,
    )
    out_path = OUT / result_filename
    out_doc = {
        "candidate_pool": args.candidate_pool,
        "shuffle_seed": int(args.shuffle_seed),
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
        "extrapolated_production_minutes": float(
            calib_dt / args.n_shuffles * 20 * (999 / args.B) / 60
        ),
        "n_anchors": len(works),
        "n_edges_total": int(n_edges_total),
    }
    if out_path.exists():
        # Per-config files MUST be immutable once written: if the same
        # (candidate_pool, B, n_shuffles, shuffle_seed) tuple has already
        # been recorded, refuse to overwrite.  Operator must move/delete
        # the existing artifact explicitly.
        raise FileExistsError(
            f"Refusing to overwrite existing per-config artifact: {out_path}. "
            "Per-config naming (hygiene fix h1) requires distinct configs to "
            "produce distinct files; identical configs must not silently "
            "re-stamp. Move or delete the existing file to re-run."
        )
    out_path.write_text(json.dumps(out_doc, indent=2))
    # Non-authoritative pointer to the most recent run (records which
    # per-config artifact was just written; safe to overwrite).
    latest_pointer = {
        "latest_result_path": out_path.name,
        "candidate_pool": args.candidate_pool,
        "B": int(args.B),
        "n_shuffles": int(args.n_shuffles),
        "shuffle_seed": int(args.shuffle_seed),
    }
    (OUT / "result.latest.json").write_text(json.dumps(latest_pointer, indent=2))
    log.info(f"\nWrote {out_path}")
    log.info(f"Updated pointer: {OUT / 'result.latest.json'}")
    log.info(f"Extrapolated production (20 shuffles × B=999): "
             f"~{out_doc['extrapolated_production_minutes']:.1f} min single-core")
    log.info(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
