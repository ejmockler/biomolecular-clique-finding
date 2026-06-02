"""M2.4 full B=100 sanity pass — all anchors in E_WASC, v1.0.2 axes default.

After the 2-anchor smoke (`run_m2_4_smoke.py`) returned 0 degenerate
permutations and ~0.05s wall-clock per anchor, this scales to ALL anchors
in E_WASC to validate:

  1. Per-anchor timing distribution + total wall-clock at full scale.
  2. n_degenerate distribution: does any anchor's matched-cell sampler
     hit bin-empty edge cases under the v1.0.2 2-axis default?  If so,
     the §4 ±1-decile fallback policy is load-bearing and must ship
     before B=9999.  If not, the fallback is a no-op on this dataset.
  3. Per-edge p-value distribution at B=100 — informative but coarse
     (smallest resolvable p = 1/(B+1) ≈ 0.0099).
  4. Sanity Gate 7 inputs: n-eligible-candidates-per-cell distribution
     per anchor (the load-bearing diagnostic for whether 2-axis sampling
     saturates the eligible pool).

This run does NOT compute production p-values — B=100 is too coarse to
trip BY-FDR at q=0.10.  It is an instrumentation pass.

Output:
  output/wasc/m2_4_full_b100/
    full_b100.jsonl           — per-anchor null loop results
    full_b100_summary.json    — aggregate statistics
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
from cliquefinder.stats.wasc.null import (  # noqa: E402
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    load_completed_anchors,
    run_null_serial,
)
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.4.full")


OUT = REPO / "output" / "wasc" / "m2_4_full_b100"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=100,
                        help="Number of permutation iterations (default: 100)")
    parser.add_argument("--min-valid-perms", type=int, default=20,
                        help="Edge gets p=NaN if fewer than this many finite null draws (default: 20)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N anchors (default: all)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Remove existing checkpoint before running")
    parser.add_argument("--candidate-pool", choices=["theme", "all"], default="theme",
                        help="theme: spec §4 canonical (M_T per anchor's theme). "
                             "all: build-plan prong (c) sensitivity (full proteome). "
                             "Default: theme (canonical primary).")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== M2.4 full sanity pass ===")
    log.info(f"B={args.B}, min_valid_perms={args.min_valid_perms}, candidate_pool={args.candidate_pool}")

    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs

    group_order = ("C9ORF72", "SPORADIC", "CONTROL")
    col_index = {s: i for i, s in enumerate(abundance.columns)}
    A_full = abundance.values
    abundance_by_group, sample_index_by_group, X_cov_by_group = {}, {}, {}
    for g in group_order:
        d = designs[g]
        cols = np.array([col_index[s] for s in d.sample_ids if s in col_index],
                        dtype=np.int64)
        sample_index_by_group[g] = cols
        abundance_by_group[g] = A_full[:, cols]
        X_cov_by_group[g] = d.X_cov
    uniprot_to_row = {p: i for i, p in enumerate(abundance.index)}

    # Q_obs + E_WASC
    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())

    # CANONICAL-DIRECTION CONVENTION (spec §1/M1, build_plan §3):
    # WascEdges are oriented anchor < target.  M2.2 fits each edge ONCE
    # in canonical direction; per-anchor null/Brown's must match.
    #
    # THEME-RESTRICTED CANONICAL (spec §4 line 192): for each (anchor a,
    # theme T), the candidate pool is `M_T \ {a} \ N_a^obs`.  Anchors
    # with edges in multiple themes get one work unit per theme.
    # ALL-PROTEIN-POOL PRONG-C VARIANT: ignore theme; pool = full proteome.
    anchor_theme_targets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for e in edges_doc["edges"]:
        key = (e["anchor_uniprot"], e["theme"])
        anchor_theme_targets[key].append((e["edge_id"], e["target_uniprot"]))

    work_keys = sorted(anchor_theme_targets.keys())
    if args.limit:
        work_keys = work_keys[:args.limit]
    log.info(f"Will process {len(work_keys)} (anchor, theme) work units")

    degrees = load_measured_degrees()

    # Theme cluster members M_T for theme-restricted pool
    cluster_doc = json.loads((REPO / "data" / "wasc" / "cluster_members_v1.json").read_text())
    m_t: dict[str, set[str]] = {
        theme: set(tdata["measured_uniprots"])
        for theme, tdata in cluster_doc["themes"].items()
    }
    log.info(f"M_T sizes: {{ {', '.join(f'{t}={len(s)}' for t, s in m_t.items())} }}")

    log.info("Building per-(anchor, theme) AnchorBins...")
    t_bins = time.time()
    bins_by_key: dict[tuple[str, str], object] = {}
    cells_per_anchor = []
    for i, (a, theme) in enumerate(work_keys):
        if args.candidate_pool == "theme":
            eligible = m_t.get(theme)
        else:
            eligible = None  # all-protein-pool variant
        b = build_anchor_bins(a, abundance, degrees, eligible_proteins=eligible)
        bins_by_key[(a, theme)] = b
        cells_per_anchor.append(len(b.cells))
        if (i + 1) % 50 == 0:
            log.info(f"  built bins for {i + 1}/{len(work_keys)}")
    bin_dt = time.time() - t_bins
    log.info(f"Bin build complete: {len(work_keys)} units in {bin_dt:.1f}s "
             f"(avg {bin_dt / len(work_keys) * 1000:.0f}ms per unit)")

    works = []
    for (a, theme) in work_keys:
        targets = anchor_theme_targets[(a, theme)]
        edge_ids = tuple(e for (e, _) in targets)
        true_targets = tuple(t for (_, t) in targets)
        Q_obs = np.array([obs_by_edge.get(e, np.nan) for e in edge_ids], dtype=np.float64)
        # AnchorWork keyed on the anchor only — but the seed is salted with
        # theme so multi-theme anchors get independent permutation sequences.
        works.append(AnchorWork(
            anchor_uniprot=a, edge_ids=edge_ids, true_targets=true_targets,
            Q_obs=Q_obs,
            seed=anchor_seed(
                a,
                global_salt=f"wasc-v1.0.2-full-b{args.B}-pool{args.candidate_pool}-{theme}",
            ),
        ))

    ctx = NullLoopContext(
        abundance_by_group=abundance_by_group,
        sample_index_by_group=sample_index_by_group,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=X_cov_by_group,
        min_n_per_group={"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15},
        group_order=group_order,
    )

    ckpt = OUT / f"full_b100_pool{args.candidate_pool}.jsonl"
    if args.reset_checkpoint and ckpt.exists():
        log.warning(f"Removing pre-existing checkpoint {ckpt}")
        ckpt.unlink()

    log.info(f"Running null loop on {len(works)} (anchor, theme) units at B={args.B}...")
    # Manual loop to avoid the bins_by_anchor → str-key constraint, which
    # would collide for multi-theme anchors.  Pair works[i] with
    # bins_by_key[work_keys[i]] explicitly.
    from cliquefinder.stats.wasc.null import (
        append_checkpoint, compute_anchor_null, load_completed_anchors,
    )
    completed_keys = set()
    if ckpt.exists() and not args.reset_checkpoint:
        completed_keys = load_completed_anchors(ckpt)
    t_run = time.time()
    results = []
    for (work, (a, theme)) in zip(works, work_keys):
        ck_key = f"{a}|{theme}"
        if ck_key in completed_keys:
            continue
        r = compute_anchor_null(
            work=work,
            anchor_bins=bins_by_key[(a, theme)],
            abundance_by_group=ctx.abundance_by_group,
            sample_index_by_group=ctx.sample_index_by_group,
            uniprot_to_row=ctx.uniprot_to_row,
            X_cov_by_group=ctx.X_cov_by_group,
            B=args.B,
            min_n_per_group=ctx.min_n_per_group,
            min_valid_perms=args.min_valid_perms,
            group_order=ctx.group_order,
        )
        # Stamp the checkpoint key with theme so multi-theme anchors are
        # distinguishable
        r_serialized = {
            "anchor": f"{r.anchor_uniprot}|{theme}",
            "edge_ids": list(r.edge_ids),
            "Q_obs": [float(x) if np.isfinite(x) else None for x in r.Q_obs],
            "p_values": [float(x) if np.isfinite(x) else None for x in r.p_values],
            "n_degenerate": [int(x) for x in r.n_degenerate_per_edge],
        }
        with ckpt.open("a") as fh:
            fh.write(json.dumps(r_serialized, sort_keys=True) + "\n")
        results.append(r)
    run_dt = time.time() - t_run

    # Aggregate stats
    n_total_edges = sum(len(r.edge_ids) for r in results)
    finite_p_count = sum(int(np.isfinite(r.p_values).sum()) for r in results)
    all_n_deg = np.concatenate([r.n_degenerate_per_edge for r in results])
    all_p = np.concatenate([r.p_values for r in results])
    finite_p = all_p[np.isfinite(all_p)]

    log.info("\n=== Aggregate ===")
    log.info(f"  total anchors  = {len(results)}")
    log.info(f"  total edges    = {n_total_edges}")
    log.info(f"  finite p       = {finite_p_count}/{n_total_edges} "
             f"({finite_p_count / n_total_edges:.2%})")
    log.info(f"  n_deg per edge: min={all_n_deg.min()} max={all_n_deg.max()} "
             f"median={int(np.median(all_n_deg))} mean={all_n_deg.mean():.2f}")
    log.info(f"  edges with any degenerate perm: {int((all_n_deg > 0).sum())}/{n_total_edges}")
    log.info(f"  edges with > {args.B // 5} degenerate perms (Gate 7 > 20% trigger): "
             f"{int((all_n_deg > args.B // 5).sum())}/{n_total_edges}")
    if finite_p.size:
        log.info(f"  p-value distribution (finite n={finite_p.size}):")
        for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]:
            log.info(f"    p{int(q*100):02d} = {float(np.quantile(finite_p, q)):.4f}")
    log.info(f"\nWall-clock: bin build {bin_dt:.1f}s + null loop {run_dt:.1f}s "
             f"= {bin_dt + run_dt:.1f}s total")
    log.info(f"  per-anchor avg (null only): {run_dt / len(results):.3f}s")
    log.info(f"  extrapolated B=9999 single-core, {len(results)} anchors: "
             f"~{(run_dt / len(results)) * (9999 / args.B) * len(results) / 60:.1f} min")

    # Persist aggregate summary
    summary = {
        "B": args.B,
        "min_valid_perms": args.min_valid_perms,
        "v1_0_2_axes_default": ["degree", "corr"],
        "n_anchors": int(len(results)),
        "n_total_edges": int(n_total_edges),
        "finite_p_count": int(finite_p_count),
        "finite_p_fraction": float(finite_p_count / n_total_edges) if n_total_edges else None,
        "n_degenerate_per_edge_stats": {
            "min": int(all_n_deg.min()),
            "max": int(all_n_deg.max()),
            "median": float(np.median(all_n_deg)),
            "mean": float(all_n_deg.mean()),
            "p95": float(np.quantile(all_n_deg, 0.95)),
            "p99": float(np.quantile(all_n_deg, 0.99)),
            "edges_with_any_degenerate": int((all_n_deg > 0).sum()),
            "edges_above_gate7_threshold_20pct": int((all_n_deg > args.B // 5).sum()),
        },
        "p_value_quantiles_finite": {
            f"p{int(q*100):02d}": float(np.quantile(finite_p, q))
            for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
        } if finite_p.size else None,
        "cells_per_anchor_stats": {
            "min": int(min(cells_per_anchor)) if cells_per_anchor else None,
            "max": int(max(cells_per_anchor)) if cells_per_anchor else None,
            "median": float(np.median(cells_per_anchor)) if cells_per_anchor else None,
        },
        "wall_clock_seconds": {
            "bin_build_total": float(bin_dt),
            "null_loop_total": float(run_dt),
            "per_anchor_null_avg": float(run_dt / len(results)) if results else None,
        },
        "extrapolated_b9999_minutes_single_core": (
            float((run_dt / len(results)) * (9999 / args.B) * len(results) / 60)
            if results else None
        ),
    }
    (OUT / "full_b100_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"\nSummary written to {OUT / 'full_b100_summary.json'}")
    log.info(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
