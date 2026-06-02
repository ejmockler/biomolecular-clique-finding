"""M3 smoke — Brown's combination + BY-FDR on the M2.4 full B=100 results.

Re-runs the full B=100 null loop (fast: ~20s) to get the per-anchor
AnchorNullResult objects in memory, then applies:

  1. compute_brown_per_anchor → per-anchor (chi2, df, c, p_brown, p_fisher)
  2. by_fdr on the per-edge raw p-values → per-edge BY-q at alpha=0.10
  3. by_fdr on per-anchor p_brown → per-anchor BY-q

Produces output/wasc/m3_smoke/{
  per_edge.csv,
  per_anchor.csv,
  summary.json,
}

This is a SMOKE pass — B=100 is too coarse for BY-FDR at q=0.10 to fire
meaningfully (effective threshold ≈ 0.013; raw-p floor is 1/(B+1) ≈ 0.01).
It validates the M3 wiring end-to-end and produces the table schema we'll
use at production (B=9999).
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

from cliquefinder.stats.wasc.bins import build_anchor_bins, load_measured_degrees  # noqa: E402
from cliquefinder.stats.wasc.combination import (  # noqa: E402
    by_fdr,
    compute_brown_per_anchor,
)
from cliquefinder.stats.wasc.null import (  # noqa: E402
    AnchorWork, NullLoopContext, anchor_seed, run_null_serial,
)
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m3.smoke")

OUT = REPO / "output" / "wasc" / "m3_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--min-valid-perms", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== M3 smoke — Brown's + BY-FDR ===")

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

    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())
    anchor_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges_doc["edges"]:
        anchor_targets[e["anchor_uniprot"]].append((e["edge_id"], e["target_uniprot"]))
        anchor_targets[e["target_uniprot"]].append((e["edge_id"], e["anchor_uniprot"]))

    all_anchors = sorted(anchor_targets.keys())
    degrees = load_measured_degrees()

    log.info("Building AnchorBins + AnchorWorks...")
    t_b = time.time()
    bins_by_anchor = {a: build_anchor_bins(a, abundance, degrees) for a in all_anchors}
    works = []
    for a in all_anchors:
        targets = anchor_targets[a]
        edge_ids = tuple(e for (e, _) in targets)
        true_targets = tuple(t for (_, t) in targets)
        Q_obs = np.array([obs_by_edge.get(e, np.nan) for e in edge_ids], dtype=np.float64)
        works.append(AnchorWork(
            anchor_uniprot=a, edge_ids=edge_ids, true_targets=true_targets,
            Q_obs=Q_obs, seed=anchor_seed(a, global_salt=f"wasc-v1.0.2-m3-b{args.B}"),
        ))
    log.info(f"  built in {time.time() - t_b:.1f}s")

    ctx = NullLoopContext(
        abundance_by_group=abundance_by_group,
        sample_index_by_group=sample_index_by_group,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=X_cov_by_group,
        min_n_per_group={"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15},
        group_order=group_order,
    )

    log.info(f"Running null loop B={args.B} across {len(works)} anchors...")
    t_run = time.time()
    results = run_null_serial(
        works=works, anchor_bins_by_anchor=bins_by_anchor, ctx=ctx,
        B=args.B, min_valid_perms=args.min_valid_perms,
        checkpoint_path=None,  # in-memory only
    )
    run_dt = time.time() - t_run
    log.info(f"  null loop: {run_dt:.1f}s")

    log.info("Computing per-anchor Brown's combination...")
    t_brown = time.time()
    brown_table = compute_brown_per_anchor(results)
    brown_dt = time.time() - t_brown
    log.info(f"  Brown's: {brown_dt:.1f}s ({brown_dt / len(results) * 1000:.1f}ms per anchor)")

    log.info("Applying BY-FDR...")
    # Per-edge table
    per_edge_rows = []
    for r in results:
        for i, eid in enumerate(r.edge_ids):
            per_edge_rows.append({
                "edge_id": eid,
                "anchor": r.anchor_uniprot,
                "Q_obs": float(r.Q_obs[i]) if np.isfinite(r.Q_obs[i]) else np.nan,
                "p_raw": float(r.p_values[i]) if np.isfinite(r.p_values[i]) else np.nan,
                "n_degenerate": int(r.n_degenerate_per_edge[i]),
            })
    per_edge_df = pd.DataFrame(per_edge_rows).drop_duplicates(subset=["edge_id"])
    rejected_e, q_e = by_fdr(per_edge_df["p_raw"].values, alpha=args.alpha)
    per_edge_df["q_by"] = q_e
    per_edge_df["rejected"] = rejected_e

    # Per-anchor table
    per_anchor_df = pd.DataFrame({
        "anchor": brown_table.anchors,
        "n_edges": brown_table.n_edges,
        "chi2_obs": brown_table.chi2_obs,
        "df": brown_table.df,
        "c": brown_table.c,
        "p_brown": brown_table.p_brown,
        "p_fisher": brown_table.p_fisher,
    })
    rejected_a, q_a = by_fdr(per_anchor_df["p_brown"].values, alpha=args.alpha)
    per_anchor_df["q_by"] = q_a
    per_anchor_df["rejected"] = rejected_a

    per_edge_df.to_csv(OUT / "per_edge.csv", index=False)
    per_anchor_df.to_csv(OUT / "per_anchor.csv", index=False)

    # Summary
    log.info("\n=== Summary ===")
    log.info(f"  per-edge rows (deduped):    {len(per_edge_df)}")
    log.info(f"  edges with finite p_raw:    {int(per_edge_df['p_raw'].notna().sum())}")
    log.info(f"  edges rejected at q<={args.alpha}: {int(rejected_e.sum())}")
    log.info(f"  per-anchor rows:            {len(per_anchor_df)}")
    log.info(f"  anchors with finite p_brown:{int(per_anchor_df['p_brown'].notna().sum())}")
    log.info(f"  anchors rejected at q<={args.alpha}: {int(rejected_a.sum())}")
    log.info("")
    log.info("  Per-anchor c distribution (>1 = positive dependence):")
    finite_c = per_anchor_df["c"][per_anchor_df["c"].notna()].values
    if len(finite_c):
        log.info(f"    n={len(finite_c)}, min={finite_c.min():.3f}, "
                 f"median={float(np.median(finite_c)):.3f}, "
                 f"p75={float(np.quantile(finite_c, 0.75)):.3f}, "
                 f"max={finite_c.max():.3f}")
        log.info(f"    fraction with c > 1.5: {(finite_c > 1.5).mean():.2%}")
    log.info("")
    log.info("  Per-anchor df shrinkage (df / (2*n_edges) = 1 ⇒ Fisher):")
    finite_df = per_anchor_df[["df", "n_edges"]].dropna()
    if len(finite_df):
        shrink = finite_df["df"] / (2.0 * finite_df["n_edges"])
        log.info(f"    n={len(shrink)}, min={shrink.min():.3f}, "
                 f"median={float(np.median(shrink)):.3f}, "
                 f"max={shrink.max():.3f}")

    summary = {
        "B": args.B,
        "alpha": args.alpha,
        "n_edges_total": int(len(per_edge_df)),
        "n_edges_finite_p": int(per_edge_df["p_raw"].notna().sum()),
        "n_edges_rejected": int(rejected_e.sum()),
        "n_anchors": int(len(per_anchor_df)),
        "n_anchors_finite_pbrown": int(per_anchor_df["p_brown"].notna().sum()),
        "n_anchors_rejected": int(rejected_a.sum()),
        "wall_clock_seconds": {
            "null_loop": float(run_dt),
            "brown": float(brown_dt),
            "total": float(time.time() - t0),
        },
        "per_anchor_c_quantiles": (
            {
                f"p{int(q*100):02d}": float(np.quantile(finite_c, q))
                for q in [0.05, 0.25, 0.50, 0.75, 0.95]
            } if len(finite_c) else None
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"\nWrote {OUT}")
    log.info(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
