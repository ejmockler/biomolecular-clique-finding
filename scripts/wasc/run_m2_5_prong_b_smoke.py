"""M2.5 prong (b) — SPORADIC down-sample to n=25 + edge-set overlap.

Per build plan §10 / spec §12: tests whether the C9-vs-SPOR contrast
pattern persists when SPOR is down-sampled to match C9's cohort size
(n=25, from n=294).  If the WASC-positive edge set is largely preserved,
the C9-driven pattern is NOT a group-size confound.  If it collapses,
primary is suspended.

Diagnostic: Jaccard overlap of "positive" edge sets between the full-N
run and the down-sampled run.  At B=999, smallest resolvable raw p is
0.001 and BY effective threshold at q=0.10 ≈ 0.013, so a small number
of BY-rejections is achievable.  Smoke uses raw p < 0.10 for a richer
diagnostic and reports BOTH thresholds.

This is a SMOKE pass — the production tripwire run uses B=9999 and
will live in the run_wasc orchestrator (M7).

Output: output/wasc/m2_5_prong_b_smoke/
    full_n_positives.csv     — edge_id + p + (BY q) for full-N run
    downsampled_positives.csv — same for SPOR-25 run
    summary.json              — Jaccard + counts + timings
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
from cliquefinder.stats.wasc.combination import by_fdr  # noqa: E402
from cliquefinder.stats.wasc.null import (  # noqa: E402
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    compute_anchor_null,
    run_null_serial,
)
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402
from cliquefinder.stats.wasc.sanity import (  # noqa: E402
    _fit_observed_q_for_works,
    downsample_group,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.5.prong_b.smoke")

OUT = REPO / "output" / "wasc" / "m2_5_prong_b_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def _build_anchor_works(
    edges_doc: dict,
    obs_by_edge: dict[str, float],
    salt: str,
) -> list[AnchorWork]:
    """Canonical-direction anchor work units."""
    anchor_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in edges_doc["edges"]:
        anchor_targets[e["anchor_uniprot"]].append(
            (e["edge_id"], e["target_uniprot"])
        )
    works = []
    for a in sorted(anchor_targets):
        targets = anchor_targets[a]
        edge_ids = tuple(eid for (eid, _) in targets)
        true_targets = tuple(t for (_, t) in targets)
        Q_obs = np.array([obs_by_edge.get(eid, np.nan) for eid in edge_ids],
                         dtype=np.float64)
        works.append(AnchorWork(
            anchor_uniprot=a, edge_ids=edge_ids, true_targets=true_targets,
            Q_obs=Q_obs, seed=anchor_seed(a, global_salt=salt),
        ))
    return works


def _run_pipeline(
    works: list[AnchorWork],
    bins_by_anchor: dict,
    ctx: NullLoopContext,
    B: int,
    min_valid_perms: int,
    alpha: float,
) -> pd.DataFrame:
    """Run null loop → per-edge DataFrame with p_raw + BY q."""
    results = run_null_serial(
        works=works, anchor_bins_by_anchor=bins_by_anchor, ctx=ctx,
        B=B, min_valid_perms=min_valid_perms, checkpoint_path=None,
    )
    rows = []
    for r in results:
        for i, eid in enumerate(r.edge_ids):
            rows.append({
                "edge_id": eid,
                "anchor": r.anchor_uniprot,
                "Q_obs": float(r.Q_obs[i]) if np.isfinite(r.Q_obs[i]) else np.nan,
                "p_raw": float(r.p_values[i]) if np.isfinite(r.p_values[i]) else np.nan,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["edge_id"])
    _, q = by_fdr(df["p_raw"].values, alpha=alpha)
    df["q_by"] = q
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=999)
    parser.add_argument("--min-valid-perms", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--spor-n", type=int, default=25,
                        help="Target SPOR size for down-sample (default: 25, matches C9)")
    parser.add_argument("--downsample-seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== M2.5 prong (b) — SPORADIC down-sample smoke ===")
    log.info(f"B={args.B}, min_valid_perms={args.min_valid_perms}, "
             f"alpha={args.alpha}, spor_n={args.spor_n}")

    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs

    obs_df = pd.read_csv(REPO / "output" / "wasc" / "concordance_per_edge_m2_2.csv")
    obs_by_edge = dict(zip(obs_df["edge_id"], obs_df["Q"]))
    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())
    degrees = load_measured_degrees()
    uniprot_to_row = {p: i for i, p in enumerate(abundance.index)}

    log.info("Building AnchorBins for canonical anchors...")
    t_b = time.time()
    works_template = _build_anchor_works(edges_doc, obs_by_edge, "wasc-v1.0.2-prongb-full")
    bins_by_anchor = {w.anchor_uniprot: build_anchor_bins(w.anchor_uniprot, abundance, degrees)
                      for w in works_template}
    log.info(f"  {len(works_template)} anchors, bins built in {time.time() - t_b:.1f}s")

    # --- FULL-N RUN ---
    log.info(f"--- FULL N run (SPOR={len(designs['SPORADIC'].sample_ids)}) ---")
    group_order = ("C9ORF72", "SPORADIC", "CONTROL")
    col_index = {s: i for i, s in enumerate(abundance.columns)}
    A_full = abundance.values
    full_sample_index, full_abundance_by_group, full_X_cov_by_group = {}, {}, {}
    for g in group_order:
        d = designs[g]
        cols = np.array([col_index[s] for s in d.sample_ids if s in col_index],
                        dtype=np.int64)
        full_sample_index[g] = cols
        full_abundance_by_group[g] = A_full[:, cols]
        full_X_cov_by_group[g] = d.X_cov
    full_ctx = NullLoopContext(
        abundance_by_group=full_abundance_by_group,
        sample_index_by_group=full_sample_index,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=full_X_cov_by_group,
        min_n_per_group={"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15},
        group_order=group_order,
    )
    t_full = time.time()
    full_df = _run_pipeline(works_template, bins_by_anchor, full_ctx,
                            B=args.B, min_valid_perms=args.min_valid_perms,
                            alpha=args.alpha)
    full_dt = time.time() - t_full
    log.info(f"  full-N pipeline: {full_dt:.1f}s, {len(full_df)} edges")
    full_df.to_csv(OUT / "full_n_positives.csv", index=False)

    # --- DOWN-SAMPLED RUN ---
    log.info(f"--- DOWN-SAMPLED SPOR (n={args.spor_n}) run ---")
    rng_ds = np.random.default_rng(args.downsample_seed)
    ds_designs, ds_sample_index, ds_abundance_by_group = downsample_group(
        designs, abundance, "SPORADIC", args.spor_n, rng_ds,
    )
    ds_X_cov_by_group = {g: d.X_cov for g, d in ds_designs.items()}
    log.info(f"  SPORADIC samples after down-sample: {len(ds_designs['SPORADIC'].sample_ids)}")
    log.info(f"  SPORADIC X_cov shape: {ds_X_cov_by_group['SPORADIC'].shape}")

    # CRITICAL: re-fit observed Q under the down-sampled context.
    t_obsq = time.time()
    ds_obs_q = _fit_observed_q_for_works(
        works_template, ds_abundance_by_group, ds_X_cov_by_group,
        uniprot_to_row,
        # Loosen C9 / CTRL floors since their n hasn't changed; keep SPOR
        # floor honest at the down-sampled size.
        min_n_per_group={
            "C9ORF72": 10, "SPORADIC": min(15, args.spor_n - 5), "CONTROL": 15,
        },
    )
    log.info(f"  observed Q refit: {time.time() - t_obsq:.1f}s")

    # Re-derive AnchorWorks with down-sampled-context Q_obs.  Re-seed via
    # a distinct salt so the down-sampled null draws are an independent
    # permutation sequence (not the same draws as the full-N null).
    ds_works = [
        AnchorWork(
            anchor_uniprot=w.anchor_uniprot,
            edge_ids=w.edge_ids,
            true_targets=w.true_targets,
            Q_obs=ds_obs_q[w.anchor_uniprot],
            seed=anchor_seed(w.anchor_uniprot,
                             global_salt=f"wasc-v1.0.2-prongb-ds{args.spor_n}-s{args.downsample_seed}"),
        )
        for w in works_template
    ]
    ds_ctx = NullLoopContext(
        abundance_by_group=ds_abundance_by_group,
        sample_index_by_group=ds_sample_index,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=ds_X_cov_by_group,
        min_n_per_group={
            "C9ORF72": 10, "SPORADIC": min(15, args.spor_n - 5), "CONTROL": 15,
        },
        group_order=group_order,
    )
    t_ds = time.time()
    ds_df = _run_pipeline(ds_works, bins_by_anchor, ds_ctx,
                          B=args.B, min_valid_perms=args.min_valid_perms,
                          alpha=args.alpha)
    ds_dt = time.time() - t_ds
    log.info(f"  down-sampled pipeline: {ds_dt:.1f}s, {len(ds_df)} edges")
    ds_df.to_csv(OUT / "downsampled_positives.csv", index=False)

    # --- COMPARISON ---
    log.info("\n=== Overlap diagnostics ===")

    def positives_at(df: pd.DataFrame, p_col: str, threshold: float) -> set[str]:
        mask = (df[p_col] < threshold) & df[p_col].notna()
        return set(df.loc[mask, "edge_id"])

    # Raw p < 0.10
    full_raw = positives_at(full_df, "p_raw", 0.10)
    ds_raw = positives_at(ds_df, "p_raw", 0.10)
    inter_raw = full_raw & ds_raw
    union_raw = full_raw | ds_raw
    jaccard_raw = len(inter_raw) / len(union_raw) if union_raw else float("nan")
    log.info(f"  Raw p<0.10: full={len(full_raw)}, "
             f"down={len(ds_raw)}, intersection={len(inter_raw)}, "
             f"Jaccard={jaccard_raw:.4f}")

    # BY q <= alpha
    full_by = positives_at(full_df, "q_by", args.alpha)
    ds_by = positives_at(ds_df, "q_by", args.alpha)
    inter_by = full_by & ds_by
    union_by = full_by | ds_by
    jaccard_by = len(inter_by) / len(union_by) if union_by else float("nan")
    log.info(f"  BY q<={args.alpha}: full={len(full_by)}, "
             f"down={len(ds_by)}, intersection={len(inter_by)}, "
             f"Jaccard={jaccard_by:.4f}")

    # Per-theme + per-anchor breakdown (raw p<0.10)
    edge_to_theme = {e["edge_id"]: e["theme"] for e in edges_doc["edges"]}
    log.info("  Raw-p positives per theme:")
    for theme in ("Splicing", "Chromatin", "Transport"):
        f_theme = [e for e in full_raw if edge_to_theme.get(e) == theme]
        d_theme = [e for e in ds_raw if edge_to_theme.get(e) == theme]
        inter_theme = set(f_theme) & set(d_theme)
        log.info(f"    {theme}: full={len(f_theme)}, down={len(d_theme)}, "
                 f"intersection={len(inter_theme)}")

    summary = {
        "B": args.B,
        "alpha": args.alpha,
        "spor_n_downsampled": args.spor_n,
        "spor_n_original": len(designs["SPORADIC"].sample_ids),
        "downsample_seed": args.downsample_seed,
        "n_edges_full": len(full_df),
        "n_edges_downsampled": len(ds_df),
        "raw_p_lt_010": {
            "full_count": len(full_raw),
            "downsampled_count": len(ds_raw),
            "intersection": len(inter_raw),
            "union": len(union_raw),
            "jaccard": jaccard_raw if not np.isnan(jaccard_raw) else None,
        },
        f"by_q_le_{args.alpha}": {
            "full_count": len(full_by),
            "downsampled_count": len(ds_by),
            "intersection": len(inter_by),
            "union": len(union_by),
            "jaccard": jaccard_by if not np.isnan(jaccard_by) else None,
        },
        "wall_clock_seconds": {
            "full_n_pipeline": float(full_dt),
            "downsampled_pipeline": float(ds_dt),
            "total": float(time.time() - t0),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"\nWrote {OUT}")
    log.info(f"Total elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
