"""M2.5 prong (d) — F-W vs statsmodels.OLS numerical-identity on PRODUCTION design.

Pre-registered DoD (locked_bounds_v1.json::prong_d_FW_vs_OLS_tolerance):
    |β_FW - β_OLS|  and  SE relative diff  must each be <= 1e-8
    across all sampled (edge, group) triples fit on the PRODUCTION
    GroupDesign (intercept + sex_female + age_z + tissue dummies — the
    same X_cov the F-W kernel consumes in M2.2 and downstream).

This script is the on-disk re-stamp of the prong-(d) computation that
previously ran only as an inline /tmp script in workflow wuynarvdj.
Numbers from that earlier ad-hoc run were β abs 2.39e-10, SE rel 3.14e-10
— this script produces a versioned, repo-rooted artifact that any later
brutalist can locate by path alone.

Procedure
---------
1. Load WascDataBundle (real proteomics + enriched metadata + GroupDesigns).
2. Load the frozen E_WASC_v1.json (944 edges).
3. Sample 50 edges with numpy default_rng(seed=42).
4. For each (edge, group) triple where both endpoints are measured and the
   group's complete-case n >= min_n_per_group (10 for C9ORF72, 15 otherwise):
       - F-W: fit_fwl_per_pair(target_y, anchor_y, X_cov)
       - OLS: statsmodels.OLS(target_y_kept, [X_cov_kept, anchor_y_kept]).fit()
   Apply the SAME per-pair complete-case mask used by the F-W kernel so
   both fits operate on identical row subsets.
5. Record per-triple |β_FW - β_OLS|, |SE_FW - SE_OLS|, |SE_FW - SE_OLS|/SE_OLS.
   Triples with non-converged F-W (collinearity, n < min_n) are excluded
   from the identity comparison and counted under `n_skipped_unconverged`.
6. Compute max over all converged triples + the 5 worst per metric.
7. Write output/wasc/m2_5_prong_d/result.json.

The pre-registered bound is 1e-8 on β absolute and SE relative
(matching test_fit.py::test_50_random_triples_agree_to_1e8 which uses
absolute on β and relative on SE; the locked_bounds description text
says "absolute tolerance ... on β and SE" but the on-record observed
SE diff of 3.14e-10 is relative — the gate is checked against both to
remove any ambiguity).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, (REPO / "src").as_posix())

from cliquefinder.stats.wasc.fit import fit_fwl_per_pair  # noqa: E402
from cliquefinder.stats.wasc.preprocess import build_wasc_data_bundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m2.5.prong_d")

OUT = REPO / "output" / "wasc" / "m2_5_prong_d"
OUT.mkdir(parents=True, exist_ok=True)

# Per spec §2.3 / fit.py::fit_edges_per_group default.
MIN_N_PER_GROUP = {"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15}

# Pre-registered bound (locked_bounds_v1.json::prong_d_FW_vs_OLS_tolerance).
PRONG_D_BOUND = 1e-8


def _fit_ols_anchor(
    target_y: np.ndarray,
    anchor_y: np.ndarray,
    X_cov: np.ndarray,
    min_n: int,
) -> dict | None:
    """Run statsmodels.OLS on the per-pair complete-case subset (same mask
    as fit_fwl_per_pair) and return the anchor coefficient + SE + df.

    Returns None if the post-mask design fails the same conditions the
    F-W kernel uses (n < min_n, df <= 0, or non-finite anchor).
    """
    import statsmodels.api as sm

    nan_mask = (
        np.isnan(target_y)
        | np.isnan(anchor_y)
        | np.isnan(X_cov).any(axis=1)
    )
    keep = ~nan_mask
    n = int(keep.sum())
    p_cov = X_cov.shape[1]
    p_total = p_cov + 1
    df = n - p_total
    if n < min_n or df <= 0:
        return None

    y = target_y[keep]
    a = anchor_y[keep]
    X = X_cov[keep]
    X_full = np.column_stack([X, a])
    try:
        res = sm.OLS(y, X_full).fit()
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("statsmodels.OLS failed: %s", exc)
        return None
    return {
        "beta": float(res.params[-1]),
        "se": float(res.bse[-1]),
        "df": int(res.df_resid),
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-edges", type=int, default=50,
                        help="Number of edges to sample (default 50; spec uses 50 real triples).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    log.info("=== M2.5 prong (d) — F-W vs statsmodels.OLS production-design identity ===")
    log.info(f"n_edges={args.n_edges}, seed={args.seed}, bound={PRONG_D_BOUND:.0e}")

    # --- Load data ------------------------------------------------------
    bundle = build_wasc_data_bundle()
    abundance = bundle.abundance
    designs = bundle.designs
    log.info("Production design columns per group:")
    for g, d in designs.items():
        log.info(f"  {g}: n={len(d.sample_ids)}, p_cov={d.X_cov.shape[1]}, "
                 f"cols={d.column_names}")

    edges_doc = json.loads((REPO / "data" / "wasc" / "E_WASC_v1.json").read_text())
    all_edges = edges_doc["edges"]
    log.info(f"E_WASC_v1.json: |E|={len(all_edges)} edges loaded")

    # --- Sample 50 edges -----------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.n_edges >= len(all_edges):
        sampled_idx = np.arange(len(all_edges))
    else:
        sampled_idx = rng.choice(len(all_edges), size=args.n_edges, replace=False)
    sampled_edges = [all_edges[int(i)] for i in sampled_idx]

    # --- Per-(edge, group) fit pairs -----------------------------------
    abundance_index = abundance.index
    uniprot_to_row: dict[str, int] = {}
    for e in sampled_edges:
        for up in (e["anchor_uniprot"], e["target_uniprot"]):
            if up not in uniprot_to_row:
                try:
                    uniprot_to_row[up] = abundance_index.get_loc(up)
                except KeyError:
                    uniprot_to_row[up] = -1

    sample_index = {
        g: [abundance.columns.get_loc(s) for s in d.sample_ids]
        for g, d in designs.items()
    }
    A = abundance.values  # rows=proteins, cols=samples

    triples: list[dict] = []
    n_skipped_unmeasured = 0
    n_skipped_unconverged = 0
    n_skipped_ols_failed = 0
    for e in sampled_edges:
        a_up = e["anchor_uniprot"]
        t_up = e["target_uniprot"]
        a_row = uniprot_to_row[a_up]
        t_row = uniprot_to_row[t_up]
        if a_row < 0 or t_row < 0:
            n_skipped_unmeasured += len(designs)
            continue
        for g, d in designs.items():
            cols = sample_index[g]
            target_y = A[t_row, cols]
            anchor_y = A[a_row, cols]
            X = d.X_cov
            min_n = MIN_N_PER_GROUP.get(g, 10)

            fw = fit_fwl_per_pair(target_y, anchor_y, X, min_n=min_n)
            if not fw.converged:
                n_skipped_unconverged += 1
                continue
            ols = _fit_ols_anchor(target_y, anchor_y, X, min_n=min_n)
            if ols is None:
                n_skipped_ols_failed += 1
                continue
            abs_beta = abs(fw.beta - ols["beta"])
            abs_se = abs(fw.se - ols["se"])
            rel_se = abs_se / max(abs(ols["se"]), 1e-300)
            triples.append({
                "edge_id":   e["edge_id"],
                "anchor":    a_up,
                "target":    t_up,
                "theme":     e["theme"],
                "group":     g,
                "n":         int(fw.n),
                "df":        int(fw.df),
                "beta_fw":   float(fw.beta),
                "beta_ols":  float(ols["beta"]),
                "se_fw":     float(fw.se),
                "se_ols":    float(ols["se"]),
                "abs_beta_diff": float(abs_beta),
                "abs_se_diff":   float(abs_se),
                "rel_se_diff":   float(rel_se),
            })

    if not triples:
        log.error("No converged triples — cannot evaluate prong (d). Aborting.")
        return 2

    abs_beta_arr = np.array([t["abs_beta_diff"] for t in triples])
    abs_se_arr = np.array([t["abs_se_diff"] for t in triples])
    rel_se_arr = np.array([t["rel_se_diff"] for t in triples])

    max_abs_beta = float(abs_beta_arr.max())
    max_abs_se = float(abs_se_arr.max())
    max_rel_se = float(rel_se_arr.max())
    median_abs_beta = float(np.median(abs_beta_arr))
    median_rel_se = float(np.median(rel_se_arr))

    # Pre-registered gate: BOTH metrics under 1e-8.
    # The on-record convention (test_fit.py + workflow wuynarvdj report)
    # uses beta-absolute and SE-relative.
    pass_against_bound = bool(max_abs_beta < PRONG_D_BOUND and max_rel_se < PRONG_D_BOUND)
    # Also report the stricter SE-absolute check for completeness.
    pass_against_bound_se_absolute_too = bool(max_abs_se < PRONG_D_BOUND)

    # Worst-5 by each metric (for diagnostic surface).
    by_abs_beta = sorted(triples, key=lambda t: -t["abs_beta_diff"])[:5]
    by_rel_se = sorted(triples, key=lambda t: -t["rel_se_diff"])[:5]
    worst_triples = {
        "by_abs_beta_diff_top5": by_abs_beta,
        "by_rel_se_diff_top5": by_rel_se,
    }

    out_doc = {
        "schema_version": "v1.0",
        "prong": "d",
        "description": (
            "F-W (fit_fwl_per_pair) vs statsmodels.OLS numerical identity "
            "on the PRODUCTION GroupDesign (intercept + sex_female + age_z "
            "+ tissue dummies). 50 random edges × 3 groups; per-triple "
            "complete-case mask applied identically to both fits."
        ),
        "config": {
            "n_edges_sampled":  int(len(sampled_edges)),
            "seed":             int(args.seed),
            "min_n_per_group":  MIN_N_PER_GROUP,
            "bound":            PRONG_D_BOUND,
            "bound_metrics":    ["max_abs_beta_diff", "max_rel_se_diff"],
        },
        "production_design": {
            g: {
                "n_samples": len(d.sample_ids),
                "p_cov":     int(d.X_cov.shape[1]),
                "columns":   list(d.column_names),
            }
            for g, d in designs.items()
        },
        "n_triples":              int(len(triples)),
        "n_skipped_unmeasured":   int(n_skipped_unmeasured),
        "n_skipped_unconverged":  int(n_skipped_unconverged),
        "n_skipped_ols_failed":   int(n_skipped_ols_failed),
        "max_abs_beta_diff":      max_abs_beta,
        "max_abs_se_diff":        max_abs_se,
        "max_rel_se_diff":        max_rel_se,
        "median_abs_beta_diff":   median_abs_beta,
        "median_rel_se_diff":     median_rel_se,
        "pass_against_1e-8_bound": pass_against_bound,
        "pass_against_1e-8_bound_se_absolute_too": pass_against_bound_se_absolute_too,
        "worst_triples":          worst_triples,
        "wall_clock_seconds":     float(time.time() - t0),
    }

    # Per-config naming (hygiene fix h1 pattern, applied to prong-d so a
    # later re-run with a different --n-edges / --seed cannot silently
    # overwrite this PASS artifact).  result.latest.json is a convenience
    # pointer to the most recent run; not authoritative.
    per_config_name = f"result.n{int(args.n_edges)}_seed{int(args.seed)}.json"
    out_path = OUT / per_config_name
    if out_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {out_path}.  Delete or rename the "
            f"existing file if a fresh run is intended."
        )
    out_path.write_text(json.dumps(out_doc, indent=2))
    (OUT / "result.latest.json").write_text(json.dumps({
        "latest_run_path": str(out_path.relative_to(REPO)),
        "wrote_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    log.info("Wrote %s (per-config naming)", out_path)
    log.info("=== M2.5 prong (d) summary ===")
    log.info("  n_triples         : %d", out_doc["n_triples"])
    log.info("  max |β_FW - β_OLS|: %.3e (bound %.0e)", max_abs_beta, PRONG_D_BOUND)
    log.info("  max rel SE diff   : %.3e (bound %.0e)", max_rel_se, PRONG_D_BOUND)
    log.info("  max abs SE diff   : %.3e", max_abs_se)
    log.info("  pass (β-abs + SE-rel both < 1e-8): %s", pass_against_bound)
    log.info("  pass (also SE-abs < 1e-8)        : %s", pass_against_bound_se_absolute_too)
    log.info("  wall-clock        : %.1fs", out_doc["wall_clock_seconds"])
    return 0 if pass_against_bound else 1


if __name__ == "__main__":
    sys.exit(main())
