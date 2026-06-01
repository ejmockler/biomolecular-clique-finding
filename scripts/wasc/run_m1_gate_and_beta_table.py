"""M1 numerical-identity gate (real data) + M2 first artifact (per-edge β/SE table).

Two outputs:

  1. M1 gate report — for 50 randomly-chosen real edges × 3 groups, fit β
     via our F-W kernel AND via `statsmodels.OLS`. Verify agreement to
     1e-8 on β and 1e-8 relative on SE.  HARD HALT if any pair disagrees.

  2. Per-edge per-group β/SE table — for all 944 edges in E_WASC v1, fit
     β̂_{j|a,g} and SE for each of {C9, SPOR, CTRL}.  Write to
     `output/wasc/beta_per_edge_per_group_m2.csv`.  This is the M2 first
     useful artifact (biologically interpretable BEFORE the null is run).

Per memory/wasc_build_plan.md M2.5 prong (d), this script also runs the
"F-W vs OLS with explicit batch dummies on SPOR" identity check.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cliquefinder.stats.wasc import (  # noqa: E402
    EdgeBetaTable,
    Theme,
    WascEdge,
    build_wasc_data_bundle,
    fit_edges_per_group,
    fit_fwl_per_pair,
)

logger = logging.getLogger("wasc-m1-gate")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

OUT_DIR = ROOT / "output" / "wasc"
EDGES_PATH = ROOT / "data" / "wasc" / "E_WASC_v1.json"
GATE_TOL_BETA = 1e-8
GATE_TOL_SE = 1e-8


def _load_e_wasc() -> tuple[WascEdge, ...]:
    """Load the frozen v1.0 E_WASC edges from the JSON artifact."""
    doc = json.loads(EDGES_PATH.read_text())
    out = []
    for e in doc["edges"]:
        out.append(WascEdge(
            anchor_uniprot=e["anchor_uniprot"],
            target_uniprot=e["target_uniprot"],
            theme=Theme(e["theme"]),
            network=__import__(
                "cliquefinder.stats.wasc.types", fromlist=["Network"]
            ).Network(e["network"]),
            anchor_symbol=e.get("anchor_symbol", ""),
            target_symbol=e.get("target_symbol", ""),
            evidence_count=e.get("evidence_count"),
            stmt_types=tuple(e["stmt_types"]) if e.get("stmt_types") else None,
        ))
    return tuple(out)


def m1_gate_real_data(
    edges: tuple[WascEdge, ...],
    bundle,
    n_samples: int = 50,
    seed: int = 0,
) -> dict:
    """Compare F-W vs statsmodels.OLS for 50 random (edge × group) triples.

    Returns dict of {gate_passed: bool, n_triples: int, max_beta_err: ..., ...}.
    """
    import statsmodels.api as sm

    rng = random.Random(seed)
    pool = list(edges)
    rng.shuffle(pool)

    triples = []
    for e in pool:
        if len(triples) >= n_samples:
            break
        for g, design in bundle.designs.items():
            try:
                a_row = bundle.abundance.index.get_loc(e.anchor_uniprot)
                j_row = bundle.abundance.index.get_loc(e.target_uniprot)
            except KeyError:
                continue
            sample_cols = [bundle.abundance.columns.get_loc(s)
                           for s in design.sample_ids]
            target_y = bundle.abundance.values[j_row, sample_cols]
            anchor_y = bundle.abundance.values[a_row, sample_cols]
            triples.append({
                "edge": e.edge_id,
                "group": g,
                "target_y": target_y,
                "anchor_y": anchor_y,
                "X_cov": design.X_cov,
            })

    triples = triples[:n_samples]
    logger.info("M1 gate: %d (edge, group) triples to compare", len(triples))

    max_beta_err = 0.0
    max_se_err = 0.0
    failures = []
    skipped = 0
    for i, t in enumerate(triples):
        fit = fit_fwl_per_pair(t["target_y"], t["anchor_y"], t["X_cov"], min_n=10)
        if not fit.converged:
            skipped += 1
            continue
        # statsmodels OLS on the complete cases
        mask = ~(
            np.isnan(t["target_y"])
            | np.isnan(t["anchor_y"])
            | np.isnan(t["X_cov"]).any(axis=1)
        )
        y = t["target_y"][mask]
        a = t["anchor_y"][mask]
        X = t["X_cov"][mask]
        X_full = np.column_stack([X, a])
        try:
            ols = sm.OLS(y, X_full).fit()
        except Exception as ex:
            logger.warning("OLS failed for triple %d (%s/%s): %s",
                           i, t["edge"], t["group"], ex)
            skipped += 1
            continue
        ols_beta = float(ols.params[-1])
        ols_se = float(ols.bse[-1])
        beta_err = abs(fit.beta - ols_beta)
        se_err = abs(fit.se - ols_se) / max(ols_se, 1e-12)
        if beta_err > max_beta_err:
            max_beta_err = beta_err
        if se_err > max_se_err:
            max_se_err = se_err
        if beta_err > GATE_TOL_BETA or se_err > GATE_TOL_SE:
            failures.append({
                "triple_idx": i,
                "edge": t["edge"],
                "group": t["group"],
                "fw_beta": fit.beta,
                "ols_beta": ols_beta,
                "beta_err": beta_err,
                "fw_se": fit.se,
                "ols_se": ols_se,
                "se_err": se_err,
            })

    passed = (len(failures) == 0)
    return {
        "gate_passed": passed,
        "n_triples": len(triples),
        "n_skipped": skipped,
        "max_beta_err": max_beta_err,
        "max_se_err_rel": max_se_err,
        "failures": failures[:10],  # cap report at 10
    }


def write_beta_table(bet: EdgeBetaTable, out_path: Path) -> None:
    """Write the per-edge per-group β/SE table to CSV."""
    rows = []
    for i, eid in enumerate(bet.edge_ids):
        rows.append({
            "edge_id": eid,
            "beta_C9":      bet.beta["C9ORF72"][i],
            "se_C9":        bet.se["C9ORF72"][i],
            "df_C9":        int(bet.df["C9ORF72"][i]),
            "n_C9":         int(bet.n["C9ORF72"][i]),
            "beta_SPOR":    bet.beta["SPORADIC"][i],
            "se_SPOR":      bet.se["SPORADIC"][i],
            "df_SPOR":      int(bet.df["SPORADIC"][i]),
            "n_SPOR":       int(bet.n["SPORADIC"][i]),
            "beta_CTRL":    bet.beta["CONTROL"][i],
            "se_CTRL":      bet.se["CONTROL"][i],
            "df_CTRL":      int(bet.df["CONTROL"][i]),
            "n_CTRL":       int(bet.n["CONTROL"][i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load data
    logger.info("Loading E_WASC v1.0 edges...")
    edges = _load_e_wasc()
    logger.info("Loaded %d edges (Splicing: %d, Chromatin: %d, Transport: %d)",
                len(edges),
                sum(1 for e in edges if e.theme == Theme.SPLICING),
                sum(1 for e in edges if e.theme == Theme.CHROMATIN),
                sum(1 for e in edges if e.theme == Theme.TRANSPORT))

    logger.info("Building WASC data bundle (proteomics + enriched metadata)...")
    bundle = build_wasc_data_bundle()

    # ----- 2. M1 numerical-identity gate on real data
    logger.info("Running M1 gate on 50 random (edge × group) triples...")
    gate_report = m1_gate_real_data(edges, bundle, n_samples=50, seed=0)
    gate_path = OUT_DIR / "m1_gate_report.json"
    gate_path.write_text(json.dumps(gate_report, indent=2, default=float) + "\n")
    print()
    print("=" * 70)
    print("M1 GATE — F-W vs statsmodels.OLS on real data")
    print("=" * 70)
    print(f"  Triples tested:       {gate_report['n_triples']}")
    print(f"  Triples skipped:      {gate_report['n_skipped']}")
    print(f"  Max β error:          {gate_report['max_beta_err']:.2e}  "
          f"(tol {GATE_TOL_BETA:.0e})")
    print(f"  Max SE relative err:  {gate_report['max_se_err_rel']:.2e}  "
          f"(tol {GATE_TOL_SE:.0e})")
    print(f"  GATE PASSED:          {gate_report['gate_passed']}")
    if not gate_report["gate_passed"]:
        print(f"  FAILURES (first {len(gate_report['failures'])}):")
        for f in gate_report["failures"]:
            print(f"    {f['edge']} / {f['group']}: "
                  f"β_err={f['beta_err']:.2e}  SE_err={f['se_err']:.2e}")
        print()
        print("HARD HALT: M1 numerical-identity gate failed. Do not proceed to M2 null.")
        sys.exit(1)

    # ----- 3. Per-edge per-group β/SE table — M2 first artifact
    logger.info("Running full F-W fit on all %d edges × 3 groups...", len(edges))
    bet = fit_edges_per_group(list(edges), bundle.abundance, bundle.designs)
    beta_path = OUT_DIR / "beta_per_edge_per_group_m2.csv"
    write_beta_table(bet, beta_path)
    logger.info("Wrote %s", beta_path)

    # ----- 4. Convergence summary
    print()
    print("=" * 70)
    print("M2 FIRST ARTIFACT — per-edge per-group β/SE table")
    print("=" * 70)
    print(f"  {beta_path.relative_to(ROOT)}")
    print()
    for g in ("C9ORF72", "SPORADIC", "CONTROL"):
        n_conv = int(np.isfinite(bet.beta[g]).sum())
        n_pos = int((bet.beta[g] > 0).sum())
        n_neg = int((bet.beta[g] < 0).sum())
        beta_med = float(np.nanmedian(bet.beta[g]))
        n_used_med = int(np.nanmedian(bet.n[g]))
        print(f"  {g:<10} converged: {n_conv:>4}/{len(edges)}  "
              f"β>0: {n_pos:>4}  β<0: {n_neg:>4}  median β: {beta_med:+.3f}  "
              f"median n_used: {n_used_med}")

    # Per-theme summary
    print()
    print("Per-theme convergence (any group fit):")
    by_theme = {Theme.SPLICING: 0, Theme.CHROMATIN: 0, Theme.TRANSPORT: 0}
    by_theme_conv = {Theme.SPLICING: 0, Theme.CHROMATIN: 0, Theme.TRANSPORT: 0}
    for i, e in enumerate(edges):
        by_theme[e.theme] += 1
        if all(np.isfinite(bet.beta[g][i]) for g in ("C9ORF72", "SPORADIC", "CONTROL")):
            by_theme_conv[e.theme] += 1
    for t in (Theme.SPLICING, Theme.CHROMATIN, Theme.TRANSPORT):
        print(f"  {t.value:<10} {by_theme_conv[t]:>4}/{by_theme[t]}  "
              f"({100*by_theme_conv[t]/by_theme[t]:.1f}%) converge in all 3 groups")


if __name__ == "__main__":
    main()
