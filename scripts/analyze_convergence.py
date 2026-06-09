"""Wave 24l convergence diagnostic — per-anchor "natural max_hops".

Loads landscape result.json files (from unbounded BFS runs) and
reports:
  - Per-anchor distribution of deepest hop reached
  - Cumulative reach fraction at h=1, 2, 3, 4, 5+
  - "Convergence hop" per anchor: smallest h where adding the next
    shell increases reach by < 5% (or < 10 features)
  - Correlation of convergence hop with anchor degree (when graph
    degrees are available in the meta sidecar)

This is the empirical answer to "what's INDRA-regulatory-on-measured
diameter on this proteome?"  Under bounded h=2 every anchor's
deepest hop is 2 (trivial); under unbounded the result is informative.

Run:
    .venv/bin/python scripts/analyze_convergence.py \
        --result-dir output/landscape_proteome_measured_only_unbounded \
        --out-dir   output/landscape_convergence_c9spor

Outputs:
  <out-dir>/per_anchor_shells.csv  — one row per anchor
  <out-dir>/summary.json           — population summary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy-free for simplicity)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--epsilon-frac", type=float, default=0.05,
        help="Convergence: marginal reach < this fraction of current "
             "cumulative reach.  Default 0.05.",
    )
    parser.add_argument(
        "--epsilon-abs", type=int, default=10,
        help="Convergence: marginal reach < this many features.  "
             "Default 10.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("convergence")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.result_dir / "result.json"
    log.info("Loading %s", result_path)
    data = json.loads(result_path.read_text())
    pf = data["per_feature"]
    log.info("per_feature: %d anchors", len(pf))

    rows: list[dict] = []
    max_hops_seen = 0
    for r in pf:
        shells = r.get("shells", []) or []
        if not shells:
            continue
        shells_sorted = sorted(shells, key=lambda s: s["hop"])
        max_hops_seen = max(
            max_hops_seen, shells_sorted[-1]["hop"],
        )
        cum_reach: list[int] = []
        cum = 0
        for s in shells_sorted:
            cum += s["n_genes"]
            cum_reach.append(cum)
        deepest_hop = shells_sorted[-1]["hop"]
        total_reach = cum_reach[-1]
        # Convergence hop: smallest h where shell n_genes / cum_reach
        # < epsilon_frac OR shell n_genes < epsilon_abs.
        conv_hop = deepest_hop
        for idx, s in enumerate(shells_sorted):
            if idx == 0:
                continue
            cum_at_h = cum_reach[idx]
            margin = s["n_genes"]
            if (
                margin < args.epsilon_abs
                or (cum_at_h > 0 and margin / cum_at_h < args.epsilon_frac)
            ):
                conv_hop = s["hop"]
                break

        row = {
            "seed": r["seed"],
            "slope": r.get("slope"),
            "slope_pvalue": r.get("slope_pvalue"),
            "n_genes_total": r.get("n_genes_total", total_reach),
            "deepest_hop": deepest_hop,
            "convergence_hop": conv_hop,
            "total_reach": total_reach,
        }
        for h in range(1, 8):
            in_shell = next(
                (s for s in shells_sorted if s["hop"] == h), None,
            )
            row[f"n_hop{h}"] = in_shell["n_genes"] if in_shell else 0
            row[f"cum_hop{h}"] = (
                cum_reach[h - 1]
                if h - 1 < len(cum_reach)
                else (cum_reach[-1] if cum_reach else 0)
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "per_anchor_shells.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d anchors)", out_csv, len(df))

    log.info("==== Per-anchor hop summary ====")
    log.info("Max hop observed across any anchor: %d", max_hops_seen)
    deepest_dist = Counter(df["deepest_hop"].tolist())
    for h in sorted(deepest_dist):
        log.info(
            "  deepest_hop=%d : %d anchors (%.1f%%)",
            h, deepest_dist[h], 100 * deepest_dist[h] / len(df),
        )
    conv_dist = Counter(df["convergence_hop"].tolist())
    log.info("---- Convergence hop (epsilon=%.2f frac OR %d abs) ----",
             args.epsilon_frac, args.epsilon_abs)
    for h in sorted(conv_dist):
        log.info(
            "  conv_hop=%d : %d anchors (%.1f%%)",
            h, conv_dist[h], 100 * conv_dist[h] / len(df),
        )

    log.info("---- Cumulative reach distribution ----")
    for h in range(1, max(max_hops_seen + 1, 5)):
        col = f"cum_hop{h}"
        if col not in df.columns:
            continue
        vals = df[col].dropna().tolist()
        if not vals:
            continue
        log.info(
            "  cum_hop%d : min=%d  p25=%.0f  median=%.0f  "
            "p75=%.0f  max=%d",
            h, int(min(vals)),
            _percentile(vals, 0.25),
            _percentile(vals, 0.50),
            _percentile(vals, 0.75),
            int(max(vals)),
        )

    summary = {
        "n_anchors": int(len(df)),
        "max_hops_observed": int(max_hops_seen),
        "deepest_hop_distribution": {
            int(h): int(c) for h, c in deepest_dist.items()
        },
        "convergence_hop_distribution": {
            int(h): int(c) for h, c in conv_dist.items()
        },
        "epsilon_frac": float(args.epsilon_frac),
        "epsilon_abs": int(args.epsilon_abs),
    }
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_json)
    log.info("CONVERGENCE_DONE")


if __name__ == "__main__":
    main()
