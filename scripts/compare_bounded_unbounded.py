"""Wave 24l comparison: bounded h=2 vs unbounded BFS slope-GSEA.

Side-by-side report for the 8 pre-registered cluster terms across
three contrasts.  Loads confirmatory_8terms_robust.csv (or all/) from
both bounded and unbounded runs and writes a unified comparison.

Decision rule (per goal):
  >= 6/8 pass in both C9 AND 0/8 spctrl under unbounded → depth-invariant
  fewer than bounded → bounded h=2 is the right operating point
  more than bounded → bounded was conservative; switch to unbounded

Run:
    .venv/bin/python scripts/compare_bounded_unbounded.py \
        --bounded-dir   output/landscape_confirmatory_{c9spor,c9ctrl,spctrl}_measured_only \
        --unbounded-dir output/landscape_confirmatory_{c9spor,c9ctrl,spctrl}_measured_only_unbounded \
        --out output/wave_24l_bounded_vs_unbounded.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


CONTRASTS = ["c9spor", "c9ctrl", "spctrl"]


def _load_confirmatory(
    dirs_template: str, contrast: str, scope: str,
) -> pd.DataFrame | None:
    path = Path(
        dirs_template.replace("{c9spor,c9ctrl,spctrl}", contrast)
    ) / f"confirmatory_8terms_{scope}.csv"
    if not path.exists():
        logging.getLogger("compare").warning("Missing %s", path)
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bounded-dir", required=True,
        help="Template with {c9spor,c9ctrl,spctrl} placeholder",
    )
    parser.add_argument(
        "--unbounded-dir", required=True,
        help="Template with {c9spor,c9ctrl,spctrl} placeholder",
    )
    parser.add_argument("--scope", default="robust",
                        choices=("robust", "all"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("compare")

    lines: list[str] = []
    lines.append(
        "# Wave 24l bounded h=2 vs unbounded BFS — confirmatory comparison\n"
    )
    lines.append(f"Scope: `{args.scope}`\n")

    pass_counts = {}
    for contrast in CONTRASTS:
        bounded = _load_confirmatory(args.bounded_dir, contrast, args.scope)
        unbounded = _load_confirmatory(
            args.unbounded_dir, contrast, args.scope,
        )
        if bounded is None or unbounded is None:
            lines.append(
                f"## {contrast}\n\n_Missing inputs; skipped._\n"
            )
            continue
        merged = bounded.merge(
            unbounded,
            on=("cluster", "db", "term_id", "term"),
            suffixes=("_h2", "_unb"),
        )
        b_pass = int(bounded["bonferroni_pass"].sum())
        u_pass = int(unbounded["bonferroni_pass"].sum())
        pass_counts[contrast] = (b_pass, u_pass)
        lines.append(f"## {contrast} — bounded {b_pass}/8 vs unbounded {u_pass}/8\n")
        cols = [
            "cluster", "term", "NES_h2", "raw_p_h2", "bonferroni_pass_h2",
            "NES_unb", "raw_p_unb", "bonferroni_pass_unb",
        ]
        present_cols = [c for c in cols if c in merged.columns]
        lines.append(
            merged[present_cols].to_markdown(index=False, floatfmt=".4f"),
        )
        lines.append("")

    # Verdict per goal decision rule.
    lines.append("## Verdict\n")
    c9_pairs = [pass_counts.get(c, (0, 0)) for c in ("c9spor", "c9ctrl")]
    sp = pass_counts.get("spctrl", (0, 0))
    b_min_c9 = min(p[0] for p in c9_pairs)
    u_min_c9 = min(p[1] for p in c9_pairs)

    if u_min_c9 >= 6 and sp[1] == 0:
        verdict = (
            "**Depth-invariant.** Unbounded passes ≥6/8 in both C9 contrasts "
            f"(c9spor={pass_counts['c9spor'][1]}/8, c9ctrl={pass_counts['c9ctrl'][1]}/8) "
            f"and 0/8 in spctrl.  Prefer parameter-free unbounded for parsimony."
        )
    elif u_min_c9 < b_min_c9:
        verdict = (
            f"**Bounded h=2 is the right operating point.** Bonferroni-8 "
            f"passes drop under unbounded (c9 min: {b_min_c9} → {u_min_c9}); "
            f"deeper shells dilute the signal.  Report unbounded as a "
            f"sensitivity-analysis footnote."
        )
    elif u_min_c9 > b_min_c9:
        verdict = (
            f"**Bounded h=2 was conservative.** Bonferroni-8 passes increase "
            f"under unbounded (c9 min: {b_min_c9} → {u_min_c9}); switch to "
            f"unbounded as the primary regime."
        )
    else:
        verdict = (
            f"**Tied.** Same Bonferroni-8 outcome under both regimes "
            f"(c9 min: {b_min_c9}/8).  Prefer parameter-free unbounded for "
            f"parsimony."
        )
    lines.append(verdict)
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    log.info("Wrote %s", args.out)
    log.info("Pass counts: %s", pass_counts)
    log.info("Verdict: %s", verdict.split(".")[0] + ".")


if __name__ == "__main__":
    main()
