"""Test the c9ctrl-collapse interpretation from wave_24l_unbounded.

Wave 24l unbounded showed C9-vs-Control's Bonferroni-8 score went
from 6/8 (bounded) to 0/8 (unbounded), with 5 of 8 terms having
NEGATIVE NES.  We interpreted this as: "regional ALS-vs-healthy
network perturbation at deeper hops swamps the local cluster signal."

That's interpretation, not test.  Falsifier:
  - Compute per-anchor slope_delta = unbounded_slope - bounded_slope
  - Identify cluster-member anchors (union of HGNC IDs in the 8 terms)
  - Compare slope_delta distribution: cluster vs degree-matched non-cluster

If the regional-noise hypothesis is correct:
  - All anchors should shift toward positive slope (regional perturbation
    at h=3,4,5 is higher than h=1 typically)
  - Cluster anchors with strongly-negative bounded slopes have the most
    to lose → bigger positive shifts
  - But the SHIFT MAGNITUDE shouldn't be cluster-specific after
    controlling for the bounded slope's magnitude

If something cluster-specific is broken:
  - Cluster anchors degrade beyond what their bounded slope magnitude
    predicts
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats


CLUSTER_TERMS = [
    ("go", "GO:0005643", "nuclear pore"),
    ("go", "GO:0006913", "nucleocytoplasmic transport"),
    ("reactome", "R-HSA-180910", "Vpr-mediated nuclear import of PICs"),
    ("reactome", "R-HSA-72172", "mRNA Splicing"),
    ("reactome", "R-HSA-72203", "Processing of Capped Intron-Containing Pre-mRNA"),
    ("go", "GO:0000398", "mRNA splicing, via spliceosome"),
    ("go", "GO:0005694", "chromosome"),
    ("go", "GO:0000785", "chromatin"),
]


def load_slopes(path: Path) -> dict[str, dict]:
    """Return {seed_uniprot: {slope, n_hop1, n_total}}."""
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for r in data["per_feature"]:
        shells = r.get("shells", []) or []
        n_hop1 = next(
            (s["n_genes"] for s in shells if s["hop"] == 1), 0,
        )
        out[r["seed"]] = {
            "slope": float(r["slope"]),
            "n_hop1": int(n_hop1),
            "n_total": int(r.get("n_genes_total", 0)),
            "shells": shells,
        }
    return out


def get_cluster_members_uniprot() -> set[str]:
    """Fetch the union of HGNC IDs in the 8 pre-registered terms via
    the INDRA CoGEx ``[:associated_with]`` edge, then map to UniProt."""
    from pathlib import Path as _Path
    from cliquefinder.knowledge.cogex import CoGExClient
    from indra.databases import hgnc_client, uniprot_client

    hgnc_ids: set[str] = set()
    q = """
    MATCH (g:BioEntity)-[:associated_with]->(t:BioEntity {id: $id})
    WHERE g.id STARTS WITH 'hgnc:'
    RETURN DISTINCT g.id AS hgnc_id, g.name AS name
    """
    with CoGExClient(env_file=_Path(".env")) as c:
        for db, term_id, name in CLUSTER_TERMS:
            if db == "go":
                cogex_id = f"go:{term_id.lower().replace('go:', '')}"
            elif db == "reactome":
                cogex_id = f"reactome:{term_id}"
            else:
                continue
            rows = c._execute_query(q, id=cogex_id)
            for row in rows:
                raw = row[0]
                hgnc_id = raw.replace("hgnc:", "") if raw else ""
                if hgnc_id:
                    hgnc_ids.add(hgnc_id)
            logging.info("term %s (%s): %d members (cum total HGNC %d)",
                         cogex_id, name, len(rows), len(hgnc_ids))

    uniprots: set[str] = set()
    unmapped = 0
    for hgnc_id in hgnc_ids:
        # hgnc_client returns a comma-separated string for some entries.
        up_raw = hgnc_client.get_uniprot_id(hgnc_id)
        if up_raw:
            for up in str(up_raw).split(","):
                up = up.strip()
                if up:
                    uniprots.add(up)
        else:
            unmapped += 1
    logging.info("Mapped %d HGNC → %d UniProts (%d unmapped)",
                 len(hgnc_ids), len(uniprots), unmapped)
    return uniprots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--unbounded", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("c9ctrl-collapse-test")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading bounded slopes from %s", args.bounded)
    bounded = load_slopes(args.bounded)
    log.info("Loading unbounded slopes from %s", args.unbounded)
    unbounded = load_slopes(args.unbounded)

    common = sorted(set(bounded) & set(unbounded))
    log.info("Common anchors (in both): %d", len(common))

    log.info("Fetching cluster-member UniProts from INDRA...")
    try:
        cluster_uniprots = get_cluster_members_uniprot()
    except Exception as exc:
        log.error("INDRA fetch failed: %s.  Falling back to GSEA leading-edge "
                  "approach not implemented; aborting.", exc)
        raise
    log.info("Cluster-member UniProts: %d", len(cluster_uniprots))

    rows: list[dict] = []
    for seed in common:
        b = bounded[seed]
        u = unbounded[seed]
        rows.append({
            "seed": seed,
            "is_cluster": seed in cluster_uniprots,
            "bounded_slope": b["slope"],
            "unbounded_slope": u["slope"],
            "slope_delta": u["slope"] - b["slope"],
            "abs_loss": abs(b["slope"]) - abs(u["slope"]),
            "n_hop1": b["n_hop1"],
            "n_total_bounded": b["n_total"],
            "n_total_unbounded": u["n_total"],
        })
    df = pd.DataFrame(rows)
    n_cluster = int(df["is_cluster"].sum())
    log.info("Anchors in cluster intersection (measured ∩ cluster terms): %d",
             n_cluster)

    out_csv = args.out_dir / "per_anchor_collapse.csv"
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    cluster_df = df[df["is_cluster"]]
    nc_df = df[~df["is_cluster"]]

    log.info("==== bounded_slope (h=2) ====")
    log.info("Cluster   (n=%d): mean=%.4f  median=%.4f  std=%.4f",
             len(cluster_df), cluster_df["bounded_slope"].mean(),
             cluster_df["bounded_slope"].median(),
             cluster_df["bounded_slope"].std())
    log.info("Non-clus  (n=%d): mean=%.4f  median=%.4f  std=%.4f",
             len(nc_df), nc_df["bounded_slope"].mean(),
             nc_df["bounded_slope"].median(),
             nc_df["bounded_slope"].std())

    log.info("==== unbounded_slope ====")
    log.info("Cluster   : mean=%.4f  median=%.4f  std=%.4f",
             cluster_df["unbounded_slope"].mean(),
             cluster_df["unbounded_slope"].median(),
             cluster_df["unbounded_slope"].std())
    log.info("Non-clus  : mean=%.4f  median=%.4f  std=%.4f",
             nc_df["unbounded_slope"].mean(),
             nc_df["unbounded_slope"].median(),
             nc_df["unbounded_slope"].std())

    log.info("==== slope_delta (unbounded - bounded) ====")
    log.info("Cluster   : mean=%.4f  median=%.4f",
             cluster_df["slope_delta"].mean(),
             cluster_df["slope_delta"].median())
    log.info("Non-clus  : mean=%.4f  median=%.4f",
             nc_df["slope_delta"].mean(),
             nc_df["slope_delta"].median())

    # Mann-Whitney: does cluster vs non-cluster differ in slope_delta?
    u_stat, p_val = stats.mannwhitneyu(
        cluster_df["slope_delta"], nc_df["slope_delta"],
        alternative="two-sided",
    )
    log.info("Mann-Whitney slope_delta cluster vs non-cluster: "
             "U=%.0f, p=%.4g", u_stat, p_val)

    # Stratify by bounded slope quartile to control for "you can't lose
    # what you didn't have": does cluster membership predict degradation
    # AFTER conditioning on the bounded slope?
    qs = np.quantile(df["bounded_slope"], [0.25, 0.5, 0.75])
    df["bs_quartile"] = (
        (df["bounded_slope"] > qs[0]).astype(int)
        + (df["bounded_slope"] > qs[1]).astype(int)
        + (df["bounded_slope"] > qs[2]).astype(int)
    )
    log.info("==== slope_delta by bounded-slope quartile × cluster ====")
    for q in range(4):
        sub = df[df["bs_quartile"] == q]
        c = sub[sub["is_cluster"]]
        nc = sub[~sub["is_cluster"]]
        if len(c) < 2 or len(nc) < 2:
            log.info("  Q%d: too few cluster/non-cluster anchors (skip)", q)
            continue
        try:
            uq, pq = stats.mannwhitneyu(
                c["slope_delta"], nc["slope_delta"],
                alternative="two-sided",
            )
        except Exception:
            uq, pq = float("nan"), float("nan")
        log.info(
            "  Q%d (bounded ∈ %s): cluster n=%d  Δmedian=%.4f vs "
            "non-cluster n=%d  Δmedian=%.4f  MWU p=%.4g",
            q,
            f"[{sub['bounded_slope'].min():.3f}, "
            f"{sub['bounded_slope'].max():.3f}]",
            len(c), c["slope_delta"].median(),
            len(nc), nc["slope_delta"].median(),
            pq,
        )

    # Per-hop mean|t| diagnostic for cluster anchors under unbounded:
    # does the regional-noise story hold (hop>=3 has elevated |t|)?
    log.info("==== Cluster anchors' shell mean|t| under unbounded ====")
    by_hop: dict[int, list[float]] = {}
    for seed in cluster_df["seed"]:
        u = unbounded.get(seed, {})
        for s in u.get("shells", []):
            by_hop.setdefault(s["hop"], []).append(s["mean_abs_t"])
    for h in sorted(by_hop):
        v = np.array(by_hop[h])
        log.info("  hop %d  (n_anchors=%d, n_shells_present=%d): "
                 "mean=%.3f  median=%.3f",
                 h, len(cluster_df), len(v), v.mean(), np.median(v))

    log.info("==== Verdict ====")
    cluster_median_delta = cluster_df["slope_delta"].median()
    nc_median_delta = nc_df["slope_delta"].median()
    if abs(cluster_median_delta - nc_median_delta) < 0.01:
        v = (
            "Slope deltas similar — cluster anchors don't degrade "
            "more than non-cluster.  Generic regional shift."
        )
    elif cluster_median_delta > nc_median_delta:
        v = (
            "Cluster anchors shift more positive than non-cluster.  "
            "Consistent with regional-noise: cluster anchors had "
            "more negative bounded slopes (more to lose) AND/OR "
            "deeper shells have higher |t| disproportionately for "
            "cluster anchors."
        )
    else:
        v = (
            "Cluster anchors shift LESS positive than non-cluster.  "
            "Refutes simple regional-noise — cluster anchors are "
            "actually relatively resistant to the unbounded shift."
        )
    log.info(v)

    summary = {
        "n_cluster_anchors": int(n_cluster),
        "n_non_cluster_anchors": int(len(nc_df)),
        "cluster_bounded_slope_median": float(
            cluster_df["bounded_slope"].median()
        ),
        "non_cluster_bounded_slope_median": float(
            nc_df["bounded_slope"].median()
        ),
        "cluster_unbounded_slope_median": float(
            cluster_df["unbounded_slope"].median()
        ),
        "non_cluster_unbounded_slope_median": float(
            nc_df["unbounded_slope"].median()
        ),
        "cluster_slope_delta_median": float(
            cluster_df["slope_delta"].median()
        ),
        "non_cluster_slope_delta_median": float(
            nc_df["slope_delta"].median()
        ),
        "mwu_slope_delta_p": float(p_val),
        "verdict": v,
    }
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
