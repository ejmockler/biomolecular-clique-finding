"""ANGLE C — Cluster-member missingness profile diagnostic.

Loads proteomics matrix via the WASC preprocess loader, intersects with the 377
Splicing+Chromatin+Transport cluster members, computes per-protein NaN-rate
distribution, scans the repo for sibling imputation-flag artifacts, and flags
all-zero / all-identical / quasi-constant rows that would behave LIKE missingness
inside the F-W regression (zero residual variance after covariate projection).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cliquefinder.stats.wasc.preprocess import (
    DEFAULT_PROTEOMICS_CSV_PATH,
    load_proteomics,
)

REPO = Path(__file__).resolve().parents[2]
CLUSTERS_JSON = REPO / "data" / "wasc" / "cluster_members_v1.json"


def load_cluster_members() -> tuple[dict[str, list[str]], list[str]]:
    doc = json.loads(CLUSTERS_JSON.read_text())
    by_theme: dict[str, list[str]] = {}
    flat: list[str] = []
    for theme, payload in doc["themes"].items():
        members = list(payload["measured_uniprots"])
        by_theme[theme] = members
        flat.extend(members)
    return by_theme, flat


def nan_rate_summary(label: str, sub: pd.DataFrame) -> dict:
    rates = sub.isna().mean(axis=1).values
    return {
        "label": label,
        "n_proteins": int(sub.shape[0]),
        "n_samples": int(sub.shape[1]),
        "min": float(np.min(rates)),
        "median": float(np.median(rates)),
        "p95": float(np.percentile(rates, 95)),
        "max": float(np.max(rates)),
        "frac_any_nan": float(np.mean(rates > 0.0)),
        "frac_ge_50pct_nan": float(np.mean(rates >= 0.5)),
    }


def zero_and_constant_diagnostics(sub: pd.DataFrame) -> dict:
    arr = sub.values.astype(float)
    finite_mask = np.isfinite(arr)

    # All-zero rows (post-imputation zero is a flat signal).
    all_zero = np.sum(np.all((arr == 0.0) | ~finite_mask, axis=1) & np.any(finite_mask, axis=1))

    # All-identical (zero variance) — every finite value equal.
    all_identical = 0
    quasi_constant = 0  # std/|mean| < 1e-6 OR std < 1e-12
    near_constant_005 = 0  # CV < 0.005 (0.5%)
    for i in range(arr.shape[0]):
        row = arr[i][finite_mask[i]]
        if row.size <= 1:
            continue
        rmin = float(np.min(row))
        rmax = float(np.max(row))
        if rmin == rmax:
            all_identical += 1
            continue
        std = float(np.std(row, ddof=0))
        mean_abs = float(np.abs(np.mean(row))) + 1e-30
        if std < 1e-12 or (std / mean_abs) < 1e-6:
            quasi_constant += 1
        if (std / mean_abs) < 5e-3:
            near_constant_005 += 1

    # Per-sample-group constancy is the actual WASC concern: a protein constant
    # WITHIN a donor group has zero within-group variance even if globally variable.
    # We do not have group labels in scope here without metadata; flag the global
    # one and recommend group-stratified follow-up.
    return {
        "all_zero_rows": int(all_zero),
        "all_identical_rows": int(all_identical),
        "quasi_constant_rows_cv_1e6": int(quasi_constant),
        "near_constant_rows_cv_5e3": int(near_constant_005),
    }


def search_sibling_artifacts() -> list[str]:
    """Look for files that could carry an imputation/detection mask."""
    candidates: list[Path] = []
    for root in [REPO / "output", REPO / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(tok in name for tok in
                   ("imput", "detect", "mask", "missing", "nan_flag", "is_present")):
                candidates.append(p)
    # Also: any companion to the proteomics CSV?
    prot_dir = DEFAULT_PROTEOMICS_CSV_PATH.parent
    companions = [p for p in prot_dir.iterdir() if p.is_file()]
    return [str(p.relative_to(REPO)) for p in candidates] + \
           ["__proteomics_dir__:" + str(c.relative_to(REPO)) for c in companions]


def main():
    print(f"Loading proteomics from: {DEFAULT_PROTEOMICS_CSV_PATH}")
    abundance = load_proteomics()
    print(f"  shape: {abundance.shape}")
    print(f"  global NaN rate: {abundance.isna().mean().mean():.6f}")
    print(f"  total NaN cells: {int(abundance.isna().sum().sum())}")
    print()

    by_theme, flat = load_cluster_members()
    cluster_set = sorted(set(flat))
    print(f"Cluster members (total across themes, deduped): {len(cluster_set)}")
    for theme, members in by_theme.items():
        print(f"  {theme}: {len(members)}")
    print()

    present = [u for u in cluster_set if u in abundance.index]
    missing = [u for u in cluster_set if u not in abundance.index]
    print(f"Intersection with proteomics index: {len(present)}/{len(cluster_set)}")
    if missing:
        print(f"  NOT in matrix ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print()

    sub = abundance.loc[present]

    print("=== Per-protein NaN-rate summary ===")
    for row in [
        nan_rate_summary("ALL_PROTEINS", abundance),
        nan_rate_summary("CLUSTER_MEMBERS_ALL", sub),
    ]:
        print(json.dumps(row, indent=2))

    for theme, members in by_theme.items():
        idx = [u for u in members if u in abundance.index]
        print(json.dumps(nan_rate_summary(f"THEME_{theme}", abundance.loc[idx]), indent=2))
    print()

    print("=== Zero / constant / quasi-constant rows ===")
    print("Scope: ALL proteins")
    print(json.dumps(zero_and_constant_diagnostics(abundance), indent=2))
    print("Scope: cluster members only")
    print(json.dumps(zero_and_constant_diagnostics(sub), indent=2))
    print()

    print("=== Distribution snapshot of cluster-member values ===")
    vals = sub.values.flatten()
    finite = vals[np.isfinite(vals)]
    print(f"  finite cells: {finite.size} / {vals.size}")
    print(f"  min={finite.min():.4g}  median={np.median(finite):.4g}  max={finite.max():.4g}")
    print(f"  fraction == 0.0: {float(np.mean(finite == 0.0)):.6f}")
    print(f"  fraction <= 1e-6: {float(np.mean(np.abs(finite) <= 1e-6)):.6f}")
    print()

    print("=== Sibling-artifact search ===")
    hits = search_sibling_artifacts()
    if not hits:
        print("  (no candidates found)")
    for h in hits:
        print(f"  {h}")


if __name__ == "__main__":
    main()
