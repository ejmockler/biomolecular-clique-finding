"""M2.2.5 — Compute measured-only INDRA hop-1 degree per protein.

This wraps the earlier inline computation into a reproducible script.
The output is the FIRST AXIS of the WASC 3-axis matched-bin null.

Source of truth
---------------
`output/landscape_proteome/distances.npz` (CSR sparse) holds the
measured-only INDRA distance matrix:
    - d=0   : self loops (3,258 entries — one per measured protein with a
              regulatory neighbor; a handful of measured proteins are
              fully isolated in INDRA)
    - d=1   : 137,314 entries = 2 × 68,657 undirected hop-1 edges
    - d=2   : 10.5M entries

The "measured-only" hop-1 degree of protein p is the number of d==1
entries in its row of the distance matrix (equivalently, the count of
measured proteins directly connected via INDRA regulatory edges).

Why this matters (per the v1.0 WASC spec)
-----------------------------------------
The null draws non-neighbors matched on degree decile so the test isn't
biased by hub anchors having structurally richer neighborhoods than
peripheral anchors.  The cache lives at `data/wasc/measured_degrees_v1.json`
and is loaded by `cliquefinder.stats.wasc.bins.load_measured_degrees`.

Output
------
data/wasc/measured_degrees_v1.json with:
    version, derived_from, derived_from_sha256, n_proteins, metric,
    degree_quantiles, mean_degree, zero_degree_count, degrees (dict).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DISTANCES = REPO / "output" / "landscape_proteome" / "distances.npz"
DEFAULT_OUTPUT = REPO / "data" / "wasc" / "measured_degrees_v1.json"


def _sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(distances_path: Path = DEFAULT_DISTANCES,
         output_path: Path = DEFAULT_OUTPUT) -> None:
    if not distances_path.exists():
        print(f"FATAL: distances file not found: {distances_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading distances from {distances_path}")
    npz = np.load(distances_path, allow_pickle=True)
    # CSR storage (scipy.sparse.save_npz format)
    data = npz["data"]
    indices = npz["indices"]
    indptr = npz["indptr"]
    shape = tuple(int(x) for x in npz["shape"])
    # Protein labels live alongside in distances.meta.json
    meta_path = distances_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    proteins = list(meta["feature_names"])
    print(f"  {shape[0]} × {shape[1]} matrix, nnz = {len(data)}")
    print(f"  {len(proteins)} protein labels (from {meta_path.name})")

    csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

    # Distance distribution sanity check
    unique, counts = np.unique(data, return_counts=True)
    print(f"  distance distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Hop-1 degree per row: count of d==1 entries
    print("Computing hop-1 degree per protein...")
    degrees = np.zeros(shape[0], dtype=np.int64)
    for i in range(shape[0]):
        row_data = data[indptr[i]:indptr[i + 1]]
        degrees[i] = int((row_data == 1).sum())

    # Stats
    finite = degrees[degrees > 0]
    quantiles = {
        "p5":  float(np.quantile(degrees, 0.05)),
        "p25": float(np.quantile(degrees, 0.25)),
        "p50": float(np.quantile(degrees, 0.50)),
        "p75": float(np.quantile(degrees, 0.75)),
        "p95": float(np.quantile(degrees, 0.95)),
    }
    mean_deg = float(np.mean(degrees))
    n_zero = int((degrees == 0).sum())

    print(f"  degree min={degrees.min()} max={degrees.max()} mean={mean_deg:.1f}")
    print(f"  quantiles: {quantiles}")
    print(f"  zero-degree proteins: {n_zero}")

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": "v1.0",
        "derived_from": distances_path.relative_to(REPO).as_posix(),
        "derived_from_sha256": _sha256_of(distances_path),
        "n_proteins": int(shape[0]),
        "metric": "measured-only INDRA hop-1 degree (count of d==1 entries per row)",
        "degree_quantiles": quantiles,
        "mean_degree": mean_deg,
        "zero_degree_count": n_zero,
        "degrees": {p: int(degrees[i]) for i, p in enumerate(proteins)},
    }
    output_path.write_text(json.dumps(doc, indent=2, sort_keys=False))
    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
