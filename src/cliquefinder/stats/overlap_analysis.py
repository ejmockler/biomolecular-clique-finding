"""
Gene set overlap quantification and effective independent test estimation.

When testing many gene sets from a dense regulatory graph (e.g., INDRA at
hop 3+), massive overlap inflates the apparent number of discoveries because
a single proteomic shift lights up hundreds of overlapping sets.  Standard
FDR (Storey's BH) assumes independence or weak dependence.

This module quantifies the overlap and estimates the effective number of
independent tests using Li & Ji (2005), a method originally developed for
correlated SNPs in GWAS linkage disequilibrium blocks.

Li & Ji (2005) method:
    1. Build M x N binary indicator matrix (M gene sets, N genes)
    2. Compute M x M correlation matrix of the binary indicators
    3. Eigendecompose the correlation matrix
    4. M_eff = sum_i [ I(lambda_i >= 1) + (lambda_i - floor(lambda_i)) ]

Reference:
    Li, J. & Ji, L. (2005). Adjusting multiple testing in multilocus
    analyses using the eigenvalues of a correlation matrix. Heredity, 95(3).
"""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray


def compute_jaccard_matrix(gene_sets: dict[str, set[str]]) -> NDArray[np.float64]:
    """Compute pairwise Jaccard similarity matrix for gene sets.

    Parameters
    ----------
    gene_sets : dict mapping set name -> set of gene identifiers

    Returns
    -------
    np.ndarray of shape (M, M) with Jaccard similarities.
        Diagonal is 1.0.  Empty sets yield 0.0 for all pairwise comparisons.
    """
    names = list(gene_sets.keys())
    m = len(names)
    if m == 0:
        return np.empty((0, 0), dtype=np.float64)

    sets = [gene_sets[n] for n in names]
    mat = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        mat[i, i] = 1.0
        for j in range(i + 1, m):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union == 0:
                jac = 0.0
            else:
                jac = intersection / union
            mat[i, j] = jac
            mat[j, i] = jac

    return mat


def _build_binary_matrix(gene_sets: dict[str, set[str]]) -> NDArray[np.float64]:
    """Build M x N binary indicator matrix.

    Rows correspond to gene sets, columns to the union of all genes.
    Entry (i, j) = 1 if gene j is in set i, else 0.
    """
    names = list(gene_sets.keys())
    all_genes = sorted(set().union(*gene_sets.values()))
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    m = len(names)
    n = len(all_genes)
    mat = np.zeros((m, n), dtype=np.float64)

    for i, name in enumerate(names):
        for gene in gene_sets[name]:
            mat[i, gene_to_idx[gene]] = 1.0

    return mat


def effective_independent_tests(gene_sets: dict[str, set[str]]) -> dict:
    """Estimate effective number of independent tests using Li & Ji (2005).

    Builds the binary membership matrix (M tests x N genes), computes the
    M x M correlation matrix, eigendecomposes, and counts effective tests as:

        M_eff = sum_i [ I(lambda_i >= 1) + (lambda_i - floor(lambda_i)) ]

    Parameters
    ----------
    gene_sets : dict mapping set name -> set of gene identifiers

    Returns
    -------
    dict with keys:
        m_nominal : int - number of tests
        m_eff : float - effective number of independent tests
        ratio : float - m_eff / m_nominal (1.0 = fully independent)
        median_jaccard : float
        max_jaccard : float - maximum off-diagonal Jaccard similarity
        pct_pairs_above_50 : float - % of pairs with Jaccard > 0.5
        eigenvalue_summary : dict with eigenvalue statistics
    """
    m = len(gene_sets)
    if m == 0:
        return {
            "m_nominal": 0,
            "m_eff": 0.0,
            "ratio": 0.0,
            "median_jaccard": 0.0,
            "max_jaccard": 0.0,
            "pct_pairs_above_50": 0.0,
            "eigenvalue_summary": {},
        }
    if m == 1:
        return {
            "m_nominal": 1,
            "m_eff": 1.0,
            "ratio": 1.0,
            "median_jaccard": 0.0,
            "max_jaccard": 0.0,
            "pct_pairs_above_50": 0.0,
            "eigenvalue_summary": {"max": 1.0, "min": 1.0, "n_geq_1": 1},
        }

    # Build binary indicator matrix and compute correlation
    binary_mat = _build_binary_matrix(gene_sets)

    # Compute correlation matrix of the rows (M x M)
    # np.corrcoef treats each row as a variable
    corr = np.corrcoef(binary_mat)

    # Handle degenerate cases: constant rows produce NaN correlations
    # Replace NaN with 0 (zero-variance sets are effectively independent)
    corr = np.nan_to_num(corr, nan=0.0)

    # Eigendecompose
    eigenvalues = np.linalg.eigvalsh(corr)
    # eigvalsh returns ascending order; we want descending for summary
    eigenvalues = eigenvalues[::-1]

    # Li & Ji formula: M_eff = sum [ I(lambda >= 1) + (lambda - floor(lambda)) ]
    # For negative eigenvalues (numerical noise), contribution is 0
    m_eff = 0.0
    for lam in eigenvalues:
        if lam < 0:
            continue
        if lam >= 1.0:
            m_eff += 1.0 + (lam - np.floor(lam))
        else:
            m_eff += lam - np.floor(lam)  # floor(lam<1) = 0, so this is just lam

    # Ensure m_eff is at least 1 and at most m_nominal
    m_eff = float(np.clip(m_eff, 1.0, m))

    # Jaccard statistics
    jac_mat = compute_jaccard_matrix(gene_sets)
    # Extract upper triangle (off-diagonal)
    triu_idx = np.triu_indices(m, k=1)
    off_diag = jac_mat[triu_idx]

    if len(off_diag) > 0:
        median_jaccard = float(np.median(off_diag))
        max_jaccard = float(np.max(off_diag))
        pct_above_50 = float(100.0 * np.mean(off_diag > 0.5))
    else:
        median_jaccard = 0.0
        max_jaccard = 0.0
        pct_above_50 = 0.0

    return {
        "m_nominal": m,
        "m_eff": round(m_eff, 2),
        "ratio": round(m_eff / m, 4),
        "median_jaccard": round(median_jaccard, 4),
        "max_jaccard": round(max_jaccard, 4),
        "pct_pairs_above_50": round(pct_above_50, 2),
        "eigenvalue_summary": {
            "max": round(float(eigenvalues[0]), 4),
            "min": round(float(eigenvalues[-1]), 4),
            "n_geq_1": int(np.sum(eigenvalues >= 1.0)),
        },
    }


def overlap_summary(gene_sets: dict[str, set[str]]) -> dict:
    """Quick summary statistics without full eigendecomposition.

    Computes Jaccard statistics only (O(M^2) set operations but avoids
    the O(M^2 * N) correlation matrix and O(M^3) eigendecomposition).

    Parameters
    ----------
    gene_sets : dict mapping set name -> set of gene identifiers

    Returns
    -------
    dict with keys:
        m_nominal : int
        median_jaccard : float
        max_jaccard : float
        pct_pairs_above_50 : float
        mean_set_size : float
        min_set_size : int
        max_set_size : int
    """
    m = len(gene_sets)
    if m == 0:
        return {
            "m_nominal": 0,
            "median_jaccard": 0.0,
            "max_jaccard": 0.0,
            "pct_pairs_above_50": 0.0,
            "mean_set_size": 0.0,
            "min_set_size": 0,
            "max_set_size": 0,
        }

    sizes = [len(s) for s in gene_sets.values()]

    if m == 1:
        return {
            "m_nominal": 1,
            "median_jaccard": 0.0,
            "max_jaccard": 0.0,
            "pct_pairs_above_50": 0.0,
            "mean_set_size": float(sizes[0]),
            "min_set_size": sizes[0],
            "max_set_size": sizes[0],
        }

    jac_mat = compute_jaccard_matrix(gene_sets)
    triu_idx = np.triu_indices(m, k=1)
    off_diag = jac_mat[triu_idx]

    return {
        "m_nominal": m,
        "median_jaccard": round(float(np.median(off_diag)), 4),
        "max_jaccard": round(float(np.max(off_diag)), 4),
        "pct_pairs_above_50": round(float(100.0 * np.mean(off_diag > 0.5)), 2),
        "mean_set_size": round(float(np.mean(sizes)), 1),
        "min_set_size": int(min(sizes)),
        "max_set_size": int(max(sizes)),
    }


def annotate_discovery_with_overlap(
    discovery_result: dict,
    gene_sets_per_hop: dict[int, dict[str, set[str]]] | None = None,
    fdr_threshold: float = 0.05,
) -> dict:
    """Add overlap statistics to each hop in a discovery result.

    For each hop, extracts the gene sets (targets for each arm) and computes
    the effective independent test count, then annotates the hop dict with
    an "overlap" key containing:
        - m_nominal: number of tests at this hop
        - m_eff: effective independent tests
        - ratio: m_eff / m_nominal
        - median_jaccard: median pairwise Jaccard similarity
        - adjusted_fdr_threshold: nominal FDR * (m_nominal / m_eff)

    Parameters
    ----------
    discovery_result : dict
        Discovery result dictionary (as loaded from JSON).
    gene_sets_per_hop : dict mapping hop number -> {intermediary: set(targets)}
        If provided, uses these gene sets directly. Otherwise, reconstructs
        from the discovery result (requires 'targets' field in arms).
    fdr_threshold : float
        Nominal FDR threshold (default 0.05) for computing adjusted threshold.

    Returns
    -------
    dict - the discovery result with "overlap" added to each hop.
    """
    result = copy.deepcopy(discovery_result)

    for hop_data in result.get("hops", []):
        hop_num = hop_data.get("hop")
        arms = hop_data.get("all_arms", [])

        if gene_sets_per_hop is not None and hop_num in gene_sets_per_hop:
            gene_sets = gene_sets_per_hop[hop_num]
        elif arms and "targets" in arms[0]:
            # Reconstruct from arm targets
            gene_sets = {}
            for arm in arms:
                name = arm.get("intermediary", "unknown")
                targets = arm.get("targets", [])
                gene_sets[name] = set(targets)
        else:
            # Cannot reconstruct gene sets for this hop
            hop_data["overlap"] = {
                "error": "Gene sets not available (no targets in arms and "
                         "no gene_sets_per_hop provided)",
            }
            continue

        if len(gene_sets) < 2:
            hop_data["overlap"] = {
                "m_nominal": len(gene_sets),
                "m_eff": float(len(gene_sets)),
                "ratio": 1.0,
                "median_jaccard": 0.0,
                "adjusted_fdr_threshold": fdr_threshold,
            }
            continue

        stats = effective_independent_tests(gene_sets)
        m_eff = stats["m_eff"]
        m_nominal = stats["m_nominal"]

        # Adjusted FDR threshold: scale up to account for dependence
        # If tests are correlated, the effective multiple testing burden is lower
        adjusted_fdr = fdr_threshold * m_nominal / m_eff if m_eff > 0 else fdr_threshold

        hop_data["overlap"] = {
            "m_nominal": m_nominal,
            "m_eff": m_eff,
            "ratio": stats["ratio"],
            "median_jaccard": stats["median_jaccard"],
            "max_jaccard": stats["max_jaccard"],
            "pct_pairs_above_50": stats["pct_pairs_above_50"],
            "adjusted_fdr_threshold": round(adjusted_fdr, 6),
            "eigenvalue_summary": stats["eigenvalue_summary"],
        }

    return result
