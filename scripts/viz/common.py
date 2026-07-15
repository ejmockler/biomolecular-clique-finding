"""Shared constants + helpers for the C9-ALS cluster-explorer visualizations.

- TERMS: 8 discovery-derived frozen terms (cluster, short_label, full_term, term_id)
- CLUSTER_COLOR: hex color per biological cluster
- resolve_groups: AnswerALS group definitions (C9 / SPORADIC / CONTROL)
- fit_per_protein_t: fast vectorized OLS per-protein, returns t-statistic
- fetch_term_members: INDRA cogex Neo4j gene-set fetch
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ANALYSIS_PATH = ROOT / "data/publication/c9_primary_analysis.json"
with PRIMARY_ANALYSIS_PATH.open(encoding="utf-8") as _fh:
    PRIMARY_ANALYSIS = json.load(_fh)

# Discovery-derived terms frozen for the same-cohort consistency rerun.
# (cluster, short_label, full_term_name, cogex_term_id)
TERMS = [
    ("Splicing",  "mRNA Splicing",                   "mRNA Splicing",                                   "reactome:R-HSA-72172"),
    ("Splicing",  "Processing Capped Pre-mRNA",      "Processing of Capped Intron-Containing Pre-mRNA", "reactome:R-HSA-72203"),
    ("Splicing",  "mRNA splicing, via spliceosome",  "mRNA splicing, via spliceosome",                  "go:0000398"),
    ("Chromatin", "chromosome",                      "chromosome",                                      "go:0005694"),
    ("Chromatin", "chromatin",                       "chromatin",                                       "go:0000785"),
    ("Transport", "nucleocytoplasmic transport",     "nucleocytoplasmic transport",                     "go:0006913"),
    ("Transport", "nuclear pore",                    "nuclear pore",                                    "go:0005643"),
    ("Transport", "Vpr-mediated nuclear import",     "Vpr-mediated nuclear import of PICs",             "reactome:R-HSA-180910"),
]

# Distinct, accessible colors per cluster.  Used for highlighting in
# all views.  Background tints are alpha-reduced versions.
CLUSTER_COLOR = {
    "Splicing":  "#1f78b4",   # blue
    "Chromatin": "#6a3d9a",   # purple
    "Transport": "#e6550d",   # orange
}
CLUSTER_TINT = {
    "Splicing":  "rgba(31, 120, 180, 0.18)",
    "Chromatin": "rgba(106, 61, 154, 0.18)",
    "Transport": "rgba(230, 85, 13, 0.18)",
}

# Production log2(x+1), bounded measured-only h<=2 fixed-panel result.
# The tracked publication manifest is the single source of numeric truth;
# "Healthy" is retained only as the established display label in these figures.
_CONFIRMATORY = PRIMARY_ANALYSIS["confirmatory"]
_DISPLAY_CONTRAST = {
    "C9 vs Sporadic": "C9 vs Sporadic",
    "C9 vs Control": "C9 vs Healthy",
    "Sporadic vs Control": "Sporadic vs Healthy",
}
BONFERRONI_8 = {
    _DISPLAY_CONTRAST[contrast]: {
        term: (float(values["NES"]), float(values["raw_p"]))
        for term, values in result["terms"].items()
    }
    for contrast, result in _CONFIRMATORY["bounded"].items()
}

ALPHA_FAMILY = float(_CONFIRMATORY["familywise_alpha"])
N_TERMS = len(_CONFIRMATORY["term_order"])
ALPHA_PER_TEST = float(_CONFIRMATORY["per_term_alpha"])

CONTRAST_ORDER = [
    _DISPLAY_CONTRAST[contrast]
    for contrast in _CONFIRMATORY["contrast_order"]
]
# Internal short codes for filenames / column lookup
CONTRAST_CODE = {
    "C9 vs Sporadic":      "c9spor",
    "C9 vs Healthy":       "c9ctrl",
    "Sporadic vs Healthy": "spctrl",
}
CONTRAST_GROUPS = {
    "C9 vs Sporadic":      ("C9ORF72", "SPORADIC"),
    "C9 vs Healthy":       ("C9ORF72", "CONTROL"),
    "Sporadic vs Healthy": ("SPORADIC", "CONTROL"),
}


def resolve_groups(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    """AnswerALS group definitions matching scripts/run_landscape_proteome.py."""
    c9 = metadata[
        (metadata["ClinReport_Mutations_Details"] == "C9orf72")
        | (metadata["C9orf72_repeat_length"] >= 30)
    ]
    known_muts = [
        "C9orf72", "SOD1", "FUS", "TARDBP", "TARDBP (TDP43)",
        "SETX", "Multiple", "Other",
    ]
    sporadic = metadata[
        (metadata["phenotype"] == "CASE")
        & (~metadata["ClinReport_Mutations_Details"].isin(known_muts))
        & (
            (metadata["C9orf72_repeat_length"] < 30)
            | metadata["C9orf72_repeat_length"].isna()
        )
    ]
    control = metadata[metadata["phenotype"] == "CTRL"]
    return {
        "C9ORF72":  c9.index,
        "SPORADIC": sporadic.index,
        "CONTROL":  control.index,
    }


def fit_per_protein_t(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: dict[str, pd.Index],
    contrast: tuple[str, str],
) -> pd.Series:
    """Fit per-protein OLS log_abundance ~ Group + Sex.  Return t-stat
    for the Group coefficient, indexed by protein.

    Vectorized per-protein with per-protein NaN masking.  Sub-second
    for the AnswerALS scale (3,264 feature rows × ~300 samples per pair).
    """
    case, ctrl = contrast
    case_idx = groups[case].intersection(metadata.index)
    ctrl_idx = groups[ctrl].intersection(metadata.index)
    sample_ids = list(case_idx) + list(ctrl_idx)
    # Restrict to samples that exist in the data columns.
    sample_ids = [s for s in sample_ids if s in data.columns]
    if not sample_ids:
        raise ValueError(f"No common samples for contrast {contrast}")

    sub_md = metadata.loc[sample_ids].copy()
    sub_md["_group"] = sub_md.index.isin(case_idx).astype(float)

    # Sex covariate as 0/1.  If only one level present, drop it.
    sex_vals = sub_md["Sex"].fillna("Unknown")
    sex_levels = sex_vals.unique()
    has_sex = len(sex_levels) > 1
    if has_sex:
        # Take the first level as reference; encode others as 1.
        ref = sex_levels[0]
        sub_md["_sex"] = (sex_vals != ref).astype(float)

    # Design matrix: [1, group, sex?]
    n = len(sample_ids)
    cols = [np.ones(n), sub_md["_group"].values]
    if has_sex:
        cols.append(sub_md["_sex"].values)
    X = np.column_stack(cols).astype(np.float64)
    p = X.shape[1]

    Y = data[sample_ids].values.astype(np.float64)  # (n_proteins, n_samples)
    n_proteins = Y.shape[0]
    t_stats = np.full(n_proteins, np.nan, dtype=np.float64)

    # Per-protein OLS with per-protein NaN masking.  Vectorized:
    # mask each protein, solve normal equations.  For most proteins
    # the NaN pattern is similar so we batch-solve where possible.
    nan_mask = np.isnan(Y)
    fully_obs = ~nan_mask.any(axis=1)

    # Fast path: proteins with no missing values share one (X'X)^-1.
    if fully_obs.any():
        XtX_inv = np.linalg.inv(X.T @ X)
        Y_fo = Y[fully_obs]
        beta = (XtX_inv @ X.T @ Y_fo.T).T  # (n_fo, p)
        resid = Y_fo - beta @ X.T          # (n_fo, n)
        rss = (resid ** 2).sum(axis=1)
        dof = n - p
        sigma2 = rss / dof
        se_group = np.sqrt(sigma2 * XtX_inv[1, 1])
        t_stats[fully_obs] = beta[:, 1] / se_group

    # Slow path: proteins with NaN values, fit one by one.
    needs_loop = np.where(~fully_obs)[0]
    for i in needs_loop:
        mask = ~nan_mask[i]
        if mask.sum() < p + 1:
            continue
        Xi = X[mask]
        yi = Y[i, mask]
        try:
            XtX_inv_i = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError:
            continue
        beta_i = XtX_inv_i @ Xi.T @ yi
        resid_i = yi - Xi @ beta_i
        rss_i = (resid_i ** 2).sum()
        dof_i = mask.sum() - p
        if dof_i <= 0:
            continue
        sigma2_i = rss_i / dof_i
        se_i = np.sqrt(sigma2_i * XtX_inv_i[1, 1])
        if se_i > 0:
            t_stats[i] = beta_i[1] / se_i

    return pd.Series(t_stats, index=data.index, name=f"t__{case}_vs_{ctrl}")


def fetch_term_members_via_indra(
    term_ids: list[str],
    env_file: Path | None = None,
) -> dict[str, set[str]]:
    """Query INDRA cogex Neo4j for the HGNC-ID gene set of each term.

    Returns {cogex_term_id: {hgnc_ids}}.  Uses the public INDRA cogex
    endpoint via the project's CoGExClient (independent of SSH).

    Cogex uses different edge labels per source: GO uses
    ``[:associated_with]``; Reactome uses ``[:haspart]``; WikiPathways
    and HPO use other labels.  We union over any inbound edge from a
    gene-typed BioEntity, which catches the relevant ones across
    sources.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cliquefinder.knowledge.cogex import CoGExClient

    env_file = env_file or (ROOT / ".env")
    out: dict[str, set[str]] = {}
    # Union over edge types — restrict to gene-typed sources (hgnc-prefixed)
    # to avoid pulling in pathway/disease entities.
    q = """
    MATCH (g:BioEntity)-[r]-(t:BioEntity {id: $id})
    WHERE g.id STARTS WITH 'hgnc:'
      AND type(r) IN ['associated_with', 'haspart', 'has_member', 'partof']
    RETURN DISTINCT g.id AS hgnc_id
    """
    with CoGExClient(env_file=env_file) as c:
        for tid in term_ids:
            rows = c._execute_query(q, id=tid)
            out[tid] = {
                row[0].replace("hgnc:", "")
                for row in rows
                if row[0]
            }
    return out


def hgnc_ids_to_uniprots(hgnc_ids: set[str]) -> set[str]:
    """HGNC IDs → UniProt accessions via indra.databases.hgnc_client.
    Some HGNC IDs map to multiple UniProts (comma-separated)."""
    from indra.databases import hgnc_client
    out: set[str] = set()
    for h in hgnc_ids:
        raw = hgnc_client.get_uniprot_id(h)
        if not raw:
            continue
        for u in str(raw).split(","):
            u = u.strip()
            if u:
                out.add(u)
    return out


def uniprot_to_hgnc_symbol(uniprots: list[str]) -> dict[str, str]:
    """UniProt → official HGNC approved symbol (one per UniProt).

    Uses INDRA's uniprot_client.get_gene_name, which returns the
    HGNC-approved symbol (e.g. SRSF1, NUP62, MBD3) rather than the
    alphabetically-first synonym or a BAC-clone alias. Unresolved accessions
    retain their UniProt ID; publication-figure generation never falls back to
    a live identifier service."""
    from indra.databases import uniprot_client

    out: dict[str, str] = {}
    for u in uniprots:
        try:
            sym = uniprot_client.get_gene_name(u)
            if sym:
                out[u] = sym
                continue
        except Exception:
            pass
        out[u] = u
    return out
