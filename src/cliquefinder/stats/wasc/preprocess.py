"""WASC pre-processing — load proteomics + enriched metadata, build per-group design matrices.

See memory/wasc_spec.md §2 for the regression specification.

Per spec, the covariate design for each donor group g is:
  intercept + Sex + Age (imputed within group×sex) + Tissue (3-level dummy)

Batch is handled separately via within-group ComBat pre-residualization
(NOT included as a covariate column).  ComBat support is deferred to a
follow-up sub-milestone; the initial M2 vertical slice uses the design
above without batch correction.  The M2.5 tripwire validates F-W
identity against statsmodels.OLS WITH explicit batch dummies on SPOR.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default location of the v1.0.1 enriched-metadata artifact (M6a-amend).
DEFAULT_ENRICHED_METADATA_PATH = (
    Path(__file__).resolve().parents[4]
    / "data" / "wasc" / "metadata_enriched_v1.json"
)
DEFAULT_PROTEOMICS_CSV_PATH = (
    Path(__file__).resolve().parents[4]
    / "output" / "proteomics" / "all_als.data.csv"
)


class GroupDesign(NamedTuple):
    """Per-group covariate design matrix + the sample subset it covers.

    Attributes
    ----------
    group : str
        "C9ORF72" | "SPORADIC" | "CONTROL"
    sample_ids : list[str]
        Donor IDs included (non-external, post-exclusion, with complete covariates).
    X_cov : np.ndarray
        ``(n_samples_g, p_cov)`` covariate design matrix.  The first column
        is the intercept.  Subsequent columns are Sex, Age_z (z-scored
        within group), Tissue dummies (drop_first=True).
    column_names : list[str]
        Names of the columns in `X_cov`, in order.
    """
    group: str
    sample_ids: list[str]
    X_cov: np.ndarray
    column_names: list[str]


class WascDataBundle(NamedTuple):
    """All inputs the F-W kernel needs to run per-edge regressions.

    Attributes
    ----------
    abundance : pd.DataFrame
        ``(n_proteins, n_samples)`` log2-abundance matrix indexed by UniProt.
        Same as the proteomics matrix; included by reference, not copied.
    designs : dict[str, GroupDesign]
        Per-group design matrices.
    metadata : dict[str, dict]
        Per-sample enriched record (full v1.0.1 artifact, keyed by sample_id).
    """
    abundance: pd.DataFrame
    designs: dict[str, "GroupDesign"]
    metadata: dict[str, dict]


def load_proteomics(path: Path | str | None = None) -> pd.DataFrame:
    """Load the per-protein log2-abundance matrix.  Rows = UniProt; cols = sample IDs."""
    p = Path(path) if path else DEFAULT_PROTEOMICS_CSV_PATH
    df = pd.read_csv(p, index_col=0)
    logger.info("Loaded proteomics: %d proteins × %d samples from %s",
                df.shape[0], df.shape[1], p)
    return df


def load_enriched_metadata(path: Path | str | None = None) -> dict:
    """Load `data/wasc/metadata_enriched_v1.json` (v1.0.1)."""
    p = Path(path) if path else DEFAULT_ENRICHED_METADATA_PATH
    doc = json.loads(p.read_text())
    if doc.get("version") not in ("v1.0", "v1.0.1"):
        logger.warning("Unexpected metadata version: %s (expected v1.0 or v1.0.1)",
                       doc.get("version"))
    logger.info("Loaded enriched metadata: %d samples from %s",
                len(doc.get("samples", {})), p)
    return doc


def build_group_design(
    enriched_metadata: dict,
    group: str,
    *,
    eligible_sample_ids: list[str] | None = None,
) -> GroupDesign:
    """Build the per-group covariate design matrix per spec §2.

    Covariates: intercept + Sex + Age_z (z-scored within group) + Tissue dummies.

    Parameters
    ----------
    enriched_metadata
        Loaded v1.0.1 enriched metadata doc.
    group
        "C9ORF72" | "SPORADIC" | "CONTROL".
    eligible_sample_ids
        Restrict to these sample IDs (e.g., intersection with proteomics columns).
        If None, uses all non-external samples in the group from the metadata.

    Returns
    -------
    GroupDesign
    """
    rows = []
    for sid, rec in enriched_metadata["samples"].items():
        if rec.get("wasc_group") != group:
            continue
        if rec.get("is_external"):
            continue
        if eligible_sample_ids is not None and sid not in eligible_sample_ids:
            continue
        # Skip if essential fields missing.
        if rec.get("sex") is None:
            continue
        if rec.get("age_imputed") is None:
            continue
        if rec.get("tissue_collapsed") is None:
            continue
        rows.append({
            "sample_id":       sid,
            "sex":             rec["sex"],
            "age_imputed":     float(rec["age_imputed"]),
            "tissue":          rec["tissue_collapsed"],
        })
    df = pd.DataFrame(rows).set_index("sample_id")

    # Sex: binary {Male: 0, Female: 1}.
    sex_binary = (df["sex"] == "Female").astype(float).values

    # Age z-scored WITHIN GROUP (per spec §2.1).
    age = df["age_imputed"].values.astype(float)
    age_mean = float(np.nanmean(age))
    age_std = float(np.nanstd(age, ddof=1))
    age_z = (age - age_mean) / (age_std if age_std > 0 else 1.0)

    # Tissue dummies — drop_first=True with T_Cell as reference per spec.
    tissue_series = df["tissue"]
    # Stable column ordering for reproducibility.
    tissue_order = ["NT_Cell", "Bulk_or_Unknown"]
    tissue_cols = []
    tissue_names = []
    for level in tissue_order:
        if (tissue_series == level).any():
            tissue_cols.append((tissue_series == level).astype(float).values)
            tissue_names.append(f"tissue_{level}")

    intercept = np.ones(len(df))
    columns = [intercept, sex_binary, age_z] + tissue_cols
    column_names = ["intercept", "sex_female", "age_z"] + tissue_names
    X_cov = np.column_stack(columns)

    return GroupDesign(
        group=group,
        sample_ids=df.index.tolist(),
        X_cov=X_cov,
        column_names=column_names,
    )


def build_wasc_data_bundle(
    proteomics_csv: Path | str | None = None,
    enriched_metadata_path: Path | str | None = None,
) -> WascDataBundle:
    """One-shot loader: produce everything the F-W kernel needs."""
    abundance = load_proteomics(proteomics_csv)
    metadata = load_enriched_metadata(enriched_metadata_path)
    eligible = set(abundance.columns)
    designs = {
        g: build_group_design(metadata, g, eligible_sample_ids=list(eligible))
        for g in ("C9ORF72", "SPORADIC", "CONTROL")
    }
    for g, d in designs.items():
        logger.info("Design %s: n=%d samples, %d covariates [%s]",
                    g, len(d.sample_ids), d.X_cov.shape[1],
                    ", ".join(d.column_names))
    return WascDataBundle(
        abundance=abundance,
        designs=designs,
        metadata=metadata.get("samples", {}),
    )
