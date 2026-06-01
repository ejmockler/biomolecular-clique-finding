"""Generate `data/wasc/metadata_enriched_v1.json`.

Joins `output/proteomics/all_als.metadata.csv` (5 columns, project-local)
with `aals_dataportal_datatable.csv` (51 columns, AnswerALS portal export)
to produce the full WASC covariate set per spec §2:

  - Sex                              (100% coverage in portal)
  - Age_at_First_PBMC_Collection     (~94% coverage; impute by Sex within group)
  - Primary_Tissue (3-level collapse to T-Cell / NT-Cell / Bulk-or-Unknown)
  - Batch                            (50 levels; preserved + C9-coarsened key)
  - External_Control flag            (iPSC-derived, excluded from primary)

Sample-to-participant key: parse `CASE_NEUAA295HHE-9014-P_D3` → `CASE-NEUAA295HHE`
(case-insensitive on the participant suffix to capture lowercase ``i`` in `EDi***`).

The 11 EDi* samples without portal records + 21 portal-flagged External_Control=Yes
are both treated as external/iPSC controls and excluded from the primary analysis
per spec §2.3. Net cohort:

  C9    = ~25 (resolve_groups from local metadata)
  SPOR  = ~294 (resolve_groups)
  CTRL  = ~91 less external (resolve_groups intersected with non-external)

The C9 column "Batch_C9_coarsened" maps C9 donor batches to a coarser
site/year stratification to avoid within-group ComBat singleton-batch
degeneracy.  Since the underlying batch labels are integer codes without
site/year keys in the portal export, the coarsening is implemented as
quantile-binning of the integer Batch into 6 quasi-strata (~4 donors each)
for C9 donors only.  This is a pragmatic best-effort given the portal's
limited Batch metadata; the spec calls for explicit site/year if available.

Output is JSON keyed by sample_id, suitable for direct ingestion by the
WASC fit kernel.  SHA-256 is recorded in the WASC manifest (M6a-amend).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import resolve_groups  # noqa: E402

logger = logging.getLogger("wasc-metadata")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

OUT_DIR = ROOT / "data" / "wasc"
PROTEOMICS_META = ROOT / "output" / "proteomics" / "all_als.metadata.csv"
PORTAL_CSV = ROOT / "aals_dataportal_datatable.csv"

# Tissue normalization map (collapse 7+ variant strings to 3 strata per spec)
TISSUE_COLLAPSE = {
    "PBMC/T-Cell":   "T_Cell",
    "PBMC/T-Cell ":  "T_Cell",
    "PBMC/NT-Cell":  "NT_Cell",
    "PBMC/NT-Cell ": "NT_Cell",
    "PBMC/NT-cell":  "NT_Cell",
    "PBMC":          "Bulk_or_Unknown",
}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_participant_id(sample_id: str) -> str | None:
    """Convert proteomics sample-id to portal-style participant id.

    `CASE_NEUAA295HHE-9014-P_D3`  → `CASE-NEUAA295HHE`
    `CTRL_EDi021A-8041-P_B4`      → `CTRL-EDi021A`   (case-insensitive on the suffix)
    """
    m = re.match(r"^(CASE|CTRL)_([A-Za-z0-9]+)", str(sample_id))
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _impute_age_within_group(
    age: pd.Series,
    sex: pd.Series,
    group: pd.Series,
) -> pd.Series:
    """Per-group linear regression of Age on Sex; impute missing with the
    group-and-sex specific fitted mean. Returns the imputed Age series."""
    out = age.copy().astype(float)
    for g in group.dropna().unique():
        mask_g = (group == g)
        sub = pd.DataFrame({
            "age": age[mask_g].astype(float),
            "sex": sex[mask_g],
        })
        # Group means by sex
        means = sub.groupby("sex")["age"].mean()
        # Fill NaN ages with the (group, sex) mean; if (group, sex) is empty
        # use the within-group mean; if that's empty too, use overall mean.
        within_group_mean = sub["age"].mean()
        for sex_val, mu in means.items():
            cell = mask_g & (sex == sex_val) & out.isna()
            if cell.any():
                fill = mu if pd.notna(mu) else within_group_mean
                if pd.notna(fill):
                    out.loc[cell] = fill
        # Catch any still-missing within group (unknown sex)
        still_missing = mask_g & out.isna()
        if still_missing.any() and pd.notna(within_group_mean):
            out.loc[still_missing] = within_group_mean
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).isoformat()

    # Inputs
    logger.info("Loading proteomics metadata: %s", PROTEOMICS_META)
    md = pd.read_csv(PROTEOMICS_META, index_col=0)
    logger.info("Loading AnswerALS portal: %s", PORTAL_CSV)
    portal = pd.read_csv(PORTAL_CSV)

    md_sha = _sha256_of_file(PROTEOMICS_META)
    portal_sha = _sha256_of_file(PORTAL_CSV)

    # Parse participant id from sample id
    md["participant_id"] = md.index.to_series().map(_parse_participant_id)
    unparsable = md["participant_id"].isna().sum()
    if unparsable > 0:
        logger.warning("%d sample IDs could not be parsed for participant_id", unparsable)

    # Left-join portal — many-to-one is safe since participant_id is unique in portal
    portal_unique = portal.drop_duplicates(subset=["Participant_ID"], keep="first")
    joined = md.merge(
        portal_unique, left_on="participant_id", right_on="Participant_ID",
        how="left", suffixes=("_pro", "_pt"),
    )
    joined.index = md.index

    # Group resolution from project's existing rule (uses local metadata,
    # not portal — keeps existing C9 / SPOR / CTRL split consistent with
    # all prior waves)
    groups = resolve_groups(md)
    group_label = pd.Series(index=md.index, dtype=object)
    for g, idx in groups.items():
        group_label.loc[idx] = g
    joined["wasc_group"] = group_label

    # External / iPSC exclusion: portal-flagged + unmatched (EDi* without portal record)
    external_portal = (joined["External_Control"] == "Yes")
    unmatched_in_portal = joined["Participant_ID"].isnull()
    is_external = external_portal | unmatched_in_portal
    n_external_portal = int(external_portal.sum())
    n_unmatched = int(unmatched_in_portal.sum())
    n_external_total = int(is_external.sum())
    logger.info(
        "External / iPSC exclusion: %d (%d portal-flagged + %d unmatched in portal)",
        n_external_total, n_external_portal, n_unmatched,
    )

    # Covariate columns
    # Sex: prefer portal SEX (already complete), fall back to local Sex
    sex_portal = joined["SEX"]
    sex_local = joined["Sex"]
    sex_combined = sex_portal.fillna(sex_local)
    # Normalize to {Male, Female}
    sex_combined = sex_combined.astype(object)

    # Age: from portal; impute by (group, sex) within wasc_group
    age_raw = joined["Age_at_First_PBMC_Collection"].astype(float)
    age_pre_imputation_na = age_raw.isna().sum()
    # Impute only on non-external samples
    age_imputable = age_raw.copy()
    age_imputable.loc[is_external] = np.nan  # don't waste imputation on excluded
    age_imputed_nonexternal = _impute_age_within_group(
        age_imputable, sex_combined, group_label,
    )
    # Final age: imputed for non-external; raw for external (they're excluded anyway)
    age_final = age_raw.copy()
    age_final.update(age_imputed_nonexternal)
    age_post_imputation_na = age_final[~is_external].isna().sum()

    # Tissue: collapse to 3 strata
    tissue_raw = joined["Primary_Tissue"]
    tissue_collapsed = tissue_raw.map(TISSUE_COLLAPSE).fillna("Bulk_or_Unknown")

    # Batch: integer codes; preserve raw, build C9-coarsened mapping
    batch_raw = joined["Batch"]
    batch_str = batch_raw.where(batch_raw.notna(), None).astype(object)
    # For C9 donors specifically, coarsen to ~6 strata (quantile-binned)
    c9_mask = (group_label == "C9ORF72") & ~is_external
    c9_batches = batch_raw[c9_mask].dropna().astype(float)
    if len(c9_batches) >= 6:
        # 6 quantile-binned coarse strata for C9
        c9_coarse, c9_bin_edges = pd.qcut(c9_batches, q=6, labels=False, retbins=True, duplicates="drop")
        batch_coarsened = batch_raw.copy().astype(object)
        # Set C9 to "C9_strata_{N}" using the qcut result
        c9_coarse_full = pd.Series(index=batch_raw.index, dtype=object)
        c9_coarse_full.loc[c9_batches.index] = c9_coarse.map(lambda v: f"C9_S{int(v)}" if pd.notna(v) else None)
        batch_coarsened.loc[c9_mask] = c9_coarse_full.loc[c9_mask]
        # For non-C9, keep the raw integer batch as a string label
        non_c9 = ~c9_mask
        batch_coarsened.loc[non_c9] = batch_raw.loc[non_c9].apply(
            lambda v: f"B{int(v)}" if pd.notna(v) else None
        )
        coarse_n_strata = int(c9_coarse.dropna().nunique())
    else:
        batch_coarsened = batch_str.copy()
        c9_bin_edges = []
        coarse_n_strata = 0

    # Assemble per-sample record
    samples_record: dict[str, dict] = {}
    for sid in md.index:
        rec = {
            "sample_id":      str(sid),
            "participant_id": joined.at[sid, "participant_id"],
            "phenotype":      joined.at[sid, "phenotype"],
            "wasc_group":     group_label.get(sid),
            "is_external":    bool(is_external.loc[sid]),
            "sex":            sex_combined.get(sid),
            "age_raw":        None if pd.isna(age_raw.loc[sid]) else float(age_raw.loc[sid]),
            "age_imputed":    None if pd.isna(age_final.loc[sid]) else float(age_final.loc[sid]),
            "age_was_imputed": bool(pd.isna(age_raw.loc[sid]) and pd.notna(age_final.loc[sid])),
            "tissue_raw":     joined.at[sid, "Primary_Tissue"]
                              if pd.notna(joined.at[sid, "Primary_Tissue"]) else None,
            "tissue_collapsed": tissue_collapsed.loc[sid],
            "batch_raw":      None if pd.isna(batch_raw.loc[sid]) else int(batch_raw.loc[sid]),
            "batch_for_combat": batch_coarsened.loc[sid] if pd.notna(batch_coarsened.loc[sid]) else None,
            "external_control_portal": joined.at[sid, "External_Control"]
                                       if pd.notna(joined.at[sid, "External_Control"]) else None,
        }
        samples_record[str(sid)] = rec

    # Coverage / sanity summary
    def _count_by_group_non_external(predicate):
        return {
            g: int(((group_label == g) & ~is_external & predicate).sum())
            for g in ["C9ORF72", "SPORADIC", "CONTROL"]
        }

    summary = {
        "total_samples": len(md),
        "total_external_excluded": n_external_total,
        "n_external_portal_flagged": n_external_portal,
        "n_unmatched_in_portal": n_unmatched,
        "groups_pre_exclusion": {
            g: len(groups[g]) for g in ["C9ORF72", "SPORADIC", "CONTROL"]
        },
        "groups_post_exclusion": {
            g: int(((group_label == g) & ~is_external).sum())
            for g in ["C9ORF72", "SPORADIC", "CONTROL"]
        },
        "covariate_coverage_post_exclusion": {
            "sex_non_null": _count_by_group_non_external(sex_combined.notna()),
            "age_raw_non_null": _count_by_group_non_external(age_raw.notna()),
            "age_imputed_non_null": _count_by_group_non_external(age_final.notna()),
            "tissue_known": _count_by_group_non_external(tissue_raw.notna()),
            "batch_known": _count_by_group_non_external(batch_raw.notna()),
        },
        "tissue_strata_post_exclusion": {
            g: tissue_collapsed[(group_label == g) & ~is_external].value_counts().to_dict()
            for g in ["C9ORF72", "SPORADIC", "CONTROL"]
        },
        "batch_strata_post_exclusion_n_levels": {
            g: int(batch_coarsened[(group_label == g) & ~is_external].dropna().nunique())
            for g in ["C9ORF72", "SPORADIC", "CONTROL"]
        },
        "c9_batch_quantile_edges": [float(x) for x in c9_bin_edges] if len(c9_bin_edges) else [],
        "c9_coarse_n_strata": coarse_n_strata,
        "age_pre_imputation_missing": int(age_pre_imputation_na),
        "age_post_imputation_missing_in_eligible": int(age_post_imputation_na),
    }

    doc = {
        "version": "v1.0",
        "frozen_at_git_tag": "wasc-prereg-v1.0.1",
        "generated_at_utc": timestamp,
        "spec_reference": "memory/wasc_spec.md §2",
        "inputs": {
            "proteomics_metadata_csv": {
                "path": str(PROTEOMICS_META.relative_to(ROOT)),
                "sha256": md_sha,
            },
            "aals_portal_csv": {
                "path": str(PORTAL_CSV.relative_to(ROOT)),
                "sha256": portal_sha,
            },
        },
        "summary": summary,
        "samples": samples_record,
    }

    out_path = OUT_DIR / "metadata_enriched_v1.json"
    text = json.dumps(doc, indent=2, sort_keys=False)
    out_path.write_text(text + "\n")
    logger.info("Wrote %s (%d bytes)", out_path, len(text))

    print()
    print("=" * 70)
    print("WASC enriched metadata written.")
    print("=" * 70)
    print(f"  {out_path.relative_to(ROOT)}")
    print()
    print(f"  Total samples:                {summary['total_samples']}")
    print(f"  External / iPSC excluded:     {summary['total_external_excluded']}")
    print(f"    (portal-flagged: {summary['n_external_portal_flagged']}, unmatched: {summary['n_unmatched_in_portal']})")
    print()
    print(f"  Pre-exclusion groups:  {summary['groups_pre_exclusion']}")
    print(f"  Post-exclusion groups: {summary['groups_post_exclusion']}")
    print()
    print("  Covariate coverage post-exclusion:")
    cov = summary["covariate_coverage_post_exclusion"]
    for g in ["C9ORF72", "SPORADIC", "CONTROL"]:
        n_total = summary["groups_post_exclusion"][g]
        print(f"    {g:<10} n={n_total:<4} "
              f"sex={cov['sex_non_null'][g]:<4} "
              f"age_raw={cov['age_raw_non_null'][g]:<4} "
              f"age_imp={cov['age_imputed_non_null'][g]:<4} "
              f"tissue={cov['tissue_known'][g]:<4} "
              f"batch={cov['batch_known'][g]}")
    print()
    print("  C9 batch coarsening:")
    print(f"    raw C9 batches:        {summary['batch_strata_post_exclusion_n_levels']['C9ORF72']} unique levels post-coarsen")
    print(f"    quantile edges:        {[round(x,1) for x in summary['c9_batch_quantile_edges']]}")
    print()
    print("  Tissue strata post-exclusion:")
    for g, ts in summary["tissue_strata_post_exclusion"].items():
        print(f"    {g:<10} {ts}")


if __name__ == "__main__":
    main()
