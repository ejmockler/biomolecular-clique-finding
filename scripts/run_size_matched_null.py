#!/usr/bin/env python3
"""Run the canonical graph-independent size-matched pathway null.

The analysis uses the same per-feature statistic as the production landscape:

* log2(intensity + 1)
* ``RotationTestEngine`` with ``~ condition + Sex``
* empirical-Bayes moderated ``|t|``

The measured proteome is then made HGNC-keyed.  A UniProt feature that maps to
more than one row for the same HGNC gene contributes once, using the maximum
``t^2`` across its rows.  Each of the eight discovery-derived pathway terms,
fixed before the canonical method-transfer reruns, is
compared with 10,000 uniformly sampled, same-size HGNC sets from the finite
moderated-t background for that contrast.

Normal (offline) reproduction::

    uv run --no-sync python scripts/run_size_matched_null.py

Refreshing the frozen UniProt/HGNC map and pathway memberships requires the
INDRA CoGEx endpoint and is intentionally explicit::

    uv run --no-sync python scripts/run_size_matched_null.py \
        --refresh-frozen-inputs

The refresh path expands GO terms over ontology descendants, matching the GO
gene-set semantics used by ``indra_cogex`` GSEA.  Reactome terms use direct
``haspart`` membership.  Both the full library memberships and their measured
intersections are frozen so ordinary reruns never depend on live network state.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cliquefinder.panels.seed_runner import load_panel_inputs  # noqa: E402
from cliquefinder.stats.rotation import RotationTestEngine  # noqa: E402

DATA_PATH = ROOT / "output/proteomics/all_als.data.csv"
METADATA_PATH = ROOT / "output/proteomics/all_als.metadata.csv"
FROZEN_INPUTS_PATH = (
    ROOT / "data/publication/c9_size_matched_null_inputs.json"
)
DEFAULT_OUTPUT_PREFIX = ROOT / "output/size_matched_null"

N_PERMUTATIONS = 10_000
MASTER_SEED = 42
FAMILYWISE_ALPHA = 0.05

TERMS: tuple[tuple[str, str, str], ...] = (
    ("Splicing", "reactome:R-HSA-72172", "mRNA Splicing"),
    (
        "Splicing",
        "reactome:R-HSA-72203",
        "Processing of Capped Intron-Containing Pre-mRNA",
    ),
    ("Splicing", "go:0000398", "mRNA splicing, via spliceosome"),
    ("Chromatin", "go:0005694", "chromosome"),
    ("Chromatin", "go:0000785", "chromatin"),
    ("Transport", "go:0006913", "nucleocytoplasmic transport"),
    ("Transport", "go:0005643", "nuclear pore"),
    (
        "Transport",
        "reactome:R-HSA-180910",
        "Vpr-mediated nuclear import of PICs",
    ),
)

CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C9 vs Sporadic", "C9ORF72", "SPORADIC"),
    ("C9 vs Control", "C9ORF72", "CONTROL"),
    ("Sporadic vs Control", "SPORADIC", "CONTROL"),
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Record a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def stable_seed(master_seed: int, *parts: object) -> int:
    """Derive an order-independent 128-bit seed for one null cell."""
    payload = "\x1f".join([str(master_seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big"
    )


def resolve_groups(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    """Resolve the three production AnswerALS arms."""
    c9 = metadata[
        (metadata["ClinReport_Mutations_Details"] == "C9orf72")
        | (metadata["C9orf72_repeat_length"] >= 30)
    ]
    known_mutations = [
        "C9orf72",
        "SOD1",
        "FUS",
        "TARDBP",
        "TARDBP (TDP43)",
        "SETX",
        "Multiple",
        "Other",
    ]
    sporadic = metadata[
        (metadata["phenotype"] == "CASE")
        & (~metadata["ClinReport_Mutations_Details"].isin(known_mutations))
        & (
            (metadata["C9orf72_repeat_length"] < 30)
            | metadata["C9orf72_repeat_length"].isna()
        )
    ]
    control = metadata[metadata["phenotype"] == "CTRL"]
    return {
        "C9ORF72": c9.index,
        "SPORADIC": sporadic.index,
        "CONTROL": control.index,
    }


def _fetch_term_memberships() -> dict[str, dict[str, Any]]:
    """Fetch full HGNC memberships using the production library semantics."""
    from indra.databases.identifiers import get_ns_id_from_identifiers
    from indra.ontology.bio import bio_ontology
    from indra_cogex.representation import norm_id

    from cliquefinder.knowledge.cogex import CoGExClient

    bio_ontology.initialize()
    go_query = """
    MATCH (gene:BioEntity)-[:associated_with]->(term:BioEntity)
    WHERE term.id IN $ids
      AND gene.id STARTS WITH 'hgnc:'
      AND NOT gene.obsolete
    RETURN DISTINCT gene.id AS hgnc_id
    """
    reactome_query = """
    MATCH (pathway:BioEntity {id: $id})-[:haspart]-(gene:BioEntity)
    WHERE gene.id STARTS WITH 'hgnc:'
      AND NOT gene.obsolete
    RETURN DISTINCT gene.id AS hgnc_id
    """

    fetched: dict[str, dict[str, Any]] = {}
    with CoGExClient(env_file=ROOT / ".env") as client:
        for cluster, term_id, name in TERMS:
            if term_id.startswith("go:"):
                graph_ns, graph_id = term_id.split(":", maxsplit=1)
                db_ns, db_id = get_ns_id_from_identifiers(graph_ns, graph_id)
                children = bio_ontology.get_children(db_ns, db_id)
                queried_terms = sorted(
                    {term_id}
                    | {
                        norm_id(child_ns, child_id)
                        for child_ns, child_id in children
                    }
                )
                rows = client._execute_query(go_query, ids=queried_terms)
                semantics = "direct GO annotations union all ontology descendants"
            elif term_id.startswith("reactome:"):
                queried_terms = [term_id]
                rows = client._execute_query(reactome_query, id=term_id)
                semantics = "direct Reactome haspart membership"
            else:  # pragma: no cover - TERMS is a frozen constant
                raise ValueError(f"Unsupported term namespace: {term_id}")

            hgnc_ids = sorted(
                {
                    str(row[0]).removeprefix("hgnc:")
                    for row in rows
                    if row and row[0] and str(row[0]).startswith("hgnc:")
                },
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            )
            fetched[term_id] = {
                "cluster": cluster,
                "name": name,
                "membership_semantics": semantics,
                "queried_term_ids": queried_terms,
                "library_hgnc_ids": hgnc_ids,
                "library_hgnc_count": len(hgnc_ids),
            }
    return fetched


def refresh_frozen_inputs(path: Path = FROZEN_INPUTS_PATH) -> dict[str, Any]:
    """Refresh and freeze feature mapping plus pathway memberships."""
    from indra.databases import uniprot_client

    feature_ids = list(
        pd.read_csv(DATA_PATH, index_col=0, usecols=[0]).index.astype(str)
    )
    feature_to_hgnc: dict[str, str] = {}
    unmapped: list[str] = []
    for feature_id in feature_ids:
        hgnc_id = uniprot_client.get_hgnc_id(feature_id)
        if hgnc_id:
            feature_to_hgnc[feature_id] = str(hgnc_id)
        else:
            unmapped.append(feature_id)

    hgnc_to_features: dict[str, list[str]] = defaultdict(list)
    for feature_id, hgnc_id in feature_to_hgnc.items():
        hgnc_to_features[hgnc_id].append(feature_id)
    duplicate_hgnc = {
        hgnc_id: sorted(features)
        for hgnc_id, features in hgnc_to_features.items()
        if len(features) > 1
    }

    memberships = _fetch_term_memberships()
    measured_hgnc = set(hgnc_to_features)
    for record in memberships.values():
        measured = sorted(
            set(record["library_hgnc_ids"]) & measured_hgnc,
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        )
        record["measured_hgnc_ids"] = measured
        record["measured_hgnc_count"] = len(measured)

    frozen = {
        "schema_version": 1,
        "analysis_id": "c9-size-matched-null-log2-eb-2026-07",
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_files": {
            "proteomics": {
                "path": str(DATA_PATH.relative_to(ROOT)),
                "sha256": sha256_file(DATA_PATH),
            },
            "metadata": {
                "path": str(METADATA_PATH.relative_to(ROOT)),
                "sha256": sha256_file(METADATA_PATH),
            },
        },
        "feature_hgnc_mapping": {
            "source": "indra.databases.uniprot_client.get_hgnc_id",
            "aggregation": "maximum t^2 across UniProt features for each HGNC ID",
            "feature_count": len(feature_ids),
            "mapped_feature_count": len(feature_to_hgnc),
            "unique_hgnc_count": len(hgnc_to_features),
            "unmapped_features": sorted(unmapped),
            "duplicate_hgnc_to_features": duplicate_hgnc,
            "feature_to_hgnc": dict(sorted(feature_to_hgnc.items())),
        },
        "term_membership_source": {
            "service": "INDRA CoGEx",
            "go_ontology_cache": "INDRA bio ontology 1.34",
            "note": (
                "Full library membership and measured intersections are frozen; "
                "ordinary analysis reruns do not query CoGEx."
            ),
        },
        "terms": memberships,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def load_and_validate_frozen_inputs(path: Path) -> dict[str, Any]:
    """Load frozen inputs and reject source-file or term-family drift."""
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected_terms = [term_id for _, term_id, _ in TERMS]
    if list(frozen["terms"]) != expected_terms:
        raise ValueError("Frozen term order/family does not match TERMS")

    for key, source_path in (
        ("proteomics", DATA_PATH),
        ("metadata", METADATA_PATH),
    ):
        observed = sha256_file(source_path)
        expected = frozen["source_files"][key]["sha256"]
        if observed != expected:
            raise ValueError(
                f"{key} input hash drift: expected {expected}, observed {observed}; "
                "review the input and explicitly refresh frozen inputs"
            )
    return frozen


def fit_contrast_t2(
    data: np.ndarray,
    feature_ids: list[str],
    metadata: pd.DataFrame,
    groups: dict[str, pd.Index],
    case: str,
    reference: str,
    feature_to_hgnc: dict[str, str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit production EB statistics and aggregate finite ``t^2`` by HGNC."""
    selected = groups[case].union(groups[reference])
    sample_metadata = metadata.loc[metadata.index.intersection(selected)].copy()
    sample_metadata["_condition"] = None
    sample_metadata.loc[
        sample_metadata.index.isin(groups[case]), "_condition"
    ] = case
    sample_metadata.loc[
        sample_metadata.index.isin(groups[reference]), "_condition"
    ] = reference
    sample_metadata = sample_metadata.dropna(subset=["_condition"])

    index_by_sample = {sample: index for index, sample in enumerate(metadata.index)}
    sample_indices = [index_by_sample[sample] for sample in sample_metadata.index]
    log2_data = np.log2(data[:, sample_indices] + 1.0)
    if not np.all(np.isfinite(log2_data)):
        raise ValueError("log2(x+1) produced non-finite values")

    engine = RotationTestEngine(
        log2_data.copy(), list(feature_ids), sample_metadata.copy()
    )
    engine.fit(
        conditions=[case, reference],
        contrast=(case, reference),
        condition_column="_condition",
        covariates=["Sex"],
    )
    effects = engine._effects
    precomputed = engine._precomputed
    if effects is None or precomputed is None:  # pragma: no cover - engine contract
        raise RuntimeError("RotationTestEngine did not expose fitted effects")
    if effects.moderated_variances is None:
        raise RuntimeError("Production fit did not produce moderated variances")

    standard_error = np.sqrt(effects.moderated_variances)
    t_statistics = effects.U[:, 0] / standard_error
    t2_by_hgnc: dict[str, float] = {}
    finite_feature_count = 0
    mapped_finite_feature_count = 0
    for feature_id, t_statistic in zip(effects.gene_ids, t_statistics):
        if not np.isfinite(t_statistic):
            continue
        finite_feature_count += 1
        hgnc_id = feature_to_hgnc.get(feature_id)
        if hgnc_id is None:
            continue
        mapped_finite_feature_count += 1
        t2 = float(t_statistic * t_statistic)
        t2_by_hgnc[hgnc_id] = max(t2, t2_by_hgnc.get(hgnc_id, -math.inf))

    diagnostics = {
        "sample_count": len(sample_metadata),
        "arm_counts": {
            case: int((sample_metadata["_condition"] == case).sum()),
            reference: int(
                (sample_metadata["_condition"] == reference).sum()
            ),
        },
        "engine_fitted_feature_count": len(effects.gene_ids),
        "finite_feature_count": finite_feature_count,
        "mapped_finite_feature_count": mapped_finite_feature_count,
        "finite_unique_hgnc_count": len(t2_by_hgnc),
        "df_residual": int(precomputed.df_residual),
        "eb_d0": float(precomputed.eb_d0),
        "eb_s0_sq": float(precomputed.eb_s0_sq),
        "eb_df_total": float(precomputed.eb_df_total),
    }
    return t2_by_hgnc, diagnostics


def sample_null_means(
    background_values: np.ndarray,
    set_size: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Sample independent same-size sets without replacement."""
    if set_size <= 0 or set_size > len(background_values):
        raise ValueError("set_size must be within the finite background")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return np.fromiter(
        (
            float(
                np.mean(
                    rng.choice(background_values, size=set_size, replace=False)
                )
            )
            for _ in range(n_permutations)
        ),
        dtype=np.float64,
        count=n_permutations,
    )


def run_analysis(
    frozen: dict[str, Any],
    *,
    n_permutations: int = N_PERMUTATIONS,
    master_seed: int = MASTER_SEED,
    frozen_inputs_path: Path = FROZEN_INPUTS_PATH,
) -> dict[str, Any]:
    """Fit all contrasts and evaluate the eight-term null family."""
    data, feature_ids, metadata, groups = load_panel_inputs(
        data_path=DATA_PATH,
        metadata_path=METADATA_PATH,
        group_resolver=resolve_groups,
    )
    cohort_counts = {name: len(index) for name, index in groups.items()}
    expected_counts = {"C9ORF72": 25, "SPORADIC": 294, "CONTROL": 91}
    if cohort_counts != expected_counts:
        raise ValueError(
            f"Production cohort drift: expected {expected_counts}, got {cohort_counts}"
        )

    feature_map = frozen["feature_hgnc_mapping"]["feature_to_hgnc"]
    rows: list[dict[str, Any]] = []
    contrast_diagnostics: dict[str, dict[str, Any]] = {}
    alpha_per_term = FAMILYWISE_ALPHA / len(TERMS)

    for contrast_name, case, reference in CONTRASTS:
        t2_by_hgnc, diagnostics = fit_contrast_t2(
            data,
            feature_ids,
            metadata,
            groups,
            case,
            reference,
            feature_map,
        )
        contrast_diagnostics[contrast_name] = diagnostics
        background_hgnc = sorted(
            t2_by_hgnc,
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        )
        background_values = np.array(
            [t2_by_hgnc[hgnc_id] for hgnc_id in background_hgnc],
            dtype=np.float64,
        )
        diagnostics["background_mean_t2"] = float(np.mean(background_values))
        diagnostics["background_median_t2"] = float(
            np.median(background_values)
        )

        for cluster, term_id, term_name in TERMS:
            frozen_term = frozen["terms"][term_id]
            measured_members = list(frozen_term["measured_hgnc_ids"])
            in_background = sorted(set(measured_members) & set(t2_by_hgnc))
            if len(in_background) != len(measured_members):
                missing = sorted(set(measured_members) - set(in_background))
                raise ValueError(
                    f"Finite background lost frozen members for {term_id}: {missing}"
                )
            observed = float(
                np.mean([t2_by_hgnc[hgnc_id] for hgnc_id in in_background])
            )
            null_seed = stable_seed(master_seed, contrast_name, term_id)
            null_means = sample_null_means(
                background_values,
                len(in_background),
                n_permutations,
                null_seed,
            )
            exceedances = int(np.sum(null_means >= observed))
            empirical_p = (exceedances + 1) / (n_permutations + 1)
            rows.append(
                {
                    "cluster": cluster,
                    "term_id": term_id,
                    "term": term_name,
                    "contrast": contrast_name,
                    "set_size": len(in_background),
                    "background_size": len(background_values),
                    "observed_mean_t2": observed,
                    "null_median": float(np.median(null_means)),
                    "null_p95": float(np.quantile(null_means, 0.95)),
                    "null_exceedances": exceedances,
                    "empirical_p": empirical_p,
                    "bonferroni_p": min(empirical_p * len(TERMS), 1.0),
                    "bonferroni_pass": bool(empirical_p < alpha_per_term),
                    "null_seed": null_seed,
                }
            )

    pass_counts = {
        contrast: sum(
            row["bonferroni_pass"]
            for row in rows
            if row["contrast"] == contrast
        )
        for contrast, _, _ in CONTRASTS
    }
    return {
        "schema_version": 1,
        "analysis_id": "c9-size-matched-null-log2-eb-2026-07",
        "status": "canonical_auxiliary",
        # Deliberately inherit the frozen-input timestamp instead of stamping
        # wall-clock run time.  An offline reproduction is byte-identical.
        "generated_from_frozen_at_utc": frozen["frozen_at_utc"],
        "design": {
            "intensity_transform": "log2(x+1)",
            "per_feature_model": "condition + Sex",
            "per_feature_engine": (
                "cliquefinder.stats.rotation.RotationTestEngine"
            ),
            "statistic": "mean empirical-Bayes moderated t^2",
            "analysis_unit": "HGNC gene",
            "feature_to_hgnc_aggregation": (
                "maximum t^2 across UniProt features for each HGNC ID"
            ),
            "null": (
                "uniform same-size HGNC sets sampled without replacement "
                "from each contrast's finite moderated-t background"
            ),
            "alternative": "observed pathway mean t^2 is elevated",
            "n_permutations": n_permutations,
            "master_seed": master_seed,
            "bit_generator": "numpy.random.PCG64DXSM",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "term_count": len(TERMS),
            "bonferroni_alpha_per_term": alpha_per_term,
            "term_selection_status": (
                "discovery-derived on this cohort, then fixed before the "
                "canonical method-transfer reruns"
            ),
            "post_selection_fwer_guarantee": False,
            "selective_inference_correction": False,
            "empirical_p_formula": "(1 + count(null >= observed)) / (N + 1)",
        },
        "source_files": frozen["source_files"],
        "generator": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_inputs": {
            "path": portable_path(frozen_inputs_path),
            "sha256": sha256_file(frozen_inputs_path),
            "frozen_at_utc": frozen["frozen_at_utc"],
        },
        "cohort": {
            "metadata_matched_samples": len(metadata),
            "primary_arm_samples": sum(cohort_counts.values()),
            "arms": cohort_counts,
        },
        "feature_hgnc_accounting": {
            key: value
            for key, value in frozen["feature_hgnc_mapping"].items()
            if key != "feature_to_hgnc"
        },
        "contrast_diagnostics": contrast_diagnostics,
        "pass_counts_bonferroni_8": pass_counts,
        "contrast_order": [name for name, _, _ in CONTRASTS],
        "term_order": [term_id for _, term_id, _ in TERMS],
        "rows": rows,
        "interpretation_limits": [
            (
                "Controls pathway-size and finite-background heavy-tail effects "
                "within each contrast."
            ),
            (
                "Does not make overlapping pathway terms independent or "
                "provide a post-selection FWER/selective-inference guarantee."
            ),
            (
                "Does not by itself remove the contrast sample-size imbalance "
                "or license a causal, mechanistic, or external-cohort claim."
            ),
            "Contains no network or graph input.",
        ],
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _cell(row: dict[str, Any]) -> str:
    marker = "✓" if row["bonferroni_pass"] else "—"
    return (
        f"{_fmt(row['observed_mean_t2'])} "
        f"({_fmt(row['null_median'])} / {_fmt(row['null_p95'])}); "
        f"p={row['empirical_p']:.5f}; {marker}"
    )


def render_markdown(result: dict[str, Any]) -> str:
    """Render a concise publication-facing audit record."""
    n_permutations = int(result["design"]["n_permutations"])
    rows_by_key = {
        (row["term_id"], row["contrast"]): row for row in result["rows"]
    }
    lines = [
        "# Canonical size-matched HGNC-set null",
        "",
        "**Status:** canonical auxiliary analysis regenerated on the production "
        "log2(x+1) scale.",
        "",
        "## Design",
        "",
        "For each contrast, `RotationTestEngine` fits "
        "`log2(intensity+1) ~ condition + Sex` and supplies the production "
        "empirical-Bayes moderated *t*. UniProt rows are mapped to HGNC IDs; "
        "the sole duplicated HGNC measurement is aggregated by maximum "
        "*t*² so every random-set unit is one gene. For each of the eight "
        "discovery-derived fixed terms, the observed mean *t*² is compared with "
        f"{n_permutations:,} uniform same-size HGNC sets sampled without replacement from "
        "that contrast's finite moderated-*t* background.",
        "",
        "One-tailed empirical p = `(1 + count(null >= observed)) / "
        f"{n_permutations + 1:,}`. "
        "The family readout is Bonferroni-8: raw p < 0.00625.",
        "",
        "## Input and fit accounting",
        "",
        f"- Source matrix: {result['feature_hgnc_accounting']['feature_count']:,} "
        "feature rows (3,263 human UniProt rows plus the iRT standard); "
        f"{result['feature_hgnc_accounting']['mapped_feature_count']:,} mapped "
        "rows collapse to "
        f"{result['feature_hgnc_accounting']['unique_hgnc_count']:,} HGNC genes; "
        f"{len(result['feature_hgnc_accounting']['unmapped_features'])} unmapped.",
        f"- Metadata-matched samples: {result['cohort']['metadata_matched_samples']}; "
        "primary arms: C9 = 25, Sporadic = 294, Control = 91.",
    ]
    for contrast in result["contrast_order"]:
        diagnostic = result["contrast_diagnostics"][contrast]
        lines.append(
            f"- {contrast}: n={diagnostic['sample_count']}; "
            f"HGNC background={diagnostic['finite_unique_hgnc_count']:,}; "
            f"EB d0={diagnostic['eb_d0']:.4f}; "
            f"background mean/median *t*²="
            f"{diagnostic['background_mean_t2']:.3f}/"
            f"{diagnostic['background_median_t2']:.3f}."
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "Each cell is `observed mean t² (null median / null 95th); "
            "raw empirical p; Bonferroni-8 pass`.",
            "",
            "| Cluster | Term | n | C9 vs Sporadic | C9 vs Control | "
            "Sporadic vs Control |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for cluster, term_id, term_name in TERMS:
        first = rows_by_key[(term_id, result["contrast_order"][0])]
        cells = [
            _cell(rows_by_key[(term_id, contrast)])
            for contrast in result["contrast_order"]
        ]
        lines.append(
            f"| {cluster} | {term_name} | {first['set_size']} | "
            + " | ".join(cells)
            + " |"
        )

    pattern = [
        result["pass_counts_bonferroni_8"][contrast]
        for contrast in result["contrast_order"]
    ]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"The Bonferroni-8 pass pattern is **{pattern[0]}/{pattern[1]}/"
            f"{pattern[2]}** for C9-vs-Sporadic / C9-vs-Control / "
            "Sporadic-vs-Control.",
            "",
            "The terms were discovered on this same cohort and then fixed for "
            "method transfer. The eightfold threshold handles arithmetic "
            "multiplicity across the reported rerun tests; it is not a "
            "post-selection FWER or selective-inference guarantee.",
            "",
            "This graph-independent analysis controls pathway size and the "
            "heavy-tailed moderated-*t*² background within each contrast. "
            "It does not make the overlapping terms independent, remove the "
            "contrast sample-size imbalance, or license causal, mechanistic, "
            "individual-protein, or external-cohort claims.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run --no-sync python scripts/run_size_matched_null.py",
            "```",
            "",
            "The default run is offline: the full term libraries, measured "
            "intersections, and UniProt-to-HGNC map are frozen in "
            "`data/publication/c9_size_matched_null_inputs.json`. Refreshing "
            "those inputs is a separate, explicit network operation via "
            "`--refresh-frozen-inputs`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], prefix: Path) -> None:
    """Write JSON, tidy CSV, and Markdown from one result object."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(result["rows"][0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    markdown_path.write_text(render_markdown(result), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-inputs",
        type=Path,
        default=FROZEN_INPUTS_PATH,
        help="Frozen membership/mapping JSON.",
    )
    parser.add_argument(
        "--refresh-frozen-inputs",
        action="store_true",
        help="Explicitly refresh memberships and feature mapping from INDRA.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output stem; .json, .csv, and .md are written.",
    )
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_permutations <= 0:
        raise ValueError("--n-permutations must be positive")
    writes_canonical = args.output_prefix.expanduser().resolve() == (
        DEFAULT_OUTPUT_PREFIX.resolve()
    )
    canonical_configuration = (
        args.n_permutations == N_PERMUTATIONS
        and args.seed == MASTER_SEED
        and args.frozen_inputs.expanduser().resolve() == FROZEN_INPUTS_PATH.resolve()
    )
    if writes_canonical and not canonical_configuration:
        raise ValueError(
            "Noncanonical inputs, seed, or permutation count require an explicit "
            "alternate --output-prefix; refusing to overwrite publication artifacts."
        )
    if args.refresh_frozen_inputs:
        print(f"Refreshing frozen inputs: {args.frozen_inputs}", flush=True)
        refresh_frozen_inputs(args.frozen_inputs)
    frozen = load_and_validate_frozen_inputs(args.frozen_inputs)
    print(
        "Fitting production log2(x+1), Sex-adjusted EB statistics...",
        flush=True,
    )
    result = run_analysis(
        frozen,
        n_permutations=args.n_permutations,
        master_seed=args.seed,
        frozen_inputs_path=args.frozen_inputs,
    )
    write_outputs(result, args.output_prefix)
    pattern = [
        result["pass_counts_bonferroni_8"][contrast]
        for contrast in result["contrast_order"]
    ]
    print(
        f"Wrote {args.output_prefix}.{{json,csv,md}}; "
        f"Bonferroni-8 pattern={pattern[0]}/{pattern[1]}/{pattern[2]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
