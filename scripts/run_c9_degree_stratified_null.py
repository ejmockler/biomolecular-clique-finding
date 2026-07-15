#!/usr/bin/env python3
"""Run the canonical C9 F5a pathway-level degree-matched anchor null.

This is a secondary, pathway-level location test over eight terms discovered
on this cohort and then fixed before the measured-only/log2 method transfer.
It consumes the already-computed ``-slope`` anchor scores from the canonical
log2(x+1), measured-only, bounded-h2 landscapes.  It does *not* use the
per-anchor ``slope_pvalue`` field and it does *not* run or approximate GSEA.

The May 2026 F5a check drew one non-term control for each term member inside a
reciprocal 20% degree window and applied a one-sided Mann-Whitney test.  This
durable rerun keeps that matching estimand but removes dependence on one
deduplicated random draw: for every member with at least one eligible control,
it samples one degree-comparable nonmember with replacement in each of many
Monte Carlo replicates.  The fixed/canonical endpoint is the mean ``-slope``
in the fixed/canonical scope (hop-1 size >= 20).  The May all-valid-anchor
scope, a median statistic, and matching on measured hop-1 shell size are
reported as sensitivities.

The default run is offline.  Use ``--refresh-terms`` only to deliberately
replace the frozen INDRA CoGEx term-membership snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS = {
    "C9 vs Sporadic": ROOT / "output/landscape_proteome_measured_only_log2/result.json",
    "C9 vs Control": ROOT / "output/landscape_c9_vs_control_measured_only_log2/result.json",
    "Sporadic vs Control": ROOT
    / "output/landscape_sporadic_vs_control_measured_only_log2/result.json",
}
EXPECTED_CONTRASTS = {
    "C9 vs Sporadic": ("C9ORF72", "SPORADIC"),
    "C9 vs Control": ("C9ORF72", "CONTROL"),
    "Sporadic vs Control": ("SPORADIC", "CONTROL"),
}
DEFAULT_DEGREE_META = ROOT / "output/landscape_proteome_measured_only_log2/distances.meta.json"
DEFAULT_TERM_SNAPSHOT = ROOT / "data/publication/c9_degree_stratified_null_terms.json"
DEFAULT_OUTPUT_JSON = ROOT / "data/publication/c9_degree_stratified_null.json"
DEFAULT_OUTPUT_MD = ROOT / "data/publication/c9_degree_stratified_null.md"

# Ordered exactly as the confirmatory family.  Term membership is fetched by
# the same enrichment utilities used by scripts/run_landscape_gsea.py.
TERMS = [
    ("Splicing", "reactome:R-HSA-72172", "mRNA Splicing", "reactome"),
    (
        "Splicing",
        "reactome:R-HSA-72203",
        "Processing Capped Pre-mRNA",
        "reactome",
    ),
    ("Splicing", "go:0000398", "mRNA splicing, via spliceosome", "go"),
    ("Chromatin", "go:0005694", "chromosome", "go"),
    ("Chromatin", "go:0000785", "chromatin", "go"),
    ("Transport", "go:0006913", "nucleocytoplasmic transport", "go"),
    ("Transport", "go:0005643", "nuclear pore", "go"),
    (
        "Transport",
        "reactome:R-HSA-180910",
        "Vpr-mediated nuclear import",
        "reactome",
    ),
]

NOMINAL_REFERENCE_ALPHA = 0.05
N_TERMS = len(TERMS)
EIGHTFOLD_RAW_P_THRESHOLD = NOMINAL_REFERENCE_ALPHA / N_TERMS
DEGREE_RATIO_LOWER = 0.8
DEGREE_RATIO_UPPER = 1.0 / DEGREE_RATIO_LOWER
PRIMARY_DEGREE_METRIC = "full_indra_degree"
SENSITIVITY_DEGREE_METRIC = "measured_hop1_size"
PRIMARY_ANCHOR_SCOPE = "robust_hop1_ge_20"
LEGACY_ANCHOR_SCOPE = "all_valid"
N_REPLICATES = 9999
BASE_SEED = 20260712


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _portable_path(path: Path) -> str:
    """Record a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\x1f".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _split_uniprots(raw: Any) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def refresh_term_snapshot(path: Path) -> dict[str, Any]:
    """Fetch and freeze the exact CoGEx gene sets used by the GSEA helpers."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from indra.databases import hgnc_client
    from indra_cogex.client.enrichment.utils import get_go, get_reactome
    from indra_cogex.client.neo4j_client import Neo4jClient

    client = Neo4jClient()
    go_sets = get_go(client=client)
    reactome_sets = get_reactome(client=client)

    def find_set(term_id: str, source: dict[Any, Iterable[Any]]) -> tuple[str, set[str]]:
        for (curie, name), members in source.items():
            if str(curie).lower() == term_id.lower():
                return str(name), {str(member) for member in members}
        raise KeyError(f"Term {term_id!r} was absent from the CoGEx enrichment corpus")

    rows: list[dict[str, Any]] = []
    for cluster, term_id, display_name, database in TERMS:
        source = go_sets if database == "go" else reactome_sets
        source_name, hgnc_ids = find_set(term_id, source)
        uniprots: set[str] = set()
        unmapped_hgnc: list[str] = []
        for hgnc_id in sorted(hgnc_ids):
            mapped = _split_uniprots(hgnc_client.get_uniprot_id(hgnc_id))
            if mapped:
                uniprots.update(mapped)
            else:
                unmapped_hgnc.append(hgnc_id)
        rows.append(
            {
                "cluster": cluster,
                "term_id": term_id,
                "display_name": display_name,
                "source_name": source_name,
                "database": database,
                "hgnc_ids": sorted(hgnc_ids),
                "uniprot_ids": sorted(uniprots),
                "unmapped_hgnc_ids": unmapped_hgnc,
            }
        )

    snapshot = {
        "schema_version": 1,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": (
            "terms discovered on this cohort and fixed before measured-only/log2 "
            "method transfer; not prospectively preregistered"
        ),
        "source": {
            "name": "INDRA CoGEx enrichment corpus",
            "endpoint": "configured via environment; endpoint not serialized",
            "fetch_functions": [
                "indra_cogex.client.enrichment.utils.get_go",
                "indra_cogex.client.enrichment.utils.get_reactome",
            ],
            "indra_cogex_version": _package_version("indra-cogex"),
            "indra_version": _package_version("indra"),
        },
        "term_order": [term_id for _, term_id, _, _ in TERMS],
        "terms": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


@dataclass(frozen=True)
class AnchorTable:
    label: str
    score: dict[str, float]
    full_degree: dict[str, int]
    hop1_size: dict[str, int]
    design: dict[str, Any]


def load_anchor_table(
    label: str,
    result_path: Path,
    degree_meta: dict[str, Any],
) -> AnchorTable:
    """Load and strictly validate one canonical landscape."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    design = payload.get("design", {})
    if tuple(design.get("contrast", [])) != EXPECTED_CONTRASTS[label]:
        raise ValueError(
            f"{label}: expected contrast {EXPECTED_CONTRASTS[label]!r}, "
            f"found {design.get('contrast')!r}"
        )
    if design.get("transform") != "log2":
        raise ValueError(f"{label}: expected transform='log2', found {design.get('transform')!r}")
    if design.get("max_hops") != 2:
        raise ValueError(f"{label}: expected max_hops=2, found {design.get('max_hops')!r}")
    if design.get("covariates") != ["Sex"]:
        raise ValueError(
            f"{label}: expected covariates=['Sex'], found {design.get('covariates')!r}"
        )
    if design.get("n_permutations") != 999:
        raise ValueError(
            f"{label}: expected n_permutations=999, found {design.get('n_permutations')!r}"
        )
    records = payload.get("per_feature", [])
    if len(records) != 3117:
        raise ValueError(f"{label}: expected 3,117 valid anchors, found {len(records)}")

    full_degrees = degree_meta.get("graph_degrees", {})
    score: dict[str, float] = {}
    hop1_size: dict[str, int] = {}
    for record in records:
        seed = str(record["seed"])
        slope = float(record["slope"])
        if not math.isfinite(slope):
            raise ValueError(f"{label}: non-finite slope for {seed}")
        hop1 = next(
            (int(shell["n_genes"]) for shell in record.get("shells", []) if int(shell["hop"]) == 1),
            None,
        )
        if hop1 is None:
            raise ValueError(f"{label}: valid anchor {seed} has no hop-1 shell")
        if seed not in full_degrees:
            raise ValueError(f"{label}: full-INDRA degree is missing for {seed}")
        score[seed] = -slope
        hop1_size[seed] = hop1

    return AnchorTable(
        label=label,
        score=score,
        full_degree={seed: int(full_degrees[seed]) for seed in score},
        hop1_size=hop1_size,
        design=design,
    )


def candidate_scores_for_members(
    member_ids: list[str],
    nonmember_ids: list[str],
    scores: dict[str, float],
    degrees: dict[str, int],
    lower_ratio: float = DEGREE_RATIO_LOWER,
    upper_ratio: float = DEGREE_RATIO_UPPER,
) -> tuple[list[str], list[np.ndarray], list[int]]:
    """Return each matchable member and its degree-window control scores.

    The reciprocal window ``[0.8*d, 1.25*d]`` is the actual May rule.  It
    is symmetric on the multiplicative scale; calling it literal +/-20%
    would be inaccurate because the upper endpoint is +25%.
    """
    if not (0 < lower_ratio <= 1 <= upper_ratio):
        raise ValueError("degree-ratio window must contain 1 and be positive")
    matched_members: list[str] = []
    candidate_scores: list[np.ndarray] = []
    candidate_counts: list[int] = []
    for member in member_ids:
        degree = degrees[member]
        if degree == 0:
            candidates = [control for control in nonmember_ids if degrees[control] == 0]
        else:
            low = degree * lower_ratio
            high = degree * upper_ratio
            candidates = [control for control in nonmember_ids if low <= degrees[control] <= high]
        if not candidates:
            continue
        matched_members.append(member)
        candidate_scores.append(
            np.asarray([scores[control] for control in candidates], dtype=float)
        )
        candidate_counts.append(len(candidates))
    return matched_members, candidate_scores, candidate_counts


def monte_carlo_matched_null(
    observed_scores: np.ndarray,
    candidate_scores: list[np.ndarray],
    n_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Evaluate mean and median term scores against matched reference draws."""
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if len(observed_scores) != len(candidate_scores) or not candidate_scores:
        raise ValueError("one non-empty candidate array is required per observed score")

    draws = np.empty((n_replicates, len(candidate_scores)), dtype=float)
    for column, candidates in enumerate(candidate_scores):
        if candidates.size == 0:
            raise ValueError("candidate arrays must not be empty")
        indices = rng.integers(0, candidates.size, size=n_replicates)
        draws[:, column] = candidates[indices]

    null_mean = draws.mean(axis=1)
    null_median = np.median(draws, axis=1)
    observed_mean = float(np.mean(observed_scores))
    observed_median = float(np.median(observed_scores))

    def summarize(null: np.ndarray, observed: float) -> dict[str, float]:
        null_center = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1))
        return {
            "observed": observed,
            "null_mean": null_center,
            "null_sd": null_sd,
            "null_ci95_low": float(np.quantile(null, 0.025)),
            "null_ci95_high": float(np.quantile(null, 0.975)),
            "difference": observed - null_center,
            "z_against_null": ((observed - null_center) / null_sd if null_sd > 0 else math.inf),
            "empirical_p_greater": float(
                (np.count_nonzero(null >= observed) + 1) / (n_replicates + 1)
            ),
        }

    return {
        "mean_score": summarize(null_mean, observed_mean),
        "median_score": summarize(null_median, observed_median),
    }


def may_single_draw_mwu(
    member_ids: list[str],
    nonmember_ids: list[str],
    scores: dict[str, float],
    degrees: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Reproduce the May one-draw, deduplicated, one-sided MWU check.

    This is retained for historical comparability only.  The repeated matched
    reference in :func:`monte_carlo_matched_null` is the inferential endpoint.
    Sorting IDs makes this rerun stable; the transcript's ``list(set(...))``
    ordering was process-dependent.
    """
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    selected_members: list[str] = []
    for member in member_ids:
        degree = degrees[member]
        if degree == 0:
            candidates = [control for control in nonmember_ids if degrees[control] == 0]
        else:
            low = degree * DEGREE_RATIO_LOWER
            high = degree * DEGREE_RATIO_UPPER
            candidates = [control for control in nonmember_ids if low <= degrees[control] <= high]
        if candidates:
            selected_members.append(member)
            selected.append(candidates[int(rng.integers(0, len(candidates)))])

    # The May transcript selected with replacement, then used ``isin`` to form
    # the control frame, silently deduplicating repeated picks.
    unique_controls = sorted(set(selected))
    observed = np.asarray([scores[member] for member in selected_members], dtype=float)
    controls = np.asarray([scores[control] for control in unique_controls], dtype=float)
    if observed.size < 2 or controls.size < 2:
        raise ValueError("May-style MWU requires at least two values per group")
    statistic, p_value = mannwhitneyu(observed, controls, alternative="greater")
    return {
        "role": "historical_sensitivity_not_primary",
        "sampling": "one control draw per member with replacement, then deduplicated",
        "alternative": "term -slope greater than matched-control -slope",
        "n_term": int(observed.size),
        "n_unique_controls": int(controls.size),
        "observed_median": float(np.median(observed)),
        "control_median": float(np.median(controls)),
        "mann_whitney_u": float(statistic),
        "raw_p_greater": float(p_value),
    }


def analyze_term(
    table: AnchorTable,
    term: dict[str, Any],
    anchor_scope: str,
    degree_metric: str,
    n_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if anchor_scope == PRIMARY_ANCHOR_SCOPE:
        eligible = {anchor for anchor, hop1 in table.hop1_size.items() if hop1 >= 20}
    elif anchor_scope == LEGACY_ANCHOR_SCOPE:
        eligible = set(table.score)
    else:
        raise ValueError(f"Unknown anchor scope: {anchor_scope}")
    term_members = eligible.intersection(term["uniprot_ids"])
    member_ids = sorted(term_members)
    nonmember_ids = sorted(eligible.difference(term_members))
    degrees = table.full_degree if degree_metric == PRIMARY_DEGREE_METRIC else table.hop1_size
    matched_members, candidates, candidate_counts = candidate_scores_for_members(
        member_ids,
        nonmember_ids,
        table.score,
        degrees,
    )
    if len(matched_members) < 5:
        raise ValueError(
            f"{table.label}/{term['term_id']}/{anchor_scope}/{degree_metric}: only "
            f"{len(matched_members)} degree-matchable members"
        )
    observed = np.asarray([table.score[member] for member in matched_members])
    result = monte_carlo_matched_null(
        observed,
        candidates,
        n_replicates,
        np.random.default_rng(seed),
    )
    may_legacy = None
    if anchor_scope == LEGACY_ANCHOR_SCOPE and degree_metric == PRIMARY_DEGREE_METRIC:
        may_legacy = may_single_draw_mwu(
            member_ids,
            nonmember_ids,
            table.score,
            degrees,
            _stable_seed(seed, "may_single_draw_mwu"),
        )
    return {
        "cluster": term["cluster"],
        "term_id": term["term_id"],
        "term": term["display_name"],
        "anchor_scope": anchor_scope,
        "degree_metric": degree_metric,
        "n_term_eligible": len(member_ids),
        "n_term_matched": len(matched_members),
        "match_coverage": len(matched_members) / len(member_ids),
        "small_term_n_lt_10": len(matched_members) < 10,
        "candidate_count_min": min(candidate_counts),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_max": max(candidate_counts),
        "may_legacy_single_draw_mwu": may_legacy,
        **result,
    }


def _add_eightfold_threshold(results: list[dict[str, Any]]) -> None:
    """Attach the fixed-family eightfold p threshold to each result.

    This arithmetic adjustment is not a post-selection FWER guarantee: the
    eight terms were discovered on this cohort before being fixed for method
    transfer.
    """
    for row in results:
        for statistic in ("mean_score", "median_score"):
            p_value = row[statistic]["empirical_p_greater"]
            adjusted = min(p_value * N_TERMS, 1.0)
            row[statistic]["eightfold_p"] = adjusted
            row[statistic]["meets_eightfold_threshold"] = bool(
                row[statistic]["difference"] > 0 and adjusted < NOMINAL_REFERENCE_ALPHA
            )
        legacy = row.get("may_legacy_single_draw_mwu")
        if legacy is not None:
            adjusted = min(legacy["raw_p_greater"] * N_TERMS, 1.0)
            legacy["eightfold_p"] = adjusted
            legacy["meets_eightfold_threshold"] = bool(adjusted < NOMINAL_REFERENCE_ALPHA)


def _format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.4f}"


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Canonical F5a: degree-matched pathway anchor null",
        "",
        "**Status:** Current secondary analysis on the canonical "
        "log2(x+1), measured-only, bounded-h2 landscapes.",
        "",
        "## What this test is",
        "",
        "For each of eight terms discovered on this cohort and then fixed "
        "before the measured-only/log2 method transfer, this analysis asks "
        "whether member anchors have a larger pathway-level location of "
        "`-slope` than nonmember anchors with comparable regulatory-network "
        "degree. A larger `-slope` means higher mean moderated-|t| in the "
        "measured hop-1 shell than in the measured hop-2 shell.",
        "",
        "The fixed/canonical anchor scope is the same robust ranking used for "
        "the canonical fixed-term GSEA: valid anchors with at least 20 measured "
        "hop-1 neighbors. The May all-valid-anchor scope is retained as a "
        "sensitivity. The fixed/canonical match variable is the persisted "
        "full-INDRA "
        "degree used by the production per-anchor degree-preserving permutation; "
        "matching on measured hop-1 shell size is another sensitivity. The "
        "actual May window is retained: control/member degree ratio in "
        "[0.8, 1.25]. One matched control is sampled per member, with replacement, in each "
        f"of {artifact['analysis']['n_replicates']:,} deterministic Monte Carlo "
        "replicates. The fixed/canonical endpoint is the term mean `-slope`; "
        "the median is a sensitivity. Raw one-sided empirical p-values are "
        "multiplied by eight and compared with the nominal 0.05 reference, "
        "separately within each contrast, anchor scope, degree metric, and "
        "reported statistic.",
        "",
        "> **Selection integrity:** these eight terms were derived from "
        "discovery on the same cohort and were not prospectively "
        "preregistered. The eightfold threshold handles only the arithmetic "
        "multiplicity across the fixed list during this method-transfer check; "
        "it is not a post-selection FWER guarantee or selective-inference "
        "correction.",
        "",
        "> This is not an analysis of `slope_pvalue`, and it is not GSEA. It "
        "does not produce or validate an NES. It is a same-data conditional "
        "sensitivity asking whether the simple degree-matched reference "
        "reproduces the term-level location shift in the already-computed "
        "anchor scores.",
        "",
        "## Fixed/canonical result: matching on full-INDRA degree",
        "",
    ]

    for label in EXPECTED_CONTRASTS:
        block = artifact["contrasts"][label]
        rows = [
            row
            for row in block["results"]
            if row["anchor_scope"] == PRIMARY_ANCHOR_SCOPE
            and row["degree_metric"] == PRIMARY_DEGREE_METRIC
        ]
        lines.extend(
            [
                f"### {label}",
                "",
                "| Term | n matched | observed mean | matched-null mean "
                "(95% interval) | empirical p | eightfold p | meets threshold |",
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in rows:
            value = row["mean_score"]
            lines.append(
                f"| {row['term']} | {row['n_term_matched']} | "
                f"{value['observed']:+.4f} | {value['null_mean']:+.4f} "
                f"[{value['null_ci95_low']:+.4f}, "
                f"{value['null_ci95_high']:+.4f}] | "
                f"{_format_p(value['empirical_p_greater'])} | "
                f"{_format_p(value['eightfold_p'])} | "
                f"{'yes' if value['meets_eightfold_threshold'] else 'no'} |"
            )
        summary = block["summary"][PRIMARY_ANCHOR_SCOPE][PRIMARY_DEGREE_METRIC]
        lines.extend(
            [
                "",
                f"**Pattern:** {summary['mean_eightfold_threshold_count']}/8 "
                "terms meet the fixed/canonical mean-score threshold; "
                f"{summary['median_eightfold_threshold_count']}/8 also meet "
                "the median-sensitivity threshold.",
                "",
            ]
        )

    c9_failures: dict[str, list[dict[str, Any]]] = {}
    for label in ("C9 vs Sporadic", "C9 vs Control"):
        c9_failures[label] = [
            row
            for row in artifact["contrasts"][label]["results"]
            if row["anchor_scope"] == PRIMARY_ANCHOR_SCOPE
            and row["degree_metric"] == PRIMARY_DEGREE_METRIC
            and not row["mean_score"]["meets_eightfold_threshold"]
        ]
    if all(
        len(rows) == 1 and rows[0]["term_id"] == "reactome:R-HSA-180910"
        for rows in c9_failures.values()
    ):
        spor_vpr = c9_failures["C9 vs Sporadic"][0]
        ctrl_vpr = c9_failures["C9 vs Control"][0]
        lines.extend(
            [
                "**Key nuance:** Vpr-mediated nuclear import is the sole "
                "fixed/canonical-scope C9 non-threshold term (only "
                f"{spor_vpr['n_term_matched']} eligible term members). Its "
                "one-sided mean p-values are "
                f"{spor_vpr['mean_score']['empirical_p_greater']:.4f} for "
                "C9-vs-sporadic and "
                f"{ctrl_vpr['mean_score']['empirical_p_greater']:.4f} for "
                "C9-vs-control, but neither meets the eightfold threshold. The "
                "fixed/canonical-scope result is therefore 7/8, not the May "
                "artifact's all-eight claim. The May all-valid scope still "
                "returns 8/8 in both C9 contrasts, but that scope cannot "
                "substitute for the robust ranking used by the canonical "
                "fixed-term GSEA.",
                "",
            ]
        )

    lines.extend(
        [
            "## Scope and degree sensitivities",
            "",
            "| Contrast | sensitivity | mean threshold count | median threshold "
            "count | minimum match coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    sensitivity_rows = [
        (
            PRIMARY_ANCHOR_SCOPE,
            SENSITIVITY_DEGREE_METRIC,
            "fixed/canonical scope; measured hop-1 degree",
        ),
        (
            LEGACY_ANCHOR_SCOPE,
            PRIMARY_DEGREE_METRIC,
            "May all-valid scope; full-INDRA degree",
        ),
        (
            LEGACY_ANCHOR_SCOPE,
            SENSITIVITY_DEGREE_METRIC,
            "all-valid scope; measured hop-1 degree",
        ),
    ]
    for label in EXPECTED_CONTRASTS:
        for anchor_scope, degree_metric, display in sensitivity_rows:
            summary = artifact["contrasts"][label]["summary"][anchor_scope][degree_metric]
            lines.append(
                f"| {label} | {display} | "
                f"{summary['mean_eightfold_threshold_count']}/8 | "
                f"{summary['median_eightfold_threshold_count']}/8 | "
                f"{summary['minimum_match_coverage']:.1%} |"
            )

    lines.extend(
        [
            "",
            "### Exact May-method reproduction",
            "",
            "The historical one-draw procedure (all valid anchors, "
            "full-INDRA degree, one match sampled per member, duplicate "
            "controls discarded, one-sided Mann-Whitney U) is retained only "
            "as a reproducibility sensitivity. After multiplying its raw "
            "p-values by eight, its threshold counts are:",
            "",
        ]
    )
    for label in EXPECTED_CONTRASTS:
        summary = artifact["contrasts"][label]["summary"][LEGACY_ANCHOR_SCOPE][
            PRIMARY_DEGREE_METRIC
        ]
        lines.append(f"- {label}: {summary['may_legacy_mwu_eightfold_threshold_count']}/8 terms.")
    lines.extend(
        [
            "",
            "Because that result depends on one random matched set and "
            "silently reduces its size when controls repeat, it is not the "
            "fixed/canonical endpoint.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            artifact["interpretation"],
            "",
            "This same-data control is conditional on the discovery-derived "
            "term list, frozen INDRA term membership, INDRA degree snapshot in "
            "the distance sidecar, and the same INDRA regulatory graph that "
            "generated the slopes. Its 7/7/0 pattern weighs against a simple "
            "degree-location explanation for seven terms in each C9 contrast, "
            "but does not rule that explanation out. It cannot establish "
            "network independence, biological causality, external replication, "
            "individual-anchor significance, or post-selection familywise "
            "error control. Term overlap also means the eight rows are not "
            "independent.",
            "",
            "Machine-readable values and full input hashes are in "
            "`data/publication/c9_degree_stratified_null.json`; frozen term "
            "members are in "
            "`data/publication/c9_degree_stratified_null_terms.json`.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run --no-sync python scripts/run_c9_degree_stratified_null.py",
            "```",
            "",
            "The default is offline and consumes the frozen term snapshot. "
            "`--refresh-terms` deliberately replaces that snapshot from CoGEx "
            "and should be treated as a corpus update, not an ordinary rerun.",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(
    result_paths: dict[str, Path],
    degree_meta_path: Path,
    term_snapshot_path: Path,
    n_replicates: int,
    base_seed: int,
) -> dict[str, Any]:
    degree_meta = json.loads(degree_meta_path.read_text(encoding="utf-8"))
    if degree_meta.get("path_traversal") != "measured_only":
        raise ValueError("degree sidecar must record path_traversal='measured_only'")
    if degree_meta.get("max_hops") != 2:
        raise ValueError("degree sidecar must record max_hops=2")
    snapshot = json.loads(term_snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("term_order") != [term_id for _, term_id, _, _ in TERMS]:
        raise ValueError("frozen term order does not match the confirmatory family")
    term_by_id = {row["term_id"]: row for row in snapshot["terms"]}

    tables = {
        label: load_anchor_table(label, result_paths[label], degree_meta)
        for label in EXPECTED_CONTRASTS
    }
    expected_meta_sha = _sha256(degree_meta_path)
    for label, result_path in result_paths.items():
        sibling_meta = result_path.with_name("distances.meta.json")
        if not sibling_meta.exists():
            raise FileNotFoundError(f"{label}: missing sidecar {sibling_meta}")
        if _sha256(sibling_meta) != expected_meta_sha:
            raise ValueError(f"{label}: distance sidecar differs from {degree_meta_path}")
    seed_sets = {label: set(table.score) for label, table in tables.items()}
    if any(seeds != next(iter(seed_sets.values())) for seeds in seed_sets.values()):
        raise ValueError("canonical contrasts do not share the same anchor universe")

    contrasts: dict[str, Any] = {}
    for label, table in tables.items():
        rows: list[dict[str, Any]] = []
        for anchor_scope in (PRIMARY_ANCHOR_SCOPE, LEGACY_ANCHOR_SCOPE):
            for degree_metric in (
                PRIMARY_DEGREE_METRIC,
                SENSITIVITY_DEGREE_METRIC,
            ):
                for _, term_id, _, _ in TERMS:
                    row = analyze_term(
                        table,
                        term_by_id[term_id],
                        anchor_scope,
                        degree_metric,
                        n_replicates,
                        _stable_seed(
                            base_seed,
                            label,
                            term_id,
                            anchor_scope,
                            degree_metric,
                        ),
                    )
                    rows.append(row)
        _add_eightfold_threshold(rows)
        summary: dict[str, Any] = {}
        for anchor_scope in (PRIMARY_ANCHOR_SCOPE, LEGACY_ANCHOR_SCOPE):
            summary[anchor_scope] = {}
            for degree_metric in (
                PRIMARY_DEGREE_METRIC,
                SENSITIVITY_DEGREE_METRIC,
            ):
                selected = [
                    row
                    for row in rows
                    if row["anchor_scope"] == anchor_scope and row["degree_metric"] == degree_metric
                ]
                summary[anchor_scope][degree_metric] = {
                    "mean_eightfold_threshold_count": sum(
                        row["mean_score"]["meets_eightfold_threshold"] for row in selected
                    ),
                    "median_eightfold_threshold_count": sum(
                        row["median_score"]["meets_eightfold_threshold"] for row in selected
                    ),
                    "minimum_match_coverage": min(row["match_coverage"] for row in selected),
                }
                if anchor_scope == LEGACY_ANCHOR_SCOPE and degree_metric == PRIMARY_DEGREE_METRIC:
                    summary[anchor_scope][degree_metric][
                        "may_legacy_mwu_eightfold_threshold_count"
                    ] = sum(
                        row["may_legacy_single_draw_mwu"]["meets_eightfold_threshold"]
                        for row in selected
                    )
        contrasts[label] = {
            "design": table.design,
            "n_valid_anchors": len(table.score),
            "n_scope_anchors": {
                PRIMARY_ANCHOR_SCOPE: sum(hop1 >= 20 for hop1 in table.hop1_size.values()),
                LEGACY_ANCHOR_SCOPE: len(table.score),
            },
            "result_path": _portable_path(result_paths[label]),
            "result_sha256": _sha256(result_paths[label]),
            "summary": summary,
            "results": rows,
        }

    c9_fixed_threshold_counts = {
        label: contrasts[label]["summary"][PRIMARY_ANCHOR_SCOPE][PRIMARY_DEGREE_METRIC][
            "mean_eightfold_threshold_count"
        ]
        for label in ("C9 vs Sporadic", "C9 vs Control")
    }
    spctrl_fixed_threshold_count = contrasts["Sporadic vs Control"]["summary"][
        PRIMARY_ANCHOR_SCOPE
    ][PRIMARY_DEGREE_METRIC]["mean_eightfold_threshold_count"]
    interpretation = (
        "In this same-data conditional sensitivity on the fixed/canonical "
        "scope, the mean `-slope` exceeds the full-INDRA degree-matched "
        "reference at the eightfold threshold for "
        f"{c9_fixed_threshold_counts['C9 vs Sporadic']}/8 C9-vs-sporadic "
        f"terms and {c9_fixed_threshold_counts['C9 vs Control']}/8 "
        "C9-vs-control terms; the corresponding sporadic-vs-control count is "
        f"{spctrl_fixed_threshold_count}/8. The 7/7/0 pattern weighs against a "
        "simple degree-location explanation for those seven terms, but does "
        "not rule it out, provide selective-inference control, or show that "
        "the GSEA result is network-independent."
    )

    primary_summary: dict[str, Any] = {}
    may_legacy_summary: dict[str, Any] = {}
    for label, block in contrasts.items():
        primary_rows = [
            row
            for row in block["results"]
            if row["anchor_scope"] == PRIMARY_ANCHOR_SCOPE
            and row["degree_metric"] == PRIMARY_DEGREE_METRIC
        ]
        primary_summary[label] = {
            "threshold_count": sum(
                row["mean_score"]["meets_eightfold_threshold"] for row in primary_rows
            ),
            "terms_meeting_threshold": [
                row["term"]
                for row in primary_rows
                if row["mean_score"]["meets_eightfold_threshold"]
            ],
            "terms_not_meeting_threshold": [
                row["term"]
                for row in primary_rows
                if not row["mean_score"]["meets_eightfold_threshold"]
            ],
        }
        legacy_rows = [
            row
            for row in block["results"]
            if row["anchor_scope"] == LEGACY_ANCHOR_SCOPE
            and row["degree_metric"] == PRIMARY_DEGREE_METRIC
        ]
        may_legacy_summary[label] = {
            "threshold_count": sum(
                row["may_legacy_single_draw_mwu"]["meets_eightfold_threshold"]
                for row in legacy_rows
            ),
            "terms_meeting_threshold": [
                row["term"]
                for row in legacy_rows
                if row["may_legacy_single_draw_mwu"]["meets_eightfold_threshold"]
            ],
        }

    return {
        "schema_version": 1,
        "analysis": {
            "name": "canonical_c9_f5a_degree_matched_anchor_null",
            "score": "-slope",
            "input_scale": "log2(x+1) Sex-adjusted EB-moderated |t|",
            "path_traversal": "measured_only",
            "max_hops": 2,
            "n_valid_anchors": 3117,
            "fixed_canonical_anchor_scope": PRIMARY_ANCHOR_SCOPE,
            "fixed_canonical_anchor_scope_rule": "hop1_size >= 20",
            "legacy_anchor_scope": LEGACY_ANCHOR_SCOPE,
            "fixed_canonical_degree_metric": PRIMARY_DEGREE_METRIC,
            "sensitivity_degree_metric": SENSITIVITY_DEGREE_METRIC,
            "degree_ratio_window": [DEGREE_RATIO_LOWER, DEGREE_RATIO_UPPER],
            "control_sampling": "one match per member with replacement per replicate",
            "historical_method_sensitivity": (
                "May all-valid/full-degree single matched draw, deduplicated "
                "controls, one-sided Mann-Whitney U"
            ),
            "n_replicates": n_replicates,
            "base_seed": base_seed,
            "fixed_canonical_statistic": "mean_score",
            "sensitivity_statistic": "median_score",
            "alternative": "term score greater than degree-matched reference",
            "term_selection_status": (
                "discovery-derived on this cohort, then fixed before "
                "measured-only/log2 method transfer"
            ),
            "n_fixed_terms": N_TERMS,
            "nominal_eightfold_reference_alpha": NOMINAL_REFERENCE_ALPHA,
            "eightfold_raw_p_threshold": EIGHTFOLD_RAW_P_THRESHOLD,
            "post_selection_fwer_guarantee": False,
            "selective_inference_correction": False,
            "explicit_non_inputs": [
                "per-anchor slope_pvalue",
                "GSEA NES",
                "GSEA nominal p-value",
                "GSEA FDR q-value",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
        },
        "inputs": {
            "analysis_script_path": _portable_path(Path(__file__)),
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
            "degree_meta_path": _portable_path(degree_meta_path),
            "degree_meta_sha256": expected_meta_sha,
            "term_snapshot_path": _portable_path(term_snapshot_path),
            "term_snapshot_sha256": _sha256(term_snapshot_path),
            "term_snapshot_content_sha256": hashlib.sha256(
                _canonical_json_bytes(snapshot)
            ).hexdigest(),
        },
        "term_order": [term_id for _, term_id, _, _ in TERMS],
        "summary": {
            "fixed_canonical_scope_full_degree_mean": primary_summary,
            "may_legacy_all_valid_single_draw_mwu": may_legacy_summary,
        },
        "contrasts": contrasts,
        "interpretation": interpretation,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-meta", type=Path, default=DEFAULT_DEGREE_META)
    parser.add_argument("--term-snapshot", type=Path, default=DEFAULT_TERM_SNAPSHOT)
    parser.add_argument(
        "--refresh-terms",
        action="store_true",
        help="Fetch CoGEx GO/Reactome terms and replace --term-snapshot.",
    )
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    result_destinations = {
        "C9 vs Sporadic": "c9_vs_sporadic",
        "C9 vs Control": "c9_vs_control",
        "Sporadic vs Control": "sporadic_vs_control",
    }
    for label, default in DEFAULT_RESULTS.items():
        option = "--" + label.lower().replace(" ", "-")
        parser.add_argument(
            option,
            dest=result_destinations[label],
            type=Path,
            default=default,
        )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_replicates < 999:
        raise ValueError(
            "Use at least 999 replicates; fewer cannot resolve the fixed-list "
            "eightfold threshold reliably."
        )
    if args.refresh_terms:
        refresh_term_snapshot(args.term_snapshot)
    if not args.term_snapshot.exists():
        raise FileNotFoundError(f"Missing {args.term_snapshot}; run once with --refresh-terms")

    result_paths = {
        "C9 vs Sporadic": args.c9_vs_sporadic,
        "C9 vs Control": args.c9_vs_control,
        "Sporadic vs Control": args.sporadic_vs_control,
    }
    writes_canonical = (
        args.output_json.expanduser().resolve() == DEFAULT_OUTPUT_JSON.resolve()
        or args.output_md.expanduser().resolve() == DEFAULT_OUTPUT_MD.resolve()
    )
    canonical_configuration = (
        args.n_replicates == N_REPLICATES
        and args.seed == BASE_SEED
        and args.degree_meta.expanduser().resolve() == DEFAULT_DEGREE_META.resolve()
        and args.term_snapshot.expanduser().resolve() == DEFAULT_TERM_SNAPSHOT.resolve()
        and all(
            result_paths[label].expanduser().resolve() == DEFAULT_RESULTS[label].resolve()
            for label in DEFAULT_RESULTS
        )
    )
    if writes_canonical and not canonical_configuration:
        raise ValueError(
            "Noncanonical inputs, seed, or replicate count require explicit alternate "
            "--output-json and --output-md paths; refusing to overwrite publication artifacts."
        )
    artifact = build_artifact(
        result_paths,
        args.degree_meta,
        args.term_snapshot,
        args.n_replicates,
        args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(artifact), encoding="utf-8")
    print(artifact["interpretation"])
    print(f"Wrote {_portable_path(args.output_json)}")
    print(f"Wrote {_portable_path(args.output_md)}")


if __name__ == "__main__":
    main()
