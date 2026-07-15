"""Guards for the canonical C9 publication state.

The analysis outputs are intentionally too large for Git.  This module keeps
the small, tracked publication snapshot internally consistent and, when the
local outputs are available, verifies that the snapshot still matches them.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "publication" / "c9_primary_analysis.json"

CONTRAST_PATHS = {
    "C9 vs Sporadic": "c9spor",
    "C9 vs Control": "c9ctrl",
    "Sporadic vs Control": "spctrl",
}
CONTRAST_GROUPS = {
    "C9 vs Sporadic": ["C9ORF72", "SPORADIC"],
    "C9 vs Control": ["C9ORF72", "CONTROL"],
    "Sporadic vs Control": ["SPORADIC", "CONTROL"],
}
LANDSCAPE_PATHS = {
    "C9 vs Sporadic": "landscape_proteome_measured_only_log2",
    "C9 vs Control": "landscape_c9_vs_control_measured_only_log2",
    "Sporadic vs Control": "landscape_sporadic_vs_control_measured_only_log2",
}
EXPECTED_ROBUST_GSEA_COUNTS = {
    "C9 vs Sporadic": {"go": 161, "reactome": 114, "wikipathways": 9, "phenotype": 0},
    "C9 vs Control": {"go": 157, "reactome": 64, "wikipathways": 13, "phenotype": 26},
    "Sporadic vs Control": {"go": 0, "reactome": 0, "wikipathways": 0, "phenotype": 0},
}


def _state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _resolve_primary_groups(metadata):
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


def test_canonical_publication_state_is_internally_consistent():
    state = _state()
    accounting = state["feature_accounting"]

    assert state["status"] == "canonical_primary"
    assert state["cohort"] == {
        "measured_samples": 436,
        "metadata_matched_samples": 423,
        "primary_arm_samples": 410,
        "arms": {"C9ORF72": 25, "SPORADIC": 294, "CONTROL": 91},
    }
    assert state["design"] == {
        "intensity_transform": "log2(x+1)",
        "covariates": ["Sex"],
        "path_traversal": "measured_only",
        "distance_direction": "undirected",
        "max_hops": 2,
        "landscape_permutations": 999,
        "gsea_permutations": 1000,
        "confirmatory_scope": "robust",
        "confirmatory_scope_definition": "hop-1 measured neighborhood size >= 20",
    }
    assert accounting["attempted"] == (
        accounting["valid_two_shell_gradients"]
        + accounting["no_reachable_measured_neighbor"]
        + accounting["below_minimum_measurable_neighborhood"]
    )
    assert (
        accounting["attempted"],
        accounting["valid_two_shell_gradients"],
        accounting["no_reachable_measured_neighbor"],
        accounting["below_minimum_measurable_neighborhood"],
    ) == (3264, 3117, 137, 10)
    assert accounting["primary_confirmatory_anchors"] == 1407
    assert accounting["primary_confirmatory_anchors"] < accounting[
        "valid_two_shell_gradients"
    ]
    assert accounting["attempted"] == (
        accounting["attempted_human_uniprot_features"]
        + accounting["attempted_internal_retention_time_standards"]
    )
    assert state["cohort"]["primary_arm_samples"] == sum(
        state["cohort"]["arms"].values()
    )
    assert state["empirical_bayes_prior_df"] == {
        "C9 vs Sporadic": 4.98469971792816,
        "C9 vs Control": 5.509674728521206,
        "Sporadic vs Control": 4.921881586146681,
    }

    confirmatory = state["confirmatory"]
    assert confirmatory["selection_status"] == (
        "same-cohort consistency check; not independent confirmation or "
        "prospective preregistration"
    )
    assert confirmatory["post_selection_fwer_guarantee"] is False
    assert "not selective inference" in confirmatory["threshold_interpretation"]
    assert [item["term"] for item in confirmatory["term_definitions"]] == (
        confirmatory["term_order"]
    )
    assert [item["term_id"] for item in confirmatory["term_definitions"]] == [
        "reactome:R-HSA-72172",
        "reactome:R-HSA-72203",
        "go:0000398",
        "go:0005694",
        "go:0000785",
        "go:0006913",
        "go:0005643",
        "reactome:R-HSA-180910",
    ]
    bounded_counts = [
        confirmatory["bounded"][contrast]["pass_count"]
        for contrast in confirmatory["contrast_order"]
    ]
    unbounded_counts = [
        confirmatory["unbounded"][contrast]["pass_count"]
        for contrast in confirmatory["contrast_order"]
    ]
    assert bounded_counts == confirmatory["bounded_pass_pattern"] == [8, 6, 0]
    assert unbounded_counts == confirmatory["unbounded_pass_pattern"] == [6, 0, 0]

    for contrast in confirmatory["contrast_order"]:
        term_results = confirmatory["bounded"][contrast]["terms"]
        assert list(term_results) == confirmatory["term_order"]
        assert all(
            term["pass"]
            is (
                term["raw_p"] < confirmatory["per_term_alpha"]
                and term["NES"] > 0
            )
            for term in term_results.values()
        )
        assert sum(term["pass"] for term in term_results.values()) == (
            confirmatory["bounded"][contrast]["pass_count"]
        )

    passing_sets = [
        {
            term
            for term, result in confirmatory["bounded"][contrast]["terms"].items()
            if result["pass"]
        }
        for contrast in confirmatory["contrast_order"]
    ]
    expected_core = (passing_sets[0] & passing_sets[1]) - passing_sets[2]
    assert set(confirmatory["six_term_core"]) == expected_core
    exploratory = confirmatory[
        "exploratory_full_database_robust_fdr_lt_0_05_rows"
    ]
    assert "not unique or independent pathways" in exploratory["unit"]
    assert [exploratory[contrast] for contrast in confirmatory["contrast_order"]] == [
        284,
        260,
        0,
    ]

    unbounded = state["unbounded_feature_accounting"]
    assert unbounded["attempted"] == (
        unbounded["valid_gradients"]
        + unbounded["no_reachable_measured_neighbor"]
        + unbounded["below_minimum_measurable_neighborhood"]
    )
    assert (
        unbounded["attempted"],
        unbounded["valid_gradients"],
        unbounded["no_reachable_measured_neighbor"],
        unbounded["below_minimum_measurable_neighborhood"],
    ) == (3264, 3125, 137, 2)
    assert state["source_artifacts"]["canonical_auxiliaries"] == {
        "size_matched_result": state["auxiliary_evidence"]
        ["size_matched_gene_set_null"]["result"],
        "degree_matched_result": state["auxiliary_evidence"]
        ["pathway_level_degree_stratified_null"]["result"],
    }
    assert state["source_artifacts"]["gsea_provenance"] == (
        "data/publication/c9_gsea_provenance.json"
    )


@pytest.mark.parametrize("contrast", tuple(CONTRAST_PATHS))
def test_local_bounded_landscape_matches_publication_snapshot(contrast: str):
    """Validate the ignored full result when it exists in a research checkout."""
    state = _state()
    relative_path = state["source_artifacts"]["bounded_landscapes"][contrast]
    assert relative_path == f"output/{LANDSCAPE_PATHS[contrast]}/result.json"
    result_path = ROOT / relative_path
    if not result_path.exists():
        pytest.skip(f"local research artifact is absent: {result_path}")

    expected = state["feature_accounting"]
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["design"]["transform"] == "log2"
    assert result["design"]["contrast"] == CONTRAST_GROUPS[contrast]
    assert result["design"]["max_hops"] == 2
    assert result["design"]["n_permutations"] == 999
    assert result["design"]["covariates"] == ["Sex"]
    assert result["n_features_input"] == expected["attempted"]
    assert len(result["per_feature"]) == expected["valid_two_shell_gradients"]
    assert len(result["degenerate_features"]) == expected["no_reachable_measured_neighbor"]
    assert len(result["error_features"]) == expected["below_minimum_measurable_neighborhood"]
    assert {
        item["error_type"] for item in result["degenerate_features"]
    } == {"DisconnectedFeature"}
    assert all(
        item["error_message"] == "no measured features reachable within max_hops"
        for item in result["degenerate_features"]
    )
    assert all(
        item["error_type"] == "ValueError"
        and re.fullmatch(
            r"Only [1-9] measurable genes in graph neighborhood\. Need at least 10\.",
            item["error_message"],
        )
        for item in result["error_features"]
    )
    assert sum(
        bool(item.get("shells")) and item["shells"][0]["n_genes"] >= 20
        for item in result["per_feature"]
    ) == expected["primary_confirmatory_anchors"]
    assert all(
        [shell["hop"] for shell in item["shells"]] == [1, 2]
        for item in result["per_feature"]
    )
    assert expected["internal_retention_time_standard_id"] in {
        item["seed"] for item in result["degenerate_features"]
    }

    meta_path = result_path.with_name("distances.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["path_traversal"] == "measured_only"
    assert meta["max_hops"] == 2
    assert len(meta["feature_names"]) == expected["attempted"]
    assert expected["internal_retention_time_standard_id"] in meta["feature_names"]


@pytest.mark.parametrize("contrast", tuple(CONTRAST_PATHS))
def test_local_unbounded_landscape_matches_publication_snapshot(contrast: str):
    state = _state()
    relative_path = state["source_artifacts"]["unbounded_landscapes"][contrast]
    result_path = ROOT / relative_path
    if not result_path.exists():
        pytest.skip(f"local research artifact is absent: {result_path}")

    expected = state["unbounded_feature_accounting"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["design"]["transform"] == "log2"
    assert result["design"]["contrast"] == CONTRAST_GROUPS[contrast]
    assert result["design"]["max_hops"] is None
    assert result["design"]["n_permutations"] == 999
    assert result["design"]["covariates"] == ["Sex"]
    assert result["n_features_input"] == expected["attempted"]
    assert len(result["per_feature"]) == expected["valid_gradients"]
    assert len(result["degenerate_features"]) == expected[
        "no_reachable_measured_neighbor"
    ]
    assert len(result["error_features"]) == expected[
        "below_minimum_measurable_neighborhood"
    ]
    assert sum(
        bool(item.get("shells")) and item["shells"][0]["n_genes"] >= 20
        for item in result["per_feature"]
    ) == expected["primary_confirmatory_anchors"]

    meta = json.loads(
        result_path.with_name("distances.meta.json").read_text(encoding="utf-8")
    )
    assert meta["path_traversal"] == "measured_only"
    assert meta["max_hops"] is None


def test_auxiliary_evidence_ledger_matches_canonical_and_withdrawn_artifacts():
    state = _state()
    auxiliary = state["auxiliary_evidence"]

    size_null = auxiliary["size_matched_gene_set_null"]
    degree_null = auxiliary["pathway_level_degree_stratified_null"]
    assert size_null["status"] == "canonical_log2_auxiliary"
    assert size_null["eightfold_threshold_pattern"] == [8, 8, 0]
    assert degree_null["status"] == "canonical_log2_auxiliary"
    assert degree_null["eightfold_threshold_pattern"] == [7, 7, 0]

    size_result = json.loads((ROOT / size_null["result"]).read_text(encoding="utf-8"))
    assert [
        size_result["pass_counts_bonferroni_8"][contrast]
        for contrast in size_result["contrast_order"]
    ] == [8, 8, 0]
    degree_result = json.loads(
        (ROOT / degree_null["result"]).read_text(encoding="utf-8")
    )
    assert [
        degree_result["contrasts"][contrast]["summary"]["robust_hop1_ge_20"]
        ["full_indra_degree"]["mean_eightfold_threshold_count"]
        for contrast in CONTRAST_GROUPS
    ] == [7, 7, 0]

    assert auxiliary["string_alternative_network"]["publication_use"] == "none"
    assert auxiliary["matched_rna"]["publication_use"] == "none"
    assert "post-transcriptional" not in " ".join(state["claim_scope"]["licensed"])
    assert "C9-specific" not in " ".join(state["claim_scope"]["licensed"])
    assert all(
        auxiliary[key]["status"].startswith(("legacy_", "withdrawn_"))
        for key in (
            "abundance_stratified_null",
            "string_alternative_network",
            "matched_rna",
            "age_partial_r2",
            "five_analysis_sensitivities",
        )
    )


def test_local_log2_empirical_bayes_priors_match_publication_snapshot():
    """Recompute d0 from the local canonical inputs when research data exist."""
    import numpy as np

    from cliquefinder.panels.seed_runner import load_panel_inputs
    from cliquefinder.stats.rotation import RotationTestEngine
    data_path = ROOT / "output" / "proteomics" / "all_als.data.csv"
    metadata_path = ROOT / "output" / "proteomics" / "all_als.metadata.csv"
    if not data_path.exists() or not metadata_path.exists():
        pytest.skip("local canonical proteomics inputs are absent")

    data, feature_ids, metadata, groups = load_panel_inputs(
        data_path=data_path,
        metadata_path=metadata_path,
        group_resolver=_resolve_primary_groups,
    )
    sample_index = {sample: i for i, sample in enumerate(metadata.index)}
    expected = _state()["empirical_bayes_prior_df"]

    for label, (condition_1, condition_2) in zip(
        CONTRAST_GROUPS,
        CONTRAST_GROUPS.values(),
        strict=True,
    ):
        keep = groups[condition_1].union(groups[condition_2])
        sample_metadata = metadata.loc[metadata.index.intersection(keep)].copy()
        sample_metadata["_condition"] = None
        sample_metadata.loc[
            sample_metadata.index.isin(groups[condition_1]), "_condition"
        ] = condition_1
        sample_metadata.loc[
            sample_metadata.index.isin(groups[condition_2]), "_condition"
        ] = condition_2
        sample_metadata = sample_metadata.dropna(subset=["_condition"])
        columns = [sample_index[sample] for sample in sample_metadata.index]
        log2_data = np.log2(data[:, columns] + 1.0)

        engine = RotationTestEngine(
            log2_data.copy(), list(feature_ids), sample_metadata.copy()
        ).fit(
            conditions=[condition_1, condition_2],
            contrast=(condition_1, condition_2),
            condition_column="_condition",
            covariates=["Sex"],
        )
        assert engine._precomputed is not None
        assert engine._precomputed.eb_d0 == pytest.approx(expected[label], abs=1e-12)


@pytest.mark.parametrize("contrast", tuple(CONTRAST_PATHS))
def test_local_confirmatory_table_matches_publication_snapshot(contrast: str):
    """Validate the ignored confirmatory CSV when it exists locally."""
    state = _state()
    code = CONTRAST_PATHS[contrast]
    expected_relative = (
        f"output/landscape_confirmatory_{code}_measured_only_log2/"
        "confirmatory_8terms_robust.csv"
    )
    relative_path = state["source_artifacts"]["bounded_confirmatory"][contrast]
    assert relative_path == expected_relative
    table_path = ROOT / relative_path
    if not table_path.exists():
        pytest.skip(f"local research artifact is absent: {table_path}")

    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    observed = {row["term"]: row for row in rows}
    short_to_full = {
        "Processing Capped Pre-mRNA": "Processing of Capped Intron-Containing Pre-mRNA",
        "Vpr-mediated nuclear import": "Vpr-mediated nuclear import of PICs",
    }
    expected_terms = state["confirmatory"]["bounded"][contrast]["terms"]
    assert len(rows) == len(expected_terms) == 8

    for short_label, expected in expected_terms.items():
        row = observed[short_to_full.get(short_label, short_label)]
        assert float(row["NES"]) == pytest.approx(expected["NES"])
        assert float(row["raw_p"]) == pytest.approx(expected["raw_p"])
        assert (row["bonferroni_pass"] == "True") is expected["pass"]


@pytest.mark.parametrize("contrast", tuple(CONTRAST_PATHS))
def test_local_unbounded_confirmatory_pass_pattern(contrast: str):
    state = _state()
    relative_path = state["source_artifacts"]["unbounded_confirmatory"][contrast]
    table_path = ROOT / relative_path
    if not table_path.exists():
        pytest.skip(f"local research artifact is absent: {table_path}")

    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    observed_passes = sum(row["bonferroni_pass"] == "True" for row in rows)
    assert observed_passes == state["confirmatory"]["unbounded"][contrast][
        "pass_count"
    ]
    assert all(
        (row["bonferroni_pass"] == "True")
        is (
            float(row["raw_p"]) < state["confirmatory"]["per_term_alpha"]
            and float(row["NES"]) > 0
        )
        for row in rows
    )


@pytest.mark.parametrize("contrast", tuple(CONTRAST_PATHS))
def test_local_robust_gsea_summary_matches_database_row_counts(contrast: str):
    state = _state()
    relative_path = state["source_artifacts"]["bounded_gsea_summaries"][contrast]
    summary_path = ROOT / relative_path
    if not summary_path.exists():
        pytest.skip(f"local research artifact is absent: {summary_path}")

    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scope"] == "robust"]
    observed = {row["db"]: int(row["n_fdr_lt_05"]) for row in rows}
    assert observed == EXPECTED_ROBUST_GSEA_COUNTS[contrast]

    expected_total = state["confirmatory"][
        "exploratory_full_database_robust_fdr_lt_0_05_rows"
    ][contrast]
    assert sum(observed.values()) == expected_total


def test_current_publication_narratives_do_not_reintroduce_superseded_primary():
    """Guard the canonical design, selection boundary, and auxiliary ledger."""
    narrative_paths = [
        ROOT / "output" / "analytical_workflow_methods.md",
        ROOT / "output" / "analytical_workflow_breakdown.md",
        ROOT / "output" / "c9_triangulation_report.md",
    ]
    forbidden = (
        "d₀≈0.60",
        "d0≈0.60",
        "on raw-linear values",
        "3,257",
        "130 single-shell",
        "Land 7/8 and 6/8 first",
        "56.5% FPR",
        "FPR 56.5%",
        "run anticonservative",
        "pre-registered confirmatory",
        "graph-invariant core",
        "STRING gives the opposite",
        "supports a protein-level, post-transcriptional",
        "within-ALS cluster claim is robust to age adjustment",
        "3,117 valid slopes become one preranked list",
        "size-matched gene-set null is being rerun",
        "degree-stratified check must be rerun",
    )

    for path in narrative_paths:
        text = path.read_text(encoding="utf-8")
        for stale in forbidden:
            assert stale not in text, f"{path.relative_to(ROOT)} retains {stale!r}"
        assert "1,407" in text
        assert "3,264" in text
        assert "3,117" in text
        assert "137" in text
        assert "8/6/0" in text
        assert "6/0/0" in text
        assert "8/8/0" in text
        assert "7/7/0" in text
        assert "iRT" in text
        assert "withdraw" in text.lower()
        assert "same-cohort" in text.lower() or "discovery-derived" in text.lower()
        assert re.search(r"log.?2", text, flags=re.IGNORECASE)
        assert re.search(r"4\.98", text)


def test_workflow_docs_retain_audited_method_boundaries():
    """Guard implementation-level corrections that headline token tests miss."""
    workflow_paths = [
        ROOT / "output" / "analytical_workflow_methods.md",
        ROOT / "output" / "analytical_workflow_breakdown.md",
    ]
    forbidden = (
        "one dominant confound",
        "they tend to carry larger $|t|$",
        "a hub's value can only be exchanged",
        "threshold across the reported rerun tests",
        "covariate-residualized anchor",
        "does not evaluate or name individual proteins",
        "pre-specified guardrail",
        "mean squared moderated-$t$",
        "puts sets of every size on a common scale",
        "same-size sets from the measured universe",
        "compared the set's mean $|t|$ with the complement's mean $|t|$",
        "planned maximum $B=9{,}999$",
        "raw $p\\le0.10$",
        "raw-$p\\le0.10$",
        "raw $p≤0.10$",
        "raw-$p≤0.10$",
        "distance files and hashes preserve",
    )
    required = (
        "rank blocks of 100",
        "separately within each contrast",
        "1,000 gene-set permutations",
        "904",
        "manually maintained",
        "measured non-target pool",
        "random-set test",
        "B=99{,}999",
        "metadata JSON, not the NPZ",
        "current reference reconstruction",
        "historical target-set branch used the provider-corrected/imputed linear-intensity matrix without a repository log transform",
        "target rotation used evidence weights",
        "random control sets were unweighted",
        "random-set calls left their seed at the `None` default",
        "embedded neither $d_0$ nor $s_0^2$",
    )

    for path in workflow_paths:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        for stale in forbidden:
            assert stale not in normalized, (
                f"{path.relative_to(ROOT)} retains {stale!r}"
            )
        for boundary in required:
            assert boundary in normalized, (
                f"{path.relative_to(ROOT)} omits {boundary!r}"
            )
        assert re.search(r"raw-?\s*\$p<0\.10\$", normalized), (
            f"{path.relative_to(ROOT)} omits the strict WASC selector"
        )
        assert re.search(
            r"did not verify[^.]{0,160}graph non-neighbor", normalized
        ), f"{path.relative_to(ROOT)} reverses the WASC sampler caveat"
        assert "version-controlled together" in normalized, (
            f"{path.relative_to(ROOT)} omits the Git-history boundary"
        )
        assert "do not yet have committed-git provenance" not in normalized
        assert re.search(
            r"sizes and SHA-256 values declared in the WASC manifest.{0,240}?exclude.{0,40}?final LF",
            normalized,
        ), f"{path.relative_to(ROOT)} omits the WASC final-LF hash boundary"
        assert "upstream" in normalized and "regeneration" in normalized, (
            f"{path.relative_to(ROOT)} omits the upstream-regeneration boundary"
        )


def test_string_figure_is_a_withdrawal_receipt_not_a_numeric_comparison():
    """Prevent the invalid legacy STRING matrix from returning to the report."""
    script = ROOT / "scripts" / "viz" / "report_figures.py"
    spec = importlib.util.spec_from_file_location("report_figures", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    figure = module.build_fig6_string()
    assert set(figure) == {"html", "css", "js"}
    assert "WITHDRAWN LEGACY COMPARATOR" in figure["html"]
    assert "licenses no" in figure["html"]
    assert "fig6-string" not in figure["html"]
    assert figure["js"] == ""
