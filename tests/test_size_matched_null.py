"""Focused guards for the canonical log2 EB size-matched HGNC null."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / "data/publication/c9_size_matched_null_inputs.json"
RESULT_PATH = ROOT / "output/size_matched_null.json"

_SPEC = importlib.util.spec_from_file_location(
    "canonical_size_matched_null", ROOT / "scripts/run_size_matched_null.py"
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def _frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def _result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_frozen_hgnc_inputs_are_closed_and_hash_pinned():
    frozen = _frozen()
    mapping = frozen["feature_hgnc_mapping"]

    assert frozen["analysis_id"] == "c9-size-matched-null-log2-eb-2026-07"
    assert mapping["feature_count"] == 3264
    assert mapping["mapped_feature_count"] == len(mapping["feature_to_hgnc"]) == 3262
    assert mapping["unique_hgnc_count"] == len(
        set(mapping["feature_to_hgnc"].values())
    ) == 3261
    assert mapping["unmapped_features"] == ["1/iRT_protein", "P30042"]
    assert mapping["duplicate_hgnc_to_features"] == {
        "11875": ["P42166", "P42167"]
    }
    assert mapping["aggregation"] == (
        "maximum t^2 across UniProt features for each HGNC ID"
    )



def test_frozen_source_hashes_match_when_local_inputs_exist():
    frozen = _frozen()
    sources = list(frozen["source_files"].values())
    missing = [
        ROOT / source["path"]
        for source in sources
        if not (ROOT / source["path"]).exists()
    ]
    if missing:
        pytest.skip("local canonical proteomics inputs are absent")

    for source in sources:
        source_path = ROOT / source["path"]
        assert analysis.sha256_file(source_path) == source["sha256"]


def test_frozen_term_family_and_measured_memberships_are_exact():
    frozen = _frozen()
    expected_sizes = {
        "reactome:R-HSA-72172": (212, 137),
        "reactome:R-HSA-72203": (278, 183),
        "go:0000398": (201, 118),
        "go:0005694": (1074, 321),
        "go:0000785": (620, 174),
        "go:0006913": (167, 81),
        "go:0005643": (72, 40),
        "reactome:R-HSA-180910": (31, 26),
    }
    assert list(frozen["terms"]) == [term_id for _, term_id, _ in analysis.TERMS]

    measured_background = set(
        frozen["feature_hgnc_mapping"]["feature_to_hgnc"].values()
    )
    for term_id, (library_size, measured_size) in expected_sizes.items():
        term = frozen["terms"][term_id]
        library = set(term["library_hgnc_ids"])
        measured = set(term["measured_hgnc_ids"])
        assert term["library_hgnc_count"] == len(library) == library_size
        assert term["measured_hgnc_count"] == len(measured) == measured_size
        assert measured == library & measured_background


def test_canonical_result_contract_and_bonferroni_arithmetic():
    result = _result()
    frozen = _frozen()
    design = result["design"]

    assert result["status"] == "canonical_auxiliary"
    assert design["intensity_transform"] == "log2(x+1)"
    assert design["per_feature_model"] == "condition + Sex"
    assert design["per_feature_engine"].endswith("RotationTestEngine")
    assert design["analysis_unit"] == "HGNC gene"
    assert design["n_permutations"] == 10_000
    assert design["master_seed"] == 42
    assert design["bonferroni_alpha_per_term"] == 0.00625
    assert result["generator"] == {
        "path": "scripts/run_size_matched_null.py",
        "sha256": analysis.sha256_file(ROOT / "scripts/run_size_matched_null.py"),
    }
    assert result["frozen_inputs"]["sha256"] == analysis.sha256_file(FROZEN_PATH)
    assert result["pass_counts_bonferroni_8"] == {
        "C9 vs Sporadic": 8,
        "C9 vs Control": 8,
        "Sporadic vs Control": 0,
    }
    assert len(result["rows"]) == 24
    assert len({(row["contrast"], row["term_id"]) for row in result["rows"]}) == 24

    for row in result["rows"]:
        assert row["set_size"] == frozen["terms"][row["term_id"]][
            "measured_hgnc_count"
        ]
        assert row["background_size"] == 3261
        expected_p = (row["null_exceedances"] + 1) / 10_001
        assert row["empirical_p"] == pytest.approx(expected_p, abs=1e-15)
        assert row["bonferroni_p"] == pytest.approx(
            min(8 * expected_p, 1.0), abs=1e-15
        )
        assert row["bonferroni_pass"] is (expected_p < 0.00625)
        assert row["null_seed"] == analysis.stable_seed(
            42, row["contrast"], row["term_id"]
        )

    assert all(
        row["bonferroni_pass"]
        for row in result["rows"]
        if row["contrast"].startswith("C9")
    )
    assert not any(
        row["bonferroni_pass"]
        for row in result["rows"]
        if row["contrast"] == "Sporadic vs Control"
    )


def test_production_eb_diagnostics_match_the_primary_snapshot():
    result = _result()
    expected = {
        "C9 vs Sporadic": (319, 4.98469971792816),
        "C9 vs Control": (116, 5.509674728521206),
        "Sporadic vs Control": (385, 4.921881586146681),
    }
    for contrast, (sample_count, d0) in expected.items():
        diagnostic = result["contrast_diagnostics"][contrast]
        assert diagnostic["sample_count"] == sample_count
        assert diagnostic["engine_fitted_feature_count"] == 3264
        assert diagnostic["finite_feature_count"] == 3264
        assert diagnostic["mapped_finite_feature_count"] == 3262
        assert diagnostic["finite_unique_hgnc_count"] == 3261
        assert diagnostic["eb_d0"] == pytest.approx(d0, abs=1e-12)


def test_current_engine_reproduces_all_observed_pathway_statistics():
    """Refit all three contrasts; use a tiny null because only t² is checked."""
    if not analysis.DATA_PATH.exists() or not analysis.METADATA_PATH.exists():
        pytest.skip("local canonical proteomics inputs are absent")

    frozen = analysis.load_and_validate_frozen_inputs(FROZEN_PATH)
    fresh = analysis.run_analysis(frozen, n_permutations=2, master_seed=42)
    canonical = _result()
    fresh_rows = {
        (row["contrast"], row["term_id"]): row for row in fresh["rows"]
    }
    for row in canonical["rows"]:
        rerun = fresh_rows[(row["contrast"], row["term_id"])]
        assert rerun["observed_mean_t2"] == pytest.approx(
            row["observed_mean_t2"], abs=1e-12
        )
        assert rerun["set_size"] == row["set_size"]
        assert rerun["background_size"] == row["background_size"]


def test_null_sampler_is_deterministic_and_without_replacement():
    background = np.arange(1.0, 11.0)
    seed = analysis.stable_seed(42, "contrast", "term")
    first = analysis.sample_null_means(background, 4, 12, seed)
    second = analysis.sample_null_means(background, 4, 12, seed)
    other = analysis.sample_null_means(background, 4, 12, seed + 1)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(first >= np.mean(background[:4]))
    assert np.all(first <= np.mean(background[-4:]))


def test_machine_outputs_and_markdown_share_one_result(tmp_path: Path):
    result = _result()
    prefix = tmp_path / "size_matched_null"
    analysis.write_outputs(result, prefix)

    assert prefix.with_suffix(".json").read_bytes() == RESULT_PATH.read_bytes()
    assert prefix.with_suffix(".md").read_bytes() == (
        ROOT / "output/size_matched_null.md"
    ).read_bytes()
    with prefix.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 24
    assert [row["term_id"] for row in csv_rows] == [
        row["term_id"] for row in result["rows"]
    ]

    markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert "Bonferroni-8 pass pattern is **8/8/0**" in markdown
    assert "log2(intensity+1) ~ condition + Sex" in markdown
    assert "does not" in markdown and "sample-size imbalance" in markdown


def test_noncanonical_run_cannot_overwrite_publication_outputs(tmp_path: Path):
    with pytest.raises(ValueError, match="alternate --output-prefix"):
        analysis.main(["--n-permutations", "9"])
    with pytest.raises(ValueError, match="alternate --output-prefix"):
        analysis.main(["--frozen-inputs", str(tmp_path / "inputs.json")])


def test_markdown_and_provenance_support_sensitivity_paths(tmp_path: Path):
    result = _result()
    result["design"]["n_permutations"] = 9
    markdown = analysis.render_markdown(result)

    assert "9 uniform same-size HGNC sets" in markdown
    assert "/ 10`" in markdown
    external = tmp_path / "inputs.json"
    assert analysis.portable_path(external) == str(external.resolve())
