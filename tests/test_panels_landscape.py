"""Landscape module — pure-Python tests (no Neo4j, no INDRA).

Compute_landscape itself is tested via the integration smoke; these
tests cover the data model, FeatureDistanceMatrix sparse handling,
LandscapeResult validation, and analyze_landscape statistical
behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cliquefinder.panels import (
    AdjustedFeatureResult,
    FailedSeed,
    FeatureDistanceMatrix,
    LandscapeAnalysis,
    LandscapeDesign,
    LandscapeResult,
    PerSeedResult,
    ShellSummary,
    analyze_landscape,
)


# --- Helpers ----------------------------------------------------------------


def _design() -> LandscapeDesign:
    return LandscapeDesign(
        contrast=("CASE", "CTRL"),
        max_hops=2,
        n_permutations=999,
        covariates=("Sex",),
        description="test landscape",
    )


def _per_feature(seed: str, slope: float, p: float) -> PerSeedResult:
    from cliquefinder.panels.landscape import LANDSCAPE_FEATURE_STRATUM_LABEL
    return PerSeedResult(
        seed=seed,
        stratum=LANDSCAPE_FEATURE_STRATUM_LABEL,
        slope=slope, slope_pvalue=p,
        spearman_rho=0.0, spearman_pvalue=1.0,
        shells=(
            ShellSummary(hop=1, n_genes=10, mean_abs_t=1.0, median_abs_t=1.0),
        ),
        n_genes_total=10,
        elapsed_seconds=0.0,
    )


# --- LandscapeDesign --------------------------------------------------------


class TestLandscapeDesign:
    def test_validates_contrast(self):
        with pytest.raises(ValueError, match="distinct"):
            LandscapeDesign(
                contrast=("a", "a"), max_hops=2, n_permutations=999,
                covariates=(),
            )

    def test_validates_max_hops(self):
        with pytest.raises(ValueError, match="max_hops must be >= 1"):
            LandscapeDesign(
                contrast=("a", "b"), max_hops=0, n_permutations=999,
                covariates=(),
            )

    def test_validates_n_permutations(self):
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            LandscapeDesign(
                contrast=("a", "b"), max_hops=2, n_permutations=0,
                covariates=(),
            )

    def test_coerces_lists_to_tuples(self):
        d = LandscapeDesign(
            contrast=["a", "b"],  # type: ignore[arg-type]
            max_hops=2,
            n_permutations=999,
            covariates=["Sex", "Age"],  # type: ignore[arg-type]
        )
        assert isinstance(d.contrast, tuple)
        assert isinstance(d.covariates, tuple)

    def test_yaml_round_trip(self, tmp_path: Path):
        d = _design()
        path = tmp_path / "manifest.yaml"
        d.save_yaml(path)
        recovered = LandscapeDesign.load_yaml(path)
        assert recovered == d

    def test_dict_round_trip(self):
        d = _design()
        assert LandscapeDesign.from_dict(d.to_dict()) == d


# --- FeatureDistanceMatrix --------------------------------------------------


class TestFeatureDistanceMatrix:
    def test_from_distance_dict_builds_sparse(self):
        distances = {
            "A": {"A": 0, "B": 1, "C": 2},
            "B": {"A": 1, "B": 0, "C": 1},
            "C": {"A": 2, "B": 1, "C": 0},
        }
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B", "C"],
            max_hops=2,
        )
        assert matrix.feature_names == ("A", "B", "C")
        assert matrix.distances.shape == (3, 3)
        assert matrix.max_hops == 2

    def test_distances_from_recovers_full_row(self):
        distances = {"A": {"A": 0, "B": 1, "C": 2}}
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B", "C"],
            max_hops=2,
        )
        assert matrix.distances_from("A") == {"A": 0, "B": 1, "C": 2}

    def test_hop_neighbors_filters_to_exact_distance(self):
        distances = {"A": {"A": 0, "B": 1, "C": 1, "D": 2}}
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B", "C", "D"],
            max_hops=2,
        )
        assert matrix.hop_neighbors("A", 1) == {"B", "C"}
        assert matrix.hop_neighbors("A", 2) == {"D"}
        assert matrix.hop_neighbors("A", 0) == {"A"}

    def test_targets_outside_feature_names_dropped(self):
        """compute_all_pairs may report distances to non-measured nodes;
        the matrix construction must drop them.
        """
        distances = {"A": {"A": 0, "B": 1, "Unmeasured": 1}}
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B"],  # Unmeasured intentionally absent
            max_hops=2,
        )
        assert matrix.distances_from("A") == {"A": 0, "B": 1}

    def test_unknown_feature_raises_keyerror(self):
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances={"A": {"A": 0}},
            feature_names=["A"], max_hops=2,
        )
        with pytest.raises(KeyError, match="not in matrix"):
            matrix.index_of("ZZZ")
        with pytest.raises(KeyError):
            matrix.distances_from("ZZZ")

    def test_npz_round_trip(self, tmp_path: Path):
        distances = {
            "A": {"A": 0, "B": 1, "C": 2},
            "B": {"A": 1, "B": 0, "C": 1},
        }
        original = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B", "C"],
            max_hops=2,
            unmatched={"C"},
        )
        path = tmp_path / "distances.npz"
        original.save_npz(path)
        recovered = FeatureDistanceMatrix.load_npz(path)
        assert recovered.feature_names == original.feature_names
        assert recovered.max_hops == original.max_hops
        assert recovered.unmatched == original.unmatched
        assert recovered.distances_from("A") == original.distances_from("A")

    def test_shape_validation(self):
        # 2x2 matrix but 3 feature names → invalid.
        bad = sp.csr_matrix(np.zeros((2, 2), dtype=np.int16))
        with pytest.raises(ValueError, match="shape"):
            FeatureDistanceMatrix(
                feature_names=("A", "B", "C"),
                distances=bad,
                max_hops=2,
            )

    def test_unmatched_distinguished_from_isolated(self):
        """A feature in unmatched should NOT have its own row populated."""
        distances = {"A": {"A": 0, "B": 1}, "B": {"B": 0, "A": 1}}
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B", "ZZZ"],  # ZZZ in matrix but unmatched
            max_hops=2,
            unmatched={"ZZZ"},
        )
        assert "ZZZ" in matrix.unmatched
        # ZZZ has no row in the dict, so distances_from returns just empty
        assert matrix.distances_from("ZZZ") == {}


# --- LandscapeResult --------------------------------------------------------


class TestLandscapeResult:
    def _result(
        self,
        completed: list[tuple[str, float, float]] | None = None,
        degenerate: list[str] | None = None,
        errored: list[tuple[str, str, str]] | None = None,
    ) -> LandscapeResult:
        completed = completed or [
            ("A", -0.10, 0.05),
            ("B", -0.05, 0.30),
            ("C", +0.02, 0.70),
        ]
        degenerate = degenerate or []
        errored = errored or []
        per_feature = tuple(
            _per_feature(seed, slope, p)
            for seed, slope, p in completed
        )
        return LandscapeResult(
            design=_design(),
            per_feature=per_feature,
            degenerate_features=tuple(
                FailedSeed(s, "DisconnectedFeature", "no neighbors")
                for s in degenerate
            ),
            error_features=tuple(
                FailedSeed(s, etype, msg) for s, etype, msg in errored
            ),
            distance_matrix_path="distances.npz",
            n_features_input=len(completed) + len(degenerate) + len(errored),
        )

    def test_validates_partition(self):
        with pytest.raises(ValueError, match="!= n_features_input"):
            LandscapeResult(
                design=_design(),
                per_feature=(_per_feature("A", -0.1, 0.05),),
                degenerate_features=(),
                error_features=(),
                distance_matrix_path="x.npz",
                n_features_input=5,  # mismatched
            )

    def test_rejects_seed_in_completed_and_degenerate(self):
        with pytest.raises(ValueError, match="multiple outcome buckets"):
            LandscapeResult(
                design=_design(),
                per_feature=(_per_feature("A", -0.1, 0.05),),
                degenerate_features=(FailedSeed("A", "DisconnectedFeature", "x"),),
                error_features=(),
                distance_matrix_path="x.npz",
                n_features_input=2,
            )

    def test_rejects_seed_in_completed_and_errored(self):
        with pytest.raises(ValueError, match="multiple outcome buckets"):
            LandscapeResult(
                design=_design(),
                per_feature=(_per_feature("A", -0.1, 0.05),),
                degenerate_features=(),
                error_features=(FailedSeed("A", "Err", "y"),),
                distance_matrix_path="x.npz",
                n_features_input=2,
            )

    def test_rejects_duplicates_in_per_feature(self):
        with pytest.raises(ValueError, match="duplicate"):
            LandscapeResult(
                design=_design(),
                per_feature=(
                    _per_feature("A", -0.1, 0.05),
                    _per_feature("A", -0.2, 0.10),
                ),
                degenerate_features=(),
                error_features=(),
                distance_matrix_path="x.npz",
                n_features_input=2,
            )

    def test_failed_features_property_combines(self):
        """Backwards-compat property: failed_features = degenerate + errored."""
        result = self._result(
            completed=[("A", -0.1, 0.05)],
            degenerate=["B"],
            errored=[("C", "TimeoutError", "slow")],
        )
        assert {f.seed for f in result.failed_features} == {"B", "C"}

    def test_json_round_trip(self, tmp_path: Path):
        result = self._result(
            completed=[("A", -0.1, 0.05), ("B", 0.05, 0.5)],
            degenerate=["C"],
            errored=[("D", "RuntimeError", "boom")],
        )
        path = tmp_path / "result.json"
        result.save_json(path)
        recovered = LandscapeResult.load_json(path)
        assert recovered == result


# --- analyze_landscape ------------------------------------------------------


def _result_from_specs(
    completed: list[tuple[str, float, float]],
    *,
    degenerate: list[str] | None = None,
    errored: list[tuple[str, str, str]] | None = None,
) -> LandscapeResult:
    degenerate = degenerate or []
    errored = errored or []
    return LandscapeResult(
        design=_design(),
        per_feature=tuple(
            _per_feature(s, slope, p) for s, slope, p in completed
        ),
        degenerate_features=tuple(
            FailedSeed(s, "DisconnectedFeature", "x") for s in degenerate
        ),
        error_features=tuple(
            FailedSeed(s, etype, msg) for s, etype, msg in errored
        ),
        distance_matrix_path="d.npz",
        n_features_input=len(completed) + len(degenerate) + len(errored),
    )


class TestAnalyzeLandscape:
    def test_attaches_bh_q_per_feature(self):
        result = _result_from_specs([
            ("A", -0.20, 0.001),
            ("B", -0.10, 0.01),
            ("C", -0.05, 0.10),
            ("D", +0.05, 0.50),
        ])
        analysis = analyze_landscape(result)
        for adj in analysis.feature_results_adjusted:
            assert 0.0 <= adj.bh_qvalue <= 1.0
            assert adj.bh_qvalue >= adj.slope_pvalue

    def test_degenerate_and_errored_inflate_correction_family(self):
        completed_specs = [
            ("A", -0.20, 0.001),
            ("B", -0.10, 0.01),
        ]
        result = _result_from_specs(
            completed=completed_specs,
            degenerate=["DEG1"],
            errored=[("ERR1", "Timeout", "slow")],
        )
        analysis = analyze_landscape(result)
        # Family = 4 (2 completed + 1 degenerate + 1 errored)
        for adj in analysis.feature_results_adjusted:
            raw_p = next(p for s, _, p in completed_specs if s == adj.seed)
            expected = min(raw_p * 4, 1.0)
            assert adj.bonferroni_pvalue == pytest.approx(expected)

    def test_rank_matches_output_position(self):
        """Per-feature rank == position+1 in feature_results_adjusted."""
        result = _result_from_specs([
            ("A", -0.20, 0.05),
            ("B", -0.10, 0.05),
            ("C", +0.05, 0.05),
            ("D", +0.10, 0.05),
        ])
        analysis = analyze_landscape(result)
        for position, adj in enumerate(analysis.feature_results_adjusted):
            assert adj.rank_left_tail == position + 1

    def test_rank_breaks_ties_by_seed_name(self):
        """Two equal slopes get adjacent ranks, ordered by seed name."""
        result = _result_from_specs([
            ("Z", -0.10, 0.05),
            ("A", -0.10, 0.05),  # same slope, earlier alphabetical
        ])
        analysis = analyze_landscape(result)
        # A < Z lexicographically → A rank 1, Z rank 2
        assert analysis.feature_results_adjusted[0].seed == "A"
        assert analysis.feature_results_adjusted[0].rank_left_tail == 1
        assert analysis.feature_results_adjusted[1].seed == "Z"
        assert analysis.feature_results_adjusted[1].rank_left_tail == 2

    def test_discovery_flag_at_q_threshold(self):
        result = _result_from_specs([
            ("A", -0.20, 0.001),
            ("B", -0.10, 0.04),
            ("C", +0.05, 0.50),
        ])
        analysis = analyze_landscape(result, q_threshold=0.05)
        for adj in analysis.feature_results_adjusted:
            assert adj.discovery == (adj.bh_qvalue < 0.05)

    def test_empty_completed_returns_empty_analysis(self):
        result = LandscapeResult(
            design=_design(),
            per_feature=(),
            degenerate_features=(FailedSeed("A", "DisconnectedFeature", "x"),),
            error_features=(),
            distance_matrix_path="d.npz",
            n_features_input=1,
        )
        analysis = analyze_landscape(result)
        assert analysis.feature_results_adjusted == ()
        assert analysis.n_completed == 0
        assert analysis.n_failed == 1

    def test_nan_pvalue_does_not_become_discovery(self):
        result = _result_from_specs([
            ("A", -0.20, 0.001),
            ("B", -0.10, float("nan")),
        ])
        analysis = analyze_landscape(result)
        b_adj = next(a for a in analysis.feature_results_adjusted if a.seed == "B")
        assert b_adj.discovery is False


class TestStratumLabelReservation:
    def test_landscape_feature_label_rejected_as_panel_stratum(self):
        """`<feature>` is reserved like `<target>` — cannot be a user
        PanelStratum name.
        """
        from cliquefinder.panels import PanelStratum
        with pytest.raises(ValueError, match="reserved"):
            PanelStratum(name="<feature>", members=("A",))


class TestCheckpointParse:
    """JSONL checkpoint round-trip + recovery semantics."""

    def test_empty_path_returns_empty_collections(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        completed, degenerate, errored, seen = _load_checkpoint(ck)
        assert completed == []
        assert degenerate == []
        assert errored == []
        assert seen == set()

    def test_round_trip_via_writer(self, tmp_path: Path):
        from cliquefinder.panels.landscape import (
            _checkpoint_writer,
            _load_checkpoint,
        )
        ck = tmp_path / "checkpoint.jsonl"
        writer = _checkpoint_writer(ck)
        a = _per_feature("A", -0.1, 0.05)
        b = FailedSeed("B", "DisconnectedFeature", "no neighbors")
        c = FailedSeed("C", "RuntimeError", "boom")
        writer("completed", a.to_dict())
        writer("degenerate", b.to_dict())
        writer("errored", c.to_dict())

        completed, degenerate, errored, seen = _load_checkpoint(ck)
        assert len(completed) == 1 and completed[0].seed == "A"
        assert len(degenerate) == 1 and degenerate[0].seed == "B"
        assert len(errored) == 1 and errored[0].seed == "C"
        assert seen == {"A", "B", "C"}

    def test_trailing_partial_line_dropped(self, tmp_path: Path):
        """Process killed mid-write leaves a partial trailing line —
        skip it rather than fail.
        """
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        good = json.dumps({
            "type": "completed",
            "result": _per_feature("A", -0.1, 0.05).to_dict(),
        })
        partial = '{"type": "completed", "result": {"seed":'
        ck.write_text(good + "\n" + partial)
        completed, _, _, seen = _load_checkpoint(ck)
        assert len(completed) == 1
        assert seen == {"A"}

    def test_malformed_non_trailing_line_raises(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        good_a = json.dumps({
            "type": "completed",
            "result": _per_feature("A", -0.1, 0.05).to_dict(),
        })
        bad = '{"this is not": "valid'
        good_b = json.dumps({
            "type": "completed",
            "result": _per_feature("B", -0.2, 0.10).to_dict(),
        })
        ck.write_text(good_a + "\n" + bad + "\n" + good_b + "\n")
        with pytest.raises(RuntimeError, match="malformed line"):
            _load_checkpoint(ck)

    def test_duplicate_seed_raises(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        first = json.dumps({
            "type": "completed",
            "result": _per_feature("A", -0.1, 0.05).to_dict(),
        })
        dup = json.dumps({
            "type": "errored",
            "result": FailedSeed("A", "TimeoutError", "x").to_dict(),
        })
        ck.write_text(first + "\n" + dup + "\n")
        with pytest.raises(RuntimeError, match="duplicate entry"):
            _load_checkpoint(ck)

    def test_unknown_type_raises(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        rec = json.dumps({
            "type": "weird_type",
            "result": _per_feature("A", -0.1, 0.05).to_dict(),
        })
        ck.write_text(rec + "\n")
        with pytest.raises(RuntimeError, match="unknown type"):
            _load_checkpoint(ck)

    def test_writer_returns_none_when_path_is_none(self):
        from cliquefinder.panels.landscape import _checkpoint_writer
        assert _checkpoint_writer(None) is None


class TestPerFeatureLoopWithCheckpoint:
    """The gradient loop respects skip_seeds and invokes the writer
    for each new result.  Exercised without Neo4j by patching
    run_gradient_test.
    """

    def _make_matrix(
        self, seeds: list[str],
    ) -> "FeatureDistanceMatrix":
        # All seeds reachable from each other at distance 1.
        from cliquefinder.panels import FeatureDistanceMatrix
        distances = {
            s: {**{t: 1 for t in seeds if t != s}, s: 0}
            for s in seeds
        }
        return FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=seeds,
            max_hops=2,
        )

    def test_skip_seeds_excluded_from_loop(self, tmp_path: Path):
        """skip_seeds members must not have run_gradient_test called."""
        from unittest.mock import patch
        from cliquefinder.panels.landscape import (
            _per_feature_gradient_loop,
        )

        seeds = ["S1", "S2", "S3"]
        matrix = self._make_matrix(seeds)
        abs_t = {s: 1.0 for s in seeds}

        called_with = []

        def fake_gradient(*args, **kw):
            called_with.append(kw["seed"])
            from cliquefinder.stats.perturbation_gradient import GradientResult, HopShellStats
            return GradientResult(
                seed_gene=kw["seed"],
                shells=(HopShellStats(hop=1, n_genes=2, mean_abs_t=1.0,
                                       median_abs_t=1.0, std_abs_t=0.0,
                                       genes=("X", "Y")),),
                slope=-0.1, slope_pvalue=0.05,
                spearman_rho=-0.1, spearman_pvalue=0.05,
                active_horizon=1, background_mean_abs_t=0.5,
                n_permutations=49, n_genes_total=2,
            )

        with patch(
            "cliquefinder.panels.landscape.run_gradient_test",
            side_effect=fake_gradient,
        ):
            completed, _, _ = _per_feature_gradient_loop(
                measured_feature_ids=seeds,
                abs_t_per_feature=abs_t,
                distance_matrix=matrix,
                graph_degrees={s: 5 for s in seeds},
                unmatched=set(),
                max_hops=2,
                n_permutations=49,
                rng_base=42,
                skip_seeds={"S2"},  # skip the middle one
            )
        assert {r.seed for r in completed} == {"S1", "S3"}
        assert called_with == ["S1", "S3"]  # S2 not called

    def test_writer_invoked_per_completed_seed(self, tmp_path: Path):
        from unittest.mock import patch
        from cliquefinder.panels.landscape import (
            _checkpoint_writer,
            _load_checkpoint,
            _per_feature_gradient_loop,
        )

        seeds = ["S1", "S2"]
        matrix = self._make_matrix(seeds)
        abs_t = {s: 1.0 for s in seeds}
        ck = tmp_path / "checkpoint.jsonl"

        def fake_gradient(*args, **kw):
            from cliquefinder.stats.perturbation_gradient import GradientResult, HopShellStats
            return GradientResult(
                seed_gene=kw["seed"],
                shells=(HopShellStats(hop=1, n_genes=1, mean_abs_t=1.0,
                                       median_abs_t=1.0, std_abs_t=0.0,
                                       genes=("X",)),),
                slope=-0.1, slope_pvalue=0.05,
                spearman_rho=-0.1, spearman_pvalue=0.05,
                active_horizon=1, background_mean_abs_t=0.5,
                n_permutations=49, n_genes_total=1,
            )

        with patch(
            "cliquefinder.panels.landscape.run_gradient_test",
            side_effect=fake_gradient,
        ):
            _per_feature_gradient_loop(
                measured_feature_ids=seeds,
                abs_t_per_feature=abs_t,
                distance_matrix=matrix,
                graph_degrees={s: 5 for s in seeds},
                unmatched=set(),
                max_hops=2,
                n_permutations=49,
                rng_base=42,
                checkpoint_writer=_checkpoint_writer(ck),
            )
        completed, _, _, seen = _load_checkpoint(ck)
        assert seen == {"S1", "S2"}
        assert len(completed) == 2

    def test_degenerate_seed_also_checkpointed(self, tmp_path: Path):
        from cliquefinder.panels import FeatureDistanceMatrix
        from cliquefinder.panels.landscape import (
            _checkpoint_writer,
            _load_checkpoint,
            _per_feature_gradient_loop,
        )

        # S2 has no neighbors → degenerate.  S1 has self-only.
        distances = {
            "S1": {"S1": 0},
            "S2": {"S2": 0},
        }
        matrix = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["S1", "S2"],
            max_hops=2,
        )
        ck = tmp_path / "checkpoint.jsonl"
        _per_feature_gradient_loop(
            measured_feature_ids=["S1", "S2"],
            abs_t_per_feature={"S1": 1.0, "S2": 1.0},
            distance_matrix=matrix,
            graph_degrees={"S1": 0, "S2": 0},
            unmatched={"S2"},
            max_hops=2,
            n_permutations=49,
            rng_base=42,
            checkpoint_writer=_checkpoint_writer(ck),
        )
        _, degenerate, _, _ = _load_checkpoint(ck)
        assert len(degenerate) == 2
        # The unmatched message variant should differ.
        msgs = {f.seed: f.error_message for f in degenerate}
        assert "did not resolve to a BioEntity" in msgs["S2"]
        assert "no measured features reachable" in msgs["S1"]


def _make_dummy_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Create dummy CSV inputs to satisfy fingerprinting + load_panel_inputs.

    Tests that don't actually want compute_landscape to succeed past
    Phase 1 still need the fingerprinting to produce stable hashes.
    """
    data_path = tmp_path / "data.csv"
    meta_path = tmp_path / "meta.csv"
    data_path.write_text("feature_id,S1,S2\nP1,1.0,2.0\nP2,3.0,4.0\n")
    meta_path.write_text("sample_id,group\nS1,A\nS2,B\n")
    return data_path, meta_path


class TestComputeLandscapeResume:
    """End-to-end resume semantics for the validation gate.

    These tests verify the resume preconditions (lock, manifest,
    fingerprint, stale checkpoint) without exercising Phase 3/4.
    Each test triggers an early raise before any expensive work.
    """

    def test_resume_refuses_design_mismatch(self, tmp_path: Path):
        """A design saved in manifest.yaml that differs from the
        in-memory design must abort the resume.
        """
        from cliquefinder.panels.landscape import (
            INPUTS_FINGERPRINT_FILENAME,
            _file_fingerprint,
            compute_landscape,
        )
        out = tmp_path / "ls"
        out.mkdir()
        data_path, meta_path = _make_dummy_csvs(tmp_path)

        # Pre-write a matching inputs fingerprint AND a different design.
        prior = LandscapeDesign(
            contrast=("A", "B"),
            max_hops=2, n_permutations=49, covariates=(),
        )
        prior.save_yaml(out / "manifest.yaml")
        with open(out / INPUTS_FINGERPRINT_FILENAME, "w") as f:
            json.dump({
                "data": _file_fingerprint(data_path),
                "metadata": _file_fingerprint(meta_path),
            }, f)

        new = LandscapeDesign(
            contrast=("X", "Y"),  # different contrast → mismatch
            max_hops=2, n_permutations=49, covariates=(),
        )
        with pytest.raises(RuntimeError, match="different LandscapeDesign"):
            compute_landscape(
                new,
                data_path=data_path,
                metadata_path=meta_path,
                group_resolver=lambda _meta: {},
                indra_env_file=None,
                output_dir=out,
                resume=True,
            )

    def test_resume_refuses_missing_manifest(self, tmp_path: Path):
        """resume=True with no manifest = orphan artifacts; refuse."""
        from cliquefinder.panels.landscape import compute_landscape
        out = tmp_path / "ls"
        out.mkdir()
        data_path, meta_path = _make_dummy_csvs(tmp_path)

        design = LandscapeDesign(
            contrast=("A", "B"), max_hops=2, n_permutations=49, covariates=(),
        )
        with pytest.raises(RuntimeError, match="manifest.yaml does not exist"):
            compute_landscape(
                design,
                data_path=data_path, metadata_path=meta_path,
                group_resolver=lambda _meta: {},
                indra_env_file=None, output_dir=out,
                resume=True,
            )

    def test_resume_refuses_missing_inputs_fingerprint(self, tmp_path: Path):
        """A pre-Wave-24h output_dir lacks inputs.json — refuse to resume
        rather than silently merge against unknown input lineage.
        """
        from cliquefinder.panels.landscape import compute_landscape
        out = tmp_path / "ls"
        out.mkdir()
        data_path, meta_path = _make_dummy_csvs(tmp_path)

        design = LandscapeDesign(
            contrast=("A", "B"), max_hops=2, n_permutations=49, covariates=(),
        )
        design.save_yaml(out / "manifest.yaml")
        # Deliberately do NOT write inputs.json.
        with pytest.raises(RuntimeError, match="inputs.json does not exist"):
            compute_landscape(
                design,
                data_path=data_path, metadata_path=meta_path,
                group_resolver=lambda _meta: {},
                indra_env_file=None, output_dir=out,
                resume=True,
            )

    def test_resume_refuses_data_csv_changed(self, tmp_path: Path):
        """Data CSV swap between runs → resume must refuse."""
        from cliquefinder.panels.landscape import (
            INPUTS_FINGERPRINT_FILENAME,
            _file_fingerprint,
            compute_landscape,
        )
        out = tmp_path / "ls"
        out.mkdir()
        data_path, meta_path = _make_dummy_csvs(tmp_path)
        design = LandscapeDesign(
            contrast=("A", "B"), max_hops=2, n_permutations=49, covariates=(),
        )
        design.save_yaml(out / "manifest.yaml")
        # Persist a fingerprint of the ORIGINAL data.
        original_data = _file_fingerprint(data_path)
        with open(out / INPUTS_FINGERPRINT_FILENAME, "w") as f:
            json.dump({
                "data": original_data,
                "metadata": _file_fingerprint(meta_path),
            }, f)
        # Now mutate the data CSV.
        data_path.write_text("feature_id,S1,S2\nP1,99.0,99.0\nP2,99.0,99.0\n")

        with pytest.raises(RuntimeError, match="data CSV fingerprint changed"):
            compute_landscape(
                design,
                data_path=data_path, metadata_path=meta_path,
                group_resolver=lambda _meta: {},
                indra_env_file=None, output_dir=out,
                resume=True,
            )

    def test_fresh_run_refuses_existing_checkpoint(self, tmp_path: Path):
        """resume=False but checkpoint.jsonl exists from a prior run →
        appending would create duplicate seeds → refuse loudly.
        """
        from cliquefinder.panels.landscape import (
            CHECKPOINT_FILENAME,
            compute_landscape,
        )
        out = tmp_path / "ls"
        out.mkdir()
        data_path, meta_path = _make_dummy_csvs(tmp_path)
        # Stale checkpoint from a prior run.
        (out / CHECKPOINT_FILENAME).write_text(
            json.dumps({
                "type": "completed",
                "result": _per_feature("STALE", -0.1, 0.05).to_dict(),
            }) + "\n",
        )

        design = LandscapeDesign(
            contrast=("A", "B"), max_hops=2, n_permutations=49, covariates=(),
        )
        with pytest.raises(RuntimeError, match="already exists from a prior run"):
            compute_landscape(
                design,
                data_path=data_path, metadata_path=meta_path,
                group_resolver=lambda _meta: {},
                indra_env_file=None, output_dir=out,
                resume=False, checkpoint=True,
            )


class TestLandscapeLock:
    def test_lock_refuses_concurrent_holders(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _LandscapeLock
        lock_path = tmp_path / "landscape.lock"
        with _LandscapeLock(lock_path):
            with pytest.raises(RuntimeError, match="Another landscape run"):
                with _LandscapeLock(lock_path):
                    pass

    def test_lock_released_on_exit(self, tmp_path: Path):
        """After __exit__, a second acquisition should succeed."""
        from cliquefinder.panels.landscape import _LandscapeLock
        lock_path = tmp_path / "landscape.lock"
        with _LandscapeLock(lock_path):
            pass
        # Lock file is removed.
        assert not lock_path.exists()
        # Second acquisition succeeds.
        with _LandscapeLock(lock_path):
            pass


class TestFileFingerprint:
    def test_fingerprint_changes_with_content(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _file_fingerprint
        p = tmp_path / "x.csv"
        p.write_text("a,b\n1,2\n")
        fp1 = _file_fingerprint(p)
        p.write_text("a,b\n1,3\n")  # different content
        fp2 = _file_fingerprint(p)
        assert fp1["sha256"] != fp2["sha256"]

    def test_fingerprint_stable_across_calls(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _file_fingerprint
        p = tmp_path / "x.csv"
        p.write_text("a,b\n1,2\n")
        assert _file_fingerprint(p) == _file_fingerprint(p)


class TestSaveNpzWithDegrees:
    def test_degrees_persisted_in_meta(self, tmp_path: Path):
        from cliquefinder.panels import FeatureDistanceMatrix
        distances = {
            "A": {"A": 0, "B": 1},
            "B": {"B": 0, "A": 1},
        }
        m = FeatureDistanceMatrix.from_distance_dict(
            distances=distances,
            feature_names=["A", "B"], max_hops=2,
        )
        path = tmp_path / "d.npz"
        m.save_npz(path, graph_degrees={"A": 5, "B": 7})
        meta = json.loads((path.with_suffix(".meta.json")).read_text())
        assert meta["graph_degrees"] == {"A": 5, "B": 7}

    def test_no_degrees_when_omitted(self, tmp_path: Path):
        from cliquefinder.panels import FeatureDistanceMatrix
        distances = {"A": {"A": 0}}
        m = FeatureDistanceMatrix.from_distance_dict(
            distances=distances, feature_names=["A"], max_hops=2,
        )
        path = tmp_path / "d.npz"
        m.save_npz(path)  # no graph_degrees
        meta = json.loads((path.with_suffix(".meta.json")).read_text())
        assert "graph_degrees" not in meta


class TestPathTraversalTag:
    """Wave 24l: distance matrices are stamped with path_traversal."""

    def test_save_writes_measured_only_tag(self, tmp_path: Path):
        from cliquefinder.panels import FeatureDistanceMatrix
        m = FeatureDistanceMatrix.from_distance_dict(
            distances={"A": {"A": 0}},
            feature_names=["A"], max_hops=2,
        )
        path = tmp_path / "d.npz"
        m.save_npz(path)
        meta = json.loads((path.with_suffix(".meta.json")).read_text())
        assert meta["path_traversal"] == "measured_only"

    def test_resume_refuses_untagged_matrix(self, tmp_path: Path):
        """An untagged matrix (pre-wave-24l artifact) must not be
        reused during a measured-only-paths resume."""
        from cliquefinder.panels import FeatureDistanceMatrix
        from cliquefinder.panels.landscape import (
            _load_or_build_distance_matrix,
        )
        m = FeatureDistanceMatrix.from_distance_dict(
            distances={"A": {"A": 0, "B": 1}, "B": {"A": 1, "B": 0}},
            feature_names=["A", "B"], max_hops=2,
        )
        path = tmp_path / "d.npz"
        m.save_npz(path)
        # Remove the path_traversal tag to simulate a stale artifact.
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        del meta["path_traversal"]
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(
            RuntimeError, match="path_traversal=None"
        ):
            _load_or_build_distance_matrix(
                matrix_path=path, resume=True, bridge=None,
                measured_symbols=["A", "B"],
                measured_feature_ids=["A", "B"],
                sym_to_feat={"A": "A", "B": "B"},
                max_hops=2, seed_batch_size=500,
            )

    def test_resume_refuses_wrong_traversal_tag(self, tmp_path: Path):
        from cliquefinder.panels import FeatureDistanceMatrix
        from cliquefinder.panels.landscape import (
            _load_or_build_distance_matrix,
        )
        m = FeatureDistanceMatrix.from_distance_dict(
            distances={"A": {"A": 0, "B": 1}, "B": {"A": 1, "B": 0}},
            feature_names=["A", "B"], max_hops=2,
        )
        path = tmp_path / "d.npz"
        m.save_npz(path)
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["path_traversal"] = "with_intermediates"
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(
            RuntimeError, match="path_traversal='with_intermediates'"
        ):
            _load_or_build_distance_matrix(
                matrix_path=path, resume=True, bridge=None,
                measured_symbols=["A", "B"],
                measured_feature_ids=["A", "B"],
                sym_to_feat={"A": "A", "B": "B"},
                max_hops=2, seed_batch_size=500,
            )


class TestCheckpointRobustness:
    def test_invalid_utf8_trailing_dropped(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        good = json.dumps({
            "type": "completed",
            "result": _per_feature("A", -0.1, 0.05).to_dict(),
        })
        # Append bytes that form an incomplete UTF-8 multi-byte sequence.
        with open(ck, "wb") as f:
            f.write(good.encode("utf-8") + b"\n")
            f.write(b"\xc3")  # truncated UTF-8 lead byte (no continuation)
        completed, _, _, seen = _load_checkpoint(ck)
        assert seen == {"A"}
        assert len(completed) == 1

    def test_non_dict_json_raises(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        ck.write_text("[1, 2, 3]\n")
        with pytest.raises(RuntimeError, match="expected JSON object"):
            _load_checkpoint(ck)

    def test_non_dict_result_raises(self, tmp_path: Path):
        from cliquefinder.panels.landscape import _load_checkpoint
        ck = tmp_path / "checkpoint.jsonl"
        ck.write_text(json.dumps({"type": "completed", "result": "string"}) + "\n")
        with pytest.raises(RuntimeError, match="result.*must be a JSON object"):
            _load_checkpoint(ck)
