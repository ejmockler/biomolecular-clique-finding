"""Landscape module — pure-Python tests (no Neo4j, no INDRA).

Compute_landscape itself is tested via the integration smoke; these
tests cover the data model, FeatureDistanceMatrix sparse handling,
LandscapeResult validation, and analyze_landscape statistical
behavior.
"""
from __future__ import annotations

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
