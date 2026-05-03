"""analyze_panel: BH-FDR, Bonferroni, Mann-Whitney U, target rank."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cliquefinder.panels import (
    FailedSeed,
    PanelDesign,
    PanelResult,
    PanelStratum,
    PerSeedResult,
    ShellSummary,
    TARGET_STRATUM_LABEL,
    analyze_panel,
)


def _make_seed_result(
    seed: str, stratum: str, slope: float, p: float
) -> PerSeedResult:
    return PerSeedResult(
        seed=seed,
        stratum=stratum,
        slope=slope,
        slope_pvalue=p,
        spearman_rho=0.0,
        spearman_pvalue=1.0,
        shells=(
            ShellSummary(hop=1, n_genes=10, mean_abs_t=1.0, median_abs_t=1.0),
            ShellSummary(hop=2, n_genes=20, mean_abs_t=0.9, median_abs_t=0.9),
        ),
        n_genes_total=30,
        elapsed_seconds=10.0,
    )


def _make_panel_result(
    target_p: float = 0.01,
    target_slope: float = -0.16,
    panel_specs: list[tuple[str, str, float, float]] | None = None,
) -> PanelResult:
    """Build a minimal PanelResult.  panel_specs: [(seed, stratum, slope, p), ...]."""
    if panel_specs is None:
        panel_specs = [
            ("HNRNPK", "RNA_RBP", -0.14, 0.02),
            ("G3BP1", "RNA_RBP", -0.10, 0.05),
            ("VCL", "Cytoskeletal", -0.01, 0.50),
            ("MSN", "Cytoskeletal", +0.03, 0.70),
        ]
    strata_dict: dict[str, list[str]] = {}
    for seed, stratum, _, _ in panel_specs:
        strata_dict.setdefault(stratum, []).append(seed)
    design = PanelDesign(
        target_seed="C9orf72",
        strata=tuple(
            PanelStratum(name=name, members=tuple(sorted(members)))
            for name, members in strata_dict.items()
        ),
        contrast=("C9ORF72", "SPORADIC"),
        max_hops=2,
        n_permutations=999,
        covariates=("Sex",),
        selection_rng_seed=42,
    )
    target = _make_seed_result(
        "C9orf72", TARGET_STRATUM_LABEL, target_slope, target_p,
    )
    per_seed = tuple(
        _make_seed_result(seed, stratum, slope, p)
        for seed, stratum, slope, p in panel_specs
    )
    return PanelResult(
        design=design, target_result=target, per_seed=per_seed,
    )


# --- BH-FDR backed by statsmodels (via fdr_correction) ----------------------


class TestBHIntegration:
    """We delegate to ``cliquefinder.stats.differential.fdr_correction``,
    which is statsmodels-backed.  These tests exist to ensure the
    delegation flows correctly through analyze_panel — the BH
    correctness itself is statsmodels' problem.
    """

    def test_bh_q_attached_to_each_panel_seed(self):
        result = _make_panel_result()
        analysis = analyze_panel(result)
        for adj in analysis.panel_seeds_adjusted:
            assert 0.0 <= adj.bh_qvalue <= 1.0
            assert adj.bh_qvalue >= adj.slope_pvalue  # BH is monotone with raw p

    def test_nan_panel_pvalue_does_not_poison_others(self):
        """A NaN p-value must not silently propagate NaN into every
        other seed's q.  (statsmodels' fdr_correction handles NaN by
        masking; the panel layer relies on that contract.)
        """
        result = _make_panel_result(panel_specs=[
            ("HNRNPK", "RNA_RBP", -0.14, 0.02),
            ("G3BP1", "RNA_RBP", -0.10, float("nan")),
            ("VCL", "Cytoskeletal", -0.01, 0.50),
            ("MSN", "Cytoskeletal", +0.03, 0.70),
        ])
        analysis = analyze_panel(result)
        finite_qs = [
            adj.bh_qvalue for adj in analysis.panel_seeds_adjusted
            if not np.isnan(adj.bh_qvalue)
        ]
        # 3 of 4 should be finite (the non-NaN inputs).
        assert len(finite_qs) == 3
        # NaN input should yield NaN q AND non-discovery.
        nan_seed = next(
            adj for adj in analysis.panel_seeds_adjusted
            if adj.seed == "G3BP1"
        )
        assert np.isnan(nan_seed.bh_qvalue)
        assert nan_seed.discovery is False


# --- analyze_panel basics ---------------------------------------------------


class TestPanelAnalysis:
    def test_runs_and_returns_complete_struct(self):
        result = _make_panel_result()
        analysis = analyze_panel(result)
        assert analysis.design == result.design
        assert len(analysis.panel_seeds_adjusted) == len(result.per_seed)
        assert analysis.target_adjusted.seed == "C9orf72"

    def test_panel_seeds_get_bh_qvalues(self):
        result = _make_panel_result()
        analysis = analyze_panel(result)
        for adj, raw in zip(analysis.panel_seeds_adjusted, result.per_seed):
            assert adj.slope_pvalue == raw.slope_pvalue
            assert adj.bh_qvalue >= adj.slope_pvalue
            assert 0.0 <= adj.bh_qvalue <= 1.0

    def test_panel_seeds_get_bonferroni(self):
        result = _make_panel_result()  # n=4 panel, 0 failed → family = 4
        analysis = analyze_panel(result)
        for adj, raw in zip(analysis.panel_seeds_adjusted, result.per_seed):
            expected = min(raw.slope_pvalue * 4, 1.0)
            assert adj.bonferroni_pvalue == pytest.approx(expected)

    def test_target_treated_as_single_test(self):
        """Target's q-value and Bonferroni p both collapse to raw p."""
        result = _make_panel_result(target_p=0.01)
        analysis = analyze_panel(result)
        assert analysis.target_adjusted.bh_qvalue == pytest.approx(0.01)
        assert analysis.target_adjusted.bonferroni_pvalue == pytest.approx(0.01)

    def test_discovery_flag_at_q_threshold(self):
        result = _make_panel_result()
        analysis = analyze_panel(result, q_threshold=0.05)
        for adj in analysis.panel_seeds_adjusted:
            assert adj.discovery == (adj.bh_qvalue < 0.05)

    def test_target_uses_same_threshold_as_panel(self):
        """No two-knob asymmetry: target and panel share q_threshold."""
        result = _make_panel_result(target_p=0.04)
        analysis = analyze_panel(result, q_threshold=0.05)
        assert analysis.target_adjusted.discovery is True
        analysis_strict = analyze_panel(result, q_threshold=0.01)
        assert analysis_strict.target_adjusted.discovery is False

    def test_failed_seeds_inflate_correction_family(self):
        """Failed seeds must NOT shrink the multiple-testing family —
        otherwise non-random failures make discoveries anti-conservative.
        """
        # Build a panel with 4 completed seeds and 2 failed.
        design = PanelDesign(
            target_seed="C9orf72",
            strata=(
                PanelStratum(name="RNA_RBP", members=("G3BP1", "HNRNPK")),
                PanelStratum(
                    name="Cytoskeletal",
                    members=("FAILA", "FAILB", "MSN", "VCL"),
                ),
            ),
            contrast=("C9ORF72", "SPORADIC"),
            max_hops=2,
            n_permutations=999,
            covariates=("Sex",),
            selection_rng_seed=42,
        )
        target = _make_seed_result(
            "C9orf72", TARGET_STRATUM_LABEL, -0.16, 0.01,
        )
        per_seed = (
            _make_seed_result("G3BP1", "RNA_RBP", -0.10, 0.05),
            _make_seed_result("HNRNPK", "RNA_RBP", -0.14, 0.02),
            _make_seed_result("MSN", "Cytoskeletal", +0.03, 0.70),
            _make_seed_result("VCL", "Cytoskeletal", -0.01, 0.50),
        )
        result_with_fail = PanelResult(
            design=design,
            target_result=target,
            per_seed=per_seed,
            failed_seeds=(
                FailedSeed("FAILA", "RuntimeError", "synthetic"),
                FailedSeed("FAILB", "RuntimeError", "synthetic"),
            ),
        )
        # Bonferroni multiplier should be 6 (attempted), not 4 (completed).
        analysis = analyze_panel(result_with_fail)
        for adj, raw in zip(analysis.panel_seeds_adjusted, per_seed):
            expected = min(raw.slope_pvalue * 6, 1.0)
            assert adj.bonferroni_pvalue == pytest.approx(expected), (
                f"Bonferroni for {adj.seed}: family must be 6 attempted, "
                f"not 4 completed"
            )


# --- Mann-Whitney stratum comparisons --------------------------------------


class TestStratumComparisons:
    def test_emits_one_row_per_unordered_pair(self):
        """Two strata → one comparison row, not two.  Reciprocal
        direction is recoverable from u_statistic + n_a*n_b.
        """
        result = _make_panel_result()  # 2 strata: RNA_RBP, Cytoskeletal
        analysis = analyze_panel(result)
        assert len(analysis.stratum_comparisons) == 1
        c = analysis.stratum_comparisons[0]
        # Sorted lexicographically: Cytoskeletal < RNA_RBP
        assert (c.stratum_a, c.stratum_b) == ("Cytoskeletal", "RNA_RBP")

    def test_one_sided_correctly_oriented(self):
        """alternative='less' tests stratum_a < stratum_b.  Cyto
        slopes (~0) > RNA slopes (~-0.12), so the test direction
        Cyto<RNA should fail (large p) and we infer RNA<Cyto from
        the small u_statistic.
        """
        result = _make_panel_result()
        analysis = analyze_panel(result)
        c = analysis.stratum_comparisons[0]
        # Cyto medians > RNA medians
        assert c.median_a > c.median_b  # cyto > rna
        # "Cyto < RNA" should be a large p (wrong direction)
        assert c.pvalue > 0.5

    def test_three_strata_three_unordered_pairs(self):
        """3 strata → 3 unordered pairs, not 6."""
        result = _make_panel_result(panel_specs=[
            ("HNRNPK", "RNA_RBP", -0.14, 0.02),
            ("G3BP1", "RNA_RBP", -0.10, 0.05),
            ("VCL", "Cytoskeletal", -0.01, 0.50),
            ("MSN", "Cytoskeletal", +0.03, 0.70),
            ("ACLY", "Metabolic", -0.05, 0.30),
            ("AIFM1", "Metabolic", -0.08, 0.20),
        ])
        analysis = analyze_panel(result)
        assert len(analysis.stratum_comparisons) == 3

    def test_single_stratum_yields_no_comparisons(self):
        """One stratum → zero pairs."""
        result = _make_panel_result(panel_specs=[
            ("HNRNPK", "RNA_RBP", -0.14, 0.02),
            ("G3BP1", "RNA_RBP", -0.10, 0.05),
        ])
        analysis = analyze_panel(result)
        assert len(analysis.stratum_comparisons) == 0


# --- Target rank ------------------------------------------------------------


class TestTargetRank:
    def test_target_most_negative_in_panel(self):
        """Target slope (-0.16) is more negative than all panel members
        (-0.14, -0.10, -0.01, +0.03)."""
        result = _make_panel_result(target_slope=-0.16)
        analysis = analyze_panel(result)
        # 0 panel slopes <= target → rank 1, p = 1/(4+1) = 0.2
        assert analysis.target_position.rank_left_tail == 1
        assert analysis.target_position.empirical_p_left == pytest.approx(0.2)

    def test_target_least_negative_in_panel(self):
        """Target slope (+0.10) is more positive than all panel members."""
        result = _make_panel_result(target_slope=+0.10)
        analysis = analyze_panel(result)
        # All 4 panel slopes <= target → rank 5, p = 5/5 = 1.0
        assert analysis.target_position.rank_left_tail == 5
        assert analysis.target_position.empirical_p_left == pytest.approx(1.0)

    def test_target_middle_of_pack(self):
        """Target slope (-0.05) sits between -0.10 and -0.01."""
        result = _make_panel_result(target_slope=-0.05)
        analysis = analyze_panel(result)
        # Panel slopes: -0.14, -0.10, -0.01, +0.03
        # 2 are <= -0.05 → rank 3, p = 3/5 = 0.6
        assert analysis.target_position.rank_left_tail == 3
        assert analysis.target_position.empirical_p_left == pytest.approx(0.6)

    def test_target_position_panel_n(self):
        result = _make_panel_result()
        analysis = analyze_panel(result)
        assert analysis.target_position.panel_n == 4


# --- Round-trip serialization for results ----------------------------------


class TestPanelResultSerialization:
    def test_to_dict_then_from_dict(self):
        result = _make_panel_result()
        recovered = PanelResult.from_dict(result.to_dict())
        assert recovered == result

    def test_save_load_json(self, tmp_path: Path):
        result = _make_panel_result()
        path = tmp_path / "panel.json"
        result.save_json(path)
        recovered = PanelResult.load_json(path)
        assert recovered == result

    def test_target_seed_mismatch_raises(self):
        result = _make_panel_result()
        bad_target = PerSeedResult(
            seed="WRONG",
            stratum=TARGET_STRATUM_LABEL,
            slope=0.0, slope_pvalue=1.0, spearman_rho=0.0, spearman_pvalue=1.0,
            shells=(), n_genes_total=0, elapsed_seconds=0.0,
        )
        with pytest.raises(ValueError, match="does not match"):
            PanelResult(
                design=result.design,
                target_result=bad_target,
                per_seed=result.per_seed,
            )

    def test_target_stratum_label_enforced(self):
        result = _make_panel_result()
        bad_target = PerSeedResult(
            seed=result.design.target_seed,
            stratum="RNA_RBP",  # wrong: must be TARGET_STRATUM_LABEL
            slope=0.0, slope_pvalue=1.0, spearman_rho=0.0, spearman_pvalue=1.0,
            shells=(), n_genes_total=0, elapsed_seconds=0.0,
        )
        with pytest.raises(ValueError, match="must be"):
            PanelResult(
                design=result.design,
                target_result=bad_target,
                per_seed=result.per_seed,
            )


class TestPanelResultCoherence:
    """PanelResult.__post_init__ must enforce that per_seed matches
    the bundled design.  Without this, analyze_panel will compute
    against mismatched data.
    """

    def test_rejects_per_seed_with_extra_seed(self):
        result = _make_panel_result()
        extra = _make_seed_result("EXTRA", "RNA_RBP", -0.1, 0.05)
        with pytest.raises(ValueError, match="unexpected seeds"):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=result.per_seed + (extra,),
            )

    def test_rejects_per_seed_with_missing_seed(self):
        result = _make_panel_result()
        # Drop the last seed without marking it failed.
        with pytest.raises(ValueError, match="missing seeds"):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=result.per_seed[:-1],
            )

    def test_failed_seeds_complete_partition(self):
        """Failed + completed must equal design.selected_seeds()."""
        result = _make_panel_result()
        # Drop last seed from per_seed and mark it failed.
        last_seed = result.per_seed[-1].seed
        ok = PanelResult(
            design=result.design,
            target_result=result.target_result,
            per_seed=result.per_seed[:-1],
            failed_seeds=(
                FailedSeed(last_seed, "RuntimeError", "synthetic"),
            ),
        )
        assert ok.failed_seeds[0].seed == last_seed

    def test_rejects_seed_in_both_completed_and_failed(self):
        result = _make_panel_result()
        with pytest.raises(ValueError, match="appear in both"):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=result.per_seed,
                failed_seeds=(
                    FailedSeed(
                        result.per_seed[0].seed,
                        "RuntimeError", "synthetic",
                    ),
                ),
            )

    def test_rejects_per_seed_with_wrong_stratum_label(self):
        result = _make_panel_result()
        # Swap a per_seed row's stratum to a different valid stratum name.
        bad_per_seed = list(result.per_seed)
        bad_row = bad_per_seed[0]  # is in "RNA_RBP"
        bad_per_seed[0] = PerSeedResult(
            seed=bad_row.seed,
            stratum="Cytoskeletal",  # WRONG: design says RNA_RBP
            slope=bad_row.slope,
            slope_pvalue=bad_row.slope_pvalue,
            spearman_rho=bad_row.spearman_rho,
            spearman_pvalue=bad_row.spearman_pvalue,
            shells=bad_row.shells,
            n_genes_total=bad_row.n_genes_total,
            elapsed_seconds=bad_row.elapsed_seconds,
        )
        with pytest.raises(ValueError, match="design assigns it to"):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=tuple(bad_per_seed),
            )

    def test_rejects_target_in_per_seed(self):
        result = _make_panel_result()
        # Construct a design where target is also a panel member
        # (impossible at PanelDesign-level, but we test PanelResult's
        # defense-in-depth check).
        rogue_target_in_panel = _make_seed_result(
            result.design.target_seed,
            result.per_seed[0].stratum,
            -0.1, 0.05,
        )
        # Replace one panel member with the target-named row.
        bad_per_seed = (rogue_target_in_panel,) + result.per_seed[1:]
        # PanelDesign forbade target in any stratum, so this also
        # mismatches the design (extra C9orf72, missing the original).
        with pytest.raises(ValueError):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=bad_per_seed,
            )

    def test_rejects_duplicate_seeds_in_per_seed(self):
        result = _make_panel_result()
        bad_per_seed = result.per_seed + (result.per_seed[0],)
        with pytest.raises(ValueError, match="duplicate seeds"):
            PanelResult(
                design=result.design,
                target_result=result.target_result,
                per_seed=bad_per_seed,
            )


class TestFrozenCoercion:
    """List inputs to constructor must be coerced to tuples so the
    frozen guarantee covers collection mutability, not just rebinding.
    """

    def test_panel_stratum_accepts_list_and_coerces(self):
        s = PanelStratum(name="X", members=["A", "B"])  # type: ignore[arg-type]
        assert isinstance(s.members, tuple)
        assert s.members == ("A", "B")

    def test_panel_design_accepts_lists_and_coerces(self):
        d = PanelDesign(
            target_seed="C9orf72",
            strata=[PanelStratum("X", ("A",))],  # type: ignore[arg-type]
            contrast=["a", "b"],  # type: ignore[arg-type]
            max_hops=2, n_permutations=999,
            covariates=["Sex"],  # type: ignore[arg-type]
            selection_rng_seed=0,
        )
        assert isinstance(d.strata, tuple)
        assert isinstance(d.contrast, tuple)
        assert isinstance(d.covariates, tuple)


class TestStratumNameReservation:
    def test_target_stratum_label_rejected_as_user_stratum(self):
        with pytest.raises(ValueError, match="reserved"):
            PanelStratum(name=TARGET_STRATUM_LABEL, members=("A",))
