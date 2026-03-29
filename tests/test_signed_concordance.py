"""Tests for signed concordance between INDRA predictions and observed data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.signed_concordance import (
    SignedConcordanceResult,
    compute_signed_concordance,
    predict_direction,
)
from cliquefinder.stats.target_set import TargetSet


class TestPredictDirection:
    def test_activation_only_lof(self):
        edges = [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}]
        assert predict_direction(edges, loss_of_function=True) == "predicted_down"

    def test_repression_only_lof(self):
        edges = [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}]
        assert predict_direction(edges, loss_of_function=True) == "predicted_up"

    def test_mixed_returns_none(self):
        edges = [
            {"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1},
            {"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1},
        ]
        assert predict_direction(edges) is None

    def test_phosphorylation_only_returns_none(self):
        edges = [{"regulation_type": "phosphorylation", "sources": ["reach"], "evidence_count": 1}]
        assert predict_direction(edges) is None

    def test_gain_of_function_reverses(self):
        edges = [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}]
        assert predict_direction(edges, loss_of_function=False) == "predicted_up"


def _make_protein_df(t_stats: dict[str, float], targets: set[str]) -> pd.DataFrame:
    """Helper: build a protein_df with feature_id, t_statistic, is_target."""
    rows = []
    for fid, t in t_stats.items():
        rows.append({"feature_id": fid, "t_statistic": t, "is_target": fid in targets})
    return pd.DataFrame(rows)


def _make_target_set(
    targets: dict[str, str],
    edge_meta: dict[str, list[dict]],
) -> TargetSet:
    return TargetSet.from_query(
        targets_in_data=targets,
        gene_symbol="TEST",
        min_evidence=1,
        n_hops=1,
        edge_metadata=edge_meta,
    )


class TestComputeSignedConcordance:
    def test_all_concordant(self):
        """All activation targets down, all repression targets up."""
        targets = {"A": "P1", "B": "P2", "C": "P3"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "B": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "C": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
        }
        t_stats = {"P1": -2.5, "P2": -1.3, "P3": 1.8, "P4": -0.5, "P5": 0.3}
        protein_df = _make_protein_df(t_stats, {"P1", "P2", "P3"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        assert result.n_unambiguous == 3
        assert result.n_concordant == 3
        assert result.concordance_rate == 1.0
        assert result.n_predicted_down == 2
        assert result.n_predicted_up == 1

    def test_all_discordant(self):
        """All predictions wrong."""
        targets = {"A": "P1", "B": "P2"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "B": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
        }
        t_stats = {"P1": 2.0, "P2": -1.5, "P3": 0.1}
        protein_df = _make_protein_df(t_stats, {"P1", "P2"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        assert result.n_concordant == 0
        assert result.concordance_rate == 0.0

    def test_subgroup_breakdown(self):
        """Activation and repression groups reported separately."""
        targets = {"A": "P1", "B": "P2", "C": "P3"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "B": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "C": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
        }
        # A,B concordant (activation, t<0), C discordant (repression, t<0)
        t_stats = {"P1": -1.0, "P2": -2.0, "P3": -1.5, "P4": 0.5}
        protein_df = _make_protein_df(t_stats, {"P1", "P2", "P3"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        assert result.activation_subgroup is not None
        assert result.activation_subgroup["n"] == 2
        assert result.activation_subgroup["n_concordant"] == 2

        assert result.repression_subgroup is not None
        assert result.repression_subgroup["n"] == 1
        assert result.repression_subgroup["n_concordant"] == 0

    def test_mixed_targets_excluded(self):
        targets = {"A": "P1", "B": "P2"}
        edge_meta = {
            "A": [
                {"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1},
                {"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1},
            ],
            "B": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
        }
        t_stats = {"P1": -1.0, "P2": -2.0}
        protein_df = _make_protein_df(t_stats, {"P1", "P2"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        assert result.n_unambiguous == 1
        assert result.n_mixed_excluded == 1

    def test_no_edge_metadata_returns_empty(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1"},
            gene_symbol="X", min_evidence=1, n_hops=1,
        )
        protein_df = _make_protein_df({"P1": -1.0}, {"P1"})
        result = compute_signed_concordance(protein_df, ts)

        assert result.n_unambiguous == 0
        assert result.binomial_pvalue == 1.0
        assert result.permutation_pvalue == 1.0

    def test_gof_sensitivity(self):
        """Gain-of-function model tested as sensitivity analysis."""
        targets = {"A": "P1", "B": "P2"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
            "B": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
        }
        # GoF: activation→up, repression→down. A has t>0 (concordant under GoF), B has t<0 (concordant under GoF)
        t_stats = {"P1": 2.0, "P2": -1.5, "P3": 0.1, "P4": -0.1}
        protein_df = _make_protein_df(t_stats, {"P1", "P2"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        # LoF: 0/2 concordant (A:act→down but t>0, B:rep→up but t<0)
        assert result.concordance_rate == 0.0
        # GoF: 2/2 concordant
        assert result.gof_concordance_rate == 1.0

    def test_permutation_pvalue_exists(self):
        """Permutation null produces a valid p-value."""
        n = 20
        targets = {f"G{i}": f"P{i}" for i in range(n)}
        edge_meta = {
            f"G{i}": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}]
            for i in range(n)
        }
        t_stats = {f"P{i}": -2.0 for i in range(n)}
        t_stats.update({f"BG{i}": (1.0 if i % 2 == 0 else -1.0) for i in range(100)})
        protein_df = _make_protein_df(t_stats, {f"P{i}" for i in range(n)})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=200, seed=42)

        assert 0.0 < result.permutation_pvalue <= 1.0
        assert result.n_permutations == 200

    def test_background_excludes_targets(self):
        """Background rate should exclude target proteins."""
        targets = {"A": "P1"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
        }
        # Only background proteins: all positive
        t_stats = {"P1": -1.0, "BG1": 1.0, "BG2": 1.0, "BG3": 1.0}
        protein_df = _make_protein_df(t_stats, {"P1"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        # Background frac_negative should be 0/3 = 0.0 (only BG proteins)
        # not 1/4 = 0.25 (which would include P1)
        # background_rate = p_pred_down * 0.0 + p_pred_up * 1.0
        # With 1 predicted_down, 0 predicted_up: bg = 1.0 * 0.01 (clamped) = 0.01
        assert result.background_concordance_rate < 0.05

    def test_result_to_dict(self):
        result = SignedConcordanceResult(
            n_unambiguous=10, n_concordant=7, concordance_rate=0.7,
            background_concordance_rate=0.5, binomial_pvalue=0.05,
            permutation_pvalue=0.04,
            n_predicted_down=6, n_predicted_up=4,
            n_mixed_excluded=2, n_no_tstat=1,
        )
        d = result.to_dict()
        assert d["n_unambiguous"] == 10
        assert d["permutation_pvalue"] == 0.04
        assert "activation_subgroup" in d
        assert "gof_concordance_rate" in d
        assert "best_model" in d

    def test_best_model_selection(self):
        """best_model reflects which model fits better."""
        targets = {"A": "P1"}
        edge_meta = {
            "A": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
        }
        # Neither model significant with n=1
        t_stats = {"P1": -1.0, "BG1": 0.5, "BG2": -0.5}
        protein_df = _make_protein_df(t_stats, {"P1"})

        ts = _make_target_set(targets, edge_meta)
        result = compute_signed_concordance(protein_df, ts, n_permutations=100, seed=42)

        assert result.best_model == "neither"
