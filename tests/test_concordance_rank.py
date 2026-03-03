"""Tests for compute_concordance_rank (SCI-III-2 mitigation).

Covers:
    - Basic ranking by cross-method significance count
    - Tie-breaking by geometric mean p-value
    - Dense ranking semantics (ties share same rank)
    - Edge cases: single method, single gene set, all invalid results
    - Alpha threshold sensitivity
    - Empty input raises ValueError
    - Invalid results are excluded
"""

from __future__ import annotations

import numpy as np
import pytest

from cliquefinder.stats.concordance import compute_concordance_rank
from cliquefinder.stats.method_comparison_types import (
    MethodName,
    UnifiedCliqueResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    clique_id: str,
    method: MethodName,
    p_value: float,
    effect_size: float = 1.0,
) -> UnifiedCliqueResult:
    """Build a minimal UnifiedCliqueResult for testing."""
    return UnifiedCliqueResult(
        clique_id=clique_id,
        method=method,
        effect_size=effect_size,
        effect_size_se=0.1,
        p_value=p_value,
        statistic_value=2.0,
        statistic_type="t",
        degrees_of_freedom=10.0,
        n_proteins=10,
        n_proteins_found=8,
    )


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------


class TestConcordanceRankBasic:
    """Basic ranking by n_methods_significant then geomean p-value."""

    def test_rank_by_significance_count(self):
        """Gene set significant in more methods ranks higher."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.001),
                _make_result("B", MethodName.OLS, p_value=0.001),
                _make_result("C", MethodName.OLS, p_value=0.5),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.001),
                _make_result("B", MethodName.ROAST_MSQ, p_value=0.5),
                _make_result("C", MethodName.ROAST_MSQ, p_value=0.5),
            ],
            MethodName.PERMUTATION_COMPETITIVE: [
                _make_result("A", MethodName.PERMUTATION_COMPETITIVE, p_value=0.001),
                _make_result("B", MethodName.PERMUTATION_COMPETITIVE, p_value=0.5),
                _make_result("C", MethodName.PERMUTATION_COMPETITIVE, p_value=0.5),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A: significant in 3 methods -> rank 1
        # B: significant in 1 method  -> rank 2
        # C: significant in 0 methods -> rank 3
        assert ranks["A"] == 1
        assert ranks["B"] == 2
        assert ranks["C"] == 3

    def test_tiebreak_by_geomean_pvalue(self):
        """Gene sets with same significance count are ranked by geomean p."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.001),
                _make_result("B", MethodName.OLS, p_value=0.01),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.001),
                _make_result("B", MethodName.ROAST_MSQ, p_value=0.01),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # Both significant in 2 methods, but A has lower geomean p
        assert ranks["A"] < ranks["B"]

    def test_single_method(self):
        """Ranking works with a single method."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.001),
                _make_result("B", MethodName.OLS, p_value=0.1),
                _make_result("C", MethodName.OLS, p_value=0.01),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A and C significant (1 method each), B not significant (0 methods)
        # A has lower p than C -> rank 1
        assert ranks["A"] == 1
        assert ranks["C"] == 2
        assert ranks["B"] == 3

    def test_single_gene_set(self):
        """Ranking works with a single gene set."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.01),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.02),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)
        assert ranks == {"A": 1}


# ---------------------------------------------------------------------------
# Dense ranking and ties
# ---------------------------------------------------------------------------


class TestConcordanceRankTies:
    """Dense ranking semantics: ties share rank, next rank is offset."""

    def test_tied_significance_and_geomean(self):
        """Gene sets with identical (n_sig, geomean_p) get same rank."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.01),
                _make_result("B", MethodName.OLS, p_value=0.01),
                _make_result("C", MethodName.OLS, p_value=0.5),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.01),
                _make_result("B", MethodName.ROAST_MSQ, p_value=0.01),
                _make_result("C", MethodName.ROAST_MSQ, p_value=0.5),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A and B have same n_sig (2) and same geomean (0.01) -> tied at rank 1
        assert ranks["A"] == ranks["B"] == 1
        # C has n_sig=0 -> rank 3 (not rank 2, because positions 1,2 taken)
        assert ranks["C"] == 3

    def test_dense_ranking_gap(self):
        """After a tie, the next rank skips appropriately."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.01),
                _make_result("B", MethodName.OLS, p_value=0.01),
                _make_result("C", MethodName.OLS, p_value=0.01),
                _make_result("D", MethodName.OLS, p_value=0.5),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A, B, C all significant with same p-value -> tied at rank 1
        assert ranks["A"] == ranks["B"] == ranks["C"] == 1
        # D not significant -> rank 4 (positions 1, 2, 3 occupied by the tie)
        assert ranks["D"] == 4


# ---------------------------------------------------------------------------
# Alpha threshold sensitivity
# ---------------------------------------------------------------------------


class TestConcordanceRankAlpha:
    """Alpha threshold affects significance classification."""

    def test_stricter_alpha_changes_ranking(self):
        """Using alpha=0.001 instead of 0.05 changes counts."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.0001),
                _make_result("B", MethodName.OLS, p_value=0.01),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.0001),
                _make_result("B", MethodName.ROAST_MSQ, p_value=0.01),
            ],
        }

        # At alpha=0.05, both A and B are significant in 2 methods
        ranks_05 = compute_concordance_rank(results_by_method, alpha=0.05)
        # Both significant in 2 methods -> A ranks higher by geomean
        assert ranks_05["A"] < ranks_05["B"]

        # At alpha=0.001, only A is significant in 2 methods, B in 0
        ranks_001 = compute_concordance_rank(results_by_method, alpha=0.001)
        assert ranks_001["A"] == 1
        assert ranks_001["B"] == 2

    def test_alpha_boundary(self):
        """Gene set with p-value exactly at alpha is NOT significant."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.05),  # not < 0.05
                _make_result("B", MethodName.OLS, p_value=0.049),  # < 0.05
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)
        # B is significant (p < 0.05), A is not (p == 0.05)
        assert ranks["B"] < ranks["A"]


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestConcordanceRankEdgeCases:
    """Edge cases: empty inputs, invalid results, extreme p-values."""

    def test_empty_input_raises(self):
        """Empty results_by_method raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_concordance_rank({})

    def test_all_invalid_returns_empty(self):
        """All results invalid (NaN p-values) returns empty dict."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=float("nan"),
                             effect_size=float("nan")),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)
        assert ranks == {}

    def test_mixed_valid_invalid(self):
        """Invalid results are excluded; only valid results contribute."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.01),
                _make_result("B", MethodName.OLS, p_value=float("nan"),
                             effect_size=float("nan")),
            ],
            MethodName.ROAST_MSQ: [
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.02),
                _make_result("B", MethodName.ROAST_MSQ, p_value=0.03),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A: sig in 2 methods; B: sig in 1 (OLS result invalid, only ROAST counts)
        assert ranks["A"] < ranks["B"]
        assert "A" in ranks
        assert "B" in ranks

    def test_very_small_pvalues(self):
        """Extremely small p-values (near machine epsilon) handled correctly."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=1e-300),
                _make_result("B", MethodName.OLS, p_value=1e-10),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)
        # Both significant, but A has smaller p -> rank 1
        assert ranks["A"] == 1
        assert ranks["B"] == 2

    def test_zero_pvalue(self):
        """p-value of exactly 0.0 does not cause log(0) crash."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.0),
                _make_result("B", MethodName.OLS, p_value=0.01),
            ],
        }

        # Should not raise
        ranks = compute_concordance_rank(results_by_method, alpha=0.05)
        assert ranks["A"] == 1
        assert ranks["B"] == 2

    def test_partial_method_coverage(self):
        """Gene sets not tested by all methods are still ranked."""
        results_by_method = {
            MethodName.OLS: [
                _make_result("A", MethodName.OLS, p_value=0.001),
                _make_result("B", MethodName.OLS, p_value=0.001),
            ],
            MethodName.ROAST_MSQ: [
                # Only A tested by ROAST
                _make_result("A", MethodName.ROAST_MSQ, p_value=0.001),
            ],
        }

        ranks = compute_concordance_rank(results_by_method, alpha=0.05)

        # A: sig in 2 methods, B: sig in 1 method
        assert ranks["A"] < ranks["B"]


# ---------------------------------------------------------------------------
# Integration with MethodComparisonResult (import chain)
# ---------------------------------------------------------------------------


class TestConcordanceRankImportChain:
    """Verify the function is accessible via all re-export paths."""

    def test_import_from_concordance(self):
        from cliquefinder.stats.concordance import compute_concordance_rank as fn
        assert callable(fn)

    def test_import_from_method_comparison(self):
        from cliquefinder.stats.method_comparison import compute_concordance_rank as fn
        assert callable(fn)

    def test_import_from_stats_init(self):
        from cliquefinder.stats import compute_concordance_rank as fn
        assert callable(fn)
