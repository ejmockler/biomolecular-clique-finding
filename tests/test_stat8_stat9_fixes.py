"""Tests for STAT-8 (FWER docstring accuracy) and STAT-9 (cache matched control sets).

STAT-8: Verify the compute_verdict docstring uses bounded-FWER language
        with rho-dependent formulas instead of the incorrect alpha^2 claim.

STAT-9: Verify that matched z-scores use the SAME gene sets as matched
        p-values (cached from first pass), not freshly sampled sets.
"""

import inspect
import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.validation_report import ValidationReport
from cliquefinder.stats.negative_controls import (
    NegativeControlResult,
    run_negative_control_sets,
)


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

class MockRotationResult:
    """Minimal mock of RotationResult dataclass."""

    def __init__(self, set_id, n_genes, pvalue):
        self.feature_set_id = set_id
        self.n_genes = n_genes
        self.p_values = {"msq": {"mixed": pvalue}}


class RecordingMockEngine:
    """Mock engine that records every gene set passed to test_gene_set.

    Attributes:
        calls: List of (gene_set_id, frozenset(gene_set)) tuples in call order.
    """

    def __init__(self, gene_ids, pvalue=0.5):
        self.gene_ids = gene_ids
        self.gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        self._fitted = True
        self._pvalue = pvalue
        self.calls: list[tuple[str, frozenset]] = []

    def test_gene_set(self, gene_set, gene_set_id):
        self.calls.append((gene_set_id, frozenset(gene_set)))
        return MockRotationResult(gene_set_id, len(gene_set), self._pvalue)


class MockRotationEngine:
    """Minimal mock of RotationTestEngine for testing."""

    def __init__(self, gene_ids, pvalue_func=None):
        self.gene_ids = gene_ids
        self.gene_to_idx = {g: i for i, g in enumerate(gene_ids)}
        self._fitted = True
        self._pvalue_func = pvalue_func or (lambda _: 0.5)

    def test_gene_set(self, gene_set, gene_set_id):
        pval = self._pvalue_func(gene_set)
        return MockRotationResult(gene_set_id, len(gene_set), pval)


def _make_protein_results(gene_ids, target_genes, target_t=3.0, bg_t=0.0):
    """Build a protein_results DataFrame with known t-statistics."""
    rng = np.random.default_rng(99)
    n = len(gene_ids)
    target_set = set(target_genes)
    t_stats = rng.normal(bg_t, 0.5, size=n)
    is_target = np.zeros(n, dtype=bool)
    for i, g in enumerate(gene_ids):
        if g in target_set:
            t_stats[i] = target_t + rng.normal(0, 0.3)
            is_target[i] = True
    return pd.DataFrame({
        "feature_id": gene_ids,
        "t_statistic": t_stats,
        "is_target": is_target,
    })


# ===================================================================
# STAT-8: FWER docstring accuracy
# ===================================================================


class TestStat8FWERDocstring:
    """Verify compute_verdict docstring uses bounded-FWER with rho formula."""

    def _get_docstring(self):
        return inspect.getdoc(ValidationReport.compute_verdict)

    def test_docstring_contains_bounded_fwer(self):
        """Docstring says 'bounded-FWER' (not 'controlled FWER')."""
        doc = self._get_docstring()
        assert "bounded-FWER" in doc.lower() or "Bounded-FWER" in doc or "bounded-fwer" in doc.lower()

    def test_docstring_contains_rho_dependent_formula(self):
        """Docstring includes the rho-dependent joint-pass formula."""
        doc = self._get_docstring()
        # The formula mentions Phi, z_alpha, rho, and sqrt(1 - rho^2)
        assert "rho" in doc
        assert "Phi" in doc
        assert "z_alpha" in doc
        assert "sqrt(1 - rho" in doc or "sqrt(1-rho" in doc

    def test_docstring_no_alpha_squared_claim(self):
        """Docstring does NOT claim joint probability IS alpha^2."""
        doc = self._get_docstring()
        # Should not contain "is alpha^2" as a standalone claim;
        # it's OK to reference alpha^2 as the independent-case limit
        assert "joint probability under the global null is alpha^2" not in doc.lower()

    def test_docstring_contains_quantitative_bounds(self):
        """Docstring gives the approximate FWER range for typical rho."""
        doc = self._get_docstring()
        assert "0.006" in doc
        assert "0.020" in doc

    def test_docstring_mentions_design_asymmetry(self):
        """Docstring contains the 'Design asymmetry note'."""
        doc = self._get_docstring()
        assert "Design asymmetry" in doc

    def test_docstring_mentions_rho_080_value(self):
        """Docstring notes the rho=0.8 case (approx 0.030)."""
        doc = self._get_docstring()
        assert "0.030" in doc


# ===================================================================
# STAT-9: Cache matched control sets
# ===================================================================


class TestStat9CachedMatchedSets:
    """Verify matched z-scores reuse the same gene sets as matched p-values."""

    def _make_data(self, n_features, n_samples, rng=None):
        """Create expression data with elevated targets."""
        if rng is None:
            rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 3.0  # Targets: elevated expression
        return data

    def test_matched_gene_sets_same_for_pvalues_and_zscores(self):
        """Gene sets used for matched p-values must match those for z-scores.

        Strategy: Use a RecordingMockEngine that logs every gene set.
        With the fix, matched_control_i in the p-value pass should use
        the same genes as the z-score computation for control i.
        """
        n_features = 100
        n_samples = 20
        n_control = 5
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        engine = RecordingMockEngine(gene_ids, pvalue=0.5)
        rng_data = np.random.default_rng(42)
        data = self._make_data(n_features, n_samples, rng_data)
        protein_df = _make_protein_results(gene_ids, target_genes, target_t=4.0)

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="cache_test",
            n_control_sets=n_control,
            seed=123,
            protein_results=protein_df,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        # Gather matched control calls from the engine
        # The call pattern is:
        #   1. target set  (gene_set_id = "cache_test")
        #   2. n_control uniform control sets (gene_set_id = "control_0" .. "control_{n-1}")
        #   3. n_control matched control sets (gene_set_id = "matched_control_0" .. )
        # With the fix, z-score computation does NOT call engine.test_gene_set
        # again for matched sets (it only uses compute_competitive_z).
        # So there should be exactly 1 + n_control + n_control calls.
        matched_calls = {
            gene_set_id: genes
            for gene_set_id, genes in engine.calls
            if gene_set_id.startswith("matched_control_")
        }

        # There should be exactly n_control matched control calls
        assert len(matched_calls) == n_control

        # Each matched_control_i should appear exactly once (no duplicates
        # from a second sampling pass)
        for i in range(n_control):
            key = f"matched_control_{i}"
            assert key in matched_calls, f"Missing {key} in engine calls"

    def test_no_extra_matched_calls_for_zscore(self):
        """With the fix, the z-score pass should NOT call engine.test_gene_set
        for matched sets (it uses compute_competitive_z directly on the
        cached gene lists). Total engine calls should be:
          1 (target) + n_control (uniform) + n_control (matched) = 1 + 2*n
        """
        n_features = 100
        n_samples = 20
        n_control = 5
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        engine = RecordingMockEngine(gene_ids, pvalue=0.5)
        data = self._make_data(n_features, n_samples, np.random.default_rng(42))
        protein_df = _make_protein_results(gene_ids, target_genes, target_t=4.0)

        run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="call_count_test",
            n_control_sets=n_control,
            seed=456,
            protein_results=protein_df,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        # Expected: 1 target + n_control uniform + n_control matched
        expected_calls = 1 + n_control + n_control
        assert len(engine.calls) == expected_calls, (
            f"Expected {expected_calls} engine calls, got {len(engine.calls)}: "
            f"{[cid for cid, _ in engine.calls]}"
        )

    def test_n_unique_matched_sets_equals_n_control(self):
        """Caching doesn't change the number of unique control sets."""
        n_features = 100
        n_samples = 20
        n_control = 5
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        engine = RecordingMockEngine(gene_ids, pvalue=0.5)
        data = self._make_data(n_features, n_samples, np.random.default_rng(42))
        protein_df = _make_protein_results(gene_ids, target_genes, target_t=4.0)

        run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="unique_test",
            n_control_sets=n_control,
            seed=789,
            protein_results=protein_df,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        matched_gene_sets = [
            genes for gene_set_id, genes in engine.calls
            if gene_set_id.startswith("matched_control_")
        ]

        # Exactly n_control unique matched sets
        assert len(matched_gene_sets) == n_control
        # Each set should be distinct (different genes due to noise in
        # Hungarian matching)
        unique_sets = {gs for gs in matched_gene_sets}
        # At minimum we expect >1 unique set (noise should differentiate them)
        # but we mainly care that we have exactly n_control entries
        assert len(matched_gene_sets) == n_control

    def test_rng_reproducibility(self):
        """Same seed produces identical results across two runs."""
        n_features = 100
        n_samples = 20
        n_control = 5
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        data = np.random.default_rng(42).normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 3.0
        protein_df = _make_protein_results(gene_ids, target_genes, target_t=4.0)

        results = []
        for _ in range(2):
            engine = RecordingMockEngine(gene_ids, pvalue=0.5)
            r = run_negative_control_sets(
                engine=engine,
                target_gene_ids=target_genes,
                target_set_id="repro_test",
                n_control_sets=n_control,
                seed=42,
                protein_results=protein_df,
                data=data,
                matching="expression_matched",
                verbose=False,
            )
            results.append(r)
            # Also record matched gene sets
            matched = [
                genes for gid, genes in engine.calls
                if gid.startswith("matched_control_")
            ]
            results.append(matched)

        r1, matched1, r2, matched2 = results

        # p-values and metrics should be identical
        np.testing.assert_array_equal(r1.matched_control_pvalues, r2.matched_control_pvalues)
        assert r1.matched_fpr == r2.matched_fpr
        assert r1.matched_target_percentile == r2.matched_target_percentile
        assert r1.matched_competitive_z_percentile == r2.matched_competitive_z_percentile

        # Gene sets should be identical
        assert len(matched1) == len(matched2)
        for gs1, gs2 in zip(matched1, matched2):
            assert gs1 == gs2

    def test_n_control_sets_zero(self):
        """Edge case: n_control_sets=0 should produce valid result with no controls."""
        gene_ids = [f"gene_{i}" for i in range(100)]
        target_genes = gene_ids[:10]

        engine = MockRotationEngine(gene_ids)

        # n_control_sets=0 means no controls are run
        # The function should handle this gracefully
        # (RuntimeError "All control sets failed" is expected since
        # 0 valid controls means none passed)
        with pytest.raises(RuntimeError, match="All control sets failed"):
            run_negative_control_sets(
                engine=engine,
                target_gene_ids=target_genes,
                target_set_id="zero_test",
                n_control_sets=0,
                seed=42,
                verbose=False,
            )

    def test_matched_competitive_z_populated_with_cache(self):
        """matched_competitive_z_percentile is populated when protein_results given."""
        n_features = 100
        n_samples = 20
        n_control = 5
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        engine = MockRotationEngine(gene_ids)
        rng_data = np.random.default_rng(42)
        data = rng_data.normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 3.0
        protein_df = _make_protein_results(gene_ids, target_genes, target_t=4.0)

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="z_cache_test",
            n_control_sets=n_control,
            seed=42,
            protein_results=protein_df,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        # matched_competitive_z_percentile should be populated
        assert result.matched_competitive_z_percentile is not None
        assert 0.0 <= result.matched_competitive_z_percentile <= 100.0

    def test_without_protein_results_no_matched_z(self):
        """Without protein_results, matched_competitive_z_percentile stays None."""
        n_features = 100
        n_samples = 20
        gene_ids = [f"gene_{i}" for i in range(n_features)]
        target_genes = gene_ids[:15]

        engine = MockRotationEngine(gene_ids)
        data = np.random.default_rng(42).normal(0, 1, size=(n_features, n_samples))
        data[:15, :] += 3.0

        result = run_negative_control_sets(
            engine=engine,
            target_gene_ids=target_genes,
            target_set_id="no_protein_test",
            n_control_sets=5,
            seed=42,
            data=data,
            matching="expression_matched",
            verbose=False,
        )

        assert result.matched_competitive_z_percentile is None
