"""Tests for competitive z-score enrichment test.

Tests cover:
- Basic competitive z computation with known synthetic data
- VIF correction inflates SE and reduces z-score
- VIF=1 equivalence with no correlation matrix
- Integration with DiscoveryBridge.test_gene_set(use_competitive=True)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cliquefinder.stats.competitive_z import competitive_z_test, _mean_off_diagonal


class TestCompetitiveZBasic:
    """Test competitive_z_test on synthetic data."""

    def test_target_higher_than_background(self):
        """Target set with higher |t| should produce positive z and low p."""
        rng = np.random.default_rng(42)
        n_genes = 200
        # Background: |t| ~ N(1, 0.5)
        t_stats = rng.normal(0, 1.0, size=n_genes)
        # Targets: inflate |t| significantly
        target_indices = np.arange(10)
        t_stats[target_indices] = rng.normal(0, 3.0, size=10)

        z, p = competitive_z_test(t_stats, target_indices)

        assert z > 0, "Targets with higher |t| should have positive z"
        assert p < 0.05, f"Enriched set should be significant, got p={p}"

    def test_target_same_as_background(self):
        """Target set drawn from same distribution should not be significant."""
        rng = np.random.default_rng(123)
        n_genes = 1000
        t_stats = rng.normal(0, 1.0, size=n_genes)
        target_indices = np.arange(30)

        z, p = competitive_z_test(t_stats, target_indices)

        # With 1000 genes from same distribution, z should be near 0
        assert abs(z) < 3.0, f"Same-distribution set should have small z, got {z}"
        # p should not be extremely small
        assert p > 0.001

    def test_empty_target_set(self):
        """Empty target set returns degenerate (0, 1)."""
        t_stats = np.array([1.0, 2.0, 3.0])
        z, p = competitive_z_test(t_stats, np.array([], dtype=np.intp))
        assert z == 0.0
        assert p == 1.0

    def test_all_genes_in_target(self):
        """Target covering all genes returns degenerate (0, 1)."""
        t_stats = np.array([1.0, 2.0, 3.0])
        z, p = competitive_z_test(t_stats, np.arange(3))
        assert z == 0.0
        assert p == 1.0

    def test_two_sided_pvalue(self):
        """P-value is two-sided (both tails)."""
        # Create a set with LOWER |t| than background
        rng = np.random.default_rng(99)
        n_genes = 200
        # Background has high |t| with some variance
        t_stats = rng.normal(5.0, 1.0, size=n_genes)
        target_indices = np.arange(10)
        # Targets have low |t|
        t_stats[target_indices] = rng.normal(0.0, 0.3, size=10)

        z, p = competitive_z_test(t_stats, target_indices)
        assert z < 0, "Targets with lower |t| should have negative z"
        # Two-sided p should still be small for this large difference
        assert p < 0.05


class TestCompetitiveZVIF:
    """Test VIF correction from inter-gene correlation."""

    def test_vif_inflates_se_reduces_z(self):
        """Positive correlation (VIF > 1) should reduce |z| vs VIF=1."""
        rng = np.random.default_rng(42)
        n_genes = 200
        t_stats = rng.normal(0, 1.0, size=n_genes)
        target_indices = np.arange(10)
        t_stats[target_indices] = rng.normal(0, 3.0, size=10)

        # No VIF
        z_no_vif, p_no_vif = competitive_z_test(t_stats, target_indices)

        # With positive correlation -> VIF > 1
        k = len(target_indices)
        corr_matrix = np.full((k, k), 0.3)  # rho_bar = 0.3
        np.fill_diagonal(corr_matrix, 1.0)

        z_with_vif, p_with_vif = competitive_z_test(
            t_stats, target_indices, correlation_matrix=corr_matrix
        )

        assert abs(z_with_vif) < abs(z_no_vif), (
            f"VIF correction should reduce |z|: "
            f"|z_vif|={abs(z_with_vif):.3f} vs |z_no|={abs(z_no_vif):.3f}"
        )
        assert p_with_vif > p_no_vif, (
            f"VIF correction should increase p-value: "
            f"p_vif={p_with_vif:.4f} vs p_no={p_no_vif:.4f}"
        )

    def test_vif_1_equals_no_correction(self):
        """VIF=1 (identity correlation) gives same result as no matrix."""
        rng = np.random.default_rng(42)
        n_genes = 200
        t_stats = rng.normal(0, 1.0, size=n_genes)
        target_indices = np.arange(10)
        t_stats[target_indices] = rng.normal(0, 3.0, size=10)

        z_none, p_none = competitive_z_test(t_stats, target_indices)

        # Identity matrix -> rho_bar = 0 -> VIF = 1
        k = len(target_indices)
        corr_identity = np.eye(k)
        z_identity, p_identity = competitive_z_test(
            t_stats, target_indices, correlation_matrix=corr_identity
        )

        assert abs(z_none - z_identity) < 1e-10
        assert abs(p_none - p_identity) < 1e-10

    def test_negative_correlation_floored(self):
        """Negative mean correlation is floored at 0 (conservative)."""
        rng = np.random.default_rng(42)
        n_genes = 200
        t_stats = rng.normal(0, 1.0, size=n_genes)
        target_indices = np.arange(10)
        t_stats[target_indices] = rng.normal(0, 3.0, size=10)

        z_none, p_none = competitive_z_test(t_stats, target_indices)

        # Negative correlation -> rho_bar floored at 0 -> VIF = 1
        k = len(target_indices)
        corr_neg = np.full((k, k), -0.05)
        np.fill_diagonal(corr_neg, 1.0)
        z_neg, p_neg = competitive_z_test(
            t_stats, target_indices, correlation_matrix=corr_neg
        )

        assert abs(z_none - z_neg) < 1e-10
        assert abs(p_none - p_neg) < 1e-10


class TestMeanOffDiagonal:
    """Test _mean_off_diagonal helper."""

    def test_identity_matrix(self):
        """Identity matrix has zero off-diagonal mean."""
        assert _mean_off_diagonal(np.eye(5)) == 0.0

    def test_uniform_correlation(self):
        """Uniform off-diagonal correlation."""
        k = 5
        mat = np.full((k, k), 0.4)
        np.fill_diagonal(mat, 1.0)
        rho = _mean_off_diagonal(mat)
        assert abs(rho - 0.4) < 1e-10

    def test_handles_nan(self):
        """NaN entries are excluded from mean."""
        mat = np.array([
            [1.0, 0.3, np.nan],
            [0.3, 1.0, 0.5],
            [np.nan, 0.5, 1.0],
        ])
        rho = _mean_off_diagonal(mat)
        # Valid off-diagonal: 0.3, 0.3, 0.5, 0.5 -> mean = 0.4
        assert abs(rho - 0.4) < 1e-10

    def test_single_gene(self):
        """Single gene returns 0."""
        assert _mean_off_diagonal(np.array([[1.0]])) == 0.0


class TestDiscoveryBridgeCompetitive:
    """Test DiscoveryBridge with use_competitive=True."""

    def _make_bridge(self, use_competitive=False):
        """Create a bridge with a mock engine that has realistic effects."""
        from cliquefinder.stats.discovery_bridge import DiscoveryBridge

        engine = MagicMock()
        n_genes = 100
        engine.gene_to_idx = {f"P{i:05d}": i for i in range(n_genes)}
        engine.gene_ids = [f"P{i:05d}" for i in range(n_genes)]

        # Mock _effects with realistic data
        rng = np.random.default_rng(42)
        effects = MagicMock()
        # U[:, 0] are the contrast effects
        U = rng.normal(0, 1.0, size=(n_genes, 5))
        # Make first 10 genes have large effects (our targets)
        U[:10, 0] = rng.normal(3.0, 0.5, size=10)
        effects.U = U
        effects.moderated_variances = np.full(n_genes, 0.5)
        effects.sample_variances = np.full(n_genes, 0.6)
        engine._effects = effects
        engine._fitted = True

        # Mock data for correlation estimation
        engine.data = rng.normal(0, 1, size=(n_genes, 20))

        bridge = DiscoveryBridge(
            engine, {}, use_competitive=use_competitive
        )
        return bridge

    def test_competitive_returns_pvalue(self):
        """use_competitive=True returns a valid p-value."""
        bridge = self._make_bridge(use_competitive=True)
        target_fids = [f"P{i:05d}" for i in range(10)]

        p = bridge.test_gene_set(target_fids, "test_set")

        assert 0.0 <= p <= 1.0
        # Targets have inflated effects -> should be significant
        assert p < 0.05

    def test_competitive_vs_roast_different(self):
        """Competitive p-value differs from ROAST p-value."""
        bridge = self._make_bridge(use_competitive=False)
        target_fids = [f"P{i:05d}" for i in range(10)]

        # Mock ROAST result
        mock_result = MagicMock()
        mock_result.p_values = {"msq": {"mixed": 0.03}}
        bridge.engine.test_gene_set.return_value = mock_result

        p_roast = bridge.test_gene_set(target_fids, "test_set")
        p_comp = bridge.test_gene_set(
            target_fids, "test_set", use_competitive=True
        )

        assert p_roast == 0.03  # From mock
        assert p_comp != p_roast  # Competitive computes differently

    def test_per_call_override(self):
        """use_competitive parameter overrides instance setting."""
        bridge = self._make_bridge(use_competitive=True)
        target_fids = [f"P{i:05d}" for i in range(10)]

        # Override to False -> should call engine.test_gene_set (ROAST)
        mock_result = MagicMock()
        mock_result.p_values = {"msq": {"mixed": 0.07}}
        bridge.engine.test_gene_set.return_value = mock_result

        p = bridge.test_gene_set(target_fids, "test_set", use_competitive=False)
        assert p == 0.07

    def test_insufficient_genes_returns_1(self):
        """Fewer than 2 valid genes returns p=1.0 regardless of mode."""
        bridge = self._make_bridge(use_competitive=True)
        p = bridge.test_gene_set(["P00000", "UNKNOWN"], "test_set")
        assert p == 1.0

    def test_constructor_flag_stored(self):
        """use_competitive flag is accessible on the instance."""
        from cliquefinder.stats.discovery_bridge import DiscoveryBridge

        engine = MagicMock()
        engine.gene_to_idx = {}
        bridge = DiscoveryBridge(engine, {}, use_competitive=True)
        assert bridge.use_competitive is True

        bridge2 = DiscoveryBridge(engine, {}, use_competitive=False)
        assert bridge2.use_competitive is False

        # Default is False
        bridge3 = DiscoveryBridge(engine, {})
        assert bridge3.use_competitive is False
