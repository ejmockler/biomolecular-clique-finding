"""Tests for shared competitive enrichment z-score computation."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.enrichment_z import compute_competitive_z


class TestComputeCompetitiveZ:
    """Tests for compute_competitive_z()."""

    def test_basic_computation(self):
        """Known values produce expected z-score."""
        # 5 targets with high |t|, 95 background with lower |t|
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:5] = [3.0, -3.5, 2.8, -4.0, 3.2]  # targets: high |t|
        is_target = np.zeros(100, dtype=bool)
        is_target[:5] = True

        z = compute_competitive_z(t_stats, is_target)

        # Target mean |t| should be much higher than background
        target_mean = np.mean(np.abs(t_stats[:5]))
        bg_mean = np.mean(np.abs(t_stats[5:]))
        bg_std = np.std(np.abs(t_stats[5:]), ddof=1)
        expected_z = (target_mean - bg_mean) / bg_std

        assert z == pytest.approx(expected_z)
        assert z > 2.0  # strong signal

    def test_no_targets_returns_zero(self):
        """No targets → z = 0.0."""
        t_stats = np.random.default_rng(42).normal(0, 1, size=50)
        is_target = np.zeros(50, dtype=bool)
        assert compute_competitive_z(t_stats, is_target) == 0.0

    def test_all_targets_returns_zero(self):
        """All features are targets → z = 0.0 (no background)."""
        t_stats = np.random.default_rng(42).normal(0, 1, size=50)
        is_target = np.ones(50, dtype=bool)
        assert compute_competitive_z(t_stats, is_target) == 0.0

    def test_zero_variance_background(self):
        """Constant background → z = 0.0 (division by zero guard)."""
        t_stats = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        is_target = np.array([True, True, False, False, False, False, False, False])
        # Background |t| = [1, 1, 1, 1, 1, 1], std = 0
        assert compute_competitive_z(t_stats, is_target) == 0.0

    def test_matches_extract_enrichment_z(self):
        """compute_competitive_z matches _extract_enrichment_z on DataFrame."""
        from cliquefinder.stats.label_permutation import _extract_enrichment_z

        rng = np.random.default_rng(42)
        n = 100
        t_stats = rng.normal(0, 1, size=n)
        t_stats[:10] = rng.normal(2, 0.5, size=10)
        is_target = np.zeros(n, dtype=bool)
        is_target[:10] = True

        # Build DataFrame as run_protein_differential would
        df = pd.DataFrame({
            "feature_id": [f"gene_{i}" for i in range(n)],
            "t_statistic": t_stats,
            "is_target": is_target,
        })

        z_array = compute_competitive_z(t_stats, is_target)
        z_df = _extract_enrichment_z(df)

        assert z_array == pytest.approx(z_df)

    def test_negative_z_when_targets_lower(self):
        """Targets with lower |t| than background produce negative z."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 2, size=100)  # background with high |t|
        t_stats[:5] = [0.1, -0.1, 0.05, -0.05, 0.0]  # targets: near zero
        is_target = np.zeros(100, dtype=bool)
        is_target[:5] = True

        z = compute_competitive_z(t_stats, is_target)
        assert z < 0  # targets are less extreme than background
