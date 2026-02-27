"""Tests for shared competitive enrichment z-score computation."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.enrichment_z import (
    compute_competitive_z,
    compute_robust_competitive_z,
)


class TestComputeCompetitiveZ:
    """Tests for compute_competitive_z()."""

    def test_basic_computation(self):
        """Known values produce expected z-score (SE denominator)."""
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
        k = 5
        # STAT-3 fix: denominator is SE = bg_std / sqrt(k)
        expected_z = (target_mean - bg_mean) / (bg_std / np.sqrt(k))

        assert z == pytest.approx(expected_z)
        assert z > 2.0  # strong signal (even stronger now with SE)

    def test_no_targets_returns_zero(self):
        """No targets -> z = 0.0."""
        t_stats = np.random.default_rng(42).normal(0, 1, size=50)
        is_target = np.zeros(50, dtype=bool)
        assert compute_competitive_z(t_stats, is_target) == 0.0

    def test_all_targets_returns_zero(self):
        """All features are targets -> z = 0.0 (no background)."""
        t_stats = np.random.default_rng(42).normal(0, 1, size=50)
        is_target = np.ones(50, dtype=bool)
        assert compute_competitive_z(t_stats, is_target) == 0.0

    def test_zero_variance_background(self):
        """Constant background -> z = 0.0 (division by zero guard)."""
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


class TestRobustCompetitiveZ:
    """Tests for robust (median+MAD) competitive z-score."""

    def test_robust_z_basic(self):
        """Known values produce expected median+MAD z-score (SE denominator)."""
        # Construct deterministic data so we can hand-compute the answer.
        # 5 targets with |t| = [3.0, 3.5, 2.8, 4.0, 3.2]
        # 10 background with |t| = [1.0]*10 except one = 1.5
        t_stats = np.array(
            [3.0, -3.5, 2.8, -4.0, 3.2,     # targets
             1.0, 1.0, 1.0, 1.0, 1.0,        # background
             1.0, 1.0, 1.0, 1.0, 1.5],       # background
            dtype=np.float64,
        )
        is_target = np.zeros(len(t_stats), dtype=bool)
        is_target[:5] = True

        z = compute_competitive_z(t_stats, is_target, robust=True)

        # Manual calculation:
        # target |t|: [3.0, 3.5, 2.8, 4.0, 3.2] -> median = 3.2
        # bg |t|: [1.0]*9 + [1.5] -> median = 1.0
        # deviations from bg median: [0]*9 + [0.5] -> median deviation = 0.0
        # MAD = 1.4826 * 0.0 = 0.0 -> zero guard -> 0.0
        # Actually with 9 ones and 1 value of 1.5, median of deviations is 0.
        # So this should return 0.0.
        assert z == 0.0

        # Now use a background with actual spread.
        bg = np.array([0.5, 1.0, 1.2, 1.5, 2.0, 0.8, 1.1, 1.3, 1.7, 0.9])
        t_stats2 = np.concatenate([np.array([3.0, -3.5, 2.8, -4.0, 3.2]), bg])
        is_target2 = np.zeros(len(t_stats2), dtype=bool)
        is_target2[:5] = True

        z2 = compute_competitive_z(t_stats2, is_target2, robust=True)

        # Manual calculation:
        # target |t|: [3.0, 3.5, 2.8, 4.0, 3.2] -> median = 3.2
        # bg |t| = bg (all positive): sorted = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
        # bg median = (1.1 + 1.2) / 2 = 1.15
        # deviations: |bg - 1.15| = [0.65, 0.35, 0.25, 0.15, 0.05, 0.05, 0.15, 0.35, 0.55, 0.85]
        # sorted devs: [0.05, 0.05, 0.15, 0.15, 0.25, 0.35, 0.35, 0.55, 0.65, 0.85]
        # median dev = (0.25 + 0.35) / 2 = 0.30
        # MAD = 1.4826 * 0.30 = 0.44478
        # STAT-3 fix: SE = MAD / sqrt(k) = 0.44478 / sqrt(5) = 0.198918
        # z = (3.2 - 1.15) / 0.198918 = 2.05 / 0.198918 ~ 10.306
        k = 5
        expected = (3.2 - 1.15) / (1.4826 * 0.30 / np.sqrt(k))
        assert z2 == pytest.approx(expected, rel=1e-6)
        assert z2 > 4.0

    def test_robust_z_outlier_resistant(self):
        """Robust z is less affected by extreme outliers than standard z."""
        rng = np.random.default_rng(42)
        t_stats = rng.normal(0, 1, size=200)
        # Strong target signal
        t_stats[:10] = rng.normal(3, 0.3, size=10)
        is_target = np.zeros(200, dtype=bool)
        is_target[:10] = True

        z_standard_clean = compute_competitive_z(t_stats, is_target, robust=False)
        z_robust_clean = compute_competitive_z(t_stats, is_target, robust=True)

        # Inject extreme outlier in background
        t_stats_outlier = t_stats.copy()
        t_stats_outlier[50] = 100.0  # massive outlier in background

        z_standard_outlier = compute_competitive_z(t_stats_outlier, is_target, robust=False)
        z_robust_outlier = compute_competitive_z(t_stats_outlier, is_target, robust=True)

        # Standard z should be heavily affected by the outlier
        standard_change = abs(z_standard_clean - z_standard_outlier)
        robust_change = abs(z_robust_clean - z_robust_outlier)

        # Robust z should change much less than standard z
        assert robust_change < standard_change
        # The robust change should be small (median/MAD barely move)
        assert robust_change < 0.5 * standard_change

    def test_robust_z_matches_standard_for_normal(self):
        """For normal data, robust and standard z agree in sign and rough magnitude."""
        rng = np.random.default_rng(123)
        # Large sample from normal distribution -- median ~ mean, MAD ~ std
        t_stats = rng.normal(0, 1, size=5000)
        t_stats[:50] = rng.normal(2, 0.5, size=50)  # moderate target signal
        is_target = np.zeros(5000, dtype=bool)
        is_target[:50] = True

        z_standard = compute_competitive_z(t_stats, is_target, robust=False)
        z_robust = compute_competitive_z(t_stats, is_target, robust=True)

        # Same sign
        assert np.sign(z_standard) == np.sign(z_robust)
        # Similar magnitude (within factor of 2 for large normal samples)
        ratio = z_robust / z_standard
        assert 0.5 < ratio < 2.0

    def test_robust_z_zero_mad(self):
        """Zero MAD returns 0.0 (same guard as zero-variance in standard mode)."""
        # Background all identical -> MAD = 0
        t_stats = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        is_target = np.array([True, True, False, False, False, False, False, False, False, False])
        z = compute_competitive_z(t_stats, is_target, robust=True)
        assert z == 0.0

    def test_convenience_function(self):
        """compute_robust_competitive_z() delegates to robust=True."""
        rng = np.random.default_rng(99)
        t_stats = rng.normal(0, 1, size=100)
        t_stats[:5] = [3.0, -3.5, 2.8, -4.0, 3.2]
        is_target = np.zeros(100, dtype=bool)
        is_target[:5] = True

        z_direct = compute_competitive_z(t_stats, is_target, robust=True)
        z_convenience = compute_robust_competitive_z(t_stats, is_target)
        assert z_direct == z_convenience


class TestNetworkEnrichmentResult:
    """Tests for NetworkEnrichmentResult dataclass (H7 audit finding)."""

    def test_network_enrichment_result_fields(self):
        """Verify dataclass has all expected fields with correct types."""
        from cliquefinder.stats.differential import NetworkEnrichmentResult

        result = NetworkEnrichmentResult(
            observed_mean_abs_t=2.5,
            null_mean=1.0,
            null_std=0.3,
            z_score=5.0,
            empirical_pvalue=0.001,
            n_targets=50,
            n_background=500,
            pct_down=60.0,
            direction_pvalue=0.1,
            mannwhitney_pvalue=0.005,
        )

        # Verify all expected attributes exist and are accessible
        assert result.observed_mean_abs_t == 2.5
        assert result.null_mean == 1.0
        assert result.null_std == 0.3
        assert result.z_score == 5.0
        assert result.empirical_pvalue == 0.001
        assert result.n_targets == 50
        assert result.n_background == 500
        assert result.pct_down == 60.0
        assert result.direction_pvalue == 0.1
        assert result.mannwhitney_pvalue == 0.005
        # New VIF fields have defaults
        assert result.variance_inflation_factor == 1.0
        assert result.mean_pairwise_correlation == 0.0

        # Verify frozen (immutable)
        with pytest.raises(AttributeError):
            result.z_score = 10.0

        # Verify dict-style access (backward compatibility)
        assert result["z_score"] == 5.0
        assert result.get("z_score") == 5.0
        assert result.get("nonexistent", 42) == 42

    def test_network_enrichment_result_vif_fields(self):
        """Verify VIF fields can be set explicitly."""
        from cliquefinder.stats.differential import NetworkEnrichmentResult

        result = NetworkEnrichmentResult(
            observed_mean_abs_t=2.5,
            null_mean=1.0,
            null_std=0.3,
            z_score=5.0,
            empirical_pvalue=0.001,
            n_targets=50,
            n_background=500,
            pct_down=60.0,
            direction_pvalue=0.1,
            mannwhitney_pvalue=0.005,
            variance_inflation_factor=3.5,
            mean_pairwise_correlation=0.05,
        )

        assert result.variance_inflation_factor == 3.5
        assert result.mean_pairwise_correlation == 0.05
        assert result["variance_inflation_factor"] == 3.5
        assert result["mean_pairwise_correlation"] == 0.05

    def test_network_enrichment_to_dict(self):
        """Verify to_dict() produces correct dict and roundtrips."""
        from cliquefinder.stats.differential import NetworkEnrichmentResult

        result = NetworkEnrichmentResult(
            observed_mean_abs_t=2.5,
            null_mean=1.0,
            null_std=0.3,
            z_score=5.0,
            empirical_pvalue=0.001,
            n_targets=50,
            n_background=500,
            pct_down=60.0,
            direction_pvalue=0.1,
            mannwhitney_pvalue=0.005,
        )

        d = result.to_dict()

        # Verify dict has all expected keys (including new VIF fields)
        expected_keys = {
            'observed_mean_abs_t', 'null_mean', 'null_std', 'z_score',
            'empirical_pvalue', 'n_targets', 'n_background', 'pct_down',
            'direction_pvalue', 'mannwhitney_pvalue',
            'variance_inflation_factor', 'mean_pairwise_correlation',
        }
        assert set(d.keys()) == expected_keys

        # Verify values match
        assert d['observed_mean_abs_t'] == 2.5
        assert d['null_mean'] == 1.0
        assert d['null_std'] == 0.3
        assert d['z_score'] == 5.0
        assert d['empirical_pvalue'] == 0.001
        assert d['n_targets'] == 50
        assert d['n_background'] == 500
        assert d['pct_down'] == 60.0
        assert d['direction_pvalue'] == 0.1
        assert d['mannwhitney_pvalue'] == 0.005
        assert d['variance_inflation_factor'] == 1.0
        assert d['mean_pairwise_correlation'] == 0.0

        # Verify JSON-serializable
        import json
        json_str = json.dumps(d)
        roundtripped = json.loads(json_str)
        assert roundtripped == d

        # Verify ** unpacking works (used in validate_baselines.py)
        merged = {**d, "extra_key": "value"}
        assert merged["z_score"] == 5.0
        assert merged["extra_key"] == "value"
