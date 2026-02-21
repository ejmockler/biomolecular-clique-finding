"""Tests for bootstrap stability analysis."""

import numpy as np
import pytest

from cliquefinder.stats.bootstrap_stability import run_bootstrap_stability


def _make_data(n_features=100, n_samples=30, n_targets=10, effect=0.0, seed=42):
    """Build synthetic expression data with optional target effect."""
    rng = np.random.default_rng(seed)
    n_a = n_samples // 2
    n_b = n_samples - n_a
    data = rng.normal(0, 1, size=(n_features, n_samples))
    # Inject effect into targets in group A
    for i in range(n_targets):
        data[i, :n_a] += effect
    feature_ids = [f"gene_{i}" for i in range(n_features)]
    target_gene_ids = feature_ids[:n_targets]
    condition = np.array(["A"] * n_a + ["B"] * n_b)
    contrast = ("A", "B")
    return data, feature_ids, condition, contrast, target_gene_ids


class TestRunBootstrapStability:
    """Tests for run_bootstrap_stability()."""

    def test_signal_data_high_stability(self):
        """Strong signal → stability > 0.7."""
        data, fids, cond, contrast, targets = _make_data(effect=3.0, seed=42)

        result = run_bootstrap_stability(
            data=data,
            feature_ids=fids,
            sample_condition=cond,
            contrast=contrast,
            target_gene_ids=targets,
            n_bootstraps=50,
            seed=42,
            verbose=False,
        )

        assert result["stability"] > 0.7
        assert result["n_bootstraps"] > 0
        assert len(result["z_scores"]) == result["n_bootstraps"]

    def test_null_data_low_stability(self):
        """No signal → stability < 0.3."""
        data, fids, cond, contrast, targets = _make_data(effect=0.0, seed=42)

        result = run_bootstrap_stability(
            data=data,
            feature_ids=fids,
            sample_condition=cond,
            contrast=contrast,
            target_gene_ids=targets,
            n_bootstraps=50,
            z_threshold=1.5,
            seed=42,
            verbose=False,
        )

        assert result["stability"] < 0.3

    def test_ci_bounds(self):
        """Z-score CI should have lower < upper."""
        data, fids, cond, contrast, targets = _make_data(effect=2.0, seed=42)

        result = run_bootstrap_stability(
            data=data,
            feature_ids=fids,
            sample_condition=cond,
            contrast=contrast,
            target_gene_ids=targets,
            n_bootstraps=50,
            seed=42,
            verbose=False,
        )

        lo, hi = result["z_ci"]
        assert lo <= hi

    def test_with_covariates(self):
        """Bootstrap stability works with covariates."""
        data, fids, cond, contrast, targets = _make_data(
            effect=3.0, n_samples=40, seed=42,
        )
        import pandas as pd
        rng = np.random.default_rng(42)
        cov = pd.DataFrame({"Sex": rng.choice(["M", "F"], size=len(cond))})

        result = run_bootstrap_stability(
            data=data,
            feature_ids=fids,
            sample_condition=cond,
            contrast=contrast,
            target_gene_ids=targets,
            covariates_df=cov,
            n_bootstraps=30,
            seed=42,
            verbose=False,
        )

        assert 0.0 <= result["stability"] <= 1.0
        assert result["n_bootstraps"] > 0


class TestBootstrapReportAnnotation:
    """Tests for bootstrap stability integration with ValidationReport."""

    def test_low_stability_annotation(self):
        """Low bootstrap stability annotates summary."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.bootstrap_stability = 0.5
        report.bootstrap_ci = (0.3, 1.2)
        report.compute_verdict()

        assert "low bootstrap stability" in report.summary

    def test_high_stability_no_annotation(self):
        """High bootstrap stability does not annotate summary."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {"empirical_pvalue": 0.01})
        report.add_phase("label_permutation", {"permutation_pvalue": 0.01})
        report.bootstrap_stability = 0.85
        report.bootstrap_ci = (1.0, 3.5)
        report.compute_verdict()

        assert "bootstrap stability" not in report.summary

    def test_serialization_with_bootstrap(self):
        """to_dict() includes bootstrap fields when populated."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.bootstrap_stability = 0.62
        report.bootstrap_ci = (0.8, 2.1)

        d = report.to_dict()
        assert d["bootstrap_stability"] == pytest.approx(0.62)
        assert d["bootstrap_ci"] == pytest.approx([0.8, 2.1])

    def test_serialization_without_bootstrap(self):
        """to_dict() omits bootstrap fields when None."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        d = report.to_dict()
        assert "bootstrap_stability" not in d
