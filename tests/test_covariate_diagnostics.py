"""Tests for covariate confounding diagnostics.

Verifies that the covariate_diagnostics module correctly:
- Detects categorical confounding (chi-squared)
- Detects continuous confounding (ANOVA / t-test)
- Handles NaN values in covariates
- Formats the confounding report table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.covariate_diagnostics import (
    CovariateReport,
    CovariateDiagnostic,
    assess_covariate_confounding,
    _is_categorical,
    _summarize_categorical,
    _summarize_continuous,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def confounded_categorical_data():
    """Synthetic data where sex is strongly associated with group.

    Group A is 90% Male, Group B is 90% Female.
    """
    rng = np.random.default_rng(42)
    n = 200
    group = np.array(["A"] * 100 + ["B"] * 100)
    # Group A: 90% M, Group B: 90% F
    sex_a = rng.choice(["M", "F"], size=100, p=[0.9, 0.1])
    sex_b = rng.choice(["M", "F"], size=100, p=[0.1, 0.9])
    sex = np.concatenate([sex_a, sex_b])

    return pd.DataFrame({
        "group": group,
        "Sex": sex,
    })


@pytest.fixture
def balanced_categorical_data():
    """Synthetic data where sex is balanced across groups."""
    rng = np.random.default_rng(42)
    n = 200
    group = np.array(["A"] * 100 + ["B"] * 100)
    sex = rng.choice(["M", "F"], size=n, p=[0.5, 0.5])

    return pd.DataFrame({
        "group": group,
        "Sex": sex,
    })


@pytest.fixture
def confounded_continuous_data():
    """Synthetic data where age differs significantly between groups."""
    rng = np.random.default_rng(42)
    n = 200
    group = np.array(["A"] * 100 + ["B"] * 100)
    # Group A mean=30, Group B mean=60 — very different
    age_a = rng.normal(30, 5, size=100)
    age_b = rng.normal(60, 5, size=100)
    age = np.concatenate([age_a, age_b])

    return pd.DataFrame({
        "group": group,
        "Age": age,
    })


@pytest.fixture
def balanced_continuous_data():
    """Synthetic data where age is similar across groups."""
    rng = np.random.default_rng(42)
    n = 200
    group = np.array(["A"] * 100 + ["B"] * 100)
    age = rng.normal(50, 10, size=n)

    return pd.DataFrame({
        "group": group,
        "Age": age,
    })


@pytest.fixture
def three_group_data():
    """Three-group data for ANOVA testing."""
    rng = np.random.default_rng(42)
    group = np.array(["C9ORF72"] * 30 + ["SPORADIC"] * 150 + ["CONTROL"] * 80)
    # Age is confounded: C9 patients are younger
    age_c9 = rng.normal(45, 5, size=30)
    age_spor = rng.normal(58, 10, size=150)
    age_ctrl = rng.normal(55, 12, size=80)
    age = np.concatenate([age_c9, age_spor, age_ctrl])
    sex = rng.choice(["M", "F"], size=260, p=[0.55, 0.45])

    return pd.DataFrame({
        "group": group,
        "Age": age,
        "Sex": sex,
    })


@pytest.fixture
def data_with_nans():
    """Data with NaN values in covariates."""
    rng = np.random.default_rng(42)
    n = 100
    group = np.array(["A"] * 50 + ["B"] * 50)
    age = rng.normal(50, 10, size=n).astype(float)
    sex = rng.choice(["M", "F"], size=n).astype(object)
    # Introduce NaN in 20% of age values
    nan_idx = rng.choice(n, size=20, replace=False)
    age[nan_idx] = np.nan
    # Introduce NaN in sex too
    sex_nan_idx = rng.choice(n, size=5, replace=False)
    sex[sex_nan_idx] = None

    return pd.DataFrame({
        "group": group,
        "Age": age,
        "Sex": sex,
    })


# ─── Tests: Categorical detection ──────────────────────────────────────


class TestCategoricalDetection:
    """Test detection of categorical confounding via chi-squared."""

    def test_confounded_categorical_detected(self, confounded_categorical_data):
        """Strong sex-group association should be flagged as confounded."""
        report = assess_covariate_confounding(
            metadata=confounded_categorical_data,
            group_column="group",
            covariates=["Sex"],
        )
        assert len(report.diagnostics) == 1
        diag = report.diagnostics[0]
        assert diag.covariate == "Sex"
        assert diag.covariate_type == "categorical"
        assert diag.test_name == "chi-sq"
        assert diag.p_value < 0.001  # Very strong association
        assert diag.confounded is True
        assert report.has_confounded() is True

    def test_balanced_categorical_not_confounded(self, balanced_categorical_data):
        """Balanced sex across groups should not be flagged."""
        report = assess_covariate_confounding(
            metadata=balanced_categorical_data,
            group_column="group",
            covariates=["Sex"],
        )
        diag = report.diagnostics[0]
        assert diag.p_value > 0.05
        assert diag.confounded is False
        assert report.has_confounded() is False

    def test_categorical_summary_format(self, confounded_categorical_data):
        """Check that summaries show percentages."""
        report = assess_covariate_confounding(
            metadata=confounded_categorical_data,
            group_column="group",
            covariates=["Sex"],
        )
        diag = report.diagnostics[0]
        # Group A should show ~90% M
        summary_a = diag.group_summaries["A"]
        assert "%" in summary_a
        assert "M" in summary_a


# ─── Tests: Continuous detection ────────────────────────────────────────


class TestContinuousDetection:
    """Test detection of continuous confounding via ANOVA/t-test."""

    def test_confounded_continuous_detected(self, confounded_continuous_data):
        """Strong age difference between groups should be flagged."""
        report = assess_covariate_confounding(
            metadata=confounded_continuous_data,
            group_column="group",
            covariates=["Age"],
        )
        diag = report.diagnostics[0]
        assert diag.covariate == "Age"
        assert diag.covariate_type == "continuous"
        assert diag.test_name == "t-test"  # 2 groups → t-test
        assert diag.p_value < 0.001
        assert diag.confounded is True

    def test_balanced_continuous_not_confounded(self, balanced_continuous_data):
        """Similar age distribution should not be flagged."""
        report = assess_covariate_confounding(
            metadata=balanced_continuous_data,
            group_column="group",
            covariates=["Age"],
        )
        diag = report.diagnostics[0]
        assert diag.p_value > 0.05
        assert diag.confounded is False

    def test_three_group_uses_anova(self, three_group_data):
        """Three groups should trigger ANOVA for continuous covariates."""
        report = assess_covariate_confounding(
            metadata=three_group_data,
            group_column="group",
            covariates=["Age"],
        )
        diag = report.diagnostics[0]
        assert diag.test_name == "ANOVA"
        # Age is confounded in our synthetic data (C9 is younger)
        assert diag.p_value < 0.05
        assert diag.confounded is True

    def test_continuous_summary_format(self, confounded_continuous_data):
        """Check that summaries show mean +/- std."""
        report = assess_covariate_confounding(
            metadata=confounded_continuous_data,
            group_column="group",
            covariates=["Age"],
        )
        diag = report.diagnostics[0]
        # Should contain +/- format
        for summary in diag.group_summaries.values():
            assert "+/-" in summary


# ─── Tests: NaN handling ────────────────────────────────────────────────


class TestNaNHandling:
    """Test behavior when covariates contain NaN values."""

    def test_nan_in_covariates_still_computes(self, data_with_nans):
        """NaN values should be excluded from computation, not crash."""
        report = assess_covariate_confounding(
            metadata=data_with_nans,
            group_column="group",
            covariates=["Age", "Sex"],
        )
        assert len(report.diagnostics) == 2
        # Both should produce finite p-values (enough data remains)
        for diag in report.diagnostics:
            assert np.isfinite(diag.p_value)

    def test_all_nan_covariate_in_group(self):
        """A covariate that is all-NaN in one group shows N/A summary."""
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "score": [1.0, 2.0, np.nan, np.nan],
        })
        report = assess_covariate_confounding(
            metadata=df,
            group_column="group",
            covariates=["score"],
        )
        diag = report.diagnostics[0]
        assert diag.group_summaries["B"] == "N/A"

    def test_missing_column_skipped(self):
        """A covariate not present in metadata should be skipped."""
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "Sex": ["M", "F", "M", "F"],
        })
        report = assess_covariate_confounding(
            metadata=df,
            group_column="group",
            covariates=["Sex", "NONEXISTENT"],
        )
        # Only Sex is reported
        assert len(report.diagnostics) == 1
        assert report.diagnostics[0].covariate == "Sex"


# ─── Tests: Report formatting ──────────────────────────────────────────


class TestReportFormatting:
    """Test the human-readable report output."""

    def test_format_table_basic_structure(self, three_group_data):
        """Report table should have header and data rows."""
        report = assess_covariate_confounding(
            metadata=three_group_data,
            group_column="group",
            covariates=["Age", "Sex"],
        )
        table = report.format_table()

        assert "Covariate Confounding Report" in table
        assert "\u2500" in table  # Unicode horizontal line
        assert "Age" in table
        assert "Sex" in table
        assert "p-value" in table

    def test_confounded_flagged_with_stars(self, confounded_continuous_data):
        """Confounded covariates should be flagged with ***."""
        report = assess_covariate_confounding(
            metadata=confounded_continuous_data,
            group_column="group",
            covariates=["Age"],
        )
        table = report.format_table()
        assert "***" in table

    def test_non_confounded_no_stars(self, balanced_continuous_data):
        """Non-confounded covariates should not have ***."""
        report = assess_covariate_confounding(
            metadata=balanced_continuous_data,
            group_column="group",
            covariates=["Age"],
        )
        table = report.format_table()
        assert "***" not in table

    def test_group_sizes_in_header(self, three_group_data):
        """Group sample sizes should appear in header."""
        report = assess_covariate_confounding(
            metadata=three_group_data,
            group_column="group",
            covariates=["Age"],
        )
        table = report.format_table()
        assert "n=30" in table  # C9ORF72
        assert "n=150" in table  # SPORADIC
        assert "n=80" in table  # CONTROL

    def test_empty_report(self):
        """Empty covariate list produces informative message."""
        report = CovariateReport(diagnostics=[], group_sizes={"A": 10})
        table = report.format_table()
        assert "No covariates to report" in table


# ─── Tests: Helper functions ────────────────────────────────────────────


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_is_categorical_string(self):
        """String columns should always be categorical."""
        s = pd.Series(["M", "F", "M", "F"])
        assert _is_categorical(s) is True

    def test_is_categorical_few_integers(self):
        """Integer columns with few unique values are categorical."""
        s = pd.Series([1, 2, 3, 1, 2, 3])
        assert _is_categorical(s) is True

    def test_is_categorical_many_integers(self):
        """Integer columns with many unique values are continuous."""
        s = pd.Series(list(range(50)))
        assert _is_categorical(s) is False

    def test_is_categorical_float(self):
        """Float columns with many unique values are continuous."""
        s = pd.Series(np.random.randn(100))
        assert _is_categorical(s) is False

    def test_is_categorical_bool(self):
        """Boolean columns are categorical."""
        s = pd.Series([True, False, True, False])
        assert _is_categorical(s) is True

    def test_summarize_categorical(self):
        """Categorical summary shows percentages."""
        s = pd.Series(["M", "M", "M", "F"])
        result = _summarize_categorical(s)
        assert "75% M" in result
        assert "25% F" in result

    def test_summarize_continuous(self):
        """Continuous summary shows mean +/- std."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = _summarize_continuous(s)
        assert "+/-" in result
        assert "25.0" in result  # mean

    def test_group_filtering(self):
        """Only specified groups should be included in analysis."""
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B", "C", "C"],
            "Age": [20, 25, 50, 55, 40, 45],
        })
        report = assess_covariate_confounding(
            metadata=df,
            group_column="group",
            covariates=["Age"],
            groups=["A", "B"],  # Exclude C
        )
        assert "C" not in report.group_sizes
        assert set(report.group_sizes.keys()) == {"A", "B"}
