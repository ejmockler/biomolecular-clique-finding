"""Covariate confounding diagnostics for multi-group analyses.

Provides utilities to assess whether covariates are confounded with
experimental groups. For each covariate, tests whether the distribution
differs across groups using chi-squared (categorical) or ANOVA/t-test
(continuous). This helps flag covariates that may introduce bias into
differential analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CovariateDiagnostic:
    """Result of a single covariate confounding test.

    Attributes:
        covariate: Name of the covariate column.
        covariate_type: "categorical" or "continuous".
        group_summaries: Dict mapping group name to summary string
            (e.g., "60% M / 40% F" or "52.3 +/- 8.1").
        test_name: Name of the statistical test used.
        p_value: P-value from the confounding test.
        confounded: True if p < 0.05.
    """

    covariate: str
    covariate_type: str
    group_summaries: dict[str, str]
    test_name: str
    p_value: float
    confounded: bool


@dataclass
class CovariateReport:
    """Complete confounding assessment for all covariates.

    Attributes:
        diagnostics: List of per-covariate results.
        group_sizes: Dict mapping group name to sample count.
    """

    diagnostics: list[CovariateDiagnostic]
    group_sizes: dict[str, int]

    def has_confounded(self) -> bool:
        """Return True if any covariate is confounded with group."""
        return any(d.confounded for d in self.diagnostics)

    def format_table(self) -> str:
        """Format a human-readable confounding report table."""
        if not self.diagnostics:
            return "No covariates to report."

        group_names = list(self.group_sizes.keys())

        # Build header
        lines = []
        lines.append("Covariate Confounding Report")
        lines.append("\u2500" * 60)

        # Column headers
        header_parts = [f"{'':12}"]
        for gname in group_names:
            n = self.group_sizes[gname]
            header_parts.append(f"{gname} (n={n})")
        header_parts.append("Test")
        header_parts.append("p-value")
        lines.append("   ".join(header_parts))

        # Data rows
        for diag in self.diagnostics:
            parts = [f"{diag.covariate:12}"]
            for gname in group_names:
                summary = diag.group_summaries.get(gname, "N/A")
                parts.append(f"{summary:>20}")
            parts.append(f"{diag.test_name:>10}")
            flag = " ***" if diag.confounded else ""
            parts.append(f"{diag.p_value:.3f}{flag}")
            lines.append("   ".join(parts))

        return "\n".join(lines)


def _is_categorical(series: pd.Series, max_unique: int = 10) -> bool:
    """Determine if a series should be treated as categorical."""
    non_null = series.dropna()
    if non_null.dtype == object or non_null.dtype.name == "category":
        return True
    if pd.api.types.is_bool_dtype(non_null):
        return True
    n_unique = non_null.nunique()
    return n_unique < max_unique


def _summarize_categorical(series: pd.Series) -> str:
    """Summarize a categorical series as 'X% A / Y% B / ...'."""
    counts = series.value_counts(normalize=True)
    parts = []
    for val, prop in counts.items():
        parts.append(f"{prop*100:.0f}% {val}")
    return " / ".join(parts)


def _summarize_continuous(series: pd.Series) -> str:
    """Summarize a continuous series as 'mean +/- std'."""
    mean = series.mean()
    std = series.std()
    return f"{mean:.1f} +/- {std:.1f}"


def assess_covariate_confounding(
    metadata: pd.DataFrame,
    group_column: str,
    covariates: list[str],
    groups: Optional[list[str]] = None,
    max_categorical_unique: int = 10,
) -> CovariateReport:
    """Assess whether covariates are confounded with experimental groups.

    For each covariate:
    - If categorical (< max_categorical_unique unique values): chi-squared test
    - If continuous: one-way ANOVA (3+ groups) or t-test (2 groups)

    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata with group and covariate columns.
    group_column : str
        Column name identifying the experimental group.
    covariates : list[str]
        List of covariate column names to assess.
    groups : list[str], optional
        Specific groups to include. If None, uses all unique values.
    max_categorical_unique : int
        Maximum unique values before treating a column as continuous.

    Returns
    -------
    CovariateReport
        Complete confounding assessment.
    """
    from scipy import stats as sp_stats

    # Filter to relevant groups
    if groups is not None:
        mask = metadata[group_column].isin(groups)
        df = metadata.loc[mask].copy()
    else:
        df = metadata.copy()

    # Get group labels
    group_labels = sorted(df[group_column].dropna().unique().tolist())
    group_sizes = {g: int((df[group_column] == g).sum()) for g in group_labels}

    diagnostics = []
    for cov in covariates:
        if cov not in df.columns:
            # Skip missing columns — caller should have warned
            continue

        col = df[cov]
        is_cat = _is_categorical(col, max_unique=max_categorical_unique)

        # Build per-group summaries
        group_summaries = {}
        for g in group_labels:
            g_data = col[df[group_column] == g].dropna()
            if len(g_data) == 0:
                group_summaries[g] = "N/A"
            elif is_cat:
                group_summaries[g] = _summarize_categorical(g_data)
            else:
                group_summaries[g] = _summarize_continuous(g_data)

        # Statistical test
        if is_cat:
            # Chi-squared test of independence
            contingency = pd.crosstab(df[group_column], col)
            # Only include specified groups
            contingency = contingency.loc[
                contingency.index.isin(group_labels)
            ]
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                p_value = np.nan
                test_name = "chi-sq"
            else:
                chi2, p_value, _, _ = sp_stats.chi2_contingency(contingency)
                test_name = "chi-sq"
        else:
            # Continuous: ANOVA or t-test
            group_data = []
            for g in group_labels:
                g_values = pd.to_numeric(
                    col[df[group_column] == g], errors="coerce"
                ).dropna()
                if len(g_values) > 0:
                    group_data.append(g_values.values)

            if len(group_data) < 2:
                p_value = np.nan
                test_name = "N/A"
            elif len(group_data) == 2:
                t_stat, p_value = sp_stats.ttest_ind(
                    group_data[0], group_data[1], equal_var=False
                )
                test_name = "t-test"
            else:
                f_stat, p_value = sp_stats.f_oneway(*group_data)
                test_name = "ANOVA"

        confounded = bool(np.isfinite(p_value) and p_value < 0.05)

        diagnostics.append(CovariateDiagnostic(
            covariate=cov,
            covariate_type="categorical" if is_cat else "continuous",
            group_summaries=group_summaries,
            test_name=test_name,
            p_value=float(p_value) if np.isfinite(p_value) else float("nan"),
            confounded=confounded,
        ))

    return CovariateReport(diagnostics=diagnostics, group_sizes=group_sizes)
