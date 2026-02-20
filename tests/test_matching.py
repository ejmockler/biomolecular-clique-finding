"""Tests for exact covariate matching."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.matching import MatchResult, exact_match_covariates


class TestExactMatchCovariates:
    """Tests for exact_match_covariates()."""

    @pytest.fixture
    def imbalanced_metadata(self):
        """Metadata with sex imbalance (mimics ALS dataset)."""
        # C9ORF72: 17M, 8F = 25 total
        # Sporadic: 207M, 77F = 284 total
        # Control: 32M, 59F = 91 total
        rows = (
            [{"group": "C9ORF72", "Sex": "M"}] * 17
            + [{"group": "C9ORF72", "Sex": "F"}] * 8
            + [{"group": "SPORADIC", "Sex": "M"}] * 207
            + [{"group": "SPORADIC", "Sex": "F"}] * 77
            + [{"group": "CONTROL", "Sex": "M"}] * 32
            + [{"group": "CONTROL", "Sex": "F"}] * 59
        )
        return pd.DataFrame(rows)

    def test_basic_matching(self, imbalanced_metadata):
        """Basic sex matching produces balanced groups."""
        result = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        assert isinstance(result, MatchResult)
        assert result.n_matched < result.n_original

        # Verify balance: each group should have same count per sex
        matched = imbalanced_metadata.iloc[result.matched_indices]
        counts = matched.groupby(["group", "Sex"]).size().unstack(fill_value=0)

        # C9ORF72 has 17M/8F → min is 8F (if all groups have ≥8F)
        # For Males: min(17, 207, 32) = 17
        # For Females: min(8, 77, 59) = 8
        assert counts.loc["C9ORF72", "M"] == counts.loc["SPORADIC", "M"]
        assert counts.loc["C9ORF72", "M"] == counts.loc["CONTROL", "M"]
        assert counts.loc["C9ORF72", "F"] == counts.loc["SPORADIC", "F"]
        assert counts.loc["C9ORF72", "F"] == counts.loc["CONTROL", "F"]

    def test_two_group_matching(self, imbalanced_metadata):
        """Match only two groups, ignoring the third."""
        result = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            groups=["C9ORF72", "SPORADIC"],
            seed=42,
        )

        matched = imbalanced_metadata.iloc[result.matched_indices]

        # Only two groups in result
        assert set(matched["group"].unique()) == {"C9ORF72", "SPORADIC"}

        counts = matched.groupby(["group", "Sex"]).size().unstack(fill_value=0)
        assert counts.loc["C9ORF72", "M"] == counts.loc["SPORADIC", "M"]
        assert counts.loc["C9ORF72", "F"] == counts.loc["SPORADIC", "F"]

    def test_balance_audit(self, imbalanced_metadata):
        """Balance audit shows before/after counts."""
        result = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        audit = result.balance_audit
        assert "original_count" in audit.columns
        assert "matched_count" in audit.columns
        assert "dropped" in audit.columns
        assert len(audit) > 0

    def test_dropped_counts(self, imbalanced_metadata):
        """Dropped counts are computed correctly."""
        result = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        # Sporadic should drop the most (largest group)
        assert result.dropped_per_group["SPORADIC"] > result.dropped_per_group["C9ORF72"]

    def test_reproducible_with_seed(self, imbalanced_metadata):
        """Same seed produces same result."""
        r1 = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )
        r2 = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        np.testing.assert_array_equal(r1.matched_indices, r2.matched_indices)

    def test_different_seeds_differ(self, imbalanced_metadata):
        """Different seeds may produce different subsets."""
        r1 = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )
        r2 = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=123,
        )

        # They should have the same size but possibly different samples
        assert r1.n_matched == r2.n_matched

    def test_missing_column_raises(self):
        """Missing column → ValueError."""
        metadata = pd.DataFrame({"group": ["A", "B"], "Sex": ["M", "F"]})
        with pytest.raises(ValueError, match="not found"):
            exact_match_covariates(metadata, "group", ["Missing"])

    def test_empty_stratum_skipped(self):
        """Stratum with zero in one group is skipped."""
        metadata = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "Sex": ["M", "M", "F", "M", "M"],  # No F in group B
        })

        result = exact_match_covariates(
            metadata=metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        matched = metadata.iloc[result.matched_indices]
        # Only Males should be matched (F stratum skipped due to 0 in B)
        assert all(matched["Sex"] == "M")

    def test_multiple_match_vars(self):
        """Matching on multiple variables creates cross-strata."""
        metadata = pd.DataFrame({
            "group": ["A"] * 8 + ["B"] * 8,
            "Sex": ["M", "M", "F", "F", "M", "M", "F", "F"] * 2,
            "Batch": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"] * 2,
        })

        result = exact_match_covariates(
            metadata=metadata,
            group_col="group",
            match_vars=["Sex", "Batch"],
            seed=42,
        )

        matched = metadata.iloc[result.matched_indices]
        counts = matched.groupby(["group", "Sex", "Batch"]).size()
        # Each stratum should be balanced
        for (_, sex, batch), count in counts.items():
            partner = counts.get(("A" if _ == "B" else "B", sex, batch), 0)
            assert count == partner

    def test_to_dict(self, imbalanced_metadata):
        """Result serializes to dict."""
        result = exact_match_covariates(
            metadata=imbalanced_metadata,
            group_col="group",
            match_vars=["Sex"],
            seed=42,
        )

        d = result.to_dict()
        assert "n_original" in d
        assert "n_matched" in d
        assert "balance_audit" in d
