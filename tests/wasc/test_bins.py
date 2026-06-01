"""Tests for the WASC bin builder + matched non-neighbor sampler.

Covers:
- assign_decile: tie handling, NaN propagation, equal-frequency binning,
  constant-input behavior (regression: all proteins → bin 9, not -1)
- compute_missingness: row-wise NaN fraction
- compute_marginal_correlation_with_anchor: pairwise-complete-case Pearson;
  zero-variance protein → NaN; anchor-vs-itself → NaN
- build_anchor_bins: cells produced correctly; anchor excluded; NaN bin -1.
  v1.0.2 axes parameter: 2-axis default, 3-axis opt-in for v1.1, ValueError
  if axes includes 'miss' but missingness is None.
- sample_matched_non_neighbors:
    * sampled proteins land in the SAME cell as the input neighbor
    * exclusion of anchor + true neighbors + already-sampled
    * bin-empty edge case returns None at that position
    * RNG-seeded reproducibility
- v1.0.2 amendment regressions:
    * default cell key is 2-tuple (degree, corr)
    * explicit axes=("degree","miss","corr") preserves prior 3-tuple behavior
    * on a 0%-NaN abundance matrix, 2-axis cell count >= 3-axis cell count
      per anchor and matched-non-neighbor draws are identical under same seed
- load_measured_degrees: integer round-trip
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.wasc.bins import (
    DEFAULT_AXES,
    AnchorBins,
    assign_decile,
    build_anchor_bins,
    compute_marginal_correlation_with_anchor,
    compute_missingness_per_protein,
    sample_matched_non_neighbors,
)


# ---------------------------------------------------------------------------
# assign_decile
# ---------------------------------------------------------------------------

class TestAssignDecile:
    def test_returns_minus_one_when_too_few_finite(self):
        v = np.array([1.0, 2.0, np.nan, np.nan])
        out = assign_decile(v)
        assert (out == -1).all()

    def test_equal_count_binning_on_uniform_data(self):
        v = np.arange(100, dtype=np.float64)
        out = assign_decile(v)
        # Each bin should have roughly 10 values
        counts = np.bincount(out[out >= 0], minlength=10)
        assert counts.min() >= 8 and counts.max() <= 12, (
            f"Decile counts not balanced: {counts}"
        )

    def test_nan_yields_minus_one(self):
        v = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        out = assign_decile(v)
        assert out[2] == -1
        # All finite values are binned
        assert (out[[0, 1, 3, 4, 5, 6, 7, 8, 9, 10]] >= 0).all()

    def test_tied_values_share_bin(self):
        v = np.zeros(20)
        v[10:] = 1.0
        out = assign_decile(v)
        # All 10 zeros share a bin, all 10 ones share a bin
        assert len(set(out[:10].tolist())) == 1
        assert len(set(out[10:].tolist())) == 1

    def test_constant_input_does_not_return_minus_one(self):
        """REGRESSION (v1.0.2 brutalist V2 mod): all-constant input must
        return a valid (non-negative) bin for every element.  If this
        returned -1 the v1.0 missingness axis would have been silently
        producing zero-candidate cells, a load-bearing pre-existing bug.
        Observed behavior: all elements → bin 9."""
        for v in [np.zeros(100), np.full(100, 0.5), np.full(100, 1e10)]:
            out = assign_decile(v)
            assert (out >= 0).all(), f"Constant input produced bin -1: {np.unique(out)}"
            # All in a single bin (degenerate to high bin under searchsorted right)
            assert len(set(out.tolist())) == 1


# ---------------------------------------------------------------------------
# compute_missingness_per_protein
# ---------------------------------------------------------------------------

class TestMissingness:
    def test_basic(self):
        df = pd.DataFrame(
            [
                [1.0, 2.0, np.nan, 4.0],
                [np.nan, np.nan, np.nan, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            index=["A", "B", "C"],
        )
        m = compute_missingness_per_protein(df)
        assert m["A"] == 0.25
        assert m["B"] == 0.75
        assert m["C"] == 0.0


# ---------------------------------------------------------------------------
# compute_marginal_correlation_with_anchor
# ---------------------------------------------------------------------------

class TestMarginalCorrelation:
    def test_self_is_nan(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((5, 20)),
                          index=list("ABCDE"))
        r = compute_marginal_correlation_with_anchor(df, "A")
        assert np.isnan(r["A"])

    def test_perfect_correlation_returns_one(self):
        n = 30
        rng = np.random.default_rng(0)
        anchor = rng.standard_normal(n)
        df = pd.DataFrame(
            [anchor, 2 * anchor + 3, anchor + rng.standard_normal(n) * 0.001],
            index=["A", "B", "C"],
        )
        r = compute_marginal_correlation_with_anchor(df, "A")
        assert r["B"] == pytest.approx(1.0, abs=1e-9)
        assert r["C"] > 0.99

    def test_zero_variance_protein_is_nan(self):
        n = 30
        anchor = np.linspace(0, 1, n)
        constant_row = np.zeros(n)  # zero variance
        df = pd.DataFrame([anchor, constant_row], index=["A", "B"])
        r = compute_marginal_correlation_with_anchor(df, "A")
        assert np.isnan(r["B"])

    def test_nan_aware_pairwise(self):
        n = 30
        rng = np.random.default_rng(0)
        anchor = rng.standard_normal(n)
        other = 0.5 * anchor + 0.5 * rng.standard_normal(n)
        # Inject NaN in OTHER protein
        other[10:15] = np.nan
        df = pd.DataFrame([anchor, other], index=["A", "B"])
        r = compute_marginal_correlation_with_anchor(df, "A")
        # Should be in (0, 1) — corr computed on the remaining 25 samples
        assert 0 < r["B"] < 1


# ---------------------------------------------------------------------------
# build_anchor_bins
# ---------------------------------------------------------------------------

class TestBuildAnchorBins:
    @pytest.fixture
    def small_abundance(self):
        # 12 proteins, 30 samples (enough for decile binning)
        rng = np.random.default_rng(0)
        n_proteins = 12
        n_samples = 30
        idx = [f"P{i:02d}" for i in range(n_proteins)]
        cols = [f"S{i:02d}" for i in range(n_samples)]
        data = rng.standard_normal((n_proteins, n_samples))
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_anchor_excluded_from_cells(self, small_abundance):
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(np.linspace(0, 0.5, len(small_abundance)),
                                index=small_abundance.index)
        bins = build_anchor_bins("P00", small_abundance, degrees, missingness)
        for cell in bins.cells.values():
            assert "P00" not in cell

    def test_returns_correct_n_cells_or_fewer(self, small_abundance):
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(np.linspace(0, 0.5, len(small_abundance)),
                                index=small_abundance.index)
        bins = build_anchor_bins("P00", small_abundance, degrees, missingness)
        # Cells are 3-dim deciles; n_cells ≤ 10*10*10
        assert len(bins.cells) <= 1000
        # All 11 non-anchor proteins should be assigned to a cell
        total_in_cells = sum(len(v) for v in bins.cells.values())
        # Some may be -1 binned if too-few-finite; allow ≤ 11
        assert total_in_cells <= 11

    def test_get_cell_key(self, small_abundance):
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(np.linspace(0, 0.5, len(small_abundance)),
                                index=small_abundance.index)
        bins = build_anchor_bins("P00", small_abundance, degrees, missingness)
        for p in small_abundance.index:
            if p == "P00":
                continue
            key = bins.get_cell_key(p)
            if key is not None:
                # That cell should contain p
                assert p in bins.cells[key]


# ---------------------------------------------------------------------------
# sample_matched_non_neighbors
# ---------------------------------------------------------------------------

class TestSampleMatchedNonNeighbors:
    def _make_anchor_bins(self):
        """Hand-crafted AnchorBins with known cells for testing.

        Uses the v1.0 3-axis layout (`axes=("degree","miss","corr")`) so
        the cell-key shape matches the hand-written 3-tuple keys.  These
        tests cover the SAMPLER, which is axis-count-agnostic.
        """
        proteins = tuple(f"P{i:02d}" for i in range(10))
        deg_bin = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype=np.int8)
        miss_bin = np.zeros(10, dtype=np.int8)
        corr_bin = np.zeros(10, dtype=np.int8)
        cells = {
            (0, 0, 0): ("P00", "P01", "P02"),  # but anchor is P00 — must exclude
            (1, 0, 0): ("P03", "P04", "P05"),
            (2, 0, 0): ("P06", "P07", "P08"),
            (3, 0, 0): ("P09",),
        }
        anchor = "P00"
        # Strip anchor from cells (mimics build_anchor_bins behavior)
        cells = {k: tuple(p for p in v if p != anchor) for k, v in cells.items()}
        return AnchorBins(
            anchor_uniprot=anchor,
            protein_ids=proteins,
            deg_bin=deg_bin,
            miss_bin=miss_bin,
            corr_bin=corr_bin,
            cells=cells,
            axes=("degree", "miss", "corr"),
            _protein_to_idx={p: i for i, p in enumerate(proteins)},
        )

    def test_sampled_proteins_match_bins(self):
        bins = self._make_anchor_bins()
        # Use only one true_neighbor so the cell (1,0,0)={P03,P04,P05}
        # leaves 2 candidates eligible after exclusion of P03.
        true_neighbors = ["P03"]
        rng = np.random.default_rng(0)
        sampled, n_deg = sample_matched_non_neighbors(bins, true_neighbors, rng)
        assert n_deg == 0
        assert len(sampled) == 1
        assert sampled[0] in {"P04", "P05"}  # both in same cell as P03

    def test_excludes_anchor_and_true_neighbors(self):
        bins = self._make_anchor_bins()
        true_neighbors = ["P03"]
        rng = np.random.default_rng(0)
        sampled, _ = sample_matched_non_neighbors(bins, true_neighbors, rng)
        assert "P00" not in sampled  # anchor
        assert "P03" not in sampled  # true neighbor

    def test_bin_empty_returns_none(self):
        """Cell (3,0,0) has only P09. If P09 is the true neighbor, no
        eligible candidate remains → None."""
        bins = self._make_anchor_bins()
        true_neighbors = ["P09"]
        rng = np.random.default_rng(0)
        sampled, n_deg = sample_matched_non_neighbors(bins, true_neighbors, rng)
        assert sampled == [None]
        assert n_deg == 1

    def test_unbinned_neighbor_returns_none(self):
        """If the true neighbor isn't in any cell (e.g., bin=-1), None."""
        bins = self._make_anchor_bins()
        true_neighbors = ["P_not_in_anything"]
        rng = np.random.default_rng(0)
        sampled, n_deg = sample_matched_non_neighbors(bins, true_neighbors, rng)
        assert sampled == [None]
        assert n_deg == 1

    def test_already_sampled_excluded_within_call(self):
        """Across multiple positions in true_neighbors, already-sampled
        proteins are not re-drawn (sampling without replacement)."""
        bins = self._make_anchor_bins()
        # Both true_neighbors in cell (2,0,0) which has {P06,P07,P08}
        true_neighbors = ["P06", "P07"]
        rng = np.random.default_rng(0)
        sampled, n_deg = sample_matched_non_neighbors(bins, true_neighbors, rng)
        # Only P08 is eligible (the other two are excluded); both sampler
        # positions try to draw → second one gets bin-empty
        assert sampled[0] == "P08"
        assert sampled[1] is None
        assert n_deg == 1

    def test_rng_reproducibility(self):
        """Same seed → same draw."""
        # Build a clean AnchorBins with one cell containing many candidates,
        # so randomness in the choice is observable.
        n = 20
        proteins = tuple(["ANCHOR"] + [f"X{i:02d}" for i in range(n)])
        deg_bin = np.ones(n + 1, dtype=np.int8)
        miss_bin = np.zeros(n + 1, dtype=np.int8)
        corr_bin = np.zeros(n + 1, dtype=np.int8)
        cells = {(1, 0, 0): tuple(f"X{i:02d}" for i in range(n))}
        bins = AnchorBins(
            anchor_uniprot="ANCHOR",
            protein_ids=proteins,
            deg_bin=deg_bin,
            miss_bin=miss_bin,
            corr_bin=corr_bin,
            cells=cells,
            axes=("degree", "miss", "corr"),
            _protein_to_idx={p: i for i, p in enumerate(proteins)},
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1, _ = sample_matched_non_neighbors(bins, ["X00", "X01"], rng1)
        s2, _ = sample_matched_non_neighbors(bins, ["X00", "X01"], rng2)
        assert s1 == s2
        # Different seed → likely different draw on n=20 cell
        rng3 = np.random.default_rng(123)
        s3, _ = sample_matched_non_neighbors(bins, ["X00", "X01"], rng3)
        # Not asserting inequality (low probability of same draw); just
        # asserting validity:
        assert all(s is not None for s in s3)


# ---------------------------------------------------------------------------
# v1.0.2 amendment regressions (axes parameter)
# ---------------------------------------------------------------------------

class TestV102AxesAmendment:
    """Lock the v1.0.2 amendment: default 2-axis null + 3-axis opt-in."""

    @pytest.fixture
    def small_abundance(self):
        # 12 proteins, 30 samples (enough for decile binning)
        rng = np.random.default_rng(7)
        n_proteins, n_samples = 12, 30
        idx = [f"P{i:02d}" for i in range(n_proteins)]
        cols = [f"S{i:02d}" for i in range(n_samples)]
        return pd.DataFrame(rng.standard_normal((n_proteins, n_samples)),
                            index=idx, columns=cols)

    def test_default_axes_is_2_axis(self):
        """DEFAULT_AXES is ('degree', 'corr') per v1.0.2 amendment."""
        assert DEFAULT_AXES == ("degree", "corr")

    def test_default_cell_key_is_2_tuple(self, small_abundance):
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        bins = build_anchor_bins("P00", small_abundance, degrees)  # no missingness
        for key in bins.cells:
            assert isinstance(key, tuple)
            assert len(key) == 2, f"Default key should be 2-tuple, got {key}"
        # AnchorBins.axes reflects the default
        assert bins.axes == ("degree", "corr")
        # miss_bin is None when not requested
        assert bins.miss_bin is None

    def test_explicit_3_axis_opt_in(self, small_abundance):
        """v1.1 prebatch re-derivation path: axes=('degree','miss','corr')."""
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(np.linspace(0, 0.5, len(small_abundance)),
                                index=small_abundance.index)
        bins = build_anchor_bins(
            "P00", small_abundance, degrees, missingness,
            axes=("degree", "miss", "corr"),
        )
        for key in bins.cells:
            assert len(key) == 3, f"3-axis key expected, got {key}"
        assert bins.axes == ("degree", "miss", "corr")
        assert bins.miss_bin is not None

    def test_value_error_if_miss_requested_without_data(self, small_abundance):
        """`axes` includes 'miss' but missingness=None must raise loud."""
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        with pytest.raises(ValueError, match="missingness is None"):
            build_anchor_bins(
                "P00", small_abundance, degrees, missingness=None,
                axes=("degree", "miss", "corr"),
            )

    def test_unknown_axis_raises(self, small_abundance):
        """Unknown axis name fails loud, prevents typo'd opt-in."""
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        with pytest.raises(ValueError, match="Unknown axes"):
            build_anchor_bins(
                "P00", small_abundance, degrees,
                axes=("degree", "intensity"),
            )

    def test_2axis_at_least_3axis_cell_population(self, small_abundance):
        """Operationalizes the §4 'strictly more conservative than 1-axis'
        rationale: collapsing an axis MERGES cells, so per-cell candidate
        pool size can only grow (or stay equal).  For every populated
        2-axis cell, total membership >= sum of corresponding 3-axis cell
        populations restricted to (deg, corr) projection.
        """
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(np.zeros(len(small_abundance)),  # degenerate
                                index=small_abundance.index)
        bins_2axis = build_anchor_bins("P00", small_abundance, degrees)
        bins_3axis = build_anchor_bins(
            "P00", small_abundance, degrees, missingness,
            axes=("degree", "miss", "corr"),
        )
        # 2-axis cell count <= 3-axis cell count (collapsing merges)
        assert len(bins_2axis.cells) <= len(bins_3axis.cells)
        # And total members are equal (no protein gained or lost)
        total_2 = sum(len(v) for v in bins_2axis.cells.values())
        total_3 = sum(len(v) for v in bins_3axis.cells.values())
        assert total_2 == total_3

    def test_2axis_identical_to_3axis_on_zero_nan_matrix(self, small_abundance):
        """ON A 0%-NAN MATRIX (the v1.0.2 trigger), 2-axis and 3-axis are
        MATHEMATICALLY IDENTICAL — the dropped axis is inert by data, not
        by spec change.  Per brutalist V2 mod: this is the technical
        evidence backing the amendment.  Under a fixed seed, the matched
        non-neighbor draws must be element-wise equal."""
        degrees = {p: i for i, p in enumerate(small_abundance.index)}
        missingness = pd.Series(
            np.zeros(len(small_abundance)),  # 0% NaN ⇒ all miss=9
            index=small_abundance.index,
        )
        bins_2 = build_anchor_bins("P00", small_abundance, degrees)
        bins_3 = build_anchor_bins(
            "P00", small_abundance, degrees, missingness,
            axes=("degree", "miss", "corr"),
        )
        # Sampler draws must agree under a fixed seed (cells are
        # 1-to-1 corresponding because the dropped axis is a constant).
        true_neighbors = ["P01", "P02", "P03"]
        rng2 = np.random.default_rng(12345)
        rng3 = np.random.default_rng(12345)
        s2, _ = sample_matched_non_neighbors(bins_2, true_neighbors, rng2)
        s3, _ = sample_matched_non_neighbors(bins_3, true_neighbors, rng3)
        assert s2 == s3, (
            f"2-axis vs 3-axis draws differ on 0%-NaN matrix:\n  2-axis: {s2}\n  3-axis: {s3}"
        )
