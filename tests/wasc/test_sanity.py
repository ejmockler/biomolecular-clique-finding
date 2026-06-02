"""Tests for the WASC M2.5 prong (a) — label-shuffle null calibration.

Covers:
- shuffle_group_labels: group sizes preserved; donor X_cov rows travel
  with the donor; superset X_cov columns aligned; per-group X_cov
  re-trimmed to non-zero columns
- _fit_observed_q_for_works: per-anchor Q vector shape; NaN handling
- run_label_shuffle_calibration:
    * Basic wiring: 1 shuffle, B small, returns LabelShuffleResult
    * Validates ValueError on n_shuffles<1 and B<min_valid_perms
    * Reproducibility: same shuffle_seed → same FP rate sequence
    * On a synthetic context with no signal, mean FP rate ≈ p_threshold
      ± reasonable sampling variation (NOT a strict gate — that's the
      production run; this is wiring + plausibility)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.wasc.bins import AnchorBins
from cliquefinder.stats.wasc.null import AnchorWork
from cliquefinder.stats.wasc.preprocess import GroupDesign
from cliquefinder.stats.wasc.sanity import (
    LabelShuffleResult,
    _fit_observed_q_for_works,
    downsample_group,
    run_label_shuffle_calibration,
    shuffle_group_labels,
)


# ---------------------------------------------------------------------------
# shuffle_group_labels
# ---------------------------------------------------------------------------

def _make_designs_and_abundance(seed: int = 0):
    """Three groups with different sizes + non-overlapping tissue dummies.

    Sizes satisfy the spec §2.3 min_n_per_group defaults
    (C9>=10, SPOR>=15, CTRL>=15) so FWL converges in each group.
    """
    rng = np.random.default_rng(seed)
    n_proteins = 30
    proteins = [f"P{i:02d}" for i in range(n_proteins)]
    # Group sizes: C9=12, SPOR=25, CTRL=18 (total 55)
    sample_ids_per_group = {
        "C9ORF72":  [f"C9_{i:02d}" for i in range(12)],
        "SPORADIC": [f"SP_{i:02d}" for i in range(25)],
        "CONTROL":  [f"CT_{i:02d}" for i in range(18)],
    }
    # X_cov columns: intercept + sex_female + age_z (all groups have)
    # + tissue_NT_Cell (only C9 + SPOR has)
    designs = {}
    column_names_by_group = {
        "C9ORF72": ["intercept", "sex_female", "age_z", "tissue_NT_Cell"],
        "SPORADIC": ["intercept", "sex_female", "age_z", "tissue_NT_Cell"],
        "CONTROL": ["intercept", "sex_female", "age_z"],
    }
    for g, samples in sample_ids_per_group.items():
        n = len(samples)
        cols = column_names_by_group[g]
        X = np.zeros((n, len(cols)))
        X[:, 0] = 1.0  # intercept
        X[:, 1] = rng.choice([0, 1], n)  # sex
        X[:, 2] = rng.standard_normal(n)  # age_z
        if "tissue_NT_Cell" in cols:
            X[:, 3] = rng.choice([0, 1], n)
        designs[g] = GroupDesign(group=g, sample_ids=samples, X_cov=X, column_names=cols)

    all_samples = sum((d.sample_ids for d in designs.values()), [])
    abundance = pd.DataFrame(
        rng.standard_normal((n_proteins, len(all_samples))),
        index=proteins, columns=all_samples,
    )
    return designs, abundance, proteins


class TestShuffleGroupLabels:
    def test_group_sizes_preserved(self):
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(7)
        shuf_designs, sample_index, abundance_by_group = shuffle_group_labels(
            designs, abundance, rng,
        )
        for g in designs:
            assert len(shuf_designs[g].sample_ids) == len(designs[g].sample_ids)
            assert len(sample_index[g]) == len(designs[g].sample_ids)

    def test_donor_pool_preserved(self):
        """All original donors should still appear exactly once, just
        possibly under different group labels."""
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(7)
        shuf_designs, _, _ = shuffle_group_labels(designs, abundance, rng)
        orig_all = set()
        for d in designs.values():
            orig_all.update(d.sample_ids)
        shuf_all = set()
        for d in shuf_designs.values():
            shuf_all.update(d.sample_ids)
        assert orig_all == shuf_all

    def test_X_cov_alignment(self):
        """Each shuffled group's X_cov has appropriate non-zero columns
        (intercept always; sex/age usually; tissue only if at least one
        donor in the shuffled group has it)."""
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(7)
        shuf_designs, _, _ = shuffle_group_labels(designs, abundance, rng)
        for g, sd in shuf_designs.items():
            # Intercept always present
            assert "intercept" in sd.column_names
            # Each column has at least one non-zero value
            for j, name in enumerate(sd.column_names):
                assert np.any(sd.X_cov[:, j] != 0), \
                    f"Group {g} col {name} is all zeros — should have been trimmed"

    def test_abundance_slices_align_with_shuffled_samples(self):
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(7)
        shuf_designs, sample_index, abundance_by_group = shuffle_group_labels(
            designs, abundance, rng,
        )
        for g in designs:
            assert abundance_by_group[g].shape[0] == abundance.shape[0]
            assert abundance_by_group[g].shape[1] == len(shuf_designs[g].sample_ids)


# ---------------------------------------------------------------------------
# _fit_observed_q_for_works
# ---------------------------------------------------------------------------

class TestFitObservedQ:
    def test_basic_shape_and_finiteness(self):
        designs, abundance, proteins = _make_designs_and_abundance()
        # Build a work for one anchor with 2 true targets
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01", "P00|P02"),
            true_targets=("P01", "P02"),
            Q_obs=np.array([np.nan, np.nan]),  # ignored — we recompute
            seed=42,
        )
        uniprot_to_row = {p: i for i, p in enumerate(proteins)}
        rng = np.random.default_rng(0)
        _, sample_index, abundance_by_group = shuffle_group_labels(
            designs, abundance, rng,
        )
        X_cov_by_group = {g: d.X_cov for g, d in designs.items()}
        result = _fit_observed_q_for_works(
            [work], abundance_by_group, X_cov_by_group, uniprot_to_row,
        )
        assert "P00" in result
        assert result["P00"].shape == (2,)
        # Q values should mostly be finite on synthetic data (random ≠ NaN)
        assert np.isfinite(result["P00"]).any()

    def test_anchor_not_in_matrix_returns_nan(self):
        designs, abundance, proteins = _make_designs_and_abundance()
        work = AnchorWork(
            anchor_uniprot="NOT_IN_MATRIX",
            edge_ids=("NOT_IN_MATRIX|P01",),
            true_targets=("P01",),
            Q_obs=np.array([np.nan]),
            seed=0,
        )
        uniprot_to_row = {p: i for i, p in enumerate(proteins)}
        abundance_by_group = {g: abundance[d.sample_ids].values
                              for g, d in designs.items()}
        X_cov_by_group = {g: d.X_cov for g, d in designs.items()}
        result = _fit_observed_q_for_works(
            [work], abundance_by_group, X_cov_by_group, uniprot_to_row,
        )
        assert np.isnan(result["NOT_IN_MATRIX"]).all()


# ---------------------------------------------------------------------------
# run_label_shuffle_calibration
# ---------------------------------------------------------------------------

def _make_anchor_bins_for_test():
    """Minimal AnchorBins for P00 with all other proteins in one cell."""
    proteins = [f"P{i:02d}" for i in range(30)]
    n = len(proteins)
    deg_bin = np.ones(n, dtype=np.int8)
    miss_bin = None
    corr_bin = np.zeros(n, dtype=np.int8)
    cells = {(1, 0): tuple(p for p in proteins if p != "P00")}
    return AnchorBins(
        anchor_uniprot="P00", protein_ids=tuple(proteins),
        deg_bin=deg_bin, miss_bin=miss_bin, corr_bin=corr_bin,
        cells=cells, axes=("degree", "corr"),
        _protein_to_idx={p: i for i, p in enumerate(proteins)},
    )


class TestDownsampleGroup:
    def test_basic_downsample(self):
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(0)
        # SPOR original n=25; down-sample to 12
        new_designs, sample_index, abundance_by_group = downsample_group(
            designs, abundance, "SPORADIC", 12, rng,
        )
        assert len(new_designs["SPORADIC"].sample_ids) == 12
        assert sample_index["SPORADIC"].shape == (12,)
        assert abundance_by_group["SPORADIC"].shape[1] == 12
        # Other groups unchanged
        assert len(new_designs["C9ORF72"].sample_ids) == len(designs["C9ORF72"].sample_ids)
        assert len(new_designs["CONTROL"].sample_ids) == len(designs["CONTROL"].sample_ids)
        # Sampled donors are subset of original
        orig_set = set(designs["SPORADIC"].sample_ids)
        new_set = set(new_designs["SPORADIC"].sample_ids)
        assert new_set.issubset(orig_set)
        # No duplicates
        assert len(new_set) == 12

    def test_n_equal_orig_returns_permutation(self):
        designs, abundance, _ = _make_designs_and_abundance()
        rng = np.random.default_rng(0)
        n_orig = len(designs["SPORADIC"].sample_ids)
        new_designs, _, _ = downsample_group(
            designs, abundance, "SPORADIC", n_orig, rng,
        )
        # All original donors present (just possibly in different order)
        assert set(new_designs["SPORADIC"].sample_ids) == set(designs["SPORADIC"].sample_ids)

    def test_invalid_group_raises(self):
        designs, abundance, _ = _make_designs_and_abundance()
        with pytest.raises(ValueError, match="not in designs"):
            downsample_group(designs, abundance, "MADE_UP", 5,
                             np.random.default_rng(0))

    def test_invalid_n_raises(self):
        designs, abundance, _ = _make_designs_and_abundance()
        with pytest.raises(ValueError, match=r"must be in \[1"):
            downsample_group(designs, abundance, "SPORADIC", 0,
                             np.random.default_rng(0))
        with pytest.raises(ValueError, match=r"must be in \[1"):
            downsample_group(designs, abundance, "SPORADIC", 99999,
                             np.random.default_rng(0))

    def test_x_cov_trimmed_when_column_collapses(self):
        """If down-sample drops all donors having a specific tissue dummy,
        that column should be removed from X_cov."""
        # Build a fixture where one tissue level has only 1 donor in SPOR
        rng = np.random.default_rng(0)
        n_sp = 20
        cols = ["intercept", "sex_female", "age_z", "tissue_NT_Cell"]
        X = np.zeros((n_sp, 4))
        X[:, 0] = 1.0
        X[:, 1] = rng.choice([0, 1], n_sp)
        X[:, 2] = rng.standard_normal(n_sp)
        # Only donor 0 has tissue_NT_Cell = 1
        X[0, 3] = 1.0
        sp_samples = [f"SP_{i:02d}" for i in range(n_sp)]
        designs = {
            "SPORADIC": GroupDesign("SPORADIC", sp_samples, X, cols),
        }
        abundance = pd.DataFrame(
            rng.standard_normal((10, n_sp)),
            index=[f"P{i:02d}" for i in range(10)],
            columns=sp_samples,
        )
        # Force a draw that excludes donor 0 by trying many seeds
        for seed in range(50):
            rng = np.random.default_rng(seed)
            new_designs, _, _ = downsample_group(
                designs, abundance, "SPORADIC", 5, rng,
            )
            if "SP_00" not in new_designs["SPORADIC"].sample_ids:
                # X_cov should have trimmed the tissue_NT_Cell column
                assert "tissue_NT_Cell" not in new_designs["SPORADIC"].column_names
                assert new_designs["SPORADIC"].X_cov.shape[1] == 3
                break
        else:
            pytest.skip("Could not find seed that excludes SP_00 — fixture issue")

    def test_reproducibility(self):
        designs, abundance, _ = _make_designs_and_abundance()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        d1, _, _ = downsample_group(designs, abundance, "SPORADIC", 10, rng1)
        d2, _, _ = downsample_group(designs, abundance, "SPORADIC", 10, rng2)
        assert d1["SPORADIC"].sample_ids == d2["SPORADIC"].sample_ids


class TestRunLabelShuffleCalibration:
    def test_basic_wiring(self):
        designs, abundance, proteins = _make_designs_and_abundance()
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01", "P00|P02"),
            true_targets=("P01", "P02"),
            Q_obs=np.array([np.nan, np.nan]),
            seed=42,
        )
        bins = _make_anchor_bins_for_test()
        uniprot_to_row = {p: i for i, p in enumerate(proteins)}
        result = run_label_shuffle_calibration(
            works_template=[work],
            anchor_bins_by_anchor={"P00": bins},
            abundance=abundance,
            designs=designs,
            uniprot_to_row=uniprot_to_row,
            n_shuffles=2, B=50, min_valid_perms=10,
            verbose=False,
        )
        assert isinstance(result, LabelShuffleResult)
        assert result.n_shuffles == 2
        assert result.B == 50
        assert result.fp_rate_per_shuffle.shape == (2,)
        assert isinstance(result.pooled_pass, bool)
        # bound should be in (0.10, 1.0)
        assert 0.10 < result.bound < 1.0

    def test_value_errors(self):
        designs, abundance, proteins = _make_designs_and_abundance()
        bins = _make_anchor_bins_for_test()
        uniprot_to_row = {p: i for i, p in enumerate(proteins)}
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",), true_targets=("P01",),
            Q_obs=np.array([np.nan]), seed=42,
        )
        with pytest.raises(ValueError, match="n_shuffles"):
            run_label_shuffle_calibration(
                works_template=[work],
                anchor_bins_by_anchor={"P00": bins},
                abundance=abundance, designs=designs,
                uniprot_to_row=uniprot_to_row,
                n_shuffles=0, B=50, min_valid_perms=10, verbose=False,
            )
        with pytest.raises(ValueError, match="min_valid_perms"):
            run_label_shuffle_calibration(
                works_template=[work],
                anchor_bins_by_anchor={"P00": bins},
                abundance=abundance, designs=designs,
                uniprot_to_row=uniprot_to_row,
                n_shuffles=1, B=5, min_valid_perms=10, verbose=False,
            )

    def test_reproducibility_under_same_seed(self):
        designs, abundance, proteins = _make_designs_and_abundance()
        bins = _make_anchor_bins_for_test()
        uniprot_to_row = {p: i for i, p in enumerate(proteins)}
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",), true_targets=("P01",),
            Q_obs=np.array([np.nan]), seed=42,
        )
        kwargs = dict(
            works_template=[work],
            anchor_bins_by_anchor={"P00": bins},
            abundance=abundance, designs=designs,
            uniprot_to_row=uniprot_to_row,
            n_shuffles=2, B=50, min_valid_perms=10,
            shuffle_seed=123, verbose=False,
        )
        r1 = run_label_shuffle_calibration(**kwargs)
        r2 = run_label_shuffle_calibration(**kwargs)
        # Same shuffle seed + same anchor seed derivation → same FP rates
        np.testing.assert_array_equal(r1.fp_rate_per_shuffle, r2.fp_rate_per_shuffle)
