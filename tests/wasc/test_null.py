"""Tests for the WASC permutation null loop.

Covers:
- anchor_seed: deterministic, salt-aware
- compute_anchor_null:
    * shape and dtype of null_Q (n_edges, B)
    * p-values are in (0, 1]
    * NaN handling: bin-empty positions → null_Q[i,b] = NaN
    * min_valid_perms gate: p = NaN if too few finite null draws
    * anchor not in matrix → entire null is NaN
    * reproducibility: same seed → identical null_Q
- Checkpoint I/O:
    * append_checkpoint round-trip via load_completed_anchors
    * skip_completed in run_null_serial
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.wasc.bins import AnchorBins
from cliquefinder.stats.wasc.null import (
    AnchorNullResult,
    AnchorWork,
    NullLoopContext,
    anchor_seed,
    append_checkpoint,
    compute_anchor_null,
    load_completed_anchors,
    run_null_serial,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestAnchorSeed:
    def test_deterministic(self):
        assert anchor_seed("P12345") == anchor_seed("P12345")

    def test_different_proteins_different_seeds(self):
        # Hash collisions are astronomically unlikely on 4-byte hashes
        s1 = anchor_seed("P12345")
        s2 = anchor_seed("Q99880")
        assert s1 != s2

    def test_salt_changes_seed(self):
        s1 = anchor_seed("P12345", global_salt="v1.0")
        s2 = anchor_seed("P12345", global_salt="v2.0")
        assert s1 != s2

    def test_seed_in_uint32_range(self):
        s = anchor_seed("ABCDEFG")
        assert 0 <= s < 2 ** 32


# ---------------------------------------------------------------------------
# compute_anchor_null fixtures
# ---------------------------------------------------------------------------

def _make_test_context(
    n_proteins: int = 50,
    n_samples_per_group: int = 25,
    seed: int = 0,
) -> tuple[NullLoopContext, dict[str, AnchorBins], dict[str, int], list[str]]:
    """Build a synthetic 3-group context with a 50×n_g abundance matrix.

    Returns context, bins-by-anchor (one anchor: P00), uniprot_to_row, proteins.
    """
    rng = np.random.default_rng(seed)
    proteins = [f"P{i:02d}" for i in range(n_proteins)]
    groups = ("C9ORF72", "SPORADIC", "CONTROL")

    abundance_by_group = {}
    X_cov_by_group = {}
    sample_index_by_group = {}
    for g in groups:
        A = rng.standard_normal((n_proteins, n_samples_per_group))
        abundance_by_group[g] = A
        # Simple X_cov: intercept only
        X_cov_by_group[g] = np.ones((n_samples_per_group, 1))
        sample_index_by_group[g] = np.arange(n_samples_per_group)

    uniprot_to_row = {p: i for i, p in enumerate(proteins)}

    # Build AnchorBins for P00 with all proteins in cell (1,0,0)
    # so the sampler always has many candidates.
    # Uses the v1.0 3-axis layout — the sampler is axis-count-agnostic.
    n = n_proteins
    deg_bin = np.ones(n, dtype=np.int8)
    miss_bin = np.zeros(n, dtype=np.int8)
    corr_bin = np.zeros(n, dtype=np.int8)
    cells = {(1, 0, 0): tuple(p for p in proteins if p != "P00")}
    bins = AnchorBins(
        anchor_uniprot="P00",
        protein_ids=tuple(proteins),
        deg_bin=deg_bin,
        miss_bin=miss_bin,
        corr_bin=corr_bin,
        cells=cells,
        axes=("degree", "miss", "corr"),
        _protein_to_idx={p: i for i, p in enumerate(proteins)},
    )

    ctx = NullLoopContext(
        abundance_by_group=abundance_by_group,
        sample_index_by_group=sample_index_by_group,
        uniprot_to_row=uniprot_to_row,
        X_cov_by_group=X_cov_by_group,
        min_n_per_group={"C9ORF72": 10, "SPORADIC": 10, "CONTROL": 10},
        group_order=groups,
    )
    return ctx, {"P00": bins}, uniprot_to_row, proteins


# ---------------------------------------------------------------------------
# compute_anchor_null
# ---------------------------------------------------------------------------

class TestComputeAnchorNull:
    def test_basic_shape(self):
        ctx, bins_by_anchor, _, _ = _make_test_context()
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01", "P00|P02"),
            true_targets=("P01", "P02"),
            Q_obs=np.array([0.5, 1.0]),
            seed=42,
        )
        r = compute_anchor_null(
            work, bins_by_anchor["P00"],
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=100, min_valid_perms=10,
            group_order=ctx.group_order,
        )
        assert r.null_Q.shape == (2, 100)
        assert r.p_values.shape == (2,)
        assert np.all(np.isfinite(r.p_values))  # both ≥10 valid draws
        # p-values in (0, 1]
        assert ((r.p_values > 0) & (r.p_values <= 1)).all()

    def test_min_valid_perms_gate(self):
        """If too few finite null draws, p = NaN."""
        ctx, bins_by_anchor, _, _ = _make_test_context()
        # Use a bins object with an empty cell so sampler always returns None
        bins_empty = AnchorBins(
            anchor_uniprot="P00",
            protein_ids=bins_by_anchor["P00"].protein_ids,
            deg_bin=np.full(50, -1, dtype=np.int8),  # all unbinned → None always
            miss_bin=np.zeros(50, dtype=np.int8),
            corr_bin=np.zeros(50, dtype=np.int8),
            cells={},
            axes=("degree", "miss", "corr"),
            _protein_to_idx=bins_by_anchor["P00"]._protein_to_idx,
        )
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.5]),
            seed=42,
        )
        r = compute_anchor_null(
            work, bins_empty,
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=100, min_valid_perms=10,
            group_order=ctx.group_order,
        )
        assert np.isnan(r.p_values[0])
        assert r.n_degenerate_per_edge[0] == 100

    def test_anchor_not_in_matrix(self):
        ctx, bins_by_anchor, _, _ = _make_test_context()
        work = AnchorWork(
            anchor_uniprot="MISSING_PROTEIN",
            edge_ids=("X|Y",),
            true_targets=("Y",),
            Q_obs=np.array([0.5]),
            seed=42,
        )
        r = compute_anchor_null(
            work, bins_by_anchor["P00"],  # bins still resolves but anchor absent
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=50, min_valid_perms=10,
            group_order=ctx.group_order,
        )
        # All null draws NaN, p-value NaN
        assert np.isnan(r.p_values[0])
        assert r.n_degenerate_per_edge[0] == 50
        assert np.isnan(r.null_Q).all()

    def test_p_value_is_lower_tail(self):
        """REGRESSION (spec §4 line 207 / lower-tail convention):
        small Q_obs ⇒ small p ⇒ WASC-positive.  An anchor with Q_obs
        smaller than every null draw should yield p = 1/(B+1).  An
        anchor with Q_obs larger than every null draw should yield
        p = 1.0 (1 + B / (B+1) = (B+1)/(B+1))."""
        ctx, bins_by_anchor, _, _ = _make_test_context()

        # Q_obs = 0.0 (smaller than any chi2(2) sample) → p should hit
        # the floor (1/(B+1) under min_valid_perms=10).
        work_low = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.0]),
            seed=42,
        )
        r_low = compute_anchor_null(
            work_low, bins_by_anchor["P00"],
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=200, min_valid_perms=10, group_order=ctx.group_order,
        )
        # Lower-tail: very small Q_obs → very small p (close to floor)
        assert r_low.p_values[0] <= 0.05, (
            f"Q_obs=0 should give very small p; got {r_low.p_values[0]}"
        )

        # Q_obs = 1e6 (larger than any chi2(2) sample) → p ≈ 1
        work_high = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([1e6]),
            seed=42,
        )
        r_high = compute_anchor_null(
            work_high, bins_by_anchor["P00"],
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=200, min_valid_perms=10, group_order=ctx.group_order,
        )
        # Lower-tail: huge Q_obs → p ≈ 1 (every null is ≤ Q_obs)
        assert r_high.p_values[0] >= 0.95, (
            f"Q_obs=1e6 should give p ≈ 1; got {r_high.p_values[0]}"
        )

    def test_min_unique_q_values_guard(self):
        """REGRESSION (workflow wf_45fe2105-641 V1 verdict): edges whose
        null distribution has fewer than min_unique_q_values distinct
        values must get p=NaN.  The pathology: sparse-cell matched-bin
        sampling can draw the same fake t' for many iterations,
        producing a CONSTANT Q_null where the lower-tail formula is
        deterministic.  Guard catches this regardless of n_valid_perms.
        """
        # Build a context where the matched-bin cell has exactly 1
        # eligible candidate (after exclusion) — forces the sampler to
        # pick the same protein for every permutation.
        ctx, bins_by_anchor, _, _ = _make_test_context(n_proteins=50)
        # Replace bins with a 1-candidate cell
        bins_one = AnchorBins(
            anchor_uniprot="P00",
            protein_ids=bins_by_anchor["P00"].protein_ids,
            deg_bin=bins_by_anchor["P00"].deg_bin,
            miss_bin=bins_by_anchor["P00"].miss_bin,
            corr_bin=bins_by_anchor["P00"].corr_bin,
            cells={(1, 0, 0): ("P05",)},  # singleton cell
            axes=("degree", "miss", "corr"),
            _protein_to_idx=bins_by_anchor["P00"]._protein_to_idx,
        )
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.5]),
            seed=42,
        )
        r = compute_anchor_null(
            work, bins_one,
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=100, min_valid_perms=20, min_unique_q_values=5,
            group_order=ctx.group_order,
        )
        # With only 1 candidate in the cell, every permutation picks the
        # same fake → Q_null is constant → guard flags p=NaN.
        # Verify the null draws ARE all (nearly) identical
        finite_null = r.null_Q[0][np.isfinite(r.null_Q[0])]
        if len(finite_null) >= 5:
            assert len(np.unique(finite_null)) < 5, (
                f"Expected ≤4 unique null Q values; got {len(np.unique(finite_null))}"
            )
        # Guard should suppress p-value to NaN even though n_finite >= 20
        assert np.isnan(r.p_values[0]), (
            f"Expected p=NaN under constant-Q_null guard; got {r.p_values[0]}"
        )

    def test_min_unique_q_values_disabled_at_one(self):
        """Setting min_unique_q_values=1 disables the guard — equivalent
        to pre-guard behavior."""
        ctx, bins_by_anchor, _, _ = _make_test_context(n_proteins=50)
        bins_one = AnchorBins(
            anchor_uniprot="P00",
            protein_ids=bins_by_anchor["P00"].protein_ids,
            deg_bin=bins_by_anchor["P00"].deg_bin,
            miss_bin=bins_by_anchor["P00"].miss_bin,
            corr_bin=bins_by_anchor["P00"].corr_bin,
            cells={(1, 0, 0): ("P05",)},
            axes=("degree", "miss", "corr"),
            _protein_to_idx=bins_by_anchor["P00"]._protein_to_idx,
        )
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.5]),
            seed=42,
        )
        r = compute_anchor_null(
            work, bins_one,
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=100, min_valid_perms=20, min_unique_q_values=1,
            group_order=ctx.group_order,
        )
        # With guard disabled, p-value is computed (likely 0.01 or 1.0
        # depending on whether Q_obs is below/above the constant Q_null)
        assert np.isfinite(r.p_values[0])

    def test_reproducibility(self):
        """Same seed → identical null_Q matrix."""
        ctx, bins_by_anchor, _, _ = _make_test_context()
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.5]),
            seed=999,
        )
        r1 = compute_anchor_null(
            work, bins_by_anchor["P00"],
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=50, group_order=ctx.group_order,
        )
        r2 = compute_anchor_null(
            work, bins_by_anchor["P00"],
            ctx.abundance_by_group, ctx.sample_index_by_group,
            ctx.uniprot_to_row, ctx.X_cov_by_group,
            B=50, group_order=ctx.group_order,
        )
        # Compare element-wise, treating NaN as equal
        eq = np.where(
            np.isnan(r1.null_Q) & np.isnan(r2.null_Q),
            True,
            r1.null_Q == r2.null_Q,
        )
        assert eq.all(), "Same seed should produce identical null_Q"
        np.testing.assert_array_equal(r1.p_values, r2.p_values)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_roundtrip_single_anchor(self, tmp_path):
        ckpt = tmp_path / "ckpt.jsonl"
        r = AnchorNullResult(
            anchor_uniprot="P12345",
            edge_ids=("P12345|Q1", "P12345|Q2"),
            Q_obs=np.array([0.5, 1.0]),
            null_Q=np.array([[0.1, 0.2], [0.3, 0.4]]),
            p_values=np.array([0.05, 0.10]),
            n_degenerate_per_edge=np.array([0, 1]),
        )
        append_checkpoint(ckpt, r)
        assert ckpt.exists()
        done = load_completed_anchors(ckpt)
        assert done == {"P12345"}

    def test_multiple_anchors(self, tmp_path):
        ckpt = tmp_path / "ckpt.jsonl"
        for upid in ["P00001", "P00002", "P00003"]:
            r = AnchorNullResult(
                anchor_uniprot=upid,
                edge_ids=(f"{upid}|X",),
                Q_obs=np.array([0.5]),
                null_Q=np.array([[0.1]]),
                p_values=np.array([0.5]),
                n_degenerate_per_edge=np.array([0]),
            )
            append_checkpoint(ckpt, r)
        done = load_completed_anchors(ckpt)
        assert done == {"P00001", "P00002", "P00003"}

    def test_skip_completed(self, tmp_path):
        ctx, bins_by_anchor, _, _ = _make_test_context()
        # Pre-mark P00 as done in checkpoint
        ckpt = tmp_path / "ckpt.jsonl"
        ckpt.write_text(json.dumps({"anchor": "P00", "edge_ids": [],
                                    "Q_obs": [], "p_values": [],
                                    "n_degenerate": []}) + "\n")
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01",),
            true_targets=("P01",),
            Q_obs=np.array([0.5]),
            seed=42,
        )
        results = run_null_serial(
            [work], bins_by_anchor, ctx,
            B=10, min_valid_perms=5,
            checkpoint_path=ckpt, skip_completed=True,
        )
        assert results == []  # nothing computed; pre-existing skip

    def test_nan_in_record_serializes(self, tmp_path):
        ckpt = tmp_path / "ckpt.jsonl"
        r = AnchorNullResult(
            anchor_uniprot="P12345",
            edge_ids=("P12345|Q1",),
            Q_obs=np.array([np.nan]),
            null_Q=np.array([[np.nan]]),
            p_values=np.array([np.nan]),
            n_degenerate_per_edge=np.array([100]),
        )
        append_checkpoint(ckpt, r)
        # Should not crash and the file should be valid JSONL
        with ckpt.open() as fh:
            line = fh.readline()
        rec = json.loads(line)
        assert rec["anchor"] == "P12345"
        assert rec["Q_obs"] == [None]
        assert rec["p_values"] == [None]


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

class TestRunNullSerial:
    def test_e2e_smoke(self, tmp_path):
        ctx, bins_by_anchor, _, _ = _make_test_context()
        work = AnchorWork(
            anchor_uniprot="P00",
            edge_ids=("P00|P01", "P00|P02"),
            true_targets=("P01", "P02"),
            Q_obs=np.array([0.5, 1.0]),
            seed=42,
        )
        ckpt = tmp_path / "smoke.jsonl"
        results = run_null_serial(
            [work], bins_by_anchor, ctx,
            B=200, min_valid_perms=20,
            checkpoint_path=ckpt,
        )
        assert len(results) == 1
        assert results[0].anchor_uniprot == "P00"
        assert ckpt.exists()
        # Resume: re-running should skip
        results2 = run_null_serial(
            [work], bins_by_anchor, ctx,
            B=200, min_valid_perms=20,
            checkpoint_path=ckpt, skip_completed=True,
        )
        assert results2 == []
