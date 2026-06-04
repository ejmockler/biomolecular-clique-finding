"""CI test for the WASC v1.0.5 C17′ B-feasibility table.

Asserts the (B → testable-rank-count) mapping from independent
arithmetic, NOT from text-quoted numbers.  Prevents v1.0.4-class
C17 bugs (sign inversion / B mis-attribution) from re-occurring.

Per v1.0.5 amendment lessons section: future amendments that
introduce NEW arithmetic claims (vs correcting OLD ones) should
require an extra brutalist verifier whose sole job is to re-derive
the claim from first principles.  This test is the durable form
of that gate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, (REPO / "scripts" / "wasc").as_posix())

from verify_by_feasibility import feasibility_at_B, H_N, N, Q


def test_H_N_matches_locked_value():
    """H_944 = 7.4279 (4 dp) per v1.0.4 C1 + v1.0.5 C17′."""
    assert N == 944
    assert Q == 0.10
    assert abs(H_N - 7.4279) < 1e-4, f"H_944 = {H_N:.6f}, expected 7.4279"


def test_rank1_raw_p_threshold():
    """Rank-1 BY threshold per spec §6 v1.0.4 C1: q/(N·H_N) = 1.43e-5."""
    rank1 = Q / (N * H_N)
    assert abs(rank1 - 1.4261e-5) < 1e-8, f"rank-1 = {rank1:.6e}"


def test_B999_sensitivity_tier():
    """At B=999: ranks 1..70 untestable; 874 testable."""
    r = feasibility_at_B(999)
    assert r["k_min"] == 71
    assert r["n_untestable"] == 70
    assert r["n_testable"] == 874


def test_B9999_primary():
    """At B=9999: ranks 1..7 untestable; 937 testable."""
    r = feasibility_at_B(9999)
    assert r["k_min"] == 8
    assert r["n_untestable"] == 7
    assert r["n_testable"] == 937


def test_B99999_floor_tie_rerun():
    """At B=99999: all 944 ranks testable."""
    r = feasibility_at_B(99999)
    assert r["k_min"] == 1
    assert r["n_untestable"] == 0
    assert r["n_testable"] == 944


def test_v104_C17_was_wrong():
    """REGRESSION: v1.0.4 C17 said 'at B=99999 only ranks 1..69 testable'.
    Locks the correction in v1.0.5: at B=99999, all 944 ranks ARE testable.
    The number 69 came from a confused calc at B=999 (where ranks 1..70 are
    UNTESTABLE), not B=99999 (where 0 are untestable)."""
    r99999 = feasibility_at_B(99999)
    assert r99999["n_testable"] == 944, "v1.0.5 must lock all-ranks-testable at B=99999"
    r999 = feasibility_at_B(999)
    assert r999["n_untestable"] == 70, "the 69-from-confusion was actually 70 untestable at B=999"
