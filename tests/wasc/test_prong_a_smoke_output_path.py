"""Tests for hygiene fix h1: per-config output path computation in
scripts/wasc/run_m2_5_prong_a_smoke.py.

PROBLEM (h1): The script previously wrote output/wasc/m2_5_prong_a_smoke/
result.json regardless of (candidate_pool, B, n_shuffles, shuffle_seed).
A later theme-restricted re-stamp silently overwrote the all-protein-pool
production result.

FIX: Output filename now embeds the configuration:
    result.{candidate_pool}_b{B}_n{n_shuffles}_seed{shuffle_seed}.json

These tests pin the string-level path computation; they do NOT run the
actual smoke (which requires Neo4j + the data bundle).  If anyone changes
the naming convention, they must update this test deliberately.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "wasc" / "run_m2_5_prong_a_smoke.py"


def _load_script_module():
    """Import the script as a module so we can call its helper directly.

    The script lives under scripts/wasc/, not under a package — load it
    by file path with importlib.
    """
    spec = importlib.util.spec_from_file_location(
        "run_m2_5_prong_a_smoke_h1test", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Filename format (the load-bearing string)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "candidate_pool,B,n_shuffles,shuffle_seed,expected",
    [
        # The two configs that collided in the bug we're fixing:
        ("all", 999, 20, 42,  "result.all_b999_n20_seed42.json"),
        ("theme", 999, 20, 42, "result.theme_b999_n20_seed42.json"),
        # Smoke-scale configs:
        ("all", 99,  5,  42,  "result.all_b99_n5_seed42.json"),
        ("theme", 99,  5,  42, "result.theme_b99_n5_seed42.json"),
        # Distinct seeds → distinct files (so prong-b-style multi-seed
        # extensions of the same calibration can coexist):
        ("all", 999, 20, 7,   "result.all_b999_n20_seed7.json"),
        ("all", 999, 20, 99,  "result.all_b999_n20_seed99.json"),
    ],
)
def test_filename_format(candidate_pool, B, n_shuffles, shuffle_seed, expected):
    mod = _load_script_module()
    got = mod.per_config_result_filename(
        candidate_pool=candidate_pool,
        B=B,
        n_shuffles=n_shuffles,
        shuffle_seed=shuffle_seed,
    )
    assert got == expected


# ---------------------------------------------------------------------------
# Distinctness (the core invariant of hygiene fix h1)
# ---------------------------------------------------------------------------

def test_distinct_pools_give_distinct_filenames():
    """The exact collision that caused the data loss must not recur:
    all-pool production vs theme-restricted re-stamp at the same B /
    n_shuffles / seed must write to different files."""
    mod = _load_script_module()
    allpool = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=42,
    )
    theme = mod.per_config_result_filename(
        candidate_pool="theme", B=999, n_shuffles=20, shuffle_seed=42,
    )
    assert allpool != theme
    assert "all" in allpool
    assert "theme" in theme


def test_distinct_B_give_distinct_filenames():
    mod = _load_script_module()
    f_99 = mod.per_config_result_filename(
        candidate_pool="all", B=99, n_shuffles=20, shuffle_seed=42,
    )
    f_999 = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=42,
    )
    assert f_99 != f_999


def test_distinct_n_shuffles_give_distinct_filenames():
    mod = _load_script_module()
    f_5 = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=5, shuffle_seed=42,
    )
    f_20 = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=42,
    )
    assert f_5 != f_20


def test_distinct_shuffle_seeds_give_distinct_filenames():
    mod = _load_script_module()
    f_42 = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=42,
    )
    f_7 = mod.per_config_result_filename(
        candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=7,
    )
    assert f_42 != f_7


def test_filename_is_deterministic():
    """Same inputs → same filename (no implicit randomness, no time)."""
    mod = _load_script_module()
    kw = dict(candidate_pool="all", B=999, n_shuffles=20, shuffle_seed=42)
    assert (
        mod.per_config_result_filename(**kw)
        == mod.per_config_result_filename(**kw)
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_unknown_candidate_pool():
    mod = _load_script_module()
    with pytest.raises(ValueError, match="candidate_pool"):
        mod.per_config_result_filename(
            candidate_pool="bogus", B=999, n_shuffles=20, shuffle_seed=42,
        )


@pytest.mark.parametrize("B", [0, -1, -999])
def test_rejects_nonpositive_B(B):
    mod = _load_script_module()
    with pytest.raises(ValueError):
        mod.per_config_result_filename(
            candidate_pool="all", B=B, n_shuffles=20, shuffle_seed=42,
        )


@pytest.mark.parametrize("n_shuffles", [0, -1, -20])
def test_rejects_nonpositive_n_shuffles(n_shuffles):
    mod = _load_script_module()
    with pytest.raises(ValueError):
        mod.per_config_result_filename(
            candidate_pool="all", B=999, n_shuffles=n_shuffles, shuffle_seed=42,
        )


# ---------------------------------------------------------------------------
# CLI surface (the script must accept --shuffle-seed; previously hardcoded)
# ---------------------------------------------------------------------------

def test_script_exposes_shuffle_seed_cli_arg():
    """Previously shuffle_seed was hardcoded to 42 inside main(), which
    meant the only way two configs could differ was candidate_pool/B/
    n_shuffles.  Hygiene fix h1 promotes shuffle_seed to a CLI arg so
    multi-seed prong-(a) runs are addressable in the path."""
    text = SCRIPT.read_text()
    assert "--shuffle-seed" in text, (
        "Script must expose --shuffle-seed as a CLI argument (h1)."
    )


def test_script_no_longer_writes_unsuffixed_result_json():
    """Defense in depth: the literal old write target
    OUT / 'result.json' must not appear in the script — that was the
    line that silently overwrote the production artifact."""
    text = SCRIPT.read_text()
    assert "\"result.json\"" not in text and "'result.json'" not in text, (
        "Script must not write to a fixed 'result.json' path (h1). "
        "Use per_config_result_filename() instead."
    )
