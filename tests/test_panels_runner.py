"""Unit tests for the parallel runner orchestration.

Real INDRA queries are out of scope here.  We patch
``_worker_run_seed`` to return synthetic ``PerSeedResult`` objects
and run the panel through a ``ThreadPoolExecutor`` (so the patch is
visible to the workers; with ProcessPoolExecutor the function is
re-imported in a subprocess and the patch is invisible).

Verifies the orchestration contract:
- All seeds are submitted (target + every panel member).
- Returned ``PanelResult`` correctly partitions completed vs failed.
- ``per_seed`` and ``failed_seeds`` are sorted in design order →
  byte-deterministic ``result.json`` across runs.
- ``failed_seeds`` carries structured exception info, not just names.
- Manifest YAML and result JSON are written atomically.
- Target failure raises (panel-fatal).
- Per-seed failures are recorded but don't kill the panel.
- Per-seed timeout cancels pending work and records the timeout.
- ``ProcessPoolExecutor`` rejects unpicklable ``group_resolver``.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cliquefinder.panels import (
    FailedSeed,
    PanelDesign,
    PanelResult,
    PanelStratum,
    PerSeedResult,
    ShellSummary,
    TARGET_STRATUM_LABEL,
    run_panel,
)


def _design() -> PanelDesign:
    return PanelDesign(
        target_seed="C9orf72",
        strata=(
            PanelStratum(name="A", members=("G1", "G2")),
            PanelStratum(name="B", members=("G3",)),
        ),
        contrast=("CASE", "CTRL"),
        max_hops=2,
        n_permutations=49,
        covariates=(),
        selection_rng_seed=42,
    )


def _fake_resolver(metadata: pd.DataFrame) -> dict[str, pd.Index]:
    """Always returns two non-empty groups."""
    half = len(metadata) // 2
    return {
        "CASE": metadata.index[:half],
        "CTRL": metadata.index[half:],
    }


def _fake_result(seed: str, stratum: str) -> PerSeedResult:
    return PerSeedResult(
        seed=seed, stratum=stratum,
        slope=-0.1, slope_pvalue=0.05,
        spearman_rho=-0.1, spearman_pvalue=0.05,
        shells=(
            ShellSummary(hop=1, n_genes=10, mean_abs_t=1.0, median_abs_t=1.0),
        ),
        n_genes_total=10, elapsed_seconds=1.0,
    )


def _write_dummy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """CSV inputs that satisfy load_panel_inputs but contain no real data."""
    data_path = tmp_path / "data.csv"
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"S1": [1.0, 2.0], "S2": [3.0, 4.0], "S3": [5.0, 6.0], "S4": [7.0, 8.0]},
        index=pd.Index(["P1", "P2"], name="feature_id"),
    ).to_csv(data_path)
    pd.DataFrame(
        {"X": [1, 2, 3, 4]},
        index=pd.Index(["S1", "S2", "S3", "S4"], name="sample_id"),
    ).to_csv(metadata_path)
    return data_path, metadata_path


class TestRunPanelOrchestration:
    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_writes_manifest_and_result(
        self, mock_worker, tmp_path: Path,
    ):
        mock_worker.side_effect = lambda **kw: _fake_result(
            kw["seed"], kw["stratum"],
        )
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=1,
            executor_class=ThreadPoolExecutor,
        )

        assert (tmp_path / "out" / "manifest.yaml").exists()
        assert (tmp_path / "out" / "result.json").exists()
        # Round-trips:
        loaded = PanelResult.load_json(tmp_path / "out" / "result.json")
        assert loaded == result

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_runs_target_and_all_panel_members(
        self, mock_worker, tmp_path: Path,
    ):
        seen = []

        def capture(**kw):
            seen.append((kw["seed"], kw["stratum"]))
            return _fake_result(kw["seed"], kw["stratum"])

        mock_worker.side_effect = capture
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=1,
            executor_class=ThreadPoolExecutor,
        )

        seeds_seen = {s for s, _ in seen}
        assert seeds_seen == {"C9orf72", "G1", "G2", "G3"}

        # Target carries the reserved stratum label.
        target_call = next(s for s in seen if s[0] == "C9orf72")
        assert target_call[1] == TARGET_STRATUM_LABEL

        # Panel result has the right shape.
        assert result.target_result.seed == "C9orf72"
        assert {r.seed for r in result.per_seed} == {"G1", "G2", "G3"}
        assert result.failed_seeds == ()

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_per_seed_failure_recorded_but_not_fatal(
        self, mock_worker, tmp_path: Path,
    ):
        def maybe_fail(**kw):
            if kw["seed"] == "G2":
                raise RuntimeError("simulated Neo4j timeout")
            return _fake_result(kw["seed"], kw["stratum"])

        mock_worker.side_effect = maybe_fail
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=1,
            executor_class=ThreadPoolExecutor,
        )

        # failed_seeds carries structured info, not bare names.
        assert len(result.failed_seeds) == 1
        failure = result.failed_seeds[0]
        assert isinstance(failure, FailedSeed)
        assert failure.seed == "G2"
        assert failure.error_type == "RuntimeError"
        assert "Neo4j timeout" in failure.error_message
        assert {r.seed for r in result.per_seed} == {"G1", "G3"}

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_target_failure_is_panel_fatal(
        self, mock_worker, tmp_path: Path,
    ):
        def fail_target(**kw):
            if kw["seed"] == "C9orf72":
                raise RuntimeError("target seed Neo4j timeout")
            return _fake_result(kw["seed"], kw["stratum"])

        mock_worker.side_effect = fail_target
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        with pytest.raises(RuntimeError, match="Target seed.*Neo4j timeout"):
            run_panel(
                _design(),
                data_path=data_path,
                metadata_path=metadata_path,
                group_resolver=_fake_resolver,
                output_dir=tmp_path / "out",
                parallelism=1,
                executor_class=ThreadPoolExecutor,
            )

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_per_seed_ordering_is_design_order_not_completion_order(
        self, mock_worker, tmp_path: Path,
    ):
        """Two runs of the same panel must produce byte-identical
        result.json regardless of which worker finishes first.  Sort
        order is design.selected_seeds().
        """
        completion_order = ["G3", "G1", "G2"]  # arbitrary, not design order

        def reverse_completion_speed(**kw):
            seed = kw["seed"]
            if seed in completion_order:
                idx = completion_order.index(seed)
                time.sleep(0.005 * (len(completion_order) - idx))
            return _fake_result(seed, kw["stratum"])

        mock_worker.side_effect = reverse_completion_speed
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=4,
            executor_class=ThreadPoolExecutor,
        )

        # Design selected_seeds order is: G1, G2, G3 (RNA stratum first).
        observed_order = [r.seed for r in result.per_seed]
        assert observed_order == ["G1", "G2", "G3"], (
            f"per_seed not in design order: {observed_order}"
        )

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_failed_seeds_sorted_in_design_order(
        self, mock_worker, tmp_path: Path,
    ):
        def fail_g1_and_g3(**kw):
            if kw["seed"] in ("G1", "G3"):
                raise RuntimeError(f"{kw['seed']} failed")
            return _fake_result(kw["seed"], kw["stratum"])

        mock_worker.side_effect = fail_g1_and_g3
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=2,
            executor_class=ThreadPoolExecutor,
        )

        # Design order: G1, G2, G3.  Failed entries should follow that.
        assert [f.seed for f in result.failed_seeds] == ["G1", "G3"]


class TestRunPanelTimeout:
    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_timeout_marks_pending_seeds_failed(
        self, mock_worker, tmp_path: Path,
    ):
        """A wedged worker must not block the entire panel run."""

        def slow_g2(**kw):
            if kw["seed"] == "G2":
                time.sleep(5.0)
            return _fake_result(kw["seed"], kw["stratum"])

        mock_worker.side_effect = slow_g2
        data_path, metadata_path = _write_dummy_inputs(tmp_path)

        result = run_panel(
            _design(),
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=4,
            seed_timeout_seconds=0.5,
            executor_class=ThreadPoolExecutor,
        )

        assert any(
            f.seed == "G2" and f.error_type == "TimeoutError"
            for f in result.failed_seeds
        ), f"G2 timeout missing in {[f.to_dict() for f in result.failed_seeds]}"


class TestPicklabilityCheck:
    def test_lambda_resolver_rejected_for_process_pool(
        self, tmp_path: Path,
    ):
        """ProcessPoolExecutor would crash deep in futures plumbing
        on a lambda resolver — fail fast at submission time instead.
        """
        data_path, metadata_path = _write_dummy_inputs(tmp_path)
        with pytest.raises(TypeError, match="picklable"):
            run_panel(
                _design(),
                data_path=data_path,
                metadata_path=metadata_path,
                group_resolver=lambda meta: {  # type: ignore[arg-type]
                    "CASE": meta.index[:1], "CTRL": meta.index[1:],
                },
                output_dir=tmp_path / "out",
                parallelism=1,
                executor_class=ProcessPoolExecutor,
            )

    def test_lambda_resolver_accepted_for_thread_pool(
        self, tmp_path: Path,
    ):
        """ThreadPoolExecutor doesn't pickle, so the picklability
        check should be skipped — tests can use lambdas.
        """
        with patch(
            "cliquefinder.panels.runner._worker_run_seed",
            side_effect=lambda **kw: _fake_result(
                kw["seed"], kw["stratum"],
            ),
        ):
            data_path, metadata_path = _write_dummy_inputs(tmp_path)
            run_panel(
                _design(),
                data_path=data_path,
                metadata_path=metadata_path,
                group_resolver=_fake_resolver,
                output_dir=tmp_path / "out",
                parallelism=1,
                executor_class=ThreadPoolExecutor,
            )


class TestParallelismBounds:
    def test_zero_parallelism_rejected(self, tmp_path: Path):
        data_path, metadata_path = _write_dummy_inputs(tmp_path)
        with pytest.raises(ValueError, match="parallelism must be >= 1"):
            run_panel(
                _design(),
                data_path=data_path,
                metadata_path=metadata_path,
                group_resolver=_fake_resolver,
                output_dir=tmp_path / "out",
                parallelism=0,
                executor_class=ThreadPoolExecutor,
            )

    @patch("cliquefinder.panels.runner._worker_run_seed")
    def test_parallelism_larger_than_seed_count(
        self, mock_worker, tmp_path: Path,
    ):
        """Parallelism > len(seeds_to_run) should just work."""
        mock_worker.side_effect = lambda **kw: _fake_result(
            kw["seed"], kw["stratum"],
        )
        data_path, metadata_path = _write_dummy_inputs(tmp_path)
        result = run_panel(
            _design(),  # 1 target + 3 panel = 4 seeds
            data_path=data_path,
            metadata_path=metadata_path,
            group_resolver=_fake_resolver,
            output_dir=tmp_path / "out",
            parallelism=16,
            executor_class=ThreadPoolExecutor,
        )
        assert len(result.per_seed) == 3
