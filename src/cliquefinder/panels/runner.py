"""Parallel panel orchestration via ProcessPoolExecutor.

Architecture
------------
One subprocess worker per seed.  Each worker:
- Loads the proteomics CSV from disk (cheap; avoids pickling ~50MB
  of array data through the executor).
- Resolves cohort groups via the supplied ``group_resolver`` (a
  picklable callable; the panel layer is dataset-agnostic).
- Opens its own Neo4j connection inside ``DiscoveryBridge``.
- Returns a ``PerSeedResult`` (frozen, picklable).

Failures
--------
A seed that raises does not kill the panel run.  The exception is
logged and the seed is recorded in ``PanelResult.failed_seeds``.
The multiple-testing family in :func:`analyze_panel` uses the full
attempted set, so failures don't make discoveries anti-conservative.

Provenance
----------
Each completed ``PerSeedResult`` carries its own elapsed time.
``run_panel`` writes the panel manifest to ``output_dir/manifest.yaml``
before launching workers, and the rolled-up ``PanelResult`` to
``output_dir/result.json`` when finished — both via the project's
atomic write helpers.
"""
from __future__ import annotations

import logging
import pickle
from concurrent.futures import (
    BrokenExecutor,
    Executor,
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from pathlib import Path

from .analysis import (
    FailedSeed,
    PanelResult,
    PerSeedResult,
    TARGET_STRATUM_LABEL,
)
from .design import PanelDesign
from .seed_runner import GroupResolver, load_panel_inputs, run_seed_gradient

logger = logging.getLogger(__name__)


def _worker_run_seed(
    *,
    seed: str,
    stratum: str,
    contrast: tuple[str, str],
    data_path: str,
    metadata_path: str,
    group_resolver: GroupResolver,
    indra_env_file: str | None,
    covariates: tuple[str, ...],
    max_hops: int,
    n_permutations: int,
    rng_seed: int,
    transform: str,
) -> PerSeedResult:
    """Subprocess entry point: load data, run seed, return result.

    Path arguments are strings (not Path) so they're unconditionally
    picklable across platforms.  Pure positional kwargs throughout
    so the function is identifiable by reference at module level.
    """
    data, feature_ids, metadata, groups = load_panel_inputs(
        data_path=data_path,
        metadata_path=metadata_path,
        group_resolver=group_resolver,
    )
    return run_seed_gradient(
        seed=seed,
        stratum=stratum,
        contrast=contrast,
        data=data,
        feature_ids=feature_ids,
        metadata=metadata,
        groups=groups,
        indra_env_file=indra_env_file,
        covariates=covariates,
        max_hops=max_hops,
        n_permutations=n_permutations,
        rng_seed=rng_seed,
        transform=transform,
    )


def run_panel(
    design: PanelDesign,
    *,
    data_path: Path | str,
    metadata_path: Path | str,
    group_resolver: GroupResolver,
    indra_env_file: Path | str | None = None,
    output_dir: Path | str,
    parallelism: int = 4,
    rng_seed: int = 42,
    seed_timeout_seconds: float | None = None,
    executor_class: type[Executor] = ProcessPoolExecutor,
) -> PanelResult:
    """Run every seed in ``design`` (target + panel members) in parallel.

    Parameters
    ----------
    design
        Locked panel manifest.  Saved to
        ``output_dir/manifest.yaml`` before workers launch — the
        on-disk manifest is the authoritative record.
    data_path, metadata_path
        Proteomics CSV inputs.  Each worker re-reads these from
        disk; do NOT pre-load and pass arrays.
    group_resolver
        Callable mapping aligned metadata → ``{condition: sample
        index}``.  Must be picklable (module-level function or
        importable callable).
    indra_env_file
        ``.env`` containing INDRA Neo4j credentials.
    output_dir
        Directory for ``manifest.yaml`` and ``result.json``.  Created
        if missing.
    parallelism
        Worker count.  Practical cap is ~4–6 against the shared INDRA
        Neo4j endpoint; larger values risk timeouts.
    rng_seed
        Forwarded to every seed's permutation null.  All seeds in the
        same run share this base seed; per-seed reproducibility is via
        the seed name + this value.
    seed_timeout_seconds
        Per-seed wall-clock timeout.  ``None`` (default) means no
        timeout.  A wedged Neo4j connection inside one worker will
        otherwise block the entire panel run forever.
    executor_class
        Concurrent-futures Executor implementation.  Defaults to
        :class:`ProcessPoolExecutor` for true parallelism; pass
        :class:`~concurrent.futures.ThreadPoolExecutor` for tests
        that patch ``_worker_run_seed`` (the patch must be visible
        to the worker, which only happens with thread executors).

    Raises
    ------
    TypeError
        If ``group_resolver`` is not picklable (lambdas, nested
        functions, etc. break ProcessPoolExecutor at submission
        time; we catch this early with a clear error).
    RuntimeError
        If the target seed fails (panel-fatal — without target
        there is nothing to anchor analysis against).
    BrokenExecutor
        If the underlying executor's worker pool dies (segfault, OOM
        kill, etc.).  This is a halt condition, not a per-seed
        failure: continuing would mark every remaining seed as failed
        without actually running it.

    Returns
    -------
    PanelResult
        Includes design, target_result, per_seed (completed), and
        failed_seeds.  Saved to ``output_dir/result.json`` before
        return.
    """
    if parallelism < 1:
        raise ValueError(
            f"parallelism must be >= 1, got {parallelism}"
        )

    # Fail fast on unpicklable group_resolver: ProcessPoolExecutor
    # would crash deep inside the futures plumbing with an opaque
    # PicklingError after the manifest had already been written.
    # Skip the check for ThreadPool (no IPC) so test injection of
    # mocks/lambdas keeps working.
    if executor_class is ProcessPoolExecutor:
        try:
            pickle.dumps(group_resolver)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise TypeError(
                f"group_resolver must be picklable for ProcessPoolExecutor "
                f"(lambdas / nested functions are not).  Use a "
                f"module-level function or functools.partial.  "
                f"Underlying error: {exc}"
            ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.yaml"
    design.save_yaml(manifest_path)
    logger.info("Wrote panel manifest to %s", manifest_path)

    # Build the worker queue: target first, then panel members.
    # Stratum label is part of the worker payload so PerSeedResult
    # can be constructed without round-tripping through the design.
    seeds_to_run: list[tuple[str, str]] = [
        (design.target_seed, TARGET_STRATUM_LABEL)
    ]
    for s in design.strata:
        for member in s.members:
            seeds_to_run.append((member, s.name))

    common_kwargs = dict(
        contrast=design.contrast,
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        group_resolver=group_resolver,
        indra_env_file=str(indra_env_file) if indra_env_file else None,
        covariates=tuple(design.covariates),
        max_hops=design.max_hops,
        n_permutations=design.n_permutations,
        rng_seed=rng_seed,
        transform=design.transform,
    )

    completed: dict[str, PerSeedResult] = {}
    failed: dict[str, FailedSeed] = {}

    logger.info(
        "Running panel: %d seeds (1 target + %d panel members), "
        "parallelism=%d, timeout=%s",
        len(seeds_to_run), len(seeds_to_run) - 1, parallelism,
        f"{seed_timeout_seconds}s" if seed_timeout_seconds else "none",
    )

    with executor_class(max_workers=parallelism) as executor:
        future_to_seed = {
            executor.submit(
                _worker_run_seed,
                seed=seed,
                stratum=stratum,
                **common_kwargs,
            ): seed
            for seed, stratum in seeds_to_run
        }
        pending = set(future_to_seed.keys())
        while pending:
            done, pending = wait(
                pending,
                timeout=seed_timeout_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                # Timeout: cancel everything still pending and mark
                # those seeds as failed with a timeout error record.
                timed_out_seeds = [future_to_seed[f] for f in pending]
                for f in pending:
                    f.cancel()
                for seed in timed_out_seeds:
                    failed[seed] = FailedSeed(
                        seed=seed,
                        error_type="TimeoutError",
                        error_message=(
                            f"exceeded seed_timeout_seconds="
                            f"{seed_timeout_seconds}"
                        ),
                    )
                    logger.error(
                        "Seed %s timed out after %ss", seed,
                        seed_timeout_seconds,
                    )
                pending = set()
                break
            for future in done:
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    completed[seed] = result
                    logger.info(
                        "Seed %s completed: slope=%.4f p=%.4f (%.1fs)",
                        seed, result.slope, result.slope_pvalue,
                        result.elapsed_seconds,
                    )
                except BrokenExecutor:
                    # Pool died — every subsequent future.result()
                    # would also raise BrokenExecutor.  Halt rather
                    # than mark every remaining seed as a "failure"
                    # masking the infrastructure collapse.
                    raise
                except Exception as exc:  # noqa: BLE001
                    error_msg = str(exc) or repr(exc)
                    failed[seed] = FailedSeed(
                        seed=seed,
                        error_type=type(exc).__name__,
                        error_message=error_msg[:500],
                    )
                    logger.error(
                        "Seed %s failed: %s: %s",
                        seed, type(exc).__name__, error_msg,
                        exc_info=True,
                    )

    target_result = completed.pop(design.target_seed, None)
    if target_result is None:
        # Target failure is a panel-fatal error: without the target
        # there is nothing to anchor the analysis against.
        target_failure = failed.get(design.target_seed)
        detail = (
            f" ({target_failure.error_type}: {target_failure.error_message})"
            if target_failure else ""
        )
        raise RuntimeError(
            f"Target seed {design.target_seed!r} failed{detail}; "
            f"cannot build PanelResult.  Other failed seeds: "
            f"{sorted(failed.keys())}"
        )

    # Sort per_seed and failed_seeds by design order (target first
    # in design.selected_seeds()) for byte-deterministic JSON output.
    selected_order = list(design.selected_seeds())
    panel_seeds = tuple(
        completed[s] for s in selected_order if s in completed
    )
    failed_sorted = tuple(
        failed[s] for s in selected_order if s in failed
    )

    result = PanelResult(
        design=design,
        target_result=target_result,
        per_seed=panel_seeds,
        failed_seeds=failed_sorted,
    )

    result_path = output_dir / "result.json"
    result.save_json(result_path)
    logger.info("Wrote panel result to %s", result_path)

    return result
