# Test Suite

The suite covers the production package from low-level numerical kernels to
CLI and pipeline integration. It contains 2,200+ tests across 100+
`test_*.py` modules. The collection command below is the source of truth as
the suite evolves.

## Running tests

Create the locked development environment and run the full suite:

```bash
uv sync --extra dev
uv run --no-sync pytest
```

Equivalent one-shot invocation:

```bash
uv run --extra dev pytest
```

Useful focused commands:

```bash
# Confirm collection and report the current test count
uv run --no-sync pytest --collect-only -q

# Exclude the explicitly marked calibration/benchmark tests
uv run --no-sync pytest -m "not slow"

# Run one module, directory, or named test
uv run --no-sync pytest tests/test_rotation.py
uv run --no-sync pytest tests/wasc/
uv run --no-sync pytest tests/test_rotation.py::TestRotationEngine
```

`slow` tests are included by default. Use `-m "not slow"` only for a quicker
development pass; run the complete suite before release or publication builds.

## Organization

- `conftest.py` contains shared deterministic fixtures and generators.
- Core, I/O, and quality tests cover matrix invariants, loading, transforms,
  outlier handling, and imputation.
- Statistical tests cover differential models, empirical Bayes moderation,
  ROAST, permutation engines, enrichment, graph rewiring, gradients, and
  numerical edge cases.
- Knowledge, CLI, and validation tests cover INDRA integration boundaries,
  argument validation, target snapshots, matching, negative controls, and
  report generation.
- Panel tests cover the multi-contrast landscape runner and selection logic.
- `wasc/` covers edge enumeration, preprocessing, FWL fits, concordance,
  matched nulls, calibration, Brown combination, and locked feasibility bounds.

Tests that require unavailable optional hardware or packages should use an
explicit pytest skip condition. Ordinary tests must be deterministic, assert
their outcomes directly, and avoid returning pass/fail booleans.

## Adding tests

Place tests beside the closest existing domain, name modules `test_*.py`, and
prefer focused assertions over printed diagnostics. Register any new custom
pytest marker in `pyproject.toml` before using it. For stochastic methods, use a
fixed seed and test statistical invariants with a justified tolerance rather
than retrying failures.
