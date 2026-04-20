# Architecture & Infrastructure Audit Findings

**Source:** Brutalist multi-agent review (Claude + Gemini), 2026-02-26
**Scope:** `src/cliquefinder/` — pipeline architecture, infrastructure resilience, concurrency

---

## Finding Index

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| ARCH-1 | [Silent method failure in run_method_comparison()](#arch-1-silent-method-failure) | CRITICAL | Valid |
| ARCH-2 | [DB connection failures become biological assertions](#arch-2-db-connection-failures-become-biology) | HIGH | Valid |
| ARCH-3 | [No failure threshold in parallel execution](#arch-3-no-failure-threshold-in-parallel-execution) | HIGH | Valid |
| ARCH-4 | [Dead Neo4j connection reused after failure](#arch-4-dead-neo4j-connection-reused) | HIGH | Valid |
| ARCH-5 | [Process pool memory multiplication](#arch-5-process-pool-memory-multiplication) | HIGH | Valid |
| ARCH-6 | [Validation orchestrator — no resume/checkpoint](#arch-6-no-resume-checkpoint) | HIGH | Valid |
| ARCH-7 | [method_comparison.py 2,819-line monolith](#arch-7-method-comparison-monolith) | MEDIUM | Valid |
| ARCH-8 | [Protocol abstraction leaks (isinstance on concrete)](#arch-8-protocol-abstraction-leaks) | MEDIUM | Valid |
| ARCH-9 | [Frozen dataclasses with mutable fields](#arch-9-frozen-dataclasses-with-mutable-fields) | HIGH | Valid |
| ARCH-10 | [Seed-addition anti-pattern](#arch-10-seed-addition-anti-pattern) | MEDIUM | Partially valid |
| ARCH-11 | [Non-atomic file writes](#arch-11-non-atomic-file-writes) | MEDIUM | Valid |
| ARCH-12 | [No CLI parameter bounds checking](#arch-12-no-cli-bounds-checking) | MEDIUM | Valid |
| ARCH-13 | [O(n) unbuffered external API calls in gene resolution](#arch-13-unbuffered-gene-resolution) | HIGH | Valid |
| ARCH-14 | [Unbounded Neo4j result sets](#arch-14-unbounded-neo4j-results) | HIGH | Valid |
| ARCH-15 | [Confirmation bias in verdict structure](#arch-15-confirmation-bias-in-verdict) | HIGH | Valid |
| ARCH-16 | [Sequential inner loop defeats GPU acceleration](#arch-16-sequential-loop-defeats-gpu) | MEDIUM | Valid |
| ARCH-17 | [Verdict over-conservative downgrade for single-contrast](#arch-17-verdict-over-conservative-downgrade) | MEDIUM | Valid |
| ARCH-18 | [Mixed warnings.warn vs logger.warning](#arch-18-mixed-warning-patterns) | LOW | Valid |

---

## ARCH-1: Silent Method Failure in run_method_comparison()

### Status: VALID — CRITICAL (statistical validity impact)

### Location
`stats/method_comparison.py:2704-2707`

### Problem
```python
except Exception as e:
    if verbose:
        print(f"FAILED: {e}")
    results_by_method[method.name] = []  # Silently empty
```

When a statistical method crashes (singular matrix, convergence failure, etc.), its results are set to `[]`. Concordance metrics are computed over only the surviving methods. The output reports "N methods compared" without indicating a method failed.

### Impact
- User sees concordance results that appear to validate findings when a disagreeing method crashed
- Method failure is only visible in verbose console output (not in the returned data structures)
- Downstream code processes the empty list without error, producing incomplete comparison

### Solution
```python
# Track failures in the returned result
failed_methods: dict[MethodName, str] = {}

for method in methods:
    try:
        results = method.test(experiment)
        results_by_method[method.name] = results
    except Exception as e:
        failed_methods[method.name] = str(e)
        results_by_method[method.name] = []
        logger.warning(f"Method {method.name.value} failed: {e}")

# Include failures in MethodComparisonResult
result = MethodComparisonResult(
    ...,
    failed_methods=failed_methods,
)

# In MethodComparisonResult.summary():
if self.failed_methods:
    lines.append(f"WARNING: {len(self.failed_methods)} methods failed:")
    for name, err in self.failed_methods.items():
        lines.append(f"  {name.value}: {err}")
```

### Pitfalls
- Some methods are expected to fail for certain designs (e.g., LMM with no replicates) — distinguish expected vs unexpected failures
- Add a `min_methods` parameter: if fewer than N methods succeed, raise an error rather than producing incomplete concordance
- Ensure the wide_format DataFrame has NaN columns (not missing columns) for failed methods

### Priority: P0 — Fix this week

---

## ARCH-2: DB Connection Failures Become Biological Assertions

### Status: VALID — HIGH

### Location
`knowledge/cogex.py:535, 638, 956` (and all query methods)

### Problem
Neo4j query methods wrap Cypher execution in `try... except Exception`. On any exception (timeout, connection drop, network partition), the function logs the error and returns an empty list `[]`. The caller interprets `[]` as "this gene has zero downstream targets" and continues.

### Impact
- Infrastructure failure → false biological conclusion
- "TP53 has zero targets" because the DB timed out → pipeline computes empty cliques → results say TP53 is not a regulator
- The error IS logged, but the pipeline continues as if the result is valid

### Solution

**Option A: Distinguish "no results" from "query failed" (recommended)**
```python
@dataclass
class QueryResult:
    edges: list[INDRAEdge]
    status: Literal["success", "error", "timeout"]
    error_message: str | None = None

def get_downstream_targets(self, regulator, ...) -> QueryResult:
    try:
        results = client.query_tx(query, ...)
        return QueryResult(edges=results, status="success")
    except Exception as e:
        logger.error(f"Query failed for {regulator}: {e}")
        return QueryResult(edges=[], status="error", error_message=str(e))
```

Callers then check `result.status` before proceeding:
```python
result = client.get_downstream_targets(gene)
if result.status != "success":
    raise RuntimeError(f"INDRA query failed for {gene}: {result.error_message}")
```

**Option B: Raise exceptions, let callers decide**
Remove the try/except from query methods entirely. Let the Neo4j exception propagate. Callers that want retry/skip behavior can add their own handling.

**Option C: Add retry with exponential backoff**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _execute_query(self, query, **params):
    return self._get_client().query_tx(query, **params)
```

### Pitfalls
- **Option A** requires updating every caller of `get_downstream_targets` — many callsites
- **Option B** is the cleanest but may break existing pipelines that rely on silent failure
- **Option C** adds a dependency (tenacity) but is the most robust for production use
- Consider a circuit breaker: after N consecutive failures, skip all remaining queries rather than retrying each one

### Priority: P1 — Fix this month

---

## ARCH-3: No Failure Threshold in Parallel Execution

### Status: VALID — HIGH

### Location
`cli/_analyze_core.py:545-557` (thread path), `cli/_analyze_core.py:870-882` (process path), `cli/_analyze_core.py:949-950` (sequential path)

### Problem
Per-regulator exceptions are caught and logged, but no aggregate tracking occurs:
```python
except Exception as e:
    logger.warning(f"Failed for {regulator_name}: {e}")
    return None  # or: continue
```

If 90% of regulators fail (e.g., Neo4j connection dies), the pipeline returns results for 10% with a clean exit code.

### Impact
- Systematic failure (bad connection, API change, corrupted data) → partial results treated as complete
- No distinction between "40 regulators, all succeeded" and "400 regulators, 360 failed"

### Solution
```python
n_total = len(regulators)
n_failed = 0
n_succeeded = 0
failure_reasons: dict[str, int] = defaultdict(int)

for future in as_completed(futures):
    try:
        result = future.result()
        if result is not None:
            n_succeeded += 1
            all_results.append(result)
        else:
            n_failed += 1
    except Exception as e:
        n_failed += 1
        failure_reasons[type(e).__name__] += 1

# Abort if too many failures
failure_rate = n_failed / n_total if n_total > 0 else 0
if failure_rate > 0.5:
    raise RuntimeError(
        f"Analysis aborted: {n_failed}/{n_total} regulators failed "
        f"({failure_rate:.0%}). Reasons: {dict(failure_reasons)}"
    )
elif failure_rate > 0.1:
    logger.warning(
        f"{n_failed}/{n_total} regulators failed ({failure_rate:.0%}). "
        f"Results may be incomplete."
    )

# Include failure statistics in output
result_summary.n_attempted = n_total
result_summary.n_succeeded = n_succeeded
result_summary.n_failed = n_failed
```

### Pitfalls
- The 50% abort threshold is somewhat arbitrary — expose as a CLI parameter
- Some regulators may legitimately have no targets (not a failure) — distinguish "no results" from "exception thrown"
- The failure counting must work correctly with all three execution paths (thread, process, sequential)

### Priority: P1 — Fix this month

---

## ARCH-4: Dead Neo4j Connection Reused After Failure

### Status: VALID — HIGH

### Location
`knowledge/cogex.py:502-514`

### Problem
`_get_client()` initializes `self._client` once lazily. If the connection drops mid-query, the exception is caught in the query method, but `self._client` is **never reset to None**. The next query reuses the dead connection.

`ping()` exists but is never called internally before queries. No reconnection logic, no circuit breaker.

### Solution
```python
def _get_client(self, force_reconnect=False):
    if self._client is not None and not force_reconnect:
        return self._client
    # Create new connection
    self._client = IndraOntology.from_neo4j(...)
    return self._client

def _execute_query_with_reconnect(self, query, **params):
    try:
        return self._get_client().query_tx(query, **params)
    except (ConnectionError, TimeoutError, neo4j.exceptions.ServiceUnavailable) as e:
        logger.warning(f"Connection lost, attempting reconnect: {e}")
        self._client = None
        return self._get_client(force_reconnect=True).query_tx(query, **params)
```

### Pitfalls
- Need to identify the specific Neo4j exception types (depends on the driver version)
- Don't retry indefinitely — cap at 2-3 attempts
- Thread safety: if multiple threads share a CoGExClient, the reconnection must be synchronized

### Priority: P1 — Fix this month

---

## ARCH-5: Process Pool Memory Multiplication

### Status: VALID — HIGH

### Location
`cli/_analyze_core.py:182-211`, `cli/_analyze_core.py:834-847`

### Problem
Each `ProcessPoolExecutor` worker receives a full copy of the expression matrix via pickle serialization. The worker then reconstructs a BioMatrix and creates a new CliqueValidator (which precomputes correlation matrices).

For 20k features × 500 samples:
- Base matrix per process: ~80 MB
- Correlation cache per process: potentially hundreds of MB
- With `n_workers=4`: 1.2+ GB just for duplicated data

### Solution

**Option A: Shared memory (Python 3.8+)**
```python
from multiprocessing.shared_memory import SharedMemory

# Create shared memory for the expression matrix
shm = SharedMemory(create=True, size=data.nbytes)
shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shared_data[:] = data[:]

# Workers access shared memory by name
def worker_fn(shm_name, shape, dtype, regulator):
    existing_shm = SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    # data is read-only (no copy)
    ...
```

**Option B: Memory-mapped arrays**
```python
import tempfile
tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
np.save(tmp.name, data)

# Workers load via memory mapping (OS handles sharing)
def worker_fn(mmap_path, regulator):
    data = np.load(mmap_path, mmap_mode='r')  # Read-only, shared
    ...
```

**Option C: Reduce worker count based on available memory**
```python
import psutil
available_mem = psutil.virtual_memory().available
per_worker_mem = data.nbytes * 2  # Conservative estimate (data + correlation cache)
max_workers = max(1, int(available_mem * 0.7 / per_worker_mem))
n_workers = min(n_workers, max_workers)
```

### Pitfalls
- **Option A**: SharedMemory requires manual cleanup (`shm.close()`, `shm.unlink()`)
- **Option B**: Memory-mapped arrays are read-only; workers cannot modify the data (which is correct for analysis)
- **Option C**: Simple but doesn't actually reduce memory per worker — just limits total memory
- CliqueValidator precomputation (correlation matrices) is per-condition and should be computed once and shared, not per-worker
- ThreadPoolExecutor shares memory natively (no copy) but is limited by GIL for CPU-bound work

### Priority: P2 — Fix for production use

---

## ARCH-6: Validation Orchestrator — No Resume/Checkpoint

### Status: VALID — HIGH

### Location
`cli/validate_baselines.py:163`

### Problem
The orchestrator creates a fresh `ValidationReport()` on every run. Phase results are saved to JSON after each phase, but there is no mechanism to:
1. Load previously completed phase results on restart
2. Skip already-completed phases
3. Resume Phase 3 permutations from an intermediate point

Phase 3 runs 500+ label permutations (each calling `run_protein_differential`). If killed at permutation 400, all 400 are lost.

### Solution
```python
def run_validate_baselines(args):
    # Load existing report if resuming
    report_path = args.output / "validation_report.json"
    if report_path.exists() and not args.force_restart:
        report = ValidationReport.load(report_path)
        print(f"Resuming from existing report ({len(report.phases)} phases complete)")
    else:
        report = ValidationReport()

    # Skip completed phases
    if "covariate_adjusted" not in report.phases:
        # Phase 1
        ...

    if "specificity" not in report.phases and len(contrasts) > 1:
        # Phase 2
        ...

    # Phase 3 with intermediate checkpointing
    if "label_permutation" not in report.phases:
        checkpoint_path = args.output / "phase3_checkpoint.npz"
        start_perm = 0
        null_z_scores = np.full(n_permutations, np.nan)

        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path)
            null_z_scores[:len(checkpoint['z'])] = checkpoint['z']
            start_perm = int(checkpoint['completed'])
            print(f"  Resuming from permutation {start_perm}")

        for i in range(start_perm, n_permutations):
            null_z_scores[i] = ...

            if (i + 1) % 100 == 0:
                np.savez(checkpoint_path, z=null_z_scores[:i+1], completed=i+1)
```

### Pitfalls
- Phase results must be fully JSON-serializable for loading — NumPy arrays need `.tolist()`
- Checkpointing must preserve RNG state to maintain reproducibility on resume — save `rng.bit_generator.state`
- "Completed" detection must be robust: a phase that failed (`"status": "failed"`) should not be skipped on resume
- Add `--force-restart` flag to override resume behavior

### Priority: P1 — Fix this month

---

## ARCH-7: method_comparison.py 2,819-Line Monolith

### Status: VALID — MEDIUM

### Location
`stats/method_comparison.py`

### Problem
Single file containing: enums, 5 dataclasses, 1 Protocol, PreparedCliqueExperiment + factory, 4 method implementations (OLS, LMM, ROAST, Permutation), concordance computation, MethodComparisonResult (433-line class), and the main entry point. The file documents its own splitting plan (lines 53-63) with 7 extraction targets.

### Impact
- Merge conflict magnet: any change touches the same file
- Testing burden: import of any component imports all components
- Cognitive load: 2,819 lines of context to understand any single method

### Solution
Follow the documented splitting plan at lines 53-63:
1. `method_comparison_types.py`: Enums, dataclasses, Protocol (~450 lines)
2. `experiment.py`: PreparedCliqueExperiment + prepare_experiment (~415 lines)
3. `methods/ols.py`: OLSMethod (~190 lines)
4. `methods/lmm.py`: LMMMethod (~220 lines)
5. `methods/roast.py`: ROASTMethod (~240 lines)
6. `methods/permutation.py`: PermutationMethod (~200 lines)
7. `concordance.py`: concordance functions + MethodComparisonResult (~720 lines)
8. `method_comparison.py`: run_method_comparison + re-exports (~300 lines)

### Pitfalls
- **All importers must be updated** — grep for `from .method_comparison import` and `from cliquefinder.stats.method_comparison import`
- The re-export file (item 8) should maintain backward compatibility:
  ```python
  # method_comparison.py — re-exports for backward compatibility
  from .method_comparison_types import *
  from .experiment import *
  from .concordance import *
  from .methods.ols import OLSMethod
  ...
  ```
- OLSMethod and LMMMethod share ~80% code — extract a `_BaseLinearMethod` before splitting
- Test imports must be updated — use the re-export module for test stability

### Priority: P2 — Split this month

---

## ARCH-8: Protocol Abstraction Leaks (isinstance on Concrete)

### Status: VALID — MEDIUM

### Location
`stats/permutation_framework.py:453, 597, 641`

### Problem
The code uses `isinstance(design, MetadataDerivedDesign)` (concrete class check) instead of Protocol-based dispatch. `MetadataDerivedDesign.derive_conditions()` is not part of the `ExperimentalDesign` Protocol, forcing concrete class checks wherever conditions need derivation.

### Solution
Add `derive_conditions` to the Protocol (as an optional method with default implementation):

```python
@runtime_checkable
class ExperimentalDesign(Protocol):
    @property
    def condition_column(self) -> str: ...

    def sample_mask(self, metadata: pd.DataFrame) -> NDArray[np.bool_]: ...

    def get_conditions(self, metadata: pd.DataFrame) -> NDArray:
        """Return condition labels. Default: read from condition_column."""
        return metadata[self.condition_column].values
```

Then `MetadataDerivedDesign` overrides `get_conditions()` with its derivation logic, and callers use the Protocol method instead of isinstance:

```python
# Instead of:
if isinstance(design, MetadataDerivedDesign):
    conditions = design.derive_conditions(metadata).values
else:
    conditions = metadata[design.condition_column].values

# Use:
conditions = design.get_conditions(metadata)
```

### Pitfalls
- Adding methods to a Protocol is a breaking change for any external implementors
- The `get_conditions()` default implementation must handle the common case correctly
- Test all 3 concrete ExperimentalDesign implementations after the change

### Priority: P3 — Refactor when touching the permutation framework

---

## ARCH-9: Frozen Dataclasses with Mutable Fields

### Status: VALID — HIGH

### Location
Multiple files: `rotation.py`, `method_comparison.py`, `permutation_framework.py`

### Problem
`@dataclass(frozen=True)` classes contain `NDArray`, `dict`, and `list` fields. `frozen=True` prevents attribute reassignment (`obj.x = new_value`) but NOT mutation of the referenced object (`obj.x[0] = new_value`, `obj.x.append(...)`, `obj.x['key'] = val`).

Affected classes:
- `RotationPrecomputed`: `Q2: NDArray`
- `GeneEffects`: `U, rho_sq, gene_ids: list, sample_variances`
- `PreparedCliqueExperiment`: `data: NDArray, feature_to_idx: dict, clique_to_feature_indices: dict`
- `UnifiedCliqueResult`: `method_metadata: dict`
- `TestResult`: `additional: dict`

### Impact
- Violates the reproducibility contract these dataclasses advertise
- Creates concurrency hazards if objects are shared between threads/processes
- Currently safe because code doesn't appear to mutate these objects, but no enforcement

### Solution

**Option A: Defensive copy on construction (recommended)**
```python
@dataclass(frozen=True)
class PreparedCliqueExperiment:
    data: NDArray[np.float64]

    def __post_init__(self):
        # Make arrays read-only
        object.__setattr__(self, 'data', self.data.copy())
        self.data.flags.writeable = False
```

For dicts and lists, use `types.MappingProxyType` and `tuple`:
```python
def __post_init__(self):
    from types import MappingProxyType
    object.__setattr__(self, 'feature_to_idx',
                       MappingProxyType(dict(self.feature_to_idx)))
    object.__setattr__(self, 'method_metadata',
                       MappingProxyType(dict(self.method_metadata)))
```

**Option B: Document the contract without enforcement**
Add a docstring note: "Fields are conceptually immutable. Do not mutate contained objects."

### Pitfalls
- `__post_init__` on frozen dataclasses requires `object.__setattr__` (the frozen attribute prevents normal assignment)
- `MappingProxyType` is read-only but still references the underlying dict — if the original dict is modified elsewhere, the proxy sees the changes. Use `MappingProxyType(dict(...))` to copy first.
- Making NumPy arrays non-writeable (`flags.writeable = False`) prevents accidental mutation but can be bypassed with `arr.flags.writeable = True`
- The copy adds memory overhead — for `PreparedCliqueExperiment.data` this can be significant (full expression matrix). Consider making it a view with read-only flag instead of a copy.

### Priority: P2 — Fix for high-traffic dataclasses (PreparedCliqueExperiment, UnifiedCliqueResult)

---

## ARCH-10: Seed-Addition Anti-Pattern

### Status: PARTIALLY VALID — MEDIUM

### Location
`cli/validate_baselines.py:175-186`

### Problem
Phase seeds are generated by adding constants to a base seed:
```python
_seed_phase3_strat = _base_seed + 1000
_seed_phase3_free = _base_seed + 2000
```

The Gemini review claimed this "guarantees highly correlated random streams" for MT19937. However, **NumPy's `default_rng()` uses PCG64** (not MT19937), which has a 128-bit state space. Adjacent integer seeds for PCG64 do NOT produce correlated streams (unlike MT19937). The PCG64 generator is specifically designed to handle sequential seeds well.

### Revised Assessment
- For PCG64: The current approach is **acceptable** — sequential seeds produce independent streams
- For MT19937: The critique would be valid — MT19937 has known issues with correlated streams from nearby seeds
- Using `SeedSequence` is still best practice and more robust across generator backends

### Solution (best practice, not urgent)
```python
from numpy.random import SeedSequence

ss = SeedSequence(_base_seed)
child_seeds = ss.spawn(5)
_seed_bootstrap = child_seeds[0]
_seed_phase3_strat = child_seeds[1]
_seed_phase3_free = child_seeds[2]
_seed_phase4 = child_seeds[3]
_seed_phase5 = child_seeds[4]

# In each phase:
rng = np.random.default_rng(_seed_phase3_strat)
```

### Pitfalls
- `SeedSequence.spawn()` returns `SeedSequence` objects, not integers — pass them directly to `default_rng()`
- This changes all random sequences (different results from same base seed) — breaking change for reproducibility
- The current approach is functional and correct with PCG64; this is a future-proofing improvement

### Priority: P3 — Adopt SeedSequence in next major version

---

## ARCH-11: Non-Atomic File Writes

### Status: VALID — MEDIUM

### Location
~20+ locations using `to_csv()`, `json.dump()`, `np.save()` directly

### Problem
All output writes use direct file operations with no temp-file-then-rename pattern. If the process crashes mid-write (OOM, kill signal, disk full), the output file is left in a corrupted partial state. On resume, the corrupted file may be loaded as valid input.

### Solution
```python
import tempfile, os

def atomic_write_json(path, data):
    """Write JSON atomically via temp file + rename."""
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=dir_path, suffix='.tmp', delete=False
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)  # Atomic on POSIX
```

### Priority: P3 — Apply to checkpoint and report files first

---

## ARCH-12: No CLI Parameter Bounds Checking

### Status: VALID — MEDIUM

### Location
`cli/validate_baselines.py`, `cli/analyze.py`, `cli/differential.py`

### Problem
Numeric CLI parameters accept arbitrary values: `--n-rotations -5`, `--alpha 2.0`, `--label-permutations 999999999`.

### Solution
Add validation in `register_parser()`:
```python
parser.add_argument(
    "--n-rotations", type=int, default=9999,
    choices=range(100, 100001),  # argparse validates automatically
    metavar="N",
)
# Or use a custom type:
def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

parser.add_argument("--n-rotations", type=positive_int, default=9999)
parser.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold (0 < alpha < 1)")
```

Add post-parsing validation:
```python
if not 0 < args.alpha < 1:
    parser.error(f"--alpha must be between 0 and 1, got {args.alpha}")
```

### Priority: P3 — Add bounds for safety-critical parameters (alpha, n_rotations, seed)

---

## ARCH-13: O(n) Unbuffered External API Calls in Gene Resolution

### Status: VALID — HIGH

### Location
`knowledge/cogex.py:712-727` (`discover_regulators`), `cogex.py:883-960` (`resolve_gene_name`)

### Problem
Gene resolution loops over every gene calling `hgnc_client.get_current_hgnc_id()` sequentially, with no caching. The same genes are resolved redundantly in multiple code paths. `resolve_gene_name()` has a fallback to MyGene.info where `mygene.MyGeneInfo()` is instantiated fresh per call.

### Impact
- 10,000+ external calls for a large gene universe
- No connection pooling for MyGene.info
- 15-minute latency wall for large datasets before any analysis begins

### Solution
Add an LRU cache to `resolve_gene_name`:
```python
from functools import lru_cache

@lru_cache(maxsize=50000)
def resolve_gene_name(self, gene_name: str) -> GeneId | None:
    ...
```

Or use a module-level cache dict:
```python
_gene_resolution_cache: dict[str, GeneId | None] = {}

def resolve_gene_name(self, gene_name: str) -> GeneId | None:
    if gene_name in _gene_resolution_cache:
        return _gene_resolution_cache[gene_name]
    result = self._resolve_gene_name_uncached(gene_name)
    _gene_resolution_cache[gene_name] = result
    return result
```

For MyGene.info, create a single client instance:
```python
_mygene_client = None

def _get_mygene_client():
    global _mygene_client
    if _mygene_client is None:
        import mygene
        _mygene_client = mygene.MyGeneInfo()
    return _mygene_client
```

### Priority: P1 — Fix this month (major performance impact)

---

## ARCH-14: Unbounded Neo4j Result Sets

### Status: VALID — HIGH

### Location
`knowledge/cogex.py:748-755`

### Problem
`discover_regulators()` fires a single Cypher query with up to 20k CURIEs and loads all matching edges into memory at once. No pagination, no result size limit.

### Solution
1. Add pagination to large queries
2. Add a hard limit with warning:
   ```python
   MAX_RESULTS = 100_000
   results = client.query_tx(query, target_ids=target_curies, limit=MAX_RESULTS)
   if len(results) >= MAX_RESULTS:
       logger.warning(f"Query returned {MAX_RESULTS} results (limit reached). "
                      "Some edges may be missing. Increase --min-evidence to reduce results.")
   ```
3. For very large gene universes, chunk the CURIEs:
   ```python
   CHUNK_SIZE = 5000
   all_results = []
   for i in range(0, len(target_curies), CHUNK_SIZE):
       chunk = target_curies[i:i+CHUNK_SIZE]
       results = client.query_tx(query, target_ids=chunk, ...)
       all_results.extend(results)
   ```

### Priority: P2 — Fix before production deployment

---

## ARCH-15: Confirmation Bias in Verdict Structure

### Status: VALID — HIGH

### Location
`stats/validation_report.py:100-238`

### Problem
The verdict structure is asymmetric:
- **"Validated"**: Phase 1 AND Phase 3 pass (+ at least one supplementary pass)
- **"Refuted"**: Phase 1 AND Phase 3 BOTH fail
- **Everything else**: "Inconclusive"

The path to "refuted" requires both mandatory gates to fail simultaneously. A truly null signal must beat BOTH tests to be called refuted, while a truly positive signal only needs both to pass. Combined with the positive correlation between Phase 1 and Phase 3 (same data, same gene set), the framework is structurally biased toward "inconclusive" rather than "refuted."

Additionally, Phase 2 classifies results as "specific" or "shared" — both are positive outcomes. Only "inconclusive" is negative for Phase 2.

### Solution

**Option A: Symmetric verdict logic**
```python
if gate_adjusted and gate_permutation:
    verdict = "validated"
elif not gate_adjusted and not gate_permutation:
    verdict = "refuted"
elif gate_adjusted != gate_permutation:
    verdict = "inconclusive"  # One gate passes, one fails
```

**Option B: Add explicit "refuted" path for single-gate failure**
If Phase 3 (label permutation) fails significantly (p > 0.5, not just > alpha), upgrade "inconclusive" to "refuted" — the signal is not just non-significant but actively null-consistent.

**Option C: Document the asymmetry honestly**
```python
"""
Note: This framework is designed for VALIDATION of a candidate finding,
not for balanced hypothesis testing. The verdict is asymmetric by design:
- "Validated" requires positive evidence from multiple complementary tests
- "Refuted" requires strong evidence of absence from both mandatory gates
- "Inconclusive" is the default for ambiguous or mixed results

For formal two-sided hypothesis testing, use the individual phase p-values
with appropriate multiplicity correction.
"""
```

### Pitfalls
- **Option A** changes the scientific interpretation of the framework — discuss with domain collaborators
- **Option B** requires choosing a threshold for "actively null-consistent" (e.g., p > 0.5) — another researcher degree of freedom
- **Option C** is the most honest approach if the framework is intended for validation rather than discovery

### Priority: P1 — At minimum, document the asymmetry (Option C). Consider Option A/B for next major version.

---

## ARCH-16: Sequential Inner Loop Defeats GPU Acceleration

### Status: VALID — MEDIUM

### Location
`stats/permutation_gpu.py:940-944`

### Problem
The "vectorized" random index precomputation is actually a Python for-loop:
```python
for i in range(total_samples):
    all_samples[i] = rng.choice(pool_size, size=size, replace=False)
```

For 831 cliques × 1000 permutations = 831,000 sequential `rng.choice()` calls. Additionally, `rotation.py:1086` forces a GPU→CPU transfer in the hot loop.

### Solution
Replace sequential sampling with vectorized:
```python
# Vectorized sampling without replacement (approximation for small size/pool_size ratios)
random_values = rng.random((total_samples, pool_size))
all_samples = np.argpartition(random_values, size, axis=1)[:, :size]
```

For the GPU→CPU transfer, batch transfers across rotations instead of per-rotation.

### Priority: P3 — Performance optimization

---

## ARCH-17: Verdict Over-Conservative Downgrade for Single-Contrast

### Status: VALID — MEDIUM

### Location
`stats/validation_report.py:185-192`

### Problem
If both mandatory gates pass but ALL supplementary phases either fail or show no signal:
```python
if supplementary_total > 0 and supplementary_pass == 0:
    verdict = "inconclusive"
```

For single-contrast datasets (Phase 2 doesn't apply) with small samples (Phase 4 matching fails), this can downgrade to "inconclusive" despite passing both strongest statistical gates.

### Solution
Count skipped phases separately from failed phases:
```python
supplementary_failed = supplementary_total - supplementary_pass
supplementary_skipped = expected_supplementary_total - supplementary_total

if supplementary_failed > 0 and supplementary_pass == 0 and supplementary_skipped == 0:
    verdict = "inconclusive"  # All supplementary phases ran AND failed
elif supplementary_total == 0 or supplementary_pass > 0:
    verdict = "validated"  # No supplementary phases ran, or at least one passed
```

### Priority: P2 — Fix this month

---

## ARCH-18: Mixed warnings.warn vs logger.warning

### Status: VALID — LOW

### Location
286 `logger.*` calls across 21 files, plus scattered `warnings.warn` calls

### Problem
Inconsistent use of Python's two warning systems:
- `warnings.warn()`: user-facing, can be filtered, captured by pytest
- `logger.warning()`: developer-facing, goes to log handlers

### Solution
Adopt a convention:
- `warnings.warn()`: For issues the **end user** should see (e.g., deprecated parameters, low sample size, convergence warnings)
- `logger.warning()`: For issues **developers/operators** should see (e.g., missing data, fallback behavior, retry logic)

### Priority: P4 — Standardize opportunistically when touching files
