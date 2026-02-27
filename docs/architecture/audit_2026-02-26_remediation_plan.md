# Audit Remediation Plan — 2026-02-26

## Wave Structure

### Wave 1: Critical Security & Statistical Correctness (P0)

**Scope:** 4 findings, 6 files modified, ~300 lines changed

| Finding | File(s) | Agent | Description |
|---------|---------|-------|-------------|
| S-1 | `viz/differential.py:377` | security | `eval()` → `ast.literal_eval()` |
| S-3a | `viz/differential.py:380` | security | Narrow bare `except:` → `(ValueError, SyntaxError)` |
| S-3b | `viz/cliques.py:210` | security | Narrow bare `except:` → `(ValueError, KeyError)` |
| S-3c | `quality/outliers.py:719` | security | Narrow bare `except:` → `(ValueError,)` |
| S-3d | `stats/differential.py:826` | stats | Narrow bare `except:` + add logging |
| STAT-1 | `stats/differential.py:480-535,1164-1171` | stats | Per-pattern grouped OLS replacing shared `(X'X)^-1` |
| ARCH-1 | `stats/method_comparison.py:2704-2707` | arch | Failed method tracking in `MethodComparisonResult` |

**Agents:**
- `security`: S-1 + S-3(a,b,c) — surgical edits + tests
- `stats`: STAT-1 + S-3d — mathematical OLS fix + bare except in same file
- `arch`: ARCH-1 — method failure tracking + tests

**Manual review between waves:** Verify mathematical correctness of STAT-1, run full test suite, check for regressions.

**Status: COMPLETE** — All 7 sub-findings fixed. 56 new tests, 129 existing tests passing.

#### Wave 1 Completion Log

| Finding | Status | Tests Added | Manual Review Notes |
|---------|--------|-------------|---------------------|
| S-1 | Fixed | 7 tests (malicious string rejection, ast.literal_eval parsing) | Clean — surgical replacement |
| S-3a | Fixed | 5 tests (narrowed except, KeyboardInterrupt propagation) | Correct exception types for ast.literal_eval |
| S-3b | Fixed | 4 tests (degenerate distance matrix, Jaccard fallback) | FloatingPointError covers pdist edge cases |
| S-3c | Fixed | 5 tests (student-t logpdf, extreme residuals) | scipy RuntimeWarning not caught (correct — it's a warning, not exception) |
| S-3d | Fixed | 2 tests (source check, typed except) | logger.warning added with %s formatting |
| STAT-1 | Fixed | 19 tests (bias detection, GPU/CPU agreement, edge cases) | Per-pattern grouped OLS correct. EB uses median_df — pragmatic approximation. Inline squeeze_var matches original formula. GPU path now pure NumPy (correctness > speed). |
| ARCH-1 | Fixed | 14 tests (failure tracking, summary, backward compat) | `failed_methods: dict[str,str]` with default_factory=dict for backward compat |

**New findings from manual review:**
- STAT-1-OPT: GPU path no longer uses MLX — future optimization can restore MLX for the fast path (all features same pattern = no NaN). P4 priority.
- The `squeeze_var()` in `permutation_gpu.py` only accepts scalar `df` — not a problem now (inline formula used) but should be unified in Wave 4. P4 priority.

---

### Wave 2: Statistical Integrity & Infrastructure Resilience (P1)

**Scope:** 10 findings, 12 files modified, ~2,500 lines changed

| Finding | File(s) | Agent | Description |
|---------|---------|-------|-------------|
| S-2 | `validation/{id_mapping,entity_resolver,annotation_providers}.py`, `knowledge/cross_modal_mapper.py` | cache-security | pickle → JSON cache migration |
| STAT-2 | `stats/enrichment_z.py`, `stats/differential.py` | stat-z-vif | Camera VIF inter-gene correlation adjustment |
| STAT-3 | `stats/enrichment_z.py` | stat-z-vif | z-score: population σ → SE of mean |
| STAT-15 | `cli/validate_baselines.py`, `stats/validation_report.py` | pipeline-hardening | Expose hardcoded constants as CLI params |
| ARCH-2 | `knowledge/cogex.py` | neo4j-resilience | Error routing through `_execute_query()` |
| ARCH-3 | `cli/_analyze_core.py` | pipeline-hardening | Failure rate tracking + abort threshold (50%) + warn (10%) |
| ARCH-4 | `knowledge/cogex.py` | neo4j-resilience | Connection reset + single-retry reconnect |
| ARCH-6 | `cli/validate_baselines.py` | pipeline-hardening | Phase checkpointing + `--force-restart` resume |
| ARCH-13 | `knowledge/cogex.py` | neo4j-resilience | Dict cache for gene resolution + mygene singleton |
| ARCH-15 | `stats/validation_report.py` | pipeline-hardening | Honest bounded-FWER analysis replacing α² claim |

**Agents (4 parallel, isolated worktrees):**
- `stat-z-vif`: STAT-2 + STAT-3 — Camera VIF formula + SE denominator correction
- `neo4j-resilience`: ARCH-2 + ARCH-4 + ARCH-13 — query reconnection, gene cache, mygene singleton
- `pipeline-hardening`: ARCH-3 + ARCH-6 + ARCH-15 + STAT-15 — failure thresholds, checkpointing, configurable verdict
- `cache-security`: S-2 — pickle deserialization elimination across 4 cache files

**Manual review between waves:** Verified Camera VIF formula `1 + (k-1)*ρ̄` matches Wu & Smyth 2012. Confirmed SE denominator `σ/√k` for proper calibration. Verified checkpoint/resume logic with phase-skip guards. Confirmed no file conflicts between branches. Re-installed editable package after worktree contaminated Python path.

**Status: COMPLETE** — All 10 sub-findings fixed. 104 new tests (Wave 2), 160 cumulative audit tests, 513 total tests passing.

#### Wave 2 Completion Log

| Finding | Status | Tests Added | Manual Review Notes |
|---------|--------|-------------|---------------------|
| S-2 | Fixed | 36 tests (JSON roundtrip, set serialization, corruption recovery, migration) | Tagged set wrapper `{"__set__": True, "values": [...]}` for JSON. Sorted values for deterministic output. Error handling for corrupt cache files. |
| STAT-2 | Fixed | 20 tests (VIF formula, scaling, backward compat) | `VIF = 1 + (k-1)*ρ̄` correct per Camera paper. ρ̄ floored at 0 (negative correlation is conservative). `estimate_inter_gene_correlation()` uses off-diagonal mean of `np.corrcoef`. |
| STAT-3 | Fixed | (included in STAT-2 tests) | SE = `σ/√k` instead of σ. k=1 fallback preserves original behavior. Robust mode updated analogously: `MAD/√k * √VIF`. |
| STAT-15 | Fixed | (included in pipeline tests) | `--specificity-z-threshold`, `--negative-control-percentile`, `--interaction-n-perms` CLI args with sensible defaults. `neg_ctrl_percentile` threaded to `compute_verdict()`. |
| ARCH-2 | Fixed | (included in neo4j tests) | Both `get_downstream_targets` and `discover_regulators` route through `_execute_query()`. |
| ARCH-3 | Fixed | 17 tests (failure threshold, abort, warning) | 50% abort, 10% warn. Tracks `n_failed` across all 3 executor paths (ProcessPool, ThreadPool, sequential). |
| ARCH-4 | Fixed | 17 tests (reconnection, retry, error classification) | Heuristic keyword-based connection error detection. Single retry with fresh client. Non-connection errors re-raised immediately. |
| ARCH-6 | Fixed | (included in pipeline tests) | `validation_checkpoint.json` in output dir. `_load_checkpoint()` / `_save_checkpoint()`. `--force-restart` removes stale checkpoint. Phase-skip guards on all 5 phases. |
| ARCH-13 | Fixed | (included in neo4j tests) | `self._gene_cache: Dict[str, Optional[GeneId]]` on `INDRAModuleExtractor`. `_get_mygene_client()` lazy singleton. `resolve_gene_name()` → `_resolve_gene_name_uncached()` split. |
| ARCH-15 | Fixed | (included in pipeline tests) | Replaced misleading α² independence claim with honest bounded-FWER analysis + "Design asymmetry note" documenting conservative P1+P3 gate structure. |

**New findings from manual review:**
- W2-INSTALL: Agent worktrees that run `pip install -e .` can contaminate the main repo's editable install path. Future waves should avoid `pip install -e .` in worktrees, or the orchestrator should re-install after merge. Operational note, not a code fix.
- ARCH-4-NOTE: Connection error detection is heuristic (keyword matching on error string). Could miss exotic Neo4j errors or false-positive on query data containing "timeout"/"connection". Acceptable for now; could formalize with Neo4j exception class hierarchy in Wave 4. P4 priority.

---

### Wave 3: Statistical Refinements & Architecture Hardening (P2)

| Finding | Description |
|---------|-------------|
| STAT-4 | Satterthwaite df: containment df fallback |
| STAT-5 | Q2 sign convention: contrast projection method |
| STAT-8 | FWER docstring correction |
| STAT-9 | Cache matched control sets |
| ARCH-9 | Frozen dataclass immutability enforcement |
| ARCH-14 | Neo4j result pagination |
| ARCH-17 | Single-contrast verdict fix |

---

### Wave 4: Polish & Documentation (P3/P4)

| Finding | Description |
|---------|-------------|
| STAT-6 | Increase specificity permutations, report null_corr |
| STAT-10 | Document p-value sidedness conventions |
| STAT-11 | GPU float precision documentation + `--exact-precision` flag |
| STAT-12 | Inverse-variance weighting for subject aggregation |
| STAT-13 | FDR docstring revision |
| STAT-14 | Rotation negative variance: exclude instead of truncate |
| ARCH-5 | Shared memory for process pool |
| ARCH-7 | method_comparison.py monolith split |
| ARCH-8 | Protocol abstraction: add `get_conditions()` |
| ARCH-10 | SeedSequence adoption |
| ARCH-11 | Atomic file writes for checkpoints |
| ARCH-12 | CLI parameter bounds checking |
| ARCH-16 | Vectorized random index sampling |
| ARCH-18 | Standardize warnings vs logging |

---

## Execution Protocol

1. **Agent execution** with isolated worktrees per agent
2. **Manual review** — verify correctness, run tests, check for missed edge cases
3. **Document** — update audit docs with completion status, capture new findings
4. **Structure next wave** — adjust scope based on what manual review uncovered
