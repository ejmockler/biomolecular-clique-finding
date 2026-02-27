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

**Scope:** 7 findings, 9 files modified, ~2,000 lines changed

| Finding | File(s) | Agent | Description |
|---------|---------|-------|-------------|
| STAT-4 | `stats/differential.py` | stat-satterthwaite | Containment df primary + Satterthwaite refinement |
| STAT-5 | `stats/rotation.py` | stat-rotation | Q2 sign: contrast projection replaces correlation heuristic |
| STAT-8 | `stats/validation_report.py` | stat-docs-controls | Bounded-FWER docstring with ρ-dependent bivariate formula |
| STAT-9 | `stats/negative_controls.py` | stat-docs-controls | Cache matched gene sets for paired z-score/p-value computation |
| ARCH-9 | `stats/rotation.py`, `stats/permutation_framework.py`, `stats/method_comparison.py` | arch-verdict-pagination | MappingProxyType + read-only arrays + list→tuple |
| ARCH-14 | `knowledge/cogex.py` | arch-verdict-pagination | Chunked CURIE queries (5000/batch) + max_results limit |
| ARCH-17 | `stats/validation_report.py` | arch-verdict-pagination | Skipped vs failed supplementary phase distinction |

**Agents (4 parallel, isolated worktrees):**
- `stat-satterthwaite`: STAT-4 — containment df (SAS PROC MIXED default) with Satterthwaite refinement
- `stat-rotation`: STAT-5 — Q2 sign via `Q2[:,0] @ (X @ c)` projection, both simple and general paths
- `stat-docs-controls`: STAT-8 + STAT-9 — docstring formula + cached matched gene sets
- `arch-verdict-pagination`: ARCH-9 + ARCH-14 + ARCH-17 — frozen immutability, chunked queries, verdict logic

**Manual review between waves:** Verified containment df formula `n_subjects - p` is correct per SAS PROC MIXED. Confirmed Satterthwaite refinement accepted only when `>= containment_df * 0.5`. Verified Q2 projection `Q2[:,0] @ Xc` is nonzero because Xc is not in reduced model's column space. Confirmed STAT-9 caching eliminates RNG advancement mismatch. Resolved 2 merge conflicts (cogex.py: combined ARCH-14 chunking with ARCH-4 `_execute_query()` reconnection; validation_report.py: took STAT-8 formula over Wave 2 partial fix). Updated Wave 2 test assertion for STAT-8 docstring refinement.

**Status: COMPLETE** — All 7 sub-findings fixed. 87 new tests (Wave 3), 247 cumulative audit tests, 599 total tests passing.

#### Wave 3 Completion Log

| Finding | Status | Tests Added | Manual Review Notes |
|---------|--------|-------------|---------------------|
| STAT-4 | Fixed | 19 tests (balanced, unbalanced, bounds, containment fallback, backward compat) | Containment df `n_groups - n_params` as primary. Satterthwaite refinement only when `>= containment * 0.5`. Diagnostic warning on clipping. 8 pre-existing tests still pass. |
| STAT-5 | Fixed | 21 tests (balanced groups, UP/DOWN consistency, MSQ invariance, multi-group, near-degenerate) | Projection `Q2[:,0] @ Xc` is deterministic regardless of group balance. Both general and simple paths updated. 28 pre-existing rotation tests pass. |
| STAT-8 | Fixed | 6 tests (bounded-FWER language, rho formula, quantitative bounds, no alpha^2 claim) | Full bivariate normal formula with ρ-dependence. Quantitative bounds: ρ∈[0.3,0.7] → FWER ≈ 0.006-0.020. Updated Wave 2 test that expected earlier phrasing. |
| STAT-9 | Fixed | 7 tests (gene set pairing, call count, RNG reproducibility, edge cases) | `matched_gene_sets` list caches first-pass gene sets. Second pass uses `enumerate(matched_gene_sets)`. No extra RNG advancement. |
| ARCH-9 | Fixed | 23 tests (read-only arrays, MappingProxyType, tuple conversion, FrozenInstanceError, construction) | `RotationPrecomputed`, `GeneEffects`, `PreparedCliqueExperiment`, `UnifiedCliqueResult`, `TestResult` all enforced. `to_dict()` and `is_valid` work through MappingProxyType. |
| ARCH-14 | Fixed | 4 tests (single chunk, multi-chunk, max_results warning, empty CURIEs) | `CURIE_CHUNK_SIZE = 5000`. Combined with `_execute_query()` for auto-reconnection. `max_results=100_000` default with warning. |
| ARCH-17 | Fixed | 7 tests (single-contrast validated, no supplementary, all fail, mixed, gates fail) | Three-way branch: `supplementary_total==0` → validated, `failed>0 && pass==0` → inconclusive, else → validated. |

**New findings from manual review:**
- STAT-4-NOTE: Satterthwaite clipping warning fires for unbalanced designs where `df_satt_raw > n_obs - 1`. This is expected behavior (Satterthwaite can exceed residual df when random effects dominate) but the warning is informative. No code fix needed.
- ARCH-9-NOTE: `MappingProxyType` is a read-only view but not deeply immutable (nested mutable objects in dict values are still mutable). Acceptable for current usage — no nested mutable values observed in practice.

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
