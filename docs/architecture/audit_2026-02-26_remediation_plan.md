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

**Scope:** 10 findings, ~15 files modified

| Finding | File(s) | Description |
|---------|---------|-------------|
| S-2 | `validation/{id_mapping,entity_resolver,annotation_providers}.py`, `knowledge/cross_modal_mapper.py` | pickle → JSON cache migration |
| STAT-2 | `stats/differential.py`, `stats/permutation_framework.py` | Camera VIF inter-gene correlation adjustment |
| STAT-3 | `stats/enrichment_z.py` | z-score: population σ → SE of mean (coordinated with STAT-2) |
| STAT-15 | `stats/specificity.py`, `stats/validation_report.py`, CLI files | Expose hardcoded constants as CLI params |
| ARCH-2 | `knowledge/cogex.py` | QueryResult status enum, distinguish error from empty |
| ARCH-3 | `cli/_analyze_core.py` | Failure rate tracking + abort threshold |
| ARCH-4 | `knowledge/cogex.py` | Connection reset + reconnect logic |
| ARCH-6 | `cli/validate_baselines.py` | Phase checkpointing + resume |
| ARCH-13 | `knowledge/cogex.py` | LRU cache for gene resolution |
| ARCH-15 | `stats/validation_report.py` | Document verdict asymmetry |

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
