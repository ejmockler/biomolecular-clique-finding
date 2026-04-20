# Audit VIII — Findings and Solutions

**Date:** 2026-03-06
**Critics:** 5 agents (Claude) across ROAST stats, GPU permutation, KG coherence, validation, architecture
**Raw findings:** ~34 → **12 valid actionable** after dedup, rejection, and critical assessment

## Status: IN PROGRESS

---

## Cycle 1: Semantic/Statistical Fixes

### RC-VIII-4: Permutation p-value is graph-level but assigned per-community (HIGH→MEDIUM)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` lines 1169, 1193, 132-136
- **Problem:** `permutation_null()` computes a single p-value for whole-graph modularity. This same value is assigned to every `CommunityResult` via `permutation_pvalue=perm_pvalue_pos` (line 1169) and `perm_pvalue_neg` (line 1193). A small noisy 3-gene community gets the same p-value as a large dense community. `is_significant` (line 132) checks `self.permutation_pvalue < 0.05`, so weak communities appear significant just because the overall graph is.
- **Fix:** (a) Rename field to `graph_permutation_pvalue` in CommunityResult; (b) Update `is_significant` to require BOTH graph-level significance AND per-community density/size; (c) Add docstring clarifying this is graph-level.
- **Tests:** Verify `is_significant` requires density >= 0.5 even when graph p-value < 0.05.
- **Status:** [ ]

### VAL-VIII-6: Phase 1 status="failed" leads to "refuted" instead of "inconclusive" (MEDIUM)
- **File:** `src/cliquefinder/stats/validation_report.py` lines 167-172, 259, 325
- **Problem:** When Phase 1 has `status="failed"` (runtime error, not statistical failure), line 169 skips the if-block (`gate_adjusted` stays False). At line 259, `not gate_adjusted and not cov` is `True and False` = `False` (cov is a non-empty dict). If Phase 3 also fails, we fall to line 325: `verdict = "refuted"`. But a phase that errored ≠ statistical refutation — should be "inconclusive".
- **Fix:** Detect when both mandatory phases have `status="failed"` and route to "inconclusive" with appropriate summary.
- **Tests:** Test with Phase 1 status="failed" + Phase 3 absent → should be "inconclusive".
- **Status:** [ ]

### VAL-VIII-3: Degenerate strata permutation proceeds with no frozen_fraction metadata (MEDIUM)
- **File:** `src/cliquefinder/stats/label_permutation.py` lines 141-152
- **Problem:** When a stratum has only one condition value, the warning fires but permutation proceeds with those samples frozen (labels never change). The `LabelPermutationResult` has no field recording what fraction of samples were frozen, so users cannot assess severity.
- **Fix:** Add `frozen_fraction: float = 0.0` field to `LabelPermutationResult`; compute it in `generate_stratified_permutation` (return frozen count); thread through `run_label_permutation_null`.
- **Tests:** Verify frozen_fraction > 0 when degenerate strata exist; verify it appears in to_dict().
- **Status:** [ ]

### RC-VIII-1: _compute_corr doesn't handle NaN or diagonal — callers do it inconsistently (MEDIUM)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` lines 815-827 vs 878-890 vs 982-993
- **Problem:** The `_compute_corr` helper returns raw correlation without filling diagonal to 1.0 or zeroing NaN. Every caller (bootstrap_stability, permutation_null) must manually do both steps. NaN count logging includes diagonal NaN entries, overcounting.
- **Fix:** Make `_compute_corr` zero NaN and fill diagonal internally. Return count of NaN entries (excluding diagonal) for logging.
- **Tests:** Verify _compute_corr returns matrix with diagonal=1.0 and no NaN even for zero-variance input.
- **Status:** [ ]

---

## Cycle 2: Immutability & Safety Fixes

### ARCH-VIII-1: get_clique_data() returns writable slice of immutable array (MEDIUM)
- **File:** `src/cliquefinder/stats/experiment.py` line 147
- **Problem:** Fancy indexing on read-only array returns a writable copy. Callers can mutate the returned subset, breaking the immutability contract.
- **Fix:** Set `data_subset.flags.writeable = False` before returning.
- **Tests:** Verify returned array raises ValueError on write attempt.
- **Status:** [ ]

### ARCH-VIII-3: BioMatrix properties expose internal mutable arrays (MEDIUM)
- **File:** `src/cliquefinder/core/biomatrix.py` lines 181-186, 190-212
- **Problem:** `data` and `quality_flags` properties return direct references. Callers can mutate `matrix.data[0,0] = 999`.
- **Fix:** Set `self._data.flags.writeable = False` and `self._quality_flags.flags.writeable = False` in `__init__`, after validation.
- **Tests:** Verify `matrix.data[0,0] = X` raises ValueError.
- **Status:** [ ]

### NEG-VIII-1: NegativeControlResult.to_dict() can crash on empty control_competitive_z_scores (MEDIUM)
- **File:** `src/cliquefinder/stats/negative_controls.py` lines 115-127
- **Problem:** Guard checks `target_competitive_z is not None` but not whether `control_competitive_z_scores` is None or empty. `np.percentile` on empty array raises ValueError.
- **Fix:** Add guard: `if self.target_competitive_z is not None and self.control_competitive_z_scores is not None and len(self.control_competitive_z_scores) > 0`.
- **Tests:** Test to_dict with target_competitive_z set but control_competitive_z_scores=None and empty array.
- **Status:** [ ]

### RC-VIII-5: _match_communities mixes positive and negative communities (MEDIUM)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` lines 1433-1434
- **Problem:** Cross-condition matching combines `positive_communities + negative_communities` into one pool. A positive community in condition A can match a negative community in B based on gene overlap. This conflates co-activation with anti-correlation.
- **Fix:** Match positive-to-positive and negative-to-negative separately. Return both match types.
- **Tests:** Verify that positive communities only match positive, negative only negative.
- **Status:** [ ]

---

## Cycle 3: Minor Polish

### ARCH-VIII-4: BioMatrix.__repr__ crashes on empty matrix (LOW)
- **File:** `src/cliquefinder/core/biomatrix.py` lines 360-367
- **Problem:** `self.feature_ids[0]` raises IndexError when n_features=0.
- **Fix:** Guard with length check before indexing.
- **Tests:** Create empty BioMatrix and call repr().
- **Status:** [ ]

### RC-VIII-2: compute_community_stats uses O(n) list.index() per gene (LOW)
- **File:** `src/cliquefinder/knowledge/regulatory_coherence.py` line 785
- **Problem:** `gene_list.index(g)` is O(n) per lookup. Should use dict.
- **Fix:** Build `{gene: idx}` dict once, use O(1) lookups.
- **Tests:** Verify same results with dict-based lookup.
- **Status:** [ ]

### SPEC-VIII-1: specificity.py uses iloc with boolean mask (LOW)
- **File:** `src/cliquefinder/stats/specificity.py` line 204
- **Problem:** `covariates_df.iloc[mask]` works but `iloc` is designed for integer positions. Boolean mask should use `.loc` or `[mask]`.
- **Fix:** Change to `covariates_df.loc[mask]` or `covariates_df[mask]`.
- **Tests:** Existing tests should pass.
- **Status:** [ ]

### ARCH-VIII-7: Double DataFrame copy in prepare_experiment (LOW)
- **File:** `src/cliquefinder/stats/experiment.py` lines 444 and 116
- **Problem:** Factory copies DataFrame, then `__post_init__` copies again.
- **Fix:** Remove the copy in the factory (let __post_init__ handle it).
- **Tests:** Existing tests should pass.
- **Status:** [ ]

---

## Rejected Findings (with rationale)

| Finding | Reason |
|---|---|
| ROAST-1: C-matrix contrast normalization | P-values are invariant (scaling cancels). Active proportion threshold is a secondary diagnostic. Document but don't change the math — normalization improves MGS stability. |
| ROAST-2: Two-group path fragile Q2 | Already guarded to exactly 2-group designs (line 777-783). Would need relaxation to cause issues. |
| ROAST-3: Floormean UP/DOWN zeros opposite | Intentional design choice (SET-TEST-15), documented. Diverges from limma but is internally consistent. |
| ROAST-4: Mean50 weight normalization | Only affects non-uniform weights, which this codebase doesn't use. |
| ROAST-5: MSQ directional zeros | Documented as intentional (SET-TEST-15). |
| ROAST-6: L matrix hardcoded | Guarded to 2-group designs. |
| ROAST-7: n_rotations from config | n_valid_rotations field is correct; n_rotations is config metadata, not a bug. |
| ROAST-8: GPU/CPU transpose locations | Both produce correct final shape. Style difference only. |
| GPU-1: squeeze_var NaN propagation | Already addressed in Audit III (DATA-III-1). |
| GPU-2: trigamma abs(-dif/y) | The negation inside abs() is a no-op. Matches R limma convention. Not a bug. |
| GPU-3: np.linalg.inv vs solve | Conscious design choice: explicit inverse needed for batched beta = XtX_inv @ X' @ Y reuse. |
| GPU-4: c_var_factor < 0 abs | Regularization prevents this. Warning is appropriate. |
| VAL-2: Competitive z percentile semantics | Actually consistent with ROAST percentile (both: low = good). |
| VAL-4: Free fallback structured tracking | Details string annotation is sufficient. |
| KG-6: VarianceFilteredUniverse ddof=0 | Rank-preserving — no functional impact on gene selection. |
| KG-7: stability.py Pearson only | Separate module from regulatory_coherence.py. Low priority. |
| ARCH-5: _ID_MAPPING_CACHE thread safety | The cache is only called from main thread in practice. ProcessPoolExecutor (not Thread) is used. |
| ARCH-6: identify_disagreements O(M*N) | Performance only, not correctness. Small N in practice. |
| ARCH-8: cliques tuple mutable elements | Linked to CliqueDefinition frozen — too large blast radius for this audit. |
| ARCH-9: _analyze_core globals | Standard multiprocessing initializer pattern. Process-safe by design. |

---

## Implementation Plan

### Cycle 1: Semantic/Statistical (4 findings)
Files: `regulatory_coherence.py`, `validation_report.py`, `label_permutation.py`
- RC-VIII-4: Rename permutation_pvalue → graph_permutation_pvalue, fix is_significant
- VAL-VIII-6: Phase 1 status="failed" → inconclusive
- VAL-VIII-3: frozen_fraction in LabelPermutationResult
- RC-VIII-1: _compute_corr handle NaN/diagonal internally

### Cycle 2: Immutability & Safety (4 findings)
Files: `experiment.py`, `biomatrix.py`, `negative_controls.py`, `regulatory_coherence.py`
- ARCH-VIII-1: get_clique_data writeable=False
- ARCH-VIII-3: BioMatrix._data writeable=False
- NEG-VIII-1: to_dict empty guard
- RC-VIII-5: Match communities within same sign only

### Cycle 3: Minor Polish (4 findings)
Files: `biomatrix.py`, `regulatory_coherence.py`, `specificity.py`, `experiment.py`
- ARCH-VIII-4: __repr__ empty guard
- RC-VIII-2: dict-based lookup in compute_community_stats
- SPEC-VIII-1: iloc → loc
- ARCH-VIII-7: Remove double copy
