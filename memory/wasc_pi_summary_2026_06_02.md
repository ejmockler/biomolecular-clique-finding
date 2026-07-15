# WASC pipeline — status memo for PI (2026-06-02)

> **SUPERSEDED INTERPRETATION (audited July 2026).** The retained calibration
> result is narrower than the memo below: this raw-p selector/WASC pipeline was
> unstable when SPOR was reduced from 294 to 25 (mean selected-set Jaccard
> 0.285 versus the frozen 0.70 gate). The gate blocked B=9999 and licenses no
> edge verdict; it does not establish universal structural underpowering at
> n=25. The current pathway state is bounded log2 measured-only **8/6/0**;
> legacy STRING and matched-RNA claims are withdrawn. Content below preserves
> the June decision trail and is not current publication authority.

## TL;DR

The WASC test (Within-cluster Anchor-Slope Concordance) ran through five pre-registered amendments today, surfaced two pre-existing arithmetic bugs that survived four prior audit gates, and produced **one substantive scientific result and one structural ceiling**:

- **Substantive**: under proper SPOR-cohort-size-matched sensitivity testing, the C9-vs-SPOR contrast at n=25 C9 donors **does not survive** the pre-registered ≥70% Jaccard overlap bound (observed 0.285). The C9 contrast pattern at full SPOR cohort depth (n=294) is partly explained by SPOR cohort size. **M2.5 prong (b) HARD FAILS** the pre-registered tripwire.
- **Structural ceiling**: a corrected arithmetic check shows that even with infinite per-edge effect size, only **rank 1–69 of 944 edges** can clear BY-FDR q≤0.10 under the pre-registered design at B=99999. The primary outcome (count + per-theme breakdown) is feasible but bounded.

The methodology itself is sound. The data is what it is.

---

## What WASC tests

For each of 944 pre-enumerated edges in three pathway clusters (Splicing 434, Chromatin 443, Transport 67 — with 40 cross-theme dupes), test whether the per-edge regression slope (target_y ~ anchor_y + Sex + Age + Tissue + Batch) is concordant across {C9, Sporadic, Control} donor groups via Cochran-Q. Calibrate against a matched-bin permutation null (degree × |Pearson| deciles).

**Pre-registered primary outcome**: count of WASC-positive edges (BY-FDR q ≤ 0.10) + per-theme breakdown.

**Pre-registered tripwire (HARD HALT)**: four sanity prongs must pass before the primary B=9999 run is licensed.

---

## Where we are (5 tagged amendments since this morning)

| Tag | Commit | What changed | Trigger |
|---|---|---|---|
| `v1.0` | 157f841 | M1 baseline freeze | initial pre-reg |
| `v1.0.1` | 7f1e10c | Enriched metadata (E10, E11) | metadata join discovered |
| `v1.0.2` | 4e04b79 | Dropped missingness axis (3-axis → 2-axis null) | AnswerALS source is RF-imputed upstream → axis empirically inert |
| `v1.0.3` | 0a9bff0 | Canonical primary substrate swap: theme-restricted → all-protein-pool | theme-restricted FAILED H0 calibration (FP=0.261 vs bound 0.120) due to sparse-cell sampling pathology in 42–190-protein clusters |
| `v1.0.4` | 59e52dc | 7 spec corrections from foundational audit + locked-bounds CI gate | foundational audit found 37 issues (3 CRITICAL); 4 rescue items deferred to v1.1 |

Each amendment followed the same protocol: review-gated workflow (3+ adversarial verifiers), signed decision log with SHA-256 pins, no-Q-exposure attestation.

---

## What's calibrated and what isn't

| Component | Status | Evidence |
|---|---|---|
| FWL kernel ≡ statsmodels.OLS (inline check) | **PASS** | code internally consistent; full prong (d) result reported below |
| M2.5 prong (a) — label-shuffle null FP | **PASS** | mean FP=0.1114 vs bound 0.1195 at production 20×B=999 (all-protein-pool substrate) — see `output/wasc/m2_5_prong_a_smoke/result.allpool_b999_production.json` |
| M2.5 prong (b) — SPOR n=25 down-sample Jaccard | **HARD FAIL** | Jaccard 0.285 ± 0.022 (3 seeds) vs pre-registered bound ≥0.70 — see `output/wasc/m2_5_prong_b_b999/seed{42,7,99}/summary.json` (note: artifacts use deduped 904 edges, not 944; see Open Item below) |
| M2.5 prong (d) — FW vs OLS production design | **PASS (in-workflow report)** | β abs 2.39e-10, SE rel 3.14e-10 — both 30–40× margin under spec 1e-8. Reported by workflow ``'s prong-d agent; numbers transcribed into `data/wasc/locked_bounds_v1.json` but **no on-disk result file** — needs separate re-run if PI wants an audit trail. |
| min_unique_q_values guard (constant-Q_null defense) | **COMMITTED** | inert under all-protein-pool (cells ~34 candidates; K=5 floor never triggered) |
| locked-bounds CI gate | **COMMITTED** | 4 tests; spec/code drift fails loud |

---

## What the tripwire FAIL means scientifically

Prong (b) asks: if we shrink the SPORADIC cohort to match C9 (n=25 instead of n=294), do we recover ≥70% of the full-N WASC-positive edges?

Observed: only ~40% of full-N raw-p positives replicate under SPOR-25 down-sampling. Stable across three random SPOR-25 draws (Jaccard std = 0.022). This is not seed noise; it is a real loss of power when SPOR is reduced to C9-cohort depth.

**Implication**: the C9-vs-SPOR contrast at full N is partly enabled by SPOR's larger n. Some WASC-positive edges in the full-N run may not be detectable in a cohort-size-matched comparison. The pre-registered ≥70% bound was specifically designed to detect this confounding pattern; it triggered as designed.

This is not a methodology bug. It is a **statistical power finding** about the C9 cohort. The methodology correctly identified that the cohort cannot support the pre-registered robustness test.

---

## Three honest paths from here

### Path A — Publish as INCONCLUSIVE
Document prong (b) FAIL as the substantive finding. The WASC primary count cannot be honestly reported when its pre-registered SPOR-size-matched robustness check fails. The publishable result is the methodology + the finding that the C9 n=25 cohort is structurally underpowered for this design.

**Pros**: most defensible; respects pre-registration discipline; clean scientific narrative.
**Cons**: no per-edge or per-pathway "WASC-positive" count to report.

### Path B — Pivot methodology
Replace the matched-bin permutation null with a hierarchical model that borrows strength across edges and themes (e.g., empirical Bayes shrinkage of per-edge betas, then joint test). This would not require SPOR-size matching as a robustness gate because the model would explicitly account for sample-size differences.

**Pros**: potentially produces a publishable WASC-style result on this cohort.
**Cons**: requires new pre-registration with its own audit gates; introduces model assumptions (random-effect prior shape, etc.) that need their own validation.

### Path C — Acquire more C9 samples
Wait for more C9 donors. With C9 n closer to SPOR n, prong (b) ≥70% may become satisfiable.

**Pros**: keeps the current methodology intact.
**Cons**: indefinite timeline.

---

## What this WASC exercise has produced regardless of which path you choose

1. **A pre-registered methodology** with documented arithmetic, full code coverage (168/168 tests), CI-enforced bound locking, and signed decision logs across 5 tagged amendments.

2. **A reusable bin-matched permutation framework** (`cliquefinder.stats.wasc`) for testing slope-concordance across donor groups within graph neighborhoods. The framework supports both theme-restricted and all-protein-pool variants and includes calibration tools (label-shuffle FP rate, F-W vs OLS identity, SPOR-downsample Jaccard).

3. **A foundational lesson** for future pre-registrations: audit gates that scope to "the amendment being made" miss arithmetic bugs in the unchanged base. The CI locked-bounds infrastructure shipped today (`tests/wasc/test_locked_bounds.py`) closes this gap for WASC; the pattern transfers to any future pre-registered test.

4. **An empirical map of where the C9 n=25 cohort can and cannot support inference**: it can support label-shuffle null calibration (prong (a)), it can support per-edge β/SE estimation (prong (d)), it cannot support cohort-size-matched robustness (prong (b)) under the pre-registered ≥70% bar.

---

## Known artifact-hygiene issues (surfaced 2026-06-04 by the Path A scaffold workflow)

The Path A publication-draft workflow (``, 4 drafters + 3 reviewers) returned **3/3 REJECT** verdicts and surfaced load-bearing artifact-hygiene problems in the existing record:

1. **`scripts/wasc/run_m2_5_prong_a_smoke.py` overwrites a single `result.json` per run.** The all-protein-pool production calibration (FP=0.1114 vs bound 0.1195, 20×B=999, task ``) was overwritten by a later theme-restricted re-stamp (FP=0.1302 vs bound 0.1313). The original production result has been **reconstructed** from the task log to `output/wasc/m2_5_prong_a_smoke/result.allpool_b999_production.json`; the overwriting file moved to `result.theme_b999_restamp_OVERWROTE_allpool.json`. Script needs per-config output paths.

2. **Prong (b) artifacts use deduped 904 edges, not 944.** `scripts/wasc/run_m2_5_prong_b_smoke.py` silently drops the 40 cross-theme duplicates via `drop_duplicates(subset=['edge_id'])`. This pre-empts the v1.0.4 deferred decision about which denominator (944 or 904) is binding. v1.1 must reconcile this with the script behavior before any production run.

3. **Prong (d) has no on-disk result file.** The β=2.39e-10 / SE=3.14e-10 numbers come from workflow ``'s prong-d agent (which wrote a `/tmp/wasc_prong_d.py` script and ran it inline) but the script + output were not promoted to the repo. Re-running prong (d) cleanly into `output/wasc/m2_5_prong_d/` is a follow-on task.

4. **Path A publication scaffold v0 NOT SAVED.** The first scaffold attempt produced a ~5000-word draft; review caught factual errors (FP number-stew, prong-(b) substrate misattribution, denominator mismatch, fabricated prong-d citation, forbidden-language contamination, prong-(c) status silently omitted, Path-A-vs-A/B/C scope contradiction). Saving the draft would propagate the errors. A careful corrected scaffold (`memory/wasc_path_a_v1.md`) was produced subsequently.

These artifact-hygiene issues are independent of the methodology + science findings above — but they illustrate why the locked-bounds CI gate shipped today matters: it would have caught problem #1 (script writes to a path the locked-bounds JSON pins) if extended to output artifacts.

---

## Open items deferred to v1.1 (separate audit gates required)

- **Promote B=9999 → B=99999** as primary (arithmetic correction shows B=9999 cannot resolve BY q≤0.10 at rank 1; B=99999 unlocks ranks 1–69).
- **Pre-register BY-FDR denominator** (944 with cross-theme dupes counted under each theme vs 904 unique edges) — choice has rescue-risk for prong (a) verdict.
- **Pre-register Gate 2 metric** (RAW-p vs BY-q) — RAW-p was the metric under which prong (a) PASSED; BY-q under which the verdict could differ. Asymmetric ratification not allowed in v1.0.4.
- **K-sweep tertiary diagnostics** for the min_unique_q_values guard — explicit v1.0.3 prohibition; needs re-tag.

---

## Recommended next step for you

Decide between Path A, B, or C. Each is defensible; each has costs.

If A: I can prepare a publishable methodology + findings document. Estimated 1–2 weeks.

If B: I can scope a v1.1 pre-registration for a hierarchical-model alternative. Estimated 2–4 weeks to design + audit-gate; then re-implement against the cohort.

If C: WASC pipeline is on ice pending cohort growth. Existing artifacts (5 tags, 168 tests, decision logs, locked-bounds JSON) stay in repo as reproducibility infrastructure.

I do not recommend a fourth path — "find a way to make the existing WASC pipeline produce a positive result" — because that would require either rescuing prong (b) (forbidden by v1.0.3 prohibitions without re-tag) or weakening the pre-registered bound (post-hoc threshold-relaxation). The audit gates this project committed to make those moves unavailable.
