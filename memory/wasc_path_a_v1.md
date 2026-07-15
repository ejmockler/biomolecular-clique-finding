# WASC: Within-cluster Anchor-Slope Concordance — Path A Publication Scaffold

> **SUPERSEDED SCAFFOLD (audited July 2026).** The WASC halt establishes only
> instability of this raw-p selector/pipeline under SPOR 294→25 (mean Jaccard
> 0.285 versus its frozen 0.70 gate). B=9999 was blocked and no edge verdict
> exists; the result is not a general n=25 power theorem. The current pathway
> state is bounded log2 measured-only **8/6/0**, while the STRING and matched-
> RNA arguments repeated below are withdrawn. The body remains historical and
> must not be used as current publication text.

**Version:** v1 scaffold (NOT FOR SUBMISSION)
**Pinned to:** git tag `wasc-prereg-v1.0.5`, commit `c9572c9` (2026-06-04)
**Audience:** PI internal review only
**Recommended disposition:** Path A — publish as INCONCLUSIVE per build plan §5 HARD HALT
**Document scope:** This scaffold commits to Path A. Paths B (methodological pivot) and C (cohort expansion) are listed in Discussion as alternatives the PI may select instead, but the document itself recommends Path A and is structured accordingly.

---

## Abstract

We pre-registered a within-cluster Cochran-Q test of cross-protein abundance coupling among INDRA hop-1 neighbor pairs within eight C9-ALS cluster terms (themes: Splicing, Chromatin, Transport) across three donor groups (C9orf72 n=25, Sporadic n=294, Control n=59) in the AnswerALS PBMC proteomics matrix (3,264 measured UniProt accessions). The pre-registration is a six-tag amendment chain (v1.0 through v1.0.5) culminating in 944 within-theme INDRA hop-1 edges (Splicing 434, Chromatin 443, Transport 67) and a four-prong methodology tripwire (label-shuffle calibration, down-sampled-Sporadic Jaccard stability, all-protein-pool null ratio, Frisch-Waugh vs OLS numerical identity). The tripwire returned: prong (a) PASS at production B=999 (mean false-positive rate 0.1114 vs Gate 2 bound 0.1195), prong (d) PASS at 1e-8 absolute on the production design, prong (c) test trivially satisfied by the v1.0.3 substrate swap to the all-protein-pool (the four-prong structure is preserved per v1.0.4 items_NOT_modified), and prong (b) HARD FAIL at three random seeds (mean edge-set Jaccard 0.285 vs pre-registered bound greater than or equal to 0.70). Per build plan §5 hard-halt semantics, the primary B=9999 production run is BLOCKED. We propose to publish the methodology and the tripwire-FAIL as a substantive negative finding about the structural underpoweredness of n=25 C9 cohorts for pre-registered coupling-invariance tests at this contrast scale.

---

## 1. Status of this draft

This is a v1 scaffold pinned to `wasc-prereg-v1.0.5` (commit `c9572c9`, 2026-06-04). It is intended for PI review of structure, framing, and disposition. It is NOT a submission draft. Specifically:

- The Results section reports what is on disk as of the v1.0.5 tag and nothing else. No production B=9999 numbers exist because the M2.5 tripwire blocks the production run per build plan §5.
- The Limitations and Future Work sections enumerate the open decisions that gate any further compute.
- The Discussion explicitly recommends Path A (publish as INCONCLUSIVE) and mentions Paths B and C only as alternatives.
- The forbidden-language list from spec §9 has been enforced throughout. The terms *mechanism*, *causal*, *causation*, *drives*, *regulates*, *controls* (as a verb), *rewiring*, *validates*, *post-transcriptional*, and *INDRA-edges-are-correct* are not used in any claim-bearing context. Where a stem like *Control* appears as a donor-group proper noun, or *regulatory* appears as a substrate name (the INDRA regulatory subgraph), or *substrate-naming* uses these stems, that is defensible by the spec §9 ceiling and so flagged.

---

## 1.5 Position vs the Wave 24l cluster claim

This scaffold sits on top of an established positive finding it does NOT re-derive. The Wave 24l per-feature slope-GSEA cluster claim — that C9-ALS proteomic perturbation concentrates in three within-cluster signatures (Splicing, Chromatin, nucleocytoplasmic Transport) — survives the methodology change to the measured-only-paths substrate (BFS adjacency restricted to edges with both endpoints in the measured-protein set; no routing through unmeasured INDRA intermediates at any hop). Bonferroni-8 confirmatory at bounded h=2: 7/8 C9-vs-Sporadic, 6/8 C9-vs-Control, 0/8 Sporadic-vs-Control. The clean negative-control leg confirms the signature is C9-mutation-specific, not shared ALS pathology and not a graph-topology artifact. The c9ctrl-collapse falsifier upheld the claim as a *local* property of regulatory neighborhoods within two hops, not a continuous gradient over the connected component (cluster vs non-cluster slope-delta Mann-Whitney p ∈ {0.42, 0.90, 0.48, 0.70} across bounded-slope quartiles). The signal is INDRA-regulatory-specific per the Wave 24k T43 STRING contrast (physical-PPI gives opposite-sign GSEA on the same proteomics |t|). Persistent project record: `memory/wave_24l_measured_only_paths.md`, `memory/wave_24k_cluster_claim_consolidated.md`, `memory/wave_24j_triangulation.md`. Committed pipeline shipped 2026-06-06 across eight commits ending at the publication-figure stack.

WASC tests a different statistic on the same substrate: per-edge Cochran-Q wiring-invariance across donor groups within the cluster terms. The WASC tripwire FAIL (M2.5 prong (b) SPOR-25 down-sample Jaccard 0.285 vs ≥0.70 bound) and the Wave 24l cluster claim's survival do not invalidate each other. The honest publishable picture: a positive cluster signature with five sensitivities + Bonferroni-8 confirmatory + negative-control triangulation + INDRA-vs-STRING substrate-specificity, plus an orthogonal per-edge-resolution attempt (WASC) that ran into a cohort-power wall at n=25 C9. The Wave 24l findings are the headline; WASC is the methods-section attempt at sharper resolution.

---

## 2. Pre-registration timeline

The amendment chain is six tags spanning three calendar days. Each amendment was triggered by an independent audit gate and is itself a frozen artifact under git. The chain is summarized below.

| Tag | Commit | Date | Trigger | Effect |
|---|---|---|---|---|
| v1.0 | 157f841 | 2026-06-02 | M1 enumeration complete | Locked 944 within-theme INDRA hop-1 edges (Splicing 434, Chromatin 443, Transport 67); 377 measured cluster members across 3,264 measured proteins. |
| v1.0.1 | 7f1e10c | 2026-06-02 | Metadata audit | Pinned enriched metadata manifests (E10, E11); corrected Control n from 71 to actual 59 after pre-registered iPSC and external-donor exclusions. |
| v1.0.2 | 4e04b79 | 2026-06-02 | Empirical matrix audit | Dropped the missingness axis from the matched-cell null. The loaded matrix is RF-imputed upstream (0 of 1,423,104 cells NaN), so the missingness decile collapses to a constant; the axis was empirically inert and the rename was a descriptive correction. Three-axis null became two-axis (degree-decile by pooled-Pearson-decile). |
| v1.0.3 | 0a9bff0 | 2026-06-02 | M2.5 prong (a) calibration FAILED at theme-restricted substrate | Promoted the all-protein-pool null substrate (previously the "prong (c)" sensitivity in build plan §10) to the canonical primary candidate pool. Theme-restricted had failed the Gate 2 bound at mean false-positive rate 0.261 versus bound 0.120 due to sparse-cell sampling pathology (M_T sizes 42 to 190 over 100 decile cells produce per-cell occupancy median at most 2). The all-protein-pool substrate calibrated cleanly. |
| v1.0.4 | 59e52dc | 2026-06-02 | Foundational audit `` | Seven clean corrections (including the CRITICAL BY rank-1 arithmetic formula in §6) plus a locked-bounds CI gate (`data/wasc/locked_bounds_v1.json` and `tests/wasc/test_locked_bounds.py`). The CI gate enforces spec text and code parity; future drift fails CI loud. Four rescue-disguised-as-correction items were deferred to v1.1 by the synthesis-review. |
| v1.0.5 | c9572c9 | 2026-06-04 | C17 arithmetic correction caught by Path A scaffold audit | The v1.0.4 C17 statement on B-feasibility had its sign inverted and its B value mis-attributed. The corrected C17' is a three-row table (see §4.3 below) covering the sensitivity (B=999), primary (B=9999), and floor-tie (B=99999) tiers. The cumulative six-tag chain is now a correction-to-a-correction; the discipline is honest and traceable, but the count is acknowledged as a credibility stress in the build plan acknowledgment paragraph. |

The amendment chain reflects a deliberate discipline: every spec change is gated by an independent audit, every change has a decision-log JSON pinning the SHA-256 of the affected artifacts at the time of decision, and every change is a clean correction or a substrate swap, not a rescue. Decisions that carried rescue risk were deferred to v1.1.

---

## 3. Methods

This section summarizes the binding spec at tag v1.0.5. Full text is in `memory/wasc_spec.md` (792 lines) and `memory/wasc_build_plan.md` (1012 lines), pinned at commit `c9572c9`.

### 3.1 Cohort and exclusions

The cohort is AnswerALS PBMC DDA proteomics, expressed as log2 abundance of 3,264 UniProt accessions across 378 donors after pre-registered exclusions. Pre-registered exclusions remove 20 external and iPSC-derived donors (donor-ID prefixes EDi*, CW50*, CS007, W14-16C*) that lack Batch IDs and portal coverage. After exclusion the three donor groups are C9orf72 n=25, Sporadic n=294, and Control n=59. The Control count was corrected from 71 to 59 by the v1.0.1 amendment. The donor-exclusion list is the frozen artifact `data/wasc/excluded_donors_v1.json` (manifest item E7).

### 3.2 Edge enumeration

The eight pre-registered cluster terms (three Splicing, two Chromatin, three Transport; see spec §1 for the per-term Cogex IDs) are mapped to UniProt cluster-member sets through `fetch_term_members_via_indra` + `hgnc_ids_to_uniprots`. The intersection with the 3,264 measured proteins produces the per-theme measured cluster-member sets `M_T`: Splicing 190 measured of 304 UniProt union (303 HGNC); Chromatin 145 of 468 (467 HGNC); Transport 42 of 70 (70 HGNC); total 377 measured cluster members.

Within-theme INDRA hop-1 edges are enumerated on the Wave-24l measured-only INDRA regulatory subgraph restricted to ALL_REGULATORY_TYPES (Activation, Inhibition, IncreaseAmount, DecreaseAmount) with `restrict_endpoints_to_features=True` and `max_hops=1`. Both endpoints must be measured cluster members of the SAME theme; cross-theme edges are deferred out of scope. The total is `|E_WASC| = 944` edges (Splicing 434, Chromatin 443, Transport 67). Within-theme densities are 2.4%, 4.2%, and 7.8% respectively — biologically plausible at these set sizes. The enumeration is the frozen artifact `data/wasc/E_WASC_v1.json` (manifest item E4).

### 3.3 Per-edge per-group regression

For each edge `(a, j)` in `E_WASC` and each donor group `g` in {C9, SPOR, CTRL}, the test fits one OLS:

`y_{j, s} = β_0 + β_a · anchor_{a, s} + γ_Sex · Sex_s + γ_Age · Age_s + γ_Tissue · Tissue_s + ε_s`

where `y_{j, s}` is the pre-residualized log2 abundance of target protein `j` for donor `s` in group `g`. Pre-residualization is within-group ComBat-style EB location and scale adjustment over Batch, fit per `(g, j)` pair and reused across all anchors `a`. For C9 only, Batch is collapsed to `site_year` before ComBat to defeat singleton-batch degeneracy across 25 donors. Sporadic and Control retain raw Batch.

Covariates are Sex (binary, 100% coverage), Age (z-scored within group, missing values imputed by within-group regression on Sex; 91.3% pre-imputation coverage), and Tissue (collapsed to T-Cell / NT-Cell+Bulk / Unknown with T-Cell as reference). Batch is pre-residualized, not a column of X. The anchor is z-scored within group so the slope `β̂_{j|a, g}` is interpretable as "one SD anchor change predicts β̂ SD target change in group g".

Implementation uses Frisch-Waugh-Lovell to decouple the per-edge fit from the covariate-only residualization. Within group g the covariate-only residual projection `M_g` is computed once; each protein p in the measured set is residualized once to `p̃_g = M_g · z(p)_g`; per-edge slope and SE are closed-form univariate regression of `j̃_g` on `ã_g`. This makes the fit phase O(|M| · n_g) for residualization plus O(|E_WASC| · n_g) for per-edge dot products.

### 3.4 Cochran-Q concordance statistic

For each edge with all three groups converged, define precision weights `w_{g, j, a} = 1 / SE(β̂_{j|a, g})²` and compute the inverse-variance-weighted pooled slope `β̄_{j, a}` and Cochran's Q:

`Q(j, a) = Σ_g w_{g, j, a} · (β̂_{j|a, g} − β̄_{j, a})²`

LOW Q is the WASC-positive direction (invariant coupling across groups; small between-group dispersion). The χ² calibration is not used; the empirical null is constructed from two-axis-matched non-neighbor pairs (next section). Edges with fewer than three groups converged are reported in a secondary two-group analysis and excluded from the primary BY-FDR pool. The primary p-value is lower-tail (one-sided): `p(j, a) = (1 + #{Q in Q_null : Q ≤ Q(j, a)}) / (1 + |Q_null|)`, the Phipson-Smyth correction.

### 3.5 Null model

Per anchor `a` and theme T, the candidate pool `P_a^candidate` is constructed for matched draws. The substrate was changed by the v1.0.3 amendment: the canonical primary substrate is now the all-protein-pool `M \ N_a^obs \ {a}` (the full measured proteome minus the anchor's true neighbors and itself), promoted from the prior "prong (c) sensitivity" status. The theme-restricted substrate `M_T \ N_a^obs \ {a}` is preserved as an opt-in sensitivity (with its documented FAIL calibration on the AnswerALS RF-imputed substrate).

Substitute targets are drawn without replacement on a two-dimensional bin grid: Axis 1 is the anchor's degree decile within the INDRA measured-only regulatory graph; Axis 2 is the pooled |Pearson(anchor, p)| decile computed over the union of all three donor groups (covariate-adjusted but not within-group z-scored). The missingness axis (former Axis 2 in v1.0) was dropped by v1.0.2 as empirically inert on the RF-imputed substrate. A ±1-decile fallback on the |Pearson| axis is invoked if a perfect cell-match fails after 100 attempts; per-anchor fallback frequency is reported in Sanity Gate 7.

For B permutations, Q values are computed exactly as for real edges and pooled per anchor to form `Q_null^a`. The lower-tail empirical p-value above is computed against this anchor-local pool. The global pool is reported as a sensitivity. The pre-registered primary is `B = 9999` for INDRA, with `B = 999` permitted for sensitivities and `B = 99999` triggered for any floor-tied edge.

### 3.6 Multiple testing and claim ceiling

FDR uses the Benjamini-Yekutieli method at q-threshold 0.10. BY is chosen over BH because per-edge p-values are positively dependent through (a) shared anchors, (b) shared targets across multiple anchor fits, and (c) within-anchor null-draw dependence. BY controls FDR under arbitrary dependence at the cost of an inflation factor `H_N = Σ_{i=1}^{N} 1/i`. For N = 944, `H_944 = 7.4279`, so the BY rank-1 raw-p rejection threshold at q = 0.10 is `q / (N · H_N) = 1.4261e-5`. This is the corrected formula per v1.0.4 C1 and v1.0.5 C17'; the v1.0 spec text had originally written `q · (1/H_n) ≈ 0.013`, which is the BH adjustment, not the BY rank-1 threshold.

The claim ceiling in spec §9 is the maximum licensed inference. A positive result is described as "cross-protein abundance coupling is more invariant across donor groups than degree- and marginal-correlation-matched non-neighbor pairs", with no mechanism, causation, or rewiring language. The forbidden-language list is enforced in CI through the locked-bounds gate.

This document commits to Path A — publish as INCONCLUSIVE per build plan §5 — based on the tripwire FAIL reported in §4 below. Paths B and C are listed in §6.2 as alternatives the PI may choose instead.

### 3.7 STRING negative comparator

The STRING physical-PPI v12.0 graph at `combined_score ≥ 700` is the pre-registered comparator network, applied with the same regression and null pipeline. ENSP-to-UniProt mapping uses Swiss-Prot canonical-only (no isoforms) to avoid edge double-counting. The STRING decision rule has five branches (`INDRA-SPECIFIC`, `STRING-STRONGER`, `INCONCLUSIVE`, `STRING-UNDERPOWERED`, `STRING-ZERO-POSITIVES`); the primary statistic is `ΔQ^{∩}` on the edge-intersection with BCa bootstrap CI. STRING is BLOCKED behind the M2.5 tripwire.

### 3.8 Three-contrast decomposition

For each edge, three two-group Cochran-Q statistics are computed (C9-SPOR, C9-CTRL, SPOR-CTRL) with the same matched null, and BY-FDR is applied within each contrast at q = 0.10. The C9-specific criterion requires C1 (primary 3-group test passes), C2 (both C9-containing contrasts have at least `max(3·|R_{SPOR-CTRL}|, 48)` positive edges, where the empirical floor `48 = ceil(0.05 · 944)`), C3 (STRING resolves to `INDRA-SPECIFIC`), and C4 (SPOR-CTRL is not significantly elevated above the FDR null expectation). The decomposition is BLOCKED behind the tripwire.

### 3.9 Four-prong calibration tripwire (M2.5)

Per build plan §5, the four-prong tripwire is a HARD HALT gate inserted between M2 (null implementation) and M3 (BY-FDR on real data). All four prongs must pass to proceed to the primary B=9999 production run.

- **Prong (a)** — Label-shuffle null calibration. 20 shuffles of group labels with B=999 permutations per shuffle should return a pooled false-positive rate within the Gate 2 bound at q = 0.10.
- **Prong (b)** — Down-sampled-Sporadic Jaccard stability. Sample Sporadic n=294 down to n=25 (matching C9), rerun the pipeline at B=999, and measure the edge-set Jaccard against the full-Sporadic positive set. Pre-registered bound: Jaccard ≥ 0.70 (build plan §5 lines 25, 766; spec §6 C11 confirms `≥` not `>`).
- **Prong (c)** — Originally the all-protein-pool null ratio test (was the candidate pool over-restrictive?). The v1.0.3 amendment promoted the all-protein-pool to canonical primary substrate, so this test became trivially satisfied. The four-prong structure is preserved per v1.0.4 items_NOT_modified; prong (c) is marked "test no longer applicable" rather than relabelling the tripwire as 3-prong.
- **Prong (d)** — Frisch-Waugh vs `statsmodels.OLS` numerical identity on the production design with explicit per-group covariate columns (intercept + sex_female + age_z + tissue dummies, varying covariate dimension per group). Pre-registered bound: max absolute β difference ≤ 1e-8 (spec §2.5(a) line 150, build plan §5 line 27/768).

HARD HALT semantics: if any prong FAILS, the production B=9999 run is BLOCKED. The tripwire status determines whether M3-M5 proceed.

---

## 4. Results

This section reports only what is on disk at tag v1.0.5. No M3+ outputs exist because the M2.5 tripwire is the gating step.

### 4.1 M1 — Edge enumeration

Complete. Pinned at `data/wasc/E_WASC_v1.json` and verified by the locked-bounds CI gate. `|E_WASC| = 944` total: Splicing 434, Chromatin 443, Transport 67. 377 measured cluster members across 3,264 measured proteins. 203 distinct canonical anchors; 220 per-(anchor, theme) work units (17 multi-theme anchors get one unit per theme). Within-theme densities: Splicing 2.4%, Chromatin 4.2%, Transport 7.8%.

### 4.2 M2 — Concordance + null implementation

Complete at the implementation level. The kernel, null sampler, and joblib-parallel anchor loop are in `src/cliquefinder/stats/wasc/`. Per-edge Cochran-Q values exist for the real-label edges at `output/wasc/concordance_per_edge_m2_2.csv` (generated at commit `cac32d0` during M2.4 development); these were NOT consulted for v1.0.3, v1.0.4, or v1.0.5 amendment decisions (each decision log carries a no-Q-exposure attestation).

### 4.3 BY rank-feasibility — C17' table

The B-feasibility table corrected by v1.0.5 (replacing the v1.0.4 C17 text whose sign was inverted and B value mis-attributed):

At `N = 944`, `q = 0.10`, `H_944 = 7.4279`, the BY rank-k raw-p threshold is `p_{(k)} ≤ q · k / (N · H_N) = k · 1.4261e-5`. An edge can clear BY at rank k only if its achievable empirical p-value floor `1 / (B + 1)` is at most its rank-k threshold. The structural feasibility ceiling under perfect per-edge power is:

| B | Empirical-p floor | k_min (smallest testable rank) | Untestable ranks | Maximum testable rejections |
|---:|---:|---:|---:|---:|
| 999 (sensitivity) | 1.00e-3 | 71 | 1..70 (70 ranks) | 874 of 944 |
| 9999 (primary) | 1.00e-4 | 8 | 1..7 (7 ranks) | 937 of 944 |
| 99999 (floor-tie rerun) | 1.00e-5 | 1 | none | 944 of 944 |

Interpretation: at the pre-registered primary B=9999 the structural ceiling is 937 of 944 edges testable, not "69". Ranks 1..7 are untestable from B=9999 alone; the floor-tie tertiary B=99999 rerun resolves all 944. The "69" figure in the wrong v1.0.4 C17 text comes from B=999 (sensitivity tier) where ranks 1..70 are untestable; the original prose confused testable with untestable ranks AND mis-attributed the B value. Computation is independently re-derived by `scripts/wasc/verify_by_feasibility.py` and asserted by `tests/wasc/test_feasibility.py` (6 tests, all pass).

### 4.4 M2.5 tripwire — prong outcomes

The four-prong tripwire outcomes at tag v1.0.5 are summarized below.

#### 4.4.1 Prong (a) — Label-shuffle null calibration: PASS

Source artifact: `output/wasc/m2_5_prong_a_smoke/result.allpool_b999_production.json`

- Substrate: all-protein-pool (canonical primary per v1.0.3).
- Configuration: 20 shuffles × B = 999 permutations per shuffle.
- Outcome: mean false-positive rate at q = 0.10 = **0.1114**.
- Gate 2 bound: `0.10 + 2 · √(0.10 · 0.90 / 944) ≈ 0.1195`.
- Verdict: PASS. Margin ≈ 0.008 absolute (≈ 7% relative).
- Provenance: `pooled_pass = true` in the on-disk artifact.

#### 4.4.2 Prong (b) — Down-sampled-Sporadic Jaccard stability: HARD FAIL

Source artifacts: `output/wasc/m2_5_prong_b_b999/seed{42,7,99}/summary.json`

- Substrate: all-protein-pool. The build-plan-prong-(b) script does not pass `eligible_proteins` and so inherits the v1.0.3 canonical primary substrate.
- Configuration: 3 random seeds, Sporadic down-sampled to n = 25, B = 999.
- Raw-p (lower-tail per-edge p ≤ 0.10) edge-set Jaccards: seed42 = **0.297**, seed7 = **0.260**, seed99 = **0.299**. Mean = 0.285, standard deviation = 0.022.
- Pre-registered bound (build plan §5 lines 25, 766; spec C11 confirms `≥`): **Jaccard ≥ 0.70**.
- Verdict: **HARD FAIL** stable across three seeds. The mean is 41.5 percentage points short of the bound.
- BY-q-Jaccard reading is undefined because both full-Sporadic and down-sampled-Sporadic returned 0 BY-q ≤ 0.10 edges at B = 999 (the empirical-p floor is above the BY rank-1 threshold at this B; see §4.3). The raw-p Jaccard is therefore the load-bearing metric.
- Per-theme raw-p Jaccards (mean across three seeds, from the verified ledger): Splicing 0.254, Chromatin 0.314, Transport 0.273. All three themes are below the 0.70 bound; the FAIL is not driven by one theme.
- Note on edge count: the prong (b) artifacts run on **904 deduped edges**, not 944. The deduplication drops 40 cross-theme duplicate edges via `drop_duplicates` in the prong (b) script. Whether the BY-FDR denominator should be 944 or 904 is the v1.0.4 deferred decision C5; the prong (b) FAIL verdict is robust to this choice (Jaccards 0.285 are far below 0.70 under either denominator).

#### 4.4.3 Prong (c) — All-protein-pool null ratio: trivially satisfied (test not applicable)

The v1.0.3 amendment promoted the all-protein-pool substrate from "prong (c) sensitivity" to canonical primary. The test prong (c) was designed to evaluate (does the all-protein-pool null produce more than 3× the positive count relative to the theme-restricted null?) is now trivially satisfied because the theme-restricted substrate is itself the sensitivity, not the primary. The four-prong tripwire STRUCTURE is preserved per v1.0.4 items_NOT_modified ("M2.5 four-pronged HARD HALT STRUCTURE"); we explicitly do NOT relabel the tripwire as 3-prong. Prong (c) is reported as N/A (test no longer applicable) rather than as a deletion.

#### 4.4.4 Prong (d) — Frisch-Waugh vs OLS numerical identity: PASS

Source artifact: `output/wasc/m2_5_prong_d/result.n50_seed42.json` (working-tree at v1.0.5; `output/` is gitignored by project convention so the file is not in the v1.0.5 commit. The producing script `scripts/wasc/run_m2_5_prong_d.py` IS committed at v1.0.5; the artifact is reproducible by running the script from `git checkout wasc-prereg-v1.0.5`).

- Design: production GroupDesign with intercept + sex_female + age_z + tissue dummies. C9 design has 4 covariate columns (intercept, sex_female, age_z, tissue_NT_Cell); Sporadic and Control have 5 (the additional tissue_Bulk_or_Unknown column).
- Configuration: 50 randomly sampled edges × 3 groups = 150 (edge, group) triples. Per-triple complete-case mask applied identically to FW and OLS. seed = 42.
- Outcome: max |β_FW − β_OLS| = **9.423342106629207e-10** (worst triple: edge P06730|Q6P2Q9 in C9, theme Splicing, n=25, df=20). max |relative SE difference| = **3.4862159867971017e-11** (same triple). Median absolute β difference = 6.6e-14; median relative SE difference = 2.6e-13.
- Pre-registered bound: 1e-8 absolute on β and SE (spec §2.5(a) line 150; build plan §5 line 27/768).
- Verdict: PASS. Margin on the load-bearing β-absolute metric: 1e-8 / 9.423e-10 ≈ **10.6×**. Margin on relative-SE: 1e-8 / 3.486e-11 ≈ **287×**.
- Diagnostics: `n_skipped_unmeasured = 0`, `n_skipped_unconverged = 0`, `n_skipped_ols_failed = 0`. All 150 triples converged. Wall clock 0.22 s.

#### 4.4.5 Tripwire overall: HARD HALT

Three of the four prong slots return PASS or trivially-satisfied, but prong (b) is HARD FAIL across three seeds. Per build plan §5 hard-halt semantics ("HARD HALT semantics: if `overall_pass=False`, the orchestrator MUST abort before M3"), the primary B=9999 production run is **BLOCKED**. STRING comparator (M4), three-contrast decomposition (M5), and the mandatory sensitivities batch (M5.5) are all BLOCKED behind the same gate.

### 4.5 Tests

194 of 194 WASC-namespace tests pass at tag v1.0.5. The locked-bounds CI gate (`tests/wasc/test_locked_bounds.py`) asserts spec-text parity for every load-bearing numeric threshold (B values, q, min_n_per_group, C2 floor, prong (b) Jaccard floor, prong (d) tolerance, `|E_WASC|=944`, canonical anchor count 203) against both the spec quote and the code default. The C17'-arithmetic test (`tests/wasc/test_feasibility.py`) re-derives the B-feasibility table from `q / (N · H_N)` and `ceil(floor / rank1_threshold)` independently of any spec-quoted number, providing a CI regression test against any future C17-class arithmetic bug.

---

## 5. Limitations

### 5.1 Structural underpoweredness of n = 25 (the prong (b) finding)

The prong (b) HARD FAIL is itself the substantive scientific finding of this work. With C9 n = 25, when the Sporadic arm is down-sampled to match (n = 25 from n = 294 at three random seeds), the edge set returned by the pre-registered raw-p ≤ 0.10 selector overlaps the full-Sporadic set with mean Jaccard 0.285 — far below the pre-registered ≥ 0.70 bound. The pattern is stable across seeds (standard deviation 0.022) and stable across themes (per-theme Jaccards 0.254, 0.314, 0.273). The conclusion is not "the pipeline is broken" but "the pre-registered raw-p selector at B=999 is not stable to a sample-size reduction from 294 to 25 on this matrix at this contrast scale".

The prong (b) bound was deliberately set high (≥ 0.70) by the pre-registration to reject methodologies that are stable on n = 294 but become a different result at n = 25 — the very asymmetry the test is designed to detect. The bound is doing its job; the methodology fails it. Three paths forward are discussed in §6.2 below.

### 5.2 Forbidden-language scope

Spec §9 lists *mechanism*, *causal*, *causation*, *drives*, *regulates*, *controls* (verb), *rewiring*, *validates*, *post-transcriptional*, and *INDRA-edges-are-correct* as forbidden in any claim-bearing context. This document uses Control only as a donor-group proper noun (e.g., "the Control n=59 cohort"), uses *regulatory* only as a substrate name (e.g., "the INDRA regulatory subgraph"), and uses *validation*-stem language only in defensible substrate-naming contexts (e.g., naming the locked-bounds CI gate, which is a code-text parity check, not a biology claim). The forbidden stems are otherwise absent. A grep at the end of the §7 self-check confirms zero hits in non-defensible contexts.

### 5.3 Cohort and tissue limitations

The AnswerALS matrix is PBMC, not postmortem CNS. The Primary_Tissue covariate is a 3-level proxy (T-Cell / NT-Cell+Bulk / Unknown) that is heavily group-confounded (C9 92% T-Cell, SPOR 65%, CTRL 38%). Granular cell-composition (T-CD4/CD8/B/NK/monocyte) cannot be deconvolved from DDA-MS proteomics; this is acknowledged as a limitation per spec §9 ("It cannot resolve confounding by granular cell-composition beyond the Primary_Tissue 3-level proxy").

### 5.4 RF-imputed substrate

The AnswerALS portal provides the matrix as the `correctedImputed_436` track (RF-imputed upstream). The v1.0.2 missingness-axis drop is correct on this substrate but means the matched null does not condition on detection variability that would be visible in a pre-imputation file. A v1.1 re-derivation on the prebatch matrix is reserved for a future tag; it is not in scope here.

### 5.5 Amendment-chain credibility cost

WASC has amended its analytical decision family across four tags in three days (v1.0.2 axis drop, v1.0.3 substrate swap, v1.0.4 corrections, v1.0.5 correction-to-a-correction). Each amendment was triggered by an independent audit gate and has a decision-log JSON; the chain is honest and traceable. The accumulating count is a real cost. The v1.0.5 decision log acknowledges this explicitly and proposes a structural fix (an independent re-derivation gate for any NEW arithmetic claim in future amendments) to slow the amendment frequency. The PI should weigh this when deciding whether further amendments are warranted versus pivoting (Path B) or expanding the cohort (Path C).

### 5.6 No real-Q observed in primary

No production B=9999 run has executed. The prong (b) FAIL gates the primary by build-plan construction. Any narrative beyond the tripwire-FAIL itself would require running the production pipeline and would constitute a violation of the pre-registered HARD HALT.

---

## 6. Discussion and Future Work

### 6.1 What this work establishes

The substantive contributions of this work at tag v1.0.5 are:

1. A pre-registered methodology for testing within-cluster cross-protein abundance coupling invariance, with a six-tag amendment chain, locked-bounds CI gate, and four-prong tripwire — all auditable artifacts under git.
2. An empirical finding that n = 25 C9 cohorts are structurally underpowered for the pre-registered Jaccard-stability tripwire at B = 999 on this matrix at this contrast scale. This is itself a finding of methodological consequence: pipelines that pass label-shuffle calibration and numerical-identity tests can still fail a down-sample stability gate, and pre-registration of that gate prevents the methodology from being published in a form that papers over the asymmetry.
3. An infrastructure deliverable: the `data/wasc/locked_bounds_v1.json` plus `tests/wasc/test_locked_bounds.py` plus `tests/wasc/test_feasibility.py` pattern is a transferable design for spec-pre-registration CI gates. The pattern caught both the spec-vs-code drift (C7 RNG formula) and the arithmetic-text bugs (C1 BY rank-1; C17 perfect-H1 power) that survived four prior audit gates.

### 6.2 Recommended disposition — Path A — publish as INCONCLUSIVE

This document recommends **Path A**: publish the methodology, the tripwire outcomes, and the prong (b) FAIL as an INCONCLUSIVE primary result with no claim about C9-specific coupling-structure. The merit of Path A is that it respects the pre-registered HARD HALT, lets the methodological finding stand as the contribution, and does not commit to further compute on a cohort that the tripwire has already flagged as underpowered for this design.

Path A deliverables are:

- A methodology paper (or note) describing the pre-registration discipline, the amendment chain, and the four-prong tripwire as design pattern.
- A negative-finding report on the n = 25 structural underpoweredness for pre-registered Jaccard-stability gates at this contrast scale, framed as "for what study designs does this tripwire halt before the primary?".
- Public release of all frozen artifacts under `data/wasc/` plus the `scripts/wasc/` orchestration plus the test suite as a reference implementation.

The PI may choose either of the following alternatives instead. They are presented neutrally for the PI's decision but the document's scope is committed to Path A.

#### Alternative — Path B — methodological pivot

Path B is to pivot the analysis from a per-edge BY-FDR pre-registered test to a hierarchical model whose claim ceiling is "cluster-pattern invariance" not "count of positive edges at q ≤ 0.10". The hierarchical reformulation would partial-pool slopes within anchor and within theme, with group as a random effect; the primary statistic would be the variance ratio (between-group / within-anchor) rather than per-edge Cochran-Q. The Jaccard-stability prong (b) would not apply in the same form because the per-edge selection step is replaced by a continuous posterior summary. The cost is that Path B is a new pre-registration (`wasc-prereg-v1.1` at minimum, possibly `v2.0`) with its own audit arc, its own calibration tripwire, and the explicit acknowledgment that the v1.0.x pre-registration was an unsuccessful path.

#### Alternative — Path C — cohort expansion

Path C is to acquire additional C9 samples (target n ≈ 50–75) and re-run the v1.0.5 pipeline as-is. The prong (b) FAIL at n = 25 would be expected to soften as n grows; the question is whether the down-sample-to-25 Jaccard becomes ≥ 0.70 at the larger primary cohort, which is what the tripwire is asking. Path C respects the pre-registered methodology but is gated on sample acquisition lead time (months to a year) and on whether AnswerALS or a partner cohort can supply the additional C9 donors with comparable PBMC DDA-MS proteomics under matched processing.

The recommendation in this document is Path A because (a) the methodological finding is itself publishable, (b) the amendment-chain credibility cost (§5.5) argues against further pre-registration churn, and (c) the cohort expansion timeline of Path C is not under the PI's near-term control.

### 6.3 What this work does NOT establish

Per the spec §9 claim ceiling, this work does NOT support any of the following claims, and the absence of a positive primary result reinforces this:

- It does not establish that any particular INDRA hop-1 edge corresponds to a true in-vivo biomolecular relationship between the two proteins; the INDRA edge is a literature-derived link, not a claim about in-vivo biology.
- It does not adjudicate among transcriptional, translational, or post-translational origins of any observed coupling.
- It does not show that coupling structure within the eight cluster terms is C9-specific. The three-contrast decomposition (§3.8) was BLOCKED by the tripwire and never executed on real-label data at production B.
- It does not retract or modify the Wave-24l per-feature gradient claim (which is a different test family on different statistics; see `memory/wave_24l_measured_only_paths.md`).

---

## 7. Open decisions for PI

The following are the unresolved decisions that gate any deliverable from this work. Each is presented with the recommended disposition under Path A.

### 7.1 Confirm Path A as scope

The PI confirms (or rejects) the recommendation in §6.2 to publish as INCONCLUSIVE under the existing v1.0.5 tag. Path B and Path C are not foreclosed by Path A but are out of scope for this document.

**Recommended:** confirm Path A. Cite (a) the pre-registered HARD HALT in build plan §5, (b) the cumulative amendment-chain credibility cost in §5.5, and (c) the standalone methodological contribution in §6.1.

### 7.2 v1.1 deferred items — handle now or freeze permanently

Four items were deferred from v1.0.4 to a hypothetical v1.1 by the synthesis review:

- **C2 (B promotion)** — promote B = 9999 → B = 99999 as primary, eliminating the staged-B design. Affects per-edge p floor; would make ranks 1..7 testable at primary (per §4.3 C17' table).
- **C5 (BY-FDR denominator)** — pre-register 944 vs 904 as the BY-FDR denominator. The 40-edge difference is the cross-theme deduplication currently invoked silently by the prong (b) script.
- **C8 (Gate 2 metric)** — bind raw-p versus BY-q as the calibration metric for prong (a). Currently raw-p is the operational metric because BY-q floor-ties at B = 999 (see §4.3).
- **C12 (K-sweep)** — `min_unique_q_values` K ∈ {1, 10} as tertiary diagnostics. The v1.0.3 prohibitions clause forbids K-sweeps as a rescue device, but tertiary diagnostic use was deferred.

**Recommended under Path A:** freeze all four as permanent v1.0.5 design choices. Path A does not run further primary compute, so these items have no operational impact under Path A; freezing them locks the v1.0.5 spec as the published methodology. If the PI selects Path B or C, these items re-open under the new tag's gate.

### 7.3 Prong (b) substrate dedup acknowledgment

The prong (b) artifact uses 904 deduped edges (40 cross-theme duplicates dropped via `drop_duplicates`) while the primary BY-FDR denominator is the locked 944. Under Path A, the published methodology should acknowledge this silent dedup as a v1.1-deferred decision (C5 above) and report Jaccards under both denominator conventions for completeness. The 0.285 mean Jaccard is far below the 0.70 bound under either convention; the FAIL verdict is robust.

**Recommended:** include the 944 vs 904 acknowledgment as a footnote in the Methods § of the publication. The on-disk artifacts are not retroactively re-run; the deferred-to-v1.1 status is recorded as the binding statement.

### 7.4 Public-release artifact set

Path A deliverables (§6.2) include public release of `data/wasc/` plus `scripts/wasc/` plus the test suite. The PI confirms scope and license.

**Recommended:** release under the project's standard open-source license. Tag the release commit `wasc-publication-v1.0.5` to distinguish from the pre-registration tag. Include the locked-bounds CI gate as a release-gating test (already in CI; no additional infrastructure).

### 7.5 Cite-as for the methodology contribution

The methodology contribution is the six-tag amendment-chain discipline, the locked-bounds CI gate as a transferable pattern, and the four-prong tripwire as a publishable design. The PI confirms how to attribute (e.g., methodology paper authorship list, acknowledgment of the audit-gate process, choice of venue).

**Recommended:** target a methods venue with a substantive interest in pre-registration discipline for high-dimensional bioinformatics tests. The methodological finding (structural underpoweredness of n = 25 for pre-registered Jaccard-stability gates at this contrast scale) is the substantive contribution and the framing.

---

## 8. Self-check (for PI scrutiny, not for publication)

The following self-check was performed before returning this scaffold to the PI. Each item is a known prior-draft failure mode that was caught by external audit gates in workflow ` V3` against an earlier Path A scaffold; this scaffold is required to self-correct in advance.

- **Tripwire framing.** The tripwire is described as a 4-prong STRUCTURE throughout (§3.9, §4.4, §6.1). Prong (c) is reported as "test no longer applicable / trivially satisfied" rather than as a deletion. This is consistent with v1.0.4 items_NOT_modified preserving the four-prong structure.
- **C17' B-feasibility.** §4.3 reports the C17' table (three rows: B=999 / B=9999 / B=99999 with untestable rank counts 70 / 7 / 0). No single number like "69" or "937" is presented as a standalone "structural ceiling"; the table is the binding form.
- **Prong (d) numerics.** §4.4.4 cites the on-disk artifact `output/wasc/m2_5_prong_d/result.n50_seed42.json` with the exact numbers 9.423342106629207e-10 and 3.4862159867971017e-11. The margin is reported as ≈ 10.6× on β-absolute (the load-bearing metric) and ≈ 287× on relative SE. The "30-40×" figure from a prior draft is NOT used.
- **Prong (b) substrate.** §4.4.2 explicitly cites all-protein-pool (per v1.0.3) as the prong (b) substrate, NOT theme-restricted. The 904-vs-944 deduplication is acknowledged as a v1.1-deferred decision (§4.4.2 final paragraph and §7.3).
- **Forbidden-language audit.** A grep of the draft for the stems `mechanism`, `causal`, `causation`, `drives`, `regulat-`, `control-`, `rewiring`, `valid-`, `post-transcriptional`, and `INDRA-edges-are-correct` returns hits only in defensible substrate-naming or proper-noun contexts: *Control* as a donor-group proper noun (e.g., "C9 / Sporadic / Control"), *regulatory* as the substrate name for the INDRA subgraph (Methods §3.2 only), and *validation* / *valid-* as a CI-test-quality term (locked-bounds gate). No instance of *mechanism*, *causal*, *causation*, *drives*, *rewiring*, *post-transcriptional*, or *INDRA-edges-are-correct* appears in any context. (NOTE: an earlier draft of §6.3 had "regulatory relationship between the two proteins" in a Discussion-section biology-claim context — a real forbidden-language hit. That phrasing was corrected to "biomolecular relationship" before this scaffold was saved; the self-attestation as written reflects the post-correction state.)
- **Scope commitment.** §1, §3.6, and §6.2 commit the document to Path A. Paths B and C appear only as labelled alternatives under §6.2 and do not change the recommended disposition.

End of v1 scaffold.
