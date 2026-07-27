# C9-ALS Deck — Technical-Walkthrough Re-Architecture

> [!CAUTION]
> **SUPERSEDED PLANNING SNAPSHOT — not a current analysis authority (2026-07-12).** This document preserves the technical-walkthrough rationale but predates the production `log2(x+1)` rerun. Current state lives in `data/publication/c9_primary_analysis.json`: EB prior df `4.98 / 5.51 / 4.92`; bounded measured-only `h≤2` confirmatory `8 / 6 / 0`; unbounded sensitivity `6 / 0 / 0`; feature accounting `3,264 attempted = 3,117 valid + 137 without a reachable measured neighbor + 10 below the ≥10-neighbor guardrail`. Any raw-linear, `d₀≈0.60`, `7 / 6 / 0`, or `3,257 / 7 / 130 single-shell` statement below is historical context only.
> **Current integrity note:** this is a historical plan, not presentation authority. The audited same-data auxiliaries are now HGNC size-matched `8/8/0` and robust-scope degree-matched mean(`-slope`) `7/7/0` (Vpr is the sole C9 non-pass). STRING, matched RNA, age, abundance, legacy F5b, and five legacy sensitivity claims below remain outside the evidence stack. The historical 56.5% random observed-set rejection is not an FPR/calibration estimate.

**Reframe:** make the methodology the load-bearing spine. Today the real method lives in 31 footer MethodStrips and 4 `SKIP FOR A GENERAL AUDIENCE` interludes — "minimalism as avoidance." The fix is *relocation and cogency*, not detail-dumping: every analytical step becomes a first-class beat (claim · on-ramp · method · validity/kills-alternative · handoff), intuition as the on-ramp **to** each step rather than a substitute for it.

**Current numerical authority:** `data/publication/c9_primary_analysis.json`. The methods narrative remains in `output/analytical_workflow_breakdown.md` (8 steps + §3 disambiguation + §4 claim ceiling).
**Target files:** `slides.md`, `components/*.vue`, inheriting the §1 design spine already in `REWORK_PLAN.md`.

At the time it was written, this document superseded the *structure* of REWORK_PLAN §2–§3 and inherited its §1 Design Spine and §5 "6 Slidev pitfalls." Its numerical grounding is now historical; slide indices still refer to the single-file `slides.md` numbering used in that rework.

---

## 1. Architecture decision

### (a) The 4 SKIP-interludes → **DISSOLVE the skip framing; promote 3 into the spine, keep 1 as a labeled READ-ledger.**

The `FOR THE METHODOLOGIST · SKIP FOR A GENERAL AUDIENCE` eyebrow is the exact anti-pattern for a technical audience: it tells the reader to skip the only slide where the method lives. For a technical talk, *delete the skip eyebrow from all four* and re-cast them:

| interlude | now | decision |
|---|---|---|
| **s12** "why the standard test failed" (3-null table + cascade-collapse) | skip-aside | **DISSOLVE into spine.** The competitive-z → Camera-VIF → matched-null cascade and the cascade-collapse row are the *substantive content of Act 1*. Distribute into promoted s11 (Camera-VIF beat) + a new cascade-collapse slide. Keep a slimmed table as a non-skip "robustness ledger" receipt slide titled **"Three references, one verdict"** — READ, not skip. |
| **s20** "the null that earns the slope" (hub hazard/fix + degree bins) | skip-aside | **DISSOLVE into spine.** The hub-confound rationale *is* the validity backbone of the gradient. Merge the hazard/fix card-pair into a re-titled, non-skip s17 ("How we know it's not shapes in clouds") and add the VCL falsifier. No interlude remains. |
| **s24** "reading the three comparisons" (4-pattern truth table) | skip-aside | **PROMOTE to a main slide between s23 and s25.** The truth table is the single deductive move that licenses C9-specificity. Un-skip, re-title **"Why an empty third cell is the proof."** |
| **s30** "the robustness ledger" (7-row kills-table) | skip-aside | **KEEP as a consolidated MAIN closing slide of Act 2, re-titled "One claim, seven falsifiers," eyebrow removed.** For a technical audience a kills-ledger is a feature, not an aside — but each row must also be *earned* on its own beat upstream (distribute the ledger). |

**Net: zero `SKIP` eyebrows survive.** The interlude *visual treatment* (recessed bordered ledger, mono voice, no motion) is retained for the two surviving ledger slides — that density is honest — but the "skip" instruction is gone. The §1 rule "load-bearing narrative conclusions must NEVER live only on a skippable interlude" becomes "…must live on the spine," and there are no skippable interludes.

### (b) The 31 method-strips → **auditable-provenance only; load-bearing physics promoted into bodies.**

Re-scope the MethodStrip from "where the method hides" to "the mono forensic receipt the PI reads to audit." A strip's content is legitimate only if it is *citation + exact parameters + provenance* (file, perm count, thresholds, version tags). Anything **load-bearing for following the logic** moves into the slide body. Concretely, these strips get their physics promoted into the body and the strip demoted to citation:

- s11 (`competitive z=1.25 → 0.69, Camera VIF, ρ̄=0.185`) → body owns the VIF chain; strip keeps `Wu 2010 · enrichment_z.py`.
- s14/VolumeKnob (`|EB-moderated t| · limma variance shrinkage`) → body owns "effect ÷ SE, magnitude" + the EB rationale; strip keeps the formula + `d0≈0.60, raw-linear`.
- s18/GradientSlopeAnatomy (`at depth 2 ≡ ring2 − ring1`) → body owns the two-point-line identity; strip keeps the general inverse-variance WLS case.
- s17/s20 (`999 perms within degree bins, Guney 2016`) → body owns the hub hazard/fix; strip keeps perm count + p-formula + floor.
- s21/NeighborhoodTally (`gseapy.prerank · rank by −slope · NES-histogram FDR`) → body owns "one ranked list, gene-set permutation, score=−slope sign convention"; strip keeps gseapy params (perm=**1000**, weight=1, seed=123).
- s33/CochranQ (`inverse-variance Cochran-Q · BY-FDR`) → body owns "LOW Q = invariant," the matched-bin null; strip keeps the BY-FDR rationale + bounds.

The remaining ~24 strips stay as pure provenance (cohort denominators, edge counts, version tags). **Rule going forward:** *cover the strip — the body must still let a technical reader follow the step.* If it can't, the strip held something load-bearing and that content moves up.

### (c) Dual-track model → **keep intuition as the on-ramp; make method the first-class spine immediately after — for a technical primary audience.**

Keep the dual-track *device* but invert which track is load-bearing. Today: intuition is the spine, method is the footnote. Target: **intuition is the 5-second on-ramp into each beat, the method beat IS the slide.** The earthquake, the smoke-detector, the tide/yardstick, "cheat against yourself," "the tilt is the link" all stay — they help technical people too — but each is followed *on the same or the very next slide* by the named statistic, the formula or its plain algebra, and the alternative-it-kills. The speaker-note instructions that *suppress* method ("Say loudness, not t-statistic," "Don't say those words to the room," "avoid the word sign") are **deleted or inverted** for the technical cut — a technical deck must name self-contained vs competitive, must say "moderated t = effect ÷ SE," must show the minus sign.

**Recommended model, in one breath a technical reviewer would endorse:** *Every analytical step is one beat — an intuition on-ramp, then the named statistic with its formula or plain algebra, then the specific alternative it rules out, then what it hands to the next step. Strips carry only the auditable parameters and citations; nothing load-bearing for following the logic lives in a footer or a slide stamped "skip." The narrative spine and the methods spine are the same spine.*

---

## 2. The cogent technical spine (the target the DO restructures toward)

Each step: **claim · on-ramp · method · validity/kills · handoff → slide(s).** New/promoted beats marked ★.

### ACT 0 — Setup (s0–s7) — unchanged spine, grounding fixes only
**S1. Data & cohort.** *Claim:* one number per protein per person, three groups. *On-ramp:* "we weighed the dishes in blood." *Method:* AnswerALS PBMC, **3,264 measured proteins → 423 metadata-matched samples → 3,257 fitted** (7 graph-disconnected dropped); **UniProt accession as the unit of analysis** (one value per protein, no HGNC-alias inflation, Wave 22); three groups **C9 = 25 · Sporadic = 294 · Control = 91**; **Sex the sole covariate.** *Validity/kills:* why UniProt (kills alias inflation); why Sex (Table 1 SPOR_LIMB 78% M vs Control 52% M imbalance is the dominant confound); **raw-linear intensities** owned as a caveat (pathway direction robust, |t| r≈0.93 raw-vs-log2; gene-level naming ~30% churn). *Handoff:* each contrast subset feeds the |t| engine. → **s3 (What we measured) + s4 (Three groups).** ★ promote denominators + unit choice + raw-linear caveat into body. **Fix: Control = 91 everywhere (footnote WASC 59 as the post-exclusion sub-cohort).**

**S0b. The map.** INDRA: ~129,000 edges, 4 directional regulatory edge types, median ~15 measured neighbors, **distance read undirected.** → s5. *(strip-level; add "undirected distance" — currently absent.)*

**S0c. The one question.** H₀: |t| independent of graph position; test = contrast-specific spatial concentration. → s7. Unchanged.

### ACT 1 — The failed gene-set gate (s8–s13)
★**S2-foundational. What |t| is.** *Claim:* one number per protein — signal-to-noise of the abundance shift. *On-ramp:* the volume knob (loud = big + confident; uncertain-big gets turned down). *Method:* **moderated t = contrast effect ÷ its standard error**; model **intensity ~ group + Sex**, group is the tested coefficient, Sex the nuisance column; **|t| keeps magnitude, drops direction** (that is why the gradient measures perturbation *size*). *Validity/kills:* a freak-small-variance protein would otherwise top the ranking — names the failure mode that motivates EB. ★**EB half-beat:** borrow strength across ~3,200 proteins; posterior variance = blend of own variance + a fitted genome-wide prior (limma fitFDist, Smyth 2004); shrinkage is *weak here* (d0≈0.60 ≪ df) because raw-scale variances span ~10 orders — doubles as the on-ramp to the raw-vs-log2 caveat. *Handoff:* "this one number per protein is the only thing every later step reads." → **promote VolumeKnob (s14) earlier OR add a dedicated `|t|` slide before s9.** This is the single most important addition — the foundational statistic gets a full beat, not a footer.

**S3a. The obvious first test.** *Claim:* ask a curated group at once. *On-ramp:* group → ROAST box → YES. *Method:* **self-contained ROAST rotation** of a **curated ~47-gene C9 target set**; MSQ = mean of *squared* EB-moderated t under MIXED, scored against a zero-effect rotation null — **set vs no-change, never vs the rest of the proteome.** *Validity hook:* "this is the standard move; next slide shows why it's the wrong reference here." → **s9.** ★ promote "curated 47-gene / self-contained" off strip into body.

**S3b. We fed it pure noise.** *Claim:* 56.5% FPR. *Method:* 200 size-matched random sets through the identical engine, **113/200 clear p<0.05**; ★ add the specificity corollary — **the curated set sits at the 12.5th percentile of those 200 (below the median random set).** → **s10.** Already strong; add one line.

**S3c. Why it lied — the wrong yardstick.** *Claim:* on a broadly-perturbed proteome, almost any set is "DE." *On-ramp:* the tide rose; every group floats. *Method:* ★ show the mechanism panel — **random mean|t| ≈ 1.0 vs target ≈ 1.15 (~13% above background)**; MSQ averages squared t against a zero-effect null, so global elevation alone clears the bar. Re-ask **competitively: z = (mean|t|_set − mean|t|_bg)/SE; z = 1.25, p = 0.21 → ordinary.** *Validity:* name **self-contained vs competitive** as the conceptual hinge — on a main slide, in words. → **s11** (promote competitive-z into the RIGHT panel body).

★**S3d. Camera-VIF — co-regulation collapses it to null.** *Claim:* 47 co-regulated targets are not 47 independent votes. *On-ramp:* "they move together." *Method:* **VIF = 1 + (k−1)·ρ̄; ρ̄ = 0.185 → VIF = 9.5; SE inflates by √9.5 = 3.08; z 1.25 → 0.41, p 0.21 → 0.69 → null.** *Validity/kills:* not goalpost-moving — brutalist 3/3 confirmed VIF corrects genuine co-regulation, not signal. → **own half-slide or climactic panel of s11.** Promoted from skip-table.

★**S3e. We falsified our own day-one triumph.** *Claim:* the celebrated cascade WAS the artifact. *On-ramp:* "day one looked spectacular — a 5-hop cascade, hop-1 p=0.019, 46 of 46 two-hop sets significant." *Method:* that was raw self-contained ROAST — the same gate that fails at 56.5% FPR; re-run under corrected competitive-z. *Kill:* **hop-1 0.019 → 0.69, 46/46 → 0/46 (π₀ 0.02 → 1.0).** → **new main slide between s10 and s13** (the deck's strongest integrity beat; the author's own note begs "say it plainly… not a footnote").

**S3f. Three references, one verdict.** Slimmed receipts ledger (READ, not skip): self-contained FPR 56.5% · competitive z 1.25/p 0.21 · +VIF z 0.41/p 0.69 · **sex-matched subsample (n=50) p=0.65.** *Claim:* triangulated to null across three different reference models. → **re-cast s12 (eyebrow removed).**

★**S3g. What survived — and the verdict.** *Claim:* every set-level test died, but one shape didn't. *Evidence:* network proximity — across **938 distance-1 genes, mean|t| 1.31 (hop 1) vs 1.06 (hop 2), permutation p = 0.001** — a topology–expression gradient, *not* differential abundance of the 47 targets. *Verdict:* the residual is **real but tiny (~0.13–0.17 moderated-t units)**; on n=25 a binary yes/no gate (~1–2 effective independent tests) cannot resolve it, so **we stop asking yes/no and measure a continuous per-protein quantity.** → **fold into s13 pivot.** Converts the pivot from a vibe into a data-forced consequence; pre-installs the n=25 power-wall that recurs at WASC.

### ACT 2a — The gradient and its null (s13–s20)
**S4-bridge. Does it fade with distance?** *Thesis line:* Act 1 asked *did-it-move* (fakeable on a broadly-perturbed proteome); we now ask *does perturbation concentrate near an anchor* — a per-protein decay slope; unit of analysis changes from one curated set to every protein-as-anchor. → **s13** (earthquake on-ramp kept).

**S5. The shell & the hops.** *Method:* distance = **minimum hop count, BFS, undirected**; a **shell = the ring of measured proteins at exactly h hops; its statistic = the arithmetic mean of |t| over that ring.** → **s15.** ★ establish the shell-as-summarized-object and "undirected" (sets up s18).

**S6. Measured-only — and why it gives the slope power.** *Method:* BFS may only step on measured proteins. ★*Validity/kills:* with intermediates, the 2-hop ring saturates near ~3,240 for *every* anchor (flat, non-discriminating); measured-only makes each ring its own size (**median ~2,000, range 0–~3,000**), and *that variation is the source of the slope's discriminating power* — kills the loose topological-proximity-via-unmeasured-bystander reading. → **s16.**

★**S7-centerpiece. The slope = ring2 − ring1.** *Claim:* the per-protein number is one transparent subtraction, not a black box. *On-ramp:* loud-near / quiet-far. *Method:* annotate the two ring means on the figure (e.g. ring1 = 1.8, ring2 = 1.05); **slope = (ring2 − ring1)/(2 − 1); at depth 2 the inverse-variance WLS reduces algebraically to ring2 − ring1 — the weights cancel** (verified for all weight choices). *Validity:* negative = perturbation concentrates near the anchor. → **s18** (show the minus sign; delete the "don't get into the minus sign" note). Strip keeps the general n_shells>2 WLS case.

★**S8. The degree-binned null + the VCL falsifier.** *Claim:* a negative slope means nothing without the *right* null. *On-ramp:* "cheat against yourself — reshuffle loudness on a fixed map." *Method:* shuffle |t| **within INDRA-degree bins (≥100, Guney 2016)**, 999×, recompute slope, p = (#≤obs + 1)/1000, floor 1e-3. ★*Validity/kills (the hub hazard/fix):* hubs sit closer to every anchor and are better-measured, so a uniform shuffle piles big |t| into near shells and fakes a steep slope; binning forces a hub's |t| to swap only with another hub — topology fixed, only biology moves. ★*Empirical falsifier:* **VCL (Vinculin, degree 3759 ≈ C9orf72 3734) gives slope +0.029, p ≈ 0.92 — no gradient;** if the C9 slope were a hub artifact, VCL would show it too. *Handoff:* this earns each per-anchor slope's p-value; the GSEA null downstream is a separate question. → **dissolve s20 into a re-titled s17; add VCL.** Upgrade `PermutationNull.vue` to show two degree lanes swapping only within a lane.

★**S9. Why depth 2 — interpretive ceiling.** *Method:* depth-2 leaves ~39% of each anchor's reachable proteins untouched (≈61% captured) — a deliberate stop, not exhaustion; at h≥3 the slope stops measuring cascade-decay and starts measuring component-membership. *Validity:* a bin-stratified bounded-vs-unbounded test — bin anchors by bounded slope, compare the bounded→unbounded shift within each bin — found cluster and non-cluster anchors behave identically within every bin; cluster anchors move more only because they start steeper. Distinguish **structural ceiling (reachability exhausts at h=4)** from **interpretive ceiling (slope stops measuring cascade at h≥3)** — the claim rests on the interpretive one. → **fold into s15/s16 back-half OR a short beat before s21.** (Currently speaker-note-only.)

**S10. Every protein its own anchor.** **3,257 reachable; ~3,120 yield a usable slope** (130 single-shell anchors have only one ring, so a two-point slope is undefined — reconciles the two counts); do the steep ones pile up by shared job? → **s19.** ★ add the 3,257-vs-3,120 half-sentence.

### ACT 2b — Slope-GSEA, triangulation, robustness (s21–s30)
★**S11. What preranked GSEA is, and the score=−slope sign convention.** *Claim:* per-anchor slopes become one ranked list; do a pathway's members bunch at the steep end? *On-ramp:* "total the steep slopes by shared job." *Method:* ~3,120 slopes → **one preranked list (score = −slope)**; running-sum ES walks the fixed list; significance from **gene-set (label) permutation of the fixed ranking — NOT phenotype permutation**; **score = −slope is negated so the most-negative slope ranks first → a POSITIVE NES means the pathway's neighborhoods concentrate perturbation** (name this as the T45 hinge; a sibling |t|-GSEA uses score=−|t| where the sign means the opposite — misreading it *was* a corrected integrity error). FDR = gseapy NES-histogram (not BH), **perm = 1000**, per database (GO/Reactome/WikiPathways/HPO). *Validity:* gene-set permutation asks whether members are non-randomly positioned given the fixed ranking. *Handoff:* a per-term NES+FDR for one contrast — next, the identical pipeline on three contrasts. → **split s21 into two slides (A intuition / B method).** Separate the GSEA FDR from the Bonferroni-8 (which belongs to s25).

**S12. Three neighborhoods light up.** Splicing · chromatin · nucleocytoplasmic transport; ★ surface **cross-cluster Jaccard < 0.06 (near-disjoint membership)** onto the slide (kills "one gene-set wearing three hats"). → **s22.**

★**S13. Why an empty third cell is the proof.** *Claim:* the 4 pass/fail patterns map to 4 interpretations; only one survives. *Method:* truth table — **✓✓— = C9-specific · —✓✓ = shared ALS · ✓✓✓ = topology artifact · ——✓ = sporadic-specific.** *Validity/kills:* the empty **SPOR-vs-CTRL** cell simultaneously excludes shared-ALS pathology *and* graph-topology artifact, because both would light up SPOR-vs-CTRL. *Handoff:* so ✓✓— in the real data leaves only the C9-specific reading. → **promote s24 to a main slide between s23 and s25** (highest-leverage single relocation in the deck).

**S14. The headline + Bonferroni-8.** **7/8 (C9-vs-SPOR) · 6/8 (C9-vs-CTRL) · 0/8 (SPOR-vs-CTRL).** ★ add two load-bearing facts: **the 8 terms were frozen before looking** (what makes Bonferroni-8 legitimate, not post-hoc), and this is a **confirmatory readout** of the s21 GSEA tables (raw permutation p < 0.05/8 = 0.00625 AND NES>0), not a new pipeline. → **s25.** Empty column gray = win.

**S15. Why the zero is the point + the size-matched null.** ★ Put the two-threshold reconciliation **on-slide:** strict bar → 0/8; loose FDR → 372/422/8 (~50× drop); **those 8 loose hits clear 0 strict bars — same story, two thresholds** (kills the "control lit up" misread). ★ Give the **size-matched null its own labeled beat:** observed cluster mean(t²) vs **10,000** size-matched random sets, **p<0.001 in both C9 contrasts, not sporadic** — graph-free, kills gene-set-size inflation independently of the network argument. → **s26 (split the load).**

**S16. The six-term core.** 6/8 pass both C9 contrasts + quiet in control (3 splicing · chromosome · chromatin · transport). ★ promote the two-term honesty on-slide: **Vpr-mediated nuclear import = smallest term/size-artifact; nuclear pore = C9-vs-SPOR only** (disclosed exceptions raise credibility). Defer "graph-invariant" until s29 defines it. → **s27.**

**S17. STRING comparator — why opposite sign is the validating outcome.** *Method:* swap INDRA literature edges for **STRING physical-PPI (v12.0, ≥700, 173k edges)**, identical pipeline → **0/8 sign-agreement, opposite direction.** ★*Validity (the mechanism of informativeness):* **INDRA edge = regulator→regulatee** (perturb the regulator ⇒ regulatees co-vary ⇒ slope sees it); **STRING edge = stable-complex partners** (co-stabilize ⇒ buffered stoichiometry ⇒ need not co-vary). So opposite sign is *expected* if the effect is a property of regulatory relationships — STRING is an informative comparator that shows the network leg is **INDRA-edge-specific.** → **s28** (delete "avoid the word sign"). *(Hazard to verify in DO: STRING hop1−hop2 vs production hop2−hop1 sign convention — confirm the 0/8 headline isn't a double-negation artifact.)*

★**S17b. Cross-modality RNA-seq discordance.** *Claim:* the network-aware fingerprint is a **protein-level** property. *On-ramp:* "does the same fingerprint show up in the matching RNA?" *Method:* same measured-only graph, same 8 terms, **OLS ~ Sex + Batch + Group on 463 dual-modality donors, 10k perms.** *Result (count+pattern):* **opposite-sign NES at every cluster term** (nuclear pore +2.42 protein / −2.15 RNA; splicing +2.30 / −1.82; chromatin +2.35 / −1.53). *Validity/kills:* three rejected alternatives — **R-β statistical-frame** (same OLS frame), **R-γ batch** (47-level Batch factor), **R-δ cohort-overlap** (restricted to 463 dual donors) — so it is a real discordance, not a pipeline difference. *Scope consequence:* bounds the cluster claim to the protein level. → **new main slide alongside s28** (both are "change an input, re-run the identical pipeline"). **Strictly count+pattern+scope; flag the post-transcriptional reading as an interpretation made elsewhere (T44), not licensed by the slope.** This is a *completed* result the deck currently omits and contradicts.

**S18. Local property (h≤2).** Bounded h=2 (13 passes) > unbounded (6). ★ show the bin-stratified evidence on-slide ("within every bin cluster and non-cluster anchors shift identically — they move more only because they start steeper"); state structural vs interpretive ceilings separately. → **s29.**

★**S18b. Age — is the disease-vs-healthy contrast just aging?** *On-ramp:* C9-vs-Sporadic = two sick groups, same age (a no-op); C9-vs-Control = sick-vs-healthy where age could leak. *Method:* per-protein **partial R²(Age)** from OLS protein ~ Sex+group vs +Age; age imputed at within-Sex median, cross-checked with 5-draw MI (agree to 4 decimals). *Result:* **C9-vs-SPOR median 0.1–0.4% (age-robust); C9-vs-CTRL median ≤1.7%, leading-edge p95 ≤4.5%, 45.6% of proteins R²>0.01.** *Validity/honesty:* kills "your control contrast is an aging signature," with a number; **we deliberately did not run a full age-adjusted slope-GSEA rerun** (too small to reclassify a q≈0.001 cluster — and we say so; F1 disclosed as a |t|-level proxy). → **new beat in the robustness stretch** (pairs with the C9-vs-CTRL leg). Currently only a row in the skip-ledger.

**S19. One claim, seven falsifiers.** Consolidated ledger (READ): triangulation → shared-ALS + topology · size-matched → size inflation · degree-binned → hub · STRING → network-generic · bounded-vs-unbounded → deep-shell/component · 5 sensitivities → analytic-choice · age → confounding. → **re-cast s30** (eyebrow removed); every row already earned upstream.

### ACT 3 — WASC frontier + close (s31–s44) — already the best-built arc
**S20. Level → relationship.** gradient = a level (magnitude vs distance); WASC = a relationship (two proteins' co-movement). → **s32** (promote the strip one-liner to an axis caption). Already cogent — keep.

**S21. The per-pair anchor-slope.** Regress protein-B on anchor-A within a group; the tilt is the link. ★ add the spec: **anchor FWL-residualized vs Sex/Age_z/Tissue (Batch pre-residualized), slope in log2-abundance units** (currently absent entirely). → **s34.**

**S22. Fit in C9/SPOR/CTRL; name WASC.** **944 within-cluster hop-1 edges**; ★ render as a segmented bar **Splicing 434 · Chromatin 443 · Transport 67** (pre-arms the per-theme tripwire). → **s35–37.**

★**S23. The referee — Cochran-Q.** ★ Method on-slide: **Q = Σ wₖ(βₖ − β̄)², wₖ = 1/SEₖ²; LOW Q = invariant = WASC-positive** (lower-tail Phipson-Smyth — a technical reader will otherwise invert the direction). ★ Externalize the **matched-bin permutation null** (degree-decile × pooled-|Pearson|-decile over the all-protein pool) as a histogram with the lower tail shaded — "matched on degree × co-expression so within-pathway correlation cannot fake invariance." ★ **BY-FDR not BH** because edges share anchors/targets/null-draws. → **s33** (split into intuition + scoring micro-arc).

**S24. The 4-prong tripwire (3 PASS + 1 FAIL).** ★ Show all four prongs: **(a) null calibration FP 0.111 vs 0.120 — PASS · (b) SPOR-25 Jaccard ≥ 0.70 — the gate · (c) substrate N/A after v1.0.3 · (d) FW-vs-OLS exactness 9.4e-10 vs 1e-8 — PASS.** → **s39 setup.** Pre-registration is the validity hook.

**S25. The tripwire fired.** **Jaccard = 0.285 ≪ 0.70 → HARD HALT; production B=9999 blocked; no per-pair verdicts.** ★ surface "stable across 3 seeds AND all 3 themes (0.254/0.314/0.273)"; ★ the mechanism: **SPOR n=294 carried the precision; matching down to 25 recovers ~40% of full-N positives — a power finding, not a bug.** Three documented paths (A/B/C). → **s40.** Already exemplary; add the two precision lines.

**S26. The wall is the disease.** ★ add the **aggregate-at-25 receipt:** "the aggregate cluster claim cleared a 10k size-matched null at p<0.001 with these same 25 (Act 2); the per-pair gate did not" — grounds the level-asymmetry in a validated result instead of analogy. → **s41.**

**S27. Claim ceiling.** ★ add **"protein-level only (mRNA discordant)"** to the *cannot say / restrictions* column (the boundary the cross-modality result forces). Keep the forbidden-language discipline: the slope statistic alone licenses no post-transcriptional reading; the cross-modality sign-flip is what supplies it — a discordance to reconcile, not a mechanism claim. → **s42.**

**S28. Close / what's next.** ★ **reorder Next:** cross-modality reconciliation is the FIRST named pre-publication item (a must-resolve, *not* future work), then external replication, then more carriers for WASC; clarify the 100k-perm gate as "single-protein resolution only; does not move the pathway claim." **Fix the self-contradiction** (lines 1126/1131 + the note at 1138). → **s43–s44.**

---

## 3. Promotion & addition table

Priority: **P0** load-bearing break / integrity contradiction · **P1** technical thread breaks · **P2** cogency strengthener.

| slide(s) | step | lives now | cogency | promotion (strip/interlude → body) | currently OMITTED — must add | pri |
|---|---|---|---|---|---|---|
| s3/s4 | Data & cohort | main + strip | footnoted | denominators (3,264→423→3,257), UniProt unit, raw-linear caveat → body | **Control = 91 (not 59); 3,257-vs-3,264 reason; why-Sex (Table 1 imbalance)** | **P0** |
| new pre-s9 / s14 | **What \|t\| is + EB** | strip (s14, s11) | implied | **moderated t = effect ÷ SE; \|t\| = magnitude; model intensity~group+Sex; EB borrows strength across ~3,200** → first-class slide | the effect-over-SE definition; EB rationale (kills freak-small-variance); d0≈0.60 weak-shrinkage | **P0** |
| s9 | Self-contained ROAST | main + strip | footnoted | "curated 47-gene · self-contained MSQ · set vs no-change" → body | self-contained vs competitive named as the hinge | P1 |
| s10 | 56.5% FPR | main | delivered | — | target set = **12.5th percentile** of 200 controls | P2 |
| s11 | Wrong yardstick + competitive z | main + skip-table | implied | **mean\|t\| 1.0 vs 1.15 (~13%); competitive z=1.25/p=0.21** → body | name competitive z + the self-contained/competitive contrast | P1 |
| s11 | **Camera-VIF** | skip-interlude | footnoted | **VIF=1+(k−1)ρ̄; ρ̄=0.185→VIF=9.5→SE×3.08→z 0.41/p 0.69** → own half-slide | VIF chain + "corrects co-regulation not signal (3/3)" | P1 |
| new s10→s13 | **Cascade-collapse** | skip-table row | footnoted | **hop-1 0.019→0.69; 46/46→0/46 (π₀ 0.02→1.0)** → own main slide | the whole beat (deck's strongest integrity moment) | **P0** |
| s12 | Three-null ledger | skip-interlude | footnoted | un-skip → "Three references, one verdict" READ-ledger | matched-null p=0.65 as 3rd independent reference | P1 |
| s13 | **What survived + verdict** | speaker-notes only | omitted | RWR proximity into s13 body | **938 dist-1 genes, mean\|t\| 1.31 vs 1.06, p=0.001; effect real-but-tiny ~0.13–0.17u; n=25 power wall** | **P0** |
| s15 | Shell / hops | main + strip | footnoted | shell-as-ring-mean; **undirected** → body | shell statistic definition; "undirected" never stated | P1 |
| s16 | Measured-only | main + strip | footnoted | **with-intermediates saturates ~3,240 → flat; measured-only varies (median ~2,000) = slope power** → body | the validity payload (why it gives discriminating power) | P1 |
| s18 | **slope = ring2 − ring1** | strip | footnoted | **two-point WLS ≡ ring2 − ring1, weights cancel** → centerpiece body | the algebraic identity (the one equation; note says don't say the minus sign) | **P0** |
| s17 / s20 | **Degree-binned null + hub** | skip-interlude | footnoted | hub hazard/fix → re-titled non-skip s17 | hub-confound rationale; **VCL falsifier (3759 vs 3734, p≈0.92)** | **P0** |
| s15/16 | **Why depth 2** | notes only | omitted | new short beat | ~61% coverage = deliberate stop; cascade→component drift; bin-stratified test; structural vs interpretive ceiling | P1 |
| s19 | Every anchor | main + strip | delivered | — | 3,257-vs-~3,120 reconciliation (130 single-shell) | P2 |
| s21 | **Preranked GSEA + score=−slope** | strip | footnoted | **one ranked list; gene-set permutation; score=−slope → +NES = concentration** → split into 2 slides | what preranked GSEA is; the sign convention (T45 hinge); FDR=NES-histogram not BH; perm=1000 not 10k | **P0** |
| s22 | Three clusters | main + strip | delivered | Jaccard < 0.06 → on-slide | — | P2 |
| s24 → new | **4-pattern truth table** | skip-interlude | footnoted | **promote to main slide between s23/s25** | the licensing logic (empty cell kills shared-ALS + topology) | **P0** |
| s25 | Headline + Bonferroni-8 | main + strip | footnoted | pre-registration + "confirmatory readout" → body | 8 terms frozen a priori; raw-p<0.00625 ∧ NES>0 | P1 |
| s26 | Zero + size-matched null | main + strip | footnoted | two-threshold reconciliation on-slide; size-matched null its own beat | strict-0/8 vs loose-8 same story; 10k size-matched p<0.001 | P1 |
| s27 | Six-term core | main + notes | delivered | two-term honesty on-slide | Vpr (size-artifact) / nuclear-pore (SPOR-only) caveats | P2 |
| s28 | STRING | main + strip | footnoted | **regulator→regulatee vs stable-complex** rationale → body | why opposite sign is *validating* not failing | P1 |
| new ~s28 | **Cross-modality RNA-seq** | omitted | omitted | new main slide | **opposite-sign NES 8 terms (T44, OLS~Sex+Batch+Group, 463 donors); R-β/R-γ/R-δ rejected; protein-level scope** | **P0** |
| s29 | Local property | main + strip | footnoted | bin-stratified test on-slide; two ceilings separately | the evidence behind "two hops isn't a knob" | P1 |
| new ~s29 | **Age robustness** | skip-ledger row | footnoted | own beat | partial-R²(Age) asymmetry; MI agreement; deliberate no-rerun decision | P1 |
| s30 | Robustness ledger | skip-interlude | footnoted | un-skip → "One claim, seven falsifiers" main slide | every row also earned upstream | P2 |
| s33 | Cochran-Q referee | main + strip | footnoted | **Q formula; LOW Q = positive; matched-bin null; BY-FDR rationale** → body (2-slide micro-arc) | the statistic + direction + null + BY-vs-BH | P1 |
| s34 | Anchor-slope | main + strip | delivered | FWL-residualization → strip/body | per-edge covariate residualization | P2 |
| s35–37 | Name WASC | main + strip | footnoted | **944 → Splicing 434/Chromatin 443/Transport 67** segmented bar | per-theme decomposition | P2 |
| s39 | Tripwire setup | main + strip | delivered | 4-prong ledger | 3 PASS + 1 gate structure | P1 |
| s40 | Tripwire fired | main + strip | delivered | seed/theme stability + mechanism → body | stable across 3 seeds & 3 themes; ~40%-recovery power mechanism | P2 |
| s41 | The wall | main + strip | delivered | aggregate-at-25 receipt → body | tie to 10k size-matched null (Act 2) | P2 |
| s42 | Claim ceiling | main + strip | footnoted | "protein-level only (mRNA discordant)" → restrictions column | the protein-level restriction | **P0** |
| s43/44 | Close | main + strip + note | footnoted | reorder Next; cross-modality = pre-pub item | **fix self-contradiction (matched-mRNA framed as future while completed opposite-sign)** | **P0** |

---

## 4. What the deck must ADD (currently omitted, needed for cogency)

Grouped by where each belongs:

**Act 1 (foundational statistic + integrity):**
1. ★ **The `|t|` definition** — effect ÷ SE, magnitude not direction, model intensity~group+Sex. *(new pre-s9 slide or promoted s14)* — **the single most important addition.**
2. ★ **EB variance moderation** — borrow strength across ~3,200; posterior = own + fitted prior; weak shrinkage (d0≈0.60); on-ramp to raw-vs-log2. *(paired with #1)*
3. ★ **Camera-VIF chain** — ρ̄=0.185 → VIF=9.5 → z 0.41/p 0.69; "corrects co-regulation not signal." *(s11)*
4. ★ **Cascade-collapse** — 0.019→0.69, 46/46→0/46. *(new slide s10→s13)*
5. ★ **What survived** — RWR proximity p=0.001 (938 genes, 1.31 vs 1.06); + the **verdict** (real-but-tiny ~0.13–0.17u, n=25 power wall). *(s13)*
6. self-contained-vs-competitive named hinge; 12.5th-percentile specificity number. *(s11/s10)*

**Act 2a (gradient mechanics + null):**
7. ★ **slope = ring2 − ring1** algebraic identity. *(s18)*
8. ★ **hub hazard/fix + VCL falsifier** (p≈0.92). *(s17, dissolving s20)*
9. ★ **depth-2 interpretive ceiling** (~61% coverage, cascade→component drift, bin-stratified test, structural vs interpretive). *(s15/16)*
10. shell-statistic definition; "undirected"; measured-only validity payload; 3,257-vs-3,120 reconciliation.

**Act 2b (GSEA + triangulation):**
11. ★ **what preranked GSEA is + score=−slope sign convention (T45 hinge).** *(split s21)*
12. ★ **4-pattern truth table promoted to main slide.** *(s24→main)*
13. ★ **size-matched null as its own labeled beat** (10k, p<0.001, graph-free). *(s26)*
14. ★ **cross-modality RNA-seq discordance** (T44, opposite-sign 8 terms, R-β/γ/δ rejected, protein-level scope). *(new ~s28)*
15. ★ **age robustness beat** (partial-R²(Age) asymmetry, MI, deliberate no-rerun). *(new ~s29)*
16. pre-registration of 8 terms; confirmatory-readout framing; two-threshold reconciliation on-slide; STRING regulator→regulatee rationale; bin-stratified depth evidence; six-term two-term honesty; Jaccard<0.06 on-slide; enumerate the three nested nulls (degree-binned per-anchor · GSEA gene-set per-term · size-matched per-protein).

**Act 3 (WASC + close):**
17. ★ **Cochran-Q statistic + LOW-Q-=-positive direction + matched-bin null + BY-vs-BH.** *(s33)*
18. ★ **protein-level restriction in the claim ceiling.** *(s42)*
19. ★ **fix the cross-modality self-contradiction in the close** (matched-mRNA is completed/opposite-sign, a pre-pub reconcile item — not future work). *(s43/44)*
20. 4-prong tripwire structure; FWL-residualization; 944 per-theme split; seed/theme stability + power mechanism; aggregate-at-25 receipt.

---

## 5. DO → REVIEW batching (low-risk-first)

Inherits REWORK_PLAN §1 design spine and the **6 Slidev pitfalls** verbatim as the per-batch REVIEW checklist: (1) bake alpha into `rgba()`, never bind `:opacity`/`:fill-opacity`; (2) SVG text needs **px** font-size; (3) figures need `max-width` caps to clear the 48px strip reserve; (4) verify absolute strips on the normal `/N` route; (5) `layout:section` centers — SVG below the centered title; (6) table cells use inline `style="color:#…"`. **Gate:** do not start Batch N+1 until Batch N passes REVIEW.

**Cross-cutting invariants every batch must preserve:** the forensic-spiral arc + dark forensic-nocturne atmosphere; claim discipline (count+pattern, never mechanism/cause/drives/regulates [except naming INDRA edges]/controls/rewiring/validates/post-transcriptional); grounded numbers traceable to the breakdown; intuition kept as on-ramp; the gradient machine and WASC machine evolve (not repeat); the descent ends quieter than its loudest moment.

---

### BATCH 0 — Grounding & contradiction P0s (markdown/strip + one component literal; LAND FIRST, ~zero render risk)
Fold the still-valid grounding fixes from the prior plan so the rework builds on correct numbers.
- **Changes:** Control 91 everywhere (CohortBars literal 59→91; strips; ContrastTriangle); raw-linear strips (s3, s14) + transform-invariance footnote; s21 strip perms 10k→**1000**, "BH-FDR"→"NES-histogram FDR"; carrier number ~50–75; s17 body/notes rewrite to |t|-within-degree-bin shuffle (remove group-label-shuffle language); Jaccard <0.06 unify; "median 18"→"~15"; **fix the s43 close self-contradiction** (matched-mRNA completed/opposite-sign).
- **Preserve:** atmosphere, claim discipline, the existing strong centers.
- **REVIEW verifies:** every number traces to the breakdown; CohortBars shows 91 and agrees with its strip; no log2 claim survives; s17 body agrees with its own strip; the close no longer frames matched-mRNA as open future work.

### BATCH 1 — Dissolve interludes / kill SKIP framing (markdown-only; no SVG)
- **Changes:** delete the `FOR THE METHODOLOGIST · SKIP` eyebrow from s12, s20, s24, s30; re-title s12 → "Three references, one verdict" (READ ledger), s24 → "Why an empty third cell is the proof" (relocate between s23/s25), s30 → "One claim, seven falsifiers"; dissolve s20's hazard/fix card-pair into s17 (defer the VCL/figure work to Batch 4). Delete the speaker-note suppression instructions ("Say loudness not t-statistic," "Don't say those words," "avoid the word sign," "don't get into the minus sign").
- **Preserve:** the recessed-ledger *visual treatment* for the two surviving ledgers (density is honest); claim discipline.
- **REVIEW verifies:** zero `SKIP` eyebrows in `slides.md`; no load-bearing conclusion lives only on a ledger; s24 truth table is on the main path before the headline; speaker notes no longer instruct hiding method.

### BATCH 2 — Promote method into bodies + add omitted text beats (markdown-only; highest narrative payload, no SVG)
This is the core re-architecture: every promoted/new beat from §2–§4 that is *text*.
- **Changes:** new `|t|` + EB slide (foundational); Camera-VIF beat; cascade-collapse slide; "what survived + verdict" into s13; shell/undirected + measured-only validity into s15/s16; depth-2 interpretive-ceiling beat; slope=ring2−ring1 body on s18; degree-null hub rationale body on s17; split s21 into GSEA-intuition + GSEA-method (score=−slope hinge); pre-registration + confirmatory framing on s25; two-threshold + size-matched-null split on s26; STRING regulator→regulatee body on s28; **new cross-modality RNA-seq slide**; **new age-robustness beat**; bin-stratified evidence on s29; Cochran-Q method body + matched-bin null + BY-FDR on s33; FWL-residualization on s34; 944 per-theme split; 4-prong tripwire setup; aggregate-at-25 receipt on s41; protein-level restriction on s42; reordered Next on s43.
- **Preserve:** intuition on-ramps; claim discipline (cross-modality and STRING stay count+pattern+scope; post-transcriptional flagged as elsewhere-made); grounded numbers; the slow WASC build's pedagogical pacing.
- **REVIEW verifies (the cogency gate):** for each step, *cover the strip — can a technical reader still follow the step from the body?* Yes for every promoted beat. The score=−slope convention, LOW-Q=positive, and ring2−ring1 identity each appear in a body. Cross-modality and age are first-class beats, not ledger rows. No mechanism/post-transcriptional overclaim leaks past the discipline line. Term-introduction order still holds (no symbol used before defined: |t| before mean|t| before slope before NES before Q).

### BATCH 3 — Re-sequence (markdown-only; verify flow after moves)
- **Changes:** confirm the new running order: `|t|/EB` before ROAST; cascade-collapse between s10 and s13; depth-2 beat before s21; s24 truth-table before headline; cross-modality + age in the s28–s30 robustness stretch; "seven falsifiers" as Act-2 closer. Adjust act-divider promises and inter-slide handoff lines so each beat names what it hands forward.
- **Preserve:** the three-act tension→resolution arc; one strong center per slide; the quieter-ending descent.
- **REVIEW verifies:** no orphaned references (each "we now ask…" has its antecedent on a prior main slide); each step's handoff line names its consumer; the arc still reads as a forensic spiral end-to-end; <2s strong-center test passes per slide.

### BATCH 4 — Component edits for the new method visuals (SVG; isolate — highest render risk)
- **Changes:** `GradientSlopeAnatomy.vue` annotate the two ring means + show the subtraction; `PermutationNull.vue` two degree lanes swapping within-lane + the VCL flat-slope companion; new/extended component for the 4-pattern truth table (or styled HTML table) and the cross-modality paired protein(+)/RNA(−) NES barchart; `CochranQTriplet` referee stage gains the matched-bin null histogram with lower tail shaded; `TripwireBars` setup gains the 4-prong checklist; WASC-name slide segmented 944 bar; NeighborhoodTally split visuals for the two GSEA slides.
- **Preserve:** mark semantics constant deck-wide (dot=protein/anchor, ring=hop, bar=term, tilt=coupling); palette semantics (purple=knowledge, cyan=data, gray=correctly-quiet, amber=caution); motion only at the sanctioned moments (250–350ms ease-out, no loop, none on ledgers); the machine-evolves-not-repeats rule.
- **REVIEW verifies:** each new/edited SVG renders AND any motion fires once on the `/N` route (not just presenter); no `:opacity`/`:fill-opacity` bindings (alpha baked in `rgba()`); px font-size on all new SVG text; `max-width` caps clear the 48px strip reserve; the ring2−ring1 subtraction is legible in <2s; the cross-modality sign-flip reads as the picture; full-deck grounding + atmosphere regression pass.

**Why this order:** Batches 0–3 are markdown/strip text (cannot break renders) and carry the entire structural re-architecture — a technical reader could follow the whole walkthrough from the main flow after Batch 3 even with the old figures. Batch 4 (SVG) only sharpens visuals that the promoted text already makes followable, isolating the UnoCSS-crush and `/N`-route render risks to last.
