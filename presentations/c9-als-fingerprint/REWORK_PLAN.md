# C9-ALS-Fingerprint Deck — MAP→DO Rework Plan

> [!CAUTION]
> **SUPERSEDED PLANNING SNAPSHOT — not a current analysis authority (2026-07-12).** This document preserves the deck's design rationale but predates the production `log2(x+1)` rerun. Current state lives in `data/publication/c9_primary_analysis.json`: EB prior df `4.98 / 5.51 / 4.92`; bounded measured-only `h≤2` confirmatory `8 / 6 / 0`; unbounded sensitivity `6 / 0 / 0`; feature accounting `3,264 attempted = 3,117 valid + 137 without a reachable measured neighbor + 10 below the ≥10-neighbor guardrail`. Any raw-linear, `d₀≈0.60`, `7 / 6 / 0`, or `3,257 / 7 / 130 single-shell` statement below is historical context only.
> **Current integrity note:** this is a historical plan, not presentation authority. The audited same-data auxiliaries are now HGNC size-matched `8/8/0` and robust-scope degree-matched mean(`-slope`) `7/7/0` (Vpr is the sole C9 non-pass). STRING, matched RNA, age, abundance, legacy F5b, and five legacy sensitivity claims below remain outside the evidence stack. The historical 56.5% random observed-set rejection is not an FPR/calibration estimate.

> Historical DO→REVIEW snapshot. Slide indices use the **actual single-file `slides.md` numbering** (the deck has no offset; the ARC MAPs' "s14–s23 / s25–39 / s61–83" labels are reconciled here to real indices **s0–s44**). Numerical grounding reflects the superseded run unless the caution banner says otherwise.

---

## 1. Design Spine — the contract every slide inherits

**ATMOSPHERE — Forensic Nocturne.** Deep near-black ground (`#0a0f1a` / `#121a2d`). One disciplined accent does the pointing. The felt quality is an evidence board at 2am: something is being *tested*, not displayed. Domain-specific, never generic data-viz — cyan signal laid onto a purple literature-map, perturbation that radiates and fades through a network. The deck distrusts its own first answer; restraint is visible (no triumphant color, no celebratory motion until earned). The finding lands as a subtle glow, not a fanfare.

**COLOR SEMANTICS (strict, peripheral-channel, 2 bits/s — color always means a layer or a verdict, never decoration).** Bake alpha into `rgba()`; never bind `:opacity`/`:fill-opacity` (UnoCSS crushes to ~0.01). Table cells use inline `style="color:#xxx"` (theme `td` rule out-specifies utilities).

| Token | Hex | Means |
|---|---|---|
| CYAN | `#4ecadf` | DATA layer — blood proteins, loudness/\|t\|, the observed signal, the C9-positive answer |
| PURPLE | `#bf6ff7` | KNOWLEDGE layer — INDRA map, edges, neighborhoods, anything map-side |
| GREEN | `#7dd629` | biology/control AND **correct-silence-as-evidence** (0/8 empty column, "tilts agree", concord, aggregate-stands) — green = a thing being quiet when it should be |
| AMBER | `#f59e0b` | caution / the tripwire / "weak" verdicts / honest limits |
| RED | `#ef4444` | **RESERVED ONLY for the test that lied** (ROAST 56.5% FPR). Appears in Act 1 and essentially never again — its single use brands the betrayal |
| GREY | `#94a3b8` (quiet/not-yet) · `#64748b` (de-emphasis) | verdict greys |

**Signature pairing:** cyan-on-purple (introduced s5) must recur wherever data meets map. **Patient-label axis** (C9 mutation) is NEITHER cyan nor purple — use a neutral off-white/desaturated-gold accent.

**MOTION GRAMMAR — rationed to the single most important state-change, nowhere else.** Three sanctioned uses only, always ease-out/decelerate (never bounce):
1. **ARRIVAL of the one number that matters** — 56.5% (s10), 0/8 (s45), 0.285 tripwire (s40): 250–350ms settle so the gut registers before the eye reads.
2. **LAYERING gesture** — cyan data settling onto the purple map (s5, echoed at s42 cluster-on-map, optionally s0 cover).
3. **TRIPWIRE FIRING** — the one moment motion carries dread, an amber threshold crossed (s40).

NO ambient/looping motion. NO entrance animation on body text. **NO motion on interludes (stillness IS the skip-signal).** Act-divider transitions are active boundaries: a deliberate tonal shift, not a hard cut.

**TYPOGRAPHIC SCALE — three tiers / three channels.** FOCAL: one readable sentence at `text-lg/xl`, max one block per slide. HERO NUMBER: load-bearing figure at `2xl+`/mono, isolated with a whitespace moat, peak-shifted. PERIPHERAL/PROVENANCE (glance-only): MethodStrip mono `text-xs/sm` at foot; cyan SKIP eyebrow `font-mono text-xs tracking-widest` on interludes; secondary captions `#94a3b8`. **Mono = forensic/auditable voice** (every number, null spec, provenance). **Sans (Inter) = human/narrative voice.** Contract: nobody must read mono to follow the story; the PI reads ONLY mono to audit. SVG text uses **px font-size units** (UnoCSS strips unitless); figures need `max-width` caps to clear the 48px method-strip reserve.

**LEVELS OF SCALE — nested coherent wholes.** ACT (tension→resolution, marked by a colored `layout:section` divider + one-line promise) → SLIDE (one strong center: one focal block + one hero mark + one provenance strip) → FIGURE (a single component EVOLVING across slides, not repeating — the gradient machine `HopShells→MeasuredOnlyPaths→PermutationNull→GradientSlopeAnatomy→AllAnchorsField` is ONE machine assembled piece by piece; the WASC machine `CochranQTriplet[one]→[three]→[referee]` likewise) → MARK (dot = one protein/anchor; shell ring = one hop; bar = one term; tilt = one coupling — constant meaning deck-wide). **Anti-pattern to hunt: a component reused without visible evolution** (BonferroniMatrix s45/s47; CochranQTriplet skipping its `[three]` stage).

**STRONG-CENTER RULE.** Every slide answers in <2s: *what is this ONE slide about, and where do I look first (<200ms)?* ONE focal claim, ONE hero mark/number that IS the answer, everything else demoted (color, strip, caption). If two things compete for the center, cut or demote one. Hero gets isolation (whitespace moat) + peak-shift. Test: cover strip+caption — do focal block + hero mark still tell the whole story? Interludes are the deliberate exception (dense ledgers, density signaled honestly by the skip eyebrow).

**INTERLUDE TREATMENT — "a glass door you CHOOSE not to open."** DISTINCT: fixed cyan mono eyebrow top-left (`FOR THE METHODOLOGIST · SKIP FOR A GENERAL AUDIENCE`, tracking-widest, text-xs); body is a dense bordered ledger (table or hazard/fix card-pair); NO motion; recessed surface (slightly darker panel, hairline `#334155`/`#1a2540` borders). NOT-SEPARATE: same palette semantics, same mono voice, same dark ground. **NO MethodStrip** (the interlude IS provenance). Normal-flow content only (never absolute-positioned strips); table cells use inline `style` color. **Load-bearing narrative conclusions must NEVER live only on a skippable interlude.**

---

## 2. Deck Experiential Arc + Term-Introduction Order

**ARC (forensic spiral — each act asks, answers, distrusts the answer, asks finer):**
- **SETUP (s0–s7):** plant the human hook (two identical patients, one carries C9), teach the four nouns (ALS, C9-label, protein/loudness, the three groups + INDRA two-layer color), end on THE ONE QUESTION. Calm, low tension — the **25 is a loaded gun on the mantel**.
- **ACT 1 — THE TEST THAT LIED (s8–s13):** ROAST says loud YES → fed pure noise → fires 56.5% → wrong yardstick → [interlude: three nulls] → earthquake pivot. Resolves into a **deliberate dead-end**; pivot line: *presence is fakeable, change the question to SHAPE.*
- **ACT 2 — A BETTER QUESTION (s13–s50):** build the machine one window at a time → tally by neighborhood → THREE LIGHT UP → triangulation (the empty SPOR-vs-CTRL is load-bearing silence) → HEADLINE 7/8·6/8·0/8 → six-term core → tried to break it (STRING, local) → [interludes: degree-null s19, matrix s44, robustness s50]. Every consummation immediately stress-tested.
- **ACT 3 — A FINER QUESTION & THE WALL (s33–s44):** build WASC slowly → pre-registered tripwire → IT FIRES (0.285 vs 0.70) → **we STOP**. The wall is the disease (n=25), not the work. **End QUIETER than the loudest moment.** The 25 from the mantel has fired.

**TERM-INTRODUCTION ORDER (checkable):**

| Term | First-defined-at | Status / note |
|---|---|---|
| ALS | s1 | ✅ plain language, before use |
| C9orf72 / C9 carrier (LABEL, not map-center) | s2 | ⚠️ cyan misuse (P1) + s3 contradiction (P0) |
| protein vs gene; PBMC / blood draw | s3 | ⚠️ log2 strip (P0); C9orf72 exemplar contradicts s2 (P0) |
| differential model (defined as "loudness" at s13) | s3 strip | 🔴 **must be RAW-LINEAR, not log2** (P0) |
| the three groups / cohort (C9 25 · Spor 294 · Ctrl 91) | s4 | 🔴 **CohortBars hardcodes Ctrl=59** (P0) |
| INDRA / knowledge graph / cyan-on-purple two layers | s5 | ⚠️ signature shown only verbally (P1); "median 18"→~15 (P1) |
| the one question / H0 | s7 | ✅ verify \|t\| only in demoted mono |
| ROAST / self-contained / gene-set test | s8/s9 | ⚠️ green YES = lie wearing success-color (P1) |
| FPR / false-positive / negative control | s10 | ✅ 56.5% grounded; needs arrival motion (P1) |
| competitive z / Camera VIF / ρ̄ | s11 + interlude s12 | ✅ quarantined to mono |
| **shape (vs presence)** | s10 close / s13 | 🔴 pivot partly trapped behind skip-door s12 (P1) |
| epicenter / distance-decay / earthquake | s13 | ⚠️ single-center mental-model debt (P1) |
| loudness = \|moderated t\| | s13/s14 | 🔴 second log2 strip (P0) |
| hop / shell / neighborhood | s14/s15 | ✅ (directed-arrow vs undirected-distance minor, P2) |
| measured-only / no ghosts / BFS | s15/s16 | ⚠️ red X breaks palette (P1) |
| permutation null / shuffle | s16/s17 | 🔴 **body says group-label shuffle; method = \|t\|-within-degree-bin** (P0) |
| slope | s17/s18 | ✅ (radial-ring repeat, P2) |
| anchor / every protein its own epicenter | s18/s19 | ✅ |
| degree-binned null / hub / Guney 2016 | interlude s19/s20 | ⚠️ pre-empted by s17 strip (P1) |
| GSEA / slope-GSEA / named job; Bonferroni | s20/s41 | 🔴 **"10,000 perms"→1000; "BH-FDR"→NES-histogram FDR** (P1) |
| three neighborhoods / signature | s21/s42 | ⚠️ "<12%" vs "<0.06" same-slide contradiction (P1) |
| triangulation / three contrasts | s22/s43 | 🔴 **ContrastTriangle Ctrl=59** (P0) |
| headline 7/8·6/8·0/8 / NES | s24/s45 | ✅ grounded; needs arrival motion + green 0/8 (P1) |
| ~50× drop / 372·422·8 | s25/s46 | ⚠️ 0-vs-8 collision unbridged (P1) |
| six-term graph-invariant core | s26/s47 | ⚠️ evolution subtle, crush-risk (P1) |
| STRING / physical-PPI / opposite-direction | s27/s48 | ⚠️ sign unverified (presenter brief) + grey=null mis-color (P1) |
| local property / bounded-vs-unbounded / depth-2 | s28/s49 | ⚠️ cyan zero-bar (P2) |
| robustness ledger | interlude s29/s50 | ✅ ; age caveat one-sided (P2) |
| finer question / coupling / rise-and-fall-together | s30/s33 | ✅ |
| gradient (level) vs WASC (relationship) | s31/s34 | ✅ retro-names gradient |
| the tilt / the link | s32/s35 | ✅ |
| three tilts / per-group fit | s33/s36 | ⚠️ verdict-color drift; unused `[three]` stage (P1) |
| WASC = Within-cluster Anchor-Slope Concordance | s34/s37 | ✅ |
| Cochran-Q / referee / inverse-variance | s35/s38 | ✅ ; valence gap (P2) |
| tripwire / pre-registered gate / Jaccard | s36/s39 | ✅ |
| tripwire fires / HARD HALT | s37/s40 | ⚠️ needs dread-motion + 3 PI paths (P1) |
| wall is the disease / power wall | s38/s41 | 🔴 **"≈35–40 carriers" contradicts breakdown n≈50–75** (P1) |
| can / cannot say | s39/s42 | ✅ ; add WASC retraction + post-transcriptional scope (P2) |
| what it means / next | s40/s43 | 🔴 **"matched mRNA = next" contradicts COMPLETED opposite-sign analysis** (P1) |
| bookend answered | s41/s44 | ✅ quiet close |

---

## 3. Prioritized Rework Backlog

Priority key: **P0** = grounding error or broken strong center · **P1** = high (palette/term-order/missing sanctioned motion/load-bearing trap) · **P2** = medium · **P3** = polish.

| idx | Strong center (target) | Top issues (G/P/Ped) | Proposed rework | Pri | Risk | Components |
|---|---|---|---|---|---|---|
| s0 | Bookend question, unanswered | P: no domain atmosphere; signature not previewed | Add recessive purple INDRA-map fragment + 1–2 cyan proteins settling (low-intensity LAYERING). Title stays strong center | P2 | low | slides.md (s0 bg) |
| s1 | ALS = move-wires die; no cure | G: prognosis number verbal-only (ok); P: section centers; Ped: "signal fades" pre-primes s13 | Peak-shift "die" (enlarge+moat); **remove "the signal fades" caption**; keep SVG below title, verify on /N | P3 | low | slides.md (s1) |
| s2 | C9 = repeat that LABELS a group (not a map place) | G: s3 contradiction; **P: repeat segment is CYAN (data-layer error)**; P: duplicated in-SVG caption | **Recolor repeat segment cyan→neutral patient-label accent** (off-white/desat-gold); remove duplicated in-SVG caption | P1 | low | MutationAnchor.vue, slides.md |
| s3 | 3,264 proteins weighed in blood | **G: strip says log2 (fit is RAW-LINEAR)**; G: "436 samples"→423 analyzed; **G: ProteinReadout lists C9orf72 (contradicts s2)**; P: 6 exemplars over-spend focal | **Strip→raw-linear + transform-invariance footnote**; "436"→"423 analyzed (436 measured)"; **drop C9orf72 exemplar**, cut to 2–3 | **P0** | low | slides.md, ProteinReadout.vue |
| s4 | Three groups, one tiny (C9=25, loaded gun) | **G: CohortBars hardcodes Healthy=59** (contradicts own strip "91" + breakdown CTRL=91) — wrong number is the one shown | **CohortBars count 59→91**; verify "91" mono label clears viewBox right edge; keep all color semantics | **P0** | low | CohortBars.vue |
| s5 | INDRA map; PURPLE=knowledge, CYAN=data | **P: signature ASSERTED not SHOWN (no cyan on slide)**; G: "median 18"→~15 | **Add 2–3 cyan dots settling onto purple nodes (LAYERING motion, 300ms ease-out, once)**; strip "median 18"→"median ~15"; verify absolute strip on /N | P1 | med | KnowledgeGraphIntro.vue, slides.md |
| s7 | The one question | P: best-composed; Ped: verify \|t\| only in demoted mono | **Verify-only**: \|t\| appears nowhere in general-track text pre-s13. Optional calm ghost-map. Else leave alone | P3 | low | slides.md (verify) |
| s8 | The obvious test gave a loud answer — and it lied (Act-1 divider) | P: red brand not foreshadowed | Render "a lie"/"lied" in `#ef4444` to plant the betrayal color; optional gyori tonal settle on entry (250ms). **No number on divider** | P3 | low | slides.md (s8) |
| s9 | ROAST said a loud YES | **P: green YES = lie wearing the success/correct-silence color** | **Recolor YES neutral-bright (`#f1f5f9`/`#cbd5e1`), remove green checkmark** — assertion voice, not verdict color; prep for s10 retraction | P1 | low | FirstTest.vue |
| s10 | 56.5% — smoke detector fired on nothing | G: grounded-perfect; **P: hero number static (sanctioned arrival motion missing)** | **Add arrival settle to 56.5% (250–350ms ease-out)**; optional left-to-right barcode fill; optional struck-through green YES echo | P1 | low | SmokeDetector.vue |
| s11 | Wrong yardstick = change-from-zero in a body where everything moved | P: 3 competing centers; P: right panel reads as "rescue"; G: strip "p=0.21→0.69" ambiguous | **Cut opening to ONE focal sentence**; demote/inset or remove right panel (let s12 ledger deliver "ordinary"); **move SHAPE pivot to s13**; fix strip "p=0.21 → p=0.69 (z 1.25→0.41…)"; rhyme crowd-cloud with s16 | P1 | med | WrongYardstick.vue, slides.md |
| s12 | Three nulls; ROAST survives none that discriminate (interlude) | **Ped: SHAPE pivot trapped behind skip-door**; **G: cascade-collapse (0.019→0.69, 46→0) missing**; G: matched-row label | **Move pivot to s11 close / s13 open**; **add cascade-collapse row + 3 provenance footnotes** (FPR artifact-recorded; competitive triad sex-only/no-JSON; 25/294 vs 23/282 caveat); refine matched row→"sex-matched subsample / p=0.65"; no motion | P1 | med | slides.md (s12) |
| s13 | Test the SHAPE — distance-decay (Act-2 divider, earthquake) | **P: single-epicenter model (deck must dismantle s17–18)**; P: radial-ring repeat; Ped: own the SHAPE intro | De-jargon strip (drop premature "mean\|t\|"/"per-anchor"); **generic/multi-epicenter cue** ("every protein is its own epicenter"); single radiate-once motion (ease-out, no loop); be sole SHAPE intro | P1 | med | EarthquakeIntuition.vue, slides.md |
| s14 | Loudness = one number per protein (magnitude, never why) | **G: strip says log2 (RAW-LINEAR) — deck's 2nd log2 lie** | **Strip→"loudness = \|EB-moderated t\| from raw-intensity ~ group + sex · magnitude, not direction"**; optional "louder→" cue | **P0** | low | slides.md, VolumeKnob.vue |
| s15 | Distance = hop count; neighborhood = within 2 steps | G: directed arrows vs undirected distance; P: clean | Soften arrowheads to plain lines OR strip note "arrows=literature direction; distance undirected"; defer "measured-only" to s16 | P2 | low | HopShells.vue, slides.md |
| s16 | A RULE: walk only on measured proteins (no ghosts) | **P: ghost X is RED (breaks red-reserved-for-ROAST)** | **Recolor ghost X red→amber (`#f59e0b`)** or drop X, keep gray-dashed + "forbidden" label; own "measured-only" here | P1 | low | MeasuredOnlyPaths.vue |
| s17 | Shuffle to a random cloud; the real answer beats it | **G: body/notes say GROUP-LABEL shuffle; method = \|t\|-WITHIN-DEGREE-BIN shuffle** (self-contradicts own strip + s20 + breakdown) | **Rewrite body+notes to \|t\|-shuffle** ("deal loudness numbers to random proteins on the map…"); **trim degree-bin/Guney specifics OUT of strip** (leave general perm idea + p-formula) so s20 earns first disclosure | **P0** | med | slides.md (s17 body/strip/notes), PermutationNull.vue |
| s18 | One protein's slope — the tilt | P: left ghost-ring panel = 3rd radial repeat, competes with hero line; G: "inverse-variance" wording | **Cut left ghost-ring panel** (or strongly commit "rings become distance" bridge) so the tilted line is sole hero; soften "inverse-variance weighted by shell size"→"weighted by shell size" | P2 | low | GradientSlopeAnatomy.vue, slides.md |
| s19 | 3,257 anchors; do steep slopes pile up? | **G: "≈90 min full proteome pass" ungrounded** | **Remove/source "≈90 min"** ("every measured protein run as an anchor · ~3,120 valid slopes"); optional: let pile-up glow win first fixation | P1 | low | slides.md, AllAnchorsField.vue |
| s20 | Naive shuffle wrong (hubs); fix = degree-bin perm (interlude) | Ped: redundant w/ s17 strip (term-order "hub first here") | Make this the FIRST hub/degree-bin/Guney disclosure (coupled w/ s17 strip trim); optional honest line "binning uses full-INDRA degree; traversal measured-only (disclosed asymmetry)"; no motion | P2 | low | slides.md (s20 + s17 coord) |
| s41 (Tally) | Steep slopes pile up in a few named jobs | **G: strip "10,000 perms"→1000; "BH-FDR"→NES-histogram FDR**; G: invented decimals 1.00/0.86/0.71 | **Strip perms→1000; FDR→"GSEA NES-histogram FDR (gseapy/Subramanian)"**; relabel y-axis honest, drop invented decimals; surface pre-registration on-slide; peak-shift 3 hot bars + faint purple cue | P1 | low | NeighborhoodTally.vue, slides.md |
| s42 (Clusters) | Three barely-overlapping neighborhoods on the map | **G: "<0.06" (SVG) vs "<12%" (strip) same-slide contradiction**; P: layering static; P: duplicate SVG title | **Unify both to "Jaccard < 0.06"**; "Transport"→"transport"; cut SVG title; **add LAYERING motion** (clusters settle onto purple, ~300ms); verify opacity on /N; body names 3 + speaks "signature" + bridges neighborhood/job | P1 | low | ThreeClusterSignature.vue, slides.md |
| s43 (Triangle) | Same test, three comparisons | **G: ContrastTriangle Ctrl=59 (contradicts s4's 91)** | **ContrastTriangle 59→91**; re-treat SPOR-vs-CTRL edge as visual focus (amber, "should be empty if C9-specific"); put prediction on-slide; name "triangulation" | **P0** | low | ContrastTriangle.vue, slides.md |
| s44 (Reading matrix) | Four patterns, only one = C9-specific (interlude) | Ped: "names the cause" over-reads; rhythm: pre-loads before peak | "names the cause"→"attributes the signal to the C9 mutation"; **consider relocating after s45** (design→headline→interlude); link to s19 degree-null | P2 | med | slides.md |
| s45 (Headline) | 7/8 · 6/8 · 0/8 — control correctly empty | G: grounded; **P: 0/8 static (sanctioned arrival motion); 0/8 grey not green**; G: row-name drift | **Stage reveal so 0/8 settles LAST (250–350ms)**; consider tinting 0/8 GREEN (correct-silence); "Capped-intron…"→"Processing Capped Pre-mRNA"; verify/soften "~5 effective"; cyan checks dominate grey gaps | P1 | med | BonferroniMatrix.vue, slides.md |
| s46 (Drop) | Negative control ~50× quieter — silence IS evidence | **Ped: 0/8→8 collision unbridged** (reader thinks control lit up); P: verbal-green vs grey-stub | **Bridge 0/8→8** ("looser count: 372 & 422 vs 8 — and zero clear the strict bar"); green-accent the stub; keep static (reserve motion for s45) | P1 | low | TriangulationDrop.vue, slides.md |
| s47 (Core) | Six-term graph-invariant core | **P: redundant center — evolution subtle (`fill-opacity 0.08`, crush-risk) → reads as verbatim s45 repeat** | **Bake band alpha into rgba()** (`rgba(78,202,223,0.08)`), verify on /N; **dim non-core ROW CELLS** for isolation; on-slide note for 2 exceptions; guard against "drives/master-regulator" | P1 | med | BonferroniMatrix.vue, slides.md |
| s48 (STRING) | Swap the map; pattern mirrors → network part is INDRA-specific | **P: STRING drawn GREY (reads as null, not "a different knowledge map")**; G: sign convention UNVERIFIED (presenter brief); P: too flat | **Recolor STRING as distinct knowledge hue carrying same cyan data** ("same signal, different map"); make taller, enlarge near/far; cut SVG title; **brief presenter on unverified sign + pinned-constant status**; don't harden on-slide | P1 | med | SubstrateMirror.vue, slides.md |
| s49 (Local) | Strongest within 2 hops; local (13) beats unbounded (6) | **P: unbounded C9-vs-Ctrl "0" bar is CYAN (other zeros grey)**; P: totals too small | **Zero-bar cyan→grey**; make totals (13, 6) dominant; strip: "local (h≤2) interpretive ceiling — graph reaches h=4 but slope stops measuring cascade past h=2" | P2 | low | LocalVsGlobal.vue, slides.md |
| s50 (Ledger) | Each test kills a specific alternative; claim survives all (interlude) | Ped: age caveat one-sided; "age partial-R²" cold | Add C9-vs-CTRL age caveat ("0.1–0.4% C9-vs-SPOR; ≤1.7% C9-vs-CTRL, disclosed"); gloss age-as-confounder ("we model Sex; checked age doesn't drive it"); ensure s50→s51 re-tints to cogex | P2 | low | slides.md |
| s33 (Act-3 divider) | Found WHERE; now ask if two proteins move together same way in C9 | P: "rise and fall together" tinted purple (it's a DATA relationship) | Tint "rise and fall together" CYAN; leave "where" purple. Optional foreshadow of the 25 | P3 | low | slides.md (s33) |
| s34 (AxisContrast) | The axis swap IS the question: level vs relationship | P: both panels equal-weight (left is the OLD thing); Ped: "anchor" reuse | **Demote left (level) panel to grey/recessed**, keep right full-cyan; 3–4-word "anchor" bridge to s18; no coupling numbers here | P2 | low | AxisContrast.vue, slides.md |
| s35 (CochranQ[one]) | Two map-linked proteins per patient; the tilt IS the link | G: strip omits covariate residualization (PI's first objection) | Add "covariate-residualized anchor (Sex · Age · Tissue · Batch)" to strip; keep focal "the tilt IS the link" | P2 | low | slides.md (s35) |
| s36 (CouplingOutcomes) | Fit the tilt 3× — line up (same link) or fan apart | **P: "same link" cyan / "different link" AMBER (mis-cues "different=bad", pre-spends tripwire color)**; P: unused `[three]` stage | **Recolor: same/concord→GREEN, different→neutral GREY**; keep amber exclusive to s40; **prefer rendering via CochranQTriplet stage='three'** (machine evolves in place) or make s38 referee descend from this fan | P1 | med | CouplingOutcomes.vue, CochranQTriplet.vue, slides.md |
| s37 (Name WASC) | WASC = Within-neighborhood · Anchor-Slope · Concordance | G: strip missing per-theme breakdown | Add "Splicing 434 · Chromatin 443 · Transport 67" to strip; half-line that 944 pairs live inside the SAME three neighborhoods | P3 | low | slides.md (s37) |
| s38 (CochranQ[referee]) | Cochran-Q referee: small Q = concord | Ped: valence gap (small Q = the HOPED outcome); P: doesn't descend from s36 fan | Extend small-Q line with valence (green: "the link holds = what we'd hope to find"); surface "each of 944 pairs gets one Q"; make referee frame descend from s36; still | P2 | low | CochranQTriplet.vue, slides.md |
| s39 (Tripwire setup) | Shrink Sporadic 294→25, require Jaccard ≥0.70 before any verdict | G: "70% of edges reproduce" ≠ Jaccard (intersection/union) | Soften body gloss ("the same pairs must mostly come back — overlap ≥0.70"); keep "Jaccard ≥0.70" in strip; note 1-of-4 prongs | P2 | low | TripwireBars.vue, slides.md |
| s40 (Tripwire fired) | 0.285 amber falls short of 0.70 green — we STOPPED | **P: dread-motion missing (the deck's designated tripwire-firing motion)**; **G: 3 PI paths (A/B/C) absent** | **Add the one sanctioned tripwire-firing motion** (amber bar settling, 250–350ms ease-out); **add 3 PI paths** (A publish-inconclusive · B hierarchical pivot · C acquire ~50–75 carriers); soften "29% held up"; one-clause bridge to s41 | P1 | med | TripwireBars.vue, slides.md |
| s41 (The wall) | n=25 = rare-mutation fact; aggregate stands, per-pair doesn't | **G: "≈35–40 carriers" contradicts breakdown n≈50–75** | **"≈35–40"→"~50–75 carriers (Path C)"** (strip + body line 1029); make 25 callback visual (s5 cyan styling); optional Act-1 power-wall callback; say headline does NOT depend on WASC | P1 | low | slides.md (s41) |
| s42 (Can/Cannot) | CAN vs CANNOT discipline ledger | Ped: WASC retraction implicit; missing post-transcriptional scope | Make WASC retraction explicit in CANNOT ("whether the link changes — WASC halted"); add post-transcriptional/mRNA scope item (or ensure s43 fixed); cyan-CAN/grey-CANNOT; no motion | P2 | low | slides.md (s42) |
| s43 (Means/Next) | Blood-detectable C9-specific signature; honest scope + next | **G: "matched mRNA = next" contradicts COMPLETED opposite-sign cross-modality analysis** | **Remove "matched mRNA" from next OR reframe to disclose completed opposite-sign result** (post-transcriptional reading in the discordance); align carrier number with s41 (~50–75); keep replication/100k-perm/sub-strat; keep quiet | P1 | low | slides.md (s43) |
| s44 (Bookend) | YES — a fingerprint in 3 neighborhoods; caveat: group pattern, not per-patient | ✅ correct quiet close | Keep as-is; ensure "three neighborhoods" matches splicing/chromatin/transport; optional nod to the tripwire-halted per-patient question | P3 | low | slides.md (s44) |

---

## 4. Deck-Wide Issues (cross-cutting)

1. **🔴 P0 — LOG2 vs RAW (s3, s14, + any recurring differential strip).** Breakdown is emphatic: the production fit is on **RAW LINEAR intensities**, not log2. The deck asserts `log₂(intensity)~group+sex` in **two** strips. This contradicts the source of truth AND the deck's own credibility currency (the PI trusts it *because* it volunteers the raw-not-log2 disclosure). Fix every strip to raw-linear; add the transform-invariance footnote (`|t|` r≈0.93, NES same-sign; pathway result stable; gene-level naming churns ~30%) at least once so the disclosure becomes a strength. **Single most important deck-wide fix.**

2. **🔴 P0 — HEALTHY 59 vs 91 (s4 CohortBars, s43 ContrastTriangle).** Both hardcode `59` (the WASC post-exclusion sub-cohort) against production CTRL=91. The wrong number is the one the audience SEES (a bar; a triangle vertex). The breakdown explicitly warns against laundering the WASC "59" into the main-story cohort. Two one-line fixes (`59→91` in each). *Note: the WASC arc's Control IS 59 — but no on-slide count is shown there, so leave it; if any WASC count is ever surfaced it must be 25/294/**59** with a footnote, never 91.*

3. **🔴 P0 — PERMUTATION-NULL CONTRADICTION (s17).** Body/notes describe a **group-label/phenotype shuffle** ("rip the C9-vs-other labels off the patients"); the production null is a **\|t\|-within-degree-bin shuffle** (holds the differential vector fixed, reassigns across degree-matched nodes). The s17 strip and s20 interlude both correctly describe the degree-binned \|t\| permutation — so the deck contradicts itself within one slide AND across the s17→s20 handoff. Rewrite the body to the \|t\|-shuffle (a plain-language version preserves "shapes in clouds").

4. **P1 — REDUNDANT/UN-EVOLVED COMPONENTS (the deck's named anti-pattern).** (a) **BonferroniMatrix s45/s47** — evolution via `:highlight-core` exists but is subtle (`fill-opacity 0.08`, SLIDEV crush-risk) and with s46 between, eye goes matrix→bars→matrix; if the band crushes it reads as a verbatim repeat. Bake band alpha into rgba(), dim non-core cells, verify on /N. (b) **CochranQTriplet `[three]` stage unused** — the WASC machine should evolve one→three→referee in one component; the deck routes s35 (`[one]`)→s38 (`[referee]`) and substitutes CouplingOutcomes for the middle, so the machine doesn't visibly evolve in place. (c) **Radial-decay motif** repeated at s13 / s18-left / s19-glow with too little evolution — s15 (dashed hop-rings) and s19 (2D pile-up) are legitimate evolutions; s13 and s18-left are near-duplicates → cut s18-left.

5. **P1 — RED PALETTE LEAK (s16 MeasuredOnlyPaths ghost X).** Red is reserved ONLY for the test that lied (ROAST). A second red in Act 2 breaks the 2-bit/s color contract. Recolor to amber or drop the X.

6. **P1 — SIGNATURE ASSERTED-NOT-SHOWN (s5).** The keystone slide that DEFINES cyan-on-purple renders only purple; it tells the audience the pairing in body text but never demonstrates cyan settling onto purple — the literal act the method performs, and the sanctioned LAYERING motion. Teach it visually here, preview at s0, echo at s42.

7. **P1 — SHAPE PIVOT TRAPPED BEHIND SKIP-DOOR (s12).** The Act-1→Act-2 hinge ("abandon gene-set enrichment for a SHAPE statistic randomness can't fake") sits on the interlude explicitly labeled SKIP. A general reader who takes the invited skip arrives at the earthquake without the presence→shape bridge. Move the pivot to s11 close / s13 open.

8. **P1 — MISSING CASCADE-COLLAPSE (s12).** The breakdown's dramatic heart of Step 3 — the celebrated early cascade (hop-1 p=0.019, 46/46 two-hop "significant") WAS raw ROAST, and the corrected statistic collapses it (0.019→0.69, 46→0) — appears **nowhere** in the deck. It is the strongest "we adversarially attacked our own day-one triumph" evidence and the PI's biggest trust-builder. Home: the s12 interlude.

9. **P1 — MOTION RATION UNSPENT AT EVERY ARRIVAL.** The three sanctioned motions are all missing where the GROUND demands them: 56.5% arrival (s10), the LAYERING gesture (s5/s42), 0/8 settling-in-last (s45), and the tripwire firing (s40). Spend the ration at exactly these points and nowhere else (no ambient motion, none on interludes).

10. **P1 — VERDICT-COLOR DRIFT.** Green is correct-silence/concord-as-success. Violations: s9 lie wears green; s36 "same link/concord" drawn cyan (should be green) and "different link" amber (pre-spends the tripwire's signature dread color); s45 0/8 grey not green; s46 verbal-green vs grey stub; s49 collapsed-to-zero bar cyan. Harmonize so green = quiet-when-it-should-be and amber stays exclusive to the s40 tripwire.

11. **P1 — UNGROUNDED/CONTRADICTED RESOLUTION NUMBERS in the descent (where the PI is most attentive).** s41 "≈35–40 carriers" contradicts the breakdown's Path-C target n≈50–75; s43 "matched mRNA = next" contradicts a COMPLETED opposite-sign cross-modality analysis the breakdown says "must be resolved before publication." Both self-inflicted punctures to the arc's central virtue (honest bounding).

12. **P1 — THE THREE PI PATHS (A/B/C) ABSENT (s40/s43).** The breakdown's Step-8 "hands off" IS the structured decision; the deck resolves the tripwire as "we stopped" + "more carriers." For Track A this reads as "we gave up" rather than "pre-registered halt + three documented moves." Surface compactly on s40.

13. **P1 — GSEA STRIP ERRORS (s41).** "10,000 perms" → the headline 372/422/8 were produced at **1000** perms (10k belongs only to the per-protein size-matched null, correct on s46). "BH-FDR" → the code emits **GSEA NES-histogram FDR**; the BH path is unreachable.

14. **P1 — SAME-SLIDE CONTRADICTION (s42).** SVG caption "cross-overlap < 0.06" vs strip "<12%". Grounded value is Jaccard <0.06; "<12%" is unsourced. Unify.

15. **P2 — INTERLUDE-RHYTHM INVERSION (s44).** s44 is the only interlude that PRE-loads its payoff (sits before the s45 headline) instead of discharging a prior slide's debt (s12/s19/s50 follow theirs). Consider moving after s45 (design→headline→interlude), which also makes s45/s47 adjacent and eases the matrix→bars→matrix whiplash.

16. **P2 — PRESENTER-AWARENESS BRIEFS (no slide change).** (a) s48 STRING "opposite-direction" rests on a sign convention the breakdown flags UNVERIFIED (possible silent double-negation), and STRING is pinned-constants not code-reproducible. (b) s45/s46 provenance split: 7/8·6/8·0/8 are measured-only counts; 372/422/8 are with-intermediates Wave-24j counts — defensible single story, but no slide should assert they come from the identical run.

17. **✅ POSITIVES — do NOT "fix" into blandness.** The tension→resolution spine is sound across all three acts; transitions are live (no dead cuts; s6→s7, s8→s9, s10→s11→s12, s19→s21, s37→s40→s41→s44 all hand off cleanly). The slow WASC build (s34→s35→s36→s37→s38) honors term-order precisely — its slowness IS the pedagogy; do not collapse it. The descent to a precisely-bounded claim (not a crescendo) is the intended RELEASE; the deck must END quieter than its loudest moment.

---

## 5. DO→REVIEW Batching Plan

**Sequencing principle:** low-risk markdown/strip-only text edits land FIRST (cannot break renders); component recolors next (one prop/attr each); SVG layout/motion rewrites isolated LAST (highest render risk). Within each batch, every change inherits the §1 design spine. After EACH batch: build + `/N` print-route render check, grounding re-check against the values in this doc, the 6 Slidev pitfalls, atmosphere coherence, strong-center <2s, term-order.

> **The 6 Slidev pitfalls (REVIEW checklist for every batch):** (1) bake alpha into `rgba()` — never bind `:opacity`/`:fill-opacity` (UnoCSS crushes to ~0.01); (2) SVG text needs **px** font-size units; (3) figures need `max-width` caps to clear the 48px strip reserve; (4) verify absolute-positioned strips render on the **normal `/N` route** (not just presenter); (5) `layout:section` CENTERS content — keep SVG below the centered title; (6) table cells use inline `style="color:#xxx"` (theme `td` out-specifies utilities).

---

### BATCH 1 — P0 grounding, markdown/strip-only (LAND FIRST, zero render risk)
**Changes:** s3 strip → raw-linear + transform-invariance footnote; s3 "436"→"423 analyzed (436 measured)"; s14 strip → raw-linear; s41 strip "10,000"→"1000" and "BH-FDR"→"GSEA NES-histogram FDR"; s19 strip remove "≈90 min"; s17 body+notes rewrite to \|t\|-shuffle + trim degree-bin specifics from strip; s42 strip "<12%"→"Jaccard < 0.06"; s5 strip "median 18"→"median ~15".
**Must honor:** mono = auditable voice; Track-B never forced to read mono; the deck's credibility = volunteering the raw-not-log2 disclosure.
**REVIEW verifies:** every number traces to this doc's verified values; s17 body now AGREES with its own strip + s20 + breakdown (no group-label-shuffle language remains); transform-invariance footnote present at least once; render unchanged on /N (text-only); no \|t\| in general-track text before s13.

### BATCH 2 — P0 component count fixes (one literal each, near-zero risk)
**Changes:** CohortBars `count: 59→91`; ContrastTriangle `control.count '59'→'91'`; ProteinReadout drop C9orf72 exemplar (cut to 2–3); MutationAnchor repeat segment cyan→neutral patient-label accent + remove duplicated in-SVG caption.
**Must honor:** color=layer contract (cyan=data, purple=knowledge, neutral=patient-label); the 25 is the loaded gun — keep its cyan/amber treatment; green Healthy bar = substantial control arm but must not steal focus from the cyan-25 stub.
**REVIEW verifies:** CohortBars strip "91" now AGREES with the bar; "91" mono label clears viewBox right edge; ContrastTriangle Ctrl=91 agrees with s4; s3 ProteinReadout no longer lists C9orf72 (agrees with s2); MutationAnchor repeat is no longer cyan; render on /N; strong-center (25 stub / triangle SPOR-CTRL edge) still <2s.

### BATCH 3 — P0/P1 narrative restructure, markdown-only (text moves, no SVG)
**Changes:** move SHAPE pivot from s12 → s11 close (isolated peak-shifted) / s13 open; cut s11 opening block to ONE focal sentence; fix s11 strip "p=0.21 → p=0.69 (z 1.25→0.41)"; s12 add cascade-collapse row + 3 provenance footnotes + matched-row label fix; s13 own the SHAPE intro + de-jargon strip; s40 add 3 PI paths + bridge to s41 + soften "29% held up"; s41 "≈35–40"→"~50–75 carriers"; s43 fix matched-mRNA contradiction; s46 bridge 0/8→8; s42 body (names 3 + "signature" + neighborhood/job bridge); s50 age caveat + gloss; s33 cyan-tint; s34 anchor bridge; s35 covariate strip note; s37 per-theme strip; s38 valence line; s39 Jaccard gloss; s45 row-name fix; s44/s48/s49 strip wording.
**Must honor:** load-bearing conclusions never live only on a skippable interlude; claim ceiling = count+pattern (no mechanism/drives); descent ends quieter; SHAPE introduced once.
**REVIEW verifies:** SHAPE appears on a non-skippable slide before s13; cascade-collapse grounded (0.019→0.69, 46→0); s43 no longer presents matched-mRNA as open (or honestly discloses opposite-sign); carrier number = ~50–75 everywhere; 0/8→8 bridge prevents "control lit up" misread; 3 PI paths present on s40; no mechanism/post-transcriptional overclaim; term-order table still holds.

### BATCH 4 — P1 component recolors (single attr/prop each, low render risk)
**Changes:** FirstTest YES green→neutral-bright (`#f1f5f9`), remove green checkmark; MeasuredOnlyPaths ghost X red→amber; CouplingOutcomes same-link→green / different-link→grey; LocalVsGlobal unbounded-zero-bar cyan→grey; SubstrateMirror STRING grey→distinct knowledge hue (keep cyan data); TriangulationDrop stub green accent; ContrastTriangle SPOR-CTRL edge → amber focus.
**Must honor:** RED reserved for ROAST only (Act 1); GREEN = concord/correct-silence; AMBER exclusive to the s40 tripwire (s36 must NOT use amber); the lie wears neutral, not green; STRING = a *different knowledge map*, not "null/grey".
**REVIEW verifies:** grep components for stray `#ef4444` outside ROAST slides (s9/s10/s12 region) → none; grep for `#f59e0b` → only s40 (+ s4 "25" note, s46 ~50× bracket, s16 amber X); s36 concord green / different grey; s49 all zero-bars grey; baked-alpha pitfall on any new fills; /N render; <2s strong center per slide.

### BATCH 5 — P1 SVG layout + the sanctioned motions (ISOLATE — highest render risk)
**Changes:** s5 KnowledgeGraphIntro add cyan dots settling onto purple (LAYERING motion); s10 SmokeDetector 56.5% arrival settle; s13 EarthquakeIntuition generic/multi-epicenter + radiate-once; s18 cut left ghost-ring panel; s42 ThreeClusterSignature LAYERING motion + cut SVG title; s45 BonferroniMatrix staged reveal (0/8 last) + optional green 0/8; s47 BonferroniMatrix bake band alpha into rgba() + dim non-core cells; s48 SubstrateMirror taller + enlarge labels + cut SVG title; s40 TripwireBars amber-bar dread-settle; s36/s38 CochranQTriplet machine continuity (referee descends from s36 fan, or route s36 through `[three]`).
**Must honor:** motion ONLY at the 3 sanctioned moments (arrival / layering / tripwire), 250–350ms ease-out, no loop, no bounce, none on interludes; one strong center per slide after panel cuts; the gradient machine and WASC machine must visibly EVOLVE, not repeat.
**REVIEW verifies:** **each animated SVG renders AND the motion fires once on /N** (not just presenter); no `:opacity`/`:fill-opacity` bindings introduced (alpha baked in rgba) — especially the s47 band (would re-trigger the redundant-center anti-pattern if crushed); s47 visibly differs from s45 in <2s; s18/s42 have a single uncontested center after panel/title cuts; px font-size on all new SVG text; max-width caps clear the strip reserve; s36→s38 read as one evolving figure; atmosphere (forensic nocturne, subtle glow not fanfare) intact; full deck term-order + grounding regression pass.

**Gate between batches:** do not start Batch N+1 until Batch N passes its REVIEW. Batches 1–3 (markdown) can be reviewed together if the build is clean; Batches 4 and 5 must each be reviewed independently because component edits carry the UnoCSS-crush and /N-route render risks that text edits cannot.
