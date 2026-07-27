---
theme: ../slidev-theme-gyori-cogex
title: "A within-cohort C9-associated protein-neighborhood pattern"
info: |
  ## An audited C9-ALS proteomic consistency analysis
  Gradient discovery on a knowledge graph, with the evidence boundary made explicit.

  Gyori Lab — Northeastern University
author: Eric Jing Mockler
transition: slide-left
highlighter: shiki
mdc: true
---

::title::
# Does your blood know<br>what the doctor can't?

::meta::
**Eric Jing Mockler**<br>
_Gyori Lab, Northeastern University_

June 2026

<!--
Read it slowly to a tired room; plant the question, explain nothing.

Picture two people. Same age. Same first symptom — a weak hand. Same diagnosis: ALS. Same prognosis. To a neurologist, in the clinic, they are the same patient twice.

But one of them carries a single broken stretch of DNA — a gene called C9orf72 — and the other doesn't.

Here's the question for the next twenty minutes: if you drew a tube of blood from each of them, could the PROTEINS in that blood tell them apart?

We answer this at the population level — a within-cohort group pattern, not as a per-patient clinical test. Land the hook now; qualify at the end.
-->

---
layout: section
color: bio
---

# First — what is ALS?

<div class="text-xl mt-6 text-[#cbd5e1] max-w-3xl">
The nerves that carry "move" from your brain to your muscles <span class="text-[#7dd629] font-semibold">die</span>.
Slowly, then completely. There is no cure.
</div>

<div class="mt-8 flex justify-center">
  <svg viewBox="0 0 520 150" width="520">
    <!-- a single motor neuron, dimming -->
    <circle cx="70" cy="75" r="26" fill="none" stroke="#7dd629" stroke-width="2"/>
    <circle cx="70" cy="75" r="9" fill="#7dd629"/>
    <!-- axon -->
    <line x1="96" y1="75" x2="430" y2="75" stroke="#4a5568" stroke-width="3"/>
    <!-- dimming dendrites -->
    <line x1="70" y1="49" x2="55" y2="28" stroke="#4a5568" stroke-width="2"/>
    <line x1="70" y1="49" x2="88" y2="28" stroke="#4a5568" stroke-width="2"/>
    <!-- muscle, gone gray -->
    <rect x="430" y="55" width="44" height="40" rx="6" fill="#1a2540" stroke="#4a5568" stroke-width="1.5"/>
    <text x="70" y="128" text-anchor="middle" fill="#94a3b8" font-family="Inter, sans-serif" font-size="13px">motor neuron</text>
    <text x="452" y="128" text-anchor="middle" fill="#94a3b8" font-family="Inter, sans-serif" font-size="13px">muscle</text>
    <text x="250" y="60" text-anchor="middle" fill="#94a3b8" font-family="Inter, sans-serif" font-size="12px" font-style="italic">the signal fades</text>
  </svg>
</div>

<!--
Motor neurons are the wires that carry the command to move from your brain to your muscles. In ALS, those wires die. First you drop things. Then you can't walk, then can't speak, then can't breathe. Most people die within two to five years. No cure.

That's the stakes. Say it plainly and let it sit.
-->

---
layout: default
---

# Most ALS has no known cause — but some does

<div class="grid grid-cols-1 gap-6 mt-4">
  <div class="text-lg text-[#cbd5e1] max-w-4xl">
    For most patients we can't point to a culprit. But the single most common <em>genetic</em> cause is a gene called <span class="text-[#cbb994] font-semibold">C9orf72</span> — carriers have a short piece of DNA repeated over and over.
  </div>
  <MutationAnchor />
  <div class="text-base text-[#94a3b8] max-w-4xl">
    That repeat is what <span class="text-[#f1f5f9]">defines who is a "C9 carrier"</span> — a confirmed <span class="font-mono">G4C2 repeat ≥30</span> — and it splits patients into a group we can study. It is a known fact about the <em>person</em>, not a landmark on any map (C9orf72 itself is not measured).
  </div>
</div>

<!--
Among all ALS, a subset carries one identifiable culprit: a repeated chunk of DNA — GGGGCC, thirty-plus copies — in C9orf72.

We do NOT need the biology of why the repeat is harmful. We need only this: it cleanly labels a group of patients. "C9 carriers" vs everyone else. That label is the thing we'll compare against.

C9orf72 itself is not even measured in our blood data, and it is NOT the center of any map. It defines the GROUP. The whole rest of the talk compares groups — it never walks outward from C9orf72.
-->

---
layout: default
---

# What we actually measured

<div class="grid grid-cols-1 gap-4 mt-2">
  <div class="text-lg text-[#cbd5e1] max-w-4xl">
    A gene is a <span class="text-[#94a3b8]">recipe</span>. A <span class="text-[#4ecadf] font-semibold">protein</span> is the dish it builds — the working part that actually does things in a cell. We weighed the dishes.
  </div>
  <ProteinReadout />
  <div class="text-base text-[#94a3b8] max-w-4xl">
    From a routine blood draw — immune cells, <span class="text-[#f1f5f9]">not</span> motor neurons. One level for each measured feature, for each person.
  </div>
  <div class="text-xs text-[#94a3b8] max-w-5xl">
    <span class="text-[#cbd5e1]">AnswerALS</span> · <span class="font-mono">3,264</span> feature rows = <span class="font-mono">3,263</span> human UniProt + <span class="font-mono">1</span> iRT standard · <span class="font-mono">423</span> metadata-matched samples · <span class="font-mono">log₂(x+1) ~ group + Sex</span>
  </div>
</div>

<!--
Proteins are the working parts of a cell. The matrix has 3,264 feature rows: 3,263 human UniProt rows plus one iRT technical standard. We measured them in PBMCs — immune cells from an ordinary blood draw. Say it explicitly: this is blood, not motor neurons, and the iRT row is not a human protein.

One number per protein per person. That table of numbers is everything we have to work with.
-->

---
layout: default
---

# Three groups of people

<CohortBars />

<div class="text-base text-[#94a3b8] max-w-4xl mt-6">
Same measurement, three groups. Remember that <span class="text-[#4ecadf] font-semibold">25</span> — C9 is rare, and that small number comes back at the end.
</div>

<!--
The three groups, working sizes: C9 carriers 25, sporadic ALS 294, healthy controls 91.

Say out loud that 25 is small. C9 is a rare mutation. Foreshadow without dwelling — "hold on to that 25." It comes back at the end.
-->

---
layout: default
---

# How loud did each protein change?

<VolumeKnob />

<div class="text-base text-[#94a3b8] max-w-4xl mt-5">
One number per protein: <span class="text-[#4ecadf] font-semibold">loudness</span> = <span class="font-mono">|EB-moderated t|</span>. We never claim <em>why</em> it changed — only how loud.
</div>

<!--
For each protein we get one number — how loudly its level changed between groups. Loud = a big, confident change. A big change we're unsure about gets turned down — noise counts against you.

That's the whole idea. It's the data (cyan) we lay onto the map (purple). For a technical room, name it: loudness = |moderated t| = effect ÷ SE — magnitude, not direction (not *why* it moved, just how-many-SEs it moved).
-->

---
layout: default
---

# One number per protein: |t|

<div class="text-base text-[#94a3b8] max-w-4xl">
The volume knob, named. <span class="text-[#4ecadf] font-semibold">Loud</span> = a big change we're confident is real. Here is the exact number behind it.
</div>

<div class="grid grid-cols-5 gap-5 max-w-5xl mt-4">

<div class="col-span-3 p-4 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.05)]">
<div class="text-[#4ecadf] font-semibold text-sm mb-2">the statistic — a signal-to-noise ratio</div>
<div class="text-center text-[#e2e8f0] text-lg my-3" style="font-family:'JetBrains Mono',monospace;">
moderated&nbsp;<span class="text-[#4ecadf]">t</span>&nbsp;=&nbsp;<span class="text-[#cbd5e1]">contrast effect</span>&nbsp;÷&nbsp;<span class="text-[#cbd5e1]">its standard error</span>
</div>
<div class="text-sm text-[#cbd5e1] mt-2">
Fit one model per protein: <span style="font-family:'JetBrains Mono',monospace;color:#cbd5e1;">log₂(intensity+1) ~ <span style="color:#4ecadf;">group</span> + <span style="color:#94a3b8;">Sex</span></span> — <span class="text-[#4ecadf]">group</span> is the tested coefficient (the C9-vs-other shift); <span class="text-[#94a3b8]">Sex</span> is a nuisance column held aside. Divide the shift by how unsure we are of it.
</div>
<div class="text-sm text-[#cbd5e1] mt-3">
Then take the <strong>magnitude</strong>: <span style="font-family:'JetBrains Mono',monospace;color:#e2e8f0;">|t|</span> keeps SIZE, drops direction. That's deliberate — the later walk asks <em>how perturbed</em> a neighborhood is, not which way each protein leaned.
</div>
</div>

<div class="col-span-2 p-4 rounded-lg border border-[#bf6ff7] bg-[rgba(191,111,247,0.05)]">
<div class="text-[#bf6ff7] font-semibold text-sm mb-2">borrowing strength — empirical Bayes</div>
<div class="text-sm text-[#cbd5e1]">
The residual variance is shrunk toward a <span class="text-[#bf6ff7]">genome-wide prior</span> fitted across ~3,200 proteins (limma; Smyth 2004):
</div>
<div class="text-center text-[#e2e8f0] my-3" style="font-family:'JetBrains Mono',monospace;font-size:13px;">
s²<sub>post</sub> = blend( own&nbsp;s² , <span style="color:#bf6ff7;">prior</span> )
</div>
<div class="text-sm text-[#cbd5e1]">
<span class="text-[#7dd629] font-semibold">Kills:</span> a protein with a <em>freak-small</em> sample variance can't divide by near-zero and rocket to the top of the ranking — the prior pulls it back.
</div>
<div class="text-sm text-[#94a3b8] mt-2">
The fitted prior is <strong>well behaved</strong>: <span style="font-family:'JetBrains Mono',monospace;">d₀=4.98 · 5.51 · 4.92</span> across the three contrasts — about 5 prior df beside residual df in the hundreds.
</div>
</div>

</div>

<div class="text-sm text-[#94a3b8] max-w-5xl mt-3">
<span class="text-[#4ecadf] font-semibold">log₂(x+1) is the production scale</span>, not a sensitivity panel. The prior stabilizes unusually small variances without dominating the ranking. <span class="text-[#e2e8f0] font-semibold">This one number per protein is the only thing every later step reads.</span>
</div>

<!--
This is the foundational slide the deck was missing: |t| was only ever the volume-knob picture and never defined for a technical room. Lead with the 5-second on-ramp (loud = a big change you're confident is real), then the METHOD is the slide.

The statistic: moderated t = (the group shift) ÷ (its standard error) — a signal-to-noise ratio. The production model fit per protein is log2(intensity+1) ~ group + Sex: group is the coefficient we test, Sex is the nuisance column we adjust away (the Table 1 sex imbalance, 78% M in SPOR_LIMB, is why). A big shift we're unsure of gets a big SE and a small t — noise counts against you.

|t| takes the magnitude and drops direction. That is the bridge to the gradient: the downstream walk measures perturbation SIZE in a neighborhood, regardless of which way each protein moved.

EB half-beat: limma empirical-Bayes moderation (Smyth 2004) shrinks each protein's variance toward a fitted genome-wide prior, borrowing strength across ~3,200 proteins. Posterior variance = a blend of the protein's own sample variance and that prior. The failure mode it exists to kill: a protein with a freakishly small sample variance would otherwise divide by near-zero and top the ranking on noise. On the production log2(x+1) scale the fitted prior degrees of freedom are 4.98, 5.51, and 4.92 across the three contrasts — approximately 5 each. With residual degrees of freedom in the hundreds, this is a modest, well-behaved stabilizer rather than a prior-dominated fit.

Be explicit that log2(x+1) is the production analysis used by every downstream gradient, GSEA, and displayed pass count. The raw-linear run is historical sensitivity context, not the result being presented.

Handoff, said plainly: this single per-protein |t| is the only thing every later step reads — the gradient, the null, the GSEA all consume it.

Color: cyan = data/the statistic; purple = knowledge/the borrowed prior; green = the correct-silence the kill enforces. No amber/red here (no tripwire, no test-that-lied on this slide). Claim discipline held: magnitude, not direction; no mechanism/causal language.
-->

---
layout: two-cols
---

# A map nobody had to draw

<div class="pr-4">

Every time a paper reported **"protein A acts on protein B,"** someone — or some machine — drew an arrow.

Do that for **millions of papers** and you get a map: a kind of **Wikipedia of which protein acts on which.**

We didn't build it. The literature did. It's called **INDRA**.

<div class="mt-4 text-sm text-[#94a3b8]">
This is the <span class="text-[#bf6ff7] font-semibold">knowledge</span> layer — purple, all deck.
The protein measurements are the <span class="text-[#4ecadf] font-semibold">data</span> layer — cyan.
</div>

<div class="mt-3 text-sm text-[#94a3b8] max-w-4xl">
INDRA CoGEx · <span class="font-mono">~129,000</span> belief-noise-filtered edges of <span class="font-mono">4</span> directional kinds (activate / inhibit / increase / decrease) · median <span class="font-mono">~15</span> measured neighbors per protein.
</div>

</div>

::right::

<KnowledgeGraphIntro />

<!--
INDRA is a knowledge graph: arrows for "A acts on B," assembled from millions of papers. Directional. We didn't curate it; the field did.

Two layers from here on: purple = the knowledge map; cyan = our blood data. We're about to lay the cyan onto the purple.

("Regulates" is fine when naming this map — INDRA encodes regulatory statements. It is not a claim about our findings.)
-->

---
layout: section
color: gradient
---

# The one question

<div class="text-2xl mt-6 text-[#f1f5f9] max-w-3xl leading-snug">
On that map, does the disturbance pile up in <span class="text-[#4ecadf]">tight neighborhoods</span> —
and is that piling-up <span class="text-[#4ecadf]">special to C9 carriers?</span>
</div>


<div class="mt-10 font-mono text-sm text-[#94a3b8] max-w-3xl">
<span class="text-[#4ecadf]">H₀:</span> |t| independent of regulatory-graph position &nbsp;·&nbsp; <span class="text-[#4ecadf]">test:</span> contrast-specific spatial concentration of perturbation
</div>

<!--
No statistics yet.

We lay the blood data onto the literature map and ask one thing: does the disturbance concentrate in small neighborhoods of the map — and is that concentration something you see in C9 carriers but NOT in generic ALS?

We do NOT measure distance from C9orf72. Every protein is used in turn as an anchor, and each ring pools incoming and outgoing regulatory partners. C9 association comes from comparing groups, not from a privileged source node.

Everything after this is earning a trustworthy answer.
-->

---
layout: section
color: gyori
---

# Act 1 — The question that could not discriminate

<div class="text-xl mt-6 text-[#cbd5e1] max-w-3xl">
We started with a valid self-contained question: did a set move away from zero? It answered loudly — but that was not the competitive question we needed.
</div>

<!--
Act divider: ROAST validly asks whether a set differs from no change. Our scientific question was whether the curated set was more unusual than comparable sets in these observed data. The distinction is question mismatch, not a broken or miscalibrated test.
-->

---
layout: default
---

# The obvious first test

<div class="text-lg text-[#cbd5e1] max-w-4xl">
The obvious move: grab a group of proteins and ask the whole group at once — <span class="text-[#4ecadf]">"are these disturbed in C9 carriers?"</span> There's a published gene-set test for exactly that, called <strong>ROAST</strong>.
</div>

<div class="text-base text-[#94a3b8] max-w-4xl mt-3">
Concretely: a <strong>self-contained ROAST rotation</strong> of a <span class="text-[#cbb994] font-semibold">curated ~47-gene C9 target set</span>, scored by its <span class="font-mono">MSQ</span> (direction-agnostic) statistic on the moderated <span class="font-mono">t</span> — it asks whether <em>that set</em> moved versus <strong>no change at all</strong>, never versus the rest of the proteome.
</div>

<FirstTest />

<div class="text-base text-[#94a3b8] max-w-4xl">
On the real data, <span class="text-[#f1f5f9]">ROAST</span> said a loud <span class="text-[#7dd629] font-semibold">yes</span> — group after group. It looked like we'd found the fingerprint on day one.
</div>

<!--
ROAST is a published gene-set test (Wu et al. 2010): point it at a group of proteins, it returns "significant or not." On the real proteome it lit up — group after group said yes. Felt like a triumph. It's a reasonable, off-the-shelf first choice — NOT a strawman we rigged, but also not necessarily THE field standard; just a sensible default. That's what makes the next slide land: a sensible test, fooled. The gut-check comes next.
-->

---
layout: default
---

# Random observed-data sets often rejected too

<SmokeDetector />

<!--
One beat of silence before the number.

We drew 200 random gene lists from the observed proteome and ran the same self-contained test. It rejected for 113 of them: 56.5%.

That is <strong>not</strong> a false-positive-rate or null-calibration estimate: random observed-data sets can contain genuine effects. It is a specificity check for our competitive question. If random sets often reject too, a rejection does not show that the curated C9 set is unusual relative to its proteomic background.

That same distinction changes the day-one result below. The five-hop expansion showed movement relative to zero; it did not survive a competitive reference. The random observed sets are not null simulations, so they cannot diagnose ROAST calibration.
-->

---
layout: default
---

# Day one looked spectacular

<div class="text-lg text-[#cbd5e1] max-w-4xl">
Our first answer was a <strong>five-hop recursive expansion</strong>: hop-1 at <span class="font-mono" style="color:#f59e0b">p=0.019</span>, and <strong style="color:#f59e0b">46 of 46</strong> two-hop gene sets calling "significant" against no change. We celebrated it.
</div>

<div class="text-base text-[#94a3b8] max-w-4xl mt-5">
Then we read the estimand: every call was <strong>raw self-contained ROAST</strong>. Random observed-data sets also rejected <span class="font-mono">56.5%</span> of the time, so those calls showed movement relative to zero but <strong>not competitive specificity</strong> for the curated expansion.
</div>

<div class="mt-7 grid grid-cols-[1fr_auto_1fr] items-center gap-6 max-w-5xl">

<div class="rounded-lg border border-[#3a1518] bg-[#1a0c0e] px-5 py-4">
<div class="text-xs uppercase tracking-wider text-[#94a3b8] mb-2">raw self-contained ROAST</div>
<div class="font-mono text-sm" style="color:#ef4444">hop-1 &nbsp;p = 0.019</div>
<div class="font-mono text-sm" style="color:#ef4444">hop-2 &nbsp;46 / 46 significant</div>
<div class="font-mono text-sm" style="color:#ef4444">π₀ = 0.02</div>
<div class="text-xs text-[#94a3b8] mt-2 italic">movement relative to zero</div>
</div>

<div class="text-3xl text-[#64748b] font-mono">→</div>

<div class="rounded-lg border border-[#1e3a2a] bg-[#0c1612] px-5 py-4">
<div class="text-xs uppercase tracking-wider text-[#94a3b8] mb-2">competitive reference + Camera VIF</div>
<div class="font-mono text-sm" style="color:#7dd629">hop-1 &nbsp;p = 0.69</div>
<div class="font-mono text-sm" style="color:#7dd629">hop-2 &nbsp;0 / 46 significant</div>
<div class="font-mono text-sm" style="color:#7dd629">π₀ = 1.0</div>
<div class="text-xs text-[#94a3b8] mt-2 italic">silence, correctly</div>
</div>

</div>

<div class="text-base text-[#cbd5e1] max-w-5xl mt-5">
Against the competitive reference, hop-1 changes <span class="font-mono">0.019→0.69</span> and <span class="font-mono">46/46→0/46</span>; the best hop-2 p changes <span class="font-mono">0.004→0.25</span>. <strong>The day-one expansion showed movement, not specificity.</strong>
</div>

<!--
This is the integrity beat — the spine of Act 1, not a footnote. Say it plainly even to a general room.

THE ON-RAMP (5 seconds): day one looked spectacular. A five-hop recursive expansion — hop-1 p=0.019, all 46 of 46 two-hop sets significant against no change. We celebrated it.

THE METHOD (the slide): the whole expansion used raw self-contained ROAST. The 56.5% random observed-set rejection rate is not a false-positive-rate estimate: those sets can contain real effects. It shows that the self-contained rejection does not discriminate the curated set from the observed proteomic background.

THE COMPETITIVE CHECK: compare the same sets with OTHER sets and account for target co-regulation (ρ̄=0.185, VIF=9.5). Hop-1 0.019 → 0.69. The 46/46 becomes 0/46 under that different reference. The best hop-2 set changes from VPS4A p=0.004 to AFG3L2 p=0.25.

TONE: self-contained and competitive tests answer different questions. Random observed-data sets do not test null calibration.

HANDOFF: movement relative to zero was not selective for our target set. Next slide names the question mismatch, then we ask whether mean |t| differs between direct regulatory partners and two-hop partners.
-->

---
layout: default
---

# Why it could not answer the competitive question

<div class="text-lg text-[#cbd5e1] max-w-5xl">
ROAST scored each set against <span class="text-[#f59e0b] font-semibold">no change</span>. Random sets drawn from the observed proteome can contain real effects, so their 56.5% rejection rate is <strong>not a false-positive-rate estimate</strong>. It tells us the self-contained result has poor specificity for a different question: did the curated C9 set change <span class="text-[#4ecadf] font-semibold">more than comparable sets?</span>
</div>

<WrongYardstick />

<div class="text-sm text-[#94a3b8] max-w-5xl mt-1">
Self-contained asks “moved from zero?”; competitive asks “more than comparable sets?” The latter reads ordinary (<span class="font-mono">p=0.69</span> with VIF; matched <span class="font-mono">p=0.65</span>). Next: <span class="text-[#4ecadf] font-semibold">compare mean |t| in direct-partner and two-hop rings.</span>
</div>

<!--
This is the WHY in one breath. ROAST is self-contained: it asks whether each set differs from no change, not whether the curated set exceeds comparable sets. The random lists were sampled from observed data, not generated under a true null; they can carry genuine effects. Therefore 113/200 is an observed random-set rejection fraction, not a false-positive rate or calibration experiment. Its use here is narrower: many random sets also clear the self-contained bar, so that bar is poorly specific for the competitive C9-set question.

The right yardstick (right panel): don't test against no-change, compare to RANDOM groups drawn from the SAME sick patients — the crowd it actually lives in. Against THAT crowd, our candidate sits right in the middle. Ordinary, not special — which is exactly what the competitive test found: z=1.25, p=0.21, and p=0.69 after VIF correction for the targets' co-regulation. INTEGRITY NOTE: the right yardstick does NOT rescue the gene set — it confirms it's ordinary. Don't tell the room competitive testing found the signal.

The right panel asks the competitive question directly: compare the curated set with other sets drawn from the same observed proteomic background.

The closing line sets up the pivot: instead of another set-level presence test, compare two undirected rings around every anchor — direct regulatory partners versus two-hop partners. This is a spatial contrast, not a propagation mechanism.

ROAST is a "self-contained" test (group vs zero); we needed a "competitive" one (group vs other groups). Name both — self-contained vs competitive is the conceptual hinge of Act 1.
-->

---
layout: default
---

# 47 targets, but not 47 votes

<div class="text-lg text-[#cbd5e1] max-w-4xl">
The competitive test said <em>weak</em> (<span class="font-mono">z=1.25, p=0.21</span>) — barely above ordinary. But it counted all <span class="text-[#9b8cff] font-semibold">47 targets</span> as 47 independent witnesses. They aren't. Curated C9 targets are <span class="text-[#9b8cff] font-semibold">co-regulated</span> — <strong>they move together</strong>, so they testify together. Count correlated witnesses as independent and you fake your own confidence.
</div>

<div class="text-base text-[#cbd5e1] max-w-4xl mt-5">
The Camera correction prices that in. One number — average pairwise correlation among the targets, <span class="font-mono">ρ̄ = 0.185</span> — converts <span class="font-mono">k=47</span> entangled genes into how many independent votes they're actually worth, and reflates the error bar by the missing count:
</div>

<div class="max-w-4xl mt-5 font-mono text-[#e2e8f0]" style="line-height:2.05">
<div>VIF = 1 + (k−1)·ρ̄ = 1 + 46 × 0.185 = <span style="color:#f59e0b" class="font-semibold">9.5</span></div>
<div>SE inflates by <span style="color:#f59e0b" class="font-semibold">√9.5 ≈ 3.08</span><span class="text-[#94a3b8]">×</span></div>
<div>z: 1.25 ÷ 3.08 = <span style="color:#7dd629" class="font-semibold">0.41</span> &nbsp;→&nbsp; p: 0.21 → <span style="color:#7dd629" class="font-semibold">0.69</span> <span class="text-[#7dd629] font-semibold">(null)</span></div>
</div>

<div class="text-base text-[#94a3b8] max-w-4xl mt-6">
Not goalpost-moving: the Camera VIF addresses co-regulation for the competitive question. Under that reference the set is ordinary, which sends us to a different statistic: the contrast between two undirected neighborhood rings.
</div>

<!--
This is the mechanism behind the "weak → null" row on the next slide. The on-ramp is the jury: 47 co-regulated targets are not 47 independent votes — they move together, so they speak with fewer independent voices than their count suggests. The competitive z treated them as independent, which is why it read a too-confident "weak" p=0.21 instead of null.

The METHOD is the slide, not a footnote. VIF — variance inflation factor — is one number: ρ̄ = 0.185 is the average pairwise correlation among the 47 targets (mean off-diagonal Pearson). Plug into VIF = 1 + (k−1)·ρ̄ = 1 + 46×0.185 = 9.5. That says the 47 entangled genes carry the information of only ~47/9.5 ≈ 5 independent ones. The standard error was understated by a factor of √9.5 ≈ 3.08, so the honest z is 1.25/3.08 = 0.41, and p goes 0.21 → 0.69. Null.

The integrity beat — say it out loud: this is NOT moving the goalposts to kill our own result. A 3/3 brutalist consensus confirmed the VIF corrects real co-regulation, not real signal — the tell is that ρ̄ stays in the residuals after the model fit, i.e. the correlation is structural co-regulation, not the effect we're testing for. If the VIF were eating signal, the correlation would vanish in residuals; it doesn't.

Caveat for the methodologist in the room: ρ̄ is estimated from raw expression rather than residuals (enrichment_z.py:154-178), which errs CONSERVATIVE — it can only over-deflate z, never inflate it. So if anything the true z is ≥0.41; the null verdict is safe either way. (k=47 is the curated target-set size; the exact count varies slightly by run but the algebra and verdict do not.)

Handoff: self-contained ROAST detects movement relative to zero; competitive z and matched single-gene references find the curated set ordinary relative to alternatives. These are different estimands. We next compare mean |t| for direct regulatory partners with mean |t| for two-hop partners around every anchor. Color discipline: amber marks the competitive adjustment, green the competitive null, and purple co-regulation.
-->

---
layout: default
---

# Three references, one competitive verdict

<div class="text-base text-[#cbd5e1] max-w-5xl mb-5">
The self-contained result establishes movement from zero. The competitive references ask the question we need: is the curated set unusual relative to alternatives?
</div>

<table class="w-full max-w-5xl text-sm border-collapse">
<thead>
<tr class="text-[#94a3b8] border-b border-[#334155]">
<th class="text-left py-2 font-semibold">test</th>
<th class="text-left py-2 font-semibold">reference</th>
<th class="text-left py-2 font-semibold">result</th>
<th class="text-left py-2 font-semibold">verdict</th>
</tr>
</thead>
<tbody class="text-[#cbd5e1]">
<tr class="border-b border-[#1a2540]">
<td class="py-2.5">ROAST <span class="text-[#94a3b8]">(self-contained)</span></td>
<td>perturbed vs <em>no change</em>?</td>
<td class="font-mono">113/200 random observed sets reject</td>
<td class="font-semibold" style="color:#f59e0b">poor competitive specificity</td>
</tr>
<tr class="border-b border-[#1a2540]">
<td class="py-2.5">competitive z</td>
<td>more than random sets?</td>
<td class="font-mono">z=1.25, p=0.21</td>
<td class="font-semibold" style="color:#f59e0b">weak</td>
</tr>
<tr class="border-b border-[#1a2540]">
<td class="py-2.5">+ Camera VIF</td>
<td>correcting co-regulation (ρ̄=0.185, VIF=9.5)</td>
<td class="font-mono">z=0.41, p=0.69</td>
<td class="font-semibold" style="color:#7dd629">null</td>
</tr>
<tr>
<td class="py-2.5">matched single-gene</td>
<td>per-target reanalysis</td>
<td class="font-mono">p=0.65</td>
<td class="font-semibold" style="color:#7dd629">null</td>
</tr>
</tbody>
</table>

<div class="text-sm text-[#94a3b8] max-w-5xl mt-3">
The <span class="font-mono">113/200</span> fraction is not calibration; competitive references read the curated set as ordinary. Next: an undirected two-ring statistic.
</div>


<!--
The 56.5% value is an observed random-set rejection fraction, not a null-calibration result; those sets may contain real effects. Its high value only shows that a self-contained rejection has poor specificity for the competitive question. Competitive z is weak (p=0.21) and reads null (p=0.69) under Camera VIF; matched single-gene p=0.65 agrees. The five-hop day-one expansion therefore showed movement from zero but not competitive specificity.
-->

---
layout: section
color: gradient
---

# A better question:<br>is perturbation concentrated nearby?

<div class="mt-6 flex justify-center">
  <EarthquakeIntuition />
</div>


<div class="mt-8 text-center font-mono text-sm text-[#94a3b8] max-w-4xl mx-auto">
pivot: compare <span class="text-[#4ecadf]">two undirected rings</span> around each anchor &nbsp;·&nbsp; mean |t| among direct regulatory partners versus two-hop partners
</div>

<!--
For each anchor, ring 1 contains every measured direct regulatory partner: both proteins that regulate the anchor and proteins the anchor regulates. Ring 2 contains measured partners two undirected edges away.

The statistic asks whether mean moderated |t| is larger in ring 1 than ring 2. Because distance is symmetrized, the slope carries no source→target direction, temporal order, propagation, or mechanistic-cascade meaning.
-->

---
layout: default
---

# Walking the map, one hop at a time

<div class="grid grid-cols-[1.05fr_0.95fr] gap-6 items-start mt-2">
  <div>
    <HopShells />
    <div class="text-sm text-[#94a3b8] mt-1">
      One hop = a direct regulatory partner. Two hops = a partner two undirected edges away.
    </div>
  </div>

  <div class="space-y-3 mt-2">
  <div class="p-3 rounded-lg border border-[#bf6ff7] bg-[rgba(191,111,247,0.06)]">
    <div class="text-[#bf6ff7] font-semibold mb-1 text-sm">two <span style="color:#4ecadf">rings</span>, defined precisely</div>
    <div class="text-sm text-[#cbd5e1]"><strong>Ring 1</strong> contains measured direct regulatory partners. <strong>Ring 2</strong> contains measured proteins exactly two undirected edges away. Each ring contributes its <strong style="color:#4ecadf">arithmetic mean |t|</strong>.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#94a3b8] font-semibold mb-1 text-sm">distance is <span style="color:#f1f5f9">undirected</span></div>
    <div class="text-sm text-[#cbd5e1]">INDRA stores directed statements, but distance symmetrizes them. Ring 1 therefore includes both the anchor's <span class="text-[#bf6ff7]">regulators and regulatees</span>. The slope has no source→target or propagation direction.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#f8b84e] bg-[rgba(248,184,78,0.05)] text-sm text-[#94a3b8]">
    <strong class="text-[#cbd5e1]">Why stop at two — the declared bounded operating point.</strong> The audited depth evidence is the later fixed-panel sensitivity: <strong style="color:#4ecadf">8 / 6 / 0</strong> at <span style="font-family:'JetBrains Mono',monospace">h≤2</span> versus <strong style="color:#94a3b8">6 / 0 / 0</strong> unbounded. Legacy interpretive-ceiling claims remain <strong style="color:#f8b84e">withdrawn</strong>.
  </div>
  </div>
</div>

<!--
Ring 1 contains measured direct regulatory partners, regardless of which way the INDRA statement points: regulators plus regulatees. Ring 2 contains measured proteins two edges away after symmetrizing the graph.

The slope is ring2 mean |t| minus ring1 mean |t|. It is a topological concentration contrast, not an arrow of time, source→target propagation, or mechanistic cascade. We use h≤2 as the declared bounded operating point; the audited depth comparison is 8/6/0 bounded versus 6/0/0 unbounded. Legacy interpretive-ceiling analyses are not evidence.
-->

---
layout: default
---

# No shortcuts through ghosts

<div class="grid grid-cols-[1.05fr_0.95fr] gap-6 items-start mt-2">
<div>
  <MeasuredOnlyPaths />
  <div class="text-sm text-[#94a3b8] mt-1">
    <span class="text-[#4ecadf] font-semibold">3,263 human UniProt rows</span> + one iRT standard. Distance paths may include measured human proteins only.
  </div>
</div>

<div class="space-y-3 mt-2">
  <div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#94a3b8] font-semibold mb-1 text-sm">if ghosts <em>were</em> allowed</div>
    <div class="text-sm text-[#cbd5e1]">Unmeasured intermediates make almost everything two hops from everything. The ring saturates near <strong>~3,240</strong>; anchors become flat and non-discriminating.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.06)]">
    <div class="text-[#4ecadf] font-semibold mb-1 text-sm">measured-only ⇒ each ring is its <em>own</em> size</div>
    <div class="text-sm text-[#cbd5e1]">Restricting paths to measured proteins preserves anchor-specific 2-hop rings: <span style="font-family:'JetBrains Mono',monospace">median ≈2,000; range 0–~3,000</span>.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#475569] text-sm text-[#94a3b8]">
    A path through an unmeasured bystander would manufacture closeness with no measured node behind it. Measured-only forbids that shortcut.
  </div>
</div>
</div>

<!--
The map contains far more proteins than we measured. If our walk hops THROUGH a protein we never measured, that's a stepping-stone we can't see — we'd be trusting the map's wiring with no data to check it.

The rule: only ever step on proteins we actually measured. It's a rule we set, not a result we found.
-->

---
layout: default
---

# One diagnostic per anchor — not the pathway test

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><PermutationNull /></div>

<div>
<div class="text-base text-[#cbd5e1] mb-3">
For one anchor, deal each protein's loudness to a random protein of <strong>similar connectivity</strong>, recompute the slope 999 times, and locate the observed slope in that cloud.
</div>
<div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
<div class="text-[#94a3b8] font-semibold mb-1 text-sm">what it answers</div>
<div class="text-sm text-[#cbd5e1]">Is <em>this anchor's</em> slope unusual under a degree-matched reassignment of the fixed <span class="font-mono">|t|</span> vector?</div>
</div>
<div class="p-3 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.06)] mt-3">
<div class="text-[#4ecadf] font-semibold mb-1 text-sm">what it does not answer</div>
<div class="text-sm text-[#cbd5e1]">These p-values are <strong>diagnostics only</strong>. GSEA never consumes them, so they do not establish that pathway enrichment is free of topology effects.</div>
</div>
<div class="text-sm text-[#94a3b8] mt-3">
<span class="font-mono">999</span> degree-binned permutations per anchor · empirical <span class="font-mono">p = (#≤obs + 1)/1000</span> · separate from the downstream GSEA null.
</div>
</div>

</div>

<!--
This is an anchor-level diagnostic: keep the map fixed, reassign |t| within degree bins, and compare one observed slope with 999 shuffled slopes. It does not feed the pathway ranking. Downstream GSEA ranks the slope itself and generates its own gene-set null, so do not use this slide to claim the pathway result is topology-proof.
-->

---
layout: default
---

# One protein's slope

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><GradientSlopeAnatomy /></div>

<div>
<div class="p-3 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.05)]">
<div class="text-[#4ecadf] font-semibold text-sm mb-1">the slope is one transparent subtraction</div>
<div class="text-sm text-[#cbd5e1]">Two undirected rings, two mean-|t| numbers: <span style="font-family:'JetBrains Mono',monospace;color:#f1f5f9">ring1 ≈ 1.8</span> (direct partners), <span style="font-family:'JetBrains Mono',monospace;color:#f1f5f9">ring2 ≈ 1.05</span> (two-hop partners).</div>
<div class="my-2 font-mono text-[#e2e8f0]" style="line-height:1.7;font-size:13px">slope = ring2 − ring1 = 1.05 − 1.8 = <span style="color:#4ecadf" class="font-semibold">−0.75</span></div>
<div class="text-sm text-[#cbd5e1]">A <strong style="color:#4ecadf">negative</strong> slope means ring 1 has the larger mean |t|. It does <strong>not</strong> encode regulatory direction or propagation.</div>
</div>

<div class="p-3 rounded-lg border border-[#bf6ff7] bg-[rgba(191,111,247,0.05)] mt-3">
<div class="text-[#bf6ff7] font-semibold text-sm mb-1">why no weights appear</div>
<div class="text-sm text-[#cbd5e1]">The general fit is inverse-variance <strong>WLS</strong> of mean|t| on hop. But at depth 2 a line through two points is exact, so <span style="font-family:'JetBrains Mono',monospace;color:#e2e8f0">WLS ≡ ring2 − ring1</span> — the <strong>weights cancel</strong> (verified for every weight choice).</div>
</div>
</div>

</div>

<!--
For one anchor, compare mean |t| among direct regulatory partners with mean |t| among two-hop partners on the symmetrized graph. A negative slope is exactly ring2 − ring1 < 0.

State the boundary explicitly: regulators and regulatees are pooled in ring 1. The statistic says nothing about which protein acts first or whether a disturbance propagates.
-->

---
layout: default
---

# Now do it for every protein

<AllAnchorsField />

<div class="text-sm text-[#94a3b8] max-w-5xl mt-2">
<span class="text-[#f1f5f9] font-semibold">3,264 attempted</span> → <span class="text-[#f1f5f9] font-semibold">3,117 valid</span> → <span class="text-[#4ecadf] font-semibold">1,407 in the primary robust ranking</span> (<span class="font-mono">hop-1 ≥20</span>). Excluded before gradients: <span class="font-mono">137 + 10</span>.
</div>

<!--
Each protein is used in turn as an anchor for the same undirected two-ring contrast; there is no single center.

We attempt the same bounded, measured-only calculation for all 3,264 feature rows, including the iRT standard. Exactly 3,117 return a valid two-shell gradient; 137 have no reachable measured neighbor within h≤2, and 10 have fewer than the required 10 measurable neighborhood proteins. The primary pathway ranking then applies the declared robust-scope rule, hop-1 neighborhood size ≥20, leaving 1,407 anchors.

The real question: do the steep ones pile up in proteins that share a job?
-->

---
layout: default
---

# Adding up the signal by neighborhood

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><NeighborhoodTally /></div>

<div>
<div class="text-base text-[#94a3b8] mb-3">
Rank the <span class="text-[#4ecadf] font-semibold">1,407 robust anchors</span> that meet the primary <span class="font-mono">hop-1 ≥20</span> rule. Then ask where eight discovery-derived pathway terms land.
</div>
<div class="p-3 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.05)]">
<div class="text-[#4ecadf] font-semibold text-sm mb-1">one ranked list, read once</div>
<div class="text-sm text-[#cbd5e1]">The <span style="font-family:'JetBrains Mono',monospace;color:#f1f5f9">1,407</span> robust slopes become one preranked list, ranked by <span style="font-family:'JetBrains Mono',monospace;color:#4ecadf">score = −slope</span> (most-negative on top). A term scores high if its members cluster there → a <strong style="color:#4ecadf">positive NES</strong> = its neighborhoods concentrate the perturbation. <span class="text-[#94a3b8]">(gseapy.prerank; four libraries.)</span></div>
</div>
<div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)] mt-3">
<div class="text-[#94a3b8] font-semibold text-sm mb-1">where significance comes from</div>
<div class="text-sm text-[#cbd5e1]">Ranking held <strong>fixed</strong>; shuffle gene-set membership <span style="font-family:'JetBrains Mono',monospace">1000×</span>. This GSEA null is separate from the per-anchor degree shuffles. The eight terms were selected in discovery on this cohort, then frozen for an <strong>internal consistency check</strong> — not independent confirmation.</div>
</div>
</div>

</div>

<!--
We group proteins by known job (named gene sets) and total up the slope strength in each. Which jobs are most concentrated?

The method is the slide. Of 3,117 valid gradients, 1,407 meet the primary robust-scope rule of at least 20 measured hop-1 neighbors. Those 1,407 slopes become ONE preranked list. Score = −slope, so the most-negative slopes sit at the top. Standard preranked GSEA asks whether a term's members cluster there. Positive NES means concentration. Run across GO, Reactome, WikiPathways, and HPO.

Where the p-value comes from matters: the ranked list is FIXED, and we permute the GENE-SET labels — which proteins count as members of "this job" — 1000 times. We do NOT permute the patient phenotypes. FDR is gseapy's own NES-histogram FDR, computed per database — not Benjamini-Hochberg.

The sign convention is load-bearing: score = −slope here, so positive NES = concentration. The separate size-matched and pathway-level degree-matched auxiliaries shown later do not consume per-anchor degree p-values or GSEA NES values.

The eight terms were outcome-selected during discovery on this same cohort and then frozen before the measured-only/log2 consistency rerun. We apply an eightfold reporting threshold (raw p<0.00625 and NES>0), but there is no post-selection FWER or selective-inference guarantee and no independent confirmation.
-->

---
layout: default
---

# The fixed panel spans three themes

<ThreeClusterSignature />

<!--
The eight discovery-derived terms group into three themes: mRNA splicing, chromatin, and nucleocytoplasmic transport. Treat these as same-cohort hypotheses, not independently discovered neighborhoods.

Frame this as "the fixed panel groups into three themes." Do not call the themes independent evidence; they were discovery-derived in the same cohort. Never say "C9 drives splicing" or anything causal.
-->

---
layout: default
---

# Same pipeline, three comparisons

<ContrastTriangle />

<!--
Same map, same loudness, same 8 frozen jobs. We only change WHO we compare: C9 vs sporadic, C9 vs control, and sporadic vs control.

That last one — sporadic vs control — is the within-cohort specificity check. A C9-associated pattern should be stronger in both C9 contrasts and weak in Sporadic-vs-Control. That pattern is informative, but it does not establish mutation causality or external replication.
-->

---
layout: default
---

# What the three contrasts do — and do not — tell us

<div class="text-base text-[#cbd5e1] max-w-5xl mb-5">
Three contrasts, one pipeline. Their <strong>pattern</strong> supports a C9-associated reading <em>within this cohort</em>; it does not by itself identify cause or remove graph topology.
</div>

<table class="w-full max-w-5xl text-sm border-collapse">
<thead>
<tr class="text-[#94a3b8] border-b border-[#334155]">
<th class="text-center py-2 font-semibold">C9 vs SPOR</th>
<th class="text-center py-2 font-semibold">C9 vs CTRL</th>
<th class="text-center py-2 font-semibold">SPOR vs CTRL</th>
<th class="text-left py-2 font-semibold pl-6">interpretation</th>
</tr>
</thead>
<tbody>
<tr class="border border-[#4ecadf] bg-[rgba(78,202,223,0.08)]">
<td class="text-center py-2.5 font-bold" style="color:#4ecadf">✓</td>
<td class="text-center font-bold" style="color:#4ecadf">✓</td>
<td class="text-center text-[#94a3b8]">—</td>
<td class="text-[#f1f5f9] font-semibold pl-6">C9-associated in this cohort &nbsp;←&nbsp; observed</td>
</tr>
<tr class="border-b border-[#1a2540] text-[#94a3b8]">
<td class="text-center py-2.5">—</td><td class="text-center">✓</td><td class="text-center">✓</td>
<td class="pl-6">shared ALS-vs-control pattern</td>
</tr>
<tr class="border-b border-[#1a2540] text-[#94a3b8]">
<td class="text-center py-2.5">✓</td><td class="text-center">✓</td><td class="text-center">✓</td>
<td class="pl-6">non-specific across all contrasts</td>
</tr>
<tr class="text-[#94a3b8]">
<td class="text-center py-2.5">—</td><td class="text-center">—</td><td class="text-center">✓</td>
<td class="pl-6">sporadic-ALS-specific</td>
</tr>
</tbody>
</table>

<div class="text-base text-[#94a3b8] max-w-5xl mt-5">
The quiet <span class="text-[#f1f5f9]">SPOR-vs-CTRL</span> leg argues against a generic sporadic-ALS pattern here. <strong class="text-[#cbd5e1]">It does not rule out pathway-topology effects, discovery selection, or cohort-specific structure.</strong>
</div>


<!--
The three-contrast pattern is a within-cohort specificity check. C9-vs-SPOR ✓, C9-vs-CTRL ✓, SPOR-vs-CTRL — is consistent with a C9-associated signal in this cohort. Do not call it mutation causality, a topology falsifier, or replication.-->

---
layout: default
---

# The fixed-panel consistency check

<BonferroniMatrix />

<div class="text-sm text-[#94a3b8] max-w-5xl mt-1">
<strong class="text-[#cbd5e1]">Discovery-derived, same-cohort panel.</strong> An eightfold reporting threshold is applied (<span class="font-mono">raw p&lt;0.00625 ∧ NES&gt;0</span>); it provides <strong>no post-selection FWER or selective-inference guarantee</strong>. Internal consistency — not independent confirmation.
</div>

<!--
Stage it slowly. Land 8/8 and 6/8 first.

Of the 8 frozen jobs: all 8 pass in C9-vs-sporadic, 6 of 8 in C9-vs-control.

Then pause. "And the within-cohort specificity leg..." — reveal Sporadic-vs-Control: zero of eight. This supports a C9-associated reading here; it does not prove mutation causality.

The empty column is gray, not red. It is the within-cohort specificity pattern we hoped to see, but say "consistent with" rather than "proof of."

Integrity point: these are not preregistered independent hypotheses. They were outcome-selected during discovery on this cohort and frozen only for the later log2/measured-only consistency rerun. The raw-p<0.00625 rule is an eightfold reporting threshold, not a post-selection FWER or selective-inference correction. Present 8/6/0 as an internal consistency result and a hypothesis for external replication.
-->

---
layout: default
---

# Two auxiliary receipts — same data, different references

<AuxiliaryReceipts />

<!--
These two auxiliaries are current and canonical.

The graph-independent size-matched analysis uses the production log2 empirical-Bayes t statistics, one HGNC gene per unit, and 10,000 uniform same-size sets. Its eightfold-threshold pattern is 8/8/0.

The graph-conditional degree-matched analysis uses the same 1,407-anchor robust scope as primary GSEA, matches fixed-term members to nonmembers on full-INDRA degree, and compares term mean -slope across 9,999 references. Its pattern is 7/7/0; Vpr-mediated nuclear import is the sole C9 non-pass.

Keep three nulls distinct: per-anchor degree shuffles diagnose one slope; GSEA permutes gene-set membership; the 7/7/0 auxiliary matches pathway members to degree-comparable nonmembers and tests mean -slope. Neither auxiliary is independent because the terms and cohort are reused. The eightfold threshold has no post-selection FWER guarantee.
-->

---
layout: default
---

# A broader view — descriptive only

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><TriangulationDrop /></div>

<div>
<div class="text-base text-[#94a3b8] mb-3">
At <span class="font-mono">FDR&lt;0.05</span>, the bounded log₂ run contains <span class="text-[#f1f5f9]">284</span>, <span class="text-[#f1f5f9]">260</span>, and <span class="text-[#94a3b8] font-semibold">0</span> significant <strong>database-term rows</strong> across the three contrasts.
</div>
<div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
<div class="text-[#cbd5e1] font-semibold text-sm mb-1">what those bars are — and are not</div>
<div class="text-sm text-[#cbd5e1]">They sum rows from <strong>GO · Reactome · WikiPathways · HPO</strong>. The same or overlapping biology can appear more than once across libraries, so these are <strong>not unique pathways</strong>. They are same-cohort descriptive context, not a second validation layer.</div>
</div>
</div>

</div>

<!--
Easiest place to lose people — go slow.

These counts are FDR<0.05 rows summed across four databases, not deduplicated biological pathways. They describe the breadth of the same bounded log2 analysis. Do not present them as independent corroboration or as 284 and 260 unique mechanisms. The audited size-matched HGNC auxiliary is now a separate canonical 8/8/0 receipt on the preceding slide.
-->

---
layout: default
---

# The six-term intersection

<BonferroniMatrix :highlight-core="true" />

<div class="text-base text-[#94a3b8] max-w-4xl mt-3">
Six of eight pass <span class="text-[#4ecadf] font-semibold">both</span> C9 comparisons and stay quiet in Sporadic-vs-Control — the same-cohort intersection spanning splicing, chromatin, and transport.
</div>

<!--
Six terms pass BOTH C9 comparisons and stay silent in sporadic-vs-control: three splicing terms, chromosome, chromatin, and nucleocytoplasmic transport. This is an intersection inside a discovery-derived fixed panel, not a replicated core. Stay at count-and-pattern; do not invoke mechanism, a master regulator, or independent validation.
-->

---
layout: default
---

# What remains outside the evidence stack

<div class="text-xs tracking-widest text-[#f59e0b] mb-4">WITHDRAWN / LEGACY — NOT CURRENT SUPPORT</div>

<div class="grid grid-cols-2 gap-5 max-w-5xl">
  <div class="p-4 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#cbd5e1] font-semibold mb-2">Removed as supporting evidence</div>
    <ul class="text-sm text-[#94a3b8] space-y-1">
      <li>STRING alternative-network result</li>
      <li>matched RNA / post-transcriptional reading</li>
      <li>age-robustness proxy</li>
      <li>abundance-stratified null</li>
    </ul>
  </div>
  <div class="p-4 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#cbd5e1] font-semibold mb-2">Also withdrawn from the battery</div>
    <ul class="text-sm text-[#94a3b8] space-y-1">
      <li>legacy interpretive-ceiling / F5b claim</li>
      <li>five sensitivity reruns</li>
      <li>pathway-topology claims from anchor permutations</li>
    </ul>
  </div>
</div>

<div class="text-base text-[#cbd5e1] max-w-5xl mt-5">
These analyses are <strong>not backup evidence</strong>. The audited size-matched <span class="font-mono">8/8/0</span> and robust-scope degree-matched <span class="font-mono">7/7/0</span> auxiliaries are current and were shown separately.
</div>

<!--
Read this as a clean evidence withdrawal, not a caveat. Do not confuse the withdrawn historical artifacts with the two newly audited auxiliaries: the log2 HGNC size-matched result is 8/8/0, and the canonical robust-scope degree-matched mean(-slope) result is 7/7/0 with Vpr the sole C9 non-pass.
-->

---
layout: default
---

# It's a local property

<LocalVsGlobal />

<div class="text-sm text-[#94a3b8] max-w-5xl mt-2">
Bounded <span class="font-mono">h≤2</span> gives <span class="font-mono text-[#4ecadf]">8/6/0</span>; unbounded depth gives <span class="font-mono">6/0/0</span>. This supports a bounded-depth statistical pattern—not a physical cascade.
</div>

<!--
Frame it as an observed two-ring contrast in this cohort, not a propagation cascade or universal biological boundary.

When depth is unbounded, the fixed-panel pass pattern changes from 8/6/0 to 6/0/0. This is attenuation of the statistical pattern, not evidence of a physical signal propagating or dissipating.

Do not speculate that the deep walk "drags in noise" as an established mechanism. The licensed statement is only the observed attenuation: bounded 8/6/0 versus unbounded 6/0/0.
-->

---
layout: default
---

# The evidence boundary after audit

<div class="grid grid-cols-2 gap-6 max-w-5xl mt-2">
  <div class="p-4 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.06)]">
    <div class="text-[#4ecadf] font-semibold mb-3">Licensed now</div>
    <ul class="text-sm text-[#cbd5e1] space-y-2">
      <li>production <span class="font-mono">log₂(x+1)</span>, measured-only, bounded <span class="font-mono">h≤2</span></li>
      <li><span class="font-mono">1,407</span> robust anchors ranked from <span class="font-mono">3,117</span> valid gradients</li>
      <li>discovery-derived fixed panel: <span class="font-mono">8/6/0</span></li>
      <li>same-size HGNC auxiliary: <span class="font-mono">8/8/0</span></li>
      <li>robust degree-matched mean(−slope): <span class="font-mono">7/7/0</span></li>
      <li>unbounded depth sensitivity: <span class="font-mono">6/0/0</span></li>
      <li>cautious three-contrast pattern, within this cohort</li>
    </ul>
  </div>
  <div class="p-4 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#94a3b8] font-semibold mb-3">Not licensed</div>
    <ul class="text-sm text-[#94a3b8] space-y-2">
      <li>independent confirmation or external replication</li>
      <li>mutation causality, mechanism, or diagnosis</li>
      <li>pathway topology ruled out by anchor diagnostics</li>
      <li>support from withdrawn STRING, RNA, age, abundance, legacy F5b, or five sensitivity reruns</li>
    </ul>
  </div>
</div>

<div class="text-base text-[#cbd5e1] max-w-5xl mt-5">
The result is a <strong>same-cohort, hypothesis-generating consistency pattern</strong>. External replication is the next evidentiary gate.
</div>

<!--
This is the complete current battery. The 8/8/0 and 7/7/0 auxiliaries reuse the same cohort and outcome-selected terms, so they are consistency receipts, not independent confirmation. Per-anchor degree-shuffle p-values remain separate: they do not enter GSEA or the pathway-level degree-matched mean(-slope) auxiliary and do not rule out topology.
-->

---
layout: section
color: cogex
---

# Act 3 — A finer question

<div class="text-xl mt-6 text-[#cbd5e1] max-w-3xl">
The fixed panel points to <span class="text-[#bf6ff7]">where</span> the within-cohort pattern concentrates — three themes.
An exploratory extension asked a finer question: do paired proteins <span class="text-[#4ecadf]">rise and fall together</span> the same way across groups?
</div>

<!--
So far we measured a same-cohort pathway pattern. The finer WASC question is separate and exploratory; its preregistered stability gate later fails, so it produces no edge-level conclusion.

We name this test WASC a few slides on. Don't say "gradient" (the audience never met that word), "rewiring," or "coordination." Say: do two proteins still rise and fall together the same way across groups.
-->

---
layout: default
---

# So far: one protein at a time

<div class="text-lg text-[#cbd5e1] max-w-5xl">
So far we measured how <strong>loud</strong> each protein's change was — one number per protein. But loudness alone can't say whether two proteins still <strong>rise and fall together</strong> across patients. That needs a different chart.
</div>

<AxisContrast />

<!--
Don't say "gradient" or "coordination."

The left chart is what we just did: watched ONE protein — how loud its change was vs how far it sits on the map. That's a level. The new question needs BOTH proteins' levels in the SAME person (right chart): protein A across, protein B up, one dot per patient. Point at the axes — that swap is the whole difference. Measuring one protein at a time simply can't see whether two proteins move together; we need this second chart.
-->

---
layout: default
---

# What the link looks like

<div class="text-lg text-[#cbd5e1] max-w-5xl">
Take two proteins the map links. In every patient we have <em>both</em> their levels — so plot one against the other. When A is high, is B high too? If they rise and fall together, the dots <strong>tilt</strong> — and that tilt is the link.
</div>

<CochranQTriplet stage="one" />

<div class="text-sm text-[#94a3b8] max-w-5xl mt-3">
<strong class="text-[#cbd5e1]">The spec, for one within-cluster edge.</strong> Regress protein-B's level on the <span class="text-[#4ecadf] font-semibold">anchor</span> (protein-A) <em>within a single group</em>; the fitted slope <span style="font-family:'JetBrains Mono',monospace;color:#4ecadf">β̂(B | A)</span> is the link. The anchor is first <strong>FWL-residualized</strong> against <span style="font-family:'JetBrains Mono',monospace">Sex · Age<sub>z</sub> · Tissue</span> (Batch pre-residualized) so the tilt isn't a covariate masquerading as coupling. The slope is in <span style="font-family:'JetBrains Mono',monospace;color:#f1f5f9">log₂-abundance units</span> — <strong>the tilt is the link.</strong>
</div>

<!--
Two linked proteins, A and B. Each patient is one dot: their A level (across) and B level (up). If B reliably rises when A rises, the dots tilt upward — that tilt IS the link. A steep, tight tilt = they track each other closely; a flat, loose cloud = they barely track.

Stay slow here — this picture is the whole basis of the test.
-->

---
layout: default
---

# Measure that tilt in each group

<div class="text-lg text-[#cbd5e1] max-w-5xl">
Fit the tilt three times — once in C9, once in Sporadic, once in Control. Two things can happen.
</div>

<CouplingOutcomes />

<!--
Same two proteins, but split the people into the three groups and fit a tilt in each. Three tilts. Either they sit on top of each other — the link is the same across groups — or they fan apart — the link differs.

That's the finer question, for a single pair. We never say why; only whether the tilt is the same or different across groups.
-->

---
layout: default
---

# So now we can name it: WASC

<div class="flex flex-col gap-4 mt-10 max-w-5xl">
  <div class="flex items-baseline gap-5">
    <div class="text-[#cbd5e1] text-lg flex-1">both proteins in the same neighborhood — <span class="text-[#bf6ff7] font-semibold">944</span> linked pairs in all</div>
    <div class="text-[#bf6ff7] font-bold text-2xl w-64 shrink-0 text-right">Within-neighborhood</div>
  </div>
  <div class="flex items-baseline gap-5">
    <div class="text-[#cbd5e1] text-lg flex-1">the tilt of one protein plotted against its partner</div>
    <div class="text-[#4ecadf] font-bold text-2xl w-64 shrink-0 text-right">Anchor-Slope</div>
  </div>
  <div class="flex items-baseline gap-5">
    <div class="text-[#cbd5e1] text-lg flex-1">do the three groups <em>agree</em> on that tilt?</div>
    <div class="text-[#7dd629] font-bold text-2xl w-64 shrink-0 text-right">Concordance</div>
  </div>
</div>

<div class="text-lg text-[#f1f5f9] max-w-5xl mt-8">
<strong>WASC:</strong> within one neighborhood, do two proteins' tilts agree across C9, Sporadic, and Control?
</div>

<div class="text-sm text-[#94a3b8] max-w-5xl mt-4">
Those <span class="text-[#bf6ff7] font-semibold">944</span> are every within-cluster hop-1 link, split by neighborhood: <span class="text-[#bf6ff7] font-semibold">Splicing 434</span> · <span class="text-[#bf6ff7] font-semibold">Chromatin 443</span> · <span class="text-[#bf6ff7] font-semibold">Transport 67</span> (<span class="font-mono">904</span> unique edges, 40 cross-theme duplicates) — all living inside the <em>same three neighborhoods</em> Act 2 lit up.
</div>

<!--
Within-neighborhood = both proteins in the same neighborhood (944 linked pairs); Anchor-Slope = the tilt of one protein vs its partner; Concordance = do the groups agree. So WASC = within one neighborhood, do two proteins' tilts agree across the three groups? A link-coherence test — never mechanism, never rewiring.

If a specialist asks: in the literature these neighborhoods are called "clusters," so the formal name is "Within-cluster Anchor-Slope Concordance" — same thing. We use "neighborhood" so nobody has to learn a synonym mid-talk.
-->

---
layout: default
---

# The referee: a single number

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><CochranQTriplet stage="referee" /></div>

<div>
<div class="text-base text-[#cbd5e1] mb-3">
To call it, we need one number. <strong>Cochran-Q</strong> scores how much the three tilts disagree, giving bigger groups more say. A small score = the tilts <em>concord</em>.
</div>
<div class="p-3 rounded-lg border border-[#7dd629] bg-[rgba(125,214,41,0.05)]">
<div class="text-[#7dd629] font-semibold text-sm mb-1">the number — and which way is "agree"</div>
<div class="my-1 font-mono text-[#e2e8f0]" style="font-size:13px;line-height:1.8">Q = Σ wₖ (βₖ − β̄)² , &nbsp; wₖ = 1 / SEₖ²</div>
<div class="text-sm text-[#cbd5e1]">Each tilt βₖ weighted by its own precision (<strong>inverse-variance</strong>). <strong style="color:#7dd629">LOW Q = tilts agree = invariant = WASC-positive</strong> — a <strong>lower-tail</strong> test (small Q is the <em>win</em>, the reverse of a usual heterogeneity test).</div>
</div>
<div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)] mt-3">
<div class="text-[#94a3b8] font-semibold text-sm mb-1">so within-pathway correlation can't fake "agree"</div>
<div class="text-sm text-[#cbd5e1]">The null shuffles each anchor's partner within <strong>degree-decile × pooled-|Pearson|-decile bins</strong> — matched connectivity <em>and</em> co-expression. FDR is <strong>BY</strong> (Benjamini-<em>Yekutieli</em>) at <span style="font-family:'JetBrains Mono',monospace">q=0.10</span>, not BH, because edges share anchors, targets, and null draws.</div>
</div>
</div>

</div>

<!--
The referee just turns "do they agree?" into a number.

Cochran-Q is a standard meta-analysis statistic. All it does here: measure how far apart the three tilts are, weighting each group by how sure we are — more people = more weight, so n matters (remember the 25). Small Q = the tilts agree. Don't show the formula.

944 linked pairs, each judged this way. The claim is count-and-pattern, never mechanism.
-->

---
layout: default
---

# We set a tripwire — before looking

<div class="grid grid-cols-2 gap-6 mt-1 items-center">

<div><TripwireBars mode="setup" /></div>

<div>
<div class="text-base text-[#94a3b8] mb-3">
Pre-registered honesty check — fixed in writing <em>before</em> any verdict. <strong>Four prongs</strong>; all must clear, or the run aborts.
</div>
<div class="space-y-2 text-sm">
<div class="pl-3 border-l-2 border-[#475569]"><span class="text-[#cbd5e1] font-semibold">(a) null calibration</span> — FP <span style="font-family:'JetBrains Mono',monospace">0.111</span> vs <span style="font-family:'JetBrains Mono',monospace">0.120</span> · <span style="color:#7dd629">PASS</span></div>
<div class="pl-3 border-l-2" style="border-color:#4ecadf"><span style="color:#4ecadf" class="font-semibold">(b) SPOR-25 down-sample — THE gate</span> — shrink SPOR 294→25, require <span style="font-family:'JetBrains Mono',monospace">Jaccard ≥ 0.70</span></div>
<div class="pl-3 border-l-2 border-[#475569]"><span class="text-[#cbd5e1] font-semibold">(c) substrate-robustness</span> — <span style="font-family:'JetBrains Mono',monospace">N/A</span> after the all-protein-pool fix</div>
<div class="pl-3 border-l-2 border-[#475569]"><span class="text-[#cbd5e1] font-semibold">(d) FW-vs-OLS exactness</span> — <span style="font-family:'JetBrains Mono',monospace">9.4e-10</span> vs <span style="font-family:'JetBrains Mono',monospace">1e-8</span> · <span style="color:#7dd629">PASS</span></div>
</div>
<div class="text-sm text-[#94a3b8] mt-3">
Prong <span style="color:#4ecadf;font-weight:600">(b)</span> is load-bearing: does this raw-p selector/WASC pipeline return the same pairs when SPOR is reduced from 294 to <span class="text-[#4ecadf] font-semibold">25</span>? The preregistered <span style="font-family:'JetBrains Mono',monospace">0.70</span> bar is a pipeline-stability gate, not a general power theorem for n=25.
</div>
</div>

</div>

<!--
Set it up as foresight, not hindsight.

Before running the production selector, we wrote down a stability check: shrink sporadic from 294 to 25 and ask whether this exact raw-p selector/WASC pipeline returns the same pairs.

We required at least 70% overlap and fixed that bar in advance. Passing or failing this gate speaks to this pipeline under this down-sampling experiment, not every possible pairwise method at n=25.
-->

---
layout: default
---

# The tripwire fired

<div class="grid grid-cols-[1.05fr_0.95fr] gap-6 items-start mt-2">
<div>
  <TripwireBars mode="result" />
  <div class="text-sm text-[#cbd5e1] mt-1">
    Mean Jaccard <span class="font-mono text-[#f59e0b]">0.285</span> missed the preregistered <span class="font-mono text-[#7dd629]">0.70</span> gate. Production halted; WASC produced no per-pair verdicts.
  </div>
</div>

<div class="space-y-3 mt-2">
  <div class="p-3 rounded-lg border border-[#f59e0b] bg-[rgba(245,158,11,0.05)]">
    <div class="text-sm font-semibold text-[#f59e0b]">what failed</div>
    <div class="text-sm text-[#cbd5e1] mt-1">This raw-p selector/WASC pipeline was unstable under <span class="font-mono">SPOR 294→25</span>. The low overlap repeats across three seeds and all three themes.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-sm font-semibold text-[#cbd5e1]">what it does not establish</div>
    <div class="text-sm text-[#94a3b8] mt-1">This calibration is not a general theorem that every edge model is underpowered at <span class="font-mono">n=25</span>, and it yields no carrier-count requirement.</div>
  </div>
  <div class="p-3 rounded-lg border border-[#475569]">
    <div class="text-sm font-semibold text-[#cbd5e1]">next evidentiary move</div>
    <div class="text-sm text-[#94a3b8] mt-1">Any different model or larger-cohort design needs a fresh prospective calibration and preregistration.</div>
  </div>
</div>
</div>

<!--
Not a downer, not a refutation — the gate did its job.

When we shrank sporadic to 25, mean Jaccard was 0.285 — below the preregistered 0.70 gate. This establishes instability of this raw-p selector/WASC pipeline under the SPOR 294→25 calibration. It does not show that every pairwise estimand or model is structurally underpowered at n=25.

So we halted the full run, by our own pre-registered rule. Say this plainly and without apology: WASC never produced a single per-pair result. We report the tripwire firing — not an invariance finding, not a difference finding. We didn't torture the data until it confessed.

The cards are possible next moves, not conclusions: report the failed gate; develop a different model under a fresh preregistration; or design a larger prospective study with its own calibration. None is on record as chosen, and no carrier-count requirement follows from this check.
-->

---
layout: default
---

# Two questions, two evidence states

<div class="grid grid-cols-[1.15fr_0.85fr] gap-6 mt-4 items-center">

<div class="pr-4 text-base text-[#cbd5e1]">

n = 25 is the available C9 cohort. It supports neither overstatement nor an edge-level shortcut.

<div class="mt-4">
<span class="text-[#4ecadf] font-semibold">Pathway-level result:</span> the same-cohort fixed-panel pattern is <span class="font-mono">8/6/0</span>. It remains hypothesis-generating until external replication.
</div>

<div class="mt-2 text-sm text-[#94a3b8]">Canonical same-data auxiliaries: same-size HGNC <span class="font-mono">8/8/0</span>; robust degree-matched mean(−slope) <span class="font-mono">7/7/0</span>.</div>

<div class="mt-3">
<span class="text-[#94a3b8] font-semibold">Edge-level extension:</span> the preregistered stability gate failed, so WASC produced no per-pair verdict.
</div>

<div class="mt-4 text-sm text-[#94a3b8]">
Do not compare these states as "25 is enough" versus "25 is not enough." They are different estimands with different evidence, and neither licenses individual-patient inference.
</div>

</div>

<div class="flex flex-col gap-4 justify-center h-full">
  <div class="p-5 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.07)]">
    <div class="text-[#4ecadf] font-semibold text-lg">Pathway-level pattern</div>
    <div class="text-sm text-[#94a3b8] mt-1">internal consistency · same cohort</div>
    <div class="text-[#4ecadf] text-xl mt-2">hypothesis-generating</div>
  </div>
  <div class="p-5 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.06)]">
    <div class="text-[#94a3b8] font-semibold text-lg">Edge-level WASC</div>
    <div class="text-sm text-[#94a3b8] mt-1">stability gate failed</div>
    <div class="text-[#94a3b8] text-2xl mt-2">withheld</div>
  </div>
</div>

</div>

<!--
Keep the evidence states separate. The pathway panel and its two canonical auxiliaries are same-cohort consistency results awaiting replication. WASC is withheld because its preregistered stability gate failed. Neither the 8/8/0 size-matched nor 7/7/0 degree-matched auxiliary establishes that 25 carriers are generally sufficient.
-->

---
layout: default
---

# What we can and cannot say

<div class="grid grid-cols-2 gap-6 max-w-5xl mt-4">

<div class="p-4 rounded-lg border border-[#4ecadf] bg-[rgba(78,202,223,0.06)]">
<div class="text-[#4ecadf] font-semibold mb-2">We can say</div>
<ul class="text-sm text-[#cbd5e1] space-y-1">
<li>the fixed panel shows <strong>8/6/0</strong> in this cohort</li>
<li>the primary ranking uses <strong>1,407 robust anchors</strong></li>
<li>same-size HGNC sets give <strong>8/8/0</strong></li>
<li>robust degree matching gives <strong>7/7/0</strong>; Vpr alone misses both C9 thresholds</li>
<li>bounded <strong>h≤2</strong> gives 8/6/0; unbounded gives <strong>6/0/0</strong></li>
</ul>
</div>

<div class="p-4 rounded-lg border border-[#475569] bg-[rgba(100,116,139,0.05)]">
<div class="text-[#94a3b8] font-semibold mb-2">We cannot say</div>
<ul class="text-sm text-[#94a3b8] space-y-1">
<li>independent confirmation or replication</li>
<li>mutation causality, mechanism, or diagnosis</li>
<li>source→target propagation from an undirected slope</li>
<li>that pathway topology has been ruled out</li>
<li>an individual-protein or edge-level discovery</li>
<li>support from withdrawn STRING, RNA, age, abundance, legacy F5b, or five sensitivity analyses</li>
</ul>
</div>

</div>

<!--
What we can report is the exact within-cohort battery: primary GSEA 8/6/0, graph-independent same-size HGNC 8/8/0, robust-scope degree-matched mean(-slope) 7/7/0 with Vpr the sole C9 non-pass, and unbounded attenuation to 6/0/0. The auxiliaries reuse the same selected terms and cohort; neither is independent confirmation.
-->

---
layout: default
---

# What it means — and what's next

<div class="grid grid-cols-1 gap-4 mt-4 max-w-4xl">
  <div class="text-lg text-[#cbd5e1]">
    A <span class="text-[#4ecadf] font-semibold">within-cohort, C9-associated protein-neighborhood pattern</span> across discovery-derived splicing, chromatin, and transport themes.
  </div>
  <div class="grid grid-cols-3 gap-4 mt-2">
    <div class="p-4 rounded-lg border border-[#4ecadf]/40 bg-[rgba(78,202,223,0.05)]">
      <div class="text-[#4ecadf] font-semibold text-sm">Current result</div>
      <div class="text-xs text-[#94a3b8] mt-1">GSEA 8/6/0 · size 8/8/0 · degree 7/7/0</div>
    </div>
    <div class="p-4 rounded-lg border border-[#bf6ff7]/40 bg-[rgba(191,111,247,0.05)]">
      <div class="text-[#bf6ff7] font-semibold text-sm">Current scope</div>
      <div class="text-xs text-[#94a3b8] mt-1">same-cohort consistency, not confirmation</div>
    </div>
    <div class="p-4 rounded-lg border border-[#7dd629]/40 bg-[rgba(125,214,41,0.05)]">
      <div class="text-[#7dd629] font-semibold text-sm">Next gate</div>
      <div class="text-xs text-[#94a3b8] mt-1">external replication on a fixed protocol</div>
    </div>
  </div>
  <div class="text-base text-[#94a3b8] mt-2">
    Next, in order: <strong class="text-[#cbd5e1]">freeze the production protocol and replicate externally</strong>; keep the auxiliaries explicitly same-data; rerun or reconcile remaining withdrawn analyses before restoring them; then revisit WASC under a separately justified design.
  </div>
</div>

<!--
What it means: this cohort contains a reproducible-within-pipeline pattern worth testing elsewhere. The two canonical auxiliaries probe pathway size and simple degree location but remain same-data checks on outcome-selected terms. This is not a validated biomarker, directional mechanism, or causal biology result.
-->

---
layout: end
---

# Does the blood know what the doctor can't?

<div class="text-xl mt-6 text-[#cbd5e1] max-w-3xl">
Within this cohort, C9 carriers show a <span class="text-[#4ecadf]">local protein-neighborhood pattern</span> across three discovery-derived themes.
</div>

<div class="text-base mt-4 text-[#94a3b8] max-w-3xl">
A hypothesis for external replication — <span class="text-[#f1f5f9]">not confirmation, causality, or a test for any one patient.</span>
</div>

<div class="mt-8 text-sm text-[#94a3b8]">
Eric Jing Mockler · Gyori Lab, Northeastern University<br>
Happy to go deeper in questions — the fixed panel, the protein map, or the evidence boundary.
</div>

<!--
Back to the opening question with the audited answer: within this cohort, the blood proteome contains a local C9-associated group pattern. That is a hypothesis, not an independent confirmation or individual diagnostic.

Thank you. In questions, keep four references distinct: per-anchor degree shuffles diagnose individual slopes; GSEA permutes gene-set membership; same-size HGNC sets give 8/8/0; pathway-level robust degree matching of mean(-slope) gives 7/7/0. None is external replication, and the undirected slope is not a propagation direction.
-->
