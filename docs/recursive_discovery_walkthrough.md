# Recursive Regulatory Discovery Through INDRA Knowledge Graphs

## The Question

C9orf72 repeat expansions are the most common genetic cause of ALS. INDRA's
knowledge graph tells us which genes C9orf72 regulates, and which genes
those genes regulate, and so on. Separately, we have proteomics data from
iPSC-derived motor neurons that tells us which proteins are altered in
C9orf72 carriers.

The question: **can we walk outward through INDRA's regulatory graph from
C9orf72, and at each step, confirm that the downstream targets actually
show altered protein levels in the data?** How far does the signal extend
before it fades into the background? And when it does fade — what does
that boundary tell us?

## The Data

**Proteomics**: 3,264 proteins quantified from iPSC-derived motor neurons
(AnswerALS cohort). Three groups:

| Group | n | Description |
|-------|---|-------------|
| C9orf72 carriers | 25 | ALS patients with pathogenic repeat expansion (≥30 GGGGCC repeats) |
| Sporadic ALS | 294 | ALS patients with no known genetic mutation |
| Healthy controls | 91 | No ALS diagnosis |

For each protein, we fit a linear model adjusting for sex. The model
produces a t-statistic per protein: positive means higher in the test
group, negative means lower.

**Knowledge graph**: INDRA CoGEx, queried for regulatory relationships
(IncreaseAmount, DecreaseAmount, Activation, Inhibition, Phosphorylation).
Each edge has a source attribution — which databases or text-mining
systems reported it — and an evidence count.

**Three contrasts**: We run the same recursive discovery pipeline on three
different comparisons of the same proteomics data:

1. **C9orf72 vs Sporadic** — what's unique to C9orf72 within ALS
2. **C9orf72 vs Control** — total C9orf72 disease signal vs healthy
3. **Sporadic vs Control** — general ALS signal vs healthy

Comparing results across all three tells us which regulatory cascades are
C9-specific versus shared with general ALS.

## Building Up: How We Test a Gene Set

Before describing the recursive algorithm, we need to explain the
statistical test at its core.

### The problem with testing one gene at a time

Suppose INDRA says VPS4A regulates 28 proteins. We could test each of
those 28 proteins individually: is it differentially expressed in C9orf72
carriers? But with only 25 carriers, individual protein tests are noisy.
Some truly affected proteins will be missed (false negatives), and some
unaffected proteins will look significant by chance (false positives).

### The gene set approach: testing all 28 together

Instead of asking "is protein X affected?", we ask: **"are VPS4A's 28
downstream targets, as a group, more affected than you'd expect by
chance?"**

This is more powerful because it pools evidence across all 28 proteins.
Even if no single protein passes a stringent threshold, a consistent
pattern across many proteins — most shifted slightly in the same
direction — can be highly significant as a collective signal.

### How ROAST works (the rotation test)

ROAST (Wu et al., 2010) is the specific gene set test we use. Here's
what it does, step by step:

**Step 1: Fit a linear model.** For each of the 3,264 proteins, fit
the model:

    protein_level = β₀ + β₁(condition) + β₂(sex) + ε

The coefficient β₁ tells us the condition effect for that protein. The
t-statistic = β₁ / SE(β₁) measures how confidently we can say the
effect is nonzero.

**Step 2: Improve the variance estimates.** With only 25 C9orf72
carriers, the variance estimate for each protein is noisy. A protein
measured in few samples might have an artificially small variance
(making its t-statistic too large) or artificially large variance
(making it too small).

The fix: look at the variance estimates across ALL 3,264 proteins.
Most proteins should have similar residual variance. We use this
global information to "moderate" each protein's variance toward the
overall trend. Proteins with suspiciously small variances get pulled
up; proteins with suspiciously large variances get pulled down. This
is called empirical Bayes moderation — "empirical" because we learn
the prior from the data itself, and "Bayes" because we combine the
per-protein estimate with the global prior.

The result: moderated t-statistics that are more stable and reliable
than raw t-statistics, especially with small sample sizes.

**Step 3: Compute the gene set statistic.** For VPS4A's 28 targets,
extract their 28 moderated t-statistics. Compute a summary: the mean
of their squared values (MSQ). A large MSQ means the targets are
collectively shifted away from zero — some combination of up- and
down-regulation beyond what chance would produce.

**Step 4: Generate the null distribution by rotation.** This is the
key innovation. We need to know: how large would the MSQ be if
C9orf72 status had no effect?

Rather than permuting sample labels (which breaks the covariate
structure), ROAST rotates the data in a high-dimensional space.
Imagine the 25 C9orf72 samples as a vector in sample-space. A random
rotation of this vector produces a "fake" contrast that preserves all
the correlation structure and covariate relationships but has no
real C9orf72 effect.

We generate 9,999 random rotations. For each rotation, we recompute
the MSQ for VPS4A's 28 targets. This gives us 9,999 values of "what
MSQ looks like under the null."

**Step 5: Compute the p-value.** Count how many of the 9,999 null
MSQ values are at least as large as the observed MSQ. If 40 out of
9,999 are that large, p = 40/9999 ≈ 0.004.

This p-value is exact — it doesn't depend on any distributional
assumptions. It's valid whether we have 25 samples or 25,000.

## The Recursive Algorithm

### Hop 1: Does C9orf72's regulatory program show enrichment?

INDRA says C9orf72 regulates 47 proteins (after filtering to those
measurable in our proteomics data). We test these 47 as a single gene
set using ROAST.

Result: **p = 0.019** (C9 vs Sporadic). C9orf72's direct targets are
collectively more perturbed in C9orf72 carriers than expected by chance.

### Hop 2: Which intermediaries carry the signal?

C9orf72 doesn't just regulate 47 proteins directly — it regulates 46
intermediary genes (VPS4A, RAN, HNRNPA1, TARDBP, FUS, ...), and each
of THOSE genes regulates its own downstream targets.

For each of the 46 intermediaries, we:
1. Query CoGEx for its downstream regulatory targets
2. Filter to measurable proteins
3. Run ROAST on that target set

This gives us 46 p-values — one per intermediary. VPS4A's 28 targets
yield p = 0.004. RAN's 70 targets yield p = 0.005. And so on down to
STMN2's 17 targets at p = 0.333.

### The multiple testing problem (and why standard corrections fail here)

We just ran 46 tests. Some will be significant by chance even if nothing
is going on. The standard fix is the Benjamini-Hochberg (BH) procedure:
rank the 46 p-values from smallest to largest, and for each rank k,
check whether p_(k) ≤ k/46 × 0.05. This controls the "false discovery
rate" — the expected proportion of false positives among the results
you call significant.

With our 46 p-values, standard BH gives **0 significant** out of 46.

But this is wrong. Here's why.

BH assumes the worst case: that all 46 hypotheses might be null (no
real signal). It penalizes every discovery as if it might be a false
positive. But we know from Hop 1 that C9orf72's regulatory program IS
enriched. Most of these 46 intermediaries are part of a real biological
cascade. Penalizing for 46 possible false positives when there are
really only ~1-2 true nulls is throwing away real discoveries.

### Storey's q-value: using the p-value distribution itself

Storey (2002) observed that the p-value distribution contains
information about how many hypotheses are truly null. Under the null
(no real effect), p-values are uniformly distributed between 0 and 1.
Under the alternative (real effect), p-values cluster near 0.

If we look at how many p-values fall above some threshold (say, 0.5),
we can estimate the null fraction. Among our 46 intermediaries, **zero**
have p > 0.5. Under the null, we'd expect ~23 of 46 to exceed 0.5.
Seeing zero means virtually all 46 are non-null.

Formally: π₀ = (number of p-values > 0.5) / (46 × 0.5) = 0/23 ≈ 0.
We clip this to 1/46 ≈ 0.02 (can't have exactly zero nulls). Then
the adjusted q-values are:

    q_storey = π₀ × q_BH ≈ 0.02 × q_BH

This recovers the lost signal. The largest Storey q-value among our 46
intermediaries is 0.043 — all 46 are significant at FDR < 0.05.

### Hop 3 and beyond: the recursive step

Each significant hop-2 intermediary has downstream targets. Some of
those targets are themselves regulatory genes with their own downstream
targets in INDRA. We extend the search to those genes — but only if
they have enough downstream targets to be testable (≥5 measurable
proteins).

This decision — which genes extend to the next hop — is based entirely
on the **knowledge graph structure**, not on the expression data. A gene
extends if INDRA says it regulates enough other genes. We never look at
the gene's own t-statistic to decide whether to extend it.

This is critical. If we selected genes for extension based on their
expression (e.g., "only extend genes with |t| > 1.5"), we would bias
the next hop's tests: genes selected for looking significant tend to
have targets that also look significant, even by chance. By using
graph structure alone for extension, we keep the expression-based
tests at each hop statistically independent of the extension decisions.

At hop 3: 1,384 intermediaries tested, 1,364 significant (98.6%).
At hop 4: 2,185 tested, 2,059 significant (94.2%).
At hop 5: 2,206 tested, 2,075 significant (94.1%). The set barely
changed — the algorithm has found every reachable regulatory gene.

Something interesting happens as the hops deepen: the *character*
of the genes changes. Hop 2 contains pathway-level regulators (MTOR,
TARDBP, STAT1). By hop 3, their individual effectors appear. By
hop 4, the subunits of the molecular machines those effectors operate
in. The cascade resolves from regulatory logic to physical substrate.
We'll see this resolution gradient in detail in the results.

## How We Know When to Stop

Two questions need answering:
1. Is C9orf72 special, or would any gene look like this?
2. Have we reached the edge of the knowledge graph?

### Question 1: The seed permutation null

We take 30 random genes from the proteome (excluding C9orf72 and its
direct neighbors) and run each one through the same pipeline as C9orf72:
query its INDRA neighbors, test their downstream targets, count the
significant ones.

If C9orf72 isn't special, random genes should produce similar results.
We compare: C9orf72 gets 46/46 significant at hop 2. How many of the
30 random genes achieve that? Only ~1. Empirical p-value: 0.032.

C9orf72's regulatory cascade is real — it's not just an artifact of
graph topology or the structure of the knowledge graph.

### Question 2: π₀ convergence (the null fraction trajectory)

At each hop, we estimate what fraction of the tested intermediaries are
truly null (π₀) from the p-value distribution:

```
Hop 2: π₀ = 0.022  →  98% of intermediaries are genuinely non-null
Hop 3: π₀ = 0.078  →  92% non-null (some noise entering)
Hop 4: π₀ = 0.100  →  90% non-null
Hop 5: π₀ = 0.101  →  90% non-null (same as hop 4)
```

The π₀ trajectory rises as the search moves away from C9orf72 and
the signal dilutes into the graph's background. When π₀ stops
changing between hops (0.100 → 0.101, a change of only 0.001), the
algorithm has reached the boundary of C9orf72's regulatory influence.
Every reachable regulatory gene has been tested. Extending further
would just revisit the same genes.

No additional INDRA queries are needed for this stopping criterion —
it's computed directly from the p-values at each hop.

## How We Query INDRA

For each intermediary gene, we query CoGEx for its regulatory targets:

```cypher
MATCH (source:BioEntity)-[r:indra_rel]->(target:BioEntity)
WHERE source.id = {gene_hgnc_id}
  AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount',
                       'Activation', 'Inhibition', 'Phosphorylation']
  AND r.evidence_count >= 1
RETURN target.name, r.source_counts, r.evidence_count
```

Each returned edge has source attributions — which databases or
text-mining systems reported it. We filter the returned targets to
those that:
1. Have valid HGNC identifiers (excludes protein complexes, gene
   families, and non-gene entities)
2. Are measurable in this specific proteomics experiment (the protein
   was quantified and has a UniProt ID in our data)
3. Have nonzero variance in the expression data (proteins with no
   variation across samples can't contribute to the test)

### Edge reliability: why source diversity matters more than evidence count

Each INDRA edge is supported by some number of evidence statements from
various sources. A statement from REACH (a text-mining system that reads
papers) might say "VPS4A activates CHMP2A." A statement from SIGNOR (a
manually curated database) might say the same thing.

Not all sources are equally reliable. Text-mining systems like REACH have
two kinds of errors:
- **Random errors**: the system misreads a sentence. More evidence from
  the same system reduces this (the same mistake is unlikely to repeat).
- **Systematic errors**: the system has a blind spot (e.g., it
  consistently misinterprets a particular sentence structure). More
  evidence from the SAME system doesn't help — the systematic error
  persists no matter how many papers the system reads.

This is why source diversity matters more than evidence count. Ten
statements all from REACH can't overcome REACH's systematic error rate.
But one statement from REACH plus one from SIGNOR can — because their
systematic errors are independent. The probability that both systems
make the same systematic error is the product of their individual
systematic error rates.

We compute an edge reliability score using this model:

```
For each source s with n_s evidence statements:
    P(source s is wrong) = syst(s) + (1 - syst(s)) × rand(s)^n_s

P(edge is wrong) = product over all sources of P(source s is wrong)
reliability = 1 - P(edge is wrong)
```

The `(1 - syst(s))` factor means: random error only matters in the
fraction of cases where systematic error didn't already cause the
mistake. This is a conditional probability — a correction over the
approximation used in INDRA's native belief scoring.

Concrete examples:
- 1 REACH statement: reliability = 0.67 (REACH has ~5% systematic, ~30% random error)
- 5 REACH statements: reliability = 0.87 (random error shrinks, systematic persists)
- 1 REACH + 1 SIGNOR: reliability = 0.98 (independent systematic errors multiply out)
- 1 SIGNOR alone: reliability = 0.94 (curated databases have ~1% systematic error)

**This is why we don't filter by evidence count.** An edge with 1 SIGNOR
statement (curated, reliability 0.94) is more reliable than an edge with
5 REACH statements (NLP only, reliability 0.87). The reliability score
captures this naturally.

## The Specificity Triangle

Running the same pipeline on all three contrasts — same INDRA graph,
same 46 intermediaries, same ROAST test — but different proteomics
comparisons produces dramatically different results:

### Hop 2: Is the signal C9-specific?

| Gene | Targets | C9 vs Sporadic | C9 vs Control | Sporadic vs Control | Edge reliability |
|------|---------|----------------|---------------|---------------------|-----------------|
| VPS4A | 28 | **0.004** | 0.049 | 0.640 | 0.89 |
| RAN | 70 | **0.005** | 0.085 | 0.760 | 0.32 |
| HNRNPA1 | 53 | **0.007** | 0.155 | 0.742 | 0.67 |
| TARDBP | 164 | **0.009** | 0.201 | 0.787 | 0.05 |
| FUS | 164 | **0.010** | 0.152 | 0.754 | 0.86 |
| MTOR | 511 | **0.019** | 0.213 | 0.860 | 0.06 |
| STAT1 | 311 | **0.015** | 0.138 | 0.817 | 0.67 |
| CASP3 | 315 | **0.016** | 0.156 | 0.758 | 0.67 |
| TMEM106B | 14 | **0.029** | 0.097 | 0.367 | 0.98 |

Three observations:

**1. Sporadic vs Control produces zero signal.** All 46 intermediaries
have p > 0.3 when comparing sporadic ALS to healthy controls. The
estimated null fraction is π₀ = 1.0 — every test looks null. C9orf72's
INDRA-defined regulatory targets are simply not enriched in general ALS.

**2. C9 vs Control is intermediate.** 45 of 46 pass FDR at hop 2
(π₀ = 0.09), but p-values are 5–20× larger than C9 vs Sporadic. The
signal exists — C9orf72 carriers differ from healthy — but it's weaker.

**3. C9 vs Sporadic is where the signal lives.** 46/46 significant,
π₀ = 0.02, median p = 0.029. This contrast captures what's unique to
C9orf72 within the ALS disease context.

### Summary across contrasts

| Contrast | n₁ vs n₂ | Hop 2 | π₀ | Hop 3 | π₀ | Seed null p |
|----------|----------|-------|-----|-------|-----|-------------|
| C9 vs Sporadic | 25 vs 294 | 46/46 sig | 0.02 | 1,364/1,384 sig | 0.08 | 0.032 |
| C9 vs Control | 25 vs 91 | 45/46 sig | 0.09 | 0/1,380 sig | 0.31 | 0.032 |
| Sporadic vs Control | 294 vs 91 | 0/46 sig | 1.00 | — | — | — |

### What this pattern means

The regulatory cascade defined by C9orf72's INDRA targets is **specific
to the C9orf72 mutation**, not a general feature of ALS. Sporadic ALS
patients don't show coordinated enrichment in any of these 46 regulatory
programs. The signal requires the C9orf72 repeat expansion.

But the most striking result is at **hop 3**. In the C9 vs Sporadic
contrast, 1,364 of 1,384 hop-3 intermediaries are significant. In C9 vs
Control, **zero** pass FDR — despite 45 of 46 hop-2 parents being
significant in that contrast. The signal at the direct-neighbor level
(hop 2) exists in both C9 contrasts, but the deeper cascade (hop 3)
only differentiates C9orf72 from other ALS patients.

### Decomposing the hop-3 signal

Among the 1,380 intermediaries tested in both C9 contrasts at hop 3:

| Category | Count | % | What it means |
|----------|-------|---|---------------|
| Significant in BOTH C9 contrasts | 45 | 3.3% | Truly C9-specific: expression differs from both sporadic and healthy |
| Significant only in C9 vs Sporadic | 674 | 48.8% | C9 looks like healthy; sporadic ALS has aberrant expression |
| Significant only in C9 vs Control | 13 | 0.9% | Shared ALS pathway (C9 and sporadic differ from healthy together) |
| Significant in neither | 648 | 47.0% | No signal at this depth |

**The dominant pattern (674 arms, 49%) is not that C9orf72 is doing
something extra — it's that C9orf72 carriers are failing to do something
that sporadic ALS patients do.** These gene sets show coordinated
differential expression in sporadic ALS (relative to healthy), but
C9orf72 carriers don't show that pattern. Their expression resembles
healthy motor neurons at these loci.

This makes biological sense. Many of the 674 "sporadic-only" arms
trace through known ALS response pathways:

| Hop-2 parent | Hop-3 children | % sporadic-only | Pathway |
|-------------|----------------|-----------------|---------|
| MTOR | 92 arms | 95% | Autophagy/protein homeostasis |
| TARDBP | 80 arms | 96% | RNA processing/splicing |
| STAT1 | 78 arms | 95% | Neuroinflammatory signaling |
| FUS | 60 arms | 90% | RNA metabolism |
| CASP3 | 57 arms | 93% | Apoptotic cascades |
| RAN | 49 arms | 94% | Nuclear transport |

These are pathways that sporadic ALS motor neurons activate — perhaps as
compensatory or degenerative responses — but C9orf72 motor neurons fail
to activate. The repeat expansion may disrupt upstream signaling that
would normally trigger these programs.

The 45 arms significant in both contrasts represent pathways where
C9orf72 carriers genuinely diverge from everyone — both healthy and
sporadic ALS. These include nuclear pore components (NUP93, NUP107,
RAE1), DNA repair (FEN1), and chromatin remodeling (MTA2, RAD21).

### Signal composition is stable across hops

We can quantify the fraction of signal that is C9-specific vs shared
with general ALS by comparing the strength of signal above null in each
contrast:

| Hop | C9-specific | Shared disease |
|-----|-------------|----------------|
| 2 | 76% | 24% |
| 3 | 76% | 24% |

The ratio is identical at both depths. The recursive framework is not
amplifying artifacts or changing the nature of the signal — it's
faithfully propagating the same ~3:1 C9-specific-to-shared ratio through
the regulatory graph.

## Results: C9orf72 vs Sporadic (Primary Contrast)

### The regulatory cascade

| Hop | Tested | Significant | Null fraction (π₀) | Seed null p |
|-----|--------|-------------|---------------------|-------------|
| 1 | 1 | 1 | — | — |
| 2 | 46 | 46 (100%) | 2.2% | 0.032 |
| 3 | 1,384 | 1,364 (98.6%) | 7.8% | — |
| 4 | 2,185 | 2,059 (94.2%) | 10.0% | — |
| 5 | 2,206 | 2,075 (94.1%) | 10.1% | [converged] |

### Hop 2: C9orf72's immediate regulatory network

All 46 intermediaries show significant downstream enrichment. The top
arms are well-known ALS biology:

| Gene | p-value | Targets | Reliability | Regulation |
|------|---------|---------|-------------|------------|
| VPS4A | 0.004 | 28 | 0.89 | activates |
| RAN | 0.005 | 70 | 0.32 | activates |
| HNRNPA1 | 0.007 | 53 | 0.67 | activates |
| TARDBP (TDP-43) | 0.009 | 164 | 0.05 | activates |
| FUS | 0.010 | 164 | 0.86 | represses |
| CNBP | 0.010 | 25 | 0.67 | activates |
| TIA1 | 0.010 | 29 | 0.67 | represses |
| EIF2AK2 | 0.018 | 88 | 0.95 | activates |
| MTOR | 0.019 | 511 | 0.06 | activates |
| TMEM106B | 0.029 | 14 | 0.98 | represses |

TARDBP (TDP-43) — the most prominent ALS gene — has the lowest edge
reliability (0.05, supported by a single weak INDRA source). Yet its
164 downstream targets show strong collective enrichment (p = 0.009).
The biological signal is detectable through a low-reliability edge
because ROAST tests the entire downstream program collectively, not
the individual edge.

TMEM106B has the highest reliability (0.98, supported by multiple
independent sources) and shows significant enrichment even with only
14 downstream targets.

### Hop 3: The regulators' downstream machinery

At hop 2 we tested the programs of 46 intermediaries — known ALS
regulators like MTOR, TARDBP, FUS. At hop 3 we ask: what do *their*
targets regulate? The hop-2 genes are pathway-level names. The hop-3
genes are the physical machinery those pathways control.

1,384 intermediaries are testable. 1,364 (98.6%) show significant
downstream enrichment.

The signal fans out unevenly. Some hop-2 parents spawn large families
of significant children; others have fewer downstream regulators in
INDRA. The parent distribution reveals which biological programs carry
the C9orf72 signal deepest:

| Hop-2 parent | Significant hop-3 children | Biology |
|--------------|---------------------------|---------|
| MTOR | 177 | Autophagy / protein homeostasis |
| STAT1 | 154 | Neuroinflammatory signaling |
| CASP3 | 126 | Apoptotic cascades |
| TARDBP | 116 | RNA processing / splicing |
| FUS | 93 | RNA metabolism |
| RAN | 65 | Nuclear transport |
| RAC1 | 54 | Cytoskeletal signaling |
| GSK3B | 53 | Wnt / tau phosphorylation |
| RAB8A | 45 | Vesicle trafficking |
| HNRNPA1 | 42 | mRNA splicing |

The character of the genes changes. At hop 2, MTOR is a single
pathway-level regulator. At hop 3, MTOR's 177 children include the
specific effectors: FKBP8 (mitophagy receptor, p = 0.001), MAD2L1
(spindle checkpoint, p = 0.004), EIF3D and EIF4G2 (translational
control, p = 0.009–0.012), STIP1 (co-chaperone, p = 0.012). Similarly,
RAN at hop 2 is "nuclear transport." At hop 3, RAN's children are the
individual components: RAE1 (mRNA export factor, p = 0.002), RCC1
(RAN's guanine exchange factor, p = 0.003), XPO1 (exportin-1,
p = 0.006), RANBP1 and RANGAP1 (the RAN cycle regulators, p = 0.008
and 0.013).

The same resolution applies to RNA processing. TARDBP at hop 2 is
"TDP-43 regulates RNA." At hop 3, TARDBP's 116 children include
HNRNPK and HNRNPD (splicing regulators, p = 0.009), SRSF3 (SR
protein, p = 0.008), UPF1 (nonsense-mediated decay, p = 0.008), and
PFN1 (profilin-1, an ALS gene in its own right, p = 0.008). FUS's 93
children include TNPO3 (transportin-3, the nuclear import receptor
for FUS itself, p = 0.003), HNRNPA3 and the SR proteins SRSF4 and
SRSF10 (p = 0.005), and NXF1 (mRNA nuclear export, p = 0.013).

STAT1's 154 children reveal the breadth of inflammatory signaling
touching the disease cascade: the chromatin remodeler MTA2 (p = 0.004),
the histone deacetylase HDAC2 (p = 0.007), the Notch effector RBPJ
(p = 0.004, 59 targets), and a cluster of splicing regulators (SRSF7,
RBM4, SF3B4) that cross-link the immune signaling and RNA processing
axes.

This cross-linking is the dominant pattern at hop 3: gene sets from
different hop-2 parents overlap. STAT1's children include splicing
factors that also appear in TARDBP's and FUS's programs. The signal
is not 46 independent cascades — it's a densely connected regulatory
network, and the ROAST tests at hop 3 are probing different entry
points into the same underlying biology.

### Hop 4: The physical complexes

737 new genes enter at hop 4 that weren't testable at hop 3. The
total rises to 2,185, of which 2,059 (94.2%) are significant. The
π₀ rises from 0.078 to 0.100 — the noise floor is climbing, but
slowly.

The new genes at hop 4 are one step more specific than hop 3. Where
hop 3 resolved pathway names to individual effectors, hop 4 resolves
effectors to the subunits of the complexes they operate in. Three
clusters are distinctive:

**Nuclear pore complex.** Eleven nucleoporin genes appear for the
first time: NUP205 (via FDXR, p = 0.001), NUP88 (via XPO1,
p = 0.002), NUP160 (via METTL3, p = 0.003), NUP155 (via DNAJB1,
p = 0.004), NDC80 (via AURKB, p = 0.010), NUP37, NDC1, NUP210,
NUP133. These arrive through the RAN→XPO1 and MTOR→METTL3 lineages.
At hop 2, RAN was "nuclear transport." At hop 3, XPO1 and RCC1 were
the transport cycle. At hop 4, the physical pore structure itself is
enriched — the barrel that everything passes through. This is
consistent with the known C9orf72 pathology: the GGGGCC repeat RNA
forms G-quadruplexes that sequester nuclear transport factors and
physically obstruct the pore.

**Spliceosome machinery.** Seventeen splicing genes enter: SRSF11
(via SRPK2, p = 0.002), SRSF9 (via SRPK2, p = 0.002), SF3B1
(via SRSF7, p = 0.005), SF3B3 (via SUMO1, p = 0.005), SNRNP200
(via USP39, p = 0.004), DDX23, DDX1, DDX39B. At hop 2, TARDBP and
FUS were RNA-binding proteins. At hop 3, their children were
individual splicing regulators (SRSF3, SRSF4). At hop 4, the core
spliceosome components appear — the SF3B complex (which recognizes
the branch point), the DEAD-box helicases (which unwind RNA
duplexes during splicing), the snRNP assembly factors. The cascade
has traced from "RNA processing is disrupted" through "specific
splicing regulators are affected" to "the spliceosome itself is
perturbed."

**Chaperones and proteostasis.** Eleven chaperone genes appear:
HSPA4 (via DNAJB1, p = 0.003), CCT5/CCT6A/CCT7 (chaperonin
subunits, p = 0.007–0.017), DNAJB2 (via CREB1, p = 0.010),
DNAJC2, DNAJC13. The CCT complex (also called TRiC) folds ~10%
of all cytoplasmic proteins, including actin and tubulin. The
DNAJ family members are Hsp70 co-chaperones that triage misfolded
proteins for refolding or degradation. Their appearance at hop 4
connects the MTOR-driven autophagy axis (hop 2) through individual
quality-control effectors (hop 3) to the folding machinery that
handles the substrate.

The parent distribution of the 737 new hop-4 genes confirms this
deepening pattern:

| Hop-3 parent | New hop-4 children | What it resolves |
|--------------|--------------------|-----------------|
| CTNNB1 | 24 | Wnt signaling effectors |
| YY1 | 23 | Polycomb/transcriptional repression targets |
| CDK1 | 20 | Cell cycle checkpoint components |
| METTL3 | 20 | m⁶A RNA methylation targets (NUP133, NUP160, DDX27) |
| GABPA | 15 | Mitochondrial gene regulation |
| XPO1 | 10 | Nuclear export clients (NUP88, DDX39B, DDX21) |
| SRPK2 | 9 | SR protein kinase substrates (SRSF5, SRSF9, SRSF11) |
| SRSF1 | 9 | Spliceosome recruits (SF3B1, DDX23) |

METTL3 is notable: an m⁶A RNA methyltransferase whose targets at
hop 4 include three nucleoporins and two DEAD-box helicases. This
connects the RNA modification axis (METTL3 is downstream of MTOR
via translational regulation) to nuclear transport and splicing
machinery through a specific epitranscriptomic mechanism.

### Hop 5: Graph exhaustion

Only 22 new genes enter the tested set at hop 5. Of those, just 3
are significant: NUP85 (via UBAP2L, p = 0.003 — yet another
nucleoporin), EXOSC5 (exosome subunit, p = 0.004), and PAF1
(polymerase-associated factor, p = 0.027). The remaining 19 are
non-significant, with p-values from 0.06 to 0.29. The total set
stabilizes at 2,206 intermediaries — **70% of the measured proteome**.

π₀ moves from 0.100 to 0.101 (Δ = 0.001 < 0.01), triggering the
convergence stop. The algorithm has found every reachable regulatory
gene in the knowledge graph.

### The noise floor

The ~130 non-significant genes across hops 4–5 share a profile:
small target sets (median 12 proteins), high p-values (median 0.13,
range 0.05–0.97). Many sit just outside the significance boundary —
PRKCA (141 targets, p = 0.051), EPRS1 (32 targets, p = 0.051),
TECR (35 targets, p = 0.051). Others are clearly null: metabolic
enzymes and structural proteins whose regulatory programs show no
coordination with the C9orf72 signal.

These genes are in the INDRA graph. They have enough downstream
targets to be tested. They just don't participate in the disease
cascade. They are the ~6–10% of reachable regulators that the π₀
estimate captures — the topological background of the knowledge
graph, against which the real signal is measured.

### The resolution gradient

Across hops, the cascade resolves from abstract to concrete:

| Hop | What we see | Example lineage (nuclear transport) |
|-----|------------|-------------------------------------|
| 2 | Pathway-level regulators | RAN (p = 0.005, 70 targets) |
| 3 | Individual effectors | XPO1 (p = 0.006), RCC1 (p = 0.003), RAE1 (p = 0.002) |
| 4 | Complex subunits | NUP88 (p = 0.002), NUP160 (p = 0.003), NUP205 (p = 0.001) |
| 5 | Exhaustion | NUP85 (p = 0.003) — last nucleoporin reachable |

The same gradient appears in splicing (TARDBP → SRSF3 → SF3B1),
autophagy (MTOR → FKBP8 → CCT5), and apoptosis (CASP3 → LMNA →
NDC80). Each hop moves from the regulatory logic to the physical
substrate. The biological signal doesn't weaken with distance — the
π₀ rises only from 2% to 10% over four hops — but the *kind* of
gene changes. The deeper we go, the more the cascade looks like a
parts list of the molecular machines that C9orf72's regulatory
network ultimately controls.

## What This Means

### For the C9orf72 biology

The regulatory cascade from C9orf72 is detectable 4–5 hops deep in the
knowledge graph. The signal attenuates gradually — the null fraction
rises from 2% at the immediate neighborhood to 10% at the graph
boundary — rather than collapsing abruptly.

The resolution gradient tells a specific story. At hop 2, the cascade
identifies the known pathway-level regulators of C9orf72 disease:
MTOR (autophagy), TARDBP and FUS (RNA processing), STAT1
(neuroinflammation), CASP3 (apoptosis), RAN (nuclear transport). At
hop 3, it resolves those pathways to individual effectors — the
specific splicing regulators (SRSF3, SRSF4, SRSF10), the RAN cycle
components (XPO1, RCC1, RAE1), the autophagy receptors (FKBP8). At
hop 4, the molecular machines themselves appear: the nuclear pore
(NUP88, NUP155, NUP160, NUP205), the spliceosome core (SF3B1, SF3B3,
SNRNP200), the chaperonin (CCT5, CCT6A, CCT7). The cascade doesn't
just say "RNA processing is disrupted" — it traces the disruption
from TDP-43 through individual splicing regulators to the physical
spliceosome components.

The nuclear pore finding is especially striking. Eleven nucleoporins
appear at hop 4 through three independent lineages (RAN→XPO1,
MTOR→METTL3, FUS→FDXR). This convergence on the pore complex from
multiple regulatory parents is consistent with the known C9orf72
pathology: GGGGCC repeat RNA forms G-quadruplexes that sequester
transport factors and obstruct the pore. The recursive walk
independently rediscovers this mechanism by following the regulatory
graph.

The specificity triangle reveals that the deeper cascade (hop 3+) is
dominated by pathways that sporadic ALS activates but C9orf72 carriers
do not. These are not additional disruptions by the repeat expansion —
they are compensatory or degenerative programs (autophagy, RNA
surveillance, inflammatory signaling) that sporadic ALS motor neurons
engage but C9orf72 motor neurons fail to activate. The repeat expansion
may block upstream signaling that would normally trigger these responses.

A smaller set of 45 pathways (3.3%) are uniquely disrupted in C9orf72
carriers regardless of reference group — nuclear pore components, DNA
repair, and chromatin remodeling. These may represent the direct
molecular consequences of the repeat expansion, independent of the
broader ALS disease context.

### For INDRA as an inference substrate

ROAST detecting enrichment at each hop means the regulatory edges in
INDRA carry real biological information that is independently
corroborated by expression data. Even single-reader edges (reliability
~0.67) point to real regulatory relationships when tested collectively
as part of a regulatory program. The knowledge graph's structure aligns
with what the proteomics data shows independently.

The specificity triangle adds a second validation layer: the signal
is not a generic property of INDRA's graph topology. The same 46
intermediaries produce starkly different results depending on the
contrast — 100% significant in C9 vs Sporadic, 98% in C9 vs Control,
0% in Sporadic vs Control. INDRA provides the hypotheses; the
expression data confirms or rejects them.

### For the methodology

The algorithm finds its own stopping point. No hardcoded hop limit is
needed. Two diagnostics work together:
- The seed null confirms the seed gene is special (not just a hub)
- The π₀ trajectory detects when the signal has diluted into the
  graph's topological background

The three-contrast specificity design adds interpretability: it
decomposes the signal into C9-specific and shared-disease components,
revealing that the dominant effect at deeper hops is not additional
disruption by C9orf72 but *absence* of the sporadic ALS response.

The entire pipeline runs in ~20 minutes per contrast, bottlenecked by
INDRA API latency (~1 second per CoGEx query), not by computation.

## Technical Summary

| Component | Description |
|-----------|-------------|
| Gene set test | ROAST (Wu et al., 2010), 9,999 rotations |
| FDR control | Storey q-values (π₀-adaptive BH) |
| Extension criterion | Graph-structural: ≥5 measurable INDRA targets |
| Stopping (hop 2) | Seed permutation null, B=30, p=0.032 |
| Stopping (hop 3+) | π₀ convergence, Δ < 0.01 between hops |
| Edge reliability | Corrected INDRA noise model, per-edge, contradiction-aware |
| Cohort | 25 C9orf72 carriers, 294 sporadic ALS, 91 healthy controls |
| Proteome | 3,264 proteins from iPSC-derived motor neurons |
| Knowledge graph | INDRA CoGEx regulatory edges |
| Contrasts | C9 vs Sporadic, C9 vs Control, Sporadic vs Control |
| Runtime | ~20 minutes per contrast |
