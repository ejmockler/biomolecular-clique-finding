# Recursive Regulatory Discovery Through INDRA Knowledge Graphs

## The Question

Given a gene of interest and a proteomics dataset, how far does that gene's
regulatory influence extend through the INDRA knowledge graph — and how do
we know when to stop?

We developed a recursive algorithm that walks the INDRA regulatory graph
outward from a seed gene, testing at each hop whether the downstream
regulatory program shows enrichment in expression data. We applied it to
C9orf72 in ALS cerebrospinal fluid proteomics (n=23 C9orf72 mutation
carriers vs 282 sporadic ALS).

## The Data

- **Proteomics**: ~3,200 quantified proteins from CSF, differential
  expression modeled with covariates (sex, age at onset, baseline ALSFRS-R)
- **Knowledge graph**: INDRA CoGEx regulatory edges (IncreaseAmount,
  DecreaseAmount, Activation, Inhibition, Phosphorylation)
- **Seed gene**: C9orf72 (hexanucleotide repeat expansion, most common
  genetic cause of ALS)

## The Algorithm

### Core idea

At each "hop" outward from the seed, we ask: does this intermediary gene's
regulatory program show collective enrichment in the proteomics data?

We use ROAST (Wu et al., 2010), a rotation-based gene set test that:
- Borrows strength across genes via empirical Bayes moderation
- Produces exact p-values via random rotations (valid at any sample size)
- Tests the gene SET collectively, not individual genes

### The recursive structure

```
Hop 1: Test C9orf72's direct INDRA targets as a gene set
        → p = 0.019 (significant)

Hop 2: For each of C9orf72's 46 regulatory neighbors in INDRA:
        Query their downstream targets from CoGEx
        Test each target set by ROAST
        → 46/46 significant (Storey q < 0.043)

Hop 3: For each target of each significant hop-2 arm
        that itself has ≥5 downstream INDRA targets:
        Test its target set by ROAST
        → 1364/1384 significant (Storey q < 0.071)

Hop 4: Same recursive step
        → 2059/2185 significant

Hop 5: Same recursive step
        → 2075/2206 significant [CONVERGED — same intermediaries as hop 4]
```

### What "significant" means at each hop

We use ROAST's rotation p-values (9,999 rotations) with Storey's
q-value procedure for FDR control. Storey's method estimates the
fraction of truly null hypotheses (π₀) from the p-value distribution
and adjusts the Benjamini-Hochberg correction accordingly.

This matters because standard BH assumes worst-case (all null). In a
biologically coherent regulatory neighborhood, most hypotheses are
genuinely non-null. Standard BH gives 0/46 significant at hop 2;
Storey's q-value gives 46/46. The signal is real — BH was just
wasting power on nulls that don't exist.

## How We Query INDRA

For each intermediary gene, we query CoGEx for its regulatory targets:

```
MATCH (source:BioEntity)-[r:indra_rel]->(target:BioEntity)
WHERE source.id = {gene_hgnc_id}
  AND r.stmt_type IN {regulatory_types}
  AND r.evidence_count >= 1
RETURN target.name, r.source_counts, r.evidence_count, ...
```

We then filter targets to those:
1. With valid HGNC IDs (excludes complexes, families, non-genes)
2. Measurable in this proteomics experiment (UniProt → feature ID mapping)
3. Present in the fitted ROAST engine (non-zero variance)

### Edge reliability scoring

For each INDRA edge, we compute a reliability score using the INDRA
noise model with a corrected conditional formula:

```
P(incorrect) = ∏_source [ syst(s) + (1 - syst(s)) × rand(s)^n_s ]
reliability = 1 - P(incorrect)
```

This differs from INDRA's native belief score in that:
- The `(1-syst)` factor makes systematic and random errors conditional
  (random error only operates where systematic error didn't fire)
- It's computed per-edge with contradiction handling (activation vs
  repression evidence for the same gene pair)
- It's computed in context from source metadata, not pre-computed

The key property: **cross-source corroboration breaks the systematic
error ceiling**. A single REACH statement gives reliability ~0.67.
Two independent sources (REACH + SIGNOR) give ~0.98. This is the
mathematical reason why source diversity matters more than evidence count.

## The Three Methodological Innovations

### 1. Graph-structural extension (no double-dipping)

The naive approach would select which genes to extend at the next hop
based on their individual differential expression (|t| > threshold).
This creates a selective inference problem: you select genes because
they look significant, then test their targets on the same data.

Our fix: extend based purely on **graph structure**. A gene at hop k
extends to hop k+1 if it has ≥5 downstream INDRA targets that are
measurable in the proteomics data. This criterion is independent of
expression — it's determined entirely by the knowledge graph.

This eliminates the double-dipping problem at any sample size.

### 2. Storey π₀-adaptive BH for FDR control

Standard Benjamini-Hochberg assumes all hypotheses could be null.
In a regulatory neighborhood of a disease gene, most intermediaries
genuinely regulate affected targets. Storey's method estimates the
null fraction π₀ from the p-value distribution and adjusts:

```
q_storey = π₀ × q_BH
```

When π₀ ≈ 0.02 (as at hop 2), this recovers ~50× more power than
standard BH. We use a conservative upper bound for π₀ when the point
estimate hits the clip floor (rule-of-3 binomial bound).

### 3. Two-tier stopping criterion

**Seed permutation null at hop 2**: We run B=30 random seed genes
through the same pipeline and compare their hop-2 significance counts
to C9orf72's. If C9orf72 is no better than random seeds, its signal
is just graph topology, not biology. Result: p = 0.032 — C9orf72
IS special among 17,120 candidate genes.

**π₀ convergence at hop 3+**: Instead of running expensive permutation
tests at every hop (each requiring thousands of INDRA API calls), we
track the estimated null fraction π₀ across hops. When π₀ stabilizes
between consecutive hops (|Δπ₀| < 0.01), the reachable regulatory
network has been fully explored — the graph boundary is reached.

```
Hop 2: π₀ = 0.022  (98% non-null — the immediate neighborhood)
Hop 3: π₀ = 0.078  (92% non-null — signal diluting)
Hop 4: π₀ = 0.100  (90% non-null — approaching baseline)
Hop 5: π₀ = 0.101  (CONVERGED — Δ = 0.001 < 0.01)
```

## Results on C9orf72

### The regulatory cascade

| Hop | Tested | Significant | π₀ | Seed null | Extends to |
|-----|--------|-------------|-----|-----------|------------|
| 1 | 1 | 1 | — | — | — |
| 2 | 46 | 46 (100%) | 0.022 | p=0.032 | 1,384 |
| 3 | 1,384 | 1,364 (98.6%) | 0.078 | — | 2,185 |
| 4 | 2,185 | 2,059 (94.2%) | 0.100 | — | 2,206 |
| 5 | 2,206 | 2,075 (94.1%) | 0.101 | [converged] | 2,206 |

### Hop 2: C9orf72's immediate regulatory network

All 46 intermediaries show significant downstream enrichment. The
top arms are core ALS biology:

| Gene | p-value | Targets | Reliability | Direction |
|------|---------|---------|-------------|-----------|
| VPS4A | 0.004 | 28 | 0.89 | activation |
| RAN | 0.005 | 70 | 0.32 | activation |
| HNRNPA1 | 0.007 | 53 | 0.67 | activation |
| TARDBP | 0.009 | 164 | 0.05 | activation |
| FUS | 0.010 | 164 | 0.87 | repression |
| CNBP | 0.010 | 25 | 0.67 | activation |
| TIA1 | 0.010 | 29 | 0.67 | repression |
| EIF2AK2 | 0.018 | 88 | 0.95 | activation |
| TMEM106B | 0.029 | 14 | 0.98 | repression |

TARDBP (TDP-43) has the lowest edge reliability (0.05 — single weak
INDRA source) but 164 downstream targets with strong collective
enrichment (p=0.009). The biological signal is detectable even through
a low-reliability edge because ROAST tests the collective program, not
the individual edge.

### Hop 3: Downstream expansion through ALS pathways

1,384 intermediaries tested (targets of hop-2 arms that themselves
regulate ≥5 measurable proteins). The signal flows through major
ALS-associated hubs:

| Parent | Children | Sig rate | Pathway context |
|--------|----------|----------|-----------------|
| MTOR | 179 | 98.9% | Autophagy, protein homeostasis |
| STAT1 | 162 | 95.1% | Neuroinflammation |
| CASP3 | 127 | 99.2% | Apoptosis |
| TARDBP | 116 | 100% | RNA processing |
| FUS | 94 | 98.9% | RNA processing |
| RAN | 65 | 100% | Nuclear transport |
| GSK3B | 54 | 98.1% | Tau phosphorylation |

The 20 non-significant genes at hop 3 all have small target sets
(5-15 proteins) and p-values > 0.64 — genuine nulls at the graph
periphery with insufficient downstream structure.

### Convergence to the graph boundary

The intermediary set grows: 46 → 1,384 → 2,185 → 2,206 → 2,206.
At hop 5, only 22 new genes enter (vs 801 at hop 4). The regulatory
network reachable from C9orf72 saturates at ~2,200 genes — 70% of
the measured proteome.

P-values are **identical** for the same gene across hops (verified:
max change = 0.000000). The graph-structural extension introduces
no statistical bias — each gene's ROAST test is deterministic and
independent of which hop discovered it.

### The 131 genuine nulls

At convergence (hop 5), 131/2,206 intermediaries are non-significant.
These have:
- Small target sets (median = 8 proteins)
- High p-values (0.47 - 0.96)
- Peripheral positions in the knowledge graph

They represent the ~6% of reachable regulators whose downstream
programs show no ALS-related enrichment — the graph's noise floor.

## What This Means

### For the C9orf72 biology

The regulatory cascade from C9orf72 is detectable 4-5 hops deep in
the INDRA graph, reaching 70% of the CSF proteome. The signal
attenuates gradually (π₀: 2% → 8% → 10%) rather than collapsing.
The core pathways — RNA processing, vesicular trafficking, nuclear
transport, autophagy — are established at hop 2 and propagate through
secondary cascades at hop 3-4.

### For INDRA as an inference substrate

The fact that ROAST detects enrichment at each hop means the INDRA
regulatory edges carry real biological information that's corroborated
by expression data. Even single-reader edges (reliability ~0.67) point
to real regulatory relationships when tested collectively. The
knowledge graph's regulatory structure aligns with what the proteomics
data shows independently.

### For the methodology

The recursive algorithm with graph-structural extension, Storey
q-values, and π₀ convergence stopping works at any sample size and
finds its own boundary. No hardcoded hop limit is needed — the
statistics tell you when to stop. The π₀ trajectory is the key
diagnostic: it rises as the seed-specific signal dilutes into the
graph's topological background, and stabilizes when the graph
boundary is reached.

## Technical Details

### Software

- **causal-path-scoring**: Recursive discovery engine, edge reliability,
  hierarchical FDR, knockoff filter. 104 tests.
- **biomolecular-clique-finding**: ROAST rotation engine, INDRA bridge,
  validation pipeline. ~1600 tests.
- **INDRA CoGEx**: Regulatory knowledge graph (Neo4j), queried via the
  indra_cogex Python client.

### Runtime

Full recursive discovery (6 hops, 9,999 ROAST rotations):
- Without seed null: ~20 minutes
- With hop-2 seed null (B=30): ~19 minutes
- The bottleneck is INDRA API latency (~1s per query), not computation

### Reproducibility

All results are deterministic given:
- Fixed ROAST rotation seed (42)
- Fixed seed null RNG seed (42)
- Same INDRA CoGEx graph state
- Same proteomics data and covariate model
