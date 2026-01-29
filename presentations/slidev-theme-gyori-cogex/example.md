---
theme: ./
title: INDRA CoGEx Analysis of Proteomic Cliques in ALS
info: |
  ## Cliquefinding Pipeline Analysis
  Mechanistic interpretation of differential protein cliques using INDRA CoGEx MCP.

  Gyori Lab - Computational Biomedicine
author: Your Name
transition: slide-left
highlighter: shiki
drawings:
  persist: false
mdc: true
---

# INDRA CoGEx Analysis
## Mechanistic Interpretation of Differential Protein Cliques in ALS

<div class="pt-12">

**Your Name**
_Gyori Lab - Laboratory of Systems Pharmacology_
_Harvard Medical School_

</div>

<div class="abs-br m-6 text-sm opacity-50">
January 2026
</div>

---
layout: section
color: gyori
---

# Background
<hr>
From correlation networks to mechanistic insight

---
layout: two-cols
gap: lg
---

# The Cliquefinding Pipeline

::left::

## Correlation Networks

Proteins that correlate across samples form **cliques** - fully connected subgraphs.

- High-throughput proteomics data
- Pearson/Spearman correlation
- Graph-based clique enumeration

::right::

## Differential Analysis

ROAST rotation testing identifies cliques with:

<div class="mt-4 space-y-2">

- **Coordinated upregulation** in disease
- **Coordinated downregulation** in disease
- **Sex-specific** effects

</div>

---
layout: default
---

# Key Results: ALS Proteomics

<div class="grid grid-cols-4 gap-4 mt-8">

<StatBox value="1,777" label="Cliques Tested" color="gyori" />
<StatBox value="847" label="Significant (p<0.05)" color="bio" />
<StatBox value="312" label="Upregulated" color="gyori" />
<StatBox value="535" label="Downregulated" color="cogex" />

</div>

<div class="mt-8">

### Top Differential Cliques

| Clique | Size | Direction | p-value | Key Proteins |
|--------|------|-----------|---------|--------------|
| 42 | 8 | ↑ Up | 0.0001 | *SOD1*, *TARDBP*, *FUS* |
| 156 | 12 | ↓ Down | 0.0002 | *NEFL*, *NEFM*, *NEFH* |
| 78 | 6 | ↑ Up | 0.0005 | *SQSTM1*, *OPTN*, *TBK1* |

</div>

---
layout: section
color: cogex
---

# INDRA CoGEx
<hr>
Causal knowledge graph analysis

---
layout: side-title
color: cogex
---

::title::

# What is INDRA CoGEx?

Integrated Network and Dynamical Reasoning Assembler

::default::

## Knowledge Graph Integration

CoGEx combines multiple biomedical knowledge sources:

- **INDRA Statements** - Machine-read causal assertions
- **Pathway databases** - Reactome, KEGG, WikiPathways
- **Gene Ontology** - Biological processes, molecular functions
- **Disease associations** - DisGeNET, OMIM

## MCP Server Access

Query causal mechanisms directly via Model Context Protocol:

```python
# Example: Find mechanisms connecting SOD1 to neurodegeneration
results = cogex.get_causal_paths(
    source="SOD1",
    target="neurodegeneration",
    max_depth=3
)
```

---
layout: default
---

# Clique Mechanistic Analysis

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

## Clique 42: Autophagy Regulators

<CliqueCard
  id="42"
  name="Autophagy-related proteins"
  :size="8"
  direction="up"
  :pvalue="0.0001"
  :genes="['SQSTM1', 'OPTN', 'TBK1', 'CALCOCO2', 'NBR1']"
  showGenes
/>

</div>

<div>

## INDRA Pathway Analysis

<StatementCard
  subject="TBK1"
  object="OPTN"
  type="activation"
  :evidence="47"
  :sources="['PubMed', 'Reactome']"
/>

<StatementCard
  subject="SQSTM1"
  object="autophagosome"
  type="complex"
  :evidence="89"
  :sources="['GO', 'KEGG']"
  class="mt-3"
/>

</div>

</div>

---
layout: default
---

# Causal Mechanism Discovery

<div class="mt-6 mb-8">

<CausalPath :nodes="[
  { name: 'SOD1 mutation', type: 'gene' },
  { name: 'Protein aggregation', type: 'process' },
  { name: 'SQSTM1', type: 'protein' },
  { name: 'Autophagy activation', type: 'process' },
  { name: 'Cellular stress', type: 'process' }
]" :edges="[
  { type: 'activation' },
  { type: 'activation' },
  { type: 'activation' },
  { type: 'inhibition' }
]" />

</div>

<div class="grid grid-cols-2 gap-6">

<ResultsHighlight
  title="Novel TBK1-SQSTM1 Connection"
  finding="CoGEx identifies a previously underappreciated phosphorylation cascade linking TBK1 kinase activity to SQSTM1-mediated selective autophagy in ALS motor neurons."
  type="discovery"
  :pvalue="0.0001"
  effect="8-protein clique"
/>

<ResultsHighlight
  title="Validated by Literature"
  finding="The identified TBK1→OPTN→SQSTM1 axis is supported by 47 independent publications and matches known ALS pathogenic mechanisms."
  type="validation"
/>

</div>

---
layout: default
---

# Supporting Evidence

<div class="mt-4">

<EvidenceList :items="[
  {
    pmid: '28245873',
    text: 'TBK1 phosphorylates OPTN at Ser177, enhancing its interaction with LC3 and promoting selective autophagy of protein aggregates.',
    source: 'Cell Reports',
    year: 2017
  },
  {
    pmid: '27523608',
    text: 'Loss-of-function mutations in TBK1 cause familial ALS through impaired autophagy and accumulation of misfolded proteins.',
    source: 'Nature Neuroscience',
    year: 2016
  },
  {
    pmid: '29224782',
    text: 'SQSTM1/p62 bodies are present in sporadic ALS spinal cord motor neurons, suggesting autophagy pathway involvement.',
    source: 'Acta Neuropathol',
    year: 2018
  }
]" />

</div>

---
layout: two-cols
gap: md
---

# Sex-Stratified Analysis

::left::

## Male-Specific Effects

<div class="space-y-3 mt-4">

<CliqueCard
  id="89"
  :size="5"
  direction="up"
  :pvalue="0.002"
/>

<CliqueCard
  id="234"
  :size="7"
  direction="down"
  :pvalue="0.008"
/>

</div>

::right::

## Female-Specific Effects

<div class="space-y-3 mt-4">

<CliqueCard
  id="112"
  :size="6"
  direction="down"
  :pvalue="0.001"
/>

<CliqueCard
  id="445"
  :size="4"
  direction="up"
  :pvalue="0.015"
/>

</div>

---
layout: section
color: bio
---

# Conclusions
<hr>
Integrating data-driven discovery with mechanistic knowledge

---
layout: default
---

# Summary

<div class="grid grid-cols-2 gap-8 mt-8">

<div>

## Key Findings

1. **847 significant cliques** identified in ALS proteomics

2. **Autophagy pathway** strongly implicated through clique analysis

3. **Sex-specific effects** reveal distinct disease mechanisms

4. **INDRA CoGEx** provides mechanistic interpretation

</div>

<div>

## Methods Innovation

- **ROAST rotation testing** for clique-level inference
- **MCP integration** for real-time knowledge graph queries
- **Perceptual engineering** for visualization design

<div class="mt-6 p-4 bg-gyori-50 dark:bg-gyori-900 rounded-lg border-2 border-gyori-500">

### Future Directions

- Multi-omics clique integration
- Causal network inference
- Drug target prioritization

</div>

</div>

</div>

---
layout: end
---

# Thank You

Questions and Discussion

<div class="contact-info">
  <div class="contact-item">
    <carbon-email /> your.email@institution.edu
  </div>
  <div class="contact-item">
    <carbon-logo-github /> github.com/username
  </div>
  <div class="contact-item">
    <carbon-logo-twitter /> @username
  </div>
</div>

<div class="acknowledgments">

**Acknowledgments:** Gyori Lab, Harvard Medical School
**Funding:** NIH Grant #XXXXX

</div>
