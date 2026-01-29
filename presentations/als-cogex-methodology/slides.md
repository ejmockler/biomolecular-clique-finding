---
theme: ../slidev-theme-gyori-cogex
title: "AI-Augmented Biomedical Discovery: Knowledge Graphs and LLM Agents for ALS Research"
info: |
  ## AI-Augmented Biomedical Discovery
  Using INDRA CoGEx and LLM Agents to Investigate Sporadic ALS

  A methodology presentation demonstrating how Model Context Protocol (MCP)
  enables systematic knowledge integration in computational biology.

  Gyori Lab - Northeastern University
author: Eric Jing Mockler
transition: slide-left
highlighter: shiki
drawings:
  persist: false
mdc: true
---

::title::
# AI-Augmented Biomedical Discovery

::meta::
**Eric Jing Mockler**<br>
_Gyori Lab, Northeastern University_

January 2026

<!--
SPEAKER NOTES:
Welcome. Today I'm going to show you something that has fundamentally changed how we do computational biology research.

We're going to walk through an actual investigation of sporadic ALS using a new methodology that combines large language models with biomedical knowledge graphs.

This isn't a polished product demo - it's a real research workflow that we used to generate novel insights about ALS pathobiology. By the end, you'll understand not just the biology, but also how to apply this approach to your own research.

[40 minutes total - pace yourself]
-->

---
layout: section
color: gyori
---

# Part 1: The Problem
<hr>
Why biomedical knowledge integration is broken

<!--
Let's start with a problem that everyone in this room has faced.
-->

---
layout: default
---

# The Scale of Biomedical Knowledge

<ScaleOfKnowledge />

<!--
SPEAKER NOTES:
[Let the numbers land - they're designed to be visible from the back of the room]

35 million papers. 60 million protein interactions. This is the ocean we're swimming in.

Now zoom in. Pick ONE protein - β-catenin. 500 pathways. A thousand interactors. Twelve thousand citations. That's still more than any human can hold in working memory.

Miller's law: we can hold 4 plus-or-minus 1 items in working memory at once. You're looking at thousands. For ONE protein.

In your next proteomics experiment, you'll have hundreds of differentially expressed proteins. How do you integrate all of this? The honest answer is: traditionally, we don't. We cherry-pick, or we run enrichment analyses that give us GO terms but miss the mechanistic connections.

That's the problem we're solving.
-->

---
layout: default
---

# The Translational Bottleneck

<TranslationalBottleneck />

<!--
SPEAKER NOTES:
Here's the pattern I see in most omics papers.

[Point to the green flow] This part is easy. Generate data - we're great at that. Run statistics - no problem. Publish a list of 500 dysregulated genes - done.

[Point to the red bottleneck] Then we hit THIS. The interpretation phase.

What do these proteins actually DO together? Not what GO terms they share - what's the mechanism? Which findings are actionable - meaning, can we actually do something about them? What drugs already exist that target these pathways?

These questions require integrating information across databases, across papers, across domains. That's the bottleneck.

[Point to the result] And the result? Supplementary tables that nobody reads. Or worse - we focus only on the genes we already expected, confirmation bias baked into the process.
-->

---
layout: default
---

# What Interpretation Usually Looks Like

<InterpretationTrilemma />

<!--
SPEAKER NOTES:
[Let them read the cards - the GET/LOSE tradeoffs make the dilemma visceral]

These aren't really options - they're compromises. Pick your poison.

Manual literature review gives you deep understanding, but you lose scale. Weeks of work, maybe 20 genes covered. Your proteomics experiment has 500.

Pathway enrichment gives you genome-wide scale, but you lose mechanism. GO terms tell you "cell adhesion" - they don't tell you HOW these proteins interact or which drugs target them.

Expert intuition is fast and gives you a narrative, but you lose objectivity. Confirmation bias is baked in.

[Pause on the question - let it land]

What if there was another way? What if you could query all of biomedical knowledge - 35 million papers, 60 million interactions - as naturally as asking a colleague who has read everything?

That's what I'm going to show you.
-->

---
layout: section
color: cogex
---

# Part 2: The Approach
<hr>
LLMs, Agents, and Model Context Protocol

<!--
Let's talk about how this actually works. I'm going to assume most of you have used ChatGPT or similar tools. But what we're doing is fundamentally different from a chatbot.
-->

---
layout: side-title
color: cogex
titleWidth: wide
---

::title::

# What is a Large Language Model?

A reasoning engine, not a database

::default::

<LlmCapabilities />

<!--
SPEAKER NOTES:
[Let them read the Can/Cannot split - it's the setup]

An LLM is not a database. It's a pattern matcher trained on scientific text.

Think of it this way: if you read 35 million papers, you'd develop intuitions. You'd know "CTNNB1 and WNT signaling probably go together" or "mitochondrial dysfunction shows up in neurodegeneration." That's what an LLM has.

[Point to the green side] It CAN interpret your question, reason about biology, explain relationships.

[Point to the red side - this is the key] But it CANNOT access real-time data. It can't look things up. It might confidently tell you something that was true in 2021 but isn't anymore. Or worse - something that was never true.

[Point to the hook] So how do we ground this reasoning capability in actual, current data? That's where agents come in.
-->

---
layout: default
---

# From Chatbot to Agent

<ChatbotToAgent />

<!--
SPEAKER NOTES:
[The question at top anchors both sides - same input, different outputs]

Here's the key evolution. Both see the same question: "What drugs target ROCK1?"

[Point to the left] The chatbot searches its memory. It gives you fasudil and Y-27632 - which is true! But it's from 2021 training data. What's been discovered since? What's it missing?

[Point to the + TOOLS in the middle] Add tools.

[Point to the right] The agent doesn't guess - it queries. It runs query_database("ROCK1") and returns 38 current modulators. Specific. Verifiable. Complete.

This is the difference between "I think I remember..." and "Let me look that up for you."

The reasoning capability is the same. But one is grounded in current knowledge, one is not.
-->

---
layout: default
---

# What is MCP?

<div class="text-3xl text-[#4ecadf] -mt-6 mb-4 text-center w-full">Model Context Protocol</div>

<McpExplainer />

<!--
SPEAKER NOTES:
MCP - Model Context Protocol - is the universal adapter that makes this work.

[Point to the human question at top] You ask in plain English. "What genes interact with LRRK2?"

[Point to the MCP port] MCP translates that into structured queries - Cypher, SQL, API calls - whatever each tool needs.

[Point to the connected tools] And it can connect to anything: CoGEx for protein interactions, DrugBank for compounds, PubMed for literature, UniProt for sequences.

[Point to the crossed-out Cypher query] See that query at the bottom? That's what you'd have to write without MCP. MATCH, WHERE, RETURN... You never write it. You just ask in English, and MCP handles the translation.

That's the key benefit: the complexity is hidden. You work at the level of scientific questions, not database syntax.
-->

---
layout: default
---

# The MCP Architecture

<McpArchitecture />

<!--
SPEAKER NOTES:
This is the actual architecture of the INDRA CoGEx MCP server.

The key innovation is the Intelligent Gateway layer. Instead of exposing 190+ individual API functions - which would overwhelm any LLM's context - we expose just 4 gateway tools.

[Point to each tool]
- ground_entity: Converts natural language to database identifiers using GILDA. Semantic filtering means "ALS" becomes the disease or the gene depending on context.
- suggest_endpoints: Given what you have, shows what you can reach. Navigation hints.
- call_endpoint: The unified interface. One tool that can call any of the 190+ underlying functions.
- get_nav_schema: The map of the knowledge graph - what connects to what.

The processing pipeline handles validation, query execution, token-aware pagination, and result enrichment. Progressive disclosure means results adapt to how much detail you need.

This architecture is what makes conversational knowledge graph queries practical.
-->

---
layout: default
---

# INDRA CoGEx
## Integrated Network and Dynamical Reasoning Assembler

<CoGExOverview />

<!--
SPEAKER NOTES:
You know this system better than anyone - you built it. What I want to highlight is what makes CoGEx uniquely suited for LLM-driven discovery.

The scale - 92 million entities, 573 million relationships - is what makes the MCP gateway necessary. No LLM could process raw Cypher results at this scale without the pagination and token management we built.

The evidence layer is crucial: 45 million evidence records with embeddings means the agent can assess confidence, not just retrieve facts. When it returns "LRRK2 phosphorylates RAB10" with belief 0.98 and 47 evidence sources, that's actionable.

The mechanistic vocabulary - those 31 statement types - is what distinguishes this from simpler drug-target databases. The agent can reason about activation vs inhibition, phosphorylation vs ubiquitination. That semantic richness is what enables the hypothesis generation we'll show later.

The research ecosystem integration - trials, NIH projects, publications - lets the agent traverse from molecular mechanism to clinical relevance in a single query chain.
-->

---
layout: default
---

# What You Can Ask

<QueryExamples />

<!--
SPEAKER NOTES:
Here are the kinds of questions you can ask. Notice that these are plain English - you don't need to know any query language.

[Point to each category] Drug queries, disease queries, pathway queries, gene function. Each category has examples of the kinds of questions the agent can handle.

Behind the scenes, the agent is translating "What drugs target ROCK1?" into a structured call to the knowledge graph. It handles entity grounding - figuring out that ROCK1 means HGNC:10251, not some other protein with a similar name.

[Point to the flow at bottom] You ask in natural language, the agent translates to structured queries, you get back interpreted answers. That's the magic of combining LLMs with knowledge graphs.
-->

---
layout: default
---

# The Setup We Used

<div class="mt-8 p-6 bg-gray-900 text-green-400 font-mono rounded-lg">

```bash
$ claude --mcp indra-cogex

Connected to INDRA CoGEx MCP server
Available: ground_entity, get_drugs_for_gene,
           get_pathways_for_gene, get_go_terms...

> What drugs target ROCK1?

Querying... Found 38 modulators for ROCK1 (HGNC:10251):
- fasudil (CHEBI:31766) - 17 ALS trials
- baricitinib (CHEBI:135684) - JAK/ROCK inhibitor
...
```

</div>

<div class="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">

**Key feature:** Results include database IDs (HGNC:10251, CHEBI:31766) - fully traceable and verifiable

</div>

<!--
SPEAKER NOTES:
This is what it actually looks like. It's a terminal session - not a polished web app. You type questions, it queries the knowledge graph, and it returns interpreted results.

The key thing to notice is that the results include database identifiers - HGNC:10251, CHEBI:31766. This is important because it means the answers are traceable. You can verify them.

This is not a black box giving you unverifiable claims. It's grounded in real databases that you can check.
-->

---
layout: section
color: bio
---

# Part 3: The ALS Investigation
<hr>
A real case study from proteomics to mechanism

<!--
Now let's see this in action. I'm going to walk through our actual investigation of sporadic ALS proteomics data.
-->

---
layout: default
---

# The Starting Point: sALS Proteomics

<SalsSetup />

<!--
SPEAKER NOTES:
Here's where we started. CSF proteomics from sporadic ALS patients, stratified by sex.

[Point to the transformation visual] The key innovation: we don't test individual proteins - we test CLIQUES. A clique is a group of proteins that correlate with each other across samples.

[Point to scattered dots → grouped cliques] See the difference? Individual proteins give you a multiple testing nightmare. Grouping them into co-regulated modules captures the biology - proteins that function together change together.

[Point to ROAST] ROAST is the statistical engine. Unlike other tests that break correlations, ROAST preserves them by rotating residuals. Critical for cliques where correlation IS the biology.

[Point to 1,777] We tested 1,777 cliques. That's 1,777 regulatory modules, each representing coordinated biology.

The question: which modules are dysregulated in ALS?
-->

---
layout: default
---

# Initial Results: The Lists

<div class="grid grid-cols-4 gap-4 mt-6">

<StatBox value="1,777" label="Cliques Tested" color="neutral" />
<StatBox value="1" label="Female Top Hits" color="bio" />
<StatBox value="5" label="Male Top Hits" color="cogex" />
<StatBox value="1,777" label="Interaction Hits" color="gyori" />

</div>

<div class="mt-8 text-center text-xl">

**ALL 1,777 cliques showed significant sex × disease interaction (p < 0.05)**

Sporadic ALS is fundamentally sex-dimorphic

</div>

<!--
SPEAKER NOTES:
Here's what the analysis produced.

In females, one clique stood out - RAP1A, containing RHOA and related proteins. Strongly downregulated.

In males, the SWI/SNF chromatin remodeling complex was upregulated.

But look at the interaction analysis - ALL 1,777 cliques showed significant sex-by-disease interaction. This tells us that sporadic ALS is not one disease - it's sex-dimorphic at a fundamental level.

Now, this is where traditional analysis usually stops. We have lists. We could write a paper saying "RHOA signaling is suppressed in female sALS" and cite some literature. But that's not very satisfying.

Let's dig deeper using the knowledge graph.
-->

---
layout: default
---

# The Agent-Assisted Investigation Begins

<div class="p-6 bg-gray-900 text-green-400 font-mono rounded-lg text-sm">

```
> RHOA is downregulated in female ALS. What drugs target this pathway?

Grounding RHOA... HGNC:10001
Querying drug modulators...

RHOA downstream effector ROCK1 has 38 modulators:
  fasudil      - ROCK inhibitor, 17 ALS trials
  baricitinib  - JAK/ROCK, FDA approved
```

</div>

<div class="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">

**First query → Actionable drug candidates in 30 seconds**

The RHOA→ROCK1 axis inhibits axon regeneration. ROCK1 inhibitors promote axonal growth.

</div>

<!--
SPEAKER NOTES:
Here's what happened when we asked about drug targets.

RHOA itself doesn't have drugs targeting it directly. But the agent knew to look at the downstream pathway - ROCK1. And ROCK1 has 38 known modulators, including several FDA-approved drugs.

Even more interesting: fasudil, a ROCK inhibitor, already has 17 ALS clinical trials. We didn't know this before querying.

In 30 seconds, we went from "RHOA is downregulated" to "here are FDA-approved drugs that target this pathway, and one is already being tested in ALS."

This is the power of systematic knowledge integration.
-->

---
layout: default
---

# Discovery: The Drug Landscape

| Target | # Drugs | Status |
|--------|---------|--------|
| ROCK1 | 38 | 17 ALS trials |
| SIGMAR1 | 50+ | ALS gene, SSRIs |
| **SOD1** | **0** | **Critical gap** |
| **TDP-43** | **0** | **Critical gap** |

<div class="mt-8 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-l-4 border-red-500">

**Major finding:** The two most important ALS genes (SOD1, TDP-43) have NO known drug modulators in any database.

</div>

<div class="mt-6">

This emerged not from reading one paper, but from systematically querying the knowledge graph.

</div>

<!--
SPEAKER NOTES:
This systematic querying revealed the drug landscape for ALS.

Some targets have many drugs - ROCK1 has 38, and trials are already happening. Good - we can monitor those.

But look at SOD1 and TDP-43. These are the two most important genes in ALS biology. SOD1 mutations cause familial ALS. TDP-43 aggregation is the hallmark of sporadic ALS.

And neither has any known drug modulators.

This is a critical gap. It emerged not from reading one paper, but from systematically querying the knowledge graph. This is exactly the kind of insight that gets lost when you do selective literature review.
-->

---
layout: default
---

# Digging Into Mechanisms

<div class="p-6 bg-gray-900 text-green-400 font-mono rounded-lg text-sm mt-4">

```
> What proteins appear most frequently in sex-interaction cliques (p<0.001)?

Frequency analysis of 586 cliques:
  CTNNB1 (β-catenin)    181 appearances
  BAX                     87 appearances
  CDH1 (E-cadherin)      70 appearances
  APP                     45 appearances
```

</div>

<div class="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border-l-4 border-yellow-500">

**Key insight:** CTNNB1 (β-catenin) appears in 181 cliques. Its GO terms include "response to estrogen" — connecting to sex dimorphism.

</div>

<!--
SPEAKER NOTES:
We asked: which proteins appear most often in the sex-interaction cliques?

CTNNB1 - β-catenin - appeared 181 times. That's extraordinary. BAX, an apoptosis protein, appeared 87 times.

Then we asked about β-catenin's functions. Look at the GO terms - cell adhesion, Wnt signaling, apoptosis regulation. And notably: response to estrogen.

This connects to the sex-differential effect. β-catenin is involved in estrogen response. Maybe this explains why males and females show different disease patterns?

This hypothesis emerged from the data, not from our preconceptions.
-->

---
layout: default
---

# Mechanistic Convergence

<div class="mt-6">

The top proteins from interaction cliques converge on **mitochondrial apoptosis**:

</div>

<div class="flex justify-center items-center gap-4 mt-8 text-lg">

<div class="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-center">
<div class="text-2xl font-bold">DNM1L</div>
<div class="text-xs text-gray-500">Fission</div>
</div>

<div class="text-3xl">→</div>

<div class="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-center">
<div class="text-2xl font-bold">BAX</div>
<div class="text-xs text-gray-500">Permeabilization</div>
</div>

<div class="text-3xl">→</div>

<div class="p-4 bg-red-100 dark:bg-red-900/30 rounded-lg text-center">
<div class="text-2xl font-bold">CYCS</div>
<div class="text-xs text-gray-500">Release</div>
</div>

</div>

<div class="flex justify-center mt-6">
<div class="text-3xl">↓</div>
</div>

<div class="flex justify-center">
<div class="p-4 bg-red-200 dark:bg-red-800/30 rounded-lg text-center text-xl">
Apoptosome → Caspases → Motor Neuron Death
</div>
</div>

<!--
SPEAKER NOTES:
When we looked up the pathways for our top proteins, they all converged on the same place: mitochondrial apoptosis.

DNM1L causes mitochondrial fission - fragmenting mitochondria before cell death. BAX permeabilizes the mitochondrial membrane. CYCS is cytochrome c, which gets released and activates caspases.

This is the intrinsic apoptosis pathway. And our sex-interaction cliques are full of it.

We didn't go looking for apoptosis. We asked "what functions do these proteins share?" and apoptosis emerged as the answer. This is hypothesis generation from data, not hypothesis confirmation.
-->

---
layout: default
---

# Tissue Localization: Validating Relevance

<div class="p-6 bg-gray-900 text-green-400 font-mono rounded-lg text-sm mt-4">

```
> Do these proteins express in ALS-relevant tissues?

Querying tissue expression for top proteins...

All 5 proteins expressed in:
  - Cervical spinal cord    ← Primary ALS site
  - Prefrontal cortex       ← Upper motor neurons
  - Skeletal muscle         ← Denervation target
```

</div>

<div class="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-l-4 border-green-500">

**Validation:** These proteins are expressed exactly where ALS pathology occurs. Not a random statistical artifact.

</div>

<!--
SPEAKER NOTES:
We also asked: do these proteins actually express in relevant tissues?

Yes. All of them are expressed in the cervical spinal cord, motor cortex, peripheral nerves, and muscle - exactly the tissues affected in ALS.

This is important. It validates that our findings aren't statistical artifacts. These proteins are present where the disease happens.

This kind of cross-validation is easy when you have access to a knowledge graph. It would take hours to look up manually.
-->

---
layout: two-cols
gap: md
---

# The Mechanistic Model

::left::

```
┌────────────────────┐
│    GENETIC         │
│ (SOD1, TDP-43)     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  SEX MODIFIERS     │
│ Female: DNMT1      │
│ Male: SWI/SNF      │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ CELLULAR STRESS    │
│ • Mitochondria     │
│ • Junctions        │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ MOTOR NEURON DEATH │
└────────────────────┘
```

::right::

## Key insight

sALS is a **sex-dimorphic syndrome**

### Female pathway
- DNMT1 downregulation
- Estrogen-β-catenin axis

### Male pathway
- SWI/SNF activation
- RNA splicing defects

### Shared endpoint
- Mitochondrial apoptosis
- Motor neuron death

<!--
SPEAKER NOTES:
Putting it all together, here's the mechanistic model that emerged.

Sporadic ALS has sex-specific upstream mechanisms converging on a shared endpoint - motor neuron death via mitochondrial apoptosis.

In females, we see epigenetic changes - DNMT1 downregulation. Combined with the estrogen-β-catenin connection, this suggests a pathway involving hormonal protection.

In males, we see chromatin remodeling activation and RNA splicing changes - more reminiscent of the TDP-43 splicing pathology.

This model was CONSTRUCTED through iterative querying. It wasn't something we hypothesized and confirmed. It emerged from the data and the knowledge graph together.
-->

---
layout: section
color: gradient
---

# Part 4: Pitfalls and Mitigations
<hr>
What can go wrong and how to handle it

<!--
Now let's talk about what can go wrong. This is important - I want you to use this approach, but I want you to use it correctly.
-->

---
layout: default
---

# Pitfall 1: Hallucination

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-300">

### The Problem

```
> What trials exist for drug X?

"Drug X has been tested in 5 trials
 for ALS, showing promising results..."

 [No such trials exist]
```

</div>

<div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-300">

### The Mitigation

```
> What trials exist for drug X?

Querying ClinicalTrials.gov...
No trials found for drug X in ALS.

Result: 0 trials
```

</div>

</div>

<div class="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border-l-4 border-yellow-500">

**Rule:** Trust the data source, not the prose.

</div>

<!--
SPEAKER NOTES:
Hallucination is the biggest risk with LLMs. They can confidently make things up.

But this is exactly why we use MCP and knowledge graphs. When the agent says "fasudil has 17 ALS trials," that number came from ClinicalTrials.gov via the knowledge graph. It's not fabricated.

The key practice: always note the source. If the agent says "querying DrugBank..." then you know the answer is grounded. If it just says "based on my knowledge..." be skeptical.
-->

---
layout: default
---

# Pitfall 2: Entity Ambiguity

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-300">

### The Problem

- **"ALS"** → Disease? Or SOD1 gene?
- **"CDH1"** → E-cadherin? Or FZR1?
- **"p53"** → Protein? Gene? Pathway?

</div>

<div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-300">

### The Mitigation

```python
# With semantic context
ground_entity("ALS",
  param_name="disease")
# MESH:D000690

ground_entity("ALS",
  param_name="gene")
# HGNC:396 (SOD1)
```

</div>

</div>

<!--
SPEAKER NOTES:
Entity ambiguity is a subtle but serious problem.

ALS can mean the disease or it can be an alias for the SOD1 gene. CDH1 can be E-cadherin or FZR1, which are completely different proteins.

The MCP uses GILDA for entity grounding with semantic context. If you say "the disease ALS," it knows to look for MESH:D000690. If you say "the gene ALS," it knows to look for HGNC:396.

But you should always check the CURIE - the standardized identifier. That's your verification that the right entity was used.
-->

---
layout: default
---

# Pitfall 3: Confirmation Bias

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-300">

### The Problem

- Only ask expected questions
- Find only what you seek
- Selection bias in queries

</div>

<div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-300">

### The Mitigation

- Query broadly first (unbiased)
- Ask for counter-examples
- Let agent suggest queries

</div>

</div>

<!--
SPEAKER NOTES:
Confirmation bias is a human problem, not a technical one. But it's amplified when you have a powerful query tool.

If you only ask about what you expect to find, you'll find it. The agent will dutifully return results about apoptosis if that's all you ask about.

The mitigation is methodological: query broadly first. Ask "what pathways contain these proteins?" not "are these proteins in apoptosis?" Let the data tell you what's there.

Also, actively ask for counter-examples. "What genes in these cliques are NOT in the apoptosis pathway?" This keeps you honest.
-->

---
layout: default
---

# Pitfall 4: Knowledge Graph Incompleteness

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-300">

### The Problem

**Absence of evidence ≠ evidence of absence**

- No drugs found for TDP-43
- Means: not in THIS database
- NOT: no drugs exist

</div>

<div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-2 border-green-300">

### The Mitigation

- Cross-reference with literature
- Note gaps explicitly
- Use multiple sources

</div>

</div>

<!--
SPEAKER NOTES:
This is important: the knowledge graph is comprehensive but not complete.

When we found that SOD1 and TDP-43 have no drug modulators, that means they're not in DrugBank or ChEMBL. There might be experimental compounds in development that haven't been registered yet.

Always note what database the answer came from, and be precise in your language. "No modulators in DrugBank" is different from "no modulators exist."

And for important findings, cross-reference with other sources. Don't rely solely on any single database.
-->

---
layout: default
---

# Best Practices Summary

<div class="mt-6 space-y-3">

<div class="p-4 bg-gyori-50 dark:bg-gyori-900/30 rounded-lg">

**1. Verify Entity Grounding** - Check the CURIE (database ID)

</div>

<div class="p-4 bg-cogex-50 dark:bg-cogex-900/30 rounded-lg">

**2. Trace Provenance** - Note where facts came from

</div>

<div class="p-4 bg-bio-50 dark:bg-bio-900/30 rounded-lg">

**3. Query Broadly First** - Let data guide hypotheses

</div>

<div class="p-4 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">

**4. Document Conversation** - Save sessions as methods

</div>

</div>

<!--
SPEAKER NOTES:
Let me summarize the best practices.

1. Verify Entity Grounding: Always check the CURIE (database identifier) for queried entities.

2. Trace Provenance: Note where each fact came from. "DrugBank says..." is verifiable.

3. Query Broadly First: Ask open-ended questions before specific ones. Let the data guide your hypotheses.

4. Document the Conversation: Save your query session. It's your methods section. Others should be able to reproduce your reasoning.

The overarching point is: this is a powerful tool, but it requires discipline. Used well, it amplifies your expertise. Used poorly, it amplifies your biases.
-->

---
layout: section
color: bio
---

# Part 5: Results and Implications
<hr>
What we learned and where it leads

<!--
Let's wrap up with the concrete results and what they mean for ALS research.
-->

---
layout: default
---

# Actionable Findings for ALS Research

<div class="mt-6 space-y-3 text-sm">

| Finding | Implication |
|---------|-------------|
| ROCK inhibitors in trials | Monitor fasudil results |
| SIGMAR1 ↔ SSRIs | Repurposing opportunity |
| SOD1/TDP-43 no modulators | Critical drug discovery gap |
| Sex × disease interaction | Stratify future trials by sex |

</div>

<!--
SPEAKER NOTES:
Here are the concrete, actionable findings.

Some are immediately translatable - ROCK inhibitors are already in trials, we just need to watch those results. SSRIs target SIGMAR1, which is an ALS gene - maybe worth an epidemiological study looking at ALS risk in long-term SSRI users.

Some identify gaps - the lack of drugs for SOD1 and TDP-43 should be a priority for drug discovery efforts.

Some are methodological - the sex-interaction findings suggest current trials might need stratification.
-->

---
layout: default
---

# What This Approach Enabled

<div class="grid grid-cols-3 gap-6 mt-8">

<div class="p-6 bg-gray-100 dark:bg-gray-800 rounded-lg text-center">

<div class="text-5xl font-bold text-gyori-500">50+</div>

<div class="text-sm mt-2">queries in 2 hours</div>

</div>

<div class="p-6 bg-gray-100 dark:bg-gray-800 rounded-lg text-center">

<div class="text-5xl font-bold text-cogex-500">100%</div>

<div class="text-sm mt-2">coverage, no cherry-picking</div>

</div>

<div class="p-6 bg-gray-100 dark:bg-gray-800 rounded-lg text-center">

<div class="text-5xl font-bold text-bio-500">3+</div>

<div class="text-sm mt-2">unexpected connections</div>

</div>

</div>

<!--
SPEAKER NOTES:
Let me quantify what this approach enabled.

We ran 50+ structured queries in about 2 hours. That same coverage would take weeks of manual literature review.

We covered 100% of our top differential proteins systematically. No cherry-picking based on what we expected.

And we found unexpected connections - the CTNNB1-estrogen link, the drug target gaps, the extent of sex dimorphism.

The fundamental shift is from "what do we already know?" to "what does ALL of biomedical knowledge tell us?"
-->

---
layout: default
---

# Future Directions

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

## For This ALS Project

- Experimental validation
- Clinical collaboration

</div>

<div>

## For the Methodology

- Broader application
- More knowledge sources

</div>

</div>

<!--
SPEAKER NOTES:
Looking forward, there are two threads.

For ALS specifically, we need experimental validation. The mitochondrial fission hypothesis needs testing in models. The sex differences need confirmation in independent cohorts.

For the methodology, this is generalizable. Any omics experiment can benefit from this approach. And there are MCP servers for many other databases - UniProt, PubMed, ChEMBL.

Saved query sessions document your analytical reasoning. Someone else can re-run your queries and see exactly how you reached your conclusions.
-->

---
layout: default
---

# Summary

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

## The Methodology

- LLMs + MCP + Knowledge Graphs
- Grounded in real databases
- Pitfalls are manageable

</div>

<div>

## The Biology

- sALS is sex-dimorphic
- Mitochondrial apoptosis converges
- Drug targets exist, gaps remain

</div>

</div>

<div class="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">

**Amplifying expertise, not replacing it**

</div>

<!--
SPEAKER NOTES:
Let me leave you with two summaries - one for methodology, one for biology.

The bottom line: this is amplification, not replacement. You still need domain expertise. You still need to design good experiments. You still need to think critically about the results.

But the bottleneck of "how do I integrate all this knowledge?" is now addressable. You can ask questions at scale and get grounded answers. That changes what's possible.
-->

---
layout: end
---

# Thank You

Questions and Discussion

<div class="contact-info">
  <div class="contact-item">
    <carbon-logo-github /> github.com/gyorilab
  </div>
  <div class="contact-item">
    <carbon-link /> indra.bio
  </div>
</div>

<div class="acknowledgments">

**INDRA CoGEx:** Benjamin Gyori, John Bachman, Northeastern University

**Cliquefinding Pipeline:** Developed for the ALS proteomics collaboration

**MCP Infrastructure:** Anthropic

</div>

<!--
SPEAKER NOTES:
Thank you. I'm happy to take questions.

If you're interested in trying this yourself:
- INDRA CoGEx is available at indra.bio
- The MCP server documentation is on GitHub
- Feel free to reach out if you want help setting it up for your data

Questions?
-->
