# Computational Biology Glossary

## For Engineers

This document bridges the engineering and biology domains. Understanding these concepts is critical for making good design decisions in the platform.

---

## Core Data Types

### Expression/Count Data
**What it is:** A matrix where rows = genes/proteins, columns = samples, values = abundance.

**Example:**
```
             Sample1  Sample2  Sample3
Gene_BRCA1      1250     980     1150
Gene_TP53        450     520      380
Gene_ACTB       8900    9200     8800
```

**Engineering analogy:** Think of it as a feature matrix in ML, but features are genes and samples are observations.

**Key properties:**
- **High-dimensional:** 20K-60K genes, hundreds to thousands of samples
- **Sparse:** Many zeros (genes not expressed in certain conditions)
- **Heavy-tailed:** A few genes have very high counts, most are low
- **Noisy:** Biological and technical variation

### RNA-seq vs. Proteomics
**RNA-seq:** Measures mRNA abundance (transcriptomics)
- Tells you what genes are being transcribed
- Proxy for protein levels (but not perfect)
- Cheaper, more comprehensive

**Proteomics:** Measures protein abundance directly
- More relevant biologically (proteins do the work)
- More expensive, harder to measure
- Our current dataset is proteomic

### Gene Identifiers
**ENSG00000139618** - Ensembl gene ID (standard nomenclature)
- ENSG = Ensembl Gene
- Numbers uniquely identify each gene
- Stable across versions (mostly)

**Alternative IDs:** HGNC symbols (e.g., BRCA1), RefSeq IDs, Entrez IDs
- **Engineering note:** Always store canonical IDs (ENSG), keep symbol mappings separate

---

## Biological Concepts

### Transcription Factors (TFs)
**What they are:** Proteins that bind to DNA and control which genes are "turned on"

**Why they matter:** Master regulators of cellular behavior
- Cancer often involves TF dysregulation
- ALS may involve TF network disruption

**Example:**
- TF "MYC" is activated
- MYC binds to DNA near genes involved in cell growth
- Those genes get transcribed more
- Cell grows faster (can lead to cancer)

**Engineering analogy:** TFs are like config files that control which services (genes) start up in a system.

### Cliques in Gene Networks
**What they are:** Fully-connected subgraphs in a gene co-expression network

**Why they matter:** Genes in a clique are all highly correlated with each other
- Suggests they're regulated together (co-regulation)
- May be controlled by the same TF
- Form functional modules (e.g., all involved in same pathway)

**Example:**
```
Correlation network:
  A --- B
  |  X  |
  C --- D

{A, B, C, D} is a clique (all 4 genes highly correlated)
```

**Engineering analogy:** Think of microservices that always scale together - they're functionally coupled.

**Our task:** Find cliques in genes regulated by the same TF. This reveals TF regulatory modules disrupted in ALS.

### Case vs. Control
**Case:** Patients with the disease (ALS in our dataset)
**Control:** Healthy individuals

**Sample ID format:**
- `CASE-NEUVM674HUA-5257-T_P003` - ALS patient, cohort 5257
- `CTRL-NEUEU392AE8-5190-T_P001` - Healthy control, cohort 5190

**Why it matters for us:** Differential analysis will compare CASE vs. CTRL to find genes/networks changed in disease.

---

## Statistical Concepts

### Outliers in Omics Data
**Why they occur:**
1. **Technical artifacts:** Sample prep, instrument issues
2. **Biological outliers:** Actual extreme biology (rare but real)
3. **Batch effects:** Different samples processed at different times

**Why we care:**
- Ruin statistical tests (especially parametric ones)
- Distort correlations (one outlier can create false edge in network)
- Mess up visualizations (scatterplot squished to tiny range)

**Why we impute instead of remove:**
- Removing would create missing values
- Missing values complicate downstream analysis (can't compute correlation with NaN)
- Imputation preserves matrix structure while fixing the problem

**Our approach:**
1. Detect outliers (MAD-Z robust method)
2. Impute with KNN (borrow information from similar genes)
3. Track which values were imputed (quality flags)

### MAD (Median Absolute Deviation)
**Formula:** MAD = median(|x - median(x)|)

**Why better than standard deviation for outlier detection:**
- Standard deviation is itself affected by outliers (circular problem)
- Median is robust (50% of data would need to be outliers to fool it)

**MAD-Z Score:**
```
Modified Z = 0.6745 × (x - median) / MAD
```
- 0.6745 factor makes it comparable to standard Z under normality
- Threshold of 3.5 means "3.5 MADs away from median"

**Engineering analogy:** MAD is like using median response time instead of mean - not fooled by occasional 10-second requests.

### KNN Imputation
**Idea:** To impute gene G in sample S:
1. Find k most similar genes to G (by Euclidean distance across all samples)
2. Take the median of those k genes' values in sample S
3. Use that as the imputed value

**Why it works:**
- Similar genes (neighbors) likely have similar values
- Median of neighbors is a good guess for what the real value should be
- Better than just using overall gene median (context-aware)

**k=5 is typical default:** Balance between local specificity and robustness

**Engineering analogy:** Collaborative filtering (Netflix recommendations) - if you like the same movies as 5 other users, you'll probably like movies they liked that you haven't seen.

### Normalization
**Why needed:** Raw counts are affected by technical factors
- Library size (some samples just have more total RNA)
- Gene length (longer genes have more reads)
- GC content (affects sequencing efficiency)

**Common methods:**
- **TPM/FPKM:** Transcripts per million (normalize by length and library size)
- **TMM:** Trimmed mean of M-values (robust normalization)
- **DESeq2 size factors:** Account for library size differences

**When to apply:** Usually before statistical tests, but NOT before outlier detection (outliers should be detected in raw space)

---

## Network Biology

### Co-expression Networks
**Construction:**
1. Compute pairwise correlation between all genes
2. Create graph: nodes = genes, edges = high correlation
3. Threshold: only keep edges with |correlation| > 0.7 (example)

**Result:** Graph where clusters represent co-regulated gene modules

**Clique finding:** Enumerate all fully-connected subgraphs
- Computationally expensive (NP-complete)
- NetworkX has efficient algorithms (Bron-Kerbosch)

### TF Regulatory Networks
**Structure:** Bipartite or layered graph
- TFs (regulators) on one side
- Target genes (regulated) on other side
- Edges = "TF binds and regulates this gene"

**Data sources:**
- ChIP-seq experiments (direct binding evidence)
- Motif scanning (predicted binding sites)
- Literature curation (databases like ENCODE)

**Our analysis:** For each TF, find cliques in its target genes
- If targets form cliques, they're co-regulated by that TF
- Disrupted cliques in ALS → dysregulated TF module

---

## ALS Biology Context

### ALS (Amyotrophic Lateral Sclerosis)
**What it is:** Neurodegenerative disease affecting motor neurons
- Progressive muscle weakness
- No cure
- Usually fatal within 2-5 years

**Why proteomic analysis:**
- Understand molecular mechanisms
- Identify drug targets
- Find biomarkers for early detection/prognosis

**Our contribution:**
- Identify TF regulatory modules disrupted in ALS
- These modules may be:
  - Causal (drive disease)
  - Protective (compensatory response)
  - Biomarkers (early indicators)

---

## Data Quality Concepts

### Batch Effects
**Problem:** Samples processed at different times/places show systematic differences unrelated to biology

**Example:**
- Cohort 1-3 processed in January
- Cohort 4-6 processed in March
- Different technician, different reagent lot → systematic shift

**Detection:** PCA plot shows clustering by batch rather than phenotype

**Correction:** ComBat, Harmony (linear models that remove batch variance)

### Missing Values
**Types:**
1. **MCAR:** Missing completely at random (rare, benign)
2. **MAR:** Missing at random given observed data (common)
3. **MNAR:** Missing not at random (e.g., value too low to detect)

**Why our dataset has none:** Already filtered to well-detected proteins

**Why outliers are like missing:** Both disrupt downstream analysis, but outliers are worse (they're wrong rather than absent)

### Quality Metrics
**For proteomics:**
- **Coverage:** What % of proteome detected?
- **Dynamic range:** Can we detect both abundant and rare proteins?
- **Reproducibility:** Technical replicates should be highly correlated

**Our quality flags encode this:** LOW_CONFIDENCE flag for measurements near detection limit

---

## Computational Patterns

### Why These Abstractions Matter

**BioMatrix = DataFrame + NumPy array + metadata**
- Most computational biology is matrix operations
- But we need to keep track of what rows/columns mean
- And quality/provenance metadata
- **Anti-pattern:** Separate data/metadata files that get out of sync

**Transform = Functional operation**
- Omics pipelines are long sequences of operations
- Each step can fail or need parameter tuning
- Immutability enables:
  - Rollback (try different parameters)
  - Parallelization (no race conditions)
  - Reproducibility (no hidden state)
- **Anti-pattern:** Scripts that modify CSV files in place (can't undo)

**Quality flags = Audit trail**
- Reviewers will ask "which values were imputed?"
- Need to prove analysis validity
- Some tests should exclude imputed values
- **Anti-pattern:** Impute and forget (no way to check later)

---

## Key Principles

1. **Biology is noisy:** Expect high variance, outliers, batch effects. Robust methods always.

2. **Metadata is as important as data:** A matrix without sample labels is useless. Keep them together.

3. **Provenance matters:** Science requires reproducibility. Track everything.

4. **Domain expertise in code:** Encode best practices (e.g., log-transform counts) as defaults, not user responsibility.

5. **Validate with biology:** Statistical significance ≠ biological relevance. Collaborate closely with domain expert.

---

## Questions to Ask

When designing any new operation:

1. **What is the input shape?** (genes × samples? samples × genes? square matrix?)
2. **What are the assumptions?** (normality? independence? positive values?)
3. **How should missing values be handled?** (error? impute? skip?)
4. **Does it make sense to apply per-gene? per-sample? globally?**
5. **What metadata is needed?** (phenotypes? batch? technical covariates?)
6. **What should be tracked in quality flags?** (is this operation lossy?)
7. **What would a biologist expect as default parameters?** (ask collaborator!)
8. **How does it scale?** (10K samples? 100K genes?)

---

## Further Reading

**For engineers entering computational biology:**
- "Computational Biology: A Practical Introduction" (Shasha & Lazzeroni)
- "RNA-seqlopedia" - https://rnaseq.uoregon.edu/
- "StatQuest" YouTube channel (Josh Starmer) - excellent intuition-building

**Papers on methods we're using:**
- MAD for outlier detection: Leys et al. (2013)
- KNN imputation: Troyanskaya et al. (2001)
- Clique finding: Bron-Kerbosch algorithm

**Databases:**
- Ensembl (gene annotations): https://ensembl.org
- ENCODE (TF binding data): https://www.encodeproject.org
- STRING (protein interactions): https://string-db.org
