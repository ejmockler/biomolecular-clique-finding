# C9-ALS proteome triangulation: a regulatory-network-aware re-analysis of AnswerALS

## Summary

**On the primary $\log_2(x+1)$ analysis, eight of eight fixed pathway terms pass the eightfold reporting threshold in C9-vs-sporadic, six of eight pass in C9-vs-healthy, and zero of eight pass in sporadic-vs-healthy.** The six-term intersection covers mRNA splicing, chromatin organization, and nucleocytoplasmic transport. The terms were discovery-derived on this same cohort and then fixed before the July method-transfer reruns, so 8/6/0 is a same-cohort consistency pattern, not independent confirmation. It supports a C9-carrier-associated pathway pattern within these samples rather than mutation causality.

We re-analyzed the AnswerALS peripheral-blood proteomics cohort to ask whether C9-carrier status is associated with a pathway-level pattern distinct from sporadic ALS. The analysis layers a literature-derived map of regulatory relationships on top of the protein-by-protein comparison and asks whether each contrast's differential magnitude concentrates around particular anchors under undirected graph distance.

In the regulatory-neighborhood analysis, all eight fixed terms pass the eightfold reporting threshold in C9-vs-sporadic, six pass in C9-vs-healthy, and none pass in sporadic-vs-healthy. A newly rerun graph-independent size-matched null using the same log2, Sex-adjusted empirical-Bayes statistics gives 8/8/0 as well. These are complementary within-dataset analyses sharing samples, statistics, and discovery-derived term definitions; they are not statistical replication.

The bounded 8/6/0 pattern attenuates to 6/0/0 in the measured-only unbounded rerun, showing sensitivity to the local depth bound. That comparison does not by itself identify a deep-shell mechanism.

Two inherited May claims are withdrawn. The STRING artifact reverses the production slope convention and has no durable derivation, so it cannot support a regulatory-versus-physical conclusion. The matched-RNA artifact reports an impossible shared-donor denominator, reuses a legacy distance matrix, and uses an unmoderated RNA statistic, so no cross-modality or post-transcriptional conclusion is licensed.

## The question

Standard proteomics on a disease cohort typically returns hundreds of proteins that differ between groups, each by a modest amount, with no single one crossing strict significance after correcting for the many tests done in parallel. The footprint of the disease is *somewhere* in this list, but reading it directly from per-protein numbers is hard.

The approach here brings outside information into the analysis. If differential magnitudes are organized around particular biological processes, nearby proteins in a regulatory-relationship map may form a spatial pattern even when no single protein clears a strict discovery threshold. We test that undirected proximity pattern.

The benefit is statistical power even when no single protein is individually significant. The cost is that the answer is conditional on the regulatory map being faithful and on the spatial-clustering picture being a reasonable description of the biology. Most of this report is organized around testing those conditions empirically.

## The cohort

The AnswerALS matrix comprises **3,264 measured rows across 436 peripheral blood mononuclear cell (PBMC) samples** collected from living donors via venipuncture: 3,263 human UniProt features plus one internal retention-time standard (`1/iRT_protein`). Metadata intersection retains 423 donors. The three primary arms below contain 410; 13 other-mutation cases are outside them. (Because the tissue is blood from living donors rather than postmortem brain, postmortem interval does not apply here.)

- **C9-ALS** — 25 donors carrying a confirmed C9orf72 mutation or a hexanucleotide-repeat expansion of length ≥ 30
- **Sporadic ALS** — 294 donors with ALS but no known causal mutation
- **Healthy** — 91 donors without ALS or related neurodegenerative diagnosis

The imbalance between groups is biologically natural (C9 is a rare mutation, sporadic dominates clinical ALS) and shapes what statistical claims we can make at the individual-protein level.

## How each protein differs between groups

For each of the three pairwise comparisons — C9 vs sporadic, C9 vs healthy, sporadic vs healthy — we compute one number per protein. This number summarizes how reliably the protein's abundance separates the two groups, after adjusting for sex. It accounts for both the size of the difference and how variable the measurement is across donors; small differences in stably-measured proteins can be more reliable than large differences in noisy ones. The estimate is stabilized by borrowing information across proteins so that any one protein's variance estimate isn't dominated by chance.

We then take the *absolute value* of this number — high values mean strongly differential, regardless of whether the protein goes up or down in the case group. We work with absolute values because the question is whether an anchor's nearby regulatory partners are loud, not which direction abundance moves.

> **Operationally.** Per protein, we first transform abundance as $\log_2(\mathrm{intensity}+1)$ and then fit `log2_intensity ~ group + sex`. Limma-style empirical Bayes shrinkage of the residual variances produces the moderated $t$ used throughout; the fitted prior degrees of freedom are $d_0=4.98$ for C9-vs-sporadic, $5.51$ for C9-vs-healthy, and $4.92$ for sporadic-vs-healthy (approximately 5 in every contrast). Variance shrinkage is what "borrows information across proteins": it prevents a protein with a freakishly small sample variance from spuriously dominating the ranking. In the codebase, the ROAST engine (Wu et al. 2010) supplies this moderated-$t$ computation, but ROAST's set-level rotation null is **not** applied to the cluster terms; pathway-level inference comes from the slope-GSEA pipeline below. The executed production analysis, all reported pass counts, and any descriptive rankings in this report use the $\log_2(x+1)$ moderated-$|t|$ scale.

Everything to this point is conventional differential-abundance modeling. No graph information has entered.

## The regulatory map

We use **INDRA** (Integrated Network and Dynamical Reasoning Assembler), a public resource that extracts claims of biological causation from the scientific literature. INDRA's CoGEx (Context Graph Extension) organizes these claims into a graph where each node is a biological entity and each edge is a published assertion that one protein regulates another in a specific way.

For this analysis we keep four kinds of regulatory assertions: X activates Y, inhibits Y, increases the amount of Y, or decreases the amount of Y. We exclude co-mention and complex-membership edges because they do not assert a regulatory relationship. The assertions retain direction in the source graph, but the distance calculation deliberately symmetrizes them.

The directional regulatory subgraph linking pairs of measured proteins contains roughly **129,000 edges**. A typical measured protein has 18 directly-connected regulatory neighbors that are themselves measured; well-studied hubs can have several hundred.

> **Operationally.** Edges are kept if they assert one of the four directional types AND carry at least one supporting evidence statement in INDRA's literature corpus; very-low-confidence edges per INDRA's belief-score noise model are filtered. The graph is *built* directional (`A→B` distinct from `B→A`), but the distance metric below treats it as undirected adjacency. Consequently, rings mix regulators and regulatees and do not trace causal flow.

## Concentration near a candidate anchor

Pick any protein in the regulatory map and look outward. Its *direct regulatory neighbors* (one edge away) form a small ring. Its *neighbors-of-neighbors* (two edges away) form a larger ring around the first.

Now ask: across both rings, does the strength of the differential signal *fall off* with distance from the candidate? If it does — strongest near the candidate, weaker as you move outward — the candidate is the kind of "anchor" the analysis is looking for. The candidate's regulatory neighborhood is unusually loud relative to the surrounding proteome.

Concretely, for each candidate anchor we fit a line through the data points (one per ring) describing how the average strength of differential signal changes with distance. A negative slope says the perturbation falls off with undirected regulatory proximity; a flat slope says the anchor is not distinguished from its surrounding region. We call this number the **slope** for the anchor.

> **Operationally.** "Distance" is the number of hops in a breadth-first search through the measured-only adjacency; each protein's distance to the anchor is its minimum hop count. The shell statistic at each distance is the mean of `|t|` across proteins in that shell. The slope is a weighted-least-squares fit of `mean(|t|)` against hop number, weighted by shell size (proportional to inverse variance under CLT). With our chosen depth of two, the slope reduces algebraically to `mean(|t|)_ring2 − mean(|t|)_ring1` — the regression machinery is exact for two points and generalizes if depth changes.

### What counts as the second ring

A subtle but consequential design decision is what counts as "two regulatory steps from the anchor." We require the intermediate to be measured. A—unmeasured X—C is *not* a valid two-hop path in this analysis, regardless of the source-edge arrow directions.

The reasoning is that "two hops from A" should describe two regulatory-relationship edges through a measured intermediate. Because distance is undirected, this is a measured regulatory-neighborhood proximity rule, not a directional cascade.

A useful empirical consequence: each anchor's reachable second-ring set now varies (median around 2,000 measured proteins, range from 0 to nearly 3,000), instead of being saturated at essentially the full proteome for every anchor. That anchor-by-anchor variation is what gives the slope its discriminating power.

### Why we stop at two regulatory steps

There's a temptation to push the rings further. Among anchors shared by the bounded and unbounded runs, depth two captures about **55.0% on average** of eventually reachable measured proteins (median **60.85%**). So why not continue deeper?

The reason isn't compute. It's that the slope statistic stops asking a coherent question.

At one step, the ring contains the anchor's direct regulatory partners in either arrow direction. At two steps, it contains partners connected through one measured intermediate, again without enforcing arrow direction. At greater depths, reachable sets expand and anchors in the same component increasingly share deep rings.

So as the rings get deeper, the statistic shifts from a local regulatory-proximity contrast toward a comparison between the anchor's local neighborhood and a large, mostly shared component background.

The current empirical comparison is narrower: the fixed-term pass pattern changes from bounded 8/6/0 to unbounded 6/0/0. This demonstrates depth sensitivity. It does not prove that all deeper rings are noise or that depth two is universally optimal; it supports describing the present result as local to the chosen $h\le2$ statistic.

## What counts as concentrated

Each anchor also receives a degree-binned permutation diagnostic.

To construct that comparison, we repeatedly reshuffle the differential-strength values across proteins and recompute the slope. Critically, we don't shuffle uniformly: we shuffle only within groups of proteins that have similar regulatory connectedness. This way, a hub stays where a hub would be after shuffling, and a sparsely-connected protein stays sparse. This matters because hub proteins are systematically closer to any anchor in the regulatory map; failing to preserve this would generate spurious "enrichment" of hub-heavy neighborhoods.

We generate 999 shuffled slopes per anchor and report how often a shuffle produces a slope as extreme as the real one. The smallest reportable diagnostic p-value is one in a thousand.

> **Operationally.** Proteins are sorted by INDRA degree into roughly 100-feature chunks (the final bin may be smaller). For each of 999 permutations, the $|t|$ labels are shuffled without replacement within each fixed bin and the slope is recomputed. The empirical p-value is `(# permuted slopes ≤ observed + 1) / (999 + 1)`, giving a strict floor of 1/1000.

This diagnostic is not an inferential input to pathway GSEA: the pathway code ranks observed `−slope` and never consumes the per-anchor p-value. A separate canonical pathway-level degree-matched null is reported below.

## Running every protein as a candidate

Rather than pre-selecting anchors, we attempt the analysis around all 3,264 measured features. The bounded, measured-only run produces **3,117 valid two-shell gradients** per comparison. Of the 147 exclusions, **137** have no reachable measured neighbor (`DisconnectedFeature`) and **10** fail the pre-specified guardrail requiring at least 10 measurable genes in the neighborhood; the iRT standard is among the disconnected rows. The primary `robust` GSEA then retains **1,407** valid anchors with at least 20 measured hop-1 neighbors. Those 1,407, not all 3,117, form the primary ranking.

A complete pass over the proteome takes about 90 minutes on a single multi-core machine.

## Translating slopes into biology

We then ask which pathways are over-represented at the most-concentrated-perturbation end of the anchor list. A pathway in this sense is a curated list of genes thought to work together on a particular biological function — coming, for example, from the Gene Ontology, Reactome, WikiPathways, or the Human Phenotype Ontology. For each pathway in each database we get an enrichment score and a multiple-testing-corrected significance value.

We use four pathway annotation databases. Their terms overlap and all consume the same ranking, so cross-database recurrence is descriptive concordance, not independent replication.

> **Operationally.** We use `gseapy.prerank` with 1000 permutations, ranking anchors by `−slope` so that most-negative slopes rank highest (= "most concentrated perturbation"). Pathway terms are scored with the magnitude-weighted enrichment statistic (`weighted_score_type=1`); the GSEA NES-histogram FDR (gseapy/Subramanian) is applied within each database separately. The "raw p" below is the GSEA permutation p before the eightfold reporting threshold.

## Triangulating across comparisons

We ran the entire pipeline three times, once for each pairwise group comparison. The three pass/fail patterns across those comparisons map to four possible interpretations:

| C9 vs sporadic | C9 vs healthy | Sporadic vs healthy | Pattern consistent with |
|:---:|:---:|:---:|---|
| ✓ | ✓ | — | **C9-carrier-associated within this cohort** |
| — | ✓ | ✓ | shared ALS pathology |
| ✓ | ✓ | ✓ | uniformly recurring fixed-graph pattern |
| — | — | ✓ | sporadic-ALS-specific |

The observed first-row pattern argues against a shared sporadic-ALS pattern and a uniformly recurring fixed-graph pattern. It does not prove mutation causality or exclude every graph-by-contrast interaction.

## The eight pathway terms put to the test

Discovery on this proteomics dataset identified eight pathway terms that recurrently came up in the network-aware analysis as cluster anchors. They group into three biological themes.

**These terms were identified through earlier discovery on the same data**, so the rerun is *consistency-checking*, not independent confirmation. We apply an eightfold multiplicity threshold: each term must have raw p<0.00625 and positive NES. Because selection was outcome-guided on this cohort, this is not a valid post-selection FWER guarantee.

| Theme | Source | Term |
|---|---|---|
| Nuclear pore / transport | GO | nuclear pore (GO:0005643) |
| Nuclear pore / transport | GO | nucleocytoplasmic transport (GO:0006913) |
| Nuclear pore / transport | Reactome | Vpr-mediated nuclear import of PICs (R-HSA-180910) |
| Splicing | Reactome | mRNA Splicing (R-HSA-72172) |
| Splicing | Reactome | Processing of Capped Intron-Containing Pre-mRNA (R-HSA-72203) |
| Splicing | GO | mRNA splicing, via spliceosome (GO:0000398) |
| Chromatin | GO | chromosome (GO:0005694) |
| Chromatin | GO | chromatin (GO:0000785) |

The eight terms overlap within biological themes. The eightfold threshold does not make them independent or repair their original same-data selection.

Two complementary canonical auxiliaries challenge narrower artifacts. A graph-independent 10,000-draw size-matched HGNC null tests set-size inflation in mean moderated $t^2$. A pathway-level degree-matched null tests whether mean observed `−slope` exceeds matched anchors on the same 1,407-anchor robust scope. Both reuse the cohort, statistics, and discovery-derived terms; neither is independent replication.

## At the protein level
*Do the cluster proteins, taken as a set, carry more differential signal than random gene sets of the same size?*

In the canonical rerun, all eight terms pass Bonferroni-8 in both C9 comparisons. C9-vs-sporadic raw p-values are all 0.00009999; C9-vs-healthy has seven at 0.00009999 and the GO spliceosome term at 0.00019998.

In sporadic-vs-healthy, zero of eight pass; raw p-values range from 0.14659 to 0.76432.

This 8/8/0 auxiliary shows that the fixed sets' mean moderated $t^2$ exceeds same-sized random HGNC sets under the canonical statistic. It is a same-data, graph-independent corroboration, not independent confirmation.

## In the regulatory neighborhoods
*Do the fixed pathway terms enrich toward the concentrated end of the primary slope ranking?*

The eightfold-threshold outcome on the fixed eight terms:

| Comparison | Terms passing | Detail |
|---|---|---|
| **C9 vs sporadic** | **8 of 8** | all fixed terms pass on the primary $\log_2(x+1)$ analysis |
| **C9 vs healthy** | **6 of 8** | nuclear pore and Vpr-mediated nuclear import do not clear the eightfold threshold |
| **Sporadic vs healthy** | **0 of 8** | no fixed term clears the eightfold threshold |

**Six terms pass the threshold in both C9 comparisons and miss in sporadic-vs-healthy:** mRNA Splicing, Processing of Capped Intron-Containing Pre-mRNA, mRNA splicing via spliceosome, chromosome, chromatin, and nucleocytoplasmic transport. These six form the cross-contrast core of the pathway-level claim. Nuclear pore and Vpr-mediated nuclear import both pass in C9-vs-sporadic but not in C9-vs-healthy.

**Observed pattern: ✓ ✓ —.** This supports a C9-carrier-associated within-cohort reading and argues against a uniformly recurring fixed-graph pattern. It does not exclude every graph-by-contrast interaction.

### Canonical pathway-level degree-matched auxiliary

On the same 1,407-anchor robust scope, a repeated degree-matched null compares each term's mean observed `−slope` with same-size sets sampled from matched degree bins. Seven of eight terms pass Bonferroni-8 in each C9 contrast and zero of eight pass in sporadic-vs-healthy. Vpr-mediated nuclear import is the sole C9 non-pass (C9-vs-sporadic raw p=0.0231; C9-vs-healthy p=0.0397). This 7/7/0 result supports the six-term cross-contrast core plus nuclear pore; it must not be reported as all-eight $p<0.001$ protection.

## The three biological clusters

**Splicing and pre-mRNA processing.** Cluster members span spliceosomal proteins — the SR-family splicing factors (SRSF1, SRSF2, SRSF5, SRSF7, SRSF9), U2-snRNP components (U2AF2, SF3B1, SF3B3, SF3B4), pre-mRNA processing factors (PRPF3, PRPF8, PRPF19), and heterogeneous nuclear ribonucleoproteins (the HNRNP family). These names describe pathway membership, not statistically discovered individual-protein hits; the present analysis licenses only the pathway-level pattern. The grouping is consistent with established RNA-binding-protein sequestration biology in C9-ALS, but the slope itself does not establish that mechanism.

**Chromosome and chromatin organization.** Cluster members include chromatin-associated proteins (MBD3, RBBP4, RBBP7, TRRAP, CBX5), DNA replication and repair components (MCM4, MCM5, MCM7), and cell-cycle proteins (RAD21, MAD2L1, RCC1). Again, these are pathway annotations rather than individual-protein discoveries. A mechanistic link between C9 and chromatin organization is less established than the links to splicing or nuclear pore. Scattered reports describe altered nucleosome positioning around C9 repeats, R-loop-mediated DNA damage response activation, and disrupted Polycomb repressive complex function in C9 patient iPSC-derived neurons, but no consensus mechanism has emerged. This signature is therefore the most novel of the three and the one we would prioritize for independent-cohort replication.

**Nuclear pore complex and nucleocytoplasmic transport.** Cluster members include structural nucleoporins (NUP35, NUP54, NUP62, NUP85, NUP88, NUP93, NUP107, NUP133, NUP160, NUP188, NUP205) and nuclear transport receptors (importins KPNA1 and IPO4, exportin XPO7, RAE1, RANBP1). This aligns with the established nuclear-pore-disruption mechanism in C9-ALS, attributed to dipeptide-repeat protein toxicity — particularly poly-PR and poly-GR — which disrupts nucleoporin function and impairs nuclear import in postmortem motor cortex and patient-derived neurons. The present pathway pattern in PBMCs is consistent with that related biological theme; it does not establish the mechanism or tissue extension.

The three labels describe distinct biological themes, but their exact gene-set overlap depends on the frozen database membership and should be reported from that snapshot rather than summarized by an unsourced percentage.

## Withdrawn auxiliary comparisons

The legacy STRING comparison is withdrawn, not interpreted. Its report defines `slope = hop1 − hop2` while production uses `hop2 − hop1`, then negates the slope for GSEA. That reversal can manufacture the reported negative NES pattern, and no durable runnable result accompanies it. A same-orientation log2 measured-only rerun is required before any regulatory-versus-physical claim.

The legacy matched-RNA comparison is also withdrawn. It claims 463 donors shared by both modalities despite the proteomics matrix containing only 436 measured and 423 metadata-matched donors; it reuses a with-intermediates distance matrix and applies an unmoderated RNA statistic. It licenses no same-donor discordance, molecular-layer localization, or post-transcriptional mechanism.

## What we'd defend, what we wouldn't yet

**Defensible from the present analysis:**

- The bounded $\log_2(x+1)$, measured-only fixed-term GSEA produces **8/6/0** on a 1,407-anchor robust ranking. Six terms pass in both C9 contrasts and not sporadic-vs-healthy.
- A canonical graph-independent, size-matched mean-$t^2$ null gives **8/8/0** after Bonferroni-8.
- A canonical pathway-level degree-matched mean-`−slope` null on the same robust scope gives **7/7/0**; Vpr-mediated nuclear import is the sole C9 non-pass.
- The unbounded log2 measured-only sensitivity gives **6/0/0**, so the fixed-term pattern attenuates when the local depth bound is removed.
- Together these support a C9-carrier-associated pathway pattern within this cohort. They do not establish mutation causality, independent replication, STRING specificity, RNA localization, or an age-adjusted result.

**Hypotheses we would not yet defend:**

- Sub-pathology structure within sporadic ALS. The diffuse-signal finding could reflect a mixture of sub-clinical groupings, but we have no within-dataset test that would resolve this.

**What the analysis cannot tell us:**

- Per-anchor p-values have a one-in-a-thousand floor and are diagnostics only; they are not used by GSEA and license no individual-protein discovery.
- External replication in a separate C9-ALS proteomics cohort remains the publication gate.
- No edge-level wiring claim is made. WASC calibration gave mean Jaccard 0.285 versus a required 0.70 after Sporadic was downsampled 294→25, so the hard halt blocked the primary run; that is pipeline-specific, not a universal $n=25$ power claim.
- Age remains unresolved. The May artifact used raw-scale incremental $\Delta R^2$, not partial $R^2$, and no canonical age-adjusted slope/GSEA rerun exists.
- STRING and matched RNA remain withdrawn for the method and accounting defects stated above.
- The literature-derived regulatory edges in INDRA inherit publication biases: well-studied genes have more documented connections, which may interact with pathway-database gene-set membership in ways that are not always easy to predict.

## Next experiments

In rough priority order:

1. **Separate-cohort replication** of the cluster claim on external C9-ALS proteomics. This is the publication gate.
2. **Sub-pathology stratification of sporadic ALS** by clinical features (limb vs bulbar onset, age-of-onset bins, survival quantiles), to test whether the diffuse sporadic signal in fact comprises structured sub-signals that average out. Requires careful pre-registration to avoid confounders associated with sporadic-ALS stratification.
3. **Optional deeper per-anchor permutation** for the two C9 comparisons. This is separate from, and does not alter, the present pathway-level claim.
4. **Audited auxiliary reruns**: rebuild STRING with the production slope orientation and rerun matched RNA only after reconciling the true same-donor cohort and statistical frame.
5. **An explicit WASC decision**: report the pipeline-specific calibration halt, specify a redesigned partial-pooling analysis, or acquire a larger C9 cohort before revisiting edge-level inference.

## Where the results live

Full local landscape and GSEA artifacts live under `output/`; the durable publication contract and auxiliary snapshots live under `data/publication/`. Landscape runs carry design metadata and input accounting. The historical GSEA runs queried live CoGEx and did not embed the complete upstream corpus, producer package revisions, or effective seed/default in their outputs, so their local CSVs can be integrity-checked but exact upstream regeneration is not yet guaranteed. The two current auxiliaries freeze their mappings/memberships and generator/input hashes explicitly.

---

*Report prepared for AnswerALS collaborators. Methods and code: `scripts/run_landscape_*.py` and `src/cliquefinder/`.*
