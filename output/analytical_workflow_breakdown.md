# A Methods Deep-Dive: Finding a C9orf72 Pathway Fingerprint in ALS Blood Proteomics

This is a companion walkthrough to the Methods. Where the Methods states each step once and tersely, this document *teaches* the pipeline: what question each stage answers, which risks it diagnoses, why the naive first attempt fails, and how the failure reshaped the whole study. It is meant to be read on its own by a statistically literate reader who has never seen this analysis.

A one-paragraph map of the argument, so the reader knows where the narrative is going. We reduce each measured feature to a Sex-adjusted, empirical-Bayes moderated $|t|$, form local two-shell gradients on a measured-only regulatory graph, and rank the robust subset of those gradients for pathway enrichment. We run the same July 2026 log2 pipeline across three contrasts and compare the bounded 8/6/0 fixed-term pattern with the unbounded 6/0/0 sensitivity. Degree-binned permutations are per-anchor diagnostics, not inputs to pathway GSEA. The canonically rerun size-matched and pathway-level degree-matched nulls are current auxiliaries; legacy versions, abundance, STRING, RNA, age, and five-analysis artifacts are explicitly separated from them.

The authoritative state is analysis `c9-als-fingerprint-log2-measured-only-h2-2026-07`, recorded 12 July 2026 in `data/publication/c9_primary_analysis.json`; retained GSEA integrity and its regeneration limits are recorded in `data/publication/c9_gsea_provenance.json`. If this walkthrough conflicts with either record, the machine-readable record controls.

## The question, stated so it can be falsified

The *C9orf72* hexanucleotide-repeat expansion is the most common single-gene cause of amyotrophic lateral sclerosis. We want to know whether carrying that expansion leaves a **pathway-level fingerprint** in the proteome of peripheral blood mononuclear cells (PBMCs) — a fingerprint that *sporadic* ALS does not leave.

Any signal we find has several competing readings:

1. **C9-carrier-associated in this cohort.** The signal appears in both C9 contrasts but not Sporadic-vs-Control.
2. **Shared ALS pathology.** The signal reflects motor-neuron disease in general and would appear in sporadic patients too.
3. **A graph-by-data interaction.** The fixed graph and contrast-specific magnitudes jointly manufacture the pattern.

The three contrasts argue against a uniformly recurring fixed-graph artifact and against a shared sporadic-ALS pattern. They do not prove mutation causality or exclude every graph-by-data interaction.

## Stage 1 — The cohort and the data matrix

The input is a PBMC proteomics matrix of **3,264 measured rows × 436 samples**. The rows are 3,263 human UniProt features plus one internal retention-time standard (`1/iRT_protein`). The data provider has already batch-corrected the intensities, imputed missing values by random forest, and rolled peptides up to protein-level abundances, so the matrix arrives **complete** — no missing cells.

After intersecting the proteomics with donor metadata, **423 donors** remain. The three primary arms contain **410**; 13 other-mutation cases sit outside them:

| Group | n | Role |
|---|---|---|
| C9 (expansion carriers) | 25 | the perturbation of interest |
| Sporadic ALS | 294 | disease without the mutation |
| Control | 91 | neither |

Operationally, C9 includes donors whose mutation detail names C9orf72 or whose
repeat length is at least 30. Sporadic requires CASE phenotype, excludes the
mutation labels `C9orf72`, `SOD1`, `FUS`, `TARDBP`, `TARDBP (TDP43)`, `SETX`,
`Multiple`, and `Other`, and has repeat length below 30 or missing. Control is
the CTRL phenotype arm.

The three-arm structure is not incidental; it is the instrument. C9-vs-Sporadic compares carrier status within ALS; C9-vs-Control combines carrier and disease-status differences; Sporadic-vs-Control compares ALS without a known causal mutation against controls. We read these jointly without treating any observational contrast as a causal intervention.

**Covariate.** The only nuisance variable modeled is **Sex**, and it earns its place empirically: Sporadic is 212/294 (72.1%) male, versus 47/91 (51.6%) in Control and 17/25 (68.0%) in C9. An uncorrected comparison would partly confound sex-linked abundance with group. Age is not in the canonical model and has not been resolved by a canonical age-adjusted rerun.

**Intensity scale.** The canonical primary, depth sensitivity, and two current auxiliary analyses operate on the log-transformed scale

$$\tilde{x}_{ij} = \log_2(x_{ij} + 1),$$

for feature $i$ and donor $j$. The $\log_2$ compresses the heavy right tail typical of intensity data so that variances are comparable across the dynamic range, and the $+1$ absorbs exact zeros without discarding them. This is the scale of the July bounded/unbounded analyses; historical diagnostics below are labeled rather than implied to share it.

All **3,264 measured features** are attempted as anchors in the bounded, measured-only gradient analysis. Of those, **3,117 yield valid two-shell gradients**. The remaining 147 fail for one of two recorded reasons: **137** have no reachable measured neighbor (`DisconnectedFeature`), and **10** fall below the implemented guardrail requiring at least 10 measurable genes in the neighborhood. The iRT standard is among the disconnected rows. These are network-statistic exclusions, not missing-intensity exclusions.

## Stage 2 — One number per protein: the moderated $|t|$

Every stage of the primary landscape/GSEA pipeline consumes a single per-protein quantity: how strongly a protein's abundance shifts between two groups, measured in standard errors. Fix one protein $i$ and one of the three pairwise contrasts, so we are comparing two arms holding $n$ donors between them. Gather that protein's log2 abundances across those $n$ donors into a column vector $\tilde{x}_i \in \mathbb{R}^{n}$, and model each donor's value as a baseline plus a group effect plus a sex effect:

$$\tilde{x}_i = X\beta_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2 I_n).$$

The **design matrix** $X$ is what turns "baseline plus group plus sex" into linear algebra. It has one row per donor and one column per model term — an all-ones column for the intercept, a 0/1 indicator $d_{\text{group}}$ of which arm each donor is in, and a 0/1 indicator $d_{\text{Sex}}$ of sex — so it is $n \times 3$:

$$X = [\,\mathbf{1} \mid d_{\text{group}} \mid d_{\text{Sex}}\,].$$

Reading the matrix equation one donor $j$ at a time recovers an ordinary regression line,

$$\tilde{x}_{ij} = \beta_{i,0} + \beta_{i,\text{group}}\, d_{\text{group},j} + \beta_{i,\text{Sex}}\, d_{\text{Sex},j} + \varepsilon_{ij},$$

so the coefficient vector $\beta_i = (\beta_{i,0},\, \beta_{i,\text{group}},\, \beta_{i,\text{Sex}})^\top$ has one entry per design column. The first, $\beta_{i,0}$, is the **intercept**: the subscript $0$ marks it as the coefficient of the constant all-ones column, and it equals protein $i$'s baseline log2 abundance when both indicators are $0$ — that is, in the reference arm and the reference sex. The other two entries, $\beta_{i,\text{group}}$ and $\beta_{i,\text{Sex}}$, are the additive shifts away from that baseline for the non-reference arm and the non-reference sex, and $\varepsilon_i$ is the donor-level noise. The single entry we ultimately test is $\beta_{i,\text{group}}$, the between-arm difference with sex held fixed — carrying $d_{\text{Sex}}$ in the model is precisely what "adjusting for sex" means.

The coefficients are unknown population quantities, so we estimate them from the data by ordinary least squares,

$$\hat\beta_i = (X^\top X)^{-1} X^\top \tilde{x}_i,$$

which returns the full estimated vector; its group entry $\hat\beta_{i,\text{group}}$ is the estimated log2 abundance difference between the two arms — the number the rest of the analysis is built on. How precisely that entry is pinned down is set by a single element of $(X^\top X)^{-1}$. That matrix has one row and one column per model term, so naming an element takes two labels, a row and a column; the group coefficient's variance sits on the diagonal, where the row and the column are both the group term:
$$v_i = \big[(X^\top X)^{-1}\big]_{\text{group},\text{group}}$$
(the two "group" labels are that one diagonal address, not two quantities). Concretely $\operatorname{Var}(\hat\beta_{i,\text{group}}) = \sigma_i^2\,v_i$, so $v_i$ is the purely design-driven part of the coefficient's variance — how the donors are split and balanced across group and sex — while $\sigma_i^2$ (estimated by $s_i^2$) is the noise part. A balanced, well-populated contrast makes $v_i$ small and the estimate precise.

A raw $t = \hat\beta_{i,\text{group}} / (s_i\sqrt{v_i})$ — the estimated shift divided by its estimated standard error, with $s_i^2$ the ordinary residual variance — would be dangerous here. With modest sample sizes, some proteins land a freakishly small residual variance $s_i^2$ by chance; their $t$ balloons and floats to the top of the ranking for no biological reason. The fix is the empirical-Bayes variance-moderation framework of Smyth (2004): **borrow strength across all proteins**. We estimate a prior variance $(s_0^2, d_0)$ once, by matching moments of the observed variance distribution across the ~3,200 proteins, then shrink each protein's variance toward that prior:

$$\tilde{s}_i^2 = \frac{d_0\,s_0^2 + d_i\,s_i^2}{d_0 + d_i}, \qquad t_i = \frac{\hat\beta_{i,\text{group}}}{\tilde{s}_i\sqrt{v_i}} \ \sim\ t_{\,d_0 + d_i}.$$

Here $d_i = n - 3$ is the residual degrees of freedom for protein $i$ — the $n$ donors minus the three fitted coefficients — $v_i$ is the same design-driven precision factor introduced above, and $d_0$ is the prior degrees of freedom, literally how many "pseudo-observations" of variance the prior contributes. The moderated $t_i$ is built from exactly the same estimated shift $\hat\beta_{i,\text{group}}$ as the raw $t$; only the variance in the denominator has changed, from the protein's own $s_i^2$ to the shrunken $\tilde{s}_i^2$.

The shrunken variance is a weighted average of two variance estimates, and the weights are exactly the degrees of freedom each estimate carries: the prior contributes $d_0$ pseudo-observations of variance $s_0^2$, and the data contribute $d_i$ residual degrees of freedom to $s_i^2$. Written that way, the formula is a degrees-of-freedom-weighted mean, and its limits are the intuition:

- As $d_0 \to \infty$, the shrunken variance $\tilde{s}_i^2 \to s_0^2$: infinite prior confidence collapses every protein onto a single common variance, and the moderated $t$ becomes a $z$-like statistic that depends only on the coefficient. This is maximal borrowing — every protein is assumed to share the same underlying noise, and a protein with a lucky-small sample variance is given none of that luck.
- As $d_0 \to 0$, $\tilde{s}_i^2 \to s_i^2$: the prior contributes nothing and we recover the ordinary per-protein $t$. This is no borrowing.
- The moderated statistic is referenced to $d_0+d_i$ degrees of freedom when tail probabilities are calculated. This added reference df does not itself change the ranked number: variance shrinkage changes the numerical moderated $t$, while the reference df changes its p-value distribution.

The production code implements this Smyth-style method of moments locally; it does not call the R `limma` package. The July landscape result embedded neither $d_0$ nor $s_0^2$. The retrospective primary receipt records the same three $d_0$ values; both parameters below come from the current-code log2/Sex refit retained with the canonical size-matched null, and the posterior weights are derived from them. The table is a current reference reconstruction of the local method, not a byte-level record of every July empirical-Bayes hyperparameter:

| Contrast | residual df $d_i$ | $d_0$ | $s_0^2$ | prior weight |
|---|---:|---:|---:|---:|
| C9-vs-Sporadic | 316 | 4.9847 | 0.059531 | 1.55% |
| C9-vs-Control | 113 | 5.5097 | 0.064427 | 4.65% |
| Sporadic-vs-Control | 382 | 4.9219 | 0.056273 | 1.27% |

So the prior contributes a small, reported fraction of each posterior variance. These fitted values are estimates rather than tuned settings, and they do not by themselves prove variance homogeneity or model adequacy.

Two more deliberate choices define what the downstream statistic is *about*.

**We take the magnitude $|t_i|$, not the signed $t_i$.** The gradient and enrichment stages ask where perturbation *concentrates*, not which direction abundance moves. A protein pushed hard up and a protein pushed hard down are both strongly perturbed. Taking the absolute value makes the analysis direction-agnostic and measures perturbation *magnitude*. The consequence to keep in mind: no result derived from $|t|$ can speak to direction or to activation-versus-inhibition. That scope limit is inherited by everything downstream.

**$|t_i|$, not a p-value or a fold-change.** The $t$ folds effect size and its uncertainty into one number on a common scale across proteins, which is exactly what a shell-average or an enrichment rank needs.

Having reduced each protein to one honest number, we can now ask the study's real question. The first, natural attempt to ask it fails — instructively.

## Stage 3 — What the historical target-set attempt actually establishes

This stage is design history, not canonical July evidence. Crucially, the retained files are not one pipeline whose p-values can be arranged into a clean progression.

The historical target-set branch used the provider-corrected/imputed linear-intensity matrix without a repository log transform. This scale applies to both the complete-case and matched diagnostics below.

The identified 28 March run used a different complete-case model: C9 $n=23$, Sporadic $n=282$, with Sex, age at symptom onset, and baseline ALSFRS-R as covariates. It tested a 47-protein *C9orf72* set with a **self-contained rotation test**. The current retained engine transforms moderated $t$ statistics to normal scores and then uses a mixed-direction mean-squared normal score (MSQ), but the March artifact did not embed its exact producer revision, so that implementation detail is a reference description rather than proven historical provenance:

$$p = \frac{b + 1}{B + 1}, \qquad b = \#\{\text{rotated MSQ} \ge \text{observed MSQ}\}.$$

That run reports target $p=0.0157$. But 113 of 200 uniformly sampled, same-size sets drawn without replacement from the measured non-target pool also returned self-contained $p<0.05$: **56.5%**. Those are observed-data sets, not global-null sets, and can contain genuine group effects. The percentage is therefore **not** a false-positive rate or type-I calibration estimate. It demonstrates a question mismatch: the self-contained test asks whether a set moves at all, while the competitive question asks whether it moves more than the measured background. The target rotation used evidence weights, whereas the 200 random control sets were unweighted, so this is not a like-for-like weighted target-versus-control calibration.

The retained competitive producer instead uses a random-set test. In each of $B=10{,}000$ draws, it samples a same-size set without replacement from all finite protein rows; because the pool is not target-excluded, a random set can overlap the target set. If $T_{\mathrm{obs}}$ is the target mean $|t|$ and $T_b^*$ is a random-set mean,

$$z=\frac{T_{\mathrm{obs}}-\overline{T^*}}{s(T^*)}, \qquad
p=\frac{1+\#\{T_b^*\ge T_{\mathrm{obs}}\}}{B+1}.$$

For the complete-case fit, $T_{\mathrm{obs}}=1.298$, the null mean is 1.133, the null SD is 0.111, $z=1.479$, and $p=0.075$. The same procedure in the Sex-matched 50-donor artifact gives $T_{\mathrm{obs}}=0.886$, null mean 0.919, null SD 0.089, $z=-0.380$, and $p=0.649$. Both artifacts carry VIF and mean-correlation fields equal to 1 and 0, respectively, but those fields do not enter this statistic. The cohort analyses are related diagnostics, not successive corrections to one test. Both random-set calls left their seed at the `None` default; their exact 10,000 null draws and p-values cannot be regenerated, even though the broader validation command recorded seed 42.

Earlier drafts quoted a Sex-only sequence near 0.21 → 0.69 → 0.65 and a mean-$|t|$ difference of 0.13–0.17. No identified machine-readable producer artifact for that sequence remains, so this walkthrough no longer treats it as evidence. The durable lesson is narrower: the self-contained result lacked competitive specificity, and retained competitive analyses did not establish target-set enrichment.

That lesson suggested a different, continuous hypothesis: **network distance may predict perturbation magnitude**. The July log2 measured-only analysis below is the current test of that hypothesis.

## Stage 4 — The per-protein perturbation gradient

The testable hypothesis is geometric: differential magnitude may be larger among an anchor's nearby regulatory partners than among more distant partners. Because distance is symmetrized, this is an **undirected regulatory-neighborhood proximity** hypothesis, not a model of causal propagation from a source.

Treat **every measured feature row as an anchor**, including the iRT row for complete accounting. Around a biological anchor $a$, walk outward through a regulatory network extracted from live INDRA CoGEx using `Activation`, `Inhibition`, `IncreaseAmount`, and `DecreaseAmount` statements with at least one item of evidence. A live MyGene query maps each UniProt row to symbols and aliases; distance takes the minimum finite distance over resolved alias pairs. Two properties of the walk are deliberate:

- **Distance is measured undirectedly.** We use edge *direction* to define what "regulatory" means, but we measure graph distance ignoring direction, because a target is regulatorily close to its regulator whether the arrow points in or out. The question is proximity in the regulatory wiring, not causal reachability.
- **The walk visits measured proteins only.** We never route a path through an unmeasured intermediate. A shell at distance $h$ therefore contains only proteins we actually have $|t|$ values for. This keeps every shell average an average over real measurements and prevents phantom nodes from bending the distance metric.

The retained `distances.npz` files contain the graph-derived matrices that were analyzed, while `distances.meta.json` preserves feature labels, degrees, and unmatched features. The canonical GSEA receipt hashes the metadata JSON, not the NPZ matrix bytes. The substrate is retained locally, but matrix-byte integrity is outside the receipt; the full producing CoGEx corpus/version is also absent, so exact extraction from the upstream live resource is not guaranteed.

Formally, the shell at hop distance $h$ around anchor $a$ is

$$R_h(a) = \{\, i : d(a, i) = h \,\},$$

with $d$ the undirected distance over measured proteins, and the shell's mean signal is

$$\bar{S}_h = \frac{1}{|R_h|} \sum_{i \in R_h} |t_i|.$$

The anchor's **gradient slope** is the least-squares slope of shell-mean-$|t|$ against hop distance, weighted by shell size:

$$\hat{m}_a = \frac{\sum_h w_h\,(h - \bar{h})(\bar{S}_h - \bar{S})}{\sum_h w_h\,(h - \bar{h})^2}, \qquad w_h = |R_h|,$$

where $\bar{h} = \sum_h w_h h / \sum_h w_h$ and $\bar{S} = \sum_h w_h \bar{S}_h / \sum_h w_h$ are the shell-size-weighted means of hop distance and of shell-mean-$|t|$. The weights $w_h = |R_h|$ give a shell with many proteins more say than a shell with two, because its mean is better estimated. A **negative** slope means the near shells carry higher $|t|$ than the far shells — perturbation is concentrated at the anchor and decays outward. A flat or positive slope means the anchor sits in no such gradient.

**Why depth 2, and why it is a decision rather than a limit.** The analysis is bounded at $h \le 2$. This is not because deeper shells are unreachable — they usually exist — but because of *what the statistic means* at each depth:

- At depth 2, the slope compares an anchor's direct regulatory partners (ring 1) with two-hop regulatory partners reached through a measured intermediate (ring 2). This is a local proximity question, not a directional cascade.
- Beyond depth 2, the far shells expand to engulf most of the anchor's connected component. The slope then quietly changes its meaning into "does the anchor's neighborhood differ from its whole connected component?" — a weaker, more diffuse question that is dominated by global component structure rather than local concentration.

So bounding at 2 keeps the statistic aimed at the local question. The unbounded rerun later tests how sensitive the fixed-term pattern is to removing that bound.

There is a clean simplification at depth 2 worth internalizing, because it demystifies the statistic entirely. A least-squares line through only two points passes exactly through both of them, so its slope is simply the rise over the run between them. With shells at $h = 1$ and $h = 2$ the run is $2 - 1 = 1$, and the rise is $\bar{S}_2 - \bar{S}_1$; the shell-size weights cancel because a line fit to two points is determined regardless of how the two points are weighted. The weighted-slope formula therefore collapses to

$$\hat{m}_a = \bar{S}_2 - \bar{S}_1.$$

So the depth-2 gradient is nothing more mysterious than "how much lower is the second ring's mean $|t|$ than the first ring's." Negative means the first ring is hotter than the second: local concentration. Every anchor's slope is, at bottom, that one comparison. This is not just a tidy identity; it makes the null in the next stage transparent, because we are permuting the inputs to a plain difference of two means rather than to an opaque regression.

## Stage 5 — The per-anchor degree-stratified diagnostic

Network degree is a plausible structural nuisance because high- and low-degree proteins can occupy distance shells differently. The canonical data do **not** establish the stronger claim that hubs carry larger $|t|$; degree control is therefore a conservative robustness diagnostic, not correction for an observed dominant degree–magnitude association.

For each anchor, remove that anchor from the label pool. Assign every remaining measured feature its full-INDRA count of incident regulatory relationships, taking the maximum over its resolved aliases. This is not the count of unique measured hop-1 neighbors, and its query uses the statement-type filter without reapplying the shell extraction's explicit evidence-count predicate. Sort features by degree, split the ordering into consecutive rank blocks of 100, and shuffle $|t|$ within each block. This approximately retains degree location; it is not exact degree matching, and “hub swaps only with hub” would overstate it. For $B=999$ permutations producing null slopes $\{\hat m_a^{(b)}\}$, the one-sided decay p-value is

$$p_a = \frac{1 + \#\{\hat{m}_a^{(b)} \le \hat{m}_a\}}{B + 1}.$$

Read this as: “among slopes achievable by shuffling magnitudes within similar-degree blocks, how often is the shuffled slope at least as negative as the observed one?” The $+1$ correction prevents a zero p-value. Base seed 42 plus a stable MD5-derived anchor identifier makes each anchor's shuffles repeatable. Crucially, `run_landscape_gsea.py` ranks the observed $-\hat m_a$ and never reads $p_a$. The null calibrates individual-anchor diagnostics only; it does not degree-adjust the 8/6/0 pathway inference. A separately rerun pathway-level degree-matched sensitivity gives 7/7/0 on the robust scope and is an auxiliary, not part of GSEA.

## Stage 6 — From per-protein slopes to pathways

We now hold a slope $\hat{m}_a$ and a diagnostic permutation p-value for every valid anchor. A per-anchor list is not yet a biological statement; the pathway analysis ranks the slopes themselves.

Of the 3,117 valid slopes, retain the **1,407 robust anchors** whose measured hop-1 neighborhood has at least 20 features. All 1,407 map to unique HGNC identifiers in the retained runs. Rank them by

$$r_a = -\hat{m}_a,$$

so the most sharply-decaying anchors sit at the top. For annotation set $S$, define $S'$ as its intersection with the ranked HGNC universe and $k=|S'|$. Walk down the primary list of $N=1{,}407$ anchors, accumulating a signed running score $RS_i(S)$ that adds weight at measured set members and subtracts at non-members:

$$RS_i(S) = \sum_{\substack{a \le i \\ a \in S'}} \frac{|r_a|^{\kappa}}{\sum_{a' \in S'} |r_{a'}|^{\kappa}} - \sum_{\substack{a \le i \\ a \notin S'}} \frac{1}{N - k}, \qquad \kappa = 1,$$

$$ES(S)=RS_{i^*}(S), \qquad i^*=\arg\max_i |RS_i(S)|.$$

The sign of the greatest absolute excursion is retained, so a bottom-concentrated set can have negative ES/NES. This corrects the common but incompatible shorthand $\max_i|RS_i|$, which would erase negative enrichment.

Significance comes from **gene-set (label) permutation over the fixed ranking**: repeatedly draw random gene sets and recompute $ES$ to build a null, then divide by the mean absolute ES among null scores with the same sign to obtain NES. NES reduces set-size dependence rather than placing every possible set on a guaranteed common scale. Note what is *not* permuted: we never shuffle the phenotype. The label permutation keeps the expensive $|t|$-and-slope computation fixed.

Operationally, the retained July producer calls the INDRA CoGEx wrappers around `gseapy.prerank` **separately for each database**, using 1,000 gene-set permutations, weight exponent 1, minimum matched size 1, and no top-score trimming. The current reference wrapper permits sets up to 50,000 members. Each database run produces its own nominal p-values and FDR q-values; there is no single across-database FDR family.

We run this against four annotation databases: Gene Ontology, Reactome, WikiPathways, and Human Phenotype Ontology. Their terms overlap and all consume the same samples and ranking, so recurrence across them is descriptive concordance, not independent replication. The current exploratory totals, 284/260/0, are FDR<0.05 **database-term rows summed after four separate FDR procedures**, not unique pathways.

No seed argument was passed to the historical July producer. The reference environment now contains gseapy 1.2.1, whose default seed is 123, but the effective producer default, package revisions, and live CoGEx corpus/version were not embedded in the outputs. Retained hashes allow exact byte, count, and decision verification offline; they do not guarantee upstream regeneration.

## Stage 7 — Three-contrast triangulation

The fixed pipeline is run for C9-vs-Sporadic, C9-vs-Control, and Sporadic-vs-Control. The bounded fixed-term counts are (8, 6, 0). That pattern argues against a uniformly recurring fixed-graph artifact and against a shared sporadic-ALS pattern in this cohort. It supports the narrower phrase **C9-carrier-associated within-cohort pattern**. It does not prove that the repeat expansion caused the pattern, and a graph-by-contrast interaction could still differ across the three rankings.

Three pathway groupings carry the signal:

1. **mRNA splicing and processing**
2. **chromosome and chromatin organization**
3. **nuclear pore and nucleocytoplasmic transport**

**Reading the empty cell carefully.** The current analysis establishes only that none of the eight fixed terms clears the eightfold reporting threshold in Sporadic-vs-Control. It does not distinguish diffuse biology, subtype averaging, power, or another source of heterogeneity.

## Stage 7, continued — The fixed-term consistency check

The eight terms came from earlier discovery on this same cohort and an older graph traversal. They were fixed before the July log2/measured-only method-transfer reruns, so this is a **same-cohort consistency check**, not prospective preregistration or independent confirmation. A term passes if

$$p < \frac{0.05}{8} = 0.00625 \quad\text{and}\quad \mathrm{NES} > 0.$$

On the bounded log2 analysis, the pass counts are **8/8, 6/8, and 0/8**. Bonferroni-8 is applied separately to the eight rows within each contrast and analysis regime; it is not joint correction across all 24 contrast-by-term cells, bounded and unbounded runs, scopes, four databases, or auxiliary analyses. The same-cohort outcome-guided selection also means it is not a post-selection FWER guarantee. Six terms pass in both C9 contrasts and not Sporadic-vs-Control: the three splicing terms, chromosome, chromatin, and nucleocytoplasmic transport. This is the **six-term cross-contrast core**, not a graph-invariant result.

The current depth sensitivity is bounded **8/6/0** versus unbounded **6/0/0**. This licenses only the statement that the fixed-term pattern attenuates when the local depth bound is removed.

### Auxiliary evidence: current, withheld, or withdrawn

- The canonical size-matched HGNC-set null refits the log2, Sex-adjusted empirical-Bayes model. UniProt rows yield 3,261 HGNC genes after mapping; the sole duplicate gene is aggregated by maximum $t^2$. For each fixed term, its mean $t^2$ is compared with 10,000 uniform same-size sets sampled without replacement from that contrast's finite HGNC background. Plus-one empirical p-values receive Bonferroni-8 separately within each contrast. The pattern is **8/8/0**. Term memberships, measured intersections, and the mapping are frozen in `data/publication/c9_size_matched_null_inputs.json`, making the default rerun offline.
- The canonical pathway-level degree-matched null uses the same 1,407-anchor robust scope and the persisted full-INDRA degree. For each of 9,999 deterministic replicates, it samples one nonmember control per term member, with replacement, from the reciprocal 20% degree window and compares the observed term mean $-\text{slope}$ with the matched distribution. Bonferroni-8 is again applied separately within each contrast. Its pattern is **7/7/0**; Vpr-mediated nuclear import is the sole C9 non-pass. This is separate from the per-anchor p-values that GSEA ignores and does not make GSEA intrinsically degree-adjusted. Its term snapshot is frozen in `data/publication/c9_degree_stratified_null_terms.json`.
- The abundance control and five analytic sensitivities remain May raw artifacts and are withheld pending canonical reruns.
- The STRING comparator is withdrawn: its artifact reverses the production slope convention and has no durable derivation.
- The matched-RNA artifact is withdrawn: it reports an impossible shared-donor denominator, reuses a legacy with-intermediates distance matrix, and uses an unmoderated RNA statistic. No post-transcriptional claim follows.
- The age artifact is a raw-scale incremental-$\Delta R^2$ proxy, not a canonical age-adjusted pathway rerun. Age remains an unresolved limitation.

## Stage 8 — Cross-modality status

The available matched-RNA artifact is withdrawn for accounting and method drift. It does not establish a same-donor, same-statistic comparison and cannot localize the pathway pattern to a post-transcriptional layer. A future audited rerun must first reconcile the cohort intersection, use a stated moderated statistical frame, and recompute distances under the canonical graph design.

## Stage 9 — The edge-level extension, reported as a calibration halt

WASC asks whether fitted cross-protein abundance coupling is similarly dispersed across donor groups. It is a separate, amended analysis with a different cohort and model, not a finer-resolution result from the canonical landscape pipeline. Frozen exclusions leave C9 $n=25$, Sporadic $n=294$, and Control $n=59$—378 donors rather than the primary analysis's 410.

The frozen edge file contains 944 within-theme rows from the eight terms: 434 Splicing, 443 Chromatin, and 67 Transport. These represent 904 unique unordered INDRA hop-1 UniProt pairs because 40 pairs occur in more than one theme. Bidirectional INDRA statements are deduplicated and each pair receives a lexicographic anchor/target orientation. “Anchor” and “target” are regression labels here, not biological arrow direction.

For each oriented pair $(a,j)$ and group $g$, the executed Frisch–Waugh–Lovell kernel residualizes **both** protein vectors against an intercept, Sex, within-group age z-score, and Tissue dummies, then regresses residualized $j$ on residualized $a$. It produces $\hat\beta_g$ and $\mathrm{SE}(\hat\beta_g)$. With $w_g=1/\mathrm{SE}(\hat\beta_g)^2$ and $\bar\beta=\sum_gw_g\hat\beta_g/\sum_gw_g$,

$$Q=\sum_gw_g(\hat\beta_g-\bar\beta)^2.$$

Lower $Q$ means the fitted slopes are more similar across groups. It does not establish a biological interaction or direction.

The executed calibration differs materially from its frozen prose. The prose specified log2 values, within-group protein standardization, and additional within-group ComBat-style batch adjustment. The code loaded the provider-corrected/imputed **linear-intensity** matrix without another transform, did not standardize anchor or target, and explicitly deferred additional ComBat. Its covariate design did include Sex, imputed Age, and Tissue. This is an implementation gap, not a completed version of the stated method.

The executed null used the full 3,264-row matrix as its candidate pool, including no explicit iRT exclusion. It matched substitute targets on measured-only INDRA degree decile and absolute Pearson-correlation decile. The correlations were computed on raw abundance across all 436 matrix columns because the calibration scripts supplied no WASC-donor subset. Sampling excluded the anchor, the current work unit's recorded true-target list, and controls already drawn within an iteration; it did not verify graph non-neighbor status beyond that recorded list. For $n_{\mathrm{finite}}$ usable null values, the lower-tail p-value was

$$p=\frac{1+\#\{Q_b^*\le Q\}}{n_{\mathrm{finite}}+1}.$$

The frozen primary plan specified $B=9{,}999$, BY-FDR at $q=0.10$ for edges, and empirical Brown combination plus BY for a parallel anchor table. The corrected rank-feasibility audit at $N=944$, $q=0.10$, and $H_{944}=7.4279$ shows that $B=999$ leaves ranks 1–70 structurally untestable, $B=9{,}999$ leaves ranks 1–7 untestable while ranks 8–944 are feasible, and the frozen $B=99{,}999$ floor-tie rerun makes all ranks feasible. Neither the primary nor tertiary run became a result because the calibration halted first.

The hard halt occurred earlier. With $B=999$ and a raw-$p<0.10$ selector, downsampling Sporadic from 294 to 25 produced full-versus-downsampled edge-set Jaccards of 0.297, 0.260, and 0.299 across seeds 42, 7, and 99: mean **0.285**, below the frozen **0.70** gate. These artifacts used 904 deduplicated pairs rather than 944 theme rows, but the denominator ambiguity cannot rescue that miss. The halt blocked the primary run. The only licensed WASC statement is instability of this executed selector/pipeline under that downsampling design—no per-edge, per-anchor, biological-coupling, or general $n=25$ verdict exists.

WASC's artifact provenance is weaker than the July pathway receipt. The all-pool calibration JSON was reconstructed from a task log after its original output was overwritten, the downsampling summaries contain no producer hash, and the sizes and SHA-256 values declared in the WASC manifest for `cluster_members_v1.json` and `E_WASC_v1.json` exclude the files' final LF. That does not change the recorded hard fail, but it means these calibration artifacts are not byte-reproducible evidence of the same grade as the canonical pathway files.

## Age as an unresolved limitation

The May artifact computed a raw-scale per-protein incremental $\Delta R^2=(\mathrm{SSE}_{\mathrm{reduced}}-\mathrm{SSE}_{\mathrm{full}})/\mathrm{SST}$. That is a semi-partial increment, not partial $R^2$, and the full age-adjusted slope/GSEA pipeline was never rerun. Its descriptive values are therefore historical context only. The canonical claim remains unadjusted for Age.

## Reproducibility and other unresolved confounding

The canonical model adjusts only Sex. In enriched metadata for the primary arms before WASC exclusions, the coarse Primary_Tissue proxy is strongly group-imbalanced: T-cell labels are 23/25 (92.0%) in C9, 190/294 (64.6%) in Sporadic, and 35/91 (38.5%) in Control. After WASC excludes 32 external controls, its Control fraction is 35/59 (59.3%). The primary model does not include this proxy, and granular PBMC cell composition remains unresolved. Provider-side batch correction, random-forest imputation, and protein roll-up happen before the delivered matrix, so this repository cannot reconstruct missingness or detection effects from a pre-imputation source. The contrast sample sizes differ sharply, which means their pass counts are not equally powered measurements.

The July local files preserve and hash the analyzed result tables well enough to verify bytes, row counts, FDR counts, and fixed-term decisions offline. They do not freeze the live MyGene alias mapping, full CoGEx graph and term corpus, historical producer revisions, or effective GSEA seed/default, so exact upstream regeneration is not guaranteed. The two workflow Markdown files are manually maintained and are not themselves included in the GSEA integrity hash set. Both narratives and the current `data/publication/` authority/input files are version-controlled together. Git preserves their revision history, but it does not add these Markdown files to the GSEA receipt's narrower integrity hash set.

## Current evidence ledger

| Evidence | Current status | What it licenses |
|---|---|---|
| bounded log2 measured-only fixed-term GSEA | current: 8/6/0 on 1,407 robust anchors | same-cohort C9-carrier-associated pathway pattern |
| unbounded log2 measured-only sensitivity | current: 6/0/0 | attenuation when the depth bound is removed |
| exploratory four-database totals | current: 284/260/0 database-term rows | descriptive concordance, not unique-pathway counts |
| per-anchor degree-binned permutations | current diagnostic | individual-slope calibration only; not pathway protection |
| size-matched HGNC-set null | current: 8/8/0 | graph-independent same-size-set corroboration |
| pathway degree-matched null | current: 7/7/0 on robust scope | separate hub-structure sensitivity; Vpr does not pass |
| abundance control and five sensitivities | legacy raw | withheld pending canonical reruns |
| STRING and matched RNA | withdrawn | no network-substrate or post-transcriptional claim |
| age proxy | legacy raw $\Delta R^2$ | unresolved limitation only |
| WASC extension | calibration hard halt at 0.285 versus 0.70 | executed-pipeline instability only; no edge or anchor result |

## What the pipeline licenses, and what it does not

The value of a study is bounded by the discipline of its claims. Here is the boundary, drawn explicitly.

**What the current evidence licenses.** A pathway-level count-and-pattern claim within this cohort: the eight discovery-derived terms transfer to the bounded log2, measured-only analysis with an 8/6/0 pattern on 1,407 robust anchors, and six terms form a cross-contrast core. The graph-independent 8/8/0 size-null and separate 7/7/0 degree-matched sensitivity complement that pattern; the unbounded 6/0/0 result shows attenuation when the depth bound is removed. This is a C9-carrier-associated local regulatory-neighborhood pattern, not a mutation-causal or externally replicated result.

**What it does not license:**

- **No mechanism, causation, or direction from the slope alone.** The gradient describes where differential *magnitude* concentrates; because it is built on $|t|$, it says nothing about the sign or the causal direction of any regulation.
- **No per-protein discovery.** The workflow computes slopes and diagnostic p-values for valid anchors, but it does not license or report individual proteins as discoveries or hits.
- **No canonical post-transcriptional or network-substrate claim.** The RNA and STRING artifacts are withdrawn pending audited reruns.
- **No claim that per-anchor permutation p-values degree-adjust GSEA.** They are not consumed by the pathway ranking.
- **No postmortem-confound story.** This is living-donor peripheral blood, so post-mortem interval and related autopsy confounds simply do not apply — they are neither a threat nor a variable here.
- **No cross-cohort generalization yet.** External replication in a separate cohort is the stated publication gate, and it has not been done. The pattern has not yet been shown to transfer.
- **The edge-level coupling question is unresolved.** Its pipeline-specific calibration halted before the primary run; no per-edge or per-anchor verdict exists.

The through-line is the same one the failed first attempt taught in Stage 3: a significant number is not an answer to a spatial question. The current answer comes from a continuous gradient, a clearly defined robust ranking, a fixed-term same-cohort check, and an explicit boundary around every auxiliary artifact that has not earned its way into the canonical evidence.
