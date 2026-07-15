# Methods

## Overview and design logic

We ask whether the *C9orf72* hexanucleotide-repeat expansion — the most common
monogenic cause of amyotrophic lateral sclerosis (ALS) — leaves a pathway-level
fingerprint in peripheral-blood mononuclear cell (PBMC) proteomics that sporadic
ALS does not.

Any signal we find admits several competing explanations. It could be associated
with C9-carrier status in this cohort, it could reflect a pattern shared with
sporadic ALS, or it could arise from an interaction between the fixed regulatory
graph and the contrast-specific data.

We resolve between these not with a single test but with a triangulation. The
same statistic is computed for three pairwise contrasts — C9-vs-Sporadic,
C9-vs-Control, and Sporadic-vs-Control — and the pattern of which contrasts
pass and which fail is used as a within-cohort triangulation. A two-C9-pass,
sporadic-control-fail pattern argues against a uniformly recurring fixed-graph
artifact and against a shared sporadic-ALS pattern. It does not establish
mutation causality or exclude every graph-by-data interaction.

The Methods below follow the reasoning that produced the design. We first build
the per-protein differential statistic used by the historical target-set
diagnostic and every stage of the primary landscape/GSEA pipeline.
We then show why the obvious first analysis — a single gene-set test on a curated
*C9orf72* target set — has poor specificity for the competitive question on this
dataset, and how re-asking the question against the observed proteome drives its
apparent signal toward null while exposing the structure that actually carries
information: perturbation magnitude decays with network distance. That continuous
observation motivates a continuous statistic — a per-feature perturbation
gradient — aggregated to pathways by preranked enrichment and triangulated
across the three contrasts. Degree-binned permutations provide a per-anchor
diagnostic but are not consumed by the pathway ranking. The current evidence is
the July 2026 log2, measured-only bounded analysis, its unbounded depth
sensitivity, and two canonically rerun auxiliaries: the size-matched gene-set
null and the pathway-level degree-matched null. Legacy versions of those checks,
the abundance check, STRING, matched RNA, age, and five-analysis sensitivities
are not silently pooled into the current evidence; their status is stated
explicitly below. An edge-level extension is reported as a pipeline-specific
calibration halt, not as an edge result.

The authoritative state for the current pathway analysis is analysis
`c9-als-fingerprint-log2-measured-only-h2-2026-07`, recorded 12 July 2026 in
`data/publication/c9_primary_analysis.json`. The retained GSEA files and their
integrity boundary are inventoried in
`data/publication/c9_gsea_provenance.json`. If this prose and either
machine-readable record disagree, the machine-readable record controls.

## Cohort, data, and covariate

The proteomics are AnswerALS PBMC intensities: a matrix of 3,264 measured rows by
436 samples. The rows comprise 3,263 human UniProt features and one internal
retention-time standard (`1/iRT_protein`). The provider processes intensities
upstream — batch correction, then
random-forest imputation, then protein roll-up — so the matrix delivered to us is
complete and carries no missing values. Metadata intersection retains 423 donors;
the three analyzed arms, defined by genotype and phenotype, comprise 410 of
these. The other 13 metadata-matched cases carry other known mutations and are
outside the primary arms:

| Arm | $n$ |
|---|---|
| C9 (repeat-expansion carriers) | 25 |
| Sporadic ALS | 294 |
| Control | 91 |

Operationally, the C9 arm includes donors whose mutation detail identifies
C9orf72 or whose repeat length is at least 30. Sporadic requires CASE phenotype,
excludes the known-mutation labels `C9orf72`, `SOD1`, `FUS`, `TARDBP`,
`TARDBP (TDP43)`, `SETX`, `Multiple`, and `Other`, and has repeat length below
30 or missing. Control is the CTRL phenotype arm.

The canonical bounded/unbounded analyses and the two current auxiliaries use
$$\tilde{x}_{ij} = \log_2(x_{ij} + 1),$$
for protein $i$ and donor $j$, where the $+1$ absorbs exact zeros. Working in log
space converts multiplicative effects into additive ones, which is the scale on
which the primary linear models below are specified. Historical target-set, age,
and WASC artifacts are explicitly labeled and must not be assumed to share this
scale.

We adjust for exactly one nuisance covariate, Sex, and we include it because of a
real, arm-dependent imbalance rather than by reflex. The sporadic arm is 212/294
(72.1%) male, against 47/91 (51.6%) in controls and 17/25 (68.0%) in C9, so an
unadjusted group coefficient would partly encode sex differences rather than
disease. Age is not a covariate in the canonical model. A legacy raw-scale
incremental-$R^2$ exploration exists, but no canonical age-adjusted pathway rerun
has been performed; age therefore remains an unresolved limitation.

All 3,264 measured features are attempted as anchors in the bounded,
measured-only network analysis. Of these, 3,117 yield valid two-shell gradients;
137 have no reachable measured neighbor (`DisconnectedFeature`), and 10 more
fail the implemented guardrail requiring at least 10 measurable genes in the
neighborhood. These are network-statistic exclusions, not missing-intensity or
data-quality exclusions. The iRT standard is one of the 137 disconnected rows.

## The per-protein moderated-t statistic

Every stage of the primary landscape/GSEA pipeline consumes a single
per-protein number: how many standard errors a protein's abundance shifts
between two groups. This is the atom of that pipeline, so we build it carefully.

For protein $i$ we fit a linear model
$$\tilde{x}_i = X\beta_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2 I),$$
with design $X = [\,\mathbf{1} \mid d_{\text{group}} \mid d_{\text{Sex}}\,]$ — an
intercept, a group indicator, and a sex indicator — and we test the group
coefficient $\hat\beta_i$, the modeled abundance difference between the two arms.

Fitting each protein in complete isolation would let a protein with a freakishly
small sample variance spuriously top the ranking, because the test statistic
divides by that variance. To prevent this we use a local, limma-style empirical-
Bayes variance moderation following Smyth (2004); the production analysis does
not call the R `limma` package. The ordinary residual variance $s_i^2$,
estimated on $d_i$ residual degrees of freedom, is shrunk toward a prior
$(s_0^2, d_0)$ estimated once across all proteins by the local method-of-moments
implementation:
$$\tilde s_i^2 = \frac{d_0\,s_0^2 + d_i\,s_i^2}{d_0 + d_i}, \qquad
  t_i = \frac{\hat\beta_i}{\tilde s_i \sqrt{v_i}}\ \sim\ t_{\,d_0 + d_i},$$
where $s_0^2$ and $d_0$ are the prior variance and prior degrees of freedom,
$v_i$ is the contrast's diagonal entry of $(X^\top X)^{-1}$, and $t_i$ is the
moderated $t$-statistic. In words, each protein's variance becomes a degrees-of-
freedom-weighted compromise
between its own noisy estimate and the proteome-wide prior, so no single protein
is trusted purely on a too-good-to-be-true variance of its own. As $d_0 \to
\infty$ the moderated variance collapses entirely to the common prior $s_0^2$; at
the other extreme $d_0 \to 0$ recovers the ordinary per-protein $t$. The July
landscape result embedded neither $d_0$ nor $s_0^2$. The retrospective primary
receipt records the same three $d_0$ values; both parameters below come from the
current-code log2/Sex refit retained with the canonical size-matched null, and
the posterior weights are derived from them. The table is therefore a current
reference reconstruction of the local method, not a byte-level record of every
July empirical-Bayes hyperparameter:

| Contrast | $d_i$ | $d_0$ | $s_0^2$ | prior weight $d_0/(d_0+d_i)$ |
|---|---:|---:|---:|---:|
| C9-vs-Sporadic | 316 | 4.9847 | 0.059531 | 1.55% |
| C9-vs-Control | 113 | 5.5097 | 0.064427 | 4.65% |
| Sporadic-vs-Control | 382 | 4.9219 | 0.056273 | 1.27% |

Thus the prior contributes a small, reported share of each posterior variance.
Variance shrinkage changes the numerical moderated $t$ used in the ranking;
$d_0+d_i$ separately defines the reference $t$ distribution used for tail
probabilities. The fitted values are estimates, not tuned hyperparameters or by
themselves a goodness-of-fit test.

The downstream signal is the magnitude $|t_i|$. Taking the absolute value makes
the analysis about perturbation *size* and direction-agnostic. This is the
correct primitive for asking where in the network abundance is disturbed: a
protein that is strongly down-regulated is as informative about the location of a
perturbation as one that is strongly up-regulated, and only their common
magnitude should enter a distance-decay analysis.

## Historical target-set diagnostic and its provenance boundary

The first design question used a curated set of 47 *C9orf72*-related proteins.
It is historical motivation, not part of the canonical July log2 evidence, and
the retained files do not form one analysis that can be narrated as a sequence
of directly comparable p-values.

The historical target-set branch used the provider-corrected/imputed
linear-intensity matrix without a repository log transform. This scale applies
to both the complete-case and matched diagnostics below.

The identified 28 March run used a different complete-case cohort (C9 $n=23$,
Sporadic $n=282$) and adjusted for Sex, age at symptom onset, and baseline
ALSFRS-R. The current retained rotation implementation converts moderated $t$
statistics to normal scores before computing a mixed-direction mean-squared
normal-score statistic (MSQ); the March output did not embed its exact producer
revision, so that implementation detail cannot be proved retrospectively. For
$B$ rotations,
$$p = \frac{b + 1}{B + 1}, \qquad
  b = \#\{\mathrm{rotated\ MSQ} \ge \mathrm{observed\ MSQ}\}.$$
The retained target-set p-value is 0.0157. Of 200 uniformly sampled, same-size
sets drawn without replacement from the measured non-target pool, 113 also had
self-contained $p<0.05$ (56.5%).
Those are observed-data comparison sets, not global-null sets; 56.5% is
therefore neither a false-positive rate nor a type-I error estimate. It shows
that significance in the self-contained test was not specific for the desired
competitive question. The target rotation used evidence weights, whereas the
200 random control sets were unweighted, so the count is not a like-for-like
weighted target-versus-control calibration.

The retained competitive producer used a random-set test, not an analytic
complement standard error or Camera/VIF correction. In each of $B=10{,}000$
draws it sampled a same-size set without replacement from all finite protein
rows; because the pool was not target-excluded, a null set could overlap the
target set. If $T_{\mathrm{obs}}$ is the target mean $|t|$ and $T_b^*$ is a
random-set mean, it reported
$$z=\frac{T_{\mathrm{obs}}-\overline{T^*}}{s(T^*)}, \qquad
  p=\frac{1+\#\{T_b^*\ge T_{\mathrm{obs}}\}}{B+1}.$$
For the complete-case fit, $T_{\mathrm{obs}}=1.298$, the null mean was 1.133,
the null SD was 0.111, $z=1.479$, and $p=0.075$. The same procedure in the
Sex-matched 50-donor artifact gave $T_{\mathrm{obs}}=0.886$, null mean 0.919,
null SD 0.089, $z=-0.380$, and $p=0.649$. Both artifacts carry VIF and mean-
correlation fields equal to 1 and 0, respectively, but those fields do not enter
this random-set statistic. The two cohort analyses are related diagnostics, not
successive corrections to one test. Both random-set calls left their seed at the
`None` default; their exact 10,000 null draws and p-values cannot be regenerated,
even though the broader validation command recorded seed 42.

Earlier drafts quoted a Sex-only sequence of approximately 0.21, 0.69, and 0.65
and a mean-$|t|$ difference of 0.13–0.17. No identified machine-readable
producer artifact for that sequence remains in the repository, so those values
are not used as evidence here. The durable design lesson is narrower: the
self-contained result lacked competitive specificity, and the retained
competitive analyses did not establish target-set enrichment. That motivated
the continuous network-distance question tested by the July analysis below.

## The perturbation gradient

We attempt every measured feature row as an anchor and ask how perturbation magnitude
decays as we move away from it through the regulatory network.

The network is extracted from live INDRA CoGEx using four statement types
(`Activation`, `Inhibition`, `IncreaseAmount`, and `DecreaseAmount`) with at
least one item of evidence. Both endpoints and every traversed intermediate are
restricted to the measured feature universe. The UniProt-to-symbol/alias bridge
is queried live through MyGene; distance is the minimum finite distance over a
row's resolved symbol/alias pairs.
The source statements are directed, but distance is measured *undirected*,
because the question is regulatory-neighborhood proximity rather than causal
flow. Thus a reported hop distance never routes through an unmeasured protein.
The retained `distances.npz` files contain the analyzed distance matrices, while
`distances.meta.json` preserves their feature labels, degrees, and unmatched
features. The canonical GSEA receipt hashes the metadata JSON, not the NPZ
matrix bytes. The analyzed substrate is therefore retained locally but the
distance-matrix byte integrity lies outside that receipt; the producing live
CoGEx corpus/version was also not embedded and cannot be reconstructed exactly
from the repository alone.

For anchor $a$, the shell at hop distance $h$ is the set of measured proteins
exactly $h$ steps away,
$$R_h(a) = \{\,i : d(a, i) = h\,\},$$
with $d$ the undirected regulatory graph-distance over measured proteins, and the
shell's mean signal is
$$\bar S_h = \frac{1}{|R_h|}\sum_{i \in R_h} |t_i|.$$
The anchor's gradient slope is the shell-size-weighted least-squares slope of
$\bar S_h$ on $h$, with weights $w_h = |R_h|$ so that larger and better-estimated
shells count more:
$$\hat m_a = \frac{\sum_h w_h\,(h - \bar h)(\bar S_h - \bar S)}{\sum_h w_h\,(h - \bar h)^2},$$
where $\bar h = (\sum_h w_h h)/(\sum_h w_h)$ and $\bar S = (\sum_h w_h \bar
S_h)/(\sum_h w_h)$ are the weight-averaged hop distance and shell mean. A
negative $\hat m_a$ means perturbation concentrates near the anchor and decays
with distance — local concentration, which is the pattern a real,
spatially-organized perturbation should leave.

The analysis is bounded at depth 2 by reasoned choice, not by a reachability
limit. At depth two the statistic compares the anchor's direct regulatory
partners with its two-hop partners under an **undirected** proximity metric. It
does not trace directional propagation. Extending deeper changes the question
rather than sharpening it: it
stops measuring local decay and starts asking whether the anchor's neighborhood
differs from its entire connected component, a weaker and more diffuse
comparison. The current unbounded sensitivity attenuates the pass pattern from
8/6/0 to 6/0/0. That establishes depth sensitivity; it does not by itself
identify the deeper-shell signal as a particular kind of biological noise.

At depth 2, with shells $h \in \{1, 2\}$, the weighted slope reduces exactly to
$$\hat m_a = \bar S_2 - \bar S_1,$$
independent of the weights — the mean $|t|$ of the second ring minus that of the
first. This transparency is a virtue: the headline statistic is a plain
difference of two shell means, with no hidden weighting to interrogate.

## Per-anchor degree-stratified diagnostic

Network degree is a plausible graph-structural nuisance because proteins of
different degree can be represented differently across distance shells. We did
not establish a positive degree–$|t|$ association in the canonical data, so the
analysis does not treat larger hub $|t|$ as an observed dominant confound.
Instead, degree stratification is a conservative diagnostic of how a slope
compares with label arrangements that approximately retain degree location.

For each anchor, production removes the anchor from the label pool. Each
remaining measured feature receives its full-INDRA count of incident regulatory
relationships, taking the maximum over that UniProt row's resolved aliases. This
is not the number of unique measured hop-1 neighbors; the degree query applies
the statement-type filter but does not reapply the shell extraction's explicit
evidence-count predicate. Features are sorted by degree and
partitioned into consecutive rank blocks of 100; $|t|$ is shuffled within each
block. These are similar-degree blocks, not exact degree matches. Randomness uses
base seed 42 plus a stable MD5-derived anchor identifier. The gradient's
one-sided (decay) empirical p-value over $B=999$ permutations is
$$p_a = \frac{1 + \#\{\hat m_a^{(b)} \le \hat m_a\}}{B + 1},$$
where $\hat m_a^{(b)}$ is the slope recomputed on permutation $b$. This p-value
calibrates an individual anchor's slope against this degree-stratified null. The
software boundary is load-bearing: pathway GSEA ranks the observed
$-\hat m_a$ values and does **not** consume $p_a$. This permutation is therefore
a per-anchor diagnostic, not protection of the 8/6/0 pathway inference. A
separate pathway-level degree-matched rerun now gives 7/7/0 on the canonical
robust scope; it remains a distinct auxiliary, not an input to GSEA.

## From slopes to pathways

A per-protein slope tells us where perturbation concentrates around one anchor; to
reach biology we aggregate those slopes into pathways.

The bounded run yields 3,117 valid slopes. The primary `robust` scope then keeps
the 1,407 anchors whose measured hop-1 neighborhood contains at least 20
features. All 1,407 map to unique HGNC identifiers in the retained runs. We rank
them by $r_a = -\hat m_a$, so that the most concentrated anchors (most negative
slope) sit at the top of the ranking. We then run preranked gene-set enrichment
analysis (GSEA;
Subramanian et al. 2005) against four annotation databases: Gene
Ontology, Reactome, WikiPathways, and the Human Phenotype Ontology. Let $N$ be
the number of ranked anchors, let $S'=S$ intersect the ranked HGNC universe, and
let $k=|S'|$. We walk the ranking
$i = 1, \dots, N$ and track the running difference between weighted hits and
misses. Let $RS_i(S)$ denote that signed running sum,
$$RS_i(S)=\sum_{\substack{a \le i \\ a \in S'}}
  \frac{|r_a|^{\kappa}}{\sum_{a' \in S'} |r_{a'}|^{\kappa}}
  \ -\ \sum_{\substack{a \le i \\ a \notin S'}} \frac{1}{N - k}, \qquad \kappa = 1.$$
The signed enrichment score is the excursion with greatest absolute magnitude,
$$ES(S)=RS_{i^*}(S), \qquad i^*=\arg\max_i |RS_i(S)|.$$
Thus a set concentrated at the bottom can have negative ES/NES; the sign is not
discarded. The weighting
exponent $\kappa$ controls how strongly a member's rank magnitude — not merely
its presence — drives the enrichment; $\kappa = 1$ is the standard weighted
Kolmogorov–Smirnov choice, in which each hit contributes in proportion to its
score. A large positive enrichment means the set's measured members cluster
toward the concentrated end of the ranking.

The retained July runs call the INDRA CoGEx continuous-enrichment wrappers,
which execute `gseapy.prerank` separately for each database. Each call uses
1,000 gene-set permutations, weighted-score exponent $\kappa=1$, minimum matched
set size 1, and no top-score trimming; the current reference wrapper permits
sets up to 50,000 members. No phenotype labels are permuted.

Significance comes from gene-set (label) permutation over the fixed ranking: the
members of $S$ are the unit of permutation, never the phenotype. We ask whether a
pathway's members sit non-randomly toward the concentrated end of a ranking we
hold fixed. Raw enrichment scores grow systematically with gene-set size, so a
raw $ES$ is not comparable across pathways of different sizes. GSEA normalizes
against the magnitude of the **same-sign** permutation null,
$$D_s(S)=\mathbb{E}\!\left[|ES^{\mathrm{null}}(S)|\mid
\operatorname{sign}(ES^{\mathrm{null}})=s\right], \qquad
\mathrm{NES}(S)=\frac{ES(S)}{D_{\operatorname{sign}(ES(S))}(S)},$$
which reduces the dependence of enrichment magnitude on matched set size. Both
the nominal $p$-value and false-discovery $q$-value are estimated from the GSEA
null. Each database is executed and FDR-calibrated separately; there is no
single across-database FDR family.

The four databases provide complementary annotations, but their terms overlap
and the same samples, slopes, and discovery history feed every database.
Cross-database recurrence is descriptive concordance, not independent
replication. Exploratory FDR<0.05 totals are 284/260/0 database-term rows summed
across the four separate database runs, not counts of unique or independent
pathways.

No seed argument was passed to the July GSEA producer. The reference environment
now contains gseapy 1.2.1, whose default seed is 123, but the effective
historical default, producer package revisions, and live CoGEx term corpus were
not embedded in the output files. The provenance snapshot verifies retained
bytes, row counts, FDR counts, and fixed-term decisions offline; it does not
guarantee exact regeneration from upstream resources.

## Three-contrast triangulation

Having a pathway-level statistic, we run the entire pipeline — same network, same
$|t|$ computation, same slope, same GSEA — three times, once per pairwise
contrast, and read the pass/fail pattern. The pattern is diagnostic because the
three competing explanations predict distinct signatures:

| Pattern (C9-vs-Spor, C9-vs-Ctrl, Spor-vs-Ctrl) | Interpretation |
|---|---|
| (pass, pass, —) | consistent with a C9-carrier-associated pattern |
| (—, pass, pass) | consistent with shared ALS pathology |
| (pass, pass, pass) | consistent with a uniformly recurring fixed-graph pattern |
| (—, —, pass) | consistent with a sporadic-associated pattern |

The observed fixed-term pattern is (pass, pass, —), or 8/6/0. It argues against a
uniformly recurring fixed-graph pattern and against a shared sporadic-ALS pattern
in these samples. Other graph-by-contrast interactions remain possible, and the
design supports association with C9-carrier status rather than mutation
causality. The current analysis does not decide whether the weaker
Sporadic-vs-Control pathway pattern reflects biological diffusion,
subtype-averaging, or another source of heterogeneity.

### Discovery-derived fixed-term consistency check

The eight terms were selected during earlier discovery on the same cohort and an
older graph traversal, then fixed before the log2/measured-only method-transfer
reruns. This makes the July readout a same-cohort consistency check, not a
prospective preregistration or independent confirmation. For each fixed term we
require
$$p < \frac{0.05}{8} = 0.00625 \quad\text{and}\quad \mathrm{NES} > 0,$$
using the raw GSEA permutation p-value. This is an eightfold multiplicity
threshold applied separately to the eight term rows within each contrast and
analysis regime. It is not a joint correction across all 24 contrast-by-term
cells, bounded and unbounded runs, four databases, or auxiliary analyses.
Because the overlapping terms were selected using the same cohort, it is not a
valid post-selection FWER or selective-inference guarantee.
On the bounded log2 analysis, 8/8 pass in C9-vs-Sporadic, 6/8 in C9-vs-Control,
and 0/8 in Sporadic-vs-Control. The six terms shared by the two C9 contrasts form
a **cross-contrast core**: the three splicing terms, chromosome, chromatin, and
nucleocytoplasmic transport.

### Current depth sensitivity

The bounded depth-2 pattern is 8/6/0; the measured-only unbounded rerun is 6/0/0.
This attenuation supports a narrow claim that the fixed-term result is sensitive
to the local depth bound. It does not prove a particular deep-shell mechanism or
show that bounded depth is globally superior.

### Auxiliary-evidence boundary

The bounded/unbounded log2 measured-only landscapes and July GSEA are primary.
Two auxiliary checks have now been rerun canonically; the other inherited May
artifacts remain outside the evidence:

- The graph-independent size-matched HGNC-set null refits the canonical log2,
  Sex-adjusted empirical-Bayes model. UniProt rows map to 3,261 HGNC genes; the
  sole duplicate HGNC gene is aggregated by maximum $t^2$. For each term it
  compares observed mean $t^2$ with 10,000 uniform same-size sets sampled
  without replacement from that contrast's finite HGNC background. Its
  one-sided empirical p-value uses the plus-one correction, and Bonferroni-8 is
  applied separately within each contrast. The pass pattern is 8/8/0.
  Term memberships, measured intersections, and the UniProt-to-HGNC map are
  frozen in `data/publication/c9_size_matched_null_inputs.json`, so the default
  rerun is offline.
- The separate pathway-level degree-matched null uses the same 1,407-anchor
  robust scope and persisted full-INDRA degrees. In each of 9,999 deterministic
  replicates, it samples one nonmember control per term member, with replacement,
  from a reciprocal 20% degree window, and compares the term's observed mean
  `−slope` with the matched distribution. Bonferroni-8 is again applied
  separately within each contrast. Its pass pattern is 7/7/0; Vpr-mediated
  nuclear import is the sole C9 non-pass. This is distinct from the per-anchor
  p-values above and does not convert GSEA into a degree-adjusted test. Its term
  memberships are frozen in
  `data/publication/c9_degree_stratified_null_terms.json`.
- The abundance-stratified check and five analytic sensitivities exist only on
  the legacy raw analysis and are withheld from the canonical evidence battery.

The same selection boundary applies to both current auxiliaries: their eightfold
counts are same-data conditional sensitivities, not post-selection FWER control
or independent confirmation.

- The legacy STRING comparator is withdrawn. Its document uses the opposite
  slope orientation from production and has no durable derivation, so its
  all-negative NES pattern cannot support a regulatory-versus-physical claim.
- The legacy matched-RNA artifact is withdrawn. It reports an impossible shared
  donor denominator, reuses a with-intermediates distance matrix, and uses an
  unmoderated RNA statistic. No canonical cross-modality or post-transcriptional
  claim is licensed.
- The age artifact is a raw-scale per-protein proxy, not an age-adjusted rerun of
  the pathway analysis. Its formula is incremental $\Delta R^2 =
  (\mathrm{SSE}_{\mathrm{reduced}}-\mathrm{SSE}_{\mathrm{full}})/
  \mathrm{SST}$, not partial $R^2$. Age remains a limitation.

## Edge-level coupling-invariance extension

WASC (Within-cluster Anchor-Slope Concordance) is a distinct, amended extension,
not another stage of the canonical landscape pipeline. Its frozen exclusions
leave 378 donors: C9 $n=25$, Sporadic $n=294$, and Control $n=59$. It enumerates
944 within-theme rows from the eight fixed terms, corresponding to 904 unique
unordered INDRA hop-1 UniProt pairs because 40 pairs occur in more than one
theme. Bidirectional INDRA statements are deduplicated, and each pair is assigned
an arbitrary lexicographic anchor/target orientation; this is not biological
regulator-to-target direction.

For each oriented pair $(a,j)$ and group $g$, the executed Frisch–Waugh–Lovell
kernel residualizes **both** abundance vectors against an intercept, Sex,
within-group age z-score, and three-level Tissue dummies, then regresses the
residualized $j$ vector on the residualized $a$ vector. This yields
$\hat\beta_g$ and $\mathrm{SE}(\hat\beta_g)$. With
$w_g=1/\mathrm{SE}(\hat\beta_g)^2$ and
$\bar\beta=(\sum_g w_g\hat\beta_g)/(\sum_g w_g)$, the cross-group dispersion is
$$Q = \sum_g w_g\,(\hat\beta_g-\bar\beta)^2,$$
where lower $Q$ means more similar fitted coupling across groups.

There is an important specification-to-execution gap. The frozen WASC prose
described log2 abundance, within-group standardization, and an additional
within-group ComBat-style batch adjustment. The calibration code instead loaded
the provider-corrected/imputed linear-intensity matrix without a log transform,
did not standardize the two protein vectors, and did not implement an additional
ComBat step. Its group designs did include Sex, imputed Age, and Tissue. This gap
is one more reason the calibration cannot be promoted to a primary edge result.

The executed lower-tail null used the full 3,264-row matrix as its candidate
pool, with no explicit iRT exclusion. It matched substitute targets within
anchor-specific cells defined by measured-only INDRA degree decile and absolute
Pearson-correlation decile. The sampler excluded the anchor, the current work
unit's recorded true-target list, and controls already drawn in that iteration;
it did not verify that candidates were graph non-neighbors beyond that recorded
list. Because the calibration scripts supplied no WASC donor subset, those
correlations used raw abundance across all 436 matrix columns. The p-value used
the number of finite null draws,
$$p = \frac{1+\#\{Q_b^*\le Q\}}{n_{\mathrm{finite}}+1},$$
and the frozen primary plan specified $B=9{,}999$, followed by
Benjamini–Yekutieli (BY) FDR at $q=0.10$ for the edge family and an
empirical-Brown combination followed by BY for a parallel anchor-level table.
The corrected rank-feasibility audit at
$N=944$, $q=0.10$, and $H_{944}=7.4279$ shows that $B=999$ leaves ranks 1–70
structurally untestable, $B=9{,}999$ leaves ranks 1–7 untestable while ranks
8–944 are feasible, and the frozen $B=99{,}999$ floor-tie rerun makes every rank
feasible. Neither the primary nor tertiary run became a result because the
calibration halted first.

The calibration halt occurred before that primary plan ran. The
Sporadic-downsampling tripwire used
$B=999$ and selected edges at raw $p<0.10$. Across seeds 42, 7, and 99,
downsampling Sporadic from 294 to 25 produced Jaccards 0.297, 0.260, and 0.299
(mean 0.285), below the frozen 0.70 requirement. Those artifacts operated on the
904 deduplicated pairs rather than the 944 theme-counted rows; the unresolved
denominator choice does not change a miss of this size. The hard halt blocked
the primary run. WASC therefore establishes only instability of this executed
selector/pipeline under that downsampling design—no edge, anchor, coupling, or
general $n=25$ conclusion.

WASC's retained provenance is also weaker than the July pathway receipt: the
all-pool calibration JSON was reconstructed from a task log after an overwrite,
the downsampling summaries do not embed producer hashes, and the sizes and
SHA-256 values declared in the WASC manifest for `cluster_members_v1.json` and
`E_WASC_v1.json` exclude the files' final LF. These limitations do not change
the recorded hard fail, but they preclude treating WASC as byte-reproducible
evidence of the same grade as the canonical pathway artifacts.

## Reproducibility and unresolved confounding

The primary model adjusts only Sex. It has no canonical Age-adjusted pathway
rerun. In enriched metadata for the primary arms before WASC exclusions, the
coarse Primary_Tissue proxy was strongly group-imbalanced: T-cell labels were
23/25 (92.0%) in C9, 190/294 (64.6%) in Sporadic, and 35/91 (38.5%) in Control.
After WASC excluded 32 external controls, its Control fraction was 35/59
(59.3%). The canonical primary model does not include this proxy, and granular
PBMC cell composition remains unresolved. The three contrasts also have sharply
different sample sizes, so pass-count differences must not be read as equally
powered comparisons.

Provider-side batch correction, random-forest imputation, and protein roll-up
precede the delivered matrix. The repository has no corresponding detection
mask or pre-imputation matrix with which to reconstruct missingness effects. The
eight terms remain discovery-derived on this cohort, and no external cohort has
been analyzed. Finally, the live CoGEx corpus/version, historical GSEA producer
revisions, and effective GSEA seed were not embedded in the July outputs.
Retained hashes make the analyzed artifacts auditable offline, but exact upstream
regeneration is not guaranteed. These two Markdown narratives are manually maintained
and are not themselves included in the GSEA integrity hash set.
Both narratives and the current `data/publication/` authority/input files are
version-controlled together. Git preserves their revision history, but it does
not add these Markdown files to the GSEA receipt's narrower integrity hash set.

## What this licenses — and what it does not

### Licenses

The canonical analysis supports a pathway-level count-and-pattern claim within
this cohort: the eight discovery-derived terms transfer to the bounded log2,
measured-only analysis with an 8/6/0 pattern, and six terms form a cross-contrast
core. The unbounded 6/0/0 result shows that this pattern attenuates when the depth
bound is removed. The graph-independent 8/8/0 size-null and separate 7/7/0
degree-matched null complement that result without constituting independent
replication. The safe interpretation is a C9-carrier-associated, local
regulatory-neighborhood pattern in these PBMC proteomics data. Cross-database
recurrence is complementary annotation of the same ranking, not independent
replication.

### Does not license

The slope statistic describes where differential *magnitude* concentrates; it
does not identify mechanism, causation, direction of regulation, or a
post-transcriptional layer. No per-protein discovery is licensed. The primary
GSEA still ranks raw observed slopes; the separate 7/7/0 auxiliary does not turn
it into an intrinsically degree-adjusted test. The withdrawn STRING and RNA
artifacts license no network-substrate or cross-modality conclusion; the legacy
age analysis does not resolve age confounding. Nothing here bears on
postmortem-specific confounds—this is living-donor peripheral blood, so
post-mortem interval does not apply. No cross-cohort generalization is claimed:
external replication is the stated publication gate and has not been performed.
The edge-level coupling-invariance question also remains unresolved because its
pipeline-specific calibration gate halted before the primary run.
