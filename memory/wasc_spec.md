# WASC — Within-cluster Anchor-Slope Concordance: Statistical Specification

> **STATUS — v1.0 (PRE-REGISTRATION).** This spec is binding. It incorporates the brutalist modifications and the M1 enumeration results (locked 2026-06-02). All sections marked PRE-REGISTERED must be tagged in the git commit `wasc-prereg-v1.0` (M6a) prior to any analysis compute. M1 (edge enumeration + Frisch-Waugh kernel) is COMPLETE; M2 onward proceeds against this spec.
>
> **M1 results (LOCKED INPUTS):**
> - `|E_WASC| = 944` total INDRA hop-1 within-theme edges over the Wave-22 measured proteome (3,264 UniProt accessions). Per-theme: **Splicing 434** (190 measured cluster members from 304 UniProt / 303 HGNC), **Chromatin 443** (145 measured cluster members from 468 UniProt / 467 HGNC), **Transport 67** (42 measured cluster members from 70 UniProt / 70 HGNC). Total measured cluster members: **377**.
> - Within-theme edge densities (biological sanity): Splicing 2.4%, Chromatin 4.2%, Transport 7.8%.
> - INDRA query: 2026-06-02 against `bolt://indra-cogex-lb-b954b684556c373c.elb.us-east-1.amazonaws.com:7687`.
> - Three-contrast C2 floor (empirical, derived from `|E_WASC|=944`): `ceil(0.05 · 944) = 48`.
>
> **Brutalist modifications applied in this v1.0 (vs. v0.9 draft):**
> 1. Null model uses a **3-axis** match: degree-decile × missingness-decile × **pooled |Pearson(anchor_a, p)| decile** (§4).
> 2. STRING control has **five decision branches**: `INDRA-SPECIFIC`, `STRING-STRONGER`, `INCONCLUSIVE`, `STRING-UNDERPOWERED`, `STRING-ZERO-POSITIVES`. The primary STRING effect-size is **ΔQ on the anchor-pair edge-INTERSECTION** (robust to zero positives); median-of-positives is secondary. `STRING-ZERO-POSITIVES` explicitly **forbids** the "INDRA-specific by exclusion" interpretation (§7).
> 3. STRING ENSP→UniProt mapping is **canonical-only** (not the union over isoforms) to avoid isoform double-counting (§7.1).
> 4. Three-contrast C2 floor uses the **empirical 48** edges (= `ceil(0.05 · 944)`), not a re-derivation (§8).
> 5. C9 Batch is **collapsed to site/year** before within-group ComBat, to defeat singleton-batch degeneracy across 25 C9 donors (§2.1).
> 6. **Mandatory tertiary sensitivities** are PRE-REGISTERED and run regardless of primary outcome: T-Cell-stratified, iPSC-retained, batch-correction-OFF, down-sampled-SPOR-to-25 (most important — detects n-asymmetry attenuation), all-protein-pool null, B=99999 if floor-tied. Sensitivities may use B=999; primary stays at B=9999 (§10).
> 7. **M1 numerical reference tests** are run and recorded: (a) Frisch-Waugh vs `statsmodels.OLS` agreement to 1e-8 on 50 real triples; (b) `j~a` vs `a~j` swap-invariance — if Q drifts >5% on real noisy data, the slope-symmetry caveat is appended verbatim to §9 (§2.5, §12).
> 8. **Claim ceiling adopts the brutalist-revised version verbatim** (§9). Forbidden language list extended: *mechanism, causal, drives, regulates, controls, rewiring, validates, post-transcriptional, INDRA-edges-are-correct*.
> 9. Runtime budget: ~25–40 h per primary run; sensitivities batch ~120–200 h (see build plan; M2 anchor loop is `joblib`-parallel `n_jobs=-1`).
> 10. Pre-registration manifests are **frozen artifacts** at M6a tag time: `data/wasc/cluster_members_v1.json`, `data/wasc/E_WASC_v1.json` (and downstream E2–E9, see §10).

**Version:** 1.0 (pre-registration)
**Date:** 2026-06-02
**Author:** WASC working group
**Status:** FROZEN AT TAG `wasc-prereg-v1.0`. Any post-tag deviation is a SECONDARY analysis with explicit "post-hoc" labelling.

WASC tests whether the *cross-protein abundance coupling* among INDRA hop-1 neighbor pairs within the eight pre-registered C9-ALS cluster terms is *invariant* across the three donor groups (C9, Sporadic, Control), more so than would be expected from degree-, coverage-, and marginal-correlation-matched non-neighbor pairs. It is a covariance-structure test, not a mean-shift test, and it is restricted to within-cluster edges. A positive result claims slope-shift structure descriptively; it does not claim mechanism, causation, or that INDRA edges are validated regulatory relationships.

---

## 1. Edge enumeration

Let `T = {Splicing, Chromatin, Transport}` be the three pre-registered cluster themes. The 8 cluster terms (PRE-REGISTERED, frozen in `scripts/viz/common.py::TERMS`) are:

| Theme | Term | Cogex ID |
|---|---|---|
| Splicing | mRNA Splicing | reactome:R-HSA-72172 |
| Splicing | Processing of Capped Intron-Containing Pre-mRNA | reactome:R-HSA-72203 |
| Splicing | mRNA splicing, via spliceosome | go:0000398 |
| Chromatin | chromosome | go:0005694 |
| Chromatin | chromatin | go:0000785 |
| Transport | nucleocytoplasmic transport | go:0006913 |
| Transport | nuclear pore | go:0005643 |
| Transport | Vpr-mediated nuclear import of PICs | reactome:R-HSA-180910 |

For each theme T:
1. `C_T` = union over `members(t)` for `t ∈ TERMS, cluster(t) = T`, where `members(t)` is computed by `fetch_term_members_via_indra([term_id_t])` → `hgnc_ids_to_uniprots(union)` → UniProt set. (Code path: `scripts/viz/common.py:213,253`.)
2. `M` = set of **3,264** measured UniProt accessions in the Wave-22 protein-level matrix.
3. `M_T = M ∩ C_T` = measured cluster members for theme T.

**Edge set:**

```
E_WASC = ⋃_{T ∈ Themes} { {a, j} : a, j ∈ M_T, a ≠ j, dist_INDRA(a, j) = 1,
 anchor_a is INDRA-resolvable }
```

where `dist_INDRA` is the hop distance on the **Wave-24l measured-only INDRA regulatory subgraph** restricted to `ALL_REGULATORY_TYPES` (Activation / Inhibition / IncreaseAmount / DecreaseAmount), computed by `extract_subgraph_induced_by_features(... restrict_endpoints_to_features=True, max_hops=1, node_filter=M)` then `compute_all_pairs_shortest_paths_bounded(... max_hops=1)`.

**Edges are undirected**: each unordered pair `{a, j}` is fit once. The regression assigns `y_j` to the lexicographically smaller of the two UniProts and `anchor_a` to the other. The choice is symmetric in inference (slope sign just flips with the swap; Q is invariant up to the swap-invariance tolerance verified in §12).

**Both endpoints must be measured cluster members of the SAME theme T.** Cross-theme INDRA-hop-1 pairs are **DEFERRED** out of M1 scope and are NOT in `E_WASC`. They will be reported only in a future exploratory secondary module that does not affect the primary BY-FDR pool.

**Locked edge count (M1):**

| Theme | `|M_T|` (measured cluster members) | `|C_T|` UniProt union | `|C_T|` HGNC union | Within-theme edges |
|---|---:|---:|---:|---:|
| Splicing | 190 | 304 | 303 | **434** |
| Chromatin | 145 | 468 | 467 | **443** |
| Transport | 42 | 70 | 70 | **67** |
| **TOTAL** | **377** | — | — | **`|E_WASC| = 944`** |

Within-theme densities (sanity): Splicing 2.4%, Chromatin 4.2%, Transport 7.8% — all biologically plausible (Transport is dense because nuclear-pore + transport machinery is heavily annotated; Splicing is sparse because of the size of the spliceosome union).

This count is **frozen** in `data/wasc/E_WASC_v1.json` at M6a (pre-registration item E4).

---

## 2. Per-edge per-group regression

For each edge `(a, j) ∈ E_WASC` and each group `g ∈ G = {C9, SPOR, CTRL}`, fit a single OLS:

```
y_{j, s} = β_0 + β_a · anchor_{a, s} + γ_Sex · Sex_s + γ_Age · Age_s
 + γ_Tissue · Tissue_s + ε_s for s ∈ donors(g)
```

with the contrast vector `c = (0, 1, 0, 0,...)` zero-padded for all covariates, so that the SE targets the anchor-slope coefficient `β_a` only. The estimate of interest is `β̂_{j|a, g}` (the partial regression slope of target on anchor, within group g, adjusted for covariates).

### 2.1 Response and Batch handling

`y_{j, s}` = pre-residualized log2 abundance of target protein j in donor s, defined as:

```
y_{j, s} = log2_abundance_{j, s} − ComBat_location_{j, batch*(s)}^{(g)}
 / ComBat_scale_{j, batch*(s)}^{(g)}
```

That is: within each group g, for each target protein j, residualize Batch via ComBat-style EB location/scale adjustment fit on the within-group donors only. The pre-residualization is the SAME for every edge whose target is j and whose group is g, so it is computed once per `(g, j)` pair and reused across all anchors a.

**C9-Batch collapse (brutalist mod 5, PRE-REGISTERED, item E9).** The raw Batch field has up to 50 levels across 25 C9 donors, producing singleton batches that defeat ComBat EB estimation. **For the C9 group only**, Batch is collapsed to `site_year(s) := concat(donor_site(s), collection_year(s))` BEFORE ComBat is fit. The mapping `Batch_raw → site_year` is computed at pre-registration time and saved to `data/wasc/c9_batch_collapse_v1.json`. SPOR and CTRL retain their raw `Batch` because their donor counts (294, 71) make singleton batches rare; the audit at M6a will verify ≥3 donors per batch in those groups or extend the collapse rule accordingly.

`anchor_{a, s}` = `log2_abundance_{a, s}` z-scored within group g (mean-centered, unit variance). Z-scoring within-group makes `β̂_{j|a,g}` interpretable as "1-SD-of-anchor change predicts β̂ SD-of-target change in group g" and makes the across-group comparison scale-comparable.

### 2.2 Covariates (from the metadata audit)

- **Sex:** binary (Male=0 / Female=1). 100% coverage. Fit directly.
- **Age** = `Age_at_First_PBMC_Collection`, z-scored within group g. Missing values imputed by linear regression on Sex *within group g* (pre-registered imputation rule; see `memory/wave_24l_measured_only_paths.md`). 91.3% pre-imputation coverage.
- **Tissue** = `Primary_Tissue` collapsed to 3 levels (T-Cell / NT-Cell+Bulk / Unknown), dummy-coded with T-Cell as reference (drop_first=True via `build_covariate_design_matrix`). This is the only available cell-composition proxy and is heavily group-confounded (C9 92% T-Cell, SPOR 65%, CTRL 38%), making it load-bearing.
- **Batch:** PRE-RESIDUALIZED (see §2.1, with C9 site_year collapse), NOT a column of X.
- **Excluded donors:** the 20 external/iPSC-derived controls (EDi*, CW50*, CS007, W14-16C* prefixes) have no Batch ID and no portal coverage; they are EXCLUDED from all WASC analyses (PRE-REGISTERED, item E7). Net C9 = 25, SPOR ≈ 294, CTRL ≈ 71 (post-exclusion).

### 2.3 Missingness handling

For each `(j, a, g)`, define `S_{j, a, g}` = donors s in group g where:
- y_{j,s} is observed (after pre-residualization), AND
- anchor_{a,s} is observed, AND
- Sex_s, Age_s (post-imputation), Tissue_s are all defined, AND
- s is not in the 20 excluded external/iPSC donors.

Fit OLS on `S_{j, a, g}` only. Require `|S_{j, a, g}| ≥ 10` for C9 and `|S_{j, a, g}| ≥ 15` for SPOR/CTRL; otherwise the per-group fit is marked degenerate.

### 2.4 Output per (j, a, g)

- `β̂_{j|a, g}` — the anchor-slope point estimate
- `SE(β̂_{j|a, g})` = `√(σ̂²_{g,(j,a)} · [(X_g^T X_g)^{-1}]_{a,a})` where the `[a,a]` element corresponds to the anchor column (column 1, after intercept)
- `df_{g, (j, a)}` = `|S_{j, a, g}| − rank(X_g)`
- `converged`: True iff `|S_{j, a, g}|` thresholds met AND `X_g` full rank AND `SE > 0`

**Variance moderation (optional sensitivity):** apply EB shrinkage (`fit_f_dist` + `squeeze_var` from `permutation_gpu.py`) across all `(j, a)` pairs *within group g* to stabilize small-sample C9 variances. Report Q both with and without EB; the primary test uses the un-moderated SE (because EB shrinkage across edges introduces dependence between edges and complicates the Q null distribution).

### 2.5 Implementation pattern

Use Frisch-Waugh-Lovell to avoid O(|E| × |G|) full OLS:

1. Per group g, build the covariate-only design `X_g^{cov} = [intercept | Sex | Age | Tissue_dummies]` once. Compute the residual projection `M_g = I − X_g^{cov} (X_g^{cov T} X_g^{cov})^{-1} X_g^{cov T}` once. This is O(n_g²) per group.
2. For each protein p ∈ M (anchors and targets), compute `p̃_g = M_g @ z(p)_g` once — the covariate-residualized within-group expression vector for p. Cache.
3. For each edge `(a, j)`, the anchor-slope and SE become the closed-form univariate regression of `j̃_g` on `ã_g`:
 - `β̂_{j|a, g} = (ã_g^T j̃_g) / (ã_g^T ã_g)`
 - `RSS = j̃_g^T j̃_g − β̂² · (ã_g^T ã_g)`
 - `σ̂² = RSS / (n_g − rank(X_g^{cov}) − 1)`
 - `SE = √(σ̂² / (ã_g^T ã_g))`

This makes the WASC fit phase O(|M| · n_g) for residualization + O(|E_WASC| · n_g) for the per-edge dot products — fast enough for 944 edges × 3 groups × (25 + 294 + 71) samples on CPU without GPU. The anchor loop in M2 is `joblib.Parallel(n_jobs=-1)` over anchors (PRE-REGISTERED, brutalist mod 4); single-threaded scale is infeasible at `|E_WASC| = 944`.

Reuse: `build_covariate_design_matrix` (`design_matrix.py:70`) for X_g^{cov} construction; reuse the numerical guards (rank check, condition number). Do NOT reuse `precompute_ols_matrices` (the batched paths assume shared X across the batch; the WASC kernel has shared X within group but different per-edge regression — Frisch-Waugh decouples this cleanly).

**M1 numerical reference tests (brutalist mod 7, completed at M1 tag time, recorded in `data/wasc/m1_numerical_reference_v1.json`):**
- **(a) Frisch-Waugh ≡ statsmodels.OLS.** On 50 randomly sampled real `(j, a, g)` triples, the absolute difference `|β̂_{FW} − β̂_{statsmodels}|` and `|SE_{FW} − SE_{statsmodels}|` must each be ≤ `1e-8`. M1 result: **PASS** (max abs diff `< 1e-10`).
- **(b) Swap invariance.** On the same 50 triples, refit with `a` and `j` swapped (regress anchor on target instead of target on anchor) and recompute Q. If `|Q_{j~a} − Q_{a~j}| / Q_{j~a} > 5%` on the median triple, the slope-symmetry caveat below is appended verbatim to §9. M1 result: **PASS** (median drift 0.8%, max 3.2% on real noisy triples); §9 caveat NOT triggered.

---

## 3. Concordance statistic

Define the precision weights `w_{g, j, a} = 1 / SE(β̂_{j|a, g})²` for each `(j, a, g)` with converged regression. Let `G_obs(j, a) = {g : (j, a, g) converged}`.

**Primary edge statistic:**

```
β̄_{j, a} = Σ_{g ∈ G_obs} w_{g, j, a} · β̂_{j|a, g} / Σ_{g ∈ G_obs} w_{g, j, a}

Q(j, a) = Σ_{g ∈ G_obs} w_{g, j, a} · (β̂_{j|a, g} − β̄_{j, a})²
```

This is Cochran's Q from inverse-variance-weighted meta-analysis. Under H0 of identical true slopes across groups (`β_{j|a, g} = β_{j, a}` for all g) and exact-Normal sampling, `Q ~ χ²_{|G_obs|−1}`. We do NOT use the χ² calibration; we use the empirical null from 3-axis-matched non-neighbor pairs (§4).

**Interpretation:** **LOW Q = invariant coupling across groups = WASC-positive.** HIGH Q = group-dependent coupling.

**Primary edge inclusion criterion:** `|G_obs(j, a)| = 3` (all three groups converged). Edges with `|G_obs| < 3` are reported in a SECONDARY two-group analysis and excluded from the primary BY-FDR pool.

**Descriptive auxiliaries (reported, not tested):**
- `I²(j, a) = max(0, (Q − (|G_obs|−1)) / Q)`
- `τ̂²(j, a)` = DerSimonian-Laird random-effects variance estimate
- `direction-pattern(j, a)` = sign tuple `(sign(β̂_{j|a, C9}), sign(β̂_{j|a, SPOR}), sign(β̂_{j|a, CTRL}))` for narrative reporting only

**Rationale for choosing Q over alternatives:**
- **Kendall's W (unweighted ordinal):** rejected — discards the precision asymmetry between n=25 C9 and n=294 SPOR; small samples should not vote equally with large.
- **I²:** monotone-derivable from Q but bounded in [0, 1]; compresses signal range, saturates at 0 for many edges, complicates permutation null reuse.
- **τ² (DerSimonian-Laird):** requires Q − df as intermediate; adds method-of-moments step whose small-k=3 sampling distribution is non-pivotal.
- **Q:** pivotal under exact H0, directly weights by precision (handles n=25 vs n=294 asymmetry as required), works on continuous scale that survives permutation reshuffling.

---

## 4. Null model

The Q statistic's permutation null is constructed **per anchor**, preserving anchor-degree, target-coverage, **and pooled marginal correlation with the anchor** (brutalist mod 1). The two-axis (degree × missingness) null was over-conservative because within-theme cluster members co-express by pathway selection, inflating the substitute targets' marginal correlation with the anchor relative to true neighbors; the third axis corrects this.

For each anchor `a ∈ M_T` that participates in at least one edge in `E_WASC` and each theme T:
1. Let `N_a^obs` = set of true measured within-theme INDRA hop-1 neighbors of a in theme T (the targets of a's WASC edges in theme T). Let `n_a = |N_a^obs|`.
2. Let `P_a^candidate` = `M_T \ N_a^obs \ {a}` (measured cluster members in theme T, excluding a's true neighbors and a itself).
3. Bin proteins in `P_a^candidate` on a **3-D grid**:
 - **Axis 1 — degree decile** within the INDRA measured-only regulatory graph. Degree is the full regulatory degree from `query_gene_degrees_batched`, cached in `distances.meta.json`.
 - **Axis 2 — missingness decile** within the proteomics matrix. Missingness rate is `missing_per_feature / n_samples_total` from `analyze_missing_values`.
 - **Axis 3 — pooled |Pearson(anchor_a, p)| decile**, where the Pearson correlation is computed on the union of all three groups' donors (after the same Sex/Age/Tissue residualization used in §2 but BEFORE the per-group covariate-adjustment), restricted to donors where both `anchor_a` and `p` are observed. Bin edges are computed per-anchor (deciles of `|r(a, p)|` over `p ∈ M_T \ {a}`).
4. For each true neighbor `j ∈ N_a^obs`, identify its (degree-decile, missingness-decile, |r|-decile) cell.
5. **Per-anchor permutation:** for `b = 1,..., B`:
 - Draw n_a substitute targets without replacement from `P_a^candidate`, with the constraint that the multiset of (degree-decile, missingness-decile, |r|-decile) cells matches that of `N_a^obs` exactly. If a perfect 3-D match draw fails after 100 attempts, fall back to a relaxed match where the |r|-decile constraint is widened by ±1 decile (record the relaxation per-anchor in the null-diagnostics manifest). If still failing after 100 further attempts, mark anchor a's permutation as failed for iteration b (record but do not impute).
 - For each substitute target `j'`, fit the per-group regressions §2, compute `Q^{(b)}(j', a)` exactly as for the real edges.

6. Pool null Q values **across all anchors and all iterations** to form the global null distribution `Q_null = {Q^{(b)}(j', a) : a ∈ anchors, b ∈ 1..B, j' ∈ substitute set b}`. Total size: `|E_WASC| × B = 944 × 9999 ≈ 9.4 × 10^6`.

**Per-edge p-value:**
```
p(j, a) = (1 + #{Q ∈ Q_null : Q ≤ Q(j, a)}) / (1 + |Q_null|)
```
(one-sided, lower tail; small Q = invariant = WASC-positive). The "+1 / +1" is the Phipson-Smyth correction.

**Per-anchor null pooling (PRE-REGISTERED as primary):** pool null Qs only within the same anchor a and same `n_a`, i.e., `Q_null^a = {Q^{(b)}(j', a) : b ∈ 1..B, j' ∈ substitute set b}`. This gives an anchor-local p-value that controls for anchor-specific noise structure. The global pool is reported as a sensitivity. The primary p-value uses the anchor-local pool.

**B (number of permutations):** **`B = 9999` for the primary run.** Rationale: with `|E_WASC| = 944` edges, the smallest resolvable per-edge p-value is `1 / (B + 1) ≈ 10^{-4}`. After BY-FDR (§6) at q = 0.10 the most stringent rejection threshold is `q · (1/H_n) ≈ 0.10 / 7.5 ≈ 0.013` (`H_{944} ≈ 7.5`); B = 9999 provides ~7× safety margin against tied-rank ambiguity. **Sensitivity runs may use `B = 999`** (1/1000 floor) to fit the ~120–200 h sensitivities batch budget. If the primary BY q-table is **floor-tied** (i.e., the smallest p-value equals `1/(B+1)` at B=9999), a single `B = 99999` rerun is triggered (tertiary, pre-registered, item E10 in §10).

**Three-axis binning rationale:** the audit confirms degree-only matching was the prior practice; missingness (Axis 2) was added in v0.9; pooled |Pearson| (Axis 3) is new in v1.0 in response to the brutalist's over-conservativeness diagnosis. Within-cluster proteins co-express by pathway selection, so unmatched substitutes systematically have lower marginal correlation with the anchor than true neighbors — making the null too easy to beat. Decile-matching on `|r(a, p)|` neutralizes this and produces a strictly more conservative null than 2-axis matching.

**Implementation note:** the new sampling primitive `sample_n_3axis_matched_non_neighbors(anchor, n, joint_bins_3d, exclude_set, rng) -> list[str]` is in `src/cliquefinder/stats/wasc/null.py`. The per-anchor RNG seeding pattern (`md5(anchor_uniprot + "wasc-v1.0" + iteration)`) from `_per_feature_gradient_loop` (`landscape.py:1033`) is reused for reproducibility under multi-process parallelism. Parallelism via `joblib.Parallel(n_jobs=-1)` over anchors (PRE-REGISTERED, brutalist mod 4).

---

## 5. Per-anchor combination

Each anchor a contributes `n_a` edge tests. To combine into a per-anchor statistic, use **empirical Brown's method** (Poole et al. 2016).

```
χ²_a (combined) = −2 · Σ_{j ∈ N_a^obs} log(p(j, a))

E[χ²_a] = 2 · n_a
Var[χ²_a] = 4 · n_a + 2 · Σ_{i ≠ j} cov_emp(−2 log p(i, a), −2 log p(j, a))

c_a = Var[χ²_a] / (2 · E[χ²_a])
df_a = 2 · E[χ²_a]² / Var[χ²_a]

p_a^Brown = Pr(χ²_{df_a} ≥ χ²_a / c_a)
```

The covariance `cov_emp` is estimated **from the null distribution** of `(−2 log p(i, a), −2 log p(j, a))` pairs across the B permutation iterations — this is the "empirical" variant and correctly captures within-anchor dependence (the null tests share donors, share anchor expression vector, share covariate design). Brown's method reduces to Fisher's combination if the covariance is zero.

**Pre-registered combination decision:** Brown's chosen because (a) within-anchor edge tests are positively correlated through the shared anchor expression vector, (b) the audit confirms no Brown's implementation exists in the repo — this is new code with the empirical estimator landing in `src/cliquefinder/stats/combination.py`, re-exported through `stats/__init__.py`. Fisher's method is reported as a sensitivity and would over-state significance under positive dependence.

**Reporting:**
- Primary table 1: per-edge `(j, a, Q, p, q_BY)` for all `(j, a) ∈ E_WASC`.
- Primary table 2: per-anchor `(a, n_a, χ²_a, p_a^Brown, q_a^BY)` for all anchors.
- Per-edge and per-anchor analyses are reported in PARALLEL, not as a hierarchy. The primary hypothesis is on the per-edge BY-FDR table (§6); per-anchor combination is reported alongside as a complementary view.

---

## 6. Multiple testing

**Primary:** Benjamini-Yekutieli FDR over the per-edge p-values, q-threshold **q = 0.10**.

```
q(j, a) = BY-adjusted p(j, a) via fdr_correction(p_values, method="BY")
```

Edge is "WASC-positive" iff `q(j, a) ≤ 0.10`.

**Rationale for BY over BH:** the per-edge p-values are positively dependent through (a) shared anchors (same X column reused), (b) shared targets (same y vector reused across multiple anchor regressions), (c) the within-anchor null draws are themselves dependent. BH controls FDR only under PRDS (positive regression dependency on a subset), which the WASC p-value graph does NOT satisfy generally — anchor-sharing produces dependence between edges with the SAME anchor that is not strictly monotone in any natural ordering. BY controls FDR under arbitrary dependence at the cost of an inflation factor `H_n = Σ_{i=1}^{n} 1/i` (for `n = |E_WASC| = 944`, `H_n ≈ 7.5`). The effective per-edge threshold at q = 0.10 is therefore approximately `0.10 / 7.5 ≈ 0.013` at the smallest rank.

**Implementation:** `fdr_correction(pvalues, method="BY", alpha=0.10)` from `src/cliquefinder/stats/differential.py:242`. No new code; BY is already wired through.

**q-threshold = 0.10 (not 0.05), framed as "count + cluster pattern":** WASC is an exploratory invariance test of structural equality, not a confirmatory mean-shift gate. The **primary claim is the COMPLEX — the count of WASC-positive edges and their per-theme cluster-membership pattern — not any single per-edge mechanism** (brutalist mod 3, user-confirmed M1 decision). At q = 0.05 with BY's inflation, power for 944 edges with n=25 in C9 is dangerously low; q = 0.10 is the conventional FDR threshold for exploratory genomics. The threshold is PRE-REGISTERED at q = 0.10 and cannot be changed post-hoc.

**Per-anchor BY:** apply `fdr_correction(p_a^Brown_values, method="BY", alpha=0.10)` independently on the per-anchor table.

**Sensitivity:** report results at q = 0.05 and q = 0.20 as descriptive, but the binary decision rule uses q = 0.10.

**Effective-tests diagnostic (secondary):** compute `estimate_effective_tests(correlation_matrix_of_edge_p_values)` from `stats/correlation_tests.py:628` for transparency on the dependence structure, but do not use M_eff to lower the threshold.

---

## 7. STRING-PPI negative control

PRE-REGISTERED. Same anchor set, same regression spec (§2), same Q statistic (§3), same null model (§4), same FDR procedure (§6) — only the edge set changes.

### 7.1 STRING edge set construction

Implement (audit gap — no loader exists):

1. Load `data/string/9606.protein.physical.links.v12.0.txt.gz`. Filter rows to `combined_score ≥ 700`. Result: ≈ 173,038 edges among ≈ 10,746 ENSP.
2. **Map ENSP → UniProt CANONICAL ONLY** (brutalist mod 3). For each ENSP, the canonical UniProt accession is the **reviewed Swiss-Prot primary accession** for the gene that the ENSP belongs to, as resolved by `mygene` (`species=human`, `fields=uniprot.Swiss-Prot`) with the unreviewed/isoform accessions discarded. This avoids isoform double-counting that inflates STRING density (multiple ENSP isoforms of the same gene would otherwise each contribute an edge per partner). If `mygene` returns multiple Swiss-Prot accessions, take the alphabetically-first **after** filtering to reviewed entries; the mapping is recorded with the version date in `data/wasc/string_ensp_to_uniprot_canonical_v1.json` (E5).
3. Collapse to UniProt adjacency under the canonical-only policy: there is an edge between canonical UniProts u1, u2 iff there exists an ENSP-edge (e1, e2) with `canonical_UniProt(e1) = u1` and `canonical_UniProt(e2) = u2`. Expected measured-coverage: comparable to T43's ~71% but somewhat lower because isoform-only matches are dropped.

### 7.2 STRING edge set for WASC negative control

```
E_STRING = ⋃_{T ∈ Themes} { {a, j} : a, j ∈ M_T, a ≠ j, dist_STRING(a, j) = 1 }
```

Same theme-purity and within-cluster constraints as INDRA. Edge count will be different (STRING physical density differs from INDRA regulatory density on these proteins).

### 7.3 Decision rule

**Primary STRING effect-size (brutalist mod 2): ΔQ on the anchor-pair edge-INTERSECTION.** Let `E_BOTH = E_WASC ∩ E_STRING` (edges that appear in *both* networks over the WASC anchor pool). Compute the mean (or median) Q over `E_BOTH` separately under the INDRA regression (`Q̄_INDRA^{∩}`) and under the STRING regression (`Q̄_STRING^{∩}` — note: same regression spec, same data; only the null-set membership differs, but for the intersection-effect we compute Q directly on `E_BOTH` and compare its distribution to the corresponding intersection-restricted null under each network). The intersection-restricted effect-size is:

```
ΔQ^{∩} = Q̄_STRING^{∩} − Q̄_INDRA^{∩} (positive = INDRA more invariant on shared edges)
```

This is **robust to the zero-positives degenerate case** because it does not depend on either network having a non-empty WASC-positive set — it compares the full Q distributions on the same edge population.

**Secondary STRING effect-size: median Q across WASC-positive edges.** As in v0.9: compute `Q̃_INDRA` over INDRA's BY-positive edges and `Q̃_STRING` over STRING's BY-positive edges. The effect-size is `ΔQ̃ = Q̃_STRING − Q̃_INDRA`. **This statistic is UNDEFINED when STRING has zero BY-positive edges** and triggers the `STRING-ZERO-POSITIVES` branch below.

Confidence intervals on both ΔQ-statistics via 1000-resample BCa bootstrap over edges (within each network independently, then differenced). For the intersection effect-size, the bootstrap resamples edges of `E_BOTH`.

**Decision rule (PRE-REGISTERED, FIVE BRANCHES):**

| Branch | Trigger | Interpretation |
|---|---|---|
| **INDRA-SPECIFIC** | Primary `ΔQ^{∩}` BCa 95% CI lower bound > 0 AND STRING coverage gate (§12) passes AND STRING has ≥ 1 BY-positive edge | INDRA-specific coupling-invariance interpretation is SUPPORTED. Secondary `ΔQ̃` is reported for context. |
| **STRING-STRONGER** | Primary `ΔQ^{∩}` BCa 95% CI upper bound < 0 | The result is STRING-stronger. INDRA-specific interpretation is **REJECTED**. Result is reported as a generic within-cluster coupling-structure finding without regulatory-edge framing. |
| **INCONCLUSIVE** | Primary `ΔQ^{∩}` BCa 95% CI straddles 0 AND STRING coverage gate passes AND STRING has ≥ 1 BY-positive edge | Edge-source specificity is undetermined. Both networks' Q distributions are reported side-by-side. Do NOT claim INDRA-specific wiring. |
| **STRING-UNDERPOWERED** | STRING measured-coverage on cluster members < 60% of INDRA's coverage (per §12 gate 3) | STRING control is statistically uninformative. Result section reports STRING as underpowered; no INDRA-specific claim from STRING is licensed. |
| **STRING-ZERO-POSITIVES** | STRING has 0 BY-positive edges (secondary `ΔQ̃` undefined) BUT STRING coverage gate passes | The primary `ΔQ^{∩}` BCa CI is still the decision input, but the **`STRING-ZERO-POSITIVES` branch EXPLICITLY FORBIDS the "INDRA-specific by exclusion" interpretation** even if `ΔQ^{∩}` lower bound > 0. Report as: "STRING returned zero WASC-positive edges; the intersection-effect-size remains the only quantifiable comparison and is reported, but no claim of INDRA-edge-source specificity by exclusion can be made on the secondary statistic alone." The phrasing "STRING has no WASC-positive edges, therefore the result must be INDRA-specific" is **explicitly forbidden** as it confuses absence of evidence with evidence of absence at the BY-positive level. |

**Branch precedence:** STRING-UNDERPOWERED > STRING-ZERO-POSITIVES > {INDRA-SPECIFIC, STRING-STRONGER, INCONCLUSIVE based on `ΔQ^{∩}` CI}.

**Secondary tabulation:** report the count of edges WASC-positive in INDRA, in STRING, and in `E_BOTH`. If the BY-significant edge sets are disjoint or anti-correlated (as T43's slope-GSEA was), this strengthens the INDRA-specific reading qualitatively but does not override the primary intersection-`ΔQ^{∩}` decision.

**Rationale:** T43's slope-GSEA showed STRING and INDRA give *opposite-sign* GSEA enrichments on the same proteomics |t|. WASC must therefore be tested on both networks to determine whether the coupling-invariance result is a property of the INDRA regulatory-edge curation specifically or a generic within-cluster topology effect. Without this control, the "regulatory wiring" interpretation is unwarranted. The intersection-`ΔQ^{∩}` primary statistic is a brutalist response to the over-reliance on median-of-positives in v0.9, which is undefined precisely when the most informative degenerate case (STRING-ZERO-POSITIVES) occurs.

---

## 8. Three-contrast decomposition

PRE-REGISTERED. Before claiming "C9-specific coupling-structure shift," decompose the per-edge invariance into three pairwise contrasts:

For each edge `(j, a)`, compute three two-group Q statistics:
- `Q^{C9-SPOR}(j, a)` = inverse-variance-weighted Q on `{β̂_{j|a, C9}, β̂_{j|a, SPOR}}` (with k=2 groups)
- `Q^{C9-CTRL}(j, a)` = same for `{C9, CTRL}`
- `Q^{SPOR-CTRL}(j, a)` = same for `{SPOR, CTRL}`

For k=2 groups, Cochran Q reduces to `Q = (β̂_1 − β̂_2)² / (SE_1² + SE_2²)` (the squared standardized difference). Get its permutation p-value `p^{XY}(j, a)` from the corresponding two-group null draws (same 3-axis matching, but restricted to k=2).

Apply BY-FDR within each contrast separately at q = 0.10. Let:
- `R_{C9-SPOR}` = set of edges with `q^{C9-SPOR}(j, a) ≤ 0.10`
- `R_{C9-CTRL}` = set of edges with `q^{C9-CTRL}(j, a) ≤ 0.10`
- `R_{SPOR-CTRL}` = set of edges with `q^{SPOR-CTRL}(j, a) ≤ 0.10`

**C9-specific coupling-shift criterion (PRE-REGISTERED, ALL conditions required):**

- **(C1)** Primary 3-group test passes: there exist edges with `q(j, a) ≤ 0.10` after primary BY-FDR (§6).
- **(C2, empirical floor 48)** Both of the following hold:
 - `|R_{C9-SPOR}| ≥ max(3 · |R_{SPOR-CTRL}|, 48)`
 - `|R_{C9-CTRL}| ≥ max(3 · |R_{SPOR-CTRL}|, 48)`

 The floor **48 = `ceil(0.05 · |E_WASC|) = ceil(0.05 · 944)`** (brutalist mod 4, M1 user decision 2). The 3× ratio echoes the wave_24j 50× drop framing, scaled to the WASC pool; the absolute floor prevents vacuous 3-of-3 vs 1-of-3 passes when the contrast counts are small.
- **(C3)** STRING control (§7) returns `ΔQ^{∩}` BCa lower bound > 0 AND the resolved branch is `INDRA-SPECIFIC` (i.e., not `STRING-STRONGER`, `INCONCLUSIVE`, `STRING-UNDERPOWERED`, or `STRING-ZERO-POSITIVES`).
- **(C4)** `|R_{SPOR-CTRL}|` is not significantly elevated above the FDR null expectation: a one-sided binomial test on `|R_{SPOR-CTRL}|` vs `E[|R| | H0] = 0.10 · |E_WASC| = 94.4` should NOT reject at α = 0.05.

If C1 holds but C2-C4 do not all hold: the result is a within-cluster coupling-structure shift that is NOT cleanly C9-specific. Report it as such; do NOT use "C9-mutation-specific" language. The wave_24j precedent (which retracted C9-specific framing once SPOR-vs-CTRL hits appeared) is binding.

---

## 9. Claim ceiling

A positive WASC result (all of C1-C4 satisfied AND STRING branch resolves to `INDRA-SPECIFIC`) licenses the following sentence and NO MORE (**brutalist-revised verbatim**, brutalist mod 8):

> Within the 8 pre-registered C9-ALS cluster terms, the **cross-protein abundance coupling** of INDRA hop-1 neighbor pairs — quantified by inverse-variance-weighted between-group dispersion of per-edge partial regression slopes adjusted for sex, age, primary tissue, and batch — is significantly more invariant across C9, Sporadic, and Control donors than degree-, coverage-, and marginal-correlation-matched non-neighbor pairs in the same cluster (BY-FDR q < 0.10). The invariance is preserved across the SPORADIC-vs-CONTROL contrast while breaking selectively in C9-containing contrasts. The finding is INDRA-edge-specific: a parallel analysis on the STRING physical-PPI graph over the same anchors does not reach the same magnitude on the edge-intersection (ΔQ BCa lower bound > 0). This is consistent with multiple non-mutually-exclusive scenarios — including regulatory rewiring, differential post-translational processing, differential cell-composition not captured by Primary_Tissue, or differential measurement-noise structure — and does not distinguish among them. The result is descriptive of slope-shift structure only.

**What WASC CANNOT support, even with a maximal positive result:**
- It does not establish **mechanism**.
- It does not establish **causation** in any direction.
- It does not show that any particular INDRA edge **drives**, **regulates**, or **controls** the abundance of its partner.
- It does not validate INDRA edges as true in-vivo regulatory relationships.
- It does not adjudicate **post-transcriptional** vs transcriptional vs translational origin of the coupling shift.
- It does not show that the coupling shift is a **rewiring** of regulatory architecture (the term "rewiring" is reserved for results with paired perturbation evidence; coupling-structure shift on observational data does not warrant it).
- It cannot resolve confounding by **granular cell-composition** (T-CD4/CD8/B/NK/monocyte fractions) beyond the `Primary_Tissue` 3-level proxy; residual cell-composition shift across groups remains a stated limitation.

**Forbidden language list (PRE-REGISTERED, brutalist mod 8):**

The following words and phrases are **forbidden** in any reporting of WASC results regardless of outcome:
- *mechanism*
- *causal*, *causation*
- *drives*, *regulates*, *controls*
- *rewiring* (use only "coupling-structure shift" or "slope-shift structure")
- *validates* the cluster (WASC is a coupling-invariance test, not a per-protein or per-edge validation)
- *post-transcriptional* (out of scope; not adjudicated by the test)
- *INDRA-edges-are-correct* (they are an analysis substrate, not a validated ground truth)

**Per-edge claim ceiling:** at the per-edge level, a WASC-positive edge can only be described as "having passed the structural-coupling-invariance test in the pool of 944 pre-registered cluster edges at BY q ≤ 0.10." It is **NOT** described as a mechanism, a regulatory relationship, or a validated edge. The primary claim is at the cluster-pattern level (count + per-theme breakdown), per the M1 user decision 7.

**Slope-symmetry caveat (conditional, M1 verified PASS):** the M1 swap-invariance test (§2.5(b)) returned median Q-drift 0.8% (well below the 5% threshold). The slope-symmetry caveat is therefore NOT triggered in v1.0. Were a future re-derivation on a different dataset to fail the 5% threshold, the following sentence would be appended verbatim to the claim ceiling: "*Q is sensitive to the choice of regression direction at the >5% level on this dataset; per-edge Q values should be interpreted with a slope-asymmetry caveat.*"

---

## 10. Pre-registration items

The following must be locked in a tagged git commit `wasc-prereg-v1.0` prior to any M2-onward compute. Re-running the pipeline after this commit MUST reproduce identical inputs.

**Frozen inputs (E1–E9, brutalist mod 8 added E9):**

- **E1.** The 8 cluster terms (already frozen in `scripts/viz/common.py:20::TERMS`).
- **E2.** The cluster-member set per theme `{C_T : T ∈ Themes}` computed at commit time and saved to **`data/wasc/cluster_members_v1.json`** (frozen UniProt sets, with provenance: cogex query timestamp 2026-06-02, INDRA HGNC mapping date, per-theme counts: Splicing 304 UniProt / 303 HGNC / 190 measured, Chromatin 468 / 467 / 145, Transport 70 / 70 / 42).
- **E3.** The Wave-24l measured-only INDRA distance matrix snapshot (re-use `output/landscape_c9_vs_*/distances.meta.json` from the existing wave_24l runs; SHA-256-pin its file hash).
- **E4.** The exhaustive enumeration `E_WASC` (UniProt-pair list) computed and frozen, saved to **`data/wasc/E_WASC_v1.json`**, with edge count `|E_WASC| = 944` recorded and per-theme breakdown (434 / 443 / 67).
- **E5.** The STRING v12.0 graph: file hash of `data/string/9606.protein.physical.links.v12.0.txt.gz`, ENSP→**canonical-UniProt** mapping version (mygene query date, Swiss-Prot-reviewed-only policy), `combined_score ≥ 700` threshold, resulting UniProt adjacency saved to `data/wasc/string_uniprot_adj_v1.json` plus `data/wasc/string_ensp_to_uniprot_canonical_v1.json`.
- **E6.** The STRING-derived `E_STRING` enumeration over `M_T` for each theme, frozen.
- **E7.** The 20-donor exclusion list (external/iPSC-derived, no Batch ID) — saved to `data/wasc/excluded_donors_v1.json`.
- **E8.** The Age-imputation model coefficients per group g (regression-on-Sex within-arm), saved to `data/wasc/age_imputation_v1.json`.
- **E9. (NEW, brutalist mod 5)** The C9-Batch collapse mapping `Batch_raw → site_year` for all 25 C9 donors, saved to **`data/wasc/c9_batch_collapse_v1.json`**, with the policy note: SPOR and CTRL retain `Batch_raw` pending the M6a audit; if any SPOR/CTRL batch has < 3 donors after exclusions, this manifest is extended to cover the affected group and re-saved before tag.

**Frozen analysis decisions (D1–D10):**

- **D1.** Primary test: per-edge Cochran Q on 3-group inverse-variance-weighted slopes (§3).
- **D2.** Permutation `B = 9999` for primary; `B = 999` floor permitted for sensitivities; `B = 99999` rerun triggered on floor-tied primary q-table.
- **D3.** Null model: anchor-local, **3-axis** (degree-decile × missingness-decile × pooled-|Pearson|-decile) matched, sampled WITHOUT replacement (§4).
- **D4.** FDR: BY method, q-threshold 0.10 (§6).
- **D5.** Per-anchor combination: empirical Brown's method (§5).
- **D6.** STRING decision rule: primary `ΔQ^{∩}` with BCa 95% CI, 5-branch decision (§7); `STRING-ZERO-POSITIVES` forbids INDRA-by-exclusion.
- **D7.** Three-contrast C9-specific criterion: C1 ∧ C2 ∧ C3 ∧ C4 with the empirical floor `|R_{C9-X}| ≥ max(3·|R_{SPOR-CTRL}|, 48)` (§8).
- **D8.** Claim ceiling: §9 verbatim (no modification permitted post-hoc). Forbidden language list enforced.
- **D9.** Covariate design: `[1, anchor_z, Sex, Age_z, Tissue_dummies]` with Batch pre-residualization via within-group ComBat, using C9-Batch collapsed to `site_year` (§2.1).
- **D10.** Random-number generation: per-anchor seed = `int(md5(uniprot + "wasc-v1.0").hexdigest[:8], 16)`, per-iteration seed derived via `np.random.SeedSequence`.

**Primary outcome (singular, brutalist-explicit):**

> **Count of WASC-positive edges (BY-FDR q ≤ 0.10) in the 3-group test, with the per-theme breakdown `(|R ∩ E_Splicing|, |R ∩ E_Chromatin|, |R ∩ E_Transport|)`, against the pre-registered context `|E_WASC| = 944` (434 / 443 / 67).** The primary claim is the count + cluster pattern, not any per-edge mechanism.

**Secondary outcomes (pre-registered, do not affect primary inference):**

- Per-anchor BY-FDR q-values (Brown's combination).
- Three-contrast decomposition counts `|R_{C9-SPOR}|`, `|R_{C9-CTRL}|`, `|R_{SPOR-CTRL}|` and the C1-C4 evaluation.
- STRING control primary `ΔQ^{∩}` with BCa 95% CI; secondary median-of-positives `ΔQ̃` with BCa 95% CI; resolved branch.
- Direction-pattern table (sign triples) for WASC-positive edges.
- I² and τ̂² descriptive distributions for WASC-positive edges.

**Tertiary (PRE-REGISTERED, run regardless of primary outcome; brutalist mod 6):**

These sensitivities are **mandatory** and reported alongside the primary regardless of whether the primary returns positive. Sensitivity runs may use `B = 999` (compute-budget concession; primary stays at `B = 9999`).

| Tertiary | Description | Purpose |
|---|---|---|
| **T-Cell-stratified** | Restrict all three groups to `Primary_Tissue == T-Cell` donors only, rerun WASC primary | Tests whether the result is driven by cell-composition confounding |
| **iPSC-retained** | Re-include the 20 external/iPSC-derived donors (with `iPSC` Tissue level added to design) | Tests donor-pool sensitivity |
| **Batch-correction-OFF** | Skip §2.1 ComBat pre-residualization | Tests whether the ComBat EB step is generating false invariance |
| **Down-sampled-SPOR-to-25** *(MOST IMPORTANT)* | Random-subsample SPOR to n=25 (matching C9), rerun WASC primary, B=999, repeat 20 subsamples and report distribution | Detects n-asymmetry attenuation: if the primary positive disappears at n_SPOR=25, the result is partly an artifact of the precision asymmetry between groups |
| **All-protein-pool null** | Substitute draws from ALL measured proteins (not just `M_T`), maintaining 3-axis match | Tests whether the within-theme null is over-restrictive |
| **B=99999 if floor-tied** | If primary smallest p-value equals `1/(B+1)` at B=9999, rerun the affected anchors at B=99999 | Resolves rank-tie at BY threshold |
| **EB-moderated SE** | Apply `fit_f_dist` + `squeeze_var` across edges within group g | Reported as descriptive; not primary |
| **Cross-theme exploratory** (DEFERRED) | Out of M1 scope; reported in a future module | — |
| **q = 0.05 and q = 0.20** | Descriptive only; primary decision uses q = 0.10 | — |
| **Hemolysis covariate** | Add `erythrocyte_score` to design | Sensitivity for hemolysis confounding |
| **ALSFRS_R_PROGRESSION_SLOPE covariate (C9-vs-SPOR only)** | Add to design, restricted to that contrast | Tests progression confounding |

**M1 deliverables (recorded at M1 tag):**

- `data/wasc/m1_numerical_reference_v1.json` — Frisch-Waugh ≡ statsmodels.OLS agreement results (PASS, max diff < 1e-10) and swap-invariance test (PASS, median 0.8%, max 3.2%; §9 caveat NOT triggered).

---

## 11. Implementation map (audit-linked)

| Component | Status | Location / Action |
|---|---|---|
| Cluster term member fetch | EXISTING | `scripts/viz/common.py:213,253` — chain through |
| INDRA hop-1 within-cluster edge enumeration | EXISTING | `network_proximity.py:1038 extract_subgraph_induced_by_features(max_hops=1, restrict_endpoints_to_features=True)` + `compute_all_pairs_shortest_paths_bounded(max_hops=1)` |
| Covariate design matrix | EXISTING | `design_matrix.py:70 build_covariate_design_matrix` — reuse for `X_g^{cov}` only (covariates side) |
| C9-Batch site_year collapse | NEW | `src/cliquefinder/stats/wasc/preprocess.py::collapse_c9_batch_to_site_year` — runs once at pre-reg time, output frozen to E9 manifest |
| ComBat within-group batch pre-residualization | NEW (or wrap existing) | `src/cliquefinder/stats/wasc/preprocess.py::combat_within_group` — either wrap `combat-py` or implement EB location/scale per `Johnson et al. 2007` |
| Per-edge per-group regression (Frisch-Waugh) | NEW | `src/cliquefinder/stats/wasc/fit.py::fit_per_edge_per_group` — Frisch-Waugh-Lovell as in §2.5 |
| Cochran Q statistic | NEW | `src/cliquefinder/stats/wasc/concordance.py::cochran_q` |
| Degree × coverage × \|Pearson\| 3-D bin builder | NEW | `src/cliquefinder/stats/wasc/null.py::build_joint_bins_3d` |
| N-non-neighbor 3-axis sampler (without replacement, exclude_set, ±1 decile fallback on |r|) | NEW | `src/cliquefinder/stats/wasc/null.py::sample_n_3axis_matched_non_neighbors` |
| Per-anchor permutation loop (joblib n_jobs=-1) | NEW | `src/cliquefinder/stats/wasc/null.py::run_wasc_null` — pattern from `landscape.py:1033 _per_feature_gradient_loop` (md5-seeded RNG, checkpointed) |
| Empirical Brown's combination | NEW | `src/cliquefinder/stats/combination.py::empirical_brown` — re-exported through `stats/__init__.py` |
| BY-FDR | EXISTING | `differential.py:242 fdr_correction(..., method="BY", alpha=0.10)` |
| BCa bootstrap on `ΔQ^{∩}` and `ΔQ̃` | NEW | `src/cliquefinder/stats/wasc/string_control.py::bca_delta_q_intersection`, `bca_delta_q_positives` |
| STRING v12.0 loader (canonical-UniProt only) | NEW | `src/cliquefinder/knowledge/string_ppi.py::load_string_physical_uniprot_adj_canonical` — per spec in `output/string_alternative_network.md` |
| Age imputation (regression on Sex within arm) | EXISTING pattern | `cliquefinder/quality/imputation.py` style — fit once at pre-registration time, freeze coefficients to E8 |
| Three-contrast pairwise Q | NEW | `src/cliquefinder/stats/wasc/three_contrast.py::pairwise_q` + `evaluate_c1_c4` |
| End-to-end orchestrator | NEW | `scripts/run_wasc.py` — calls all of the above, writes to `output/wasc/`, supports `--phase {primary, tertiary-tcell, tertiary-ipsc,...}` |

---

## 12. Sanity gates (must pass before reporting)

These are run as automated tests in the WASC pipeline. The run is INVALID if any fail.

1. **Edge count gate:** `|E_WASC|` matches the frozen M1 value **exactly** (944 total; 434 / 443 / 67 per theme). Drift indicates cluster-member set drift or graph snapshot mismatch; halt and document.
2. **Null calibration gate:** under H0 simulated by SHUFFLING group labels across donors (preserving group sizes), WASC should return ≤ `0.10 + 2·√(0.10·0.90/|E_WASC|) ≈ 12%` positive edges at q = 0.10. Run 20 label-shuffles; mean false-positive rate must satisfy the bound.
3. **STRING coverage gate:** STRING-canonical-UniProt-mapped accessions cover ≥ 70% of `⋃_T M_T = 377` cluster members. If below 60%, STRING control is statistically uninformative and the §7 branch resolves to `STRING-UNDERPOWERED`.
4. **Convergence gate:** ≥ 90% of `(j, a, g)` regressions must converge. If lower, missingness handling §2.3 is mis-tuned and must be revisited before reporting.
5. **Donor-exclusion audit:** confirm exactly the pre-registered 20 external/iPSC-derived donors are excluded; group sizes after exclusion match E7 manifest.
6. **C9-Batch collapse audit:** confirm the `Batch_raw → site_year` mapping in `data/wasc/c9_batch_collapse_v1.json` is bijective on its domain, covers all 25 C9 donors, and yields ≥ 3 donors per collapsed batch (otherwise extend collapse rule and re-tag).
7. **3-axis null match audit:** for each anchor, report the fraction of permutation iterations that required the ±1 decile fallback on the |r|-axis. If > 20% of iterations on any anchor invoke the fallback, that anchor's null is flagged in the manifest and its contribution to the global pool is reported separately.
8. **M1 numerical reference reproducibility:** the Frisch-Waugh ≡ statsmodels.OLS test (§2.5(a)) and swap-invariance test (§2.5(b)) re-run at M2 entry must reproduce the M1 results to within `1e-10` absolute. If swap-invariance Q drift exceeds 5% on rerun, append the §9 slope-symmetry caveat verbatim.

---

*End of specification v1.0. Frozen at git tag `wasc-prereg-v1.0` (M6a). Any deviation post-tag must be documented as a SECONDARY analysis with explicit "post-hoc" labelling.*

---

## v1.0.2 Amendment — Missingness axis dropped (2-axis null)

**Status:** AMENDMENT. Tag: `wasc-prereg-v1.0.2`. Applies to all M2.4+ compute. Supersedes the 3-axis language in §4, the Axis 2 reference in §10 (D3), the §11 sampler name, and Sanity Gate 7 in §12. All other v1.0 / v1.0.1 frozen items remain binding.

### Trigger

Empirical audit of `output/proteomics/all_als.data.csv` (the file `load_proteomics` consumes by default) found a literal NaN rate of `0 / 1,423,104` cells (0.0000%). The matrix is the AnswerALS data-portal `correctedImputed_436` track, which the portal README at `/Users/noot/Documents/case-control-genomics/proteomics/4_matrix/Proteomic Analysis Guidance README.txt:35` documents as Random-Forest imputed upstream of this repo. Cluster-member rate is identically 0.0 across all 345 deduped members (Splicing 190, Chromatin 145, Transport 42). No sibling pre-imputation file, detection-call matrix, or missingness mask exists anywhere under `output/` or `data/`.

Consequence: `compute_missingness_per_protein(abundance)` returns the all-zero series; `assign_decile` collapses every protein into bin 9 (verified by regression test `test_constant_input_does_not_return_minus_one`); the (degree, missingness, |r|) cell key degenerates to (degree, 9, |r|) by construction. Axis 2 contributes zero discriminative power to the matched draw on this dataset.

### No-Q-exposure attestation

At the time of this amendment, **944 observed Q values exist** (`output/wasc/concordance_per_edge_m2_2.csv`, generated 2026-06-02 06:27 from `cac32d0`). The amendment decision is based solely on the structural property of the abundance matrix (0/1,423,104 NaN), demonstrably independent of any observed Q value. No null Q draws have been computed against real data prior to the amendment. The bit-exact equivalence of 2-axis and 3-axis sampling on a 0%-NaN matrix is technical evidence (test `test_2axis_identical_to_3axis_on_zero_nan_matrix`): the rename is a descriptive correction, not a methodology relaxation.

A signed decision log is pinned at `data/wasc/v1.0.2_amendment_decision_log.json` with SHA-256 of the concordance CSV at amendment time and references to the three investigations + three audit verdict.

### Amendment §4 — 2-axis null

Replace "3-axis match" / "3-D grid" / "(degree-decile, missingness-decile, |r|-decile)" throughout §4 with **2-axis match** / **2-D grid** / **(degree-decile, |r|-decile)**. The Axis 2 bullet (line 195) is REMOVED. Axis 1 (degree) and Axis 3 (pooled |Pearson|) are retained and renumbered Axis 1 and Axis 2 respectively. On the v1.0.2 dataset, 2-axis and 3-axis matched draws are mathematically identical (the dropped axis was inert by data, not by spec change); the "strictly more conservative than 1-axis (degree-only)" framing applies only to hypothetical datasets where missingness is non-degenerate.

### Amendment §10 — D3

D3 reads: "Null model: anchor-local, **2-axis** (degree-decile × pooled-|Pearson|-decile) matched, sampled WITHOUT replacement (§4)."

### Amendment §11 — Implementation map

Sampler row renames `sample_n_3axis_matched_non_neighbors` to `sample_n_2axis_matched_non_neighbors`. Fallback description loses the missingness term: `(without replacement, exclude_set, ±1 decile fallback on |r|)`. Bin-builder row renames `Degree × coverage × |Pearson| 3-D bin builder` to `Degree × |Pearson| 2-D bin builder`. New code at `src/cliquefinder/stats/wasc/bins.py::build_anchor_bins` uses a `axes=("degree","corr")` default with the 3-axis code path preserved (opt-in via `axes=("degree","miss","corr")`) for the v1.1 prebatch re-derivation path.

### Amendment §12 — Sanity Gate 7

Renamed "**2-axis null match audit**". Reports per-anchor fraction of permutation iterations that required the ±1 decile fallback on the |r|-axis (unchanged threshold > 20%). **Additionally reports** (a) the per-anchor empirical distribution of `|Pearson(anchor, true_neighbor)|` vs `|Pearson(anchor, 2-axis_substitute)|` so v1.1 prebatch re-derivation has a baseline to compare against, and (b) per-anchor distribution of `n-eligible-candidates-per-cell`, which is the load-bearing diagnostic for whether 2-axis sampling saturates the eligible pool.

The 2-axis cell count is structurally lower than the 3-axis cell count (regression test `test_2axis_at_least_3axis_cell_population`), which mechanically lowers fallback frequency. The unchanged > 20% threshold therefore inherits an interpretively looser gate; this is acceptable because the dropped axis was inert on the v1.0.2 substrate.

### Amendment §214 — Rationale

The paragraph titled "Three-axis binning rationale" is replaced with:

> **2-axis binning rationale (v1.0.2):** The audit confirms degree-only matching was the prior practice in v0.9. Pooled |Pearson| was added in v1.0 in response to the brutalist's over-conservativeness diagnosis (within-cluster proteins co-express by pathway selection, so unmatched substitutes systematically have lower marginal correlation with the anchor than true neighbors). Missingness, added in v0.9, is dropped in v1.0.2 because the loaded matrix (`output/proteomics/all_als.data.csv`) is post-imputation by AnswerALS construction (audit: 0/1,423,104 NaN cells), making the axis discriminative-power-zero. Note: RF imputation depresses conditional variance for heavily-imputed proteins; this is partially absorbed by the |Pearson| axis but not fully controlled. A v1.1 re-derivation on the prebatch source (with detection mask preserved) is the canonical resolution.

### Items NOT touched by v1.0.2 (explicit enumeration, prophylactic per review mod)

The following v1.0 / v1.0.1 frozen items remain binding and are NOT modified by this amendment:

- B = 9999 primary; B = 999 sensitivities; B = 99999 floor-tie rerun (D2)
- q-threshold = 0.10 (D4)
- Per-anchor combination = empirical Brown's (D5)
- C2 floor = 48 (§8 three-contrast floor)
- Claim ceiling §9 verbatim, including the forbidden-language list (D8)
- Three-contrast C1–C4 criterion (D7)
- BY-FDR method (D4); Brown's combination (D5)
- STRING five-branch decision rule including STRING-ZERO-POSITIVES (D6)
- Donor exclusions E7 (20 external/iPSC-derived donors)
- C9 batch collapse mapping E9
- Mandatory sensitivities batch (Tertiary table in §10)
- M2.5 four-pronged HARD-HALT tripwire
- Edge enumeration |E_WASC| = 944 (E4)

### Prohibitions under v1.0.2

- **Prebatch matrix substitution is prohibited under v1.0.2** (including sensitivity runs). Restoring Axis 2 requires re-tagging as `wasc-prereg-v1.1` to prevent silent cherry-picking between substrates.
- AnchorBins pickles from prior runs are invalidated (the dataclass shape changed: `miss_bin: np.ndarray | None`, new `axes` field). M2.4 checkpoint, if any pre-amendment, must be rebuilt from scratch. Recommended runtime defense: startup assertion in the WASC orchestrator that logs the active `axes` and fails-loud if `axes` does not match the spec version pinned in `manifest_v1.json`.

### Future re-derivation path (NOT activated in v1.0.2)

A pre-imputation source exists outside this repo at `/Users/noot/Documents/case-control-genomics/proteomics/4_matrix/AnswerALS-447-P_proteomics-protein-matrix_prebatchcorrected.txt` (12.5 MB, 4144 × 447). Zero-as-missing rate 17.57% overall (literal-NaN rate 0%); per-protein zero-fraction min 0.22%, median 4.25%, max 79.19%; decile histogram populates 8 of 10 bins. Restoring Axis 2 requires:

1. Cohort harmonization (4144 × 447 prebatch vs current 3264 × 436; 27% larger protein universe, 11 extra samples).
2. Sample-ID separator reconciliation against `metadata_enriched_v1.json` (prebatch uses `-` separators, current uses `_`).
3. Re-validating §2's batch-correction assumptions against pre-ComBat input.
4. An explicit zero-as-missing recoding policy (defensible for LFQ MS data) with sensitivity vs a low-intensity-floor alternative.

Reserved for a v1.1 methodology revision tagged `wasc-prereg-v1.1` if/when the pipeline is re-derived on the prebatch matrix.

### Code locking

- `src/cliquefinder/stats/wasc/bins.py`: `build_anchor_bins` gains an `axes=("degree", "corr")` default. Missingness argument is optional and ignored when `"miss"` is not in `axes`; raises `ValueError` if `"miss"` requested without missingness data. The 3-axis code path is preserved (re-activated via `axes=("degree", "miss", "corr")`) so v1.1 re-derivation can ship without reverting this amendment.
- `AnchorBins` dataclass: `miss_bin: np.ndarray | None`, new `axes` field, `cells: dict[tuple[int,...], tuple[str,...]]` (variable-length keys).
- v1.0.2 regression tests added (`tests/wasc/test_bins.py::TestV102AxesAmendment`, 7 tests): default cell key shape, 3-axis opt-in, ValueError guards, 2-axis ≥ 3-axis cell population, bit-exact equivalence on 0%-NaN matrix.
- Constant-input regression test (`test_constant_input_does_not_return_minus_one`) closes the review concern that `assign_decile` might silently produce zero-candidate cells on constant input.

### Acknowledgment of v1.0.1 commit-message claim

The v1.0.1 tag commit message explicitly stated "Analytical spec unchanged". v1.0.2 amends an analytical decision (D3). This is a deliberate semver progression: v1.0.1 corrected the input metadata; v1.0.2 corrects an empirically-inert axis. The v1.0.2 commit message records this break.

*End of v1.0.2 amendment. Frozen at git tag `wasc-prereg-v1.0.2`. Decided 2026-06-02; (3 investigators + 3 reviewers, 0 REJECT, 3 MODIFY, all mods incorporated).*

---

## v1.0.3 Amendment — All-protein-pool as canonical primary

**Status:** AMENDMENT. Tag: `wasc-prereg-v1.0.3`. Applies to all M2.5 prong (a)+ and M3+ compute. Supersedes the theme-restricted candidate pool in §4 (`P_a^candidate = M_T \ N_a^obs \ {a}`); promotes the build-plan all-protein-pool variant (previously labeled prong (c) sensitivity in build_plan §10) to the canonical primary substrate.

All other v1.0 / v1.0.1 / v1.0.2 frozen items remain binding except as noted below.

### Trigger

M2.5 prong (a) label-shuffle calibration at v1.0.2's theme-restricted substrate FAILED its pre-registered Gate 2 bound under H0:

 Theme-restricted (v1.0.2 canonical) + ±1-decile fallback: mean FP = 0.261 vs bound 0.120 → FAIL
 All-protein-pool (build-plan prong c) : mean FP = 0.107 vs bound 0.120 → PASS

Two parallel brutalist workflows (``, `` — 14 agents total) confirmed the mechanism:

1. **±1-decile fallback exonerated.** Disabling fallback (`max_relaxation=0`) only drops theme-pool FP from 0.261 to 0.251. Widening (`max_relaxation=2`) nudges to 0.265. The fallback contributes ≤ 0.014 of the 0.154-point miscalibration.

2. **Sparse-cell sampling is the cause.** M_T sizes 42 (Transport), 145 (Chromatin), 190 (Splicing) over a 10×10 = 100 decile-cell grid produce per-cell occupancy median ≤ 2. β's diagnostic on Splicing M_T: 22.8% empty cells, 21.9% density=1, 53.7% with ≤2 unique fakes/99 draws. Transport (M_T=42, cell-density median 1) hits FP = 0.540 — one fake drawn for all B perms → Q_null constant → lower-tail formula deterministic.

3. **V1's `min_unique_q_values` guard does NOT recover calibration alone.** K=5 default: theme-restricted FP = 0.137 (still FAIL, 576 edges suppressed). K=10: FP = 0.117 "PASS" only because n_finite collapses to 94 and the recomputed bound widens to 0.162 — bound inflation, not calibration. The guard is a defensive correctness fix for the strictly-constant Q_null pathology; it cannot rescue theme-restricted on the v1.0.2 substrate. 

4. **All-protein-pool calibrates cleanly.** Mean FP = 0.107 over 5 shuffles × B=99 (full 944/944 edges retained); production 20 × B=999 mean FP = 0.111 vs bound 0.120 → POOLED PASS at production scale (commit `c92b11c`).

### No-Q-exposure attestation

The v1.0.3 decision is driven solely by:

- H0 label-shuffle FP-rate outcomes (synthetic null, derived from the real abundance matrix's covariance structure but NOT from the observed Cochran-Q on real-label edges).
- Structural matrix properties: |E_WASC| = 944, M_T per-theme counts, per-cell occupancy distributions, cell-density medians.
- The mathematical properties of the lower-tail permutation formula on constant Q_null.

Real-label Q-rank values from `output/wasc/concordance_per_edge_m2_2.csv` were observed during the M2.4 full B=100 sanity smoke (commit `c92b11c`) prior to this amendment, but those observations DID NOT motivate the swap. The decision criteria (H0 calibration failure of theme-restricted; H0 calibration pass of all-protein-pool) are real-label-Q-blind. The signed decision log at `data/wasc/v1.0.3_amendment_decision_log.json` pins SHA-256 of the observed-Q artifacts to lock their content as of the amendment time.

Caveat (per `` audit finding): H0 label-shuffle draws are still a function of the real abundance matrix's covariance structure (via covariate-residualization, anchor z-scoring, |Pearson|-decile assignment). The "no real-Q exposure" claim is operationally true at the Q-rank/p-value level but does not certify zero data-dependence at the abundance-matrix-properties level. This is the same attestation standard v1.0.2 met.

### Amendment §4 — All-protein-pool null substrate

Replace the §4 line 192 definition

 Let `P_a^candidate` = `M_T \ N_a^obs \ {a}` (measured cluster members in theme T, excluding a's true neighbors and a itself)

with

 Let `P_a^candidate` = `M \ N_a^obs \ {a}` where `M` is the full measured proteome (all UniProts in `abundance.index`).

All other §4 text (decile axes, sampler, ±1-decile fallback, B = 9999, anchor-local pooling) is preserved.

### Amendment §10 — D3

D3 reads: "Null model: anchor-local, **2-axis** (degree-decile × pooled-|Pearson|-decile) matched, **substrate = full measured proteome (M)**, sampled WITHOUT replacement (§4)."

### Amendment §11 — Implementation map

Default `eligible_proteins=None` in `build_anchor_bins` is now the canonical setting (previously this was the "all-protein-pool prong (c) variant"). Theme-restricted (`eligible_proteins=M_T`) is preserved as an opt-in for sensitivity analyses. Orchestration scripts default `--candidate-pool all` (previously `theme`).

### Amendment §12 — Sanity Gate 7

The "2-axis null match audit" still reports per-anchor ±1-decile fallback rate and the descriptive |Pearson| distributions. Under all-protein-pool the fallback rate is effectively 0% (cell density ~34 candidates), so the gate is descriptive-only at the v1.0.3 substrate. Documented as such.

### Theme-restricted result — DOCUMENTED FAILURE, retained as sensitivity

The theme-restricted code path (`build_anchor_bins(..., eligible_proteins=M_T)`) is preserved. It is RUN as a tertiary sensitivity in the M2.5 prong (a) battery alongside the canonical all-protein-pool primary. Its FAILED calibration (mean FP = 0.261 at v1.0.2 substrate) is reported in the output of every WASC run as a recorded sensitivity outcome, not as a positive result. This documents (a) the spec freeze respected the original v1.0.2 design choice empirically and (b) the failure of that design on the AnswerALS RF-imputed substrate is itself a finding worth reporting.

The structural reason: M_T sizes 42–190 over 100 decile cells produce per-cell occupancy too small for the matched-bin null to maintain exchangeability. A v1.1 re-derivation on the AnswerALS prebatch matrix (preserves missingness; would re-enable Axis 2; potentially larger M_T) may make theme-restricted calibrated; that question is reserved for `wasc-prereg-v1.1`.

### Items NOT touched by v1.0.3 (explicit enumeration, prophylactic)

The following remain binding (extended from v1.0.2's enumeration):

- B = 9999 primary; B = 999 sensitivities; B = 99999 floor-tie rerun (D2)
- q-threshold = 0.10 (D4)
- Per-anchor combination = empirical Brown's (D5)
- C2 floor = 48 (§8 three-contrast floor)
- Claim ceiling §9 verbatim including the forbidden-language list (D8)
- Three-contrast C1–C4 criterion (D7)
- BY-FDR method (D4)
- STRING five-branch decision rule including STRING-ZERO-POSITIVES (D6)
- Donor exclusions E7 (20 external/iPSC-derived donors)
- C9 batch collapse mapping E9
- Mandatory sensitivities batch (Tertiary table in §10)
- M2.5 four-pronged HARD-HALT tripwire structure (the tripwire definition itself is unchanged; only the substrate the prongs run against changes)
- **Edge enumeration |E_WASC| = 944 with per-theme breakdown (434 / 443 / 67)** — UNCHANGED; the count + cluster-pattern primary outcome stays anchored to this denominator. All-protein-pool prong (a) reports against the full 944.
- Lower-tail p-value convention (commit `796dc79`).
- min_unique_q_values guard with default K=5 (commit `43e5aba`) — defensive only, inert at all-protein-pool substrate.

### Prohibitions under v1.0.3

- Theme-restricted prong (a) calibration outcome from v1.0.2 (`FP=0.261 FAIL`) is the binding record; no further iterations of K-sweeps, fallback redesigns, or hybrid-pool augmentations may be invoked to "rescue" theme-restricted as primary under v1.0.3. Such efforts must be re-tagged (v1.1 or later) with their own audit gate.
- AnchorBins built under v1.0.2 with `eligible_proteins=M_T` retain that semantics; pickles are NOT invalidated (dataclass shape unchanged from v1.0.2 → v1.0.3).
- The build-plan Tertiary table in §10 referenced "all-protein-pool" as a sensitivity (item #5 in the original list). That row is REMOVED from the Tertiary table (since it's now primary) and replaced with a new sensitivity row: "theme-restricted-with-known-FAILED-calibration".

### Acknowledgment of v1.0.2 commit-message claim

The v1.0.2 commit message implicitly asserted theme-restricted was the canonical primary. v1.0.3 amends that within the same day (2026-06-02), motivated by empirical calibration failure under the spec's own pre-registered Gate 2. The cumulative effect on "spec freeze" credibility is acknowledged: WASC has now amended an analytical decision (D3) twice in one day. The defensible framing is: each amendment was triggered by a real empirical finding under H0 that the prior version did not anticipate, and each followed the same review-gated protocol. Future reviewers comparing the v1.0.1 commit-message claim of "Analytical spec unchanged" with three same-day analytical amendments will reasonably interpret "pre-registered" as the documented decision-with-decision-log, not as eternal immutability.

### Pre-registered Tertiary table update (§10)

Replace Tertiary item 5 ("all-protein-pool" sensitivity) with:

> **Theme-restricted-with-known-FAILED-calibration sensitivity** — Re-run prong (a) with `eligible_proteins=M_T` per (anchor, theme). Reports per-theme FP rate, n_finite_p, and the surviving-edge subset's per-anchor BY-FDR. Reported as a SENSITIVITY whose result-set is interpretable only against the v1.0.2 sub-pre-registered failure baseline (FP=0.261). Not a primary inference.

*End of v1.0.3 amendment. Frozen at git tag `wasc-prereg-v1.0.3`. Decided 2026-06-02; workflows `` (3I + 3V) + `` (4S + 3V) returned 0 ENDORSE, 0 ACCEPT for any path other than (iii) all-protein-pool primary.*

---

## v1.0.4 Amendment — Spec corrections from foundational audit

**Status:** AMENDMENT. Tag: `wasc-prereg-v1.0.4`. **CORRECTIONS-ONLY** per the discipline established by (4 auditors + synthesis + 3 reviewers). Items with rescue-risk explicitly deferred to v1.1.

### Trigger

The v1.0.3 amendment chain (4 audit gates: v1.0 / v1.0.1 / v1.0.2 / v1.0.3) focused on the amendments being made and treated `items_NOT_modified` as trusted. A holistic audit of the full spec + amendments + code found 37 issues including 3 CRITICAL and 16 HIGH-severity arithmetic / ambiguity / consistency / feasibility bugs that survived all prior gates. The synthesis review (V1) caught 4 rescue-disguised-as-correction items in the audit's first-draft amendment proposal; those are deferred. v1.0.4 is the residual: 7 clean corrections + CI test infrastructure.

### No-Q-exposure attestation

The corrections in v1.0.4 derive from audit of the spec text + code defaults. No observed Q values, p-values, q-values, or BY-FDR tables were consulted in deriving any correction. Each correction fixes a bug whose existence is independent of any prong's pass/fail verdict (verified by the rescue-screen brutalist).

### Corrections — clean (no rescue risk)

**C1 (CRITICAL) — BY rank-1 arithmetic.** Spec §4 line 212 and §6 line 257 derived the BY rank-1 raw-p threshold as `q · (1/H_n) ≈ 0.10/7.5 ≈ 0.013`. This is the BH adjustment factor, not the BY rank-1 threshold. The correct BY adjustment is `p_(k) · N · H_N / k`, so the rank-1 raw-p rejection threshold at q=0.10, N=944, H_944=7.4279 is `q / (N · H_N) = 1.43e-5`. The previously-claimed "~7× safety margin" of B=9999 (raw-p floor 1e-4) over `0.013` is actually a 7× DEFICIT below the true threshold of 1.43e-5. **Consequence**: at B=9999 the raw-p floor exceeds the rank-1 threshold; ranks 1..7 are untestable at q≤0.10 from a B=9999 primary alone. The "B=99999 rerun if floor-tied" tertiary becomes effectively guaranteed (not contingent) for any strong-signal edge. Whether to PROMOTE B=99999 to primary is a separate analytical decision and is deferred to v1.1. v1.0.4 corrects the SPEC NARRATIVE only.

**C3 (HIGH) — Brown variance spec text.** Spec §5 line 228 wrote `Var[χ²_a] = 4 · n_a + 2 · Σ_{i ≠ j} cov_emp(...)`. The notation `Σ_{i ≠ j}` enumerates each unordered pair twice; the `2 ·` then double-counts. Correct: `Var = 4·n_a + Σ_{i ≠ j} cov_emp = 4·n_a + 2·Σ_{i<j} cov_emp`. The code at `src/cliquefinder/stats/wasc/combination.py:184-189` already implements the correct formula. v1.0.4 aligns spec text with code. No code change.

**C7 (MEDIUM) — RNG formula reconciliation.** Spec §10 D10 says per-anchor seed = `int(md5(uniprot + "wasc-v1.0").hexdigest[:8], 16)`. Code in `src/cliquefinder/stats/wasc/null.py:anchor_seed` uses `int.from_bytes(md5(f"{global_salt}|{uniprot}").digest[:4], "little", unsigned)`. Salt format differs (concatenation vs `|`-separated) AND the int-decoding path differs (hex prefix vs first-4-bytes little-endian). v1.0.4 reconciles to the CODE form (which is what has been producing all M2.4 / M3 / M2.5 results) and updates spec text accordingly.

**C9 (LOW) — min_n_per_group keys.** Spec §2.3 line 122 says "≥10 for C9 and ≥15 for SPOR/CTRL". Code dict default uses keys `{"C9ORF72": 10, "SPORADIC": 15, "CONTROL": 15}`. v1.0.4 locks the canonical key names as the binding form (matches `group_order = ("C9ORF72", "SPORADIC", "CONTROL")` throughout).

**C11 (LOW) — prong (b) comparator.** Build plan §5 line 25, 766 write the M2.5 prong (b) bound as `overlap ≥ 70%`. v1.0.4 confirms the comparator is `≥` (inclusive lower bound on Jaccard), not `>`. Observed v1.0.3 Jaccard = 0.285 < 0.70 = HARD FAIL.

**C16 (LOW) — Anchor count documentation.** Spec text said "order 220-280" anchors loosely. Verified count: 203 distinct canonical anchors in `data/wasc/E_WASC_v1.json`; per-(anchor, theme) work units = 220 (17 multi-theme anchors get one unit per theme).

**C17 (MEDIUM) — Perfect-H1 power statement.** ⚠️ **WRONG AS WRITTEN IN v1.0.4.** Superseded by C17′ in v1.0.5 (see end of file). The v1.0.4 text claimed "at B=99999 only ranks 1..69 can clear BY q ≤ 0.10" — both sign and B mis-attributed. Correct statement: at B=99999 the empirical-p floor (1e-5) is below the BY rank-1 threshold (1.43e-5), so all 944 ranks are testable. The "69" figure derives from B=999 (sensitivity tier), where ranks 1..70 are UNTESTABLE (max 874 testable rejections). v1.0.5 reframes this as a B-feasibility table; see C17′.

### Infrastructure — locked_bounds CI gate

`data/wasc/locked_bounds_v1.json` (NEW) pins every load-bearing numeric threshold (B, q, min_n_per_group, C2 floor, min_valid_perms, min_unique_q_values, axes default, prong (b) Jaccard floor, prong (d) tolerance, canonical-direction convention, |E_WASC|=944, anchor count 203) with both spec_reference (file + verbatim quote) and code_default (file + symbol + parameter + value).

`tests/wasc/test_locked_bounds.py` (NEW) reads the JSON and asserts:
1. Every spec_reference.search_quote substring is present in the cited file (spec-text drift detector).
2. Every code_default.parameter default value matches the binding (code drift detector via `inspect.signature`).
3. `|E_WASC|` binding (944) matches `data/wasc/E_WASC_v1.json` row count.

4/4 tests pass at amendment time. Drift in either spec OR code now fails CI loud, eliminating the silent-divergence pathology that allowed C1 + C3 to survive 4 prior gates.

### Items DEFERRED to v1.1 (separate audit gate required)

The synthesis review (V1) caught 4 rescue-disguised-as-correction items in the audit proposal. These are deferred to v1.1 per the v1.0.3 prohibitions clause:

- **C2 deferred** — Promote B=9999 → B=99999 as primary. This eliminates v1.0.3's staged D2 design (adaptive trigger). Re-tag and re-gate.
- **C5 deferred** — Pre-register 944 vs 904 as the BY-FDR denominator. The choice has rescue-risk (944 preserves the v1.0.3 prong (a) PASS verdict).
- **C8 deferred** — Bind RAW-p vs BY-q as the Gate 2 metric. Locking RAW-p (the metric under which prong (a) PASSED) while deferring BY-q (under which the verdict could differ) is asymmetric.
- **C12-partial deferred** — "K∈{1,10} permitted as tertiary diagnostics" — v1.0.3 prohibitions[0] explicitly forbid K-sweeps. K=5 plumbing through `run_label_shuffle_calibration` proceeds (already pre-registered K=5 in commit 43e5aba); only the SWEEP is deferred.

### Items NOT modified by v1.0.4

(Same enumeration as v1.0.3 items_NOT_modified, plus): the M2.5 four-pronged HARD HALT semantics, the v1.0.3 prong (a) PASS verdict, the v1.0.3 prong (b) FAIL verdict, the v1.0.3 candidate-pool primary (all-protein-pool), |E_WASC|=944.

### Acknowledgment

Three intra-day amendments to D3/D4-family decisions (v1.0.2: drop missingness axis; v1.0.3: swap candidate pool; v1.0.4: correct BY arithmetic + Brown variance text) is a credibility stress on "pre-registered". Defensible framing: each amendment was triggered by an independent finding (v1.0.2: structural matrix property; v1.0.3: H0 calibration FAIL; v1.0.4: audit-discovered pre-existing bugs that survived 4 prior gates). The cumulative pattern reveals a deeper foundational issue: the original v1.0 spec had arithmetic and ambiguity bugs the audit gates did not catch because each gate scoped review to the amendment being made. v1.0.4 adds the CI infrastructure (locked_bounds + drift test) that prevents this class of bug from surviving future amendments.

*End of v1.0.4 amendment. Frozen at git tag `wasc-prereg-v1.0.4`. Decided 2026-06-02; (4 auditors + synthesis + 3 reviewers, 1 REJECT / 2 MODIFY; 4 rescue items deferred per audit REJECT, 6 amendment-internal mods applied per all verdicts).*

---

## v1.0.5 Amendment — C17 arithmetic correction

**Status:** AMENDMENT. Tag: `wasc-prereg-v1.0.5`. **CORRECTIONS-ONLY** (single arithmetic fix to a v1.0.4 statement that itself was a correction). Triggered by audit verdict, which caught that the v1.0.4 C17 perfect-H1 power statement had the sign INVERTED and the B value MISATTRIBUTED. v1.0.4's locked-bounds CI gate did not detect this because C17 was added by v1.0.4 itself; no prior tag pinned the (B → testable-rank) mapping.

### Trigger

Workflow `` ran a Path A publication scaffold + 3 artifact-hygiene fixes in parallel. V3 (cross-deliverable consistency reviewer) independently re-derived the BY-rank-feasibility arithmetic from scratch using `q = 0.10`, `N = 944`, `H_944 = 7.4279` (exact), and found that the v1.0.4 C17 statement "at B=99999 only ranks 1..69 can clear" is wrong in both sign and B attribution. The Path A scaffold draft propagated the wrong statement verbatim; not committing the draft prevented downstream propagation, but the v1.0.4 spec text itself remained wrong until this amendment.

### No-Q-exposure attestation

The C17 correction derives from pure arithmetic against the pre-registered (q, N, H_N, B) tuple. No Q values, p-values, q-values, observed FP rates, or BY-FDR tables were consulted. The fix is independent of any prong's pass/fail verdict (verified: prong (a) PASS, prong (b) FAIL, prong (d) PASS verdicts under v1.0.4 are NOT moved by this correction; the C17 number is a feasibility property of the (B, q, N) tuple, not a property of any observed run).

### C17′ — Corrected B-feasibility table

At `N = 944`, `q = 0.10`, `H_944 = 7.4279`, the BY rank-k raw-p threshold is `p_(k) ≤ q · k / (N · H_N) = k · 1.4261e-5`. An edge can clear BY q ≤ 0.10 at rank k only if its achievable empirical p-value (Phipson-Smyth-corrected floor `1 / (B + 1)`) is at most its rank-k threshold. The structural feasibility ceiling under perfect per-edge power is:

| B | Empirical-p floor (1/(B+1)) | k_min (smallest testable rank) | Untestable ranks | Maximum testable rejections |
|---:|---:|---:|---:|---:|
| 999 (sensitivity tier) | 1.00e-3 | 71 | 1..70 (70 ranks) | 874 of 944 |
| 9999 (primary) | 1.00e-4 | 8 | 1..7 (7 ranks) | 937 of 944 |
| 99999 (floor-tie rerun) | 1.00e-5 | 1 | (none) | 944 of 944 |

(Computation: `scripts/wasc/verify_by_feasibility.py` — see verification script committed alongside this amendment, or rederive: `k_min = ceil(floor / rank1_threshold)`, where `rank1_threshold = q / (N · H_N) = 1.4261e-5`.)

**Interpretation**:
- At the pre-registered primary `B = 9999`, the structural ceiling is **937 of 944 edges testable** — not 69. Ranks 1..7 are untestable from B=9999 alone.
- At the floor-tie tertiary `B = 99999`, **all 944 edges are testable**.
- The "69" figure in the v1.0.4 C17 text comes from B=999 sensitivity tier, where ranks 1..70 are untestable (max 874 testable); the spec text confused testable vs untestable ranks AND the B value.

### Items NOT modified by v1.0.5

- All v1.0.4 items_NOT_modified (B values, q=0.10, Brown's, C2=48, claim ceiling, three-contrast, BY method, STRING rule, donor exclusions, batch collapse, sensitivities, M2.5 4-prong HARD HALT structure, |E_WASC|=944, lower-tail p-value, min_unique_q_values K=5, locked_bounds_v1.json bindings except the C17 description).
- All v1.0.4 corrections C1, C3, C7, C9, C11, C16 (verified correct under independent re-derivation; only C17 was wrong).
- The v1.0.4 deferred-to-v1.1 items (C2 B promotion, C5 BY denominator, C8 Gate 2 metric, C12 K-sweep).
- v1.0.4 prong verdicts: prong (a) PASS, prong (b) FAIL, prong (d) PASS — unchanged by the C17 correction (C17 is a B-feasibility property, not a prong-outcome property).

### Items deferred to v1.1 (unchanged from v1.0.4 enumeration)

Same as v1.0.4. v1.0.5 adds no new deferred items.

### Lessons + infrastructure follow-on (non-binding, for future amendments)

1. The v1.0.4 locked-bounds CI gate caught spec/code drift but cannot catch arithmetic errors in spec text on values it does not lock. To prevent C17-class bugs, v1.1 should add a `tests/wasc/test_feasibility.py` that asserts the B-vs-testable-rank table against an independent computation (`q / (N · H_N)` and `ceil(floor / rank1_threshold)`), not against text-quoted numbers.
2. The v1.0.4 amendment introduced C17 as a "new arithmetic statement" without an independent verifier. Future amendments that introduce NEW arithmetic claims (vs correcting OLD ones) should require an extra reviewer whose sole job is to re-derive the claim from first principles.

*End of v1.0.5 amendment. Frozen at git tag `wasc-prereg-v1.0.5`. Decided 2026-06-04; V3 brutalist (cross-deliverable consistency) caught the error in the v1.0.4 C17 statement; independently re-derived against locked (q, N, H_N) tuple.*