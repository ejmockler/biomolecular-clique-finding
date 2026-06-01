# WASC — Build Plan

> **STATUS — v1.0 (PRE-REGISTRATION, BINDING).** Frozen 2026-06-02. Brutalist modifications applied; M1 numerical outputs incorporated; milestone sequence reordered; mandatory sensitivities batch added; calibration tripwire inserted as a hard halt. This document is now authoritative and binding. Deviations require an explicit `wasc-plan-v1.1` tag and a written rationale.
>
> **Frozen M1 numerical inputs (locked, 2026-06-02):**
> - `|E_WASC|` = **944** edges. Per-theme: Splicing 434 / Chromatin 443 / Transport 67.
> - Cluster members: 377 measured UniProt accessions total (Splicing 190 from 304 UniProt union of 303 HGNC; Chromatin 145 from 468 UniProt union of 467 HGNC; Transport 42 from 70 UniProt union of 70 HGNC).
> - Within-theme densities: Splicing 2.4%, Chromatin 4.2%, Transport 7.8% (biologically sane; flagged as expected at this scale).
> - Measured proteome: 3,264 UniProt accessions (Wave-22 protein-level matrix).
> - INDRA query date: 2026-06-02 against `bolt://indra-cogex-lb-b954b684556c373c.elb.us-east-1.amazonaws.com:7687`.
>
> **Binding consequence of |E_WASC| = 944:**
> - C2 contrast-specific floor recalculated: `ceil(0.05 × 944) = 48` contrast-specific edges per C9 comparison.
> - M2 anchor loop is **mandatorily** `joblib`-parallelized (`n_jobs=-1`). Single-threaded execution at this scale is infeasible.
> - Primary run wall-clock revised to **25–40 h**. Full sensitivities batch budget: **120–200 h**.
> - Sensitivity reruns may use B = 999 (1/1000 floor). The primary INDRA and primary STRING runs MUST stay at B = 9999.
> - BY-FDR threshold remains q = 0.10; the primary claim is framed as **"count + cluster pattern"** invariance, not per-edge mechanism.
> - Cross-theme edges remain **DEFERRED** to a future exploratory secondary module; out of M1/M2/M3 scope.
>
> **Required milestone-sequence modifications (all applied below):**
> 1. **M6a (pre-registration manifest + git tag) MOVED BEFORE M2.** M1 produces |E_WASC| = 944 (DONE 2026-06-02); the pre-reg tag is applied next. No anchor-local null Q values may be computed on real data until tag `wasc-prereg-v1.0` is in place. M2 (concordance + null on real data) is **BLOCKED** on M6a.
> 2. **M1 numerical-identity gate added.** Frisch-Waugh kernel vs `statsmodels.OLS` (with explicit batch dummies on SPOR group) must agree to 1e-8 on 50 real (anchor, target, group) triples BEFORE M2 starts.
> 3. **M2.5 — Four-pronged calibration tripwire (HARD HALT)** inserted after M2 (null implementation), before M3. All four prongs must pass; HALT BUILD if any fails:
>    - (a) Label-shuffle FP rate ∈ [0.05, 0.14], stratified by degree decile AND SE quintile;
>    - (b) Down-sampled-SPOR-to-25 overlap with full-N WASC-positive set ≥ 70%;
>    - (c) All-measured-protein-pool null does not explode positives (>3× = over-restrictive within-theme null);
>    - (d) Frisch-Waugh kernel vs `statsmodels.OLS` with explicit batch dummies on SPOR group — identity to 1e-8 on the production batch design.
> 4. **M5.5 — Mandatory pre-registered sensitivities batch** inserted after M5. ALL five sensitivities run regardless of primary outcome: T-Cell-stratified, iPSC-retained, batch-correction-OFF, down-sampled-SPOR-to-25, all-protein-pool. B = 999 permitted for sensitivity reruns; B = 99999 rerun mandatory if any floor-tied edges exist at primary B = 9999.
> 5. **Revised total effort estimate: 180–220 h** (was ~140 h).
> 6. **M7 disposition reframed.** A negative WASC is a **publishable finding of structural-coupling-flatness**, not a retraction of wave_24l. Report integration is conditional on result, not blocked by it.

**Within-cluster Anchor-Slope Concordance**
Companion to: `data/wasc/spec_v1.0.md` (the statistical spec, frozen at git tag `wasc-prereg-v1.0`).
Status: pre-registration. Authoritative.

This document is the engineering build plan that operationalizes the statistical spec on the existing biomolecular-clique-finding codebase. It is binding for the WASC implementation: deviations from this plan must be recorded as TODOs and reviewed before merging.

---

## 0. Scope and non-scope

**In scope.** Per-edge inverse-variance-weighted Cochran-Q test on partial regression slopes within the 8 pre-registered C9-ALS cluster terms (Splicing / Chromatin / Transport themes) across {C9, SPOR, CTRL}; degree-and-coverage-matched anchor-local permutation null; BY-FDR over per-edge p-values at q = 0.10 (framed as count + cluster pattern, not per-edge mechanism); empirical-Brown's per-anchor combination; STRING physical-PPI negative control on the same anchors; three-contrast decomposition; pre-registration manifest with SHA-256 input fingerprinting; sanity gates; mandatory pre-registered sensitivities batch.

**Out of scope.** Granular cell-composition deconvolution (no T-CD4/CD8/B/NK/monocyte covariates available in DDA proteomics); mechanism claims; direction-of-effect claims; per-edge validation as a regulatory ground truth; **cross-theme edges** (explicitly DEFERRED — reported as a future exploratory secondary module, not v1.0); RNA-seq cross-modality (separate wave); ALSFRS slope per-patient as a covariate in the primary fit (reported tertiary only); any operation that changes the locked `|E_WASC| = 944` pre-registered edge count. The claim ceiling in §9 of the spec is the maximum licensed inference.

**Reuse principle.** Do not modify existing utilities. Use Frisch-Waugh-Lovell to fit a brand-new kernel rather than retrofitting `RotationTestEngine`, `run_protein_differential`, or `batched_ols_contrast_test` — those test condition-effect contrasts and have the wrong X structure for per-pair regression. Reuse `build_covariate_design_matrix` for covariate-side construction, `fit_f_dist` + `squeeze_var` for the EB-moderated sensitivity, `fdr_correction(method="BY")` for FDR, `extract_subgraph_induced_by_features` + `compute_all_pairs_shortest_paths_bounded` for hop-1 enumeration, `TERMS` + `fetch_term_members_via_indra` + `hgnc_ids_to_uniprots` for cluster-member fetch, and the landscape's `_per_feature_gradient_loop` RNG-seeding + checkpoint pattern verbatim for the per-anchor null loop.

---

## 1. Module placement

WASC is a **statistically distinct test family** (not a different parameterization of the existing landscape gradient). It deserves a dedicated module rather than sprawling functions in `stats/network_proximity.py`. Confirmed placement:

```
src/cliquefinder/stats/wasc/
    __init__.py            re-exports the public surface
    types.py               dataclasses (Theme, Network, WascEdge, ... WascResults)
    edges.py               edge enumeration on INDRA and STRING
    preprocess.py          within-group ComBat batch pre-residualization; age imputation
    fit.py                 Frisch–Waugh per-(edge, group) regression kernel
    concordance.py         Cochran Q, I², τ²_DL, pooled β̄
    null.py                degree × coverage 2-D bin builder; matched non-neighbor sampler
                           per-anchor null loop (md5-seeded, checkpointed, joblib-parallel)
    combination.py         empirical Brown's (Poole 2016); Fisher fallback
    contrasts.py           three pairwise 2-group Qs + C1..C4 decision rule
    string_control.py      INDRA-vs-STRING ΔQ with BCa bootstrap
    sanity.py              the 5 pre-registered gates + tripwire (M2.5)
    api.py                 the top-level orchestrator: run_wasc, run_wasc_string_control

src/cliquefinder/knowledge/
    string_ppi.py          NEW — STRING v12.0 loader + ENSP↔UniProt collapse
```

**Public surface** (re-exported through `src/cliquefinder/stats/wasc/__init__.py`):

```python
from cliquefinder.stats.wasc import (
    # types
    Theme, Network, WascEdge, PerGroupFit, EdgeResult, AnchorResult,
    ContrastEdgeResult, ContrastDecomp, StringComparison, SanityGates,
    WascManifest, WascResults,
    # edge enumeration
    enumerate_wasc_indra_edges, enumerate_wasc_string_edges,
    # fit
    build_group_residual_cache, fit_per_group_beta,
    # concordance
    cochran_q, pooled_beta, i_squared, dersimonian_laird_tau2,
    # null
    build_degree_coverage_bins, sample_n_matched_non_neighbors,
    run_anchor_null,
    # combination
    empirical_brown, fisher_combine,
    # contrasts
    decompose_three_contrast,
    # STRING control
    compare_indra_vs_string,
    # orchestrator
    run_wasc, run_wasc_string_control,
    # sanity
    run_sanity_gates, run_calibration_tripwire,
)
```

These are **also** re-exported through `src/cliquefinder/stats/__init__.py` (alongside the existing `fdr_correction` etc.) so callers can do `from cliquefinder.stats import run_wasc`. The empirical-Brown combiner lands additionally as a stand-alone `from cliquefinder.stats import empirical_brown` because it is generic enough to be reused outside WASC.

**STRING loader placement.** `src/cliquefinder/knowledge/string_ppi.py` (sibling to `cogex.py` and `indra_source.py`). Not a generic external-DB module — this loader is specific to STRING v12.0 physical-links flat files and the canonical UniProt-collapse we use for WASC. If a second STRING use-case appears later it can be lifted into a more general module.

**Script entry points.**
```
scripts/run_wasc.py                  end-to-end orchestrator (the only script users invoke)
scripts/prepare_wasc_prereg.py       one-shot, freezes E1–E8 to data/wasc/*.json
scripts/wasc/                        helper scripts (sanity audits, diagnostic plots,
                                     calibration tripwire harness)
```

The orchestrator script writes to `output/wasc/{run_label}/` and follows the panels/landscape output convention (`results.json`, `edges.csv`, `anchors.csv`, `contrasts.csv`, `string_comparison.json`, `sanity.json`, `tripwire.json`, `manifest.json`, `wasc.log`).

---

## 2. Function signatures

All signatures are precise and type-annotated. Where an existing utility is reused, the spec is the call site, not a re-declaration.

### 2.1 Edge enumeration

```python
# wasc/edges.py

def enumerate_wasc_indra_edges(
    cluster_terms: list[tuple[Theme, str]],        # [(Theme.SPLICING, "go:0000398"), ...]
    measured_uniprots: frozenset[str],             # M, the 3,264 proteome accessions
    cogex_client: CoGExClient,
    *,
    snapshot_distances: FeatureDistanceMatrix | None = None,
    require_belief: bool = False,                  # if True, populate edge_reliability
) -> tuple[WascEdge, ...]:
    """
    Enumerate within-cluster INDRA hop-1 edges per the spec §1.
    Locked output for v1.0: |E_WASC| = 944 (434 Splicing / 443 Chromatin / 67 Transport).

    Pipeline:
      1. fetch_term_members_via_indra(term_ids) → {term_id: {hgnc_id}}
      2. group by theme; union members per theme
      3. hgnc_ids_to_uniprots(union_per_theme) → {Theme: {uniprot}}
      4. intersect with measured_uniprots → {Theme: M_T}
      5. extract_subgraph_induced_by_features(
             cogex_client,
             features=sorted(uniprot_to_hgnc_symbol(M_T_all).values()),
             max_hops=1,
             restrict_endpoints_to_features=True,
             node_filter=hgnc_symbols(M),
         ) → edge list with stmt_type + evidence_count + (optional) source_counts
      6. Map HGNC symbols on edge endpoints back to UniProt; keep edges where
         BOTH endpoints are in the SAME M_T.
      7. Deduplicate to undirected pairs (edge_id = "min|max").
      8. If require_belief: call indra_belief.noise_model.compute_edge_reliability
         per edge using source_counts.
    """


def enumerate_wasc_string_edges(
    cluster_terms_by_theme: Mapping[Theme, frozenset[str]],  # {Theme: M_T} from INDRA run
    string_adj: StringUniprotAdjacency,                       # see string_ppi.py
) -> tuple[WascEdge, ...]:
    """
    Same theme-pure within-cluster rule, but over the STRING physical-PPI graph.
    Uses the cluster-member sets pre-computed during INDRA enumeration to ensure
    the anchor universe is identical between the two networks (spec §7.2).
    """
```

### 2.2 Preprocessing

```python
# wasc/preprocess.py

@dataclass(frozen=True)
class AgeImputationModel:
    coefficients_per_group: Mapping[str, dict[str, float]]  # {"C9": {"intercept": ..., "sex": ...}, ...}
    fit_at_timestamp: str
    n_imputed_per_group: Mapping[str, int]


def fit_age_imputation(
    metadata: pd.DataFrame,
    *,
    age_col: str = "Age_at_First_PBMC_Collection",
    sex_col: str = "Sex",
    group_col: str = "diagnosis_group",
) -> AgeImputationModel:
    """Pre-registration step. Fit one regression (Age ~ Sex) per group g
    on donors with observed age. Frozen at pre-reg commit; saved to JSON."""


def impute_age_from_model(
    metadata: pd.DataFrame, model: AgeImputationModel
) -> pd.Series:
    """Apply pre-fit imputation. Returns full-coverage age series (z-scored
    within-group is applied later, at design-matrix-build time)."""


def combat_within_group(
    data: pd.DataFrame,        # (n_features, n_samples) log2 abundance
    sample_metadata: pd.DataFrame,
    *,
    group_col: str = "diagnosis_group",
    batch_col: str = "batch",
    parametric: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, NDArray]]]:
    """ComBat-style EB location/scale adjustment fit WITHIN each group g
    separately. Returns (adjusted_data, params) where params is keyed by
    (group, feature_id) for cache reuse. Implementation: wrap `combat-py`
    (PyPI: combat) if available, else native EB on within-group residuals."""
```

### 2.3 Per-edge per-group regression (Frisch-Waugh kernel)

```python
# wasc/fit.py

def build_group_residual_cache(
    group: str,
    sample_ids: Sequence[str],
    covariate_design: pd.DataFrame,   # (n_g, p_cov): intercept | Sex | Age_z | Tissue_dummies
) -> GroupResidualCache:
    """Compute M_g = I − X_cov (X_cov^T X_cov)^{-1} X_cov^T once per group.
    Stores rank, n_g, sample_ids. Residualization of protein vectors is lazy
    (added on first request)."""


def residualize_protein(
    cache: GroupResidualCache,
    protein_uniprot: str,
    expression_vec: NDArray[np.float64],   # length n_g, after batch pre-residualization
) -> NDArray[np.float64]:
    """Compute and cache p̃_g = M_g @ z(p)_g. Standardization (z-score) is
    applied to the input vector BEFORE projection (within-group mean/sd).
    Idempotent on repeated calls for the same protein."""


def fit_per_group_beta(
    edge: WascEdge,
    cache_by_group: Mapping[str, GroupResidualCache],
    batch_residualized_data: pd.DataFrame,   # output of combat_within_group, post-pre-residual
    *,
    min_n_c9: int = 10,
    min_n_other: int = 15,
) -> tuple[PerGroupFit, ...]:
    """
    For each group g, fit (target ~ anchor + covariates) via Frisch–Waugh:
        ã_g, j̃_g already in cache (or compute now).
        β̂  = (ã^T j̃) / (ã^T ã)
        RSS = j̃^T j̃ − β̂² · (ã^T ã)
        σ̂² = RSS / (n_g − rank_X_cov − 1)
        SE  = sqrt(σ̂² / (ã^T ã))
        df  = n_g − rank_X_cov − 1
    Convergence: |S_g| ≥ threshold, X_cov full rank (already verified in cache),
    SE > 0. Returns one PerGroupFit per group, including non-converged ones
    (caller decides whether to drop).
    """
```

### 2.4 Concordance statistic

```python
# wasc/concordance.py

def pooled_beta(fits: Sequence[PerGroupFit]) -> float | None:
    """β̄ = Σ w·β̂ / Σ w, where w_g = 1/SE². Returns None if fewer than
    2 converged groups."""


def cochran_q(fits: Sequence[PerGroupFit]) -> float | None:
    """Q = Σ_g w_g · (β̂_g − β̄)². Returns None if fewer than 2 converged
    groups. Computed in float64 with Kahan-stable summation guard."""


def i_squared(q: float, k: int) -> float:
    """max(0, (Q − (k−1)) / Q). Reported only as descriptor."""


def dersimonian_laird_tau2(fits: Sequence[PerGroupFit], q: float) -> float:
    """DL τ̂². Reported only as descriptor."""
```

### 2.5 Null model

```python
# wasc/null.py

@dataclass(frozen=True)
class JointBins:
    """Degree-decile × missingness-decile binning."""
    bin_of: Mapping[str, tuple[int, int]]               # uniprot → (deg_bin, miss_bin)
    members: Mapping[tuple[int, int], tuple[str, ...]]  # cell → tuple of uniprots


def build_degree_coverage_bins(
    candidate_pool: Iterable[str],            # M_T \ {anchor} \ N_a^obs for a given theme/anchor
    graph_degrees: Mapping[str, int],         # full INDRA regulatory degree per uniprot
    missing_rate_per_feature: Mapping[str, float],
    *,
    n_deg_bins: int = 10,                     # deciles
    n_miss_bins: int = 10,
) -> JointBins:
    """Deciles within the candidate pool (NOT global), so the matching
    is anchor-local-bin-relative. Edge buckets sweep to nearest neighbor
    when a cell is empty (≤ 1-bin sweep allowed; record sweep count)."""


def sample_n_matched_non_neighbors(
    anchor_uniprot: str,
    true_neighbors: frozenset[str],          # N_a^obs
    joint_bins: JointBins,
    n_to_draw: int,                          # = |N_a^obs|
    rng: np.random.Generator,
    *,
    exclude_set: frozenset[str] = frozenset(),  # always includes anchor + true_neighbors
    max_attempts: int = 100,
) -> tuple[str, ...] | None:
    """Sample n_to_draw substitute targets WITHOUT replacement, such that
    the multiset of (deg_bin, miss_bin) cells matches that of true_neighbors
    exactly. Returns None if no matching draw found in max_attempts."""


def run_anchor_null(
    anchor_uniprot: str,
    edges_for_anchor: Sequence[WascEdge],   # all WASC edges with this anchor
    cache_by_group: Mapping[str, GroupResidualCache],
    batch_residualized_data: pd.DataFrame,
    joint_bins: JointBins,
    *,
    n_permutations: int = 9999,
    rng_namespace: str = "wasc-v1.0",
    checkpoint_path: Path | None = None,
) -> AnchorNullResult:
    """
    For b = 1..B:
      1. Draw n_a substitute targets via sample_n_matched_non_neighbors.
      2. For each substitute (anchor, j'), call fit_per_group_beta, cochran_q.
      3. Record n_a null Q values for iteration b.

    Per-edge p-value (anchor-local pool):
      p(j, a) = (1 + #{Q_null_local ≤ Q_obs}) / (1 + |Q_null_local|)

    Also tracks the global pool. RNG seeded as
      seed = int(md5(anchor_uniprot + rng_namespace).hexdigest()[:8], 16)
    spawned per-iteration via SeedSequence (pattern from
    landscape._per_feature_gradient_loop). Checkpoint writes one JSONL record
    per anchor with all B null Qs (fsync per write).
    """


def run_all_anchor_nulls_parallel(
    anchors: Sequence[str],
    edges_by_anchor: Mapping[str, Sequence[WascEdge]],
    cache_by_group: Mapping[str, GroupResidualCache],
    batch_residualized_data: pd.DataFrame,
    joint_bins: JointBins,
    *,
    n_permutations: int = 9999,
    rng_namespace: str = "wasc-v1.0",
    checkpoint_dir: Path,
    n_jobs: int = -1,                          # MANDATORY at |E_WASC|=944
) -> Mapping[str, AnchorNullResult]:
    """
    joblib.Parallel(n_jobs=n_jobs, backend='loky') over anchors.
    Required at |E_WASC|=944 — sequential execution is infeasible.
    Per-anchor checkpointing makes resume-on-failure trivial.
    """
```

### 2.6 Empirical Brown's combination

```python
# wasc/combination.py

def empirical_brown(
    p_observed: Sequence[float],          # per-edge p-values for this anchor (length n_a)
    p_null_matrix: NDArray[np.float64],   # (B, n_a) — per-permutation p-values for same edges
) -> tuple[float, float, float, float]:
    """
    Returns (chi2_obs, df_brown, scale_c, p_brown).
    Empirical covariance of −2 log(p_i), −2 log(p_j) is taken across rows of
    p_null_matrix (Poole et al. 2016 variant). Falls back to Fisher's
    χ²_{2n_a} if all off-diagonal covariances are within machine ε of 0.
    """


def fisher_combine(p_observed: Sequence[float]) -> tuple[float, int, float]:
    """Sensitivity-only. Returns (chi2, df=2·k, p)."""
```

### 2.7 Three-contrast decomposition

```python
# wasc/contrasts.py

# At |E_WASC| = 944, the C2 contrast-specific floor is:
#   C2_FLOOR = math.ceil(0.05 * 944) = 48
# i.e. each C9-vs-X contrast must show ≥ 48 contrast-specific edges to satisfy C2.
C2_CONTRAST_SPECIFIC_FLOOR: int = 48


def decompose_three_contrast(
    edge_results: Sequence[EdgeResult],
    anchor_to_null_two_group: Mapping[
        tuple[str, frozenset[str]],   # (anchor, group_pair) → 2-group null Qs
        NDArray[np.float64],
    ],
    *,
    fdr_q: float = 0.10,
    ratio_threshold: float = 3.0,
    contrast_specific_floor: int = C2_CONTRAST_SPECIFIC_FLOOR,
    string_decision: StringComparison | None,   # for C3
) -> ContrastDecomp:
    """
    For each edge, compute 2-group Q for {C9,SPOR}, {C9,CTRL}, {SPOR,CTRL}.
    BY-FDR within each contrast separately. Then evaluate C1..C4 (§8) with the
    pre-registered C2 floor of 48 contrast-specific edges (= ceil(0.05·944)).
    Two-group null Qs are precomputed during run_anchor_null (cheap: same
    substitutes, just restricted to k=2 fits).
    """
```

### 2.8 STRING control

```python
# wasc/string_control.py

def compare_indra_vs_string(
    indra_results: WascResults,
    string_results: WascResults,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> StringComparison:
    """
    Q̃ = median Q over q ≤ 0.10 positives in each network.
    BCa 95% CI on ΔQ = Q̃_STRING − Q̃_INDRA via 1000 edge-resamples per network,
    then differenced. Decision per spec §7.3.
    """
```

### 2.9 STRING loader

```python
# knowledge/string_ppi.py

@dataclass(frozen=True)
class StringUniprotAdjacency:
    """UniProt-collapsed STRING physical-PPI adjacency."""
    neighbors_of: Mapping[str, frozenset[str]]   # uniprot → its STRING-neighbor uniprots
    combined_score_max_of: Mapping[
        tuple[str, str], int
    ]                                            # max combined_score across collapsed ENSP-edges
    n_covered_uniprots: int                      # for gate 3
    source_file_sha256: str
    mygene_query_date: str
    score_threshold: int                          # 700


def load_string_physical_uniprot_adj(
    string_links_path: Path,
    measured_uniprots: frozenset[str],
    *,
    combined_score_threshold: int = 700,
    cache_path: Path | None = None,             # data/wasc/string_uniprot_adj_v1.json
) -> StringUniprotAdjacency:
    """Load + filter STRING physical links. Map ENSP → UniProt via mygene
    (one batch call, cached to local pickle on first run). Collapse to UniProt
    adjacency: edge between u1, u2 iff any (e1, e2) with e1 ∈ ENSP(u1),
    e2 ∈ ENSP(u2). Cache the result keyed on file SHA + measured_uniprots SHA."""
```

### 2.10 Orchestrator

```python
# wasc/api.py

def run_wasc(
    expression_matrix: pd.DataFrame,             # (n_features, n_samples), log2, post-normalization
    metadata: pd.DataFrame,                      # donor rows; includes batch, age, sex, tissue
    *,
    cogex_client: CoGExClient,
    cluster_terms: list[tuple[Theme, str]] = DEFAULT_8_TERMS,
    n_permutations: int = 9999,                  # primary; sensitivities may use 999
    fdr_alpha: float = 0.10,
    output_dir: Path,
    resume: bool = False,
    n_jobs: int = -1,                            # MANDATORY joblib parallel at |E_WASC|=944
    rng_namespace: str = "wasc-v1.0",
    prereg_manifest_path: Path | None = None,
) -> WascResults:
    """
    Phase A. Pre-registration check: load manifest (or fail loud).
             Verifies |E_WASC| == 944 (per-theme 434/443/67) and SHA-256 of
             cluster_members_v1.json. Mismatch → refuse-loud abort.
    Phase B. Edge enumeration (INDRA). Sanity gate 1.
    Phase C. Donor exclusion + age imputation + within-group ComBat batch
             pre-residualization. Sanity gate 5.
    Phase D. Build GroupResidualCache for each group.
    Phase E. joblib-parallel per-anchor: residualize all needed protein vectors,
             fit_per_group_beta for each true edge, build joint bins,
             run_anchor_null with checkpoint. Sanity gate 4.
    Phase F. Empirical-Brown per-anchor. BY-FDR per-edge AND per-anchor.
    Phase G. Three-contrast decomposition (C2 floor = 48).
    Phase H. Write results.
    """


def run_wasc_string_control(
    indra_results: WascResults,
    expression_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    string_links_path: Path,
    output_dir: Path,
    n_permutations: int | None = None,           # default: same as INDRA run
    n_jobs: int = -1,
) -> WascResults:
    """Re-runs Phases B (with STRING enumeration) and E–G on the same
    anchors using STRING edges. Sanity gate 3.  Then calls
    compare_indra_vs_string and attaches the result to indra_results."""


def run_sanity_gates(
    e_wasc_count: int,
    label_shuffle_fp_rates: Sequence[float],
    string_coverage: float,
    convergence_rate: float,
    excluded_donor_set: frozenset[str],
    prereg_excluded: frozenset[str],
) -> SanityGates:
    """Pure function; raises no exceptions — just returns the gates dataclass.
    The orchestrator decides whether to abort or warn."""


def run_calibration_tripwire(
    expression_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    cogex_client: CoGExClient,
    e_wasc_anchors: Sequence[str],
    *,
    output_dir: Path,
    n_jobs: int = -1,
) -> CalibrationTripwireResult:
    """M2.5 — four-pronged calibration tripwire (HARD HALT on any failure).
    Returns a result with per-prong pass/fail and the offending statistic;
    the orchestrator aborts M3 if any prong fails."""
```

---

## 3. Data structures

See `data_structures` section. All dataclasses are frozen except the residualization caches (which grow mutably during a run). Edge identity is the **unordered** pair via `edge_id = f"{min(a, j)}|{max(a, j)}"`; this is the canonical key for joining across the per-edge table, the anchor table, the contrast table, and the STRING comparison.

A new frozen dataclass `CalibrationTripwireResult` (in `wasc/sanity.py`) records the four prongs:

```python
@dataclass(frozen=True)
class CalibrationTripwireResult:
    label_shuffle_fp_rate: float
    label_shuffle_stratified_pass: bool            # prong (a)
    spor_downsampled_overlap: float
    spor_downsampled_pass: bool                    # prong (b)
    allprotein_pool_positive_ratio: float
    allprotein_pool_pass: bool                     # prong (c)
    fw_vs_ols_max_abs_diff: float
    fw_vs_ols_pass: bool                           # prong (d)
    overall_pass: bool                              # AND over the four
    tripwire_timestamp: str
```

---

## 4. Test plan

All tests live under `tests/wasc/`. Mirror the module split for unit tests; add an `integration/` subdir.

### 4.1 Unit tests (`tests/wasc/`)

**`test_edges.py`**
- `test_enumerate_indra_edges_within_theme_only`: synthetic CoGEx mock returning known hop-1 edges spanning two themes; assert cross-theme edges are excluded.
- `test_enumerate_indra_edges_undirected_dedupe`: assert {a,j} and {j,a} collapse to one WascEdge with lexicographic id.
- `test_enumerate_indra_edges_requires_both_endpoints_measured`: mock with one endpoint outside M_T; assert excluded.
- `test_enumerate_indra_edges_v1_count_lock`: with the v1.0 pre-registered manifest mock, assert `|E_WASC| == 944` and per-theme counts (434/443/67).
- `test_enumerate_string_edges_uses_indra_membership`: STRING adjacency given, assert only within-theme edges produced.
- `test_string_loader_filters_combined_score`: load a 10-line synthetic STRING file; assert combined_score < 700 dropped.
- `test_string_loader_ensp_to_uniprot_collapses_correctly`: synthetic ENSP1→ENSP2 maps to UniProt {U1}→{U2,U3}; assert edges {U1,U2} and {U1,U3} appear with max combined_score.
- `test_belief_population`: assert `edge_reliability` field is populated when `require_belief=True` and `source_counts_json` present.

**`test_preprocess.py`**
- `test_combat_within_group_invariance`: synthetic data with known batch effect; assert post-ComBat between-batch means within ε within group.
- `test_combat_within_group_does_not_leak_across_groups`: assert ComBat params for group A do not affect group B residuals (each group fit independently).
- `test_age_imputation_uses_pre_fit_coefficients`: fit on one set, apply to another; assert reproducibility byte-for-byte.

**`test_fit.py`**
- `test_frisch_waugh_matches_full_ols_synthetic`: build synthetic y, X = [1, anchor, sex, age, tissue], fit via Frisch–Waugh and via `statsmodels.OLS`; assert β, SE agree to 1e-10.
- `test_fit_per_group_beta_handles_singletons`: anchor with constant value within group; assert non-converged, β=nan, se=nan.
- `test_fit_per_group_beta_respects_min_n_thresholds`: synthetic group with n=9 (C9 needs ≥10); assert converged=False.
- `test_residualize_protein_idempotent`: call twice; assert second call hits cache (instrumented).

**`test_concordance.py`**
- `test_cochran_q_zero_under_identical_betas`: three groups with same β; assert Q ≈ 0.
- `test_cochran_q_known_value`: hand-computed example; assert Q matches to 1e-12.
- `test_cochran_q_returns_none_under_one_group`: only 1 converged; assert None.
- `test_cochran_q_invariant_under_anchor_target_swap`: fit (j ~ a) and (a ~ j) on synthetic; assert Q identical (β flips sign but Q is invariant under that flip when β̄ flips too).
- `test_pooled_beta_inverse_variance_weighted`: hand-computed example; check β̄ matches.

**`test_null.py`**
- `test_build_degree_coverage_bins_decile_split`: 100 uniprots with known degree and missingness; assert 10×10 cells populated correctly.
- `test_sample_n_matched_non_neighbors_excludes_true_neighbors`: assert no overlap.
- `test_sample_n_matched_non_neighbors_matches_joint_distribution`: assert the multiset of (deg, miss) cells matches the true-neighbor multiset.
- `test_sample_n_matched_non_neighbors_no_replacement`: assert uniqueness within draw.
- `test_sample_n_matched_non_neighbors_returns_none_on_exhaust`: too-small candidate pool; assert None after max_attempts.
- `test_anchor_null_rng_reproducibility`: same anchor_uniprot, same rng_namespace → identical null Q draws across runs and across processes.
- `test_anchor_null_checkpoint_roundtrip`: write, kill, resume; assert no recomputation and identical final p-values.
- `test_anchor_null_phipson_smyth_p_value`: synthetic null with known support; assert p formula `(1 + #) / (1 + B)`.
- `test_run_all_anchor_nulls_parallel_matches_serial`: with B=99 on 5 synthetic anchors, joblib parallel result equals serial result byte-for-byte.

**`test_combination.py`**
- `test_empirical_brown_reduces_to_fisher_under_zero_covariance`: independent p-values; assert p_brown = p_fisher to numerical precision.
- `test_empirical_brown_increases_p_under_positive_covariance`: dependent p-values; assert p_brown > p_fisher.
- `test_empirical_brown_one_test_returns_input`: n_a=1; assert p_brown = p_observed.

**`test_contrasts.py`**
- `test_two_group_q_is_squared_standardized_diff`: synthetic; assert Q = (β1−β2)² / (SE1²+SE2²).
- `test_c2_ratio_check`: construct R sizes that satisfy / violate the 3× rule; assert decision matches.
- `test_c2_floor_locked_to_48`: assert `C2_CONTRAST_SPECIFIC_FLOOR == 48` and that a contrast with exactly 47 contrast-specific edges fails C2 at the floor (regardless of ratio).
- `test_c4_binomial_check`: construct R_SPOR_CTRL = 0.10·|E_WASC| (= 94) exactly; assert C4 passes.
- `test_c9_specific_requires_all_four_conditions`: each Ci individually fails; assert c9_specific=False; all pass → True.

**`test_string_control.py`**
- `test_delta_q_bca_excludes_zero_under_synthetic_indra_specific_data`: construct INDRA-positive edges with low Q and STRING-positive with high Q; assert decision="INDRA-SPECIFIC".
- `test_delta_q_bca_straddles_zero_under_null`: identical Q distributions; assert decision="INCONCLUSIVE".
- `test_string_underpowered_branch`: gate 3 below 0.60; assert decision="STRING-UNDERPOWERED" (orchestrator).

**`test_sanity.py`**
- `test_gate1_edge_count_band`: count inside / outside ±30% of 944; assert bool.
- `test_gate2_label_shuffle_fp_rate_bound`: synthetic shuffle rates; assert bound.
- `test_gate5_donor_set_equality_strict`: extra or missing donor → False.
- `test_calibration_tripwire_halt_on_any_prong_failure`: construct each prong failing individually; assert `overall_pass=False` and orchestrator aborts.

### 4.2 Integration tests (`tests/wasc/integration/`)

**`test_run_wasc_end_to_end_synthetic.py`**
Builds a tiny synthetic universe:
- 30 measured proteins; 3 themes with 10 members each; ~15 known hop-1 within-theme edges per theme via a mock CoGEx client.
- 30 donors: 10 C9, 15 SPOR, 5 CTRL (small but exercises 3-group convergence).
- Synthetic expression: for 5 designated "invariant" edges, β identical across groups; for 5 "variant" edges, β differs across groups; rest random.
- B = 199 permutations (fast).
- Asserts:
  - WASC-positive set includes ≥ 4 of 5 invariant edges, ≤ 1 of 5 variant edges (sensitivity floor, specificity ceiling).
  - Per-anchor Brown's p-values are well-defined and monotone in per-edge support.
  - BY-FDR q-values are >= raw p-values.
  - Three-contrast decomposition runs to completion.

**`test_run_wasc_known_invariant_pathway.py`** (positive control)
- Use the housekeeping-gene-like subset (ribosomal proteins or NF-κB regulon subset already in INDRA hop-1) as a "known invariant" cluster. Run on real data restricted to that subset.
- Expected: high fraction of edges with q ≤ 0.10 (invariant); low τ²; STRING ΔQ noisy because both networks contain ribosomal proteins.
- This is a *qualitative* positive control; assert "≥ 30% positives" rather than a precise number.

**`test_run_wasc_random_anchors_negative_control.py`** (negative control)
- Sample 220 random within-cluster edges from a non-cluster theme (e.g., metabolism — present in measured proteome but not in the 8 pre-reg terms).
- Expected: ~10% positives by construction (uniform under H0 at q = 0.10).
- Assert fraction of positives is within 0.10 + 3·√(0.10·0.90/220) ≈ 0.17.

**`test_run_wasc_label_shuffle_null_calibration.py`**
- Run 20 label-shuffles (preserve group sizes) on the real data; assert mean false-positive rate ∈ [0.05, 0.14], stratified by degree decile AND SE quintile (M2.5 prong a).
- This is slow but is the load-bearing null-calibration check.

**`test_run_wasc_string_control_integration.py`**
- Load real STRING file; run STRING enumeration on the real cluster members; assert coverage ≥ 0.70 (gate 3) on the real measured set.
- Run a mini WASC (B = 199) on STRING; assert run completes; assert `compare_indra_vs_string` returns a well-formed `StringComparison`.

**`test_calibration_tripwire_end_to_end.py`** (M2.5)
- Run all four prongs against a synthetic harness sized to mimic the real run topology.
- Assert each prong returns a well-formed result and that overall pass/fail is the conjunction.

### 4.3 Numerical regression tests

**`test_frisch_waugh_vs_statsmodels_real_data.py`** (M1 numerical-identity gate)
- Sample 50 real (anchor, target, group) triples from the proteome.
- Fit Frisch–Waugh kernel vs `statsmodels.OLS` with full design (including explicit batch dummies on the SPOR group).
- Assert β, SE, df agree to **1e-8** (float64). This test is the gate; it MUST pass before M2 begins.

### 4.4 Sanity gates as tests

The five sanity gates from spec §12 are run as automated tests in `tests/wasc/integration/test_sanity_gates.py`, invoked against the latest committed real-data run. The build is marked invalid if any gate fails. The M2.5 calibration tripwire is run as a separate test in `test_calibration_tripwire_end_to_end.py` and is a HARD HALT for the build pipeline.

---

## 5. Build sequence (milestones)

Each milestone is a self-contained vertical slice with a definition of done (DoD), dependencies, and an effort estimate. Effort is calendar-hours of focused work (not wall-clock); double for ambient overhead.

**REORDERING NOTE (v1.0):** M6a (pre-registration manifest + git tag) now sits BETWEEN M1 and M2. M1 produced |E_WASC| = 944 on 2026-06-02; M6a freezes that count and all input SHAs to the git tag `wasc-prereg-v1.0`. No anchor-local null Q values may be computed on real data until M6a is complete. M2 is blocked on M6a.

### Milestone 1 — Edge enumeration + per-group regression + numerical-identity gate

**STATUS: edge enumeration COMPLETE 2026-06-02 (|E_WASC| = 944).**

**DoD.**
- `wasc/edges.py::enumerate_wasc_indra_edges` returns the frozen `E_WASC` UniProt-pair list given a real CoGEx client + measured set + the 8 TERMS. **DONE: 944 edges** (434 Splicing / 443 Chromatin / 67 Transport); saved to `data/wasc/cluster_members_v1.json` and `data/wasc/e_wasc_v1.json` (E2, E4).
- `wasc/preprocess.py::fit_age_imputation` + `combat_within_group` run on real data; output saved to `data/wasc/age_imputation_v1.json` (E8) and a temp adjusted matrix.
- `wasc/fit.py::build_group_residual_cache` + `fit_per_group_beta` produce `PerGroupFit` objects for ≥ 90% of `E_WASC × {C9, SPOR, CTRL}` regressions on real data.
- A smoke script (`scripts/wasc/smoke_fit.py`) prints a 10-row table of `(edge_id, β_C9, SE_C9, β_SPOR, SE_SPOR, β_CTRL, SE_CTRL)` to stdout.
- **NEW (v1.0 gate):** `tests/wasc/test_frisch_waugh_vs_statsmodels_real_data.py` passes on 50 real triples to 1e-8 with explicit batch dummies on the SPOR group. **M1 is not complete until this gate passes.**
- All unit tests in `test_edges.py`, `test_preprocess.py`, `test_fit.py` pass.

**Dependencies.** None (uses only existing utilities: `extract_subgraph_induced_by_features`, `build_covariate_design_matrix`, `TERMS`, `fetch_term_members_via_indra`, `hgnc_ids_to_uniprots`).

**Effort.** 22 h (was 18 h; +4 h for the numerical-identity gate and the SPOR batch-dummy reconstruction).
- 5 h edges (DONE).
- 4 h ComBat-within-group (decide: wrap PyPI `combat` vs hand-roll EB; hand-roll is ~80 lines).
- 3 h age imputation (trivial sklearn LinearRegression per group, but pre-reg JSON serialization needs care).
- 4 h Frisch–Waugh kernel + caches.
- 4 h numerical-identity gate (real-data triple sampler, statsmodels OLS harness with batch dummies, tolerance hardening to 1e-8).
- 2 h smoke + unit tests.

### Milestone 6a — Pre-registration manifest + git tag (BLOCKS M2)

**DoD.**
- `scripts/prepare_wasc_prereg.py` runs end-to-end, producing all E1–E8 artifacts under `data/wasc/`, computes SHA-256 of each, writes `data/wasc/manifest_v1.json` listing all SHAs and the locked counts `{|E_WASC|: 944, splicing: 434, chromatin: 443, transport: 67, n_cluster_members: 377, measured_proteome: 3264}`.
- A git tag `wasc-prereg-v1.0` is applied to the commit that includes these artifacts.
- `WascManifest` dataclass populated from the JSON on every subsequent run; mismatch → loud failure (refuse-loud pattern from `wave_24h_landscape_resume`).
- Memory entry `wave_25_wasc_spec.md` written, summarizing: question, decision rules, what a positive / negative / inconclusive result licenses (with the §9 ceiling verbatim), pre-reg manifest location, integration map to M1–M7 of this plan.
- `data/wasc/spec_v1.0.md` is committed at the same tag (the full statistical spec).

**Dependencies.** M1 complete.

**Effort.** 6 h.
- 3 h `prepare_wasc_prereg.py` orchestration + SHA-256 fingerprinting.
- 2 h `wave_25_wasc_spec.md` (mirrors the wave_24l template).
- 1 h `WascManifest` validation on load (refuse-loud on any drift, including |E_WASC| ≠ 944).

### Milestone 2 — Concordance + permutation null (joblib-parallel, BLOCKED ON M6a)

**DoD.**
- `wasc/concordance.py::cochran_q`, `pooled_beta`, `i_squared`, `dersimonian_laird_tau2` implemented; unit tests pass.
- `wasc/null.py::build_degree_coverage_bins`, `sample_n_matched_non_neighbors`, `run_anchor_null`, `run_all_anchor_nulls_parallel` implemented; unit tests pass including reproducibility-across-processes and checkpoint roundtrip and parallel-equals-serial.
- A second smoke script (`scripts/wasc/smoke_null.py`) runs the full pipeline on a single anchor with B = 199 in < 60 s; prints per-edge raw permutation p-values.
- joblib parallel implementation in `run_all_anchor_nulls_parallel` (n_jobs=-1 default) is wired and tested.
- The "label shuffle null calibration" integration test runs (slow; not required to pass at this milestone, but must run to completion — it is itself prong (a) of M2.5).

**Dependencies.** M1, M6a (cannot touch real data null draws until the manifest is tagged).

**Effort.** 28 h (was 22 h; +6 h for joblib infrastructure and per-anchor checkpoint sharding to make the 944-edge load tractable).
- 3 h concordance (small functions, but Kahan summation + numerical edge cases need care).
- 4 h joint bin builder (the empty-cell sweep logic + decile boundaries are subtle).
- 4 h matched non-neighbor sampler (without replacement + multiset matching is the trickiest single function in the build).
- 6 h anchor null loop (lift the md5-seeded pattern from `landscape.py:1033`, adapt checkpoint format).
- 6 h joblib parallel orchestration (`run_all_anchor_nulls_parallel`, per-anchor JSONL checkpoint sharding, resume-on-failure semantics, `loky` backend smoke against real Neo4j cache).
- 3 h two-group null bookkeeping (record per-iteration 2-group Qs alongside the 3-group ones — same substitutes, just restricted fits).
- 2 h smoke + unit tests.

### Milestone 2.5 — Calibration tripwire (HARD HALT)

**DoD.**
- `wasc/sanity.py::run_calibration_tripwire` and `scripts/wasc/run_tripwire.py` implemented.
- All four prongs evaluated against the real-data run:
  - (a) Label-shuffle FP rate ∈ [0.05, 0.14], stratified by degree decile AND SE quintile;
  - (b) Down-sampled-SPOR-to-25 overlap with full-N WASC-positive set ≥ 70%;
  - (c) All-measured-protein-pool null does not explode positives (>3× = over-restrictive within-theme null);
  - (d) Frisch-Waugh kernel vs `statsmodels.OLS` with explicit batch dummies on SPOR group — identity check to 1e-8 on the production batch design.
- Tripwire result serialized to `output/wasc/{run_label}/tripwire.json` with per-prong stat and pass/fail.
- **HARD HALT semantics:** if `overall_pass=False`, the orchestrator MUST abort before M3 (BY-FDR + combination) is applied to real data.
- `tests/wasc/integration/test_calibration_tripwire_end_to_end.py` passes.

**Dependencies.** M2.

**Effort.** 14 h.
- 4 h prong (a): stratified shuffle harness (degree-decile × SE-quintile cells, FP-rate calculation per cell, pooled bound test).
- 3 h prong (b): SPOR down-sampling harness (sample 25 from SPOR, rerun WASC, compute Jaccard overlap with full-N positive set).
- 3 h prong (c): all-protein-pool null variant (override `build_degree_coverage_bins` candidate pool to be the full measured proteome, not M_T); ratio check vs theme-restricted positives.
- 2 h prong (d): production-design batch dummies harness (extends the 50-triple gate from M1 to the production SPOR batch structure).
- 2 h tripwire dataclass + orchestrator abort semantics + tests.

### Milestone 3 — BY-FDR + per-anchor combination

**DoD.**
- `wasc/combination.py::empirical_brown` + `fisher_combine` implemented; unit tests pass including the "reduces to Fisher under independence" property.
- `wasc/api.py::run_wasc` Phase F (BY-FDR per-edge + per-anchor + empirical Brown) wired.
- A WASC results object is produced end-to-end on real data at B = 999 (development fallback); orchestrator output dir contains `edges.csv`, `anchors.csv`, `manifest.json`.
- Primary claim framing: **"count + cluster pattern at q = 0.10"** appears in the result language; per-edge mechanism claims are explicitly NOT made.
- Integration test `test_run_wasc_end_to_end_synthetic.py` passes.
- `re-export through stats/__init__.py` done.

**Dependencies.** M2, M2.5 (tripwire MUST have passed).

**Effort.** 14 h.
- 5 h empirical Brown's (Poole et al. 2016 — the empirical-covariance estimator from null draws is the load-bearing detail).
- 4 h Phase F orchestration (FDR application is one-line per `fdr_correction(method="BY")` calls, but wiring per-anchor + per-edge tables in parallel needs structure).
- 2 h CSV / JSON writers.
- 3 h integration test.

### Milestone 4 — STRING-PPI control

**DoD.**
- `knowledge/string_ppi.py::load_string_physical_uniprot_adj` implemented and loads the v12.0 file from `data/string/9606.protein.physical.links.v12.0.txt.gz` in < 60 s; ENSP→UniProt cache is durable (pickled with file SHA + measured-set SHA keys); unit tests pass.
- `wasc/edges.py::enumerate_wasc_string_edges` produces `E_STRING` from the cached adjacency + the same M_T sets.
- `wasc/string_control.py::compare_indra_vs_string` implemented with BCa bootstrap; unit tests pass.
- `wasc/api.py::run_wasc_string_control` runs end-to-end at B = 9999 with joblib parallel; produces `string_comparison.json`.
- Sanity gate 3 evaluated and recorded.

**Dependencies.** M3 (needs an INDRA `WascResults` to compare against).

**Effort.** 16 h.
- 6 h STRING loader (file is 8.7 MB compressed but ENSP→UniProt via mygene needs batching, retry, and caching to a local pickle).
- 4 h `enumerate_wasc_string_edges` (mostly reuses the within-theme machinery).
- 4 h BCa bootstrap on ΔQ (decision rule with three branches + 1000-resample bootstrap — use `scipy.stats.bootstrap(method='BCa')`).
- 2 h orchestrator wiring + sanity gate 3.

### Milestone 5 — Three-contrast decomposition

**DoD.**
- `wasc/contrasts.py::decompose_three_contrast` implemented with `C2_CONTRAST_SPECIFIC_FLOOR = 48`; unit tests pass.
- C1–C4 evaluated and serialized; `contrasts.csv` written.
- "C9-specific wiring change" boolean is computed end-to-end on real data using the locked C2 floor of 48 (= ceil(0.05 × 944)).

**Dependencies.** M4 (C3 needs `StringComparison`).

**Effort.** 8 h.
- 4 h two-group Q + per-contrast BY-FDR (mostly bookkeeping; the 2-group Q is closed-form for k=2).
- 3 h C1–C4 decision logic (one-sided binomial for C4 via `scipy.stats.binom.cdf`; C2 floor locked at 48).
- 1 h unit tests + CSV writer.

### Milestone 5.5 — Mandatory pre-registered sensitivities batch

**DoD.**
- All FIVE pre-registered sensitivities run regardless of primary outcome and serialized to `output/wasc/{run_label}/sensitivities/`:
  1. **T-Cell-stratified** — re-run with a T-cell-composition proxy added as a covariate (PBMC composition proxy from existing metadata if available; else mark as best-effort with a TODO).
  2. **iPSC-retained** — re-run with the 20 iPSC/external donors retained (not excluded), to bound exclusion-rule sensitivity.
  3. **batch-correction-OFF** — re-run with `combat_within_group` skipped; raw within-group log2 abundance only.
  4. **down-sampled-SPOR-to-25** — re-run with SPOR randomly down-sampled to n=25 (10 independent draws; report distribution of WASC-positive counts).
  5. **all-protein-pool** — re-run with the null candidate pool set to the full measured proteome (M), not the theme-restricted set (M_T). Reports whether the positive count is robust to relaxing the within-theme matching.
- **B = 999** is permitted for these reruns; primary INDRA + STRING stay at B = 9999.
- **B = 99999 rerun mandatory** if any floor-tied edges exist at primary B = 9999 (i.e., any edge whose raw p-value is exactly at the 1/(B+1) = 1e-4 floor).
- Each sensitivity writes a result row to a master comparison table `sensitivities_summary.csv` with columns `(sensitivity, n_positive, jaccard_to_primary, c9_specific_bool, notes)`.

**Wall-clock budget.** ~25–40 h per primary run (already counted in M3); ~120–200 h sensitivities batch total. Run on a multi-core machine with `n_jobs=-1`; budget 5–7 calendar days for the sensitivities sweep.

**Dependencies.** M5 (needs full primary WascResults including ContrastDecomp).

**Effort.** 16 h focused-implementation (in addition to the wall-clock budget).
- 3 h T-cell covariate harness.
- 2 h iPSC-retained donor-list switch.
- 2 h batch-correction-OFF orchestrator branch.
- 3 h SPOR down-sample loop + Jaccard summary.
- 2 h all-protein-pool null candidate override.
- 2 h floor-tie detector + B=99999 rerun branch.
- 2 h `sensitivities_summary.csv` writer + comparison plots.

### Milestone 6 — Wave-style memory entry (post-tag, post-run)

> **NOTE:** the pre-registration tag itself is M6a (placed before M2). M6 here is the post-result memory entry.

**DoD.**
- Memory entry `wave_25_wasc_result.md` written, paralleling `wave_24k_cluster_claim_consolidated.md` in structure; includes per-prong tripwire outcomes and the full sensitivities batch summary.

**Dependencies.** M5.5 (sensitivities batch complete).

**Effort.** 4 h.

### Milestone 7 — Report integration (conditional, NEVER blocked by negative result)

**DoD.**
- An `§11.5 — WASC` section added to `c9_triangulation_report.md`, populated with the per-edge / per-anchor counts (at |E_WASC| = 944), the STRING ΔQ result, the three-contrast decomposition (with C2 floor 48), the §9 claim ceiling, and explicit "what we did NOT show" lines lifted verbatim from spec §9 forbidden-language list.
- **Disposition rule (v1.0):** a negative WASC is a **publishable finding of structural-coupling-flatness**, not a retraction of wave_24l. Report integration is conditional on result, **not blocked by it**:
  - **Positive WASC** (≥ 5% per-edge q ≤ 0.10 AND c9_specific=True AND STRING decision ≠ "INDRA-WEAKER"): full §11.5 with cluster-pattern claim at the §9 ceiling.
  - **Negative WASC** (none of the above): §11.5 framed as "structural-coupling-flatness" — wave_24l's per-feature gradient claim stands; WASC adds the orthogonal finding that coupling structure within those clusters is not detectably C9-specific.
  - **Inconclusive WASC** (tripwire prong failure, or BCa CI straddles zero, or STRING-underpowered): §11.5 documents the inconclusive outcome and lists which prong / branch caused it; no claim either way.
- Updated `output/c9_triangulation_report.html` regenerated in all three cases.
- All result language passes brutalist review against spec §9 forbidden-language list.

**Dependencies.** M5.5 (full sensitivities batch + tripwire result), real-data run completed.

**Effort.** 10 h (most of this is brutalist review + writing the result language to spec §9 ceiling, not coding).

### Total estimated effort

**180–220 h focused implementation + brutalist review cycles** (was ~140 h). Driven by:
- 3-axis bin builder + matched non-neighbor sampler hardening at scale (|E_WASC| = 944).
- ComBat-EB-within-group hand-roll.
- STRING loader + mygene caching from scratch.
- Empirical Brown's with full per-anchor null-matrix accounting.
- M1 numerical-identity gate against statsmodels OLS with batch dummies.
- M2 mandatory joblib parallel infrastructure + per-anchor checkpoint sharding.
- M2.5 four-prong calibration tripwire with hard-halt semantics.
- M5.5 mandatory sensitivities batch (5 reruns + floor-tied B=99999 conditional).
- M7 conditional report-integration logic (3 branches: positive / negative / inconclusive).

At sustainable pace (~25 h/week): **7–9 calendar weeks** end-to-end including the ~25–40 h primary wall-clock and the 120–200 h sensitivities sweep.

---

## 6. Integration with the existing pipeline

**Position relative to landscape.** WASC is a **parallel test family**, not a downstream consumer of the landscape gradient pipeline. It shares: (a) the cluster-term manifest (`scripts/viz/common.py::TERMS`), (b) the measured-only INDRA distance matrix from a saved landscape run (re-used via `distances.meta.json` SHA-256 pinning), (c) the proteome matrix and metadata. It does NOT consume landscape per-feature slopes — WASC's signal is a **covariance-structure invariance**, while landscape's signal is a **perturbation-decay gradient**. They are independent statistical lenses on the same cluster.

**Position relative to panels.** WASC is NOT a panel. Panels test individual seeds against degree-matched control seeds under a single contrast; WASC tests within-cluster *edge pairs* across all three groups simultaneously. WASC borrows the `ProcessPoolExecutor` parallelism pattern and the per-anchor RNG seeding pattern from `panels/landscape.py` but lives in `stats/wasc/`, not `panels/`. The `panels/` module is for seed-based gradient tests; WASC is for edge-based concordance tests. WASC's parallelism is `joblib` (loky backend) rather than raw `ProcessPoolExecutor` to fit cleanly with the per-anchor checkpoint sharding.

**Position relative to differential / ROAST.** WASC is orthogonal. ROAST tests gene-set enrichment under a condition contrast; WASC tests within-edge slope invariance across groups. They could in principle be combined into a meta-analysis (covariance-of-effect-sizes), but that is out of scope for v1.

**Calling convention.** A single orchestrator `scripts/run_wasc.py` runs the entire pipeline. It is invoked once with `--phase prereg` to freeze inputs (M6a), once with `--phase indra` to run the primary test, once with `--phase tripwire` to run the M2.5 calibration tripwire (HARD HALT on failure), once with `--phase string` to run the negative control, once with `--phase sensitivities` to run the M5.5 batch, and once with `--phase report` to emit the report fragment. Each phase is idempotent and checkpointed; re-running with the same manifest is a no-op if completed.

**Memory entry placement.**
- `memory/wave_25_wasc_spec.md` — pre-registration spec summary (M6a).
- `memory/wave_25_wasc_result.md` — post-run result + brutalist audit log + tripwire outcomes + sensitivities summary (M6/M7).
- `data/wasc/spec_v1.0.md` — the full §1–§12 statistical spec, frozen at git tag `wasc-prereg-v1.0`.

Both memory entries follow the wave_24l template (one-paragraph summary, then numbered findings, then "open" / "closed" tracker).

**MEMORY.md update.** After M6a, add to the project memory file the cross-reference line for `wave_25_wasc_spec.md`. After M7, add for `wave_25_wasc_result.md`. Use the existing pattern: short bullet under the "Signal Analysis & Reproducibility" section.

---

## 7. Run-time budget

At the locked |E_WASC| = 944, sequential CPU execution is **infeasible**. joblib parallel (`n_jobs=-1`) over anchors is **mandatory** for M2 and downstream.

### 7.1 Compute volume

Let |E| = 944 edges (locked), B = 9999 permutations (primary) or 999 (sensitivities), |G| = 3 groups, n_C9 ≈ 25, n_SPOR ≈ 294, n_CTRL ≈ 71.

Per real edge: 3 group-fits × 1 Frisch–Waugh univariate regression = ~3 cheap dot products (residualization is cached). ≈ 2,832 fits total for real edges.

Number of distinct anchors at |E_WASC| = 944 is approximately the unique-endpoint count across the three theme edge lists, estimated at **220–280 distinct anchors** (with ~3–5 average true neighbors per anchor across themes). Per anchor: `n_a × B ≈ 4 × 9999 ≈ 40,000` substitute-target fits per anchor; ~250 anchors total ⇒ ≈ **10 M null fits** for the 3-group test. Two-group null Qs add no extra fits — they reuse the same 3-group β̂ values.

Each Frisch–Waugh fit on residualized vectors costs ≈ 3 · n_g floating-point ops, ≈ 1 µs at n = 100. So 10 M fits × 1 µs ≈ **10 s of pure compute** for the null regressions — but the real cost is **per-anchor RNG sampling + matched-non-neighbor draws** (rejection sampling for the multiset-of-cells match) plus Python dispatch overhead.

Empirically estimate **~6–10 minutes per anchor sequential**, **~250 anchors ⇒ 25–40 hours sequential** at B = 9999.

### 7.2 Parallelism (MANDATORY)

`joblib.Parallel(n_jobs=-1, backend='loky')` over anchors. Anchors are independent given the cached `GroupResidualCache` and per-theme `JointBins`. On a 16-core machine: **~2–3 h wall-clock** for the INDRA primary run at B = 9999. On an 8-core machine: **~4–5 h**.

The cache poses a subprocess-init cost: `GroupResidualCache` for each of 3 groups is ~(n_g × n_g) float64 = ~700 KB total. The batch-pre-residualized expression matrix is ~(3264 × 390) float64 = ~10 MB. Both fit easily in shared memory; pickle-copy to each subprocess is acceptable. Use `forkserver` start method on Linux/macOS for low init cost. joblib's `loky` backend handles this cleanly.

### 7.3 Memory

- Expression matrix: ~10 MB.
- `GroupResidualCache.residualized`: max ~3264 proteins × 3 groups × 700 floats ≈ 50 MB (filled lazily; in practice only ~1000 proteins touched given |E_WASC| = 944).
- Per-anchor null Q array: B × n_a × 8 bytes ≈ 400 KB per anchor; ~100 MB total if held in memory simultaneously. Stream to disk via the JSONL per-anchor checkpoint to avoid the all-at-once allocation.
- STRING adjacency: ~5 MB UniProt-keyed dict.

Total < 300 MB per worker; comfortably fits in commodity multi-core. No GPU required; MLX backend not used.

### 7.4 STRING control wall-clock

STRING edge count will likely be **larger** than INDRA (denser physical PPI within clusters), say 1,500–2,500 edges. Same B, same parallelism on 16 cores ⇒ **~5–8 h wall-clock**.

### 7.5 Total budget (REVISED for |E_WASC| = 944)

- **Pre-reg run (M6a):** ~10 min (STRING loader + ENSP↔UniProt mygene call on first run; cached after).
- **Primary INDRA at B = 9999, n_jobs=-1, 16 cores:** ~25–40 h wall-clock (single primary run).
- **Primary STRING at B = 9999, n_jobs=-1, 16 cores:** ~5–8 h.
- **M2.5 calibration tripwire (20 label-shuffles at B = 999 + 3 other prongs):** ~15–25 h.
- **M5.5 sensitivities batch (5 sensitivities at B = 999 each):** ~60–100 h.
- **Conditional B = 99999 rerun if any floor-tied edges:** add ~250–400 h (only if triggered).
- **Total primary cycle:** ~50–80 h wall-clock.
- **Total with full sensitivities batch:** ~120–200 h wall-clock.

At B = 999 during development: divide by ~10 across the board ⇒ ~5–8 h primary; use B = 999 during M1–M5 development; promote to B = 9999 only for the pre-registered final run.

### 7.6 What NOT to GPU-accelerate

The existing `batched_ols_contrast_test` MLX path is the wrong kernel — it requires a shared X across the batch. WASC's per-(anchor, target) model violates that. The Frisch–Waugh trick reduces the per-edge fit to three CPU dot products, which is faster than GPU transfer overhead for this volume. **Do not add MLX to WASC.** If a future scale-up needs GPU, the right kernel is batched dot products of all `j̃` columns against a single `ã` column per anchor — but this is a 10× speedup on what is already a multi-hour job, so not worth the implementation cost in v1.

---

## 8. Open questions for the reviewer

> Note: open questions 4 (C2 ratio + floor) and 1 (excluded-donor list) are now CLOSED by the M1 decisions log below.

1. ~~**Excluded-donor list provenance.**~~ **CLOSED 2026-06-02:** authoritative list frozen in `data/wasc/excluded_donors_v1.json` at M6a tag; populated from regex on donor_id prefix (EDi*, CW50*, CS007, W14-16C*) and reviewed donor-by-donor.

2. **ComBat library choice.** PyPI `combat` is unmaintained (last release 2017). Two options: (a) wrap `combat-py` and freeze the version; (b) hand-roll the ~80-line within-group EB location/scale (Johnson et al. 2007 equations). **Decision: hand-roll** for reproducibility — adds 1 day to M1; locks the algorithm to a SHA-pinned implementation file in `wasc/preprocess.py`.

3. **Two-group null storage.** The two-group Q values per substitute fit are computed essentially for free during the 3-group loop. Default to keeping them all in the JSONL per-anchor checkpoint (3× storage). At ~100 MB × 3 = ~300 MB per run this is fine; flagging here for visibility.

4. ~~**Reseating the brutalist's "1×3 ratio" in §8 condition C2.**~~ **CLOSED 2026-06-02:** with |E_WASC| = 944 locked, C2 floor = `ceil(0.05 × 944) = 48` contrast-specific edges per C9 comparison. The 3× ratio AND the 48-edge floor must BOTH be satisfied. C2 floor recorded in code as `C2_CONTRAST_SPECIFIC_FLOOR = 48`.

5. **STRING-vs-INDRA cluster-member set difference.** The spec uses the **same** M_T sets for both networks (intersected with the *INDRA*-derived cluster-member set). If a STRING-only protein is in a cluster term but missing from INDRA's `fetch_term_members_via_indra` result, it is excluded from both networks. This is the right choice for a fair head-to-head, but worth flagging.

6. **Cross-theme edges (DEFERRED).** v1.0 only enumerates within-theme edges. A future exploratory secondary module may enumerate cross-theme edges (e.g., Splicing-anchor → Chromatin-target) to test whether C9-specific coupling crosses theme boundaries. Out of scope for v1.0.

---

## 9. Decisions Log (v1.0)

This log records binding decisions taken during the v0.9 → v1.0 transition. Future revisits to WASC scope must consult this log before proposing changes to any item listed here.

| Date | Decision | Rationale | Bound at |
|---|---|---|---|
| 2026-06-02 | **Lock |E_WASC| = 944** (Splicing 434 / Chromatin 443 / Transport 67). | M1 edge enumeration completed against the live INDRA-CoGEx Neo4j (bolt://indra-cogex-lb-...:7687). Within-theme densities (2.4% / 4.2% / 7.8%) are biologically sane. | git tag `wasc-prereg-v1.0` (M6a) |
| 2026-06-02 | **C2 contrast-specific floor = 48** edges per C9 comparison (= `ceil(0.05 × 944)`). | Pre-registered as a function of |E_WASC|; locked once |E_WASC| was locked. Encoded as `C2_CONTRAST_SPECIFIC_FLOOR = 48` in `wasc/contrasts.py`. | git tag `wasc-prereg-v1.0` (M6a) |
| 2026-06-02 | **BY-FDR q = 0.10 retained**, primary claim reframed as "count + cluster pattern" rather than per-edge mechanism. | At |E_WASC| = 944 the per-edge mechanism claim is not licensable from a structural-coupling-invariance test. The claim ceiling in spec §9 caps inference at "passed structural-coupling-invariance test in pool of 944". | spec §9 + this plan |
| 2026-06-02 | **M2 anchor loop MUST be joblib-parallelized** (`n_jobs=-1`, mandatory not optional). | At |E_WASC| = 944, sequential execution is ~25–40 h wall-clock per run. Parallel is the only feasible path. Encoded as default in `run_all_anchor_nulls_parallel`. | M2 DoD |
| 2026-06-02 | **Sensitivity reruns may use B = 999**; primary INDRA + STRING stay at B = 9999. | The 1/1000 floor on sensitivity p-values is acceptable given that sensitivities are interpreted relative to primary, not in absolute terms. Reduces sensitivities batch from ~600 h to ~120–200 h. | M5.5 |
| 2026-06-02 | **Cross-theme edges DEFERRED** to a future exploratory secondary module. | v1.0 hypothesis is within-theme coupling invariance. Cross-theme coupling is a distinct (and weaker) hypothesis that does not belong in the pre-registered primary. | §0 non-scope |
| 2026-06-02 | **Claim ceiling: per-edge level says "passed structural-coupling-invariance test in pool of 944" — NOT mechanism.** | Hard ceiling per spec §9 forbidden-language list. Brutalist review identified that any per-edge mechanism framing at this scale is unlicensable. | spec §9 + M7 disposition rules |
| 2026-06-02 | **M6a (pre-reg tag) MOVED BEFORE M2.** | No anchor-local null Q values may be computed on real data without a frozen manifest. Without this reorder, the pre-reg discipline is theatrical, not binding. | This plan §5 |
| 2026-06-02 | **M2.5 four-pronged calibration tripwire is a HARD HALT.** | Each prong addresses a specific failure mode (FP rate calibration, SPOR-size sensitivity, within-theme-pool over-restriction, FW vs OLS numerical identity). Any prong failure invalidates the null model; proceeding would produce uninterpretable results. | This plan §5, M2.5 |
| 2026-06-02 | **M5.5 sensitivities batch is MANDATORY regardless of primary outcome.** | Five sensitivities are pre-registered. Running them only on positive results would constitute a form of selection. Running all five always is the only defensible discipline. | This plan §5, M5.5 |
| 2026-06-02 | **M7: negative WASC is a publishable structural-coupling-flatness finding**, not a retraction of wave_24l. | Wave_24l's per-feature gradient claim is on a different statistical object (perturbation decay) than WASC (covariance structure). They are orthogonal; a null on one does not retract the other. | This plan §5 M7 disposition rules |

---

*End of build plan v1.0. Bound at git tag `wasc-prereg-v1.0` alongside the spec on completion of M6a.*