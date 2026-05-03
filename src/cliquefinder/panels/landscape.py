"""Landscape — gradient analysis with every measured feature as a seed.

Where a ``Panel`` is a small, structured set of seeds with a target
and biological strata, a ``Landscape`` is the exhaustive computation
across every feature in the measured set.  No target seed; no strata.
The natural reference distribution is the landscape itself: every
feature's slope is one point in a 3,000+-feature distribution.

Architecture
------------
The per-seed Cypher round-trip that the panel runner uses is wasteful
at this scale (3,256 seeds × ~3 min each = days of work).  Landscape
extracts the regulatory subgraph induced by *all* measured features
in one Cypher call (via
:func:`extract_subgraph_induced_by_features`), computes all-pairs
shortest paths *locally* (via
:func:`compute_all_pairs_shortest_paths_bounded`), then runs the
existing per-seed degree-binned label-permutation null
(``run_gradient_test``) against the precomputed shells.  Total cost
collapses from days (per-seed Neo4j) to ~30-90 minutes (one extraction +
local APSP + per-seed permutations) — perms remain CPU-bound and are
the dominant cost at scale.

UniProt-keyed throughout
------------------------
The landscape delegates |t|, alias collapse, and degree aggregation
to ``DiscoveryBridge`` so it inherits Wave 22's protein-level fix
(one observation per measured UniProt, not per HGNC alias).
``feature_names`` in ``FeatureDistanceMatrix`` are UniProt
accessions; HGNC symbols are an internal implementation detail of
the INDRA query layer.

The landscape does **not** parameterize edge scope — like the panel
runner, it inherits Wave 24d's commitment to ``ALL_REGULATORY_TYPES``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from cliquefinder.stats.differential import fdr_correction
from cliquefinder.utils.fileio import atomic_write_json, atomic_write_text

from .analysis import FailedSeed, PerSeedResult, ShellSummary
from .seed_runner import GroupResolver, load_panel_inputs

logger = logging.getLogger(__name__)


# Reserved stratum identifier for landscape per-feature results — like
# TARGET_STRATUM_LABEL ("<target>"), this is not a valid user stratum.
# Validated in panels/design.py:_RESERVED_STRATUM_NAMES below.
LANDSCAPE_FEATURE_STRATUM_LABEL = "<feature>"


@dataclass(frozen=True)
class LandscapeDesign:
    """Locked specification for a landscape (every-feature-as-seed) run.

    Attributes
    ----------
    contrast
        ``(case, control)`` condition labels.
    max_hops
        BFS depth for shell construction.  Inherits Wave 23: the
        regulatory subgraph saturates at h=2 in this dataset.
    n_permutations
        Degree-binned label-permutation null sample size.  Sets the
        floor of empirical p-values at ``1 / (n_permutations + 1)``.
    covariates
        Metadata columns for the design matrix.
    description
        Optional free-text note.

    Notes
    -----
    No ``target_seed``, no ``strata`` — every measured feature is a
    seed.  Edge scope is ``ALL_REGULATORY_TYPES``, inherited from the
    gradient pipeline.
    """

    contrast: tuple[str, str]
    max_hops: int
    n_permutations: int
    covariates: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.contrast, tuple):
            object.__setattr__(self, "contrast", tuple(self.contrast))
        if not isinstance(self.covariates, tuple):
            object.__setattr__(self, "covariates", tuple(self.covariates))

        if len(self.contrast) != 2 or self.contrast[0] == self.contrast[1]:
            raise ValueError(
                f"LandscapeDesign.contrast must be a 2-tuple of distinct "
                f"labels, got {self.contrast!r}"
            )
        if self.max_hops < 1:
            raise ValueError(
                f"LandscapeDesign.max_hops must be >= 1, got {self.max_hops}"
            )
        if self.n_permutations < 1:
            raise ValueError(
                f"LandscapeDesign.n_permutations must be >= 1, "
                f"got {self.n_permutations}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast": list(self.contrast),
            "max_hops": int(self.max_hops),
            "n_permutations": int(self.n_permutations),
            "covariates": list(self.covariates),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LandscapeDesign:
        contrast = data["contrast"]
        return cls(
            contrast=(str(contrast[0]), str(contrast[1])),
            max_hops=int(data["max_hops"]),
            n_permutations=int(data["n_permutations"]),
            covariates=tuple(str(c) for c in data.get("covariates", [])),
            description=str(data.get("description", "")),
        )

    def save_yaml(self, path: Path | str) -> None:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            self.to_dict(), sort_keys=False, default_flow_style=False,
        )
        atomic_write_text(path, text)

    @classmethod
    def load_yaml(cls, path: Path | str) -> LandscapeDesign:
        import yaml
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass(frozen=True)
class FeatureDistanceMatrix:
    """Sparse all-pairs shortest-path distances among measured features.

    Distances are integers in ``[0, max_hops]``; an absent entry means
    the target is not reachable from the source within ``max_hops``.

    The diagonal (source-to-self distance 0) is stored explicitly only
    if the source is in the measured set — a feature seed that wasn't
    matched in INDRA (unresolved) is marked as such via ``unmatched``.

    Attributes
    ----------
    feature_names
        Tuple of feature gene symbols in row/column order.  Indexable
        by name via ``index_of``.
    distances
        Sparse CSR matrix of shape ``(n_features, n_features)``,
        dtype ``int16``.  ``distances[i, j]`` is the shortest-path
        hop count from feature i to feature j; missing means not
        reachable within ``max_hops``.  Note CSR's "explicit zero"
        for the diagonal is meaningful (distance 0); use ``.toarray``
        with care.
    max_hops
        BFS depth used to compute the distances.
    unmatched
        Frozenset of feature names that did NOT resolve to a
        ``BioEntity`` in INDRA — distinguish "biologically isolated"
        from "couldn't be looked up."

    Notes
    -----
    Use ``hop_neighbors(seed, hop)`` to recover ``{features at this
    hop count from seed}`` — the set lookup is the load-bearing
    downstream operation for "what's in seed's hop-1 neighborhood."
    """

    feature_names: tuple[str, ...]
    distances: sp.csr_matrix
    max_hops: int
    unmatched: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.feature_names, tuple):
            object.__setattr__(
                self, "feature_names", tuple(self.feature_names),
            )
        if not isinstance(self.unmatched, frozenset):
            object.__setattr__(self, "unmatched", frozenset(self.unmatched))
        n = len(self.feature_names)
        if self.distances.shape != (n, n):
            raise ValueError(
                f"distances shape {self.distances.shape} does not match "
                f"len(feature_names)={n}"
            )
        if self.max_hops < 1:
            raise ValueError(
                f"max_hops must be >= 1, got {self.max_hops}"
            )

    def index_of(self, feature: str) -> int:
        try:
            return self.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(f"feature {feature!r} not in matrix") from exc

    def distances_from(self, seed: str) -> dict[str, int]:
        """Return ``{target: distance}`` for every reachable target from ``seed``.

        Distance 0 is the seed itself; other distances are positive
        integers up to ``max_hops``.  Unreachable targets are absent.
        """
        i = self.index_of(seed)
        row = self.distances.getrow(i)
        result: dict[str, int] = {}
        # CSR explicit-zero handling: a zero in .data is a real
        # stored distance (the diagonal).  Iterate the sparse format
        # directly.
        for j_idx in range(row.indptr[0], row.indptr[1]):
            j = int(row.indices[j_idx])
            d = int(row.data[j_idx])
            result[self.feature_names[j]] = d
        return result

    def hop_neighbors(self, seed: str, hop: int) -> set[str]:
        """Features at exactly ``hop`` distance from ``seed``."""
        return {
            f for f, d in self.distances_from(seed).items() if d == hop
        }

    def save_npz(self, path: Path | str) -> None:
        """Persist to a sidecar pair: ``<path>.npz`` (matrix) +
        ``<path>.meta.json`` (feature names, max_hops, unmatched).

        Atomic via temp-file + ``os.replace`` for the .npz; the meta
        sidecar uses :func:`atomic_write_json`.  An interrupted run
        will not leave a partially written .npz next to a complete
        .meta.json.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # scipy.sparse.save_npz isn't atomic; wrap with temp+rename.
        # NOTE: save_npz appends ".npz" to the path if missing, so we
        # use a temp directory location that ends in .npz to make the
        # final filename predictable.
        dir_path = str(path.parent) or "."
        tmp_handle = tempfile.NamedTemporaryFile(
            dir=dir_path, suffix=".npz.tmp", delete=False,
        )
        tmp_path = tmp_handle.name
        tmp_handle.close()
        # Remove the empty placeholder so save_npz can write fresh.
        # save_npz will treat tmp_path as having the extension already.
        os.unlink(tmp_path)
        try:
            sp.save_npz(tmp_path, self.distances)
            # save_npz writes to tmp_path verbatim (since it already
            # ends in a recognized extension after we removed the file).
            actual_tmp = (
                tmp_path if os.path.exists(tmp_path)
                else tmp_path + ".npz"
            )
            os.replace(actual_tmp, str(path))
        except BaseException:
            for candidate in (tmp_path, tmp_path + ".npz"):
                if candidate and os.path.exists(candidate):
                    os.unlink(candidate)
            raise
        meta = {
            "feature_names": list(self.feature_names),
            "max_hops": int(self.max_hops),
            "unmatched": sorted(self.unmatched),
        }
        atomic_write_json(path.with_suffix(".meta.json"), meta)

    @classmethod
    def load_npz(cls, path: Path | str) -> FeatureDistanceMatrix:
        path = Path(path)
        distances = sp.load_npz(path).tocsr()
        with open(path.with_suffix(".meta.json")) as f:
            meta = json.load(f)
        return cls(
            feature_names=tuple(meta["feature_names"]),
            distances=distances,
            max_hops=int(meta["max_hops"]),
            unmatched=frozenset(meta.get("unmatched", [])),
        )

    @classmethod
    def from_distance_dict(
        cls,
        distances: dict[str, dict[str, int]],
        feature_names: list[str],
        max_hops: int,
        unmatched: set[str] | None = None,
    ) -> FeatureDistanceMatrix:
        """Build from the nested-dict output of
        ``compute_all_pairs_shortest_paths_bounded``.

        ``feature_names`` defines the row/column order of the matrix.
        Distances to features NOT in ``feature_names`` (e.g.,
        unmeasured intermediaries) are dropped.
        """
        n = len(feature_names)
        index = {f: i for i, f in enumerate(feature_names)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for source, targets in distances.items():
            i = index.get(source)
            if i is None:
                continue
            for target, dist in targets.items():
                j = index.get(target)
                if j is None:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(dist)

        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.int16,
        ).tocsr()
        return cls(
            feature_names=tuple(feature_names),
            distances=matrix,
            max_hops=max_hops,
            unmatched=frozenset(unmatched or set()),
        )


@dataclass(frozen=True)
class LandscapeResult:
    """Rolled-up output of a landscape run.

    The result distinguishes three feature outcomes:

    - ``per_feature``: features that produced a real gradient
      (slope + p-value + shells).
    - ``degenerate_features``: features with no measured neighbors
      reachable within ``max_hops`` (biologically isolated under
      the regulatory edge scope, OR unmatched in INDRA).  Not a
      software failure — the gradient is mathematically degenerate.
    - ``error_features``: features whose run threw an unexpected
      exception (timeout, out-of-memory, etc.).  Diagnose via the
      attached ``error_type`` and ``error_message``.

    The multiple-testing family in :func:`analyze_landscape` is
    ``per_feature + degenerate_features + error_features`` so that
    no class of failure makes discoveries anti-conservative.
    """

    design: LandscapeDesign
    per_feature: tuple[PerSeedResult, ...]
    degenerate_features: tuple[FailedSeed, ...]
    error_features: tuple[FailedSeed, ...]
    distance_matrix_path: str  # relative to result.json's directory
    n_features_input: int

    def __post_init__(self) -> None:
        for fname in ("per_feature", "degenerate_features", "error_features"):
            value = getattr(self, fname)
            if not isinstance(value, tuple):
                object.__setattr__(self, fname, tuple(value))

        completed = [r.seed for r in self.per_feature]
        if len(set(completed)) != len(completed):
            raise ValueError(
                "LandscapeResult.per_feature has duplicate seeds"
            )
        degenerate = [f.seed for f in self.degenerate_features]
        if len(set(degenerate)) != len(degenerate):
            raise ValueError(
                "LandscapeResult.degenerate_features has duplicates"
            )
        errored = [f.seed for f in self.error_features]
        if len(set(errored)) != len(errored):
            raise ValueError(
                "LandscapeResult.error_features has duplicates"
            )
        all_seeds = set(completed) | set(degenerate) | set(errored)
        observed_n = len(completed) + len(degenerate) + len(errored)
        if observed_n != len(all_seeds):
            overlap = sorted(
                s for s in all_seeds
                if (s in completed) + (s in degenerate) + (s in errored) > 1
            )
            raise ValueError(
                f"seeds appear in multiple outcome buckets: {overlap}"
            )
        if observed_n != self.n_features_input:
            raise ValueError(
                f"completed ({len(completed)}) + degenerate "
                f"({len(degenerate)}) + errored ({len(errored)}) = "
                f"{observed_n} != n_features_input ({self.n_features_input})"
            )

    @property
    def failed_features(self) -> tuple[FailedSeed, ...]:
        """All non-completed features (degenerate + errored), for
        backwards compatibility with the multiple-testing family math.
        """
        return self.degenerate_features + self.error_features

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "per_feature": [r.to_dict() for r in self.per_feature],
            "degenerate_features": [
                f.to_dict() for f in self.degenerate_features
            ],
            "error_features": [
                f.to_dict() for f in self.error_features
            ],
            "distance_matrix_path": self.distance_matrix_path,
            "n_features_input": int(self.n_features_input),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LandscapeResult:
        return cls(
            design=LandscapeDesign.from_dict(data["design"]),
            per_feature=tuple(
                PerSeedResult.from_dict(r) for r in data["per_feature"]
            ),
            degenerate_features=tuple(
                FailedSeed.from_dict(f) for f in data.get("degenerate_features", [])
            ),
            error_features=tuple(
                FailedSeed.from_dict(f) for f in data.get("error_features", [])
            ),
            distance_matrix_path=str(data["distance_matrix_path"]),
            n_features_input=int(data["n_features_input"]),
        )

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: Path | str) -> LandscapeResult:
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class AdjustedFeatureResult:
    """Per-feature result with multiple-testing-adjusted p-values + rank."""

    seed: str
    slope: float
    slope_pvalue: float
    bh_qvalue: float
    bonferroni_pvalue: float
    rank_left_tail: int  # 1-indexed rank in landscape, ascending
    discovery: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "slope": float(self.slope),
            "slope_pvalue": float(self.slope_pvalue),
            "bh_qvalue": float(self.bh_qvalue),
            "bonferroni_pvalue": float(self.bonferroni_pvalue),
            "rank_left_tail": int(self.rank_left_tail),
            "discovery": bool(self.discovery),
        }


@dataclass(frozen=True)
class LandscapeAnalysis:
    """Statistical summary of a landscape.

    Attributes
    ----------
    design
        The originating manifest.
    q_threshold
        BH-FDR significance threshold for ``discovery`` flag.
    feature_results_adjusted
        Per-feature with BH q, Bonferroni p, and landscape rank.
        Ordered by ``slope`` ascending (most negative first).
    n_completed, n_failed
        Counts.  Multiple-testing family = completed + failed (failed
        features expand the family so non-random failures don't bias
        toward discoveries).
    """

    design: LandscapeDesign
    q_threshold: float
    feature_results_adjusted: tuple[AdjustedFeatureResult, ...]
    n_completed: int
    n_failed: int

    def __post_init__(self) -> None:
        if not isinstance(self.feature_results_adjusted, tuple):
            object.__setattr__(
                self, "feature_results_adjusted",
                tuple(self.feature_results_adjusted),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "q_threshold": float(self.q_threshold),
            "feature_results_adjusted": [
                r.to_dict() for r in self.feature_results_adjusted
            ],
            "n_completed": int(self.n_completed),
            "n_failed": int(self.n_failed),
        }


def analyze_landscape(
    result: LandscapeResult,
    *,
    q_threshold: float = 0.05,
) -> LandscapeAnalysis:
    """Apply BH-FDR + Bonferroni + per-feature rank to a landscape.

    The multiple-testing family is the *attempted* feature set
    (``per_feature + degenerate_features + error_features``).
    Degenerate features (no measured neighbors) and errored features
    are both real attempts that produced no discovery — including
    them in the denominator prevents non-random failures from making
    discoveries anti-conservative.

    Per-feature rank uses ``(slope, seed)`` as the sort key, so the
    ``rank_left_tail`` of an ``AdjustedFeatureResult`` matches its
    position in ``feature_results_adjusted`` exactly (no
    rank/output-order mismatch on ties).
    """
    completed = list(result.per_feature)
    n_completed = len(completed)
    n_attempted = result.n_features_input

    if n_completed == 0:
        return LandscapeAnalysis(
            design=result.design,
            q_threshold=q_threshold,
            feature_results_adjusted=(),
            n_completed=0,
            n_failed=n_attempted,
        )

    p_arr = np.array([r.slope_pvalue for r in completed], dtype=np.float64)

    # BH on the attempted family: pad with 1.0 for non-completed
    # features so the denominator reflects the actual attempt count.
    if n_attempted > n_completed:
        padded = np.concatenate([
            p_arr,
            np.ones(n_attempted - n_completed, dtype=np.float64),
        ])
        bh_q = fdr_correction(padded, method="BH")[:n_completed]
    else:
        bh_q = fdr_correction(p_arr, method="BH")
    bonf_p = np.minimum(p_arr * n_attempted, 1.0)

    # Build a list of completed-with-adjustments, then sort by
    # (slope, seed) ascending and assign rank by sorted position so
    # rank == position+1 always holds on output.
    enriched = list(zip(completed, bh_q, bonf_p))
    enriched.sort(key=lambda x: (x[0].slope, x[0].seed))

    adjusted = tuple(
        AdjustedFeatureResult(
            seed=r.seed,
            slope=r.slope,
            slope_pvalue=r.slope_pvalue,
            bh_qvalue=float(q),
            bonferroni_pvalue=float(b),
            rank_left_tail=position + 1,  # 1-indexed; matches output order
            discovery=(
                bool(q < q_threshold) if not np.isnan(q) else False
            ),
        )
        for position, (r, q, b) in enumerate(enriched)
    )

    return LandscapeAnalysis(
        design=result.design,
        q_threshold=q_threshold,
        feature_results_adjusted=adjusted,
        n_completed=n_completed,
        n_failed=n_attempted - n_completed,
    )


def _fit_engine_for_contrast(
    design: LandscapeDesign,
    data: np.ndarray,
    feature_ids: list[str],
    metadata: "pd.DataFrame",
    groups: dict[str, "pd.Index"],
):
    """Subset + fit RotationTestEngine for one contrast.  Helper for testability."""
    from cliquefinder.stats.rotation import RotationTestEngine

    cond1, cond2 = design.contrast
    keep_samples = groups[cond1].union(groups[cond2])
    sub_meta = metadata.loc[metadata.index.intersection(keep_samples)].copy()
    sub_meta["_condition"] = None
    sub_meta.loc[sub_meta.index.isin(groups[cond1]), "_condition"] = cond1
    sub_meta.loc[sub_meta.index.isin(groups[cond2]), "_condition"] = cond2
    sub_meta = sub_meta.dropna(subset=["_condition"])
    sample_id_to_idx = {s: i for i, s in enumerate(metadata.index)}
    aligned_indices = [sample_id_to_idx[s] for s in sub_meta.index]
    sub_data = data[:, aligned_indices]

    engine = RotationTestEngine(sub_data, list(feature_ids), sub_meta)
    fit_covariates = [
        c for c in design.covariates if c in sub_meta.columns
    ]
    engine.fit(
        conditions=[cond1, cond2],
        contrast=(cond1, cond2),
        condition_column="_condition",
        covariates=fit_covariates,
    )
    return engine


def _build_distance_matrix(
    cogex_client,
    measured_symbols: list[str],
    measured_feature_ids: list[str],
    sym_to_feat: dict[str, str],
    max_hops: int,
    seed_batch_size: int = 500,
) -> tuple[FeatureDistanceMatrix, set[str], dict[str, int], list[tuple[str, str, dict]]]:
    """Extract regulatory subgraph and build per-protein distance matrix.

    Aggregates HGNC-keyed all-pairs distances back to UniProt-keyed
    per-protein distances using the bridge's min-over-aliases convention
    (Wave 22).  Returns:

    - ``FeatureDistanceMatrix`` keyed by UniProt feature_ids.
    - Set of unmatched feature_ids (UniProt accessions whose aliases
      did not resolve to a BioEntity in INDRA).
    - Per-feature_id graph degree (max-over-aliases against full INDRA),
      for the degree-binned null.
    - Raw edge list (returned for caller's optional reuse).
    """
    from cliquefinder.stats.network_proximity import (
        compute_all_pairs_shortest_paths_bounded,
        extract_subgraph_induced_by_features,
        query_gene_degrees_batched,
    )

    logger.info(
        "Extracting regulatory subgraph induced by %d HGNC aliases "
        "(%d UniProt proteins)",
        len(measured_symbols), len(measured_feature_ids),
    )
    t_extract = time.time()
    edges, matched_symbols = extract_subgraph_induced_by_features(
        cogex_client=cogex_client,
        features=measured_symbols,
        max_hops=max_hops,
        min_evidence=1,
        seed_batch_size=seed_batch_size,
    )
    logger.info(
        "Extracted %d regulatory edges; %d/%d HGNC aliases matched "
        "in INDRA (%.1fs)",
        len(edges), len(matched_symbols), len(measured_symbols),
        time.time() - t_extract,
    )

    logger.info("Computing all-pairs shortest paths (bounded at h=%d)",
                max_hops)
    t_apsp = time.time()
    distances_hgnc_dict = compute_all_pairs_shortest_paths_bounded(
        edges=edges,
        source_nodes=measured_symbols,
        max_hops=max_hops,
        target_filter=set(measured_symbols),
    )
    logger.info("All-pairs done (%.1fs)", time.time() - t_apsp)

    # Full-INDRA degrees per HGNC alias — needed for degree-bin matching
    # consistent with the bridge's per-seed gradient.
    logger.info(
        "Querying full-INDRA regulatory degrees for %d HGNC aliases",
        len(measured_symbols),
    )
    t_deg = time.time()
    degrees_hgnc = query_gene_degrees_batched(
        cogex_client=cogex_client,
        gene_names=measured_symbols,
        batch_size=500,
    )
    logger.info("Degrees done (%.1fs)", time.time() - t_deg)

    # Aggregate to UniProt level (Wave 22 convention):
    # - distance: min over aliases that have a finite distance.
    # - degree: max over aliases (hub-status ceiling).
    # - unmatched at protein level: NO alias of this UniProt resolved.
    feat_to_syms: dict[str, list[str]] = {}
    for sym, fid in sym_to_feat.items():
        if fid in measured_feature_ids:
            feat_to_syms.setdefault(fid, []).append(sym)

    distances_prot: dict[str, dict[str, int]] = {}
    degrees_prot: dict[str, int] = {}
    unmatched_proteins: set[str] = set()
    for fid, aliases in feat_to_syms.items():
        # Degree: max over aliases.
        degrees_prot[fid] = max(
            (degrees_hgnc.get(a, 0) for a in aliases), default=0,
        )
        # Unmatched at protein level if NO alias matched in INDRA.
        if matched_symbols.isdisjoint(aliases):
            unmatched_proteins.add(fid)

    # Build per-source UniProt distance dict by aggregating alias rows.
    # For each source UniProt, take the min distance to each target
    # UniProt across all (source_alias, target_alias) pairs.
    feature_id_set = set(measured_feature_ids)
    for source_fid, source_aliases in feat_to_syms.items():
        per_target: dict[str, int] = {source_fid: 0}
        for source_alias in source_aliases:
            alias_dists = distances_hgnc_dict.get(source_alias, {})
            for target_alias, dist in alias_dists.items():
                target_fid = sym_to_feat.get(target_alias)
                if target_fid is None or target_fid not in feature_id_set:
                    continue
                if target_fid == source_fid:
                    continue  # self-distance handled above
                if 1 <= dist <= max_hops:
                    existing = per_target.get(target_fid)
                    if existing is None or dist < existing:
                        per_target[target_fid] = dist
        distances_prot[source_fid] = per_target

    matrix = FeatureDistanceMatrix.from_distance_dict(
        distances=distances_prot,
        feature_names=measured_feature_ids,
        max_hops=max_hops,
        unmatched=unmatched_proteins,
    )
    return matrix, unmatched_proteins, degrees_prot, edges


def _per_feature_gradient_loop(
    *,
    measured_feature_ids: list[str],
    abs_t_per_feature: dict[str, float],
    distance_matrix: FeatureDistanceMatrix,
    graph_degrees: dict[str, int],
    unmatched: set[str],
    max_hops: int,
    n_permutations: int,
    rng_base: int,
    progress_every: int = 200,
) -> tuple[list[PerSeedResult], list[FailedSeed], list[FailedSeed]]:
    """Per-feature gradient over the precomputed distance matrix.

    Returns ``(completed, degenerate, errored)`` tuples.  Degenerate
    features had no measured neighbors in their max_hops shells (either
    biologically isolated or unresolved in INDRA); errored features
    threw an unexpected exception.

    Each feature's null permutation gets a deterministic per-feature
    RNG seed derived from ``rng_base`` and the feature id, so 3,256
    Monte Carlo tests do not all share a single shuffle sequence.
    """
    from cliquefinder.stats.perturbation_gradient import run_gradient_test

    completed: list[PerSeedResult] = []
    degenerate: list[FailedSeed] = []
    errored: list[FailedSeed] = []
    n = len(measured_feature_ids)
    t0 = time.time()

    for i, seed_fid in enumerate(measured_feature_ids):
        try:
            seed_distances = distance_matrix.distances_from(seed_fid)
            shells: dict[int, set[str]] = {}
            for target, d in seed_distances.items():
                if 1 <= d <= max_hops and target != seed_fid:
                    shells.setdefault(d, set()).add(target)
            if not shells:
                degenerate.append(FailedSeed(
                    seed=seed_fid,
                    error_type="DisconnectedFeature",
                    error_message=(
                        "feature did not resolve to a BioEntity in INDRA"
                        if seed_fid in unmatched
                        else "no measured features reachable within max_hops"
                    ),
                ))
                continue
            # Drop seed from |t| pool (consistent with bridge — seed's
            # own signal must not contaminate its own null/background).
            abs_t_no_seed = {
                fid: t for fid, t in abs_t_per_feature.items()
                if fid != seed_fid
            }
            # Stable per-feature RNG seed: Python's built-in hash() is
            # randomized per process (PYTHONHASHSEED), so using it
            # would make landscape runs non-reproducible across
            # subprocess boundaries.  md5 is process-stable.
            seed_rng = rng_base + int.from_bytes(
                hashlib.md5(seed_fid.encode()).digest()[:4], "big",
            )
            t_seed = time.time()
            result = run_gradient_test(
                adjacency={},
                abs_t_stats=abs_t_no_seed,
                seed=seed_fid,
                max_hops=max_hops,
                n_permutations=n_permutations,
                rng_seed=seed_rng,
                precomputed_shells=shells,
                graph_degrees=graph_degrees,
                verbose=False,
            )
            completed.append(PerSeedResult(
                seed=seed_fid,
                stratum=LANDSCAPE_FEATURE_STRATUM_LABEL,
                slope=float(result.slope),
                slope_pvalue=float(result.slope_pvalue),
                spearman_rho=float(result.spearman_rho),
                spearman_pvalue=float(result.spearman_pvalue),
                shells=tuple(
                    ShellSummary(
                        hop=int(s.hop),
                        n_genes=int(s.n_genes),
                        mean_abs_t=float(s.mean_abs_t),
                        median_abs_t=float(s.median_abs_t),
                    )
                    for s in result.shells
                ),
                n_genes_total=int(result.n_genes_total),
                elapsed_seconds=round(time.time() - t_seed, 3),
            ))
        except (MemoryError, RecursionError):
            # Resource-exhaustion errors are systemic; let them
            # propagate so the caller halts rather than mark every
            # remaining feature as a per-feature failure.
            raise
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc) or repr(exc)
            errored.append(FailedSeed(
                seed=seed_fid,
                error_type=type(exc).__name__,
                error_message=error_msg[:500],
            ))
            logger.debug(
                "Seed %s errored: %s: %s",
                seed_fid, type(exc).__name__, error_msg,
            )
        if (i + 1) % progress_every == 0:
            logger.info(
                "  Progress: %d/%d features (%d ok, %d degenerate, "
                "%d errored, %.1fs)",
                i + 1, n,
                len(completed), len(degenerate), len(errored),
                time.time() - t0,
            )
    return completed, degenerate, errored


def compute_landscape(
    design: LandscapeDesign,
    *,
    data_path: Path | str,
    metadata_path: Path | str,
    group_resolver: GroupResolver,
    indra_env_file: Path | str | None,
    output_dir: Path | str,
    rng_seed: int = 42,
    seed_batch_size: int = 500,
) -> LandscapeResult:
    """Run the full landscape: fit engine, extract subgraph, per-feature gradient.

    UniProt-keyed throughout (Wave 22 convention): every measured
    protein contributes one observation, regardless of how many HGNC
    aliases it has.  Delegates |t|, alias collapse, and degree
    aggregation to ``DiscoveryBridge`` so the landscape's per-feature
    inputs are exactly what a per-seed bridge gradient would compute.

    Saves:
      - ``output_dir/manifest.yaml`` — frozen LandscapeDesign
      - ``output_dir/distances.npz`` + ``.meta.json`` — distance matrix
      - ``output_dir/result.json`` — LandscapeResult

    All writes are atomic.
    """
    from cliquefinder.stats.clique_analysis import map_feature_ids_to_symbols
    from cliquefinder.stats.discovery_bridge import DiscoveryBridge

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.yaml"
    design.save_yaml(manifest_path)
    logger.info("Wrote landscape manifest to %s", manifest_path)

    # Phase 1: load data + fit engine.
    logger.info("Loading data and fitting ROAST engine")
    data, feature_ids, metadata, groups = load_panel_inputs(
        data_path=data_path,
        metadata_path=metadata_path,
        group_resolver=group_resolver,
    )
    engine = _fit_engine_for_contrast(
        design, data, feature_ids, metadata, groups,
    )

    sym_to_feat = map_feature_ids_to_symbols(
        list(feature_ids), verbose=False,
    )

    # Phase 2: bridge gives us protein-level |t| (Wave 22).
    with DiscoveryBridge(
        engine, sym_to_feat,
        env_file=indra_env_file,
        min_evidence=1,
        min_reliability=0.0,
        min_sources=1,
    ) as bridge:
        abs_t_per_feature = bridge.get_abs_t_per_feature()
        cogex_client = bridge._indra_source.client if bridge._indra_source else None
        if cogex_client is None:
            bridge._ensure_indra()
            cogex_client = bridge._indra_source.client

        measured_feature_ids = sorted(abs_t_per_feature.keys())
        n_features_input = len(measured_feature_ids)
        # HGNC aliases for every measured UniProt — what we actually
        # query INDRA for.  Sorted for deterministic Cypher batching.
        feat_to_syms: dict[str, list[str]] = {}
        for sym, fid in sym_to_feat.items():
            if fid in abs_t_per_feature:
                feat_to_syms.setdefault(fid, []).append(sym)
        measured_symbols = sorted({
            sym for syms in feat_to_syms.values() for sym in syms
        })
        logger.info(
            "Measured features: %d UniProt proteins (%d HGNC aliases)",
            n_features_input, len(measured_symbols),
        )

        # Phase 3: distance matrix at protein level (Wave 22 aggregation).
        distance_matrix, unmatched, graph_degrees, _edges = _build_distance_matrix(
            cogex_client=cogex_client,
            measured_symbols=measured_symbols,
            measured_feature_ids=measured_feature_ids,
            sym_to_feat=sym_to_feat,
            max_hops=design.max_hops,
            seed_batch_size=seed_batch_size,
        )

    matrix_path = output_dir / "distances.npz"
    distance_matrix.save_npz(matrix_path)
    logger.info(
        "Wrote distance matrix to %s (%d × %d, %d unmatched)",
        matrix_path, n_features_input, n_features_input, len(unmatched),
    )

    # Phase 4: per-feature gradient.  No Neo4j; pure local compute.
    logger.info(
        "Running per-feature gradient over %d UniProt seeds",
        n_features_input,
    )
    completed, degenerate, errored = _per_feature_gradient_loop(
        measured_feature_ids=measured_feature_ids,
        abs_t_per_feature=abs_t_per_feature,
        distance_matrix=distance_matrix,
        graph_degrees=graph_degrees,
        unmatched=unmatched,
        max_hops=design.max_hops,
        n_permutations=design.n_permutations,
        rng_base=rng_seed,
    )
    logger.info(
        "Per-feature gradient done: %d completed, %d degenerate, %d errored",
        len(completed), len(degenerate), len(errored),
    )

    landscape_result = LandscapeResult(
        design=design,
        per_feature=tuple(completed),
        degenerate_features=tuple(degenerate),
        error_features=tuple(errored),
        distance_matrix_path=matrix_path.name,
        n_features_input=n_features_input,
    )
    result_path = output_dir / "result.json"
    landscape_result.save_json(result_path)
    logger.info("Wrote landscape result to %s", result_path)
    return landscape_result
