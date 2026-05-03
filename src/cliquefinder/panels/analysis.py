"""Panel analysis — per-seed results, multiple-testing correction,
stratum comparisons, and the target seed's empirical position.

The analysis stack answers three questions, in order of statistical
care:

1. **Per-seed: did the seed produce a real gradient?**
   The intrinsic null is the degree-binned label permutation
   (``slope_pvalue`` from the gradient pipeline).  Multiple-testing
   correction across the panel is applied via Benjamini–Hochberg
   (primary, ``q < 0.05``) with Bonferroni reported as a sensitivity
   check.  Both delegate to
   :func:`cliquefinder.stats.differential.fdr_correction`, which is
   backed by statsmodels and handles NaN p-values.  The target seed
   is corrected separately — it is the inferential subject, not an
   exchangeable panel member.

2. **Across strata: does one biological category gradient harder?**
   For each unordered pair of strata we emit one one-sided
   Mann–Whitney U test in the direction "stratum_a slopes are *less
   than* (more negative than) stratum_b slopes," with the convention
   ``stratum_a < stratum_b`` lexicographically.  Reciprocal direction
   is recoverable from the same row's ``u_statistic`` and ``n_*``;
   exposing both as separate "hypotheses" would inflate the
   apparent test count.  No multiple-testing correction across pairs
   is applied — that policy is a downstream choice.

3. **Target rank: where does the target sit in the panel?**
   Panel slopes form an empirical reference distribution; the
   target's empirical p-value is its rank among them (left-tail for
   negative slopes).  This is the model-free question — no
   distributional assumptions, no covariate model.

All p-values use the Phipson–Smyth ``(B + 1) / (N + 1)`` convention
(no zero floor), consistent with the rest of the framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from cliquefinder.stats.differential import fdr_correction
from cliquefinder.utils.fileio import atomic_write_json

from .design import PanelDesign


@dataclass(frozen=True)
class ShellSummary:
    """Per-shell aggregates from a single seed's gradient."""

    hop: int
    n_genes: int
    mean_abs_t: float
    median_abs_t: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": int(self.hop),
            "n_genes": int(self.n_genes),
            "mean_abs_t": float(self.mean_abs_t),
            "median_abs_t": float(self.median_abs_t),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShellSummary:
        return cls(
            hop=int(data["hop"]),
            n_genes=int(data["n_genes"]),
            mean_abs_t=float(data["mean_abs_t"]),
            median_abs_t=float(data["median_abs_t"]),
        )


@dataclass(frozen=True)
class PerSeedResult:
    """Single-seed gradient outcome.

    Attributes
    ----------
    seed
        Gene symbol.
    stratum
        Stratum name (or ``"<target>"`` for the target seed).
    slope
        WLS slope of mean ``|t|`` against hop distance.  Negative
        slopes mean perturbation decays with distance.
    slope_pvalue
        Phipson–Smyth empirical p-value from the degree-binned
        label-permutation null.
    spearman_rho, spearman_pvalue
        Rank correlation alternative diagnostic.
    shells
        Per-hop summaries.
    n_genes_total
        Total measured proteins reachable from the seed.
    elapsed_seconds
        Wall-clock cost of the run, for cost accounting across
        panels.
    """

    seed: str
    stratum: str
    slope: float
    slope_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    shells: tuple[ShellSummary, ...]
    n_genes_total: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "stratum": self.stratum,
            "slope": float(self.slope),
            "slope_pvalue": float(self.slope_pvalue),
            "spearman_rho": float(self.spearman_rho),
            "spearman_pvalue": float(self.spearman_pvalue),
            "shells": [s.to_dict() for s in self.shells],
            "n_genes_total": int(self.n_genes_total),
            "elapsed_seconds": float(self.elapsed_seconds),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerSeedResult:
        return cls(
            seed=str(data["seed"]),
            stratum=str(data["stratum"]),
            slope=float(data["slope"]),
            slope_pvalue=float(data["slope_pvalue"]),
            spearman_rho=float(data["spearman_rho"]),
            spearman_pvalue=float(data["spearman_pvalue"]),
            shells=tuple(
                ShellSummary.from_dict(s) for s in data["shells"]
            ),
            n_genes_total=int(data["n_genes_total"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
        )


# Stratum identifier used for the implicit target stratum.  Reserved
# in PanelStratum.__post_init__ so a user-defined stratum cannot
# collide with it.
TARGET_STRATUM_LABEL = "<target>"


@dataclass(frozen=True)
class FailedSeed:
    """Structured failure record for a seed that crashed during a run.

    Carries enough information to diagnose the failure from
    ``result.json`` alone — the on-disk artifact is the authoritative
    record, not the stderr log.
    """

    seed: str
    error_type: str  # exception class name (e.g., "RuntimeError")
    error_message: str  # str(exc), truncated to a sane length

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailedSeed:
        return cls(
            seed=str(data["seed"]),
            error_type=str(data.get("error_type", "Unknown")),
            error_message=str(data.get("error_message", "")),
        )


@dataclass(frozen=True)
class PanelResult:
    """Rolled-up output of a panel run.

    The target seed is reported separately from the panel members.
    Multiple-testing and stratum comparisons treat the target
    asymmetrically because it is the inferential subject, not an
    exchangeable panel member.

    Validation
    ----------
    ``__post_init__`` enforces that the result is internally consistent
    with its bundled design:
    - Target seed and stratum match the design.
    - Per-seed seeds, plus failed seeds, partition the design's seed
      set exactly (no duplicates, no extras, no missing).
    - Each per-seed row's stratum matches the design's stratum
      assignment for that seed.
    """

    design: PanelDesign
    target_result: PerSeedResult
    per_seed: tuple[PerSeedResult, ...]
    failed_seeds: tuple[FailedSeed, ...] = ()

    def __post_init__(self) -> None:
        # Coerce inputs to tuples so frozen-ness extends to collections.
        if not isinstance(self.per_seed, tuple):
            object.__setattr__(self, "per_seed", tuple(self.per_seed))
        if not isinstance(self.failed_seeds, tuple):
            object.__setattr__(
                self, "failed_seeds", tuple(self.failed_seeds),
            )

        if self.target_result.seed != self.design.target_seed:
            raise ValueError(
                f"PanelResult.target_result.seed "
                f"({self.target_result.seed!r}) does not match "
                f"design.target_seed ({self.design.target_seed!r})"
            )
        if self.target_result.stratum != TARGET_STRATUM_LABEL:
            raise ValueError(
                f"target_result.stratum must be {TARGET_STRATUM_LABEL!r}, "
                f"got {self.target_result.stratum!r}"
            )

        per_seed_names = [r.seed for r in self.per_seed]
        if len(set(per_seed_names)) != len(per_seed_names):
            duplicates = sorted(
                {n for n in per_seed_names if per_seed_names.count(n) > 1}
            )
            raise ValueError(
                f"PanelResult.per_seed has duplicate seeds: {duplicates}"
            )

        completed = set(per_seed_names)
        failed_names = [f.seed for f in self.failed_seeds]
        if len(set(failed_names)) != len(failed_names):
            raise ValueError(
                f"PanelResult.failed_seeds has duplicate entries"
            )
        failed = set(failed_names)
        if completed & failed:
            raise ValueError(
                f"PanelResult: seeds appear in both per_seed and "
                f"failed_seeds: {sorted(completed & failed)}"
            )

        expected = set(self.design.selected_seeds())
        observed = completed | failed
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"PanelResult does not match design: "
                f"missing seeds {missing}, unexpected seeds {extra}"
            )

        if self.design.target_seed in observed:
            raise ValueError(
                f"PanelResult: target seed {self.design.target_seed!r} "
                f"must not appear in per_seed or failed_seeds"
            )

        for r in self.per_seed:
            expected_stratum = self.design.stratum_for(r.seed)
            if r.stratum != expected_stratum:
                raise ValueError(
                    f"PanelResult: seed {r.seed!r} reports stratum "
                    f"{r.stratum!r} but design assigns it to "
                    f"{expected_stratum!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "target_result": self.target_result.to_dict(),
            "per_seed": [r.to_dict() for r in self.per_seed],
            "failed_seeds": [f.to_dict() for f in self.failed_seeds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PanelResult:
        # Backwards-compatible failed_seeds parsing: accept either a
        # list of dicts (new schema) or a list of strings (legacy
        # placeholder during 24e iteration).  String entries become
        # FailedSeed with placeholder error info.
        raw_failed = data.get("failed_seeds", [])
        failed: list[FailedSeed] = []
        for entry in raw_failed:
            if isinstance(entry, str):
                failed.append(FailedSeed(
                    seed=entry,
                    error_type="Unknown",
                    error_message="(no detail in legacy schema)",
                ))
            else:
                failed.append(FailedSeed.from_dict(entry))
        return cls(
            design=PanelDesign.from_dict(data["design"]),
            target_result=PerSeedResult.from_dict(data["target_result"]),
            per_seed=tuple(
                PerSeedResult.from_dict(r) for r in data["per_seed"]
            ),
            failed_seeds=tuple(failed),
        )

    def save_json(self, path: Path | str) -> None:
        """Atomic JSON write via :func:`atomic_write_json`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load_json(cls, path: Path | str) -> PanelResult:
        import json
        path = Path(path)
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class AdjustedSeedResult:
    """A ``PerSeedResult`` augmented with multiple-testing-adjusted
    p-values and a discovery-status flag.
    """

    seed: str
    stratum: str
    slope: float
    slope_pvalue: float
    bh_qvalue: float
    bonferroni_pvalue: float
    discovery: bool  # True iff bh_qvalue < q_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "stratum": self.stratum,
            "slope": float(self.slope),
            "slope_pvalue": float(self.slope_pvalue),
            "bh_qvalue": float(self.bh_qvalue),
            "bonferroni_pvalue": float(self.bonferroni_pvalue),
            "discovery": bool(self.discovery),
        }


@dataclass(frozen=True)
class StratumComparison:
    """One pairwise Mann–Whitney U comparison between two strata.

    Convention: ``stratum_a < stratum_b`` lexicographically, with the
    one-sided alternative "stratum_a slopes are *less than* (more
    negative than) stratum_b slopes."  The reciprocal direction is
    not emitted as a separate row — it is recoverable from
    ``u_statistic`` and ``n_a * n_b`` via
    ``U_reverse = n_a * n_b - U`` and the SciPy continuity-corrected
    p-value of the reversed alternative.  Emitting both directions
    would inflate the apparent number of hypotheses and invite
    multiple-testing confusion.
    """

    stratum_a: str
    stratum_b: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    u_statistic: float
    pvalue: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratum_a": self.stratum_a,
            "stratum_b": self.stratum_b,
            "n_a": int(self.n_a),
            "n_b": int(self.n_b),
            "median_a": float(self.median_a),
            "median_b": float(self.median_b),
            "u_statistic": float(self.u_statistic),
            "pvalue": float(self.pvalue),
        }


@dataclass(frozen=True)
class TargetPosition:
    """Empirical position of the target seed within the panel.

    Attributes
    ----------
    target_slope
        Target seed's slope.
    panel_n
        Number of panel members compared against (excludes the target).
    rank_left_tail
        1-indexed rank from the most negative end (1 = strictly more
        negative than every panel member).
    empirical_p_left
        Phipson–Smyth p-value for the test "panel slope ≤ target slope":
        ``(rank_left_tail) / (panel_n + 1)``.  Conservative: ties are
        counted as ``≤``.
    """

    target_slope: float
    panel_n: int
    rank_left_tail: int
    empirical_p_left: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_slope": float(self.target_slope),
            "panel_n": int(self.panel_n),
            "rank_left_tail": int(self.rank_left_tail),
            "empirical_p_left": float(self.empirical_p_left),
        }


@dataclass(frozen=True)
class PanelAnalysis:
    """Statistical summary of a panel run.

    Notes
    -----
    The target seed and the panel members share the same
    ``q_threshold`` for their respective ``discovery`` flags.  The
    target's BH q-value collapses to its raw p (single-test BH); its
    Bonferroni p collapses similarly.  Using one threshold is more
    coherent than two — a future user changing it gets consistent
    semantics across both kinds of seed.
    """

    design: PanelDesign
    q_threshold: float
    panel_seeds_adjusted: tuple[AdjustedSeedResult, ...]
    target_adjusted: AdjustedSeedResult
    stratum_comparisons: tuple[StratumComparison, ...]
    target_position: TargetPosition

    def __post_init__(self) -> None:
        if not isinstance(self.panel_seeds_adjusted, tuple):
            object.__setattr__(
                self, "panel_seeds_adjusted",
                tuple(self.panel_seeds_adjusted),
            )
        if not isinstance(self.stratum_comparisons, tuple):
            object.__setattr__(
                self, "stratum_comparisons",
                tuple(self.stratum_comparisons),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "q_threshold": float(self.q_threshold),
            "panel_seeds_adjusted": [
                r.to_dict() for r in self.panel_seeds_adjusted
            ],
            "target_adjusted": self.target_adjusted.to_dict(),
            "stratum_comparisons": [
                c.to_dict() for c in self.stratum_comparisons
            ],
            "target_position": self.target_position.to_dict(),
        }


def analyze_panel(
    result: PanelResult,
    *,
    q_threshold: float = 0.05,
) -> PanelAnalysis:
    """Compute multiple-testing corrections and stratum comparisons.

    Parameters
    ----------
    result
        Panel run output.
    q_threshold
        Significance threshold for both BH-q (panel members) and the
        target seed's single-test discovery flag (default 0.05).

    Notes
    -----
    The multiple-testing family for both BH-FDR and Bonferroni is the
    set of *attempted* panel seeds (completed + failed), not just the
    ones that produced a result.  This prevents non-random failures
    from making discoveries anti-conservative.  Failed seeds get no
    ``AdjustedSeedResult``; only completed ones are returned.
    """
    panel = list(result.per_seed)
    n_completed = len(panel)
    n_attempted = n_completed + len(result.failed_seeds)

    if n_completed > 0:
        panel_pvalues = np.array(
            [r.slope_pvalue for r in panel], dtype=np.float64
        )
        # BH on the attempted family: pad with 1.0 for failed seeds so
        # the denominator reflects the panel's actual size.  The
        # padded values get adjusted q-values that we discard.
        if n_attempted > n_completed:
            padded = np.concatenate([
                panel_pvalues,
                np.ones(n_attempted - n_completed, dtype=np.float64),
            ])
            bh_q = fdr_correction(padded, method="BH")[:n_completed]
            bonf_p = np.minimum(panel_pvalues * n_attempted, 1.0)
        else:
            bh_q = fdr_correction(panel_pvalues, method="BH")
            bonf_p = np.minimum(panel_pvalues * n_completed, 1.0)
    else:
        bh_q = np.empty(0, dtype=np.float64)
        bonf_p = np.empty(0, dtype=np.float64)

    panel_adjusted = tuple(
        AdjustedSeedResult(
            seed=r.seed,
            stratum=r.stratum,
            slope=r.slope,
            slope_pvalue=r.slope_pvalue,
            bh_qvalue=float(bh_q[i]),
            bonferroni_pvalue=float(bonf_p[i]),
            # NaN-safe: NaN < x is False, so a NaN p never becomes a
            # discovery (silently — this is consistent with statsmodels).
            discovery=bool(bh_q[i] < q_threshold) if not np.isnan(bh_q[i]) else False,
        )
        for i, r in enumerate(panel)
    )

    target = result.target_result
    # Target is a single test; BH-q and Bonferroni p both collapse to
    # raw p.  Same threshold as the panel for coherent semantics.
    target_p = target.slope_pvalue
    target_adjusted = AdjustedSeedResult(
        seed=target.seed,
        stratum=target.stratum,
        slope=target.slope,
        slope_pvalue=target_p,
        bh_qvalue=float(target_p),
        bonferroni_pvalue=float(min(target_p, 1.0)) if not np.isnan(target_p) else float("nan"),
        discovery=bool(target_p < q_threshold) if not np.isnan(target_p) else False,
    )

    # Stratum-vs-stratum Mann–Whitney U (one-sided: a < b lexically,
    # tests "a slopes more negative than b slopes").  One row per
    # unordered pair; reciprocal direction recoverable from u + n_a*n_b.
    by_stratum: dict[str, list[float]] = {}
    for r in panel:
        by_stratum.setdefault(r.stratum, []).append(r.slope)
    strata_present = sorted(by_stratum.keys())
    comparisons: list[StratumComparison] = []
    for stratum_a, stratum_b in combinations(strata_present, 2):
        slopes_a = by_stratum[stratum_a]
        slopes_b = by_stratum[stratum_b]
        stat = sp_stats.mannwhitneyu(
            slopes_a, slopes_b, alternative="less",
        )
        comparisons.append(StratumComparison(
            stratum_a=stratum_a,
            stratum_b=stratum_b,
            n_a=len(slopes_a),
            n_b=len(slopes_b),
            median_a=float(np.median(slopes_a)),
            median_b=float(np.median(slopes_b)),
            u_statistic=float(stat.statistic),
            pvalue=float(stat.pvalue),
        ))

    # Target empirical rank in the panel.  Conservative: ties counted
    # as ``<=``.  PanelResult validation guarantees n_completed > 0
    # only if the panel has members; for an empty panel we still
    # report a degenerate position rather than skip the field.
    panel_slopes = np.array([r.slope for r in panel], dtype=np.float64)
    n_le = int(np.sum(panel_slopes <= target.slope))
    rank_left = n_le + 1  # 1-indexed
    emp_p_left = float((n_le + 1) / (n_completed + 1))

    target_position = TargetPosition(
        target_slope=target.slope,
        panel_n=n_completed,
        rank_left_tail=rank_left,
        empirical_p_left=emp_p_left,
    )

    return PanelAnalysis(
        design=result.design,
        q_threshold=q_threshold,
        panel_seeds_adjusted=panel_adjusted,
        target_adjusted=target_adjusted,
        stratum_comparisons=tuple(comparisons),
        target_position=target_position,
    )
