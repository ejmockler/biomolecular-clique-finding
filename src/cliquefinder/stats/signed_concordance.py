"""Signed concordance test for INDRA-predicted regulatory directions.

Tests whether the DIRECTION of each target's differential expression matches
the prediction from the INDRA knowledge graph edge type.  For loss-of-function
(e.g., C9orf72 repeat expansion → haploinsufficiency):

- Activation targets: predicted DOWN (regulator lost → less activation)
- Inhibition targets: predicted UP (regulator lost → less inhibition)

Reports subgroup-level concordance (activation vs repression separately)
to avoid masking heterogeneous effects in a pooled statistic.  Uses a
permutation null (label shuffle) as the primary test to handle inter-gene
correlation, with binomial as a secondary reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from cliquefinder.stats.target_set import TargetSet


@dataclass
class SubgroupResult:
    """Concordance result for one prediction class (activation or repression)."""
    n: int
    n_concordant: int
    concordance_rate: float
    background_rate: float
    binomial_pvalue: float  # one-sided: greater (concordance) or less (anti-concordance)


@dataclass
class SignedConcordanceResult:
    """Result of signed concordance test."""

    n_unambiguous: int
    n_concordant: int
    concordance_rate: float
    background_concordance_rate: float
    binomial_pvalue: float  # two-sided pooled binomial
    permutation_pvalue: float  # label-shuffle null (primary)

    n_predicted_down: int
    n_predicted_up: int
    n_mixed_excluded: int
    n_no_tstat: int

    # Subgroup breakdown (Finding 1: don't hide heterogeneity)
    activation_subgroup: Optional[dict] = None  # SubgroupResult as dict
    repression_subgroup: Optional[dict] = None

    # Sensitivity: gain-of-function model (Finding 4)
    gof_concordance_rate: float = 0.0
    gof_binomial_pvalue: float = 1.0
    best_model: str = "neither"  # "lof", "gof", or "neither"

    n_permutations: int = 0
    target_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "n_unambiguous": self.n_unambiguous,
            "n_concordant": self.n_concordant,
            "concordance_rate": self.concordance_rate,
            "background_concordance_rate": self.background_concordance_rate,
            "binomial_pvalue": self.binomial_pvalue,
            "permutation_pvalue": self.permutation_pvalue,
            "n_predicted_down": self.n_predicted_down,
            "n_predicted_up": self.n_predicted_up,
            "n_mixed_excluded": self.n_mixed_excluded,
            "n_no_tstat": self.n_no_tstat,
            "activation_subgroup": self.activation_subgroup,
            "repression_subgroup": self.repression_subgroup,
            "gof_concordance_rate": self.gof_concordance_rate,
            "gof_binomial_pvalue": self.gof_binomial_pvalue,
            "best_model": self.best_model,
            "n_permutations": self.n_permutations,
            "target_details": self.target_details,
        }
        return d


def predict_direction(
    edge_list: list[dict],
    loss_of_function: bool = True,
) -> Optional[str]:
    """Predict expected direction from INDRA edge types.

    Returns "predicted_down", "predicted_up", or None (ambiguous/unknown).
    """
    reg_types = {e.get("regulation_type") for e in edge_list}
    has_act = "activation" in reg_types
    has_rep = "repression" in reg_types

    if has_act and has_rep:
        return None  # mixed → ambiguous

    if has_act:
        return "predicted_down" if loss_of_function else "predicted_up"
    if has_rep:
        return "predicted_up" if loss_of_function else "predicted_down"

    return None  # phosphorylation-only or unknown


def _compute_subgroup(
    n_subgroup: int,
    n_concordant: int,
    background_rate: float,
) -> dict:
    """Compute subgroup concordance with appropriate test direction."""
    if n_subgroup == 0:
        return {"n": 0, "n_concordant": 0, "concordance_rate": 0.0,
                "background_rate": background_rate, "binomial_pvalue": 1.0}

    rate = n_concordant / n_subgroup

    # Use two-sided test to detect both concordance AND anti-concordance
    bg = max(0.01, min(0.99, background_rate))
    p = float(scipy_stats.binomtest(
        n_concordant, n_subgroup, p=bg, alternative="two-sided",
    ).pvalue)

    return {"n": n_subgroup, "n_concordant": n_concordant,
            "concordance_rate": rate, "background_rate": background_rate,
            "binomial_pvalue": p}


def compute_signed_concordance(
    protein_df: pd.DataFrame,
    target_set: TargetSet,
    loss_of_function: bool = True,
    n_permutations: int = 1000,
    seed: int | None = None,
    alpha: float = 0.05,
) -> SignedConcordanceResult:
    """Test whether observed t-stat signs match INDRA-predicted directions.

    Uses a permutation null (label shuffle) as the primary test to handle
    inter-gene correlation.  Also reports binomial tests per subgroup
    (activation vs repression) to expose heterogeneous effects.

    Args:
        protein_df: DataFrame with columns ``feature_id``, ``t_statistic``,
            ``is_target``.  Typically from Phase 1 ``run_protein_differential``.
        target_set: TargetSet with populated ``edge_metadata``.
        loss_of_function: If True (default), activation → predicted DOWN.
        n_permutations: Number of label shuffles for permutation null.
        seed: Random seed for permutation reproducibility.

    Returns:
        SignedConcordanceResult with subgroup breakdown and permutation p-value.
    """
    _empty = SignedConcordanceResult(
        n_unambiguous=0, n_concordant=0, concordance_rate=0.0,
        background_concordance_rate=0.5, binomial_pvalue=1.0,
        permutation_pvalue=1.0,
        n_predicted_down=0, n_predicted_up=0,
        n_mixed_excluded=0, n_no_tstat=0,
    )

    if not target_set.edge_metadata:
        return _empty

    # Get predictions for unambiguous targets
    predictions = target_set.get_unambiguous_targets(
        loss_of_function=loss_of_function
    )
    n_mixed = len(target_set.get_mixed_targets())

    # Build feature_id → t_statistic lookup from protein_df
    target_fids = set(target_set.targets.values())
    t_stats: dict[str, float] = {}
    if "feature_id" in protein_df.columns:
        for _, row in protein_df.iterrows():
            fid = row["feature_id"]
            t = row.get("t_statistic")
            if pd.notna(t):
                t_stats[fid] = float(t)
    elif not protein_df.empty:
        for idx, row in protein_df.iterrows():
            t = row.get("t_statistic")
            if pd.notna(t):
                t_stats[str(idx)] = float(t)

    # Match predictions to observed t-statistics
    details: list[dict] = []
    act_concordant = 0
    rep_concordant = 0
    n_predicted_down = 0
    n_predicted_up = 0
    n_no_tstat = 0

    for sym, predicted in predictions.items():
        fid = target_set.targets.get(sym)
        if fid is None:
            continue
        t_val = t_stats.get(fid)
        if t_val is None:
            n_no_tstat += 1
            continue

        if predicted == "predicted_down":
            n_predicted_down += 1
            concordant = t_val < 0
            if concordant:
                act_concordant += 1
        else:  # predicted_up
            n_predicted_up += 1
            concordant = t_val > 0
            if concordant:
                rep_concordant += 1

        details.append({
            "symbol": sym,
            "feature_id": fid,
            "predicted": predicted,
            "t_statistic": round(t_val, 4),
            "concordant": concordant,
        })

    n_unambiguous = n_predicted_down + n_predicted_up
    n_concordant = act_concordant + rep_concordant

    if n_unambiguous == 0:
        _empty.n_mixed_excluded = n_mixed
        _empty.n_no_tstat = n_no_tstat
        _empty.target_details = details
        return _empty

    concordance_rate = n_concordant / n_unambiguous

    # Background rate: exclude targets to avoid circularity (Finding 3)
    bg_t_values = [t for fid, t in t_stats.items() if fid not in target_fids]
    if bg_t_values:
        bg_frac_negative = sum(1 for t in bg_t_values if t < 0) / len(bg_t_values)
    else:
        bg_frac_negative = 0.5

    p_pred_down = n_predicted_down / n_unambiguous
    p_pred_up = n_predicted_up / n_unambiguous
    background_rate = (
        p_pred_down * bg_frac_negative
        + p_pred_up * (1.0 - bg_frac_negative)
    )
    background_rate = max(0.01, min(0.99, background_rate))

    # Pooled binomial: two-sided to detect both concordance AND anti-concordance
    pooled_binom = scipy_stats.binomtest(
        n_concordant, n_unambiguous, p=background_rate, alternative="two-sided",
    )

    # Subgroup tests (Finding 1: don't pool heterogeneous groups)
    act_subgroup = _compute_subgroup(
        n_predicted_down, act_concordant, bg_frac_negative,
    )
    rep_subgroup = _compute_subgroup(
        n_predicted_up, rep_concordant, 1.0 - bg_frac_negative,
    )

    # Permutation null (Finding 11): shuffle prediction labels among targets
    # to account for inter-gene correlation.  Keeps the 21/15 prediction
    # split fixed, shuffles which targets get which label.
    rng = np.random.default_rng(seed)
    target_t_values = np.array([d["t_statistic"] for d in details])
    target_predictions = np.array([d["predicted"] for d in details])
    n_perm_ge = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(target_predictions)
        perm_concordant = 0
        for t_val, pred in zip(target_t_values, shuffled):
            if pred == "predicted_down" and t_val < 0:
                perm_concordant += 1
            elif pred == "predicted_up" and t_val > 0:
                perm_concordant += 1
        if perm_concordant >= n_concordant:
            n_perm_ge += 1

    perm_pvalue = (n_perm_ge + 1) / (n_permutations + 1)

    # Gain-of-function sensitivity (Finding 4): reverse predictions
    gof_concordant = 0
    for d in details:
        t_val = d["t_statistic"]
        pred = d["predicted"]
        # GoF reverses: predicted_down becomes predicted_up and vice versa
        if pred == "predicted_down" and t_val > 0:
            gof_concordant += 1
        elif pred == "predicted_up" and t_val < 0:
            gof_concordant += 1

    gof_rate = gof_concordant / n_unambiguous if n_unambiguous > 0 else 0.0
    # GoF background rate is 1 - LoF background rate
    gof_bg = 1.0 - background_rate
    gof_bg = max(0.01, min(0.99, gof_bg))
    gof_binom = scipy_stats.binomtest(
        gof_concordant, n_unambiguous, p=gof_bg, alternative="two-sided",
    )

    # Determine best model
    if float(pooled_binom.pvalue) < alpha:
        best_model = "lof"
    elif float(gof_binom.pvalue) < alpha:
        best_model = "gof"
    else:
        best_model = "neither"

    return SignedConcordanceResult(
        n_unambiguous=n_unambiguous,
        n_concordant=n_concordant,
        concordance_rate=concordance_rate,
        background_concordance_rate=background_rate,
        binomial_pvalue=float(pooled_binom.pvalue),
        permutation_pvalue=float(perm_pvalue),
        n_predicted_down=n_predicted_down,
        n_predicted_up=n_predicted_up,
        n_mixed_excluded=n_mixed,
        n_no_tstat=n_no_tstat,
        activation_subgroup=act_subgroup,
        repression_subgroup=rep_subgroup,
        gof_concordance_rate=gof_rate,
        gof_binomial_pvalue=float(gof_binom.pvalue),
        best_model=best_model,
        n_permutations=n_permutations,
        target_details=details,
    )
