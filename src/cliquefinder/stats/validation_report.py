"""
Validation report aggregation.

Collects results from all baseline validation phases into a single
report for JSON serialization and console output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json


@dataclass
class ValidationReport:
    """Aggregated validation report across all phases.

    Attributes:
        phases: Dict of phase name → phase results dict.
        summary: Overall assessment string.
        verdict: "validated", "inconclusive", or "refuted".
    """

    phases: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    verdict: str = "inconclusive"
    bootstrap_stability: float | None = None
    bootstrap_ci: tuple[float, float] | None = None

    def add_phase(self, name: str, result: dict) -> None:
        """Add a phase result to the report."""
        self.phases[name] = result

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "verdict": self.verdict,
            "summary": self.summary,
            "phases": self.phases,
        }
        if self.bootstrap_stability is not None:
            d["bootstrap_stability"] = self.bootstrap_stability
            d["bootstrap_ci"] = list(self.bootstrap_ci) if self.bootstrap_ci else None
        return d

    def save(self, path) -> None:
        """Save report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def compute_verdict(
        self,
        *,
        alpha: float = 0.05,
        neg_ctrl_percentile: float = 10.0,
    ) -> None:
        """Compute overall verdict from phase results.

        Uses hierarchical logic rather than equal-weight voting:

        - Phase 1 (covariate-adjusted) and Phase 3 (label permutation)
          are MANDATORY gates -- both must pass for "validated".
        - Phase 2 (specificity) characterizes the signal as "specific"
          or "shared" -- a "shared" signal is still biologically valid.
        - Phase 4 (matched) and Phase 5 (negative controls) provide
          supplementary evidence, weighted as supporting.

        The overall verdict reflects whether the core signal survives
        confound correction and permutation null, with nuance from
        supplementary phases.

        Args:
            alpha: Significance threshold applied to Phase 1 (covariate-
                adjusted enrichment), Phase 3 (label permutation null),
                and Phase 4 (matched subsampling). Default is 0.05.
            neg_ctrl_percentile: Percentile threshold for Phase 5
                negative control pass/fail (default: 10.0). The target
                gene set must rank below this percentile among random
                control sets to pass.

        Bounded-FWER rationale
        ----------------------
        Three phases (1, 3, 4) apply the same ``alpha`` threshold:

        1. These phases test *different aspects* of the same underlying
           enrichment signal -- confound correction (Phase 1), label
           robustness (Phase 3), and sample-composition sensitivity
           (Phase 4) -- rather than independent hypotheses about
           unrelated effects. Bonferroni or Benjamini-Hochberg
           adjustment is therefore overly conservative because the
           test statistics are positively correlated (all driven by
           the same target-gene signal).

        2. The hierarchical gating structure provides bounded-FWER
           control: Phase 1 AND Phase 3 must *both* pass for a
           "validated" verdict, which is strictly more stringent than
           requiring either alone.

           **Design asymmetry note:** Phase 1 (covariate-adjusted
           enrichment) and Phase 3 (label permutation null) are *not*
           independent -- they test the same gene set on the same data.
           If their test statistics have correlation rho under the
           global null, the joint-pass probability is:

               P(both pass) = alpha * Phi((z_alpha - rho * z_alpha)
                              / sqrt(1 - rho^2))

           where z_alpha = Phi^{-1}(1 - alpha).

           For alpha = 0.05 and typical rho in [0.3, 0.7], the
           effective FWER is bounded by approximately 0.006 to 0.020,
           which is still well below the nominal alpha = 0.05. The
           gating therefore provides meaningful multiplicity reduction
           even under positive dependence, but the exact bound depends
           on the inter-phase correlation and is *not* alpha^2 unless
           the phases are independent (rho = 0). At moderate
           correlation (rho = 0.5) the bound is approximately 0.012;
           at high correlation (rho = 0.8) it rises to approximately
           0.030 -- still below alpha but 12x the independent-case
           value of 0.0025.

        3. Phase 4 is supplementary and cannot override a "refuted"
           verdict from the mandatory gates -- it only modulates between
           "validated" and "inconclusive" when the gates pass.

        Design asymmetry note
        ---------------------
        This framework is designed for VALIDATION of a candidate
        finding, not balanced hypothesis testing. The verdict is
        intentionally asymmetric:

        - "Validated" requires positive evidence from both mandatory
          gates
        - "Refuted" requires both gates to fail simultaneously
        - "Inconclusive" is the default for mixed or ambiguous results

        This asymmetry biases toward "inconclusive" rather than
        "refuted" for null signals. Users who want formal two-sided
        testing should use the individual phase p-values with
        Bonferroni-Holm correction across the mandatory gates
        (alpha/2 each).

        For conservative family-wise error control, set ``alpha=0.01``.
        """
        details: dict[str, str] = {}

        # --- P-value sidedness conventions (STAT-10) ---
        # Phase 1 (covariate-adjusted): one-sided (enrichment direction)
        # Phase 2 (specificity interaction): two-sided (|Δz| in either direction)
        # Phase 3 (label permutation): one-sided (null z >= observed z)
        # Phase 4 (matched subsampling): one-sided (enrichment direction)
        # Phase 5 (negative controls): ROAST MSQ mixed = effectively two-sided
        #
        # When comparing p-values across phases, note that one-sided
        # p-values (Phases 1, 3, 4) test enrichment specifically, while
        # two-sided p-values (Phase 2 interaction, Phase 5 ROAST mixed)
        # test deviation in either direction.

        # --- Mandatory gates ---
        # Phase 1: Covariate-adjusted enrichment
        cov = self.phases.get("covariate_adjusted")
        gate_adjusted = False
        if cov and cov.get("status") != "failed":
            p = cov.get("empirical_pvalue", 1.0)
            gate_adjusted = p < alpha
            details["covariate_adjusted"] = f"p={p:.4f} ({'pass' if gate_adjusted else 'fail'})"

        # Phase 3: Label permutation (use both stratified and free)
        perm = self.phases.get("label_permutation")
        gate_permutation = False
        if perm and perm.get("status") != "failed":
            # Use stratified p-value as primary gate
            strat = perm.get("stratified", perm)
            strat_p = strat.get("permutation_pvalue", perm.get("permutation_pvalue", 1.0))
            gate_permutation = strat_p < alpha
            details["label_permutation_stratified"] = f"p={strat_p:.4f}"

            # Also report free permutation
            free = perm.get("free", {})
            free_p = free.get("permutation_pvalue", None)
            if free_p is not None:
                details["label_permutation_free"] = f"p={free_p:.4f}"
                # If stratified passes but free fails, flag potential issue
                if gate_permutation and free_p >= alpha:
                    details["permutation_warning"] = (
                        "Stratified passes but free fails — signal may "
                        "partly reflect covariate structure."
                    )

        # --- Supplementary evidence ---
        supplementary_pass = 0
        supplementary_total = 0

        # Phase 2: Specificity (characterization, not gate)
        spec = self.phases.get("specificity")
        specificity_label = "not_tested"
        if spec and spec.get("status") != "failed":
            specificity_label = spec.get("specificity_label", "inconclusive")
            details["specificity"] = specificity_label
            # "shared" is still valid biology — only "inconclusive" is a concern
            if specificity_label in ("specific", "shared"):
                supplementary_pass += 1
            supplementary_total += 1

        # Phase 4: Matched subsampling
        matched = self.phases.get("matched_reanalysis")
        if matched and matched.get("status") != "failed":
            p = matched.get("empirical_pvalue", 1.0)
            n_matched = matched.get("n_matched", 0)
            passed = p < alpha
            supplementary_pass += int(passed)
            supplementary_total += 1
            details["matched_reanalysis"] = (
                f"p={p:.4f}, n={n_matched} ({'pass' if passed else 'fail'})"
            )

        # Phase 5: Negative controls
        # Prefer competitive z metrics when available (cross-phase consistency
        # with Phases 1/3/4); fall back to ROAST percentile
        neg = self.phases.get("negative_controls")
        if neg and neg.get("status") != "failed":
            comp_z = neg.get("competitive_z", {})
            if comp_z:
                percentile = comp_z.get("percentile", neg.get("target_percentile", 100))
                fpr = comp_z.get("fpr", neg.get("fpr", 1.0))
            else:
                percentile = neg.get("target_percentile", 100)
                fpr = neg.get("fpr", 1.0)
            passed = percentile < neg_ctrl_percentile
            supplementary_pass += int(passed)
            supplementary_total += 1
            details["negative_controls"] = (
                f"percentile={percentile:.1f}%, FPR={fpr:.3f} "
                f"({'pass' if passed else 'fail'})"
            )

        # --- Compute verdict ---
        # ARCH-17: Distinguish skipped supplementary phases from failed ones.
        # Phases that didn't run (e.g., Phase 2 for single-contrast datasets)
        # should NOT count as failures.  ``supplementary_total`` only counts
        # phases that actually executed and produced a result.
        supplementary_failed = supplementary_total - supplementary_pass

        if not gate_adjusted and not cov:
            self.verdict = "inconclusive"
            self.summary = "Core phases not completed."
        elif gate_adjusted and gate_permutation:
            if supplementary_total == 0:
                # No supplementary phases ran — don't penalise for missing data
                self.verdict = "validated"
                qualifier = ""
                if specificity_label == "shared":
                    qualifier = " (shared across disease subtypes)"
                elif specificity_label == "specific":
                    qualifier = " (disease-subtype specific)"
                self.summary = (
                    f"Signal validated{qualifier}: survives covariate "
                    f"adjustment and label permutation null. "
                    f"No supplementary phases ran."
                )
            elif supplementary_failed > 0 and supplementary_pass == 0:
                # All supplementary phases that ran actually FAILED
                self.verdict = "inconclusive"
                self.summary = (
                    f"Core tests pass (adjusted enrichment + permutation null) "
                    f"but all {supplementary_total} supplementary phases fail. "
                    f"Signal survives confound correction but calibration "
                    f"is concerning."
                )
            else:
                # At least one supplementary phase passed
                self.verdict = "validated"
                qualifier = ""
                if specificity_label == "shared":
                    qualifier = " (shared across disease subtypes)"
                elif specificity_label == "specific":
                    qualifier = " (disease-subtype specific)"
                self.summary = (
                    f"Signal validated{qualifier}: survives covariate "
                    f"adjustment and label permutation null. "
                    f"Supplementary: {supplementary_pass}/{supplementary_total} pass."
                )
        elif gate_adjusted and not gate_permutation and perm:
            self.verdict = "inconclusive"
            self.summary = (
                "Covariate-adjusted enrichment is significant but label "
                "permutation null is not — effect may not be robust to "
                "label reassignment."
            )
        elif not gate_adjusted and gate_permutation:
            self.verdict = "inconclusive"
            self.summary = (
                "Label permutation is significant but covariate-adjusted "
                "enrichment is not — signal may reflect confounding."
            )
        else:
            self.verdict = "refuted"
            self.summary = (
                "Neither covariate-adjusted enrichment nor label permutation "
                "null reaches significance — signal is likely spurious."
            )

        # Annotate with any failed phases
        failed = [
            name for name, data in self.phases.items()
            if isinstance(data, dict) and data.get("status") == "failed"
        ]
        if failed:
            self.summary += f" [Phases failed: {', '.join(failed)}]"

        # Annotate bootstrap stability (report annotation, not a gate)
        if self.bootstrap_stability is not None and self.bootstrap_stability < 0.7:
            self.summary += (
                " [low bootstrap stability — result sensitive to sample composition]"
            )

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        print(f"Verdict: {self.verdict.upper()}")
        print(f"Summary: {self.summary}")
        print()

        for phase_name, phase_data in self.phases.items():
            print(f"  {phase_name}:")
            if isinstance(phase_data, dict):
                for key, val in phase_data.items():
                    if not isinstance(val, (dict, list)):
                        print(f"    {key}: {val}")
            print()
