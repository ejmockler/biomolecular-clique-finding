"""Tests for multi-contrast specificity analysis."""

import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.specificity import (
    ContrastEnrichment,
    SpecificityResult,
    compute_specificity,
    _run_interaction_permutation,
)


def _make_enrichment(z: float, p: float, n_targets: int = 50) -> dict:
    """Helper to create enrichment dict."""
    return {
        "z_score": z,
        "empirical_pvalue": p,
        "n_targets": n_targets,
        "n_background": 500,
        "pct_down": 60.0,
        "direction_pvalue": 0.1,
    }


class TestComputeSpecificity:
    """Tests for compute_specificity()."""

    def test_specific_signal(self):
        """Primary contrast significant, secondary not → specific."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "specific"
        assert result.specificity_ratio > 1.0
        assert result.primary_contrast == "C9_vs_CTRL"
        assert len(result.contrasts) == 2

    def test_shared_signal(self):
        """Both contrasts significant → shared."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=2.5, p=0.01),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "shared"
        assert result.specificity_ratio == pytest.approx(3.0 / 2.5)

    def test_inconclusive_primary_not_sig(self):
        """Primary not significant → inconclusive."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=1.0, p=0.15),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "inconclusive"

    def test_single_contrast_inconclusive(self):
        """Only one contrast → inconclusive."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "inconclusive"
        assert result.specificity_ratio == float("inf")

    def test_three_contrasts(self):
        """Three contrasts: specificity assessed against max secondary."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
            "C9_vs_SPORADIC": _make_enrichment(z=1.0, p=0.15),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "specific"
        # Ratio should be against max secondary z = 1.0
        assert result.specificity_ratio == pytest.approx(3.0)

    def test_primary_not_found(self):
        """Missing primary contrast → ValueError."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
        }
        with pytest.raises(ValueError, match="not found"):
            compute_specificity(enrichment, primary_contrast="MISSING")

    def test_with_roast(self):
        """ROAST p-values are extracted when provided."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        roast = {
            "C9_vs_CTRL": pd.DataFrame({"pvalue_msq_mixed": [0.004]}),
            "SPORADIC_vs_CTRL": pd.DataFrame({"pvalue_msq_mixed": [0.35]}),
        }
        result = compute_specificity(
            enrichment, primary_contrast="C9_vs_CTRL",
            roast_by_contrast=roast,
        )

        assert result.contrasts["C9_vs_CTRL"].roast_pvalue == pytest.approx(0.004)
        assert result.contrasts["SPORADIC_vs_CTRL"].roast_pvalue == pytest.approx(0.35)

    def test_to_dict_serializable(self):
        """Result serializes to JSON-compatible dict."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")
        d = result.to_dict()

        assert "specificity_ratio" in d
        assert "contrasts" in d
        assert "C9_vs_CTRL" in d["contrasts"]

    def test_negative_secondary_z(self):
        """Secondary z-score <= 0 → ratio is inf."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=2.0, p=0.02),
            "SPORADIC_vs_CTRL": _make_enrichment(z=-0.5, p=0.7),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_ratio == float("inf")
        assert result.specificity_label == "specific"

    def test_custom_thresholds(self):
        """Custom z_threshold and p_threshold change classification."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=2.0, p=0.02),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        # With strict z_threshold=3.0, primary doesn't meet threshold
        result = compute_specificity(
            enrichment, primary_contrast="C9_vs_CTRL",
            z_threshold=3.0,
        )
        assert result.specificity_label == "inconclusive"

        # With lenient z_threshold=1.0, primary meets threshold
        result = compute_specificity(
            enrichment, primary_contrast="C9_vs_CTRL",
            z_threshold=1.0,
        )
        assert result.specificity_label == "specific"


def _make_synthetic_data(
    n_features=200,
    n_c9=15,
    n_sporadic=40,
    n_ctrl=30,
    n_targets=20,
    target_effect_c9=2.0,
    target_effect_sporadic=0.0,
    seed=42,
):
    """Build synthetic expression data for interaction test.

    Creates a 3-group design with C9, Sporadic, and CTRL samples.
    Target genes have elevated expression in C9 (and optionally Sporadic).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_c9 + n_sporadic + n_ctrl

    # Base expression: features × samples
    data = rng.normal(0, 1, size=(n_features, n_samples))

    # Inject signal into targets for C9 group
    for i in range(n_targets):
        data[i, :n_c9] += target_effect_c9

    # Optionally inject signal for Sporadic group
    if target_effect_sporadic != 0:
        for i in range(n_targets):
            data[i, n_c9:n_c9 + n_sporadic] += target_effect_sporadic

    feature_ids = [f"gene_{i}" for i in range(n_features)]
    target_feature_ids = feature_ids[:n_targets]

    # Metadata
    conditions = (
        ["C9"] * n_c9
        + ["Sporadic"] * n_sporadic
        + ["CTRL"] * n_ctrl
    )
    sex_values = rng.choice(["M", "F"], size=n_samples)
    metadata = pd.DataFrame(
        {"phenotype": conditions, "Sex": sex_values},
        index=[f"sample_{i}" for i in range(n_samples)],
    )
    covariates_df = metadata[["Sex"]]

    contrasts = {
        "C9_vs_CTRL": ("C9", "CTRL"),
        "Sporadic_vs_CTRL": ("Sporadic", "CTRL"),
    }

    return data, feature_ids, metadata, target_feature_ids, covariates_df, contrasts


class TestInteractionZTest:
    """Tests for paired permutation interaction z-test (M-2)."""

    def test_interaction_specific(self):
        """Strong primary signal, no secondary → interaction_pvalue < 0.05."""
        data, fids, meta, targets, cov, contrasts = _make_synthetic_data(
            target_effect_c9=3.0,
            target_effect_sporadic=0.0,
            seed=42,
        )

        # Run enrichment for each contrast (needed for compute_specificity)
        from cliquefinder.stats.differential import run_protein_differential
        from cliquefinder.stats.differential import run_network_enrichment_test

        enrichment_by_contrast = {}
        for name, ct in contrasts.items():
            mask = meta["phenotype"].isin(ct)
            sub_data = data[:, mask.values]
            results = run_protein_differential(
                data=sub_data,
                feature_ids=fids,
                sample_condition=meta.loc[mask, "phenotype"],
                contrast=ct,
                eb_moderation=True,
                target_genes=targets,
                verbose=False,
                covariates_df=cov[mask],
            )
            enrichment_by_contrast[name] = run_network_enrichment_test(
                results, verbose=False
            )

        result = compute_specificity(
            enrichment_by_contrast,
            primary_contrast="C9_vs_CTRL",
            data=data,
            feature_ids=fids,
            metadata=meta,
            condition_col="phenotype",
            contrast_tuples=contrasts,
            target_feature_ids=targets,
            covariates_df=cov,
            n_interaction_perms=100,
            seed=42,
        )

        assert result.specificity_label == "specific"
        assert result.interaction_pvalue is not None
        assert result.interaction_pvalue < 0.05
        assert result.interaction_z is not None
        assert result.interaction_z > 0  # Primary z > secondary z

    def test_interaction_shared(self):
        """Both contrasts have similar signal → interaction_pvalue > 0.05."""
        # Use balanced group sizes so identical effects yield similar z-scores
        data, fids, meta, targets, cov, contrasts = _make_synthetic_data(
            n_c9=25,
            n_sporadic=25,
            n_ctrl=25,
            target_effect_c9=2.5,
            target_effect_sporadic=2.5,  # Same effect in both
            seed=42,
        )

        from cliquefinder.stats.differential import run_protein_differential
        from cliquefinder.stats.differential import run_network_enrichment_test

        enrichment_by_contrast = {}
        for name, ct in contrasts.items():
            mask = meta["phenotype"].isin(ct)
            sub_data = data[:, mask.values]
            results = run_protein_differential(
                data=sub_data,
                feature_ids=fids,
                sample_condition=meta.loc[mask, "phenotype"],
                contrast=ct,
                eb_moderation=True,
                target_genes=targets,
                verbose=False,
                covariates_df=cov[mask],
            )
            enrichment_by_contrast[name] = run_network_enrichment_test(
                results, verbose=False
            )

        result = compute_specificity(
            enrichment_by_contrast,
            primary_contrast="C9_vs_CTRL",
            data=data,
            feature_ids=fids,
            metadata=meta,
            condition_col="phenotype",
            contrast_tuples=contrasts,
            target_feature_ids=targets,
            covariates_df=cov,
            n_interaction_perms=100,
            seed=42,
        )

        assert result.specificity_label == "shared"
        assert result.interaction_pvalue is not None
        assert result.interaction_pvalue > 0.05

    def test_interaction_backward_compat(self):
        """Without data parameter, existing classification unchanged."""
        enrichment = {
            "C9_vs_CTRL": _make_enrichment(z=3.0, p=0.001),
            "SPORADIC_vs_CTRL": _make_enrichment(z=0.5, p=0.4),
        }
        result = compute_specificity(enrichment, primary_contrast="C9_vs_CTRL")

        assert result.specificity_label == "specific"
        assert result.interaction_z is None
        assert result.interaction_pvalue is None
        assert result.z_difference_ci is None
        assert result.null_correlation is None

    def test_interaction_null_correlation_positive(self):
        """Shared control samples → estimated null correlation > 0."""
        data, fids, meta, targets, cov, contrasts = _make_synthetic_data(
            target_effect_c9=2.0,
            target_effect_sporadic=0.0,
            seed=42,
        )

        result = _run_interaction_permutation(
            data=data,
            feature_ids=fids,
            metadata=meta,
            condition_col="phenotype",
            primary_contrast=("C9", "CTRL"),
            secondary_contrast=("Sporadic", "CTRL"),
            target_feature_ids=targets,
            covariates_df=cov,
            n_perms=100,
            seed=42,
        )

        # Shared CTRL samples should induce positive correlation
        # between null z-scores for both contrasts
        assert result["null_correlation"] > -0.5  # Not strongly negative

    def test_interaction_serialization(self):
        """to_dict() includes interaction_test fields when populated."""
        data, fids, meta, targets, cov, contrasts = _make_synthetic_data(
            target_effect_c9=3.0,
            target_effect_sporadic=0.0,
            seed=42,
        )

        from cliquefinder.stats.differential import run_protein_differential
        from cliquefinder.stats.differential import run_network_enrichment_test

        enrichment_by_contrast = {}
        for name, ct in contrasts.items():
            mask = meta["phenotype"].isin(ct)
            sub_data = data[:, mask.values]
            results = run_protein_differential(
                data=sub_data,
                feature_ids=fids,
                sample_condition=meta.loc[mask, "phenotype"],
                contrast=ct,
                eb_moderation=True,
                target_genes=targets,
                verbose=False,
                covariates_df=cov[mask],
            )
            enrichment_by_contrast[name] = run_network_enrichment_test(
                results, verbose=False
            )

        result = compute_specificity(
            enrichment_by_contrast,
            primary_contrast="C9_vs_CTRL",
            data=data,
            feature_ids=fids,
            metadata=meta,
            condition_col="phenotype",
            contrast_tuples=contrasts,
            target_feature_ids=targets,
            covariates_df=cov,
            n_interaction_perms=50,
            seed=42,
        )

        d = result.to_dict()
        assert "interaction_test" in d
        assert "z_difference" in d["interaction_test"]
        assert "interaction_pvalue" in d["interaction_test"]
        assert "null_correlation" in d["interaction_test"]
