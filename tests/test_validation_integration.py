"""Integration test for full 5-phase validation pipeline."""
import numpy as np
import pandas as pd
import pytest

from cliquefinder.stats.differential import run_protein_differential, run_network_enrichment_test
from cliquefinder.stats.label_permutation import run_label_permutation_null
from cliquefinder.stats.validation_report import ValidationReport


class TestValidationPipelineIntegration:
    """End-to-end test of the validation pipeline on synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Build synthetic proteomics data with known signal."""
        rng = np.random.default_rng(42)
        n_features = 200
        n_samples = 60
        n_targets = 15

        # 30 case, 30 control
        data = rng.normal(0, 1, (n_features, n_samples))
        # Inject strong signal in targets for group A
        data[:n_targets, :30] += 2.5

        feature_ids = [f"gene_{i}" for i in range(n_features)]
        target_gene_ids = feature_ids[:n_targets]
        condition = np.array(["CASE"] * 30 + ["CTRL"] * 30)
        contrast = ("CASE", "CTRL")

        covariates = pd.DataFrame({
            "Sex": rng.choice(["M", "F"], size=n_samples),
        })

        return {
            "data": data, "feature_ids": feature_ids,
            "condition": condition, "contrast": contrast,
            "target_gene_ids": target_gene_ids,
            "covariates": covariates,
        }

    def test_full_pipeline_signal_data(self, synthetic_data):
        """Strong signal should produce 'validated' verdict."""
        d = synthetic_data

        # Phase 1: Covariate-adjusted enrichment
        protein_df = run_protein_differential(
            data=d["data"], feature_ids=d["feature_ids"],
            sample_condition=d["condition"], contrast=d["contrast"],
            target_gene_ids=d["target_gene_ids"],
            covariates_df=d["covariates"], verbose=False,
        )
        enrichment = run_network_enrichment_test(protein_df, verbose=False)

        # Phase 3: Label permutation
        perm_result = run_label_permutation_null(
            data=d["data"], feature_ids=d["feature_ids"],
            sample_condition=d["condition"], contrast=d["contrast"],
            target_gene_ids=d["target_gene_ids"],
            n_permutations=99, seed=42, verbose=False,
        )

        # Assemble report (mirror orchestrator's nested structure)
        report = ValidationReport()
        report.add_phase("covariate_adjusted", enrichment.to_dict())
        report.add_phase("label_permutation", {
            "stratified": perm_result.to_dict(),
            "free": {},
            "permutation_pvalue": perm_result.permutation_pvalue,
        })
        report.compute_verdict()

        assert report.verdict == "validated"
        assert enrichment.z_score > 1.5
        assert perm_result.permutation_pvalue < 0.05

    def test_null_pipeline_no_signal(self, synthetic_data):
        """Null data should NOT produce 'validated' verdict."""
        d = synthetic_data
        # Zero out the signal
        d["data"][:15, :30] -= 2.5

        protein_df = run_protein_differential(
            data=d["data"], feature_ids=d["feature_ids"],
            sample_condition=d["condition"], contrast=d["contrast"],
            target_gene_ids=d["target_gene_ids"],
            verbose=False,
        )
        enrichment = run_network_enrichment_test(protein_df, verbose=False)

        perm_result = run_label_permutation_null(
            data=d["data"], feature_ids=d["feature_ids"],
            sample_condition=d["condition"], contrast=d["contrast"],
            target_gene_ids=d["target_gene_ids"],
            n_permutations=49, seed=42, verbose=False,
        )

        report = ValidationReport()
        report.add_phase("covariate_adjusted", enrichment.to_dict())
        report.add_phase("label_permutation", {
            "stratified": perm_result.to_dict(),
            "free": {},
            "permutation_pvalue": perm_result.permutation_pvalue,
        })
        report.compute_verdict()

        assert report.verdict != "validated"
