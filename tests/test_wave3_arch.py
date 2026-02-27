"""
Tests for Wave 3 Architecture Fixes: ARCH-9, ARCH-14, ARCH-17.

ARCH-9:  Frozen dataclass immutability enforcement
ARCH-14: Chunked Neo4j queries with max_results
ARCH-17: Single-contrast verdict over-conservative downgrade fix
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# =============================================================================
# ARCH-9: Frozen Dataclasses with Mutable Fields
# =============================================================================


class TestRotationPrecomputedImmutability:
    """Test that RotationPrecomputed enforces true immutability."""

    def _make_precomputed(self):
        from cliquefinder.stats.rotation import RotationPrecomputed

        Q2 = np.random.default_rng(42).standard_normal((10, 5))
        return RotationPrecomputed(
            Q2=Q2,
            df_residual=4,
            contrast_name="test",
        )

    def test_q2_not_writeable(self):
        """NDArray Q2 field should not be writeable after construction."""
        pc = self._make_precomputed()
        assert not pc.Q2.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            pc.Q2[0, 0] = 999.0

    def test_q2_is_copy(self):
        """Q2 should be a copy of the original array, not a view."""
        from cliquefinder.stats.rotation import RotationPrecomputed

        original = np.ones((4, 3))
        pc = RotationPrecomputed(Q2=original, df_residual=2, contrast_name="c")
        # Mutate the original — the dataclass copy should be unaffected
        original[0, 0] = -999.0
        assert pc.Q2[0, 0] == 1.0

    def test_frozen_attribute_reassignment_raises(self):
        """Frozen dataclass should still prevent attribute reassignment."""
        pc = self._make_precomputed()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.Q2 = np.zeros((10, 5))

    def test_construction_still_works(self):
        """Ensure normal construction patterns are not broken."""
        pc = self._make_precomputed()
        assert pc.df_residual == 4
        assert pc.contrast_name == "test"
        assert pc.Q2.shape == (10, 5)
        assert pc.residual_dims == 5


class TestGeneEffectsImmutability:
    """Test that GeneEffects enforces true immutability."""

    def _make_effects(self, with_moderated=False):
        from cliquefinder.stats.rotation import GeneEffects

        rng = np.random.default_rng(42)
        U = rng.standard_normal((5, 4))
        rho_sq = rng.random(5)
        sv = rng.random(5)
        mv = rng.random(5) if with_moderated else None

        return GeneEffects(
            U=U,
            rho_sq=rho_sq,
            gene_ids=["A", "B", "C", "D", "E"],
            sample_variances=sv,
            moderated_variances=mv,
            df_total=10.0 if with_moderated else None,
        )

    def test_ndarrays_not_writeable(self):
        """All NDArray fields should be read-only."""
        effects = self._make_effects(with_moderated=True)
        for attr in ("U", "rho_sq", "sample_variances", "moderated_variances"):
            arr = getattr(effects, attr)
            assert not arr.flags.writeable, f"{attr} should not be writeable"
            with pytest.raises(ValueError, match="read-only"):
                arr.flat[0] = 999.0

    def test_gene_ids_is_tuple(self):
        """gene_ids should be converted from list to tuple."""
        effects = self._make_effects()
        assert isinstance(effects.gene_ids, tuple)
        assert effects.gene_ids == ("A", "B", "C", "D", "E")

    def test_gene_ids_immutable(self):
        """Tuple gene_ids cannot be mutated."""
        effects = self._make_effects()
        with pytest.raises(TypeError):
            effects.gene_ids[0] = "X"  # type: ignore[index]

    def test_frozen_attribute_reassignment(self):
        """Frozen dataclass prevents attribute reassignment."""
        effects = self._make_effects()
        with pytest.raises(dataclasses.FrozenInstanceError):
            effects.U = np.zeros((5, 4))

    def test_construction_still_works(self):
        """Normal construction and property access works."""
        effects = self._make_effects()
        assert effects.n_genes == 5
        assert effects.residual_dims == 4


class TestTestResultImmutability:
    """Test that TestResult (permutation_framework) enforces immutability."""

    def _make_result(self, additional=None):
        from cliquefinder.stats.permutation_framework import TestResult

        return TestResult(
            feature_set_id="test",
            effect_size=1.5,
            test_statistic=2.3,
            p_value=0.01,
            n_test=10,
            n_reference=10,
            additional=additional or {},
        )

    def test_additional_is_mapping_proxy(self):
        """additional dict should be wrapped in MappingProxyType."""
        result = self._make_result({"key": "value"})
        assert isinstance(result.additional, MappingProxyType)
        assert result.additional["key"] == "value"

    def test_additional_not_mutable(self):
        """Cannot add keys to the additional MappingProxyType."""
        result = self._make_result({"key": "value"})
        with pytest.raises(TypeError):
            result.additional["new_key"] = "new_value"  # type: ignore[index]

    def test_empty_additional(self):
        """Empty additional dict should still become MappingProxyType."""
        result = self._make_result()
        assert isinstance(result.additional, MappingProxyType)
        assert len(result.additional) == 0

    def test_frozen_attribute_reassignment(self):
        """Frozen dataclass prevents attribute reassignment."""
        result = self._make_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.additional = {}  # type: ignore[misc]

    def test_construction_and_access(self):
        """Normal construction and field access works."""
        result = self._make_result({"model": "mixed"})
        assert result.feature_set_id == "test"
        assert result.additional["model"] == "mixed"


class TestUnifiedCliqueResultImmutability:
    """Test that UnifiedCliqueResult enforces immutability on method_metadata."""

    def _make_result(self, metadata=None):
        from cliquefinder.stats.method_comparison import UnifiedCliqueResult, MethodName

        return UnifiedCliqueResult(
            clique_id="TP53",
            method=MethodName.OLS,
            effect_size=1.5,
            effect_size_se=0.3,
            p_value=0.001,
            statistic_value=5.0,
            statistic_type="t",
            degrees_of_freedom=45.0,
            n_proteins=25,
            n_proteins_found=20,
            method_metadata=metadata or {},
        )

    def test_method_metadata_is_mapping_proxy(self):
        """method_metadata should be wrapped in MappingProxyType."""
        result = self._make_result({"converged": True})
        assert isinstance(result.method_metadata, MappingProxyType)
        assert result.method_metadata["converged"] is True

    def test_method_metadata_not_mutable(self):
        """Cannot add keys to method_metadata."""
        result = self._make_result({"converged": True})
        with pytest.raises(TypeError):
            result.method_metadata["new"] = "value"  # type: ignore[index]

    def test_to_dict_still_works(self):
        """to_dict() should iterate over MappingProxyType without issues."""
        result = self._make_result({"model_type": "ols", "converged": True})
        d = result.to_dict()
        assert d["meta_model_type"] == "ols"
        assert d["meta_converged"] is True
        assert d["clique_id"] == "TP53"

    def test_is_valid_still_works(self):
        """is_valid property should work correctly."""
        result = self._make_result()
        assert result.is_valid


class TestPreparedCliqueExperimentImmutability:
    """Test that PreparedCliqueExperiment enforces immutability."""

    def _make_experiment(self):
        from cliquefinder.stats.method_comparison import PreparedCliqueExperiment

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 5))
        return PreparedCliqueExperiment(
            data=data,
            feature_ids=tuple(f"gene_{i}" for i in range(10)),
            feature_to_idx={f"gene_{i}": i for i in range(10)},
            sample_metadata=None,
            condition_column="cond",
            subject_column=None,
            conditions=("A", "B"),
            n_samples=5,
            cliques=(),
            clique_to_feature_indices={"clq1": (0, 1, 2)},
            symbol_to_feature={"SYM1": "gene_0"},
            contrast=("A", "B"),
            contrast_name="A_vs_B",
            preprocessing_params={"method": "median"},
            creation_timestamp="2026-01-01T00:00:00",
        )

    def test_data_not_writeable(self):
        """data NDArray should not be writeable."""
        exp = self._make_experiment()
        assert not exp.data.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            exp.data[0, 0] = 999.0

    def test_dicts_are_mapping_proxy(self):
        """Dict fields should be MappingProxyType."""
        exp = self._make_experiment()
        for attr in ("feature_to_idx", "clique_to_feature_indices",
                     "symbol_to_feature", "preprocessing_params"):
            val = getattr(exp, attr)
            assert isinstance(val, MappingProxyType), f"{attr} should be MappingProxyType"

    def test_dicts_not_mutable(self):
        """Cannot modify dict fields."""
        exp = self._make_experiment()
        with pytest.raises(TypeError):
            exp.feature_to_idx["new_gene"] = 99  # type: ignore[index]
        with pytest.raises(TypeError):
            exp.clique_to_feature_indices["new_clq"] = (5,)  # type: ignore[index]

    def test_frozen_attribute_reassignment(self):
        """Frozen class prevents reassignment."""
        exp = self._make_experiment()
        with pytest.raises(dataclasses.FrozenInstanceError):
            exp.data = np.zeros((10, 5))

    def test_n_features_property_works(self):
        """Properties that depend on data still work."""
        exp = self._make_experiment()
        assert exp.n_features == 10
        assert exp.n_cliques == 0  # cliques tuple is empty


# =============================================================================
# ARCH-14: Unbounded Neo4j Result Sets — Chunked Queries
# =============================================================================


class TestDiscoverRegulatorsChunking:
    """Test that discover_regulators() chunks large CURIE lists."""

    def _make_client(self):
        """Create a CoGExClient with mocked Neo4j connection."""
        from cliquefinder.knowledge import cogex as cogex_mod

        client = cogex_mod.CoGExClient.__new__(cogex_mod.CoGExClient)
        client._client = MagicMock()
        client._url = "bolt://fake:7687"
        return client

    def _make_mock_row(self, reg_name, target_name, target_id_suffix):
        """Create a mock Neo4j result row."""
        return [
            f"hgnc:{reg_name}",     # reg_id
            reg_name,               # reg_name
            f"hgnc:{target_id_suffix}",  # target_id
            target_name,            # target_name
            "IncreaseAmount",       # stmt_type
            5,                      # evidence_count
            "hash123",              # stmt_hash
            '{"reach": 3}',         # source_counts
        ]

    @patch("cliquefinder.knowledge.cogex.INDRA_AVAILABLE", True)
    @patch("cliquefinder.knowledge.cogex.hgnc_client")
    @patch("cliquefinder.knowledge.cogex.norm_id")
    def test_small_query_single_chunk(self, mock_norm_id, mock_hgnc):
        """Queries with < CHUNK_SIZE CURIEs should use a single query call."""
        client = self._make_client()

        # Set up gene resolution: 100 genes
        gene_names = [f"GENE{i}" for i in range(100)]
        mock_hgnc.get_current_hgnc_id.side_effect = lambda name: f"id_{name}"
        mock_norm_id.side_effect = lambda ns, id_: f"hgnc:{id_}"

        # Return 5+ targets for regulator "REG1"
        mock_neo4j = client._client
        rows = [self._make_mock_row("REG1", f"T{i}", f"t{i}") for i in range(10)]
        mock_neo4j.query_tx.return_value = rows

        result = client.discover_regulators(
            gene_universe=gene_names,
            min_targets=5,
        )

        # Should be exactly 1 query call (no chunking)
        assert mock_neo4j.query_tx.call_count == 1
        assert "REG1" in result
        assert len(result["REG1"]) == 10

    @patch("cliquefinder.knowledge.cogex.INDRA_AVAILABLE", True)
    @patch("cliquefinder.knowledge.cogex.hgnc_client")
    @patch("cliquefinder.knowledge.cogex.norm_id")
    def test_large_query_chunked(self, mock_norm_id, mock_hgnc):
        """Queries with > CHUNK_SIZE CURIEs should produce multiple query calls."""
        client = self._make_client()

        # Use a smaller chunk size for testing
        client.CURIE_CHUNK_SIZE = 50

        # Set up gene resolution: 120 genes -> 3 chunks of 50/50/20
        gene_names = [f"GENE{i}" for i in range(120)]
        mock_hgnc.get_current_hgnc_id.side_effect = lambda name: f"id_{name}"
        mock_norm_id.side_effect = lambda ns, id_: f"hgnc:{id_}"

        # Return 10 targets for regulator "REG1" per chunk call
        rows = [self._make_mock_row("REG1", f"T{i}", f"t{i}") for i in range(10)]
        mock_neo4j = client._client
        mock_neo4j.query_tx.return_value = rows

        result = client.discover_regulators(
            gene_universe=gene_names,
            min_targets=5,
        )

        # Should be 3 query calls: ceil(120 / 50) = 3
        assert mock_neo4j.query_tx.call_count == 3
        # Check that each chunk was the right size
        call_args_list = mock_neo4j.query_tx.call_args_list
        chunk_sizes = [len(c.kwargs.get("target_ids", c[1].get("target_ids", [])))
                       if c.kwargs else len(c[1].get("target_ids", []))
                       for c in call_args_list]
        # query_tx is called with keyword args
        for c in call_args_list:
            # Get target_ids from keyword arguments
            if c.kwargs:
                tid = c.kwargs.get("target_ids", [])
            else:
                tid = []
            assert len(tid) <= 50

    @patch("cliquefinder.knowledge.cogex.INDRA_AVAILABLE", True)
    @patch("cliquefinder.knowledge.cogex.hgnc_client")
    @patch("cliquefinder.knowledge.cogex.norm_id")
    def test_max_results_warning(self, mock_norm_id, mock_hgnc):
        """Warning should be emitted when max_results is reached."""
        client = self._make_client()
        client.CURIE_CHUNK_SIZE = 50

        gene_names = [f"GENE{i}" for i in range(120)]
        mock_hgnc.get_current_hgnc_id.side_effect = lambda name: f"id_{name}"
        mock_norm_id.side_effect = lambda ns, id_: f"hgnc:{id_}"

        # Return many rows per chunk
        rows = [self._make_mock_row("REG1", f"T{i}", f"t{i}") for i in range(100)]
        mock_neo4j = client._client
        mock_neo4j.query_tx.return_value = rows

        import logging
        with patch.object(logging.getLogger("cliquefinder.knowledge.cogex"),
                          "warning") as mock_warn:
            result = client.discover_regulators(
                gene_universe=gene_names,
                min_targets=5,
                max_results=150,  # Hit limit after 2 chunks (100 per chunk)
            )
            # Warning should be emitted
            mock_warn.assert_called_once()
            assert "max_results" in mock_warn.call_args[0][0]

        # Should have stopped early (only 2 calls instead of 3)
        assert mock_neo4j.query_tx.call_count == 2

    @patch("cliquefinder.knowledge.cogex.INDRA_AVAILABLE", True)
    @patch("cliquefinder.knowledge.cogex.hgnc_client")
    def test_empty_curie_list(self, mock_hgnc):
        """Empty gene universe (no resolvable genes) returns empty dict."""
        client = self._make_client()

        # All gene name resolutions fail
        mock_hgnc.get_current_hgnc_id.return_value = None

        result = client.discover_regulators(
            gene_universe=["FAKEGENE1", "FAKEGENE2"],
            min_targets=1,
        )

        assert result == {}
        # No queries should have been made
        client._client.query_tx.assert_not_called()


# =============================================================================
# ARCH-17: Verdict Over-Conservative Downgrade for Single-Contrast
# =============================================================================


class TestVerdictSingleContrast:
    """Test that single-contrast datasets are not penalised for skipped phases."""

    def _make_report_with_gates_passing(self, alpha=0.05):
        """Create a report where both mandatory gates pass."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        # Phase 1: covariate-adjusted passes
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.001,
            "status": "completed",
        })
        # Phase 3: label permutation passes
        report.add_phase("label_permutation", {
            "permutation_pvalue": 0.002,
            "status": "completed",
        })
        return report

    def test_single_contrast_gates_pass_phase5_passes_validated(self):
        """
        Single-contrast: Phase 2 skipped, gates pass, Phase 5 passes.
        Should be 'validated', not 'inconclusive'.
        """
        report = self._make_report_with_gates_passing()
        # Phase 2 (specificity) NOT added — simulates single-contrast skip
        # Phase 5 (negative controls) passes
        report.add_phase("negative_controls", {
            "target_percentile": 5.0,
            "fpr": 0.03,
            "status": "completed",
        })
        report.compute_verdict()
        assert report.verdict == "validated"
        assert "Supplementary: 1/1 pass" in report.summary

    def test_single_contrast_gates_pass_no_supplementary_validated(self):
        """
        Single-contrast: both gates pass, no supplementary phases ran.
        Should be 'validated' (not inconclusive).
        """
        report = self._make_report_with_gates_passing()
        # No supplementary phases at all
        report.compute_verdict()
        assert report.verdict == "validated"
        assert "No supplementary phases ran" in report.summary

    def test_all_supplementary_fail_inconclusive(self):
        """
        Both gates pass but all supplementary phases fail.
        Should be 'inconclusive'.
        """
        report = self._make_report_with_gates_passing()
        # Phase 2: specificity inconclusive (counts as fail)
        report.add_phase("specificity", {
            "specificity_label": "inconclusive",
            "status": "completed",
        })
        # Phase 4: matched fails
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.5,
            "n_matched": 10,
            "status": "completed",
        })
        # Phase 5: negative controls fail
        report.add_phase("negative_controls", {
            "target_percentile": 80.0,
            "fpr": 0.5,
            "status": "completed",
        })
        report.compute_verdict()
        assert report.verdict == "inconclusive"
        assert "supplementary phases fail" in report.summary

    def test_mixed_supplementary_some_pass_validated(self):
        """
        Gates pass, some supplementary pass and some fail.
        Should be 'validated'.
        """
        report = self._make_report_with_gates_passing()
        # Phase 2: specific (pass)
        report.add_phase("specificity", {
            "specificity_label": "specific",
            "status": "completed",
        })
        # Phase 4: matched fails
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.5,
            "n_matched": 10,
            "status": "completed",
        })
        # Phase 5: negative controls pass
        report.add_phase("negative_controls", {
            "target_percentile": 5.0,
            "fpr": 0.03,
            "status": "completed",
        })
        report.compute_verdict()
        assert report.verdict == "validated"
        assert "Supplementary: 2/3 pass" in report.summary

    def test_both_gates_fail_refuted(self):
        """Both mandatory gates fail -> 'refuted' (unchanged behaviour)."""
        from cliquefinder.stats.validation_report import ValidationReport

        report = ValidationReport()
        report.add_phase("covariate_adjusted", {
            "empirical_pvalue": 0.5,
            "status": "completed",
        })
        report.add_phase("label_permutation", {
            "permutation_pvalue": 0.6,
            "status": "completed",
        })
        report.compute_verdict()
        assert report.verdict == "refuted"

    def test_phase2_skipped_phase4_and_phase5_both_pass(self):
        """
        Phase 2 skipped (single-contrast), Phase 4 and 5 both pass.
        Verdict should be 'validated' with 2/2 supplementary.
        """
        report = self._make_report_with_gates_passing()
        report.add_phase("matched_reanalysis", {
            "empirical_pvalue": 0.001,
            "n_matched": 20,
            "status": "completed",
        })
        report.add_phase("negative_controls", {
            "target_percentile": 3.0,
            "fpr": 0.02,
            "status": "completed",
        })
        report.compute_verdict()
        assert report.verdict == "validated"
        assert "Supplementary: 2/2 pass" in report.summary

    def test_failed_phase_not_counted_as_supplementary(self):
        """
        A phase with status='failed' should not count as supplementary.
        """
        report = self._make_report_with_gates_passing()
        # Phase 2 failed (error during execution)
        report.add_phase("specificity", {
            "status": "failed",
            "error": "Not enough contrasts",
        })
        report.compute_verdict()
        # No supplementary phases actually ran -> validated
        assert report.verdict == "validated"
        assert "No supplementary phases ran" in report.summary
