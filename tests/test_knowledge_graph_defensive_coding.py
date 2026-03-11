"""Tests for knowledge graph defensive coding: bounded gene cache, CURIE parsing,
error propagation, condition delimiters, and discover_regulators LSP compliance.
"""

import logging
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: mock INDRA so tests don't require a live Neo4j connection
# ---------------------------------------------------------------------------

MOCK_TFS = ["TP53", "MYC", "JUN"]
MOCK_KINASES = ["AKT1", "MAPK1"]
MOCK_PHOSPHATASES = ["PTEN"]


@pytest.fixture(autouse=True)
def _mock_indra():
    """Patch hgnc_client on the already-imported cogex module."""
    mock_hgnc = MagicMock()
    mock_hgnc.tfs = MOCK_TFS
    mock_hgnc.kinases = MOCK_KINASES
    mock_hgnc.phosphatases = MOCK_PHOSPHATASES
    mock_hgnc.get_current_hgnc_id = MagicMock(return_value=None)

    import cliquefinder.knowledge.cogex as cogex_mod
    orig_hgnc = cogex_mod.hgnc_client
    orig_available = cogex_mod.INDRA_AVAILABLE
    cogex_mod.hgnc_client = mock_hgnc
    cogex_mod.INDRA_AVAILABLE = True
    yield mock_hgnc
    cogex_mod.hgnc_client = orig_hgnc
    cogex_mod.INDRA_AVAILABLE = orig_available


from cliquefinder.knowledge.cogex import (  # noqa: E402
    CoGExClient,
    INDRAModuleExtractor,
    INDRAEdge,
    GeneId,
)


class TestBoundedGeneCache:
    """Gene symbol lookup cache must not grow without bound."""

    def test_cache_is_ordered_dict(self, _mock_indra):
        """Cache uses OrderedDict for LRU semantics."""
        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)
        assert isinstance(ext._gene_cache, OrderedDict)

    def test_cache_maxsize_default(self, _mock_indra):
        """Default maxsize is 50,000."""
        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)
        assert ext._gene_cache_maxsize == 50_000

    def test_cache_evicts_oldest_when_full(self, _mock_indra):
        """When cache exceeds maxsize, oldest entry (LRU) is evicted."""
        _mock_indra.get_current_hgnc_id.side_effect = lambda name: name

        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)
        ext._gene_cache_maxsize = 5  # Small limit for testing

        # Fill cache with 5 entries
        for i in range(5):
            ext.resolve_gene_name(f"GENE{i}")
        assert len(ext._gene_cache) == 5
        assert "GENE0" in ext._gene_cache

        # Add one more -- should evict GENE0 (oldest)
        ext.resolve_gene_name("GENE5")
        assert len(ext._gene_cache) == 5
        assert "GENE0" not in ext._gene_cache
        assert "GENE5" in ext._gene_cache

    def test_cache_lru_access_refreshes(self, _mock_indra):
        """Accessing a cached entry moves it to the end (LRU)."""
        _mock_indra.get_current_hgnc_id.side_effect = lambda name: name

        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)
        ext._gene_cache_maxsize = 5

        # Fill cache
        for i in range(5):
            ext.resolve_gene_name(f"GENE{i}")

        # Access GENE0 (oldest) to refresh it
        ext.resolve_gene_name("GENE0")

        # Now add a new entry -- GENE1 should be evicted (now oldest)
        ext.resolve_gene_name("GENE_NEW")
        assert len(ext._gene_cache) == 5
        assert "GENE0" in ext._gene_cache  # Was refreshed, so kept
        assert "GENE1" not in ext._gene_cache  # Became oldest, evicted

    def test_clear_gene_cache(self, _mock_indra):
        """clear_gene_cache() empties the cache entirely."""
        _mock_indra.get_current_hgnc_id.side_effect = lambda name: name

        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)

        ext.resolve_gene_name("TP53")
        assert len(ext._gene_cache) == 1

        ext.clear_gene_cache()
        assert len(ext._gene_cache) == 0

    def test_cache_maxsize_never_exceeded(self, _mock_indra):
        """Cache never exceeds maxsize even with rapid insertions."""
        _mock_indra.get_current_hgnc_id.side_effect = lambda name: name

        mock_client = MagicMock(spec=CoGExClient)
        ext = INDRAModuleExtractor(mock_client)
        ext._gene_cache_maxsize = 10

        for i in range(100):
            ext.resolve_gene_name(f"G{i}")
            assert len(ext._gene_cache) <= 10


class TestCURIEParsingDefensive:
    """Malformed CURIE records must be skipped, not crash."""

    def _make_query_row(
        self, reg_id="hgnc:11998", reg_name="TP53",
        target_id="hgnc:1234", target_name="MDM2",
        stmt_type="IncreaseAmount", evidence=3,
        stmt_hash=12345, source_counts='{"reach": 2}'
    ):
        """Helper: produce a row tuple as returned by Neo4j query."""
        return (reg_id, reg_name, target_id, target_name,
                stmt_type, evidence, stmt_hash, source_counts)

    def test_malformed_regulator_curie_skipped(self, _mock_indra, caplog):
        """Missing ':' in regulator CURIE should skip row, not crash."""
        import cliquefinder.knowledge.cogex as cogex_mod

        good_row = self._make_query_row()
        bad_row = self._make_query_row(reg_id="MALFORMED_NO_COLON")

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [bad_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with caplog.at_level(logging.WARNING):
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                # Good row parsed, bad row skipped
                assert len(edges) == 1
                assert edges[0].target_name == "MDM2"
                assert "malformed CURIE" in caplog.text.lower() or "Skipping malformed" in caplog.text

    def test_malformed_target_curie_skipped(self, _mock_indra, caplog):
        """Missing ':' in target CURIE should skip row, not crash."""
        import cliquefinder.knowledge.cogex as cogex_mod

        bad_row = self._make_query_row(target_id="NO_COLON")
        good_row = self._make_query_row(target_id="hgnc:5555", target_name="BAX")

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [bad_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with caplog.at_level(logging.WARNING):
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                assert len(edges) == 1
                assert edges[0].target_name == "BAX"

    def test_none_curie_skipped(self, _mock_indra, caplog):
        """None value in CURIE field should skip row (AttributeError on split)."""
        import cliquefinder.knowledge.cogex as cogex_mod

        bad_row = self._make_query_row(reg_id=None)
        good_row = self._make_query_row()

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [bad_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with caplog.at_level(logging.WARNING):
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                assert len(edges) == 1

    def test_empty_string_curie_skipped(self, _mock_indra, caplog):
        """Empty string CURIE should skip row (no ':' to split on)."""
        import cliquefinder.knowledge.cogex as cogex_mod

        bad_row = self._make_query_row(reg_id="")
        good_row = self._make_query_row()

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [bad_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with caplog.at_level(logging.WARNING):
                    edges = client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                # Empty string splits to [""], which has only 1 element
                # so split(":", 1) returns [""] without a ":" -> ValueError
                assert len(edges) == 1

    def test_discover_regulators_malformed_curie_skipped(self, _mock_indra, caplog):
        """discover_regulators also skips malformed CURIE records."""
        import cliquefinder.knowledge.cogex as cogex_mod

        _mock_indra.get_current_hgnc_id.return_value = "11998"

        good_row = self._make_query_row(
            reg_id="hgnc:11998", reg_name="TP53",
            target_id="hgnc:1234", target_name="MDM2",
        )
        bad_row = self._make_query_row(
            reg_id="BROKEN", reg_name="BAD",
            target_id="hgnc:9999", target_name="XYZ",
        )

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [bad_row, good_row]

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with caplog.at_level(logging.WARNING):
                    result = client.discover_regulators(
                        gene_universe=["MDM2"],
                        min_evidence=2,
                        min_targets=1
                    )

                # Only the good row's regulator should appear
                assert "TP53" in result
                assert "Skipping malformed" in caplog.text


class TestNoDoubleWrappedRuntimeError:
    """RuntimeError from _execute_query must propagate without re-wrapping."""

    def test_runtime_error_not_double_wrapped(self, _mock_indra):
        """RuntimeError from _execute_query should propagate directly."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        # Simulate a connection error that _execute_query wraps as RuntimeError
        mock_client.query_tx.side_effect = Exception("Connection timeout")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError) as exc_info:
                    client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                # Should be the _execute_query RuntimeError, not a wrapping one
                msg = str(exc_info.value)
                assert "Query failed after" in msg
                # Should NOT contain "Query failed: Query failed after" (double-wrap)
                assert msg.count("Query failed") == 1

    def test_discover_regulators_runtime_error_not_double_wrapped(self, _mock_indra):
        """discover_regulators also avoids double-wrapping RuntimeError."""
        import cliquefinder.knowledge.cogex as cogex_mod

        _mock_indra.get_current_hgnc_id.return_value = "11998"

        mock_client = MagicMock()
        mock_client.query_tx.side_effect = Exception("Connection reset by peer")

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError) as exc_info:
                    client.discover_regulators(
                        gene_universe=["TP53", "MDM2"],
                        min_evidence=2,
                        min_targets=1
                    )

                msg = str(exc_info.value)
                # Only one level of wrapping
                assert msg.count("failed") <= 2  # "Query failed after N attempts: ..."

    def test_non_runtime_error_still_wrapped(self, _mock_indra):
        """Non-RuntimeError (e.g., ValueError in parsing) should be wrapped."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        # Return valid query results but with data that causes a non-RuntimeError
        # in the parsing loop (e.g., index out of bounds)
        mock_client.query_tx.return_value = [(1,)]  # Row too short -> IndexError

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError, match="Query failed"):
                    client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

    def test_original_exception_preserved_in_chain(self, _mock_indra):
        """For non-RuntimeError wrapping, __cause__ preserves the original."""
        import cliquefinder.knowledge.cogex as cogex_mod

        mock_client = MagicMock()
        mock_client.query_tx.return_value = [(1,)]  # Causes IndexError

        with patch.object(cogex_mod, "Neo4jClient", return_value=mock_client):
            with patch.object(cogex_mod, "norm_id", return_value="hgnc:11998"):
                client = CoGExClient(url="bolt://fake:7687", user="neo4j", password="secret")

                with pytest.raises(RuntimeError) as exc_info:
                    client.get_downstream_targets(
                        regulator=("HGNC", "11998"),
                        stmt_types=["IncreaseAmount"],
                        min_evidence=2
                    )

                # __cause__ should be set (from 'raise ... from e')
                assert exc_info.value.__cause__ is not None


class TestConditionDelimiter:
    """Condition strings use '||' delimiter to handle underscored metadata."""

    @pytest.fixture
    def matrix_with_underscore_metadata(self):
        """Create BioMatrix with metadata values containing underscores."""
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag

        n_genes, n_samples = 50, 20
        data = np.random.default_rng(42).standard_normal((n_genes, n_samples))
        feature_ids = pd.Index([f"GENE{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        metadata = pd.DataFrame({
            'phenotype': ['late_onset'] * 10 + ['early_onset'] * 10,
            'Sex': ['Male'] * 5 + ['Female'] * 5 + ['Male'] * 5 + ['Female'] * 5,
        }, index=sample_ids)
        quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=int)
        return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)

    @pytest.fixture
    def simple_matrix(self):
        """Create simple BioMatrix without underscore metadata."""
        from cliquefinder.core.biomatrix import BioMatrix
        from cliquefinder.core.quality import QualityFlag

        n_genes, n_samples = 50, 20
        data = np.random.default_rng(42).standard_normal((n_genes, n_samples))
        feature_ids = pd.Index([f"GENE{i}" for i in range(n_genes)])
        sample_ids = pd.Index([f"S{i}" for i in range(n_samples)])
        metadata = pd.DataFrame({
            'phenotype': ['CASE'] * 10 + ['CTRL'] * 10,
            'Sex': ['Male'] * 5 + ['Female'] * 5 + ['Male'] * 5 + ['Female'] * 5,
        }, index=sample_ids)
        quality_flags = np.full((n_genes, n_samples), QualityFlag.ORIGINAL, dtype=int)
        return BioMatrix(data, feature_ids, sample_ids, metadata, quality_flags)

    def test_delimiter_constant_is_pipe(self):
        """CONDITION_DELIMITER is '||'."""
        from cliquefinder.knowledge.clique_validator import CONDITION_DELIMITER
        assert CONDITION_DELIMITER == "||"

    def test_conditions_use_pipe_delimiter(self, simple_matrix):
        """get_available_conditions uses '||' not '_'."""
        from cliquefinder.knowledge.clique_validator import CliqueValidator

        validator = CliqueValidator(simple_matrix, stratify_by=['phenotype', 'Sex'])
        conditions = validator.get_available_conditions()

        # Conditions should use '||' delimiter
        for cond in conditions:
            if cond != 'all':
                assert '||' in cond, f"Expected '||' in condition '{cond}'"
                assert '_' not in cond or cond.count('||') > 0

    def test_underscore_metadata_roundtrips(self, matrix_with_underscore_metadata):
        """Conditions with underscored values parse correctly."""
        from cliquefinder.knowledge.clique_validator import CliqueValidator

        validator = CliqueValidator(
            matrix_with_underscore_metadata,
            stratify_by=['phenotype', 'Sex'],
            min_samples=3,
        )
        conditions = validator.get_available_conditions()

        # Should have 4 conditions
        assert len(conditions) == 4

        # 'late_onset' should be kept intact (not split at underscore)
        late_onset_conds = [c for c in conditions if 'late_onset' in c]
        assert len(late_onset_conds) == 2

        # Each condition should have exactly 2 parts when split on '||'
        for cond in conditions:
            parts = cond.split('||')
            assert len(parts) == 2, (
                f"Condition '{cond}' should split into 2 parts on '||', "
                f"got {len(parts)}: {parts}"
            )

    def test_mask_correct_for_underscore_metadata(self, matrix_with_underscore_metadata):
        """Condition masks correctly select samples with underscored values."""
        from cliquefinder.knowledge.clique_validator import CliqueValidator

        validator = CliqueValidator(
            matrix_with_underscore_metadata,
            stratify_by=['phenotype', 'Sex'],
            min_samples=3,
        )

        # Get a condition with underscore in value
        conditions = validator.get_available_conditions()
        late_male = [c for c in conditions if 'late_onset' in c and 'Male' in c]
        assert len(late_male) == 1

        mask = validator._get_condition_mask(late_male[0])
        # Should select 5 samples (late_onset + Male)
        assert mask.sum() == 5

    def test_old_underscore_delimiter_raises(self, simple_matrix):
        """Using '_' as delimiter in condition string raises ValueError."""
        from cliquefinder.knowledge.clique_validator import CliqueValidator

        validator = CliqueValidator(
            simple_matrix,
            stratify_by=['phenotype', 'Sex'],
            min_samples=3,
        )

        # Old-style condition with '_' should fail to parse (wrong number of parts)
        # because 'CASE_Male' splits on '||' -> 1 part, but 2 expected
        with pytest.raises(ValueError, match="has 1 parts but stratification requires 2"):
            validator._compute_condition_mask_internal("CASE_Male")

    def test_no_stratification_returns_all(self, simple_matrix):
        """Without stratification, 'all' is returned (no delimiter involved)."""
        from cliquefinder.knowledge.clique_validator import CliqueValidator

        validator = CliqueValidator(simple_matrix, stratify_by=[])
        assert validator.get_available_conditions() == ['all']


class TestDiscoverRegulatorsLSP:
    """INDRAKnowledgeSource.discover_regulators signature matches parent."""

    def test_parameter_order_matches_parent(self):
        """Child signature has same positional params as parent, extras at end."""
        import inspect
        from cliquefinder.knowledge.base import KnowledgeSource
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        parent_sig = inspect.signature(KnowledgeSource.discover_regulators)
        child_sig = inspect.signature(INDRAKnowledgeSource.discover_regulators)

        parent_params = list(parent_sig.parameters.keys())
        child_params = list(child_sig.parameters.keys())

        # Parent params (excluding 'self') should be a prefix of child params
        parent_params_no_self = [p for p in parent_params if p != 'self']
        child_params_no_self = [p for p in child_params if p != 'self']

        for i, parent_param in enumerate(parent_params_no_self):
            assert child_params_no_self[i] == parent_param, (
                f"Parameter at position {i} differs: parent has '{parent_param}', "
                f"child has '{child_params_no_self[i]}'"
            )

    def test_max_targets_is_last(self):
        """max_targets (child-only param) is after all parent params."""
        import inspect
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        sig = inspect.signature(INDRAKnowledgeSource.discover_regulators)
        params = list(sig.parameters.keys())

        assert 'max_targets' in params
        # max_targets should be after min_evidence
        assert params.index('max_targets') > params.index('min_evidence')

    def test_parent_positional_call_works(self):
        """Calling with parent's positional args doesn't misassign max_targets."""
        import inspect
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        sig = inspect.signature(INDRAKnowledgeSource.discover_regulators)
        params = sig.parameters

        # Simulate parent-style positional call:
        # discover_regulators(target_universe, min_targets, relationship_types, min_evidence)
        param_list = list(params.keys())
        # 'self', 'target_universe', 'min_targets', 'relationship_types', 'min_evidence', 'max_targets'
        assert param_list[0] == 'self'
        assert param_list[1] == 'target_universe'
        assert param_list[2] == 'min_targets'
        assert param_list[3] == 'relationship_types'
        assert param_list[4] == 'min_evidence'
        assert param_list[5] == 'max_targets'

    def test_max_targets_has_default_none(self):
        """max_targets has default=None so it's optional."""
        import inspect
        from cliquefinder.knowledge.indra_source import INDRAKnowledgeSource

        sig = inspect.signature(INDRAKnowledgeSource.discover_regulators)
        max_targets_param = sig.parameters['max_targets']
        assert max_targets_param.default is None
