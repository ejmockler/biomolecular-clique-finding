"""Tests for TargetSet serialization round-trip and validation pipeline integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliquefinder.stats.target_set import TargetSet


class TestTargetSetRoundTrip:
    """Serialize → deserialize produces identical objects."""

    def _make_target_set(self) -> TargetSet:
        ts = TargetSet.from_query(
            targets_in_data={"TP53": "P04637", "MDM2": "Q00987", "MAPT": "P10636"},
            gene_symbol="C9orf72",
            min_evidence=3,
            n_hops=1,
            n_indra_edges_raw=115,
        )
        ts.attach_adjacency(
            adjacency={
                "REG1": ["TP53", "MDM2", "MAPT"],
                "REG2": ["TP53", "MAPT"],
            },
            min_evidence=3,
            min_targets=2,
        )
        return ts

    def test_round_trip_targets(self, tmp_path: Path):
        ts = self._make_target_set()
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.targets == ts.targets
        assert loaded.gene_symbol == ts.gene_symbol
        assert loaded.min_evidence == ts.min_evidence
        assert loaded.n_hops == ts.n_hops
        assert loaded.min_intermediaries == ts.min_intermediaries
        assert loaded.n_indra_edges_raw == ts.n_indra_edges_raw

    def test_round_trip_adjacency(self, tmp_path: Path):
        ts = self._make_target_set()
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.adjacency == ts.adjacency
        assert loaded.adjacency_min_evidence == 3
        assert loaded.adjacency_min_targets == 2

    def test_round_trip_preserves_timestamp(self, tmp_path: Path):
        ts = self._make_target_set()
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.query_timestamp == ts.query_timestamp

    def test_feature_ids_property(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1", "B": "P2"},
            gene_symbol="X",
            min_evidence=1,
            n_hops=1,
        )
        assert set(ts.feature_ids) == {"P1", "P2"}
        assert set(ts.symbols) == {"A", "B"}

    def test_no_adjacency_round_trips(self, tmp_path: Path):
        ts = TargetSet.from_query(
            targets_in_data={"TP53": "P04637"},
            gene_symbol="C9orf72",
            min_evidence=1,
            n_hops=1,
        )
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.adjacency == {}
        assert loaded.adjacency_min_evidence is None

    def test_format_version_check(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "format_version": 99,
            "targets": {},
            "gene_symbol": "X",
            "min_evidence": 1,
            "n_hops": 1,
        }))
        with pytest.raises(ValueError, match="Unsupported target set format version 99"):
            TargetSet.load(path)

    def test_valid_json_on_disk(self, tmp_path: Path):
        ts = self._make_target_set()
        path = ts.save(tmp_path / "targets.json")
        blob = json.loads(path.read_text())

        assert blob["format_version"] == 1
        assert blob["n_targets"] == 3
        assert "TP53" in blob["targets"]
        assert blob["adjacency"]["n_regulators"] == 2

    def test_attach_adjacency_sorts_targets(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1"},
            gene_symbol="X",
            min_evidence=1,
            n_hops=1,
        )
        ts.attach_adjacency(
            adjacency={"REG": ["Z", "A", "M"]},
            min_evidence=1,
        )
        assert ts.adjacency["REG"] == ["A", "M", "Z"]


class TestTargetSetInValidation:
    """Integration: --target-set flag skips INDRA re-query."""

    def test_loaded_target_set_provides_feature_ids(self, tmp_path: Path):
        """When loaded, .feature_ids gives the UniProt IDs for ROAST."""
        ts = TargetSet.from_query(
            targets_in_data={"SOD1": "P00441", "FUS": "P35637"},
            gene_symbol="C9orf72",
            min_evidence=3,
            n_hops=1,
        )
        path = ts.save(tmp_path / "indra_targets.json")
        loaded = TargetSet.load(path)

        target_gene_ids = list(loaded.targets.values())
        assert set(target_gene_ids) == {"P00441", "P35637"}

    def test_loaded_adjacency_used_for_graph_permutation(self, tmp_path: Path):
        """Adjacency from file should be usable without INDRA query."""
        ts = TargetSet.from_query(
            targets_in_data={"SOD1": "P00441"},
            gene_symbol="C9orf72",
            min_evidence=3,
            n_hops=1,
        )
        ts.attach_adjacency(
            adjacency={"TF1": ["SOD1", "FUS"], "TF2": ["SOD1", "MAPT"]},
            min_evidence=3,
            min_targets=2,
        )
        path = ts.save(tmp_path / "indra_targets.json")
        loaded = TargetSet.load(path)

        assert len(loaded.adjacency) == 2
        assert "TF1" in loaded.adjacency
        assert loaded.adjacency["TF1"] == ["FUS", "SOD1"]

    def test_missing_feature_ids_detected(self, tmp_path: Path):
        """Feature IDs not in proteomics should be flagged."""
        ts = TargetSet.from_query(
            targets_in_data={"FAKE": "PXXXXX"},
            gene_symbol="C9orf72",
            min_evidence=1,
            n_hops=1,
        )
        path = ts.save(tmp_path / "indra_targets.json")
        loaded = TargetSet.load(path)

        proteomics_ids = {"P00441", "P35637"}
        missing = [fid for fid in loaded.targets.values()
                   if fid not in proteomics_ids]
        assert missing == ["PXXXXX"]


class TestTargetSet2Hop:
    """Verify 2-hop metadata preserved."""

    def test_min_intermediaries_preserved(self, tmp_path: Path):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1", "B": "P2"},
            gene_symbol="C9orf72",
            min_evidence=3,
            n_hops=2,
            min_intermediaries=4,
        )
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.n_hops == 2
        assert loaded.min_intermediaries == 4

    def test_repr(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1", "B": "P2"},
            gene_symbol="C9orf72",
            min_evidence=3,
            n_hops=1,
        )
        r = repr(ts)
        assert "C9orf72" in r
        assert "2 targets" in r
        assert "min_ev=3" in r


class TestEdgeMetadataV2:
    """Tests for v2 format: per-edge metadata, source filtering, concordance."""

    def _make_edge_metadata(self) -> dict[str, list[dict]]:
        return {
            "SOD1": [{"regulation_type": "activation", "sources": ["reach", "sparser"], "evidence_count": 2}],
            "FUS": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
            "MAPT": [
                {"regulation_type": "activation", "sources": ["reach"], "evidence_count": 2},
                {"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1},
            ],
            "TARDBP": [{"regulation_type": "activation", "sources": ["reach", "sparser", "signor"], "evidence_count": 19}],
        }

    def _make_v2_target_set(self) -> TargetSet:
        return TargetSet.from_query(
            targets_in_data={"SOD1": "P00441", "FUS": "P35637", "MAPT": "P10636", "TARDBP": "Q13148"},
            gene_symbol="C9orf72",
            min_evidence=1,
            n_hops=1,
            edge_metadata=self._make_edge_metadata(),
            min_sources=None,
        )

    def test_v2_round_trip(self, tmp_path: Path):
        ts = self._make_v2_target_set()
        path = ts.save(tmp_path / "targets.json")
        loaded = TargetSet.load(path)

        assert loaded.edge_metadata == ts.edge_metadata
        assert len(loaded.edge_metadata["SOD1"]) == 1
        assert len(loaded.edge_metadata["MAPT"]) == 2

    def test_v2_format_version_on_disk(self, tmp_path: Path):
        ts = self._make_v2_target_set()
        path = ts.save(tmp_path / "targets.json")
        blob = json.loads(path.read_text())
        assert blob["format_version"] == 2

    def test_v1_loads_with_empty_edge_metadata(self, tmp_path: Path):
        """v1 files load with empty edge_metadata (backward compat)."""
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1"},
            gene_symbol="X",
            min_evidence=1,
            n_hops=1,
        )
        path = ts.save(tmp_path / "v1.json")
        blob = json.loads(path.read_text())
        assert blob["format_version"] == 1  # no edge_metadata → v1

        loaded = TargetSet.load(path)
        assert loaded.edge_metadata == {}

    def test_get_unambiguous_targets_lof(self):
        ts = self._make_v2_target_set()
        unambiguous = ts.get_unambiguous_targets(loss_of_function=True)

        assert unambiguous["SOD1"] == "predicted_down"  # activation → down when LoF
        assert unambiguous["FUS"] == "predicted_up"  # repression → up when LoF
        assert unambiguous["TARDBP"] == "predicted_down"
        assert "MAPT" not in unambiguous  # mixed → excluded

    def test_get_unambiguous_targets_gof(self):
        ts = self._make_v2_target_set()
        unambiguous = ts.get_unambiguous_targets(loss_of_function=False)

        assert unambiguous["SOD1"] == "predicted_up"  # activation → up when GoF
        assert unambiguous["FUS"] == "predicted_down"

    def test_get_mixed_targets(self):
        ts = self._make_v2_target_set()
        mixed = ts.get_mixed_targets()
        assert mixed == {"MAPT"}

    def test_filter_by_min_sources_2(self):
        ts = self._make_v2_target_set()
        filtered = ts.filter_by_min_sources(2)

        assert "SOD1" in filtered  # reach + sparser = 2
        assert "TARDBP" in filtered  # reach + sparser + signor = 3
        assert "FUS" not in filtered  # reach only = 1
        assert "MAPT" not in filtered  # reach only = 1

    def test_filter_by_min_sources_3(self):
        ts = self._make_v2_target_set()
        filtered = ts.filter_by_min_sources(3)
        assert filtered == {"TARDBP": "Q13148"}  # only one with 3 sources

    def test_filter_no_metadata_returns_all(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1", "B": "P2"},
            gene_symbol="X", min_evidence=1, n_hops=1,
        )
        filtered = ts.filter_by_min_sources(2)
        assert filtered == {"A": "P1", "B": "P2"}

    def test_evidence_weights_multi_source(self):
        ts = self._make_v2_target_set()
        w = ts.evidence_weights()
        assert w["SOD1"] == 1.0    # reach + sparser = 2 sources
        assert w["TARDBP"] == 1.0  # reach + sparser + signor = 3 sources
        assert w["FUS"] == 0.2     # reach only, evidence_count=1

    def test_evidence_weights_single_source_high_evidence(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1"},
            gene_symbol="X", min_evidence=1, n_hops=1,
            edge_metadata={"A": [{"regulation_type": "activation",
                                  "sources": ["reach"], "evidence_count": 5}]},
        )
        w = ts.evidence_weights()
        assert w["A"] == 0.5  # single source, evidence >= 3

    def test_evidence_weights_no_metadata_all_ones(self):
        ts = TargetSet.from_query(
            targets_in_data={"A": "P1", "B": "P2"},
            gene_symbol="X", min_evidence=1, n_hops=1,
        )
        w = ts.evidence_weights()
        assert w == {"A": 1.0, "B": 1.0}

    def test_to_weighted_feature_ids(self):
        ts = self._make_v2_target_set()
        ids, weights = ts.to_weighted_feature_ids()
        assert len(ids) == len(weights) == 4
        # SOD1 (P00441) should have weight 1.0
        idx = ids.index("P00441")
        assert weights[idx] == 1.0

    def test_unambiguous_ignores_targets_not_in_set(self):
        """edge_metadata may have symbols not in targets (e.g., filtered out)."""
        ts = TargetSet.from_query(
            targets_in_data={"SOD1": "P00441"},  # only SOD1 in targets
            gene_symbol="C9orf72", min_evidence=1, n_hops=1,
            edge_metadata={
                "SOD1": [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}],
                "REMOVED": [{"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1}],
            },
        )
        unambiguous = ts.get_unambiguous_targets()
        assert "SOD1" in unambiguous
        assert "REMOVED" not in unambiguous
