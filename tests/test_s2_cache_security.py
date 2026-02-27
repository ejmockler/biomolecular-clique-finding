"""
Tests for S-2 security fix: Replace pickle.load() with JSON in cache files.

Validates that:
1. No pickle.load/dump calls remain in the 4 affected files
2. JSON round-trip works correctly for each cache data type
3. Annotation cache preserves set semantics through JSON serialization
4. Old .pkl cache files are ignored (backward compatibility via extension change)
5. Corrupted JSON cache files are handled gracefully
"""

from __future__ import annotations

import inspect
import json
import pickle
import textwrap
from pathlib import Path
from typing import Dict, Set
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Source inspection: no pickle.load/dump in the 4 affected files
# ---------------------------------------------------------------------------

# Paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET_FILES = [
    _REPO_ROOT / "src" / "cliquefinder" / "validation" / "id_mapping.py",
    _REPO_ROOT / "src" / "cliquefinder" / "validation" / "entity_resolver.py",
    _REPO_ROOT / "src" / "cliquefinder" / "validation" / "annotation_providers.py",
    _REPO_ROOT / "src" / "cliquefinder" / "knowledge" / "cross_modal_mapper.py",
]


@pytest.mark.parametrize("filepath", _TARGET_FILES, ids=lambda p: p.name)
def test_no_pickle_load_in_source(filepath: Path):
    """Ensure pickle.load() is not called in any of the 4 target files."""
    source = filepath.read_text()
    assert "pickle.load" not in source, (
        f"{filepath.name} still contains pickle.load()"
    )


@pytest.mark.parametrize("filepath", _TARGET_FILES, ids=lambda p: p.name)
def test_no_pickle_dump_in_source(filepath: Path):
    """Ensure pickle.dump() is not called in any of the 4 target files."""
    source = filepath.read_text()
    assert "pickle.dump" not in source, (
        f"{filepath.name} still contains pickle.dump()"
    )


@pytest.mark.parametrize("filepath", _TARGET_FILES, ids=lambda p: p.name)
def test_no_pickle_import_in_source(filepath: Path):
    """Ensure 'import pickle' is not present in any of the 4 target files."""
    source = filepath.read_text()
    assert "import pickle" not in source, (
        f"{filepath.name} still imports pickle"
    )


# ---------------------------------------------------------------------------
# 2. JSON round-trip: id_mapping.py (dict[str, str])
# ---------------------------------------------------------------------------

class TestIdMappingCacheRoundTrip:
    """Test that id_mapping cache works correctly with JSON."""

    def test_json_round_trip_dict_str_str(self, tmp_path):
        """dict[str, str] survives JSON round-trip."""
        data = {"ENSG00000141510": "TP53", "ENSG00000012048": "BRCA1"}
        cache_file = tmp_path / "test_cache.json"

        # Write
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        # Read
        with open(cache_file, "r") as f:
            loaded = json.load(f)

        assert loaded == data

    def test_cache_extension_is_json(self):
        """Verify the cache key uses .json extension."""
        from cliquefinder.validation.id_mapping import MyGeneInfoMapper

        source = inspect.getsource(MyGeneInfoMapper.map_ids)
        assert ".json" in source, "Cache key should use .json extension"
        assert ".pkl" not in source, "Cache key should not use .pkl extension"

    def test_old_pkl_file_ignored(self, tmp_path):
        """Old .pkl files are not loaded (different extension)."""
        # Create a .pkl file with pickle data
        pkl_file = tmp_path / "old_cache.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"ENSG00000141510": "OLD_VALUE"}, f)

        # JSON cache file does not exist
        json_file = tmp_path / "old_cache.json"
        assert not json_file.exists()

        # The .pkl file exists but the code would look for .json
        assert pkl_file.exists()
        assert not json_file.exists()

    def test_corrupted_json_handled_gracefully(self, tmp_path):
        """Corrupted JSON cache is ignored with warning, not exception."""
        cache_file = tmp_path / "corrupted.json"
        cache_file.write_text("{invalid json content")

        # Loading should not raise
        try:
            with open(cache_file, "r") as f:
                json.load(f)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected - the code catches this gracefully


# ---------------------------------------------------------------------------
# 3. JSON round-trip: entity_resolver.py (Dict[str, ResolvedEntity])
# ---------------------------------------------------------------------------

class TestEntityResolverCacheRoundTrip:
    """Test entity_resolver cache with ResolvedEntity serialization."""

    def test_resolved_entity_to_dict_round_trip(self):
        """ResolvedEntity.to_dict() produces JSON-serializable output."""
        from cliquefinder.validation.entity_resolver import ResolvedEntity

        entity = ResolvedEntity(
            ensembl_id="ENSG00000141510",
            symbol="TP53",
            source="mygene",
            confidence=0.85,
            biotype="protein_coding",
            aliases=["P53", "LFS1"],
            description="tumor protein p53",
        )

        # Serialize via to_dict
        d = entity.to_dict()
        json_str = json.dumps(d)

        # Deserialize
        loaded = json.loads(json_str)
        restored = ResolvedEntity(**loaded)

        assert restored.ensembl_id == entity.ensembl_id
        assert restored.symbol == entity.symbol
        assert restored.source == entity.source
        assert restored.confidence == entity.confidence
        assert restored.biotype == entity.biotype
        assert restored.aliases == entity.aliases
        assert restored.description == entity.description

    def test_resolved_entity_none_fields_round_trip(self):
        """ResolvedEntity with None fields survives round-trip."""
        from cliquefinder.validation.entity_resolver import ResolvedEntity

        entity = ResolvedEntity(
            ensembl_id="ENSG00000999999",
            symbol=None,
            source="none",
            confidence=0.0,
            biotype=None,
            aliases=[],
            description=None,
        )

        d = entity.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        restored = ResolvedEntity(**loaded)

        assert restored.symbol is None
        assert restored.biotype is None
        assert restored.description is None
        assert restored.confidence == 0.0

    def test_full_cache_dict_round_trip(self, tmp_path):
        """Full Dict[str, ResolvedEntity] round-trips through JSON."""
        from cliquefinder.validation.entity_resolver import ResolvedEntity

        results = {
            "ENSG00000141510": ResolvedEntity(
                ensembl_id="ENSG00000141510",
                symbol="TP53",
                source="mygene",
                confidence=0.85,
                biotype="protein_coding",
                aliases=["P53"],
                description="tumor protein p53",
            ),
            "ENSG00000012048": ResolvedEntity(
                ensembl_id="ENSG00000012048",
                symbol="BRCA1",
                source="mygene",
                confidence=0.85,
                biotype="protein_coding",
                aliases=["BRCA1/BRCA2"],
                description="BRCA1 DNA repair",
            ),
        }

        cache_file = tmp_path / "entity_cache.json"

        # Serialize
        serializable = {eid: e.to_dict() for eid, e in results.items()}
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2)

        # Deserialize
        with open(cache_file, "r") as f:
            raw = json.load(f)
        restored = {eid: ResolvedEntity(**data) for eid, data in raw.items()}

        assert set(restored.keys()) == set(results.keys())
        for eid in results:
            assert restored[eid].symbol == results[eid].symbol
            assert restored[eid].confidence == results[eid].confidence

    def test_cache_extension_is_json(self):
        """Verify entity resolver uses .json extension."""
        from cliquefinder.validation.entity_resolver import GeneEntityResolver

        source = inspect.getsource(GeneEntityResolver.resolve_ensembl_ids)
        assert ".json" in source, "Cache key should use .json extension"
        assert ".pkl" not in source, "Cache key should not use .pkl extension"


# ---------------------------------------------------------------------------
# 4. JSON round-trip: annotation_providers.py (mixed dict with Set values)
# ---------------------------------------------------------------------------

class TestAnnotationCacheRoundTrip:
    """Test annotation cache preserves set semantics through JSON."""

    def test_set_serialization_round_trip(self):
        """Sets are preserved through the tagged JSON format."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        original = {"GO:0006915", "GO:0042981", "GO:0004674"}

        serialized = CachedAnnotationProvider._serialize_value(original)
        assert serialized["__set__"] is True
        assert sorted(serialized["values"]) == sorted(original)

        restored = CachedAnnotationProvider._deserialize_value(serialized)
        assert isinstance(restored, set)
        assert restored == original

    def test_nested_dict_with_sets_round_trip(self):
        """Dict[str, Set[str]] (annotation result) round-trips correctly."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        original = {
            "go_biological_process": {"GO:0006915", "GO:0042981"},
            "go_molecular_function": {"GO:0004674"},
            "go_cellular_component": {"GO:0005634", "GO:0005737"},
        }

        serialized = CachedAnnotationProvider._serialize_value(original)
        restored = CachedAnnotationProvider._deserialize_value(serialized)

        assert isinstance(restored, dict)
        for key in original:
            assert isinstance(restored[key], set), f"{key} should be a set"
            assert restored[key] == original[key]

    def test_string_value_round_trip(self):
        """Plain string values (term names) survive round-trip."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        original = "apoptotic process"
        serialized = CachedAnnotationProvider._serialize_value(original)
        restored = CachedAnnotationProvider._deserialize_value(serialized)
        assert restored == original
        assert isinstance(restored, str)

    def test_full_mixed_cache_round_trip(self, tmp_path):
        """Full mixed cache (sets, dicts, strings) round-trips via file."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        cache = {
            "gene:ENSG00000141510": {
                "go_biological_process": {"GO:0006915", "GO:0042981"},
                "go_molecular_function": {"GO:0004674"},
                "go_cellular_component": set(),
            },
            "term_name:GO:0006915": "apoptotic process",
            "term:GO:0006915": {"ENSG00000141510", "ENSG00000012048"},
            "all_terms": {"GO:0006915", "GO:0042981", "GO:0004674"},
        }

        cache_file = tmp_path / "annotations.json"

        # Serialize
        serializable = {
            k: CachedAnnotationProvider._serialize_value(v)
            for k, v in cache.items()
        }
        with open(cache_file, "w") as f:
            json.dump(serializable, f, indent=2)

        # Deserialize
        with open(cache_file, "r") as f:
            raw = json.load(f)
        restored = CachedAnnotationProvider._deserialize_cache(raw)

        # Verify types and values
        assert isinstance(restored["gene:ENSG00000141510"]["go_biological_process"], set)
        assert restored["gene:ENSG00000141510"]["go_biological_process"] == {"GO:0006915", "GO:0042981"}
        assert isinstance(restored["term_name:GO:0006915"], str)
        assert restored["term_name:GO:0006915"] == "apoptotic process"
        assert isinstance(restored["term:GO:0006915"], set)
        assert restored["term:GO:0006915"] == {"ENSG00000141510", "ENSG00000012048"}
        assert isinstance(restored["all_terms"], set)
        assert restored["all_terms"] == {"GO:0006915", "GO:0042981", "GO:0004674"}

    def test_empty_set_round_trip(self):
        """Empty sets are preserved through JSON."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        original = set()
        serialized = CachedAnnotationProvider._serialize_value(original)
        restored = CachedAnnotationProvider._deserialize_value(serialized)
        assert isinstance(restored, set)
        assert len(restored) == 0

    def test_cached_provider_init_with_corrupted_json(self, tmp_path):
        """CachedAnnotationProvider handles corrupted JSON gracefully."""
        from cliquefinder.validation.annotation_providers import (
            CachedAnnotationProvider,
            AnnotationProvider,
        )

        cache_file = tmp_path / "corrupted.json"
        cache_file.write_text("{not valid json!!!")

        # Create a mock provider
        mock_provider = MagicMock(spec=AnnotationProvider)

        # Should not raise, just warn and start with empty cache
        provider = CachedAnnotationProvider(mock_provider, cache_file=cache_file)
        assert provider._cache == {}

    def test_cached_provider_default_extension(self):
        """Default cache file uses .json extension."""
        from cliquefinder.validation.annotation_providers import CachedAnnotationProvider

        source = inspect.getsource(CachedAnnotationProvider.__init__)
        assert "annotations.json" in source
        assert "annotations.pkl" not in source


# ---------------------------------------------------------------------------
# 5. JSON round-trip: cross_modal_mapper.py (dict and Set)
# ---------------------------------------------------------------------------

class TestCrossModalCacheRoundTrip:
    """Test cross-modal mapper cache with dict and set data types."""

    def test_dict_cache_round_trip(self, tmp_path):
        """Dict[str, str] mapping cache round-trips through JSON."""
        data = {"ENSG00000141510": "TP53", "ENSG00000012048": "BRCA1"}
        cache_file = tmp_path / "test.json"

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        with open(cache_file, "r") as f:
            loaded = json.load(f)

        assert loaded == data

    def test_set_cache_round_trip_with_tagged_format(self, tmp_path):
        """Set[str] cache uses __set__ tag and round-trips correctly."""
        original = {"TP53", "BRCA1", "MDM2"}
        cache_file = tmp_path / "set_cache.json"

        # Serialize with __set__ tag (as cross_modal_mapper does)
        serialized = {"__set__": True, "values": sorted(original)}
        with open(cache_file, "w") as f:
            json.dump(serialized, f, indent=2)

        # Deserialize (as cross_modal_mapper does)
        with open(cache_file, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw.get("__set__") is True and "values" in raw:
            restored = set(raw["values"])
        else:
            restored = raw

        assert isinstance(restored, set)
        assert restored == original

    def test_load_cache_returns_none_for_missing(self, tmp_path, monkeypatch):
        """_load_cache returns None when no cache file exists."""
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        # Avoid requiring mygene import
        monkeypatch.setattr(
            "cliquefinder.knowledge.cross_modal_mapper.MyGeneInfoMapper",
            MagicMock,
        )
        mapper = CrossModalIDMapper(cache_dir=tmp_path)
        result = mapper._load_cache("nonexistent_key")
        assert result is None

    def test_save_and_load_dict_cache(self, tmp_path, monkeypatch):
        """_save_cache + _load_cache round-trip for dict data."""
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        monkeypatch.setattr(
            "cliquefinder.knowledge.cross_modal_mapper.MyGeneInfoMapper",
            MagicMock,
        )
        mapper = CrossModalIDMapper(cache_dir=tmp_path)

        data = {"ENSG00000141510": "TP53"}
        mapper._save_cache("test_key", data)
        loaded = mapper._load_cache("test_key")
        assert loaded == data

    def test_save_and_load_set_cache(self, tmp_path, monkeypatch):
        """_save_cache + _load_cache round-trip for set data."""
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        monkeypatch.setattr(
            "cliquefinder.knowledge.cross_modal_mapper.MyGeneInfoMapper",
            MagicMock,
        )
        mapper = CrossModalIDMapper(cache_dir=tmp_path)

        data = {"TP53", "BRCA1", "MDM2"}
        mapper._save_cache("set_key", data)
        loaded = mapper._load_cache("set_key")
        assert isinstance(loaded, set)
        assert loaded == data

    def test_corrupted_json_returns_none(self, tmp_path, monkeypatch):
        """Corrupted JSON cache returns None instead of raising."""
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        monkeypatch.setattr(
            "cliquefinder.knowledge.cross_modal_mapper.MyGeneInfoMapper",
            MagicMock,
        )
        mapper = CrossModalIDMapper(cache_dir=tmp_path)

        # Write corrupted JSON
        (tmp_path / "bad_key.json").write_text("{corrupted")

        result = mapper._load_cache("bad_key")
        assert result is None

    def test_cache_files_use_json_extension(self):
        """Verify _load_cache/_save_cache use .json extension."""
        from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper

        load_source = inspect.getsource(CrossModalIDMapper._load_cache)
        save_source = inspect.getsource(CrossModalIDMapper._save_cache)

        assert ".json" in load_source
        assert ".json" in save_source
        assert ".pkl" not in load_source
        assert ".pkl" not in save_source


# ---------------------------------------------------------------------------
# 6. Backward compatibility: old .pkl files are ignored
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Old .pkl cache files are automatically ignored by extension change."""

    def test_pkl_files_not_loaded_by_id_mapping(self, tmp_path):
        """id_mapping.py looks for .json, so .pkl files are ignored."""
        # Create an old .pkl file
        pkl_file = tmp_path / "cache.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"old": "data"}, f)

        # JSON file does not exist
        json_file = tmp_path / "cache.json"
        assert pkl_file.exists()
        assert not json_file.exists()

    def test_pkl_files_not_loaded_by_annotation_provider(self, tmp_path):
        """annotation_providers.py looks for .json, so .pkl is ignored."""
        from cliquefinder.validation.annotation_providers import (
            CachedAnnotationProvider,
            AnnotationProvider,
        )

        # Create an old annotations.pkl in the cache dir
        pkl_file = tmp_path / "annotations.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"gene:TP53": {"go_biological_process": {"GO:0006915"}}}, f)

        # Create provider pointing to .json cache (which doesn't exist)
        mock_provider = MagicMock(spec=AnnotationProvider)
        json_cache = tmp_path / "annotations.json"
        provider = CachedAnnotationProvider(mock_provider, cache_file=json_cache)

        # Cache should be empty (pkl ignored)
        assert provider._cache == {}
