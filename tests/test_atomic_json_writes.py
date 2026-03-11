"""
Tests for atomic JSON writes in CLI modules.

Verifies that:
1. atomic_write_json produces valid JSON with indent and default params
2. atomic_write_json cleans up temp files on failure
3. All three CLI files use atomic_write_json instead of json.dump
4. The default= parameter works for non-serializable types
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Tests for atomic_write_json utility
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """Tests for the atomic_write_json function itself."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """atomic_write_json writes valid JSON to the target path."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "out.json"
        data = {"key": "value", "number": 42}
        atomic_write_json(target, data)

        assert target.exists()
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_indent_default(self, tmp_path: Path) -> None:
        """Default indent is 2 spaces."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "indented.json"
        data = {"a": 1}
        atomic_write_json(target, data)

        content = target.read_text()
        # json.dumps with indent=2 produces exactly this
        expected = json.dumps(data, indent=2)
        assert content == expected

    def test_custom_indent(self, tmp_path: Path) -> None:
        """Custom indent is respected."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "indented4.json"
        data = {"a": 1}
        atomic_write_json(target, data, indent=4)

        content = target.read_text()
        expected = json.dumps(data, indent=4)
        assert content == expected

    def test_default_str_serializer(self, tmp_path: Path) -> None:
        """default=str converts non-serializable objects (e.g. Path, datetime)."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "defaults.json"
        data = {
            "path": Path("/some/path"),
            "timestamp": datetime(2026, 2, 28, 12, 0, 0),
            "number": 42,
        }
        atomic_write_json(target, data, default=str)

        with open(target) as f:
            loaded = json.load(f)
        assert loaded["path"] == "/some/path"
        assert loaded["timestamp"] == "2026-02-28 12:00:00"
        assert loaded["number"] == 42

    def test_default_none_raises_on_non_serializable(self, tmp_path: Path) -> None:
        """Without default=, non-serializable objects raise TypeError."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "fail.json"
        data = {"path": Path("/some/path")}

        with pytest.raises(TypeError):
            atomic_write_json(target, data)

        # File should NOT be written on failure
        assert not target.exists()

    def test_custom_default_callable(self, tmp_path: Path) -> None:
        """Custom default callable is passed through to json.dump."""
        from cliquefinder.utils.fileio import atomic_write_json

        def custom_default(obj):
            if isinstance(obj, set):
                return sorted(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        target = tmp_path / "custom.json"
        data = {"genes": {"TP53", "AKT1", "BRAF"}}
        atomic_write_json(target, data, default=custom_default)

        with open(target) as f:
            loaded = json.load(f)
        assert loaded["genes"] == ["AKT1", "BRAF", "TP53"]

    def test_atomic_no_partial_writes(self, tmp_path: Path) -> None:
        """If write fails, the original file is preserved (not corrupted)."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "existing.json"
        original = {"version": 1}
        target.write_text(json.dumps(original))

        # Try to write non-serializable data (without default=)
        with pytest.raises(TypeError):
            atomic_write_json(target, {"bad": object()})

        # Original file must be intact
        with open(target) as f:
            assert json.load(f) == original

    def test_no_temp_files_on_failure(self, tmp_path: Path) -> None:
        """Temp files are cleaned up even if write fails."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "noclutter.json"
        with pytest.raises(TypeError):
            atomic_write_json(target, {"bad": object()})

        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        """Atomic write replaces existing file content."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "overwrite.json"
        atomic_write_json(target, {"version": 1})
        atomic_write_json(target, {"version": 2})

        with open(target) as f:
            assert json.load(f) == {"version": 2}

    def test_nested_data_structure(self, tmp_path: Path) -> None:
        """Deeply nested data is serialized correctly."""
        from cliquefinder.utils.fileio import atomic_write_json

        target = tmp_path / "nested.json"
        data = {
            "analysis": {
                "regulators": ["TP53", "MYC"],
                "params": {"alpha": 0.05, "method": "ROAST"},
                "results": [{"gene": "BRAF", "pvalue": 0.01}],
            }
        }
        atomic_write_json(target, data)

        with open(target) as f:
            assert json.load(f) == data

    def test_pathlib_path_accepted(self, tmp_path: Path) -> None:
        """Both str and Path objects are accepted for path."""
        from cliquefinder.utils.fileio import atomic_write_json

        # Path object
        p1 = tmp_path / "path_obj.json"
        atomic_write_json(p1, {"a": 1})
        assert p1.exists()

        # str
        p2 = str(tmp_path / "str_path.json")
        atomic_write_json(p2, {"b": 2})
        assert os.path.exists(p2)


# ---------------------------------------------------------------------------
# Tests verifying CLI modules use atomic writes (source inspection)
# ---------------------------------------------------------------------------


class TestCliModulesUseAtomicWrites:
    """Verify that the three CLI files no longer use raw json.dump for output."""

    @staticmethod
    def _get_source(module_path: str) -> str:
        """Read source code of a module."""
        full = Path(__file__).resolve().parent.parent / "src" / module_path
        return full.read_text()

    def test_validate_baselines_no_raw_json_dump(self) -> None:
        """validate_baselines.py should not use json.dump() for writes."""
        src = self._get_source("cliquefinder/cli/validate_baselines.py")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dump"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                pytest.fail(
                    "validate_baselines.py still contains json.dump() — "
                    "should use atomic_write_json"
                )

    def test_validate_baselines_imports_atomic(self) -> None:
        """validate_baselines.py imports atomic_write_json."""
        src = self._get_source("cliquefinder/cli/validate_baselines.py")
        assert "atomic_write_json" in src

    def test_differential_no_raw_json_dump(self) -> None:
        """differential.py should not use json.dump() for writes."""
        src = self._get_source("cliquefinder/cli/differential.py")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dump"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                pytest.fail(
                    "differential.py still contains json.dump() — "
                    "should use atomic_write_json"
                )

    def test_differential_imports_atomic(self) -> None:
        """differential.py imports atomic_write_json."""
        src = self._get_source("cliquefinder/cli/differential.py")
        assert "atomic_write_json" in src

    def test_analyze_core_no_raw_json_dump(self) -> None:
        """_analyze_core.py should not use json.dump() for writes."""
        src = self._get_source("cliquefinder/cli/_analyze_core.py")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dump"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                pytest.fail(
                    "_analyze_core.py still contains json.dump() — "
                    "should use atomic_write_json"
                )

    def test_analyze_core_imports_atomic(self) -> None:
        """_analyze_core.py imports atomic_write_json."""
        src = self._get_source("cliquefinder/cli/_analyze_core.py")
        assert "atomic_write_json" in src


# ---------------------------------------------------------------------------
# Tests for atomic_write_json default= parameter signature
# ---------------------------------------------------------------------------


class TestAtomicWriteJsonSignature:
    """Verify the function signature supports the default parameter."""

    def test_default_is_keyword_only(self) -> None:
        """The default parameter should be keyword-only."""
        import inspect
        from cliquefinder.utils.fileio import atomic_write_json

        sig = inspect.signature(atomic_write_json)
        param = sig.parameters["default"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_indent_is_keyword_only(self) -> None:
        """The indent parameter should be keyword-only."""
        import inspect
        from cliquefinder.utils.fileio import atomic_write_json

        sig = inspect.signature(atomic_write_json)
        param = sig.parameters["indent"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default == 2
