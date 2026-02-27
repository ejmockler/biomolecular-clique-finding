"""
Atomic file-write utilities.

Prevents corrupted output when a process is interrupted mid-write by
writing to a temporary file in the same directory and then performing an
atomic ``os.replace()`` (POSIX rename guarantee).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def atomic_write_json(path: str | os.PathLike, data: Any, *, indent: int = 2) -> None:
    """Write *data* as JSON atomically via temp-file + rename.

    The data is first serialized to a temporary file in the same
    directory as *path*, then moved into place with ``os.replace()``.
    On POSIX systems this is an atomic operation — readers will see
    either the old file or the new file, never a partially written one.

    Parameters
    ----------
    path:
        Destination file path.
    data:
        JSON-serializable object.
    indent:
        JSON indentation (default 2).
    """
    path = str(path)
    dir_path = os.path.dirname(path) or "."
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_text(path: str | os.PathLike, content: str) -> None:
    """Write *content* as text atomically via temp-file + rename.

    Parameters
    ----------
    path:
        Destination file path.
    content:
        Text content to write.
    """
    path = str(path)
    dir_path = os.path.dirname(path) or "."
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
