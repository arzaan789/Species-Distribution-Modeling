from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from provenance_sdm.manifests import sha256_file, write_manifest


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"abc")

    assert sha256_file(source) == hashlib.sha256(b"abc").hexdigest()


def test_manifest_json_has_stable_sorted_serialization(tmp_path: Path) -> None:
    path = write_manifest({"b": 2, "a": 1}, tmp_path / "manifest.json")

    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_identical_manifest_can_be_written_again(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest({"a": 1}, path)

    assert write_manifest({"a": 1}, path) == path
    assert path.read_text(encoding="utf-8") == '{\n  "a": 1\n}\n'


def test_changed_manifest_is_not_overwritten_by_default(tmp_path: Path) -> None:
    path = write_manifest({"a": 1}, tmp_path / "manifest.json")

    with pytest.raises(FileExistsError, match="differs"):
        write_manifest({"a": 2}, path)

    assert path.read_text(encoding="utf-8") == '{\n  "a": 1\n}\n'


def test_changed_manifest_requires_explicit_replacement(tmp_path: Path) -> None:
    path = write_manifest({"a": 1}, tmp_path / "manifest.json")

    write_manifest({"a": 2}, path, allow_replace=True)

    assert path.read_text(encoding="utf-8") == '{\n  "a": 2\n}\n'
