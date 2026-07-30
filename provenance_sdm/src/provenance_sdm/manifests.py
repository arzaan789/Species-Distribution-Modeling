"""Stable hashes and immutable JSON manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path


CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hexadecimal digest of *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    payload: Mapping[str, object],
    path: Path,
    *,
    allow_replace: bool = False,
) -> Path:
    """Write stable JSON while refusing implicit changes to an existing file."""

    destination = Path(path)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing == serialized:
            return destination
        if not allow_replace:
            raise FileExistsError(
                f"existing manifest differs from requested content: {destination}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, destination)
    return destination
