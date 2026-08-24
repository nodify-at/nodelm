from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from nodelm.models import stable_model_id


class ArtifactCollisionError(RuntimeError):
    """A deterministic artifact path already contains different data."""


@dataclass(frozen=True)
class ArtifactWriteResult:
    path: Path
    digest: str
    created: bool


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def content_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_artifact_name(prefix: str, inputs: Any, suffix: str = ".json") -> str:
    return f"{prefix}-{stable_model_id(inputs)[:16]}{suffix}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def write_immutable_stream(
    path: Path,
    writer: Callable[[BinaryIO], object],
    *,
    before_publish: Callable[[], object] | None = None,
) -> ArtifactWriteResult:
    """Publish writer output atomically without buffering the artifact in memory.

    ``before_publish`` runs after the temporary file is flushed and hashed, immediately
    before the no-clobber link. A raised exception aborts publication and removes the
    temporary file, allowing callers to revalidate streamed inputs without leaving a
    mislabeled immutable artifact behind.
    """

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        digest = _file_digest(temporary_path)
        if before_publish is not None:
            before_publish()
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if not _files_equal(temporary_path, path):
                raise ArtifactCollisionError(
                    f"refusing to overwrite differing artifact at {path}; "
                    f"existing={_file_digest(path)} requested={digest}"
                ) from None
            created = False
        else:
            created = True
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)

    return ArtifactWriteResult(path=path, digest=digest, created=created)


def write_immutable_json(path: Path, value: Any) -> ArtifactWriteResult:
    data = canonical_json_bytes(value)
    return write_immutable_stream(path, lambda stream: stream.write(data))
