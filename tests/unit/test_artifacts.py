from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

import pytest

from nodelm.artifacts import (
    ArtifactCollisionError,
    canonical_json_bytes,
    write_immutable_json,
    write_immutable_stream,
)


def test_canonical_json_is_stable_and_newline_terminated() -> None:
    assert canonical_json_bytes({"z": 1, "a": "å"}) == ('{"a":"å","z":1}\n'.encode())


def test_immutable_write_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    first = write_immutable_json(path, {"rows": [1, 2]})
    second = write_immutable_json(path, {"rows": [1, 2]})

    assert first.created is True
    assert second.created is False
    assert first.digest == second.digest
    assert json.loads(path.read_text()) == {"rows": [1, 2]}


def test_immutable_write_refuses_different_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_immutable_json(path, {"value": 1})

    with pytest.raises(ArtifactCollisionError):
        write_immutable_json(path, {"value": 2})


def test_concurrent_immutable_writes_never_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    def publish(value: int) -> int | ArtifactCollisionError:
        try:
            write_immutable_json(path, {"value": value})
        except ArtifactCollisionError as error:
            return error
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (1, 2)))

    assert sum(isinstance(outcome, ArtifactCollisionError) for outcome in outcomes) == 1
    assert json.loads(path.read_text()) in ({"value": 1}, {"value": 2})


def test_streaming_immutable_write_is_idempotent_and_collision_safe(tmp_path: Path) -> None:
    path = tmp_path / "streamed.json"

    first = write_immutable_stream(path, lambda stream: stream.write(b'{"ok":true}\n'))
    second = write_immutable_stream(path, lambda stream: stream.write(b'{"ok":true}\n'))

    assert first.created is True
    assert second.created is False
    assert first.digest == second.digest
    with pytest.raises(ArtifactCollisionError):
        write_immutable_stream(path, lambda stream: stream.write(b'{"ok":false}\n'))


def test_pre_publish_verification_failure_leaves_no_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    events: list[str] = []

    def writer(stream: BinaryIO) -> None:
        stream.write(b'{"ok":true}\n')
        events.append("written")

    def reject_publication() -> None:
        events.append("verified")
        raise RuntimeError("source changed")

    with pytest.raises(RuntimeError, match="source changed"):
        write_immutable_stream(path, writer, before_publish=reject_publication)

    assert events == ["written", "verified"]
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
