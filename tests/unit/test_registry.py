from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nodelm.datasets.registry import DatasetRegistry, RegistryError
from nodelm.models import VerificationStatus


def test_registry_loads_unverified_source_without_inventing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "datasets.yaml"
    path.write_text(
        """
schema_version: nodelm.dataset-registry/v1
sources:
  - name: open-swe-traces
    repository_id: nvidia/Open-SWE-Traces
    status: UNVERIFIED
    notes: Exact identifier requires primary-source verification.
""".lstrip()
    )

    registry = DatasetRegistry.load(path)

    assert registry.sources[0].status is VerificationStatus.UNVERIFIED
    assert registry.sources[0].revision is None


def test_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "datasets.yaml"
    path.write_text(
        """
schema_version: nodelm.dataset-registry/v1
sources:
  - name: duplicate
    repository_id: owner/one
    status: UNVERIFIED
  - name: duplicate
    repository_id: owner/two
    status: UNVERIFIED
""".lstrip()
    )

    with pytest.raises(RegistryError, match="duplicate dataset name"):
        DatasetRegistry.load(path)


def test_registry_from_bytes_passes_the_exact_buffer_to_yaml_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"exact registry bytes"
    observed: bytes | None = None

    def fake_safe_load(candidate: bytes) -> dict[str, Any]:
        nonlocal observed
        observed = candidate
        return {
            "schema_version": "nodelm.dataset-registry/v1",
            "sources": [
                {
                    "name": "fixture",
                    "repository_id": "owner/fixture",
                    "status": "UNVERIFIED",
                }
            ],
        }

    monkeypatch.setattr(yaml, "safe_load", fake_safe_load)

    registry = DatasetRegistry.from_bytes(payload)

    assert observed is payload
    assert registry.sources[0].name == "fixture"


def test_registry_load_reads_the_path_once_then_parses_that_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "datasets.yaml"
    payload = b"""
schema_version: nodelm.dataset-registry/v1
sources:
  - name: first-read
    repository_id: owner/fixture
    status: UNVERIFIED
""".lstrip()
    read_count = 0
    original_read_bytes = Path.read_bytes

    def read_bytes_once(candidate: Path) -> bytes:
        nonlocal read_count
        if candidate != path:
            return original_read_bytes(candidate)
        read_count += 1
        if read_count > 1:
            return payload.replace(b"first-read", b"second-read")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    registry = DatasetRegistry.load(path)

    assert read_count == 1
    assert registry.sources[0].name == "first-read"
