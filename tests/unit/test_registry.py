from __future__ import annotations

from pathlib import Path

import pytest

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
