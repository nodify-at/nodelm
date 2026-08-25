from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nodelm.artifacts import canonical_json_bytes, content_digest
from nodelm.datasets.lineage import build_snapshot_transfer_receipt, capture_snapshot_identity
from nodelm.datasets.partitions import (
    PartitionContractError,
    TracePartition,
    TracePartitionContract,
)
from nodelm.datasets.registry import DatasetRegistry, RegistryError
from nodelm.datasets.seals import SnapshotSealError, require_authorized_snapshot_seal
from nodelm.models import DatasetSource, VerificationStatus


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


def test_open_swe_partition_contract_records_all_real_leaf_partitions() -> None:
    contract = TracePartitionContract.load(Path("configs/datasets/open-swe-trace-partitions.yaml"))

    expected = {
        "minisweagent/qwen36_27b/scale-swe",
        "minisweagent/qwen36_27b/swe-rebench-v2",
        "openhands/deepseek-v4-flash/scale-swe",
        "openhands/minimax_m25/swe-rebench-v2",
        "openhands/qwen35_122b/swe-rebench-v2",
        "openhands/qwen36_27b/scale-swe",
        "openhands/qwen36_27b/swe-rebench-v2",
        "sweagent/minimax_m25/swe-rebench-v2",
        "sweagent/qwen35_122b/swe-rebench-v2",
        "sweagent/qwen36_27b/scale-swe",
        "sweagent/qwen36_27b/swe-rebench-v2",
    }

    assert {partition.name for partition in contract.partitions} == expected
    deepseek = contract.by_name("openhands/deepseek-v4-flash/scale-swe")
    assert deepseek.file_patterns == ("data/openhands/deepseek-v4-flash/scale-swe/*.parquet",)
    assert "deepseek_v4_flash" not in "\n".join(
        pattern for partition in contract.partitions for pattern in partition.file_patterns
    )


def test_partition_contract_rejects_escaping_patterns() -> None:
    payload = b"""
schema_version: nodelm.trace-partition-contract/v1
source_name: fixture
source_repository_id: owner/fixture
source_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
sealed_registry_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
transfer_receipt_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
snapshot_sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
snapshot_file_count: 1
partitions:
  - name: fixture/model/tasks
    harness: fixture
    generating_model: source-label:model
    upstream_source: tasks
    row_dataset_name: owner/tasks
    normalization_status: BLOCKED
    file_patterns: [../outside.parquet]
"""

    with pytest.raises(PartitionContractError, match="contained relative glob"):
        TracePartitionContract.from_bytes(payload)


def test_partition_contract_rejects_source_revision_mismatch() -> None:
    contract = TracePartitionContract.load(Path("configs/datasets/open-swe-trace-partitions.yaml"))

    with pytest.raises(PartitionContractError, match="source revision"):
        contract.require_source("open-swe-traces", "b" * 40)


def test_open_swe_partition_contract_digest_is_code_authorized() -> None:
    path = Path("configs/datasets/open-swe-trace-partitions.yaml")
    payload = path.read_bytes()
    contract = TracePartitionContract.from_bytes(payload)

    contract.require_authorized_digest(content_digest(payload))

    with pytest.raises(PartitionContractError, match="not authorized"):
        contract.require_authorized_digest("0" * 64)


def test_real_snapshot_receipts_are_code_authorized() -> None:
    require_authorized_snapshot_seal(
        source_name="swe-rebench-v2",
        source_revision="475dd5e8703bb5fb22dd3c60b5d038b019eba1e0",
        transfer_receipt_sha256=(
            "fbcd4fbb2b9c4b887ef15f368f3673c07d82d4ba81d2b0d0eed7e3dd6d1fe254"
        ),
        snapshot_sha256=("4f4328b560d27918da8f2d251c037789add5b5f7566c46825eeed91aa9d9c117"),
        snapshot_file_count=1,
    )

    with pytest.raises(SnapshotSealError, match="not authorized"):
        require_authorized_snapshot_seal(
            source_name="swe-rebench-v2",
            source_revision="475dd5e8703bb5fb22dd3c60b5d038b019eba1e0",
            transfer_receipt_sha256="0" * 64,
            snapshot_sha256=("4f4328b560d27918da8f2d251c037789add5b5f7566c46825eeed91aa9d9c117"),
            snapshot_file_count=1,
        )


def test_partition_contract_binds_complete_disjoint_receipt_coverage(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    one = snapshot / "data" / "one" / "model" / "tasks"
    two = snapshot / "data" / "two" / "model" / "tasks"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    (one / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (two / "b.jsonl").write_text("{}\n", encoding="utf-8")
    source = DatasetSource(
        name="fixture",
        repository_id="owner/fixture",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=2,
        evidence_urls=("https://example.invalid/evidence",),
        status=VerificationStatus.PASS,
    )
    snapshot_identity = capture_snapshot_identity(snapshot)
    receipt = build_snapshot_transfer_receipt(
        source=source,
        registry_sha256="b" * 64,
        registry_bytes=1,
        snapshot=snapshot_identity,
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json"))
    contract = TracePartitionContract.model_validate(
        {
            "schema_version": "nodelm.trace-partition-contract/v1",
            "source_name": source.name,
            "source_repository_id": source.repository_id,
            "source_revision": source.revision,
            "sealed_registry_sha256": "b" * 64,
            "transfer_receipt_sha256": content_digest(receipt_payload),
            "snapshot_sha256": snapshot_identity.snapshot_sha256,
            "snapshot_file_count": 2,
            "partitions": [
                {
                    "name": "one/model/tasks",
                    "harness": "one",
                    "generating_model": "source-label:model",
                    "upstream_source": "tasks",
                    "row_dataset_name": "owner/tasks",
                    "normalization_status": "BLOCKED",
                    "file_patterns": ["data/one/model/tasks/*.jsonl"],
                },
                {
                    "name": "two/model/tasks",
                    "harness": "two",
                    "generating_model": "source-label:model",
                    "upstream_source": "tasks",
                    "row_dataset_name": "owner/tasks",
                    "normalization_status": "BLOCKED",
                    "file_patterns": ["data/two/model/tasks/*.jsonl"],
                },
            ],
        }
    )

    bound = contract.bind_transfer_receipt(receipt_payload)

    assert bound.snapshot == snapshot_identity


def test_partition_contract_rejects_cross_partition_pattern() -> None:
    payload = b"""
schema_version: nodelm.trace-partition-contract/v1
source_name: fixture
source_repository_id: owner/fixture
source_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
sealed_registry_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
transfer_receipt_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
snapshot_sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
snapshot_file_count: 1
partitions:
  - name: one/model/tasks
    harness: one
    generating_model: source-label:model
    upstream_source: tasks
    row_dataset_name: owner/tasks
    normalization_status: BLOCKED
    file_patterns: [data/two/*.jsonl]
"""

    with pytest.raises(PartitionContractError, match="named leaf"):
        TracePartitionContract.from_bytes(payload)


@pytest.mark.parametrize("field", ["task_source_name", "task_source_revision"])
def test_blocked_partition_rejects_partial_task_source(field: str) -> None:
    partition = {
        "name": "fixture/model/tasks",
        "harness": "fixture",
        "generating_model": "source-label:model",
        "upstream_source": "tasks",
        "row_dataset_name": "owner/tasks",
        "normalization_status": "BLOCKED",
        "file_patterns": ["data/fixture/model/tasks/*.jsonl"],
        field: "fixture-tasks" if field == "task_source_name" else "a" * 40,
    }

    with pytest.raises(ValidationError, match="must omit both"):
        TracePartition.model_validate(partition)


@pytest.mark.parametrize("missing", ["task_source_name", "task_source_revision"])
def test_pass_partition_requires_both_task_source_fields(missing: str) -> None:
    partition = {
        "name": "fixture/model/tasks",
        "harness": "fixture",
        "generating_model": "source-label:model",
        "upstream_source": "tasks",
        "row_dataset_name": "owner/tasks",
        "normalization_status": "PASS",
        "task_source_name": "fixture-tasks",
        "task_source_revision": "a" * 40,
        "file_patterns": ["data/fixture/model/tasks/*.jsonl"],
    }
    del partition[missing]

    with pytest.raises(ValidationError, match="require both"):
        TracePartition.model_validate(partition)
