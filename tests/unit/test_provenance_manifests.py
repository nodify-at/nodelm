from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from nodelm.provenance.manifests import (
    NormalizationManifestV2,
    SnapshotMaterializationManifestV1,
    SnapshotMaterializationManifestV2,
    TaskProvenanceProjectionManifestV1,
)


def _file(path: str = "data/fixture/model/tasks/traces.jsonl") -> dict[str, Any]:
    return {"path": path, "sha256": "f" * 64, "bytes": 123}


def _materialization_v1() -> dict[str, Any]:
    return {
        "schema_version": "nodelm.snapshot-materialization/v1",
        "status": "PASS",
        "source_name": "fixture-traces",
        "source_repository_id": "owner/fixture-traces",
        "source_revision": "a" * 40,
        "registry_sha256": "b" * 64,
        "file_patterns": ["data/fixture/model/tasks/*.jsonl"],
        "row_count": 2,
        "max_rows": 2,
        "files": [_file()],
        "output": "raw.jsonl",
        "output_sha256": "c" * 64,
        "output_bytes": 456,
    }


def _materialization_v2() -> dict[str, Any]:
    return {
        **_materialization_v1(),
        "schema_version": "nodelm.snapshot-materialization/v2",
        "materialization_scope": "canary",
        "partition_contract_sha256": "d" * 64,
        "partition_contract_bytes": 789,
        "transfer_receipt_sha256": "e" * 64,
        "transfer_receipt_bytes": 987,
        "partition_name": "fixture/model/tasks",
        "harness": "fixture",
        "generating_model": "source-label:model",
        "upstream_source": "tasks",
        "row_dataset_name": "owner/fixture-tasks",
        "task_source_name": "fixture-tasks",
        "task_source_revision": "1" * 40,
        "normalization_status": "PASS",
    }


def _task_projection() -> dict[str, Any]:
    return {
        "schema_version": "nodelm.task-provenance-projection/v1",
        "status": "PASS",
        "source_name": "fixture-tasks",
        "source_repository_id": "owner/fixture-tasks",
        "source_revision": "1" * 40,
        "registry_sha256": "b" * 64,
        "registry_bytes": 321,
        "transfer_receipt_sha256": "e" * 64,
        "transfer_receipt_bytes": 987,
        "snapshot_sha256": "2" * 64,
        "projection_scope": "complete-snapshot",
        "file_patterns": [],
        "files": [_file("tasks.jsonl")],
        "safe_fields": [
            "instance_id",
            "repository",
            "base_commit",
            "repository_license",
            "language",
            "source_dataset",
            "source_dataset_revision",
        ],
        "admitted_count": 2,
        "rejected_count": 1,
        "rejection_counts_by_code": {"disallowed_license": 1},
        "output": "tasks.safe.jsonl",
        "output_sha256": "3" * 64,
        "output_bytes": 654,
        "rejection_artifact": "tasks.safe.rejections.jsonl",
        "rejection_sha256": "4" * 64,
        "rejection_bytes": 111,
    }


def _normalization() -> dict[str, Any]:
    return {
        "schema_version": "nodelm.normalization-manifest/v2",
        "status": "PASS",
        "source_name": "fixture-traces",
        "source_repository_id": "owner/fixture-traces",
        "source_revision": "a" * 40,
        "partition_name": "fixture/model/tasks",
        "harness": "fixture",
        "generating_model": "source-label:model",
        "upstream_source": "tasks",
        "row_dataset_name": "owner/fixture-tasks",
        "input_sha256": "5" * 64,
        "input_bytes": 1_000,
        "registry_sha256": "b" * 64,
        "materialization_manifest_sha256": "6" * 64,
        "materialization_manifest_bytes": 1_001,
        "partition_contract_sha256": "d" * 64,
        "partition_contract_bytes": 789,
        "transfer_receipt_sha256": "e" * 64,
        "transfer_receipt_bytes": 987,
        "task_provenance_sha256": "3" * 64,
        "task_provenance_bytes": 654,
        "task_provenance_manifest_sha256": "7" * 64,
        "task_provenance_manifest_bytes": 1_002,
        "task_transfer_receipt_sha256": "8" * 64,
        "task_transfer_receipt_bytes": 1_003,
        "task_source_name": "fixture-tasks",
        "task_source_revision": "1" * 40,
        "materialization_replay": "PASS",
        "task_provenance_replay": "PASS",
        "uniqueness_scope": "complete-partition",
        "input_row_count": 3,
        "accepted_count": 1,
        "rejected_count": 2,
        "rejection_counts_by_code": {"unknown_resolution": 2},
        "unique_rollout_key_count": 2,
        "duplicate_trace_row_count": 0,
        "conflicting_rollout_identity_count": 0,
        "conflicting_rollout_row_count": 0,
        "gold_exposure_audit": "NOT RUN",
        "normalized_artifact": "normalized.jsonl",
        "normalized_sha256": "9" * 64,
        "normalized_bytes": 700,
        "rejection_artifact": "normalized.rejections.jsonl",
        "rejection_sha256": "0" * 64,
        "rejection_bytes": 300,
    }


@pytest.mark.parametrize(
    ("model", "payload_factory"),
    [
        (SnapshotMaterializationManifestV1, _materialization_v1),
        (SnapshotMaterializationManifestV2, _materialization_v2),
        (TaskProvenanceProjectionManifestV1, _task_projection),
        (NormalizationManifestV2, _normalization),
    ],
)
def test_manifest_models_round_trip_producer_payloads(
    model: type[BaseModel],
    payload_factory: Callable[[], dict[str, Any]],
) -> None:
    payload = payload_factory()

    assert model.model_validate(payload).model_dump(mode="json") == payload


def test_materialization_v2_preserves_a_blocked_normalization_partition() -> None:
    payload = _materialization_v2()
    payload["normalization_status"] = "BLOCKED"

    manifest = SnapshotMaterializationManifestV2.model_validate(payload)

    assert manifest.normalization_status == "BLOCKED"


@pytest.mark.parametrize(
    ("model", "payload_factory"),
    [
        (SnapshotMaterializationManifestV1, _materialization_v1),
        (SnapshotMaterializationManifestV2, _materialization_v2),
        (TaskProvenanceProjectionManifestV1, _task_projection),
        (NormalizationManifestV2, _normalization),
    ],
)
def test_manifest_models_forbid_extra_fields(
    model: type[BaseModel],
    payload_factory: Callable[[], dict[str, Any]],
) -> None:
    payload = payload_factory()
    payload["ignored_tampering"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload_factory", "field", "value"),
    [
        (SnapshotMaterializationManifestV1, _materialization_v1, "row_count", True),
        (SnapshotMaterializationManifestV2, _materialization_v2, "output_bytes", 456.0),
        (TaskProvenanceProjectionManifestV1, _task_projection, "admitted_count", True),
        (TaskProvenanceProjectionManifestV1, _task_projection, "registry_bytes", 321.0),
        (NormalizationManifestV2, _normalization, "accepted_count", True),
        (NormalizationManifestV2, _normalization, "normalized_bytes", 700.0),
    ],
)
def test_manifest_models_reject_bool_and_float_counts(
    model: type[BaseModel],
    payload_factory: Callable[[], dict[str, Any]],
    field: str,
    value: object,
) -> None:
    payload = payload_factory()
    payload[field] = value

    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("model", "payload_factory", "updates", "message"),
    [
        (
            SnapshotMaterializationManifestV1,
            _materialization_v1,
            {"row_count": 3},
            "max_rows",
        ),
        (
            SnapshotMaterializationManifestV2,
            _materialization_v2,
            {"max_rows": None, "materialization_scope": "canary"},
            "materialization_scope",
        ),
        (
            SnapshotMaterializationManifestV2,
            _materialization_v2,
            {"status": "FAIL"},
            "status",
        ),
        (
            TaskProvenanceProjectionManifestV1,
            _task_projection,
            {"file_patterns": ["tasks/*.jsonl"]},
            "projection_scope",
        ),
        (
            TaskProvenanceProjectionManifestV1,
            _task_projection,
            {"rejection_counts_by_code": {"disallowed_license": 2}},
            "rejected_count",
        ),
        (
            TaskProvenanceProjectionManifestV1,
            _task_projection,
            {"safe_fields": ["instance_id"]},
            "safe_fields",
        ),
        (
            NormalizationManifestV2,
            _normalization,
            {"input_row_count": 4},
            "accepted_count.*rejected_count",
        ),
        (
            NormalizationManifestV2,
            _normalization,
            {"duplicate_trace_row_count": 1},
            "duplicate_trace_row_count",
        ),
    ],
)
def test_manifest_models_reject_inconsistent_scope_status_and_counts(
    model: type[BaseModel],
    payload_factory: Callable[[], dict[str, Any]],
    updates: dict[str, Any],
    message: str,
) -> None:
    payload = payload_factory()
    payload.update(updates)

    with pytest.raises(ValidationError, match=message):
        model.model_validate(payload)
