from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from nodelm.provenance.manifests import (
    NormalizationManifestV2,
    ResolutionCanaryExecutionManifestV1,
    ResolutionRecoveryManifestV1,
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


def _resolution_recovery() -> dict[str, Any]:
    return {
        "schema_version": "nodelm.resolution-recovery/v1",
        "derivation_status": "PASS",
        "admission_status": "BLOCKED",
        "admission_blocker": "harness_canary_pending",
        "source_name": "fixture-traces",
        "source_repository_id": "owner/fixture-traces",
        "source_revision": "a" * 40,
        "task_source_name": "fixture-tasks",
        "task_source_revision": "1" * 40,
        "partition_contract_sha256": "b" * 64,
        "partition_contract_bytes": 1_000,
        "transfer_receipt_sha256": "c" * 64,
        "transfer_receipt_bytes": 2_000,
        "labeled_partitions": [
            {
                "partition_name": "openhands/model-a/tasks",
                "row_count": 8,
                "files": [_file("data/openhands/model-a/tasks/part-0.parquet")],
            },
            {
                "partition_name": "swe-agent/model-a/tasks",
                "row_count": 12,
                "files": [_file("data/swe-agent/model-a/tasks/part-0.parquet")],
            },
        ],
        "target_partitions": [
            {
                "partition_name": "openhands/model-b/tasks",
                "row_count": 9,
                "files": [_file("data/openhands/model-b/tasks/part-0.parquet")],
            },
            {
                "partition_name": "swe-agent/model-b/tasks",
                "row_count": 11,
                "files": [_file("data/swe-agent/model-b/tasks/part-0.parquet")],
            },
        ],
        "language_filter": ["JavaScript", "TypeScript"],
        "candidate_artifact": "resolution-candidates.jsonl",
        "candidate_sha256": "2" * 64,
        "candidate_bytes": 2_000,
        "queue_artifact": "resolution-queue.jsonl",
        "queue_sha256": "3" * 64,
        "queue_bytes": 3_000,
        "target_row_count": 20,
        "ineligible_row_count": 3,
        "already_known_row_count": 2,
        "candidate_row_count": 5,
        "candidate_unique_count": 4,
        "candidate_resolved_count": 3,
        "candidate_unresolved_count": 2,
        "queued_fanout_row_count": 10,
        "queue_unique_count": 8,
        "conflict_count": 0,
    }


def _resolution_recovery_with_empty_outputs() -> dict[str, Any]:
    payload = _resolution_recovery()
    payload.update(
        {
            "ineligible_row_count": 18,
            "candidate_row_count": 0,
            "candidate_unique_count": 0,
            "candidate_resolved_count": 0,
            "candidate_unresolved_count": 0,
            "queued_fanout_row_count": 0,
            "queue_unique_count": 0,
            "candidate_bytes": 0,
            "queue_bytes": 0,
        }
    )
    return payload


def _resolution_canary_execution() -> dict[str, Any]:
    return {
        "schema_version": "nodelm.resolution-canary-execution/v1",
        "execution_status": "PASS",
        "admission_status": "PASS",
        "admission_blocker": None,
        "code_commit": "a" * 40,
        "recovery_manifest_sha256": "1" * 64,
        "workset_manifest_sha256": "2" * 64,
        "workset_manifest_bytes": 1_000,
        "workset_sha256": "3" * 64,
        "workset_bytes": 2_000,
        "image_lock_sha256": "4" * 64,
        "image_lock_bytes": 3_000,
        "evaluator_repository_id": "SWE-rebench/SWE-rebench-V2",
        "evaluator_revision": "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d",
        "evaluator_log_parsers_sha256": "5" * 64,
        "evaluator_script_sha256": "6" * 64,
        "evaluator_constants_sha256": "8" * 64,
        "sandbox_backend": "rootless-podman",
        "sandbox_network": "none",
        "sandbox_cpus_per_attempt": 2,
        "sandbox_memory_per_attempt": "4g",
        "results_artifact": "canary.results.jsonl",
        "results_sha256": "7" * 64,
        "results_bytes": 4_000,
        "case_count": 4,
        "passed_case_count": 4,
        "failed_case_count": 0,
        "transfer_control_count": 2,
        "transfer_label_agreement_count": 2,
        "evaluation_request_count": 2,
        "evaluation_resolved_count": 1,
        "evaluation_unresolved_count": 1,
        "image_count": 4,
        "failure_counts_by_reason": {},
    }


@pytest.mark.parametrize(
    ("model", "payload_factory"),
    [
        (SnapshotMaterializationManifestV1, _materialization_v1),
        (SnapshotMaterializationManifestV2, _materialization_v2),
        (TaskProvenanceProjectionManifestV1, _task_projection),
        (NormalizationManifestV2, _normalization),
        (ResolutionRecoveryManifestV1, _resolution_recovery),
        (ResolutionCanaryExecutionManifestV1, _resolution_canary_execution),
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
        (ResolutionRecoveryManifestV1, _resolution_recovery),
        (ResolutionCanaryExecutionManifestV1, _resolution_canary_execution),
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
        (ResolutionRecoveryManifestV1, _resolution_recovery, "candidate_row_count", True),
        (ResolutionRecoveryManifestV1, _resolution_recovery, "queue_bytes", 3_000.0),
        (
            ResolutionCanaryExecutionManifestV1,
            _resolution_canary_execution,
            "case_count",
            True,
        ),
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
        (
            ResolutionRecoveryManifestV1,
            _resolution_recovery,
            {"ineligible_row_count": 4},
            "target recovery accounting",
        ),
        (
            ResolutionRecoveryManifestV1,
            _resolution_recovery,
            {"candidate_resolved_count": 4},
            "candidate outcome counts",
        ),
        (
            ResolutionRecoveryManifestV1,
            _resolution_recovery,
            {"candidate_unique_count": 6},
            "candidate_unique_count",
        ),
        (
            ResolutionRecoveryManifestV1,
            _resolution_recovery,
            {"queue_unique_count": 11},
            "queue_unique_count",
        ),
        (
            ResolutionRecoveryManifestV1,
            _resolution_recovery,
            {"conflict_count": 1},
            "conflict_count",
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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"derivation_status": "FAIL"}, "derivation_status"),
        ({"admission_status": "PASS"}, "admission_status"),
        ({"admission_blocker": "none"}, "admission_blocker"),
    ],
)
def test_resolution_recovery_manifest_is_never_training_admissible(
    updates: dict[str, Any],
    message: str,
) -> None:
    payload = _resolution_recovery()
    payload.update(updates)

    with pytest.raises(ValidationError, match=message):
        ResolutionRecoveryManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"passed_case_count": 3}, "case_count"),
        ({"transfer_label_agreement_count": 1}, "PASS canary"),
        ({"evaluation_unresolved_count": 0}, "PASS canary"),
        (
            {
                "execution_status": "FAIL",
                "admission_status": "PASS",
                "admission_blocker": None,
                "passed_case_count": 3,
                "failed_case_count": 1,
                "failure_counts_by_reason": {"oracle_failed": 1},
            },
            "admission status",
        ),
    ],
)
def test_resolution_canary_execution_fails_closed_on_inconsistent_admission(
    updates: dict[str, Any],
    message: str,
) -> None:
    payload = _resolution_canary_execution()
    payload.update(updates)

    with pytest.raises(ValidationError, match=message):
        ResolutionCanaryExecutionManifestV1.model_validate(payload)


def test_resolution_recovery_manifest_rejects_unsupported_language_filter() -> None:
    payload = _resolution_recovery()
    payload["language_filter"] = ["Python"]

    with pytest.raises(ValidationError):
        ResolutionRecoveryManifestV1.model_validate(payload)


@pytest.mark.parametrize("field", ["candidate_bytes", "queue_bytes"])
def test_resolution_recovery_manifest_requires_bytes_for_nonempty_artifacts(field: str) -> None:
    payload = _resolution_recovery()
    payload[field] = 0

    with pytest.raises(ValidationError, match="bytes must be zero exactly when"):
        ResolutionRecoveryManifestV1.model_validate(payload)


@pytest.mark.parametrize("field", ["candidate_bytes", "queue_bytes"])
def test_resolution_recovery_manifest_rejects_bytes_for_empty_artifacts(field: str) -> None:
    payload = _resolution_recovery_with_empty_outputs()
    payload[field] = 1

    with pytest.raises(ValidationError, match="bytes must be zero exactly when"):
        ResolutionRecoveryManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "candidate_row_count": 0,
            "candidate_unique_count": 1,
            "candidate_resolved_count": 0,
            "candidate_unresolved_count": 0,
            "queued_fanout_row_count": 15,
        },
        {"queue_unique_count": 0},
    ],
)
def test_resolution_recovery_manifest_rejects_inconsistent_zero_counts(
    updates: dict[str, Any],
) -> None:
    payload = _resolution_recovery()
    payload.update(updates)

    with pytest.raises(ValidationError, match="unique count must be zero exactly when"):
        ResolutionRecoveryManifestV1.model_validate(payload)


def test_resolution_recovery_manifest_binds_all_target_rows() -> None:
    payload = _resolution_recovery()
    payload["target_partitions"][0]["row_count"] = 8

    with pytest.raises(ValidationError, match="target partition rows"):
        ResolutionRecoveryManifestV1.model_validate(payload)


def test_resolution_recovery_manifest_is_immutable() -> None:
    manifest = ResolutionRecoveryManifestV1.model_validate(_resolution_recovery())

    with pytest.raises(ValidationError, match="frozen"):
        manifest.target_row_count = 21


def test_resolution_recovery_manifest_allows_terminal_empty_outputs() -> None:
    payload = _resolution_recovery_with_empty_outputs()

    manifest = ResolutionRecoveryManifestV1.model_validate(payload)

    assert manifest.derivation_status == "PASS"
    assert manifest.admission_status == "BLOCKED"


def test_resolution_partition_input_rejects_bool_row_count() -> None:
    payload = _resolution_recovery()
    payload["target_partitions"][0]["row_count"] = True

    with pytest.raises(ValidationError):
        ResolutionRecoveryManifestV1.model_validate(payload)
