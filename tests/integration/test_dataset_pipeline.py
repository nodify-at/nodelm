from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from nodelm.artifacts import canonical_json_bytes, file_identity, write_immutable_json
from nodelm.cli import app
from nodelm.datasets.lineage import build_snapshot_transfer_receipt, capture_snapshot_identity
from nodelm.datasets.partitions import AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION
from nodelm.datasets.registry import DatasetRegistry
from nodelm.datasets.seals import (
    AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION,
    SnapshotSeal,
)
from nodelm.decontamination.split import AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256
from nodelm.provenance.gold import AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256


def test_snapshot_materializes_normalizes_splits_and_builds_a_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    partition_directory = snapshot / "data" / "fixture" / "model" / "tasks"
    partition_directory.mkdir(parents=True)
    (partition_directory / "traces.jsonl").write_text(
        json.dumps(
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-1",
                "repo": "acme/widget",
                "trajectory_id": "rollout-1",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "inspect and patch"}],
                "model_patch": "diff --git a/a.ts b/a.ts",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (partition_directory / "traces.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "hf_dataset_name": "owner/fixture-tasks",
                    "instance_id": "acme__widget-unknown",
                    "repo": "acme/widget",
                    "trajectory_id": "rollout-unknown",
                    "resolved": -1,
                    "trajectory": [{"role": "assistant", "content": "inspect unknown"}],
                }
            )
            + "\n"
        )
        for row in (
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-1",
                "repo": "acme/widget",
                "trajectory_id": "rollout-1",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "inspect and patch"}],
                "model_patch": "diff --git a/a.ts b/a.ts",
            },
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-conflict",
                "repo": "acme/widget",
                "trajectory_id": "rollout-conflict",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "conflict A"}],
            },
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-conflict",
                "repo": "acme/widget",
                "trajectory_id": "rollout-conflict",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "conflict A"}],
            },
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-conflict",
                "repo": "acme/widget",
                "trajectory_id": "rollout-conflict",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "conflict B"}],
            },
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-mixed",
                "repo": "acme/widget",
                "trajectory_id": "rollout-mixed",
                "resolved": -1,
                "trajectory": [{"role": "assistant", "content": "mixed unknown"}],
            },
            {
                "hf_dataset_name": "owner/fixture-tasks",
                "instance_id": "acme__widget-mixed",
                "repo": "acme/widget",
                "trajectory_id": "rollout-mixed",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "mixed valid"}],
            },
        ):
            stream.write(json.dumps(row) + "\n")
    task_snapshot = tmp_path / "task-snapshot"
    task_snapshot.mkdir()
    (task_snapshot / "tasks.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "acme__widget-1",
                "repo": "acme/widget",
                "base_commit": "b" * 40,
                "license": "MIT",
                "language": "TypeScript",
                "problem_statement": "repair the fixture widget retry implementation",
                "patch": "gold patch is join-excluded",
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "acme__widget-unknown",
                "repo": "acme/widget",
                "base_commit": "c" * 40,
                "license": "MIT",
                "language": "ts",
                "problem_statement": "unknown rows remain outside normalized sample v1",
                "patch": "second gold patch is join-excluded",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (task_snapshot / "tasks.jsonl").open("a", encoding="utf-8") as stream:
        for instance_id, base_commit in (
            ("acme__widget-conflict", "d" * 40),
            ("acme__widget-mixed", "e" * 40),
        ):
            stream.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "acme/widget",
                        "base_commit": base_commit,
                        "license": "MIT",
                        "language": "TypeScript",
                        "problem_statement": "join-excluded fixture task",
                        "patch": "join-excluded fixture gold",
                    }
                )
                + "\n"
            )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture-traces\n"
        "    repository_id: owner/fixture-traces\n"
        f"    revision: {'a' * 40}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        "    observed_rows: 8\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n"
        "  - name: fixture-tasks\n"
        "    repository_id: owner/fixture-tasks\n"
        f"    revision: {'c' * 40}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        "    observed_rows: 4\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n",
        encoding="utf-8",
    )
    registry_identity = file_identity(registry)
    snapshot_identity = capture_snapshot_identity(snapshot)
    receipt = build_snapshot_transfer_receipt(
        source=DatasetRegistry.load(registry).by_name("fixture-traces"),
        registry_sha256=registry_identity[0],
        registry_bytes=registry_identity[1],
        snapshot=snapshot_identity,
    )
    receipt_path = tmp_path / "fixture.transfer.json"
    receipt_result = write_immutable_json(receipt_path, receipt.model_dump(mode="json"))
    task_snapshot_identity = capture_snapshot_identity(task_snapshot)
    task_receipt = build_snapshot_transfer_receipt(
        source=DatasetRegistry.load(registry).by_name("fixture-tasks"),
        registry_sha256=registry_identity[0],
        registry_bytes=registry_identity[1],
        snapshot=task_snapshot_identity,
    )
    task_receipt_path = tmp_path / "fixture-tasks.transfer.json"
    write_immutable_json(task_receipt_path, task_receipt.model_dump(mode="json"))
    partition_contract = tmp_path / "partitions.yaml"
    partition_contract.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nodelm.trace-partition-contract/v1",
                "source_name": "fixture-traces",
                "source_repository_id": "owner/fixture-traces",
                "source_revision": "a" * 40,
                "sealed_registry_sha256": registry_identity[0],
                "transfer_receipt_sha256": receipt_result.digest,
                "snapshot_sha256": snapshot_identity.snapshot_sha256,
                "snapshot_file_count": 1,
                "partitions": [
                    {
                        "name": "fixture/model/tasks",
                        "harness": "fixture",
                        "generating_model": "source-label:model",
                        "upstream_source": "tasks",
                        "row_dataset_name": "owner/fixture-tasks",
                        "normalization_status": "PASS",
                        "task_source_name": "fixture-tasks",
                        "task_source_revision": "c" * 40,
                        "file_patterns": ["data/fixture/model/tasks/*.jsonl"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION,
        ("fixture-traces", "a" * 40),
        file_identity(partition_contract)[0],
    )
    monkeypatch.setitem(
        AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION,
        ("fixture-traces", "a" * 40),
        SnapshotSeal(
            transfer_receipt_sha256=file_identity(receipt_path)[0],
            snapshot_sha256=snapshot_identity.snapshot_sha256,
            snapshot_file_count=len(snapshot_identity.files),
        ),
    )
    monkeypatch.setitem(
        AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION,
        ("fixture-tasks", "c" * 40),
        SnapshotSeal(
            transfer_receipt_sha256=file_identity(task_receipt_path)[0],
            snapshot_sha256=task_snapshot_identity.snapshot_sha256,
            snapshot_file_count=len(task_snapshot_identity.files),
        ),
    )
    raw = tmp_path / "raw.jsonl"
    task_provenance = tmp_path / "tasks.safe.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    split = tmp_path / "split.json"
    benchmark = tmp_path / "public-benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "benchmark_id": "public-unrelated-1",
                "task": "replace an unrelated database migration planner",
                "patch": "diff --git a/migration.sql b/migration.sql\n+CREATE INDEX unrelated;",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot.json"
    runner = CliRunner()

    materialize = runner.invoke(
        app,
        [
            "datasets",
            "materialize",
            "--source",
            "fixture-traces",
            "--snapshot",
            str(snapshot),
            "--output",
            str(raw),
            "--partition-contract",
            str(partition_contract),
            "--transfer-receipt",
            str(receipt_path),
            "--partition",
            "fixture/model/tasks",
            "--config",
            str(registry),
        ],
    )
    assert materialize.exit_code == 0, materialize.output
    materialization_manifest = tmp_path / "raw.manifest.json"
    materialization_payload = json.loads(materialization_manifest.read_text(encoding="utf-8"))
    assert materialization_payload["schema_version"] == "nodelm.snapshot-materialization/v2"
    assert materialization_payload["partition_name"] == "fixture/model/tasks"
    assert materialization_payload["harness"] == "fixture"
    assert materialization_payload["generating_model"] == "source-label:model"
    assert materialization_payload["upstream_source"] == "tasks"
    assert materialization_payload["output_bytes"] == raw.stat().st_size

    projection = runner.invoke(
        app,
        [
            "datasets",
            "project-task-provenance",
            "--source",
            "fixture-tasks",
            "--snapshot",
            str(task_snapshot),
            "--output",
            str(task_provenance),
            "--transfer-receipt",
            str(task_receipt_path),
            "--config",
            str(registry),
        ],
    )
    assert projection.exit_code == 0, projection.output
    safe_task_payloads = [
        json.loads(line) for line in task_provenance.read_text(encoding="utf-8").splitlines()
    ]
    assert {payload["repository_license"] for payload in safe_task_payloads} == {"MIT"}
    assert "gold patch" not in task_provenance.read_text(encoding="utf-8")
    assert "problem_statement" not in task_provenance.read_text(encoding="utf-8")

    task_manifest_path = tmp_path / "tasks.safe.manifest.json"
    normalize_arguments = [
        "datasets",
        "normalize",
        "--source",
        "fixture-traces",
        "--snapshot",
        str(snapshot),
        "--input",
        str(raw),
        "--output",
        str(normalized),
        "--materialization-manifest",
        str(materialization_manifest),
        "--partition-contract",
        str(partition_contract),
        "--transfer-receipt",
        str(receipt_path),
        "--expect-harness",
        "fixture",
        "--expect-generating-model",
        "source-label:model",
        "--task-provenance",
        str(task_provenance),
        "--task-provenance-manifest",
        str(task_manifest_path),
        "--task-transfer-receipt",
        str(task_receipt_path),
        "--task-snapshot",
        str(task_snapshot),
        "--config",
        str(registry),
    ]

    def assert_manifest_tampering_rejected(
        path: Path,
        field: str,
        value: object,
    ) -> None:
        original = path.read_bytes()
        payload = json.loads(original)
        payload[field] = value
        path.write_bytes(canonical_json_bytes(payload))
        try:
            result = runner.invoke(app, normalize_arguments)
        finally:
            path.write_bytes(original)
        assert result.exit_code == 2, result.output

    task_manifest_payload = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    for field, value in (
        ("source_repository_id", "attacker/fixture-tasks"),
        ("registry_bytes", task_manifest_payload["registry_bytes"] + 1),
        ("file_patterns", ["tasks.jsonl"]),
    ):
        assert_manifest_tampering_rejected(task_manifest_path, field, value)
    assert_manifest_tampering_rejected(
        materialization_manifest,
        "file_patterns",
        ["data/fixture/model/tasks/attacker-*.jsonl"],
    )

    normalize = runner.invoke(app, normalize_arguments)
    assert normalize.exit_code == 0, normalize.output
    normalization_manifest = json.loads(
        (tmp_path / "normalized.manifest.json").read_text(encoding="utf-8")
    )
    assert normalization_manifest["accepted_count"] == 1
    assert normalization_manifest["input_row_count"] == 8
    assert normalization_manifest["rejected_count"] == 7
    assert normalization_manifest["unique_rollout_key_count"] == 4
    assert normalization_manifest["duplicate_trace_row_count"] == 1
    assert normalization_manifest["conflicting_rollout_identity_count"] == 2
    assert normalization_manifest["conflicting_rollout_row_count"] == 5
    assert normalization_manifest["rejection_counts_by_code"] == {
        "conflicting_rollout_identity": 5,
        "duplicate_trace_row": 1,
        "unknown_resolution": 1,
    }
    normalization_rejections = [
        json.loads(line)
        for line in (tmp_path / "normalized.rejections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [rejection["row_index"] for rejection in normalization_rejections] == list(range(1, 8))
    assert normalization_rejections[5]["cause_code"] == "unknown_resolution"
    assert all(len(rejection["raw_row_sha256"]) == 64 for rejection in normalization_rejections)
    normalized_rows = [
        json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()
    ]
    assert len({row["sample_id"] for row in normalized_rows}) == 1

    split_result = runner.invoke(
        app,
        [
            "split",
            "build",
            "--input",
            str(normalized),
            "--output",
            str(split),
            "--task-metadata",
            str(task_snapshot / "tasks.jsonl"),
            "--benchmark",
            str(benchmark),
            "--near-duplicate-threshold",
            "0.85",
            "--seed",
            "0",
            "--evaluation-fraction",
            "0.1",
        ],
    )
    assert split_result.exit_code == 0, split_result.output
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    assert split_payload["decontamination"]["sample_count"] == 1
    assert split_payload["decontamination"]["benchmark_entry_count"] == 1
    assert split_payload["repositories"]["excluded"] == []
    assert "gold patch is join-excluded" not in split.read_text(encoding="utf-8")
    monkeypatch.setitem(
        AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256,
        file_identity(normalized)[0],
        file_identity(split)[0],
    )

    normalized_identity = file_identity(normalized)
    normalization_manifest_path = tmp_path / "normalized.manifest.json"
    normalization_identity = file_identity(normalization_manifest_path)
    findings_path = tmp_path / "gold.findings.jsonl"
    findings_path.write_bytes(b"")
    attestation_path = tmp_path / "oracle-isolation.json"
    attestation_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.oracle-isolation-attestation/v1",
                "method_version": "nodelm.oracle-isolation-review/v1",
                "status": "PASS",
                "source_name": "fixture-traces",
                "source_revision": "a" * 40,
                "partition_name": "fixture/model/tasks",
                "normalized_sha256": normalized_identity[0],
                "normalized_bytes": normalized_identity[1],
                "covered_sample_count": 1,
            }
        )
    )
    findings_identity = file_identity(findings_path)
    attestation_identity = file_identity(attestation_path)
    gold_audit = tmp_path / "gold-exposure.audit.json"
    gold_audit.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.gold-exposure-audit/v1",
                "method_version": "nodelm.gold-exposure-audit-method/v1",
                "status": "PASS",
                "normalization_manifest_artifact": normalization_manifest_path.name,
                "normalization_manifest_sha256": normalization_identity[0],
                "normalization_manifest_bytes": normalization_identity[1],
                "normalized_artifact": normalized.name,
                "normalized_sha256": normalized_identity[0],
                "normalized_bytes": normalized_identity[1],
                "expected_sample_count": 1,
                "audited_sample_count": 1,
                "structural_scan": {"status": "PASS", "finding_count": 0},
                "oracle_isolation": {
                    "status": "PASS",
                    "attestation_artifact": attestation_path.name,
                    "attestation_sha256": attestation_identity[0],
                    "attestation_bytes": attestation_identity[1],
                    "covered_sample_count": 1,
                },
                "findings_artifact": findings_path.name,
                "findings_sha256": findings_identity[0],
                "findings_bytes": findings_identity[1],
            }
        )
    )
    monkeypatch.setitem(
        AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256,
        normalized_identity[0],
        file_identity(gold_audit)[0],
    )

    pilot_result = runner.invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(normalized),
            "--output",
            str(pilot),
            "--normalization-manifest",
            str(normalization_manifest_path),
            "--gold-exposure-audit",
            str(gold_audit),
            "--split-manifest",
            str(split),
            "--registry",
            str(registry),
        ],
    )
    assert pilot_result.exit_code == 0, pilot_result.output

    payload = json.loads(pilot.read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 1
    pilot_samples = tmp_path / "pilot.samples.jsonl"
    assert json.loads(pilot_samples.read_text(encoding="utf-8"))["trajectory"]
    assert "gold patch" not in pilot_samples.read_text(encoding="utf-8")
