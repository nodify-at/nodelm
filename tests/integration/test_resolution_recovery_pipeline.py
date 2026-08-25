from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nodelm.artifacts import file_identity, write_immutable_json
from nodelm.cli import app
from nodelm.datasets.lineage import build_snapshot_transfer_receipt, capture_snapshot_identity
from nodelm.datasets.partitions import AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION
from nodelm.datasets.registry import DatasetRegistry
from nodelm.datasets.seals import (
    AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION,
    SnapshotSeal,
)
from nodelm.provenance.manifests import ResolutionRecoveryManifestV1


def _trace(
    instance_id: str,
    rollout_id: str,
    patch: str,
    *,
    resolved: int,
    language: str,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "trajectory_id": rollout_id,
        "resolved": resolved,
        "language": language,
        "hf_dataset_name": "owner/fixture-tasks",
        "trajectory": [{"golden_patch": "MUST_NOT_CROSS_RECOVERY_PROJECTION"}],
        "metadata": {
            "model_patch": {"patch": patch},
            "reference_patch": {"patch": "MUST_NOT_CROSS_RECOVERY_PROJECTION"},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_resolution_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    labeled_rows: list[dict[str, Any]],
    openhands_target_rows: list[dict[str, Any]],
    sweagent_target_rows: list[dict[str, Any]],
) -> tuple[Any, Path, Path, Path]:
    trace_revision = "a" * 40
    task_revision = "c" * 40
    snapshot = tmp_path / "snapshot"
    _write_jsonl(
        snapshot / "data/openhands/teacher/tasks/part.jsonl",
        labeled_rows,
    )
    _write_jsonl(
        snapshot / "data/openhands/qwen36/tasks/part.jsonl",
        openhands_target_rows,
    )
    _write_jsonl(
        snapshot / "data/sweagent/qwen36/tasks/part.jsonl",
        sweagent_target_rows,
    )

    observed_rows = len(labeled_rows) + len(openhands_target_rows) + len(sweagent_target_rows)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture-traces\n"
        "    repository_id: owner/fixture-traces\n"
        f"    revision: {trace_revision}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-25T00:00:00Z'\n"
        f"    observed_rows: {observed_rows}\n"
        "    evidence_urls: [https://example.invalid/traces]\n"
        "    status: PASS\n"
        "  - name: fixture-tasks\n"
        "    repository_id: owner/fixture-tasks\n"
        f"    revision: {task_revision}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-25T00:00:00Z'\n"
        "    observed_rows: 4\n"
        "    evidence_urls: [https://example.invalid/tasks]\n"
        "    status: PASS\n",
        encoding="utf-8",
    )
    registry_identity = file_identity(registry_path)
    registry = DatasetRegistry.load(registry_path)
    snapshot_identity = capture_snapshot_identity(snapshot)
    receipt = build_snapshot_transfer_receipt(
        source=registry.by_name("fixture-traces"),
        registry_sha256=registry_identity[0],
        registry_bytes=registry_identity[1],
        snapshot=snapshot_identity,
    )
    receipt_path = tmp_path / "traces.transfer.json"
    receipt_result = write_immutable_json(receipt_path, receipt.model_dump(mode="json"))
    partitions = [
        ("openhands/teacher/tasks", "openhands", "teacher"),
        ("openhands/qwen36/tasks", "openhands", "qwen36"),
        ("sweagent/qwen36/tasks", "sweagent", "qwen36"),
    ]
    contract_path = tmp_path / "partitions.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "nodelm.trace-partition-contract/v1",
                "source_name": "fixture-traces",
                "source_repository_id": "owner/fixture-traces",
                "source_revision": trace_revision,
                "sealed_registry_sha256": registry_identity[0],
                "transfer_receipt_sha256": receipt_result.digest,
                "snapshot_sha256": snapshot_identity.snapshot_sha256,
                "snapshot_file_count": len(snapshot_identity.files),
                "partitions": [
                    {
                        "name": name,
                        "harness": harness,
                        "generating_model": f"source-label:{model}",
                        "upstream_source": "tasks",
                        "row_dataset_name": "owner/fixture-tasks",
                        "normalization_status": "PASS",
                        "task_source_name": "fixture-tasks",
                        "task_source_revision": task_revision,
                        "file_patterns": [f"data/{name}/*.jsonl"],
                    }
                    for name, harness, model in partitions
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION,
        ("fixture-traces", trace_revision),
        file_identity(contract_path)[0],
    )
    monkeypatch.setitem(
        AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION,
        ("fixture-traces", trace_revision),
        SnapshotSeal(
            transfer_receipt_sha256=file_identity(receipt_path)[0],
            snapshot_sha256=snapshot_identity.snapshot_sha256,
            snapshot_file_count=len(snapshot_identity.files),
        ),
    )

    candidates = tmp_path / "recovery.candidates.jsonl"
    queue = tmp_path / "recovery.queue.jsonl"
    manifest = tmp_path / "recovery.manifest.json"
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-resolution-recovery",
            "--source",
            "fixture-traces",
            "--snapshot",
            str(snapshot),
            "--partition-contract",
            str(contract_path),
            "--transfer-receipt",
            str(receipt_path),
            "--labeled-partition",
            "openhands/teacher/tasks",
            "--target-partition",
            "openhands/qwen36/tasks",
            "--target-partition",
            "sweagent/qwen36/tasks",
            "--language",
            "ts",
            "--language",
            "js",
            "--candidates-output",
            str(candidates),
            "--queue-output",
            str(queue),
            "--manifest-output",
            str(manifest),
            "--config",
            str(registry_path),
        ],
    )
    return result, candidates, queue, manifest


def test_build_resolution_recovery_emits_blocked_sidecar_and_deduplicated_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched_patch = "diff --git a/matched.ts b/matched.ts\n+repair();\n"
    queued_patch = "diff --git a/queued.js b/queued.js\n+retry();\n"
    result, candidates, queue, manifest = _run_resolution_recovery(
        tmp_path,
        monkeypatch,
        labeled_rows=[_trace("task-match", "teacher-1", matched_patch, resolved=1, language="ts")],
        openhands_target_rows=[
            _trace("task-match", "target-1", matched_patch, resolved=-1, language="TypeScript"),
            _trace("task-queue", "target-2", queued_patch, resolved=-1, language="js"),
            _trace("task-known", "target-3", queued_patch, resolved=0, language="JavaScript"),
        ],
        sweagent_target_rows=[
            _trace("task-queue", "target-4", queued_patch, resolved=-1, language="javascript"),
            _trace("task-python", "target-5", queued_patch, resolved=-1, language="Python"),
        ],
    )

    assert result.exit_code == 0, result.output
    recovery = ResolutionRecoveryManifestV1.model_validate_json(manifest.read_bytes())
    assert recovery.admission_status == "BLOCKED"
    assert recovery.target_row_count == 5
    assert recovery.ineligible_row_count == 1
    assert recovery.already_known_row_count == 1
    assert recovery.candidate_row_count == 1
    assert recovery.candidate_unique_count == 1
    assert recovery.candidate_resolved_count == 1
    assert recovery.candidate_unresolved_count == 0
    assert recovery.queued_fanout_row_count == 2
    assert recovery.queue_unique_count == 1
    assert recovery.conflict_count == 0
    candidate_rows = [json.loads(line) for line in candidates.read_text().splitlines()]
    queue_rows = [json.loads(line) for line in queue.read_text().splitlines()]
    assert candidate_rows[0]["resolved"] is True
    assert len(queue_rows[0]["target_references"]) == 2
    published = candidates.read_text() + queue.read_text() + manifest.read_text()
    assert "MUST_NOT_CROSS_RECOVERY_PROJECTION" not in published


def test_build_resolution_recovery_conflict_fails_before_publishing_any_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conflicted_patch = "diff --git a/conflicted.ts b/conflicted.ts\n+repair();\n"
    result, candidates, queue, manifest = _run_resolution_recovery(
        tmp_path,
        monkeypatch,
        labeled_rows=[
            _trace(
                "task-conflict",
                "teacher-pass",
                conflicted_patch,
                resolved=1,
                language="ts",
            ),
            _trace(
                "task-conflict",
                "teacher-fail",
                conflicted_patch,
                resolved=0,
                language="TypeScript",
            ),
        ],
        openhands_target_rows=[
            _trace(
                "task-conflict",
                "target-conflict",
                conflicted_patch,
                resolved=-1,
                language="ts",
            )
        ],
        sweagent_target_rows=[
            _trace(
                "task-python",
                "target-python",
                "python patch",
                resolved=-1,
                language="Python",
            )
        ],
    )

    assert result.exit_code != 0
    assert "resolution label conflicts prevent artifact publication" in result.output
    assert "count=1" in result.output
    assert not candidates.exists()
    assert not queue.exists()
    assert not manifest.exists()


def test_build_resolution_recovery_duplicate_target_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched_patch = "diff --git a/duplicate.ts b/duplicate.ts\n+repair();\n"
    duplicate_target = _trace(
        "task-duplicate",
        "duplicate-rollout",
        matched_patch,
        resolved=-1,
        language="ts",
    )
    result, candidates, queue, manifest = _run_resolution_recovery(
        tmp_path,
        monkeypatch,
        labeled_rows=[
            _trace(
                "task-duplicate",
                "teacher-pass",
                matched_patch,
                resolved=1,
                language="TypeScript",
            )
        ],
        openhands_target_rows=[duplicate_target, dict(duplicate_target)],
        sweagent_target_rows=[
            _trace(
                "task-python",
                "target-python",
                "python patch",
                resolved=-1,
                language="Python",
            )
        ],
    )

    assert result.exit_code != 0
    assert "resolution target accounting is incomplete before" in result.output
    assert "publication: accounted=2 target=3" in result.output
    assert not candidates.exists()
    assert not queue.exists()
    assert not manifest.exists()


def test_build_resolution_recovery_task_family_mismatch_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matched_patch = "diff --git a/mismatch.ts b/mismatch.ts\n+repair();\n"
    mislabeled = _trace(
        "task-mismatch",
        "teacher-mismatch",
        matched_patch,
        resolved=1,
        language="TypeScript",
    )
    mislabeled["hf_dataset_name"] = "owner/not-the-bound-task-family"

    result, candidates, queue, manifest = _run_resolution_recovery(
        tmp_path,
        monkeypatch,
        labeled_rows=[mislabeled],
        openhands_target_rows=[
            _trace(
                "task-mismatch",
                "target-mismatch",
                matched_patch,
                resolved=-1,
                language="TypeScript",
            )
        ],
        sweagent_target_rows=[],
    )

    assert result.exit_code != 0
    assert "trace hf_dataset_name does not" in result.output
    assert "match the bound partition task family" in result.output
    assert not candidates.exists()
    assert not queue.exists()
    assert not manifest.exists()
