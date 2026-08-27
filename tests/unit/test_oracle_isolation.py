from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nodelm.artifacts import ArtifactCollisionError, canonical_json_bytes, file_identity
from nodelm.cli import app
from nodelm.models import NormalizedSample, stable_model_id
from nodelm.provenance.gold import (
    AUTHORIZED_ORACLE_ATTESTATION_SHA256_BY_NORMALIZED_SHA256,
)
from nodelm.provenance.manifests import TASK_PROVENANCE_SAFE_FIELDS
from nodelm.provenance.normalize import normalize_sample
from nodelm.provenance.oracle_isolation import (
    inspect_recorded_model_context,
    review_oracle_isolation_artifacts,
)
from nodelm.provenance.pipeline import normalization_evidence_lineage

SOURCE_REVISION = "ed95cef24df8d8bd79b4ceb0192cb420fde06521"
TASK_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
PARTITION = "openhands/minimax_m25/swe-rebench-v2"
RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_oracle_isolation_reviews.sh"


def _raw_trace(
    *, prompt: str = "Fix the reported bug", include_reference_patch: bool = True
) -> dict[str, object]:
    row: dict[str, object] = {
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "license": "MIT",
        "language": "TypeScript",
        "trajectory_id": "rollout-1",
        "resolved": 1,
        "trajectory": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "I will inspect the repository.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "execute_bash",
                            "arguments": '{"command":"sed -n 1,80p src/a.ts"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "export const value = 1;"},
        ],
        "tools": [
            '{"type":"function","function":{"name":"execute_bash",'
            '"parameters":{"properties":{"command":{"type":"string"}}}}}'
        ],
        "model_patch": "diff --git a/src/a.ts b/src/a.ts\n-old\n+new",
    }
    if include_reference_patch:
        row["metadata"] = {
            "reference_patch": {
                "patch": "diff --git a/src/a.ts b/src/a.ts\n-old\n+new",
                "num_modified_files": 1,
                "num_modified_lines": 2,
            }
        }
    return row


def test_recorded_context_review_allows_separate_reference_patch_metadata() -> None:
    inspection = inspect_recorded_model_context(_raw_trace())

    assert inspection.reference_patch_present
    assert inspection.reason_codes == ()


def test_recorded_context_review_rejects_reference_patch_in_initial_prompt() -> None:
    gold = "diff --git a/src/a.ts b/src/a.ts\n-old\n+new"
    inspection = inspect_recorded_model_context(
        _raw_trace(prompt=f"Use this solution exactly:\n{gold}")
    )

    assert inspection.reason_codes == ("reference_patch_in_initial_prompt",)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row, patch: row["trajectory"].append(
            {"role": "tool", "content": f"late tool result:\n{patch}"}
        ),
        lambda row, patch: row["trajectory"].append(
            {"role": "future-input-role", "content": f"late context:\n{patch}"}
        ),
        lambda row, patch: row.__setitem__(
            "tools",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": f"Do not expose: {patch}",
                    },
                }
            ],
        ),
    ],
)
def test_recorded_context_review_rejects_patch_in_every_model_input_boundary(
    mutate: object,
) -> None:
    row = _raw_trace()
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    reference_patch = metadata["reference_patch"]
    assert isinstance(reference_patch, dict)
    patch = reference_patch["patch"]
    assert isinstance(patch, str)
    assert callable(mutate)

    mutate(row, patch)

    inspection = inspect_recorded_model_context(row)

    assert inspection.reason_codes == ("reference_patch_in_initial_prompt",)


def test_recorded_context_review_excludes_genuine_assistant_output() -> None:
    row = _raw_trace()
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    reference_patch = metadata["reference_patch"]
    assert isinstance(reference_patch, dict)
    patch = reference_patch["patch"]
    assert isinstance(patch, str)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory.append({"role": "assistant", "content": patch})

    assert inspect_recorded_model_context(row).reason_codes == ()


def test_recorded_context_review_requires_structured_reference_patch() -> None:
    row = _raw_trace()
    row["metadata"] = {}

    inspection = inspect_recorded_model_context(row)

    assert inspection.reference_patch_present is False
    assert inspection.reason_codes == ("unsupported_reference_patch_location",)


def test_recorded_context_review_rejects_gold_fields_in_tool_definitions() -> None:
    row = _raw_trace()
    row["tools"] = [
        '{"type":"function","function":{"name":"read_oracle",'
        '"parameters":{"properties":{"gold_patch":{"type":"string"}}}}}'
    ]

    inspection = inspect_recorded_model_context(row)

    assert inspection.reason_codes == ("gold_field_in_recorded_model_context",)


def test_recorded_context_review_rejects_unapproved_gold_field_location() -> None:
    row = _raw_trace()
    row["reference_patch"] = "hidden"

    inspection = inspect_recorded_model_context(row)

    assert inspection.reason_codes == ("unsupported_reference_patch_location",)


def test_recorded_context_review_rejects_malformed_reference_patch_boundary() -> None:
    row = _raw_trace()
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    metadata["reference_patch"] = "not a structured reference patch"

    inspection = inspect_recorded_model_context(row)

    assert inspection.reason_codes == ("unsupported_reference_patch_location",)


def _write_review_inputs(
    tmp_path: Path,
    *,
    prompt: str = "Fix the reported bug",
    include_reference_patch: bool = True,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    raw_row = _raw_trace(prompt=prompt, include_reference_patch=include_reference_patch)
    raw = tmp_path / "leaf.raw.jsonl"
    raw.write_bytes(canonical_json_bytes(raw_row))
    raw_identity = file_identity(raw)
    materialization_manifest = tmp_path / "leaf.raw.manifest.json"
    materialization_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.snapshot-materialization/v2",
                "status": "PASS",
                "source_name": "open-swe-traces",
                "source_repository_id": "nvidia/Open-SWE-Traces",
                "source_revision": SOURCE_REVISION,
                "registry_sha256": "1" * 64,
                "row_count": 1,
                "max_rows": None,
                "output": raw.name,
                "output_sha256": raw_identity[0],
                "output_bytes": raw_identity[1],
                "file_patterns": ["data/openhands/minimax_m25/swe-rebench-v2/*.parquet"],
                "files": [
                    {
                        "path": ("data/openhands/minimax_m25/swe-rebench-v2/part.parquet"),
                        "sha256": "2" * 64,
                        "bytes": 2,
                    }
                ],
                "materialization_scope": "complete-partition",
                "partition_contract_sha256": "3" * 64,
                "partition_contract_bytes": 3,
                "transfer_receipt_sha256": "4" * 64,
                "transfer_receipt_bytes": 4,
                "partition_name": PARTITION,
                "harness": "openhands",
                "generating_model": "source-label:minimax_m25",
                "upstream_source": "swe-rebench-v2",
                "row_dataset_name": "nebius/SWE-rebench-V2",
                "task_source_name": "swe-rebench-v2",
                "task_source_revision": TASK_REVISION,
                "normalization_status": "PASS",
            }
        )
    )
    materialization_identity = file_identity(materialization_manifest)
    task_provenance = tmp_path / "swe-rebench-v2.safe.jsonl"
    task_provenance.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.task-provenance/v1",
                "source_dataset": "swe-rebench-v2",
                "source_dataset_revision": TASK_REVISION,
                "instance_id": "acme__widget-1",
                "repository": "acme/widget",
                "base_commit": "b" * 40,
                "repository_license": "MIT",
                "language": "TypeScript",
            }
        )
    )
    task_identity = file_identity(task_provenance)
    task_rejections = tmp_path / "swe-rebench-v2.safe.rejections.jsonl"
    task_rejections.write_bytes(b"")
    task_provenance_manifest = tmp_path / "swe-rebench-v2.safe.manifest.json"
    task_provenance_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.task-provenance-projection/v1",
                "status": "PASS",
                "source_name": "swe-rebench-v2",
                "source_repository_id": "princeton-nlp/SWE-bench",
                "source_revision": TASK_REVISION,
                "registry_sha256": "1" * 64,
                "registry_bytes": 1,
                "transfer_receipt_sha256": "7" * 64,
                "transfer_receipt_bytes": 7,
                "snapshot_sha256": "9" * 64,
                "projection_scope": "complete-snapshot",
                "file_patterns": [],
                "files": [{"path": "data/tasks.parquet", "sha256": "a" * 64, "bytes": 1}],
                "safe_fields": list(TASK_PROVENANCE_SAFE_FIELDS),
                "admitted_count": 1,
                "rejected_count": 0,
                "rejection_counts_by_code": {},
                "output": task_provenance.name,
                "output_sha256": task_identity[0],
                "output_bytes": task_identity[1],
                "rejection_artifact": task_rejections.name,
                "rejection_sha256": file_identity(task_rejections)[0],
                "rejection_bytes": 0,
            }
        )
    )
    task_manifest_identity = file_identity(task_provenance_manifest)
    evidence_lineage = normalization_evidence_lineage(
        materialization_manifest_sha256=materialization_identity[0],
        partition_name=PARTITION,
        upstream_source="swe-rebench-v2",
        task_source_name="swe-rebench-v2",
        task_source_revision=TASK_REVISION,
        task_provenance_sha256=task_identity[0],
    )
    sample = normalize_sample(
        raw_row,
        source_dataset="open-swe-traces",
        source_revision=SOURCE_REVISION,
        harness="openhands",
        generating_model="source-label:minimax_m25",
        lineage=(
            f"hf-dataset:nvidia/Open-SWE-Traces@{SOURCE_REVISION}",
            "instance:acme__widget-1",
            f"raw-row:{stable_model_id(raw_row)}",
            "task-metadata:acme__widget-1",
            *evidence_lineage,
        ),
    )
    normalized = tmp_path / "leaf.normalized.jsonl"
    normalized.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    normalized_identity = file_identity(normalized)
    normalization_manifest = tmp_path / "leaf.normalized.manifest.json"
    normalization_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.normalization-manifest/v2",
                "status": "PASS",
                "source_name": "open-swe-traces",
                "source_repository_id": "nvidia/Open-SWE-Traces",
                "source_revision": SOURCE_REVISION,
                "partition_name": PARTITION,
                "harness": "openhands",
                "generating_model": "source-label:minimax_m25",
                "upstream_source": "swe-rebench-v2",
                "row_dataset_name": "nebius/SWE-rebench-V2",
                "input_sha256": raw_identity[0],
                "input_bytes": raw_identity[1],
                "registry_sha256": "1" * 64,
                "materialization_manifest_sha256": materialization_identity[0],
                "materialization_manifest_bytes": materialization_identity[1],
                "partition_contract_sha256": "3" * 64,
                "partition_contract_bytes": 3,
                "transfer_receipt_sha256": "4" * 64,
                "transfer_receipt_bytes": 4,
                "task_provenance_sha256": task_identity[0],
                "task_provenance_bytes": task_identity[1],
                "task_provenance_manifest_sha256": task_manifest_identity[0],
                "task_provenance_manifest_bytes": task_manifest_identity[1],
                "task_transfer_receipt_sha256": "7" * 64,
                "task_transfer_receipt_bytes": 7,
                "task_source_name": "swe-rebench-v2",
                "task_source_revision": TASK_REVISION,
                "materialization_replay": "PASS",
                "task_provenance_replay": "PASS",
                "uniqueness_scope": "complete-partition",
                "input_row_count": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "rejection_counts_by_code": {},
                "unique_rollout_key_count": 1,
                "duplicate_trace_row_count": 0,
                "conflicting_rollout_identity_count": 0,
                "conflicting_rollout_row_count": 0,
                "gold_exposure_audit": "NOT RUN",
                "normalized_artifact": normalized.name,
                "normalized_sha256": normalized_identity[0],
                "normalized_bytes": normalized_identity[1],
                "rejection_artifact": "leaf.normalized.rejections.jsonl",
                "rejection_sha256": "8" * 64,
                "rejection_bytes": 0,
            }
        )
    )
    return (
        raw,
        materialization_manifest,
        normalized,
        normalization_manifest,
        task_provenance,
        task_provenance_manifest,
        tmp_path / "leaf.oracle.json",
        tmp_path / "leaf.oracle.findings.jsonl",
    )


def _invoke_review(inputs: tuple[Path, Path, Path, Path, Path, Path, Path, Path]):
    (
        raw,
        materialization,
        normalized,
        normalization,
        task_provenance,
        task_provenance_manifest,
        output,
        findings,
    ) = inputs
    return CliRunner().invoke(
        app,
        [
            "datasets",
            "review-oracle-isolation",
            "--raw-input",
            str(raw),
            "--materialization-manifest",
            str(materialization),
            "--input",
            str(normalized),
            "--normalization-manifest",
            str(normalization),
            "--task-provenance",
            str(task_provenance),
            "--task-provenance-manifest",
            str(task_provenance_manifest),
            "--output",
            str(output),
            "--findings-output",
            str(findings),
        ],
    )


def test_oracle_review_command_emits_bound_pass_attestation(
    tmp_path: Path,
) -> None:
    inputs = _write_review_inputs(tmp_path)

    result = _invoke_review(inputs)

    assert result.exit_code == 0, result.output
    payload = json.loads(inputs[6].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "nodelm.oracle-isolation-attestation/v2"
    assert payload["status"] == "PASS"
    assert payload["raw_row_count"] == payload["covered_sample_count"] == 1
    assert payload["reference_patch_row_count"] == 1
    assert all(check["status"] == "PASS" for check in payload["checks"])
    assert inputs[7].read_bytes() == b""


def test_oracle_review_requires_reference_patch_for_every_row(tmp_path: Path) -> None:
    inputs = _write_review_inputs(tmp_path, include_reference_patch=False)

    result = _invoke_review(inputs)

    assert result.exit_code == 1, result.output
    payload = json.loads(inputs[6].read_text(encoding="utf-8"))
    finding = json.loads(inputs[7].read_text(encoding="utf-8"))
    assert payload["reference_patch_row_count"] == 0
    assert {check["name"]: check["status"] for check in payload["checks"]}[
        "reference-patch-coverage"
    ] == "FAIL"
    assert finding["reason_code"] == "unsupported_reference_patch_location"


@pytest.mark.parametrize(
    "updates",
    [
        {"repository": "other/project"},
        {"repository_license": "Apache-2.0"},
        {"base_commit": "c" * 40},
        {"language": "JavaScript"},
        {"resolved": False},
        {
            "patch_metadata": {
                "sha256": "0" * 64,
                "bytes": 0,
                "added_lines": 0,
                "removed_lines": 0,
                "source_field": "pred_patch",
            }
        },
    ],
)
def test_oracle_review_rejects_each_training_visible_projection_drift(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    inputs = _write_review_inputs(tmp_path)
    normalized_payload = json.loads(inputs[2].read_text(encoding="utf-8"))
    normalized_payload.update(updates)
    normalized_payload.pop("sample_id")
    mutated = NormalizedSample.model_validate(normalized_payload)
    inputs[2].write_bytes(canonical_json_bytes(mutated.model_dump(mode="json")))
    normalized_identity = file_identity(inputs[2])
    manifest_payload = json.loads(inputs[3].read_text(encoding="utf-8"))
    manifest_payload["normalized_sha256"] = normalized_identity[0]
    manifest_payload["normalized_bytes"] = normalized_identity[1]
    inputs[3].write_bytes(canonical_json_bytes(manifest_payload))

    result = _invoke_review(inputs)

    assert result.exit_code == 1, result.output
    finding = json.loads(inputs[7].read_text(encoding="utf-8"))
    assert finding["reason_code"] == "raw_normalized_binding_mismatch"


def test_oracle_review_command_fails_safely_when_prompt_contains_gold(
    tmp_path: Path,
) -> None:
    gold = "diff --git a/src/a.ts b/src/a.ts\n-old\n+new"
    inputs = _write_review_inputs(tmp_path, prompt=f"Copy this answer:\n{gold}")

    result = _invoke_review(inputs)

    assert result.exit_code == 1, result.output
    payload = json.loads(inputs[6].read_text(encoding="utf-8"))
    finding = json.loads(inputs[7].read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert finding["reason_code"] == "reference_patch_in_initial_prompt"
    assert set(finding) == {"row_index", "sample_id", "reason_code", "reason"}
    assert gold not in result.output
    assert gold not in inputs[6].read_text(encoding="utf-8")
    assert gold not in inputs[7].read_text(encoding="utf-8")


def test_oracle_review_replay_completes_with_identical_existing_findings(
    tmp_path: Path,
) -> None:
    inputs = _write_review_inputs(tmp_path)
    first = _invoke_review(inputs)
    assert first.exit_code == 0, first.output
    findings_before = inputs[7].read_bytes()
    inputs[6].unlink()

    result, attestation = review_oracle_isolation_artifacts(
        raw_input=inputs[0],
        materialization_manifest=inputs[1],
        normalized_input=inputs[2],
        normalization_manifest=inputs[3],
        task_provenance=inputs[4],
        task_provenance_manifest=inputs[5],
        output=inputs[6],
        findings_output=inputs[7],
    )

    assert result.created
    assert attestation.status.value == "PASS"
    assert inputs[7].read_bytes() == findings_before


def test_oracle_review_replay_rejects_forged_self_consistent_pass_pair(
    tmp_path: Path,
) -> None:
    gold = "diff --git a/src/a.ts b/src/a.ts\n-old\n+new"
    inputs = _write_review_inputs(tmp_path, prompt=f"Copy this answer:\n{gold}")
    first = _invoke_review(inputs)
    assert first.exit_code == 1, first.output

    # This forged pair is internally consistent and binds every artifact identity,
    # but lies about the review result. Replaying the deterministic writer must
    # collide with its empty findings file instead of trusting the claim.
    forged_findings = b""
    inputs[7].write_bytes(forged_findings)
    forged = json.loads(inputs[6].read_text(encoding="utf-8"))
    forged["status"] = "PASS"
    forged["checks"] = [{"name": check["name"], "status": "PASS"} for check in forged["checks"]]
    forged["findings_sha256"] = file_identity(inputs[7])[0]
    forged["findings_bytes"] = 0
    forged["finding_count"] = 0
    inputs[6].write_bytes(canonical_json_bytes(forged))

    with pytest.raises(ArtifactCollisionError):
        review_oracle_isolation_artifacts(
            raw_input=inputs[0],
            materialization_manifest=inputs[1],
            normalized_input=inputs[2],
            normalization_manifest=inputs[3],
            task_provenance=inputs[4],
            task_provenance_manifest=inputs[5],
            output=inputs[6],
            findings_output=inputs[7],
        )


def test_optimized_terminal_pair_validation_rejects_non_pass_attestation(
    tmp_path: Path,
) -> None:
    gold = "diff --git a/src/a.ts b/src/a.ts\n-old\n+new"
    inputs = _write_review_inputs(tmp_path, prompt=f"Copy this answer:\n{gold}")
    first = _invoke_review(inputs)
    assert first.exit_code == 1, first.output

    script = RUNNER.read_text(encoding="utf-8")
    function_start = script.index("validate_terminal_pair() {")
    python_start = script.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    python_end = script.index("\nPY\n}", python_start)
    validation_program = script[python_start:python_end]
    environment = os.environ | {"PYTHONPATH": str(RUNNER.parents[1] / "src")}
    validation = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            validation_program,
            str(inputs[6]),
            str(inputs[7]),
            str(inputs[0]),
            str(inputs[1]),
            str(inputs[2]),
            str(inputs[3]),
            str(inputs[4]),
            str(inputs[5]),
            PARTITION,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert validation.returncode != 0
    assert "status is not PASS" in validation.stderr


def test_oracle_review_pass_can_enter_gold_audit_only_after_digest_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_review_inputs(tmp_path)
    result = _invoke_review(inputs)
    assert result.exit_code == 0, result.output
    normalized_identity = file_identity(inputs[2])
    monkeypatch.setitem(
        AUTHORIZED_ORACLE_ATTESTATION_SHA256_BY_NORMALIZED_SHA256,
        normalized_identity[0],
        file_identity(inputs[6])[0],
    )
    audit = tmp_path / "leaf.gold-audit.json"
    gold_findings = tmp_path / "leaf.gold-findings.jsonl"

    audit_result = CliRunner().invoke(
        app,
        [
            "datasets",
            "audit-gold-exposure",
            "--input",
            str(inputs[2]),
            "--normalization-manifest",
            str(inputs[3]),
            "--oracle-isolation-attestation",
            str(inputs[6]),
            "--output",
            str(audit),
            "--findings-output",
            str(gold_findings),
        ],
    )

    assert audit_result.exit_code == 0, audit_result.output
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "PASS"
