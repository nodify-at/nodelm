from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nodelm.artifacts import file_identity, write_immutable_json, write_immutable_stream
from nodelm.cli import app
from nodelm.evaluation.resolution_canary import (
    SWE_REBENCH_EVALUATOR_REVISION,
    PinnedContainerImage,
    ResolutionCanaryImageLock,
    ResolutionCanaryOracle,
    ResolutionCanaryPrivateCaseEvidence,
    SWERebenchTask,
    resolution_canary_output,
)
from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.provenance.manifests import (
    ResolutionCanaryExecutionManifestV1,
    ResolutionCanaryWorksetManifestV1,
)
from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
    evaluation_request_from_target_row,
)
from nodelm.provenance.resolution_canary import (
    build_evaluation_case,
    build_transfer_control_case,
)


def _request(suffix: str) -> ResolutionEvaluationRequest:
    return evaluation_request_from_target_row(
        "openhands/qwen36_27b/swe-rebench-v2",
        {
            "instance_id": "owner__repo",
            "trajectory_id": f"rollout-{suffix}",
            "resolved": -1,
            "language": "TypeScript",
            "hf_dataset_name": "nebius/SWE-rebench-V2",
            "metadata": {
                "model_patch": {"patch": f"diff --git a/src/{suffix}.ts b/src/{suffix}.ts\n"}
            },
        },
        trace_source_revision="a" * 40,
        task_source_revision="b" * 40,
    )


def _command(output: str, *, resolved: bool) -> CommandResult:
    return CommandResult(
        argv=("fixture",),
        cwd=Path.cwd(),
        outcome=OutcomeCategory.SUCCESS if resolved else OutcomeCategory.TEST_FAILURE,
        exit_code=0 if resolved else 1,
        stdout=output,
        stderr="",
        duration_seconds=0.1,
    )


def _parser(_: str, output: str) -> dict[str, str]:
    return dict(line.split(":", 1) for line in output.splitlines() if ":" in line)


def test_canary_commands_publish_a_safe_resumable_pass_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = SWERebenchTask(
        task_source_revision="b" * 40,
        instance_id="owner__repo",
        repository="owner/repo",
        base_commit="c" * 40,
        language="TypeScript",
        image_name="docker.io/swerebenchv2/owner-repo:fixture",
        test_patch="PRIVATE TEST PATCH",
        fail_to_pass=("fixes bug",),
        pass_to_pass=("keeps behavior",),
        test_commands=("npm test -- --verbose",),
        log_parser="parse_log_jest",
    )
    control_request = _request("control")
    control = ExactResolutionCandidate(
        resolution_key=control_request.resolution_key,
        instance_id=control_request.instance_id,
        language=control_request.language,
        model_patch_sha256=control_request.model_patch_sha256,
        resolved=True,
        trace_source_revision=control_request.trace_source_revision,
        task_source_revision=control_request.task_source_revision,
        label_evidence=control_request.target_references,
        target_reference=control_request.target_references[0],
    )
    cases = tuple(
        sorted(
            (
                build_transfer_control_case(control, control_request, task),
                build_evaluation_case(_request("queue"), task),
            ),
            key=lambda case: case.case_id,
        )
    )
    workset = tmp_path / "canary.private.jsonl"
    write_immutable_stream(
        workset,
        lambda stream: [
            stream.write((json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n").encode())
            for case in cases
        ],
    )
    workset_identity = file_identity(workset)
    workset_manifest = tmp_path / "canary.workset.manifest.json"
    workset_record = ResolutionCanaryWorksetManifestV1(
        schema_version="nodelm.resolution-canary-workset/v1",
        materialization_status="PASS",
        execution_status="NOT RUN",
        admission_status="BLOCKED",
        admission_blocker="canary_execution_pending",
        recovery_manifest_sha256="1" * 64,
        recovery_manifest_bytes=100,
        candidate_sha256="2" * 64,
        candidate_bytes=200,
        queue_sha256="3" * 64,
        queue_bytes=300,
        trace_source_name="fixture-traces",
        trace_source_revision="a" * 40,
        trace_transfer_receipt_sha256="4" * 64,
        task_source_name="fixture-tasks",
        task_source_revision="b" * 40,
        task_transfer_receipt_sha256="5" * 64,
        selection_algorithm="nodelm.resolution-canary-cover/v1",
        minimum_per_kind=1,
        maximum_per_kind=2,
        evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
        evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
        workset_artifact=workset.name,
        workset_sha256=workset_identity[0],
        workset_bytes=workset_identity[1],
        case_count=2,
        transfer_control_count=1,
        evaluation_request_count=1,
        languages=("TypeScript",),
        target_partitions=("openhands/qwen36_27b/swe-rebench-v2",),
    )
    write_immutable_json(workset_manifest, workset_record.model_dump(mode="json"))
    locked_image = PinnedContainerImage(
        source_image=task.image_name,
        image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "d" * 64,
    )

    class FakeLocker:
        def __init__(self, *, executable: str) -> None:
            assert executable == "podman"

        def lock(
            self,
            source_images: tuple[str, ...],
            *,
            workspace: Path,
            workset_sha256: str,
        ) -> ResolutionCanaryImageLock:
            assert set(source_images) == {task.image_name}
            assert workspace == tmp_path
            return ResolutionCanaryImageLock(
                workset_sha256=workset_sha256,
                evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
                evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
                images=(locked_image,),
            )

    monkeypatch.setattr("nodelm.cli.PodmanImageLocker", FakeLocker)
    image_lock = tmp_path / "canary.images.json"
    lock_result = CliRunner().invoke(
        app,
        [
            "datasets",
            "lock-resolution-canary-images",
            "--workset",
            str(workset),
            "--workset-manifest",
            str(workset_manifest),
            "--output",
            str(image_lock),
        ],
    )
    assert lock_result.exit_code == 0, lock_result.output

    private_sentinel = "PRIVATE_SCHEMA_SENTINEL_MUST_NOT_REACH_CLI"
    invalid_workset = tmp_path / "invalid.private.jsonl"
    invalid_rows = [json.loads(line) for line in workset.read_text().splitlines()]
    invalid_rows[0]["kind"] = private_sentinel
    invalid_workset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in invalid_rows),
        encoding="utf-8",
    )
    invalid_identity = file_identity(invalid_workset)
    invalid_manifest = tmp_path / "invalid.workset.manifest.json"
    write_immutable_json(
        invalid_manifest,
        workset_record.model_copy(
            update={
                "workset_sha256": invalid_identity[0],
                "workset_bytes": invalid_identity[1],
            }
        ).model_dump(mode="json"),
    )
    rejected_private = CliRunner().invoke(
        app,
        [
            "datasets",
            "lock-resolution-canary-images",
            "--workset",
            str(invalid_workset),
            "--workset-manifest",
            str(invalid_manifest),
            "--output",
            str(tmp_path / "invalid.images.json"),
        ],
    )
    assert rejected_private.exit_code != 0
    normalized_error = " ".join(rejected_private.output.split())
    assert "private workset" in normalized_error
    assert "failed schema validation" in normalized_error
    assert private_sentinel not in rejected_private.output
    assert "PRIVATE TEST PATCH" not in rejected_private.output

    evidence_dir = tmp_path / "private-case-evidence"
    evidence_dir.mkdir()
    oracle = ResolutionCanaryOracle(_parser)
    for case in cases:
        baseline = _command("fixes bug:FAILED\nkeeps behavior:PASSED", resolved=False)
        is_control = case.kind == "transfer_control"
        candidate = _command(
            (
                "fixes bug:PASSED\nkeeps behavior:PASSED"
                if is_control
                else "fixes bug:FAILED\nkeeps behavior:PASSED"
            ),
            resolved=is_control,
        )
        safe = oracle.evaluate(
            case,
            image=locked_image,
            baseline=baseline,
            candidate=candidate,
            sandbox_evidence={"backend": "fixture"},
        )
        private = ResolutionCanaryPrivateCaseEvidence(
            result=safe,
            baseline_output=resolution_canary_output(baseline),
            candidate_output=resolution_canary_output(candidate),
        )
        write_immutable_json(
            evidence_dir / f"{case.case_id}.json",
            private.model_dump(mode="json"),
        )

    evaluator = tmp_path / "evaluator"
    (evaluator / "lib" / "agent").mkdir(parents=True)
    (evaluator / "scripts").mkdir()
    (evaluator / "lib" / "agent" / "log_parsers.py").write_text("fixture")
    (evaluator / "lib" / "agent" / "swe_constants.py").write_text("constants")
    (evaluator / "scripts" / "eval.py").write_text("fixture")
    parser_sha256 = file_identity(evaluator / "lib" / "agent" / "log_parsers.py")[0]
    eval_sha256 = file_identity(evaluator / "scripts" / "eval.py")[0]
    constants_sha256 = file_identity(evaluator / "lib" / "agent" / "swe_constants.py")[0]

    class FakeParser:
        def __init__(self, evaluator_root: Path) -> None:
            assert evaluator_root == tmp_path / "evaluator"
            self.parser_sha256 = parser_sha256
            self.eval_sha256 = eval_sha256
            self.constants_sha256 = constants_sha256

    monkeypatch.setattr("nodelm.cli.PinnedEvaluatorLogParser", FakeParser)
    monkeypatch.setattr(
        "nodelm.cli.SWE_REBENCH_LOG_PARSERS_SHA256",
        parser_sha256,
    )
    monkeypatch.setattr("nodelm.cli.SWE_REBENCH_EVAL_SCRIPT_SHA256", eval_sha256)
    monkeypatch.setattr("nodelm.cli.SWE_REBENCH_CONSTANTS_SHA256", constants_sha256)
    results = tmp_path / "canary.results.jsonl"
    execution_manifest = tmp_path / "canary.execution.manifest.json"
    execution = CliRunner().invoke(
        app,
        [
            "datasets",
            "run-resolution-canary",
            "--workset",
            str(workset),
            "--workset-manifest",
            str(workset_manifest),
            "--image-lock",
            str(image_lock),
            "--evaluator-root",
            str(evaluator),
            "--case-evidence-dir",
            str(evidence_dir),
            "--results-output",
            str(results),
            "--manifest-output",
            str(execution_manifest),
            "--code-commit",
            "e" * 40,
        ],
    )

    assert execution.exit_code == 0, execution.output
    manifest = ResolutionCanaryExecutionManifestV1.model_validate_json(
        execution_manifest.read_bytes()
    )
    assert manifest.execution_status == "PASS"
    assert manifest.admission_status == "PASS"
    assert manifest.transfer_label_agreement_count == 1
    assert manifest.evaluation_unresolved_count == 1
    assert "PRIVATE TEST PATCH" not in results.read_text()
    assert "fixes bug" not in results.read_text()
