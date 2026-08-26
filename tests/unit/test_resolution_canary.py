from __future__ import annotations

from pathlib import Path

import pytest

from nodelm.evaluation.resolution_canary import (
    PinnedContainerImage,
    ResolutionCanaryCase,
    ResolutionCanaryError,
    ResolutionCanaryOracle,
    SWERebenchTask,
    project_swe_rebench_task,
)
from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.models import VerificationStatus
from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
    ResolutionRowReference,
    evaluation_request_from_target_row,
)
from nodelm.provenance.resolution_canary import (
    build_evaluation_case,
    build_transfer_control_case,
    select_resolution_canary_sources,
)

TRACE_REVISION = "a" * 40
TASK_REVISION = "b" * 40


def _reference(partition: str, suffix: str) -> ResolutionRowReference:
    return ResolutionRowReference(
        partition_name=partition,
        rollout_id=f"rollout-{suffix}",
        projected_row_sha256=suffix * 64,
    )


def _request(
    suffix: str,
    *,
    language: str = "TypeScript",
    partitions: tuple[str, ...] = ("openhands/qwen36_27b/swe-rebench-v2",),
) -> ResolutionEvaluationRequest:
    patch = f"diff --git a/src/{suffix}.ts b/src/{suffix}.ts\n"
    row = {
        "instance_id": f"owner__repo-{suffix}",
        "trajectory_id": f"rollout-{suffix}",
        "resolved": -1,
        "language": language,
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "metadata": {"model_patch": {"patch": patch}},
    }
    first = evaluation_request_from_target_row(
        partitions[0],
        row,
        trace_source_revision=TRACE_REVISION,
        task_source_revision=TASK_REVISION,
    )
    return ResolutionEvaluationRequest(
        resolution_key=first.resolution_key,
        instance_id=first.instance_id,
        language=first.language,
        model_patch=first.model_patch,
        model_patch_sha256=first.model_patch_sha256,
        trace_source_revision=first.trace_source_revision,
        task_source_revision=first.task_source_revision,
        target_references=tuple(
            sorted(
                (
                    ResolutionRowReference(
                        partition_name=partition,
                        rollout_id=f"rollout-{suffix}-{index}",
                        projected_row_sha256=f"{index + 1:064x}",
                    )
                    for index, partition in enumerate(partitions)
                ),
                key=lambda item: (
                    item.partition_name,
                    item.rollout_id,
                    item.projected_row_sha256,
                ),
            )
        ),
    )


def _candidate(
    request: ResolutionEvaluationRequest,
    *,
    resolved: bool,
) -> ExactResolutionCandidate:
    return ExactResolutionCandidate(
        resolution_key=request.resolution_key,
        instance_id=request.instance_id,
        language=request.language,
        model_patch_sha256=request.model_patch_sha256,
        resolved=resolved,
        trace_source_revision=request.trace_source_revision,
        task_source_revision=request.task_source_revision,
        label_evidence=(_reference("openhands/minimax_m25/swe-rebench-v2", "c"),),
        target_reference=request.target_references[0],
    )


def _task(*, language: str = "TypeScript") -> SWERebenchTask:
    return SWERebenchTask(
        task_source_revision=TASK_REVISION,
        instance_id="owner__repo-a",
        repository="owner/repo",
        base_commit="d" * 40,
        language=language,
        image_name="docker.io/swerebenchv2/owner-repo:1-deadbee",
        test_patch="diff --git a/test/a.test.ts b/test/a.test.ts\n",
        fail_to_pass=("fixes bug",),
        pass_to_pass=("keeps behavior",),
        test_commands=("npm test -- --verbose",),
        log_parser="parse_log_jest",
    )


def _command(output: str, *, exit_code: int = 1) -> CommandResult:
    return CommandResult(
        argv=("podman", "run"),
        cwd=Path("/tmp"),
        outcome=(OutcomeCategory.SUCCESS if exit_code == 0 else OutcomeCategory.TEST_FAILURE),
        exit_code=exit_code,
        stdout=output,
        stderr="",
        duration_seconds=1.25,
    )


def _parse_status_lines(_name: str, output: str) -> dict[str, str]:
    return {line.split(":", 1)[0]: line.split(":", 1)[1] for line in output.splitlines()}


def test_canary_selection_is_deterministic_and_covers_languages_partitions_and_labels() -> None:
    partitions = (
        "minisweagent/qwen36_27b/swe-rebench-v2",
        "openhands/qwen36_27b/swe-rebench-v2",
        "sweagent/qwen36_27b/swe-rebench-v2",
    )
    requests = tuple(
        _request(
            chr(ord("a") + index),
            language="TypeScript" if index % 2 == 0 else "JavaScript",
            partitions=(partition,),
        )
        for index, partition in enumerate(partitions * 2)
    )
    candidates = tuple(
        _candidate(request, resolved=index % 2 == 0) for index, request in enumerate(requests)
    )

    first = select_resolution_canary_sources(candidates, requests, minimum_per_kind=4)
    second = select_resolution_canary_sources(
        reversed(candidates), reversed(requests), minimum_per_kind=4
    )

    assert first == second
    assert len(first.transfer_controls) >= 4
    assert len(first.evaluation_requests) >= 4
    assert {item.resolved for item in first.transfer_controls} == {False, True}
    assert {item.language for item in first.transfer_controls} == {"JavaScript", "TypeScript"}
    assert {
        reference.partition_name
        for item in first.transfer_controls
        for reference in (item.target_reference,)
    } == set(partitions)
    assert {
        reference.partition_name
        for item in first.evaluation_requests
        for reference in item.target_references
    } == set(partitions)


def test_transfer_control_reconstructs_only_the_exact_bound_target_patch() -> None:
    request = _request("a")
    candidate = _candidate(request, resolved=True)
    task = _task()

    case = build_transfer_control_case(candidate, request, task)

    assert case.kind == "transfer_control"
    assert case.expected_resolved is True
    assert case.model_patch == request.model_patch
    assert case.task == task

    wrong_patch = request.model_copy(update={"model_patch": "different"})
    with pytest.raises(ResolutionCanaryError, match="patch"):
        build_transfer_control_case(candidate, wrong_patch, task)


def test_task_projection_keeps_private_oracle_fields_but_drops_gold_patch() -> None:
    row = {
        "instance_id": "owner__repo-a",
        "repo": "owner/repo",
        "base_commit": "d" * 40,
        "language": "ts",
        "image_name": "docker.io/swerebenchv2/owner-repo:1-deadbee",
        "patch": "GOLD MUST NOT ENTER THE CANARY CASE",
        "test_patch": "diff --git a/test/a.test.ts b/test/a.test.ts\n",
        "FAIL_TO_PASS": ["fixes bug"],
        "PASS_TO_PASS": ["keeps behavior"],
        "install_config": {
            "test_cmd": "npm test -- --verbose",
            "log_parser": "parse_log_jest",
        },
    }

    task = project_swe_rebench_task(row, task_source_revision=TASK_REVISION)

    serialized = task.model_dump_json()
    assert task.language == "TypeScript"
    assert "test_patch" in serialized
    assert "GOLD MUST NOT ENTER" not in serialized
    assert '"patch"' not in serialized


def test_case_identity_binds_source_patch_task_and_expected_label() -> None:
    request = _request("a")
    case = build_evaluation_case(request, _task())

    assert case.case_id
    with pytest.raises(ValueError, match="case_id"):
        ResolutionCanaryCase.model_validate(
            {
                **case.model_dump(mode="json"),
                "task": {
                    **case.task.model_dump(mode="json"),
                    "base_commit": "f" * 40,
                },
            }
        )


def test_oracle_requires_baseline_failure_and_reports_a_resolved_queue_case() -> None:
    case = build_evaluation_case(_request("a"), _task())
    oracle = ResolutionCanaryOracle(_parse_status_lines)

    result = oracle.evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command("fixes bug:FAILED\nkeeps behavior:PASSED"),
        candidate=_command("fixes bug:PASSED\nkeeps behavior:PASSED", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.PASS
    assert result.task_resolved is True
    assert result.label_agreement is None
    assert result.baseline.output_sha256
    assert "fixes bug" not in result.model_dump_json()


def test_oracle_fails_closed_when_a_transfer_label_disagrees() -> None:
    request = _request("a")
    case = build_transfer_control_case(_candidate(request, resolved=True), request, _task())
    result = ResolutionCanaryOracle(_parse_status_lines).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command("fixes bug:FAILED\nkeeps behavior:PASSED"),
        candidate=_command("fixes bug:FAILED\nkeeps behavior:PASSED"),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.FAIL
    assert result.task_resolved is False
    assert result.label_agreement is False


def test_oracle_fails_closed_when_expected_tests_are_missing() -> None:
    case = build_evaluation_case(_request("a"), _task())
    result = ResolutionCanaryOracle(_parse_status_lines).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command("fixes bug:FAILED"),
        candidate=_command("fixes bug:PASSED", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.FAIL
    assert result.task_resolved is None
    assert result.reason == "incomplete_expected_test_evidence"
