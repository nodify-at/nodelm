from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

import nodelm.evaluation.resolution_canary as resolution_canary_module
from nodelm.evaluation.resolution_canary import (
    SWE_REBENCH_EVALUATOR_REVISION,
    PinnedContainerImage,
    PodmanImageLocker,
    ResolutionCanaryCase,
    ResolutionCanaryError,
    ResolutionCanaryImageLock,
    ResolutionCanaryOracle,
    SkopeoChrootImageLocker,
    SWERebenchPodmanSandbox,
    SWERebenchSeccompChrootSandbox,
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


def _task(*, language: Literal["TypeScript", "JavaScript"] = "TypeScript") -> SWERebenchTask:
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
        "test_patch": "\n diff --git a/test/a.test.ts b/test/a.test.ts\n\n",
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
    assert task.test_patch == row["test_patch"]
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


def test_oracle_requires_every_fail_to_pass_test_to_fail_on_the_baseline() -> None:
    task = _task().model_copy(update={"fail_to_pass": ("fixes bug", "fixes other bug")})
    case = build_evaluation_case(_request("a"), task)
    result = ResolutionCanaryOracle(_parse_status_lines).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command("fixes bug:FAILED\nfixes other bug:PASSED\nkeeps behavior:PASSED"),
        candidate=_command(
            "fixes bug:PASSED\nfixes other bug:PASSED\nkeeps behavior:PASSED",
            exit_code=0,
        ),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.FAIL
    assert result.reason == "failing_baseline_not_reproduced"


def test_oracle_recovers_exact_numbered_failures_omitted_by_pinned_js_parser() -> None:
    task = _task().model_copy(update={"log_parser": "parse_log_js_4"})
    case = build_evaluation_case(_request("a"), task)

    def pinned_like_parser(_name: str, output: str) -> dict[str, str]:
        parsed = {"keeps behavior": "PASSED"}
        if "✓ fixes bug" in output:
            parsed["fixes bug"] = "PASSED"
        return parsed

    result = ResolutionCanaryOracle(pinned_like_parser).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command("  1) fixes bug\n  ✓ keeps behavior\n  2) keeps behavior"),
        candidate=_command("  ✓ fixes bug\n  ✓ keeps behavior", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.PASS
    assert result.task_resolved is True


def test_oracle_recovers_exact_jest_cross_failures_omitted_by_pinned_js_parser() -> None:
    task = _task().model_copy(update={"log_parser": "parse_log_js_4"})
    case = build_evaluation_case(_request("a"), task)

    def pinned_like_parser(_name: str, output: str) -> dict[str, str]:
        parsed = {"keeps behavior": "PASSED"}
        if "✓ fixes bug" in output:
            parsed["fixes bug"] = "PASSED"
        return parsed

    result = ResolutionCanaryOracle(pinned_like_parser).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command(
            "  ✕ fixes bug (12ms)\n"
            "  ✓ keeps behavior\n"
            "\n"
            "  ● suite \u203a fixes bug\n"
            "\n"
            "Tests:       1 failed, 1 passed, 2 total"
        ),
        candidate=_command("  ✓ fixes bug\n  ✓ keeps behavior", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.PASS
    assert result.task_resolved is True


def test_oracle_recovers_exact_ava_rejection_failures_omitted_by_pinned_js_parser() -> None:
    task = _task().model_copy(update={"log_parser": "parse_log_js_4"})
    case = build_evaluation_case(_request("a"), task)

    def pinned_like_parser(_name: str, output: str) -> dict[str, str]:
        parsed = {"keeps behavior": "PASSED"}
        if "✓ fixes bug" in output:
            parsed["fixes bug"] = "PASSED"
        elif "Rejected promise returned by test" in output:
            parsed["fixes bug Rejected promise returned by test"] = "FAILED"
        return parsed

    result = ResolutionCanaryOracle(pinned_like_parser).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command(
            "  ✘ [fail]: fixes bug Rejected promise returned by test\n"
            "  ✓ keeps behavior\n"
            "\n"
            "  fixes bug\n"
            "\n"
            "  1 test failed"
        ),
        candidate=_command("  ✓ fixes bug\n  ✓ keeps behavior", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.PASS
    assert result.task_resolved is True


def test_oracle_does_not_recover_ambiguous_or_already_parsed_symbol_failures() -> None:
    task = _task().model_copy(update={"log_parser": "parse_log_js_4"})
    case = build_evaluation_case(_request("a"), task)

    def pinned_like_parser(_name: str, _output: str) -> dict[str, str]:
        return {"fixes bug": "PASSED", "keeps behavior": "PASSED"}

    result = ResolutionCanaryOracle(pinned_like_parser).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command(
            "  ✕ fixes bug with ambiguous suffix\n"
            "  ✘ [fail]: fixes bug with ambiguous suffix Rejected promise returned by test"
        ),
        candidate=_command("  ✓ fixes bug\n  ✓ keeps behavior", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.FAIL
    assert result.reason == "failing_baseline_not_reproduced"


@pytest.mark.parametrize(
    "fabricated_output",
    [
        "  ✕ fixes bug (12ms)",
        "  ✕ fixes bug (12ms)\n  ● suite \u203a fixes bug",
        "  ✕ fixes bug (12ms)\nTests:       1 failed, 1 passed, 2 total",
        "  ✘ [fail]: fixes bug Rejected promise returned by test",
        "  ✘ [fail]: fixes bug Rejected promise returned by test\n  fixes bug",
        "  ✘ [fail]: fixes bug Rejected promise returned by test\n  1 test failed",
    ],
)
def test_oracle_requires_complete_reporter_context_for_symbol_failures(
    fabricated_output: str,
) -> None:
    task = _task().model_copy(update={"log_parser": "parse_log_js_4"})
    case = build_evaluation_case(_request("a"), task)

    def pinned_like_parser(_name: str, output: str) -> dict[str, str]:
        parsed = {"keeps behavior": "PASSED"}
        if "✓ fixes bug" in output:
            parsed["fixes bug"] = "PASSED"
        return parsed

    result = ResolutionCanaryOracle(pinned_like_parser).evaluate(
        case,
        image=PinnedContainerImage(
            source_image=case.task.image_name,
            image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
        ),
        baseline=_command(f"application output\n{fabricated_output}\n  ✓ keeps behavior"),
        candidate=_command("  ✓ fixes bug\n  ✓ keeps behavior", exit_code=0),
        sandbox_evidence={"backend": "fake"},
    )

    assert result.status is VerificationStatus.FAIL
    assert result.reason == "incomplete_expected_test_evidence"


def test_image_digest_selection_requires_the_source_repository() -> None:
    selected = PodmanImageLocker._select_repo_digest(
        "registry.example:5000/team/repo:canary",
        '["registry.example:5000/team/repo@sha256:' + "a" * 64 + '"]',
    )

    assert selected == "registry.example:5000/team/repo@sha256:" + "a" * 64
    with pytest.raises(ResolutionCanaryError, match="matching immutable"):
        PodmanImageLocker._select_repo_digest(
            "registry.example:5000/team/repo:canary",
            '["registry.example:5000/other/repo@sha256:' + "a" * 64 + '"]',
        )


def test_seccomp_chroot_image_lock_requires_local_oci_manifest_identity() -> None:
    image = PinnedContainerImage(
        source_image="docker.io/swerebenchv2/owner-repo:fixture",
        image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
    )

    with pytest.raises(ValueError, match="local OCI manifest"):
        ResolutionCanaryImageLock(
            workset_sha256="a" * 64,
            evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
            evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
            runtime="seccomp-chroot",
            images=(image,),
        )

    pinned = image.model_copy(
        update={"runtime_artifact_sha256": "f" * 64, "runtime_artifact_bytes": 2_135}
    )
    lock = ResolutionCanaryImageLock(
        workset_sha256="a" * 64,
        evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
        evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
        runtime="seccomp-chroot",
        images=(pinned,),
    )
    assert lock.runtime == "seccomp-chroot"
    with pytest.raises(ValueError, match="Podman image locks"):
        ResolutionCanaryImageLock(
            workset_sha256="a" * 64,
            evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
            evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
            images=(pinned,),
        )


def test_oci_runtime_configuration_rejects_unsafe_paths_and_bounds(tmp_path: Path) -> None:
    SkopeoChrootImageLocker(image_root=tmp_path)
    SWERebenchSeccompChrootSandbox(image_root=tmp_path)

    with pytest.raises(ValueError, match="absolute non-symlink"):
        SkopeoChrootImageLocker(image_root=Path("relative"))
    with pytest.raises(ValueError, match="non-empty NUL-free"):
        SkopeoChrootImageLocker(image_root=tmp_path, executable="")
    with pytest.raises(ValueError, match="greater than zero"):
        SkopeoChrootImageLocker(image_root=tmp_path, pull_timeout_seconds=0)
    with pytest.raises(ValueError, match="non-empty NUL-free"):
        SWERebenchSeccompChrootSandbox(image_root=tmp_path, skopeo="")
    with pytest.raises(ValueError, match="greater than zero"):
        SWERebenchSeccompChrootSandbox(image_root=tmp_path, sandbox_uid=0)

    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="absolute non-symlink"):
        SWERebenchSeccompChrootSandbox(image_root=linked)


def test_execution_patch_copy_adds_only_a_missing_terminal_line_feed() -> None:
    complete = "diff --git a/src/a.ts b/src/a.ts\n"
    unterminated = complete.rstrip("\n")

    assert resolution_canary_module._execution_patch_text(complete) == complete
    assert resolution_canary_module._execution_patch_text(unterminated) == complete


def test_seccomp_chroot_preserves_image_environment_with_safe_overrides(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        '{"process":{"cwd":"/KaTeX","env":["PATH=/custom/bin","HOME=/root","FEATURE=yes"]}}',
        encoding="utf-8",
    )

    environment = SWERebenchSeccompChrootSandbox._image_environment(bundle)

    assert SWERebenchSeccompChrootSandbox._image_workdir(bundle) == "/KaTeX"
    assert "PATH=/custom/bin" in environment
    assert "FEATURE=yes" in environment
    assert "HOME=/tmp/nodelm-home" in environment
    assert "CI=true" in environment


def test_seccomp_chroot_rejects_unsafe_oci_workdir(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        '{"process":{"cwd":"../KaTeX","env":[]}}',
        encoding="utf-8",
    )

    with pytest.raises(ResolutionCanaryError, match="working directory"):
        SWERebenchSeccompChrootSandbox._image_workdir(bundle)


def test_seccomp_chroot_prepares_case_sensitive_oci_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "KaTeX" / "src").mkdir(parents=True)
    (rootfs / "dev").mkdir()
    corepack = rootfs / "root" / ".cache" / "node" / "corepack"
    yarn = corepack / "v1" / "yarn" / "1.22.22"
    yarn.mkdir(parents=True)
    (corepack / "lastKnownGood.json").write_text('{"yarn":"1.22.22"}\n')
    (yarn / ".corepack").write_text('{"hash":"sha512.fixture"}\n')
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")
    monkeypatch.setattr("nodelm.evaluation.resolution_canary.os.chown", lambda *_: None)
    monkeypatch.setattr("nodelm.evaluation.resolution_canary.os.lchown", lambda *_: None)

    sandbox._prepare_rootfs(rootfs, "/KaTeX")

    assert (rootfs / "nodelm-input").is_dir()
    assert (rootfs / "tmp" / "nodelm-home").is_dir()
    copied = rootfs / "tmp" / "nodelm-home" / ".cache" / "node" / "corepack"
    assert (copied / "lastKnownGood.json").read_text() == '{"yarn":"1.22.22"}\n'
    assert (copied / "v1" / "yarn" / "1.22.22" / ".corepack").is_file()
    assert "nodelm-canary:x:61000:61000:" in (rootfs / "etc" / "passwd").read_text()
    assert "nodelm-canary:x:61000:" in (rootfs / "etc" / "group").read_text()


def test_seccomp_chroot_rejects_conflicting_sandbox_uid_before_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "repository").mkdir(parents=True)
    (rootfs / "dev").mkdir()
    (rootfs / "etc").mkdir()
    (rootfs / "etc" / "passwd").write_text(
        "other:x:61000:61000:conflict:/tmp:/bin/false\n", encoding="utf-8"
    )
    (rootfs / "etc" / "group").write_text("other:x:61000:\n", encoding="utf-8")
    chown_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.chown",
        lambda *args: chown_calls.append(args),
    )
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.lchown",
        lambda *args: chown_calls.append(args),
    )
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")

    with pytest.raises(ResolutionCanaryError, match="sandbox UID"):
        sandbox._prepare_rootfs(rootfs, "/repository")

    assert chown_calls == []


def test_seccomp_chroot_rejects_symlinked_passwd_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = tmp_path / "rootfs"
    outside = tmp_path / "outside-passwd"
    (rootfs / "repository").mkdir(parents=True)
    (rootfs / "dev").mkdir()
    (rootfs / "etc").mkdir()
    outside.write_text("host-content\n", encoding="utf-8")
    (rootfs / "etc" / "passwd").symlink_to(outside)
    chown_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.chown",
        lambda *args: chown_calls.append(args),
    )
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.lchown",
        lambda *args: chown_calls.append(args),
    )
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")

    with pytest.raises(ResolutionCanaryError, match="identity file"):
        sandbox._prepare_rootfs(rootfs, "/repository")

    assert outside.read_text(encoding="utf-8") == "host-content\n"
    assert chown_calls == []


def test_seccomp_chroot_rejects_symlinked_corepack_cache_before_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = tmp_path / "rootfs"
    outside = tmp_path / "outside"
    (rootfs / "repository").mkdir(parents=True)
    (rootfs / "dev").mkdir()
    (rootfs / "root" / ".cache" / "node").mkdir(parents=True)
    outside.mkdir()
    (rootfs / "root" / ".cache" / "node" / "corepack").symlink_to(outside, target_is_directory=True)
    chown_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.chown",
        lambda *args: chown_calls.append(args),
    )
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.lchown",
        lambda *args: chown_calls.append(args),
    )
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")

    with pytest.raises(ResolutionCanaryError, match="Corepack cache"):
        sandbox._prepare_rootfs(rootfs, "/repository")

    assert chown_calls == []


@pytest.mark.parametrize("linked_component", [".cache", ".cache/node"])
def test_seccomp_chroot_rejects_existing_home_with_intermediate_cache_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_component: str
) -> None:
    rootfs = tmp_path / "rootfs"
    outside = tmp_path / "outside"
    (rootfs / "repository").mkdir(parents=True)
    (rootfs / "dev").mkdir()
    source_cache = rootfs / "root" / ".cache" / "node" / "corepack"
    source_cache.mkdir(parents=True)
    (source_cache / "lastKnownGood.json").write_text('{"yarn":"1.22.22"}\n')
    home = rootfs / "tmp" / "nodelm-home"
    home.mkdir(parents=True)
    outside.mkdir()
    linked_path = home / linked_component
    linked_path.parent.mkdir(parents=True, exist_ok=True)
    linked_path.symlink_to(outside, target_is_directory=True)
    chown_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.chown",
        lambda *args: chown_calls.append(args),
    )
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.lchown",
        lambda *args: chown_calls.append(args),
    )
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")

    with pytest.raises(ResolutionCanaryError, match="sandbox home"):
        sandbox._prepare_rootfs(rootfs, "/repository")

    assert list(outside.iterdir()) == []
    assert chown_calls == []


def test_seccomp_chroot_rejects_intermediate_workdir_symlinks_before_chown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = tmp_path / "rootfs"
    outside = tmp_path / "outside"
    (rootfs / "dev").mkdir(parents=True)
    (outside / "repository").mkdir(parents=True)
    (rootfs / "escape").symlink_to(outside, target_is_directory=True)
    chown_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.chown",
        lambda *args: chown_calls.append(args),
    )
    monkeypatch.setattr(
        "nodelm.evaluation.resolution_canary.os.lchown",
        lambda *args: chown_calls.append(args),
    )
    sandbox = SWERebenchSeccompChrootSandbox(image_root=tmp_path / "images")

    with pytest.raises(ResolutionCanaryError, match="symlink"):
        sandbox._prepare_rootfs(rootfs, "/escape/repository")

    assert chown_calls == []


def test_real_repository_sandbox_command_is_offline_bounded_and_digest_pinned(
    tmp_path: Path,
) -> None:
    case = build_evaluation_case(_request("a"), _task())
    image = PinnedContainerImage(
        source_image=case.task.image_name,
        image_digest="docker.io/swerebenchv2/owner-repo@sha256:" + "e" * 64,
    )
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    cidfile = tmp_path / "container.cid"
    sandbox = SWERebenchPodmanSandbox(executable="podman")

    baseline = sandbox.command(
        case,
        image,
        patch_dir=patch_dir,
        include_model_patch=False,
        container_name="nodelm-resolution-canary-" + "a" * 24,
        cidfile=cidfile,
    )
    candidate = sandbox.command(
        case,
        image,
        patch_dir=patch_dir,
        include_model_patch=True,
        container_name="nodelm-resolution-canary-" + "b" * 24,
        cidfile=cidfile,
    )

    assert "--network=none" in baseline
    assert "--cap-drop=ALL" in baseline
    assert "--security-opt=no-new-privileges" in baseline
    assert "--pids-limit=512" in baseline
    assert "--memory=4g" in baseline
    assert "--cpus=2" in baseline
    assert "--env=_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false" in baseline
    assert image.image_digest in baseline
    assert image.source_image not in baseline
    assert not any(argument.startswith("--workdir=") for argument in baseline)
    volume_arguments = tuple(argument for argument in baseline if argument.startswith("--volume="))
    assert any(argument.endswith(":/nodelm-input:ro") for argument in volume_arguments)
    assert all(":rw" not in argument for argument in volume_arguments)
    assert "/nodelm-input/model.patch" not in baseline[-1]
    assert "/nodelm-input/model.patch" in candidate[-1]
    assert case.task.base_commit in baseline[-1]
    assert "git reset --hard HEAD" in baseline[-1]
