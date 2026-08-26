from __future__ import annotations

import json
import os
import re
import secrets
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from nodelm.artifacts import canonical_json_bytes, content_digest, file_identity
from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.harness import CommandExecutor, CommandPolicy, CommandResult, OutcomeCategory
from nodelm.models import VerificationStatus
from nodelm.provenance.resolution import (
    ResolutionRowReference,
    model_patch_sha256,
    resolution_key_sha256,
)
from nodelm.provenance.task_provenance import TaskProjectionError, canonical_language

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{40}$")]
NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
SupportedLanguage = Literal["TypeScript", "JavaScript"]
CaseKind = Literal["transfer_control", "evaluation_request"]

_CASE_IDENTITY_SCHEMA = "nodelm.resolution-canary-case-identity/v1"
_IMAGE_DIGEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
_TIMING_NORMALIZE_RES = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)
_EVALUATION_OUTCOMES = frozenset({OutcomeCategory.SUCCESS, OutcomeCategory.TEST_FAILURE})
_NUMBERED_JS_FAILURE = re.compile(r"^\s*\d+\)\s+(.+?)(?::)?\s*$")

SWE_REBENCH_EVALUATOR_REPOSITORY_ID = "SWE-rebench/SWE-rebench-V2"
SWE_REBENCH_EVALUATOR_REVISION = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"
SWE_REBENCH_LOG_PARSERS_SHA256 = "a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c"
SWE_REBENCH_EVAL_SCRIPT_SHA256 = "4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda"
SWE_REBENCH_CONSTANTS_SHA256 = "823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5"


class ResolutionCanaryError(ValueError):
    """Resolution evidence cannot cross the real repository canary safely."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SWERebenchTask(_StrictFrozenModel):
    """Private oracle material required to evaluate one SWE-rebench V2 patch."""

    schema_version: Literal["nodelm.swe-rebench-task/v1"] = "nodelm.swe-rebench-task/v1"
    task_source_revision: CommitSha
    instance_id: NonEmptyStr
    repository: NonEmptyStr
    base_commit: CommitSha
    language: SupportedLanguage
    image_name: NonEmptyStr
    test_patch: NonEmptyStr
    fail_to_pass: tuple[NonEmptyStr, ...] = Field(min_length=1)
    pass_to_pass: tuple[NonEmptyStr, ...] = ()
    test_commands: tuple[NonEmptyStr, ...] = Field(min_length=1)
    log_parser: NonEmptyStr

    @field_validator("task_source_revision", "base_commit")
    @classmethod
    def canonicalize_commit(cls, value: str) -> str:
        return value.casefold()

    @field_validator("repository")
    @classmethod
    def canonicalize_repository(cls, value: str) -> str:
        return canonical_repository(value)

    @field_validator("fail_to_pass", "pass_to_pass", "test_commands")
    @classmethod
    def require_unique_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("SWE-rebench task lists must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_private_task(self) -> SWERebenchTask:
        if any(character.isspace() or character == "\0" for character in self.image_name):
            raise ValueError("SWE-rebench image name must be whitespace-free and NUL-free")
        if set(self.fail_to_pass) & set(self.pass_to_pass):
            raise ValueError("FAIL_TO_PASS and PASS_TO_PASS must be disjoint")
        return self


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResolutionCanaryError(f"SWE-rebench task requires non-empty {field}")
    return value.strip()


def _required_raw_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResolutionCanaryError(f"SWE-rebench task requires non-empty {field}")
    return value


def _string_list(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str):
        values: object = [value]
    else:
        values = value
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ):
        raise ResolutionCanaryError(f"SWE-rebench task requires string list {field}")
    normalized = tuple(item.strip() for item in values)
    if not normalized and not allow_empty:
        raise ResolutionCanaryError(f"SWE-rebench task requires non-empty {field}")
    return normalized


def project_swe_rebench_task(
    row: Mapping[str, Any],
    *,
    task_source_revision: str,
) -> SWERebenchTask:
    """Project a private task row without retaining its gold solution patch."""

    install_config = row.get("install_config")
    if not isinstance(install_config, Mapping):
        raise ResolutionCanaryError("SWE-rebench task requires install_config")
    try:
        language = canonical_language(row.get("language"))
    except TaskProjectionError as error:
        raise ResolutionCanaryError(str(error)) from error
    if language not in {"TypeScript", "JavaScript"}:
        raise ResolutionCanaryError("resolution canary task must be TypeScript or JavaScript")
    parser = install_config.get("log_parser")
    if not isinstance(parser, str) or not parser.strip():
        raise ResolutionCanaryError("SWE-rebench task requires install_config.log_parser")
    try:
        return SWERebenchTask(
            task_source_revision=task_source_revision,
            instance_id=_required_string(row, "instance_id"),
            repository=canonical_repository(_required_string(row, "repo")),
            base_commit=_required_string(row, "base_commit"),
            language=cast(SupportedLanguage, language),
            image_name=_required_string(row, "image_name"),
            test_patch=_required_raw_string(row, "test_patch"),
            fail_to_pass=_string_list(
                row.get("FAIL_TO_PASS"), field="FAIL_TO_PASS", allow_empty=False
            ),
            pass_to_pass=_string_list(
                row.get("PASS_TO_PASS", ()), field="PASS_TO_PASS", allow_empty=True
            ),
            test_commands=_string_list(
                install_config.get("test_cmd"),
                field="install_config.test_cmd",
                allow_empty=False,
            ),
            log_parser=parser.strip(),
        )
    except ValueError as error:
        if isinstance(error, ResolutionCanaryError):
            raise
        raise ResolutionCanaryError("private SWE-rebench task failed schema validation") from error


class ResolutionCanaryCase(_StrictFrozenModel):
    """Private, content-bound evaluator input for one selected patch."""

    schema_version: Literal["nodelm.resolution-canary-case/v1"] = "nodelm.resolution-canary-case/v1"
    case_id: str = Field(default="", pattern=r"^[0-9a-f]{64}$")
    kind: CaseKind
    source_id: Sha256
    resolution_key: Sha256
    instance_id: NonEmptyStr
    language: SupportedLanguage
    model_patch: StrictStr
    model_patch_sha256: Sha256
    trace_source_revision: CommitSha
    task_source_revision: CommitSha
    expected_resolved: StrictBool | None = None
    target_references: tuple[ResolutionRowReference, ...] = Field(min_length=1)
    task: SWERebenchTask

    @model_validator(mode="after")
    def validate_case_identity(self) -> ResolutionCanaryCase:
        if self.kind == "transfer_control" and self.expected_resolved is None:
            raise ValueError("transfer controls require an expected resolution label")
        if self.kind == "evaluation_request" and self.expected_resolved is not None:
            raise ValueError("evaluation requests must not carry a transferred label")
        if self.model_patch_sha256 != model_patch_sha256(self.model_patch):
            raise ValueError("model patch digest does not match canary patch")
        expected_key = resolution_key_sha256(
            instance_id=self.instance_id,
            model_patch=self.model_patch,
            trace_source_revision=self.trace_source_revision,
            task_source_revision=self.task_source_revision,
        )
        if self.resolution_key != expected_key:
            raise ValueError("resolution key does not match canary task and patch")
        if (
            self.task.instance_id != self.instance_id
            or self.task.language != self.language
            or self.task.task_source_revision.casefold() != self.task_source_revision.casefold()
        ):
            raise ValueError("private task does not match canary source identity")
        expected_id = resolution_canary_case_digest(self)
        if not self.case_id:
            object.__setattr__(self, "case_id", expected_id)
        elif self.case_id != expected_id:
            raise ValueError("case_id does not match canary source, patch, task, and label")
        return self


def resolution_canary_case_digest(case: ResolutionCanaryCase) -> str:
    return content_digest(
        f"{_CASE_IDENTITY_SCHEMA}\0".encode()
        + canonical_json_bytes(case.model_dump(mode="json", exclude={"case_id"}))
    )


class PinnedContainerImage(_StrictFrozenModel):
    schema_version: Literal["nodelm.pinned-container-image/v1"] = "nodelm.pinned-container-image/v1"
    source_image: NonEmptyStr
    image_digest: NonEmptyStr
    runtime_artifact_sha256: Sha256 | None = None
    runtime_artifact_bytes: int | None = Field(default=None, ge=1)

    @field_validator("image_digest")
    @classmethod
    def require_digest_pin(cls, value: str) -> str:
        if _IMAGE_DIGEST.fullmatch(value) is None:
            raise ValueError("container image must be pinned by sha256 digest")
        return value

    @model_validator(mode="after")
    def require_complete_runtime_artifact_identity(self) -> PinnedContainerImage:
        if (self.runtime_artifact_sha256 is None) != (self.runtime_artifact_bytes is None):
            raise ValueError("runtime artifact identity must include digest and byte count")
        return self


class ResolutionCanaryImageLock(_StrictFrozenModel):
    schema_version: Literal["nodelm.resolution-canary-image-lock/v1"] = (
        "nodelm.resolution-canary-image-lock/v1"
    )
    workset_sha256: Sha256
    evaluator_repository_id: Literal["SWE-rebench/SWE-rebench-V2"]
    evaluator_revision: CommitSha
    runtime: Literal["rootless-podman", "seccomp-chroot"] = "rootless-podman"
    images: tuple[PinnedContainerImage, ...] = Field(min_length=1)

    @field_validator("evaluator_revision")
    @classmethod
    def require_evaluator_revision(cls, value: str) -> str:
        normalized = value.casefold()
        if normalized != SWE_REBENCH_EVALUATOR_REVISION:
            raise ValueError("image lock evaluator revision is not the approved pin")
        return normalized

    @field_validator("images")
    @classmethod
    def require_sorted_unique_images(
        cls,
        images: tuple[PinnedContainerImage, ...],
    ) -> tuple[PinnedContainerImage, ...]:
        names = tuple(image.source_image for image in images)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("image lock source images must be unique and sorted")
        return images

    @model_validator(mode="after")
    def require_runtime_artifact_identities(self) -> ResolutionCanaryImageLock:
        if self.runtime == "seccomp-chroot" and any(
            image.runtime_artifact_sha256 is None for image in self.images
        ):
            raise ValueError("seccomp chroot images require local OCI manifest identities")
        if self.runtime == "rootless-podman" and any(
            image.runtime_artifact_sha256 is not None for image in self.images
        ):
            raise ValueError("Podman image locks must not carry local OCI identities")
        return self

    def by_source_image(self) -> dict[str, PinnedContainerImage]:
        return {image.source_image: image for image in self.images}


class ResolutionCanaryAttemptEvidence(_StrictFrozenModel):
    outcome: OutcomeCategory
    exit_code: int | None
    duration_seconds: float = Field(ge=0)
    output_sha256: Sha256
    output_bytes: int = Field(ge=0)
    expected_test_count: int = Field(ge=0)
    observed_expected_test_count: int = Field(ge=0)


class ResolutionCanaryCaseResult(_StrictFrozenModel):
    schema_version: Literal["nodelm.resolution-canary-case-result/v1"] = (
        "nodelm.resolution-canary-case-result/v1"
    )
    case_id: Sha256
    kind: CaseKind
    source_id: Sha256
    status: VerificationStatus
    reason: NonEmptyStr
    task_resolved: StrictBool | None
    expected_resolved: StrictBool | None
    label_agreement: StrictBool | None
    image: PinnedContainerImage
    baseline: ResolutionCanaryAttemptEvidence
    candidate: ResolutionCanaryAttemptEvidence
    sandbox_evidence: dict[str, Any] = Field(min_length=1)


class ResolutionCanaryPrivateCaseEvidence(_StrictFrozenModel):
    """Private raw logs paired atomically with their sanitized case result."""

    schema_version: Literal["nodelm.resolution-canary-private-case-evidence/v1"] = (
        "nodelm.resolution-canary-private-case-evidence/v1"
    )
    result: ResolutionCanaryCaseResult
    baseline_output: StrictStr
    candidate_output: StrictStr

    @model_validator(mode="after")
    def verify_output_identities(self) -> ResolutionCanaryPrivateCaseEvidence:
        baseline = self.baseline_output.encode("utf-8")
        candidate = self.candidate_output.encode("utf-8")
        if (
            content_digest(baseline) != self.result.baseline.output_sha256
            or len(baseline) != self.result.baseline.output_bytes
        ):
            raise ValueError("private baseline output does not match sanitized evidence")
        if (
            content_digest(candidate) != self.result.candidate.output_sha256
            or len(candidate) != self.result.candidate.output_bytes
        ):
            raise ValueError("private candidate output does not match sanitized evidence")
        return self


LogParser = Callable[[str, str], Mapping[str, str]]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def resolution_canary_output(result: CommandResult) -> str:
    return result.stdout + "\n" + result.stderr


def _attempt_evidence(
    result: CommandResult,
    *,
    expected: frozenset[str],
    parsed: Mapping[str, str],
) -> ResolutionCanaryAttemptEvidence:
    output = resolution_canary_output(result).encode("utf-8")
    return ResolutionCanaryAttemptEvidence(
        outcome=result.outcome,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        output_sha256=content_digest(output),
        output_bytes=len(output),
        expected_test_count=len(expected),
        observed_expected_test_count=len(expected & set(parsed)),
    )


class ResolutionCanaryOracle:
    """Convert isolated test attempts into a fail-closed resolution observation."""

    def __init__(self, parser: LogParser) -> None:
        self._parser = parser

    def _parse(self, case: ResolutionCanaryCase, result: CommandResult) -> dict[str, str]:
        parsed = self._parser(case.task.log_parser, resolution_canary_output(result))
        normalized = {
            _normalize_test_name(name): status.upper()
            for name, status in parsed.items()
            if isinstance(name, str) and isinstance(status, str)
        }
        if case.task.log_parser == "parse_log_js_4":
            expected = {
                _normalize_test_name(name)
                for name in (*case.task.fail_to_pass, *case.task.pass_to_pass)
            }
            for line in resolution_canary_output(result).splitlines():
                match = _NUMBERED_JS_FAILURE.fullmatch(line)
                if match is None:
                    continue
                name = _normalize_test_name(match.group(1).removesuffix(":"))
                if name in expected and name not in normalized:
                    normalized[name] = "FAILED"
        return normalized

    def evaluate(
        self,
        case: ResolutionCanaryCase,
        *,
        image: PinnedContainerImage,
        baseline: CommandResult,
        candidate: CommandResult,
        sandbox_evidence: dict[str, Any],
    ) -> ResolutionCanaryCaseResult:
        expected_fail = frozenset(_normalize_test_name(item) for item in case.task.fail_to_pass)
        expected_pass = frozenset(_normalize_test_name(item) for item in case.task.pass_to_pass)
        expected = expected_fail | expected_pass
        baseline_parsed = self._parse(case, baseline)
        candidate_parsed = self._parse(case, candidate)
        baseline_evidence = _attempt_evidence(baseline, expected=expected, parsed=baseline_parsed)
        candidate_evidence = _attempt_evidence(
            candidate, expected=expected, parsed=candidate_parsed
        )

        reason = "canary_oracle_validated"
        task_resolved: bool | None = None
        label_agreement: bool | None = None
        status = VerificationStatus.FAIL
        if baseline.outcome not in _EVALUATION_OUTCOMES:
            reason = "baseline_sandbox_execution_failed"
        elif candidate.outcome not in _EVALUATION_OUTCOMES:
            reason = "candidate_sandbox_execution_failed"
        elif (
            baseline.stdout_truncated
            or baseline.stderr_truncated
            or candidate.stdout_truncated
            or candidate.stderr_truncated
        ):
            reason = "sandbox_output_truncated"
        elif not expected.issubset(baseline_parsed) or not expected.issubset(candidate_parsed):
            reason = "incomplete_expected_test_evidence"
        elif any(baseline_parsed[name] != "PASSED" for name in expected_pass) or any(
            baseline_parsed[name] != "FAILED" for name in expected_fail
        ):
            reason = "failing_baseline_not_reproduced"
        else:
            task_resolved = all(candidate_parsed[name] == "PASSED" for name in expected)
            if task_resolved and candidate.outcome is not OutcomeCategory.SUCCESS:
                task_resolved = None
                reason = "candidate_exit_status_contradicts_test_evidence"
            else:
                label_agreement = (
                    None
                    if case.expected_resolved is None
                    else task_resolved is case.expected_resolved
                )
                if label_agreement is not False:
                    status = VerificationStatus.PASS
                else:
                    reason = "transferred_label_disagrees_with_repository_oracle"

        return ResolutionCanaryCaseResult(
            case_id=case.case_id,
            kind=case.kind,
            source_id=case.source_id,
            status=status,
            reason=reason,
            task_resolved=task_resolved,
            expected_resolved=case.expected_resolved,
            label_agreement=label_agreement,
            image=image,
            baseline=baseline_evidence,
            candidate=candidate_evidence,
            sandbox_evidence=sandbox_evidence,
        )


def _module_path(module: ModuleType) -> Path:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise ResolutionCanaryError("pinned evaluator module has no source path")
    return Path(source).resolve()


class PinnedEvaluatorLogParser:
    """Load only the integrity-pinned upstream parser implementation."""

    def __init__(
        self,
        evaluator_root: Path,
        *,
        expected_parser_sha256: str = SWE_REBENCH_LOG_PARSERS_SHA256,
        expected_eval_sha256: str = SWE_REBENCH_EVAL_SCRIPT_SHA256,
        expected_constants_sha256: str = SWE_REBENCH_CONSTANTS_SHA256,
    ) -> None:
        if evaluator_root.is_symlink():
            raise ResolutionCanaryError("pinned evaluator root must not be a symlink")
        root = evaluator_root.resolve()
        parser_path = root / "lib" / "agent" / "log_parsers.py"
        constants_path = root / "lib" / "agent" / "swe_constants.py"
        eval_path = root / "scripts" / "eval.py"
        for path in (parser_path, constants_path, eval_path):
            if not path.is_file() or path.is_symlink():
                raise ResolutionCanaryError("pinned evaluator checkout is incomplete or unsafe")
        parser_identity = file_identity(parser_path)
        eval_identity = file_identity(eval_path)
        constants_identity = file_identity(constants_path)
        if parser_identity[0] != expected_parser_sha256:
            raise ResolutionCanaryError("pinned evaluator log parser digest mismatch")
        if eval_identity[0] != expected_eval_sha256:
            raise ResolutionCanaryError("pinned evaluator script digest mismatch")
        if constants_identity[0] != expected_constants_sha256:
            raise ResolutionCanaryError("pinned evaluator constants digest mismatch")

        module_name = f"_nodelm_swe_rebench_log_parsers_{parser_identity[0]}"
        specification = spec_from_file_location(module_name, parser_path)
        if specification is None or specification.loader is None:
            raise ResolutionCanaryError("pinned evaluator log parser cannot be loaded")
        module = module_from_spec(specification)
        sys.path.insert(0, str(root))
        try:
            specification.loader.exec_module(module)
        except Exception as error:
            raise ResolutionCanaryError("pinned evaluator log parser failed to load") from error
        finally:
            with suppress(ValueError):
                sys.path.remove(str(root))
        constants_module = sys.modules.get("lib.agent.swe_constants")
        if constants_module is None or _module_path(constants_module) != constants_path.resolve():
            raise ResolutionCanaryError("pinned evaluator constants resolved outside checkout")
        parsers = getattr(module, "NAME_TO_PARSER", None)
        if not isinstance(parsers, Mapping) or not parsers:
            raise ResolutionCanaryError("pinned evaluator exposes no named log parsers")
        self._parsers = dict(parsers)
        self.evaluator_root = root
        self.parser_sha256 = parser_identity[0]
        self.eval_sha256 = eval_identity[0]
        self.constants_sha256 = constants_identity[0]

    def __call__(self, parser_name: str, output: str) -> Mapping[str, str]:
        parser = self._parsers.get(parser_name)
        if not callable(parser):
            raise ResolutionCanaryError("private task requests an unknown pinned log parser")
        parsed = parser(output)
        if not isinstance(parsed, Mapping) or any(
            not isinstance(name, str) or not isinstance(status, str)
            for name, status in parsed.items()
        ):
            raise ResolutionCanaryError("pinned evaluator log parser returned invalid evidence")
        return cast(Mapping[str, str], parsed)


class PodmanImageLocker:
    """Pull selected source tags, then record their immutable registry digests."""

    def __init__(
        self,
        *,
        executable: str = "podman",
        pull_timeout_seconds: float = 7_200,
    ) -> None:
        if not executable or "\0" in executable:
            raise ValueError("image locker executable must be a non-empty NUL-free string")
        if pull_timeout_seconds <= 0:
            raise ValueError("image pull timeout must be greater than zero")
        self.executable = executable
        self.pull_timeout_seconds = pull_timeout_seconds

    @staticmethod
    def _repository_name(source_image: str) -> str:
        without_digest = source_image.split("@", 1)[0]
        last_slash = without_digest.rfind("/")
        last_colon = without_digest.rfind(":")
        if last_colon > last_slash:
            return without_digest[:last_colon]
        return without_digest

    @classmethod
    def _select_repo_digest(cls, source_image: str, output: str) -> str:
        try:
            values = cast(object, json.loads(output))
        except (TypeError, ValueError) as error:
            raise ResolutionCanaryError("Podman returned invalid image digest evidence") from error
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ResolutionCanaryError("Podman returned invalid image digest evidence")
        repository = cls._repository_name(source_image)
        matches = sorted(
            value
            for value in cast(list[str], values)
            if value.startswith(f"{repository}@sha256:")
            and _IMAGE_DIGEST.fullmatch(value) is not None
        )
        if len(matches) != 1:
            raise ResolutionCanaryError(
                "Podman did not expose one matching immutable repository digest"
            )
        return matches[0]

    def lock(
        self,
        source_images: tuple[str, ...],
        *,
        workspace: Path,
        workset_sha256: str,
    ) -> ResolutionCanaryImageLock:
        names = tuple(sorted(set(source_images)))
        if not names:
            raise ResolutionCanaryError("image locking requires at least one source image")
        executor = CommandExecutor(workspace, default_max_output_bytes=1024 * 1024)
        policy = CommandPolicy(workspace)
        rootless = executor.run(
            policy.generic(
                (self.executable, "info", "--format", "{{.Host.Security.Rootless}}"),
                trusted_local=True,
                timeout_seconds=20,
            )
        )
        if (
            rootless.outcome is not OutcomeCategory.SUCCESS
            or rootless.stdout.strip().lower() != "true"
        ):
            raise ResolutionCanaryError("Podman rootless capability probe failed")

        locked: list[PinnedContainerImage] = []
        for source_image in names:
            pull = executor.run(
                policy.generic(
                    (self.executable, "pull", "--quiet", source_image),
                    trusted_local=True,
                    timeout_seconds=self.pull_timeout_seconds,
                    max_output_bytes=1024 * 1024,
                )
            )
            if pull.outcome is not OutcomeCategory.SUCCESS:
                raise ResolutionCanaryError("selected canary image pull failed")
            inspect = executor.run(
                policy.generic(
                    (
                        self.executable,
                        "image",
                        "inspect",
                        "--format",
                        "{{json .RepoDigests}}",
                        source_image,
                    ),
                    trusted_local=True,
                    timeout_seconds=30,
                )
            )
            if inspect.outcome is not OutcomeCategory.SUCCESS:
                raise ResolutionCanaryError("selected canary image inspection failed")
            locked.append(
                PinnedContainerImage(
                    source_image=source_image,
                    image_digest=self._select_repo_digest(source_image, inspect.stdout.strip()),
                )
            )
        return ResolutionCanaryImageLock(
            workset_sha256=workset_sha256,
            evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
            evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
            images=tuple(locked),
        )


def _image_cache_key(source_image: str) -> str:
    return content_digest(f"nodelm.oci-image-cache/v1\0{source_image}".encode())


class SkopeoChrootImageLocker:
    """Pull digest-pinned images into local OCI layouts without nested namespaces."""

    def __init__(
        self,
        *,
        image_root: Path,
        executable: str = "skopeo",
        pull_timeout_seconds: float = 7_200,
    ) -> None:
        if not executable or "\0" in executable:
            raise ValueError("image locker executable must be a non-empty NUL-free string")
        if pull_timeout_seconds <= 0:
            raise ValueError("image pull timeout must be greater than zero")
        if not image_root.is_absolute() or image_root.is_symlink():
            raise ValueError("OCI image root must be an absolute non-symlink path")
        self.image_root = image_root.resolve()
        self.executable = executable
        self.pull_timeout_seconds = pull_timeout_seconds

    def _run(
        self,
        executor: CommandExecutor,
        policy: CommandPolicy,
        command: tuple[str, ...],
        *,
        timeout_seconds: float,
        max_output_bytes: int = 1024 * 1024,
    ) -> CommandResult:
        return executor.run(
            policy.generic(
                command,
                trusted_local=True,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
        )

    def lock(
        self,
        source_images: tuple[str, ...],
        *,
        workspace: Path,
        workset_sha256: str,
    ) -> ResolutionCanaryImageLock:
        names = tuple(sorted(set(source_images)))
        if not names:
            raise ResolutionCanaryError("image locking requires at least one source image")
        self.image_root.mkdir(parents=True, exist_ok=True)
        if self.image_root.is_symlink() or not self.image_root.is_dir():
            raise ResolutionCanaryError("OCI image root is unsafe")
        executor = CommandExecutor(workspace, default_max_output_bytes=1024 * 1024)
        policy = CommandPolicy(workspace)
        locked: list[PinnedContainerImage] = []
        for source_image in names:
            inspected = self._run(
                executor,
                policy,
                (
                    self.executable,
                    "inspect",
                    "--format",
                    "{{.Digest}}",
                    f"docker://{source_image}",
                ),
                timeout_seconds=60,
            )
            digest = inspected.stdout.strip()
            if (
                inspected.outcome is not OutcomeCategory.SUCCESS
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise ResolutionCanaryError("selected canary image digest inspection failed")
            repository = PodmanImageLocker._repository_name(source_image)
            pinned = f"{repository}@{digest}"
            layout = self.image_root / _image_cache_key(source_image) / "layout"
            layout.mkdir(parents=True, exist_ok=True)
            copied = self._run(
                executor,
                policy,
                (
                    self.executable,
                    "copy",
                    "--format",
                    "oci",
                    f"docker://{pinned}",
                    f"oci:{layout}:canary",
                ),
                timeout_seconds=self.pull_timeout_seconds,
            )
            if copied.outcome is not OutcomeCategory.SUCCESS:
                raise ResolutionCanaryError("selected canary OCI image pull failed")
            raw = self._run(
                executor,
                policy,
                (self.executable, "inspect", "--raw", f"oci:{layout}:canary"),
                timeout_seconds=30,
            )
            if raw.outcome is not OutcomeCategory.SUCCESS or not raw.stdout:
                raise ResolutionCanaryError("local OCI image manifest inspection failed")
            manifest = raw.stdout.encode("utf-8")
            locked.append(
                PinnedContainerImage(
                    source_image=source_image,
                    image_digest=pinned,
                    runtime_artifact_sha256=content_digest(manifest),
                    runtime_artifact_bytes=len(manifest),
                )
            )
        return ResolutionCanaryImageLock(
            workset_sha256=workset_sha256,
            evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
            evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
            runtime="seccomp-chroot",
            images=tuple(locked),
        )


def _attempt_script(case: ResolutionCanaryCase, *, include_model_patch: bool) -> str:
    commands = [
        "set -euo pipefail",
        f'test "$(git rev-parse HEAD)" = "{case.task.base_commit}"',
        "git reset --hard HEAD",
    ]
    if include_model_patch:
        commands.append(
            "git apply -v --3way --recount --ignore-space-change "
            "--whitespace=nowarn /nodelm-input/model.patch"
        )
    commands.append(
        "git apply -v --3way --recount --ignore-space-change "
        "--whitespace=nowarn /nodelm-input/test.patch"
    )
    commands.extend(case.task.test_commands)
    return "\n".join(commands)


class SWERebenchSeccompChrootSandbox:
    """Run fresh OCI rootfs clones with seccomp, chroot, UID, and resource isolation."""

    def __init__(
        self,
        *,
        image_root: Path,
        skopeo: str = "skopeo",
        umoci: str = "umoci",
        timeout_seconds: float = 1_800,
        max_output_bytes: int = 16 * 1024 * 1024,
        sandbox_uid: int = 61_000,
    ) -> None:
        if not image_root.is_absolute() or image_root.is_symlink():
            raise ValueError("OCI image root must be an absolute non-symlink path")
        if any(not executable or "\0" in executable for executable in (skopeo, umoci)):
            raise ValueError("sandbox executables must be non-empty NUL-free strings")
        if timeout_seconds <= 0 or max_output_bytes <= 0 or sandbox_uid <= 0:
            raise ValueError("sandbox bounds must be greater than zero")
        self.image_root = image_root.resolve()
        self.skopeo = skopeo
        self.umoci = umoci
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.sandbox_uid = sandbox_uid
        self._ready_bundles: dict[str, Path] = {}
        self._manifest_probe: CommandResult | None = None
        self._unpack: CommandResult | None = None
        self._clone: CommandResult | None = None
        self._cleanup_verified = False

    @staticmethod
    def _manifest_identity(result: CommandResult) -> tuple[str, int]:
        payload = result.stdout.encode("utf-8")
        return content_digest(payload), len(payload)

    def _ensure_ready(self, workspace: Path, image: PinnedContainerImage) -> Path:
        cached = self._ready_bundles.get(image.image_digest)
        if cached is not None:
            return cached
        if image.runtime_artifact_sha256 is None or image.runtime_artifact_bytes is None:
            raise ResolutionCanaryError("seccomp chroot image lacks a local OCI identity")
        cache = self.image_root / _image_cache_key(image.source_image)
        layout = cache / "layout"
        bundle = cache / "bundle"
        executor = CommandExecutor(workspace, default_max_output_bytes=1024 * 1024)
        policy = CommandPolicy(workspace)
        if not (layout / "index.json").is_file():
            layout.mkdir(parents=True, exist_ok=True)
            restored = executor.run(
                policy.generic(
                    (
                        self.skopeo,
                        "copy",
                        "--format",
                        "oci",
                        f"docker://{image.image_digest}",
                        f"oci:{layout}:canary",
                    ),
                    trusted_local=True,
                    timeout_seconds=7_200,
                    max_output_bytes=1024 * 1024,
                )
            )
            if restored.outcome is not OutcomeCategory.SUCCESS:
                raise ResolutionCanaryError("locked OCI image restoration failed")
        manifest_probe = executor.run(
            policy.generic(
                (self.skopeo, "inspect", "--raw", f"oci:{layout}:canary"),
                trusted_local=True,
                timeout_seconds=30,
                max_output_bytes=1024 * 1024,
            )
        )
        self._manifest_probe = manifest_probe
        if manifest_probe.outcome is not OutcomeCategory.SUCCESS or self._manifest_identity(
            manifest_probe
        ) != (image.runtime_artifact_sha256, image.runtime_artifact_bytes):
            raise ResolutionCanaryError("local OCI image does not match its image lock")
        marker = bundle / ".nodelm-oci-manifest"
        if bundle.exists():
            if (
                bundle.is_symlink()
                or not (bundle / "rootfs").is_dir()
                or not (bundle / "config.json").is_file()
                or not marker.is_file()
                or marker.is_symlink()
                or marker.read_text(encoding="ascii") != image.runtime_artifact_sha256
            ):
                raise ResolutionCanaryError("cached OCI bundle is incomplete or unsafe")
        else:
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".unpack-", dir=cache) as temporary_name:
                temporary = Path(temporary_name).resolve()
                staged = temporary / "bundle"
                unpack = executor.run(
                    policy.generic(
                        (self.umoci, "unpack", "--image", f"{layout}:canary", str(staged)),
                        trusted_local=True,
                        timeout_seconds=1_800,
                        max_output_bytes=1024 * 1024,
                    )
                )
                self._unpack = unpack
                if unpack.outcome is not OutcomeCategory.SUCCESS:
                    raise ResolutionCanaryError("local OCI image unpack failed")
                marker_path = staged / ".nodelm-oci-manifest"
                marker_path.write_text(image.runtime_artifact_sha256, encoding="ascii")
                staged.rename(bundle)
        self._ready_bundles[image.image_digest] = bundle
        return bundle

    @staticmethod
    def _image_environment(bundle: Path) -> tuple[str, ...]:
        try:
            value = cast(object, json.loads((bundle / "config.json").read_bytes()))
            process = cast(dict[str, object], value).get("process")
            environment = cast(dict[str, object], process).get("env")
        except (AttributeError, TypeError, ValueError) as error:
            raise ResolutionCanaryError("OCI runtime configuration is invalid") from error
        if not isinstance(environment, list) or any(
            not isinstance(item, str) or "=" not in item or "\0" in item for item in environment
        ):
            raise ResolutionCanaryError("OCI runtime environment is invalid")
        merged = {item.partition("=")[0]: item for item in cast(list[str], environment)}
        merged.update(
            {
                "CI": "CI=true",
                "HOME": "HOME=/tmp/nodelm-home",
                "_JAVA_OPTIONS": "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
            }
        )
        if "PATH" not in merged:
            merged["PATH"] = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        return tuple(merged[key] for key in sorted(merged))

    @staticmethod
    def _validated_workdir(workdir: object) -> str:
        if (
            not isinstance(workdir, str)
            or "\0" in workdir
            or not workdir.startswith("/")
            or workdir == "/"
            or any(part in {"", ".", ".."} for part in workdir.split("/")[1:])
        ):
            raise ResolutionCanaryError("OCI runtime working directory is invalid")
        return workdir

    @classmethod
    def _image_workdir(cls, bundle: Path) -> str:
        try:
            value = cast(object, json.loads((bundle / "config.json").read_bytes()))
            process = cast(dict[str, object], value).get("process")
            workdir = cast(dict[str, object], process).get("cwd")
        except (AttributeError, TypeError, ValueError) as error:
            raise ResolutionCanaryError("OCI runtime working directory is invalid") from error
        return cls._validated_workdir(workdir)

    def _prepare_rootfs(self, rootfs: Path, workdir: str) -> None:
        workdir = self._validated_workdir(workdir)
        try:
            resolved_rootfs = rootfs.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ResolutionCanaryError("OCI rootfs is missing or unsafe") from error
        if rootfs.is_symlink() or not resolved_rootfs.is_dir():
            raise ResolutionCanaryError("OCI rootfs is missing or unsafe")
        repository = resolved_rootfs
        for part in workdir.split("/")[1:]:
            repository /= part
            if repository.is_symlink():
                raise ResolutionCanaryError("OCI repository workdir contains a symlink")
        try:
            resolved_repository = repository.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ResolutionCanaryError(
                "OCI rootfs does not contain the expected repository"
            ) from error
        if (
            not resolved_repository.is_relative_to(resolved_rootfs)
            or resolved_repository != repository
            or not resolved_repository.is_dir()
        ):
            raise ResolutionCanaryError("OCI rootfs does not contain the expected repository")
        repository = resolved_repository
        inputs = rootfs / "nodelm-input"
        inputs.mkdir(mode=0o700)
        home = rootfs / "tmp" / "nodelm-home"
        home.mkdir(parents=True, mode=0o700, exist_ok=True)
        (rootfs / "tmp").chmod(0o1777)
        devices = rootfs / "dev"
        for entry in devices.iterdir():
            mode = entry.lstat().st_mode
            if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
                raise ResolutionCanaryError("OCI rootfs contains a host device node")
        for device in ("null", "zero", "random", "urandom"):
            path = devices / device
            if not path.exists():
                if device == "zero":
                    with path.open("wb") as stream:
                        stream.truncate(16 * 1024 * 1024)
                elif device in {"random", "urandom"}:
                    path.write_bytes(os.urandom(1024 * 1024))
                else:
                    path.touch()
                path.chmod(0o666)
        for parent, directories, files in os.walk(repository):
            os.chown(parent, self.sandbox_uid, self.sandbox_uid)
            for name in (*directories, *files):
                os.lchown(Path(parent) / name, self.sandbox_uid, self.sandbox_uid)
        os.chown(inputs, self.sandbox_uid, self.sandbox_uid)
        os.chown(home, self.sandbox_uid, self.sandbox_uid)

    def run(
        self,
        case: ResolutionCanaryCase,
        image: PinnedContainerImage,
        *,
        include_model_patch: bool,
    ) -> CommandResult:
        if os.geteuid() != 0:
            raise ResolutionCanaryError("seccomp chroot sandbox requires a root launcher")
        if image.source_image != case.task.image_name:
            raise ResolutionCanaryError("pinned image does not match private task image")
        attempt_root = self.image_root.parent / "attempts"
        attempt_root.mkdir(parents=True, exist_ok=True)
        bundle = self._ensure_ready(attempt_root, image)
        with tempfile.TemporaryDirectory(
            prefix="nodelm-resolution-canary-", dir=attempt_root
        ) as temporary_name:
            temporary = Path(temporary_name).resolve()
            rootfs = temporary / "rootfs"
            executor = CommandExecutor(temporary, default_max_output_bytes=self.max_output_bytes)
            policy = CommandPolicy(temporary)
            clone = executor.run(
                policy.generic(
                    (
                        "cp",
                        "-a",
                        "--reflink=always",
                        str(bundle / "rootfs"),
                        str(rootfs),
                    ),
                    trusted_local=True,
                    timeout_seconds=600,
                    max_output_bytes=1024 * 1024,
                )
            )
            self._clone = clone
            if clone.outcome is not OutcomeCategory.SUCCESS:
                raise ResolutionCanaryError("fresh OCI rootfs clone failed")
            workdir = self._image_workdir(bundle)
            self._prepare_rootfs(rootfs, workdir)
            inputs = rootfs / "nodelm-input"
            if include_model_patch:
                (inputs / "model.patch").write_text(case.model_patch, encoding="utf-8")
            (inputs / "test.patch").write_text(case.task.test_patch, encoding="utf-8")
            for path in inputs.iterdir():
                os.chown(path, self.sandbox_uid, self.sandbox_uid)
                path.chmod(0o400)
            sched_getaffinity = getattr(os, "sched_getaffinity", None)
            if not callable(sched_getaffinity):
                raise ResolutionCanaryError("CPU affinity is unavailable")
            cpus = tuple(sorted(sched_getaffinity(0))[:2])
            if len(cpus) != 2:
                raise ResolutionCanaryError("seccomp chroot sandbox requires two available CPUs")
            command = [
                sys.executable,
                str(Path(__file__).with_name("chroot_launcher.py").resolve()),
                "--rootfs",
                str(rootfs),
                "--workdir",
                workdir,
                "--uid",
                str(self.sandbox_uid),
                "--gid",
                str(self.sandbox_uid),
                "--cpus",
                ",".join(str(cpu) for cpu in cpus),
                "--memory-bytes",
                str(4 * 1024 * 1024 * 1024),
                "--pids",
                "512",
            ]
            for item in self._image_environment(bundle):
                command.extend(("--env", item))
            command.extend(
                (
                    "--",
                    "/bin/bash",
                    "-lc",
                    _attempt_script(case, include_model_patch=include_model_patch),
                )
            )
            result = executor.run(
                policy.generic(
                    tuple(command),
                    trusted_local=True,
                    timeout_seconds=self.timeout_seconds,
                    failure_outcome=OutcomeCategory.TEST_FAILURE,
                    max_output_bytes=self.max_output_bytes,
                )
            )
        self._cleanup_verified = not temporary.exists()
        if not self._cleanup_verified:
            return replace(
                result,
                outcome=OutcomeCategory.INTERNAL_FAILURE,
                stderr=(result.stderr.rstrip() + "\nsandbox rootfs cleanup failed").lstrip(),
            )
        return result

    def evidence(self) -> dict[str, Any]:
        def summary(result: CommandResult | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {
                "outcome": result.outcome.value,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }

        return {
            "schema_version": "nodelm.swe-rebench-seccomp-chroot/v1",
            "backend": "seccomp-chroot",
            "network": "seccomp-denied",
            "implicit_pull": False,
            "pids_limit": 512,
            "memory": "4g monitored aggregate and per-process address space",
            "cpus": 2,
            "uid": self.sandbox_uid,
            "manifest_probe": summary(self._manifest_probe),
            "unpack": summary(self._unpack),
            "clone": summary(self._clone),
            "cleanup_verified": self._cleanup_verified,
        }


class SWERebenchPodmanSandbox:
    """Run model-authored patches in preloaded, digest-pinned rootless Podman images."""

    def __init__(
        self,
        *,
        executable: str = "podman",
        timeout_seconds: float = 1_800,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not executable or "\0" in executable:
            raise ValueError("sandbox executable must be a non-empty NUL-free string")
        if timeout_seconds <= 0:
            raise ValueError("sandbox timeout must be greater than zero")
        if max_output_bytes <= 0:
            raise ValueError("sandbox output bound must be greater than zero")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self._ready_images: set[str] = set()
        self._rootless_probe: CommandResult | None = None
        self._image_probe: CommandResult | None = None
        self._cleanup: CommandResult | None = None

    @staticmethod
    def _script(case: ResolutionCanaryCase, *, include_model_patch: bool) -> str:
        return _attempt_script(case, include_model_patch=include_model_patch)

    def command(
        self,
        case: ResolutionCanaryCase,
        image: PinnedContainerImage,
        *,
        patch_dir: Path,
        include_model_patch: bool,
        container_name: str,
        cidfile: Path,
    ) -> tuple[str, ...]:
        if image.source_image != case.task.image_name:
            raise ResolutionCanaryError("pinned image does not match private task image")
        if _IMAGE_DIGEST.fullmatch(image.image_digest) is None:
            raise ResolutionCanaryError("sandbox image is not digest-pinned")
        resolved_patches = patch_dir.resolve()
        if not resolved_patches.is_dir():
            raise ResolutionCanaryError("sandbox patch directory does not exist")
        if re.fullmatch(r"nodelm-resolution-canary-[0-9a-f]{24}", container_name) is None:
            raise ResolutionCanaryError("sandbox container name is invalid")
        resolved_cidfile = cidfile.resolve()
        if resolved_cidfile.exists() or not resolved_cidfile.parent.is_dir():
            raise ResolutionCanaryError("sandbox cidfile must be new inside an existing directory")
        return (
            self.executable,
            "run",
            "--rm",
            f"--name={container_name}",
            f"--cidfile={resolved_cidfile}",
            "--label=io.nodelm.resolution-canary=true",
            f"--label=io.nodelm.resolution-canary.case={case.case_id}",
            "--pull=never",
            "--network=none",
            "--pid=private",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=512",
            "--memory=4g",
            "--cpus=2",
            "--ipc=private",
            "--ulimit=nofile=4096:4096",
            "--ulimit=fsize=1073741824:1073741824",
            "--ulimit=core=0:0",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=1073741824",
            f"--volume={resolved_patches}:/nodelm-input:ro",
            "--env=_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
            "--entrypoint=/bin/bash",
            image.image_digest,
            "-lc",
            self._script(case, include_model_patch=include_model_patch),
        )

    def _ensure_ready(self, workspace: Path, image: PinnedContainerImage) -> None:
        if image.image_digest in self._ready_images:
            return
        executor = CommandExecutor(workspace)
        policy = CommandPolicy(workspace)
        if self._rootless_probe is None:
            rootless = executor.run(
                policy.generic(
                    (self.executable, "info", "--format", "{{.Host.Security.Rootless}}"),
                    trusted_local=True,
                    timeout_seconds=20,
                )
            )
            self._rootless_probe = rootless
        rootless = self._rootless_probe
        if (
            rootless.outcome is not OutcomeCategory.SUCCESS
            or rootless.stdout.strip().lower() != "true"
        ):
            raise ResolutionCanaryError("Podman rootless capability probe failed")
        image_probe = executor.run(
            policy.generic(
                (self.executable, "image", "exists", image.image_digest),
                trusted_local=True,
                timeout_seconds=20,
            )
        )
        self._image_probe = image_probe
        if image_probe.outcome is not OutcomeCategory.SUCCESS:
            raise ResolutionCanaryError("digest-pinned canary image is not preloaded")
        self._ready_images.add(image.image_digest)

    def run(
        self,
        case: ResolutionCanaryCase,
        image: PinnedContainerImage,
        *,
        include_model_patch: bool,
    ) -> CommandResult:
        with tempfile.TemporaryDirectory(prefix="nodelm-resolution-canary-") as temporary_name:
            temporary = Path(temporary_name).resolve()
            if include_model_patch:
                (temporary / "model.patch").write_text(case.model_patch, encoding="utf-8")
            (temporary / "test.patch").write_text(case.task.test_patch, encoding="utf-8")
            self._ensure_ready(temporary, image)
            container_name = f"nodelm-resolution-canary-{secrets.token_hex(12)}"
            cidfile = temporary / "container.cid"
            executor = CommandExecutor(temporary, default_max_output_bytes=self.max_output_bytes)
            policy = CommandPolicy(temporary)
            result: CommandResult | None = None
            cleanup: CommandResult
            try:
                result = executor.run(
                    policy.generic(
                        self.command(
                            case,
                            image,
                            patch_dir=temporary,
                            include_model_patch=include_model_patch,
                            container_name=container_name,
                            cidfile=cidfile,
                        ),
                        trusted_local=True,
                        timeout_seconds=self.timeout_seconds,
                        failure_outcome=OutcomeCategory.TEST_FAILURE,
                        max_output_bytes=self.max_output_bytes,
                    )
                )
            finally:
                cleanup = executor.run(
                    policy.generic(
                        (
                            self.executable,
                            "rm",
                            "--force",
                            "--ignore",
                            "--time=0",
                            "--",
                            container_name,
                        ),
                        trusted_local=True,
                        timeout_seconds=20,
                    )
                )
                self._cleanup = cleanup
            assert result is not None
            if cleanup.outcome is not OutcomeCategory.SUCCESS:
                return replace(
                    result,
                    outcome=OutcomeCategory.INTERNAL_FAILURE,
                    stderr=(
                        result.stderr.rstrip() + "\nsandbox container cleanup could not be verified"
                    ).lstrip(),
                )
            return result

    def evidence(self) -> dict[str, Any]:
        def command_summary(result: CommandResult | None) -> dict[str, Any] | None:
            if result is None:
                return None
            return {
                "outcome": result.outcome.value,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }

        return {
            "schema_version": "nodelm.swe-rebench-podman/v1",
            "backend": "rootless-podman",
            "network": "none",
            "implicit_pull": False,
            "pids_limit": 512,
            "memory": "4g",
            "cpus": 2,
            "rootless_probe": command_summary(self._rootless_probe),
            "image_probe": command_summary(self._image_probe),
            "cleanup": command_summary(self._cleanup),
        }


__all__ = [
    "SWE_REBENCH_CONSTANTS_SHA256",
    "SWE_REBENCH_EVALUATOR_REPOSITORY_ID",
    "SWE_REBENCH_EVALUATOR_REVISION",
    "SWE_REBENCH_EVAL_SCRIPT_SHA256",
    "SWE_REBENCH_LOG_PARSERS_SHA256",
    "PinnedContainerImage",
    "PinnedEvaluatorLogParser",
    "PodmanImageLocker",
    "ResolutionCanaryCase",
    "ResolutionCanaryCaseResult",
    "ResolutionCanaryError",
    "ResolutionCanaryImageLock",
    "ResolutionCanaryOracle",
    "ResolutionCanaryPrivateCaseEvidence",
    "SWERebenchPodmanSandbox",
    "SWERebenchSeccompChrootSandbox",
    "SWERebenchTask",
    "SkopeoChrootImageLocker",
    "project_swe_rebench_task",
    "resolution_canary_output",
]
