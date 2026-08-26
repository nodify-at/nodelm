from __future__ import annotations

import re
from collections.abc import Callable, Mapping
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

from nodelm.artifacts import canonical_json_bytes, content_digest
from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.harness import CommandResult, OutcomeCategory
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
_IMAGE_DIGEST = re.compile(
    r"^(?:sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64})$"
)
_TIMING_NORMALIZE_RES = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)
_EVALUATION_OUTCOMES = frozenset({OutcomeCategory.SUCCESS, OutcomeCategory.TEST_FAILURE})


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
            test_patch=_required_string(row, "test_patch"),
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
        raise ResolutionCanaryError(str(error)) from error


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

    @field_validator("image_digest")
    @classmethod
    def require_digest_pin(cls, value: str) -> str:
        if _IMAGE_DIGEST.fullmatch(value) is None:
            raise ValueError("container image must be pinned by sha256 digest")
        return value


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


LogParser = Callable[[str, str], Mapping[str, str]]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _combined_output(result: CommandResult) -> str:
    return result.stdout + "\n" + result.stderr


def _attempt_evidence(
    result: CommandResult,
    *,
    expected: frozenset[str],
    parsed: Mapping[str, str],
) -> ResolutionCanaryAttemptEvidence:
    output = _combined_output(result).encode("utf-8")
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
        parsed = self._parser(case.task.log_parser, _combined_output(result))
        return {
            _normalize_test_name(name): status.upper()
            for name, status in parsed.items()
            if isinstance(name, str) and isinstance(status, str)
        }

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
        elif not expected.issubset(baseline_parsed) or not expected.issubset(candidate_parsed):
            reason = "incomplete_expected_test_evidence"
        elif any(baseline_parsed[name] != "PASSED" for name in expected_pass) or all(
            baseline_parsed[name] == "PASSED" for name in expected_fail
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


__all__ = [
    "PinnedContainerImage",
    "ResolutionCanaryCase",
    "ResolutionCanaryCaseResult",
    "ResolutionCanaryError",
    "ResolutionCanaryOracle",
    "SWERebenchTask",
    "project_swe_rebench_task",
]
