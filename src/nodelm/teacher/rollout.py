from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodelm.artifacts import ArtifactWriteResult, canonical_json_bytes, write_immutable_stream
from nodelm.models import CheckResult, SolveContext, VerificationStatus, stable_model_id


class RolloutToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class RolloutObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_index: int = Field(ge=0)
    status: VerificationStatus
    output: str


class RolloutAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concise_state: str = Field(min_length=1)
    tool_calls: tuple[RolloutToolCall, ...]
    observations: tuple[RolloutObservation, ...]
    patch: str = Field(min_length=1, max_length=1_000_000)
    generation_status: VerificationStatus
    status: VerificationStatus
    result_summary: str = Field(min_length=1)
    execution_results: tuple[CheckResult, ...] = ()
    task_resolved: bool | None = None
    regression_tests_passed: bool | None = None

    @model_validator(mode="after")
    def pass_requires_repository_execution(self) -> RolloutAttempt:
        if self.status is not VerificationStatus.PASS:
            return self
        checks = {check.name: check.status for check in self.execution_results}
        required = {"base_restore", "patch_apply", "tests"}
        if (
            self.generation_status is not VerificationStatus.PASS
            or not required.issubset(checks)
            or any(checks[name] is not VerificationStatus.PASS for name in required)
            or self.task_resolved is not True
            or self.regression_tests_passed is not True
        ):
            raise ValueError(
                "PASS rollout requires successful repository execution evidence for "
                "base_restore, patch_apply, and tests"
            )
        return self


class TeacherAdapter(Protocol):
    model_id: str
    revision: str
    decoding_config: dict[str, Any]

    def solve(self, context: SolveContext, *, seed: int) -> RolloutAttempt: ...


class TeacherExecutionRunner(Protocol):
    def execute(self, context: SolveContext, attempt: RolloutAttempt) -> RolloutAttempt: ...


class RolloutRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "nodelm.teacher-rollout/v1"
    rollout_id: str
    repository: str
    base_commit: str
    task: str
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    decoding_config: dict[str, Any]
    seed: int
    attempt: RolloutAttempt


def run_teacher_rollouts(
    adapter: TeacherAdapter,
    context: SolveContext,
    *,
    seeds: tuple[int, ...],
    runner: TeacherExecutionRunner | None = None,
) -> tuple[RolloutRecord, ...]:
    if not seeds:
        raise ValueError("teacher rollout requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("teacher rollout seeds must be unique")
    records: list[RolloutRecord] = []
    for seed in seeds:
        generated = adapter.solve(context, seed=seed)
        if runner is None:
            if (
                generated.status is VerificationStatus.PASS
                or generated.execution_results
                or generated.task_resolved is not None
                or generated.regression_tests_passed is not None
            ):
                raise ValueError("adapter-supplied repository execution evidence is not trusted")
            attempt = generated
        else:
            untrusted_generation = RolloutAttempt.model_validate(
                {
                    **generated.model_dump(mode="json"),
                    "status": VerificationStatus.UNVERIFIED,
                    "execution_results": (),
                    "task_resolved": None,
                    "regression_tests_passed": None,
                    "result_summary": "teacher generation awaits repository execution",
                }
            )
            attempt = RolloutAttempt.model_validate(
                runner.execute(context, untrusted_generation).model_dump(mode="json")
            )
        rollout_id = stable_model_id(
            {
                "context": context,
                "model_id": adapter.model_id,
                "revision": adapter.revision,
                "decoding_config": adapter.decoding_config,
                "seed": seed,
            }
        )
        records.append(
            RolloutRecord(
                rollout_id=rollout_id,
                repository=context.repository,
                base_commit=context.base_commit,
                task=context.task,
                model_id=adapter.model_id,
                model_revision=adapter.revision,
                decoding_config=adapter.decoding_config,
                seed=seed,
                attempt=attempt,
            )
        )
    return tuple(records)


def write_teacher_rollouts(
    records: tuple[RolloutRecord, ...],
    *,
    output: Path,
) -> ArtifactWriteResult:
    """Persist successful and failed rollout records in one immutable JSONL artifact."""

    if not records:
        raise ValueError("teacher rollout artifact requires at least one record")

    def write_records(stream: BinaryIO) -> None:
        for record in records:
            stream.write(canonical_json_bytes(record.model_dump(mode="json")))

    return write_immutable_stream(output, write_records)
