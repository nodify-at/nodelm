from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodelm.models import VerificationStatus


class TrainingLifecycleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "nodelm.training-lifecycle/v1"
    model_id: str
    revision: str
    output_dir: Path
    seed: int = 42
    learning_rate: float = Field(default=2e-5, gt=0)


class TrainingBackend(Protocol):
    def load(self, config: TrainingLifecycleConfig) -> None: ...

    def tokenize(self, samples: tuple[str, ...]) -> None: ...

    def train_step(self) -> None: ...

    def save_checkpoint(self, path: Path) -> None: ...

    def reload_checkpoint(self, path: Path) -> None: ...

    def resume_train_step(self) -> None: ...

    def infer(self, prompt: str) -> str: ...


class TrainingLifecycleReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "nodelm.training-lifecycle-report/v1"
    status: VerificationStatus
    completed_stages: tuple[str, ...]
    stage_durations_seconds: dict[str, float]
    wall_clock_seconds: float = Field(ge=0)
    failed_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    inference_output: str | None = None
    resumed_optimizer_step_completed: bool

    @model_validator(mode="after")
    def require_consistent_resume_evidence(self) -> TrainingLifecycleReport:
        stage_completed = "resume_train_step" in self.completed_stages
        if self.resumed_optimizer_step_completed != stage_completed:
            raise ValueError("resume completion flag must match completed lifecycle stages")
        if self.status is VerificationStatus.PASS and not stage_completed:
            raise ValueError("PASS requires a resumed optimizer step")
        return self


def run_training_lifecycle(
    backend: TrainingBackend,
    config: TrainingLifecycleConfig,
    *,
    samples: tuple[str, ...],
    prompt: str,
) -> TrainingLifecycleReport:
    completed: list[str] = []
    durations: dict[str, float] = {}
    lifecycle_started = perf_counter()
    stages: tuple[tuple[str, Callable[[], None]], ...] = (
        ("load", lambda: backend.load(config)),
        ("tokenize", lambda: backend.tokenize(samples)),
        ("train_step", backend.train_step),
        ("save_checkpoint", lambda: backend.save_checkpoint(config.output_dir)),
        ("reload_checkpoint", lambda: backend.reload_checkpoint(config.output_dir)),
        ("resume_train_step", backend.resume_train_step),
    )
    for name, operation in stages:
        stage_started = perf_counter()
        try:
            operation()
        except Exception as error:
            durations[name] = perf_counter() - stage_started
            return TrainingLifecycleReport(
                status=VerificationStatus.FAIL,
                completed_stages=tuple(completed),
                stage_durations_seconds=durations,
                wall_clock_seconds=perf_counter() - lifecycle_started,
                failed_stage=name,
                error_type=type(error).__name__,
                error_message=str(error),
                resumed_optimizer_step_completed="resume_train_step" in completed,
            )
        durations[name] = perf_counter() - stage_started
        completed.append(name)

    stage_started = perf_counter()
    try:
        output = backend.infer(prompt)
    except Exception as error:
        durations["infer"] = perf_counter() - stage_started
        return TrainingLifecycleReport(
            status=VerificationStatus.FAIL,
            completed_stages=tuple(completed),
            stage_durations_seconds=durations,
            wall_clock_seconds=perf_counter() - lifecycle_started,
            failed_stage="infer",
            error_type=type(error).__name__,
            error_message=str(error),
            resumed_optimizer_step_completed=True,
        )
    durations["infer"] = perf_counter() - stage_started
    completed.append("infer")
    return TrainingLifecycleReport(
        status=VerificationStatus.PASS,
        completed_stages=tuple(completed),
        stage_durations_seconds=durations,
        wall_clock_seconds=perf_counter() - lifecycle_started,
        inference_output=output,
        resumed_optimizer_step_completed=True,
    )
