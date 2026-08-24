from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from nodelm.evaluation.fixture import (
    ExactSourceTransition,
    FixturePatchReport,
    evaluate_model_patch_fixture,
)
from nodelm.evaluation.sandbox import FixtureSandbox
from nodelm.models import SolveContext, VerificationStatus


class CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    repository_id: str
    revision: str | None = None
    license: str | None = None
    parameter_count: int | None = Field(default=None, gt=0)
    architecture: str | None = None
    context_limit: int | None = Field(default=None, gt=0)
    status: VerificationStatus

    @model_validator(mode="after")
    def verified_candidates_are_pinned(self) -> CandidateModel:
        if self.status is VerificationStatus.PASS:
            if not self.revision or re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None:
                raise ValueError("PASS candidate requires a full 40-hex immutable revision")
            if not self.license:
                raise ValueError("PASS candidate requires a license")
        return self


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0)
    gpu_memory_bytes: int | None = Field(default=None, ge=0)
    prompt_tokens_per_second: float | None = Field(default=None, ge=0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0)


class ModelAdapter(Protocol):
    def generate(self, context: SolveContext) -> ModelResponse: ...


class HarnessEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    tool_calls_valid: bool | None = None
    task_resolved: bool | None = None
    regression_tests_passed: bool | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def pass_requires_positive_execution_evidence(self) -> HarnessEvaluation:
        if self.status is VerificationStatus.PASS and (
            self.tool_calls_valid is not True
            or self.task_resolved is not True
            or self.regression_tests_passed is not True
            or not self.evidence
        ):
            raise ValueError("PASS harness evaluation requires all execution gates")
        return self


class EvaluationHarness(Protocol):
    def evaluate(self, response: ModelResponse, context: SolveContext) -> HarnessEvaluation: ...


class FixturePatchHarness:
    """Common protected-fixture evaluation backed by an explicit OS sandbox."""

    def __init__(
        self,
        fixture: Path,
        sandbox: FixtureSandbox | None,
        *,
        expected_context: SolveContext,
        exact_source_transitions: tuple[ExactSourceTransition, ...],
    ) -> None:
        self.fixture = fixture
        self.sandbox = sandbox
        self.expected_context = expected_context
        self.exact_source_transitions = exact_source_transitions

    def evaluate(self, response: ModelResponse, context: SolveContext) -> HarnessEvaluation:
        if context != self.expected_context:
            return HarnessEvaluation(
                status=VerificationStatus.FAIL,
                tool_calls_valid=False,
                task_resolved=False,
                regression_tests_passed=False,
                evidence={
                    "schema_version": "nodelm.candidate-fixture-evidence/v1",
                    "reason": "evaluation context does not match the fixture binding",
                    "expected_context": self.expected_context.model_dump(mode="json"),
                },
            )
        report: FixturePatchReport = evaluate_model_patch_fixture(
            response.text,
            fixture=self.fixture,
            exact_source_transitions=self.exact_source_transitions,
            sandbox=self.sandbox,
        )
        passed = report.status is VerificationStatus.PASS
        return HarnessEvaluation(
            status=report.status,
            tool_calls_valid=all(
                isinstance(call.get("name"), str)
                and bool(call["name"].strip())
                and isinstance(call.get("arguments"), dict)
                for call in response.tool_calls
            ),
            task_resolved=passed,
            regression_tests_passed=passed,
            evidence={
                "schema_version": "nodelm.candidate-fixture-evidence/v1",
                "context": self.expected_context.model_dump(mode="json"),
                "fixture_patch": report.model_dump(mode="json"),
            },
        )


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: CandidateModel
    precision: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    generation_status: VerificationStatus
    status: VerificationStatus
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    tool_calls_valid: bool | None = None
    task_resolved: bool | None = None
    regression_tests_passed: bool | None = None
    gpu_memory_bytes: int | None = None
    prompt_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    harness_evidence: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @model_validator(mode="after")
    def pass_requires_harness_evidence(self) -> CandidateEvaluation:
        if self.status is VerificationStatus.PASS:
            if self.generation_status is not VerificationStatus.PASS:
                raise ValueError("PASS evaluation requires successful generation")
            if (
                self.tool_calls_valid is not True
                or self.task_resolved is not True
                or self.regression_tests_passed is not True
                or not self.harness_evidence
            ):
                raise ValueError(
                    "PASS evaluation requires valid-tool, resolved-task, and "
                    "regression-test evidence"
                )
        return self


def evaluate_adapter(
    candidate: CandidateModel,
    adapter: ModelAdapter,
    context: SolveContext,
    *,
    precision: str,
    backend: str,
    harness: EvaluationHarness | None = None,
) -> CandidateEvaluation:
    if candidate.status is not VerificationStatus.PASS:
        raise ValueError("candidate must be verified and pinned before evaluation")
    response = adapter.generate(context)
    harness_result = (
        harness.evaluate(response, context)
        if harness is not None
        else HarnessEvaluation(status=VerificationStatus.UNVERIFIED)
    )
    return CandidateEvaluation(
        candidate=candidate,
        precision=precision,
        backend=backend,
        generation_status=VerificationStatus.PASS,
        status=harness_result.status,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        wall_clock_seconds=response.wall_clock_seconds,
        tool_call_count=len(response.tool_calls),
        tool_calls_valid=harness_result.tool_calls_valid,
        task_resolved=harness_result.task_resolved,
        regression_tests_passed=harness_result.regression_tests_passed,
        gpu_memory_bytes=response.gpu_memory_bytes,
        prompt_tokens_per_second=response.prompt_tokens_per_second,
        decode_tokens_per_second=response.decode_tokens_per_second,
        harness_evidence=harness_result.evidence,
    )
