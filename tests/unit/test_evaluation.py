from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nodelm.evaluation.adapters import (
    CandidateEvaluation,
    CandidateModel,
    FixturePatchHarness,
    HarnessEvaluation,
    ModelResponse,
    evaluate_adapter,
)
from nodelm.evaluation.fixture import MODEL_TASK_EXACT_SOURCE_TRANSITIONS
from nodelm.models import SolveContext, VerificationStatus


class FakeAdapter:
    def generate(self, context: SolveContext) -> ModelResponse:
        return ModelResponse(
            text="Use the repository tools",
            tool_calls=({"name": "search", "arguments": {"pattern": "TODO"}},),
            prompt_tokens=10,
            completion_tokens=5,
            wall_clock_seconds=0.1,
        )


def test_candidate_evaluation_records_measured_fields_without_inventing_benchmark_result() -> None:
    candidate = CandidateModel(
        name="fixture",
        repository_id="owner/model",
        revision="a" * 40,
        license="Apache-2.0",
        status=VerificationStatus.PASS,
    )
    context = SolveContext(repository="acme/widget", base_commit="b" * 40, task="Inspect")

    result = evaluate_adapter(
        candidate,
        FakeAdapter(),
        context,
        precision="bfloat16",
        backend="transformers",
    )

    assert result.generation_status is VerificationStatus.PASS
    assert result.status is VerificationStatus.UNVERIFIED
    assert result.total_tokens == 15
    assert result.model_dump()["total_tokens"] == 15
    assert result.task_resolved is None
    assert result.regression_tests_passed is None
    assert result.precision == "bfloat16"
    assert result.backend == "transformers"


def test_candidate_adapter_can_return_real_common_harness_evidence() -> None:
    class PassingHarness:
        def evaluate(self, response: ModelResponse, context: SolveContext) -> HarnessEvaluation:
            assert response.text
            assert context.repository == "acme/widget"
            return HarnessEvaluation(
                status=VerificationStatus.PASS,
                tool_calls_valid=True,
                task_resolved=True,
                regression_tests_passed=True,
                evidence={"fixture": "passed"},
            )

    candidate = CandidateModel(
        name="fixture",
        repository_id="owner/model",
        revision="a" * 40,
        license="Apache-2.0",
        status=VerificationStatus.PASS,
    )
    context = SolveContext(repository="acme/widget", base_commit="b" * 40, task="Inspect")

    result = evaluate_adapter(
        candidate,
        FakeAdapter(),
        context,
        precision="float32",
        backend="fixture-backend",
        harness=PassingHarness(),
    )

    assert result.status is VerificationStatus.PASS
    assert result.tool_calls_valid is True
    assert result.task_resolved is True
    assert result.regression_tests_passed is True
    assert result.harness_evidence == {"fixture": "passed"}


def test_pass_candidate_rejects_symbolic_revision() -> None:
    with pytest.raises(ValidationError, match="40-hex"):
        CandidateModel(
            name="fixture",
            repository_id="owner/model",
            revision="main",
            license="Apache-2.0",
            status=VerificationStatus.PASS,
        )


def test_pass_evaluation_requires_harness_outcomes() -> None:
    candidate = CandidateModel(
        name="fixture",
        repository_id="owner/model",
        revision="a" * 40,
        license="Apache-2.0",
        status=VerificationStatus.PASS,
    )

    with pytest.raises(ValidationError, match="valid-tool"):
        CandidateEvaluation(
            candidate=candidate,
            precision="float32",
            backend="fixture",
            generation_status=VerificationStatus.PASS,
            status=VerificationStatus.PASS,
            prompt_tokens=1,
            completion_tokens=1,
            wall_clock_seconds=0.1,
            tool_call_count=0,
        )


def test_fixture_harness_rejects_a_context_other_than_its_exact_binding() -> None:
    expected = SolveContext(
        repository="acme/widget",
        base_commit="b" * 40,
        task="Repair multiply",
    )
    actual = SolveContext(
        repository="acme/other",
        base_commit="c" * 40,
        task="Repair multiply",
    )
    harness = FixturePatchHarness(
        Path("tests/fixtures/model-task"),
        None,
        expected_context=expected,
        exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
    )
    response = ModelResponse(
        text="not executed",
        prompt_tokens=1,
        completion_tokens=1,
        wall_clock_seconds=0.1,
    )

    result = harness.evaluate(response, actual)

    assert result.status is VerificationStatus.FAIL
    assert result.task_resolved is False
    assert result.evidence["expected_context"]["repository"] == "acme/widget"
