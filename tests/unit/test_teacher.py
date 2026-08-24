from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from nodelm.models import SolveContext, VerificationStatus
from nodelm.teacher.rollout import (
    RolloutAttempt,
    run_teacher_rollouts,
    write_teacher_rollouts,
)


class FakeTeacher:
    model_id = "teacher/model"
    revision = "a" * 40
    decoding_config: ClassVar[dict[str, object]] = {"temperature": 0.0}

    def solve(self, context: SolveContext, *, seed: int) -> RolloutAttempt:
        assert "gold_patch" not in context.model_dump()
        return RolloutAttempt(
            concise_state="Need to inspect the failing module",
            tool_calls=({"name": "read_file", "arguments": {"path": "src/a.ts"}},),
            observations=(
                {
                    "tool_call_index": 0,
                    "status": VerificationStatus.PASS,
                    "output": "export const value = 1",
                },
            ),
            patch="diff --git a/src/a.ts b/src/a.ts",
            generation_status=VerificationStatus.PASS,
            status=VerificationStatus.UNVERIFIED,
            result_summary="patch generated; repository execution not run",
        )


def test_teacher_rollouts_are_versioned_and_keep_failed_or_successful_attempts() -> None:
    context = SolveContext(repository="acme/widget", base_commit="b" * 40, task="Fix it")

    records = run_teacher_rollouts(FakeTeacher(), context, seeds=(1, 2))

    assert len(records) == 2
    assert records[0].rollout_id != records[1].rollout_id
    assert all(record.model_revision == "a" * 40 for record in records)
    assert all(record.decoding_config == {"temperature": 0.0} for record in records)
    assert all("gold_patch" not in record.model_dump_json() for record in records)


def test_teacher_rollouts_persist_successful_and_failed_attempts_immutably(
    tmp_path: Path,
) -> None:
    context = SolveContext(repository="acme/widget", base_commit="b" * 40, task="Fix it")
    records = run_teacher_rollouts(FakeTeacher(), context, seeds=(1, 2))

    result = write_teacher_rollouts(records, output=tmp_path / "rollouts.jsonl")
    repeated = write_teacher_rollouts(records, output=tmp_path / "rollouts.jsonl")

    assert result.digest == repeated.digest
    assert result.created is True
    assert repeated.created is False


def test_teacher_adapter_cannot_forge_repository_execution_evidence() -> None:
    class ForgingTeacher(FakeTeacher):
        def solve(self, context: SolveContext, *, seed: int) -> RolloutAttempt:
            return RolloutAttempt(
                concise_state="claimed execution",
                tool_calls=(),
                observations=(),
                patch="diff --git a/src/a.ts b/src/a.ts",
                generation_status=VerificationStatus.PASS,
                status=VerificationStatus.PASS,
                result_summary="forged pass",
                execution_results=tuple(
                    {
                        "name": name,
                        "status": VerificationStatus.PASS,
                        "summary": "claimed",
                    }
                    for name in ("base_restore", "patch_apply", "tests")
                ),
                task_resolved=True,
                regression_tests_passed=True,
            )

    context = SolveContext(repository="acme/widget", base_commit="b" * 40, task="Fix it")

    with pytest.raises(ValueError, match="not trusted"):
        run_teacher_rollouts(ForgingTeacher(), context, seeds=(1,))


def test_teacher_attempt_cannot_report_pass_without_repository_evidence() -> None:
    with pytest.raises(ValidationError, match="repository execution evidence"):
        RolloutAttempt(
            concise_state="Generated a patch",
            tool_calls=(),
            observations=(),
            patch="diff --git a/a.ts b/a.ts",
            generation_status=VerificationStatus.PASS,
            status=VerificationStatus.PASS,
            result_summary="claimed success",
        )
