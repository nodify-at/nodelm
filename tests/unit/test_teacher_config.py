from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nodelm.models import VerificationStatus
from nodelm.teacher.config import (
    PrimaryTeacherConfig,
    TeacherConfigError,
    TeacherDecodingConfig,
)


def test_checked_in_primary_teacher_is_pinned_but_execution_is_not_run() -> None:
    config = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))

    assert config.model.repository_id == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert config.model.revision == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    assert config.model.metadata_status is VerificationStatus.PASS
    assert config.execution_status is VerificationStatus.NOT_RUN
    assert config.execution_evidence == ()
    assert config.decoding.reasoning_effort == "max"
    assert config.decoding.temperature == 1.0
    assert config.decoding.top_p == 0.95


def test_teacher_decoding_rejects_invalid_sampling_bounds() -> None:
    with pytest.raises(ValidationError):
        TeacherDecodingConfig(reasoning_effort="max", temperature=1.0, top_p=0.0)


@pytest.mark.parametrize("reasoning_effort", ("low", "high", "max"))
def test_teacher_decoding_accepts_supported_reasoning_efforts(reasoning_effort: str) -> None:
    config = TeacherDecodingConfig.model_validate(
        {"reasoning_effort": reasoning_effort, "temperature": 1.0, "top_p": 0.95}
    )

    assert config.reasoning_effort == reasoning_effort


def test_teacher_decoding_rejects_medium_reasoning_effort() -> None:
    with pytest.raises(ValidationError):
        TeacherDecodingConfig(reasoning_effort="medium", temperature=1.0, top_p=0.95)


def test_teacher_pass_execution_requires_evidence() -> None:
    checked_in = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))

    with pytest.raises(ValidationError, match="PASS teacher execution requires evidence"):
        PrimaryTeacherConfig.model_validate(
            {
                **checked_in.model_dump(mode="json"),
                "execution_status": VerificationStatus.PASS,
            }
        )


def test_teacher_pass_execution_requires_verified_model_metadata() -> None:
    checked_in = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))
    payload = checked_in.model_dump(mode="json")
    payload["model"]["metadata_status"] = VerificationStatus.NOT_RUN
    payload["execution_status"] = VerificationStatus.PASS
    payload["execution_evidence"] = ("artifacts/reports/teacher-execution.json",)

    with pytest.raises(
        ValidationError,
        match="PASS teacher execution requires PASS model metadata",
    ):
        PrimaryTeacherConfig.model_validate(payload)


def test_teacher_rejects_blank_execution_evidence() -> None:
    checked_in = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))

    with pytest.raises(ValidationError):
        PrimaryTeacherConfig.model_validate(
            {
                **checked_in.model_dump(mode="json"),
                "execution_status": VerificationStatus.PASS,
                "execution_evidence": ("   ",),
            }
        )


def test_teacher_strips_execution_evidence_whitespace() -> None:
    checked_in = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))
    config = PrimaryTeacherConfig.model_validate(
        {
            **checked_in.model_dump(mode="json"),
            "execution_status": VerificationStatus.PASS,
            "execution_evidence": ("  artifacts/reports/teacher-execution.json  ",),
        }
    )

    assert config.execution_evidence == ("artifacts/reports/teacher-execution.json",)


def test_teacher_not_run_rejects_execution_evidence() -> None:
    checked_in = PrimaryTeacherConfig.load(Path("configs/teacher/primary.yaml"))

    with pytest.raises(ValidationError, match="NOT RUN teacher execution cannot have evidence"):
        PrimaryTeacherConfig.model_validate(
            {
                **checked_in.model_dump(mode="json"),
                "execution_evidence": ("artifacts/reports/claimed-run.json",),
            }
        )


def test_teacher_loader_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "teacher.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TeacherConfigError, match="root must be a mapping"):
        PrimaryTeacherConfig.load(path)
