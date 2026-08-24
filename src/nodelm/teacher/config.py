from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nodelm.model_metadata import EvidenceReference, HuggingFaceModelMetadata
from nodelm.models import VerificationStatus


class TeacherConfigError(ValueError):
    """The primary teacher configuration is malformed or internally inconsistent."""


class TeacherDecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: Literal["low", "high", "max"]
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)


class PrimaryTeacherConfig(BaseModel):
    """Pinned teacher metadata and planned decoding, independent of execution state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.primary-teacher/v1"]
    model: HuggingFaceModelMetadata
    decoding: TeacherDecodingConfig
    execution_status: VerificationStatus
    execution_evidence: tuple[EvidenceReference, ...] = ()
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_execution_state(self) -> PrimaryTeacherConfig:
        if self.execution_status is VerificationStatus.NOT_RUN and self.execution_evidence:
            raise ValueError("NOT RUN teacher execution cannot have evidence")
        if self.execution_status is VerificationStatus.PASS:
            if self.model.metadata_status is not VerificationStatus.PASS:
                raise ValueError("PASS teacher execution requires PASS model metadata")
            if not self.execution_evidence:
                raise ValueError("PASS teacher execution requires evidence")
        return self

    @classmethod
    def load(cls, path: Path) -> PrimaryTeacherConfig:
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise TeacherConfigError(
                f"unable to read primary teacher configuration {path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise TeacherConfigError("primary teacher configuration root must be a mapping")
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise TeacherConfigError(f"invalid primary teacher configuration: {error}") from error
