from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nodelm.model_metadata import EvidenceReference, HuggingFaceModelMetadata
from nodelm.models import VerificationStatus


class CandidateRegistryError(ValueError):
    """The candidate registry is malformed or internally inconsistent."""


class CandidateModel(HuggingFaceModelMetadata):
    metadata_kind: ClassVar[str] = "candidate"

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")

    @model_validator(mode="after")
    def pass_requires_context_limit(self) -> CandidateModel:
        if self.metadata_status is VerificationStatus.PASS and self.context_limit is None:
            raise ValueError("PASS candidate metadata requires context_limit")
        return self


class CandidateRegistry(BaseModel):
    """Strict candidate metadata plus honest execution and bake-off state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.candidate-registry/v1"]
    metadata_status: VerificationStatus
    execution_status: VerificationStatus
    bakeoff_status: VerificationStatus
    candidates: tuple[CandidateModel, ...] = Field(min_length=1)
    selected_candidate: str | None = None
    execution_evidence: tuple[EvidenceReference, ...] = ()
    bakeoff_evidence: tuple[EvidenceReference, ...] = ()
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry_state(self) -> CandidateRegistry:
        duplicate_names = sorted(
            name
            for name, count in Counter(item.name for item in self.candidates).items()
            if count > 1
        )
        if duplicate_names:
            raise ValueError(f"duplicate candidate name: {', '.join(duplicate_names)}")

        duplicate_repositories = sorted(
            repository
            for repository, count in Counter(item.repository_id for item in self.candidates).items()
            if count > 1
        )
        if duplicate_repositories:
            raise ValueError(
                "duplicate candidate repository_id: " + ", ".join(duplicate_repositories)
            )

        if self.metadata_status is VerificationStatus.PASS and any(
            item.metadata_status is not VerificationStatus.PASS for item in self.candidates
        ):
            raise ValueError("PASS registry metadata requires every candidate metadata to PASS")

        if self.execution_status is VerificationStatus.NOT_RUN and self.execution_evidence:
            raise ValueError("NOT RUN candidate execution cannot have evidence")
        if self.execution_status is VerificationStatus.PASS:
            if self.metadata_status is not VerificationStatus.PASS or any(
                item.metadata_status is not VerificationStatus.PASS for item in self.candidates
            ):
                raise ValueError(
                    "PASS candidate execution requires PASS registry and candidate metadata"
                )
            if not self.execution_evidence:
                raise ValueError("PASS candidate execution requires evidence")

        if self.bakeoff_status is VerificationStatus.NOT_RUN and self.bakeoff_evidence:
            raise ValueError("NOT RUN bake-off cannot have evidence")
        if self.bakeoff_status is VerificationStatus.PASS and (
            self.execution_status is not VerificationStatus.PASS
            or not self.bakeoff_evidence
            or self.selected_candidate is None
        ):
            raise ValueError(
                "PASS bake-off requires PASS candidate execution, evidence, and a selection"
            )
        candidate_names = {item.name for item in self.candidates}
        if self.selected_candidate is not None and self.selected_candidate not in candidate_names:
            raise ValueError("selected_candidate must name a registered candidate")
        if (
            self.bakeoff_status is not VerificationStatus.PASS
            and self.selected_candidate is not None
        ):
            raise ValueError("a candidate selection requires a PASS bake-off")
        return self

    @classmethod
    def load(cls, path: Path) -> CandidateRegistry:
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise CandidateRegistryError(
                f"unable to read candidate registry {path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise CandidateRegistryError("candidate registry root must be a mapping")
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise CandidateRegistryError(f"invalid candidate registry: {error}") from error
