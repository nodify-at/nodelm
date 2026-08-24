from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Annotated, ClassVar, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nodelm.models import VerificationStatus

EvidenceReference: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class HuggingFaceLoaderClass(StrEnum):
    AUTO_MODEL_FOR_CAUSAL_LM = "AutoModelForCausalLM"
    AUTO_MODEL_FOR_MULTIMODAL_LM = "AutoModelForMultimodalLM"


class HuggingFaceModelMetadata(BaseModel):
    """Immutable, source-backed Hugging Face model metadata.

    PASS verifies only this metadata contract. It does not imply that the model has
    been downloaded, loaded, executed, evaluated, or trained by NodeLM.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    metadata_kind: ClassVar[str] = "model"

    repository_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str | None = None
    license: str | None = Field(default=None, min_length=1)
    parameter_count: int | None = Field(default=None, gt=0)
    active_parameter_count: int | None = Field(default=None, gt=0)
    architecture: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    model_type: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    loader_class: HuggingFaceLoaderClass | None = None
    context_limit: int | None = Field(default=None, gt=0)
    metadata_verified_on: date | None = None
    evidence_urls: tuple[str, ...] = ()
    metadata_status: VerificationStatus
    notes: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_metadata_evidence(self) -> HuggingFaceModelMetadata:
        if (
            self.active_parameter_count is not None
            and self.parameter_count is not None
            and self.active_parameter_count > self.parameter_count
        ):
            raise ValueError("active parameter count cannot exceed total parameter count")

        if self.metadata_status is not VerificationStatus.PASS:
            return self

        missing: list[str] = []
        if not self.revision or re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None:
            missing.append("a full 40-hex immutable revision")
        required = {
            "license": self.license,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "loader_class": self.loader_class,
            "metadata_verified_on": self.metadata_verified_on,
            "evidence_urls": self.evidence_urls,
        }
        missing.extend(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"PASS {self.metadata_kind} metadata requires " + ", ".join(missing))
        revision = self.revision
        if revision is None:  # Kept explicit for static narrowing after aggregate validation.
            raise ValueError(f"PASS {self.metadata_kind} metadata requires an immutable revision")

        repository_url = f"https://huggingface.co/{self.repository_id}"
        if any(
            not (url == repository_url or url.startswith(repository_url + "/"))
            for url in self.evidence_urls
        ):
            raise ValueError(
                f"PASS {self.metadata_kind} evidence must use the official Hugging Face repository"
            )
        if not any(revision in url for url in self.evidence_urls):
            raise ValueError(
                f"PASS {self.metadata_kind} evidence must reference the immutable revision"
            )
        return self
