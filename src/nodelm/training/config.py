from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodelm.models import VerificationStatus


class TrainingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_id: str | None = Field(default=None, pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    license: str | None = Field(default=None, min_length=1)


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    save: bool
    reload: bool
    resume: bool


class TransformersRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["transformers-peft"]
    device: Literal["cpu", "cuda"]
    padding_policy: Literal["require-existing", "reuse-eos", "add-token"]
    added_padding_token: str | None = None
    max_length: int = Field(ge=8)
    max_new_tokens: int = Field(ge=1)
    use_lora: bool
    target_modules: tuple[str, ...] = ()
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def require_explicit_architecture_choices(self) -> TransformersRuntimeConfig:
        if self.use_lora and not self.target_modules:
            raise ValueError("LoRA target_modules must be explicitly selected for the model")
        if self.padding_policy == "add-token" and not self.added_padding_token:
            raise ValueError("add-token padding policy requires added_padding_token")
        return self


class TrainingSmokeConfig(BaseModel):
    """Strict contract for the checked-in tiny LoRA/PEFT smoke configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.training-config/v1"]
    status: VerificationStatus
    purpose: str = Field(min_length=1)
    model: TrainingModelConfig
    runtime: TransformersRuntimeConfig | None
    precision: Literal["float32", "float16", "bfloat16"] | None
    seed: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    checkpoint: CheckpointConfig
    inference_after_reload: bool
    reason: str | None = None

    @model_validator(mode="after")
    def pass_requires_a_pinned_runnable_configuration(self) -> TrainingSmokeConfig:
        if self.status is VerificationStatus.PASS and (
            self.model.repository_id is None
            or self.model.revision is None
            or self.model.license is None
            or self.runtime is None
            or self.precision is None
            or not self.checkpoint.save
            or not self.checkpoint.reload
            or not self.checkpoint.resume
            or not self.inference_after_reload
        ):
            raise ValueError("PASS requires a pinned model and complete lifecycle checks")
        return self

    @property
    def is_runnable(self) -> bool:
        return (
            self.status in {VerificationStatus.PASS, VerificationStatus.UNVERIFIED}
            and self.model.repository_id is not None
            and self.model.revision is not None
            and self.model.license is not None
            and self.runtime is not None
            and self.precision is not None
            and self.checkpoint.save
            and self.checkpoint.reload
            and self.checkpoint.resume
            and self.inference_after_reload
        )


def parse_training_smoke_config(data: bytes) -> TrainingSmokeConfig:
    raw: Any = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training smoke config must be a YAML mapping")
    return TrainingSmokeConfig.model_validate(raw)


def load_training_smoke_config(path: Path) -> TrainingSmokeConfig:
    return parse_training_smoke_config(path.read_bytes())


def training_config_identity(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)
