from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodelm.models import VerificationStatus


class DependencyInstallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal[False]
    ignore_scripts: Literal[True]


class HarnessConfig(BaseModel):
    """Capabilities the trusted-local harness can actually enforce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.harness-config/v1"]
    backend: Literal["trusted-local"]
    status: VerificationStatus
    timeout_seconds: float = Field(gt=0, le=3_600)
    max_output_bytes: int = Field(gt=0, le=16 * 1024 * 1024)
    network_enabled: bool
    network_isolation_enforced: bool
    dependency_install: DependencyInstallConfig
    allowed_tools: tuple[str, ...] = Field(min_length=1)
    security_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_unenforceable_claims(self) -> HarnessConfig:
        if not self.network_enabled:
            raise ValueError(
                "trusted-local cannot enforce network isolation; set network_enabled: true "
                "or use an external sandbox backend"
            )
        if self.network_isolation_enforced:
            raise ValueError("trusted-local cannot attest network isolation")
        if "node" not in self.allowed_tools:
            raise ValueError("trusted-local fixture verification requires node in allowed_tools")
        return self
