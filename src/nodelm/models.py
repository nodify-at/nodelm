from __future__ import annotations

import hashlib
import json
import re
from enum import Enum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT RUN"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"


JsonFieldType = Literal["array", "boolean", "integer", "null", "number", "object", "string"]


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.dataset-source/v1"] = "nodelm.dataset-source/v1"
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    repository_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str | None = None
    dataset_license: str | None = None
    license_evidence_url: str | None = None
    snapshot_timestamp_utc: str | None = None
    observed_rows: int | None = Field(default=None, ge=0)
    evidence_urls: tuple[str, ...] = ()
    config_name: str | None = None
    splits: tuple[str, ...] = ()
    status: VerificationStatus
    notes: str | None = None
    trust_remote_code: Literal[False] = False

    @model_validator(mode="after")
    def require_verified_evidence(self) -> DatasetSource:
        if self.status is VerificationStatus.PASS:
            if not self.revision or re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None:
                raise ValueError("PASS requires a full 40-hex immutable revision")
            if (
                not self.dataset_license
                or not self.snapshot_timestamp_utc
                or self.observed_rows is None
                or not self.evidence_urls
            ):
                raise ValueError("PASS requires a license, timestamp, row count, and evidence")
        return self


class NormalizedSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.normalized-sample/v1"] = "nodelm.normalized-sample/v1"
    sample_id: str = Field(default="", pattern=r"^[0-9a-f]{64}$")
    source_dataset: str = Field(min_length=1)
    source_dataset_revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    repository: str = Field(min_length=1)
    repository_license: str = Field(min_length=1)
    base_commit: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    issue_or_pr_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    harness: str = Field(min_length=1)
    generating_model: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    resolved: bool
    trajectory: tuple[dict[str, Any], ...] = ()
    generated_patch: str | None = Field(default=None, min_length=1)
    patch_metadata: dict[str, Any]
    provenance_lineage: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def populate_and_verify_sample_id(self) -> NormalizedSample:
        identity = stable_model_id(
            {
                "source_dataset": self.source_dataset,
                "source_dataset_revision": self.source_dataset_revision,
                "repository": self.repository,
                "repository_license": self.repository_license,
                "base_commit": self.base_commit,
                "issue_or_pr_id": self.issue_or_pr_id,
                "language": self.language,
                "harness": self.harness,
                "generating_model": self.generating_model,
                "rollout_id": self.rollout_id,
                "resolved": self.resolved,
                "trajectory": self.trajectory,
                "generated_patch": self.generated_patch,
                "patch_metadata": self.patch_metadata,
                "provenance_lineage": self.provenance_lineage,
            }
        )
        if not self.sample_id:
            object.__setattr__(self, "sample_id", identity)
        elif self.sample_id != identity:
            raise ValueError("sample_id does not match normalized provenance identity")
        return self


class SolveContext(BaseModel):
    """Model-visible task context. Gold/reference patches are deliberately not representable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str
    base_commit: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    task: str


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: VerificationStatus
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DatasetAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.dataset-audit/v1"] = "nodelm.dataset-audit/v1"
    status: VerificationStatus
    source_name: str
    source_repository_id: str
    source_revision: str | None
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_bytes: int | None = Field(default=None, ge=0)
    input_scope: Literal["complete-snapshot", "partial-snapshot"]
    logical_rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    declared_row_count: int | None = Field(default=None, ge=0)
    matches_declared_row_count: bool | None
    schema_fields: tuple[str, ...]
    schema_field_types: dict[str, tuple[JsonFieldType, ...]]
    language_distribution: dict[str, int]
    resolved_distribution: dict[str, int]
    unique_repositories: int = Field(ge=0)
    duplicate_instance_id_count: int = Field(ge=0)
    duplicate_instance_ids: tuple[str, ...]
    duplicate_instance_id_sample_cap: int = Field(ge=0)
    duplicate_instance_ids_truncated: bool
    trajectory_lengths: dict[str, int | None]
    patch_sizes: dict[str, int | None]
    distribution_sample_cap: int = Field(gt=0)
    distribution_percentiles_approximate: bool
    license_distribution: dict[str, int]
    rejected_row_count: int = Field(ge=0)
    rejected_rows: tuple[dict[str, Any], ...]
    rejected_rows_truncated: bool
    rejection_ledger_artifact: str | None = None
    rejection_ledger_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rejection_ledger_rows: int | None = Field(default=None, ge=0)
    issues: tuple[str, ...]

    @model_validator(mode="after")
    def input_identity_is_complete(self) -> DatasetAuditReport:
        if (self.input_sha256 is None) != (self.input_bytes is None):
            raise ValueError("input_sha256 and input_bytes must be supplied together")
        if self.rejected_rows_truncated != (self.rejected_row_count > len(self.rejected_rows)):
            raise ValueError("rejected row truncation metadata is inconsistent")
        if self.duplicate_instance_ids_truncated != (
            self.duplicate_instance_id_count > len(self.duplicate_instance_ids)
        ):
            raise ValueError("duplicate instance ID truncation metadata is inconsistent")
        if len(self.duplicate_instance_ids) > self.duplicate_instance_id_sample_cap:
            raise ValueError("duplicate instance ID sample exceeds its declared cap")
        ledger_fields = (
            self.rejection_ledger_artifact,
            self.rejection_ledger_sha256,
            self.rejection_ledger_rows,
        )
        if any(value is None for value in ledger_fields) and any(
            value is not None for value in ledger_fields
        ):
            raise ValueError("rejection ledger identity must be supplied together")
        if (
            self.rejection_ledger_rows is not None
            and self.rejection_ledger_rows != self.rejected_row_count
        ):
            raise ValueError("rejection ledger row count must match rejected_row_count")
        return self


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_model_id(value: Any) -> str:
    canonical = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
