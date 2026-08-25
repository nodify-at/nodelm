from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

Sha256: TypeAlias = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha: TypeAlias = Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{40}$")]
NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
ByteCount: TypeAlias = Annotated[StrictInt, Field(ge=0)]
RowCount: TypeAlias = Annotated[StrictInt, Field(ge=0)]
PositiveRowCount: TypeAlias = Annotated[StrictInt, Field(gt=0)]
RejectionCode: TypeAlias = Annotated[
    StrictStr,
    Field(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
]
TerminalStatus: TypeAlias = Literal["PASS", "FAIL"]

TASK_PROVENANCE_SAFE_FIELDS = (
    "instance_id",
    "repository",
    "base_commit",
    "repository_license",
    "language",
    "source_dataset",
    "source_dataset_revision",
)


class _DerivedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManifestFileIdentity(_DerivedManifest):
    path: NonEmptyStr
    sha256: Sha256
    bytes: ByteCount

    @field_validator("path")
    @classmethod
    def require_normalized_relative_posix_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            value == "."
            or "\\" in value
            or "\x00" in value
            or parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("manifest file path must be a normalized relative POSIX path")
        return value


class _ArtifactManifest(_DerivedManifest):
    @field_validator("*", check_fields=False)
    @classmethod
    def reject_nul_in_strings(cls, value: object) -> object:
        if isinstance(value, str) and "\x00" in value:
            raise ValueError("manifest strings must not contain NUL bytes")
        return value


class _FileBoundManifest(_ArtifactManifest):
    file_patterns: tuple[NonEmptyStr, ...]
    files: tuple[ManifestFileIdentity, ...] = Field(min_length=1)

    @field_validator("file_patterns")
    @classmethod
    def require_normalized_file_patterns(
        cls,
        patterns: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(pattern != pattern.strip() or "\x00" in pattern for pattern in patterns):
            raise ValueError("file patterns must not contain surrounding whitespace")
        return patterns

    @field_validator("files")
    @classmethod
    def require_sorted_unique_files(
        cls,
        files: tuple[ManifestFileIdentity, ...],
    ) -> tuple[ManifestFileIdentity, ...]:
        paths = tuple(identity.path for identity in files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("manifest files must have unique, strictly sorted paths")
        return files


class _SnapshotMaterializationManifest(_FileBoundManifest):
    status: TerminalStatus
    source_name: NonEmptyStr
    source_repository_id: Annotated[
        StrictStr,
        Field(pattern=r"^[^/\s]+/[^/\s]+$"),
    ]
    source_revision: CommitSha
    registry_sha256: Sha256
    row_count: RowCount
    max_rows: Annotated[StrictInt, Field(gt=0)] | None
    output: NonEmptyStr
    output_sha256: Sha256
    output_bytes: ByteCount

    @model_validator(mode="after")
    def verify_status_and_row_bound(self) -> _SnapshotMaterializationManifest:
        expected_status = "PASS" if self.row_count else "FAIL"
        if self.status != expected_status:
            raise ValueError("materialization status must reflect whether rows were emitted")
        if self.max_rows is not None and self.row_count > self.max_rows:
            raise ValueError("materialization row_count cannot exceed max_rows")
        return self


class SnapshotMaterializationManifestV1(_SnapshotMaterializationManifest):
    schema_version: Literal["nodelm.snapshot-materialization/v1"]


class SnapshotMaterializationManifestV2(_SnapshotMaterializationManifest):
    schema_version: Literal["nodelm.snapshot-materialization/v2"]
    materialization_scope: Literal["canary", "complete-partition"]
    partition_contract_sha256: Sha256
    partition_contract_bytes: ByteCount
    transfer_receipt_sha256: Sha256
    transfer_receipt_bytes: ByteCount
    partition_name: Annotated[
        StrictStr,
        Field(pattern=r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$"),
    ]
    harness: Annotated[StrictStr, Field(pattern=r"^[a-z0-9._-]+$")]
    generating_model: NonEmptyStr
    upstream_source: Annotated[StrictStr, Field(pattern=r"^[a-z0-9._-]+$")]
    row_dataset_name: Annotated[StrictStr, Field(pattern=r"^[^/\s]+/[^/\s]+$")]
    task_source_name: Annotated[
        StrictStr,
        Field(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
    ]
    task_source_revision: CommitSha
    normalization_status: Literal["PASS", "BLOCKED"]

    @model_validator(mode="after")
    def verify_materialization_scope(self) -> SnapshotMaterializationManifestV2:
        expected_scope = "canary" if self.max_rows is not None else "complete-partition"
        if self.materialization_scope != expected_scope:
            raise ValueError("materialization_scope must match the max_rows bound")
        return self


class TaskProvenanceProjectionManifestV1(_FileBoundManifest):
    schema_version: Literal["nodelm.task-provenance-projection/v1"]
    status: TerminalStatus
    source_name: NonEmptyStr
    source_repository_id: Annotated[
        StrictStr,
        Field(pattern=r"^[^/\s]+/[^/\s]+$"),
    ]
    source_revision: CommitSha
    registry_sha256: Sha256
    registry_bytes: ByteCount
    transfer_receipt_sha256: Sha256
    transfer_receipt_bytes: ByteCount
    snapshot_sha256: Sha256
    projection_scope: Literal["filtered", "complete-snapshot"]
    safe_fields: tuple[NonEmptyStr, ...]
    admitted_count: RowCount
    rejected_count: RowCount
    rejection_counts_by_code: dict[RejectionCode, PositiveRowCount]
    output: NonEmptyStr
    output_sha256: Sha256
    output_bytes: ByteCount
    rejection_artifact: NonEmptyStr
    rejection_sha256: Sha256
    rejection_bytes: ByteCount

    @field_validator("safe_fields")
    @classmethod
    def require_exact_safe_field_contract(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if fields != TASK_PROVENANCE_SAFE_FIELDS:
            raise ValueError("safe_fields must equal the task provenance safe-field contract")
        return fields

    @model_validator(mode="after")
    def verify_projection_accounting(self) -> TaskProvenanceProjectionManifestV1:
        expected_status = "PASS" if self.admitted_count else "FAIL"
        if self.status != expected_status:
            raise ValueError("task projection status must reflect whether rows were admitted")
        expected_scope = "filtered" if self.file_patterns else "complete-snapshot"
        if self.projection_scope != expected_scope:
            raise ValueError("projection_scope must reflect whether file_patterns are present")
        if sum(self.rejection_counts_by_code.values()) != self.rejected_count:
            raise ValueError("rejection count sum must equal rejected_count")
        return self


class NormalizationManifestV2(_ArtifactManifest):
    schema_version: Literal["nodelm.normalization-manifest/v2"]
    status: TerminalStatus
    source_name: NonEmptyStr
    source_repository_id: Annotated[
        StrictStr,
        Field(pattern=r"^[^/\s]+/[^/\s]+$"),
    ]
    source_revision: CommitSha
    partition_name: Annotated[
        StrictStr,
        Field(pattern=r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$"),
    ]
    harness: Annotated[StrictStr, Field(pattern=r"^[a-z0-9._-]+$")]
    generating_model: NonEmptyStr
    upstream_source: Annotated[StrictStr, Field(pattern=r"^[a-z0-9._-]+$")]
    row_dataset_name: Annotated[StrictStr, Field(pattern=r"^[^/\s]+/[^/\s]+$")]
    input_sha256: Sha256
    input_bytes: ByteCount
    registry_sha256: Sha256
    materialization_manifest_sha256: Sha256
    materialization_manifest_bytes: ByteCount
    partition_contract_sha256: Sha256
    partition_contract_bytes: ByteCount
    transfer_receipt_sha256: Sha256
    transfer_receipt_bytes: ByteCount
    task_provenance_sha256: Sha256
    task_provenance_bytes: ByteCount
    task_provenance_manifest_sha256: Sha256
    task_provenance_manifest_bytes: ByteCount
    task_transfer_receipt_sha256: Sha256
    task_transfer_receipt_bytes: ByteCount
    task_source_name: Annotated[
        StrictStr,
        Field(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
    ]
    task_source_revision: CommitSha
    materialization_replay: Literal["PASS"]
    task_provenance_replay: Literal["PASS"]
    uniqueness_scope: Literal["canary", "complete-partition"]
    input_row_count: RowCount
    accepted_count: RowCount
    rejected_count: RowCount
    rejection_counts_by_code: dict[RejectionCode, PositiveRowCount]
    unique_rollout_key_count: RowCount
    duplicate_trace_row_count: RowCount
    conflicting_rollout_identity_count: RowCount
    conflicting_rollout_row_count: RowCount
    gold_exposure_audit: Literal["NOT RUN"]
    normalized_artifact: NonEmptyStr
    normalized_sha256: Sha256
    normalized_bytes: ByteCount
    rejection_artifact: NonEmptyStr
    rejection_sha256: Sha256
    rejection_bytes: ByteCount

    @model_validator(mode="after")
    def verify_normalization_accounting(self) -> NormalizationManifestV2:
        expected_status = "PASS" if self.accepted_count else "FAIL"
        if self.status != expected_status:
            raise ValueError("normalization status must reflect whether rows were accepted")
        if self.accepted_count + self.rejected_count != self.input_row_count:
            raise ValueError("accepted_count plus rejected_count must equal input_row_count")
        if sum(self.rejection_counts_by_code.values()) != self.rejected_count:
            raise ValueError("rejection count sum must equal rejected_count")
        if self.duplicate_trace_row_count != self.rejection_counts_by_code.get(
            "duplicate_trace_row", 0
        ):
            raise ValueError(
                "duplicate_trace_row_count must match its rejection_counts_by_code entry"
            )
        if self.conflicting_rollout_row_count != self.rejection_counts_by_code.get(
            "conflicting_rollout_identity", 0
        ):
            raise ValueError(
                "conflicting_rollout_row_count must match its rejection_counts_by_code entry"
            )
        if self.unique_rollout_key_count > self.input_row_count:
            raise ValueError("unique_rollout_key_count cannot exceed input_row_count")
        if (
            self.accepted_count + self.conflicting_rollout_identity_count
            > self.unique_rollout_key_count
        ):
            raise ValueError(
                "accepted and conflicting rollout identities cannot exceed unique rollout keys"
            )
        if self.conflicting_rollout_row_count < 2 * self.conflicting_rollout_identity_count:
            raise ValueError("each conflicting rollout identity must cover at least two rows")
        if self.duplicate_trace_row_count and not self.accepted_count:
            raise ValueError("duplicate trace rejections require an accepted rollout identity")
        return self
