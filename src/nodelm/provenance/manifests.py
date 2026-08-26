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
PartitionName: TypeAlias = Annotated[
    StrictStr,
    Field(pattern=r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$"),
]
ResolutionLanguage: TypeAlias = Literal["TypeScript", "JavaScript"]

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


class ResolutionPartitionInput(_DerivedManifest):
    partition_name: PartitionName
    row_count: RowCount
    files: tuple[ManifestFileIdentity, ...] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def require_sorted_partition_files(
        cls,
        files: tuple[ManifestFileIdentity, ...],
    ) -> tuple[ManifestFileIdentity, ...]:
        paths = tuple(identity.path for identity in files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("resolution partition files must have unique, sorted paths")
        return files

    @model_validator(mode="after")
    def require_files_inside_partition(self) -> ResolutionPartitionInput:
        partition_prefix = f"data/{self.partition_name}/"
        if any(not identity.path.startswith(partition_prefix) for identity in self.files):
            raise ValueError("resolution partition files must stay inside their partition")
        return self


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


class ResolutionRecoveryManifestV1(_ArtifactManifest):
    schema_version: Literal["nodelm.resolution-recovery/v1"]
    derivation_status: Literal["PASS"]
    admission_status: Literal["BLOCKED"]
    admission_blocker: Literal["harness_canary_pending"]
    source_name: NonEmptyStr
    source_repository_id: Annotated[
        StrictStr,
        Field(pattern=r"^[^/\s]+/[^/\s]+$"),
    ]
    source_revision: CommitSha
    task_source_name: Annotated[
        StrictStr,
        Field(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
    ]
    task_source_revision: CommitSha
    partition_contract_sha256: Sha256
    partition_contract_bytes: ByteCount
    transfer_receipt_sha256: Sha256
    transfer_receipt_bytes: ByteCount
    labeled_partitions: tuple[ResolutionPartitionInput, ...] = Field(min_length=1)
    target_partitions: tuple[ResolutionPartitionInput, ...] = Field(min_length=1)
    language_filter: tuple[ResolutionLanguage, ...] = Field(min_length=1)
    candidate_artifact: NonEmptyStr
    candidate_sha256: Sha256
    candidate_bytes: ByteCount
    queue_artifact: NonEmptyStr
    queue_sha256: Sha256
    queue_bytes: ByteCount
    target_row_count: RowCount
    ineligible_row_count: RowCount
    already_known_row_count: RowCount
    candidate_row_count: RowCount
    candidate_unique_count: RowCount
    candidate_resolved_count: RowCount
    candidate_unresolved_count: RowCount
    queued_fanout_row_count: RowCount
    queue_unique_count: RowCount
    conflict_count: RowCount

    @field_validator("labeled_partitions", "target_partitions")
    @classmethod
    def require_sorted_unique_partitions(
        cls,
        partitions: tuple[ResolutionPartitionInput, ...],
    ) -> tuple[ResolutionPartitionInput, ...]:
        names = tuple(partition.partition_name for partition in partitions)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("resolution partitions must have unique, sorted names")
        return partitions

    @field_validator("language_filter")
    @classmethod
    def require_sorted_unique_languages(cls, languages: tuple[str, ...]) -> tuple[str, ...]:
        if any(language != language.strip() for language in languages):
            raise ValueError("language_filter entries must not contain surrounding whitespace")
        if languages != tuple(sorted(languages)) or len(languages) != len(set(languages)):
            raise ValueError("language_filter entries must be unique and sorted")
        return languages

    @model_validator(mode="after")
    def verify_recovery_contract(self) -> ResolutionRecoveryManifestV1:
        labeled_names = {partition.partition_name for partition in self.labeled_partitions}
        target_names = {partition.partition_name for partition in self.target_partitions}
        if labeled_names & target_names:
            raise ValueError("labeled and target resolution partitions must be disjoint")

        all_files = [
            identity.path
            for partition in (*self.labeled_partitions, *self.target_partitions)
            for identity in partition.files
        ]
        if len(all_files) != len(set(all_files)):
            raise ValueError("resolution partition file identities must be globally unique")

        target_partition_rows = sum(partition.row_count for partition in self.target_partitions)
        if target_partition_rows != self.target_row_count:
            raise ValueError("target partition rows must equal target_row_count")

        accounted_rows = (
            self.ineligible_row_count
            + self.already_known_row_count
            + self.candidate_row_count
            + self.queued_fanout_row_count
        )
        if accounted_rows != self.target_row_count:
            raise ValueError("target recovery accounting must equal target_row_count exactly")

        if (
            self.candidate_resolved_count + self.candidate_unresolved_count
            != self.candidate_row_count
        ):
            raise ValueError("candidate outcome counts must equal candidate_row_count")
        if (self.candidate_unique_count == 0) != (self.candidate_row_count == 0):
            raise ValueError(
                "candidate unique count must be zero exactly when candidate rows are zero"
            )
        if (self.queue_unique_count == 0) != (self.queued_fanout_row_count == 0):
            raise ValueError(
                "queue unique count must be zero exactly when queued fanout rows are zero"
            )
        if self.candidate_unique_count > self.candidate_row_count:
            raise ValueError("candidate_unique_count cannot exceed candidate_row_count")
        if self.queue_unique_count > self.queued_fanout_row_count:
            raise ValueError("queue_unique_count cannot exceed queued_fanout_row_count")
        if (self.candidate_bytes == 0) != (self.candidate_row_count == 0):
            raise ValueError("candidate bytes must be zero exactly when candidate rows are zero")
        if (self.queue_bytes == 0) != (self.queue_unique_count == 0):
            raise ValueError("queue bytes must be zero exactly when queue rows are zero")
        if self.conflict_count:
            raise ValueError("conflict_count must be zero before a PASS manifest is published")
        if self.candidate_artifact == self.queue_artifact:
            raise ValueError("candidate and queue artifacts must be distinct")
        return self


class ResolutionCanaryWorksetManifestV1(_ArtifactManifest):
    schema_version: Literal["nodelm.resolution-canary-workset/v1"]
    materialization_status: Literal["PASS"]
    execution_status: Literal["NOT RUN"]
    admission_status: Literal["BLOCKED"]
    admission_blocker: Literal["canary_execution_pending"]
    recovery_manifest_sha256: Sha256
    recovery_manifest_bytes: ByteCount
    candidate_sha256: Sha256
    candidate_bytes: ByteCount
    queue_sha256: Sha256
    queue_bytes: ByteCount
    trace_source_name: NonEmptyStr
    trace_source_revision: CommitSha
    trace_transfer_receipt_sha256: Sha256
    task_source_name: NonEmptyStr
    task_source_revision: CommitSha
    task_transfer_receipt_sha256: Sha256
    selection_algorithm: Literal["nodelm.resolution-canary-cover/v1"]
    minimum_per_kind: Annotated[StrictInt, Field(gt=0)]
    maximum_per_kind: Annotated[StrictInt, Field(gt=0)]
    evaluator_repository_id: Annotated[
        StrictStr,
        Field(pattern=r"^[^/\s]+/[^/\s]+$"),
    ]
    evaluator_revision: CommitSha
    workset_artifact: NonEmptyStr
    workset_sha256: Sha256
    workset_bytes: ByteCount
    case_count: PositiveRowCount
    transfer_control_count: PositiveRowCount
    evaluation_request_count: PositiveRowCount
    languages: tuple[ResolutionLanguage, ...] = Field(min_length=1)
    target_partitions: tuple[PartitionName, ...] = Field(min_length=1)

    @field_validator("languages", "target_partitions")
    @classmethod
    def require_sorted_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("resolution canary coverage values must be unique and sorted")
        return values

    @model_validator(mode="after")
    def verify_workset_accounting(self) -> ResolutionCanaryWorksetManifestV1:
        if self.minimum_per_kind > self.maximum_per_kind:
            raise ValueError("minimum_per_kind must not exceed maximum_per_kind")
        if self.transfer_control_count + self.evaluation_request_count != self.case_count:
            raise ValueError("resolution canary case counts must equal case_count")
        if (
            self.transfer_control_count < self.minimum_per_kind
            or self.evaluation_request_count < self.minimum_per_kind
            or self.transfer_control_count > self.maximum_per_kind
            or self.evaluation_request_count > self.maximum_per_kind
        ):
            raise ValueError("resolution canary case counts violate selection bounds")
        return self


class ResolutionCanaryExecutionManifestV1(_ArtifactManifest):
    schema_version: Literal["nodelm.resolution-canary-execution/v1"]
    execution_status: TerminalStatus
    admission_status: Literal["PASS", "BLOCKED"]
    admission_blocker: Literal["canary_case_failed"] | None
    code_commit: CommitSha
    recovery_manifest_sha256: Sha256
    workset_manifest_sha256: Sha256
    workset_manifest_bytes: ByteCount
    workset_sha256: Sha256
    workset_bytes: ByteCount
    image_lock_sha256: Sha256
    image_lock_bytes: ByteCount
    evaluator_repository_id: Literal["SWE-rebench/SWE-rebench-V2"]
    evaluator_revision: Literal["c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"]
    evaluator_log_parsers_sha256: Sha256
    evaluator_script_sha256: Sha256
    evaluator_constants_sha256: Sha256
    sandbox_backend: Literal["rootless-podman", "seccomp-chroot"]
    sandbox_network: Literal["none"]
    sandbox_cpus_per_attempt: Literal[2]
    sandbox_memory_per_attempt: Literal["4g"]
    results_artifact: NonEmptyStr
    results_sha256: Sha256
    results_bytes: ByteCount
    case_count: PositiveRowCount
    passed_case_count: RowCount
    failed_case_count: RowCount
    transfer_control_count: PositiveRowCount
    transfer_label_agreement_count: RowCount
    evaluation_request_count: PositiveRowCount
    evaluation_resolved_count: RowCount
    evaluation_unresolved_count: RowCount
    image_count: PositiveRowCount
    failure_counts_by_reason: dict[RejectionCode, PositiveRowCount]

    @model_validator(mode="after")
    def verify_execution_accounting(self) -> ResolutionCanaryExecutionManifestV1:
        if self.passed_case_count + self.failed_case_count != self.case_count:
            raise ValueError("canary result counts must equal case_count")
        if self.transfer_control_count + self.evaluation_request_count != self.case_count:
            raise ValueError("canary kind counts must equal case_count")
        if self.transfer_label_agreement_count > self.transfer_control_count:
            raise ValueError("label agreements cannot exceed transfer controls")
        if (
            self.evaluation_resolved_count + self.evaluation_unresolved_count
            > self.evaluation_request_count
        ):
            raise ValueError("evaluation outcomes cannot exceed evaluation requests")
        if sum(self.failure_counts_by_reason.values()) != self.failed_case_count:
            raise ValueError("failure reasons must account for every failed case")
        expected_execution = "PASS" if self.failed_case_count == 0 else "FAIL"
        if self.execution_status != expected_execution:
            raise ValueError("execution status must reflect failed_case_count")
        expected_admission = "PASS" if self.execution_status == "PASS" else "BLOCKED"
        if self.admission_status != expected_admission:
            raise ValueError("admission status must reflect execution status")
        expected_blocker = None if self.admission_status == "PASS" else "canary_case_failed"
        if self.admission_blocker != expected_blocker:
            raise ValueError("admission blocker must reflect admission status")
        if self.execution_status == "PASS" and (
            self.transfer_label_agreement_count != self.transfer_control_count
            or self.evaluation_resolved_count + self.evaluation_unresolved_count
            != self.evaluation_request_count
        ):
            raise ValueError("PASS canary must resolve every case and agree with every control")
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
