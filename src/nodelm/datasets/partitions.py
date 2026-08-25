from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nodelm.artifacts import content_digest
from nodelm.datasets.lineage import DatasetSnapshotTransferReceipt
from nodelm.models import VerificationStatus

AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION = {
    (
        "open-swe-traces",
        "ed95cef24df8d8bd79b4ceb0192cb420fde06521",
    ): "aec2ae095a926dda09a5fe3eefede7a59fbd494b24fffd503fff4cb366b389b5",
}


class PartitionContractError(ValueError):
    """A trace partition contract is malformed or does not bind the requested source."""


class TracePartition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$")
    harness: str = Field(pattern=r"^[a-z0-9._-]+$")
    generating_model: str = Field(min_length=1)
    upstream_source: str = Field(pattern=r"^[a-z0-9._-]+$")
    row_dataset_name: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    normalization_status: Literal[VerificationStatus.PASS, VerificationStatus.BLOCKED]
    task_source_name: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    task_source_revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    file_patterns: tuple[str, ...] = Field(min_length=1)
    notes: str | None = None

    @field_validator("generating_model")
    @classmethod
    def generating_model_is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("generating_model must not contain surrounding whitespace")
        return value

    @field_validator("file_patterns")
    @classmethod
    def patterns_are_contained(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(patterns)) != len(patterns):
            raise ValueError("partition file patterns must be unique")
        for pattern in patterns:
            candidate = Path(pattern)
            if (
                not pattern
                or candidate.is_absolute()
                or ".." in candidate.parts
                or not pattern.startswith("data/")
            ):
                raise ValueError(f"file pattern must be a contained relative glob: {pattern}")
        return patterns

    def model_post_init(self, __context: object) -> None:
        harness_slug, model_slug, upstream_slug = self.name.split("/")
        if (
            self.harness != harness_slug
            or self.generating_model != f"source-label:{model_slug}"
            or self.upstream_source != upstream_slug
        ):
            raise ValueError("partition labels must match the three source path components")
        partition_prefix = f"data/{self.name}/"
        if any(not pattern.startswith(partition_prefix) for pattern in self.file_patterns):
            raise ValueError("partition file patterns must stay inside their named leaf")
        task_fields_present = (
            self.task_source_name is not None,
            self.task_source_revision is not None,
        )
        if self.normalization_status is VerificationStatus.PASS and not all(task_fields_present):
            raise ValueError("PASS partitions require both pinned task source fields")
        if self.normalization_status is VerificationStatus.BLOCKED and any(task_fields_present):
            raise ValueError("BLOCKED partitions must omit both pinned task source fields")


class TracePartitionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.trace-partition-contract/v1"]
    source_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source_repository_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    source_revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    sealed_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transfer_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_file_count: int = Field(gt=0)
    partitions: tuple[TracePartition, ...] = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        counts = Counter(partition.name for partition in self.partitions)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate partition name: {', '.join(duplicates)}")

    @classmethod
    def load(cls, path: Path) -> TracePartitionContract:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise PartitionContractError(
                f"unable to read partition contract {path}: {error}"
            ) from error
        return cls.from_bytes(payload)

    @classmethod
    def from_bytes(cls, payload: bytes) -> TracePartitionContract:
        try:
            raw = yaml.safe_load(payload)
            return cls.model_validate(raw)
        except (UnicodeError, yaml.YAMLError, ValidationError, ValueError) as error:
            raise PartitionContractError(f"invalid trace partition contract: {error}") from error

    def by_name(self, name: str) -> TracePartition:
        for partition in self.partitions:
            if partition.name == name:
                return partition
        raise PartitionContractError(f"unknown trace partition: {name}")

    def require_source(self, name: str, revision: str) -> None:
        if self.source_name != name:
            raise PartitionContractError(
                f"partition contract source name mismatch: {self.source_name} != {name}"
            )
        if self.source_revision.casefold() != revision.casefold():
            raise PartitionContractError(
                "partition contract source revision does not match the dataset registry"
            )

    def require_authorized_digest(self, digest: str) -> None:
        expected = AUTHORIZED_CONTRACT_SHA256_BY_SOURCE_REVISION.get(
            (self.source_name, self.source_revision.casefold())
        )
        if expected is None:
            raise PartitionContractError(
                "no authorized partition contract for this source revision"
            )
        if digest != expected:
            raise PartitionContractError(
                "partition contract digest is not authorized for this source revision"
            )

    def bind_transfer_receipt(self, payload: bytes) -> DatasetSnapshotTransferReceipt:
        """Validate complete receipt coverage and return its typed, immutable identity."""

        if content_digest(payload) != self.transfer_receipt_sha256:
            raise PartitionContractError("transfer receipt digest does not match the contract")
        try:
            receipt = DatasetSnapshotTransferReceipt.model_validate_json(payload)
        except ValidationError as error:
            raise PartitionContractError(f"invalid transfer receipt: {error}") from error
        if receipt.snapshot_scope != "complete":
            raise PartitionContractError("partition contract requires a complete transfer receipt")
        if (
            receipt.source.name != self.source_name
            or receipt.source.repository_id != self.source_repository_id
            or receipt.source.revision is None
            or receipt.source.revision.casefold() != self.source_revision.casefold()
        ):
            raise PartitionContractError("transfer receipt source does not match the contract")
        if receipt.registry_sha256 != self.sealed_registry_sha256:
            raise PartitionContractError("transfer receipt registry does not match the contract")
        if (
            receipt.snapshot.snapshot_sha256 != self.snapshot_sha256
            or len(receipt.snapshot.files) != self.snapshot_file_count
        ):
            raise PartitionContractError("transfer receipt snapshot does not match the contract")

        matched_partitions: Counter[str] = Counter()
        for identity in receipt.snapshot.files:
            matches = [
                partition.name
                for partition in self.partitions
                if any(
                    PurePosixPath(identity.path).match(pattern)
                    for pattern in partition.file_patterns
                )
            ]
            if len(matches) != 1:
                raise PartitionContractError(
                    f"receipt file must belong to exactly one partition: {identity.path}"
                )
            matched_partitions[matches[0]] += 1
        empty_partitions = sorted(
            partition.name
            for partition in self.partitions
            if matched_partitions[partition.name] == 0
        )
        if empty_partitions:
            raise PartitionContractError(
                f"receipt has no files for partition: {', '.join(empty_partitions)}"
            )
        return receipt
