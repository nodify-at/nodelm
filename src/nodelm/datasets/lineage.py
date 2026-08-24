from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nodelm.artifacts import canonical_json_bytes, content_digest
from nodelm.datasets.materialize import discover_snapshot_files
from nodelm.models import DatasetAuditReport, DatasetSource, VerificationStatus

SNAPSHOT_IDENTITY_SCHEMA: Final = "nodelm.dataset-snapshot-identity/v1"
LINEAGE_MANIFEST_SCHEMA: Final = "nodelm.dataset-lineage/v1"


class SnapshotFileIdentity(BaseModel):
    """Content identity for one file, independent of the snapshot's absolute root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)

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
            raise ValueError("snapshot file path must be a normalized relative POSIX path")
        return value


def _aggregate_snapshot_digest(
    files: tuple[SnapshotFileIdentity, ...],
    snapshot_bytes: int,
) -> str:
    payload = {
        "schema_version": SNAPSHOT_IDENTITY_SCHEMA,
        "snapshot_bytes": snapshot_bytes,
        "files": [identity.model_dump(mode="json") for identity in files],
    }
    domain = f"{SNAPSHOT_IDENTITY_SCHEMA}\0".encode()
    return content_digest(domain + canonical_json_bytes(payload))


class DatasetSnapshotIdentity(BaseModel):
    """Self-validating aggregate identity for every supported file in a snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.dataset-snapshot-identity/v1"] = SNAPSHOT_IDENTITY_SCHEMA
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_bytes: int = Field(ge=0)
    files: tuple[SnapshotFileIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_aggregate_identity(self) -> DatasetSnapshotIdentity:
        paths = tuple(identity.path for identity in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("snapshot files must have unique, strictly sorted paths")
        total_bytes = sum(identity.bytes for identity in self.files)
        if self.snapshot_bytes != total_bytes:
            raise ValueError("snapshot_bytes must equal the sum of file bytes")
        expected_digest = _aggregate_snapshot_digest(self.files, self.snapshot_bytes)
        if self.snapshot_sha256 != expected_digest:
            raise ValueError("snapshot_sha256 does not match the canonical snapshot identity")
        return self


class DeferredAuditChecks(BaseModel):
    """Phase 0 checks that need later tokenizer/benchmark policy decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenized_trajectory_lengths: Literal[VerificationStatus.NOT_RUN] = VerificationStatus.NOT_RUN
    exact_and_near_patch_duplicates: Literal[VerificationStatus.NOT_RUN] = (
        VerificationStatus.NOT_RUN
    )
    public_evaluation_overlap: Literal[VerificationStatus.NOT_RUN] = VerificationStatus.NOT_RUN


class DatasetLineageManifest(BaseModel):
    """Completion marker binding a pinned source to immutable snapshot audit evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.dataset-lineage/v1"] = LINEAGE_MANIFEST_SCHEMA
    status: VerificationStatus
    source: DatasetSource
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_bytes: int = Field(ge=0)
    snapshot: DatasetSnapshotIdentity
    audit_artifact: str = Field(min_length=1)
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_row_count: int = Field(ge=0)
    audit_logical_rows_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rejection_ledger_artifact: str = Field(min_length=1)
    rejection_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rejection_ledger_rows: int = Field(ge=0)
    deferred_checks: DeferredAuditChecks = Field(default_factory=DeferredAuditChecks)

    @model_validator(mode="after")
    def verify_terminal_source_contract(self) -> DatasetLineageManifest:
        if self.source.status is not VerificationStatus.PASS or self.source.revision is None:
            raise ValueError("lineage requires an exact registry-PASS pinned source")
        if self.status not in {VerificationStatus.PASS, VerificationStatus.FAIL}:
            raise ValueError("lineage status must be terminal PASS or FAIL")
        if self.rejection_ledger_rows > self.audit_row_count:
            raise ValueError("rejection ledger rows cannot exceed audited rows")
        if (
            self.status is VerificationStatus.PASS
            and self.audit_row_count != self.source.observed_rows
        ):
            raise ValueError("PASS lineage requires an exact declared row count")
        return self


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def capture_snapshot_identity(snapshot: Path) -> DatasetSnapshotIdentity:
    """Capture every supported snapshot file without retaining an absolute root."""

    resolved_snapshot = snapshot.resolve()
    files = discover_snapshot_files(resolved_snapshot)
    relative_root = resolved_snapshot if resolved_snapshot.is_dir() else resolved_snapshot.parent
    identities = tuple(
        SnapshotFileIdentity(
            path=path.relative_to(relative_root).as_posix(),
            sha256=digest,
            bytes=byte_count,
        )
        for path in files
        for digest, byte_count in (_file_identity(path),)
    )
    if discover_snapshot_files(resolved_snapshot) != files:
        raise ValueError("snapshot file set changed while its identity was captured")
    snapshot_bytes = sum(identity.bytes for identity in identities)
    return DatasetSnapshotIdentity(
        snapshot_sha256=_aggregate_snapshot_digest(identities, snapshot_bytes),
        snapshot_bytes=snapshot_bytes,
        files=identities,
    )


def verify_snapshot_identity(snapshot: Path, expected: DatasetSnapshotIdentity) -> None:
    """Fail closed if the supported file set, paths, sizes, or contents changed."""

    try:
        current = capture_snapshot_identity(snapshot)
    except (OSError, ValueError) as error:
        raise ValueError(f"snapshot changed while it was being processed: {error}") from error
    if current != expected:
        raise ValueError("snapshot changed while it was being processed")


def build_dataset_lineage_manifest(
    *,
    source: DatasetSource,
    registry_sha256: str,
    registry_bytes: int,
    snapshot: DatasetSnapshotIdentity,
    report: DatasetAuditReport,
    audit_artifact: str,
    audit_sha256: str,
    rejection_ledger_artifact: str,
    rejection_ledger_sha256: str,
    rejection_ledger_rows: int,
) -> DatasetLineageManifest:
    """Gate a lineage marker on truthful, complete, cross-bound audit evidence."""

    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise ValueError("lineage requires an exact registry-PASS pinned source")
    if (
        report.source_name != source.name
        or report.source_repository_id != source.repository_id
        or report.source_revision != source.revision
    ):
        raise ValueError("audit report source identity does not match the registry source")
    if report.input_scope != "complete-snapshot":
        raise ValueError("lineage requires a complete-snapshot audit")
    if (
        report.input_sha256 != snapshot.snapshot_sha256
        or report.input_bytes != snapshot.snapshot_bytes
    ):
        raise ValueError("audit report snapshot identity does not match the captured snapshot")

    matches_declared = report.row_count == source.observed_rows
    if (
        report.declared_row_count != source.observed_rows
        or report.matches_declared_row_count is not matches_declared
    ):
        raise ValueError("audit declared/observed row count evidence is inconsistent")
    if report.status not in {VerificationStatus.PASS, VerificationStatus.FAIL}:
        raise ValueError("audit report must have terminal PASS or FAIL status")
    if report.status is VerificationStatus.PASS and (not matches_declared or report.issues):
        raise ValueError("PASS audit requires an exact row count and no issues")
    if report.status is VerificationStatus.FAIL and not report.issues:
        raise ValueError("FAIL audit requires recorded issues")

    expected_audit_sha256 = content_digest(canonical_json_bytes(report.model_dump(mode="json")))
    if audit_sha256 != expected_audit_sha256:
        raise ValueError("canonical audit report digest does not match audit_sha256")
    if (
        not report.rejection_ledger_artifact
        or report.rejection_ledger_sha256 != rejection_ledger_sha256
        or report.rejection_ledger_rows != rejection_ledger_rows
        or rejection_ledger_rows != report.rejected_row_count
    ):
        raise ValueError("audit rejection ledger binding is incomplete or inconsistent")

    return DatasetLineageManifest(
        status=report.status,
        source=source,
        registry_sha256=registry_sha256,
        registry_bytes=registry_bytes,
        snapshot=snapshot,
        audit_artifact=audit_artifact,
        audit_sha256=audit_sha256,
        audit_row_count=report.row_count,
        audit_logical_rows_sha256=report.logical_rows_sha256,
        rejection_ledger_artifact=rejection_ledger_artifact,
        rejection_ledger_sha256=rejection_ledger_sha256,
        rejection_ledger_rows=rejection_ledger_rows,
    )
