from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import ValidationError

from nodelm.artifacts import (
    ArtifactCollisionError,
    ArtifactWriteResult,
    canonical_json_bytes,
    content_digest,
    file_identity,
    write_immutable_stream,
)
from nodelm.datasets.audit import audit_rows
from nodelm.datasets.lineage import (
    DatasetSnapshotTransferReceipt,
    build_dataset_lineage_manifest,
    capture_snapshot_identity,
    verify_snapshot_identity,
)
from nodelm.datasets.materialize import discover_snapshot_files, iter_snapshot_rows
from nodelm.datasets.registry import DatasetRegistry
from nodelm.models import (
    SNAPSHOT_IDENTITY_SCHEMA,
    DatasetAuditReport,
    DatasetSource,
    VerificationStatus,
)


class SnapshotAuditError(ValueError):
    """A snapshot cannot be audited or published under the immutable contract."""


@dataclass(frozen=True)
class SnapshotAuditResult:
    report: DatasetAuditReport
    audit_result: ArtifactWriteResult
    rejection_ledger_result: ArtifactWriteResult
    lineage_result: ArtifactWriteResult


def _require_file_identity(path: Path, expected: tuple[str, int]) -> None:
    if file_identity(path) != expected:
        raise ValueError(f"input changed while it was being processed: {path}")


def _load_transfer_receipt(
    path: Path,
) -> tuple[DatasetSnapshotTransferReceipt, tuple[str, int]]:
    payload = path.read_bytes()
    identity = (content_digest(payload), len(payload))
    receipt = DatasetSnapshotTransferReceipt.model_validate_json(payload)
    if payload != canonical_json_bytes(receipt.model_dump(mode="json")):
        raise ValueError("transfer receipt is not the canonical immutable artifact")
    return receipt, identity


def _resolved_inventory(
    snapshot: Path,
    receipt: DatasetSnapshotTransferReceipt,
) -> tuple[Path, ...]:
    resolved_snapshot = snapshot.resolve()
    discovered = discover_snapshot_files(resolved_snapshot)
    relative_root = resolved_snapshot if resolved_snapshot.is_dir() else resolved_snapshot.parent
    observed = {path.relative_to(relative_root).as_posix(): path for path in discovered}
    expected_paths = tuple(identity.path for identity in receipt.snapshot.files)
    if tuple(sorted(observed)) != expected_paths:
        raise ValueError("snapshot inventory does not match transfer receipt")
    return tuple(observed[path] for path in expected_paths)


def _stage_snapshot(
    source_files: tuple[Path, ...],
    receipt: DatasetSnapshotTransferReceipt,
    staging_snapshot: Path,
) -> tuple[Path, ...]:
    for source_path, expected in zip(source_files, receipt.snapshot.files, strict=True):
        destination = staging_snapshot / expected.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    staged_identity = capture_snapshot_identity(staging_snapshot)
    if staged_identity != receipt.snapshot:
        raise ValueError("staged snapshot identity does not match transfer receipt")
    return tuple(staging_snapshot / identity.path for identity in receipt.snapshot.files)


def _validate_artifact_paths(
    *,
    snapshot: Path,
    config: Path,
    receipt_path: Path,
    output: Path,
    ledger_path: Path,
    lineage_path: Path,
    staging_root: Path,
) -> None:
    artifact_paths = tuple(path.resolve() for path in (output, ledger_path, lineage_path))
    resolved_staging_root = staging_root.resolve()
    if len(set(artifact_paths)) != len(artifact_paths):
        raise SnapshotAuditError("audit, rejection ledger, and lineage outputs must be distinct")
    if resolved_staging_root in artifact_paths:
        raise SnapshotAuditError("staging root must be distinct from audit output artifacts")

    resolved_snapshot = snapshot.resolve()
    if resolved_snapshot.is_dir():
        if any(path.is_relative_to(resolved_snapshot) for path in artifact_paths):
            raise SnapshotAuditError("all audit outputs must be outside a directory snapshot")
        if resolved_staging_root.is_relative_to(resolved_snapshot):
            raise SnapshotAuditError("staging root must be outside a directory snapshot")
    elif resolved_snapshot in artifact_paths:
        raise SnapshotAuditError("audit outputs must be distinct from the snapshot input")

    protected_inputs = {config.resolve(), receipt_path.resolve()}
    if resolved_staging_root in protected_inputs:
        raise SnapshotAuditError("staging root must be distinct from audit input files")
    if any(path in protected_inputs for path in artifact_paths):
        raise SnapshotAuditError(
            "audit outputs must be distinct from registry and transfer receipt inputs"
        )


def audit_snapshot(
    *,
    source_name: str,
    snapshot: Path,
    receipt_path: Path,
    output: Path,
    config: Path,
    lineage_output: Path | None = None,
    rejections_output: Path | None = None,
    staging_root: Path | None = None,
) -> SnapshotAuditResult:
    """Audit a receipt-bound local snapshot from a private, receipt-verified staging copy."""

    lineage_path = lineage_output or output.with_name(f"{output.stem}.lineage.json")
    ledger_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    staging_base = (staging_root or output.resolve().parent).resolve()
    _validate_artifact_paths(
        snapshot=snapshot,
        config=config,
        receipt_path=receipt_path,
        output=output,
        ledger_path=ledger_path,
        lineage_path=lineage_path,
        staging_root=staging_base,
    )

    try:
        registry_payload = config.read_bytes()
        registry_identity = (content_digest(registry_payload), len(registry_payload))
        registry = DatasetRegistry.from_bytes(registry_payload)
        _require_file_identity(config, registry_identity)
        source = registry.by_name(source_name)
        receipt, receipt_identity = _load_transfer_receipt(receipt_path)
        _require_file_identity(receipt_path, receipt_identity)
    except (OSError, ValueError, ValidationError) as error:
        raise SnapshotAuditError(f"invalid snapshot audit inputs: {error}") from error

    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise SnapshotAuditError("snapshot audit requires a registry-verified pinned source")
    if receipt.source != source:
        raise SnapshotAuditError("transfer receipt source does not match the registry source")
    if (
        receipt.registry_sha256 != registry_identity[0]
        or receipt.registry_bytes != registry_identity[1]
    ):
        raise SnapshotAuditError("transfer receipt registry identity does not match --config")
    if receipt.snapshot_scope != "complete" or receipt.allow_patterns:
        raise SnapshotAuditError("snapshot audit requires a complete immutable transfer receipt")

    try:
        source_files = _resolved_inventory(snapshot, receipt)
        staging_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".nodelm-snapshot-audit-",
            dir=staging_base,
        ) as temporary_name:
            staging_snapshot = Path(temporary_name)
            staged_files = _stage_snapshot(source_files, receipt, staging_snapshot)
            return _audit_staged_snapshot(
                source=source,
                staged_files=staged_files,
                staging_snapshot=staging_snapshot,
                receipt=receipt,
                receipt_path=receipt_path,
                receipt_identity=receipt_identity,
                config=config,
                registry_identity=registry_identity,
                output=output,
                ledger_path=ledger_path,
                lineage_path=lineage_path,
            )
    except SnapshotAuditError:
        raise
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise SnapshotAuditError(f"snapshot audit failed: {error}") from error


def _audit_staged_snapshot(
    *,
    source: DatasetSource,
    staged_files: tuple[Path, ...],
    staging_snapshot: Path,
    receipt: DatasetSnapshotTransferReceipt,
    receipt_path: Path,
    receipt_identity: tuple[str, int],
    config: Path,
    registry_identity: tuple[str, int],
    output: Path,
    ledger_path: Path,
    lineage_path: Path,
) -> SnapshotAuditResult:
    ledger_row_count = 0
    ledger_byte_count = 0
    report: DatasetAuditReport | None = None

    def write_rejection(rejection: dict[str, Any], stream: BinaryIO) -> None:
        nonlocal ledger_byte_count, ledger_row_count
        encoded = canonical_json_bytes(rejection)
        stream.write(encoded)
        ledger_row_count += 1
        ledger_byte_count += len(encoded)

    def write_rejection_ledger(stream: BinaryIO) -> None:
        nonlocal report
        report = audit_rows(
            source,
            iter_snapshot_rows(staged_files),
            input_sha256=receipt.snapshot.snapshot_sha256,
            input_bytes=receipt.snapshot.snapshot_bytes,
            input_identity_schema=SNAPSHOT_IDENTITY_SCHEMA,
            expect_complete_snapshot=True,
            rejection_sink=lambda rejection: write_rejection(rejection, stream),
        )

    def verify_common_boundary() -> None:
        _require_file_identity(config, registry_identity)
        _require_file_identity(receipt_path, receipt_identity)
        verify_snapshot_identity(staging_snapshot, receipt.snapshot)

    try:
        ledger_result = write_immutable_stream(
            ledger_path,
            write_rejection_ledger,
            before_publish=verify_common_boundary,
        )
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise SnapshotAuditError(f"snapshot audit publication failed: {error}") from error
    if report is None:  # pragma: no cover - immutable writer always invokes its writer
        raise RuntimeError("snapshot audit writer did not produce a report")
    ledger_identity = (ledger_result.digest, ledger_byte_count)

    report = type(report).model_validate(
        {
            **report.model_dump(mode="json"),
            "rejection_ledger_artifact": os.path.relpath(
                ledger_result.path,
                start=output.resolve().parent,
            ),
            "rejection_ledger_sha256": ledger_result.digest,
            "rejection_ledger_rows": ledger_row_count,
        }
    )
    report_bytes = canonical_json_bytes(report.model_dump(mode="json"))

    def verify_report_boundary() -> None:
        _require_file_identity(config, registry_identity)
        _require_file_identity(receipt_path, receipt_identity)
        verify_snapshot_identity(staging_snapshot, receipt.snapshot)
        _require_file_identity(ledger_result.path, ledger_identity)

    try:
        report_result = write_immutable_stream(
            output,
            lambda stream: stream.write(report_bytes),
            before_publish=verify_report_boundary,
        )
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise SnapshotAuditError(f"snapshot audit publication failed: {error}") from error
    report_identity = (report_result.digest, len(report_bytes))

    def verify_lineage_boundary() -> None:
        _require_file_identity(config, registry_identity)
        _require_file_identity(receipt_path, receipt_identity)
        verify_snapshot_identity(staging_snapshot, receipt.snapshot)
        _require_file_identity(ledger_result.path, ledger_identity)
        _require_file_identity(report_result.path, report_identity)

    try:
        manifest = build_dataset_lineage_manifest(
            source=source,
            registry_sha256=registry_identity[0],
            registry_bytes=registry_identity[1],
            snapshot=receipt.snapshot,
            transfer_receipt=receipt,
            transfer_receipt_artifact=os.path.relpath(
                receipt_path.resolve(),
                start=lineage_path.resolve().parent,
            ),
            transfer_receipt_sha256=receipt_identity[0],
            transfer_receipt_bytes=receipt_identity[1],
            report=report,
            audit_artifact=os.path.relpath(
                report_result.path,
                start=lineage_path.resolve().parent,
            ),
            audit_sha256=report_result.digest,
            rejection_ledger_artifact=os.path.relpath(
                ledger_result.path,
                start=lineage_path.resolve().parent,
            ),
            rejection_ledger_sha256=ledger_result.digest,
            rejection_ledger_rows=ledger_row_count,
        )
        lineage_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
        lineage_result = write_immutable_stream(
            lineage_path,
            lambda stream: stream.write(lineage_bytes),
            before_publish=verify_lineage_boundary,
        )
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise SnapshotAuditError(f"snapshot lineage publication failed: {error}") from error

    return SnapshotAuditResult(
        report=report,
        audit_result=report_result,
        rejection_ledger_result=ledger_result,
        lineage_result=lineage_result,
    )
