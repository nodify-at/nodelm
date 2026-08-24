from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from nodelm.artifacts import (
    ArtifactCollisionError,
    ArtifactWriteResult,
    canonical_json_bytes,
    content_digest,
    file_identity,
    write_immutable_stream,
)
from nodelm.datasets.lineage import (
    DatasetSnapshotTransferReceipt,
    build_snapshot_transfer_receipt,
    capture_snapshot_identity,
    verify_snapshot_identity,
)
from nodelm.datasets.registry import DatasetRegistry
from nodelm.models import DatasetSource, VerificationStatus


class SnapshotDownloader(Protocol):
    def __call__(
        self,
        source: DatasetSource,
        *,
        destination: Path,
        allow_patterns: tuple[str, ...],
    ) -> Path: ...


class SnapshotTransferError(ValueError):
    """A pinned snapshot transfer cannot be completed or receipted safely."""


@dataclass(frozen=True)
class SnapshotTransferResult:
    snapshot_path: Path
    receipt: DatasetSnapshotTransferReceipt
    receipt_result: ArtifactWriteResult


def _require_file_identity(path: Path, expected: tuple[str, int]) -> None:
    if file_identity(path) != expected:
        raise ValueError(f"input changed while it was being processed: {path}")


def _validate_transfer_paths(
    *,
    destination: Path,
    receipt_output: Path,
    config: Path,
) -> None:
    resolved_destination = destination.resolve()
    resolved_receipt = receipt_output.resolve()
    if resolved_receipt == resolved_destination or resolved_receipt.is_relative_to(
        resolved_destination
    ):
        raise SnapshotTransferError("transfer receipt output must be outside the destination")
    if resolved_receipt == config.resolve():
        raise SnapshotTransferError("transfer receipt output must be distinct from the registry")


def transfer_snapshot(
    *,
    source_name: str,
    destination: Path,
    config: Path,
    downloader: SnapshotDownloader,
    allow_patterns: tuple[str, ...] = (),
    receipt_output: Path | None = None,
) -> SnapshotTransferResult:
    """Download a pinned snapshot and publish its content-bound transfer receipt last."""

    if len(allow_patterns) != len(set(allow_patterns)):
        raise SnapshotTransferError("download allow patterns must be unique")
    normalized_patterns = tuple(sorted(allow_patterns))
    if any(not pattern for pattern in normalized_patterns):
        raise SnapshotTransferError("download allow patterns must be non-empty")
    receipt_path = receipt_output or destination.with_name(f"{destination.name}.transfer.json")
    _validate_transfer_paths(
        destination=destination,
        receipt_output=receipt_path,
        config=config,
    )

    if destination.exists():
        if not destination.is_dir():
            raise SnapshotTransferError(f"download destination must be a directory: {destination}")
        if any(destination.iterdir()):
            raise SnapshotTransferError(f"download destination must be new or empty: {destination}")

    try:
        registry_payload = config.read_bytes()
        registry_identity = (content_digest(registry_payload), len(registry_payload))
        registry = DatasetRegistry.from_bytes(registry_payload)
        _require_file_identity(config, registry_identity)
        source = registry.by_name(source_name)
    except (OSError, ValueError, ValidationError) as error:
        raise SnapshotTransferError(f"invalid dataset registry: {error}") from error
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise SnapshotTransferError("snapshot download requires a registry-verified pinned source")

    try:
        downloaded_path = downloader(
            source,
            destination=destination,
            allow_patterns=normalized_patterns,
        ).resolve()
        if downloaded_path != destination.resolve():
            raise ValueError("downloaded snapshot path does not match the requested destination")
        snapshot_identity = capture_snapshot_identity(downloaded_path)
        receipt = build_snapshot_transfer_receipt(
            source=source,
            registry_sha256=registry_identity[0],
            registry_bytes=registry_identity[1],
            snapshot=snapshot_identity,
            snapshot_scope="filtered" if normalized_patterns else "complete",
            allow_patterns=normalized_patterns,
        )
        receipt_bytes = canonical_json_bytes(receipt.model_dump(mode="json"))

        def verify_receipt_boundary() -> None:
            _require_file_identity(config, registry_identity)
            verify_snapshot_identity(downloaded_path, snapshot_identity)

        receipt_result = write_immutable_stream(
            receipt_path,
            lambda stream: stream.write(receipt_bytes),
            before_publish=verify_receipt_boundary,
        )
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise SnapshotTransferError(f"snapshot transfer failed: {error}") from error

    return SnapshotTransferResult(
        snapshot_path=downloaded_path,
        receipt=receipt,
        receipt_result=receipt_result,
    )
