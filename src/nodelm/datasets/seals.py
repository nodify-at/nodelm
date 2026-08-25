from __future__ import annotations

from dataclasses import dataclass


class SnapshotSealError(ValueError):
    """Snapshot evidence is not authorized by the checked-in trust root."""


@dataclass(frozen=True)
class SnapshotSeal:
    transfer_receipt_sha256: str
    snapshot_sha256: str
    snapshot_file_count: int


AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION: dict[tuple[str, str], SnapshotSeal] = {
    (
        "open-swe-traces",
        "ed95cef24df8d8bd79b4ceb0192cb420fde06521",
    ): SnapshotSeal(
        transfer_receipt_sha256=(
            "44ea157ebd802a5604301c82e8785003d67f90d0ed64efcc079059dfd4290a84"
        ),
        snapshot_sha256=("218df319b86b21be12f22284e25cdbaf90e77fdbc3e2d996c152ec8c54c03aa3"),
        snapshot_file_count=231,
    ),
    (
        "swe-rebench-v2",
        "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0",
    ): SnapshotSeal(
        transfer_receipt_sha256=(
            "fbcd4fbb2b9c4b887ef15f368f3673c07d82d4ba81d2b0d0eed7e3dd6d1fe254"
        ),
        snapshot_sha256=("4f4328b560d27918da8f2d251c037789add5b5f7566c46825eeed91aa9d9c117"),
        snapshot_file_count=1,
    ),
    (
        "swe-rebench-v2-prs",
        "fbf0ecf50f268d5344149e2f0097db6bede83737",
    ): SnapshotSeal(
        transfer_receipt_sha256=(
            "d7e6c8e4abb7a8488c62588a0cc95c089bc12603c22c9cf85f2acad2a1c59570"
        ),
        snapshot_sha256=("c2e6edf039c1e49f4cc4193c0b5cb26eb3376b21f71bae7148b17b157b463780"),
        snapshot_file_count=3,
    ),
}


def require_authorized_snapshot_seal(
    *,
    source_name: str,
    source_revision: str,
    transfer_receipt_sha256: str,
    snapshot_sha256: str,
    snapshot_file_count: int,
) -> None:
    seal = AUTHORIZED_SNAPSHOT_SEALS_BY_SOURCE_REVISION.get(
        (source_name, source_revision.casefold())
    )
    if seal is None:
        raise SnapshotSealError(f"no authorized snapshot seal for {source_name}@{source_revision}")
    if seal != SnapshotSeal(
        transfer_receipt_sha256=transfer_receipt_sha256,
        snapshot_sha256=snapshot_sha256,
        snapshot_file_count=snapshot_file_count,
    ):
        raise SnapshotSealError(
            f"snapshot evidence is not authorized for {source_name}@{source_revision}"
        )
