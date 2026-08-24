from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from nodelm.artifacts import canonical_json_bytes, content_digest
from nodelm.datasets.audit import audit_rows
from nodelm.datasets.lineage import (
    DatasetLineageManifest,
    DatasetSnapshotIdentity,
    DeferredAuditChecks,
    SnapshotFileIdentity,
    build_dataset_lineage_manifest,
    capture_snapshot_identity,
    verify_snapshot_identity,
)
from nodelm.models import DatasetAuditReport, DatasetSource, VerificationStatus


def _source(*, observed_rows: int = 2) -> DatasetSource:
    return DatasetSource(
        name="fixture",
        repository_id="owner/fixture",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=observed_rows,
        evidence_urls=("https://example.invalid/official",),
        status=VerificationStatus.PASS,
    )


def _write_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "z.jsonl").write_text(
        '{"instance_id":"two","repo":"acme/widget","license":"MIT","resolved":false}\n',
        encoding="utf-8",
    )
    nested = root / "nested"
    nested.mkdir()
    (nested / "a.jsonl").write_text(
        '{"instance_id":"one","repo":"acme/widget","license":"MIT","resolved":true}\n',
        encoding="utf-8",
    )
    return root


def _report_with_ledger(
    source: DatasetSource,
    snapshot: DatasetSnapshotIdentity,
) -> DatasetAuditReport:
    report = audit_rows(
        source,
        (
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "license": "MIT",
                "resolved": True,
            },
            {
                "instance_id": "two",
                "repo": "acme/widget",
                "license": "MIT",
                "resolved": False,
            },
        ),
        input_sha256=snapshot.snapshot_sha256,
        input_bytes=snapshot.snapshot_bytes,
    )
    return DatasetAuditReport.model_validate(
        {
            **report.model_dump(mode="json"),
            "rejection_ledger_artifact": "audit.rejections.jsonl",
            "rejection_ledger_sha256": hashlib.sha256(b"").hexdigest(),
            "rejection_ledger_rows": 0,
        }
    )


def _build_manifest(
    source: DatasetSource,
    snapshot: DatasetSnapshotIdentity,
    report: DatasetAuditReport,
) -> DatasetLineageManifest:
    report_sha256 = content_digest(canonical_json_bytes(report.model_dump(mode="json")))
    return build_dataset_lineage_manifest(
        source=source,
        registry_sha256="b" * 64,
        registry_bytes=123,
        snapshot=snapshot,
        report=report,
        audit_artifact="audit.json",
        audit_sha256=report_sha256,
        rejection_ledger_artifact="audit.rejections.jsonl",
        rejection_ledger_sha256=hashlib.sha256(b"").hexdigest(),
        rejection_ledger_rows=0,
    )


def test_snapshot_identity_is_root_independent_sorted_and_content_addressed(
    tmp_path: Path,
) -> None:
    first_root = _write_snapshot(tmp_path / "first-root")
    second_root = _write_snapshot(tmp_path / "renamed-root")

    first = capture_snapshot_identity(first_root)
    second = capture_snapshot_identity(second_root)

    assert first == second
    assert [item.path for item in first.files] == ["nested/a.jsonl", "z.jsonl"]
    assert first.snapshot_bytes == sum(item.bytes for item in first.files)

    (second_root / "nested" / "a.jsonl").rename(second_root / "nested" / "b.jsonl")
    renamed = capture_snapshot_identity(second_root)
    assert renamed.snapshot_sha256 != first.snapshot_sha256

    (second_root / "nested" / "b.jsonl").write_text("{}\n", encoding="utf-8")
    changed = capture_snapshot_identity(second_root)
    assert changed.snapshot_sha256 != renamed.snapshot_sha256


@pytest.mark.parametrize(
    "path",
    (".", "/absolute.jsonl", "../escape.jsonl", "nested\\file.jsonl", "./file.jsonl", "a//b"),
)
def test_snapshot_file_identity_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="normalized relative POSIX"):
        SnapshotFileIdentity(path=path, sha256="a" * 64, bytes=1)


def test_snapshot_identity_rejects_forged_bytes_digest_and_order() -> None:
    files = (
        SnapshotFileIdentity(path="b.jsonl", sha256="b" * 64, bytes=2),
        SnapshotFileIdentity(path="a.jsonl", sha256="a" * 64, bytes=1),
    )
    with pytest.raises(ValidationError, match="strictly sorted"):
        DatasetSnapshotIdentity(snapshot_sha256="c" * 64, snapshot_bytes=3, files=files)

    file_identity = SnapshotFileIdentity(path="a.jsonl", sha256="a" * 64, bytes=1)
    with pytest.raises(ValidationError, match="snapshot_bytes"):
        DatasetSnapshotIdentity(
            snapshot_sha256="c" * 64,
            snapshot_bytes=2,
            files=(file_identity,),
        )
    with pytest.raises(ValidationError, match="snapshot_sha256"):
        DatasetSnapshotIdentity(
            snapshot_sha256="c" * 64,
            snapshot_bytes=1,
            files=(file_identity,),
        )


def test_snapshot_verification_detects_file_set_and_content_drift(tmp_path: Path) -> None:
    root = _write_snapshot(tmp_path / "snapshot")
    identity = capture_snapshot_identity(root)
    verify_snapshot_identity(root, identity)

    (root / "new.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify_snapshot_identity(root, identity)


def test_lineage_builder_binds_complete_canonical_audit_and_ledger(tmp_path: Path) -> None:
    snapshot = capture_snapshot_identity(_write_snapshot(tmp_path / "snapshot"))
    source = _source()
    report = _report_with_ledger(source, snapshot)

    manifest = _build_manifest(source, snapshot, report)

    assert manifest.status is VerificationStatus.PASS
    assert manifest.source == source
    assert manifest.snapshot == snapshot
    assert manifest.audit_row_count == 2
    assert manifest.audit_logical_rows_sha256 == report.logical_rows_sha256
    assert manifest.deferred_checks == DeferredAuditChecks()
    assert set(manifest.deferred_checks.model_dump(mode="json").values()) == {"NOT RUN"}


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"input_scope": "partial-snapshot"}, "complete-snapshot"),
        ({"input_sha256": "f" * 64}, "snapshot identity"),
        ({"source_revision": "f" * 40}, "source identity"),
        ({"rejection_ledger_sha256": "f" * 64}, "rejection ledger"),
    ),
)
def test_lineage_builder_rejects_cross_artifact_mismatches(
    tmp_path: Path,
    update: dict[str, object],
    message: str,
) -> None:
    snapshot = capture_snapshot_identity(_write_snapshot(tmp_path / "snapshot"))
    source = _source()
    report = _report_with_ledger(source, snapshot).model_copy(update=update)

    with pytest.raises(ValueError, match=message):
        _build_manifest(source, snapshot, report)


def test_lineage_builder_accepts_truthful_row_drift_failure(tmp_path: Path) -> None:
    snapshot = capture_snapshot_identity(_write_snapshot(tmp_path / "snapshot"))
    source = _source(observed_rows=3)
    report = _report_with_ledger(source, snapshot)

    manifest = _build_manifest(source, snapshot, report)

    assert report.status is VerificationStatus.FAIL
    assert manifest.status is VerificationStatus.FAIL
    assert manifest.audit_row_count == 2


def test_lineage_builder_rejects_noncanonical_report_digest(tmp_path: Path) -> None:
    snapshot = capture_snapshot_identity(_write_snapshot(tmp_path / "snapshot"))
    source = _source()
    report = _report_with_ledger(source, snapshot)

    with pytest.raises(ValueError, match="canonical audit report digest"):
        build_dataset_lineage_manifest(
            source=source,
            registry_sha256="b" * 64,
            registry_bytes=123,
            snapshot=snapshot,
            report=report,
            audit_artifact="audit.json",
            audit_sha256="f" * 64,
            rejection_ledger_artifact="audit.rejections.jsonl",
            rejection_ledger_sha256=hashlib.sha256(b"").hexdigest(),
            rejection_ledger_rows=0,
        )
