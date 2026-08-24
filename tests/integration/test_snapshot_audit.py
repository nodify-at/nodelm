from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from typer.testing import CliRunner

from nodelm.artifacts import canonical_json_bytes, file_identity
from nodelm.cli import app
from nodelm.datasets.lineage import (
    DatasetLineageManifest,
    DatasetSnapshotIdentity,
    DatasetSnapshotTransferReceipt,
    build_snapshot_transfer_receipt,
    capture_snapshot_identity,
)
from nodelm.datasets.registry import DatasetRegistry


def _write_registry(path: Path, *, observed_rows: int = 2) -> Path:
    path.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture\n"
        "    repository_id: owner/fixture\n"
        f"    revision: {'a' * 40}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        f"    observed_rows: {observed_rows}\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n",
        encoding="utf-8",
    )
    return path


def _write_mixed_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "a.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "license": "MIT",
                "language": "TypeScript",
                "resolved": True,
                "trajectory": [{"action": "read"}],
                "patch": "+one\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parquet.write_table(
        pa.table(
            {
                "instance_id": ["two"],
                "repo": ["beta/service"],
                "license": ["GPL-3.0"],
                "language": ["JavaScript"],
                "resolved": [False],
                "trajectory": [[]],
                "patch": ["+two\n"],
            }
        ),
        root / "b.parquet",
    )
    return root


def _write_transfer_receipt(
    path: Path,
    *,
    snapshot: Path,
    registry: Path,
) -> Path:
    registry_sha256, registry_bytes = file_identity(registry)
    source = DatasetRegistry.load(registry).by_name("fixture")
    receipt = build_snapshot_transfer_receipt(
        source=source,
        registry_sha256=registry_sha256,
        registry_bytes=registry_bytes,
        snapshot=capture_snapshot_identity(snapshot),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt.model_dump(mode="json")))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact(owner: Path, reference: str) -> Path:
    referenced = Path(reference)
    path = referenced if referenced.is_absolute() else owner.resolve().parent / referenced
    return path.resolve()


def _invoke(
    snapshot: Path,
    registry: Path,
    output: Path,
    *extra: str,
    receipt: Path | None = None,
) -> object:
    transfer_receipt = receipt or _write_transfer_receipt(
        registry.with_name("snapshot.transfer.json"),
        snapshot=snapshot,
        registry=registry,
    )
    return CliRunner().invoke(
        app,
        [
            "datasets",
            "audit-snapshot",
            "--source",
            "fixture",
            "--snapshot",
            str(snapshot),
            "--receipt",
            str(transfer_receipt),
            "--output",
            str(output),
            "--config",
            str(registry),
            *extra,
        ],
    )


def test_complete_snapshot_audit_is_offline_bound_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = tmp_path / "artifacts" / "audit.json"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network or download path must not be called")

    monkeypatch.setattr("nodelm.cli.download_pinned_snapshot", forbidden)
    monkeypatch.setattr("nodelm.cli.verify_hub_source", forbidden)

    first = _invoke(snapshot, registry, output)
    second = _invoke(snapshot, registry, output)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    receipt_path = registry.with_name("snapshot.transfer.json")
    receipt = DatasetSnapshotTransferReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    audit = json.loads(output.read_text(encoding="utf-8"))
    ledger = output.with_name("audit.rejections.jsonl")
    lineage_path = output.with_name("audit.lineage.json")
    lineage = DatasetLineageManifest.model_validate_json(lineage_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["schema_version"] == "nodelm.dataset-audit/v2"
    assert audit["input_scope"] == "complete-snapshot"
    assert audit["input_identity_schema"] == "nodelm.dataset-snapshot-identity/v1"
    assert audit["row_count"] == 2
    assert audit["language_distribution"] == {"JavaScript": 1, "TypeScript": 1}
    assert json.loads(ledger.read_text(encoding="utf-8"))["disposition"] == "REJECT"
    assert lineage.status.value == "PASS"
    assert lineage.snapshot.snapshot_sha256 == audit["input_sha256"]
    assert lineage.snapshot.snapshot_bytes == audit["input_bytes"]
    assert lineage.rejection_ledger_rows == 1
    assert lineage.source.revision == "a" * 40

    assert receipt.snapshot == lineage.snapshot
    assert receipt.source == lineage.source
    assert receipt.registry_sha256 == lineage.registry_sha256 == _sha256(registry)
    assert receipt.registry_bytes == lineage.registry_bytes == registry.stat().st_size
    assert lineage.transfer_receipt_sha256 == _sha256(receipt_path)
    assert lineage.transfer_receipt_bytes == receipt_path.stat().st_size
    assert _resolve_artifact(lineage_path, lineage.transfer_receipt_artifact) == receipt_path

    report_artifact = _resolve_artifact(lineage_path, lineage.audit_artifact)
    ledger_from_lineage = _resolve_artifact(lineage_path, lineage.rejection_ledger_artifact)
    ledger_from_report = _resolve_artifact(output, audit["rejection_ledger_artifact"])
    assert report_artifact == output
    assert ledger_from_lineage == ledger_from_report == ledger
    assert lineage.audit_sha256 == _sha256(report_artifact)
    assert lineage.rejection_ledger_sha256 == _sha256(ledger_from_lineage)
    assert canonical_json_bytes(audit) == output.read_bytes()
    assert canonical_json_bytes(lineage.model_dump(mode="json")) == lineage_path.read_bytes()
    assert canonical_json_bytes(receipt.model_dump(mode="json")) == receipt_path.read_bytes()

    relative_root = snapshot.resolve()
    for identity in lineage.snapshot.files:
        persisted_file = relative_root / identity.path
        assert persisted_file.stat().st_size == identity.bytes
        assert _sha256(persisted_file) == identity.sha256


def test_complete_snapshot_rows_are_streamed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = tmp_path / "artifacts" / "audit.json"
    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    from nodelm.datasets import snapshot_audit as snapshot_audit_service

    original_rows = snapshot_audit_service.iter_snapshot_rows
    original_verify = snapshot_audit_service.verify_snapshot_identity
    row_passes = 0
    streamed_paths: tuple[Path, ...] = ()
    verified_paths: list[Path] = []

    def counted_rows(paths: tuple[Path, ...]) -> Iterator[dict[str, Any]]:
        nonlocal row_passes, streamed_paths
        row_passes += 1
        streamed_paths = paths
        yield from original_rows(paths)

    def counted_verification(path: Path, expected: DatasetSnapshotIdentity) -> None:
        verified_paths.append(path.resolve())
        original_verify(path, expected)

    monkeypatch.setattr(snapshot_audit_service, "iter_snapshot_rows", counted_rows)
    monkeypatch.setattr(
        snapshot_audit_service,
        "verify_snapshot_identity",
        counted_verification,
    )

    result = _invoke(
        snapshot,
        registry,
        output,
        "--staging-root",
        str(staging_root),
    )

    assert result.exit_code == 0, result.output
    assert row_passes == 1
    assert streamed_paths
    assert all(path.is_relative_to(staging_root) for path in streamed_paths)
    assert all(not path.is_relative_to(snapshot) for path in streamed_paths)
    assert len(verified_paths) == 3
    assert len(set(verified_paths)) == 1
    assert verified_paths[0] != snapshot.resolve()
    assert verified_paths[0].is_relative_to(staging_root)
    assert not verified_paths[0].exists()


def test_snapshot_audit_parses_the_same_registry_bytes_it_identifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    replacement_registry = _write_registry(
        tmp_path / "replacement-registry.yaml",
        observed_rows=3,
    )
    receipt = _write_transfer_receipt(
        tmp_path / "snapshot.transfer.json",
        snapshot=snapshot,
        registry=registry,
    )
    output = tmp_path / "artifacts" / "audit.json"
    original_payload = registry.read_bytes()
    replacement_payload = replacement_registry.read_bytes()

    from nodelm.datasets import snapshot_audit as snapshot_audit_service

    original_file_identity = snapshot_audit_service.file_identity
    original_read_bytes = Path.read_bytes
    registry_reads = 0
    identity_was_read_before_parse = False

    def tracked_file_identity(candidate: Path) -> tuple[str, int]:
        nonlocal identity_was_read_before_parse
        identity = original_file_identity(candidate)
        if candidate.resolve() == registry.resolve() and registry_reads == 0:
            identity_was_read_before_parse = True
        return identity

    def registry_aba(candidate: Path) -> bytes:
        nonlocal registry_reads
        if candidate.resolve() != registry.resolve():
            return original_read_bytes(candidate)
        registry_reads += 1
        if identity_was_read_before_parse or registry_reads > 1:
            return replacement_payload
        return original_payload

    monkeypatch.setattr(snapshot_audit_service, "file_identity", tracked_file_identity)
    monkeypatch.setattr(Path, "read_bytes", registry_aba)

    result = _invoke(snapshot, registry, output, receipt=receipt)

    assert result.exit_code == 0, result.output
    assert registry_reads == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source_repository_id"] == "owner/fixture"
    assert report["declared_row_count"] == 2


def test_row_drift_publishes_fail_lineage_evidence(tmp_path: Path) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml", observed_rows=3)
    output = tmp_path / "artifacts" / "audit.json"

    result = _invoke(snapshot, registry, output)

    assert result.exit_code == 1, result.output
    audit = json.loads(output.read_text(encoding="utf-8"))
    lineage = json.loads(output.with_name("audit.lineage.json").read_text(encoding="utf-8"))
    assert audit["status"] == "FAIL"
    assert audit["matches_declared_row_count"] is False
    assert lineage["status"] == "FAIL"
    assert lineage["audit_row_count"] == 2


def test_receipt_for_same_size_unrelated_snapshot_fails_without_lineage(
    tmp_path: Path,
) -> None:
    correct_snapshot = _write_mixed_snapshot(tmp_path / "correct")
    unrelated_snapshot = _write_mixed_snapshot(tmp_path / "unrelated")
    unrelated_jsonl = unrelated_snapshot / "a.jsonl"
    unrelated_jsonl.write_text(
        unrelated_jsonl.read_text(encoding="utf-8").replace('"one"', '"eno"'),
        encoding="utf-8",
    )
    registry = _write_registry(tmp_path / "registry.yaml")
    receipt = _write_transfer_receipt(
        tmp_path / "correct.transfer.json",
        snapshot=correct_snapshot,
        registry=registry,
    )
    output = tmp_path / "artifacts" / "audit.json"

    result = _invoke(
        unrelated_snapshot,
        registry,
        output,
        receipt=receipt,
    )

    assert result.exit_code == 2
    assert "receipt" in result.output.casefold() or "snapshot" in result.output.casefold()
    assert not output.exists()
    assert not output.with_name("audit.lineage.json").exists()


def test_directory_snapshot_outputs_must_be_outside_snapshot(tmp_path: Path) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = snapshot / "audit.json"

    result = _invoke(snapshot, registry, output)

    assert result.exit_code == 2
    assert "outside a directory snapshot" in result.output
    assert not output.with_name("audit.lineage.json").exists()


@pytest.mark.parametrize("dependency", ("snapshot", "registry"))
@pytest.mark.parametrize(
    ("publication_boundary", "expected_artifacts"),
    (
        pytest.param(1, (False, False, False), id="ledger"),
        pytest.param(2, (True, False, False), id="report"),
        pytest.param(3, (True, True, False), id="lineage"),
    ),
)
def test_dependency_drift_at_each_publication_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    publication_boundary: int,
    expected_artifacts: tuple[bool, bool, bool],
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    receipt = _write_transfer_receipt(
        tmp_path / "snapshot.transfer.json",
        snapshot=snapshot,
        registry=registry,
    )
    output = tmp_path / "artifacts" / "audit.json"
    ledger = output.with_name("audit.rejections.jsonl")
    lineage = output.with_name("audit.lineage.json")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    from nodelm.datasets import snapshot_audit as snapshot_audit_service

    original_writer = snapshot_audit_service.write_immutable_stream
    publication_count = 0

    def intercept_writer(
        path: Path,
        writer: Any,
        *,
        before_publish: Any = None,
    ) -> Any:
        nonlocal publication_count
        publication_count += 1
        current_boundary = publication_count

        def mutate_then_verify() -> object | None:
            if current_boundary == publication_boundary:
                if dependency == "registry":
                    registry.write_text(
                        registry.read_text(encoding="utf-8") + "# drift\n",
                        encoding="utf-8",
                    )
                else:
                    staged_snapshot = next(staging_root.glob(".nodelm-snapshot-audit-*"))
                    next(staged_snapshot.rglob("*.jsonl")).write_text(
                        "{}\n",
                        encoding="utf-8",
                    )
            return before_publish() if before_publish is not None else None

        return original_writer(path, writer, before_publish=mutate_then_verify)

    monkeypatch.setattr(snapshot_audit_service, "write_immutable_stream", intercept_writer)

    result = _invoke(
        snapshot,
        registry,
        output,
        "--staging-root",
        str(staging_root),
        receipt=receipt,
    )

    assert result.exit_code == 2
    assert "publication failed" in result.output
    assert publication_count == publication_boundary
    assert (ledger.exists(), output.exists(), lineage.exists()) == expected_artifacts


@pytest.mark.parametrize("dependency_action", ("delete-report", "mutate-ledger"))
def test_dependency_failure_before_fail_lineage_is_diagnostic_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_action: str,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml", observed_rows=3)
    receipt = _write_transfer_receipt(
        tmp_path / "snapshot.transfer.json",
        snapshot=snapshot,
        registry=registry,
    )
    output = tmp_path / "artifacts" / "audit.json"
    ledger = output.with_name("audit.rejections.jsonl")
    lineage = output.with_name("audit.lineage.json")

    from nodelm.datasets import snapshot_audit as snapshot_audit_service

    original_writer = snapshot_audit_service.write_immutable_stream
    publication_count = 0

    def intercept_writer(
        path: Path,
        writer: Any,
        *,
        before_publish: Any = None,
    ) -> Any:
        nonlocal publication_count
        publication_count += 1
        current_boundary = publication_count

        def break_dependency_then_verify() -> object | None:
            if current_boundary == 3:
                if dependency_action == "delete-report":
                    output.unlink()
                else:
                    ledger.write_text("tampered\n", encoding="utf-8")
            return before_publish() if before_publish is not None else None

        return original_writer(path, writer, before_publish=break_dependency_then_verify)

    monkeypatch.setattr(snapshot_audit_service, "write_immutable_stream", intercept_writer)

    result = _invoke(snapshot, registry, output, receipt=receipt)

    assert result.exit_code == 2
    assert result.exit_code != 1
    assert "publication failed" in result.output
    assert not lineage.exists()


def test_source_snapshot_aba_during_staged_iteration_cannot_change_audit_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    receipt_path = _write_transfer_receipt(
        tmp_path / "snapshot.transfer.json",
        snapshot=snapshot,
        registry=registry,
    )
    receipt = DatasetSnapshotTransferReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    output = tmp_path / "artifacts" / "audit.json"
    source_jsonl = snapshot / "a.jsonl"
    original_bytes = source_jsonl.read_bytes()

    from nodelm.datasets import snapshot_audit as snapshot_audit_service

    original_rows = snapshot_audit_service.iter_snapshot_rows
    streamed_paths: tuple[Path, ...] = ()

    def source_aba(paths: tuple[Path, ...]) -> Iterator[dict[str, Any]]:
        nonlocal streamed_paths
        streamed_paths = paths
        source_jsonl.write_text(
            json.dumps(
                {
                    "instance_id": "replacement",
                    "repo": "attacker/replacement",
                    "license": "MIT",
                    "language": "Rust",
                    "resolved": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            yield from original_rows(paths)
        finally:
            source_jsonl.write_bytes(original_bytes)

    monkeypatch.setattr(snapshot_audit_service, "iter_snapshot_rows", source_aba)

    result = _invoke(snapshot, registry, output, receipt=receipt_path)

    assert result.exit_code == 0, result.output
    assert source_jsonl.read_bytes() == original_bytes
    assert streamed_paths
    assert all(not path.is_relative_to(snapshot) for path in streamed_paths)
    report = json.loads(output.read_text(encoding="utf-8"))
    lineage = DatasetLineageManifest.model_validate_json(
        output.with_name("audit.lineage.json").read_text(encoding="utf-8")
    )
    assert report["language_distribution"] == {"JavaScript": 1, "TypeScript": 1}
    assert "Rust" not in report["language_distribution"]
    assert lineage.snapshot == receipt.snapshot


def test_artifact_collision_aborts_without_lineage(tmp_path: Path) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = tmp_path / "artifacts" / "audit.json"
    output.parent.mkdir()
    output.with_name("audit.rejections.jsonl").write_text("different\n", encoding="utf-8")

    result = _invoke(snapshot, registry, output)

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert not output.with_name("audit.lineage.json").exists()
