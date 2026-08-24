from __future__ import annotations

from pathlib import Path

import pytest

from nodelm.artifacts import content_digest
from nodelm.datasets import snapshot_transfer as snapshot_transfer_service
from nodelm.datasets.snapshot_transfer import transfer_snapshot
from nodelm.models import DatasetSource


def _registry_payload(*, repository_id: str, revision: str) -> bytes:
    return (
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture\n"
        f"    repository_id: {repository_id}\n"
        f"    revision: {revision}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        "    observed_rows: 1\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n"
    ).encode()


def test_snapshot_transfer_parses_the_same_registry_bytes_it_identifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry.yaml"
    original_payload = _registry_payload(
        repository_id="owner/original",
        revision="a" * 40,
    )
    replacement_payload = _registry_payload(
        repository_id="attacker/replacement",
        revision="b" * 40,
    )
    registry.write_bytes(original_payload)
    destination = tmp_path / "snapshot"
    receipt_output = tmp_path / "snapshot.transfer.json"
    downloaded_source: DatasetSource | None = None

    original_file_identity = snapshot_transfer_service.file_identity
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

    def downloader(
        source: DatasetSource,
        *,
        destination: Path,
        allow_patterns: tuple[str, ...],
    ) -> Path:
        nonlocal downloaded_source
        downloaded_source = source
        assert allow_patterns == ()
        destination.mkdir()
        (destination / "data.jsonl").write_text('{"instance_id":"one"}\n', encoding="utf-8")
        return destination

    monkeypatch.setattr(snapshot_transfer_service, "file_identity", tracked_file_identity)
    monkeypatch.setattr(Path, "read_bytes", registry_aba)

    result = transfer_snapshot(
        source_name="fixture",
        destination=destination,
        config=registry,
        downloader=downloader,
        receipt_output=receipt_output,
    )

    assert registry_reads == 1
    assert downloaded_source is not None
    assert downloaded_source.repository_id == "owner/original"
    assert result.receipt.source == downloaded_source
    assert result.receipt.registry_sha256 == content_digest(original_payload)
    assert result.receipt.registry_bytes == len(original_payload)
