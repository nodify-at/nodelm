from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from typer.testing import CliRunner

from nodelm.cli import app
from nodelm.datasets.lineage import DatasetLineageManifest


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


def _invoke(
    snapshot: Path,
    registry: Path,
    output: Path,
    *extra: str,
) -> object:
    return CliRunner().invoke(
        app,
        [
            "datasets",
            "audit-snapshot",
            "--source",
            "fixture",
            "--snapshot",
            str(snapshot),
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
    audit = json.loads(output.read_text(encoding="utf-8"))
    ledger = output.with_name("audit.rejections.jsonl")
    lineage_path = output.with_name("audit.lineage.json")
    lineage = DatasetLineageManifest.model_validate_json(lineage_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["input_scope"] == "complete-snapshot"
    assert audit["row_count"] == 2
    assert audit["language_distribution"] == {"JavaScript": 1, "TypeScript": 1}
    assert json.loads(ledger.read_text(encoding="utf-8"))["disposition"] == "REJECT"
    assert lineage.status.value == "PASS"
    assert lineage.snapshot.snapshot_sha256 == audit["input_sha256"]
    assert lineage.snapshot.snapshot_bytes == audit["input_bytes"]
    assert lineage.rejection_ledger_rows == 1
    assert lineage.source.revision == "a" * 40


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


def test_directory_snapshot_outputs_must_be_outside_snapshot(tmp_path: Path) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = snapshot / "audit.json"

    result = _invoke(snapshot, registry, output)

    assert result.exit_code == 2
    assert "outside a directory snapshot" in result.output
    assert not output.with_name("audit.lineage.json").exists()


def test_snapshot_drift_aborts_before_report_and_lineage_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _write_mixed_snapshot(tmp_path / "snapshot")
    registry = _write_registry(tmp_path / "registry.yaml")
    output = tmp_path / "artifacts" / "audit.json"

    from nodelm import cli

    original_verify = cli.verify_snapshot_identity
    verification_count = 0

    def mutate_before_second_publication(path: Path, expected: object) -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            (snapshot / "a.jsonl").write_text("{}\n", encoding="utf-8")
        original_verify(path, expected)

    monkeypatch.setattr(cli, "verify_snapshot_identity", mutate_before_second_publication)

    result = _invoke(snapshot, registry, output)

    assert result.exit_code == 2
    assert "snapshot changed" in result.output
    assert not output.exists()
    assert not output.with_name("audit.lineage.json").exists()


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
