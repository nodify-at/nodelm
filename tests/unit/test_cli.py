from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

import nodelm.cli as cli_module
from nodelm.artifacts import canonical_json_bytes, file_identity
from nodelm.cli import app
from nodelm.datasets.lineage import (
    DatasetSnapshotTransferReceipt,
    capture_snapshot_identity,
)
from nodelm.datasets.pilot import AUTHORIZED_PILOT_MANIFEST_SHA256_BY_SAMPLES_SHA256
from nodelm.decontamination.split import AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256
from nodelm.evaluation.fixture import MODEL_TASK_FIXTURE_IDENTITY
from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.models import NormalizedSample
from nodelm.provenance.gold import AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256


def _write_pilot_safety_evidence(
    tmp_path: Path,
    input_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_name: str = "fixture",
    source_revision: str = "a" * 40,
    partition_name: str = "fixture/model/tasks",
    sample_count: int = 1,
    uniqueness_scope: str = "complete-partition",
) -> tuple[Path, Path]:
    input_identity = file_identity(input_path)
    normalization_manifest = tmp_path / "normalized.manifest.json"
    normalization_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.normalization-manifest/v2",
                "status": "PASS",
                "source_name": source_name,
                "source_repository_id": f"owner/{source_name}",
                "source_revision": source_revision,
                "partition_name": partition_name,
                "harness": "fixture",
                "generating_model": "fixture@revision",
                "upstream_source": "tasks",
                "row_dataset_name": "owner/fixture-tasks",
                "input_sha256": "1" * 64,
                "input_bytes": 1,
                "registry_sha256": "2" * 64,
                "materialization_manifest_sha256": "3" * 64,
                "materialization_manifest_bytes": 1,
                "partition_contract_sha256": "4" * 64,
                "partition_contract_bytes": 1,
                "transfer_receipt_sha256": "5" * 64,
                "transfer_receipt_bytes": 1,
                "task_provenance_sha256": "6" * 64,
                "task_provenance_bytes": 1,
                "task_provenance_manifest_sha256": "7" * 64,
                "task_provenance_manifest_bytes": 1,
                "task_transfer_receipt_sha256": "8" * 64,
                "task_transfer_receipt_bytes": 1,
                "task_source_name": "fixture-tasks",
                "task_source_revision": "c" * 40,
                "materialization_replay": "PASS",
                "task_provenance_replay": "PASS",
                "uniqueness_scope": uniqueness_scope,
                "gold_exposure_audit": "NOT RUN",
                "input_row_count": sample_count,
                "accepted_count": sample_count,
                "rejected_count": 0,
                "rejection_counts_by_code": {},
                "unique_rollout_key_count": sample_count,
                "duplicate_trace_row_count": 0,
                "conflicting_rollout_identity_count": 0,
                "conflicting_rollout_row_count": 0,
                "normalized_artifact": input_path.name,
                "normalized_sha256": input_identity[0],
                "normalized_bytes": input_identity[1],
                "rejection_artifact": "normalized.rejections.jsonl",
                "rejection_sha256": "9" * 64,
                "rejection_bytes": 0,
            }
        )
    )
    findings = tmp_path / "gold.findings.jsonl"
    findings.write_bytes(b"")
    attestation = tmp_path / "oracle-isolation.json"
    attestation.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.oracle-isolation-attestation/v1",
                "method_version": "nodelm.oracle-isolation-review/v1",
                "status": "PASS",
                "source_name": source_name,
                "source_revision": source_revision,
                "partition_name": partition_name,
                "normalized_sha256": input_identity[0],
                "normalized_bytes": input_identity[1],
                "covered_sample_count": sample_count,
            }
        )
    )
    normalization_identity = file_identity(normalization_manifest)
    findings_identity = file_identity(findings)
    attestation_identity = file_identity(attestation)
    audit = tmp_path / "gold-exposure.audit.json"
    audit.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.gold-exposure-audit/v1",
                "method_version": "nodelm.gold-exposure-audit-method/v1",
                "status": "PASS",
                "normalization_manifest_artifact": normalization_manifest.name,
                "normalization_manifest_sha256": normalization_identity[0],
                "normalization_manifest_bytes": normalization_identity[1],
                "normalized_artifact": input_path.name,
                "normalized_sha256": input_identity[0],
                "normalized_bytes": input_identity[1],
                "expected_sample_count": sample_count,
                "audited_sample_count": sample_count,
                "structural_scan": {"status": "PASS", "finding_count": 0},
                "oracle_isolation": {
                    "status": "PASS",
                    "attestation_artifact": attestation.name,
                    "attestation_sha256": attestation_identity[0],
                    "attestation_bytes": attestation_identity[1],
                    "covered_sample_count": sample_count,
                },
                "findings_artifact": findings.name,
                "findings_sha256": findings_identity[0],
                "findings_bytes": findings_identity[1],
            }
        )
    )
    monkeypatch.setitem(
        AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256,
        input_identity[0],
        file_identity(audit)[0],
    )
    return normalization_manifest, audit


def _split_sample(identifier: str, repository: str) -> NormalizedSample:
    return NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="a" * 40,
        repository=repository,
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id=identifier,
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id=f"rollout-{identifier}",
        resolved=True,
        trajectory=({"role": "assistant", "content": f"repair {identifier}"},),
        generated_patch=f"diff --git a/{identifier}.ts b/{identifier}.ts\n+repair();\n",
        patch_metadata={"bytes": 1},
        provenance_lineage=(f"raw:{identifier}",),
    )


def _write_split_gate_inputs(
    tmp_path: Path,
    samples: tuple[NormalizedSample, ...],
) -> tuple[Path, Path, Path]:
    input_path = tmp_path / "samples.jsonl"
    input_path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples),
        encoding="utf-8",
    )
    task_metadata_path = tmp_path / "tasks.jsonl"
    task_metadata_path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": sample.issue_or_pr_id,
                    "repo": sample.repository,
                    "problem_statement": f"task statement for {sample.issue_or_pr_id}",
                    "patch": f"reference patch for {sample.issue_or_pr_id}",
                }
            )
            + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "benchmark_id": "public-one",
                "task": "an unrelated public benchmark task",
                "patch": "an unrelated public benchmark patch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return input_path, task_metadata_path, benchmark_path


def test_dataset_registry_validate_command_reports_verified_sources() -> None:
    result = CliRunner().invoke(
        app,
        ["datasets", "validate", "--config", "configs/datasets/registry.yaml", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert '"source_count":3' in result.output
    assert '"status":"PASS"' in result.output


def test_split_command_writes_deterministic_manifest(tmp_path: Path) -> None:
    input_path, task_metadata_path, benchmark_path = _write_split_gate_inputs(
        tmp_path,
        (
            _split_sample("one", "acme/widget"),
            _split_sample("two", "acme/widget"),
        ),
    )
    output_path = tmp_path / "split.json"

    result = CliRunner().invoke(
        app,
        [
            "split",
            "build",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--task-metadata",
            str(task_metadata_path),
            "--benchmark",
            str(benchmark_path),
            "--near-duplicate-threshold",
            "0.85",
            "--seed",
            "3",
            "--evaluation-fraction",
            "0.2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "nodelm.repository-split/v1"
    assert payload["sample_count"] == 2
    assert len(payload["input_sha256"]) == 64
    assert payload["decontamination"]["near_duplicate_threshold"] == 0.85
    assert len(payload["decontamination"]["task_metadata_sha256"]) == 64
    assert len(payload["decontamination"]["benchmark_sha256"]) == 64


def test_split_command_applies_versioned_repository_aliases(tmp_path: Path) -> None:
    input_path, task_metadata_path, benchmark_path = _write_split_gate_inputs(
        tmp_path,
        (
            _split_sample("one", "acme/widget"),
            _split_sample("two", "mirror/widget"),
        ),
    )
    aliases_path = tmp_path / "aliases.yaml"
    aliases_path.write_text(
        "schema_version: nodelm.repository-aliases/v1\naliases:\n  mirror/widget: acme/widget\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "split.json"

    result = CliRunner().invoke(
        app,
        [
            "split",
            "build",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--task-metadata",
            str(task_metadata_path),
            "--benchmark",
            str(benchmark_path),
            "--near-duplicate-threshold",
            "0.85",
            "--aliases",
            str(aliases_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len({item["split"] for item in payload["assignments"]}) == 1
    assert len(payload["aliases_sha256"]) == 64


def test_doctor_and_fixture_harness_commands_run_real_local_checks() -> None:
    doctor = CliRunner().invoke(app, ["doctor", "--json"])
    harness = CliRunner().invoke(app, ["harness", "verify", "--json"])

    assert doctor.exit_code == 0, doctor.output
    assert '"status":"PASS"' in doctor.output
    assert harness.exit_code == 0, harness.output
    assert '"outcome":"success"' in harness.output
    payload = json.loads(harness.output)
    assert payload["command"]["schema_version"] == "nodelm.command-result/v1"
    assert len(payload["config_sha256"]) == 64


def test_harness_verify_fails_closed_for_an_empty_workspace(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["harness", "verify", "--workspace", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    assert '"status":"FAIL"' in result.output
    assert "no package.json" in result.output


def test_trusted_local_harness_rejects_a_false_network_isolation_claim(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "harness.yaml"
    config.write_text(
        "schema_version: nodelm.harness-config/v1\n"
        "backend: trusted-local\n"
        "status: UNVERIFIED\n"
        "timeout_seconds: 10\n"
        "max_output_bytes: 1024\n"
        "network_enabled: false\n"
        "network_isolation_enforced: false\n"
        "dependency_install:\n"
        "  enabled: false\n"
        "  ignore_scripts: true\n"
        "allowed_tools: [node]\n"
        "security_note: trusted fixture only\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "harness",
            "verify",
            "--workspace",
            str(workspace),
            "--config",
            str(config),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "trusted-local cannot enforce network isolation" in result.output


def test_offline_dataset_audit_command_writes_rejections(tmp_path: Path) -> None:
    config = tmp_path / "registry.yaml"
    config.write_text(
        """
schema_version: nodelm.dataset-registry/v1
sources:
  - name: fixture
    repository_id: owner/fixture
    status: UNVERIFIED
    observed_rows: 1
""".lstrip(),
        encoding="utf-8",
    )
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        '{"instance_id":"one","repo":"acme/widget","license":"GPL-3.0"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "audit",
            "--source",
            "fixture",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--config",
            str(config),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rejected_rows"][0]["disposition"] == "REJECT"
    assert payload["status"] == "UNVERIFIED"
    assert len(payload["input_sha256"]) == 64
    assert payload["input_bytes"] == input_path.stat().st_size
    ledger = tmp_path / "audit.rejections.jsonl"
    assert ledger.exists()
    assert json.loads(ledger.read_text(encoding="utf-8"))["disposition"] == "REJECT"
    assert payload["rejection_ledger_rows"] == 1
    assert len(payload["rejection_ledger_sha256"]) == 64


def test_pilot_command_validates_and_writes_normalized_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="a" * 40,
        repository="acme/widget",
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id="one",
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id="rollout-one",
        resolved=True,
        trajectory=({"role": "assistant", "content": "inspect and patch"},),
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 1},
        provenance_lineage=("raw:one",),
    )
    input_path = tmp_path / "normalized.jsonl"
    input_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    input_identity = file_identity(input_path)
    normalization_manifest, gold_audit = _write_pilot_safety_evidence(
        tmp_path,
        input_path,
        monkeypatch,
        uniqueness_scope="canary",
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nodelm.repository-split/v1",
                "repositories": {
                    "train": ["github.com/acme/widget"],
                    "evaluation": [],
                },
                "input_sha256": input_identity[0],
                "input_bytes": input_identity[1],
                "sample_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256,
        input_identity[0],
        file_identity(split_manifest)[0],
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture\n"
        "    repository_id: owner/fixture\n"
        f"    revision: {'a' * 40}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        "    observed_rows: 1\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n",
        encoding="utf-8",
    )
    output = tmp_path / "pilot.json"

    canary_result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--normalization-manifest",
            str(normalization_manifest),
            "--gold-exposure-audit",
            str(gold_audit),
            "--split-manifest",
            str(split_manifest),
            "--registry",
            str(registry),
        ],
    )

    assert canary_result.exit_code == 2
    assert "complete-partition" in canary_result.output
    assert not output.exists()

    normalization_manifest, gold_audit = _write_pilot_safety_evidence(
        tmp_path,
        input_path,
        monkeypatch,
    )

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--normalization-manifest",
            str(normalization_manifest),
            "--gold-exposure-audit",
            str(gold_audit),
            "--split-manifest",
            str(split_manifest),
            "--registry",
            str(registry),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 1
    assert payload["status"] == "UNVERIFIED"
    samples = tmp_path / "pilot.samples.jsonl"
    assert samples.exists()
    assert json.loads(samples.read_text(encoding="utf-8"))["trajectory"]
    assert all(
        len(payload[field]) == 64
        for field in (
            "input_sha256",
            "policy_sha256",
            "registry_sha256",
            "split_manifest_sha256",
            "normalization_manifest_sha256",
            "gold_exposure_audit_sha256",
        )
    )
    assert payload["gold_exposure_audit"] == "PASS"


def test_pilot_rechecks_training_surface_despite_authorized_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = NormalizedSample(
        source_dataset="open-swe-traces",
        source_dataset_revision="ed95cef24df8d8bd79b4ceb0192cb420fde06521",
        repository="acme/widget",
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id="one",
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id="rollout-one",
        resolved=True,
        trajectory=({"reference": {"patch": "SECRET_GOLD"}},),
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 1},
        provenance_lineage=("raw:one",),
    )
    input_path = tmp_path / "normalized.jsonl"
    input_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    input_identity = file_identity(input_path)
    normalization_manifest, gold_audit = _write_pilot_safety_evidence(
        tmp_path,
        input_path,
        monkeypatch,
        source_name="open-swe-traces",
        source_revision="ed95cef24df8d8bd79b4ceb0192cb420fde06521",
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "nodelm.repository-split/v1",
                "repositories": {
                    "train": ["github.com/acme/widget"],
                    "evaluation": [],
                },
                "input_sha256": input_identity[0],
                "input_bytes": input_identity[1],
                "sample_count": 1,
            }
        )
    )
    monkeypatch.setitem(
        AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256,
        input_identity[0],
        file_identity(split_manifest)[0],
    )
    output = tmp_path / "pilot.json"

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--normalization-manifest",
            str(normalization_manifest),
            "--gold-exposure-audit",
            str(gold_audit),
            "--split-manifest",
            str(split_manifest),
        ],
    )

    assert result.exit_code == 2
    assert "forbidden gold/reference patch" in result.output
    assert not output.exists()


def test_large_download_requires_explicit_confirmation(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "download",
            "--source",
            "open-swe-traces",
            "--destination",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 2
    assert "confirm-large-download" in result.output


def test_large_download_rejects_nonempty_destination_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "data"
    destination.mkdir()
    marker = destination / "partial-snapshot"
    marker.write_text("preserve me\n", encoding="utf-8")
    download_called = False

    def fake_download(*args: object, **kwargs: object) -> Path:
        nonlocal download_called
        download_called = True
        return destination

    monkeypatch.setattr("nodelm.cli.download_pinned_snapshot", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "download",
            "--source",
            "open-swe-traces",
            "--destination",
            str(destination),
            "--confirm-large-download",
        ],
    )

    assert result.exit_code == 2
    assert "destination must be new or empty" in result.output
    assert download_called is False
    assert marker.read_text(encoding="utf-8") == "preserve me\n"


@pytest.mark.parametrize("destination_exists", (False, True))
def test_large_download_allows_new_or_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_exists: bool,
) -> None:
    destination = tmp_path / "data"
    receipt_output = tmp_path / "data.transfer.json"
    if destination_exists:
        destination.mkdir()
    download_called = False

    def fake_download(*args: object, **kwargs: object) -> Path:
        nonlocal download_called
        download_called = True
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "snapshot.jsonl").write_text(
            '{"instance_id":"synthetic"}\n',
            encoding="utf-8",
        )
        return destination

    monkeypatch.setattr("nodelm.cli.download_pinned_snapshot", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "download",
            "--source",
            "open-swe-traces",
            "--destination",
            str(destination),
            "--confirm-large-download",
            "--receipt-output",
            str(receipt_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert download_called is True
    receipt = DatasetSnapshotTransferReceipt.model_validate_json(
        receipt_output.read_text(encoding="utf-8")
    )
    assert receipt.snapshot_scope == "complete"
    assert receipt.allow_patterns == ()
    assert receipt.snapshot == capture_snapshot_identity(destination)
    assert receipt.source.name == "open-swe-traces"
    assert canonical_json_bytes(receipt.model_dump(mode="json")) == receipt_output.read_bytes()


def test_filtered_download_receipt_records_exact_requested_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "data"
    receipt_output = tmp_path / "filtered.transfer.json"
    observed_patterns: tuple[str, ...] | None = None

    def fake_download(*args: object, **kwargs: object) -> Path:
        nonlocal observed_patterns
        patterns = kwargs["allow_patterns"]
        assert isinstance(patterns, tuple)
        assert all(isinstance(pattern, str) for pattern in patterns)
        observed_patterns = patterns
        destination.mkdir()
        (destination / "snapshot.jsonl").write_text("{}\n", encoding="utf-8")
        return destination

    monkeypatch.setattr("nodelm.cli.download_pinned_snapshot", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "download",
            "--source",
            "open-swe-traces",
            "--destination",
            str(destination),
            "--allow-pattern",
            "**/*.jsonl",
            "--confirm-large-download",
            "--receipt-output",
            str(receipt_output),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = DatasetSnapshotTransferReceipt.model_validate_json(
        receipt_output.read_text(encoding="utf-8")
    )
    assert observed_patterns == ("**/*.jsonl",)
    assert receipt.snapshot_scope == "filtered"
    assert receipt.allow_patterns == ("**/*.jsonl",)
    assert receipt.snapshot == capture_snapshot_identity(destination)


def test_unresolved_model_and_training_commands_report_honest_statuses() -> None:
    model = CliRunner().invoke(app, ["models", "smoke", "--json"])
    training = CliRunner().invoke(app, ["training", "smoke", "--dry-run", "--json"])
    blocked_training = CliRunner().invoke(app, ["training", "smoke", "--json"])

    assert model.exit_code == 0
    model_payload = json.loads(model.output)
    assert model_payload["metadata_status"] == "PASS"
    assert model_payload["status"] == "NOT RUN"
    assert model_payload["bakeoff_status"] == "NOT RUN"
    assert model_payload["candidate_count"] == 3
    assert model_payload["selected_candidate"] is None
    assert training.exit_code == 0
    assert '"status":"NOT RUN"' in training.output
    assert blocked_training.exit_code == 2
    assert '"status":"BLOCKED"' in blocked_training.output
    assert "same-harness candidate bake-off" in blocked_training.output


def test_training_dry_run_rejects_an_arbitrary_mapping(tmp_path: Path) -> None:
    config = tmp_path / "not-training.yaml"
    config.write_text("hello: world\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["training", "smoke", "--config", str(config), "--dry-run", "--json"],
    )

    assert result.exit_code == 2
    assert "invalid training configuration" in result.output


def test_real_training_lifecycle_blocks_before_importing_a_model_without_pins(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "training",
            "run-lifecycle",
            "--samples",
            str(samples),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "report.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert '"status":"BLOCKED"' in result.output


def test_training_lifecycle_command_requires_model_authored_patch_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def __init__(self, settings: object) -> None:
            self.settings = settings

        def load(self, config: object) -> None:
            pass

        def tokenize(self, samples: tuple[str, ...]) -> None:
            assert samples

        def train_step(self) -> None:
            pass

        def save_checkpoint(self, path: Path) -> None:
            path.mkdir()

        def reload_checkpoint(self, path: Path) -> None:
            assert path.is_dir()

        def resume_train_step(self) -> None:
            pass

        def infer(self, prompt: str) -> str:
            assert "multiply" in prompt
            return (
                "diff --git a/src/math.js b/src/math.js\n"
                "--- a/src/math.js\n"
                "+++ b/src/math.js\n"
                "@@ -4,5 +4,5 @@ export function add(left, right) {\n"
                " \n"
                " export function multiply(left, right) {\n"
                "   // Deliberate model-verification task: repair this implementation.\n"
                "-  return left + right;\n"
                "+  return left * right;\n"
                " }\n"
            )

        def measured_evidence(self) -> dict[str, object]:
            return {
                "optimizer_steps": 2,
                "resumed_optimizer_steps": 1,
                "optimizer_state_reloaded": True,
            }

    class FakeSandbox:
        def __init__(self, image: str) -> None:
            self.image = image

        def run_node_tests(self, workspace: Path) -> CommandResult:
            repaired = "return left * right;" in (workspace / "src/math.js").read_text(
                encoding="utf-8"
            )
            return CommandResult(
                argv=("fake-sandbox", "node", "--test"),
                cwd=workspace,
                outcome=(OutcomeCategory.SUCCESS if repaired else OutcomeCategory.TEST_FAILURE),
                exit_code=0 if repaired else 1,
                stdout="# tests 2\n",
                stderr="",
                duration_seconds=0.01,
            )

        def evidence(self) -> dict[str, object]:
            return {"backend": "fake-isolated", "image": self.image}

    monkeypatch.setattr("nodelm.cli.TransformersSmokeBackend", FakeBackend)
    monkeypatch.setattr("nodelm.cli.PodmanFixtureSandbox", FakeSandbox)
    config = tmp_path / "training.yaml"
    config.write_text(
        "schema_version: nodelm.training-config/v1\n"
        "status: UNVERIFIED\n"
        "purpose: fixture lifecycle\n"
        "model:\n"
        "  repository_id: owner/model\n"
        f"  revision: {'a' * 40}\n"
        "  license: Apache-2.0\n"
        "runtime:\n"
        "  backend: transformers-peft\n"
        "  device: cpu\n"
        "  padding_policy: require-existing\n"
        "  added_padding_token: null\n"
        "  max_length: 128\n"
        "  max_new_tokens: 128\n"
        "  use_lora: false\n"
        "  target_modules: []\n"
        "  lora_rank: 8\n"
        "  lora_alpha: 16\n"
        "  lora_dropout: 0.0\n"
        "precision: float32\n"
        "seed: 42\n"
        "max_steps: 1\n"
        "batch_size: 1\n"
        "gradient_accumulation_steps: 1\n"
        "learning_rate: 0.00002\n"
        "checkpoint: {save: true, reload: true, resume: true}\n"
        "inference_after_reload: true\n"
        "reason: fixture\n",
        encoding="utf-8",
    )
    sample = NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="b" * 40,
        repository="acme/widget",
        repository_license="MIT",
        base_commit="c" * 40,
        issue_or_pr_id="one",
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id="rollout-one",
        resolved=True,
        trajectory=({"role": "assistant", "content": "repair"},),
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 1},
        provenance_lineage=("raw:one",),
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    samples_sha256 = hashlib.sha256(samples.read_bytes()).hexdigest()
    pilot_manifest = tmp_path / "pilot.json"
    pilot_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nodelm.pilot-subset/v1",
                "status": "UNVERIFIED",
                "accepted_count": 1,
                "samples_artifact": samples.name,
                "samples_sha256": samples_sha256,
                "samples_bytes": samples.stat().st_size,
                "gold_exposure_audit": "PASS",
                "normalization_manifest_sha256": "d" * 64,
                "gold_exposure_audit_sha256": "e" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        AUTHORIZED_PILOT_MANIFEST_SHA256_BY_SAMPLES_SHA256,
        samples_sha256,
        file_identity(pilot_manifest)[0],
    )
    report = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "training",
            "run-lifecycle",
            "--config",
            str(config),
            "--samples",
            str(samples),
            "--pilot-manifest",
            str(pilot_manifest),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--output",
            str(report),
            "--sandbox-image",
            f"docker.io/library/node@sha256:{'d' * 64}",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["lifecycle"]["status"] == "PASS"
    assert payload["lifecycle"]["resumed_optimizer_step_completed"] is True
    assert payload["checkpoint_resume_required"] is True
    assert payload["measurements"]["resumed_optimizer_steps"] == 1
    assert payload["fixture_evaluation"]["status"] == "PASS"
    assert payload["evaluation_fixture_sha256"] == MODEL_TASK_FIXTURE_IDENTITY.tree_sha256
    assert payload["evaluation_fixture_file_count"] == MODEL_TASK_FIXTURE_IDENTITY.file_count

    altered_fixture = tmp_path / "altered-fixture"
    shutil.copytree(Path("tests/fixtures/model-task"), altered_fixture)
    (altered_fixture / "test" / "math.test.js").write_text(
        "import test from 'node:test';\ntest('untrusted', () => {});\n",
        encoding="utf-8",
    )
    altered_result = CliRunner().invoke(
        app,
        [
            "training",
            "run-lifecycle",
            "--config",
            str(config),
            "--samples",
            str(samples),
            "--pilot-manifest",
            str(pilot_manifest),
            "--checkpoint-dir",
            str(tmp_path / "altered-checkpoint"),
            "--evaluation-workspace",
            str(altered_fixture),
            "--output",
            str(tmp_path / "altered-report.json"),
            "--sandbox-image",
            f"docker.io/library/node@sha256:{'d' * 64}",
            "--json",
        ],
    )

    assert altered_result.exit_code == 2
    assert "evaluation fixture" in altered_result.output
    assert not (tmp_path / "altered-checkpoint").exists()

    original_writer = cli_module.write_immutable_json

    def mutate_on_publish(path, value, *, before_publish=None):
        def mutate_then_verify() -> object:
            samples.write_bytes(samples.read_bytes() + b" ")
            return before_publish() if before_publish is not None else None

        return original_writer(path, value, before_publish=mutate_then_verify)

    monkeypatch.setattr(cli_module, "write_immutable_json", mutate_on_publish)
    changed_report = tmp_path / "changed-report.json"
    changed_result = CliRunner().invoke(
        app,
        [
            "training",
            "run-lifecycle",
            "--config",
            str(config),
            "--samples",
            str(samples),
            "--pilot-manifest",
            str(pilot_manifest),
            "--checkpoint-dir",
            str(tmp_path / "changed-checkpoint"),
            "--output",
            str(changed_report),
            "--sandbox-image",
            f"docker.io/library/node@sha256:{'d' * 64}",
            "--json",
        ],
    )

    assert changed_result.exit_code == 2
    assert "input changed while it was being processed" in changed_result.output
    assert not changed_report.exists()


def test_infrastructure_doctor_emits_machine_readable_report(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["infra", "doctor", "--workspace", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "nodelm.infrastructure/v2"
    assert payload["gpu_count"] >= 0
    assert payload["total_gpu_memory_bytes"] >= 0
    assert payload["cuda_runtime"]["status"] in {"PASS", "FAIL", "NOT RUN"}
    assert payload["cuda_toolkit"]["status"] in {"PASS", "FAIL", "NOT RUN"}


def test_harness_rejects_a_successful_node_run_with_zero_tests(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{}\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["harness", "verify", "--workspace", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "FAIL"
    assert payload["outcome"] == "success"
    assert payload["test_count"] == 0
    assert "without evidence" in payload["reason"]


def test_invalid_normalized_pilot_input_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"repository":"acme/widget"}\n', encoding="utf-8")
    input_identity = file_identity(input_path)
    normalization_manifest, gold_audit = _write_pilot_safety_evidence(
        tmp_path,
        input_path,
        monkeypatch,
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nodelm.repository-split/v1",
                "repositories": {"train": [], "evaluation": []},
                "input_sha256": input_identity[0],
                "input_bytes": input_identity[1],
                "sample_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        AUTHORIZED_SPLIT_SHA256_BY_NORMALIZED_SHA256,
        input_identity[0],
        file_identity(split_manifest)[0],
    )

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "pilot.json"),
            "--normalization-manifest",
            str(normalization_manifest),
            "--gold-exposure-audit",
            str(gold_audit),
            "--split-manifest",
            str(split_manifest),
        ],
    )

    assert result.exit_code == 2
    assert "invalid normalized pilot input" in result.output


def test_dataset_registry_validate_preserves_explicit_fail(tmp_path: Path) -> None:
    config = tmp_path / "registry.yaml"
    config.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: failed\n"
        "    repository_id: owner/failed\n"
        "    status: FAIL\n"
        "  - name: pending\n"
        "    repository_id: owner/pending\n"
        "    status: UNVERIFIED\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["datasets", "validate", "--config", str(config), "--json"],
    )

    assert result.exit_code == 1
    assert '"status":"FAIL"' in result.output


def _gold_audit_sample(
    *,
    trajectory: tuple[dict[str, object], ...] = (
        {"role": "assistant", "content": "inspect and patch"},
    ),
) -> NormalizedSample:
    return NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="a" * 40,
        repository="acme/widget",
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id="one",
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id="rollout-one",
        resolved=True,
        trajectory=trajectory,
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 1},
        provenance_lineage=("raw:one",),
    )


def _gold_audit_command_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sample: NormalizedSample | None = None,
    sample_count: int = 1,
    uniqueness_scope: str = "complete-partition",
) -> tuple[Path, Path, Path, Path, Path]:
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(
        (sample or _gold_audit_sample()).model_dump_json() + "\n",
        encoding="utf-8",
    )
    manifest, _ = _write_pilot_safety_evidence(
        tmp_path,
        normalized,
        monkeypatch,
        sample_count=sample_count,
        uniqueness_scope=uniqueness_scope,
    )
    return (
        normalized,
        manifest,
        tmp_path / "oracle-isolation.json",
        tmp_path / "produced-gold.audit.json",
        tmp_path / "produced-gold.findings.jsonl",
    )


def _invoke_gold_audit(
    normalized: Path,
    manifest: Path,
    audit: Path,
    findings: Path,
    *,
    attestation: Path | None = None,
):
    arguments = [
        "datasets",
        "audit-gold-exposure",
        "--input",
        str(normalized),
        "--normalization-manifest",
        str(manifest),
        "--output",
        str(audit),
        "--findings-output",
        str(findings),
    ]
    if attestation is not None:
        arguments.extend(("--oracle-isolation-attestation", str(attestation)))
    return CliRunner().invoke(app, arguments)


def _replace_gold_audit_input(
    normalized: Path,
    manifest: Path,
    raw_rows: bytes,
) -> None:
    normalized.write_bytes(raw_rows)
    normalized_identity = file_identity(normalized)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "normalized_sha256": normalized_identity[0],
            "normalized_bytes": normalized_identity[1],
        }
    )
    manifest.write_bytes(canonical_json_bytes(manifest_payload))


def test_gold_audit_command_passes_complete_population_with_reviewed_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["expected_sample_count"] == payload["audited_sample_count"] == 1
    assert payload["structural_scan"] == {"status": "PASS", "finding_count": 0}
    assert payload["oracle_isolation"]["status"] == "PASS"
    assert payload["findings_artifact"] == findings.name
    assert findings.read_bytes() == b""


def test_gold_audit_command_blocks_without_oracle_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["structural_scan"] == {"status": "PASS", "finding_count": 0}
    assert payload["oracle_isolation"] == {
        "status": "BLOCKED",
        "attestation_artifact": None,
        "attestation_sha256": None,
        "attestation_bytes": None,
        "covered_sample_count": 0,
    }


def test_gold_audit_command_blocks_canary_even_with_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
        uniqueness_scope="canary",
    )

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["oracle_isolation"]["status"] == "PASS"


@pytest.mark.parametrize(
    "trajectory",
    [
        ({"reference": {"patch": "TOP_SECRET_REFERENCE_PATCH"}},),
        ({"golden_patch": "TOP_SECRET_REFERENCE_PATCH"},),
        ({"payload": {"golden_patch": "TOP_SECRET_REFERENCE_PATCH"}},),
    ],
)
def test_gold_audit_command_fails_safely_on_forbidden_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trajectory: tuple[dict[str, object], ...],
) -> None:
    secret = "TOP_SECRET_REFERENCE_PATCH"
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
        sample=_gold_audit_sample(trajectory=trajectory),
    )

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    finding = json.loads(findings.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["structural_scan"] == {"status": "FAIL", "finding_count": 1}
    assert finding["reason_code"] == "forbidden_gold_reference_patch"
    assert set(finding) == {"row_index", "sample_id", "reason_code", "reason"}
    assert secret not in result.output
    assert secret not in audit.read_text(encoding="utf-8")
    assert secret not in findings.read_text(encoding="utf-8")


def test_gold_audit_command_sanitizes_invalid_sample_validation_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    secret = "RAW_VALIDATION_SECRET"
    sample_payload = json.loads(normalized.read_text(encoding="utf-8"))
    sample_payload["unexpected_gold_field"] = secret
    normalized.write_bytes(canonical_json_bytes(sample_payload))
    normalized_identity = file_identity(normalized)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "normalized_sha256": normalized_identity[0],
            "normalized_bytes": normalized_identity[1],
        }
    )
    manifest.write_bytes(canonical_json_bytes(manifest_payload))

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    finding = json.loads(findings.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert finding["reason_code"] == "invalid_normalized_sample"
    assert finding["sample_id"] is None
    assert secret not in result.output
    assert secret not in audit.read_text(encoding="utf-8")
    assert secret not in findings.read_text(encoding="utf-8")


def test_gold_audit_command_sanitizes_invalid_normalization_manifest_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    secret = "NORMALIZATION_MANIFEST_SECRET_SENTINEL"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["unexpected_secret"] = secret
    manifest.write_bytes(canonical_json_bytes(manifest_payload))

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 2
    assert "invalid normalization manifest evidence" in result.output
    assert secret not in result.output
    assert not audit.exists()
    assert not findings.exists()


@pytest.mark.parametrize("duplicate_scope", ["top-level", "nested"])
def test_gold_audit_command_rejects_duplicate_object_keys_recursively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_scope: str,
) -> None:
    secret = "SHADOWED_GOLDEN_PATCH_SECRET"
    sample = (
        _gold_audit_sample()
        if duplicate_scope == "top-level"
        else _gold_audit_sample(trajectory=({"role": "tool", "payload": {"note": "safe"}},))
    )
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
        sample=sample,
    )
    payload = json.loads(normalized.read_text(encoding="utf-8"))
    raw_row = canonical_json_bytes(payload).decode("utf-8")
    if duplicate_scope == "top-level":
        safe_value = json.dumps(
            payload["trajectory"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        needle = f'"trajectory":{safe_value}'
        replacement = f'"trajectory":[{{"golden_patch":"{secret}"}}],{needle}'
    else:
        needle = '"payload":{"note":"safe"}'
        replacement = f'"payload":{{"golden_patch":"{secret}"}},"payload":{{"note":"safe"}}'
    assert raw_row.count(needle) == 1
    _replace_gold_audit_input(
        normalized,
        manifest,
        raw_row.replace(needle, replacement).encode("utf-8"),
    )

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 1, result.output
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    finding = json.loads(findings.read_text(encoding="utf-8"))
    assert audit_payload["status"] == "FAIL"
    assert audit_payload["structural_scan"] == {"status": "FAIL", "finding_count": 1}
    assert finding["reason_code"] == "invalid_normalized_sample"
    assert finding["sample_id"] is None
    assert secret not in result.output
    assert secret not in audit.read_text(encoding="utf-8")
    assert secret not in findings.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "raw_row",
    [
        b'{"secret":"MALFORMED_JSON_SECRET"\n',
        b'["NON_OBJECT_JSON_SECRET"]\n',
    ],
)
def test_gold_audit_command_maps_malformed_or_non_object_rows_to_sanitized_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_row: bytes,
) -> None:
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    _replace_gold_audit_input(normalized, manifest, raw_row)

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    finding = json.loads(findings.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert finding["reason_code"] == "invalid_normalized_sample"
    assert finding["sample_id"] is None
    for secret in ("MALFORMED_JSON_SECRET", "NON_OBJECT_JSON_SECRET"):
        assert secret not in result.output
        assert secret not in audit.read_text(encoding="utf-8")
        assert secret not in findings.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutated_field", "mutated_value", "message"),
    [
        ("normalized_sha256", "f" * 64, "coverage is inconsistent"),
        ("unexpected", "field", "invalid oracle-isolation attestation"),
    ],
)
def test_gold_audit_command_rejects_malformed_or_mismatched_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_field: str,
    mutated_value: str,
    message: str,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload[mutated_field] = mutated_value
    attestation.write_bytes(canonical_json_bytes(payload))

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 2
    assert message in result.output
    assert mutated_value not in result.output
    assert not audit.exists()
    assert not findings.exists()


def test_gold_audit_command_rejects_an_unbound_normalized_path_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    alias = tmp_path / "normalized-alias.jsonl"
    alias.write_bytes(normalized.read_bytes())

    result = _invoke_gold_audit(
        alias,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 2
    assert "artifact path does not match" in result.output
    assert not audit.exists()
    assert not findings.exists()


def test_gold_audit_command_rejects_preexisting_audit_output_before_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    audit.write_text("preserve me\n", encoding="utf-8")

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 2
    assert "audit output already exists" in result.output
    assert audit.read_text(encoding="utf-8") == "preserve me\n"
    assert not findings.exists()


def test_gold_audit_command_rejects_colliding_output_paths_before_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, _ = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )

    same_output = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        audit,
        attestation=attestation,
    )
    staged_input_collision = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        normalized,
        attestation=attestation,
    )

    assert same_output.exit_code == 2
    assert "must be distinct" in same_output.output
    assert staged_input_collision.exit_code == 2
    assert "must not collide with staged inputs" in staged_input_collision.output
    assert not audit.exists()


def test_gold_audit_command_publishes_fail_for_population_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
        sample_count=2,
    )

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["expected_sample_count"] == 2
    assert payload["audited_sample_count"] == 1
    assert payload["structural_scan"] == {"status": "FAIL", "finding_count": 0}
    assert payload["oracle_isolation"]["status"] == "FAIL"


def test_gold_audit_command_never_passes_an_empty_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, _, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    normalized.write_bytes(b"")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "status": "FAIL",
            "input_row_count": 0,
            "accepted_count": 0,
            "unique_rollout_key_count": 0,
            "normalized_sha256": file_identity(normalized)[0],
            "normalized_bytes": 0,
        }
    )
    manifest.write_bytes(canonical_json_bytes(manifest_payload))

    result = _invoke_gold_audit(normalized, manifest, audit, findings)

    assert result.exit_code == 1, result.output
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["expected_sample_count"] == payload["audited_sample_count"] == 0


def test_gold_audit_command_rechecks_inputs_at_audit_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized, manifest, attestation, audit, findings = _gold_audit_command_inputs(
        tmp_path,
        monkeypatch,
    )
    original_writer = cli_module.write_immutable_json

    def mutate_on_publish(path, value, *, before_publish=None):
        def mutate_then_verify() -> object:
            normalized.write_bytes(normalized.read_bytes() + b" ")
            return before_publish() if before_publish is not None else None

        return original_writer(path, value, before_publish=mutate_then_verify)

    monkeypatch.setattr(cli_module, "write_immutable_json", mutate_on_publish)

    result = _invoke_gold_audit(
        normalized,
        manifest,
        audit,
        findings,
        attestation=attestation,
    )

    assert result.exit_code == 2
    assert "input changed while it was being processed" in result.output
    assert findings.exists()
    assert not audit.exists()
