from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nodelm.cli import app
from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.models import NormalizedSample


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


def test_pilot_command_validates_and_writes_normalized_samples(tmp_path: Path) -> None:
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
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nodelm.repository-split/v1",
                "repositories": {
                    "train": ["github.com/acme/widget"],
                    "evaluation": [],
                },
            }
        ),
        encoding="utf-8",
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

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(input_path),
            "--output",
            str(output),
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
        )
    )


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
    if destination_exists:
        destination.mkdir()
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

    assert result.exit_code == 0, result.output
    assert download_called is True


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
    monkeypatch: object,
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
            }
        )
        + "\n",
        encoding="utf-8",
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


def test_invalid_normalized_pilot_input_is_actionable(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"repository":"acme/widget"}\n', encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "nodelm.repository-split/v1",
                "repositories": {"train": [], "evaluation": []},
            }
        ),
        encoding="utf-8",
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
            "--split-manifest",
            str(split_manifest),
        ],
    )

    assert result.exit_code == 2
    assert "invalid normalized sample" in result.output


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
