from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nodelm.cli import app


def test_snapshot_materializes_normalizes_splits_and_builds_a_pilot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "traces.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "acme__widget-1",
                "repo": "acme/widget",
                "trajectory_id": "rollout-1",
                "resolved": 1,
                "trajectory": [{"role": "assistant", "content": "inspect and patch"}],
                "model_patch": "diff --git a/a.ts b/a.ts",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_metadata = tmp_path / "tasks.jsonl"
    task_metadata.write_text(
        json.dumps(
            {
                "instance_id": "acme__widget-1",
                "repo": "acme/widget",
                "base_commit": "b" * 40,
                "license": "MIT",
                "language": "TypeScript",
                "problem_statement": "repair the fixture widget retry implementation",
                "patch": "gold patch is join-excluded",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: nodelm.dataset-registry/v1\n"
        "sources:\n"
        "  - name: fixture-traces\n"
        "    repository_id: owner/fixture-traces\n"
        f"    revision: {'a' * 40}\n"
        "    dataset_license: cc-by-4.0\n"
        "    snapshot_timestamp_utc: '2026-08-24T00:00:00Z'\n"
        "    observed_rows: 1\n"
        "    evidence_urls: [https://example.invalid/evidence]\n"
        "    status: PASS\n",
        encoding="utf-8",
    )
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    split = tmp_path / "split.json"
    benchmark = tmp_path / "public-benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "benchmark_id": "public-unrelated-1",
                "task": "replace an unrelated database migration planner",
                "patch": "diff --git a/migration.sql b/migration.sql\n+CREATE INDEX unrelated;",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot.json"
    runner = CliRunner()

    materialize = runner.invoke(
        app,
        [
            "datasets",
            "materialize",
            "--source",
            "fixture-traces",
            "--snapshot",
            str(snapshot),
            "--output",
            str(raw),
            "--config",
            str(registry),
        ],
    )
    assert materialize.exit_code == 0, materialize.output

    normalize = runner.invoke(
        app,
        [
            "datasets",
            "normalize",
            "--source",
            "fixture-traces",
            "--input",
            str(raw),
            "--output",
            str(normalized),
            "--harness",
            "fixture-harness",
            "--generating-model",
            "source-config:fixture",
            "--task-metadata",
            str(task_metadata),
            "--config",
            str(registry),
        ],
    )
    assert normalize.exit_code == 0, normalize.output

    split_result = runner.invoke(
        app,
        [
            "split",
            "build",
            "--input",
            str(normalized),
            "--output",
            str(split),
            "--task-metadata",
            str(task_metadata),
            "--benchmark",
            str(benchmark),
            "--near-duplicate-threshold",
            "0.85",
            "--seed",
            "0",
            "--evaluation-fraction",
            "0.1",
        ],
    )
    assert split_result.exit_code == 0, split_result.output
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    assert split_payload["decontamination"]["sample_count"] == 1
    assert split_payload["decontamination"]["benchmark_entry_count"] == 1
    assert split_payload["repositories"]["excluded"] == []
    assert "gold patch is join-excluded" not in split.read_text(encoding="utf-8")

    pilot_result = runner.invoke(
        app,
        [
            "datasets",
            "build-pilot",
            "--input",
            str(normalized),
            "--output",
            str(pilot),
            "--split-manifest",
            str(split),
            "--registry",
            str(registry),
        ],
    )
    assert pilot_result.exit_code == 0, pilot_result.output

    payload = json.loads(pilot.read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 1
    pilot_samples = tmp_path / "pilot.samples.jsonl"
    assert json.loads(pilot_samples.read_text(encoding="utf-8"))["trajectory"]
    assert "gold patch" not in pilot_samples.read_text(encoding="utf-8")
