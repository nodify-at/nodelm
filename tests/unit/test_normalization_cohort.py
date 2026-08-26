from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nodelm.artifacts import canonical_json_bytes, file_identity
from nodelm.cli import app
from nodelm.models import NormalizedSample
from nodelm.provenance import cohort as cohort_module
from nodelm.provenance.cohort import (
    NormalizationCohortError,
    build_normalization_cohort,
)
from nodelm.provenance.manifests import NormalizationCohortManifestV1


def _sample(*, issue: str, harness: str, model: str) -> NormalizedSample:
    partition = f"{harness}/{model}/tasks"
    return NormalizedSample(
        source_dataset="fixture-traces",
        source_dataset_revision="a" * 40,
        repository="acme/widget",
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id=issue,
        language="TypeScript",
        harness=harness,
        generating_model=f"source-label:{model}",
        rollout_id=f"rollout-{issue}",
        resolved=True,
        trajectory=({"role": "assistant", "content": f"repair {issue}"},),
        generated_patch=f"diff --git a/{issue}.ts b/{issue}.ts\n+repair();\n",
        patch_metadata={"bytes": 1},
        provenance_lineage=(
            f"hf-dataset:owner/fixture-traces@{'a' * 40}",
            f"instance:{issue}",
            f"raw-row:{hashlib.sha256(issue.encode()).hexdigest()}",
            f"task-metadata:{issue}",
            f"materialization:{'3' * 64}",
            f"trace-partition:{partition}",
            "upstream-source:tasks",
            f"task-provenance:fixture-tasks@{'c' * 40}",
            f"task-provenance-artifact:{'6' * 64}",
        ),
    )


def _with_lineage(sample: NormalizedSample, lineage: tuple[str, ...]) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            **sample.model_dump(mode="json", exclude={"sample_id"}),
            "provenance_lineage": lineage,
        }
    )


def _write_member(
    root: Path,
    *,
    partition: str,
    samples: tuple[NormalizedSample, ...],
    status: str = "PASS",
    uniqueness_scope: str = "complete-partition",
) -> tuple[Path, Path]:
    harness, model, _ = partition.split("/")
    member_root = root / partition.replace("/", "-")
    member_root.mkdir()
    normalized = member_root / "normalized.jsonl"
    normalized.write_bytes(
        b"".join(canonical_json_bytes(sample.model_dump(mode="json")) for sample in samples)
    )
    normalized_identity = file_identity(normalized)
    manifest = member_root / "normalized.manifest.json"
    payload: dict[str, Any] = {
        "schema_version": "nodelm.normalization-manifest/v2",
        "status": status,
        "source_name": "fixture-traces",
        "source_repository_id": "owner/fixture-traces",
        "source_revision": "a" * 40,
        "partition_name": partition,
        "harness": harness,
        "generating_model": f"source-label:{model}",
        "upstream_source": "tasks",
        "row_dataset_name": "owner/fixture-tasks",
        "input_sha256": "1" * 64,
        "input_bytes": len(samples),
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
        "input_row_count": len(samples),
        "accepted_count": len(samples),
        "rejected_count": 0,
        "rejection_counts_by_code": {},
        "unique_rollout_key_count": len(samples),
        "duplicate_trace_row_count": 0,
        "conflicting_rollout_identity_count": 0,
        "conflicting_rollout_row_count": 0,
        "gold_exposure_audit": "NOT RUN",
        "normalized_artifact": normalized.name,
        "normalized_sha256": normalized_identity[0],
        "normalized_bytes": normalized_identity[1],
        "rejection_artifact": "normalized.rejections.jsonl",
        "rejection_sha256": "9" * 64,
        "rejection_bytes": 0,
    }
    if status == "FAIL":
        payload.update(
            {
                "input_row_count": 1,
                "accepted_count": 0,
                "rejected_count": 1,
                "rejection_counts_by_code": {"unknown_resolution": 1},
                "unique_rollout_key_count": 1,
            }
        )
    manifest.write_bytes(canonical_json_bytes(payload))
    return manifest, normalized


def test_build_cohort_sorts_members_and_binds_exact_concatenated_population(
    tmp_path: Path,
) -> None:
    second_manifest, second_normalized = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    first_manifest, first_normalized = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    output = tmp_path / "normalized-cohort.manifest.json"

    result, built_cohort = build_normalization_cohort((second_manifest, first_manifest), output)

    assert result.path == output.resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    cohort = NormalizationCohortManifestV1.model_validate(payload)
    assert cohort == built_cohort
    assert tuple(member.partition_name for member in cohort.members) == (
        "openhands/model-a/tasks",
        "sweagent/model-b/tasks",
    )
    assert cohort.member_count == 2
    assert cohort.sample_count == cohort.unique_sample_id_count == 2
    population = first_normalized.read_bytes() + second_normalized.read_bytes()
    assert cohort.population_sha256 == hashlib.sha256(population).hexdigest()
    assert cohort.population_bytes == len(population)
    assert all(".." not in member.normalized_artifact.path for member in cohort.members)


def test_build_cohort_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    duplicate = _sample(issue="one", harness="fixture", model="model")
    first, _ = _write_member(
        tmp_path,
        partition="fixture/model/tasks",
        samples=(duplicate, duplicate),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model"),),
    )

    with pytest.raises(NormalizationCohortError, match="duplicate sample_id"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


def test_build_cohort_rejects_contradictory_reserved_lineage(tmp_path: Path) -> None:
    sample = _sample(issue="one", harness="openhands", model="model-a")
    contradictory = _with_lineage(
        sample,
        (*sample.provenance_lineage, "trace-partition:attacker/model/tasks"),
    )
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(contradictory,),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    with pytest.raises(NormalizationCohortError, match="row lineage"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


@pytest.mark.parametrize("mutation", ("missing-raw-row", "wrong-source", "duplicate-instance"))
def test_build_cohort_rejects_incomplete_producer_lineage(
    tmp_path: Path,
    mutation: str,
) -> None:
    sample = _sample(issue="one", harness="openhands", model="model-a")
    lineage = sample.provenance_lineage
    if mutation == "missing-raw-row":
        changed_lineage = tuple(item for item in lineage if not item.startswith("raw-row:"))
    elif mutation == "wrong-source":
        changed_lineage = ("hf-dataset:attacker/traces@" + "a" * 40, *lineage[1:])
    else:
        changed_lineage = (*lineage, "instance:one")
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_with_lineage(sample, changed_lineage),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    with pytest.raises(NormalizationCohortError, match="row lineage"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


def test_build_normalization_cohort_cli_publishes_manifest(tmp_path: Path) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    output = tmp_path / "cohort.json"

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-normalization-cohort",
            "--member-manifest",
            str(second),
            "--member-manifest",
            str(first),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "members=2 samples=2" in result.output
    assert NormalizationCohortManifestV1.model_validate_json(output.read_bytes())


@pytest.mark.parametrize(
    ("status", "scope", "message"),
    [
        ("FAIL", "complete-partition", "PASS normalization"),
        ("PASS", "canary", "complete-partition"),
    ],
)
def test_build_cohort_rejects_non_admissible_member_evidence(
    tmp_path: Path,
    status: str,
    scope: str,
    message: str,
) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
        status=status,
        uniqueness_scope=scope,
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    with pytest.raises(NormalizationCohortError, match=message):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


def test_build_cohort_rejects_row_lineage_inconsistent_with_member(tmp_path: Path) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="wrong", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    with pytest.raises(NormalizationCohortError, match="row lineage"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


def test_build_cohort_rejects_symlinked_directory_escape(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    first, _ = _write_member(
        evidence,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, normalized = _write_member(
        evidence,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_normalized = outside / "normalized.jsonl"
    outside_normalized.write_bytes(normalized.read_bytes())
    escape = second.parent / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["normalized_artifact"] = "escape/normalized.jsonl"
    second.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(NormalizationCohortError, match="contained"):
        build_normalization_cohort((first, second), evidence / "cohort.json")


def test_build_cohort_rejects_same_size_artifact_identity_mismatch(tmp_path: Path) -> None:
    first, normalized = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    changed = bytearray(normalized.read_bytes())
    changed[0] = ord("[")
    normalized.write_bytes(changed)

    with pytest.raises(NormalizationCohortError, match="staged input"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


@pytest.mark.parametrize("member_to_swap", ("first", "final"))
def test_build_cohort_rechecks_every_member_containment_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_to_swap: str,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    first, first_normalized = _write_member(
        evidence,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, final_normalized = _write_member(
        evidence,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    normalized_to_swap = first_normalized if member_to_swap == "first" else final_normalized
    outside = tmp_path / f"outside-{member_to_swap}.normalized.jsonl"
    outside.write_bytes(normalized_to_swap.read_bytes())
    output = evidence / "cohort.json"
    original_writer = cohort_module.write_immutable_json

    def swap_then_write(path: Path, value: Any, *, before_publish: Any = None) -> Any:
        normalized_to_swap.unlink()
        normalized_to_swap.symlink_to(outside)
        return original_writer(path, value, before_publish=before_publish)

    monkeypatch.setattr(cohort_module, "write_immutable_json", swap_then_write)

    with pytest.raises(NormalizationCohortError, match="contained"):
        build_normalization_cohort((first, second), output)
    assert not output.exists()


def test_build_cohort_pins_resolved_output_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(evidence, target_is_directory=True)
    first, _ = _write_member(
        evidence,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        evidence,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    original_writer = cohort_module.write_immutable_json

    def retarget_then_write(path: Path, value: Any, *, before_publish: Any = None) -> Any:
        alias.unlink()
        alias.symlink_to(outside, target_is_directory=True)
        return original_writer(path, value, before_publish=before_publish)

    monkeypatch.setattr(cohort_module, "write_immutable_json", retarget_then_write)

    result, _ = build_normalization_cohort((first, second), alias / "cohort.json")

    assert result.path == evidence / "cohort.json"
    assert result.path.exists()
    assert not (outside / "cohort.json").exists()


def test_build_cohort_translates_sqlite_operational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    def fail_connect(*args: Any, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(cohort_module.sqlite3, "connect", fail_connect)

    with pytest.raises(NormalizationCohortError, match="sample-id index: disk full"):
        build_normalization_cohort((first, second), tmp_path / "cohort.json")


def test_build_cohort_cli_translates_differing_output_collision(tmp_path: Path) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    output = tmp_path / "cohort.json"
    output.write_bytes(b'{"different":true}\n')

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "build-normalization-cohort",
            "--member-manifest",
            str(first),
            "--member-manifest",
            str(second),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.split())
    assert "invalid normalization cohort" in normalized_output
    assert "existing=" in normalized_output
    assert "requested=" in normalized_output


def test_build_cohort_rejects_symlinked_manifest(tmp_path: Path) -> None:
    first, _ = _write_member(
        tmp_path,
        partition="openhands/model-a/tasks",
        samples=(_sample(issue="one", harness="openhands", model="model-a"),),
    )
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    alias = tmp_path / "member-link.json"
    alias.symlink_to(first)

    with pytest.raises(NormalizationCohortError, match="real regular file"):
        build_normalization_cohort((alias, second), tmp_path / "cohort.json")


def test_build_cohort_rejects_oversized_manifest(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )

    with pytest.raises(NormalizationCohortError, match="exceeds 1048576 bytes"):
        build_normalization_cohort((oversized, second), tmp_path / "cohort.json")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_build_cohort_rejects_fifo_manifest_without_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fifo = tmp_path / "member.fifo"
    os.mkfifo(fifo)
    second, _ = _write_member(
        tmp_path,
        partition="sweagent/model-b/tasks",
        samples=(_sample(issue="two", harness="sweagent", model="model-b"),),
    )
    original_read_bytes = Path.read_bytes

    def fail_if_fifo_is_read(path: Path) -> bytes:
        if path == fifo:
            pytest.fail("FIFO manifest must be rejected before a blocking read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_if_fifo_is_read)

    with pytest.raises(NormalizationCohortError, match="real regular file"):
        build_normalization_cohort((fifo, second), tmp_path / "cohort.json")
