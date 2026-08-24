from __future__ import annotations

import pytest

from nodelm.models import DatasetSource, VerificationStatus
from nodelm.provenance.normalize import NormalizationError
from nodelm.provenance.pipeline import normalize_trace_sample, task_metadata_index


def _source() -> DatasetSource:
    return DatasetSource(
        name="open-swe-traces",
        repository_id="nvidia/Open-SWE-Traces",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=1,
        evidence_urls=("https://example.invalid/evidence",),
        status=VerificationStatus.PASS,
    )


def test_trace_normalization_joins_only_provenance_safe_task_fields() -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "license": "MIT",
        "language": "TypeScript",
        "patch": "gold patch must not cross the join",
        "problem_statement": "also not copied",
    }
    trace = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "trajectory_id": "rollout-1",
        "resolved": 1,
        "trajectory": [{"role": "assistant", "content": "inspect"}],
        "model_patch": "diff --git a/a.ts b/a.ts",
    }

    with task_metadata_index((task,)) as lookup:
        sample = normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-config:model",
            task_lookup=lookup,
        )

    payload = sample.model_dump(mode="json")
    assert payload["base_commit"] == "b" * 40
    assert payload["repository_license"] == "MIT"
    assert payload["generated_patch"] == trace["model_patch"]
    assert "gold patch" not in sample.model_dump_json()
    assert "problem_statement" not in payload
    assert len(payload["sample_id"]) == 64


def test_trace_normalization_works_without_a_join_when_fields_are_present() -> None:
    sample = normalize_trace_sample(
        {
            "instance_id": "one",
            "repo": "acme/widget",
            "base_commit": "b" * 40,
            "license": "Apache-2.0",
            "language": "JavaScript",
            "trajectory_id": "rollout-one",
            "resolved": True,
            "trajectory": [{"role": "tool", "content": "tests pass"}],
            "model_patch": "diff --git a/a.js b/a.js",
        },
        source=_source(),
        harness="openhands",
        generating_model="source-config:model",
    )

    assert sample.language == "JavaScript"
    assert sample.trajectory


@pytest.mark.parametrize(
    ("field", "conflicting_value", "expected_message"),
    [
        ("repo", "other/widget", "repository"),
        ("base_commit", "c" * 40, "base_commit"),
        ("license", "Apache-2.0", "license"),
        ("language", "JavaScript", "language"),
    ],
)
def test_trace_normalization_rejects_conflicting_joined_metadata(
    field: str,
    conflicting_value: str,
    expected_message: str,
) -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "license": "MIT",
        "language": "TypeScript",
    }
    trace = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "license": "MIT",
        "language": "TypeScript",
        "trajectory_id": "rollout-1",
        "resolved": True,
        "trajectory": [{"role": "assistant", "content": "inspect"}],
        field: conflicting_value,
    }

    with (
        task_metadata_index((task,)) as lookup,
        pytest.raises(
            NormalizationError,
            match=rf"trace/task {expected_message} mismatch",
        ),
    ):
        normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-config:model",
            task_lookup=lookup,
        )


def test_trace_normalization_rejects_repository_alias_conflict() -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "base_commit": "b" * 40,
        "license": "MIT",
        "language": "TypeScript",
    }
    trace = {
        "instance_id": "acme__widget-1",
        "repository": "other/widget",
        "trajectory_id": "rollout-1",
        "resolved": True,
        "trajectory": [{"role": "assistant", "content": "inspect"}],
    }

    with (
        task_metadata_index((task,)) as lookup,
        pytest.raises(
            NormalizationError,
            match="trace/task repository mismatch",
        ),
    ):
        normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-config:model",
            task_lookup=lookup,
        )
