from __future__ import annotations

import pytest

from nodelm.models import DatasetSource, VerificationStatus, stable_model_id
from nodelm.provenance.normalize import NormalizationError, UnknownResolutionError
from nodelm.provenance.pipeline import (
    has_exact_normalization_evidence_lineage,
    has_exact_normalized_sample_lineage,
    normalization_evidence_lineage,
    normalize_trace_sample,
    task_metadata_index,
    trace_rollout_key,
)


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


def test_normalization_evidence_lineage_has_exact_reserved_namespaces() -> None:
    expected = normalization_evidence_lineage(
        materialization_manifest_sha256="1" * 64,
        partition_name="sweagent/model/tasks",
        upstream_source="tasks",
        task_source_name="fixture-tasks",
        task_source_revision="2" * 40,
        task_provenance_sha256="3" * 64,
    )

    assert has_exact_normalization_evidence_lineage(("raw:one", *expected), expected)
    assert not has_exact_normalization_evidence_lineage(
        (*expected, "trace-partition:attacker/model/tasks"),
        expected,
    )
    assert not has_exact_normalization_evidence_lineage(expected[:-1], expected)


def test_normalized_sample_lineage_matches_the_complete_producer_contract() -> None:
    expected = normalization_evidence_lineage(
        materialization_manifest_sha256="1" * 64,
        partition_name="sweagent/model/tasks",
        upstream_source="tasks",
        task_source_name="fixture-tasks",
        task_source_revision="2" * 40,
        task_provenance_sha256="3" * 64,
    )
    lineage = (
        f"hf-dataset:owner/traces@{'4' * 40}",
        "instance:acme__widget-1",
        f"raw-row:{'5' * 64}",
        "task-metadata:acme__widget-1",
        *expected,
    )

    assert has_exact_normalized_sample_lineage(
        lineage,
        source_repository_id="owner/traces",
        source_revision="4" * 40,
        instance_id="acme__widget-1",
        evidence_lineage=expected,
    )
    assert not has_exact_normalized_sample_lineage(
        (*lineage, "instance:attacker"),
        source_repository_id="owner/traces",
        source_revision="4" * 40,
        instance_id="acme__widget-1",
        evidence_lineage=expected,
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


def test_trace_normalization_requires_a_matching_safe_task_projection() -> None:
    trace = {
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "instance_id": "missing",
        "repo": "acme/widget",
        "trajectory_id": "rollout-1",
        "resolved": 1,
        "trajectory": [{"role": "assistant", "content": "inspect"}],
    }

    with (
        task_metadata_index(()) as lookup,
        pytest.raises(NormalizationError, match="missing required task provenance"),
    ):
        normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
            task_lookup=lookup,
            require_task_match=True,
            expected_row_dataset_name="nebius/SWE-rebench-V2",
        )


def test_trace_normalization_canonicalizes_license_and_language_aliases() -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repository": "Acme/Widget",
        "base_commit": "B" * 40,
        "repository_license": "MIT",
        "language": "TypeScript",
    }
    trace = {
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "instance_id": "acme__widget-1",
        "repo": "https://github.com/acme/widget.git",
        "base_commit": "b" * 40,
        "license": "mit license",
        "language": "ts",
        "trajectory_id": "rollout-1",
        "resolved": 1,
        "trajectory": [{"role": "assistant", "content": "inspect"}],
    }

    with task_metadata_index((task,)) as lookup:
        sample = normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
            task_lookup=lookup,
            require_task_match=True,
            expected_row_dataset_name="nebius/SWE-rebench-V2",
            extra_lineage=("materialization:a", "trace-partition:harness/model/tasks"),
        )

    assert sample.repository_license == "MIT"
    assert sample.language == "TypeScript"
    assert sample.base_commit == "B" * 40
    assert sample.provenance_lineage[-2:] == (
        "materialization:a",
        "trace-partition:harness/model/tasks",
    )
    assert f"raw-row:{stable_model_id(trace)}" in sample.provenance_lineage


def test_raw_row_lineage_binds_fields_omitted_from_normalized_content() -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repository": "Acme/Widget",
        "base_commit": "b" * 40,
        "repository_license": "MIT",
        "language": "TypeScript",
    }
    trace = {
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "instance_id": "acme__widget-1",
        "trajectory_id": "rollout-1",
        "resolved": 1,
        "tools": ["first-omitted-tool"],
    }

    with task_metadata_index((task,)) as lookup:
        first = normalize_trace_sample(
            trace,
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
            task_lookup=lookup,
            require_task_match=True,
            expected_row_dataset_name="nebius/SWE-rebench-V2",
        )
        second = normalize_trace_sample(
            {**trace, "tools": ["second-omitted-tool"]},
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
            task_lookup=lookup,
            require_task_match=True,
            expected_row_dataset_name="nebius/SWE-rebench-V2",
        )

    assert first.sample_id != second.sample_id


def test_trace_rollout_key_is_scoped_to_exact_partition_leaf() -> None:
    task = {
        "instance_id": "acme__widget-1",
        "repository": "Acme/Widget",
        "base_commit": "b" * 40,
        "repository_license": "MIT",
        "language": "TypeScript",
    }
    trace = {
        "hf_dataset_name": "nebius/SWE-rebench-V2",
        "instance_id": "acme__widget-1",
        "trajectory_id": "rollout-1",
        "resolved": -1,
    }

    with task_metadata_index((task,)) as lookup:
        first = trace_rollout_key(
            trace,
            source=_source(),
            partition_name="openhands/model/swe-rebench-v2",
            row_dataset_name="nebius/SWE-rebench-V2",
            task_lookup=lookup,
        )
        second = trace_rollout_key(
            {**trace, "hf_dataset_name": "AweAI-Team/Scale-SWE"},
            source=_source(),
            partition_name="openhands/model/scale-swe",
            row_dataset_name="AweAI-Team/Scale-SWE",
            task_lookup=lookup,
        )

    assert first != second


def test_trace_normalization_rejects_wrong_row_task_family() -> None:
    with pytest.raises(NormalizationError, match="hf_dataset_name"):
        normalize_trace_sample(
            {
                "hf_dataset_name": "AweAI-Team/Scale-SWE",
                "instance_id": "one",
            },
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
            expected_row_dataset_name="nebius/SWE-rebench-V2",
        )


@pytest.mark.parametrize("resolved", (-1, None))
def test_trace_normalization_rejects_unknown_resolution_explicitly(
    resolved: object,
) -> None:
    with pytest.raises(UnknownResolutionError):
        normalize_trace_sample(
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "base_commit": "b" * 40,
                "license": "MIT",
                "language": "TypeScript",
                "trajectory_id": "rollout-one",
                "resolved": resolved,
            },
            source=_source(),
            harness="sweagent",
            generating_model="source-label:model",
        )
