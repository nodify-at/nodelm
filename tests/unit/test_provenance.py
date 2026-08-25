from __future__ import annotations

import pytest

from nodelm.provenance.normalize import (
    NormalizationError,
    normalize_sample,
    parse_resolution_status,
)


def test_normalize_sample_preserves_complete_lineage() -> None:
    sample = normalize_sample(
        {
            "instance_id": "acme__widget-17",
            "repo": "acme/widget",
            "license": "Apache-2.0",
            "base_commit": "a" * 40,
            "language": "TypeScript",
            "resolved": 1,
            "model_patch": "diff --git a/a.ts b/a.ts",
            "trajectory": [{"role": "assistant", "content": "patch"}],
            "trajectory_id": "rollout-2",
        },
        source_dataset="open-swe-traces",
        source_revision="b" * 40,
        harness="sweagent",
        generating_model="model@revision",
        lineage=("raw:row-2", "normalized:v1"),
    )

    assert sample.source_dataset_revision == "b" * 40
    assert sample.repository_license == "Apache-2.0"
    assert sample.issue_or_pr_id == "acme__widget-17"
    assert sample.patch_metadata["sha256"]
    assert sample.generated_patch == "diff --git a/a.ts b/a.ts"
    assert len(sample.trajectory) == 1
    assert sample.provenance_lineage == ("raw:row-2", "normalized:v1")


def test_normalize_sample_rejects_missing_base_commit() -> None:
    with pytest.raises(NormalizationError, match="base_commit"):
        normalize_sample(
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "license": "MIT",
                "language": "TypeScript",
                "resolved": True,
                "trajectory_id": "one",
            },
            source_dataset="fixture",
            source_revision="a" * 40,
            harness="fixture",
            generating_model="fixture",
            lineage=("raw:one",),
        )


@pytest.mark.parametrize(
    "trajectory",
    [
        [{"role": "tool", "payload": {"gold_patch": "SECRET_GOLD"}}],
        [{"role": "tool", "reference": {"patch": "SECRET_REFERENCE"}}],
        [{"golden_patch": "SECRET_GOLDEN_DIRECT"}],
        [{"role": "tool", "payload": {"golden_patch": "SECRET_GOLDEN_NESTED"}}],
    ],
)
def test_normalize_sample_rejects_gold_or_reference_patch(
    trajectory: list[dict[str, object]],
) -> None:
    with pytest.raises(NormalizationError, match="forbidden gold/reference patch"):
        normalize_sample(
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "license": "MIT",
                "base_commit": "a" * 40,
                "language": "TypeScript",
                "resolved": True,
                "trajectory_id": "one",
                "trajectory": trajectory,
            },
            source_dataset="fixture",
            source_revision="a" * 40,
            harness="fixture",
            generating_model="fixture",
            lineage=("raw:one",),
        )


@pytest.mark.parametrize("resolved", ["false", "true", 2, None])
def test_normalize_sample_rejects_ambiguous_resolved_values(resolved: object) -> None:
    with pytest.raises(NormalizationError, match="resolved"):
        normalize_sample(
            {
                "instance_id": "one",
                "repo": "acme/widget",
                "license": "MIT",
                "base_commit": "a" * 40,
                "language": "TypeScript",
                "resolved": resolved,
                "trajectory_id": "one",
            },
            source_dataset="fixture",
            source_revision="a" * 40,
            harness="fixture",
            generating_model="fixture",
            lineage=("raw:one",),
        )


@pytest.mark.parametrize(("raw", "expected"), [(True, True), (False, False), (1, True), (0, False)])
def test_normalize_sample_accepts_explicit_resolved_values(raw: object, expected: bool) -> None:
    sample = normalize_sample(
        {
            "instance_id": "one",
            "repo": "acme/widget",
            "license": "MIT",
            "base_commit": "a" * 40,
            "language": "TypeScript",
            "resolved": raw,
            "trajectory_id": "one",
        },
        source_dataset="fixture",
        source_revision="a" * 40,
        harness="fixture",
        generating_model="fixture",
        lineage=("raw:one",),
    )

    assert sample.resolved is expected


def test_parse_resolution_status_preserves_documented_unknown_marker() -> None:
    assert parse_resolution_status(-1) is None
    assert parse_resolution_status(None) is None
