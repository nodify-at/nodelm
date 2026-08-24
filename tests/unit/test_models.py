from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodelm.models import (
    DatasetSource,
    NormalizedSample,
    SolveContext,
    VerificationStatus,
    stable_model_id,
)


def test_verified_dataset_requires_revision_and_license() -> None:
    with pytest.raises(ValidationError):
        DatasetSource(
            name="example",
            repository_id="owner/dataset",
            status=VerificationStatus.PASS,
        )


def test_verified_dataset_rejects_symbolic_revision() -> None:
    with pytest.raises(ValidationError, match="40-hex"):
        DatasetSource(
            name="example",
            repository_id="owner/dataset",
            revision="main",
            dataset_license="Apache-2.0",
            snapshot_timestamp_utc="2026-08-24T00:00:00Z",
            observed_rows=1,
            evidence_urls=("https://example.invalid/evidence",),
            status=VerificationStatus.PASS,
        )


def test_unverified_dataset_may_preserve_unresolved_fields() -> None:
    source = DatasetSource(
        name="example",
        repository_id="owner/dataset",
        status=VerificationStatus.UNVERIFIED,
        notes="Awaiting primary-source verification",
    )

    assert source.revision is None
    assert source.dataset_license is None


def test_normalized_sample_rejects_missing_provenance() -> None:
    with pytest.raises(ValidationError):
        NormalizedSample.model_validate(
            {
                "source_dataset": "example",
                "repository": "acme/widget",
                "language": "TypeScript",
            }
        )


def test_solve_context_structurally_rejects_gold_patch() -> None:
    with pytest.raises(ValidationError):
        SolveContext.model_validate(
            {
                "repository": "acme/widget",
                "base_commit": "a" * 40,
                "task": "Repair the failing test",
                "gold_patch": "do not leak me",
            }
        )


def test_solve_context_rejects_nested_metadata_escape_hatch() -> None:
    with pytest.raises(ValidationError):
        SolveContext.model_validate(
            {
                "repository": "acme/widget",
                "base_commit": "a" * 40,
                "task": "Repair the failing test",
                "metadata": {"gold_patch": "do not leak me"},
            }
        )


def test_stable_model_id_is_order_independent() -> None:
    first = stable_model_id({"b": [2, 1], "a": "value"})
    second = stable_model_id({"a": "value", "b": [2, 1]})

    assert first == second
    assert len(first) == 64


def test_normalized_sample_has_a_deterministic_verified_sample_id() -> None:
    payload = {
        "source_dataset": "fixture",
        "source_dataset_revision": "a" * 40,
        "repository": "acme/widget",
        "repository_license": "MIT",
        "base_commit": "b" * 40,
        "issue_or_pr_id": "issue-one",
        "language": "TypeScript",
        "harness": "fixture",
        "generating_model": "fixture@revision",
        "rollout_id": "rollout-one",
        "resolved": True,
        "patch_metadata": {"bytes": 1},
        "provenance_lineage": ("raw:one",),
    }
    sample = NormalizedSample.model_validate(payload)
    restored = NormalizedSample.model_validate(sample.model_dump(mode="json"))

    assert len(sample.sample_id) == 64
    assert restored.sample_id == sample.sample_id

    with pytest.raises(ValidationError, match="sample_id"):
        NormalizedSample.model_validate({**sample.model_dump(mode="json"), "sample_id": "a" * 64})

    changed_payload = sample.model_dump(mode="json", exclude={"sample_id"})
    changed_payload["trajectory"] = [{"role": "assistant", "content": "changed"}]
    changed = NormalizedSample.model_validate(changed_payload)
    assert changed.sample_id != sample.sample_id
