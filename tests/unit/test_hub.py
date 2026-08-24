from __future__ import annotations

from nodelm.datasets.hub import compare_hub_metadata
from nodelm.models import DatasetSource, VerificationStatus


def _source() -> DatasetSource:
    return DatasetSource(
        name="fixture",
        repository_id="owner/data",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=1,
        evidence_urls=("https://example.invalid",),
        status=VerificationStatus.PASS,
    )


def test_hub_metadata_comparison_requires_exact_revision_and_license() -> None:
    verified = compare_hub_metadata(_source(), sha="a" * 40, dataset_license="cc-by-4.0")
    drifted = compare_hub_metadata(_source(), sha="b" * 40, dataset_license="cc-by-4.0")

    assert verified.status is VerificationStatus.PASS
    assert drifted.status is VerificationStatus.FAIL
    assert "revision" in drifted.issues[0]
