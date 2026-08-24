from __future__ import annotations

import pytest

from nodelm.datasets.audit import audit_rows
from nodelm.models import DatasetSource, VerificationStatus
from nodelm.provenance.normalize import NormalizationError


def _source() -> DatasetSource:
    return DatasetSource(
        name="fixture",
        repository_id="owner/fixture",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=3,
        evidence_urls=("https://example.invalid/official",),
        status=VerificationStatus.PASS,
    )


def test_audit_computes_required_distributions_and_keeps_rejections() -> None:
    rows = [
        {
            "instance_id": "one",
            "repo": "Acme/Widget",
            "license": " MIT",
            "language": "TypeScript",
            "resolved": 1,
            "trajectory": [{"action": "read"}, {"action": "patch"}],
            "patch": "+const ok = true;\n",
        },
        {
            "instance_id": "one",
            "repo": "Acme/Widget",
            "license": "GPL-3.0",
            "language": "TypeScript",
            "resolved": 0,
            "trajectory": [],
            "patch": "",
        },
        {
            "instance_id": "three",
            "repo": "Beta/Service",
            "license": None,
            "language": "JavaScript",
            "resolved": True,
            "trajectory": [{"action": "test"}],
        },
    ]

    report = audit_rows(_source(), rows)

    assert report.row_count == 3
    assert report.status is VerificationStatus.PASS
    assert report.matches_declared_row_count is True
    assert report.unique_repositories == 2
    assert report.duplicate_instance_id_count == 1
    assert report.duplicate_instance_ids == ("one",)
    assert report.duplicate_instance_ids_truncated is False
    assert report.language_distribution == {"JavaScript": 1, "TypeScript": 2}
    assert report.resolved_distribution == {"resolved": 2, "unresolved": 1, "unknown": 0}
    assert report.trajectory_lengths == {"count": 3, "min": 0, "p50": 1, "p95": 2, "max": 2}
    assert report.license_distribution == {"ALLOW": 1, "REJECT": 1, "UNKNOWN": 1}
    assert report.rejected_row_count == 2
    assert len(report.rejected_rows) == 2
    assert report.rejected_rows_truncated is False
    assert len(report.logical_rows_sha256) == 64


def test_audit_records_open_swe_patch_shapes_and_observed_schema_types() -> None:
    rows = [
        {
            "instance_id": "one",
            "repo": "Acme/Widget",
            "license": "MIT",
            "language": "TypeScript",
            "resolved": 1,
            "trajectory": [],
            "model_patch": "+one\n",
        },
        {
            "instance_id": "two",
            "repo": "Acme/Widget",
            "license": "MIT",
            "language": "TypeScript",
            "resolved": True,
            "trajectory": [],
            "metadata": {"model_patch": {"patch": "+nested\n"}},
        },
        {
            "instance_id": "three",
            "repo": "Beta/Service",
            "license": "MIT",
            "language": "JavaScript",
            "resolved": False,
            "trajectory": [],
            "model_patch": {"patch": "+mapping\n"},
        },
    ]

    report = audit_rows(_source(), rows)

    assert report.patch_sizes == {"count": 3, "min": 5, "p50": 8, "p95": 9, "max": 9}
    assert report.schema_fields == (
        "instance_id",
        "language",
        "license",
        "metadata",
        "model_patch",
        "repo",
        "resolved",
        "trajectory",
    )
    assert report.schema_field_types == {
        "instance_id": ("string",),
        "language": ("string",),
        "license": ("string",),
        "metadata": ("object",),
        "model_patch": ("object", "string"),
        "repo": ("string",),
        "resolved": ("boolean", "integer"),
        "trajectory": ("array",),
    }


def test_audit_preserves_zero_valued_instance_ids() -> None:
    source = _source().model_copy(update={"observed_rows": 2})
    rows = [
        {"instance_id": 0, "repo": "acme/widget", "license": "MIT", "resolved": True},
        {"instance_id": 0, "repo": "acme/widget", "license": "MIT", "resolved": False},
    ]

    report = audit_rows(source, rows)

    assert report.duplicate_instance_id_count == 1
    assert report.duplicate_instance_ids == ("0",)


def test_audit_reports_declared_row_count_drift() -> None:
    report = audit_rows(_source(), [])

    assert report.matches_declared_row_count is False
    assert report.status is VerificationStatus.FAIL
    assert any("row count drift" in issue for issue in report.issues)


def test_partial_snapshot_audit_does_not_misreport_expected_slice_size_as_drift() -> None:
    report = audit_rows(
        _source(),
        (
            {
                "instance_id": "one",
                "repo": "Acme/Widget",
                "license": "MIT",
                "resolved": True,
            },
        ),
        expect_complete_snapshot=False,
    )

    assert report.input_scope == "partial-snapshot"
    assert report.matches_declared_row_count is None
    assert not any("row count drift" in issue for issue in report.issues)


def test_audit_bounds_rejection_examples_but_counts_every_rejection() -> None:
    report = audit_rows(
        _source(),
        [
            {
                "instance_id": str(index),
                "repo": "Acme/Widget",
                "license": None,
                "resolved": False,
            }
            for index in range(3)
        ],
        max_rejected_examples=1,
    )

    assert report.rejected_row_count == 3
    assert len(report.rejected_rows) == 1
    assert report.rejected_rows_truncated is True


def test_audit_bounds_distribution_samples_and_labels_approximate_percentiles() -> None:
    report = audit_rows(
        _source(),
        [
            {
                "instance_id": str(index),
                "repo": "Acme/Widget",
                "license": "MIT",
                "resolved": False,
                "trajectory": list(range(index)),
                "patch": "x" * index,
            }
            for index in range(3)
        ],
        max_distribution_samples=1,
    )

    assert report.trajectory_lengths["count"] == 3
    assert report.distribution_sample_cap == 1
    assert report.distribution_percentiles_approximate is True


def test_audit_bounds_duplicate_examples_but_counts_every_duplicate() -> None:
    rows = [
        {
            "instance_id": instance_id,
            "repo": "Acme/Widget",
            "license": "MIT",
            "resolved": False,
        }
        for instance_id in ("one", "one", "two", "two", "three", "three")
    ]

    report = audit_rows(_source(), rows, max_duplicate_examples=1)

    assert report.duplicate_instance_id_count == 3
    assert report.duplicate_instance_ids == ("one",)
    assert report.duplicate_instance_id_sample_cap == 1
    assert report.duplicate_instance_ids_truncated is True


def test_audit_records_supplied_input_identity_and_stable_logical_identity() -> None:
    rows = (
        {
            "instance_id": "one",
            "repo": "Acme/Widget",
            "license": "MIT",
            "resolved": True,
        },
    )
    first = audit_rows(_source(), rows, input_sha256="b" * 64, input_bytes=123)
    second = audit_rows(_source(), rows)

    assert first.input_sha256 == "b" * 64
    assert first.input_bytes == 123
    assert first.logical_rows_sha256 == second.logical_rows_sha256


@pytest.mark.parametrize("resolved", ["false", "true", 2])
def test_audit_rejects_ambiguous_resolved_values(resolved: object) -> None:
    with pytest.raises(NormalizationError, match="resolved"):
        audit_rows(
            _source(),
            (
                {
                    "instance_id": "one",
                    "repo": "Acme/Widget",
                    "license": "MIT",
                    "resolved": resolved,
                },
            ),
        )


@pytest.mark.parametrize("row", [{}, {"resolved": None}])
def test_audit_classifies_missing_resolved_as_unknown(row: dict[str, object]) -> None:
    report = audit_rows(
        _source(),
        (
            {
                "instance_id": "one",
                "repo": "Acme/Widget",
                "license": "MIT",
                **row,
            },
        ),
    )

    assert report.resolved_distribution == {"resolved": 0, "unresolved": 0, "unknown": 1}


def test_audit_classifies_documented_negative_one_as_unknown() -> None:
    report = audit_rows(
        _source(),
        (
            {
                "instance_id": "one",
                "repo": "Acme/Widget",
                "license": "MIT",
                "resolved": -1,
            },
        ),
    )

    assert report.resolved_distribution == {"resolved": 0, "unresolved": 0, "unknown": 1}
