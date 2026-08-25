from __future__ import annotations

import json

import pytest

from nodelm.models import DatasetSource, VerificationStatus
from nodelm.provenance.task_provenance import (
    TaskProjectionError,
    project_task_provenance,
    task_provenance_projection,
)


def _source() -> DatasetSource:
    return DatasetSource(
        name="swe-rebench-v2",
        repository_id="nebius/SWE-rebench-V2",
        revision="a" * 40,
        dataset_license="cc-by-4.0",
        snapshot_timestamp_utc="2026-08-24T00:00:00Z",
        observed_rows=1,
        evidence_urls=("https://example.invalid/evidence",),
        status=VerificationStatus.PASS,
    )


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": "acme__widget-1",
        "repo": "Acme/Widget",
        "base_commit": "b" * 40,
        "license": "mit license",
        "language": "ts",
        "problem_statement": "must never cross the safe projection",
        "patch": "gold patch must never cross the safe projection",
        "test_patch": "gold tests must never cross the safe projection",
    }
    row.update(overrides)
    return row


def test_task_projection_emits_only_safe_canonical_provenance() -> None:
    record = project_task_provenance(_row(), source=_source())
    payload = record.model_dump(mode="json")

    assert payload == {
        "schema_version": "nodelm.task-provenance/v1",
        "source_dataset": "swe-rebench-v2",
        "source_dataset_revision": "a" * 40,
        "instance_id": "acme__widget-1",
        "repository": "github.com/acme/widget",
        "base_commit": "b" * 40,
        "repository_license": "MIT",
        "language": "TypeScript",
    }
    serialized = record.model_dump_json()
    assert "problem_statement" not in serialized
    assert "gold patch" not in serialized
    assert "test_patch" not in serialized


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"license": None}, "license_unknown"),
        ({"license": "GPL-3.0"}, "license_reject"),
        ({"language": None}, "missing_language"),
        ({"base_commit": "not-a-commit"}, "invalid_base_commit"),
    ],
)
def test_task_projection_rejects_unsafe_or_incomplete_rows(
    overrides: dict[str, object],
    reason_code: str,
) -> None:
    with pytest.raises(TaskProjectionError) as raised:
        project_task_provenance(_row(**overrides), source=_source())

    assert raised.value.reason_code == reason_code


def test_task_projection_excludes_every_conflicting_duplicate() -> None:
    rows = (
        _row(),
        _row(base_commit="c" * 40),
        _row(instance_id="acme__other-2", language="js"),
    )

    with task_provenance_projection(rows, source=_source()) as projection:
        admitted = tuple(projection.iter_admitted())
        rejections = tuple(projection.iter_rejections())

    assert [record.instance_id for record in admitted] == ["acme__other-2"]
    assert [rejection["row_index"] for rejection in rejections] == [0, 1]
    assert {rejection["reason_code"] for rejection in rejections} == {"conflicting_task_provenance"}
    assert "gold patch" not in json.dumps(rejections)


def test_task_projection_reclassifies_a_a_b_as_one_conflict_group() -> None:
    rows = (_row(), _row(), _row(base_commit="c" * 40))

    with task_provenance_projection(rows, source=_source()) as projection:
        admitted = tuple(projection.iter_admitted())
        rejections = tuple(projection.iter_rejections())
        counts = projection.rejection_counts_by_code

    assert admitted == ()
    assert [rejection["row_index"] for rejection in rejections] == [0, 1, 2]
    assert {rejection["reason_code"] for rejection in rejections} == {"conflicting_task_provenance"}
    assert rejections[1]["cause_code"] == "duplicate_task_provenance"
    assert counts == {"conflicting_task_provenance": 3}


@pytest.mark.parametrize("unsafe_first", [True, False])
def test_task_projection_invalid_duplicate_poisons_the_whole_instance(
    unsafe_first: bool,
) -> None:
    unsafe = _row(license="GPL-3.0")
    valid = _row()
    rows = (unsafe, valid) if unsafe_first else (valid, unsafe)

    with task_provenance_projection(rows, source=_source()) as projection:
        admitted = tuple(projection.iter_admitted())
        rejections = tuple(projection.iter_rejections())

    unsafe_row_index = 0 if unsafe_first else 1
    assert admitted == ()
    assert [rejection["row_index"] for rejection in rejections] == [0, 1]
    assert {rejection["reason_code"] for rejection in rejections} == {"conflicting_task_provenance"}
    assert rejections[unsafe_row_index]["cause_code"] == "license_reject"
    assert "copyleft license" in rejections[unsafe_row_index]["cause"]


def test_task_projection_treats_repository_aliases_as_duplicates() -> None:
    rows = (
        _row(repo="Acme/Widget"),
        _row(repo="https://github.com/acme/widget.git"),
    )

    with task_provenance_projection(rows, source=_source()) as projection:
        admitted = tuple(projection.iter_admitted())
        rejections = tuple(projection.iter_rejections())

    assert [record.repository for record in admitted] == ["github.com/acme/widget"]
    assert [rejection["reason_code"] for rejection in rejections] == ["duplicate_task_provenance"]


def test_task_projection_treats_commit_hex_case_as_duplicate() -> None:
    rows = (
        _row(base_commit="B" * 40),
        _row(base_commit="b" * 40),
    )

    with task_provenance_projection(rows, source=_source()) as projection:
        admitted = tuple(projection.iter_admitted())
        rejections = tuple(projection.iter_rejections())

    assert [record.base_commit for record in admitted] == ["b" * 40]
    assert [rejection["reason_code"] for rejection in rejections] == ["duplicate_task_provenance"]
