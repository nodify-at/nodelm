from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.licenses.gate import LicenseDisposition, evaluate_license
from nodelm.models import DatasetSource, NormalizedSample, VerificationStatus, stable_model_id
from nodelm.provenance.normalize import (
    NormalizationError,
    UnknownResolutionError,
    normalize_sample,
    parse_resolution_status,
)
from nodelm.provenance.task_provenance import TaskProjectionError, canonical_language

TaskMetadata = dict[str, str]
TaskMetadataLookup = Callable[[str], TaskMetadata | None]


def normalization_evidence_lineage(
    *,
    materialization_manifest_sha256: str,
    partition_name: str,
    upstream_source: str,
    task_source_name: str,
    task_source_revision: str,
    task_provenance_sha256: str,
) -> tuple[str, ...]:
    """Serialize the reserved lineage namespace shared by producer and verifier."""

    return (
        f"materialization:{materialization_manifest_sha256}",
        f"trace-partition:{partition_name}",
        f"upstream-source:{upstream_source}",
        f"task-provenance:{task_source_name}@{task_source_revision}",
        f"task-provenance-artifact:{task_provenance_sha256}",
    )


def has_exact_normalization_evidence_lineage(
    lineage: tuple[str, ...],
    expected: tuple[str, ...],
) -> bool:
    """Require exactly one expected value in each reserved lineage namespace."""

    return all(
        tuple(item for item in lineage if item.startswith(f"{value.partition(':')[0]}:"))
        == (value,)
        for value in expected
    )


def has_exact_normalized_sample_lineage(
    lineage: tuple[str, ...],
    *,
    source_repository_id: str,
    source_revision: str,
    instance_id: str,
    evidence_lineage: tuple[str, ...],
) -> bool:
    """Match the complete ordered lineage emitted by trace normalization."""

    if len(lineage) != 4 + len(evidence_lineage):
        return False
    raw_row = lineage[2]
    return (
        lineage[0] == f"hf-dataset:{source_repository_id}@{source_revision}"
        and lineage[1] == f"instance:{instance_id}"
        and re.fullmatch(r"raw-row:[0-9a-f]{64}", raw_row) is not None
        and lineage[3] == f"task-metadata:{instance_id}"
        and lineage[4:] == evidence_lineage
    )


def trace_rollout_key(
    row: Mapping[str, Any],
    *,
    source: DatasetSource,
    partition_name: str,
    row_dataset_name: str,
    task_lookup: TaskMetadataLookup,
) -> str:
    """Build a leaf-scoped rollout identity before resolution-dependent normalization."""

    if _optional_string(row, "hf_dataset_name") != row_dataset_name:
        raise NormalizationError(
            "trace hf_dataset_name does not match the bound partition task family"
        )
    instance_id = _required_identifier(row, "instance_id")
    rollout_id = _required_one_of_string(row, "trajectory_id", "rollout_id")
    task = task_lookup(instance_id)
    if task is None:
        raise NormalizationError(f"missing required task provenance for {instance_id}")
    try:
        repository = canonical_repository(task["repo"])
    except (KeyError, ValueError) as error:
        raise NormalizationError(
            f"invalid task repository for rollout identity {instance_id}: {error}"
        ) from error
    return stable_model_id(
        {
            "schema_version": "nodelm.trace-rollout-key/v1",
            "source_dataset": source.name,
            "source_dataset_revision": source.revision,
            "partition_name": partition_name,
            "row_dataset_name": row_dataset_name,
            "repository": repository,
            "instance_id": instance_id,
            "rollout_id": rollout_id,
        }
    )


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise NormalizationError(f"task metadata requires string {field}")


def _required_one_of_string(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise NormalizationError(f"task metadata requires string {' or '.join(fields)}")


def _canonical_license(value: str) -> str:
    decision = evaluate_license(value)
    if decision.disposition is not LicenseDisposition.ALLOW or decision.normalized_spdx is None:
        raise NormalizationError(f"task metadata license is not allowed: {value!r}")
    return decision.normalized_spdx


def _canonical_language(value: str) -> str:
    try:
        return canonical_language(value)
    except TaskProjectionError as error:
        raise NormalizationError(str(error)) from error


def _required_identifier(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise NormalizationError(f"task metadata requires {field}")


def _optional_string(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first_optional_string(row: Mapping[str, Any], *fields: str) -> str | None:
    for field in fields:
        value = _optional_string(row, field)
        if value is not None:
            return value
    return None


@contextmanager
def task_metadata_index(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_source_name: str | None = None,
    expected_source_revision: str | None = None,
) -> Iterator[TaskMetadataLookup]:
    """Index only provenance-safe join fields from task rows on disk.

    Gold patches and problem statements are deliberately never stored in the join index.
    """

    descriptor, database_name = tempfile.mkstemp(prefix="nodelm-task-metadata-", suffix=".sqlite3")
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE tasks ("
            "instance_id TEXT PRIMARY KEY NOT NULL, "
            "repository TEXT NOT NULL, base_commit TEXT NOT NULL, "
            "repository_license TEXT NOT NULL, language TEXT NOT NULL) WITHOUT ROWID"
        )
        for row in rows:
            if expected_source_name is not None and (
                _required_string(row, "source_dataset") != expected_source_name
            ):
                raise NormalizationError("task provenance source_dataset mismatch")
            if expected_source_revision is not None and (
                _required_string(row, "source_dataset_revision").casefold()
                != expected_source_revision.casefold()
            ):
                raise NormalizationError("task provenance source revision mismatch")
            values = (
                _required_identifier(row, "instance_id"),
                _required_one_of_string(row, "repo", "repository"),
                _required_string(row, "base_commit"),
                _canonical_license(_required_one_of_string(row, "license", "repository_license")),
                _canonical_language(_required_string(row, "language")),
            )
            existing = connection.execute(
                "SELECT repository, base_commit, repository_license, language "
                "FROM tasks WHERE instance_id = ?",
                (values[0],),
            ).fetchone()
            if existing is not None:
                if tuple(str(item) for item in existing) != values[1:]:
                    raise NormalizationError(
                        f"conflicting task metadata for instance_id: {values[0]}"
                    )
                continue
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
                values,
            )
        connection.commit()

        def lookup(instance_id: str) -> TaskMetadata | None:
            result = connection.execute(
                "SELECT repository, base_commit, repository_license, language "
                "FROM tasks WHERE instance_id = ?",
                (instance_id,),
            ).fetchone()
            if result is None:
                return None
            return {
                "repo": str(result[0]),
                "base_commit": str(result[1]),
                "license": str(result[2]),
                "language": str(result[3]),
            }

        yield lookup
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()


def normalize_trace_sample(
    row: Mapping[str, Any],
    *,
    source: DatasetSource,
    harness: str,
    generating_model: str,
    task_lookup: TaskMetadataLookup | None = None,
    require_task_match: bool = False,
    expected_row_dataset_name: str | None = None,
    extra_lineage: tuple[str, ...] = (),
) -> NormalizedSample:
    """Normalize one generated trace without allowing task gold-patch fields to cross over."""

    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise NormalizationError("trace normalization requires a registry-verified source")
    if not harness.strip() or not generating_model.strip():
        raise NormalizationError("harness and generating_model must be non-empty")
    if any(not item.strip() for item in extra_lineage):
        raise NormalizationError("extra lineage entries must be non-empty")
    if expected_row_dataset_name is not None:
        observed_row_dataset = _optional_string(row, "hf_dataset_name")
        if observed_row_dataset != expected_row_dataset_name:
            raise NormalizationError(
                "trace hf_dataset_name does not match the bound partition task family"
            )

    instance_id = _required_identifier(row, "instance_id")
    merged = dict(row)
    resolution = parse_resolution_status(row.get("resolved"))
    if resolution is None:
        raise UnknownResolutionError(
            "resolved is unknown; normalized-sample/v1 requires boolean evidence"
        )
    merged["resolved"] = resolution
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for field in ("base_commit", "license", "language"):
            if _optional_string(merged, field) is None:
                nested = _optional_string(metadata, field)
                if nested is not None:
                    merged[field] = nested

    task = task_lookup(instance_id) if task_lookup is not None else None
    if require_task_match and task is None:
        raise NormalizationError(f"missing required task provenance for {instance_id}")
    if task is not None:
        joined_fields = (
            ("repo", ("repo", "repository"), "repository"),
            ("base_commit", ("base_commit",), "base_commit"),
            ("license", ("license", "repository_license"), "license"),
            ("language", ("language",), "language"),
        )
        for task_field, trace_fields, label in joined_fields:
            trace_value = _first_optional_string(merged, *trace_fields)
            task_value = task[task_field]
            if trace_value is None:
                merged[task_field] = task_value
                continue
            if task_field == "repo":
                try:
                    values_match = canonical_repository(trace_value) == canonical_repository(
                        task_value
                    )
                except ValueError as error:
                    raise NormalizationError(
                        f"invalid trace/task repository for {instance_id}: {error}"
                    ) from error
            elif task_field == "base_commit":
                values_match = trace_value.casefold() == task_value.casefold()
            elif task_field == "license":
                values_match = _canonical_license(trace_value) == task_value
            elif task_field == "language":
                values_match = _canonical_language(trace_value) == task_value
            else:  # pragma: no cover - joined_fields is closed above
                values_match = trace_value == task_value
            if not values_match:
                raise NormalizationError(f"trace/task {label} mismatch for {instance_id}")
            merged[task_field] = task_value

    lineage = [
        f"hf-dataset:{source.repository_id}@{source.revision}",
        f"instance:{instance_id}",
        f"raw-row:{stable_model_id(row)}",
    ]
    if task is not None:
        lineage.append(f"task-metadata:{instance_id}")
    lineage.extend(item.strip() for item in extra_lineage)
    return normalize_sample(
        merged,
        source_dataset=source.name,
        source_revision=source.revision,
        harness=harness.strip(),
        generating_model=generating_model.strip(),
        lineage=tuple(lineage),
    )
