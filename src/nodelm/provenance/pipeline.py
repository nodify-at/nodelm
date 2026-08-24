from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.models import DatasetSource, NormalizedSample, VerificationStatus
from nodelm.provenance.normalize import NormalizationError, normalize_sample

TaskMetadata = dict[str, str]
TaskMetadataLookup = Callable[[str], TaskMetadata | None]


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise NormalizationError(f"task metadata requires string {field}")


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
def task_metadata_index(rows: Iterable[Mapping[str, Any]]) -> Iterator[TaskMetadataLookup]:
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
            values = (
                _required_identifier(row, "instance_id"),
                _required_string(row, "repo"),
                _required_string(row, "base_commit"),
                _required_string(row, "license"),
                _required_string(row, "language"),
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
) -> NormalizedSample:
    """Normalize one generated trace without allowing task gold-patch fields to cross over."""

    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise NormalizationError("trace normalization requires a registry-verified source")
    if not harness.strip() or not generating_model.strip():
        raise NormalizationError("harness and generating_model must be non-empty")

    instance_id = _required_identifier(row, "instance_id")
    merged = dict(row)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for field in ("base_commit", "license", "language"):
            if _optional_string(merged, field) is None:
                nested = _optional_string(metadata, field)
                if nested is not None:
                    merged[field] = nested

    task = task_lookup(instance_id) if task_lookup is not None else None
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
            else:
                values_match = trace_value == task_value
            if not values_match:
                raise NormalizationError(f"trace/task {label} mismatch for {instance_id}")

    lineage = [
        f"hf-dataset:{source.repository_id}@{source.revision}",
        f"instance:{instance_id}",
    ]
    if task is not None:
        lineage.append(f"task-metadata:{instance_id}")
    return normalize_sample(
        merged,
        source_dataset=source.name,
        source_revision=source.revision,
        harness=harness.strip(),
        generating_model=generating_model.strip(),
        lineage=tuple(lineage),
    )
