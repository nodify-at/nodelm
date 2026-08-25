from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.licenses.gate import LicenseDisposition, evaluate_license
from nodelm.models import DatasetSource, VerificationStatus


class TaskProjectionError(ValueError):
    """A raw task row cannot enter the gold-free provenance projection."""

    def __init__(self, reason_code: str, reason: str) -> None:
        super().__init__(reason)
        self.reason_code = reason_code


_CONFLICT_REASON_CODE = "conflicting_task_provenance"
_CONFLICT_REASON = "instance_id has conflicting task provenance"


class TaskProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.task-provenance/v1"] = "nodelm.task-provenance/v1"
    source_dataset: str = Field(min_length=1)
    source_dataset_revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    instance_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    base_commit: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    repository_license: str = Field(min_length=1)
    language: str = Field(min_length=1)


_LANGUAGE_ALIASES = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
}


def canonical_language(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskProjectionError("missing_language", "task language is missing")
    candidate = value.strip()
    return _LANGUAGE_ALIASES.get(candidate.casefold(), candidate.casefold())


def _required_identifier(row: Mapping[str, Any]) -> str:
    value = row.get("instance_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise TaskProjectionError("missing_instance_id", "task instance_id is missing")


def _required_repository(row: Mapping[str, Any]) -> str:
    value = row.get("repo") or row.get("repository")
    if not isinstance(value, str) or not value.strip():
        raise TaskProjectionError("missing_repository", "task repository is missing")
    candidate = value.strip()
    try:
        return canonical_repository(candidate)
    except ValueError as error:
        raise TaskProjectionError("invalid_repository", str(error)) from error


def _required_base_commit(row: Mapping[str, Any]) -> str:
    value = row.get("base_commit")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value.strip()) is None:
        raise TaskProjectionError(
            "invalid_base_commit", "task base_commit must be a full 40-hex commit"
        )
    return value.strip().lower()


def project_task_provenance(
    row: Mapping[str, Any],
    *,
    source: DatasetSource,
) -> TaskProvenanceRecord:
    """Project one task row into the only fields allowed to cross the trace join."""

    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise TaskProjectionError(
            "unverified_source", "task projection requires a registry-verified pinned source"
        )
    raw_license = row.get("license") or row.get("repository_license")
    license_decision = evaluate_license(raw_license if isinstance(raw_license, str) else None)
    if license_decision.disposition is not LicenseDisposition.ALLOW:
        raise TaskProjectionError(
            f"license_{license_decision.disposition.value.casefold()}",
            license_decision.reason,
        )
    if license_decision.normalized_spdx is None:  # pragma: no cover - ALLOW contract
        raise TaskProjectionError("license_unknown", "allowed license lacks normalized SPDX")

    return TaskProvenanceRecord(
        source_dataset=source.name,
        source_dataset_revision=source.revision,
        instance_id=_required_identifier(row),
        repository=_required_repository(row),
        base_commit=_required_base_commit(row),
        repository_license=license_decision.normalized_spdx,
        language=canonical_language(row.get("language")),
    )


def _safe_rejection(
    row_index: int,
    row: Mapping[str, Any],
    *,
    reason_code: str,
    reason: str,
    cause_code: str | None = None,
    cause: str | None = None,
) -> dict[str, Any]:
    try:
        instance_id = _required_identifier(row)
    except TaskProjectionError:
        instance_id = None
    repository = row.get("repo") or row.get("repository")
    rejection = {
        "row_index": row_index,
        "instance_id": instance_id,
        "repository": repository if isinstance(repository, str) else None,
        "reason_code": reason_code,
        "reason": reason,
    }
    if cause_code is not None:
        rejection["cause_code"] = cause_code
    if cause is not None:
        rejection["cause"] = cause
    return rejection


@dataclass(frozen=True)
class TaskProvenanceProjection:
    connection: sqlite3.Connection

    def iter_admitted(self) -> Iterator[TaskProvenanceRecord]:
        rows = self.connection.execute("SELECT payload FROM tasks ORDER BY instance_id")
        for (payload,) in rows:
            yield TaskProvenanceRecord.model_validate_json(str(payload))

    def iter_rejections(self) -> Iterator[dict[str, Any]]:
        rows = self.connection.execute("SELECT payload FROM rejections ORDER BY row_index")
        for (payload,) in rows:
            value = json.loads(str(payload))
            if not isinstance(value, dict):  # pragma: no cover - internal SQLite contract
                raise RuntimeError("task rejection payload is not an object")
            yield {str(key): item for key, item in value.items()}

    @property
    def admitted_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    @property
    def rejected_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM rejections").fetchone()[0])

    @property
    def rejection_counts_by_code(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT reason_code, COUNT(*) FROM rejections GROUP BY reason_code ORDER BY reason_code"
        )
        return {str(code): int(count) for code, count in rows}


def _insert_rejection(
    connection: sqlite3.Connection,
    rejection: Mapping[str, Any],
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO rejections VALUES (?, ?, ?, ?)",
        (
            int(rejection["row_index"]),
            str(rejection["instance_id"]) if rejection.get("instance_id") is not None else None,
            str(rejection["reason_code"]),
            json.dumps(rejection, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
    )


def _reclassify_rejections_as_conflict(
    connection: sqlite3.Connection,
    *,
    instance_id: str,
) -> None:
    prior_rejections = connection.execute(
        "SELECT row_index, payload FROM rejections WHERE instance_id = ? ORDER BY row_index",
        (instance_id,),
    )
    for row_index, prior_payload in prior_rejections:
        prior_rejection = json.loads(str(prior_payload))
        if not isinstance(prior_rejection, dict):  # pragma: no cover - SQLite contract
            raise RuntimeError("task rejection payload is not an object")
        prior_reason_code = prior_rejection.get("reason_code")
        prior_reason = prior_rejection.get("reason")
        if prior_reason_code != _CONFLICT_REASON_CODE:
            if isinstance(prior_reason_code, str):
                prior_rejection["cause_code"] = prior_reason_code
            if isinstance(prior_reason, str):
                prior_rejection["cause"] = prior_reason
        prior_rejection.update(
            {
                "reason_code": _CONFLICT_REASON_CODE,
                "reason": _CONFLICT_REASON,
            }
        )
        connection.execute(
            "UPDATE rejections SET reason_code = ?, payload = ? WHERE row_index = ?",
            (
                _CONFLICT_REASON_CODE,
                json.dumps(
                    prior_rejection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                int(row_index),
            ),
        )


def _mark_instance_conflict(
    connection: sqlite3.Connection,
    *,
    instance_id: str,
    admitted_row_index: int | None = None,
    admitted_repository: str | None = None,
) -> None:
    connection.execute("DELETE FROM tasks WHERE instance_id = ?", (instance_id,))
    connection.execute("INSERT OR IGNORE INTO conflicts VALUES (?)", (instance_id,))
    _reclassify_rejections_as_conflict(connection, instance_id=instance_id)
    if admitted_row_index is not None and admitted_repository is not None:
        _insert_rejection(
            connection,
            {
                "row_index": admitted_row_index,
                "instance_id": instance_id,
                "repository": admitted_repository,
                "reason_code": _CONFLICT_REASON_CODE,
                "reason": _CONFLICT_REASON,
            },
        )


@contextmanager
def task_provenance_projection(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: DatasetSource,
) -> Iterator[TaskProvenanceProjection]:
    """Build a disk-backed, unique, gold-free task provenance projection."""

    descriptor, database_name = tempfile.mkstemp(
        prefix="nodelm-task-provenance-", suffix=".sqlite3"
    )
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE tasks (instance_id TEXT PRIMARY KEY NOT NULL, "
            "row_index INTEGER NOT NULL, repository TEXT NOT NULL, payload TEXT NOT NULL) "
            "WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE conflicts (instance_id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE rejections (row_index INTEGER PRIMARY KEY NOT NULL, "
            "instance_id TEXT, reason_code TEXT NOT NULL, payload TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE INDEX rejections_instance_id ON rejections (instance_id, row_index)"
        )

        for row_index, row in enumerate(rows):
            try:
                record = project_task_provenance(row, source=source)
            except TaskProjectionError as error:
                rejection = _safe_rejection(
                    row_index,
                    row,
                    reason_code=error.reason_code,
                    reason=str(error),
                )
                instance_id = rejection["instance_id"]
                if instance_id is None:
                    _insert_rejection(connection, rejection)
                    continue
                if connection.execute(
                    "SELECT 1 FROM conflicts WHERE instance_id = ?", (instance_id,)
                ).fetchone():
                    _insert_rejection(
                        connection,
                        _safe_rejection(
                            row_index,
                            row,
                            reason_code=_CONFLICT_REASON_CODE,
                            reason=_CONFLICT_REASON,
                            cause_code=error.reason_code,
                            cause=str(error),
                        ),
                    )
                    continue
                existing = connection.execute(
                    "SELECT row_index, repository FROM tasks WHERE instance_id = ?",
                    (instance_id,),
                ).fetchone()
                _insert_rejection(
                    connection,
                    rejection,
                )
                if existing is not None:
                    _mark_instance_conflict(
                        connection,
                        instance_id=str(instance_id),
                        admitted_row_index=int(existing[0]),
                        admitted_repository=str(existing[1]),
                    )
                continue

            if connection.execute(
                "SELECT 1 FROM conflicts WHERE instance_id = ?", (record.instance_id,)
            ).fetchone():
                _insert_rejection(
                    connection,
                    _safe_rejection(
                        row_index,
                        row,
                        reason_code=_CONFLICT_REASON_CODE,
                        reason=_CONFLICT_REASON,
                    ),
                )
                continue

            payload = record.model_dump_json()
            existing = connection.execute(
                "SELECT row_index, repository, payload FROM tasks WHERE instance_id = ?",
                (record.instance_id,),
            ).fetchone()
            if existing is None:
                if connection.execute(
                    "SELECT 1 FROM rejections WHERE instance_id = ? LIMIT 1",
                    (record.instance_id,),
                ).fetchone():
                    _mark_instance_conflict(connection, instance_id=record.instance_id)
                    _insert_rejection(
                        connection,
                        _safe_rejection(
                            row_index,
                            row,
                            reason_code=_CONFLICT_REASON_CODE,
                            reason=_CONFLICT_REASON,
                        ),
                    )
                    continue
                connection.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?)",
                    (record.instance_id, row_index, record.repository, payload),
                )
                continue
            if str(existing[2]) == payload:
                _insert_rejection(
                    connection,
                    _safe_rejection(
                        row_index,
                        row,
                        reason_code="duplicate_task_provenance",
                        reason="instance_id repeats identical task provenance",
                    ),
                )
                continue

            _mark_instance_conflict(
                connection,
                instance_id=record.instance_id,
                admitted_row_index=int(existing[0]),
                admitted_repository=str(existing[1]),
            )
            _insert_rejection(
                connection,
                _safe_rejection(
                    row_index,
                    row,
                    reason_code=_CONFLICT_REASON_CODE,
                    reason=_CONFLICT_REASON,
                ),
            )
        connection.commit()
        yield TaskProvenanceProjection(connection)
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()
