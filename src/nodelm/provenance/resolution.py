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
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from nodelm.artifacts import canonical_json_bytes, content_digest
from nodelm.models import stable_model_id
from nodelm.provenance.normalize import NormalizationError, parse_resolution_status
from nodelm.provenance.task_provenance import TaskProjectionError, canonical_language

_RESOLUTION_KEY_SCHEMA = "nodelm.resolution-key/v1"
_PROJECTED_ROW_IDENTITY_SCHEMA = "nodelm.resolution-projected-row-identity/v1"
_CANDIDATE_IDENTITY_SCHEMA = "nodelm.exact-resolution-candidate-identity/v1"
_REQUEST_IDENTITY_SCHEMA = "nodelm.resolution-evaluation-request-identity/v1"
_CONFLICT_IDENTITY_SCHEMA = "nodelm.resolution-label-conflict-identity/v1"
_PARTITION_NAME_PATTERN = r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$"

Sha256: TypeAlias = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha: TypeAlias = Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{40}$")]
NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
PartitionName: TypeAlias = Annotated[
    StrictStr,
    Field(pattern=_PARTITION_NAME_PATTERN),
]
SupportedLanguage: TypeAlias = Literal["TypeScript", "JavaScript"]
PartitionedRow: TypeAlias = tuple[str, Mapping[str, Any]]


class ResolutionRecoveryError(ValueError):
    """Resolution evidence cannot be recovered without weakening provenance."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolutionRowReference(_StrictFrozenModel):
    """A content-bound pointer to one source row without retaining its trajectory."""

    partition_name: PartitionName
    rollout_id: NonEmptyStr
    projected_row_sha256: Sha256


def _reference_sort_key(reference: ResolutionRowReference) -> tuple[str, str, str]:
    return (
        reference.partition_name,
        reference.rollout_id,
        reference.projected_row_sha256,
    )


def _require_sorted_unique_references(
    references: tuple[ResolutionRowReference, ...],
) -> tuple[ResolutionRowReference, ...]:
    expected = tuple(sorted(references, key=_reference_sort_key))
    identities = tuple(_reference_sort_key(reference) for reference in references)
    if references != expected or len(identities) != len(set(identities)):
        raise ValueError("resolution row references must be unique and strictly sorted")
    return references


class ExactResolutionCandidate(_StrictFrozenModel):
    """A proposed label transfer backed by exact task-and-model-patch evidence."""

    schema_version: Literal["nodelm.exact-resolution-candidate/v1"] = (
        "nodelm.exact-resolution-candidate/v1"
    )
    candidate_id: str = Field(default="", pattern=r"^[0-9a-f]{64}$")
    resolution_key: Sha256
    instance_id: NonEmptyStr
    language: SupportedLanguage
    model_patch_sha256: Sha256
    resolved: StrictBool
    trace_source_revision: CommitSha
    task_source_revision: CommitSha
    label_evidence: tuple[ResolutionRowReference, ...] = Field(min_length=1)
    target_reference: ResolutionRowReference

    @field_validator("label_evidence")
    @classmethod
    def require_sorted_unique_label_evidence(
        cls,
        references: tuple[ResolutionRowReference, ...],
    ) -> tuple[ResolutionRowReference, ...]:
        return _require_sorted_unique_references(references)

    @model_validator(mode="after")
    def populate_and_verify_candidate_id(self) -> ExactResolutionCandidate:
        _require_matching_resolution_key(
            resolution_key=self.resolution_key,
            instance_id=self.instance_id,
            model_patch_sha256_value=self.model_patch_sha256,
            trace_source_revision=self.trace_source_revision,
            task_source_revision=self.task_source_revision,
        )
        expected = exact_resolution_candidate_digest(self)
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", expected)
        elif self.candidate_id != expected:
            raise ValueError("candidate_id does not match exact resolution evidence")
        return self


class ResolutionEvaluationRequest(_StrictFrozenModel):
    """A unique task-and-patch request safe to pass to a repository evaluator."""

    schema_version: Literal["nodelm.resolution-evaluation-request/v1"] = (
        "nodelm.resolution-evaluation-request/v1"
    )
    request_id: str = Field(default="", pattern=r"^[0-9a-f]{64}$")
    resolution_key: Sha256
    instance_id: NonEmptyStr
    language: SupportedLanguage
    model_patch: StrictStr
    model_patch_sha256: Sha256
    trace_source_revision: CommitSha
    task_source_revision: CommitSha
    target_references: tuple[ResolutionRowReference, ...] = Field(min_length=1)

    @field_validator("target_references")
    @classmethod
    def require_sorted_unique_target_references(
        cls,
        references: tuple[ResolutionRowReference, ...],
    ) -> tuple[ResolutionRowReference, ...]:
        return _require_sorted_unique_references(references)

    @model_validator(mode="after")
    def verify_patch_and_request_id(self) -> ResolutionEvaluationRequest:
        if self.model_patch_sha256 != model_patch_sha256(self.model_patch):
            raise ValueError("model_patch_sha256 does not match model_patch")
        _require_matching_resolution_key(
            resolution_key=self.resolution_key,
            instance_id=self.instance_id,
            model_patch_sha256_value=self.model_patch_sha256,
            trace_source_revision=self.trace_source_revision,
            task_source_revision=self.task_source_revision,
        )
        expected = resolution_evaluation_request_digest(self)
        if not self.request_id:
            object.__setattr__(self, "request_id", expected)
        elif self.request_id != expected:
            raise ValueError("request_id does not match evaluation request evidence")
        return self


class ResolutionLabelConflict(_StrictFrozenModel):
    """Known labels that disagree for one exact task-and-patch identity."""

    schema_version: Literal["nodelm.resolution-label-conflict/v1"] = (
        "nodelm.resolution-label-conflict/v1"
    )
    conflict_id: str = Field(default="", pattern=r"^[0-9a-f]{64}$")
    resolution_key: Sha256
    instance_id: NonEmptyStr
    model_patch_sha256: Sha256
    trace_source_revision: CommitSha
    task_source_revision: CommitSha
    false_evidence: tuple[ResolutionRowReference, ...] = Field(min_length=1)
    true_evidence: tuple[ResolutionRowReference, ...] = Field(min_length=1)

    @field_validator("false_evidence", "true_evidence")
    @classmethod
    def require_sorted_unique_conflict_evidence(
        cls,
        references: tuple[ResolutionRowReference, ...],
    ) -> tuple[ResolutionRowReference, ...]:
        return _require_sorted_unique_references(references)

    @model_validator(mode="after")
    def populate_and_verify_conflict_id(self) -> ResolutionLabelConflict:
        _require_matching_resolution_key(
            resolution_key=self.resolution_key,
            instance_id=self.instance_id,
            model_patch_sha256_value=self.model_patch_sha256,
            trace_source_revision=self.trace_source_revision,
            task_source_revision=self.task_source_revision,
        )
        expected = resolution_label_conflict_digest(self)
        if not self.conflict_id:
            object.__setattr__(self, "conflict_id", expected)
        elif self.conflict_id != expected:
            raise ValueError("conflict_id does not match conflicting label evidence")
        return self


def _domain_digest(schema: str, payload: object) -> str:
    return content_digest(f"{schema}\0".encode() + canonical_json_bytes(payload))


def _canonical_instance_id(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ResolutionRecoveryError("resolution row requires a non-empty instance_id")


def _canonical_commit_revision(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ResolutionRecoveryError(f"{field} must be a full 40-hex revision")
    return value.casefold()


def model_patch_sha256(model_patch: str) -> str:
    """Hash the exact UTF-8 model patch without newline or whitespace normalization."""

    if not isinstance(model_patch, str):
        raise TypeError("model_patch must be a string")
    return content_digest(model_patch.encode())


def _resolution_key_from_patch_sha256(
    *,
    instance_id: str,
    model_patch_sha256_value: str,
    trace_source_revision: str,
    task_source_revision: str,
) -> str:
    canonical_instance_id = _canonical_instance_id(instance_id)
    canonical_trace_revision = _canonical_commit_revision(
        trace_source_revision,
        field="trace_source_revision",
    )
    canonical_task_revision = _canonical_commit_revision(
        task_source_revision,
        field="task_source_revision",
    )
    if re.fullmatch(r"[0-9a-f]{64}", model_patch_sha256_value) is None:
        raise ResolutionRecoveryError("model_patch_sha256 must be a lowercase SHA-256 digest")
    return _domain_digest(
        _RESOLUTION_KEY_SCHEMA,
        {
            "instance_id": canonical_instance_id,
            "model_patch_sha256": model_patch_sha256_value,
            "trace_source_revision": canonical_trace_revision,
            "task_source_revision": canonical_task_revision,
        },
    )


def _require_matching_resolution_key(
    *,
    resolution_key: str,
    instance_id: str,
    model_patch_sha256_value: str,
    trace_source_revision: str,
    task_source_revision: str,
) -> None:
    expected = _resolution_key_from_patch_sha256(
        instance_id=instance_id,
        model_patch_sha256_value=model_patch_sha256_value,
        trace_source_revision=trace_source_revision,
        task_source_revision=task_source_revision,
    )
    if resolution_key != expected:
        raise ValueError("resolution_key does not match canonical task and patch identity")


def resolution_key_sha256(
    *,
    instance_id: str,
    model_patch: str,
    trace_source_revision: str,
    task_source_revision: str,
) -> str:
    """Return a revision-bound identity for one exact task and model patch."""

    return _resolution_key_from_patch_sha256(
        instance_id=instance_id,
        model_patch_sha256_value=model_patch_sha256(model_patch),
        trace_source_revision=trace_source_revision,
        task_source_revision=task_source_revision,
    )


def exact_resolution_candidate_digest(candidate: ExactResolutionCandidate) -> str:
    return _domain_digest(
        _CANDIDATE_IDENTITY_SCHEMA,
        candidate.model_dump(mode="json", exclude={"candidate_id"}),
    )


def resolution_evaluation_request_digest(request: ResolutionEvaluationRequest) -> str:
    return _domain_digest(
        _REQUEST_IDENTITY_SCHEMA,
        request.model_dump(mode="json", exclude={"request_id"}),
    )


def resolution_label_conflict_digest(conflict: ResolutionLabelConflict) -> str:
    return _domain_digest(
        _CONFLICT_IDENTITY_SCHEMA,
        conflict.model_dump(mode="json", exclude={"conflict_id"}),
    )


def _projected_row_digest(row: Mapping[str, Any]) -> str:
    if any(not isinstance(key, str) for key in row):
        raise ResolutionRecoveryError("resolution rows require string field names")
    return stable_model_id(
        {
            "schema_version": _PROJECTED_ROW_IDENTITY_SCHEMA,
            "row": dict(row),
        }
    )


def _required_rollout_id(row: Mapping[str, Any]) -> str:
    for field in ("trajectory_id", "rollout_id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ResolutionRecoveryError("resolution row requires trajectory_id or rollout_id")


def _model_patch(row: Mapping[str, Any]) -> str:
    for field in ("model_patch", "pred_patch", "generated_patch"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("model_patch")
        if isinstance(nested, Mapping) and isinstance(nested.get("patch"), str):
            return str(nested["patch"])
    raise ResolutionRecoveryError("resolution row requires an exact model patch field")


def _row_language(row: Mapping[str, Any]) -> str:
    value = row.get("language")
    if not isinstance(value, str) or not value.strip():
        metadata = row.get("metadata")
        value = metadata.get("language") if isinstance(metadata, Mapping) else None
    try:
        return canonical_language(value)
    except TaskProjectionError as error:
        raise ResolutionRecoveryError(str(error)) from error


def _resolution(row: Mapping[str, Any]) -> bool | None:
    try:
        return parse_resolution_status(row.get("resolved"))
    except NormalizationError as error:
        raise ResolutionRecoveryError(str(error)) from error


def _canonical_supported_languages(languages: Iterable[str]) -> frozenset[str]:
    canonical: set[str] = set()
    for language in languages:
        try:
            item = canonical_language(language)
        except TaskProjectionError as error:
            raise ResolutionRecoveryError(str(error)) from error
        if item not in {"TypeScript", "JavaScript"}:
            raise ResolutionRecoveryError(
                "resolution recovery only supports TypeScript and JavaScript"
            )
        canonical.add(item)
    if not canonical:
        raise ResolutionRecoveryError("resolution recovery languages must not be empty")
    return frozenset(canonical)


def _row_reference(partition_name: str, row: Mapping[str, Any]) -> ResolutionRowReference:
    try:
        return ResolutionRowReference(
            partition_name=partition_name,
            rollout_id=_required_rollout_id(row),
            projected_row_sha256=_projected_row_digest(row),
        )
    except ValueError as error:
        if isinstance(error, ResolutionRecoveryError):
            raise
        raise ResolutionRecoveryError(str(error)) from error


def _validated_partition_name(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(_PARTITION_NAME_PATTERN, value) is None:
        raise ResolutionRecoveryError(
            "partition_name must be a normalized harness/model/task-family path"
        )
    return value


def _reference_id(reference: ResolutionRowReference) -> str:
    return stable_model_id(reference)


def _json_payload(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_reference(payload: object) -> ResolutionRowReference:
    return ResolutionRowReference.model_validate_json(str(payload))


def _references_for_label(
    connection: sqlite3.Connection,
    *,
    resolution_key: str,
    resolved: bool,
    language: str | None = None,
) -> tuple[ResolutionRowReference, ...]:
    query = (
        "SELECT reference_payload FROM label_observations WHERE resolution_key = ? AND resolved = ?"
    )
    parameters: tuple[object, ...] = (resolution_key, int(resolved))
    if language is not None:
        query += " AND language = ?"
        parameters += (language,)
    query += " ORDER BY partition_name, rollout_id, projected_row_sha256"
    return tuple(_parse_reference(row[0]) for row in connection.execute(query, parameters))


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE label_observations ("
        "resolution_key TEXT NOT NULL, resolved INTEGER NOT NULL, reference_id TEXT NOT NULL, "
        "instance_id TEXT NOT NULL, language TEXT NOT NULL, model_patch_sha256 TEXT NOT NULL, "
        "partition_name TEXT NOT NULL, rollout_id TEXT NOT NULL, "
        "projected_row_sha256 TEXT NOT NULL, "
        "reference_payload TEXT NOT NULL, "
        "PRIMARY KEY (resolution_key, resolved, reference_id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE targets ("
        "resolution_key TEXT NOT NULL, reference_id TEXT NOT NULL, instance_id TEXT NOT NULL, "
        "language TEXT NOT NULL, model_patch TEXT NOT NULL, model_patch_sha256 TEXT NOT NULL, "
        "partition_name TEXT NOT NULL, rollout_id TEXT NOT NULL, "
        "projected_row_sha256 TEXT NOT NULL, "
        "reference_payload TEXT NOT NULL, "
        "PRIMARY KEY (resolution_key, reference_id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY NOT NULL, "
        "resolution_key TEXT NOT NULL, resolved INTEGER NOT NULL, payload TEXT NOT NULL) "
        "WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE requests (request_id TEXT PRIMARY KEY NOT NULL, "
        "resolution_key TEXT NOT NULL, target_count INTEGER NOT NULL, payload TEXT NOT NULL) "
        "WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE conflicts (conflict_id TEXT PRIMARY KEY NOT NULL, "
        "resolution_key TEXT NOT NULL, payload TEXT NOT NULL) "
        "WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE partition_accounting ("
        "row_kind TEXT NOT NULL, partition_name TEXT NOT NULL, total_rows INTEGER NOT NULL, "
        "ineligible_rows INTEGER NOT NULL, already_known_rows INTEGER NOT NULL, "
        "PRIMARY KEY (row_kind, partition_name)) WITHOUT ROWID"
    )


def _account_partition_row(
    connection: sqlite3.Connection,
    *,
    row_kind: Literal["labeled", "target"],
    partition_name: str,
    ineligible: bool = False,
    already_known: bool = False,
) -> None:
    connection.execute(
        "INSERT INTO partition_accounting VALUES (?, ?, 0, 0, 0) "
        "ON CONFLICT (row_kind, partition_name) DO NOTHING",
        (row_kind, partition_name),
    )
    connection.execute(
        "UPDATE partition_accounting SET total_rows = total_rows + 1, "
        "ineligible_rows = ineligible_rows + ?, "
        "already_known_rows = already_known_rows + ? "
        "WHERE row_kind = ? AND partition_name = ?",
        (int(ineligible), int(already_known), row_kind, partition_name),
    )


def _mark_partition_row(
    connection: sqlite3.Connection,
    *,
    row_kind: Literal["labeled", "target"],
    partition_name: str,
    field: Literal["ineligible_rows", "already_known_rows"],
) -> None:
    connection.execute(
        f"UPDATE partition_accounting SET {field} = {field} + 1 "
        "WHERE row_kind = ? AND partition_name = ?",
        (row_kind, partition_name),
    )


def _index_labeled_rows(
    connection: sqlite3.Connection,
    rows: Iterable[PartitionedRow],
    *,
    trace_source_revision: str,
    task_source_revision: str,
    languages: frozenset[str],
) -> None:
    for partition_name, row in rows:
        partition_name = _validated_partition_name(partition_name)
        _account_partition_row(
            connection,
            row_kind="labeled",
            partition_name=partition_name,
        )
        language = _row_language(row)
        if language not in languages:
            _mark_partition_row(
                connection,
                row_kind="labeled",
                partition_name=partition_name,
                field="ineligible_rows",
            )
            continue
        resolved = _resolution(row)
        if resolved is None:
            continue
        instance_id = _canonical_instance_id(row.get("instance_id"))
        model_patch = _model_patch(row)
        patch_sha256 = model_patch_sha256(model_patch)
        key = resolution_key_sha256(
            instance_id=instance_id,
            model_patch=model_patch,
            trace_source_revision=trace_source_revision,
            task_source_revision=task_source_revision,
        )
        reference = _row_reference(partition_name, row)
        connection.execute(
            "INSERT OR IGNORE INTO label_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                int(resolved),
                _reference_id(reference),
                instance_id,
                language,
                patch_sha256,
                reference.partition_name,
                reference.rollout_id,
                reference.projected_row_sha256,
                _json_payload(reference),
            ),
        )


def _index_target_rows(
    connection: sqlite3.Connection,
    rows: Iterable[PartitionedRow],
    *,
    trace_source_revision: str,
    task_source_revision: str,
    languages: frozenset[str],
) -> None:
    for partition_name, row in rows:
        partition_name = _validated_partition_name(partition_name)
        _account_partition_row(
            connection,
            row_kind="target",
            partition_name=partition_name,
        )
        language = _row_language(row)
        if language not in languages:
            _mark_partition_row(
                connection,
                row_kind="target",
                partition_name=partition_name,
                field="ineligible_rows",
            )
            continue
        if _resolution(row) is not None:
            _mark_partition_row(
                connection,
                row_kind="target",
                partition_name=partition_name,
                field="already_known_rows",
            )
            continue
        instance_id = _canonical_instance_id(row.get("instance_id"))
        model_patch = _model_patch(row)
        patch_sha256 = model_patch_sha256(model_patch)
        key = resolution_key_sha256(
            instance_id=instance_id,
            model_patch=model_patch,
            trace_source_revision=trace_source_revision,
            task_source_revision=task_source_revision,
        )
        reference = _row_reference(partition_name, row)
        connection.execute(
            "INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                _reference_id(reference),
                instance_id,
                language,
                model_patch,
                patch_sha256,
                reference.partition_name,
                reference.rollout_id,
                reference.projected_row_sha256,
                _json_payload(reference),
            ),
        )


def _materialize_conflicts(
    connection: sqlite3.Connection,
    *,
    trace_source_revision: str,
    task_source_revision: str,
) -> None:
    rows = connection.execute(
        "SELECT resolution_key, MIN(instance_id), MIN(model_patch_sha256) "
        "FROM label_observations "
        "WHERE resolution_key IN (SELECT resolution_key FROM targets) "
        "GROUP BY resolution_key HAVING MIN(resolved) != MAX(resolved) "
        "ORDER BY resolution_key"
    )
    for resolution_key, instance_id, patch_sha256 in rows:
        conflict = ResolutionLabelConflict(
            resolution_key=str(resolution_key),
            instance_id=str(instance_id),
            model_patch_sha256=str(patch_sha256),
            trace_source_revision=trace_source_revision,
            task_source_revision=task_source_revision,
            false_evidence=_references_for_label(
                connection,
                resolution_key=str(resolution_key),
                resolved=False,
            ),
            true_evidence=_references_for_label(
                connection,
                resolution_key=str(resolution_key),
                resolved=True,
            ),
        )
        connection.execute(
            "INSERT INTO conflicts VALUES (?, ?, ?)",
            (conflict.conflict_id, conflict.resolution_key, _json_payload(conflict)),
        )


def _target_groups(
    connection: sqlite3.Connection,
) -> Iterator[
    tuple[
        str,
        str,
        str,
        str,
        str,
        tuple[ResolutionRowReference, ...],
    ]
]:
    rows = connection.execute(
        "SELECT resolution_key, instance_id, language, model_patch, model_patch_sha256, "
        "reference_payload FROM targets "
        "ORDER BY resolution_key, partition_name, rollout_id, projected_row_sha256"
    )
    current_key: str | None = None
    current_identity: tuple[str, str, str, str] | None = None
    references: list[ResolutionRowReference] = []
    for raw_key, raw_instance_id, raw_language, raw_patch, raw_patch_sha256, payload in rows:
        key = str(raw_key)
        identity = (
            str(raw_instance_id),
            str(raw_language),
            str(raw_patch),
            str(raw_patch_sha256),
        )
        if current_key is not None and key != current_key:
            if current_identity is None:  # pragma: no cover - internal grouping contract
                raise RuntimeError("target group has no identity")
            yield current_key, *current_identity, tuple(references)
            references = []
            current_identity = None
        if current_identity is None:
            current_key = key
            current_identity = identity
        elif identity != current_identity:
            raise ResolutionRecoveryError(
                "one resolution key has conflicting target task, patch, or language metadata"
            )
        references.append(_parse_reference(payload))
    if current_key is not None:
        if current_identity is None:  # pragma: no cover - internal grouping contract
            raise RuntimeError("target group has no identity")
        yield current_key, *current_identity, tuple(references)


def _unanimous_label(
    connection: sqlite3.Connection,
    *,
    resolution_key: str,
    language: str,
) -> bool | None:
    aggregate = connection.execute(
        "SELECT MIN(resolved), MAX(resolved), COUNT(*) FROM label_observations "
        "WHERE resolution_key = ?",
        (resolution_key,),
    ).fetchone()
    if aggregate is None or int(aggregate[2]) == 0 or int(aggregate[0]) != int(aggregate[1]):
        return None
    matching_language_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM label_observations WHERE resolution_key = ? AND language = ?",
            (resolution_key, language),
        ).fetchone()[0]
    )
    if matching_language_count == 0:
        return None
    return bool(int(aggregate[0]))


def _materialize_outputs(
    connection: sqlite3.Connection,
    *,
    trace_source_revision: str,
    task_source_revision: str,
) -> None:
    _materialize_conflicts(
        connection,
        trace_source_revision=trace_source_revision,
        task_source_revision=task_source_revision,
    )
    for key, instance_id, language, model_patch, patch_sha256, references in _target_groups(
        connection
    ):
        if language not in {"TypeScript", "JavaScript"}:  # pragma: no cover - index contract
            raise RuntimeError("target index contains an unsupported language")
        supported_language = cast(SupportedLanguage, language)
        resolved = _unanimous_label(
            connection,
            resolution_key=key,
            language=language,
        )
        if resolved is None:
            request = ResolutionEvaluationRequest(
                resolution_key=key,
                instance_id=instance_id,
                language=supported_language,
                model_patch=model_patch,
                model_patch_sha256=patch_sha256,
                trace_source_revision=trace_source_revision,
                task_source_revision=task_source_revision,
                target_references=references,
            )
            connection.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?)",
                (
                    request.request_id,
                    request.resolution_key,
                    len(request.target_references),
                    _json_payload(request),
                ),
            )
            continue
        label_evidence = _references_for_label(
            connection,
            resolution_key=key,
            resolved=resolved,
            language=language,
        )
        for target_reference in references:
            candidate = ExactResolutionCandidate(
                resolution_key=key,
                instance_id=instance_id,
                language=supported_language,
                model_patch_sha256=patch_sha256,
                resolved=resolved,
                trace_source_revision=trace_source_revision,
                task_source_revision=task_source_revision,
                label_evidence=label_evidence,
                target_reference=target_reference,
            )
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                (
                    candidate.candidate_id,
                    candidate.resolution_key,
                    int(candidate.resolved),
                    _json_payload(candidate),
                ),
            )


@dataclass(frozen=True)
class ResolutionRecovery:
    """A deterministic view over disk-backed recovery candidates and evaluation work."""

    connection: sqlite3.Connection

    def iter_candidates(self) -> Iterator[ExactResolutionCandidate]:
        rows = self.connection.execute("SELECT payload FROM candidates ORDER BY candidate_id")
        for (payload,) in rows:
            yield ExactResolutionCandidate.model_validate_json(str(payload))

    def iter_evaluation_requests(self) -> Iterator[ResolutionEvaluationRequest]:
        rows = self.connection.execute("SELECT payload FROM requests ORDER BY request_id")
        for (payload,) in rows:
            yield ResolutionEvaluationRequest.model_validate_json(str(payload))

    def iter_conflicts(self) -> Iterator[ResolutionLabelConflict]:
        rows = self.connection.execute("SELECT payload FROM conflicts ORDER BY conflict_id")
        for (payload,) in rows:
            yield ResolutionLabelConflict.model_validate_json(str(payload))

    @property
    def candidate_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    @property
    def evaluation_request_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0])

    @property
    def conflict_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])

    def _partition_total(self, row_kind: Literal["labeled", "target"]) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(SUM(total_rows), 0) FROM partition_accounting WHERE row_kind = ?",
                (row_kind,),
            ).fetchone()[0]
        )

    def _partition_counter(
        self,
        row_kind: Literal["labeled", "target"],
        field: Literal["ineligible_rows", "already_known_rows"],
    ) -> int:
        return int(
            self.connection.execute(
                f"SELECT COALESCE(SUM({field}), 0) FROM partition_accounting WHERE row_kind = ?",
                (row_kind,),
            ).fetchone()[0]
        )

    def _row_counts_by_partition(
        self,
        row_kind: Literal["labeled", "target"],
    ) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT partition_name, total_rows FROM partition_accounting "
            "WHERE row_kind = ? ORDER BY partition_name",
            (row_kind,),
        )
        return {str(partition_name): int(total) for partition_name, total in rows}

    @property
    def labeled_row_count(self) -> int:
        return self._partition_total("labeled")

    @property
    def target_row_count(self) -> int:
        return self._partition_total("target")

    @property
    def target_ineligible_count(self) -> int:
        return self._partition_counter("target", "ineligible_rows")

    @property
    def target_already_known_count(self) -> int:
        return self._partition_counter("target", "already_known_rows")

    @property
    def candidate_row_count(self) -> int:
        return self.candidate_count

    @property
    def candidate_unique_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT resolution_key) FROM candidates"
            ).fetchone()[0]
        )

    @property
    def candidate_resolved_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE resolved = 1"
            ).fetchone()[0]
        )

    @property
    def candidate_unresolved_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM candidates WHERE resolved = 0"
            ).fetchone()[0]
        )

    @property
    def queued_target_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(SUM(target_count), 0) FROM requests"
            ).fetchone()[0]
        )

    @property
    def labeled_row_counts_by_partition(self) -> dict[str, int]:
        return self._row_counts_by_partition("labeled")

    @property
    def target_row_counts_by_partition(self) -> dict[str, int]:
        return self._row_counts_by_partition("target")


@contextmanager
def build_resolution_recovery(
    labeled_rows: Iterable[PartitionedRow],
    target_rows: Iterable[PartitionedRow],
    *,
    trace_source_revision: str,
    task_source_revision: str,
    languages: Iterable[str] = ("TypeScript", "JavaScript"),
) -> Iterator[ResolutionRecovery]:
    """Build a global, disk-backed exact-label index and unique evaluation queue.

    Input mappings are only read. SQLite retains exact model patches for evaluator requests,
    row references, and label evidence; it never retains trajectory, gold, or test payloads.
    """

    canonical_trace_revision = _canonical_commit_revision(
        trace_source_revision,
        field="trace_source_revision",
    )
    canonical_task_revision = _canonical_commit_revision(
        task_source_revision,
        field="task_source_revision",
    )
    canonical_languages = _canonical_supported_languages(languages)
    descriptor, database_name = tempfile.mkstemp(
        prefix="nodelm-resolution-recovery-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        _create_schema(connection)
        _index_labeled_rows(
            connection,
            labeled_rows,
            trace_source_revision=canonical_trace_revision,
            task_source_revision=canonical_task_revision,
            languages=canonical_languages,
        )
        _index_target_rows(
            connection,
            target_rows,
            trace_source_revision=canonical_trace_revision,
            task_source_revision=canonical_task_revision,
            languages=canonical_languages,
        )
        _materialize_outputs(
            connection,
            trace_source_revision=canonical_trace_revision,
            task_source_revision=canonical_task_revision,
        )
        connection.commit()
        yield ResolutionRecovery(connection)
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()
