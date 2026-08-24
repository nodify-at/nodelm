from __future__ import annotations

import hashlib
import heapq
import math
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.licenses.gate import LicenseDisposition, evaluate_license
from nodelm.models import (
    DatasetAuditReport,
    DatasetSource,
    JsonFieldType,
    VerificationStatus,
    stable_model_id,
)
from nodelm.provenance.normalize import parse_resolution_status

DEFAULT_MAX_REJECTED_EXAMPLES = 1_000
DEFAULT_MAX_DISTRIBUTION_SAMPLES = 100_000
DEFAULT_MAX_DUPLICATE_EXAMPLES = 1_000


def _nearest_rank(ordered: list[int], percentile: float) -> int | None:
    if not ordered:
        return None
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class _BoundedDistribution:
    def __init__(self, sample_cap: int) -> None:
        self.sample_cap = sample_cap
        self.count = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self._samples: list[tuple[int, int]] = []

    def add(self, value: int, *, identity: str) -> None:
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        score = int.from_bytes(hashlib.sha256(f"{identity}:{value}".encode()).digest()[:8], "big")
        item = (-score, value)
        if len(self._samples) < self.sample_cap:
            heapq.heappush(self._samples, item)
        elif item > self._samples[0]:
            heapq.heapreplace(self._samples, item)

    @property
    def approximate(self) -> bool:
        return self.count > len(self._samples)

    def summary(self) -> dict[str, int | None]:
        ordered = sorted(value for _, value in self._samples)
        return {
            "count": self.count,
            "min": self.minimum,
            "p50": _nearest_rank(ordered, 0.50),
            "p95": _nearest_rank(ordered, 0.95),
            "max": self.maximum,
        }


@contextmanager
def _disk_backed_identity_index() -> Iterator[sqlite3.Connection]:
    """Keep dataset-cardinality state off the Python heap."""

    descriptor, database_name = tempfile.mkstemp(prefix="nodelm-audit-", suffix=".sqlite3")
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE instance_ids (value TEXT PRIMARY KEY NOT NULL, count INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE repositories (value TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID"
        )
        yield connection
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()


def _patch_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("patch"), str):
        return str(value["patch"])
    return None


def _patch_size(row: Mapping[str, Any]) -> int | None:
    for field in ("model_patch", "patch", "pred_patch", "diff"):
        patch = _patch_text(row.get(field))
        if patch is not None:
            return len(patch.encode("utf-8"))
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for field in ("model_patch", "patch"):
            patch = _patch_text(metadata.get(field))
            if patch is not None:
                return len(patch.encode("utf-8"))
    return None


def _json_field_type(value: Any) -> JsonFieldType:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    raise ValueError(f"unsupported JSON field type: {type(value).__name__}")


def _instance_id(row: Mapping[str, Any], fallback: int) -> str:
    value = row.get("instance_id")
    if value is None:
        value = row.get("id")
    return str(fallback if value is None else value)


def iter_license_rejections(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield the complete repository-license rejection ledger for a raw row stream."""

    for index, row in enumerate(rows):
        raw_repository = row.get("repo") or row.get("repository")
        instance_id = _instance_id(row, index)
        raw_license = row.get("license")
        decision = evaluate_license(raw_license if isinstance(raw_license, str) else None)
        if decision.disposition is not LicenseDisposition.ALLOW:
            yield {
                "row_index": index,
                "instance_id": instance_id,
                "repository": raw_repository,
                "raw_license": raw_license,
                "disposition": decision.disposition.value,
                "reason": decision.reason,
            }


def audit_rows(
    source: DatasetSource,
    rows: Iterable[Mapping[str, Any]],
    *,
    input_sha256: str | None = None,
    input_bytes: int | None = None,
    max_rejected_examples: int = DEFAULT_MAX_REJECTED_EXAMPLES,
    max_distribution_samples: int = DEFAULT_MAX_DISTRIBUTION_SAMPLES,
    max_duplicate_examples: int = DEFAULT_MAX_DUPLICATE_EXAMPLES,
    expect_complete_snapshot: bool = True,
) -> DatasetAuditReport:
    if (input_sha256 is None) != (input_bytes is None):
        raise ValueError("input_sha256 and input_bytes must be supplied together")
    if max_rejected_examples < 0:
        raise ValueError("max_rejected_examples must be non-negative")
    if max_distribution_samples <= 0:
        raise ValueError("max_distribution_samples must be greater than zero")
    if max_duplicate_examples < 0:
        raise ValueError("max_duplicate_examples must be non-negative")

    languages: Counter[str] = Counter()
    resolved: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    schema_fields: set[str] = set()
    schema_field_types: dict[str, set[JsonFieldType]] = {}
    trajectory_lengths = _BoundedDistribution(max_distribution_samples)
    patch_sizes = _BoundedDistribution(max_distribution_samples)
    rejected_rows: list[dict[str, Any]] = []
    rejected_row_count = 0
    logical_rows_digest = hashlib.sha256()
    row_count = 0

    with _disk_backed_identity_index() as identities:
        for index, row in enumerate(rows):
            row_count += 1
            row_identity = stable_model_id(row)
            logical_rows_digest.update(row_identity.encode("ascii"))
            logical_rows_digest.update(b"\n")
            for raw_key, value in row.items():
                key = str(raw_key)
                schema_fields.add(key)
                schema_field_types.setdefault(key, set()).add(_json_field_type(value))
            instance_id = _instance_id(row, index)
            identities.execute(
                "INSERT INTO instance_ids (value, count) VALUES (?, 1) "
                "ON CONFLICT(value) DO UPDATE SET count = count + 1",
                (instance_id,),
            )
            language = str(row.get("language") or "UNKNOWN")
            languages[language] += 1
            parsed_resolved = parse_resolution_status(row.get("resolved"))
            if parsed_resolved is None:
                resolved["unknown"] += 1
            else:
                resolved["resolved" if parsed_resolved else "unresolved"] += 1

            raw_repository = row.get("repo") or row.get("repository")
            if isinstance(raw_repository, str):
                try:
                    repository = canonical_repository(raw_repository)
                except ValueError:
                    repository = raw_repository.strip().lower()
                identities.execute(
                    "INSERT OR IGNORE INTO repositories (value) VALUES (?)",
                    (repository,),
                )

            trajectory = row.get("trajectory")
            trajectory_lengths.add(
                len(trajectory) if isinstance(trajectory, list) else 0,
                identity=f"{index}:{row_identity}:trajectory",
            )
            patch_size = _patch_size(row)
            if patch_size is not None:
                patch_sizes.add(
                    patch_size,
                    identity=f"{index}:{row_identity}:patch",
                )

            raw_license = row.get("license")
            decision = evaluate_license(raw_license if isinstance(raw_license, str) else None)
            licenses[decision.disposition.value] += 1
            if decision.disposition is not LicenseDisposition.ALLOW:
                rejected_row_count += 1
                if len(rejected_rows) < max_rejected_examples:
                    rejected_rows.append(
                        {
                            "row_index": index,
                            "instance_id": instance_id,
                            "repository": raw_repository,
                            "raw_license": raw_license,
                            "disposition": decision.disposition.value,
                            "reason": decision.reason,
                        }
                    )

        repository_row = identities.execute("SELECT COUNT(*) FROM repositories").fetchone()
        duplicate_row = identities.execute(
            "SELECT COUNT(*) FROM instance_ids WHERE count > 1"
        ).fetchone()
        if repository_row is None or duplicate_row is None:  # pragma: no cover - SQLite contract
            raise RuntimeError("identity index count query returned no row")
        unique_repository_count = int(repository_row[0])
        duplicate_instance_id_count = int(duplicate_row[0])
        duplicate_instance_ids = tuple(
            str(row[0])
            for row in identities.execute(
                "SELECT value FROM instance_ids WHERE count > 1 ORDER BY value LIMIT ?",
                (max_duplicate_examples,),
            )
        )

    matches = (
        source.observed_rows == row_count
        if expect_complete_snapshot and source.observed_rows is not None
        else None
    )
    issues: list[str] = []
    if matches is False:
        issues.append(f"row count drift: registry={source.observed_rows} observed={row_count}")
    if unique_repository_count == 0 and row_count:
        issues.append("no repository identity found in audited rows")

    return DatasetAuditReport(
        status=(
            VerificationStatus.FAIL
            if issues
            else (
                VerificationStatus.PASS
                if source.status is VerificationStatus.PASS
                else VerificationStatus.UNVERIFIED
            )
        ),
        source_name=source.name,
        source_repository_id=source.repository_id,
        source_revision=source.revision,
        input_sha256=input_sha256,
        input_bytes=input_bytes,
        input_scope="complete-snapshot" if expect_complete_snapshot else "partial-snapshot",
        logical_rows_sha256=logical_rows_digest.hexdigest(),
        row_count=row_count,
        declared_row_count=source.observed_rows,
        matches_declared_row_count=matches,
        schema_fields=tuple(sorted(schema_fields)),
        schema_field_types={
            field: tuple(sorted(schema_field_types[field])) for field in sorted(schema_field_types)
        },
        language_distribution=dict(sorted(languages.items())),
        resolved_distribution={
            "resolved": resolved["resolved"],
            "unresolved": resolved["unresolved"],
            "unknown": resolved["unknown"],
        },
        unique_repositories=unique_repository_count,
        duplicate_instance_id_count=duplicate_instance_id_count,
        duplicate_instance_ids=duplicate_instance_ids,
        duplicate_instance_id_sample_cap=max_duplicate_examples,
        duplicate_instance_ids_truncated=(
            duplicate_instance_id_count > len(duplicate_instance_ids)
        ),
        trajectory_lengths=trajectory_lengths.summary(),
        patch_sizes=patch_sizes.summary(),
        distribution_sample_cap=max_distribution_samples,
        distribution_percentiles_approximate=(
            trajectory_lengths.approximate or patch_sizes.approximate
        ),
        license_distribution={
            disposition.value: licenses[disposition.value] for disposition in LicenseDisposition
        },
        rejected_row_count=rejected_row_count,
        rejected_rows=tuple(rejected_rows),
        rejected_rows_truncated=rejected_row_count > len(rejected_rows),
        issues=tuple(issues),
    )
