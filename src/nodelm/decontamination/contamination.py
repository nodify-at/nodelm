from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from nodelm.decontamination.fingerprints import (
    canonical_repository,
    exact_fingerprint,
    is_near_duplicate,
    normalize_fingerprint_text,
)
from nodelm.models import stable_model_id

ContentKind = Literal["task", "patch"]
SplitName = Literal["train", "evaluation", "excluded"]


@dataclass(frozen=True)
class ContaminationSample:
    sample_id: str
    repository: str
    task_text: str
    patch_texts: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkEntry:
    benchmark_id: str
    task_text: str
    patch_text: str


@dataclass(frozen=True)
class TaskDecontaminationMetadata:
    repository: str
    task_text: str
    reference_patch: str


TaskMetadataLookup = Callable[[str], TaskDecontaminationMetadata | None]


class DecontaminationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.decontamination-evidence/v1"] = (
        "nodelm.decontamination-evidence/v1"
    )
    near_duplicate_threshold: float = Field(gt=0.0, le=1.0)
    task_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=0)
    unique_task_count: int = Field(ge=0)
    unique_patch_count: int = Field(ge=0)
    benchmark_entry_count: int = Field(ge=0)
    unique_benchmark_task_count: int = Field(ge=0)
    unique_benchmark_patch_count: int = Field(ge=0)
    exact_task_duplicate_group_count: int = Field(ge=0)
    exact_patch_duplicate_group_count: int = Field(ge=0)
    near_task_comparison_count: int = Field(ge=0)
    near_patch_comparison_count: int = Field(ge=0)
    near_task_duplicate_pair_count: int = Field(ge=0)
    near_patch_duplicate_pair_count: int = Field(ge=0)
    benchmark_exact_task_fingerprint_count: int = Field(ge=0)
    benchmark_exact_patch_fingerprint_count: int = Field(ge=0)
    benchmark_task_comparison_count: int = Field(ge=0)
    benchmark_patch_comparison_count: int = Field(ge=0)
    benchmark_near_task_pair_count: int = Field(ge=0)
    benchmark_near_patch_pair_count: int = Field(ge=0)
    benchmark_overlap_sample_count: int = Field(ge=0)
    benchmark_overlap_repository_count: int = Field(ge=0)
    contamination_group_count: int = Field(ge=0)
    multi_repository_group_count: int = Field(ge=0)
    excluded_group_count: int = Field(ge=0)
    excluded_repository_count: int = Field(ge=0)


@dataclass(frozen=True)
class ContaminationIndex:
    connection: sqlite3.Connection
    config_digest: str
    evidence: DecontaminationEvidence

    @property
    def sample_count(self) -> int:
        return _scalar(self.connection, "SELECT COUNT(*) FROM samples")

    def iter_assignments(self) -> Iterator[tuple[str, str, str, SplitName]]:
        for row in self.connection.execute(
            "SELECT r.parent, s.repository, s.sample_id, r.split "
            "FROM samples AS s JOIN repositories AS r USING (repository) "
            "ORDER BY s.sample_id, s.repository, r.split"
        ):
            yield str(row[0]), str(row[1]), str(row[2]), cast(SplitName, str(row[3]))

    def iter_repositories(self, split: SplitName) -> Iterator[str]:
        for row in self.connection.execute(
            "SELECT repository FROM repositories WHERE split = ? ORDER BY repository",
            (split,),
        ):
            yield str(row[0])


def _required_text(row: Mapping[str, object], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"required field is missing: {' or '.join(fields)}")


@contextmanager
def decontamination_task_metadata_index(
    rows: Iterable[Mapping[str, object]],
    *,
    temp_directory: Path,
) -> Iterator[TaskMetadataLookup]:
    """Index task text and reference patches on disk solely for the split gate."""

    temp_directory.mkdir(parents=True, exist_ok=True)
    descriptor, database_name = tempfile.mkstemp(
        prefix=".nodelm-decontamination-tasks-",
        suffix=".sqlite3",
        dir=temp_directory,
    )
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE tasks ("
            "task_id TEXT PRIMARY KEY NOT NULL, repository TEXT NOT NULL, "
            "task_text TEXT NOT NULL, reference_patch TEXT NOT NULL) WITHOUT ROWID"
        )
        for row in rows:
            task_id = _required_text(row, "instance_id", "task_id")
            values = (
                canonical_repository(_required_text(row, "repo", "repository")),
                _required_text(
                    row,
                    "problem_statement",
                    "task",
                    "task_description",
                    "pr_description",
                ),
                _required_text(row, "patch", "reference_patch"),
            )
            existing = connection.execute(
                "SELECT repository, task_text, reference_patch FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if existing is not None:
                if tuple(str(item) for item in existing) != values:
                    raise ValueError(f"conflicting decontamination task metadata: {task_id}")
                continue
            connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?)",
                (task_id, *values),
            )
        connection.commit()

        def lookup(task_id: str) -> TaskDecontaminationMetadata | None:
            result = connection.execute(
                "SELECT repository, task_text, reference_patch FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if result is None:
                return None
            return TaskDecontaminationMetadata(
                repository=str(result[0]),
                task_text=str(result[1]),
                reference_patch=str(result[2]),
            )

        yield lookup
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()


def _validate_digest(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _repository_bucket(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE repositories (
            repository TEXT PRIMARY KEY NOT NULL,
            parent TEXT NOT NULL,
            split TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE samples (
            sample_id TEXT PRIMARY KEY NOT NULL,
            repository TEXT NOT NULL,
            benchmark_overlap INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID;
        CREATE TABLE content_values (
            kind TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            text TEXT NOT NULL,
            length INTEGER NOT NULL,
            PRIMARY KEY (kind, fingerprint)
        ) WITHOUT ROWID;
        CREATE INDEX content_values_length ON content_values (kind, length, fingerprint);
        CREATE TABLE value_repositories (
            kind TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            repository TEXT NOT NULL,
            PRIMARY KEY (kind, fingerprint, repository)
        ) WITHOUT ROWID;
        CREATE TABLE value_samples (
            kind TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            PRIMARY KEY (kind, fingerprint, sample_id)
        ) WITHOUT ROWID;
        CREATE TABLE benchmark_ids (benchmark_id TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID;
        CREATE TABLE benchmark_values (
            kind TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            text TEXT NOT NULL,
            length INTEGER NOT NULL,
            PRIMARY KEY (kind, fingerprint)
        ) WITHOUT ROWID;
        CREATE INDEX benchmark_values_length ON benchmark_values (kind, length, fingerprint);
        CREATE TABLE excluded_roots (root TEXT PRIMARY KEY NOT NULL) WITHOUT ROWID;
        """
    )


def _insert_content_value(
    connection: sqlite3.Connection,
    *,
    kind: ContentKind,
    text: str,
    sample_id: str,
    repository: str,
) -> None:
    normalized = normalize_fingerprint_text(text)
    if not normalized:
        raise ValueError(f"{kind} text must not be empty")
    fingerprint = exact_fingerprint(normalized)
    connection.execute(
        "INSERT OR IGNORE INTO content_values VALUES (?, ?, ?, ?)",
        (kind, fingerprint, normalized, len(normalized)),
    )
    stored = connection.execute(
        "SELECT text FROM content_values WHERE kind = ? AND fingerprint = ?",
        (kind, fingerprint),
    ).fetchone()
    if stored is None or str(stored[0]) != normalized:
        raise ValueError(f"{kind} fingerprint collision detected")
    connection.execute(
        "INSERT OR IGNORE INTO value_repositories VALUES (?, ?, ?)",
        (kind, fingerprint, repository),
    )
    connection.execute(
        "INSERT OR IGNORE INTO value_samples VALUES (?, ?, ?)",
        (kind, fingerprint, sample_id),
    )


def _insert_benchmark_value(
    connection: sqlite3.Connection,
    *,
    kind: ContentKind,
    text: str,
) -> None:
    normalized = normalize_fingerprint_text(text)
    if not normalized:
        raise ValueError(f"benchmark {kind} text must not be empty")
    fingerprint = exact_fingerprint(normalized)
    connection.execute(
        "INSERT OR IGNORE INTO benchmark_values VALUES (?, ?, ?, ?)",
        (kind, fingerprint, normalized, len(normalized)),
    )
    stored = connection.execute(
        "SELECT text FROM benchmark_values WHERE kind = ? AND fingerprint = ?",
        (kind, fingerprint),
    ).fetchone()
    if stored is None or str(stored[0]) != normalized:
        raise ValueError(f"benchmark {kind} fingerprint collision detected")


def _find_root(connection: sqlite3.Connection, repository: str) -> str:
    path: list[str] = []
    current = repository
    while True:
        row = connection.execute(
            "SELECT parent FROM repositories WHERE repository = ?", (current,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown repository in contamination graph: {current}")
        parent = str(row[0])
        if parent == current:
            root = current
            break
        path.append(current)
        current = parent
    for item in path:
        connection.execute(
            "UPDATE repositories SET parent = ? WHERE repository = ?",
            (root, item),
        )
    return root


def _union_repositories(connection: sqlite3.Connection, repositories: Iterable[str]) -> None:
    roots = {_find_root(connection, repository) for repository in repositories}
    if len(roots) < 2:
        return
    root = min(roots)
    for child in roots - {root}:
        connection.execute(
            "UPDATE repositories SET parent = ? WHERE parent = ?",
            (root, child),
        )


def _union_value_repositories(
    connection: sqlite3.Connection,
    kind: ContentKind,
    fingerprints: tuple[str, ...],
) -> None:
    placeholders = ",".join("?" for _ in fingerprints)
    repositories = (
        str(row[0])
        for row in connection.execute(
            f"SELECT DISTINCT repository FROM value_repositories "
            f"WHERE kind = ? AND fingerprint IN ({placeholders}) ORDER BY repository",
            (kind, *fingerprints),
        )
    )
    _union_repositories(connection, repositories)


def _union_exact_groups(connection: sqlite3.Connection, kind: ContentKind) -> int:
    count = 0
    for row in connection.execute(
        "SELECT fingerprint FROM value_repositories WHERE kind = ? "
        "GROUP BY fingerprint HAVING COUNT(*) > 1 ORDER BY fingerprint",
        (kind,),
    ):
        count += 1
        _union_value_repositories(connection, kind, (str(row[0]),))
    return count


def _candidate_length_bounds(length: int, threshold: float) -> tuple[int, int]:
    if threshold == 0.0:
        return 0, 2**63 - 1
    minimum = max(0, math.floor(length * threshold / (2.0 - threshold)))
    maximum = math.ceil(length * (2.0 / threshold - 1.0))
    return minimum, maximum


def _union_near_groups(
    connection: sqlite3.Connection,
    kind: ContentKind,
    threshold: float,
) -> tuple[int, int]:
    comparisons = 0
    matches = 0
    left_cursor = connection.execute(
        "SELECT fingerprint, text, length FROM content_values WHERE kind = ? ORDER BY fingerprint",
        (kind,),
    )
    for left_fingerprint, left_text, left_length in left_cursor:
        minimum, maximum = _candidate_length_bounds(int(left_length), threshold)
        for right_fingerprint, right_text in connection.execute(
            "SELECT fingerprint, text FROM content_values "
            "WHERE kind = ? AND fingerprint > ? AND length BETWEEN ? AND ? "
            "ORDER BY fingerprint",
            (kind, str(left_fingerprint), minimum, maximum),
        ):
            comparisons += 1
            if not is_near_duplicate(str(left_text), str(right_text), threshold=threshold):
                continue
            matches += 1
            _union_value_repositories(
                connection,
                kind,
                (str(left_fingerprint), str(right_fingerprint)),
            )
    return comparisons, matches


def _mark_value_samples(
    connection: sqlite3.Connection,
    kind: ContentKind,
    fingerprint: str,
) -> None:
    connection.execute(
        "UPDATE samples SET benchmark_overlap = 1 WHERE sample_id IN ("
        "SELECT sample_id FROM value_samples WHERE kind = ? AND fingerprint = ?)",
        (kind, fingerprint),
    )


def _mark_exact_benchmark_overlap(connection: sqlite3.Connection, kind: ContentKind) -> int:
    count = 0
    for row in connection.execute(
        "SELECT c.fingerprint FROM content_values AS c "
        "JOIN benchmark_values AS b USING (kind, fingerprint) "
        "WHERE c.kind = ? ORDER BY c.fingerprint",
        (kind,),
    ):
        count += 1
        _mark_value_samples(connection, kind, str(row[0]))
    return count


def _mark_near_benchmark_overlap(
    connection: sqlite3.Connection,
    kind: ContentKind,
    threshold: float,
) -> tuple[int, int]:
    comparisons = 0
    matches = 0
    for fingerprint, text, length in connection.execute(
        "SELECT fingerprint, text, length FROM content_values WHERE kind = ? ORDER BY fingerprint",
        (kind,),
    ):
        minimum, maximum = _candidate_length_bounds(int(length), threshold)
        for _benchmark_fingerprint, benchmark_text in connection.execute(
            "SELECT fingerprint, text FROM benchmark_values "
            "WHERE kind = ? AND fingerprint != ? AND length BETWEEN ? AND ? "
            "ORDER BY fingerprint",
            (kind, str(fingerprint), minimum, maximum),
        ):
            comparisons += 1
            if not is_near_duplicate(str(text), str(benchmark_text), threshold=threshold):
                continue
            matches += 1
            _mark_value_samples(connection, kind, str(fingerprint))
    return comparisons, matches


def _compress_repository_groups(connection: sqlite3.Connection) -> None:
    cursor = connection.execute("SELECT repository FROM repositories ORDER BY repository")
    while rows := cursor.fetchmany(1_024):
        for row in rows:
            repository = str(row[0])
            root = _find_root(connection, repository)
            connection.execute(
                "UPDATE repositories SET parent = ? WHERE repository = ?",
                (root, repository),
            )


def _assign_splits(
    connection: sqlite3.Connection,
    *,
    seed: int,
    evaluation_fraction: float,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO excluded_roots "
        "SELECT DISTINCT r.parent FROM samples AS s "
        "JOIN repositories AS r USING (repository) WHERE s.benchmark_overlap = 1"
    )
    connection.execute(
        "UPDATE repositories SET split = 'excluded' "
        "WHERE parent IN (SELECT root FROM excluded_roots)"
    )
    for row in connection.execute(
        "SELECT DISTINCT parent FROM repositories "
        "WHERE parent NOT IN (SELECT root FROM excluded_roots) ORDER BY parent"
    ):
        root = str(row[0])
        split = "evaluation" if _repository_bucket(root, seed) < evaluation_fraction else "train"
        connection.execute(
            "UPDATE repositories SET split = ? WHERE parent = ?",
            (split, root),
        )


def _scalar(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise RuntimeError("contamination count query returned no row")
    return int(row[0])


@contextmanager
def build_contamination_index(
    samples: Iterable[ContaminationSample],
    *,
    benchmarks: Iterable[BenchmarkEntry],
    near_duplicate_threshold: float,
    aliases: Mapping[str, str],
    task_metadata_sha256: str,
    benchmark_sha256: str,
    seed: int,
    evaluation_fraction: float,
    temp_directory: Path,
) -> Iterator[ContaminationIndex]:
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be greater than 0 and at most 1")
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1")
    _validate_digest(task_metadata_sha256, "task_metadata_sha256")
    _validate_digest(benchmark_sha256, "benchmark_sha256")

    temp_directory.mkdir(parents=True, exist_ok=True)
    descriptor, database_name = tempfile.mkstemp(
        prefix=".nodelm-contamination-",
        suffix=".sqlite3",
        dir=temp_directory,
    )
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        _initialize_database(connection)
        for sample in samples:
            if not sample.sample_id:
                raise ValueError("sample_id must not be empty")
            repository = canonical_repository(sample.repository)
            repository = aliases.get(repository, repository)
            connection.execute(
                "INSERT OR IGNORE INTO repositories VALUES (?, ?, '')",
                (repository, repository),
            )
            try:
                connection.execute(
                    "INSERT INTO samples (sample_id, repository) VALUES (?, ?)",
                    (sample.sample_id, repository),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"duplicate sample_id: {sample.sample_id}") from None
            _insert_content_value(
                connection,
                kind="task",
                text=sample.task_text,
                sample_id=sample.sample_id,
                repository=repository,
            )
            if not sample.patch_texts:
                raise ValueError(f"sample has no patch text: {sample.sample_id}")
            seen_patch_fingerprints: set[str] = set()
            for patch_text in sample.patch_texts:
                fingerprint = exact_fingerprint(patch_text)
                if fingerprint in seen_patch_fingerprints:
                    continue
                seen_patch_fingerprints.add(fingerprint)
                _insert_content_value(
                    connection,
                    kind="patch",
                    text=patch_text,
                    sample_id=sample.sample_id,
                    repository=repository,
                )

        if _scalar(connection, "SELECT COUNT(*) FROM samples") == 0:
            raise ValueError("split input contains no samples")

        benchmark_entry_count = 0
        for benchmark in benchmarks:
            if not benchmark.benchmark_id:
                raise ValueError("benchmark_id must not be empty")
            try:
                connection.execute(
                    "INSERT INTO benchmark_ids VALUES (?)", (benchmark.benchmark_id,)
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"duplicate benchmark_id: {benchmark.benchmark_id}") from None
            benchmark_entry_count += 1
            _insert_benchmark_value(connection, kind="task", text=benchmark.task_text)
            _insert_benchmark_value(connection, kind="patch", text=benchmark.patch_text)
        if benchmark_entry_count == 0:
            raise ValueError("public benchmark input contains no entries")

        exact_task_groups = _union_exact_groups(connection, "task")
        exact_patch_groups = _union_exact_groups(connection, "patch")
        task_comparisons, near_task_pairs = _union_near_groups(
            connection, "task", near_duplicate_threshold
        )
        patch_comparisons, near_patch_pairs = _union_near_groups(
            connection, "patch", near_duplicate_threshold
        )
        benchmark_exact_tasks = _mark_exact_benchmark_overlap(connection, "task")
        benchmark_exact_patches = _mark_exact_benchmark_overlap(connection, "patch")
        benchmark_task_comparisons, benchmark_near_tasks = _mark_near_benchmark_overlap(
            connection, "task", near_duplicate_threshold
        )
        benchmark_patch_comparisons, benchmark_near_patches = _mark_near_benchmark_overlap(
            connection, "patch", near_duplicate_threshold
        )
        _compress_repository_groups(connection)
        _assign_splits(
            connection,
            seed=seed,
            evaluation_fraction=evaluation_fraction,
        )
        connection.commit()

        evidence = DecontaminationEvidence(
            near_duplicate_threshold=near_duplicate_threshold,
            task_metadata_sha256=task_metadata_sha256,
            benchmark_sha256=benchmark_sha256,
            sample_count=_scalar(connection, "SELECT COUNT(*) FROM samples"),
            unique_task_count=_scalar(
                connection, "SELECT COUNT(*) FROM content_values WHERE kind = 'task'"
            ),
            unique_patch_count=_scalar(
                connection, "SELECT COUNT(*) FROM content_values WHERE kind = 'patch'"
            ),
            benchmark_entry_count=benchmark_entry_count,
            unique_benchmark_task_count=_scalar(
                connection, "SELECT COUNT(*) FROM benchmark_values WHERE kind = 'task'"
            ),
            unique_benchmark_patch_count=_scalar(
                connection, "SELECT COUNT(*) FROM benchmark_values WHERE kind = 'patch'"
            ),
            exact_task_duplicate_group_count=exact_task_groups,
            exact_patch_duplicate_group_count=exact_patch_groups,
            near_task_comparison_count=task_comparisons,
            near_patch_comparison_count=patch_comparisons,
            near_task_duplicate_pair_count=near_task_pairs,
            near_patch_duplicate_pair_count=near_patch_pairs,
            benchmark_exact_task_fingerprint_count=benchmark_exact_tasks,
            benchmark_exact_patch_fingerprint_count=benchmark_exact_patches,
            benchmark_task_comparison_count=benchmark_task_comparisons,
            benchmark_patch_comparison_count=benchmark_patch_comparisons,
            benchmark_near_task_pair_count=benchmark_near_tasks,
            benchmark_near_patch_pair_count=benchmark_near_patches,
            benchmark_overlap_sample_count=_scalar(
                connection, "SELECT COUNT(*) FROM samples WHERE benchmark_overlap = 1"
            ),
            benchmark_overlap_repository_count=_scalar(
                connection,
                "SELECT COUNT(DISTINCT repository) FROM samples WHERE benchmark_overlap = 1",
            ),
            contamination_group_count=_scalar(
                connection, "SELECT COUNT(DISTINCT parent) FROM repositories"
            ),
            multi_repository_group_count=_scalar(
                connection,
                "SELECT COUNT(*) FROM (SELECT parent FROM repositories "
                "GROUP BY parent HAVING COUNT(*) > 1)",
            ),
            excluded_group_count=_scalar(connection, "SELECT COUNT(*) FROM excluded_roots"),
            excluded_repository_count=_scalar(
                connection, "SELECT COUNT(*) FROM repositories WHERE split = 'excluded'"
            ),
        )
        config_digest = stable_model_id(
            {
                "aliases": dict(sorted(aliases.items())),
                "benchmark_sha256": benchmark_sha256,
                "evaluation_fraction": evaluation_fraction,
                "near_duplicate_threshold": near_duplicate_threshold,
                "seed": seed,
                "task_metadata_sha256": task_metadata_sha256,
            }
        )
        yield ContaminationIndex(
            connection=connection,
            config_digest=config_digest,
            evidence=evidence,
        )
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()
