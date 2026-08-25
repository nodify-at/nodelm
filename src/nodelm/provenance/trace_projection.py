from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path


class TraceProjectionError(ValueError):
    """Trace rollout classification cannot produce a unique fail-closed projection."""


@dataclass(frozen=True)
class TraceRowDecision:
    row_index: int
    rollout_key: str | None
    raw_row_sha256: str
    sample_id: str | None
    admitted: bool
    reason_code: str | None
    cause_code: str | None
    reason: str | None
    instance_id: str | None
    repository: str | None
    rollout_id: str | None


@dataclass
class TraceNormalizationProjection:
    connection: sqlite3.Connection
    database_path: Path
    finalized: bool = False

    def add_row(
        self,
        *,
        row_index: int,
        rollout_key: str | None,
        raw_row_sha256: str,
        sample_id: str | None,
        reason_code: str | None,
        reason: str | None,
        instance_id: str | None,
        repository: str | None,
        rollout_id: str | None,
    ) -> None:
        if self.finalized:
            raise TraceProjectionError("cannot add rows after rollout projection finalization")
        if (sample_id is None) != (reason_code is not None):
            raise TraceProjectionError(
                "each trace row must carry exactly one of sample_id or rejection reason"
            )
        self.connection.execute(
            "INSERT INTO rows ("
            "row_index, rollout_key, raw_row_sha256, sample_id, original_reason_code, "
            "original_reason, instance_id, repository, rollout_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row_index,
                rollout_key,
                raw_row_sha256,
                sample_id,
                reason_code,
                reason,
                instance_id,
                repository,
                rollout_id,
            ),
        )

    def finalize(self) -> None:
        if self.finalized:
            raise TraceProjectionError("rollout projection was already finalized")
        invalid_valid_rows = self.connection.execute(
            "SELECT row_index FROM rows WHERE sample_id IS NOT NULL AND rollout_key IS NULL LIMIT 1"
        ).fetchone()
        if invalid_valid_rows is not None:
            raise TraceProjectionError(
                f"normalized row lacks a rollout key: {int(invalid_valid_rows[0])}"
            )
        collision = self.connection.execute(
            "SELECT sample_id FROM rows WHERE sample_id IS NOT NULL "
            "GROUP BY sample_id HAVING COUNT(DISTINCT rollout_key) > 1 LIMIT 1"
        ).fetchone()
        if collision is not None:
            raise TraceProjectionError(f"sample_id maps to multiple rollout keys: {collision[0]!s}")

        self.connection.execute(
            "UPDATE rows SET decision = 'reject', reason_code = original_reason_code, "
            "reason = original_reason WHERE rollout_key IS NULL"
        )
        groups = self.connection.execute(
            "SELECT rollout_key, COUNT(*), COUNT(DISTINCT raw_row_sha256), "
            "SUM(CASE WHEN sample_id IS NOT NULL THEN 1 ELSE 0 END), "
            "COUNT(DISTINCT sample_id) "
            "FROM rows WHERE rollout_key IS NOT NULL GROUP BY rollout_key ORDER BY rollout_key"
        )
        for rollout_key, row_count, raw_digest_count, valid_count, sample_id_count in groups:
            key = str(rollout_key)
            total = int(row_count)
            raw_count = int(raw_digest_count)
            valid = int(valid_count)
            distinct_sample_ids = int(sample_id_count)
            if valid == total and raw_count == 1 and distinct_sample_ids != 1:
                raise TraceProjectionError(
                    f"one rollout key maps to multiple normalized sample IDs: {key}"
                )
            if raw_count > 1 or valid not in {0, total}:
                self.connection.execute(
                    "INSERT INTO conflicting_rollouts VALUES (?)",
                    (key,),
                )
                self.connection.execute(
                    "UPDATE rows SET decision = 'reject', "
                    "cause_code = original_reason_code, "
                    "reason_code = 'conflicting_rollout_identity', "
                    "reason = 'rollout key maps to conflicting raw trace rows' "
                    "WHERE rollout_key = ?",
                    (key,),
                )
                continue
            if valid == 0:
                self.connection.execute(
                    "UPDATE rows SET decision = 'reject', reason_code = original_reason_code, "
                    "reason = original_reason WHERE rollout_key = ?",
                    (key,),
                )
                continue

            admitted_row = int(
                self.connection.execute(
                    "SELECT MIN(row_index) FROM rows WHERE rollout_key = ?",
                    (key,),
                ).fetchone()[0]
            )
            self.connection.execute(
                "UPDATE rows SET decision = 'admit' WHERE row_index = ?",
                (admitted_row,),
            )
            self.connection.execute(
                "UPDATE rows SET decision = 'reject', reason_code = 'duplicate_trace_row', "
                "reason = 'exact trace row repeats an already admitted rollout' "
                "WHERE rollout_key = ? AND row_index <> ?",
                (key, admitted_row),
            )

        undecided = self.connection.execute(
            "SELECT row_index FROM rows WHERE decision IS NULL LIMIT 1"
        ).fetchone()
        if undecided is not None:
            raise TraceProjectionError(f"trace row was not classified: {int(undecided[0])}")
        missing_reason = self.connection.execute(
            "SELECT row_index FROM rows WHERE decision = 'reject' AND reason_code IS NULL LIMIT 1"
        ).fetchone()
        if missing_reason is not None:
            raise TraceProjectionError(
                f"rejected trace row lacks a reason: {int(missing_reason[0])}"
            )
        self.connection.commit()
        self.finalized = True

    def iter_decisions(self) -> Iterator[TraceRowDecision]:
        self._require_finalized()
        rows = self.connection.execute(
            "SELECT row_index, rollout_key, raw_row_sha256, sample_id, decision, "
            "reason_code, cause_code, reason, instance_id, repository, rollout_id "
            "FROM rows ORDER BY row_index"
        )
        for row in rows:
            yield TraceRowDecision(
                row_index=int(row[0]),
                rollout_key=str(row[1]) if row[1] is not None else None,
                raw_row_sha256=str(row[2]),
                sample_id=str(row[3]) if row[3] is not None else None,
                admitted=str(row[4]) == "admit",
                reason_code=str(row[5]) if row[5] is not None else None,
                cause_code=str(row[6]) if row[6] is not None else None,
                reason=str(row[7]) if row[7] is not None else None,
                instance_id=str(row[8]) if row[8] is not None else None,
                repository=str(row[9]) if row[9] is not None else None,
                rollout_id=str(row[10]) if row[10] is not None else None,
            )

    @property
    def input_row_count(self) -> int:
        return self._count("SELECT COUNT(*) FROM rows")

    @property
    def admitted_count(self) -> int:
        self._require_finalized()
        return self._count("SELECT COUNT(*) FROM rows WHERE decision = 'admit'")

    @property
    def rejected_count(self) -> int:
        self._require_finalized()
        return self._count("SELECT COUNT(*) FROM rows WHERE decision = 'reject'")

    @property
    def unique_rollout_key_count(self) -> int:
        return self._count(
            "SELECT COUNT(DISTINCT rollout_key) FROM rows WHERE rollout_key IS NOT NULL"
        )

    @property
    def duplicate_trace_row_count(self) -> int:
        self._require_finalized()
        return self._count("SELECT COUNT(*) FROM rows WHERE reason_code = 'duplicate_trace_row'")

    @property
    def conflicting_rollout_identity_count(self) -> int:
        self._require_finalized()
        return self._count("SELECT COUNT(*) FROM conflicting_rollouts")

    @property
    def conflicting_rollout_row_count(self) -> int:
        self._require_finalized()
        return self._count(
            "SELECT COUNT(*) FROM rows WHERE reason_code = 'conflicting_rollout_identity'"
        )

    @property
    def rejection_counts_by_code(self) -> dict[str, int]:
        self._require_finalized()
        rows = self.connection.execute(
            "SELECT reason_code, COUNT(*) FROM rows WHERE decision = 'reject' "
            "GROUP BY reason_code ORDER BY reason_code"
        )
        return {str(reason_code): int(count) for reason_code, count in rows}

    def _count(self, query: str) -> int:
        result = self.connection.execute(query).fetchone()
        if result is None:  # pragma: no cover - aggregate SQL contract
            raise TraceProjectionError("trace projection aggregate returned no row")
        return int(result[0])

    def _require_finalized(self) -> None:
        if not self.finalized:
            raise TraceProjectionError("rollout projection must be finalized first")


@contextmanager
def trace_normalization_projection(
    *,
    temp_directory: Path | None = None,
) -> Iterator[TraceNormalizationProjection]:
    """Create a bounded, disk-backed trace rollout classification index."""

    descriptor, database_name = tempfile.mkstemp(
        prefix="nodelm-trace-projection-",
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
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "CREATE TABLE rows ("
            "row_index INTEGER PRIMARY KEY NOT NULL, rollout_key TEXT, "
            "raw_row_sha256 TEXT NOT NULL, sample_id TEXT, "
            "original_reason_code TEXT, original_reason TEXT, "
            "instance_id TEXT, repository TEXT, rollout_id TEXT, "
            "decision TEXT, reason_code TEXT, cause_code TEXT, reason TEXT"
            ")"
        )
        connection.execute("CREATE INDEX rows_rollout_key ON rows (rollout_key, row_index)")
        connection.execute("CREATE INDEX rows_sample_id ON rows (sample_id)")
        connection.execute(
            "CREATE TABLE conflicting_rollouts (rollout_key TEXT PRIMARY KEY NOT NULL) "
            "WITHOUT ROWID"
        )
        yield TraceNormalizationProjection(connection, database_path)
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()
