from __future__ import annotations

from pathlib import Path

import pytest

from nodelm.provenance.trace_projection import (
    TraceNormalizationProjection,
    TraceProjectionError,
    trace_normalization_projection,
)


def _add(
    projection: TraceNormalizationProjection,
    row_index: int,
    *,
    key: str | None,
    raw: str,
    sample: str | None,
    reason_code: str | None = None,
) -> None:
    projection.add_row(
        row_index=row_index,
        rollout_key=key,
        raw_row_sha256=raw,
        sample_id=sample,
        reason_code=reason_code,
        reason=f"reason:{reason_code}" if reason_code else None,
        instance_id=f"instance-{row_index}",
        repository="acme/widget",
        rollout_id=f"rollout-{row_index}",
    )


def test_trace_projection_deduplicates_exact_rows_and_allows_distinct_rollouts() -> None:
    with trace_normalization_projection() as projection:
        _add(projection, 0, key="key-a", raw="raw-a", sample="sample-a")
        _add(projection, 1, key="key-a", raw="raw-a", sample="sample-a")
        _add(projection, 2, key="key-b", raw="raw-a", sample="sample-b")
        projection.finalize()

        decisions = tuple(projection.iter_decisions())

        assert [decision.admitted for decision in decisions] == [True, False, True]
        assert decisions[1].reason_code == "duplicate_trace_row"
        assert projection.admitted_count == 2
        assert projection.rejected_count == 1
        assert projection.duplicate_trace_row_count == 1
        assert projection.unique_rollout_key_count == 2


def test_trace_projection_retroactively_rejects_a_a_b_conflict() -> None:
    with trace_normalization_projection() as projection:
        _add(projection, 0, key="key-a", raw="raw-a", sample="sample-a")
        _add(projection, 1, key="key-a", raw="raw-a", sample="sample-a")
        _add(projection, 2, key="key-a", raw="raw-b", sample="sample-b")
        projection.finalize()

        decisions = tuple(projection.iter_decisions())

        assert not any(decision.admitted for decision in decisions)
        assert {decision.reason_code for decision in decisions} == {"conflicting_rollout_identity"}
        assert projection.conflicting_rollout_identity_count == 1
        assert projection.conflicting_rollout_row_count == 3


def test_trace_projection_unknown_and_valid_same_key_conflict_with_cause() -> None:
    with trace_normalization_projection() as projection:
        _add(
            projection,
            0,
            key="key-a",
            raw="raw-unknown",
            sample=None,
            reason_code="unknown_resolution",
        )
        _add(projection, 1, key="key-a", raw="raw-valid", sample="sample-valid")
        projection.finalize()

        decisions = tuple(projection.iter_decisions())

        assert decisions[0].cause_code == "unknown_resolution"
        assert {decision.reason_code for decision in decisions} == {"conflicting_rollout_identity"}


def test_trace_projection_retains_singleton_unknown_resolution() -> None:
    with trace_normalization_projection() as projection:
        _add(
            projection,
            0,
            key="key-a",
            raw="raw-unknown",
            sample=None,
            reason_code="unknown_resolution",
        )
        projection.finalize()

        decision = next(projection.iter_decisions())

        assert decision.reason_code == "unknown_resolution"
        assert projection.rejection_counts_by_code == {"unknown_resolution": 1}


def test_trace_projection_rejects_sample_id_collision_across_rollout_keys() -> None:
    with trace_normalization_projection() as projection:
        _add(projection, 0, key="key-a", raw="raw-a", sample="same-sample")
        _add(projection, 1, key="key-b", raw="raw-b", sample="same-sample")

        with pytest.raises(TraceProjectionError, match="multiple rollout keys"):
            projection.finalize()


def test_trace_projection_rejects_multiple_samples_for_same_exact_raw_rollout() -> None:
    with trace_normalization_projection() as projection:
        _add(projection, 0, key="key-a", raw="raw-a", sample="sample-a")
        _add(projection, 1, key="key-a", raw="raw-a", sample="sample-b")

        with pytest.raises(TraceProjectionError, match="multiple normalized sample IDs"):
            projection.finalize()


def test_trace_projection_is_disk_backed_bounded_and_cleaned_up(tmp_path: Path) -> None:
    with trace_normalization_projection(temp_directory=tmp_path) as projection:
        for row_index in range(1_000):
            _add(
                projection,
                row_index,
                key=f"key-{row_index}",
                raw=f"raw-{row_index}",
                sample=f"sample-{row_index}",
            )
        projection.finalize()
        database_path = projection.database_path
        cache_size = int(projection.connection.execute("PRAGMA cache_size").fetchone()[0])
        temp_store = int(projection.connection.execute("PRAGMA temp_store").fetchone()[0])
        columns = {
            str(row[1])
            for row in projection.connection.execute("PRAGMA table_info(rows)").fetchall()
        }

        assert database_path.exists()
        assert cache_size == -8192
        assert temp_store == 1  # SQLite FILE
        assert "payload" not in columns
        assert "trajectory" not in columns

    assert not database_path.exists()


def test_trace_projection_cleans_up_after_error(tmp_path: Path) -> None:
    database_path: Path | None = None
    with (
        pytest.raises(RuntimeError, match="boom"),
        trace_normalization_projection(temp_directory=tmp_path) as projection,
    ):
        database_path = projection.database_path
        raise RuntimeError("boom")

    assert database_path is not None
    assert not database_path.exists()
