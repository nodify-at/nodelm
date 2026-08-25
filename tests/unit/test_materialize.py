from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from nodelm.datasets.materialize import discover_snapshot_files, iter_snapshot_rows


def test_snapshot_rows_stream_from_sorted_jsonl_and_parquet_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    parquet.write_table(
        pa.table({"instance_id": ["two"], "resolved": [False]}),
        snapshot / "b.parquet",
    )
    (snapshot / "a.jsonl").write_text(
        json.dumps({"instance_id": "one", "resolved": True}) + "\n",
        encoding="utf-8",
    )
    (snapshot / "README.md").write_text("ignored", encoding="utf-8")

    files = discover_snapshot_files(snapshot)
    rows = tuple(iter_snapshot_rows(files))

    assert [path.name for path in files] == ["a.jsonl", "b.parquet"]
    assert [row["instance_id"] for row in rows] == ["one", "two"]


def test_snapshot_rows_project_nested_columns_without_loading_other_fields(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    parquet.write_table(
        pa.table(
            {
                "instance_id": ["one"],
                "trajectory": [[{"role": "assistant", "content": "large trace"}]],
                "metadata": [
                    {
                        "model_patch": {"patch": "diff --git a/a.ts b/a.ts\n+fix();\n"},
                        "reference_patch": {"patch": "must not cross projection"},
                    }
                ],
            }
        ),
        snapshot / "data.parquet",
    )

    rows = tuple(
        iter_snapshot_rows(
            discover_snapshot_files(snapshot),
            columns=("instance_id", "metadata.model_patch.patch"),
        )
    )

    assert rows == (
        {
            "instance_id": "one",
            "metadata": {"model_patch": {"patch": "diff --git a/a.ts b/a.ts\n+fix();\n"}},
        },
    )


def test_parquet_projection_rejects_missing_nested_columns(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    parquet.write_table(
        pa.table(
            {
                "instance_id": ["one"],
                "metadata": [{"model_patch": {"other": "not-the-requested-patch"}}],
            }
        ),
        snapshot / "data.parquet",
    )

    with pytest.raises(
        ValueError,
        match=r"missing projected column.*metadata\.model_patch\.patch",
    ):
        tuple(
            iter_snapshot_rows(
                discover_snapshot_files(snapshot),
                columns=("instance_id", "metadata.model_patch.patch"),
            )
        )


def test_snapshot_patterns_cannot_escape_the_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (tmp_path / "outside.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contained relative glob"):
        discover_snapshot_files(snapshot, patterns=("../outside.jsonl",))


def test_snapshot_discovery_fails_when_no_supported_data_exists(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("metadata only", encoding="utf-8")

    with pytest.raises(ValueError, match="no supported data files"):
        discover_snapshot_files(tmp_path)


def test_snapshot_discovery_rejects_supported_symlink_outside_root(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    linked = snapshot / "linked.jsonl"
    try:
        linked.symlink_to(outside)
    except (NotImplementedError, OSError) as error:  # pragma: no cover - platform capability
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="resolves outside snapshot"):
        discover_snapshot_files(snapshot)
