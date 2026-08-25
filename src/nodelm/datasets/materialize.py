from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

parquet: Any = importlib.import_module("pyarrow.parquet")

SUPPORTED_SNAPSHOT_SUFFIXES = frozenset({".jsonl", ".parquet"})


def discover_snapshot_files(
    snapshot: Path,
    *,
    patterns: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    """Resolve a deterministic, contained set of supported snapshot data files."""

    root = snapshot.resolve()
    candidates: tuple[Path, ...]
    if root.is_file():
        if patterns:
            raise ValueError("file patterns cannot be used when snapshot is a file")
        candidates = (root,)
        containment_root = root.parent
    elif root.is_dir():
        containment_root = root
        if patterns:
            for pattern in patterns:
                pattern_path = Path(pattern)
                if pattern_path.is_absolute() or ".." in pattern_path.parts:
                    raise ValueError(
                        f"snapshot pattern must be a contained relative glob: {pattern}"
                    )
            candidates = tuple(path for pattern in patterns for path in root.glob(pattern))
        else:
            candidates = tuple(root.rglob("*"))
    else:
        raise ValueError(f"snapshot path does not exist: {snapshot}")

    selected: set[Path] = set()
    for candidate in candidates:
        if (
            not candidate.is_file()
            or candidate.suffix.casefold() not in SUPPORTED_SNAPSHOT_SUFFIXES
        ):
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(containment_root):
            raise ValueError(f"snapshot data file resolves outside snapshot: {candidate}")
        selected.add(resolved)
    if not selected:
        suffixes = ", ".join(sorted(SUPPORTED_SNAPSHOT_SUFFIXES))
        raise ValueError(f"snapshot contains no supported data files ({suffixes})")
    return tuple(sorted(selected))


def iter_snapshot_rows(
    paths: tuple[Path, ...],
    *,
    columns: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream records, optionally projecting a deterministic set of nested columns."""

    if not paths:
        raise ValueError("at least one snapshot data file is required")
    if columns is not None and (
        not columns
        or len(columns) != len(set(columns))
        or any(
            not column or column != column.strip() or any(not part for part in column.split("."))
            for column in columns
        )
    ):
        raise ValueError("projected columns must be unique non-empty dotted paths")
    for path in paths:
        suffix = path.suffix.casefold()
        if suffix == ".jsonl":
            yield from _iter_jsonl(path, columns=columns)
        elif suffix == ".parquet":
            yield from _iter_parquet(path, columns=columns)
        else:
            raise ValueError(f"unsupported snapshot data file: {path}")


def _project_mapping(value: Mapping[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for column in columns:
        parts = column.split(".")
        source: Any = value
        for part in parts:
            if not isinstance(source, Mapping) or part not in source:
                raise ValueError(f"snapshot row is missing projected column: {column}")
            source = source[part]
        destination = projected
        for part in parts[:-1]:
            child = destination.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"projected columns overlap incompatibly: {column}")
            destination = child
        destination[parts[-1]] = source
    return projected


def _iter_jsonl(
    path: Path,
    *,
    columns: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"snapshot row is not an object: {path}:{line_number}")
            yield value if columns is None else _project_mapping(value, columns)


def _iter_parquet(
    path: Path,
    *,
    columns: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        source = parquet.ParquetFile(path)
        if columns is not None:
            leaf_paths = tuple(
                source.schema.column(index).path for index in range(len(source.schema))
            )
            missing = tuple(
                column
                for column in columns
                if not any(
                    leaf_path == column or leaf_path.startswith(f"{column}.")
                    for leaf_path in leaf_paths
                )
            )
            if missing:
                raise ValueError("snapshot row is missing projected column: " + ", ".join(missing))
        for batch in source.iter_batches(batch_size=1_024, columns=columns):
            for value in batch.to_pylist():
                if not isinstance(value, Mapping):  # pragma: no cover - Arrow table contract
                    raise ValueError(f"snapshot row is not a mapping: {path}")
                yield {str(key): item for key, item in value.items()}
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to read Parquet snapshot file {path}: {error}") from error
