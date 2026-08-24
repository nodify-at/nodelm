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


def iter_snapshot_rows(paths: tuple[Path, ...]) -> Iterator[dict[str, Any]]:
    """Stream records from deterministic JSONL or Parquet snapshot files."""

    if not paths:
        raise ValueError("at least one snapshot data file is required")
    for path in paths:
        suffix = path.suffix.casefold()
        if suffix == ".jsonl":
            yield from _iter_jsonl(path)
        elif suffix == ".parquet":
            yield from _iter_parquet(path)
        else:
            raise ValueError(f"unsupported snapshot data file: {path}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
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
            yield value


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=1_024):
            for value in batch.to_pylist():
                if not isinstance(value, Mapping):  # pragma: no cover - Arrow table contract
                    raise ValueError(f"snapshot row is not a mapping: {path}")
                yield {str(key): item for key, item in value.items()}
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to read Parquet snapshot file {path}: {error}") from error
