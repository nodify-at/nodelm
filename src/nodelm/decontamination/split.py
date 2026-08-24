from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

import yaml
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    DocumentEndEvent,
    DocumentStartEvent,
    Event,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
    StreamEndEvent,
    StreamStartEvent,
)

from nodelm.artifacts import ArtifactWriteResult, write_immutable_stream
from nodelm.decontamination.contamination import (
    BenchmarkEntry,
    ContaminationSample,
    build_contamination_index,
)
from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.models import stable_model_id

SplitName = Literal["train", "evaluation", "excluded"]


@dataclass(frozen=True, order=True)
class SplitAssignment:
    sample_id: str
    repository: str
    split: SplitName


@dataclass(frozen=True)
class RepositorySplitManifest:
    schema_version: str
    seed: int
    evaluation_fraction: float
    config_digest: str
    assignments: tuple[SplitAssignment, ...]
    repositories: dict[SplitName, tuple[str, ...]]


def _repository_bucket(repository: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{repository}".encode()).digest()
    numerator = int.from_bytes(digest[:8], "big")
    return numerator / (2**64 - 1)


def _resolved_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
    direct: dict[str, str] = {}
    for raw_alias, raw_target in aliases.items():
        alias = canonical_repository(raw_alias)
        target = canonical_repository(raw_target)
        if alias == target:
            continue
        previous = direct.get(alias)
        if previous is not None and previous != target:
            raise ValueError(f"conflicting repository alias targets for {alias}")
        direct[alias] = target

    resolved: dict[str, str] = {}
    for origin in sorted(direct):
        if origin in resolved:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current = origin
        while current in direct and current not in resolved:
            if current in positions:
                cycle = [*path[positions[current] :], current]
                raise ValueError(f"repository alias cycle: {' -> '.join(cycle)}")
            positions[current] = len(path)
            path.append(current)
            current = direct[current]

        target = resolved.get(current, current)
        for alias in reversed(path):
            resolved[alias] = target
    return dict(sorted(resolved.items()))


def _next_yaml_event(events: Iterator[Event], context: str) -> Event:
    try:
        return next(events)
    except StopIteration:
        raise ValueError(f"repository split manifest ended while reading {context}") from None


def _skip_yaml_value(first: Event, events: Iterator[Event]) -> None:
    if isinstance(first, (ScalarEvent, AliasEvent)):
        return
    if not isinstance(first, CollectionStartEvent):
        raise ValueError("repository split manifest contains an invalid value")

    depth = 1
    while depth:
        event = _next_yaml_event(events, "a manifest value")
        if isinstance(event, CollectionStartEvent):
            depth += 1
        elif isinstance(event, CollectionEndEvent):
            depth -= 1


def _read_repository_mapping(
    first: Event,
    events: Iterator[Event],
) -> dict[SplitName, list[str]]:
    if not isinstance(first, MappingStartEvent):
        raise ValueError("split manifest requires a repositories mapping")

    repositories: dict[SplitName, list[str]] = {
        "train": [],
        "evaluation": [],
        "excluded": [],
    }
    seen: set[str] = set()
    while True:
        key_event = _next_yaml_event(events, "repositories")
        if isinstance(key_event, MappingEndEvent):
            break
        if not isinstance(key_event, ScalarEvent):
            raise ValueError("split manifest repository keys must be strings")
        key = key_event.value
        if key in seen:
            raise ValueError(f"duplicate repositories key: {key}")
        seen.add(key)
        value_event = _next_yaml_event(events, f"repositories.{key}")
        if key not in repositories:
            _skip_yaml_value(value_event, events)
            continue
        if not isinstance(value_event, SequenceStartEvent):
            raise ValueError("split manifest train/evaluation repositories must be lists")

        split = cast(SplitName, key)
        values = repositories[split]
        while True:
            item_event = _next_yaml_event(events, f"repositories.{split}")
            if isinstance(item_event, SequenceEndEvent):
                break
            if not isinstance(item_event, ScalarEvent):
                raise ValueError("split manifest repository identities must be strings")
            values.append(item_event.value)

    if not {"train", "evaluation"}.issubset(seen):
        raise ValueError("split manifest requires train and evaluation repository lists")
    return repositories


def read_repository_split_repositories(
    path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stream a split manifest and retain only its repository-level partition.

    PyYAML's event API validates and traverses the complete JSON/YAML document lazily, so
    large per-sample assignment arrays are never materialized in memory.
    """

    schema_version: str | None = None
    repository_mapping: dict[SplitName, list[str]] | None = None
    try:
        with path.open(encoding="utf-8") as source:
            events = iter(yaml.parse(source, Loader=yaml.SafeLoader))
            if not isinstance(_next_yaml_event(events, "stream start"), StreamStartEvent):
                raise ValueError("repository split manifest has no stream start")
            if not isinstance(_next_yaml_event(events, "document start"), DocumentStartEvent):
                raise ValueError("repository split manifest has no document start")
            if not isinstance(_next_yaml_event(events, "document root"), MappingStartEvent):
                raise ValueError("repository split manifest root must be a mapping")

            seen: set[str] = set()
            while True:
                key_event = _next_yaml_event(events, "manifest root")
                if isinstance(key_event, MappingEndEvent):
                    break
                if not isinstance(key_event, ScalarEvent):
                    raise ValueError("repository split manifest keys must be strings")
                key = key_event.value
                if key in seen:
                    raise ValueError(f"duplicate repository split manifest key: {key}")
                seen.add(key)
                value_event = _next_yaml_event(events, key)
                if key == "schema_version":
                    if not isinstance(value_event, ScalarEvent):
                        raise ValueError("repository split schema_version must be a string")
                    schema_version = value_event.value
                elif key == "repositories":
                    repository_mapping = _read_repository_mapping(value_event, events)
                else:
                    _skip_yaml_value(value_event, events)

            if not isinstance(_next_yaml_event(events, "document end"), DocumentEndEvent):
                raise ValueError("repository split manifest has an invalid document ending")
            if not isinstance(_next_yaml_event(events, "stream end"), StreamEndEvent):
                raise ValueError("repository split manifest must contain exactly one document")
    except yaml.YAMLError as error:
        raise ValueError(f"invalid repository split manifest: {error}") from error

    if schema_version != "nodelm.repository-split/v1":
        raise ValueError("unsupported repository split manifest schema_version")
    if repository_mapping is None:
        raise ValueError("split manifest requires a repositories mapping")

    try:
        train = tuple(sorted({canonical_repository(item) for item in repository_mapping["train"]}))
        evaluation = tuple(
            sorted({canonical_repository(item) for item in repository_mapping["evaluation"]})
        )
        excluded = tuple(
            sorted({canonical_repository(item) for item in repository_mapping["excluded"]})
        )
    except ValueError as error:
        raise ValueError(f"invalid split manifest repository identity: {error}") from error
    if (
        set(train) & set(evaluation)
        or set(train) & set(excluded)
        or set(evaluation) & set(excluded)
    ):
        raise ValueError("split manifest repository lists overlap")
    return train, evaluation


def build_repository_split(
    samples: Iterable[tuple[str, str]],
    *,
    seed: int,
    evaluation_fraction: float,
    aliases: Mapping[str, str] | None = None,
    max_samples: int = 100_000,
) -> RepositorySplitManifest:
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1")
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")

    normalized_aliases = _resolved_aliases(aliases or {})
    normalized_samples: list[tuple[str, str]] = []
    seen_sample_ids: set[str] = set()
    for sample_id, raw_repository in samples:
        if len(normalized_samples) >= max_samples:
            raise ValueError(
                f"in-memory split limit exceeded ({max_samples}); "
                "use write_repository_split_manifest for streamed inputs"
            )
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        repository = canonical_repository(raw_repository)
        repository = normalized_aliases.get(repository, repository)
        normalized_samples.append((sample_id, repository))

    repository_splits: dict[str, SplitName] = {}
    for repository in sorted({repository for _, repository in normalized_samples}):
        repository_splits[repository] = (
            "evaluation" if _repository_bucket(repository, seed) < evaluation_fraction else "train"
        )

    assignments = tuple(
        sorted(
            SplitAssignment(
                sample_id=sample_id,
                repository=repository,
                split=repository_splits[repository],
            )
            for sample_id, repository in normalized_samples
        )
    )
    repositories: dict[SplitName, tuple[str, ...]] = {
        "train": tuple(
            repository
            for repository, split in sorted(repository_splits.items())
            if split == "train"
        ),
        "evaluation": tuple(
            repository
            for repository, split in sorted(repository_splits.items())
            if split == "evaluation"
        ),
    }
    config_digest = stable_model_id(
        {
            "seed": seed,
            "evaluation_fraction": evaluation_fraction,
            "aliases": normalized_aliases,
        }
    )
    return RepositorySplitManifest(
        schema_version="nodelm.repository-split/v1",
        seed=seed,
        evaluation_fraction=evaluation_fraction,
        config_digest=config_digest,
        assignments=assignments,
        repositories=repositories,
    )


def write_repository_split_manifest(
    samples: Iterable[ContaminationSample],
    *,
    benchmarks: Iterable[BenchmarkEntry],
    near_duplicate_threshold: float,
    task_metadata_sha256: str,
    benchmark_sha256: str,
    output: Path,
    seed: int,
    evaluation_fraction: float,
    aliases: Mapping[str, str] | None = None,
    input_sha256: str,
    input_bytes: int,
    aliases_sha256: str | None = None,
    before_publish: Callable[[], object] | None = None,
) -> ArtifactWriteResult:
    """Build a contamination-safe split through disk-backed indexes and stream its artifact."""

    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be strictly between 0 and 1")
    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in input_sha256
    ):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    if input_bytes < 0:
        raise ValueError("input_bytes must be non-negative")

    normalized_aliases = _resolved_aliases(aliases or {})
    output = output.resolve()
    with build_contamination_index(
        samples,
        benchmarks=benchmarks,
        near_duplicate_threshold=near_duplicate_threshold,
        aliases=normalized_aliases,
        task_metadata_sha256=task_metadata_sha256,
        benchmark_sha256=benchmark_sha256,
        seed=seed,
        evaluation_fraction=evaluation_fraction,
        temp_directory=output.parent,
    ) as index:

        def write_json(stream: BinaryIO) -> None:
            def encoded(value: object) -> bytes:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")

            stream.write(b'{"aliases_sha256":')
            stream.write(encoded(aliases_sha256))
            stream.write(b',"assignments":[')
            first = True
            for contamination_group, repository, sample_id, split in index.iter_assignments():
                if not first:
                    stream.write(b",")
                first = False
                stream.write(
                    encoded(
                        {
                            "contamination_group": contamination_group,
                            "repository": repository,
                            "sample_id": sample_id,
                            "split": split,
                        }
                    )
                )
            stream.write(b'],"config_digest":')
            stream.write(encoded(index.config_digest))
            stream.write(b',"decontamination":')
            stream.write(encoded(index.evidence.model_dump(mode="json")))
            stream.write(b',"evaluation_fraction":')
            stream.write(encoded(evaluation_fraction))
            stream.write(b',"input_bytes":')
            stream.write(encoded(input_bytes))
            stream.write(b',"input_sha256":')
            stream.write(encoded(input_sha256))
            stream.write(b',"repositories":{')
            split_order: tuple[SplitName, ...] = ("evaluation", "excluded", "train")
            for split_index, split in enumerate(split_order):
                if split_index:
                    stream.write(b"],")
                stream.write(encoded(split))
                stream.write(b":[")
                first = True
                for repository in index.iter_repositories(split):
                    if not first:
                        stream.write(b",")
                    first = False
                    stream.write(encoded(repository))
            stream.write(b']},"sample_count":')
            stream.write(encoded(index.sample_count))
            stream.write(b',"schema_version":"nodelm.repository-split/v1","seed":')
            stream.write(encoded(seed))
            stream.write(b"}\n")

        return write_immutable_stream(output, write_json, before_publish=before_publish)
