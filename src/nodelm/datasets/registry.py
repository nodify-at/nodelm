from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nodelm.models import DatasetSource


class RegistryError(ValueError):
    """The dataset registry is malformed or internally inconsistent."""


@dataclass(frozen=True)
class DatasetRegistry:
    schema_version: str
    sources: tuple[DatasetSource, ...]

    @classmethod
    def load(cls, path: Path) -> DatasetRegistry:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise RegistryError(f"unable to read dataset registry {path}: {error}") from error

        if not isinstance(raw, dict):
            raise RegistryError("dataset registry root must be a mapping")
        if raw.get("schema_version") != "nodelm.dataset-registry/v1":
            raise RegistryError("unsupported dataset registry schema_version")
        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise RegistryError("dataset registry sources must be a non-empty list")

        try:
            sources = tuple(DatasetSource.model_validate(item) for item in raw_sources)
        except ValidationError as error:
            raise RegistryError(f"invalid dataset source: {error}") from error

        name_counts = Counter(source.name for source in sources)
        duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicate_names:
            raise RegistryError(f"duplicate dataset name: {', '.join(duplicate_names)}")

        return cls(schema_version=str(raw["schema_version"]), sources=sources)

    def by_name(self, name: str) -> DatasetSource:
        for source in self.sources:
            if source.name == name:
                return source
        raise RegistryError(f"unknown dataset source: {name}")

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.model_dump(mode="json") for source in self.sources],
        }
