from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_IGNORED_DIRECTORIES = frozenset(
    {".git", ".next", ".turbo", "build", "coverage", "dist", "node_modules"}
)


@dataclass(frozen=True, slots=True)
class PackageManifest:
    path: Path
    name: str | None
    private: bool | None
    package_manager: str | None
    scripts: tuple[tuple[str, str], ...]
    workspace_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TypeScriptWorkspace:
    root: Path
    package_manager: str | None
    package_manifests: tuple[PackageManifest, ...]
    workspace_packages: tuple[PackageManifest, ...]
    tsconfig_paths: tuple[Path, ...]


def discover_package_manifests(workspace_root: Path | str) -> tuple[PackageManifest, ...]:
    root = _validated_root(workspace_root)
    paths, _ = _discover_workspace_files(root)
    return tuple(_load_package_manifest(path) for path in paths)


def discover_tsconfigs(workspace_root: Path | str) -> tuple[Path, ...]:
    root = _validated_root(workspace_root)
    _, paths = _discover_workspace_files(root)
    return paths


def discover_typescript_workspace(workspace_root: Path | str) -> TypeScriptWorkspace:
    root = _validated_root(workspace_root)
    manifest_paths, tsconfig_paths = _discover_workspace_files(root)
    manifests = tuple(_load_package_manifest(path) for path in manifest_paths)
    root_manifest = next((manifest for manifest in manifests if manifest.path.parent == root), None)
    workspace_paths = _expand_workspace_patterns(
        root, () if root_manifest is None else root_manifest.workspace_patterns
    )
    by_path = {manifest.path: manifest for manifest in manifests}
    workspace_packages = tuple(
        by_path[path] for path in workspace_paths if path in by_path and path.parent != root
    )
    return TypeScriptWorkspace(
        root=root,
        package_manager=None if root_manifest is None else root_manifest.package_manager,
        package_manifests=manifests,
        workspace_packages=workspace_packages,
        tsconfig_paths=tsconfig_paths,
    )


def _discover_workspace_files(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    manifests: list[Path] = []
    tsconfigs: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_names[:] = sorted(
            name for name in directory_names if name not in _IGNORED_DIRECTORIES
        )
        for file_name in sorted(file_names):
            is_manifest = file_name == "package.json"
            is_tsconfig = file_name == "tsconfig.json" or (
                file_name.startswith("tsconfig.") and file_name.endswith(".json")
            )
            if not is_manifest and not is_tsconfig:
                continue
            path = (Path(directory) / file_name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"discovered path resolves outside workspace: {path}")
            (manifests if is_manifest else tsconfigs).append(path)
    return tuple(sorted(manifests)), tuple(sorted(tsconfigs))


def _load_package_manifest(path: Path) -> PackageManifest:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in package manifest {path}: {error.msg}") from error
    except OSError as error:
        raise ValueError(f"could not read package manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"package manifest must contain a JSON object: {path}")

    name = value.get("name")
    private = value.get("private")
    package_manager = value.get("packageManager")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"package name must be a string in {path}")
    if private is not None and not isinstance(private, bool):
        raise ValueError(f"package private flag must be a boolean in {path}")
    if package_manager is not None and not isinstance(package_manager, str):
        raise ValueError(f"packageManager must be a string in {path}")

    scripts_value = value.get("scripts", {})
    if not isinstance(scripts_value, dict) or any(
        not isinstance(script, str) or not isinstance(command, str)
        for script, command in scripts_value.items()
    ):
        raise ValueError(f"package scripts must map strings to strings in {path}")

    return PackageManifest(
        path=path,
        name=name,
        private=private,
        package_manager=package_manager,
        scripts=tuple(sorted(scripts_value.items())),
        workspace_patterns=_workspace_patterns(value.get("workspaces"), path),
    )


def _workspace_patterns(value: Any, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        value = value.get("packages", [])
    if not isinstance(value, list) or any(not isinstance(pattern, str) for pattern in value):
        raise ValueError(f"workspaces must be a string list in {path}")
    patterns = tuple(value)
    for pattern in patterns:
        pure = PurePosixPath(pattern)
        if pure.is_absolute() or ".." in pure.parts or pattern.startswith("!"):
            raise ValueError(
                f"workspace pattern escapes or excludes the workspace in {path}: {pattern}"
            )
    return patterns


def _expand_workspace_patterns(root: Path, patterns: tuple[str, ...]) -> tuple[Path, ...]:
    matches: set[Path] = set()
    for pattern in patterns:
        for candidate in root.glob(pattern):
            manifest = (candidate / "package.json" if candidate.is_dir() else candidate).resolve()
            if not manifest.is_relative_to(root):
                raise ValueError(f"workspace package resolves outside workspace: {candidate}")
            if manifest.name == "package.json" and manifest.is_file():
                matches.add(manifest)
    return tuple(sorted(matches))


def _validated_root(workspace_root: Path | str) -> Path:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root is not an existing directory: {root}")
    return root
