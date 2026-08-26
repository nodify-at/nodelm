from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nodelm.artifacts import canonical_json_bytes, content_digest, file_identity

REGULAR_FILE_TREE_IDENTITY_SCHEMA = "nodelm.regular-file-tree-identity/v1"


class VerifiedStagingError(ValueError):
    """An input could not be copied into a private identity-verified staging view."""


@dataclass(frozen=True, order=True)
class RegularFileIdentity:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or "\x00" in self.path
            or relative.is_absolute()
            or relative.as_posix() != self.path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("regular-file identity paths must be normalized relative POSIX paths")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("regular-file identities require lowercase sha256 digests")
        if self.bytes < 0:
            raise ValueError("regular-file identity bytes must be non-negative")


@dataclass(frozen=True)
class RegularFileTreeIdentity:
    schema_version: str
    tree_sha256: str
    tree_bytes: int
    files: tuple[RegularFileIdentity, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)


def regular_file_tree_identity(
    files: Sequence[RegularFileIdentity],
) -> RegularFileTreeIdentity:
    ordered = tuple(sorted(files))
    paths = tuple(item.path for item in ordered)
    if not ordered or len(paths) != len(set(paths)):
        raise ValueError("regular-file tree identities require unique files")
    tree_bytes = sum(item.bytes for item in ordered)
    payload = {
        "schema_version": REGULAR_FILE_TREE_IDENTITY_SCHEMA,
        "tree_bytes": tree_bytes,
        "files": [
            {"path": item.path, "sha256": item.sha256, "bytes": item.bytes} for item in ordered
        ],
    }
    domain = f"{REGULAR_FILE_TREE_IDENTITY_SCHEMA}\0".encode()
    return RegularFileTreeIdentity(
        schema_version=REGULAR_FILE_TREE_IDENTITY_SCHEMA,
        tree_sha256=content_digest(domain + canonical_json_bytes(payload)),
        tree_bytes=tree_bytes,
        files=ordered,
    )


def _expected_directories(identity: RegularFileTreeIdentity) -> set[str]:
    directories: set[str] = set()
    for item in identity.files:
        parts = PurePosixPath(item.path).parts[:-1]
        for end in range(1, len(parts) + 1):
            directories.add(PurePosixPath(*parts[:end]).as_posix())
    return directories


def _capture_regular_file_tree(root: Path) -> tuple[RegularFileTreeIdentity, set[str]]:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise VerifiedStagingError(
            f"unable to inspect regular-file tree {root}: {error}"
        ) from error
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise VerifiedStagingError(f"regular-file tree root must be a real directory: {root}")

    files: list[RegularFileIdentity] = []
    directories: set[str] = set()

    def visit(directory: Path, relative_directory: PurePosixPath | None = None) -> None:
        try:
            with os.scandir(directory) as directory_entries:
                entries = sorted(directory_entries, key=lambda entry: entry.name)
        except OSError as error:
            raise VerifiedStagingError(
                f"unable to enumerate regular-file tree {directory}: {error}"
            ) from error
        for entry in entries:
            relative = (
                PurePosixPath(entry.name)
                if relative_directory is None
                else relative_directory / entry.name
            )
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise VerifiedStagingError(
                    f"unable to inspect tree entry {path}: {error}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise VerifiedStagingError(f"regular-file tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative.as_posix())
                visit(path, relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise VerifiedStagingError(f"regular-file tree contains a special file: {relative}")
            try:
                digest, byte_count = file_identity(path)
                final_metadata = path.lstat()
            except OSError as error:
                raise VerifiedStagingError(f"unable to read tree file {path}: {error}") from error
            if not stat.S_ISREG(final_metadata.st_mode) or stat.S_ISLNK(final_metadata.st_mode):
                raise VerifiedStagingError(f"tree file changed type while being read: {relative}")
            files.append(
                RegularFileIdentity(
                    path=relative.as_posix(),
                    sha256=digest,
                    bytes=byte_count,
                )
            )

    visit(root)
    try:
        identity = regular_file_tree_identity(files)
    except ValueError as error:
        raise VerifiedStagingError(f"invalid regular-file tree {root}: {error}") from error
    return identity, directories


def verify_regular_file_tree(root: Path, expected: RegularFileTreeIdentity) -> None:
    observed, directories = _capture_regular_file_tree(root)
    if observed != expected or directories != _expected_directories(expected):
        raise VerifiedStagingError(
            f"regular-file tree does not match its authorized identity: {root}"
        )


@contextmanager
def verified_staged_files(
    inputs: Sequence[tuple[Path, tuple[str, int]]],
) -> Iterator[tuple[Path, ...]]:
    """Copy inputs privately and expose only copies matching their captured identities.

    The destination names preserve source suffixes so format dispatch remains deterministic.
    A swap during copying changes the staged digest and fails before any transformation reads it.
    """

    if not inputs:
        raise VerifiedStagingError("at least one file is required for verified staging")
    with tempfile.TemporaryDirectory(prefix="nodelm-verified-stage-") as temporary_name:
        staging_root = Path(temporary_name)
        staged: list[Path] = []
        for index, (source, expected) in enumerate(inputs):
            destination = staging_root / f"{index:08d}{source.suffix.casefold()}"
            try:
                shutil.copyfile(source, destination)
            except OSError as error:
                raise VerifiedStagingError(f"unable to stage input {source}: {error}") from error
            if file_identity(destination) != expected:
                raise VerifiedStagingError(
                    f"staged input does not match its captured identity: {source}"
                )
            staged.append(destination)

        for source, expected in inputs:
            if file_identity(source) != expected:
                raise VerifiedStagingError(
                    f"input changed while it was copied into staging: {source}"
                )
        yield tuple(staged)


@contextmanager
def verified_staged_file(
    source: Path,
    expected: tuple[str, int],
) -> Iterator[Path]:
    """Expose one private identity-verified copy and recheck its source on exit.

    This is the bounded-space counterpart to :func:`verified_staged_files` for callers
    that can consume a sequence one file at a time. At most one source-sized copy exists.
    """

    with verified_staged_files(((source, expected),)) as staged_files:
        try:
            yield staged_files[0]
        finally:
            try:
                observed = file_identity(source)
            except OSError as error:
                raise VerifiedStagingError(
                    f"unable to recheck input while staged {source}: {error}"
                ) from error
            if observed != expected:
                raise VerifiedStagingError(f"input changed while staged: {source}")


@contextmanager
def verified_staged_regular_file_tree(
    source_root: Path,
    expected: RegularFileTreeIdentity,
) -> Iterator[Path]:
    """Expose a private tree containing only identity-authorized regular files."""

    verify_regular_file_tree(source_root, expected)
    with tempfile.TemporaryDirectory(prefix="nodelm-verified-tree-") as temporary_name:
        staged_root = Path(temporary_name) / "tree"
        staged_root.mkdir()
        for item in expected.files:
            source = source_root / Path(item.path)
            destination = staged_root / Path(item.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(source, destination, follow_symlinks=False)
            except OSError as error:
                raise VerifiedStagingError(
                    f"unable to stage tree file {source}: {error}"
                ) from error
        verify_regular_file_tree(staged_root, expected)
        verify_regular_file_tree(source_root, expected)
        yield staged_root
