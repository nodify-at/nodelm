from __future__ import annotations

from pathlib import Path

import pytest

from nodelm.artifacts import file_identity
from nodelm.datasets.staging import (
    RegularFileIdentity,
    VerifiedStagingError,
    regular_file_tree_identity,
    verified_staged_files,
    verified_staged_regular_file_tree,
)


def test_verified_staging_preserves_order_suffix_and_content(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.parquet"
    first.write_bytes(b'{"row":1}\n')
    second.write_bytes(b"parquet fixture")

    with verified_staged_files(
        ((first, file_identity(first)), (second, file_identity(second)))
    ) as staged:
        assert tuple(path.suffix for path in staged) == (".jsonl", ".parquet")
        assert tuple(path.read_bytes() for path in staged) == (
            first.read_bytes(),
            second.read_bytes(),
        )
        staged_root = staged[0].parent

    assert not staged_root.exists()


def test_verified_staging_rejects_wrong_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with (
        pytest.raises(VerifiedStagingError, match="captured identity"),
        verified_staged_files(((source, ("0" * 64, source.stat().st_size)),)),
    ):
        raise AssertionError("unreachable")


def test_verified_tree_staging_is_private_and_aba_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    source_file = source / "src" / "main.js"
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    digest, byte_count = file_identity(source_file)
    expected = regular_file_tree_identity(
        (RegularFileIdentity(path="src/main.js", sha256=digest, bytes=byte_count),)
    )

    with verified_staged_regular_file_tree(source, expected) as staged:
        staged_file = staged / "src" / "main.js"
        source_file.write_text("export const value = 2;\n", encoding="utf-8")
        source_file.write_text("export const value = 1;\n", encoding="utf-8")

        assert staged_file.read_text(encoding="utf-8") == "export const value = 1;\n"
        staged_root = staged

    assert not staged_root.exists()


@pytest.mark.parametrize("alteration", ["extra", "missing", "symlink"])
def test_verified_tree_staging_rejects_non_authorized_tree(
    tmp_path: Path,
    alteration: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "main.js"
    source_file.write_text("export {};\n", encoding="utf-8")
    digest, byte_count = file_identity(source_file)
    expected = regular_file_tree_identity(
        (RegularFileIdentity(path="main.js", sha256=digest, bytes=byte_count),)
    )
    if alteration == "extra":
        (source / "extra.js").write_text("export const extra = true;\n", encoding="utf-8")
    elif alteration == "missing":
        source_file.unlink()
    else:
        source_file.unlink()
        source_file.symlink_to(tmp_path / "outside.js")

    with (
        pytest.raises(VerifiedStagingError),
        verified_staged_regular_file_tree(source, expected),
    ):
        raise AssertionError("unreachable")
