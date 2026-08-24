from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodelm.decontamination.contamination import BenchmarkEntry, ContaminationSample
from nodelm.decontamination.fingerprints import (
    canonical_repository,
    exact_fingerprint,
    is_near_duplicate,
)
from nodelm.decontamination.split import (
    build_repository_split,
    read_repository_split_repositories,
    write_repository_split_manifest,
)


def test_repository_identity_normalizes_common_git_forms() -> None:
    assert canonical_repository("https://github.com/Acme/Widget.git") == "github.com/acme/widget"
    assert canonical_repository("git@github.com:Acme/Widget.git") == "github.com/acme/widget"
    assert canonical_repository("Acme/Widget") == "github.com/acme/widget"


def test_fingerprints_ignore_line_endings_and_trailing_space() -> None:
    assert exact_fingerprint("one  \r\ntwo\r\n") == exact_fingerprint("one\ntwo\n")


def test_near_duplicate_threshold_is_explicit() -> None:
    assert is_near_duplicate("fix retry timeout", "fix the retry timeout", threshold=0.85)
    assert not is_near_duplicate("fix retry timeout", "replace database driver", threshold=0.85)
    left_patch = "diff --git a/a.ts b/a.ts\n+return retry();\n"
    right_patch = "diff --git a/b.ts b/b.ts\n+return timeout();\n"
    assert is_near_duplicate(left_patch, right_patch, threshold=0.85)
    assert is_near_duplicate(right_patch, left_patch, threshold=0.85)


def test_repository_split_is_deterministic_and_disjoint() -> None:
    samples = [
        ("one", "acme/a"),
        ("two", "acme/a"),
        ("three", "acme/b"),
        ("four", "acme/c"),
        ("five", "acme/d"),
    ]

    first = build_repository_split(samples, seed=7, evaluation_fraction=0.25)
    second = build_repository_split(reversed(samples), seed=7, evaluation_fraction=0.25)

    assert first == second
    assignments = {item.sample_id: item.split for item in first.assignments}
    assert assignments["one"] == assignments["two"]
    train_repos = set(first.repositories["train"])
    eval_repos = set(first.repositories["evaluation"])
    assert train_repos.isdisjoint(eval_repos)


def test_declared_mirrors_resolve_transitively_to_one_repository() -> None:
    samples = [
        ("one", "https://github.com/acme/widget.git"),
        ("two", "mirror/widget"),
        ("three", "legacy/widget"),
    ]
    manifest = build_repository_split(
        samples,
        seed=11,
        evaluation_fraction=0.5,
        aliases={
            "mirror/widget": "legacy/widget",
            "legacy/widget": "acme/widget",
        },
    )

    assert len({item.split for item in manifest.assignments}) == 1
    assert {item.repository for item in manifest.assignments} == {"github.com/acme/widget"}


def test_repository_alias_cycles_are_rejected() -> None:
    with pytest.raises(ValueError, match="repository alias cycle"):
        build_repository_split(
            [("one", "acme/widget")],
            seed=11,
            evaluation_fraction=0.5,
            aliases={
                "acme/widget": "mirror/widget",
                "mirror/widget": "acme/widget",
            },
        )


def test_in_memory_split_api_requires_an_explicit_safe_bound() -> None:
    with pytest.raises(ValueError, match="in-memory split limit"):
        build_repository_split(
            [("one", "acme/one"), ("two", "acme/two")],
            seed=11,
            evaluation_fraction=0.5,
            max_samples=1,
        )


def test_streamed_split_groups_exact_and_near_content_and_excludes_benchmark_overlap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "split.json"
    shared_patch = "diff --git a/a.ts b/a.ts\n+return retry();\n"
    samples = (
        ContaminationSample(
            sample_id="one",
            repository="acme/one",
            task_text="fix retry timeout",
            patch_texts=(shared_patch,),
        ),
        ContaminationSample(
            sample_id="two",
            repository="acme/two",
            task_text="fix the retry timeout",
            patch_texts=("diff --git a/b.ts b/b.ts\n+return timeout();\n",),
        ),
        ContaminationSample(
            sample_id="three",
            repository="acme/three",
            task_text="repair an unrelated parser",
            patch_texts=(shared_patch,),
        ),
        ContaminationSample(
            sample_id="four",
            repository="acme/four",
            task_text="add a standalone formatter",
            patch_texts=("entirely unrelated checksum serialization payload",),
        ),
    )

    write_repository_split_manifest(
        samples,
        benchmarks=(
            BenchmarkEntry(
                benchmark_id="public-one",
                task_text="fix retry timeout",
                patch_text=shared_patch,
            ),
        ),
        near_duplicate_threshold=0.85,
        task_metadata_sha256="b" * 64,
        benchmark_sha256="c" * 64,
        output=output,
        seed=11,
        evaluation_fraction=0.5,
        input_sha256="a" * 64,
        input_bytes=42,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assignments = {item["sample_id"]: item for item in payload["assignments"]}
    assert {assignments[sample_id]["split"] for sample_id in ("one", "two", "three")} == {
        "excluded"
    }
    assert assignments["one"]["contamination_group"] == assignments["two"]["contamination_group"]
    assert assignments["one"]["contamination_group"] == assignments["three"]["contamination_group"]
    assert assignments["four"]["split"] in {"train", "evaluation"}
    evidence = payload["decontamination"]
    assert evidence["exact_patch_duplicate_group_count"] == 1
    assert evidence["near_task_duplicate_pair_count"] >= 1
    assert evidence["near_patch_duplicate_pair_count"] >= 1
    assert evidence["benchmark_exact_task_fingerprint_count"] == 1
    assert evidence["benchmark_exact_patch_fingerprint_count"] == 1
    assert evidence["benchmark_near_task_pair_count"] >= 1
    assert evidence["benchmark_near_patch_pair_count"] >= 1
    assert evidence["benchmark_overlap_sample_count"] == 3
    assert evidence["excluded_repository_count"] == 3
    assert "fix retry timeout" not in output.read_text(encoding="utf-8")
    assert shared_patch not in output.read_text(encoding="utf-8")


def test_streamed_split_verification_failure_leaves_no_manifest(tmp_path: Path) -> None:
    output = tmp_path / "split.json"

    def reject_publication() -> None:
        raise RuntimeError("input changed")

    with pytest.raises(RuntimeError, match="input changed"):
        write_repository_split_manifest(
            (
                ContaminationSample(
                    sample_id="one",
                    repository="acme/widget",
                    task_text="repair widget behavior",
                    patch_texts=("diff --git a/a.ts b/a.ts\n+repair();\n",),
                ),
            ),
            benchmarks=(
                BenchmarkEntry(
                    benchmark_id="public-one",
                    task_text="unrelated benchmark task",
                    patch_text="unrelated benchmark patch",
                ),
            ),
            near_duplicate_threshold=0.85,
            task_metadata_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            output=output,
            seed=11,
            evaluation_fraction=0.5,
            input_sha256="a" * 64,
            input_bytes=42,
            before_publish=reject_publication,
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_contamination_gate_rejects_zero_near_duplicate_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        write_repository_split_manifest(
            (
                ContaminationSample(
                    sample_id="one",
                    repository="acme/widget",
                    task_text="repair widget behavior",
                    patch_texts=("diff --git a/a.ts b/a.ts\n+repair();\n",),
                ),
            ),
            benchmarks=(
                BenchmarkEntry(
                    benchmark_id="public-one",
                    task_text="unrelated benchmark task",
                    patch_text="unrelated benchmark patch",
                ),
            ),
            near_duplicate_threshold=0.0,
            task_metadata_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            output=tmp_path / "split.json",
            seed=11,
            evaluation_fraction=0.5,
            input_sha256="a" * 64,
            input_bytes=42,
        )


def test_streaming_repository_reader_ignores_assignments_and_normalizes_lists(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        """
{
  "schema_version": "nodelm.repository-split/v1",
  "assignments": [
    {"sample_id": "repositories", "repository": "acme/train", "split": "train"},
    {"sample_id": "two", "repository": "acme/eval", "split": "evaluation"}
  ],
  "repositories": {
    "train": ["Acme/Train", "github.com/acme/train"],
    "evaluation": ["acme/eval"]
  }
}
""".lstrip(),
        encoding="utf-8",
    )

    train, evaluation = read_repository_split_repositories(manifest)

    assert train == ("github.com/acme/train",)
    assert evaluation == ("github.com/acme/eval",)


def test_streaming_repository_reader_rejects_overlap_after_normalization(tmp_path: Path) -> None:
    manifest = tmp_path / "split.json"
    manifest.write_text(
        '{"schema_version":"nodelm.repository-split/v1",'
        '"repositories":{"train":["acme/widget"],'
        '"evaluation":["https://github.com/acme/widget.git"]}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        read_repository_split_repositories(manifest)
