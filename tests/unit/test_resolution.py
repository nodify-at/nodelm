from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
    ResolutionLabelConflict,
    build_resolution_recovery,
    model_patch_sha256,
    resolution_key_sha256,
)

TRACE_REVISION = "a" * 40
TASK_REVISION = "b" * 40


def _row(
    instance_id: str,
    rollout_id: str,
    patch: str,
    *,
    resolved: int = -1,
    language: str = "TypeScript",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "trajectory_id": rollout_id,
        "model_patch": patch,
        "resolved": resolved,
        "language": language,
        **extra,
    }


def _build(
    labeled: list[tuple[str, Mapping[str, Any]]],
    targets: list[tuple[str, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    with build_resolution_recovery(
        labeled,
        targets,
        trace_source_revision=TRACE_REVISION,
        task_source_revision=TASK_REVISION,
    ) as recovery:
        candidates = [item.model_dump(mode="json") for item in recovery.iter_candidates()]
        requests = [item.model_dump(mode="json") for item in recovery.iter_evaluation_requests()]
        conflicts = [item.model_dump(mode="json") for item in recovery.iter_conflicts()]
    return candidates, requests, conflicts


def test_exact_transfer_is_revision_bound_and_never_mutates_source_rows() -> None:
    patch = "diff --git a/src/a.ts b/src/a.ts\n+export const answer = 42;\n"
    labeled_row = _row("task-1", "teacher-1", patch, resolved=1, language="ts")
    target_row = _row("task-1", "qwen-1", patch, language="typescript")
    originals = copy.deepcopy((labeled_row, target_row))

    with build_resolution_recovery(
        [("openhands/minimax/swe-rebench-v2", labeled_row)],
        [("openhands/qwen36/swe-rebench-v2", target_row)],
        trace_source_revision=TRACE_REVISION,
        task_source_revision=TASK_REVISION,
    ) as recovery:
        candidates = list(recovery.iter_candidates())
        assert recovery.candidate_count == 1
        assert recovery.evaluation_request_count == 0

    assert (labeled_row, target_row) == originals
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.resolved is True
    assert candidate.instance_id == "task-1"
    assert candidate.model_patch_sha256 == model_patch_sha256(patch)
    assert candidate.resolution_key == resolution_key_sha256(
        instance_id="task-1",
        model_patch=patch,
        trace_source_revision=TRACE_REVISION,
        task_source_revision=TASK_REVISION,
    )
    assert candidate.trace_source_revision == TRACE_REVISION
    assert candidate.task_source_revision == TASK_REVISION
    assert candidate.label_evidence[0].partition_name.startswith("openhands/minimax")
    assert candidate.target_reference.partition_name.startswith("openhands/qwen36")
    with pytest.raises(ValidationError, match="frozen"):
        candidate.resolved = False


def test_duplicate_targets_fan_out_candidates_and_unmatched_requests_deduplicate() -> None:
    matched_patch = "diff --git a/a.js b/a.js\n+retry();\n"
    unmatched_patch = "diff --git a/b.ts b/b.ts\n+await repair();\n"
    labeled = [
        (
            "sweagent/minimax/swe-rebench-v2",
            _row("matched", "known-1", matched_patch, resolved=0, language="js"),
        )
    ]
    targets = [
        (
            "openhands/qwen36/swe-rebench-v2",
            _row("matched", "target-1", matched_patch, language="JavaScript"),
        ),
        (
            "sweagent/qwen36/swe-rebench-v2",
            _row("matched", "target-2", matched_patch, language="javascript"),
        ),
        (
            "openhands/qwen36/swe-rebench-v2",
            _row(
                "unmatched",
                "target-3",
                unmatched_patch,
                trajectory=[{"golden_patch": "MUST_NOT_BE_RETAINED"}],
            ),
        ),
        (
            "sweagent/qwen36/swe-rebench-v2",
            _row(
                "unmatched",
                "target-4",
                unmatched_patch,
                tests="MUST_NOT_BE_RETAINED",
            ),
        ),
    ]

    candidates, requests, conflicts = _build(labeled, targets)

    assert sorted(item["target_reference"]["rollout_id"] for item in candidates) == [
        "target-1",
        "target-2",
    ]
    assert all(item["resolved"] is False for item in candidates)
    assert len(requests) == 1
    assert requests[0]["instance_id"] == "unmatched"
    assert requests[0]["model_patch"] == unmatched_patch
    assert [item["rollout_id"] for item in requests[0]["target_references"]] == [
        "target-3",
        "target-4",
    ]
    assert "MUST_NOT_BE_RETAINED" not in json.dumps(
        {"candidates": candidates, "requests": requests, "conflicts": conflicts}
    )
    serialized = json.dumps(
        {"candidates": candidates, "requests": requests, "conflicts": conflicts}
    )
    assert "projected_row_sha256" in serialized
    assert "raw_row_sha256" not in serialized
    assert conflicts == []


def test_conflicting_known_labels_are_recorded_and_queued_for_real_evaluation() -> None:
    patch = "diff --git a/conflict.ts b/conflict.ts\n+conflict();\n"
    labeled = [
        ("openhands/minimax/swe-rebench-v2", _row("task-c", "known-1", patch, resolved=1)),
        ("sweagent/minimax/swe-rebench-v2", _row("task-c", "known-2", patch, resolved=0)),
    ]
    targets = [
        ("openhands/qwen36/swe-rebench-v2", _row("task-c", "target-1", patch)),
    ]

    candidates, requests, conflicts = _build(labeled, targets)

    assert candidates == []
    assert len(requests) == 1
    assert requests[0]["resolution_key"] == conflicts[0]["resolution_key"]
    assert [item["rollout_id"] for item in conflicts[0]["false_evidence"]] == ["known-2"]
    assert [item["rollout_id"] for item in conflicts[0]["true_evidence"]] == ["known-1"]


def test_unrelated_labeled_conflict_is_not_reported_for_the_target_run() -> None:
    conflict_patch = "diff --git a/conflict.ts b/conflict.ts\n+conflict();\n"
    queued_patch = "diff --git a/queued.ts b/queued.ts\n+queued();\n"
    labeled = [
        (
            "openhands/minimax/swe-rebench-v2",
            _row("unrelated", "known-1", conflict_patch, resolved=1),
        ),
        (
            "sweagent/minimax/swe-rebench-v2",
            _row("unrelated", "known-2", conflict_patch, resolved=0),
        ),
    ]
    targets = [
        (
            "openhands/qwen36/swe-rebench-v2",
            _row("target-only", "target-1", queued_patch),
        ),
    ]

    candidates, requests, conflicts = _build(labeled, targets)

    assert candidates == []
    assert [request["instance_id"] for request in requests] == ["target-only"]
    assert conflicts == []


def test_known_target_resolution_is_never_overridden_or_queued() -> None:
    patch = "diff --git a/a.ts b/a.ts\n+known();\n"
    labeled = [("openhands/minimax/swe-rebench-v2", _row("task-1", "known", patch, resolved=1))]
    targets = [("openhands/qwen36/swe-rebench-v2", _row("task-1", "target", patch, resolved=0))]

    candidates, requests, conflicts = _build(labeled, targets)

    assert candidates == []
    assert requests == []
    assert conflicts == []


def test_language_filter_canonicalizes_typescript_and_javascript_aliases() -> None:
    ts_patch = "diff --git a/a.ts b/a.ts\n+ts();\n"
    js_patch = "diff --git a/a.js b/a.js\n+js();\n"
    python_patch = "diff --git a/a.py b/a.py\n+python()\n"
    labeled = [
        ("source/ts/tasks", _row("ts", "known-ts", ts_patch, resolved=1, language="TS")),
        (
            "source/python/tasks",
            _row("py", "known-py", python_patch, resolved=1, language="Python"),
        ),
    ]
    targets = [
        ("target/ts/tasks", _row("ts", "target-ts", ts_patch, language="TypeScript")),
        ("target/js/tasks", _row("js", "target-js", js_patch, language="JS")),
        ("target/python/tasks", _row("py", "target-py", python_patch, language="python")),
    ]

    candidates, requests, conflicts = _build(labeled, targets)

    assert [item["instance_id"] for item in candidates] == ["ts"]
    assert [item["instance_id"] for item in requests] == ["js"]
    assert requests[0]["language"] == "JavaScript"
    assert conflicts == []


def test_output_order_and_ids_are_independent_of_input_order() -> None:
    patch_a = "diff --git a/a.ts b/a.ts\n+a();\n"
    patch_b = "diff --git a/b.js b/b.js\n+b();\n"
    patch_c = "diff --git a/c.ts b/c.ts\n+c();\n"
    labeled = [
        ("source/z/tasks", _row("a", "known-z", patch_a, resolved=1, language="ts")),
        ("source/a/tasks", _row("a", "known-a", patch_a, resolved=1, language="TypeScript")),
        ("source/b/tasks", _row("b", "known-b", patch_b, resolved=0, language="js")),
    ]
    targets = [
        ("target/z/tasks", _row("c", "target-c2", patch_c)),
        ("target/a/tasks", _row("a", "target-a", patch_a)),
        ("target/b/tasks", _row("b", "target-b", patch_b, language="javascript")),
        ("target/a/tasks", _row("c", "target-c1", patch_c)),
    ]

    first = _build(labeled, targets)
    second = _build(list(reversed(labeled)), list(reversed(targets)))

    assert first == second
    candidate_ids = [item["candidate_id"] for item in first[0]]
    request_ids = [item["request_id"] for item in first[1]]
    assert candidate_ids == sorted(candidate_ids)
    assert request_ids == sorted(request_ids)
    assert all(len(identifier) == 64 for identifier in candidate_ids + request_ids)


def test_candidate_models_reject_coercion_and_extra_fields() -> None:
    patch = "diff --git a/a.ts b/a.ts\n+strict();\n"
    candidates, _, _ = _build(
        [("source/a/tasks", _row("a", "known", patch, resolved=1))],
        [("target/a/tasks", _row("a", "target", patch))],
    )
    payload = candidates[0]

    with pytest.raises(ValidationError, match="valid boolean"):
        ExactResolutionCandidate.model_validate({**payload, "resolved": 1})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExactResolutionCandidate.model_validate({**payload, "trajectory": []})


def test_artifact_models_reject_resolution_keys_unbound_to_their_payloads() -> None:
    matched_patch = "diff --git a/matched.ts b/matched.ts\n+matched();\n"
    queued_patch = "diff --git a/queued.ts b/queued.ts\n+queued();\n"
    conflicted_patch = "diff --git a/conflicted.ts b/conflicted.ts\n+conflicted();\n"
    candidate = _build(
        [("source/a/tasks", _row("matched", "known", matched_patch, resolved=1))],
        [("target/a/tasks", _row("matched", "target", matched_patch))],
    )[0][0]
    request = _build(
        [],
        [("target/a/tasks", _row("queued", "target", queued_patch))],
    )[1][0]
    conflict = _build(
        [
            ("source/a/tasks", _row("conflicted", "known-1", conflicted_patch, resolved=1)),
            ("source/b/tasks", _row("conflicted", "known-2", conflicted_patch, resolved=0)),
        ],
        [("target/a/tasks", _row("conflicted", "target", conflicted_patch))],
    )[2][0]

    for model, payload, identity_field in (
        (ExactResolutionCandidate, candidate, "candidate_id"),
        (ResolutionEvaluationRequest, request, "request_id"),
        (ResolutionLabelConflict, conflict, "conflict_id"),
    ):
        malformed = {**payload, "resolution_key": "f" * 64}
        malformed.pop(identity_field)
        with pytest.raises(ValidationError, match="resolution_key does not match"):
            model.model_validate(malformed)


def test_nested_open_swe_model_patch_schema_is_supported() -> None:
    patch = "diff --git a/nested.ts b/nested.ts\n+nested();\n"
    labeled = {
        "instance_id": "nested",
        "trajectory_id": "known-nested",
        "resolved": 1,
        "metadata": {
            "language": "ts",
            "model_patch": {"patch": patch},
        },
    }
    target = {
        "instance_id": "nested",
        "trajectory_id": "target-nested",
        "resolved": -1,
        "metadata": {
            "language": "TypeScript",
            "model_patch": {"patch": patch},
        },
    }

    candidates, requests, conflicts = _build(
        [("source/nested/tasks", labeled)],
        [("target/nested/tasks", target)],
    )

    assert len(candidates) == 1
    assert candidates[0]["model_patch_sha256"] == model_patch_sha256(patch)
    assert requests == []
    assert conflicts == []


def test_recovery_exposes_global_and_per_partition_accounting() -> None:
    patch_a = "diff --git a/a.ts b/a.ts\n+a();\n"
    patch_b = "diff --git a/b.js b/b.js\n+b();\n"
    patch_c = "diff --git a/c.ts b/c.ts\n+c();\n"
    patch_d = "diff --git a/d.js b/d.js\n+d();\n"
    labeled = [
        ("source/one/tasks", _row("a", "known-a1", patch_a, resolved=1)),
        ("source/two/tasks", _row("a", "known-a2", patch_a, resolved=1, language="ts")),
        ("source/one/tasks", _row("c", "known-c1", patch_c, resolved=1)),
        ("source/two/tasks", _row("c", "known-c2", patch_c, resolved=0)),
        ("source/one/tasks", _row("d", "known-d", patch_d, resolved=0, language="js")),
        (
            "source/one/tasks",
            _row("python", "known-py", "python", resolved=1, language="Python"),
        ),
    ]
    targets = [
        ("target/one/tasks", _row("a", "target-a1", patch_a)),
        ("target/two/tasks", _row("a", "target-a2", patch_a, language="ts")),
        ("target/one/tasks", _row("b", "target-b1", patch_b, language="js")),
        ("target/two/tasks", _row("b", "target-b2", patch_b, language="JavaScript")),
        ("target/one/tasks", _row("c", "target-c", patch_c)),
        ("target/one/tasks", _row("d", "target-d", patch_d, language="javascript")),
        ("target/one/tasks", _row("a", "already-known", patch_a, resolved=0)),
        ("target/two/tasks", _row("python", "target-py", "python", language="Python")),
    ]

    with build_resolution_recovery(
        labeled,
        targets,
        trace_source_revision=TRACE_REVISION,
        task_source_revision=TASK_REVISION,
    ) as recovery:
        assert recovery.labeled_row_count == 6
        assert recovery.target_row_count == 8
        assert recovery.target_ineligible_count == 1
        assert recovery.target_already_known_count == 1
        assert recovery.candidate_count == recovery.candidate_row_count == 3
        assert recovery.candidate_unique_count == 2
        assert recovery.candidate_resolved_count == 2
        assert recovery.candidate_unresolved_count == 1
        assert recovery.queued_target_count == 3
        assert recovery.evaluation_request_count == 2
        assert recovery.conflict_count == 1
        assert recovery.labeled_row_counts_by_partition == {
            "source/one/tasks": 4,
            "source/two/tasks": 2,
        }
        assert recovery.target_row_counts_by_partition == {
            "target/one/tasks": 5,
            "target/two/tasks": 3,
        }
