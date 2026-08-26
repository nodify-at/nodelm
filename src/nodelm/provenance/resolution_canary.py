from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from nodelm.evaluation.resolution_canary import (
    ResolutionCanaryCase,
    ResolutionCanaryError,
    SWERebenchTask,
)
from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
    evaluation_request_from_target_row,
    model_patch_sha256,
)


@dataclass(frozen=True)
class ResolutionCanarySelection:
    transfer_controls: tuple[ExactResolutionCandidate, ...]
    evaluation_requests: tuple[ResolutionEvaluationRequest, ...]


T = TypeVar("T")


def _bounded_cover(
    items: Iterable[T],
    *,
    identity: Callable[[T], str],
    features: Callable[[T], set[tuple[str, Hashable]]],
    minimum: int,
    maximum: int,
) -> tuple[T, ...]:
    if minimum <= 0 or maximum < minimum:
        raise ResolutionCanaryError("resolution canary selection bounds are invalid")
    feature_choices: dict[tuple[str, Hashable], tuple[str, T]] = {}
    smallest: list[tuple[str, T]] = []
    seen_ids: set[str] = set()
    for item in items:
        item_id = identity(item)
        if item_id in seen_ids:
            raise ResolutionCanaryError("resolution canary source IDs must be unique")
        seen_ids.add(item_id)
        smallest.append((item_id, item))
        smallest.sort(key=lambda pair: pair[0])
        del smallest[maximum:]
        for feature in features(item):
            current = feature_choices.get(feature)
            if current is None or item_id < current[0]:
                feature_choices[feature] = (item_id, item)
    if not seen_ids:
        raise ResolutionCanaryError("resolution canary requires non-empty source artifacts")
    selected_by_id = {item_id: item for item_id, item in feature_choices.values()}
    if len(selected_by_id) > maximum:
        raise ResolutionCanaryError("canary selection bound cannot cover observed strata")
    for item_id, item in smallest:
        if len(selected_by_id) >= minimum:
            break
        selected_by_id[item_id] = item
    if len(selected_by_id) < minimum:
        raise ResolutionCanaryError("resolution artifacts contain too few unique canary sources")
    return tuple(selected_by_id[item_id] for item_id in sorted(selected_by_id))


def select_resolution_canary_sources(
    candidates: Iterable[ExactResolutionCandidate],
    requests: Iterable[ResolutionEvaluationRequest],
    *,
    minimum_per_kind: int = 6,
    maximum_per_kind: int = 12,
) -> ResolutionCanarySelection:
    """Select a small deterministic set that covers observed language/partition/label strata."""

    controls = _bounded_cover(
        candidates,
        identity=lambda item: item.candidate_id,
        features=lambda item: {
            ("language", item.language),
            ("partition", item.target_reference.partition_name),
            ("label", item.resolved),
        },
        minimum=minimum_per_kind,
        maximum=maximum_per_kind,
    )
    evaluation_requests = _bounded_cover(
        requests,
        identity=lambda item: item.request_id,
        features=lambda item: {
            ("language", item.language),
            *(("partition", reference.partition_name) for reference in item.target_references),
        },
        minimum=minimum_per_kind,
        maximum=maximum_per_kind,
    )
    return ResolutionCanarySelection(
        transfer_controls=controls,
        evaluation_requests=evaluation_requests,
    )


def _validate_task(
    request: ResolutionEvaluationRequest,
    task: SWERebenchTask,
) -> None:
    if (
        task.instance_id != request.instance_id
        or task.language != request.language
        or task.task_source_revision.casefold() != request.task_source_revision.casefold()
    ):
        raise ResolutionCanaryError("private task does not match selected resolution source")


def build_transfer_control_case(
    candidate: ExactResolutionCandidate,
    recovered_request: ResolutionEvaluationRequest,
    task: SWERebenchTask,
) -> ResolutionCanaryCase:
    if (
        recovered_request.model_patch_sha256 != model_patch_sha256(recovered_request.model_patch)
        or candidate.resolution_key != recovered_request.resolution_key
        or candidate.instance_id != recovered_request.instance_id
        or candidate.language != recovered_request.language
        or candidate.model_patch_sha256 != recovered_request.model_patch_sha256
        or candidate.trace_source_revision.casefold()
        != recovered_request.trace_source_revision.casefold()
        or candidate.task_source_revision.casefold()
        != recovered_request.task_source_revision.casefold()
        or recovered_request.target_references != (candidate.target_reference,)
    ):
        raise ResolutionCanaryError("recovered target patch does not match transfer candidate")
    _validate_task(recovered_request, task)
    return ResolutionCanaryCase(
        kind="transfer_control",
        source_id=candidate.candidate_id,
        resolution_key=candidate.resolution_key,
        instance_id=candidate.instance_id,
        language=candidate.language,
        model_patch=recovered_request.model_patch,
        model_patch_sha256=candidate.model_patch_sha256,
        trace_source_revision=candidate.trace_source_revision,
        task_source_revision=candidate.task_source_revision,
        expected_resolved=candidate.resolved,
        target_references=(candidate.target_reference,),
        task=task,
    )


def build_evaluation_case(
    request: ResolutionEvaluationRequest,
    task: SWERebenchTask,
) -> ResolutionCanaryCase:
    _validate_task(request, task)
    return ResolutionCanaryCase(
        kind="evaluation_request",
        source_id=request.request_id,
        resolution_key=request.resolution_key,
        instance_id=request.instance_id,
        language=request.language,
        model_patch=request.model_patch,
        model_patch_sha256=request.model_patch_sha256,
        trace_source_revision=request.trace_source_revision,
        task_source_revision=request.task_source_revision,
        target_references=request.target_references,
        task=task,
    )


def recover_transfer_control_requests(
    candidates: Iterable[ExactResolutionCandidate],
    target_rows: Iterable[tuple[str, Mapping[str, object]]],
    *,
    trace_source_revision: str,
    task_source_revision: str,
) -> dict[str, ResolutionEvaluationRequest]:
    """Replay selected raw target rows into exact evaluator requests for transfer controls."""

    candidates_by_reference = {
        (
            candidate.target_reference.partition_name,
            candidate.target_reference.rollout_id,
            candidate.target_reference.projected_row_sha256,
        ): candidate
        for candidate in candidates
    }
    if not candidates_by_reference:
        raise ResolutionCanaryError("resolution canary requires transfer controls")
    selected_rollouts = {
        (partition_name, rollout_id) for partition_name, rollout_id, _ in candidates_by_reference
    }
    recovered: dict[str, ResolutionEvaluationRequest] = {}
    for partition_name, row in target_rows:
        rollout_id = row.get("trajectory_id") or row.get("rollout_id")
        if (
            not isinstance(rollout_id, str)
            or (
                partition_name,
                rollout_id.strip(),
            )
            not in selected_rollouts
        ):
            continue
        request = evaluation_request_from_target_row(
            partition_name,
            row,
            trace_source_revision=trace_source_revision,
            task_source_revision=task_source_revision,
        )
        reference = request.target_references[0]
        candidate = candidates_by_reference.get(
            (
                reference.partition_name,
                reference.rollout_id,
                reference.projected_row_sha256,
            )
        )
        if candidate is None:
            continue
        if candidate.candidate_id in recovered:
            raise ResolutionCanaryError("transfer-control target row is duplicated")
        if (
            candidate.resolution_key != request.resolution_key
            or candidate.model_patch_sha256 != request.model_patch_sha256
            or candidate.instance_id != request.instance_id
        ):
            raise ResolutionCanaryError("transfer-control target replay changed identity")
        recovered[candidate.candidate_id] = request
    missing = sorted(
        candidate.candidate_id
        for candidate in candidates_by_reference.values()
        if candidate.candidate_id not in recovered
    )
    if missing:
        raise ResolutionCanaryError(
            f"selected transfer-control rows were not recovered: count={len(missing)}"
        )
    return recovered


def materialize_resolution_canary_cases(
    selection: ResolutionCanarySelection,
    *,
    recovered_controls: Mapping[str, ResolutionEvaluationRequest],
    tasks_by_instance: Mapping[str, SWERebenchTask],
) -> tuple[ResolutionCanaryCase, ...]:
    cases: list[ResolutionCanaryCase] = []
    for candidate in selection.transfer_controls:
        request = recovered_controls.get(candidate.candidate_id)
        task = tasks_by_instance.get(candidate.instance_id)
        if request is None or task is None:
            raise ResolutionCanaryError("transfer control lacks recovered patch or private task")
        cases.append(build_transfer_control_case(candidate, request, task))
    for request in selection.evaluation_requests:
        task = tasks_by_instance.get(request.instance_id)
        if task is None:
            raise ResolutionCanaryError("evaluation request lacks private task")
        cases.append(build_evaluation_case(request, task))
    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    if len({case.case_id for case in ordered}) != len(ordered):
        raise ResolutionCanaryError("resolution canary cases must be unique")
    return ordered


__all__ = [
    "ResolutionCanarySelection",
    "build_evaluation_case",
    "build_transfer_control_case",
    "materialize_resolution_canary_cases",
    "recover_transfer_control_requests",
    "select_resolution_canary_sources",
]
