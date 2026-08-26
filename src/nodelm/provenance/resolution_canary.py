from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
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
    model_patch_sha256,
)


@dataclass(frozen=True)
class ResolutionCanarySelection:
    transfer_controls: tuple[ExactResolutionCandidate, ...]
    evaluation_requests: tuple[ResolutionEvaluationRequest, ...]


T = TypeVar("T")


def _greedy_cover(
    items: tuple[T, ...],
    *,
    identity: Callable[[T], str],
    features: Callable[[T], set[tuple[str, Hashable]]],
    minimum: int,
    maximum: int,
) -> tuple[T, ...]:
    if not items:
        raise ResolutionCanaryError("resolution canary requires non-empty source artifacts")
    if minimum <= 0 or maximum < minimum:
        raise ResolutionCanaryError("resolution canary selection bounds are invalid")
    features_by_id = {str(identity(item)): frozenset(features(item)) for item in items}
    uncovered = frozenset().union(*features_by_id.values())
    remaining = {str(identity(item)): item for item in items}
    selected: list[T] = []
    while uncovered and remaining and len(selected) < maximum:
        candidate_id = min(
            remaining,
            key=lambda item_id: (-len(features_by_id[item_id] & uncovered), item_id),
        )
        if not (features_by_id[candidate_id] & uncovered):
            break
        selected.append(remaining.pop(candidate_id))
        uncovered = uncovered - features_by_id[candidate_id]
    if uncovered:
        raise ResolutionCanaryError("canary selection bound cannot cover observed strata")
    for item_id in sorted(remaining):
        if len(selected) >= minimum:
            break
        if len(selected) >= maximum:
            raise ResolutionCanaryError("canary selection maximum is below its minimum")
        selected.append(remaining[item_id])
    if len(selected) < minimum:
        raise ResolutionCanaryError("resolution artifacts contain too few unique canary sources")
    return tuple(sorted(selected, key=identity))


def select_resolution_canary_sources(
    candidates: Iterable[ExactResolutionCandidate],
    requests: Iterable[ResolutionEvaluationRequest],
    *,
    minimum_per_kind: int = 6,
    maximum_per_kind: int = 12,
) -> ResolutionCanarySelection:
    """Select a small deterministic set that covers observed language/partition/label strata."""

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    ordered_requests = tuple(sorted(requests, key=lambda item: item.request_id))
    controls = _greedy_cover(
        ordered_candidates,
        identity=lambda item: item.candidate_id,
        features=lambda item: {
            ("language", item.language),
            ("partition", item.target_reference.partition_name),
            ("label", item.resolved),
        },
        minimum=minimum_per_kind,
        maximum=maximum_per_kind,
    )
    evaluation_requests = _greedy_cover(
        ordered_requests,
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


__all__ = [
    "ResolutionCanarySelection",
    "build_evaluation_case",
    "build_transfer_control_case",
    "select_resolution_canary_sources",
]
