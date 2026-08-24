from __future__ import annotations

import json
from collections.abc import Iterable

from nodelm.models import NormalizedSample
from nodelm.provenance.normalize import validate_gold_free_trajectory


def format_training_sample(sample: NormalizedSample) -> str:
    """Serialize one gold-free trajectory into deterministic SFT-visible text."""

    if not sample.trajectory:
        raise ValueError(f"training sample has no trajectory: {sample.sample_id}")
    validate_gold_free_trajectory(sample.trajectory)
    payload = {
        "repository": sample.repository,
        "base_commit": sample.base_commit,
        "issue_or_pr_id": sample.issue_or_pr_id,
        "trajectory": sample.trajectory,
        "generated_patch": sample.generated_patch,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def take_training_texts(
    samples: Iterable[NormalizedSample],
    *,
    count: int,
) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("training sample count must be greater than zero")
    selected: list[str] = []
    for sample in samples:
        selected.append(format_training_sample(sample))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} training samples but found {len(selected)}")
    return tuple(selected)
