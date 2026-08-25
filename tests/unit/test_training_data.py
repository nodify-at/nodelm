from __future__ import annotations

import pytest

from nodelm.models import NormalizedSample
from nodelm.training.data import format_training_sample, take_training_texts


def _sample() -> NormalizedSample:
    return NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="a" * 40,
        repository="acme/widget",
        repository_license="MIT",
        base_commit="b" * 40,
        issue_or_pr_id="one",
        language="TypeScript",
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id="rollout-one",
        resolved=True,
        trajectory=({"role": "assistant", "content": "inspect"},),
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 1},
        provenance_lineage=("raw:one",),
    )


def test_training_text_is_deterministic_and_gold_free() -> None:
    text = format_training_sample(_sample())

    assert '"trajectory"' in text
    assert '"generated_patch"' in text
    assert "gold_patch" not in text


@pytest.mark.parametrize(
    "trajectory",
    [
        ({"role": "tool", "payload": {"gold_patch": "SECRET_GOLD"}},),
        ({"role": "tool", "reference": {"patch": "SECRET_REFERENCE"}},),
        ({"golden_patch": "SECRET_GOLDEN_DIRECT"},),
        ({"role": "tool", "payload": {"golden_patch": "SECRET_GOLDEN_NESTED"}},),
    ],
)
def test_training_text_rejects_gold_or_reference_patch(
    trajectory: tuple[dict[str, object], ...],
) -> None:
    unsafe_sample = _sample().model_copy(update={"trajectory": trajectory})

    with pytest.raises(ValueError, match="forbidden gold/reference patch"):
        format_training_sample(unsafe_sample)


def test_training_text_selection_requires_the_requested_real_batch() -> None:
    with pytest.raises(ValueError, match="requested 2"):
        take_training_texts((_sample(),), count=2)
