from __future__ import annotations

from nodelm.datasets.pilot import PilotFilter, build_pilot_subset
from nodelm.models import NormalizedSample


def _sample(
    identifier: str,
    *,
    language: str,
    resolved: bool = True,
    repository: str | None = None,
    repository_license: str = "MIT",
    issue_or_pr_id: str | None = None,
    rollout_id: str | None = None,
) -> NormalizedSample:
    return NormalizedSample(
        source_dataset="fixture",
        source_dataset_revision="a" * 40,
        repository=repository or f"acme/{identifier}",
        repository_license=repository_license,
        base_commit="b" * 40,
        issue_or_pr_id=issue_or_pr_id or identifier,
        language=language,
        harness="fixture",
        generating_model="fixture@revision",
        rollout_id=rollout_id or f"rollout-{identifier}",
        resolved=resolved,
        trajectory=({"role": "assistant", "content": "inspect and patch"},),
        generated_patch="diff --git a/a.ts b/a.ts",
        patch_metadata={"bytes": 100},
        provenance_lineage=(f"raw:{identifier}",),
    )


def test_pilot_subset_prioritizes_resolved_ts_then_js_and_reports_rejections() -> None:
    samples = [
        _sample("js", language="JavaScript"),
        _sample("py", language="Python"),
        _sample("ts-failed", language="TypeScript", resolved=False),
        _sample("ts", language="TypeScript"),
    ]

    result = build_pilot_subset((sample for sample in samples), PilotFilter(max_samples=10))

    assert [sample.issue_or_pr_id for sample in result.accepted] == ["ts", "js"]
    assert result.rejection_reasons == {"language": 1, "unresolved": 1}


def test_pilot_target_is_a_cap_not_forced_padding() -> None:
    result = build_pilot_subset(
        [_sample("only", language="TypeScript")], PilotFilter(max_samples=10_000)
    )

    assert len(result.accepted) == 1


def test_pilot_selection_is_bounded_and_deterministic() -> None:
    samples = (_sample(identifier, language="TypeScript") for identifier in ("z", "b", "a", "c"))

    result = build_pilot_subset(samples, PilotFilter(max_samples=2))

    assert [sample.issue_or_pr_id for sample in result.accepted] == ["a", "b"]


def test_pilot_rejects_unapproved_licenses_and_evaluation_repositories() -> None:
    samples = [
        _sample("allowed", language="TypeScript"),
        _sample("copyleft", language="TypeScript", repository_license="GPL-3.0"),
        _sample(
            "evaluation",
            language="TypeScript",
            repository="git@github.com:Acme/Evaluation.git",
        ),
    ]

    result = build_pilot_subset(
        samples,
        PilotFilter(
            max_samples=10,
            excluded_repositories=("https://github.com/acme/evaluation",),
        ),
    )

    assert [sample.issue_or_pr_id for sample in result.accepted] == ["allowed"]
    assert result.rejection_reasons == {"evaluation_repository": 1, "license": 1}


def test_pilot_rejects_duplicate_repository_issue_rollout_identity() -> None:
    samples = [
        _sample(
            "first",
            language="TypeScript",
            repository="Acme/Widget",
            issue_or_pr_id="42",
            rollout_id="rollout-42",
        ),
        _sample(
            "duplicate",
            language="TypeScript",
            repository="https://github.com/acme/widget.git",
            issue_or_pr_id="42",
            rollout_id="rollout-42",
        ),
    ]

    result = build_pilot_subset(samples, PilotFilter(max_samples=10))

    assert len(result.accepted) == 1
    assert result.rejection_reasons == {"duplicate": 1}


def test_pilot_requires_and_bounds_training_visible_trajectories() -> None:
    empty = _sample("empty", language="TypeScript").model_copy(update={"trajectory": ()})
    long = _sample("long", language="TypeScript").model_copy(
        update={"trajectory": ({"step": 1}, {"step": 2})}
    )

    result = build_pilot_subset(
        [empty, long],
        PilotFilter(max_samples=10, max_trajectory_steps=1),
    )

    assert result.accepted == ()
    assert result.rejection_reasons == {"empty_trajectory": 1, "trajectory_length": 1}
