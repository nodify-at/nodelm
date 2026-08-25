from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.licenses.gate import LicenseDisposition, evaluate_license
from nodelm.models import NormalizedSample, VerificationStatus, stable_model_id


class PilotAuthorizationError(ValueError):
    """A pilot manifest is not part of the reviewed training trust root."""


AUTHORIZED_PILOT_MANIFEST_SHA256_BY_SAMPLES_SHA256: dict[str, str] = {}


def require_authorized_pilot_manifest(
    *,
    samples_sha256: str,
    pilot_manifest_sha256: str,
) -> None:
    expected = AUTHORIZED_PILOT_MANIFEST_SHA256_BY_SAMPLES_SHA256.get(samples_sha256)
    if expected is None:
        raise PilotAuthorizationError(
            "no reviewed pilot manifest is authorized for this samples artifact"
        )
    if pilot_manifest_sha256 != expected:
        raise PilotAuthorizationError(
            "pilot manifest digest is not authorized for this samples artifact"
        )


class PilotPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.pilot-filter/v1"]
    status: VerificationStatus
    target_samples: int = Field(gt=0)
    languages: tuple[str, ...] = Field(min_length=1)
    require_resolved: bool
    require_nonempty_trajectory: bool
    repository_disjoint_from_evaluation: Literal[True]
    allowed_repository_licenses: tuple[str, ...] = Field(min_length=1)
    max_patch_bytes: int | None = Field(default=None, gt=0)
    max_trajectory_steps: int | None = Field(default=None, gt=0)
    notes: str | None = None


@dataclass(frozen=True)
class PilotFilter:
    max_samples: int = 10_000
    languages: tuple[str, ...] = ("TypeScript", "JavaScript")
    require_resolved: bool = True
    require_nonempty_trajectory: bool = True
    max_patch_bytes: int | None = None
    max_trajectory_steps: int | None = None
    allowed_repository_licenses: tuple[str, ...] = (
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
    )
    training_repositories: tuple[str, ...] | None = None
    excluded_repositories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")
        if not self.languages:
            raise ValueError("languages must not be empty")
        if self.max_patch_bytes is not None and self.max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes must be greater than zero")
        if self.max_trajectory_steps is not None and self.max_trajectory_steps <= 0:
            raise ValueError("max_trajectory_steps must be greater than zero")
        normalized_licenses: set[str] = set()
        for item in self.allowed_repository_licenses:
            decision = evaluate_license(item)
            if (
                decision.disposition is not LicenseDisposition.ALLOW
                or decision.normalized_spdx is None
            ):
                raise ValueError(f"pilot policy contains a disallowed license: {item}")
            normalized_licenses.add(decision.normalized_spdx)
        object.__setattr__(self, "allowed_repository_licenses", tuple(sorted(normalized_licenses)))
        if self.training_repositories is not None:
            object.__setattr__(
                self,
                "training_repositories",
                tuple(sorted({canonical_repository(item) for item in self.training_repositories})),
            )
        object.__setattr__(
            self,
            "excluded_repositories",
            tuple(sorted({canonical_repository(item) for item in self.excluded_repositories})),
        )


@dataclass(frozen=True)
class PilotSubset:
    accepted: tuple[NormalizedSample, ...]
    rejection_reasons: dict[str, int]
    filter_digest: str


def build_pilot_subset(samples: Iterable[NormalizedSample], policy: PilotFilter) -> PilotSubset:
    rejections: Counter[str] = Counter()
    language_rank = {language.casefold(): index for index, language in enumerate(policy.languages)}
    allowed_licenses = set(policy.allowed_repository_licenses)
    training_repositories = (
        None if policy.training_repositories is None else set(policy.training_repositories)
    )
    excluded_repositories = set(policy.excluded_repositories)
    seen_identities: set[tuple[str, str, str]] = set()

    def eligible() -> Iterator[tuple[tuple[int, str, str, str], int, NormalizedSample]]:
        for ordinal, sample in enumerate(samples):
            try:
                repository = canonical_repository(sample.repository)
            except ValueError:
                rejections["repository"] += 1
                continue
            if repository in excluded_repositories:
                rejections["evaluation_repository"] += 1
                continue
            if training_repositories is not None and repository not in training_repositories:
                rejections["not_in_frozen_split"] += 1
                continue
            license_decision = evaluate_license(sample.repository_license)
            if (
                license_decision.disposition is not LicenseDisposition.ALLOW
                or license_decision.normalized_spdx not in allowed_licenses
            ):
                rejections["license"] += 1
                continue
            normalized_language = sample.language.casefold()
            if normalized_language not in language_rank:
                rejections["language"] += 1
                continue
            if policy.require_resolved and not sample.resolved:
                rejections["unresolved"] += 1
                continue
            if policy.require_nonempty_trajectory and not sample.trajectory:
                rejections["empty_trajectory"] += 1
                continue
            if (
                policy.max_trajectory_steps is not None
                and len(sample.trajectory) > policy.max_trajectory_steps
            ):
                rejections["trajectory_length"] += 1
                continue
            patch_bytes = sample.patch_metadata.get("bytes")
            if (
                policy.max_patch_bytes is not None
                and isinstance(patch_bytes, int)
                and patch_bytes > policy.max_patch_bytes
            ):
                rejections["patch_size"] += 1
                continue
            identity = (repository, sample.issue_or_pr_id, sample.rollout_id)
            if identity in seen_identities:
                rejections["duplicate"] += 1
                continue
            seen_identities.add(identity)
            sort_key = (
                language_rank[normalized_language],
                repository,
                sample.issue_or_pr_id,
                sample.rollout_id,
            )
            yield sort_key, ordinal, sample

    selected = heapq.nsmallest(
        policy.max_samples,
        eligible(),
        key=lambda item: (item[0], item[1]),
    )
    return PilotSubset(
        accepted=tuple(item[2] for item in selected),
        rejection_reasons=dict(sorted(rejections.items())),
        filter_digest=stable_model_id(asdict(policy)),
    )
