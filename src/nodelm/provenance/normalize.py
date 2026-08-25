from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from nodelm.artifacts import content_digest
from nodelm.licenses.gate import LicenseDisposition, evaluate_license
from nodelm.models import NormalizedSample


class NormalizationError(ValueError):
    """A raw record cannot satisfy the normalized provenance contract."""


class UnknownResolutionError(NormalizationError):
    """A trace has no verified solved/unsolved label and cannot enter sample v1."""


_PATCH_TOKENS = frozenset({"patch", "diff"})
_REFERENCE_TOKENS = frozenset({"gold", "reference", "ref", "oracle", "groundtruth"})


def _field_tokens(field: object) -> frozenset[str]:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(field))
    return frozenset(re.findall(r"[a-z0-9]+", camel_split.casefold()))


def validate_gold_free_trajectory(
    value: Any,
    *,
    _reference_scope: bool = False,
) -> None:
    """Reject recursively nested gold/reference patch fields from model-visible traces."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            tokens = _field_tokens(raw_key)
            references_gold = bool(tokens & _REFERENCE_TOKENS) or {
                "ground",
                "truth",
            }.issubset(tokens)
            contains_patch = bool(tokens & _PATCH_TOKENS)
            if contains_patch and (_reference_scope or references_gold):
                raise NormalizationError("trajectory contains a forbidden gold/reference patch")
            validate_gold_free_trajectory(
                item,
                _reference_scope=_reference_scope or references_gold,
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_gold_free_trajectory(item, _reference_scope=_reference_scope)


def parse_resolved(value: Any) -> bool:
    """Parse the dataset's resolved marker without Python truthiness coercion."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise NormalizationError("resolved must be a boolean or integer 0/1")


def parse_resolution_status(value: Any) -> bool | None:
    """Parse source audit labels, including the documented -1/unknown marker."""

    if value is None or (isinstance(value, int) and not isinstance(value, bool) and value == -1):
        return None
    return parse_resolved(value)


def _required_string(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise NormalizationError(f"required field is missing: {' or '.join(fields)}")


def _required_identifier(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    raise NormalizationError(f"required identifier is missing: {' or '.join(fields)}")


def _patch(row: Mapping[str, Any]) -> tuple[str, str]:
    for field in ("model_patch", "pred_patch", "generated_patch"):
        value = row.get(field)
        if isinstance(value, str):
            return value, field
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        model_patch = metadata.get("model_patch")
        if isinstance(model_patch, Mapping) and isinstance(model_patch.get("patch"), str):
            return str(model_patch["patch"]), "metadata.model_patch.patch"
    return "", "absent"


def _trajectory(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    value = row.get("trajectory")
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(step, Mapping) for step in value):
        raise NormalizationError("trajectory must be a list of mappings")
    validate_gold_free_trajectory(value)
    return tuple({str(key): item for key, item in step.items()} for step in value)


def _patch_metadata(patch: str, source_field: str) -> dict[str, Any]:
    encoded_patch = patch.encode("utf-8")
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return {
        "sha256": content_digest(encoded_patch),
        "bytes": len(encoded_patch),
        "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in lines),
        "removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in lines),
        "source_field": source_field,
    }


def normalize_sample(
    row: Mapping[str, Any],
    *,
    source_dataset: str,
    source_revision: str,
    harness: str,
    generating_model: str,
    lineage: Sequence[str],
) -> NormalizedSample:
    raw_license = _required_string(row, "license", "repository_license")
    license_decision = evaluate_license(raw_license)
    if license_decision.disposition is not LicenseDisposition.ALLOW:
        raise NormalizationError(
            f"repository license is not allowed: {raw_license!r} ({license_decision.reason})"
        )

    patch, patch_field = _patch(row)
    try:
        return NormalizedSample(
            source_dataset=source_dataset,
            source_dataset_revision=source_revision,
            repository=_required_string(row, "repo", "repository"),
            repository_license=license_decision.normalized_spdx or raw_license,
            base_commit=_required_string(row, "base_commit"),
            issue_or_pr_id=_required_identifier(
                row, "instance_id", "issue_or_pr_id", "pull_number", "issue_id"
            ),
            language=_required_string(row, "language"),
            harness=harness,
            generating_model=generating_model,
            rollout_id=_required_string(row, "trajectory_id", "rollout_id"),
            resolved=parse_resolved(row.get("resolved")),
            trajectory=_trajectory(row),
            generated_patch=patch or None,
            patch_metadata=_patch_metadata(patch, patch_field),
            provenance_lineage=tuple(lineage),
        )
    except ValidationError as error:
        raise NormalizationError(str(error)) from error
