from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nodelm.artifacts import (
    ArtifactWriteResult,
    canonical_json_bytes,
    content_digest,
    file_identity,
    write_immutable_json,
    write_immutable_stream,
)
from nodelm.datasets.staging import VerifiedStagingError, verified_staged_file
from nodelm.models import NormalizedSample, VerificationStatus, stable_model_id
from nodelm.provenance.gold import OracleIsolationAttestation
from nodelm.provenance.manifests import (
    TASK_PROVENANCE_SAFE_FIELDS,
    NormalizationManifestV2,
    SnapshotMaterializationManifestV2,
    TaskProvenanceProjectionManifestV1,
)
from nodelm.provenance.normalize import (
    NormalizationError,
    extract_model_patch,
    is_reference_patch_field,
    model_patch_metadata,
    parse_resolution_status,
    validate_gold_free_trajectory,
)
from nodelm.provenance.pipeline import (
    has_exact_normalized_sample_lineage,
    normalization_evidence_lineage,
)
from nodelm.provenance.task_provenance import TaskProvenanceRecord


class OracleIsolationReviewError(ValueError):
    """The supplied artifacts cannot support a fail-closed oracle-isolation review."""


OracleIsolationFindingReasonCode: TypeAlias = Literal[
    "invalid_recorded_model_context",
    "gold_field_in_recorded_model_context",
    "reference_patch_in_initial_prompt",
    "unsupported_reference_patch_location",
    "raw_normalized_binding_mismatch",
    "population_binding_mismatch",
]

OracleIsolationFindingReason: TypeAlias = Literal[
    "recorded model context is invalid",
    "recorded model context contains forbidden gold/reference patch metadata",
    "initial model prompt contains the bound reference patch",
    "raw row contains a reference patch outside the reviewed metadata boundary",
    "raw row does not match its normalized training-visible projection",
    "raw and normalized populations do not have complete identity coverage",
]

_SAFE_REASON_BY_CODE: dict[
    OracleIsolationFindingReasonCode,
    OracleIsolationFindingReason,
] = {
    "invalid_recorded_model_context": "recorded model context is invalid",
    "gold_field_in_recorded_model_context": (
        "recorded model context contains forbidden gold/reference patch metadata"
    ),
    "reference_patch_in_initial_prompt": (
        "initial model prompt contains the bound reference patch"
    ),
    "unsupported_reference_patch_location": (
        "raw row contains a reference patch outside the reviewed metadata boundary"
    ),
    "raw_normalized_binding_mismatch": (
        "raw row does not match its normalized training-visible projection"
    ),
    "population_binding_mismatch": (
        "raw and normalized populations do not have complete identity coverage"
    ),
}

_REASON_ORDER: tuple[OracleIsolationFindingReasonCode, ...] = (
    "invalid_recorded_model_context",
    "gold_field_in_recorded_model_context",
    "reference_patch_in_initial_prompt",
    "unsupported_reference_patch_location",
    "raw_normalized_binding_mismatch",
    "population_binding_mismatch",
)
_ALLOWED_REFERENCE_PATCH_PATH = ("metadata", "reference_patch")


@dataclass(frozen=True, slots=True)
class RecordedModelContextInspection:
    reference_patch_present: bool
    reason_codes: tuple[OracleIsolationFindingReasonCode, ...]


class SanitizedOracleIsolationFinding(BaseModel):
    """A review finding that cannot serialize recorded context or patch content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: int = Field(ge=0)
    sample_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reason_code: OracleIsolationFindingReasonCode
    reason: OracleIsolationFindingReason

    @classmethod
    def from_reason_code(
        cls,
        *,
        row_index: int,
        sample_id: str | None,
        reason_code: OracleIsolationFindingReasonCode,
    ) -> SanitizedOracleIsolationFinding:
        return cls(
            row_index=row_index,
            sample_id=sample_id,
            reason_code=reason_code,
            reason=_SAFE_REASON_BY_CODE[reason_code],
        )

    @model_validator(mode="after")
    def require_fixed_safe_reason(self) -> SanitizedOracleIsolationFinding:
        if self.reason != _SAFE_REASON_BY_CODE[self.reason_code]:
            raise ValueError("finding reason must match its fixed safe reason code")
        return self


@dataclass(frozen=True, slots=True)
class UpstreamOracleReview:
    review_id: str
    harness: str
    generating_model: str
    evidence_url: str


_SOURCE_NAME = "open-swe-traces"
_SOURCE_REPOSITORY_ID = "nvidia/Open-SWE-Traces"
_SOURCE_REVISION = "ed95cef24df8d8bd79b4ceb0192cb420fde06521"
_UPSTREAM_REVIEW_ID = "open-swe-traces-v1.0-paper-git-hacking-review/v1"
_UPSTREAM_EVIDENCE_URL = "https://arxiv.org/abs/2606.16038"

AUTHORIZED_UPSTREAM_ORACLE_REVIEWS: dict[
    tuple[str, str, str, str],
    UpstreamOracleReview,
] = {
    (_SOURCE_NAME, _SOURCE_REPOSITORY_ID, _SOURCE_REVISION, partition): UpstreamOracleReview(
        review_id=_UPSTREAM_REVIEW_ID,
        harness=harness,
        generating_model=generating_model,
        evidence_url=_UPSTREAM_EVIDENCE_URL,
    )
    for partition, harness, generating_model in (
        (
            "openhands/minimax_m25/swe-rebench-v2",
            "openhands",
            "source-label:minimax_m25",
        ),
        (
            "openhands/qwen35_122b/swe-rebench-v2",
            "openhands",
            "source-label:qwen35_122b",
        ),
        (
            "sweagent/minimax_m25/swe-rebench-v2",
            "sweagent",
            "source-label:minimax_m25",
        ),
        (
            "sweagent/qwen35_122b/swe-rebench-v2",
            "sweagent",
            "source-label:qwen35_122b",
        ),
    )
}


def _reference_patch(row: Mapping[str, Any]) -> tuple[str | None, bool]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return None, False
    if "reference_patch" not in metadata:
        return None, False
    value = metadata.get("reference_patch")
    if not isinstance(value, Mapping):
        return None, False
    patch = value.get("patch")
    if not isinstance(patch, str) or not patch:
        return None, False
    return patch, True


def _has_unsupported_reference_patch_field(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    ignored_root_keys: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if not path and key in ignored_root_keys:
                continue
            child_path = (*path, key)
            if is_reference_patch_field(key) and child_path != _ALLOWED_REFERENCE_PATCH_PATH:
                return True
            if _has_unsupported_reference_patch_field(
                item,
                path=child_path,
                ignored_root_keys=ignored_root_keys,
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _has_unsupported_reference_patch_field(
                item,
                path=path,
                ignored_root_keys=ignored_root_keys,
            )
            for item in value
        )
    return False


def _parsed_tool_definitions(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("tools must be a sequence")
    parsed: list[Any] = []
    for item in value:
        if isinstance(item, str):
            parsed.append(json.loads(item))
        elif isinstance(item, Mapping):
            parsed.append(item)
        else:
            raise ValueError("tool definitions must be mappings or serialized JSON")
    return tuple(parsed)


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _recorded_model_input_contains_patch(trajectory: Any, tools: Any, patch: str | None) -> bool:
    """Fail closed over all recorded input that could be shown to a model.

    A literal ``assistant`` role is the sole representation treated as model output.
    Tool results, developer/system/user messages, and unknown or malformed roles remain
    input-visible so a novel producer role cannot silently create an exclusion.
    """

    if patch is None or not isinstance(trajectory, list):
        return False
    normalized_patch = patch.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_patch:
        return False
    for step in trajectory:
        if not isinstance(step, Mapping):
            return False
        if step.get("role") == "assistant":
            continue
        if any(
            normalized_patch in text.replace("\r\n", "\n").replace("\r", "\n")
            for text in _iter_strings(step)
        ):
            return True
    try:
        parsed_tools = _parsed_tool_definitions(tools)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return any(
        normalized_patch in text.replace("\r\n", "\n").replace("\r", "\n")
        for tool in parsed_tools
        for text in _iter_strings(tool)
    )


def _validate_recorded_model_inputs(trajectory: list[Mapping[str, Any]], tools: Any) -> None:
    """Apply the gold-field policy only to input-visible trajectory content and tools."""

    for step in trajectory:
        if step.get("role") != "assistant":
            validate_gold_free_trajectory(step)
    validate_gold_free_trajectory(_parsed_tool_definitions(tools))


def inspect_recorded_model_context(
    row: Mapping[str, Any],
) -> RecordedModelContextInspection:
    """Inspect only recorded model-visible context; never return source content."""

    reasons: set[OracleIsolationFindingReasonCode] = set()
    trajectory = row.get("trajectory")
    tools = row.get("tools")
    try:
        if not isinstance(trajectory, list) or any(
            not isinstance(step, Mapping) for step in trajectory
        ):
            raise ValueError("trajectory must be a list of mappings")
        _validate_recorded_model_inputs(trajectory, tools)
    except NormalizationError:
        reasons.add("gold_field_in_recorded_model_context")
    except (json.JSONDecodeError, TypeError, ValueError):
        reasons.add("invalid_recorded_model_context")

    if _has_unsupported_reference_patch_field(
        row,
        ignored_root_keys=frozenset({"trajectory", "tools"}),
    ):
        reasons.add("unsupported_reference_patch_location")

    reference_patch, reference_patch_valid = _reference_patch(row)
    if not reference_patch_valid:
        reasons.add("unsupported_reference_patch_location")
    if _recorded_model_input_contains_patch(trajectory, tools, reference_patch):
        reasons.add("reference_patch_in_initial_prompt")

    return RecordedModelContextInspection(
        reference_patch_present=reference_patch is not None,
        reason_codes=tuple(reason for reason in _REASON_ORDER if reason in reasons),
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _iter_jsonl_mappings(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=_unique_json_object)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                raise OracleIsolationReviewError(
                    f"invalid JSONL evidence at row {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise OracleIsolationReviewError(
                    f"JSONL evidence row {line_number} is not an object"
                )
            yield value


def _raw_model_patch(row: Mapping[str, Any]) -> str | None:
    return extract_model_patch(row)[0] or None


def _raw_identifier(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _raw_rollout_id(row: Mapping[str, Any]) -> str | None:
    for field in ("trajectory_id", "rollout_id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True, slots=True)
class _IndexedNormalizedSample:
    sample_id: str
    instance_id: str
    rollout_id: str
    trajectory_sha256: str
    model_patch_sha256: str
    resolved: bool
    patch_metadata: dict[str, Any]
    task_binding_matches: bool


@dataclass(slots=True)
class _NormalizedOracleIndex:
    connection: sqlite3.Connection
    inserted_count: int = 0
    covered_count: int = 0

    def add_task(self, record: TaskProvenanceRecord) -> None:
        try:
            self.connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
                (
                    record.instance_id,
                    record.repository,
                    record.repository_license,
                    record.base_commit.casefold(),
                    record.language,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise OracleIsolationReviewError(
                "task provenance artifact contains duplicate instance identities"
            ) from error

    @property
    def task_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    def add(self, sample: NormalizedSample, manifest: NormalizationManifestV2) -> None:
        evidence_lineage = normalization_evidence_lineage(
            materialization_manifest_sha256=manifest.materialization_manifest_sha256,
            partition_name=manifest.partition_name,
            upstream_source=manifest.upstream_source,
            task_source_name=manifest.task_source_name,
            task_source_revision=manifest.task_source_revision,
            task_provenance_sha256=manifest.task_provenance_sha256,
        )
        if (
            sample.source_dataset != manifest.source_name
            or sample.source_dataset_revision.casefold() != manifest.source_revision.casefold()
            or sample.harness != manifest.harness
            or sample.generating_model != manifest.generating_model
            or not has_exact_normalized_sample_lineage(
                sample.provenance_lineage,
                source_repository_id=manifest.source_repository_id,
                source_revision=manifest.source_revision,
                instance_id=sample.issue_or_pr_id,
                evidence_lineage=evidence_lineage,
            )
        ):
            raise OracleIsolationReviewError(
                "normalized sample does not satisfy the exact raw-row lineage contract"
            )
        task = self.connection.execute(
            "SELECT repository, repository_license, base_commit, language "
            "FROM tasks WHERE instance_id = ?",
            (sample.issue_or_pr_id,),
        ).fetchone()
        task_binding_matches = task is not None and (
            sample.repository,
            sample.repository_license,
            sample.base_commit.casefold(),
            sample.language,
        ) == tuple(str(item) for item in task)
        raw_sha256 = sample.provenance_lineage[2].removeprefix("raw-row:")
        try:
            self.connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    raw_sha256,
                    sample.sample_id,
                    sample.issue_or_pr_id,
                    sample.rollout_id,
                    stable_model_id(sample.trajectory),
                    stable_model_id(sample.generated_patch),
                    int(sample.resolved),
                    json.dumps(
                        sample.patch_metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    int(task_binding_matches),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise OracleIsolationReviewError(
                "multiple normalized samples claim the same raw-row identity"
            ) from error
        self.inserted_count += 1

    def finalize(self, expected_sample_count: int) -> None:
        self.connection.commit()
        if self.inserted_count != expected_sample_count:
            raise OracleIsolationReviewError(
                "normalized artifact row count does not match its manifest"
            )

    @property
    def sample_count(self) -> int:
        return self.inserted_count

    @property
    def covered_sample_count(self) -> int:
        return self.covered_count

    def claim(self, raw_sha256: str) -> _IndexedNormalizedSample | None:
        result = self.connection.execute(
            "UPDATE samples SET seen = 1 WHERE raw_sha256 = ? AND seen = 0 "
            "RETURNING sample_id, instance_id, rollout_id, trajectory_sha256, "
            "model_patch_sha256, resolved, patch_metadata, task_binding_matches",
            (raw_sha256,),
        ).fetchone()
        if result is None:
            return None
        self.covered_count += 1
        return _IndexedNormalizedSample(
            sample_id=str(result[0]),
            instance_id=str(result[1]),
            rollout_id=str(result[2]),
            trajectory_sha256=str(result[3]),
            model_patch_sha256=str(result[4]),
            resolved=bool(result[5]),
            patch_metadata=json.loads(str(result[6])),
            task_binding_matches=bool(result[7]),
        )


@contextmanager
def _normalized_oracle_index() -> Iterator[_NormalizedOracleIndex]:
    descriptor, database_name = tempfile.mkstemp(
        prefix="nodelm-oracle-isolation-", suffix=".sqlite3"
    )
    os.close(descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "CREATE TABLE samples ("
            "raw_sha256 TEXT PRIMARY KEY NOT NULL, sample_id TEXT NOT NULL, "
            "instance_id TEXT NOT NULL, rollout_id TEXT NOT NULL, "
            "trajectory_sha256 TEXT NOT NULL, model_patch_sha256 TEXT NOT NULL, "
            "resolved INTEGER NOT NULL, patch_metadata TEXT NOT NULL, "
            "task_binding_matches INTEGER NOT NULL, "
            "seen INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE tasks (instance_id TEXT PRIMARY KEY NOT NULL, "
            "repository TEXT NOT NULL, repository_license TEXT NOT NULL, "
            "base_commit TEXT NOT NULL, language TEXT NOT NULL) WITHOUT ROWID"
        )
        yield _NormalizedOracleIndex(connection)
    finally:
        connection.close()
        with suppress(FileNotFoundError):
            database_path.unlink()


@dataclass(frozen=True, slots=True)
class OracleIsolationScanResult:
    raw_row_count: int
    covered_sample_count: int
    reference_patch_row_count: int
    finding_count: int
    findings_bytes: int
    reason_counts: dict[OracleIsolationFindingReasonCode, int]


def _scan_raw_population(
    raw_path: Path,
    *,
    index: _NormalizedOracleIndex,
    normalization: NormalizationManifestV2,
    materialization: SnapshotMaterializationManifestV2,
    stream: BinaryIO,
) -> OracleIsolationScanResult:
    raw_row_count = 0
    reference_patch_row_count = 0
    finding_count = 0
    findings_bytes = 0
    reason_counts: dict[OracleIsolationFindingReasonCode, int] = {}

    def record(
        *,
        row_index: int,
        sample_id: str | None,
        reason_code: OracleIsolationFindingReasonCode,
    ) -> None:
        nonlocal finding_count, findings_bytes
        finding = SanitizedOracleIsolationFinding.from_reason_code(
            row_index=row_index,
            sample_id=sample_id,
            reason_code=reason_code,
        )
        payload = canonical_json_bytes(finding.model_dump(mode="json"))
        stream.write(payload)
        finding_count += 1
        findings_bytes += len(payload)
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1

    for row_index, row in enumerate(_iter_jsonl_mappings(raw_path)):
        raw_row_count += 1
        raw_sha256 = stable_model_id(row)
        indexed = index.claim(raw_sha256)
        if indexed is None:
            continue

        trajectory = row.get("trajectory")
        raw_patch, raw_patch_field = extract_model_patch(row)
        raw_resolution = parse_resolution_status(row.get("resolved"))
        binding_matches = (
            indexed.task_binding_matches
            and _raw_identifier(row, "instance_id") == indexed.instance_id
            and _raw_rollout_id(row) == indexed.rollout_id
            and row.get("hf_dataset_name") == normalization.row_dataset_name
            and isinstance(trajectory, list)
            and stable_model_id(trajectory) == indexed.trajectory_sha256
            and stable_model_id(raw_patch or None) == indexed.model_patch_sha256
            and raw_resolution is not None
            and raw_resolution == indexed.resolved
            and model_patch_metadata(raw_patch, raw_patch_field) == indexed.patch_metadata
        )
        if not binding_matches:
            record(
                row_index=row_index,
                sample_id=indexed.sample_id,
                reason_code="raw_normalized_binding_mismatch",
            )

        inspection = inspect_recorded_model_context(row)
        if inspection.reference_patch_present:
            reference_patch_row_count += 1
        for reason_code in inspection.reason_codes:
            record(
                row_index=row_index,
                sample_id=indexed.sample_id,
                reason_code=reason_code,
            )

    covered_sample_count = index.covered_sample_count
    if (
        raw_row_count != materialization.row_count
        or covered_sample_count != normalization.accepted_count
    ):
        record(
            row_index=raw_row_count,
            sample_id=None,
            reason_code="population_binding_mismatch",
        )
    return OracleIsolationScanResult(
        raw_row_count=raw_row_count,
        covered_sample_count=covered_sample_count,
        reference_patch_row_count=reference_patch_row_count,
        finding_count=finding_count,
        findings_bytes=findings_bytes,
        reason_counts=reason_counts,
    )


def _read_manifest(
    path: Path,
    model: (
        type[SnapshotMaterializationManifestV2]
        | type[NormalizationManifestV2]
        | type[TaskProvenanceProjectionManifestV1]
    ),
) -> tuple[
    SnapshotMaterializationManifestV2
    | NormalizationManifestV2
    | TaskProvenanceProjectionManifestV1,
    tuple[str, int],
]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        parsed = model.model_validate(value)
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
        ValidationError,
    ) as error:
        raise OracleIsolationReviewError(f"invalid evidence manifest: {path.name}") from error
    return parsed, (content_digest(payload), len(payload))


def resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    path = Path(artifact)
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return path.resolve()


def _require_identity(path: Path, expected: tuple[str, int]) -> None:
    try:
        observed = file_identity(path)
    except OSError as error:
        raise OracleIsolationReviewError(f"unable to recheck evidence input: {path}") from error
    if observed != expected:
        raise OracleIsolationReviewError(f"evidence input changed while reviewed: {path}")


def _require_supported_upstream_review(
    normalization: NormalizationManifestV2,
) -> UpstreamOracleReview:
    key = (
        normalization.source_name,
        normalization.source_repository_id,
        normalization.source_revision,
        normalization.partition_name,
    )
    review = AUTHORIZED_UPSTREAM_ORACLE_REVIEWS.get(key)
    if review is None:
        raise OracleIsolationReviewError(
            "no reviewed upstream oracle-isolation claim covers this exact source partition"
        )
    if (
        review.harness != normalization.harness
        or review.generating_model != normalization.generating_model
    ):
        raise OracleIsolationReviewError(
            "upstream oracle review does not match the normalized harness/model leaf"
        )
    return review


def review_oracle_isolation_artifacts(
    *,
    raw_input: Path,
    materialization_manifest: Path,
    normalized_input: Path,
    normalization_manifest: Path,
    task_provenance: Path,
    task_provenance_manifest: Path,
    output: Path,
    findings_output: Path,
) -> tuple[ArtifactWriteResult, OracleIsolationAttestation]:
    """Review one complete normalized leaf against its exact materialized raw rows."""

    all_inputs = (
        raw_input,
        materialization_manifest,
        normalized_input,
        normalization_manifest,
        task_provenance,
        task_provenance_manifest,
    )
    if output.resolve() == findings_output.resolve():
        raise OracleIsolationReviewError("review output and findings output must be distinct")
    if any(
        destination.resolve() == source.resolve()
        for destination in (output, findings_output)
        for source in all_inputs
    ):
        raise OracleIsolationReviewError("review outputs must not collide with evidence inputs")
    materialization_value, materialization_identity = _read_manifest(
        materialization_manifest, SnapshotMaterializationManifestV2
    )
    normalization_value, normalization_identity = _read_manifest(
        normalization_manifest, NormalizationManifestV2
    )
    task_manifest_value, task_manifest_identity = _read_manifest(
        task_provenance_manifest, TaskProvenanceProjectionManifestV1
    )
    assert isinstance(materialization_value, SnapshotMaterializationManifestV2)
    assert isinstance(normalization_value, NormalizationManifestV2)
    assert isinstance(task_manifest_value, TaskProvenanceProjectionManifestV1)
    materialization = materialization_value
    normalization = normalization_value
    task_manifest = task_manifest_value
    raw_identity = file_identity(raw_input)
    normalized_identity = file_identity(normalized_input)
    task_identity = file_identity(task_provenance)

    if (
        materialization.status != "PASS"
        or materialization.materialization_scope != "complete-partition"
        or materialization.max_rows is not None
        or normalization.status != "PASS"
        or normalization.uniqueness_scope != "complete-partition"
        or normalization.accepted_count < 1
    ):
        raise OracleIsolationReviewError(
            "oracle isolation requires complete-partition PASS materialization and normalization"
        )
    if (
        resolve_artifact(materialization_manifest, materialization.output) != raw_input.resolve()
        or (materialization.output_sha256, materialization.output_bytes) != raw_identity
        or resolve_artifact(normalization_manifest, normalization.normalized_artifact)
        != normalized_input.resolve()
        or (normalization.normalized_sha256, normalization.normalized_bytes) != normalized_identity
    ):
        raise OracleIsolationReviewError("evidence manifests do not bind their artifact paths")
    shared_fields = (
        "source_name",
        "source_repository_id",
        "source_revision",
        "partition_name",
        "harness",
        "generating_model",
        "upstream_source",
        "row_dataset_name",
        "task_source_name",
        "task_source_revision",
    )
    if any(
        getattr(materialization, field) != getattr(normalization, field) for field in shared_fields
    ):
        raise OracleIsolationReviewError(
            "materialization and normalization describe different source leaves"
        )
    if (
        (normalization.input_sha256, normalization.input_bytes) != raw_identity
        or normalization.input_row_count != materialization.row_count
        or (
            normalization.materialization_manifest_sha256,
            normalization.materialization_manifest_bytes,
        )
        != materialization_identity
    ):
        raise OracleIsolationReviewError(
            "normalization does not bind the exact materialized raw population"
        )
    if (
        task_manifest.status != "PASS"
        or task_manifest.projection_scope != "complete-snapshot"
        or task_manifest.file_patterns
        or task_manifest.safe_fields != TASK_PROVENANCE_SAFE_FIELDS
        or task_manifest.source_name != normalization.task_source_name
        or task_manifest.source_revision.casefold() != normalization.task_source_revision.casefold()
        or task_manifest.output_sha256 != task_identity[0]
        or task_manifest.output_bytes != task_identity[1]
        or resolve_artifact(task_provenance_manifest, task_manifest.output)
        != task_provenance.resolve()
        or (
            normalization.task_provenance_sha256,
            normalization.task_provenance_bytes,
        )
        != task_identity
        or (
            normalization.task_provenance_manifest_sha256,
            normalization.task_provenance_manifest_bytes,
        )
        != task_manifest_identity
    ):
        raise OracleIsolationReviewError(
            "normalization does not bind exact complete task provenance evidence"
        )
    upstream_review = _require_supported_upstream_review(normalization)

    captured_inputs = (
        (raw_input, raw_identity),
        (materialization_manifest, materialization_identity),
        (normalized_input, normalized_identity),
        (normalization_manifest, normalization_identity),
        (task_provenance, task_identity),
        (task_provenance_manifest, task_manifest_identity),
    )

    def verify_inputs() -> None:
        for path, identity in captured_inputs:
            _require_identity(path, identity)

    with _normalized_oracle_index() as index:
        try:
            with verified_staged_file(task_provenance, task_identity) as staged_task_provenance:
                for row in _iter_jsonl_mappings(staged_task_provenance):
                    try:
                        record = TaskProvenanceRecord.model_validate(row)
                    except ValidationError as error:
                        raise OracleIsolationReviewError(
                            "task provenance artifact contains an invalid record"
                        ) from error
                    if (
                        record.source_dataset != normalization.task_source_name
                        or record.source_dataset_revision.casefold()
                        != normalization.task_source_revision.casefold()
                    ):
                        raise OracleIsolationReviewError(
                            "task provenance record does not match the normalized task source"
                        )
                    index.add_task(record)
            if index.task_count != task_manifest.admitted_count:
                raise OracleIsolationReviewError(
                    "task provenance artifact row count does not match its manifest"
                )
            with verified_staged_file(normalized_input, normalized_identity) as staged_normalized:
                for row in _iter_jsonl_mappings(staged_normalized):
                    try:
                        sample = NormalizedSample.model_validate(row)
                    except ValidationError as error:
                        raise OracleIsolationReviewError(
                            "normalized artifact contains an invalid sample"
                        ) from error
                    index.add(sample, normalization)
            index.finalize(normalization.accepted_count)
        except VerifiedStagingError as error:
            raise OracleIsolationReviewError(
                f"unable to stage normalized evidence: {error}"
            ) from error

        scan_result: OracleIsolationScanResult | None = None

        def write_findings(stream: BinaryIO) -> None:
            nonlocal scan_result
            try:
                with verified_staged_file(raw_input, raw_identity) as staged_raw:
                    scan_result = _scan_raw_population(
                        staged_raw,
                        index=index,
                        normalization=normalization,
                        materialization=materialization,
                        stream=stream,
                    )
            except VerifiedStagingError as error:
                raise OracleIsolationReviewError(
                    f"unable to stage raw evidence: {error}"
                ) from error

        findings_result = write_immutable_stream(
            findings_output,
            write_findings,
            before_publish=verify_inputs,
        )
    if scan_result is None:  # pragma: no cover - immutable writer contract
        raise OracleIsolationReviewError("oracle-isolation scan produced no result")

    binding_reasons: set[OracleIsolationFindingReasonCode] = {
        "raw_normalized_binding_mismatch",
        "population_binding_mismatch",
    }
    context_reasons: set[OracleIsolationFindingReasonCode] = {
        "invalid_recorded_model_context",
        "gold_field_in_recorded_model_context",
        "unsupported_reference_patch_location",
    }
    prompt_reasons: set[OracleIsolationFindingReasonCode] = {"reference_patch_in_initial_prompt"}
    reference_patch_coverage_status = (
        VerificationStatus.PASS.value
        if (
            scan_result.reference_patch_row_count == scan_result.raw_row_count
            and scan_result.reference_patch_row_count == scan_result.covered_sample_count
        )
        else VerificationStatus.FAIL.value
    )

    def check_status(reasons: set[OracleIsolationFindingReasonCode]) -> str:
        return (
            VerificationStatus.FAIL.value
            if any(scan_result.reason_counts.get(reason, 0) for reason in reasons)
            else VerificationStatus.PASS.value
        )

    checks = (
        {
            "name": "raw-normalized-population-binding",
            "status": check_status(binding_reasons),
        },
        {
            "name": "recorded-model-context-boundary",
            "status": check_status(context_reasons),
        },
        {
            "name": "recorded-model-input-gold-absence",
            "status": check_status(prompt_reasons),
        },
        {"name": "reference-patch-coverage", "status": reference_patch_coverage_status},
        {"name": "upstream-git-hacking-review", "status": "PASS"},
    )
    overall_status = (
        VerificationStatus.PASS
        if all(check["status"] == VerificationStatus.PASS.value for check in checks)
        else VerificationStatus.FAIL
    )
    attestation = OracleIsolationAttestation.model_validate(
        {
            "schema_version": "nodelm.oracle-isolation-attestation/v2",
            "method_version": "nodelm.oracle-isolation-recorded-context-review/v2",
            "status": overall_status.value,
            "review_scope": "recorded-model-context-and-upstream-curation",
            "upstream_review_id": upstream_review.review_id,
            "source_name": normalization.source_name,
            "source_repository_id": normalization.source_repository_id,
            "source_revision": normalization.source_revision,
            "partition_name": normalization.partition_name,
            "harness": normalization.harness,
            "generating_model": normalization.generating_model,
            "materialization_manifest_artifact": os.path.relpath(
                materialization_manifest, start=output.resolve().parent
            ),
            "materialization_manifest_sha256": materialization_identity[0],
            "materialization_manifest_bytes": materialization_identity[1],
            "raw_artifact": os.path.relpath(raw_input, start=output.resolve().parent),
            "raw_sha256": raw_identity[0],
            "raw_bytes": raw_identity[1],
            "raw_row_count": scan_result.raw_row_count,
            "task_provenance_artifact": os.path.relpath(
                task_provenance, start=output.resolve().parent
            ),
            "task_provenance_sha256": task_identity[0],
            "task_provenance_bytes": task_identity[1],
            "task_provenance_manifest_artifact": os.path.relpath(
                task_provenance_manifest, start=output.resolve().parent
            ),
            "task_provenance_manifest_sha256": task_manifest_identity[0],
            "task_provenance_manifest_bytes": task_manifest_identity[1],
            "normalization_manifest_artifact": os.path.relpath(
                normalization_manifest, start=output.resolve().parent
            ),
            "normalization_manifest_sha256": normalization_identity[0],
            "normalization_manifest_bytes": normalization_identity[1],
            "normalized_artifact": os.path.relpath(normalized_input, start=output.resolve().parent),
            "normalized_sha256": normalized_identity[0],
            "normalized_bytes": normalized_identity[1],
            "expected_sample_count": normalization.accepted_count,
            "covered_sample_count": scan_result.covered_sample_count,
            "reference_patch_row_count": scan_result.reference_patch_row_count,
            "checks": checks,
            "findings_artifact": os.path.relpath(
                findings_result.path, start=output.resolve().parent
            ),
            "findings_sha256": findings_result.digest,
            "findings_bytes": scan_result.findings_bytes,
            "finding_count": scan_result.finding_count,
        }
    )

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_identity(
            findings_result.path,
            (findings_result.digest, scan_result.findings_bytes),
        )

    result = write_immutable_json(
        output,
        attestation.model_dump(mode="json"),
        before_publish=verify_completion_boundary,
    )
    return result, attestation
