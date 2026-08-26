from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nodelm.artifacts import (
    ArtifactWriteResult,
    canonical_json_bytes,
    content_digest,
    file_identity,
    write_immutable_json,
)
from nodelm.datasets.staging import VerifiedStagingError, verified_staged_file
from nodelm.models import NormalizedSample, VerificationStatus
from nodelm.provenance.manifests import (
    ManifestFileIdentity,
    NormalizationCohortManifestV1,
    NormalizationCohortMemberV1,
    NormalizationManifestV2,
)
from nodelm.provenance.pipeline import (
    has_exact_normalization_evidence_lineage,
    normalization_evidence_lineage,
)


class NormalizationCohortError(ValueError):
    """A selected normalization population is not fully content-bound and valid."""


@dataclass(frozen=True)
class _BoundMember:
    manifest_path: Path
    manifest_identity: tuple[str, int]
    manifest: NormalizationManifestV2
    normalized_path: Path
    normalized_identity: tuple[str, int]


_SHARED_FIELDS = (
    "source_name",
    "source_repository_id",
    "source_revision",
    "upstream_source",
    "row_dataset_name",
    "registry_sha256",
    "partition_contract_sha256",
    "partition_contract_bytes",
    "transfer_receipt_sha256",
    "transfer_receipt_bytes",
    "task_provenance_sha256",
    "task_provenance_bytes",
    "task_provenance_manifest_sha256",
    "task_provenance_manifest_bytes",
    "task_transfer_receipt_sha256",
    "task_transfer_receipt_bytes",
    "task_source_name",
    "task_source_revision",
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _contained_path(path: Path, *, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise NormalizationCohortError(
            f"cohort evidence must be contained by its output directory: {path}"
        ) from error
    return resolved


def _read_normalization_manifest(path: Path, *, evidence_root: Path) -> _BoundMember:
    manifest_path = _contained_path(path, root=evidence_root)
    try:
        payload = manifest_path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
        if not isinstance(value, dict):
            raise ValueError("manifest root must be a mapping")
        manifest = NormalizationManifestV2.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, ValidationError) as error:
        raise NormalizationCohortError(
            f"invalid normalization member manifest {manifest_path}: {error}"
        ) from error

    if manifest.status != VerificationStatus.PASS.value:
        raise NormalizationCohortError(
            f"cohort requires a PASS normalization member: {manifest.partition_name}"
        )
    if manifest.uniqueness_scope != "complete-partition":
        raise NormalizationCohortError(
            f"cohort requires complete-partition evidence: {manifest.partition_name}"
        )
    if manifest.accepted_count < 1:
        raise NormalizationCohortError(
            f"cohort member has no accepted samples: {manifest.partition_name}"
        )
    partition_harness, partition_model, partition_upstream = manifest.partition_name.split("/")
    if (
        manifest.harness != partition_harness
        or manifest.generating_model != f"source-label:{partition_model}"
        or manifest.upstream_source != partition_upstream
    ):
        raise NormalizationCohortError(
            f"normalization member labels disagree with its partition: {manifest.partition_name}"
        )

    try:
        normalized_reference = ManifestFileIdentity(
            path=manifest.normalized_artifact,
            sha256=manifest.normalized_sha256,
            bytes=manifest.normalized_bytes,
        )
        unresolved_normalized_path = manifest_path.parent / normalized_reference.path
        normalized_path = _contained_path(unresolved_normalized_path, root=evidence_root)
        metadata = unresolved_normalized_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError("normalized member must be a real regular file")
        normalized_identity = file_identity(normalized_path)
    except (OSError, ValidationError) as error:
        raise NormalizationCohortError(
            f"unable to read normalized member {manifest.partition_name}: {error}"
        ) from error
    if normalized_identity != (manifest.normalized_sha256, manifest.normalized_bytes):
        raise NormalizationCohortError(
            f"normalization manifest does not bind its artifact: {manifest.partition_name}"
        )
    return _BoundMember(
        manifest_path=manifest_path,
        manifest_identity=(content_digest(payload), len(payload)),
        manifest=manifest,
        normalized_path=normalized_path,
        normalized_identity=normalized_identity,
    )


def _contained_identity(
    path: Path,
    identity: tuple[str, int],
    *,
    root: Path,
) -> ManifestFileIdentity:
    relative = _contained_path(path, root=root).relative_to(root.resolve()).as_posix()
    return ManifestFileIdentity(path=relative, sha256=identity[0], bytes=identity[1])


def _require_unchanged(member: _BoundMember) -> None:
    try:
        manifest_identity = file_identity(member.manifest_path)
        normalized_identity = file_identity(member.normalized_path)
    except OSError as error:
        raise NormalizationCohortError(
            f"unable to recheck cohort member {member.manifest.partition_name}: {error}"
        ) from error
    if manifest_identity != member.manifest_identity:
        raise NormalizationCohortError(
            f"normalization manifest changed during cohort construction: "
            f"{member.manifest.partition_name}"
        )
    if normalized_identity != member.normalized_identity:
        raise NormalizationCohortError(
            f"normalized artifact changed during cohort construction: "
            f"{member.manifest.partition_name}"
        )


def build_normalization_cohort(
    member_manifest_paths: Sequence[Path],
    output: Path,
) -> tuple[ArtifactWriteResult, NormalizationCohortManifestV1]:
    """Validate and bind selected complete partitions without concatenating them on disk."""

    if len(member_manifest_paths) < 2:
        raise NormalizationCohortError("a normalization cohort requires at least two members")
    output_root = output.resolve().parent
    members = tuple(
        sorted(
            (
                _read_normalization_manifest(path, evidence_root=output_root)
                for path in member_manifest_paths
            ),
            key=lambda item: item.manifest.partition_name,
        )
    )
    partitions = tuple(member.manifest.partition_name for member in members)
    if len(partitions) != len(set(partitions)):
        raise NormalizationCohortError("normalization cohort partitions must be unique")

    first = members[0].manifest
    for member in members[1:]:
        for field in _SHARED_FIELDS:
            if getattr(member.manifest, field) != getattr(first, field):
                raise NormalizationCohortError(
                    f"normalization cohort members disagree on shared provenance: {field}"
                )

    if any(
        output.resolve() in {member.manifest_path, member.normalized_path} for member in members
    ):
        raise NormalizationCohortError("cohort output must not collide with a member input")

    population_digest = hashlib.sha256()
    population_bytes = 0
    cohort_members: list[NormalizationCohortMemberV1] = []
    sample_count = 0
    with tempfile.TemporaryDirectory(prefix="nodelm-cohort-index-") as index_directory:
        connection = sqlite3.connect(Path(index_directory) / "sample-ids.sqlite3")
        try:
            connection.execute("CREATE TABLE sample_ids (sample_id TEXT PRIMARY KEY) WITHOUT ROWID")
            for member in members:
                manifest = member.manifest
                member_count = 0
                expected_lineage = normalization_evidence_lineage(
                    materialization_manifest_sha256=(manifest.materialization_manifest_sha256),
                    partition_name=manifest.partition_name,
                    upstream_source=manifest.upstream_source,
                    task_source_name=manifest.task_source_name,
                    task_source_revision=manifest.task_source_revision,
                    task_provenance_sha256=manifest.task_provenance_sha256,
                )
                try:
                    with (
                        verified_staged_file(
                            member.normalized_path,
                            member.normalized_identity,
                        ) as staged,
                        staged.open("rb") as source,
                    ):
                        for line_number, raw_row in enumerate(source, start=1):
                            population_digest.update(raw_row)
                            population_bytes += len(raw_row)
                            if not raw_row.strip():
                                raise NormalizationCohortError(
                                    f"blank normalized row in "
                                    f"{manifest.partition_name}:{line_number}"
                                )
                            try:
                                value = json.loads(
                                    raw_row,
                                    object_pairs_hook=_unique_json_object,
                                )
                                if not isinstance(value, dict):
                                    raise ValueError("normalized row must be a JSON object")
                                sample = NormalizedSample.model_validate(value)
                            except (
                                UnicodeError,
                                json.JSONDecodeError,
                                ValueError,
                                ValidationError,
                            ) as error:
                                raise NormalizationCohortError(
                                    f"invalid normalized row in "
                                    f"{manifest.partition_name}:{line_number}: {error}"
                                ) from error
                            if raw_row != canonical_json_bytes(sample.model_dump(mode="json")):
                                raise NormalizationCohortError(
                                    f"normalized row is not canonical JSONL: "
                                    f"{manifest.partition_name}:{line_number}"
                                )
                            try:
                                connection.execute(
                                    "INSERT INTO sample_ids(sample_id) VALUES (?)",
                                    (sample.sample_id,),
                                )
                            except sqlite3.IntegrityError as error:
                                raise NormalizationCohortError(
                                    f"duplicate sample_id across normalization cohort: "
                                    f"{sample.sample_id}"
                                ) from error
                            if (
                                sample.source_dataset != manifest.source_name
                                or sample.source_dataset_revision.casefold()
                                != manifest.source_revision.casefold()
                                or sample.harness != manifest.harness
                                or sample.generating_model != manifest.generating_model
                                or not has_exact_normalization_evidence_lineage(
                                    sample.provenance_lineage,
                                    expected_lineage,
                                )
                            ):
                                raise NormalizationCohortError(
                                    f"normalized row lineage disagrees with member manifest: "
                                    f"{manifest.partition_name}:{line_number}"
                                )
                            member_count += 1
                except VerifiedStagingError as error:
                    raise NormalizationCohortError(
                        f"unable to stage normalized member {manifest.partition_name}: {error}"
                    ) from error
                if member_count != manifest.accepted_count:
                    raise NormalizationCohortError(
                        f"normalized row count does not match member manifest: "
                        f"{manifest.partition_name}"
                    )
                sample_count += member_count
                cohort_members.append(
                    NormalizationCohortMemberV1(
                        partition_name=manifest.partition_name,
                        harness=manifest.harness,
                        generating_model=manifest.generating_model,
                        normalization_manifest=_contained_identity(
                            member.manifest_path,
                            member.manifest_identity,
                            root=output_root,
                        ),
                        normalized_artifact=_contained_identity(
                            member.normalized_path,
                            member.normalized_identity,
                            root=output_root,
                        ),
                        accepted_count=member_count,
                    )
                )
        finally:
            connection.close()

    def verify_inputs() -> None:
        for member in members:
            _require_unchanged(member)

    cohort = NormalizationCohortManifestV1(
        schema_version="nodelm.normalization-cohort-manifest/v1",
        status="PASS",
        cohort_scope="complete-selected-members",
        member_order="partition-name-ascending",
        population_identity="sha256-exact-member-concatenation/v1",
        source_name=first.source_name,
        source_repository_id=first.source_repository_id,
        source_revision=first.source_revision,
        upstream_source=first.upstream_source,
        row_dataset_name=first.row_dataset_name,
        registry_sha256=first.registry_sha256,
        partition_contract_sha256=first.partition_contract_sha256,
        partition_contract_bytes=first.partition_contract_bytes,
        trace_transfer_receipt_sha256=first.transfer_receipt_sha256,
        trace_transfer_receipt_bytes=first.transfer_receipt_bytes,
        task_provenance_sha256=first.task_provenance_sha256,
        task_provenance_bytes=first.task_provenance_bytes,
        task_provenance_manifest_sha256=first.task_provenance_manifest_sha256,
        task_provenance_manifest_bytes=first.task_provenance_manifest_bytes,
        task_transfer_receipt_sha256=first.task_transfer_receipt_sha256,
        task_transfer_receipt_bytes=first.task_transfer_receipt_bytes,
        task_source_name=first.task_source_name,
        task_source_revision=first.task_source_revision,
        member_count=len(cohort_members),
        members=tuple(cohort_members),
        sample_count=sample_count,
        unique_sample_id_count=sample_count,
        population_sha256=population_digest.hexdigest(),
        population_bytes=population_bytes,
        gold_exposure_audit="NOT RUN",
    )
    result = write_immutable_json(
        output,
        cohort.model_dump(mode="json"),
        before_publish=verify_inputs,
    )
    return result, cohort
