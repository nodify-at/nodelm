from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import typer
import yaml
from pydantic import ValidationError

from nodelm.artifacts import (
    canonical_json_bytes,
    content_digest,
    file_identity,
    write_immutable_json,
    write_immutable_stream,
)
from nodelm.datasets.audit import audit_rows, iter_license_rejections
from nodelm.datasets.hub import download_pinned_snapshot, verify_hub_source
from nodelm.datasets.lineage import DatasetSnapshotTransferReceipt
from nodelm.datasets.materialize import discover_snapshot_files, iter_snapshot_rows
from nodelm.datasets.partitions import (
    PartitionContractError,
    TracePartition,
    TracePartitionContract,
)
from nodelm.datasets.pilot import (
    PilotAuthorizationError,
    PilotFilter,
    PilotPolicyConfig,
    build_pilot_subset,
    require_authorized_pilot_manifest,
)
from nodelm.datasets.registry import DatasetRegistry, RegistryError
from nodelm.datasets.seals import SnapshotSealError, require_authorized_snapshot_seal
from nodelm.datasets.snapshot_audit import (
    SnapshotAuditError,
    audit_snapshot,
)
from nodelm.datasets.snapshot_transfer import (
    SnapshotTransferError,
    transfer_snapshot,
)
from nodelm.datasets.staging import (
    VerifiedStagingError,
    verified_staged_files,
    verified_staged_regular_file_tree,
    verify_regular_file_tree,
)
from nodelm.decontamination.contamination import (
    BenchmarkEntry,
    ContaminationSample,
    TaskMetadataLookup,
    decontamination_task_metadata_index,
)
from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.decontamination.split import (
    SplitAuthorizationError,
    read_repository_split_evidence,
    require_authorized_repository_split,
    write_repository_split_manifest,
)
from nodelm.doctor import build_doctor_report
from nodelm.evaluation.fixture import (
    MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
    MODEL_TASK_FIXTURE_IDENTITY,
    FixturePatchReport,
    evaluate_model_patch_fixture,
)
from nodelm.evaluation.registry import CandidateRegistry, CandidateRegistryError
from nodelm.evaluation.resolution_canary import (
    SWE_REBENCH_CONSTANTS_SHA256,
    SWE_REBENCH_EVAL_SCRIPT_SHA256,
    SWE_REBENCH_EVALUATOR_REPOSITORY_ID,
    SWE_REBENCH_EVALUATOR_REVISION,
    SWE_REBENCH_LOG_PARSERS_SHA256,
    PinnedContainerImage,
    PinnedEvaluatorLogParser,
    PodmanImageLocker,
    ResolutionCanaryCase,
    ResolutionCanaryCaseResult,
    ResolutionCanaryError,
    ResolutionCanaryImageLock,
    ResolutionCanaryOracle,
    ResolutionCanaryPrivateCaseEvidence,
    SkopeoChrootImageLocker,
    SWERebenchPodmanSandbox,
    SWERebenchSeccompChrootSandbox,
    SWERebenchTask,
    project_swe_rebench_task,
    resolution_canary_output,
)
from nodelm.evaluation.sandbox import PodmanFixtureSandbox
from nodelm.harness import (
    CommandExecutor,
    CommandPolicy,
    OutcomeCategory,
    parse_node_test_count,
)
from nodelm.harness.config import HarnessConfig
from nodelm.harness.discovery import discover_typescript_workspace
from nodelm.infra.doctor import collect_infrastructure_report
from nodelm.logging import configure_structured_logging
from nodelm.models import NormalizedSample, VerificationStatus, stable_model_id
from nodelm.provenance.cohort import (
    NormalizationCohortError,
    build_normalization_cohort,
)
from nodelm.provenance.gold import (
    GoldExposureAudit,
    GoldExposureAuthorizationError,
    OracleIsolationAttestation,
    OracleIsolationAuthorizationError,
    SanitizedGoldExposureFinding,
    require_authorized_gold_audit,
    require_authorized_oracle_attestation,
)
from nodelm.provenance.manifests import (
    TASK_PROVENANCE_SAFE_FIELDS,
    ManifestFileIdentity,
    NormalizationManifestV2,
    ResolutionCanaryExecutionManifestV1,
    ResolutionCanaryWorksetManifestV1,
    ResolutionLanguage,
    ResolutionPartitionInput,
    ResolutionRecoveryManifestV1,
    SnapshotMaterializationManifestV1,
    SnapshotMaterializationManifestV2,
    TaskProvenanceProjectionManifestV1,
)
from nodelm.provenance.normalize import (
    NormalizationError,
    UnknownResolutionError,
    validate_gold_free_trajectory,
)
from nodelm.provenance.oracle_isolation import (
    OracleIsolationReviewError,
    review_oracle_isolation_artifacts,
)
from nodelm.provenance.pipeline import (
    normalization_evidence_lineage,
    normalize_trace_sample,
    task_metadata_index,
    trace_rollout_key,
)
from nodelm.provenance.resolution import (
    ExactResolutionCandidate,
    ResolutionEvaluationRequest,
    ResolutionRecoveryError,
    build_resolution_recovery,
)
from nodelm.provenance.resolution_canary import (
    materialize_resolution_canary_cases,
    recover_transfer_control_requests,
    select_resolution_canary_sources,
)
from nodelm.provenance.task_provenance import (
    TaskProjectionError,
    canonical_language,
    task_provenance_projection,
)
from nodelm.provenance.trace_projection import (
    TraceProjectionError,
    trace_normalization_projection,
)
from nodelm.training.config import (
    load_training_smoke_config,
    parse_training_smoke_config,
    training_config_identity,
)
from nodelm.training.data import take_training_texts
from nodelm.training.lifecycle import TrainingLifecycleConfig, run_training_lifecycle
from nodelm.training.transformers_backend import (
    TransformersSmokeBackend,
    TransformersSmokeSettings,
)

app = typer.Typer(no_args_is_help=True, help="NodeLM reproducible research tooling")
datasets_app = typer.Typer(no_args_is_help=True, help="Dataset registry and audit commands")
split_app = typer.Typer(no_args_is_help=True, help="Contamination-safe split commands")
harness_app = typer.Typer(no_args_is_help=True, help="Repository harness commands")
infra_app = typer.Typer(no_args_is_help=True, help="Infrastructure verification commands")
models_app = typer.Typer(no_args_is_help=True, help="Model candidate commands")
training_app = typer.Typer(no_args_is_help=True, help="Training preparation commands")
app.add_typer(datasets_app, name="datasets")
app.add_typer(split_app, name="split")
app.add_typer(harness_app, name="harness")
app.add_typer(infra_app, name="infra")
app.add_typer(models_app, name="models")
app.add_typer(training_app, name="training")


@app.callback()
def configure_cli() -> None:
    configure_structured_logging()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise typer.BadParameter(
                    f"invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise typer.BadParameter(f"line {line_number} is not a JSON object")
            yield value


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    return _read_yaml_mapping_with_identity(path)[0]


def _read_yaml_mapping_with_identity(
    path: Path,
) -> tuple[dict[str, Any], tuple[str, int]]:
    try:
        payload = path.read_bytes()
        value = yaml.safe_load(payload)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise typer.BadParameter(f"unable to read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise typer.BadParameter(f"configuration root must be a mapping: {path}")
    return (
        {str(key): item for key, item in value.items()},
        (content_digest(payload), len(payload)),
    )


def _read_json_mapping_with_identity(path: Path) -> tuple[dict[str, Any], tuple[str, int]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"unable to read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise typer.BadParameter(f"JSON manifest root must be a mapping: {path}")
    identity = (content_digest(payload), len(payload))
    return ({str(key): item for key, item in value.items()}, identity)


def _load_registry_with_identity(path: Path) -> tuple[DatasetRegistry, tuple[str, int]]:
    try:
        payload = path.read_bytes()
        registry = DatasetRegistry.from_bytes(payload)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"unable to read dataset registry {path}: {error}") from error
    identity = (content_digest(payload), len(payload))
    _require_unchanged_file(path, identity)
    return registry, identity


def _read_repository_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = _read_yaml_mapping(path)
    if set(value) != {"schema_version", "aliases"}:
        raise typer.BadParameter(
            "repository alias manifest requires only schema_version and aliases"
        )
    if value["schema_version"] != "nodelm.repository-aliases/v1":
        raise typer.BadParameter("unsupported repository alias manifest schema_version")
    aliases = value["aliases"]
    if not isinstance(aliases, dict) or any(
        not isinstance(alias, str) or not isinstance(target, str)
        for alias, target in aliases.items()
    ):
        raise typer.BadParameter("repository alias manifest aliases must map strings to strings")
    return aliases


def _require_unchanged_file(path: Path, expected: tuple[str, int]) -> None:
    if file_identity(path) != expected:
        raise typer.BadParameter(f"input changed while it was being processed: {path}")


def _published_artifact_identity(
    path: Path,
    expected_digest: str,
) -> tuple[str, int]:
    identity = file_identity(path)
    if identity[0] != expected_digest:
        raise typer.BadParameter(f"published artifact changed unexpectedly: {path}")
    return identity


def _canonical_jsonl_identity(rows: Iterable[Any]) -> tuple[tuple[str, int], int]:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    for row in rows:
        payload = canonical_json_bytes(row)
        digest.update(payload)
        byte_count += len(payload)
        row_count += 1
    return (digest.hexdigest(), byte_count), row_count


def _receipt_bound_snapshot_files(
    snapshot: Path,
    *,
    expected_by_path: dict[str, tuple[str, int]],
    patterns: tuple[str, ...] = (),
) -> tuple[tuple[Path, tuple[str, int]], ...]:
    files = discover_snapshot_files(snapshot, patterns=patterns)
    resolved_snapshot = snapshot.resolve()
    root = resolved_snapshot if resolved_snapshot.is_dir() else resolved_snapshot.parent
    actual_by_path = {
        path.relative_to(root).as_posix(): (path, file_identity(path)) for path in files
    }
    actual_identities = {
        relative_path: identity for relative_path, (_, identity) in actual_by_path.items()
    }
    if actual_identities != expected_by_path:
        raise typer.BadParameter("snapshot files do not match the authorized transfer receipt")
    return tuple(actual_by_path[path] for path in sorted(expected_by_path))


def _validate_pilot_manifest(
    path: Path,
    *,
    samples_sha256: str,
    samples_bytes: int,
    pilot_manifest_sha256: str,
    artifact_base: Path,
    expected_samples_path: Path,
    required_samples: int,
) -> None:
    value = _read_yaml_mapping(path)
    if value.get("schema_version") != "nodelm.pilot-subset/v1":
        raise typer.BadParameter("unsupported pilot manifest schema_version")
    if value.get("status") not in {
        VerificationStatus.PASS.value,
        VerificationStatus.UNVERIFIED.value,
    }:
        raise typer.BadParameter("training lifecycle requires a reviewed pilot manifest")
    try:
        require_authorized_pilot_manifest(
            samples_sha256=samples_sha256,
            pilot_manifest_sha256=pilot_manifest_sha256,
        )
    except PilotAuthorizationError as error:
        raise typer.BadParameter(str(error)) from error
    if value.get("samples_sha256") != samples_sha256:
        raise typer.BadParameter("pilot manifest sample digest does not match --samples")
    if value.get("samples_bytes") != samples_bytes:
        raise typer.BadParameter("pilot manifest sample bytes do not match --samples")
    if value.get("gold_exposure_audit") != VerificationStatus.PASS.value:
        raise typer.BadParameter("pilot manifest lacks a PASS gold-exposure gate")
    for field in ("normalization_manifest_sha256", "gold_exposure_audit_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise typer.BadParameter(f"pilot manifest is missing {field}")
    accepted_count = value.get("accepted_count")
    if not isinstance(accepted_count, int) or accepted_count < required_samples:
        raise typer.BadParameter("pilot manifest does not contain the required training batch")
    artifact = value.get("samples_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise typer.BadParameter("pilot manifest is missing its samples_artifact")
    artifact_path = Path(artifact)
    if not artifact_path.is_absolute():
        artifact_path = artifact_base.resolve() / artifact_path
    if artifact_path.resolve() != expected_samples_path.resolve():
        raise typer.BadParameter("pilot manifest samples_artifact does not match --samples")


@app.command("doctor")
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    report = build_doctor_report(finder=shutil.which)
    payload = report.model_dump(mode="json")
    typer.echo(_dump(payload) if json_output else f"{report.status.value}: local environment")
    if report.status is VerificationStatus.FAIL:
        raise typer.Exit(code=1)


@datasets_app.command("validate")
def datasets_validate(
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    registry = DatasetRegistry.load(config)
    statuses = {source.status for source in registry.sources}
    if VerificationStatus.FAIL in statuses:
        status = VerificationStatus.FAIL
    elif statuses == {VerificationStatus.PASS}:
        status = VerificationStatus.PASS
    else:
        status = VerificationStatus.UNVERIFIED
    payload = {
        "schema_version": registry.schema_version,
        "source_count": len(registry.sources),
        "status": status.value,
        "sources": [
            source.model_dump(
                mode="json",
                include={"name", "repository_id", "revision", "status"},
            )
            for source in registry.sources
        ],
    }
    typer.echo(
        _dump(payload) if json_output else f"{status.value}: {len(registry.sources)} sources"
    )
    if status is VerificationStatus.FAIL:
        raise typer.Exit(code=1)


@datasets_app.command("audit")
def datasets_audit(
    source_name: str = typer.Option(..., "--source"),
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    allow_partial_snapshot: bool = typer.Option(False, "--allow-partial-snapshot"),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    registry = DatasetRegistry.load(config)
    input_identity = file_identity(input_path)
    input_sha256, input_bytes = input_identity
    try:
        report = audit_rows(
            registry.by_name(source_name),
            _read_jsonl(input_path),
            input_sha256=input_sha256,
            input_bytes=input_bytes,
            expect_complete_snapshot=not allow_partial_snapshot,
        )
    except NormalizationError as error:
        raise typer.BadParameter(f"invalid audited row: {error}") from error
    _require_unchanged_file(input_path, input_identity)
    ledger_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    ledger_row_count = 0

    def write_rejection_ledger(stream: BinaryIO) -> None:
        nonlocal ledger_row_count
        for rejection in iter_license_rejections(_read_jsonl(input_path)):
            stream.write(canonical_json_bytes(rejection))
            ledger_row_count += 1

    ledger_result = write_immutable_stream(
        ledger_path,
        write_rejection_ledger,
        before_publish=lambda: _require_unchanged_file(input_path, input_identity),
    )
    _require_unchanged_file(input_path, input_identity)
    ledger_artifact = os.path.relpath(
        ledger_result.path,
        start=output.resolve().parent,
    )
    report = type(report).model_validate(
        {
            **report.model_dump(mode="json"),
            "rejection_ledger_artifact": ledger_artifact,
            "rejection_ledger_sha256": ledger_result.digest,
            "rejection_ledger_rows": ledger_row_count,
        }
    )
    result = write_immutable_json(output, report.model_dump(mode="json"))
    typer.echo(f"wrote {result.path} sha256={result.digest}")
    if report.status is VerificationStatus.FAIL:
        raise typer.Exit(code=1)


@datasets_app.command("audit-snapshot")
def datasets_audit_snapshot(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    receipt: Path = typer.Option(..., "--receipt", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    lineage_output: Path | None = typer.Option(None, "--lineage-output", dir_okay=False),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    staging_root: Path | None = typer.Option(None, "--staging-root", file_okay=False),
    config: Path = typer.Option(
        Path("configs/datasets/registry.yaml"), exists=True, dir_okay=False
    ),
) -> None:
    """Audit every supported file in a local pinned snapshot without network access."""

    try:
        result = audit_snapshot(
            source_name=source_name,
            snapshot=snapshot,
            receipt_path=receipt,
            output=output,
            lineage_output=lineage_output,
            rejections_output=rejections_output,
            staging_root=staging_root,
            config=config,
        )
    except SnapshotAuditError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"wrote {result.audit_result.path} rows={result.report.row_count} "
        f"sha256={result.audit_result.digest}; lineage={result.lineage_result.path}"
    )
    if result.report.status is VerificationStatus.FAIL:
        raise typer.Exit(code=1)


@datasets_app.command("verify")
def datasets_verify(
    source_name: str | None = typer.Option(None, "--source"),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    registry = DatasetRegistry.load(config)
    sources = (registry.by_name(source_name),) if source_name is not None else registry.sources
    results = [verify_hub_source(source) for source in sources]
    status = (
        VerificationStatus.PASS
        if all(result.status is VerificationStatus.PASS for result in results)
        else VerificationStatus.FAIL
    )
    payload = {
        "status": status.value,
        "results": [result.model_dump(mode="json") for result in results],
    }
    typer.echo(_dump(payload) if json_output else f"{status.value}: Hub metadata")
    if status is VerificationStatus.FAIL:
        raise typer.Exit(code=1)


@datasets_app.command("download")
def datasets_download(
    source_name: str = typer.Option(..., "--source"),
    destination: Path = typer.Option(..., "--destination", file_okay=False),
    allow_pattern: list[str] | None = typer.Option(None, "--allow-pattern"),
    receipt_output: Path | None = typer.Option(None, "--receipt-output", dir_okay=False),
    confirm_large_download: bool = typer.Option(False, "--confirm-large-download"),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    if not confirm_large_download:
        raise typer.BadParameter(
            "dataset snapshots may be very large; pass --confirm-large-download"
        )
    try:
        result = transfer_snapshot(
            source_name=source_name,
            destination=destination,
            allow_patterns=tuple(allow_pattern or ()),
            receipt_output=receipt_output,
            config=config,
            downloader=download_pinned_snapshot,
        )
    except SnapshotTransferError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"downloaded pinned snapshot to {result.snapshot_path}; "
        f"receipt={result.receipt_result.path}"
    )


@datasets_app.command("materialize")
def datasets_materialize(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    file_pattern: list[str] | None = typer.Option(None, "--file-pattern"),
    partition_contract: Path | None = typer.Option(
        None, "--partition-contract", exists=True, dir_okay=False
    ),
    transfer_receipt: Path | None = typer.Option(
        None, "--transfer-receipt", exists=True, dir_okay=False
    ),
    partition_name: str | None = typer.Option(None, "--partition"),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
    manifest_output: Path | None = typer.Option(None, "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    registry, config_identity = _load_registry_with_identity(config)
    source = registry.by_name(source_name)
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("materialization requires a registry-verified pinned source")
    partition_values = (partition_contract, transfer_receipt, partition_name)
    if any(value is not None for value in partition_values) and not all(
        value is not None for value in partition_values
    ):
        raise typer.BadParameter(
            "--partition-contract, --transfer-receipt, and --partition must be supplied together"
        )
    if file_pattern and partition_name is not None:
        raise typer.BadParameter("--file-pattern cannot be combined with --partition")

    contract_identity: tuple[str, int] | None = None
    receipt_identity: tuple[str, int] | None = None
    selected_partition = None
    receipt = None
    if (
        partition_contract is not None
        and transfer_receipt is not None
        and partition_name is not None
    ):
        try:
            contract_payload = partition_contract.read_bytes()
            contract_identity = (content_digest(contract_payload), len(contract_payload))
            contract = TracePartitionContract.from_bytes(contract_payload)
            contract.require_source(source.name, source.revision)
            contract.require_authorized_digest(contract_identity[0])
            if contract.source_repository_id != source.repository_id:
                raise PartitionContractError(
                    "partition contract repository does not match the dataset registry"
                )
            if contract.sealed_registry_sha256 != config_identity[0]:
                raise PartitionContractError(
                    "partition contract registry digest does not match --config"
                )
            receipt_payload = transfer_receipt.read_bytes()
            receipt_identity = (content_digest(receipt_payload), len(receipt_payload))
            receipt = contract.bind_transfer_receipt(receipt_payload)
            require_authorized_snapshot_seal(
                source_name=source.name,
                source_revision=source.revision,
                transfer_receipt_sha256=receipt_identity[0],
                snapshot_sha256=receipt.snapshot.snapshot_sha256,
                snapshot_file_count=len(receipt.snapshot.files),
            )
            selected_partition = contract.by_name(partition_name)
            patterns = selected_partition.file_patterns
        except (OSError, PartitionContractError, SnapshotSealError) as error:
            raise typer.BadParameter(f"invalid partition evidence: {error}") from error
    else:
        patterns = tuple(file_pattern or ())
    try:
        files = discover_snapshot_files(snapshot, patterns=patterns)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    identities = tuple((path, file_identity(path)) for path in files)
    snapshot_root = snapshot.resolve() if snapshot.is_dir() else snapshot.resolve().parent
    if receipt is not None:
        expected_by_path = {
            identity.path: (identity.sha256, identity.bytes)
            for identity in receipt.snapshot.files
            if any(PurePosixPath(identity.path).match(pattern) for pattern in patterns)
        }
        actual_by_path = {
            path.relative_to(snapshot_root).as_posix(): identity for path, identity in identities
        }
        if actual_by_path != expected_by_path:
            raise typer.BadParameter(
                "selected partition files do not match the sealed transfer receipt"
            )
    row_count = 0

    def verify_inputs() -> None:
        try:
            current_files = discover_snapshot_files(snapshot, patterns=patterns)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if current_files != files:
            raise typer.BadParameter("snapshot file set changed while it was being processed")
        for path, identity in identities:
            _require_unchanged_file(path, identity)
        _require_unchanged_file(config, config_identity)
        if partition_contract is not None and contract_identity is not None:
            _require_unchanged_file(partition_contract, contract_identity)
        if transfer_receipt is not None and receipt_identity is not None:
            _require_unchanged_file(transfer_receipt, receipt_identity)

    try:
        with verified_staged_files(identities) as staged_files:

            def write_rows(stream: BinaryIO) -> None:
                nonlocal row_count
                for row in iter_snapshot_rows(staged_files):
                    if max_rows is not None and row_count >= max_rows:
                        break
                    stream.write(canonical_json_bytes(row))
                    row_count += 1

            result = write_immutable_stream(output, write_rows, before_publish=verify_inputs)
    except (ValueError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"snapshot materialization failed: {error}") from error
    output_identity = _published_artifact_identity(result.path, result.digest)
    manifest_path = manifest_output or output.with_name(f"{output.stem}.manifest.json")
    manifest_payload = {
        "schema_version": (
            "nodelm.snapshot-materialization/v2"
            if selected_partition is not None
            else "nodelm.snapshot-materialization/v1"
        ),
        "status": (VerificationStatus.PASS.value if row_count else VerificationStatus.FAIL.value),
        "source_name": source.name,
        "source_repository_id": source.repository_id,
        "source_revision": source.revision,
        "registry_sha256": config_identity[0],
        "file_patterns": patterns,
        "row_count": row_count,
        "max_rows": max_rows,
        "files": [
            {
                "path": path.relative_to(snapshot_root).as_posix(),
                "sha256": identity[0],
                "bytes": identity[1],
            }
            for path, identity in identities
        ],
        "output": os.path.relpath(result.path, start=manifest_path.resolve().parent),
        "output_sha256": result.digest,
        "output_bytes": output_identity[1],
    }
    manifest: SnapshotMaterializationManifestV1 | SnapshotMaterializationManifestV2
    if (
        selected_partition is not None
        and contract_identity is not None
        and receipt_identity is not None
    ):
        manifest_payload.update(
            {
                "materialization_scope": (
                    "canary" if max_rows is not None else "complete-partition"
                ),
                "partition_contract_sha256": contract_identity[0],
                "partition_contract_bytes": contract_identity[1],
                "transfer_receipt_sha256": receipt_identity[0],
                "transfer_receipt_bytes": receipt_identity[1],
                "partition_name": selected_partition.name,
                "harness": selected_partition.harness,
                "generating_model": selected_partition.generating_model,
                "upstream_source": selected_partition.upstream_source,
                "row_dataset_name": selected_partition.row_dataset_name,
                "task_source_name": selected_partition.task_source_name,
                "task_source_revision": selected_partition.task_source_revision,
                "normalization_status": selected_partition.normalization_status.value,
            }
        )
        manifest = SnapshotMaterializationManifestV2.model_validate(manifest_payload)
    else:
        manifest = SnapshotMaterializationManifestV1.model_validate(manifest_payload)

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_unchanged_file(result.path, output_identity)

    manifest_result = write_immutable_json(
        manifest_path,
        manifest.model_dump(mode="json"),
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        f"wrote {result.path} rows={row_count} sha256={result.digest}; "
        f"manifest={manifest_result.path}"
    )
    if row_count == 0:
        raise typer.Exit(code=1)


@datasets_app.command("build-resolution-recovery")
def datasets_build_resolution_recovery(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    partition_contract: Path = typer.Option(
        ..., "--partition-contract", exists=True, dir_okay=False
    ),
    transfer_receipt: Path = typer.Option(..., "--transfer-receipt", exists=True, dir_okay=False),
    labeled_partition: list[str] | None = typer.Option(None, "--labeled-partition"),
    target_partition: list[str] | None = typer.Option(None, "--target-partition"),
    language: list[str] | None = typer.Option(None, "--language"),
    candidates_output: Path = typer.Option(..., "--candidates-output", dir_okay=False),
    queue_output: Path = typer.Option(..., "--queue-output", dir_okay=False),
    manifest_output: Path = typer.Option(..., "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    """Build exact-label candidates and a blocked evaluator queue from sealed leaves."""

    labeled_names = tuple(labeled_partition or ())
    target_names = tuple(target_partition or ())
    if not labeled_names or not target_names:
        raise typer.BadParameter(
            "at least one --labeled-partition and --target-partition are required"
        )
    if len(labeled_names) != len(set(labeled_names)):
        raise typer.BadParameter("--labeled-partition values must be unique")
    if len(target_names) != len(set(target_names)):
        raise typer.BadParameter("--target-partition values must be unique")
    if set(labeled_names) & set(target_names):
        raise typer.BadParameter("labeled and target partitions must be disjoint")
    output_paths = {
        candidates_output.resolve(),
        queue_output.resolve(),
        manifest_output.resolve(),
    }
    if len(output_paths) != 3:
        raise typer.BadParameter("candidate, queue, and manifest outputs must be distinct")

    language_values = tuple(language or ("TypeScript", "JavaScript"))
    try:
        canonical_languages = cast(
            tuple[ResolutionLanguage, ...],
            tuple(sorted({canonical_language(item) for item in language_values})),
        )
    except TaskProjectionError as error:
        raise typer.BadParameter(f"invalid resolution language: {error}") from error
    if not canonical_languages or any(
        item not in {"TypeScript", "JavaScript"} for item in canonical_languages
    ):
        raise typer.BadParameter("resolution recovery only supports TypeScript and JavaScript")

    registry, config_identity = _load_registry_with_identity(config)
    try:
        source = registry.by_name(source_name)
    except RegistryError as error:
        raise typer.BadParameter(str(error)) from error
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter(
            "resolution recovery requires a registry-verified pinned trace source"
        )
    trace_revision = source.revision

    try:
        contract_payload = partition_contract.read_bytes()
        contract_identity = (content_digest(contract_payload), len(contract_payload))
        contract = TracePartitionContract.from_bytes(contract_payload)
        contract.require_source(source.name, trace_revision)
        contract.require_authorized_digest(contract_identity[0])
        if contract.source_repository_id != source.repository_id:
            raise PartitionContractError(
                "partition contract repository does not match the dataset registry"
            )
        if contract.sealed_registry_sha256 != config_identity[0]:
            raise PartitionContractError(
                "partition contract registry digest does not match --config"
            )
        receipt_payload = transfer_receipt.read_bytes()
        receipt_identity = (content_digest(receipt_payload), len(receipt_payload))
        receipt = contract.bind_transfer_receipt(receipt_payload)
        if (
            receipt.source != source
            or receipt.registry_sha256 != config_identity[0]
            or receipt.registry_bytes != config_identity[1]
        ):
            raise PartitionContractError(
                "transfer receipt does not match the pinned registry source"
            )
        require_authorized_snapshot_seal(
            source_name=source.name,
            source_revision=trace_revision,
            transfer_receipt_sha256=receipt_identity[0],
            snapshot_sha256=receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(receipt.snapshot.files),
        )
        labeled_partitions = tuple(
            sorted(
                (contract.by_name(name) for name in labeled_names),
                key=lambda partition: partition.name,
            )
        )
        target_partitions = tuple(
            sorted(
                (contract.by_name(name) for name in target_names),
                key=lambda partition: partition.name,
            )
        )
    except (OSError, PartitionContractError, SnapshotSealError) as error:
        raise typer.BadParameter(f"invalid resolution partition evidence: {error}") from error

    selected_partitions = (*labeled_partitions, *target_partitions)
    blocked_partitions = tuple(
        partition.name
        for partition in selected_partitions
        if partition.normalization_status is not VerificationStatus.PASS
    )
    if blocked_partitions:
        raise typer.BadParameter(
            "resolution recovery requires PASS partitions: " + ", ".join(sorted(blocked_partitions))
        )
    task_bindings = {
        (partition.task_source_name, partition.task_source_revision)
        for partition in selected_partitions
    }
    if len(task_bindings) != 1:
        raise typer.BadParameter("all selected partitions must bind the same pinned task source")
    task_source_name, task_source_revision = next(iter(task_bindings))
    if task_source_name is None or task_source_revision is None:  # pragma: no cover - PASS model
        raise typer.BadParameter("selected PASS partitions lack pinned task source evidence")
    try:
        task_source = registry.by_name(task_source_name)
    except RegistryError as error:
        raise typer.BadParameter(str(error)) from error
    if (
        task_source.status is not VerificationStatus.PASS
        or task_source.revision is None
        or task_source.revision.casefold() != task_source_revision.casefold()
    ):
        raise typer.BadParameter(
            "selected partitions' task source is not registry-verified at its revision"
        )
    canonical_task_revision = task_source.revision

    receipt_files_by_partition = {
        partition.name: tuple(
            identity
            for identity in receipt.snapshot.files
            if any(
                PurePosixPath(identity.path).match(pattern) for pattern in partition.file_patterns
            )
        )
        for partition in selected_partitions
    }
    if any(not identities for identities in receipt_files_by_partition.values()):
        raise typer.BadParameter("a selected partition has no receipt-bound snapshot files")
    selected_receipt_files = tuple(
        sorted(
            (
                identity
                for partition in selected_partitions
                for identity in receipt_files_by_partition[partition.name]
            ),
            key=lambda identity: identity.path,
        )
    )
    selected_paths = tuple(identity.path for identity in selected_receipt_files)
    if len(selected_paths) != len(set(selected_paths)):
        raise typer.BadParameter("selected partition files overlap in the transfer receipt")
    expected_by_path = {
        identity.path: (identity.sha256, identity.bytes) for identity in selected_receipt_files
    }
    selected_patterns = tuple(
        sorted(
            {pattern for partition in selected_partitions for pattern in partition.file_patterns}
        )
    )
    try:
        snapshot_inputs = _receipt_bound_snapshot_files(
            snapshot,
            expected_by_path=expected_by_path,
            patterns=selected_patterns,
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid resolution snapshot: {error}") from error
    original_snapshot_paths = tuple(path for path, _ in snapshot_inputs)
    resolved_snapshot = snapshot.resolve()
    snapshot_root = resolved_snapshot if resolved_snapshot.is_dir() else resolved_snapshot.parent
    if (
        tuple(path.relative_to(snapshot_root).as_posix() for path in original_snapshot_paths)
        != selected_paths
    ):
        raise typer.BadParameter("resolution snapshot input ordering is inconsistent")

    def verify_inputs() -> None:
        try:
            current_files = discover_snapshot_files(snapshot, patterns=selected_patterns)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if current_files != original_snapshot_paths:
            raise typer.BadParameter(
                "resolution snapshot file set changed while it was being processed"
            )
        for path, identity in snapshot_inputs:
            _require_unchanged_file(path, identity)
        _require_unchanged_file(config, config_identity)
        _require_unchanged_file(partition_contract, contract_identity)
        _require_unchanged_file(transfer_receipt, receipt_identity)

    projected_columns = (
        "instance_id",
        "trajectory_id",
        "resolved",
        "language",
        "hf_dataset_name",
        "metadata.model_patch.patch",
    )
    manifest_parent = manifest_output.resolve().parent
    try:
        with verified_staged_files(snapshot_inputs) as staged_files:
            staged_by_path = dict(zip(selected_paths, staged_files, strict=True))

            def partitioned_rows(
                partitions: tuple[TracePartition, ...],
            ) -> Iterator[tuple[str, dict[str, Any]]]:
                for partition in partitions:
                    partition_paths = tuple(
                        staged_by_path[identity.path]
                        for identity in receipt_files_by_partition[partition.name]
                    )
                    for row in iter_snapshot_rows(
                        partition_paths,
                        columns=projected_columns,
                    ):
                        observed_task_family = row.get("hf_dataset_name")
                        if (
                            not isinstance(observed_task_family, str)
                            or observed_task_family.strip() != partition.row_dataset_name
                        ):
                            raise ResolutionRecoveryError(
                                "trace hf_dataset_name does not match the bound partition "
                                "task family"
                            )
                        yield partition.name, row

            with build_resolution_recovery(
                partitioned_rows(labeled_partitions),
                partitioned_rows(target_partitions),
                trace_source_revision=trace_revision,
                task_source_revision=canonical_task_revision,
                languages=canonical_languages,
            ) as recovery:
                if recovery.conflict_count:
                    raise typer.BadParameter(
                        "resolution label conflicts prevent artifact publication: "
                        f"count={recovery.conflict_count}"
                    )
                labeled_row_counts = recovery.labeled_row_counts_by_partition
                target_row_counts = recovery.target_row_counts_by_partition
                accounted_target_rows = (
                    recovery.target_ineligible_count
                    + recovery.target_already_known_count
                    + recovery.candidate_count
                    + recovery.queued_target_count
                )
                if accounted_target_rows != recovery.target_row_count:
                    raise typer.BadParameter(
                        "resolution target accounting is incomplete before publication: "
                        f"accounted={accounted_target_rows} "
                        f"target={recovery.target_row_count}"
                    )
                if (
                    sum(labeled_row_counts.values()) != recovery.labeled_row_count
                    or sum(target_row_counts.values()) != recovery.target_row_count
                ):
                    raise typer.BadParameter(
                        "resolution per-partition accounting is incomplete before publication"
                    )
                if (
                    recovery.candidate_resolved_count + recovery.candidate_unresolved_count
                    != recovery.candidate_count
                    or recovery.candidate_unique_count > recovery.candidate_count
                    or (recovery.candidate_unique_count == 0) != (recovery.candidate_count == 0)
                    or recovery.evaluation_request_count > recovery.queued_target_count
                    or (recovery.evaluation_request_count == 0)
                    != (recovery.queued_target_count == 0)
                ):
                    raise typer.BadParameter(
                        "resolution output accounting is inconsistent before publication"
                    )

                def write_candidates(stream: BinaryIO) -> None:
                    for candidate in recovery.iter_candidates():
                        stream.write(canonical_json_bytes(candidate.model_dump(mode="json")))

                candidate_result = write_immutable_stream(
                    candidates_output,
                    write_candidates,
                    before_publish=verify_inputs,
                )
                candidate_identity = _published_artifact_identity(
                    candidate_result.path,
                    candidate_result.digest,
                )

                def write_queue(stream: BinaryIO) -> None:
                    for request in recovery.iter_evaluation_requests():
                        stream.write(canonical_json_bytes(request.model_dump(mode="json")))

                def verify_queue_boundary() -> None:
                    verify_inputs()
                    _require_unchanged_file(candidate_result.path, candidate_identity)

                queue_result = write_immutable_stream(
                    queue_output,
                    write_queue,
                    before_publish=verify_queue_boundary,
                )
                queue_identity = _published_artifact_identity(
                    queue_result.path,
                    queue_result.digest,
                )

                def partition_input(
                    partition: TracePartition,
                    row_count: int,
                ) -> ResolutionPartitionInput:
                    return ResolutionPartitionInput(
                        partition_name=partition.name,
                        row_count=row_count,
                        files=tuple(
                            ManifestFileIdentity(
                                path=identity.path,
                                sha256=identity.sha256,
                                bytes=identity.bytes,
                            )
                            for identity in receipt_files_by_partition[partition.name]
                        ),
                    )

                manifest = ResolutionRecoveryManifestV1(
                    schema_version="nodelm.resolution-recovery/v1",
                    derivation_status="PASS",
                    admission_status="BLOCKED",
                    admission_blocker="harness_canary_pending",
                    source_name=source.name,
                    source_repository_id=source.repository_id,
                    source_revision=trace_revision,
                    task_source_name=task_source_name,
                    task_source_revision=canonical_task_revision,
                    partition_contract_sha256=contract_identity[0],
                    partition_contract_bytes=contract_identity[1],
                    transfer_receipt_sha256=receipt_identity[0],
                    transfer_receipt_bytes=receipt_identity[1],
                    labeled_partitions=tuple(
                        partition_input(
                            partition,
                            labeled_row_counts.get(partition.name, 0),
                        )
                        for partition in labeled_partitions
                    ),
                    target_partitions=tuple(
                        partition_input(
                            partition,
                            target_row_counts.get(partition.name, 0),
                        )
                        for partition in target_partitions
                    ),
                    language_filter=canonical_languages,
                    candidate_artifact=os.path.relpath(
                        candidate_result.path,
                        start=manifest_parent,
                    ),
                    candidate_sha256=candidate_result.digest,
                    candidate_bytes=candidate_identity[1],
                    queue_artifact=os.path.relpath(
                        queue_result.path,
                        start=manifest_parent,
                    ),
                    queue_sha256=queue_result.digest,
                    queue_bytes=queue_identity[1],
                    target_row_count=recovery.target_row_count,
                    ineligible_row_count=recovery.target_ineligible_count,
                    already_known_row_count=recovery.target_already_known_count,
                    candidate_row_count=recovery.candidate_row_count,
                    candidate_unique_count=recovery.candidate_unique_count,
                    candidate_resolved_count=recovery.candidate_resolved_count,
                    candidate_unresolved_count=recovery.candidate_unresolved_count,
                    queued_fanout_row_count=recovery.queued_target_count,
                    queue_unique_count=recovery.evaluation_request_count,
                    conflict_count=recovery.conflict_count,
                )

                def verify_completion_boundary() -> None:
                    verify_queue_boundary()
                    _require_unchanged_file(queue_result.path, queue_identity)

                manifest_result = write_immutable_json(
                    manifest_output,
                    manifest.model_dump(mode="json"),
                    before_publish=verify_completion_boundary,
                )
    except (ResolutionRecoveryError, VerifiedStagingError, ValidationError, ValueError) as error:
        raise typer.BadParameter(f"resolution recovery failed: {error}") from error

    typer.echo(
        f"wrote candidates={candidate_result.path} rows={manifest.candidate_row_count}; "
        f"queue={queue_result.path} unique={manifest.queue_unique_count}; "
        f"manifest={manifest_result.path} admission={manifest.admission_status}"
    )


@datasets_app.command("build-resolution-canary-workset")
def datasets_build_resolution_canary_workset(
    recovery_manifest: Path = typer.Option(..., "--recovery-manifest", exists=True, dir_okay=False),
    candidates: Path = typer.Option(..., "--candidates", exists=True, dir_okay=False),
    queue: Path = typer.Option(..., "--queue", exists=True, dir_okay=False),
    trace_snapshot: Path = typer.Option(..., "--trace-snapshot", exists=True),
    trace_transfer_receipt: Path = typer.Option(
        ..., "--trace-transfer-receipt", exists=True, dir_okay=False
    ),
    partition_contract: Path = typer.Option(
        ..., "--partition-contract", exists=True, dir_okay=False
    ),
    task_snapshot: Path = typer.Option(..., "--task-snapshot", exists=True),
    task_transfer_receipt: Path = typer.Option(
        ..., "--task-transfer-receipt", exists=True, dir_okay=False
    ),
    workset_output: Path = typer.Option(..., "--workset-output", dir_okay=False),
    manifest_output: Path = typer.Option(..., "--manifest-output", dir_okay=False),
    minimum_per_kind: int = typer.Option(6, "--minimum-per-kind", min=1),
    maximum_per_kind: int = typer.Option(12, "--maximum-per-kind", min=1),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    """Materialize a small private, provenance-bound real-repository canary workset."""

    if minimum_per_kind > maximum_per_kind:
        raise typer.BadParameter("--minimum-per-kind must not exceed --maximum-per-kind")
    if workset_output.resolve() == manifest_output.resolve():
        raise typer.BadParameter("workset and manifest outputs must be distinct")

    recovery_identity = file_identity(recovery_manifest)
    candidate_identity = file_identity(candidates)
    queue_identity = file_identity(queue)
    contract_identity = file_identity(partition_contract)
    trace_receipt_identity = file_identity(trace_transfer_receipt)
    task_receipt_identity = file_identity(task_transfer_receipt)
    try:
        recovery = ResolutionRecoveryManifestV1.model_validate_json(recovery_manifest.read_bytes())
        if candidate_identity != (recovery.candidate_sha256, recovery.candidate_bytes):
            raise ResolutionCanaryError("candidate artifact does not match recovery manifest")
        if queue_identity != (recovery.queue_sha256, recovery.queue_bytes):
            raise ResolutionCanaryError("queue artifact does not match recovery manifest")
        if contract_identity != (
            recovery.partition_contract_sha256,
            recovery.partition_contract_bytes,
        ):
            raise ResolutionCanaryError("partition contract does not match recovery manifest")
        if trace_receipt_identity != (
            recovery.transfer_receipt_sha256,
            recovery.transfer_receipt_bytes,
        ):
            raise ResolutionCanaryError("trace receipt does not match recovery manifest")

        candidate_count = 0

        def candidate_rows() -> Iterator[ExactResolutionCandidate]:
            nonlocal candidate_count
            for row in _read_jsonl(candidates):
                candidate_count += 1
                yield ExactResolutionCandidate.model_validate(row)

        request_count = 0

        def request_rows() -> Iterator[ResolutionEvaluationRequest]:
            nonlocal request_count
            for row in _read_jsonl(queue):
                request_count += 1
                yield ResolutionEvaluationRequest.model_validate(row)

        selection = select_resolution_canary_sources(
            candidate_rows(),
            request_rows(),
            minimum_per_kind=minimum_per_kind,
            maximum_per_kind=maximum_per_kind,
        )
        if candidate_count != recovery.candidate_row_count:
            raise ResolutionCanaryError("candidate row count does not match recovery manifest")
        if request_count != recovery.queue_unique_count:
            raise ResolutionCanaryError("queue row count does not match recovery manifest")

        registry, registry_identity = _load_registry_with_identity(config)
        trace_source = registry.by_name(recovery.source_name)
        task_source = registry.by_name(recovery.task_source_name)
        if (
            trace_source.revision is None
            or trace_source.revision.casefold() != recovery.source_revision.casefold()
            or task_source.revision is None
            or task_source.revision.casefold() != recovery.task_source_revision.casefold()
        ):
            raise ResolutionCanaryError("registry revisions do not match recovery manifest")

        contract = TracePartitionContract.from_bytes(partition_contract.read_bytes())
        contract.require_source(recovery.source_name, recovery.source_revision)
        trace_receipt_payload = trace_transfer_receipt.read_bytes()
        trace_receipt = contract.bind_transfer_receipt(trace_receipt_payload)
        if (
            trace_receipt.source != trace_source
            or trace_receipt.registry_sha256 != registry_identity[0]
            or trace_receipt.registry_bytes != registry_identity[1]
        ):
            raise ResolutionCanaryError("trace receipt does not bind the selected registry")
        require_authorized_snapshot_seal(
            source_name=trace_source.name,
            source_revision=recovery.source_revision,
            transfer_receipt_sha256=trace_receipt_identity[0],
            snapshot_sha256=trace_receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(trace_receipt.snapshot.files),
        )

        task_receipt = DatasetSnapshotTransferReceipt.model_validate_json(
            task_transfer_receipt.read_bytes()
        )
        if (
            task_receipt.source != task_source
            or task_receipt.registry_sha256 != registry_identity[0]
            or task_receipt.registry_bytes != registry_identity[1]
        ):
            raise ResolutionCanaryError("task receipt does not bind the selected registry")
        require_authorized_snapshot_seal(
            source_name=task_source.name,
            source_revision=recovery.task_source_revision,
            transfer_receipt_sha256=task_receipt_identity[0],
            snapshot_sha256=task_receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(task_receipt.snapshot.files),
        )

        trace_expected = {
            identity.path: (identity.sha256, identity.bytes)
            for partition in recovery.target_partitions
            for identity in partition.files
        }
        trace_inputs = _receipt_bound_snapshot_files(
            trace_snapshot,
            expected_by_path=trace_expected,
            patterns=tuple(sorted(trace_expected)),
        )
        task_expected = {
            identity.path: (identity.sha256, identity.bytes)
            for identity in task_receipt.snapshot.files
        }
        task_inputs = _receipt_bound_snapshot_files(
            task_snapshot,
            expected_by_path=task_expected,
            patterns=tuple(sorted(task_expected)),
        )

        selected_instance_ids = {item.instance_id for item in selection.transfer_controls} | {
            item.instance_id for item in selection.evaluation_requests
        }
        trace_paths = tuple(sorted(trace_expected))
        with (
            verified_staged_files(trace_inputs) as staged_trace_files,
            verified_staged_files(task_inputs) as staged_task_files,
        ):
            staged_trace_by_path = dict(zip(trace_paths, staged_trace_files, strict=True))

            def target_rows() -> Iterator[tuple[str, dict[str, Any]]]:
                columns = (
                    "instance_id",
                    "trajectory_id",
                    "resolved",
                    "language",
                    "hf_dataset_name",
                    "metadata.model_patch.patch",
                )
                for partition in recovery.target_partitions:
                    paths = tuple(
                        staged_trace_by_path[identity.path] for identity in partition.files
                    )
                    for row in iter_snapshot_rows(paths, columns=columns):
                        yield partition.partition_name, row

            recovered_controls = recover_transfer_control_requests(
                selection.transfer_controls,
                target_rows(),
                trace_source_revision=recovery.source_revision,
                task_source_revision=recovery.task_source_revision,
            )

            tasks_by_instance: dict[str, SWERebenchTask] = {}
            task_columns = (
                "instance_id",
                "repo",
                "base_commit",
                "language",
                "image_name",
                "test_patch",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
                "install_config",
            )
            for row in iter_snapshot_rows(staged_task_files, columns=task_columns):
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str) or instance_id not in selected_instance_ids:
                    continue
                task = project_swe_rebench_task(
                    row,
                    task_source_revision=recovery.task_source_revision,
                )
                existing = tasks_by_instance.get(task.instance_id)
                if existing is not None and existing != task:
                    raise ResolutionCanaryError(
                        "selected instance has conflicting private task rows"
                    )
                tasks_by_instance[task.instance_id] = task
            if set(tasks_by_instance) != selected_instance_ids:
                raise ResolutionCanaryError(
                    "private task snapshot does not contain every selected instance"
                )

            try:
                cases = materialize_resolution_canary_cases(
                    selection,
                    recovered_controls=recovered_controls,
                    tasks_by_instance=tasks_by_instance,
                )
            except ValidationError as error:
                raise ResolutionCanaryError(
                    "private canary cases failed schema validation"
                ) from error

            def verify_inputs() -> None:
                for path, identity in (
                    *trace_inputs,
                    *task_inputs,
                    (recovery_manifest, recovery_identity),
                    (candidates, candidate_identity),
                    (queue, queue_identity),
                    (partition_contract, contract_identity),
                    (trace_transfer_receipt, trace_receipt_identity),
                    (task_transfer_receipt, task_receipt_identity),
                    (config, registry_identity),
                ):
                    _require_unchanged_file(path, identity)

            def write_workset(stream: BinaryIO) -> None:
                for case in cases:
                    stream.write(canonical_json_bytes(case.model_dump(mode="json")))

            workset_result = write_immutable_stream(
                workset_output,
                write_workset,
                before_publish=verify_inputs,
            )
            workset_identity = _published_artifact_identity(
                workset_result.path,
                workset_result.digest,
            )

            canary_manifest = ResolutionCanaryWorksetManifestV1(
                schema_version="nodelm.resolution-canary-workset/v1",
                materialization_status="PASS",
                execution_status="NOT RUN",
                admission_status="BLOCKED",
                admission_blocker="canary_execution_pending",
                recovery_manifest_sha256=recovery_identity[0],
                recovery_manifest_bytes=recovery_identity[1],
                candidate_sha256=candidate_identity[0],
                candidate_bytes=candidate_identity[1],
                queue_sha256=queue_identity[0],
                queue_bytes=queue_identity[1],
                trace_source_name=recovery.source_name,
                trace_source_revision=recovery.source_revision,
                trace_transfer_receipt_sha256=trace_receipt_identity[0],
                task_source_name=recovery.task_source_name,
                task_source_revision=recovery.task_source_revision,
                task_transfer_receipt_sha256=task_receipt_identity[0],
                selection_algorithm="nodelm.resolution-canary-cover/v1",
                minimum_per_kind=minimum_per_kind,
                maximum_per_kind=maximum_per_kind,
                evaluator_repository_id=SWE_REBENCH_EVALUATOR_REPOSITORY_ID,
                evaluator_revision=SWE_REBENCH_EVALUATOR_REVISION,
                workset_artifact=os.path.relpath(
                    workset_result.path,
                    start=manifest_output.resolve().parent,
                ),
                workset_sha256=workset_identity[0],
                workset_bytes=workset_identity[1],
                case_count=len(cases),
                transfer_control_count=len(selection.transfer_controls),
                evaluation_request_count=len(selection.evaluation_requests),
                languages=tuple(sorted({case.language for case in cases})),
                target_partitions=tuple(
                    sorted(
                        {
                            reference.partition_name
                            for case in cases
                            for reference in case.target_references
                        }
                    )
                ),
            )

            def verify_manifest_boundary() -> None:
                verify_inputs()
                _require_unchanged_file(workset_result.path, workset_identity)

            manifest_result = write_immutable_json(
                manifest_output,
                canary_manifest.model_dump(mode="json"),
                before_publish=verify_manifest_boundary,
            )
    except (
        OSError,
        PartitionContractError,
        RegistryError,
        ResolutionCanaryError,
        ResolutionRecoveryError,
        SnapshotSealError,
        ValidationError,
        ValueError,
        VerifiedStagingError,
    ) as error:
        raise typer.BadParameter(f"resolution canary workset failed: {error}") from error

    typer.echo(
        f"wrote workset={workset_result.path} cases={canary_manifest.case_count}; "
        f"manifest={manifest_result.path} execution={canary_manifest.execution_status}"
    )


def _load_resolution_canary_workset(
    workset: Path,
    workset_manifest: Path,
) -> tuple[
    tuple[ResolutionCanaryCase, ...],
    ResolutionCanaryWorksetManifestV1,
    tuple[str, int],
    tuple[str, int],
]:
    workset_identity = file_identity(workset)
    manifest_identity = file_identity(workset_manifest)
    manifest = ResolutionCanaryWorksetManifestV1.model_validate_json(workset_manifest.read_bytes())
    if workset_identity != (manifest.workset_sha256, manifest.workset_bytes):
        raise ResolutionCanaryError("private workset does not match its manifest")
    if (
        manifest.evaluator_repository_id != SWE_REBENCH_EVALUATOR_REPOSITORY_ID
        or manifest.evaluator_revision.casefold() != SWE_REBENCH_EVALUATOR_REVISION
    ):
        raise ResolutionCanaryError("private workset does not bind the approved evaluator")
    try:
        cases = tuple(ResolutionCanaryCase.model_validate(row) for row in _read_jsonl(workset))
    except ValidationError as error:
        raise ResolutionCanaryError("private workset failed schema validation") from error
    if tuple(case.case_id for case in cases) != tuple(sorted({case.case_id for case in cases})):
        raise ResolutionCanaryError("private workset cases must be unique and sorted")
    controls = sum(case.kind == "transfer_control" for case in cases)
    requests = sum(case.kind == "evaluation_request" for case in cases)
    if (
        len(cases) != manifest.case_count
        or controls != manifest.transfer_control_count
        or requests != manifest.evaluation_request_count
        or tuple(sorted({case.language for case in cases})) != manifest.languages
    ):
        raise ResolutionCanaryError("private workset contents do not match manifest accounting")
    return cases, manifest, workset_identity, manifest_identity


@datasets_app.command("lock-resolution-canary-images")
def datasets_lock_resolution_canary_images(
    workset: Path = typer.Option(..., "--workset", exists=True, dir_okay=False),
    workset_manifest: Path = typer.Option(..., "--workset-manifest", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    runtime: str = typer.Option("rootless-podman", "--runtime"),
    podman: str = typer.Option("podman", "--podman"),
    skopeo: str = typer.Option("skopeo", "--skopeo"),
    image_root: Path = typer.Option(Path("/var/lib/nodelm-canary/images"), "--image-root"),
) -> None:
    """Pull only selected canary images and publish immutable digest pins."""

    try:
        cases, _, workset_identity, manifest_identity = _load_resolution_canary_workset(
            workset,
            workset_manifest,
        )
        output.resolve().parent.mkdir(parents=True, exist_ok=True)
        locker: PodmanImageLocker | SkopeoChrootImageLocker
        if runtime == "rootless-podman":
            locker = PodmanImageLocker(executable=podman)
        elif runtime == "seccomp-chroot":
            locker = SkopeoChrootImageLocker(image_root=image_root, executable=skopeo)
        else:
            raise ResolutionCanaryError("unsupported resolution canary runtime")
        lock = locker.lock(
            tuple(case.task.image_name for case in cases),
            workspace=output.resolve().parent,
            workset_sha256=workset_identity[0],
        )

        def verify_inputs() -> None:
            _require_unchanged_file(workset, workset_identity)
            _require_unchanged_file(workset_manifest, manifest_identity)

        result = write_immutable_json(
            output,
            lock.model_dump(mode="json"),
            before_publish=verify_inputs,
        )
    except (OSError, ResolutionCanaryError, ValidationError, ValueError) as error:
        raise typer.BadParameter(f"resolution canary image locking failed: {error}") from error
    typer.echo(f"wrote image_lock={result.path} images={len(lock.images)}")


def _load_private_case_evidence(
    path: Path,
    *,
    case: ResolutionCanaryCase,
    image: PinnedContainerImage,
) -> ResolutionCanaryPrivateCaseEvidence:
    try:
        evidence = ResolutionCanaryPrivateCaseEvidence.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise ResolutionCanaryError("private case evidence failed schema validation") from error
    if (
        evidence.result.case_id != case.case_id
        or evidence.result.kind != case.kind
        or evidence.result.source_id != case.source_id
        or evidence.result.image != image
    ):
        raise ResolutionCanaryError("private case evidence does not match selected canary case")
    return evidence


@datasets_app.command("run-resolution-canary")
def datasets_run_resolution_canary(
    workset: Path = typer.Option(..., "--workset", exists=True, dir_okay=False),
    workset_manifest: Path = typer.Option(..., "--workset-manifest", exists=True, dir_okay=False),
    image_lock: Path = typer.Option(..., "--image-lock", exists=True, dir_okay=False),
    evaluator_root: Path = typer.Option(..., "--evaluator-root", exists=True, file_okay=False),
    case_evidence_dir: Path = typer.Option(..., "--case-evidence-dir", file_okay=False),
    results_output: Path = typer.Option(..., "--results-output", dir_okay=False),
    manifest_output: Path = typer.Option(..., "--manifest-output", dir_okay=False),
    code_commit: str = typer.Option(..., "--code-commit"),
    podman: str = typer.Option("podman", "--podman"),
    skopeo: str = typer.Option("skopeo", "--skopeo"),
    umoci: str = typer.Option("umoci", "--umoci"),
    image_root: Path = typer.Option(Path("/var/lib/nodelm-canary/images"), "--image-root"),
) -> None:
    """Run the private real-repository canary with offline, bounded containers."""

    if results_output.resolve() == manifest_output.resolve():
        raise typer.BadParameter("results and manifest outputs must be distinct")
    try:
        cases, workset_record, workset_identity, workset_manifest_identity = (
            _load_resolution_canary_workset(workset, workset_manifest)
        )
        image_lock_identity = file_identity(image_lock)
        locked = ResolutionCanaryImageLock.model_validate_json(image_lock.read_bytes())
        if locked.workset_sha256 != workset_identity[0]:
            raise ResolutionCanaryError("image lock does not bind the private workset")
        images = locked.by_source_image()
        required_images = {case.task.image_name for case in cases}
        if set(images) != required_images:
            raise ResolutionCanaryError("image lock does not exactly cover selected images")

        parser = PinnedEvaluatorLogParser(evaluator_root)
        oracle = ResolutionCanaryOracle(parser)
        sandbox: SWERebenchPodmanSandbox | SWERebenchSeccompChrootSandbox
        if locked.runtime == "rootless-podman":
            sandbox = SWERebenchPodmanSandbox(executable=podman)
        else:
            sandbox = SWERebenchSeccompChrootSandbox(
                image_root=image_root,
                skopeo=skopeo,
                umoci=umoci,
            )
        case_evidence_dir.mkdir(parents=True, exist_ok=True)
        if case_evidence_dir.is_symlink() or not case_evidence_dir.is_dir():
            raise ResolutionCanaryError("private case evidence directory is unsafe")
        expected_names = {f"{case.case_id}.json" for case in cases}
        actual_names = {path.name for path in case_evidence_dir.iterdir()}
        if not actual_names.issubset(expected_names):
            raise ResolutionCanaryError("private case evidence directory has unexpected entries")

        safe_results: list[ResolutionCanaryCaseResult] = []
        evidence_identities: list[tuple[Path, tuple[str, int]]] = []
        for case in cases:
            image = images[case.task.image_name]
            evidence_path = case_evidence_dir / f"{case.case_id}.json"
            if evidence_path.exists():
                if not evidence_path.is_file() or evidence_path.is_symlink():
                    raise ResolutionCanaryError("private case evidence path is unsafe")
                evidence = _load_private_case_evidence(
                    evidence_path,
                    case=case,
                    image=image,
                )
            else:
                baseline = sandbox.run(case, image, include_model_patch=False)
                candidate = sandbox.run(case, image, include_model_patch=True)
                result = oracle.evaluate(
                    case,
                    image=image,
                    baseline=baseline,
                    candidate=candidate,
                    sandbox_evidence=sandbox.evidence(),
                )
                evidence = ResolutionCanaryPrivateCaseEvidence(
                    result=result,
                    baseline_output=resolution_canary_output(baseline),
                    candidate_output=resolution_canary_output(candidate),
                )
                write_immutable_json(evidence_path, evidence.model_dump(mode="json"))
            safe_results.append(evidence.result)
            evidence_identities.append((evidence_path, file_identity(evidence_path)))

        def verify_inputs() -> None:
            for path, identity in (
                (workset, workset_identity),
                (workset_manifest, workset_manifest_identity),
                (image_lock, image_lock_identity),
                *evidence_identities,
            ):
                _require_unchanged_file(path, identity)
            if (
                file_identity(evaluator_root / "lib" / "agent" / "log_parsers.py")[0]
                != SWE_REBENCH_LOG_PARSERS_SHA256
                or file_identity(evaluator_root / "scripts" / "eval.py")[0]
                != SWE_REBENCH_EVAL_SCRIPT_SHA256
                or file_identity(evaluator_root / "lib" / "agent" / "swe_constants.py")[0]
                != SWE_REBENCH_CONSTANTS_SHA256
            ):
                raise ResolutionCanaryError("pinned evaluator changed during canary execution")

        def write_results(stream: BinaryIO) -> None:
            for result in safe_results:
                stream.write(canonical_json_bytes(result.model_dump(mode="json")))

        results_result = write_immutable_stream(
            results_output,
            write_results,
            before_publish=verify_inputs,
        )
        results_identity = _published_artifact_identity(
            results_result.path,
            results_result.digest,
        )
        failures = Counter(
            result.reason for result in safe_results if result.status is VerificationStatus.FAIL
        )
        failed_count = sum(failures.values())
        controls = tuple(result for result in safe_results if result.kind == "transfer_control")
        requests = tuple(result for result in safe_results if result.kind == "evaluation_request")
        execution = ResolutionCanaryExecutionManifestV1(
            schema_version="nodelm.resolution-canary-execution/v1",
            execution_status="PASS" if failed_count == 0 else "FAIL",
            admission_status="PASS" if failed_count == 0 else "BLOCKED",
            admission_blocker=None if failed_count == 0 else "canary_case_failed",
            code_commit=code_commit,
            recovery_manifest_sha256=workset_record.recovery_manifest_sha256,
            workset_manifest_sha256=workset_manifest_identity[0],
            workset_manifest_bytes=workset_manifest_identity[1],
            workset_sha256=workset_identity[0],
            workset_bytes=workset_identity[1],
            image_lock_sha256=image_lock_identity[0],
            image_lock_bytes=image_lock_identity[1],
            evaluator_repository_id="SWE-rebench/SWE-rebench-V2",
            evaluator_revision="c71902a8cf8d2b725f63d51f199f4d3e56f68d2d",
            evaluator_log_parsers_sha256=parser.parser_sha256,
            evaluator_script_sha256=parser.eval_sha256,
            evaluator_constants_sha256=parser.constants_sha256,
            sandbox_backend=locked.runtime,
            sandbox_network="none",
            sandbox_cpus_per_attempt=2,
            sandbox_memory_per_attempt="4g",
            results_artifact=os.path.relpath(
                results_result.path,
                start=manifest_output.resolve().parent,
            ),
            results_sha256=results_identity[0],
            results_bytes=results_identity[1],
            case_count=len(safe_results),
            passed_case_count=len(safe_results) - failed_count,
            failed_case_count=failed_count,
            transfer_control_count=len(controls),
            transfer_label_agreement_count=sum(
                result.label_agreement is True for result in controls
            ),
            evaluation_request_count=len(requests),
            evaluation_resolved_count=sum(result.task_resolved is True for result in requests),
            evaluation_unresolved_count=sum(result.task_resolved is False for result in requests),
            image_count=len(locked.images),
            failure_counts_by_reason=dict(sorted(failures.items())),
        )

        def verify_manifest_boundary() -> None:
            verify_inputs()
            _require_unchanged_file(results_result.path, results_identity)

        manifest_result = write_immutable_json(
            manifest_output,
            execution.model_dump(mode="json"),
            before_publish=verify_manifest_boundary,
        )
    except (
        OSError,
        ResolutionCanaryError,
        ValidationError,
        ValueError,
    ) as error:
        raise typer.BadParameter(f"resolution canary execution failed: {error}") from error
    typer.echo(
        f"wrote results={results_result.path} cases={execution.case_count}; "
        f"manifest={manifest_result.path} admission={execution.admission_status}"
    )


@datasets_app.command("project-task-provenance")
def datasets_project_task_provenance(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    transfer_receipt: Path = typer.Option(..., "--transfer-receipt", exists=True, dir_okay=False),
    file_pattern: list[str] | None = typer.Option(None, "--file-pattern"),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    manifest_output: Path | None = typer.Option(None, "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    """Project pinned task rows into a license-safe, gold-free join artifact."""

    registry, config_identity = _load_registry_with_identity(config)
    source = registry.by_name(source_name)
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("task projection requires a registry-verified pinned source")
    try:
        receipt_payload = transfer_receipt.read_bytes()
        receipt_identity = (content_digest(receipt_payload), len(receipt_payload))
        receipt = DatasetSnapshotTransferReceipt.model_validate_json(receipt_payload)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(f"invalid transfer receipt: {error}") from error
    if receipt.snapshot_scope != "complete":
        raise typer.BadParameter("task projection requires a complete transfer receipt")
    if receipt.source != source:
        raise typer.BadParameter("transfer receipt source does not match --source")
    if (
        receipt.registry_sha256 != config_identity[0]
        or receipt.registry_bytes != config_identity[1]
    ):
        raise typer.BadParameter("transfer receipt registry identity does not match --config")
    try:
        require_authorized_snapshot_seal(
            source_name=source.name,
            source_revision=source.revision,
            transfer_receipt_sha256=receipt_identity[0],
            snapshot_sha256=receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(receipt.snapshot.files),
        )
    except SnapshotSealError as error:
        raise typer.BadParameter(str(error)) from error

    patterns = tuple(file_pattern or ())
    try:
        files = discover_snapshot_files(snapshot, patterns=patterns)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    identities = tuple((path, file_identity(path)) for path in files)
    snapshot_root = snapshot.resolve() if snapshot.is_dir() else snapshot.resolve().parent
    expected_by_path = {
        identity.path: (identity.sha256, identity.bytes)
        for identity in receipt.snapshot.files
        if not patterns or any(PurePosixPath(identity.path).match(pattern) for pattern in patterns)
    }
    actual_by_path = {
        path.relative_to(snapshot_root).as_posix(): identity for path, identity in identities
    }
    if actual_by_path != expected_by_path:
        raise typer.BadParameter("selected task files do not match the sealed transfer receipt")

    def verify_inputs() -> None:
        if discover_snapshot_files(snapshot, patterns=patterns) != files:
            raise typer.BadParameter("task snapshot file set changed while it was processed")
        for path, identity in identities:
            _require_unchanged_file(path, identity)
        _require_unchanged_file(config, config_identity)
        _require_unchanged_file(transfer_receipt, receipt_identity)

    rejection_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    try:
        with (
            verified_staged_files(identities) as staged_files,
            task_provenance_projection(
                iter_snapshot_rows(staged_files), source=source
            ) as projection,
        ):

            def write_admitted(stream: BinaryIO) -> None:
                for record in projection.iter_admitted():
                    stream.write(canonical_json_bytes(record.model_dump(mode="json")))

            admitted_result = write_immutable_stream(
                output,
                write_admitted,
                before_publish=verify_inputs,
            )
            admitted_identity = _published_artifact_identity(
                admitted_result.path,
                admitted_result.digest,
            )

            def write_rejections(stream: BinaryIO) -> None:
                for rejection in projection.iter_rejections():
                    stream.write(canonical_json_bytes(rejection))

            def verify_rejection_boundary() -> None:
                verify_inputs()
                _require_unchanged_file(admitted_result.path, admitted_identity)

            rejection_result = write_immutable_stream(
                rejection_path,
                write_rejections,
                before_publish=verify_rejection_boundary,
            )
            rejection_identity = _published_artifact_identity(
                rejection_result.path,
                rejection_result.digest,
            )
            admitted_count = projection.admitted_count
            rejected_count = projection.rejected_count
            rejection_counts = projection.rejection_counts_by_code
    except (ValueError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"task provenance projection failed: {error}") from error

    verify_inputs()
    manifest_path = manifest_output or output.with_name(f"{output.stem}.manifest.json")
    manifest = TaskProvenanceProjectionManifestV1.model_validate(
        {
            "schema_version": "nodelm.task-provenance-projection/v1",
            "status": (
                VerificationStatus.PASS.value if admitted_count else VerificationStatus.FAIL.value
            ),
            "source_name": source.name,
            "source_repository_id": source.repository_id,
            "source_revision": source.revision,
            "registry_sha256": config_identity[0],
            "registry_bytes": config_identity[1],
            "transfer_receipt_sha256": receipt_identity[0],
            "transfer_receipt_bytes": receipt_identity[1],
            "snapshot_sha256": receipt.snapshot.snapshot_sha256,
            "projection_scope": "filtered" if patterns else "complete-snapshot",
            "file_patterns": patterns,
            "files": [
                {
                    "path": path.relative_to(snapshot_root).as_posix(),
                    "sha256": identity[0],
                    "bytes": identity[1],
                }
                for path, identity in identities
            ],
            "safe_fields": TASK_PROVENANCE_SAFE_FIELDS,
            "admitted_count": admitted_count,
            "rejected_count": rejected_count,
            "rejection_counts_by_code": rejection_counts,
            "output": os.path.relpath(admitted_result.path, start=manifest_path.resolve().parent),
            "output_sha256": admitted_result.digest,
            "output_bytes": admitted_identity[1],
            "rejection_artifact": os.path.relpath(
                rejection_result.path, start=manifest_path.resolve().parent
            ),
            "rejection_sha256": rejection_result.digest,
            "rejection_bytes": rejection_identity[1],
        }
    )

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_unchanged_file(admitted_result.path, admitted_identity)
        _require_unchanged_file(rejection_result.path, rejection_identity)

    manifest_result = write_immutable_json(
        manifest_path,
        manifest.model_dump(mode="json"),
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        f"wrote {admitted_result.path} admitted={admitted_count} rejected={rejected_count} "
        f"sha256={admitted_result.digest}; manifest={manifest_result.path}"
    )
    if admitted_count == 0:
        raise typer.Exit(code=1)


@datasets_app.command("normalize")
def datasets_normalize(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    materialization_manifest: Path = typer.Option(
        ..., "--materialization-manifest", exists=True, dir_okay=False
    ),
    partition_contract: Path = typer.Option(
        ..., "--partition-contract", exists=True, dir_okay=False
    ),
    transfer_receipt: Path = typer.Option(..., "--transfer-receipt", exists=True, dir_okay=False),
    task_provenance: Path = typer.Option(..., "--task-provenance", exists=True, dir_okay=False),
    task_provenance_manifest: Path = typer.Option(
        ..., "--task-provenance-manifest", exists=True, dir_okay=False
    ),
    task_transfer_receipt: Path = typer.Option(
        ..., "--task-transfer-receipt", exists=True, dir_okay=False
    ),
    task_snapshot: Path = typer.Option(..., "--task-snapshot", exists=True),
    expected_harness: str | None = typer.Option(None, "--expect-harness", "--harness"),
    expected_generating_model: str | None = typer.Option(
        None, "--expect-generating-model", "--generating-model"
    ),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    manifest_output: Path | None = typer.Option(None, "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    registry, config_identity = _load_registry_with_identity(config)
    source = registry.by_name(source_name)
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("normalization requires a registry-verified pinned source")
    input_identity = file_identity(input_path)
    task_identity = file_identity(task_provenance)
    materialization_payload, materialization_identity = _read_json_mapping_with_identity(
        materialization_manifest
    )
    task_manifest_payload, task_manifest_identity = _read_json_mapping_with_identity(
        task_provenance_manifest
    )
    try:
        materialization = SnapshotMaterializationManifestV2.model_validate(materialization_payload)
        task_manifest = TaskProvenanceProjectionManifestV1.model_validate(task_manifest_payload)
    except ValidationError as error:
        raise typer.BadParameter(f"invalid normalization manifest evidence: {error}") from error
    try:
        contract_payload = partition_contract.read_bytes()
        contract_identity = (content_digest(contract_payload), len(contract_payload))
        contract = TracePartitionContract.from_bytes(contract_payload)
        contract.require_source(source.name, source.revision)
        contract.require_authorized_digest(contract_identity[0])
        receipt_payload = transfer_receipt.read_bytes()
        receipt_identity = (content_digest(receipt_payload), len(receipt_payload))
        receipt = contract.bind_transfer_receipt(receipt_payload)
        require_authorized_snapshot_seal(
            source_name=source.name,
            source_revision=source.revision,
            transfer_receipt_sha256=receipt_identity[0],
            snapshot_sha256=receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(receipt.snapshot.files),
        )
    except (OSError, PartitionContractError, SnapshotSealError) as error:
        raise typer.BadParameter(f"invalid partition evidence: {error}") from error
    if (
        contract.source_repository_id != source.repository_id
        or contract.sealed_registry_sha256 != config_identity[0]
    ):
        raise typer.BadParameter("partition contract does not match the dataset registry")

    if materialization.status != VerificationStatus.PASS.value:
        raise typer.BadParameter("materialization manifest is not PASS")
    if (
        materialization.source_name != source.name
        or materialization.source_repository_id != source.repository_id
        or materialization.source_revision != source.revision
        or materialization.registry_sha256 != config_identity[0]
    ):
        raise typer.BadParameter("materialization manifest source does not match normalization")
    if (
        materialization.output_sha256 != input_identity[0]
        or materialization.output_bytes != input_identity[1]
    ):
        raise typer.BadParameter("materialization manifest output identity does not match --input")
    materialized_path = Path(materialization.output)
    if not materialized_path.is_absolute():
        materialized_path = materialization_manifest.resolve().parent / materialized_path
    if materialized_path.resolve() != input_path.resolve():
        raise typer.BadParameter("materialization manifest output path does not match --input")
    if (
        materialization.partition_contract_sha256 != contract_identity[0]
        or materialization.partition_contract_bytes != contract_identity[1]
        or materialization.transfer_receipt_sha256 != receipt_identity[0]
        or materialization.transfer_receipt_bytes != receipt_identity[1]
    ):
        raise typer.BadParameter("materialization manifest partition evidence is inconsistent")

    try:
        selected_partition = contract.by_name(materialization.partition_name)
    except PartitionContractError as error:
        raise typer.BadParameter(str(error)) from error
    if selected_partition.normalization_status is not VerificationStatus.PASS:
        raise typer.BadParameter(
            f"trace partition is blocked from normalization: {selected_partition.name}"
        )
    if materialization.file_patterns != selected_partition.file_patterns:
        raise typer.BadParameter(
            "materialization file patterns do not match the selected partition"
        )
    receipt_partition_files = {
        identity.path: (identity.sha256, identity.bytes)
        for identity in receipt.snapshot.files
        if any(
            PurePosixPath(identity.path).match(pattern)
            for pattern in selected_partition.file_patterns
        )
    }
    materialized_files = {
        identity.path: (identity.sha256, identity.bytes) for identity in materialization.files
    }
    if (
        len(materialized_files) != len(materialization.files)
        or materialized_files != receipt_partition_files
    ):
        raise typer.BadParameter("materialization files do not match the receipt-bound partition")
    materialized_row_count = materialization.row_count
    materialized_max_rows = materialization.max_rows
    expected_materialization_scope = (
        "canary" if materialized_max_rows is not None else "complete-partition"
    )
    if materialization.materialization_scope != expected_materialization_scope:
        raise typer.BadParameter("materialization scope does not match its max_rows bound")

    trace_snapshot_inputs = _receipt_bound_snapshot_files(
        snapshot,
        expected_by_path=receipt_partition_files,
        patterns=selected_partition.file_patterns,
    )
    try:
        with verified_staged_files(trace_snapshot_inputs) as staged_trace_files:

            def replay_materialized_rows() -> Iterator[dict[str, Any]]:
                for row_index, row in enumerate(iter_snapshot_rows(staged_trace_files)):
                    if materialized_max_rows is not None and row_index >= materialized_max_rows:
                        break
                    yield row

            replayed_input_identity, replayed_row_count = _canonical_jsonl_identity(
                replay_materialized_rows()
            )
    except (ValueError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"unable to replay materialization: {error}") from error
    if replayed_input_identity != input_identity or replayed_row_count != materialized_row_count:
        raise typer.BadParameter(
            "materialized input is not the deterministic projection of its sealed snapshot"
        )
    expected_partition_fields = {
        "harness": selected_partition.harness,
        "generating_model": selected_partition.generating_model,
        "upstream_source": selected_partition.upstream_source,
        "row_dataset_name": selected_partition.row_dataset_name,
        "task_source_name": selected_partition.task_source_name,
        "task_source_revision": selected_partition.task_source_revision,
        "normalization_status": VerificationStatus.PASS.value,
    }
    if any(
        getattr(materialization, key) != value for key, value in expected_partition_fields.items()
    ):
        raise typer.BadParameter("materialization manifest partition labels are inconsistent")
    if expected_harness is not None and expected_harness != selected_partition.harness:
        raise typer.BadParameter("--expect-harness does not match the bound partition")
    if (
        expected_generating_model is not None
        and expected_generating_model != selected_partition.generating_model
    ):
        raise typer.BadParameter("--expect-generating-model does not match the bound partition")

    task_source_name = selected_partition.task_source_name
    task_source_revision = selected_partition.task_source_revision
    if task_source_name is None or task_source_revision is None:  # pragma: no cover - PASS model
        raise typer.BadParameter("normalizable partition lacks a pinned task source")
    task_source = registry.by_name(task_source_name)
    if (
        task_source.status is not VerificationStatus.PASS
        or task_source.revision != task_source_revision
    ):
        raise typer.BadParameter("partition task source is not registry-verified at its revision")
    try:
        task_receipt_payload = task_transfer_receipt.read_bytes()
        task_receipt_identity = (
            content_digest(task_receipt_payload),
            len(task_receipt_payload),
        )
        task_receipt = DatasetSnapshotTransferReceipt.model_validate_json(task_receipt_payload)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(f"invalid task transfer receipt: {error}") from error
    if (
        task_receipt.snapshot_scope != "complete"
        or task_receipt.source != task_source
        or task_receipt.registry_sha256 != config_identity[0]
        or task_receipt.registry_bytes != config_identity[1]
    ):
        raise typer.BadParameter("task transfer receipt does not match the bound task source")
    try:
        require_authorized_snapshot_seal(
            source_name=task_source.name,
            source_revision=task_source_revision,
            transfer_receipt_sha256=task_receipt_identity[0],
            snapshot_sha256=task_receipt.snapshot.snapshot_sha256,
            snapshot_file_count=len(task_receipt.snapshot.files),
        )
    except SnapshotSealError as error:
        raise typer.BadParameter(str(error)) from error
    if task_manifest.status != VerificationStatus.PASS.value:
        raise typer.BadParameter("task provenance manifest is not PASS")
    if (
        task_manifest.source_name != task_source_name
        or task_manifest.source_repository_id != task_source.repository_id
        or task_manifest.source_revision != task_source_revision
        or task_manifest.registry_sha256 != config_identity[0]
        or task_manifest.registry_bytes != config_identity[1]
        or task_manifest.projection_scope != "complete-snapshot"
        or task_manifest.file_patterns
        or task_manifest.transfer_receipt_sha256 != task_receipt_identity[0]
        or task_manifest.transfer_receipt_bytes != task_receipt_identity[1]
        or task_manifest.snapshot_sha256 != task_receipt.snapshot.snapshot_sha256
    ):
        raise typer.BadParameter("task provenance manifest does not match the bound task source")
    if (
        task_manifest.output_sha256 != task_identity[0]
        or task_manifest.output_bytes != task_identity[1]
    ):
        raise typer.BadParameter("task provenance manifest output does not match its artifact")
    task_artifact_path = Path(task_manifest.output)
    if not task_artifact_path.is_absolute():
        task_artifact_path = task_provenance_manifest.resolve().parent / task_artifact_path
    if task_artifact_path.resolve() != task_provenance.resolve():
        raise typer.BadParameter("task provenance manifest output path does not match artifact")
    if task_manifest.safe_fields != TASK_PROVENANCE_SAFE_FIELDS:
        raise typer.BadParameter("task provenance manifest safe field contract is incomplete")
    task_receipt_files = {
        identity.path: (identity.sha256, identity.bytes) for identity in task_receipt.snapshot.files
    }
    task_manifest_files = {
        identity.path: (identity.sha256, identity.bytes) for identity in task_manifest.files
    }
    if (
        len(task_manifest_files) != len(task_manifest.files)
        or task_manifest_files != task_receipt_files
    ):
        raise typer.BadParameter("task provenance files do not match the sealed task receipt")

    task_rejection_path = Path(task_manifest.rejection_artifact)
    if not task_rejection_path.is_absolute():
        task_rejection_path = task_provenance_manifest.resolve().parent / task_rejection_path
    try:
        task_rejection_identity = file_identity(task_rejection_path)
    except OSError as error:
        raise typer.BadParameter(f"unable to read task rejection artifact: {error}") from error
    if (
        task_manifest.rejection_sha256 != task_rejection_identity[0]
        or task_manifest.rejection_bytes != task_rejection_identity[1]
    ):
        raise typer.BadParameter("task rejection artifact identity does not match its manifest")

    task_snapshot_inputs = _receipt_bound_snapshot_files(
        task_snapshot,
        expected_by_path=task_receipt_files,
    )
    try:
        with (
            verified_staged_files(task_snapshot_inputs) as staged_task_files,
            task_provenance_projection(
                iter_snapshot_rows(staged_task_files), source=task_source
            ) as replayed_projection,
        ):
            replayed_task_identity, replayed_admitted_count = _canonical_jsonl_identity(
                record.model_dump(mode="json") for record in replayed_projection.iter_admitted()
            )
            replayed_rejection_identity, replayed_rejected_count = _canonical_jsonl_identity(
                replayed_projection.iter_rejections()
            )
            replayed_rejection_counts = replayed_projection.rejection_counts_by_code
    except (ValueError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"unable to replay task provenance: {error}") from error
    if (
        replayed_task_identity != task_identity
        or replayed_rejection_identity != task_rejection_identity
        or task_manifest.admitted_count != replayed_admitted_count
        or task_manifest.rejected_count != replayed_rejected_count
        or task_manifest.rejection_counts_by_code != replayed_rejection_counts
    ):
        raise typer.BadParameter(
            "task provenance is not the deterministic projection of its sealed snapshot"
        )

    rejection_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    extra_lineage = normalization_evidence_lineage(
        materialization_manifest_sha256=materialization_identity[0],
        partition_name=selected_partition.name,
        upstream_source=selected_partition.upstream_source,
        task_source_name=task_source_name,
        task_source_revision=task_source_revision,
        task_provenance_sha256=task_identity[0],
    )

    def verify_inputs() -> None:
        for path, identity in trace_snapshot_inputs:
            _require_unchanged_file(path, identity)
        for path, identity in task_snapshot_inputs:
            _require_unchanged_file(path, identity)
        _require_unchanged_file(input_path, input_identity)
        _require_unchanged_file(config, config_identity)
        _require_unchanged_file(materialization_manifest, materialization_identity)
        _require_unchanged_file(partition_contract, contract_identity)
        _require_unchanged_file(transfer_receipt, receipt_identity)
        _require_unchanged_file(task_provenance, task_identity)
        _require_unchanged_file(task_provenance_manifest, task_manifest_identity)
        _require_unchanged_file(task_transfer_receipt, task_receipt_identity)
        _require_unchanged_file(task_rejection_path, task_rejection_identity)

    try:
        with (
            verified_staged_files(
                ((input_path, input_identity), (task_provenance, task_identity))
            ) as staged_derived_files,
            task_metadata_index(
                _read_jsonl(staged_derived_files[1]),
                expected_source_name=task_source_name,
                expected_source_revision=task_source_revision,
            ) as lookup,
            trace_normalization_projection() as projection,
        ):
            for row_index, row in enumerate(_read_jsonl(staged_derived_files[0])):
                raw_row_sha256 = stable_model_id(row)
                try:
                    rollout_key = trace_rollout_key(
                        row,
                        source=source,
                        partition_name=selected_partition.name,
                        row_dataset_name=selected_partition.row_dataset_name,
                        task_lookup=lookup,
                    )
                except NormalizationError:
                    rollout_key = None
                sample_id: str | None
                reason_code: str | None
                reason: str | None
                try:
                    sample = normalize_trace_sample(
                        row,
                        source=source,
                        harness=selected_partition.harness,
                        generating_model=selected_partition.generating_model,
                        task_lookup=lookup,
                        require_task_match=True,
                        expected_row_dataset_name=selected_partition.row_dataset_name,
                        extra_lineage=extra_lineage,
                    )
                except NormalizationError as error:
                    sample_id = None
                    reason_code = (
                        "unknown_resolution"
                        if isinstance(error, UnknownResolutionError)
                        else "normalization_error"
                    )
                    reason = str(error)
                else:
                    sample_id = sample.sample_id
                    reason_code = None
                    reason = None
                instance_id = row.get("instance_id")
                repository = row.get("repo") or row.get("repository")
                rollout_id = row.get("trajectory_id") or row.get("rollout_id")
                projection.add_row(
                    row_index=row_index,
                    rollout_key=rollout_key,
                    raw_row_sha256=raw_row_sha256,
                    sample_id=sample_id,
                    reason_code=reason_code,
                    reason=reason,
                    instance_id=(
                        str(instance_id)
                        if isinstance(instance_id, (str, int)) and not isinstance(instance_id, bool)
                        else None
                    ),
                    repository=repository if isinstance(repository, str) else None,
                    rollout_id=rollout_id if isinstance(rollout_id, str) else None,
                )
            projection.finalize()
            input_row_count = projection.input_row_count
            accepted_count = projection.admitted_count
            rejected_count = projection.rejected_count
            rejection_counts = projection.rejection_counts_by_code
            unique_rollout_key_count = projection.unique_rollout_key_count
            duplicate_trace_row_count = projection.duplicate_trace_row_count
            conflicting_rollout_identity_count = projection.conflicting_rollout_identity_count
            conflicting_rollout_row_count = projection.conflicting_rollout_row_count
            if (
                input_row_count != materialized_row_count
                or input_row_count != accepted_count + rejected_count
            ):
                raise TraceProjectionError("normalization row accounting invariant failed")

            def write_normalized(stream: BinaryIO) -> None:
                rows = _read_jsonl(staged_derived_files[0])
                for row, decision in zip(rows, projection.iter_decisions(), strict=True):
                    if stable_model_id(row) != decision.raw_row_sha256:
                        raise TraceProjectionError(
                            f"trace row changed after classification: {decision.row_index}"
                        )
                    if not decision.admitted:
                        continue
                    sample = normalize_trace_sample(
                        row,
                        source=source,
                        harness=selected_partition.harness,
                        generating_model=selected_partition.generating_model,
                        task_lookup=lookup,
                        require_task_match=True,
                        expected_row_dataset_name=selected_partition.row_dataset_name,
                        extra_lineage=extra_lineage,
                    )
                    if sample.sample_id != decision.sample_id:
                        raise TraceProjectionError(
                            f"normalized sample changed after classification: {decision.row_index}"
                        )
                    stream.write(canonical_json_bytes(sample.model_dump(mode="json")))

            normalized_result = write_immutable_stream(
                output,
                write_normalized,
                before_publish=verify_inputs,
            )
            normalized_identity = _published_artifact_identity(
                normalized_result.path,
                normalized_result.digest,
            )

            def write_rejections(stream: BinaryIO) -> None:
                for decision in projection.iter_decisions():
                    if decision.admitted:
                        continue
                    stream.write(
                        canonical_json_bytes(
                            {
                                "row_index": decision.row_index,
                                "rollout_key": decision.rollout_key,
                                "raw_row_sha256": decision.raw_row_sha256,
                                "instance_id": decision.instance_id,
                                "repository": decision.repository,
                                "rollout_id": decision.rollout_id,
                                "reason_code": decision.reason_code,
                                "cause_code": decision.cause_code,
                                "reason": decision.reason,
                            }
                        )
                    )

            def verify_rejection_boundary() -> None:
                verify_inputs()
                _require_unchanged_file(normalized_result.path, normalized_identity)

            rejection_result = write_immutable_stream(
                rejection_path,
                write_rejections,
                before_publish=verify_rejection_boundary,
            )
            rejection_identity = _published_artifact_identity(
                rejection_result.path,
                rejection_result.digest,
            )
    except (NormalizationError, TraceProjectionError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"invalid task provenance: {error}") from error

    verify_inputs()
    manifest_path = manifest_output or output.with_name(f"{output.stem}.manifest.json")
    manifest = NormalizationManifestV2.model_validate(
        {
            "schema_version": "nodelm.normalization-manifest/v2",
            "status": (
                VerificationStatus.PASS.value if accepted_count else VerificationStatus.FAIL.value
            ),
            "source_name": source.name,
            "source_repository_id": source.repository_id,
            "source_revision": source.revision,
            "partition_name": selected_partition.name,
            "harness": selected_partition.harness,
            "generating_model": selected_partition.generating_model,
            "upstream_source": selected_partition.upstream_source,
            "row_dataset_name": selected_partition.row_dataset_name,
            "input_sha256": input_identity[0],
            "input_bytes": input_identity[1],
            "registry_sha256": config_identity[0],
            "materialization_manifest_sha256": materialization_identity[0],
            "materialization_manifest_bytes": materialization_identity[1],
            "partition_contract_sha256": contract_identity[0],
            "partition_contract_bytes": contract_identity[1],
            "transfer_receipt_sha256": receipt_identity[0],
            "transfer_receipt_bytes": receipt_identity[1],
            "task_provenance_sha256": task_identity[0],
            "task_provenance_bytes": task_identity[1],
            "task_provenance_manifest_sha256": task_manifest_identity[0],
            "task_provenance_manifest_bytes": task_manifest_identity[1],
            "task_transfer_receipt_sha256": task_receipt_identity[0],
            "task_transfer_receipt_bytes": task_receipt_identity[1],
            "task_source_name": task_source_name,
            "task_source_revision": task_source_revision,
            "materialization_replay": VerificationStatus.PASS.value,
            "task_provenance_replay": VerificationStatus.PASS.value,
            "uniqueness_scope": expected_materialization_scope,
            "input_row_count": input_row_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "rejection_counts_by_code": rejection_counts,
            "unique_rollout_key_count": unique_rollout_key_count,
            "duplicate_trace_row_count": duplicate_trace_row_count,
            "conflicting_rollout_identity_count": conflicting_rollout_identity_count,
            "conflicting_rollout_row_count": conflicting_rollout_row_count,
            "gold_exposure_audit": VerificationStatus.NOT_RUN.value,
            "normalized_artifact": os.path.relpath(
                normalized_result.path, start=manifest_path.resolve().parent
            ),
            "normalized_sha256": normalized_result.digest,
            "normalized_bytes": normalized_identity[1],
            "rejection_artifact": os.path.relpath(
                rejection_result.path, start=manifest_path.resolve().parent
            ),
            "rejection_sha256": rejection_result.digest,
            "rejection_bytes": rejection_identity[1],
        }
    )

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_unchanged_file(normalized_result.path, normalized_identity)
        _require_unchanged_file(rejection_result.path, rejection_identity)

    manifest_result = write_immutable_json(
        manifest_path,
        manifest.model_dump(mode="json"),
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        f"wrote {normalized_result.path} accepted={accepted_count} rejected={rejected_count} "
        f"sha256={normalized_result.digest}; manifest={manifest_result.path}"
    )
    if accepted_count == 0:
        raise typer.Exit(code=1)


@datasets_app.command("build-normalization-cohort")
def datasets_build_normalization_cohort(
    member_manifests: list[Path] = typer.Option(
        ...,
        "--member-manifest",
        exists=True,
        dir_okay=False,
        help="Repeat for each complete-partition normalization manifest",
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False),
) -> None:
    """Bind selected complete normalized leaves without concatenating their data files."""

    try:
        result, cohort = build_normalization_cohort(tuple(member_manifests), output)
    except (NormalizationCohortError, OSError, ValidationError) as error:
        raise typer.BadParameter(f"invalid normalization cohort: {error}") from error
    typer.echo(
        f"wrote {result.path} members={cohort.member_count} "
        f"samples={cohort.sample_count} sha256={result.digest}"
    )


@datasets_app.command("review-oracle-isolation")
def datasets_review_oracle_isolation(
    raw_input: Path = typer.Option(..., "--raw-input", exists=True, dir_okay=False),
    materialization_manifest: Path = typer.Option(
        ..., "--materialization-manifest", exists=True, dir_okay=False
    ),
    normalized_input: Path = typer.Option(
        ..., "--input", "--normalized", exists=True, dir_okay=False
    ),
    normalization_manifest: Path = typer.Option(
        ..., "--normalization-manifest", exists=True, dir_okay=False
    ),
    task_provenance: Path = typer.Option(..., "--task-provenance", exists=True, dir_okay=False),
    task_provenance_manifest: Path = typer.Option(
        ..., "--task-provenance-manifest", exists=True, dir_okay=False
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    findings_output: Path | None = typer.Option(None, "--findings-output", dir_okay=False),
) -> None:
    """Review recorded model context against exact raw and normalized populations."""

    findings_path = findings_output or output.with_name(f"{output.stem}.findings.jsonl")
    try:
        result, attestation = review_oracle_isolation_artifacts(
            raw_input=raw_input,
            materialization_manifest=materialization_manifest,
            normalized_input=normalized_input,
            normalization_manifest=normalization_manifest,
            task_provenance=task_provenance,
            task_provenance_manifest=task_provenance_manifest,
            output=output,
            findings_output=findings_path,
        )
    except (OSError, OracleIsolationReviewError, ValidationError) as error:
        raise typer.BadParameter(f"oracle-isolation review failed: {error}") from error
    typer.echo(
        f"wrote {result.path} status={attestation.status.value} "
        f"covered={attestation.covered_sample_count}/"
        f"{attestation.expected_sample_count} findings={attestation.finding_count} "
        f"sha256={result.digest}"
    )
    if attestation.status is not VerificationStatus.PASS:
        raise typer.Exit(code=1)


def _oracle_attestation_matches_normalization(
    attestation: OracleIsolationAttestation,
    *,
    normalization: NormalizationManifestV2,
    normalization_identity: tuple[str, int],
    normalized_identity: tuple[str, int],
    expected_sample_count: int,
) -> bool:
    return (
        attestation.source_name == normalization.source_name
        and attestation.source_repository_id == normalization.source_repository_id
        and attestation.source_revision == normalization.source_revision
        and attestation.partition_name == normalization.partition_name
        and attestation.harness == normalization.harness
        and attestation.generating_model == normalization.generating_model
        and attestation.materialization_manifest_sha256
        == normalization.materialization_manifest_sha256
        and attestation.materialization_manifest_bytes
        == normalization.materialization_manifest_bytes
        and attestation.raw_sha256 == normalization.input_sha256
        and attestation.raw_bytes == normalization.input_bytes
        and attestation.raw_row_count == normalization.input_row_count
        and attestation.normalization_manifest_sha256 == normalization_identity[0]
        and attestation.normalization_manifest_bytes == normalization_identity[1]
        and attestation.normalized_sha256 == normalized_identity[0]
        and attestation.normalized_bytes == normalized_identity[1]
        and attestation.expected_sample_count == expected_sample_count
        and attestation.covered_sample_count == expected_sample_count
    )


def _resolve_attestation_evidence_chain(
    *,
    attestation_path: Path,
    attestation: OracleIsolationAttestation,
    normalization_manifest: Path,
    normalization: NormalizationManifestV2,
    normalization_identity: tuple[str, int],
    normalized_input: Path,
    normalized_identity: tuple[str, int],
) -> tuple[tuple[Path, tuple[str, int]], ...]:
    """Resolve and bind the complete retained evidence chain behind a v2 PASS claim."""

    def resolve(reference: Path, artifact: str) -> Path:
        path = Path(artifact)
        return path if path.is_absolute() else reference.resolve().parent / path

    materialization_path = resolve(attestation_path, attestation.materialization_manifest_artifact)
    raw_path = resolve(attestation_path, attestation.raw_artifact)
    task_path = resolve(attestation_path, attestation.task_provenance_artifact)
    task_manifest_path = resolve(attestation_path, attestation.task_provenance_manifest_artifact)
    attestation_normalization_path = resolve(
        attestation_path, attestation.normalization_manifest_artifact
    )
    attestation_normalized_path = resolve(attestation_path, attestation.normalized_artifact)
    findings_path = resolve(attestation_path, attestation.findings_artifact)
    try:
        materialization_payload, materialization_identity = _read_json_mapping_with_identity(
            materialization_path
        )
        task_manifest_payload, task_manifest_identity = _read_json_mapping_with_identity(
            task_manifest_path
        )
        materialization = SnapshotMaterializationManifestV2.model_validate(materialization_payload)
        task_manifest = TaskProvenanceProjectionManifestV1.model_validate(task_manifest_payload)
        raw_identity = file_identity(raw_path)
        task_identity = file_identity(task_path)
        findings_identity = file_identity(findings_path)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(f"invalid oracle-isolation evidence artifact: {error}") from error

    if (
        attestation_normalization_path.resolve() != normalization_manifest.resolve()
        or attestation_normalized_path.resolve() != normalized_input.resolve()
        or materialization_identity
        != (
            attestation.materialization_manifest_sha256,
            attestation.materialization_manifest_bytes,
        )
        or raw_identity != (attestation.raw_sha256, attestation.raw_bytes)
        or task_identity != (attestation.task_provenance_sha256, attestation.task_provenance_bytes)
        or task_manifest_identity
        != (
            attestation.task_provenance_manifest_sha256,
            attestation.task_provenance_manifest_bytes,
        )
        or findings_identity != (attestation.findings_sha256, attestation.findings_bytes)
        or not _oracle_attestation_matches_normalization(
            attestation,
            normalization=normalization,
            normalization_identity=normalization_identity,
            normalized_identity=normalized_identity,
            expected_sample_count=normalization.accepted_count,
        )
    ):
        raise typer.BadParameter("oracle-isolation attestation does not bind its evidence paths")

    if (
        materialization.status != VerificationStatus.PASS.value
        or materialization.materialization_scope != "complete-partition"
        or materialization.max_rows is not None
        or resolve(materialization_path, materialization.output).resolve() != raw_path.resolve()
        or (materialization.output_sha256, materialization.output_bytes) != raw_identity
        or materialization.row_count != attestation.raw_row_count
        or (normalization.input_sha256, normalization.input_bytes) != raw_identity
        or normalization.input_row_count != materialization.row_count
        or (
            normalization.materialization_manifest_sha256,
            normalization.materialization_manifest_bytes,
        )
        != materialization_identity
    ):
        raise typer.BadParameter("oracle-isolation raw materialization evidence is inconsistent")

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
        raise typer.BadParameter("oracle-isolation manifests describe different source leaves")

    if (
        task_manifest.status != VerificationStatus.PASS.value
        or task_manifest.projection_scope != "complete-snapshot"
        or task_manifest.file_patterns
        or task_manifest.safe_fields != TASK_PROVENANCE_SAFE_FIELDS
        or task_manifest.source_name != normalization.task_source_name
        or task_manifest.source_revision.casefold() != normalization.task_source_revision.casefold()
        or resolve(task_manifest_path, task_manifest.output).resolve() != task_path.resolve()
        or (task_manifest.output_sha256, task_manifest.output_bytes) != task_identity
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
        raise typer.BadParameter("oracle-isolation task provenance evidence is inconsistent")

    return (
        (raw_path, raw_identity),
        (materialization_path, materialization_identity),
        (task_path, task_identity),
        (task_manifest_path, task_manifest_identity),
        (normalization_manifest, normalization_identity),
        (normalized_input, normalized_identity),
        (findings_path, findings_identity),
    )


@datasets_app.command("audit-gold-exposure")
def datasets_audit_gold_exposure(
    input_path: Path = typer.Option(..., "--input", "--normalized", exists=True, dir_okay=False),
    normalization_manifest: Path = typer.Option(
        ..., "--normalization-manifest", exists=True, dir_okay=False
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    findings_output: Path | None = typer.Option(None, "--findings-output", dir_okay=False),
    oracle_isolation_attestation: Path | None = typer.Option(
        None,
        "--oracle-isolation-attestation",
        exists=True,
        dir_okay=False,
    ),
) -> None:
    """Audit a normalized population without serializing trajectory or gold content."""

    findings_path = findings_output or output.with_name(f"{output.stem}.findings.jsonl")
    staged_source_paths = [normalization_manifest, input_path]
    if oracle_isolation_attestation is not None:
        staged_source_paths.append(oracle_isolation_attestation)

    def same_existing_file(left: Path, right: Path) -> bool:
        return left.exists() and right.exists() and os.path.samefile(left, right)

    if output.resolve() == findings_path.resolve() or same_existing_file(output, findings_path):
        raise typer.BadParameter("--output and --findings-output must be distinct paths")
    if any(
        artifact_path.resolve() == source_path.resolve()
        or same_existing_file(artifact_path, source_path)
        for artifact_path in (output, findings_path)
        for source_path in staged_source_paths
    ):
        raise typer.BadParameter("audit output paths must not collide with staged inputs")
    if output.exists():
        raise typer.BadParameter("audit output already exists")

    normalization_payload, normalization_identity = _read_json_mapping_with_identity(
        normalization_manifest
    )
    try:
        normalization = NormalizationManifestV2.model_validate(normalization_payload)
    except ValidationError as error:
        raise typer.BadParameter("invalid normalization manifest evidence") from error

    input_identity = file_identity(input_path)
    if input_identity != (
        normalization.normalized_sha256,
        normalization.normalized_bytes,
    ):
        raise typer.BadParameter("normalization manifest does not bind --input")
    normalized_path = Path(normalization.normalized_artifact)
    if not normalized_path.is_absolute():
        normalized_path = normalization_manifest.resolve().parent / normalized_path
    if normalized_path.resolve() != input_path.resolve():
        raise typer.BadParameter("normalization manifest artifact path does not match --input")

    attestation: OracleIsolationAttestation | None = None
    attestation_identity: tuple[str, int] | None = None
    attestation_authorized = False
    attestation_evidence: tuple[tuple[Path, tuple[str, int]], ...] = ()
    if oracle_isolation_attestation is not None:
        attestation_payload, attestation_identity = _read_json_mapping_with_identity(
            oracle_isolation_attestation
        )
        try:
            attestation = OracleIsolationAttestation.model_validate(attestation_payload)
        except ValidationError as error:
            raise typer.BadParameter("invalid oracle-isolation attestation") from error
        if attestation.status is not VerificationStatus.PASS:
            raise typer.BadParameter("oracle-isolation attestation is not PASS")
        if not _oracle_attestation_matches_normalization(
            attestation,
            normalization=normalization,
            normalization_identity=normalization_identity,
            normalized_identity=input_identity,
            expected_sample_count=normalization.accepted_count,
        ):
            raise typer.BadParameter("oracle-isolation attestation coverage is inconsistent")
        try:
            require_authorized_oracle_attestation(
                normalized_sha256=input_identity[0],
                attestation_sha256=attestation_identity[0],
            )
        except OracleIsolationAuthorizationError:
            pass
        else:
            attestation_authorized = True
            attestation_evidence = _resolve_attestation_evidence_chain(
                attestation_path=oracle_isolation_attestation,
                attestation=attestation,
                normalization_manifest=normalization_manifest,
                normalization=normalization,
                normalization_identity=normalization_identity,
                normalized_input=input_path,
                normalized_identity=input_identity,
            )

    captured_inputs: list[tuple[Path, tuple[str, int]]] = [
        (normalization_manifest, normalization_identity),
        (input_path, input_identity),
    ]
    if oracle_isolation_attestation is not None and attestation_identity is not None:
        captured_inputs.append((oracle_isolation_attestation, attestation_identity))
    captured_inputs.extend(attestation_evidence)
    staged_inputs = list(
        {path.resolve(): (path, identity) for path, identity in captured_inputs}.values()
    )

    def verify_inputs() -> None:
        for path, identity in staged_inputs:
            _require_unchanged_file(path, identity)

    audited_sample_count = 0
    finding_count = 0
    findings_bytes = 0
    try:
        with verified_staged_files(tuple(staged_inputs)) as staged_files:
            staged_normalized = staged_files[1]

            def write_findings(stream: BinaryIO) -> None:
                nonlocal audited_sample_count, finding_count, findings_bytes

                with staged_normalized.open("rb") as normalized_stream:
                    for raw_row in normalized_stream:
                        if not raw_row.strip():
                            continue
                        row_index = audited_sample_count
                        audited_sample_count += 1
                        try:
                            value = json.loads(
                                raw_row,
                                object_pairs_hook=_unique_json_object,
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                            value = None
                        row = value if isinstance(value, dict) else None
                        if row is None:
                            finding = SanitizedGoldExposureFinding.from_reason_code(
                                row_index=row_index,
                                sample_id=None,
                                reason_code="invalid_normalized_sample",
                            )
                        else:
                            try:
                                sample = NormalizedSample.model_validate(row)
                            except ValidationError:
                                finding = SanitizedGoldExposureFinding.from_reason_code(
                                    row_index=row_index,
                                    sample_id=None,
                                    reason_code="invalid_normalized_sample",
                                )
                            else:
                                try:
                                    validate_gold_free_trajectory(sample.trajectory)
                                except NormalizationError:
                                    finding = SanitizedGoldExposureFinding.from_reason_code(
                                        row_index=row_index,
                                        sample_id=sample.sample_id,
                                        reason_code="forbidden_gold_reference_patch",
                                    )
                                else:
                                    continue
                        finding_bytes = canonical_json_bytes(finding.model_dump(mode="json"))
                        stream.write(finding_bytes)
                        findings_bytes += len(finding_bytes)
                        finding_count += 1

            findings_result = write_immutable_stream(
                findings_path,
                write_findings,
                before_publish=verify_inputs,
            )
            findings_identity = (findings_result.digest, findings_bytes)
    except VerifiedStagingError as error:
        raise typer.BadParameter(f"unable to audit verified normalized input: {error}") from error

    expected_sample_count = normalization.accepted_count
    count_mismatch = audited_sample_count != expected_sample_count
    structural_status = (
        VerificationStatus.FAIL if finding_count or count_mismatch else VerificationStatus.PASS
    )
    if (
        finding_count
        or count_mismatch
        or expected_sample_count == 0
        or normalization.status != VerificationStatus.PASS.value
    ):
        overall_status = VerificationStatus.FAIL
    elif (
        normalization.uniqueness_scope != "complete-partition"
        or attestation is None
        or not attestation_authorized
    ):
        overall_status = VerificationStatus.BLOCKED
    else:
        overall_status = VerificationStatus.PASS

    if attestation is None or attestation_identity is None or not attestation_authorized:
        oracle_evidence: dict[str, object] = {
            "status": VerificationStatus.BLOCKED.value,
            "attestation_artifact": None,
            "attestation_sha256": None,
            "attestation_bytes": None,
            "covered_sample_count": 0,
        }
    elif count_mismatch:
        oracle_evidence = {
            "status": VerificationStatus.FAIL.value,
            "attestation_artifact": None,
            "attestation_sha256": None,
            "attestation_bytes": None,
            "covered_sample_count": 0,
        }
    else:
        assert oracle_isolation_attestation is not None
        oracle_evidence = {
            "status": VerificationStatus.PASS.value,
            "attestation_artifact": os.path.relpath(
                oracle_isolation_attestation,
                start=output.resolve().parent,
            ),
            "attestation_sha256": attestation_identity[0],
            "attestation_bytes": attestation_identity[1],
            "covered_sample_count": attestation.covered_sample_count,
        }

    audit = GoldExposureAudit.model_validate(
        {
            "schema_version": "nodelm.gold-exposure-audit/v1",
            "method_version": "nodelm.gold-exposure-audit-method/v1",
            "status": overall_status.value,
            "normalization_manifest_artifact": os.path.relpath(
                normalization_manifest,
                start=output.resolve().parent,
            ),
            "normalization_manifest_sha256": normalization_identity[0],
            "normalization_manifest_bytes": normalization_identity[1],
            "normalized_artifact": os.path.relpath(
                input_path,
                start=output.resolve().parent,
            ),
            "normalized_sha256": input_identity[0],
            "normalized_bytes": input_identity[1],
            "expected_sample_count": expected_sample_count,
            "audited_sample_count": audited_sample_count,
            "structural_scan": {
                "status": structural_status.value,
                "finding_count": finding_count,
            },
            "oracle_isolation": oracle_evidence,
            "findings_artifact": os.path.relpath(
                findings_result.path,
                start=output.resolve().parent,
            ),
            "findings_sha256": findings_identity[0],
            "findings_bytes": findings_identity[1],
        }
    )

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_unchanged_file(findings_result.path, findings_identity)

    audit_result = write_immutable_json(
        output,
        audit.model_dump(mode="json"),
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        f"wrote {audit_result.path} status={overall_status.value} "
        f"audited={audited_sample_count} findings={finding_count} "
        f"sha256={audit_result.digest}"
    )
    if overall_status is not VerificationStatus.PASS:
        raise typer.Exit(code=1)


@datasets_app.command("build-pilot")
def datasets_build_pilot(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    samples_output: Path | None = typer.Option(None, "--samples-output", dir_okay=False),
    normalization_manifest: Path = typer.Option(
        ..., "--normalization-manifest", exists=True, dir_okay=False
    ),
    gold_exposure_audit: Path = typer.Option(
        ..., "--gold-exposure-audit", exists=True, dir_okay=False
    ),
    split_manifest: Path = typer.Option(..., "--split-manifest", exists=True, dir_okay=False),
    policy_config: Path = typer.Option(
        Path("configs/datasets/pilot.yaml"), "--policy", exists=True, dir_okay=False
    ),
    registry_config: Path = typer.Option(
        Path("configs/datasets/registry.yaml"), "--registry", exists=True, dir_okay=False
    ),
    max_samples: int | None = typer.Option(None, min=1),
    max_patch_bytes: int | None = typer.Option(None, min=1),
) -> None:
    input_identity = file_identity(input_path)
    normalization_payload, normalization_identity = _read_json_mapping_with_identity(
        normalization_manifest
    )
    audit_payload, audit_identity = _read_json_mapping_with_identity(gold_exposure_audit)
    policy_payload, policy_identity = _read_yaml_mapping_with_identity(policy_config)
    try:
        normalization = NormalizationManifestV2.model_validate(normalization_payload)
        pilot_policy = PilotPolicyConfig.model_validate(policy_payload)
        audit = GoldExposureAudit.model_validate(audit_payload)
    except ValidationError as error:
        raise typer.BadParameter(f"invalid pilot safety evidence: {error}") from error
    registry, registry_identity = _load_registry_with_identity(registry_config)
    split_identity = file_identity(split_manifest)
    try:
        require_authorized_repository_split(
            normalized_sha256=input_identity[0],
            split_sha256=split_identity[0],
        )
    except SplitAuthorizationError as error:
        raise typer.BadParameter(str(error)) from error

    if normalization.status != VerificationStatus.PASS.value:
        raise typer.BadParameter("pilot requires a replay-verified PASS normalization manifest")
    if normalization.uniqueness_scope != "complete-partition":
        raise typer.BadParameter("pilot requires complete-partition normalization evidence")
    if (
        normalization.normalized_sha256 != input_identity[0]
        or normalization.normalized_bytes != input_identity[1]
    ):
        raise typer.BadParameter("normalization manifest does not bind --input")
    normalized_path = Path(normalization.normalized_artifact)
    if not normalized_path.is_absolute():
        normalized_path = normalization_manifest.resolve().parent / normalized_path
    if normalized_path.resolve() != input_path.resolve():
        raise typer.BadParameter("normalization manifest artifact path does not match --input")
    normalized_count = normalization.accepted_count
    if normalized_count < 1:
        raise typer.BadParameter("normalization manifest has no admitted samples")

    if audit.status is not VerificationStatus.PASS:
        raise typer.BadParameter("gold-exposure audit is not PASS")
    try:
        require_authorized_gold_audit(
            normalized_sha256=input_identity[0],
            audit_sha256=audit_identity[0],
        )
    except GoldExposureAuthorizationError as error:
        raise typer.BadParameter(str(error)) from error
    if (
        audit.normalization_manifest_sha256 != normalization_identity[0]
        or audit.normalization_manifest_bytes != normalization_identity[1]
        or audit.normalized_sha256 != input_identity[0]
        or audit.normalized_bytes != input_identity[1]
        or audit.expected_sample_count != normalized_count
        or audit.audited_sample_count != normalized_count
    ):
        raise typer.BadParameter("gold-exposure audit does not bind the normalized population")
    audit_normalization_path = Path(audit.normalization_manifest_artifact)
    if not audit_normalization_path.is_absolute():
        audit_normalization_path = gold_exposure_audit.resolve().parent / audit_normalization_path
    audit_normalized_path = Path(audit.normalized_artifact)
    if not audit_normalized_path.is_absolute():
        audit_normalized_path = gold_exposure_audit.resolve().parent / audit_normalized_path
    if (
        audit_normalization_path.resolve() != normalization_manifest.resolve()
        or audit_normalized_path.resolve() != input_path.resolve()
    ):
        raise typer.BadParameter("gold-exposure audit artifact paths are inconsistent")

    findings_path = Path(audit.findings_artifact)
    if not findings_path.is_absolute():
        findings_path = gold_exposure_audit.resolve().parent / findings_path
    attestation_artifact = audit.oracle_isolation.attestation_artifact
    if attestation_artifact is None:  # pragma: no cover - PASS model invariant
        raise typer.BadParameter("PASS oracle isolation is missing its attestation")
    attestation_path = Path(attestation_artifact)
    if not attestation_path.is_absolute():
        attestation_path = gold_exposure_audit.resolve().parent / attestation_path
    verified_sources = {source.name: source for source in registry.sources}
    try:
        findings_identity = file_identity(findings_path)
        attestation_payload, attestation_identity = _read_json_mapping_with_identity(
            attestation_path
        )
        attestation = OracleIsolationAttestation.model_validate(attestation_payload)
    except (OSError, ValidationError) as error:
        raise typer.BadParameter(f"invalid gold-exposure evidence artifact: {error}") from error
    if attestation.status is not VerificationStatus.PASS:
        raise typer.BadParameter("oracle-isolation attestation is not PASS")
    if (
        findings_identity != (audit.findings_sha256, audit.findings_bytes)
        or findings_identity[1] != 0
    ):
        raise typer.BadParameter("PASS structural gold scan requires an empty bound findings file")
    oracle_findings_path = Path(attestation.findings_artifact)
    if not oracle_findings_path.is_absolute():
        oracle_findings_path = attestation_path.resolve().parent / oracle_findings_path
    try:
        oracle_findings_identity = file_identity(oracle_findings_path)
    except OSError as error:
        raise typer.BadParameter(f"invalid oracle-isolation findings artifact: {error}") from error
    if (
        attestation_identity
        != (
            audit.oracle_isolation.attestation_sha256,
            audit.oracle_isolation.attestation_bytes,
        )
        or oracle_findings_identity != (attestation.findings_sha256, attestation.findings_bytes)
        or not _oracle_attestation_matches_normalization(
            attestation,
            normalization=normalization,
            normalization_identity=normalization_identity,
            normalized_identity=input_identity,
            expected_sample_count=normalized_count,
        )
    ):
        raise typer.BadParameter("oracle-isolation attestation coverage is inconsistent")
    attestation_evidence = _resolve_attestation_evidence_chain(
        attestation_path=attestation_path,
        attestation=attestation,
        normalization_manifest=normalization_manifest,
        normalization=normalization,
        normalization_identity=normalization_identity,
        normalized_input=input_path,
        normalized_identity=input_identity,
    )

    captured_pilot_evidence = [
        (input_path, input_identity),
        (split_manifest, split_identity),
        (normalization_manifest, normalization_identity),
        (gold_exposure_audit, audit_identity),
        (findings_path, findings_identity),
        (attestation_path, attestation_identity),
        *attestation_evidence,
    ]
    staged_pilot_sources = list(
        {path.resolve(): (path, identity) for path, identity in captured_pilot_evidence}.values()
    )

    sample_count = 0
    try:
        with verified_staged_files(tuple(staged_pilot_sources)) as staged_pilot_files:
            split_evidence = read_repository_split_evidence(staged_pilot_files[1])
            if (
                split_evidence.input_sha256 != input_identity[0]
                or split_evidence.input_bytes != input_identity[1]
                or split_evidence.sample_count != normalized_count
            ):
                raise typer.BadParameter("split manifest does not bind the normalized population")

            def validated_samples() -> Iterator[NormalizedSample]:
                nonlocal sample_count
                for row in _read_jsonl(staged_pilot_files[0]):
                    sample = NormalizedSample.model_validate(row)
                    validate_gold_free_trajectory(sample.trajectory)
                    source = verified_sources.get(sample.source_dataset)
                    if source is None:
                        raise typer.BadParameter(
                            f"normalized sample references unknown source: {sample.source_dataset}"
                        )
                    if source.status is not VerificationStatus.PASS:
                        raise typer.BadParameter(
                            f"normalized sample source is not verified: {sample.source_dataset}"
                        )
                    if sample.source_dataset_revision != source.revision:
                        raise typer.BadParameter(
                            "normalized sample source revision does not match registry: "
                            f"{sample.source_dataset}"
                        )
                    sample_count += 1
                    yield sample

            subset = build_pilot_subset(
                validated_samples(),
                PilotFilter(
                    max_samples=max_samples or pilot_policy.target_samples,
                    languages=pilot_policy.languages,
                    require_resolved=pilot_policy.require_resolved,
                    require_nonempty_trajectory=pilot_policy.require_nonempty_trajectory,
                    max_patch_bytes=(
                        max_patch_bytes
                        if max_patch_bytes is not None
                        else pilot_policy.max_patch_bytes
                    ),
                    max_trajectory_steps=pilot_policy.max_trajectory_steps,
                    allowed_repository_licenses=pilot_policy.allowed_repository_licenses,
                    training_repositories=split_evidence.train,
                    excluded_repositories=(split_evidence.evaluation + split_evidence.excluded),
                ),
            )
    except (NormalizationError, ValidationError, ValueError) as error:
        raise typer.BadParameter(f"invalid normalized pilot input: {error}") from error
    if sample_count != normalized_count:
        raise typer.BadParameter("normalized sample count does not match safety evidence")

    def verify_inputs() -> None:
        _require_unchanged_file(policy_config, policy_identity)
        _require_unchanged_file(registry_config, registry_identity)
        for path, identity in staged_pilot_sources:
            _require_unchanged_file(path, identity)

    verify_inputs()
    samples_path = samples_output or output.with_name(f"{output.stem}.samples.jsonl")

    def write_samples(stream: BinaryIO) -> None:
        for sample in subset.accepted:
            stream.write(canonical_json_bytes(sample.model_dump(mode="json")))

    samples_result = write_immutable_stream(
        samples_path,
        write_samples,
        before_publish=verify_inputs,
    )
    samples_identity = _published_artifact_identity(samples_result.path, samples_result.digest)
    input_sha256, input_bytes = input_identity
    payload = {
        "schema_version": "nodelm.pilot-subset/v1",
        "status": (
            VerificationStatus.FAIL.value if not subset.accepted else pilot_policy.status.value
        ),
        "input_sha256": input_sha256,
        "input_bytes": input_bytes,
        "policy_sha256": policy_identity[0],
        "registry_sha256": registry_identity[0],
        "split_manifest_sha256": split_identity[0],
        "normalization_manifest_sha256": normalization_identity[0],
        "normalization_manifest_bytes": normalization_identity[1],
        "gold_exposure_audit": VerificationStatus.PASS.value,
        "gold_exposure_audit_sha256": audit_identity[0],
        "gold_exposure_audit_bytes": audit_identity[1],
        "filter_digest": subset.filter_digest,
        "accepted_count": len(subset.accepted),
        "rejection_reasons": subset.rejection_reasons,
        "samples_artifact": os.path.relpath(samples_result.path, start=output.resolve().parent),
        "samples_sha256": samples_result.digest,
        "samples_bytes": samples_identity[1],
    }

    def verify_completion_boundary() -> None:
        verify_inputs()
        _require_unchanged_file(samples_result.path, samples_identity)

    result = write_immutable_json(
        output,
        payload,
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        f"wrote {result.path} sha256={result.digest}; "
        f"samples={samples_result.path} sha256={samples_result.digest}"
    )
    if not subset.accepted:
        raise typer.Exit(code=1)


@split_app.command("build")
def split_build(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    task_metadata_path: Path = typer.Option(
        ...,
        "--task-metadata",
        exists=True,
        dir_okay=False,
        help="Task JSONL containing IDs, repositories, problem statements, and reference patches",
    ),
    benchmark_path: Path = typer.Option(
        ...,
        "--benchmark",
        exists=True,
        dir_okay=False,
        help="Public benchmark JSONL containing IDs, task text, and reference patches",
    ),
    near_duplicate_threshold: float = typer.Option(
        ...,
        "--near-duplicate-threshold",
        min=0.0,
        max=1.0,
        help="Explicit measured SequenceMatcher threshold in (0, 1]; no default is assumed",
    ),
    seed: int = typer.Option(42),
    evaluation_fraction: float = typer.Option(0.1, min=0.000001, max=0.999999),
    aliases_path: Path | None = typer.Option(None, "--aliases", exists=True, dir_okay=False),
) -> None:
    task_metadata_identity = file_identity(task_metadata_path)
    benchmark_identity = file_identity(benchmark_path)
    input_identity = file_identity(input_path)
    aliases_identity = file_identity(aliases_path) if aliases_path is not None else None

    def required_text(record: dict[str, Any], *fields: str) -> str:
        for field in fields:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(f"required field is missing: {' or '.join(fields)}")

    input_sha256, input_bytes = input_identity
    aliases_sha256 = aliases_identity[0] if aliases_identity is not None else None

    def verify_inputs() -> None:
        _require_unchanged_file(input_path, input_identity)
        _require_unchanged_file(task_metadata_path, task_metadata_identity)
        _require_unchanged_file(benchmark_path, benchmark_identity)
        if aliases_path is not None and aliases_identity is not None:
            _require_unchanged_file(aliases_path, aliases_identity)

    try:
        verify_inputs()
        staged_identities = [
            (input_path, input_identity),
            (task_metadata_path, task_metadata_identity),
            (benchmark_path, benchmark_identity),
        ]
        if aliases_path is not None and aliases_identity is not None:
            staged_identities.append((aliases_path, aliases_identity))
        with verified_staged_files(tuple(staged_identities)) as staged_split_inputs:
            staged_input = staged_split_inputs[0]
            staged_task_metadata = staged_split_inputs[1]
            staged_benchmark = staged_split_inputs[2]
            staged_aliases = (
                staged_split_inputs[3]
                if aliases_path is not None and aliases_identity is not None
                else None
            )

            def benchmarks() -> Iterator[BenchmarkEntry]:
                for record in _read_jsonl(staged_benchmark):
                    yield BenchmarkEntry(
                        benchmark_id=required_text(record, "benchmark_id", "instance_id", "id"),
                        task_text=required_text(
                            record,
                            "problem_statement",
                            "task",
                            "task_description",
                        ),
                        patch_text=required_text(record, "patch", "reference_patch"),
                    )

            def samples(task_lookup: TaskMetadataLookup) -> Iterator[ContaminationSample]:
                for record in _read_jsonl(staged_input):
                    sample = NormalizedSample.model_validate(record)
                    metadata = task_lookup(sample.issue_or_pr_id)
                    if metadata is None:
                        raise ValueError(
                            f"missing decontamination task metadata: {sample.issue_or_pr_id}"
                        )
                    if canonical_repository(sample.repository) != metadata.repository:
                        raise ValueError(
                            f"normalized sample/task repository mismatch: {sample.issue_or_pr_id}"
                        )
                    patches: tuple[str, ...] = (metadata.reference_patch,)
                    if sample.generated_patch is not None and sample.generated_patch.strip():
                        patches = (*patches, sample.generated_patch)
                    yield ContaminationSample(
                        sample_id=sample.sample_id,
                        repository=sample.repository,
                        task_text=metadata.task_text,
                        patch_texts=patches,
                    )

            with decontamination_task_metadata_index(
                _read_jsonl(staged_task_metadata),
                temp_directory=output.resolve().parent,
            ) as task_lookup:
                result = write_repository_split_manifest(
                    samples(task_lookup),
                    benchmarks=benchmarks(),
                    near_duplicate_threshold=near_duplicate_threshold,
                    task_metadata_sha256=task_metadata_identity[0],
                    benchmark_sha256=benchmark_identity[0],
                    output=output,
                    seed=seed,
                    evaluation_fraction=evaluation_fraction,
                    aliases=_read_repository_aliases(staged_aliases),
                    input_sha256=input_sha256,
                    input_bytes=input_bytes,
                    aliases_sha256=aliases_sha256,
                    before_publish=verify_inputs,
                )
    except ValueError as error:
        raise typer.BadParameter(f"invalid repository split input: {error}") from error
    typer.echo(f"wrote {result.path} sha256={result.digest}")


@harness_app.command("verify")
def harness_verify(
    workspace: Path = typer.Option(Path("tests/fixtures/ts-project"), exists=True),
    config: Path = typer.Option(Path("configs/harness/default.yaml"), exists=True, dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        harness_config = HarnessConfig.model_validate(_read_yaml_mapping(config))
    except ValidationError as error:
        raise typer.BadParameter(f"invalid harness configuration: {error}") from error
    discovery = discover_typescript_workspace(workspace)
    missing: list[str] = []
    if not discovery.package_manifests:
        missing.append("no package.json discovered")
    if not discovery.tsconfig_paths:
        missing.append("no tsconfig.json discovered")
    config_sha256 = content_digest(config.read_bytes())
    if missing:
        payload = {
            "schema_version": "nodelm.harness-verification/v1",
            "status": VerificationStatus.FAIL.value,
            "outcome": OutcomeCategory.TOOL_PROTOCOL_FAILURE.value,
            "reason": "; ".join(missing),
            "backend": harness_config.backend,
            "config_sha256": config_sha256,
            "package_count": len(discovery.package_manifests),
            "tsconfig_count": len(discovery.tsconfig_paths),
            "command": None,
        }
        typer.echo(_dump(payload) if json_output else f"FAIL: {payload['reason']}")
        raise typer.Exit(code=1)

    policy = CommandPolicy(workspace)
    command = replace(
        policy.node_test(),
        timeout_seconds=harness_config.timeout_seconds,
        max_output_bytes=harness_config.max_output_bytes,
    )
    result = CommandExecutor(workspace).run(command)
    test_count = parse_node_test_count(result.stdout)
    tests_executed = test_count is not None and test_count > 0
    succeeded = result.outcome is OutcomeCategory.SUCCESS and tests_executed
    payload = {
        "schema_version": "nodelm.harness-verification/v1",
        "status": (VerificationStatus.PASS.value if succeeded else VerificationStatus.FAIL.value),
        "outcome": result.outcome.value,
        "reason": (
            "trusted-local fixture tests completed"
            if succeeded
            else (
                "Node test command completed without evidence that any test executed"
                if result.outcome is OutcomeCategory.SUCCESS
                else "trusted-local fixture command failed"
            )
        ),
        "backend": harness_config.backend,
        "config_sha256": config_sha256,
        "package_count": len(discovery.package_manifests),
        "tsconfig_count": len(discovery.tsconfig_paths),
        "test_count": test_count,
        "command": result.to_evidence(),
    }
    typer.echo(_dump(payload) if json_output else f"{payload['status']}: {result.outcome.value}")
    if not succeeded:
        raise typer.Exit(code=1)


@infra_app.command("doctor")
def infrastructure_doctor(
    workspace: Path = typer.Option(Path("."), exists=True, file_okay=False),
    output: Path | None = typer.Option(None, "--output", dir_okay=False),
    json_output: bool = typer.Option(False, "--json"),
    require_gpu: bool = typer.Option(False, "--require-gpu"),
) -> None:
    report = collect_infrastructure_report(workspace=workspace)
    payload = report.model_dump(mode="json")
    if output is not None:
        result = write_immutable_json(output, payload)
        typer.echo(f"wrote {result.path} sha256={result.digest}")
    elif json_output:
        typer.echo(_dump(payload))
    else:
        typer.echo(f"{report.status.value}: {report.cpu_count} CPUs; GPU {report.gpu.status.value}")
    if report.status is VerificationStatus.FAIL or (
        require_gpu and report.status is not VerificationStatus.PASS
    ):
        raise typer.Exit(code=1)


@models_app.command("smoke")
def model_smoke(
    config: Path = typer.Option(Path("configs/evaluation/candidates.yaml"), exists=True),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        registry = CandidateRegistry.load(config)
    except CandidateRegistryError as error:
        raise typer.BadParameter(f"invalid candidate registry: {error}") from error
    payload = {
        "schema_version": "nodelm.model-smoke-command/v1",
        "status": registry.execution_status.value,
        "metadata_status": registry.metadata_status.value,
        "bakeoff_status": registry.bakeoff_status.value,
        "candidate_count": len(registry.candidates),
        "selected_candidate": registry.selected_candidate,
        "reason": registry.reason,
    }
    typer.echo(
        _dump(payload)
        if json_output
        else f"{registry.execution_status.value}: {len(registry.candidates)} candidates"
    )


@training_app.command("smoke")
def training_smoke(
    config: Path = typer.Option(Path("configs/training/tiny-lora.yaml"), exists=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        value = load_training_smoke_config(config)
    except (ValidationError, ValueError) as error:
        raise typer.BadParameter(f"invalid training configuration: {error}") from error
    pinned = value.model.repository_id is not None and value.model.revision is not None
    if dry_run:
        status = VerificationStatus.NOT_RUN
        reason = "configuration parsed; model load and training were deliberately not run"
    elif not pinned:
        status = VerificationStatus.BLOCKED
        reason = "student model remains unselected until the same-harness candidate bake-off passes"
    else:
        status = VerificationStatus.BLOCKED
        reason = "use the remote verification procedure after installing the training extra"
    payload = {"status": status.value, "config": str(config), "reason": reason}
    typer.echo(_dump(payload) if json_output else f"{status.value}: {reason}")
    if not dry_run and status is VerificationStatus.BLOCKED:
        raise typer.Exit(code=2)


@training_app.command("run-lifecycle")
def training_run_lifecycle(
    samples: Path = typer.Option(..., "--samples", exists=True, dir_okay=False),
    pilot_manifest: Path | None = typer.Option(
        None, "--pilot-manifest", exists=True, dir_okay=False
    ),
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", file_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    config: Path = typer.Option(
        Path("configs/training/tiny-lora.yaml"), "--config", exists=True, dir_okay=False
    ),
    evaluation_workspace: Path = typer.Option(
        Path("tests/fixtures/model-task"),
        "--evaluation-workspace",
        exists=True,
        file_okay=False,
    ),
    inference_prompt: str = typer.Option(
        "In src/math.js, multiply incorrectly adds its operands. Return only a git-style "
        "unified diff that repairs multiply without changing tests.",
        "--inference-prompt",
    ),
    sandbox_image: str | None = typer.Option(
        None,
        "--sandbox-image",
        help="Preloaded Node container image pinned as name@sha256:<digest>",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        config_bytes = config.read_bytes()
        config_identity = training_config_identity(config_bytes)
        value = parse_training_smoke_config(config_bytes)
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise typer.BadParameter(f"invalid training configuration: {error}") from error
    _require_unchanged_file(config, config_identity)
    if not value.is_runnable:
        blocked_payload = {
            "schema_version": "nodelm.training-lifecycle-command/v1",
            "status": VerificationStatus.BLOCKED.value,
            "reason": (
                "model, revision, license, runtime, precision, and every checkpoint lifecycle "
                "gate must be selected"
            ),
        }
        typer.echo(
            _dump(blocked_payload) if json_output else f"BLOCKED: {blocked_payload['reason']}"
        )
        raise typer.Exit(code=2)
    if value.max_steps != 1 or value.gradient_accumulation_steps != 1:
        raise typer.BadParameter(
            "run-lifecycle is a one-step verification command; max_steps and "
            "gradient_accumulation_steps must both equal 1"
        )
    if checkpoint_dir.exists():
        raise typer.BadParameter(f"checkpoint directory already exists: {checkpoint_dir}")
    if pilot_manifest is None:
        raise typer.BadParameter("a pilot manifest is required for lifecycle verification")
    if sandbox_image is None:
        blocked_payload = {
            "schema_version": "nodelm.training-lifecycle-command/v1",
            "status": VerificationStatus.BLOCKED.value,
            "reason": "a digest-pinned rootless Podman --sandbox-image is required",
        }
        typer.echo(
            _dump(blocked_payload) if json_output else f"BLOCKED: {blocked_payload['reason']}"
        )
        raise typer.Exit(code=2)
    try:
        sandbox = PodmanFixtureSandbox(sandbox_image)
    except ValueError as error:
        raise typer.BadParameter(f"invalid sandbox image: {error}") from error

    model_id = value.model.repository_id
    revision = value.model.revision
    model_license = value.model.license
    precision = value.precision
    runtime = value.runtime
    if (
        model_id is None
        or revision is None
        or model_license is None
        or precision is None
        or runtime is None
    ):
        raise AssertionError("runnable training configuration lost required fields")
    samples_identity = file_identity(samples)
    pilot_identity = file_identity(pilot_manifest)
    try:
        with verified_staged_files(
            ((samples, samples_identity), (pilot_manifest, pilot_identity))
        ) as staged_training_inputs:
            _validate_pilot_manifest(
                staged_training_inputs[1],
                samples_sha256=samples_identity[0],
                samples_bytes=samples_identity[1],
                pilot_manifest_sha256=pilot_identity[0],
                artifact_base=pilot_manifest.resolve().parent,
                expected_samples_path=samples,
                required_samples=value.batch_size,
            )
            settings = TransformersSmokeSettings(
                model_id=model_id,
                revision=revision,
                device=runtime.device,
                dtype=precision,
                padding_policy=runtime.padding_policy,
                added_padding_token=runtime.added_padding_token,
                max_length=runtime.max_length,
                max_new_tokens=runtime.max_new_tokens,
                use_lora=runtime.use_lora,
                target_modules=runtime.target_modules,
                lora_rank=runtime.lora_rank,
                lora_alpha=runtime.lora_alpha,
                lora_dropout=runtime.lora_dropout,
            )
            training_samples = take_training_texts(
                (
                    NormalizedSample.model_validate(row)
                    for row in _read_jsonl(staged_training_inputs[0])
                ),
                count=value.batch_size,
            )
    except (ValidationError, ValueError, VerifiedStagingError) as error:
        raise typer.BadParameter(f"training lifecycle inputs are invalid: {error}") from error

    try:
        verify_regular_file_tree(evaluation_workspace, MODEL_TASK_FIXTURE_IDENTITY)
    except VerifiedStagingError as error:
        raise typer.BadParameter(f"invalid evaluation fixture: {error}") from error
    backend = TransformersSmokeBackend(settings)
    report = run_training_lifecycle(
        backend,
        TrainingLifecycleConfig(
            model_id=model_id,
            revision=revision,
            output_dir=checkpoint_dir,
            seed=value.seed,
            learning_rate=value.learning_rate,
        ),
        samples=training_samples,
        prompt=inference_prompt,
    )
    measurements = backend.measured_evidence()
    resume_evidence_passes = (
        report.resumed_optimizer_step_completed
        and measurements.get("optimizer_steps") == 2
        and measurements.get("resumed_optimizer_steps") == 1
        and measurements.get("optimizer_state_reloaded") is True
    )

    def verify_completion_boundary() -> None:
        _require_unchanged_file(config, config_identity)
        _require_unchanged_file(samples, samples_identity)
        _require_unchanged_file(pilot_manifest, pilot_identity)
        try:
            verify_regular_file_tree(evaluation_workspace, MODEL_TASK_FIXTURE_IDENTITY)
        except VerifiedStagingError as error:
            raise typer.BadParameter(f"invalid evaluation fixture: {error}") from error

    verify_completion_boundary()
    if (
        report.status is VerificationStatus.PASS
        and resume_evidence_passes
        and report.inference_output is not None
    ):
        try:
            with verified_staged_regular_file_tree(
                evaluation_workspace,
                MODEL_TASK_FIXTURE_IDENTITY,
            ) as staged_evaluation_workspace:
                evaluation = evaluate_model_patch_fixture(
                    report.inference_output,
                    fixture=staged_evaluation_workspace,
                    exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
                    sandbox=sandbox,
                )
        except VerifiedStagingError as error:
            raise typer.BadParameter(f"invalid evaluation fixture: {error}") from error
    else:
        evaluation = FixturePatchReport(
            status=VerificationStatus.NOT_RUN,
            reason="model lifecycle did not produce verified resumed inference output",
        )
    if report.status is not VerificationStatus.PASS or not resume_evidence_passes:
        overall_status = VerificationStatus.FAIL
    else:
        overall_status = evaluation.status
    payload: dict[str, Any] = {
        "schema_version": "nodelm.training-lifecycle-command/v1",
        "status": overall_status.value,
        "config": str(config),
        "config_sha256": config_identity[0],
        "config_status": value.status.value,
        "samples": str(samples),
        "samples_sha256": samples_identity[0],
        "pilot_manifest": str(pilot_manifest),
        "pilot_manifest_sha256": pilot_identity[0],
        "model_id": model_id,
        "model_revision": revision,
        "model_license": model_license,
        "backend": runtime.backend,
        "device": runtime.device,
        "precision": precision,
        "checkpoint_save_required": value.checkpoint.save,
        "checkpoint_reload_required": value.checkpoint.reload,
        "checkpoint_resume_required": value.checkpoint.resume,
        "inference_after_reload_required": value.inference_after_reload,
        "sandbox_image": sandbox_image,
        "evaluation_fixture": str(evaluation_workspace),
        "evaluation_fixture_identity_schema": MODEL_TASK_FIXTURE_IDENTITY.schema_version,
        "evaluation_fixture_sha256": MODEL_TASK_FIXTURE_IDENTITY.tree_sha256,
        "evaluation_fixture_file_count": MODEL_TASK_FIXTURE_IDENTITY.file_count,
        "evaluation_fixture_bytes": MODEL_TASK_FIXTURE_IDENTITY.tree_bytes,
        "lifecycle": report.model_dump(mode="json"),
        "measurements": measurements,
        "resume_evidence_status": (
            VerificationStatus.PASS.value
            if resume_evidence_passes
            else VerificationStatus.FAIL.value
        ),
        "fixture_evaluation": evaluation.model_dump(mode="json"),
    }
    result = write_immutable_json(
        output,
        payload,
        before_publish=verify_completion_boundary,
    )
    typer.echo(
        _dump({**payload, "report_artifact": str(result.path), "report_sha256": result.digest})
        if json_output
        else f"{overall_status.value}: lifecycle report {result.path} sha256={result.digest}"
    )
    if overall_status is VerificationStatus.BLOCKED:
        raise typer.Exit(code=2)
    if overall_status is not VerificationStatus.PASS:
        raise typer.Exit(code=1)
