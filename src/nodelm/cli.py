from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO

import typer
import yaml
from pydantic import ValidationError

from nodelm.artifacts import (
    ArtifactCollisionError,
    canonical_json_bytes,
    content_digest,
    write_immutable_json,
    write_immutable_stream,
)
from nodelm.datasets.audit import audit_rows, iter_license_rejections
from nodelm.datasets.hub import download_pinned_snapshot, verify_hub_source
from nodelm.datasets.lineage import (
    build_dataset_lineage_manifest,
    capture_snapshot_identity,
    verify_snapshot_identity,
)
from nodelm.datasets.materialize import discover_snapshot_files, iter_snapshot_rows
from nodelm.datasets.pilot import PilotFilter, PilotPolicyConfig, build_pilot_subset
from nodelm.datasets.registry import DatasetRegistry
from nodelm.decontamination.contamination import (
    BenchmarkEntry,
    ContaminationSample,
    TaskMetadataLookup,
    decontamination_task_metadata_index,
)
from nodelm.decontamination.fingerprints import canonical_repository
from nodelm.decontamination.split import (
    read_repository_split_repositories,
    write_repository_split_manifest,
)
from nodelm.doctor import build_doctor_report
from nodelm.evaluation.fixture import (
    MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
    FixturePatchReport,
    evaluate_model_patch_fixture,
)
from nodelm.evaluation.registry import CandidateRegistry, CandidateRegistryError
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
from nodelm.models import NormalizedSample, VerificationStatus
from nodelm.provenance.normalize import NormalizationError
from nodelm.provenance.pipeline import normalize_trace_sample, task_metadata_index
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
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise typer.BadParameter(f"unable to read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise typer.BadParameter(f"configuration root must be a mapping: {path}")
    return value


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


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _require_unchanged_file(path: Path, expected: tuple[str, int]) -> None:
    if _file_identity(path) != expected:
        raise typer.BadParameter(f"input changed while it was being processed: {path}")


def _split_repositories(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return read_repository_split_repositories(path)
    except ValueError as error:
        raise typer.BadParameter(f"invalid repository split manifest: {error}") from error


def _validate_pilot_manifest(
    path: Path,
    *,
    samples: Path,
    samples_sha256: str,
    required_samples: int,
) -> None:
    value = _read_yaml_mapping(path)
    if value.get("schema_version") != "nodelm.pilot-subset/v1":
        raise typer.BadParameter("unsupported pilot manifest schema_version")
    if value.get("status") not in {
        VerificationStatus.PASS.value,
        VerificationStatus.UNVERIFIED.value,
    }:
        raise typer.BadParameter("pilot manifest is not eligible for lifecycle verification")
    if value.get("samples_sha256") != samples_sha256:
        raise typer.BadParameter("pilot manifest sample digest does not match --samples")
    accepted_count = value.get("accepted_count")
    if not isinstance(accepted_count, int) or accepted_count < required_samples:
        raise typer.BadParameter("pilot manifest does not contain the required training batch")
    artifact = value.get("samples_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise typer.BadParameter("pilot manifest is missing its samples_artifact")
    artifact_path = Path(artifact)
    if not artifact_path.is_absolute():
        artifact_path = path.resolve().parent / artifact_path
    if artifact_path.resolve() != samples.resolve():
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
    input_identity = _file_identity(input_path)
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
    output: Path = typer.Option(..., "--output", dir_okay=False),
    lineage_output: Path | None = typer.Option(None, "--lineage-output", dir_okay=False),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    config: Path = typer.Option(
        Path("configs/datasets/registry.yaml"), exists=True, dir_okay=False
    ),
) -> None:
    """Audit every supported file in a local pinned snapshot without network access."""

    lineage_path = lineage_output or output.with_name(f"{output.stem}.lineage.json")
    ledger_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    artifact_paths = tuple(path.resolve() for path in (output, ledger_path, lineage_path))
    if len(set(artifact_paths)) != len(artifact_paths):
        raise typer.BadParameter("audit, rejection ledger, and lineage outputs must be distinct")

    resolved_snapshot = snapshot.resolve()
    if resolved_snapshot.is_dir():
        if any(path.is_relative_to(resolved_snapshot) for path in artifact_paths):
            raise typer.BadParameter("all audit outputs must be outside a directory snapshot")
    elif resolved_snapshot in artifact_paths:
        raise typer.BadParameter("audit outputs must be distinct from the snapshot input")
    if config.resolve() in artifact_paths:
        raise typer.BadParameter("audit outputs must be distinct from the registry input")

    try:
        registry_identity = _file_identity(config)
        registry = DatasetRegistry.load(config)
        source = registry.by_name(source_name)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"invalid dataset registry: {error}") from error
    try:
        _require_unchanged_file(config, registry_identity)
    except OSError as error:
        raise typer.BadParameter(f"dataset registry changed while loading: {error}") from error
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("snapshot audit requires a registry-verified pinned source")

    try:
        snapshot_identity = capture_snapshot_identity(snapshot)
        snapshot_files = discover_snapshot_files(snapshot)
    except (OSError, ValueError, ValidationError) as error:
        raise typer.BadParameter(f"unable to capture snapshot identity: {error}") from error

    def verify_inputs() -> None:
        try:
            _require_unchanged_file(config, registry_identity)
            verify_snapshot_identity(snapshot, snapshot_identity)
        except (OSError, ValueError, ValidationError) as error:
            raise typer.BadParameter(str(error)) from error

    verify_inputs()
    try:
        report = audit_rows(
            source,
            iter_snapshot_rows(snapshot_files),
            input_sha256=snapshot_identity.snapshot_sha256,
            input_bytes=snapshot_identity.snapshot_bytes,
            expect_complete_snapshot=True,
        )
    except (OSError, ValueError, NormalizationError) as error:
        raise typer.BadParameter(f"snapshot audit failed: {error}") from error

    ledger_row_count = 0
    ledger_byte_count = 0

    def write_rejection_ledger(stream: BinaryIO) -> None:
        nonlocal ledger_byte_count, ledger_row_count
        for rejection in iter_license_rejections(iter_snapshot_rows(snapshot_files)):
            encoded = canonical_json_bytes(rejection)
            stream.write(encoded)
            ledger_row_count += 1
            ledger_byte_count += len(encoded)

    try:
        ledger_result = write_immutable_stream(
            ledger_path,
            write_rejection_ledger,
            before_publish=verify_inputs,
        )
    except (ArtifactCollisionError, OSError, ValueError) as error:
        raise typer.BadParameter(f"snapshot audit publication failed: {error}") from error
    ledger_identity = (ledger_result.digest, ledger_byte_count)

    report_ledger_artifact = os.path.relpath(
        ledger_result.path,
        start=output.resolve().parent,
    )
    report = type(report).model_validate(
        {
            **report.model_dump(mode="json"),
            "rejection_ledger_artifact": report_ledger_artifact,
            "rejection_ledger_sha256": ledger_result.digest,
            "rejection_ledger_rows": ledger_row_count,
        }
    )
    report_bytes = canonical_json_bytes(report.model_dump(mode="json"))

    def verify_report_inputs() -> None:
        verify_inputs()
        _require_unchanged_file(ledger_result.path, ledger_identity)

    try:
        report_result = write_immutable_stream(
            output,
            lambda stream: stream.write(report_bytes),
            before_publish=verify_report_inputs,
        )
    except (ArtifactCollisionError, OSError, ValueError) as error:
        raise typer.BadParameter(f"snapshot audit publication failed: {error}") from error
    report_identity = (report_result.digest, len(report_bytes))

    def verify_lineage_inputs() -> None:
        verify_report_inputs()
        _require_unchanged_file(report_result.path, report_identity)

    verify_lineage_inputs()
    try:
        manifest = build_dataset_lineage_manifest(
            source=source,
            registry_sha256=registry_identity[0],
            registry_bytes=registry_identity[1],
            snapshot=snapshot_identity,
            report=report,
            audit_artifact=os.path.relpath(
                report_result.path,
                start=lineage_path.resolve().parent,
            ),
            audit_sha256=report_result.digest,
            rejection_ledger_artifact=os.path.relpath(
                ledger_result.path,
                start=lineage_path.resolve().parent,
            ),
            rejection_ledger_sha256=ledger_result.digest,
            rejection_ledger_rows=ledger_row_count,
        )
        lineage_result = write_immutable_stream(
            lineage_path,
            lambda stream: stream.write(canonical_json_bytes(manifest.model_dump(mode="json"))),
            before_publish=verify_lineage_inputs,
        )
    except (ArtifactCollisionError, OSError, ValueError, ValidationError) as error:
        raise typer.BadParameter(f"snapshot lineage publication failed: {error}") from error

    typer.echo(
        f"wrote {report_result.path} rows={report.row_count} sha256={report_result.digest}; "
        f"lineage={lineage_result.path}"
    )
    if report.status is VerificationStatus.FAIL:
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
    confirm_large_download: bool = typer.Option(False, "--confirm-large-download"),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    if not confirm_large_download:
        raise typer.BadParameter(
            "dataset snapshots may be very large; pass --confirm-large-download"
        )
    if destination.exists() and any(destination.iterdir()):
        raise typer.BadParameter(f"download destination must be new or empty: {destination}")
    source = DatasetRegistry.load(config).by_name(source_name)
    path = download_pinned_snapshot(
        source, destination=destination, allow_patterns=tuple(allow_pattern or ())
    )
    typer.echo(f"downloaded pinned snapshot to {path}")


@datasets_app.command("materialize")
def datasets_materialize(
    source_name: str = typer.Option(..., "--source"),
    snapshot: Path = typer.Option(..., "--snapshot", exists=True),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    file_pattern: list[str] | None = typer.Option(None, "--file-pattern"),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
    manifest_output: Path | None = typer.Option(None, "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    source = DatasetRegistry.load(config).by_name(source_name)
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("materialization requires a registry-verified pinned source")
    config_identity = _file_identity(config)
    patterns = tuple(file_pattern or ())
    try:
        files = discover_snapshot_files(snapshot, patterns=patterns)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    identities = tuple((path, _file_identity(path)) for path in files)
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

    def write_rows(stream: BinaryIO) -> None:
        nonlocal row_count
        for row in iter_snapshot_rows(files):
            if max_rows is not None and row_count >= max_rows:
                break
            stream.write(canonical_json_bytes(row))
            row_count += 1

    try:
        result = write_immutable_stream(output, write_rows, before_publish=verify_inputs)
    except ValueError as error:
        raise typer.BadParameter(f"snapshot materialization failed: {error}") from error
    manifest_path = manifest_output or output.with_name(f"{output.stem}.manifest.json")
    snapshot_root = snapshot.resolve() if snapshot.is_dir() else snapshot.resolve().parent
    manifest = {
        "schema_version": "nodelm.snapshot-materialization/v1",
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
                "path": os.path.relpath(path, start=snapshot_root),
                "sha256": identity[0],
                "bytes": identity[1],
            }
            for path, identity in identities
        ],
        "output": os.path.relpath(result.path, start=manifest_path.resolve().parent),
        "output_sha256": result.digest,
    }
    manifest_result = write_immutable_json(manifest_path, manifest)
    typer.echo(
        f"wrote {result.path} rows={row_count} sha256={result.digest}; "
        f"manifest={manifest_result.path}"
    )
    if row_count == 0:
        raise typer.Exit(code=1)


@datasets_app.command("normalize")
def datasets_normalize(
    source_name: str = typer.Option(..., "--source"),
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    harness: str = typer.Option(..., "--harness"),
    generating_model: str = typer.Option(..., "--generating-model"),
    task_metadata: Path | None = typer.Option(None, "--task-metadata", exists=True, dir_okay=False),
    rejections_output: Path | None = typer.Option(None, "--rejections-output", dir_okay=False),
    manifest_output: Path | None = typer.Option(None, "--manifest-output", dir_okay=False),
    config: Path = typer.Option(Path("configs/datasets/registry.yaml"), exists=True),
) -> None:
    source = DatasetRegistry.load(config).by_name(source_name)
    if source.status is not VerificationStatus.PASS or source.revision is None:
        raise typer.BadParameter("normalization requires a registry-verified pinned source")
    input_identity = _file_identity(input_path)
    task_identity = _file_identity(task_metadata) if task_metadata is not None else None
    config_identity = _file_identity(config)
    rejection_path = rejections_output or output.with_name(f"{output.stem}.rejections.jsonl")
    accepted_count = 0
    rejected_count = 0

    def verify_inputs() -> None:
        _require_unchanged_file(input_path, input_identity)
        _require_unchanged_file(config, config_identity)
        if task_metadata is not None and task_identity is not None:
            _require_unchanged_file(task_metadata, task_identity)

    metadata_rows = _read_jsonl(task_metadata) if task_metadata is not None else ()
    try:
        with task_metadata_index(metadata_rows) as lookup:

            def write_normalized(stream: BinaryIO) -> None:
                nonlocal accepted_count
                for row in _read_jsonl(input_path):
                    try:
                        sample = normalize_trace_sample(
                            row,
                            source=source,
                            harness=harness,
                            generating_model=generating_model,
                            task_lookup=lookup if task_metadata is not None else None,
                        )
                    except NormalizationError:
                        continue
                    stream.write(canonical_json_bytes(sample.model_dump(mode="json")))
                    accepted_count += 1

            normalized_result = write_immutable_stream(
                output,
                write_normalized,
                before_publish=verify_inputs,
            )

            def write_rejections(stream: BinaryIO) -> None:
                nonlocal rejected_count
                for row_index, row in enumerate(_read_jsonl(input_path)):
                    try:
                        normalize_trace_sample(
                            row,
                            source=source,
                            harness=harness,
                            generating_model=generating_model,
                            task_lookup=lookup if task_metadata is not None else None,
                        )
                    except NormalizationError as error:
                        stream.write(
                            canonical_json_bytes(
                                {
                                    "row_index": row_index,
                                    "instance_id": row.get("instance_id"),
                                    "repository": row.get("repo") or row.get("repository"),
                                    "reason": str(error),
                                }
                            )
                        )
                        rejected_count += 1

            rejection_result = write_immutable_stream(
                rejection_path,
                write_rejections,
                before_publish=verify_inputs,
            )
    except NormalizationError as error:
        raise typer.BadParameter(f"invalid task metadata: {error}") from error

    verify_inputs()
    manifest_path = manifest_output or output.with_name(f"{output.stem}.manifest.json")
    manifest = {
        "schema_version": "nodelm.normalization-manifest/v1",
        "status": (
            VerificationStatus.PASS.value if accepted_count else VerificationStatus.FAIL.value
        ),
        "source_name": source.name,
        "source_repository_id": source.repository_id,
        "source_revision": source.revision,
        "harness": harness,
        "generating_model": generating_model,
        "input_sha256": input_identity[0],
        "input_bytes": input_identity[1],
        "registry_sha256": config_identity[0],
        "task_metadata_sha256": task_identity[0] if task_identity is not None else None,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "normalized_artifact": os.path.relpath(
            normalized_result.path, start=manifest_path.resolve().parent
        ),
        "normalized_sha256": normalized_result.digest,
        "rejection_artifact": os.path.relpath(
            rejection_result.path, start=manifest_path.resolve().parent
        ),
        "rejection_sha256": rejection_result.digest,
    }
    manifest_result = write_immutable_json(manifest_path, manifest)
    typer.echo(
        f"wrote {normalized_result.path} accepted={accepted_count} rejected={rejected_count} "
        f"sha256={normalized_result.digest}; manifest={manifest_result.path}"
    )
    if accepted_count == 0:
        raise typer.Exit(code=1)


@datasets_app.command("build-pilot")
def datasets_build_pilot(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    samples_output: Path | None = typer.Option(None, "--samples-output", dir_okay=False),
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
    policy_identity = _file_identity(policy_config)
    registry_identity = _file_identity(registry_config)
    split_identity = _file_identity(split_manifest)
    try:
        pilot_policy = PilotPolicyConfig.model_validate(_read_yaml_mapping(policy_config))
    except ValidationError as error:
        raise typer.BadParameter(f"invalid pilot policy: {error}") from error
    registry = DatasetRegistry.load(registry_config)
    verified_sources = {source.name: source for source in registry.sources}
    training_repositories, evaluation_repositories = _split_repositories(split_manifest)
    input_identity = _file_identity(input_path)

    def validated_samples() -> Iterator[NormalizedSample]:
        for row in _read_jsonl(input_path):
            sample = NormalizedSample.model_validate(row)
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
                    f"normalized sample source revision does not match registry: "
                    f"{sample.source_dataset}"
                )
            yield sample

    try:
        subset = build_pilot_subset(
            validated_samples(),
            PilotFilter(
                max_samples=max_samples or pilot_policy.target_samples,
                languages=pilot_policy.languages,
                require_resolved=pilot_policy.require_resolved,
                require_nonempty_trajectory=pilot_policy.require_nonempty_trajectory,
                max_patch_bytes=(
                    max_patch_bytes if max_patch_bytes is not None else pilot_policy.max_patch_bytes
                ),
                max_trajectory_steps=pilot_policy.max_trajectory_steps,
                allowed_repository_licenses=pilot_policy.allowed_repository_licenses,
                training_repositories=training_repositories,
                excluded_repositories=evaluation_repositories,
            ),
        )
    except ValidationError as error:
        raise typer.BadParameter(f"invalid normalized sample: {error}") from error

    def verify_inputs() -> None:
        _require_unchanged_file(input_path, input_identity)
        _require_unchanged_file(policy_config, policy_identity)
        _require_unchanged_file(registry_config, registry_identity)
        _require_unchanged_file(split_manifest, split_identity)

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
        "filter_digest": subset.filter_digest,
        "accepted_count": len(subset.accepted),
        "rejection_reasons": subset.rejection_reasons,
        "samples_artifact": os.path.relpath(samples_result.path, start=output.resolve().parent),
        "samples_sha256": samples_result.digest,
    }
    verify_inputs()
    result = write_immutable_json(output, payload)
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
    task_metadata_identity = _file_identity(task_metadata_path)
    benchmark_identity = _file_identity(benchmark_path)
    input_identity = _file_identity(input_path)
    aliases_identity = _file_identity(aliases_path) if aliases_path is not None else None

    def required_text(record: dict[str, Any], *fields: str) -> str:
        for field in fields:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ValueError(f"required field is missing: {' or '.join(fields)}")

    def benchmarks() -> Iterator[BenchmarkEntry]:
        for record in _read_jsonl(benchmark_path):
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
        for record in _read_jsonl(input_path):
            sample = NormalizedSample.model_validate(record)
            metadata = task_lookup(sample.issue_or_pr_id)
            if metadata is None:
                raise ValueError(f"missing decontamination task metadata: {sample.issue_or_pr_id}")
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
        with decontamination_task_metadata_index(
            _read_jsonl(task_metadata_path),
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
                aliases=_read_repository_aliases(aliases_path),
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
    samples_identity = _file_identity(samples)
    pilot_identity = _file_identity(pilot_manifest)
    _validate_pilot_manifest(
        pilot_manifest,
        samples=samples,
        samples_sha256=samples_identity[0],
        required_samples=value.batch_size,
    )
    try:
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
            (NormalizedSample.model_validate(row) for row in _read_jsonl(samples)),
            count=value.batch_size,
        )
    except (ValidationError, ValueError) as error:
        raise typer.BadParameter(f"training lifecycle inputs are invalid: {error}") from error

    evaluation_files = tuple(
        sorted(path.resolve() for path in evaluation_workspace.rglob("*") if path.is_file())
    )
    evaluation_identities = tuple((path, _file_identity(path)) for path in evaluation_files)
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
    _require_unchanged_file(config, config_identity)
    _require_unchanged_file(samples, samples_identity)
    _require_unchanged_file(pilot_manifest, pilot_identity)
    current_evaluation_files = tuple(
        sorted(path.resolve() for path in evaluation_workspace.rglob("*") if path.is_file())
    )
    if current_evaluation_files != evaluation_files:
        raise typer.BadParameter("evaluation fixture file set changed during verification")
    for path, identity in evaluation_identities:
        _require_unchanged_file(path, identity)
    if (
        report.status is VerificationStatus.PASS
        and resume_evidence_passes
        and report.inference_output is not None
    ):
        evaluation = evaluate_model_patch_fixture(
            report.inference_output,
            fixture=evaluation_workspace,
            exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
            sandbox=sandbox,
        )
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
        "lifecycle": report.model_dump(mode="json"),
        "measurements": measurements,
        "resume_evidence_status": (
            VerificationStatus.PASS.value
            if resume_evidence_passes
            else VerificationStatus.FAIL.value
        ),
        "fixture_evaluation": evaluation.model_dump(mode="json"),
    }
    result = write_immutable_json(output, payload)
    typer.echo(
        _dump({**payload, "report_artifact": str(result.path), "report_sha256": result.digest})
        if json_output
        else f"{overall_status.value}: lifecycle report {result.path} sha256={result.digest}"
    )
    if overall_status is VerificationStatus.BLOCKED:
        raise typer.Exit(code=2)
    if overall_status is not VerificationStatus.PASS:
        raise typer.Exit(code=1)
