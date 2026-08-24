from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, ConfigDict

from nodelm.models import DatasetSource, VerificationStatus


class HubMetadataComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str
    status: VerificationStatus
    expected_revision: str | None
    observed_revision: str | None
    expected_license: str | None
    observed_license: str | None
    issues: tuple[str, ...]


def compare_hub_metadata(
    source: DatasetSource, *, sha: str | None, dataset_license: str | None
) -> HubMetadataComparison:
    issues: list[str] = []
    if not source.revision:
        issues.append("registry revision is not pinned")
    elif sha != source.revision:
        issues.append(f"revision drift: registry={source.revision} hub={sha}")
    if not source.dataset_license:
        issues.append("registry dataset license is missing")
    elif (dataset_license or "").casefold() != source.dataset_license.casefold():
        issues.append(
            f"dataset license drift: registry={source.dataset_license} hub={dataset_license}"
        )
    return HubMetadataComparison(
        source_name=source.name,
        status=VerificationStatus.FAIL if issues else VerificationStatus.PASS,
        expected_revision=source.revision,
        observed_revision=sha,
        expected_license=source.dataset_license,
        observed_license=dataset_license,
        issues=tuple(issues),
    )


def verify_hub_source(source: DatasetSource, *, api: HfApi | None = None) -> HubMetadataComparison:
    client = api or HfApi()
    info = client.dataset_info(repo_id=source.repository_id, revision=source.revision)
    card_data = info.card_data
    observed_license: str | None = None
    if card_data is not None:
        license_value = card_data.get("license")
        if isinstance(license_value, str):
            observed_license = license_value
    return compare_hub_metadata(source, sha=info.sha, dataset_license=observed_license)


def download_pinned_snapshot(
    source: DatasetSource,
    *,
    destination: Path,
    allow_patterns: tuple[str, ...] = (),
) -> Path:
    if not source.revision:
        raise ValueError(f"dataset source {source.name} has no pinned revision")
    resolved = snapshot_download(
        repo_id=source.repository_id,
        repo_type="dataset",
        revision=source.revision,
        local_dir=destination,
        allow_patterns=list(allow_patterns) or None,
    )
    return Path(resolved).resolve()
