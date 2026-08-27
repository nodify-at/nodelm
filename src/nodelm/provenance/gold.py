from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from nodelm.models import VerificationStatus


class GoldExposureAuthorizationError(ValueError):
    """A gold-exposure PASS artifact is not part of the reviewed trust root."""


class OracleIsolationAuthorizationError(ValueError):
    """An oracle-isolation PASS artifact is not part of the reviewed trust root."""


GoldExposureFindingReasonCode: TypeAlias = Literal[
    "invalid_normalized_sample",
    "forbidden_gold_reference_patch",
]
GoldExposureFindingReason: TypeAlias = Literal[
    "normalized row does not satisfy the sample schema",
    "trajectory contains forbidden gold/reference patch metadata",
]

_SAFE_FINDING_REASON_BY_CODE: dict[
    GoldExposureFindingReasonCode,
    GoldExposureFindingReason,
] = {
    "invalid_normalized_sample": "normalized row does not satisfy the sample schema",
    "forbidden_gold_reference_patch": (
        "trajectory contains forbidden gold/reference patch metadata"
    ),
}


class SanitizedGoldExposureFinding(BaseModel):
    """A finding that cannot serialize model-visible trajectory or gold content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[StrictInt, Field(ge=0)]
    sample_id: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    reason_code: GoldExposureFindingReasonCode
    reason: GoldExposureFindingReason

    @classmethod
    def from_reason_code(
        cls,
        *,
        row_index: int,
        sample_id: str | None,
        reason_code: GoldExposureFindingReasonCode,
    ) -> SanitizedGoldExposureFinding:
        return cls(
            row_index=row_index,
            sample_id=sample_id,
            reason_code=reason_code,
            reason=_SAFE_FINDING_REASON_BY_CODE[reason_code],
        )

    @model_validator(mode="after")
    def require_fixed_safe_reason(self) -> SanitizedGoldExposureFinding:
        if self.reason != _SAFE_FINDING_REASON_BY_CODE[self.reason_code]:
            raise ValueError("finding reason must match its fixed safe reason code")
        return self


class StructuralGoldScan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    finding_count: int = Field(ge=0)


class OracleIsolationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    attestation_artifact: str | None = None
    attestation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attestation_bytes: int | None = Field(default=None, ge=0)
    covered_sample_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_pass_attestation(self) -> OracleIsolationEvidence:
        attestation = (
            self.attestation_artifact,
            self.attestation_sha256,
            self.attestation_bytes,
        )
        if self.status is VerificationStatus.PASS and any(item is None for item in attestation):
            raise ValueError("PASS oracle isolation requires a complete attestation identity")
        if self.status is not VerificationStatus.PASS and any(
            item is not None for item in attestation
        ):
            raise ValueError("non-PASS oracle isolation must not claim a PASS attestation")
        return self


OracleIsolationCheckName: TypeAlias = Literal[
    "raw-normalized-population-binding",
    "recorded-model-context-boundary",
    "recorded-model-input-gold-absence",
    "reference-patch-coverage",
    "upstream-git-hacking-review",
]

_ORACLE_ISOLATION_CHECK_ORDER: tuple[OracleIsolationCheckName, ...] = (
    "raw-normalized-population-binding",
    "recorded-model-context-boundary",
    "recorded-model-input-gold-absence",
    "reference-patch-coverage",
    "upstream-git-hacking-review",
)
_REFERENCE_PATCH_COVERAGE_PARTITIONS = frozenset(
    {
        "openhands/minimax_m25/swe-rebench-v2",
        "openhands/qwen35_122b/swe-rebench-v2",
        "sweagent/minimax_m25/swe-rebench-v2",
        "sweagent/qwen35_122b/swe-rebench-v2",
    }
)


class OracleIsolationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: OracleIsolationCheckName
    status: Literal[VerificationStatus.PASS, VerificationStatus.FAIL]


class OracleIsolationAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.oracle-isolation-attestation/v2"]
    method_version: Literal["nodelm.oracle-isolation-recorded-context-review/v2"]
    status: Literal[VerificationStatus.PASS, VerificationStatus.FAIL]
    review_scope: Literal["recorded-model-context-and-upstream-curation"]
    upstream_review_id: Literal["open-swe-traces-v1.0-paper-git-hacking-review/v1"]
    source_name: StrictStr = Field(min_length=1)
    source_repository_id: StrictStr = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    source_revision: StrictStr = Field(pattern=r"^[0-9a-fA-F]{40}$")
    partition_name: StrictStr = Field(pattern=r"^[a-z0-9._-]+/[a-z0-9._-]+/[a-z0-9._-]+$")
    harness: StrictStr = Field(pattern=r"^[a-z0-9._-]+$")
    generating_model: StrictStr = Field(min_length=1)
    materialization_manifest_artifact: StrictStr = Field(min_length=1)
    materialization_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    materialization_manifest_bytes: StrictInt = Field(ge=0)
    raw_artifact: StrictStr = Field(min_length=1)
    raw_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    raw_bytes: StrictInt = Field(ge=0)
    raw_row_count: StrictInt = Field(gt=0)
    task_provenance_artifact: StrictStr = Field(min_length=1)
    task_provenance_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    task_provenance_bytes: StrictInt = Field(ge=0)
    task_provenance_manifest_artifact: StrictStr = Field(min_length=1)
    task_provenance_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    task_provenance_manifest_bytes: StrictInt = Field(ge=0)
    normalization_manifest_artifact: StrictStr = Field(min_length=1)
    normalization_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_manifest_bytes: StrictInt = Field(ge=0)
    normalized_artifact: StrictStr = Field(min_length=1)
    normalized_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_bytes: StrictInt = Field(ge=0)
    expected_sample_count: StrictInt = Field(gt=0)
    covered_sample_count: StrictInt = Field(ge=0)
    reference_patch_row_count: StrictInt = Field(ge=0)
    checks: tuple[OracleIsolationCheck, ...]
    findings_artifact: StrictStr = Field(min_length=1)
    findings_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    findings_bytes: StrictInt = Field(ge=0)
    finding_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_pass_evidence(self) -> OracleIsolationAttestation:
        if tuple(check.name for check in self.checks) != _ORACLE_ISOLATION_CHECK_ORDER:
            raise ValueError("oracle isolation requires the complete ordered check set")
        checks_pass = all(check.status is VerificationStatus.PASS for check in self.checks)
        if self.status is VerificationStatus.PASS and (
            not checks_pass
            or self.finding_count
            or self.findings_bytes
            or self.covered_sample_count != self.expected_sample_count
        ):
            raise ValueError("PASS oracle isolation requires complete PASS checks")
        if self.status is VerificationStatus.FAIL and checks_pass:
            raise ValueError("FAIL oracle isolation requires at least one failed check")
        if self.covered_sample_count > self.raw_row_count:
            raise ValueError("covered samples cannot exceed raw rows")
        if self.reference_patch_row_count > self.raw_row_count:
            raise ValueError("reference patch rows cannot exceed raw rows")
        if (
            self.status is VerificationStatus.PASS
            and self.source_name == "open-swe-traces"
            and self.partition_name in _REFERENCE_PATCH_COVERAGE_PARTITIONS
            and (
                self.reference_patch_row_count != self.raw_row_count
                or self.reference_patch_row_count != self.covered_sample_count
            )
        ):
            raise ValueError(
                "selected oracle-isolation PASS requires complete reference patch coverage"
            )
        return self


class GoldExposureAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.gold-exposure-audit/v1"]
    method_version: Literal["nodelm.gold-exposure-audit-method/v1"]
    status: VerificationStatus
    normalization_manifest_artifact: str = Field(min_length=1)
    normalization_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_manifest_bytes: int = Field(ge=0)
    normalized_artifact: str = Field(min_length=1)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_bytes: int = Field(ge=0)
    expected_sample_count: int = Field(ge=0)
    audited_sample_count: int = Field(ge=0)
    structural_scan: StructuralGoldScan
    oracle_isolation: OracleIsolationEvidence
    findings_artifact: str = Field(min_length=1)
    findings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    findings_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def pass_requires_complete_zero_finding_coverage(self) -> GoldExposureAudit:
        if self.status is not VerificationStatus.PASS:
            return self
        if (
            self.structural_scan.status is not VerificationStatus.PASS
            or self.structural_scan.finding_count != 0
            or self.oracle_isolation.status is not VerificationStatus.PASS
            or self.expected_sample_count == 0
            or self.expected_sample_count != self.audited_sample_count
            or self.oracle_isolation.covered_sample_count != self.audited_sample_count
        ):
            raise ValueError(
                "PASS gold-exposure audit requires a non-empty population, zero findings, "
                "and complete oracle coverage"
            )
        return self


AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256: dict[str, str] = {}
AUTHORIZED_ORACLE_ATTESTATION_SHA256_BY_NORMALIZED_SHA256: dict[str, str] = {}


def require_authorized_oracle_attestation(
    *, normalized_sha256: str, attestation_sha256: str
) -> None:
    expected = AUTHORIZED_ORACLE_ATTESTATION_SHA256_BY_NORMALIZED_SHA256.get(normalized_sha256)
    if expected is None:
        raise OracleIsolationAuthorizationError(
            "no reviewed oracle-isolation attestation is authorized for this normalized artifact"
        )
    if attestation_sha256 != expected:
        raise OracleIsolationAuthorizationError(
            "oracle-isolation attestation digest is not authorized for this normalized artifact"
        )


def require_authorized_gold_audit(*, normalized_sha256: str, audit_sha256: str) -> None:
    expected = AUTHORIZED_GOLD_AUDIT_SHA256_BY_NORMALIZED_SHA256.get(normalized_sha256)
    if expected is None:
        raise GoldExposureAuthorizationError(
            "no reviewed gold-exposure audit is authorized for this normalized artifact"
        )
    if audit_sha256 != expected:
        raise GoldExposureAuthorizationError(
            "gold-exposure audit digest is not authorized for this normalized artifact"
        )
