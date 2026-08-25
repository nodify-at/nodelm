from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodelm.provenance.gold import (
    GoldExposureAudit,
    GoldExposureAuthorizationError,
    SanitizedGoldExposureFinding,
    require_authorized_gold_audit,
)


def _audit_payload() -> dict[str, object]:
    return {
        "schema_version": "nodelm.gold-exposure-audit/v1",
        "method_version": "nodelm.gold-exposure-audit-method/v1",
        "status": "PASS",
        "normalization_manifest_artifact": "normalized.manifest.json",
        "normalization_manifest_sha256": "a" * 64,
        "normalization_manifest_bytes": 1,
        "normalized_artifact": "normalized.jsonl",
        "normalized_sha256": "b" * 64,
        "normalized_bytes": 2,
        "expected_sample_count": 1,
        "audited_sample_count": 1,
        "structural_scan": {"status": "PASS", "finding_count": 0},
        "oracle_isolation": {
            "status": "PASS",
            "attestation_artifact": "oracle.json",
            "attestation_sha256": "c" * 64,
            "attestation_bytes": 3,
            "covered_sample_count": 1,
        },
        "findings_artifact": "findings.jsonl",
        "findings_sha256": "d" * 64,
        "findings_bytes": 0,
    }


def test_gold_exposure_pass_requires_complete_zero_finding_coverage() -> None:
    payload = _audit_payload()
    payload["structural_scan"] = {"status": "FAIL", "finding_count": 1}

    with pytest.raises(ValidationError, match="zero findings"):
        GoldExposureAudit.model_validate(payload)


def test_gold_exposure_pass_requires_a_nonempty_population() -> None:
    payload = _audit_payload()
    payload["expected_sample_count"] = 0
    payload["audited_sample_count"] = 0
    oracle_isolation = payload["oracle_isolation"]
    assert isinstance(oracle_isolation, dict)
    oracle_isolation["covered_sample_count"] = 0

    with pytest.raises(ValidationError, match="non-empty population"):
        GoldExposureAudit.model_validate(payload)


def test_unreviewed_gold_audit_is_not_authorized() -> None:
    with pytest.raises(GoldExposureAuthorizationError, match="no reviewed"):
        require_authorized_gold_audit(
            normalized_sha256="e" * 64,
            audit_sha256="f" * 64,
        )


def test_sanitized_gold_finding_has_a_fixed_safe_reason_and_forbids_extras() -> None:
    payload = {
        "row_index": 0,
        "sample_id": "a" * 64,
        "reason_code": "forbidden_gold_reference_patch",
        "reason": "trajectory contains forbidden gold/reference patch metadata",
    }

    finding = SanitizedGoldExposureFinding.model_validate(payload)

    assert finding.model_dump(mode="json") == payload
    assert (
        SanitizedGoldExposureFinding.from_reason_code(
            row_index=0,
            sample_id="a" * 64,
            reason_code="forbidden_gold_reference_patch",
        ).model_dump(mode="json")
        == payload
    )

    with pytest.raises(ValidationError, match="reason must match"):
        SanitizedGoldExposureFinding.model_validate(
            {
                **payload,
                "reason": "normalized row does not satisfy the sample schema",
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SanitizedGoldExposureFinding.model_validate({**payload, "trajectory": "secret"})
