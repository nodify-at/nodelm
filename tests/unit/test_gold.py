from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodelm.provenance.gold import (
    GoldExposureAudit,
    GoldExposureAuthorizationError,
    OracleIsolationAttestation,
    OracleIsolationAuthorizationError,
    SanitizedGoldExposureFinding,
    require_authorized_gold_audit,
    require_authorized_oracle_attestation,
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


def _oracle_attestation_payload() -> dict[str, object]:
    return {
        "schema_version": "nodelm.oracle-isolation-attestation/v2",
        "method_version": "nodelm.oracle-isolation-recorded-context-review/v2",
        "status": "PASS",
        "review_scope": "recorded-model-context-and-upstream-curation",
        "upstream_review_id": "open-swe-traces-v1.0-paper-git-hacking-review/v1",
        "source_name": "open-swe-traces",
        "source_repository_id": "nvidia/Open-SWE-Traces",
        "source_revision": "a" * 40,
        "partition_name": "openhands/minimax_m25/swe-rebench-v2",
        "harness": "openhands",
        "generating_model": "source-label:minimax_m25",
        "materialization_manifest_artifact": "raw.manifest.json",
        "materialization_manifest_sha256": "1" * 64,
        "materialization_manifest_bytes": 1,
        "raw_artifact": "raw.jsonl",
        "raw_sha256": "2" * 64,
        "raw_bytes": 2,
        "raw_row_count": 1,
        "task_provenance_artifact": "tasks.safe.jsonl",
        "task_provenance_sha256": "6" * 64,
        "task_provenance_bytes": 6,
        "task_provenance_manifest_artifact": "tasks.safe.manifest.json",
        "task_provenance_manifest_sha256": "7" * 64,
        "task_provenance_manifest_bytes": 7,
        "normalization_manifest_artifact": "normalized.manifest.json",
        "normalization_manifest_sha256": "3" * 64,
        "normalization_manifest_bytes": 3,
        "normalized_artifact": "normalized.jsonl",
        "normalized_sha256": "4" * 64,
        "normalized_bytes": 4,
        "expected_sample_count": 1,
        "covered_sample_count": 1,
        "reference_patch_row_count": 1,
        "checks": [
            {"name": "raw-normalized-population-binding", "status": "PASS"},
            {"name": "recorded-model-context-boundary", "status": "PASS"},
            {"name": "recorded-model-input-gold-absence", "status": "PASS"},
            {"name": "reference-patch-coverage", "status": "PASS"},
            {"name": "upstream-git-hacking-review", "status": "PASS"},
        ],
        "findings_artifact": "oracle.findings.jsonl",
        "findings_sha256": "5" * 64,
        "findings_bytes": 0,
        "finding_count": 0,
    }


def test_oracle_attestation_pass_requires_complete_zero_finding_evidence() -> None:
    payload = _oracle_attestation_payload()
    checks = payload["checks"]
    assert isinstance(checks, list)
    checks[0] = {"name": "raw-normalized-population-binding", "status": "FAIL"}

    with pytest.raises(ValidationError, match="complete PASS checks"):
        OracleIsolationAttestation.model_validate(payload)


def test_selected_oracle_attestation_pass_requires_reference_patch_coverage() -> None:
    payload = _oracle_attestation_payload()
    payload["reference_patch_row_count"] = 0

    with pytest.raises(ValidationError, match="complete reference patch coverage"):
        OracleIsolationAttestation.model_validate(payload)


def test_unreviewed_oracle_attestation_is_not_authorized() -> None:
    with pytest.raises(OracleIsolationAuthorizationError, match="no reviewed"):
        require_authorized_oracle_attestation(
            normalized_sha256="e" * 64,
            attestation_sha256="f" * 64,
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
