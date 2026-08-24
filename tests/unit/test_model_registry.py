from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nodelm.evaluation.registry import (
    CandidateModel,
    CandidateRegistry,
    CandidateRegistryError,
)
from nodelm.models import VerificationStatus


def _verified_candidate(**overrides: object) -> CandidateModel:
    payload: dict[str, object] = {
        "name": "fixture",
        "repository_id": "owner/model",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "parameter_count": 27_000_000_000,
        "active_parameter_count": 27_000_000_000,
        "architecture": "ExampleForConditionalGeneration",
        "model_type": "example",
        "loader_class": "AutoModelForMultimodalLM",
        "context_limit": 262_144,
        "metadata_verified_on": "2026-08-24",
        "evidence_urls": (
            "https://huggingface.co/owner/model",
            f"https://huggingface.co/owner/model/commit/{'a' * 40}",
        ),
        "metadata_status": VerificationStatus.PASS,
    }
    payload.update(overrides)
    return CandidateModel.model_validate(payload)


def test_checked_in_candidate_registry_has_three_verified_models_and_no_run_claims() -> None:
    registry = CandidateRegistry.load(Path("configs/evaluation/candidates.yaml"))

    assert registry.metadata_status is VerificationStatus.PASS
    assert registry.execution_status is VerificationStatus.NOT_RUN
    assert registry.bakeoff_status is VerificationStatus.NOT_RUN
    assert registry.selected_candidate is None
    assert [candidate.repository_id for candidate in registry.candidates] == [
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3-Coder-Next",
    ]
    assert all(
        candidate.metadata_status is VerificationStatus.PASS for candidate in registry.candidates
    )


@pytest.mark.parametrize(
    "missing_field",
    (
        "revision",
        "license",
        "parameter_count",
        "architecture",
        "model_type",
        "loader_class",
        "context_limit",
        "metadata_verified_on",
        "evidence_urls",
    ),
)
def test_pass_candidate_requires_complete_immutable_metadata(missing_field: str) -> None:
    with pytest.raises(ValidationError, match="PASS candidate metadata requires"):
        _verified_candidate(**{missing_field: None if missing_field != "evidence_urls" else ()})


def test_candidate_evidence_must_be_official_and_bound_to_the_repository() -> None:
    with pytest.raises(ValidationError, match="official Hugging Face repository"):
        _verified_candidate(evidence_urls=("https://example.invalid/model",))


def test_candidate_rejects_active_parameter_count_above_total() -> None:
    with pytest.raises(ValidationError, match="active parameter count"):
        _verified_candidate(active_parameter_count=27_000_000_001)


def test_registry_rejects_duplicate_candidate_repository_ids() -> None:
    first = _verified_candidate(name="first")
    second = _verified_candidate(name="second")

    with pytest.raises(ValidationError, match="duplicate candidate repository_id"):
        CandidateRegistry(
            schema_version="nodelm.candidate-registry/v1",
            metadata_status=VerificationStatus.PASS,
            execution_status=VerificationStatus.NOT_RUN,
            bakeoff_status=VerificationStatus.NOT_RUN,
            candidates=(first, second),
            reason="Execution and bake-off have not run.",
        )


@pytest.mark.parametrize(
    ("registry_metadata_status", "candidate_metadata_status"),
    (
        (VerificationStatus.NOT_RUN, VerificationStatus.PASS),
        (VerificationStatus.NOT_RUN, VerificationStatus.FAIL),
    ),
)
def test_pass_execution_requires_verified_registry_and_candidate_metadata(
    registry_metadata_status: VerificationStatus,
    candidate_metadata_status: VerificationStatus,
) -> None:
    with pytest.raises(
        ValidationError,
        match="PASS candidate execution requires PASS registry and candidate metadata",
    ):
        CandidateRegistry(
            schema_version="nodelm.candidate-registry/v1",
            metadata_status=registry_metadata_status,
            execution_status=VerificationStatus.PASS,
            bakeoff_status=VerificationStatus.NOT_RUN,
            candidates=(_verified_candidate(metadata_status=candidate_metadata_status),),
            execution_evidence=("artifacts/reports/candidate-execution.json",),
            reason="Invalid claimed execution.",
        )


def test_registry_rejects_blank_execution_evidence() -> None:
    with pytest.raises(ValidationError):
        CandidateRegistry(
            schema_version="nodelm.candidate-registry/v1",
            metadata_status=VerificationStatus.PASS,
            execution_status=VerificationStatus.PASS,
            bakeoff_status=VerificationStatus.NOT_RUN,
            candidates=(_verified_candidate(),),
            execution_evidence=("   ",),
            reason="Invalid claimed execution.",
        )


def test_registry_rejects_blank_bakeoff_evidence() -> None:
    with pytest.raises(ValidationError):
        CandidateRegistry(
            schema_version="nodelm.candidate-registry/v1",
            metadata_status=VerificationStatus.PASS,
            execution_status=VerificationStatus.PASS,
            bakeoff_status=VerificationStatus.PASS,
            candidates=(_verified_candidate(),),
            selected_candidate="fixture",
            execution_evidence=("artifacts/reports/candidate-execution.json",),
            bakeoff_evidence=("\t",),
            reason="Invalid claimed bake-off.",
        )


def test_registry_strips_evidence_reference_whitespace() -> None:
    registry = CandidateRegistry(
        schema_version="nodelm.candidate-registry/v1",
        metadata_status=VerificationStatus.PASS,
        execution_status=VerificationStatus.PASS,
        bakeoff_status=VerificationStatus.PASS,
        candidates=(_verified_candidate(),),
        selected_candidate="fixture",
        execution_evidence=("  artifacts/reports/candidate-execution.json  ",),
        bakeoff_evidence=("  artifacts/reports/student-bakeoff.json  ",),
        reason="Measured execution and bake-off.",
    )

    assert registry.execution_evidence == ("artifacts/reports/candidate-execution.json",)
    assert registry.bakeoff_evidence == ("artifacts/reports/student-bakeoff.json",)


def test_registry_rejects_pass_bakeoff_without_execution_evidence_and_selection() -> None:
    with pytest.raises(ValidationError, match="PASS bake-off"):
        CandidateRegistry(
            schema_version="nodelm.candidate-registry/v1",
            metadata_status=VerificationStatus.PASS,
            execution_status=VerificationStatus.NOT_RUN,
            bakeoff_status=VerificationStatus.PASS,
            candidates=(_verified_candidate(),),
            reason="Invalid claimed result.",
        )


def test_registry_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "schema_version: nodelm.candidate-registry/v1\n"
        "metadata_status: NOT RUN\n"
        "execution_status: NOT RUN\n"
        "bakeoff_status: NOT RUN\n"
        "candidates: []\n"
        "reason: fixture\n"
        "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(CandidateRegistryError, match="invalid candidate registry"):
        CandidateRegistry.load(path)
