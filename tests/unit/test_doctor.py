from __future__ import annotations

from nodelm.doctor import build_doctor_report
from nodelm.models import VerificationStatus


def test_doctor_distinguishes_required_and_optional_missing_tools() -> None:
    available = {
        "git": "/usr/bin/git",
        "python3": "/usr/bin/python3",
        "rg": "/usr/bin/rg",
        "uv": "/usr/bin/uv",
        "node": "/usr/bin/node",
    }

    report = build_doctor_report(
        finder=available.get,
        node_version_reader=lambda path: "v20.18.0",
    )

    assert report.status is VerificationStatus.PASS
    checks = {check.name: check for check in report.checks}
    assert checks["git"].status is VerificationStatus.PASS
    assert checks["node"].status is VerificationStatus.PASS


def test_doctor_fails_when_required_tool_is_missing() -> None:
    report = build_doctor_report(
        finder=lambda name: None,
        node_version_reader=lambda path: None,
    )

    assert report.status is VerificationStatus.FAIL
    assert any(
        check.required and check.status is VerificationStatus.FAIL for check in report.checks
    )


def test_doctor_rejects_an_unsupported_node_major() -> None:
    report = build_doctor_report(
        finder=lambda name: f"/usr/bin/{name}",
        node_version_reader=lambda path: "v18.20.0",
    )

    assert report.status is VerificationStatus.FAIL
    node = next(check for check in report.checks if check.name == "node")
    assert node.status is VerificationStatus.FAIL
    assert node.version == "v18.20.0"
