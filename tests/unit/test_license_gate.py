from __future__ import annotations

import pytest

from nodelm.licenses.gate import LicenseDisposition, evaluate_license


@pytest.mark.parametrize(
    ("raw", "spdx"),
    [
        ("MIT", "MIT"),
        ("MIT License", "MIT"),
        ("Apache License 2.0", "Apache-2.0"),
        ("BSD 2-Clause", "BSD-2-Clause"),
        ("BSD-3-Clause", "BSD-3-Clause"),
    ],
)
def test_v1_allowlist_accepts_only_normalized_permissive_licenses(raw: str, spdx: str) -> None:
    decision = evaluate_license(raw)

    assert decision.disposition is LicenseDisposition.ALLOW
    assert decision.normalized_spdx == spdx


@pytest.mark.parametrize("raw", [None, "", "unknown", "GPL-3.0", "AGPL-3.0", "MIT OR GPL-3.0"])
def test_unknown_and_copyleft_licenses_are_rejected_but_auditable(raw: str | None) -> None:
    decision = evaluate_license(raw)

    assert decision.disposition is not LicenseDisposition.ALLOW
    assert decision.reason
