from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LicenseDisposition(StrEnum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LicenseDecision:
    raw_license: str | None
    normalized_spdx: str | None
    disposition: LicenseDisposition
    reason: str


_NORMALIZED_ALLOWLIST = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
}


def evaluate_license(raw_license: str | None) -> LicenseDecision:
    if raw_license is None or not raw_license.strip():
        return LicenseDecision(
            raw_license=raw_license,
            normalized_spdx=None,
            disposition=LicenseDisposition.UNKNOWN,
            reason="repository license is missing; V1 policy rejects unknown licenses",
        )

    normalized = _NORMALIZED_ALLOWLIST.get(raw_license.strip().lower())
    if normalized:
        return LicenseDecision(
            raw_license=raw_license,
            normalized_spdx=normalized,
            disposition=LicenseDisposition.ALLOW,
            reason="license is in the conservative V1 allowlist",
        )

    lowered = raw_license.lower()
    if "gpl" in lowered:
        disposition = LicenseDisposition.REJECT
        reason = "copyleft license is outside the V1 allowlist"
    else:
        disposition = LicenseDisposition.UNKNOWN
        reason = "license is not recognized by the conservative V1 allowlist"
    return LicenseDecision(
        raw_license=raw_license,
        normalized_spdx=None,
        disposition=disposition,
        reason=reason,
    )
