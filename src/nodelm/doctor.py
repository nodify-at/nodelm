from __future__ import annotations

import platform
import re
import subprocess
import sys
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from nodelm.models import VerificationStatus


class ToolCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    required: bool
    status: VerificationStatus
    path: str | None
    version: str | None = None


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "nodelm.doctor/v1"
    status: VerificationStatus
    python_version: str
    platform: str
    checks: tuple[ToolCheck, ...]


def _read_node_version(path: str) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={"LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_doctor_report(
    *,
    finder: Callable[[str], str | None],
    node_version_reader: Callable[[str], str | None] = _read_node_version,
) -> DoctorReport:
    tools = (
        ("git", True),
        ("python3", True),
        ("rg", True),
        ("uv", True),
        ("node", True),
        ("npm", False),
        ("pnpm", False),
        ("nvidia-smi", False),
    )
    checks: list[ToolCheck] = []
    for name, required in tools:
        path = finder(name)
        version: str | None = None
        if path and name == "node":
            version = node_version_reader(path)
            match = re.fullmatch(r"v?(\d+)(?:\.\d+){1,2}", version or "")
            status = (
                VerificationStatus.PASS
                if match is not None and int(match.group(1)) >= 20
                else VerificationStatus.FAIL
            )
        elif path:
            status = VerificationStatus.PASS
        elif required:
            status = VerificationStatus.FAIL
        else:
            status = VerificationStatus.NOT_RUN
        checks.append(
            ToolCheck(
                name=name,
                required=required,
                status=status,
                path=path,
                version=version,
            )
        )

    overall = (
        VerificationStatus.FAIL
        if any(check.required and check.status is VerificationStatus.FAIL for check in checks)
        else VerificationStatus.PASS
    )
    return DoctorReport(
        status=overall,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        checks=tuple(checks),
    )
