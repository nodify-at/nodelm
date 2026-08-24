from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OutcomeCategory(StrEnum):
    """Stable outcome vocabulary for model and command orchestration."""

    SUCCESS = "success"
    MODEL_FAILURE = "model_failure"
    TEST_FAILURE = "test_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    DEPENDENCY_INSTALL_FAILURE = "dependency_install_failure"
    TIMEOUT = "timeout"
    TOOL_PROTOCOL_FAILURE = "tool_protocol_failure"
    INTERNAL_FAILURE = "internal_failure"


_NONZERO_OUTCOMES = frozenset(OutcomeCategory) - {
    OutcomeCategory.SUCCESS,
    OutcomeCategory.TIMEOUT,
}


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """A validated argv-only command request.

    ``cwd`` is interpreted relative to the executor's workspace root. Environment values
    supplement a small inherited allowlist; the full host environment is never forwarded.
    """

    argv: tuple[str, ...]
    cwd: Path | str | None = None
    timeout_seconds: float = 60.0
    failure_outcome: OutcomeCategory = OutcomeCategory.ENVIRONMENT_FAILURE
    success_exit_codes: frozenset[int] = field(default_factory=lambda: frozenset({0}))
    env: Mapping[str, str] = field(default_factory=dict)
    redact_values: tuple[str, ...] = ()
    max_output_bytes: int | None = None

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if not argv or not argv[0]:
            raise ValueError("argv must contain a non-empty executable")
        if any(not isinstance(argument, str) or "\0" in argument for argument in argv):
            raise ValueError("argv entries must be strings without NUL bytes")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        if self.failure_outcome not in _NONZERO_OUTCOMES:
            raise ValueError("failure_outcome must describe a nonzero command outcome")

        success_exit_codes = frozenset(self.success_exit_codes)
        if not success_exit_codes or any(not isinstance(code, int) for code in success_exit_codes):
            raise ValueError("success_exit_codes must contain at least one integer")
        if self.max_output_bytes is not None and self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")

        environment = dict(self.env)
        for name, value in environment.items():
            if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            if not isinstance(value, str) or "\0" in value:
                raise ValueError(f"environment value for {name!r} must be a NUL-free string")

        redact_values = tuple(value for value in self.redact_values if value)
        if any(not isinstance(value, str) for value in redact_values):
            raise ValueError("redact_values entries must be strings")

        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "success_exit_codes", success_exit_codes)
        object.__setattr__(self, "env", MappingProxyType(environment))
        object.__setattr__(self, "redact_values", redact_values)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded, redacted observation from one command attempt."""

    argv: tuple[str, ...]
    cwd: Path
    outcome: OutcomeCategory
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return self.outcome is OutcomeCategory.SUCCESS

    def to_evidence(self) -> dict[str, object]:
        """Return a versioned, JSON-serializable command observation."""

        return {
            "schema_version": "nodelm.command-result/v1",
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "outcome": self.outcome.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }
