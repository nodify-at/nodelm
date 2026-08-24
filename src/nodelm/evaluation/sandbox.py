from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.harness.sandbox import RootlessPodmanExecutor, SandboxUnavailableError


class FixtureSandbox(Protocol):
    """Isolation boundary used to execute tests against a model-authored patch."""

    def run_node_tests(self, workspace: Path) -> CommandResult: ...

    def evidence(self) -> dict[str, object]: ...


class PodmanFixtureSandbox:
    """Run protected fixture tests through the generic rootless Podman backend."""

    def __init__(self, image: str, *, executable: str = "podman") -> None:
        self._executor = RootlessPodmanExecutor(image, executable=executable)

    @property
    def image(self) -> str:
        return self._executor.image

    def command(self, workspace: Path) -> tuple[str, ...]:
        return self._executor.command(
            workspace,
            ("node", "--test", "test/math.test.js"),
        )

    def run_node_tests(self, workspace: Path) -> CommandResult:
        return self._executor.run(
            workspace,
            ("node", "--test", "test/math.test.js"),
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def evidence(self) -> dict[str, object]:
        return {
            **self._executor.evidence(),
            "schema_version": "nodelm.fixture-sandbox/v1",
        }


__all__ = [
    "FixtureSandbox",
    "PodmanFixtureSandbox",
    "SandboxUnavailableError",
]
