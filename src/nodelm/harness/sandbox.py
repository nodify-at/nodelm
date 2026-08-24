from __future__ import annotations

import os
import re
import secrets
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from nodelm.harness.executor import CommandExecutor
from nodelm.harness.models import CommandResult, OutcomeCategory
from nodelm.harness.policy import CommandPolicy

_DIGEST_PINNED_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")


class SandboxUnavailableError(RuntimeError):
    """The requested isolation backend cannot safely execute repository code."""


class RootlessPodmanExecutor:
    """Execute argv-only commands inside a constrained rootless Podman container."""

    def __init__(self, image: str, *, executable: str = "podman") -> None:
        if not _DIGEST_PINNED_IMAGE.fullmatch(image):
            raise ValueError("sandbox image must be an exact sha256 digest reference")
        if not executable or "\0" in executable:
            raise ValueError("sandbox executable must be a non-empty NUL-free string")
        self.image = image
        self.executable = executable
        self._ready_workspace: Path | None = None
        self._rootless_command: CommandResult | None = None
        self._image_command: CommandResult | None = None
        self._last_container_id: str | None = None
        self._last_cleanup_command: CommandResult | None = None

    def _ensure_ready(self, workspace: Path) -> None:
        resolved = workspace.resolve()
        if self._ready_workspace == resolved:
            return
        policy = CommandPolicy(resolved)
        executor = CommandExecutor(resolved)
        rootless = executor.run(
            policy.generic(
                (self.executable, "info", "--format", "{{.Host.Security.Rootless}}"),
                trusted_local=True,
                timeout_seconds=20,
            )
        )
        self._rootless_command = rootless
        if rootless.outcome is not OutcomeCategory.SUCCESS:
            raise SandboxUnavailableError("Podman availability/rootless probe failed")
        if rootless.stdout.strip().lower() != "true":
            raise SandboxUnavailableError("Podman is not running in rootless mode")

        image_check = executor.run(
            policy.generic(
                (self.executable, "image", "exists", self.image),
                trusted_local=True,
                timeout_seconds=20,
            )
        )
        self._image_command = image_check
        if image_check.outcome is not OutcomeCategory.SUCCESS:
            raise SandboxUnavailableError(
                "digest-pinned sandbox image is not preloaded; implicit pulls are forbidden"
            )
        self._ready_workspace = resolved

    def command(
        self,
        workspace: Path,
        argv: Sequence[str],
        *,
        cwd: Path | str = ".",
        writable_workspace: bool = False,
        _container_name: str | None = None,
        _cidfile: Path | None = None,
    ) -> tuple[str, ...]:
        resolved = workspace.resolve()
        if not resolved.is_dir():
            raise ValueError(f"sandbox workspace is not an existing directory: {resolved}")
        normalized_argv = tuple(argv)
        if not normalized_argv or any(
            not isinstance(argument, str) or not argument or "\0" in argument
            for argument in normalized_argv
        ):
            raise ValueError("sandbox argv must contain non-empty NUL-free strings")
        if writable_workspace:
            raise ValueError(
                "writable host bind mounts are forbidden; use quota-backed scratch storage"
            )
        lifecycle_arguments: tuple[str, ...] = ()
        if (_container_name is None) != (_cidfile is None):
            raise ValueError("sandbox container name and cidfile must be supplied together")
        if _container_name is not None and _cidfile is not None:
            if re.fullmatch(r"nodelm-[0-9a-f]{32}", _container_name) is None:
                raise ValueError("sandbox container name is invalid")
            cidfile = _cidfile.resolve()
            if cidfile.exists() or not cidfile.parent.is_dir():
                raise ValueError("sandbox cidfile must be a new file in an existing directory")
            lifecycle_arguments = (
                f"--name={_container_name}",
                f"--cidfile={cidfile}",
            )
        candidate_cwd = Path(cwd)
        if candidate_cwd.is_absolute():
            raise ValueError("sandbox cwd must be relative to the workspace")
        host_cwd = (resolved / candidate_cwd).resolve()
        if not host_cwd.is_dir() or not host_cwd.is_relative_to(resolved):
            raise ValueError("sandbox cwd must resolve to an existing workspace directory")
        relative_cwd = os.path.relpath(host_cwd, start=resolved)
        container_cwd = "/workspace" if relative_cwd == "." else f"/workspace/{relative_cwd}"
        return (
            self.executable,
            "run",
            "--rm",
            *lifecycle_arguments,
            "--pull=never",
            "--network=none",
            "--pid=private",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            "--memory=512m",
            "--cpus=1",
            "--user=65534:65534",
            "--ipc=none",
            "--ulimit=nofile=256:256",
            "--ulimit=fsize=16777216:16777216",
            "--ulimit=core=0:0",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16777216",
            f"--volume={resolved}:/workspace:ro",
            f"--workdir={container_cwd}",
            f"--entrypoint={normalized_argv[0]}",
            self.image,
            *normalized_argv[1:],
        )

    def run(
        self,
        workspace: Path,
        argv: Sequence[str],
        *,
        cwd: Path | str = ".",
        writable_workspace: bool = False,
        timeout_seconds: float = 60,
        failure_outcome: OutcomeCategory = OutcomeCategory.ENVIRONMENT_FAILURE,
    ) -> CommandResult:
        resolved = workspace.resolve()
        self._ensure_ready(resolved)
        policy = CommandPolicy(resolved)
        executor = CommandExecutor(resolved)
        container_name = f"nodelm-{secrets.token_hex(16)}"
        self._last_container_id = None
        self._last_cleanup_command = None
        with tempfile.TemporaryDirectory(prefix="nodelm-podman-") as temporary_directory:
            cidfile = Path(temporary_directory).resolve() / "container.cid"
            result: CommandResult | None = None
            cleanup: CommandResult
            try:
                result = executor.run(
                    policy.generic(
                        self.command(
                            resolved,
                            argv,
                            cwd=cwd,
                            writable_workspace=writable_workspace,
                            _container_name=container_name,
                            _cidfile=cidfile,
                        ),
                        trusted_local=True,
                        timeout_seconds=timeout_seconds,
                        failure_outcome=failure_outcome,
                    )
                )
                try:
                    if cidfile.is_file() and not cidfile.is_symlink():
                        container_id = cidfile.read_text(encoding="ascii").strip()
                        if re.fullmatch(r"[0-9a-f]{64}", container_id) is not None:
                            self._last_container_id = container_id
                except (OSError, UnicodeError):
                    self._last_container_id = None
            finally:
                cleanup = executor.run(
                    policy.generic(
                        (
                            self.executable,
                            "rm",
                            "--force",
                            "--ignore",
                            "--time=0",
                            "--",
                            container_name,
                        ),
                        trusted_local=True,
                        timeout_seconds=20,
                    )
                )
                self._last_cleanup_command = cleanup
            assert result is not None
            if cleanup.outcome is not OutcomeCategory.SUCCESS:
                return replace(
                    result,
                    outcome=OutcomeCategory.INTERNAL_FAILURE,
                    stderr=(
                        result.stderr.rstrip() + "\nsandbox container cleanup could not be verified"
                    ).lstrip(),
                )
            return result

    def evidence(self) -> dict[str, object]:
        return {
            "schema_version": "nodelm.rootless-podman/v1",
            "backend": "rootless-podman",
            "image": self.image,
            "implicit_pull": False,
            "network": "none",
            "workspace": "read-only",
            "container_id": self._last_container_id,
            "cleanup": (
                self._last_cleanup_command.to_evidence()
                if self._last_cleanup_command is not None
                else None
            ),
            "rootless_probe": (
                self._rootless_command.to_evidence() if self._rootless_command is not None else None
            ),
            "image_probe": (
                self._image_command.to_evidence() if self._image_command is not None else None
            ),
        }
