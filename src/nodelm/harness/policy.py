from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from nodelm.harness.models import CommandSpec, OutcomeCategory

_PACKAGE_SCRIPT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]*$")
_GIT_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@{}^~:+-]*$")
_INSTALL_SUBCOMMANDS = frozenset({"add", "ci", "i", "install"})


class CommandPolicyError(ValueError):
    """A requested command is outside the explicit trusted-repository policy."""


class CommandPolicy:
    """Create argv-only commands for a validated local workspace.

    This policy reduces accidental command injection and makes failure classification
    explicit. It is not a security sandbox: package scripts, compilers, and test runners
    execute repository code and therefore require a trusted local repository or fixture.
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self.workspace_root = _validated_root(workspace_root)

    def repo_tree(self, *, cwd: Path | str | None = None) -> CommandSpec:
        return self._spec(
            ("git", "ls-files", "--cached", "--others", "--exclude-standard"), cwd=cwd
        )

    def search(
        self,
        pattern: str,
        *,
        paths: Sequence[Path | str] = (".",),
        cwd: Path | str | None = None,
    ) -> CommandSpec:
        if "\0" in pattern:
            raise CommandPolicyError("search pattern cannot contain a NUL byte")
        resolved_cwd = self._cwd(cwd)
        arguments = tuple(self._path_argument(path, resolved_cwd) for path in paths)
        if not arguments:
            raise CommandPolicyError("search requires at least one workspace path")
        return CommandSpec(
            argv=(
                "rg",
                "--line-number",
                "--hidden",
                "--glob",
                "!.git/**",
                "--",
                pattern,
                *arguments,
            ),
            cwd=resolved_cwd,
            success_exit_codes=frozenset({0, 1}),
        )

    def read_file(self, path: Path | str, *, cwd: Path | str | None = None) -> CommandSpec:
        resolved_cwd = self._cwd(cwd)
        return CommandSpec(
            argv=("cat", "--", self._path_argument(path, resolved_cwd)), cwd=resolved_cwd
        )

    def git_status(self, *, cwd: Path | str | None = None) -> CommandSpec:
        return self._spec(("git", "status", "--short", "--branch"), cwd=cwd)

    def git_diff(
        self,
        *,
        paths: Sequence[Path | str] = (),
        cwd: Path | str | None = None,
    ) -> CommandSpec:
        resolved_cwd = self._cwd(cwd)
        arguments = tuple(self._path_argument(path, resolved_cwd) for path in paths)
        return CommandSpec(
            argv=("git", "diff", "--no-ext-diff", "--", *arguments), cwd=resolved_cwd
        )

    def git_log(self, *, max_count: int = 20, cwd: Path | str | None = None) -> CommandSpec:
        if max_count <= 0 or max_count > 1_000:
            raise CommandPolicyError("max_count must be between 1 and 1000")
        return self._spec(
            ("git", "log", f"--max-count={max_count}", "--oneline", "--decorate=no"),
            cwd=cwd,
        )

    def git_show(self, revision: str, *, cwd: Path | str | None = None) -> CommandSpec:
        if not _GIT_REVISION.fullmatch(revision):
            raise CommandPolicyError("git revision contains unsupported characters")
        return self._spec(("git", "show", "--no-ext-diff", "--format=fuller", revision), cwd=cwd)

    def npm(self, arguments: Sequence[str], *, cwd: Path | str | None = None) -> CommandSpec:
        return self._node_tool("npm", arguments, cwd=cwd)

    def pnpm(self, arguments: Sequence[str], *, cwd: Path | str | None = None) -> CommandSpec:
        return self._node_tool("pnpm", arguments, cwd=cwd)

    def node(self, arguments: Sequence[str], *, cwd: Path | str | None = None) -> CommandSpec:
        return self._spec(("node", *_arguments(arguments)), cwd=cwd)

    def node_test(
        self, arguments: Sequence[str] = (), *, cwd: Path | str | None = None
    ) -> CommandSpec:
        return self._spec(
            ("node", "--test", *_arguments(arguments)),
            cwd=cwd,
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def jest(self, arguments: Sequence[str] = (), *, cwd: Path | str | None = None) -> CommandSpec:
        return self._spec(
            ("jest", *_arguments(arguments)),
            cwd=cwd,
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def vitest(
        self, arguments: Sequence[str] = (), *, cwd: Path | str | None = None
    ) -> CommandSpec:
        return self._spec(
            ("vitest", *_arguments(arguments)),
            cwd=cwd,
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def tsc(
        self,
        *,
        project: Path | str | None = None,
        arguments: Sequence[str] = (),
        cwd: Path | str | None = None,
    ) -> CommandSpec:
        resolved_cwd = self._cwd(cwd)
        project_arguments: tuple[str, ...] = ()
        if project is not None:
            project_arguments = ("--project", self._path_argument(project, resolved_cwd))
        return CommandSpec(
            argv=("tsc", *project_arguments, *_arguments(arguments), "--noEmit"),
            cwd=resolved_cwd,
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def eslint(
        self, arguments: Sequence[str] = (), *, cwd: Path | str | None = None
    ) -> CommandSpec:
        return self._spec(
            ("eslint", *_arguments(arguments)),
            cwd=cwd,
            failure_outcome=OutcomeCategory.TEST_FAILURE,
        )

    def package_script(
        self,
        manager: str,
        script: str,
        *,
        arguments: Sequence[str] = (),
        cwd: Path | str | None = None,
    ) -> CommandSpec:
        if manager not in {"npm", "pnpm"}:
            raise CommandPolicyError("package script manager must be npm or pnpm")
        if not _PACKAGE_SCRIPT.fullmatch(script):
            raise CommandPolicyError("package script name contains unsupported characters")
        failure = (
            OutcomeCategory.TEST_FAILURE
            if script == "test" or script.startswith("test:")
            else OutcomeCategory.ENVIRONMENT_FAILURE
        )
        return self._spec(
            (manager, "run", script, "--", *_arguments(arguments)),
            cwd=cwd,
            failure_outcome=failure,
        )

    def dependency_install(
        self,
        manager: str,
        *,
        frozen: bool = True,
        ignore_scripts: bool = True,
        cwd: Path | str | None = None,
    ) -> CommandSpec:
        if manager == "npm":
            arguments = ["ci" if frozen else "install"]
        elif manager == "pnpm":
            arguments = ["install"]
            if frozen:
                arguments.append("--frozen-lockfile")
        else:
            raise CommandPolicyError("dependency manager must be npm or pnpm")
        if ignore_scripts:
            arguments.append("--ignore-scripts")
        return self._spec(
            (manager, *arguments),
            cwd=cwd,
            failure_outcome=OutcomeCategory.DEPENDENCY_INSTALL_FAILURE,
        )

    def generic(
        self,
        argv: Sequence[str],
        *,
        trusted_local: bool = False,
        cwd: Path | str | None = None,
        timeout_seconds: float = 60.0,
        failure_outcome: OutcomeCategory = OutcomeCategory.ENVIRONMENT_FAILURE,
        env: dict[str, str] | None = None,
        redact_values: Sequence[str] = (),
        max_output_bytes: int | None = None,
    ) -> CommandSpec:
        if not trusted_local:
            raise CommandPolicyError(
                "generic commands require explicit trusted local execution opt-in"
            )
        return CommandSpec(
            argv=_arguments(argv),
            cwd=self._cwd(cwd),
            timeout_seconds=timeout_seconds,
            failure_outcome=failure_outcome,
            env={} if env is None else env,
            redact_values=tuple(redact_values),
            max_output_bytes=max_output_bytes,
        )

    def _node_tool(
        self,
        executable: str,
        arguments: Sequence[str],
        *,
        cwd: Path | str | None,
    ) -> CommandSpec:
        normalized = _arguments(arguments)
        command = normalized[0] if normalized else ""
        if command in _INSTALL_SUBCOMMANDS:
            failure = OutcomeCategory.DEPENDENCY_INSTALL_FAILURE
        elif command in {"test", "t"} or (
            command == "run"
            and len(normalized) > 1
            and (normalized[1] == "test" or normalized[1].startswith("test:"))
        ):
            failure = OutcomeCategory.TEST_FAILURE
        else:
            failure = OutcomeCategory.ENVIRONMENT_FAILURE
        return self._spec((executable, *normalized), cwd=cwd, failure_outcome=failure)

    def _spec(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | str | None,
        failure_outcome: OutcomeCategory = OutcomeCategory.ENVIRONMENT_FAILURE,
    ) -> CommandSpec:
        return CommandSpec(argv=argv, cwd=self._cwd(cwd), failure_outcome=failure_outcome)

    def _cwd(self, cwd: Path | str | None) -> Path:
        if cwd is None:
            return self.workspace_root
        candidate = Path(cwd)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise CommandPolicyError(f"command cwd resolves outside workspace: {cwd}")
        if not resolved.is_dir():
            raise CommandPolicyError(f"command cwd is not an existing directory: {cwd}")
        return resolved

    def _path_argument(self, path: Path | str, cwd: Path) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise CommandPolicyError(f"path resolves outside workspace: {path}")
        return os.path.relpath(resolved, start=cwd)


def _arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    if isinstance(arguments, (str, bytes)):
        raise CommandPolicyError("command arguments must be a sequence, not a string")
    normalized = tuple(arguments)
    if any(not isinstance(argument, str) or "\0" in argument for argument in normalized):
        raise CommandPolicyError("command arguments must be strings without NUL bytes")
    return normalized


def _validated_root(workspace_root: Path | str) -> Path:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise CommandPolicyError(f"workspace root is not an existing directory: {root}")
    return root
