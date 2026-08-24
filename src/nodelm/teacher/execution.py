from __future__ import annotations

import hashlib
import stat
import tempfile
from pathlib import Path
from typing import Protocol

from nodelm.harness import CommandExecutor, CommandPolicy, CommandResult, OutcomeCategory
from nodelm.harness.patches import validate_text_git_patch
from nodelm.harness.sandbox import SandboxUnavailableError
from nodelm.models import CheckResult, SolveContext, VerificationStatus
from nodelm.teacher.rollout import RolloutAttempt

_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
}


class RepositorySandbox(Protocol):
    def run(
        self,
        workspace: Path,
        argv: tuple[str, ...],
        *,
        cwd: Path | str = ".",
        writable_workspace: bool = False,
        timeout_seconds: float = 60,
        failure_outcome: OutcomeCategory = OutcomeCategory.ENVIRONMENT_FAILURE,
    ) -> CommandResult: ...

    def evidence(self) -> dict[str, object]: ...


def _protected_file_digest(workspace: Path, relative_path: str) -> str | None:
    root = workspace.resolve()
    candidate = root
    for part in Path(relative_path).parts:
        candidate /= part
        try:
            metadata = candidate.lstat()
        except OSError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root) or not stat.S_ISREG(candidate.lstat().st_mode):
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _updated_attempt(attempt: RolloutAttempt, **updates: object) -> RolloutAttempt:
    return RolloutAttempt.model_validate({**attempt.model_dump(mode="json"), **updates})


class LocalGitTeacherRunner:
    """Restore a local Git source, apply a teacher patch, and test it in a sandbox."""

    def __init__(
        self,
        *,
        source_checkout: Path,
        expected_repository: str,
        sandbox: RepositorySandbox,
        test_argv: tuple[str, ...],
        protected_paths: tuple[str, ...],
    ) -> None:
        source = source_checkout.resolve()
        if not source.is_dir() or not (source / ".git").exists():
            raise ValueError("teacher source checkout must be an existing Git repository")
        if not expected_repository.strip():
            raise ValueError("expected repository must be non-empty")
        if not test_argv:
            raise ValueError("teacher runner requires an explicit test command")
        if not protected_paths:
            raise ValueError("teacher runner requires explicit protected test paths")
        for path in protected_paths:
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("protected paths must be contained relative paths")
        self.source_checkout = source
        self.expected_repository = expected_repository.strip()
        self.sandbox = sandbox
        self.test_argv = test_argv
        self.protected_paths = protected_paths

    def execute(self, context: SolveContext, attempt: RolloutAttempt) -> RolloutAttempt:
        if context.repository != self.expected_repository:
            raise ValueError("teacher context does not match the configured repository")
        if attempt.generation_status is not VerificationStatus.PASS:
            return _updated_attempt(
                attempt,
                status=VerificationStatus.FAIL,
                execution_results=(),
                task_resolved=False,
                regression_tests_passed=False,
                result_summary="teacher generation did not pass",
            )
        try:
            validate_text_git_patch(attempt.patch)
        except ValueError as error:
            return _updated_attempt(
                attempt,
                status=VerificationStatus.FAIL,
                execution_results=(
                    CheckResult(
                        name="patch_apply",
                        status=VerificationStatus.FAIL,
                        summary=f"teacher patch was rejected before execution: {error}",
                    ),
                ),
                task_resolved=False,
                regression_tests_passed=False,
                result_summary="teacher patch did not pass text-patch validation",
            )

        checks: list[CheckResult] = []
        with tempfile.TemporaryDirectory(prefix="nodelm-teacher-run-") as temporary_directory:
            root = Path(temporary_directory).resolve()
            workspace = root / "workspace"
            policy = CommandPolicy(root)
            executor = CommandExecutor(root)
            clone = executor.run(
                policy.generic(
                    (
                        "git",
                        "clone",
                        "--no-checkout",
                        "--no-hardlinks",
                        "--no-local",
                        "--",
                        str(self.source_checkout),
                        workspace.name,
                    ),
                    trusted_local=True,
                    env=_GIT_ENVIRONMENT,
                )
            )
            if clone.outcome is not OutcomeCategory.SUCCESS:
                return self._failed_attempt(attempt, "base_restore", clone.to_evidence())

            checkout = executor.run(
                policy.generic(
                    (
                        "git",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "checkout",
                        "--detach",
                        context.base_commit,
                    ),
                    trusted_local=True,
                    cwd=workspace,
                    env=_GIT_ENVIRONMENT,
                )
            )
            head = executor.run(
                policy.generic(
                    ("git", "rev-parse", "HEAD"),
                    trusted_local=True,
                    cwd=workspace,
                    env=_GIT_ENVIRONMENT,
                )
            )
            restored = (
                checkout.outcome is OutcomeCategory.SUCCESS
                and head.outcome is OutcomeCategory.SUCCESS
                and head.stdout.strip().casefold() == context.base_commit.casefold()
            )
            checks.append(
                CheckResult(
                    name="base_restore",
                    status=(VerificationStatus.PASS if restored else VerificationStatus.FAIL),
                    summary=(
                        "exact base commit restored"
                        if restored
                        else "failed to restore the exact base commit"
                    ),
                    evidence={
                        "clone": clone.to_evidence(),
                        "checkout": checkout.to_evidence(),
                        "head": head.to_evidence(),
                    },
                )
            )
            if not restored:
                return self._with_checks(attempt, checks, "base restore failed")

            protected_before = {
                path: _protected_file_digest(workspace, path) for path in self.protected_paths
            }
            if any(digest is None for digest in protected_before.values()):
                checks.append(
                    CheckResult(
                        name="patch_apply",
                        status=VerificationStatus.FAIL,
                        summary="a configured protected test path is missing or unsafe",
                    )
                )
                return self._with_checks(attempt, checks, "protected tests are unavailable")

            patch_path = root / "teacher.patch"
            patch_path.write_text(attempt.patch, encoding="utf-8")
            apply_result = executor.run(
                policy.generic(
                    ("git", "apply", "--whitespace=error-all", str(patch_path)),
                    trusted_local=True,
                    cwd=workspace,
                    failure_outcome=OutcomeCategory.MODEL_FAILURE,
                    env=_GIT_ENVIRONMENT,
                )
            )
            protected_after = {
                path: _protected_file_digest(workspace, path) for path in self.protected_paths
            }
            patch_applied = (
                apply_result.outcome is OutcomeCategory.SUCCESS
                and protected_after == protected_before
            )
            checks.append(
                CheckResult(
                    name="patch_apply",
                    status=(VerificationStatus.PASS if patch_applied else VerificationStatus.FAIL),
                    summary=(
                        "teacher patch applied without modifying protected tests"
                        if patch_applied
                        else "teacher patch failed or modified protected tests"
                    ),
                    evidence={"command": apply_result.to_evidence()},
                )
            )
            if not patch_applied:
                return self._with_checks(attempt, checks, "patch application failed")

            try:
                test_result = self.sandbox.run(
                    workspace,
                    self.test_argv,
                    writable_workspace=False,
                    failure_outcome=OutcomeCategory.TEST_FAILURE,
                )
            except SandboxUnavailableError as error:
                checks.append(
                    CheckResult(
                        name="tests",
                        status=VerificationStatus.BLOCKED,
                        summary=str(error),
                        evidence={"sandbox": self.sandbox.evidence()},
                    )
                )
                return self._with_checks(attempt, checks, "sandboxed tests were blocked")
            command_succeeded = test_result.outcome is OutcomeCategory.SUCCESS
            checks.append(
                CheckResult(
                    name="tests",
                    status=(
                        VerificationStatus.UNVERIFIED
                        if command_succeeded
                        else VerificationStatus.FAIL
                    ),
                    summary=(
                        "sandboxed test command exited successfully; generic repository tests "
                        "are not an integrity-attested oracle"
                        if command_succeeded
                        else (
                            f"sandboxed repository test command failed: {test_result.outcome.value}"
                        )
                    ),
                    evidence={
                        "command": test_result.to_evidence(),
                        "sandbox": self.sandbox.evidence(),
                    },
                )
            )
            return _updated_attempt(
                attempt,
                status=(
                    VerificationStatus.UNVERIFIED if command_succeeded else VerificationStatus.FAIL
                ),
                execution_results=tuple(checks),
                task_resolved=None if command_succeeded else False,
                regression_tests_passed=None if command_succeeded else False,
                result_summary=(
                    "teacher patch executed successfully but awaits an integrity-attested oracle"
                    if command_succeeded
                    else "teacher patch failed its sandboxed repository test command"
                ),
            )

    @staticmethod
    def _failed_attempt(
        attempt: RolloutAttempt,
        stage: str,
        evidence: dict[str, object],
    ) -> RolloutAttempt:
        return _updated_attempt(
            attempt,
            status=VerificationStatus.FAIL,
            execution_results=(
                CheckResult(
                    name=stage,
                    status=VerificationStatus.FAIL,
                    summary=f"{stage} command failed",
                    evidence=evidence,
                ),
            ),
            task_resolved=False,
            regression_tests_passed=False,
            result_summary=f"{stage} failed",
        )

    @staticmethod
    def _with_checks(
        attempt: RolloutAttempt,
        checks: list[CheckResult],
        summary: str,
    ) -> RolloutAttempt:
        status = (
            VerificationStatus.BLOCKED
            if any(check.status is VerificationStatus.BLOCKED for check in checks)
            else VerificationStatus.FAIL
        )
        return _updated_attempt(
            attempt,
            status=status,
            execution_results=tuple(checks),
            task_resolved=False,
            regression_tests_passed=False,
            result_summary=summary,
        )
