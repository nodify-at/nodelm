from __future__ import annotations

import subprocess
from pathlib import Path

from nodelm.harness import CommandResult, OutcomeCategory
from nodelm.models import SolveContext, VerificationStatus
from nodelm.teacher.execution import LocalGitTeacherRunner
from nodelm.teacher.rollout import RolloutAttempt


class FakeIsolatedRepositorySandbox:
    def __init__(self) -> None:
        self.calls = 0
        self.options: dict[str, object] = {}

    def run(
        self,
        workspace: Path,
        argv: tuple[str, ...],
        **options: object,
    ) -> CommandResult:
        self.calls += 1
        self.options = options
        repaired = "return left * right;" in (workspace / "src/math.js").read_text(encoding="utf-8")
        return CommandResult(
            argv=("fake-isolated-sandbox", *argv),
            cwd=workspace,
            outcome=(OutcomeCategory.SUCCESS if repaired else OutcomeCategory.TEST_FAILURE),
            exit_code=0 if repaired else 1,
            stdout="# tests 1\n",
            stderr="",
            duration_seconds=0.01,
        )

    def evidence(self) -> dict[str, object]:
        return {"backend": "fake-isolated-repository-sandbox"}


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str, Path]:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "NodeLM fixture")
    _git(repository, "config", "user.email", "fixture@nodelm.invalid")
    (repository / "src").mkdir()
    (repository / "test").mkdir()
    (repository / "src/math.js").write_text(
        "export function multiply(left, right) {\n  return left + right;\n}\n",
        encoding="utf-8",
    )
    (repository / "test/math.test.js").write_text(
        "// protected fixture test\n",
        encoding="utf-8",
    )
    host_marker = tmp_path / "host-side-effect"
    (repository / ".nodelm-teacher.patch").symlink_to(host_marker)
    _git(repository, "add", "src/math.js", "test/math.test.js", ".nodelm-teacher.patch")
    _git(repository, "commit", "-m", "fixture base")
    return repository, _git(repository, "rev-parse", "HEAD"), host_marker


def test_teacher_runner_restores_applies_and_tests_in_an_isolated_workspace(
    tmp_path: Path,
) -> None:
    repository, base_commit, host_marker = _source_repository(tmp_path)
    sandbox = FakeIsolatedRepositorySandbox()
    runner = LocalGitTeacherRunner(
        source_checkout=repository,
        expected_repository="acme/widget",
        sandbox=sandbox,
        test_argv=("node", "--test", "test/math.test.js"),
        protected_paths=("test/math.test.js",),
    )
    attempt = RolloutAttempt(
        concise_state="repair multiply",
        tool_calls=(),
        observations=(),
        patch="""diff --git a/src/math.js b/src/math.js
--- a/src/math.js
+++ b/src/math.js
@@ -1,3 +1,3 @@
 export function multiply(left, right) {
-  return left + right;
+  return left * right;
 }
""",
        generation_status=VerificationStatus.PASS,
        status=VerificationStatus.UNVERIFIED,
        result_summary="generated",
    )
    context = SolveContext(
        repository="acme/widget",
        base_commit=base_commit,
        task="Fix multiply",
    )

    result = runner.execute(context, attempt)

    assert result.status is VerificationStatus.UNVERIFIED
    assert result.task_resolved is None
    assert result.regression_tests_passed is None
    assert [check.name for check in result.execution_results] == [
        "base_restore",
        "patch_apply",
        "tests",
    ]
    assert result.execution_results[-1].status is VerificationStatus.UNVERIFIED
    assert sandbox.calls == 1
    assert sandbox.options["writable_workspace"] is False
    assert "left + right" in (repository / "src/math.js").read_text(encoding="utf-8")
    assert not host_marker.exists()


def test_teacher_runner_rejects_symlinks_in_protected_path_components(
    tmp_path: Path,
) -> None:
    repository, _, _ = _source_repository(tmp_path)
    host_tests = tmp_path / "host-tests"
    host_tests.mkdir()
    (host_tests / "secret.test.js").write_text("host content\n", encoding="utf-8")
    (repository / "test/current").symlink_to(host_tests, target_is_directory=True)
    _git(repository, "add", "test/current")
    _git(repository, "commit", "-m", "add malicious protected-path symlink")
    base_commit = _git(repository, "rev-parse", "HEAD")
    sandbox = FakeIsolatedRepositorySandbox()
    runner = LocalGitTeacherRunner(
        source_checkout=repository,
        expected_repository="acme/widget",
        sandbox=sandbox,
        test_argv=("node", "--test", "test/current/secret.test.js"),
        protected_paths=("test/current/secret.test.js",),
    )
    attempt = RolloutAttempt(
        concise_state="repair multiply",
        tool_calls=(),
        observations=(),
        patch="""diff --git a/src/math.js b/src/math.js
--- a/src/math.js
+++ b/src/math.js
@@ -1,3 +1,3 @@
 export function multiply(left, right) {
-  return left + right;
+  return left * right;
 }
""",
        generation_status=VerificationStatus.PASS,
        status=VerificationStatus.UNVERIFIED,
        result_summary="generated",
    )
    context = SolveContext(
        repository="acme/widget",
        base_commit=base_commit,
        task="Fix multiply",
    )

    result = runner.execute(context, attempt)

    assert result.status is VerificationStatus.FAIL
    assert "protected tests are unavailable" in result.result_summary
    assert sandbox.calls == 0
