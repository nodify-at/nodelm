from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nodelm.harness import (
    CommandExecutor,
    CommandPolicy,
    CommandPolicyError,
    CommandResult,
    CommandSpec,
    OutcomeCategory,
    RootlessPodmanExecutor,
    discover_typescript_workspace,
    parse_node_test_count,
)


def test_outcome_categories_are_exhaustive() -> None:
    assert {outcome.value for outcome in OutcomeCategory} == {
        "success",
        "model_failure",
        "test_failure",
        "environment_failure",
        "dependency_install_failure",
        "timeout",
        "tool_protocol_failure",
        "internal_failure",
    }


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("TAP version 13\n1..3\n# tests 3\n# pass 3\n", 3),
        ("\u2139 tests 0\n\u2139 suites 0\n\u2139 pass 0\n", 0),
        ("    # tests 9\n# tests 2\n", 2),
        ("test output without a summary\n", None),
    ],
)
def test_parse_node_test_count(output: str, expected: int | None) -> None:
    assert parse_node_test_count(output) == expected


def test_command_policy_exposes_supported_tools_and_classification(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    policy = CommandPolicy(tmp_path)

    specs = (
        policy.repo_tree(),
        policy.search("needle", paths=("src",)),
        policy.read_file("package.json"),
        policy.git_status(),
        policy.git_diff(paths=("src",)),
        policy.git_log(max_count=5),
        policy.git_show("HEAD"),
        policy.npm(("test",)),
        policy.pnpm(("test",)),
        policy.node(("--version",)),
        policy.node_test(("tests/example.test.js",)),
        policy.jest(("--runInBand",)),
        policy.vitest(("run",)),
        policy.tsc(project="tsconfig.json"),
        policy.eslint(("src",)),
        policy.package_script("npm", "test:unit", arguments=("--runInBand",)),
    )

    assert all(spec.argv and isinstance(spec.argv, tuple) for spec in specs)
    assert policy.npm(("install",)).failure_outcome is OutcomeCategory.DEPENDENCY_INSTALL_FAILURE
    assert policy.pnpm(("install",)).failure_outcome is OutcomeCategory.DEPENDENCY_INSTALL_FAILURE
    assert policy.node_test().failure_outcome is OutcomeCategory.TEST_FAILURE
    assert policy.tsc().argv[-1] == "--noEmit"


def test_generic_command_requires_explicit_trusted_local_opt_in(tmp_path: Path) -> None:
    policy = CommandPolicy(tmp_path)

    with pytest.raises(CommandPolicyError, match="trusted local"):
        policy.generic((sys.executable, "--version"))

    spec = policy.generic((sys.executable, "--version"), trusted_local=True)

    assert spec.argv == (sys.executable, "--version")


def test_repository_path_arguments_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = CommandPolicy(workspace)

    with pytest.raises(CommandPolicyError, match="outside workspace"):
        policy.read_file("../outside.txt")


def test_executor_runs_real_argv_with_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--require=/tmp/untrusted-preload.js")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-inherited")
    executor = CommandExecutor(tmp_path)
    spec = CommandSpec(
        argv=(
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ.get('NODE_OPTIONS', '<missing>')); "
                "print(os.environ.get('AWS_SECRET_ACCESS_KEY', '<missing>'))"
            ),
        )
    )

    result = executor.run(spec)

    assert result.outcome is OutcomeCategory.SUCCESS
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["<missing>", "<missing>"]
    assert result.stderr == ""
    assert result.cwd == tmp_path.resolve()


def test_executor_rejects_cwd_escape_without_starting_process(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    marker = outside / "process-ran"
    executor = CommandExecutor(workspace)
    spec = CommandSpec(
        argv=(sys.executable, "-c", "from pathlib import Path; Path('process-ran').touch()"),
        cwd=outside,
    )

    result = executor.run(spec)

    assert result.outcome is OutcomeCategory.TOOL_PROTOCOL_FAILURE
    assert result.exit_code is None
    assert "outside workspace" in result.stderr
    assert not marker.exists()


def test_executor_times_out_and_terminates_process(tmp_path: Path) -> None:
    executor = CommandExecutor(tmp_path, termination_grace_seconds=0.1)
    spec = CommandSpec(
        argv=(sys.executable, "-c", "import time; time.sleep(30)"),
        timeout_seconds=0.05,
    )

    result = executor.run(spec)

    assert result.outcome is OutcomeCategory.TIMEOUT
    assert result.timed_out is True
    assert result.exit_code is not None
    assert result.duration_seconds < 5


def test_executor_caps_and_redacts_output(tmp_path: Path) -> None:
    executor = CommandExecutor(tmp_path, default_max_output_bytes=80)
    secret = "very-secret-token"
    spec = CommandSpec(
        argv=(
            sys.executable,
            "-c",
            f"print('token={secret}'); print('x' * 5000)",
        ),
        redact_values=(secret,),
    )

    result = executor.run(spec)

    assert result.outcome is OutcomeCategory.SUCCESS
    assert secret not in result.stdout
    assert "[REDACTED]" in result.stdout
    assert result.stdout_truncated is True
    assert len(result.stdout.encode("utf-8")) <= 80


def test_executor_redacts_quoted_and_json_shaped_secrets_from_all_evidence(
    tmp_path: Path,
) -> None:
    stdout_secret = "stdout secret with spaces"
    stderr_secret = "stderr secret with spaces"
    argv_secret = "argv-secret"
    code = (
        "import sys; "
        f'print(\'{{"access_token": "{stdout_secret}"}}\'); '
        f"print(\"password='{stderr_secret}'\", file=sys.stderr)"
    )
    result = CommandExecutor(tmp_path).run(
        CommandSpec(
            argv=(
                sys.executable,
                "-c",
                code,
                f'--payload={{"apiKey":"{argv_secret}"}}',
            )
        )
    )

    evidence = json.dumps(result.to_evidence())

    assert result.outcome is OutcomeCategory.SUCCESS
    assert stdout_secret not in evidence
    assert stderr_secret not in evidence
    assert argv_secret not in evidence
    assert evidence.count("[REDACTED]") >= 3
    assert result.to_evidence()["schema_version"] == "nodelm.command-result/v1"


def test_nonzero_exit_uses_declared_failure_category(tmp_path: Path) -> None:
    policy = CommandPolicy(tmp_path)
    install = policy.dependency_install("npm")
    failing_install = replace(
        install,
        argv=(sys.executable, "-c", "raise SystemExit(42)"),
    )

    result = CommandExecutor(tmp_path).run(failing_install)

    assert result.outcome is OutcomeCategory.DEPENDENCY_INSTALL_FAILURE
    assert result.exit_code == 42


def test_missing_executable_is_an_environment_failure(tmp_path: Path) -> None:
    result = CommandExecutor(tmp_path).run(
        CommandSpec(argv=("nodelm-command-that-does-not-exist",))
    )

    assert result.outcome is OutcomeCategory.ENVIRONMENT_FAILURE
    assert result.exit_code is None
    assert "not found" in result.stderr.lower()


def test_discovers_package_tsconfig_and_declared_workspace_members(tmp_path: Path) -> None:
    root_manifest = {
        "name": "fixture-root",
        "private": True,
        "packageManager": "pnpm@9.15.0",
        "workspaces": ["packages/*"],
    }
    (tmp_path / "package.json").write_text(json.dumps(root_manifest), encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    package_dir = tmp_path / "packages" / "api"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text(
        json.dumps({"name": "@fixture/api", "scripts": {"test": "node --test"}}),
        encoding="utf-8",
    )
    (package_dir / "tsconfig.build.json").write_text("{}", encoding="utf-8")
    ignored_dir = tmp_path / "node_modules" / "ignored"
    ignored_dir.mkdir(parents=True)
    (ignored_dir / "package.json").write_text('{"name":"ignored"}', encoding="utf-8")

    discovery = discover_typescript_workspace(tmp_path)

    assert discovery.package_manager == "pnpm@9.15.0"
    assert [manifest.name for manifest in discovery.package_manifests] == [
        "fixture-root",
        "@fixture/api",
    ]
    assert [manifest.name for manifest in discovery.workspace_packages] == ["@fixture/api"]
    assert discovery.tsconfig_paths == (
        (tmp_path / "packages" / "api" / "tsconfig.build.json").resolve(),
        (tmp_path / "tsconfig.json").resolve(),
    )


def test_discovery_rejects_invalid_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        discover_typescript_workspace(tmp_path)


def test_generic_rootless_sandbox_builds_an_argv_only_read_only_command(
    tmp_path: Path,
) -> None:
    image = f"registry.example/node-tooling@sha256:{'f' * 64}"
    sandbox = RootlessPodmanExecutor(image)

    command = sandbox.command(tmp_path, ("npm", "test", "--", "--runInBand"))

    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--volume=" + str(tmp_path.resolve()) + ":/workspace:ro" in command
    assert "--entrypoint=npm" in command
    assert command[-4:] == (image, "test", "--", "--runInBand")
    assert not any(argument in {"sh", "bash", "-c"} for argument in command)


def test_generic_rootless_sandbox_rejects_an_escaping_working_directory(
    tmp_path: Path,
) -> None:
    sandbox = RootlessPodmanExecutor(f"registry.example/node-tooling@sha256:{'f' * 64}")

    with pytest.raises(ValueError, match="workspace"):
        sandbox.command(tmp_path, ("node", "--version"), cwd="../outside")


def test_generic_rootless_sandbox_force_cleans_a_timed_out_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def command_result(
        argv: tuple[str, ...],
        *,
        outcome: OutcomeCategory = OutcomeCategory.SUCCESS,
        exit_code: int = 0,
        stdout: str = "",
        timed_out: bool = False,
    ) -> CommandResult:
        return CommandResult(
            argv=argv,
            cwd=tmp_path,
            outcome=outcome,
            exit_code=exit_code,
            stdout=stdout,
            stderr="",
            duration_seconds=0.01,
            timed_out=timed_out,
        )

    def fake_run(_executor: CommandExecutor, spec: CommandSpec) -> CommandResult:
        argv = spec.argv
        calls.append(argv)
        if argv[1:3] == ("info", "--format"):
            return command_result(argv, stdout="true\n")
        if argv[1:3] == ("image", "exists"):
            return command_result(argv)
        if argv[1] == "run":
            return command_result(
                argv,
                outcome=OutcomeCategory.TIMEOUT,
                exit_code=-9,
                timed_out=True,
            )
        return command_result(argv)

    monkeypatch.setattr("nodelm.harness.sandbox.CommandExecutor.run", fake_run)
    sandbox = RootlessPodmanExecutor(f"registry.example/node-tooling@sha256:{'f' * 64}")

    result = sandbox.run(tmp_path, ("node", "malicious.js"))

    cleanup = calls[-1]
    assert result.outcome is OutcomeCategory.TIMEOUT
    assert cleanup[1:6] == ("rm", "--force", "--ignore", "--time=0", "--")
    assert cleanup[-1].startswith("nodelm-")
    assert sandbox.evidence()["cleanup"] is not None


def test_generic_rootless_sandbox_force_cleans_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def command_result(argv: tuple[str, ...], *, stdout: str = "") -> CommandResult:
        return CommandResult(
            argv=argv,
            cwd=tmp_path,
            outcome=OutcomeCategory.SUCCESS,
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_seconds=0.01,
        )

    def fake_run(_executor: CommandExecutor, spec: CommandSpec) -> CommandResult:
        argv = spec.argv
        calls.append(argv)
        if argv[1:3] == ("info", "--format"):
            return command_result(argv, stdout="true\n")
        if argv[1:3] == ("image", "exists"):
            return command_result(argv)
        if argv[1] == "run":
            raise KeyboardInterrupt
        return command_result(argv)

    monkeypatch.setattr("nodelm.harness.sandbox.CommandExecutor.run", fake_run)
    sandbox = RootlessPodmanExecutor(f"registry.example/node-tooling@sha256:{'f' * 64}")

    with pytest.raises(KeyboardInterrupt):
        sandbox.run(tmp_path, ("node", "malicious.js"))

    assert calls[-1][1:6] == ("rm", "--force", "--ignore", "--time=0", "--")
    assert sandbox.evidence()["cleanup"] is not None
