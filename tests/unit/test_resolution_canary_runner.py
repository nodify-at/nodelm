from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_resolution_canary.sh"
CHROOT_LAUNCHER = PROJECT_ROOT / "src" / "nodelm" / "evaluation" / "chroot_launcher.py"


def test_resolution_canary_runner_is_valid_bash_with_durable_operator_state() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        check=False,
        capture_output=True,
        text=True,
    )
    source = RUNNER.read_text(encoding="utf-8")

    assert syntax.returncode == 0, syntax.stderr
    assert 'write_state "RUNNING"' in source
    assert 'write_state "STOPPED"' in source
    assert 'write_state "COMPLETE"' in source
    assert "private-case-evidence" in source
    assert "lock-resolution-canary-images" in source
    assert "run-resolution-canary" in source
    assert "kill -TERM" in source


def test_resolution_canary_runner_requires_rootless_execution_and_offline_tests() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '[[ "${EUID}" -ne 0 ]]' in source
    assert "export HF_HUB_OFFLINE=1" in source
    assert "export UV_OFFLINE=1" in source
    assert "SWE-rebench-V2-${EVALUATOR_REVISION}" in source
    assert "pulling selected canary images" in source
    assert "running offline real-repository canary attempts" in source


def test_resolution_canary_runner_supports_restricted_runpod_seccomp_chroot() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'CANARY_RUNTIME="${NODELM_CANARY_RUNTIME:-rootless-podman}"' in source
    assert "seccomp-chroot)" in source
    assert "require_command skopeo" in source
    assert "require_command umoci" in source
    assert '--runtime "${CANARY_RUNTIME}"' in source
    assert "execution.sandbox_backend != lock.runtime" in source
    assert 'log_event "REUSE phase=materialize' in source


def test_seccomp_chroot_launcher_contains_fail_closed_isolation_controls() -> None:
    source = CHROOT_LAUNCHER.read_text(encoding="utf-8")

    assert 'os.chroot(".")' in source
    assert "os.setuid(uid)" in source
    assert "_PR_SET_NO_NEW_PRIVS" in source
    assert "_SCMP_CMP_NE" in source
    assert "_AF_UNIX = 1" in source
    assert '"sched_setaffinity"' in source
    assert "resource.RLIMIT_AS" in source
    assert "process.children(recursive=True)" in source
