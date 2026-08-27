from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_resolution_recovery.sh"

LABELED_PARTITIONS = (
    "openhands/minimax_m25/swe-rebench-v2",
    "openhands/qwen35_122b/swe-rebench-v2",
    "sweagent/minimax_m25/swe-rebench-v2",
    "sweagent/qwen35_122b/swe-rebench-v2",
)
TARGET_PARTITIONS = (
    "minisweagent/qwen36_27b/swe-rebench-v2",
    "openhands/qwen36_27b/swe-rebench-v2",
    "sweagent/qwen36_27b/swe-rebench-v2",
)


@dataclass(frozen=True)
class RunnerHarness:
    repo: Path
    persist: Path
    commit: str
    environment: dict[str, str]
    uv_log: Path

    @property
    def run_dir(self) -> Path:
        return self.persist / "derived" / f"resolution-recovery-{self.commit}"

    def run(
        self, environment_overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / RUNNER.name)],
            cwd=self.repo,
            env={**self.environment, **(environment_overrides or {})},
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )

    def uv_calls(self) -> list[dict[str, object]]:
        if not self.uv_log.exists():
            return []
        return [json.loads(line) for line in self.uv_log.read_text().splitlines()]


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _git(repo: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "NodeLM resolution runner test",
        "GIT_AUTHOR_EMAIL": "resolution-runner-test@example.invalid",
        "GIT_COMMITTER_NAME": "NodeLM resolution runner test",
        "GIT_COMMITTER_EMAIL": "resolution-runner-test@example.invalid",
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_fake_sha256sum(path: Path) -> None:
    _write(
        path,
        f"""#!{sys.executable}
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

target = Path(sys.argv[-1])
digest = hashlib.sha256(target.read_bytes()).hexdigest()
print(f"{{digest}}  {{target}}")
""",
    )
    path.chmod(0o755)


def _make_fake_setsid(path: Path) -> None:
    _write(
        path,
        f"""#!{sys.executable}
from __future__ import annotations

import os
import sys

arguments = sys.argv[1:]
if arguments and arguments[0] == "--wait":
    arguments = arguments[1:]
os.setsid()
os.execvp(arguments[0], arguments)
""",
    )
    path.chmod(0o755)


def _make_fake_uv(path: Path) -> None:
    _write(
        path,
        f"""#!{sys.executable}
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
record = {{
    "args": arguments,
    "offline": os.environ.get("UV_OFFLINE"),
    "no_sync": os.environ.get("UV_NO_SYNC"),
    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
    "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    "pythonpath": os.environ.get("PYTHONPATH"),
}}
with Path(os.environ["FAKE_UV_LOG"]).open("a") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\\n")

if os.environ.get("FAKE_UV_LONG_RUNNING") == "1":
    import subprocess
    import time

    Path(os.environ["FAKE_UV_PID_FILE"]).write_text(str(os.getpid()))
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
    )
    Path(os.environ["FAKE_UV_CHILD_PID_FILE"]).write_text(str(child.pid))
    time.sleep(300)

python_index = arguments.index("python")
python_arguments = arguments[python_index + 1 :]
if python_arguments[0] == "-":
    sys.stdin.read()
    if os.environ.get("FAKE_UV_REJECT_VALIDATOR") == "1":
        raise SystemExit(93)
    raise SystemExit(0)

if python_arguments[:4] != ["-m", "nodelm", "datasets", "build-resolution-recovery"]:
    raise SystemExit(92)
command_arguments = python_arguments[4:]
if os.environ.get("FAKE_UV_PRODUCER_FAIL") == "1":
    raise SystemExit(17)

def value(flag: str) -> Path:
    return Path(command_arguments[command_arguments.index(flag) + 1])

for flag in ("--candidates-output", "--queue-output", "--manifest-output"):
    output = value(flag)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "{{}}\\n"
    if output.exists():
        if output.read_text() != payload:
            raise SystemExit(94)
        continue
    output.write_text(payload)
""",
    )
    path.chmod(0o755)


@pytest.fixture
def runner_harness(tmp_path: Path) -> RunnerHarness:
    repo = tmp_path / "repo"
    persist = tmp_path / "persist"
    fake_bin = tmp_path / "fake-bin"
    repo.mkdir()
    fake_bin.mkdir()

    (repo / "scripts").mkdir()
    shutil.copy2(RUNNER, repo / "scripts" / RUNNER.name)
    _write(repo / "pyproject.toml", "[project]\nname = 'runner-test'\nversion = '0'\n")
    _write(repo / "uv.lock", "version = 1\nrevision = 3\n")
    _write(repo / "src" / "nodelm" / "__init__.py", '"""Exact archived fixture."""\n')
    (repo / "configs" / "datasets").mkdir(parents=True)
    shutil.copy2(
        PROJECT_ROOT / "configs" / "datasets" / "registry.yaml",
        repo / "configs" / "datasets" / "registry.yaml",
    )
    shutil.copy2(
        PROJECT_ROOT / "configs" / "datasets" / "open-swe-trace-partitions.yaml",
        repo / "configs" / "datasets" / "open-swe-trace-partitions.yaml",
    )
    (repo / "scripts" / RUNNER.name).chmod(0o755)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "runner fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    _write(persist / "receipts" / "open-swe-traces.transfer.json", "{}\n")
    (persist / "snapshots" / "open-swe-traces").mkdir(parents=True)

    _make_fake_sha256sum(fake_bin / "sha256sum")
    _make_fake_setsid(fake_bin / "setsid")
    _write(
        fake_bin / "flock",
        '#!/usr/bin/env bash\n[[ "${FAKE_FLOCK_FAIL:-0}" != 1 ]]\n',
    )
    (fake_bin / "flock").chmod(0o755)
    fake_uv = fake_bin / "uv"
    _make_fake_uv(fake_uv)
    uv_log = tmp_path / "uv-calls.jsonl"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "NODELM_REPO_ROOT": str(repo),
        "NODELM_PERSIST_ROOT": str(persist),
        "NODELM_EXEC_TMP_ROOT": str(tmp_path / "execution-trees"),
        "NODELM_UV_BIN": str(fake_uv),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
        "FAKE_UV_LOG": str(uv_log),
    }
    python_path = Path(environment["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)
    return RunnerHarness(repo, persist, commit, environment, uv_log)


def _producer_call(calls: list[dict[str, object]]) -> dict[str, object]:
    return next(call for call in calls if "-m" in call["args"])


def _execution_root(call: dict[str, object]) -> Path:
    arguments = call["args"]
    assert isinstance(arguments, list)
    directory_index = arguments.index("--directory") + 1
    return Path(str(arguments[directory_index]))


def _establish_run_binding(runner_harness: RunnerHarness) -> None:
    stop_file = runner_harness.run_dir / "STOP"
    _write(stop_file, "establish exact run binding\n")
    result = runner_harness.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "state=STOPPED" in (runner_harness.run_dir / "run.state").read_text()
    stop_file.unlink()


def test_runner_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_embedded_terminal_validator_has_valid_python_syntax() -> None:
    script = RUNNER.read_text()
    marker = "<<'PY'\n"
    assert script.count(marker) == 1
    validator_and_tail = script.split(marker, maxsplit=1)[1]
    validator = validator_and_tail.split("\nPY\n", maxsplit=1)[0]

    compile(validator, f"{RUNNER}:embedded-validator", "exec")


def test_fake_run_is_commit_bound_offline_and_resumes_after_validation(
    runner_harness: RunnerHarness,
) -> None:
    first = runner_harness.run()

    assert first.returncode == 0, first.stdout + first.stderr
    state = (runner_harness.run_dir / "run.state").read_text()
    assert "state=COMPLETE" in state
    assert f"commit={runner_harness.commit}" in state
    assert "phase=complete" in state
    assert "pid=" in state
    binding = (runner_harness.run_dir / "run.binding").read_text()
    assert "format=nodelm-resolution-recovery-run-binding-v1" in binding
    assert f"commit={runner_harness.commit}" in binding
    first_calls = runner_harness.uv_calls()
    assert len(first_calls) == 2
    producer = _producer_call(first_calls)
    arguments = producer["args"]
    for partition in LABELED_PARTITIONS:
        index = arguments.index(partition)
        assert arguments[index - 1] == "--labeled-partition"
    for partition in TARGET_PARTITIONS:
        index = arguments.index(partition)
        assert arguments[index - 1] == "--target-partition"
    assert arguments.count("--language") == 2
    assert "TypeScript" in arguments
    assert "JavaScript" in arguments
    assert not any(token in arguments for token in ("evaluate", "evaluator", "docker", "gpu"))
    execution_roots = {_execution_root(call) for call in first_calls}
    assert len(execution_roots) == 1
    execution_root = execution_roots.pop()
    assert execution_root != runner_harness.repo
    assert execution_root.parent.parent == Path(runner_harness.environment["NODELM_EXEC_TMP_ROOT"])
    for call in first_calls:
        assert call["offline"] == "1"
        assert call["no_sync"] == "1"
        assert call["hf_hub_offline"] == "1"
        assert call["hf_datasets_offline"] == "1"
        assert call["transformers_offline"] == "1"
        assert call["pythonpath"] == str(execution_root / "src")
    assert not execution_root.parent.exists()

    runner_harness.uv_log.unlink()
    second = runner_harness.run()

    assert second.returncode == 0, second.stdout + second.stderr
    second_calls = runner_harness.uv_calls()
    assert len(second_calls) == 1
    assert all("-m" not in call["args"] for call in second_calls)
    assert (
        "RESUME validated terminal artifacts" in (runner_harness.run_dir / "events.log").read_text()
    )


def test_dirty_repo_is_rejected_before_any_python_process(
    runner_harness: RunnerHarness,
) -> None:
    _write(runner_harness.repo / "untracked.txt", "operator work\n")

    result = runner_harness.run()

    assert result.returncode == 1
    assert "production tree must be completely clean" in result.stdout
    assert runner_harness.uv_calls() == []


def test_unbound_or_partial_outputs_are_rejected_without_overwrite(
    runner_harness: RunnerHarness,
) -> None:
    _write(runner_harness.run_dir / "exact-resolution-candidates.jsonl", "partial\n")

    result = runner_harness.run()

    assert result.returncode == 1
    assert "terminal artifacts exist without a commit-bound run binding" in result.stdout
    assert runner_harness.uv_calls() == []
    assert (runner_harness.run_dir / "exact-resolution-candidates.jsonl").read_text() == "partial\n"


def test_bound_partial_artifacts_resume_through_byte_identical_immutable_reuse(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    candidate = runner_harness.run_dir / "exact-resolution-candidates.jsonl"
    _write(candidate, "{}\n")

    result = runner_harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert candidate.read_text() == "{}\n"
    assert (runner_harness.run_dir / "resolution-evaluation-queue.jsonl").read_text() == "{}\n"
    assert (runner_harness.run_dir / "resolution-recovery.manifest.json").read_text() == "{}\n"
    assert "state=COMPLETE" in (runner_harness.run_dir / "run.state").read_text()
    assert "RESUME bound_partial_artifacts=1" in (runner_harness.run_dir / "events.log").read_text()
    assert len(runner_harness.uv_calls()) == 2


def test_bound_differing_partial_artifact_is_not_overwritten(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    candidate = runner_harness.run_dir / "exact-resolution-candidates.jsonl"
    _write(candidate, "different\n")

    result = runner_harness.run()

    assert result.returncode == 1
    assert "producer exited 94" in result.stdout
    assert candidate.read_text() == "different\n"
    assert not (runner_harness.run_dir / "resolution-evaluation-queue.jsonl").exists()
    assert not (runner_harness.run_dir / "resolution-recovery.manifest.json").exists()


def test_stop_sentinel_records_stopped_state_before_producer(
    runner_harness: RunnerHarness,
) -> None:
    stop_file = runner_harness.run_dir / "STOP"
    _write(stop_file, "operator stop\n")

    result = runner_harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    state = (runner_harness.run_dir / "run.state").read_text()
    assert "state=STOPPED" in state
    assert "phase=stopped" in state
    assert f"stop_file={stop_file}" in state
    assert "kill -TERM pid during build" in state
    assert runner_harness.uv_calls() == []
    assert not (runner_harness.run_dir / "exact-resolution-candidates.jsonl").exists()
    assert not (runner_harness.run_dir / "resolution-evaluation-queue.jsonl").exists()
    assert not (runner_harness.run_dir / "resolution-recovery.manifest.json").exists()


def test_stop_sentinel_preserves_bound_partial_artifacts_without_producer(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    candidate = runner_harness.run_dir / "exact-resolution-candidates.jsonl"
    _write(candidate, "{}\n")
    _write(runner_harness.run_dir / "STOP", "operator stop\n")

    result = runner_harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    state = (runner_harness.run_dir / "run.state").read_text()
    assert "state=STOPPED" in state
    assert "terminal_artifacts_present=1" in state
    assert candidate.read_text() == "{}\n"
    assert runner_harness.uv_calls() == []


def test_stop_sentinel_cannot_bypass_mismatched_run_binding(
    runner_harness: RunnerHarness,
) -> None:
    _write(runner_harness.run_dir / "STOP", "operator stop\n")
    _write(
        runner_harness.run_dir / "run.binding",
        "format=nodelm-resolution-recovery-run-binding-v1\ncommit=wrong\n",
    )

    result = runner_harness.run()

    assert result.returncode == 1
    assert "run binding does not match the exact code and sealed inputs" in result.stdout
    state = (runner_harness.run_dir / "run.state").read_text()
    assert "state=FAILED" in state
    assert "state=STOPPED" not in state
    assert runner_harness.uv_calls() == []


def test_term_stops_the_complete_producer_process_group(
    runner_harness: RunnerHarness,
    tmp_path: Path,
) -> None:
    uv_pid_file = tmp_path / "long-uv.pid"
    child_pid_file = tmp_path / "long-child.pid"
    process = subprocess.Popen(
        ["bash", str(runner_harness.repo / "scripts" / RUNNER.name)],
        cwd=runner_harness.repo,
        env={
            **runner_harness.environment,
            "FAKE_UV_LONG_RUNNING": "1",
            "FAKE_UV_PID_FILE": str(uv_pid_file),
            "FAKE_UV_CHILD_PID_FILE": str(child_pid_file),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tracked_pids: list[int] = []
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if uv_pid_file.exists() and child_pid_file.exists():
                tracked_pids = [
                    int(uv_pid_file.read_text()),
                    int(child_pid_file.read_text()),
                ]
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        assert tracked_pids, "fake producer did not reach its long-running child"

        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)

        assert process.returncode == 130, stdout + stderr
        state = (runner_harness.run_dir / "run.state").read_text()
        assert "state=STOPPED" in state
        for tracked_pid in tracked_pids:
            for _ in range(100):
                try:
                    os.kill(tracked_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                pytest.fail(f"producer descendant survived TERM: pid={tracked_pid}")
        assert not (runner_harness.run_dir / "exact-resolution-candidates.jsonl").exists()
        assert not (runner_harness.run_dir / "resolution-evaluation-queue.jsonl").exists()
        assert not (runner_harness.run_dir / "resolution-recovery.manifest.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for tracked_pid in tracked_pids:
            with suppress(ProcessLookupError):
                os.kill(tracked_pid, signal.SIGKILL)


def test_invalid_resume_artifacts_fail_closed(runner_harness: RunnerHarness) -> None:
    first = runner_harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    runner_harness.uv_log.unlink()

    second = runner_harness.run({"FAKE_UV_REJECT_VALIDATOR": "1"})

    assert second.returncode == 1
    assert "terminal recovery artifacts are invalid" in second.stdout
    assert "state=FAILED" in (runner_harness.run_dir / "run.state").read_text()
    assert len(runner_harness.uv_calls()) == 1


def test_changed_sealed_input_invalidates_run_binding(
    runner_harness: RunnerHarness,
) -> None:
    first = runner_harness.run()
    assert first.returncode == 0, first.stdout + first.stderr
    runner_harness.uv_log.unlink()
    _write(
        runner_harness.persist / "receipts" / "open-swe-traces.transfer.json",
        '{"changed":true}\n',
    )

    second = runner_harness.run()

    assert second.returncode == 1
    assert "run binding does not match the exact code and sealed inputs" in second.stdout
    assert runner_harness.uv_calls() == []


def test_producer_failure_records_failed_state(runner_harness: RunnerHarness) -> None:
    result = runner_harness.run({"FAKE_UV_PRODUCER_FAIL": "1"})

    assert result.returncode == 1
    state = (runner_harness.run_dir / "run.state").read_text()
    assert "state=FAILED" in state
    assert "phase=build" in state
    assert "producer exited 17" in state
    assert len(runner_harness.uv_calls()) == 1


def test_single_run_lock_is_fail_closed(runner_harness: RunnerHarness) -> None:
    result = runner_harness.run({"FAKE_FLOCK_FAIL": "1"})

    assert result.returncode == 1
    assert "another recovery runner owns" in result.stdout
    assert runner_harness.uv_calls() == []


def test_noncanonical_lock_override_is_rejected_before_opening_it(
    runner_harness: RunnerHarness,
    tmp_path: Path,
) -> None:
    alternate_lock = tmp_path / "alternate.lock"

    result = runner_harness.run({"NODELM_LOCK_FILE": str(alternate_lock)})

    assert result.returncode == 1
    assert "NODELM_LOCK_FILE must equal the canonical single-run lock" in result.stdout
    assert not alternate_lock.exists()
    assert runner_harness.uv_calls() == []
