from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_full_normalization.sh"


@dataclass(frozen=True)
class RunnerHarness:
    repo: Path
    persist: Path
    commit: str
    environment: dict[str, str]
    uv_log: Path

    @property
    def run_dir(self) -> Path:
        return self.persist / "derived" / f"full-normalization-{self.commit}"

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
            timeout=30,
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
        "GIT_AUTHOR_NAME": "NodeLM runner test",
        "GIT_AUTHOR_EMAIL": "runner-test@example.invalid",
        "GIT_COMMITTER_NAME": "NodeLM runner test",
        "GIT_COMMITTER_EMAIL": "runner-test@example.invalid",
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
        """#!/usr/bin/env bash
set -eu
target="${!#}"
case "${target}" in
  */registry.yaml)
    digest=f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f ;;
  */open-swe-trace-partitions.yaml)
    digest=aec2ae095a926dda09a5fe3eefede7a59fbd494b24fffd503fff4cb366b389b5 ;;
  */open-swe-traces.transfer.json)
    digest=44ea157ebd802a5604301c82e8785003d67f90d0ed64efcc079059dfd4290a84 ;;
  */swe-rebench-v2.transfer.json)
    digest=fbcd4fbb2b9c4b887ef15f368f3673c07d82d4ba81d2b0d0eed7e3dd6d1fe254 ;;
  */swe-rebench-v2.safe.jsonl)
    digest=1e70b4d99cee7eea5dd40c4c36a553a53de3304caa7120ec45c00b5a2b6fdffd ;;
  */swe-rebench-v2.safe.manifest.json)
    digest=93f17e1f466fa0e014b29112c34d5f05830c17f39e296bcc89915f7b5567cfb5 ;;
  */swe-rebench-v2.safe.rejections.jsonl)
    digest=473679bf93386cd6bdbea8019e7991104c355fd21b30886632669c2e099d7bf2 ;;
  *)
    printf 'unexpected sha256sum input: %s\n' "${target}" >&2
    exit 91 ;;
esac
printf '%s  %s\n' "${digest}" "${target}"
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

python_index = arguments.index("python")
python_arguments = arguments[python_index + 1 :]
if python_arguments[0] == "-":
    sys.stdin.read()
    mode = python_arguments[1]
    if os.environ.get("FAKE_UV_REJECT_VALIDATOR") == mode:
        raise SystemExit(93)
    if mode == "normalization" and os.environ.get("FAKE_UV_NORMALIZATION_FAIL") == "1":
        print("FAIL")
    else:
        print("PASS")
    raise SystemExit(0)

if python_arguments[:3] != ["-m", "nodelm", "datasets"]:
    raise SystemExit(92)
command = python_arguments[3]
command_arguments = python_arguments[4:]


def value(flag: str) -> Path:
    return Path(command_arguments[command_arguments.index(flag) + 1])


outputs = [value("--output"), value("--manifest-output")]
if command == "normalize":
    outputs.append(value("--rejections-output"))
for output in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{{}}\\n" if output.suffix == ".json" else "")
if command == "normalize" and os.environ.get("FAKE_UV_NORMALIZATION_FAIL") == "1":
    raise SystemExit(1)
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

    shutil.copytree(PROJECT_ROOT / "src", repo / "src")
    (repo / "scripts").mkdir()
    shutil.copy2(RUNNER, repo / "scripts" / RUNNER.name)
    _write(repo / "pyproject.toml", "[project]\nname = 'nodelm-runner-test'\nversion = '0'\n")
    _write(repo / "configs" / "datasets" / "registry.yaml", "test: registry\n")
    _write(
        repo / "configs" / "datasets" / "open-swe-trace-partitions.yaml",
        "test: partitions\n",
    )
    (repo / "scripts" / RUNNER.name).chmod(0o755)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "runner fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    task_dir = persist / "derived" / "normalization-canary-20260825-7366ec0"
    for relative_path in (
        "receipts/open-swe-traces.transfer.json",
        "receipts/swe-rebench-v2.transfer.json",
    ):
        _write(persist / relative_path, "{}\n")
    for filename in (
        "swe-rebench-v2.safe.jsonl",
        "swe-rebench-v2.safe.manifest.json",
        "swe-rebench-v2.safe.rejections.jsonl",
    ):
        _write(task_dir / filename, "{}\n")
    (persist / "snapshots" / "open-swe-traces").mkdir(parents=True)
    (persist / "snapshots" / "swe-rebench-v2").mkdir(parents=True)

    _make_fake_sha256sum(fake_bin / "sha256sum")
    _write(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "flock").chmod(0o755)
    fake_uv = fake_bin / "uv"
    _make_fake_uv(fake_uv)
    uv_log = tmp_path / "uv-calls.jsonl"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "NODELM_REPO_ROOT": str(repo),
        "NODELM_PERSIST_ROOT": str(persist),
        "NODELM_EXEC_TMP_ROOT": str(tmp_path / "execution"),
        "NODELM_UV_BIN": str(fake_uv),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
        "FAKE_UV_LOG": str(uv_log),
    }
    python_path = Path(environment["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.symlink_to(sys.executable)
    return RunnerHarness(repo, persist, commit, environment, uv_log)


def _establish_run_binding(runner_harness: RunnerHarness) -> None:
    _write(runner_harness.run_dir / "STOP", "establish binding\n")
    preflight = runner_harness.run()
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    (runner_harness.run_dir / "STOP").unlink()


def test_stop_preflight_creates_and_enforces_commit_binding(
    runner_harness: RunnerHarness,
) -> None:
    run_dir = runner_harness.run_dir
    _write(run_dir / "STOP", "operator stop\n")

    first = runner_harness.run()

    assert first.returncode == 0, first.stdout + first.stderr
    state = (run_dir / "run.state").read_text()
    binding = (run_dir / "run.binding").read_text()
    assert "state=STOPPED" in state
    assert f"commit={runner_harness.commit}" in state
    assert f"commit={runner_harness.commit}" in binding
    assert "format=nodelm-full-normalization-run-binding-v1" in binding
    assert runner_harness.uv_calls() == []
    assert list(Path(runner_harness.environment["NODELM_EXEC_TMP_ROOT"]).iterdir()) == []

    marker = run_dir / "run.binding"
    marker.chmod(0o644)
    marker.write_text(binding.replace(runner_harness.commit, "0" * 40))
    second = runner_harness.run()

    assert second.returncode == 1
    assert "run marker does not match the exact execution tree" in second.stdout
    assert runner_harness.uv_calls() == []


def test_full_fake_run_uses_private_offline_tree_and_resumes(
    runner_harness: RunnerHarness,
) -> None:
    first = runner_harness.run()

    assert first.returncode == 0, first.stdout + first.stderr
    assert "state=COMPLETE" in (runner_harness.run_dir / "run.state").read_text()
    first_calls = runner_harness.uv_calls()
    producer_calls = [call for call in first_calls if "-m" in call["args"]]
    validator_calls = [call for call in first_calls if "-" in call["args"]]
    assert len(first_calls) == 28
    assert len(producer_calls) == 14
    assert len(validator_calls) == 14
    for call in first_calls:
        arguments = call["args"]
        execution_root = Path(arguments[arguments.index("--directory") + 1])
        assert execution_root != runner_harness.repo
        assert execution_root.parent.parent == Path(
            runner_harness.environment["NODELM_EXEC_TMP_ROOT"]
        )
        assert call["offline"] == "1"
        assert call["no_sync"] == "1"
        assert call["hf_hub_offline"] == "1"
        assert call["hf_datasets_offline"] == "1"
        assert call["transformers_offline"] == "1"
        assert call["pythonpath"] == str(execution_root / "src")
        assert "--no-sync" in arguments

    runner_harness.uv_log.unlink()
    second = runner_harness.run()

    assert second.returncode == 0, second.stdout + second.stderr
    second_calls = runner_harness.uv_calls()
    assert len(second_calls) == 14
    assert all("-m" not in call["args"] for call in second_calls)
    assert (runner_harness.run_dir / "events.log").read_text().count("RESUME leaf=") == 14


def test_partial_materialization_is_refused_without_running_a_producer(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    _write(runner_harness.run_dir / "openhands-minimax-v2.raw.jsonl", "partial\n")

    result = runner_harness.run()

    assert result.returncode == 1
    assert "conflicting partial materialization evidence" in result.stdout
    assert runner_harness.uv_calls() == []
    assert "state=FAILED" in (runner_harness.run_dir / "run.state").read_text()


def test_partial_normalization_is_refused_without_running_a_producer(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    _write(runner_harness.run_dir / "openhands-minimax-v2.raw.jsonl", "raw\n")
    _write(runner_harness.run_dir / "openhands-minimax-v2.raw.manifest.json", "{}\n")
    _write(runner_harness.run_dir / "openhands-minimax-v2.normalized.jsonl", "partial\n")

    result = runner_harness.run()

    assert result.returncode == 1
    assert "conflicting partial normalization evidence" in result.stdout
    assert len(runner_harness.uv_calls()) == 1
    assert all("-m" not in call["args"] for call in runner_harness.uv_calls())


def test_invalid_existing_terminal_checkpoint_is_rejected(
    runner_harness: RunnerHarness,
) -> None:
    _establish_run_binding(runner_harness)
    _write(runner_harness.run_dir / "openhands-minimax-v2.raw.jsonl", "raw\n")
    _write(runner_harness.run_dir / "openhands-minimax-v2.raw.manifest.json", "{}\n")

    result = runner_harness.run({"FAKE_UV_REJECT_VALIDATOR": "materialization"})

    assert result.returncode == 1
    assert "existing materialization checkpoint is invalid" in result.stdout
    assert len(runner_harness.uv_calls()) == 1
    assert all("-m" not in call["args"] for call in runner_harness.uv_calls())


def test_truthful_normalization_fail_is_sealed_and_run_completes(
    runner_harness: RunnerHarness,
) -> None:
    result = runner_harness.run({"FAKE_UV_NORMALIZATION_FAIL": "1"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "state=COMPLETE" in (runner_harness.run_dir / "run.state").read_text()
    events = (runner_harness.run_dir / "events.log").read_text()
    assert events.count("phase=normalize status=FAIL exit=1") == 7
    assert "COMPLETE" in events
