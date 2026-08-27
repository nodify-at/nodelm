from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_oracle_isolation_reviews.sh"


def test_oracle_isolation_runner_is_valid_offline_four_leaf_orchestration() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        check=False,
        capture_output=True,
        text=True,
    )
    script = RUNNER.read_text(encoding="utf-8")

    assert syntax.returncode == 0, syntax.stderr
    assert script.count("|openhands/") == 2
    assert script.count("|sweagent/") == 2
    assert "datasets review-oracle-isolation" in script
    assert "--raw-input" in script
    assert "--materialization-manifest" in script
    assert "--normalization-manifest" in script
    assert "UV_OFFLINE=1" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "STOPPED" in script
    assert "authorization=PENDING" in script
