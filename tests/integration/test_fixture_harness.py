from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nodelm.harness import CommandExecutor, CommandPolicy, OutcomeCategory
from nodelm.harness.discovery import discover_typescript_workspace


@pytest.mark.integration
def test_checked_in_node_fixture_runs_through_discovery_policy_and_executor() -> None:
    fixture = Path("tests/fixtures/ts-project")
    workspace = discover_typescript_workspace(fixture)

    result = CommandExecutor(fixture).run(CommandPolicy(fixture).node_test())

    assert workspace.package_manifests
    assert workspace.tsconfig_paths
    assert result.outcome is OutcomeCategory.SUCCESS
    assert result.exit_code == 0
    assert "pass 1" in result.stdout


@pytest.mark.integration
def test_protected_model_fixture_resists_assertion_monkeypatching(tmp_path: Path) -> None:
    fixture = tmp_path / "model-task"
    shutil.copytree(Path("tests/fixtures/model-task"), fixture)
    (fixture / "src/math.js").write_text(
        'import assert from "node:assert/strict";\n'
        "assert.equal = () => {};\n"
        "export function add() { return -1; }\n"
        "export function multiply(left, right) { return left + right; }\n",
        encoding="utf-8",
    )

    result = CommandExecutor(fixture).run(CommandPolicy(fixture).node_test(("test/math.test.js",)))

    assert result.outcome is OutcomeCategory.TEST_FAILURE
    assert result.exit_code == 1
    assert "fail 2" in result.stdout
