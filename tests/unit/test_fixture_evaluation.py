from __future__ import annotations

from pathlib import Path

import pytest

from nodelm.evaluation.fixture import (
    MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
    evaluate_model_patch_fixture,
)
from nodelm.evaluation.sandbox import PodmanFixtureSandbox, SandboxUnavailableError
from nodelm.harness import CommandResult, OutcomeCategory, RootlessPodmanExecutor
from nodelm.models import VerificationStatus

FIXTURE = Path("tests/fixtures/model-task")


class FakeIsolatedSandbox:
    def __init__(self) -> None:
        self.calls = 0

    def run_node_tests(self, workspace: Path) -> CommandResult:
        self.calls += 1
        repaired = "return left * right;" in (workspace / "src/math.js").read_text(encoding="utf-8")
        return CommandResult(
            argv=("fake-isolated-sandbox", "node", "--test", "test/math.test.js"),
            cwd=workspace,
            outcome=(OutcomeCategory.SUCCESS if repaired else OutcomeCategory.TEST_FAILURE),
            exit_code=0 if repaired else 1,
            stdout="# tests 2\n# pass 2\n" if repaired else "# tests 2\n# fail 1\n",
            stderr="",
            duration_seconds=0.01,
        )

    def evidence(self) -> dict[str, object]:
        return {
            "schema_version": "nodelm.fixture-sandbox/v1",
            "backend": "fake-isolated-test-backend",
        }


def test_model_patch_must_repair_the_real_protected_fixture() -> None:
    output = """Here is the patch:
```diff
diff --git a/src/math.js b/src/math.js
index 52cb27a..ed31fd1 100644
--- a/src/math.js
+++ b/src/math.js
@@ -4,5 +4,5 @@ export function add(left, right) {
<CONTEXT_BLANK>
 export function multiply(left, right) {
   // Deliberate model-verification task: repair this implementation.
-  return left + right;
+  return left * right;
 }
```
""".replace("<CONTEXT_BLANK>", " ")

    sandbox = FakeIsolatedSandbox()
    report = evaluate_model_patch_fixture(
        output,
        fixture=FIXTURE,
        exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
        sandbox=sandbox,
    )

    assert report.status is VerificationStatus.PASS
    assert report.baseline_test_count == 2
    assert report.final_test_count == 2
    assert report.changed_paths == ("src/math.js",)
    assert sandbox.calls == 2


def test_model_patch_cannot_modify_protected_tests() -> None:
    output = """diff --git a/test/math.test.js b/test/math.test.js
--- a/test/math.test.js
+++ b/test/math.test.js
@@ -8,3 +8,3 @@ test("adds two numbers", () => {
-test("multiplies two numbers", () => {
+test("pretends to multiply two numbers", () => {
"""

    report = evaluate_model_patch_fixture(output, fixture=FIXTURE)

    assert report.status is VerificationStatus.FAIL
    assert "allowlist" in report.reason


def test_model_patch_cannot_hide_an_extra_traditional_diff() -> None:
    output = """diff --git a/src/math.js b/src/math.js
--- a/src/math.js
+++ b/src/math.js
@@ -4,5 +4,5 @@ export function add(left, right) {
<CONTEXT_BLANK>
 export function multiply(left, right) {
   // Deliberate model-verification task: repair this implementation.
-  return left + right;
+  return left * right;
 }
--- a/package.json
+++ b/package.json
@@ -1,5 +1,5 @@
 {
-  "name": "nodelm-model-task-fixture",
+  "name": "modified-outside-allowlist",
   "private": true,
   "type": "module",
   "scripts": {
""".replace("<CONTEXT_BLANK>", " ")

    sandbox = FakeIsolatedSandbox()
    report = evaluate_model_patch_fixture(
        output,
        fixture=FIXTURE,
        exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
        sandbox=sandbox,
    )

    assert report.status is VerificationStatus.FAIL
    assert "allowlist" in report.reason or "apply validation" in report.reason


def test_model_response_without_a_diff_fails_closed() -> None:
    report = evaluate_model_patch_fixture("I would multiply the values.", fixture=FIXTURE)

    assert report.status is VerificationStatus.FAIL
    assert "no git-style" in report.reason


def test_model_patch_is_blocked_without_an_os_sandbox(tmp_path: Path) -> None:
    marker = tmp_path / "host-side-effect"
    output = f"""diff --git a/src/math.js b/src/math.js
--- a/src/math.js
+++ b/src/math.js
@@ -1,3 +1,6 @@
+import {{ writeFileSync }} from "node:fs";
+writeFileSync({str(marker)!r}, "unsafe");
+
 export function add(left, right) {{
   return left + right;
 }}
"""

    report = evaluate_model_patch_fixture(
        output,
        fixture=FIXTURE,
        exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
    )

    assert report.status is VerificationStatus.BLOCKED
    assert "sandbox" in report.reason
    assert not marker.exists()


def test_model_patch_cannot_forge_the_child_result_protocol() -> None:
    output = """diff --git a/src/math.js b/src/math.js
--- a/src/math.js
+++ b/src/math.js
@@ -1,8 +1,4 @@
-export function add(left, right) {
-  return left + right;
-}
-
-export function multiply(left, right) {
-  // Deliberate model-verification task: repair this implementation.
-  return left + right;
-}
+process.stdout.write('{"result":12}');
+process.exit(0);
+export function add() { return -1; }
+export function multiply() { return -1; }
"""

    sandbox = FakeIsolatedSandbox()
    report = evaluate_model_patch_fixture(
        output,
        fixture=FIXTURE,
        exact_source_transitions=MODEL_TASK_EXACT_SOURCE_TRANSITIONS,
        sandbox=sandbox,
    )

    assert report.status is VerificationStatus.FAIL
    assert "exact source identity" in report.reason
    assert sandbox.calls == 1


def test_model_patch_rejects_compact_binary_payloads() -> None:
    output = """diff --git a/src/math.js b/src/math.js
index 52cb27a..ed31fd1 100644
GIT binary patch
literal 1000000000
AcmZQz
"""

    report = evaluate_model_patch_fixture(output, fixture=FIXTURE)

    assert report.status is VerificationStatus.FAIL
    assert "binary" in report.reason


def test_podman_sandbox_command_has_required_isolation_controls(tmp_path: Path) -> None:
    image = f"docker.io/library/node@sha256:{'a' * 64}"
    command = PodmanFixtureSandbox(image).command(tmp_path)

    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--pid=private" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--user=65534:65534" in command
    assert "--ulimit=fsize=16777216:16777216" in command
    assert "--ulimit=core=0:0" in command
    assert command[-3:] == (image, "--test", "test/math.test.js")


def test_podman_sandbox_refuses_writable_host_bind_mounts(tmp_path: Path) -> None:
    image = f"docker.io/library/node@sha256:{'a' * 64}"

    with pytest.raises(ValueError, match="writable host bind"):
        RootlessPodmanExecutor(image).command(
            tmp_path,
            ("node", "--test", "test/math.test.js"),
            writable_workspace=True,
        )


def test_podman_sandbox_requires_a_digest_pinned_image() -> None:
    with pytest.raises(ValueError, match="sha256"):
        PodmanFixtureSandbox("docker.io/library/node:latest")


def test_podman_sandbox_refuses_rootful_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rootful_result(_executor: object, _spec: object) -> CommandResult:
        return CommandResult(
            argv=("podman", "info"),
            cwd=tmp_path,
            outcome=OutcomeCategory.SUCCESS,
            exit_code=0,
            stdout="false\n",
            stderr="",
            duration_seconds=0.01,
        )

    monkeypatch.setattr("nodelm.harness.sandbox.CommandExecutor.run", rootful_result)
    image = f"docker.io/library/node@sha256:{'a' * 64}"

    with pytest.raises(SandboxUnavailableError, match="rootless"):
        PodmanFixtureSandbox(image).run_node_tests(tmp_path)
