from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nodelm.datasets.staging import RegularFileIdentity, regular_file_tree_identity
from nodelm.evaluation.sandbox import FixtureSandbox, SandboxUnavailableError
from nodelm.harness import CommandExecutor, CommandPolicy, OutcomeCategory, parse_node_test_count
from nodelm.harness.patches import validate_text_git_patch
from nodelm.models import VerificationStatus


@dataclass(frozen=True)
class ExactSourceTransition:
    """An explicitly approved before/after source identity for a smoke fixture."""

    path: str
    before_sha256: str
    after_sha256: str

    def __post_init__(self) -> None:
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != self.path:
            raise ValueError("source-transition paths must be normalized relative paths")
        for digest in (self.before_sha256, self.after_sha256):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("source-transition identities must be lowercase sha256 digests")


MODEL_TASK_EXACT_SOURCE_TRANSITIONS = (
    ExactSourceTransition(
        path="src/math.js",
        before_sha256="c3df0312d1ff6f17ef20ce799fe706e2ed138326c81f3ee4a85a4df80e13e079",
        after_sha256="a08f7a62ed3600268595fd20d7e9e506d299d7d48ee51c092b52d7560b547670",
    ),
)

MODEL_TASK_FIXTURE_IDENTITY = regular_file_tree_identity(
    (
        RegularFileIdentity(
            path="package.json",
            sha256="40fc837ade558c6d989356feafcd6d4e5916582e1f5a77d7badddb8fda6c3f48",
            bytes=145,
        ),
        RegularFileIdentity(
            path="src/math.js",
            sha256="c3df0312d1ff6f17ef20ce799fe706e2ed138326c81f3ee4a85a4df80e13e079",
            bytes=195,
        ),
        RegularFileIdentity(
            path="test/math.test.js",
            sha256="94f45561116f330d67a965551e78710af3f41b9e2ffc33b5b445695bafb2c922",
            bytes=1755,
        ),
        RegularFileIdentity(
            path="tsconfig.json",
            sha256="49995fe59ef45d2fe885cfe2e04643f0bbb324237c40e3e4a011ab1bc45b078d",
            bytes=220,
        ),
    )
)


class FixturePatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "nodelm.fixture-patch-evaluation/v2"
    status: VerificationStatus
    reason: str
    patch_sha256: str | None = None
    changed_paths: tuple[str, ...] = ()
    baseline_test_count: int | None = None
    final_test_count: int | None = None
    baseline_command: dict[str, object] | None = None
    patch_check_command: dict[str, object] | None = None
    patch_apply_command: dict[str, object] | None = None
    final_command: dict[str, object] | None = None
    sandbox: dict[str, object] | None = None


def extract_git_diff(model_output: str, *, max_bytes: int = 1_000_000) -> str:
    encoded = model_output.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("model patch output exceeds the evaluation limit")
    if "\0" in model_output:
        raise ValueError("model patch output contains a NUL byte")
    start = model_output.find("diff --git ")
    if start < 0:
        raise ValueError("model output contains no git-style unified diff")
    patch = model_output[start:]
    fence = patch.find("\n```")
    if fence >= 0:
        patch = patch[:fence]
    patch = patch.rstrip() + "\n"
    try:
        return validate_text_git_patch(patch, max_bytes=max_bytes)
    except ValueError as error:
        raise ValueError(f"invalid model patch: {error}") from error


def _changed_paths(patch: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        fields = line.split(" ")
        if len(fields) != 4 or not fields[2].startswith("a/") or not fields[3].startswith("b/"):
            raise ValueError("model diff contains a malformed path header")
        left = fields[2][2:]
        right = fields[3][2:]
        if left != right:
            raise ValueError("model diff may not rename fixture paths")
        paths.add(left)
    if not paths:
        raise ValueError("model diff contains no changed paths")
    return tuple(sorted(paths))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_identity(root: Path) -> dict[str, tuple[int, str]]:
    identity: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IFMT(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            payload = _sha256(path)
        elif stat.S_ISLNK(metadata.st_mode):
            payload = hashlib.sha256(os.readlink(path).encode()).hexdigest()
        else:
            payload = ""
        identity[relative] = (mode, payload)
    return identity


def evaluate_model_patch_fixture(
    model_output: str,
    *,
    fixture: Path,
    allowed_paths: tuple[str, ...] = ("src/math.js",),
    exact_source_transitions: tuple[ExactSourceTransition, ...] = (),
    sandbox: FixtureSandbox | None = None,
) -> FixturePatchReport:
    """Apply a model patch and run it only through an explicit OS isolation backend."""

    try:
        patch = extract_git_diff(model_output)
        changed_paths = _changed_paths(patch)
    except ValueError as error:
        return FixturePatchReport(status=VerificationStatus.FAIL, reason=str(error))
    if not set(changed_paths).issubset(allowed_paths):
        return FixturePatchReport(
            status=VerificationStatus.FAIL,
            reason="model diff changes paths outside the fixture source allowlist",
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            changed_paths=changed_paths,
        )
    transition_paths = tuple(transition.path for transition in exact_source_transitions)
    if not exact_source_transitions or len(set(transition_paths)) != len(transition_paths):
        return FixturePatchReport(
            status=VerificationStatus.BLOCKED,
            reason="an explicit, unique exact-source transition policy is required",
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            changed_paths=changed_paths,
        )
    if set(transition_paths) != set(changed_paths):
        return FixturePatchReport(
            status=VerificationStatus.FAIL,
            reason="model diff does not exactly match the approved source-transition paths",
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            changed_paths=changed_paths,
        )
    patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
    if sandbox is None:
        return FixturePatchReport(
            status=VerificationStatus.BLOCKED,
            reason="rootless container sandbox is required to execute model-authored code",
            patch_sha256=patch_sha256,
            changed_paths=changed_paths,
            sandbox={
                "schema_version": "nodelm.fixture-sandbox/v1",
                "backend": "none",
                "availability": "NOT_CONFIGURED",
            },
        )

    with tempfile.TemporaryDirectory(prefix="nodelm-model-fixture-") as temporary_directory:
        trusted_temporary_root = Path(temporary_directory)
        workspace = trusted_temporary_root / "workspace"
        shutil.copytree(fixture.resolve(), workspace)
        test_path = workspace / "test" / "math.test.js"
        if not test_path.is_file():
            return FixturePatchReport(
                status=VerificationStatus.FAIL,
                reason="fixture test file is missing",
                changed_paths=changed_paths,
            )
        protected_test_digest = _sha256(test_path)
        protected_tree = _tree_identity(workspace)
        transition_policy = {transition.path: transition for transition in exact_source_transitions}
        if any(
            not (workspace / path).is_file()
            or (workspace / path).is_symlink()
            or _sha256(workspace / path) != transition.before_sha256
            for path, transition in transition_policy.items()
        ):
            return FixturePatchReport(
                status=VerificationStatus.FAIL,
                reason="fixture source does not match the approved pre-patch identity",
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
            )
        patch_path = trusted_temporary_root / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")

        policy = CommandPolicy(workspace)
        executor = CommandExecutor(workspace)
        try:
            baseline = sandbox.run_node_tests(workspace)
        except SandboxUnavailableError as error:
            return FixturePatchReport(
                status=VerificationStatus.BLOCKED,
                reason=str(error),
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
                sandbox=sandbox.evidence(),
            )
        baseline_count = parse_node_test_count(baseline.stdout)
        if baseline.outcome is not OutcomeCategory.TEST_FAILURE or not baseline_count:
            return FixturePatchReport(
                status=VerificationStatus.FAIL,
                reason="fixture baseline did not produce the expected failing test evidence",
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
                baseline_test_count=baseline_count,
                baseline_command=baseline.to_evidence(),
                sandbox=sandbox.evidence(),
            )

        check = executor.run(
            policy.generic(
                ("git", "apply", "--check", "--whitespace=error-all", str(patch_path)),
                trusted_local=True,
                failure_outcome=OutcomeCategory.MODEL_FAILURE,
            )
        )
        if check.outcome is not OutcomeCategory.SUCCESS:
            return FixturePatchReport(
                status=VerificationStatus.FAIL,
                reason="model diff failed git apply validation",
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
                baseline_test_count=baseline_count,
                baseline_command=baseline.to_evidence(),
                patch_check_command=check.to_evidence(),
                sandbox=sandbox.evidence(),
            )

        apply_result = executor.run(
            policy.generic(
                ("git", "apply", "--whitespace=error-all", str(patch_path)),
                trusted_local=True,
                failure_outcome=OutcomeCategory.MODEL_FAILURE,
            )
        )
        current_tree = _tree_identity(workspace)
        changed_tree_paths = {
            path
            for path in protected_tree.keys() | current_tree.keys()
            if protected_tree.get(path) != current_tree.get(path)
        }
        allowed_source_files_are_regular = all(
            (workspace / path).is_file() and not (workspace / path).is_symlink()
            for path in changed_tree_paths
            if path in allowed_paths
        )
        protected_test_unchanged = (
            test_path.is_file()
            and not test_path.is_symlink()
            and _sha256(test_path) == protected_test_digest
        )
        approved_source_transition = all(
            (workspace / path).is_file()
            and not (workspace / path).is_symlink()
            and _sha256(workspace / path) == transition.after_sha256
            for path, transition in transition_policy.items()
        )
        if (
            apply_result.outcome is not OutcomeCategory.SUCCESS
            or not protected_test_unchanged
            or not changed_tree_paths.issubset(allowed_paths)
            or not allowed_source_files_are_regular
            or not approved_source_transition
        ):
            return FixturePatchReport(
                status=VerificationStatus.FAIL,
                reason=(
                    "model diff failed to apply, escaped the allowlist, or did not produce "
                    "the approved exact source identity"
                ),
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
                baseline_test_count=baseline_count,
                baseline_command=baseline.to_evidence(),
                patch_check_command=check.to_evidence(),
                patch_apply_command=apply_result.to_evidence(),
                sandbox=sandbox.evidence(),
            )

        try:
            final = sandbox.run_node_tests(workspace)
        except SandboxUnavailableError as error:
            return FixturePatchReport(
                status=VerificationStatus.BLOCKED,
                reason=str(error),
                patch_sha256=patch_sha256,
                changed_paths=changed_paths,
                baseline_test_count=baseline_count,
                baseline_command=baseline.to_evidence(),
                patch_check_command=check.to_evidence(),
                patch_apply_command=apply_result.to_evidence(),
                sandbox=sandbox.evidence(),
            )
        final_count = parse_node_test_count(final.stdout)
        passed = (
            final.outcome is OutcomeCategory.SUCCESS and final_count is not None and final_count > 0
        )
        return FixturePatchReport(
            status=VerificationStatus.PASS if passed else VerificationStatus.FAIL,
            reason=(
                "model-authored patch passed the protected fixture tests"
                if passed
                else "model-authored patch did not pass the protected fixture tests"
            ),
            patch_sha256=patch_sha256,
            changed_paths=changed_paths,
            baseline_test_count=baseline_count,
            final_test_count=final_count,
            baseline_command=baseline.to_evidence(),
            patch_check_command=check.to_evidence(),
            patch_apply_command=apply_result.to_evidence(),
            final_command=final.to_evidence(),
            sandbox=sandbox.evidence(),
        )
