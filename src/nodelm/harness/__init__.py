"""Trusted-local Node/TypeScript command harness.

This package is not a security sandbox for untrusted repositories. Commands execute local
repository code with the current user's operating-system permissions.
"""

from nodelm.harness.discovery import (
    PackageManifest,
    TypeScriptWorkspace,
    discover_package_manifests,
    discover_tsconfigs,
    discover_typescript_workspace,
)
from nodelm.harness.evidence import parse_node_test_count
from nodelm.harness.executor import CommandExecutor
from nodelm.harness.models import CommandResult, CommandSpec, OutcomeCategory
from nodelm.harness.policy import CommandPolicy, CommandPolicyError
from nodelm.harness.sandbox import RootlessPodmanExecutor, SandboxUnavailableError

__all__ = [
    "CommandExecutor",
    "CommandPolicy",
    "CommandPolicyError",
    "CommandResult",
    "CommandSpec",
    "OutcomeCategory",
    "PackageManifest",
    "RootlessPodmanExecutor",
    "SandboxUnavailableError",
    "TypeScriptWorkspace",
    "discover_package_manifests",
    "discover_tsconfigs",
    "discover_typescript_workspace",
    "parse_node_test_count",
]
