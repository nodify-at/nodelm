from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

import psutil

from nodelm.harness.models import CommandResult, CommandSpec, OutcomeCategory

_INHERITED_ENVIRONMENT = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)
_SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:API[_-]?KEY|AUTH|BEARER|CREDENTIAL|PASSWORD|PRIVATE[_-]?KEY|SECRET|TOKEN)",
    re.IGNORECASE,
)
_KEY_VALUE_SECRET = re.compile(
    r"""(?ix)
    (?P<prefix>
        (?<![A-Za-z0-9_])
        (?P<key_quote>["']?)
        [A-Za-z0-9_-]*
        (?:api[_-]?key|auth(?:orization)?|bearer|credential|password|private[_-]?key|secret|token)
        [A-Za-z0-9_-]*
        (?P=key_quote)
        \s*[:=]\s*
    )
    (?P<value>
        "(?:\\.|[^"\\])*"
        | '(?:\\.|[^'\\])*'
        | [^\s,;}]+
    )
    """
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_TRUNCATION_MARKER = "\n...[output truncated]"
_CAPTURE_CHUNK_BYTES = 64 * 1024
_REDACTION_MARGIN_BYTES = 4 * 1024


class _WorkspaceViolation(ValueError):
    pass


class _CappedCapture:
    def __init__(self, raw_capacity: int) -> None:
        self._raw_capacity = raw_capacity
        self._buffer = bytearray()
        self.total_bytes = 0
        self.error: OSError | None = None

    def read(self, stream: BinaryIO) -> None:
        try:
            while chunk := stream.read(_CAPTURE_CHUNK_BYTES):
                self.total_bytes += len(chunk)
                available = self._raw_capacity - len(self._buffer)
                if available > 0:
                    self._buffer.extend(chunk[:available])
        except OSError as error:
            self.error = error
        finally:
            stream.close()

    @property
    def data(self) -> bytes:
        return bytes(self._buffer)


class CommandExecutor:
    """Execute bounded argv-only commands inside one trusted local workspace.

    The executor uses a sanitized environment, never invokes a shell, and terminates the
    process group/tree on timeout. These are defense-in-depth controls, not isolation. A
    repository's package scripts and tools can execute arbitrary code, so untrusted
    repositories must run in a real OS/container sandbox outside this harness.
    """

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        default_max_output_bytes: int = 256 * 1024,
        termination_grace_seconds: float = 1.0,
    ) -> None:
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise ValueError(f"workspace root is not an existing directory: {root}")
        if default_max_output_bytes <= 0:
            raise ValueError("default_max_output_bytes must be greater than zero")
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be greater than zero")
        self.workspace_root = root
        self.default_max_output_bytes = default_max_output_bytes
        self.termination_grace_seconds = termination_grace_seconds

    def run(self, spec: CommandSpec) -> CommandResult:
        started = time.monotonic()
        redactions = _redaction_values(spec)
        safe_argv = tuple(_redact_text(argument, redactions) for argument in spec.argv)
        try:
            cwd = self._resolve_cwd(spec.cwd)
        except _WorkspaceViolation as error:
            return self._failure_result(
                safe_argv,
                self.workspace_root,
                OutcomeCategory.TOOL_PROTOCOL_FAILURE,
                error,
                started,
                redactions,
            )

        environment = _sanitized_environment(spec.env)
        max_output_bytes = spec.max_output_bytes or self.default_max_output_bytes
        longest_redaction = max((len(value.encode("utf-8")) for value in redactions), default=0)
        raw_capacity = max_output_bytes + longest_redaction + _REDACTION_MARGIN_BYTES
        stdout_capture = _CappedCapture(raw_capacity)
        stderr_capture = _CappedCapture(raw_capacity)

        try:
            process = _start_process(spec.argv, cwd, environment)
        except FileNotFoundError:
            missing_executable = FileNotFoundError(f"executable not found: {spec.argv[0]}")
            return self._failure_result(
                safe_argv,
                cwd,
                OutcomeCategory.ENVIRONMENT_FAILURE,
                missing_executable,
                started,
                redactions,
            )
        except OSError as error:
            return self._failure_result(
                safe_argv,
                cwd,
                OutcomeCategory.ENVIRONMENT_FAILURE,
                error,
                started,
                redactions,
            )
        except Exception as error:  # pragma: no cover - defensive harness boundary
            return self._failure_result(
                safe_argv,
                cwd,
                OutcomeCategory.INTERNAL_FAILURE,
                error,
                started,
                redactions,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=stdout_capture.read, args=(process.stdout,), name="nodelm-command-stdout"
        )
        stderr_thread = threading.Thread(
            target=stderr_capture.read, args=(process.stderr,), name="nodelm-command-stderr"
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            exit_code = process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process_tree(process)
            exit_code = self._wait_after_termination(process)
        finally:
            if process.poll() is not None:
                self._terminate_remaining_group(process.pid)

        stdout_thread.join(timeout=self.termination_grace_seconds)
        stderr_thread.join(timeout=self.termination_grace_seconds)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            self._terminate_remaining_group(process.pid, force=True)
            stdout_thread.join(timeout=self.termination_grace_seconds)
            stderr_thread.join(timeout=self.termination_grace_seconds)

        stdout, stdout_truncated = _render_output(stdout_capture, max_output_bytes, redactions)
        stderr, stderr_truncated = _render_output(stderr_capture, max_output_bytes, redactions)
        stream_error = stdout_capture.error or stderr_capture.error
        if stream_error is not None and not timed_out:
            outcome = OutcomeCategory.INTERNAL_FAILURE
            stderr = _append_bounded(
                stderr,
                f"output capture failed: {type(stream_error).__name__}: {stream_error}",
                max_output_bytes,
                redactions,
            )
            stderr_truncated = len(stderr.encode("utf-8")) >= max_output_bytes
        elif timed_out:
            outcome = OutcomeCategory.TIMEOUT
        elif exit_code in spec.success_exit_codes:
            outcome = OutcomeCategory.SUCCESS
        else:
            outcome = spec.failure_outcome

        return CommandResult(
            argv=safe_argv,
            cwd=cwd,
            outcome=outcome,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _resolve_cwd(self, cwd: Path | str | None) -> Path:
        candidate = self.workspace_root if cwd is None else Path(cwd)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            raise _WorkspaceViolation(f"command cwd resolves outside workspace: {cwd}")
        if not resolved.is_dir():
            raise _WorkspaceViolation(f"command cwd is not an existing directory: {cwd}")
        return resolved

    def _terminate_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        # The POSIX process group is the authoritative tree boundary. Signal it before
        # best-effort psutil inspection, which macOS sandboxes may deny.
        self._terminate_remaining_group(process.pid)
        descendants: list[psutil.Process] = []
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            descendants = psutil.Process(process.pid).children(recursive=True)

        if os.name != "posix" and process.poll() is None:  # pragma: no cover - Windows
            process.terminate()

        _, alive = psutil.wait_procs(descendants, timeout=self.termination_grace_seconds)
        for child in alive:
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()
        if process.poll() is None:
            process.kill()

    def _terminate_remaining_group(self, process_group_id: int, *, force: bool = False) -> None:
        if os.name != "posix":
            return
        requested_signal = signal.SIGKILL if force else signal.SIGTERM
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process_group_id, requested_signal)

    def _wait_after_termination(self, process: subprocess.Popen[bytes]) -> int:
        try:
            return process.wait(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            self._terminate_remaining_group(process.pid, force=True)
            process.kill()
            return process.wait(timeout=self.termination_grace_seconds)

    def _failure_result(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        outcome: OutcomeCategory,
        error: Exception,
        started: float,
        redactions: tuple[str, ...],
    ) -> CommandResult:
        stderr = _redact_text(f"{type(error).__name__}: {error}", redactions)
        stderr, stderr_truncated = _cap_text(stderr, self.default_max_output_bytes, False)
        return CommandResult(
            argv=argv,
            cwd=cwd,
            outcome=outcome,
            exit_code=None,
            stdout="",
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            stderr_truncated=stderr_truncated,
        )


def _start_process(
    argv: tuple[str, ...], cwd: Path, environment: Mapping[str, str]
) -> subprocess.Popen[bytes]:
    if os.name == "posix":
        return subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )

    creation_flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        creationflags=creation_flags,
    )


def _sanitized_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items() if name in _INHERITED_ENVIRONMENT
    }
    environment.update({"CI": "1", "NO_COLOR": "1"})
    environment.update(overrides)
    return environment


def _redaction_values(spec: CommandSpec) -> tuple[str, ...]:
    sensitive_values = [
        value for name, value in spec.env.items() if value and _SENSITIVE_ENVIRONMENT.search(name)
    ]
    return tuple(sorted({*spec.redact_values, *sensitive_values}, key=len, reverse=True))


def _redact_text(text: str, redactions: tuple[str, ...]) -> str:
    for value in redactions:
        text = text.replace(value, "[REDACTED]")
    text = _KEY_VALUE_SECRET.sub(_redact_key_value, text)
    return _BEARER_SECRET.sub("Bearer [REDACTED]", text)


def _redact_key_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        redacted_value = f"{value[0]}[REDACTED]{value[-1]}"
    else:
        redacted_value = "[REDACTED]"
    return f"{match.group('prefix')}{redacted_value}"


def _render_output(
    capture: _CappedCapture, max_output_bytes: int, redactions: tuple[str, ...]
) -> tuple[str, bool]:
    text = capture.data.decode("utf-8", errors="replace")
    redacted = _redact_text(text, redactions)
    raw_was_truncated = capture.total_bytes > len(capture.data)
    return _cap_text(redacted, max_output_bytes, raw_was_truncated)


def _cap_text(text: str, max_bytes: int, already_truncated: bool) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    truncated = already_truncated or len(encoded) > max_bytes
    if not truncated:
        return text, False

    marker = _TRUNCATION_MARKER.encode("utf-8")
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("utf-8", errors="ignore"), True
    prefix = encoded[: max_bytes - len(marker)].decode("utf-8", errors="ignore")
    return prefix + _TRUNCATION_MARKER, True


def _append_bounded(
    existing: str, addition: str, max_bytes: int, redactions: tuple[str, ...]
) -> str:
    combined = f"{existing}\n{_redact_text(addition, redactions)}" if existing else addition
    return _cap_text(combined, max_bytes, False)[0]
