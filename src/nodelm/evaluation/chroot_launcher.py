from __future__ import annotations

import argparse
import ctypes
import errno
import os
import resource
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil

_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM
_SCMP_CMP_NE = 1
_BLOCKED_SYSCALLS = (
    "add_key",
    "bpf",
    "chroot",
    "delete_module",
    "finit_module",
    "init_module",
    "io_uring_enter",
    "io_uring_register",
    "io_uring_setup",
    "kexec_file_load",
    "kexec_load",
    "keyctl",
    "mount",
    "move_mount",
    "open_by_handle_at",
    "open_tree",
    "perf_event_open",
    "pivot_root",
    "process_vm_readv",
    "process_vm_writev",
    "ptrace",
    "reboot",
    "request_key",
    "sched_setaffinity",
    "setns",
    "setpgid",
    "setsid",
    "socketcall",
    "swapon",
    "swapoff",
    "umount2",
    "unshare",
    "userfaultfd",
)
_AF_UNIX = 1


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = (
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    )


def _prctl(option: int, argument: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, argument, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _drop_capability_bounding_set() -> None:
    for capability in range(64):
        try:
            _prctl(_PR_CAPBSET_DROP, capability)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise


def _load_seccomp_filter(seccomp: ctypes.CDLL) -> None:
    seccomp.seccomp_init.argtypes = (ctypes.c_uint32,)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_release.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    )
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_load.restype = ctypes.c_int

    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp filter initialization failed")
    try:
        for name in _BLOCKED_SYSCALLS:
            syscall = seccomp.seccomp_syscall_resolve_name(name.encode())
            if syscall < 0:
                continue
            if seccomp.seccomp_rule_add(context, _SCMP_ACT_ERRNO, syscall, 0) != 0:
                raise RuntimeError("seccomp syscall rule installation failed")
        socket_syscall = seccomp.seccomp_syscall_resolve_name(b"socket")
        if socket_syscall < 0:
            raise RuntimeError("seccomp socket syscall resolution failed")
        comparison = _ScmpArgCmp(0, _SCMP_CMP_NE, _AF_UNIX, 0)
        if (
            seccomp.seccomp_rule_add_array(
                context,
                _SCMP_ACT_ERRNO,
                socket_syscall,
                1,
                ctypes.byref(comparison),
            )
            != 0
        ):
            raise RuntimeError("seccomp socket rule installation failed")
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("seccomp filter activation failed")
    finally:
        seccomp.seccomp_release(context)


def _parse_environment(values: list[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or not name or "\0" in value:
            raise ValueError("sandbox environment entries must be NUL-free NAME=VALUE pairs")
        environment[name] = item
    return environment


def _resolve_rootfs(rootfs: Path, workdir: str) -> tuple[Path, str]:
    if not rootfs.is_absolute() or rootfs.is_symlink():
        raise ValueError("sandbox rootfs must be an absolute non-symlink directory")
    resolved = rootfs.resolve(strict=True)
    if not resolved.is_dir() or not workdir.startswith("/") or "\0" in workdir:
        raise ValueError("sandbox rootfs or workdir is invalid")
    host_workdir = (resolved / workdir.removeprefix("/")).resolve(strict=True)
    if not host_workdir.is_dir() or not host_workdir.is_relative_to(resolved):
        raise ValueError("sandbox workdir escapes rootfs")
    return resolved, workdir


def _child(
    *,
    rootfs: Path,
    workdir: str,
    uid: int,
    gid: int,
    cpus: tuple[int, ...],
    memory_bytes: int,
    pids: int,
    environment: dict[str, str],
    command: tuple[str, ...],
) -> None:
    os.setsid()
    sched_setaffinity = getattr(os, "sched_setaffinity", None)
    if not callable(sched_setaffinity):
        raise RuntimeError("CPU affinity is unavailable")
    sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (1_800, 1_800))
    resource.setrlimit(resource.RLIMIT_NPROC, (pids, pids))
    resource.setrlimit(resource.RLIMIT_NOFILE, (4_096, 4_096))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_073_741_824, 1_073_741_824))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    os.chdir(rootfs)
    os.chroot(".")
    os.chdir(workdir)
    _drop_capability_bounding_set()
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    _prctl(_PR_SET_NO_NEW_PRIVS, 1)
    _load_seccomp_filter(seccomp)
    os.closerange(3, 1_048_576)
    os.execve(command[0], command, environment)


def _kill_tree(pid: int) -> None:
    try:
        process = psutil.Process(pid)
    except psutil.Error:
        return
    descendants = process.children(recursive=True)
    for target in (*descendants, process):
        with suppress(psutil.Error):
            target.kill()


def _supervise(pid: int, *, memory_bytes: int, pids: int) -> int:
    stopping = False

    def forward_signal(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        _kill_tree(pid)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    resource_failure = False
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            if os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            if os.WIFSIGNALED(status):
                return 128 + os.WTERMSIG(status)
            return 125
        try:
            process = psutil.Process(pid)
            tree = (process, *process.children(recursive=True))
            rss = sum(item.memory_info().rss for item in tree if item.is_running())
            if rss > memory_bytes or len(tree) > pids:
                resource_failure = True
                _kill_tree(pid)
        except psutil.Error:
            pass
        if stopping:
            _kill_tree(pid)
        if resource_failure:
            print("sandbox resource monitor terminated the attempt", file=sys.stderr)
            resource_failure = False
        time.sleep(0.1)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--pids", type=int, required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    return arguments


def main() -> int:
    arguments = _arguments()
    if os.geteuid() != 0:
        raise SystemExit("seccomp chroot launcher requires root")
    rootfs, workdir = _resolve_rootfs(arguments.rootfs, arguments.workdir)
    cpus = tuple(int(value) for value in arguments.cpus.split(","))
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if not callable(sched_getaffinity):
        raise SystemExit("CPU affinity is unavailable")
    available = sched_getaffinity(0)
    if (
        not arguments.command
        or not cpus
        or not set(cpus).issubset(available)
        or arguments.uid <= 0
        or arguments.gid <= 0
        or arguments.memory_bytes <= 0
        or arguments.pids <= 0
    ):
        raise SystemExit("invalid seccomp chroot launcher bounds")
    environment = _parse_environment(arguments.env)
    pid = os.fork()
    if pid == 0:
        try:
            _child(
                rootfs=rootfs,
                workdir=workdir,
                uid=arguments.uid,
                gid=arguments.gid,
                cpus=cpus,
                memory_bytes=arguments.memory_bytes,
                pids=arguments.pids,
                environment=environment,
                command=tuple(arguments.command),
            )
        except BaseException as error:
            print(f"sandbox launcher failed: {type(error).__name__}", file=sys.stderr)
            os._exit(125)
    return _supervise(pid, memory_bytes=arguments.memory_bytes, pids=arguments.pids)


if __name__ == "__main__":
    raise SystemExit(main())
