from __future__ import annotations

import csv
import ctypes.util
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nodelm.models import CheckResult, VerificationStatus


class GPUDeviceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    memory_total_mib: int = Field(gt=0)
    memory_total_bytes: int = Field(gt=0)
    driver_version: str = Field(min_length=1)


class InfrastructureReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["nodelm.infrastructure/v2"] = "nodelm.infrastructure/v2"
    os: str
    architecture: str
    cpu_count: int
    host_ram_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    status: VerificationStatus
    gpu_count: int = Field(ge=0)
    total_gpu_memory_bytes: int = Field(ge=0)
    driver_versions: tuple[str, ...]
    gpu: CheckResult
    cuda_runtime: CheckResult
    cuda_toolkit: CheckResult
    nvlink_nvswitch: CheckResult
    nccl: CheckResult
    rdma_infiniband: CheckResult


def _not_run(name: str, summary: str, *, availability: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=VerificationStatus.NOT_RUN,
        summary=summary,
        evidence={"availability": availability},
    )


def _dependent_checks(summary: str) -> tuple[CheckResult, ...]:
    return (
        _not_run("cuda_runtime", summary, availability="UNKNOWN"),
        _not_run("cuda_toolkit", summary, availability="UNKNOWN"),
        _not_run("nvlink_nvswitch", summary, availability="UNKNOWN"),
        _not_run("nccl", summary, availability="UNKNOWN"),
        _not_run("rdma_infiniband", summary, availability="UNKNOWN"),
    )


def collect_infrastructure_report(
    *,
    workspace: Path,
    finder: Callable[[str], str | None] = shutil.which,
    library_finder: Callable[[str], str | None] = ctypes.util.find_library,
) -> InfrastructureReport:
    disk = shutil.disk_usage(workspace.resolve())
    gpu_binary = finder("nvidia-smi")
    devices: tuple[GPUDeviceEvidence, ...] = ()

    if gpu_binary is None:
        gpu = _not_run(
            "gpu",
            "nvidia-smi is not installed; NVIDIA GPU availability was not measured",
            availability="UNAVAILABLE",
        )
        dependent = _dependent_checks("NVIDIA GPU probe did not run")
    else:
        gpu = _probe_gpu(gpu_binary)
        devices = _gpu_devices(gpu)
        if gpu.status is not VerificationStatus.PASS:
            dependent = _dependent_checks("NVIDIA GPU probe did not pass")
        else:
            dependent = (
                _probe_cuda_runtime(library_finder),
                _probe_cuda_toolkit(finder("nvcc")),
                _probe_interconnect(gpu_binary, len(devices)),
                _probe_nccl(
                    gpu_count=len(devices),
                    library_finder=library_finder,
                    benchmark_binary=finder("all_reduce_perf"),
                ),
                _probe_rdma_infiniband(finder),
            )

    cuda_runtime, cuda_toolkit, interconnect, nccl, rdma = dependent
    overall = _overall_status(
        gpu=gpu,
        cuda_runtime=cuda_runtime,
        cuda_toolkit=cuda_toolkit,
        interconnect=interconnect,
        nccl=nccl,
        gpu_count=len(devices),
    )
    return InfrastructureReport(
        os=platform.platform(),
        architecture=platform.machine(),
        cpu_count=os.cpu_count() or 0,
        host_ram_bytes=psutil.virtual_memory().total,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        status=overall,
        gpu_count=len(devices),
        total_gpu_memory_bytes=sum(device.memory_total_bytes for device in devices),
        driver_versions=tuple(sorted({device.driver_version for device in devices})),
        gpu=gpu,
        cuda_runtime=cuda_runtime,
        cuda_toolkit=cuda_toolkit,
        nvlink_nvswitch=interconnect,
        nccl=nccl,
        rdma_infiniband=rdma,
    )


def _overall_status(
    *,
    gpu: CheckResult,
    cuda_runtime: CheckResult,
    cuda_toolkit: CheckResult,
    interconnect: CheckResult,
    nccl: CheckResult,
    gpu_count: int,
) -> VerificationStatus:
    if gpu.status is VerificationStatus.FAIL:
        return VerificationStatus.FAIL
    if gpu.status is not VerificationStatus.PASS:
        return VerificationStatus.NOT_RUN
    required = [cuda_runtime, cuda_toolkit]
    if gpu_count > 1:
        required.extend((interconnect, nccl))
    if any(check.status is VerificationStatus.FAIL for check in required):
        return VerificationStatus.FAIL
    if any(check.status is not VerificationStatus.PASS for check in required):
        return VerificationStatus.UNVERIFIED
    return VerificationStatus.PASS


def _gpu_devices(result: CheckResult) -> tuple[GPUDeviceEvidence, ...]:
    raw_devices = result.evidence.get("devices")
    if not isinstance(raw_devices, (list, tuple)):
        return ()
    devices: list[GPUDeviceEvidence] = []
    for raw in raw_devices:
        try:
            devices.append(GPUDeviceEvidence.model_validate(raw))
        except ValidationError:
            return ()
    return tuple(devices)


def _run_probe(
    argv: list[str],
) -> subprocess.CompletedProcess[str] | OSError | subprocess.TimeoutExpired:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return error


def _probe_gpu(binary: str) -> CheckResult:
    result = _run_probe(
        [
            binary,
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if isinstance(result, (OSError, subprocess.TimeoutExpired)):
        return CheckResult(
            name="gpu",
            status=VerificationStatus.FAIL,
            summary=f"nvidia-smi probe failed: {type(result).__name__}",
        )
    if result.returncode != 0:
        return CheckResult(
            name="gpu",
            status=VerificationStatus.FAIL,
            summary="nvidia-smi returned a nonzero exit code",
            evidence={"exit_code": result.returncode},
        )

    devices: list[GPUDeviceEvidence] = []
    try:
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise ValueError("unexpected column count")
            index, uuid, name, memory_mib, driver = (value.strip() for value in row)
            memory_total_mib = int(memory_mib)
            devices.append(
                GPUDeviceEvidence(
                    index=int(index),
                    uuid=uuid,
                    name=name,
                    memory_total_mib=memory_total_mib,
                    memory_total_bytes=memory_total_mib * 1024 * 1024,
                    driver_version=driver,
                )
            )
    except (ValueError, ValidationError) as error:
        return CheckResult(
            name="gpu",
            status=VerificationStatus.FAIL,
            summary="nvidia-smi returned malformed GPU evidence",
            evidence={"error_type": type(error).__name__},
        )
    if not devices:
        return CheckResult(
            name="gpu",
            status=VerificationStatus.FAIL,
            summary="nvidia-smi returned no GPU rows",
        )
    return CheckResult(
        name="gpu",
        status=VerificationStatus.PASS,
        summary=f"detected {len(devices)} NVIDIA GPU(s)",
        evidence={
            "availability": "AVAILABLE",
            "devices": tuple(device.model_dump(mode="json") for device in devices),
        },
    )


def _probe_cuda_runtime(library_finder: Callable[[str], str | None]) -> CheckResult:
    try:
        library = library_finder("cudart")
    except (OSError, ValueError) as error:
        return CheckResult(
            name="cuda_runtime",
            status=VerificationStatus.FAIL,
            summary=f"CUDA runtime library probe failed: {type(error).__name__}",
        )
    if library is None:
        return _not_run(
            "cuda_runtime",
            "CUDA runtime library was not found",
            availability="UNAVAILABLE",
        )
    version_match = re.search(r"(?:so|dylib)[.-]([0-9][0-9.]*)", library)
    return CheckResult(
        name="cuda_runtime",
        status=VerificationStatus.PASS,
        summary="CUDA runtime library is available",
        evidence={
            "availability": "AVAILABLE",
            "library": library,
            "version_from_soname": version_match.group(1).rstrip(".") if version_match else None,
        },
    )


def _probe_cuda_toolkit(binary: str | None) -> CheckResult:
    if binary is None:
        return _not_run(
            "cuda_toolkit",
            "nvcc is not installed",
            availability="UNAVAILABLE",
        )
    result = _run_probe([binary, "--version"])
    if isinstance(result, (OSError, subprocess.TimeoutExpired)):
        return CheckResult(
            name="cuda_toolkit",
            status=VerificationStatus.FAIL,
            summary=f"nvcc probe failed: {type(result).__name__}",
        )
    combined = f"{result.stdout}\n{result.stderr}"
    version_match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)*)", combined)
    if result.returncode != 0 or version_match is None:
        return CheckResult(
            name="cuda_toolkit",
            status=VerificationStatus.FAIL,
            summary="nvcc did not return valid CUDA toolkit evidence",
            evidence={"exit_code": result.returncode},
        )
    return CheckResult(
        name="cuda_toolkit",
        status=VerificationStatus.PASS,
        summary=f"CUDA toolkit {version_match.group(1)} is available",
        evidence={
            "availability": "AVAILABLE",
            "nvcc_path": binary,
            "version": version_match.group(1),
        },
    )


def _probe_interconnect(binary: str, gpu_count: int) -> CheckResult:
    if gpu_count < 2:
        return _not_run(
            "nvlink_nvswitch",
            "NVLink/NVSwitch is not applicable to a single-GPU host",
            availability="NOT_APPLICABLE",
        )
    result = _run_probe([binary, "topo", "-m"])
    if isinstance(result, (OSError, subprocess.TimeoutExpired)):
        return CheckResult(
            name="nvlink_nvswitch",
            status=VerificationStatus.FAIL,
            summary=f"GPU topology probe failed: {type(result).__name__}",
        )
    if result.returncode != 0:
        return CheckResult(
            name="nvlink_nvswitch",
            status=VerificationStatus.FAIL,
            summary="nvidia-smi topology probe returned a nonzero exit code",
            evidence={"exit_code": result.returncode},
        )
    lines = tuple(line.rstrip() for line in result.stdout.splitlines() if line.strip())
    if not lines:
        return CheckResult(
            name="nvlink_nvswitch",
            status=VerificationStatus.FAIL,
            summary="nvidia-smi topology probe returned no evidence",
        )
    nvlink_present = re.search(r"\bNV[0-9]+\b", result.stdout, re.IGNORECASE) is not None
    explicit_nvswitch = re.search(r"\bNVS(?:WITCH)?\b", result.stdout, re.IGNORECASE) is not None
    nvswitch_present: bool | None = explicit_nvswitch or (False if not nvlink_present else None)
    availability = "AVAILABLE" if nvlink_present or nvswitch_present else "UNAVAILABLE"
    nvswitch_detection = (
        "EXPLICIT" if explicit_nvswitch else "NOT_DETERMINED" if nvlink_present else "ABSENT"
    )
    return CheckResult(
        name="nvlink_nvswitch",
        status=VerificationStatus.PASS,
        summary=f"NVLink/NVSwitch topology collected; fabric is {availability.lower()}",
        evidence={
            "availability": availability,
            "nvlink_present": nvlink_present,
            "nvswitch_present": nvswitch_present,
            "nvswitch_detection": nvswitch_detection,
            "topology_lines": lines[:64],
            "truncated": len(lines) > 64,
        },
    )


def _probe_nccl(
    *,
    gpu_count: int,
    library_finder: Callable[[str], str | None],
    benchmark_binary: str | None,
) -> CheckResult:
    if gpu_count < 2:
        return _not_run(
            "nccl",
            "NCCL is not required for a single-GPU host",
            availability="NOT_APPLICABLE",
        )
    try:
        library = library_finder("nccl")
    except (OSError, ValueError) as error:
        return CheckResult(
            name="nccl",
            status=VerificationStatus.FAIL,
            summary=f"NCCL library probe failed: {type(error).__name__}",
        )
    if library is None:
        return _not_run(
            "nccl",
            "NCCL library was not found for this multi-GPU host",
            availability="UNAVAILABLE",
        )
    return CheckResult(
        name="nccl",
        status=VerificationStatus.PASS,
        summary="NCCL library is available; no collective benchmark was run",
        evidence={
            "availability": "AVAILABLE",
            "library": library,
            "all_reduce_perf_path": benchmark_binary,
            "benchmark_status": VerificationStatus.NOT_RUN.value,
        },
    )


def _probe_rdma_infiniband(finder: Callable[[str], str | None]) -> CheckResult:
    candidates: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ibv_devices", ()),
        ("ibv_devinfo", ("-l",)),
        ("rdma", ("link", "show")),
    )
    selected: tuple[str, str, tuple[str, ...]] | None = None
    for tool, arguments in candidates:
        binary = finder(tool)
        if binary is not None:
            selected = (tool, binary, arguments)
            break
    if selected is None:
        return _not_run(
            "rdma_infiniband",
            "RDMA/InfiniBand query tools are not installed",
            availability="UNAVAILABLE",
        )

    tool, binary, arguments = selected
    result = _run_probe([binary, *arguments])
    if isinstance(result, (OSError, subprocess.TimeoutExpired)):
        return CheckResult(
            name="rdma_infiniband",
            status=VerificationStatus.FAIL,
            summary=f"{tool} probe failed: {type(result).__name__}",
        )
    if result.returncode != 0:
        return CheckResult(
            name="rdma_infiniband",
            status=VerificationStatus.FAIL,
            summary=f"{tool} returned a nonzero exit code",
            evidence={"exit_code": result.returncode},
        )

    lines = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    device_lines = _rdma_device_lines(tool, lines)
    unavailable_markers = ("0 hcas found", "no devices", "no rdma devices")
    unavailable = not device_lines or any(
        marker in result.stdout.lower() for marker in unavailable_markers
    )
    availability = "UNAVAILABLE" if unavailable else "AVAILABLE"
    return CheckResult(
        name="rdma_infiniband",
        status=VerificationStatus.PASS,
        summary=f"RDMA/InfiniBand query completed; devices are {availability.lower()}",
        evidence={
            "availability": availability,
            "tool": tool,
            "device_lines": device_lines[:64],
            "truncated": len(device_lines) > 64,
        },
    )


def _rdma_device_lines(tool: str, lines: tuple[str, ...]) -> tuple[str, ...]:
    if tool == "ibv_devices":
        return tuple(
            line
            for line in lines
            if not line.lower().startswith("device") and set(line) - {"-", " ", "\t"}
        )
    if tool == "ibv_devinfo":
        count_match = re.search(r"\b([0-9]+)\s+HCAs?\s+found\b", "\n".join(lines), re.IGNORECASE)
        if count_match is not None and int(count_match.group(1)) == 0:
            return ()
        return tuple(line for line in lines if line.lower().startswith("hca_id"))
    return lines
