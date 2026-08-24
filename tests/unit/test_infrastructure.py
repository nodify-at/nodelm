from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nodelm.infra.doctor import _probe_gpu, collect_infrastructure_report
from nodelm.models import VerificationStatus

TWO_GPU_ROWS = "\n".join(
    (
        "0, GPU-zero, NVIDIA H100, 81920, 550.54",
        "1, GPU-one, NVIDIA H100, 81920, 550.54",
        "",
    )
)


def test_infrastructure_report_never_marks_missing_gpu_as_pass(tmp_path: Path) -> None:
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=lambda name: None,
    )

    assert report.disk_total_bytes > 0
    assert report.host_ram_bytes > 0
    assert report.status is VerificationStatus.NOT_RUN
    assert report.gpu_count == 0
    assert report.total_gpu_memory_bytes == 0
    assert report.driver_versions == ()
    assert report.gpu.status is VerificationStatus.NOT_RUN
    assert report.gpu.evidence["availability"] == "UNAVAILABLE"
    assert report.cuda_runtime.status is VerificationStatus.NOT_RUN
    assert report.cuda_toolkit.status is VerificationStatus.NOT_RUN
    assert report.nvlink_nvswitch.status is VerificationStatus.NOT_RUN
    assert report.nccl.status is VerificationStatus.NOT_RUN
    assert report.rdma_infiniband.status is VerificationStatus.NOT_RUN


def test_infrastructure_report_marks_gpu_probe_failure_as_overall_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "nodelm.infra.doctor.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "probe failed"),
    )

    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )

    assert report.status is VerificationStatus.FAIL
    assert report.gpu.status is VerificationStatus.FAIL


def test_gpu_probe_fails_when_nvidia_smi_returns_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nodelm.infra.doctor.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "\n", ""),
    )

    result = _probe_gpu("/usr/bin/nvidia-smi")

    assert result.status is VerificationStatus.FAIL
    assert result.summary == "nvidia-smi returned no GPU rows"


def test_gpu_probe_passes_only_with_detected_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nodelm.infra.doctor.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "0, GPU-fixture, NVIDIA A100, 81920, 550.54\n", ""
        ),
    )

    result = _probe_gpu("/usr/bin/nvidia-smi")

    assert result.status is VerificationStatus.PASS
    assert result.evidence["availability"] == "AVAILABLE"
    assert result.evidence["devices"] == (
        {
            "index": 0,
            "uuid": "GPU-fixture",
            "name": "NVIDIA A100",
            "memory_total_mib": 81920,
            "memory_total_bytes": 81920 * 1024 * 1024,
            "driver_version": "550.54",
        },
    )


def test_single_gpu_report_requires_cuda_runtime_and_toolkit_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name,memory.total,driver_version" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "0, GPU-fixture, NVIDIA H100, 81920, 550.54\n", ""
            )
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.4", "")
        raise AssertionError(f"unexpected probe: {argv}")

    monkeypatch.setattr("nodelm.infra.doctor.subprocess.run", run)
    binaries = {"nvidia-smi": "/usr/bin/nvidia-smi", "nvcc": "/usr/local/cuda/bin/nvcc"}
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=binaries.get,
        library_finder=lambda name: "libcudart.so.12" if name == "cudart" else None,
    )

    assert report.status is VerificationStatus.PASS
    assert report.gpu_count == 1
    assert report.total_gpu_memory_bytes == 81920 * 1024 * 1024
    assert report.driver_versions == ("550.54",)
    assert report.cuda_runtime.evidence["version_from_soname"] == "12"
    assert report.cuda_toolkit.evidence["version"] == "12.4"
    assert report.nvlink_nvswitch.evidence["availability"] == "NOT_APPLICABLE"
    assert report.nccl.evidence["availability"] == "NOT_APPLICABLE"


def test_multi_gpu_report_collects_fabric_nccl_and_rdma_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name,memory.total,driver_version" in argv:
            return subprocess.CompletedProcess(argv, 0, TWO_GPU_ROWS, "")
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.4", "")
        if "topo" in argv:
            return subprocess.CompletedProcess(argv, 0, "GPU0 GPU1\nGPU0 X NV4\nGPU1 NV4 X\n", "")
        if argv[0].endswith("ibv_devices"):
            return subprocess.CompletedProcess(argv, 0, "mlx5_0 0123456789abcdef\n", "")
        raise AssertionError(f"unexpected probe: {argv}")

    monkeypatch.setattr("nodelm.infra.doctor.subprocess.run", run)
    binaries = {
        "nvidia-smi": "/usr/bin/nvidia-smi",
        "nvcc": "/usr/local/cuda/bin/nvcc",
        "all_reduce_perf": "/opt/nccl-tests/all_reduce_perf",
        "ibv_devices": "/usr/bin/ibv_devices",
    }
    libraries = {"cudart": "libcudart.so.12", "nccl": "libnccl.so.2"}
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=binaries.get,
        library_finder=libraries.get,
    )

    assert report.status is VerificationStatus.PASS
    assert report.gpu_count == 2
    assert report.nvlink_nvswitch.evidence["nvlink_present"] is True
    assert report.nvlink_nvswitch.evidence["nvswitch_present"] is None
    assert report.nvlink_nvswitch.evidence["nvswitch_detection"] == "NOT_DETERMINED"
    assert report.nccl.status is VerificationStatus.PASS
    assert report.nccl.evidence["benchmark_status"] == VerificationStatus.NOT_RUN.value
    assert report.rdma_infiniband.evidence["availability"] == "AVAILABLE"


def test_multi_gpu_report_is_unverified_when_nccl_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name,memory.total,driver_version" in argv:
            return subprocess.CompletedProcess(
                argv,
                0,
                TWO_GPU_ROWS,
                "",
            )
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.4", "")
        if "topo" in argv:
            return subprocess.CompletedProcess(argv, 0, "GPU0 GPU1\nGPU0 X PHB\nGPU1 PHB X\n", "")
        raise AssertionError(f"unexpected probe: {argv}")

    monkeypatch.setattr("nodelm.infra.doctor.subprocess.run", run)
    binaries = {"nvidia-smi": "/usr/bin/nvidia-smi", "nvcc": "/usr/bin/nvcc"}
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=binaries.get,
        library_finder=lambda name: "libcudart.so.12" if name == "cudart" else None,
    )

    assert report.status is VerificationStatus.UNVERIFIED
    assert report.nvlink_nvswitch.status is VerificationStatus.PASS
    assert report.nvlink_nvswitch.evidence["availability"] == "UNAVAILABLE"
    assert report.nccl.status is VerificationStatus.NOT_RUN
    assert report.nccl.evidence["availability"] == "UNAVAILABLE"


def test_rdma_headers_without_devices_are_reported_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name,memory.total,driver_version" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "0, GPU-fixture, NVIDIA H100, 81920, 550.54\n", ""
            )
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.4", "")
        if argv[0].endswith("ibv_devices"):
            return subprocess.CompletedProcess(argv, 0, "device node GUID\n------ ---- ----\n", "")
        raise AssertionError(f"unexpected probe: {argv}")

    monkeypatch.setattr("nodelm.infra.doctor.subprocess.run", run)
    binaries = {
        "nvidia-smi": "/usr/bin/nvidia-smi",
        "nvcc": "/usr/bin/nvcc",
        "ibv_devices": "/usr/bin/ibv_devices",
    }
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=binaries.get,
        library_finder=lambda name: "libcudart.so.12" if name == "cudart" else None,
    )

    assert report.status is VerificationStatus.PASS
    assert report.rdma_infiniband.status is VerificationStatus.PASS
    assert report.rdma_infiniband.evidence["availability"] == "UNAVAILABLE"
    assert report.rdma_infiniband.evidence["device_lines"] == ()


def test_optional_rdma_probe_failure_does_not_fail_a_single_gpu_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,name,memory.total,driver_version" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "0, GPU-fixture, NVIDIA H100, 81920, 550.54\n", ""
            )
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.4", "")
        if argv[0].endswith("ibv_devices"):
            return subprocess.CompletedProcess(argv, 1, "", "permission denied")
        raise AssertionError(f"unexpected probe: {argv}")

    monkeypatch.setattr("nodelm.infra.doctor.subprocess.run", run)
    binaries = {
        "nvidia-smi": "/usr/bin/nvidia-smi",
        "nvcc": "/usr/bin/nvcc",
        "ibv_devices": "/usr/bin/ibv_devices",
    }
    report = collect_infrastructure_report(
        workspace=tmp_path,
        finder=binaries.get,
        library_finder=lambda name: "libcudart.so.12" if name == "cudart" else None,
    )

    assert report.status is VerificationStatus.PASS
    assert report.rdma_infiniband.status is VerificationStatus.FAIL
