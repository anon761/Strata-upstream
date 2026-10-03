"""What the installer needs to know about this machine: NVIDIA GPUs, RAM, CPU.

Every probe reads a file or runs one program and has a parse function of its own, so the parsing is tested without
the hardware (tools/install/test_hw.py).
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess


@dataclasses.dataclass
class GPU:
    index: int
    name: str
    vram_mib: int
    compute_cap: str  # "8.6"
    pcie_gen: int
    pcie_width: int

    @property
    def sm(self) -> int:
        """The CUDA architecture number, 86 for 8.6."""
        major, _, minor = self.compute_cap.partition(".")
        return int(major) * 10 + int(minor or 0)

    @property
    def vram_gib(self) -> float:
        return self.vram_mib / 1024


@dataclasses.dataclass
class CPU:
    model: str
    cores: int
    threads: int
    flags: set

    @property
    def avx2_ok(self) -> bool:
        """What every CPU expert kernel of the engine needs (it refuses to start without)."""
        return {"avx2", "fma", "f16c"} <= self.flags

    @property
    def avx512(self) -> bool:
        return {"avx512f", "avx512bw", "avx512vl"} <= self.flags


@dataclasses.dataclass
class Hardware:
    gpus: list
    ram_gib: float
    cpu: CPU
    driver: str  # "" without a working NVIDIA driver


NVIDIA_SMI_QUERY = "index,name,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max,driver_version"


def parse_nvidia_smi(text: str) -> tuple[list, str]:
    """nvidia-smi --query-gpu=<NVIDIA_SMI_QUERY> --format=csv,noheader,nounits -> (GPUs, driver version)."""
    gpus, driver = [], ""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            gpus.append(GPU(index=int(parts[0]), name=parts[1], vram_mib=int(float(parts[2])),
                            compute_cap=parts[3], pcie_gen=_int(parts[4]), pcie_width=_int(parts[5])))
        except ValueError:
            continue
        driver = parts[6]
    return gpus, driver


def _int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0  # "[N/A]" in some containers


def parse_meminfo(text: str) -> float:
    """/proc/meminfo -> total RAM in GiB."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def parse_cpuinfo(text: str) -> CPU:
    """/proc/cpuinfo -> model, physical cores, threads, flags (of the first processor)."""
    model, flags, threads = "", set(), 0
    cores_per_package: dict[str, int] = {}
    package = "0"
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            threads += 1
        elif key == "model name" and not model:
            model = value
        elif key == "flags" and not flags:
            flags = set(value.split())
        elif key == "physical id":
            package = value
        elif key == "cpu cores":
            cores_per_package[package] = int(value)
    cores = sum(cores_per_package.values()) or threads
    return CPU(model=model, cores=cores, threads=threads, flags=flags)


def detect() -> Hardware:
    gpus, driver = [], ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode == 0:
                gpus, driver = parse_nvidia_smi(out.stdout)
        except (OSError, subprocess.TimeoutExpired):
            pass
    with open("/proc/meminfo", encoding="utf-8") as f:
        ram = parse_meminfo(f.read())
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        cpu = parse_cpuinfo(f.read())
    return Hardware(gpus=gpus, ram_gib=ram, cpu=cpu, driver=driver)


def in_container() -> bool:
    """LXC, Docker, podman, systemd-nspawn: no kernel driver can be installed here, the host provides it."""
    if shutil.which("systemd-detect-virt"):
        r = subprocess.run(["systemd-detect-virt", "--container"], capture_output=True, text=True)
        return r.returncode == 0
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
