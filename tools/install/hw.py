"""What the installer needs to know about this machine: NVIDIA GPUs, RAM, CPU.

Every probe reads a file or runs one program and has a parse function of its own, so the parsing is tested without
the hardware (tools/install/test_hw.py).
"""
from __future__ import annotations

import dataclasses
import os
import resource
import shutil
import subprocess
from pathlib import Path


@dataclasses.dataclass
class GPU:
    index: int
    name: str
    vram_mib: int
    compute_cap: str  # "8.6"
    pcie_gen: int
    pcie_width: int
    pcie_width_now: int = 0  # the link's current width; less than pcie_width: the card sits in a narrower slot

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
    thp: str = ""            # transparent huge pages: always / madvise / never ("" unknown)
    memlock_gib: float = -1  # this shell's locked-memory limit; -1 = unlimited


@dataclasses.dataclass
class Probe:
    """What `strata --system-probe` measured."""
    pcie_gbps: dict          # GPU index -> host->device GB/s (< 0: the probe failed)
    p2p: set                 # (a, b): GPU a can read GPU b directly
    ram_gbps: float          # all threads reading RAM at once
    ram_threads: int


NVIDIA_SMI_QUERY = ("index,name,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max,driver_version,"
                    "pcie.link.width.current")


def parse_nvidia_smi(text: str) -> tuple[list, str]:
    """nvidia-smi --query-gpu=<NVIDIA_SMI_QUERY> --format=csv,noheader,nounits -> (GPUs, driver version).
    The current link width (the 8th field) is optional."""
    gpus, driver = [], ""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (7, 8):
            continue
        try:
            gpus.append(GPU(index=int(parts[0]), name=parts[1], vram_mib=int(float(parts[2])),
                            compute_cap=parts[3], pcie_gen=_int(parts[4]), pcie_width=_int(parts[5]),
                            pcie_width_now=_int(parts[7]) if len(parts) == 8 else 0))
        except ValueError:
            continue
        driver = parts[6]
    return gpus, driver


def parse_thp(text: str) -> str:
    """/sys/kernel/mm/transparent_hugepage/enabled ("always [madvise] never") -> the selected mode."""
    for word in text.split():
        if word.startswith("[") and word.endswith("]"):
            return word[1:-1]
    return ""


def parse_probe(text: str) -> Probe:
    """`strata --system-probe` output -> Probe."""
    pcie, p2p, ram, threads = {}, set(), 0.0, 0
    for line in text.splitlines():
        f = line.split()
        try:
            if f[:1] == ["pcie"] and len(f) == 3:
                pcie[int(f[1])] = float(f[2])
            elif f[:1] == ["p2p"] and len(f) == 4 and f[3] == "1":
                p2p.add((int(f[1]), int(f[2])))
            elif f[:1] == ["ram_read"] and len(f) >= 3:
                ram, threads = float(f[1]), int(f[2])
        except ValueError:
            continue
    return Probe(pcie_gbps=pcie, p2p=p2p, ram_gbps=ram, ram_threads=threads)


def free_gib(path: Path) -> float:
    """Free disk space where `path` (or its nearest existing parent) lives."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1024 ** 3


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
    thp = ""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled", encoding="utf-8") as f:
            thp = parse_thp(f.read())
    except OSError:
        pass
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    memlock = -1.0 if soft == resource.RLIM_INFINITY else soft / 1024 ** 3
    return Hardware(gpus=gpus, ram_gib=ram, cpu=cpu, driver=driver, thp=thp, memlock_gib=memlock)


def probe(engine: Path) -> Probe | None:
    """Runs `strata --system-probe` (PCIe per GPU, P2P, RAM bandwidth; ~30 s, the GPUs should be idle)."""
    try:
        out = subprocess.run([str(engine), "--system-probe"], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_probe(out.stdout) if out.returncode == 0 else None


def in_container() -> bool:
    """LXC, Docker, podman, systemd-nspawn: no kernel driver can be installed here, the host provides it."""
    if shutil.which("systemd-detect-virt"):
        r = subprocess.run(["systemd-detect-virt", "--container"], capture_output=True, text=True)
        return r.returncode == 0
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
