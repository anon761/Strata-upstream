"""The installer's system steps on Debian 12/13: packages, the NVIDIA driver and CUDA toolkit, the Python
environment, the engine build, and preparing a model (pack + MTP draft head).  Every step checks first whether its
result is already there, so running the installer again only does what is missing.

Nothing here reads or needs an environment variable: tools are called by absolute path, and what the engine needs at
run time (CUDA library directories, GPUs) goes into its config file, which serve/server.py applies itself.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_VERSION = "13.3"                      # the toolkit Strata is built and tested with
CUDA_HOME = Path(f"/usr/local/cuda-{CUDA_VERSION}")
NVCC = CUDA_HOME / "bin" / "nvcc"
CUDA_PACKAGE = "cuda-toolkit-" + CUDA_VERSION.replace(".", "-")
DRIVER_PACKAGE = "nvidia-open"             # NVIDIA's open kernel modules: RTX 20 (Turing) and newer
KEYRING_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/cuda-keyring_1.1-1_all.deb"
BASE_PACKAGES = ["build-essential", "git", "ca-certificates", "curl", "gnupg", "pciutils", "pkg-config",
                 "python3", "python3-venv", "python3-dev"]
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"
ENGINE = ROOT / "engine" / "strata"
BUILD_DIR = ROOT / "build-install"
STATE = ROOT / ".install"


class StepError(Exception):
    """A step failed; the message says what to do."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def run(cmd: list, cwd: Path | None = None, quiet: bool = False) -> None:
    say("  $ " + " ".join(str(c) for c in cmd))
    # the caller's environment passes through (a proxy, say) but nothing in it is needed
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    env["PATH"] = env.get("PATH") or "/usr/sbin:/usr/bin:/sbin:/bin"
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.DEVNULL if quiet else None)
    if r.returncode != 0:
        raise StepError(f"'{' '.join(str(c) for c in cmd[:3])} ...' failed (exit {r.returncode})")


def installed(package: str) -> bool:
    r = subprocess.run(["dpkg-query", "-W", "-f=${Status}", package], capture_output=True, text=True)
    return r.returncode == 0 and "install ok installed" in r.stdout


def debian_release() -> tuple[str, str]:
    """(ID, VERSION_ID) from /etc/os-release."""
    info = {}
    with open("/etc/os-release", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            info[k] = v.strip('"')
    return info.get("ID", ""), info.get("VERSION_ID", "")


def apt_install(packages: list) -> None:
    missing = [p for p in packages if not installed(p)]
    if not missing:
        return
    run(["apt-get", "update"], quiet=True)
    run(["apt-get", "install", "-y", "--no-install-recommends", *missing])


# ---------------------------------------------------------------------------------------------------- packages
def base_packages() -> None:
    say("[1/6] system packages")
    apt_install(BASE_PACKAGES)


def nvidia_repo(version_id: str) -> None:
    say("[2/6] NVIDIA package repository")
    if installed("cuda-keyring"):
        return
    distro = f"debian{version_id}"
    deb = STATE / "cuda-keyring.deb"
    STATE.mkdir(exist_ok=True)
    url = KEYRING_URL.format(distro=distro)
    say(f"  downloading {url}")
    urllib.request.urlretrieve(url, deb)
    run(["dpkg", "-i", deb])
    run(["apt-get", "update"], quiet=True)


def secure_boot_on() -> bool:
    if not shutil.which("mokutil"):
        return False
    r = subprocess.run(["mokutil", "--sb-state"], capture_output=True, text=True)
    return "enabled" in r.stdout.lower()


def driver(container: bool, driver_version: str, build_only: bool) -> bool:
    """Make sure an NVIDIA driver runs. Returns True when a reboot is needed before the installer can go on."""
    say("[3/6] NVIDIA driver")
    if driver_version:
        say(f"  driver {driver_version} is running")
        return False
    if container:
        if build_only:
            say("  no GPU visible in this container: building only (--cuda-arch)")
            return False
        raise StepError("no NVIDIA GPU is visible in this container. A container cannot load a kernel driver: install "
                        "the driver on the host and pass the GPUs into the container (Docker: --gpus all; LXC: the "
                        "/dev/nvidia* devices), then run ./install.sh again - or build only with --cuda-arch")
    if installed(DRIVER_PACKAGE):
        raise StepError("the NVIDIA driver is installed but not running: reboot, then run ./install.sh again "
                        "(with Secure Boot on, the unsigned module cannot load: disable it or enroll the module key)")
    kernel = os.uname().release
    apt_install([f"linux-headers-{kernel}", DRIVER_PACKAGE])
    if secure_boot_on():
        say("  Secure Boot is on: the driver's module must be signed (mokutil --import) or Secure Boot disabled")
    return True


def cuda_toolkit() -> None:
    say(f"[4/6] CUDA toolkit {CUDA_VERSION}")
    if NVCC.is_file():
        return
    apt_install([CUDA_PACKAGE])
    if not NVCC.is_file():
        raise StepError(f"{CUDA_PACKAGE} is installed but {NVCC} is missing")


def cuda_lib_dirs() -> list[str]:
    """What the engine needs on its library path at run time; the config carries it, serve/server.py applies it."""
    return [str(d) for d in (CUDA_HOME / "lib64", CUDA_HOME / "targets" / "x86_64-linux" / "lib") if d.is_dir()]


# ---------------------------------------------------------------------------------------------------- python
def python_env() -> None:
    say("[5/6] Python environment (.venv)")
    req = ROOT / "requirements.txt"
    stamp = STATE / "venv.sha256"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    if PY.is_file() and stamp.is_file() and stamp.read_text() == digest:
        return
    if not PY.is_file():
        run(["/usr/bin/python3", "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "--upgrade", "pip"], quiet=True)
    run([PY, "-m", "pip", "install", "-r", req])
    STATE.mkdir(exist_ok=True)
    stamp.write_text(digest)


# ---------------------------------------------------------------------------------------------------- engine
def source_revision() -> str:
    """The checkout's commit (+dirty with local engine changes): a new one rebuilds the engine.  safe.directory: the
    installer runs as root, the checkout usually belongs to a user, and git refuses such a repository otherwise."""
    git = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]
    r = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        return "unknown+dirty"  # no git: rebuild every time rather than keep a stale engine
    dirty = subprocess.run([*git, "status", "--porcelain", "--", "src", "include", "CMakeLists.txt"],
                           capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def build_engine(archs: list) -> None:
    say(f"[6/6] Strata engine for sm_{', sm_'.join(str(a) for a in archs)}")
    stamp = STATE / "engine.stamp"
    want = f"{source_revision()} {';'.join(str(a) for a in archs)} cuda-{CUDA_VERSION}"
    if ENGINE.is_file() and stamp.is_file() and stamp.read_text() == want and "+dirty" not in want:
        say("  up to date")
        return
    cmake, ninja = VENV / "bin" / "cmake", VENV / "bin" / "ninja"
    run([cmake, "-S", ROOT, "-B", BUILD_DIR, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
         "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={NVCC}",
         f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(str(a) for a in archs)}",
         "-DSTRATA_MMQ_KQUANTS=ON", "-DSTRATA_ORCA_Q4KS_MMQ=ON", "-DSTRATA_BUILD_TESTS=OFF"])
    run([ninja, "-C", BUILD_DIR, "strata"])
    ENGINE.parent.mkdir(exist_ok=True)
    tmp = ENGINE.with_suffix(".new")
    shutil.copy2(BUILD_DIR / "strata", tmp)
    os.replace(tmp, ENGINE)
    stamp.write_text(want)


def expert_support(combos: list, n_embd: int, n_ff: int, type_id) -> list:
    """The (gate/up, down) type combinations the built engine has no GPU kernels for."""
    bad = []
    for gu, d in combos:
        r = subprocess.run([str(ENGINE), "--expert-support", f"{type_id(gu)},{type_id(d)},{n_embd},{n_ff}"],
                           capture_output=True, text=True)
        if r.returncode == 3:
            bad.append((gu, d))
        elif r.returncode != 0:
            raise StepError(f"the engine could not check expert types {gu}/{d}: {r.stderr.strip()}")
    return bad


# ---------------------------------------------------------------------------------------------------- model
def build_pack(first_shard: Path, pack: Path) -> None:
    if (pack / "native_experts.txt").is_file():
        say(f"  pack: {pack} (exists)")
        return
    say(f"  pack: building {pack} (reads every shard, takes a while)")
    run([PY, ROOT / "tools" / "iq_pack.py", "--gguf", first_shard, "--out", pack, "--compat-bf16"], cwd=ROOT)
    if not (pack / "native_experts.txt").is_file():
        raise StepError(f"the pack in {pack} is incomplete")


def mtp_complete(rt: Path) -> bool:
    return (rt / "experts.bin").is_file() and (rt / "draft_vocab.bin").is_file()


def build_mtp(mtp_dir: Path, repo: str) -> Path:
    rt = mtp_dir / "rt"
    if mtp_complete(rt):
        say(f"  MTP draft head: {rt} (exists)")
        return rt
    say(f"  MTP draft head: building {rt} (downloads ~5 GB of MTP tensors)")
    mtp_dir.mkdir(parents=True, exist_ok=True)
    fetch = [PY, ROOT / "tools" / "mtp_fetch.py", "fetch", "--out", mtp_dir]
    if repo:
        fetch += ["--repo", repo]
    gguf = mtp_dir / "mtp-q2_0.gguf"
    run(fetch, cwd=ROOT)
    run([PY, ROOT / "tools" / "mtp_pack.py", "--src", mtp_dir, "--experts", "q2_0", "--out", gguf], cwd=ROOT)
    run([PY, ROOT / "tools" / "mtp_rt.py", "--gguf", gguf, "--out", rt], cwd=ROOT)
    shutil.copyfile(ROOT / "data" / "draft_vocab.bin", rt / "draft_vocab.bin")
    if not mtp_complete(rt):
        raise StepError(f"the MTP draft head in {rt} is incomplete")
    return rt


def engine_config(model, settings: dict, pack: Path, mtp_rt: Path, served_name: str, log: Path) -> dict:
    """The serve/server.py --config file: everything the engine needs, absolute paths, no environment."""
    args = ["--pack", str(pack), "--native", str(model.shards[0]),
            "--native-head-gguf", str(model.head_shard), "--ple-gguf", str(model.ple_shard),
            "--expert-profile", str(ROOT / "data" / "expert-profile.bin"),
            "--expert-cache", str(settings["expert_cache"]), "--prefill", str(settings["prefill"]),
            "--spec", str(settings["spec"]), "--spec-min-p", str(settings["spec_min_p"]),
            "--max-context", str(settings["max_context"]), "--kv", str(settings["kv"]),
            "--vram-reserve-mib", str(settings["vram_reserve_mib"]), "--mtp", str(mtp_rt)]
    if settings["pcie_frac"] != "":
        args += ["--pcie-frac", str(settings["pcie_frac"])]
    if settings["ple_io"]:
        args += ["--ple-io", str(settings["ple_io"])]
    if settings["conversation_cache_mib"] > 0:
        args += ["--conversation-cache-mib", str(settings["conversation_cache_mib"]),
                 "--conversation-cache-slots", str(settings["conversation_cache_slots"])]
    if settings.get("batch", 0) > 0:
        args += ["--batch", str(settings["batch"])]
    if settings["kv_resident"] > 0:
        args += ["--kv-resident", str(settings["kv_resident"])]
    if settings["resident_budget_gib"] > 0:   # the mapped mode with a RAM budget: experts from the GGUF in place
        args += ["--resident-budget-gib", str(settings["resident_budget_gib"])]
    cfg = {
        "exe": str(ENGINE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
        "model_name": served_name, "log": str(log), "lib_dirs": cuda_lib_dirs(),
        "host": str(settings["host"]), "gpu": list(settings["gpu"]),
    }
    if settings["layer_split"]:
        cfg["layer_split"] = settings["layer_split"]
    if settings["split_skip_if_fits"]:
        cfg["split_skip_if_fits"] = True
    if settings["api_key"]:
        cfg["api_key"] = str(settings["api_key"])
    return cfg


SERVICE_TEMPLATE = """[Unit]
Description=Strata ({name})
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={python} {server} --engine strata --config {config} --port {port}
WorkingDirectory={root}
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
"""


def write_launchers(name: str, config: Path, port: int, service: bool) -> Path:
    script = ROOT / f"run-{name}.sh"
    script.write_text(f"#!/bin/sh\nexec {PY} {ROOT / 'serve' / 'server.py'} --engine strata --config {config} "
                      f"--port {port} \"$@\"\n")
    script.chmod(0o755)
    if service:
        unit = Path(f"/etc/systemd/system/strata-{name}.service")
        unit.write_text(SERVICE_TEMPLATE.format(name=name, python=PY, server=ROOT / "serve" / "server.py",
                                                config=config, port=port, root=ROOT))
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "enable", "--now", unit.name])
    return script
