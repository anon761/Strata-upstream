#!/usr/bin/env python3
"""Strata fork installer for Debian 12 / 13 (amd64), started by ./install.sh.

  ./install.sh                                  install everything: packages, NVIDIA driver and CUDA toolkit,
                                                Python environment, the engine built for this machine's GPUs
  ./install.sh --gguf /models/X-00001-of-00004.gguf
                                                ... then check that GGUF, propose settings for this hardware,
                                                and after your confirmation prepare it (pack, MTP draft head)
                                                and write its config and a run script
  ./install.sh --check --gguf FILE              only show the hardware and the proposal, change nothing

Options:
  --gguf FILE          a GGUF of Qwen3.8-Flash-Next (any shard of a split model)
  --set KEY=VALUE      change a proposed setting (repeatable), e.g. --set max_context=131072
  --yes                apply the proposal without asking
  --service            also install a systemd service that starts the model at boot
  --name NAME          the model name the API reports (default: from the file name)
  --port N             API port (default 8080)        --host ADDR   listen address (default 127.0.0.1)
  --mtp-repo URL       Hugging Face resolve/main URL of the BF16 checkpoint with the MTP head (a fine-tune's own)
  --cuda-arch LIST     build for these CUDA architectures (e.g. 86,89) instead of the visible GPUs; needed where
                       no GPU is visible (a build container)

The installer needs no environment variables: everything is set by these options, and what the engine needs at run
time is written into its config file.  Running it again only does what is still missing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggufinfo  # noqa: E402
import hw  # noqa: E402
import plan as planner  # noqa: E402
import steps  # noqa: E402

REBOOT_EXIT = 10
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def parse_args(argv: list) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="install.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--service", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mtp-repo", default="")
    ap.add_argument("--cuda-arch", default="")
    a = ap.parse_args(argv)
    if not 1 <= a.port <= 65535:
        ap.error("--port must be 1-65535")
    if a.name and not NAME_RE.match(a.name):
        ap.error("--name: letters, digits, '.', '_' and '-' only")
    if a.cuda_arch and not re.fullmatch(r"\d{2,3}(,\d{2,3})*", a.cuda_arch):
        ap.error("--cuda-arch wants numbers like 86 or 86,89")
    if a.mtp_repo and not a.mtp_repo.startswith("https://"):
        ap.error("--mtp-repo wants an https:// URL")
    if (a.check or a.overrides or a.service) and not a.gguf:
        ap.error("--check, --set and --service need --gguf")
    return a


def apply_overrides(settings: dict, overrides: list) -> dict:
    """--set KEY=VALUE onto the proposal; the value takes the type of the proposed one."""
    out = dict(settings)
    for item in overrides:
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or key not in out:
            raise ValueError(f"--set {item!r}: unknown setting; known: {', '.join(sorted(out))}")
        old = out[key]
        try:
            if key == "pcie_frac":          # a share, or empty for the engine's default
                out[key] = float(raw) if raw else ""
            elif isinstance(old, list):
                out[key] = [int(x) for x in raw.split(",") if x.strip()]
            elif isinstance(old, float):
                out[key] = float(raw)
            elif isinstance(old, int):
                out[key] = int(raw)
            else:
                out[key] = raw
        except ValueError:
            raise ValueError(f"--set {item!r}: {raw!r} is not a valid value for {key}") from None
    return out


def show_hardware(h) -> None:
    steps.say("Hardware:")
    for g in h.gpus:
        steps.say(f"  GPU {g.index}: {g.name}, {g.vram_gib:.0f} GiB, sm_{g.sm}, PCIe {g.pcie_gen}.0 x{g.pcie_width}")
    if not h.gpus:
        steps.say("  no NVIDIA GPU visible")
    steps.say(f"  CPU: {h.cpu.model}, {h.cpu.cores} cores / {h.cpu.threads} threads"
              f"{', AVX-512' if h.cpu.avx512 else ''}")
    steps.say(f"  RAM: {h.ram_gib:.0f} GiB")


def show_model(m) -> None:
    gib = 1024 ** 3
    steps.say(f"Model: {m.name} ({len(m.shards)} file(s), {m.total_bytes / gib:.0f} GiB)")
    steps.say(f"  experts {m.expert_bytes / gib:.0f} GiB in RAM, n-gram table {m.ple_bytes / gib:.0f} GiB ({m.ple_type})")
    for (gu, d), layers in sorted(m.expert_combos.items()):
        steps.say(f"  experts {gu} (gate/up) / {d} (down): {len(layers)} layer(s)")


def show_plan(p, settings: dict) -> None:
    steps.say("Proposed settings:")
    for key, value, why in p.reasons:
        shown = settings[key]
        changed = " (changed by --set)" if shown != value else ""
        text = ",".join(str(x) for x in shown) if isinstance(shown, list) else ("-" if shown == "" else shown)
        steps.say(f"  {key:<26} {str(text):<12} {why}{changed}")
    for n in p.notes:
        steps.say(f"  note: {n}")
    for prob in p.problems:
        steps.say(f"  PROBLEM: {prob}")


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        raise steps.StepError("no terminal to ask on: add --yes to apply the proposal")
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes", "j", "ja")


def prepare_model(a, h) -> int:
    model = ggufinfo.analyze(a.gguf)
    show_hardware(h)
    show_model(model)
    if steps.ENGINE.is_file():
        bad = steps.expert_support(sorted(model.expert_combos), model.n_embd, model.n_ff, ggufinfo.type_id)
    else:
        bad = []
        steps.say("  (the engine is not built yet: its expert-type check runs after the install)")
    p = planner.propose(model, h, bad, port=a.port, host=a.host)
    try:
        settings = apply_overrides(p.settings, a.overrides)
    except ValueError as exc:
        raise steps.StepError(str(exc)) from None
    show_plan(p, settings)
    if p.problems:
        steps.say("This model cannot run here as it is (see PROBLEM above).")
        return 1
    if a.check:
        return 0
    if not a.yes and not confirm("Apply these settings and prepare the model?"):
        steps.say("Nothing changed.")
        return 0
    name = a.name or model.name
    if not NAME_RE.match(name):
        raise steps.StepError(f"the model name {name!r} has characters a config file name cannot take: use --name")
    base = model.shards[0].parent
    pack = base / "strata-pack"
    steps.build_pack(model.shards[0], pack)
    mtp_rt = steps.build_mtp(base / "strata-mtp", a.mtp_repo)
    configs = steps.ROOT / "configs"
    configs.mkdir(exist_ok=True)
    config = configs / f"{name}.json"
    logs = steps.ROOT / "logs"
    logs.mkdir(exist_ok=True)
    cfg = steps.engine_config(model, settings, pack, mtp_rt, name, logs / f"strata-{name}.log")
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    script = steps.write_launchers(name, config, int(settings["port"]), a.service)
    steps.say(f"Config: {config}")
    steps.say(f"Start:  {script}" + (f"   (service strata-{name} is running)" if a.service else ""))
    steps.say(f"API:    http://{settings['host']}:{settings['port']}/v1")
    return 0


def main(argv: list) -> int:
    a = parse_args(argv)
    if a.check:  # changes nothing: no root needed
        try:
            return prepare_model(a, hw.detect())
        except (steps.StepError, ggufinfo.ModelError) as exc:
            steps.say(f"\nStopped: {exc}")
            return 1
    if os.geteuid() != 0:
        steps.say("Run the installer as root (sudo ./install.sh ...): it installs system packages.")
        return 1
    distro, version = steps.debian_release()
    if distro != "debian" or version not in ("12", "13"):
        steps.say(f"This installer supports Debian 12 and 13, not {distro} {version}.")
        return 1
    h = hw.detect()
    try:
        container = hw.in_container()
        steps.base_packages()
        steps.nvidia_repo(version)
        if steps.driver(container, h.driver, build_only=bool(a.cuda_arch)):
            steps.say("")
            steps.say("The NVIDIA driver is installed. Reboot, then run the same command again to continue.")
            return REBOOT_EXIT
        steps.cuda_toolkit()
        steps.python_env()
        archs = sorted({int(x) for x in a.cuda_arch.split(",")}) if a.cuda_arch else \
            sorted({g.sm for g in h.gpus if g.sm >= planner.MIN_SM})
        if not archs:
            raise steps.StepError("no RTX 20 or newer GPU to build for: give --cuda-arch")
        steps.build_engine(archs)
        steps.say(f"Engine: {steps.ENGINE}")
        if a.gguf:
            return prepare_model(a, h)
        steps.say("Installed. Next: ./install.sh --gguf /path/to/model.gguf")
        return 0
    except (steps.StepError, ggufinfo.ModelError) as exc:
        steps.say(f"\nStopped: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
