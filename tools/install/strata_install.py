#!/usr/bin/env python3
"""Strata fork installer for Debian 12 / 13 (amd64), started by ./install.sh.

  ./install.sh                                  install everything: packages, NVIDIA driver and CUDA toolkit,
                                                Python environment, the engine built for this machine's GPUs
  ./install.sh --gguf /models/X-00001-of-00004.gguf
                                                ... then check that GGUF, propose settings for this hardware,
                                                and after your confirmation prepare it (pack, MTP draft head)
                                                and write its config and a run script
  ./install.sh --check --gguf FILE              only show the hardware and the proposal, change nothing
  ./install.sh --system-check [--gguf FILE]     check this machine and print the best settings for it - for FILE,
                                                or for the common quantizations of the model without one; offers
                                                to measure PCIe and RAM bandwidth first (y/n; the GPUs should be
                                                idle). Changes nothing, needs no root

Options:
  --gguf FILE          a GGUF of Qwen3.8-Flash-Next (any shard of a split model)
  --set KEY=VALUE      change a proposed setting (repeatable), e.g. --set max_context=131072
  --yes                apply the proposal without asking (with --system-check: measure without asking)
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
    ap.add_argument("--system-check", action="store_true")
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
    if (a.check or a.service) and not a.gguf:
        ap.error("--check and --service need --gguf")
    if a.overrides and not (a.gguf or a.system_check):
        ap.error("--set needs --gguf or --system-check")
    if a.system_check and (a.check or a.service):
        ap.error("--system-check changes nothing: not with --check or --service")
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
            if isinstance(old, bool):
                if raw.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                    raise ValueError
                out[key] = raw.lower() in ("true", "1", "yes", "on")
            elif key == "pcie_frac":          # a share, or empty for the engine's default
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


def show_hardware(h, probe=None) -> None:
    steps.say("Hardware:")
    for g in h.gpus:
        now = f" (running x{g.pcie_width_now})" if g.pcie_width_now and g.pcie_width_now != g.pcie_width else ""
        meas = ""
        if probe and g.index in probe.pcie_gbps:
            bw = probe.pcie_gbps[g.index]
            meas = f", measured {bw:.1f} GB/s host->GPU" if bw >= 0 else ", not measured (its memory is in use: stop the engine first)"
        steps.say(f"  GPU {g.index}: {g.name}, {g.vram_gib:.0f} GiB, sm_{g.sm}, PCIe {g.pcie_gen}.0 x{g.pcie_width}"
                  f"{now}{meas}")
    if not h.gpus:
        steps.say("  no NVIDIA GPU visible")
    if probe and len(h.gpus) > 1:
        pairs = sorted(probe.p2p)
        steps.say("  P2P between the GPUs: " + (", ".join(f"{a}->{b}" for a, b in pairs) if pairs else
                                                 "none (the layer split does not need it)"))
    steps.say(f"  CPU: {h.cpu.model}, {h.cpu.cores} cores / {h.cpu.threads} threads"
              f"{', AVX-512' if h.cpu.avx512 else ''}{'' if h.cpu.avx2_ok else ', NO AVX2 (required)'}")
    ram = f"  RAM: {h.ram_gib:.0f} GiB"
    if probe and probe.ram_gbps > 0:
        ram += f", measured {probe.ram_gbps:.0f} GB/s read ({probe.ram_threads} threads)"
    steps.say(ram)


def context_label(tokens: int) -> str:
    """262000 -> 262K (the model's whole window), 131072 -> 128K."""
    return "262K" if tokens >= 262000 else f"{tokens // 1024}K"


def show_system(h) -> None:
    steps.say("System:")
    steps.say(f"  NVIDIA driver: {h.driver or 'none working'}")
    thp = {"always": "on (always)", "madvise": "on (madvise)", "never": "OFF", "": "unknown"}.get(h.thp, h.thp)
    steps.say(f"  transparent huge pages: {thp}")
    steps.say("  memlock limit (this shell): " + ("unlimited" if h.memlock_gib < 0 else f"{h.memlock_gib:.1f} GiB"))
    steps.say(f"  free disk where Strata lives: {hw.free_gib(steps.ROOT):.0f} GiB")


def maybe_probe(a):
    """--system-check: measure PCIe and RAM bandwidth with the built engine, after a y/n question (or --yes)."""
    if not steps.ENGINE.is_file():
        steps.say("(the engine is not built yet: no bandwidth measurement - the settings below follow the hardware)")
        return None
    if not a.yes:
        if not sys.stdin.isatty():
            steps.say("(no terminal to ask on: no measurement; --yes measures without asking)")
            return None
        if not confirm("Measure PCIe and RAM bandwidth first (~30 s; the GPUs should be idle)?"):
            return None
    steps.say("Measuring ...")
    p = hw.probe(steps.ENGINE)
    if p is None:
        steps.say("(the measurement failed - the settings below follow the hardware)")
    return p


def show_reference(h, probe, a) -> None:
    """--system-check without --gguf: the proposal for each common quantization of the model."""
    steps.say("Best settings for the common GGUFs of Qwen3.8-Flash-Next on this machine:")
    shared = None
    for ref in planner.REFERENCE_MODELS:
        p = planner.propose(ref, h, [], port=a.port, host=a.host, probe=probe, api_key="(generated at install)")
        try:
            st = apply_overrides(p.settings, a.overrides)
        except ValueError as exc:
            raise steps.StepError(str(exc)) from None
        if p.problems:
            steps.say(f"  {ref.name:<20} does not run here: {p.problems[0]}")
            continue
        shared = shared or (p, st)
        parts = [f"{context_label(st['max_context'])} context"]
        if st["resident_budget_gib"]:
            parts.append(f"RAM-budget mode ({st['resident_budget_gib']} GiB of {ref.expert_bytes / planner.GIB:.0f})")
        if st["kv_resident"]:
            parts.append("KV streaming")
        parts.append("n-gram table in RAM" if st["ple_io"] == "ram" else "n-gram table on SSD")
        if st["conversation_cache_mib"]:
            parts.append(f"parking {st['conversation_cache_mib'] // 1024} GiB")
        if st["batch2_cells"]:
            parts.append("Batch-2")
        if st["split_skip_if_fits"]:
            parts.append("skip the split if the first card fits")
        steps.say(f"  {ref.name:<20} " + ", ".join(parts))
    if shared is None:
        return
    p, st = shared
    steps.say("For every model that runs here:")
    for key in ("gpu", "layer_split", "pcie_frac", "kv", "spec", "spec_min_p", "prefill", "api_key"):
        value = st[key]
        why = next(w for k, _, w in p.reasons if k == key)
        text = ",".join(str(x) for x in value) if isinstance(value, list) else ("-" if value == "" else value)
        steps.say(f"  {key:<26} {str(text):<12} {why}")
    for n in p.notes:
        if "do not fit in RAM" not in n:
            steps.say(f"  note: {n}")
    steps.say("For one model exactly: ./install.sh --system-check --gguf /path/to/model.gguf")


def system_check(a) -> int:
    h = hw.detect()
    show_system(h)
    probe = maybe_probe(a)
    if a.gguf:
        return prepare_model(a, h, probe)
    show_hardware(h, probe)
    show_reference(h, probe, a)
    return 0


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


def prepare_model(a, h, probe=None) -> int:
    model = ggufinfo.analyze(a.gguf)
    show_hardware(h, probe)
    show_model(model)
    if steps.ENGINE.is_file():
        bad = steps.expert_support(sorted(model.expert_combos), model.n_embd, model.n_ff, ggufinfo.type_id)
    else:
        bad = []
        steps.say("  (the engine is not built yet: its expert-type check runs after the install)")
    shown_only = a.check or a.system_check   # a key is made when the install writes the config, not for a preview
    p = planner.propose(model, h, bad, port=a.port, host=a.host, probe=probe,
                        api_key="(generated at install)" if shown_only else None)
    try:
        settings = apply_overrides(p.settings, a.overrides)
    except ValueError as exc:
        raise steps.StepError(str(exc)) from None
    show_plan(p, settings)
    if p.problems:
        steps.say("This model cannot run here as it is (see PROBLEM above).")
        return 1
    if a.check or a.system_check:
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
    if cfg.get("api_key"):
        config.chmod(0o600)   # it holds the API key
    script = steps.write_launchers(name, config, int(settings["port"]), a.service)
    steps.say(f"Config: {config}")
    steps.say(f"Start:  {script}" + (f"   (service strata-{name} is running)" if a.service else ""))
    steps.say(f"API:    http://{settings['host']}:{settings['port']}/v1")
    return 0


def main(argv: list) -> int:
    a = parse_args(argv)
    if a.system_check:  # changes nothing: no root needed
        try:
            return system_check(a)
        except (steps.StepError, ggufinfo.ModelError) as exc:
            steps.say(f"\nStopped: {exc}")
            return 1
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
