"""The settings the installer proposes for one GGUF on this machine: a pure function of the model and the hardware,
with a reason for every choice (tools/install/test_plan.py covers the rules).

The numbers come from measurements: ~5 GiB of every card go to the dense weights, buffers and graphs; the KV cache
takes 13 x 1056 bytes per context token at int8 (12 QSA layers + the draft layer, setup.py low_ram_gpu_gb); the
engine keeps all routed experts in a pinned RAM arena and wants ~24 GiB beside it for the OS, the engine and the
file cache (setup.py resident_budget_gib); on 2x RTX 3090 a second conversation slot of 65536 cells costs ~1.1 GiB
VRAM and four parallel requests reach ~97-99 tok/s against ~82 serial (docs/BATCH2.md).
"""
from __future__ import annotations

import dataclasses

GIB = 1024 ** 3
DENSE_GIB_PER_GPU = 5.0           # dense weights, buffers, CUDA graphs per card
KV_BYTES_PER_TOKEN = 13 * 1056    # --kv int8
RAM_HEADROOM_GIB = 24.0           # OS, engine, file cache beside the expert arena
MIN_GPU_VRAM_GIB = 11.0           # below this nothing useful is left for the expert cache
MIN_SM = 75                       # RTX 20 (Turing) or newer
CONTEXTS = (262000, 131072, 65536, 32768)  # 262000: the model's 262144 less room for the engine's own tokens
KV_SHARE = 0.25                   # at most this share of a card's free VRAM for the KV cache, the rest caches experts
BATCH2_CELLS = 65536
BATCH2_MIN_VRAM_GIB = 20.0
SERVER_CPU_THREADS = 32           # a CPU this wide computes expert-cache misses faster than PCIe fetches them


@dataclasses.dataclass
class Plan:
    settings: dict
    reasons: list   # (setting, value, why), in display order
    problems: list  # anything here blocks the install
    notes: list     # worth knowing, not blocking


def propose(model, hw, unsupported_combos: list, port: int = 8080, host: str = "127.0.0.1") -> Plan:
    """model: ggufinfo.ModelInfo; hw: hw.Hardware; unsupported_combos: the (gate/up, down) types the engine has no
    GPU kernels for (asked with `strata --expert-support`)."""
    s: dict = {}
    reasons: list = []
    problems: list = []
    notes: list = []

    def choose(key, value, why):
        s[key] = value
        reasons.append((key, value, why))

    for gu, d in unsupported_combos:
        layers = model.expert_combos.get((gu, d), [])
        problems.append(f"experts {gu} (gate/up) / {d} (down) in {len(layers)} layer(s): this engine has no GPU "
                        f"kernels for them - use another quantization of the model")
    if not hw.cpu.avx2_ok:
        problems.append(f"the CPU ({hw.cpu.model}) lacks AVX2/FMA/F16C, which every CPU expert kernel needs")

    usable = [g for g in hw.gpus if g.sm >= MIN_SM and g.vram_gib >= MIN_GPU_VRAM_GIB]
    for g in hw.gpus:
        if g not in usable:
            notes.append(f"GPU {g.index} ({g.name}, {g.vram_gib:.0f} GiB, sm_{g.sm}) is left out: Strata needs an RTX 20 "
                         f"or newer with at least {MIN_GPU_VRAM_GIB:.0f} GiB")
    if not usable:
        problems.append("no usable NVIDIA GPU found (RTX 20 or newer, at least 11 GiB)")
        usable_vram = []
    else:
        usable_vram = [g.vram_gib for g in usable]
    n_gpu = len(usable)
    choose("gpu", [g.index for g in usable],
           f"{n_gpu} usable card(s)" + (", the layers split across them" if n_gpu > 1 else ""))
    choose("layer_split", "auto" if n_gpu > 1 else "", "split by free VRAM" if n_gpu > 1 else "one card, no split")

    arena_gib = model.expert_bytes / GIB
    ple_gib = model.ple_bytes / GIB
    need = arena_gib + RAM_HEADROOM_GIB
    if hw.ram_gib < need:
        problems.append(f"{hw.ram_gib:.0f} GiB RAM, but the model's experts take {arena_gib:.0f} GiB and the engine "
                        f"keeps them all in RAM (+{RAM_HEADROOM_GIB:.0f} GiB for the rest): {need:.0f} GiB needed - "
                        f"use a smaller quantization or add RAM")

    # Context: the longest that keeps the KV cache to KV_SHARE of every card's free VRAM.
    ctx = CONTEXTS[-1]
    if usable_vram:
        smallest_free = min(usable_vram) - DENSE_GIB_PER_GPU
        for c in CONTEXTS:
            if c > model.context_length:
                continue
            kv_per_gpu = c * KV_BYTES_PER_TOKEN / GIB / n_gpu
            if kv_per_gpu <= KV_SHARE * smallest_free:
                ctx = c
                break
    kv_gib = ctx * KV_BYTES_PER_TOKEN / GIB
    choose("max_context", ctx, f"KV cache {kv_gib:.1f} GiB (int8) leaves most of the VRAM to the expert cache")
    choose("kv", "int8", "int8 codes: half of fp16's memory, measured lossless enough for code")
    choose("expert_cache", "auto", "the engine fills the VRAM left after the dense weights and the KV cache")

    rest = hw.ram_gib - arena_gib - RAM_HEADROOM_GIB
    if rest >= ple_gib + 8:
        choose("ple_io", "ram", f"the n-gram table ({ple_gib:.0f} GiB, {model.ple_type}) fits the RAM: no SSD reads")
        rest -= ple_gib
    else:
        choose("ple_io", "", f"the n-gram table ({ple_gib:.0f} GiB) stays on the SSD: not enough RAM beside the experts")

    if hw.cpu.threads >= SERVER_CPU_THREADS:
        choose("pcie_frac", 0, f"{hw.cpu.threads} CPU threads compute expert-cache misses faster than PCIe fetches them")
    else:
        choose("pcie_frac", "", "engine default: a desktop CPU fetches part of the misses over PCIe")

    park_mib = int(max(0.0, rest * 0.5) * 1024) // 1024 * 1024
    park_mib = min(park_mib, 65536)
    if park_mib >= 4096:
        slots = 4 if park_mib < 16384 else 8 if park_mib < 32768 else 20
        choose("conversation_cache_mib", park_mib,
               f"half of the free RAM parks conversations, so agents taking turns skip re-reading their context")
        choose("conversation_cache_slots", slots, "parked conversations at most")
    else:
        choose("conversation_cache_mib", 0, "too little free RAM to park conversations")
        choose("conversation_cache_slots", 4, "unused while parking is off")

    if n_gpu >= 2 and min(usable_vram) >= BATCH2_MIN_VRAM_GIB:
        choose("batch2_cells", BATCH2_CELLS, "two requests decode together (~1.1 GiB VRAM; 4 parallel: ~97-99 tok/s "
                                             "vs ~82 serial on 2x RTX 3090)")
    else:
        choose("batch2_cells", 0, "requests run one after the other (Batch-2 needs two cards with 20+ GiB)")

    choose("spec", 4, "MTP speculative decoding, 4-token windows")
    choose("spec_min_p", 0.5, "adapts the window to how sure the draft is")
    choose("prefill", "auto", "batched prompt reading")
    choose("vram_reserve_mib", 700, "kept free on every card for the driver and other programs")
    choose("port", port, "API and chat page")
    choose("host", host, "reachable from this machine only" if host == "127.0.0.1" else "reachable from the network")
    return Plan(settings=s, reasons=reasons, problems=problems, notes=notes)
