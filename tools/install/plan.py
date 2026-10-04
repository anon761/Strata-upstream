"""The settings the installer proposes for one GGUF on this machine: a pure function of the model, the hardware and
(optionally) what `strata --system-probe` measured, with a reason for every choice (tools/install/test_install.py
covers the rules).

The numbers come from measurements: ~5 GiB of every card go to the dense weights, buffers and graphs; the KV cache
takes 13 x 1056 bytes per context token at int8 (12 QSA layers + the draft layer, setup.py low_ram_gpu_gb); the
engine keeps all routed experts in a pinned RAM arena and wants ~24 GiB beside it for the OS, the engine and the
file cache (setup.py resident_budget_gib); on 2x RTX 3090 a second conversation slot of 65536 cells costs ~1.1 GiB
VRAM and four parallel requests reach ~97-99 tok/s against ~82 serial (docs/BATCH2.md).

Features that only work once switched on are proposed where they fit: KV streaming (setup.py: from 64K up, +23%
decode at 262K on a 12 GB card), the RAM-budget mode for experts that do not fit in RAM (docs/UNSLOTH_Q4.md, one
GPU), skipping the split when the first card holds every expert (docs/MULTI_GPU.md), an API key for a server others
can reach; and notes for what the system itself has to allow (transparent huge pages, the memlock limit).
"""
from __future__ import annotations

import dataclasses
import secrets

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
STRONG_RAM_GBPS = 80.0            # measured: RAM this fast feeds the CPU pool faster than a x16 link feeds the GPU
SLOW_LINK_GBPS = 4.0              # measured below this (an x1/x4 link), no miss pays its trip over PCIe
KV_STREAM_MIN_CONTEXT = 65536     # KV streaming: from this context up (setup.py)
KV_RESIDENT_CELLS = 32768         # the cells per QSA layer kept in VRAM while streaming (setup.py)
BUDGET_MIN_RAM_GIB = 48.0         # the RAM-budget mode needs at least this much RAM (docs/UNSLOTH_Q4.md)
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclasses.dataclass
class Plan:
    settings: dict
    reasons: list   # (setting, value, why), in display order
    problems: list  # anything here blocks the install
    notes: list     # worth knowing, not blocking


@dataclasses.dataclass
class RefModel:
    """What propose() reads of a model, for the system check without a GGUF: sizes measured with ggufinfo."""
    name: str
    expert_bytes: int
    ple_bytes: int
    ple_type: str
    context_length: int = 262144
    expert_combos: dict = dataclasses.field(default_factory=dict)


# The ordinary GGUFs of Qwen3.8-Flash-Next this fork runs, measured with tools/install/ggufinfo.py (2026-10-03).
REFERENCE_MODELS = (
    RefModel("Unsloth UD-Q4_K_XL", int(71.7 * GIB), int(26.8 * GIB), "IQ4_NL"),
    RefModel("Swift-1.5 Q4_K_L", int(75.0 * GIB), int(50.7 * GIB), "Q8_0"),
    RefModel("Q5_K_M", int(85.5 * GIB), int(35.8 * GIB), "Q5_1"),
    RefModel("Unsloth UD-Q6_K_XL", int(101.7 * GIB), int(50.7 * GIB), "Q8_0"),
)


def kv_gib(context: int) -> float:
    return context * KV_BYTES_PER_TOKEN / GIB


def propose(model, hw, unsupported_combos: list, port: int = 8080, host: str = "127.0.0.1", probe=None,
            api_key: str | None = None) -> Plan:
    """model: ggufinfo.ModelInfo (or a RefModel); hw: hw.Hardware; unsupported_combos: the (gate/up, down) types the
    engine has no GPU kernels for (asked with `strata --expert-support`); probe: hw.Probe or None; api_key: the key
    to propose for a server others can reach (None: a new random one)."""
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

    # RAM for the experts: all of them in the pinned arena, or - when they do not fit - the RAM-budget mode, which
    # keeps the most-used ones in RAM and reads the rest from the GGUF on the SSD while it answers (one GPU only)
    arena_gib = model.expert_bytes / GIB
    ple_gib = model.ple_bytes / GIB
    budget = 0
    if hw.ram_gib < arena_gib + RAM_HEADROOM_GIB:
        if hw.ram_gib < BUDGET_MIN_RAM_GIB:
            problems.append(f"{hw.ram_gib:.0f} GiB RAM: the model's experts take {arena_gib:.0f} GiB and even the "
                            f"RAM-budget mode needs {BUDGET_MIN_RAM_GIB:.0f} GiB - use a smaller quantization or add RAM")
        else:
            budget = int(hw.ram_gib - RAM_HEADROOM_GIB)
            usable = usable[:1]
            notes.append(f"the experts ({arena_gib:.0f} GiB) do not fit in RAM: {budget} GiB of them stay in RAM, the "
                         f"rest is read from the model file for every token - several times slower; it needs an NVMe "
                         f"SSD (a smaller quantization or more RAM avoids it)")
    arena_used = budget if budget else arena_gib

    n_gpu = len(usable)
    usable_vram = [g.vram_gib for g in usable]
    choose("gpu", [g.index for g in usable],
           f"{n_gpu} usable card(s)" + (", the layers split across them" if n_gpu > 1 else
                                         " (the RAM-budget mode runs on one)" if budget else ""))
    choose("layer_split", "auto" if n_gpu > 1 else "", "split by free VRAM" if n_gpu > 1 else "one card, no split")
    if n_gpu > 1 and arena_gib <= usable_vram[0] - DENSE_GIB_PER_GPU:
        choose("split_skip_if_fits", True, f"the first card ({usable_vram[0]:.0f} GiB) may hold every expert "
                                           f"({arena_gib:.0f} GiB): the engine then runs on it alone (faster)")
    else:
        choose("split_skip_if_fits", False, "the first card cannot hold every expert" if n_gpu > 1 else "one card")
    batch2 = n_gpu >= 2 and min(usable_vram) >= BATCH2_MIN_VRAM_GIB and not budget

    # Context: the longest that keeps the KV cache to KV_SHARE of every card's free VRAM.
    ctx = CONTEXTS[-1]
    if usable_vram:
        smallest_free = min(usable_vram) - DENSE_GIB_PER_GPU
        for c in CONTEXTS:
            if c > model.context_length:
                continue
            if kv_gib(c) / n_gpu <= KV_SHARE * smallest_free:
                ctx = c
                break

    rest = hw.ram_gib - arena_used - RAM_HEADROOM_GIB
    ple_ram = not budget and rest >= ple_gib + 8
    if ple_ram:
        rest -= ple_gib

    # KV streaming: the whole KV cache in RAM, 32K cells per QSA layer in VRAM; the VRAM it frees holds experts, and
    # the context is no longer bound by VRAM.  Not with Batch-2 (that combination is untested).
    longest = max((c for c in CONTEXTS if c <= model.context_length), default=CONTEXTS[-1])
    kv_stream = (not batch2 and usable_vram and longest >= KV_STREAM_MIN_CONTEXT and rest >= kv_gib(longest) + 2)
    if kv_stream:
        ctx = longest
        rest -= kv_gib(ctx)
    choose("max_context", ctx,
           f"KV cache {kv_gib(ctx):.1f} GiB (int8) in RAM, its read part in VRAM" if kv_stream else
           f"KV cache {kv_gib(ctx):.1f} GiB (int8) leaves most of the VRAM to the expert cache")
    choose("kv", "int8", "int8 codes: half of fp16's memory, measured lossless enough for code")
    if kv_stream:
        choose("kv_resident", KV_RESIDENT_CELLS, f"KV streaming: {KV_RESIDENT_CELLS} cells per layer in VRAM, the rest "
                                                 f"in RAM - the freed VRAM caches more experts")
    else:
        why = ("Batch-2 keeps the KV cache in VRAM (the two are not combined)" if batch2 and ctx >= KV_STREAM_MIN_CONTEXT
               else "the KV cache stays in VRAM" + ("" if ctx >= KV_STREAM_MIN_CONTEXT else f" (below {KV_STREAM_MIN_CONTEXT // 1024}K)"))
        choose("kv_resident", 0, why)
    choose("expert_cache", "auto", "the engine fills the VRAM left after the dense weights and the KV cache")
    choose("resident_budget_gib", budget,
           f"RAM-budget mode: {budget} GiB of experts in RAM, the rest from the SSD" if budget else
           "every expert in RAM")
    if ple_ram:
        choose("ple_io", "ram", f"the n-gram table ({ple_gib:.0f} GiB, {model.ple_type}) fits the RAM: no SSD reads")
    else:
        choose("ple_io", "", f"the n-gram table ({ple_gib:.0f} GiB) stays on the SSD: not enough RAM beside the experts")

    # The share of expert-cache misses the GPU fetches over PCIe instead of the CPU computing them
    weakest_link = min((probe.pcie_gbps.get(g.index, -1.0) for g in usable), default=-1.0) if probe else -1.0
    if probe and 0 <= weakest_link < SLOW_LINK_GBPS:
        choose("pcie_frac", 0, f"a {weakest_link:.1f} GB/s link (measured): a miss is computed faster than it crosses")
    elif probe and probe.ram_gbps >= STRONG_RAM_GBPS:
        choose("pcie_frac", 0, f"RAM read at {probe.ram_gbps:.0f} GB/s (measured): the CPU computes misses faster than "
                               f"PCIe fetches them")
    elif hw.cpu.threads >= SERVER_CPU_THREADS and not probe:
        choose("pcie_frac", 0, f"{hw.cpu.threads} CPU threads compute expert-cache misses faster than PCIe fetches them")
    else:
        choose("pcie_frac", "", "engine default: it probes the link at start and fetches part of the misses over PCIe")

    park_mib = int(max(0.0, rest * 0.5) * 1024) // 1024 * 1024
    park_mib = min(park_mib, 65536)
    if park_mib >= 4096 and not budget:
        slots = 4 if park_mib < 16384 else 8 if park_mib < 32768 else 20
        choose("conversation_cache_mib", park_mib,
               "half of the free RAM parks conversations, so agents taking turns skip re-reading their context")
        choose("conversation_cache_slots", slots, "parked conversations at most")
    else:
        choose("conversation_cache_mib", 0, "the RAM-budget mode needs the RAM for experts" if budget else
               "too little free RAM to park conversations")
        choose("conversation_cache_slots", 4, "unused while parking is off")

    if batch2:
        choose("batch2_cells", BATCH2_CELLS, "two requests decode together (~1.1 GiB VRAM; 4 parallel: ~97-99 tok/s "
                                             "vs ~82 serial on 2x RTX 3090)")
    else:
        choose("batch2_cells", 0, "requests run one after the other (Batch-2 needs two cards with 20+ GiB)")

    choose("spec", 4, "MTP speculative decoding, 4-token windows")
    choose("spec_min_p", 0.5, "adapts the window to how sure the draft is")
    choose("prefill", "auto", "batched prompt reading")
    choose("vram_reserve_mib", 700, "kept free on every card for the driver and other programs")
    choose("port", port, "API and chat page")
    local = host in LOCAL_HOSTS
    choose("host", host, "reachable from this machine only" if local else "reachable from the network")
    if local:
        choose("api_key", "", "not needed on 127.0.0.1")
    else:
        choose("api_key", api_key if api_key is not None else secrets.token_urlsafe(24),
               "others can reach the server: every /v1 request needs this key (Bearer or x-api-key)")

    # what the system itself has to allow
    if hw.thp == "never":
        notes.append("transparent huge pages are off: the expert arena runs on 4 KB pages (slower start, ~5% slower "
                     "CPU experts) - echo madvise > /sys/kernel/mm/transparent_hugepage/enabled")
    if 0 <= hw.memlock_gib < arena_used:
        notes.append(f"this shell may lock only {hw.memlock_gib:.1f} GiB of RAM: the run script needs `ulimit -l "
                     f"unlimited` (the --service unit sets it) for the pinned experts" +
                     (" and the n-gram table" if ple_ram else ""))
    for g in usable:
        if g.pcie_width_now and g.pcie_width and g.pcie_width_now < g.pcie_width:
            notes.append(f"GPU {g.index} runs at PCIe x{g.pcie_width_now} (x{g.pcie_width} possible): a wider slot "
                         f"speeds up prompts")
    return Plan(settings=s, reasons=reasons, problems=problems, notes=notes)
