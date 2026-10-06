"""What the installer needs to know about a GGUF model: is it a model this engine runs, how large are its parts, and
which shard holds what.  Reads the headers only (tools/gguf_reader.py), never tensor data.
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_reader import GGML_TYPES, GGUFFile  # noqa: E402

ARCH = "qwen4exp"  # Qwen3.8-Flash-Next, the one model family Strata runs
TYPE_IDS = {name: tid for tid, name in GGML_TYPES.items()}
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")


@dataclasses.dataclass
class ModelInfo:
    name: str
    shards: list            # every shard, in order
    head_shard: Path        # holds output.weight
    ple_shard: Path         # holds per_layer_token_embd.weight
    n_layers: int
    n_embd: int
    n_ff: int
    context_length: int
    expert_combos: dict     # (gate/up type, down type) -> layers using it
    expert_bytes: int       # all routed experts: the engine's RAM arena
    ple_bytes: int
    ple_type: str
    total_bytes: int


class ModelError(Exception):
    """The file is not a model this installer can prepare; the message says why."""


def find_shards(path: Path) -> list[Path]:
    """All shards of the split model `path` belongs to (any of them may be given), or [path] for a single file."""
    path = path.resolve()
    m = SHARD_RE.match(path.name)
    if not m:
        return [path]
    count = int(m.group("count"))
    shards = [path.with_name(f"{m.group('stem')}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [s.name for s in shards if not s.is_file()]
    if missing:
        raise ModelError(f"the split model is incomplete, missing: {', '.join(missing)}")
    return shards


def analyze(path: Path) -> ModelInfo:
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"{path}: no such file")
    shards = find_shards(path)
    files = []
    for s in shards:
        try:
            files.append(GGUFFile(s))
        except (ValueError, KeyError, OSError) as exc:
            raise ModelError(f"{s.name}: not a readable GGUF v3 file ({exc})") from None
    meta = files[0].metadata
    arch = meta.get("general.architecture", "")
    if arch != ARCH:
        raise ModelError(f"architecture {arch or 'unknown'!r}: Strata runs Qwen3.8-Flash-Next ({ARCH!r}) only")

    def key(k: str, default=0):
        return meta.get(f"{ARCH}.{k}", default)

    n_layers, n_embd, n_ff = int(key("block_count")), int(key("embedding_length")), int(key("expert_feed_forward_length"))
    if not (n_layers and n_embd and n_ff):
        raise ModelError("the model's metadata lacks block_count / embedding_length / expert_feed_forward_length")

    tensors = {t.name: (t, f.path) for f in files for t in f.tensors}
    head = tensors.get("output.weight")
    ple = tensors.get("per_layer_token_embd.weight")
    if head is None or ple is None:
        raise ModelError("the model has no output.weight or per_layer_token_embd.weight")

    combos: dict = {}
    expert_bytes = 0
    for layer in range(n_layers):
        parts = {}
        for kind in ("gate", "up", "down"):
            hit = tensors.get(f"blk.{layer}.ffn_{kind}_exps.weight")
            if hit is None:
                raise ModelError(f"layer {layer} has no ffn_{kind}_exps tensor")
            parts[kind] = hit[0]
            expert_bytes += _bytes(hit[0])
        if parts["gate"].type_name != parts["up"].type_name:
            raise ModelError(f"layer {layer}: gate experts {parts['gate'].type_name} but up experts "
                             f"{parts['up'].type_name}; the engine needs one type for both")
        combo = (parts["gate"].type_name, parts["down"].type_name)
        combos.setdefault(combo, []).append(layer)

    total = sum(s.stat().st_size for s in shards)
    name = SHARD_RE.match(shards[0].name).group("stem") if SHARD_RE.match(shards[0].name) else shards[0].stem
    return ModelInfo(name=name, shards=shards, head_shard=head[1], ple_shard=ple[1], n_layers=n_layers,
                     n_embd=n_embd, n_ff=n_ff, context_length=int(key("context_length", 262144)),
                     expert_combos=combos, expert_bytes=expert_bytes, ple_bytes=_bytes(ple[0]),
                     ple_type=ple[0].type_name, total_bytes=total)


def _bytes(t) -> int:
    n = t.expected_bytes()
    if n is None:
        raise ModelError(f"{t.name}: tensor type {t.type_name} is not one this installer knows")
    return n


def type_id(name: str) -> int:
    return TYPE_IDS[name]
