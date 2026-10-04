"""Tests for the installer's pure parts: hardware parsing, the GGUF check, the proposal, --set and the engine
config.  Standard library only, like the installer (it runs before any venv exists):

    python3 -m unittest discover -s tools/install -p 'test_*.py'
"""
from __future__ import annotations

import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggufinfo  # noqa: E402
import hw  # noqa: E402
import plan  # noqa: E402
import steps  # noqa: E402
import strata_install  # noqa: E402

GIB = 1024 ** 3


# ---------------------------------------------------------------------------------------------------- GGUF headers
def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _s(key) + struct.pack("<I", 8) + _s(value)
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path: Path, meta: dict, tensors: list) -> None:
    """A GGUF v3 header (no tensor data - the installer reads headers only). tensors: (name, shape, type name)."""
    body = b"".join(_kv(k, v) for k, v in meta.items())
    for name, shape, type_name in tensors:
        body += _s(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        body += struct.pack("<IQ", ggufinfo.type_id(type_name), 0)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta)) + body)


def model_files(folder: Path, arch="qwen4exp", layers=2, down="Q5_1", gate_up="Q4_K", split=True) -> Path:
    meta = {"general.architecture": arch, f"{arch}.block_count": layers, f"{arch}.embedding_length": 2560,
            f"{arch}.expert_feed_forward_length": 640, f"{arch}.context_length": 262144}
    head = [("output.weight", [2560, 248320], "Q8_0"), ("per_layer_token_embd.weight", [160, 320001536], "IQ4_NL")]
    experts = []
    for i in range(layers):
        experts += [(f"blk.{i}.ffn_gate_exps.weight", [2560, 640, 512], gate_up),
                    (f"blk.{i}.ffn_up_exps.weight", [2560, 640, 512], gate_up),
                    (f"blk.{i}.ffn_down_exps.weight", [640, 2560, 512], down)]
    if not split:
        p = folder / "M-Q4.gguf"
        write_gguf(p, meta, head + experts)
        return p
    write_gguf(folder / "M-Q4-00001-of-00003.gguf", meta, [])
    write_gguf(folder / "M-Q4-00002-of-00003.gguf", {}, head)
    write_gguf(folder / "M-Q4-00003-of-00003.gguf", {}, experts)
    return folder / "M-Q4-00003-of-00003.gguf"   # any shard may be given


class GGUFInfoTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_split_model(self):
        m = ggufinfo.analyze(model_files(self.dir))
        self.assertEqual(m.name, "M-Q4")
        self.assertEqual([s.name for s in m.shards], [f"M-Q4-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        self.assertEqual((m.head_shard.name, m.ple_shard.name), ("M-Q4-00002-of-00003.gguf",) * 2)
        self.assertEqual((m.n_layers, m.n_embd, m.n_ff), (2, 2560, 640))
        self.assertEqual(m.expert_combos, {("Q4_K", "Q5_1"): [0, 1]})
        per_layer = 2 * 2560 * 640 * 512 // 256 * 144 + 640 * 2560 * 512 // 32 * 24
        self.assertEqual(m.expert_bytes, 2 * per_layer)
        self.assertEqual(m.ple_type, "IQ4_NL")

    def test_single_file(self):
        m = ggufinfo.analyze(model_files(self.dir, split=False))
        self.assertEqual(len(m.shards), 1)
        self.assertEqual(m.name, "M-Q4")

    def test_refusals(self):
        with self.assertRaisesRegex(ggufinfo.ModelError, "Qwen3.8-Flash-Next"):
            ggufinfo.analyze(model_files(self.dir, arch="llama"))
        missing = Path(tempfile.mkdtemp())
        model_files(missing).unlink()  # shard 3 of 3 gone
        with self.assertRaisesRegex(ggufinfo.ModelError, "incomplete"):
            ggufinfo.analyze(missing / "M-Q4-00001-of-00003.gguf")
        bad = self.dir / "x.gguf"
        bad.write_bytes(b"not a gguf at all, just text")
        with self.assertRaisesRegex(ggufinfo.ModelError, "not a readable GGUF"):
            ggufinfo.analyze(bad)
        with self.assertRaisesRegex(ggufinfo.ModelError, "no such file"):
            ggufinfo.analyze(self.dir / "nope.gguf")


# ---------------------------------------------------------------------------------------------------- hardware
class HardwareTest(unittest.TestCase):
    def test_nvidia_smi(self):
        gpus, driver = hw.parse_nvidia_smi("0, NVIDIA GeForce RTX 3090, 24576, 8.6, 4, 16, 580.126.09\n"
                                           "1, NVIDIA GeForce RTX 3090, 24576, 8.6, [N/A], [N/A], 580.126.09\n"
                                           "garbage line\n")
        self.assertEqual(driver, "580.126.09")
        self.assertEqual([(g.index, g.sm, g.vram_mib) for g in gpus], [(0, 86, 24576), (1, 86, 24576)])
        self.assertEqual((gpus[1].pcie_gen, gpus[1].pcie_width), (0, 0))

    def test_meminfo_and_cpuinfo(self):
        self.assertAlmostEqual(hw.parse_meminfo("MemTotal:       471859200 kB\nMemFree: 1 kB\n"), 450.0)
        text = "".join(f"processor\t: {i}\nmodel name\t: AMD EPYC 7413\nphysical id\t: 0\ncpu cores\t: 24\n"
                       f"flags\t\t: fpu avx2 fma f16c avx512f avx512bw avx512vl\n\n" for i in range(48))
        cpu = hw.parse_cpuinfo(text)
        self.assertEqual((cpu.model, cpu.cores, cpu.threads), ("AMD EPYC 7413", 24, 48))
        self.assertTrue(cpu.avx2_ok and cpu.avx512)
        self.assertFalse(hw.parse_cpuinfo("processor : 0\nflags : sse4_2\n").avx2_ok)


class SystemFactsTest(unittest.TestCase):
    def test_thp_probe_and_current_link_width(self):
        self.assertEqual(hw.parse_thp("always [madvise] never\n"), "madvise")
        self.assertEqual(hw.parse_thp("[never]"), "never")
        p = hw.parse_probe("pcie 0 25.3\npcie 1 -1.0\np2p 0 1 0\np2p 1 0 1\nram_read 104.6 48 1\nnoise\n")
        self.assertEqual((p.pcie_gbps, p.p2p, p.ram_gbps, p.ram_threads), ({0: 25.3, 1: -1.0}, {(1, 0)}, 104.6, 48))
        gpus, driver = hw.parse_nvidia_smi("0, NVIDIA GeForce RTX 3090, 24576, 8.6, 4, 16, 580.82, 8\n")
        self.assertEqual((gpus[0].pcie_width, gpus[0].pcie_width_now, driver), (16, 8, "580.82"))
        self.assertGreater(hw.free_gib(Path("/nonexistent/sub/dir")), 0)


# ---------------------------------------------------------------------------------------------------- proposal
def machine(gpus=2, vram=24576, ram=450, threads=48, sm="8.6"):
    cards = [hw.GPU(i, "RTX", vram, sm, 4, 16) for i in range(gpus)]
    cpu = hw.CPU("CPU", threads // 2, threads, {"avx2", "fma", "f16c"})
    return hw.Hardware(cards, float(ram), cpu, "580")


def fake_model(expert_gib=100.0, ple_gib=28.0, combos=None):
    return ggufinfo.ModelInfo(name="M", shards=[Path("/m/M-00001-of-00002.gguf")], head_shard=Path("/m/h.gguf"),
                              ple_shard=Path("/m/p.gguf"), n_layers=48, n_embd=2560, n_ff=640,
                              context_length=262144, expert_combos=combos or {("Q4_K", "Q5_1"): list(range(48))},
                              expert_bytes=int(expert_gib * GIB), ple_bytes=int(ple_gib * GIB), ple_type="IQ4_NL",
                              total_bytes=int((expert_gib + ple_gib + 5) * GIB))


class PlanTest(unittest.TestCase):
    def test_reproduces_the_measured_dual_3090_setup(self):
        p = plan.propose(fake_model(), machine(), [])
        s = p.settings
        self.assertEqual(p.problems, [])
        self.assertEqual((s["gpu"], s["layer_split"], s["max_context"]), ([0, 1], "auto", 262000))
        self.assertEqual((s["ple_io"], s["pcie_frac"], s["batch2_cells"]), ("ram", 0, 65536))
        self.assertEqual((s["conversation_cache_mib"], s["conversation_cache_slots"]), (65536, 20))
        self.assertEqual([r[0] for r in p.reasons], list(s))  # every setting has its reason

    def test_single_12gb_card_on_a_desktop(self):
        s = plan.propose(fake_model(), machine(gpus=1, vram=12288, ram=192, threads=16), []).settings
        self.assertEqual((s["gpu"], s["layer_split"], s["batch2_cells"]), ([0], "", 0))
        self.assertEqual(s["pcie_frac"], "")
        self.assertEqual(s["ple_io"], "ram")        # 192 - 100 - 24 = 68 GiB left, the table takes 28
        # 40 GiB left: KV streaming puts the whole 262K KV cache (3.4 GiB) in RAM, so VRAM no longer bounds it
        self.assertEqual((s["max_context"], s["kv_resident"]), (262000, plan.KV_RESIDENT_CELLS))
        # no RAM to spare for the KV cache: VRAM bounds the context as before
        tight = plan.propose(fake_model(ple_gib=1), machine(gpus=1, vram=12288, ram=126, threads=16), []).settings
        self.assertEqual((tight["max_context"], tight["kv_resident"]), (131072, 0))  # 7 GiB free: 1.68 GiB < 1/4
        small = plan.propose(fake_model(ple_gib=1), machine(gpus=1, vram=11264, ram=126, threads=16), []).settings
        self.assertEqual(small["max_context"], 65536)  # 6 GiB free: 128K's 1.68 GiB is over a quarter

    def test_batch2_keeps_the_kv_cache_in_vram(self):
        p = plan.propose(fake_model(), machine(), [])
        self.assertEqual((p.settings["batch2_cells"], p.settings["kv_resident"]), (65536, 0))
        self.assertIn("Batch-2", dict((k, w) for k, _, w in p.reasons)["kv_resident"])

    def test_experts_past_the_ram_take_the_budget_mode_on_one_card(self):
        p = plan.propose(fake_model(expert_gib=100), machine(ram=96), [])
        s = p.settings
        self.assertEqual(p.problems, [])
        self.assertEqual((s["resident_budget_gib"], s["gpu"], s["layer_split"]), (72, [0], ""))
        self.assertEqual((s["batch2_cells"], s["conversation_cache_mib"], s["ple_io"]), (0, 0, ""))
        self.assertTrue(any("do not fit in RAM" in n for n in p.notes))
        self.assertTrue(any("RAM" in x for x in plan.propose(fake_model(), machine(ram=40), []).problems))

    def test_skip_the_split_when_the_first_card_holds_every_expert(self):
        self.assertTrue(plan.propose(fake_model(expert_gib=15), machine(), []).settings["split_skip_if_fits"])
        self.assertFalse(plan.propose(fake_model(), machine(), []).settings["split_skip_if_fits"])
        self.assertFalse(plan.propose(fake_model(expert_gib=5), machine(gpus=1), []).settings["split_skip_if_fits"])

    def test_an_api_key_once_others_can_reach_it(self):
        self.assertEqual(plan.propose(fake_model(), machine(), []).settings["api_key"], "")
        s = plan.propose(fake_model(), machine(), [], host="0.0.0.0", api_key="k1").settings
        self.assertEqual((s["host"], s["api_key"]), ("0.0.0.0", "k1"))
        self.assertGreaterEqual(len(plan.propose(fake_model(), machine(), [], host="0.0.0.0").settings["api_key"]), 24)

    def test_measured_bandwidth_sets_the_pcie_share(self):
        desk = machine(gpus=1, vram=12288, ram=192, threads=16)
        fast_ram = hw.Probe({0: 25.0}, set(), 100.0, 16)
        slow_link = hw.Probe({0: 1.5}, set(), 40.0, 16)
        normal = hw.Probe({0: 25.0}, set(), 45.0, 16)
        self.assertEqual(plan.propose(fake_model(), desk, [], probe=fast_ram).settings["pcie_frac"], 0)
        self.assertEqual(plan.propose(fake_model(), desk, [], probe=slow_link).settings["pcie_frac"], 0)
        self.assertEqual(plan.propose(fake_model(), desk, [], probe=normal).settings["pcie_frac"], "")
        # measured slow RAM overrides the thread count's guess
        self.assertEqual(plan.propose(fake_model(), machine(), [], probe=hw.Probe({0: 25, 1: 25}, set(), 45.0, 48))
                         .settings["pcie_frac"], "")

    def test_system_notes(self):
        h = machine()
        h.thp, h.memlock_gib = "never", 8.0
        h.gpus[1].pcie_width_now = 8
        notes = " | ".join(plan.propose(fake_model(), h, []).notes)
        self.assertIn("transparent huge pages are off", notes)
        self.assertIn("ulimit -l unlimited", notes)
        self.assertIn("GPU 1 runs at PCIe x8", notes)
        self.assertEqual(plan.propose(fake_model(), machine(), []).notes, [])

    def test_every_reference_model_has_a_plan(self):
        for ref in plan.REFERENCE_MODELS:
            p = plan.propose(ref, machine(), [])
            self.assertEqual(p.problems, [], ref.name)
            self.assertEqual([r[0] for r in p.reasons], list(p.settings), ref.name)

    def test_short_of_ram_or_gpu_or_kernels_blocks(self):
        p = plan.propose(fake_model(expert_gib=100), machine(ram=40), [])
        self.assertTrue(any("RAM" in x for x in p.problems))
        p = plan.propose(fake_model(), machine(sm="6.1"), [])
        self.assertTrue(any("no usable NVIDIA GPU" in x for x in p.problems))
        self.assertEqual(len(p.notes), 2)
        p = plan.propose(fake_model(), machine(), [("Q4_K", "Q5_1")])
        self.assertTrue(any("no GPU kernels" in x and "48 layer" in x for x in p.problems))

    def test_little_free_ram_turns_off_parking_and_ple_in_ram(self):
        s = plan.propose(fake_model(expert_gib=100, ple_gib=28), machine(ram=130), []).settings
        self.assertEqual((s["ple_io"], s["conversation_cache_mib"]), ("", 0))


# ---------------------------------------------------------------------------------------------------- CLI + config
class CliTest(unittest.TestCase):
    def test_overrides_take_the_proposed_types(self):
        s = plan.propose(fake_model(), machine(), []).settings
        out = strata_install.apply_overrides(s, ["max_context=131072", "gpu=1", "pcie_frac=", "ple_io=mmap",
                                                 "spec_min_p=0.6"])
        self.assertEqual((out["max_context"], out["gpu"], out["pcie_frac"], out["ple_io"], out["spec_min_p"]),
                         (131072, [1], "", "mmap", 0.6))
        with self.assertRaisesRegex(ValueError, "unknown setting"):
            strata_install.apply_overrides(s, ["nope=1"])
        with self.assertRaisesRegex(ValueError, "not a valid value"):
            strata_install.apply_overrides(s, ["max_context=lots"])

    def test_argument_checks(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            strata_install.parse_args(["--check"])            # needs --gguf
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            strata_install.parse_args(["--cuda-arch", "sm86"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            strata_install.parse_args(["--gguf", "m.gguf", "--name", "a/b"])
        a = strata_install.parse_args(["--gguf", "m.gguf", "--set", "kv=int8", "--yes", "--cuda-arch", "86,89"])
        self.assertEqual((a.overrides, a.yes, a.cuda_arch), (["kv=int8"], True, "86,89"))
        self.assertTrue(strata_install.parse_args(["--system-check"]).system_check)
        self.assertEqual(strata_install.parse_args(["--system-check", "--set", "max_context=65536"]).overrides,
                         ["max_context=65536"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            strata_install.parse_args(["--system-check", "--check", "--gguf", "m.gguf"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            strata_install.parse_args(["--set", "kv=int8"])                       # needs --gguf or --system-check

    def test_bool_overrides(self):
        s = plan.propose(fake_model(), machine(), []).settings
        self.assertTrue(strata_install.apply_overrides(s, ["split_skip_if_fits=yes"])["split_skip_if_fits"])
        with self.assertRaisesRegex(ValueError, "not a valid value"):
            strata_install.apply_overrides(s, ["split_skip_if_fits=maybe"])

    def test_system_check_without_a_model_prints_every_reference(self):
        a = strata_install.parse_args(["--system-check"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            strata_install.show_reference(machine(), None, a)
        text = out.getvalue()
        for ref in plan.REFERENCE_MODELS:
            self.assertIn(ref.name, text)
        self.assertIn("pcie_frac", text)

    def test_engine_config_has_everything_and_no_environment(self):
        m = fake_model()
        s = plan.propose(m, machine(), []).settings
        cfg = steps.engine_config(m, s, Path("/m/strata-pack"), Path("/m/strata-mtp/rt"), "M", Path("/l/x.log"))
        args = cfg["args"]
        for flag, value in (("--native", "/m/M-00001-of-00002.gguf"), ("--native-head-gguf", "/m/h.gguf"),
                            ("--ple-gguf", "/m/p.gguf"), ("--mtp", "/m/strata-mtp/rt"), ("--max-context", "262000"),
                            ("--pcie-frac", "0"), ("--ple-io", "ram"), ("--batch2-cells", "65536"),
                            ("--conversation-cache-mib", "65536")):
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.assertEqual((cfg["gpu"], cfg["layer_split"], cfg["host"]), ([0, 1], "auto", "127.0.0.1"))
        self.assertNotIn("env", cfg)
        s1 = plan.propose(m, machine(gpus=1, vram=12288, ram=130, threads=8), []).settings
        args1 = steps.engine_config(m, s1, Path("/p"), Path("/r"), "M", Path("/l"))["args"]
        for flag in ("--pcie-frac", "--ple-io", "--batch2-cells", "--conversation-cache-mib", "--resident-budget-gib"):
            self.assertNotIn(flag, args1)
        self.assertNotIn("api_key", cfg)
        self.assertNotIn("split_skip_if_fits", cfg)

    def test_engine_config_carries_the_opt_in_features(self):
        m = fake_model()
        kv = plan.propose(m, machine(gpus=1, vram=12288, ram=192, threads=16), []).settings
        args = steps.engine_config(m, kv, Path("/p"), Path("/r"), "M", Path("/l"))["args"]
        self.assertEqual(args[args.index("--kv-resident") + 1], str(plan.KV_RESIDENT_CELLS))
        budget = plan.propose(m, machine(ram=96), []).settings
        args = steps.engine_config(m, budget, Path("/p"), Path("/r"), "M", Path("/l"))["args"]
        self.assertEqual(args[args.index("--resident-budget-gib") + 1], "72")
        small = fake_model(expert_gib=15)
        open_s = plan.propose(small, machine(), [], host="0.0.0.0", api_key="k1").settings
        cfg = steps.engine_config(small, open_s, Path("/p"), Path("/r"), "M", Path("/l"))
        self.assertEqual((cfg["split_skip_if_fits"], cfg["api_key"]), (True, "k1"))


if __name__ == "__main__":
    unittest.main()
