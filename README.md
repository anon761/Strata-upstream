# arcfork-strata

> **A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata)** (MIT), merged with upstream `v0.1.32`.
> This fork is not affiliated with or endorsed by the Strata authors. Upstream's README is kept verbatim at the
> bottom; the engine, its design and its documentation are upstream's work.

## What this is

`arcfork-strata` is an internal fork of the **Strata** inference engine. Strata upstream is a specialised,
high-performance engine for *one* model family - **Qwen3.8-Flash-Next** - originally through ISTA-DASLab's
**GSQ-RCO** i-quant packs, and since v0.1.31 also through ordinary GGUFs such as unsloth's `UD-Q4_K_XL`.

The fork runs those ordinary GGUFs - unsloth's `UD-Q4_K_XL` and finetunes such as **Swift-1.5 Q4_K_L** - on a
**2x RTX 3090** box with a **262K context** for coding agents, and adds what that setup needs on top of upstream.
It is driven from `mayaservices` as the GGUF backend, next to stock vLLM for safetensors models.

## What the fork adds on top of upstream

- **Conversation parking with a layer split.** Upstream's conversation cache (`--conversation-cache-mib`) parks
  whole conversations in RAM but refused `--layer-split`. Here every stage parks its own part (running state,
  checkpoint states, K/V of its layers; the draft layer's with the last stage) and restores it on its own GPU,
  so clients that take turns (an agent, the web chat, the Matrix bot) no longer re-read each other's context.
  Validated with upstream's `tools/conversation_cache_parity.py` (A/B/A byte-exact state, pressure fallback).
- **The idle GPU helps short prompts.** A prompt that fits one chunk runs the split's stages one after the
  other; the idle stage's GPU streams and computes part of the active stage's experts over its own PCIe link
  (deterministic, `STRATA_PREFILL_HELP=0` turns it off): 2K-token prompts +28-34%.
- **Follow-ups resume their session.** A verify window commits only the tokens it hands out (an answer that
  ended inside an accepted draft left the session ahead of the client), and the server keeps the recent requests'
  own token ids, since a model's sampled tokens are not always the tokenizer's split of the same text.
- **Q5_0 down experts on the GPU** (Swift-1.5 Q4_K_L, OrcaRouter Q4_K_S) and a **Q8_0 PLE table** (unsloth,
  Swift-1.5).
- The expert arena on **transparent huge pages** (Linux, THP `madvise`).
- Tools: `iq_pack.py` prints `IQPACK_PROGRESS` lines (mayaservices' progress bar); `mtp_fetch.py --repo` fetches
  the MTP head from a finetune's own BF16 checkpoint.
- `setup.py` prefers the pip-installed `cmake`/`ninja` (a distro cmake is too old for the CUDA 20 dialect).

Earlier fork work that upstream now has in its own form (and that the fork follows): ordinary GGUF experts
(Q4_K/Q5_K/Q5_1/Q8_0) with MMQ, per-role expert shards, `--ple-io ram`, the prompt loans of a layer split.

## Measured results

2x **RTX 3090** (24 GB each), AMD EPYC 7413, **450 GB** RAM, driver 580, CUDA 13.3, layer split across both cards,
**262K context**, MTP speculative decoding on (`--spec 4`), `--pcie-frac 0 --ple-io ram`, conversation parking on.
Single request, OpenAI `/v1/chat/completions`, a coding workload: prompts of real C++/CUDA source, a code-writing
task for decode (768 tokens, greedy), and a follow-up turn on an 8K conversation (time to first token, the prefix
reused). Decode excludes time-to-first-token and includes the accepted drafts, so it varies with the text.

| Engine, model | Prefill 2K | 8K | 32K | Decode tok/s | Follow-up turn |
| --- | ---: | ---: | ---: | ---: | ---: |
| **this fork, unsloth `UD-Q4_K_XL`** (103.7 GiB) | **~1,010** | **~1,860** | **~3,260** | **~116** | **~0.17 s** |
| **this fork, `Swift-1.5 Q4_K_L`** | ~1,020 | ~1,830 | ~3,210 | ~117 | ~0.22 s |
| FreeToken, `Swift-1.5 NVFP4` (tensor parallel over both cards) | ~1,280 | ~1,450 | ~1,630 | ~104 | ~1.1 s |
| the fork on v0.1.28 before this work, `UD-Q4_K_XL` | ~650 | ~1,170 | ~1,370 | ~71 | ~0.4 s |

Two 8K coding conversations taking turns (A, B, A, B, ...): a follow-up's time to first token **~4.6 s without
parking, ~0.63 s with it** (the other conversation's context is restored instead of read again). Needle-in-a-
haystack recall 12/12 at 2K/32K/128K/250K.

`--pcie-frac 0` (an expert-cache miss is computed by the CPU pool instead of fetched over PCIe) suits a CPU with
many cores and memory channels (here 24 cores, ~105 GB/s); on a desktop CPU keep the default.

## Installing

On a bare **Debian 12 or 13** (amd64) the installer sets up everything - system packages, the NVIDIA driver and
CUDA toolkit 13.3 from NVIDIA's Debian repository, the Python environment, and the engine built from this fork for
the GPUs it finds. No environment variables are needed.

```bash
git clone <this fork> && cd arcfork-strata
sudo ./install.sh                                    # install; reboots once if it had to install the driver
sudo ./install.sh --gguf /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

With `--gguf` it checks the model (architecture, every layer's expert types against the engine's GPU kernels, RAM
for the expert arena), shows the settings it proposes for this hardware with a reason for each - GPUs and layer
split, context length, the n-gram table in RAM, `--pcie-frac`, conversation parking, Batch-2 - and after you
confirm builds the pack and the MTP draft head next to the model and writes `configs/<name>.json` and
`run-<name>.sh`. Features that only work once switched on are proposed where they fit: KV streaming
(`--kv-resident`, from 64K context when RAM holds the KV cache; not with Batch-2), the RAM-budget mode
(`--resident-budget-gib`, one GPU) when the experts do not fit in RAM instead of refusing the model, skipping the
split when the first card holds every expert, and an API key once `--host` is not local. It also notes what the
system has to allow: transparent huge pages off, a memlock limit below the experts, a GPU in a narrower slot.

`./install.sh --system-check` (no root, changes nothing) checks this machine - driver, transparent huge pages,
memlock, free disk, GPUs and their links, CPU, RAM - offers to measure every GPU's PCIe bandwidth, P2P and the RAM's
read bandwidth (y/n; stop a running engine first, it holds the GPUs' memory) and prints the best settings: for one
model with `--gguf`, else for the common quantizations (UD-Q4_K_XL, Swift-1.5 Q4_K_L, Q5_K_M, UD-Q6_K_XL). The
measurement decides `--pcie-frac`; the engine's `--system-probe` does it. Useful options:

| Option | |
| --- | --- |
| `--check` | only show the hardware and the proposal (no root needed, changes nothing) |
| `--system-check` | check this machine and print its best settings (with `--gguf`: for that model); `--yes` measures without asking |
| `--set KEY=VALUE` | change a proposed setting, e.g. `--set max_context=131072` (repeatable) |
| `--yes` | apply without asking |
| `--service` | also install a systemd service `strata-<name>` that starts at boot |
| `--port`, `--host`, `--name` | API port (8080), listen address (127.0.0.1), the model name the API reports |
| `--mtp-repo URL` | the BF16 checkpoint with the MTP head of a fine-tune (default: Qwen's) |
| `--cuda-arch 86,89` | build for these GPUs instead of the visible ones (a build container without GPUs) |

In a container (LXC, Docker) the driver comes from the host: pass the GPUs in, the installer installs the rest.
Running it again only does what is missing; a rebuilt engine follows a new checkout. Upstream's own one-click setup
(`setup.sh` / `START-HERE.bat`) stays for upstream's GSQ-RCO models and ready-made engines, which lack this fork's
changes.

### By hand

```bash
git clone <this fork> && cd arcfork-strata
# a source build for the ordinary GGUFs: the K-quant (and Q5_0) MMQ prompt kernels are build options, off upstream
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_ORCA_Q4KS_MMQ=ON
ninja -C build strata
```

For an **ordinary GGUF** (e.g. unsloth `UD-Q4_K_XL`), prepare a pack and the MTP head and point the server at them
(`mayaservices`' `strata-run.py` builds a missing pack and MTP head itself):

```bash
# 1) the native pack (reads every shard; --compat-bf16 dequantizes the small projections)
.venv/bin/python tools/iq_pack.py --gguf <shard1.gguf> --out /path/to/pack --compat-bf16
# 2) the MTP head (--repo: the BF16 checkpoint of a finetune, e.g. Swift-1.5's), as setup.py does
.venv/bin/python tools/mtp_fetch.py fetch --out <mtp dir> [--repo <hf resolve/main url>]
.venv/bin/python tools/mtp_pack.py --src <mtp dir> --experts q2_0 --out <mtp dir>/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf <mtp dir>/mtp-q2_0.gguf --out <mtp dir>/rt && cp data/draft_vocab.bin <mtp dir>/rt/
# 3) a run config (see serve/server.py --help) with:
#    --pack <pack> --native <shard1> --native-head-gguf <shard with output.weight> --ple-gguf <shard with PLE>
#    --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp <mtp dir>/rt
#    --max-context 262000 --kv int8 --pcie-frac 0 --ple-io ram
#    --conversation-cache-mib 16384 --conversation-cache-slots 4   and  "gpu": [0,1], "layer_split": "auto"
```

A pack written by the fork before the merge (`native_experts.txt` with space-separated per-role shards) has to be
written again with this `iq_pack.py`. `--ple-io ram` needs RAM for the table (IQ4_NL ~28 GB, Q8_0 ~54 GB; a cold
start reads it in ~1-2 min) on top of the expert arena; parking adds up to its budget.

Requirements are upstream's: an NVIDIA RTX 20+ card, a current driver, and (for a source build) a CUDA toolkit
with `nvcc` (CUDA 13.x tested). Everything else is set up by `setup.sh`.

## Known issues / limitations

- **Single GPU in an LXC** can fail `cublasCreate` when the expert cache fills VRAM (container pinning limits).
  The 2-GPU layer split is the supported configuration.
- Split parking makes a full copy on every park (upstream's retained-K/V reuse is single-GPU only); a 2K-token
  conversation parks in ~150 ms. `--split-device 0` (the one-GPU check of the hand-off) refuses parking.

## License and attribution

MIT, upstream's [`LICENSE`](LICENSE) unchanged. The engine, the kernels, the MMQ path (llama.cpp, MIT), the model
and the packs are upstream's and their authors' work - see the upstream README below and `LICENSE`.

---

<h2 align="center">Strata</h2>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But Strata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
