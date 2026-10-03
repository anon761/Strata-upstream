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
`run-<name>.sh`. Useful options:

| Option | |
| --- | --- |
| `--check` | only show the hardware and the proposal (no root needed, changes nothing) |
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

<p align="center"><b>Run a 125-billion-parameter AI model on a normal gaming PC</b><br>
one NVIDIA card (12-24 GB) + 64 GB of RAM · Windows or Linux · one click to install</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** - a large, smart AI model that
normally needs a server - on your own PC. It writes its answers at **60-95 tokens per second** (a token is about ¾
of a word): faster than you can read.

- **Free and open source.**

> **Jump to:** [How fast?](#how-fast-is-it) · [Which model?](#which-model-should-i-pick) · [Install](#install) ·
> [Using it](#using-it) · [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) ·
> [All the details](docs/DETAILS.md)

---

## How fast is it?

Measured on an RTX 5070 (12 GB), a Ryzen 5 7600 and 64 GB of RAM:

| Size | Writes answers (short chat) | Writes answers (128K context) | Reads your prompt |
| --- | ---: | ---: | ---: |
| **Q2_0** | 93 tokens/s | 74 tokens/s | 2,170 tokens/s |
| **IQ2_XS** | 79 tokens/s | 63 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 49 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 46 tokens/s | 1,620 tokens/s |
| **Coder** (IQ1_M) | 55 tokens/s | 43 tokens/s | 2,180 tokens/s |

- **Writes answers** = how fast the reply appears (tokens per second).
- **Reads your prompt** = how fast it takes in what you send (long documents, code, chat history), measured on a
  32K-token prompt; a 4K prompt reads at 910-1,580 tokens/s. A 32K prompt takes about 15 seconds with Q2_0.

A card with more VRAM is faster, because more of the model fits on the GPU: an RTX 3090 (24 GB) should do roughly
100-140 tokens per second. All measurements, long-context numbers and estimates for other cards are in the
[details](docs/DETAILS.md#speed-measured).

Every PC is different: `START-HERE.bat --calibrate` measures a few engine settings on yours and keeps the fastest
(about 5-10 minutes; on the PC above it made the Coder 7% faster).

Measured Strata on your own PC? See [Community benchmark results](docs/COMMUNITY_BENCHMARKS.md)
for a report template and how to share your results in a pull request.

**Two or three NVIDIA cards?** Just run `START-HERE.bat`: it lists your cards, says which ones Strata can use, and
asks whether to share the model across them (recommended when two can). An install made on one card asks once at
its next start. Or choose yourself: `START-HERE.bat --gpus 0,2` (both, remembered), `--gpus all`, or `--gpu 0` (one
card, this start only). Each card keeps the experts of its own layers, and prompts flow through the cards in a
pipeline: on an RTX 5080 + RTX 3090 prompts were read 18-20% faster than on the 5080 alone, decoding on par.
Every card must be an RTX 20 series or newer with 8 GB or more. See [docs/MULTI_GPU.md](docs/MULTI_GPU.md).

## Which model should I pick?

**The size** (the same model, compressed more or less):

| Model | RAM+VRAM Requirements | Speed | Quality |
| --- | ---: | --- | --- |
| **Q2_0** | 37.6 GB | fastest | good |
| **IQ2_XS** | 39.2 GB | fast | better (**recommended**) |
| **IQ3_XXS** | 47.0 GB | slower | great |
| **IQ3_S** | 54.8 GB | slowest | best: matches the full model on the published tests (original model only) |

**Will it fit?** Shard 1 is the part of the model that gets loaded when it starts: its experts go into your **RAM**,
the rest onto your graphics card (the second shard, a 29 GB lookup table, stays on the SSD). So it fits when your
**RAM is at least shard 1 + about 10 GB** for Windows and your other programs. With 64 GB of RAM every size fits
(IQ3_S with little else open); with 48 GB, Q2_0 and IQ2_XS. A bigger graphics card makes it faster, but it doesn't
lower the RAM needed.

**The version:**

- **Qwen3.8-Flash-Next** - the original.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** - ISTA-DASLab's coding
  version: half of the experts removed, keeping the ones that code, tool use and images need (91% of the full model's
  SWE-bench Verified score, 99% of LiveCodeBench, by its authors). One size (IQ1_M: its experts stored like IQ3_S):
  shard 1 is **29.6 GB**, so it fits a PC with **32 GB of RAM**, runs 262K context on 64 GB, and reads long prompts
  the fastest of all. Weaker outside coding.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** - a fine-tune by UkisAI
  that thinks much shorter before answering, so you get the answer sooner, with about the same quality. Same speed per
  token, and about the same RAM as the same size of the original (no IQ3_S). Its own license applies (see its page).

Not sure? Take **IQ2_XS** - or the **Coder** if you mainly write code, or have 32-48 GB of RAM. You can add another
one later with `SETUP.bat` (the same as `START-HERE.bat --setup`; on Linux `./setup.sh --setup`).

For **OrcaRouter's Flash-Next Uncensored IQ3_XXS**, see the [manual compatibility setup](docs/ORCA.md).
It needs an explicit packing conversion and is not an installer menu option.

**Unsloth's 4-bit UD-Q4_K_XL** (experimental) is the fourth version in setup's menu (`--family unsloth`): the closest
to the full model, but a 111 GB download whose 77 GB of experts do not fit in RAM. Strata keeps your RAM minus 24 GB
of them in RAM and reads the rest from the SSD while it answers: 7-8.5 tokens/s on a 64 GB PC with a 12 GB GPU, several
times slower than the sizes above, and long prompts are slow. It needs 48 GB of RAM or more, an NVMe SSD and one
NVIDIA GPU (no images yet). Details and measurements: [UD-Q4_K_XL](docs/UNSLOTH_Q4.md).

An **AMD Radeon RX 7900 XT / XTX, RX 9070 / 9070 XT or Radeon AI PRO R9700 on Linux** works too (experimental; the
RX 7800 XT / 7700 XT and RX 9060 XT were validated by their owners; the RX 6800 / 6900 series, gfx1030, is community-reported):
`./setup.sh --backend hip`, chosen by itself on a PC with no NVIDIA card Strata can use. It installs ROCm without sudo
and compiles the engine (no images yet; several cards with `--gpus`). Details: [AMD HIP](docs/AMD_HIP.md).

## Install

**You need:** an NVIDIA RTX 20, 30, 40 or 50 card with 12 GB of VRAM or more (RTX 20 since 0.1.27), enough RAM for the size you pick (above;
a big GPU makes up for less RAM - the [low-RAM mode](docs/DETAILS.md)),
~80 GB of free disk space (an SSD makes the first start much faster), and Windows 10/11 or Linux. The only thing you
install yourself is a current **NVIDIA driver** ([nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA
App). Everything else - Python, the engine, the model - is set up for you.

**Windows**

1. [Download this project](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Double-click **`START-HERE.bat`**.
3. Answer a few questions - or just press Enter each time for the recommended choice:
   - **Which model and size?** The original or Swift 1.5, and Q2_0, IQ2_XS, IQ3_XXS or IQ3_S - see [above](#which-model-should-i-pick)
   - **How much context?** How much text it can keep in mind at once (it suggests one for your card). 384K and
     512K (experimental) extend the model past its trained 262K by rope scaling - the setup turns it on itself (yarn and a
     covering factor; `--rope-scaling`/`--rope-scale` override) ([details](docs/DETAILS.md))
   - **Images?** Whether it should also read pictures
   - **Experimental speed projection?** Off unless you say yes - [read what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first

Then it downloads everything (the model is ~70 GB, so the first time takes a while - you can stop and it picks up
where it left off) and **starts the model**. Your browser opens the Strata app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time): Strata
> loads 35-55 GB into your RAM and locks part of it for the graphics card. That's normal - wait, and don't close the
> window. The window tells you what it is doing.

**Next time**, just double-click `START-HERE.bat` again: it starts right away, nothing is downloaded twice. Close its
window to stop the model.

**Updating:** download the new version and unzip it anywhere (or `git pull`), then run `START-HERE.bat` in it. The
model files are kept in a `Strata-data` folder next to your Strata folder, so a new copy finds them and sets itself up
the same way - nothing big is downloaded again.

**Linux:** run `./setup.sh` - same questions, same result.

**Docker (Linux):** the same idea, in a container.

1. Host: Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   and a driver **580 or newer** (CUDA 13.0).
2. Build (this compiles the engine into the image, so the container never compiles):
   `docker build -t strata .`
   `docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .` builds for one card only (faster).
   The default covers RTX 30 (86), RTX 40 (89), RTX 50 (120) and A-series (80); a card outside that
   set needs a rebuild with its own arch. Add `--build-arg BUILD_VISION=0` to skip the image encoder.
3. Run (the first start downloads the ~70 GB model, then starts; later starts go straight to serving):
   `docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data strata`

   The setup choices are env vars: `-e MODEL=IQ2_XS -e FAMILY=qwen -e CONTEXT=32768 -e VISION=no`
   (or `MODEL=Q2_0|IQ3_XXS|IQ3_S`, `FAMILY=swift|coder`; the defaults above are the recommended ones).
   `-e VISION=cpu` keeps the image encoder on the CPU. `-e KV=int8|q4_0|k8v4` picks the KV cache
   precision; `k8v4` is INT8 K with 4-bit V and keeps its KV in VRAM from 64K up.
   Only the model files, the prepared pack, the MTP layer and the install config live in the
   `strata-data` volume; the engine is part of the image. Switching between models already on the
   volume needs no setup pass: `-e MODEL=Q2_0 -e FAMILY=coder` picks that model's config. Add
   `-e REINSTALL=1` only to change settings for a model already set up (context, vision, KV, host,
   api_key, LOW_RAM), since those are recorded in its config.
   Strata loads 32-62 GB into RAM. `--gpus all` on a host with two usable cards takes both: the
   layer split is setup's recommended default ([docs/MULTI_GPU.md](docs/MULTI_GPU.md)), and a volume
   set up for one card switches to the pair on its first start there. Pin one card with `-e GPU=0`,
   or name them with `-e GPUS=0,2` and where the later card's layers start with `-e LAYER_SPLIT=18`.
   A memory limit needs `-e LOW_RAM=on`, which maps the model's experts from the pack instead of
   keeping them in RAM: setup.py measures the host's RAM, not the container's limit, so it cannot
   see a cap. LOW_RAM runs on one card.
   The server listens on `0.0.0.0:8080` by default; set `-e API_KEY=<secret>` before exposing the port
   to a network. The image has a `HEALTHCHECK` on `/health`, so `docker ps` shows the container
   healthy once the model is loaded, and `GET /v1/status` says what it is running.

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** `http://127.0.0.1:8080` - the Strata app (it opens by itself when the model starts): **Chat**, a
  live **Monitor** of the model and your GPU/CPU/RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `.venv\Scripts\python chat.py`
- **Your apps and coding agents:** add it as an "OpenAI-compatible" provider with base URL
  **`http://127.0.0.1:8080/v1`**, any API key and any model name. Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **off, low, medium or high** - in the chat page menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; high is best for hard questions.
- **Pictures:** in the chat page click **Picture**; in `chat.py` type `/image <path>`; in apps just attach them.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`, then open the
  address the server window prints; see the [details](docs/DETAILS.md#using-it).
- **Experimental speed projection (off by default):** an experimental control vector that setup can turn on; it
  changes how the model answers - read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) first.

**Good to know:** it answers one request at a time. The first message of a chat is read in full (about 1 minute per
30,000 tokens); after that it keeps the conversation and reads only what is new, so follow-ups start in seconds.

### Where things are stored

- **Your chats: only in your browser.** The Chat tab keeps the conversation, its settings and the API key you typed
  in the browser's local storage (`strata.*` keys) - not on the server and not in the Strata folder. Pictures are not
  kept, only their names. Another browser or a private window starts empty; clearing the site's data deletes them.
- **How the model starts:** `strata-<model>.json` in the Strata folder (context, GPUs, host, API key, ...), written
  by setup; next to it `run-<model>.bat` / `.sh`, the log `strata-<model>.log` and, when you use "Use for other
  apps too", `strata-<model>.shared-settings.json`.
- **The model files** (`models/`, `packs/`, `mtp/`, 70-120 GB): in **`Strata-data` next to the Strata folder**, or
  wherever `--data-dir` put them.
- **Where that data folder is:** `%APPDATA%\Strata\settings.json` on Windows, `~/.config/strata/settings.json` on
  Linux ([details](docs/DETAILS.md)).

## Something went wrong?

**My PC froze, or got very slow, the first time Strata started.**
That's normal while it starts, most of all the first time. Strata loads 35-55 GB into your RAM, locks part of it for
the graphics card, and works out how much of the model fits on your GPU. The mouse can freeze for a few minutes. **Wait, and don't close the
window.** The next starts are much faster. Still frozen after 10 minutes? Restart the PC, close other programs
(browsers use a lot of RAM) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `START-HERE.bat` again. It continues where it stopped.

**It says the NVIDIA driver is too old.**
Update it (NVIDIA App or [nvidia.com/drivers](https://www.nvidia.com/drivers)), restart the PC, and run
`START-HERE.bat` again.

**It says port 8080 is already in use.**
Strata is already running. Look for its window.

**It's very slow and the disk light keeps blinking.**
Your PC is out of free RAM. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually not enough RAM (on Linux the system then stops the engine). Just send your message again: Strata starts the
engine by itself. If it keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new chat, or run `SETUP.bat` and pick more
context.

**Still stuck?** Look in the [full troubleshooting table](docs/DETAILS.md#troubleshooting), or open an issue and
attach `strata-<model>.log` from the Strata folder.

## How does it work?

Models like this one normally run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes it fit by **sharing the work across your whole PC** - the same idea as a kitchen, where the
things you use all the time stay on the counter and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts"),** and each word it writes needs only 10 of them.
  So it doesn't have to have all of them on the graphics card at once.
- **Your graphics card** does the part of the work needed for every word, and keeps the few thousand experts that
  are asked most often. It keeps learning which ones those are while you use it.
- **Your RAM** holds every expert. When a word needs one the card doesn't have, **your processor** works on it -
  at the same time as the graphics card, so neither waits for the other.
- **Your SSD** holds a big lookup table; the model only reads a few small rows of it per word.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check.** A small, fast helper built into the model guesses the next few words, and the big model
  checks all the guesses in one go. It keeps the ones it agrees with and writes the next word itself - so one step
  often produces several words. The helper only guesses - the big model decides every word - so you get the same
  quality answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens - pieces of words - at a time), which is why a long
  document or code base is read at over 1,000 tokens per second.

Want the full picture? The [details](docs/DETAILS.md#how-it-works) explain every part and its numbers, and the
[paper](docs/paper/Strata-Paper.pdf) tells the whole story, with the measurements behind it.

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI; the experimental
  [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) by Unsloth (its support follows
  [eddoursul/Strata](https://github.com/eddoursul/Strata)). Their licenses apply to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).

## License

Strata is open source under the [MIT License](LICENSE). A few parts carry their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml), the web app's font (SIL Open Font License 1.1) and the experimental speed projection's
vector in `data/experimental-speed-projection` (Qwen Community License 1.0, from the model's activations). The
models are not part of this repository; each model's own license applies to its files.
