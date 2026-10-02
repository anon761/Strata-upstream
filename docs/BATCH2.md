# Batch-2 decode (branch `batch2`, work in progress)

Goal: decode two independent conversations in one verify window, so the dense weights are read once for both and
the CPU computes one sequence's experts while the GPU runs the other's mixer. Strata today is single-session: the
conversation cache (parking) switches between conversations, it never runs two at once.

Expected gain, before any measurement: ~1.3-1.5x aggregate decode for two sequences. Two different conversations
share fewer experts than consecutive tokens of one (the window's measured union: 1.75x one token's misses for
T=2 of ONE sequence), and the two sequences share the window width the MTP drafts use today.

## Why the window is the place

`Verifier::record_window` already runs a window as two token groups (`G = 2`, `pre(l, grp)` / `post(l, grp)`),
ordered so the CPU pool serves group A's experts while the GPU runs group B's mixer and router. Today both groups
are halves of ONE sequence. Batch-2 makes group A = sequence 1 and group B = sequence 2 (`TA` tokens and
`T - TA` tokens); B's mixer then does not even depend on A's.

Sequence-bound state inside the window (everything else is per-token scratch or shared weights):

| state | where | batch-2 |
|---|---|---|
| GDN recurrence + conv history | `ss.gdn_state` per GDN layer | group's own session |
| QSA K/V pools, indexer tail/dead/pooled/block_pos | `ss.qsa_states[qi]` | group's own session |
| indexer tail snapshot for the commit | `tail_snap_` (taken in group 0 only) | one per group |
| PLE history (+ per-token snapshots) | `ss.ple.hist`, `hist_snap_` | group's own session |
| positions / step counters | `h_step_`, `h_pos_` (already per token) | per-sequence `pos0` |
| PLE rows (n-gram hash over the previous tokens) | `ss.ple_prev` in `run` | per-sequence |

The commit (`capture_commit`) replays the kept tokens from per-token inputs stored as `[layer][token]`
(`qkv_L_`, `h_L_`, `gate_L_`, `beta_L_`, `idx_raw_L_`, `hist_snap_`). Sequence 2's tokens sit at index `TA..` in the
same arrays, so a second commit graph with a token offset, its own session, its own tail snapshot and its own
`commit_` array covers it.

## What else it touches

- **Layer split** (prod splits the 48 layers over two GPUs): the window is a chain of `Verifier` stages (`next_`),
  each with its own `SessionState` for its layer range. Sequence 2 needs a session per stage.
- **Sampling**: temperature, seed, top-p/k and the repetition-penalty history are per window today; per segment in
  batch-2 (`sample_tokens` over each segment's rows with that request's parameters and `pos0`).
- **VRAM**: a second session at `max_context` doubles the K/V pools; its context has to be capped so the expert
  cache keeps its share.
- **Drafts**: each sequence keeps its own MTP drafter state; the window width is split between the two.

## Status

**M1 done** (2026-10-02, prod 2x RTX 3090, Qwen3.8-Flash-Next-Uncensored-Q5_K_M, layer split over both GPUs, int8 K/V).
`STRATA_BATCH2_SELFTEST=<steps>` with `STRATA_BATCH2_SPLIT=na,nb` decodes two sequences alone (one-token windows)
and then teacher-forced in batch windows of na + nb tokens: the picks agree and every logit row is bitwise equal.
Passed for 1+1, 2+2, 3+2 and 1+4 over 300 steps.

First throughput numbers from the same self-test (wall time of window + commit, expert cache warm):

| windows | alone (one-token windows) | batched, both sequences |
|---|---|---|
| 1+1 | 46-51 tok/s | 58.5 tok/s |
| 2+2 (teacher-forced: every token kept) | 51.5 tok/s | 93.0 tok/s |
| 3+2 | 51.4 tok/s | 101.5 tok/s |

The 1+1 row is the clean batch gain (~1.15-1.25x). The wider rows are what a window of T tokens costs in general -
a single sequence's MTP window of the same width costs about as much, so with drafts the gain comes from two
sequences filling the window with tokens that are each kept more often than a fourth or fifth draft would be.
M4 has to measure that against the serial engine with MTP.

**M2 test done** (same day): `STRATA_BATCH2_DECODE=<tokens>` decodes the two sequences with MTP drafts, greedily,
first each alone (fixed windows of `--mtp-max-t` tokens), then in lock-step in batch windows of twice that, the second
sequence drafting with a twin drafter (`MtpDrafter::load_twin`: shared weights and draft head, its own K/V ring,
buffers and stream; ~19 MiB at 4096 cells).  400 tokens x 2, identical outputs for every width:

| tokens per sequence and window | alone | lock-step, both | gain | ms per window alone / batched |
|---|---|---|---|---|
| 4 | 49.0 tok/s | 56.9 tok/s | 1.16x | 46 / 80 |
| 3 | 54.6 tok/s | 61.1 tok/s | 1.12x | 37 / 66 |
| 2 | 77.0 tok/s | 92.1 tok/s | 1.20x | 23 / 39 |

What it says: a window's cost grows almost linearly with its rows (~10-11 ms per row on this box) - the misses of
each row's routed experts, computed on the CPU, dominate, and two different conversations share few of them.  The
part a batch shares (dense weights, the fixed cost per window) is small here, so batch-2 tops out at ~1.2x.  (The
alone column uses fixed windows; prod adapts the width with `--spec-min-p`.)

## Milestones

1. **M1 engine**: second `SessionState` per stage; `record_window` with per-group sessions and a free split `TA`;
   a commit graph per sequence; a parity program: a batch window `[A | B]` gives the same argmax (and bitwise the
   same logits) as A and B in separate windows.
2. **M2 decode loop**: two requests decode in lock-step, drafts per sequence; greedy parity with the serial run.
3. **M3 server**: `serve/server.py` admits a second request into the running batch (prompt reading stays serial).
4. **M4 measurement** on the 2x RTX 3090: aggregate tok/s with two agents vs serial, hit rates, VRAM.
