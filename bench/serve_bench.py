#!/usr/bin/env python3
"""A/B benchmark for a running Strata server over its OpenAI endpoint.

Sends a synthetic prompt of a chosen length and reports the engine's own numbers
(prompt and decode tokens/s, expert-cache hit rate) from /metrics, averaged over
several runs.  Run it against one GPU and against a layer split to compare.

  python bench/serve_bench.py --base-url http://127.0.0.1:8081 --runs 3
  python bench/serve_bench.py --prompt-tokens 4096 --max-tokens 256 --json

It needs nothing but the standard library; it does not start or stop the server.
"""
import argparse
import json
import sys
import time
import urllib.request

SENTENCE = (
    "Mixture-of-experts inference splits each token among a small number of "
    "specialists, so the memory footprint is the whole model while the compute "
    "is only a fraction of it per token. "
)


def post_json(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def make_prompt(target_tokens):
    # ~11 tokens per sentence on the Qwen tokenizer; good enough to set the length.
    repeat = max(1, target_tokens // 11)
    return SENTENCE * repeat + "\nSummarize the above in one short sentence."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="server root (not the /v1 path)")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--prompt-tokens", type=int, default=2048, help="approximate prompt length in tokens")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--json", action="store_true", help="print the raw per-run numbers as JSON")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    prompt = make_prompt(args.prompt_tokens)
    runs = []
    for i in range(args.runs):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "stream": False,
        }
        t0 = time.time()
        try:
            resp = post_json(base + "/v1/chat/completions", body, args.timeout)
        except Exception as e:  # noqa: BLE001 - report any transport/HTTP failure
            print(f"run {i + 1}: FAILED: {e}", file=sys.stderr)
            return 1
        wall = time.time() - t0
        usage = resp.get("usage", {})
        try:
            last = get_json(base + "/metrics", 30)["requests"][0]
        except Exception:  # noqa: BLE001
            last = {}
        pt = last.get("prompt_tokens") or usage.get("prompt_tokens")
        pm = last.get("prompt_ms")
        dt = last.get("decode_ms")
        ot = last.get("output_tokens") or usage.get("completion_tokens")
        run = {
            "run": i + 1,
            "prompt_tokens": pt,
            "output_tokens": ot,
            "prompt_ms": pm,
            "decode_ms": dt,
            "prompt_tok_s": round(pt / (pm / 1000), 1) if pt and pm else None,
            "decode_tok_s": round(last["decode_tok_s"], 1) if last.get("decode_tok_s") else None,
            "hit_rate": last.get("hit_rate"),
            "wall_s": round(wall, 1),
        }
        runs.append(run)
        print(
            "run {run}: prompt {pt} tok {prompt_tok_s} tok/s | decode {ot} tok "
            "{decode_tok_s} tok/s | hit {hit_rate} | {wall_s} s".format(**run)
        )

    def mean(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    summary = {"prompt_tok_s": mean("prompt_tok_s"), "decode_tok_s": mean("decode_tok_s")}
    print(f"mean: prompt {summary['prompt_tok_s']} tok/s | decode {summary['decode_tok_s']} tok/s")
    if args.json:
        print(json.dumps({"runs": runs, "mean": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
