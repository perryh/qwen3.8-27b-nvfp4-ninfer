#!/usr/bin/env python3
"""NInfer bench: decode, prefill (4K/32K), long-context generation.

Wall-clock based (NInfer's OpenAI usage block does not expose per-phase
timings). MTP acceptance is parsed from serve stderr logs when available.

Usage:
    python3 bench.py
    BASE=http://host:8080 MODEL=qwen3.8-27b python3 bench.py
"""
import json, os, time, urllib.request, urllib.error

BASE = os.environ.get("BASE", "http://localhost:8080")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")

def chat(payload):
    payload = {"model": MODEL, **payload}
    body = json.dumps(payload).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return {"ERROR": f"HTTP {e.code}: {e.read().decode(errors='ignore')[:200]}"}
    return {"wall_s": round(time.time() - t0, 2),
            "prompt_tokens": d.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": d.get("usage", {}).get("completion_tokens", 0),
            "content": d["choices"][0]["message"].get("content", "")}

def long_prompt(n_words):
    seq = ("The lighthouse keeper logged the weather, wind speed, barometric pressure, "
           "tide tables, cargo manifests, gull sightings, and the occasional ship's bell. ")
    return "Summarize the following log:\n\n" + seq * (n_words // 15)

results = {}

# warm-up (loads/compiles graphs paths)
r = chat({"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 16})
print("warmup:", r)

# 1. pure decode: short prompt, long generation
r = chat({"messages": [{"role": "user", "content": "Count from 1 to 200, digits only."}], "max_tokens": 700})
rate = r["completion_tokens"] / r["wall_s"]
results["decode"] = {"completion_tokens": r["completion_tokens"], "wall_s": r["wall_s"], "tok_s": round(rate, 1)}
print("DECODE:", results["decode"])

# 2. prefill 4K
r = chat({"messages": [{"role": "user", "content": long_prompt(3000)}], "max_tokens": 8})
results["prefill_4k"] = {"prompt_tokens": r["prompt_tokens"], "wall_s": r["wall_s"],
                         "prefill_tok_s": round(r["prompt_tokens"] / r["wall_s"], 1)}
print("PREFILL 4K:", results["prefill_4k"])

# 3. prefill 32K
r = chat({"messages": [{"role": "user", "content": long_prompt(32000)}], "max_tokens": 8})
results["prefill_32k"] = {"prompt_tokens": r["prompt_tokens"], "wall_s": r["wall_s"],
                          "prefill_tok_s": round(r["prompt_tokens"] / r["wall_s"], 1)}
print("PREFILL 32K:", results["prefill_32k"])

# 4. long-context generation
r = chat({"messages": [{"role": "user", "content": long_prompt(32000) + "\n\nNow write a 300-word story about the keeper's cat."}], "max_tokens": 500})
gen = r["completion_tokens"] / max(r["wall_s"] - r["prompt_tokens"] / results["prefill_32k"]["prefill_tok_s"], 0.01)
results["longctx_gen"] = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "wall_s": r["wall_s"], "gen_tok_s_est": round(gen, 1)}
print("LONGCTX GEN:", results["longctx_gen"])

json.dump(results, open("bench_results.json", "w"), indent=2)
print("\nSaved bench_results.json")
