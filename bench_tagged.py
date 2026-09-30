#!/usr/bin/env python3
"""NInfer bench mirroring bench.py, but each long prompt carries a unique tag so
no phase can reuse another phase's prefix cache.

    BASE=http://host:8081 MODEL=qwen3.8-27b-orcarouter python3 bench_tagged.py
"""
import json, os, secrets, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8080")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
NONCE = secrets.token_hex(4)  # keeps every prompt out of another run's prefix cache

SEQ = ("The lighthouse keeper logged the weather, wind speed, barometric pressure, "
       "tide tables, cargo manifests, gull sightings, and the occasional ship's bell. ")


def long_prompt(n_words, tag):
    head = f"Summarize the following log ({tag}-{NONCE}):\n\n"
    return head + (SEQ.replace("lighthouse", "lighthouse " + tag)) * (n_words // 15)


def chat(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return {"wall_s": time.time() - t0, "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


results = {}
print(f"# base={BASE} model={MODEL}")

chat("Say OK", 16)
r = chat("Count from 1 to 200, digits only.", 700)
results["decode"] = {"completion_tokens": r["completion_tokens"], "wall_s": round(r["wall_s"], 2),
                     "tok_s": round(r["completion_tokens"] / r["wall_s"], 1)}
print("DECODE:", results["decode"])

r = chat(long_prompt(3000, "alpha"), 8)
results["prefill_6k"] = {"prompt_tokens": r["prompt_tokens"], "wall_s": round(r["wall_s"], 2),
                         "prefill_tok_s": round(r["prompt_tokens"] / r["wall_s"], 1)}
print("PREFILL 6K:", results["prefill_6k"])

r = chat(long_prompt(32000, "bravo"), 8)
results["prefill_68k"] = {"prompt_tokens": r["prompt_tokens"], "wall_s": round(r["wall_s"], 2),
                          "prefill_tok_s": round(r["prompt_tokens"] / r["wall_s"], 1)}
print("PREFILL 68K:", results["prefill_68k"])

r = chat(long_prompt(32000, "charlie") + "\n\nNow write a 300-word story about the keeper's cat.", 500)
net = r["completion_tokens"] / max(r["wall_s"] - r["prompt_tokens"] / results["prefill_68k"]["prefill_tok_s"], 0.01)
results["longctx_gen"] = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "wall_s": round(r["wall_s"], 2), "gen_tok_s_est": round(net, 1)}
print("LONGCTX GEN:", results["longctx_gen"])

print(json.dumps(results))
