#!/usr/bin/env python3
"""Controlled decode probe: same prompt, thinking off, fixed output budget, so
decode tok/s is comparable across builds and runs.

    BASE=http://host:8080 MODEL=qwen3.8-27b python3 decode_probe.py
"""
import json, os, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8080")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
N = int(os.environ.get("SAMPLES", "2"))


def run(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      "Write a detailed 400-word explanation of how ocean tides work, "
                      "for a curious teenager. Plain prose, no lists."}],
        "max_tokens": 700,
        "reasoning_effort": "none",
        "temperature": 0,
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    u = d["usage"]
    print(f"sample {i}: {u['completion_tokens']:4d} tok / {wall:5.2f}s = "
          f"{u['completion_tokens'] / wall:6.1f} tok/s   "
          f"reasoning {u.get('completion_tokens_details', {}).get('reasoning_tokens')}   "
          f"head={d['choices'][0]['message'].get('content', '')[:40]!r}")


print(f"# base={BASE} model={MODEL} (thinking off, temperature 0)")
for i in range(1, N + 1):
    run(i)
