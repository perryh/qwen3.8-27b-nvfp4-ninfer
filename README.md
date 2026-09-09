# NInfer serving for Qwen3.8-27B NVFP4 + MTP3 (Docker)

Serves [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
(27B dense, NVFP4 weights, MTP speculative decoding) with
[NInfer](https://github.com/Neroued/ninfer) — a from-scratch C++/CUDA engine for
RTX 5090 (sm_120a) — on a single RTX 5090 + Ryzen 5950X + 128GB DDR4.

OpenAI + Anthropic compatible APIs on port 8080. Structured for Hermes-agent /
coding use: high concurrency lanes, long context, preserved reasoning.

## Profile

| Setting | Value | Why |
|---|---|---|
| Context per request | **262,144** | model's full window (`--max-context`) |
| Shared KV pool | **auto** (maximized from free VRAM) | `--kv-capacity auto` |
| Concurrency | **8 lanes** | Hermes parallel tool calls / coding agents |
| KV dtype | `fp8` | halves KV bytes vs bf16, negligible quality delta |
| Speculative | `--spec mtp --draft-tokens 3 --lm-head-draft` | MTP3, published ~49% acceptance |
| Device checkpoint slots | 2 extra | prefix reuse across agent turns |
| Host State / Host KV | 8 slots / 8192 MiB | pinned-memory continuation cache |
| Thinking | preserved (`--preserve-thinking`) | agent transcripts keep reasoning |
| CUDA Graphs | on (default) | published decode numbers assume graphs |

`--kv-capacity auto` measures free VRAM after weights+MTP+workspace and takes
the largest legal page capacity — the "use the entire GPU" option, with NInfer's
1 GiB automatic headroom kept in place (safe; the engine freezes residency at
startup).

## Requirements

- RTX 5090 (sm_120a — the build rejects anything else) + driver r580+
- Docker + Compose v2; SELinux hosts supported (`:ro,Z` model volume)
- Artifact at `~/git/ninfer/models/qwen3_8_27b_nvfp4.ninfer` (21 GB,
  tokenizer + chat template embedded)

## Run

```bash
./run.sh          # build image (first time: long C++ build), up -d, health-wait
./run.sh test     # /health + /v1/models + live chat completion
./run.sh logs     # follow
./run.sh stop
```

Server: `http://localhost:8080` — OpenAI `/v1/chat/completions`,
`/v1/models`, `/v1/responses`; Anthropic `/v1/messages`,
`/v1/messages/count_tokens`.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":64}'
```

### Knobs (env / `.env`)

| var | default | meaning |
|---|---|---|
| `PORT` | `8080` | host port |
| `MODEL_FILE` | `/models/qwen3_8_27b_nvfp4.ninfer` | artifact path in-container |
| `MAX_CONTEXT` | `262144` | per-request context ceiling |
| `MAX_CONCURRENCY` | `8` | active-request lanes (NInfer max 8) |
| `KV_CAPACITY` | `auto` | shared KV pool (`auto` = maximize from free VRAM) |
| `KV_DTYPE` | `fp8` | KV storage type |
| `EXTRA_ARGS` | `--device-state-slots 2 --host-state-slots 8 --host-kv-mib 8192 --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking` | passthrough |

## Benchmark

`bench.py` — decode (short prompt), prefill (4K / 8K / 32K), long-context
generation; wall-clock based, reports tok/s and MTP acceptance (from stderr
stats when available). Measured results: see table below.

Measured 2026-09-09, this build (262K ctx, fp8 KV, MTP3, C=4, `--kv-capacity auto`). The GPU is power-limited to 400 W (below the 5090's 575 W stock), so these numbers are a floor for an unbounded card:

| metric | ninfer (Qwen3.8-27B nvfp4, MTP3) |
|---|---|
| decode (prose, warm, C=1) | **170.3 tok/s** |
| prefill @ 6.5K | 6,660 tok/s |
| prefill @ 68K | **5,046 tok/s** |
| long-ctx gen (68K prompt, 500 out) | ~107 tok/s net |

`--max-concurrency` must be <= 4 at 262K on the 32GB card: C=8 fails startup
with "minimum Engine runtime reservation requires 11.68 GiB + 1 GiB headroom"
(only ~11.96 GiB free after weights). Upstream's own long-context example uses
C=2. Warm the server with one throwaway request before timing.

Reference points on the same machine: ik_llama MTP (Qwen3.8-Flash-Next 125B)
~21 tok/s decode / ~300 tok/s prefill; FreeToken (same model) 48 tok/s @ 16K
ctx or 22 tok/s @ 262K. NInfer publishes 143.8 tok/s C=1 / 766.6 tok/s C=8
aggregate decode for this exact artifact (INT8 KV, CUDA Graphs, MTP3).

## Files

- `Dockerfile` — two-stage build from `nvidia/cuda:13.1.2-{devel,runtime}-ubuntu24.04`
  (mirrors upstream ninfer/Dockerfile: cmake, ninja, ffmpeg dev libs; runtime
  needs libav*60/58 + libcurl4t64)
- `docker-compose.yml` — port 8080, `:ro,Z` artifact mount, GPU reservation,
  restart policy
- `entrypoint.sh` — `ninfer-serve` args from env knobs
- `run.sh` — one-command build/start/test/logs/stop
- `bench.py` — decode/prefill/long-ctx benchmark
