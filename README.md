# NInfer serving for Qwen3.8-27B NVFP4 (Docker, two variants)

Serves 27B dense NVFP4 artifacts with [NInfer](https://github.com/Neroued/ninfer) —
a from-scratch C++/CUDA engine for RTX 5090 (sm_120a) — on a single RTX 5090 +
Ryzen 5950X + 128GB DDR4. OpenAI + Anthropic compatible APIs, one container per
variant, long context, preserved reasoning, MTP/DFlash2 speculative decoding.

| variant | model | engine | port |
|---|---|---|---|
| `default` | Qwen3.8-27B NVFP4 (Neroued) | upstream NInfer @ `d44ab584`, CUDA 13.1.2 | 8080 |
| `orcarouter` | Qwen3.8-27B **Uncensored** NVFP4 (OrcaRouter, abliterated) | igorls fork @ `91ce2f2c`, CUDA 13.3.1 | 8081 |

Both stacks want the whole card, so **only one runs at a time** (`docker ps`
should show a single `ninfer*` container).

## Run

```bash
./run.sh                      # default variant: build (first time: long C++ build), up -d, health-wait
./run.sh test                 # /health + /v1/models + live chat completion
./run.sh logs
./run.sh stop

VARIANT=orcarouter ./run.sh up     # the OrcaRouter uncensored variant
VARIANT=orcarouter ./run.sh test
VARIANT=orcarouter ./run.sh stop
```

An explicit environment override wins over `variants/<name>.env`, so a one-off
profile change needs no edit:

```bash
VARIANT=orcarouter MAX_CONTEXT=32768 KV_CAPACITY=65536 ./run.sh up
```

Server on `http://localhost:<port>` — OpenAI `/v1/chat/completions`,
`/v1/models`, `/v1/responses`; Anthropic `/v1/messages`,
`/v1/messages/count_tokens`.

```bash
curl http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-orcarouter","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":64}'
```

## Variant: default (Neroued Qwen3.8-27B NVFP4)

| Setting | Value | Why |
|---|---|---|
| Context per request | **262,144** | model's full window (`--max-context`) |
| Shared KV pool | **auto** (265,152 tokens measured) | `--kv-capacity auto` |
| Concurrency | **4 lanes** (8 doesn't fit at 262K on 32GB) | Hermes parallel tool calls / coding agents |
| KV dtype | `fp8` | halves KV bytes vs bf16, negligible quality delta |
| Speculative | `--spec mtp --draft-tokens 3 --lm-head-draft` | MTP3, published ~49% acceptance |
| Device checkpoint slots | 2 extra | prefix reuse across agent turns |
| Host State / Host KV | 8 slots / 8192 MiB | pinned-memory continuation cache |
| Thinking | preserved (`--preserve-thinking`) | agent transcripts keep reasoning |
| CUDA Graphs | on (default) | published decode numbers assume graphs |

Engine: `~/git/ninfer` (upstream master, pinned to `d44ab584`), staged into the
build context by `run.sh`.

**The artifact is now v3.** The engine from 2026-09-29 rejects v2 containers:
`NInfer v2 artifact is not supported. Upgrade to v3 with:
python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer`. The
downloaded 21 GB v2 file was upgraded that way to
`qwen3_8_27b_nvfp4.v3.ninfer` (weights preserved, maintained chat template
installed, +24 KB); the v2 original is kept alongside it and is only readable by
the older images.

## Variant: orcarouter (Qwen3.8-27B Uncensored NVFP4)

| Setting | Value | Why |
|---|---|---|
| Context per request | **131,072** | largest verified here; the vendor's own example used 32K |
| Shared KV pool | **131,072 tokens** (`--kv-capacity`) | explicit; `auto` resolves to 203,840 and leaves only 1.2 GiB free |
| Concurrency | **2 lanes** | the 26 GB artifact leaves ~8.8 GiB after weights |
| KV dtype | `fp8` | as above |
| Speculative | `--spec mtp --draft-tokens 5 --lm-head-draft` | MTP K5, the card's recommended pair |
| Desktop reserve | `--desktop-reserve-gib 0` | the fork defaults to an 8 GiB floor, which cannot fit a 26 GB artifact on a 32 GB card |
| Prefill chunk | `--prefill-chunk 2048` | card's launch example |
| Thinking | preserved | as above |

Artifact — [igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer](https://huggingface.co/igorls/Qwen3.8-27B-Uncensored-NVFP4-NInfer):

| Property | Value |
|---|---|
| File | `models/orcarouter/qwen3_8_27b_orcarouter_nvfp4.ninfer` |
| Size | 26,268,462,848 bytes (24.5 GiB) |
| SHA-256 (verified on download) | `003f8c65175e262e66f83a803f7b1d286f7226ede0e672d3b3e5c4b6c75f3e7f` |
| Engine identity | `qwen3.8-27b-orcarouter` / `nvfp4`, target `qwen3_8_27b` |
| Source model | `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` @ `69d21348` (Apache-2.0) |
| Companion | `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df6` (Apache-2.0) |
| Contents | text + Vision + MTP + DFlash2 + optimized proposal head, BF16 embeddings and full output head preserved |

Engine: the fork revision the artifact's manifest pins
(`minimum_revision` = `91ce2f2c`, "feat(models): support OrcaRouter Qwen3.8-27B
NVFP4"), checked out at `~/git/ninfer-orcarouter`. That revision is **MSVC-only**
in practice, so this repo carries the Linux fixes:

- `patches/orcarouter/0001-linux-build-portability.patch`, applied by `run.sh`
  after staging: `<algorithm>` for a `std::max({...})` call, `__align__` instead
  of `alignas` on an `extern __shared__` array, explicit template arguments for
  `checked_add`/`checked_mul` calls that only deduce where `size_t` is
  `unsigned long long`.
- `CUDA_VERSION=13.3.1` (the revision's qualified toolkit). Its Flash-Next
  target includes `<cub/device/device_topk.cuh>`, which does not exist in CUDA
  13.1's CCCL at all.

Upstream and the fork's current `workstation` line have **no** OrcaRouter
support (`git grep -i orcarouter` on `Neroued/ninfer` master: no hits; the
fork's `workstation` head no longer registers the identity), so the pinned
revision is required — it cannot ride the newer engine.

Vendor qualification, for orientation: RTX PRO 6000 Blackwell 96 GB on native
Windows, CUDA 13.3, ordinary/MTP/DFlash2 decode plus image, JSON-schema and
prefix-reuse checks; the card states RTX 5090 qualification and
maximum-context operation were **not** done. What this repo adds is our own
5090 verification below — not a substitute for it.

## Profiles' memory, as the engine reports it

`default` (262K ctx, auto KV, C=4): KV 265,152 tokens fp8, runtime reservation
10.1 GiB, 1.29 GiB free after startup.

`orcarouter` (131K ctx, 131K KV, C=2): runtime reservation 5.4 GiB,
available after weights 8.8 GiB, **3.5 GiB free after startup**. With `auto`
instead of an explicit pool the engine resolves 203,840 tokens and warns
`KV pool is oversubscribed (77.8% full-context backing)` with 1.2 GiB free —
that profile boots and serves, but it is not the default here.

## Benchmarks

All on this box, 400 W power limit (the 5090's stock limit is 575 W, so every
number is a floor), fp8 KV, CUDA graphs on, nothing else on the GPU.

| metric | default (MTP3, 262K, C=4) | orcarouter (MTP5, 128K, C=2) |
|---|---|---|
| decode, prose, thinking off (`bench_decode.py`, 3 samples) | 135.5 / 135.1 / 134.5 tok/s | 143.6 / 144.0 / 143.7 tok/s |
| decode, 700-token generation (`bench_tagged.py`) | 179.8 tok/s | 193.6 tok/s |
| prefill @ 6.7K | 8,216 tok/s | 6,993 tok/s |
| prefill @ 70K | 6,440 tok/s | 4,994 tok/s |
| long-ctx gen (72.6K prompt, 500 out, net of prefill) | 110.0 tok/s | 106.8 tok/s |

Measured 2026-09-30 against the builds pinned above. Notes that matter when
re-measuring:

- **Use a unique prompt per phase.** `bench.py` reuses one long prompt across
  its prefill and long-context phases, so on a warm server those phases hit
  prefix reuse and report fictional rates (a 70K-token prefill "completing" in
  0.16 s). `bench_tagged.py` seeds each prompt with a random nonce; on this
  pair it produces 8.2K/7.0K tok/s at 6.7K and 6.4K/5.0K at 70K, which matches
  the engine's order of magnitude.
- Decode rate depends on the token stream, so compare with thinking off and a
  fixed output budget (`bench_decode.py` does that; ~144 vs ~135 tok/s prose is
  reproducible to ±1 tok/s).
- Warm up once before timing: the first request after start compiles CUDA-graph
  paths.

Also verified on the OrcaRouter stack: two concurrent decodes plus a
128,062-token prompt served together (36.3 s, ~3.5K tok/s prefill under
contention) with the container healthy afterwards — the config survives its
first real batch, not just startup. DFlash2 (`--spec dflash2 --draft-tokens 7
--lm-head-draft`) loads the companion (+1.8 GB weights, 727 tensors) and
measured 142–157 tok/s on the same prose prompt, i.e. no win over MTP5 here; it
also drops free VRAM at 131K ctx to 1.9 GiB. MTP5 is the default for that
reason.

Reference points on the same machine: ik_llama MTP (Qwen3.8-Flash-Next 125B)
~21 tok/s decode / ~300 tok/s prefill; FreeToken (same model) 48 tok/s @ 16K
ctx or 22 tok/s @ 262K. NInfer publishes 143.8 tok/s C=1 / 766.6 tok/s C=8
aggregate decode for the Neroued artifact (INT8 KV, CUDA Graphs, MTP3), and the
OrcaRouter card publishes 70.8 ordinary / 196.0 MTP K5 / 208.8 DFlash2 K7 on an
RTX PRO 6000 for a 169-token coding prompt (same order as our 5090 numbers).

## Requirements

- RTX 5090 (sm_120a — both builds reject anything else) + driver r580+;
  `orcarouter` builds and runs on CUDA 13.3.1 (driver here: 610.57.04)
- Docker + Compose v2; SELinux hosts supported (`:ro,Z` model volume)
- Engine checkouts next to this repo: `~/git/ninfer` (default) and
  `~/git/ninfer-orcarouter` (fork revision, `git worktree` of `~/git/ninfer`)
- Artifacts under `~/git/ninfer/models/` (mounted at `/models`):
  - `qwen3_8_27b_nvfp4.v3.ninfer` — default, 21.5 GB (v3 upgrade of the
    downloaded v2 file, which is kept)
  - `orcarouter/qwen3_8_27b_orcarouter_nvfp4.ninfer` — orcarouter, 26.3 GB
- On another machine, adjust `NINFER_SRC`, `MODEL_DIR` and `MODEL_FILE` in the
  variant file; `CUDA_VERSION`, `CMAKE_EXTRA_ARGS` and `SRC_DIR` describe how
  each engine is built.

## Files

- `run.sh` — variant dispatcher: stage engine source (patches applied), build,
  up, health-wait, test, logs, stop
- `variants/default.env`, `variants/orcarouter.env` — every per-variant knob
  (engine path, artifact, image/container/project names, port, profile)
- `patches/orcarouter/` — Linux build fixes for the fork revision
- `Dockerfile` — two-stage CUDA devel → runtime build (mirrors upstream's):
  `SRC_DIR` picks the staged engine tree, `CUDA_VERSION` the toolkit,
  `CMAKE_EXTRA_ARGS` extra configure flags
- `docker-compose.yml` — image/container/port/artifact/profile from the
  variant's exported env
- `entrypoint.sh` — `ninfer-serve` args from env knobs
- `bench.py` — original decode/prefill/long-ctx benchmark (see the prefix-cache
  caveat above)
- `bench_tagged.py` — same phases with a unique prompt per phase
- `bench_decode.py` — controlled decode probe (thinking off, temperature 0)
