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
| Context per request | **262,144** | same window as the default variant |
| Shared KV pool | **262,144 tokens** (`--kv-capacity`) | explicit, *not* `auto`: see the admission bug below |
| Concurrency | **2 lanes** | 4 crashed the engine (below); 2 held three full test rounds |
| KV dtype | `nvfp4` | at 262K an fp8 pool alone needs 9.66 GiB and the artifact leaves 8.77 GiB after weights; k8v4 misses by ~2 MB |
| Speculative | `--spec mtp --draft-tokens 3 --lm-head-draft` | MTP3, same as the default variant |
| Desktop reserve | `--desktop-reserve-gib 0` | the fork defaults to an 8 GiB floor, which cannot fit a 26 GB artifact on a 32 GB card |
| Prefill chunk | `--prefill-chunk 2048` | card's launch example |
| Thinking | preserved | as above |

### Two engine bugs to know about (fork revision `91ce2f2c`)

1. **Admission crash when the pool is oversubscribed.** With an `auto` pool
   (which sizes to 5,293 page groups / 338,752 tokens, *more* than the 4,096
   groups one 262K sequence needs) the engine dies mid-request with
   `fatal executor failure (St11logic_error: isolated-feasible request is
   blocked in an idle Engine)`. Reproduced with `bench_tagged.py` (a 70K prefill
   followed by a distinct 72K generation) at `MAX_CONCURRENCY=4`; the same
   config with an explicit `--kv-capacity 262144` also crashed at 4 lanes. At 2
   lanes with an explicit 262,144-token pool (exactly one full-context sequence)
   the reproduction, run three times, produced no crash. This is why the pool is
   explicit and concurrency is 2.
2. **`--clamp-concurrency-to-pool` cannot start.** It clamps the effective
   concurrency but the auto-sizer still picks the pre-clamp page count, so
   startup fails with `Main KV page count N is outside the target capacity curve
   [4096, 4096]`. With an explicit `--kv-capacity` it fails the other way (the
   clamp demands full backing for the configured lanes, i.e. 8,192 groups).
   Leave the flag off.

Neither bug is present in the engine the default variant runs, so the default
variant keeps its `auto` pool and 4 lanes.

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
revision is required — it cannot ride the newer engine. Serving this artifact on
the upstream engine is also blocked: upstream's `tools/upgrade_ninfer_v2_to_v3.py`
has an allow-list of known models and rejects this container with
`unsupported v2 input ('qwen3.8-27b-orcarouter', 'nvfp4') with 1190 objects`, so
the artifact stays v2 and only the fork revision can load it.

Vendor qualification, for orientation: RTX PRO 6000 Blackwell 96 GB on native
Windows, CUDA 13.3, ordinary/MTP/DFlash2 decode plus image, JSON-schema and
prefix-reuse checks; the card states RTX 5090 qualification and
maximum-context operation were **not** done. What this repo adds is our own
5090 verification below — not a substitute for it.

## Profiles' memory, as the engine reports it

`default` (262K ctx, auto KV, C=4): KV 265,152 tokens fp8, runtime reservation
10.1 GiB, 1.29 GiB free after startup.

`orcarouter` (262K ctx, explicit 262,144-token KV, C=2, nvfp4 KV): runtime
reservation 5.88 GiB, available after weights 8.77 GiB, **3.05 GiB free after
startup**. The pool is exactly one full-context sequence, so `--kv-capacity`
reports `kv_page_groups=4096 kv_max_page_groups=8192` and `kv_headroom_bytes=0`.
Sizing this pool with `auto` instead is what crashes the engine (see the
admission bug above): `auto` picks 5,293 groups, i.e. *more* than one sequence
and less than two, which is the state the scheduler cannot place a request in.
For the same reason there is no fp8 profile here: at 262K an fp8 pool alone
needs 9.66 GiB of the 8.77 GiB available, and `k8v4` misses by about 2 MB.

## Benchmarks

All on this box, 400 W power limit (the 5090's stock limit is 575 W, so every
number is a floor), CUDA graphs on, nothing else on the GPU; fp8 KV for the
default variant, nvfp4 KV for orcarouter (forced, see its profile above).

| metric | default (MTP3, 262K, C=4) | orcarouter (MTP3, 262K, C=2) |
|---|---|---|
| decode, prose, thinking off (`bench_decode.py`, 3 samples) | 135.5 / 135.1 / 134.5 tok/s | 141.6 / 141.6 / 141.1 tok/s |
| decode, 700-token generation (`bench_tagged.py`) | 179.8 tok/s | 160.3 tok/s |
| prefill @ 6.7K | 8,216 tok/s | 6,778 tok/s |
| prefill @ 70K | 6,440 tok/s | 4,868 tok/s |
| long-ctx gen (72.6K prompt, 500 out, net of prefill) | 110.0 tok/s | 107.1 tok/s |

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

Also verified on the OrcaRouter stack, on the config above: two concurrent
decodes plus a 192,062-token prompt served together (70.3 s, ~2.7K tok/s prefill
under contention, `cached_tokens: 0`), container healthy afterwards, and the
admission-crash reproduction run three times with no crash. Startup health is
not the bar for this profile; the batch is.

DFlash2 (`--spec dflash2 --draft-tokens 7 --lm-head-draft`) loads the artifact's
companion head (+1.8 GB weights, 727 tensors) and, measured under the earlier
131K/MTP5 profile, gave 142–157 tok/s on the same prose prompt — no win over MTP
on this card — while dropping free VRAM to 1.9 GiB. It is not used here.

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
