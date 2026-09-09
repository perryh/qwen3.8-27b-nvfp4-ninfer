#!/bin/bash
# NInfer serve entrypoint — args from env knobs (see docker-compose.yml)
set -euo pipefail

MODEL_FILE="${MODEL_FILE:-/models/qwen3_8_27b_nvfp4.ninfer}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
KV_CAPACITY="${KV_CAPACITY:-auto}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
KV_DTYPE="${KV_DTYPE:-fp8}"
EXTRA_ARGS="${EXTRA_ARGS:---device-state-slots 2 --host-state-slots 8 --host-kv-mib 8192 --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking}"

ARGS=(serve "$MODEL_FILE" --host "$HOST" --port "$PORT"
      --max-context "$MAX_CONTEXT"
      --kv-capacity "$KV_CAPACITY"
      --max-concurrency "$MAX_CONCURRENCY"
      --kv-dtype "$KV_DTYPE")

# Optional flags pass-through (word-split intentionally)
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2086
  ARGS+=( $EXTRA_ARGS )
fi

exec /usr/local/bin/ninfer-serve "${ARGS[@]}"
