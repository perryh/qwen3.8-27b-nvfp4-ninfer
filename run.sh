#!/usr/bin/env bash
# One-command manager for the NInfer Qwen3.8-27B NVFP4 containers.
#
# Each variant is described by variants/<name>.env: which engine checkout to
# stage, which artifact to serve, the image/container names and the serving
# profile. Only one variant can hold the GPU at a time.
#
#   VARIANT=default     Qwen3.8-27B NVFP4, upstream NInfer, v3 artifact
#   VARIANT=orcarouter  Qwen3.8-27B Uncensored NVFP4 (OrcaRouter), fork
#
# Both variants publish the SAME port (8080) and the same model alias, so a
# single agent profile works whichever model is loaded. Only one can hold the
# GPU at a time, so `up` stops the other variant first.
#
# usage: [VARIANT=<name>] ./run.sh [up|test|logs|status|stop]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

VARIANT="${VARIANT:-default}"
VARIANT_FILE="variants/${VARIANT}.env"
if [ ! -f "$VARIANT_FILE" ]; then
  echo ">> unknown variant '${VARIANT}'; available: $(cd variants && ls *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
  exit 2
fi
# Every knob a variant defines; an explicit environment override wins over the
# variant file (capture before sourcing, re-apply after).
VARIANT_KEYS="SRC_DIR NINFER_SRC IMAGE_TAG CONTAINER_NAME COMPOSE_PROJECT MODEL_FILE \
MODEL_ID MAX_CONTEXT KV_CAPACITY MAX_CONCURRENCY KV_DTYPE EXTRA_ARGS HOST_PORT \
CMAKE_EXTRA_ARGS CUDA_VERSION"

OVERRIDES=""
for key in $VARIANT_KEYS; do
  if [ -n "${!key+x}" ]; then
    OVERRIDES+="$(printf '%s=%q; ' "$key" "${!key}")"
  fi
done

# shellcheck disable=SC1090
set -a; . "$VARIANT_FILE"; set +a
[ -n "$OVERRIDES" ] && eval "$OVERRIDES"

for required in SRC_DIR NINFER_SRC IMAGE_TAG CONTAINER_NAME COMPOSE_PROJECT \
                MODEL_FILE MODEL_ID MAX_CONTEXT KV_CAPACITY MAX_CONCURRENCY \
                KV_DTYPE EXTRA_ARGS HOST_PORT; do
  [ -n "${!required:-}" ] || { echo ">> ${VARIANT_FILE} must define ${required}" >&2; exit 2; }
done
CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-}"
CUDA_VERSION="${CUDA_VERSION:-13.1.2}"
export SRC_DIR NINFER_SRC IMAGE_TAG CONTAINER_NAME COMPOSE_PROJECT MODEL_FILE \
       MODEL_ID MAX_CONTEXT KV_CAPACITY MAX_CONCURRENCY KV_DTYPE EXTRA_ARGS HOST_PORT \
       CMAKE_EXTRA_ARGS CUDA_VERSION

compose() { docker compose -p "$COMPOSE_PROJECT" "$@"; }

# Variant patches for staged engine sources that need them (kept in-repo so the
# exact build is reproducible). Applied right after staging, skipped when the
# stage already exists.
apply_patches() {
  local dir="patches/${VARIANT}"
  [ -d "$dir" ] || return 0
  local p
  for p in "${dir}"/*.patch; do
    [ -e "$p" ] || continue
    echo ">> applying $(basename "$p")"
    patch -p1 -d "${ROOT}/${SRC_DIR}" --forward --silent < "$p" \
      || { echo ">> failed to apply $p" >&2; exit 1; }
  done
}

# Stage the variant's engine checkout into the build context (content only, no
# .git). Skips when already staged unless NINFER_REFRESH=1.
stage_source() {
  if [ -e "${ROOT}/${SRC_DIR}" ] && [ -z "${NINFER_REFRESH:-}" ]; then
    return
  fi
  rm -rf "${ROOT:?}/${SRC_DIR}"
  if [ -e "${NINFER_SRC}/.git" ]; then
    echo ">> staging ${VARIANT} engine from ${NINFER_SRC} ($(git -C "$NINFER_SRC" rev-parse --short HEAD))"
    mkdir -p "${ROOT}/${SRC_DIR}"
    ( cd "$NINFER_SRC" && git archive HEAD ) | tar -x -C "${ROOT}/${SRC_DIR}"
  else
    echo ">> ${NINFER_SRC} is not a git checkout; cloning upstream into ${SRC_DIR}"
    git clone --depth 1 https://github.com/Neroued/ninfer.git "${ROOT}/${SRC_DIR}"
  fi
  apply_patches
}

# Both variants publish the same port, so a second one cannot start while the
# first still holds it (the engine would exit on bind failure). `up` stops the
# others first: switching models is one command, not a stop plus a start.
stop_others() {
  local f other proj name
  for f in variants/*.env; do
    other="$(basename "$f" .env)"
    [ "$other" = "$VARIANT" ] && continue
    proj="$(sed -n 's/^COMPOSE_PROJECT=//p' "$f" | head -1)"
    name="$(sed -n 's/^CONTAINER_NAME=//p' "$f" | head -1)"
    [ -n "$proj" ] && [ -n "$name" ] || continue
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      echo ">> stopping ${other}: ${name} holds port :${HOST_PORT}, one model at a time"
      docker compose -p "$proj" down >/dev/null 2>&1 || true
    fi
  done
}

case "${1:-up}" in
  up)
    stop_others
    stage_source
    compose up -d --build
    echo ">> waiting for health on :${HOST_PORT} ..."
    for i in $(seq 1 120); do
      if curl -sf "http://localhost:${HOST_PORT}/health" >/dev/null 2>&1; then
        echo ">> healthy after ~$((i*5))s"; exit 0
      fi
      sleep 5
    done
    echo ">> health check timed out; run VARIANT=${VARIANT} ./run.sh logs"; exit 1
    ;;
  test)
    echo "== variant ${VARIANT} | model ${MODEL_ID} | ${MODEL_FILE} =="
    echo "== /health =="; curl -sf "http://localhost:${HOST_PORT}/health"; echo
    echo "== /v1/models =="; curl -sf "http://localhost:${HOST_PORT}/v1/models"; echo
    echo "== chat completion =="
    curl -sf "http://localhost:${HOST_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer with just the number.\"}],\"max_tokens\":512}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("content:", repr(m.get("content"))); print("reasoning chars:", len(m.get("reasoning_content") or "")); print("usage:", d.get("usage", {}))'
    ;;
  logs)
    compose logs -f
    ;;
  stop)
    compose down
    ;;
  status)
    echo "== variants (one at a time, same port) =="
    for f in variants/*.env; do
      v="$(basename "$f" .env)"
      printf '   %-11s :%-5s model %-22s CUDA %s\n' "$v" \
        "$(sed -n 's/^HOST_PORT=//p' "$f" | head -1)" \
        "$(sed -n 's/^MODEL_ID=//p' "$f" | head -1)" \
        "$(sed -n 's/^CUDA_VERSION=//p' "$f" | head -1)"
    done
    echo "== containers =="
    docker ps -a --format '{{.Names}}\t{{.Status}}' | grep ninfer || echo "   (none)"
    echo "== :${HOST_PORT} =="
    curl -sf "http://localhost:${HOST_PORT}/v1/models" || echo "   (nothing answering)"
    echo
    ;;
  *)
    echo "usage: [VARIANT=<name>] ./run.sh [up|test|logs|status|stop]"; exit 2
    ;;
esac
