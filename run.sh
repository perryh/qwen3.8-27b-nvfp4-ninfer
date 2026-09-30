#!/usr/bin/env bash
# One-command manager for the NInfer Qwen3.8-27B NVFP4 containers.
#
# Each variant is described by variants/<name>.env: which engine checkout to
# stage, which artifact to serve, the image/container names and the serving
# profile. Only one variant can hold the GPU at a time.
#
#   VARIANT=default     Qwen3.8-27B NVFP4, upstream NInfer, v3 artifact   :8080
#   VARIANT=orcarouter  Qwen3.8-27B Uncensored NVFP4 (OrcaRouter), fork   :8081
#
# usage: [VARIANT=<name>] ./run.sh [up|test|logs|stop]
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

VARIANT="${VARIANT:-default}"
VARIANT_FILE="variants/${VARIANT}.env"
if [ ! -f "$VARIANT_FILE" ]; then
  echo ">> unknown variant '${VARIANT}'; available: $(cd variants && ls *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
  exit 2
fi
# shellcheck disable=SC1090
set -a; . "$VARIANT_FILE"; set +a

for required in SRC_DIR NINFER_SRC IMAGE_TAG CONTAINER_NAME COMPOSE_PROJECT \
                MODEL_FILE MODEL_ID MAX_CONTEXT KV_CAPACITY MAX_CONCURRENCY \
                KV_DTYPE EXTRA_ARGS HOST_PORT; do
  [ -n "${!required:-}" ] || { echo ">> ${VARIANT_FILE} must define ${required}" >&2; exit 2; }
done
export SRC_DIR NINFER_SRC IMAGE_TAG CONTAINER_NAME COMPOSE_PROJECT MODEL_FILE \
       MODEL_ID MAX_CONTEXT KV_CAPACITY MAX_CONCURRENCY KV_DTYPE EXTRA_ARGS HOST_PORT

compose() { docker compose -p "$COMPOSE_PROJECT" "$@"; }

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
}

case "${1:-up}" in
  up)
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
  *)
    echo "usage: [VARIANT=<name>] ./run.sh [up|test|logs|stop]"; exit 2
    ;;
esac
