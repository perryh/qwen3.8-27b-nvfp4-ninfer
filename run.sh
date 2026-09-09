#!/usr/bin/env bash
# One-command manager for the NInfer Qwen3.8-27B NVFP4 container, mirroring
# run.sh from qwen3.8-flash-next-mtp-mxfp4-ik_llama.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"

case "${1:-up}" in
  up)
    # stage the pinned ninfer source into the build context
    if [ ! -d ninfer-src ] || [ -n "${NINFER_REFRESH:-}" ]; then
      rm -rf ninfer-src
      if [ -d ../ninfer/.git ]; then
        echo ">> staging ninfer source from ../ninfer ($(git -C ../ninfer rev-parse --short HEAD))"
        mkdir -p ninfer-src
        (cd ../ninfer && git archive HEAD | tar -x -C "$OLDPWD/ninfer-src")
      else
        echo ">> cloning ninfer from upstream"
        git clone --depth 1 https://github.com/Neroued/ninfer.git ninfer-src
      fi
    fi
    docker compose up -d --build
    echo ">> waiting for health on :${PORT} ..."
    for i in $(seq 1 120); do
      if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo ">> healthy after ~$((i*5))s"; exit 0
      fi
      sleep 5
    done
    echo ">> health check timed out; run ./run.sh logs"; exit 1
    ;;
  test)
    echo "== /health =="; curl -sf "http://localhost:${PORT}/health"; echo
    echo "== /v1/models =="; curl -sf "http://localhost:${PORT}/v1/models"; echo
    echo "== chat completion =="
    curl -sf "http://localhost:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":512}' \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("content:", repr(m.get("content"))); print("reasoning chars:", len(m.get("reasoning_content") or "")); print("usage:", d.get("usage", {}))'
    ;;
  logs)
    docker compose logs -f
    ;;
  stop)
    docker compose down
    ;;
  *)
    echo "usage: ./run.sh [up|test|logs|stop]"; exit 2
    ;;
esac
