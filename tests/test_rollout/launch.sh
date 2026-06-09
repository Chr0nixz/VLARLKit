#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BACKEND="openvla_oft"
CONFIG_NAME="libero_spatial_grpo_openvlaoft"
PORT=5550
GPU=0
ENV_CONDA="libero"
WAIT_SECONDS=20
ENV_PID=""

PROJECT="model_backends/$BACKEND"
if [[ ! -d "$PROJECT" ]]; then
    echo "Unknown backend project: $PROJECT" >&2
    exit 2
fi

CONFIG="tests/test_rollout/configs/${CONFIG_NAME}.yaml"

cleanup() {
    if [[ -n "$ENV_PID" ]]; then
        kill -TERM -- "-$ENV_PID" 2>/dev/null || kill "$ENV_PID" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$ENV_PID" 2>/dev/null || true
    fi

    PORT_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$PORT_PIDS" ]]; then
        kill $PORT_PIDS 2>/dev/null || true
        sleep 1
        PORT_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
        [[ -z "$PORT_PIDS" ]] || kill -KILL $PORT_PIDS 2>/dev/null || true
    fi

    wait "$ENV_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

setsid -w env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$ENV_CONDA" \
    python -m env_clients.client \
    --config "$CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --rank 0 \
    --world_size 1 \
    --modes eval &
ENV_PID="$!"

sleep "$WAIT_SECONDS"

CUDA_VISIBLE_DEVICES="$GPU" uv run --project "$PROJECT" \
    python tests/test_rollout/test_rollout.py
