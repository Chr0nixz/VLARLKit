#!/bin/bash
set -euo pipefail

cd ~/codes/VLARLKit

NPROC=4
CONFIG_NAME="libero_goal_vla_mbpo"
CONFIG="examples/configs/${CONFIG_NAME}.yaml"

WM_BASE_PORT=8002
ENV_BASE_PORT=5550
LOAD_MODEL_PATH="/home/mila/s/sunyi/scratch/Bagel-WM/Bagel-libero-goal"

WM_PIDS=()
ENV_PIDS=()

cleanup() {
    for pid in "${ENV_PIDS[@]}" "${WM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# launch BAGEL world-model clients on gpu0-3
for ((i=0; i<NPROC; i++)); do
    CUDA_VISIBLE_DEVICES="$i" third_party/BAGEL/.venv/bin/python \
        -m env_clients.world_models.bagel.client \
        --load-model-path "$LOAD_MODEL_PATH" \
        --host 0.0.0.0 --port $((WM_BASE_PORT + i)) &
    WM_PIDS+=($!)
done

# launch eval environment clients on gpu4-7
for ((i=0; i<NPROC; i++)); do
    gpu=$((i + 4))
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n libero \
        python -m env_clients.client \
        --config "$CONFIG" \
        --host 0.0.0.0 --port $((ENV_BASE_PORT + i)) \
        --rank "$i" --world_size "$NPROC" \
        --modes eval &
    ENV_PIDS+=($!)
done

# launch VLA-MBPO training on gpu4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun --nproc_per_node="$NPROC" \
    examples/train_vla_mbpo.py \
    --config-name "$CONFIG_NAME" \
    world_model.base_port="$WM_BASE_PORT" \
    world_model.load_model_path="$LOAD_MODEL_PATH" \
    env.env_client_base_port="$ENV_BASE_PORT"
