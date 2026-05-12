#!/usr/bin/env bash
set -euo pipefail

IP="127.0.0.1"
PORT=8001
LOAD_MODEL_PATH="/home/mila/s/sunyi/scratch/Bagel-WM/Bagel-libero-goal"
DATA_DIR="/home/mila/s/sunyi/codes/VLARLKit/tests/test_bagel_wm/test_data"

BAGEL_ENV_PATH="third_party/BAGEL/.venv"
VLARLKIT_ENV_PATH=".venv"
SERVER_STARTUP_SLEEP=5
RECV_TIMEOUT_MS=900000
SAVE_OUTPUT_DIR="/home/mila/s/sunyi/codes/VLARLKit/tests/test_bagel_wm/test_data/rollout_results"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/BAGEL:${PYTHONPATH:-}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting BAGEL WM server at ${IP}:${PORT}"
source "${REPO_ROOT}/${BAGEL_ENV_PATH}/bin/activate"
python -m env_clients.world_models.bagel.client \
    --load-model-path "${LOAD_MODEL_PATH}" \
    --host "${IP}" \
    --port "${PORT}" &
SERVER_PID=$!
deactivate

sleep "${SERVER_STARTUP_SLEEP}"
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}"
    exit 1
fi

source "${REPO_ROOT}/${VLARLKIT_ENV_PATH}/bin/activate"
TEST_CMD=(
    python tests/test_bagel_wm/test_bagel_wm.py
    --data-dir "${DATA_DIR}"
    --host "${IP}"
    --port "${PORT}"
    --recv-timeout-ms "${RECV_TIMEOUT_MS}"
    --close-server
)

if [[ -n "${SAVE_OUTPUT_DIR}" ]]; then
    TEST_CMD+=(--save-output-dir "${SAVE_OUTPUT_DIR}")
fi

"${TEST_CMD[@]}"
deactivate
