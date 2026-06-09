#!/usr/bin/env bash
set -euo pipefail

flash_ver="2.7.4.post1"
base_url="https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_ver}"

py_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
torch_mm="$(python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"{v[0]}.{v[1]}")')"
cuda_major="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')"

wheel_name="flash_attn-${flash_ver}+cu${cuda_major}torch${torch_mm}cxx11abiFALSE-${py_tag}-${py_tag}-linux_x86_64.whl"

uv pip install "${base_url}/${wheel_name}"
