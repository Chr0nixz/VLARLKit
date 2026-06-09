import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

CONFIG_NAME = "libero_spatial_grpo_openvlaoft"
HOST = "localhost"
PORT = 5550
DEVICE = "cuda:0"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlarlkit.models import get_model
from vlarlkit.rollouts import Rollout
from vlarlkit.utils.remote_env import RemoteEnv


def load_cfg():
    config_name = CONFIG_NAME.removesuffix(".yaml")
    config_dir = os.path.abspath(CONFIG_DIR)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    with open_dict(cfg):
        cfg.env.env_client_host = HOST
        cfg.env.env_client_base_port = PORT

    OmegaConf.resolve(cfg)
    return cfg, config_name


def summarize(info, backend, config_name, elapsed_s):
    success = np.asarray(info["success_once"], dtype=np.float32).reshape(-1)
    summary = {
        "backend": backend,
        "config": config_name,
        "episodes": int(success.size),
        "success_rate": float(success.mean()),
        "elapsed_s": float(elapsed_s),
    }

    for key in ("episode_len", "return", "reward"):
        if key in info:
            values = np.asarray(info[key], dtype=np.float32).reshape(-1)
            summary[f"{key}_mean"] = float(values.mean())

    if "task_id" in info:
        task_ids = np.asarray(info["task_id"]).reshape(-1)
        summary["per_task_success"] = {
            str(int(task_id)): float(success[task_ids == task_id].mean())
            for task_id in sorted(np.unique(task_ids).tolist())
        }

    return summary


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg, config_name = load_cfg()

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for device {DEVICE!r}.")

    start = time.time()
    model = get_model(cfg.model)
    model.to(DEVICE)
    model.eval()

    env = RemoteEnv(host=HOST, port=PORT, env_mode="eval")
    try:
        rollout = Rollout(cfg, env, model, rollout_result=None, mode="eval")
        info = rollout.run_rollout(cfg.algorithm.eval_rollout_epochs)
    finally:
        env.close()

    summary = summarize(info, cfg.model.backend, config_name, time.time() - start)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
