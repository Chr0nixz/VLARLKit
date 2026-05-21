"""
Smoke test for BAGEL WM autoregressive branch rollouts.

The test reads branch rollout start states from
examples/configs/libero_goal_vla_mbpo.yaml:
    algorithm.branch_dataset_root / algorithm.branch_eval_repo_id

Each loaded start state provides the initial head/wrist observations, task, and
dataset action. If the sample only has one action, that action is reused while
the predicted observations are fed back autoregressively for the requested
number of rollout steps.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vlarlkit.utils.remote_wm import RemoteWM


ACTION_DIM = 7
DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "configs" / "libero_goal_vla_mbpo.yaml"
IMAGE_COLUMNS = ("image", "wrist_image")
OPTIONAL_SAMPLE_COLUMNS = (
    "state",
    "actions",
    "action",
    "task_id",
    "task_index",
    "episode_index",
    "frame_index",
    "index",
)


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=False))
    return dict(value)


def _load_config(config_path: Path) -> DictConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def _resolve_branch_dataset_root(cfg: DictConfig) -> Path:
    algo_cfg = cfg.algorithm
    dataset_root = Path(str(algo_cfg.branch_dataset_root)).expanduser()
    repo_id = str(algo_cfg.branch_eval_repo_id)

    if (dataset_root / "meta").exists():
        return dataset_root

    candidates = [dataset_root / repo_id, dataset_root / repo_id.split("/")[-1]]
    for candidate in candidates:
        if (candidate / "meta").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find branch eval dataset for repo_id={repo_id!r} under {dataset_root}."
    )


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Task metadata not found: {tasks_path}")

    tasks = {}
    with tasks_path.open("r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            try:
                tasks[int(item["task_index"])] = str(item["task"])
            except KeyError as exc:
                raise KeyError(f"{tasks_path}:{line_idx} is missing {exc}.") from exc
    if not tasks:
        raise ValueError(f"No tasks found in {tasks_path}.")
    return tasks


def _parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset data directory not found: {data_dir}")

    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}.")
    return files


def _read_start_samples(
    dataset_root: Path,
    sample_index: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    samples = []
    seen = 0
    target_end = sample_index + num_samples
    for parquet_path in _parquet_files(dataset_root):
        if seen >= target_end:
            return samples
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        columns = [
            column
            for column in (*IMAGE_COLUMNS, *OPTIONAL_SAMPLE_COLUMNS)
            if column in schema_names
        ]
        table = pq.read_table(parquet_path, columns=columns)
        for row in table.to_pylist():
            if seen < sample_index:
                seen += 1
                continue
            if seen >= target_end:
                return samples
            samples.append(row)
            seen += 1

    if len(samples) != num_samples:
        raise ValueError(
            f"Requested {num_samples} sample(s) from index {sample_index}, "
            f"but only found {seen} sample(s)."
        )
    return samples


def _image_to_array(image: Any) -> np.ndarray:
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        elif image.get("path") is not None:
            image = Image.open(image["path"]).convert("RGB")
        else:
            raise ValueError("Image dict must contain either 'bytes' or 'path'.")

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image [H, W, 3] or [3, H, W], got {arr.shape}.")

    if np.issubdtype(arr.dtype, np.floating):
        if arr.size > 0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _load_action_sequence(sample: dict[str, Any], sample_idx: int) -> np.ndarray:
    action_value = sample.get("actions", sample.get("action"))
    if action_value is None:
        raise KeyError(f"Sample {sample_idx} does not contain 'actions' or 'action'.")

    actions = np.asarray(action_value, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"Sample {sample_idx} actions should be [T, {ACTION_DIM}], got {actions.shape}."
        )
    return actions


def _task_description(sample: dict[str, Any], tasks: dict[int, str], sample_idx: int) -> str:
    if "task_index" not in sample:
        raise KeyError(f"Sample {sample_idx} does not contain 'task_index'.")
    task_index = int(sample["task_index"])
    if task_index not in tasks:
        raise KeyError(f"Sample {sample_idx} has unknown task_index={task_index}.")
    return tasks[task_index]


def _sample_label(sample: dict[str, Any], sample_idx: int) -> str:
    episode = sample.get("episode_index", "unknown")
    frame = sample.get("frame_index", "unknown")
    return f"sample={sample_idx}, episode={episode}, frame={frame}"


def _sample_to_obs(
    sample: dict[str, Any],
    tasks: dict[int, str],
    sample_idx: int,
) -> tuple[dict[str, Any], np.ndarray]:
    for column in IMAGE_COLUMNS:
        if column not in sample:
            raise KeyError(f"Sample {sample_idx} does not contain image column {column!r}.")

    observations = {
        "main_images": _image_to_array(sample["image"])[None],
        "wrist_images": _image_to_array(sample["wrist_image"])[None],
        "extra_view_images": None,
        "task_descriptions": [_task_description(sample, tasks, sample_idx)],
    }
    if "state" in sample and sample["state"] is not None:
        observations["states"] = np.asarray(sample["state"], dtype=np.float32)[None]

    return observations, _load_action_sequence(sample, sample_idx)


def _action_for_step(action_sequence: np.ndarray, step_idx: int) -> np.ndarray:
    action_idx = min(step_idx, action_sequence.shape[0] - 1)
    return action_sequence[action_idx][None]


def _check_prediction(next_observations: dict, sample_idx: int, step_idx: int) -> None:
    for key in ("main_images", "wrist_images"):
        images = np.asarray(next_observations[key])
        if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3:
            raise AssertionError(
                f"Sample {sample_idx} step {step_idx} predicted {key} should be "
                f"[1, H, W, 3], got {images.shape}."
            )
        if images.dtype != np.uint8:
            raise AssertionError(
                f"Sample {sample_idx} step {step_idx} predicted {key} dtype should be uint8."
            )


def _save_prediction(
    output_dir: Path,
    sample_idx: int,
    step_idx: int,
    next_observations: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, prefix in (("main_images", "head"), ("wrist_images", "wrist")):
        image = np.asarray(next_observations[key])[0]
        output_path = output_dir / f"{prefix}_sample_{sample_idx}_step_{step_idx}.png"
        Image.fromarray(image).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressive BAGEL WM rollout from configured branch eval starts."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--recv-timeout-ms", type=int, default=900_000)
    parser.add_argument("--save-output-dir", type=Path, default=None)
    parser.add_argument("--close-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")

    cfg = _load_config(args.config_path)
    dataset_root = _resolve_branch_dataset_root(cfg)
    tasks = _load_tasks(dataset_root)
    samples = _read_start_samples(
        dataset_root=dataset_root,
        sample_index=args.sample_index,
        num_samples=args.num_samples,
    )

    world_model_cfg = cfg.get("world_model", {})
    host = args.host or str(world_model_cfg.get("host", "127.0.0.1"))
    port = args.port or int(world_model_cfg.get("base_port", 8002))
    edit_kwargs = _plain_dict(world_model_cfg.get("edit_kwargs", {}))
    und_kwargs = _plain_dict(world_model_cfg.get("und_kwargs", {}))

    print(
        f"Loaded {len(samples)} branch eval start sample(s) from {dataset_root} "
        f"using {args.config_path}."
    )
    world_model = RemoteWM(
        host=host,
        port=port,
        recv_timeout_ms=args.recv_timeout_ms,
    )

    try:
        for local_idx, sample in enumerate(samples):
            sample_idx = args.sample_index + local_idx
            observations, action_sequence = _sample_to_obs(sample, tasks, sample_idx)
            task = observations["task_descriptions"][0]
            label = _sample_label(sample, sample_idx)
            print(
                f"{label}: task={task!r}, action_sequence={action_sequence.shape}, "
                f"rollout_steps={args.rollout_steps}"
            )

            for step_idx in range(args.rollout_steps):
                actions = _action_for_step(action_sequence, step_idx)
                next_observations, rewards, terminations = world_model.step(
                    observations=observations,
                    actions=actions,
                    image_key="main_images",
                    wrist_image_key="wrist_images",
                    edit_kwargs=edit_kwargs,
                    und_kwargs=und_kwargs,
                )
                _check_prediction(next_observations, sample_idx, step_idx)

                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                terminations = np.asarray(terminations, dtype=bool).reshape(-1)
                if rewards.shape != (1,) or terminations.shape != (1,):
                    raise AssertionError(
                        f"Sample {sample_idx} step {step_idx} expected scalar reward "
                        f"and termination, got {rewards.shape} and {terminations.shape}."
                    )
                print(
                    f"{label}: step={step_idx + 1}/{args.rollout_steps}, "
                    f"action_idx={min(step_idx, action_sequence.shape[0] - 1)}, "
                    f"reward={float(rewards[0]):.1f}, done={bool(terminations[0])}"
                )

                if args.save_output_dir is not None:
                    _save_prediction(
                        args.save_output_dir,
                        sample_idx,
                        step_idx,
                        next_observations,
                    )
                observations = next_observations

        print(
            f"BAGEL WM autoregressive rollout smoke test passed for "
            f"{len(samples)} sample(s) x {args.rollout_steps} step(s)."
        )
    finally:
        if args.close_server:
            try:
                world_model.close()
            except Exception as exc:
                print(f"Warning: failed to close remote WM server cleanly: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
