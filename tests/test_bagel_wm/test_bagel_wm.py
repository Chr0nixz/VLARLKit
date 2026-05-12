"""
Smoke test for BAGEL WM observation and reward prediction.

Expected data layout:
    DATA_DIR/
      images/
        head_0.jpg
        wrist_0.jpg
        ...
      action_seq.jsonl

Each jsonl row should contain image file names, a task, and an action_sequence, for example:
    {"head_image": "head_0.jpg", "wrist_image": "wrist_0.jpg",
     "task": "put the object into the target area",
     "action_sequence": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vlarlkit.utils.remote_wm import RemoteWM


ACTION_DIM = 7
IMAGE_SIZE = 256
HEAD_IMAGE_KEYS = (
    "head_image",
    "head_image_file",
    "head",
    "main_image",
    "main_image_file",
    "main",
    "image",
    "image_file",
    "start_image",
    "start_image_file",
)
WRIST_IMAGE_KEYS = ("wrist_image", "wrist_image_file", "wrist")
TASK_KEYS = ("task", "task_description", "language_instruction", "instruction")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_idx} should contain a JSON object.")
            records.append(record)
    if not records:
        raise ValueError(f"No samples found in {path}.")
    return records


def _get_from_record(record: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in record:
            return record[key]

    for container_key in ("images", "start_images", "observation_images"):
        container = record.get(container_key)
        if isinstance(container, dict):
            for key in keys:
                if key in container:
                    return container[key]
    return None


def _infer_wrist_name(head_name: str) -> str | None:
    path = Path(head_name)
    name = path.name
    if name.startswith("head_"):
        return str(path.with_name(name.replace("head_", "wrist_", 1)))
    if name.startswith("main_"):
        return str(path.with_name(name.replace("main_", "wrist_", 1)))
    return None


def _resolve_image_path(data_dir: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    candidates = [
        image_path,
        data_dir / image_path,
        data_dir / "images" / image_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image '{image_name}' under {data_dir}.")


def _load_image(data_dir: Path, image_name: str) -> np.ndarray:
    image_path = _resolve_image_path(data_dir, image_name)
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.uint8)


def _load_action_sequence(record: dict[str, Any], line_idx: int) -> np.ndarray:
    if "action_sequence" not in record:
        raise KeyError(f"Sample {line_idx} does not contain 'action_sequence'.")

    action_sequence = np.asarray(record["action_sequence"], dtype=np.float32)
    if action_sequence.ndim == 1:
        action_sequence = action_sequence[None, :]
    if action_sequence.ndim != 2 or action_sequence.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"Sample {line_idx} action_sequence should be [T, {ACTION_DIM}], "
            f"got {action_sequence.shape}."
        )
    return action_sequence


def _load_sample(data_dir: Path, record: dict[str, Any], sample_idx: int) -> tuple[dict, np.ndarray]:
    head_name = _get_from_record(record, HEAD_IMAGE_KEYS)
    if head_name is None:
        raise KeyError(
            f"Sample {sample_idx} should contain one of {HEAD_IMAGE_KEYS}, "
            "or the same keys under an 'images' field."
        )

    wrist_name = _get_from_record(record, WRIST_IMAGE_KEYS)
    if wrist_name is None:
        wrist_name = _infer_wrist_name(str(head_name))
    if wrist_name is None:
        raise KeyError(
            f"Sample {sample_idx} should contain one of {WRIST_IMAGE_KEYS}; "
            "automatic wrist name inference only supports head_*/main_* names."
        )

    observations = {
        "main_images": _load_image(data_dir, str(head_name))[None],
        "wrist_images": _load_image(data_dir, str(wrist_name))[None],
    }

    task = _get_from_record(record, TASK_KEYS)
    if task is None:
        raise KeyError(f"Sample {sample_idx} should contain one of {TASK_KEYS}.")
    observations["task_descriptions"] = [str(task)]

    actions = _load_action_sequence(record, sample_idx)[None]
    return observations, actions


def _check_prediction(next_observations: dict, sample_idx: int) -> None:
    for key in ("main_images", "wrist_images"):
        images = np.asarray(next_observations[key])
        if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3:
            raise AssertionError(
                f"Sample {sample_idx} predicted {key} should be [1, H, W, 3], "
                f"got {images.shape}."
            )
        if images.dtype != np.uint8:
            raise AssertionError(f"Sample {sample_idx} predicted {key} dtype should be uint8.")


def _save_prediction(output_dir: Path, sample_idx: int, next_observations: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, prefix in (("main_images", "head"), ("wrist_images", "wrist")):
        image = np.asarray(next_observations[key])[0]
        Image.fromarray(image).save(output_dir / f"{prefix}_pred_{sample_idx}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test BAGEL WM prediction from rollout samples.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--recv-timeout-ms", type=int, default=900_000)
    parser.add_argument("--save-output-dir", type=Path, default=None)
    parser.add_argument("--close-server", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.data_dir / "action_seq.jsonl")
    world_model = RemoteWM(
        host=args.host,
        port=args.port,
        recv_timeout_ms=args.recv_timeout_ms,
    )

    try:
        for sample_idx, record in enumerate(records):
            observations, actions = _load_sample(args.data_dir, record, sample_idx)
            print(f"Sample {sample_idx}: calling get_observations with actions={actions.shape}")
            next_observations = world_model.get_observations(
                observations=observations,
                actions=actions,
                image_key="main_images",
                wrist_image_key="wrist_images",
            )
            _check_prediction(next_observations, sample_idx)
            print(
                f"Sample {sample_idx}: predicted "
                f"main_images={np.asarray(next_observations['main_images']).shape}, "
                f"wrist_images={np.asarray(next_observations['wrist_images']).shape}"
            )

            if args.save_output_dir is not None:
                _save_prediction(args.save_output_dir, sample_idx, next_observations)

            tasks = observations["task_descriptions"]
            rewards = np.asarray(
                world_model.get_rewards(
                    next_observations,
                    tasks=tasks,
                    image_key="main_images",
                )
            )
            if rewards.shape != (1,) or rewards.dtype != np.float32:
                raise AssertionError(
                    f"Expected rewards shape (1,) float32, got {rewards.shape} {rewards.dtype}."
                )
            print(f"Sample {sample_idx}: task={tasks[0]!r}, reward={float(rewards[0]):.1f}")

        print(f"BAGEL WM prediction smoke test passed for {len(records)} sample(s).")
    finally:
        if args.close_server:
            try:
                world_model.close()
            except Exception as exc:
                print(f"Warning: failed to close remote WM server cleanly: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
