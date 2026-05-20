import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from datasets import load_dataset

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.utils.action_utils import prepare_actions
from vlarlkit.utils.conversion_utils import to_numpy


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_array(image: Any) -> np.ndarray:
    if isinstance(image, dict) and "bytes" in image:
        if image["bytes"] is None:
            image = image["path"]
        else:
            image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    arr = _as_numpy(image)
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


def _image_batch_to_array(images: Any) -> np.ndarray:
    if isinstance(images, (list, tuple)):
        return np.stack([_image_to_array(image) for image in images], axis=0)

    arr = _as_numpy(images)
    if arr.ndim == 3:
        return _image_to_array(arr)[None]
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 1, -1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected image batch [B, H, W, 3] or [B, 3, H, W], got {arr.shape}.")
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size > 0 and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    raise ValueError(f"Expected image batch with 3 or 4 dims, got {arr.shape}.")


def _collate_batch(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {k: _collate_batch([item[k] for item in items]) for k in first}
    return np.stack([_as_numpy(item) for item in items], axis=0)


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        return _batch_size(next(iter(batch.values())))
    return int(np.asarray(batch).shape[0])


def _slice_batch(batch: Any, start: int, end: int | None) -> Any:
    if isinstance(batch, dict):
        return {k: _slice_batch(v, start, end) for k, v in batch.items()}
    return batch[start:end]


def _concat_batches(left: Any, right: Any) -> Any:
    if isinstance(left, dict):
        return {k: _concat_batches(left[k], right[k]) for k in left}
    return np.concatenate([left, right], axis=0)


def _cycle(loader: DataLoader, sampler: DistributedSampler | None = None):
    epoch = 0
    carry = None
    batch_size = int(loader.batch_size)
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if carry is not None:
                batch = _concat_batches(carry, batch)
                carry = None

            while _batch_size(batch) >= batch_size:
                yield _slice_batch(batch, 0, batch_size)
                batch = _slice_batch(batch, batch_size, None)

            if _batch_size(batch) > 0:
                carry = batch
        epoch += 1


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    with open(dataset_root / "meta" / "tasks.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


class BranchRollout:
    def __init__(
        self,
        cfg,
        world_model,
        actor_model,
        rollout_result: RolloutResult | None = None,
        mode: str = "train",
        branch_data_iter: Any | None = None,
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.actor_model = actor_model
        self.actor_model.eval()
        self.rollout_result = rollout_result
        self.mode = mode

        self._algo_cfg = cfg.algorithm
        self._global_branch_sample_size = int(self._algo_cfg.branch_sample_size)
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        assert self._global_branch_sample_size % self._world_size == 0, (
            f"branch_sample_size ({self._global_branch_sample_size}) must be divisible "
            f"by world_size ({self._world_size})."
        )
        self._branch_sample_size = self._global_branch_sample_size // self._world_size
        max_steps = int(self._algo_cfg.get("branch_max_episode_steps"))
        env_cfg = self.cfg.env.get(self.mode, self.cfg.env.get("eval"))
        self._env_type = env_cfg.get("env_type", "libero")
        num_action_chunks = int(self.cfg.model.num_action_chunks)
        assert max_steps % num_action_chunks == 0, (
            f"branch_max_episode_steps ({max_steps}) must be divisible by "
            f"num_action_chunks ({num_action_chunks})."
        )
        self._num_chunk_steps = max_steps // num_action_chunks
        self._gamma = float(self._algo_cfg.get("gamma", 0.99))
        self._validate_world_model_actor_config()
        world_model_cfg = self.cfg.get("world_model", {})
        self._edit_kwargs = _plain_dict(world_model_cfg.get("edit_kwargs", {}))
        self._und_kwargs = _plain_dict(world_model_cfg.get("und_kwargs", {}))
        self._tasks = None
        self._data_iter = branch_data_iter or self._build_branch_data_iter()

        self.init_rollout()

    def init_rollout(self) -> None:
        if self.rollout_result is not None:
            self.rollout_result.clear()

    def _validate_world_model_actor_config(self) -> None:
        openpi_cfg = self.cfg.model.get("openpi", {})
        pi05 = bool(openpi_cfg.get("pi05", False))
        discrete_state_input = openpi_cfg.get("discrete_state_input", None)
        assert pi05 and discrete_state_input is False, (
            "BranchRollout with the current image-only world model requires "
            "model.openpi.pi05=True and model.openpi.discrete_state_input=False. "
            "State is kept in observations for action transforms, but must not be "
            "consumed as a VLA state token during world-model rollout."
        )

    def _build_branch_data_iter(self):
        repo_id_key = (
            "branch_eval_repo_id" if self.mode == "eval" else "branch_train_repo_id"
        )
        repo_id = self._algo_cfg.get(repo_id_key)

        dataset_root = self._resolve_dataset_root(repo_id)
        self._tasks = _load_tasks(dataset_root)

        dataset = load_dataset(
            "parquet",
            data_dir=str(dataset_root / "data"),
            split="train",
        )
        seed = int(getattr(self.cfg.runner, "seed", 0))
        sampler = None
        generator = None
        shuffle = True
        local_dataset_size = len(dataset)

        if self._world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            shuffle = False
            local_dataset_size = len(sampler)
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)

        if local_dataset_size < self._branch_sample_size:
            raise ValueError(
                f"per-rank branch_sample_size ({self._branch_sample_size}) is larger than "
                f"the per-rank branch dataset size ({local_dataset_size})."
            )

        loader = DataLoader(
            dataset,
            batch_size=self._branch_sample_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            collate_fn=_collate_batch,
            drop_last=False,
            generator=generator,
        )
        return _cycle(loader, sampler)

    def _resolve_dataset_root(self, repo_id: str) -> Path:
        root = Path(self._algo_cfg.branch_dataset_root).expanduser()
        if (root / "meta").exists():
            return root
        candidates = [root / repo_id, root / repo_id.split("/")[-1]]
        for candidate in candidates:
            if (candidate / "meta").exists():
                return candidate
        return candidates[0]

    def _get_bootstrap_values(self, obs):
        with torch.no_grad():
            _, info = self.actor_model.predict_action_batch(obs, mode=self.mode)
        values = info.get("prev_values")
        if values is None:
            return None
        values = to_numpy(values)
        if values.ndim > 1:
            values = values.squeeze(-1)
        return values.astype(np.float32)

    def _task_description(self, task_index: int) -> str:
        if self._tasks is None:
            raise RuntimeError("Branch rollout tasks are not loaded.")
        return self._tasks[int(task_index)]

    def _batch_to_obs(self, batch: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        main_images = batch["image"]
        wrist_images = batch["wrist_image"]
        states = batch["state"]
        task_indices = np.asarray(batch["task_index"]).reshape(-1).astype(np.int64)
        task_ids = np.asarray(batch["task_id"]).reshape(-1).astype(np.int64)
        task_descriptions = [
            self._task_description(int(task_index)) for task_index in task_indices
        ]

        obs = {
            "main_images": _image_batch_to_array(main_images),
            "wrist_images": _image_batch_to_array(wrist_images),
            "extra_view_images": None,
            "states": _as_numpy(states).astype(np.float32),
            "task_descriptions": task_descriptions,
        }
        return obs, task_ids

    def _sample_actions(self, obs: dict[str, Any]):
        with torch.no_grad():
            actions, info = self.actor_model.predict_action_batch(obs, mode=self.mode)
        actions = prepare_actions(
            raw_chunk_actions=actions,
            env_type=self._env_type,
            model_type=self.cfg.model.model_type,
            num_action_chunks=self.cfg.model.num_action_chunks,
            action_dim=self.cfg.model.action_dim,
            policy=self.cfg.model.get("policy_setup", None),
        )
        return actions, info

    def _step_world_model(
        self,
        obs: dict[str, Any],
        actions: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        next_obs, rewards, terminations = self.world_model.step(
            observations=obs,
            actions=actions,
            image_key="main_images",
            wrist_image_key="wrist_images",
            edit_kwargs=self._edit_kwargs,
            und_kwargs=self._und_kwargs,
        )
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1) # [B,]
        terminations = np.asarray(terminations, dtype=bool).reshape(-1) # [B,]
        return next_obs, rewards, terminations

    def rollout_one_epoch(self):
        batch = next(self._data_iter)
        obs, task_ids = self._batch_to_obs(batch)
        batch_size = int(np.asarray(obs["states"]).shape[0])

        success_once = np.zeros((batch_size,), dtype=bool)
        returns = np.zeros((batch_size,), dtype=np.float32)
        episode_len = np.full((batch_size,), self._num_chunk_steps, dtype=np.int32)

        for step in range(self._num_chunk_steps):
            actions, info = self._sample_actions(obs)
            next_obs, rewards, terminations = self._step_world_model(obs, actions)
            new_success = terminations
            first_success = np.logical_and(new_success, ~success_once)
            episode_len[first_success] = step + 1
            success_once = np.logical_or(success_once, new_success)
            terminations = success_once.copy()
            truncations = np.zeros_like(terminations, dtype=bool)
            if step == self._num_chunk_steps - 1:
                truncations[:] = True
            returns += rewards

            if self.rollout_result is not None:
                self.rollout_result.append_step(
                    obs=obs,
                    next_obs=next_obs,
                    actions=actions,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    prev_logprobs=info.get("prev_logprobs"),
                    prev_values=info.get("prev_values"),
                    forward_inputs=info.get("forward_inputs"),
                )
            obs = next_obs

        if self.rollout_result is not None:
            last_values = self._get_bootstrap_values(obs)
            if last_values is not None:
                bootstrap_mask = ~np.asarray(self.rollout_result.terminations[-1], dtype=bool)
                if bootstrap_mask.any():
                    last_rewards = self.rollout_result.rewards[-1].copy()
                    last_rewards[bootstrap_mask] += self._gamma * last_values[bootstrap_mask]
                    self.rollout_result.rewards[-1] = last_rewards

        return {
            "success_once": success_once,
            "return": returns,
            "task_id": task_ids,
        }

    def run_rollout(self, rollout_epochs: int):
        rollout_infos = []
        for _ in range(rollout_epochs):
            rollout_infos.append(self.rollout_one_epoch())
        return {
            k: np.concatenate([info[k] for info in rollout_infos])
            for k in rollout_infos[0]
        }
