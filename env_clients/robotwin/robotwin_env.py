# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# --------------------------------------------------------------------
# Modifications:
#   Modified by VLARLKit Authors on 2026-06-01.
# --------------------------------------------------------------------
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from env_clients.utils import list_of_dict_to_dict_of_list


class RoboTwinEnv(gym.Env):
    def __init__(self, cfg, num_envs, total_num_processes, rank: int = 0):
        self._cfg = cfg
        self._rank = rank
        self._seed = int(cfg.seed) + rank
        self.num_envs = int(num_envs)
        self._total_num_processes = int(total_num_processes)
        self._group_size = int(cfg.group_size)
        self._num_group = self.num_envs // self._group_size
        assert self._num_group > 0
        assert self.num_envs % self._group_size == 0

        self._ignore_terminations = bool(cfg.ignore_terminations)
        self._use_rel_reward = bool(cfg.use_rel_reward)
        self._use_custom_reward = bool(cfg.get("use_custom_reward", True))
        self._use_fixed_reset_state_ids = bool(cfg.use_fixed_reset_state_ids)
        self._center_crop = bool(cfg.get("center_crop", False))
        self._task_name = cfg.task_config.task_name

        self._generator = np.random.default_rng(seed=self._seed)
        self._ordered_generator = np.random.default_rng(seed=0)
        self._seed_pool = self._load_seed_pool()
        self._ordered_seed_pool = None
        self._ordered_start_idx = 0

        self._reset_state_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.update_reset_state_ids()

        self._venv = None
        self._current_raw_obs = None
        self._init_env()

        self._prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

    def _init_env(self):
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = str(self._resolve_path(self._cfg.assets_path))
        self._ensure_robotwin_source_on_path()

        from robotwin.envs.vector_env import VectorEnv

        task_config = self._to_container(self._cfg.task_config)
        self._venv = VectorEnv(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=self._reset_state_ids.tolist(),
        )

    def _to_container(self, value):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
        except ModuleNotFoundError:
            pass

        if isinstance(value, dict):
            return {key: self._to_container(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_container(item) for item in value]
        return value

    def _ensure_robotwin_source_on_path(self):
        repo_root = Path(__file__).resolve().parents[2]
        robotwin_source = repo_root / "third_party" / "RoboTwin"
        if robotwin_source.exists() and str(robotwin_source) not in sys.path:
            sys.path.insert(0, str(robotwin_source))

    def _resolve_path(self, path_like) -> Path:
        path = Path(str(path_like)).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    def _load_seed_pool(self) -> np.ndarray | None:
        seeds_path = self._cfg.get("seeds_path", None)
        if seeds_path is None:
            return None

        with self._resolve_path(seeds_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        seeds = data[self._task_name]["success_seeds"]
        return np.asarray(seeds, dtype=np.int64)

    def _prepare_ordered_seed_pool(self):
        if self._seed_pool is None:
            raise ValueError("use_fixed_reset_state_ids requires seeds_path.")

        reset_state_ids = self._seed_pool.copy()
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self._total_num_processes
        )
        if valid_size <= 0:
            raise ValueError(
                f"Not enough RoboTwin seeds for {self._total_num_processes} processes."
            )
        self._ordered_generator.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        self._ordered_seed_pool = reset_state_ids.reshape(
            self._total_num_processes, -1
        )
        self._ordered_start_idx = 0

    def _get_random_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self._seed_pool is None:
            return self._generator.integers(
                low=0, high=1_000_000, size=(num_reset_states,), dtype=np.int64
            )

        if len(self._seed_pool) < num_reset_states:
            raise ValueError(
                f"RoboTwin seed pool has {len(self._seed_pool)} seeds, "
                f"but {num_reset_states} are required."
            )
        reset_state_ids = self._seed_pool.copy()
        self._generator.shuffle(reset_state_ids)
        return reset_state_ids[:num_reset_states]

    def _get_ordered_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self._ordered_seed_pool is None:
            self._prepare_ordered_seed_pool()

        ordered_seed_pool = self._ordered_seed_pool[self._rank]
        if self._ordered_start_idx + num_reset_states > len(ordered_seed_pool):
            self._prepare_ordered_seed_pool()
            ordered_seed_pool = self._ordered_seed_pool[self._rank]
        if num_reset_states > len(ordered_seed_pool):
            raise ValueError(
                f"RoboTwin ordered seed shard has {len(ordered_seed_pool)} seeds, "
                f"but {num_reset_states} are required."
            )

        reset_state_ids = ordered_seed_pool[
            self._ordered_start_idx : self._ordered_start_idx + num_reset_states
        ]
        self._ordered_start_idx += num_reset_states
        return reset_state_ids

    def _sample_reset_state_ids(self, num_reset_states: int) -> np.ndarray:
        if self._use_fixed_reset_state_ids:
            return self._get_ordered_reset_state_ids(num_reset_states)
        return self._get_random_reset_state_ids(num_reset_states)

    def update_reset_state_ids(
        self, env_idx: Optional[Union[int, list[int], np.ndarray]] = None
    ):
        if env_idx is None:
            reset_state_ids = self._sample_reset_state_ids(self._num_group)
            self._reset_state_ids = reset_state_ids.repeat(self._group_size)
            return self._reset_state_ids

        env_idx = self._normalize_env_idx(env_idx)
        num_groups = int(np.ceil(len(env_idx) / self._group_size))
        reset_state_ids = self._sample_reset_state_ids(num_groups).repeat(
            self._group_size
        )
        self._reset_state_ids[env_idx] = reset_state_ids[: len(env_idx)]
        return self._reset_state_ids[env_idx]

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _init_metrics(self):
        self._success_once = np.zeros(self.num_envs, dtype=bool)
        self._fail_once = np.zeros(self.num_envs, dtype=bool)
        self._returns = np.zeros(self.num_envs, dtype=np.float32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = self._normalize_env_idx(env_idx)
        self._prev_step_reward[env_idx] = 0.0
        self._success_once[env_idx] = False
        self._fail_once[env_idx] = False
        self._returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self._returns += step_reward
        self._success_once = self._success_once | terminations
        episode_info["success_once"] = self._success_once.copy()
        episode_info["return"] = self._returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = self._returns / np.maximum(self.elapsed_steps, 1)
        infos["episode"] = episode_info
        return infos

    def _normalize_env_idx(self, env_idx):
        if isinstance(env_idx, (int, np.integer)):
            return np.asarray([env_idx], dtype=np.int64)
        return np.asarray(env_idx, dtype=np.int64).reshape(-1)

    def _center_crop_image(self, image: np.ndarray, crop_scale: float = 0.9):
        if not self._center_crop:
            return image
        height, width = image.shape[:2]
        crop_height = int(height * crop_scale)
        crop_width = int(width * crop_scale)
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        cropped = image[top : top + crop_height, left : left + crop_width]
        return np.asarray(
            Image.fromarray(cropped).resize((width, height), Image.BILINEAR)
        )

    def _parse_image(self, image) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (
            1,
            3,
            4,
        ):
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).clip(0, 255).astype(np.uint8)
        return self._center_crop_image(image)

    def _wrap_obs(self, obs_list):
        main_images = []
        wrist_image_pairs = []
        states = []
        task_descriptions = []
        has_wrist_image = False

        for obs in obs_list:
            full_image = self._parse_image(obs["full_image"])
            left_wrist_image = obs.get("left_wrist_image", None)
            right_wrist_image = obs.get("right_wrist_image", None)

            left_wrist_image = (
                self._parse_image(left_wrist_image)
                if left_wrist_image is not None
                else np.zeros_like(full_image)
            )
            right_wrist_image = (
                self._parse_image(right_wrist_image)
                if right_wrist_image is not None
                else np.zeros_like(full_image)
            )
            has_wrist_image = has_wrist_image or obs.get("left_wrist_image") is not None
            has_wrist_image = (
                has_wrist_image or obs.get("right_wrist_image") is not None
            )

            main_images.append(full_image)
            wrist_image_pairs.append(np.stack([left_wrist_image, right_wrist_image]))
            states.append(np.asarray(obs["state"], dtype=np.float32))
            task_descriptions.append(obs.get("instruction", self._task_name))

        return {
            "main_images": np.stack(main_images),
            "wrist_images": np.stack(wrist_image_pairs) if has_wrist_image else None,
            "states": np.stack(states),
            "task_descriptions": task_descriptions,
        }

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            reset_env_idx = None
            env_seeds = np.asarray(self._reset_state_ids, dtype=np.int64)
        else:
            reset_env_idx = self._normalize_env_idx(env_idx).tolist()
            if reset_state_ids is None:
                env_seeds = self._reset_state_ids[reset_env_idx]
            else:
                env_seeds = np.asarray(reset_state_ids, dtype=np.int64).reshape(-1)
                if len(env_seeds) == self.num_envs:
                    env_seeds = env_seeds[reset_env_idx]
                else:
                    assert len(env_seeds) == len(reset_env_idx)

        if reset_state_ids is not None and env_idx is None:
            env_seeds = np.asarray(reset_state_ids, dtype=np.int64).reshape(-1)
            assert len(env_seeds) == self.num_envs

        if reset_state_ids is not None:
            if reset_env_idx is None:
                self._reset_state_ids[:] = env_seeds
            else:
                self._reset_state_ids[reset_env_idx] = env_seeds

        self._venv.reset(env_idx=reset_env_idx, env_seeds=env_seeds.tolist())
        self._current_raw_obs = self._venv.get_obs()
        obs = self._wrap_obs(self._current_raw_obs)
        self._reset_metrics(reset_env_idx)
        return obs, {}

    def reset_to_states(self, *args, **kwargs):
        raise NotImplementedError("RoboTwinEnv does not support reset_to_states.")

    def step(self, actions=None):
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[:, None, :]

        obs, rewards, terminations, truncations, infos = self.chunk_step(actions)
        return (
            obs,
            rewards[:, -1],
            terminations[:, -1],
            truncations[:, -1],
            infos,
        )

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        assert chunk_actions.ndim == 3
        chunk_size = chunk_actions.shape[1]

        (
            raw_obs,
            raw_rewards,
            raw_terminations,
            raw_truncations,
            info_lists,
        ) = self._venv.step(chunk_actions)

        self._current_raw_obs = raw_obs
        self._elapsed_steps += chunk_size

        raw_rewards = self._flatten_env_values(raw_rewards, dtype=np.float32)
        raw_terminations = self._flatten_env_values(raw_terminations, dtype=bool)
        raw_truncations = self._flatten_env_values(raw_truncations, dtype=bool)
        raw_truncations = raw_truncations | (
            self.elapsed_steps >= self._cfg.max_episode_steps
        )

        if self._use_custom_reward:
            step_reward = self._calc_step_reward(raw_terminations)
        else:
            step_reward = raw_rewards.astype(np.float32)

        terminations = raw_terminations.copy()
        if self._use_custom_reward and self._use_rel_reward:
            terminations = self._prev_step_reward > 0

        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(step_reward, terminations, infos)
        if self._ignore_terminations:
            infos["episode"]["success_at_end"] = raw_terminations.copy()
            terminations[:] = False

        chunk_rewards = np.zeros((self.num_envs, chunk_size), dtype=np.float32)
        chunk_terminations = np.zeros((self.num_envs, chunk_size), dtype=bool)
        chunk_truncations = np.zeros((self.num_envs, chunk_size), dtype=bool)
        chunk_rewards[:, -1] = step_reward
        chunk_terminations[:, -1] = terminations
        chunk_truncations[:, -1] = raw_truncations

        return (
            self._wrap_obs(raw_obs),
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _flatten_env_values(self, values, dtype):
        flattened = []
        for value in values:
            value = np.asarray(value)
            if value.size == 0:
                flattened.append(0)
            else:
                flattened.append(value.reshape(-1)[-1])
        return np.asarray(flattened, dtype=dtype)

    def _calc_step_reward(self, terminations):
        termination_bonus = self._cfg.reward_coef * terminations.astype(np.float32)

        if self._use_rel_reward:
            reward_diff = termination_bonus - self._prev_step_reward
            update_mask = termination_bonus > self._prev_step_reward
            self._prev_step_reward[update_mask] = termination_bonus[update_mask]
            reward_diff[reward_diff < 0] = 0
            return reward_diff.astype(np.float32)
        return termination_bonus.astype(np.float32)

    def sample_action_space(self):
        num_action_chunks = int(self._cfg.get("num_action_chunks", 1))
        action_dim = int(self._cfg.get("action_dim", 14))
        return self._generator.normal(
            size=(self.num_envs, num_action_chunks, action_dim)
        ).astype(np.float32)

    def check_seeds(self):
        return self._venv.check_seeds(self._reset_state_ids.tolist())

    def close(self, clear_cache: bool = True):
        if self._venv is not None:
            self._venv.close(clear_cache=clear_cache)
            self._venv = None
