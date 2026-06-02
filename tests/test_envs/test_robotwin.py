from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from env_clients.client import import_env_class
from env_clients.robotwin.robotwin_env import RoboTwinEnv


class FakeVectorEnv:
    def __init__(self, num_envs: int, wrist_images: bool = True):
        self.num_envs = num_envs
        self.wrist_images = wrist_images
        self.reset_calls = []
        self.step_calls = []
        self.check_seed_arg = None
        self.close_arg = None
        self.next_rewards = [np.array([1.0]), np.array([0.0]), np.array([0.5])]
        self.next_terminations = [np.array([1]), np.array([0]), np.array([0])]
        self.next_truncations = [np.array([0]), np.array([0]), np.array([0])]

    def reset(self, env_idx=None, env_seeds=None):
        self.reset_calls.append((env_idx, list(env_seeds)))

    def get_obs(self):
        obs_list = []
        for env_id in range(self.num_envs):
            obs = {
                "full_image": np.zeros((8, 9, 3), dtype=np.uint8) + env_id,
                "state": np.arange(14, dtype=np.float32) + env_id,
                "instruction": f"instruction {env_id}",
            }
            if self.wrist_images:
                obs["left_wrist_image"] = (
                    np.ones((8, 9, 3), dtype=np.uint8) * (10 + env_id)
                )
                obs["right_wrist_image"] = (
                    np.ones((8, 9, 3), dtype=np.uint8) * (20 + env_id)
                )
            else:
                obs["left_wrist_image"] = None
                obs["right_wrist_image"] = None
            obs_list.append(obs)
        return obs_list

    def step(self, actions):
        self.step_calls.append(np.asarray(actions).copy())
        info_list = [
            {"success": bool(np.asarray(termination).reshape(-1)[-1])}
            for termination in self.next_terminations
        ]
        return (
            self.get_obs(),
            self.next_rewards[: self.num_envs],
            self.next_terminations[: self.num_envs],
            self.next_truncations[: self.num_envs],
            info_list[: self.num_envs],
        )

    def check_seeds(self, seeds):
        self.check_seed_arg = list(seeds)
        return [{"seed": seed} for seed in seeds]

    def close(self, clear_cache=True):
        self.close_arg = clear_cache


def _write_seed_file(tmp_path: Path, seeds: list[int]) -> str:
    path = tmp_path / "robotwin_seeds.json"
    path.write_text(
        json.dumps({"adjust_bottle": {"success_seeds": seeds}}),
        encoding="utf-8",
    )
    return str(path)


def _make_cfg(
    tmp_path: Path,
    *,
    ignore_terminations: bool = False,
    use_custom_reward: bool = True,
    use_rel_reward: bool = True,
    use_fixed_reset_state_ids: bool = False,
    max_episode_steps: int = 100,
    wrist_images: bool = True,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "group_size": 1,
            "ignore_terminations": ignore_terminations,
            "use_rel_reward": use_rel_reward,
            "use_custom_reward": use_custom_reward,
            "use_fixed_reset_state_ids": use_fixed_reset_state_ids,
            "center_crop": False,
            "reward_coef": 1.0,
            "max_episode_steps": max_episode_steps,
            "assets_path": str(tmp_path),
            "seeds_path": _write_seed_file(tmp_path, list(range(100, 130))),
            "test_wrist_images": wrist_images,
            "task_config": {
                "task_name": "adjust_bottle",
                "embodiment": ["aloha-agilex"],
            },
        }
    )


@pytest.fixture(autouse=True)
def fake_vector_env(monkeypatch):
    def fake_init_env(self):
        self._venv = FakeVectorEnv(
            self.num_envs,
            wrist_images=bool(self._cfg.get("test_wrist_images", True)),
        )

    monkeypatch.setattr(RoboTwinEnv, "_init_env", fake_init_env)


def test_robotwin_registry_and_reset_chunk_step_contract(tmp_path):
    assert import_env_class("robotwin") is RoboTwinEnv

    env = RoboTwinEnv(
        _make_cfg(tmp_path),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )

    obs, info = env.reset()
    assert info == {}
    assert env._venv.reset_calls[-1] == (None, env._reset_state_ids.tolist())
    assert obs["main_images"].shape == (3, 8, 9, 3)
    assert obs["wrist_images"].shape == (3, 2, 8, 9, 3)
    assert obs["states"].shape == (3, 14)
    assert obs["task_descriptions"] == [
        "instruction 0",
        "instruction 1",
        "instruction 2",
    ]

    obs, _ = env.reset(env_idx=[0, 2], reset_state_ids=[101, 303])
    assert env._venv.reset_calls[-1] == ([0, 2], [101, 303])
    assert env._reset_state_ids[[0, 2]].tolist() == [101, 303]

    obs, _ = env.reset(env_idx=[1], reset_state_ids=[7, 8, 9])
    assert env._venv.reset_calls[-1] == ([1], [8])
    assert env._reset_state_ids[1] == 8

    actions = np.zeros((3, 50, 14), dtype=np.float32)
    obs, rewards, terminations, truncations, infos = env.chunk_step(actions)
    assert env._venv.step_calls[-1].shape == (3, 50, 14)
    assert obs["main_images"].shape == (3, 8, 9, 3)
    assert rewards.shape == (3, 50)
    assert terminations.shape == (3, 50)
    assert truncations.shape == (3, 50)
    assert rewards[:, -1].tolist() == [1.0, 0.0, 0.0]
    assert terminations[:, -1].tolist() == [True, False, False]
    assert truncations.any() is np.False_
    assert infos["episode"]["success_once"].tolist() == [True, False, False]


def test_robotwin_reward_modes_ignore_terminations_and_missing_wrist(tmp_path):
    env = RoboTwinEnv(
        _make_cfg(tmp_path),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )
    actions = np.zeros((3, 50, 14), dtype=np.float32)

    _, rewards, terminations, _, _ = env.chunk_step(actions)
    assert rewards[:, -1].tolist() == [1.0, 0.0, 0.0]
    assert terminations[:, -1].tolist() == [True, False, False]

    _, rewards, terminations, _, _ = env.chunk_step(actions)
    assert rewards[:, -1].tolist() == [0.0, 0.0, 0.0]
    assert terminations[:, -1].tolist() == [True, False, False]

    raw_reward_env = RoboTwinEnv(
        _make_cfg(tmp_path, use_custom_reward=False, use_rel_reward=True),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )
    _, rewards, terminations, _, _ = raw_reward_env.chunk_step(actions)
    assert rewards[:, -1].tolist() == [1.0, 0.0, 0.5]
    assert terminations[:, -1].tolist() == [True, False, False]

    ignore_env = RoboTwinEnv(
        _make_cfg(tmp_path, ignore_terminations=True),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )
    _, _, terminations, _, infos = ignore_env.chunk_step(actions)
    assert terminations[:, -1].tolist() == [False, False, False]
    assert infos["episode"]["success_once"].tolist() == [True, False, False]
    assert infos["episode"]["success_at_end"].tolist() == [True, False, False]

    no_wrist_env = RoboTwinEnv(
        _make_cfg(tmp_path, wrist_images=False),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )
    obs, _ = no_wrist_env.reset()
    assert obs["wrist_images"] is None


def test_robotwin_fixed_seeds_truncation_check_seeds_and_close(tmp_path):
    env = RoboTwinEnv(
        _make_cfg(
            tmp_path,
            use_fixed_reset_state_ids=True,
            max_episode_steps=50,
        ),
        num_envs=3,
        total_num_processes=1,
        rank=0,
    )
    assert env._venv.reset_calls == []

    obs, rewards, terminations, truncations, infos = env.chunk_step(
        np.zeros((3, 50, 14), dtype=np.float32)
    )
    assert truncations[:, -1].tolist() == [True, True, True]

    check_results = env.check_seeds()
    assert env._venv.check_seed_arg == env._reset_state_ids.tolist()
    assert [result["seed"] for result in check_results] == env._reset_state_ids.tolist()

    fake_venv = env._venv
    env.close(clear_cache=False)
    assert fake_venv.close_arg is False
    assert env._venv is None
