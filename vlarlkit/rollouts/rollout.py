from typing import Any

from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.utils.conversion_utils import to_numpy
from vlarlkit.utils.action_utils import prepare_actions

import torch
import numpy as np


class Rollout:
    def __init__(self, cfg, env, actor_model, rollout_result: RolloutResult | None = None, mode="train",
                 is_onpolicy: bool = True):
        self.cfg = cfg
        self.env = env
        self.actor_model = actor_model
        self.actor_model.eval()
        self.rollout_result = rollout_result
        self.mode = mode
        self._is_onpolicy = is_onpolicy
        self._bootstrap_type = str(cfg.algorithm.get("bootstrap_type", "none"))
        self._gamma = float(cfg.algorithm.get("gamma", 0.99))
        self._agg_reward = str(cfg.algorithm.get("agg_reward", "sum"))
        self._reward_type = str(cfg.algorithm.get("reward_type", "chunk_level"))

        self.init_rollout()

    def init_rollout(self) -> None:
        if self.rollout_result is not None:
            self.rollout_result.clear()

    def _get_bootstrap_values(self, obs):
        """Compute V(obs) using the actor model for value bootstrapping."""
        with torch.no_grad():
            _, info = self.actor_model.predict_action_batch(
                obs,
                mode=self.mode,
                **self._sampling_kwargs(),
            )
        values = info.get("prev_values")
        if values is None:
            return None
        values = to_numpy(values)
        if values.ndim > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        return values.astype(np.float32)

    def _sampling_kwargs(self) -> dict[str, Any]:
        params = self.cfg.algorithm.get("sampling_params", {})
        kwargs = {key: params[key] for key in params} if params else {}
        train_temperature = kwargs.pop("temperature_train", None)
        eval_temperature = kwargs.pop("temperature_eval", None)
        temperature = train_temperature if self.mode == "train" else eval_temperature
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        return kwargs

    def _process_chunk_feedback(self, rewards, terminations, truncations):
        if self._reward_type == "action_level":
            cum_done = np.cumsum(np.logical_or(terminations, truncations), axis=-1)
            shifted = np.zeros_like(cum_done)
            shifted[..., 1:] = cum_done[..., :-1]
            valid_mask = shifted == 0
            return (
                rewards * valid_mask.astype(rewards.dtype),
                terminations,
                truncations,
            )

        elif self._reward_type == "chunk_level":
            if self._agg_reward == "sum":
                cum_term = np.cumsum(terminations, axis=-1)
                shifted = np.zeros_like(cum_term)
                shifted[..., 1:] = cum_term[..., :-1]
                valid_mask = (shifted == 0).astype(rewards.dtype)
                rewards = (rewards * valid_mask).sum(-1)
            elif self._agg_reward == "max":
                rewards = rewards.max(-1)
            else:
                raise ValueError(f"Unknown agg_reward: {self._agg_reward!r}, expected 'sum' or 'max'")

            terminations = terminations.any(-1)
            truncations = truncations.any(-1)
            return rewards, terminations, truncations
        else:
            raise ValueError(
                f"Unknown reward_type: {self._reward_type!r}. "
                "Expected 'chunk_level' or 'action_level'."
            )

    def _bootstrap_terminations(self) -> None:
        terms = np.stack(self.rollout_result.terminations)
        vals = np.stack(self.rollout_result.prev_values)
        if vals.ndim > terms.ndim and vals.shape[-1] == 1:
            vals = vals.squeeze(-1)

        if self._reward_type == "action_level":
            terms = terms.astype(bool)
            vals = vals.astype(np.float32)
            num_steps, _, chunk_size = terms.shape

            for t in range(num_steps):
                for c in range(chunk_size):
                    if t == num_steps - 1 and c == chunk_size - 1:
                        continue
                    mask = terms[t, :, c]
                    if not mask.any():
                        continue
                    if c + 1 < chunk_size:
                        next_values = vals[t, :, c + 1]
                    else:
                        next_values = vals[t + 1, :, 0]
                    self.rollout_result.rewards[t] = self.rollout_result.rewards[t].copy()
                    self.rollout_result.rewards[t][mask, c] += self._gamma * next_values[mask]

        elif self._reward_type == "chunk_level":
            for t in range(len(self.rollout_result.rewards) - 1):
                mask = terms[t].astype(bool)
                if not mask.any():
                    continue
                self.rollout_result.rewards[t] = self.rollout_result.rewards[t].copy()
                self.rollout_result.rewards[t][mask] += self._gamma * vals[t + 1][mask]

        else:
            raise ValueError(
                f"Unknown reward_type: {self._reward_type!r}. "
                "Expected 'chunk_level' or 'action_level'."
            )

    def _bootstrap_truncations(self, obs) -> None:
        last_values = self._get_bootstrap_values(obs)
        if last_values is None:
            return

        last_terminations = np.asarray(
            self.rollout_result.terminations[-1], dtype=bool
        )
        if self._reward_type == "action_level":
            bootstrap_mask = ~last_terminations.any(axis=-1)
            if bootstrap_mask.any():
                last_rewards = self.rollout_result.rewards[-1].copy()
                bootstrap_values = (
                    last_values[:, 0]
                    if last_values.ndim == 2
                    else last_values
                )
                last_rewards[bootstrap_mask, -1] += (
                    self._gamma * bootstrap_values[bootstrap_mask]
                )
                self.rollout_result.rewards[-1] = last_rewards
        elif self._reward_type == "chunk_level":
            bootstrap_mask = ~last_terminations
            if bootstrap_mask.any():
                last_rewards = self.rollout_result.rewards[-1].copy()
                last_rewards[bootstrap_mask] += (
                    self._gamma * last_values[bootstrap_mask]
                )
                self.rollout_result.rewards[-1] = last_rewards
        else:
            raise ValueError(
                f"Unknown reward_type: {self._reward_type!r}. "
                "Expected 'chunk_level' or 'action_level'."
            )

    def rollout_one_epoch(self):
        num_chunk_steps = (
            self.cfg.env[self.mode].max_episode_steps
            // self.cfg.model.num_action_chunks
        )

        obs, _ = self.env.reset()

        for _ in range(num_chunk_steps):
            with torch.no_grad():
                actions, info = self.actor_model.predict_action_batch(
                    obs,
                    mode=self.mode,
                    **self._sampling_kwargs(),
                )
            actions = prepare_actions(
                raw_chunk_actions=actions,
                env_type=self.cfg.env[self.mode].env_type,
                model_backend=self.cfg.model.backend,
                num_action_chunks=self.cfg.model.num_action_chunks,
                action_dim=self.cfg.model.action_dim,
                policy=self.cfg.model.get("policy_setup", None)
            )
            next_obs, rewards, terminations, truncations, env_info = self.env.chunk_step(actions)
            rewards, terminations, truncations = self._process_chunk_feedback(
                rewards,
                terminations,
                truncations,
            )

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
            obs = next_obs.copy()

        # On-policy: bootstrap the value function into the rewards.
        # Two modes:
        #   standard — only at epoch end (truncation)
        #   always   — epoch end + every intermediate termination step
        if self._bootstrap_type != "none" and self._is_onpolicy and self.rollout_result is not None:
            if self._bootstrap_type == "always":
                self._bootstrap_terminations()

            # Epoch-end: all envs are truncated, bootstrap V(last_obs)
            self._bootstrap_truncations(obs)

        # Off-policy: run one extra predict to get forward_inputs for the
        # last next_obs, then temporal-shift to build next_forward_inputs.
        if not self._is_onpolicy and self.rollout_result is not None:
            with torch.no_grad():
                _, extra_info = self.actor_model.predict_action_batch(
                    obs,
                    mode=self.mode,
                    **self._sampling_kwargs(),
                )
            extra_fi = extra_info.get("forward_inputs")
            if extra_fi is not None:
                self.rollout_result.forward_inputs.append(to_numpy(extra_fi))
                self.rollout_result.build_next_forward_inputs()

        # Update reset state ids for next epoch rollouts
        if not self.cfg.env[self.mode].use_fixed_reset_state_ids:
            self.env.update_reset_state_ids()

        return env_info["episode"]

    def run_rollout(self, rollout_epochs: int):
        rollout_infos = []
        for epoch in range(rollout_epochs):
            rollout_infos.append(self.rollout_one_epoch())
        rollout_infos = {k: np.concatenate([info[k] for info in rollout_infos]) for k in rollout_infos[0]}
        return rollout_infos
